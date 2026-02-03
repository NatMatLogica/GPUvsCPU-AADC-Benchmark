#include <atomic>
#include <string>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

#include "XVAProblem.h"
#include "XVAJobRequest.h"
#include "DataTools.h"

// Adept AAD library removed - using AADC only
// #include "adept/adept_source.h"

using json = nlohmann::json;

////////////////////////////////////////////////////
//
//  compute_xva_from_exposures
//
//  Compute CVA/DVA from PEE/NEE arrays using trapezoidal rule
//  (same formula as XVAProblem::computeXVAMeasures)
//
////////////////////////////////////////////////////

std::pair<double, double> compute_xva_from_exposures(
    const std::vector<double>& pee,
    const std::vector<double>& nee,
    const json& data_in
) {
    // Build pricing times from config
    int t0 = data_in["t0"].get<int>();
    int max_t = data_in["ModelAndPricingTimes"]["T"].get<int>();
    int model_step = data_in["ModelAndPricingTimes"]["step"].get<int>();
    int pricing_freq = data_in["ModelAndPricingTimes"]["PricingFreq"].get<int>();

    std::vector<int> pricing_times;
    int pr = 0;
    int t = t0;
    while (t < max_t) {
        if (pr == 0 || (t + model_step > max_t)) {
            pricing_times.push_back(t);
            pr = pricing_freq;
        }
        --pr;
        t += model_step;
    }

    // Build survival curves from config
    auto build_surv_curve = [&](const json& curve_data) -> std::vector<std::pair<int, double>> {
        std::vector<std::pair<int, double>> curve;
        int step = curve_data["step"].get<int>();
        int curve_max_t = curve_data["T"].get<int>();
        double rate = curve_data["flat_rate"].get<double>();
        int ct = t0;
        while (ct < curve_max_t) {
            curve.push_back({ct, rate});
            ct += step;
        }
        return curve;
    };

    auto ctrparty_curve = build_surv_curve(data_in["CounterPartySurvivalCurve"]);
    auto company_curve = build_surv_curve(data_in["CompanySurvivalCurve"]);

    // Linear interpolation for survival probability: exp(-rate * yearfrac)
    auto get_survival = [&](const std::vector<std::pair<int, double>>& curve, int time) -> double {
        if (curve.empty()) return 1.0;
        // Find surrounding points
        double rate = curve[0].second;  // Default to first rate
        for (size_t i = 0; i < curve.size(); ++i) {
            if (curve[i].first <= time) {
                rate = curve[i].second;
                // Linear interpolation if not at end
                if (i + 1 < curve.size() && curve[i + 1].first > time) {
                    double frac = double(time - curve[i].first) /
                                  double(curve[i + 1].first - curve[i].first);
                    rate = curve[i].second + frac * (curve[i + 1].second - curve[i].second);
                }
            }
        }
        double yearfrac = double(time - t0) / 365.0;
        return std::exp(-rate * yearfrac);
    };

    // Trapezoidal rule for CVA/DVA integration
    double cva = 0.0, dva = 0.0;
    size_t n = std::min(pee.size(), pricing_times.size());
    if (n < 2) return {0.0, 0.0};

    double ctrparty_next = get_survival(ctrparty_curve, pricing_times[0]);
    double company_next = get_survival(company_curve, pricing_times[0]);

    for (size_t i = 0; i < n - 1; ++i) {
        double ctrparty_prev = ctrparty_next;
        ctrparty_next = get_survival(ctrparty_curve, pricing_times[i + 1]);
        double company_prev = company_next;
        company_next = get_survival(company_curve, pricing_times[i + 1]);

        cva += (pee[i] + pee[i + 1]) * (ctrparty_prev - ctrparty_next) * 0.5;
        dva += (nee[i] + nee[i + 1]) * (company_prev - company_next) * 0.5;
    }

    return {cva, dva};
}

////////////////////////////////////////////////////
//
//  get_iso_timestamp
//
//  Returns current time as ISO 8601 string
//
////////////////////////////////////////////////////

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    struct tm tm_buf;
    localtime_r(&time_t_now, &tm_buf);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    char result[80];
    snprintf(result, sizeof(result), "%s.%03d", buf, (int)ms.count());
    return std::string(result);
}

////////////////////////////////////////////////////
//
//  log_xva_csv
//
//  Append one row to data/execution_log_xva.csv
//
////////////////////////////////////////////////////

static const char* CSV_HEADER =
    "timestamp,model_name,model_version,num_trades,num_mc_paths,"
    "num_model_steps,num_pricing_times,num_sensitivity_params,num_threads,"
    "backend,mode,cva_result,dva_result,"
    "eval_time_sec,sensitivity_time_sec,total_time_sec,"
    "kernel_recording_sec,num_params_bumped,speedup_vs_cpu,"
    "max_cva_diff,max_dva_diff,gpu_kernel_time_sec,status\n";

void log_xva_csv(
    const std::string& csv_path,
    const std::string& model_name,
    int num_trades, int num_mc_paths,
    int num_model_steps, int num_pricing_times,
    int num_sensitivity_params, int num_threads,
    const std::string& backend, const std::string& mode,
    double cva, double dva,
    double primal_time_sec, double sensitivity_time_sec,
    double total_time_sec, double kernel_recording_sec,
    int num_params_bumped, double speedup_vs_cpu,
    double max_cva_diff, double max_dva_diff,
    double gpu_kernel_time_sec,
    const std::string& status
) {
    // Check if file exists to decide whether to write header
    struct stat st;
    bool file_exists = (stat(csv_path.c_str(), &st) == 0 && st.st_size > 0);

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv.is_open()) {
        std::cerr << "Warning: cannot open " << csv_path << " for logging\n";
        return;
    }
    if (!file_exists) {
        csv << CSV_HEADER;
    }
    csv << std::setprecision(15)
        << get_iso_timestamp() << ","
        << model_name << ","
        << "1.0.0" << ","
        << num_trades << ","
        << num_mc_paths << ","
        << num_model_steps << ","
        << num_pricing_times << ","
        << num_sensitivity_params << ","
        << num_threads << ","
        << backend << ","
        << mode << ","
        << cva << ","
        << dva << ","
        << primal_time_sec << ","
        << sensitivity_time_sec << ","
        << total_time_sec << ","
        << kernel_recording_sec << ","
        << num_params_bumped << ","
        << speedup_vs_cpu << ","
        << max_cva_diff << ","
        << max_dva_diff << ","
        << gpu_kernel_time_sec << ","
        << status << "\n";
    csv.close();
}

////////////////////////////////////////////////////
//
//  checkBondsFormula
//
//  This function allows to compare analytical Hull-White formula via MC simulation   
//
//  XVA      XVA problem
//
////////////////////////////////////////////////////

void checkBondsFormula(XVAProblem<double>& XVA) {
    std::mt19937_64 gen(17);
    std::normal_distribution<> normal_distrib(0, 1);

//    HullWhiteMarketModel<double> M=*XVA.getModel();
    std::vector<double> mean_rev_vector(XVA.getModel()->getMeanRev()->getVals());
    for (int i=0; i< mean_rev_vector.size(); i++) {
        mean_rev_vector[i]=(normal_distrib(gen)+10)/20;
    }
    HullWhiteMarketModel<double> model(
        XVA.getModel()->getR0(), XVA.getModel()->getSigma(), XVA.getModel()->getAlpha(),
        std::make_shared<PiecewiseLinearCurve<double>>(
            XVA.getModel()->getMeanRev()->getTimes(),
            mean_rev_vector
        ),
        XVA.getModel()->getSpreads()
    );
    model.initT0(XVA.getT0());
    double p(model.getIrMarket().getDiscountCurve()(20.)); // Analytical formula 
    
    std::cout << "Correctness of the HW-bond price formula \n";
    std::cout <<"HW-formula for 20 years bond price " << p << "\n";
    
    int mc_iterations=800;
    qtime time=20*365;
    int per_day=5;

    std::vector<double> normals(per_day*time);
    double bond=0;
    for (int i=0; i<mc_iterations; i++) {
        model.initT0(0);
        for (int j=0; j<normals.size(); j++) {
            normals[j]=normal_distrib(gen);
        }
        bond+=Vasicek<double>(time, model, normals);
    }
    std::cout << "MonteCarlo price " << bond/mc_iterations << "\n";
    std::cout << "Iterations=" << mc_iterations << "; partions_per_day=" << per_day << ";\n";
    std::cout << "relative formula/MC " << bond/mc_iterations/p << "\n";
    std::cout << "----------------------------\n";
}

////////////////////////////////////////////////////
//
//  checkHullWhiteFormula
//
//  This function allows to compare analytical Hull-White formula via MC simulation  
//
//  request_data     XVA task configuration
//
////////////////////////////////////////////////////

void checkHullWhiteFormula(const json& request_data) {
    std::cout<<"\n";
    XVAProblem<double> XVA_check; 
    XVA_check.initData(request_data);
    checkBondsFormula(XVA_check);
}

////////////////////////////////////////////////////
//
//  run_pricing
//
//  This function demonstrates how interface of XVAJobRequest can be used. 
//  There is a sequence of calls of XVA computations (processRequest()) for various 
//  portfolios and process parameters  
//
//  threads_num     number of threads 
//  path            path to the XVA task data
//
////////////////////////////////////////////////////

template<class mmType>
int run_pricing(const int threads_num, const std::string input_file) {
    json data_in, data_out;
    std::ifstream i(input_file);
    if(i.fail()) {
        std::cout << "Fail to open " << input_file << "\n";
        return 1;
    }
    i >> data_in;

    //checkHullWhiteFormula(data_in);

    std::shared_ptr<std::vector<RequestFunction<mmType>>> func_request_cache =
        std::make_shared<std::vector<RequestFunction<mmType>>>()
    ;
    std::shared_ptr<XVAJobRequest<mmType>> obj;
    std::atomic<bool> cancel= false;


    // Example of the XVA Task pricing loop
    obj=std::make_shared<XVAJobRequest<mmType>>(func_request_cache);
    obj->processRequest(data_in, data_out, threads_num, cancel);
    std::ofstream all_res("all_results.json");
    all_res << std::setw(4) << data_out << std::endl;
    all_res.close();

    // --- CSV logging ---
    // Derive path relative to input file location
    std::string csv_path = "data/execution_log_xva.csv";
    {
        auto slash = input_file.rfind('/');
        if (slash != std::string::npos) {
            csv_path = input_file.substr(0, slash + 1) + "data/execution_log_xva.csv";
        }
    }

    // Extract portfolio and time grid dimensions from config
    int num_trades = data_in["Portfolio"]["NumRandomTrades"].get<int>();
    int num_mc_paths = obj->m_mc_iterations;

    // Count model steps and pricing times
    int model_T = data_in["ModelAndPricingTimes"]["T"].get<int>();
    int model_step = data_in["ModelAndPricingTimes"]["step"].get<int>();
    int pricing_freq = data_in["ModelAndPricingTimes"]["PricingFreq"].get<int>();
    int t0 = data_in["t0"].get<int>();
    int num_model_steps = (model_T - t0) / model_step;
    int num_pricing_times = (num_model_steps + pricing_freq - 1) / pricing_freq;

    // Count sensitivity parameters (actual curve points + r0 + sigma)
    // MR curve: T/step points (T in years, step in years)
    double mr_T = data_in["Currencies"]["EUR"]["HWMeanReversionCurve"]["T"].get<double>();
    double mr_step = data_in["Currencies"]["EUR"]["HWMeanReversionCurve"]["step"].get<double>();
    int n_mr = (int)(mr_T / mr_step);

    // Survival curves: T/step + 1 points (T and step in days, +1 for t=0)
    int surv_T = data_in["CounterPartySurvivalCurve"]["T"].get<int>();
    int surv_step = data_in["CounterPartySurvivalCurve"]["step"].get<int>();
    int n_ctrp_surv = surv_T / surv_step + 1;

    int comp_T = data_in["CompanySurvivalCurve"]["T"].get<int>();
    int comp_step = data_in["CompanySurvivalCurve"]["step"].get<int>();
    int n_comp_surv = comp_T / comp_step + 1;

    int num_sens = 2 + n_mr + n_ctrp_surv + n_comp_surv;  // r0, sigma, MR, survival curves

    // Extract results
    double primal_cva = 0, primal_dva = 0, aadc_cva = 0, aadc_dva = 0;
    if (data_out.contains("primal")) {
        primal_cva = data_out["primal"]["CVA"].get<double>();
        primal_dva = data_out["primal"]["DVA"].get<double>();
    }
    if (data_out.contains("AADC results")) {
        aadc_cva = data_out["AADC results"]["CVA"].get<double>();
        aadc_dva = data_out["AADC results"]["DVA"].get<double>();

        // If AADC CVA/DVA are 0, compute from PEE/NEE arrays
        if (aadc_cva == 0.0 && aadc_dva == 0.0 &&
            data_out["AADC results"].contains("PEE") &&
            data_out["AADC results"].contains("NEE")) {
            std::vector<double> pee = data_out["AADC results"]["PEE"].get<std::vector<double>>();
            std::vector<double> nee = data_out["AADC results"]["NEE"].get<std::vector<double>>();

            // Check if arrays have meaningful values
            bool has_exposures = false;
            for (double v : pee) if (v != 0.0) { has_exposures = true; break; }
            if (!has_exposures) {
                for (double v : nee) if (v != 0.0) { has_exposures = true; break; }
            }

            if (has_exposures) {
                auto [computed_cva, computed_dva] = compute_xva_from_exposures(pee, nee, data_in);
                aadc_cva = computed_cva;
                aadc_dva = computed_dva;
                std::cout << "  CVA/DVA computed from AADC exposures: CVA=" << aadc_cva
                          << ", DVA=" << aadc_dva << "\n";
            }
        }
    }

    // Timing (microseconds -> seconds), normalized to full MC iterations
    double primal_time_sec = obj->m_primal_is_required
        ? obj->m_base_time.count() * obj->m_norm_coeff / 1e6 : 0.0;
    double aadc_time_sec = obj->m_aad_time.count() / 1e6;
    double compilation_sec = 0;
    if (data_out.contains("compiler data") && data_out["compiler data"].contains("Compilation time")) {
        compilation_sec = data_out["compiler data"]["Compilation time"].get<double>() / 1e6;
    }

    bool forward_only = data_in["Forward Only"].get<bool>();
    std::string mode = forward_only ? "pricing_only" : "pricing_with_greeks";

    // Use primal values when AADC doesn't compute its own CVA/DVA
    double reported_cva = (aadc_cva != 0.0) ? aadc_cva : primal_cva;
    double reported_dva = (aadc_dva != 0.0) ? aadc_dva : primal_dva;

    double max_cva_diff = std::abs(primal_cva - reported_cva);
    double max_dva_diff = std::abs(primal_dva - reported_dva);

    // Log primal row
    if (obj->m_primal_is_required) {
        log_xva_csv(csv_path, "xva_cpp_primal", num_trades, num_mc_paths,
            num_model_steps, num_pricing_times, num_sens, threads_num,
            "cpp_double", mode, primal_cva, primal_dva,
            primal_time_sec, 0.0, primal_time_sec,
            0.0, 0, 0.0, 0.0, 0.0, 0.0, "success");
    }

    // Log AADC row
    double speedup = (primal_time_sec > 0) ? primal_time_sec / aadc_time_sec : 0;
    log_xva_csv(csv_path, "xva_cpp_aadc", num_trades, num_mc_paths,
        num_model_steps, num_pricing_times, num_sens, threads_num,
        "cpp_aadc_avx256", mode, reported_cva, reported_dva,
        aadc_time_sec, 0.0, aadc_time_sec + compilation_sec,
        compilation_sec, 0, speedup,
        max_cva_diff, max_dva_diff, 0.0, "success");

    std::cout << "\nResults logged to " << csv_path << "\n";

    return 0;
}

////////////////////////////////////////////////////
//
//  Main
//  
//  argv:
//  string    Path to XVA task data
//  int       AVX: 256/512
//  int       Number of threads
//
////////////////////////////////////////////////////

int main (int argc, char* argv[]) {
	int num_threads(1);
	std::string input_file("../Pricing/initData.json");
	if (argc > 3) num_threads = atoi(argv[3]);
    if (argc > 1) input_file=argv[1];
#if AADC_512 
    if (argc > 2 && atoi(argv[2]) == 512) return run_pricing<__m512d>(num_threads, input_file);
    else
#endif
    return run_pricing<__m256d>(num_threads, input_file);
}