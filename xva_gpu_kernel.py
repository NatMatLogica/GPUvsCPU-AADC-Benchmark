"""XVA Benchmark: Numba CUDA kernel for Hull-White Monte Carlo simulation.

Ports the C++ XVA-Benchmark simulation to GPU:
  - 1 GPU thread per MC path
  - HW rate evolution + bond pricing + swap pricing + CSA + PEE/NEE
  - CVA/DVA computed on CPU from averaged exposures

Supports arbitrary portfolio sizes via device global memory for per-path state.

Version: 1.1.0
"""
MODEL_VERSION = "1.1.0"

import numpy as np
import math
from numba import cuda, float64, int32
from numba.cuda import is_available as cuda_is_available

# ---------------------------------------------------------------------------
# Device Functions
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def pw_interp_device(times, vals, n_points, t):
    """Piecewise linear interpolation using binary search.
    Matches C++ PiecewiseLinearCurve::operator()."""
    lo, hi = 0, n_points
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return vals[0]
    if lo >= n_points:
        return vals[n_points - 1]
    len_t = times[lo] - times[lo - 1]
    wl = (times[lo] - t) / len_t
    wr = (t - times[lo - 1]) / len_t
    return wl * vals[lo - 1] + wr * vals[lo]


@cuda.jit(device=True)
def interpolated_index_device(times, n_points, t):
    """Binary search returning index matching C++ interpolatedIndex."""
    lo, hi = 0, n_points
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    if lo >= n_points:
        lo = n_points - 1
    return lo


@cuda.jit(device=True)
def hw_bond_price_device(r_current, sigma, alpha, time_counter,
                         mr_times, mr_vals, n_mr,
                         cumulat1, cumulat2,
                         t_years):
    """Hull-White zero-coupon bond price.
    Port of HullWhiteZDBCurve::bond()."""
    s_part = math.exp(-alpha * (t_years - time_counter))
    A_t_T = (1.0 - s_part) / alpha

    T_index = interpolated_index_device(mr_times, n_mr, t_years)
    t_cur_index = interpolated_index_device(mr_times, n_mr, time_counter)

    C_t_T = sigma * sigma / (2.0 * alpha * alpha) * (
        t_years - time_counter
        + 1.0 / (2.0 * alpha) * (1.0 - s_part * s_part)
        + 2.0 / alpha * (s_part - 1.0)
    )

    mr_at_t_cur = mr_vals[t_cur_index]

    if T_index == t_cur_index:
        integral = (t_years - time_counter - (1.0 - s_part) / alpha) * mr_at_t_cur
        C_t_T -= integral
        return math.exp(-A_t_T * r_current + C_t_T)

    # First segment
    delta_t = mr_times[t_cur_index] - time_counter
    integral = (delta_t - (
        math.exp(-alpha * (t_years - mr_times[t_cur_index])) - s_part
    ) / alpha) * mr_at_t_cur
    C_t_T -= integral

    # Middle segments via cumulatives
    if T_index - t_cur_index > 1:
        C_t_T -= (cumulat1[T_index - 1] - cumulat1[t_cur_index]
                  + (cumulat2[T_index - 1] - cumulat2[t_cur_index])
                  * math.exp(-alpha * t_years))

    # Last segment
    delta_t_last = t_years - mr_times[T_index - 1]
    mr_at_T = mr_vals[T_index]
    integral_last = (delta_t_last - (1.0 - math.exp(-alpha * delta_t_last)) / alpha) * mr_at_T
    C_t_T -= integral_last

    return math.exp(-A_t_T * r_current + C_t_T)


@cuda.jit(device=True)
def project_curve_device(r_current, sigma, alpha, time_counter,
                         mr_times, mr_vals, n_mr,
                         cumulat1, cumulat2,
                         spread_times, spread_vals, n_spread,
                         t_qtime):
    """ProjectCurve::operator()(qtime).
    Returns discount(t) * exp(-spread(t) * (t_years - time_counter))."""
    t_years = t_qtime / 365.0
    disc = hw_bond_price_device(r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr,
                                cumulat1, cumulat2, t_years)
    spread_val = pw_interp_device(spread_times, spread_vals, n_spread, t_years)
    return disc * math.exp(-spread_val * (t_years - time_counter))


@cuda.jit(device=True)
def csa_next_device(collateral, total_price, th, tl, mta_h, mta_l):
    """CSA collateral update. Port of CSARules::next()."""
    margin_high = max(total_price - collateral - th, 0.0)
    margin_low = min(total_price - collateral - tl, 0.0)
    add_high = 0.0 if margin_high < mta_h else margin_high
    add_low = margin_low if margin_low < -mta_l else 0.0
    return collateral + add_low + add_high


# ---------------------------------------------------------------------------
# Main Simulation Kernel
# ---------------------------------------------------------------------------

@cuda.jit
def simulate_xva_kernel(
    # Random numbers: (num_paths, num_model_steps)
    randoms,
    # Model scalars
    alpha, sigma, r0,
    # Mean reversion curve
    mr_times, mr_vals, n_mr,
    # Precomputed cumulatives (n_mr,)
    cumulat1, cumulat2,
    # 3 spread curves: times and vals arrays + lengths
    sp0_times, sp0_vals, n_sp0,
    sp1_times, sp1_vals, n_sp1,
    sp2_times, sp2_vals, n_sp2,
    # Simulation grid
    model_times_days,   # (num_steps,) int32
    is_pricing,         # (num_steps,) int32 (0/1)
    num_steps,
    # Fixed leg data: (num_trades, max_cf)
    fixed_amounts, fixed_times, fixed_num_cfs,
    # Float leg data: (num_trades, max_cf)
    float_notionals, float_start_times, float_end_times,
    float_pay_times, float_spread_ids, float_num_cfs,
    num_trades, max_cf,
    # CSA params
    csa_th, csa_tl, csa_mta_h, csa_mta_l, csa_init_ct,
    # Per-path state in device global memory (indexed by path within batch)
    fwd_cache,    # (batch_size, num_trades, max_cf)
    fwd_set,      # (batch_size, num_trades, max_cf) int32
    fixed_first,  # (batch_size, num_trades) int32
    float_first,  # (batch_size, num_trades) int32
    # Output: (num_paths, num_pricing_times)
    out_pee, out_nee,
):
    path_idx = cuda.grid(1)
    if path_idx >= randoms.shape[0]:
        return

    r_current = r0
    time_counter = 0.0
    collateral = csa_init_ct
    pricing_idx = 0

    # Initialize per-path state
    for ti in range(num_trades):
        fixed_first[path_idx, ti] = 0
        float_first[path_idx, ti] = 0
        for cf in range(max_cf):
            fwd_cache[path_idx, ti, cf] = 0.0
            fwd_set[path_idx, ti, cf] = 0

    for step_i in range(num_steps):
        t_days = model_times_days[step_i]
        t_years = t_days / 365.0

        if step_i > 0:
            delta_t = t_years - time_counter
            s_part = math.exp(-alpha * delta_t)
            new_tc = time_counter + delta_t
            mr_val = pw_interp_device(mr_times, mr_vals, n_mr, new_tc)
            mu = (1.0 - s_part) * mr_val
            r_current = (r_current * s_part + mu
                         + sigma * randoms[path_idx, step_i]
                         * math.sqrt((1.0 - s_part * s_part) / (2.0 * alpha)))
            time_counter = new_tc

        if is_pricing[step_i] == 1:
            # Advance first CF indices
            for ti in range(num_trades):
                while (fixed_first[path_idx, ti] < fixed_num_cfs[ti]
                       and fixed_times[ti, fixed_first[path_idx, ti]] < t_days):
                    fixed_first[path_idx, ti] += 1
                while (float_first[path_idx, ti] < float_num_cfs[ti]
                       and float_pay_times[ti, float_first[path_idx, ti]] < t_days):
                    float_first[path_idx, ti] += 1

            total_price = 0.0

            for ti in range(num_trades):
                # Fixed leg price
                for cf in range(fixed_first[path_idx, ti], fixed_num_cfs[ti]):
                    cf_time_years = fixed_times[ti, cf] / 365.0
                    bond_p = hw_bond_price_device(
                        r_current, sigma, alpha, time_counter,
                        mr_times, mr_vals, n_mr,
                        cumulat1, cumulat2, cf_time_years)
                    total_price += fixed_amounts[ti, cf] * bond_p

                # Float leg price
                sid = float_spread_ids[ti]
                for cf in range(float_first[path_idx, ti], float_num_cfs[ti]):
                    # Discount factor
                    pay_years = float_pay_times[ti, cf] / 365.0
                    disc = hw_bond_price_device(
                        r_current, sigma, alpha, time_counter,
                        mr_times, mr_vals, n_mr,
                        cumulat1, cumulat2, pay_years)

                    # Forward rate
                    st = float_start_times[ti, cf]
                    et = float_end_times[ti, cf]
                    if st >= t_days:
                        # Compute projection curve values
                        if sid == 0:
                            proj_st = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp0_times, sp0_vals, n_sp0, st)
                            proj_et = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp0_times, sp0_vals, n_sp0, et)
                        elif sid == 1:
                            proj_st = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp1_times, sp1_vals, n_sp1, st)
                            proj_et = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp1_times, sp1_vals, n_sp1, et)
                        else:
                            proj_st = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp2_times, sp2_vals, n_sp2, st)
                            proj_et = project_curve_device(
                                r_current, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp2_times, sp2_vals, n_sp2, et)
                        yf = (et - st) / 365.0
                        fwd = (proj_st / proj_et - 1.0) / yf
                        fwd_cache[path_idx, ti, cf] = fwd
                        fwd_set[path_idx, ti, cf] = 1
                    else:
                        fwd = fwd_cache[path_idx, ti, cf]

                    total_price += float_notionals[ti, cf] * fwd * disc

            # CSA update
            collateral = csa_next_device(collateral, total_price,
                                         csa_th, csa_tl, csa_mta_h, csa_mta_l)
            csa_price = total_price - collateral
            out_pee[path_idx, pricing_idx] = max(csa_price, 0.0)
            out_nee[path_idx, pricing_idx] = min(csa_price, 0.0)
            pricing_idx += 1


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------

def compute_cva_dva(pee: np.ndarray, nee: np.ndarray,
                    pricing_times_days: np.ndarray,
                    company_surv_times_years: np.ndarray,
                    company_surv_vals: np.ndarray,
                    ctrparty_surv_times_years: np.ndarray,
                    ctrparty_surv_vals: np.ndarray,
                    company_t0: int, ctrparty_t0: int):
    """Average PEE/NEE across paths, trapezoidal integration.
    Matches C++ computeXVAMeasures."""
    from xva_common import pw_interp

    avg_pee = pee.mean(axis=0)
    avg_nee = nee.mean(axis=0)

    n = len(pricing_times_days)
    cva = 0.0
    dva = 0.0

    # Evaluate survival curves at pricing times
    ctrp = np.zeros(n)
    comp = np.zeros(n)
    for i in range(n):
        t_q = int(pricing_times_days[i])
        t_y = t_q / 365.0
        comp_rate = pw_interp(company_surv_times_years, company_surv_vals, t_y)
        ctrp_rate = pw_interp(ctrparty_surv_times_years, ctrparty_surv_vals, t_y)
        comp_yf = (t_q - company_t0) / 365.0
        ctrp_yf = (t_q - ctrparty_t0) / 365.0
        comp[i] = math.exp(-comp_rate * comp_yf)
        ctrp[i] = math.exp(-ctrp_rate * ctrp_yf)

    for i in range(n - 1):
        cva += (avg_pee[i] + avg_pee[i + 1]) * (ctrp[i] - ctrp[i + 1]) * 0.5
        dva += (avg_nee[i] + avg_nee[i + 1]) * (comp[i] - comp[i + 1]) * 0.5

    return cva, dva


def _estimate_batch_size(num_trades, max_cf, num_paths, gpu_mem_bytes=None):
    """Estimate batch size to fit per-path state arrays in GPU memory."""
    if gpu_mem_bytes is None:
        try:
            device = cuda.get_current_device()
            gpu_mem_bytes = device.total_memory
        except Exception:
            gpu_mem_bytes = 4 * 1024**3  # assume 4GB

    # Reserve 20% for kernel overhead, output arrays, curve data
    usable = int(gpu_mem_bytes * 0.6)

    # Per-path state: fwd_cache(f64) + fwd_set(i32) + fixed_first(i32) + float_first(i32)
    per_path = (num_trades * max_cf * 8      # fwd_cache float64
                + num_trades * max_cf * 4    # fwd_set int32
                + num_trades * 4             # fixed_first int32
                + num_trades * 4)            # float_first int32

    if per_path == 0:
        return num_paths

    batch = usable // per_path
    batch = max(256, min(batch, num_paths))
    return batch


def run_gpu_simulation(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                       num_pricing_times, block_size=256):
    """Launch GPU kernel and return PEE/NEE arrays.
    Automatically batches paths if per-path state exceeds GPU memory."""
    num_paths = randoms.shape[0]
    num_trades = trades.num_trades
    max_cf = trades.max_cf

    batch_size = _estimate_batch_size(num_trades, max_cf, num_paths)

    # Transfer constant data to device (shared across batches)
    d_mr_times = cuda.to_device(hw.mean_rev_times)
    d_mr_vals = cuda.to_device(hw.mean_rev_vals)
    d_cumulat1 = cuda.to_device(cumulat1)
    d_cumulat2 = cuda.to_device(cumulat2)
    d_sp0_times = cuda.to_device(hw.spread_3m_times)
    d_sp0_vals = cuda.to_device(hw.spread_3m_vals)
    d_sp1_times = cuda.to_device(hw.spread_6m_times)
    d_sp1_vals = cuda.to_device(hw.spread_6m_vals)
    d_sp2_times = cuda.to_device(hw.spread_12m_times)
    d_sp2_vals = cuda.to_device(hw.spread_12m_vals)
    d_model_times = cuda.to_device(grid.model_times)
    d_is_pricing = cuda.to_device(grid.is_pricing.astype(np.int32))
    num_steps = len(grid.model_times)
    d_fixed_amounts = cuda.to_device(trades.fixed_amounts)
    d_fixed_times = cuda.to_device(trades.fixed_times)
    d_fixed_num_cfs = cuda.to_device(trades.fixed_num_cfs)
    d_float_notionals = cuda.to_device(trades.float_notionals)
    d_float_start = cuda.to_device(trades.float_start_times)
    d_float_end = cuda.to_device(trades.float_end_times)
    d_float_pay = cuda.to_device(trades.float_pay_times)
    d_float_spread = cuda.to_device(trades.float_spread_ids)
    d_float_num_cfs = cuda.to_device(trades.float_num_cfs)

    # Allocate per-path state arrays (reused across batches)
    actual_batch = min(batch_size, num_paths)
    d_fwd_cache = cuda.device_array((actual_batch, num_trades, max_cf), dtype=np.float64)
    d_fwd_set = cuda.device_array((actual_batch, num_trades, max_cf), dtype=np.int32)
    d_fixed_first = cuda.device_array((actual_batch, num_trades), dtype=np.int32)
    d_float_first = cuda.device_array((actual_batch, num_trades), dtype=np.int32)

    # Output on host
    all_pee = np.zeros((num_paths, num_pricing_times), dtype=np.float64)
    all_nee = np.zeros((num_paths, num_pricing_times), dtype=np.float64)

    # Process in batches
    offset = 0
    while offset < num_paths:
        batch_n = min(actual_batch, num_paths - offset)

        d_randoms_batch = cuda.to_device(randoms[offset:offset + batch_n])
        d_out_pee = cuda.device_array((batch_n, num_pricing_times), dtype=np.float64)
        d_out_nee = cuda.device_array((batch_n, num_pricing_times), dtype=np.float64)

        blocks = (batch_n + block_size - 1) // block_size

        simulate_xva_kernel[blocks, block_size](
            d_randoms_batch,
            hw.alpha, hw.sigma, hw.r0,
            d_mr_times, d_mr_vals, len(hw.mean_rev_times),
            d_cumulat1, d_cumulat2,
            d_sp0_times, d_sp0_vals, len(hw.spread_3m_times),
            d_sp1_times, d_sp1_vals, len(hw.spread_6m_times),
            d_sp2_times, d_sp2_vals, len(hw.spread_12m_times),
            d_model_times, d_is_pricing, num_steps,
            d_fixed_amounts, d_fixed_times, d_fixed_num_cfs,
            d_float_notionals, d_float_start, d_float_end,
            d_float_pay, d_float_spread, d_float_num_cfs,
            num_trades, max_cf,
            csa.th, csa.tl, csa.mta_h, csa.mta_l, csa.c_t,
            d_fwd_cache, d_fwd_set, d_fixed_first, d_float_first,
            d_out_pee, d_out_nee,
        )
        cuda.synchronize()

        all_pee[offset:offset + batch_n] = d_out_pee.copy_to_host()
        all_nee[offset:offset + batch_n] = d_out_nee.copy_to_host()
        offset += batch_n

    return all_pee, all_nee
