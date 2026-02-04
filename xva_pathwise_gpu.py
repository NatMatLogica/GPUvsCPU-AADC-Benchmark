"""XVA Pathwise Derivatives: GPU implementation using Numba CUDA.

Computes CVA/DVA and sensitivities in a single Monte Carlo simulation pass
using pathwise (tangent-mode) automatic differentiation.

Key advantages over bump-and-revalue:
- O(M) complexity instead of O(N×M) where N=params, M=paths
- Single kernel launch instead of N+1 launches
- Exact derivatives (no finite difference truncation error)
- Perfect GPU parallelism (paths are independent)

Sensitivity parameters computed (~284 total):
- r0 (initial short rate): 1 param, via pathwise AD
- sigma (volatility): 1 param, via pathwise AD
- Counterparty survival curve: ~141 params, analytically from exposures
- Company survival curve: ~141 params, analytically from exposures

NOT computed (would require significant kernel changes):
- Mean reversion curve θ(t): ~251 params
  (requires tracking dr/dθ[i] for each curve point per path)

For full ~535 Greeks including MR curve, use cpp_aadc (reverse-mode AD).

Version: 1.0.0
"""

import numpy as np
import math
import time
from numba import cuda, float64, int32
from numba.cuda import is_available as cuda_is_available

# Maximum theta points supported (for local array allocation)
MAX_THETA_POINTS = 512


# ---------------------------------------------------------------------------
# Device Functions: Pathwise Derivative Computation
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def pw_interp_device(times, vals, n_points, t):
    """Piecewise linear interpolation using binary search."""
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
    """Binary search returning index for interpolation."""
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
def get_interp_weights(times, n_points, t):
    """Get interpolation index and weights for a given time.

    Returns (idx_lo, idx_hi, weight_lo, weight_hi) where:
    - value = vals[idx_lo] * weight_lo + vals[idx_hi] * weight_hi
    """
    lo, hi = 0, n_points
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return 0, 0, 1.0, 0.0
    if lo >= n_points:
        return n_points - 1, n_points - 1, 1.0, 0.0

    len_t = times[lo] - times[lo - 1]
    wl = (times[lo] - t) / len_t
    wr = (t - times[lo - 1]) / len_t
    return lo - 1, lo, wl, wr


@cuda.jit(device=True)
def hw_bond_price_with_derivs(r, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                               mr_times, mr_vals, n_mr, cumulat1, cumulat2, t_years):
    """Hull-White bond price P(t,T) with pathwise derivatives.

    Returns: (P, dP_dr0, dP_dsigma)
    """
    if t_years <= time_counter:
        return 1.0, 0.0, 0.0

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
    else:
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

    P = math.exp(-A_t_T * r + C_t_T)

    # Pathwise derivatives via chain rule: dP/dθ = dP/dr × dr/dθ
    dP_dr = -A_t_T * P
    dP_dr0 = dP_dr * dr_dr0
    dP_dsigma = dP_dr * dr_dsigma

    return P, dP_dr0, dP_dsigma


@cuda.jit(device=True)
def project_curve_with_derivs(r, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                               mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                               spread_times, spread_vals, n_spread, t_qtime):
    """Projection curve value with pathwise derivatives."""
    t_years = t_qtime / 365.0
    disc, d_disc_dr0, d_disc_dsigma = hw_bond_price_with_derivs(
        r, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
        mr_times, mr_vals, n_mr, cumulat1, cumulat2, t_years
    )
    spread_val = pw_interp_device(spread_times, spread_vals, n_spread, t_years)
    exp_spread = math.exp(-spread_val * (t_years - time_counter))

    proj = disc * exp_spread
    d_proj_dr0 = d_disc_dr0 * exp_spread
    d_proj_dsigma = d_disc_dsigma * exp_spread

    return proj, d_proj_dr0, d_proj_dsigma


@cuda.jit(device=True)
def csa_update_with_derivs(collateral, d_coll_dr0, d_coll_dsigma,
                           total_price, d_price_dr0, d_price_dsigma,
                           th, tl, mta_h, mta_l):
    """CSA collateral update with subgradient derivatives.

    Returns: (new_collateral, d_coll_dr0, d_coll_dsigma)
    """
    diff = total_price - collateral
    d_diff_dr0 = d_price_dr0 - d_coll_dr0
    d_diff_dsigma = d_price_dsigma - d_coll_dsigma

    margin_high = max(diff - th, 0.0)
    margin_low = min(diff - tl, 0.0)

    # Subgradients for max/min
    d_margin_high = 1.0 if (diff - th) > 0 else 0.0
    d_margin_low = 1.0 if (diff - tl) < 0 else 0.0

    add_high = margin_high if margin_high >= mta_h else 0.0
    add_low = margin_low if margin_low < -mta_l else 0.0

    # Derivative of add_high and add_low
    d_add_high = d_margin_high if margin_high >= mta_h else 0.0
    d_add_low = d_margin_low if margin_low < -mta_l else 0.0

    new_collateral = collateral + add_low + add_high
    new_d_coll_dr0 = d_coll_dr0 + (d_add_low + d_add_high) * d_diff_dr0
    new_d_coll_dsigma = d_coll_dsigma + (d_add_low + d_add_high) * d_diff_dsigma

    return new_collateral, new_d_coll_dr0, new_d_coll_dsigma


# ---------------------------------------------------------------------------
# Main Pathwise Simulation Kernel
# ---------------------------------------------------------------------------

@cuda.jit
def simulate_xva_pathwise_kernel(
    # Random numbers: (num_paths, num_model_steps)
    randoms,
    # Model scalars
    alpha, sigma, r0,
    # Mean reversion curve
    mr_times, mr_vals, n_mr,
    # Precomputed cumulatives
    cumulat1, cumulat2,
    # Spread curves
    sp0_times, sp0_vals, n_sp0,
    sp1_times, sp1_vals, n_sp1,
    sp2_times, sp2_vals, n_sp2,
    # Simulation grid
    model_times_days, is_pricing, num_steps,
    # Trade data
    fixed_amounts, fixed_times, fixed_num_cfs,
    float_notionals, float_start_times, float_end_times,
    float_pay_times, float_spread_ids, float_num_cfs,
    num_trades, max_cf,
    # CSA params
    csa_th, csa_tl, csa_mta_h, csa_mta_l, csa_init_ct,
    # Per-path state arrays
    fwd_cache, fwd_set, fixed_first, float_first,
    # Derivative caches for forward rates
    d_fwd_dr0_cache, d_fwd_dsigma_cache,
    # Output arrays - primal
    out_pee, out_nee,
    # Output arrays - r0/sigma sensitivities
    out_d_pee_dr0, out_d_nee_dr0,
    out_d_pee_dsigma, out_d_nee_dsigma,
):
    """Pathwise derivative kernel: computes PEE/NEE and their sensitivities in one pass."""

    path_idx = cuda.grid(1)
    if path_idx >= randoms.shape[0]:
        return

    # Initialize primal state
    r_current = r0
    time_counter = 0.0
    collateral = csa_init_ct

    # Initialize pathwise derivative state
    dr_dr0 = 1.0      # dr/dr0 starts at 1 (r = r0 at t=0)
    dr_dsigma = 0.0   # dr/dsigma starts at 0

    d_collateral_dr0 = 0.0
    d_collateral_dsigma = 0.0

    pricing_idx = 0

    # Initialize per-path trade state
    for ti in range(num_trades):
        fixed_first[path_idx, ti] = 0
        float_first[path_idx, ti] = 0
        for cf in range(max_cf):
            fwd_cache[path_idx, ti, cf] = 0.0
            fwd_set[path_idx, ti, cf] = 0
            d_fwd_dr0_cache[path_idx, ti, cf] = 0.0
            d_fwd_dsigma_cache[path_idx, ti, cf] = 0.0

    for step_i in range(num_steps):
        t_days = model_times_days[step_i]
        t_years = t_days / 365.0

        if step_i > 0:
            # --- Evolve rate AND pathwise derivatives ---
            delta_t = t_years - time_counter
            S = math.exp(-alpha * delta_t)
            S2 = S * S
            V = math.sqrt((1.0 - S2) / (2.0 * alpha)) if alpha > 1e-10 else math.sqrt(delta_t)
            Z = randoms[path_idx, step_i]

            new_tc = time_counter + delta_t
            mr_val = pw_interp_device(mr_times, mr_vals, n_mr, new_tc)
            mu = (1.0 - S) * mr_val

            # Primal evolution
            r_new = r_current * S + mu + sigma * V * Z

            # Pathwise derivative evolution
            # dr(t+dt)/dr0 = S × dr(t)/dr0
            dr_dr0_new = S * dr_dr0

            # dr(t+dt)/dsigma = S × dr(t)/dsigma + V × Z
            dr_dsigma_new = S * dr_dsigma + V * Z

            r_current = r_new
            dr_dr0 = dr_dr0_new
            dr_dsigma = dr_dsigma_new
            time_counter = new_tc

        if is_pricing[step_i] == 1:
            # --- Advance first CF indices ---
            for ti in range(num_trades):
                while (fixed_first[path_idx, ti] < fixed_num_cfs[ti]
                       and fixed_times[ti, fixed_first[path_idx, ti]] < t_days):
                    fixed_first[path_idx, ti] += 1
                while (float_first[path_idx, ti] < float_num_cfs[ti]
                       and float_pay_times[ti, float_first[path_idx, ti]] < t_days):
                    float_first[path_idx, ti] += 1

            # --- Compute portfolio price AND derivatives ---
            total_price = 0.0
            d_price_dr0 = 0.0
            d_price_dsigma = 0.0

            for ti in range(num_trades):
                # Fixed leg price with derivatives
                for cf in range(fixed_first[path_idx, ti], fixed_num_cfs[ti]):
                    cf_time_years = fixed_times[ti, cf] / 365.0
                    bond_p, d_bond_dr0, d_bond_dsigma = hw_bond_price_with_derivs(
                        r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                        mr_times, mr_vals, n_mr, cumulat1, cumulat2, cf_time_years
                    )
                    amt = fixed_amounts[ti, cf]
                    total_price += amt * bond_p
                    d_price_dr0 += amt * d_bond_dr0
                    d_price_dsigma += amt * d_bond_dsigma

                # Float leg price with derivatives
                sid = float_spread_ids[ti]
                for cf in range(float_first[path_idx, ti], float_num_cfs[ti]):
                    pay_years = float_pay_times[ti, cf] / 365.0
                    disc, d_disc_dr0, d_disc_dsigma = hw_bond_price_with_derivs(
                        r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                        mr_times, mr_vals, n_mr, cumulat1, cumulat2, pay_years
                    )

                    st = float_start_times[ti, cf]
                    et = float_end_times[ti, cf]

                    if st >= t_days:
                        # Compute forward rate with derivatives
                        if sid == 0:
                            proj_st, d_proj_st_dr0, d_proj_st_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp0_times, sp0_vals, n_sp0, st
                            )
                            proj_et, d_proj_et_dr0, d_proj_et_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp0_times, sp0_vals, n_sp0, et
                            )
                        elif sid == 1:
                            proj_st, d_proj_st_dr0, d_proj_st_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp1_times, sp1_vals, n_sp1, st
                            )
                            proj_et, d_proj_et_dr0, d_proj_et_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp1_times, sp1_vals, n_sp1, et
                            )
                        else:
                            proj_st, d_proj_st_dr0, d_proj_st_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp2_times, sp2_vals, n_sp2, st
                            )
                            proj_et, d_proj_et_dr0, d_proj_et_dsigma = project_curve_with_derivs(
                                r_current, dr_dr0, dr_dsigma, sigma, alpha, time_counter,
                                mr_times, mr_vals, n_mr, cumulat1, cumulat2,
                                sp2_times, sp2_vals, n_sp2, et
                            )

                        yf = (et - st) / 365.0
                        # fwd = (proj_st / proj_et - 1) / yf
                        ratio = proj_st / proj_et
                        fwd = (ratio - 1.0) / yf

                        # d(fwd)/dθ = d(ratio)/dθ / yf
                        # d(ratio)/dθ = (d_proj_st × proj_et - proj_st × d_proj_et) / proj_et²
                        d_ratio_dr0 = (d_proj_st_dr0 * proj_et - proj_st * d_proj_et_dr0) / (proj_et * proj_et)
                        d_ratio_dsigma = (d_proj_st_dsigma * proj_et - proj_st * d_proj_et_dsigma) / (proj_et * proj_et)
                        d_fwd_dr0 = d_ratio_dr0 / yf
                        d_fwd_dsigma = d_ratio_dsigma / yf

                        # Cache for future use
                        fwd_cache[path_idx, ti, cf] = fwd
                        fwd_set[path_idx, ti, cf] = 1
                        d_fwd_dr0_cache[path_idx, ti, cf] = d_fwd_dr0
                        d_fwd_dsigma_cache[path_idx, ti, cf] = d_fwd_dsigma
                    else:
                        # Use cached forward rate
                        fwd = fwd_cache[path_idx, ti, cf]
                        d_fwd_dr0 = d_fwd_dr0_cache[path_idx, ti, cf]
                        d_fwd_dsigma = d_fwd_dsigma_cache[path_idx, ti, cf]

                    notional = float_notionals[ti, cf]
                    # price_contribution = notional × fwd × disc
                    total_price += notional * fwd * disc

                    # Product rule: d(fwd × disc) = d_fwd × disc + fwd × d_disc
                    d_price_dr0 += notional * (d_fwd_dr0 * disc + fwd * d_disc_dr0)
                    d_price_dsigma += notional * (d_fwd_dsigma * disc + fwd * d_disc_dsigma)

            # --- CSA update with derivatives ---
            collateral, d_collateral_dr0, d_collateral_dsigma = csa_update_with_derivs(
                collateral, d_collateral_dr0, d_collateral_dsigma,
                total_price, d_price_dr0, d_price_dsigma,
                csa_th, csa_tl, csa_mta_h, csa_mta_l
            )

            csa_price = total_price - collateral
            d_csa_dr0 = d_price_dr0 - d_collateral_dr0
            d_csa_dsigma = d_price_dsigma - d_collateral_dsigma

            # --- PEE/NEE with subgradient derivatives ---
            pee = max(csa_price, 0.0)
            nee = min(csa_price, 0.0)

            # Subgradient: d(max(x,0))/dx = 1 if x > 0, else 0
            d_pee_dr0 = d_csa_dr0 if csa_price > 0 else 0.0
            d_nee_dr0 = d_csa_dr0 if csa_price < 0 else 0.0
            d_pee_dsigma = d_csa_dsigma if csa_price > 0 else 0.0
            d_nee_dsigma = d_csa_dsigma if csa_price < 0 else 0.0

            # Store outputs
            out_pee[path_idx, pricing_idx] = pee
            out_nee[path_idx, pricing_idx] = nee
            out_d_pee_dr0[path_idx, pricing_idx] = d_pee_dr0
            out_d_nee_dr0[path_idx, pricing_idx] = d_nee_dr0
            out_d_pee_dsigma[path_idx, pricing_idx] = d_pee_dsigma
            out_d_nee_dsigma[path_idx, pricing_idx] = d_nee_dsigma

            pricing_idx += 1


# ---------------------------------------------------------------------------
# Host-side Functions
# ---------------------------------------------------------------------------

def compute_cva_dva_with_derivs(pee, nee, d_pee_dr0, d_nee_dr0, d_pee_dsigma, d_nee_dsigma,
                                 pricing_times_days,
                                 company_surv_times, company_surv_vals,
                                 ctrparty_surv_times, ctrparty_surv_vals,
                                 company_t0, ctrparty_t0):
    """Compute CVA/DVA and their sensitivities from pathwise exposure derivatives."""
    from xva_common import pw_interp

    # Average across paths
    avg_pee = pee.mean(axis=0)
    avg_nee = nee.mean(axis=0)
    avg_d_pee_dr0 = d_pee_dr0.mean(axis=0)
    avg_d_nee_dr0 = d_nee_dr0.mean(axis=0)
    avg_d_pee_dsigma = d_pee_dsigma.mean(axis=0)
    avg_d_nee_dsigma = d_nee_dsigma.mean(axis=0)

    n = len(pricing_times_days)

    # Evaluate survival curves at pricing times
    ctrp = np.zeros(n)
    comp = np.zeros(n)
    for i in range(n):
        t_q = int(pricing_times_days[i])
        t_y = t_q / 365.0
        comp_rate = pw_interp(company_surv_times, company_surv_vals, t_y)
        ctrp_rate = pw_interp(ctrparty_surv_times, ctrparty_surv_vals, t_y)
        comp_yf = (t_q - company_t0) / 365.0
        ctrp_yf = (t_q - ctrparty_t0) / 365.0
        comp[i] = math.exp(-comp_rate * comp_yf)
        ctrp[i] = math.exp(-ctrp_rate * ctrp_yf)

    # Trapezoidal integration for CVA/DVA
    cva = 0.0
    dva = 0.0
    d_cva_dr0 = 0.0
    d_dva_dr0 = 0.0
    d_cva_dsigma = 0.0
    d_dva_dsigma = 0.0

    for i in range(n - 1):
        d_ctrp = ctrp[i] - ctrp[i + 1]
        d_comp = comp[i] - comp[i + 1]

        cva += (avg_pee[i] + avg_pee[i + 1]) * d_ctrp * 0.5
        dva += (avg_nee[i] + avg_nee[i + 1]) * d_comp * 0.5

        d_cva_dr0 += (avg_d_pee_dr0[i] + avg_d_pee_dr0[i + 1]) * d_ctrp * 0.5
        d_dva_dr0 += (avg_d_nee_dr0[i] + avg_d_nee_dr0[i + 1]) * d_comp * 0.5

        d_cva_dsigma += (avg_d_pee_dsigma[i] + avg_d_pee_dsigma[i + 1]) * d_ctrp * 0.5
        d_dva_dsigma += (avg_d_nee_dsigma[i] + avg_d_nee_dsigma[i + 1]) * d_comp * 0.5

    return cva, dva, d_cva_dr0, d_dva_dr0, d_cva_dsigma, d_dva_dsigma


def compute_survival_curve_sensitivities(avg_pee, avg_nee, pricing_times_days,
                                          company_surv_times, company_surv_vals,
                                          ctrparty_surv_times, ctrparty_surv_vals,
                                          company_t0, ctrparty_t0):
    """Compute CVA/DVA sensitivities to survival curve parameters analytically.

    These don't require MC re-simulation - computed via chain rule from base exposures.
    """
    from xva_common import pw_interp

    n_pricing = len(pricing_times_days)
    n_ctrp = len(ctrparty_surv_vals)
    n_comp = len(company_surv_vals)

    # Evaluate survival curves at pricing times
    ctrp = np.zeros(n_pricing)
    comp = np.zeros(n_pricing)
    ctrp_yf = np.zeros(n_pricing)
    comp_yf = np.zeros(n_pricing)

    for i in range(n_pricing):
        t_q = int(pricing_times_days[i])
        t_y = t_q / 365.0
        comp_rate = pw_interp(company_surv_times, company_surv_vals, t_y)
        ctrp_rate = pw_interp(ctrparty_surv_times, ctrparty_surv_vals, t_y)
        comp_yf[i] = (t_q - company_t0) / 365.0
        ctrp_yf[i] = (t_q - ctrparty_t0) / 365.0
        comp[i] = math.exp(-comp_rate * comp_yf[i])
        ctrp[i] = math.exp(-ctrp_rate * ctrp_yf[i])

    # dCVA/d(ctrp_rate_j) via chain rule
    # CVA = Σᵢ (pee[i] + pee[i+1])/2 × (ctrp[i] - ctrp[i+1])
    # ctrp[i] = exp(-rate(tᵢ) × yfᵢ)
    # d(ctrp[i])/d(rate_j) = -yfᵢ × ctrp[i] × interp_weight(j, tᵢ)

    d_cva_d_ctrp_rate = np.zeros(n_ctrp)
    d_dva_d_comp_rate = np.zeros(n_comp)

    for i in range(n_pricing - 1):
        pee_avg = (avg_pee[i] + avg_pee[i + 1]) * 0.5
        nee_avg = (avg_nee[i] + avg_nee[i + 1]) * 0.5

        t_i = pricing_times_days[i] / 365.0
        t_ip1 = pricing_times_days[i + 1] / 365.0

        # Counterparty survival curve sensitivity (affects CVA)
        for j in range(n_ctrp):
            # Get interpolation weights
            w_i = _interp_weight(ctrparty_surv_times, j, t_i)
            w_ip1 = _interp_weight(ctrparty_surv_times, j, t_ip1)

            # d(ctrp[i])/d(rate_j) = -yf × ctrp × weight
            d_ctrp_i = -ctrp_yf[i] * ctrp[i] * w_i
            d_ctrp_ip1 = -ctrp_yf[i + 1] * ctrp[i + 1] * w_ip1

            d_cva_d_ctrp_rate[j] += pee_avg * (d_ctrp_i - d_ctrp_ip1)

        # Company survival curve sensitivity (affects DVA)
        for j in range(n_comp):
            w_i = _interp_weight(company_surv_times, j, t_i)
            w_ip1 = _interp_weight(company_surv_times, j, t_ip1)

            d_comp_i = -comp_yf[i] * comp[i] * w_i
            d_comp_ip1 = -comp_yf[i + 1] * comp[i + 1] * w_ip1

            d_dva_d_comp_rate[j] += nee_avg * (d_comp_i - d_comp_ip1)

    return d_cva_d_ctrp_rate, d_dva_d_comp_rate


def _interp_weight(times, idx, t):
    """Get interpolation weight for curve point idx at time t."""
    n = len(times)
    if n == 0:
        return 0.0

    # Find bracketing indices
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return 1.0 if idx == 0 else 0.0
    if lo >= n:
        return 1.0 if idx == n - 1 else 0.0

    # Linear interpolation weights
    t_lo = times[lo - 1]
    t_hi = times[lo]
    if abs(t_hi - t_lo) < 1e-10:
        return 1.0 if idx == lo else 0.0

    w_hi = (t - t_lo) / (t_hi - t_lo)
    w_lo = 1.0 - w_hi

    if idx == lo - 1:
        return w_lo
    elif idx == lo:
        return w_hi
    else:
        return 0.0


def run_pathwise_gpu(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                     company_surv, ctrparty_surv, mode="pricing_with_greeks",
                     block_size=256):
    """Run pathwise GPU simulation returning XVAResult with all sensitivities.

    Computes CVA/DVA and sensitivities to r0, sigma, and survival curves
    in a single Monte Carlo pass.
    """
    from xva_common import XVAResult

    if not cuda_is_available():
        print("  Pathwise GPU: CUDA not available")
        return None

    num_paths = randoms.shape[0]
    num_pricing = int(grid.is_pricing.sum())
    num_trades = trades.num_trades
    max_cf = trades.max_cf
    n_mr = len(hw.mean_rev_vals)

    print(f"  Pathwise GPU: {num_paths} paths, {len(grid.model_times)} steps, "
          f"{num_pricing} pricing times, {num_trades} trades")

    # Warm up JIT (equivalent to AADC kernel recording/compilation)
    jit_warmup_time = 0.0
    if num_paths > 1:
        print("  Warming up CUDA JIT (pathwise)...")
        t_jit_start = time.perf_counter()
        _run_pathwise_kernel_once(randoms[:1], hw, grid, trades, csa, cumulat1, cumulat2,
                                   num_pricing, block_size)
        jit_warmup_time = time.perf_counter() - t_jit_start
        print(f"  JIT warm-up done: {jit_warmup_time:.3f}s")

    t0 = time.perf_counter()

    # Transfer constant data to device
    d_randoms = cuda.to_device(randoms)
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

    # Trade data
    d_fixed_amounts = cuda.to_device(trades.fixed_amounts)
    d_fixed_times = cuda.to_device(trades.fixed_times)
    d_fixed_num_cfs = cuda.to_device(trades.fixed_num_cfs)
    d_float_notionals = cuda.to_device(trades.float_notionals)
    d_float_start = cuda.to_device(trades.float_start_times)
    d_float_end = cuda.to_device(trades.float_end_times)
    d_float_pay = cuda.to_device(trades.float_pay_times)
    d_float_spread = cuda.to_device(trades.float_spread_ids)
    d_float_num_cfs = cuda.to_device(trades.float_num_cfs)

    # Per-path state arrays
    d_fwd_cache = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)
    d_fwd_set = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.int32)
    d_fixed_first = cuda.device_array((num_paths, num_trades), dtype=np.int32)
    d_float_first = cuda.device_array((num_paths, num_trades), dtype=np.int32)
    d_d_fwd_dr0_cache = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)
    d_d_fwd_dsigma_cache = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)

    # Output arrays - primal
    d_out_pee = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_nee = cuda.device_array((num_paths, num_pricing), dtype=np.float64)

    # Output arrays - sensitivities
    d_out_d_pee_dr0 = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_nee_dr0 = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_pee_dsigma = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_nee_dsigma = cuda.device_array((num_paths, num_pricing), dtype=np.float64)

    transfer_time = time.perf_counter() - t0

    # Calculate GPU memory usage
    gpu_memory_bytes = (
        # Input arrays
        randoms.nbytes +
        hw.mean_rev_times.nbytes + hw.mean_rev_vals.nbytes +
        cumulat1.nbytes + cumulat2.nbytes +
        hw.spread_3m_times.nbytes + hw.spread_3m_vals.nbytes +
        hw.spread_6m_times.nbytes + hw.spread_6m_vals.nbytes +
        hw.spread_12m_times.nbytes + hw.spread_12m_vals.nbytes +
        grid.model_times.nbytes + grid.is_pricing.nbytes +
        trades.fixed_amounts.nbytes + trades.fixed_times.nbytes + trades.fixed_num_cfs.nbytes +
        trades.float_notionals.nbytes + trades.float_start_times.nbytes +
        trades.float_end_times.nbytes + trades.float_pay_times.nbytes +
        trades.float_spread_ids.nbytes + trades.float_num_cfs.nbytes +
        # Per-path state arrays (float64=8, int32=4)
        num_paths * num_trades * max_cf * 8 +  # d_fwd_cache
        num_paths * num_trades * max_cf * 4 +  # d_fwd_set
        num_paths * num_trades * 4 * 2 +       # d_fixed_first, d_float_first
        num_paths * num_trades * max_cf * 8 * 2 +  # d_d_fwd_dr0_cache, d_d_fwd_dsigma_cache
        # Output arrays
        num_paths * num_pricing * 8 * 6        # pee, nee, 4 sensitivity arrays
    )
    gpu_memory_mb = gpu_memory_bytes / (1024 * 1024)

    # Launch kernel
    blocks = (num_paths + block_size - 1) // block_size

    t_kernel_start = time.perf_counter()
    simulate_xva_pathwise_kernel[blocks, block_size](
        d_randoms,
        hw.alpha, hw.sigma, hw.r0,
        d_mr_times, d_mr_vals, n_mr,
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
        d_d_fwd_dr0_cache, d_d_fwd_dsigma_cache,
        d_out_pee, d_out_nee,
        d_out_d_pee_dr0, d_out_d_nee_dr0,
        d_out_d_pee_dsigma, d_out_d_nee_dsigma,
    )
    cuda.synchronize()
    kernel_time = time.perf_counter() - t_kernel_start

    # Copy results back
    pee = d_out_pee.copy_to_host()
    nee = d_out_nee.copy_to_host()
    d_pee_dr0 = d_out_d_pee_dr0.copy_to_host()
    d_nee_dr0 = d_out_d_nee_dr0.copy_to_host()
    d_pee_dsigma = d_out_d_pee_dsigma.copy_to_host()
    d_nee_dsigma = d_out_d_nee_dsigma.copy_to_host()

    # Compute CVA/DVA with sensitivities
    cva, dva, d_cva_dr0, d_dva_dr0, d_cva_dsigma, d_dva_dsigma = compute_cva_dva_with_derivs(
        pee, nee, d_pee_dr0, d_nee_dr0, d_pee_dsigma, d_nee_dsigma,
        grid.pricing_times,
        company_surv.times_years, company_surv.values,
        ctrparty_surv.times_years, ctrparty_surv.values,
        company_surv.t0, ctrparty_surv.t0
    )

    # Compute survival curve sensitivities analytically
    avg_pee = pee.mean(axis=0)
    avg_nee = nee.mean(axis=0)
    d_cva_d_ctrp, d_dva_d_comp = compute_survival_curve_sensitivities(
        avg_pee, avg_nee, grid.pricing_times,
        company_surv.times_years, company_surv.values,
        ctrparty_surv.times_years, ctrparty_surv.values,
        company_surv.t0, ctrparty_surv.t0
    )

    total_time = time.perf_counter() - t0

    # Count sensitivity parameters
    n_ctrp_surv = len(ctrparty_surv.values)
    n_comp_surv = len(company_surv.values)
    num_sens_params = 2 + n_ctrp_surv + n_comp_surv  # r0, sigma, survival curves

    # Build sensitivities dict
    sensitivities = {
        'dCVA/dr0': d_cva_dr0,
        'dDVA/dr0': d_dva_dr0,
        'dCVA/dsigma': d_cva_dsigma,
        'dDVA/dsigma': d_dva_dsigma,
        'dCVA/d_ctrp_surv': d_cva_d_ctrp,
        'dDVA/d_comp_surv': d_dva_d_comp,
    }

    print(f"  Pathwise done: CVA={cva:.10f}, DVA={dva:.10f}")
    print(f"    JIT={jit_warmup_time:.3f}s, kernel={kernel_time:.3f}s, total={total_time:.3f}s, mem={gpu_memory_mb:.1f}MB")
    print(f"    dCVA/dr0={d_cva_dr0:.6f}, dCVA/dsigma={d_cva_dsigma:.6f}")

    return XVAResult(
        backend="pathwise_gpu",
        mode=mode,
        cva=cva,
        dva=dva,
        eval_time_sec=kernel_time,              # Kernel execution time (reusable)
        sensitivity_time_sec=kernel_time,       # Greeks computed in same kernel pass (included in eval_time)
        total_time_sec=total_time,              # Full wall clock (transfer + kernel + reduction)
        kernel_recording_sec=jit_warmup_time,   # JIT compilation (like AADC kernel recording)
        gpu_kernel_time_sec=kernel_time,
        num_params_bumped=num_sens_params,      # Note: excludes MR sensitivities (not yet implemented)
        gpu_memory_mb=gpu_memory_mb,
        pee=pee,
        nee=nee,
        sensitivities=sensitivities,
    )


def _run_pathwise_kernel_once(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                               num_pricing, block_size):
    """Run kernel once for JIT warmup."""
    num_paths = randoms.shape[0]
    num_trades = trades.num_trades
    max_cf = trades.max_cf
    n_mr = len(hw.mean_rev_vals)
    num_steps = len(grid.model_times)

    d_randoms = cuda.to_device(randoms)
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
    d_fixed_amounts = cuda.to_device(trades.fixed_amounts)
    d_fixed_times = cuda.to_device(trades.fixed_times)
    d_fixed_num_cfs = cuda.to_device(trades.fixed_num_cfs)
    d_float_notionals = cuda.to_device(trades.float_notionals)
    d_float_start = cuda.to_device(trades.float_start_times)
    d_float_end = cuda.to_device(trades.float_end_times)
    d_float_pay = cuda.to_device(trades.float_pay_times)
    d_float_spread = cuda.to_device(trades.float_spread_ids)
    d_float_num_cfs = cuda.to_device(trades.float_num_cfs)

    d_fwd_cache = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)
    d_fwd_set = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.int32)
    d_fixed_first = cuda.device_array((num_paths, num_trades), dtype=np.int32)
    d_float_first = cuda.device_array((num_paths, num_trades), dtype=np.int32)
    d_d_fwd_dr0 = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)
    d_d_fwd_dsigma = cuda.device_array((num_paths, num_trades, max_cf), dtype=np.float64)

    d_out_pee = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_nee = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_pee_dr0 = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_nee_dr0 = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_pee_dsigma = cuda.device_array((num_paths, num_pricing), dtype=np.float64)
    d_out_d_nee_dsigma = cuda.device_array((num_paths, num_pricing), dtype=np.float64)

    blocks = (num_paths + block_size - 1) // block_size

    simulate_xva_pathwise_kernel[blocks, block_size](
        d_randoms,
        hw.alpha, hw.sigma, hw.r0,
        d_mr_times, d_mr_vals, n_mr,
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
        d_d_fwd_dr0, d_d_fwd_dsigma,
        d_out_pee, d_out_nee,
        d_out_d_pee_dr0, d_out_d_nee_dr0,
        d_out_d_pee_dsigma, d_out_d_nee_dsigma,
    )
    cuda.synchronize()
