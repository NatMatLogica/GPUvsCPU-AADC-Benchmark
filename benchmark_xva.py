#!/usr/bin/env python3
"""XVA Benchmark: AADC (CPU) vs Brute-Force GPU vs CPU Baseline.

Compares Hull-White Monte Carlo CVA/DVA computation across backends:
  - cpu:  Pure Python/NumPy loop (validation baseline)
  - gpu:  Numba CUDA brute-force (1 thread per MC path)
  - aadc: C++ AADC binary wrapper

Usage:
    python benchmark_xva.py --mc-paths 10000 --backends cpu gpu aadc --mode pricing_with_greeks
    python benchmark_xva.py --scale-test --backends gpu aadc

Version: 1.0.0
"""
MODEL_VERSION = "1.0.0"

import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

from xva_common import (
    load_init_data, parse_hw_model, parse_survival_curve, parse_csa,
    parse_simulation_grid, generate_portfolio, generate_randoms,
    precompute_cumulatives, pw_interp, interpolated_index,
    hw_bond_price, project_curve_eval_hw, discount_curve_eval,
    write_xva_log, build_log_row,
    HWModelParams, SurvivalCurveParams, CSAParams, TradeData,
    SimulationGrid, XVAResult,
)

BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = str(BASE_DIR / "initData.json")
LOG_FILE = str(BASE_DIR / "data" / "execution_log_xva.csv")


# ---------------------------------------------------------------------------
# CPU Baseline
# ---------------------------------------------------------------------------

def cpu_simulate_path(path_randoms, hw, grid, trades, csa, cumulat1, cumulat2):
    """Simulate one MC path on CPU. Returns (pee, nee) arrays of length num_pricing_times."""
    alpha = hw.alpha
    sigma = hw.sigma
    r_current = hw.r0
    time_counter = 0.0
    collateral = csa.c_t

    num_pricing = int(grid.is_pricing.sum())
    pee = np.zeros(num_pricing)
    nee = np.zeros(num_pricing)
    pricing_idx = 0

    num_steps = len(grid.model_times)
    num_trades = trades.num_trades

    # Forward rate cache per trade per cashflow
    fwd_cache = np.zeros((num_trades, trades.max_cf))
    fwd_set = np.zeros((num_trades, trades.max_cf), dtype=np.int32)

    # Track first active CF index per trade
    fixed_first = np.zeros(num_trades, dtype=np.int32)
    float_first = np.zeros(num_trades, dtype=np.int32)

    for step_i in range(num_steps):
        t_days = int(grid.model_times[step_i])
        t_years = t_days / 365.0

        if step_i > 0:
            delta_t = t_years - time_counter
            s_part = math.exp(-alpha * delta_t)
            new_tc = time_counter + delta_t
            mr_val = pw_interp(hw.mean_rev_times, hw.mean_rev_vals, new_tc)
            mu = (1.0 - s_part) * mr_val
            r_current = (r_current * s_part + mu
                         + sigma * path_randoms[step_i]
                         * math.sqrt((1.0 - s_part * s_part) / (2.0 * alpha)))
            time_counter = new_tc

        if grid.is_pricing[step_i]:
            # Advance first CF indices
            for ti in range(num_trades):
                while (fixed_first[ti] < trades.fixed_num_cfs[ti]
                       and trades.fixed_times[ti, fixed_first[ti]] < t_days):
                    fixed_first[ti] += 1
                while (float_first[ti] < trades.float_num_cfs[ti]
                       and trades.float_pay_times[ti, float_first[ti]] < t_days):
                    float_first[ti] += 1

            total_price = 0.0

            for ti in range(num_trades):
                # Fixed leg
                for cf in range(fixed_first[ti], trades.fixed_num_cfs[ti]):
                    cf_t_years = trades.fixed_times[ti, cf] / 365.0
                    bond_p = hw_bond_price(
                        r_current, sigma, alpha, time_counter,
                        hw.mean_rev_times, hw.mean_rev_vals,
                        cumulat1, cumulat2, cf_t_years)
                    total_price += trades.fixed_amounts[ti, cf] * bond_p

                # Float leg
                sid = int(trades.float_spread_ids[ti])
                if sid == 0:
                    sp_times, sp_vals = hw.spread_3m_times, hw.spread_3m_vals
                elif sid == 1:
                    sp_times, sp_vals = hw.spread_6m_times, hw.spread_6m_vals
                else:
                    sp_times, sp_vals = hw.spread_12m_times, hw.spread_12m_vals

                for cf in range(float_first[ti], trades.float_num_cfs[ti]):
                    pay_years = trades.float_pay_times[ti, cf] / 365.0
                    disc = hw_bond_price(
                        r_current, sigma, alpha, time_counter,
                        hw.mean_rev_times, hw.mean_rev_vals,
                        cumulat1, cumulat2, pay_years)

                    st = int(trades.float_start_times[ti, cf])
                    et = int(trades.float_end_times[ti, cf])
                    if st >= t_days:
                        proj_st = project_curve_eval_hw(
                            r_current, sigma, alpha, time_counter,
                            hw.mean_rev_times, hw.mean_rev_vals,
                            cumulat1, cumulat2, sp_times, sp_vals, st)
                        proj_et = project_curve_eval_hw(
                            r_current, sigma, alpha, time_counter,
                            hw.mean_rev_times, hw.mean_rev_vals,
                            cumulat1, cumulat2, sp_times, sp_vals, et)
                        yf = (et - st) / 365.0
                        fwd = (proj_st / proj_et - 1.0) / yf
                        fwd_cache[ti, cf] = fwd
                        fwd_set[ti, cf] = 1
                    else:
                        fwd = fwd_cache[ti, cf]

                    total_price += trades.float_notionals[ti, cf] * fwd * disc

            # CSA
            margin_high = max(total_price - collateral - csa.th, 0.0)
            margin_low = min(total_price - collateral - csa.tl, 0.0)
            add_high = 0.0 if margin_high < csa.mta_h else margin_high
            add_low = margin_low if margin_low < -csa.mta_l else 0.0
            collateral = collateral + add_low + add_high

            csa_price = total_price - collateral
            pee[pricing_idx] = max(csa_price, 0.0)
            nee[pricing_idx] = min(csa_price, 0.0)
            pricing_idx += 1

    return pee, nee


def compute_cva_dva_cpu(pee_all, nee_all, grid, company_surv, ctrparty_surv):
    """Compute CVA/DVA from averaged PEE/NEE via trapezoidal integration."""
    avg_pee = pee_all.mean(axis=0)
    avg_nee = nee_all.mean(axis=0)

    n = len(grid.pricing_times)

    # Evaluate survival curves at pricing times
    comp = np.zeros(n)
    ctrp = np.zeros(n)
    for i in range(n):
        t_q = int(grid.pricing_times[i])
        comp[i] = discount_curve_eval(company_surv, t_q)
        ctrp[i] = discount_curve_eval(ctrparty_surv, t_q)

    cva = 0.0
    dva = 0.0
    for i in range(n - 1):
        cva += (avg_pee[i] + avg_pee[i + 1]) * (ctrp[i] - ctrp[i + 1]) * 0.5
        dva += (avg_nee[i] + avg_nee[i + 1]) * (comp[i] - comp[i + 1]) * 0.5

    return cva, dva


def compute_cva_dva_cpu_from_avg(avg_pee, avg_nee, grid, company_surv, ctrparty_surv):
    """Compute CVA/DVA from pre-averaged PEE/NEE with given survival curves."""
    n = len(grid.pricing_times)

    comp = np.zeros(n)
    ctrp = np.zeros(n)
    for i in range(n):
        t_q = int(grid.pricing_times[i])
        comp[i] = discount_curve_eval(company_surv, t_q)
        ctrp[i] = discount_curve_eval(ctrparty_surv, t_q)

    cva = 0.0
    dva = 0.0
    for i in range(n - 1):
        cva += (avg_pee[i] + avg_pee[i + 1]) * (ctrp[i] - ctrp[i + 1]) * 0.5
        dva += (avg_nee[i] + avg_nee[i + 1]) * (comp[i] - comp[i + 1]) * 0.5

    return cva, dva


def run_cpu_baseline(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                     company_surv, ctrparty_surv, mode="pricing_only"):
    """Run full CPU baseline: MC simulation + CVA/DVA."""
    num_paths = randoms.shape[0]
    num_pricing = int(grid.is_pricing.sum())

    print(f"  CPU baseline: {num_paths} paths, {len(grid.model_times)} steps, "
          f"{num_pricing} pricing times, {trades.num_trades} trades")

    t0 = time.perf_counter()
    pee_all = np.zeros((num_paths, num_pricing))
    nee_all = np.zeros((num_paths, num_pricing))

    for p in range(num_paths):
        pee_all[p], nee_all[p] = cpu_simulate_path(
            randoms[p], hw, grid, trades, csa, cumulat1, cumulat2)
        if (p + 1) % max(1, num_paths // 10) == 0:
            elapsed = time.perf_counter() - t0
            print(f"    Path {p+1}/{num_paths} ({elapsed:.1f}s)")

    cva, dva = compute_cva_dva_cpu(pee_all, nee_all, grid, company_surv, ctrparty_surv)
    primal_time = time.perf_counter() - t0

    sens_time = 0.0
    num_bumped = 0
    if mode == "pricing_with_greeks":
        print("  CPU bump-and-revalue for sensitivities...")
        sens_time, num_bumped = cpu_bump_and_revalue(
            randoms, hw, grid, trades, csa, cumulat1, cumulat2,
            company_surv, ctrparty_surv, cva, dva)

    total_time = primal_time + sens_time
    print(f"  CPU done: CVA={cva:.10f}, DVA={dva:.10f}, "
          f"primal={primal_time:.2f}s, sens={sens_time:.2f}s, total={total_time:.2f}s")

    return XVAResult(
        backend="cpu_baseline", mode=mode,
        cva=cva, dva=dva,
        primal_time_sec=primal_time,
        sensitivity_time_sec=sens_time,
        total_time_sec=total_time,
        num_params_bumped=num_bumped,
        pee=pee_all, nee=nee_all,
    )


def cpu_bump_and_revalue(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                         company_surv, ctrparty_surv, base_cva, base_dva):
    """Bump-and-revalue for all sensitivity parameters on CPU."""
    bump = 1e-4
    t0 = time.perf_counter()
    num_bumped = 0
    num_paths = randoms.shape[0]
    num_pricing = int(grid.is_pricing.sum())

    # Bump r0
    hw_b = HWModelParams(
        alpha=hw.alpha, sigma=hw.sigma, r0=hw.r0 + bump,
        mean_rev_times=hw.mean_rev_times, mean_rev_vals=hw.mean_rev_vals,
        spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
        spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
        spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
    )
    pee_b = np.zeros((num_paths, num_pricing))
    nee_b = np.zeros((num_paths, num_pricing))
    for p in range(num_paths):
        pee_b[p], nee_b[p] = cpu_simulate_path(
            randoms[p], hw_b, grid, trades, csa, cumulat1, cumulat2)
    cva_b, dva_b = compute_cva_dva_cpu(pee_b, nee_b, grid, company_surv, ctrparty_surv)
    num_bumped += 1
    print(f"    r0: dCVA/dr0 = {(cva_b - base_cva) / bump:.6f}")

    # Bump sigma
    hw_b2 = HWModelParams(
        alpha=hw.alpha, sigma=hw.sigma + bump, r0=hw.r0,
        mean_rev_times=hw.mean_rev_times, mean_rev_vals=hw.mean_rev_vals,
        spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
        spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
        spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
    )
    pee_b2 = np.zeros((num_paths, num_pricing))
    nee_b2 = np.zeros((num_paths, num_pricing))
    for p in range(num_paths):
        pee_b2[p], nee_b2[p] = cpu_simulate_path(
            randoms[p], hw_b2, grid, trades, csa, cumulat1, cumulat2)
    cva_b2, dva_b2 = compute_cva_dva_cpu(pee_b2, nee_b2, grid, company_surv, ctrparty_surv)
    num_bumped += 1
    print(f"    sigma: dCVA/dsigma = {(cva_b2 - base_cva) / bump:.6f}")

    # Bump mean reversion curve points
    n_mr = len(hw.mean_rev_vals)
    print(f"    Bumping {n_mr} mean reversion points...")
    for i in range(n_mr):
        mr_bumped = hw.mean_rev_vals.copy()
        mr_bumped[i] += bump
        cum1_b, cum2_b = precompute_cumulatives(hw.mean_rev_times, mr_bumped, hw.alpha)
        hw_bi = HWModelParams(
            alpha=hw.alpha, sigma=hw.sigma, r0=hw.r0,
            mean_rev_times=hw.mean_rev_times, mean_rev_vals=mr_bumped,
            spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
            spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
            spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
        )
        pee_bi = np.zeros((num_paths, num_pricing))
        nee_bi = np.zeros((num_paths, num_pricing))
        for p in range(num_paths):
            pee_bi[p], nee_bi[p] = cpu_simulate_path(
                randoms[p], hw_bi, grid, trades, csa, cum1_b, cum2_b)
        num_bumped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"      MR point {i+1}/{n_mr} ({elapsed:.1f}s)")

    # Bump survival curves (re-integration only, matches C++ AADC sensitivity set)
    # Need base PEE/NEE for re-integration
    pee_base = np.zeros((num_paths, num_pricing))
    nee_base = np.zeros((num_paths, num_pricing))
    for p in range(num_paths):
        pee_base[p], nee_base[p] = cpu_simulate_path(
            randoms[p], hw, grid, trades, csa, cumulat1, cumulat2)
    avg_pee_base = pee_base.mean(axis=0)
    avg_nee_base = nee_base.mean(axis=0)

    n_ctrp = len(ctrparty_surv.values)
    print(f"    Bumping {n_ctrp} counterparty survival curve points (re-integration)...")
    for i in range(n_ctrp):
        bumped_vals = ctrparty_surv.values.copy()
        bumped_vals[i] += bump
        # Re-integrate with bumped counterparty survival curve
        bumped_surv = SurvivalCurveParams(
            times_days=ctrparty_surv.times_days,
            times_years=ctrparty_surv.times_years,
            values=bumped_vals, t0=ctrparty_surv.t0)
        compute_cva_dva_cpu_from_avg(avg_pee_base, avg_nee_base, grid,
                                     company_surv, bumped_surv)
        num_bumped += 1

    n_comp = len(company_surv.values)
    print(f"    Bumping {n_comp} company survival curve points (re-integration)...")
    for i in range(n_comp):
        bumped_vals = company_surv.values.copy()
        bumped_vals[i] += bump
        bumped_surv = SurvivalCurveParams(
            times_days=company_surv.times_days,
            times_years=company_surv.times_years,
            values=bumped_vals, t0=company_surv.t0)
        compute_cva_dva_cpu_from_avg(avg_pee_base, avg_nee_base, grid,
                                     bumped_surv, ctrparty_surv)
        num_bumped += 1

    sens_time = time.perf_counter() - t0
    total = num_bumped
    n_resim = 2 + n_mr  # r0 + sigma + MR points
    n_reint = n_ctrp + n_comp
    print(f"    Sensitivity: {total} params total "
          f"({n_resim} re-simulated + {n_reint} re-integrated) in {sens_time:.1f}s")
    return sens_time, num_bumped


# ---------------------------------------------------------------------------
# GPU Brute-Force
# ---------------------------------------------------------------------------

def run_gpu_bruteforce(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                       company_surv, ctrparty_surv, mode="pricing_only"):
    """Run GPU brute-force simulation."""
    try:
        from numba.cuda import is_available as cuda_is_available
        if not cuda_is_available():
            print("  GPU: CUDA not available, skipping")
            return None
        from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva
    except ImportError as e:
        print(f"  GPU: import error: {e}")
        return None

    num_paths = randoms.shape[0]
    num_pricing = int(grid.is_pricing.sum())

    print(f"  GPU brute-force: {num_paths} paths, {len(grid.model_times)} steps, "
          f"{num_pricing} pricing times, {trades.num_trades} trades")

    # Warm up CUDA JIT
    if num_paths > 1:
        print("  Warming up CUDA JIT...")
        warmup_randoms = randoms[:1].copy()
        _ = run_gpu_simulation(warmup_randoms, hw, grid, trades, csa,
                               cumulat1, cumulat2, num_pricing)
        print("  JIT warm-up done.")

    # Primal
    t0 = time.perf_counter()
    pee_gpu, nee_gpu = run_gpu_simulation(
        randoms, hw, grid, trades, csa, cumulat1, cumulat2, num_pricing)
    gpu_kernel_time = time.perf_counter() - t0

    cva, dva = compute_cva_dva(
        pee_gpu, nee_gpu, grid.pricing_times,
        company_surv.times_years, company_surv.values,
        ctrparty_surv.times_years, ctrparty_surv.values,
        company_surv.t0, ctrparty_surv.t0)
    primal_time = time.perf_counter() - t0

    sens_time = 0.0
    num_bumped = 0
    if mode == "pricing_with_greeks":
        print("  GPU bump-and-revalue for sensitivities...")
        sens_time, num_bumped = gpu_bump_and_revalue(
            randoms, hw, grid, trades, csa, cumulat1, cumulat2,
            company_surv, ctrparty_surv, cva, dva, num_pricing,
            base_pee=pee_gpu, base_nee=nee_gpu)

    total_time = primal_time + sens_time
    print(f"  GPU done: CVA={cva:.10f}, DVA={dva:.10f}, "
          f"primal={primal_time:.2f}s, kernel={gpu_kernel_time:.2f}s, "
          f"sens={sens_time:.2f}s, total={total_time:.2f}s")

    return XVAResult(
        backend="gpu_brute_force", mode=mode,
        cva=cva, dva=dva,
        primal_time_sec=primal_time,
        sensitivity_time_sec=sens_time,
        total_time_sec=total_time,
        gpu_kernel_time_sec=gpu_kernel_time,
        num_params_bumped=num_bumped,
        pee=pee_gpu, nee=nee_gpu,
    )


def gpu_bump_and_revalue(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                         company_surv, ctrparty_surv, base_cva, base_dva,
                         num_pricing, base_pee=None, base_nee=None):
    """Bump-and-revalue loop on GPU for all sensitivity parameters.

    Matches the C++ AADC sensitivity set exactly:
      - r0, sigma (require full MC re-simulation)
      - Mean reversion curve points (require full MC re-simulation)
      - Company survival curve points (re-integration only, no re-simulation)
      - Counterparty survival curve points (re-integration only, no re-simulation)
    """
    from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva

    bump = 1e-4
    t0 = time.perf_counter()
    num_resim = 0   # bumps requiring full GPU re-simulation
    num_reint = 0   # bumps requiring only re-integration (CPU-side)

    # Use pre-computed base PEE/NEE for survival curve re-integration
    if base_pee is None or base_nee is None:
        base_pee, base_nee = run_gpu_simulation(
            randoms, hw, grid, trades, csa, cumulat1, cumulat2, num_pricing)
    avg_pee_base = base_pee.mean(axis=0)
    avg_nee_base = base_nee.mean(axis=0)

    def _cva_dva(pee, nee):
        return compute_cva_dva(
            pee, nee, grid.pricing_times,
            company_surv.times_years, company_surv.values,
            ctrparty_surv.times_years, ctrparty_surv.values,
            company_surv.t0, ctrparty_surv.t0)

    def _cva_dva_with_surv(avg_pee, avg_nee, comp_vals, ctrp_vals):
        """CVA/DVA from pre-averaged exposures with custom survival curves."""
        n = len(grid.pricing_times)
        comp = np.zeros(n)
        ctrp = np.zeros(n)
        for i in range(n):
            t_q = int(grid.pricing_times[i])
            t_y = t_q / 365.0
            comp_rate = pw_interp(company_surv.times_years, comp_vals, t_y)
            ctrp_rate = pw_interp(ctrparty_surv.times_years, ctrp_vals, t_y)
            comp_yf = (t_q - company_surv.t0) / 365.0
            ctrp_yf = (t_q - ctrparty_surv.t0) / 365.0
            comp[i] = math.exp(-comp_rate * comp_yf)
            ctrp[i] = math.exp(-ctrp_rate * ctrp_yf)
        cva = 0.0
        dva = 0.0
        for i in range(n - 1):
            cva += (avg_pee[i] + avg_pee[i + 1]) * (ctrp[i] - ctrp[i + 1]) * 0.5
            dva += (avg_nee[i] + avg_nee[i + 1]) * (comp[i] - comp[i + 1]) * 0.5
        return cva, dva

    # Bump r0
    hw_b = HWModelParams(
        alpha=hw.alpha, sigma=hw.sigma, r0=hw.r0 + bump,
        mean_rev_times=hw.mean_rev_times, mean_rev_vals=hw.mean_rev_vals,
        spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
        spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
        spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
    )
    pee_b, nee_b = run_gpu_simulation(randoms, hw_b, grid, trades, csa,
                                       cumulat1, cumulat2, num_pricing)
    cva_b, dva_b = _cva_dva(pee_b, nee_b)
    num_resim += 1
    print(f"    r0: dCVA/dr0 = {(cva_b - base_cva) / bump:.6f}")

    # Bump sigma
    hw_b2 = HWModelParams(
        alpha=hw.alpha, sigma=hw.sigma + bump, r0=hw.r0,
        mean_rev_times=hw.mean_rev_times, mean_rev_vals=hw.mean_rev_vals,
        spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
        spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
        spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
    )
    pee_b2, nee_b2 = run_gpu_simulation(randoms, hw_b2, grid, trades, csa,
                                          cumulat1, cumulat2, num_pricing)
    cva_b2, dva_b2 = _cva_dva(pee_b2, nee_b2)
    num_resim += 1
    print(f"    sigma: dCVA/dsigma = {(cva_b2 - base_cva) / bump:.6f}")

    # Bump mean reversion curve points (requires full re-simulation)
    n_mr = len(hw.mean_rev_vals)
    print(f"    Bumping {n_mr} mean reversion points on GPU...")
    for i in range(n_mr):
        mr_bumped = hw.mean_rev_vals.copy()
        mr_bumped[i] += bump
        cum1_b, cum2_b = precompute_cumulatives(hw.mean_rev_times, mr_bumped, hw.alpha)
        hw_bi = HWModelParams(
            alpha=hw.alpha, sigma=hw.sigma, r0=hw.r0,
            mean_rev_times=hw.mean_rev_times, mean_rev_vals=mr_bumped,
            spread_3m_times=hw.spread_3m_times, spread_3m_vals=hw.spread_3m_vals,
            spread_6m_times=hw.spread_6m_times, spread_6m_vals=hw.spread_6m_vals,
            spread_12m_times=hw.spread_12m_times, spread_12m_vals=hw.spread_12m_vals,
        )
        run_gpu_simulation(randoms, hw_bi, grid, trades, csa,
                           cum1_b, cum2_b, num_pricing)
        num_resim += 1
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"      MR point {i+1}/{n_mr} ({elapsed:.1f}s)")

    # Bump counterparty survival curve points (re-integration only, no re-simulation)
    n_ctrp = len(ctrparty_surv.values)
    print(f"    Bumping {n_ctrp} counterparty survival curve points (re-integration)...")
    for i in range(n_ctrp):
        bumped_vals = ctrparty_surv.values.copy()
        bumped_vals[i] += bump
        cva_s, dva_s = _cva_dva_with_surv(
            avg_pee_base, avg_nee_base,
            company_surv.values, bumped_vals)
        num_reint += 1

    # Bump company survival curve points (re-integration only, no re-simulation)
    n_comp = len(company_surv.values)
    print(f"    Bumping {n_comp} company survival curve points (re-integration)...")
    for i in range(n_comp):
        bumped_vals = company_surv.values.copy()
        bumped_vals[i] += bump
        cva_s, dva_s = _cva_dva_with_surv(
            avg_pee_base, avg_nee_base,
            bumped_vals, ctrparty_surv.values)
        num_reint += 1

    sens_time = time.perf_counter() - t0
    total_bumped = num_resim + num_reint
    print(f"    GPU sensitivity: {total_bumped} params total "
          f"({num_resim} re-simulated + {num_reint} re-integrated) in {sens_time:.1f}s")
    return sens_time, total_bumped


# ---------------------------------------------------------------------------
# AADC C++ Wrapper
# ---------------------------------------------------------------------------

def run_aadc_cpp(input_file, num_mc_paths, num_threads, mode="pricing_only"):
    """Build and run C++ AADC binary, parse results."""
    build_dir = BASE_DIR / "build"
    binary = build_dir / "xva_server"

    if not binary.exists():
        print("  AADC: binary not found, attempting build...")
        build_dir.mkdir(exist_ok=True)
        # Check for AADC SDK
        aadc_dir = os.environ.get("AADC_SOURCE_DIR", os.path.expanduser("~/aadc_sdk"))
        if not os.path.isdir(aadc_dir):
            print(f"  AADC: AADC_SOURCE_DIR={aadc_dir} not found, skipping")
            return None
        try:
            subprocess.run(
                ["cmake", "..", f"-DAADC_SOURCE_DIR={aadc_dir}"],
                cwd=str(build_dir), check=True,
                capture_output=True, text=True)
            subprocess.run(
                ["make", "-j4"],
                cwd=str(build_dir), check=True,
                capture_output=True, text=True)
            print("  AADC: build successful")
        except subprocess.CalledProcessError as e:
            print(f"  AADC: build failed: {e.stderr[:200]}")
            return None

    if not binary.exists():
        print("  AADC: binary not found after build, skipping")
        return None

    # Run: ./xva_server initData.json <mc_paths> <threads>
    cmd = [str(binary), input_file, str(num_mc_paths), str(num_threads)]
    print(f"  AADC: running {' '.join(cmd)}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, cwd=str(BASE_DIR))
    except subprocess.TimeoutExpired:
        print("  AADC: timed out (600s)")
        return None
    except Exception as e:
        print(f"  AADC: execution error: {e}")
        return None

    total_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  AADC: non-zero exit code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    # Parse output for timing
    stdout = result.stdout
    primal_time = 0.0
    kernel_recording = 0.0
    sens_time = 0.0
    for line in stdout.split("\n"):
        if "Primal" in line and "ms" in line:
            try:
                primal_time = float(line.split()[-2]) / 1000.0
            except (ValueError, IndexError):
                pass
        if "Compilation" in line and "ms" in line:
            try:
                kernel_recording = float(line.split()[-2]) / 1000.0
            except (ValueError, IndexError):
                pass

    # Parse all_results.json for CVA/DVA
    results_file = BASE_DIR / "all_results.json"
    if not results_file.exists():
        print("  AADC: all_results.json not found")
        return None

    with open(results_file) as f:
        res = json.load(f)

    primal = res.get("Primal results", {})
    cva = primal.get("CVA", 0.0)
    dva = primal.get("DVA", 0.0)

    if primal_time == 0.0:
        primal_time = primal.get("Computation_time_ms", 0.0) / 1000.0

    aadc_res = res.get("AADC results", {})
    aadc_cva = aadc_res.get("CVA", cva)
    aadc_dva = aadc_res.get("DVA", dva)

    # Use AADC values if non-zero, otherwise primal
    final_cva = aadc_cva if aadc_cva != 0.0 else cva
    final_dva = aadc_dva if aadc_dva != 0.0 else dva

    sens_time = total_time - primal_time - kernel_recording
    if sens_time < 0:
        sens_time = 0.0

    print(f"  AADC done: CVA={final_cva:.10f}, DVA={final_dva:.10f}, "
          f"primal={primal_time:.2f}s, recording={kernel_recording:.2f}s, total={total_time:.2f}s")

    return XVAResult(
        backend="aadc_cpu", mode=mode,
        cva=final_cva, dva=final_dva,
        primal_time_sec=primal_time,
        sensitivity_time_sec=sens_time,
        total_time_sec=total_time,
        kernel_recording_sec=kernel_recording,
    )


# ---------------------------------------------------------------------------
# Reference Values from all_results.json
# ---------------------------------------------------------------------------

def load_reference_results():
    """Load C++ reference CVA/DVA from all_results.json if available."""
    results_file = BASE_DIR / "all_results.json"
    if not results_file.exists():
        return None, None
    with open(results_file) as f:
        res = json.load(f)
    primal = res.get("Primal results", {})
    return primal.get("CVA", None), primal.get("DVA", None)


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------

def print_results(results, ref_cva, ref_dva, num_paths, num_trades,
                  num_steps, num_pricing, num_sens_params):
    """Print formatted comparison table."""
    print()
    print("=" * 80)
    print("                         XVA Benchmark Results")
    print("=" * 80)
    print(f"Config: {num_paths:,} paths, {num_trades} trades, "
          f"{num_steps} steps, {num_pricing} pricing times")
    if ref_cva is not None:
        print(f"C++ Reference: CVA={ref_cva:.10f}, DVA={ref_dva:.10f}")
    print()

    # Find CPU baseline time for speedup
    cpu_time = 0.0
    for r in results:
        if r.backend == "cpu_baseline":
            cpu_time = r.total_time_sec
            break

    print(f"{'Backend':<22} {'Primal':>10} {'Sens':>10} {'Total':>10} {'Speedup':>10}")
    print("-" * 62)
    for r in results:
        speedup = cpu_time / r.total_time_sec if r.total_time_sec > 0 and cpu_time > 0 else 0.0
        speedup_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        if r.backend == "cpu_baseline":
            speedup_str = "1.0x"
        print(f"{r.backend:<22} {r.primal_time_sec:>9.2f}s {r.sensitivity_time_sec:>9.2f}s "
              f"{r.total_time_sec:>9.2f}s {speedup_str:>10}")

    print()
    print("CVA/DVA Comparison:")
    print("-" * 80)
    for r in results:
        cva_diff = abs(r.cva - ref_cva) if ref_cva is not None else 0.0
        dva_diff = abs(r.dva - ref_dva) if ref_dva is not None else 0.0
        ref_str = ""
        if ref_cva is not None and r.backend != "cpu_baseline":
            ref_str = f"  (cva_diff={cva_diff:.2e}, dva_diff={dva_diff:.2e})"
        print(f"  {r.backend:<20} CVA={r.cva:>14.10f}  DVA={r.dva:>14.10f}{ref_str}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XVA Benchmark: AADC vs GPU vs CPU")
    parser.add_argument("--mc-paths", type=int, default=None,
                        help="Number of MC paths (default: from JSON MCPaths)")
    parser.add_argument("--trades", type=int, default=None,
                        help="Override number of trades (default: from JSON)")
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads for AADC (default: 1)")
    parser.add_argument("--backends", nargs="+", default=["cpu", "gpu", "aadc"],
                        choices=["cpu", "gpu", "aadc"],
                        help="Backends to run (default: cpu gpu aadc)")
    parser.add_argument("--mode", default="pricing_only",
                        choices=["pricing_only", "pricing_with_greeks"],
                        help="Calculation mode")
    parser.add_argument("--input-file", default=DEFAULT_INPUT,
                        help="Path to initData.json")
    parser.add_argument("--scale-test", action="store_true",
                        help="Run scaling test: 16, 1K, 10K, 100K paths")
    parser.add_argument("--seed", type=int, default=17,
                        help="Random seed (default: 17)")
    parser.add_argument("--fast-rng", action="store_true",
                        help="Use NumPy RNG (fast but won't match C++)")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip CSV logging")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.input_file}...")
    data = load_init_data(args.input_file)
    hw = parse_hw_model(data)
    t0_days = data.get("t0", 0)
    company_surv = parse_survival_curve(data["CompanySurvivalCurve"], t0_days)
    ctrparty_surv = parse_survival_curve(data["CounterPartySurvivalCurve"], t0_days)
    csa = parse_csa(data["csa"])
    grid = parse_simulation_grid(data["ModelAndPricingTimes"], t0_days)

    num_trades = args.trades or data["Portfolio"]["NumRandomTrades"]
    num_periods = data["Portfolio"]["NumPeriods"]
    mc_paths = args.mc_paths or data.get("MCPaths", 16)
    trades = generate_portfolio(t0_days, num_trades, num_periods, seed=args.seed)

    cumulat1, cumulat2 = precompute_cumulatives(hw.mean_rev_times, hw.mean_rev_vals, hw.alpha)

    num_steps = len(grid.model_times)
    num_pricing = int(grid.is_pricing.sum())
    n_mr = len(hw.mean_rev_vals)
    n_surv = len(company_surv.values) + len(ctrparty_surv.values)
    num_sens_params = 2 + n_mr + n_surv  # r0 + sigma + MR curve + survival curves

    ref_cva, ref_dva = load_reference_results()

    print(f"Grid: {num_steps} model steps, {num_pricing} pricing times")
    print(f"Portfolio: {num_trades} trades, {num_periods} CFs each")
    print(f"Sensitivity params: {num_sens_params} (r0, sigma, {n_mr} MR, "
          f"{n_surv} survival)")
    if ref_cva is not None:
        print(f"C++ reference: CVA={ref_cva:.10f}, DVA={ref_dva:.10f}")

    if args.scale_test:
        path_counts = [16, 1000, 10000, 100000]
        for n_paths in path_counts:
            print(f"\n{'='*60}")
            print(f"  Scale test: {n_paths:,} paths")
            print(f"{'='*60}")
            _run_benchmark(n_paths, args, hw, grid, trades, csa,
                           cumulat1, cumulat2, company_surv, ctrparty_surv,
                           num_steps, num_pricing, num_sens_params,
                           ref_cva, ref_dva)
    else:
        _run_benchmark(mc_paths, args, hw, grid, trades, csa,
                       cumulat1, cumulat2, company_surv, ctrparty_surv,
                       num_steps, num_pricing, num_sens_params,
                       ref_cva, ref_dva)


def _run_benchmark(num_paths, args, hw, grid, trades, csa,
                   cumulat1, cumulat2, company_surv, ctrparty_surv,
                   num_steps, num_pricing, num_sens_params,
                   ref_cva, ref_dva):
    """Run benchmark for a given path count."""
    use_fast = args.fast_rng or num_paths > 256
    rng_label = "numpy" if use_fast else "mt19937_64"
    print(f"\nGenerating {num_paths:,} paths of random numbers ({rng_label})...")
    t_gen = time.perf_counter()
    randoms = generate_randoms(num_paths, num_steps, seed=args.seed, fast=use_fast)
    print(f"  Random generation: {time.perf_counter() - t_gen:.2f}s")

    results = []

    if "cpu" in args.backends:
        print(f"\n--- CPU Baseline ---")
        cpu_result = run_cpu_baseline(
            randoms, hw, grid, trades, csa, cumulat1, cumulat2,
            company_surv, ctrparty_surv, mode=args.mode)
        results.append(cpu_result)

    if "gpu" in args.backends:
        print(f"\n--- GPU Brute-Force ---")
        gpu_result = run_gpu_bruteforce(
            randoms, hw, grid, trades, csa, cumulat1, cumulat2,
            company_surv, ctrparty_surv, mode=args.mode)
        if gpu_result:
            results.append(gpu_result)

    if "aadc" in args.backends:
        print(f"\n--- AADC C++ ---")
        aadc_result = run_aadc_cpp(args.input_file, num_paths,
                                    args.threads, mode=args.mode)
        if aadc_result:
            results.append(aadc_result)

    if results:
        print_results(results, ref_cva, ref_dva, num_paths,
                      trades.num_trades, num_steps, num_pricing, num_sens_params)

        # CSV logging
        if not args.no_log:
            cpu_time = 0.0
            for r in results:
                if r.backend == "cpu_baseline":
                    cpu_time = r.total_time_sec
            log_rows = []
            for r in results:
                row = build_log_row(
                    r, grid, trades, num_paths, args.threads,
                    num_sens_params, cpu_time=cpu_time,
                    ref_cva=ref_cva, ref_dva=ref_dva)
                log_rows.append(row)
            write_xva_log(LOG_FILE, log_rows)
            print(f"\nResults logged to {LOG_FILE}")


if __name__ == "__main__":
    main()
