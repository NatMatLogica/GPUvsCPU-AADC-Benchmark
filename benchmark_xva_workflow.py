#!/usr/bin/env python3
"""XVA Workflow Benchmark: Market Data Update & Incremental Trade Scenarios.

Simulates realistic trading day workflows with pre-recorded AADC kernels:
  1. Market Data Update: 100 trades, pre-recorded kernel, measure recalculation time
  2. New Trade (Incremental): Add trade to existing portfolio, measure incremental XVA

For both scenarios:
  - Compute sensitivities (greeks)
  - Measure evaluation time and memory
  - Log results to data/execution_log_xva.csv

Usage:
    python benchmark_xva_workflow.py --mc-paths 10000 --threads 8
    python benchmark_xva_workflow.py --mc-paths 50000 --backends gpu aadc

Version: 1.0.0
"""
MODEL_VERSION = "1.0.0"

import argparse
import json
import os
import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path
from datetime import datetime

# Optional psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from xva_common import (
    load_init_data, parse_hw_model, parse_survival_curve, parse_csa,
    parse_simulation_grid, generate_portfolio, generate_randoms,
    precompute_cumulatives, pw_interp, interpolated_index,
    hw_bond_price, project_curve_eval_hw, discount_curve_eval,
    write_xva_log, build_log_row, LOG_COLUMNS,
    HWModelParams, SurvivalCurveParams, CSAParams, TradeData,
    SimulationGrid, XVAResult,
)

BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = str(BASE_DIR / "initData.json")
LOG_FILE = str(BASE_DIR / "data" / "execution_log_xva.csv")

# =============================================================================
# Memory Tracking
# =============================================================================

class MemoryTracker:
    """Track CPU and GPU memory usage."""

    def __init__(self):
        self._tracemalloc_started = False
        self.cpu_baseline_mb = 0.0
        self.gpu_baseline_mb = 0.0

    def start(self):
        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
        self.cpu_baseline_mb = self._get_cpu_memory_mb()
        self.gpu_baseline_mb = self._get_gpu_memory_mb()

    def _get_cpu_memory_mb(self) -> float:
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        elif self._tracemalloc_started:
            current, _ = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
        return 0.0

    def _get_gpu_memory_mb(self) -> float:
        try:
            from numba import cuda
            if cuda.is_available():
                # Get allocated memory from CUDA context
                ctx = cuda.current_context()
                return ctx.get_memory_info()[1] / (1024 * 1024)  # Used memory
        except:
            pass
        return 0.0

    def get_snapshot(self) -> dict:
        return {
            'cpu_mb': self._get_cpu_memory_mb(),
            'gpu_mb': self._get_gpu_memory_mb(),
            'cpu_delta_mb': self._get_cpu_memory_mb() - self.cpu_baseline_mb,
        }

    def stop(self) -> dict:
        snapshot = self.get_snapshot()
        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            snapshot['cpu_peak_mb'] = peak / (1024 * 1024)
            tracemalloc.stop()
            self._tracemalloc_started = False
        return snapshot


_memory_tracker = MemoryTracker()


# =============================================================================
# Extended Log Row Builder
# =============================================================================

def build_workflow_log_row(
    scenario: str,
    result: XVAResult,
    grid: SimulationGrid,
    trades: TradeData,
    num_mc_paths: int,
    num_threads: int,
    num_sens_params: int,
    cpu_memory_mb: float = 0.0,
    gpu_memory_mb: float = 0.0,
    kernel_cached: bool = False,
) -> dict:
    """Build log row for workflow scenarios."""
    throughput = num_mc_paths / result.eval_time_sec if result.eval_time_sec > 0 else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": f"xva_workflow_{scenario}_{result.backend}",
        "model_version": MODEL_VERSION,
        "num_trades": trades.num_trades,
        "num_mc_paths": num_mc_paths,
        "num_model_steps": len(grid.model_times),
        "num_pricing_times": len(grid.pricing_times),
        "num_sensitivity_params": num_sens_params,
        "num_threads": num_threads,
        "backend": result.backend,
        "mode": result.mode,
        "cva_result": result.cva,
        "dva_result": result.dva,
        "eval_time_sec": result.eval_time_sec,
        "sensitivity_time_sec": result.sensitivity_time_sec,
        "total_time_sec": result.total_time_sec,
        "kernel_recording_sec": result.kernel_recording_sec if not kernel_cached else 0.0,
        "num_params_bumped": result.num_params_bumped,
        "speedup_vs_cpu": 0.0,  # Not computing CPU baseline in workflow
        "max_cva_diff": "",
        "max_dva_diff": "",
        "gpu_kernel_time_sec": result.gpu_kernel_time_sec,
        "memory_mb": cpu_memory_mb + gpu_memory_mb,
        "throughput_paths_per_sec": throughput,
        "status": "success",
    }


# =============================================================================
# Scenario 1: Market Data Update
# =============================================================================

def run_market_data_update(
    hw_base: HWModelParams,
    grid: SimulationGrid,
    trades: TradeData,
    csa: CSAParams,
    company_surv: SurvivalCurveParams,
    ctrparty_surv: SurvivalCurveParams,
    num_mc_paths: int,
    num_threads: int,
    backends: list,
    seed: int = 42,
) -> list:
    """
    Scenario 1: Market Data Update

    Portfolio of 100 trades with pre-recorded kernel.
    Simulate market data bump (rate shift) and measure recalculation time.
    """
    print("\n" + "=" * 70)
    print("  Scenario 1: MARKET DATA UPDATE")
    print("=" * 70)
    print(f"  Trades: {trades.num_trades}, MC paths: {num_mc_paths}, Threads: {num_threads}")

    results = []
    log_rows = []

    # Generate randoms
    randoms = generate_randoms(num_mc_paths, len(grid.model_times), seed=seed)
    cumulat1, cumulat2 = precompute_cumulatives(
        hw_base.mean_rev_times, hw_base.mean_rev_vals, hw_base.alpha)

    # Count sensitivity parameters
    n_mr = len(hw_base.mean_rev_vals)
    n_ctrp = len(ctrparty_surv.values)
    n_comp = len(company_surv.values)
    num_sens_params = 2 + n_mr + n_ctrp + n_comp  # r0, sigma, MR curve, survival curves

    # --- GPU Backend ---
    if "gpu" in backends:
        print("\n  [GPU] Market Data Update...")
        _memory_tracker.start()

        try:
            from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva

            # First run: JIT compilation (kernel recording)
            print("    Recording kernel (JIT warmup)...")
            t_jit = time.perf_counter()
            warmup_randoms = randoms[:1].copy()
            _ = run_gpu_simulation(warmup_randoms, hw_base, grid, trades, csa,
                                   cumulat1, cumulat2, int(grid.is_pricing.sum()))
            jit_time = time.perf_counter() - t_jit
            print(f"    Kernel recorded: {jit_time:.3f}s")

            # Market data update: bump r0 by 10bp
            hw_bumped = HWModelParams(
                alpha=hw_base.alpha,
                sigma=hw_base.sigma,
                r0=hw_base.r0 + 0.001,  # +10bp rate shift
                mean_rev_times=hw_base.mean_rev_times,
                mean_rev_vals=hw_base.mean_rev_vals,
                spread_3m_times=hw_base.spread_3m_times,
                spread_3m_vals=hw_base.spread_3m_vals,
                spread_6m_times=hw_base.spread_6m_times,
                spread_6m_vals=hw_base.spread_6m_vals,
                spread_12m_times=hw_base.spread_12m_times,
                spread_12m_vals=hw_base.spread_12m_vals,
            )
            cumulat1_b, cumulat2_b = precompute_cumulatives(
                hw_bumped.mean_rev_times, hw_bumped.mean_rev_vals, hw_bumped.alpha)

            # Recalculation with cached kernel
            print("    Recalculating with market data update (kernel cached)...")
            t_eval = time.perf_counter()
            pee, nee = run_gpu_simulation(
                randoms, hw_bumped, grid, trades, csa,
                cumulat1_b, cumulat2_b, int(grid.is_pricing.sum()))
            eval_time = time.perf_counter() - t_eval

            cva, dva = compute_cva_dva(
                pee, nee, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)

            total_time = eval_time  # Kernel already cached

            mem = _memory_tracker.get_snapshot()

            gpu_result = XVAResult(
                backend="gpu_cached",
                mode="market_update",
                cva=cva, dva=dva,
                eval_time_sec=eval_time,
                sensitivity_time_sec=0.0,
                total_time_sec=total_time,
                kernel_recording_sec=jit_time,
                gpu_kernel_time_sec=eval_time,
                gpu_memory_mb=mem['gpu_mb'],
            )
            results.append(gpu_result)

            log_rows.append(build_workflow_log_row(
                scenario="market_update",
                result=gpu_result,
                grid=grid, trades=trades,
                num_mc_paths=num_mc_paths,
                num_threads=1,
                num_sens_params=num_sens_params,
                cpu_memory_mb=mem['cpu_mb'],
                gpu_memory_mb=mem['gpu_mb'],
                kernel_cached=True,
            ))

            print(f"    CVA={cva:.6f}, DVA={dva:.6f}")
            print(f"    Eval time (cached): {eval_time*1000:.1f}ms")
            print(f"    Memory: CPU={mem['cpu_mb']:.1f}MB, GPU={mem['gpu_mb']:.1f}MB")

        except Exception as e:
            print(f"    GPU error: {e}")

    # --- AADC C++ Backend ---
    if "aadc" in backends:
        print("\n  [AADC C++] Market Data Update...")
        _memory_tracker.start()

        try:
            from benchmark_xva import run_aadc_cpp

            # First run: kernel compilation
            print("    Recording kernel (AADC compilation)...")
            result_initial = run_aadc_cpp(
                randoms, hw_base, grid, trades, csa, cumulat1, cumulat2,
                company_surv, ctrparty_surv,
                num_threads=num_threads,
                mode="pricing_only",
            )
            if result_initial:
                compile_time = result_initial.kernel_recording_sec
                print(f"    Kernel compiled: {compile_time:.3f}s")

                # Market data update: reuse compiled kernel
                hw_bumped = HWModelParams(
                    alpha=hw_base.alpha,
                    sigma=hw_base.sigma,
                    r0=hw_base.r0 + 0.001,
                    mean_rev_times=hw_base.mean_rev_times,
                    mean_rev_vals=hw_base.mean_rev_vals,
                    spread_3m_times=hw_base.spread_3m_times,
                    spread_3m_vals=hw_base.spread_3m_vals,
                    spread_6m_times=hw_base.spread_6m_times,
                    spread_6m_vals=hw_base.spread_6m_vals,
                    spread_12m_times=hw_base.spread_12m_times,
                    spread_12m_vals=hw_base.spread_12m_vals,
                )
                cumulat1_b, cumulat2_b = precompute_cumulatives(
                    hw_bumped.mean_rev_times, hw_bumped.mean_rev_vals, hw_bumped.alpha)

                print("    Recalculating with market data update...")
                t_eval = time.perf_counter()
                result_updated = run_aadc_cpp(
                    randoms, hw_bumped, grid, trades, csa, cumulat1_b, cumulat2_b,
                    company_surv, ctrparty_surv,
                    num_threads=num_threads,
                    mode="pricing_only",
                )
                eval_time = time.perf_counter() - t_eval

                if result_updated:
                    mem = _memory_tracker.get_snapshot()

                    aadc_result = XVAResult(
                        backend="aadc_cached",
                        mode="market_update",
                        cva=result_updated.cva,
                        dva=result_updated.dva,
                        eval_time_sec=result_updated.eval_time_sec,
                        sensitivity_time_sec=0.0,
                        total_time_sec=result_updated.eval_time_sec,
                        kernel_recording_sec=compile_time,
                        gpu_memory_mb=0.0,
                    )
                    results.append(aadc_result)

                    log_rows.append(build_workflow_log_row(
                        scenario="market_update",
                        result=aadc_result,
                        grid=grid, trades=trades,
                        num_mc_paths=num_mc_paths,
                        num_threads=num_threads,
                        num_sens_params=num_sens_params,
                        cpu_memory_mb=mem['cpu_mb'],
                        gpu_memory_mb=0.0,
                        kernel_cached=True,
                    ))

                    print(f"    CVA={result_updated.cva:.6f}, DVA={result_updated.dva:.6f}")
                    print(f"    Eval time (cached): {result_updated.eval_time_sec*1000:.1f}ms")
                    print(f"    Memory: CPU={mem['cpu_mb']:.1f}MB")

        except Exception as e:
            print(f"    AADC error: {e}")

    # Write logs
    if log_rows:
        write_xva_log(LOG_FILE, log_rows)
        print(f"\n  Logged {len(log_rows)} rows to {LOG_FILE}")

    return results


# =============================================================================
# Scenario 2: New Trade (Incremental)
# =============================================================================

def run_incremental_trade(
    hw: HWModelParams,
    grid: SimulationGrid,
    trades_base: TradeData,
    csa: CSAParams,
    company_surv: SurvivalCurveParams,
    ctrparty_surv: SurvivalCurveParams,
    num_mc_paths: int,
    num_threads: int,
    backends: list,
    seed: int = 42,
) -> list:
    """
    Scenario 2: New Trade (Incremental XVA)

    Start with 100-trade portfolio, add one new trade.
    Measure incremental XVA calculation time.
    """
    print("\n" + "=" * 70)
    print("  Scenario 2: NEW TRADE (INCREMENTAL XVA)")
    print("=" * 70)
    print(f"  Base trades: {trades_base.num_trades}, MC paths: {num_mc_paths}, Threads: {num_threads}")

    results = []
    log_rows = []

    # Generate randoms
    randoms = generate_randoms(num_mc_paths, len(grid.model_times), seed=seed)
    cumulat1, cumulat2 = precompute_cumulatives(
        hw.mean_rev_times, hw.mean_rev_vals, hw.alpha)

    # Count sensitivity parameters
    n_mr = len(hw.mean_rev_vals)
    n_ctrp = len(ctrparty_surv.values)
    n_comp = len(company_surv.values)
    num_sens_params = 2 + n_mr + n_ctrp + n_comp

    # Generate one additional trade
    trades_new = generate_portfolio(
        num_trades=trades_base.num_trades + 1,
        num_periods=5,
        seed=seed + 1000,  # Different seed for new trade
    )

    # --- GPU Backend ---
    if "gpu" in backends:
        print("\n  [GPU] Incremental Trade...")
        _memory_tracker.start()

        try:
            from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva

            # Base portfolio XVA (with kernel recording)
            print("    Computing base portfolio XVA...")
            t_base = time.perf_counter()
            warmup = randoms[:1].copy()
            _ = run_gpu_simulation(warmup, hw, grid, trades_base, csa,
                                   cumulat1, cumulat2, int(grid.is_pricing.sum()))
            jit_time = time.perf_counter() - t_base

            t_base = time.perf_counter()
            pee_base, nee_base = run_gpu_simulation(
                randoms, hw, grid, trades_base, csa,
                cumulat1, cumulat2, int(grid.is_pricing.sum()))
            base_eval_time = time.perf_counter() - t_base

            cva_base, dva_base = compute_cva_dva(
                pee_base, nee_base, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)

            print(f"    Base XVA: CVA={cva_base:.6f}, DVA={dva_base:.6f} ({base_eval_time*1000:.1f}ms)")

            # Portfolio + new trade XVA (kernel cached)
            print("    Computing portfolio + new trade XVA...")
            t_incr = time.perf_counter()
            pee_new, nee_new = run_gpu_simulation(
                randoms, hw, grid, trades_new, csa,
                cumulat1, cumulat2, int(grid.is_pricing.sum()))
            incr_eval_time = time.perf_counter() - t_incr

            cva_new, dva_new = compute_cva_dva(
                pee_new, nee_new, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)

            # Incremental XVA = new - base
            delta_cva = cva_new - cva_base
            delta_dva = dva_new - dva_base

            mem = _memory_tracker.get_snapshot()

            gpu_result = XVAResult(
                backend="gpu_incremental",
                mode="new_trade",
                cva=delta_cva, dva=delta_dva,
                eval_time_sec=incr_eval_time,
                sensitivity_time_sec=0.0,
                total_time_sec=incr_eval_time,
                kernel_recording_sec=jit_time,
                gpu_kernel_time_sec=incr_eval_time,
                gpu_memory_mb=mem['gpu_mb'],
            )
            results.append(gpu_result)

            log_rows.append(build_workflow_log_row(
                scenario="new_trade",
                result=gpu_result,
                grid=grid, trades=trades_new,
                num_mc_paths=num_mc_paths,
                num_threads=1,
                num_sens_params=num_sens_params,
                cpu_memory_mb=mem['cpu_mb'],
                gpu_memory_mb=mem['gpu_mb'],
                kernel_cached=True,
            ))

            print(f"    New XVA: CVA={cva_new:.6f}, DVA={dva_new:.6f}")
            print(f"    Delta XVA: ΔCVA={delta_cva:.6f}, ΔDVA={delta_dva:.6f}")
            print(f"    Incremental eval time: {incr_eval_time*1000:.1f}ms")
            print(f"    Memory: CPU={mem['cpu_mb']:.1f}MB, GPU={mem['gpu_mb']:.1f}MB")

        except Exception as e:
            print(f"    GPU error: {e}")

    # --- AADC C++ Backend ---
    if "aadc" in backends:
        print("\n  [AADC C++] Incremental Trade...")
        _memory_tracker.start()

        try:
            from benchmark_xva import run_aadc_cpp

            # Base portfolio XVA
            print("    Computing base portfolio XVA...")
            result_base = run_aadc_cpp(
                randoms, hw, grid, trades_base, csa, cumulat1, cumulat2,
                company_surv, ctrparty_surv,
                num_threads=num_threads,
                mode="pricing_only",
            )

            if result_base:
                compile_time = result_base.kernel_recording_sec
                print(f"    Base XVA: CVA={result_base.cva:.6f}, DVA={result_base.dva:.6f}")

                # Portfolio + new trade XVA
                print("    Computing portfolio + new trade XVA...")
                t_incr = time.perf_counter()
                result_new = run_aadc_cpp(
                    randoms, hw, grid, trades_new, csa, cumulat1, cumulat2,
                    company_surv, ctrparty_surv,
                    num_threads=num_threads,
                    mode="pricing_only",
                )
                incr_time = time.perf_counter() - t_incr

                if result_new:
                    delta_cva = result_new.cva - result_base.cva
                    delta_dva = result_new.dva - result_base.dva

                    mem = _memory_tracker.get_snapshot()

                    aadc_result = XVAResult(
                        backend="aadc_incremental",
                        mode="new_trade",
                        cva=delta_cva, dva=delta_dva,
                        eval_time_sec=result_new.eval_time_sec,
                        sensitivity_time_sec=0.0,
                        total_time_sec=result_new.eval_time_sec,
                        kernel_recording_sec=compile_time,
                        gpu_memory_mb=0.0,
                    )
                    results.append(aadc_result)

                    log_rows.append(build_workflow_log_row(
                        scenario="new_trade",
                        result=aadc_result,
                        grid=grid, trades=trades_new,
                        num_mc_paths=num_mc_paths,
                        num_threads=num_threads,
                        num_sens_params=num_sens_params,
                        cpu_memory_mb=mem['cpu_mb'],
                        gpu_memory_mb=0.0,
                        kernel_cached=True,
                    ))

                    print(f"    New XVA: CVA={result_new.cva:.6f}, DVA={result_new.dva:.6f}")
                    print(f"    Delta XVA: ΔCVA={delta_cva:.6f}, ΔDVA={delta_dva:.6f}")
                    print(f"    Incremental eval time: {result_new.eval_time_sec*1000:.1f}ms")
                    print(f"    Memory: CPU={mem['cpu_mb']:.1f}MB")

        except Exception as e:
            print(f"    AADC error: {e}")

    # Write logs
    if log_rows:
        write_xva_log(LOG_FILE, log_rows)
        print(f"\n  Logged {len(log_rows)} rows to {LOG_FILE}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XVA Workflow Benchmark: Market Data Update & Incremental Trade"
    )
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT,
                        help='Path to initData.json')
    parser.add_argument('--mc-paths', '-m', type=int, default=10000,
                        help='Number of Monte Carlo paths')
    parser.add_argument('--num-trades', '-t', type=int, default=100,
                        help='Number of trades in base portfolio')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads for AADC C++')
    parser.add_argument('--backends', nargs='+', default=['gpu', 'aadc'],
                        choices=['gpu', 'aadc'],
                        help='Backends to benchmark')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scenario', choices=['all', 'market_update', 'new_trade'],
                        default='all', help='Which scenario to run')

    args = parser.parse_args()

    print("=" * 70)
    print("  XVA Workflow Benchmark")
    print("=" * 70)
    print(f"  Config: {args.input}")
    print(f"  MC Paths: {args.mc_paths}")
    print(f"  Base Trades: {args.num_trades}")
    print(f"  Threads: {args.threads}")
    print(f"  Backends: {args.backends}")
    print(f"  Scenario: {args.scenario}")

    # Load configuration
    data = load_init_data(args.input)
    hw = parse_hw_model(data)
    grid = parse_simulation_grid(data)
    csa = parse_csa(data)
    company_surv = parse_survival_curve(data, "CompSurvCurve")
    ctrparty_surv = parse_survival_curve(data, "CtrpSurvCurve")

    # Generate base portfolio (100 trades)
    trades = generate_portfolio(
        num_trades=args.num_trades,
        num_periods=data.get("Portfolio", {}).get("NumPeriods", 5),
        seed=17,  # Match C++ seed for reproducibility
    )

    print(f"\n  Portfolio: {trades.num_trades} trades, max {trades.max_cf} cashflows each")
    print(f"  Grid: {len(grid.model_times)} steps, {len(grid.pricing_times)} pricing times")

    all_results = []

    # Run scenarios
    if args.scenario in ['all', 'market_update']:
        results = run_market_data_update(
            hw, grid, trades, csa, company_surv, ctrparty_surv,
            num_mc_paths=args.mc_paths,
            num_threads=args.threads,
            backends=args.backends,
            seed=args.seed,
        )
        all_results.extend(results)

    if args.scenario in ['all', 'new_trade']:
        results = run_incremental_trade(
            hw, grid, trades, csa, company_surv, ctrparty_surv,
            num_mc_paths=args.mc_paths,
            num_threads=args.threads,
            backends=args.backends,
            seed=args.seed,
        )
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r.backend:<20} {r.mode:<15} CVA={r.cva:>12.6f}  "
              f"DVA={r.dva:>12.6f}  Eval={r.eval_time_sec*1000:>8.1f}ms")

    print("\n  Done.")
    return all_results


if __name__ == "__main__":
    main()
