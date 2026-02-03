#!/usr/bin/env python3
"""XVA Benchmark with Pre-Recorded Kernels: Market Data Update & Incremental Trade.

Alternative benchmark design using 100 trades with pre-generated kernels:

  Scenario 1: Market Data Update
    - Pre-record kernel with 100-trade portfolio
    - Market data update arrives (rate shift)
    - Recalculate XVA + sensitivities using cached kernel
    - Measure: eval_time, sensitivity_time, memory

  Scenario 2: New Trade (Incremental XVA)
    - Pre-record kernel with 100-trade portfolio
    - New trade arrives (pre-recorded kernel for 101 trades)
    - Compute incremental XVA + sensitivities
    - Measure: eval_time, sensitivity_time, memory

Backends:
  - pathwise_gpu: Single-pass GPU with pathwise AD (fastest)
  - gpu_brute_force: GPU with bump-and-revalue
  - cpp_aadc: C++ AADC with kernel recording

Results logged to: data/execution_log_xva.csv

Usage:
    python benchmark_xva_prerecorded.py --mc-paths 51200 --num-trades 100
    python benchmark_xva_prerecorded.py --backends pathwise gpu --scenario all
    python benchmark_xva_prerecorded.py --scenario market_update --backends pathwise

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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Optional psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from xva_common import (
    load_init_data, parse_hw_model, parse_survival_curve, parse_csa,
    parse_simulation_grid, generate_portfolio, generate_randoms,
    precompute_cumulatives, pw_interp,
    write_xva_log, LOG_COLUMNS,
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
    """Track CPU and GPU memory usage throughout benchmark."""

    def __init__(self):
        self._tracemalloc_started = False
        self.cpu_baseline_mb = 0.0
        self.gpu_baseline_mb = 0.0
        self._peak_cpu_mb = 0.0
        self._peak_gpu_mb = 0.0

    def start(self):
        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
        self.cpu_baseline_mb = self._get_cpu_memory_mb()
        self.gpu_baseline_mb = self._get_gpu_memory_mb()
        self._peak_cpu_mb = self.cpu_baseline_mb
        self._peak_gpu_mb = self.gpu_baseline_mb

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
                ctx = cuda.current_context()
                mem_info = ctx.get_memory_info()
                return (mem_info.total - mem_info.free) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def update_peak(self):
        cpu_now = self._get_cpu_memory_mb()
        gpu_now = self._get_gpu_memory_mb()
        self._peak_cpu_mb = max(self._peak_cpu_mb, cpu_now)
        self._peak_gpu_mb = max(self._peak_gpu_mb, gpu_now)

    def get_snapshot(self) -> dict:
        self.update_peak()
        return {
            'cpu_mb': self._get_cpu_memory_mb(),
            'gpu_mb': self._get_gpu_memory_mb(),
            'cpu_delta_mb': self._get_cpu_memory_mb() - self.cpu_baseline_mb,
            'gpu_delta_mb': self._get_gpu_memory_mb() - self.gpu_baseline_mb,
            'peak_cpu_mb': self._peak_cpu_mb,
            'peak_gpu_mb': self._peak_gpu_mb,
        }

    def stop(self) -> dict:
        snapshot = self.get_snapshot()
        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            snapshot['tracemalloc_peak_mb'] = peak / (1024 * 1024)
            tracemalloc.stop()
            self._tracemalloc_started = False
        return snapshot


# =============================================================================
# Pre-Recorded Kernel Context
# =============================================================================

@dataclass
class PrerecordedKernel:
    """Stores pre-recorded/compiled kernel state for reuse."""
    backend: str
    kernel_recording_sec: float
    jit_compiled: bool = False
    aadc_compiled: bool = False
    gpu_context: Optional[object] = None  # For GPU streamed context


def record_gpu_kernel(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                      num_pricing: int) -> Tuple[PrerecordedKernel, float]:
    """Pre-record GPU kernel via JIT warmup."""
    from xva_gpu_kernel import run_gpu_simulation

    t_start = time.perf_counter()
    warmup_randoms = randoms[:1].copy()
    _ = run_gpu_simulation(warmup_randoms, hw, grid, trades, csa,
                           cumulat1, cumulat2, num_pricing)
    jit_time = time.perf_counter() - t_start

    return PrerecordedKernel(
        backend="gpu_brute_force",
        kernel_recording_sec=jit_time,
        jit_compiled=True,
    ), jit_time


def record_pathwise_kernel(randoms, hw, grid, trades, csa, cumulat1, cumulat2,
                           num_pricing: int) -> Tuple[PrerecordedKernel, float]:
    """Pre-record pathwise GPU kernel via JIT warmup."""
    from xva_pathwise_gpu import _run_pathwise_kernel_once

    t_start = time.perf_counter()
    _run_pathwise_kernel_once(randoms[:1], hw, grid, trades, csa,
                              cumulat1, cumulat2, num_pricing, block_size=256)
    jit_time = time.perf_counter() - t_start

    return PrerecordedKernel(
        backend="pathwise_gpu",
        kernel_recording_sec=jit_time,
        jit_compiled=True,
    ), jit_time


# =============================================================================
# Log Row Builder
# =============================================================================

def build_prerecorded_log_row(
    scenario: str,
    result: XVAResult,
    grid: SimulationGrid,
    trades: TradeData,
    num_mc_paths: int,
    num_threads: int,
    num_sens_params: int,
    memory_mb: float = 0.0,
    kernel_cached: bool = True,
    base_cva: float = 0.0,
    base_dva: float = 0.0,
) -> dict:
    """Build log row for prerecorded kernel scenarios."""
    throughput = num_mc_paths / result.eval_time_sec if result.eval_time_sec > 0 else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": f"xva_prerecorded_{scenario}_{result.backend}",
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
        "kernel_recording_sec": 0.0 if kernel_cached else result.kernel_recording_sec,
        "num_params_bumped": result.num_params_bumped,
        "speedup_vs_cpu": 0.0,  # Computed later if CPU baseline available
        "max_cva_diff": abs(result.cva - base_cva) if base_cva != 0 else "",
        "max_dva_diff": abs(result.dva - base_dva) if base_dva != 0 else "",
        "gpu_kernel_time_sec": result.gpu_kernel_time_sec,
        "memory_mb": memory_mb,
        "throughput_paths_per_sec": throughput,
        "status": "success",
    }


# =============================================================================
# Scenario 1: Market Data Update
# =============================================================================

def run_market_data_update_scenario(
    hw_base: HWModelParams,
    grid: SimulationGrid,
    trades: TradeData,
    csa: CSAParams,
    company_surv: SurvivalCurveParams,
    ctrparty_surv: SurvivalCurveParams,
    num_mc_paths: int,
    num_threads: int,
    backends: List[str],
    input_file: str,
    rate_bump_bp: float = 10.0,
    seed: int = 42,
    skip_mr_bumps: bool = True,
) -> List[dict]:
    """
    Scenario 1: Market Data Update with Pre-Recorded Kernel

    Steps:
    1. Pre-record kernel with base market data (100 trades)
    2. Market data update arrives (rate shift of +10bp)
    3. Recalculate XVA + sensitivities using cached kernel
    4. Measure and log: eval_time, sensitivity_time, memory
    """
    print("\n" + "=" * 75)
    print("  SCENARIO 1: MARKET DATA UPDATE (Pre-Recorded Kernel)")
    print("=" * 75)
    print(f"  Portfolio: {trades.num_trades} trades")
    print(f"  MC Paths: {num_mc_paths}")
    print(f"  Rate Bump: +{rate_bump_bp}bp")
    print(f"  Backends: {backends}")

    log_rows = []
    memory_tracker = MemoryTracker()

    # Generate randoms (shared across backends for consistency)
    randoms = generate_randoms(num_mc_paths, len(grid.model_times), seed=seed, fast=True)
    cumulat1, cumulat2 = precompute_cumulatives(
        hw_base.mean_rev_times, hw_base.mean_rev_vals, hw_base.alpha)

    # Count sensitivity parameters
    n_mr = len(hw_base.mean_rev_vals)
    n_ctrp = len(ctrparty_surv.values)
    n_comp = len(company_surv.values)
    num_sens_params = 2 + n_mr + n_ctrp + n_comp  # r0, sigma, MR curve, survival curves

    # Create bumped market data
    rate_bump = rate_bump_bp / 10000.0  # Convert bp to decimal
    hw_bumped = HWModelParams(
        alpha=hw_base.alpha,
        sigma=hw_base.sigma,
        r0=hw_base.r0 + rate_bump,
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

    num_pricing = int(grid.is_pricing.sum())

    # --- Pathwise GPU Backend ---
    if "pathwise" in backends:
        print("\n  [Pathwise GPU] Market Data Update with Sensitivities...")
        memory_tracker.start()

        try:
            from xva_pathwise_gpu import run_pathwise_gpu

            # Step 1: Pre-record kernel (JIT warmup)
            print("    Step 1: Pre-recording kernel (JIT)...")
            kernel, jit_time = record_pathwise_kernel(
                randoms, hw_base, grid, trades, csa, cumulat1, cumulat2, num_pricing)
            print(f"    Kernel recorded: {jit_time:.3f}s")

            # Step 2: Run with bumped market data (kernel cached)
            print("    Step 2: Recalculating with market data update...")
            t_start = time.perf_counter()
            result = run_pathwise_gpu(
                randoms, hw_bumped, grid, trades, csa, cumulat1_b, cumulat2_b,
                company_surv, ctrparty_surv, mode="pricing_with_greeks")
            recalc_time = time.perf_counter() - t_start

            if result:
                mem = memory_tracker.get_snapshot()
                result.kernel_recording_sec = jit_time

                log_rows.append(build_prerecorded_log_row(
                    scenario="market_update",
                    result=result,
                    grid=grid, trades=trades,
                    num_mc_paths=num_mc_paths,
                    num_threads=1,
                    num_sens_params=result.num_params_bumped,
                    memory_mb=mem['gpu_mb'] + mem['cpu_mb'],
                    kernel_cached=True,
                ))

                print(f"    CVA={result.cva:.6f}, DVA={result.dva:.6f}")
                print(f"    Eval time: {result.eval_time_sec*1000:.1f}ms")
                print(f"    Sensitivity time: {result.sensitivity_time_sec*1000:.1f}ms (included in eval)")
                print(f"    Total recalc: {recalc_time*1000:.1f}ms")
                print(f"    Memory: GPU={mem['gpu_mb']:.1f}MB, CPU={mem['cpu_mb']:.1f}MB")
                print(f"    Sensitivities: {result.num_params_bumped} params computed")

        except Exception as e:
            print(f"    Pathwise GPU error: {e}")
            import traceback
            traceback.print_exc()

    # --- GPU Brute-Force Backend ---
    if "gpu" in backends:
        print("\n  [GPU Brute-Force] Market Data Update with Sensitivities...")
        memory_tracker.start()

        try:
            from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva
            from benchmark_xva import gpu_bump_and_revalue

            # Step 1: Pre-record kernel
            print("    Step 1: Pre-recording kernel (JIT)...")
            kernel, jit_time = record_gpu_kernel(
                randoms, hw_base, grid, trades, csa, cumulat1, cumulat2, num_pricing)
            print(f"    Kernel recorded: {jit_time:.3f}s")

            # Step 2: Run primal with bumped market data
            print("    Step 2: Recalculating primal with market data update...")
            t_primal = time.perf_counter()
            pee, nee = run_gpu_simulation(
                randoms, hw_bumped, grid, trades, csa,
                cumulat1_b, cumulat2_b, num_pricing)
            primal_time = time.perf_counter() - t_primal

            cva, dva = compute_cva_dva(
                pee, nee, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)

            print(f"    Primal done: CVA={cva:.6f}, DVA={dva:.6f} ({primal_time*1000:.1f}ms)")

            # Step 3: Compute sensitivities via bump-and-revalue
            print("    Step 3: Computing sensitivities (bump-and-revalue)...")
            sens_time, num_bumped = gpu_bump_and_revalue(
                randoms, hw_bumped, grid, trades, csa, cumulat1_b, cumulat2_b,
                company_surv, ctrparty_surv, cva, dva, num_pricing,
                base_pee=pee, base_nee=nee, skip_mr_bumps=skip_mr_bumps)

            total_time = primal_time + sens_time
            mem = memory_tracker.get_snapshot()

            result = XVAResult(
                backend="gpu_brute_force",
                mode="pricing_with_greeks",
                cva=cva, dva=dva,
                eval_time_sec=primal_time,
                sensitivity_time_sec=sens_time,
                total_time_sec=total_time,
                kernel_recording_sec=jit_time,
                gpu_kernel_time_sec=primal_time,
                num_params_bumped=num_bumped,
                gpu_memory_mb=mem['gpu_mb'],
            )

            log_rows.append(build_prerecorded_log_row(
                scenario="market_update",
                result=result,
                grid=grid, trades=trades,
                num_mc_paths=num_mc_paths,
                num_threads=1,
                num_sens_params=num_bumped,
                memory_mb=mem['gpu_mb'] + mem['cpu_mb'],
                kernel_cached=True,
            ))

            print(f"    Total time: {total_time*1000:.1f}ms")
            print(f"    Memory: GPU={mem['gpu_mb']:.1f}MB, CPU={mem['cpu_mb']:.1f}MB")
            print(f"    Sensitivities: {num_bumped} params bumped")

        except Exception as e:
            print(f"    GPU Brute-Force error: {e}")
            import traceback
            traceback.print_exc()

    # --- AADC C++ Backend (Production Mode with Kernel Reuse) ---
    if "aadc" in backends:
        print("\n  [AADC C++] Market Data Update with Sensitivities (Production Mode)...")
        memory_tracker.start()

        try:
            from benchmark_xva import run_aadc_cpp_production

            # Run production mode: cold start + warm run (kernel reused)
            print("    Running AADC production mode (cold + warm with kernel reuse)...")
            cold_result, warm_result = run_aadc_cpp_production(
                input_file, num_mc_paths, num_threads, num_trades=trades.num_trades)

            if cold_result and warm_result:
                mem = memory_tracker.get_snapshot()

                # Log cold start (full_portfolio)
                log_rows.append(build_prerecorded_log_row(
                    scenario="full_portfolio",
                    result=cold_result,
                    grid=grid, trades=trades,
                    num_mc_paths=num_mc_paths,
                    num_threads=num_threads,
                    num_sens_params=cold_result.num_params_bumped,
                    memory_mb=mem['cpu_mb'],
                    kernel_cached=False,
                ))

                # Log warm run (market_update with kernel reused)
                log_rows.append(build_prerecorded_log_row(
                    scenario="market_update",
                    result=warm_result,
                    grid=grid, trades=trades,
                    num_mc_paths=num_mc_paths,
                    num_threads=num_threads,
                    num_sens_params=warm_result.num_params_bumped,
                    memory_mb=mem['cpu_mb'],
                    kernel_cached=True,  # Kernel was reused!
                ))

                print(f"    Cold start: CVA={cold_result.cva:.6f}, compile={cold_result.kernel_recording_sec:.2f}s, total={cold_result.total_time_sec:.2f}s")
                print(f"    Warm run:   CVA={warm_result.cva:.6f}, compile={warm_result.kernel_recording_sec:.2f}s (REUSED), total={warm_result.total_time_sec:.2f}s")
                print(f"    Sensitivities: {warm_result.num_params_bumped} params computed in both runs")

        except Exception as e:
            print(f"    AADC error: {e}")
            import traceback
            traceback.print_exc()

    return log_rows


# =============================================================================
# Scenario 2: New Trade (Incremental XVA)
# =============================================================================

def run_new_trade_scenario(
    hw: HWModelParams,
    grid: SimulationGrid,
    trades_base: TradeData,
    csa: CSAParams,
    company_surv: SurvivalCurveParams,
    ctrparty_surv: SurvivalCurveParams,
    num_mc_paths: int,
    num_threads: int,
    backends: List[str],
    input_file: str,
    t0: int = 0,
    num_periods: int = 5,
    seed: int = 42,
    skip_mr_bumps: bool = True,
) -> List[dict]:
    """
    Scenario 2: New Trade with Pre-Recorded Kernel (Incremental XVA)

    Steps:
    1. Pre-record kernel with base portfolio (100 trades)
    2. Compute base XVA + sensitivities
    3. New trade arrives → generate 101-trade portfolio
    4. Compute new XVA + sensitivities using cached kernel
    5. Calculate incremental XVA = new - base
    6. Measure and log: eval_time, sensitivity_time, memory
    """
    print("\n" + "=" * 75)
    print("  SCENARIO 2: NEW TRADE (Incremental XVA with Pre-Recorded Kernel)")
    print("=" * 75)
    print(f"  Base Portfolio: {trades_base.num_trades} trades")
    print(f"  New Portfolio: {trades_base.num_trades + 1} trades")
    print(f"  MC Paths: {num_mc_paths}")
    print(f"  Backends: {backends}")

    log_rows = []
    memory_tracker = MemoryTracker()

    # Generate randoms
    randoms = generate_randoms(num_mc_paths, len(grid.model_times), seed=seed, fast=True)
    cumulat1, cumulat2 = precompute_cumulatives(
        hw.mean_rev_times, hw.mean_rev_vals, hw.alpha)

    # Count sensitivity parameters
    n_mr = len(hw.mean_rev_vals)
    n_ctrp = len(ctrparty_surv.values)
    n_comp = len(company_surv.values)
    num_sens_params = 2 + n_mr + n_ctrp + n_comp

    num_pricing = int(grid.is_pricing.sum())

    # Generate portfolio with one additional trade
    trades_new = generate_portfolio(
        t0=t0,
        num_trades=trades_base.num_trades + 1,
        num_periods=num_periods,
        seed=seed + 1000,  # Different seed for variation
    )

    # --- Pathwise GPU Backend ---
    if "pathwise" in backends:
        print("\n  [Pathwise GPU] Incremental XVA with Sensitivities...")
        memory_tracker.start()

        try:
            from xva_pathwise_gpu import run_pathwise_gpu

            # Step 1: Pre-record kernel
            print("    Step 1: Pre-recording kernel (JIT)...")
            kernel, jit_time = record_pathwise_kernel(
                randoms, hw, grid, trades_base, csa, cumulat1, cumulat2, num_pricing)
            print(f"    Kernel recorded: {jit_time:.3f}s")

            # Step 2: Compute base portfolio XVA + sensitivities
            print("    Step 2: Computing base portfolio XVA + sensitivities...")
            t_base = time.perf_counter()
            result_base = run_pathwise_gpu(
                randoms, hw, grid, trades_base, csa, cumulat1, cumulat2,
                company_surv, ctrparty_surv, mode="pricing_with_greeks")
            base_time = time.perf_counter() - t_base

            if result_base:
                print(f"    Base XVA: CVA={result_base.cva:.6f}, DVA={result_base.dva:.6f} ({base_time*1000:.1f}ms)")

                # Step 3: Compute new portfolio XVA + sensitivities (kernel cached)
                print("    Step 3: Computing new portfolio XVA + sensitivities...")
                t_new = time.perf_counter()
                result_new = run_pathwise_gpu(
                    randoms, hw, grid, trades_new, csa, cumulat1, cumulat2,
                    company_surv, ctrparty_surv, mode="pricing_with_greeks")
                new_time = time.perf_counter() - t_new

                if result_new:
                    # Calculate incremental XVA
                    delta_cva = result_new.cva - result_base.cva
                    delta_dva = result_new.dva - result_base.dva

                    mem = memory_tracker.get_snapshot()

                    # Log incremental result
                    incr_result = XVAResult(
                        backend="pathwise_gpu",
                        mode="incremental_with_greeks",
                        cva=delta_cva,
                        dva=delta_dva,
                        eval_time_sec=result_new.eval_time_sec,
                        sensitivity_time_sec=result_new.sensitivity_time_sec,
                        total_time_sec=new_time,
                        kernel_recording_sec=jit_time,
                        gpu_kernel_time_sec=result_new.gpu_kernel_time_sec,
                        num_params_bumped=result_new.num_params_bumped,
                        gpu_memory_mb=mem['gpu_mb'],
                    )

                    log_rows.append(build_prerecorded_log_row(
                        scenario="new_trade",
                        result=incr_result,
                        grid=grid, trades=trades_new,
                        num_mc_paths=num_mc_paths,
                        num_threads=1,
                        num_sens_params=result_new.num_params_bumped,
                        memory_mb=mem['gpu_mb'] + mem['cpu_mb'],
                        kernel_cached=True,
                        base_cva=result_base.cva,
                        base_dva=result_base.dva,
                    ))

                    print(f"    New XVA: CVA={result_new.cva:.6f}, DVA={result_new.dva:.6f} ({new_time*1000:.1f}ms)")
                    print(f"    Incremental XVA: ΔCVA={delta_cva:.6f}, ΔDVA={delta_dva:.6f}")
                    print(f"    Memory: GPU={mem['gpu_mb']:.1f}MB, CPU={mem['cpu_mb']:.1f}MB")
                    print(f"    Sensitivities: {result_new.num_params_bumped} params computed")

        except Exception as e:
            print(f"    Pathwise GPU error: {e}")
            import traceback
            traceback.print_exc()

    # --- GPU Brute-Force Backend ---
    if "gpu" in backends:
        print("\n  [GPU Brute-Force] Incremental XVA with Sensitivities...")
        memory_tracker.start()

        try:
            from xva_gpu_kernel import run_gpu_simulation, compute_cva_dva
            from benchmark_xva import gpu_bump_and_revalue

            # Step 1: Pre-record kernel
            print("    Step 1: Pre-recording kernel (JIT)...")
            kernel, jit_time = record_gpu_kernel(
                randoms, hw, grid, trades_base, csa, cumulat1, cumulat2, num_pricing)
            print(f"    Kernel recorded: {jit_time:.3f}s")

            # Step 2: Compute base portfolio
            print("    Step 2: Computing base portfolio XVA...")
            t_base = time.perf_counter()
            pee_base, nee_base = run_gpu_simulation(
                randoms, hw, grid, trades_base, csa, cumulat1, cumulat2, num_pricing)
            cva_base, dva_base = compute_cva_dva(
                pee_base, nee_base, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)
            base_time = time.perf_counter() - t_base
            print(f"    Base XVA: CVA={cva_base:.6f}, DVA={dva_base:.6f} ({base_time*1000:.1f}ms)")

            # Step 3: Compute new portfolio primal
            print("    Step 3: Computing new portfolio XVA...")
            t_new = time.perf_counter()
            pee_new, nee_new = run_gpu_simulation(
                randoms, hw, grid, trades_new, csa, cumulat1, cumulat2, num_pricing)
            cva_new, dva_new = compute_cva_dva(
                pee_new, nee_new, grid.pricing_times,
                company_surv.times_years, company_surv.values,
                ctrparty_surv.times_years, ctrparty_surv.values,
                company_surv.t0, ctrparty_surv.t0)
            primal_time = time.perf_counter() - t_new

            # Step 4: Compute sensitivities for new portfolio
            print("    Step 4: Computing sensitivities for new portfolio...")
            sens_time, num_bumped = gpu_bump_and_revalue(
                randoms, hw, grid, trades_new, csa, cumulat1, cumulat2,
                company_surv, ctrparty_surv, cva_new, dva_new, num_pricing,
                base_pee=pee_new, base_nee=nee_new, skip_mr_bumps=skip_mr_bumps)

            # Calculate incremental XVA
            delta_cva = cva_new - cva_base
            delta_dva = dva_new - dva_base

            total_time = primal_time + sens_time
            mem = memory_tracker.get_snapshot()

            result = XVAResult(
                backend="gpu_brute_force",
                mode="incremental_with_greeks",
                cva=delta_cva,
                dva=delta_dva,
                eval_time_sec=primal_time,
                sensitivity_time_sec=sens_time,
                total_time_sec=total_time,
                kernel_recording_sec=jit_time,
                gpu_kernel_time_sec=primal_time,
                num_params_bumped=num_bumped,
                gpu_memory_mb=mem['gpu_mb'],
            )

            log_rows.append(build_prerecorded_log_row(
                scenario="new_trade",
                result=result,
                grid=grid, trades=trades_new,
                num_mc_paths=num_mc_paths,
                num_threads=1,
                num_sens_params=num_bumped,
                memory_mb=mem['gpu_mb'] + mem['cpu_mb'],
                kernel_cached=True,
                base_cva=cva_base,
                base_dva=dva_base,
            ))

            print(f"    New XVA: CVA={cva_new:.6f}, DVA={dva_new:.6f}")
            print(f"    Incremental XVA: ΔCVA={delta_cva:.6f}, ΔDVA={delta_dva:.6f}")
            print(f"    Total time: {total_time*1000:.1f}ms")
            print(f"    Memory: GPU={mem['gpu_mb']:.1f}MB")
            print(f"    Sensitivities: {num_bumped} params bumped")

        except Exception as e:
            print(f"    GPU Brute-Force error: {e}")
            import traceback
            traceback.print_exc()

    # --- AADC C++ Backend ---
    if "aadc" in backends:
        print("\n  [AADC C++] New Trade Scenario...")
        print("    NOTE: AADC monolithic kernel includes all trades.")
        print("    Adding a new trade requires FULL kernel recompilation.")
        print("    (No kernel reuse possible for new trades with AADC)")
        memory_tracker.start()

        try:
            from benchmark_xva import run_aadc_cpp_production

            # Run production mode to show what a new trade would cost
            # This demonstrates the recompilation overhead
            print("    Running AADC production mode (shows recompilation cost)...")
            cold_result, warm_result = run_aadc_cpp_production(
                input_file, num_mc_paths, num_threads, num_trades=trades_base.num_trades)

            if cold_result:
                mem = memory_tracker.get_snapshot()

                # For new trade, we report the COLD result (full recompilation required)
                log_rows.append(build_prerecorded_log_row(
                    scenario="new_trade",
                    result=cold_result,
                    grid=grid, trades=trades_base,
                    num_mc_paths=num_mc_paths,
                    num_threads=num_threads,
                    num_sens_params=cold_result.num_params_bumped,
                    memory_mb=mem['cpu_mb'],
                    kernel_cached=False,  # Kernel NOT reused for new trade
                ))

                print(f"    New trade requires: compile={cold_result.kernel_recording_sec:.2f}s + exec={cold_result.eval_time_sec:.2f}s = {cold_result.total_time_sec:.2f}s")
                print(f"    CVA={cold_result.cva:.6f}, Sensitivities={cold_result.num_params_bumped} params")

        except Exception as e:
            print(f"    AADC error: {e}")

    return log_rows


# =============================================================================
# Summary and Results Display
# =============================================================================

def print_summary(all_log_rows: List[dict]):
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 75)
    print("  BENCHMARK SUMMARY")
    print("=" * 75)

    if not all_log_rows:
        print("  No results to display.")
        return

    # Group by scenario
    market_update = [r for r in all_log_rows if "market_update" in r["model_name"]]
    new_trade = [r for r in all_log_rows if "new_trade" in r["model_name"]]

    def _print_scenario(rows, title):
        if not rows:
            return
        print(f"\n  {title}:")
        print(f"  {'Backend':<20} {'Eval(ms)':>10} {'Sens(ms)':>10} {'Total(ms)':>12} {'Mem(MB)':>10}")
        print("  " + "-" * 62)
        for r in rows:
            backend = r["backend"]
            eval_ms = r["eval_time_sec"] * 1000
            sens_ms = r["sensitivity_time_sec"] * 1000
            total_ms = r["total_time_sec"] * 1000
            mem_mb = r["memory_mb"]
            print(f"  {backend:<20} {eval_ms:>10.1f} {sens_ms:>10.1f} {total_ms:>12.1f} {mem_mb:>10.1f}")

    _print_scenario(market_update, "Scenario 1: Market Data Update")
    _print_scenario(new_trade, "Scenario 2: New Trade (Incremental)")

    print("\n" + "=" * 75)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XVA Benchmark with Pre-Recorded Kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_xva_prerecorded.py --mc-paths 51200 --num-trades 100
  python benchmark_xva_prerecorded.py --backends pathwise gpu --scenario all
  python benchmark_xva_prerecorded.py --scenario market_update --backends pathwise
        """
    )
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT,
                        help='Path to initData.json')
    parser.add_argument('--mc-paths', '-m', type=int, default=10000,
                        help='Number of Monte Carlo paths (default: 51200)')
    parser.add_argument('--num-trades', '-t', type=int, default=100,
                        help='Number of trades in base portfolio (default: 100)')
    parser.add_argument('--threads', type=int, default=16,
                        help='Number of threads for AADC C++ (default: 16)')
    parser.add_argument('--backends', nargs='+', default=['pathwise', 'gpu'],
                        choices=['pathwise', 'gpu', 'aadc'],
                        help='Backends to benchmark (default: pathwise gpu)')
    parser.add_argument('--scenario', choices=['all', 'market_update', 'new_trade'],
                        default='all', help='Which scenario to run (default: all)')
    parser.add_argument('--rate-bump', type=float, default=10.0,
                        help='Rate bump in basis points for market update (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-log', action='store_true',
                        help='Skip CSV logging')
    parser.add_argument('--include-mr-bumps', action='store_true',
                        help='Include mean reversion curve bumps for GPU (adds ~250 params)')

    args = parser.parse_args()

    print("=" * 75)
    print("  XVA Benchmark with Pre-Recorded Kernels")
    print("=" * 75)
    print(f"  Config: {args.input}")
    print(f"  MC Paths: {args.mc_paths}")
    print(f"  Base Trades: {args.num_trades}")
    print(f"  Threads: {args.threads}")
    print(f"  Backends: {args.backends}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Rate Bump: {args.rate_bump}bp")

    # Load configuration
    data = load_init_data(args.input)
    hw = parse_hw_model(data)
    t0_days = data.get("t0", 0)
    grid = parse_simulation_grid(data["ModelAndPricingTimes"], t0_days)
    csa = parse_csa(data["csa"])
    company_surv = parse_survival_curve(data["CompanySurvivalCurve"], t0_days)
    ctrparty_surv = parse_survival_curve(data["CounterPartySurvivalCurve"], t0_days)

    # Generate base portfolio
    num_periods = data.get("Portfolio", {}).get("NumPeriods", 5)
    trades = generate_portfolio(
        t0=t0_days,
        num_trades=args.num_trades,
        num_periods=num_periods,
        seed=17,  # Match C++ seed for reproducibility
    )

    print(f"\n  Portfolio: {trades.num_trades} trades, max {trades.max_cf} cashflows each")
    print(f"  Grid: {len(grid.model_times)} steps, {len(grid.pricing_times)} pricing times")

    all_log_rows = []

    # Run scenarios
    if args.scenario in ['all', 'market_update']:
        rows = run_market_data_update_scenario(
            hw, grid, trades, csa, company_surv, ctrparty_surv,
            num_mc_paths=args.mc_paths,
            num_threads=args.threads,
            backends=args.backends,
            input_file=args.input,
            rate_bump_bp=args.rate_bump,
            seed=args.seed,
            skip_mr_bumps=not args.include_mr_bumps,
        )
        all_log_rows.extend(rows)

    if args.scenario in ['all', 'new_trade']:
        rows = run_new_trade_scenario(
            hw, grid, trades, csa, company_surv, ctrparty_surv,
            num_mc_paths=args.mc_paths,
            num_threads=args.threads,
            backends=args.backends,
            input_file=args.input,
            t0=t0_days,
            num_periods=num_periods,
            seed=args.seed,
            skip_mr_bumps=not args.include_mr_bumps,
        )
        all_log_rows.extend(rows)

    # Print summary
    print_summary(all_log_rows)

    # Write logs
    if not args.no_log and all_log_rows:
        write_xva_log(LOG_FILE, all_log_rows)
        print(f"\n  Results logged to {LOG_FILE}")

    print("\n  Done.")
    return all_log_rows


if __name__ == "__main__":
    main()
