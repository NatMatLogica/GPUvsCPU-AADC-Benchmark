#!/usr/bin/env python3
"""XVA AADC Modular Kernel: SIMM-style architecture for kernel reuse.

Architecture (mirrors GPU_AAD SIMM approach):
  1. Trade Valuation: Computed outside AADC (Python/NumPy)
  2. Aggregation: V_portfolio(t) = sum of V_trade(t)
  3. CSA+CVA Kernel: AADC kernel takes V_portfolio(t) as input

This allows:
  - Kernel size O(pricing_times), NOT O(trades)
  - Adding new trades: just update V_portfolio, reuse kernel
  - Market data update: recompute trade values, reuse kernel

Usage:
    python xva_aadc_modular.py --num-trades 100 --mc-paths 1000 --threads 8
    python xva_aadc_modular.py --scenario new_trade --num-trades 100 --threads 16

Version: 1.0.0
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Try to import AADC
try:
    import aadc
    AADC_AVAILABLE = True
except ImportError:
    AADC_AVAILABLE = False
    print("Warning: AADC not available, using NumPy fallback")

from xva_common import write_xva_log

BASE_DIR = Path(__file__).parent
LOG_FILE = str(BASE_DIR / "data" / "execution_log_xva.csv")


# =============================================================================
# CSA + CVA AADC Kernel (takes aggregated portfolio values as input)
# =============================================================================

class AADCCSACVAKernel:
    """AADC kernel for CSA + CVA computation.

    Takes V_portfolio(t) as input, computes:
      - Collateral evolution (path-dependent)
      - PEE/NEE (positive/negative expected exposure)
      - CVA/DVA via trapezoidal integration

    Kernel size: O(num_pricing_times), independent of trade count.
    """

    def __init__(self, num_pricing_times: int, csa_params: dict,
                 ctrparty_surv: np.ndarray, company_surv: np.ndarray,
                 num_threads: int = 4):
        self.num_pricing_times = num_pricing_times
        self.csa_params = csa_params
        self.ctrparty_surv = ctrparty_surv
        self.company_surv = company_surv
        self.num_threads = num_threads
        self.kernel_recorded = False
        self.funcs = None
        self.v_handles = []
        self.cva_output = None
        self.dva_output = None
        self.recording_time = 0.0

    def record_kernel(self):
        """Record AADC kernel: V_portfolio(t) -> CVA, DVA."""
        if not AADC_AVAILABLE:
            self.kernel_recorded = True
            return

        start = time.perf_counter()

        # Smooth max/min approximations for AADC (differentiable everywhere)
        # smooth_max(x, 0) ≈ (x + sqrt(x^2 + eps)) / 2
        _EPS = 1e-6

        def smooth_max_zero(x):
            """Smooth approximation of max(x, 0)"""
            return (x + (x * x + _EPS).sqrt()) * 0.5

        def smooth_min_zero(x):
            """Smooth approximation of min(x, 0)"""
            return (x - (x * x + _EPS).sqrt()) * 0.5

        with aadc.record_kernel() as funcs:
            # Mark portfolio values at each pricing time as inputs
            v_portfolio = []
            self.v_handles = []
            for t in range(self.num_pricing_times):
                v_t = aadc.idouble(0.0)
                handle = v_t.mark_as_input()
                self.v_handles.append(handle)
                v_portfolio.append(v_t)

            # CSA parameters
            mta_h = self.csa_params['mta_h']
            mta_l = self.csa_params['mta_l']
            th = self.csa_params['th']
            tl = self.csa_params['tl']

            # Collateral evolution (sequential across time)
            collateral = aadc.idouble(0.0)
            pee_list = []
            nee_list = []

            for t in range(self.num_pricing_times):
                v = v_portfolio[t]

                # Margin calls using smooth max/min
                margin_high = smooth_max_zero(v - collateral - th)
                margin_low = smooth_min_zero(v - collateral - tl)

                # Apply MTA thresholds using smooth step (sigmoid)
                # smooth_step(x > threshold) ≈ sigmoid((x - threshold) / scale)
                scale = 0.1
                exp_high = (-(margin_high - mta_h) / scale).exp()
                exp_low = ((margin_low + mta_l) / scale).exp()
                step_high = 1.0 / (1.0 + exp_high)
                step_low = 1.0 / (1.0 + exp_low)

                collateral = collateral + margin_high * step_high
                collateral = collateral + margin_low * step_low

                # Exposure after collateral
                csa_price = v - collateral
                pee = smooth_max_zero(csa_price)
                nee = smooth_min_zero(csa_price)
                pee_list.append(pee)
                nee_list.append(nee)

            # CVA/DVA integration (trapezoidal rule)
            cva = aadc.idouble(0.0)
            dva = aadc.idouble(0.0)

            for t in range(self.num_pricing_times - 1):
                # Survival probability changes
                d_ctrp = self.ctrparty_surv[t] - self.ctrparty_surv[t + 1]
                d_comp = self.company_surv[t] - self.company_surv[t + 1]

                # Trapezoidal integration
                cva = cva + 0.5 * (pee_list[t] + pee_list[t + 1]) * d_ctrp
                dva = dva + 0.5 * (nee_list[t] + nee_list[t + 1]) * d_comp

            # Mark outputs
            self.cva_output = cva.mark_as_output()
            self.dva_output = dva.mark_as_output()

        self.funcs = funcs
        self.kernel_recorded = True
        self.recording_time = time.perf_counter() - start

    def evaluate(self, v_portfolio_paths: np.ndarray,
                 compute_sensitivities: bool = False) -> Tuple[float, float, Optional[dict]]:
        """Evaluate kernel on portfolio values.

        Args:
            v_portfolio_paths: (num_paths, num_pricing_times) array
            compute_sensitivities: if True, compute d(CVA)/d(V_t) and d(DVA)/d(V_t)

        Returns:
            (cva, dva, sensitivities) where sensitivities is None or dict with:
              - 'dcva_dv': (num_pricing_times,) array of d(CVA)/d(V_t)
              - 'ddva_dv': (num_pricing_times,) array of d(DVA)/d(V_t)
        """
        if not self.kernel_recorded:
            self.record_kernel()

        num_paths = v_portfolio_paths.shape[0]

        if not AADC_AVAILABLE:
            # NumPy fallback (no sensitivities)
            cva, dva = self._evaluate_numpy(v_portfolio_paths)
            return cva, dva, None

        # AADC evaluation - batch all paths at once
        # inputs: dict mapping handle -> array of values (one per path)
        inputs = {self.v_handles[t]: v_portfolio_paths[:, t]
                  for t in range(self.num_pricing_times)}

        # request: which outputs to compute and which gradients
        if compute_sensitivities:
            # Request gradients w.r.t. all V_t inputs
            request = {
                self.cva_output: self.v_handles,
                self.dva_output: self.v_handles,
            }
        else:
            request = {self.cva_output: [], self.dva_output: []}

        # Create thread pool
        workers = aadc.ThreadPool(self.num_threads)

        # Evaluate kernel for all paths at once
        results = aadc.evaluate(self.funcs, request, inputs, workers)

        # Extract results (arrays of length num_paths)
        cva_paths = np.array(results[0][self.cva_output])
        dva_paths = np.array(results[0][self.dva_output])

        sensitivities = None
        if compute_sensitivities and len(results) > 1:
            # results[1] contains gradients: {output_handle: {input_handle: gradient_array}}
            dcva_dv = np.zeros(self.num_pricing_times)
            ddva_dv = np.zeros(self.num_pricing_times)

            for t in range(self.num_pricing_times):
                handle = self.v_handles[t]
                # Average gradient across paths
                if self.cva_output in results[1] and handle in results[1][self.cva_output]:
                    dcva_dv[t] = np.mean(results[1][self.cva_output][handle])
                if self.dva_output in results[1] and handle in results[1][self.dva_output]:
                    ddva_dv[t] = np.mean(results[1][self.dva_output][handle])

            sensitivities = {
                'dcva_dv': dcva_dv,
                'ddva_dv': ddva_dv,
                'num_params': self.num_pricing_times * 2,  # CVA + DVA sensitivities
            }

        return np.mean(cva_paths), np.mean(dva_paths), sensitivities

    def _evaluate_numpy(self, v_portfolio_paths: np.ndarray) -> Tuple[float, float]:
        """NumPy fallback when AADC not available."""
        num_paths, num_times = v_portfolio_paths.shape
        mta_h = self.csa_params['mta_h']
        mta_l = self.csa_params['mta_l']
        th = self.csa_params['th']
        tl = self.csa_params['tl']

        cva_total = 0.0
        dva_total = 0.0

        for path in range(num_paths):
            collateral = 0.0
            pee_list = []
            nee_list = []

            for t in range(num_times):
                v = v_portfolio_paths[path, t]

                margin_high = max(v - collateral - th, 0.0)
                margin_low = min(v - collateral - tl, 0.0)

                if margin_high > mta_h:
                    collateral += margin_high
                if margin_low < -mta_l:
                    collateral += margin_low

                csa_price = v - collateral
                pee_list.append(max(csa_price, 0.0))
                nee_list.append(min(csa_price, 0.0))

            # Integration
            cva = 0.0
            dva = 0.0
            for t in range(num_times - 1):
                d_ctrp = self.ctrparty_surv[t] - self.ctrparty_surv[t + 1]
                d_comp = self.company_surv[t] - self.company_surv[t + 1]
                cva += 0.5 * (pee_list[t] + pee_list[t + 1]) * d_ctrp
                dva += 0.5 * (nee_list[t] + nee_list[t + 1]) * d_comp

            cva_total += cva
            dva_total += dva

        return cva_total / num_paths, dva_total / num_paths


# =============================================================================
# Trade Valuation (outside AADC - pure Python/NumPy)
# =============================================================================

def value_trade_hw(trade: dict, rates: np.ndarray, pricing_times: np.ndarray,
                   hw_params: dict) -> np.ndarray:
    """Value a single trade at all pricing times using Hull-White model.

    Args:
        trade: Trade parameters (notional, fixed_rate, cf_times, etc.)
        rates: (num_paths, num_steps) short rate paths
        pricing_times: (num_pricing_times,) array
        hw_params: Hull-White parameters

    Returns:
        (num_paths, num_pricing_times) trade values
    """
    num_paths = rates.shape[0]
    num_pricing_times = len(pricing_times)
    alpha = hw_params['alpha']
    sigma = hw_params['sigma']
    r0 = hw_params['r0']

    values = np.zeros((num_paths, num_pricing_times))

    for pt_idx, t in enumerate(pricing_times):
        # Get rate at this pricing time
        step_idx = min(int(t), rates.shape[1] - 1)
        r_t = rates[:, step_idx]

        # Fixed leg
        fixed_value = np.zeros(num_paths)
        for cf_time, cf_amount in zip(trade['fixed_cf_times'], trade['fixed_cf_amounts']):
            if cf_time > t:
                tau = (cf_time - t) / 365.0
                if tau > 0:
                    B = (1.0 - np.exp(-alpha * tau)) / alpha if alpha > 1e-10 else tau
                    A = np.exp((B - tau) * (alpha * alpha * r0 - 0.5 * sigma * sigma) / (alpha * alpha)
                               - (sigma * sigma * B * B) / (4.0 * alpha))
                    df = A * np.exp(-B * r_t)
                    fixed_value += trade['notional'] * cf_amount * df

        # Float leg (simplified - just use discount factor)
        float_value = np.zeros(num_paths)
        for cf_time, cf_amount in zip(trade['float_cf_times'], trade['float_cf_amounts']):
            if cf_time > t:
                tau = (cf_time - t) / 365.0
                if tau > 0:
                    B = (1.0 - np.exp(-alpha * tau)) / alpha if alpha > 1e-10 else tau
                    A = np.exp((B - tau) * (alpha * alpha * r0 - 0.5 * sigma * sigma) / (alpha * alpha)
                               - (sigma * sigma * B * B) / (4.0 * alpha))
                    df = A * np.exp(-B * r_t)
                    # Approximate forward rate
                    float_value -= trade['notional'] * cf_amount * r_t * tau * df

        values[:, pt_idx] = fixed_value + float_value

    return values


def simulate_rates_hw(num_paths: int, num_steps: int, hw_params: dict,
                      seed: int = 42) -> np.ndarray:
    """Simulate Hull-White short rate paths."""
    np.random.seed(seed)
    alpha = hw_params['alpha']
    sigma = hw_params['sigma']
    r0 = hw_params['r0']
    theta = hw_params.get('theta', r0)
    dt = 1.0 / 365.0

    rates = np.zeros((num_paths, num_steps))
    rates[:, 0] = r0

    for t in range(1, num_steps):
        z = np.random.randn(num_paths)
        drift = (theta - alpha * rates[:, t-1]) * dt
        diffusion = sigma * np.sqrt(dt) * z
        rates[:, t] = rates[:, t-1] + drift + diffusion

    return rates


# =============================================================================
# Modular XVA Engine (AADC version)
# =============================================================================

class AADCModularXVAEngine:
    """AADC-based modular XVA engine with SIMM-style kernel reuse."""

    def __init__(self, num_pricing_times: int, hw_params: dict,
                 csa_params: dict, ctrparty_surv: np.ndarray,
                 company_surv: np.ndarray, num_threads: int = 4):
        self.num_pricing_times = num_pricing_times
        self.hw_params = hw_params
        self.csa_params = csa_params
        self.ctrparty_surv = ctrparty_surv
        self.company_surv = company_surv
        self.num_threads = num_threads

        # CSA+CVA kernel (recorded once, reused for all evaluations)
        self.csa_cva_kernel = AADCCSACVAKernel(
            num_pricing_times, csa_params, ctrparty_surv, company_surv, num_threads
        )

    def record_kernel(self):
        """Record the CSA+CVA kernel (one-time cost)."""
        self.csa_cva_kernel.record_kernel()

    def value_portfolio(self, trades: List[dict], rates: np.ndarray,
                        pricing_times: np.ndarray) -> np.ndarray:
        """Value entire portfolio (outside AADC).

        Returns:
            (num_paths, num_pricing_times) portfolio values
        """
        num_paths = rates.shape[0]
        portfolio_values = np.zeros((num_paths, self.num_pricing_times))

        for trade in trades:
            trade_values = value_trade_hw(trade, rates, pricing_times, self.hw_params)
            portfolio_values += trade_values

        return portfolio_values

    def compute_cva_dva(self, portfolio_values: np.ndarray,
                        compute_sensitivities: bool = False) -> Tuple[float, float, Optional[dict]]:
        """Compute CVA/DVA using cached AADC kernel.

        Args:
            portfolio_values: (num_paths, num_pricing_times) array
            compute_sensitivities: if True, compute d(CVA)/d(V_t) and d(DVA)/d(V_t)

        Returns:
            (cva, dva, sensitivities) where sensitivities is None or dict
        """
        return self.csa_cva_kernel.evaluate(portfolio_values, compute_sensitivities)

    def run_full_xva(self, trades: List[dict], rates: np.ndarray,
                     pricing_times: np.ndarray,
                     compute_sensitivities: bool = False) -> dict:
        """Run full XVA calculation.

        Args:
            trades: List of trade dictionaries
            rates: (num_paths, num_steps) short rate paths
            pricing_times: (num_pricing_times,) pricing grid
            compute_sensitivities: if True, compute XVA sensitivities via AAD

        Returns:
            dict with CVA, DVA, timings, and optionally sensitivities
        """
        # Record kernel if not done
        if not self.csa_cva_kernel.kernel_recorded:
            self.record_kernel()

        # Value portfolio
        t0 = time.perf_counter()
        portfolio_values = self.value_portfolio(trades, rates, pricing_times)
        valuation_time = time.perf_counter() - t0

        # Compute CVA/DVA (and sensitivities if requested)
        t1 = time.perf_counter()
        cva, dva, sensitivities = self.compute_cva_dva(portfolio_values, compute_sensitivities)
        cva_time = time.perf_counter() - t1

        # Separate timing for sensitivities if computed
        sensitivity_time = cva_time if compute_sensitivities else 0.0

        result = {
            'cva': cva,
            'dva': dva,
            'portfolio_values': portfolio_values,
            'valuation_time_sec': valuation_time,
            'cva_time_sec': cva_time,
            'kernel_recording_sec': self.csa_cva_kernel.recording_time,
            'sensitivity_time_sec': sensitivity_time,
        }

        if sensitivities is not None:
            result['sensitivities'] = sensitivities
            result['num_sensitivity_params'] = sensitivities.get('num_params', 0)

        return result

    def add_trade_incremental(self, new_trade: dict,
                              existing_portfolio_values: np.ndarray,
                              rates: np.ndarray,
                              pricing_times: np.ndarray,
                              compute_sensitivities: bool = False) -> dict:
        """Add new trade incrementally (reuses kernel).

        Args:
            new_trade: New trade dictionary
            existing_portfolio_values: (num_paths, num_pricing_times) current portfolio
            rates: (num_paths, num_steps) short rate paths
            pricing_times: (num_pricing_times,) pricing grid
            compute_sensitivities: if True, compute XVA sensitivities via AAD

        Returns:
            dict with CVA, DVA, timings, and optionally sensitivities
        """
        # Value new trade
        t0 = time.perf_counter()
        new_trade_values = value_trade_hw(new_trade, rates, pricing_times, self.hw_params)
        valuation_time = time.perf_counter() - t0

        # Update portfolio
        updated_portfolio = existing_portfolio_values + new_trade_values

        # Compute CVA/DVA (reuses kernel)
        t1 = time.perf_counter()
        cva, dva, sensitivities = self.compute_cva_dva(updated_portfolio, compute_sensitivities)
        cva_time = time.perf_counter() - t1

        # Separate timing for sensitivities if computed
        sensitivity_time = cva_time if compute_sensitivities else 0.0

        result = {
            'cva': cva,
            'dva': dva,
            'new_trade_values': new_trade_values,
            'updated_portfolio_values': updated_portfolio,
            'valuation_time_sec': valuation_time,
            'cva_time_sec': cva_time,
            'sensitivity_time_sec': sensitivity_time,
            'kernel_reused': True,  # Always reused after initial recording
        }

        if sensitivities is not None:
            result['sensitivities'] = sensitivities
            result['num_sensitivity_params'] = sensitivities.get('num_params', 0)

        return result


# =============================================================================
# Helper Functions
# =============================================================================

def generate_test_trades(num_trades: int, seed: int = 42) -> List[dict]:
    """Generate test swap trades."""
    np.random.seed(seed)
    trades = []

    for i in range(num_trades):
        num_cfs = np.random.choice([8, 12, 20, 40])  # 2Y, 3Y, 5Y, 10Y
        cf_spacing = 90  # quarterly

        trades.append({
            'trade_id': i,
            'notional': 1_000_000 * (1 + np.random.random()),
            'fixed_rate': 0.02 + 0.01 * np.random.random(),
            'fixed_cf_times': np.arange(1, num_cfs + 1) * cf_spacing,
            'fixed_cf_amounts': np.ones(num_cfs) * cf_spacing / 365.0,
            'float_cf_times': np.arange(1, num_cfs + 1) * cf_spacing,
            'float_cf_amounts': np.ones(num_cfs) * cf_spacing / 365.0,
        })

    return trades


def build_survival_curve(num_times: int, hazard_rate: float = 0.02) -> np.ndarray:
    """Build survival probability curve."""
    times = np.arange(num_times) * 30 / 365.0
    return np.exp(-hazard_rate * times)


def log_result(scenario: str, num_trades: int, num_paths: int,
               num_pricing_times: int, num_threads: int, cva: float, dva: float,
               valuation_time: float, cva_time: float,
               kernel_recording: float, kernel_reused: bool,
               sensitivity_time: float = 0.0):
    """Log result to CSV.

    Args:
        scenario: Scenario name
        num_trades: Number of trades
        num_paths: Number of MC paths
        num_pricing_times: Number of pricing times
        num_threads: Number of AADC threads
        cva: CVA result
        dva: DVA result
        valuation_time: Trade valuation time (sec)
        cva_time: CVA/DVA computation time (sec)
        kernel_recording: Kernel recording time (sec)
        kernel_reused: Whether kernel was reused
        sensitivity_time: Time spent on sensitivity computation (sec)
    """
    # For AADC, sensitivity computation is included in cva_time
    # (single reverse sweep gets all gradients)
    total_eval_time = valuation_time + cva_time

    row = {
        'timestamp': datetime.now().isoformat(),
        'model_name': f'xva_aadc_modular_{scenario}',
        'model_version': '1.0.0',
        'num_trades': num_trades,
        'num_mc_paths': num_paths,
        'num_model_steps': 365,
        'num_pricing_times': num_pricing_times,
        'num_sensitivity_params': num_pricing_times * 2,  # CVA + DVA sensitivities
        'num_threads': num_threads,
        'backend': 'aadc_modular',
        'mode': 'pricing_with_greeks' if sensitivity_time > 0 or scenario == 'sensitivities' else 'pricing_only',
        'cva_result': cva,
        'dva_result': dva,
        'eval_time_sec': total_eval_time,
        'sensitivity_time_sec': sensitivity_time if sensitivity_time > 0 else cva_time,
        'total_time_sec': total_eval_time + kernel_recording,
        'kernel_recording_sec': kernel_recording,
        'num_params_bumped': 0 if kernel_reused else 1,
        'speedup_vs_cpu': 1 if kernel_reused else 0,
        'max_cva_diff': 0.0,
        'max_dva_diff': 0.0,
        'gpu_kernel_time_sec': 0.0,
        'memory_mb': 0.0,
        'throughput_paths_per_sec': num_paths / total_eval_time if total_eval_time > 0 else 0,
        'status': 'success',
    }
    write_xva_log(LOG_FILE, [row])
    print(f"  Results logged to {LOG_FILE}")


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(num_trades: int = 100, num_paths: int = 10000,
                  num_threads: int = 4, scenario: str = 'all'):
    """Run AADC modular XVA benchmark."""

    print("=" * 70)
    print("  AADC Modular XVA Benchmark (SIMM-style architecture)")
    print("=" * 70)
    print(f"  Trades: {num_trades}")
    print(f"  MC Paths: {num_paths}")
    print(f"  Threads: {num_threads}")
    print(f"  AADC available: {AADC_AVAILABLE}")
    print()

    # Configuration
    num_pricing_times = 122
    num_steps = 365

    hw_params = {
        'alpha': 0.03,
        'sigma': 0.01,
        'r0': 0.025,
        'theta': 0.025,
    }

    csa_params = {
        'mta_h': 1.0,
        'mta_l': 1.0,
        'th': 10.0,
        'tl': -10.0,
    }

    # Generate data
    print("  Generating trades and simulating rates...")
    trades = generate_test_trades(num_trades)
    rates = simulate_rates_hw(num_paths, num_steps, hw_params)
    pricing_times = np.linspace(0, 365 * 10, num_pricing_times)
    ctrparty_surv = build_survival_curve(num_pricing_times, 0.02)
    company_surv = build_survival_curve(num_pricing_times, 0.01)

    # Create engine
    engine = AADCModularXVAEngine(
        num_pricing_times, hw_params, csa_params, ctrparty_surv, company_surv,
        num_threads=num_threads
    )

    # === Scenario 1: Full Portfolio (record kernel) ===
    if scenario in ['all', 'full']:
        print()
        print("=" * 70)
        print("  SCENARIO 1: Full Portfolio (Record Kernel)")
        print("=" * 70)

        result = engine.run_full_xva(trades, rates, pricing_times)

        print(f"\n  Results:")
        print(f"    CVA: {result['cva']:.6f}")
        print(f"    DVA: {result['dva']:.6f}")
        print(f"\n  Timing:")
        print(f"    Kernel recording: {result['kernel_recording_sec']*1000:.1f}ms (one-time)")
        print(f"    Trade valuation:  {result['valuation_time_sec']*1000:.1f}ms")
        print(f"    CVA/DVA calc:     {result['cva_time_sec']*1000:.1f}ms")

        log_result('full_portfolio', num_trades, num_paths, num_pricing_times,
                   num_threads, result['cva'], result['dva'],
                   result['valuation_time_sec'], result['cva_time_sec'],
                   result['kernel_recording_sec'], kernel_reused=False)

    # === Scenario 2: Market Data Update (reuse kernel) ===
    if scenario in ['all', 'market_update']:
        print()
        print("=" * 70)
        print("  SCENARIO 2: Market Data Update (Reuse Kernel)")
        print("=" * 70)

        # Bump rates
        hw_params_bumped = hw_params.copy()
        hw_params_bumped['r0'] = hw_params['r0'] * 1.1
        print(f"  r0: {hw_params['r0']} -> {hw_params_bumped['r0']}")

        # Re-simulate rates
        rates2 = simulate_rates_hw(num_paths, num_steps, hw_params_bumped, seed=43)

        # Re-value portfolio (kernel already recorded)
        t0 = time.perf_counter()
        portfolio_values2 = engine.value_portfolio(trades, rates2, pricing_times)
        valuation_time2 = time.perf_counter() - t0

        t1 = time.perf_counter()
        cva2, dva2, sens2 = engine.compute_cva_dva(portfolio_values2, compute_sensitivities=True)
        cva_time2 = time.perf_counter() - t1

        print(f"\n  Results (kernel reused, with sensitivities):")
        print(f"    CVA: {cva2:.6f}")
        print(f"    DVA: {dva2:.6f}")
        if sens2:
            print(f"    Sensitivity params: {sens2.get('num_params', 0)}")
        print(f"\n  Timing:")
        print(f"    Kernel recording: 0.0ms (REUSED)")
        print(f"    Trade valuation:  {valuation_time2*1000:.1f}ms")
        print(f"    CVA/DVA + sens:   {cva_time2*1000:.1f}ms")

        log_result('market_update', num_trades, num_paths, num_pricing_times,
                   num_threads, cva2, dva2, valuation_time2, cva_time2,
                   kernel_recording=0.0, kernel_reused=True)

    # === Scenario 3: New Trade (reuse kernel) ===
    if scenario in ['all', 'new_trade']:
        print()
        print("=" * 70)
        print("  SCENARIO 3: New Trade (Reuse Kernel)")
        print("=" * 70)

        # Get base portfolio values
        portfolio_values = engine.value_portfolio(trades, rates, pricing_times)

        # Add new trade
        new_trade = generate_test_trades(1, seed=999)[0]
        print(f"  Adding trade: notional={new_trade['notional']:.0f}")

        result3 = engine.add_trade_incremental(
            new_trade, portfolio_values, rates, pricing_times
        )

        print(f"\n  Results (kernel reused):")
        print(f"    CVA: {result3['cva']:.6f}")
        print(f"    DVA: {result3['dva']:.6f}")
        print(f"\n  Timing:")
        print(f"    Kernel recording: 0.0ms (REUSED)")
        print(f"    New trade valuation: {result3['valuation_time_sec']*1000:.1f}ms")
        print(f"    CVA/DVA calc:        {result3['cva_time_sec']*1000:.1f}ms")

        log_result('new_trade', num_trades + 1, num_paths, num_pricing_times,
                   num_threads, result3['cva'], result3['dva'],
                   result3['valuation_time_sec'], result3['cva_time_sec'],
                   kernel_recording=0.0, kernel_reused=True)

    # === Scenario 4: Sensitivity Computation ===
    if scenario in ['all', 'sensitivities']:
        print()
        print("=" * 70)
        print("  SCENARIO 4: Sensitivity Computation (AADC Reverse-Mode AD)")
        print("=" * 70)

        # Run full XVA with sensitivities
        result_sens = engine.run_full_xva(trades, rates, pricing_times,
                                          compute_sensitivities=True)

        print(f"\n  Results:")
        print(f"    CVA: {result_sens['cva']:.6f}")
        print(f"    DVA: {result_sens['dva']:.6f}")
        if 'sensitivities' in result_sens:
            sens = result_sens['sensitivities']
            print(f"    Sensitivity params: {sens.get('num_params', 0)}")
            # Show sample sensitivities
            dcva = sens.get('dcva_dv', np.array([]))
            ddva = sens.get('ddva_dv', np.array([]))
            if len(dcva) > 0:
                print(f"    d(CVA)/d(V_0): {dcva[0]:.6f}")
                print(f"    d(CVA)/d(V_mid): {dcva[len(dcva)//2]:.6f}")
                print(f"    d(DVA)/d(V_0): {ddva[0]:.6f}")
                print(f"    d(DVA)/d(V_mid): {ddva[len(ddva)//2]:.6f}")
        print(f"\n  Timing:")
        print(f"    Kernel recording:  {result_sens['kernel_recording_sec']*1000:.1f}ms")
        print(f"    Trade valuation:   {result_sens['valuation_time_sec']*1000:.1f}ms")
        print(f"    CVA/DVA + sens:    {result_sens['cva_time_sec']*1000:.1f}ms")
        print(f"    Sensitivity time:  {result_sens.get('sensitivity_time_sec', 0)*1000:.1f}ms")
        print()
        print("  Note: AADC computes sensitivities in a SINGLE reverse sweep!")
        print("  No bump-and-revalue required. O(1) vs O(N) complexity.")

        log_result('sensitivities', num_trades, num_paths, num_pricing_times,
                   num_threads, result_sens['cva'], result_sens['dva'],
                   result_sens['valuation_time_sec'], result_sens['cva_time_sec'],
                   result_sens['kernel_recording_sec'], kernel_reused=True)

    # === Summary ===
    print()
    print("=" * 70)
    print("  SUMMARY: AADC Modular Kernel Architecture")
    print("=" * 70)
    print(f"  Kernel size: O({num_pricing_times} pricing times)")
    print(f"  Independent of trade count!")
    print()
    print("  Key insight: CSA+CVA kernel takes V_portfolio(t) as input.")
    print("  Trade valuation happens OUTSIDE the kernel (pure Python).")
    print("  Adding new trades just updates V_portfolio and reuses kernel.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AADC Modular XVA Benchmark')
    parser.add_argument('--num-trades', '-t', type=int, default=100)
    parser.add_argument('--mc-paths', '-m', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of AADC threads (default: 4)')
    parser.add_argument('--scenario', choices=['all', 'full', 'market_update', 'new_trade', 'sensitivities'],
                        default='all')

    args = parser.parse_args()

    run_benchmark(
        num_trades=args.num_trades,
        num_paths=args.mc_paths,
        num_threads=args.threads,
        scenario=args.scenario
    )
