"""XVA Benchmark: shared data structures, parsers, and utilities.

Ports the C++ XVA-Benchmark data loading and Hull-White model
to Python for CPU baseline and GPU kernel validation.

Version: 1.0.0
"""
MODEL_VERSION = "1.0.0"

import json
import math
import os
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HWModelParams:
    alpha: float
    sigma: float
    r0: float
    mean_rev_times: np.ndarray   # (N,) float64, in years
    mean_rev_vals: np.ndarray    # (N,) float64
    spread_3m_times: np.ndarray
    spread_3m_vals: np.ndarray
    spread_6m_times: np.ndarray
    spread_6m_vals: np.ndarray
    spread_12m_times: np.ndarray
    spread_12m_vals: np.ndarray


@dataclass
class SurvivalCurveParams:
    times_days: np.ndarray   # (N,) int32 qtime
    times_years: np.ndarray  # (N,) float64, times_days / 365
    values: np.ndarray       # (N,) float64, zero rates
    t0: int


@dataclass
class CSAParams:
    th: float
    tl: float
    mta_h: float
    mta_l: float
    c_t: float


@dataclass
class TradeData:
    """All trades, struct-of-arrays layout."""
    # Fixed legs: (num_trades, max_cashflows)
    fixed_amounts: np.ndarray
    fixed_times: np.ndarray       # int32 qtime
    fixed_num_cfs: np.ndarray     # int32
    # Float legs: (num_trades, max_cashflows)
    float_notionals: np.ndarray
    float_start_times: np.ndarray  # int32 qtime
    float_end_times: np.ndarray    # int32 qtime
    float_pay_times: np.ndarray    # int32 qtime
    float_spread_ids: np.ndarray   # int32
    float_num_cfs: np.ndarray      # int32
    num_trades: int
    max_cf: int


@dataclass
class SimulationGrid:
    model_times: np.ndarray    # int32 qtime (num_steps,)
    is_pricing: np.ndarray     # bool (num_steps,)
    pricing_times: np.ndarray  # int32 qtime (num_pricing,)
    t0: int


@dataclass
class XVAResult:
    backend: str
    mode: str
    cva: float
    dva: float
    primal_time_sec: float = 0.0
    sensitivity_time_sec: float = 0.0
    total_time_sec: float = 0.0
    kernel_recording_sec: float = 0.0
    gpu_kernel_time_sec: float = 0.0
    num_params_bumped: int = 0
    pee: Optional[np.ndarray] = None
    nee: Optional[np.ndarray] = None
    sensitivities: Optional[Dict] = None


# ---------------------------------------------------------------------------
# JSON Parsers
# ---------------------------------------------------------------------------

def load_init_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_pw_curve(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Build piecewise linear curve arrays from JSON spec.
    Returns (times_years, values)."""
    step = data["step"]
    max_t = data["T"]
    flat_rate = data["flat_rate"]
    times = []
    vals = []
    t = 0.0
    while t < max_t:
        times.append(t)
        vals.append(flat_rate)
        t += step
    return np.array(times, dtype=np.float64), np.array(vals, dtype=np.float64)


def parse_hw_model(data: dict) -> HWModelParams:
    eur = data["Currencies"]["EUR"]
    mr_t, mr_v = _build_pw_curve(eur["HWMeanReversionCurve"])
    s3_t, s3_v = _build_pw_curve(eur["ProjectionSpread.3M"])
    s6_t, s6_v = _build_pw_curve(eur["ProjectionSpread.6M"])
    s12_t, s12_v = _build_pw_curve(eur["ProjectionSpread.12M"])
    return HWModelParams(
        alpha=eur["alpha"],
        sigma=eur["sigma"],
        r0=eur["r0"],
        mean_rev_times=mr_t, mean_rev_vals=mr_v,
        spread_3m_times=s3_t, spread_3m_vals=s3_v,
        spread_6m_times=s6_t, spread_6m_vals=s6_v,
        spread_12m_times=s12_t, spread_12m_vals=s12_v,
    )


def parse_survival_curve(data: dict, t0: int) -> SurvivalCurveParams:
    step = data["step"]
    max_t = data["T"]
    flat_rate = data["flat_rate"]
    times = []
    t = t0
    while t < max_t:
        times.append(t)
        t += step
    td = np.array(times, dtype=np.int32)
    ty = td.astype(np.float64) / 365.0
    vals = np.full(len(times), flat_rate, dtype=np.float64)
    return SurvivalCurveParams(times_days=td, times_years=ty, values=vals, t0=t0)


def parse_csa(data: dict) -> CSAParams:
    return CSAParams(
        th=data["th"], tl=data["tl"],
        mta_h=data["mta_h"], mta_l=data["mta_l"],
        c_t=data["c_t"],
    )


def parse_simulation_grid(data: dict, t0: int) -> SimulationGrid:
    """Port of readModelAndPricingTimes from DataTools.h."""
    max_t = data["T"]
    model_step = data["step"]
    pricing_freq = data["PricingFreq"]

    model_times = []
    is_pricing = []
    pricing_times = []
    pr = 0
    t = t0
    while t < max_t:
        model_times.append(t)
        if pr == 0 or (t + model_step > max_t):
            pricing_times.append(t)
            pr = pricing_freq
            is_pricing.append(True)
        else:
            is_pricing.append(False)
        pr -= 1
        t += model_step

    return SimulationGrid(
        model_times=np.array(model_times, dtype=np.int32),
        is_pricing=np.array(is_pricing, dtype=np.bool_),
        pricing_times=np.array(pricing_times, dtype=np.int32),
        t0=t0,
    )


# ---------------------------------------------------------------------------
# Portfolio Generation (mt19937_64 port)
# ---------------------------------------------------------------------------

class MT19937_64:
    """Minimal port of C++ std::mt19937_64 for reproducibility.

    Uses the exact same 64-bit Mersenne Twister algorithm as libstdc++.
    This is necessary because numpy uses a 32-bit MT internally.
    """

    _w, _n, _m, _r = 64, 312, 156, 31
    _a = 0xB5026F5AA96619E9
    _u, _d = 29, 0x5555555555555555
    _s, _b = 17, 0x71D67FFFEDA60000
    _t, _c = 37, 0xFFF7EEE000000000
    _l = 43
    _mask = (1 << 64) - 1
    _lower_mask = (1 << _r) - 1
    _upper_mask = _mask & ~_lower_mask

    def __init__(self, seed: int = 5489):
        self.mt = [0] * self._n
        self.index = self._n + 1
        self._has_cached = False
        self._cached_normal = 0.0
        self._seed(seed)

    def _seed(self, seed: int):
        self.mt[0] = seed & self._mask
        for i in range(1, self._n):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & self._mask
        self.index = self._n

    def _twist(self):
        for i in range(self._n):
            x = (self.mt[i] & self._upper_mask) | (self.mt[(i + 1) % self._n] & self._lower_mask)
            xA = x >> 1
            if x & 1:
                xA ^= self._a
            self.mt[i] = self.mt[(i + self._m) % self._n] ^ xA
        self.index = 0

    def __call__(self) -> int:
        if self.index >= self._n:
            self._twist()
        y = self.mt[self.index]
        y ^= (y >> self._u) & self._d
        y ^= (y << self._s) & self._b
        y ^= (y << self._t) & self._c
        y ^= y >> self._l
        self.index += 1
        return y & self._mask

    def uniform(self) -> float:
        """Equivalent to std::uniform_real_distribution<>(0, 1)."""
        # libstdc++ uses generate_canonical: draw one 64-bit int, divide by 2^64
        # Actually it uses (gen() - gen.min()) / (gen.max() - gen.min() + 1.0)
        return self() / (self._mask + 1.0)

    def uniform_int(self, a: int, b: int) -> int:
        """Equivalent to std::uniform_int_distribution<int>(a, b)."""
        range_size = b - a + 1
        return a + (self() % range_size)

    def normal(self) -> float:
        """Equivalent to std::normal_distribution<>(0, 1).

        libstdc++ uses the Marsaglia polar method.  It returns v*m first
        and caches u*m for the next call (see libstdc++ bits/random.tcc).
        """
        if self._has_cached:
            self._has_cached = False
            return self._cached_normal

        while True:
            u = 2.0 * self.uniform() - 1.0
            v = 2.0 * self.uniform() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                break
        m = math.sqrt(-2.0 * math.log(s) / s)
        self._cached_normal = u * m
        self._has_cached = True
        return v * m


def generate_portfolio(t0: int, num_trades: int, num_periods: int, seed: int = 17) -> TradeData:
    """Port of readPortfolio from DataTools.h.

    Uses the same mt19937_64(seed=17) as C++ for identical trade generation.
    """
    gen = MT19937_64(seed)

    fixed_amounts_list = []
    fixed_times_list = []
    float_notionals_list = []
    float_start_list = []
    float_end_list = []
    float_pay_list = []
    float_spread_list = []

    for ti in range(num_trades):
        # generateFixedLeg
        f_amounts = []
        f_times = []
        start = t0 + int(10 * 365 * gen.uniform())
        period = int(2 * 365 * gen.uniform() + 30)
        for _ in range(num_periods):
            start += period
            f_times.append(start)
            f_amounts.append(0.15 * (gen.uniform() * 2.0 - 1.0))
        fixed_amounts_list.append(f_amounts)
        fixed_times_list.append(f_times)

        # generateFloatLeg
        fl_notionals = []
        fl_starts = []
        fl_ends = []
        start2 = t0 + int(10 * 365 * gen.uniform())
        period2 = int(2 * 365 * gen.uniform() + 30)
        for _ in range(num_periods):
            fl_starts.append(start2)
            start2 += period2
            fl_ends.append(start2)
            fl_notionals.append(2 * gen.uniform() - 0.985)
        spread_id = gen.uniform_int(0, 2)
        float_notionals_list.append(fl_notionals)
        float_start_list.append(fl_starts)
        float_end_list.append(fl_ends)
        float_pay_list.append(fl_ends[:])  # pay_times = end_times in C++
        float_spread_list.append(spread_id)

    max_cf = num_periods

    def pad2d(lst, dtype=np.float64):
        arr = np.zeros((num_trades, max_cf), dtype=dtype)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                arr[i, j] = val
        return arr

    return TradeData(
        fixed_amounts=pad2d(fixed_amounts_list),
        fixed_times=pad2d(fixed_times_list, np.int32),
        fixed_num_cfs=np.full(num_trades, num_periods, dtype=np.int32),
        float_notionals=pad2d(float_notionals_list),
        float_start_times=pad2d(float_start_list, np.int32),
        float_end_times=pad2d(float_end_list, np.int32),
        float_pay_times=pad2d(float_pay_list, np.int32),
        float_spread_ids=np.array(float_spread_list, dtype=np.int32),
        float_num_cfs=np.full(num_trades, num_periods, dtype=np.int32),
        num_trades=num_trades,
        max_cf=max_cf,
    )


def generate_randoms(num_paths: int, num_steps: int, seed: int = 17,
                     fast: bool = False) -> np.ndarray:
    """Generate (num_paths, num_steps) normal random matrix.

    If fast=False (default), uses exact mt19937_64 + Marsaglia polar method
    matching C++ for bit-identical results.  Slow for large path counts.

    If fast=True, uses NumPy's RNG for speed.  Results won't match C++ but
    converge to the same distribution at large path counts.
    """
    if fast:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((num_paths, num_steps))

    gen = MT19937_64(seed)
    randoms = np.zeros((num_paths, num_steps), dtype=np.float64)
    for path in range(num_paths):
        for step in range(num_steps):
            randoms[path, step] = gen.normal()
    return randoms


# ---------------------------------------------------------------------------
# Curve Interpolation (matching C++ PiecewiseLinearCurve)
# ---------------------------------------------------------------------------

def pw_interp(times: np.ndarray, vals: np.ndarray, t: float) -> float:
    """Piecewise linear interpolation matching C++ lower_bound logic."""
    idx = np.searchsorted(times, t, side='left')
    if idx == 0:
        return float(vals[0])
    if idx >= len(times):
        return float(vals[-1])
    len_t = times[idx] - times[idx - 1]
    wl = (times[idx] - t) / len_t
    wr = (t - times[idx - 1]) / len_t
    return float(wl * vals[idx - 1] + wr * vals[idx])


def interpolated_index(times: np.ndarray, t: float) -> int:
    """Match C++ PiecewiseLinearCurve::interpolatedIndex."""
    idx = np.searchsorted(times, t, side='left')
    if idx >= len(times):
        idx = len(times) - 1
    return int(idx)


def discount_curve_eval(surv: SurvivalCurveParams, t_qtime: int) -> float:
    """Evaluate LinearInterpDiscountCurve: exp(-rate(t) * yearfrac(t0, t))."""
    t_years = t_qtime / 365.0
    rate = pw_interp(surv.times_years, surv.values, t_years)
    yf = (t_qtime - surv.t0) / 365.0
    return math.exp(-rate * yf)


# ---------------------------------------------------------------------------
# Precompute Cumulatives for HW Bond Pricing
# ---------------------------------------------------------------------------

def precompute_cumulatives(mr_times: np.ndarray, mr_vals: np.ndarray,
                           alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cumulat1 and cumulat2 for HW bond pricing.

    These only depend on the mean reversion curve and alpha.
    Matches the HullWhiteZDBCurve constructor logic.
    """
    n = len(mr_times)
    cumulat1 = np.zeros(n, dtype=np.float64)
    cumulat2 = np.zeros(n, dtype=np.float64)

    for u in range(n - 1):
        delta_t = mr_times[u + 1] - mr_times[u]
        mr_avg = mr_vals[u + 1]  # C++ uses getVals()[u+1]
        cumulat1[u + 1] = cumulat1[u] + delta_t * mr_avg
        cumulat2[u + 1] = cumulat2[u] - (
            math.exp(alpha * mr_times[u + 1]) - math.exp(alpha * mr_times[u])
        ) * mr_avg / alpha

    return cumulat1, cumulat2


# ---------------------------------------------------------------------------
# HW Bond Pricing (CPU reference)
# ---------------------------------------------------------------------------

def hw_bond_price(r_current: float, sigma: float, alpha: float,
                  time_counter: float,
                  mr_times: np.ndarray, mr_vals: np.ndarray,
                  cumulat1: np.ndarray, cumulat2: np.ndarray,
                  t_years: float) -> float:
    """Port of HullWhiteZDBCurve::bond()."""
    s_part = math.exp(-alpha * (t_years - time_counter))
    A_t_T = (1.0 - s_part) / alpha

    T_index = interpolated_index(mr_times, t_years)
    t_cur_index = interpolated_index(mr_times, time_counter)

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


def project_curve_eval_hw(r_current: float, sigma: float, alpha: float,
                          time_counter: float,
                          mr_times: np.ndarray, mr_vals: np.ndarray,
                          cumulat1: np.ndarray, cumulat2: np.ndarray,
                          spread_times: np.ndarray, spread_vals: np.ndarray,
                          t_qtime: int) -> float:
    """Port of ProjectCurve::operator()(qtime)."""
    t_years = t_qtime / 365.0
    disc = hw_bond_price(r_current, sigma, alpha, time_counter,
                         mr_times, mr_vals, cumulat1, cumulat2, t_years)
    spread_val = pw_interp(spread_times, spread_vals, t_years)
    return disc * math.exp(-spread_val * (t_years - time_counter))


# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------

LOG_COLUMNS = [
    "timestamp", "model_name", "model_version",
    "num_trades", "num_mc_paths", "num_model_steps", "num_pricing_times",
    "num_sensitivity_params", "num_threads",
    "backend", "mode",
    "cva_result", "dva_result",
    "primal_time_sec", "sensitivity_time_sec", "total_time_sec",
    "kernel_recording_sec", "num_params_bumped",
    "speedup_vs_cpu", "max_cva_diff", "max_dva_diff",
    "gpu_kernel_time_sec", "status",
]


def write_xva_log(filepath: str, rows: List[dict]):
    """Append rows to execution_log_xva.csv."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_log_row(result: XVAResult, grid: SimulationGrid,
                  trades: TradeData, num_mc_paths: int,
                  num_threads: int, num_sens_params: int,
                  cpu_time: float = 0.0,
                  ref_cva: float = 0.0, ref_dva: float = 0.0) -> dict:
    speedup = cpu_time / result.total_time_sec if result.total_time_sec > 0 and cpu_time > 0 else 0.0
    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": f"xva_{result.mode}_{result.backend}",
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
        "primal_time_sec": result.primal_time_sec,
        "sensitivity_time_sec": result.sensitivity_time_sec,
        "total_time_sec": result.total_time_sec,
        "kernel_recording_sec": result.kernel_recording_sec,
        "num_params_bumped": result.num_params_bumped,
        "speedup_vs_cpu": speedup,
        "max_cva_diff": abs(result.cva - ref_cva) if ref_cva is not None else "",
        "max_dva_diff": abs(result.dva - ref_dva) if ref_dva is not None else "",
        "gpu_kernel_time_sec": result.gpu_kernel_time_sec,
        "status": "success",
    }
