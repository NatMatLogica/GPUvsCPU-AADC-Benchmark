# Modular Two-Kernel XVA Architecture

## Problem: AADC Kernel Size Scales with Trade Count

The current AADC implementation **unrolls all trades** into the compiled kernel:

| Trades | Kernel Size | Compile Time |
|--------|-------------|--------------|
| 5 | 8 MB | 3s |
| 50 | 26 MB | 7s |
| 100 | 170 MB | 15s |
| 200 | 220 MB | 25s+ |

Adding a single new trade requires **recompiling the entire kernel**.

## Solution: Two-Kernel Architecture

Separate trade valuation from portfolio aggregation:

```
┌─────────────────────────────────────────────────────────────────┐
│  Kernel 1: Trade Valuation (per trade type)                     │
│  ─────────────────────────────────────────                      │
│  • Compiled ONCE per trade type (e.g., IRS_5Y, IRS_10Y)         │
│  • Reused for ALL trades of that type                           │
│  • 100 trades × 5 types = 5 kernel compiles                     │
│  • Size: O(1) per trade type                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Sum trade values: V_portfolio(t) = Σ V_trade(t)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Kernel 2: CSA + CVA (portfolio level)                          │
│  ─────────────────────────────────────                          │
│  • Takes aggregated V_portfolio(t) as input                     │
│  • Sequential collateral evolution (path-dependent)             │
│  • Computes PEE/NEE → CVA/DVA                                   │
│  • Size: O(pricing_times), independent of trade count           │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Works

### Trade Valuation is Separable

Each trade's value at time t depends only on:
- Market state (rates, spreads)
- Trade parameters (notional, cash flows)

Trades of the **same type** (same CF structure) use the **same kernel** with different parameters.

### CSA is the Bottleneck (But It's Cheap)

Collateral evolution is path-dependent:
```
C(t) = f(V_portfolio(t), C(t-1), CSA_rules)
```

This creates sequential dependency across time steps. However:
- CSA is pure arithmetic (max/min operations)
- No trade loops inside CSA kernel
- Runs in ~40ms regardless of portfolio size

## Benchmark Results

### Configuration
- 100 trades, 5 trade types
- 51,200 MC paths
- 122 pricing times

### Scenario 1: Full Portfolio Valuation

| Metric | Value |
|--------|-------|
| Trade types compiled | 5 |
| Kernel reuses | 95 |
| Total compile time | 1.55s |
| Total valuation time | 4.4s |
| CVA/DVA time | 0.17s |
| **Total** | **5.6s** |

### Scenario 2: Market Data Update (Kernel Reuse)

| Metric | Value |
|--------|-------|
| Kernels compiled | **0** |
| Kernel reuses | **100** |
| Total time | **2.75s** |

All kernels reused - no recompilation needed for market data changes.

### Scenario 3: New Trade (Incremental)

| Trade Type | Compile Time | Valuation | CVA/DVA | Total |
|------------|--------------|-----------|---------|-------|
| Existing type | **0ms** | 67ms | 35ms | **102ms** |
| New type | 305ms | 733ms | 57ms | **1.1s** |

## Performance Comparison

| Scenario | Monolithic Kernel | Modular Kernel | Speedup |
|----------|-------------------|----------------|---------|
| Add trade (existing type) | 15s recompile | **0ms** | **∞** |
| Add trade (new type) | 15s recompile | **305ms** | **50x** |
| Market data update | 15s recompile | **0ms compile** | **∞** |
| 100→200 trades | 25s+ recompile | **~1s** (new types only) | **25x** |

## Kernel Cache Statistics

| Metric | Value |
|--------|-------|
| Trade types (kernels) | 6 |
| Total kernel reuses | 296 |
| Amortized cost/trade | **6.15ms** |

## Usage

```bash
source /home/x13-root171/GPU_AAD/venv/bin/activate

# Full benchmark (all scenarios)
python xva_modular_kernel.py --num-trades 100 --num-trade-types 5 --mc-paths 51200

# Just market update scenario
python xva_modular_kernel.py --scenario market_update --num-trades 100

# Just new trade scenario
python xva_modular_kernel.py --scenario new_trade --num-trades 100

# Scale test: 1000 trades, 10 types
python xva_modular_kernel.py --num-trades 1000 --num-trade-types 10 --mc-paths 51200
```

## Architecture Comparison: SIMM vs XVA

| Aspect | SIMM (GPU_AAD) | XVA (This Implementation) |
|--------|----------------|---------------------------|
| Aggregation | Sum sensitivities | Sum trade values |
| Non-linearity | SIMM formula | max() for exposure |
| Path dependency | None | Collateral evolution |
| Kernel size | O(risk_factors) | O(trade_types) |
| New trade | Add to agg_S | Run Kernel 1, update sum |

Both achieve **O(1) kernel reuse** for new trades of existing types.

## Limitations

1. **Collateral is sequential**: Cannot parallelize across time steps due to C(t) → C(t+1) dependency

2. **Trade type granularity**: Each unique CF structure requires a separate kernel. Very heterogeneous portfolios may have many types.

3. **Memory for trade values**: Must store V_trade(t) for all trades before aggregation. For 1000 trades × 122 pricing times × 51K paths × 8 bytes = ~50GB.

## Future Work

1. **Batch trades by type**: Process all trades of same type in single kernel launch
2. **Stream processing**: Don't store all trade values; aggregate on-the-fly
3. **AADC integration**: Apply same architecture to C++ AADC for consistent speedups

## AADC Implementation (SIMM-style)

The `xva_aadc_modular.py` implements the same architecture using AADC:

```python
# Record CSA+CVA kernel (takes V_portfolio as input)
with aadc.record_kernel() as funcs:
    v_portfolio = [aadc.idouble(0.0).mark_as_input() for t in range(num_pricing_times)]
    # ... CSA evolution, CVA/DVA integration ...
    cva.mark_as_output()
    dva.mark_as_output()

# Evaluate (reuses kernel for all paths)
inputs = {v_handles[t]: portfolio_values[:, t] for t in range(num_pricing_times)}
results = aadc.evaluate(funcs, request, inputs, workers)
```

### AADC Results (100 trades, 1000 paths)

| Scenario | Kernel Recording | Trade Valuation | CVA/DVA | Total |
|----------|------------------|-----------------|---------|-------|
| Full Portfolio | 66ms | 1840ms | 3ms | 1909ms |
| Market Update | **0ms (reused)** | 1828ms | 3ms | 1830ms |
| New Trade | **0ms (reused)** | 3ms | 3ms | **6ms** |

### Trade-off: Kernel Reuse vs Sensitivities

**Important:** The modular architecture trades sensitivity computation for kernel reuse.

**C++ AADC (XVAServer.cpp) - Full AD:**
```
Kernel records EVERYTHING:
  r0, σ, θ(t) → Rate simulation → Bond prices → Trade values → CSA → CVA/DVA
  └─────────────────────── All inside AADC kernel ───────────────────────────┘

Result: CVA + sensitivities to r0, σ, θ[0..250], survival curves (~535 params)
```

**Python Modular (xva_aadc_modular.py) - Partial AD:**
```
NumPy (no AD):           AADC Kernel:
  r0, σ → rates → trades → V_portfolio(t) → CSA → CVA/DVA
  └──── Outside kernel ────┘               └─ Inside kernel ─┘

Result: CVA only (no sensitivities to r0, σ, θ!)
```

The modular version **loses the ability to compute rate sensitivities** because trade valuation happens outside AADC in pure NumPy:

```python
# xva_aadc_modular.py - trade valuation in NumPy (NOT differentiable)
def value_trade_hw(trade, rates, pricing_times, hw_params):
    # Pure NumPy - AADC doesn't see this
    values = np.zeros((num_paths, num_pricing_times))
    for pt_idx, t in enumerate(pricing_times):
        # ... bond pricing in NumPy ...
    return values
```

### Comparison: Full AD vs Modular

| Aspect | C++ AADC (Full) | Python Modular |
|--------|-----------------|----------------|
| Rate sensitivities (∂CVA/∂r0, ∂CVA/∂σ) | ✓ Yes | ✗ No |
| Mean reversion θ(t) sensitivities | ✓ Yes | ✗ No |
| Survival curve sensitivities | ✓ Yes | ✗ No (could add analytically) |
| Kernel reuse for new trade | ✗ No (recompile all) | ✓ Yes |
| Kernel reuse for market update | ✓ Yes (eval only) | ✓ Yes |
| Kernel size | O(trades × CFs) | O(pricing_times) |

### Timing Comparison: Why cpp_aadc Appears Slower

**IMPORTANT:** Direct timing comparisons between `cpp_aadc` and `aadc_modular` are misleading because they compute fundamentally different things.

**Benchmark at 50 trades, 10,000 paths:**

| Component | aadc_modular | cpp_aadc |
|-----------|--------------|----------|
| Kernel recording | 100ms | 5.5s |
| Trade valuation | 2949ms **(NumPy)** | **(Inside AADC)** |
| CSA+CVA computation | 20ms (AADC) | **(Inside AADC)** |
| **Total AADC kernel time** | **20ms** | **36.9s** |
| **Total wall clock** | **~3.1s** | **~42s** |

**What's happening:**

```
aadc_modular workflow:
  NumPy (no AD):                    AADC kernel (20ms):
  r0,σ → rates → bond prices → V(t) → CSA → CVA
  └────── 2.9s, fast, no AD ───────┘  └─ Only this is recorded ─┘

cpp_aadc workflow:
  ┌────────────────── Entire computation in AADC kernel (36.9s) ──────────────────┐
  │ r0,σ → rates → bond prices (50 trades × 20 CFs × 365 steps) → CSA → CVA      │
  │ Every operation recorded for reverse-mode AD                                   │
  └────────────────────────────────────────────────────────────────────────────────┘
```

**Why the 12× timing difference:**

1. **aadc_modular** uses NumPy for trade valuation — highly optimized vectorized code with no AD overhead
2. **cpp_aadc** records every operation in the AD tape — significant overhead but enables full sensitivity computation

**What you get for the extra time:**

| Metric | aadc_modular (3.1s) | cpp_aadc (42s) |
|--------|---------------------|----------------|
| CVA/DVA | ✓ | ✓ |
| ∂CVA/∂r0 | ✗ | ✓ |
| ∂CVA/∂σ | ✗ | ✓ |
| ∂CVA/∂θ(t) (120 points) | ✗ | ✓ |
| ∂CVA/∂survival (122 points) | ✗ | ✓ |
| **Total sensitivities** | **0** | **244** |

**Per-path efficiency (with sensitivities):**

```
cpp_aadc: 36.9s / 10,000 paths = 3.7ms per path for CVA + 244 sensitivities
          Equivalent bump-and-revalue: 244 × 3.7ms = 902ms per path
          AD speedup: ~244×
```

**Bottom line:** The modular architecture trades sensitivity computation for speed. If you need sensitivities, cpp_aadc is actually very efficient (244× faster than bump-and-revalue). If you only need CVA values, the modular approach is faster.

### Clarification: What "Sensitivity Params" Actually Means

**WARNING:** The CSV `num_sensitivity_params` column is misleading across backends:

| Backend | CSV Reported | Actually Computed | Discrepancy Reason |
|---------|--------------|-------------------|-------------------|
| `cpp_aadc` | 242 | **~535** | CSV logs `num_pricing_times × 2` (bug) |
| `pathwise_gpu` | 284 | **~284** | ✓ Correct (but NO MR curve) |
| `gpu_brute_force` | 284-535 | **284-535** | ✓ Correct (depends on --skip-mr-bumps) |
| `aadc_modular` | 244 | **0 market Greeks** | 244 = exposure-level derivatives only |
| `modular_gpu` | 5 | **0 market Greeks** | 5 = trade type count, NOT sensitivities |

**Critical distinction:**
- `cpp_aadc` computes **~535 market Greeks**: r0, σ, θ(t)×251, survival×282
- `pathwise_gpu` computes **~284 market Greeks**: r0, σ, survival×282 (NO MR curve!)
- `aadc_modular` computes **exposure-level derivatives** (dCVA/dV(t) — NOT market Greeks)
- `modular_gpu` reports trade type count as "sensitivity params" (logging bug)

### Sensitivity Parameter Reconciliation (CRITICAL)

**WARNING:** The three backends compute DIFFERENT sets of market Greeks. Direct comparisons are misleading unless you account for this!

#### What Each Backend Actually Computes

| Parameter Class | cpp_aadc | pathwise_gpu | gpu_brute_force |
|-----------------|----------|--------------|-----------------|
| r0 (initial rate) | ✓ AD | ✓ Pathwise | ✓ Bump (re-sim) |
| σ (volatility) | ✓ AD | ✓ Pathwise | ✓ Bump (re-sim) |
| Mean reversion θ(t) (~251 pts) | ✓ AD | ✗ **NOT IMPLEMENTED** | Optional (--skip-mr-bumps) |
| Counterparty survival (~141 pts) | ✓ AD | ✓ Analytical | ✓ Bump (re-integration) |
| Company survival (~141 pts) | ✓ AD | ✓ Analytical | ✓ Bump (re-integration) |
| **TOTAL** | **~535** | **~284** | **284-535** |

#### Curve Sizes (from initData.json)

```
Mean reversion θ(t):     T=50 years, step=0.2 years → ~251 points
Counterparty survival:   T=14000 days, step=100 days → ~141 points
Company survival:        T=14000 days, step=100 days → ~141 points
```

**Parameter counts:**
- **Full set (cpp_aadc):** 1 + 1 + 251 + 141 + 141 = **535 params**
- **No MR (pathwise_gpu):** 1 + 1 + 141 + 141 = **284 params**

#### Why pathwise_gpu Doesn't Have MR Sensitivities

The pathwise kernel tracks `dr/dr0` and `dr/dσ` but NOT `dr/dθ[i]` for each MR curve point:

```python
# From xva_pathwise_gpu.py kernel:
# State tracked per path:
dr_dr0, dr_dsigma         # ✓ Implemented
# dr_dtheta[0..n_mr]       # ✗ NOT implemented (would require O(n_mr) state per path)
```

Adding MR sensitivities to pathwise would require:
- ~251 additional derivative states per path
- ~251× more arithmetic per time step
- Significant memory increase (paths × pricing_times × 251 × 8 bytes)

The docstring in `xva_pathwise_gpu.py` claims MR support but the implementation doesn't include it (see line 801 comment: "excludes MR sensitivities").

#### Why gpu_brute_force Timings Are Not 535× Single Eval

The 0.85s sensitivity time for gpu_brute_force includes:
- r0, sigma: 2 full GPU re-simulations
- MR curve: 251 re-simulations (if not skipped) **BUT** using CUDA streams for concurrency
- Survival curves: 282 re-integrations (CPU-side, ~0.001s each)

With 16 CUDA streams, MR bumps run in batches of 16 concurrently. The actual GPU time is:
```
MR bumps: ceil(251 / 16) = 16 batches × ~0.03s/batch ≈ 0.5s
Survival re-integration: 282 × 0.001s ≈ 0.3s
Total: ~0.85s (matches observed)
```

### Production Benchmark: cpp_aadc at Full Capacity (16 threads)

**Configuration:** 50 trades, 10,000 paths, 16 threads

| Phase | Compilation | Execution | Total |
|-------|-------------|-----------|-------|
| Cold start | 5.7s | 10.5s | **16.7s** |
| Warm (kernel reused) | 0.0s | 10.5s | **10.5s** |

### Honest Comparison Table (Different Param Counts!)

| Backend | Eval Time | Sens Time | Market Greeks | MR Curve? | Method |
|---------|-----------|-----------|---------------|-----------|--------|
| pathwise_gpu | 0.13s | 0.13s* | ~284 | ✗ No | Pathwise AD |
| gpu_brute_force (no MR) | 0.14s | 0.85s | ~284 | ✗ Skipped | Finite diff + streams |
| gpu_brute_force (full) | 0.14s | ~2.5s | ~535 | ✓ Yes | Finite diff + streams |
| cpp_aadc (16T, warm) | 10.5s | 10.5s | ~535 | ✓ Yes | Full reverse-mode AD |

*pathwise_gpu computes sensitivities in the same kernel pass as primal (no additional time)

**Key insight:** cpp_aadc computes ~535 Greeks in 10.5s. pathwise_gpu computes ~284 in 0.13s.
- cpp_aadc is 13× slower but computes ~2× more sensitivities (including MR curve)
- Per-Greek efficiency: cpp_aadc = 535/10.5 = 51 Greeks/sec, pathwise_gpu = 284/0.13 = 2185 Greeks/sec

### Narrative for Publication

> "Reverse-mode AD on CPU (AADC, 16 threads) computes CVA plus **535 analytic market Greeks** including mean reversion curve sensitivities in **10.5s** with kernel reuse. GPU pathwise differentiation achieves **284 Greeks** (excluding MR curve) in **0.13s** — a **43× speedup per Greek computed**. For applications not requiring MR sensitivities, GPU pathwise offers the best throughput. For full risk management requiring all curve sensitivities, CPU AADC provides the complete solution."

### Getting Both: Possible Approaches

To achieve **both** kernel reuse **and** full sensitivities:

1. **GPU Pathwise Derivatives**: Compute sensitivities alongside primal in single pass (implemented in `xva_pathwise_gpu.py`)

2. **Hybrid AADC**: Record trade valuation kernels per trade type (like GPU modular), then chain with CSA kernel

3. **Analytical Sensitivities**: For survival curves, sensitivities can be computed analytically from exposures (no AD needed)

## CSV Log Model Names Explained

The benchmark logs results to `data/execution_log_xva.csv` with different model names. Understanding these names requires understanding the fundamental difference in how GPU and AADC cache kernels.

### GPU Model Names

GPU kernels are cached **per trade type** (based on cash flow structure):

| Model Name | Description |
|------------|-------------|
| `xva_modular_full_portfolio` | Initial run: compiles kernels for all trade types |
| `xva_modular_market_update` | Market data changed: all kernels reused (0 compiles) |
| `xva_modular_new_trade_existing_type` | New trade with known CF structure: kernel reused |
| `xva_modular_new_trade_new_type` | New trade with new CF structure: must compile new kernel |

**Why two "new trade" scenarios?**

GPU groups trades by **trade type** (e.g., IRS_5Y, IRS_10Y, FRA_3M). Each type has a unique cash flow structure that determines the kernel:

```
IRS_5Y:  20 quarterly payments → kernel with 20 CF iterations
IRS_10Y: 40 quarterly payments → kernel with 40 CF iterations (different kernel!)
```

When a new trade arrives:
- **Existing type** (e.g., another IRS_5Y): Reuse cached kernel → **0ms compile**
- **New type** (e.g., first IRS_15Y): Must compile new kernel → **305ms compile**

### AADC Model Names (Python Modular)

AADC kernel is cached **per pricing time grid** (independent of trades):

| Model Name | Description |
|------------|-------------|
| `xva_aadc_modular_full_portfolio` | Initial run: records kernel with V_portfolio inputs |
| `xva_aadc_modular_market_update` | Market data changed: kernel reused |
| `xva_aadc_modular_new_trade` | Any new trade: kernel always reused |

### C++ AADC Model Names (Monolithic)

The C++ AADC implementation uses a **monolithic kernel** (all trades unrolled):

| Model Name | Description |
|------------|-------------|
| `xva_cpp_aadc_full_portfolio` | Cold start: kernel compilation + evaluation |
| `xva_cpp_aadc_market_update` | Warm run: kernel reused, only evaluation |

**Note:** The C++ monolithic kernel benefits less from reuse (~10s savings) because:
- Kernel size is O(trades) - all trades unrolled into single kernel
- Large kernel = slower evaluation even when cached
- The modular architecture (Python) achieves better separation

**Why only one "new trade" scenario?**

AADC kernel takes **aggregated portfolio values** at each pricing time as inputs:

```python
# Kernel inputs: V_portfolio(t) for t in [0, 1, ..., 121]
# Kernel does NOT know about individual trades
inputs = {v_handles[t]: portfolio_values[:, t] for t in range(122)}
```

The kernel size is O(pricing_times), **not** O(trade_count) or O(trade_types). Whether the portfolio has 100 trades or 1000 trades, or whether the new trade is IRS_5Y or IRS_15Y, the kernel is the same:

```
New IRS_5Y trade:  recompute V_portfolio → feed to SAME kernel → 0ms compile
New IRS_15Y trade: recompute V_portfolio → feed to SAME kernel → 0ms compile
```

### Architecture Comparison

```
GPU Kernel Cache:                    AADC Kernel Cache:
┌────────────────────┐              ┌────────────────────┐
│ Trade Type → Kernel│              │ Grid → Kernel      │
├────────────────────┤              ├────────────────────┤
│ IRS_5Y   → K1      │              │ 122 pricing times  │
│ IRS_10Y  → K2      │              │        ↓           │
│ IRS_15Y  → K3      │              │   Single kernel    │
│ FRA_3M   → K4      │              │   (CSA + CVA)      │
│ ...      → ...     │              │                    │
└────────────────────┘              └────────────────────┘
     5-10 kernels                        1 kernel
```

| Aspect | GPU Modular | AADC Modular | C++ AADC Monolithic |
|--------|-------------|--------------|---------------------|
| Cache key | Trade type (CF structure) | Pricing time grid | All trades combined |
| Typical cache size | 5-10 kernels | 1 kernel | 1 kernel |
| New trade (existing type) | Reuse kernel | Reuse kernel | Recompile all |
| New trade (new type) | Compile new kernel | Reuse kernel | Recompile all |
| Kernel size | O(CF payments per type) | O(pricing times) | O(trades × CFs) |
| Market update reuse | Yes | Yes | Yes |

### When Does AADC Need Recompilation?

AADC only needs to recompile if the **pricing time grid changes**:
- Changing from 122 to 200 pricing times → recompile
- Adding more granular pricing dates → recompile
- Extending portfolio maturity beyond grid → recompile

In production, the grid is fixed, so AADC achieves **true O(1) kernel reuse** for all trade operations.

## Files

| File | Purpose |
|------|---------|
| `xva_modular_kernel.py` | GPU two-kernel implementation |
| `xva_aadc_modular.py` | AADC two-kernel implementation (SIMM-style) |
| `MODULAR_KERNEL_ARCHITECTURE.md` | This documentation |
