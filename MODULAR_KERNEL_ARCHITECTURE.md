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

## Files

| File | Purpose |
|------|---------|
| `xva_modular_kernel.py` | Two-kernel implementation with benchmarks |
| `MODULAR_KERNEL_ARCHITECTURE.md` | This documentation |
