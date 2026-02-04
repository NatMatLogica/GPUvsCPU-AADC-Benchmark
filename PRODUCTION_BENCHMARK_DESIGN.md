# Production XVA Benchmark Design

## Motivation

Current benchmarks measure cold-start performance (kernel compilation + execution). In production:
- Kernels are pre-compiled and cached
- Market data updates happen frequently (reuse kernel)
- New trades arrive incrementally (potentially reuse kernel)

## Benchmark Scenarios

### Scenario 1: Market Data Update (Intraday Repricing)

**Setup:**
- Portfolio: 100 trades (pre-recorded kernel)
- Kernel already compiled and cached

**Benchmark:**
1. First run: Compile kernel (cold start) - measure compilation time
2. Second run: Update r0, sigma, curves - reuse kernel - measure execution time only
3. Compute sensitivities

**What this measures:** Real-world intraday repricing performance

### Scenario 2: New Trade Arrival (Incremental XVA)

**Setup:**
- Existing portfolio: 100 trades (pre-recorded kernel)
- New trade arrives: 1 additional trade

**Options:**

**Option A: Full Recomputation**
- Recompile kernel for 101 trades
- Run full XVA calculation
- Measures: Compilation overhead for portfolio changes

**Option B: Incremental Calculation** (if supported)
- Keep 100-trade kernel
- Compute new trade's contribution separately
- Combine results
- Measures: Incremental update efficiency

## Implementation Plan

### 1. Modify XVAServer.cpp

Add `--benchmark-mode production` flag:

```cpp
// Production benchmark mode
if (benchmark_mode == "production") {
    // Phase 1: Cold start (compile kernel)
    auto t1 = now();
    obj->run(data_in, data_out, threads_num, cancel);
    auto compile_time = now() - t1;

    // Phase 2: Warm run (reuse kernel, update market data)
    modify_market_data(data_in);  // Change r0, sigma
    auto t2 = now();
    obj->run(data_in, data_out, threads_num, cancel);
    auto warm_time = now() - t2;

    // Report both times
    log("cold_start", compile_time);
    log("warm_run", warm_time);
}
```

### 2. Modify benchmark_xva.py

Add `--mode production` that:
1. Runs GPU with JIT warmup (first kernel compile)
2. Runs GPU again with updated market data (reuse JIT)
3. Reports both times separately

### 3. New Columns in CSV Log

| Column | Description |
|--------|-------------|
| `cold_compile_sec` | Time to compile kernel (first run) |
| `warm_eval_sec` | Execution time with cached kernel |
| `market_data_update` | True if this was a market data update run |

## Expected Results

| Backend | Cold Start (100 trades) | Warm Run (market update) | Speedup |
|---------|------------------------|-------------------------|---------|
| AADC | ~170s (compile + exec) | ~160s (exec only) | 1.06x |
| GPU BF | ~10s (JIT + exec) | ~3s (exec only) | 3.3x |
| Pathwise | ~8s (JIT + exec) | ~5s (exec only) | 1.6x |

Note: AADC's compilation time (~15s) is small relative to execution (~160s), so warm runs don't help much. The real issue is the unrolled kernel size causing cache misses.

## Key Insight

For AADC, the benefit of kernel caching is limited because:
- Compilation: ~15s (9% of total)
- Execution: ~160s (91% of total)

The execution is slow because the 220MB reverse kernel doesn't fit in CPU cache.

For GPU, JIT caching matters more:
- JIT compilation: ~3s (30% of total)
- Execution: ~3s (30% of total)
- Sensitivity bumps: ~4s (40% of total)

## Files to Modify

1. `XVAServer.cpp` - Add production benchmark mode
2. `benchmark_xva.py` - Add `--mode production`
3. `run_benchmark.sh` - Add `-m production` flag
4. `xva_common.py` - Add new CSV columns
