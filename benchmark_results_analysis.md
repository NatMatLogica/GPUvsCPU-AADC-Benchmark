# XVA-Benchmark Execution Log Analysis

## How the C++ AADC Actually Works

Looking at `XVAJobRequest.h:778-836`, `processRequest()` does this:

1. **`primal()`** - runs the double baseline (only if `"Primal Is Requred": true`)
2. **Cache lookup** - checks if an AADC kernel already exists for this structure
3. **`compileAADFunction()`** - records the kernel (only on cache miss)
4. **`aADExecution()`** - runs the compiled AADC kernel (forward + reverse per MC path, multi-threaded with AVX vectorization)
5. **`bumpAndRevalue()`** - runs bump-and-revalue (only if configured)

The AADC path does **not** re-run the primal. It runs the compiled kernel directly. The primal is only there for validation/comparison.

## Small Portfolio (5 trades, 16 MC paths, 8 sensitivity params)

| Measurement | Value |
|---|---|
| **C++ primal (double)** | 0.129s for 16 paths |
| **AADC compilation** | 0.772s (one-time) |
| **AADC evaluation** (Fw+Rev(CVA)+Rev(DVA)) | **0.024s** for 16 paths |
| **AADC eval speedup vs primal** | **5.3x** |
| **Relative performance** | 18.75% (from `all_results.json`) |
| **GPU pricing-only** | 0.018s kernel time |

The `"Relative performance": 0.1875` in `all_results.json` means AADC evaluation time was **18.75% of the primal** time (normalized). So for the same 16 paths, AADC does forward + 2 reverses (CVA + DVA gradients) in ~1/5 the cost of a single double forward pass.

## Medium Portfolio (50 trades, 4096 MC paths) - `all_results1.json`

| Measurement | Value |
|---|---|
| **Primal time** | 25.83s (from `"primal time": 25827320` us) |
| **Relative performance** | **0.78%** |
| **Implied AADC eval time** | ~0.20s |
| **AADC compilation** | 1.87s |
| **Compilation/base iteration ratio** | 150x one iteration |

The **0.78% relative performance** means the full AADC pass (forward + 2 reverses for all 4096 paths) costs under 1% of what the double primal takes. That's effectively a **~128x speedup** for getting prices + all Greeks vs just prices.

## Large Portfolio - `all_results2.json`

| Measurement | Value |
|---|---|
| **Primal time** | 125.04s (from `"primal time": 125038088` us) |
| **Relative performance** | **0.70%** |
| **Implied AADC eval time** | ~0.88s |
| **AADC compilation** | 8.10s |
| **Code size forward** | 106.7 MB |
| **Compilation/base iteration ratio** | 134x one iteration |

The **0.70%** relative performance means ~**142x speedup** - AADC scales better with larger portfolios.

## What the 50-Trade Execution Log Rows Actually Show

The log rows showing `primal_time_sec` of 59-70s for `xva_cpp_aadc` are **mislabeled** in the CSV logging code. Looking at `XVAServer.cpp:254-256`:

```cpp
double primal_time_sec = obj->m_primal_is_required
    ? obj->m_base_time.count() * obj->m_norm_coeff / 1e6 : 0.0;
double aadc_time_sec = obj->m_aad_time.count() / 1e6;
```

Then at line 286:
```cpp
log_xva_csv(..., "xva_cpp_aadc", ...,
    aadc_time_sec, 0.0, aadc_time_sec + compilation_sec, ...);
```

So for the AADC row, `primal_time_sec` in the CSV is actually `m_aad_time` (the AADC evaluation time). The 50-trade rows show:

| Run | AADC eval (logged as "primal") | Compilation | Total |
|-----|------|-------------|-------|
| 1 | 69.8s | 7.4s | 77.2s |
| 2 | 66.7s | 5.9s | 72.6s |
| 3 | 70.9s | 6.7s | 77.6s |
| 4 | 59.7s | 7.0s | 66.7s |

These are the **AADC evaluation** times (not primal), but they're surprisingly large. The `all_results1.json` shows 0.78% relative performance for the medium portfolio, implying AADC eval should be ~0.2s. The discrepancy suggests these 50-trade log rows were run with a **different, much larger config** (365 model steps, 122 pricing times vs 286/96 for the small portfolio) - and possibly with `"Primal Is Requred": false`, meaning the `m_base_time` reference was zero and only AADC ran.

## GPU Scaling Test (5 trades, pricing_only)

| MC Paths | GPU Kernel Time | Total Time |
|----------|----------------|------------|
| 16 | 0.018s | 0.019s |
| 1,000 | 0.019s | 0.020s |
| 10,000 | 0.024s | 0.027s |
| 100,000 | 0.110s | 0.136s |

The GPU shows near-constant overhead up to ~10K paths, then scales linearly - characteristic of good GPU utilization.

## GPU vs AADC - The Real Comparison

| Scenario | GPU (pricing only) | AADC (pricing + all Greeks) |
|---|---|---|
| 5 trades, 16 paths | 0.018s | 0.024s eval + 0.77s compile |
| 50 trades, 4096 paths | 0.72s | ~0.2s eval + 1.87s compile |

For **repeated evaluations** (where the kernel is compiled once and reused via `m_func_request_cache`):
- AADC gives you **prices + full gradient** in roughly the same order as GPU gives you prices-only
- The 0.70-0.78% relative performance means AADC's (forward + 2 reverse) costs < 1% of a single primal - the kernel reuse is working as intended

## CVA/DVA Values Across Backends

### Small portfolio (5 trades, 16 MC paths)

| Backend | CVA | DVA |
|---|---|---|
| C++ primal | 0.01758 | -0.00440 |
| GPU brute-force | 0.01590 | -0.00540 |

The ~10% difference is due to different random number generators (C++ mt19937_64 vs NumPy).

### Medium portfolio (`all_results1.json`)

| Source | CVA | DVA |
|---|---|---|
| Primal | 0.09433 | -0.13082 |
| AADC | 0.09433 | -0.13082 |

AADC matches primal exactly - confirming kernel correctness.

### Large portfolio (`all_results2.json`)

| Source | CVA | DVA |
|---|---|---|
| Primal | 0.11319 | -0.09426 |
| AADC | 0.11319 | -0.09426 |

Again, exact match between AADC and primal.

## AADC Compiler Data Comparison

| Metric | Small (5 trades) | Medium | Large |
|---|---|---|---|
| Code size forward | 1.66 MB | 25.95 MB | 106.74 MB |
| Code size reverse | 3.23 MB | 0.08 MB | 0.40 MB |
| Compilation time | 0.77s | 1.87s | 8.10s |
| Const data size | 5,469 | 1,734,379 | 8,284,280 |
| Work array size | 2,890 | 6,733 | 12,144 |
| Stack size | 5,963 | 0 | 0 |
| CheckPoint size | 0 | 0 | 0 |

The forward code size grows roughly with portfolio complexity. The medium/large portfolios show zero stack size and much smaller reverse code, suggesting a different compilation strategy (checkpointing vs full taping).
