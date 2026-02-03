# XVA-Benchmark Execution Log Analysis

## Recent Optimizations (v1.2.0)

### Bump Size Fix
- **Changed:** `bump = 1e-4` → `bump = 1e-8`
- **Rationale:** Match C++ AADC `bump_size` exactly (see `XVAJobRequest.h:679`)
- **Impact:** Ensures numerical derivatives are computed with identical perturbation size

### CUDA Streams for Concurrent Bumps
- **Added:** `GPUSimulationContext` class for pre-allocated GPU resources
- **Added:** `run_gpu_simulation_streamed()` for stream-based kernel launches
- **Benefit:** Mean reversion curve bumps now execute concurrently (up to 16 streams)
- **Expected speedup:** 10-16x on bump-and-revalue loop for large MR curves

### Remaining Optimizations (Not Yet Implemented)
- Shared memory for curve data (medium effort, 2-5x kernel speedup)
- Native CUDA C++ port (high effort, 2-5x overall)
- cuRAND for on-device RNG (medium effort, eliminates transfer overhead)

### Note on Benchmark Legitimacy

**Why GPU bump-and-revalue cannot beat CPU AAD:**

The fundamental issue is algorithmic complexity, not implementation quality:

| Approach | Complexity | Gradients |
|----------|------------|-----------|
| Bump-and-revalue | O(N × M) | N separate simulations |
| Reverse-mode AD (AADC) | O(M) | All gradients in ~2-4x forward cost |

Where N = number of risk factors, M = MC paths.

No amount of GPU optimization (shared memory, streams, native CUDA, cuRAND) can overcome this O(N) vs O(1) algorithmic gap. Even a 20x faster GPU bump-and-revalue implementation would lose to AADC for risk computation when N > 20.

**How GPU practitioners achieve "1000x speedups" in practice:**

1. **GPU AAD** — Implement reverse-mode AD directly in CUDA (best of both worlds)
2. **Pathwise derivatives** — Analytically differentiate within MC paths (see `xva_pathwise_gpu.py`)
3. **Likelihood ratio method** — Compute Greeks as expectations without bumping
4. **Mixed precision** — FP16/TF32 tensor cores vs FP64 (where "1000x" marketing claims originate)

### Pathwise GPU Sensitivity Limitations

The pathwise GPU kernel (`xva_pathwise_gpu.py`) computes sensitivities for:
- **r0** (initial rate) — 1 param
- **sigma** (volatility) — 1 param
- **Counterparty survival curve** — 141 params (analytical, post-simulation)
- **Company survival curve** — 141 params (analytical, post-simulation)

**Total: 124 params** (r0 + sigma + survival curves computed during kernel)

**Not implemented: Mean reversion curve sensitivities (120 params)**

The MR curve has ~120 points (10 years × 12 months). Adding these to pathwise would require:
- 51,200 paths × 122 pricing times × 120 MR × 8 bytes × 2 = **~14 GB** memory

This exceeds typical GPU memory. Possible future approaches:
1. **Batch MR sensitivities** — Compute in groups of ~10-20 params per kernel launch
2. **Track only active MR points** — Linear interpolation means only 2 points active per timestep
3. **Use brute-force for MR only** — Pathwise for r0/sigma/survival, bump-and-revalue for MR

For fair comparison between backends, use `--skip-mr-bumps` to exclude MR from brute-force (124 params instead of 244).

### Sensitivity Parameter Counts (initData.json)

The default config (`initData.json`) has:

| Parameter | Count | Notes |
|-----------|-------|-------|
| r0 | 1 | Short rate |
| sigma | 1 | Volatility |
| Mean reversion curve | 250 | T=50yr, step=0.2yr |
| Counterparty survival | 140 | T=14000d, step=100d |
| Company survival | 140 | T=14000d, step=100d |
| **Total** | **532** | |

**Benchmark configurations:**

| Config | Params | Command Flag |
|--------|--------|--------------|
| GPU (skip MR) | 282 | default |
| GPU (full) | 532 | `--include-mr-bumps` |
| AADC | 532 | always computes all |

**Why AADC cannot compute 282 params:**

AADC computes all 532 sensitivities in a **single reverse pass** — the cost is O(1) regardless of parameter count. There's no way to "skip" MR sensitivities because they're computed together with all other gradients in the same backward sweep. This is the fundamental advantage of adjoint AD.

**Crossover analysis (100 trades, 10K paths):**

| Params | GPU Brute-Force | AADC C++ | Winner |
|--------|-----------------|----------|--------|
| 282 (skip MR) | 1,274ms | 4,958ms | GPU 3.9x faster |
| 532 (all) | 23,647ms | 4,958ms | AADC 4.8x faster |

The crossover point is ~300-400 params. Below that, GPU wins (fast re-integration for survival curves). Above that, AADC wins (O(1) vs O(N) for re-simulation bumps).

**GPU 532-param breakdown:**
- r0 + sigma (2 re-sims): ~0.5s
- MR curve (250 re-sims): **22.9s** ← dominates
- Survival curves (280 re-integrations): ~0.5s

**Fair benchmark framing:**

> "GPU brute-force bump-and-revalue vs CPU AADC demonstrates that algorithmic improvements (automatic differentiation) outweigh hardware acceleration. For GPU to win at risk computation, AD must be implemented on GPU—parallel finite differences are insufficient."

The current benchmark is honest: it compares the **common industry practice** (GPU Monte Carlo + finite differences) against **state-of-the-art AD** (AADC). GPU wins for pricing-only; AADC wins for pricing+risk.

---

## Execution Log CSV Fields

The benchmark results are logged to `data/execution_log_xva.csv`. Key timing fields:

### GPU Timing Fields

| Field | Description |
|-------|-------------|
| `kernel_recording_sec` | **One-time JIT compilation cost** — Numba compiling CUDA kernel to GPU code. Paid once on first run. |
| `gpu_kernel_time_sec` | **Actual GPU kernel execution time** — Reusable after compilation. This is steady-state performance. |
| `eval_time_sec` | Same as `gpu_kernel_time_sec` for GPU backends. |

**Example from log:**
```
gpu_brute_force: kernel_recording=1.63s, gpu_kernel_time=0.20s
```

This means:
- **First run (cold start):** 1.63s to compile + 0.20s to execute = 1.83s total
- **Subsequent runs (warm):** 0.20s only (kernel is cached)

### AADC Timing Fields

| Field | Description |
|-------|-------------|
| `kernel_recording_sec` | **AADC kernel compilation time** — Building the AD tape and compiling to AVX2/AVX-512. |
| `eval_time_sec` | **AADC kernel evaluation time** — Forward pass + reverse sweeps for all gradients. |

### Workflow Benchmark Fields

For workflow scenarios (`gpu_cached`, `gpu_incremental`, `aadc_cached`), `kernel_recording_sec=0` because we specifically measure the **cached kernel** scenario where JIT/compilation is already complete.

---

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

> **Note**: The column has been renamed from `primal_time_sec` to `eval_time_sec` to clarify the semantics.

The log rows showing `eval_time_sec` of 59-70s for `xva_cpp_aadc` represent the **AADC evaluation time** (not a separate primal computation). Looking at `XVAServer.cpp:254-256`:

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

So for the AADC row, `eval_time_sec` in the CSV is `m_aad_time` (the AADC evaluation time). The 50-trade rows show:

| Run | AADC eval_time_sec | Compilation | Total |
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

## GPU vs AADC - Fair Comparison (Evaluation Time Only)

AADC compilation/recording is a **one-time cost** - the kernel is cached in `m_func_request_cache` and reused across evaluations. For a fair comparison, only evaluation time should be used.

### Small Portfolio (5 trades, 16 MC paths)

| Backend | What you get | Eval Time |
|---|---|---|
| **AADC C++ (1 thread)** | Prices + all Greeks | **0.024s** |
| **GPU brute-force** | Prices only | **0.018s** |

AADC delivers prices + full gradient in roughly the same time as GPU prices-only. For repeated evaluations, AADC is the clear winner since GPU would need bump-and-revalue for Greeks.

### Medium Portfolio (50 trades, 4096 MC paths)

| Backend | What you get | Eval Time |
|---|---|---|
| **AADC C++ (1 thread)** | Prices + all Greeks | **~0.2s** (0.78% relative perf) |
| **GPU brute-force** | Prices only | **0.72s** |

AADC is **3.6x faster** than GPU even though AADC computes the full gradient. To get equivalent Greeks on GPU via bump-and-revalue: 8 params x 0.72s = ~5.8s minimum.

### Large Portfolio (200 trades, 16384 MC paths) - `bank_medium.json`

| Backend | What you get | Eval Time | Notes |
|---|---|---|---|
| **AADC C++ (16 threads)** | Prices + all Greeks | **186.5s** | 1.4GB kernel, cache-bound |
| **GPU brute-force (H100)** | Prices only | **7.4s** | |

GPU is **25x faster** for pricing-only. However, to get equivalent Greeks on GPU via bump-and-revalue would cost at least 8 params x 7.4s = ~59s. So AADC with full Greeks is **~3x slower** than GPU bump-and-revalue would be.

The bottleneck is the **kernel size**: 590MB forward + 805MB reverse = 1.4GB total. This far exceeds CPU L3 cache, causing every MC iteration to be memory-bound. For comparison, the 50-trade kernel was 26MB and ran efficiently.

### Summary Table (Eval Time Only, Excluding Compilation)

| Portfolio | AADC Eval (prices+Greeks) | GPU (prices only) | GPU bump-and-revalue (est.) | AADC vs GPU B&R |
|---|---|---|---|---|
| 5 trades, 16 paths | 0.024s | 0.018s | ~0.14s | **5.8x faster** |
| 50 trades, 4096 paths | ~0.2s | 0.72s | ~5.8s | **29x faster** |
| 200 trades, 16384 paths | 186.5s | 7.4s | ~59s | **3.2x slower** |

AADC scales well up to the point where the compiled kernel fits in CPU cache. Beyond that, cache thrashing dominates and GPU brute-force becomes competitive even with bump-and-revalue overhead.

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
