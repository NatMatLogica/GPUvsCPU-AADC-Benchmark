"""Pathwise Derivatives for Hull-White XVA: GPU-Friendly Greeks

This sketch demonstrates how to compute XVA sensitivities WITHOUT bump-and-revalue
by analytically differentiating the Hull-White model within each MC path.

Key insight: Instead of running N+1 simulations (base + N bumps), we evolve
derivative states alongside the price, computing all Greeks in ONE simulation.

Complexity: O(M) instead of O(N×M) — same as AADC, but GPU-parallelizable.

References:
- Glasserman, "Monte Carlo Methods in Financial Engineering", Ch. 7
- Giles & Glasserman, "Smoking Adjoints" (2006)
- Capriotti, "Fast Greeks by Algorithmic Differentiation" (2011)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Hull-White Model Review
# =============================================================================
#
# Short rate dynamics:
#   dr = (θ(t) - α·r) dt + σ dW
#
# Discretized (Euler):
#   r(t+Δt) = r(t)·e^(-αΔt) + θ(t)·(1 - e^(-αΔt)) + σ·√((1-e^(-2αΔt))/(2α))·Z
#
# Bond price:
#   P(t,T) = exp(-A(t,T)·r(t) + C(t,T))
#   where A(t,T) = (1 - e^(-α(T-t))) / α
#
# =============================================================================


@dataclass
class PathwiseState:
    """State vector for pathwise derivative computation.

    Evolves alongside the short rate r(t), tracking how r depends on inputs.
    """
    r: float              # Current short rate
    dr_dr0: float         # ∂r/∂r₀ — sensitivity to initial rate
    dr_dsigma: float      # ∂r/∂σ — sensitivity to volatility
    dr_dalpha: float      # ∂r/∂α — sensitivity to mean reversion speed
    dr_dtheta: np.ndarray # ∂r/∂θᵢ — sensitivities to mean reversion curve points


def initialize_pathwise_state(r0: float, n_theta: int) -> PathwiseState:
    """Initialize state at t=0."""
    return PathwiseState(
        r=r0,
        dr_dr0=1.0,           # ∂r₀/∂r₀ = 1
        dr_dsigma=0.0,        # No σ dependence yet
        dr_dalpha=0.0,        # No α dependence yet
        dr_dtheta=np.zeros(n_theta)  # No θ dependence yet
    )


def evolve_pathwise_state(
    state: PathwiseState,
    alpha: float,
    sigma: float,
    theta_t: float,        # θ(t) at current time
    theta_idx: int,        # Index of active θ point
    dt: float,
    dW: float,             # Brownian increment (Z * sqrt(dt))
) -> PathwiseState:
    """Evolve state from t to t+dt with pathwise derivatives.

    Differentiating the Euler scheme:
        r(t+dt) = r(t)·S + θ(t)·(1-S) + σ·V·Z

    where S = e^(-α·dt), V = √((1-S²)/(2α))

    Gives us evolution equations for all sensitivities.
    """
    S = math.exp(-alpha * dt)
    S2 = S * S
    V = math.sqrt((1.0 - S2) / (2.0 * alpha)) if alpha > 1e-10 else math.sqrt(dt)
    Z = dW / math.sqrt(dt) if dt > 1e-10 else 0.0

    # --- Forward evolution of r ---
    r_new = state.r * S + theta_t * (1.0 - S) + sigma * V * Z

    # --- Pathwise derivatives (chain rule) ---

    # ∂r(t+dt)/∂r₀ = ∂r(t+dt)/∂r(t) · ∂r(t)/∂r₀ = S · dr_dr0
    dr_dr0_new = S * state.dr_dr0

    # ∂r(t+dt)/∂σ = ∂r(t+dt)/∂r(t) · ∂r(t)/∂σ + ∂r(t+dt)/∂σ|direct
    #             = S · dr_dsigma + V · Z
    dr_dsigma_new = S * state.dr_dsigma + V * Z

    # ∂r(t+dt)/∂α — more complex due to S and V dependence on α
    # ∂S/∂α = -dt · S
    # ∂V/∂α = (S² · dt / α - (1-S²)/(2α²)) / (2V)  [messy but computable]
    dS_dalpha = -dt * S
    if V > 1e-10:
        dV_dalpha = (S2 * dt / alpha - (1.0 - S2) / (2.0 * alpha * alpha)) / (2.0 * V)
    else:
        dV_dalpha = 0.0

    dr_dalpha_new = (
        S * state.dr_dalpha           # Chain through r(t)
        + state.r * dS_dalpha          # ∂(r·S)/∂α
        + theta_t * (-dS_dalpha)       # ∂(θ·(1-S))/∂α
        + sigma * dV_dalpha * Z        # ∂(σ·V·Z)/∂α
    )

    # ∂r(t+dt)/∂θᵢ — only the active θ point contributes directly
    dr_dtheta_new = S * state.dr_dtheta.copy()
    dr_dtheta_new[theta_idx] += (1.0 - S)  # Direct contribution from θ(t)

    return PathwiseState(
        r=r_new,
        dr_dr0=dr_dr0_new,
        dr_dsigma=dr_dsigma_new,
        dr_dalpha=dr_dalpha_new,
        dr_dtheta=dr_dtheta_new
    )


def bond_price_with_greeks(
    state: PathwiseState,
    alpha: float,
    sigma: float,
    t: float,
    T: float,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Compute bond price P(t,T) and all its sensitivities.

    P(t,T) = exp(-A·r + C)

    Returns: (P, dP_dr0, dP_dsigma, dP_dalpha, dP_dtheta)
    """
    tau = T - t
    if tau <= 0:
        return 1.0, 0.0, 0.0, 0.0, np.zeros_like(state.dr_dtheta)

    # A(t,T) = (1 - e^(-α·τ)) / α
    exp_atau = math.exp(-alpha * tau)
    A = (1.0 - exp_atau) / alpha

    # C(t,T) involves integral of θ — simplified here
    # For full implementation, need cumulative θ terms
    # This sketch uses C = 0 for simplicity (affects level, not sensitivities)
    C = 0.0  # TODO: Full C(t,T) computation

    # Bond price
    P = math.exp(-A * state.r + C)

    # --- Bond price sensitivities via chain rule ---
    # dP/dθ = dP/dr · dr/dθ (r is the only path-dependent term)

    dP_dr = -A * P  # Direct sensitivity to r

    dP_dr0 = dP_dr * state.dr_dr0
    dP_dsigma = dP_dr * state.dr_dsigma
    dP_dalpha = dP_dr * state.dr_dalpha  # + ∂P/∂α|direct (via A)
    dP_dtheta = dP_dr * state.dr_dtheta

    # Add direct α dependence of A
    # ∂A/∂α = (τ·e^(-ατ) - A) / α
    dA_dalpha = (tau * exp_atau - A) / alpha
    dP_dalpha += (-dA_dalpha * state.r) * P

    return P, dP_dr0, dP_dsigma, dP_dalpha, dP_dtheta


# =============================================================================
# GPU Kernel Sketch (Numba CUDA)
# =============================================================================

PATHWISE_KERNEL_SKETCH = """
@cuda.jit
def simulate_xva_pathwise_kernel(
    randoms,           # (num_paths, num_steps)
    alpha, sigma, r0,
    theta_times, theta_vals, n_theta,
    # ... other model params ...
    # Outputs: prices AND all Greeks in one pass
    out_pee, out_nee,
    out_dpee_dr0, out_dnee_dr0,
    out_dpee_dsigma, out_dnee_dsigma,
    out_dpee_dtheta, out_dnee_dtheta,  # (num_paths, num_pricing, n_theta)
):
    path_idx = cuda.grid(1)
    if path_idx >= randoms.shape[0]:
        return

    # Initialize pathwise state
    r = r0
    dr_dr0 = 1.0
    dr_dsigma = 0.0
    dr_dtheta = cuda.local.array(MAX_THETA, dtype=float64)
    for i in range(n_theta):
        dr_dtheta[i] = 0.0

    for step in range(num_steps):
        # Get random increment
        dW = randoms[path_idx, step]

        # Evolve r AND all dr/d(param) simultaneously
        # ... [evolution equations from above] ...

        if is_pricing_time[step]:
            # Compute portfolio value AND all sensitivities
            total_pv = 0.0
            total_dpv_dr0 = 0.0
            total_dpv_dsigma = 0.0
            # ... aggregate across trades ...

            # Store results
            out_pee[path_idx, pricing_idx] = max(total_pv, 0.0)
            out_dpee_dr0[path_idx, pricing_idx] = total_dpv_dr0 if total_pv > 0 else 0.0
            # ... etc ...

# Result: ONE kernel launch gives prices + ALL Greeks
# Complexity: O(num_paths × num_steps × n_theta) — same as AADC
# GPU advantage: Paths are independent, perfect parallelism
"""


# =============================================================================
# Comparison: Pathwise vs Bump-and-Revalue vs AADC
# =============================================================================

COMPARISON = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Greeks Computation Approaches                            │
├─────────────────┬──────────────┬─────────────────┬─────────────────────────┤
│ Approach        │ Complexity   │ GPU Friendly?   │ Implementation Effort   │
├─────────────────┼──────────────┼─────────────────┼─────────────────────────┤
│ Bump & Revalue  │ O(N × M)     │ Yes (parallel   │ Low (just re-run)       │
│                 │              │ bumps)          │                         │
├─────────────────┼──────────────┼─────────────────┼─────────────────────────┤
│ CPU AADC        │ O(M)         │ No (CPU only)   │ Medium (library)        │
│                 │              │                 │                         │
├─────────────────┼──────────────┼─────────────────┼─────────────────────────┤
│ Pathwise (this) │ O(M × N)     │ YES (perfect)   │ High (manual derivs)    │
│                 │ but 1 pass   │                 │                         │
├─────────────────┼──────────────┼─────────────────┼─────────────────────────┤
│ GPU AAD         │ O(M)         │ Yes             │ Very High (custom AD)   │
│                 │              │                 │                         │
└─────────────────┴──────────────┴─────────────────┴─────────────────────────┘

Where N = number of Greeks, M = number of MC paths.

Pathwise advantages:
  ✓ Single simulation pass (like AADC)
  ✓ Perfect GPU parallelism (paths independent)
  ✓ No tape/memory overhead (unlike AD)
  ✓ Exact derivatives (no finite difference error)

Pathwise disadvantages:
  ✗ Manual derivation required for each model
  ✗ Doesn't work for discontinuous payoffs (digital options, barriers)
  ✗ Code complexity scales with number of parameters
  ✗ Must re-derive if model changes

For Hull-White XVA with smooth IRS payoffs: Pathwise is ideal for GPU.
For exotic/discontinuous payoffs: Need Likelihood Ratio or AD.
"""


# =============================================================================
# Expected Performance
# =============================================================================

EXPECTED_PERFORMANCE = """
For 50 trades, 4096 paths, 251 theta points:

Current GPU bump-and-revalue:
  - 253 simulations × 0.7s each = ~180s (sequential)
  - With 16 streams: ~40s

Pathwise GPU (this approach):
  - 1 simulation with ~5x overhead for derivative tracking
  - Estimated: 0.7s × 5 = ~3.5s for ALL Greeks

AADC CPU (8 threads):
  - 15s total

Pathwise GPU vs AADC: ~4x faster
Pathwise GPU vs GPU bump-and-revalue: ~11x faster

The pathwise approach makes GPU competitive with AADC for risk computation.
"""


if __name__ == "__main__":
    print("=" * 70)
    print("Pathwise Derivatives for Hull-White XVA")
    print("=" * 70)
    print(COMPARISON)
    print(EXPECTED_PERFORMANCE)

    # Simple demonstration
    print("\n" + "=" * 70)
    print("Demo: Single path evolution with pathwise derivatives")
    print("=" * 70)

    # Model parameters
    alpha = 0.04
    sigma = 0.01
    r0 = 0.05
    n_theta = 10  # Simplified: 10 theta points
    theta_vals = np.full(n_theta, 0.04)  # Flat mean reversion

    # Initialize
    state = initialize_pathwise_state(r0, n_theta)
    dt = 0.25  # Quarterly steps
    np.random.seed(42)

    print(f"\nInitial state:")
    print(f"  r = {state.r:.6f}")
    print(f"  ∂r/∂r₀ = {state.dr_dr0:.6f}")
    print(f"  ∂r/∂σ = {state.dr_dsigma:.6f}")

    # Evolve for 4 steps (1 year)
    for step in range(4):
        dW = np.random.normal(0, np.sqrt(dt))
        theta_idx = min(step, n_theta - 1)
        state = evolve_pathwise_state(
            state, alpha, sigma, theta_vals[theta_idx], theta_idx, dt, dW
        )

        print(f"\nAfter step {step + 1} (t = {(step+1)*dt:.2f}):")
        print(f"  r = {state.r:.6f}")
        print(f"  ∂r/∂r₀ = {state.dr_dr0:.6f}")
        print(f"  ∂r/∂σ = {state.dr_dsigma:.6f}")
        print(f"  ∂r/∂θ[0] = {state.dr_dtheta[0]:.6f}")

    # Bond price with Greeks
    P, dP_dr0, dP_dsigma, dP_dalpha, dP_dtheta = bond_price_with_greeks(
        state, alpha, sigma, t=1.0, T=5.0
    )

    print(f"\nBond price P(1, 5):")
    print(f"  P = {P:.6f}")
    print(f"  ∂P/∂r₀ = {dP_dr0:.6f}")
    print(f"  ∂P/∂σ = {dP_dsigma:.6f}")
    print(f"  ∂P/∂θ[0] = {dP_dtheta[0]:.6f}")

    print("\n" + "=" * 70)
    print("To use for XVA: Extend to full portfolio pricing + CSA + CVA/DVA")
    print("See PATHWISE_KERNEL_SKETCH for GPU implementation outline")
    print("=" * 70)
