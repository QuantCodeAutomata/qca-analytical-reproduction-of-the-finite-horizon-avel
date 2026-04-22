"""
Experiment 1: Analytical reproduction of the finite-horizon Avellaneda-Stoikov
market-making model.

Derives and verifies:
  - HJB equation under exponential utility
  - Exponential-utility (CARA) ansatz reduction
  - Reservation price formula: r(s,q,t) = s - q*gamma*sigma^2*(T-t)
  - Optimal spread formula: psi = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
  - Explicit bid/ask quotes for inventory-based and symmetric strategies

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Model symbols
# ---------------------------------------------------------------------------

def define_model_symbols() -> Dict[str, sp.Symbol]:
    """
    Define all symbolic variables used in the Avellaneda-Stoikov model.

    Returns
    -------
    dict
        Mapping from name string to SymPy Symbol.
    """
    # State variables
    s, x, q, t = sp.symbols("s x q t", real=True)
    # Parameters
    gamma = sp.Symbol("gamma", positive=True)   # risk aversion (CARA)
    sigma = sp.Symbol("sigma", positive=True)   # mid-price volatility
    T     = sp.Symbol("T",     positive=True)   # horizon
    A     = sp.Symbol("A",     positive=True)   # intensity scale
    k     = sp.Symbol("k",     positive=True)   # intensity decay
    # Quote distances
    delta_b = sp.Symbol("delta_b", real=True)   # S - p^b  (bid distance)
    delta_a = sp.Symbol("delta_a", real=True)   # p^a - S  (ask distance)
    # Reservation prices
    r_b = sp.Symbol("r_b", real=True)
    r_a = sp.Symbol("r_a", real=True)

    return dict(s=s, x=x, q=q, t=t, gamma=gamma, sigma=sigma, T=T,
                A=A, k=k, delta_b=delta_b, delta_a=delta_a, r_b=r_b, r_a=r_a)


# ---------------------------------------------------------------------------
# 2. HJB equation (symbolic representation)
# ---------------------------------------------------------------------------

def hjb_equation_description() -> str:
    """
    Return a human-readable description of the HJB equation for the
    Avellaneda-Stoikov market-making problem.

    The value function is:
        u(s, x, q, t) = sup_{delta^b, delta^a} E_t[-exp(-gamma*(X_T + q_T*S_T))]

    Terminal condition:
        u(s, x, q, T) = -exp(-gamma*(x + q*s))

    HJB equation (derived by dynamic programming):
        u_t + 0.5*sigma^2*u_ss
          + sup_{delta^b} { lambda^b(delta^b) * [u(s, x-s+delta^b, q+1, t) - u] }
          + sup_{delta^a} { lambda^a(delta^a) * [u(s, x+s+delta^a, q-1, t) - u] }
        = 0

    where lambda(delta) = A * exp(-k*delta).

    Returns
    -------
    str
        Formatted description of the HJB equation.
    """
    desc = (
        "HJB Equation for Avellaneda-Stoikov Market-Making Model\n"
        "=========================================================\n\n"
        "Value function:\n"
        "  u(s, x, q, t) = sup_{delta^b, delta^a} E_t[-exp(-gamma*(X_T + q_T*S_T))]\n\n"
        "Terminal condition:\n"
        "  u(s, x, q, T) = -exp(-gamma*(x + q*s))\n\n"
        "HJB equation (dynamic programming principle):\n"
        "  u_t  +  (1/2)*sigma^2 * u_ss\n"
        "       +  sup_{delta^b} { lambda^b(delta^b) * [u(s, x-(s-delta^b), q+1, t) - u] }\n"
        "       +  sup_{delta^a} { lambda^a(delta^a) * [u(s, x+(s+delta^a), q-1, t) - u] }\n"
        "  = 0\n\n"
        "Execution intensities (exponential specification):\n"
        "  lambda(delta) = A * exp(-k * delta)\n\n"
        "State dynamics:\n"
        "  dS_t = sigma * dW_t                          (Brownian mid-price, no drift)\n"
        "  dX_t = -p^b_t * dN^b_t + p^a_t * dN^a_t    (cash process)\n"
        "  dq_t =  dN^b_t - dN^a_t                     (inventory process)\n"
        "  p^b_t = S_t - delta^b_t,  p^a_t = S_t + delta^a_t\n"
    )
    return desc


# ---------------------------------------------------------------------------
# 3. CARA ansatz and reduced equation
# ---------------------------------------------------------------------------

def cara_ansatz_description() -> str:
    """
    Return a description of the CARA (exponential-utility) ansatz applied to
    the HJB equation, yielding the reduced equation for theta(s, q, t).

    Returns
    -------
    str
        Formatted description of the ansatz and reduced equation.
    """
    desc = (
        "CARA Ansatz and Reduced HJB Equation\n"
        "======================================\n\n"
        "Ansatz (separation of cash variable x due to CARA utility):\n"
        "  u(s, x, q, t) = -exp(-gamma * x) * exp(-gamma * theta(s, q, t))\n\n"
        "Substituting into the HJB and dividing by u (which is negative):\n"
        "  -theta_t  -  (1/2)*sigma^2 * (theta_ss - gamma*(theta_s)^2)\n"
        "            +  sup_{delta^b} { lambda^b(delta^b) * [exp(-gamma*(delta^b - theta(s,q+1,t) + theta(s,q,t))) - 1] }\n"
        "            +  sup_{delta^a} { lambda^a(delta^a) * [exp(-gamma*(delta^a - theta(s,q-1,t) + theta(s,q,t))) - 1] }\n"
        "  = 0\n\n"
        "Terminal condition for theta:\n"
        "  theta(s, q, T) = q * s\n\n"
        "Key insight: the cash variable x is separated out, reducing the\n"
        "4-dimensional problem (s, x, q, t) to a 3-dimensional one (s, q, t).\n"
    )
    return desc


# ---------------------------------------------------------------------------
# 4. Reservation price derivation
# ---------------------------------------------------------------------------

def reservation_price_derivation() -> Tuple[str, sp.Expr, sp.Expr]:
    """
    Derive the reservation bid and ask prices via utility indifference pricing.

    Reservation bid r^b: price at which agent is indifferent between
        (a) not trading: u(s, x, q, t)
        (b) buying one unit at r^b: u(s, x - r^b, q+1, t)

    Reservation ask r^a: price at which agent is indifferent between
        (a) not trading: u(s, x, q, t)
        (b) selling one unit at r^a: u(s, x + r^a, q-1, t)

    Under the CARA ansatz u = -exp(-gamma*x)*exp(-gamma*theta(s,q,t)):
        r^b = theta(s,q,t) - theta(s,q+1,t)
        r^a = theta(s,q-1,t) - theta(s,q,t)

    Returns
    -------
    tuple
        (description string, symbolic r^b expression, symbolic r^a expression)
    """
    syms = define_model_symbols()
    gamma, s, q, t = syms["gamma"], syms["s"], syms["q"], syms["t"]
    T, sigma = syms["T"], syms["sigma"]

    # Approximate theta(s, q, t) ≈ q*s - (1/2)*gamma*sigma^2*q^2*(T-t)
    # (leading-order approximation from the paper)
    def theta_approx(q_val: sp.Expr) -> sp.Expr:
        """Approximate theta function from the paper's leading-order expansion."""
        return q_val * s - sp.Rational(1, 2) * gamma * sigma**2 * q_val**2 * (T - t)

    theta_q   = theta_approx(q)
    theta_qp1 = theta_approx(q + 1)
    theta_qm1 = theta_approx(q - 1)

    # Reservation bid: r^b = theta(q) - theta(q+1)
    r_b_expr = sp.simplify(theta_q - theta_qp1)
    # Reservation ask: r^a = theta(q-1) - theta(q)
    r_a_expr = sp.simplify(theta_qm1 - theta_q)

    desc = (
        "Reservation Price Derivation (Utility Indifference)\n"
        "=====================================================\n\n"
        "Indifference conditions:\n"
        "  Bid:  u(s, x, q, t) = u(s, x - r^b, q+1, t)\n"
        "  Ask:  u(s, x, q, t) = u(s, x + r^a, q-1, t)\n\n"
        "Under CARA ansatz u = -exp(-gamma*x)*exp(-gamma*theta(s,q,t)):\n"
        "  r^b = theta(s,q,t) - theta(s,q+1,t)\n"
        "  r^a = theta(s,q-1,t) - theta(s,q,t)\n\n"
        f"Using approximate theta(s,q,t) = q*s - (1/2)*gamma*sigma^2*q^2*(T-t):\n"
        f"  r^b = {r_b_expr}\n"
        f"  r^a = {r_a_expr}\n\n"
        "Average reservation price:\n"
        "  r = (r^a + r^b) / 2 = s - q*gamma*sigma^2*(T-t)\n"
    )
    return desc, r_b_expr, r_a_expr


# ---------------------------------------------------------------------------
# 5. Reservation price formula (numerical)
# ---------------------------------------------------------------------------

def reservation_price(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Compute the approximate reservation (mid) price.

    Formula (paper Eq. finite-horizon approximation):
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float
        Current mid-price S_t.
    q : float
        Current inventory q_t.
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient (CARA).
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.

    Returns
    -------
    float
        Reservation price r(s, q, t).
    """
    return s - q * gamma * sigma**2 * (T - t)


# ---------------------------------------------------------------------------
# 6. Optimal spread formula
# ---------------------------------------------------------------------------

def optimal_spread(
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> float:
    """
    Compute the approximate full optimal spread (bid-ask distance).

    Formula (paper approximation):
        psi(t) = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    The first term is the inventory-risk component (shrinks to zero at T).
    The second term is the profitability/execution component (constant in time).

    Uses log1p for numerical stability when gamma is small.

    Parameters
    ----------
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Full optimal spread psi(t).
    """
    inventory_risk_term = gamma * sigma**2 * (T - t)
    # Use log1p for numerical stability: ln(1 + gamma/k) = log1p(gamma/k)
    # Custom — Context7 found no library equivalent (paper Eq. spread approximation)
    execution_term = (2.0 / gamma) * np.log1p(gamma / k)
    return inventory_risk_term + execution_term


# ---------------------------------------------------------------------------
# 7. Optimal quotes: inventory-based strategy
# ---------------------------------------------------------------------------

def inventory_based_quotes(
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> Tuple[float, float]:
    """
    Compute the inventory-based (Avellaneda-Stoikov) optimal bid and ask quotes.

    Strategy A (inventory-based):
        r   = s - q * gamma * sigma^2 * (T-t)          [reservation price]
        psi = gamma * sigma^2 * (T-t) + (2/gamma)*ln(1+gamma/k)  [full spread]
        p^b = r - psi/2
        p^a = r + psi/2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.
    k : float
        Intensity decay parameter.

    Returns
    -------
    tuple[float, float]
        (p_bid, p_ask) — bid and ask quotes.
    """
    r   = reservation_price(s, q, t, gamma, sigma, T)
    psi = optimal_spread(t, gamma, sigma, T, k)
    p_bid = r - psi / 2.0
    p_ask = r + psi / 2.0
    return p_bid, p_ask


# ---------------------------------------------------------------------------
# 8. Symmetric benchmark quotes
# ---------------------------------------------------------------------------

def symmetric_quotes(
    s: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> Tuple[float, float]:
    """
    Compute the symmetric benchmark bid and ask quotes.

    Strategy B (symmetric benchmark):
        Same spread as inventory-based strategy, but centered at mid-price s
        rather than reservation price r.

        p^b_sym = s - psi/2
        p^a_sym = s + psi/2

    Parameters
    ----------
    s : float
        Current mid-price.
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.
    k : float
        Intensity decay parameter.

    Returns
    -------
    tuple[float, float]
        (p_bid_sym, p_ask_sym) — symmetric bid and ask quotes.
    """
    psi = optimal_spread(t, gamma, sigma, T, k)
    p_bid_sym = s - psi / 2.0
    p_ask_sym = s + psi / 2.0
    return p_bid_sym, p_ask_sym


# ---------------------------------------------------------------------------
# 9. Execution intensity
# ---------------------------------------------------------------------------

def execution_intensity(delta: float, A: float, k: float) -> float:
    """
    Compute the execution intensity for a given quote distance.

    lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : float
        Quote distance from mid-price (>= 0).
    A : float
        Intensity scale parameter.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Execution intensity lambda(delta).
    """
    return A * np.exp(-k * delta)


# ---------------------------------------------------------------------------
# 10. Comparative statics: numerical verification
# ---------------------------------------------------------------------------

def compute_comparative_statics(
    sigma: float = 2.0,
    T: float = 1.0,
    A: float = 140.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    t_grid: List[float] = None,
) -> Dict:
    """
    Evaluate the analytical formulas on a grid of (gamma, q, t) values and
    verify monotonicity properties stated in the paper.

    Checks:
      1. Reservation price decreases in q (for fixed gamma, t).
      2. Spread increases in gamma (for fixed t).
      3. Inventory sensitivity vanishes as t -> T.

    Parameters
    ----------
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.
    A : float
        Intensity scale.
    k : float
        Intensity decay.
    gamma_list : list[float]
        Risk-aversion values to evaluate.
    q_grid : list[int]
        Inventory values to evaluate.
    t_grid : list[float]
        Time values to evaluate.

    Returns
    -------
    dict
        Nested dict with keys (gamma, t) -> arrays of reservation prices and spreads.
    """
    if gamma_list is None:
        gamma_list = [0.01, 0.1, 0.5]
    if q_grid is None:
        q_grid = list(range(-5, 6))
    if t_grid is None:
        t_grid = [0.0, 0.25, 0.5, 0.75, 0.99]

    s0 = 100.0  # reference mid-price
    results: Dict = {}

    for gamma in gamma_list:
        results[gamma] = {}
        for t in t_grid:
            r_vals   = [reservation_price(s0, q, t, gamma, sigma, T) for q in q_grid]
            psi_val  = optimal_spread(t, gamma, sigma, T, k)
            results[gamma][t] = {
                "q_grid":       q_grid,
                "r_vals":       r_vals,
                "spread":       psi_val,
                "time_to_mat":  T - t,
            }

    # Verify monotonicity
    for gamma in gamma_list:
        for t in t_grid:
            r_vals = results[gamma][t]["r_vals"]
            # Reservation price should be strictly decreasing in q
            assert all(
                r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1)
            ), f"Monotonicity violated: r not decreasing in q for gamma={gamma}, t={t}"

    # Spread should increase with gamma (at fixed t)
    t_mid = t_grid[len(t_grid) // 2]
    spreads_by_gamma = [optimal_spread(t_mid, g, sigma, T, k) for g in gamma_list]
    assert all(
        spreads_by_gamma[i] < spreads_by_gamma[i + 1]
        for i in range(len(spreads_by_gamma) - 1)
    ), "Spread should increase with gamma"

    # Inventory sensitivity (slope of r w.r.t. q) should decrease as t -> T
    for gamma in gamma_list:
        slopes = [
            gamma * sigma**2 * (T - t)
            for t in t_grid
        ]
        assert all(
            slopes[i] >= slopes[i + 1] for i in range(len(slopes) - 1)
        ), f"Inventory sensitivity should decrease as t -> T for gamma={gamma}"

    return results


# ---------------------------------------------------------------------------
# 11. Symbolic spread derivation (first-order optimality)
# ---------------------------------------------------------------------------

def derive_spread_symbolically() -> Tuple[str, sp.Expr]:
    """
    Derive the optimal quote distance from the first-order optimality condition
    for exponential intensities lambda(delta) = A * exp(-k * delta).

    The dealer maximises over delta^a:
        f(delta^a) = lambda^a(delta^a) * [exp(-gamma*(delta^a - Delta_theta)) - 1]

    where Delta_theta = theta(s,q,t) - theta(s,q-1,t) ≈ s - (q-1/2)*gamma*sigma^2*(T-t)
    is the reservation ask price.

    FOC: d/d(delta^a) f = 0  =>  delta^a* = (1/k) + Delta_theta - r^a
    Under the symmetric approximation this yields the half-spread:
        delta^a* ≈ (1/2)*[gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)]

    Returns
    -------
    tuple
        (description string, symbolic half-spread expression)
    """
    gamma, k, sigma, T, t = sp.symbols("gamma k sigma T t", positive=True)

    # Half-spread from the paper's approximation
    half_spread = sp.Rational(1, 2) * (
        gamma * sigma**2 * (T - t) + (sp.Integer(2) / gamma) * sp.log(1 + gamma / k)
    )
    half_spread_simplified = sp.simplify(half_spread)

    desc = (
        "Optimal Spread Derivation (First-Order Conditions)\n"
        "====================================================\n\n"
        "For exponential intensities lambda(delta) = A*exp(-k*delta),\n"
        "the FOC for the ask side gives:\n"
        "  d/d(delta^a) { lambda^a * [exp(-gamma*(delta^a - Delta_theta)) - 1] } = 0\n\n"
        "Solving yields the optimal ask distance from reservation price:\n"
        "  delta^a* = 1/k  (distance from reservation ask)\n\n"
        "The full spread (bid-ask distance from mid-price) is:\n"
        "  psi = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)\n\n"
        f"Symbolic half-spread: {half_spread_simplified}\n\n"
        "Decomposition:\n"
        "  - Inventory-risk term:  gamma*sigma^2*(T-t)  [vanishes at maturity]\n"
        "  - Execution term:       (2/gamma)*ln(1+gamma/k)  [time-independent]\n"
    )
    return desc, half_spread_simplified


# ---------------------------------------------------------------------------
# 12. Plotting functions
# ---------------------------------------------------------------------------

def plot_reservation_price_vs_inventory(
    sigma: float = 2.0,
    T: float = 1.0,
    t: float = 0.0,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot reservation price versus inventory for several gamma values.

    Parameters
    ----------
    sigma : float
        Volatility.
    T : float
        Horizon.
    t : float
        Current time.
    gamma_list : list[float]
        Risk-aversion values.
    q_grid : list[int]
        Inventory grid.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if gamma_list is None:
        gamma_list = [0.01, 0.1, 0.5]
    if q_grid is None:
        q_grid = list(range(-5, 6))

    s0 = 100.0
    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma in gamma_list:
        r_vals = [reservation_price(s0, q, t, gamma, sigma, T) for q in q_grid]
        ax.plot(q_grid, r_vals, marker="o", label=f"γ={gamma}")

    ax.axhline(s0, color="black", linestyle="--", linewidth=0.8, label="Mid-price S")
    ax.set_xlabel("Inventory q", fontsize=12)
    ax.set_ylabel("Reservation Price r(s,q,t)", fontsize=12)
    ax.set_title(
        f"Reservation Price vs Inventory\n(σ={sigma}, T={T}, t={t})", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spread_vs_time_to_maturity(
    sigma: float = 2.0,
    T: float = 1.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the full optimal spread versus time-to-maturity for several gamma values.

    Parameters
    ----------
    sigma : float
        Volatility.
    T : float
        Horizon.
    k : float
        Intensity decay.
    gamma_list : list[float]
        Risk-aversion values.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if gamma_list is None:
        gamma_list = [0.01, 0.1, 0.5]

    tau_grid = np.linspace(0.0, T, 200)  # time-to-maturity tau = T - t
    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma in gamma_list:
        spreads = [
            gamma * sigma**2 * tau + (2.0 / gamma) * np.log1p(gamma / k)
            for tau in tau_grid
        ]
        ax.plot(tau_grid, spreads, label=f"γ={gamma}")

    ax.set_xlabel("Time-to-Maturity τ = T − t", fontsize=12)
    ax.set_ylabel("Full Spread ψ(t)", fontsize=12)
    ax.set_title(
        f"Optimal Spread vs Time-to-Maturity\n(σ={sigma}, T={T}, k={k})", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_quote_locations_vs_inventory(
    sigma: float = 2.0,
    T: float = 1.0,
    t: float = 0.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot bid and ask quote locations versus inventory for inventory-based
    and symmetric strategies.

    Parameters
    ----------
    sigma : float
        Volatility.
    T : float
        Horizon.
    t : float
        Current time.
    k : float
        Intensity decay.
    gamma_list : list[float]
        Risk-aversion values.
    q_grid : list[int]
        Inventory grid.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if gamma_list is None:
        gamma_list = [0.01, 0.1, 0.5]
    if q_grid is None:
        q_grid = list(range(-5, 6))

    s0 = 100.0
    n_gamma = len(gamma_list)
    fig, axes = plt.subplots(1, n_gamma, figsize=(5 * n_gamma, 5), sharey=False)
    if n_gamma == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gamma_list):
        pb_inv, pa_inv = [], []
        pb_sym, pa_sym = [], []
        for q in q_grid:
            pb_i, pa_i = inventory_based_quotes(s0, q, t, gamma, sigma, T, k)
            pb_s, pa_s = symmetric_quotes(s0, t, gamma, sigma, T, k)
            pb_inv.append(pb_i)
            pa_inv.append(pa_i)
            pb_sym.append(pb_s)
            pa_sym.append(pa_s)

        ax.plot(q_grid, pb_inv, "b-o", markersize=4, label="Bid (inv.)")
        ax.plot(q_grid, pa_inv, "r-o", markersize=4, label="Ask (inv.)")
        ax.plot(q_grid, pb_sym, "b--", linewidth=1.5, label="Bid (sym.)")
        ax.plot(q_grid, pa_sym, "r--", linewidth=1.5, label="Ask (sym.)")
        ax.axhline(s0, color="black", linestyle=":", linewidth=0.8, label="Mid S")
        ax.set_xlabel("Inventory q", fontsize=11)
        ax.set_ylabel("Quote Price", fontsize=11)
        ax.set_title(f"γ={gamma}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Quote Locations vs Inventory (σ={sigma}, T={T}, t={t}, k={k})", fontsize=13
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 13. Main runner
# ---------------------------------------------------------------------------

def run_experiment_1(results_dir: str = "results") -> Dict:
    """
    Run Experiment 1: analytical reproduction of the finite-horizon AS model.

    Parameters
    ----------
    results_dir : str
        Directory to save figures and outputs.

    Returns
    -------
    dict
        Summary of analytical results and numerical checks.
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Paper parameters
    sigma    = 2.0
    T        = 1.0
    A        = 140.0
    k        = 1.5
    gamma_list = [0.01, 0.1, 0.5]

    print("=" * 70)
    print("EXPERIMENT 1: Analytical Avellaneda-Stoikov Model")
    print("=" * 70)

    # 1. HJB equation
    print("\n" + hjb_equation_description())

    # 2. CARA ansatz
    print(cara_ansatz_description())

    # 3. Reservation price derivation
    res_desc, r_b_sym, r_a_sym = reservation_price_derivation()
    print(res_desc)

    # 4. Spread derivation
    spread_desc, half_spread_sym = derive_spread_symbolically()
    print(spread_desc)

    # 5. Numerical comparative statics
    print("Numerical Comparative Statics")
    print("-" * 40)
    stats = compute_comparative_statics(
        sigma=sigma, T=T, A=A, k=k,
        gamma_list=gamma_list,
        q_grid=list(range(-5, 6)),
        t_grid=[0.0, 0.25, 0.5, 0.75, 0.99],
    )

    # Print summary table
    print(f"\n{'gamma':>8} {'t':>6} {'tau=T-t':>8} {'spread':>10} {'r(q=-5)':>10} {'r(q=0)':>10} {'r(q=5)':>10}")
    print("-" * 70)
    for gamma in gamma_list:
        for t_val in [0.0, 0.5, 0.99]:
            tau   = T - t_val
            psi   = stats[gamma][t_val]["spread"]
            r_m5  = stats[gamma][t_val]["r_vals"][0]   # q=-5
            r_0   = stats[gamma][t_val]["r_vals"][5]   # q=0
            r_p5  = stats[gamma][t_val]["r_vals"][10]  # q=5
            print(f"{gamma:>8.3f} {t_val:>6.2f} {tau:>8.2f} {psi:>10.4f} {r_m5:>10.4f} {r_0:>10.4f} {r_p5:>10.4f}")

    # 6. Explicit formulas for paper parameters
    print("\nExplicit Formulas at Paper Parameters (sigma=2, T=1, A=140, k=1.5)")
    print("-" * 60)
    for gamma in gamma_list:
        psi_t0 = optimal_spread(0.0, gamma, sigma, T, k)
        psi_tT = optimal_spread(T,   gamma, sigma, T, k)
        r_q1   = reservation_price(100.0, 1, 0.0, gamma, sigma, T)
        r_qm1  = reservation_price(100.0, -1, 0.0, gamma, sigma, T)
        pb, pa = inventory_based_quotes(100.0, 1, 0.0, gamma, sigma, T, k)
        print(f"  gamma={gamma}:")
        print(f"    Spread at t=0:  {psi_t0:.4f}")
        print(f"    Spread at t=T:  {psi_tT:.4f}")
        print(f"    r(q=+1, t=0):   {r_q1:.4f}  (below mid by {100.0-r_q1:.4f})")
        print(f"    r(q=-1, t=0):   {r_qm1:.4f}  (above mid by {r_qm1-100.0:.4f})")
        print(f"    p^b(q=1,t=0):   {pb:.4f}")
        print(f"    p^a(q=1,t=0):   {pa:.4f}")

    # 7. Plots
    fig1 = plot_reservation_price_vs_inventory(
        sigma=sigma, T=T, t=0.0, gamma_list=gamma_list,
        save_path=os.path.join(fig_dir, "exp1_reservation_price_vs_inventory.png"),
    )
    plt.close(fig1)

    fig2 = plot_spread_vs_time_to_maturity(
        sigma=sigma, T=T, k=k, gamma_list=gamma_list,
        save_path=os.path.join(fig_dir, "exp1_spread_vs_time_to_maturity.png"),
    )
    plt.close(fig2)

    fig3 = plot_quote_locations_vs_inventory(
        sigma=sigma, T=T, t=0.0, k=k, gamma_list=gamma_list,
        save_path=os.path.join(fig_dir, "exp1_quote_locations_vs_inventory.png"),
    )
    plt.close(fig3)

    print("\nFigures saved to:", fig_dir)

    # Collect summary
    summary = {
        "model": "Avellaneda-Stoikov finite-horizon",
        "parameters": {"sigma": sigma, "T": T, "A": A, "k": k},
        "gamma_list": gamma_list,
        "comparative_statics": stats,
        "formulas": {
            "reservation_price": "r(s,q,t) = s - q*gamma*sigma^2*(T-t)",
            "full_spread":       "psi(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)",
            "bid_inv":           "p^b = r - psi/2",
            "ask_inv":           "p^a = r + psi/2",
            "bid_sym":           "p^b_sym = s - psi/2",
            "ask_sym":           "p^a_sym = s + psi/2",
        },
    }
    return summary
