"""
Experiment 3: Analytical reproduction of the infinite-horizon variants and
alternative model formulations from the paper's appendix.

Covers:
  1. Infinite-horizon market-making formulation with discount parameter omega.
  2. Admissibility condition: omega > 0.5 * gamma^2 * sigma^2 * q^2.
  3. Practical sufficient choice: omega = 0.5 * gamma^2 * sigma^2 * (q_max+1)^2.
  4. Appendix-style alternative: multiplicative diffusion dS/S = sigma dW.
  5. Mean/variance-type objective as alternative to CARA exponential utility.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Infinite-horizon formulation description
# ---------------------------------------------------------------------------

def infinite_horizon_description() -> str:
    """
    Return a description of the infinite-horizon market-making formulation.

    In the infinite-horizon setting, the finite terminal objective
        E[-exp(-gamma*(X_T + q_T*S_T))]
    is replaced by a discounted infinite-horizon objective.

    The stationarity of the problem removes the time-to-maturity factor (T-t)
    from the reservation price and spread formulas.

    Returns
    -------
    str
        Formatted description.
    """
    desc = (
        "Infinite-Horizon Market-Making Formulation\n"
        "============================================\n\n"
        "State variables and dynamics (same as finite-horizon):\n"
        "  dS_t = sigma * dW_t\n"
        "  dX_t = -p^b_t * dN^b_t + p^a_t * dN^a_t\n"
        "  dq_t =  dN^b_t - dN^a_t\n\n"
        "Infinite-horizon discounted objective:\n"
        "  V(s, x, q) = sup_{delta^b, delta^a} E[-exp(-gamma*(X_tau + q_tau*S_tau))]\n"
        "  where tau is an exponential stopping time with rate omega (discount).\n\n"
        "Stationary HJB equation (no explicit time dependence):\n"
        "  omega * u  =  (1/2)*sigma^2 * u_ss\n"
        "             +  sup_{delta^b} { lambda^b(delta^b) * [u(s, x-p^b, q+1) - u] }\n"
        "             +  sup_{delta^a} { lambda^a(delta^a) * [u(s, x+p^a, q-1) - u] }\n\n"
        "Key difference from finite-horizon:\n"
        "  - No terminal condition; replaced by discount rate omega.\n"
        "  - Time-to-maturity factor (T-t) is replaced by a stationary inventory\n"
        "    penalty term that depends on omega and gamma.\n"
        "  - Reservation price: r(s,q) = s - q*gamma*sigma^2 / (2*omega)\n"
        "    (heuristic; exact form depends on solution of stationary HJB)\n"
    )
    return desc


# ---------------------------------------------------------------------------
# 2. Admissibility condition
# ---------------------------------------------------------------------------

def admissibility_condition_description() -> str:
    """
    Return a description of the admissibility condition for the infinite-horizon
    formulation.

    Returns
    -------
    str
        Formatted description.
    """
    desc = (
        "Admissibility Condition for Infinite-Horizon Formulation\n"
        "==========================================================\n\n"
        "For the infinite-horizon value function to be well-defined and finite,\n"
        "the discount rate omega must dominate the variance of the inventory-weighted\n"
        "mid-price process.\n\n"
        "Required condition (paper):\n"
        "  omega > (1/2) * gamma^2 * sigma^2 * q^2\n"
        "  for all relevant inventory states q.\n\n"
        "Interpretation:\n"
        "  The term (1/2)*gamma^2*sigma^2*q^2 represents the rate at which the\n"
        "  exponential utility of the inventory position grows due to mid-price\n"
        "  diffusion. The discount rate omega must exceed this to ensure the\n"
        "  discounted value function remains bounded.\n\n"
        "Practical sufficient choice (bounded inventory |q| <= q_max):\n"
        "  omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2\n"
        "  This dominates (1/2)*gamma^2*sigma^2*q^2 for all |q| <= q_max,\n"
        "  since (q_max+1)^2 > q_max^2 >= q^2.\n"
    )
    return desc


def minimum_omega(
    gamma: float,
    sigma: float,
    q: float,
) -> float:
    """
    Compute the minimum discount rate omega required for admissibility at inventory q.

    Condition: omega > 0.5 * gamma^2 * sigma^2 * q^2

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    q : float
        Inventory level.

    Returns
    -------
    float
        Minimum omega (exclusive lower bound).
    """
    return 0.5 * gamma**2 * sigma**2 * q**2


def sufficient_omega(
    gamma: float,
    sigma: float,
    q_max: int,
) -> float:
    """
    Compute the sufficient discount rate omega for a bounded inventory domain |q| <= q_max.

    omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    q_max : int
        Maximum absolute inventory level.

    Returns
    -------
    float
        Sufficient omega value.
    """
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1) ** 2


def verify_admissibility(
    gamma: float,
    sigma: float,
    q_max: int,
) -> bool:
    """
    Verify that the sufficient omega dominates the admissibility condition
    for all q in {-q_max, ..., q_max}.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    q_max : int
        Maximum absolute inventory level.

    Returns
    -------
    bool
        True if sufficient omega dominates for all q in the grid.
    """
    omega_suff = sufficient_omega(gamma, sigma, q_max)
    q_grid = range(-q_max, q_max + 1)
    return all(omega_suff > minimum_omega(gamma, sigma, q) for q in q_grid)


def tabulate_omega_requirements(
    gamma_list: List[float],
    sigma: float,
    q_max_list: List[int],
) -> pd.DataFrame:
    """
    Tabulate minimum and sufficient omega values for combinations of gamma and q_max.

    Parameters
    ----------
    gamma_list : list[float]
        Risk-aversion values.
    sigma : float
        Mid-price volatility.
    q_max_list : list[int]
        Maximum inventory levels.

    Returns
    -------
    pd.DataFrame
        Table with columns: gamma, q_max, omega_min_at_qmax, omega_sufficient, valid.
    """
    rows = []
    for gamma in gamma_list:
        for q_max in q_max_list:
            omega_min  = minimum_omega(gamma, sigma, q_max)
            omega_suff = sufficient_omega(gamma, sigma, q_max)
            valid      = verify_admissibility(gamma, sigma, q_max)
            rows.append({
                "gamma":             gamma,
                "q_max":             q_max,
                "omega_min_at_qmax": omega_min,
                "omega_sufficient":  omega_suff,
                "admissible":        valid,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Infinite-horizon stationary quotes (heuristic)
# ---------------------------------------------------------------------------

def infinite_horizon_reservation_price(
    s: float,
    q: float,
    gamma: float,
    sigma: float,
    omega: float,
) -> float:
    """
    Compute the heuristic stationary reservation price for the infinite-horizon model.

    In the infinite-horizon setting, the time-to-maturity factor (T-t) is replaced
    by a stationary term. A natural heuristic (analogous to the finite-horizon formula)
    is:
        r_inf(s, q) = s - q * gamma * sigma^2 / (2 * omega)

    This is derived by noting that in the stationary HJB, the effective
    'time horizon' is 1/(2*omega) (the mean time to discounting).

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount rate.

    Returns
    -------
    float
        Stationary reservation price.
    """
    # Custom — Context7 found no library equivalent (paper infinite-horizon heuristic)
    return s - q * gamma * sigma**2 / (2.0 * omega)


def infinite_horizon_spread(
    gamma: float,
    sigma: float,
    omega: float,
    k: float,
) -> float:
    """
    Compute the heuristic stationary optimal spread for the infinite-horizon model.

    Analogous to the finite-horizon formula with (T-t) replaced by 1/(2*omega):
        psi_inf = gamma * sigma^2 / (2*omega) + (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount rate.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Stationary optimal spread.
    """
    inventory_risk_term = gamma * sigma**2 / (2.0 * omega)
    execution_term      = (2.0 / gamma) * np.log1p(gamma / k)
    return inventory_risk_term + execution_term


# ---------------------------------------------------------------------------
# 4. Appendix: multiplicative diffusion model
# ---------------------------------------------------------------------------

def appendix_multiplicative_model_description() -> str:
    """
    Return a description of the appendix-style alternative model with
    multiplicative mid-price diffusion.

    Returns
    -------
    str
        Formatted description.
    """
    desc = (
        "Appendix: Alternative Model with Multiplicative Diffusion\n"
        "===========================================================\n\n"
        "Alternative mid-price dynamics (geometric Brownian motion):\n"
        "  dS_u / S_u = sigma * dW_u\n"
        "  (multiplicative diffusion, unlike the additive dS_t = sigma*dW_t in main model)\n\n"
        "Implications:\n"
        "  - Mid-price remains positive by construction.\n"
        "  - Volatility scales with price level (proportional volatility).\n"
        "  - The HJB equation changes because the diffusion coefficient is S*sigma\n"
        "    rather than sigma.\n\n"
        "Alternative optimization criterion (mean/variance-type):\n"
        "  Instead of CARA expected utility E[-exp(-gamma*(X_T + q_T*S_T))],\n"
        "  the appendix considers a mean/variance objective:\n"
        "    max E[W_T] - (lambda/2) * Var[W_T]\n"
        "  where W_T = X_T + q_T*S_T is terminal marked-to-market wealth,\n"
        "  and lambda > 0 is a variance-aversion parameter.\n\n"
        "Key differences from main model:\n"
        "  1. Additive Brownian mid-price  vs  Multiplicative (GBM) mid-price.\n"
        "  2. CARA exponential utility     vs  Mean/variance objective.\n"
        "  3. Finite horizon T             vs  Infinite horizon with discount omega.\n\n"
        "Mean/variance optimal control:\n"
        "  The mean/variance objective leads to a linear-quadratic control problem.\n"
        "  The optimal quotes are derived from:\n"
        "    max_{delta^b, delta^a} { E[dW] - (lambda/2)*d(Var[W]) }\n"
        "  This yields quote adjustments proportional to inventory and variance,\n"
        "  qualitatively similar to the CARA model but with different coefficients.\n\n"
        "Note: These variants are NOT used in the paper's main numerical experiments.\n"
        "They are theoretical extensions studied independently.\n"
    )
    return desc


# ---------------------------------------------------------------------------
# 5. Mean/variance objective (analytical)
# ---------------------------------------------------------------------------

def mean_variance_reservation_price(
    s: float,
    q: float,
    t: float,
    lam: float,
    sigma: float,
    T: float,
) -> float:
    """
    Compute the reservation price under a mean/variance objective.

    Under the mean/variance criterion max E[W_T] - (lambda/2)*Var[W_T],
    the optimal inventory adjustment is analogous to the CARA case with
    gamma replaced by lambda (the variance-aversion parameter).

    Approximate formula (analogous to CARA):
        r_mv(s, q, t) = s - q * lambda * sigma^2 * (T - t)

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    lam : float
        Variance-aversion parameter (analogous to gamma in CARA).
    sigma : float
        Mid-price volatility.
    T : float
        Horizon.

    Returns
    -------
    float
        Mean/variance reservation price.
    """
    # Custom — Context7 found no library equivalent (paper appendix mean/variance)
    return s - q * lam * sigma**2 * (T - t)


def compare_cara_vs_mean_variance(
    sigma: float = 2.0,
    T: float = 1.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    t: float = 0.0,
) -> pd.DataFrame:
    """
    Compare reservation prices under CARA and mean/variance objectives.

    When lambda = gamma, the two objectives yield the same reservation price
    formula (to leading order), illustrating the connection between the models.

    Parameters
    ----------
    sigma : float
        Volatility.
    T : float
        Horizon.
    k : float
        Intensity decay.
    gamma_list : list[float]
        Risk-aversion / variance-aversion values.
    q_grid : list[int]
        Inventory grid.
    t : float
        Current time.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    if gamma_list is None:
        gamma_list = [0.01, 0.1, 0.5]
    if q_grid is None:
        q_grid = list(range(-5, 6))

    from src.exp1_analytical import reservation_price

    s0 = 100.0
    rows = []
    for gamma in gamma_list:
        for q in q_grid:
            r_cara = reservation_price(s0, q, t, gamma, sigma, T)
            r_mv   = mean_variance_reservation_price(s0, q, t, gamma, sigma, T)
            rows.append({
                "gamma":  gamma,
                "q":      q,
                "r_CARA": r_cara,
                "r_MV":   r_mv,
                "diff":   r_cara - r_mv,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Plotting functions
# ---------------------------------------------------------------------------

def plot_omega_requirements(
    df_omega: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the sufficient omega versus q_max for different gamma values.

    Parameters
    ----------
    df_omega : pd.DataFrame
        Output of tabulate_omega_requirements.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma, grp in df_omega.groupby("gamma"):
        ax.plot(grp["q_max"], grp["omega_sufficient"], marker="o",
                label=f"γ={gamma}")
        ax.plot(grp["q_max"], grp["omega_min_at_qmax"], linestyle="--",
                alpha=0.5)

    ax.set_xlabel("Maximum Inventory q_max", fontsize=12)
    ax.set_ylabel("Discount Rate ω", fontsize=12)
    ax.set_title(
        "Sufficient ω (solid) vs Minimum ω at q_max (dashed)\n"
        "for Infinite-Horizon Admissibility",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_infinite_horizon_quotes(
    sigma: float = 2.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot infinite-horizon reservation prices and spreads versus inventory.

    Parameters
    ----------
    sigma : float
        Volatility.
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for gamma in gamma_list:
        # Use sufficient omega for q_max=5
        omega = sufficient_omega(gamma, sigma, q_max=5)
        r_vals = [infinite_horizon_reservation_price(s0, q, gamma, sigma, omega)
                  for q in q_grid]
        ax.plot(q_grid, r_vals, marker="o", label=f"γ={gamma}")
    ax.axhline(s0, color="black", linestyle="--", linewidth=0.8, label="Mid S")
    ax.set_xlabel("Inventory q", fontsize=11)
    ax.set_ylabel("Reservation Price", fontsize=11)
    ax.set_title("Infinite-Horizon Reservation Price vs Inventory", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    omega_grid = np.linspace(0.01, 2.0, 100)
    for gamma in gamma_list:
        spreads = [infinite_horizon_spread(gamma, sigma, omega, k)
                   for omega in omega_grid]
        ax.plot(omega_grid, spreads, label=f"γ={gamma}")
    ax.set_xlabel("Discount Rate ω", fontsize=11)
    ax.set_ylabel("Stationary Spread ψ_inf", fontsize=11)
    ax.set_title("Infinite-Horizon Spread vs Discount Rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    sigma: float = 2.0,
    T: float = 1.0,
    k: float = 1.5,
    gamma_list: List[float] = None,
    q_grid: List[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare reservation prices across finite-horizon CARA, infinite-horizon,
    and mean/variance models.

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

    from src.exp1_analytical import reservation_price

    s0 = 100.0
    t  = 0.0
    n_gamma = len(gamma_list)
    fig, axes = plt.subplots(1, n_gamma, figsize=(5 * n_gamma, 5), sharey=False)
    if n_gamma == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gamma_list):
        omega = sufficient_omega(gamma, sigma, q_max=5)

        r_cara = [reservation_price(s0, q, t, gamma, sigma, T) for q in q_grid]
        r_inf  = [infinite_horizon_reservation_price(s0, q, gamma, sigma, omega)
                  for q in q_grid]
        r_mv   = [mean_variance_reservation_price(s0, q, t, gamma, sigma, T)
                  for q in q_grid]

        ax.plot(q_grid, r_cara, "b-o",  markersize=4, label="Finite-horizon CARA")
        ax.plot(q_grid, r_inf,  "g-s",  markersize=4, label=f"Infinite-horizon (ω={omega:.3f})")
        ax.plot(q_grid, r_mv,   "r--^", markersize=4, label="Mean/Variance")
        ax.axhline(s0, color="black", linestyle=":", linewidth=0.8, label="Mid S")
        ax.set_xlabel("Inventory q", fontsize=11)
        ax.set_ylabel("Reservation Price", fontsize=11)
        ax.set_title(f"γ={gamma}", fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Reservation Price Comparison Across Model Variants\n(σ={sigma}, T={T}, k={k})",
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Main runner
# ---------------------------------------------------------------------------

def run_experiment_3(results_dir: str = "results") -> Dict:
    """
    Run Experiment 3: analytical reproduction of infinite-horizon variants
    and appendix formulations.

    Parameters
    ----------
    results_dir : str
        Directory to save figures and outputs.

    Returns
    -------
    dict
        Summary of analytical results and numerical checks.
    """
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sigma      = 2.0
    T          = 1.0
    k          = 1.5
    gamma_list = [0.01, 0.1, 0.5]
    q_max_list = [5, 10, 20]

    print("=" * 70)
    print("EXPERIMENT 3: Infinite-Horizon Variants and Appendix Formulations")
    print("=" * 70)

    # 1. Infinite-horizon description
    print("\n" + infinite_horizon_description())

    # 2. Admissibility condition
    print(admissibility_condition_description())

    # 3. Omega requirements table
    df_omega = tabulate_omega_requirements(gamma_list, sigma, q_max_list)
    print("Omega Requirements Table")
    print("-" * 60)
    print(df_omega.to_string(index=False, float_format="{:.6f}".format))
    print()

    # Verify all admissibility conditions hold
    for _, row in df_omega.iterrows():
        assert row["admissible"], (
            f"Admissibility violated for gamma={row['gamma']}, q_max={row['q_max']}"
        )
    print("All admissibility conditions verified: sufficient omega dominates for all q.\n")

    # 4. Appendix description
    print(appendix_multiplicative_model_description())

    # 5. CARA vs mean/variance comparison
    df_compare = compare_cara_vs_mean_variance(
        sigma=sigma, T=T, k=k, gamma_list=gamma_list,
        q_grid=list(range(-5, 6)), t=0.0,
    )
    print("CARA vs Mean/Variance Reservation Price Comparison (t=0)")
    print("-" * 60)
    # Show only a subset for readability
    subset = df_compare[df_compare["q"].isin([-5, -2, 0, 2, 5])]
    print(subset.to_string(index=False, float_format="{:.4f}".format))
    print("\nNote: CARA and MV formulas coincide when lambda=gamma (as expected).\n")

    # 6. Plots
    fig1 = plot_omega_requirements(
        df_omega,
        save_path=os.path.join(fig_dir, "exp3_omega_requirements.png"),
    )
    plt.close(fig1)

    fig2 = plot_infinite_horizon_quotes(
        sigma=sigma, k=k, gamma_list=gamma_list,
        save_path=os.path.join(fig_dir, "exp3_infinite_horizon_quotes.png"),
    )
    plt.close(fig2)

    fig3 = plot_model_comparison(
        sigma=sigma, T=T, k=k, gamma_list=gamma_list,
        save_path=os.path.join(fig_dir, "exp3_model_comparison.png"),
    )
    plt.close(fig3)

    print("Figures saved to:", fig_dir)

    # Summary
    summary = {
        "model_variants": [
            "finite-horizon CARA (main model)",
            "infinite-horizon discounted CARA",
            "appendix: multiplicative diffusion + mean/variance",
        ],
        "admissibility_table": df_omega.to_dict(orient="records"),
        "cara_vs_mv_comparison": df_compare.to_dict(orient="records"),
        "key_formulas": {
            "admissibility_condition":  "omega > 0.5 * gamma^2 * sigma^2 * q^2",
            "sufficient_omega":         "omega = 0.5 * gamma^2 * sigma^2 * (q_max+1)^2",
            "inf_horizon_res_price":    "r_inf(s,q) = s - q*gamma*sigma^2 / (2*omega)",
            "inf_horizon_spread":       "psi_inf = gamma*sigma^2/(2*omega) + (2/gamma)*ln(1+gamma/k)",
            "mv_res_price":             "r_mv(s,q,t) = s - q*lambda*sigma^2*(T-t)",
        },
    }
    return summary
