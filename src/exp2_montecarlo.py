"""
Experiment 2: Monte Carlo reproduction of the paper's main numerical comparison.

Simulates Brownian mid-price paths and stochastic order executions under
exponential fill intensities, comparing:
  - Strategy A: Inventory-based Avellaneda-Stoikov approximate quoting
  - Strategy B: Symmetric benchmark (same spread, centered at mid-price)

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional

from src.exp1_analytical import (
    reservation_price,
    optimal_spread,
    inventory_based_quotes,
    symmetric_quotes,
)


# ---------------------------------------------------------------------------
# 1. Simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "S0":     100.0,   # initial mid-price
    "T":      1.0,     # horizon
    "dt":     0.005,   # time step
    "N":      200,     # number of steps (T/dt)
    "sigma":  2.0,     # mid-price volatility
    "A":      140.0,   # intensity scale
    "k":      1.5,     # intensity decay
    "q0":     0,       # initial inventory
    "X0":     0.0,     # initial cash
    "n_sim":  1000,    # number of Monte Carlo paths
}


# ---------------------------------------------------------------------------
# 2. Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    strategy: str,
    gamma: float,
    params: Dict,
    rng: np.random.Generator,
    store_full: bool = False,
) -> Dict:
    """
    Simulate one Monte Carlo path for a given strategy.

    Parameters
    ----------
    strategy : str
        "inventory" for Strategy A (Avellaneda-Stoikov) or
        "symmetric" for Strategy B (symmetric benchmark).
    gamma : float
        Risk-aversion coefficient.
    params : dict
        Simulation parameters (S0, T, dt, N, sigma, A, k, q0, X0).
    rng : np.random.Generator
        Random number generator for reproducibility.
    store_full : bool
        If True, store full time series of state variables.

    Returns
    -------
    dict
        Terminal outcomes and optionally full time series.
    """
    S0    = params["S0"]
    T     = params["T"]
    dt    = params["dt"]
    N     = params["N"]
    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    q0    = params["q0"]
    X0    = params["X0"]

    S = float(S0)
    q = int(q0)
    X = float(X0)

    if store_full:
        S_path = np.empty(N + 1)
        q_path = np.empty(N + 1, dtype=int)
        X_path = np.empty(N + 1)
        r_path = np.empty(N + 1)
        pb_path = np.empty(N + 1)
        pa_path = np.empty(N + 1)
        S_path[0] = S
        q_path[0] = q
        X_path[0] = X

    n_bid_fills = 0
    n_ask_fills = 0

    # Pre-draw all random numbers for efficiency
    Z_brownian = rng.standard_normal(N)          # Brownian increments
    U_bid      = rng.uniform(0.0, 1.0, N)        # bid execution draws
    U_ask      = rng.uniform(0.0, 1.0, N)        # ask execution draws

    for n in range(N):
        t = n * dt

        # Compute quotes
        if strategy == "inventory":
            p_bid, p_ask = inventory_based_quotes(S, q, t, gamma, sigma, T, k)
        else:  # symmetric
            p_bid, p_ask = symmetric_quotes(S, t, gamma, sigma, T, k)

        # Quote distances from mid-price
        delta_b = S - p_bid   # >= 0 by construction
        delta_a = p_ask - S   # >= 0 by construction

        # Execution intensities
        lambda_b = A * np.exp(-k * delta_b)
        lambda_a = A * np.exp(-k * delta_a)

        # Per-step execution probabilities (exact Bernoulli approximation)
        # Custom — Context7 found no library equivalent (paper discrete-time approx)
        prob_b = 1.0 - np.exp(-lambda_b * dt)
        prob_a = 1.0 - np.exp(-lambda_a * dt)

        # Execute bid fill (buy one unit)
        if U_bid[n] < prob_b:
            q += 1
            X -= p_bid
            n_bid_fills += 1

        # Execute ask fill (sell one unit) — independent of bid fill
        if U_ask[n] < prob_a:
            q -= 1
            X += p_ask
            n_ask_fills += 1

        # Store state before price update
        if store_full:
            r_val = reservation_price(S, q, t, gamma, sigma, T)
            r_path[n]  = r_val
            pb_path[n] = p_bid
            pa_path[n] = p_ask

        # Evolve mid-price
        S += sigma * np.sqrt(dt) * Z_brownian[n]

        if store_full:
            S_path[n + 1] = S
            q_path[n + 1] = q
            X_path[n + 1] = X

    # Terminal marked-to-market wealth
    W_T = X + q * S

    result = {
        "W_T":          W_T,
        "q_T":          q,
        "X_T":          X,
        "S_T":          S,
        "n_bid_fills":  n_bid_fills,
        "n_ask_fills":  n_ask_fills,
    }

    if store_full:
        r_path[N]  = reservation_price(S, q, T, gamma, sigma, T)
        pb_path[N] = p_bid  # last computed quotes
        pa_path[N] = p_ask
        result.update({
            "S_path":  S_path,
            "q_path":  q_path,
            "X_path":  X_path,
            "r_path":  r_path,
            "pb_path": pb_path,
            "pa_path": pa_path,
        })

    return result


# ---------------------------------------------------------------------------
# 3. Full Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    gamma: float,
    strategy: str,
    params: Dict,
    seed: int = 42,
    n_full_paths: int = 3,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Run the full Monte Carlo simulation for a given gamma and strategy.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    strategy : str
        "inventory" or "symmetric".
    params : dict
        Simulation parameters.
    seed : int
        Random seed for reproducibility.
    n_full_paths : int
        Number of paths for which full time series are stored.

    Returns
    -------
    tuple
        (DataFrame of terminal outcomes, list of full-path dicts)
    """
    n_sim = params["n_sim"]
    rng   = np.random.default_rng(seed)

    records    = []
    full_paths = []

    for i in range(n_sim):
        store = i < n_full_paths
        result = simulate_path(strategy, gamma, params, rng, store_full=store)
        records.append({
            "path_id":      i,
            "W_T":          result["W_T"],
            "q_T":          result["q_T"],
            "n_bid_fills":  result["n_bid_fills"],
            "n_ask_fills":  result["n_ask_fills"],
        })
        if store:
            full_paths.append(result)

    df = pd.DataFrame(records)
    return df, full_paths


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for terminal outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns W_T (terminal P&L) and q_T (final inventory).

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    return {
        "mean_PnL":       float(df["W_T"].mean()),
        "std_PnL":        float(df["W_T"].std()),
        "skew_PnL":       float(df["W_T"].skew()),
        "kurt_PnL":       float(df["W_T"].kurt()),
        "q05_PnL":        float(df["W_T"].quantile(0.05)),
        "q95_PnL":        float(df["W_T"].quantile(0.95)),
        "mean_inv":       float(df["q_T"].mean()),
        "std_inv":        float(df["q_T"].std()),
        "mean_bid_fills": float(df["n_bid_fills"].mean()),
        "mean_ask_fills": float(df["n_ask_fills"].mean()),
    }


# ---------------------------------------------------------------------------
# 5. Plotting functions
# ---------------------------------------------------------------------------

def plot_sample_paths(
    full_paths_inv: List[Dict],
    full_paths_sym: List[Dict],
    gamma: float,
    params: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sample paths showing mid-price, reservation price, quotes, and inventory.
    Reproduces the qualitative message of Figure 1 in the paper.

    Parameters
    ----------
    full_paths_inv : list[dict]
        Full path data for inventory-based strategy.
    full_paths_sym : list[dict]
        Full path data for symmetric strategy.
    gamma : float
        Risk-aversion coefficient.
    params : dict
        Simulation parameters.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dt = params["dt"]
    N  = params["N"]
    t_grid = np.arange(N + 1) * dt

    path = full_paths_inv[0]  # use first path

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.35)

    # Top panel: prices
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_grid, path["S_path"],  "k-",  linewidth=1.2, label="Mid-price S_t")
    ax1.plot(t_grid, path["r_path"],  "g--", linewidth=1.2, label="Reservation price r_t")
    ax1.plot(t_grid, path["pb_path"], "b:",  linewidth=1.0, label="Bid p^b_t")
    ax1.plot(t_grid, path["pa_path"], "r:",  linewidth=1.0, label="Ask p^a_t")
    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title(
        f"Sample Path — Inventory Strategy (γ={gamma})\n"
        f"σ={params['sigma']}, T={params['T']}, A={params['A']}, k={params['k']}",
        fontsize=12,
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: inventory
    ax2 = fig.add_subplot(gs[1])
    ax2.step(t_grid, path["q_path"], "purple", linewidth=1.2, where="post",
             label="Inventory q_t (inv. strategy)")
    if full_paths_sym:
        path_sym = full_paths_sym[0]
        ax2.step(t_grid, path_sym["q_path"], "orange", linewidth=1.0,
                 linestyle="--", where="post", label="Inventory q_t (symmetric)")
    ax2.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Time t", fontsize=11)
    ax2.set_ylabel("Inventory q_t", fontsize=11)
    ax2.set_title("Inventory Path", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pnl_histograms(
    df_inv: pd.DataFrame,
    df_sym: pd.DataFrame,
    gamma: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlaid histograms of terminal P&L for both strategies.
    Reproduces the qualitative message of Figures 2-4 in the paper.

    Parameters
    ----------
    df_inv : pd.DataFrame
        Terminal outcomes for inventory-based strategy.
    df_sym : pd.DataFrame
        Terminal outcomes for symmetric strategy.
    gamma : float
        Risk-aversion coefficient.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # P&L histogram
    ax = axes[0]
    bins = np.linspace(
        min(df_inv["W_T"].min(), df_sym["W_T"].min()),
        max(df_inv["W_T"].max(), df_sym["W_T"].max()),
        50,
    )
    ax.hist(df_inv["W_T"], bins=bins, alpha=0.6, color="steelblue",
            label=f"Inventory (std={df_inv['W_T'].std():.2f})")
    ax.hist(df_sym["W_T"], bins=bins, alpha=0.6, color="tomato",
            label=f"Symmetric (std={df_sym['W_T'].std():.2f})")
    ax.set_xlabel("Terminal P&L (W_T)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Terminal P&L Distribution (γ={gamma})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Inventory histogram
    ax = axes[1]
    q_min = min(df_inv["q_T"].min(), df_sym["q_T"].min())
    q_max = max(df_inv["q_T"].max(), df_sym["q_T"].max())
    bins_q = np.arange(q_min - 0.5, q_max + 1.5, 1.0)
    ax.hist(df_inv["q_T"], bins=bins_q, alpha=0.6, color="steelblue",
            label=f"Inventory (std={df_inv['q_T'].std():.2f})")
    ax.hist(df_sym["q_T"], bins=bins_q, alpha=0.6, color="tomato",
            label=f"Symmetric (std={df_sym['q_T'].std():.2f})")
    ax.set_xlabel("Final Inventory q_T", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Final Inventory Distribution (γ={gamma})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary_comparison(
    all_stats: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a summary comparison of std(P&L) and std(inventory) across gamma values.

    Parameters
    ----------
    all_stats : dict
        Nested dict: all_stats[gamma][strategy] -> stats dict.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    gamma_list = sorted(all_stats.keys())
    std_pnl_inv = [all_stats[g]["inventory"]["std_PnL"] for g in gamma_list]
    std_pnl_sym = [all_stats[g]["symmetric"]["std_PnL"] for g in gamma_list]
    std_inv_inv = [all_stats[g]["inventory"]["std_inv"] for g in gamma_list]
    std_inv_sym = [all_stats[g]["symmetric"]["std_inv"] for g in gamma_list]

    x = np.arange(len(gamma_list))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x - width / 2, std_pnl_inv, width, label="Inventory", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, std_pnl_sym, width, label="Symmetric", color="tomato",    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"γ={g}" for g in gamma_list], fontsize=11)
    ax.set_ylabel("Std Dev of Terminal P&L", fontsize=11)
    ax.set_title("P&L Volatility Comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - width / 2, std_inv_inv, width, label="Inventory", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, std_inv_sym, width, label="Symmetric", color="tomato",    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"γ={g}" for g in gamma_list], fontsize=11)
    ax.set_ylabel("Std Dev of Final Inventory", fontsize=11)
    ax.set_title("Inventory Dispersion Comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Main runner
# ---------------------------------------------------------------------------

def run_experiment_2(
    results_dir: str = "results",
    params: Optional[Dict] = None,
    seed: int = 42,
) -> Dict:
    """
    Run Experiment 2: Monte Carlo comparison of inventory-based vs symmetric strategy.

    Parameters
    ----------
    results_dir : str
        Directory to save figures and outputs.
    params : dict, optional
        Simulation parameters (defaults to DEFAULT_PARAMS).
    seed : int
        Random seed.

    Returns
    -------
    dict
        All summary statistics and hypothesis test results.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    gamma_list = [0.01, 0.1, 0.5]
    strategies = ["inventory", "symmetric"]

    print("=" * 70)
    print("EXPERIMENT 2: Monte Carlo Simulation")
    print(f"  n_sim={params['n_sim']}, N={params['N']}, dt={params['dt']}")
    print(f"  sigma={params['sigma']}, A={params['A']}, k={params['k']}")
    print("=" * 70)

    all_stats: Dict = {}
    all_dfs:   Dict = {}

    for gamma in gamma_list:
        all_stats[gamma] = {}
        all_dfs[gamma]   = {}
        print(f"\n--- gamma = {gamma} ---")

        full_paths_by_strategy: Dict = {}

        for strategy in strategies:
            # Use different seeds per (gamma, strategy) for independence
            strat_seed = seed + int(gamma * 1000) + (0 if strategy == "inventory" else 500)
            df, full_paths = run_monte_carlo(
                gamma=gamma,
                strategy=strategy,
                params=params,
                seed=strat_seed,
                n_full_paths=3,
            )
            stats = compute_summary_stats(df)
            all_stats[gamma][strategy] = stats
            all_dfs[gamma][strategy]   = df
            full_paths_by_strategy[strategy] = full_paths

            print(f"  {strategy:12s}: mean_PnL={stats['mean_PnL']:8.3f}, "
                  f"std_PnL={stats['std_PnL']:7.3f}, "
                  f"mean_inv={stats['mean_inv']:6.3f}, "
                  f"std_inv={stats['std_inv']:6.3f}")

        # Sample path plot
        fig_path = plot_sample_paths(
            full_paths_by_strategy["inventory"],
            full_paths_by_strategy["symmetric"],
            gamma=gamma,
            params=params,
            save_path=os.path.join(fig_dir, f"exp2_sample_path_gamma{gamma}.png"),
        )
        plt.close(fig_path)

        # P&L and inventory histograms
        fig_hist = plot_pnl_histograms(
            all_dfs[gamma]["inventory"],
            all_dfs[gamma]["symmetric"],
            gamma=gamma,
            save_path=os.path.join(fig_dir, f"exp2_histograms_gamma{gamma}.png"),
        )
        plt.close(fig_hist)

    # Summary comparison plot
    fig_sum = plot_summary_comparison(
        all_stats,
        save_path=os.path.join(fig_dir, "exp2_summary_comparison.png"),
    )
    plt.close(fig_sum)

    # Hypothesis evaluation
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    print(f"\n{'gamma':>8} {'std_PnL_inv':>12} {'std_PnL_sym':>12} {'red%':>8} "
          f"{'std_inv_inv':>12} {'std_inv_sym':>12} {'red%':>8}")
    print("-" * 80)

    hypothesis_results = {}
    for gamma in gamma_list:
        s_inv = all_stats[gamma]["inventory"]
        s_sym = all_stats[gamma]["symmetric"]
        pnl_red = 100.0 * (1.0 - s_inv["std_PnL"] / s_sym["std_PnL"])
        inv_red = 100.0 * (1.0 - s_inv["std_inv"] / s_sym["std_inv"])
        print(f"{gamma:>8.3f} {s_inv['std_PnL']:>12.4f} {s_sym['std_PnL']:>12.4f} "
              f"{pnl_red:>7.1f}% {s_inv['std_inv']:>12.4f} {s_sym['std_inv']:>12.4f} "
              f"{inv_red:>7.1f}%")
        hypothesis_results[gamma] = {
            "pnl_reduction_pct":      pnl_red,
            "inv_reduction_pct":      inv_red,
            "inv_lower_std_pnl":      s_inv["std_PnL"] < s_sym["std_PnL"],
            "inv_lower_std_inv":      s_inv["std_inv"] < s_sym["std_inv"],
        }

    # Save per-path results to CSV
    for gamma in gamma_list:
        for strategy in strategies:
            csv_path = os.path.join(
                results_dir,
                f"exp2_paths_gamma{gamma}_{strategy}.csv",
            )
            all_dfs[gamma][strategy].to_csv(csv_path, index=False)

    print("\nFigures and CSVs saved to:", results_dir)

    return {
        "all_stats":          all_stats,
        "hypothesis_results": hypothesis_results,
        "params":             params,
    }
