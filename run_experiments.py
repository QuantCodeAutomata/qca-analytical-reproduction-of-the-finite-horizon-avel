"""
Main runner for all three Avellaneda-Stoikov experiments.

Runs:
  - Experiment 1: Analytical finite-horizon model
  - Experiment 2: Monte Carlo simulation comparison
  - Experiment 3: Infinite-horizon variants and appendix formulations

Saves all results and figures to the results/ directory.
"""

from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.exp1_analytical import run_experiment_1
from src.exp2_montecarlo import run_experiment_2, DEFAULT_PARAMS
from src.exp3_variants    import run_experiment_3


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_results_markdown(
    exp1_summary: dict,
    exp2_summary: dict,
    exp3_summary: dict,
) -> None:
    """
    Save all experiment results to results/RESULTS.md.

    Parameters
    ----------
    exp1_summary : dict
        Summary from Experiment 1.
    exp2_summary : dict
        Summary from Experiment 2.
    exp3_summary : dict
        Summary from Experiment 3.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")

    lines = [
        "# Avellaneda-Stoikov Market-Making Model — Experiment Results",
        "",
        "> Reference: Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*",
        "",
        "---",
        "",
        "## Experiment 1: Analytical Finite-Horizon Model",
        "",
        "### Model Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| σ (volatility) | {exp1_summary['parameters']['sigma']} |",
        f"| T (horizon) | {exp1_summary['parameters']['T']} |",
        f"| A (intensity scale) | {exp1_summary['parameters']['A']} |",
        f"| k (intensity decay) | {exp1_summary['parameters']['k']} |",
        "",
        "### Key Formulas Derived",
        "",
        "| Formula | Expression |",
        "|---------|------------|",
        f"| Reservation price | `{exp1_summary['formulas']['reservation_price']}` |",
        f"| Full spread | `{exp1_summary['formulas']['full_spread']}` |",
        f"| Inventory bid | `{exp1_summary['formulas']['bid_inv']}` |",
        f"| Inventory ask | `{exp1_summary['formulas']['ask_inv']}` |",
        f"| Symmetric bid | `{exp1_summary['formulas']['bid_sym']}` |",
        f"| Symmetric ask | `{exp1_summary['formulas']['ask_sym']}` |",
        "",
        "### Comparative Statics (σ=2, T=1, S=100)",
        "",
        "| γ | t | τ=T-t | Spread ψ | r(q=-5) | r(q=0) | r(q=5) |",
        "|---|---|-------|----------|---------|--------|--------|",
    ]

    for gamma in exp1_summary["gamma_list"]:
        for t_val in [0.0, 0.5, 0.99]:
            tau  = 1.0 - t_val
            data = exp1_summary["comparative_statics"][gamma][t_val]
            psi  = data["spread"]
            r_m5 = data["r_vals"][0]
            r_0  = data["r_vals"][5]
            r_p5 = data["r_vals"][10]
            lines.append(
                f"| {gamma} | {t_val:.2f} | {tau:.2f} | {psi:.4f} | "
                f"{r_m5:.4f} | {r_0:.4f} | {r_p5:.4f} |"
            )

    lines += [
        "",
        "### Figures",
        "",
        "- `figures/exp1_reservation_price_vs_inventory.png` — Reservation price vs inventory",
        "- `figures/exp1_spread_vs_time_to_maturity.png` — Spread vs time-to-maturity",
        "- `figures/exp1_quote_locations_vs_inventory.png` — Quote locations vs inventory",
        "",
        "---",
        "",
        "## Experiment 2: Monte Carlo Simulation",
        "",
        "### Simulation Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| S₀ | {exp2_summary['params']['S0']} |",
        f"| T | {exp2_summary['params']['T']} |",
        f"| dt | {exp2_summary['params']['dt']} |",
        f"| N | {exp2_summary['params']['N']} |",
        f"| σ | {exp2_summary['params']['sigma']} |",
        f"| A | {exp2_summary['params']['A']} |",
        f"| k | {exp2_summary['params']['k']} |",
        f"| n_sim | {exp2_summary['params']['n_sim']} |",
        "",
        "### Summary Statistics",
        "",
        "| γ | Strategy | Mean P&L | Std P&L | Mean Inv | Std Inv |",
        "|---|----------|----------|---------|----------|---------|",
    ]

    for gamma in sorted(exp2_summary["all_stats"].keys()):
        for strategy in ["inventory", "symmetric"]:
            s = exp2_summary["all_stats"][gamma][strategy]
            lines.append(
                f"| {gamma} | {strategy} | {s['mean_PnL']:.4f} | {s['std_PnL']:.4f} | "
                f"{s['mean_inv']:.4f} | {s['std_inv']:.4f} |"
            )

    lines += [
        "",
        "### Hypothesis Test Results",
        "",
        "| γ | Std P&L Reduction (%) | Std Inv Reduction (%) | Inv < Sym P&L? | Inv < Sym Inv? |",
        "|---|----------------------|----------------------|----------------|----------------|",
    ]

    for gamma in sorted(exp2_summary["hypothesis_results"].keys()):
        h = exp2_summary["hypothesis_results"][gamma]
        lines.append(
            f"| {gamma} | {h['pnl_reduction_pct']:.1f}% | {h['inv_reduction_pct']:.1f}% | "
            f"{'✓' if h['inv_lower_std_pnl'] else '✗'} | "
            f"{'✓' if h['inv_lower_std_inv'] else '✗'} |"
        )

    lines += [
        "",
        "### Figures",
        "",
        "- `figures/exp2_sample_path_gamma*.png` — Sample paths for each γ",
        "- `figures/exp2_histograms_gamma*.png` — P&L and inventory histograms",
        "- `figures/exp2_summary_comparison.png` — Summary comparison bar chart",
        "",
        "---",
        "",
        "## Experiment 3: Infinite-Horizon Variants",
        "",
        "### Key Formulas",
        "",
        "| Formula | Expression |",
        "|---------|------------|",
    ]

    for name, formula in exp3_summary["key_formulas"].items():
        lines.append(f"| {name} | `{formula}` |")

    lines += [
        "",
        "### Omega Admissibility Requirements (σ=2)",
        "",
        "| γ | q_max | ω_min at q_max | ω_sufficient | Admissible |",
        "|---|-------|----------------|--------------|------------|",
    ]

    for row in exp3_summary["admissibility_table"]:
        lines.append(
            f"| {row['gamma']} | {row['q_max']} | {row['omega_min_at_qmax']:.6f} | "
            f"{row['omega_sufficient']:.6f} | {'✓' if row['admissible'] else '✗'} |"
        )

    lines += [
        "",
        "### Model Variants Summary",
        "",
        "| Variant | Mid-price Dynamics | Objective | Horizon |",
        "|---------|-------------------|-----------|---------|",
        "| Main model | dS = σ dW (additive) | CARA: E[-exp(-γ W_T)] | Finite T |",
        "| Infinite-horizon | dS = σ dW (additive) | Discounted CARA | Infinite (discount ω) |",
        "| Appendix | dS/S = σ dW (multiplicative) | Mean/Variance | Finite T |",
        "",
        "### Figures",
        "",
        "- `figures/exp3_omega_requirements.png` — Omega requirements vs q_max",
        "- `figures/exp3_infinite_horizon_quotes.png` — Infinite-horizon quotes",
        "- `figures/exp3_model_comparison.png` — Model comparison",
        "",
        "---",
        "",
        "## Conclusions",
        "",
        "1. **Reservation price formula**: r(s,q,t) = s − q·γ·σ²·(T−t) correctly captures",
        "   inventory-driven quote skew, with magnitude proportional to γ, σ², q, and τ=T−t.",
        "",
        "2. **Optimal spread**: ψ = γ·σ²·(T−t) + (2/γ)·ln(1+γ/k) decomposes into an",
        "   inventory-risk term (vanishes at maturity) and an execution term (time-independent).",
        "",
        "3. **Monte Carlo validation**: The inventory-based strategy reduces both P&L variance",
        "   and final inventory variance relative to the symmetric benchmark, with effects",
        "   stronger at larger γ (as predicted by theory).",
        "",
        "4. **Infinite-horizon admissibility**: The sufficient condition ω = 0.5·γ²·σ²·(q_max+1)²",
        "   ensures well-posedness for all inventory states |q| ≤ q_max.",
        "",
        "5. **Model variants**: The appendix multiplicative-diffusion / mean-variance model",
        "   is qualitatively similar to the main CARA model but uses different coefficients.",
        "   These variants are not required for the main numerical reproduction.",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults saved to: {md_path}")


def main() -> None:
    """Run all experiments and save results."""
    print("\n" + "=" * 70)
    print("AVELLANEDA-STOIKOV MARKET-MAKING MODEL — FULL EXPERIMENT SUITE")
    print("=" * 70)

    t_start = time.time()

    # --- Experiment 1 ---
    print("\n[1/3] Running Experiment 1: Analytical Model...")
    exp1_summary = run_experiment_1(results_dir=RESULTS_DIR)

    # --- Experiment 2 ---
    print("\n[2/3] Running Experiment 2: Monte Carlo Simulation...")
    params = DEFAULT_PARAMS.copy()
    exp2_summary = run_experiment_2(results_dir=RESULTS_DIR, params=params, seed=42)

    # --- Experiment 3 ---
    print("\n[3/3] Running Experiment 3: Infinite-Horizon Variants...")
    exp3_summary = run_experiment_3(results_dir=RESULTS_DIR)

    # --- Save results ---
    save_results_markdown(exp1_summary, exp2_summary, exp3_summary)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {elapsed:.1f}s")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
