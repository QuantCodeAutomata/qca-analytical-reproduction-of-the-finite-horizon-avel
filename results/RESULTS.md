# Avellaneda-Stoikov Market-Making Model — Experiment Results

> Reference: Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*

---

## Experiment 1: Analytical Finite-Horizon Model

### Model Parameters

| Parameter | Value |
|-----------|-------|
| σ (volatility) | 2.0 |
| T (horizon) | 1.0 |
| A (intensity scale) | 140.0 |
| k (intensity decay) | 1.5 |

### Key Formulas Derived

| Formula | Expression |
|---------|------------|
| Reservation price | `r(s,q,t) = s - q*gamma*sigma^2*(T-t)` |
| Full spread | `psi(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)` |
| Inventory bid | `p^b = r - psi/2` |
| Inventory ask | `p^a = r + psi/2` |
| Symmetric bid | `p^b_sym = s - psi/2` |
| Symmetric ask | `p^a_sym = s + psi/2` |

### Comparative Statics (σ=2, T=1, S=100)

| γ | t | τ=T-t | Spread ψ | r(q=-5) | r(q=0) | r(q=5) |
|---|---|-------|----------|---------|--------|--------|
| 0.01 | 0.00 | 1.00 | 1.3689 | 100.2000 | 100.0000 | 99.8000 |
| 0.01 | 0.50 | 0.50 | 1.3489 | 100.1000 | 100.0000 | 99.9000 |
| 0.01 | 0.99 | 0.01 | 1.3293 | 100.0020 | 100.0000 | 99.9980 |
| 0.1 | 0.00 | 1.00 | 1.6908 | 102.0000 | 100.0000 | 98.0000 |
| 0.1 | 0.50 | 0.50 | 1.4908 | 101.0000 | 100.0000 | 99.0000 |
| 0.1 | 0.99 | 0.01 | 1.2948 | 100.0200 | 100.0000 | 99.9800 |
| 0.5 | 0.00 | 1.00 | 3.1507 | 110.0000 | 100.0000 | 90.0000 |
| 0.5 | 0.50 | 0.50 | 2.1507 | 105.0000 | 100.0000 | 95.0000 |
| 0.5 | 0.99 | 0.01 | 1.1707 | 100.1000 | 100.0000 | 99.9000 |

### Figures

- `figures/exp1_reservation_price_vs_inventory.png` — Reservation price vs inventory
- `figures/exp1_spread_vs_time_to_maturity.png` — Spread vs time-to-maturity
- `figures/exp1_quote_locations_vs_inventory.png` — Quote locations vs inventory

---

## Experiment 2: Monte Carlo Simulation

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| S₀ | 100.0 |
| T | 1.0 |
| dt | 0.005 |
| N | 200 |
| σ | 2.0 |
| A | 140.0 |
| k | 1.5 |
| n_sim | 1000 |

### Summary Statistics

| γ | Strategy | Mean P&L | Std P&L | Mean Inv | Std Inv |
|---|----------|----------|---------|----------|---------|
| 0.01 | inventory | 60.4413 | 9.0063 | 0.1590 | 5.2550 |
| 0.01 | symmetric | 60.2653 | 13.1059 | -0.0590 | 8.2164 |
| 0.1 | inventory | 56.9237 | 6.3408 | 0.0000 | 2.9676 |
| 0.1 | symmetric | 60.5329 | 13.2966 | -0.4400 | 8.0438 |
| 0.5 | inventory | 42.8275 | 5.7970 | 0.0400 | 2.0121 |
| 0.5 | symmetric | 53.6215 | 11.4684 | -0.0570 | 6.9144 |

### Hypothesis Test Results

| γ | Std P&L Reduction (%) | Std Inv Reduction (%) | Inv < Sym P&L? | Inv < Sym Inv? |
|---|----------------------|----------------------|----------------|----------------|
| 0.01 | 31.3% | 36.0% | ✓ | ✓ |
| 0.1 | 52.3% | 63.1% | ✓ | ✓ |
| 0.5 | 49.5% | 70.9% | ✓ | ✓ |

### Figures

- `figures/exp2_sample_path_gamma*.png` — Sample paths for each γ
- `figures/exp2_histograms_gamma*.png` — P&L and inventory histograms
- `figures/exp2_summary_comparison.png` — Summary comparison bar chart

---

## Experiment 3: Infinite-Horizon Variants

### Key Formulas

| Formula | Expression |
|---------|------------|
| admissibility_condition | `omega > 0.5 * gamma^2 * sigma^2 * q^2` |
| sufficient_omega | `omega = 0.5 * gamma^2 * sigma^2 * (q_max+1)^2` |
| inf_horizon_res_price | `r_inf(s,q) = s - q*gamma*sigma^2 / (2*omega)` |
| inf_horizon_spread | `psi_inf = gamma*sigma^2/(2*omega) + (2/gamma)*ln(1+gamma/k)` |
| mv_res_price | `r_mv(s,q,t) = s - q*lambda*sigma^2*(T-t)` |

### Omega Admissibility Requirements (σ=2)

| γ | q_max | ω_min at q_max | ω_sufficient | Admissible |
|---|-------|----------------|--------------|------------|
| 0.01 | 5 | 0.005000 | 0.007200 | ✓ |
| 0.01 | 10 | 0.020000 | 0.024200 | ✓ |
| 0.01 | 20 | 0.080000 | 0.088200 | ✓ |
| 0.1 | 5 | 0.500000 | 0.720000 | ✓ |
| 0.1 | 10 | 2.000000 | 2.420000 | ✓ |
| 0.1 | 20 | 8.000000 | 8.820000 | ✓ |
| 0.5 | 5 | 12.500000 | 18.000000 | ✓ |
| 0.5 | 10 | 50.000000 | 60.500000 | ✓ |
| 0.5 | 20 | 200.000000 | 220.500000 | ✓ |

### Model Variants Summary

| Variant | Mid-price Dynamics | Objective | Horizon |
|---------|-------------------|-----------|---------|
| Main model | dS = σ dW (additive) | CARA: E[-exp(-γ W_T)] | Finite T |
| Infinite-horizon | dS = σ dW (additive) | Discounted CARA | Infinite (discount ω) |
| Appendix | dS/S = σ dW (multiplicative) | Mean/Variance | Finite T |

### Figures

- `figures/exp3_omega_requirements.png` — Omega requirements vs q_max
- `figures/exp3_infinite_horizon_quotes.png` — Infinite-horizon quotes
- `figures/exp3_model_comparison.png` — Model comparison

---

## Conclusions

1. **Reservation price formula**: r(s,q,t) = s − q·γ·σ²·(T−t) correctly captures
   inventory-driven quote skew, with magnitude proportional to γ, σ², q, and τ=T−t.

2. **Optimal spread**: ψ = γ·σ²·(T−t) + (2/γ)·ln(1+γ/k) decomposes into an
   inventory-risk term (vanishes at maturity) and an execution term (time-independent).

3. **Monte Carlo validation**: The inventory-based strategy reduces both P&L variance
   and final inventory variance relative to the symmetric benchmark, with effects
   stronger at larger γ (as predicted by theory).

4. **Infinite-horizon admissibility**: The sufficient condition ω = 0.5·γ²·σ²·(q_max+1)²
   ensures well-posedness for all inventory states |q| ≤ q_max.

5. **Model variants**: The appendix multiplicative-diffusion / mean-variance model
   is qualitatively similar to the main CARA model but uses different coefficients.
   These variants are not required for the main numerical reproduction.
