# Repository: qca-analytical-reproduction-of-the-finite-horizon-avel

## Overview
Python implementation of the Avellaneda-Stoikov (2008) market-making model.

## Structure
```
src/
  exp1_analytical.py   — Analytical finite-horizon AS model (formulas, plots)
  exp2_montecarlo.py   — Monte Carlo simulation (inventory vs symmetric strategy)
  exp3_variants.py     — Infinite-horizon variants and appendix formulations
tests/
  test_exp1_analytical.py
  test_exp2_montecarlo.py
  test_exp3_variants.py
run_experiments.py     — Main runner (saves results to results/)
results/
  RESULTS.md           — All metrics and findings
  figures/             — All generated plots
```

## Key Formulas
- Reservation price: `r(s,q,t) = s - q*gamma*sigma^2*(T-t)`
- Full spread: `psi = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)`
- Inventory bid: `p^b = r - psi/2`
- Inventory ask: `p^a = r + psi/2`
- Symmetric bid: `p^b_sym = s - psi/2`
- Symmetric ask: `p^a_sym = s + psi/2`

## Paper Parameters
- sigma=2, T=1, A=140, k=1.5, S0=100, dt=0.005, N=200, n_sim=1000
- gamma in {0.01, 0.1, 0.5}

## Running
```bash
python run_experiments.py   # Run all experiments
pytest tests/ -v            # Run all tests
```

## Notes
- Uses `log1p(gamma/k)` for numerical stability with small gamma
- Exact Bernoulli approximation `1-exp(-lambda*dt)` for execution probabilities
- All plots saved as PNG to results/figures/
- Fixed random seed=42 for reproducibility
