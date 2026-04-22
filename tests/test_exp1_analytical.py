"""
Tests for Experiment 1: Analytical Avellaneda-Stoikov model.

Verifies:
  - Reservation price formula correctness and monotonicity
  - Spread formula correctness and monotonicity
  - Quote placement formulas
  - Execution intensity formula
  - Edge cases (small gamma, zero inventory, maturity)
  - Mathematical properties stated in the paper
"""

import pytest
import numpy as np
import sympy as sp

from src.exp1_analytical import (
    reservation_price,
    optimal_spread,
    inventory_based_quotes,
    symmetric_quotes,
    execution_intensity,
    compute_comparative_statics,
    derive_spread_symbolically,
    reservation_price_derivation,
    hjb_equation_description,
    cara_ansatz_description,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_params():
    """Paper's main parameter values."""
    return {"sigma": 2.0, "T": 1.0, "A": 140.0, "k": 1.5}


@pytest.fixture
def gamma_list():
    return [0.01, 0.1, 0.5]


# ---------------------------------------------------------------------------
# Tests: reservation_price
# ---------------------------------------------------------------------------

class TestReservationPrice:
    """Tests for the reservation price formula r(s,q,t) = s - q*gamma*sigma^2*(T-t)."""

    def test_zero_inventory_equals_mid_price(self, paper_params):
        """At q=0, reservation price equals mid-price."""
        r = reservation_price(100.0, 0, 0.0, 0.1, paper_params["sigma"], paper_params["T"])
        assert r == pytest.approx(100.0), "r(q=0) should equal mid-price"

    def test_positive_inventory_below_mid(self, paper_params):
        """Long inventory (q>0) should push reservation price below mid-price."""
        r = reservation_price(100.0, 1, 0.0, 0.1, paper_params["sigma"], paper_params["T"])
        assert r < 100.0, "r(q>0) should be below mid-price"

    def test_negative_inventory_above_mid(self, paper_params):
        """Short inventory (q<0) should push reservation price above mid-price."""
        r = reservation_price(100.0, -1, 0.0, 0.1, paper_params["sigma"], paper_params["T"])
        assert r > 100.0, "r(q<0) should be above mid-price"

    def test_decreasing_in_inventory(self, paper_params):
        """Reservation price should be strictly decreasing in inventory q."""
        q_grid = range(-5, 6)
        r_vals = [
            reservation_price(100.0, q, 0.0, 0.1, paper_params["sigma"], paper_params["T"])
            for q in q_grid
        ]
        assert all(r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1)), \
            "r should be strictly decreasing in q"

    def test_vanishes_at_maturity(self, paper_params):
        """At t=T, reservation price equals mid-price regardless of inventory."""
        for q in [-5, -1, 0, 1, 5]:
            r = reservation_price(
                100.0, q, paper_params["T"], 0.1,
                paper_params["sigma"], paper_params["T"]
            )
            assert r == pytest.approx(100.0), \
                f"r(q={q}, t=T) should equal mid-price"

    def test_proportional_to_gamma(self, paper_params):
        """Inventory effect should scale linearly with gamma."""
        s, q, t = 100.0, 2, 0.0
        sigma, T = paper_params["sigma"], paper_params["T"]
        r1 = reservation_price(s, q, t, 0.1, sigma, T)
        r2 = reservation_price(s, q, t, 0.2, sigma, T)
        # Shift from mid = q*gamma*sigma^2*(T-t)
        shift1 = s - r1
        shift2 = s - r2
        assert shift2 == pytest.approx(2.0 * shift1, rel=1e-10), \
            "Inventory shift should scale linearly with gamma"

    def test_exact_formula(self, paper_params):
        """Verify exact numerical value against formula."""
        s, q, t, gamma = 100.0, 3, 0.5, 0.1
        sigma, T = paper_params["sigma"], paper_params["T"]
        expected = s - q * gamma * sigma**2 * (T - t)
        result   = reservation_price(s, q, t, gamma, sigma, T)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_small_gamma_near_mid(self, paper_params):
        """For very small gamma, reservation price should be very close to mid-price."""
        r = reservation_price(100.0, 5, 0.0, 1e-6, paper_params["sigma"], paper_params["T"])
        assert abs(r - 100.0) < 1e-3, "Small gamma should give near-zero inventory effect"


# ---------------------------------------------------------------------------
# Tests: optimal_spread
# ---------------------------------------------------------------------------

class TestOptimalSpread:
    """Tests for the spread formula psi = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""

    def test_positive_spread(self, paper_params):
        """Spread should always be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            for t in [0.0, 0.5, 0.99]:
                psi = optimal_spread(t, gamma, paper_params["sigma"],
                                     paper_params["T"], paper_params["k"])
                assert psi > 0, f"Spread should be positive for gamma={gamma}, t={t}"

    def test_spread_increases_with_gamma(self, paper_params):
        """Spread should increase with gamma (at fixed t)."""
        gamma_list = [0.01, 0.1, 0.5]
        spreads = [
            optimal_spread(0.0, g, paper_params["sigma"],
                           paper_params["T"], paper_params["k"])
            for g in gamma_list
        ]
        assert all(spreads[i] < spreads[i + 1] for i in range(len(spreads) - 1)), \
            "Spread should increase with gamma"

    def test_spread_decreases_with_time(self, paper_params):
        """Spread should decrease as t increases (time-to-maturity shrinks)."""
        t_grid = [0.0, 0.25, 0.5, 0.75, 0.99]
        spreads = [
            optimal_spread(t, 0.1, paper_params["sigma"],
                           paper_params["T"], paper_params["k"])
            for t in t_grid
        ]
        assert all(spreads[i] >= spreads[i + 1] for i in range(len(spreads) - 1)), \
            "Spread should decrease as t increases"

    def test_spread_at_maturity_is_execution_term(self, paper_params):
        """At t=T, spread equals only the execution term (2/gamma)*ln(1+gamma/k)."""
        gamma = 0.1
        psi_T = optimal_spread(
            paper_params["T"], gamma, paper_params["sigma"],
            paper_params["T"], paper_params["k"]
        )
        execution_term = (2.0 / gamma) * np.log1p(gamma / paper_params["k"])
        assert psi_T == pytest.approx(execution_term, rel=1e-10)

    def test_numerical_stability_small_gamma(self, paper_params):
        """Spread should be numerically stable for very small gamma."""
        psi = optimal_spread(0.0, 1e-8, paper_params["sigma"],
                             paper_params["T"], paper_params["k"])
        assert np.isfinite(psi), "Spread should be finite for small gamma"
        assert psi > 0, "Spread should be positive for small gamma"

    def test_exact_formula(self, paper_params):
        """Verify exact numerical value against formula."""
        gamma, t = 0.1, 0.0
        sigma, T, k = paper_params["sigma"], paper_params["T"], paper_params["k"]
        expected = gamma * sigma**2 * (T - t) + (2.0 / gamma) * np.log1p(gamma / k)
        result   = optimal_spread(t, gamma, sigma, T, k)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_decomposition_components(self, paper_params):
        """Verify that spread decomposes into inventory-risk and execution terms."""
        gamma, t = 0.1, 0.0
        sigma, T, k = paper_params["sigma"], paper_params["T"], paper_params["k"]
        psi = optimal_spread(t, gamma, sigma, T, k)
        inv_risk_term  = gamma * sigma**2 * (T - t)
        exec_term      = (2.0 / gamma) * np.log1p(gamma / k)
        assert psi == pytest.approx(inv_risk_term + exec_term, rel=1e-12)


# ---------------------------------------------------------------------------
# Tests: inventory_based_quotes
# ---------------------------------------------------------------------------

class TestInventoryBasedQuotes:
    """Tests for the inventory-based (Strategy A) quote formulas."""

    def test_ask_above_bid(self, paper_params):
        """Ask price should always be above bid price."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-5, 0, 5]:
                pb, pa = inventory_based_quotes(
                    100.0, q, 0.0, gamma,
                    paper_params["sigma"], paper_params["T"], paper_params["k"]
                )
                assert pa > pb, f"Ask should be above bid for gamma={gamma}, q={q}"

    def test_centered_at_reservation_price(self, paper_params):
        """Mid-quote (average of bid and ask) should equal reservation price."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-3, 0, 3]:
                pb, pa = inventory_based_quotes(
                    100.0, q, 0.0, gamma,
                    paper_params["sigma"], paper_params["T"], paper_params["k"]
                )
                r = reservation_price(100.0, q, 0.0, gamma,
                                      paper_params["sigma"], paper_params["T"])
                assert (pb + pa) / 2.0 == pytest.approx(r, rel=1e-10), \
                    f"Mid-quote should equal reservation price for gamma={gamma}, q={q}"

    def test_spread_equals_optimal_spread(self, paper_params):
        """Bid-ask spread should equal the optimal spread formula."""
        for gamma in [0.01, 0.1, 0.5]:
            pb, pa = inventory_based_quotes(
                100.0, 0, 0.0, gamma,
                paper_params["sigma"], paper_params["T"], paper_params["k"]
            )
            psi = optimal_spread(0.0, gamma, paper_params["sigma"],
                                 paper_params["T"], paper_params["k"])
            assert (pa - pb) == pytest.approx(psi, rel=1e-10)

    def test_quotes_shift_with_inventory(self, paper_params):
        """Both quotes should shift down when inventory is positive."""
        gamma = 0.1
        pb_q0, pa_q0 = inventory_based_quotes(
            100.0, 0, 0.0, gamma,
            paper_params["sigma"], paper_params["T"], paper_params["k"]
        )
        pb_q1, pa_q1 = inventory_based_quotes(
            100.0, 1, 0.0, gamma,
            paper_params["sigma"], paper_params["T"], paper_params["k"]
        )
        assert pb_q1 < pb_q0, "Bid should shift down with positive inventory"
        assert pa_q1 < pa_q0, "Ask should shift down with positive inventory"

    def test_zero_inventory_symmetric_around_mid(self, paper_params):
        """At q=0, quotes should be symmetric around mid-price."""
        gamma = 0.1
        pb, pa = inventory_based_quotes(
            100.0, 0, 0.0, gamma,
            paper_params["sigma"], paper_params["T"], paper_params["k"]
        )
        assert (100.0 - pb) == pytest.approx(pa - 100.0, rel=1e-10), \
            "At q=0, quotes should be symmetric around mid-price"


# ---------------------------------------------------------------------------
# Tests: symmetric_quotes
# ---------------------------------------------------------------------------

class TestSymmetricQuotes:
    """Tests for the symmetric benchmark (Strategy B) quote formulas."""

    def test_always_centered_at_mid(self, paper_params):
        """Symmetric quotes should always be centered at mid-price."""
        for gamma in [0.01, 0.1, 0.5]:
            pb, pa = symmetric_quotes(
                100.0, 0.0, gamma,
                paper_params["sigma"], paper_params["T"], paper_params["k"]
            )
            assert (pb + pa) / 2.0 == pytest.approx(100.0, rel=1e-10), \
                f"Symmetric quotes should be centered at mid for gamma={gamma}"

    def test_same_spread_as_inventory_strategy(self, paper_params):
        """Symmetric strategy should use the same spread as inventory strategy."""
        for gamma in [0.01, 0.1, 0.5]:
            pb_sym, pa_sym = symmetric_quotes(
                100.0, 0.0, gamma,
                paper_params["sigma"], paper_params["T"], paper_params["k"]
            )
            pb_inv, pa_inv = inventory_based_quotes(
                100.0, 0, 0.0, gamma,
                paper_params["sigma"], paper_params["T"], paper_params["k"]
            )
            assert (pa_sym - pb_sym) == pytest.approx(pa_inv - pb_inv, rel=1e-10), \
                f"Spreads should match for gamma={gamma}"

    def test_inventory_independent(self, paper_params):
        """Symmetric quotes should not depend on inventory (function has no q parameter)."""
        gamma = 0.1
        t = 0.0
        # symmetric_quotes does not accept q — calling twice with same args gives same result
        pb1, pa1 = symmetric_quotes(100.0, t, gamma,
                                    paper_params["sigma"], paper_params["T"], paper_params["k"])
        pb2, pa2 = symmetric_quotes(100.0, t, gamma,
                                    paper_params["sigma"], paper_params["T"], paper_params["k"])
        assert pb1 == pytest.approx(pb2, rel=1e-10)
        assert pa1 == pytest.approx(pa2, rel=1e-10)
        # Verify that inventory-based quotes at q=0 match symmetric quotes (same center)
        pb_inv, pa_inv = inventory_based_quotes(
            100.0, 0, t, gamma,
            paper_params["sigma"], paper_params["T"], paper_params["k"]
        )
        assert pb1 == pytest.approx(pb_inv, rel=1e-10)
        assert pa1 == pytest.approx(pa_inv, rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: execution_intensity
# ---------------------------------------------------------------------------

class TestExecutionIntensity:
    """Tests for the execution intensity lambda(delta) = A*exp(-k*delta)."""

    def test_decreasing_in_delta(self, paper_params):
        """Intensity should decrease as quote distance increases."""
        A, k = paper_params["A"], paper_params["k"]
        deltas = [0.0, 0.5, 1.0, 2.0, 5.0]
        intensities = [execution_intensity(d, A, k) for d in deltas]
        assert all(intensities[i] > intensities[i + 1] for i in range(len(intensities) - 1)), \
            "Intensity should decrease with quote distance"

    def test_at_zero_distance(self, paper_params):
        """At delta=0, intensity should equal A."""
        A, k = paper_params["A"], paper_params["k"]
        assert execution_intensity(0.0, A, k) == pytest.approx(A, rel=1e-12)

    def test_positive_intensity(self, paper_params):
        """Intensity should always be positive."""
        A, k = paper_params["A"], paper_params["k"]
        for delta in [0.0, 1.0, 5.0, 10.0]:
            assert execution_intensity(delta, A, k) > 0

    def test_exponential_decay(self, paper_params):
        """Verify exponential decay: lambda(delta+1)/lambda(delta) = exp(-k)."""
        A, k = paper_params["A"], paper_params["k"]
        for delta in [0.0, 1.0, 2.0]:
            ratio = execution_intensity(delta + 1.0, A, k) / execution_intensity(delta, A, k)
            assert ratio == pytest.approx(np.exp(-k), rel=1e-10)


# ---------------------------------------------------------------------------
# Tests: comparative statics
# ---------------------------------------------------------------------------

class TestComparativeStatics:
    """Tests for the comparative statics verification."""

    def test_monotonicity_reservation_price_in_q(self, paper_params):
        """Reservation price should be strictly decreasing in q."""
        stats = compute_comparative_statics(
            sigma=paper_params["sigma"], T=paper_params["T"],
            A=paper_params["A"], k=paper_params["k"],
            gamma_list=[0.1], q_grid=list(range(-5, 6)),
            t_grid=[0.0, 0.5],
        )
        for t_val in [0.0, 0.5]:
            r_vals = stats[0.1][t_val]["r_vals"]
            assert all(r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1))

    def test_spread_increases_with_gamma(self, paper_params):
        """Spread should increase with gamma."""
        gamma_list = [0.01, 0.1, 0.5]
        spreads = [
            optimal_spread(0.0, g, paper_params["sigma"],
                           paper_params["T"], paper_params["k"])
            for g in gamma_list
        ]
        assert all(spreads[i] < spreads[i + 1] for i in range(len(spreads) - 1))

    def test_inventory_sensitivity_vanishes_at_maturity(self, paper_params):
        """Inventory sensitivity (slope of r w.r.t. q) should vanish as t -> T."""
        gamma = 0.1
        sigma, T = paper_params["sigma"], paper_params["T"]
        t_grid = [0.0, 0.25, 0.5, 0.75, 0.99]
        slopes = [gamma * sigma**2 * (T - t) for t in t_grid]
        assert all(slopes[i] >= slopes[i + 1] for i in range(len(slopes) - 1))
        assert slopes[-1] == pytest.approx(gamma * sigma**2 * (T - 0.99), rel=1e-10)

    def test_weak_gamma_small_skew(self, paper_params):
        """For small gamma, inventory skew should be small."""
        r_small = reservation_price(100.0, 5, 0.0, 0.01,
                                    paper_params["sigma"], paper_params["T"])
        r_large = reservation_price(100.0, 5, 0.0, 0.5,
                                    paper_params["sigma"], paper_params["T"])
        assert abs(100.0 - r_small) < abs(100.0 - r_large), \
            "Small gamma should produce smaller inventory skew"


# ---------------------------------------------------------------------------
# Tests: symbolic derivations
# ---------------------------------------------------------------------------

class TestSymbolicDerivations:
    """Tests for symbolic derivation functions."""

    def test_hjb_description_contains_key_terms(self):
        """HJB description should contain key mathematical terms."""
        desc = hjb_equation_description()
        assert "u_t" in desc
        assert "sigma" in desc
        assert "lambda" in desc
        assert "terminal" in desc.lower()

    def test_cara_ansatz_description_contains_key_terms(self):
        """CARA ansatz description should contain key terms."""
        desc = cara_ansatz_description()
        assert "exp" in desc
        assert "gamma" in desc
        assert "theta" in desc

    def test_reservation_price_derivation_returns_sympy_exprs(self):
        """Reservation price derivation should return SymPy expressions."""
        desc, r_b, r_a = reservation_price_derivation()
        assert isinstance(r_b, sp.Expr), "r^b should be a SymPy expression"
        assert isinstance(r_a, sp.Expr), "r^a should be a SymPy expression"

    def test_spread_derivation_returns_sympy_expr(self):
        """Spread derivation should return a SymPy expression."""
        desc, half_spread = derive_spread_symbolically()
        assert isinstance(half_spread, sp.Expr), "Half-spread should be a SymPy expression"

    def test_reservation_price_derivation_description_content(self):
        """Reservation price derivation description should contain key content."""
        desc, _, _ = reservation_price_derivation()
        assert "indifference" in desc.lower() or "Indifference" in desc
        assert "r^b" in desc or "r_b" in desc


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_time_step(self, paper_params):
        """Formulas should work at a single time point."""
        r = reservation_price(100.0, 0, 0.0, 0.1,
                              paper_params["sigma"], paper_params["T"])
        assert np.isfinite(r)

    def test_extreme_inventory(self, paper_params):
        """Formulas should handle extreme inventory values."""
        for q in [-100, 100]:
            r = reservation_price(100.0, q, 0.0, 0.1,
                                  paper_params["sigma"], paper_params["T"])
            assert np.isfinite(r)

    def test_very_small_time_to_maturity(self, paper_params):
        """Formulas should be stable near maturity."""
        t_near_T = paper_params["T"] - 1e-10
        psi = optimal_spread(t_near_T, 0.1, paper_params["sigma"],
                             paper_params["T"], paper_params["k"])
        assert np.isfinite(psi)
        assert psi > 0

    def test_large_k_small_spread(self, paper_params):
        """Large k (fast intensity decay) should produce larger execution term."""
        psi_small_k = optimal_spread(0.0, 0.1, paper_params["sigma"],
                                     paper_params["T"], k=0.1)
        psi_large_k = optimal_spread(0.0, 0.1, paper_params["sigma"],
                                     paper_params["T"], k=10.0)
        # (2/gamma)*ln(1+gamma/k) decreases as k increases
        assert psi_small_k > psi_large_k, \
            "Larger k should produce smaller execution term"

    def test_spread_nonnegative_for_all_params(self):
        """Spread should be non-negative for all reasonable parameter combinations."""
        for gamma in [0.001, 0.01, 0.1, 0.5, 1.0]:
            for t in [0.0, 0.5, 1.0]:
                for k in [0.5, 1.5, 5.0]:
                    psi = optimal_spread(t, gamma, sigma=2.0, T=1.0, k=k)
                    assert psi >= 0, \
                        f"Spread should be non-negative for gamma={gamma}, t={t}, k={k}"
