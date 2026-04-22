"""
Tests for Experiment 3: Infinite-horizon variants and appendix formulations.

Verifies:
  - Admissibility condition for infinite-horizon model
  - Sufficient omega formula
  - Infinite-horizon reservation price and spread
  - Mean/variance objective formulas
  - Model comparison properties
  - Edge cases
"""

import pytest
import numpy as np
import pandas as pd

from src.exp3_variants import (
    minimum_omega,
    sufficient_omega,
    verify_admissibility,
    tabulate_omega_requirements,
    infinite_horizon_reservation_price,
    infinite_horizon_spread,
    mean_variance_reservation_price,
    compare_cara_vs_mean_variance,
    infinite_horizon_description,
    admissibility_condition_description,
    appendix_multiplicative_model_description,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paper_params():
    return {"sigma": 2.0, "T": 1.0, "A": 140.0, "k": 1.5}


@pytest.fixture
def gamma_list():
    return [0.01, 0.1, 0.5]


# ---------------------------------------------------------------------------
# Tests: minimum_omega
# ---------------------------------------------------------------------------

class TestMinimumOmega:
    """Tests for the minimum omega admissibility condition."""

    def test_zero_inventory_zero_omega(self, paper_params):
        """At q=0, minimum omega is 0."""
        omega_min = minimum_omega(0.1, paper_params["sigma"], 0)
        assert omega_min == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_nonzero_inventory(self, paper_params):
        """Minimum omega should be positive for nonzero inventory."""
        omega_min = minimum_omega(0.1, paper_params["sigma"], 1)
        assert omega_min > 0

    def test_increases_with_inventory_squared(self, paper_params):
        """Minimum omega should scale as q^2."""
        omega_q1 = minimum_omega(0.1, paper_params["sigma"], 1)
        omega_q2 = minimum_omega(0.1, paper_params["sigma"], 2)
        assert omega_q2 == pytest.approx(4.0 * omega_q1, rel=1e-10)

    def test_increases_with_gamma_squared(self, paper_params):
        """Minimum omega should scale as gamma^2."""
        omega_g1 = minimum_omega(0.1, paper_params["sigma"], 3)
        omega_g2 = minimum_omega(0.2, paper_params["sigma"], 3)
        assert omega_g2 == pytest.approx(4.0 * omega_g1, rel=1e-10)

    def test_exact_formula(self, paper_params):
        """Verify exact numerical value."""
        gamma, sigma, q = 0.1, 2.0, 3
        expected = 0.5 * gamma**2 * sigma**2 * q**2
        result   = minimum_omega(gamma, sigma, q)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_symmetric_in_q(self, paper_params):
        """Minimum omega should be symmetric in q (depends on q^2)."""
        for q in [1, 2, 5]:
            assert minimum_omega(0.1, paper_params["sigma"], q) == pytest.approx(
                minimum_omega(0.1, paper_params["sigma"], -q), rel=1e-12
            )


# ---------------------------------------------------------------------------
# Tests: sufficient_omega
# ---------------------------------------------------------------------------

class TestSufficientOmega:
    """Tests for the sufficient omega formula."""

    def test_exceeds_minimum_at_qmax(self, paper_params):
        """Sufficient omega should exceed minimum omega at q=q_max."""
        for gamma in [0.01, 0.1, 0.5]:
            for q_max in [5, 10]:
                omega_suff = sufficient_omega(gamma, paper_params["sigma"], q_max)
                omega_min  = minimum_omega(gamma, paper_params["sigma"], q_max)
                assert omega_suff > omega_min, \
                    f"Sufficient omega should exceed minimum at q_max={q_max}"

    def test_exact_formula(self, paper_params):
        """Verify exact numerical value."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        expected = 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2
        result   = sufficient_omega(gamma, sigma, q_max)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_increases_with_qmax(self, paper_params):
        """Sufficient omega should increase with q_max."""
        gamma = 0.1
        omegas = [sufficient_omega(gamma, paper_params["sigma"], q_max)
                  for q_max in [1, 5, 10, 20]]
        assert all(omegas[i] < omegas[i + 1] for i in range(len(omegas) - 1))

    def test_increases_with_gamma(self, paper_params):
        """Sufficient omega should increase with gamma."""
        q_max = 5
        omegas = [sufficient_omega(g, paper_params["sigma"], q_max)
                  for g in [0.01, 0.1, 0.5]]
        assert all(omegas[i] < omegas[i + 1] for i in range(len(omegas) - 1))


# ---------------------------------------------------------------------------
# Tests: verify_admissibility
# ---------------------------------------------------------------------------

class TestVerifyAdmissibility:
    """Tests for the admissibility verification function."""

    def test_sufficient_omega_always_admissible(self, paper_params):
        """Sufficient omega should always satisfy admissibility."""
        for gamma in [0.01, 0.1, 0.5]:
            for q_max in [5, 10]:
                assert verify_admissibility(gamma, paper_params["sigma"], q_max), \
                    f"Admissibility should hold for gamma={gamma}, q_max={q_max}"

    def test_all_inventory_states_covered(self, paper_params):
        """Admissibility should hold for all q in {-q_max, ..., q_max}."""
        gamma, q_max = 0.1, 5
        omega_suff = sufficient_omega(gamma, paper_params["sigma"], q_max)
        for q in range(-q_max, q_max + 1):
            omega_min = minimum_omega(gamma, paper_params["sigma"], q)
            assert omega_suff > omega_min, \
                f"Sufficient omega should dominate at q={q}"


# ---------------------------------------------------------------------------
# Tests: tabulate_omega_requirements
# ---------------------------------------------------------------------------

class TestTabulateOmegaRequirements:
    """Tests for the omega requirements table."""

    def test_returns_dataframe(self, paper_params, gamma_list):
        """Should return a DataFrame."""
        df = tabulate_omega_requirements(gamma_list, paper_params["sigma"], [5, 10])
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, paper_params, gamma_list):
        """Should have gamma_list * q_max_list rows."""
        q_max_list = [5, 10]
        df = tabulate_omega_requirements(gamma_list, paper_params["sigma"], q_max_list)
        assert len(df) == len(gamma_list) * len(q_max_list)

    def test_all_admissible(self, paper_params, gamma_list):
        """All rows should be admissible."""
        df = tabulate_omega_requirements(gamma_list, paper_params["sigma"], [5, 10])
        assert df["admissible"].all(), "All configurations should be admissible"

    def test_sufficient_exceeds_minimum(self, paper_params, gamma_list):
        """Sufficient omega should exceed minimum omega in all rows."""
        df = tabulate_omega_requirements(gamma_list, paper_params["sigma"], [5, 10])
        assert (df["omega_sufficient"] > df["omega_min_at_qmax"]).all()


# ---------------------------------------------------------------------------
# Tests: infinite_horizon_reservation_price
# ---------------------------------------------------------------------------

class TestInfiniteHorizonReservationPrice:
    """Tests for the infinite-horizon reservation price."""

    def test_zero_inventory_equals_mid(self, paper_params):
        """At q=0, reservation price equals mid-price."""
        omega = sufficient_omega(0.1, paper_params["sigma"], 5)
        r = infinite_horizon_reservation_price(100.0, 0, 0.1, paper_params["sigma"], omega)
        assert r == pytest.approx(100.0, rel=1e-10)

    def test_decreasing_in_inventory(self, paper_params):
        """Reservation price should decrease with inventory."""
        gamma = 0.1
        omega = sufficient_omega(gamma, paper_params["sigma"], 5)
        q_grid = range(-5, 6)
        r_vals = [
            infinite_horizon_reservation_price(100.0, q, gamma, paper_params["sigma"], omega)
            for q in q_grid
        ]
        assert all(r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1))

    def test_exact_formula(self, paper_params):
        """Verify exact numerical value."""
        gamma, sigma, q, omega = 0.1, 2.0, 3, 0.5
        expected = 100.0 - q * gamma * sigma**2 / (2.0 * omega)
        result   = infinite_horizon_reservation_price(100.0, q, gamma, sigma, omega)
        assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Tests: infinite_horizon_spread
# ---------------------------------------------------------------------------

class TestInfiniteHorizonSpread:
    """Tests for the infinite-horizon spread."""

    def test_positive_spread(self, paper_params):
        """Spread should be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            omega = sufficient_omega(gamma, paper_params["sigma"], 5)
            psi = infinite_horizon_spread(gamma, paper_params["sigma"], omega, paper_params["k"])
            assert psi > 0

    def test_decreases_with_omega(self, paper_params):
        """Spread should decrease as omega increases (less inventory risk)."""
        gamma = 0.1
        omegas = [0.1, 0.5, 1.0, 2.0]
        spreads = [
            infinite_horizon_spread(gamma, paper_params["sigma"], omega, paper_params["k"])
            for omega in omegas
        ]
        assert all(spreads[i] > spreads[i + 1] for i in range(len(spreads) - 1))

    def test_numerical_stability_small_gamma(self, paper_params):
        """Spread should be numerically stable for small gamma."""
        psi = infinite_horizon_spread(1e-8, paper_params["sigma"], 0.1, paper_params["k"])
        assert np.isfinite(psi)
        assert psi > 0


# ---------------------------------------------------------------------------
# Tests: mean_variance_reservation_price
# ---------------------------------------------------------------------------

class TestMeanVarianceReservationPrice:
    """Tests for the mean/variance reservation price."""

    def test_zero_inventory_equals_mid(self, paper_params):
        """At q=0, MV reservation price equals mid-price."""
        r = mean_variance_reservation_price(100.0, 0, 0.0, 0.1,
                                            paper_params["sigma"], paper_params["T"])
        assert r == pytest.approx(100.0, rel=1e-10)

    def test_coincides_with_cara_when_lambda_equals_gamma(self, paper_params):
        """MV and CARA formulas should coincide when lambda=gamma."""
        from src.exp1_analytical import reservation_price
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-3, 0, 3]:
                r_cara = reservation_price(100.0, q, 0.0, gamma,
                                           paper_params["sigma"], paper_params["T"])
                r_mv   = mean_variance_reservation_price(100.0, q, 0.0, gamma,
                                                         paper_params["sigma"], paper_params["T"])
                assert r_cara == pytest.approx(r_mv, rel=1e-10), \
                    f"CARA and MV should coincide for gamma={gamma}, q={q}"

    def test_decreasing_in_inventory(self, paper_params):
        """MV reservation price should decrease with inventory."""
        q_grid = range(-5, 6)
        r_vals = [
            mean_variance_reservation_price(100.0, q, 0.0, 0.1,
                                            paper_params["sigma"], paper_params["T"])
            for q in q_grid
        ]
        assert all(r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1))


# ---------------------------------------------------------------------------
# Tests: compare_cara_vs_mean_variance
# ---------------------------------------------------------------------------

class TestCompareCaravellaVsMeanVariance:
    """Tests for the CARA vs mean/variance comparison."""

    def test_returns_dataframe(self, paper_params, gamma_list):
        """Should return a DataFrame."""
        df = compare_cara_vs_mean_variance(
            sigma=paper_params["sigma"], T=paper_params["T"],
            k=paper_params["k"], gamma_list=gamma_list,
        )
        assert isinstance(df, pd.DataFrame)

    def test_diff_is_zero_when_lambda_equals_gamma(self, paper_params, gamma_list):
        """Difference between CARA and MV should be zero when lambda=gamma."""
        df = compare_cara_vs_mean_variance(
            sigma=paper_params["sigma"], T=paper_params["T"],
            k=paper_params["k"], gamma_list=gamma_list,
        )
        assert np.allclose(df["diff"].values, 0.0, atol=1e-10), \
            "CARA and MV should coincide when lambda=gamma"

    def test_required_columns(self, paper_params, gamma_list):
        """DataFrame should have required columns."""
        df = compare_cara_vs_mean_variance(
            sigma=paper_params["sigma"], T=paper_params["T"],
            k=paper_params["k"], gamma_list=gamma_list,
        )
        for col in ["gamma", "q", "r_CARA", "r_MV", "diff"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Tests: description functions
# ---------------------------------------------------------------------------

class TestDescriptionFunctions:
    """Tests for description/documentation functions."""

    def test_infinite_horizon_description_content(self):
        """Infinite-horizon description should contain key terms."""
        desc = infinite_horizon_description()
        assert "omega" in desc
        assert "stationary" in desc.lower() or "Stationary" in desc
        assert "HJB" in desc

    def test_admissibility_description_content(self):
        """Admissibility description should contain the condition."""
        desc = admissibility_condition_description()
        assert "omega" in desc
        assert "gamma" in desc
        assert "q_max" in desc

    def test_appendix_description_content(self):
        """Appendix description should mention multiplicative diffusion."""
        desc = appendix_multiplicative_model_description()
        assert "multiplicative" in desc.lower() or "Multiplicative" in desc
        assert "mean" in desc.lower()
        assert "variance" in desc.lower()


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in the infinite-horizon model."""

    def test_very_large_omega(self, paper_params):
        """Very large omega should give near-zero inventory adjustment."""
        r = infinite_horizon_reservation_price(100.0, 5, 0.1, paper_params["sigma"], 1e6)
        assert abs(r - 100.0) < 1e-3, "Very large omega should give near-zero adjustment"

    def test_very_small_omega(self, paper_params):
        """Very small omega should give large inventory adjustment."""
        r = infinite_horizon_reservation_price(100.0, 5, 0.1, paper_params["sigma"], 0.001)
        assert abs(r - 100.0) > 1.0, "Very small omega should give large adjustment"

    def test_admissibility_for_all_paper_gammas(self, paper_params):
        """Admissibility should hold for all paper gamma values."""
        for gamma in [0.01, 0.1, 0.5]:
            for q_max in [5, 10]:
                assert verify_admissibility(gamma, paper_params["sigma"], q_max)

    def test_omega_table_nonnegative(self, paper_params):
        """All omega values in the table should be non-negative."""
        df = tabulate_omega_requirements([0.01, 0.1, 0.5], paper_params["sigma"], [5, 10])
        assert (df["omega_sufficient"] >= 0).all()
        assert (df["omega_min_at_qmax"] >= 0).all()
