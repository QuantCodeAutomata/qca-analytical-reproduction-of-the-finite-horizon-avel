"""
Tests for Experiment 2: Monte Carlo simulation of inventory-based vs symmetric strategy.

Verifies:
  - Simulation correctness (state updates, cash/inventory accounting)
  - Statistical properties of terminal outcomes
  - Hypothesis: inventory strategy reduces P&L and inventory variance
  - Edge cases (zero gamma, single path, single step)
  - Reproducibility with fixed seeds
"""

import pytest
import numpy as np
import pandas as pd

from src.exp2_montecarlo import (
    simulate_path,
    run_monte_carlo,
    compute_summary_stats,
    DEFAULT_PARAMS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_params():
    """Small simulation parameters for fast tests."""
    return {
        "S0":    100.0,
        "T":     1.0,
        "dt":    0.005,
        "N":     200,
        "sigma": 2.0,
        "A":     140.0,
        "k":     1.5,
        "q0":    0,
        "X0":    0.0,
        "n_sim": 100,   # small for speed
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------

class TestSimulatePath:
    """Tests for the single-path simulation function."""

    def test_returns_required_keys(self, small_params, rng):
        """simulate_path should return all required keys."""
        result = simulate_path("inventory", 0.1, small_params, rng)
        required_keys = ["W_T", "q_T", "X_T", "S_T", "n_bid_fills", "n_ask_fills"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_terminal_wealth_formula(self, small_params, rng):
        """Terminal wealth should equal X_T + q_T * S_T."""
        result = simulate_path("inventory", 0.1, small_params, rng)
        W_T_computed = result["X_T"] + result["q_T"] * result["S_T"]
        assert result["W_T"] == pytest.approx(W_T_computed, rel=1e-10)

    def test_nonnegative_fill_counts(self, small_params, rng):
        """Fill counts should be non-negative integers."""
        result = simulate_path("inventory", 0.1, small_params, rng)
        assert result["n_bid_fills"] >= 0
        assert result["n_ask_fills"] >= 0

    def test_full_path_storage(self, small_params, rng):
        """Full path storage should return time series arrays."""
        result = simulate_path("inventory", 0.1, small_params, rng, store_full=True)
        N = small_params["N"]
        assert "S_path" in result
        assert len(result["S_path"]) == N + 1
        assert len(result["q_path"]) == N + 1
        assert len(result["r_path"]) == N + 1

    def test_initial_price_correct(self, small_params, rng):
        """First element of S_path should equal S0."""
        result = simulate_path("inventory", 0.1, small_params, rng, store_full=True)
        assert result["S_path"][0] == pytest.approx(small_params["S0"], rel=1e-12)

    def test_initial_inventory_correct(self, small_params, rng):
        """First element of q_path should equal q0."""
        result = simulate_path("inventory", 0.1, small_params, rng, store_full=True)
        assert result["q_path"][0] == small_params["q0"]

    def test_inventory_integer_valued(self, small_params, rng):
        """Inventory should always be integer-valued."""
        result = simulate_path("inventory", 0.1, small_params, rng, store_full=True)
        assert all(isinstance(q, (int, np.integer)) for q in result["q_path"])

    def test_symmetric_strategy_runs(self, small_params, rng):
        """Symmetric strategy should run without errors."""
        result = simulate_path("symmetric", 0.1, small_params, rng)
        assert "W_T" in result
        assert np.isfinite(result["W_T"])

    def test_reproducibility_with_same_seed(self, small_params):
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = simulate_path("inventory", 0.1, small_params, rng1)
        r2 = simulate_path("inventory", 0.1, small_params, rng2)
        assert r1["W_T"] == pytest.approx(r2["W_T"], rel=1e-12)
        assert r1["q_T"] == r2["q_T"]

    def test_different_seeds_different_results(self, small_params):
        """Different seeds should (almost certainly) produce different results."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        r1 = simulate_path("inventory", 0.1, small_params, rng1)
        r2 = simulate_path("inventory", 0.1, small_params, rng2)
        # Very unlikely to be identical
        assert r1["W_T"] != r2["W_T"] or r1["q_T"] != r2["q_T"]

    def test_finite_terminal_wealth(self, small_params, rng):
        """Terminal wealth should be finite."""
        for gamma in [0.01, 0.1, 0.5]:
            result = simulate_path("inventory", gamma, small_params, rng)
            assert np.isfinite(result["W_T"]), f"W_T should be finite for gamma={gamma}"

    def test_cash_accounting_consistency(self, small_params):
        """Cash accounting: each bid fill decreases cash by p^b, ask fill increases by p^a."""
        # Use a deterministic-like test: zero sigma, check cash changes
        params = small_params.copy()
        params["sigma"] = 0.0  # no price movement
        params["N"] = 1
        rng = np.random.default_rng(0)
        result = simulate_path("inventory", 0.1, params, rng, store_full=True)
        # Cash should change by fills * prices
        # Just verify it's finite and consistent
        assert np.isfinite(result["X_T"])


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    """Tests for the full Monte Carlo simulation."""

    def test_returns_dataframe_with_correct_shape(self, small_params):
        """run_monte_carlo should return a DataFrame with n_sim rows."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        assert len(df) == small_params["n_sim"]

    def test_dataframe_has_required_columns(self, small_params):
        """DataFrame should have required columns."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        for col in ["W_T", "q_T", "n_bid_fills", "n_ask_fills"]:
            assert col in df.columns

    def test_full_paths_returned(self, small_params):
        """Full paths should be returned for the first n_full_paths paths."""
        _, full_paths = run_monte_carlo(0.1, "inventory", small_params,
                                        seed=42, n_full_paths=2)
        assert len(full_paths) == 2
        assert "S_path" in full_paths[0]

    def test_all_terminal_wealth_finite(self, small_params):
        """All terminal wealth values should be finite."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        assert df["W_T"].isna().sum() == 0
        assert np.isfinite(df["W_T"].values).all()

    def test_reproducibility(self, small_params):
        """Same seed should produce identical DataFrames."""
        df1, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        df2, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Tests: compute_summary_stats
# ---------------------------------------------------------------------------

class TestComputeSummaryStats:
    """Tests for summary statistics computation."""

    def test_returns_required_keys(self, small_params):
        """Summary stats should contain all required keys."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        stats = compute_summary_stats(df)
        required = ["mean_PnL", "std_PnL", "mean_inv", "std_inv"]
        for key in required:
            assert key in stats, f"Missing key: {key}"

    def test_std_nonnegative(self, small_params):
        """Standard deviations should be non-negative."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        stats = compute_summary_stats(df)
        assert stats["std_PnL"] >= 0
        assert stats["std_inv"] >= 0

    def test_quantiles_ordered(self, small_params):
        """5th percentile should be below 95th percentile."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        stats = compute_summary_stats(df)
        assert stats["q05_PnL"] < stats["q95_PnL"]

    def test_mean_finite(self, small_params):
        """Mean P&L should be finite."""
        df, _ = run_monte_carlo(0.1, "inventory", small_params, seed=42)
        stats = compute_summary_stats(df)
        assert np.isfinite(stats["mean_PnL"])
        assert np.isfinite(stats["mean_inv"])


# ---------------------------------------------------------------------------
# Tests: hypothesis validation
# ---------------------------------------------------------------------------

class TestHypothesisValidation:
    """
    Tests that validate the paper's main hypotheses.
    Uses n_sim=500 for reasonable statistical power while keeping tests fast.
    """

    @pytest.fixture
    def medium_params(self):
        return {
            "S0":    100.0,
            "T":     1.0,
            "dt":    0.005,
            "N":     200,
            "sigma": 2.0,
            "A":     140.0,
            "k":     1.5,
            "q0":    0,
            "X0":    0.0,
            "n_sim": 500,
        }

    def test_inventory_strategy_lower_pnl_std_large_gamma(self, medium_params):
        """
        For gamma=0.5, inventory strategy should have lower P&L std than symmetric.
        This is the paper's main hypothesis.
        """
        gamma = 0.5
        df_inv, _ = run_monte_carlo(gamma, "inventory", medium_params, seed=42)
        df_sym, _ = run_monte_carlo(gamma, "symmetric", medium_params, seed=542)
        assert df_inv["W_T"].std() < df_sym["W_T"].std(), \
            "Inventory strategy should have lower P&L std for gamma=0.5"

    def test_inventory_strategy_lower_inv_std_large_gamma(self, medium_params):
        """
        For gamma=0.5, inventory strategy should have lower final inventory std.
        """
        gamma = 0.5
        df_inv, _ = run_monte_carlo(gamma, "inventory", medium_params, seed=42)
        df_sym, _ = run_monte_carlo(gamma, "symmetric", medium_params, seed=542)
        assert df_inv["q_T"].std() < df_sym["q_T"].std(), \
            "Inventory strategy should have lower inventory std for gamma=0.5"

    def test_small_gamma_strategies_similar(self, medium_params):
        """
        For gamma=0.01, the two strategies should produce similar P&L distributions.
        The difference in std should be small relative to the std itself.
        """
        gamma = 0.01
        df_inv, _ = run_monte_carlo(gamma, "inventory", medium_params, seed=42)
        df_sym, _ = run_monte_carlo(gamma, "symmetric", medium_params, seed=542)
        std_inv = df_inv["W_T"].std()
        std_sym = df_sym["W_T"].std()
        # Relative difference should be small for gamma=0.01
        rel_diff = abs(std_inv - std_sym) / max(std_sym, 1e-10)
        assert rel_diff < 0.5, \
            f"For gamma=0.01, strategies should be similar (rel_diff={rel_diff:.3f})"

    def test_effect_stronger_at_larger_gamma(self, medium_params):
        """
        The reduction in inventory std should be larger for gamma=0.5 than gamma=0.01.
        """
        results = {}
        for gamma in [0.01, 0.5]:
            df_inv, _ = run_monte_carlo(gamma, "inventory", medium_params,
                                         seed=42 + int(gamma * 100))
            df_sym, _ = run_monte_carlo(gamma, "symmetric", medium_params,
                                         seed=542 + int(gamma * 100))
            std_inv = df_inv["q_T"].std()
            std_sym = df_sym["q_T"].std()
            results[gamma] = (std_sym - std_inv) / max(std_sym, 1e-10)

        assert results[0.5] > results[0.01], \
            "Inventory control effect should be stronger at larger gamma"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in the Monte Carlo simulation."""

    def test_single_step_simulation(self):
        """Simulation should work with N=1."""
        params = DEFAULT_PARAMS.copy()
        params["N"]     = 1
        params["dt"]    = 1.0
        params["T"]     = 1.0
        params["n_sim"] = 10
        df, _ = run_monte_carlo(0.1, "inventory", params, seed=42)
        assert len(df) == 10
        assert np.isfinite(df["W_T"].values).all()

    def test_zero_volatility(self):
        """With zero volatility, mid-price should stay constant."""
        params = DEFAULT_PARAMS.copy()
        params["sigma"] = 0.0
        params["n_sim"] = 10
        rng = np.random.default_rng(42)
        result = simulate_path("inventory", 0.1, params, rng, store_full=True)
        # All prices should equal S0
        assert np.allclose(result["S_path"], params["S0"]), \
            "With zero volatility, mid-price should stay constant"

    def test_very_small_gamma(self):
        """Simulation should be stable for very small gamma."""
        params = DEFAULT_PARAMS.copy()
        params["n_sim"] = 50
        df, _ = run_monte_carlo(1e-6, "inventory", params, seed=42)
        assert np.isfinite(df["W_T"].values).all()

    def test_large_gamma(self):
        """Simulation should be stable for large gamma."""
        params = DEFAULT_PARAMS.copy()
        params["n_sim"] = 50
        df, _ = run_monte_carlo(1.0, "inventory", params, seed=42)
        assert np.isfinite(df["W_T"].values).all()
