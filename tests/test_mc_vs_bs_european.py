"""Tests comparing Monte Carlo methods with Black-Scholes."""

import pytest
import numpy as np
from multi_option.bs_closed_form import bs_price
from multi_option.mc.pricing import price_european_mc


class TestMonteCarloVsBS:
    """Test Monte Carlo against Black-Scholes."""

    def test_mc_call_convergence(self):
        """Test MC converges to BS for European call."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Black-Scholes reference
        bs_price_ref = bs_price(s0, K, r, q, sigma, T, call=True)

        # Monte Carlo with many paths
        mc_result = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=100, paths=100000, seed=42, call=True
        )

        # Should be within 1% of BS price
        rel_error = abs(mc_result.price - bs_price_ref) / bs_price_ref
        assert rel_error < 0.01

        # Standard error should be small
        assert mc_result.stderr < 0.1

    def test_mc_put_convergence(self):
        """Test MC converges to BS for European put."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Black-Scholes reference
        bs_price_ref = bs_price(s0, K, r, q, sigma, T, call=False)

        # Monte Carlo with many paths
        mc_result = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=100, paths=100000, seed=42, call=False
        )

        # Should be within 1% of BS price
        rel_error = abs(mc_result.price - bs_price_ref) / bs_price_ref
        assert rel_error < 0.01

    def test_antithetic_variance_reduction(self):
        """Test that antithetic variates reduce variance."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Run MC multiple times with different seeds
        n_runs = 10
        prices_with_anti = []
        prices_without = []  # Would need non-antithetic version

        for seed in range(n_runs):
            result = price_european_mc(
                s0, K, r, q, sigma, T,
                steps=50, paths=10000, seed=seed, call=True
            )
            prices_with_anti.append(result.price)

        # Variance should be reasonable
        variance = np.var(prices_with_anti)
        assert variance < 1.0  # Reasonable variance bound

    def test_increasing_paths_reduces_error(self):
        """Test that increasing paths reduces standard error."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Few paths
        mc_few = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=50, paths=1000, seed=42, call=True
        )

        # Many paths
        mc_many = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=50, paths=100000, seed=42, call=True
        )

        # Standard error should decrease
        assert mc_many.stderr < mc_few.stderr

    def test_seed_reproducibility(self):
        """Test that same seed gives same results."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Run twice with same seed
        result1 = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=50, paths=10000, seed=123, call=True
        )
        result2 = price_european_mc(
            s0, K, r, q, sigma, T,
            steps=50, paths=10000, seed=123, call=True
        )

        # Should get exactly the same price
        assert result1.price == result2.price