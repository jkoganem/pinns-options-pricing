"""Tests comparing finite-difference PDE methods with Black-Scholes."""

import pytest
import numpy as np
from multi_option.bs_closed_form import bs_price
from multi_option.fd_pde.cn_solver import price_european_cn


class TestCrankNicolsonVsBS:
    """Test Crank-Nicolson against Black-Scholes."""

    def test_european_call_convergence(self):
        """Test CN converges to BS for European call."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Black-Scholes reference
        bs_price_ref = bs_price(s0, K, r, q, sigma, T, call=True)

        # Crank-Nicolson with fine grid
        cn_price = price_european_cn(
            s0, K, r, q, sigma, T,
            s_max=300, ns=201, nt=200, call=True
        )

        # Should be within 0.1% of BS price
        rel_error = abs(cn_price - bs_price_ref) / bs_price_ref
        assert rel_error < 0.001

    def test_european_put_convergence(self):
        """Test CN converges to BS for European put."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0

        # Black-Scholes reference
        bs_price_ref = bs_price(s0, K, r, q, sigma, T, call=False)

        # Crank-Nicolson with fine grid
        cn_price = price_european_cn(
            s0, K, r, q, sigma, T,
            s_max=300, ns=201, nt=200, call=False
        )

        # Should be within 0.1% of BS price
        rel_error = abs(cn_price - bs_price_ref) / bs_price_ref
        assert rel_error < 0.001

    def test_grid_refinement_convergence(self):
        """Test that refining grid improves accuracy."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0, 0.2, 1.0
        bs_price_ref = bs_price(s0, K, r, q, sigma, T, call=True)

        # Coarse grid
        coarse_price = price_european_cn(
            s0, K, r, q, sigma, T,
            s_max=300, ns=51, nt=50, call=True
        )
        coarse_error = abs(coarse_price - bs_price_ref)

        # Fine grid
        fine_price = price_european_cn(
            s0, K, r, q, sigma, T,
            s_max=300, ns=201, nt=200, call=True
        )
        fine_error = abs(fine_price - bs_price_ref)

        # Fine grid should be more accurate
        assert fine_error < coarse_error

    def test_various_strikes(self):
        """Test CN accuracy for various strikes."""
        s0, r, q, sigma, T = 100, 0.05, 0, 0.2, 1.0
        strikes = [80, 90, 100, 110, 120]

        for K in strikes:
            # Call option
            bs_call = bs_price(s0, K, r, q, sigma, T, call=True)
            cn_call = price_european_cn(
                s0, K, r, q, sigma, T,
                s_max=300, ns=151, nt=150, call=True
            )
            rel_error_call = abs(cn_call - bs_call) / bs_call
            assert rel_error_call < 0.01  # Within 1%

            # Put option
            bs_put = bs_price(s0, K, r, q, sigma, T, call=False)
            cn_put = price_european_cn(
                s0, K, r, q, sigma, T,
                s_max=300, ns=151, nt=150, call=False
            )
            rel_error_put = abs(cn_put - bs_put) / bs_put
            assert rel_error_put < 0.01  # Within 1%