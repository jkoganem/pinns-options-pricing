"""Tests for Black-Scholes closed-form pricing."""

import pytest
import numpy as np
from multi_option.bs_closed_form import bs_price, bs_greeks


class TestBSPrice:
    """Test Black-Scholes pricing."""

    def test_call_price_atm(self):
        """Test ATM call pricing."""
        price = bs_price(100, 100, 0.05, 0, 0.2, 1.0, call=True)
        assert price == pytest.approx(10.450583572185565, rel=1e-6)

    def test_put_price_atm(self):
        """Test ATM put pricing."""
        price = bs_price(100, 100, 0.05, 0, 0.2, 1.0, call=False)
        assert price == pytest.approx(5.573526022256971, rel=1e-6)

    def test_put_call_parity(self):
        """Test put-call parity."""
        s0, K, r, q, sigma, T = 100, 100, 0.05, 0.02, 0.25, 1.0

        call_price = bs_price(s0, K, r, q, sigma, T, call=True)
        put_price = bs_price(s0, K, r, q, sigma, T, call=False)

        # Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
        parity_lhs = call_price - put_price
        parity_rhs = s0 * np.exp(-q * T) - K * np.exp(-r * T)

        assert abs(parity_lhs - parity_rhs) < 1e-10

    def test_deep_itm_call(self):
        """Test deep ITM call approaches intrinsic value."""
        price = bs_price(150, 100, 0.05, 0, 0.2, 0.01, call=True)
        intrinsic = 150 - 100
        assert abs(price - intrinsic) < 0.1

    def test_deep_otm_put(self):
        """Test deep OTM put approaches zero."""
        price = bs_price(150, 100, 0.05, 0, 0.2, 0.01, call=False)
        assert price < 0.01

    def test_zero_volatility_call(self):
        """Test zero volatility gives intrinsic value."""
        s0, K, r = 110, 100, 0.05
        price = bs_price(s0, K, r, 0, 0.001, 0.1, call=True)  # Very small vol
        expected = (s0 - K * np.exp(-r * 0.1))
        assert abs(price - expected) < 0.1


class TestBSGreeks:
    """Test Black-Scholes Greeks."""

    def test_call_delta_bounds(self):
        """Test call delta is between 0 and 1."""
        greeks = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, call=True)
        assert 0 <= greeks.delta <= 1

    def test_put_delta_bounds(self):
        """Test put delta is between -1 and 0."""
        greeks = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, call=False)
        assert -1 <= greeks.delta <= 0

    def test_gamma_positive(self):
        """Test gamma is always positive."""
        for call in [True, False]:
            greeks = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, call=call)
            assert greeks.gamma >= 0

    def test_atm_gamma_maximum(self):
        """Test gamma is maximum near ATM."""
        strikes = [80, 90, 100, 110, 120]
        gammas = []

        for K in strikes:
            greeks = bs_greeks(100, K, 0.05, 0, 0.2, 1.0, call=True)
            gammas.append(greeks.gamma)

        # ATM gamma should be close to the maximum (within one strike bucket)
        atm_idx = 2  # K=100
        max_idx = int(np.argmax(gammas))
        assert abs(atm_idx - max_idx) <= 1

    def test_vega_positive(self):
        """Test vega is always positive."""
        for call in [True, False]:
            greeks = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, call=call)
            assert greeks.vega >= 0

    def test_theta_negative_long_positions(self):
        """Test theta is generally negative for long positions."""
        # Call theta (without dividends)
        greeks_call = bs_greeks(100, 100, 0.05, 0, 0.2, 1.0, call=True)
        assert greeks_call.theta < 0

        # OTM put theta
        greeks_put = bs_greeks(100, 110, 0.05, 0, 0.2, 1.0, call=False)
        assert greeks_put.theta < 0
