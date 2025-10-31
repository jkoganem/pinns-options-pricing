"""Black-Scholes closed-form option pricing formulas.

This module implements the analytical Black-Scholes formulas for
European option pricing and Greeks computation.
"""

import numpy as np
from scipy import stats
from multi_option.datatypes import GreeksResult


def bs_price(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    call: bool
) -> float:
    """Compute European option price using Black-Scholes formula.

    Args:
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        call: True for call, False for put.

    Returns:
        Option price.
    """
    if T <= 0:
        # At expiration
        if call:
            return max(s0 - K, 0.0)
        else:
            return max(K - s0, 0.0)

    # Calculate d1 and d2
    d1 = (np.log(s0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard normal CDF
    N = stats.norm.cdf

    if call:
        # Call option price
        price = s0 * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    else:
        # Put option price
        price = K * np.exp(-r * T) * N(-d2) - s0 * np.exp(-q * T) * N(-d1)

    return float(price)


def bs_greeks(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    call: bool
) -> GreeksResult:
    """Compute option Greeks using Black-Scholes formulas.

    Args:
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        call: True for call, False for put.

    Returns:
        GreeksResult with all Greeks.
    """
    if T <= 0:
        # At expiration, most Greeks are zero or undefined
        if call:
            delta = 1.0 if s0 > K else 0.0
        else:
            delta = -1.0 if s0 < K else 0.0

        return GreeksResult(
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0
        )

    # Calculate d1 and d2
    d1 = (np.log(s0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard normal CDF and PDF
    N = stats.norm.cdf
    n = stats.norm.pdf

    # Compute Greeks
    # Delta: dV/dS
    if call:
        delta = np.exp(-q * T) * N(d1)
    else:
        delta = -np.exp(-q * T) * N(-d1)

    # Gamma: d2V/dS2
    gamma = np.exp(-q * T) * n(d1) / (s0 * sigma * np.sqrt(T))

    # Theta: -dV/dT (negative of time derivative)
    term1 = -s0 * n(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
    if call:
        term2 = q * s0 * np.exp(-q * T) * N(d1)
        term3 = -r * K * np.exp(-r * T) * N(d2)
        theta = (term1 - term2 + term3) / 365  # Convert to per-day theta
    else:
        term2 = -q * s0 * np.exp(-q * T) * N(-d1)
        term3 = r * K * np.exp(-r * T) * N(-d2)
        theta = (term1 - term2 + term3) / 365  # Convert to per-day theta

    # Vega: dV/dsigma
    vega = s0 * np.exp(-q * T) * n(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% vol move

    # Rho: dV/dr
    if call:
        rho = K * T * np.exp(-r * T) * N(d2) / 100  # Divide by 100 for 1% rate move
    else:
        rho = -K * T * np.exp(-r * T) * N(-d2) / 100  # Divide by 100 for 1% rate move

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho)
    )


def bs_vega_numeric(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    call: bool,
    dvol: float = 0.001
) -> float:
    """Compute vega using finite difference for verification.

    Args:
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        call: True for call, False for put.
        dvol: Volatility bump for finite difference.

    Returns:
        Vega (per 1% volatility move).
    """
    priceup = bs_price(s0, K, r, q, sigma + dvol, T, call)
    price_down = bs_price(s0, K, r, q, sigma - dvol, T, call)
    vega = (priceup - price_down) / (2 * dvol) / 100  # Per 1% move
    return float(vega)