"""Finite difference methods for computing Greeks.

This module implements numerical differentiation methods for
computing option Greeks when analytical formulas are not available.
"""

from typing import Callable, Dict, Optional
import numpy as np

from multi_option.datatypes import GreeksResult


def compute_greeks_fd(
    price_func: Callable,
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    ds: float = 0.01,
    dsigma: float = 0.001,
    dr: float = 0.0001,
    dt: float = 0.001,
    **kwargs
) -> GreeksResult:
    """Compute Greeks using finite differences.

    Args:
        price_func: Function that returns option price given parameters.
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        ds: Stock price bump for delta/gamma.
        dsigma: Volatility bump for vega.
        dr: Interest rate bump for rho.
        dt: Time bump for theta.
        **kwargs: Additional arguments for price_func.

    Returns:
        GreeksResult with all Greeks.
    """
    # Base price
    price_base = price_func(s0, K, r, q, sigma, T, **kwargs)

    # Delta: dV/dS (central difference)
    priceup = price_func(s0 + ds, K, r, q, sigma, T, **kwargs)
    price_down = price_func(s0 - ds, K, r, q, sigma, T, **kwargs)
    delta = (priceup - price_down) / (2 * ds)

    # Gamma: d2V/dS2 (central second difference)
    gamma = (priceup - 2 * price_base + price_down) / (ds ** 2)

    # Vega: dV/d_sigma (forward difference)
    price_sigmaup = price_func(s0, K, r, q, sigma + dsigma, T, **kwargs)
    vega = (price_sigmaup - price_base) / dsigma / 100  # Per 1% vol move

    # Rho: dV/dr (forward difference)
    price_rup = price_func(s0, K, r + dr, q, sigma, T, **kwargs)
    rho = (price_rup - price_base) / dr / 100  # Per 1% rate move

    # Theta: -dV/dT (backward difference)
    if T > dt:
        price_t_minus = price_func(s0, K, r, q, sigma, T - dt, **kwargs)
        theta = -(price_base - price_t_minus) / dt / 365  # Per day
    else:
        theta = 0.0

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho)
    )


def compute_delta_fd(
    price_func: Callable,
    s0: float,
    ds: float = 0.01,
    method: str = 'central',
    **price_kwargs
) -> float:
    """Compute delta using finite differences.

    Args:
        price_func: Pricing function.
        s0: Current stock price.
        ds: Price bump.
        method: 'central', 'forward', or 'backward'.
        **price_kwargs: Arguments for price_func.

    Returns:
        Delta value.
    """
    if method == 'central':
        priceup = price_func(s0 + ds, **price_kwargs)
        price_down = price_func(s0 - ds, **price_kwargs)
        delta = (priceup - price_down) / (2 * ds)
    elif method == 'forward':
        price_base = price_func(s0, **price_kwargs)
        priceup = price_func(s0 + ds, **price_kwargs)
        delta = (priceup - price_base) / ds
    elif method == 'backward':
        price_base = price_func(s0, **price_kwargs)
        price_down = price_func(s0 - ds, **price_kwargs)
        delta = (price_base - price_down) / ds
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(delta)


def compute_gamma_fd(
    price_func: Callable,
    s0: float,
    ds: float = 0.01,
    **price_kwargs
) -> float:
    """Compute gamma using finite differences.

    Args:
        price_func: Pricing function.
        s0: Current stock price.
        ds: Price bump.
        **price_kwargs: Arguments for price_func.

    Returns:
        Gamma value.
    """
    price_base = price_func(s0, **price_kwargs)
    priceup = price_func(s0 + ds, **price_kwargs)
    price_down = price_func(s0 - ds, **price_kwargs)
    gamma = (priceup - 2 * price_base + price_down) / (ds ** 2)

    return float(gamma)


def compute_vega_fd(
    price_func: Callable,
    sigma: float,
    dsigma: float = 0.001,
    **price_kwargs
) -> float:
    """Compute vega using finite differences.

    Args:
        price_func: Pricing function.
        sigma: Current volatility.
        dsigma: Volatility bump.
        **price_kwargs: Arguments for price_func.

    Returns:
        Vega value (per 1% vol move).
    """
    price_base = price_func(sigma=sigma, **price_kwargs)
    priceup = price_func(sigma=sigma + dsigma, **price_kwargs)
    vega = (priceup - price_base) / dsigma / 100

    return float(vega)


def compute_theta_fd(
    price_func: Callable,
    T: float,
    dt: float = 0.001,
    **price_kwargs
) -> float:
    """Compute theta using finite differences.

    Args:
        price_func: Pricing function.
        T: Time to maturity.
        dt: Time bump.
        **price_kwargs: Arguments for price_func.

    Returns:
        Theta value (per day).
    """
    if T <= dt:
        return 0.0

    price_base = price_func(T=T, **price_kwargs)
    price_minus = price_func(T=T - dt, **price_kwargs)
    theta = -(price_base - price_minus) / dt / 365

    return float(theta)


def compute_rho_fd(
    price_func: Callable,
    r: float,
    dr: float = 0.0001,
    **price_kwargs
) -> float:
    """Compute rho using finite differences.

    Args:
        price_func: Pricing function.
        r: Risk-free rate.
        dr: Rate bump.
        **price_kwargs: Arguments for price_func.

    Returns:
        Rho value (per 1% rate move).
    """
    price_base = price_func(r=r, **price_kwargs)
    priceup = price_func(r=r + dr, **price_kwargs)
    rho = (priceup - price_base) / dr / 100

    return float(rho)


def adaptive_finite_difference(
    price_func: Callable,
    param: float,
    param_name: str,
    base_bump: float = 0.01,
    tol: float = 1e-6,
    max_refine: int = 5,
    **price_kwargs
) -> float:
    """Compute derivative using adaptive finite differences.

    Uses Richardson extrapolation for improved accuracy.

    Args:
        price_func: Pricing function.
        param: Parameter value.
        param_name: Name of parameter to differentiate.
        base_bump: Initial bump size.
        tol: Convergence tolerance.
        max_refine: Maximum refinement iterations.
        **price_kwargs: Arguments for price_func.

    Returns:
        Derivative estimate.
    """
    # Richardson extrapolation table
    R = np.zeros((max_refine, max_refine))

    for i in range(max_refine):
        h = base_bump / (2 ** i)

        # Central difference
        kwargsup = {**price_kwargs, param_name: param + h}
        kwargs_down = {**price_kwargs, param_name: param - h}

        priceup = price_func(**kwargsup)
        price_down = price_func(**kwargs_down)

        R[i, 0] = (priceup - price_down) / (2 * h)

        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

        # Check convergence
        if i > 0:
            error = abs(R[i, i] - R[i-1, i-1])
            if error < tol:
                return float(R[i, i])

    return float(R[-1, -1])


def compute_cross_gammas(
    price_func: Callable,
    s0: float,
    sigma: float,
    r: float,
    ds: float = 0.01,
    dsigma: float = 0.001,
    dr: float = 0.0001,
    **price_kwargs
) -> Dict[str, float]:
    """Compute cross-derivatives (Vanna, Charm, etc.).

    Args:
        price_func: Pricing function.
        s0: Stock price.
        sigma: Volatility.
        r: Risk-free rate.
        ds: Stock price bump.
        dsigma: Volatility bump.
        dr: Rate bump.
        **price_kwargs: Arguments for price_func.

    Returns:
        Dictionary with cross-derivatives.
    """
    # Vanna: d2V/dS d_sigma
    price_base = price_func(s0=s0, sigma=sigma, r=r, **price_kwargs)
    price_sup_sigup = price_func(s0=s0 + ds, sigma=sigma + dsigma, r=r, **price_kwargs)
    price_sup = price_func(s0=s0 + ds, sigma=sigma, r=r, **price_kwargs)
    price_sigup = price_func(s0=s0, sigma=sigma + dsigma, r=r, **price_kwargs)

    vanna = (price_sup_sigup - price_sup - price_sigup + price_base) / (ds * dsigma)

    # Charm (Delta decay): d2V/dS dT
    if 'T' in price_kwargs and price_kwargs['T'] > 0.001:
        T = price_kwargs['T']
        dt = 0.001

        price_sup_t_minus = price_func(s0=s0 + ds, T=T - dt, sigma=sigma, r=r,
                                       **{k: v for k, v in price_kwargs.items() if k != 'T'})
        price_t_minus = price_func(s0=s0, T=T - dt, sigma=sigma, r=r,
                                  **{k: v for k, v in price_kwargs.items() if k != 'T'})

        charm = -(price_sup_t_minus - price_sup - price_t_minus + price_base) / (ds * dt)
    else:
        charm = 0.0

    # Vomma (Volga): d2V/d_sigma2
    price_sigup2 = price_func(s0=s0, sigma=sigma + 2*dsigma, r=r, **price_kwargs)
    vomma = (price_sigup2 - 2 * price_sigup + price_base) / (dsigma ** 2)

    return {
        'vanna': float(vanna),
        'charm': float(charm),
        'vomma': float(vomma)
    }