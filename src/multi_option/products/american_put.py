"""American put option pricing methods.

This module provides pricing methods for American put options,
primarily using finite-difference PDE methods with PSOR.
"""

from typing import Dict, Optional
import numpy as np

from multi_option.datatypes import EngineConfig, PriceResult
from multi_option.fd_pde.lcp_psor import price_american_put_at_spot
from multi_option.bs_closed_form import bs_price


def price_american_put_methods(cfg: EngineConfig) -> Dict[str, PriceResult]:
    """Price American put using available methods.

    Args:
        cfg: Engine configuration.

    Returns:
        Dictionary mapping method names to PriceResult objects.
    """
    results = {}

    # PDE with PSOR (primary method for American options)
    pde_price = price_american_put_at_spot(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )

    results['pde'] = PriceResult(
        method='pde',
        product='american_put',
        price=pde_price,
        stderr=0.0,
        meta={
            'ns': float(cfg.ns),
            'nt': float(cfg.nt),
            'solver': 'PSOR'
        }
    )

    # Binomial tree benchmark (optional)
    binomial_price = price_american_put_binomial(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        steps=min(cfg.nt, 500)  # Limit steps for performance
    )

    results['binomial'] = PriceResult(
        method='binomial',
        product='american_put',
        price=binomial_price,
        stderr=0.0,
        meta={'steps': float(min(cfg.nt, 500))}
    )

    # European put for comparison (lower bound)
    european_price = bs_price(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T, call=False
    )

    results['european_bound'] = PriceResult(
        method='bs',
        product='european_put',
        price=european_price,
        stderr=0.0,
        meta={'bound_type': 'lower'}
    )

    # Calculate early exercise premium
    early_ex_premium = pde_price - european_price
    results['early_exercise_premium'] = PriceResult(
        method='analytical',
        product='premium',
        price=early_ex_premium,
        stderr=0.0,
        meta={'american_price': pde_price, 'european_price': european_price}
    )

    return results


def price_american_put_binomial(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int = 100
) -> float:
    """Price American put using binomial tree.

    Cox-Ross-Rubinstein binomial model with early exercise.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.

    Returns:
        American put option price.
    """
    dt = T / steps

    # Binomial parameters
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

    # Discount factor per step
    disc = np.exp(-r * dt)

    # Initialize stock price tree
    # We only need to store the final layer and work backwards
    S = np.zeros(steps + 1)
    for j in range(steps + 1):
        S[j] = s0 * (u ** j) * (d ** (steps - j))

    # Initialize option values at maturity
    V = np.maximum(K - S, 0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        # Update stock prices for this time step
        for j in range(i + 1):
            S[j] = s0 * (u ** j) * (d ** (i - j))

        # Calculate option values
        for j in range(i + 1):
            # Continuation value (discounted expected value)
            continuation = disc * (p * V[j + 1] + (1 - p) * V[j])

            # Immediate exercise value
            exercise = max(K - S[j], 0)

            # American option value is max of continuation and exercise
            V[j] = max(continuation, exercise)

    return float(V[0])


def compute_american_put_greeks_fd(
    cfg: EngineConfig,
    ds: float = 0.01,
    dsigma: float = 0.001,
    dr: float = 0.0001,
    dt: float = 0.001
) -> Dict[str, float]:
    """Compute American put Greeks using finite differences.

    Args:
        cfg: Engine configuration.
        ds: Stock price bump for delta/gamma.
        dsigma: Volatility bump for vega.
        dr: Interest rate bump for rho.
        dt: Time bump for theta.

    Returns:
        Dictionary with Greek values.
    """
    # Base price
    price_base = price_american_put_at_spot(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )

    # Delta: dV/dS
    priceup = price_american_put_at_spot(
        cfg.s0 + ds, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )
    price_down = price_american_put_at_spot(
        cfg.s0 - ds, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )
    delta = (priceup - price_down) / (2 * ds)

    # Gamma: d2V/dS2
    gamma = (priceup - 2 * price_base + price_down) / (ds ** 2)

    # Vega: dV/d_sigma
    price_sigmaup = price_american_put_at_spot(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma + dsigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )
    vega = (price_sigmaup - price_base) / dsigma / 100  # Per 1% vol move

    # Rho: dV/dr
    price_rup = price_american_put_at_spot(
        cfg.s0, cfg.K, cfg.r + dr, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt
    )
    rho = (price_rup - price_base) / dr / 100  # Per 1% rate move

    # Theta: -dV/dT
    if cfg.T > dt:
        price_t_minus = price_american_put_at_spot(
            cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T - dt,
            cfg.s_max, cfg.ns, cfg.nt
        )
        theta = -(price_base - price_t_minus) / dt / 365  # Per day
    else:
        theta = 0.0

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'vega': float(vega),
        'rho': float(rho),
        'theta': float(theta)
    }


def check_early_exercise_optimality(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int
) -> Dict[str, float]:
    """Check if early exercise is optimal at current state.

    Args:
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.

    Returns:
        Dictionary with exercise analysis.
    """
    # American put price
    american_price = price_american_put_at_spot(
        s0, K, r, q, sigma, T, s_max, ns, nt
    )

    # Immediate exercise value
    intrinsic_value = max(K - s0, 0)

    # European put price (continuation value approximation)
    european_price = bs_price(s0, K, r, q, sigma, T, call=False)

    # Early exercise premium
    early_ex_premium = american_price - european_price

    # Time value
    time_value = american_price - intrinsic_value

    return {
        'american_price': float(american_price),
        'intrinsic_value': float(intrinsic_value),
        'time_value': float(time_value),
        'european_price': float(european_price),
        'early_exercise_premium': float(early_ex_premium),
        'should_exercise': bool(time_value < 0.01),  # Small threshold
        'moneyness': float(s0 / K)
    }