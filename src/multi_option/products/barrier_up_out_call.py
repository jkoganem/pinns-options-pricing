"""Barrier up-and-out call option pricing.

This module implements pricing methods for barrier options,
specifically up-and-out calls.
"""

from typing import Dict, Optional
import numpy as np

from multi_option.datatypes import EngineConfig, PriceResult
from multi_option.mc.pricing import price_barrierup_out_call_mc


def price_barrier_methods(cfg: EngineConfig) -> Dict[str, PriceResult]:
    """Price barrier up-and-out call using available methods.

    Args:
        cfg: Engine configuration.

    Returns:
        Dictionary mapping method names to PriceResult objects.
    """
    if cfg.barrier is None:
        raise ValueError("Barrier level must be specified for barrier options")

    if cfg.barrier <= cfg.s0:
        raise ValueError("Barrier must be above initial stock price for up-and-out")

    if cfg.barrier <= cfg.K:
        raise ValueError("Barrier must be above strike for up-and-out call")

    results = {}

    # Monte Carlo (primary method for barrier options)
    mc_result = price_barrierup_out_call_mc(
        cfg.s0, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    results['mc'] = mc_result

    # Closed-form approximation (if available)
    cf_price = barrier_call_closed_form(
        cfg.s0, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T
    )
    if cf_price is not None:
        results['closed_form'] = PriceResult(
            method='analytical',
            product='barrierup_out_call',
            price=cf_price,
            stderr=0.0,
            meta={'barrier': cfg.barrier, 'formula': 'Merton-Reiner'}
        )

    # PDE method (optional - more complex implementation)
    # pde_price = price_barrier_pde(cfg)
    # results['pde'] = pde_price

    return results


def barrier_call_closed_form(
    s0: float,
    K: float,
    B: float,
    r: float,
    q: float,
    sigma: float,
    T: float
) -> Optional[float]:
    """Closed-form price for up-and-out call (when available).

    Uses the Merton-Reiner formula for continuously monitored barriers.

    Args:
        s0: Initial stock price.
        K: Strike price.
        B: Barrier level.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.

    Returns:
        Option price or None if formula not applicable.
    """
    from scipy.stats import norm
    from multi_option.bs_closed_form import bs_price

    if s0 >= B:
        return 0.0  # Already knocked out

    # Parameters
    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    lambda_val = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)

    # Standard Black-Scholes call price
    vanilla_call = bs_price(s0, K, r, q, sigma, T, call=True)

    # Barrier adjustment terms
    x1 = np.log(s0 / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(s0 / B) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

    y1 = np.log(B ** 2 / (s0 * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(B / s0) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

    # Up-and-out call price components
    if K > B:
        # Strike above barrier - option worthless
        return 0.0

    # Calculate price using barrier formula
    term1 = s0 * np.exp(-q * T) * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
    term2 = s0 * np.exp(-q * T) * norm.cdf(x2) - K * np.exp(-r * T) * norm.cdf(x2 - sigma * np.sqrt(T))

    term3 = s0 * np.exp(-q * T) * (B / s0) ** (2 * (mu + 1)) * norm.cdf(y1)
    term4 = K * np.exp(-r * T) * (B / s0) ** (2 * mu) * norm.cdf(y1 - sigma * np.sqrt(T))

    term5 = s0 * np.exp(-q * T) * (B / s0) ** (2 * (mu + 1)) * norm.cdf(y2)
    term6 = K * np.exp(-r * T) * (B / s0) ** (2 * mu) * norm.cdf(y2 - sigma * np.sqrt(T))

    if K < B:
        price = term1 - term2 - term3 + term4 + term5 - term6
    else:
        price = 0.0

    # Ensure non-negative
    return max(0.0, float(price))


def compute_barrier_greeks_mc(
    cfg: EngineConfig,
    ds: float = 0.01,
    db: float = 0.01,
    dsigma: float = 0.001
) -> Dict[str, float]:
    """Compute barrier option Greeks using Monte Carlo.

    Args:
        cfg: Engine configuration.
        ds: Stock price bump for delta/gamma.
        db: Barrier bump for barrier delta.
        dsigma: Volatility bump for vega.

    Returns:
        Dictionary with Greek values.
    """
    if cfg.barrier is None:
        raise ValueError("Barrier level must be specified")

    # Base price
    base_result = price_barrierup_out_call_mc(
        cfg.s0, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    price_base = base_result.price

    # Delta: dV/dS
    up_result = price_barrierup_out_call_mc(
        cfg.s0 + ds, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    down_result = price_barrierup_out_call_mc(
        cfg.s0 - ds, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    delta = (up_result.price - down_result.price) / (2 * ds)

    # Gamma: d2V/dS2
    gamma = (up_result.price - 2 * price_base + down_result.price) / (ds ** 2)

    # Vega: dV/d_sigma
    vega_result = price_barrierup_out_call_mc(
        cfg.s0, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma + dsigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    vega = (vega_result.price - price_base) / dsigma / 100  # Per 1% vol move

    # Barrier Delta: dV/dB
    barrierup_result = price_barrierup_out_call_mc(
        cfg.s0, cfg.K, cfg.barrier + db, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed
    )
    barrier_delta = (barrierup_result.price - price_base) / db

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'vega': float(vega),
        'barrier_delta': float(barrier_delta),
        'knock_out_prob': float(base_result.meta.get('knock_out_prob', 0))
    }


def analyze_barrier_sensitivity(
    cfg: EngineConfig,
    barrier_range: tuple = (1.1, 1.5),
    n_points: int = 20
) -> Dict[str, np.ndarray]:
    """Analyze option price sensitivity to barrier level.

    Args:
        cfg: Engine configuration.
        barrier_range: Range of barrier multipliers (relative to spot).
        n_points: Number of barrier levels to test.

    Returns:
        Dictionary with barrier levels and corresponding prices.
    """
    barrier_multipliers = np.linspace(barrier_range[0], barrier_range[1], n_points)
    barrier_levels = cfg.s0 * barrier_multipliers

    prices = []
    knock_out_probs = []

    for barrier in barrier_levels:
        if barrier > cfg.K:  # Valid barrier for up-and-out call
            result = price_barrierup_out_call_mc(
                cfg.s0, cfg.K, barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.mc_steps // 10,  # Use fewer paths for speed
                cfg.mc_paths, cfg.seed
            )
            prices.append(result.price)
            knock_out_probs.append(result.meta.get('knock_out_prob', 0))
        else:
            prices.append(0.0)
            knock_out_probs.append(1.0)

    return {
        'barrier_levels': barrier_levels,
        'barrier_multipliers': barrier_multipliers,
        'prices': np.array(prices),
        'knock_out_probs': np.array(knock_out_probs)
    }