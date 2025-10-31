"""Monte Carlo option pricing methods.

This module implements Monte Carlo simulation methods for pricing
various option types with variance reduction techniques.
"""

import numpy as np
from multi_option.datatypes import PriceResult
from multi_option.mc.rng import gbm_paths_antithetic, gbm_paths_bridge
from multi_option.bs_closed_form import bs_price


def price_european_mc(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int,
    call: bool = True
) -> PriceResult:
    """Price European option using Monte Carlo simulation.

    Uses antithetic variates for variance reduction.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed for reproducibility.
        call: True for call, False for put.

    Returns:
        PriceResult with price and standard error.
    """
    # Generate paths with antithetic variates
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)

    # Get terminal prices
    ST = paths_df.iloc[:, -1].values

    # Calculate payoffs
    if call:
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # Calculate price and standard error
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs) / np.sqrt(paths)

    return PriceResult(
        method="mc",
        product="european_call" if call else "european_put",
        price=float(price),
        stderr=float(stderr),
        meta={
            "paths": float(paths),
            "steps": float(steps)
        }
    )


def price_barrierup_out_call_mc(
    s0: float,
    K: float,
    B: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int
) -> PriceResult:
    """Price up-and-out barrier call using Monte Carlo.

    The option knocks out (becomes worthless) if the stock price
    touches or exceeds the barrier B during the option's life.

    Args:
        s0: Initial stock price.
        K: Strike price.
        B: Barrier level (must be > s0 and > K for up-and-out).
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        PriceResult with price and standard error.
    """
    if B <= s0:
        raise ValueError("Barrier must be above initial stock price for up-and-out")
    if B <= K:
        raise ValueError("Barrier must be above strike for up-and-out call")

    # Generate paths with antithetic variates (bridge had bugs - reverted)
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)

    # Check for barrier breach
    max_prices = paths_df.max(axis=1).values
    not_knocked_out = max_prices < B

    # Get terminal prices for paths that didn't knock out
    ST = paths_df.iloc[:, -1].values

    # Calculate payoffs (zero if knocked out)
    payoffs = np.where(not_knocked_out, np.maximum(ST - K, 0), 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # Calculate price and standard error
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs) / np.sqrt(paths)

    # Calculate knock-out probability for diagnostics
    knock_out_prob = 1 - np.mean(not_knocked_out)

    return PriceResult(
        method="mc",
        product="barrierup_out_call",
        price=float(price),
        stderr=float(stderr),
        meta={
            "paths": float(paths),
            "steps": float(steps),
            "barrier": float(B),
            "knock_out_prob": float(knock_out_prob)
        }
    )


def price_asian_arith_call_mc(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int
) -> PriceResult:
    """Price arithmetic average Asian call using Monte Carlo.

    The payoff depends on the arithmetic average of the stock price
    over the option's life.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps (averaging points).
        paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        PriceResult with price and standard error.
    """
    # Generate paths with antithetic variates
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)

    # Calculate arithmetic average for each path
    # Include all points from t=0 to t=T
    avg_prices = paths_df.mean(axis=1).values

    # Calculate payoffs based on average
    payoffs = np.maximum(avg_prices - K, 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # Calculate price and standard error
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs) / np.sqrt(paths)

    # Calculate some statistics for diagnostics
    in_the_money_prob = np.mean(payoffs > 0)
    avg_stock_mean = np.mean(avg_prices)

    return PriceResult(
        method="mc",
        product="asian_arith_call",
        price=float(price),
        stderr=float(stderr),
        meta={
            "paths": float(paths),
            "steps": float(steps),
            "itm_prob": float(in_the_money_prob),
            "avg_stock_mean": float(avg_stock_mean)
        }
    )


def price_lookback_put_mc(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int
) -> PriceResult:
    """Price floating strike lookback put using Monte Carlo.

    The payoff is max(S_max - ST, 0) where S_max is the maximum
    price during the option's life.

    Args:
        s0: Initial stock price.
        K: Not used for floating strike (included for interface consistency).
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        PriceResult with price and standard error.
    """
    # Generate paths with antithetic variates
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)

    # Get maximum price for each path
    S_max = paths_df.max(axis=1).values

    # Get terminal price
    ST = paths_df.iloc[:, -1].values

    # Calculate payoffs
    payoffs = np.maximum(S_max - ST, 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # Calculate price and standard error
    price = np.mean(discounted_payoffs)
    stderr = np.std(discounted_payoffs) / np.sqrt(paths)

    return PriceResult(
        method="mc",
        product="lookback_put",
        price=float(price),
        stderr=float(stderr),
        meta={
            "paths": float(paths),
            "steps": float(steps),
            "avg_max_price": float(np.mean(S_max))
        }
    )