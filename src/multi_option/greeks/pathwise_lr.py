"""Pathwise and likelihood ratio methods for Greeks computation.

This module implements advanced Monte Carlo methods for computing Greeks
using pathwise differentiation and likelihood ratio methods.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from multi_option.datatypes import GreeksResult
from multi_option.mc.rng import gbm_paths_antithetic


def compute_delta_pathwise(
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
) -> Tuple[float, float]:
    """Compute delta using pathwise method.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed.
        call: True for call, False for put.

    Returns:
        Tuple of (delta, standard_error).
    """
    # Generate paths
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)
    ST = paths_df.iloc[:, -1].values

    # Pathwise derivative: dST/dS_0 = ST/S_0
    path_derivative = ST / s0

    # Payoff indicator (1 if ITM, 0 otherwise)
    if call:
        indicator = (ST > K).astype(float)
    else:
        indicator = (ST < K).astype(float)
        path_derivative = -path_derivative  # Put has negative delta

    # Delta estimate
    discount_factor = np.exp(-r * T)
    delta_samples = discount_factor * path_derivative * indicator
    delta = np.mean(delta_samples)
    stderr = np.std(delta_samples) / np.sqrt(paths)

    return float(delta), float(stderr)


def compute_vega_pathwise(
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
) -> Tuple[float, float]:
    """Compute vega using pathwise method.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed.
        call: True for call, False for put.

    Returns:
        Tuple of (vega, standard_error).
    """
    np.random.seed(seed)

    # Generate standard normal shocks
    Z = np.random.standard_normal((paths, steps))

    # Compute paths
    dt = T / steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_S = np.zeros((paths, steps + 1))
    log_S[:, 0] = np.log(s0)

    for t in range(1, steps + 1):
        log_S[:, t] = log_S[:, t-1] + drift + diffusion * Z[:, t-1]

    ST = np.exp(log_S[:, -1])

    # Pathwise derivative: dST/d_sigma
    vega_derivative = ST * np.sum(Z, axis=1) * np.sqrt(dt)

    # Payoff calculation
    if call:
        payoff = np.maximum(ST - K, 0)
        indicator = (ST > K).astype(float)
    else:
        payoff = np.maximum(K - ST, 0)
        indicator = (ST < K).astype(float)

    # Vega estimate
    discount_factor = np.exp(-r * T)
    vega_samples = discount_factor * vega_derivative * indicator
    vega = np.mean(vega_samples) / 100  # Per 1% vol move
    stderr = np.std(vega_samples) / np.sqrt(paths) / 100

    return float(vega), float(stderr)


def compute_greeks_likelihood_ratio(
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
) -> GreeksResult:
    """Compute Greeks using likelihood ratio method.

    This method is useful for discontinuous payoffs.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed.
        call: True for call, False for put.

    Returns:
        GreeksResult with computed Greeks.
    """
    np.random.seed(seed)

    # Generate paths
    dt = T / steps
    Z = np.random.standard_normal((paths, steps))

    # Simulate stock prices
    log_S = np.log(s0)
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(steps):
        log_S = log_S + drift + diffusion * Z[:, t]

    ST = np.exp(log_S)

    # Payoffs
    if call:
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # Likelihood ratio scores
    WT = np.sum(Z, axis=1) * np.sqrt(dt)  # Terminal Brownian motion

    # Delta using LR
    score_delta = WT / (s0 * sigma * np.sqrt(T))
    delta_samples = discounted_payoffs * score_delta
    delta = np.mean(delta_samples)

    # Vega using LR
    score_vega = (WT ** 2 - T) / sigma
    vega_samples = discounted_payoffs * score_vega
    vega = np.mean(vega_samples) / 100

    # Gamma using LR (second-order score)
    score_gamma = (WT ** 2 - T) / (s0 ** 2 * sigma ** 2 * T)
    gamma_samples = discounted_payoffs * score_gamma
    gamma = np.mean(gamma_samples)

    # Theta (using finite difference as LR is complex)
    theta = 0.0  # Simplified

    # Rho using LR
    score_rho = T
    rho_samples = discounted_payoffs * score_rho
    rho = np.mean(rho_samples) / 100

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
        rho=float(rho)
    )


def compute_delta_mixed(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int,
    call: bool = True,
    smooth_param: float = 0.01
) -> Tuple[float, float]:
    """Compute delta using mixed pathwise-LR method.

    Uses smoothing for better convergence near the strike.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed.
        call: True for call, False for put.
        smooth_param: Smoothing parameter for payoff.

    Returns:
        Tuple of (delta, standard_error).
    """
    # Generate paths
    paths_df = gbm_paths_antithetic(s0, r, q, sigma, T, steps, paths, seed)
    ST = paths_df.iloc[:, -1].values

    # Smooth approximation to indicator function
    from scipy.stats import norm
    if call:
        # Smooth call payoff derivative
        d = (ST - K) / (smooth_param * K)
        smooth_indicator = norm.cdf(d)
    else:
        # Smooth put payoff derivative
        d = (K - ST) / (smooth_param * K)
        smooth_indicator = norm.cdf(d)

    # Pathwise derivative
    path_derivative = ST / s0
    if not call:
        path_derivative = -path_derivative

    # Delta estimate
    discount_factor = np.exp(-r * T)
    delta_samples = discount_factor * path_derivative * smooth_indicator
    delta = np.mean(delta_samples)
    stderr = np.std(delta_samples) / np.sqrt(paths)

    return float(delta), float(stderr)


def malliavin_greeks(
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
) -> Dict[str, float]:
    """Compute Greeks using Malliavin calculus approach.

    Simplified implementation for demonstration.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of simulation paths.
        seed: Random seed.
        call: True for call, False for put.

    Returns:
        Dictionary with Greek values.
    """
    np.random.seed(seed)

    # Generate Brownian paths
    dt = T / steps
    dW = np.random.standard_normal((paths, steps)) * np.sqrt(dt)
    W = np.cumsum(dW, axis=1)

    # Stock price at maturity
    drift = (r - q - 0.5 * sigma ** 2) * T
    ST = s0 * np.exp(drift + sigma * W[:, -1])

    # Payoff
    if call:
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    discount_factor = np.exp(-r * T)

    # Malliavin weight for delta
    weight_delta = W[:, -1] / (sigma * T)
    delta = discount_factor * np.mean(payoff * weight_delta) / s0

    # Malliavin weight for gamma
    weight_gamma = ((W[:, -1] ** 2) / (sigma ** 2 * T) - 1) / (sigma * T)
    gamma = discount_factor * np.mean(payoff * weight_gamma) / (s0 ** 2)

    # Malliavin weight for vega
    weight_vega = (W[:, -1] ** 2 - T) / sigma
    vega = discount_factor * np.mean(payoff * weight_vega) / 100

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'vega': float(vega)
    }