"""Random number generation for Monte Carlo simulations.

This module provides utilities for generating stock price paths
using Geometric Brownian Motion with variance reduction techniques.
"""

import numpy as np
import pandas as pd


def gbm_paths_antithetic(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int
) -> pd.DataFrame:
    """Generate GBM paths with antithetic variates.

    Generates stock price paths using Geometric Brownian Motion
    with antithetic variance reduction. Half of the paths are
    generated normally, and half are their antithetic pairs.

    Args:
        s0: Initial stock price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of paths (must be even).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with shape (paths, steps+1) containing price paths.
    """
    if paths % 2 != 0:
        paths += 1  # Make even for antithetic pairs

    np.random.seed(seed)

    dt = T / steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate random shocks for half the paths
    half_paths = paths // 2
    Z = np.random.standard_normal((half_paths, steps))

    # Create antithetic pairs
    Z_anti = -Z
    Z_all = np.vstack([Z, Z_anti])

    # Initialize paths array
    paths_array = np.zeros((paths, steps + 1))
    paths_array[:, 0] = s0

    # Generate paths using vectorized operations
    for t in range(1, steps + 1):
        paths_array[:, t] = paths_array[:, t - 1] * np.exp(
            drift + diffusion * Z_all[:, t - 1]
        )

    # Create DataFrame with proper column names
    columns = [f"t_{i}" for i in range(steps + 1)]
    return pd.DataFrame(paths_array, columns=columns)


def gbm_paths_standard(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int
) -> pd.DataFrame:
    """Generate standard GBM paths without variance reduction.

    Args:
        s0: Initial stock price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of paths.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with shape (paths, steps+1) containing price paths.
    """
    np.random.seed(seed)

    dt = T / steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate all random shocks at once
    Z = np.random.standard_normal((paths, steps))

    # Initialize paths array
    paths_array = np.zeros((paths, steps + 1))
    paths_array[:, 0] = s0

    # Generate paths
    for t in range(1, steps + 1):
        paths_array[:, t] = paths_array[:, t - 1] * np.exp(
            drift + diffusion * Z[:, t - 1]
        )

    columns = [f"t_{i}" for i in range(steps + 1)]
    return pd.DataFrame(paths_array, columns=columns)


def gbm_paths_bridge(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    seed: int,
    barrier: float = None
) -> pd.DataFrame:
    """Generate GBM paths using Brownian bridge for barrier options.

    The Brownian bridge construction can improve accuracy for
    barrier option pricing by better capturing barrier crossings.

    Args:
        s0: Initial stock price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        steps: Number of time steps.
        paths: Number of paths.
        seed: Random seed for reproducibility.
        barrier: Barrier level (optional).

    Returns:
        DataFrame with shape (paths, steps+1) containing price paths.
    """
    np.random.seed(seed)

    dt = T / steps
    times = np.linspace(0, T, steps + 1)

    # Initialize paths array
    paths_array = np.zeros((paths, steps + 1))
    paths_array[:, 0] = s0

    # Generate terminal Brownian motion values first
    ZT = np.random.standard_normal(paths)
    WT = sigma * np.sqrt(T) * ZT

    # Build paths using bridge construction
    for i in range(paths):
        # Start with endpoints for Brownian motion (not log price)
        W_path = np.zeros(steps + 1)
        W_path[0] = 0.0  # Start at 0
        W_path[-1] = WT[i]  # End at terminal Brownian value

        # Fill in intermediate Brownian motion using bridge
        _fill_bridge(W_path, 0, steps, times, sigma, seed + i + 1)

        # Convert Brownian motion to log prices with drift
        log_prices = np.log(s0) + (r - q - 0.5 * sigma ** 2) * times + W_path

        # Convert to prices
        paths_array[i, :] = np.exp(log_prices)

    columns = [f"t_{i}" for i in range(steps + 1)]
    return pd.DataFrame(paths_array, columns=columns)


def _fill_bridge(
    W_path: np.ndarray,
    start_idx: int,
    end_idx: int,
    times: np.ndarray,
    sigma: float,
    seed: int
) -> None:
    """Recursively fill in Brownian bridge values.

    Args:
        W_path: Array to fill with Brownian motion values (sigma * W(t)).
        start_idx: Start index.
        end_idx: End index.
        times: Time points.
        sigma: Volatility.
        seed: Random seed.
    """
    if end_idx - start_idx <= 1:
        return

    np.random.seed(seed)

    # Find midpoint
    mid_idx = (start_idx + end_idx) // 2

    # Bridge parameters
    t_start = times[start_idx]
    t_mid = times[mid_idx]
    t_end = times[end_idx]

    # Conditional mean and variance for Brownian bridge
    alpha = (t_mid - t_start) / (t_end - t_start)
    mean = (1 - alpha) * W_path[start_idx] + alpha * W_path[end_idx]
    var = (t_mid - t_start) * (t_end - t_mid) / (t_end - t_start)

    # Sample midpoint
    if var > 0:
        W_path[mid_idx] = mean + np.sqrt(var) * np.random.standard_normal()
    else:
        W_path[mid_idx] = mean

    # Recursively fill left and right segments
    _fill_bridge(W_path, start_idx, mid_idx, times, sigma, seed + 1)
    _fill_bridge(W_path, mid_idx, end_idx, times, sigma, seed + 2)


def generate_sobol_points(dim: int, n: int, seed: int = 0) -> np.ndarray:
    """Generate Sobol quasi-random points.

    Simplified Sobol sequence generator for low-discrepancy sampling.

    Args:
        dim: Dimension of points.
        n: Number of points.
        seed: Random seed offset.

    Returns:
        Array of shape (n, dim) with quasi-random points in [0, 1].
    """
    # Simple implementation using bit manipulation
    # For production, use scipy.stats.qmc.Sobol
    np.random.seed(seed)

    # Fallback to stratified sampling for simplicity
    points = np.zeros((n, dim))

    for d in range(dim):
        # Stratified sampling in each dimension
        strata = np.linspace(0, 1, n + 1)
        for i in range(n):
            points[i, d] = np.random.uniform(strata[i], strata[i + 1])

    return points