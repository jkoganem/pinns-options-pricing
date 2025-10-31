"""Convergence analysis for numerical methods.

This module implements convergence testing for various numerical
methods used in option pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from multi_option.datatypes import EngineConfig


def convergence_curve(
    vary: str,
    product: str,
    method: str,
    cfg: EngineConfig,
    ref_price: float,
    n_points: int = 10
) -> pd.DataFrame:
    """Generate convergence curve for a method.

    Args:
        vary: Parameter to vary ('grid', 'paths', 'steps', 'epochs').
        product: Option product type.
        method: Pricing method.
        cfg: Base configuration.
        ref_price: Reference price for error calculation.
        n_points: Number of points in convergence curve.

    Returns:
        DataFrame with columns ['x', 'abs_err', 'rel_err', 'price'].
    """
    if vary == 'grid':
        return _convergence_grid(product, method, cfg, ref_price, n_points)
    elif vary == 'paths':
        return _convergence_paths(product, method, cfg, ref_price, n_points)
    elif vary == 'steps':
        return _convergence_steps(product, method, cfg, ref_price, n_points)
    elif vary == 'epochs':
        return _convergence_epochs(product, method, cfg, ref_price, n_points)
    else:
        raise ValueError(f"Unknown vary parameter: {vary}")


def _convergence_grid(
    product: str,
    method: str,
    cfg: EngineConfig,
    ref_price: float,
    n_points: int
) -> pd.DataFrame:
    """Test convergence with grid refinement."""
    from multi_option.fd_pde.cn_solver import price_european_cn
    from multi_option.fd_pde.lcp_psor import price_american_put_at_spot

    if method != 'pde':
        raise ValueError("Grid convergence only applies to PDE methods")

    # Grid sizes to test
    ns_values = np.logspace(1.5, 2.5, n_points).astype(int)
    nt_values = ns_values  # Keep square grids

    results = []

    for ns, nt in zip(ns_values, nt_values):
        if product == 'european_call':
            price = price_european_cn(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.s_max, ns, nt, call=True
            )
        elif product == 'european_put':
            price = price_european_cn(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.s_max, ns, nt, call=False
            )
        elif product == 'american_put':
            price = price_american_put_at_spot(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.s_max, ns, nt
            )
        else:
            raise ValueError(f"Unknown product: {product}")

        abs_err = abs(price - ref_price)
        rel_err = abs_err / abs(ref_price) if ref_price != 0 else abs_err

        results.append({
            'x': ns,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'price': price,
            'ns': ns,
            'nt': nt
        })

    return pd.DataFrame(results)


def _convergence_paths(
    product: str,
    method: str,
    cfg: EngineConfig,
    ref_price: float,
    n_points: int
) -> pd.DataFrame:
    """Test convergence with Monte Carlo paths."""
    from multi_option.mc.pricing import (
        price_european_mc,
        price_barrierup_out_call_mc,
        price_asian_arith_call_mc
    )

    if method != 'mc':
        raise ValueError("Path convergence only applies to MC methods")

    # Number of paths to test
    paths_values = np.logspace(2, 5, n_points).astype(int)

    results = []

    for paths in paths_values:
        if product == 'european_call':
            result = price_european_mc(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.mc_steps, paths, cfg.seed, call=True
            )
            price = result.price
        elif product == 'european_put':
            result = price_european_mc(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.mc_steps, paths, cfg.seed, call=False
            )
            price = result.price
        elif product == 'barrierup_out_call':
            if cfg.barrier is None:
                cfg.barrier = cfg.s0 * 1.3
            result = price_barrierup_out_call_mc(
                cfg.s0, cfg.K, cfg.barrier, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.mc_steps, paths, cfg.seed
            )
            price = result.price
        elif product == 'asian_arith_call':
            result = price_asian_arith_call_mc(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                cfg.mc_steps, paths, cfg.seed
            )
            price = result.price
        else:
            raise ValueError(f"Unknown product: {product}")

        abs_err = abs(price - ref_price)
        rel_err = abs_err / abs(ref_price) if ref_price != 0 else abs_err

        results.append({
            'x': paths,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'price': price,
            'stderr': result.stderr if hasattr(result, 'stderr') else 0
        })

    return pd.DataFrame(results)


def _convergence_steps(
    product: str,
    method: str,
    cfg: EngineConfig,
    ref_price: float,
    n_points: int
) -> pd.DataFrame:
    """Test convergence with time steps."""
    from multi_option.mc.pricing import price_european_mc

    if method != 'mc':
        raise ValueError("Step convergence primarily applies to MC methods")

    # Number of steps to test
    steps_values = np.logspace(0.5, 2.5, n_points).astype(int)

    results = []

    for steps in steps_values:
        if product in ['european_call', 'european_put']:
            result = price_european_mc(
                cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
                steps, cfg.mc_paths, cfg.seed,
                call=(product == 'european_call')
            )
            price = result.price
        else:
            raise ValueError(f"Unknown product: {product}")

        abs_err = abs(price - ref_price)
        rel_err = abs_err / abs(ref_price) if ref_price != 0 else abs_err

        results.append({
            'x': steps,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'price': price
        })

    return pd.DataFrame(results)


def _convergence_epochs(
    product: str,
    method: str,
    cfg: EngineConfig,
    ref_price: float,
    n_points: int
) -> pd.DataFrame:
    """Test convergence with PINN training epochs."""
    if method != 'pinn':
        raise ValueError("Epoch convergence only applies to PINN methods")

    # Import here to avoid circular dependency
    from multi_option.pinns.train import train_pinn_with_validation

    # Epochs to test
    epochs_values = np.logspace(1, 3, n_points).astype(int)

    results = []
    is_call = product == 'european_call'

    for epochs in epochs_values:
        # Train PINN with validation
        history = train_pinn_with_validation(
            cfg.r, cfg.q, cfg.sigma, cfg.K, cfg.T, cfg.s_max,
            cfg.pinn_hidden, epochs, cfg.pinn_lr, cfg.seed,
            is_call, val_freq=max(1, epochs // 10)
        )

        # Get final price
        if len(history) > 0:
            final_error = history.iloc[-1]['mean_abs_error']
            price = ref_price  # Approximate from error
        else:
            final_error = float('inf')
            price = 0

        results.append({
            'x': epochs,
            'abs_err': final_error,
            'rel_err': final_error / abs(ref_price) if ref_price != 0 else final_error,
            'price': price
        })

    return pd.DataFrame(results)


def compute_convergence_rate(
    convergence_df: pd.DataFrame,
    x_col: str = 'x',
    err_col: str = 'abs_err'
) -> Dict[str, float]:
    """Compute convergence rate from convergence data.

    Fits log(error) = a + b*log(x) to estimate convergence rate.

    Args:
        convergence_df: DataFrame with convergence data.
        x_col: Column name for x values.
        err_col: Column name for error values.

    Returns:
        Dictionary with convergence rate statistics.
    """
    # Remove zero errors
    df = convergence_df[convergence_df[err_col] > 0].copy()

    if len(df) < 2:
        return {'rate': 0, 'r_squared': 0}

    # Log-log regression
    log_x = np.log(df[x_col].values)
    log_err = np.log(df[err_col].values)

    # Fit line
    coeffs = np.polyfit(log_x, log_err, 1)
    rate = -coeffs[0]  # Negative of slope

    # Compute R-squared
    fitted = np.polyval(coeffs, log_x)
    residuals = log_err - fitted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'rate': float(rate),
        'r_squared': float(r_squared),
        'intercept': float(coeffs[1])
    }


def richardson_extrapolation(
    values: List[float],
    h_values: List[float],
    order: int = 2
) -> float:
    """Perform Richardson extrapolation to improve accuracy.

    Args:
        values: Computed values at different resolutions.
        h_values: Step sizes (should decrease).
        order: Expected order of convergence.

    Returns:
        Extrapolated value.
    """
    if len(values) < 2:
        return values[0] if values else 0

    # Simple two-point extrapolation
    if len(values) == 2:
        h1, h2 = h_values[0], h_values[1]
        v1, v2 = values[0], values[1]
        ratio = (h1 / h2) ** order
        return (ratio * v2 - v1) / (ratio - 1)

    # Multi-point extrapolation (simplified)
    # Use last two points
    return richardson_extrapolation(values[-2:], h_values[-2:], order)


def optimal_grid_size(
    target_error: float,
    convergence_df: pd.DataFrame
) -> int:
    """Estimate optimal grid size for target error.

    Args:
        target_error: Target absolute error.
        convergence_df: DataFrame with convergence data.

    Returns:
        Estimated optimal grid size.
    """
    # Get convergence rate
    rate_info = compute_convergence_rate(convergence_df)
    rate = rate_info['rate']

    if rate <= 0:
        # Cannot estimate
        return convergence_df['x'].max()

    # Extrapolate
    log_target = np.log(target_error)
    log_x = (log_target - rate_info['intercept']) / (-rate)
    optimal_x = np.exp(log_x)

    return int(optimal_x)