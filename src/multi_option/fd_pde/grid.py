"""Grid generation for finite-difference PDE methods.

This module provides utilities for creating spatial and temporal grids
for finite-difference option pricing.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def make_grids(
    s_max: float,
    ns: int,
    T: float,
    nt: int
) -> Tuple[pd.Index, pd.Index]:
    """Create spatial and temporal grids for finite-difference methods.

    Args:
        s_max: Maximum stock price.
        ns: Number of spatial grid points.
        T: Time to maturity.
        nt: Number of time grid points.

    Returns:
        Tuple of (spatial_grid, time_grid) as pandas Index objects.
    """
    # Create spatial grid from 0 to s_max
    s_grid = pd.Index(np.linspace(0, s_max, ns), name="S")

    # Create time grid from 0 to T
    t_grid = pd.Index(np.linspace(0, T, nt), name="t")

    return s_grid, t_grid


def get_grid_spacing(
    s_max: float,
    ns: int,
    T: float,
    nt: int
) -> Tuple[float, float]:
    """Calculate grid spacing for finite-difference methods.

    Args:
        s_max: Maximum stock price.
        ns: Number of spatial grid points.
        T: Time to maturity.
        nt: Number of time grid points.

    Returns:
        Tuple of (ds, dt) grid spacings.
    """
    ds = s_max / (ns - 1)
    dt = T / (nt - 1)
    return ds, dt


def find_nearest_grid_index(grid: pd.Index, value: float) -> int:
    """Find the nearest grid index for a given value.

    Args:
        grid: The grid (pandas Index).
        value: The value to find.

    Returns:
        Index of the nearest grid point.
    """
    differences = np.abs(grid - value)
    return int(differences.argmin())


def interpolate_from_grid(
    grid: pd.Index,
    values: np.ndarray,
    target: float
) -> float:
    """Linear interpolation from grid values.

    Args:
        grid: The grid points.
        values: Values at grid points.
        target: Target point for interpolation.

    Returns:
        Interpolated value.
    """
    if target <= grid[0]:
        return float(values[0])
    if target >= grid[-1]:
        return float(values[-1])

    # Find surrounding grid points
    idx = np.searchsorted(grid, target)
    if idx == 0:
        return float(values[0])
    if idx >= len(grid):
        return float(values[-1])

    # Linear interpolation
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = values[idx - 1], values[idx]

    # Handle exact match
    if x1 == x0:
        return float(y0)

    alpha = (target - x0) / (x1 - x0)
    return float(y0 + alpha * (y1 - y0))