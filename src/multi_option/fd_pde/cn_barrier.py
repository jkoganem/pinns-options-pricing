"""Crank-Nicolson solver for barrier options.

This module implements the Crank-Nicolson finite-difference method
for solving the Black-Scholes PDE for barrier options (up-and-out, down-and-out).
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from multi_option.fd_pde.grid import make_grids, interpolate_from_grid


def price_barrierup_out_call_cn(
    s0: float,
    K: float,
    B: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    ns: int = 401,
    nt: int = 400
) -> float:
    """Price up-and-out barrier call using Crank-Nicolson method.

    An up-and-out call is knocked out if the stock price reaches or exceeds
    the barrier B at any time before expiration.

    Args:
        s0: Initial stock price.
        K: Strike price.
        B: Barrier level (knock-out if S >= B).
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        ns: Number of spatial grid points.
        nt: Number of time grid points.

    Returns:
        Option price at (s0, T).

    Raises:
        ValueError: If barrier conditions are invalid.
    """
    # Validation
    if B <= s0:
        return 0.0  # Already knocked out
    if B <= K:
        return 0.0  # Barrier below strike - worthless for call

    # Set s_max to slightly above barrier to capture the boundary
    s_max = B * 1.5

    # Create grids - ensure barrier falls on a grid point
    s_grid = np.linspace(0, s_max, ns)

    # Find the barrier index (closest grid point to B)
    barrier_idx = np.argmin(np.abs(s_grid - B))

    # Adjust the barrier to fall exactly on a grid point
    B_grid = s_grid[barrier_idx]

    t_grid = np.linspace(0, T, nt)
    dt = t_grid[1] - t_grid[0]

    # Initialize solution matrix
    V = np.zeros((ns, nt))

    # Set terminal condition at t = T (only up to barrier)
    for i in range(barrier_idx + 1):  # Only up to barrier
        S = s_grid[i]
        V[i, -1] = max(S - K, 0)

    # Barrier condition: V = 0 at and above barrier for all times
    V[barrier_idx:, :] = 0.0

    # Lower boundary condition: V(0,t) = 0 for call
    V[0, :] = 0

    # Build tridiagonal matrices for Crank-Nicolson
    # We solve from t=T backwards to t=0
    ds = s_grid[1] - s_grid[0]

    # Coefficients for interior points
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    gamma = np.zeros(ns)

    for i in range(1, barrier_idx):  # Only interior points below barrier
        S = s_grid[i]
        alpha[i] = 0.25 * dt * (sigma**2 * S**2 / ds**2 - (r - q) * S / ds)
        beta[i] = -0.5 * dt * (sigma**2 * S**2 / ds**2 + r)
        gamma[i] = 0.25 * dt * (sigma**2 * S**2 / ds**2 + (r - q) * S / ds)

    # Number of interior points (excluding boundaries and barrier)
    n_interior = barrier_idx - 1

    if n_interior < 2:
        raise ValueError(f"Barrier too close to lower boundary. Need more grid points or adjust barrier.")

    # Create matrices for interior points only
    # Matrix for implicit part (left-hand side)
    M1_diag = 1 - beta[1:barrier_idx]
    M1_lower = -alpha[2:barrier_idx]
    M1upper = -gamma[1:barrier_idx-1]

    M1 = diags([M1_lower, M1_diag, M1upper], [-1, 0, 1],
               shape=(n_interior, n_interior), format='csc')

    # Matrix for explicit part (right-hand side)
    M2_diag = 1 + beta[1:barrier_idx]
    M2_lower = alpha[2:barrier_idx]
    M2upper = gamma[1:barrier_idx-1]

    M2 = diags([M2_lower, M2_diag, M2upper], [-1, 0, 1],
               shape=(n_interior, n_interior), format='csc')

    # Time stepping - backward from T to 0
    for j in range(nt - 2, -1, -1):
        # Set up right-hand side
        rhs = M2 @ V[1:barrier_idx, j + 1]

        # Apply lower boundary condition (S=0): V(0,t) = 0
        rhs[0] += alpha[1] * (V[0, j + 1] + V[0, j])

        # Apply upper boundary condition (at barrier): V(B,t) = 0
        rhs[-1] += gamma[barrier_idx - 1] * (V[barrier_idx, j + 1] + V[barrier_idx, j])

        # Solve linear system for interior points
        V[1:barrier_idx, j] = spsolve(M1, rhs)

        # Enforce barrier: V = 0 at and above barrier
        V[barrier_idx:, j] = 0.0

    # Interpolate to get price at s0
    if s0 >= B:
        return 0.0  # Already knocked out

    price = interpolate_from_grid(s_grid, V[:, 0], s0)

    return max(0.0, float(price))


def price_barrier_down_out_put_cn(
    s0: float,
    K: float,
    B: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    ns: int = 401,
    nt: int = 400
) -> float:
    """Price down-and-out barrier put using Crank-Nicolson method.

    A down-and-out put is knocked out if the stock price reaches or falls below
    the barrier B at any time before expiration.

    Args:
        s0: Initial stock price.
        K: Strike price.
        B: Barrier level (knock-out if S <= B).
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        ns: Number of spatial grid points.
        nt: Number of time grid points.

    Returns:
        Option price at (s0, T).

    Raises:
        ValueError: If barrier conditions are invalid.
    """
    # Validation
    if B >= s0:
        return 0.0  # Already knocked out
    if B >= K:
        return 0.0  # Barrier above strike - worthless for put

    # Set s_max appropriately
    s_max = max(K * 2, s0 * 2)

    # Create grids - ensure barrier falls on a grid point
    s_grid = np.linspace(0, s_max, ns)

    # Find the barrier index (closest grid point to B)
    barrier_idx = np.argmin(np.abs(s_grid - B))

    # Adjust the barrier to fall exactly on a grid point
    B_grid = s_grid[barrier_idx]

    t_grid = np.linspace(0, T, nt)
    dt = t_grid[1] - t_grid[0]

    # Initialize solution matrix
    V = np.zeros((ns, nt))

    # Set terminal condition at t = T (only above barrier)
    for i in range(barrier_idx, ns):  # Only above barrier
        S = s_grid[i]
        V[i, -1] = max(K - S, 0)

    # Barrier condition: V = 0 at and below barrier for all times
    V[:barrier_idx + 1, :] = 0.0

    # Upper boundary condition: V(S_max,t) = 0 for put
    V[-1, :] = 0

    # Build tridiagonal matrices for Crank-Nicolson
    ds = s_grid[1] - s_grid[0]

    # Coefficients
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    gamma = np.zeros(ns)

    for i in range(barrier_idx + 1, ns - 1):  # Only interior points above barrier
        S = s_grid[i]
        alpha[i] = 0.25 * dt * (sigma**2 * S**2 / ds**2 - (r - q) * S / ds)
        beta[i] = -0.5 * dt * (sigma**2 * S**2 / ds**2 + r)
        gamma[i] = 0.25 * dt * (sigma**2 * S**2 / ds**2 + (r - q) * S / ds)

    # Number of interior points (between barrier and upper boundary)
    n_interior = ns - barrier_idx - 2

    if n_interior < 2:
        raise ValueError(f"Barrier too close to upper boundary. Need more grid points or adjust barrier.")

    # Create matrices for interior points only
    idx_start = barrier_idx + 1
    idx_end = ns - 1

    M1_diag = 1 - beta[idx_start:idx_end]
    M1_lower = -alpha[idx_start + 1:idx_end]
    M1upper = -gamma[idx_start:idx_end - 1]

    M1 = diags([M1_lower, M1_diag, M1upper], [-1, 0, 1],
               shape=(n_interior, n_interior), format='csc')

    M2_diag = 1 + beta[idx_start:idx_end]
    M2_lower = alpha[idx_start + 1:idx_end]
    M2upper = gamma[idx_start:idx_end - 1]

    M2 = diags([M2_lower, M2_diag, M2upper], [-1, 0, 1],
               shape=(n_interior, n_interior), format='csc')

    # Time stepping - backward from T to 0
    for j in range(nt - 2, -1, -1):
        # Set up right-hand side
        rhs = M2 @ V[idx_start:idx_end, j + 1]

        # Apply lower boundary condition (at barrier): V(B,t) = 0
        rhs[0] += alpha[idx_start] * (V[barrier_idx, j + 1] + V[barrier_idx, j])

        # Apply upper boundary condition (S=S_max): V(S_max,t) = 0
        rhs[-1] += gamma[idx_end - 1] * (V[-1, j + 1] + V[-1, j])

        # Solve linear system
        V[idx_start:idx_end, j] = spsolve(M1, rhs)

        # Enforce barrier: V = 0 at and below barrier
        V[:barrier_idx + 1, j] = 0.0

    # Interpolate to get price at s0
    if s0 <= B:
        return 0.0  # Already knocked out

    price = interpolate_from_grid(s_grid, V[:, 0], s0)

    return max(0.0, float(price))
