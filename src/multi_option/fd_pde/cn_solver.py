"""Crank-Nicolson solver for European options.

This module implements the Crank-Nicolson finite-difference method
for solving the Black-Scholes PDE for European options.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from multi_option.fd_pde.grid import make_grids, interpolate_from_grid


def price_european_cn(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int,
    call: bool
) -> float:
    """Price European option using Crank-Nicolson method.

    The Crank-Nicolson method is a second-order accurate implicit-explicit
    finite-difference scheme that is unconditionally stable.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.
        call: True for call, False for put.

    Returns:
        Option price at (s0, T).
    """
    # Create grids
    s_grid, t_grid = make_grids(s_max, ns, T, nt)
    ds = s_grid[1] - s_grid[0]
    dt = t_grid[1] - t_grid[0]

    # Initialize solution matrix
    V = np.zeros((ns, nt))

    # Set terminal condition at t = T
    for i in range(ns):
        S = s_grid[i]
        if call:
            V[i, -1] = max(S - K, 0)
        else:
            V[i, -1] = max(K - S, 0)

    # Set boundary conditions
    if call:
        # For call: V(0,t) = 0, V(S_max,t) ~ S_max - K*exp(-r*(T-t))
        V[0, :] = 0
        for j in range(nt):
            tau = T - t_grid[j]  # Time to maturity
            V[-1, j] = s_max - K * np.exp(-r * tau)
    else:
        # For put: V(0,t) = K*exp(-r*(T-t)), V(S_max,t) = 0
        for j in range(nt):
            tau = T - t_grid[j]  # Time to maturity
            V[0, j] = K * np.exp(-r * tau)
        V[-1, :] = 0

    # Build tridiagonal matrices for Crank-Nicolson
    # We solve from t=T backwards to t=0
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    gamma = np.zeros(ns)

    for i in range(1, ns - 1):
        S = s_grid[i]
        alpha[i] = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
        beta[i] = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma[i] = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)

    # Create matrices
    # Matrix for implicit part (left-hand side)
    M1_diag = 1 - beta[1:-1]
    M1_lower = -alpha[2:-1]
    M1upper = -gamma[1:-2]

    M1 = diags([M1_lower, M1_diag, M1upper], [-1, 0, 1], shape=(ns-2, ns-2), format='csc')

    # Matrix for explicit part (right-hand side)
    M2_diag = 1 + beta[1:-1]
    M2_lower = alpha[2:-1]
    M2upper = gamma[1:-2]

    M2 = diags([M2_lower, M2_diag, M2upper], [-1, 0, 1], shape=(ns-2, ns-2), format='csc')

    # Time stepping - backward from T to 0
    for j in range(nt - 2, -1, -1):
        # Set up right-hand side
        rhs = M2 @ V[1:-1, j + 1]

        # Apply boundary conditions
        rhs[0] += alpha[1] * (V[0, j + 1] + V[0, j])
        rhs[-1] += gamma[-2] * (V[-1, j + 1] + V[-1, j])

        # Solve linear system
        V[1:-1, j] = spsolve(M1, rhs)

    # Interpolate to get price at s0
    price = interpolate_from_grid(s_grid, V[:, 0], s0)

    return float(price)


def cn_theta_scheme(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int,
    call: bool,
    theta: float = 0.5
) -> float:
    """General theta-scheme for European options.

    Args:
        s0: Initial stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.
        call: True for call, False for put.
        theta: Theta parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit).

    Returns:
        Option price at (s0, T).
    """
    if abs(theta - 0.5) < 1e-10:
        # Use optimized Crank-Nicolson implementation
        return price_european_cn(s0, K, r, q, sigma, T, s_max, ns, nt, call)

    # General theta-scheme implementation
    s_grid, t_grid = make_grids(s_max, ns, T, nt)
    ds = s_grid[1] - s_grid[0]
    dt = t_grid[1] - t_grid[0]

    # Initialize solution matrix
    V = np.zeros((ns, nt))

    # Set terminal condition
    for i in range(ns):
        S = s_grid[i]
        if call:
            V[i, -1] = max(S - K, 0)
        else:
            V[i, -1] = max(K - S, 0)

    # Set boundary conditions
    if call:
        V[0, :] = 0
        for j in range(nt):
            tau = T - t_grid[j]
            V[-1, j] = s_max - K * np.exp(-r * tau)
    else:
        for j in range(nt):
            tau = T - t_grid[j]
            V[0, j] = K * np.exp(-r * tau)
        V[-1, :] = 0

    # Build coefficient matrices
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    gamma = np.zeros(ns)

    for i in range(1, ns - 1):
        S = s_grid[i]
        alpha[i] = 0.5 * dt * (sigma**2 * i**2 / ds**2 - (r - q) * i / ds)
        beta[i] = -dt * (sigma**2 * i**2 / ds**2 + r)
        gamma[i] = 0.5 * dt * (sigma**2 * i**2 / ds**2 + (r - q) * i / ds)

    # Time stepping
    for j in range(nt - 2, -1, -1):
        # Solve using theta-scheme
        V[1:-1, j] = _solve_theta_step(
            V[:, j + 1], alpha, beta, gamma, theta, V[0, j], V[-1, j]
        )

    # Interpolate to get price at s0
    price = interpolate_from_grid(s_grid, V[:, 0], s0)
    return float(price)


def _solve_theta_step(
    V_next: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    theta: float,
    bc_lower: float,
    bcupper: float
) -> np.ndarray:
    """Solve one time step of the theta-scheme.

    Args:
        V_next: Solution at next time step.
        alpha: Lower diagonal coefficients.
        beta: Main diagonal coefficients.
        gamma: Upper diagonal coefficients.
        theta: Theta parameter.
        bc_lower: Lower boundary condition.
        bcupper: Upper boundary condition.

    Returns:
        Solution at current time step (interior points).
    """
    n = len(V_next) - 2  # Number of interior points

    # Build matrices
    # Left-hand side (implicit part)
    M1_diag = 1 - theta * beta[1:-1]
    M1_lower = -theta * alpha[2:-1]
    M1upper = -theta * gamma[1:-2]
    M1 = diags([M1_lower, M1_diag, M1upper], [-1, 0, 1], shape=(n, n))

    # Right-hand side (explicit part)
    M2_diag = 1 + (1 - theta) * beta[1:-1]
    M2_lower = (1 - theta) * alpha[2:-1]
    M2upper = (1 - theta) * gamma[1:-2]
    M2 = diags([M2_lower, M2_diag, M2upper], [-1, 0, 1], shape=(n, n))

    # Set up right-hand side
    rhs = M2 @ V_next[1:-1]

    # Apply boundary conditions
    rhs[0] += theta * alpha[1] * bc_lower + (1 - theta) * alpha[1] * V_next[0]
    rhs[-1] += theta * gamma[-2] * bcupper + (1 - theta) * gamma[-2] * V_next[-1]

    # Solve linear system
    return spsolve(M1, rhs)