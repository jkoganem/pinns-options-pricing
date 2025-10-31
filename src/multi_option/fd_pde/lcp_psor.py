"""Projected SOR solver for American options.

This module implements the Projected Successive Over-Relaxation (PSOR)
method for solving the linear complementarity problem (LCP) that arises
in American option pricing.
"""

import numpy as np
from typing import Tuple
from multi_option.fd_pde.grid import make_grids, interpolate_from_grid


def price_american_put_cn_psor(
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int,
    omega: float = 1.2,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """Price American put using Crank-Nicolson with PSOR.

    The American put pricing problem leads to a linear complementarity
    problem (LCP) which we solve using Projected SOR iteration.

    Args:
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.
        omega: SOR relaxation parameter (1 < omega < 2).
        tol: Convergence tolerance.
        max_iter: Maximum number of PSOR iterations.

    Returns:
        Array of option values at t=0 for all stock prices.
    """
    # Create grids
    s_grid, t_grid = make_grids(s_max, ns, T, nt)
    ds = s_grid[1] - s_grid[0]
    dt = t_grid[1] - t_grid[0]

    # Initialize solution matrix
    V = np.zeros((ns, nt))

    # Set terminal condition (payoff at maturity)
    for i in range(ns):
        S = s_grid[i]
        V[i, -1] = max(K - S, 0)

    # Precompute coefficients for the finite difference scheme
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    gamma = np.zeros(ns)

    for i in range(1, ns - 1):
        S = s_grid[i]
        sig2 = sigma ** 2
        alpha[i] = 0.25 * dt * (sig2 * (S/ds)**2 - (r - q) * S/ds)
        beta[i] = -0.5 * dt * (sig2 * (S/ds)**2 + r)
        gamma[i] = 0.25 * dt * (sig2 * (S/ds)**2 + (r - q) * S/ds)

    # Time stepping - backward from T to 0
    for j in range(nt - 2, -1, -1):
        # Set boundary conditions
        V[0, j] = K * np.exp(-r * (T - t_grid[j]))  # S=0 boundary
        V[-1, j] = 0  # S=s_max boundary (far OTM)

        # Initial guess for PSOR
        V_old = V[:, j + 1].copy()

        # PSOR iteration to solve the LCP
        for iteration in range(max_iter):
            V_new = np.zeros(ns)
            V_new[0] = V[0, j]
            V_new[-1] = V[-1, j]

            # Update interior points
            for i in range(1, ns - 1):
                # Crank-Nicolson discretization
                rhs = (1 + beta[i]) * V[i, j + 1]
                if i > 1:
                    rhs += alpha[i] * V[i - 1, j + 1]
                if i < ns - 2:
                    rhs += gamma[i] * V[i + 1, j + 1]

                # Add contributions from already updated values
                if i > 1:
                    rhs -= alpha[i] * V_new[i - 1]
                if i < ns - 2:
                    rhs -= gamma[i] * V_old[i + 1]

                # Solve for V_new[i]
                denominator = 1 - beta[i]
                if abs(denominator) < 1e-10:
                    denominator = 1e-10 * np.sign(denominator) if denominator != 0 else 1e-10

                v_implicit = rhs / denominator

                # Apply SOR update
                v_sor = (1 - omega) * V_old[i] + omega * v_implicit

                # Apply early exercise constraint (projection step)
                payoff = max(K - s_grid[i], 0)
                V_new[i] = max(v_sor, payoff)

            # Check convergence
            error = np.max(np.abs(V_new - V_old))
            if error < tol:
                break

            V_old = V_new.copy()

        V[:, j] = V_new

    return V[:, 0]


def price_american_put_at_spot(
    s0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int,
    omega: float = 1.2,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> float:
    """Price American put at a specific spot price.

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
        omega: SOR relaxation parameter.
        tol: Convergence tolerance.
        max_iter: Maximum number of PSOR iterations.

    Returns:
        American put option price at s0.
    """
    # Get full solution grid
    V_grid = price_american_put_cn_psor(K, r, q, sigma, T, s_max, ns, nt, omega, tol, max_iter)

    # Create spatial grid
    s_grid, _ = make_grids(s_max, ns, T, nt)

    # Interpolate to get price at s0
    price = interpolate_from_grid(s_grid, V_grid, s0)

    return float(price)


def compute_early_exercise_boundary(
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    s_max: float,
    ns: int,
    nt: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the early exercise boundary for an American put.

    Args:
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.

    Returns:
        Tuple of (times, boundary_prices).
    """
    # Create grids
    s_grid, t_grid = make_grids(s_max, ns, T, nt)

    # Get full solution
    V_grid = price_american_put_cn_psor(K, r, q, sigma, T, s_max, ns, nt)

    # Find exercise boundary at each time
    times = []
    boundary = []

    for j in range(nt):
        # Find the highest stock price where early exercise is optimal
        for i in range(ns - 1, -1, -1):
            S = s_grid[i]
            payoff = max(K - S, 0)

            # Check if we should exercise
            if abs(V_grid[i] - payoff) < 1e-6 and payoff > 0:
                times.append(t_grid[j])
                boundary.append(S)
                break

    return np.array(times), np.array(boundary)