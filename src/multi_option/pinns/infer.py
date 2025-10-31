"""Inference utilities for trained PINN models.

This module provides functions for loading and using trained PINN models
for option pricing.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from multi_option.pinns.model import make_pinn


def price_european_pinn(
    s0: float,
    T: float,
    weights_path: str,
    hidden: int = 64
) -> float:
    """Price European option using trained PINN.

    Args:
        s0: Initial stock price.
        T: Time to maturity.
        weights_path: Path to saved model weights.
        hidden: Number of hidden units (must match trained model).

    Returns:
        Option price at (s0, T).
    """
    # Load model
    model = load_pinn_model(weights_path, hidden)
    model.eval()

    # Device
    device = next(model.parameters()).device

    # Convert inputs to tensors
    s_tensor = torch.tensor([s0], dtype=torch.float32, device=device)
    t_tensor = torch.tensor([T], dtype=torch.float32, device=device)

    # Inference
    with torch.no_grad():
        price = model(s_tensor, t_tensor).squeeze().item()

    return float(price)


def load_pinn_model(
    weights_path: str,
    hidden: int = 64
) -> torch.nn.Module:
    """Load a trained PINN model.

    Args:
        weights_path: Path to saved model weights.
        hidden: Number of hidden units.

    Returns:
        Loaded model.
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = make_pinn(hidden)

    # Load weights
    if Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model.to(device)
    return model


def batch_price_pinn(
    stock_prices: Union[List[float], np.ndarray],
    times: Union[List[float], np.ndarray],
    weights_path: str,
    hidden: int = 64
) -> np.ndarray:
    """Price options for multiple stock prices and times using PINN.

    Args:
        stock_prices: Array of stock prices.
        times: Array of times to maturity.
        weights_path: Path to saved model weights.
        hidden: Number of hidden units.

    Returns:
        Array of option prices.
    """
    # Load model
    model = load_pinn_model(weights_path, hidden)
    model.eval()

    device = next(model.parameters()).device

    # Convert to tensors
    s_tensor = torch.tensor(stock_prices, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(times, dtype=torch.float32, device=device)

    # Ensure same length
    if len(s_tensor) != len(t_tensor):
        raise ValueError("Stock prices and times must have same length")

    # Batch inference
    with torch.no_grad():
        prices = model(s_tensor, t_tensor).squeeze().cpu().numpy()

    return prices


def compute_greeks_pinn(
    s0: float,
    T: float,
    weights_path: str,
    hidden: int = 64,
    ds: float = 0.01,
    dt: float = 0.001
) -> dict:
    """Compute Greeks using finite differences on PINN.

    Args:
        s0: Stock price.
        T: Time to maturity.
        weights_path: Path to saved model weights.
        hidden: Number of hidden units.
        ds: Stock price bump for finite difference.
        dt: Time bump for finite difference.

    Returns:
        Dictionary with Greeks.
    """
    # Load model
    model = load_pinn_model(weights_path, hidden)
    model.eval()

    device = next(model.parameters()).device

    # Price at current point
    v_0 = price_european_pinn(s0, T, weights_path, hidden)

    # Delta: dV/dS
    vup = price_european_pinn(s0 + ds, T, weights_path, hidden)
    v_down = price_european_pinn(s0 - ds, T, weights_path, hidden)
    delta = (vup - v_down) / (2 * ds)

    # Gamma: d2V/dS2
    gamma = (vup - 2 * v_0 + v_down) / (ds ** 2)

    # Theta: -dV/dT (note negative sign)
    if T > dt:
        v_t_minus = price_european_pinn(s0, T - dt, weights_path, hidden)
        theta = -(v_0 - v_t_minus) / dt / 365  # Per day
    else:
        theta = 0.0

    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'price': float(v_0)
    }


def create_price_surface_pinn(
    weights_path: str,
    s_range: tuple = (50, 150),
    t_range: tuple = (0.01, 1.0),
    n_points: int = 50,
    hidden: int = 64
) -> dict:
    """Create option price surface using PINN.

    Args:
        weights_path: Path to saved model weights.
        s_range: Range of stock prices (min, max).
        t_range: Range of times (min, max).
        n_points: Number of points in each dimension.
        hidden: Number of hidden units.

    Returns:
        Dictionary with mesh grids and prices.
    """
    # Load model
    model = load_pinn_model(weights_path, hidden)
    model.eval()

    # Create grid
    s_vals = np.linspace(s_range[0], s_range[1], n_points)
    t_vals = np.linspace(t_range[0], t_range[1], n_points)
    S_mesh, T_mesh = np.meshgrid(s_vals, t_vals)

    # Flatten for batch processing
    s_flat = S_mesh.flatten()
    t_flat = T_mesh.flatten()

    # Get prices
    prices_flat = batch_price_pinn(s_flat, t_flat, weights_path, hidden)

    # Reshape to grid
    prices_grid = prices_flat.reshape(S_mesh.shape)

    return {
        'S': S_mesh,
        'T': T_mesh,
        'V': prices_grid,
        's_vals': s_vals,
        't_vals': t_vals
    }


def validate_pinn_against_bs(
    weights_path: str,
    r: float,
    q: float,
    sigma: float,
    K: float,
    T: float,
    call: bool,
    hidden: int = 64,
    n_test: int = 100
) -> dict:
    """Validate PINN predictions against Black-Scholes.

    Args:
        weights_path: Path to saved model weights.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        K: Strike price.
        T: Time to maturity.
        call: True for call, False for put.
        hidden: Number of hidden units.
        n_test: Number of test points.

    Returns:
        Dictionary with validation metrics.
    """
    from multi_option.bs_closed_form import bs_price

    # Test points around the strike
    s_test = np.linspace(0.5 * K, 1.5 * K, n_test)
    t_test = np.ones(n_test) * T

    # PINN predictions
    pinn_prices = batch_price_pinn(s_test, t_test, weights_path, hidden)

    # Black-Scholes prices
    bs_prices = np.array([
        bs_price(s, K, r, q, sigma, T, call) for s in s_test
    ])

    # Compute errors
    abs_errors = np.abs(pinn_prices - bs_prices)
    rel_errors = np.abs((pinn_prices - bs_prices) / (bs_prices + 1e-8))

    return {
        'mean_abs_error': float(np.mean(abs_errors)),
        'max_abs_error': float(np.max(abs_errors)),
        'mean_rel_error': float(np.mean(rel_errors)),
        'max_rel_error': float(np.max(rel_errors)),
        'rmse': float(np.sqrt(np.mean((pinn_prices - bs_prices) ** 2)))
    }


class PINNPricer:
    """Wrapper class for PINN-based option pricing."""

    def __init__(self, weights_path: str, hidden: int = 64):
        """Initialize PINN pricer.

        Args:
            weights_path: Path to saved model weights.
            hidden: Number of hidden units.
        """
        self.model = load_pinn_model(weights_path, hidden)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def price(self, s0: float, T: float) -> float:
        """Price option at given point.

        Args:
            s0: Stock price.
            T: Time to maturity.

        Returns:
            Option price.
        """
        s_tensor = torch.tensor([s0], dtype=torch.float32, device=self.device)
        t_tensor = torch.tensor([T], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            price = self.model(s_tensor, t_tensor).squeeze().item()

        return float(price)

    def price_batch(
        self,
        stock_prices: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """Price options in batch.

        Args:
            stock_prices: Array of stock prices.
            times: Array of times.

        Returns:
            Array of option prices.
        """
        s_tensor = torch.tensor(stock_prices, dtype=torch.float32, device=self.device)
        t_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prices = self.model(s_tensor, t_tensor).squeeze().cpu().numpy()

        return prices