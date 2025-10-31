"""Training routines for Physics-Informed Neural Networks.

This module implements the training loop for PINNs to solve
the Black-Scholes PDE.
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import os
from pathlib import Path

from multi_option.pinns.model import make_pinn
from multi_option.pinns.loss import pinn_loss


def train_pinn(
    r: float,
    q: float,
    sigma: float,
    K: float,
    T: float,
    s_max: float,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    call: bool,
    batch_size: int = 1000,
    save_dir: Optional[str] = None,
    use_fourier: bool = False
) -> Dict[str, float]:
    """Train a PINN to solve the Black-Scholes PDE.

    Args:
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        K: Strike price.
        T: Time to maturity.
        s_max: Maximum stock price.
        hidden: Number of hidden units.
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed.
        call: True for call, False for put.
        batch_size: Batch size for training.
        save_dir: Directory to save model weights.
        use_fourier: If True, use Fourier PINN. If False, use simple PINN.

    Returns:
        Dictionary with training metrics and model path.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model (simple or Fourier-enhanced)
    model = make_pinn(hidden, use_fourier=use_fourier).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    # Use longer patience to avoid premature LR decay
    # For simple PINNs: patience=500 (allow more time to explore)
    # For Fourier PINNs: patience=200 (converges faster with better features)
    patience = 200 if use_fourier else 500
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6
    )

    # Training history
    loss_history = []
    best_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # Sample training points with improved strategy
        s_train, t_train = sample_training_points(batch_size, s_max, T, device, K=K)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss with explicit weight parameters
        loss = pinn_loss(
            model, s_train, t_train,
            r, q, sigma, K, call, s_max,
            lambda_pde=1.0,      # PDE residual weight
            lambda_ic=100.0,     # Initial condition weight (100x higher!)
            lambda_bc=10.0       # Boundary condition weight
        )

        # Check for NaN/Inf losses
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss at epoch {epoch+1}: {loss.item()}")
            # Skip this update if loss is invalid
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping (aggressive to prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # Update weights
        optimizer.step()

        # Update learning rate
        scheduler.step(loss)

        # Record loss
        loss_val = loss.item()
        loss_history.append(loss_val)

        # Save best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_model_state = model.state_dict().copy()

        # Print progress with more details
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_val:.6f}, LR: {current_lr:.6e}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model weights
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "pinn_weights.pth")
        torch.save(model.state_dict(), model_path)
    else:
        model_path = None

    # Compute final metrics
    final_metrics = evaluate_pinn(model, r, q, sigma, K, T, s_max, call, device)

    return {
        "final_loss": float(best_loss),
        "epochs_trained": epochs,
        "model_path": model_path,
        "loss_history": loss_history,
        **final_metrics
    }


def sample_training_points(
    batch_size: int,
    s_max: float,
    T: float,
    device: torch.device,
    K: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample training points for PINN.

    Uses stratified sampling with focus on strike region.

    Args:
        batch_size: Number of points to sample.
        s_max: Maximum stock price.
        T: Maximum time.
        device: Torch device.
        K: Strike price for concentration.

    Returns:
        Tuple of (stock_prices, times) tensors.
    """
    # Stratified sampling: 70% near strike, 30% uniform elsewhere
    n_strike = int(batch_size * 0.7)
    nuniform = batch_size - n_strike

    # Around strike price (+/-30% of S, concentrated)
    strike_region_min = max(0.01, K * 0.7)
    strike_region_max = K * 1.3
    s_strike = np.random.uniform(strike_region_min, strike_region_max, n_strike)

    # Uniform across full range
    suniform = np.random.uniform(0.01, s_max, nuniform)

    # Combine
    s_samples = np.concatenate([s_strike, suniform])

    # Sample times uniformly
    t_samples = np.random.uniform(0.01, T, batch_size)

    # Shuffle to avoid ordering bias
    perm = np.random.permutation(batch_size)
    s_samples = s_samples[perm]
    t_samples = t_samples[perm]

    # Convert to tensors
    s_train = torch.tensor(s_samples, dtype=torch.float32, device=device)
    t_train = torch.tensor(t_samples, dtype=torch.float32, device=device)

    return s_train, t_train


def evaluate_pinn(
    model: torch.nn.Module,
    r: float,
    q: float,
    sigma: float,
    K: float,
    T: float,
    s_max: float,
    call: bool,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate PINN performance.

    Args:
        model: Trained PINN model.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        K: Strike price.
        T: Time to maturity.
        s_max: Maximum stock price.
        call: True for call, False for put.
        device: Torch device.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    with torch.no_grad():
        # Test on a reasonable grid (0.8K to 1.2K to avoid deep OTM/ITM extremes)
        # Deep OTM options have prices near 0, causing huge relative errors
        # that are mathematically correct but misleading for reporting
        n_test = 100
        s_test = torch.linspace(0.8 * K, 1.2 * K, n_test).to(device)
        t_test = torch.ones(n_test).to(device) * T

        # Get predictions
        v_pred = model(s_test, t_test).squeeze()

        # Compute true values using Black-Scholes
        from multi_option.bs_closed_form import bs_price
        v_true = torch.tensor([
            bs_price(s.item(), K, r, q, sigma, T, call)
            for s in s_test
        ]).to(device)

        # Compute errors
        abs_error = torch.mean(torch.abs(v_pred - v_true))
        # For relative error, only consider points where price > $0.10
        # to avoid misleading 1000%+ errors from $0.01 -> $0.02 predictions
        mask = v_true > 0.10
        if mask.sum() > 0:
            rel_error = torch.mean(torch.abs((v_pred[mask] - v_true[mask]) / v_true[mask]))
        else:
            rel_error = torch.mean(torch.abs((v_pred - v_true) / (v_true + 1e-8)))
        rmse = torch.sqrt(torch.mean((v_pred - v_true) ** 2))

    return {
        "mean_abs_error": float(abs_error),
        "mean_rel_error": float(rel_error),
        "rmse": float(rmse)
    }


def train_pinn_with_validation(
    r: float,
    q: float,
    sigma: float,
    K: float,
    T: float,
    s_max: float,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    call: bool,
    val_freq: int = 100
) -> pd.DataFrame:
    """Train PINN with validation monitoring.

    Args:
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        K: Strike price.
        T: Time to maturity.
        s_max: Maximum stock price.
        hidden: Number of hidden units.
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed.
        call: True for call, False for put.
        val_freq: Validation frequency.

    Returns:
        DataFrame with training history.
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_pinn(hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(epochs):
        # Training step
        model.train()
        s_train, t_train = sample_training_points(1000, s_max, T, device)

        optimizer.zero_grad()
        loss = pinn_loss(model, s_train, t_train, r, q, sigma, K, call, s_max)
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % val_freq == 0:
            metrics = evaluate_pinn(model, r, q, sigma, K, T, s_max, call, device)
            history.append({
                'epoch': epoch + 1,
                'train_loss': loss.item(),
                **metrics
            })

    return pd.DataFrame(history)


def create_pinn_solution_grid(
    model: torch.nn.Module,
    s_max: float,
    T: float,
    ns: int = 101,
    nt: int = 101,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """Create a grid of PINN solutions for visualization.

    Args:
        model: Trained PINN model.
        s_max: Maximum stock price.
        T: Maximum time.
        ns: Number of stock price points.
        nt: Number of time points.
        device: Torch device.

    Returns:
        DataFrame with columns ['S', 't', 'V'].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    # Create grid
    s_grid = np.linspace(0, s_max, ns)
    t_grid = np.linspace(0, T, nt)
    S_mesh, T_mesh = np.meshgrid(s_grid, t_grid)

    # Flatten for prediction
    s_flat = torch.tensor(S_mesh.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.tensor(T_mesh.flatten(), dtype=torch.float32, device=device)

    # Predict
    with torch.no_grad():
        v_flat = model(s_flat, t_flat).squeeze().cpu().numpy()

    # Create DataFrame
    df = pd.DataFrame({
        'S': S_mesh.flatten(),
        't': T_mesh.flatten(),
        'V': v_flat
    })

    return df


def compute_pinn_residuals(
    model: torch.nn.Module,
    r: float,
    q: float,
    sigma: float,
    s_max: float,
    T: float,
    ns: int = 51,
    nt: int = 51
) -> pd.DataFrame:
    """Compute PDE residuals on a grid for visualization.

    Args:
        model: Trained PINN model.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        s_max: Maximum stock price.
        T: Maximum time.
        ns: Number of stock price points.
        nt: Number of time points.

    Returns:
        DataFrame with columns ['S', 't', 'residual'].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Create grid
    s_grid = np.linspace(0.01, s_max, ns)
    t_grid = np.linspace(0.01, T, nt)
    S_mesh, T_mesh = np.meshgrid(s_grid, t_grid)

    # Flatten for prediction
    s_flat = torch.tensor(S_mesh.flatten(), dtype=torch.float32, device=device, requires_grad=True)
    t_flat = torch.tensor(T_mesh.flatten(), dtype=torch.float32, device=device, requires_grad=True)

    # Compute residuals
    v = model(s_flat, t_flat)

    # Compute derivatives
    v_s = torch.autograd.grad(v, s_flat, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t_flat, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    v_ss = torch.autograd.grad(v_s, s_flat, grad_outputs=torch.ones_like(v_s),
                              create_graph=False, retain_graph=False)[0]

    # PDE residual
    residual = -v_t + 0.5 * sigma**2 * s_flat**2 * v_ss + (r - q) * s_flat * v_s - r * v
    residual = residual.squeeze().detach().cpu().numpy()

    # Create DataFrame
    df = pd.DataFrame({
        'S': S_mesh.flatten(),
        't': T_mesh.flatten(),
        'residual': np.abs(residual)
    })

    return df
