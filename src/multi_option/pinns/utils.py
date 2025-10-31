"""
Utility functions and components for Physics-Informed Neural Networks.

This module provides reusable components for PINN training including:
- Fourier feature mapping for high-frequency learning
- Exponential moving average (EMA) for parameter stabilization
- Early stopping logic
- Custom learning rate schedulers

Author: Multi-Option Pricing Team
Date: October 2025
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class FourierFeatureMapping(nn.Module):
    """
    Fourier feature mapping for improved high-frequency learning in PINNs.

    Based on: Tancik et al. (2020) "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains"

    Maps input coordinates through random Fourier features:
        [sin(2pi B x), cos(2pi B x)]
    where B is a random Gaussian matrix.

    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for (S, tau))
        mapping_size: Number of Fourier features
        scale: Standard deviation for random Fourier matrix
    """

    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 64,
        scale: float = 1.0
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size

        # Random Fourier matrix (not trainable)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Fourier features of shape (batch_size, 2 * mapping_size)
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for improved gradient flow.

    Architecture: x -> Linear -> Tanh -> Linear -> (+x) -> output

    Args:
        hidden_dim: Dimension of hidden layers
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block with skip connection."""
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x + residual


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a moving average of model parameters for more stable predictions.
    Often improves generalization and reduces variance in final model.

    Args:
        model: PyTorch model to track
        decay: EMA decay rate (typically 0.995-0.999)

    Example:
        >>> model = MyPINN()
        >>> ema = ExponentialMovingAverage(model, decay=0.999)
        >>> for epoch in range(num_epochs):
        >>>     # ... training step ...
        >>>     ema.update()
        >>> ema.apply_shadow()  # Use EMA parameters for inference
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Apply EMA parameters to model (use for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original parameters (use after inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for accuracy

    Example:
        >>> early_stop = EarlyStopping(patience=2000)
        >>> for epoch in range(max_epochs):
        >>>     loss = train_step()
        >>>     if early_stop(loss, epoch):
        >>>         print(f"Early stopping at epoch {epoch}")
        >>>         break
    """

    def __init__(
        self,
        patience: int = 2000,
        min_delta: float = 1e-7,
        mode: str = 'min'
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, loss: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch
            return False

        # Check for improvement
        if self.mode == 'min':
            improved = loss < (self.best_loss - self.min_delta)
        else:
            improved = loss > (self.best_loss + self.min_delta)

        if improved:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def state_dict(self) -> Dict:
        """Return state for checkpointing."""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch
        }


def initialize_weights(
    model: nn.Module,
    method: str = 'xavier_normal',
    gain: float = 1.0
) -> None:
    """
    Initialize model weights with specified method.

    Args:
        model: PyTorch model to initialize
        method: Initialization method ('xavier_normal', 'xavieruniform', 'kaiming')
        gain: Gain factor for Xavier initialization
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif method == 'xavieruniform':
                nn.init.xavieruniform_(m.weight, gain=gain)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_inputs(
    S: torch.Tensor,
    tau: torch.Tensor,
    s_max: float,
    T: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize inputs to [0, 1] range for better neural network training.

    Args:
        S: Stock price tensor
        tau: Time to maturity tensor
        s_max: Maximum stock price for normalization
        T: Maximum time to maturity

    Returns:
        Normalized (S, tau) tensors
    """
    S_norm = S / s_max
    tau_norm = tau / T
    return S_norm, tau_norm


def compute_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    q: float,
    sigma: float
) -> torch.Tensor:
    """
    Compute Black-Scholes PDE residual for European call option.

    PDE: dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS - rV

    Args:
        V: Option value tensor (requires grad)
        S: Stock price tensor (requires grad)
        tau: Time to maturity tensor (requires grad)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        PDE residual tensor
    """
    # First derivatives
    V_S = torch.autograd.grad(
        V, S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]

    V_tau = torch.autograd.grad(
        V, tau,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]

    # Second derivative
    V_SS = torch.autograd.grad(
        V_S, S,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
        retain_graph=True
    )[0]

    # Black-Scholes PDE
    pde = V_tau - (0.5 * sigma**2 * S**2 * V_SS + (r - q) * S * V_S - r * V)

    return pde
