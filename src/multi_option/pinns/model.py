"""Physics-Informed Neural Network model for option pricing.

This module defines the neural network architecture for solving
the Black-Scholes PDE using PINNs.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PINN(nn.Module):
    """Physics-Informed Neural Network for Black-Scholes PDE."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        activation: str = "tanh"
    ):
        """Initialize PINN model.

        Args:
            hidden_dim: Number of neurons in hidden layers.
            n_hidden_layers: Number of hidden layers.
            activation: Activation function ('tanh', 'relu', or 'sigmoid').
        """
        super(PINN, self).__init__()

        # Input dimension: (S, t)
        input_dim = 2
        # Output dimension: V(S, t)
        output_dim = 1

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function.

        Args:
            activation: Name of activation function.

        Returns:
            Activation module.
        """
        if activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            s: Stock prices tensor.
            t: Time tensor.

        Returns:
            Option values V(S, t).
        """
        if s.dim() == 0:
            s = s.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Stack inputs
        x = torch.cat([s, t], dim=1)
        return self.network(x)


def make_pinn(hidden: int, use_fourier: bool = False) -> torch.nn.Module:
    """Create a PINN model with specified hidden dimension.

    Args:
        hidden: Number of hidden units per layer.
        use_fourier: If True, use BasePINN with Fourier features.
                     If False, use simple PINN without Fourier features.

    Returns:
        PINN model (simple or Fourier-enhanced).
    """
    if use_fourier:
        from .architectures import BasePINN
        return BasePINN(hidden_dim=hidden, num_layers=5, fourier_mapping_size=64)
    else:
        # Simple PINN without Fourier features (for baseline comparison)
        return PINN(hidden_dim=hidden, n_hidden_layers=4, activation="tanh")


class AdaptivePINN(nn.Module):
    """Adaptive PINN with attention mechanism for better boundary handling."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        use_attention: bool = True
    ):
        """Initialize Adaptive PINN model.

        Args:
            hidden_dim: Number of neurons in hidden layers.
            n_hidden_layers: Number of hidden layers.
            use_attention: Whether to use attention mechanism.
        """
        super(AdaptivePINN, self).__init__()

        self.use_attention = use_attention
        input_dim = 2
        output_dim = 1

        # Main network
        self.main_network = PINN(hidden_dim, n_hidden_layers, "tanh")

        if use_attention:
            # Attention network for adaptive weighting
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

            # Boundary network
            self.boundary_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, output_dim)
            )

            self._initialize_attention_weights()

    def _initialize_attention_weights(self):
        """Initialize attention and boundary network weights."""
        for network in [self.attention, self.boundary_network]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adaptive network.

        Args:
            s: Stock prices tensor.
            t: Time tensor.

        Returns:
            Option values V(S, t).
        """
        if not self.use_attention:
            return self.main_network(s, t)

        # Main network prediction
        v_main = self.main_network(s, t)

        # Attention weights
        x = torch.stack([s, t], dim=1)
        alpha = self.attention(x)

        # Boundary correction
        v_boundary = self.boundary_network(x)

        # Weighted combination
        v = alpha * v_main + (1 - alpha) * v_boundary

        return v


class ResidualPINN(nn.Module):
    """Residual PINN with skip connections for improved training."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_blocks: int = 2
    ):
        """Initialize Residual PINN model.

        Args:
            hidden_dim: Number of neurons in hidden layers.
            n_blocks: Number of residual blocks.
        """
        super(ResidualPINN, self).__init__()

        input_dim = 2
        output_dim = 1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.xavier_normal_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual network.

        Args:
            s: Stock prices tensor.
            t: Time tensor.

        Returns:
            Option values V(S, t).
        """
        # Input projection
        x = torch.stack([s, t], dim=1)
        h = torch.tanh(self.input_proj(x))

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)

        # Output projection
        v = self.output_proj(h)

        return v


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, hidden_dim: int):
        """Initialize residual block.

        Args:
            hidden_dim: Dimension of hidden layers.
        """
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize block weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor.

        Returns:
            Output with residual connection.
        """
        return torch.tanh(x + self.layers(x))
