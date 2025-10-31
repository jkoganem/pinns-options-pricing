"""
PINN Architecture Definitions for Exotic Options.

This module implements Physics-Informed Neural Network architectures
for pricing exotic options including American, Barrier, Asian, and Lookback.

Each architecture solves the appropriate PDE with specialized boundary conditions
and constraints:

1. American Options: Black-Scholes PDE + early exercise constraint (LCP)
2. Barrier Options: Black-Scholes PDE + barrier boundary condition
3. Asian Options: Extended 3D PDE with running average state
4. Lookback Options: Extended 3D PDE with running max/min state

References:
    - Raissi et al. (2019) J. Comput. Phys. - PINN framework
    - Han, Jentzen & E (2018) PNAS - High-dimensional PDEs
    - Sirignano & Spiliopoulos (2018) - DGM for PDEs
    - Song et al. (2024) Scientific Reports - American options with PINNs
    - Chen et al. (2020) arXiv:2009.07971 - American options

Author: Multi-Option Pricing Team
Date: October 2025
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

from .utils import FourierFeatureMapping, ResidualBlock, initialize_weights
from .adaptive_weights import AdaptiveLossWeights


class BarrierPINN(nn.Module):
    """
    PINN for Barrier Options (Up-and-Out Call).

    Solves the Black-Scholes PDE with Dirichlet boundary condition at barrier:
        dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS - rV
        V(S=H, tau) = 0  (barrier condition)
        V(S, 0) = max(S - K, 0)  (terminal condition)

    where H is the barrier level.

    Key Features:
        - Hard-enforces barrier condition using trial solution
        - Trial solution: V(S,tau) = (1 - S/H)^p * N(S,tau) for S < H
        - Automatic barrier satisfaction by construction
        - Supports rebate values

    Architecture:
        Same as AdaptiveWeightPINN (5x96) but with barrier enforcement

    References:
        - Jang & Lee (2021) Appl. Math. Comput. - Mesh-free barrier pricing
        - Raissi & Karniadakis (2019) arXiv:1912.10091 - PINNs for barriers

    Args:
        barrier_level: Barrier price H
        rebate: Rebate payment when barrier is hit (default: 0.0)
        barrier_power: Power p in trial solution (default: 2.0)
        bs_reference: Black-Scholes reference for normalization
        fourier_mapping_size: Number of Fourier features (default: 64)
        hidden_dim: Hidden layer dimension (default: 96)
        num_layers: Number of hidden layers (default: 5)
        s_max: Maximum stock price for normalization (default: 300.0)
        T: Maximum time to maturity (default: 1.0)
    """

    def __init__(
        self,
        barrier_level: float,
        rebate: float = 0.0,
        barrier_power: float = 2.0,
        bs_reference: float = 10.0,
        fourier_mapping_size: int = 64,
        fourier_scale: float = 1.0,
        hidden_dim: int = 96,
        num_layers: int = 5,
        s_max: float = 300.0,
        T: float = 1.0
    ) -> None:
        super().__init__()
        self.barrier_level = barrier_level
        self.rebate = rebate
        self.barrier_power = barrier_power
        self.s_max = s_max
        self.T = T
        self.bs_reference = bs_reference

        # Fourier feature mapping
        self.fourier = FourierFeatureMapping(
            input_dim=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        )

        # Input layer
        fourier_output_dim = 2 * fourier_mapping_size
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.Tanh()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        initialize_weights(self, method='xavier_normal', gain=1.0)

    def forward(
        self,
        S: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with barrier enforcement.

        Uses trial solution: V(S,tau) = (1 - S/H)^p * N(S,tau) + rebate * I(S>=H)
        This ensures V(H,tau) = rebate automatically.

        Args:
            S: Stock price tensor
            tau: Time to maturity tensor

        Returns:
            Option value tensor satisfying barrier condition
        """
        # Normalize inputs
        S_norm = S / self.s_max
        tau_norm = tau / self.T

        if S_norm.dim() == 0:
            S_norm = S_norm.unsqueeze(0)
        if tau_norm.dim() == 0:
            tau_norm = tau_norm.unsqueeze(0)
        if S_norm.dim() == 1:
            S_norm = S_norm.unsqueeze(1)
        if tau_norm.dim() == 1:
            tau_norm = tau_norm.unsqueeze(1)

        # Stack inputs
        x = torch.cat([S_norm, tau_norm], dim=1)

        # Fourier features
        x = self.fourier(x)

        # Input layer
        x = self.input_layer(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Raw network output
        N = torch.abs(self.output_layer(x))

        # Trial solution enforcing barrier condition
        # V(S,tau) = (1 - S/H)^p * N(S,tau) for S < H
        #        = rebate for S >= H
        barrier_factor = torch.clamp(1.0 - S / self.barrier_level, min=0.0, max=1.0)
        barrier_weight = torch.pow(barrier_factor, self.barrier_power)

        # Combine: interior value weighted by barrier distance
        V = barrier_weight * N * self.bs_reference

        # Add rebate for points at/beyond barrier
        at_barrier = (S >= self.barrier_level).float()
        V = V + at_barrier * self.rebate

        return V


class AmericanPINN(nn.Module):
    """
    PINN for American Options using Penalty Method.

    Solves the Linear Complementarity Problem (LCP):
        min{ dV/dtau - LV, V - g(S) } = 0

    where:
        L is the Black-Scholes operator
        g(S) = max(K - S, 0) is the intrinsic value (American put)

    Implementation uses penalty method:
        Total Loss = PDE Loss + lambda_penalty * max(0, g(S) - V)2

    This enforces V(S,tau) >= g(S) (early exercise constraint) via soft penalty.

    Key Features:
        - Penalty-based early exercise enforcement
        - Adaptive penalty weight lambda
        - Can learn optimal exercise boundary implicitly
        - Works with AdaptiveLossWeights for multi-scale problems

    Architecture:
        Same as AdaptiveWeightPINN (5x96) with penalty loss

    References:
        - Song et al. (2024) Scientific Reports - PINN for fractional BS American
        - Chen et al. (2020) arXiv:2009.07971 - American options with PINNs
        - Beck et al. (2021) J. Sci. Comput. - Deep learning for BSDEs

    Args:
        strike: Strike price K
        option_type: 'put' or 'call'
        penalty_weight: Initial penalty weight lambda (default: 100.0)
        bs_reference: Black-Scholes reference for normalization
        fourier_mapping_size: Number of Fourier features (default: 64)
        hidden_dim: Hidden layer dimension (default: 96)
        num_layers: Number of hidden layers (default: 5)
        s_max: Maximum stock price for normalization (default: 300.0)
        T: Maximum time to maturity (default: 1.0)
    """

    def __init__(
        self,
        strike: float,
        option_type: str = 'put',
        penalty_weight: float = 100.0,
        bs_reference: float = 10.0,
        fourier_mapping_size: int = 64,
        fourier_scale: float = 1.0,
        hidden_dim: int = 96,
        num_layers: int = 5,
        s_max: float = 300.0,
        T: float = 1.0
    ) -> None:
        super().__init__()
        self.strike = strike
        self.option_type = option_type.lower()
        self.penalty_weight = penalty_weight
        self.s_max = s_max
        self.T = T
        self.bs_reference = bs_reference

        if self.option_type not in ['put', 'call']:
            raise ValueError("option_type must be 'put' or 'call'")

        # Fourier feature mapping
        self.fourier = FourierFeatureMapping(
            input_dim=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        )

        # Input layer
        fourier_output_dim = 2 * fourier_mapping_size
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.Tanh()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        initialize_weights(self, method='xavier_normal', gain=1.0)

    def forward(
        self,
        S: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for American option value.

        Args:
            S: Stock price tensor
            tau: Time to maturity tensor

        Returns:
            Option value tensor (should satisfy V >= intrinsic_value)
        """
        # Normalize inputs
        S_norm = S / self.s_max
        tau_norm = tau / self.T

        if S_norm.dim() == 0:
            S_norm = S_norm.unsqueeze(0)
        if tau_norm.dim() == 0:
            tau_norm = tau_norm.unsqueeze(0)
        if S_norm.dim() == 1:
            S_norm = S_norm.unsqueeze(1)
        if tau_norm.dim() == 1:
            tau_norm = tau_norm.unsqueeze(1)

        # Stack inputs
        x = torch.cat([S_norm, tau_norm], dim=1)

        # Fourier features
        x = self.fourier(x)

        # Input layer
        x = self.input_layer(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output with normalization
        normalized_output = torch.abs(self.output_layer(x))
        V = normalized_output * self.bs_reference

        return V

    def intrinsic_value(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic value g(S).

        Args:
            S: Stock price tensor

        Returns:
            Intrinsic value tensor
        """
        if self.option_type == 'put':
            return torch.clamp(self.strike - S, min=0.0)
        else:  # call
            return torch.clamp(S - self.strike, min=0.0)

    def penalty_loss(self, S: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty loss for early exercise constraint.

        Penalty: lambda * max(0, g(S) - V)2

        Args:
            S: Stock price tensor
            tau: Time to maturity tensor

        Returns:
            Penalty loss value
        """
        V = self.forward(S, tau)
        intrinsic = self.intrinsic_value(S)

        # Penalize V < intrinsic
        violation = torch.clamp(intrinsic - V, min=0.0)
        penalty = self.penalty_weight * torch.mean(violation ** 2)

        return penalty


class AsianPINN(nn.Module):
    """
    PINN for Asian Options (Path-Dependent Average).

    Solves extended 3D PDE with running average state:
        dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS + S dV/dI - rV

    where I_t = integral0t Su du is the running integral of stock price.

    The payoff at maturity is phi(AT) where AT = IT / T is the average.

    Key Features:
        - 3D input: (S, tau, I) or (S, tau, A)
        - Advection term S dV/dI in PDE
        - Path-dependent payoff

    Architecture:
        Extended to 3D input with same structure as AdaptiveWeightPINN

    References:
        - Kissas & Karniadakis (2022) arXiv:2204.01243 - Path-dependent options
        - Sirignano & Spiliopoulos (2020) Quant. Finance - DGM for Asian/Lookback
        - Chen & Choi (2023) NeurIPS ML Finance - Asian option embedding

    Args:
        strike: Strike price K
        option_type: 'put' or 'call'
        averaging_type: 'arithmetic' or 'geometric'
        bs_reference: Reference price for normalization
        fourier_mapping_size: Number of Fourier features (default: 64)
        hidden_dim: Hidden layer dimension (default: 128, larger for 3D)
        num_layers: Number of hidden layers (default: 6, deeper for 3D)
        s_max: Maximum stock price for normalization (default: 300.0)
        i_max: Maximum integral value for normalization (default: 300.0)
        T: Maximum time to maturity (default: 1.0)
    """

    def __init__(
        self,
        strike: float,
        option_type: str = 'call',
        averaging_type: str = 'arithmetic',
        bs_reference: float = 10.0,
        fourier_mapping_size: int = 64,
        fourier_scale: float = 1.0,
        hidden_dim: int = 128,
        num_layers: int = 6,
        s_max: float = 300.0,
        i_max: float = 300.0,
        T: float = 1.0
    ) -> None:
        super().__init__()
        self.strike = strike
        self.option_type = option_type.lower()
        self.averaging_type = averaging_type.lower()
        self.s_max = s_max
        self.i_max = i_max
        self.T = T
        self.bs_reference = bs_reference

        # 3D input: (S, tau, I)
        input_dim = 3

        # Fourier feature mapping for 3D input
        self.fourier = FourierFeatureMapping(
            input_dim=input_dim,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        )

        # Input layer
        fourier_output_dim = 2 * fourier_mapping_size
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.Tanh()
        )

        # Residual blocks (more layers for 3D complexity)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        initialize_weights(self, method='xavier_normal', gain=1.5)

    def forward(
        self,
        S: torch.Tensor,
        tau: torch.Tensor,
        I: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Asian option value.

        Args:
            S: Stock price tensor
            tau: Time to maturity tensor
            I: Running integral tensor (integral0t Su du)

        Returns:
            Option value tensor V(S, tau, I)
        """
        # Normalize inputs
        S_norm = S / self.s_max
        tau_norm = tau / self.T
        I_norm = I / self.i_max

        tensors = []
        for tensor in (S_norm, tau_norm, I_norm):
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            tensors.append(tensor)

        # Stack 3D inputs
        x = torch.cat(tensors, dim=1)

        # Fourier features
        x = self.fourier(x)

        # Input layer
        x = self.input_layer(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output with normalization
        normalized_output = torch.abs(self.output_layer(x))
        V = normalized_output * self.bs_reference

        return V

    def payoff(
        self,
        S: torch.Tensor,
        I: torch.Tensor,
        T: float
    ) -> torch.Tensor:
        """
        Compute Asian option payoff.

        Args:
            S: Current stock price
            I: Running integral value
            T: Time to maturity (for computing average)

        Returns:
            Payoff tensor
        """
        # Average price
        A = I / T

        if self.option_type == 'call':
            return torch.clamp(A - self.strike, min=0.0)
        else:  # put
            return torch.clamp(self.strike - A, min=0.0)


class LookbackPINN(nn.Module):
    """
    PINN for Lookback Options (Path-Dependent Extrema).

    Solves extended 3D PDE with running maximum state:
        dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS - rV

    on domain 0 < S <= M (where M_t = max_{u<=t} Su).

    Boundary condition at S = M: dV/dS(M, M, tau) = 0 (smooth-fit/reflecting)

    Key Features:
        - 3D input: (S, tau, M) with constraint S <= M
        - Boundary condition along diagonal S = M
        - Payoff: phi(S, M) = M - K (lookback call)

    Architecture:
        Extended to 3D input, larger capacity for diagonal boundary

    References:
        - Liu & Lin (2022) arXiv:2208.01337 - Lookback under stochastic vol
        - Wilmott, Howison & Dewynne (1995) - Math of Financial Derivatives
        - Broadie & Glasserman (1997) J. Comput. Finance - Lookback PDEs
        - Lu et al. (2021) DeepXDE (SIAM Review)

    Args:
        strike: Strike price K
        option_type: 'call' or 'put'
        lookback_type: 'fixed' or 'floating'
        bs_reference: Reference price for normalization
        fourier_mapping_size: Number of Fourier features (default: 64)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of hidden layers (default: 6)
        s_max: Maximum stock price for normalization (default: 300.0)
        m_max: Maximum running max for normalization (default: 300.0)
        T: Maximum time to maturity (default: 1.0)
    """

    def __init__(
        self,
        strike: float,
        option_type: str = 'call',
        lookback_type: str = 'floating',
        bs_reference: float = 10.0,
        fourier_mapping_size: int = 64,
        fourier_scale: float = 1.0,
        hidden_dim: int = 128,
        num_layers: int = 6,
        s_max: float = 300.0,
        m_max: float = 300.0,
        T: float = 1.0
    ) -> None:
        super().__init__()
        self.strike = strike
        self.option_type = option_type.lower()
        self.lookback_type = lookback_type.lower()
        self.s_max = s_max
        self.m_max = m_max
        self.T = T
        self.bs_reference = bs_reference

        # 3D input: (S, tau, M)
        input_dim = 3

        # Fourier feature mapping for 3D input
        self.fourier = FourierFeatureMapping(
            input_dim=input_dim,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale
        )

        # Input layer
        fourier_output_dim = 2 * fourier_mapping_size
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.Tanh()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        initialize_weights(self, method='xavier_normal', gain=1.5)

    def forward(
        self,
        S: torch.Tensor,
        tau: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Lookback option value.

        Args:
            S: Stock price tensor
            tau: Time to maturity tensor
            M: Running maximum tensor (M_t = max_{u<=t} Su)

        Returns:
            Option value tensor V(S, tau, M)
        """
        # Enforce constraint S <= M
        S_clamped = torch.clamp(S, max=M)

        # Normalize inputs
        S_norm = S_clamped / self.s_max
        tau_norm = tau / self.T
        M_norm = M / self.m_max

        tensors = []
        for tensor in (S_norm, tau_norm, M_norm):
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            tensors.append(tensor)

        # Stack 3D inputs
        x = torch.cat(tensors, dim=1)

        # Fourier features
        x = self.fourier(x)

        # Input layer
        x = self.input_layer(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output with normalization
        normalized_output = torch.abs(self.output_layer(x))
        V = normalized_output * self.bs_reference

        return V

    def payoff(
        self,
        S: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lookback option payoff.

        Args:
            S: Current stock price
            M: Running maximum (or minimum for put)

        Returns:
            Payoff tensor
        """
        if self.lookback_type == 'floating':
            if self.option_type == 'call':
                # Floating lookback call: ST - min(S)
                # For max formulation: M - (M - ST) = ST if using min tracking
                # Here M = max, so we need different payoff
                return M - S  # Simplified for max-based
            else:  # put
                return M - S
        else:  # fixed strike
            if self.option_type == 'call':
                return torch.clamp(M - self.strike, min=0.0)
            else:  # put
                return torch.clamp(self.strike - M, min=0.0)


def create_exotic_pinn(
    option_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create exotic PINN architectures.

    Args:
        option_type: Type of exotic option ('barrier', 'american', 'asian', 'lookback')
        **kwargs: Architecture-specific arguments

    Returns:
        Exotic PINN model instance

    Example:
        >>> # Create barrier option PINN
        >>> model = create_exotic_pinn('barrier', barrier_level=120.0, strike=100.0)
        >>>
        >>> # Create American option PINN
        >>> model = create_exotic_pinn('american', strike=100.0, option_type='put')
        >>>
        >>> # Create Asian option PINN
        >>> model = create_exotic_pinn('asian', strike=100.0)
        >>>
        >>> # Create Lookback option PINN
        >>> model = create_exotic_pinn('lookback', strike=100.0)
    """
    architectures = {
        'barrier': BarrierPINN,
        'american': AmericanPINN,
        'asian': AsianPINN,
        'lookback': LookbackPINN
    }

    option_type = option_type.lower()
    if option_type not in architectures:
        raise ValueError(
            f"Unknown exotic option type '{option_type}'. "
            f"Choose from: {list(architectures.keys())}"
        )

    return architectures[option_type](**kwargs)
