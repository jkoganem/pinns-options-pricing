"""
Loss Functions for Exotic Option PINNs.

This module implements specialized loss functions for training PINNs
on exotic options including Barrier, American, Asian, and Lookback options.

Each loss function enforces:
1. PDE residual in interior domain
2. Terminal/initial conditions
3. Boundary conditions specific to the option type
4. Special constraints (barrier, early exercise, path-dependence)

References:
    - Raissi et al. (2019) J. Comput. Phys.
    - Han, Jentzen & E (2018) PNAS
    - Sirignano & Spiliopoulos (2018) J. Comput. Phys.

Author: Multi-Option Pricing Team
Date: October 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def compute_bs_operator(
    V: torch.Tensor,
    S: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    q: float,
    sigma: float
) -> torch.Tensor:
    """
    Compute Black-Scholes operator LV.

    LV = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS - rV

    Args:
        V: Option value tensor
        S: Stock price tensor (requires grad)
        tau: Time to maturity tensor (requires grad)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        Black-Scholes operator applied to V
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

    # Second derivative w.r.t. S
    V_SS = torch.autograd.grad(
        V_S, S,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
        retain_graph=True
    )[0]

    # Black-Scholes operator
    LV = 0.5 * sigma**2 * S**2 * V_SS + (r - q) * S * V_S - r * V

    return LV, V_tau


class BarrierLoss(nn.Module):
    """
    Loss function for Barrier Option PINNs.

    Components:
        1. PDE residual: |dV/dtau - LV|2
        2. Terminal condition: |V(S,0) - payoff(S)|2
        3. Barrier boundary: |V(H,tau) - rebate|2 (enforced explicitly or via trial solution)
        4. Lower boundary: V(0,tau) = 0

    Args:
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        strike: Strike price K
        barrier: Barrier level H
        T: Time to maturity
        rebate: Rebate value (default: 0.0)
        option_type: 'call' or 'put'
        barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        strike: float,
        barrier: float,
        T: float,
        rebate: float = 0.0,
        option_type: str = 'call',
        barrier_type: str = 'up-and-out'
    ):
        super().__init__()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.strike = strike
        self.barrier = barrier
        self.T = T
        self.rebate = rebate
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()

        # Loss weights
        self.w_pde = 1.0
        self.w_terminal = 10.0
        self.w_barrier = 50.0
        self.w_boundary = 10.0

    def forward(
        self,
        model: nn.Module,
        S_pde: torch.Tensor,
        tau_pde: torch.Tensor,
        S_terminal: torch.Tensor,
        S_barrier: torch.Tensor,
        tau_barrier: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            model: PINN model
            S_pde: Stock prices for PDE residual (interior points)
            tau_pde: Time to maturity for PDE residual
            S_terminal: Stock prices for terminal condition
            S_barrier: Stock prices at barrier
            tau_barrier: Time to maturity at barrier

        Returns:
            Dictionary of loss components
        """
        # Enable gradients
        S_pde.requires_grad_(True)
        tau_pde.requires_grad_(True)

        # 1. PDE residual loss
        V_pde = model(S_pde, tau_pde)
        LV, V_tau = compute_bs_operator(V_pde, S_pde, tau_pde, self.r, self.q, self.sigma)
        pde_residual = V_tau - LV
        loss_pde = torch.mean(pde_residual ** 2)

        # 2. Terminal condition loss (tau = 0)
        V_terminal = model(S_terminal, torch.zeros_like(S_terminal))
        if self.option_type == 'call':
            payoff = torch.clamp(S_terminal - self.strike, min=0.0)
        else:
            payoff = torch.clamp(self.strike - S_terminal, min=0.0)

        loss_terminal = torch.mean((V_terminal - payoff.unsqueeze(1)) ** 2)

        # 3. Barrier boundary loss (if not using trial solution)
        # Note: If using trial solution in BarrierPINN, this is automatically satisfied
        # We still compute it for monitoring
        S_barrier_tensor = torch.ones_like(tau_barrier) * self.barrier
        V_barrier = model(S_barrier_tensor, tau_barrier)
        rebate_tensor = torch.ones_like(V_barrier) * self.rebate
        loss_barrier = torch.mean((V_barrier - rebate_tensor) ** 2)

        # 4. Lower boundary loss (S -> 0)
        S_lower = torch.ones_like(tau_barrier) * 1e-6
        V_lower = model(S_lower, tau_barrier)
        if self.option_type == 'call':
            bc_lower = torch.zeros_like(V_lower)
        else:
            bc_lower = self.strike * torch.exp(-self.r * tau_barrier).unsqueeze(1)
        loss_boundary = torch.mean((V_lower - bc_lower) ** 2)

        # Total loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_terminal * loss_terminal +
            self.w_barrier * loss_barrier +
            self.w_boundary * loss_boundary
        )

        return {
            'total': total_loss,
            'pde': loss_pde,
            'terminal': loss_terminal,
            'barrier': loss_barrier,
            'boundary': loss_boundary
        }


class AmericanLoss(nn.Module):
    """
    Loss function for American Option PINNs using Penalty Method.

    Solves Linear Complementarity Problem (LCP):
        min{ dV/dtau - LV, V - g(S) } = 0

    Penalty formulation:
        L = w_pde * |ReLU(-PDE_residual)|2 +
            w_obstacle * |ReLU(g - V)|2 +
            w_complementarity * |(V - g) * ReLU(PDE_residual)|2 +
            w_terminal * |V(S,0) - g(S)|2 +
            w_boundary * boundary_loss

    Args:
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        strike: Strike price K
        T: Time to maturity
        option_type: 'call' or 'put'
        penalty_weight: Penalty weight lambda for obstacle constraint
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        strike: float,
        T: float,
        option_type: str = 'put',
        penalty_weight: float = 100.0
    ):
        super().__init__()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.strike = strike
        self.T = T
        self.option_type = option_type.lower()
        self.penalty_weight = penalty_weight

        # Loss weights
        self.w_pde = 1.0
        self.w_obstacle = penalty_weight
        self.w_complementarity = 10.0
        self.w_terminal = 10.0
        self.w_boundary = 10.0

    def intrinsic_value(self, S: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic value g(S)."""
        if self.option_type == 'put':
            return torch.clamp(self.strike - S, min=0.0)
        else:
            return torch.clamp(S - self.strike, min=0.0)

    def forward(
        self,
        model: nn.Module,
        S_pde: torch.Tensor,
        tau_pde: torch.Tensor,
        S_terminal: torch.Tensor,
        tau_boundary: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for American options.

        Args:
            model: PINN model
            S_pde: Stock prices for PDE residual
            tau_pde: Time to maturity for PDE residual
            S_terminal: Stock prices for terminal condition
            tau_boundary: Time to maturity for boundary

        Returns:
            Dictionary of loss components
        """
        # Enable gradients
        S_pde.requires_grad_(True)
        tau_pde.requires_grad_(True)

        # 1. PDE residual loss (continuation region)
        V_pde = model(S_pde, tau_pde)
        LV, V_tau = compute_bs_operator(V_pde, S_pde, tau_pde, self.r, self.q, self.sigma)
        pde_residual = V_tau - LV

        # Penalize negative residuals (should be >= 0 in continuation region)
        loss_pde = torch.mean(torch.relu(-pde_residual) ** 2)

        # 2. Obstacle constraint: V >= g(S)
        intrinsic = self.intrinsic_value(S_pde).unsqueeze(1)
        violation = torch.relu(intrinsic - V_pde)
        loss_obstacle = torch.mean(violation ** 2)

        # 3. Complementarity: (V - g) * L[V] ~ 0
        # In exercise region: V = g, so (V-g) = 0
        # In continuation region: L[V] = 0, so (V-g)*L[V] = 0
        complementarity = (V_pde - intrinsic) * torch.relu(pde_residual)
        loss_complementarity = torch.mean(complementarity ** 2)

        # 4. Terminal condition
        V_terminal = model(S_terminal, torch.zeros_like(S_terminal))
        payoff = self.intrinsic_value(S_terminal).unsqueeze(1)
        loss_terminal = torch.mean((V_terminal - payoff) ** 2)

        # 5. Boundary conditions
        # Lower boundary
        S_lower = torch.ones_like(tau_boundary) * 1e-6
        V_lower = model(S_lower, tau_boundary)
        if self.option_type == 'put':
            bc_lower = self.strike * torch.exp(-self.r * tau_boundary).unsqueeze(1)
        else:
            bc_lower = torch.zeros_like(V_lower)
        loss_boundary_lower = torch.mean((V_lower - bc_lower) ** 2)

        # Upper boundary (for put: V -> 0 as S -> infinity)
        Supper = torch.ones_like(tau_boundary) * 300.0
        Vupper = model(Supper, tau_boundary)
        if self.option_type == 'put':
            bcupper = torch.zeros_like(Vupper)
        else:
            bcupper = (Supper.unsqueeze(1) -
                       self.strike * torch.exp(-self.r * tau_boundary).unsqueeze(1))
        loss_boundaryupper = torch.mean((Vupper - bcupper) ** 2)

        loss_boundary = loss_boundary_lower + loss_boundaryupper

        # Total loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_obstacle * loss_obstacle +
            self.w_complementarity * loss_complementarity +
            self.w_terminal * loss_terminal +
            self.w_boundary * loss_boundary
        )

        return {
            'total': total_loss,
            'pde': loss_pde,
            'obstacle': loss_obstacle,
            'complementarity': loss_complementarity,
            'terminal': loss_terminal,
            'boundary': loss_boundary
        }


class AsianLoss(nn.Module):
    """
    Loss function for Asian Option PINNs.

    Solves 3D PDE:
        dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS + S dV/dI - rV

    where I is the running integral of stock price.

    Args:
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        strike: Strike price K
        T: Time to maturity
        option_type: 'call' or 'put'
        averaging_type: 'arithmetic' or 'geometric'
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        strike: float,
        T: float,
        option_type: str = 'call',
        averaging_type: str = 'arithmetic'
    ):
        super().__init__()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.strike = strike
        self.T = T
        self.option_type = option_type.lower()
        self.averaging_type = averaging_type.lower()

        # Loss weights
        self.w_pde = 1.0
        self.w_terminal = 10.0
        self.w_boundary = 10.0

    def forward(
        self,
        model: nn.Module,
        S_pde: torch.Tensor,
        tau_pde: torch.Tensor,
        I_pde: torch.Tensor,
        S_terminal: torch.Tensor,
        I_terminal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for Asian options.

        Args:
            model: PINN model
            S_pde: Stock prices for PDE residual
            tau_pde: Time to maturity for PDE residual
            I_pde: Running integral for PDE residual
            S_terminal: Stock prices for terminal condition
            I_terminal: Running integral for terminal condition

        Returns:
            Dictionary of loss components
        """
        # Enable gradients
        S_pde.requires_grad_(True)
        tau_pde.requires_grad_(True)
        I_pde.requires_grad_(True)

        # 1. PDE residual loss
        V_pde = model(S_pde, tau_pde, I_pde)

        # Compute derivatives
        V_S = torch.autograd.grad(
            V_pde, S_pde,
            grad_outputs=torch.ones_like(V_pde),
            create_graph=True,
            retain_graph=True
        )[0]

        V_tau = torch.autograd.grad(
            V_pde, tau_pde,
            grad_outputs=torch.ones_like(V_pde),
            create_graph=True,
            retain_graph=True
        )[0]

        V_I = torch.autograd.grad(
            V_pde, I_pde,
            grad_outputs=torch.ones_like(V_pde),
            create_graph=True,
            retain_graph=True
        )[0]

        V_SS = torch.autograd.grad(
            V_S, S_pde,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
            retain_graph=True
        )[0]

        # Asian PDE residual
        pde_residual = (V_tau -
                       0.5 * self.sigma**2 * S_pde**2 * V_SS -
                       (self.r - self.q) * S_pde * V_S -
                       S_pde * V_I +  # Advection term
                       self.r * V_pde)

        loss_pde = torch.mean(pde_residual ** 2)

        # 2. Terminal condition loss
        V_terminal = model(S_terminal, torch.zeros_like(S_terminal), I_terminal)

        # Payoff based on average A = I / T
        A_terminal = I_terminal / self.T
        if self.option_type == 'call':
            payoff = torch.clamp(A_terminal - self.strike, min=0.0)
        else:
            payoff = torch.clamp(self.strike - A_terminal, min=0.0)

        loss_terminal = torch.mean((V_terminal - payoff.unsqueeze(1)) ** 2)

        # 3. Boundary conditions
        # At S = 0: V(0, tau, I) = e^(-rtau) * payoff(I/T)
        n_bc = 100
        tau_bc = torch.linspace(0.01, self.T, n_bc).to(S_pde.device).unsqueeze(1)
        I_bc = torch.linspace(0.01, 300.0, n_bc).to(S_pde.device).unsqueeze(1)
        S_bc = torch.zeros_like(tau_bc) + 1e-6

        V_bc = model(S_bc, tau_bc, I_bc)
        A_bc = I_bc / self.T
        if self.option_type == 'call':
            payoff_bc = torch.clamp(A_bc - self.strike, min=0.0)
        else:
            payoff_bc = torch.clamp(self.strike - A_bc, min=0.0)

        bc_value = torch.exp(-self.r * tau_bc) * payoff_bc
        loss_boundary = torch.mean((V_bc - bc_value) ** 2)

        # Total loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_terminal * loss_terminal +
            self.w_boundary * loss_boundary
        )

        return {
            'total': total_loss,
            'pde': loss_pde,
            'terminal': loss_terminal,
            'boundary': loss_boundary
        }


class LookbackLoss(nn.Module):
    """
    Loss function for Lookback Option PINNs.

    Solves 3D PDE on domain 0 < S <= M:
        dV/dtau = 0.5sigma2S2 d2V/dS2 + (r-q)S dV/dS - rV

    with boundary condition at S = M: dV/dS(M, M, tau) = 0

    Args:
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        strike: Strike price K
        T: Time to maturity
        option_type: 'call' or 'put'
        lookback_type: 'fixed' or 'floating'
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        strike: float,
        T: float,
        option_type: str = 'call',
        lookback_type: str = 'floating'
    ):
        super().__init__()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.strike = strike
        self.T = T
        self.option_type = option_type.lower()
        self.lookback_type = lookback_type.lower()

        # Loss weights
        self.w_pde = 1.0
        self.w_terminal = 10.0
        self.w_diagonal = 50.0  # Strong enforcement of diagonal BC
        self.w_boundary = 10.0

    def forward(
        self,
        model: nn.Module,
        S_pde: torch.Tensor,
        tau_pde: torch.Tensor,
        M_pde: torch.Tensor,
        S_terminal: torch.Tensor,
        M_terminal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for Lookback options.

        Args:
            model: PINN model
            S_pde: Stock prices for PDE residual
            tau_pde: Time to maturity for PDE residual
            M_pde: Running maximum for PDE residual
            S_terminal: Stock prices for terminal condition
            M_terminal: Running maximum for terminal condition

        Returns:
            Dictionary of loss components
        """
        # Enable gradients
        S_pde.requires_grad_(True)
        tau_pde.requires_grad_(True)
        M_pde.requires_grad_(True)

        # 1. PDE residual loss (interior: S < M)
        V_pde = model(S_pde, tau_pde, M_pde)

        V_S = torch.autograd.grad(
            V_pde, S_pde,
            grad_outputs=torch.ones_like(V_pde),
            create_graph=True,
            retain_graph=True
        )[0]

        V_tau = torch.autograd.grad(
            V_pde, tau_pde,
            grad_outputs=torch.ones_like(V_pde),
            create_graph=True,
            retain_graph=True
        )[0]

        V_SS = torch.autograd.grad(
            V_S, S_pde,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
            retain_graph=True
        )[0]

        # PDE residual
        pde_residual = (V_tau -
                       0.5 * self.sigma**2 * S_pde**2 * V_SS -
                       (self.r - self.q) * S_pde * V_S +
                       self.r * V_pde)

        loss_pde = torch.mean(pde_residual ** 2)

        # 2. Terminal condition loss
        V_terminal = model(S_terminal, torch.zeros_like(S_terminal), M_terminal)

        if self.lookback_type == 'floating':
            if self.option_type == 'call':
                payoff = M_terminal  # Simplified
            else:
                payoff = M_terminal - S_terminal
        else:  # fixed
            if self.option_type == 'call':
                payoff = torch.clamp(M_terminal - self.strike, min=0.0)
            else:
                payoff = torch.clamp(self.strike - M_terminal, min=0.0)

        loss_terminal = torch.mean((V_terminal - payoff.unsqueeze(1)) ** 2)

        # 3. Diagonal boundary condition: dV/dS(S=M, M, tau) = 0
        # Sample points on diagonal
        n_diag = 100
        tau_diag = torch.linspace(0.01, self.T, n_diag).to(S_pde.device).unsqueeze(1)
        M_diag = torch.linspace(50.0, 200.0, n_diag).to(S_pde.device).unsqueeze(1)
        S_diag = M_diag.clone()  # S = M on diagonal

        S_diag.requires_grad_(True)
        V_diag = model(S_diag, tau_diag, M_diag)

        V_S_diag = torch.autograd.grad(
            V_diag, S_diag,
            grad_outputs=torch.ones_like(V_diag),
            create_graph=True,
            retain_graph=True
        )[0]

        # Enforce dV/dS = 0 on diagonal (smooth-fit condition)
        loss_diagonal = torch.mean(V_S_diag ** 2)

        # 4. Lower boundary (S = 0)
        S_lower = torch.ones_like(tau_diag) * 1e-6
        V_lower = model(S_lower, tau_diag, M_diag)

        # At S=0, option value depends on M only
        if self.lookback_type == 'fixed':
            if self.option_type == 'call':
                bc_lower = torch.exp(-self.r * tau_diag) * torch.clamp(M_diag - self.strike, min=0.0)
            else:
                bc_lower = torch.exp(-self.r * tau_diag) * torch.clamp(self.strike - M_diag, min=0.0)
        else:
            bc_lower = torch.exp(-self.r * tau_diag) * M_diag

        loss_boundary = torch.mean((V_lower - bc_lower) ** 2)

        # Total loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_terminal * loss_terminal +
            self.w_diagonal * loss_diagonal +
            self.w_boundary * loss_boundary
        )

        return {
            'total': total_loss,
            'pde': loss_pde,
            'terminal': loss_terminal,
            'diagonal': loss_diagonal,
            'boundary': loss_boundary
        }


def create_exotic_loss(
    option_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create exotic option loss functions.

    Args:
        option_type: Type of exotic option ('barrier', 'american', 'asian', 'lookback')
        **kwargs: Loss-specific arguments

    Returns:
        Loss function module

    Example:
        >>> # Create barrier loss
        >>> loss_fn = create_exotic_loss('barrier', r=0.05, q=0.02, sigma=0.2,
        ...                              strike=100.0, barrier=120.0, T=1.0)
        >>>
        >>> # Create American loss
        >>> loss_fn = create_exotic_loss('american', r=0.05, q=0.02, sigma=0.2,
        ...                              strike=100.0, T=1.0, option_type='put')
    """
    loss_functions = {
        'barrier': BarrierLoss,
        'american': AmericanLoss,
        'asian': AsianLoss,
        'lookback': LookbackLoss
    }

    option_type = option_type.lower()
    if option_type not in loss_functions:
        raise ValueError(
            f"Unknown exotic option type '{option_type}'. "
            f"Choose from: {list(loss_functions.keys())}"
        )

    return loss_functions[option_type](**kwargs)
