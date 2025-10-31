"""Loss functions for Physics-Informed Neural Networks.

This module implements the loss functions for training PINNs
to solve the Black-Scholes PDE.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


def pinn_loss(
    net: torch.nn.Module,
    s_batch: torch.Tensor,
    t_batch: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
    K: float,
    call: bool,
    s_max: float = 400.0,
    lambda_pde: float = 1.0,  # PDE residual weight
    lambda_ic: float = 100.0,  # CRITICAL: Much higher weight for initial condition!
    lambda_bc: float = 10.0   # Moderate weight for boundaries
) -> torch.Tensor:
    """Compute PINN loss for Black-Scholes PDE.

    The loss consists of:
    1. PDE residual loss (interior points)
    2. Initial condition loss (terminal condition at t=T)
    3. Boundary condition loss (at S=0 and S=S_max)

    Args:
        net: The neural network model.
        s_batch: Batch of stock prices.
        t_batch: Batch of time values (time to maturity).
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        K: Strike price.
        call: True for call, False for put.
        s_max: Maximum stock price for boundary.
        lambda_pde: Weight for PDE residual loss.
        lambda_ic: Weight for initial condition loss.
        lambda_bc: Weight for boundary condition loss.

    Returns:
        Total weighted loss.
    """
    # Enable gradients for automatic differentiation
    s_batch.requires_grad_(True)
    t_batch.requires_grad_(True)

    # Network prediction
    v = net(s_batch, t_batch)

    # Compute gradients (optimized but must retain until all computed)
    # First derivative: dV/dS (keep graph for v_ss computation)
    v_s = torch.autograd.grad(v, s_batch, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    # First derivative: dV/dt (keep graph to avoid freeing intermediate)
    v_t = torch.autograd.grad(v, t_batch, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    # Second derivative: d2V/dS2 (can release graph after this)
    v_ss = torch.autograd.grad(v_s, s_batch, grad_outputs=torch.ones_like(v_s),
                               create_graph=False, retain_graph=False)[0]

    # Black-Scholes PDE residual: dV/dtau + LV = 0
    # where L is the Black-Scholes operator
    # Note: we use tau = T - t (time to maturity), so dV/dt = -dV/dtau
    pde_residual = -v_t + 0.5 * sigma**2 * s_batch**2 * v_ss + (r - q) * s_batch * v_s - r * v

    # PDE loss (mean squared error of residual)
    loss_pde = torch.mean(pde_residual**2)

    # Initial condition loss (terminal condition at t=0, i.e., tau=T)
    # Sample points at t=0 (500 points for efficiency)
    n_ic = 500
    s_ic = torch.linspace(0.01, s_max, n_ic).to(s_batch.device)
    t_ic = torch.zeros(n_ic, dtype=s_ic.dtype).to(s_batch.device)

    v_ic = net(s_ic, t_ic).squeeze()

    if call:
        payoff_ic = torch.clamp(s_ic - K, min=0.0)
    else:
        payoff_ic = torch.clamp(K - s_ic, min=0.0)

    loss_ic = torch.mean((v_ic - payoff_ic)**2)

    # Boundary condition loss
    n_bc = 100
    t_bc = torch.linspace(0.01, 1.0, n_bc).to(s_batch.device)

    # Lower boundary (S=0)
    s_lower = torch.zeros_like(t_bc) + 1e-6
    v_lower = net(s_lower, t_bc)

    if call:
        bc_lower = torch.zeros_like(v_lower)
    else:
        bc_lower = K * torch.exp(-r * t_bc).reshape(-1, 1)

    loss_bc_lower = torch.mean((v_lower - bc_lower)**2)

    # Upper boundary (S=S_max)
    supper = torch.ones_like(t_bc) * s_max
    vupper = net(supper, t_bc)

    if call:
        bcupper = (s_max - K * torch.exp(-r * t_bc)).reshape(-1, 1)
    else:
        bcupper = torch.zeros_like(vupper)

    loss_bcupper = torch.mean((vupper - bcupper)**2)

    loss_bc = loss_bc_lower + loss_bcupper

    # Total loss
    total_loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

    return total_loss


def compute_pde_residual(
    net: torch.nn.Module,
    s: torch.Tensor,
    t: torch.Tensor,
    r: float,
    q: float,
    sigma: float
) -> torch.Tensor:
    """Compute Black-Scholes PDE residual.

    Args:
        net: The neural network model.
        s: Stock prices.
        t: Time values.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.

    Returns:
        PDE residual values.
    """
    s.requires_grad_(True)
    t.requires_grad_(True)

    v = net(s, t)

    # Compute derivatives
    v_s = torch.autograd.grad(v, s, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    v_ss = torch.autograd.grad(v_s, s, grad_outputs=torch.ones_like(v_s),
                              create_graph=True, retain_graph=True)[0]

    # PDE residual
    residual = -v_t + 0.5 * sigma**2 * s**2 * v_ss + (r - q) * s * v_s - r * v

    return residual


def adaptive_loss_weights(
    losses: Dict[str, torch.Tensor],
    alpha: float = 0.9
) -> Dict[str, float]:
    """Compute adaptive loss weights based on gradient magnitudes.

    Args:
        losses: Dictionary of individual loss components.
        alpha: Exponential moving average factor.

    Returns:
        Dictionary of adaptive weights.
    """
    weights = {}
    grad_norms = {}

    # Compute gradient norms for each loss
    for name, loss in losses.items():
        if loss.requires_grad:
            grad = torch.autograd.grad(loss, loss, retain_graph=True)[0]
            grad_norms[name] = torch.norm(grad).item()
        else:
            grad_norms[name] = 1.0

    # Normalize weights
    max_norm = max(grad_norms.values())
    if max_norm > 0:
        for name in losses:
            weights[name] = max_norm / (grad_norms[name] + 1e-8)
    else:
        for name in losses:
            weights[name] = 1.0

    return weights


class PINNLoss(nn.Module):
    """Modular PINN loss function class."""

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        K: float,
        T: float,
        call: bool,
        s_max: float = 400.0
    ):
        """Initialize PINN loss function.

        Args:
            r: Risk-free rate.
            q: Dividend yield.
            sigma: Volatility.
            K: Strike price.
            T: Time to maturity.
            call: True for call, False for put.
            s_max: Maximum stock price.
        """
        super(PINNLoss, self).__init__()
        self.r = r
        self.q = q
        self.sigma = sigma
        self.K = K
        self.T = T
        self.call = call
        self.s_max = s_max

        # Loss weights (can be updated during training)
        self.lambda_pde = 1.0   # PDE residual weight
        self.lambda_ic = 100.0  # CRITICAL: Much higher weight for initial condition!
        self.lambda_bc = 10.0   # Moderate weight for boundaries

    def forward(
        self,
        net: torch.nn.Module,
        s_batch: torch.Tensor,
        t_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            net: The neural network model.
            s_batch: Batch of stock prices.
            t_batch: Batch of time values.

        Returns:
            Dictionary with individual loss components and total loss.
        """
        # Enable gradients
        s_batch.requires_grad_(True)
        t_batch.requires_grad_(True)

        # Network prediction
        v = net(s_batch, t_batch)

        # PDE residual loss
        residual = compute_pde_residual(net, s_batch, t_batch,
                                       self.r, self.q, self.sigma)
        loss_pde = torch.mean(residual**2)

        # Initial condition loss
        loss_ic = self._initial_condition_loss(net)

        # Boundary condition loss
        loss_bc = self._boundary_condition_loss(net)

        # Total loss
        total_loss = (self.lambda_pde * loss_pde +
                     self.lambda_ic * loss_ic +
                     self.lambda_bc * loss_bc)

        return {
            'total': total_loss,
            'pde': loss_pde,
            'ic': loss_ic,
            'bc': loss_bc
        }

    def _initial_condition_loss(self, net: torch.nn.Module) -> torch.Tensor:
        """Compute initial/terminal condition loss."""
        n_points = 1000
        device = next(net.parameters()).device

        s_ic = torch.linspace(0.01, self.s_max, n_points).to(device)
        t_ic = torch.zeros(n_points).to(device)

        v_ic = net(s_ic, t_ic).squeeze()

        if self.call:
            payoff = torch.maximum(s_ic - self.K, torch.zeros_like(s_ic))
        else:
            payoff = torch.maximum(self.K - s_ic, torch.zeros_like(s_ic))

        return torch.mean((v_ic - payoff)**2)

    def _boundary_condition_loss(self, net: torch.nn.Module) -> torch.Tensor:
        """Compute boundary condition loss."""
        n_points = 100
        device = next(net.parameters()).device

        t_bc = torch.linspace(0.01, self.T, n_points).to(device)

        # Lower boundary
        s_lower = torch.ones(n_points).to(device) * 1e-6
        v_lower = net(s_lower, t_bc).squeeze()

        if self.call:
            bc_lower = torch.zeros_like(v_lower)
        else:
            bc_lower = self.K * torch.exp(-self.r * t_bc)

        loss_lower = torch.mean((v_lower - bc_lower)**2)

        # Upper boundary
        supper = torch.ones(n_points).to(device) * self.s_max
        vupper = net(supper, t_bc).squeeze()

        if self.call:
            bcupper = self.s_max - self.K * torch.exp(-self.r * t_bc)
        else:
            bcupper = torch.zeros_like(vupper)

        lossupper = torch.mean((vupper - bcupper)**2)

        return loss_lower + lossupper

    def update_weights(self, lambda_pde: float, lambda_ic: float, lambda_bc: float):
        """Update loss weights.

        Args:
            lambda_pde: Weight for PDE loss.
            lambda_ic: Weight for initial condition loss.
            lambda_bc: Weight for boundary condition loss.
        """
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc