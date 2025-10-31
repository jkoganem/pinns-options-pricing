"""Lightweight smoke tests for the exotic PINN modules."""

from __future__ import annotations

import torch

from multi_option.pinns import (
    AdaptiveLossWeights,
    AmericanLoss,
    AmericanPINN,
    BarrierLoss,
    BarrierPINN,
)


def test_barrier_pinn_forward_pass() -> None:
    model = BarrierPINN(
        barrier_level=120.0,
        rebate=0.0,
        bs_reference=12.0,
        fourier_mapping_size=16,
        hidden_dim=32,
        num_layers=2,
    )
    spot = torch.tensor([[100.0]], requires_grad=True)
    tau = torch.tensor([[1.0]], requires_grad=True)

    value = model(spot, tau)

    assert value.shape == (1, 1)


def test_barrier_loss_backward() -> None:
    model = BarrierPINN(
        barrier_level=120.0,
        rebate=0.0,
        bs_reference=12.0,
        fourier_mapping_size=16,
        hidden_dim=32,
        num_layers=2,
    )
    loss_fn = BarrierLoss(
        r=0.05,
        q=0.02,
        sigma=0.2,
        strike=100.0,
        barrier=120.0,
        T=1.0,
    )

    S_pde = torch.rand(16, 1)
    tau_pde = torch.rand(16, 1)
    S_terminal = torch.rand(8, 1)
    S_barrier = torch.ones(4, 1) * 120.0
    tau_barrier = torch.rand(4, 1)

    losses = loss_fn(model, S_pde, tau_pde, S_terminal, S_barrier, tau_barrier)
    losses["total"].backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "Expected at least one gradient to be populated"


def test_american_pinn_forward_pass() -> None:
    model = AmericanPINN(
        strike=100.0,
        option_type="put",
        penalty_weight=100.0,
        bs_reference=12.0,
        fourier_mapping_size=16,
        hidden_dim=32,
        num_layers=2,
    )
    spot = torch.tensor([[90.0]], requires_grad=True)
    tau = torch.tensor([[0.5]], requires_grad=True)

    value = model(spot, tau)
    assert value.shape == (1, 1)


def test_american_loss_backward() -> None:
    model = AmericanPINN(
        strike=100.0,
        option_type="put",
        penalty_weight=100.0,
        bs_reference=12.0,
        fourier_mapping_size=16,
        hidden_dim=32,
        num_layers=2,
    )
    loss_fn = AmericanLoss(
        r=0.05,
        q=0.02,
        sigma=0.2,
        strike=100.0,
        T=1.0,
        option_type="put",
    )

    S_pde = torch.rand(16, 1)
    tau_pde = torch.rand(16, 1)
    S_terminal = torch.rand(8, 1)
    tau_boundary = torch.rand(4, 1)

    losses = loss_fn(model, S_pde, tau_pde, S_terminal, tau_boundary)
    losses["total"].backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "Expected at least one gradient to be populated"


def test_adaptive_weights_update() -> None:
    tracker = AdaptiveLossWeights(
        initial_weights={"pde": 1.0, "terminal": 2.0},
        warmup_epochs=5,
        update_frequency=1,
    )

    for epoch in range(6):
        tracker.update({"pde": 0.1 + epoch * 0.01, "terminal": 0.2 + epoch * 0.01}, epoch)
    weights = tracker.get_weights()
    assert set(weights) == {"pde", "terminal"}
