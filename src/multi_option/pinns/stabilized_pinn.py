#!/usr/bin/env python3
"""
Highly Stabilized PINN Implementation for Black-Scholes Options Pricing

This module adds advanced stabilization techniques to reduce training oscillations:
- Cosine Annealing with Warm Restarts for smooth LR decay
- Stochastic Weight Averaging (SWA) to reduce noise
- Loss smoothing for stable scheduler decisions
- Increased collocation points for better gradient estimates
- Curriculum learning on time domain

Based on optimized_pinn.py with additional stabilization layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
import time
from typing import Dict, Tuple, Optional


def black_scholes_analytical(S, K, r, q, sigma, T, is_call=True):
    """Black-Scholes analytical price."""
    if T <= 0:
        return np.maximum(S - K, 0) if is_call else np.maximum(K - S, 0)

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if is_call:
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)


class StabilizedPINN(nn.Module):
    """PINN with Fourier features - same architecture as OptimizedPINN."""

    def __init__(self, hidden_dim=128, num_layers=5, use_fourier=True,
                 fourier_features=64, fourier_scale=3.0):
        super().__init__()

        self.use_fourier = use_fourier
        self.fourier_features = fourier_features
        self.fourier_scale = fourier_scale

        if use_fourier:
            input_dim = 2 * fourier_features
            self.register_buffer('fourier_matrix',
                               torch.randn(2, fourier_features) * fourier_scale)
        else:
            input_dim = 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_fourier_features(self, inputs):
        """Compute Fourier features."""
        projected = torch.matmul(inputs, self.fourier_matrix)
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)
        return features

    def forward(self, S, tau):
        """Forward pass with optional Fourier features."""
        if S.dim() == 1:
            S = S.unsqueeze(1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)

        S_norm = S / 100.0
        tau_norm = tau

        if self.use_fourier:
            inputs = torch.cat([S_norm, tau_norm], dim=1)
            x = self.compute_fourier_features(inputs)
        else:
            x = torch.cat([S_norm, tau_norm], dim=1)

        return self.net(x)


def compute_pde_residual(model, S, tau, r, q, sigma):
    """CORRECT Black-Scholes PDE residual."""
    S.requires_grad_(True)
    tau.requires_grad_(True)

    V = model(S, tau)

    dV = torch.autograd.grad(V.sum(), [S, tau], create_graph=True)
    V_S = dV[0]
    V_tau = dV[1]

    V_SS = torch.autograd.grad(V_S.sum(), S, create_graph=True)[0]

    residual = V_tau - (0.5 * sigma**2 * S**2 * V_SS + (r - q) * S * V_S - r * V)
    return residual


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience=1000, min_delta=1e-6, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name] +
                                   (1 - self.decay) * param.data)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def train_stabilized_pinn(
    K=100.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    T=1.0,
    n_epochs=30000,
    lr=1e-3,
    n_interior=5000,  # INCREASED from 2000
    n_boundary=500,   # INCREASED from 200
    n_initial=2000,   # INCREASED from 1000
    hidden_dim=128,
    num_layers=5,
    use_fourier=True,
    fourier_features=64,
    fourier_scale=3.0,
    use_warmup=True,
    warmup_epochs=1000,
    use_ema=True,
    ema_decay=0.999,
    use_swa=True,  # NEW: Stochastic Weight Averaging
    swa_start=0.75,  # NEW: Start SWA at 75% of training
    use_cosine_annealing=True,  # NEW: Cosine annealing schedule
    use_curriculum=True,  # NEW: Curriculum learning on time domain
    use_early_stopping=False,
    early_stop_patience=2000,
    early_stop_min_delta=1e-6,
    device=None,
    verbose=True
) -> Tuple[StabilizedPINN, Dict]:
    """
    Train stabilized PINN with advanced techniques to reduce oscillations.

    NEW STABILIZATION FEATURES:
    1. Cosine Annealing with Warm Restarts - smooth LR decay
    2. Stochastic Weight Averaging (SWA) - reduces noise in final weights
    3. Increased collocation points (5000 interior) - better gradient estimates
    4. Loss smoothing - stable scheduler decisions
    5. Curriculum learning - progressive time domain expansion
    6. Increased gradient clipping threshold - less aggressive

    Args:
        use_swa: Enable Stochastic Weight Averaging (RECOMMENDED)
        swa_start: Fraction of training to start SWA (default 0.75)
        use_cosine_annealing: Use cosine annealing LR schedule (RECOMMENDED)
        use_curriculum: Use curriculum learning on time domain (RECOMMENDED)
        (other args same as optimized_pinn.py)

    Returns:
        Tuple of (trained_model, info_dict)
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = StabilizedPINN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_fourier=use_fourier,
        fourier_features=fourier_features,
        fourier_scale=fourier_scale
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # NEW: Cosine Annealing with Warm Restarts
    if use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,      # Restart every 5000 epochs
            T_mult=1,      # Keep same period
            eta_min=1e-7   # Minimum LR
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500
        )

    # NEW: Stochastic Weight Averaging
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(swa_start * n_epochs)
    if use_swa:
        from torch.optim.swautils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05 * lr)

    # EMA (works alongside SWA)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        mode='min'
    ) if use_early_stopping else None

    # Loss weights
    w_pde = 1.0
    w_ic = 100.0
    w_bc = 10.0

    start_time = time.time()
    losses = []
    errors = []
    stopped_early = False
    stopped_epoch = n_epochs

    # NEW: Loss smoothing
    loss_smooth = None
    alpha_smooth = 0.05  # Smoothing factor

    if verbose:
        print(f"Training Stabilized PINN on {device}")
        print(f"Parameters: K={K}, r={r}, q={q}, sigma={sigma}, T={T}")
        print(f"Network: {num_layers} layers x {hidden_dim} neurons")
        print(f"Collocation points: Interior={n_interior}, IC={n_initial}, BC={n_boundary}")
        print(f"Stabilization: SWA={use_swa}, Cosine={use_cosine_annealing}, Curriculum={use_curriculum}")
        print("-" * 60)

    for epoch in range(n_epochs):
        model.train()

        # Learning rate warmup
        if use_warmup and epoch < warmup_epochs:
            current_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # NEW: Smooth curriculum learning on time domain (prevents catastrophic spikes)
        if use_curriculum:
            # SMOOTH transition instead of discrete jumps
            # Gradually expand from [0.7T, T] -> [0, T] over entire training
            progress = epoch / n_epochs  # 0.0 -> 1.0
            tau_min = (1 - progress) * 0.7 * T  # 0.7T -> 0 smoothly
            tau_max = T
        else:
            tau_min, tau_max = 0.0, T

        # Sample collocation points with stratified sampling
        n_near = int(0.7 * n_interior)
        n_far = n_interior - n_near

        S_near = K * (0.7 + 0.6 * torch.rand(n_near, 1, device=device))
        S_far = K * (0.3 + 1.4 * torch.rand(n_far, 1, device=device))
        S_interior = torch.cat([S_near, S_far], dim=0)
        tau_interior = tau_min + (tau_max - tau_min) * torch.rand(n_interior, 1, device=device)

        # Initial condition points
        S_initial = K * (0.5 + 1.0 * torch.rand(n_initial, 1, device=device))
        tau_initial = torch.zeros(n_initial, 1, device=device)

        # Boundary conditions
        S_bc_low = 0.01 * K * torch.ones(n_boundary, 1, device=device)
        S_bc_high = 2.0 * K * torch.ones(n_boundary, 1, device=device)
        tau_bc = tau_min + (tau_max - tau_min) * torch.rand(n_boundary, 1, device=device)

        # Compute losses
        residual = compute_pde_residual(model, S_interior, tau_interior, r, q, sigma)
        loss_pde = torch.mean(residual**2)

        V_initial = model(S_initial, tau_initial)
        payoff = torch.maximum(S_initial - K, torch.zeros_like(S_initial))
        loss_ic = torch.mean((V_initial - payoff)**2)

        V_bc_low = model(S_bc_low, tau_bc)
        V_bc_high = model(S_bc_high, tau_bc)
        loss_bc_low = torch.mean(V_bc_low**2)
        bc_high_target = S_bc_high - K * torch.exp(-r * tau_bc)
        loss_bc_high = torch.mean((V_bc_high - bc_high_target)**2)
        loss_bc = loss_bc_low + loss_bc_high

        # Total loss
        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # NEW: Increased gradient clipping threshold (less aggressive)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Update EMA
        if use_ema:
            ema.update(model)

        # NEW: Update SWA or regular scheduler
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if use_cosine_annealing:
                scheduler.step()
            else:
                # NEW: Use smoothed loss for ReduceLROnPlateau
                if loss_smooth is None:
                    loss_smooth = loss.item()
                else:
                    loss_smooth = alpha_smooth * loss.item() + (1 - alpha_smooth) * loss_smooth
                scheduler.step(loss_smooth)

        losses.append(loss.item())

        # Validation every 500 epochs
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                S_test = torch.tensor([[K]], device=device, dtype=torch.float32)
                tau_test = torch.tensor([[T]], device=device, dtype=torch.float32)
                V_pred = model(S_test, tau_test).item()
                V_true = black_scholes_analytical(K, K, r, q, sigma, T, is_call=True)
                error = abs(V_pred - V_true) / V_true * 100
                errors.append(error)

                if verbose:
                    curr_lr = optimizer.param_groups[0]['lr']
                    stage = "SWA" if (use_swa and epoch >= swa_start_epoch) else "Train"
                    print(f"Epoch {epoch:5d} [{stage}] | Loss: {loss.item():.2e} | "
                          f"PDE: {loss_pde.item():.2e} | IC: {loss_ic.item():.2e} | "
                          f"V(K,T): {V_pred:.4f} | True: {V_true:.4f} | Error: {error:.2f}% | LR: {curr_lr:.2e}")

                # Check early stopping (on smoothed loss)
                if use_early_stopping:
                    check_loss = loss_smooth if loss_smooth is not None else loss.item()
                    if early_stopping(check_loss, epoch):
                        stopped_early = True
                        stopped_epoch = epoch
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                        break

    training_time = time.time() - start_time

    # Apply SWA if used
    if use_swa and swa_model is not None:
        if verbose:
            print("\nApplying Stochastic Weight Averaging...")
        # Update batch norm statistics (not needed for this model, but good practice)
        # For our model without BN, just use the averaged weights
        for name, param in model.named_parameters():
            if param.requires_grad and name in swa_model.module.state_dict():
                param.data.copy_(swa_model.module.state_dict()[name])

    # Apply EMA if used and SWA wasn't
    elif use_ema and ema is not None:
        if verbose:
            print("\nApplying EMA weights...")
        ema.apply_shadow(model)

    # Final validation
    model.eval()
    with torch.no_grad():
        S_test = torch.tensor([[K]], device=device, dtype=torch.float32)
        tau_test = torch.tensor([[T]], device=device, dtype=torch.float32)
        V_pred = model(S_test, tau_test).item()
        V_true = black_scholes_analytical(K, K, r, q, sigma, T, is_call=True)
        final_error = abs(V_pred - V_true) / V_true * 100

    if verbose:
        print("-" * 60)
        print(f"Training complete in {training_time:.1f}s")
        print(f"Final V(K,T): {V_pred:.4f}")
        print(f"True V(K,T): {V_true:.4f}")
        print(f"Final error: {final_error:.2f}%")

    info = {
        'training_time': training_time,
        'final_error': final_error,
        'losses': losses,
        'errors': errors,
        'stopped_early': stopped_early,
        'stopped_epoch': stopped_epoch,
        'used_swa': use_swa,
        'used_cosine_annealing': use_cosine_annealing,
        'used_curriculum': use_curriculum
    }

    return model, info


if __name__ == "__main__":
    print("="*80)
    print("TESTING STABILIZED PINN")
    print("="*80)

    # Quick test
    model, info = train_stabilized_pinn(
        n_epochs=5000,
        use_swa=True,
        use_cosine_annealing=True,
        use_curriculum=True,
        verbose=True
    )

    print(f"\n Test complete!")
    print(f"Final error: {info['final_error']:.3f}%")
    print(f"Training time: {info['training_time']:.1f}s")
