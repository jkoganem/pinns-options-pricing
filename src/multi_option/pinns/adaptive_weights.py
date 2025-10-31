"""
Adaptive loss weight balancing for Physics-Informed Neural Networks.

This module implements research-based adaptive weight balancing techniques
for handling multi-scale problems in PINNs, where different loss components
(PDE, terminal condition, boundary conditions) have vastly different magnitudes.

Based on:
- wbPINN (2024): Weight balanced physics-informed neural networks for multi-objective learning
- IAW-PINN (2024): Improved adaptive weighting method
- Sun et al. (2024): A practical PINN framework for multi-scale problems

Author: Multi-Option Pricing Team
Date: October 2025
"""

from typing import Dict, List, Optional
import numpy as np


class AdaptiveLossWeights:
    """
    Adaptive loss weight balancing for multi-scale PINN problems.

    Dynamically adjusts weights for PDE, terminal condition, and boundary
    condition losses during training to ensure balanced optimization across
    components with different magnitude scales.

    Key Innovation:
        Instead of increasing network capacity (which can cause overfitting),
        this approach rebalances the loss function components adaptively based
        on their relative magnitudes during training.

    Algorithm:
        1. Track loss magnitudes over recent history (500 epochs)
        2. Every 500 epochs after epoch 1000, recompute optimal weights
        3. Weight update: w_i proportional to 1 / loss_mag_i (inverse relationship)
        4. Apply exponential moving average (alpha=0.1) for smooth transitions
        5. Enforce maximum weight bound to prevent instability

    Args:
        initial_weights: Starting weights for each loss component
        adaptation_rate: Alpha for exponential moving average (0-1)
        max_weight: Maximum allowed weight to prevent instability
        update_frequency: How often to update weights (in epochs)
        warmup_epochs: Number of epochs before starting adaptation

    Example:
        >>> weights = AdaptiveLossWeights(
        ...     initial_weights={'pde': 1.0, 'terminal': 10.0, 'boundary': 5.0}
        ... )
        >>> for epoch in range(max_epochs):
        ...     losses = {'pde': pde_loss, 'terminal': term_loss, 'boundary': bc_loss}
        ...     weights.update(losses, epoch)
        ...     total_loss = sum(weights.weights[k] * losses[k] for k in losses)
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.1,
        max_weight: float = 100.0,
        update_frequency: int = 500,
        warmup_epochs: int = 1000
    ) -> None:
        """Initialize adaptive loss weights."""
        if initial_weights is None:
            initial_weights = {
                'pde': 1.0,
                'terminal': 10.0,
                'boundary': 5.0
            }

        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.max_weight = max_weight
        self.update_frequency = update_frequency
        self.warmup_epochs = warmup_epochs

        # Track loss history for each component
        self.loss_history: Dict[str, List[float]] = {
            key: [] for key in self.weights.keys()
        }

    def update(self, loss_dict: Dict[str, float], epoch: int) -> None:
        """
        Update loss weights based on current loss magnitudes.

        Args:
            loss_dict: Dictionary of loss component values
            epoch: Current epoch number
        """
        # Record losses
        for key, loss_value in loss_dict.items():
            if key in self.loss_history:
                self.loss_history[key].append(float(loss_value))

        # Check if we should update weights
        shouldupdate = (
            epoch > self.warmup_epochs and
            epoch % self.update_frequency == 0 and
            all(len(hist) >= self.update_frequency for hist in self.loss_history.values())
        )

        if not shouldupdate:
            return

        # Compute mean loss magnitudes over recent history
        loss_magnitudes = {}
        for key in self.weights.keys():
            recent_losses = self.loss_history[key][-self.update_frequency:]
            loss_magnitudes[key] = np.mean(recent_losses)

        # Compute total magnitude
        total_magnitude = sum(loss_magnitudes.values())

        if total_magnitude == 0:
            return

        # Update weights: higher magnitude -> lower weight (inverse relationship)
        # This encourages the optimizer to focus on components that are lagging
        for key in self.weights.keys():
            if loss_magnitudes[key] == 0:
                continue

            # Target weight is inversely proportional to loss magnitude
            target_weight = total_magnitude / loss_magnitudes[key]

            # Apply exponential moving average for smooth transitions
            new_weight = (
                self.weights[key] * (1 - self.adaptation_rate) +
                target_weight * self.adaptation_rate
            )

            # Enforce maximum weight bound
            self.weights[key] = min(self.max_weight, new_weight)

    def get_weights(self) -> Dict[str, float]:
        """Return current weights."""
        return self.weights.copy()

    def get_weighted_loss(self, loss_dict: Dict[str, float]) -> float:
        """
        Compute total weighted loss.

        Args:
            loss_dict: Dictionary of loss component values

        Returns:
            Weighted sum of losses
        """
        return sum(
            self.weights.get(key, 1.0) * value
            for key, value in loss_dict.items()
        )

    def state_dict(self) -> Dict:
        """Return state for checkpointing."""
        return {
            'weights': self.weights.copy(),
            'loss_history': {k: list(v) for k, v in self.loss_history.items()},
            'adaptation_rate': self.adaptation_rate,
            'max_weight': self.max_weight,
            'update_frequency': self.update_frequency,
            'warmup_epochs': self.warmup_epochs
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self.weights = state['weights'].copy()
        self.loss_history = {k: list(v) for k, v in state['loss_history'].items()}
        self.adaptation_rate = state['adaptation_rate']
        self.max_weight = state['max_weight']
        self.update_frequency = state['update_frequency']
        self.warmup_epochs = state['warmup_epochs']

    def __repr__(self) -> str:
        """String representation showing current weights."""
        weight_str = ", ".join(
            f"{k}={v:.2f}" for k, v in self.weights.items()
        )
        return f"AdaptiveLossWeights({weight_str})"


def compute_loss_statistics(
    loss_history: Dict[str, List[float]],
    window: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each loss component.

    Args:
        loss_history: Dictionary mapping loss names to value lists
        window: Window size for statistics computation

    Returns:
        Dictionary containing mean, std, min, max for each component
    """
    stats = {}

    for key, values in loss_history.items():
        if len(values) == 0:
            continue

        recent_values = values[-window:] if len(values) > window else values
        stats[key] = {
            'mean': float(np.mean(recent_values)),
            'std': float(np.std(recent_values)),
            'min': float(np.min(recent_values)),
            'max': float(np.max(recent_values)),
            'current': float(values[-1]) if values else 0.0
        }

    return stats


def print_loss_summary(
    epoch: int,
    loss_dict: Dict[str, float],
    weights: Dict[str, float],
    total_loss: float
) -> str:
    """
    Create formatted summary of losses and weights.

    Args:
        epoch: Current epoch
        loss_dict: Loss component values
        weights: Current loss weights
        total_loss: Total weighted loss

    Returns:
        Formatted string for logging
    """
    loss_str = " | ".join(
        f"{k}: {v:.6f}" for k, v in loss_dict.items()
    )
    weight_str = " | ".join(
        f"w_{k}: {v:.2f}" for k, v in weights.items()
    )

    return (
        f"Epoch {epoch:5d} | Total: {total_loss:.6f} | "
        f"Losses: {loss_str} | Weights: {weight_str}"
    )
