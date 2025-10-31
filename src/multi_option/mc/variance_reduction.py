"""Variance reduction techniques for Monte Carlo simulations.

This module implements various variance reduction techniques to improve
the efficiency of Monte Carlo option pricing.
"""

import numpy as np
from typing import Tuple, Optional


def apply_control_variate(
    payoffs: np.ndarray,
    control_values: np.ndarray,
    control_mean: float,
    discount_factor: float = 1.0
) -> Tuple[float, float]:
    """Apply control variate technique to reduce variance.

    Args:
        payoffs: Original payoff values.
        control_values: Control variate values.
        control_mean: Known expectation of control variate.
        discount_factor: Discount factor for present value.

    Returns:
        Tuple of (adjusted_price, standard_error).
    """
    # Calculate optimal beta coefficient
    cov_matrix = np.cov(payoffs, control_values)
    if cov_matrix.shape == (2, 2):
        cov_pc = cov_matrix[0, 1]
        var_c = cov_matrix[1, 1]
    else:
        # Fallback if covariance calculation fails
        return discount_factor * np.mean(payoffs), np.std(payoffs) / np.sqrt(len(payoffs))

    if var_c > 0:
        beta = cov_pc / var_c
    else:
        beta = 0

    # Apply control variate adjustment
    adjusted_payoffs = payoffs - beta * (control_values - control_mean)

    # Calculate statistics
    price = discount_factor * np.mean(adjusted_payoffs)
    stderr = discount_factor * np.std(adjusted_payoffs) / np.sqrt(len(adjusted_payoffs))

    return float(price), float(stderr)


def apply_importance_sampling(
    payoffs: np.ndarray,
    likelihood_ratios: np.ndarray,
    discount_factor: float = 1.0
) -> Tuple[float, float]:
    """Apply importance sampling to improve rare event estimation.

    Args:
        payoffs: Payoff values under importance measure.
        likelihood_ratios: Radon-Nikodym derivatives.
        discount_factor: Discount factor for present value.

    Returns:
        Tuple of (adjusted_price, standard_error).
    """
    # Weight payoffs by likelihood ratios
    weighted_payoffs = payoffs * likelihood_ratios

    # Calculate statistics
    price = discount_factor * np.mean(weighted_payoffs)
    stderr = discount_factor * np.std(weighted_payoffs) / np.sqrt(len(weighted_payoffs))

    return float(price), float(stderr)


def stratified_sampling(
    n_samples: int,
    n_strata: int,
    seed: int = 42
) -> np.ndarray:
    """Generate stratified uniform samples.

    Args:
        n_samples: Total number of samples.
        n_strata: Number of strata.
        seed: Random seed.

    Returns:
        Array of stratified samples in [0, 1].
    """
    np.random.seed(seed)

    samples_per_stratum = n_samples // n_strata
    extra_samples = n_samples % n_strata

    samples = []

    for i in range(n_strata):
        # Number of samples in this stratum
        n_in_stratum = samples_per_stratum + (1 if i < extra_samples else 0)

        # Stratum boundaries
        lower = i / n_strata
        upper = (i + 1) / n_strata

        # Generate uniform samples in stratum
        stratum_samples = np.random.uniform(lower, upper, n_in_stratum)
        samples.extend(stratum_samples)

    return np.array(samples)


def moment_matching(
    samples: np.ndarray,
    target_mean: float = 0.0,
    target_std: float = 1.0
) -> np.ndarray:
    """Adjust samples to match target moments.

    Args:
        samples: Original samples.
        target_mean: Target mean.
        target_std: Target standard deviation.

    Returns:
        Adjusted samples with matched moments.
    """
    # Current moments
    current_mean = np.mean(samples)
    current_std = np.std(samples)

    if current_std == 0:
        return samples

    # Adjust to match target moments
    adjusted = (samples - current_mean) * (target_std / current_std) + target_mean

    return adjusted


def conditional_expectation(
    payoffs: np.ndarray,
    conditioning_values: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """Estimate price using conditional expectation technique.

    Args:
        payoffs: Option payoffs.
        conditioning_values: Values to condition on.
        n_bins: Number of bins for conditioning.

    Returns:
        Tuple of (price, standard_error).
    """
    # Create bins based on conditioning values
    bins = np.percentile(conditioning_values, np.linspace(0, 100, n_bins + 1))

    # Calculate conditional expectations
    conditional_means = []
    weights = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (conditioning_values >= bins[i]) & (conditioning_values < bins[i + 1])
        else:
            mask = conditioning_values >= bins[i]

        if np.sum(mask) > 0:
            conditional_means.append(np.mean(payoffs[mask]))
            weights.append(np.sum(mask))
        else:
            conditional_means.append(0)
            weights.append(0)

    # Weighted average
    weights = np.array(weights) / np.sum(weights)
    price = np.sum(np.array(conditional_means) * weights)

    # Estimate standard error (simplified)
    stderr = np.std(payoffs) / np.sqrt(len(payoffs))

    return float(price), float(stderr)


class VarianceReducer:
    """Combines multiple variance reduction techniques."""

    def __init__(
        self,
        use_antithetic: bool = True,
        use_control_variate: bool = True,
        use_moment_matching: bool = False
    ):
        """Initialize variance reducer.

        Args:
            use_antithetic: Whether to use antithetic variates.
            use_control_variate: Whether to use control variates.
            use_moment_matching: Whether to use moment matching.
        """
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.use_moment_matching = use_moment_matching

    def generate_standard_normals(
        self,
        shape: Tuple[int, ...],
        seed: int
    ) -> np.ndarray:
        """Generate standard normal random variables with variance reduction.

        Args:
            shape: Shape of output array.
            seed: Random seed.

        Returns:
            Array of standard normal samples.
        """
        np.random.seed(seed)

        if len(shape) == 2 and self.use_antithetic:
            # Generate half and create antithetic pairs
            n_paths, n_steps = shape
            if n_paths % 2 != 0:
                n_paths += 1

            half_paths = n_paths // 2
            Z_half = np.random.standard_normal((half_paths, n_steps))
            Z = np.vstack([Z_half, -Z_half])[:shape[0], :]
        else:
            Z = np.random.standard_normal(shape)

        if self.use_moment_matching:
            # Adjust each column to have exact mean 0 and std 1
            if len(shape) == 2:
                for j in range(shape[1]):
                    Z[:, j] = moment_matching(Z[:, j])
            else:
                Z = moment_matching(Z)

        return Z