"""
Physics-Informed Neural Networks for Option Pricing.

This package provides modular, PEP-compliant PINN implementations for
European and Exotic option pricing using the Black-Scholes PDE.

Modules:
    architectures: PINN network architectures (Base, Enhanced, Adaptive)
    exotic_architectures: Exotic option PINNs (Barrier, American, Asian, Lookback)
    exotic_loss: Loss functions for exotic options
    adaptive_weights: Adaptive loss weight balancing (research-based)
    utils: Utility functions (Fourier features, EMA, early stopping)
    training: Unified training logic
    loss: Loss function definitions
    model: Legacy model definitions
    infer: Inference utilities

Example:
    >>> from multi_option.pinns import create_pinn, AdaptiveLossWeights
    >>> from multi_option.pinns import create_exotic_pinn, create_exotic_loss
    >>>
    >>> # Create optimal adaptive weight PINN for European options
    >>> model = create_pinn('adaptive', bs_reference=22.54)
    >>>
    >>> # Create barrier option PINN
    >>> barrier_model = create_exotic_pinn('barrier', barrier_level=120.0, strike=100.0)
    >>>
    >>> # Create American option PINN
    >>> american_model = create_exotic_pinn('american', strike=100.0, option_type='put')
    >>>
    >>> # Create adaptive loss weights
    >>> weights = AdaptiveLossWeights(
    ...     initial_weights={'pde': 1.0, 'terminal': 10.0, 'boundary': 5.0}
    ... )

Author: Multi-Option Pricing Team
Date: October 2025
"""

from .architectures import (
    BasePINN,
    EnhancedCapacityPINN,
    AdaptiveWeightPINN,
    create_pinn,
    bs_call_price,
    bs_put_price
)

from .exotic_architectures import (
    BarrierPINN,
    AmericanPINN,
    AsianPINN,
    LookbackPINN,
    create_exotic_pinn
)

from .exotic_loss import (
    BarrierLoss,
    AmericanLoss,
    AsianLoss,
    LookbackLoss,
    create_exotic_loss
)

from .adaptive_weights import (
    AdaptiveLossWeights,
    compute_loss_statistics,
    print_loss_summary
)

from .utils import (
    FourierFeatureMapping,
    ResidualBlock,
    ExponentialMovingAverage,
    EarlyStopping,
    initialize_weights,
    count_parameters,
    normalize_inputs,
    compute_pde_residual
)

# Import the unified PINN module (RECOMMENDED for best performance)
try:
    from .unified_pinn import (
        UnifiedPINN,
        trainunified_pinn,
        train_simple_pinn as unified_train_simple,
        train_advanced_pinn as unified_train_advanced,
        compute_pde_residual as unified_compute_pde_residual,
        FourierFeatures as UnifiedFourierFeatures
    )
except ImportError:
    # Unified module not available
    UnifiedPINN = None
    trainunified_pinn = None
    unified_train_simple = None
    unified_train_advanced = None

__all__ = [
    # European Option Architectures
    'BasePINN',
    'EnhancedCapacityPINN',
    'AdaptiveWeightPINN',
    'create_pinn',
    'bs_call_price',
    'bs_put_price',

    # Exotic Option Architectures
    'BarrierPINN',
    'AmericanPINN',
    'AsianPINN',
    'LookbackPINN',
    'create_exotic_pinn',

    # Exotic Option Loss Functions
    'BarrierLoss',
    'AmericanLoss',
    'AsianLoss',
    'LookbackLoss',
    'create_exotic_loss',

    # Adaptive Weights
    'AdaptiveLossWeights',
    'compute_loss_statistics',
    'print_loss_summary',

    # Utilities
    'FourierFeatureMapping',
    'ResidualBlock',
    'ExponentialMovingAverage',
    'EarlyStopping',
    'initialize_weights',
    'count_parameters',
    'normalize_inputs',
    'compute_pde_residual',

    # Unified PINN (RECOMMENDED - Best Performance)
    'UnifiedPINN',
    'trainunified_pinn',
    'unified_train_simple',
    'unified_train_advanced',

    # Ultimate PINN with correct PDE and all features
    'UltimatePINN',
    'trainultimate_pinn',
    'TrulyFixedPINN',
    'train_truly_fixed_pinn',
]

# Import the working PINN modules
try:
    from .truly_fixed_pinn import TrulyFixedPINN, train_truly_fixed_pinn
    from .ultimate_pinn import UltimatePINN, trainultimate_pinn
except ImportError:
    TrulyFixedPINN = None
    train_truly_fixed_pinn = None
    UltimatePINN = None
    trainultimate_pinn = None

__version__ = '2.0.0'  # Major update with fixed PDE formulation
