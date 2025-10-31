"""Configuration utilities for the option pricing engine.

This module provides utilities for parsing and validating configuration.
"""

from typing import Optional, Dict, Any
from multi_option.datatypes import EngineConfig


def parse_config(args: Dict[str, Any]) -> EngineConfig:
    """Parse command-line arguments into an EngineConfig.

    Args:
        args: Dictionary of command-line arguments.

    Returns:
        Validated EngineConfig instance.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # Validate required parameters
    required = ['s0', 'r', 'q', 'sigma', 'T', 'K', 's_max', 'ns', 'nt',
                'mc_paths', 'mc_steps', 'pinn_epochs', 'pinn_lr', 'pinn_hidden',
                'seed', 'out_dir']

    for param in required:
        if param not in args or args[param] is None:
            raise ValueError(f"Required parameter '{param}' is missing")

    # Validate numeric ranges
    if args['s0'] <= 0:
        raise ValueError("Initial stock price must be positive")
    if args['K'] <= 0:
        raise ValueError("Strike price must be positive")
    if args['T'] <= 0:
        raise ValueError("Time to maturity must be positive")
    if args['sigma'] <= 0:
        raise ValueError("Volatility must be positive")
    if args['ns'] < 10 or args['nt'] < 10:
        raise ValueError("Grid dimensions must be at least 10")
    if args['mc_paths'] < 100:
        raise ValueError("Number of Monte Carlo paths must be at least 100")

    # Handle optional barrier
    barrier = args.get('barrier')
    if barrier is not None and barrier <= 0:
        raise ValueError("Barrier level must be positive if specified")

    return EngineConfig(
        s0=float(args['s0']),
        r=float(args['r']),
        q=float(args['q']),
        sigma=float(args['sigma']),
        T=float(args['T']),
        K=float(args['K']),
        s_max=float(args['s_max']),
        ns=int(args['ns']),
        nt=int(args['nt']),
        mc_paths=int(args['mc_paths']),
        mc_steps=int(args['mc_steps']),
        barrier=float(barrier) if barrier is not None else None,
        ui_json=bool(args.get('ui_json', False)),
        plots=bool(args.get('plots', False)),
        out_dir=str(args['out_dir']),
        pinn_epochs=int(args['pinn_epochs']),
        pinn_lr=float(args['pinn_lr']),
        pinn_hidden=int(args['pinn_hidden']),
        seed=int(args['seed']),
        chain_csv=args.get('chain_csv'),
        use_mid=bool(args.get('use_mid', True))
    )


def default_config() -> EngineConfig:
    """Create a default configuration for testing.

    Returns:
        Default EngineConfig instance.
    """
    return EngineConfig(
        s0=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        K=100.0,
        s_max=400.0,
        ns=401,
        nt=400,
        mc_paths=200000,
        mc_steps=252,
        barrier=None,
        ui_json=True,
        plots=True,
        out_dir="./outputs",
        pinn_epochs=300,
        pinn_lr=1e-3,
        pinn_hidden=64,
        seed=42,
        chain_csv=None,
        use_mid=True
    )