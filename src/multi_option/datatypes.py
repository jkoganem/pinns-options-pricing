"""Core data types for the option pricing engine.

This module defines the fundamental data structures used throughout
the option pricing library.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict
import pandas as pd


Method = Literal["bs", "pde", "mc", "pinn"]
Product = Literal[
    "european_call",
    "european_put",
    "american_put",
    "barrierup_out_call",
    "asian_arith_call"
]


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the option pricing engine.

    Attributes:
        s0: Initial stock price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T: Time to maturity.
        K: Strike price.
        s_max: Maximum stock price for grid.
        ns: Number of spatial grid points.
        nt: Number of time grid points.
        mc_paths: Number of Monte Carlo paths.
        mc_steps: Number of time steps in Monte Carlo.
        barrier: Barrier level for barrier options.
        ui_json: Whether to generate UI JSON bundle.
        plots: Whether to generate plots.
        out_dir: Output directory path.
        pinn_epochs: Number of PINN training epochs.
        pinn_lr: PINN learning rate.
        pinn_hidden: Number of hidden units in PINN.
        seed: Random seed for reproducibility.
        chain_csv: Path to option chain CSV file.
        use_mid: Whether to use mid prices for IV.
    """
    s0: float
    r: float
    q: float
    sigma: float
    T: float
    K: float
    s_max: float
    ns: int
    nt: int
    mc_paths: int
    mc_steps: int
    barrier: Optional[float]  # for barrier product
    ui_json: bool
    plots: bool
    out_dir: str
    pinn_epochs: int
    pinn_lr: float
    pinn_hidden: int
    seed: int
    chain_csv: Optional[str]  # path to option chain for IV
    use_mid: bool


@dataclass(frozen=True)
class PriceResult:
    """Result from a pricing method.

    Attributes:
        method: The pricing method used.
        product: The option product type.
        price: The computed option price.
        stderr: Standard error (0.0 when not applicable).
        meta: Additional metadata (e.g., grid sizes, paths, epochs).
    """
    method: Method
    product: Product
    price: float
    stderr: float  # 0.0 when not applicable
    meta: Dict[str, float]  # e.g., grid sizes, paths, epochs


@dataclass(frozen=True)
class GreeksResult:
    """Option Greeks.

    Attributes:
        delta: Rate of change of option price with respect to stock price.
        gamma: Rate of change of delta with respect to stock price.
        theta: Rate of change of option price with respect to time.
        vega: Rate of change of option price with respect to volatility.
        rho: Rate of change of option price with respect to interest rate.
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float