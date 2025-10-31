"""Multi-Method Option Pricing Engine.

A comprehensive option pricing library implementing multiple numerical methods.
"""

__version__ = "0.1.0"

from multi_option.datatypes import (
    EngineConfig,
    PriceResult,
    GreeksResult,
    Method,
    Product,
)

__all__ = [
    "EngineConfig",
    "PriceResult",
    "GreeksResult",
    "Method",
    "Product",
]