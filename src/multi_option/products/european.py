"""European option pricing across all methods.

This module provides a unified interface for pricing European options
using all available methods: Black-Scholes, PDE, Monte Carlo, and PINNs.
"""

from typing import Dict, Optional
import os
from pathlib import Path

from multi_option.datatypes import EngineConfig, PriceResult
from multi_option.bs_closed_form import bs_price
from multi_option.fd_pde.cn_solver import price_european_cn
from multi_option.mc.pricing import price_european_mc
from multi_option.pinns.train import train_pinn
from multi_option.pinns.infer import price_european_pinn


def price_european_all_methods(
    cfg: EngineConfig,
    call: bool
) -> Dict[str, PriceResult]:
    """Price European option using all available methods.

    Args:
        cfg: Engine configuration.
        call: True for call, False for put.

    Returns:
        Dictionary mapping method names to PriceResult objects.
    """
    results = {}

    # Black-Scholes closed-form
    bs_price_val = bs_price(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T, call
    )
    results['bs'] = PriceResult(
        method='bs',
        product='european_call' if call else 'european_put',
        price=bs_price_val,
        stderr=0.0,
        meta={'analytical': 1.0}
    )

    # Finite-difference PDE (Crank-Nicolson)
    pde_price_val = price_european_cn(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.s_max, cfg.ns, cfg.nt, call
    )
    results['pde'] = PriceResult(
        method='pde',
        product='european_call' if call else 'european_put',
        price=pde_price_val,
        stderr=0.0,
        meta={'ns': float(cfg.ns), 'nt': float(cfg.nt)}
    )

    # Monte Carlo
    mc_result = price_european_mc(
        cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
        cfg.mc_steps, cfg.mc_paths, cfg.seed, call
    )
    results['mc'] = mc_result

    # PINNs are NOT part of Stage 1 - only classical methods
    # PINNs are tested separately in Stage 2/3/4

    return results


def _price_european_pinn_wrapper(
    cfg: EngineConfig,
    call: bool
) -> PriceResult:
    """Wrapper for PINN pricing with training.

    Args:
        cfg: Engine configuration.
        call: True for call, False for put.

    Returns:
        PriceResult for PINN method.
    """
    # ALWAYS train/use CALL model, then apply put-call parity for puts
    model_dir = os.path.join(cfg.out_dir, 'pinn_models')
    model_path = os.path.join(model_dir, 'pinn_weights_call.pth')

    if not os.path.exists(model_path):
        # Train call model
        print(f"Training PINN call model with {cfg.pinn_epochs} epochs...")
        train_result = train_pinn(
            cfg.r, cfg.q, cfg.sigma, cfg.K, cfg.T, cfg.s_max,
            cfg.pinn_hidden, cfg.pinn_epochs, cfg.pinn_lr,
            cfg.seed, call=True, save_dir=model_dir
        )
        # Rename to call model
        old_path = train_result['model_path']
        if old_path and os.path.exists(old_path):
            os.rename(old_path, model_path)
        training_loss = train_result['final_loss']
    else:
        print(f"Using existing PINN call model from {model_path}")
        training_loss = 0.0

    # Price call using trained model
    call_price = price_european_pinn(cfg.s0, cfg.T, model_path, cfg.pinn_hidden)

    if call:
        # Return call price directly
        pinn_price = call_price
    else:
        # Apply put-call parity: Put = Call - S*e^(-qT) + K*e^(-rT)
        import numpy as np
        pinn_price = call_price - cfg.s0 * np.exp(-cfg.q * cfg.T) + cfg.K * np.exp(-cfg.r * cfg.T)

    return PriceResult(
        method='pinn',
        product='european_call' if call else 'european_put',
        price=pinn_price,
        stderr=0.0,
        meta={
            'epochs': float(cfg.pinn_epochs),
            'hidden': float(cfg.pinn_hidden),
            'final_loss': training_loss,
            'put_call_parity': not call  # Flag to indicate put was derived from call
        }
    )


def price_european_single_method(
    cfg: EngineConfig,
    method: str,
    call: bool
) -> PriceResult:
    """Price European option using a single specified method.

    Args:
        cfg: Engine configuration.
        method: Pricing method ('bs', 'pde', 'mc', or 'pinn').
        call: True for call, False for put.

    Returns:
        PriceResult for the specified method.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == 'bs':
        price = bs_price(cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T, call)
        return PriceResult(
            method='bs',
            product='european_call' if call else 'european_put',
            price=price,
            stderr=0.0,
            meta={'analytical': 1.0}
        )

    elif method == 'pde':
        price = price_european_cn(
            cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
            cfg.s_max, cfg.ns, cfg.nt, call
        )
        return PriceResult(
            method='pde',
            product='european_call' if call else 'european_put',
            price=price,
            stderr=0.0,
            meta={'ns': float(cfg.ns), 'nt': float(cfg.nt)}
        )

    elif method == 'mc':
        return price_european_mc(
            cfg.s0, cfg.K, cfg.r, cfg.q, cfg.sigma, cfg.T,
            cfg.mc_steps, cfg.mc_paths, cfg.seed, call
        )

    elif method == 'pinn':
        return _price_european_pinn_wrapper(cfg, call)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bs', 'pde', 'mc', or 'pinn'.")


def compute_european_implied_vol(
    market_price: float,
    s0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool,
    vol_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Compute implied volatility for European option.

    Uses Newton-Raphson method with Vega.

    Args:
        market_price: Market price of the option.
        s0: Current stock price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        T: Time to maturity.
        call: True for call, False for put.
        vol_guess: Initial volatility guess.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.

    Raises:
        ValueError: If convergence fails.
    """
    from multi_option.bs_closed_form import bs_price, bs_greeks

    sigma = vol_guess

    for i in range(max_iter):
        # Current price and vega
        price = bs_price(s0, K, r, q, sigma, T, call)
        greeks = bs_greeks(s0, K, r, q, sigma, T, call)
        vega = greeks.vega * 100  # Convert back to per-unit vega

        # Check convergence
        price_diff = market_price - price
        if abs(price_diff) < tol:
            return sigma

        # Newton-Raphson update
        if abs(vega) < 1e-10:
            # Use bisection fallback if vega is too small
            if price < market_price:
                sigma *= 1.1
            else:
                sigma *= 0.9
        else:
            sigma_new = sigma + price_diff / vega

            # Bounds checking
            sigma = max(0.001, min(5.0, sigma_new))

        # Check for invalid values
        if not (0 < sigma < 10):
            raise ValueError(f"Implied volatility out of bounds: {sigma}")

    raise ValueError(f"Implied volatility did not converge after {max_iter} iterations")