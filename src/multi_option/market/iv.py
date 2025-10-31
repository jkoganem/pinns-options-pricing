"""Implied volatility computation.

This module implements methods for computing implied volatility
from option prices using various numerical methods.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Optional, Tuple
from multi_option.bs_closed_form import bs_price, bs_greeks


def bs_implied_vol(
    price: float,
    s0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool,
    method: str = 'brent',
    vol_bounds: Tuple[float, float] = (0.001, 5.0),
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Compute implied volatility from option price.

    Args:
        price: Market price of option.
        s0: Spot price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        T: Time to maturity.
        call: True for call, False for put.
        method: 'brent', 'newton', or 'bisection'.
        vol_bounds: Bounds for volatility search.
        tol: Tolerance for convergence.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.

    Raises:
        ValueError: If no valid implied volatility found.
    """
    # Check for arbitrage violations
    intrinsic = max(s0 - K, 0) if call else max(K - s0, 0)
    if price < intrinsic:
        raise ValueError(f"Option price {price} below intrinsic value {intrinsic}")

    # Check upper bound
    upper_bound = s0 if call else K * np.exp(-r * T)
    if price > upper_bound:
        raise ValueError(f"Option price {price} above upper bound {upper_bound}")

    if method == 'newton':
        return _implied_vol_newton(price, s0, K, r, q, T, call, tol, max_iter)
    elif method == 'brent':
        return _implied_vol_brent(price, s0, K, r, q, T, call, vol_bounds, tol)
    elif method == 'bisection':
        return _implied_vol_bisection(price, s0, K, r, q, T, call, vol_bounds, tol, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")


def _implied_vol_newton(
    price: float,
    s0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Newton-Raphson method for implied volatility.

    Args:
        price: Market price.
        s0: Spot price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        T: Time to maturity.
        call: Option type.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.
    """
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * price / s0

    for i in range(max_iter):
        # Current price and vega
        theo_price = bs_price(s0, K, r, q, sigma, T, call)
        greeks = bs_greeks(s0, K, r, q, sigma, T, call)
        vega = greeks.vega * 100  # Convert to per-unit

        # Check convergence
        price_error = price - theo_price
        if abs(price_error) < tol:
            return sigma

        # Newton update
        if abs(vega) < 1e-10:
            # Fall back to bisection if vega too small
            return _implied_vol_bisection(price, s0, K, r, q, T, call,
                                         (0.001, 5.0), tol, max_iter)

        sigma_new = sigma + price_error / vega

        # Apply bounds
        sigma = max(0.001, min(5.0, sigma_new))

    raise ValueError(f"Newton method did not converge after {max_iter} iterations")


def _implied_vol_brent(
    price: float,
    s0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool,
    vol_bounds: Tuple[float, float],
    tol: float = 1e-6
) -> float:
    """Brent's method for implied volatility.

    Args:
        price: Market price.
        s0: Spot price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        T: Time to maturity.
        call: Option type.
        vol_bounds: Search bounds.
        tol: Tolerance.

    Returns:
        Implied volatility.
    """
    def objective(sigma):
        return bs_price(s0, K, r, q, sigma, T, call) - price

    try:
        result = optimize.brentq(objective, vol_bounds[0], vol_bounds[1], xtol=tol)
        return float(result)
    except ValueError:
        # Check if price is outside bounds
        price_low = bs_price(s0, K, r, q, vol_bounds[0], T, call)
        price_high = bs_price(s0, K, r, q, vol_bounds[1], T, call)

        if price < price_low:
            return vol_bounds[0]
        elif price > price_high:
            return vol_bounds[1]
        else:
            raise ValueError("Brent's method failed to find implied volatility")


def _implied_vol_bisection(
    price: float,
    s0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool,
    vol_bounds: Tuple[float, float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Bisection method for implied volatility.

    Args:
        price: Market price.
        s0: Spot price.
        K: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        T: Time to maturity.
        call: Option type.
        vol_bounds: Search bounds.
        tol: Tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility.
    """
    vol_low, vol_high = vol_bounds

    for i in range(max_iter):
        vol_mid = (vol_low + vol_high) / 2
        price_mid = bs_price(s0, K, r, q, vol_mid, T, call)

        if abs(price_mid - price) < tol:
            return vol_mid

        if price_mid < price:
            vol_low = vol_mid
        else:
            vol_high = vol_mid

        if vol_high - vol_low < tol:
            return vol_mid

    return vol_mid


def implied_vol_from_chain(
    df: pd.DataFrame,
    use_mid: bool,
    r: float,
    q: float,
    s0: float
) -> pd.DataFrame:
    """Compute implied volatilities for entire option chain.

    Args:
        df: Option chain DataFrame.
        use_mid: Whether to use mid prices (True) or bid/ask (False).
        r: Risk-free rate.
        q: Dividend yield.
        s0: Spot price.

    Returns:
        DataFrame with added 'iv' column.
    """
    df = df.copy()
    df['iv'] = np.nan
    df['iv_bid'] = np.nan
    df['iv_ask'] = np.nan

    for idx, row in df.iterrows():
        try:
            is_call = row['cp_flag'] == 'C'
            T = row.get('T', 0.25)  # Default to 3 months if not specified

            if use_mid:
                # Use mid price
                iv = bs_implied_vol(
                    row['mid'], s0, row['K'], r, q, T, is_call
                )
                df.at[idx, 'iv'] = iv
            else:
                # Compute both bid and ask IVs
                try:
                    iv_bid = bs_implied_vol(
                        row['bid'], s0, row['K'], r, q, T, is_call
                    )
                    df.at[idx, 'iv_bid'] = iv_bid
                except:
                    pass

                try:
                    iv_ask = bs_implied_vol(
                        row['ask'], s0, row['K'], r, q, T, is_call
                    )
                    df.at[idx, 'iv_ask'] = iv_ask
                except:
                    pass

                # Set main IV as midpoint
                if not pd.isna(df.at[idx, 'iv_bid']) and not pd.isna(df.at[idx, 'iv_ask']):
                    df.at[idx, 'iv'] = (df.at[idx, 'iv_bid'] + df.at[idx, 'iv_ask']) / 2

        except Exception as e:
            # Skip options where IV cannot be computed
            continue

    # Remove rows where IV could not be computed
    df = df.dropna(subset=['iv'])

    return df


def compute_iv_smile(
    df: pd.DataFrame,
    T: float,
    option_type: str = 'both'
) -> pd.DataFrame:
    """Extract IV smile for a given maturity.

    Args:
        df: Option chain DataFrame with 'iv' column.
        T: Target maturity.
        option_type: 'call', 'put', or 'both'.

    Returns:
        DataFrame with strike and IV columns.
    """
    # Filter by maturity (with tolerance)
    T_tol = 0.01
    mask = np.abs(df['T'] - T) < T_tol

    if option_type == 'call':
        mask &= df['cp_flag'] == 'C'
    elif option_type == 'put':
        mask &= df['cp_flag'] == 'P'

    smile_df = df[mask][['K', 'iv', 'moneyness']].copy()
    smile_df = smile_df.sort_values('K')

    return smile_df


def fit_smile_svi(
    strikes: np.ndarray,
    ivs: np.ndarray,
    forward: float,
    T: float
) -> Dict[str, float]:
    """Fit SVI (Stochastic Volatility Inspired) model to IV smile.

    The SVI parameterization:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where w = total variance = iv^2 * T

    Args:
        strikes: Strike prices.
        ivs: Implied volatilities.
        forward: Forward price.
        T: Time to maturity.

    Returns:
        Dictionary with SVI parameters.
    """
    # Convert to log-moneyness
    k = np.log(strikes / forward)
    w = ivs ** 2 * T  # Total variance

    # Initial guess
    a0 = np.mean(w)
    b0 = 0.1
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.1

    def svi_func(k, a, b, rho, m, sigma):
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def objective(params):
        a, b, rho, m, sigma = params
        w_fit = svi_func(k, a, b, rho, m, sigma)
        return np.sum((w - w_fit) ** 2)

    # Constraints for no-arbitrage
    bounds = [
        (0, None),    # a >= 0
        (0, None),    # b >= 0
        (-1, 1),      # -1 <= rho <= 1
        (None, None), # m unconstrained
        (1e-6, None)  # sigma > 0
    ]

    result = optimize.minimize(
        objective,
        [a0, b0, rho0, m0, sigma0],
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        a, b, rho, m, sigma = result.x
        return {
            'a': a,
            'b': b,
            'rho': rho,
            'm': m,
            'sigma': sigma,
            'rmse': np.sqrt(result.fun / len(k))
        }
    else:
        raise ValueError("SVI fitting failed")


def extrapolate_iv(
    strikes: np.ndarray,
    ivs: np.ndarray,
    target_strikes: np.ndarray,
    method: str = 'flat'
) -> np.ndarray:
    """Extrapolate implied volatility to new strikes.

    Args:
        strikes: Original strikes.
        ivs: Original implied volatilities.
        target_strikes: Target strikes for extrapolation.
        method: 'flat', 'linear', or 'cubic'.

    Returns:
        Extrapolated implied volatilities.
    """
    from scipy import interpolate

    if method == 'flat':
        # Flat extrapolation
        target_ivs = np.zeros_like(target_strikes)
        for i, k in enumerate(target_strikes):
            if k <= strikes[0]:
                target_ivs[i] = ivs[0]
            elif k >= strikes[-1]:
                target_ivs[i] = ivs[-1]
            else:
                # Linear interpolation
                target_ivs[i] = np.interp(k, strikes, ivs)

    elif method == 'linear':
        # Linear extrapolation
        f = interpolate.interp1d(strikes, ivs, kind='linear',
                                fill_value='extrapolate')
        target_ivs = f(target_strikes)

    elif method == 'cubic':
        # Cubic spline with natural boundary conditions
        f = interpolate.CubicSpline(strikes, ivs, bc_type='natural')
        target_ivs = f(target_strikes)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure positive volatilities
    target_ivs = np.maximum(target_ivs, 0.001)

    return target_ivs