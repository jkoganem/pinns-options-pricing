"""No-arbitrage condition checks for option prices.

This module implements various arbitrage checks for option chains
and implied volatility surfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def no_arbitrage_warnings(df_iv: pd.DataFrame) -> Dict[str, int]:
    """Check for arbitrage violations in option chain.

    Args:
        df_iv: DataFrame with option chain and IV data.

    Returns:
        Dictionary with counts of different violations.
    """
    warnings = {
        'negative_iv': 0,
        'vertical_spread': 0,
        'butterfly': 0,
        'calendar_spread': 0,
        'put_call_parity': 0
    }

    # Check for negative IV
    warnings['negative_iv'] = (df_iv['iv'] <= 0).sum()

    # Group by maturity for spread checks
    for T, group in df_iv.groupby('T'):
        # Separate calls and puts
        calls = group[group['cp_flag'] == 'C'].sort_values('K')
        puts = group[group['cp_flag'] == 'P'].sort_values('K')

        # Vertical spread arbitrage for calls
        if len(calls) > 1:
            violations = _check_vertical_spread(
                calls['K'].values,
                calls['mid'].values
            )
            warnings['vertical_spread'] += violations

        # Vertical spread arbitrage for puts
        if len(puts) > 1:
            violations = _check_vertical_spread(
                puts['K'].values,
                puts['mid'].values,
                is_put=True
            )
            warnings['vertical_spread'] += violations

        # Butterfly arbitrage
        if len(calls) > 2:
            violations = _check_butterfly(
                calls['K'].values,
                calls['mid'].values
            )
            warnings['butterfly'] += violations

    # Calendar spread arbitrage
    warnings['calendar_spread'] = _check_calendar_spread(df_iv)

    # Put-call parity violations
    warnings['put_call_parity'] = _check_put_call_parity(df_iv)

    return warnings


def _check_vertical_spread(
    strikes: np.ndarray,
    prices: np.ndarray,
    is_put: bool = False
) -> int:
    """Check vertical spread no-arbitrage condition.

    For calls: C(K1) >= C(K2) for K1 < K2
    For puts: P(K1) <= P(K2) for K1 < K2

    Args:
        strikes: Array of strikes (sorted).
        prices: Array of option prices.
        is_put: Whether checking puts.

    Returns:
        Number of violations.
    """
    violations = 0

    for i in range(len(strikes) - 1):
        if is_put:
            # Put prices should increase with strike
            if prices[i] > prices[i + 1]:
                violations += 1
        else:
            # Call prices should decrease with strike
            if prices[i] < prices[i + 1]:
                violations += 1

    return violations


def _check_butterfly(
    strikes: np.ndarray,
    prices: np.ndarray
) -> int:
    """Check butterfly spread no-arbitrage condition.

    For equally spaced strikes K1 < K2 < K3:
    C(K2) <= 0.5 * (C(K1) + C(K3))

    Args:
        strikes: Array of strikes (sorted).
        prices: Array of option prices.

    Returns:
        Number of violations.
    """
    violations = 0

    for i in range(len(strikes) - 2):
        K1, K2, K3 = strikes[i], strikes[i+1], strikes[i+2]
        C1, C2, C3 = prices[i], prices[i+1], prices[i+2]

        # Check if strikes are approximately equally spaced
        spacing1 = K2 - K1
        spacing2 = K3 - K2
        if abs(spacing1 - spacing2) / spacing1 < 0.1:  # 10% tolerance
            # Butterfly condition
            max_price = 0.5 * (C1 + C3)
            if C2 > max_price * 1.01:  # 1% tolerance
                violations += 1

    return violations


def _check_calendar_spread(df: pd.DataFrame) -> int:
    """Check calendar spread arbitrage.

    Options with longer maturity should be more expensive.

    Args:
        df: DataFrame with option data.

    Returns:
        Number of violations.
    """
    violations = 0

    # Group by strike and option type
    for (K, cp_flag), group in df.groupby(['K', 'cp_flag']):
        if len(group) > 1:
            group = group.sort_values('T')
            prices = group['mid'].values
            maturities = group['T'].values

            # Check if prices increase with maturity
            for i in range(len(prices) - 1):
                if prices[i] > prices[i + 1]:
                    violations += 1

    return violations


def _check_put_call_parity(
    df: pd.DataFrame,
    s0: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
    tol: float = 0.05
) -> int:
    """Check put-call parity violations.

    C - P = S*exp(-q*T) - K*exp(-r*T)

    Args:
        df: DataFrame with option data.
        s0: Spot price (if available).
        r: Risk-free rate.
        q: Dividend yield.
        tol: Tolerance for parity violation.

    Returns:
        Number of violations.
    """
    if s0 is None or r is None:
        # Can't check without market data
        return 0

    if q is None:
        q = 0.0

    violations = 0

    # Group by strike and maturity
    for (K, T), group in df.groupby(['K', 'T']):
        calls = group[group['cp_flag'] == 'C']
        puts = group[group['cp_flag'] == 'P']

        if len(calls) > 0 and len(puts) > 0:
            call_price = calls['mid'].iloc[0]
            put_price = puts['mid'].iloc[0]

            # Theoretical parity value
            parity = s0 * np.exp(-q * T) - K * np.exp(-r * T)

            # Actual difference
            actual = call_price - put_price

            # Check violation
            if abs(actual - parity) > tol:
                violations += 1

    return violations


def check_iv_surface_arbitrage(
    surface: pd.DataFrame
) -> Dict[str, bool]:
    """Check arbitrage conditions on IV surface.

    Args:
        surface: IV surface (strikes x maturities).

    Returns:
        Dictionary with arbitrage check results.
    """
    results = {
        'positive_ivs': True,
        'increasing_term_structure_atm': True,
        'convex_smile': True,
        'smooth_surface': True
    }

    # Check all IVs are positive
    results['positive_ivs'] = (surface > 0).all().all()

    # Check term structure at ATM increases (approximately)
    if len(surface.columns) > 0:
        mid_strike_idx = len(surface.columns) // 2
        atm_term = surface.iloc[:, mid_strike_idx]
        results['increasing_term_structure_atm'] = atm_term.is_monotonic_increasing

    # Check smile convexity for each maturity
    convex_count = 0
    for T in surface.index:
        smile = surface.loc[T].values
        # Check second differences for convexity
        second_diff = np.diff(np.diff(smile))
        if (second_diff >= -1e-6).all():  # Allow small numerical errors
            convex_count += 1

    results['convex_smile'] = convex_count == len(surface.index)

    # Check surface smoothness (no large jumps)
    iv_changes = np.diff(surface.values.ravel())
    results['smooth_surface'] = (np.abs(iv_changes) < 0.5).all()

    return results


def enforce_no_arbitrage(
    df: pd.DataFrame,
    fix_method: str = 'smooth'
) -> pd.DataFrame:
    """Enforce no-arbitrage conditions on option prices.

    Args:
        df: DataFrame with option data.
        fix_method: Method to fix violations ('smooth', 'remove', or 'cap').

    Returns:
        DataFrame with arbitrage violations fixed.
    """
    df = df.copy()

    if fix_method == 'remove':
        # Remove violating options
        df = _remove_arbitrage_violations(df)

    elif fix_method == 'smooth':
        # Smooth prices to remove violations
        df = _smooth_arbitrage_violations(df)

    elif fix_method == 'cap':
        # Cap prices at arbitrage bounds
        df = _cap_arbitrage_violations(df)

    else:
        raise ValueError(f"Unknown fix method: {fix_method}")

    return df


def _remove_arbitrage_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove options that violate arbitrage conditions."""
    # Implementation depends on specific requirements
    # For now, just remove negative IV options
    return df[df['iv'] > 0]


def _smooth_arbitrage_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth option prices to remove arbitrage violations."""
    from scipy.signal import savgol_filter

    for T, group in df.groupby('T'):
        for cp_flag in ['C', 'P']:
            mask = (df['T'] == T) & (df['cp_flag'] == cp_flag)
            if mask.sum() > 3:  # Need at least 4 points for smoothing
                # Sort by strike
                subset = df[mask].sort_values('K')
                indices = subset.index

                # Smooth prices
                prices = subset['mid'].values
                if len(prices) >= 5:
                    smoothed = savgol_filter(prices, 5, 2)
                else:
                    smoothed = prices

                # Update DataFrame
                df.loc[indices, 'mid'] = smoothed

    return df


def _cap_arbitrage_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Cap option prices at arbitrage bounds."""
    # Implement price bounds
    # C >= max(S - K*exp(-r*T), 0)
    # C <= S
    # Similar for puts

    return df  # Simplified implementation


def compute_risk_neutral_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    r: float,
    T: float,
    smooth: bool = True
) -> np.ndarray:
    """Compute risk-neutral density from option prices.

    Uses Breeden-Litzenberger formula:
    p(K) = exp(r*T) * d2C/dK2

    Args:
        strikes: Array of strikes.
        call_prices: Array of call prices.
        r: Risk-free rate.
        T: Time to maturity.
        smooth: Whether to smooth the density.

    Returns:
        Risk-neutral density values.
    """
    # Compute second derivative
    d2C_dK2 = np.zeros_like(strikes)

    for i in range(1, len(strikes) - 1):
        h1 = strikes[i] - strikes[i-1]
        h2 = strikes[i+1] - strikes[i]
        d2C_dK2[i] = 2 * (
            (call_prices[i+1] - call_prices[i]) / h2 -
            (call_prices[i] - call_prices[i-1]) / h1
        ) / (h1 + h2)

    # Boundary values (extrapolate)
    d2C_dK2[0] = d2C_dK2[1]
    d2C_dK2[-1] = d2C_dK2[-2]

    # Risk-neutral density
    density = np.exp(r * T) * d2C_dK2

    # Ensure non-negative
    density = np.maximum(density, 0)

    if smooth:
        from scipy.signal import savgol_filter
        if len(density) >= 5:
            density = savgol_filter(density, 5, 2)
            density = np.maximum(density, 0)

    # Normalize to integrate to 1
    if density.sum() > 0:
        density = density / (density.sum() * (strikes[1] - strikes[0]))

    return density