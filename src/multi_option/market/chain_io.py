"""Option chain data I/O and preprocessing.

This module handles loading and preprocessing option chain data
for implied volatility analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def load_chain_csv(path: str) -> pd.DataFrame:
    """Load option chain data from CSV file.

    Expected columns:
    - date: Trade/quote date
    - maturity: Option maturity date
    - K: Strike price
    - cp_flag: 'C' for call, 'P' for put
    - bid: Bid price
    - ask: Ask price
    - mid: Mid price (optional, computed if missing)

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with option chain data.
    """
    df = pd.read_csv(path)

    # Standardize column names
    column_mapping = {
        'strike': 'K',
        'Strike': 'K',
        'type': 'cp_flag',
        'Type': 'cp_flag',
        'call_put': 'cp_flag',
        'Bid': 'bid',
        'Ask': 'ask',
        'Mid': 'mid',
        'Date': 'date',
        'Maturity': 'maturity',
        'Expiry': 'maturity'
    }

    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required = ['K', 'cp_flag', 'bid', 'ask']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute mid price if not provided
    if 'mid' not in df.columns:
        df['mid'] = (df['bid'] + df['ask']) / 2

    # Parse dates if provided as strings
    if 'date' in df.columns and df['date'].dtype == object:
        df['date'] = pd.to_datetime(df['date'])

    if 'maturity' in df.columns and df['maturity'].dtype == object:
        df['maturity'] = pd.to_datetime(df['maturity'])

    # Standardize call/put flags
    df['cp_flag'] = df['cp_flag'].str.upper()
    df = df[df['cp_flag'].isin(['C', 'P'])]

    # Remove invalid prices
    df = df[(df['bid'] >= 0) & (df['ask'] >= 0) & (df['mid'] >= 0)]
    df = df[df['bid'] <= df['ask']]

    return df


def preprocess_chain(
    df: pd.DataFrame,
    s0: float,
    r: float,
    q: float
) -> pd.DataFrame:
    """Preprocess option chain data.

    Adds computed columns:
    - T: Time to maturity in years
    - log_moneyness: log(S/K)
    - moneyness: S/K
    - forward_moneyness: F/K where F = S*exp((r-q)*T)

    Args:
        df: Option chain DataFrame.
        s0: Current spot price.
        r: Risk-free rate.
        q: Dividend yield.

    Returns:
        Preprocessed DataFrame.
    """
    df = df.copy()

    # Calculate time to maturity if dates are available
    if 'date' in df.columns and 'maturity' in df.columns:
        df['T'] = (df['maturity'] - df['date']).dt.days / 365.25
        df = df[df['T'] > 0]  # Remove expired options
    elif 'T' not in df.columns:
        # Assume a default if not provided
        df['T'] = 0.25  # 3 months

    # Calculate moneyness measures
    df['moneyness'] = s0 / df['K']
    df['log_moneyness'] = np.log(df['moneyness'])

    # Forward price
    df['forward'] = s0 * np.exp((r - q) * df['T'])
    df['forward_moneyness'] = df['forward'] / df['K']

    # Add spot price
    df['S'] = s0

    # Filter out extreme strikes
    df = df[(df['moneyness'] > 0.5) & (df['moneyness'] < 2.0)]

    # Sort by maturity and strike
    if 'T' in df.columns:
        df = df.sort_values(['T', 'K'])

    return df


def filter_liquid_options(
    df: pd.DataFrame,
    min_spread_pct: float = 0.5,
    min_price: float = 0.05
) -> pd.DataFrame:
    """Filter for liquid options based on bid-ask spread and minimum price.

    Args:
        df: Option chain DataFrame.
        min_spread_pct: Maximum bid-ask spread as percentage of mid.
        min_price: Minimum option price.

    Returns:
        Filtered DataFrame with liquid options.
    """
    df = df.copy()

    # Calculate spread
    df['spread'] = df['ask'] - df['bid']
    df['spread_pct'] = df['spread'] / df['mid'] * 100

    # Filter based on liquidity criteria
    mask = (
        (df['spread_pct'] <= min_spread_pct) &
        (df['mid'] >= min_price)
    )

    return df[mask]


def create_synthetic_chain(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    T_values: List[float],
    K_range: tuple = (0.7, 1.3),
    n_strikes: int = 20,
    spread_bps: float = 50
) -> pd.DataFrame:
    """Create synthetic option chain for testing.

    Args:
        s0: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        T_values: List of times to maturity.
        K_range: Range of strikes as (min_mult, max_mult) of spot.
        n_strikes: Number of strikes per maturity.
        spread_bps: Bid-ask spread in basis points.

    Returns:
        Synthetic option chain DataFrame.
    """
    from multi_option.bs_closed_form import bs_price

    data = []

    for T in T_values:
        # Generate strikes
        K_min = s0 * K_range[0]
        K_max = s0 * K_range[1]
        strikes = np.linspace(K_min, K_max, n_strikes)

        for K in strikes:
            for cp_flag in ['C', 'P']:
                # Calculate theoretical price
                is_call = (cp_flag == 'C')
                theo_price = bs_price(s0, K, r, q, sigma, T, is_call)

                # Add spread
                half_spread = theo_price * spread_bps / 10000
                bid = max(0, theo_price - half_spread)
                ask = theo_price + half_spread
                mid = (bid + ask) / 2

                data.append({
                    'K': K,
                    'T': T,
                    'cp_flag': cp_flag,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'theo': theo_price,
                    'sigma_input': sigma
                })

    df = pd.DataFrame(data)

    # Add preprocessing
    df = preprocess_chain(df, s0, r, q)

    return df


def split_by_maturity(df: pd.DataFrame) -> Dict[float, pd.DataFrame]:
    """Split option chain by maturity.

    Args:
        df: Option chain DataFrame with 'T' column.

    Returns:
        Dictionary mapping maturity to DataFrame.
    """
    if 'T' not in df.columns:
        raise ValueError("DataFrame must have 'T' column")

    maturities = df['T'].unique()
    result = {}

    for T in sorted(maturities):
        result[T] = df[df['T'] == T].copy()

    return result


def split_by_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split option chain into calls and puts.

    Args:
        df: Option chain DataFrame.

    Returns:
        Tuple of (calls_df, puts_df).
    """
    calls = df[df['cp_flag'] == 'C'].copy()
    puts = df[df['cp_flag'] == 'P'].copy()

    return calls, puts


def apply_put_call_parity_filter(
    df: pd.DataFrame,
    s0: float,
    r: float,
    q: float,
    max_violation: float = 0.05
) -> pd.DataFrame:
    """Filter options based on put-call parity violations.

    Args:
        df: Option chain DataFrame.
        s0: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        max_violation: Maximum allowed parity violation.

    Returns:
        Filtered DataFrame.
    """
    # Group by strike and maturity
    grouped = df.groupby(['K', 'T'])

    valid_indices = []

    for (K, T), group in grouped:
        calls = group[group['cp_flag'] == 'C']
        puts = group[group['cp_flag'] == 'P']

        if len(calls) > 0 and len(puts) > 0:
            # Check put-call parity
            call_mid = calls['mid'].iloc[0]
            put_mid = puts['mid'].iloc[0]

            # Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
            parity_value = s0 * np.exp(-q * T) - K * np.exp(-r * T)
            actual_diff = call_mid - put_mid
            violation = abs(actual_diff - parity_value)

            if violation <= max_violation:
                valid_indices.extend(group.index.tolist())

    return df.loc[valid_indices]