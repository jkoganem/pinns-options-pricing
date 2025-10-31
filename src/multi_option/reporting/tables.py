"""Table generation for reporting.

This module provides functions for creating formatted tables
for option pricing results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    reference_method: str = 'bs'
) -> pd.DataFrame:
    """Create comparison table of methods.

    Args:
        results: Nested dict of product -> method -> price.
        reference_method: Reference method for differences.

    Returns:
        Formatted DataFrame.
    """
    rows = []

    for product, method_prices in results.items():
        row = {'Product': product}

        # Add prices for each method
        for method, price in method_prices.items():
            row[method.upper()] = price

        # Add differences from reference
        if reference_method in method_prices:
            ref_price = method_prices[reference_method]
            for method, price in method_prices.items():
                if method != reference_method:
                    row[f'{method.upper()}_diff'] = price - ref_price
                    row[f'{method.upper()}_diff_pct'] = (
                        (price - ref_price) / ref_price * 100
                    )

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('Product')

    return df


def create_greeks_table(
    greeks_by_method: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Create Greeks comparison table.

    Args:
        greeks_by_method: Dict of method -> greek_name -> value.

    Returns:
        Formatted DataFrame.
    """
    df = pd.DataFrame(greeks_by_method).T

    # Format column names
    df.columns = [col.capitalize() for col in df.columns]

    # Add statistics row
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()

    return df


def create_convergence_summary(
    convergence_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create convergence summary table.

    Args:
        convergence_data: Dict of parameter -> convergence DataFrame.

    Returns:
        Summary DataFrame.
    """
    rows = []

    for param, df in convergence_data.items():
        if len(df) > 0:
            row = {
                'Parameter': param,
                'Initial_Error': df['abs_err'].iloc[0],
                'Final_Error': df['abs_err'].iloc[-1],
                'Improvement': df['abs_err'].iloc[0] / df['abs_err'].iloc[-1],
                'Convergence_Rate': _estimate_rate(df)
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _estimate_rate(df: pd.DataFrame) -> float:
    """Estimate convergence rate from data."""
    if len(df) < 2:
        return 0.0

    # Log-log regression
    x = np.log(df['x'].values)
    y = np.log(df['abs_err'].values)

    # Remove inf/nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0

    x, y = x[mask], y[mask]

    # Fit line
    coeffs = np.polyfit(x, y, 1)
    return -coeffs[0]  # Negative of slope


def create_iv_statistics_table(
    iv_df: pd.DataFrame
) -> pd.DataFrame:
    """Create IV statistics table.

    Args:
        iv_df: DataFrame with IV data.

    Returns:
        Statistics DataFrame.
    """
    stats = []

    # Group by maturity
    for T, group in iv_df.groupby('T'):
        stats.append({
            'Maturity': T,
            'Mean_IV': group['iv'].mean(),
            'Std_IV': group['iv'].std(),
            'Min_IV': group['iv'].min(),
            'Max_IV': group['iv'].max(),
            'Skew': _compute_iv_skew(group),
            'N_Strikes': len(group)
        })

    return pd.DataFrame(stats)


def _compute_iv_skew(group: pd.DataFrame) -> float:
    """Compute IV skew metric."""
    # Simple skew: (OTM_put_IV - OTM_call_IV) at 90% moneyness
    otm_puts = group[(group['cp_flag'] == 'P') & (group['moneyness'] < 0.95)]
    otm_calls = group[(group['cp_flag'] == 'C') & (group['moneyness'] > 1.05)]

    if len(otm_puts) > 0 and len(otm_calls) > 0:
        return otm_puts['iv'].mean() - otm_calls['iv'].mean()
    return 0.0


def create_performance_table(
    timing_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Create performance comparison table.

    Args:
        timing_results: Dict of method -> metrics.

    Returns:
        Performance DataFrame.
    """
    df = pd.DataFrame(timing_results).T

    # Add relative performance
    if 'bs' in df.index and 'mean_time' in df.columns:
        bs_time = df.loc['bs', 'mean_time']
        df['relative_time'] = df['mean_time'] / bs_time

    # Sort by mean time
    df = df.sort_values('mean_time')

    return df


def format_table_for_display(
    df: pd.DataFrame,
    decimal_places: int = 4,
    percentage_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Format DataFrame for display.

    Args:
        df: DataFrame to format.
        decimal_places: Number of decimal places.
        percentage_cols: Columns to format as percentages.

    Returns:
        Formatted DataFrame.
    """
    df = df.copy()

    # Format numeric columns
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            if percentage_cols and col in percentage_cols:
                df[col] = df[col].apply(lambda x: f'{x:.2f}%')
            else:
                df[col] = df[col].apply(lambda x: f'{x:.{decimal_places}f}')

    return df


def save_tables_to_excel(
    tables: Dict[str, pd.DataFrame],
    out_path: Path
) -> Path:
    """Save multiple tables to Excel file.

    Args:
        tables: Dict of sheet_name -> DataFrame.
        out_path: Output file path.

    Returns:
        Path to saved file.
    """
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name)

    return out_path