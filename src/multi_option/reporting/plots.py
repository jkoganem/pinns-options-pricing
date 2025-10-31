"""Plotting functions for visualization.

This module provides functions for creating various plots
for option pricing analysis and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_method_comparison(
    prices: Dict[str, float],
    out_dir: Path
) -> Path:
    """Plot comparison of prices across methods.

    Args:
        prices: Dictionary mapping method names to prices.
        out_dir: Output directory for plot.

    Returns:
        Path to saved plot file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot of prices
    methods = list(prices.keys())
    values = list(prices.values())

    ax1.bar(methods, values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Option Price')
    ax1.set_title('Option Prices by Method')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (method, value) in enumerate(prices.items()):
        ax1.text(i, value + max(values) * 0.01, f'${value:.4f}',
                ha='center', va='bottom')

    # Deviation from mean
    mean_price = np.mean(values)
    deviations = [(v - mean_price) / mean_price * 100 for v in values]

    colors = ['green' if d >= 0 else 'red' for d in deviations]
    ax2.barh(methods, deviations, color=colors, alpha=0.6)
    ax2.set_xlabel('Deviation from Mean (%)')
    ax2.set_ylabel('Method')
    ax2.set_title('Price Deviation from Mean')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = out_dir / 'method_comparison.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_convergence(
    df: pd.DataFrame,
    xlabel: str,
    out_dir: Path
) -> Path:
    """Plot convergence analysis.

    Args:
        df: DataFrame with columns ['x', 'abs_err', 'rel_err'].
        xlabel: Label for x-axis.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute error
    ax1.loglog(df['x'], df['abs_err'], 'o-', color='blue', markersize=6)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Convergence: Absolute Error')
    ax1.grid(True, alpha=0.3, which='both')

    # Add convergence rate line if possible
    if len(df) > 2:
        # Fit power law
        coeffs = np.polyfit(np.log(df['x']), np.log(df['abs_err']), 1)
        rate = -coeffs[0]
        fitted = np.exp(np.polyval(coeffs, np.log(df['x'])))
        ax1.loglog(df['x'], fitted, 'r--', alpha=0.5,
                  label=f'Rate: {rate:.2f}')
        ax1.legend()

    # Relative error
    ax2.loglog(df['x'], df['rel_err'], 's-', color='green', markersize=6)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Convergence: Relative Error')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save
    out_path = out_dir / 'convergence.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_greeks(
    g: 'GreeksResult',
    out_dir: Path
) -> Path:
    """Plot Greeks values.

    Args:
        g: GreeksResult object.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Greek values
    greeks_dict = {
        'Delta': g.delta,
        'Gamma': g.gamma,
        'Theta': g.theta,
        'Vega': g.vega,
        'Rho': g.rho
    }

    # Normalize for visualization (different scales)
    names = list(greeks_dict.keys())
    values = list(greeks_dict.values())

    # Create bar plot
    bars = ax.bar(names, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    ax.set_ylabel('Value')
    ax.set_title('Option Greeks')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top')

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save
    out_path = out_dir / 'greeks.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_pinn_training(
    loss_curve: pd.Series,
    out_dir: Path
) -> Path:
    """Plot PINN training loss curve.

    Args:
        loss_curve: Series of loss values by epoch.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(len(loss_curve))
    ax.semilogy(epochs, loss_curve, color='blue', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('PINN Training Loss')
    ax.grid(True, alpha=0.3)

    # Mark minimum
    min_idx = loss_curve.argmin()
    min_val = loss_curve.iloc[min_idx]
    ax.plot(min_idx, min_val, 'ro', markersize=8,
           label=f'Min: {min_val:.2e} at epoch {min_idx}')
    ax.legend()

    plt.tight_layout()

    # Save
    out_path = out_dir / 'pinn_loss.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_pinn_surface(
    surface_df: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot PINN solution surface.

    Args:
        surface_df: DataFrame with columns ['S', 't', 'V'].
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Reshape data for surface plot
    Sunique = surface_df['S'].unique()
    tunique = surface_df['t'].unique()
    S_mesh, t_mesh = np.meshgrid(Sunique, tunique)

    V_mesh = surface_df.pivot_table(values='V', index='t', columns='S').values

    # Surface plot
    surf = ax.plot_surface(S_mesh, t_mesh, V_mesh,
                          cmap='viridis', alpha=0.8)

    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Value')
    ax.set_title('PINN Option Price Surface')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()

    # Save
    out_path = out_dir / 'pinn_surface.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_pinn_residual_heatmap(
    resid_df: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot PINN PDE residual as heatmap.

    Args:
        resid_df: DataFrame with columns ['S', 't', 'residual'].
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pivot for heatmap
    pivot = resid_df.pivot_table(values='residual', index='t', columns='S')

    # Heatmap
    im = ax.imshow(np.log10(pivot.values + 1e-10), aspect='auto',
                  cmap='hot', origin='lower')

    # Labels
    ax.set_xlabel('Stock Price Index')
    ax.set_ylabel('Time Index')
    ax.set_title('PINN PDE Residual (log10 scale)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(|Residual|)')

    plt.tight_layout()

    # Save
    out_path = out_dir / 'pinn_residual.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_iv_surface(
    iv_pivot: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot implied volatility surface.

    Args:
        iv_pivot: Pivot table with strikes as columns, maturities as index.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh
    K = iv_pivot.columns.values
    T = iv_pivot.index.values
    K_mesh, T_mesh = np.meshgrid(K, T)

    # Surface plot
    surf = ax.plot_surface(K_mesh, T_mesh, iv_pivot.values,
                          cmap='coolwarm', alpha=0.9)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()

    # Save
    out_path = out_dir / 'iv_surface.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_smile_slices(
    smiles: Dict[str, pd.DataFrame],
    out_dir: Path
) -> Path:
    """Plot IV smile slices at different maturities.

    Args:
        smiles: Dictionary mapping maturity labels to smile DataFrames.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(smiles)))

    for (label, smile_df), color in zip(smiles.items(), colors):
        ax.plot(smile_df['K'], smile_df['iv'], 'o-',
               label=label, color=color, markersize=4)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Implied Volatility Smiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = out_dir / 'iv_smiles.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path


def plot_noarb_warnings(
    stats: Dict[str, int],
    out_dir: Path
) -> Path:
    """Plot no-arbitrage violation statistics.

    Args:
        stats: Dictionary with violation counts.
        out_dir: Output directory.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of violations
    violations = list(stats.keys())
    counts = list(stats.values())

    bars = ax.bar(violations, counts, color='red', alpha=0.6)
    ax.set_ylabel('Number of Violations')
    ax.set_title('No-Arbitrage Violation Counts')
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   str(count), ha='center', va='bottom')

    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save
    out_path = out_dir / 'noarb_warnings.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    return out_path