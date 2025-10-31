"""Serialization utilities for UI and data export.

This module provides functions for serializing results to JSON
and other formats for UI consumption and data export.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def writeui_bundle(
    out_dir: Path,
    meta: Dict[str, Any],
    comparison: Dict[str, float],
    convergence: pd.DataFrame,
    greeks: Dict[str, float],
    pinn_loss: pd.DataFrame,
    pinn_surface: pd.DataFrame,
    iv_surface_df: pd.DataFrame,
    smiles_example: Dict[str, pd.DataFrame]
) -> Path:
    """Write UI bundle JSON for Next.js frontend.

    Args:
        out_dir: Output directory.
        meta: Metadata dictionary.
        comparison: Method comparison results.
        convergence: Convergence analysis data.
        greeks: Greeks values.
        pinn_loss: PINN training loss history.
        pinn_surface: PINN solution surface.
        iv_surface_df: IV surface data.
        smiles_example: Example smile slices.

    Returns:
        Path to saved JSON file.
    """
    # Create UI directory
    ui_dir = out_dir / 'ui'
    ui_dir.mkdir(parents=True, exist_ok=True)

    # Prepare bundle
    bundle = {
        'metadata': _serialize_metadata(meta),
        'comparison': _serialize_comparison(comparison),
        'convergence': _serialize_dataframe(convergence),
        'greeks': greeks,
        'pinn': {
            'loss': _serialize_series(pinn_loss) if not pinn_loss.empty else [],
            'surface': _serialize_dataframe(pinn_surface) if not pinn_surface.empty else []
        },
        'iv': {
            'surface': _serialize_dataframe(iv_surface_df) if not iv_surface_df.empty else [],
            'smiles': _serialize_smiles(smiles_example)
        },
        'timestamp': datetime.now().isoformat()
    }

    # Write JSON
    json_path = ui_dir / 'ui_bundle.json'
    with open(json_path, 'w') as f:
        json.dump(bundle, f, indent=2)

    return json_path


def _serialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize metadata for JSON."""
    serialized = {}
    for key, value in meta.items():
        if isinstance(value, (np.integer, np.floating)):
            serialized[key] = float(value)
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serialized[key] = value.to_dict('records')
        else:
            serialized[key] = value
    return serialized


def _serialize_comparison(comparison: Dict[str, float]) -> List[Dict]:
    """Serialize method comparison for JSON."""
    return [
        {'method': method, 'price': float(price)}
        for method, price in comparison.items()
    ]


def _serialize_dataframe(df: pd.DataFrame) -> List[Dict]:
    """Serialize DataFrame to list of records."""
    if df.empty:
        return []

    # Convert to records
    records = df.to_dict('records')

    # Ensure all values are JSON serializable
    for record in records:
        for key, value in record.items():
            if isinstance(value, (np.integer, np.floating)):
                record[key] = float(value)
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif pd.isna(value):
                record[key] = None

    return records


def _serialize_series(series: pd.Series) -> List[float]:
    """Serialize Series to list."""
    return [float(v) if not pd.isna(v) else None for v in series.values]


def _serialize_smiles(smiles: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
    """Serialize smile slices."""
    serialized = {}
    for label, df in smiles.items():
        serialized[label] = _serialize_dataframe(df)
    return serialized


def export_results_csv(
    results: Dict[str, Any],
    out_dir: Path
) -> List[Path]:
    """Export results to CSV files.

    Args:
        results: Dictionary of results to export.
        out_dir: Output directory.

    Returns:
        List of paths to created CSV files.
    """
    csv_dir = out_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for name, data in results.items():
        if isinstance(data, pd.DataFrame):
            csv_path = csv_dir / f'{name}.csv'
            data.to_csv(csv_path, index=True)
            paths.append(csv_path)
        elif isinstance(data, dict):
            # Convert dict to DataFrame if possible
            try:
                df = pd.DataFrame([data])
                csv_path = csv_dir / f'{name}.csv'
                df.to_csv(csv_path, index=False)
                paths.append(csv_path)
            except:
                pass

    return paths


def create_summary_report(
    config: 'EngineConfig',
    results: Dict[str, Any],
    out_dir: Path
) -> Path:
    """Create text summary report.

    Args:
        config: Engine configuration.
        results: All results.
        out_dir: Output directory.

    Returns:
        Path to summary report.
    """
    report_path = out_dir / 'summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MULTI-METHOD OPTION PRICING ENGINE - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Spot Price (S0): ${config.s0:.2f}\n")
        f.write(f"Strike Price (K): ${config.K:.2f}\n")
        f.write(f"Risk-Free Rate: {config.r:.4f}\n")
        f.write(f"Dividend Yield: {config.q:.4f}\n")
        f.write(f"Volatility: {config.sigma:.4f}\n")
        f.write(f"Time to Maturity: {config.T:.4f} years\n")
        f.write(f"Grid Size: {config.ns} x {config.nt}\n")
        f.write(f"MC Paths: {config.mc_paths:,}\n")
        f.write(f"MC Steps: {config.mc_steps}\n")
        f.write(f"PINN Epochs: {config.pinn_epochs}\n")
        f.write("\n")

        # Results
        if 'prices' in results:
            f.write("PRICING RESULTS:\n")
            f.write("-" * 40 + "\n")
            for method, price in results['prices'].items():
                f.write(f"{method.upper()}: ${price:.4f}\n")
            f.write("\n")

        # Greeks
        if 'greeks' in results:
            f.write("GREEKS:\n")
            f.write("-" * 40 + "\n")
            for greek, value in results['greeks'].items():
                f.write(f"{greek.capitalize()}: {value:.6f}\n")
            f.write("\n")

        # Convergence
        if 'convergence' in results:
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            conv = results['convergence']
            if hasattr(conv, 'tail'):
                f.write(f"Final Error: {conv['abs_err'].iloc[-1]:.6f}\n")
            f.write("\n")

        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return report_path


def create_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    out_dir: Path
) -> Path:
    """Create LaTeX table from DataFrame.

    Args:
        df: DataFrame to convert.
        caption: Table caption.
        label: LaTeX label.
        out_dir: Output directory.

    Returns:
        Path to LaTeX file.
    """
    latex_dir = out_dir / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    latex_path = latex_dir / f'{label}.tex'

    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{tab:{label}}}\n")
        f.write(df.to_latex(float_format=lambda x: f'{x:.4f}'))
        f.write("\\end{table}\n")

    return latex_path