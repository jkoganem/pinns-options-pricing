"""Method comparison and benchmarking utilities.

This module provides functions for comparing different option pricing
methods and analyzing their performance.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any
from multi_option.datatypes import EngineConfig, PriceResult


def compare_prices(results: Dict[str, PriceResult]) -> pd.DataFrame:
    """Compare prices from different methods.

    Args:
        results: Dictionary mapping method names to PriceResult objects.

    Returns:
        DataFrame with comparison statistics.
    """
    data = []

    for method, result in results.items():
        row = {
            'method': method,
            'price': result.price,
            'stderr': result.stderr,
            'product': result.product
        }

        # Add metadata
        for key, value in result.meta.items():
            row[f'meta_{key}'] = value

        data.append(row)

    df = pd.DataFrame(data)

    # Add relative differences if we have a reference (e.g., BS)
    if 'bs' in results:
        ref_price = results['bs'].price
        df['diff_from_bs'] = df['price'] - ref_price
        df['rel_diff_from_bs'] = (df['price'] - ref_price) / ref_price * 100

    # Add statistics
    df['mean_price'] = df['price'].mean()
    df['std_price'] = df['price'].std()

    return df


def benchmark_methods(
    cfg: EngineConfig,
    product: str,
    methods: List[str],
    n_runs: int = 5
) -> pd.DataFrame:
    """Benchmark performance of different methods.

    Args:
        cfg: Engine configuration.
        product: Option product to price.
        methods: List of methods to benchmark.
        n_runs: Number of runs for timing.

    Returns:
        DataFrame with benchmark results.
    """
    results = []

    for method in methods:
        times = []
        prices = []

        for run in range(n_runs):
            start_time = time.time()

            # Price option
            price = _price_with_method(cfg, product, method)

            elapsed = time.time() - start_time
            times.append(elapsed)
            prices.append(price)

        results.append({
            'method': method,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_price': np.mean(prices),
            'std_price': np.std(prices)
        })

    return pd.DataFrame(results)


def _price_with_method(
    cfg: EngineConfig,
    product: str,
    method: str
) -> float:
    """Price option with specified method.

    Args:
        cfg: Engine configuration.
        product: Option product.
        method: Pricing method.

    Returns:
        Option price.
    """
    if product == 'european_call':
        from multi_option.products.european import price_european_single_method
        result = price_european_single_method(cfg, method, call=True)
        return result.price

    elif product == 'european_put':
        from multi_option.products.european import price_european_single_method
        result = price_european_single_method(cfg, method, call=False)
        return result.price

    elif product == 'american_put':
        from multi_option.products.american_put import price_american_put_methods
        results = price_american_put_methods(cfg)
        if method in results:
            return results[method].price

    elif product == 'barrierup_out_call':
        from multi_option.products.barrierup_out_call import price_barrier_methods
        if cfg.barrier is None:
            cfg.barrier = cfg.s0 * 1.3
        results = price_barrier_methods(cfg)
        if method in results:
            return results[method].price

    raise ValueError(f"Cannot price {product} with method {method}")


def accuracy_analysis(
    cfg: EngineConfig,
    product: str,
    reference_method: str = 'bs'
) -> pd.DataFrame:
    """Analyze accuracy of different methods.

    Args:
        cfg: Engine configuration.
        product: Option product.
        reference_method: Reference method for comparison.

    Returns:
        DataFrame with accuracy metrics.
    """
    # Get reference price
    ref_price = _price_with_method(cfg, product, reference_method)

    methods = ['bs', 'pde', 'mc', 'pinn']
    results = []

    for method in methods:
        try:
            price = _price_with_method(cfg, product, method)

            abs_error = abs(price - ref_price)
            rel_error = abs_error / abs(ref_price) if ref_price != 0 else abs_error

            results.append({
                'method': method,
                'price': price,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'rel_error_pct': rel_error * 100
            })
        except:
            # Method not available for this product
            continue

    df = pd.DataFrame(results)
    df['rank'] = df['abs_error'].rank()

    return df


def parameter_sensitivity(
    cfg: EngineConfig,
    product: str,
    param_name: str,
    param_range: tuple,
    n_points: int = 20
) -> pd.DataFrame:
    """Analyze sensitivity to parameter changes.

    Args:
        cfg: Engine configuration.
        product: Option product.
        param_name: Parameter to vary ('sigma', 'r', 'q', 'K', 'T').
        param_range: Range of parameter values.
        n_points: Number of points to test.

    Returns:
        DataFrame with sensitivity analysis.
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    results = []

    # Store original value
    original_value = getattr(cfg, param_name)

    for value in param_values:
        # Update parameter
        setattr(cfg, param_name, value)

        # Price with each method
        row = {'param': param_name, 'value': value}

        for method in ['bs', 'pde', 'mc']:
            try:
                price = _price_with_method(cfg, product, method)
                row[f'{method}_price'] = price
            except:
                row[f'{method}_price'] = np.nan

        results.append(row)

    # Restore original value
    setattr(cfg, param_name, original_value)

    return pd.DataFrame(results)


def create_comparison_summary(
    all_results: Dict[str, Dict[str, PriceResult]]
) -> pd.DataFrame:
    """Create summary table of all results.

    Args:
        all_results: Nested dict of product -> method -> PriceResult.

    Returns:
        Summary DataFrame.
    """
    rows = []

    for product, method_results in all_results.items():
        for method, result in method_results.items():
            rows.append({
                'product': product,
                'method': method,
                'price': result.price,
                'stderr': result.stderr
            })

    df = pd.DataFrame(rows)

    # Pivot to get methods as columns
    pivot = df.pivot_table(
        values='price',
        index='product',
        columns='method',
        aggfunc='first'
    )

    # Add differences
    if 'bs' in pivot.columns:
        for col in pivot.columns:
            if col != 'bs':
                pivot[f'{col}_diff'] = pivot[col] - pivot['bs']
                pivot[f'{col}_diff_pct'] = (pivot[col] - pivot['bs']) / pivot['bs'] * 100

    return pivot


def compute_method_agreement(
    results: Dict[str, PriceResult],
    threshold: float = 0.01
) -> Dict[str, Any]:
    """Compute agreement metrics between methods.

    Args:
        results: Dictionary of method results.
        threshold: Relative difference threshold for agreement.

    Returns:
        Dictionary with agreement metrics.
    """
    prices = [r.price for r in results.values()]
    methods = list(results.keys())

    # Basic statistics
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    cv = std_price / mean_price if mean_price != 0 else 0

    # Pairwise agreement
    n_methods = len(methods)
    agreement_matrix = np.ones((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            price_i = results[methods[i]].price
            price_j = results[methods[j]].price
            rel_diff = abs(price_i - price_j) / max(abs(price_i), abs(price_j))
            agreement_matrix[i, j] = int(rel_diff < threshold)
            agreement_matrix[j, i] = agreement_matrix[i, j]

    # Overall agreement score
    agreement_score = (agreement_matrix.sum() - n_methods) / (n_methods * (n_methods - 1))

    return {
        'mean_price': mean_price,
        'std_price': std_price,
        'coefficient_variation': cv,
        'agreement_score': agreement_score,
        'agreement_matrix': agreement_matrix,
        'methods': methods
    }