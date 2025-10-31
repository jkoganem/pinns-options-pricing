"""Benchmark tests and reference values.

This module provides benchmark tests and known reference values
for validating option pricing implementations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkCase:
    """A benchmark test case with known solution."""
    name: str
    s0: float
    K: float
    r: float
    q: float
    sigma: float
    T: float
    option_type: str  # 'european_call', 'european_put', 'american_put'
    reference_price: float
    reference_source: str
    tolerance: float = 0.001


def get_benchmark_cases() -> List[BenchmarkCase]:
    """Get standard benchmark cases with known solutions.

    Returns:
        List of benchmark cases.
    """
    cases = [
        # European options - Hull examples
        BenchmarkCase(
            name="Hull Example 13.6",
            s0=42.0, K=40.0, r=0.1, q=0.0, sigma=0.2, T=0.5,
            option_type="european_call",
            reference_price=4.7594,
            reference_source="Hull, Options, Futures, and Other Derivatives",
            tolerance=0.001
        ),
        BenchmarkCase(
            name="Hull Example 13.7",
            s0=50.0, K=50.0, r=0.1, q=0.0, sigma=0.3, T=0.25,
            option_type="european_put",
            reference_price=2.3712,
            reference_source="Hull, Options, Futures, and Other Derivatives",
            tolerance=0.001
        ),

        # American put - Broadie-Detemple benchmark
        BenchmarkCase(
            name="Broadie-Detemple American Put",
            s0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.25, T=1.0,
            option_type="american_put",
            reference_price=10.4506,
            reference_source="Broadie & Detemple (1996)",
            tolerance=0.01
        ),

        # At-the-money cases
        BenchmarkCase(
            name="ATM Call Short Maturity",
            s0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=0.1,
            option_type="european_call",
            reference_price=2.5446,
            reference_source="Analytical Black-Scholes",
            tolerance=0.0001
        ),
        BenchmarkCase(
            name="ATM Put Long Maturity",
            s0=100.0, K=100.0, r=0.03, q=0.01, sigma=0.15, T=2.0,
            option_type="european_put",
            reference_price=11.2463,
            reference_source="Analytical Black-Scholes",
            tolerance=0.0001
        ),

        # Deep ITM/OTM cases
        BenchmarkCase(
            name="Deep ITM Call",
            s0=150.0, K=100.0, r=0.05, q=0.0, sigma=0.25, T=0.5,
            option_type="european_call",
            reference_price=51.5216,
            reference_source="Analytical Black-Scholes",
            tolerance=0.001
        ),
        BenchmarkCase(
            name="Deep OTM Put",
            s0=50.0, K=100.0, r=0.05, q=0.0, sigma=0.3, T=0.5,
            option_type="european_put",
            reference_price=47.5729,
            reference_source="Analytical Black-Scholes",
            tolerance=0.001
        ),
    ]

    return cases


def validate_against_benchmarks(
    pricing_func,
    cases: Optional[List[BenchmarkCase]] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """Validate pricing function against benchmark cases.

    Args:
        pricing_func: Function that takes benchmark parameters and returns price.
        cases: List of benchmark cases (uses default if None).
        verbose: Whether to print results.

    Returns:
        Dictionary with validation results.
    """
    if cases is None:
        cases = get_benchmark_cases()

    results = {}

    for case in cases:
        # Compute price
        if case.option_type == "european_call":
            computed = pricing_func(
                case.s0, case.K, case.r, case.q, case.sigma, case.T, call=True
            )
        elif case.option_type == "european_put":
            computed = pricing_func(
                case.s0, case.K, case.r, case.q, case.sigma, case.T, call=False
            )
        elif case.option_type == "american_put":
            computed = pricing_func(
                case.s0, case.K, case.r, case.q, case.sigma, case.T
            )
        else:
            continue

        # Compare
        abs_error = abs(computed - case.reference_price)
        rel_error = abs_error / case.reference_price
        passed = abs_error < case.tolerance

        results[case.name] = {
            'computed': computed,
            'reference': case.reference_price,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'passed': passed,
            'tolerance': case.tolerance
        }

        if verbose:
            status = "[OK]" if passed else "[FAILED]"
            print(f"{status} {case.name}: computed={computed:.4f}, "
                  f"ref={case.reference_price:.4f}, error={abs_error:.4f}")

    # Summary
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)

    if verbose:
        print(f"\nPassed {n_passed}/{n_total} benchmarks")

    return {
        'cases': results,
        'summary': {
            'passed': n_passed,
            'total': n_total,
            'success_rate': n_passed / n_total if n_total > 0 else 0
        }
    }


def generate_random_test_cases(
    n_cases: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Generate random test cases for stress testing.

    Args:
        n_cases: Number of test cases.
        seed: Random seed.

    Returns:
        List of test case dictionaries.
    """
    np.random.seed(seed)

    cases = []

    for i in range(n_cases):
        # Random parameters in reasonable ranges
        s0 = np.random.uniform(50, 150)
        K = np.random.uniform(50, 150)
        r = np.random.uniform(0, 0.1)
        q = np.random.uniform(0, 0.05)
        sigma = np.random.uniform(0.1, 0.5)
        T = np.random.uniform(0.1, 2.0)

        cases.append({
            'id': i,
            's0': s0,
            'K': K,
            'r': r,
            'q': q,
            'sigma': sigma,
            'T': T,
            'moneyness': s0 / K
        })

    return cases


def greeks_benchmarks() -> List[Dict]:
    """Get benchmark cases for Greeks computation.

    Returns:
        List of benchmark cases with known Greeks.
    """
    cases = [
        {
            'name': 'ATM Greeks',
            's0': 100.0, 'K': 100.0, 'r': 0.05, 'q': 0.0,
            'sigma': 0.2, 'T': 1.0, 'call': True,
            'greeks': {
                'delta': 0.6368,
                'gamma': 0.0188,
                'theta': -0.0183,  # Per day
                'vega': 0.3755,    # Per 1% vol
                'rho': 0.5323      # Per 1% rate
            },
            'tolerance': 0.001
        },
        {
            'name': 'ITM Put Greeks',
            's0': 90.0, 'K': 100.0, 'r': 0.05, 'q': 0.0,
            'sigma': 0.25, 'T': 0.5, 'call': False,
            'greeks': {
                'delta': -0.7257,
                'gamma': 0.0278,
                'theta': -0.0098,
                'vega': 0.1557,
                'rho': -0.3134
            },
            'tolerance': 0.001
        }
    ]

    return cases