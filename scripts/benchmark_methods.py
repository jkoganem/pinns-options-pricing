#!/usr/bin/env python3
"""
Comprehensive Benchmark: PINNs vs Crank-Nicolson vs Black-Scholes
Tests all methods for accuracy and speed.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_option.bs_closed_form import bs_price
from multi_option.fd_pde.cn_solver import price_european_cn
from multi_option.mc.pricing import price_european_mc
from multi_option.pinns.simple_pinn import train_simple_pinn, SimplePINN
from multi_option.pinns.advanced_pinn import train_advanced_pinn, AdvancedPINN


def benchmark_black_scholes(S0, K, r, q, sigma, T, test_spots):
    """Benchmark Black-Scholes analytical formula."""
    start_time = time.time()

    prices = []
    for S in test_spots:
        price = bs_price(S, K, r, q, sigma, T, call=True)
        prices.append(price)

    elapsed = time.time() - start_time

    return {
        'method': 'Black-Scholes',
        'prices': prices,
        'time': elapsed,
        'time_per_price': elapsed / len(test_spots),
        'error': 0.0  # Reference method
    }


def benchmark_crank_nicolson(S0, K, r, q, sigma, T, test_spots):
    """Benchmark Crank-Nicolson PDE solver."""
    start_time = time.time()

    # CN parameters
    s_max = 3 * K
    ns = 501  # Spatial grid points
    nt = 500  # Time steps

    prices = []
    for S in test_spots:
        price = price_european_cn(S, K, r, q, sigma, T, s_max, ns, nt, call=True)
        prices.append(price)

    elapsed = time.time() - start_time

    # Calculate errors vs Black-Scholes
    bs_prices = [bs_price(S, K, r, q, sigma, T, call=True) for S in test_spots]
    errors = [abs(p - bs) / bs * 100 for p, bs in zip(prices, bs_prices)]

    return {
        'method': 'Crank-Nicolson',
        'prices': prices,
        'time': elapsed,
        'time_per_price': elapsed / len(test_spots),
        'error': np.mean(errors),
        'max_error': np.max(errors)
    }


def benchmark_monte_carlo(S0, K, r, q, sigma, T, test_spots, n_paths=100000):
    """Benchmark Monte Carlo simulation."""
    start_time = time.time()

    prices = []
    for S in test_spots:
        result = price_european_mc(S, K, r, q, sigma, T, 252, n_paths, 42, call=True)
        prices.append(result.price)

    elapsed = time.time() - start_time

    # Calculate errors vs Black-Scholes
    bs_prices = [bs_price(S, K, r, q, sigma, T, call=True) for S in test_spots]
    errors = [abs(p - bs) / bs * 100 for p, bs in zip(prices, bs_prices)]

    return {
        'method': f'Monte Carlo ({n_paths:,} paths)',
        'prices': prices,
        'time': elapsed,
        'time_per_price': elapsed / len(test_spots),
        'error': np.mean(errors),
        'max_error': np.max(errors)
    }


def benchmark_simple_pinn(S0, K, r, q, sigma, T, test_spots, epochs=5000):
    """Benchmark Simple PINN."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training time
    start_time = time.time()

    model, info = train_simple_pinn(
        K=K, r=r, q=q, sigma=sigma, T=T,
        n_epochs=epochs,
        device=device,
        verbose=False
    )

    training_time = time.time() - start_time

    # Inference time
    start_time = time.time()

    model.eval()
    prices = []
    with torch.no_grad():
        for S in test_spots:
            S_tensor = torch.tensor([[S]], device=device, dtype=torch.float32)
            tau_tensor = torch.tensor([[T]], device=device, dtype=torch.float32)
            price = model(S_tensor, tau_tensor).item()
            prices.append(price)

    inference_time = time.time() - start_time

    # Calculate errors vs Black-Scholes
    bs_prices = [bs_price(S, K, r, q, sigma, T, call=True) for S in test_spots]
    errors = [abs(p - bs) / bs * 100 for p, bs in zip(prices, bs_prices)]

    return {
        'method': f'SimplePINN ({epochs} epochs)',
        'prices': prices,
        'training_time': training_time,
        'inference_time': inference_time,
        'total_time': training_time + inference_time,
        'time_per_price': inference_time / len(test_spots),
        'error': np.mean(errors),
        'max_error': np.max(errors)
    }


def benchmark_advanced_pinn(S0, K, r, q, sigma, T, test_spots, epochs=5000, config=None):
    """Benchmark Advanced PINN with state-of-the-art techniques."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default configuration
    if config is None:
        config = {
            'use_log_transform': True,
            'use_hard_terminal': True,
            'use_fourier': True,
            'fourier_features': 32,
            'use_rbas': True,
            'use_adaptive_weights': True,
            'activation': 'tanh',
            'use_lbfgs': True,
            'lbfgs_after': int(epochs * 0.7),
            'lbfgs_iters': 200
        }

    # Training time
    start_time = time.time()

    model, info = train_advanced_pinn(
        K=K, r=r, q=q, sigma=sigma, T=T,
        n_epochs=epochs,
        device=device,
        verbose=False,
        **config
    )

    training_time = time.time() - start_time

    # Inference time
    start_time = time.time()

    model.eval()
    prices = []
    with torch.no_grad():
        for S in test_spots:
            S_tensor = torch.tensor([[S]], device=device, dtype=torch.float64)
            tau_tensor = torch.tensor([[T]], device=device, dtype=torch.float64)
            price = model(S_tensor, tau_tensor, is_call=True).item()
            prices.append(price)

    inference_time = time.time() - start_time

    # Calculate errors vs Black-Scholes
    bs_prices = [bs_price(S, K, r, q, sigma, T, call=True) for S in test_spots]
    errors = [abs(p - bs) / bs * 100 for p, bs in zip(prices, bs_prices)]

    features = []
    if config.get('use_log_transform'): features.append('log')
    if config.get('use_hard_terminal'): features.append('hard_IC')
    if config.get('use_fourier'): features.append('Fourier')
    if config.get('use_rbas'): features.append('RBAS')
    if config.get('use_adaptive_weights'): features.append('adapt_w')
    if config.get('use_lbfgs'): features.append('L-BFGS')

    method_name = f"AdvancedPINN ({', '.join(features)})"

    return {
        'method': method_name,
        'prices': prices,
        'training_time': training_time,
        'inference_time': inference_time,
        'total_time': training_time + inference_time,
        'time_per_price': inference_time / len(test_spots),
        'error': np.mean(errors),
        'max_error': np.max(errors),
        'final_loss': info['best_loss']
    }


def print_results(results_list, test_spots, K):
    """Print benchmark results in a formatted table."""

    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)

    # Summary table
    print(f"\n{'Method':<35} {'Mean Error %':<12} {'Max Error %':<12} "
          f"{'Time (s)':<12} {'Time/Price (ms)':<15}")
    print("-"*100)

    for result in results_list:
        method = result['method']
        mean_error = result.get('error', 0.0)
        max_error = result.get('max_error', mean_error)

        if 'PINN' in method:
            total_time = result.get('total_time', 0)
            time_per = result.get('time_per_price', 0) * 1000  # Convert to ms
            print(f"{method:<35} {mean_error:<12.4f} {max_error:<12.4f} "
                  f"{total_time:<12.2f} {time_per:<15.3f}")

            if result.get('training_time'):
                print(f"  -> Training: {result['training_time']:.2f}s, "
                      f"Inference: {result['inference_time']:.2f}s")
        else:
            time_total = result.get('time', 0)
            time_per = result.get('time_per_price', 0) * 1000  # Convert to ms
            print(f"{method:<35} {mean_error:<12.4f} {max_error:<12.4f} "
                  f"{time_total:<12.2f} {time_per:<15.3f}")

    print("="*100)

    # Detailed price comparison for spots around strike
    bs_result = next(r for r in results_list if r['method'] == 'Black-Scholes')

    print(f"\nDETAILED PRICE COMPARISON (Around Strike K={K})")
    print("-"*100)

    # Select spots near strike
    strike_spots = [s for s in test_spots if 0.8*K <= s <= 1.2*K]

    header = f"{'Spot':<10}"
    for result in results_list:
        if len(result['method']) > 12:
            header += f" {result['method'][:12]:<12}"
        else:
            header += f" {result['method']:<12}"
    print(header)
    print("-"*100)

    for i, S in enumerate(strike_spots):
        row = f"{S:<10.2f}"
        for result in results_list:
            if i < len(result['prices']):
                row += f" {result['prices'][i]:<12.4f}"
        print(row)

    print("="*100)


def compare_configurations(K, r, q, sigma, T, test_spots, epochs=3000):
    """Compare different PINN configurations."""

    configs = [
        {'name': 'Baseline', 'config': {
            'use_log_transform': False,
            'use_hard_terminal': False,
            'use_fourier': False,
            'use_rbas': False,
            'use_adaptive_weights': False,
            'use_lbfgs': False
        }},
        {'name': 'With Log Transform', 'config': {
            'use_log_transform': True,
            'use_hard_terminal': False,
            'use_fourier': False,
            'use_rbas': False,
            'use_adaptive_weights': False,
            'use_lbfgs': False
        }},
        {'name': 'With Hard Terminal', 'config': {
            'use_log_transform': False,
            'use_hard_terminal': True,
            'use_fourier': False,
            'use_rbas': False,
            'use_adaptive_weights': False,
            'use_lbfgs': False
        }},
        {'name': 'With Fourier', 'config': {
            'use_log_transform': False,
            'use_hard_terminal': False,
            'use_fourier': True,
            'use_rbas': False,
            'use_adaptive_weights': False,
            'use_lbfgs': False
        }},
        {'name': 'Full Advanced', 'config': {
            'use_log_transform': True,
            'use_hard_terminal': True,
            'use_fourier': True,
            'use_rbas': True,
            'use_adaptive_weights': True,
            'use_lbfgs': True,
            'lbfgs_after': int(epochs * 0.7),
            'lbfgs_iters': 200
        }}
    ]

    print("\n" + "="*80)
    print("PINN CONFIGURATION COMPARISON")
    print("="*80)

    results = []
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}...")
        result = benchmark_advanced_pinn(
            100, K, r, q, sigma, T, test_spots, epochs, cfg['config']
        )
        result['name'] = cfg['name']
        results.append(result)
        print(f"  Error: {result['error']:.2f}%, Time: {result['total_time']:.1f}s")

    # Summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Mean Error %':<12} {'Training (s)':<12} {'Improvement':<15}")
    print("-"*80)

    baseline_error = results[0]['error']
    for result in results:
        improvement = (baseline_error - result['error']) / baseline_error * 100
        print(f"{result['name']:<20} {result['error']:<12.2f} "
              f"{result['training_time']:<12.1f} {improvement:>14.1f}%")

    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PINNs vs Classical Methods")

    # Option parameters
    parser.add_argument('--K', type=float, default=100.0, help='Strike price')
    parser.add_argument('--r', type=float, default=0.05, help='Risk-free rate')
    parser.add_argument('--q', type=float, default=0.02, help='Dividend yield')
    parser.add_argument('--sigma', type=float, default=0.2, help='Volatility')
    parser.add_argument('--T', type=float, default=1.0, help='Time to maturity')

    # Benchmark settings
    parser.add_argument('--epochs', type=int, default=5000, help='PINN training epochs')
    parser.add_argument('--mc-paths', type=int, default=100000, help='Monte Carlo paths')
    parser.add_argument('--test-spots', nargs='+', type=float,
                       help='Spot prices to test (default: 80-120 in steps of 5)')

    # Options
    parser.add_argument('--skip-mc', action='store_true', help='Skip Monte Carlo')
    parser.add_argument('--skip-simple', action='store_true', help='Skip Simple PINN')
    parser.add_argument('--compare-configs', action='store_true',
                       help='Compare different PINN configurations')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer epochs')

    args = parser.parse_args()

    # Set test spots
    if args.test_spots:
        test_spots = args.test_spots
    else:
        test_spots = list(np.linspace(80, 120, 9))

    if args.quick:
        args.epochs = 1000
        args.mc_paths = 10000

    print("\n" + "="*100)
    print("OPTIONS PRICING METHOD BENCHMARK")
    print("="*100)
    print(f"Parameters: K={args.K}, r={args.r}, q={args.q}, sigma={args.sigma}, T={args.T}")
    print(f"Test spots: {test_spots}")
    print(f"PINN epochs: {args.epochs}")
    print("="*100)

    results = []

    # Benchmark classical methods
    print("\n1. Black-Scholes (Reference)...")
    bs_result = benchmark_black_scholes(100, args.K, args.r, args.q, args.sigma, args.T, test_spots)
    results.append(bs_result)

    print("2. Crank-Nicolson PDE...")
    cn_result = benchmark_crank_nicolson(100, args.K, args.r, args.q, args.sigma, args.T, test_spots)
    results.append(cn_result)

    if not args.skip_mc:
        print(f"3. Monte Carlo ({args.mc_paths:,} paths)...")
        mc_result = benchmark_monte_carlo(100, args.K, args.r, args.q, args.sigma, args.T,
                                         test_spots, args.mc_paths)
        results.append(mc_result)

    # Benchmark PINNs
    if not args.skip_simple:
        print(f"4. Simple PINN ({args.epochs} epochs)...")
        simple_result = benchmark_simple_pinn(100, args.K, args.r, args.q, args.sigma, args.T,
                                             test_spots, args.epochs)
        results.append(simple_result)

    print(f"5. Advanced PINN (all features, {args.epochs} epochs)...")
    advanced_result = benchmark_advanced_pinn(100, args.K, args.r, args.q, args.sigma, args.T,
                                             test_spots, args.epochs)
    results.append(advanced_result)

    # Print results
    print_results(results, test_spots, args.K)

    # Compare configurations if requested
    if args.compare_configs:
        compare_configurations(args.K, args.r, args.q, args.sigma, args.T,
                             test_spots, args.epochs)

    # Final verdict
    print("\n" + "="*100)
    print("VERDICT")
    print("="*100)

    cn_error = cn_result['error']
    advanced_error = advanced_result['error']

    if advanced_error < 0.1:
        print("EXCELLENT: Advanced PINN achieves <0.1% error - BETTER than Crank-Nicolson!")
    elif advanced_error < cn_error:
        print(f"SUCCESS: Advanced PINN ({advanced_error:.3f}%) beats "
              f"Crank-Nicolson ({cn_error:.3f}%)")
    elif advanced_error < 1.0:
        print(f"GOOD: Advanced PINN achieves <1% error ({advanced_error:.3f}%)")
        print(f"  Crank-Nicolson is still better ({cn_error:.3f}%)")
    else:
        print(f"NEEDS IMPROVEMENT: Advanced PINN error ({advanced_error:.3f}%) "
              f"vs Crank-Nicolson ({cn_error:.3f}%)")

    # Speed comparison
    cn_time = cn_result['time_per_price'] * 1000  # ms
    pinn_inf_time = advanced_result['time_per_price'] * 1000  # ms

    print(f"\nSpeed comparison (inference only):")
    print(f"  Crank-Nicolson: {cn_time:.3f} ms/price")
    print(f"  Advanced PINN: {pinn_inf_time:.3f} ms/price")

    if pinn_inf_time < cn_time:
        speedup = cn_time / pinn_inf_time
        print(f"  -> PINN is {speedup:.1f}x faster!")
    else:
        slowdown = pinn_inf_time / cn_time
        print(f"  -> PINN is {slowdown:.1f}x slower")

    print("="*100)


if __name__ == "__main__":
    main()