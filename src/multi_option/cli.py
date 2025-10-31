"""Command-line interface for the option pricing engine.

This module provides the main CLI entry point for running the
multi-method option pricing engine.
"""

import click
import os
import sys
import json
import pandas as pd
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, Optional

from multi_option.config import parse_config
from multi_option.datatypes import EngineConfig, GreeksResult
from multi_option.products.european import price_european_all_methods
from multi_option.products.american_put import price_american_put_methods
from multi_option.products.barrierup_out_call import price_barrier_methods
from multi_option.bs_closed_form import bs_greeks
from multi_option.evaluation.convergence import convergence_curve
from multi_option.evaluation.compare_methods import compare_prices
from multi_option.reporting.plots import (
    plot_method_comparison, plot_convergence, plot_greeks,
    plot_pinn_training, plot_pinn_surface, plot_pinn_residual_heatmap,
    plot_iv_surface, plot_smile_slices, plot_noarb_warnings
)
from multi_option.reporting.serialize import (
    writeui_bundle, export_results_csv, create_summary_report
)
from multi_option.reporting.tables import create_comparison_table


@click.command()
@click.option('--product', type=click.Choice([
    'european_call', 'european_put', 'american_put',
    'barrierup_out_call', 'asian_arith_call'
]), default='european_call', help='Option product type')
@click.option('--s0', type=float, default=100.0, help='Initial stock price')
@click.option('--K', '--strike', 'K', type=float, default=100.0, help='Strike price')
@click.option('--T', '--maturity', 'T', type=float, default=1.0, help='Time to maturity')
@click.option('--r', '--rate', 'r', type=float, default=0.02, help='Risk-free rate')
@click.option('--q', '--dividend', 'q', type=float, default=0.0, help='Dividend yield')
@click.option('--sigma', '--vol', 'sigma', type=float, default=0.2, help='Volatility')
@click.option('--s_max', type=float, default=400.0, help='Maximum stock price for grid')
@click.option('--ns', type=int, default=401, help='Number of spatial grid points')
@click.option('--nt', type=int, default=400, help='Number of time grid points')
@click.option('--mc_paths', type=int, default=200000, help='Number of MC paths')
@click.option('--mc_steps', type=int, default=252, help='Number of MC time steps')
@click.option('--barrier', type=float, default=None, help='Barrier level')
@click.option('--pinn_epochs', type=int, default=300, help='PINN training epochs')
@click.option('--pinn_lr', type=float, default=1e-3, help='PINN learning rate')
@click.option('--pinn_hidden', type=int, default=64, help='PINN hidden layer size')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--chain_csv', type=str, default=None, help='Option chain CSV path')
@click.option('--use_mid', type=bool, default=True, help='Use mid prices for IV')
@click.option('--out', '--out_dir', 'out_dir', type=str, default='./outputs', help='Output directory')
@click.option('--plots/--no-plots', default=True, help='Generate plots')
@click.option('--ui_json/--no-ui_json', default=True, help='Generate UI JSON bundle')
@click.option('--verbose', is_flag=True, help='Verbose output')
def main(
    product: str,
    s0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    s_max: float,
    ns: int,
    nt: int,
    mc_paths: int,
    mc_steps: int,
    barrier: Optional[float],
    pinn_epochs: int,
    pinn_lr: float,
    pinn_hidden: int,
    seed: int,
    chain_csv: Optional[str],
    use_mid: bool,
    out_dir: str,
    plots: bool,
    ui_json: bool,
    verbose: bool
):
    """Multi-Method Option Pricing Engine CLI.

    Price options using multiple numerical methods and compare results.
    """
    click.echo("=" * 60)
    click.echo("MULTI-METHOD OPTION PRICING ENGINE")
    click.echo("=" * 60)

    # Create configuration
    config_dict = {
        's0': s0, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma,
        's_max': s_max, 'ns': ns, 'nt': nt,
        'mc_paths': mc_paths, 'mc_steps': mc_steps,
        'barrier': barrier,
        'pinn_epochs': pinn_epochs, 'pinn_lr': pinn_lr, 'pinn_hidden': pinn_hidden,
        'seed': seed,
        'chain_csv': chain_csv, 'use_mid': use_mid,
        'out_dir': out_dir, 'plots': plots, 'ui_json': ui_json
    }

    try:
        cfg = parse_config(config_dict)
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo("\nConfiguration:")
        click.echo(f"  Product: {product}")
        click.echo(f"  S0=${s0:.2f}, K=${K:.2f}, T={T:.2f}")
        click.echo(f"  r={r:.4f}, q={q:.4f}, sigma={sigma:.4f}")
        click.echo(f"  Grid: {ns}x{nt}")
        click.echo(f"  MC: {mc_paths:,} paths, {mc_steps} steps")
        click.echo(f"  PINN: {pinn_epochs} epochs, hidden={pinn_hidden}")

    # Price option with different methods
    click.echo("\nPricing with different methods...")

    all_results = {}
    price_results = {}

    if product in ['european_call', 'european_put']:
        is_call = (product == 'european_call')
        results = price_european_all_methods(cfg, is_call)
        all_results[product] = results
        price_results = {k: v.price for k, v in results.items()}

    elif product == 'american_put':
        results = price_american_put_methods(cfg)
        all_results[product] = results
        price_results = {k: v.price for k, v in results.items()}

    elif product == 'barrierup_out_call':
        if barrier is None:
            click.echo("Warning: Barrier not specified, using 130% of spot", err=True)
            cfg = replace(cfg, barrier=s0 * 1.3)
        results = price_barrier_methods(cfg)
        all_results[product] = results
        price_results = {k: v.price for k, v in results.items()}

    else:
        click.echo(f"Product {product} not fully implemented yet", err=True)
        sys.exit(1)

    # Display results
    click.echo("\n" + "=" * 40)
    click.echo("PRICING RESULTS:")
    click.echo("-" * 40)
    for method, price in price_results.items():
        click.echo(f"{method.upper():10s}: ${price:10.4f}")

    # Compute Greeks (for European options)
    greeks = None
    if product in ['european_call', 'european_put']:
        click.echo("\nComputing Greeks...")
        is_call = (product == 'european_call')
        greeks = bs_greeks(s0, K, r, q, sigma, T, is_call)

        click.echo("\n" + "=" * 40)
        click.echo("GREEKS:")
        click.echo("-" * 40)
        click.echo(f"Delta:  {greeks.delta:10.6f}")
        click.echo(f"Gamma:  {greeks.gamma:10.6f}")
        click.echo(f"Theta:  {greeks.theta:10.6f}")
        click.echo(f"Vega:   {greeks.vega:10.6f}")
        click.echo(f"Rho:    {greeks.rho:10.6f}")

    # Convergence analysis
    convergence_df = pd.DataFrame()
    if verbose and 'bs' in price_results:
        click.echo("\nPerforming convergence analysis...")
        ref_price = price_results['bs']

        # Grid convergence for PDE
        if 'pde' in price_results:
            convergence_df = convergence_curve(
                'grid', product, 'pde', cfg, ref_price, n_points=5
            )

    # Implied volatility analysis (if chain provided)
    iv_surface_df = pd.DataFrame()
    smiles_dict = {}
    noarb_stats = {}

    if chain_csv and os.path.exists(chain_csv):
        click.echo("\nAnalyzing option chain for implied volatility...")
        from multi_option.market.chain_io import load_chain_csv, preprocess_chain
        from multi_option.market.iv import implied_vol_from_chain
        from multi_option.market.surface import iv_surface, smile_slices
        from multi_option.market.arbitrage import no_arbitrage_warnings

        try:
            # Load and process chain
            chain_df = load_chain_csv(chain_csv)
            chain_df = preprocess_chain(chain_df, s0, r, q)

            # Compute implied volatilities
            iv_df = implied_vol_from_chain(chain_df, use_mid, r, q, s0)

            # Build surface
            iv_pivot, iv_smooth = iv_surface(iv_df)
            iv_surface_df = iv_smooth

            # Extract smile slices
            maturities = [0.25, 0.5, 1.0] if T >= 1.0 else [T]
            smiles_dict = smile_slices(iv_df, maturities)

            # Check arbitrage
            noarb_stats = no_arbitrage_warnings(iv_df)

            click.echo(f"  Loaded {len(iv_df)} options")
            click.echo(f"  IV range: {iv_df['iv'].min():.2%} - {iv_df['iv'].max():.2%}")
            click.echo(f"  Arbitrage violations: {sum(noarb_stats.values())}")

        except Exception as e:
            click.echo(f"  Warning: IV analysis failed: {e}", err=True)

    # PINN analysis
    pinn_loss_history = []
    pinn_surface_df = pd.DataFrame()
    pinn_residual_df = pd.DataFrame()

    if 'pinn' in all_results.get(product, {}):
        click.echo("\nAnalyzing PINN results...")
        pinn_result = all_results[product]['pinn']

        # Extract loss history if available
        if 'loss_history' in pinn_result.meta:
            pinn_loss_history = pinn_result.meta['loss_history']

        # Generate surface for visualization
        try:
            from multi_option.pinns.train import create_pinn_solution_grid, compute_pinn_residuals

            # This would require the trained model, simplified for now
            click.echo("  PINN surface generation simplified for this run")
        except:
            pass

    # Generate plots
    if plots:
        click.echo("\nGenerating plots...")

        # Method comparison
        if price_results:
            plot_method_comparison(price_results, out_path)
            click.echo("  [OK] Method comparison plot")

        # Convergence
        if not convergence_df.empty:
            plot_convergence(convergence_df, 'Grid Size', out_path)
            click.echo("  [OK] Convergence plot")

        # Greeks
        if greeks:
            plot_greeks(greeks, out_path)
            click.echo("  [OK] Greeks plot")

        # PINN plots
        if pinn_loss_history:
            plot_pinn_training(pd.Series(pinn_loss_history), out_path)
            click.echo("  [OK] PINN training plot")

        # IV plots
        if not iv_surface_df.empty:
            plot_iv_surface(iv_surface_df, out_path)
            click.echo("  [OK] IV surface plot")

        if smiles_dict:
            plot_smile_slices(smiles_dict, out_path)
            click.echo("  [OK] IV smile plot")

        if noarb_stats:
            plot_noarb_warnings(noarb_stats, out_path)
            click.echo("  [OK] No-arbitrage warnings plot")

    # Export results
    click.echo("\nExporting results...")

    # CSV export
    export_dict = {
        'prices': pd.DataFrame([price_results]),
        'convergence': convergence_df
    }
    if greeks:
        export_dict['greeks'] = pd.DataFrame([{
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'theta': greeks.theta,
            'vega': greeks.vega,
            'rho': greeks.rho
        }])

    csv_paths = export_results_csv(export_dict, out_path)
    click.echo(f"  [OK] Exported {len(csv_paths)} CSV files")

    # Summary report
    summary_path = create_summary_report(cfg, {
        'prices': price_results,
        'greeks': greeks.__dict__ if greeks else {},
        'convergence': convergence_df
    }, out_path)
    click.echo(f"  [OK] Summary report: {summary_path}")

    # UI JSON bundle
    if ui_json:
        ui_path = writeui_bundle(
            out_path,
            meta={'config': config_dict, 'product': product},
            comparison=price_results,
            convergence=convergence_df,
            greeks=greeks.__dict__ if greeks else {},
            pinn_loss=pd.Series(pinn_loss_history) if pinn_loss_history else pd.Series(),
            pinn_surface=pinn_surface_df,
            iv_surface_df=iv_surface_df,
            smiles_example=smiles_dict
        )
        click.echo(f"  [OK] UI bundle: {ui_path}")

    click.echo("\n" + "=" * 60)
    click.echo("COMPLETE! Results saved to: " + str(out_path))
    click.echo("=" * 60)


if __name__ == '__main__':
    main()
