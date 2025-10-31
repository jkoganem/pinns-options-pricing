"""Implied volatility surface construction and analysis.

This module implements methods for building and analyzing
implied volatility surfaces from option chain data.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Dict, Tuple, List, Optional


def iv_surface(
    df_iv: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build implied volatility surface from option chain.

    Args:
        df_iv: DataFrame with columns ['K', 'T', 'iv', 'moneyness'].

    Returns:
        Tuple of (pivot_table, smoothed_surface).
    """
    # Create pivot table
    pivot = df_iv.pivot_table(
        values='iv',
        index='K',
        columns='T',
        aggfunc='mean'
    )

    # Fill missing values with interpolation
    pivot = pivot.interpolate(method='linear', axis=0)
    pivot = pivot.interpolate(method='linear', axis=1)

    # Create smoothed surface using 2D interpolation
    smoothed = _smooth_surface(df_iv)

    return pivot, smoothed


def _smooth_surface(
    df: pd.DataFrame,
    n_strikes: int = 50,
    n_maturities: int = 20
) -> pd.DataFrame:
    """Create smoothed IV surface using 2D interpolation.

    Args:
        df: DataFrame with IV data.
        n_strikes: Number of strike points in smoothed surface.
        n_maturities: Number of maturity points in smoothed surface.

    Returns:
        Smoothed surface as DataFrame.
    """
    # Extract data points
    strikes = df['K'].values
    maturities = df['T'].values
    ivs = df['iv'].values

    # Create regular grid
    K_min, K_max = strikes.min(), strikes.max()
    T_min, T_max = maturities.min(), maturities.max()

    K_grid = np.linspace(K_min, K_max, n_strikes)
    T_grid = np.linspace(T_min, T_max, n_maturities)

    # 2D interpolation
    try:
        # Try RBF interpolation for smooth surface
        from scipy.interpolate import Rbf
        rbf = Rbf(strikes, maturities, ivs, function='thin_plate', smooth=0.1)

        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        IV_mesh = rbf(K_mesh.ravel(), T_mesh.ravel()).reshape(K_mesh.shape)

    except:
        # Fallback to griddata
        from scipy.interpolate import griddata
        points = np.column_stack([strikes, maturities])
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        grid_points = np.column_stack([K_mesh.ravel(), T_mesh.ravel()])

        IV_mesh = griddata(points, ivs, grid_points, method='cubic')
        IV_mesh = IV_mesh.reshape(K_mesh.shape)

    # Create DataFrame
    smoothed = pd.DataFrame(IV_mesh, index=T_grid, columns=K_grid)
    smoothed.index.name = 'T'
    smoothed.columns.name = 'K'

    return smoothed


def smile_slices(
    df_iv: pd.DataFrame,
    maturities: List[float]
) -> Dict[str, pd.DataFrame]:
    """Extract IV smile slices at specified maturities.

    Args:
        df_iv: DataFrame with IV data.
        maturities: List of target maturities.

    Returns:
        Dictionary mapping maturity string to smile DataFrame.
    """
    result = {}

    for T in maturities:
        # Find closest maturity in data
        uniqueT = df_iv['T'].unique()
        closestT = uniqueT[np.argmin(np.abs(uniqueT - T))]

        # Extract smile
        smile = df_iv[df_iv['T'] == closestT][['K', 'iv', 'moneyness']].copy()
        smile = smile.sort_values('K')

        # Store with formatted key
        key = f"T={T:.2f}"
        result[key] = smile

    return result


def compute_total_variance(
    df_iv: pd.DataFrame
) -> pd.DataFrame:
    """Compute total variance w = sigma2T.

    Args:
        df_iv: DataFrame with IV data.

    Returns:
        DataFrame with added 'total_var' column.
    """
    df = df_iv.copy()
    df['total_var'] = df['iv'] ** 2 * df['T']
    return df


def local_volatility_dupire(
    K: float,
    T: float,
    surface_func: callable,
    s0: float,
    r: float,
    q: float,
    dK: float = 0.1,
    dT: float = 0.001
) -> float:
    """Compute local volatility using Dupire's formula.

    sigma_loc2(K,T) = (dC/dT + (r-q)KdC/dK + qC) / (0.5K2d2C/dK2)

    Args:
        K: Strike price.
        T: Time to maturity.
        surface_func: Function that returns IV given (K, T).
        s0: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        dK: Strike bump for derivatives.
        dT: Time bump for derivatives.

    Returns:
        Local volatility.
    """
    from multi_option.bs_closed_form import bs_price

    # Get implied volatilities
    iv_0 = surface_func(K, T)
    ivT = surface_func(K, T + dT)
    iv_Kup = surface_func(K + dK, T)
    iv_K_down = surface_func(K - dK, T)
    iv_Kup2 = surface_func(K + 2*dK, T)
    iv_K_down2 = surface_func(K - 2*dK, T)

    # Compute call prices
    C_0 = bs_price(s0, K, r, q, iv_0, T, True)
    CT = bs_price(s0, K, r, q, ivT, T + dT, True)
    C_Kup = bs_price(s0, K + dK, r, q, iv_Kup, T, True)
    C_K_down = bs_price(s0, K - dK, r, q, iv_K_down, T, True)
    C_Kup2 = bs_price(s0, K + 2*dK, r, q, iv_Kup2, T, True)
    C_K_down2 = bs_price(s0, K - 2*dK, r, q, iv_K_down2, T, True)

    # Compute derivatives
    dC_dT = (CT - C_0) / dT
    dC_dK = (C_Kup - C_K_down) / (2 * dK)
    d2C_dK2 = (C_Kup - 2 * C_0 + C_K_down) / (dK ** 2)

    # Dupire formula
    numerator = dC_dT + (r - q) * K * dC_dK + q * C_0
    denominator = 0.5 * K ** 2 * d2C_dK2

    if denominator > 0:
        local_var = numerator / denominator
        if local_var > 0:
            return np.sqrt(local_var)

    # Return implied vol as fallback
    return iv_0


def fit_surface_parametric(
    df_iv: pd.DataFrame,
    model: str = 'ssvi'
) -> Dict:
    """Fit parametric model to IV surface.

    Args:
        df_iv: DataFrame with IV data.
        model: Model type ('ssvi', 'sabr', etc.).

    Returns:
        Dictionary with fitted parameters.
    """
    if model == 'ssvi':
        return _fit_ssvi(df_iv)
    elif model == 'polynomial':
        return _fit_polynomial(df_iv)
    else:
        raise ValueError(f"Unknown model: {model}")


def _fit_ssvi(df_iv: pd.DataFrame) -> Dict:
    """Fit SSVI (Surface SVI) model to IV surface.

    Simplified implementation of Gatheral's SSVI.
    """
    from scipy import optimize

    # Extract data
    moneyness = df_iv['log_moneyness'].values
    T = df_iv['T'].values
    total_var = df_iv['iv'].values ** 2 * T

    # SSVI parameterization (simplified)
    def ssvi(x, theta, rho, phi):
        k = x[:, 0]  # log-moneyness
        t = x[:, 1]  # time
        return theta * t * (1 + rho * phi * k + np.sqrt((phi * k) ** 2 + 1))

    # Initial guess
    p0 = [np.mean(total_var), 0.0, 0.5]

    # Fit
    X = np.column_stack([moneyness, T])
    popt, _ = optimize.curve_fit(ssvi, X, total_var, p0=p0)

    return {
        'model': 'ssvi',
        'theta': popt[0],
        'rho': popt[1],
        'phi': popt[2]
    }


def _fit_polynomial(df_iv: pd.DataFrame) -> Dict:
    """Fit polynomial surface to IV data."""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    # Features
    X = df_iv[['log_moneyness', 'T']].values
    y = df_iv['iv'].values

    # Create polynomial features
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Fit with regularization
    model = Ridge(alpha=0.1)
    model.fit(X_poly, y)

    return {
        'model': 'polynomial',
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_,
        'feature_names': poly.get_feature_names_out()
    }


def sticky_strike_delta(
    surface: pd.DataFrame,
    s0: float,
    ds: float = 1.0
) -> pd.DataFrame:
    """Compute sticky-strike surface delta.

    Measures how IV surface changes when spot moves.

    Args:
        surface: IV surface (strikes x maturities).
        s0: Current spot price.
        ds: Spot price bump.

    Returns:
        Surface delta.
    """
    # Shift strikes by spot move
    strikes = surface.columns
    shifted_strikes = strikes * (s0 + ds) / s0

    # Interpolate to get new IVs at original strikes
    delta_surface = surface.copy()

    for T in surface.index:
        iv_slice = surface.loc[T].values
        # Interpolate
        f = interpolate.interp1d(
            strikes, iv_slice,
            kind='linear',
            fill_value='extrapolate'
        )
        shifted_ivs = f(shifted_strikes)

        # Compute delta
        delta_surface.loc[T] = (shifted_ivs - iv_slice) / ds

    return delta_surface


def term_structure_at_strike(
    df_iv: pd.DataFrame,
    target_strike: float
) -> pd.DataFrame:
    """Extract term structure of IV at specific strike.

    Args:
        df_iv: DataFrame with IV data.
        target_strike: Target strike price.

    Returns:
        DataFrame with maturity and IV columns.
    """
    result = []

    for T in df_iv['T'].unique():
        slice_df = df_iv[df_iv['T'] == T]

        # Interpolate to target strike
        if len(slice_df) > 1:
            f = interpolate.interp1d(
                slice_df['K'].values,
                slice_df['iv'].values,
                kind='linear',
                fill_value='extrapolate'
            )
            iv = f(target_strike)
        else:
            iv = slice_df['iv'].iloc[0]

        result.append({'T': T, 'iv': iv})

    return pd.DataFrame(result).sort_values('T')