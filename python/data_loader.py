"""
Data Loading Module for DeepONet Finance
==========================================

Provides data loading utilities for:
1. Stock data (via yfinance)
2. Cryptocurrency data from Bybit (via ccxt)
3. Synthetic option pricing data (Black-Scholes / Heston)
4. Yield curve data (synthetic / Treasury)
5. DeepONet-formatted dataset creation

Each loader returns data formatted for DeepONet training:
- u_sensors: Input function values at sensor locations [n_samples, m]
- y_query: Query locations [n_samples, d]
- targets: True operator outputs [n_samples, 1]
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # General settings
    window_size: int = 60  # Historical window for branch input
    n_forecast_points: int = 20  # Number of query points for trunk
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Crypto settings
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 5000

    # Stock settings
    stock_symbols: List[str] = None
    stock_period: str = "2y"

    # Option pricing settings
    n_vol_surfaces: int = 5000
    n_vol_sensors: int = 200  # 20 strikes x 10 maturities
    n_strikes: int = 20
    n_maturities: int = 10

    # Yield curve settings
    n_macro_features: int = 10
    n_yield_maturities: int = 12

    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "SPY"]


# ═══════════════════════════════════════════════════════════════════════════════
# BYBIT / CRYPTO DATA
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_bybit_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 5000,
) -> pd.DataFrame:
    """Fetch OHLCV data from Bybit exchange via ccxt.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
        timeframe: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Number of candles to fetch

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if not CCXT_AVAILABLE:
        print("ccxt not available. Install with: pip install ccxt")
        print("Returning synthetic data instead.")
        return _generate_synthetic_ohlcv(limit)

    try:
        exchange = ccxt.bybit(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},
            }
        )

        all_ohlcv = []
        remaining = limit
        since = None

        while remaining > 0:
            batch_size = min(remaining, 1000)
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            remaining -= len(ohlcv)
            if len(ohlcv) < batch_size:
                break
            time.sleep(0.1)

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
        print(f"Fetched {len(df)} candles for {symbol} from Bybit")
        return df

    except Exception as e:
        print(f"Error fetching from Bybit: {e}")
        print("Returning synthetic data instead.")
        return _generate_synthetic_ohlcv(limit)


def _generate_synthetic_ohlcv(n_candles: int = 5000) -> pd.DataFrame:
    """Generate synthetic OHLCV data mimicking crypto behavior."""
    np.random.seed(42)

    # Geometric Brownian Motion with jumps
    dt = 1.0 / 24  # hourly
    mu = 0.0002
    sigma = 0.03
    jump_prob = 0.01
    jump_size_std = 0.05

    prices = [50000.0]  # Starting BTC-like price
    for _ in range(n_candles - 1):
        ret = mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        # Occasional jumps
        if np.random.rand() < jump_prob:
            ret += np.random.randn() * jump_size_std
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)

    # Generate OHLCV from close prices
    opens = prices * np.exp(np.random.randn(n_candles) * 0.002)
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.randn(n_candles) * 0.005))
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.randn(n_candles) * 0.005))
    volumes = np.random.lognormal(mean=15, sigma=1.5, size=n_candles)

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq="1h")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK DATA
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_stock_data(
    symbols: List[str] = None,
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch stock OHLCV data via yfinance.

    Args:
        symbols: List of stock tickers
        period: Data period (e.g., '1y', '2y', '5y')
        interval: Data interval ('1d', '1h', etc.)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "SPY"]

    if not YFINANCE_AVAILABLE:
        print("yfinance not available. Install with: pip install yfinance")
        print("Returning synthetic data instead.")
        return {sym: _generate_synthetic_ohlcv(500) for sym in symbols}

    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval)
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            data[sym] = df
            print(f"Fetched {len(df)} bars for {sym}")
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            data[sym] = _generate_synthetic_ohlcv(500)

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC OPTION PRICING DATA
# ═══════════════════════════════════════════════════════════════════════════════


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Call option price
    """
    from scipy.stats import norm

    if T <= 1e-10:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def generate_vol_surface(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """Generate a synthetic implied volatility surface.

    Uses a simplified Heston-inspired parametric model for the vol surface.

    Args:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-run variance
        xi: Vol-of-vol
        rho: Correlation
        strikes: Array of moneyness values (K/S)
        maturities: Array of maturities in years

    Returns:
        Volatility surface [n_strikes, n_maturities]
    """
    vol_surface = np.zeros((len(strikes), len(maturities)))
    base_vol = np.sqrt(v0)

    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            # Simplified parametric vol smile
            log_moneyness = np.log(K)
            term_structure = theta + (v0 - theta) * np.exp(-kappa * T)
            smile = xi * log_moneyness * (rho + 0.5 * xi * log_moneyness / np.sqrt(T + 0.01))
            vol_surface[i, j] = np.sqrt(max(term_structure + smile, 0.001))

    return vol_surface


def generate_option_pricing_data(
    n_samples: int = 5000,
    n_strikes: int = 20,
    n_maturities: int = 10,
    n_query_points: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired (vol surface, option price) data for DeepONet training.

    Args:
        n_samples: Number of volatility surface samples
        n_strikes: Number of strike grid points
        n_maturities: Number of maturity grid points
        n_query_points: Number of query points per sample
        seed: Random seed

    Returns:
        Tuple of (u_sensors, y_queries, targets)
        - u_sensors: Flattened vol surfaces [n_samples, n_strikes * n_maturities]
        - y_queries: Query locations (moneyness, T) [n_samples, n_query_points, 2]
        - targets: Option prices [n_samples, n_query_points, 1]
    """
    np.random.seed(seed)

    strikes = np.linspace(0.8, 1.2, n_strikes)
    maturities = np.linspace(0.1, 2.0, n_maturities)

    all_u = []
    all_y = []
    all_targets = []

    S = 1.0  # Normalized spot
    r = 0.02  # Risk-free rate

    for i in range(n_samples):
        # Random Heston-like parameters
        v0 = np.random.uniform(0.01, 0.09)
        kappa = np.random.uniform(0.5, 5.0)
        theta = np.random.uniform(0.01, 0.09)
        xi = np.random.uniform(0.1, 0.8)
        rho = np.random.uniform(-0.9, -0.1)

        # Generate vol surface (branch input)
        vol_surface = generate_vol_surface(v0, kappa, theta, xi, rho, strikes, maturities)
        u = vol_surface.flatten()

        # Random query points
        K_query = np.random.uniform(0.7, 1.3, n_query_points)
        T_query = np.random.uniform(0.05, 2.5, n_query_points)
        y = np.stack([K_query, T_query], axis=1)

        # Compute option prices at query points
        prices = np.zeros(n_query_points)
        for j in range(n_query_points):
            # Interpolate sigma from vol surface for this (K, T)
            k_idx = np.clip(
                np.searchsorted(strikes, K_query[j]) - 1, 0, n_strikes - 2
            )
            t_idx = np.clip(
                np.searchsorted(maturities, T_query[j]) - 1, 0, n_maturities - 2
            )
            sigma = vol_surface[k_idx, t_idx]
            prices[j] = black_scholes_call(S, K_query[j], T_query[j], r, sigma)

        all_u.append(u)
        all_y.append(y)
        all_targets.append(prices[:, np.newaxis])

    return np.array(all_u), np.array(all_y), np.array(all_targets)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC YIELD CURVE DATA
# ═══════════════════════════════════════════════════════════════════════════════


def generate_yield_curve_data(
    n_samples: int = 3000,
    n_macro_features: int = 10,
    seq_len: int = 60,
    n_query_maturities: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic macro -> yield curve data for DeepONet training.

    Args:
        n_samples: Number of macro scenarios
        n_macro_features: Number of economic indicators
        seq_len: Length of macro time series
        n_query_maturities: Number of maturity query points
        seed: Random seed

    Returns:
        Tuple of (u_macro, y_maturities, targets)
        - u_macro: Macro time series [n_samples, seq_len, n_features]
        - y_maturities: Maturity queries [n_samples, n_queries, 1]
        - targets: Yields [n_samples, n_queries, 1]
    """
    np.random.seed(seed)

    all_u = []
    all_y = []
    all_targets = []

    for _ in range(n_samples):
        # Generate correlated macro factors
        # Factor 1: level (related to short rate)
        level = np.random.uniform(0.01, 0.06)
        # Factor 2: slope
        slope = np.random.uniform(-0.01, 0.03)
        # Factor 3: curvature
        curvature = np.random.uniform(-0.02, 0.02)

        # Generate macro time series (auto-correlated)
        macro = np.zeros((seq_len, n_macro_features))
        for t in range(seq_len):
            noise = np.random.randn(n_macro_features) * 0.01
            if t > 0:
                macro[t] = 0.95 * macro[t - 1] + noise
            else:
                macro[t] = np.random.randn(n_macro_features) * 0.1
            # Embed yield curve factors in first 3 features
            macro[t, 0] += level
            macro[t, 1] += slope
            macro[t, 2] += curvature

        # Query maturities (in years)
        maturities = np.sort(np.random.uniform(0.25, 30.0, n_query_maturities))
        y = maturities.reshape(-1, 1)

        # Nelson-Siegel yield curve model
        tau = 2.0
        yields = (
            level
            + slope * (1 - np.exp(-maturities / tau)) / (maturities / tau)
            + curvature
            * (
                (1 - np.exp(-maturities / tau)) / (maturities / tau)
                - np.exp(-maturities / tau)
            )
        )
        yields += np.random.randn(n_query_maturities) * 0.001  # noise

        all_u.append(macro)
        all_y.append(y)
        all_targets.append(yields.reshape(-1, 1))

    return np.array(all_u), np.array(all_y), np.array(all_targets)


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO FORECASTING DATA
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_crypto_deeponet_data(
    df: pd.DataFrame,
    window_size: int = 60,
    n_forecast_points: int = 20,
    max_forecast_horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare crypto OHLCV data for DeepONet training.

    Branch input: Historical OHLCV window (normalized)
    Trunk input: Future time offset
    Target: Normalized price change at that offset

    Args:
        df: DataFrame with OHLCV columns
        window_size: Historical window for branch input
        n_forecast_points: Number of forecast query points per sample
        max_forecast_horizon: Maximum forecast horizon in candles

    Returns:
        Tuple of (u_sensors, y_queries, targets)
        - u_sensors: Normalized OHLCV windows [n_samples, 5, window_size]
        - y_queries: Time offsets [n_samples, n_forecast_points, 1]
        - targets: Normalized returns [n_samples, n_forecast_points, 1]
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    data = df[ohlcv_cols].values

    n = len(data)
    total_len = window_size + max_forecast_horizon

    all_u = []
    all_y = []
    all_targets = []

    for i in range(n - total_len):
        # Historical window
        window = data[i : i + window_size].copy()

        # Normalize by window's closing price mean
        price_mean = window[:, 3].mean()
        price_std = window[:, 3].std() + 1e-8

        # Normalize prices (OHLC)
        window[:, :4] = (window[:, :4] - price_mean) / price_std
        # Normalize volume separately
        vol_mean = window[:, 4].mean()
        vol_std = window[:, 4].std() + 1e-8
        window[:, 4] = (window[:, 4] - vol_mean) / vol_std

        # Transpose to [channels, seq_len]
        u = window.T  # [5, window_size]

        # Random forecast time offsets
        offsets = np.sort(
            np.random.uniform(1, max_forecast_horizon, n_forecast_points)
        )
        y = offsets.reshape(-1, 1) / max_forecast_horizon  # Normalize to [0, 1]

        # Future returns at each offset
        base_price = data[i + window_size - 1, 3]  # Last close in window
        targets = np.zeros((n_forecast_points, 1))
        for j, offset in enumerate(offsets):
            future_idx = i + window_size + int(offset) - 1
            if future_idx < n:
                future_price = data[future_idx, 3]
                targets[j, 0] = (future_price - base_price) / price_std
            else:
                targets[j, 0] = 0.0

        all_u.append(u)
        all_y.append(y)
        all_targets.append(targets)

    return np.array(all_u), np.array(all_y), np.array(all_targets)


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class DeepONetDataset(Dataset):
        """PyTorch Dataset for DeepONet training.

        Stores paired (input function, query location, target) data.

        Args:
            u_sensors: Input function values [n_samples, ...]
            y_queries: Query locations [n_samples, n_queries, d]
            targets: Target values [n_samples, n_queries, 1]
            single_query: If True, return individual (u, y, target) pairs
                         instead of batched queries
        """

        def __init__(
            self,
            u_sensors: np.ndarray,
            y_queries: np.ndarray,
            targets: np.ndarray,
            single_query: bool = False,
        ):
            self.u_sensors = torch.FloatTensor(u_sensors)
            self.y_queries = torch.FloatTensor(y_queries)
            self.targets = torch.FloatTensor(targets)
            self.single_query = single_query

            if single_query:
                # Flatten into individual (u, y, target) pairs
                n_samples, n_queries = y_queries.shape[0], y_queries.shape[1]
                self.u_flat = self.u_sensors.unsqueeze(1).expand(
                    -1, n_queries, *self.u_sensors.shape[1:]
                ).reshape(-1, *self.u_sensors.shape[1:])
                self.y_flat = self.y_queries.reshape(-1, y_queries.shape[-1])
                self.t_flat = self.targets.reshape(-1, 1)
                self._len = n_samples * n_queries
            else:
                self._len = len(u_sensors)

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            if self.single_query:
                return self.u_flat[idx], self.y_flat[idx], self.t_flat[idx]
            else:
                return self.u_sensors[idx], self.y_queries[idx], self.targets[idx]

    def create_dataloaders(
        u_sensors: np.ndarray,
        y_queries: np.ndarray,
        targets: np.ndarray,
        config: DataConfig,
        single_query: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders.

        Args:
            u_sensors: Input function values
            y_queries: Query locations
            targets: Target values
            config: Data configuration
            single_query: Whether to flatten to single queries

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        n = len(u_sensors)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        # Sequential split (important for time series)
        train_dataset = DeepONetDataset(
            u_sensors[:n_train],
            y_queries[:n_train],
            targets[:n_train],
            single_query=single_query,
        )
        val_dataset = DeepONetDataset(
            u_sensors[n_train : n_train + n_val],
            y_queries[n_train : n_train + n_val],
            targets[n_train : n_train + n_val],
            single_query=single_query,
        )
        test_dataset = DeepONetDataset(
            u_sensors[n_train + n_val :],
            y_queries[n_train + n_val :],
            targets[n_train + n_val :],
            single_query=single_query,
        )

        batch_size = 64

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
              f"Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class DeepONetNormalizer:
    """Normalize inputs and outputs for stable DeepONet training.

    Computes and stores mean/std statistics for branch inputs,
    trunk inputs, and targets. Supports fit/transform/inverse_transform.
    """

    def __init__(self):
        self.u_mean = None
        self.u_std = None
        self.y_mean = None
        self.y_std = None
        self.target_mean = None
        self.target_std = None

    def fit(
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        target_data: np.ndarray,
    ):
        """Compute normalization statistics.

        Args:
            u_data: Branch input data
            y_data: Trunk input data
            target_data: Target data
        """
        # Flatten if needed for statistics
        u_flat = u_data.reshape(-1, u_data.shape[-1]) if u_data.ndim > 2 else u_data
        y_flat = y_data.reshape(-1, y_data.shape[-1])
        t_flat = target_data.reshape(-1, target_data.shape[-1])

        self.u_mean = u_flat.mean(axis=0)
        self.u_std = u_flat.std(axis=0) + 1e-8
        self.y_mean = y_flat.mean(axis=0)
        self.y_std = y_flat.std(axis=0) + 1e-8
        self.target_mean = t_flat.mean()
        self.target_std = t_flat.std() + 1e-8

    def normalize_u(self, u: np.ndarray) -> np.ndarray:
        """Normalize branch input."""
        return (u - self.u_mean) / self.u_std

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize trunk input."""
        return (y - self.y_mean) / self.y_std

    def normalize_target(self, target: np.ndarray) -> np.ndarray:
        """Normalize target."""
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target_norm: np.ndarray) -> np.ndarray:
        """Denormalize target."""
        return target_norm * self.target_std + self.target_mean


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / DEMO
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("DeepONet Data Loading Module")
    print("=" * 50)

    # 1. Synthetic option pricing data
    print("\n1. Generating synthetic option pricing data...")
    u_opt, y_opt, t_opt = generate_option_pricing_data(
        n_samples=100, n_query_points=20
    )
    print(f"   u_sensors: {u_opt.shape}")
    print(f"   y_queries: {y_opt.shape}")
    print(f"   targets:   {t_opt.shape}")
    print(f"   Price range: [{t_opt.min():.4f}, {t_opt.max():.4f}]")

    # 2. Synthetic yield curve data
    print("\n2. Generating synthetic yield curve data...")
    u_yc, y_yc, t_yc = generate_yield_curve_data(n_samples=100)
    print(f"   u_macro:      {u_yc.shape}")
    print(f"   y_maturities: {y_yc.shape}")
    print(f"   targets:      {t_yc.shape}")
    print(f"   Yield range:  [{t_yc.min():.4f}, {t_yc.max():.4f}]")

    # 3. Crypto data
    print("\n3. Fetching crypto data...")
    df_crypto = fetch_bybit_data(limit=500)
    print(f"   Shape: {df_crypto.shape}")
    print(f"   Columns: {list(df_crypto.columns)}")

    # 4. Prepare crypto for DeepONet
    print("\n4. Preparing crypto data for DeepONet...")
    u_crypto, y_crypto, t_crypto = prepare_crypto_deeponet_data(
        df_crypto, window_size=30, n_forecast_points=10
    )
    print(f"   u_sensors: {u_crypto.shape}")
    print(f"   y_queries: {y_crypto.shape}")
    print(f"   targets:   {t_crypto.shape}")

    # 5. Normalization
    print("\n5. Testing normalization...")
    normalizer = DeepONetNormalizer()
    normalizer.fit(u_opt, y_opt, t_opt)
    u_norm = normalizer.normalize_u(u_opt)
    print(f"   u_norm mean: {u_norm.mean():.4f}, std: {u_norm.std():.4f}")

    if TORCH_AVAILABLE:
        # 6. Create PyTorch DataLoaders
        print("\n6. Creating PyTorch DataLoaders...")
        data_config = DataConfig()
        train_dl, val_dl, test_dl = create_dataloaders(
            u_opt, y_opt, t_opt, data_config
        )
        batch = next(iter(train_dl))
        print(f"   Batch shapes: u={batch[0].shape}, y={batch[1].shape}, t={batch[2].shape}")

    print("\nData loading module working correctly.")
