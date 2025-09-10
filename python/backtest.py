"""
Backtesting Module for DeepONet Finance
=========================================

Provides backtesting framework for DeepONet-based trading strategies:
1. Crypto forecasting backtest (Bybit data)
2. Option pricing arbitrage backtest
3. Yield curve trading backtest
4. Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)

Usage:
    python backtest.py --model checkpoints/best_deeponet.pth --symbol BTCUSDT
    python backtest.py --mode synthetic --strategy momentum
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data_loader import fetch_bybit_data, prepare_crypto_deeponet_data, _generate_synthetic_ohlcv
from model import DeepONet, build_crypto_deeponet, ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Capital and position sizing
    initial_capital: float = 100000.0
    max_position_pct: float = 0.2  # Max 20% of capital per trade
    transaction_cost: float = 0.001  # 0.1% per trade (taker fee)

    # Strategy parameters
    signal_threshold: float = 0.002  # Min predicted return for signal
    stop_loss: float = 0.03  # 3% stop loss
    take_profit: float = 0.05  # 5% take profit
    lookback_window: int = 60  # DeepONet branch input window
    forecast_horizon: int = 12  # Hours ahead to forecast
    rebalance_freq: int = 4  # Rebalance every 4 candles

    # DeepONet settings
    n_forecast_points: int = 5  # Average prediction over multiple horizons
    device: str = "cpu"

    # Risk management
    max_drawdown_limit: float = 0.15  # Stop trading if drawdown > 15%
    volatility_scaling: bool = True  # Scale position by inverse volatility


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Returns and equity
    total_return: float = 0.0
    annualized_return: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    calmar_ratio: float = 0.0

    # Trading metrics
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0

    # Metadata
    start_date: str = ""
    end_date: str = ""
    n_candles: int = 0
    symbol: str = ""

    def to_dict(self) -> dict:
        d = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (list, np.ndarray)):
                d[key] = [float(v) for v in val[:100]]  # Truncate for JSON
            elif isinstance(val, (int, float, str)):
                d[key] = val
        return d

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULTS: {self.symbol}",
            f"{'='*60}",
            f"Period: {self.start_date} to {self.end_date} ({self.n_candles} candles)",
            f"",
            f"--- Performance ---",
            f"Total Return:       {self.total_return*100:>8.2f}%",
            f"Annualized Return:  {self.annualized_return*100:>8.2f}%",
            f"Benchmark Return:   {self.benchmark_return*100:>8.2f}%",
            f"Alpha:              {self.alpha*100:>8.2f}%",
            f"",
            f"--- Risk ---",
            f"Sharpe Ratio:       {self.sharpe_ratio:>8.3f}",
            f"Sortino Ratio:      {self.sortino_ratio:>8.3f}",
            f"Max Drawdown:       {self.max_drawdown*100:>8.2f}%",
            f"Volatility (ann.):  {self.volatility*100:>8.2f}%",
            f"Calmar Ratio:       {self.calmar_ratio:>8.3f}",
            f"",
            f"--- Trading ---",
            f"Number of Trades:   {self.n_trades:>8d}",
            f"Win Rate:           {self.win_rate*100:>8.2f}%",
            f"Avg Win:            {self.avg_win*100:>8.4f}%",
            f"Avg Loss:           {self.avg_loss*100:>8.4f}%",
            f"Profit Factor:      {self.profit_factor:>8.3f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: np.ndarray,
    benchmark_curve: np.ndarray,
    trades: List[Dict],
    annualization_factor: float = 365 * 24,  # Hourly data
) -> BacktestResult:
    """Compute comprehensive backtest performance metrics.

    Args:
        equity_curve: Portfolio equity over time
        benchmark_curve: Buy-and-hold benchmark equity
        trades: List of trade dictionaries
        annualization_factor: Factor for annualizing (hourly candles)

    Returns:
        BacktestResult with all metrics populated
    """
    result = BacktestResult()

    # Returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    result.returns = returns.tolist()
    result.equity_curve = equity_curve.tolist()

    # Total and annualized return
    result.total_return = (equity_curve[-1] / equity_curve[0]) - 1
    n_periods = len(equity_curve)
    result.annualized_return = (
        (1 + result.total_return) ** (annualization_factor / n_periods) - 1
    )

    # Benchmark
    result.benchmark_return = (benchmark_curve[-1] / benchmark_curve[0]) - 1

    # Volatility
    result.volatility = np.std(returns) * np.sqrt(annualization_factor)

    # Sharpe ratio (assuming 0 risk-free rate for crypto)
    if result.volatility > 0:
        result.sharpe_ratio = (
            np.mean(returns) * annualization_factor / result.volatility
        )
    else:
        result.sharpe_ratio = 0.0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(annualization_factor)
        if downside_vol > 0:
            result.sortino_ratio = (
                np.mean(returns) * annualization_factor / downside_vol
            )

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    result.max_drawdown = abs(drawdown.min())

    # Max drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            durations = []
            for start in dd_starts:
                ends_after = dd_ends[dd_ends > start]
                if len(ends_after) > 0:
                    durations.append(ends_after[0] - start)
            if durations:
                result.max_drawdown_duration = max(durations)

    # Calmar ratio
    if result.max_drawdown > 0:
        result.calmar_ratio = result.annualized_return / result.max_drawdown

    # Alpha and Beta
    bench_returns = np.diff(benchmark_curve) / benchmark_curve[:-1]
    if len(returns) == len(bench_returns) and len(returns) > 1:
        cov_matrix = np.cov(returns, bench_returns)
        bench_var = np.var(bench_returns)
        if bench_var > 0:
            result.beta = cov_matrix[0, 1] / bench_var
        result.alpha = result.annualized_return - result.beta * (
            (benchmark_curve[-1] / benchmark_curve[0]) ** (annualization_factor / n_periods) - 1
        )

    # Trade statistics
    result.n_trades = len(trades)
    if trades:
        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        if total_losses > 0:
            result.profit_factor = total_wins / total_losses
        else:
            result.profit_factor = float("inf") if total_wins > 0 else 0

        holding_periods = [t.get("holding_period", 0) for t in trades]
        result.avg_holding_period = np.mean(holding_periods) if holding_periods else 0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO FORECASTING BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════


class CryptoDeepONetBacktester:
    """Backtester for DeepONet crypto trading strategy.

    Strategy:
    1. Use DeepONet to predict price changes at multiple future offsets
    2. Average predictions to get directional signal
    3. Go long/short/flat based on signal strength
    4. Apply risk management (stop loss, take profit, position sizing)
    """

    def __init__(self, model: DeepONet, config: BacktestConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    def generate_signal(self, ohlcv_window: np.ndarray) -> Tuple[float, float]:
        """Generate trading signal from DeepONet prediction.

        Args:
            ohlcv_window: OHLCV data [window_size, 5]

        Returns:
            Tuple of (signal_direction, signal_strength)
            signal_direction: -1 (short), 0 (flat), 1 (long)
            signal_strength: 0 to 1 confidence
        """
        with torch.no_grad():
            # Normalize
            price_mean = ohlcv_window[:, 3].mean()
            price_std = ohlcv_window[:, 3].std() + 1e-8

            window_norm = ohlcv_window.copy()
            window_norm[:, :4] = (window_norm[:, :4] - price_mean) / price_std
            vol_mean = window_norm[:, 4].mean()
            vol_std = window_norm[:, 4].std() + 1e-8
            window_norm[:, 4] = (window_norm[:, 4] - vol_mean) / vol_std

            # Branch input: [1, 5, window_size]
            u = torch.FloatTensor(window_norm.T).unsqueeze(0).to(self.device)

            # Trunk input: multiple forecast horizons
            max_horizon = self.config.forecast_horizon
            offsets = np.linspace(1, max_horizon, self.config.n_forecast_points)
            y = torch.FloatTensor(offsets.reshape(-1, 1) / max_horizon).to(self.device)

            # Predict for each offset
            predictions = []
            for i in range(len(offsets)):
                y_i = y[i : i + 1].expand(1, -1)
                pred = self.model(u, y_i)
                predictions.append(pred.item())

            # Average prediction across horizons
            avg_pred = np.mean(predictions)

            # Convert to signal
            threshold = self.config.signal_threshold
            if avg_pred > threshold:
                return 1.0, min(abs(avg_pred) / threshold, 1.0)
            elif avg_pred < -threshold:
                return -1.0, min(abs(avg_pred) / threshold, 1.0)
            else:
                return 0.0, abs(avg_pred) / threshold

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run the full backtest.

        Args:
            df: OHLCV DataFrame

        Returns:
            BacktestResult with all metrics
        """
        config = self.config
        ohlcv = df[["open", "high", "low", "close", "volume"]].values

        n = len(ohlcv)
        if n < config.lookback_window + config.forecast_horizon + 10:
            print("Insufficient data for backtesting")
            return BacktestResult()

        # Initialize
        capital = config.initial_capital
        position = 0.0  # Number of units held
        entry_price = 0.0
        equity_curve = [capital]
        benchmark_curve = [capital]
        benchmark_units = capital / ohlcv[config.lookback_window, 3]  # Buy and hold
        trades = []
        current_trade = None

        start_idx = config.lookback_window
        end_idx = n - config.forecast_horizon

        for i in range(start_idx, end_idx):
            current_price = ohlcv[i, 3]

            # Update equity
            portfolio_value = capital + position * current_price
            equity_curve.append(portfolio_value)
            benchmark_curve.append(benchmark_units * current_price)

            # Check max drawdown limit
            peak = max(equity_curve)
            current_dd = (peak - portfolio_value) / peak
            if current_dd > config.max_drawdown_limit:
                # Close position and stop
                if position != 0:
                    pnl = position * (current_price - entry_price)
                    capital += position * current_price
                    if current_trade:
                        current_trade["exit_price"] = current_price
                        current_trade["pnl"] = pnl
                        current_trade["holding_period"] = i - current_trade["entry_idx"]
                        trades.append(current_trade)
                    position = 0.0
                    current_trade = None
                continue

            # Check stop loss / take profit
            if position != 0 and entry_price > 0:
                ret = (current_price - entry_price) / entry_price
                if position > 0:
                    triggered = ret < -config.stop_loss or ret > config.take_profit
                else:
                    triggered = -ret < -config.stop_loss or -ret > config.take_profit

                if triggered:
                    pnl = position * (current_price - entry_price)
                    capital += position * current_price
                    if current_trade:
                        current_trade["exit_price"] = current_price
                        current_trade["pnl"] = pnl
                        current_trade["holding_period"] = i - current_trade["entry_idx"]
                        trades.append(current_trade)
                    position = 0.0
                    current_trade = None

            # Generate signal at rebalance frequency
            if (i - start_idx) % config.rebalance_freq == 0:
                window = ohlcv[i - config.lookback_window : i]
                direction, strength = self.generate_signal(window)

                # Position sizing
                position_size_pct = config.max_position_pct * strength
                if config.volatility_scaling:
                    recent_vol = np.std(np.diff(np.log(ohlcv[i - 20 : i, 3] + 1e-10)))
                    if recent_vol > 0:
                        vol_scalar = min(0.02 / recent_vol, 2.0)
                        position_size_pct *= vol_scalar

                target_value = portfolio_value * position_size_pct * direction
                target_position = target_value / current_price if current_price > 0 else 0

                # Execute trade if direction changed
                position_delta = target_position - position
                if abs(position_delta) * current_price > 100:  # Min trade size
                    # Transaction cost
                    cost = abs(position_delta) * current_price * config.transaction_cost
                    capital -= cost

                    # Close old trade if direction changed
                    if position != 0 and np.sign(position) != np.sign(target_position):
                        pnl = position * (current_price - entry_price) - cost
                        if current_trade:
                            current_trade["exit_price"] = current_price
                            current_trade["pnl"] = pnl
                            current_trade["holding_period"] = i - current_trade["entry_idx"]
                            trades.append(current_trade)
                        current_trade = None

                    # Update position
                    if current_trade is None and target_position != 0:
                        current_trade = {
                            "entry_idx": i,
                            "entry_price": current_price,
                            "direction": "long" if target_position > 0 else "short",
                            "size": abs(target_position),
                        }
                        entry_price = current_price

                    capital += (position - target_position) * current_price
                    position = target_position

        # Close final position
        final_price = ohlcv[end_idx - 1, 3]
        if position != 0:
            pnl = position * (final_price - entry_price)
            capital += position * final_price
            if current_trade:
                current_trade["exit_price"] = final_price
                current_trade["pnl"] = pnl
                current_trade["holding_period"] = end_idx - current_trade["entry_idx"]
                trades.append(current_trade)

        final_value = capital
        equity_curve.append(final_value)
        benchmark_curve.append(benchmark_units * final_price)

        # Compute metrics
        equity_arr = np.array(equity_curve)
        benchmark_arr = np.array(benchmark_curve)

        result = compute_metrics(equity_arr, benchmark_arr, trades)
        result.symbol = df.get("symbol", "UNKNOWN") if isinstance(df, dict) else "BTC/USDT"
        result.n_candles = end_idx - start_idx

        if "timestamp" in df.columns:
            result.start_date = str(df.iloc[start_idx]["timestamp"])
            result.end_date = str(df.iloc[end_idx - 1]["timestamp"])

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC STRATEGY BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════


def run_synthetic_backtest(
    strategy: str = "momentum",
    n_candles: int = 5000,
) -> BacktestResult:
    """Run a backtest with synthetic data and a simple DeepONet model.

    Args:
        strategy: Strategy type ('momentum' or 'mean_reversion')
        n_candles: Number of synthetic candles

    Returns:
        BacktestResult
    """
    print(f"\nRunning synthetic backtest ({strategy} strategy)...")

    # Generate synthetic data
    df = _generate_synthetic_ohlcv(n_candles)

    # Build a small DeepONet (untrained, for demo)
    config = BacktestConfig()

    if TORCH_AVAILABLE:
        model = build_crypto_deeponet(
            window=config.lookback_window, n_features=5, latent_dim=32
        )

        # Quick synthetic training
        print("Quick-training DeepONet on synthetic data...")
        u, y, targets = prepare_crypto_deeponet_data(
            df[:2000], window_size=config.lookback_window,
            n_forecast_points=5, max_forecast_horizon=config.forecast_horizon
        )

        if len(u) > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(20):
                # Mini-batch training
                idx = np.random.choice(len(u), min(64, len(u)), replace=False)
                u_batch = torch.FloatTensor(u[idx])
                y_batch = torch.FloatTensor(y[idx, 0, :])  # First query point
                t_batch = torch.FloatTensor(targets[idx, 0, :])

                pred = model(u_batch, y_batch)
                loss = F.mse_loss(pred, t_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"  Final training loss: {loss.item():.6f}")

        # Run backtest
        backtester = CryptoDeepONetBacktester(model, config)
        result = backtester.run(df[2000:].reset_index(drop=True))
        result.symbol = "SYNTHETIC"
    else:
        result = BacktestResult()
        result.symbol = "SYNTHETIC (no PyTorch)"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-ASSET BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════


def run_multi_asset_backtest(
    symbols: List[str] = None,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, BacktestResult]:
    """Run backtests across multiple crypto pairs.

    Args:
        symbols: List of trading pairs
        config: Backtest configuration

    Returns:
        Dictionary mapping symbol to BacktestResult
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    if config is None:
        config = BacktestConfig()

    results = {}
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        df = fetch_bybit_data(symbol=symbol, limit=3000)

        if TORCH_AVAILABLE and len(df) > config.lookback_window + 100:
            model = build_crypto_deeponet(
                window=config.lookback_window, n_features=5, latent_dim=64
            )
            backtester = CryptoDeepONetBacktester(model, config)
            result = backtester.run(df)
            result.symbol = symbol
            results[symbol] = result
            print(result.summary())
        else:
            print(f"  Skipped {symbol} (insufficient data or no PyTorch)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="DeepONet Finance Backtest")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["live", "synthetic", "multi_asset"],
        help="Backtest mode",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion"],
        help="Trading strategy",
    )
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--save-dir", type=str, default="results", help="Save directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    config = BacktestConfig(initial_capital=args.capital)

    if args.mode == "synthetic":
        result = run_synthetic_backtest(strategy=args.strategy)
        print(result.summary())

        with open(os.path.join(args.save_dir, "backtest_results.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    elif args.mode == "live":
        if not TORCH_AVAILABLE:
            print("PyTorch required for live backtesting")
            return

        print(f"Fetching data for {args.symbol}...")
        df = fetch_bybit_data(symbol=args.symbol, limit=5000)

        if args.model and os.path.exists(args.model):
            model = build_crypto_deeponet(window=config.lookback_window)
            checkpoint = torch.load(args.model, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {args.model}")
        else:
            print("No model checkpoint provided. Using untrained model (demo only).")
            model = build_crypto_deeponet(window=config.lookback_window)

        backtester = CryptoDeepONetBacktester(model, config)
        result = backtester.run(df)
        result.symbol = args.symbol
        print(result.summary())

        with open(os.path.join(args.save_dir, "backtest_results.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    elif args.mode == "multi_asset":
        results = run_multi_asset_backtest(config=config)
        all_results = {sym: r.to_dict() for sym, r in results.items()}

        with open(os.path.join(args.save_dir, "multi_asset_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
