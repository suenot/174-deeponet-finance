"""
Visualization Module for DeepONet Finance
===========================================

Provides visualization utilities for:
1. Training loss curves (data loss, PDE loss, boundary loss)
2. Operator predictions vs ground truth
3. Volatility surfaces and option price surfaces
4. Yield curve predictions
5. Crypto forecasting results
6. Regime classification probabilities
7. Latent space analysis (branch/trunk embeddings)

Usage:
    python visualize.py --results results/option_pricing_results.json
    python visualize.py --mode training_curves --metrics results/option_pricing_metrics.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns

    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_training_curves(
    metrics: Dict,
    title: str = "DeepONet Training Progress",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves.

    Args:
        metrics: Dictionary with 'train_losses', 'val_losses', 'learning_rates'
        title: Plot title
        save_path: Path to save figure (if None, shows interactively)
    """
    if not PLT_AVAILABLE:
        print("matplotlib required for visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    ax = axes[0]
    epochs = range(len(metrics["train_losses"]))
    ax.semilogy(epochs, metrics["train_losses"], label="Train", alpha=0.8)
    ax.semilogy(epochs, metrics["val_losses"], label="Validation", alpha=0.8)
    if metrics.get("best_epoch") is not None:
        ax.axvline(
            x=metrics["best_epoch"], color="r", linestyle="--", alpha=0.5, label="Best"
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate schedule
    ax = axes[1]
    if metrics.get("learning_rates"):
        ax.plot(epochs, metrics["learning_rates"], color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # Loss ratio (train/val)
    ax = axes[2]
    train = np.array(metrics["train_losses"])
    val = np.array(metrics["val_losses"])
    ratio = train / (val + 1e-10)
    ax.plot(epochs, ratio, color="purple", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train/Val Loss Ratio")
    ax.set_title("Overfitting Indicator")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pi_deeponet_losses(
    data_losses: List[float],
    pde_losses: List[float],
    bc_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
):
    """Plot Physics-Informed DeepONet loss components.

    Args:
        data_losses: Data fitting loss per epoch
        pde_losses: PDE residual loss per epoch
        bc_losses: Boundary condition loss per epoch (optional)
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(len(data_losses))
    ax.semilogy(epochs, data_losses, label="Data Loss", linewidth=2)
    ax.semilogy(epochs, pde_losses, label="PDE Residual Loss", linewidth=2)
    if bc_losses:
        ax.semilogy(epochs, bc_losses, label="Boundary Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("PI-DeepONet Loss Components", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION PRICING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_vol_surface(
    vol_surface: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    title: str = "Implied Volatility Surface",
    save_path: Optional[str] = None,
):
    """Plot a volatility surface as a 3D mesh and heatmap.

    Args:
        vol_surface: Volatility surface [n_strikes, n_maturities]
        strikes: Strike/moneyness values
        maturities: Maturity values in years
        title: Plot title
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig = plt.figure(figsize=(16, 6))

    # 3D surface
    ax1 = fig.add_subplot(121, projection="3d")
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    ax1.plot_surface(K_grid, T_grid, vol_surface, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("Strike (K/S)")
    ax1.set_ylabel("Maturity (years)")
    ax1.set_zlabel("Implied Vol")
    ax1.set_title(title)

    # Heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(
        vol_surface,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[maturities[0], maturities[-1], strikes[0], strikes[-1]],
    )
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Strike (K/S)")
    ax2.set_title(f"{title} (Heatmap)")
    plt.colorbar(im, ax=ax2, label="Implied Vol")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_option_price_comparison(
    true_prices: np.ndarray,
    pred_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    maturity_idx: int = 5,
    save_path: Optional[str] = None,
):
    """Plot true vs predicted option prices across strikes for a fixed maturity.

    Args:
        true_prices: True prices [n_strikes, n_maturities]
        pred_prices: Predicted prices [n_strikes, n_maturities]
        strikes: Strike values
        maturities: Maturity values
        maturity_idx: Which maturity to plot
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T = maturities[maturity_idx]

    # Price comparison
    ax = axes[0]
    ax.plot(strikes, true_prices[:, maturity_idx], "b-", linewidth=2, label="True")
    ax.plot(
        strikes, pred_prices[:, maturity_idx], "r--", linewidth=2, label="DeepONet"
    )
    ax.set_xlabel("Strike (K/S)")
    ax.set_ylabel("Option Price")
    ax.set_title(f"Option Prices (T = {T:.2f} years)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error
    ax = axes[1]
    error = np.abs(true_prices[:, maturity_idx] - pred_prices[:, maturity_idx])
    ax.bar(strikes, error, width=0.015, color="coral", alpha=0.8)
    ax.set_xlabel("Strike (K/S)")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"Pricing Error (T = {T:.2f} years)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# YIELD CURVE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_yield_curves(
    true_yields: np.ndarray,
    pred_yields: np.ndarray,
    maturities: np.ndarray,
    n_samples: int = 5,
    save_path: Optional[str] = None,
):
    """Plot true vs predicted yield curves for multiple scenarios.

    Args:
        true_yields: True yield curves [n_samples, n_maturities]
        pred_yields: Predicted yield curves [n_samples, n_maturities]
        maturities: Maturity values in years
        n_samples: Number of sample curves to plot
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    # Individual curves
    ax = axes[0]
    for i in range(min(n_samples, len(true_yields))):
        ax.plot(maturities, true_yields[i] * 100, "-", color=colors[i], alpha=0.8,
                label=f"True {i+1}")
        ax.plot(maturities, pred_yields[i] * 100, "--", color=colors[i], alpha=0.8,
                label=f"Pred {i+1}")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Yield Curve Predictions")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Error heatmap
    ax = axes[1]
    errors = np.abs(true_yields - pred_yields) * 100  # in bps
    if len(errors) > 20:
        errors = errors[:20]  # Show first 20 samples
    im = ax.imshow(errors, aspect="auto", cmap="Reds",
                   extent=[maturities[0], maturities[-1], 0, len(errors)])
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Sample")
    ax.set_title("Yield Curve Error (bps)")
    plt.colorbar(im, ax=ax, label="Absolute Error (bps)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO FORECASTING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_crypto_forecast(
    actual_prices: np.ndarray,
    pred_offsets: np.ndarray,
    pred_values: np.ndarray,
    window_size: int = 60,
    title: str = "DeepONet Crypto Forecast",
    save_path: Optional[str] = None,
):
    """Plot crypto price history with DeepONet forecast overlay.

    Args:
        actual_prices: Full price series
        pred_offsets: Prediction time offsets [n_forecasts]
        pred_values: Predicted price changes [n_forecasts]
        window_size: Historical window size
        title: Plot title
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Price history + forecast
    ax = axes[0]
    n_history = min(len(actual_prices), window_size + 50)
    ax.plot(range(n_history), actual_prices[:n_history], "b-", linewidth=1.5,
            label="Historical")

    # Overlay forecast
    base_price = actual_prices[window_size - 1]
    forecast_times = window_size + pred_offsets
    forecast_prices = base_price + pred_values

    ax.plot(forecast_times, forecast_prices, "r--o", markersize=4, linewidth=2,
            label="DeepONet Forecast")
    ax.axvline(x=window_size, color="gray", linestyle=":", alpha=0.7,
               label="Forecast Start")
    ax.set_xlabel("Time (candles)")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Prediction error (if actual future available)
    ax = axes[1]
    if len(actual_prices) > window_size + int(max(pred_offsets)):
        actual_future = []
        for offset in pred_offsets:
            idx = window_size + int(offset) - 1
            if idx < len(actual_prices):
                actual_future.append(actual_prices[idx])
            else:
                actual_future.append(np.nan)
        actual_future = np.array(actual_future)
        errors = forecast_prices - actual_future

        ax.bar(range(len(errors)), errors, color=["green" if e > 0 else "red" for e in errors],
               alpha=0.7)
        ax.set_xlabel("Forecast Point")
        ax.set_ylabel("Error (Price)")
        ax.set_title("Forecast Error by Horizon")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data for error analysis",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_regime_analysis(
    regime_probs: np.ndarray,
    prices: np.ndarray,
    regime_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Plot regime probabilities alongside price data.

    Args:
        regime_probs: Regime probabilities [n_samples, n_regimes]
        prices: Price data [n_samples]
        regime_names: Names for each regime
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    n_regimes = regime_probs.shape[1]
    if regime_names is None:
        regime_names = [f"Regime {i+1}" for i in range(n_regimes)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price with regime coloring
    ax = axes[0]
    dominant_regime = np.argmax(regime_probs, axis=1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))

    for i in range(n_regimes):
        mask = dominant_regime == i
        ax.scatter(np.where(mask)[0], prices[mask], c=[colors[i]], s=3,
                   label=regime_names[i], alpha=0.7)
    ax.plot(prices, "k-", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Price")
    ax.set_title("Price Series Colored by Dominant Regime")
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)

    # Regime probability stacked area
    ax = axes[1]
    x = range(len(regime_probs))
    ax.stackplot(x, regime_probs.T, labels=regime_names, colors=colors, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Regime Probability")
    ax.set_title("Regime Classification Over Time")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-FIDELITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def plot_multifidelity_comparison(
    bs_prices: np.ndarray,
    heston_prices: np.ndarray,
    lf_pred: np.ndarray,
    mf_pred: np.ndarray,
    strikes: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot multi-fidelity DeepONet results comparing BS, Heston, LF, and MF.

    Args:
        bs_prices: Black-Scholes prices
        heston_prices: Heston (true) prices
        lf_pred: Low-fidelity DeepONet predictions
        mf_pred: Multi-fidelity DeepONet predictions
        strikes: Strike values
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Price comparison
    ax = axes[0]
    ax.plot(strikes, bs_prices, "g-", linewidth=2, label="Black-Scholes", alpha=0.7)
    ax.plot(strikes, heston_prices, "b-", linewidth=2, label="Heston (true)")
    ax.plot(strikes, lf_pred, "r--", linewidth=2, label="LF DeepONet")
    ax.plot(strikes, mf_pred, "m-.", linewidth=2, label="MF DeepONet")
    ax.set_xlabel("Strike (K/S)")
    ax.set_ylabel("Option Price")
    ax.set_title("Multi-Fidelity Price Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error comparison
    ax = axes[1]
    lf_error = np.abs(lf_pred - heston_prices)
    mf_error = np.abs(mf_pred - heston_prices)
    bs_error = np.abs(bs_prices - heston_prices)

    x = np.arange(len(strikes))
    width = 0.25
    ax.bar(x - width, bs_error, width, label="BS Error", alpha=0.7, color="green")
    ax.bar(x, lf_error, width, label="LF Error", alpha=0.7, color="red")
    ax.bar(x + width, mf_error, width, label="MF Error", alpha=0.7, color="magenta")
    ax.set_xlabel("Strike Index")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Pricing Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SCATTER / PARITY PLOTS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_parity(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str = "DeepONet Prediction Parity",
    save_path: Optional[str] = None,
):
    """Plot prediction parity (true vs predicted scatter).

    Args:
        true_values: Ground truth values
        pred_values: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.scatter(true_values, pred_values, alpha=0.3, s=10, c="steelblue")

    # Perfect prediction line
    vmin = min(true_values.min(), pred_values.min())
    vmax = max(true_values.max(), pred_values.max())
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=2, label="Perfect")

    # Statistics
    mse = np.mean((true_values - pred_values) ** 2)
    r2 = 1 - np.sum((true_values - pred_values) ** 2) / (
        np.sum((true_values - true_values.mean()) ** 2) + 1e-10
    )

    ax.text(
        0.05, 0.95,
        f"MSE: {mse:.2e}\nR2: {r2:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("True Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_deeponet_architecture(save_path: Optional[str] = None):
    """Create a visual diagram of the DeepONet architecture.

    Args:
        save_path: Path to save figure
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Branch network
    branch_box = plt.Rectangle((0.5, 6), 3, 2.5, linewidth=2, edgecolor="blue",
                                facecolor="lightblue", alpha=0.5)
    ax.add_patch(branch_box)
    ax.text(2, 8, "Branch Net\n(MLP/CNN/RNN)", ha="center", va="center",
            fontsize=12, fontweight="bold")

    # Branch input
    ax.annotate("u(x_1),...,u(x_m)\n(Input Function)", xy=(2, 6), xytext=(2, 4.5),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"))

    # Trunk network
    trunk_box = plt.Rectangle((6.5, 6), 3, 2.5, linewidth=2, edgecolor="green",
                               facecolor="lightgreen", alpha=0.5)
    ax.add_patch(trunk_box)
    ax.text(8, 8, "Trunk Net\n(MLP)", ha="center", va="center",
            fontsize=12, fontweight="bold")

    # Trunk input
    ax.annotate("y = (K, T)\n(Query Location)", xy=(8, 6), xytext=(8, 4.5),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", lw=2, color="green"))

    # Branch output
    ax.text(2, 5.5, "[b_1, ..., b_p]", ha="center", fontsize=10, color="blue")

    # Trunk output
    ax.text(8, 5.5, "[t_1, ..., t_p]", ha="center", fontsize=10, color="green")

    # Dot product
    dot_circle = plt.Circle((5, 3), 0.5, linewidth=2, edgecolor="red",
                             facecolor="lightyellow")
    ax.add_patch(dot_circle)
    ax.text(5, 3, "dot", ha="center", va="center", fontsize=11, fontweight="bold")

    # Arrows to dot product
    ax.annotate("", xy=(4.5, 3), xytext=(2, 5),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"))
    ax.annotate("", xy=(5.5, 3), xytext=(8, 5),
                arrowprops=dict(arrowstyle="->", lw=2, color="green"))

    # Output
    ax.annotate("G(u)(y) = sum(b_k * t_k) + bias\n(Operator Output)",
                xy=(5, 2.5), xytext=(5, 1),
                fontsize=11, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=2, color="red"))

    ax.set_title("DeepONet Architecture", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / DEMO
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="DeepONet Finance Visualization")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--metrics", type=str, help="Path to metrics JSON file")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "training_curves", "architecture"],
        help="Visualization mode",
    )
    parser.add_argument("--save-dir", type=str, default="figures",
                       help="Directory for saving figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "architecture":
        plot_deeponet_architecture(
            save_path=os.path.join(args.save_dir, "architecture.png")
        )
        print("Architecture diagram saved.")

    elif args.mode == "training_curves" and args.metrics:
        with open(args.metrics, "r") as f:
            metrics = json.load(f)
        plot_training_curves(
            metrics,
            save_path=os.path.join(args.save_dir, "training_curves.png"),
        )
        print("Training curves saved.")

    elif args.mode == "demo":
        print("Running visualization demo with synthetic data...")

        # Demo training curves
        np.random.seed(42)
        n_epochs = 200
        train_losses = np.exp(-np.linspace(0, 3, n_epochs)) + np.random.randn(n_epochs) * 0.01
        val_losses = np.exp(-np.linspace(0, 2.5, n_epochs)) + np.random.randn(n_epochs) * 0.015
        metrics = {
            "train_losses": train_losses.tolist(),
            "val_losses": val_losses.tolist(),
            "learning_rates": (1e-3 * np.cos(np.linspace(0, 6, n_epochs)) * 0.5 + 0.5e-3).tolist(),
            "best_epoch": int(np.argmin(val_losses)),
        }
        plot_training_curves(
            metrics,
            title="Demo: DeepONet Training Progress",
            save_path=os.path.join(args.save_dir, "demo_training.png"),
        )

        # Demo parity plot
        true = np.random.randn(500) * 0.5 + 1.0
        pred = true + np.random.randn(500) * 0.05
        plot_parity(
            true, pred,
            title="Demo: Prediction Parity",
            save_path=os.path.join(args.save_dir, "demo_parity.png"),
        )

        # Demo vol surface
        strikes = np.linspace(0.8, 1.2, 20)
        maturities = np.linspace(0.1, 2.0, 10)
        K_g, T_g = np.meshgrid(strikes, maturities, indexing="ij")
        vol_surface = 0.2 + 0.1 * (K_g - 1.0) ** 2 + 0.02 * np.sqrt(T_g)
        plot_vol_surface(
            vol_surface, strikes, maturities,
            save_path=os.path.join(args.save_dir, "demo_vol_surface.png"),
        )

        # Demo architecture
        plot_deeponet_architecture(
            save_path=os.path.join(args.save_dir, "demo_architecture.png")
        )

        print(f"Demo figures saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
