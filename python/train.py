"""
Training Pipeline for DeepONet Finance
========================================

Provides training pipelines for various DeepONet configurations:
1. Standard DeepONet training (option pricing, crypto forecasting)
2. Physics-Informed DeepONet (PI-DeepONet) with Black-Scholes PDE
3. Multi-Fidelity DeepONet (two-stage: low-fi then correction)
4. Regime-Aware DeepONet with market regime classification

Usage:
    python train.py --mode option_pricing --epochs 500
    python train.py --mode crypto --symbol BTCUSDT --epochs 200
    python train.py --mode yield_curve --epochs 300
    python train.py --mode physics_informed --epochs 500
    python train.py --mode multifidelity --epochs 500
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from model import (
    DeepONet,
    MLPBranch,
    CNNBranch,
    RNNBranch,
    TrunkNet,
    PIDeepONet,
    MultiFidelityDeepONet,
    RegimeAwareDeepONet,
    ModelConfig,
    build_deeponet,
    build_option_pricing_deeponet,
    build_crypto_deeponet,
    build_yield_curve_deeponet,
)
from data_loader import (
    DataConfig,
    fetch_bybit_data,
    generate_option_pricing_data,
    generate_yield_curve_data,
    prepare_crypto_deeponet_data,
    create_dataloaders,
    DeepONetNormalizer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class TrainingMetrics:
    """Track and log training metrics."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        epoch_time: float,
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def to_dict(self) -> dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "test_losses": self.test_losses,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD DEEPONET TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_deeponet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    save_dir: str = "checkpoints",
) -> TrainingMetrics:
    """Train a standard DeepONet model.

    Args:
        model: DeepONet model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        save_dir: Directory for saving checkpoints

    Returns:
        Training metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )

    early_stopping = EarlyStopping(patience=config.patience)
    metrics = TrainingMetrics()

    print(f"\nTraining DeepONet ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"Device: {device}, Epochs: {config.epochs}, LR: {config.learning_rate}")
    print("-" * 70)

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        n_batches = 0

        for u_batch, y_batch, target_batch in train_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            target_batch = target_batch.to(device)

            pred = model(u_batch, y_batch)
            loss = F.mse_loss(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for u_batch, y_batch, target_batch in val_loader:
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                target_batch = target_batch.to(device)

                pred = model(u_batch, y_batch)
                loss = F.mse_loss(pred, target_batch)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        epoch_time = time.time() - epoch_start

        # Update metrics
        current_lr = optimizer.param_groups[0]["lr"]
        metrics.update(epoch, train_loss, val_loss, current_lr, epoch_time)

        # Save best model
        if val_loss < metrics.best_val_loss + 1e-10:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(save_dir, "best_deeponet.pth"),
            )

        # Logging
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

        # Early stopping
        if early_stopping.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}. Best: {metrics.best_val_loss:.6f}")
            break

    print(f"\nBest validation loss: {metrics.best_val_loss:.6f} at epoch {metrics.best_epoch}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_pi_deeponet(
    model: PIDeepONet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    sigma: float = 0.2,
    r: float = 0.02,
    K: float = 1.0,
    save_dir: str = "checkpoints",
) -> TrainingMetrics:
    """Train a Physics-Informed DeepONet with Black-Scholes PDE residual.

    Args:
        model: PI-DeepONet model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        sigma: Volatility for PDE residual
        r: Risk-free rate
        K: Strike price
        save_dir: Checkpoint directory

    Returns:
        Training metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    early_stopping = EarlyStopping(patience=config.patience)
    metrics = TrainingMetrics()

    print(f"\nTraining PI-DeepONet (sigma={sigma}, r={r}, K={K})")
    print(f"Lambda_data={config.lambda_data}, Lambda_pde={config.lambda_pde}, "
          f"Lambda_bc={config.lambda_bc}")
    print("-" * 70)

    for epoch in range(config.epochs):
        epoch_start = time.time()

        model.train()
        total_loss_sum = 0.0
        data_loss_sum = 0.0
        pde_loss_sum = 0.0
        n_batches = 0

        for u_batch, y_batch, target_batch in train_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            target_batch = target_batch.to(device)

            # Generate collocation points for PDE
            n_colloc = config.n_collocation
            S_colloc = torch.rand(n_colloc, 1, device=device) * 2.0  # S in [0, 2]
            t_colloc = torch.rand(n_colloc, 1, device=device) * 1.0  # t in [0, 1]

            loss, loss_dict = model.compute_loss(
                u_batch,
                y_batch,
                target_batch,
                S_colloc,
                t_colloc,
                sigma,
                r,
                K=K,
                lambda_data=config.lambda_data,
                lambda_pde=config.lambda_pde,
                lambda_bc=config.lambda_bc,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss_dict["total"]
            data_loss_sum += loss_dict["data"]
            pde_loss_sum += loss_dict["pde"]
            n_batches += 1

        scheduler.step()

        avg_total = total_loss_sum / max(n_batches, 1)
        avg_data = data_loss_sum / max(n_batches, 1)
        avg_pde = pde_loss_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for u_batch, y_batch, target_batch in val_loader:
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                target_batch = target_batch.to(device)
                pred = model(u_batch, y_batch)
                val_loss += F.mse_loss(pred, target_batch).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        metrics.update(epoch, avg_total, val_loss, current_lr, epoch_time)

        if val_loss == metrics.best_val_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_pi_deeponet.pth"))

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(
                f"Epoch {epoch:4d} | Total: {avg_total:.6f} | "
                f"Data: {avg_data:.6f} | PDE: {avg_pde:.6f} | "
                f"Val: {val_loss:.6f} | Time: {epoch_time:.1f}s"
            )

        if early_stopping.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-FIDELITY TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_multifidelity_deeponet(
    model: MultiFidelityDeepONet,
    lf_train_loader: DataLoader,
    lf_val_loader: DataLoader,
    hf_train_loader: DataLoader,
    hf_val_loader: DataLoader,
    config: ModelConfig,
    save_dir: str = "checkpoints",
) -> TrainingMetrics:
    """Two-stage training for Multi-Fidelity DeepONet.

    Stage 1: Train low-fidelity model on abundant BS data
    Stage 2: Freeze LF, train correction on scarce Heston data

    Args:
        model: MultiFidelityDeepONet model
        lf_train_loader: Low-fidelity training data
        lf_val_loader: Low-fidelity validation data
        hf_train_loader: High-fidelity training data
        hf_val_loader: High-fidelity validation data
        config: Model configuration
        save_dir: Checkpoint directory

    Returns:
        Training metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(config.device)
    model = model.to(device)

    metrics = TrainingMetrics()

    # ─── Stage 1: Train low-fidelity model ───
    print("\n" + "=" * 70)
    print("Stage 1: Training low-fidelity model on BS data")
    print("=" * 70)

    optimizer_lf = torch.optim.AdamW(
        model.get_lf_parameters(), lr=config.learning_rate
    )
    scheduler_lf = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_lf, T_max=config.lf_epochs
    )

    for epoch in range(config.lf_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for u, y, target in lf_train_loader:
            u, y, target = u.to(device), y.to(device), target.to(device)
            pred = model.forward_lf(u, y)
            loss = F.mse_loss(pred, target)
            optimizer_lf.zero_grad()
            loss.backward()
            optimizer_lf.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler_lf.step()
        train_loss /= max(n_batches, 1)

        if epoch % 20 == 0:
            print(f"  LF Epoch {epoch:4d} | Train Loss: {train_loss:.6f}")

    # ─── Stage 2: Train correction model ───
    print("\n" + "=" * 70)
    print("Stage 2: Training correction model on Heston residuals")
    print("=" * 70)

    model.freeze_lf()

    optimizer_corr = torch.optim.AdamW(
        model.get_correction_parameters(), lr=config.learning_rate * 0.1
    )
    scheduler_corr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_corr, T_max=config.hf_epochs
    )

    for epoch in range(config.hf_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for u, y, target in hf_train_loader:
            u, y, target = u.to(device), y.to(device), target.to(device)
            pred = model(u, y)  # Full model (LF + correction)
            loss = F.mse_loss(pred, target)
            optimizer_corr.zero_grad()
            loss.backward()
            optimizer_corr.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler_corr.step()
        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for u, y, target in hf_val_loader:
                u, y, target = u.to(device), y.to(device), target.to(device)
                pred = model(u, y)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        current_lr = optimizer_corr.param_groups[0]["lr"]
        metrics.update(epoch, train_loss, val_loss, current_lr, 0.0)

        if epoch % 20 == 0:
            print(
                f"  HF Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

    torch.save(model.state_dict(), os.path.join(save_dir, "best_mf_deeponet.pth"))
    print(f"\nMulti-fidelity training complete. Alpha = {model.alpha.item():.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device

    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for u_batch, y_batch, target_batch in test_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(u_batch, y_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    # Relative errors
    nonzero_mask = np.abs(targets) > 1e-8
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((preds[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask])) * 100
    else:
        mape = float("nan")

    # R-squared
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    results = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "n_samples": len(preds),
    }

    print("\nEvaluation Results:")
    print(f"  MSE:   {mse:.6f}")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  MAE:   {mae:.6f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  R2:    {r2:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════


def train_option_pricing(config: ModelConfig):
    """Full training pipeline for option pricing DeepONet."""
    print("\n" + "=" * 70)
    print("Training DeepONet for Option Pricing")
    print("=" * 70)

    # Generate data
    print("\nGenerating option pricing data...")
    u, y, targets = generate_option_pricing_data(
        n_samples=2000, n_query_points=30
    )

    data_config = DataConfig()
    train_dl, val_dl, test_dl = create_dataloaders(u, y, targets, data_config)

    # Build model
    model = build_option_pricing_deeponet(
        n_vol_sensors=u.shape[-1], latent_dim=config.latent_dim, device=config.device
    )

    # Train
    metrics = train_deeponet(model, train_dl, val_dl, config)

    # Evaluate
    results = evaluate_model(model, test_dl, config.device)

    # Save results
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "option_pricing_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, metrics, results


def train_crypto_forecasting(config: ModelConfig, symbol: str = "BTC/USDT"):
    """Full training pipeline for crypto forecasting DeepONet."""
    print("\n" + "=" * 70)
    print(f"Training DeepONet for Crypto Forecasting ({symbol})")
    print("=" * 70)

    # Fetch data
    print("\nFetching crypto data...")
    df = fetch_bybit_data(symbol=symbol, timeframe="1h", limit=5000)

    # Prepare for DeepONet
    print("Preparing data for DeepONet...")
    u, y, targets = prepare_crypto_deeponet_data(
        df, window_size=60, n_forecast_points=20, max_forecast_horizon=24
    )

    data_config = DataConfig()
    train_dl, val_dl, test_dl = create_dataloaders(
        u, y, targets, data_config, single_query=True
    )

    # Build model
    model = build_crypto_deeponet(
        window=60, n_features=5, latent_dim=config.latent_dim, device=config.device
    )

    # Train
    metrics = train_deeponet(model, train_dl, val_dl, config)

    # Evaluate
    results = evaluate_model(model, test_dl, config.device)

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "crypto_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, metrics, results


def train_yield_curve(config: ModelConfig):
    """Full training pipeline for yield curve DeepONet."""
    print("\n" + "=" * 70)
    print("Training DeepONet for Yield Curve Prediction")
    print("=" * 70)

    # Generate data
    print("\nGenerating yield curve data...")
    u, y, targets = generate_yield_curve_data(n_samples=2000)

    data_config = DataConfig()
    train_dl, val_dl, test_dl = create_dataloaders(
        u, y, targets, data_config, single_query=True
    )

    # Build model
    model = build_yield_curve_deeponet(
        n_macro_features=10, seq_len=60, latent_dim=config.latent_dim, device=config.device
    )

    # Train
    metrics = train_deeponet(model, train_dl, val_dl, config)

    # Evaluate
    results = evaluate_model(model, test_dl, config.device)

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "yield_curve_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, metrics, results


def train_physics_informed(config: ModelConfig):
    """Full training pipeline for Physics-Informed DeepONet."""
    print("\n" + "=" * 70)
    print("Training Physics-Informed DeepONet (Black-Scholes PDE)")
    print("=" * 70)

    # Generate option pricing data
    u, y, targets = generate_option_pricing_data(
        n_samples=1000, n_query_points=30
    )

    data_config = DataConfig()
    train_dl, val_dl, test_dl = create_dataloaders(u, y, targets, data_config)

    # Build PI-DeepONet
    branch = MLPBranch(u.shape[-1], [256, 256, 128], config.latent_dim)
    trunk = TrunkNet(2, [128, 128, 64], config.latent_dim)
    model = PIDeepONet(branch, trunk)

    # Train with PDE constraints
    metrics = train_pi_deeponet(
        model, train_dl, val_dl, config,
        sigma=0.2, r=0.02, K=1.0,
    )

    # Evaluate
    results = evaluate_model(model, test_dl, config.device)
    return model, metrics, results


def train_multifidelity(config: ModelConfig):
    """Full training pipeline for Multi-Fidelity DeepONet."""
    print("\n" + "=" * 70)
    print("Training Multi-Fidelity DeepONet (BS + Heston)")
    print("=" * 70)

    # Generate low-fidelity data (Black-Scholes, abundant)
    print("\nGenerating low-fidelity (BS) data...")
    u_lf, y_lf, t_lf = generate_option_pricing_data(
        n_samples=3000, n_query_points=30, seed=42
    )

    # Generate high-fidelity data (simulated Heston, scarce)
    # Add Heston-like corrections to BS prices
    print("Generating high-fidelity (Heston-like) data...")
    u_hf, y_hf, t_hf = generate_option_pricing_data(
        n_samples=500, n_query_points=30, seed=123
    )
    # Add a stochastic volatility correction term
    np.random.seed(456)
    heston_correction = np.random.randn(*t_hf.shape) * 0.01 + 0.005
    t_hf = t_hf + np.abs(t_hf) * heston_correction

    data_config = DataConfig()
    lf_train, lf_val, _ = create_dataloaders(u_lf, y_lf, t_lf, data_config)
    hf_train, hf_val, hf_test = create_dataloaders(u_hf, y_hf, t_hf, data_config)

    # Build model
    model = MultiFidelityDeepONet(
        branch_dim=u_lf.shape[-1],
        trunk_dim=y_lf.shape[-1],
        latent_dim=config.latent_dim,
    )

    # Train
    metrics = train_multifidelity_deeponet(
        model, lf_train, lf_val, hf_train, hf_val, config
    )

    # Evaluate
    results = evaluate_model(model, hf_test, config.device)
    return model, metrics, results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepONet for Finance")
    parser.add_argument(
        "--mode",
        type=str,
        default="option_pricing",
        choices=["option_pricing", "crypto", "yield_curve", "physics_informed", "multifidelity"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Crypto symbol")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ModelConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience,
    )

    if not TORCH_AVAILABLE:
        print("PyTorch is required for training. Install with: pip install torch")
        return

    if args.mode == "option_pricing":
        model, metrics, results = train_option_pricing(config)
    elif args.mode == "crypto":
        model, metrics, results = train_crypto_forecasting(config, args.symbol)
    elif args.mode == "yield_curve":
        model, metrics, results = train_yield_curve(config)
    elif args.mode == "physics_informed":
        model, metrics, results = train_physics_informed(config)
    elif args.mode == "multifidelity":
        model, metrics, results = train_multifidelity(config)
    else:
        print(f"Unknown mode: {args.mode}")
        return

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.mode}_metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)

    print(f"\nTraining complete. Results saved to results/{args.mode}_*.json")


if __name__ == "__main__":
    main()
