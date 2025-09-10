"""
DeepONet for Finance
====================

Deep Operator Networks (DeepONet) for financial applications including
option pricing, yield curve prediction, portfolio risk mapping, and
cryptocurrency trading with Bybit exchange data.

Modules:
    model       - DeepONet architecture variants (MLP, CNN, RNN branch nets)
    train       - Training pipeline with physics-informed and multi-fidelity support
    data_loader - Data loading for stocks (yfinance) and crypto (Bybit via ccxt)
    visualize   - Visualization utilities for operator learning results
    backtest    - Backtesting framework for DeepONet trading strategies
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"

from .model import (
    DeepONet,
    MLPBranch,
    CNNBranch,
    RNNBranch,
    TrunkNet,
    PIDeepONet,
    MultiFidelityDeepONet,
    RegimeAwareDeepONet,
    ModelConfig,
)

__all__ = [
    "DeepONet",
    "MLPBranch",
    "CNNBranch",
    "RNNBranch",
    "TrunkNet",
    "PIDeepONet",
    "MultiFidelityDeepONet",
    "RegimeAwareDeepONet",
    "ModelConfig",
]
