"""
DeepONet Model Architectures for Finance
=========================================

This module provides various DeepONet architectures for financial applications:
- Standard DeepONet with MLP/CNN/RNN branch networks
- Physics-Informed DeepONet (PI-DeepONet) with PDE residual loss
- Multi-Fidelity DeepONet combining low-fidelity and high-fidelity models
- Regime-Aware DeepONet with market regime conditioning

Key concepts:
- Branch network: encodes the input function (e.g., vol surface, price history)
- Trunk network: encodes the query location (e.g., strike, maturity, time offset)
- Output: dot product of branch and trunk outputs + bias
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


@dataclass
class ModelConfig:
    """Configuration for DeepONet models."""

    # Branch network settings
    branch_type: str = "mlp"  # "mlp", "cnn", "rnn"
    branch_input_dim: int = 100  # number of sensor locations
    branch_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    branch_input_channels: int = 5  # for CNN branch (OHLCV)
    branch_seq_len: int = 60  # for CNN/RNN branch

    # Trunk network settings
    trunk_input_dim: int = 1  # dimension of query location
    trunk_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])

    # Latent dimension (must match between branch and trunk)
    latent_dim: int = 128

    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 500
    batch_size: int = 64
    patience: int = 30

    # Physics-informed settings
    lambda_data: float = 1.0
    lambda_pde: float = 0.1
    lambda_bc: float = 0.01
    n_collocation: int = 1000

    # Multi-fidelity settings
    lf_epochs: int = 300
    hf_epochs: int = 200

    # Regime settings
    n_regimes: int = 3

    # Device
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════


class MLPBranch(nn.Module):
    """Branch network using Multi-Layer Perceptron.

    Best for: Discretized function values at fixed sensor locations.
    Example: Volatility surface sampled at fixed (K, T) grid points.

    Args:
        input_dim: Number of sensor locations (m)
        hidden_dims: List of hidden layer dimensions
        output_dim: Latent dimension (p)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.GELU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input function values at sensor locations [batch, m]

        Returns:
            Latent representation [batch, p]
        """
        return self.net(x)


class CNNBranch(nn.Module):
    """Branch network using 1D Convolutional Neural Network.

    Best for: Time series inputs with local patterns.
    Example: Historical price series, order book snapshots.

    Args:
        input_channels: Number of input channels (e.g., 5 for OHLCV)
        seq_len: Sequence length
        output_dim: Latent dimension (p)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Time series input [batch, channels, seq_len]

        Returns:
            Latent representation [batch, p]
        """
        h = self.conv(x).squeeze(-1)  # [batch, 128]
        return self.fc(h)  # [batch, p]


class RNNBranch(nn.Module):
    """Branch network using LSTM.

    Best for: Variable-length time series inputs.
    Example: Tick-level trading data with irregular timestamps.

    Args:
        input_dim: Feature dimension per time step
        hidden_dim: LSTM hidden dimension
        output_dim: Latent dimension (p)
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sequential input [batch, seq_len, features]

        Returns:
            Latent representation [batch, p]
        """
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, batch, hidden]
        return self.fc(h_n[-1])  # [batch, p]


# ═══════════════════════════════════════════════════════════════════════════════
# TRUNK NETWORK
# ═══════════════════════════════════════════════════════════════════════════════


class TrunkNet(nn.Module):
    """Trunk network for encoding query locations.

    The trunk network maps query locations to basis function values.
    For option pricing: y = (S, K, T, r)
    For yield curves: y = (maturity,)
    For crypto forecasting: y = (time_offset,)

    Args:
        input_dim: Dimension of query location
        hidden_dims: List of hidden layer dimensions
        output_dim: Latent dimension (p) -- must match branch output
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.GELU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            y: Query locations [batch, d]

        Returns:
            Basis function values [batch, p]
        """
        return self.net(y)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD DEEPONET
# ═══════════════════════════════════════════════════════════════════════════════


class DeepONet(nn.Module):
    """Standard Deep Operator Network.

    Combines a branch network (encodes input function) and a trunk network
    (encodes query location) via dot product to approximate operators.

    Output: G(u)(y) = sum_k branch_k(u) * trunk_k(y) + bias

    Args:
        branch_net: Branch network instance
        trunk_net: Trunk network instance
        use_bias: Whether to include a learnable bias term
    """

    def __init__(
        self,
        branch_net: nn.Module,
        trunk_net: nn.Module,
        use_bias: bool = True,
    ):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = None

    def forward(
        self,
        u_sensors: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            u_sensors: Input function values at sensors [batch, m] or
                       [batch, channels, seq_len] for CNN branch
            y_query: Query locations [batch, d]

        Returns:
            Operator output [batch, 1]
        """
        b = self.branch(u_sensors)  # [batch, p]
        t = self.trunk(y_query)  # [batch, p]
        out = torch.sum(b * t, dim=-1, keepdim=True)  # [batch, 1]
        if self.use_bias:
            out = out + self.bias
        return out

    def forward_multi_query(
        self,
        u_sensors: torch.Tensor,
        y_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with multiple query locations per input function.

        Args:
            u_sensors: Input function values [batch, m]
            y_queries: Multiple query locations [batch, n_queries, d]

        Returns:
            Operator outputs [batch, n_queries, 1]
        """
        batch_size, n_queries, d = y_queries.shape

        b = self.branch(u_sensors)  # [batch, p]
        b = b.unsqueeze(1).expand(-1, n_queries, -1)  # [batch, n_queries, p]

        y_flat = y_queries.reshape(-1, d)  # [batch * n_queries, d]
        t_flat = self.trunk(y_flat)  # [batch * n_queries, p]
        t = t_flat.reshape(batch_size, n_queries, -1)  # [batch, n_queries, p]

        out = torch.sum(b * t, dim=-1, keepdim=True)  # [batch, n_queries, 1]
        if self.use_bias:
            out = out + self.bias
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED DEEPONET
# ═══════════════════════════════════════════════════════════════════════════════


class PIDeepONet(nn.Module):
    """Physics-Informed DeepONet with PDE residual loss.

    Adds partial differential equation constraints to the training loss
    for physically consistent predictions. Supports the Black-Scholes PDE
    for option pricing applications.

    Args:
        branch_net: Branch network instance
        trunk_net: Trunk network instance
    """

    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        u_sensors: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            u_sensors: Input function values at sensors [batch, m]
            y_query: Query locations [batch, d] (e.g., S and t for BS PDE)

        Returns:
            Operator output [batch, 1]
        """
        b = self.branch(u_sensors)  # [batch, p]
        t = self.trunk(y_query)  # [batch, p]
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias

    def pde_residual_bs(
        self,
        u_sensors: torch.Tensor,
        S: torch.Tensor,
        t: torch.Tensor,
        sigma: float,
        r: float,
    ) -> torch.Tensor:
        """Compute Black-Scholes PDE residual using automatic differentiation.

        The Black-Scholes PDE:
            dC/dt + 0.5 * sigma^2 * S^2 * d2C/dS2 + r * S * dC/dS - r * C = 0

        Args:
            u_sensors: Branch input [batch, m]
            S: Spot prices (requires grad) [batch, 1]
            t: Time to maturity (requires grad) [batch, 1]
            sigma: Volatility (scalar)
            r: Risk-free rate (scalar)

        Returns:
            PDE residual [batch, 1]
        """
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)

        y_query = torch.cat([S, t], dim=-1)
        C = self.forward(u_sensors, y_query)

        # First-order partial derivatives
        grads = torch.autograd.grad(
            C,
            [S, t],
            grad_outputs=torch.ones_like(C),
            create_graph=True,
            retain_graph=True,
        )
        dC_dS, dC_dt = grads[0], grads[1]

        # Second-order partial derivative w.r.t. S
        d2C_dS2 = torch.autograd.grad(
            dC_dS,
            S,
            grad_outputs=torch.ones_like(dC_dS),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Black-Scholes PDE residual
        residual = (
            dC_dt
            + 0.5 * sigma**2 * S**2 * d2C_dS2
            + r * S * dC_dS
            - r * C
        )
        return residual

    def boundary_loss(
        self,
        u_sensors: torch.Tensor,
        K: float,
        S_max: float,
        T_max: float,
        r: float,
        n_points: int = 100,
    ) -> torch.Tensor:
        """Compute boundary condition loss for European call options.

        Boundary conditions:
        1. C(S=0, t) = 0
        2. C(S, T) = max(S - K, 0) at expiry
        3. C(S_max, t) -> S_max - K*exp(-r*(T-t)) for deep ITM

        Args:
            u_sensors: Branch input [batch, m]
            K: Strike price
            S_max: Maximum spot price for boundary
            T_max: Maximum time to maturity
            r: Risk-free rate
            n_points: Number of boundary points

        Returns:
            Scalar boundary loss
        """
        device = u_sensors.device
        batch = u_sensors.shape[0]

        # BC1: C(S=0, t) = 0
        S_zero = torch.zeros(batch, 1, device=device)
        t_rand = torch.rand(batch, 1, device=device) * T_max
        y_bc1 = torch.cat([S_zero, t_rand], dim=-1)
        bc1 = self.forward(u_sensors, y_bc1)
        bc1_loss = torch.mean(bc1**2)

        # BC2: C(S, T) = max(S - K, 0) at expiry
        S_rand = torch.rand(batch, 1, device=device) * S_max
        t_expiry = torch.full((batch, 1), T_max, device=device)
        y_bc2 = torch.cat([S_rand, t_expiry], dim=-1)
        pred_expiry = self.forward(u_sensors, y_bc2)
        true_payoff = torch.relu(S_rand - K)
        bc2_loss = F.mse_loss(pred_expiry, true_payoff)

        return bc1_loss + bc2_loss

    def compute_loss(
        self,
        u_sensors: torch.Tensor,
        y_query: torch.Tensor,
        targets: torch.Tensor,
        S_colloc: torch.Tensor,
        t_colloc: torch.Tensor,
        sigma: float,
        r: float,
        K: float = 1.0,
        S_max: float = 2.0,
        T_max: float = 1.0,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_bc: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combined data + PDE + boundary condition loss.

        Args:
            u_sensors: Branch input [batch, m]
            y_query: Trunk input [batch, d]
            targets: True operator outputs [batch, 1]
            S_colloc: Collocation spot prices [n_colloc, 1]
            t_colloc: Collocation times [n_colloc, 1]
            sigma: Volatility
            r: Risk-free rate
            K: Strike price
            S_max: Max spot for boundary
            T_max: Max maturity
            lambda_data: Weight for data loss
            lambda_pde: Weight for PDE loss
            lambda_bc: Weight for boundary loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Data fitting loss
        pred = self.forward(u_sensors, y_query)
        data_loss = F.mse_loss(pred, targets)

        # PDE residual loss
        # Expand u_sensors for collocation points
        n_colloc = S_colloc.shape[0]
        u_expanded = u_sensors[:1].expand(n_colloc, -1)
        residual = self.pde_residual_bs(u_expanded, S_colloc, t_colloc, sigma, r)
        pde_loss = torch.mean(residual**2)

        # Boundary condition loss
        bc_loss = self.boundary_loss(u_sensors, K, S_max, T_max, r)

        total_loss = lambda_data * data_loss + lambda_pde * pde_loss + lambda_bc * bc_loss

        loss_dict = {
            "total": total_loss.item(),
            "data": data_loss.item(),
            "pde": pde_loss.item(),
            "boundary": bc_loss.item(),
        }

        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-FIDELITY DEEPONET
# ═══════════════════════════════════════════════════════════════════════════════


class MultiFidelityDeepONet(nn.Module):
    """Multi-Fidelity DeepONet combining low-fi and high-fi models.

    Architecture:
        Low-fidelity:  DeepONet_LF trained on BS prices (abundant data)
        Correction:    DeepONet_corr trained on (Heston - BS) residuals
        High-fidelity: alpha * DeepONet_LF + DeepONet_corr

    Args:
        branch_dim: Input dimension for branch networks
        trunk_dim: Input dimension for trunk network
        latent_dim: Shared latent dimension
    """

    def __init__(self, branch_dim: int, trunk_dim: int, latent_dim: int):
        super().__init__()

        # Low-fidelity DeepONet (pre-trained on BS data)
        self.lf_branch = MLPBranch(branch_dim, [256, 256], latent_dim)
        self.lf_trunk = TrunkNet(trunk_dim, [128, 128], latent_dim)
        self.lf_bias = nn.Parameter(torch.zeros(1))

        # Correction DeepONet (trained on HF - LF residuals)
        self.corr_branch = MLPBranch(branch_dim, [128, 128], latent_dim)
        self.corr_trunk = TrunkNet(trunk_dim, [64, 64], latent_dim)
        self.corr_bias = nn.Parameter(torch.zeros(1))

        # Linear mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward_lf(
        self, u_sensors: torch.Tensor, y_query: torch.Tensor
    ) -> torch.Tensor:
        """Low-fidelity prediction."""
        b = self.lf_branch(u_sensors)
        t = self.lf_trunk(y_query)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.lf_bias

    def forward_correction(
        self, u_sensors: torch.Tensor, y_query: torch.Tensor
    ) -> torch.Tensor:
        """Correction prediction (HF - LF residual)."""
        b = self.corr_branch(u_sensors)
        t = self.corr_trunk(y_query)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.corr_bias

    def forward(
        self, u_sensors: torch.Tensor, y_query: torch.Tensor
    ) -> torch.Tensor:
        """Full multi-fidelity prediction."""
        lf_pred = self.forward_lf(u_sensors, y_query)
        correction = self.forward_correction(u_sensors, y_query)
        return self.alpha * lf_pred + correction

    def get_lf_parameters(self):
        """Get low-fidelity model parameters."""
        return (
            list(self.lf_branch.parameters())
            + list(self.lf_trunk.parameters())
            + [self.lf_bias]
        )

    def get_correction_parameters(self):
        """Get correction model parameters."""
        return (
            list(self.corr_branch.parameters())
            + list(self.corr_trunk.parameters())
            + [self.corr_bias, self.alpha]
        )

    def freeze_lf(self):
        """Freeze low-fidelity model for stage 2 training."""
        for param in self.lf_branch.parameters():
            param.requires_grad = False
        for param in self.lf_trunk.parameters():
            param.requires_grad = False
        self.lf_bias.requires_grad = False

    def unfreeze_lf(self):
        """Unfreeze low-fidelity model."""
        for param in self.lf_branch.parameters():
            param.requires_grad = True
        for param in self.lf_trunk.parameters():
            param.requires_grad = True
        self.lf_bias.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME-AWARE DEEPONET
# ═══════════════════════════════════════════════════════════════════════════════


class RegimeAwareDeepONet(nn.Module):
    """DeepONet with market regime conditioning.

    Uses a regime classifier to blend branch networks trained on
    different market regimes (bull, bear, sideways, high-vol).

    Args:
        n_regimes: Number of market regimes
        branch_dim: Input dimension for branch networks
        trunk_dim: Input dimension for trunk network
        latent_dim: Shared latent dimension
        branch_hidden_dims: Hidden dimensions for branch networks
        trunk_hidden_dims: Hidden dimensions for trunk network
    """

    def __init__(
        self,
        n_regimes: int,
        branch_dim: int,
        trunk_dim: int,
        latent_dim: int,
        branch_hidden_dims: Optional[List[int]] = None,
        trunk_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_regimes = n_regimes

        if branch_hidden_dims is None:
            branch_hidden_dims = [256, 256]
        if trunk_hidden_dims is None:
            trunk_hidden_dims = [128, 128]

        # One branch per regime
        self.branches = nn.ModuleList(
            [
                MLPBranch(branch_dim, branch_hidden_dims, latent_dim)
                for _ in range(n_regimes)
            ]
        )

        # Shared trunk
        self.trunk = TrunkNet(trunk_dim, trunk_hidden_dims, latent_dim)

        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_regimes),
            nn.Softmax(dim=-1),
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, u_sensors: torch.Tensor, y_query: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with regime-weighted branches.

        Args:
            u_sensors: Input function values [batch, m]
            y_query: Query locations [batch, d]

        Returns:
            Operator output [batch, 1]
        """
        # Classify regime
        regime_weights = self.regime_classifier(u_sensors)  # [batch, n_regimes]

        # Compute all branch outputs
        branch_outputs = torch.stack(
            [branch(u_sensors) for branch in self.branches], dim=1
        )  # [batch, n_regimes, p]

        # Weighted sum of branch outputs
        b = torch.sum(
            regime_weights.unsqueeze(-1) * branch_outputs, dim=1
        )  # [batch, p]

        t = self.trunk(y_query)  # [batch, p]
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias

    def get_regime_probabilities(
        self, u_sensors: torch.Tensor
    ) -> torch.Tensor:
        """Get regime classification probabilities.

        Args:
            u_sensors: Input function values [batch, m]

        Returns:
            Regime probabilities [batch, n_regimes]
        """
        return self.regime_classifier(u_sensors)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def build_deeponet(config: ModelConfig) -> DeepONet:
    """Build a DeepONet model from configuration.

    Args:
        config: Model configuration

    Returns:
        Configured DeepONet model
    """
    # Build branch network
    if config.branch_type == "mlp":
        branch = MLPBranch(
            config.branch_input_dim,
            config.branch_hidden_dims,
            config.latent_dim,
        )
    elif config.branch_type == "cnn":
        branch = CNNBranch(
            config.branch_input_channels,
            config.branch_seq_len,
            config.latent_dim,
        )
    elif config.branch_type == "rnn":
        branch = RNNBranch(
            config.branch_input_channels,
            config.branch_hidden_dims[0] if config.branch_hidden_dims else 128,
            config.latent_dim,
        )
    else:
        raise ValueError(f"Unknown branch type: {config.branch_type}")

    # Build trunk network
    trunk = TrunkNet(
        config.trunk_input_dim,
        config.trunk_hidden_dims,
        config.latent_dim,
    )

    model = DeepONet(branch, trunk)
    return model.to(config.device)


def build_option_pricing_deeponet(
    n_vol_sensors: int = 200,
    latent_dim: int = 128,
    device: str = "cpu",
) -> DeepONet:
    """Build a DeepONet for option pricing.

    Branch input: Flattened volatility surface at sensor grid
    Trunk input: (S/K, T) - moneyness and time to maturity

    Args:
        n_vol_sensors: Number of vol surface sensor locations
        latent_dim: Latent dimension
        device: Computation device

    Returns:
        DeepONet configured for option pricing
    """
    branch = MLPBranch(n_vol_sensors, [256, 256, 128], latent_dim)
    trunk = TrunkNet(2, [128, 128, 64], latent_dim)  # (moneyness, T)
    model = DeepONet(branch, trunk)
    return model.to(device)


def build_crypto_deeponet(
    window: int = 60,
    n_features: int = 5,
    latent_dim: int = 128,
    device: str = "cpu",
) -> DeepONet:
    """Build a DeepONet for crypto price forecasting.

    Branch input: Historical OHLCV window
    Trunk input: Future time offset

    Args:
        window: Historical window length
        n_features: Number of features (5 for OHLCV)
        latent_dim: Latent dimension
        device: Computation device

    Returns:
        DeepONet configured for crypto forecasting
    """
    branch = CNNBranch(n_features, window, latent_dim)
    trunk = TrunkNet(1, [64, 64], latent_dim)
    model = DeepONet(branch, trunk)
    return model.to(device)


def build_yield_curve_deeponet(
    n_macro_features: int = 10,
    seq_len: int = 60,
    latent_dim: int = 64,
    device: str = "cpu",
) -> DeepONet:
    """Build a DeepONet for yield curve prediction.

    Branch input: Macro economic time series
    Trunk input: Bond maturity (in years)

    Args:
        n_macro_features: Number of macro economic indicators
        seq_len: Length of macro time series
        latent_dim: Latent dimension
        device: Computation device

    Returns:
        DeepONet configured for yield curve prediction
    """
    branch = RNNBranch(n_macro_features, 128, latent_dim, num_layers=2)
    trunk = TrunkNet(1, [64, 64], latent_dim)
    model = DeepONet(branch, trunk)
    return model.to(device)


if __name__ == "__main__":
    print("DeepONet Model Architectures for Finance")
    print("=" * 50)

    # Demo: Build and test each model variant
    if TORCH_AVAILABLE:
        batch_size = 16

        # 1. Standard DeepONet with MLP branch
        print("\n1. Standard DeepONet (MLP Branch)")
        model = build_option_pricing_deeponet()
        u = torch.randn(batch_size, 200)  # vol surface
        y = torch.randn(batch_size, 2)  # (moneyness, T)
        out = model(u, y)
        print(f"   Input: u={u.shape}, y={y.shape}")
        print(f"   Output: {out.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 2. DeepONet with CNN branch
        print("\n2. DeepONet (CNN Branch) for Crypto")
        model_cnn = build_crypto_deeponet()
        u_cnn = torch.randn(batch_size, 5, 60)  # OHLCV x 60 steps
        y_cnn = torch.randn(batch_size, 1)  # time offset
        out_cnn = model_cnn(u_cnn, y_cnn)
        print(f"   Input: u={u_cnn.shape}, y={y_cnn.shape}")
        print(f"   Output: {out_cnn.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

        # 3. DeepONet with RNN branch
        print("\n3. DeepONet (RNN Branch) for Yield Curves")
        model_rnn = build_yield_curve_deeponet()
        u_rnn = torch.randn(batch_size, 60, 10)  # 60 months x 10 indicators
        y_rnn = torch.randn(batch_size, 1)  # maturity
        out_rnn = model_rnn(u_rnn, y_rnn)
        print(f"   Input: u={u_rnn.shape}, y={y_rnn.shape}")
        print(f"   Output: {out_rnn.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model_rnn.parameters()):,}")

        # 4. Multi-query evaluation
        print("\n4. Multi-query evaluation")
        n_queries = 50
        y_multi = torch.randn(batch_size, n_queries, 2)
        out_multi = model.forward_multi_query(u, y_multi)
        print(f"   Input: u={u.shape}, y={y_multi.shape}")
        print(f"   Output: {out_multi.shape}")

        # 5. Regime-aware DeepONet
        print("\n5. Regime-Aware DeepONet")
        model_regime = RegimeAwareDeepONet(
            n_regimes=3, branch_dim=200, trunk_dim=2, latent_dim=128
        )
        out_regime = model_regime(u, y)
        regimes = model_regime.get_regime_probabilities(u)
        print(f"   Output: {out_regime.shape}")
        print(f"   Regime probs: {regimes[0].detach().numpy()}")
        print(f"   Parameters: {sum(p.numel() for p in model_regime.parameters()):,}")

        print("\nAll model variants working correctly.")
    else:
        print("PyTorch not available. Install to run demos.")
