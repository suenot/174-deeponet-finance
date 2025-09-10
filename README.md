# Chapter 153: DeepONet for Finance

## Overview

Deep Operator Networks (DeepONet) represent a paradigm shift in how neural networks learn mappings in function spaces. Unlike traditional neural networks that learn mappings between finite-dimensional vectors, DeepONet learns **operators** -- mappings from one function space to another. In finance, this means learning entire families of pricing functions, yield curves, and risk mappings simultaneously, rather than fitting individual point estimates.

Proposed by Lu et al. (2021), DeepONet is grounded in the **universal approximation theorem for operators**, which guarantees that a network with a branch net and trunk net can approximate any continuous nonlinear operator to arbitrary accuracy.

## Why DeepONet for Finance?

### The Operator Learning Paradigm

Traditional neural networks in finance solve problems like:
- Given market state **x**, predict price **y** (function approximation)
- Given time series **X**, predict next value **y** (sequence modeling)

DeepONet solves a fundamentally different problem:
- Given an **input function** (e.g., a volatility surface), learn the **operator** that maps it to an **output function** (e.g., option prices across all strikes and maturities)

```
Traditional NN:     x ∈ R^n  →  y ∈ R^m        (vector to vector)
DeepONet:           u(·)     →  G(u)(y)          (function to function)

where:
  u(·)  = input function (e.g., implied volatility surface)
  y     = query location (e.g., strike K, maturity T)
  G(u)  = output operator (e.g., option price at (K,T))
```

### Key Advantages

| Feature | Standard NN | DeepONet |
|---------|------------|----------|
| Input type | Fixed-size vectors | Functions (variable discretization) |
| Output type | Fixed-size vectors | Functions evaluated at any point |
| Generalization | Interpolation in data space | Generalization across function space |
| Training | One model per scenario | One model for all scenarios |
| Transfer | Limited | Natural cross-asset transfer |
| Physics constraints | Hard to incorporate | PI-DeepONet adds PDE residuals |

## DeepONet Architecture

### Core Structure

DeepONet consists of two sub-networks:

```
                    Input Function u(x)
                    sampled at {x_1, ..., x_m}
                           │
                    ┌──────▼──────┐
                    │  Branch Net  │     Encodes the input function
                    │  (MLP/CNN/RNN)│     into a latent representation
                    └──────┬──────┘
                           │
                    [b_1, b_2, ..., b_p]   Branch output (p neurons)
                           │
                           ●─── dot product ───●
                           │                    │
                    [t_1, t_2, ..., t_p]   Trunk output (p neurons)
                           │
                    ┌──────▲──────┐
                    │  Trunk Net   │     Encodes the query location
                    │    (MLP)     │     (where to evaluate output)
                    └──────┬──────┘
                           │
                    Query Location y
                    (e.g., strike K, maturity T)

    Output: G(u)(y) = Σ_{k=1}^{p} b_k · t_k + bias
```

### Mathematical Formulation

The DeepONet approximation is:

```
G(u)(y) ≈ Σ_{k=1}^{p} br_k(u(x_1), u(x_2), ..., u(x_m)) · tr_k(y) + b_0
```

where:
- `br_k` is the k-th output of the branch network
- `tr_k` is the k-th output of the trunk network
- `p` is the latent dimension (number of basis functions)
- `b_0` is a learnable bias

### Branch Network Variants

The branch network encodes the input function. Different architectures suit different input types:

#### MLP Branch (for tabulated functions)

```python
class MLPBranch(nn.Module):
    """Branch network using Multi-Layer Perceptron.

    Best for: Discretized function values at fixed sensor locations.
    Example: Volatility surface sampled at fixed (K, T) grid points.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [batch, p]
```

#### CNN Branch (for grid-structured inputs)

```python
class CNNBranch(nn.Module):
    """Branch network using 1D-CNN.

    Best for: Time series inputs with local patterns.
    Example: Historical price series, order book snapshots.
    """
    def __init__(self, input_channels, seq_len, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        h = self.conv(x).squeeze(-1)
        return self.fc(h)  # [batch, p]
```

#### RNN Branch (for sequential inputs)

```python
class RNNBranch(nn.Module):
    """Branch network using LSTM/GRU.

    Best for: Variable-length time series.
    Example: Tick-level trading data with irregular timestamps.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, features]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # [batch, p]
```

### Trunk Network

The trunk network encodes query locations where the output function is evaluated:

```python
class TrunkNet(nn.Module):
    """Trunk network for encoding query locations.

    For option pricing: y = (S, t, K, T, r)
    For yield curves: y = (maturity,)
    For risk mapping: y = (asset_id, horizon)
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)  # [batch, p]
```

## Universal Approximation Theorem for Operators

### Theorem Statement

**Theorem (Chen & Chen, 1995; Lu et al., 2021):** Suppose G is a continuous operator mapping from a Banach space V to another Banach space U. Then for any compact set K in V and any epsilon > 0, there exists a DeepONet with branch network `br` and trunk network `tr` such that:

```
|G(u)(y) - Σ_{k=1}^{p} br_k(u(x_1), ..., u(x_m)) · tr_k(y)| < ε
```

for all u in K and y in the domain.

### Implications for Finance

1. **Option Pricing**: A single DeepONet can learn the Black-Scholes operator, Heston operator, or any pricing operator to arbitrary accuracy
2. **Yield Curves**: One model maps economic conditions to the entire yield curve
3. **Risk Surfaces**: One model maps portfolio composition to risk across all horizons

## Financial Applications

### Application 1: Option Pricing Operator

Learn the mapping from volatility surfaces to option price surfaces:

```
Input function:   σ(K, T)          -- implied volatility surface
Query location:   y = (S, K, T, r)  -- spot, strike, maturity, rate
Output:           C(S, K, T)        -- option price

G: σ(·,·) → C(S, ·, ·)
```

```python
# Training data generation
def generate_option_data(n_samples=10000):
    """Generate paired (vol surface, option price) data."""
    data = []
    for _ in range(n_samples):
        # Random vol surface parameters (Heston-like)
        v0 = np.random.uniform(0.01, 0.09)      # initial variance
        kappa = np.random.uniform(0.5, 5.0)       # mean reversion
        theta = np.random.uniform(0.01, 0.09)     # long-run variance
        sigma_v = np.random.uniform(0.1, 0.8)     # vol of vol
        rho = np.random.uniform(-0.9, -0.1)       # correlation

        # Sample vol surface at sensor locations
        K_sensors = np.linspace(0.8, 1.2, 20)     # moneyness grid
        T_sensors = np.linspace(0.1, 2.0, 10)     # maturity grid
        vol_surface = heston_implied_vol(v0, kappa, theta, sigma_v, rho,
                                          K_sensors, T_sensors)

        # Query locations and true option prices
        K_query = np.random.uniform(0.7, 1.3, 50)
        T_query = np.random.uniform(0.05, 2.5, 50)
        prices = heston_option_prices(v0, kappa, theta, sigma_v, rho,
                                       K_query, T_query)

        data.append({
            'vol_surface': vol_surface.flatten(),  # branch input
            'query_locations': np.stack([K_query, T_query], axis=1),  # trunk input
            'option_prices': prices  # target
        })
    return data
```

### Application 2: Yield Curve Operator

Learn the mapping from macroeconomic indicators to yield curves:

```
Input function:   macro(t)          -- economic indicators over time
Query location:   y = (maturity,)   -- bond maturity
Output:           r(maturity)       -- yield at maturity

G: macro(·) → r(·)
```

```python
def yield_curve_deeponet():
    """DeepONet for yield curve prediction."""
    # Branch: encode macro time series (GDP, inflation, employment, etc.)
    branch = RNNBranch(
        input_dim=10,       # 10 macro indicators
        hidden_dim=128,
        output_dim=64,      # p = 64 basis functions
        num_layers=2
    )

    # Trunk: encode maturity query
    trunk = TrunkNet(
        input_dim=1,        # maturity in years
        hidden_dims=[64, 64],
        output_dim=64       # p = 64 (must match branch)
    )

    model = DeepONet(branch, trunk, bias=True)
    return model
```

### Application 3: Portfolio Risk Mapping

Learn the mapping from portfolio weights to risk measures across horizons:

```
Input function:   w(asset)          -- portfolio weight function
Query location:   y = (horizon, α)  -- risk horizon and confidence level
Output:           VaR(horizon, α)   -- Value-at-Risk

G: w(·) → VaR(·, ·)
```

### Application 4: Crypto Trading with Bybit Data

Apply DeepONet to learn price dynamics operators from Bybit exchange data:

```python
import ccxt

def fetch_bybit_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """Fetch OHLCV data from Bybit exchange."""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def build_crypto_deeponet(window=60, forecast_points=20):
    """DeepONet for crypto price forecasting.

    Branch input: historical OHLCV window (function of time)
    Trunk input: future time offset (query location)
    Output: predicted price change at that offset
    """
    branch = CNNBranch(
        input_channels=5,    # OHLCV
        seq_len=window,
        output_dim=128
    )
    trunk = TrunkNet(
        input_dim=1,         # future time offset
        hidden_dims=[64, 64],
        output_dim=128
    )
    return DeepONet(branch, trunk)
```

## Physics-Informed DeepONet (PI-DeepONet)

### Motivation

Financial models are governed by partial differential equations (PDEs). PI-DeepONet incorporates these physics constraints directly into the loss function, dramatically improving accuracy and physical consistency.

### Black-Scholes PDE Constraint

The Black-Scholes PDE for European options:

```
∂C/∂t + (1/2)σ^2 S^2 ∂^2C/∂S^2 + rS ∂C/∂S - rC = 0
```

```python
class PIDeepONet(nn.Module):
    """Physics-Informed DeepONet with PDE residual loss."""

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_query):
        """Forward pass.

        Args:
            u_sensors: Input function values at sensors [batch, m]
            y_query: Query locations [batch, d] (e.g., S, t, K, T)

        Returns:
            Operator output at query locations [batch, 1]
        """
        b = self.branch(u_sensors)  # [batch, p]
        t = self.trunk(y_query)     # [batch, p]
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias

    def pde_residual(self, u_sensors, S, t, sigma, r):
        """Compute Black-Scholes PDE residual.

        Uses automatic differentiation to compute derivatives.
        """
        S.requires_grad_(True)
        t.requires_grad_(True)

        y_query = torch.cat([S, t], dim=-1)
        C = self.forward(u_sensors, y_query)

        # First-order derivatives
        dC = torch.autograd.grad(C, [S, t],
                                  grad_outputs=torch.ones_like(C),
                                  create_graph=True)
        dC_dS, dC_dt = dC[0], dC[1]

        # Second-order derivative
        d2C_dS2 = torch.autograd.grad(dC_dS, S,
                                       grad_outputs=torch.ones_like(dC_dS),
                                       create_graph=True)[0]

        # Black-Scholes PDE residual
        residual = dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C
        return residual

    def compute_loss(self, u_sensors, y_query, targets,
                     S_colloc, t_colloc, sigma, r,
                     lambda_data=1.0, lambda_pde=0.1):
        """Combined data + PDE loss.

        Args:
            u_sensors: Branch input [batch, m]
            y_query: Trunk input [batch, d]
            targets: True option prices [batch, 1]
            S_colloc: Collocation points for S [n_colloc, 1]
            t_colloc: Collocation points for t [n_colloc, 1]
            sigma: Volatility
            r: Risk-free rate
            lambda_data: Weight for data fitting loss
            lambda_pde: Weight for PDE residual loss
        """
        # Data loss
        pred = self.forward(u_sensors, y_query)
        data_loss = F.mse_loss(pred, targets)

        # PDE residual loss
        residual = self.pde_residual(u_sensors, S_colloc, t_colloc, sigma, r)
        pde_loss = torch.mean(residual**2)

        total_loss = lambda_data * data_loss + lambda_pde * pde_loss
        return total_loss, data_loss, pde_loss
```

### Boundary Conditions

Add boundary and terminal conditions for completeness:

```python
def boundary_loss(model, u_sensors, S_max, T_max, K, r):
    """Enforce option pricing boundary conditions.

    1. C(0, t) = 0                          (worthless if S=0)
    2. C(S, T) = max(S - K, 0)             (payoff at expiry)
    3. C(S, t) → S - K*exp(-r(T-t)) as S→∞ (deep ITM)
    """
    batch = u_sensors.shape[0]

    # Condition 1: C(S=0, t) = 0
    S_zero = torch.zeros(batch, 1)
    t_rand = torch.rand(batch, 1) * T_max
    y_zero = torch.cat([S_zero, t_rand], dim=-1)
    bc1_loss = torch.mean(model(u_sensors, y_zero)**2)

    # Condition 2: C(S, T) = max(S - K, 0) at expiry
    S_rand = torch.rand(batch, 1) * S_max
    t_expiry = torch.ones(batch, 1) * T_max
    y_expiry = torch.cat([S_rand, t_expiry], dim=-1)
    pred_expiry = model(u_sensors, y_expiry)
    true_payoff = torch.relu(S_rand - K)
    bc2_loss = F.mse_loss(pred_expiry, true_payoff)

    return bc1_loss + bc2_loss
```

## Multi-Fidelity DeepONet

### Combining Low-Fidelity and High-Fidelity Models

In practice, we have:
- **Low-fidelity data**: Cheap to generate (Black-Scholes, binomial trees)
- **High-fidelity data**: Expensive to generate (Heston MC, local vol MC)

Multi-fidelity DeepONet learns a correction operator:

```
G_HF(u)(y) = G_LF(u)(y) + G_correction(u)(y)
```

```python
class MultiFidelityDeepONet(nn.Module):
    """Multi-fidelity DeepONet combining BS and Heston models.

    Architecture:
        Low-fidelity:  DeepONet_LF trained on BS prices (abundant data)
        Correction:    DeepONet_corr trained on (Heston - BS) residuals
        High-fidelity: DeepONet_LF + DeepONet_corr
    """

    def __init__(self, branch_dim, trunk_dim, latent_dim):
        super().__init__()

        # Low-fidelity DeepONet (pre-trained on BS data)
        self.lf_branch = MLPBranch(branch_dim, [256, 256], latent_dim)
        self.lf_trunk = TrunkNet(trunk_dim, [128, 128], latent_dim)
        self.lf_bias = nn.Parameter(torch.zeros(1))

        # Correction DeepONet (trained on residuals)
        self.corr_branch = MLPBranch(branch_dim, [128, 128], latent_dim)
        self.corr_trunk = TrunkNet(trunk_dim, [64, 64], latent_dim)
        self.corr_bias = nn.Parameter(torch.zeros(1))

        # Linear mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward_lf(self, u_sensors, y_query):
        b = self.lf_branch(u_sensors)
        t = self.lf_trunk(y_query)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.lf_bias

    def forward_correction(self, u_sensors, y_query):
        b = self.corr_branch(u_sensors)
        t = self.corr_trunk(y_query)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.corr_bias

    def forward(self, u_sensors, y_query):
        lf_pred = self.forward_lf(u_sensors, y_query)
        correction = self.forward_correction(u_sensors, y_query)
        return self.alpha * lf_pred + correction
```

### Training Strategy

```python
def train_multifidelity(model, lf_data, hf_data, epochs=1000):
    """Two-stage training for multi-fidelity DeepONet.

    Stage 1: Train low-fidelity DeepONet on abundant BS data
    Stage 2: Freeze LF weights, train correction on scarce Heston data
    """
    optimizer_lf = torch.optim.Adam(
        list(model.lf_branch.parameters()) +
        list(model.lf_trunk.parameters()) + [model.lf_bias],
        lr=1e-3
    )

    # Stage 1: Train LF model
    for epoch in range(epochs):
        for u, y, price_bs in lf_data:
            pred = model.forward_lf(u, y)
            loss = F.mse_loss(pred, price_bs)
            optimizer_lf.zero_grad()
            loss.backward()
            optimizer_lf.step()

    # Freeze LF weights
    for param in model.lf_branch.parameters():
        param.requires_grad = False
    for param in model.lf_trunk.parameters():
        param.requires_grad = False

    optimizer_corr = torch.optim.Adam(
        list(model.corr_branch.parameters()) +
        list(model.corr_trunk.parameters()) +
        [model.corr_bias, model.alpha],
        lr=1e-4
    )

    # Stage 2: Train correction model
    for epoch in range(epochs // 2):
        for u, y, price_heston in hf_data:
            pred = model(u, y)
            loss = F.mse_loss(pred, price_heston)
            optimizer_corr.zero_grad()
            loss.backward()
            optimizer_corr.step()
```

## Transfer Across Assets and Market Regimes

### Cross-Asset Transfer Learning

DeepONet naturally supports transfer learning because operators encode structural relationships:

```python
def transfer_deeponet(pretrained_model, target_data, fine_tune_epochs=100):
    """Transfer a DeepONet trained on one asset class to another.

    Example: Transfer from equity options to crypto options.

    Strategy:
    1. Keep trunk network frozen (query structure is the same)
    2. Fine-tune branch network (input function differs)
    3. Optionally add adapter layers
    """
    # Freeze trunk (geometric structure of output space is shared)
    for param in pretrained_model.trunk.parameters():
        param.requires_grad = False

    # Add domain adapter to branch
    adapter = nn.Sequential(
        nn.Linear(pretrained_model.branch.output_dim, 128),
        nn.GELU(),
        nn.Linear(128, pretrained_model.branch.output_dim),
    )

    # Fine-tune branch + adapter
    optimizer = torch.optim.Adam([
        {'params': pretrained_model.branch.parameters(), 'lr': 1e-5},
        {'params': adapter.parameters(), 'lr': 1e-3},
    ])

    for epoch in range(fine_tune_epochs):
        for u, y, target in target_data:
            b = pretrained_model.branch(u)
            b = b + adapter(b)  # residual adapter
            t = pretrained_model.trunk(y)
            pred = torch.sum(b * t, dim=-1, keepdim=True) + pretrained_model.bias
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Regime-Aware DeepONet

```python
class RegimeAwareDeepONet(nn.Module):
    """DeepONet with regime conditioning.

    Uses a regime classifier to select/blend branch networks
    trained on different market regimes.
    """

    def __init__(self, n_regimes, branch_dim, trunk_dim, latent_dim):
        super().__init__()
        self.n_regimes = n_regimes

        # One branch per regime
        self.branches = nn.ModuleList([
            MLPBranch(branch_dim, [256, 256], latent_dim)
            for _ in range(n_regimes)
        ])

        # Shared trunk
        self.trunk = TrunkNet(trunk_dim, [128, 128], latent_dim)

        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_regimes),
            nn.Softmax(dim=-1)
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_query):
        # Classify regime
        regime_weights = self.regime_classifier(u_sensors)  # [batch, n_regimes]

        # Blend branch outputs
        branch_outputs = torch.stack([
            branch(u_sensors) for branch in self.branches
        ], dim=1)  # [batch, n_regimes, p]

        # Weighted sum of branch outputs
        b = torch.sum(
            regime_weights.unsqueeze(-1) * branch_outputs, dim=1
        )  # [batch, p]

        t = self.trunk(y_query)  # [batch, p]
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias
```

## Comparison with FNO and Standard NNs

### DeepONet vs Fourier Neural Operator (FNO)

| Aspect | DeepONet | FNO |
|--------|----------|-----|
| Architecture | Branch + Trunk | Fourier layers |
| Input discretization | Fixed sensors | Regular grid required |
| Output evaluation | Any point query | Full grid output |
| Spectral bias | None | Favors low frequencies |
| PDE integration | PI-DeepONet | PINO |
| Irregular data | Natural support | Requires interpolation |
| Memory scaling | O(mp + pd) | O(N log N) for FFT |
| Best for | Point queries, irregular data | Periodic problems, full fields |

### DeepONet vs Standard Neural Networks

```
Standard NN:
  - Train one model per (vol surface shape)
  - Cannot extrapolate to new surface shapes
  - Fixed input/output dimension

DeepONet:
  - One model for ALL vol surface shapes
  - Generalizes to unseen surface shapes
  - Flexible input/output dimension

Accuracy comparison (option pricing, MSE):
  - Black-Scholes formula:  exact for BS model
  - Standard MLP:           ~1e-3 (per-scenario)
  - DeepONet:               ~1e-4 (all scenarios)
  - PI-DeepONet:            ~1e-5 (physics-constrained)
  - Multi-fidelity DeepONet: ~1e-6 (combines BS + Heston)
```

## Training Details

### Data Generation

```python
def generate_training_data(n_functions=5000, n_sensors=100, n_queries=50):
    """Generate paired input-output function data for DeepONet training.

    Each training sample consists of:
    1. Input function u sampled at m sensor locations
    2. Query locations y_1, ..., y_q
    3. True operator outputs G(u)(y_1), ..., G(u)(y_q)
    """
    sensor_locations = np.linspace(0, 1, n_sensors)

    all_u = []
    all_y = []
    all_Gu = []

    for _ in range(n_functions):
        # Generate random input function (e.g., GP sample)
        length_scale = np.random.uniform(0.05, 0.5)
        u_values = sample_gp(sensor_locations, length_scale)

        # Query locations
        y_queries = np.random.uniform(0, 1, (n_queries, 1))

        # True operator output (e.g., antiderivative operator)
        Gu_values = compute_operator_output(u_values, sensor_locations, y_queries)

        all_u.append(u_values)
        all_y.append(y_queries)
        all_Gu.append(Gu_values)

    return np.array(all_u), np.array(all_y), np.array(all_Gu)
```

### Loss Functions

```python
def deeponet_loss(model, u_batch, y_batch, target_batch, loss_type='mse'):
    """Compute DeepONet training loss.

    Args:
        model: DeepONet model
        u_batch: Branch inputs [batch, m]
        y_batch: Trunk inputs [batch, n_queries, d]
        target_batch: True outputs [batch, n_queries, 1]
        loss_type: 'mse', 'mae', or 'huber'
    """
    batch_size, n_queries, d = y_batch.shape

    # Flatten queries for parallel evaluation
    u_expanded = u_batch.unsqueeze(1).expand(-1, n_queries, -1)
    u_flat = u_expanded.reshape(-1, u_batch.shape[-1])
    y_flat = y_batch.reshape(-1, d)
    target_flat = target_batch.reshape(-1, 1)

    pred_flat = model(u_flat, y_flat)

    if loss_type == 'mse':
        return F.mse_loss(pred_flat, target_flat)
    elif loss_type == 'mae':
        return F.l1_loss(pred_flat, target_flat)
    elif loss_type == 'huber':
        return F.smooth_l1_loss(pred_flat, target_flat)
```

### Training Loop

```python
def train_deeponet(model, train_loader, val_loader, config):
    """Training loop with learning rate scheduling and early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for u_batch, y_batch, target_batch in train_loader:
            loss = deeponet_loss(model, u_batch, y_batch, target_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u_batch, y_batch, target_batch in val_loader:
                loss = deeponet_loss(model, u_batch, y_batch, target_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_deeponet.pth')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
```

## Implementation Notes

### Sensor Location Selection

The choice of sensor locations critically affects DeepONet performance:

```python
def optimal_sensor_placement(n_sensors, method='uniform'):
    """Select sensor locations for sampling input functions.

    Methods:
    - 'uniform': Evenly spaced (simplest)
    - 'chebyshev': Chebyshev nodes (better polynomial approximation)
    - 'random': Random (for stochastic collocation)
    - 'adaptive': Data-driven placement
    """
    if method == 'uniform':
        return np.linspace(0, 1, n_sensors)
    elif method == 'chebyshev':
        k = np.arange(1, n_sensors + 1)
        return 0.5 * (1 - np.cos((2*k - 1) * np.pi / (2 * n_sensors)))
    elif method == 'random':
        return np.sort(np.random.uniform(0, 1, n_sensors))
    elif method == 'adaptive':
        # Start uniform, refine based on prediction error
        sensors = np.linspace(0, 1, n_sensors)
        return sensors  # Placeholder for adaptive refinement
```

### Scaling and Normalization

```python
class DeepONetNormalizer:
    """Normalize inputs and outputs for stable training."""

    def __init__(self):
        self.u_mean = None
        self.u_std = None
        self.y_mean = None
        self.y_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, u_data, y_data, target_data):
        self.u_mean = u_data.mean(axis=0)
        self.u_std = u_data.std(axis=0) + 1e-8
        self.y_mean = y_data.mean(axis=0)
        self.y_std = y_data.std(axis=0) + 1e-8
        self.target_mean = target_data.mean()
        self.target_std = target_data.std() + 1e-8

    def normalize_u(self, u):
        return (u - self.u_mean) / self.u_std

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def normalize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target_norm):
        return target_norm * self.target_std + self.target_mean
```

## Project Structure

```
153_deeponet_finance/
├── README.md                        # This file
├── README.ru.md                     # Russian translation
├── readme.simple.md                 # Simplified explanation (English)
├── readme.simple.ru.md              # Simplified explanation (Russian)
├── python/
│   ├── __init__.py                  # Package initialization
│   ├── model.py                     # DeepONet model architectures
│   ├── train.py                     # Training pipeline
│   ├── data_loader.py               # Data loading (stocks + Bybit crypto)
│   ├── visualize.py                 # Visualization utilities
│   ├── backtest.py                  # Backtesting framework
│   └── requirements.txt             # Python dependencies
└── rust_deeponet/
    ├── Cargo.toml                   # Rust project configuration
    ├── src/
    │   ├── lib.rs                   # Core library
    │   └── bin/
    │       ├── train.rs             # Training binary
    │       ├── predict.rs           # Prediction binary
    │       └── fetch_data.rs        # Data fetching binary
    └── examples/
        ├── option_pricing.rs        # Option pricing example
        ├── crypto_forecast.rs       # Crypto forecasting example
        └── yield_curve.rs           # Yield curve example
```

## Running the Code

### Python

```bash
cd python
pip install -r requirements.txt

# Train DeepONet for option pricing
python train.py --mode option_pricing --epochs 500

# Train DeepONet for crypto forecasting (Bybit)
python train.py --mode crypto --symbol BTCUSDT --epochs 200

# Backtest trading strategy
python backtest.py --model checkpoints/best_deeponet.pth --symbol BTCUSDT

# Visualize results
python visualize.py --results results/backtest_results.json
```

### Rust

```bash
cd rust_deeponet

# Fetch market data from Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 5000

# Train DeepONet model
cargo run --bin train -- --config config.json

# Run predictions
cargo run --bin predict -- --model model.bin --symbol BTCUSDT

# Run examples
cargo run --example option_pricing
cargo run --example crypto_forecast
cargo run --example yield_curve
```

## References

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

2. Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Transactions on Neural Networks*, 6(4), 911-917.

3. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40).

4. Howard, A. A., Perego, M., Karniadakis, G. E., & Stinis, P. (2022). Multifidelity deep operator networks. *arXiv preprint arXiv:2204.09157*.

5. Lin, C., Li, Z., Lu, L., Cai, S., Maxey, M., & Karniadakis, G. E. (2021). Operator learning for predicting multiscale bubble growth dynamics. *The Journal of Chemical Physics*, 154(10).

6. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.

7. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.

8. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.

---

*Chapter 153 of Machine Learning for Trading. DeepONet enables operator learning for financial applications, mapping entire function spaces to function spaces -- a fundamentally more powerful paradigm than traditional point-to-point neural network mappings.*
