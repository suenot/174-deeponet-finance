//! # DeepONet Finance
//!
//! Deep Operator Networks (DeepONet) for financial applications.
//! Implements operator learning for option pricing, yield curve prediction,
//! and cryptocurrency trading with Bybit exchange data.
//!
//! ## Architecture
//!
//! DeepONet = Branch Network + Trunk Network
//! - Branch: encodes input function (vol surface, price history, macro indicators)
//! - Trunk: encodes query location (strike, maturity, time offset)
//! - Output: dot product of branch and trunk outputs + bias
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `model` - DeepONet model implementation (forward pass, training)
//! - `data` - Data preprocessing and normalization
//! - `backtest` - Backtesting framework
//! - `optim` - Optimizer and training utilities

pub mod api;
pub mod backtest;
pub mod data;
pub mod model;
pub mod optim;

pub use api::{BybitClient, Kline};
pub use backtest::{BacktestConfig, BacktestEngine, BacktestResult};
pub use data::{DataNormalizer, DeepONetDataset};
pub use model::{DeepONet, DeepONetConfig, Layer};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

// ═══════════════════════════════════════════════════════════════════════════════
// API MODULE
// ═══════════════════════════════════════════════════════════════════════════════

pub mod api {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};

    /// OHLCV candle data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Kline {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    impl Kline {
        pub fn datetime(&self) -> DateTime<Utc> {
            DateTime::from_timestamp_millis(self.timestamp)
                .unwrap_or_default()
        }
    }

    /// Bybit API response wrapper
    #[derive(Debug, Deserialize)]
    pub struct BybitResponse {
        #[serde(rename = "retCode")]
        pub ret_code: i32,
        #[serde(rename = "retMsg")]
        pub ret_msg: String,
        pub result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    pub struct BybitResult {
        pub list: Vec<Vec<serde_json::Value>>,
    }

    /// Bybit API client
    pub struct BybitClient {
        base_url: String,
        client: reqwest::blocking::Client,
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
                client: reqwest::blocking::Client::new(),
            }
        }

        /// Fetch kline (OHLCV) data from Bybit
        pub fn get_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
        ) -> anyhow::Result<Vec<Kline>> {
            let url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                self.base_url, symbol, interval, limit
            );

            let response: BybitResponse = self.client.get(&url).send()?.json()?;

            if response.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", response.ret_msg);
            }

            let mut klines: Vec<Kline> = response
                .result
                .list
                .iter()
                .filter_map(|row| {
                    if row.len() >= 6 {
                        Some(Kline {
                            timestamp: row[0].as_str()?.parse().ok()?,
                            open: row[1].as_str()?.parse().ok()?,
                            high: row[2].as_str()?.parse().ok()?,
                            low: row[3].as_str()?.parse().ok()?,
                            close: row[4].as_str()?.parse().ok()?,
                            volume: row[5].as_str()?.parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // Bybit returns newest first, reverse to chronological order
            klines.reverse();
            Ok(klines)
        }

        /// Fetch klines with pagination for large requests
        pub fn get_klines_paginated(
            &self,
            symbol: &str,
            interval: &str,
            total_limit: usize,
        ) -> anyhow::Result<Vec<Kline>> {
            let mut all_klines = Vec::new();
            let batch_size = 200;
            let mut end_time: Option<i64> = None;

            while all_klines.len() < total_limit {
                let remaining = total_limit - all_klines.len();
                let limit = remaining.min(batch_size);

                let mut url = format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                    self.base_url, symbol, interval, limit
                );

                if let Some(end) = end_time {
                    url.push_str(&format!("&end={}", end));
                }

                let response: BybitResponse = self.client.get(&url).send()?.json()?;
                if response.ret_code != 0 {
                    break;
                }

                let klines: Vec<Kline> = response
                    .result
                    .list
                    .iter()
                    .filter_map(|row| {
                        if row.len() >= 6 {
                            Some(Kline {
                                timestamp: row[0].as_str()?.parse().ok()?,
                                open: row[1].as_str()?.parse().ok()?,
                                high: row[2].as_str()?.parse().ok()?,
                                low: row[3].as_str()?.parse().ok()?,
                                close: row[4].as_str()?.parse().ok()?,
                                volume: row[5].as_str()?.parse().ok()?,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if klines.is_empty() {
                    break;
                }

                // Get earliest timestamp for next pagination
                if let Some(earliest) = klines.last() {
                    end_time = Some(earliest.timestamp - 1);
                }

                all_klines.extend(klines);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            // Sort chronologically
            all_klines.sort_by_key(|k| k.timestamp);
            all_klines.dedup_by_key(|k| k.timestamp);

            if all_klines.len() > total_limit {
                all_klines.truncate(total_limit);
            }

            Ok(all_klines)
        }
    }

    /// Generate synthetic OHLCV data for testing
    pub fn generate_synthetic_klines(n: usize, seed: u64) -> Vec<Kline> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 0.03).unwrap();
        let jump_normal = Normal::new(0.0, 0.05).unwrap();

        let mut klines = Vec::with_capacity(n);
        let mut price = 50000.0_f64;
        let base_ts = 1700000000000_i64; // Approximate epoch ms

        for i in 0..n {
            let ret: f64 = normal.sample(&mut rng);
            let jump = if rng.gen::<f64>() < 0.01 {
                jump_normal.sample(&mut rng)
            } else {
                0.0
            };

            price *= (ret + jump + 0.0001).exp();
            let open = price * (1.0 + rng.gen::<f64>() * 0.002 - 0.001);
            let high = price.max(open) * (1.0 + rng.gen::<f64>().abs() * 0.005);
            let low = price.min(open) * (1.0 - rng.gen::<f64>().abs() * 0.005);
            let volume = (rng.gen::<f64>() * 1000.0 + 100.0).exp().ln() * 1000.0;

            klines.push(Kline {
                timestamp: base_ts + (i as i64) * 3600000,
                open,
                high,
                low,
                close: price,
                volume: volume.abs(),
            });
        }

        klines
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA MODULE
// ═══════════════════════════════════════════════════════════════════════════════

pub mod data {
    use ndarray::{Array1, Array2, Axis};

    use crate::api::Kline;

    /// Normalizer for DeepONet inputs/outputs
    #[derive(Debug, Clone)]
    pub struct DataNormalizer {
        pub u_mean: Array1<f64>,
        pub u_std: Array1<f64>,
        pub y_mean: Array1<f64>,
        pub y_std: Array1<f64>,
        pub target_mean: f64,
        pub target_std: f64,
    }

    impl DataNormalizer {
        /// Fit normalizer on data
        pub fn fit(
            u_data: &Array2<f64>,
            y_data: &Array2<f64>,
            targets: &Array1<f64>,
        ) -> Self {
            let u_mean = u_data.mean_axis(Axis(0)).unwrap();
            let u_std = u_data.std_axis(Axis(0), 0.0) + 1e-8;
            let y_mean = y_data.mean_axis(Axis(0)).unwrap();
            let y_std = y_data.std_axis(Axis(0), 0.0) + 1e-8;
            let target_mean = targets.mean().unwrap_or(0.0);
            let target_std = targets.std(0.0) + 1e-8;

            Self {
                u_mean,
                u_std,
                y_mean,
                y_std,
                target_mean,
                target_std,
            }
        }

        pub fn normalize_u(&self, u: &Array2<f64>) -> Array2<f64> {
            (u - &self.u_mean) / &self.u_std
        }

        pub fn normalize_y(&self, y: &Array2<f64>) -> Array2<f64> {
            (y - &self.y_mean) / &self.y_std
        }

        pub fn normalize_target(&self, t: &Array1<f64>) -> Array1<f64> {
            (t - self.target_mean) / self.target_std
        }

        pub fn denormalize_target(&self, t: &Array1<f64>) -> Array1<f64> {
            t * self.target_std + self.target_mean
        }
    }

    /// Dataset for DeepONet training
    #[derive(Debug, Clone)]
    pub struct DeepONetDataset {
        pub u_sensors: Array2<f64>,  // [n_samples, sensor_dim]
        pub y_queries: Array2<f64>,  // [n_samples, query_dim]
        pub targets: Array1<f64>,    // [n_samples]
    }

    impl DeepONetDataset {
        pub fn len(&self) -> usize {
            self.u_sensors.nrows()
        }

        pub fn is_empty(&self) -> bool {
            self.u_sensors.nrows() == 0
        }

        /// Get a batch of data
        pub fn get_batch(
            &self,
            indices: &[usize],
        ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
            let n = indices.len();
            let u_dim = self.u_sensors.ncols();
            let y_dim = self.y_queries.ncols();

            let mut u_batch = Array2::zeros((n, u_dim));
            let mut y_batch = Array2::zeros((n, y_dim));
            let mut t_batch = Array1::zeros(n);

            for (i, &idx) in indices.iter().enumerate() {
                u_batch.row_mut(i).assign(&self.u_sensors.row(idx));
                y_batch.row_mut(i).assign(&self.y_queries.row(idx));
                t_batch[i] = self.targets[idx];
            }

            (u_batch, y_batch, t_batch)
        }

        /// Split dataset into train/val/test
        pub fn split(
            &self,
            train_ratio: f64,
            val_ratio: f64,
        ) -> (Self, Self, Self) {
            let n = self.len();
            let n_train = (n as f64 * train_ratio) as usize;
            let n_val = (n as f64 * val_ratio) as usize;

            let train = Self {
                u_sensors: self.u_sensors.slice(ndarray::s![..n_train, ..]).to_owned(),
                y_queries: self.y_queries.slice(ndarray::s![..n_train, ..]).to_owned(),
                targets: self.targets.slice(ndarray::s![..n_train]).to_owned(),
            };

            let val = Self {
                u_sensors: self.u_sensors.slice(ndarray::s![n_train..n_train + n_val, ..]).to_owned(),
                y_queries: self.y_queries.slice(ndarray::s![n_train..n_train + n_val, ..]).to_owned(),
                targets: self.targets.slice(ndarray::s![n_train..n_train + n_val]).to_owned(),
            };

            let test = Self {
                u_sensors: self.u_sensors.slice(ndarray::s![n_train + n_val.., ..]).to_owned(),
                y_queries: self.y_queries.slice(ndarray::s![n_train + n_val.., ..]).to_owned(),
                targets: self.targets.slice(ndarray::s![n_train + n_val..]).to_owned(),
            };

            (train, val, test)
        }
    }

    /// Prepare crypto OHLCV data for DeepONet
    pub fn prepare_crypto_data(
        klines: &[Kline],
        window_size: usize,
        max_forecast_horizon: usize,
    ) -> DeepONetDataset {
        let n = klines.len();
        let total_len = window_size + max_forecast_horizon;

        let mut u_list = Vec::new();
        let mut y_list = Vec::new();
        let mut t_list = Vec::new();

        for i in 0..(n.saturating_sub(total_len)) {
            let window = &klines[i..i + window_size];

            // Normalize by mean close price
            let price_mean: f64 = window.iter().map(|k| k.close).sum::<f64>() / window_size as f64;
            let price_std = (window
                .iter()
                .map(|k| (k.close - price_mean).powi(2))
                .sum::<f64>()
                / window_size as f64)
                .sqrt()
                + 1e-8;

            // Flatten OHLCV into sensor vector (window_size * 5)
            let mut u_vec = Vec::with_capacity(window_size * 5);
            for k in window {
                u_vec.push((k.open - price_mean) / price_std);
                u_vec.push((k.high - price_mean) / price_std);
                u_vec.push((k.low - price_mean) / price_std);
                u_vec.push((k.close - price_mean) / price_std);
                u_vec.push(k.volume.ln().max(0.0)); // Log volume
            }

            // Multiple forecast points
            let base_price = klines[i + window_size - 1].close;
            for h in 1..=max_forecast_horizon.min(n - i - window_size) {
                let future_price = klines[i + window_size + h - 1].close;
                let ret = (future_price - base_price) / price_std;

                u_list.push(u_vec.clone());
                y_list.push(vec![h as f64 / max_forecast_horizon as f64]);
                t_list.push(ret);
            }
        }

        let n_samples = u_list.len();
        if n_samples == 0 {
            return DeepONetDataset {
                u_sensors: Array2::zeros((0, window_size * 5)),
                y_queries: Array2::zeros((0, 1)),
                targets: Array1::zeros(0),
            };
        }

        let u_dim = u_list[0].len();
        let y_dim = y_list[0].len();

        let mut u_sensors = Array2::zeros((n_samples, u_dim));
        let mut y_queries = Array2::zeros((n_samples, y_dim));
        let mut targets = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for (j, &val) in u_list[i].iter().enumerate() {
                u_sensors[[i, j]] = val;
            }
            for (j, &val) in y_list[i].iter().enumerate() {
                y_queries[[i, j]] = val;
            }
            targets[i] = t_list[i];
        }

        DeepONetDataset {
            u_sensors,
            y_queries,
            targets,
        }
    }

    /// Black-Scholes call option price
    pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        if t <= 1e-10 {
            return (s - k).max(0.0);
        }

        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();

        s * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
    }

    /// Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun)
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Generate option pricing dataset
    pub fn generate_option_dataset(
        n_samples: usize,
        n_sensors: usize,
        seed: u64,
    ) -> DeepONetDataset {
        use rand::prelude::*;
        use rand_distr::{Normal, Uniform};

        let mut rng = StdRng::seed_from_u64(seed);
        let vol_dist = Uniform::new(0.1, 0.5);
        let strike_dist = Uniform::new(0.7, 1.3);
        let maturity_dist = Uniform::new(0.1, 2.0);

        let mut u_list = Vec::new();
        let mut y_list = Vec::new();
        let mut t_list = Vec::new();

        let sensor_strikes: Vec<f64> = (0..n_sensors / 2)
            .map(|i| 0.8 + 0.4 * (i as f64) / (n_sensors as f64 / 2.0 - 1.0))
            .collect();
        let sensor_maturities: Vec<f64> = (0..n_sensors / (n_sensors / 2))
            .map(|i| 0.1 + 1.9 * (i as f64) / ((n_sensors / (n_sensors / 2)) as f64 - 1.0).max(1.0))
            .collect();

        for _ in 0..n_samples {
            let base_vol: f64 = vol_dist.sample(&mut rng);
            let normal = Normal::new(0.0, 0.05).unwrap();

            // Generate vol surface sensor values
            let mut u_vec = Vec::with_capacity(n_sensors);
            for _ in 0..n_sensors {
                let vol = (base_vol + normal.sample(&mut rng)).max(0.05);
                u_vec.push(vol);
            }

            // Query point
            let k: f64 = strike_dist.sample(&mut rng);
            let t_mat: f64 = maturity_dist.sample(&mut rng);

            // Compute BS price
            let sigma = base_vol;
            let price = black_scholes_call(1.0, k, t_mat, 0.02, sigma);

            u_list.push(u_vec);
            y_list.push(vec![k, t_mat]);
            t_list.push(price);
        }

        let u_dim = n_sensors;
        let n = u_list.len();

        let mut u_sensors = Array2::zeros((n, u_dim));
        let mut y_queries = Array2::zeros((n, 2));
        let mut targets = Array1::zeros(n);

        for i in 0..n {
            for (j, &val) in u_list[i].iter().enumerate() {
                u_sensors[[i, j]] = val;
            }
            y_queries[[i, 0]] = y_list[i][0];
            y_queries[[i, 1]] = y_list[i][1];
            targets[i] = t_list[i];
        }

        DeepONetDataset {
            u_sensors,
            y_queries,
            targets,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL MODULE
// ═══════════════════════════════════════════════════════════════════════════════

pub mod model {
    use ndarray::{Array1, Array2};
    use rand::prelude::*;
    use rand_distr::Normal;
    use serde::{Deserialize, Serialize};

    /// Configuration for DeepONet
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DeepONetConfig {
        pub branch_input_dim: usize,
        pub trunk_input_dim: usize,
        pub branch_hidden_dims: Vec<usize>,
        pub trunk_hidden_dims: Vec<usize>,
        pub latent_dim: usize,
        pub learning_rate: f64,
        pub epochs: usize,
        pub batch_size: usize,
    }

    impl Default for DeepONetConfig {
        fn default() -> Self {
            Self {
                branch_input_dim: 100,
                trunk_input_dim: 1,
                branch_hidden_dims: vec![128, 128],
                trunk_hidden_dims: vec![64, 64],
                latent_dim: 64,
                learning_rate: 0.001,
                epochs: 200,
                batch_size: 64,
            }
        }
    }

    /// Dense layer with weights and bias
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Layer {
        pub weights: Vec<Vec<f64>>,  // [output_dim][input_dim]
        pub bias: Vec<f64>,          // [output_dim]
        pub input_dim: usize,
        pub output_dim: usize,
    }

    impl Layer {
        /// Create a new layer with Xavier initialization
        pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();

            let weights: Vec<Vec<f64>> = (0..output_dim)
                .map(|_| (0..input_dim).map(|_| normal.sample(&mut rng)).collect())
                .collect();
            let bias = vec![0.0; output_dim];

            Self {
                weights,
                bias,
                input_dim,
                output_dim,
            }
        }

        /// Forward pass: output = weights @ input + bias, then GELU activation
        pub fn forward(&self, input: &[f64], apply_activation: bool) -> Vec<f64> {
            let mut output = vec![0.0; self.output_dim];

            for i in 0..self.output_dim {
                let mut sum = self.bias[i];
                for j in 0..self.input_dim {
                    sum += self.weights[i][j] * input[j];
                }
                output[i] = if apply_activation { gelu(sum) } else { sum };
            }

            output
        }

        /// Forward pass for a batch
        pub fn forward_batch(
            &self,
            input: &Array2<f64>,
            apply_activation: bool,
        ) -> Array2<f64> {
            let n = input.nrows();
            let mut output = Array2::zeros((n, self.output_dim));

            for sample in 0..n {
                for i in 0..self.output_dim {
                    let mut sum = self.bias[i];
                    for j in 0..self.input_dim {
                        sum += self.weights[i][j] * input[[sample, j]];
                    }
                    output[[sample, i]] = if apply_activation { gelu(sum) } else { sum };
                }
            }

            output
        }

        /// Compute gradients and update weights (simple SGD)
        pub fn update_weights(
            &mut self,
            input: &[f64],
            grad_output: &[f64],
            learning_rate: f64,
        ) -> Vec<f64> {
            let mut grad_input = vec![0.0; self.input_dim];

            for i in 0..self.output_dim {
                for j in 0..self.input_dim {
                    grad_input[j] += self.weights[i][j] * grad_output[i];
                    self.weights[i][j] -= learning_rate * grad_output[i] * input[j];
                }
                self.bias[i] -= learning_rate * grad_output[i];
            }

            grad_input
        }
    }

    /// GELU activation function
    fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// DeepONet model: Branch + Trunk networks with dot product
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DeepONet {
        pub branch_layers: Vec<Layer>,
        pub trunk_layers: Vec<Layer>,
        pub bias: f64,
        pub config: DeepONetConfig,
    }

    impl DeepONet {
        /// Create a new DeepONet with the given configuration
        pub fn new(config: DeepONetConfig) -> Self {
            let mut seed = 42u64;

            // Build branch network
            let mut branch_layers = Vec::new();
            let mut prev_dim = config.branch_input_dim;
            for &h_dim in &config.branch_hidden_dims {
                branch_layers.push(Layer::new(prev_dim, h_dim, seed));
                seed += 1;
                prev_dim = h_dim;
            }
            branch_layers.push(Layer::new(prev_dim, config.latent_dim, seed));
            seed += 1;

            // Build trunk network
            let mut trunk_layers = Vec::new();
            prev_dim = config.trunk_input_dim;
            for &h_dim in &config.trunk_hidden_dims {
                trunk_layers.push(Layer::new(prev_dim, h_dim, seed));
                seed += 1;
                prev_dim = h_dim;
            }
            trunk_layers.push(Layer::new(prev_dim, config.latent_dim, seed));

            Self {
                branch_layers,
                trunk_layers,
                bias: 0.0,
                config,
            }
        }

        /// Forward pass through branch network
        pub fn branch_forward(&self, u: &[f64]) -> Vec<f64> {
            let mut x = u.to_vec();
            let n_layers = self.branch_layers.len();
            for (i, layer) in self.branch_layers.iter().enumerate() {
                let activate = i < n_layers - 1; // No activation on last layer
                x = layer.forward(&x, activate);
            }
            x
        }

        /// Forward pass through trunk network
        pub fn trunk_forward(&self, y: &[f64]) -> Vec<f64> {
            let mut x = y.to_vec();
            let n_layers = self.trunk_layers.len();
            for (i, layer) in self.trunk_layers.iter().enumerate() {
                let activate = i < n_layers - 1;
                x = layer.forward(&x, activate);
            }
            x
        }

        /// Full forward pass: dot(branch(u), trunk(y)) + bias
        pub fn forward(&self, u: &[f64], y: &[f64]) -> f64 {
            let b = self.branch_forward(u);
            let t = self.trunk_forward(y);

            let mut output = self.bias;
            for k in 0..self.config.latent_dim {
                output += b[k] * t[k];
            }
            output
        }

        /// Batch prediction
        pub fn predict_batch(
            &self,
            u_batch: &Array2<f64>,
            y_batch: &Array2<f64>,
        ) -> Array1<f64> {
            let n = u_batch.nrows();
            let mut predictions = Array1::zeros(n);

            for i in 0..n {
                let u: Vec<f64> = u_batch.row(i).to_vec();
                let y: Vec<f64> = y_batch.row(i).to_vec();
                predictions[i] = self.forward(&u, &y);
            }

            predictions
        }

        /// Train on a single sample (simple gradient descent)
        pub fn train_step(
            &mut self,
            u: &[f64],
            y: &[f64],
            target: f64,
            learning_rate: f64,
        ) -> f64 {
            // Forward pass
            let pred = self.forward(u, y);
            let error = pred - target;
            let loss = error * error;

            // Backward pass (simplified numerical gradient)
            let eps = 1e-5;

            // Update bias
            self.bias -= learning_rate * 2.0 * error;

            // Update branch layers (numerical gradient)
            for layer in self.branch_layers.iter_mut() {
                for i in 0..layer.output_dim {
                    for j in 0..layer.input_dim {
                        let orig = layer.weights[i][j];

                        layer.weights[i][j] = orig + eps;
                        let pred_plus = self.forward_with_layers(u, y);

                        layer.weights[i][j] = orig - eps;
                        let pred_minus = self.forward_with_layers(u, y);

                        layer.weights[i][j] = orig;

                        let grad = (pred_plus - pred_minus) / (2.0 * eps);
                        layer.weights[i][j] -= learning_rate * 2.0 * error * grad;
                    }

                    // Bias gradient
                    let orig = layer.bias[i];
                    layer.bias[i] = orig + eps;
                    let pred_plus = self.forward_with_layers(u, y);
                    layer.bias[i] = orig - eps;
                    let pred_minus = self.forward_with_layers(u, y);
                    layer.bias[i] = orig;

                    let grad = (pred_plus - pred_minus) / (2.0 * eps);
                    layer.bias[i] -= learning_rate * 2.0 * error * grad;
                }
            }

            // Update trunk layers
            for layer in self.trunk_layers.iter_mut() {
                for i in 0..layer.output_dim {
                    for j in 0..layer.input_dim {
                        let orig = layer.weights[i][j];

                        layer.weights[i][j] = orig + eps;
                        let pred_plus = self.forward_with_layers(u, y);

                        layer.weights[i][j] = orig - eps;
                        let pred_minus = self.forward_with_layers(u, y);

                        layer.weights[i][j] = orig;

                        let grad = (pred_plus - pred_minus) / (2.0 * eps);
                        layer.weights[i][j] -= learning_rate * 2.0 * error * grad;
                    }

                    let orig = layer.bias[i];
                    layer.bias[i] = orig + eps;
                    let pred_plus = self.forward_with_layers(u, y);
                    layer.bias[i] = orig - eps;
                    let pred_minus = self.forward_with_layers(u, y);
                    layer.bias[i] = orig;

                    let grad = (pred_plus - pred_minus) / (2.0 * eps);
                    layer.bias[i] -= learning_rate * 2.0 * error * grad;
                }
            }

            loss
        }

        /// Helper for gradient computation (uses current layer state)
        fn forward_with_layers(&self, u: &[f64], y: &[f64]) -> f64 {
            let b = self.branch_forward(u);
            let t = self.trunk_forward(y);
            let mut output = self.bias;
            for k in 0..self.config.latent_dim {
                output += b[k] * t[k];
            }
            output
        }

        /// Train on a batch using stochastic gradient descent
        pub fn train_batch(
            &mut self,
            u_batch: &Array2<f64>,
            y_batch: &Array2<f64>,
            targets: &Array1<f64>,
        ) -> f64 {
            let n = u_batch.nrows();
            let lr = self.config.learning_rate;
            let mut total_loss = 0.0;

            for i in 0..n {
                let u: Vec<f64> = u_batch.row(i).to_vec();
                let y: Vec<f64> = y_batch.row(i).to_vec();
                let loss = self.train_step(&u, &y, targets[i], lr);
                total_loss += loss;
            }

            total_loss / n as f64
        }

        /// Evaluate MSE on a dataset
        pub fn evaluate(
            &self,
            u_data: &Array2<f64>,
            y_data: &Array2<f64>,
            targets: &Array1<f64>,
        ) -> f64 {
            let predictions = self.predict_batch(u_data, y_data);
            let diff = &predictions - targets;
            diff.mapv(|x| x * x).mean().unwrap_or(0.0)
        }

        /// Save model to JSON file
        pub fn save(&self, path: &str) -> anyhow::Result<()> {
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
            Ok(())
        }

        /// Load model from JSON file
        pub fn load(path: &str) -> anyhow::Result<Self> {
            let json = std::fs::read_to_string(path)?;
            let model: Self = serde_json::from_str(&json)?;
            Ok(model)
        }

        /// Get total number of parameters
        pub fn num_parameters(&self) -> usize {
            let mut count = 1; // bias
            for layer in &self.branch_layers {
                count += layer.input_dim * layer.output_dim + layer.output_dim;
            }
            for layer in &self.trunk_layers {
                count += layer.input_dim * layer.output_dim + layer.output_dim;
            }
            count
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPTIMIZER MODULE
// ═══════════════════════════════════════════════════════════════════════════════

pub mod optim {
    use crate::data::DeepONetDataset;
    use crate::model::DeepONet;
    use indicatif::{ProgressBar, ProgressStyle};
    use rand::prelude::*;

    /// Training configuration
    pub struct TrainConfig {
        pub epochs: usize,
        pub batch_size: usize,
        pub learning_rate: f64,
        pub patience: usize,
        pub verbose: bool,
    }

    impl Default for TrainConfig {
        fn default() -> Self {
            Self {
                epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
                patience: 20,
                verbose: true,
            }
        }
    }

    /// Training results
    #[derive(Debug)]
    pub struct TrainResult {
        pub train_losses: Vec<f64>,
        pub val_losses: Vec<f64>,
        pub best_epoch: usize,
        pub best_val_loss: f64,
    }

    /// Train a DeepONet model
    pub fn train(
        model: &mut DeepONet,
        train_data: &DeepONetDataset,
        val_data: &DeepONetDataset,
        config: &TrainConfig,
    ) -> TrainResult {
        let mut rng = StdRng::seed_from_u64(42);
        let n_train = train_data.len();

        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;
        let mut patience_counter = 0;

        let pb = if config.verbose {
            let pb = ProgressBar::new(config.epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };

        for epoch in 0..config.epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n_train).collect();
            indices.shuffle(&mut rng);

            // Mini-batch training
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_train).step_by(config.batch_size) {
                let batch_end = (batch_start + config.batch_size).min(n_train);
                let batch_indices: Vec<usize> =
                    indices[batch_start..batch_end].to_vec();

                let (u_batch, y_batch, t_batch) = train_data.get_batch(&batch_indices);
                let loss = model.train_batch(&u_batch, &y_batch, &t_batch);
                epoch_loss += loss;
                n_batches += 1;
            }

            epoch_loss /= n_batches as f64;
            train_losses.push(epoch_loss);

            // Validation
            let val_loss = model.evaluate(
                &val_data.u_sensors,
                &val_data.y_queries,
                &val_data.targets,
            );
            val_losses.push(val_loss);

            // Early stopping check
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= config.patience {
                    if let Some(ref pb) = pb {
                        pb.finish_with_message(format!(
                            "Early stop at epoch {}. Best val: {:.6}",
                            epoch, best_val_loss
                        ));
                    }
                    break;
                }
            }

            if let Some(ref pb) = pb {
                pb.set_position(epoch as u64);
                pb.set_message(format!(
                    "train: {:.6} val: {:.6} best: {:.6}",
                    epoch_loss, val_loss, best_val_loss
                ));
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message(format!("Done. Best val: {:.6}", best_val_loss));
        }

        TrainResult {
            train_losses,
            val_losses,
            best_epoch,
            best_val_loss,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BACKTEST MODULE
// ═══════════════════════════════════════════════════════════════════════════════

pub mod backtest {
    use crate::api::Kline;
    use crate::model::DeepONet;
    use serde::{Deserialize, Serialize};

    /// Backtest configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestConfig {
        pub initial_capital: f64,
        pub max_position_pct: f64,
        pub transaction_cost: f64,
        pub signal_threshold: f64,
        pub stop_loss: f64,
        pub take_profit: f64,
        pub window_size: usize,
        pub forecast_horizon: usize,
        pub rebalance_freq: usize,
    }

    impl Default for BacktestConfig {
        fn default() -> Self {
            Self {
                initial_capital: 100_000.0,
                max_position_pct: 0.2,
                transaction_cost: 0.001,
                signal_threshold: 0.002,
                stop_loss: 0.03,
                take_profit: 0.05,
                window_size: 60,
                forecast_horizon: 12,
                rebalance_freq: 4,
            }
        }
    }

    /// Backtest result summary
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestResult {
        pub total_return: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub n_trades: usize,
        pub win_rate: f64,
        pub profit_factor: f64,
        pub benchmark_return: f64,
        pub equity_curve: Vec<f64>,
    }

    impl std::fmt::Display for BacktestResult {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "=== Backtest Results ===")?;
            writeln!(f, "Total Return:     {:.2}%", self.total_return * 100.0)?;
            writeln!(f, "Sharpe Ratio:     {:.3}", self.sharpe_ratio)?;
            writeln!(f, "Max Drawdown:     {:.2}%", self.max_drawdown * 100.0)?;
            writeln!(f, "Trades:           {}", self.n_trades)?;
            writeln!(f, "Win Rate:         {:.2}%", self.win_rate * 100.0)?;
            writeln!(f, "Profit Factor:    {:.3}", self.profit_factor)?;
            writeln!(f, "Benchmark Return: {:.2}%", self.benchmark_return * 100.0)?;
            Ok(())
        }
    }

    /// Backtesting engine
    pub struct BacktestEngine {
        pub config: BacktestConfig,
    }

    impl BacktestEngine {
        pub fn new(config: BacktestConfig) -> Self {
            Self { config }
        }

        /// Run backtest with DeepONet model on kline data
        pub fn run(&self, model: &DeepONet, klines: &[Kline]) -> BacktestResult {
            let n = klines.len();
            let ws = self.config.window_size;
            let fh = self.config.forecast_horizon;

            if n < ws + fh + 10 {
                return BacktestResult {
                    total_return: 0.0,
                    sharpe_ratio: 0.0,
                    max_drawdown: 0.0,
                    n_trades: 0,
                    win_rate: 0.0,
                    profit_factor: 0.0,
                    benchmark_return: 0.0,
                    equity_curve: vec![self.config.initial_capital],
                };
            }

            let mut capital = self.config.initial_capital;
            let mut position: f64 = 0.0;
            let mut entry_price = 0.0_f64;
            let mut equity_curve = vec![capital];
            let mut trades = Vec::new();

            let start_idx = ws;
            let end_idx = n - fh;

            for i in start_idx..end_idx {
                let price = klines[i].close;
                let portfolio_value = capital + position * price;
                equity_curve.push(portfolio_value);

                // Stop loss / take profit
                if position != 0.0 && entry_price > 0.0 {
                    let ret = (price - entry_price) / entry_price;
                    let effective_ret = if position > 0.0 { ret } else { -ret };

                    if effective_ret < -self.config.stop_loss
                        || effective_ret > self.config.take_profit
                    {
                        let pnl = position * (price - entry_price);
                        capital += position * price;
                        trades.push(pnl);
                        position = 0.0;
                    }
                }

                // Rebalance
                if (i - start_idx) % self.config.rebalance_freq == 0 {
                    let signal = self.generate_signal(model, klines, i);

                    if signal.abs() > self.config.signal_threshold {
                        let target_value = portfolio_value
                            * self.config.max_position_pct
                            * signal.signum();
                        let target_position = target_value / price;

                        let delta = target_position - position;
                        if (delta * price).abs() > 100.0 {
                            let cost = (delta * price).abs() * self.config.transaction_cost;
                            capital -= cost;

                            if position != 0.0
                                && position.signum() != target_position.signum()
                            {
                                let pnl = position * (price - entry_price);
                                trades.push(pnl);
                            }

                            capital += (position - target_position) * price;
                            position = target_position;
                            entry_price = price;
                        }
                    }
                }
            }

            // Close final position
            if position != 0.0 {
                let final_price = klines[end_idx - 1].close;
                let pnl = position * (final_price - entry_price);
                trades.push(pnl);
                capital += position * final_price;
            }

            equity_curve.push(capital);

            // Compute metrics
            let total_return =
                (equity_curve.last().unwrap_or(&self.config.initial_capital)
                    / self.config.initial_capital)
                    - 1.0;

            let benchmark_return = (klines[end_idx - 1].close / klines[start_idx].close) - 1.0;

            let returns: Vec<f64> = equity_curve
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let var_ret = returns
                .iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std_ret = var_ret.sqrt();

            let sharpe = if std_ret > 0.0 {
                mean_ret / std_ret * (365.0 * 24.0_f64).sqrt()
            } else {
                0.0
            };

            let mut peak = equity_curve[0];
            let mut max_dd = 0.0_f64;
            for &eq in &equity_curve {
                if eq > peak {
                    peak = eq;
                }
                let dd = (peak - eq) / peak;
                max_dd = max_dd.max(dd);
            }

            let wins: Vec<f64> = trades.iter().filter(|&&p| p > 0.0).cloned().collect();
            let losses: Vec<f64> = trades.iter().filter(|&&p| p <= 0.0).cloned().collect();
            let win_rate = if trades.is_empty() {
                0.0
            } else {
                wins.len() as f64 / trades.len() as f64
            };

            let total_wins: f64 = wins.iter().sum();
            let total_losses: f64 = losses.iter().map(|l| l.abs()).sum();
            let profit_factor = if total_losses > 0.0 {
                total_wins / total_losses
            } else if total_wins > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };

            BacktestResult {
                total_return,
                sharpe_ratio: sharpe,
                max_drawdown: max_dd,
                n_trades: trades.len(),
                win_rate,
                profit_factor,
                benchmark_return,
                equity_curve,
            }
        }

        /// Generate trading signal from DeepONet prediction
        fn generate_signal(
            &self,
            model: &DeepONet,
            klines: &[Kline],
            current_idx: usize,
        ) -> f64 {
            let ws = self.config.window_size;
            if current_idx < ws {
                return 0.0;
            }

            let window = &klines[current_idx - ws..current_idx];

            // Normalize
            let price_mean: f64 =
                window.iter().map(|k| k.close).sum::<f64>() / ws as f64;
            let price_std = (window
                .iter()
                .map(|k| (k.close - price_mean).powi(2))
                .sum::<f64>()
                / ws as f64)
                .sqrt()
                + 1e-8;

            // Create branch input
            let mut u = Vec::with_capacity(ws * 5);
            for k in window {
                u.push((k.open - price_mean) / price_std);
                u.push((k.high - price_mean) / price_std);
                u.push((k.low - price_mean) / price_std);
                u.push((k.close - price_mean) / price_std);
                u.push(k.volume.ln().max(0.0));
            }

            // Truncate or pad to match model input dim
            let model_input_dim = model.config.branch_input_dim;
            u.resize(model_input_dim, 0.0);

            // Predict at multiple horizons and average
            let fh = self.config.forecast_horizon;
            let mut total_pred = 0.0;
            let n_points = 3;

            for h in 1..=n_points {
                let offset = h as f64 * fh as f64 / n_points as f64 / fh as f64;
                let y = vec![offset; model.config.trunk_input_dim];
                total_pred += model.forward(&u, &y);
            }

            total_pred / n_points as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deeponet_forward() {
        let config = model::DeepONetConfig {
            branch_input_dim: 10,
            trunk_input_dim: 2,
            branch_hidden_dims: vec![16],
            trunk_hidden_dims: vec![16],
            latent_dim: 8,
            ..Default::default()
        };

        let model = model::DeepONet::new(config);
        let u = vec![0.1; 10];
        let y = vec![0.5, 0.3];

        let output = model.forward(&u, &y);
        assert!(output.is_finite(), "Output should be finite: {}", output);
    }

    #[test]
    fn test_layer_forward() {
        let layer = model::Layer::new(4, 8, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input, true);
        assert_eq!(output.len(), 8);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_black_scholes() {
        let price = data::black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(price > 0.0);
        assert!(price < 100.0);
        // ATM call with 1yr, 20% vol, should be roughly 10
        assert!((price - 10.45).abs() < 1.0);
    }

    #[test]
    fn test_synthetic_klines() {
        let klines = api::generate_synthetic_klines(100, 42);
        assert_eq!(klines.len(), 100);
        for k in &klines {
            assert!(k.close > 0.0);
            assert!(k.high >= k.low);
            assert!(k.volume > 0.0);
        }
    }

    #[test]
    fn test_prepare_crypto_data() {
        let klines = api::generate_synthetic_klines(200, 42);
        let dataset = data::prepare_crypto_data(&klines, 30, 10);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.u_sensors.ncols(), 150); // 30 * 5
        assert_eq!(dataset.y_queries.ncols(), 1);
    }

    #[test]
    fn test_option_dataset() {
        let dataset = data::generate_option_dataset(100, 20, 42);
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.y_queries.ncols(), 2);
        for i in 0..dataset.len() {
            assert!(dataset.targets[i] >= 0.0); // Option prices are non-negative
        }
    }

    #[test]
    fn test_model_save_load() {
        let config = model::DeepONetConfig {
            branch_input_dim: 5,
            trunk_input_dim: 1,
            branch_hidden_dims: vec![8],
            trunk_hidden_dims: vec![8],
            latent_dim: 4,
            ..Default::default()
        };

        let model = model::DeepONet::new(config);
        let u = vec![0.1; 5];
        let y = vec![0.5];
        let pred1 = model.forward(&u, &y);

        // Save and load
        let path = "/tmp/test_deeponet_model.json";
        model.save(path).unwrap();
        let loaded = model::DeepONet::load(path).unwrap();
        let pred2 = loaded.forward(&u, &y);

        assert!((pred1 - pred2).abs() < 1e-10);
        std::fs::remove_file(path).ok();
    }
}
