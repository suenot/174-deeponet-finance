//! Train a DeepONet model for financial applications.
//!
//! Supports:
//! - Crypto price forecasting (using Bybit data)
//! - Option pricing operator learning
//! - Yield curve prediction
//!
//! Usage:
//!     cargo run --bin train -- --mode crypto --symbol BTCUSDT --epochs 100
//!     cargo run --bin train -- --mode option_pricing --epochs 200
//!     cargo run --bin train -- --mode crypto --synthetic --epochs 50

use clap::Parser;
use colored::Colorize;
use deeponet_finance::{
    api::{generate_synthetic_klines, BybitClient},
    data::{generate_option_dataset, prepare_crypto_data},
    model::{DeepONet, DeepONetConfig},
    optim::{self, TrainConfig},
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train DeepONet for financial applications")]
struct Args {
    /// Training mode
    #[arg(short, long, default_value = "crypto")]
    mode: String,

    /// Trading symbol for crypto mode
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Latent dimension
    #[arg(long, default_value_t = 32)]
    latent_dim: usize,

    /// Batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Window size for time series input
    #[arg(long, default_value_t = 30)]
    window_size: usize,

    /// Maximum forecast horizon
    #[arg(long, default_value_t = 12)]
    forecast_horizon: usize,

    /// Use synthetic data
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Output directory for model files
    #[arg(short, long, default_value = "models")]
    output: PathBuf,

    /// Early stopping patience
    #[arg(long, default_value_t = 20)]
    patience: usize,
}

fn train_crypto(args: &Args) -> anyhow::Result<()> {
    println!("{}", "Training DeepONet for Crypto Forecasting".green().bold());
    println!("Symbol: {} | Window: {} | Horizon: {}", args.symbol, args.window_size, args.forecast_horizon);

    // Fetch data
    let klines = if args.synthetic {
        println!("Using synthetic data...");
        generate_synthetic_klines(3000, 42)
    } else {
        println!("Fetching data from Bybit...");
        let client = BybitClient::new();
        match client.get_klines_paginated(&args.symbol, "60", 3000) {
            Ok(k) => k,
            Err(e) => {
                println!("API error: {}. Using synthetic data.", e);
                generate_synthetic_klines(3000, 42)
            }
        }
    };

    println!("Preparing dataset ({} candles)...", klines.len());
    let dataset = prepare_crypto_data(&klines, args.window_size, args.forecast_horizon);

    if dataset.is_empty() {
        anyhow::bail!("Dataset is empty. Need more data.");
    }

    let (train_data, val_data, test_data) = dataset.split(0.7, 0.15);
    println!(
        "Train: {} | Val: {} | Test: {}",
        train_data.len(),
        val_data.len(),
        test_data.len()
    );

    // Build model
    let branch_dim = args.window_size * 5; // OHLCV flattened
    let config = DeepONetConfig {
        branch_input_dim: branch_dim,
        trunk_input_dim: 1,
        branch_hidden_dims: vec![args.latent_dim * 2, args.latent_dim],
        trunk_hidden_dims: vec![args.latent_dim, args.latent_dim / 2],
        latent_dim: args.latent_dim,
        learning_rate: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
    };

    let mut model = DeepONet::new(config);
    println!("Model parameters: {}", model.num_parameters());

    // Train
    let train_config = TrainConfig {
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        patience: args.patience,
        verbose: true,
    };

    println!("\n{}", "Starting training...".yellow());
    let result = optim::train(&mut model, &train_data, &val_data, &train_config);

    // Evaluate on test set
    let test_mse = model.evaluate(&test_data.u_sensors, &test_data.y_queries, &test_data.targets);
    println!("\n{}", "Training Complete".green().bold());
    println!("Best Epoch:     {}", result.best_epoch);
    println!("Best Val Loss:  {:.6}", result.best_val_loss);
    println!("Test MSE:       {:.6}", test_mse);

    // Save model
    std::fs::create_dir_all(&args.output)?;
    let model_path = args.output.join("deeponet_crypto.json");
    model.save(model_path.to_str().unwrap())?;
    println!("\nModel saved to {}", model_path.display());

    Ok(())
}

fn train_option_pricing(args: &Args) -> anyhow::Result<()> {
    println!("{}", "Training DeepONet for Option Pricing".green().bold());

    // Generate synthetic option pricing data
    let n_sensors = 20; // Simplified vol surface
    let n_samples = 5000;

    println!("Generating {} option pricing samples...", n_samples);
    let dataset = generate_option_dataset(n_samples, n_sensors, 42);

    let (train_data, val_data, test_data) = dataset.split(0.7, 0.15);
    println!(
        "Train: {} | Val: {} | Test: {}",
        train_data.len(),
        val_data.len(),
        test_data.len()
    );

    // Build model
    let config = DeepONetConfig {
        branch_input_dim: n_sensors,
        trunk_input_dim: 2, // (moneyness, maturity)
        branch_hidden_dims: vec![64, 32],
        trunk_hidden_dims: vec![32, 16],
        latent_dim: args.latent_dim,
        learning_rate: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
    };

    let mut model = DeepONet::new(config);
    println!("Model parameters: {}", model.num_parameters());

    let train_config = TrainConfig {
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        patience: args.patience,
        verbose: true,
    };

    println!("\n{}", "Starting training...".yellow());
    let result = optim::train(&mut model, &train_data, &val_data, &train_config);

    let test_mse = model.evaluate(&test_data.u_sensors, &test_data.y_queries, &test_data.targets);
    println!("\n{}", "Training Complete".green().bold());
    println!("Best Epoch:     {}", result.best_epoch);
    println!("Best Val Loss:  {:.6}", result.best_val_loss);
    println!("Test MSE:       {:.6}", test_mse);

    // Save model
    std::fs::create_dir_all(&args.output)?;
    let model_path = args.output.join("deeponet_options.json");
    model.save(model_path.to_str().unwrap())?;
    println!("\nModel saved to {}", model_path.display());

    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    match args.mode.as_str() {
        "crypto" => train_crypto(&args)?,
        "option_pricing" | "options" => train_option_pricing(&args)?,
        _ => {
            println!("Unknown mode: {}. Use 'crypto' or 'option_pricing'.", args.mode);
        }
    }

    Ok(())
}
