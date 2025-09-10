//! Run predictions with a trained DeepONet model.
//!
//! Usage:
//!     cargo run --bin predict -- --model models/deeponet_crypto.json --symbol BTCUSDT
//!     cargo run --bin predict -- --model models/deeponet_options.json --mode options
//!     cargo run --bin predict -- --model models/deeponet_crypto.json --synthetic

use clap::Parser;
use colored::Colorize;
use deeponet_finance::{
    api::{generate_synthetic_klines, BybitClient},
    backtest::{BacktestConfig, BacktestEngine},
    data::black_scholes_call,
    model::DeepONet,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "predict", about = "Run predictions with trained DeepONet")]
struct Args {
    /// Path to trained model file
    #[arg(short, long, default_value = "models/deeponet_crypto.json")]
    model: PathBuf,

    /// Prediction mode: 'crypto' or 'options'
    #[arg(long, default_value = "crypto")]
    mode: String,

    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Use synthetic data
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Run backtest instead of single prediction
    #[arg(long, default_value_t = false)]
    backtest: bool,

    /// Window size (must match training)
    #[arg(long, default_value_t = 30)]
    window_size: usize,
}

fn predict_crypto(args: &Args) -> anyhow::Result<()> {
    println!("{}", "DeepONet Crypto Prediction".green().bold());

    // Load model
    let model = if args.model.exists() {
        println!("Loading model from {}...", args.model.display());
        DeepONet::load(args.model.to_str().unwrap())?
    } else {
        println!("{}: Model not found at {}.", "Warning".yellow(), args.model.display());
        println!("Using a fresh (untrained) model for demo purposes.\n");
        let config = deeponet_finance::model::DeepONetConfig {
            branch_input_dim: args.window_size * 5,
            trunk_input_dim: 1,
            branch_hidden_dims: vec![64, 32],
            trunk_hidden_dims: vec![32, 16],
            latent_dim: 32,
            ..Default::default()
        };
        DeepONet::new(config)
    };

    println!("Model parameters: {}", model.num_parameters());

    // Fetch data
    let klines = if args.synthetic {
        println!("Using synthetic data...");
        generate_synthetic_klines(1000, 123)
    } else {
        println!("Fetching {} from Bybit...", args.symbol);
        let client = BybitClient::new();
        match client.get_klines_paginated(&args.symbol, "60", 1000) {
            Ok(k) => k,
            Err(e) => {
                println!("API error: {}. Using synthetic data.", e);
                generate_synthetic_klines(1000, 123)
            }
        }
    };

    if args.backtest {
        // Run full backtest
        println!("\n{}", "Running Backtest...".yellow());
        let bt_config = BacktestConfig {
            window_size: args.window_size,
            ..Default::default()
        };
        let engine = BacktestEngine::new(bt_config);
        let result = engine.run(&model, &klines);
        println!("{}", result);
    } else {
        // Single prediction for the latest window
        println!("\n{}", "Latest Prediction:".cyan().bold());

        if klines.len() < args.window_size {
            println!("Not enough data for prediction (need {} candles).", args.window_size);
            return Ok(());
        }

        let window = &klines[klines.len() - args.window_size..];
        let last_price = window.last().unwrap().close;

        // Normalize
        let price_mean: f64 = window.iter().map(|k| k.close).sum::<f64>() / args.window_size as f64;
        let price_std = (window
            .iter()
            .map(|k| (k.close - price_mean).powi(2))
            .sum::<f64>()
            / args.window_size as f64)
            .sqrt()
            + 1e-8;

        let mut u = Vec::with_capacity(args.window_size * 5);
        for k in window {
            u.push((k.open - price_mean) / price_std);
            u.push((k.high - price_mean) / price_std);
            u.push((k.low - price_mean) / price_std);
            u.push((k.close - price_mean) / price_std);
            u.push(k.volume.ln().max(0.0));
        }

        // Pad to model input dim
        u.resize(model.config.branch_input_dim, 0.0);

        println!("\n  Current Price:  {:.2}", last_price);
        println!("  Price Mean:     {:.2}", price_mean);
        println!("  Price Std:      {:.2}", price_std);

        // Predict at multiple horizons
        println!("\n  {:<12} {:<15} {:<15} {:<10}", "Horizon", "Pred Return", "Pred Price", "Direction");
        println!("  {}", "-".repeat(52));

        let horizons = [1, 3, 6, 12, 24];
        for &h in &horizons {
            let y = vec![h as f64 / 24.0]; // Normalize to [0, 1]
            // Pad y to model trunk dim
            let mut y_full = y.clone();
            y_full.resize(model.config.trunk_input_dim, 0.0);

            let pred_normalized = model.forward(&u, &y_full);
            let pred_return = pred_normalized * price_std / last_price;
            let pred_price = last_price * (1.0 + pred_return);

            let direction = if pred_return > 0.002 {
                "LONG".green()
            } else if pred_return < -0.002 {
                "SHORT".red()
            } else {
                "FLAT".yellow()
            };

            println!(
                "  {:<12} {:<15.4}% {:<15.2} {}",
                format!("{}h", h),
                pred_return * 100.0,
                pred_price,
                direction
            );
        }

        println!("\n  Last candle: {}", window.last().unwrap().datetime().format("%Y-%m-%d %H:%M UTC"));
    }

    Ok(())
}

fn predict_options(args: &Args) -> anyhow::Result<()> {
    println!("{}", "DeepONet Option Pricing Predictions".green().bold());

    let model = if args.model.exists() {
        DeepONet::load(args.model.to_str().unwrap())?
    } else {
        println!("Using untrained model for demo.\n");
        let config = deeponet_finance::model::DeepONetConfig {
            branch_input_dim: 20,
            trunk_input_dim: 2,
            branch_hidden_dims: vec![64, 32],
            trunk_hidden_dims: vec![32, 16],
            latent_dim: 32,
            ..Default::default()
        };
        DeepONet::new(config)
    };

    println!("Model parameters: {}\n", model.num_parameters());

    // Demo: compare DeepONet predictions with Black-Scholes
    let base_vol = 0.25;
    let r = 0.02;
    let s = 1.0;

    // Create vol surface sensor values
    let n_sensors = model.config.branch_input_dim;
    let u: Vec<f64> = (0..n_sensors)
        .map(|i| base_vol + 0.02 * (i as f64 / n_sensors as f64 - 0.5))
        .collect();

    println!(
        "  {:<12} {:<12} {:<15} {:<15} {:<12}",
        "Strike", "Maturity", "BS Price", "DeepONet", "Error"
    );
    println!("  {}", "-".repeat(66));

    let strikes = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2];
    let maturities = [0.25, 0.5, 1.0];

    for &t_mat in &maturities {
        for &k in &strikes {
            let bs_price = black_scholes_call(s, k, t_mat, r, base_vol);
            let y = vec![k, t_mat];
            let deeponet_price = model.forward(&u, &y);
            let error = (deeponet_price - bs_price).abs();

            println!(
                "  {:<12.2} {:<12.2} {:<15.6} {:<15.6} {:<12.6}",
                k, t_mat, bs_price, deeponet_price, error
            );
        }
        println!();
    }

    println!(
        "  Note: An untrained model will show large errors. Train first with:\n  \
         cargo run --bin train -- --mode option_pricing --epochs 200"
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.mode.as_str() {
        "crypto" => predict_crypto(&args)?,
        "options" | "option_pricing" => predict_options(&args)?,
        _ => println!("Unknown mode: {}. Use 'crypto' or 'options'.", args.mode),
    }

    Ok(())
}
