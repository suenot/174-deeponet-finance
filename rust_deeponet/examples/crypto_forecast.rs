//! Crypto Forecasting with DeepONet
//!
//! Demonstrates using DeepONet for cryptocurrency price forecasting
//! using synthetic data that mimics Bybit exchange candles.
//!
//! Run: cargo run --example crypto_forecast

use deeponet_finance::{
    api::generate_synthetic_klines,
    backtest::{BacktestConfig, BacktestEngine},
    data::prepare_crypto_data,
    model::{DeepONet, DeepONetConfig},
    optim::{self, TrainConfig},
};

fn main() -> anyhow::Result<()> {
    println!("=== DeepONet for Crypto Forecasting ===\n");

    let window_size = 30;
    let forecast_horizon = 12;

    // Generate synthetic crypto data
    println!("Generating synthetic BTC-like data...");
    let klines = generate_synthetic_klines(2000, 42);
    println!("  Candles: {}", klines.len());
    println!(
        "  Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.close).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.close).fold(f64::NEG_INFINITY, f64::max)
    );

    // Prepare dataset
    println!("\nPreparing DeepONet dataset...");
    let dataset = prepare_crypto_data(&klines, window_size, forecast_horizon);
    println!("  Samples: {}", dataset.len());
    println!("  Branch dim: {} ({}x5 OHLCV)", dataset.u_sensors.ncols(), window_size);
    println!("  Trunk dim: {} (time offset)", dataset.y_queries.ncols());

    let (train_data, val_data, _test_data) = dataset.split(0.7, 0.15);

    // Build model
    let config = DeepONetConfig {
        branch_input_dim: window_size * 5,
        trunk_input_dim: 1,
        branch_hidden_dims: vec![64, 32],
        trunk_hidden_dims: vec![16, 8],
        latent_dim: 16,
        learning_rate: 0.001,
        epochs: 30,
        batch_size: 32,
    };

    let mut model = DeepONet::new(config);
    println!("\nModel parameters: {}", model.num_parameters());

    // Train
    let train_config = TrainConfig {
        epochs: 30,
        batch_size: 32,
        learning_rate: 0.001,
        patience: 10,
        verbose: true,
    };

    println!("\nTraining...");
    let result = optim::train(&mut model, &train_data, &val_data, &train_config);
    println!("Best val loss: {:.6} at epoch {}", result.best_val_loss, result.best_epoch);

    // Run backtest
    println!("\n--- Backtesting on held-out data ---");
    let test_klines = generate_synthetic_klines(500, 999);

    let bt_config = BacktestConfig {
        window_size,
        forecast_horizon,
        ..Default::default()
    };

    let engine = BacktestEngine::new(bt_config);
    let bt_result = engine.run(&model, &test_klines);
    println!("{}", bt_result);

    // Show some predictions
    println!("--- Sample Predictions ---");
    println!("{:<10} {:<15} {:<15}", "Horizon", "Pred (norm)", "Direction");
    println!("{}", "-".repeat(40));

    // Use the last window of test data
    let window = &test_klines[test_klines.len() - window_size..];
    let price_mean: f64 = window.iter().map(|k| k.close).sum::<f64>() / window_size as f64;
    let price_std = (window
        .iter()
        .map(|k| (k.close - price_mean).powi(2))
        .sum::<f64>()
        / window_size as f64)
        .sqrt()
        + 1e-8;

    let mut u = Vec::with_capacity(window_size * 5);
    for k in window {
        u.push((k.open - price_mean) / price_std);
        u.push((k.high - price_mean) / price_std);
        u.push((k.low - price_mean) / price_std);
        u.push((k.close - price_mean) / price_std);
        u.push(k.volume.ln().max(0.0));
    }

    for h in [1, 3, 6, 12] {
        let y = vec![h as f64 / forecast_horizon as f64];
        let pred = model.forward(&u, &y);
        let direction = if pred > 0.01 { "UP" } else if pred < -0.01 { "DOWN" } else { "FLAT" };
        println!("{:<10} {:<15.6} {:<15}", format!("{}h", h), pred, direction);
    }

    println!("\nDone!");
    Ok(())
}
