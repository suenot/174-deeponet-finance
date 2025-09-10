//! Option Pricing with DeepONet
//!
//! Demonstrates using DeepONet to learn the option pricing operator:
//!     vol surface → option prices
//!
//! Run: cargo run --example option_pricing

use deeponet_finance::{
    data::{black_scholes_call, generate_option_dataset},
    model::{DeepONet, DeepONetConfig},
    optim::{self, TrainConfig},
};

fn main() -> anyhow::Result<()> {
    println!("=== DeepONet for Option Pricing ===\n");

    // Generate training data
    let n_sensors = 20;
    println!("Generating option pricing dataset...");
    let dataset = generate_option_dataset(2000, n_sensors, 42);
    println!("  Samples: {}", dataset.len());
    println!("  Branch dim: {}", dataset.u_sensors.ncols());
    println!("  Trunk dim: {}", dataset.y_queries.ncols());

    let (train_data, val_data, test_data) = dataset.split(0.7, 0.15);

    // Build model
    let config = DeepONetConfig {
        branch_input_dim: n_sensors,
        trunk_input_dim: 2,
        branch_hidden_dims: vec![32, 16],
        trunk_hidden_dims: vec![16, 8],
        latent_dim: 16,
        learning_rate: 0.001,
        epochs: 50,
        batch_size: 32,
    };

    let mut model = DeepONet::new(config);
    println!("\nModel parameters: {}", model.num_parameters());

    // Train
    let train_config = TrainConfig {
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        patience: 15,
        verbose: true,
    };

    println!("\nTraining...");
    let result = optim::train(&mut model, &train_data, &val_data, &train_config);

    // Evaluate
    let test_mse = model.evaluate(&test_data.u_sensors, &test_data.y_queries, &test_data.targets);
    println!("\nTest MSE: {:.6}", test_mse);
    println!("Best epoch: {}, Best val loss: {:.6}", result.best_epoch, result.best_val_loss);

    // Compare a few predictions with Black-Scholes
    println!("\n--- Sample Predictions vs Black-Scholes ---");
    println!("{:<10} {:<10} {:<12} {:<12} {:<10}", "Strike", "Maturity", "BS Price", "DeepONet", "Error");
    println!("{}", "-".repeat(54));

    let base_vol = 0.25;
    let u: Vec<f64> = (0..n_sensors)
        .map(|i| base_vol + 0.01 * (i as f64 - n_sensors as f64 / 2.0) / n_sensors as f64)
        .collect();

    let test_cases = [(0.9, 0.5), (0.95, 1.0), (1.0, 0.25), (1.05, 0.5), (1.1, 1.0)];

    for &(k, t) in &test_cases {
        let bs = black_scholes_call(1.0, k, t, 0.02, base_vol);
        let pred = model.forward(&u, &[k, t]);
        let error = (pred - bs).abs();
        println!("{:<10.2} {:<10.2} {:<12.6} {:<12.6} {:<10.6}", k, t, bs, pred, error);
    }

    println!("\nDone!");
    Ok(())
}
