//! Yield Curve Prediction with DeepONet
//!
//! Demonstrates using DeepONet to learn the operator:
//!     macro economic conditions → yield curve
//!
//! The branch network encodes macro indicators (GDP, inflation, etc.)
//! and the trunk network encodes bond maturity, producing the yield.
//!
//! Run: cargo run --example yield_curve

use deeponet_finance::{
    data::DeepONetDataset,
    model::{DeepONet, DeepONetConfig},
    optim::{self, TrainConfig},
};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};

/// Generate synthetic yield curve data using Nelson-Siegel model
fn generate_yield_data(n_samples: usize, n_macro: usize, seed: u64) -> DeepONetDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let level_dist = Uniform::new(0.01, 0.06);
    let slope_dist = Uniform::new(-0.01, 0.03);
    let curv_dist = Uniform::new(-0.02, 0.02);
    let noise = Normal::new(0.0, 0.001).unwrap();
    let maturity_dist = Uniform::new(0.25, 30.0);

    let mut u_list = Vec::new();
    let mut y_list = Vec::new();
    let mut t_list = Vec::new();

    for _ in 0..n_samples {
        let level: f64 = level_dist.sample(&mut rng);
        let slope: f64 = slope_dist.sample(&mut rng);
        let curvature: f64 = curv_dist.sample(&mut rng);

        // Macro features (level, slope, curvature + noise features)
        let mut macro_vec = vec![level, slope, curvature];
        let macro_noise = Normal::new(0.0, 0.02).unwrap();
        for _ in 3..n_macro {
            macro_vec.push(macro_noise.sample(&mut rng));
        }

        // Random maturity query
        let maturity: f64 = maturity_dist.sample(&mut rng);

        // Nelson-Siegel yield
        let tau = 2.0;
        let factor1 = (1.0 - (-maturity / tau).exp()) / (maturity / tau);
        let factor2 = factor1 - (-maturity / tau).exp();

        let yield_val = level + slope * factor1 + curvature * factor2 + noise.sample(&mut rng);

        u_list.push(macro_vec);
        y_list.push(vec![maturity / 30.0]); // Normalize maturity to [0, 1]
        t_list.push(yield_val);
    }

    let n = u_list.len();
    let u_dim = n_macro;

    let mut u_sensors = Array2::zeros((n, u_dim));
    let mut y_queries = Array2::zeros((n, 1));
    let mut targets = Array1::zeros(n);

    for i in 0..n {
        for (j, &val) in u_list[i].iter().enumerate() {
            u_sensors[[i, j]] = val;
        }
        y_queries[[i, 0]] = y_list[i][0];
        targets[i] = t_list[i];
    }

    DeepONetDataset {
        u_sensors,
        y_queries,
        targets,
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== DeepONet for Yield Curve Prediction ===\n");

    let n_macro = 8; // Number of macro features

    // Generate data
    println!("Generating yield curve dataset...");
    let dataset = generate_yield_data(3000, n_macro, 42);
    println!("  Samples: {}", dataset.len());
    println!("  Branch dim: {} (macro features)", dataset.u_sensors.ncols());
    println!("  Trunk dim: {} (maturity)", dataset.y_queries.ncols());
    println!(
        "  Yield range: [{:.4}, {:.4}]",
        dataset.targets.iter().cloned().fold(f64::INFINITY, f64::min),
        dataset.targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    let (train_data, val_data, test_data) = dataset.split(0.7, 0.15);

    // Build model
    let config = DeepONetConfig {
        branch_input_dim: n_macro,
        trunk_input_dim: 1,
        branch_hidden_dims: vec![32, 16],
        trunk_hidden_dims: vec![16, 8],
        latent_dim: 16,
        learning_rate: 0.001,
        epochs: 80,
        batch_size: 32,
    };

    let mut model = DeepONet::new(config);
    println!("\nModel parameters: {}", model.num_parameters());

    // Train
    let train_config = TrainConfig {
        epochs: 80,
        batch_size: 32,
        learning_rate: 0.001,
        patience: 15,
        verbose: true,
    };

    println!("\nTraining...");
    let result = optim::train(&mut model, &train_data, &val_data, &train_config);
    println!("Best val loss: {:.6} at epoch {}", result.best_val_loss, result.best_epoch);

    // Evaluate on test set
    let test_mse = model.evaluate(&test_data.u_sensors, &test_data.y_queries, &test_data.targets);
    println!("\nTest MSE: {:.6}", test_mse);
    println!("Test RMSE: {:.4} ({:.1} bps)", test_mse.sqrt(), test_mse.sqrt() * 10000.0);

    // Plot yield curve for a sample scenario
    println!("\n--- Sample Yield Curve Prediction ---");
    println!("Macro scenario: Level=3%, Slope=1.5%, Curvature=0.5%\n");

    let u_sample = vec![0.03, 0.015, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0];

    println!("{:<12} {:<12} {:<12} {:<12}", "Maturity", "Predicted", "True (NS)", "Error (bps)");
    println!("{}", "-".repeat(48));

    let maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
    for &mat in &maturities {
        let y = vec![mat / 30.0]; // Normalized maturity

        let pred = model.forward(&u_sample, &y);

        // True Nelson-Siegel
        let tau = 2.0;
        let f1 = (1.0 - (-mat / tau).exp()) / (mat / tau);
        let f2 = f1 - (-mat / tau).exp();
        let true_yield = 0.03 + 0.015 * f1 + 0.005 * f2;

        let error_bps = (pred - true_yield).abs() * 10000.0;

        println!(
            "{:<12.1}y {:<12.4}% {:<12.4}% {:<12.1}",
            mat,
            pred * 100.0,
            true_yield * 100.0,
            error_bps
        );
    }

    // Compare different economic scenarios
    println!("\n--- Multiple Scenarios ---\n");

    let scenarios = [
        ("Normal", vec![0.03, 0.015, 0.005]),
        ("Inverted", vec![0.05, -0.01, 0.003]),
        ("Low Rate", vec![0.01, 0.02, -0.005]),
        ("High Rate", vec![0.06, 0.005, 0.01]),
    ];

    println!("{:<12} | 1Y     | 5Y     | 10Y    | 30Y", "Scenario");
    println!("{}", "-".repeat(58));

    for (name, factors) in &scenarios {
        let mut u = factors.clone();
        u.resize(n_macro, 0.0);

        let yields: Vec<f64> = [1.0, 5.0, 10.0, 30.0]
            .iter()
            .map(|&m| model.forward(&u, &[m / 30.0]) * 100.0)
            .collect();

        println!(
            "{:<12} | {:.2}%  | {:.2}%  | {:.2}%  | {:.2}%",
            name, yields[0], yields[1], yields[2], yields[3]
        );
    }

    println!("\nDone!");
    Ok(())
}
