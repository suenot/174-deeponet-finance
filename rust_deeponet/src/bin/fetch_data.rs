//! Fetch market data from Bybit exchange for DeepONet training.
//!
//! Usage:
//!     cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 5000
//!     cargo run --bin fetch_data -- --symbol ETHUSDT --interval 240 --limit 2000 --output data/

use clap::Parser;
use colored::Colorize;
use deeponet_finance::api::{generate_synthetic_klines, BybitClient, Kline};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch market data from Bybit for DeepONet")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 5000)]
    limit: usize,

    /// Output directory for CSV files
    #[arg(short, long, default_value = "data")]
    output: PathBuf,

    /// Use synthetic data instead of live API
    #[arg(long, default_value_t = false)]
    synthetic: bool,
}

fn save_klines_csv(klines: &[Kline], path: &PathBuf) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;

    for k in klines {
        wtr.write_record(&[
            k.timestamp.to_string(),
            format!("{:.8}", k.open),
            format!("{:.8}", k.high),
            format!("{:.8}", k.low),
            format!("{:.8}", k.close),
            format!("{:.4}", k.volume),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn print_kline_summary(klines: &[Kline], symbol: &str) {
    if klines.is_empty() {
        println!("{}", "No data available".red());
        return;
    }

    let first = &klines[0];
    let last = klines.last().unwrap();

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    let price_min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let price_max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let price_mean = prices.iter().sum::<f64>() / prices.len() as f64;

    let vol_sum: f64 = volumes.iter().sum();
    let vol_mean = vol_sum / volumes.len() as f64;

    let total_return = (last.close / first.close - 1.0) * 100.0;

    println!("\n{}", "=".repeat(60).cyan());
    println!("{}", format!("  Market Data Summary: {}", symbol).cyan().bold());
    println!("{}", "=".repeat(60).cyan());
    println!("  Candles:     {}", klines.len());
    println!("  Start:       {}", first.datetime().format("%Y-%m-%d %H:%M"));
    println!("  End:         {}", last.datetime().format("%Y-%m-%d %H:%M"));
    println!("  Price Range: {:.2} - {:.2}", price_min, price_max);
    println!("  Mean Price:  {:.2}", price_mean);
    println!("  Total Vol:   {:.0}", vol_sum);
    println!("  Avg Vol:     {:.0}", vol_mean);

    let return_color = if total_return > 0.0 { "green" } else { "red" };
    println!(
        "  Return:      {}",
        format!("{:+.2}%", total_return).color(return_color)
    );
    println!("{}", "=".repeat(60).cyan());
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("{}", "DeepONet Finance - Data Fetcher".green().bold());
    println!("Symbol: {} | Interval: {} | Limit: {}", args.symbol, args.interval, args.limit);

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    let klines = if args.synthetic {
        println!("Using {} data...", "synthetic".yellow());
        generate_synthetic_klines(args.limit, 42)
    } else {
        println!("Fetching from {}...", "Bybit API".green());
        let client = BybitClient::new();

        match client.get_klines_paginated(&args.symbol, &args.interval, args.limit) {
            Ok(k) => k,
            Err(e) => {
                println!("{}: {}", "API Error".red(), e);
                println!("Falling back to synthetic data...");
                generate_synthetic_klines(args.limit, 42)
            }
        }
    };

    // Print summary
    print_kline_summary(&klines, &args.symbol);

    // Save to CSV
    let filename = format!("{}_{}.csv", args.symbol.to_lowercase(), args.interval);
    let filepath = args.output.join(&filename);
    save_klines_csv(&klines, &filepath)?;
    println!("\nSaved {} candles to {}", klines.len(), filepath.display());

    // Also save first 5 and last 5 records for quick inspection
    println!("\n{}", "First 5 candles:".yellow());
    for k in klines.iter().take(5) {
        println!(
            "  {} | O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
            k.datetime().format("%Y-%m-%d %H:%M"),
            k.open, k.high, k.low, k.close, k.volume
        );
    }

    println!("\n{}", "Last 5 candles:".yellow());
    for k in klines.iter().rev().take(5).collect::<Vec<_>>().iter().rev() {
        println!(
            "  {} | O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
            k.datetime().format("%Y-%m-%d %H:%M"),
            k.open, k.high, k.low, k.close, k.volume
        );
    }

    Ok(())
}
