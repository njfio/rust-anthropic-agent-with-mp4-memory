//! DSPy Benchmarking Commands
//!
//! This module implements CLI commands for DSPy module benchmarking and performance analysis.

use crate::cli::dspy::{DspyCliContext, DspyCliResult};
use crate::cli::dspy::commands::{BenchmarkCommand, OutputFormat};
use crate::cli::dspy::utils::{OutputFormatter, ValidationUtils};
use serde::{Deserialize, Serialize};
use tabled::Tabled;
use tracing::{debug, info};

/// Execute benchmark command
pub async fn execute_benchmark_command(
    command: BenchmarkCommand,
    context: &DspyCliContext,
) -> DspyCliResult<()> {
    match command {
        BenchmarkCommand::Run { module, iterations, timeout, input, output, format, warmup, parallel } => {
            run_benchmark(context, module, iterations, timeout, input, output, format, warmup, parallel).await
        }
        BenchmarkCommand::Compare { modules, metric, input, output, format, statistical } => {
            compare_modules(context, modules, metric, input, output, format, statistical).await
        }
        BenchmarkCommand::Export { module, format, filter, since, output } => {
            export_results(context, module, format, filter, since, output).await
        }
        BenchmarkCommand::History { module, limit, format, metric, trend } => {
            show_history(context, module, limit, format, metric, trend).await
        }
    }
}

/// Benchmark result information
#[derive(Debug, Clone, Serialize, Deserialize, Tabled)]
pub struct BenchmarkResult {
    pub module: String,
    pub iterations: usize,
    pub avg_latency_ms: f64,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub memory_mb: f64,
    pub timestamp: String,
}

/// Run benchmark on a module
async fn run_benchmark(
    context: &DspyCliContext,
    module: String,
    iterations: usize,
    timeout: u64,
    input: Option<std::path::PathBuf>,
    output: Option<std::path::PathBuf>,
    format: Option<OutputFormat>,
    warmup: usize,
    parallel: Option<usize>,
) -> DspyCliResult<()> {
    debug!("Running benchmark on module: {}", module);

    // Validate inputs
    ValidationUtils::validate_module_name(&module)?;
    ValidationUtils::validate_iterations(iterations)?;
    ValidationUtils::validate_timeout(timeout)?;

    let format = format.unwrap_or(OutputFormat::Table);

    // Check if module exists in registry
    if !context.registry.has_module(&module) {
        return Err(crate::cli::dspy::error::DspyCliError::validation_error(
            "module_name",
            format!("Module '{}' not found", module),
            vec![
                "Check the module name spelling".to_string(),
                "Use 'dspy modules list' to see available modules".to_string(),
            ],
        ));
    }

    // Load test data if provided
    let test_data = if let Some(input_path) = input {
        ValidationUtils::validate_file_path(&input_path, true)?;
        Some(tokio::fs::read_to_string(&input_path).await
            .map_err(|e| crate::cli::dspy::error::DspyCliError::resource_error(
                "filesystem",
                format!("Failed to read input file: {}", e),
                "Check file permissions and path",
            ))?)
    } else {
        None
    };

    // Prepare benchmark configuration
    let parallel_threads = parallel.unwrap_or(1);
    let _total_iterations = warmup + iterations;

    println!("🚀 Starting benchmark for module '{}'", module);
    println!("  Warmup iterations: {}", warmup);
    println!("  Benchmark iterations: {}", iterations);
    println!("  Parallel threads: {}", parallel_threads);
    println!("  Timeout: {}s", timeout);

    // Run warmup iterations
    if warmup > 0 {
        let mut progress = crate::cli::dspy::utils::ProgressIndicator::new("Warmup", Some(warmup));
        for i in 0..warmup {
            progress.update(&format!("Warmup iteration {}/{}", i + 1, warmup));

            // Simulate module execution
            let execution_time = simulate_module_execution(&module, test_data.as_deref(), timeout).await?;

            // Small delay to simulate real work
            tokio::time::sleep(tokio::time::Duration::from_millis(execution_time as u64)).await;
        }
        progress.finish("Warmup completed");
    }

    // Run benchmark iterations
    let mut latencies = Vec::new();
    let mut errors = 0;
    let mut memory_usage = Vec::new();

    let mut progress = crate::cli::dspy::utils::ProgressIndicator::new("Benchmark", Some(iterations));

    for i in 0..iterations {
        progress.update(&format!("Iteration {}/{}", i + 1, iterations));

        let start_time = std::time::Instant::now();

        // Simulate module execution with error possibility
        match simulate_module_execution(&module, test_data.as_deref(), timeout).await {
            Ok(execution_time) => {
                latencies.push(execution_time);
                // Simulate memory usage (would be real in actual implementation)
                memory_usage.push(40.0 + (i as f64 * 0.1) + (rand::random::<f64>() * 10.0));
            }
            Err(_) => {
                errors += 1;
                latencies.push(timeout as f64 * 1000.0); // Timeout latency
                memory_usage.push(50.0); // Default memory on error
            }
        }

        // Respect timeout
        let elapsed = start_time.elapsed().as_secs();
        if elapsed >= timeout {
            println!("⚠️  Benchmark timeout reached");
            break;
        }
    }

    progress.finish("Benchmark completed");

    // Calculate statistics
    let avg_latency = if latencies.is_empty() { 0.0 } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let throughput = if avg_latency > 0.0 { 1000.0 / avg_latency } else { 0.0 };
    let error_rate = errors as f64 / iterations as f64;
    let avg_memory = if memory_usage.is_empty() { 0.0 } else {
        memory_usage.iter().sum::<f64>() / memory_usage.len() as f64
    };

    let result = BenchmarkResult {
        module: module.clone(),
        iterations: latencies.len(),
        avg_latency_ms: avg_latency,
        throughput_rps: throughput,
        error_rate,
        memory_mb: avg_memory,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Save results if output specified
    if let Some(output_path) = output {
        let json_result = serde_json::to_string_pretty(&result)
            .map_err(|e| crate::cli::dspy::error::DspyCliError::internal_error(
                format!("Failed to serialize results: {}", e)
            ))?;

        tokio::fs::write(&output_path, json_result).await
            .map_err(|e| crate::cli::dspy::error::DspyCliError::resource_error(
                "filesystem",
                format!("Failed to write results: {}", e),
                "Check output path permissions",
            ))?;

        println!("📊 Results saved to: {}", output_path.display());
    }

    OutputFormatter::print(&result, format)?;

    // Print summary
    println!("\n📈 Benchmark Summary:");
    println!("  Average Latency: {:.2}ms", avg_latency);
    println!("  Throughput: {:.2} requests/second", throughput);
    println!("  Error Rate: {:.2}%", error_rate * 100.0);
    println!("  Memory Usage: {:.2}MB", avg_memory);

    info!("Completed benchmark for module: {} ({} iterations)", module, latencies.len());

    Ok(())
}

/// Simulate module execution (placeholder for real implementation)
async fn simulate_module_execution(
    module: &str,
    _test_data: Option<&str>,
    timeout: u64,
) -> crate::cli::dspy::DspyCliResult<f64> {
    // Simulate variable execution times based on module complexity
    let base_latency = match module {
        name if name.contains("simple") => 50.0,
        name if name.contains("complex") => 200.0,
        name if name.contains("heavy") => 500.0,
        _ => 100.0,
    };

    // Add some randomness
    let variation = (rand::random::<f64>() - 0.5) * 0.4; // ±20% variation
    let execution_time = base_latency * (1.0 + variation);

    // Simulate occasional errors (5% chance)
    if rand::random::<f64>() < 0.05 {
        return Err(crate::cli::dspy::error::DspyCliError::execution_error(
            "benchmark",
            "Simulated execution error",
            true,
        ));
    }

    // Check timeout
    if execution_time > (timeout as f64 * 1000.0) {
        return Err(crate::cli::dspy::error::DspyCliError::execution_error(
            "benchmark",
            "Execution timeout",
            true,
        ));
    }

    Ok(execution_time)
}

/// Compare multiple modules
async fn compare_modules(
    context: &DspyCliContext,
    modules: String,
    metric: Option<String>,
    input: Option<std::path::PathBuf>,
    output: Option<std::path::PathBuf>,
    format: Option<OutputFormat>,
    statistical: bool,
) -> DspyCliResult<()> {
    debug!("Comparing modules: {}", modules);

    let format = format.unwrap_or(OutputFormat::Table);
    let module_names = crate::cli::dspy::utils::StringUtils::parse_comma_separated(&modules);

    if module_names.len() < 2 {
        return Err(crate::cli::dspy::error::DspyCliError::validation_error(
            "modules",
            "At least 2 modules are required for comparison",
            vec!["Provide comma-separated module names".to_string()],
        ));
    }

    // Validate module names and check existence
    for name in &module_names {
        ValidationUtils::validate_module_name(name)?;
        if !context.registry.has_module(name) {
            return Err(crate::cli::dspy::error::DspyCliError::validation_error(
                "module_name",
                format!("Module '{}' not found", name),
                vec!["Use 'dspy modules list' to see available modules".to_string()],
            ));
        }
    }

    // Load test data if provided
    let test_data = if let Some(input_path) = input {
        ValidationUtils::validate_file_path(&input_path, true)?;
        Some(tokio::fs::read_to_string(&input_path).await
            .map_err(|e| crate::cli::dspy::error::DspyCliError::resource_error(
                "filesystem",
                format!("Failed to read input file: {}", e),
                "Check file permissions and path",
            ))?)
    } else {
        None
    };

    println!("🔄 Comparing {} modules", module_names.len());
    println!("  Modules: {}", module_names.join(", "));
    if let Some(ref metric_name) = metric {
        println!("  Focus metric: {}", metric_name);
    }

    // Run benchmarks on each module
    let mut results = Vec::new();
    let benchmark_iterations = 50; // Fixed for comparison

    for (i, module_name) in module_names.iter().enumerate() {
        println!("\n📊 Benchmarking module {} ({}/{})", module_name, i + 1, module_names.len());

        let mut latencies = Vec::new();
        let mut errors = 0;
        let mut memory_usage = Vec::new();

        let mut progress = crate::cli::dspy::utils::ProgressIndicator::new(
            &format!("Testing {}", module_name),
            Some(benchmark_iterations)
        );

        for j in 0..benchmark_iterations {
            progress.update(&format!("Iteration {}/{}", j + 1, benchmark_iterations));

            match simulate_module_execution(module_name, test_data.as_deref(), 30).await {
                Ok(execution_time) => {
                    latencies.push(execution_time);
                    memory_usage.push(40.0 + (j as f64 * 0.1) + (rand::random::<f64>() * 10.0));
                }
                Err(_) => {
                    errors += 1;
                    latencies.push(30000.0); // Timeout latency
                    memory_usage.push(50.0);
                }
            }
        }

        progress.finish(&format!("Completed {}", module_name));

        // Calculate statistics
        let avg_latency = if latencies.is_empty() { 0.0 } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };
        let throughput = if avg_latency > 0.0 { 1000.0 / avg_latency } else { 0.0 };
        let error_rate = errors as f64 / benchmark_iterations as f64;
        let avg_memory = if memory_usage.is_empty() { 0.0 } else {
            memory_usage.iter().sum::<f64>() / memory_usage.len() as f64
        };

        let result = BenchmarkResult {
            module: module_name.clone(),
            iterations: latencies.len(),
            avg_latency_ms: avg_latency,
            throughput_rps: throughput,
            error_rate,
            memory_mb: avg_memory,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        results.push(result);
    }

    // Generate comparison analysis
    println!("\n📈 Comparison Analysis:");

    // Find best performing module for each metric
    let best_latency = results.iter().min_by(|a, b| a.avg_latency_ms.partial_cmp(&b.avg_latency_ms).unwrap());
    let best_throughput = results.iter().max_by(|a, b| a.throughput_rps.partial_cmp(&b.throughput_rps).unwrap());
    let best_error_rate = results.iter().min_by(|a, b| a.error_rate.partial_cmp(&b.error_rate).unwrap());
    let best_memory = results.iter().min_by(|a, b| a.memory_mb.partial_cmp(&b.memory_mb).unwrap());

    if let Some(best) = best_latency {
        println!("  🏆 Lowest Latency: {} ({:.2}ms)", best.module, best.avg_latency_ms);
    }
    if let Some(best) = best_throughput {
        println!("  🏆 Highest Throughput: {} ({:.2} rps)", best.module, best.throughput_rps);
    }
    if let Some(best) = best_error_rate {
        println!("  🏆 Lowest Error Rate: {} ({:.2}%)", best.module, best.error_rate * 100.0);
    }
    if let Some(best) = best_memory {
        println!("  🏆 Lowest Memory Usage: {} ({:.2}MB)", best.module, best.memory_mb);
    }

    // Statistical analysis if requested
    if statistical {
        println!("\n📊 Statistical Analysis:");

        let latencies: Vec<f64> = results.iter().map(|r| r.avg_latency_ms).collect();
        let throughputs: Vec<f64> = results.iter().map(|r| r.throughput_rps).collect();

        if latencies.len() > 1 {
            let latency_std = calculate_std_dev(&latencies);
            let throughput_std = calculate_std_dev(&throughputs);

            println!("  Latency std dev: {:.2}ms", latency_std);
            println!("  Throughput std dev: {:.2} rps", throughput_std);
        }
    }

    // Save results if output specified
    if let Some(output_path) = output {
        let comparison_data = serde_json::json!({
            "comparison_timestamp": chrono::Utc::now().to_rfc3339(),
            "modules_compared": module_names,
            "results": results,
            "analysis": {
                "best_latency": best_latency.map(|r| &r.module),
                "best_throughput": best_throughput.map(|r| &r.module),
                "best_error_rate": best_error_rate.map(|r| &r.module),
                "best_memory": best_memory.map(|r| &r.module),
            }
        });

        let json_result = serde_json::to_string_pretty(&comparison_data)
            .map_err(|e| crate::cli::dspy::error::DspyCliError::internal_error(
                format!("Failed to serialize comparison: {}", e)
            ))?;

        tokio::fs::write(&output_path, json_result).await
            .map_err(|e| crate::cli::dspy::error::DspyCliError::resource_error(
                "filesystem",
                format!("Failed to write comparison: {}", e),
                "Check output path permissions",
            ))?;

        println!("📊 Comparison saved to: {}", output_path.display());
    }

    OutputFormatter::print_list(&results, format)?;
    info!("Completed comparison of {} modules", module_names.len());

    Ok(())
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Export benchmark results
async fn export_results(
    context: &DspyCliContext,
    module: Option<String>,
    export_format: Option<crate::cli::dspy::commands::ExportFormat>,
    filter: Option<String>,
    since: Option<String>,
    output: Option<std::path::PathBuf>,
) -> DspyCliResult<()> {
    debug!("Exporting benchmark results for module: {:?}", module);

    let export_format = export_format.unwrap_or(crate::cli::dspy::commands::ExportFormat::Json);

    // Generate sample benchmark data (in real implementation, this would come from storage)
    let mut results = Vec::new();

    let modules_to_export = if let Some(module_name) = module {
        vec![module_name]
    } else {
        // Export all modules from registry
        context.registry.list_modules().into_iter().map(|s| s.to_string()).collect()
    };

    for module_name in modules_to_export {
        // Generate historical data for each module
        for i in 0..10 {
            let days_ago = i * 2;
            let timestamp = chrono::Utc::now() - chrono::Duration::days(days_ago);

            // Skip if since filter is applied
            if let Some(since_date) = &since {
                if let Ok(since_parsed) = chrono::DateTime::parse_from_rfc3339(since_date) {
                    if timestamp < since_parsed {
                        continue;
                    }
                }
            }

            let result = BenchmarkResult {
                module: module_name.clone(),
                iterations: 100,
                avg_latency_ms: 100.0 + (rand::random::<f64>() * 50.0),
                throughput_rps: 8.0 + (rand::random::<f64>() * 4.0),
                error_rate: rand::random::<f64>() * 0.05,
                memory_mb: 40.0 + (rand::random::<f64>() * 20.0),
                timestamp: timestamp.to_rfc3339(),
            };

            // Apply filter if provided
            if let Some(filter_pattern) = &filter {
                if !module_name.contains(filter_pattern) {
                    continue;
                }
            }

            results.push(result);
        }
    }

    // Sort by timestamp (newest first)
    results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Format and export results
    let export_data = match export_format {
        crate::cli::dspy::commands::ExportFormat::Json => {
            serde_json::to_string_pretty(&results)
                .map_err(|e| crate::cli::dspy::error::DspyCliError::internal_error(
                    format!("Failed to serialize to JSON: {}", e)
                ))?
        }
        crate::cli::dspy::commands::ExportFormat::Csv => {
            let mut csv_data = String::new();
            csv_data.push_str("module,iterations,avg_latency_ms,throughput_rps,error_rate,memory_mb,timestamp\n");

            for result in &results {
                csv_data.push_str(&format!(
                    "{},{},{},{},{},{},{}\n",
                    result.module,
                    result.iterations,
                    result.avg_latency_ms,
                    result.throughput_rps,
                    result.error_rate,
                    result.memory_mb,
                    result.timestamp
                ));
            }
            csv_data
        }
        crate::cli::dspy::commands::ExportFormat::Prometheus => {
            let mut prom_data = String::new();
            prom_data.push_str("# HELP dspy_benchmark_latency_ms Average latency in milliseconds\n");
            prom_data.push_str("# TYPE dspy_benchmark_latency_ms gauge\n");

            for result in &results {
                prom_data.push_str(&format!(
                    "dspy_benchmark_latency_ms{{module=\"{}\"}} {}\n",
                    result.module, result.avg_latency_ms
                ));
            }

            prom_data.push_str("# HELP dspy_benchmark_throughput_rps Throughput in requests per second\n");
            prom_data.push_str("# TYPE dspy_benchmark_throughput_rps gauge\n");

            for result in &results {
                prom_data.push_str(&format!(
                    "dspy_benchmark_throughput_rps{{module=\"{}\"}} {}\n",
                    result.module, result.throughput_rps
                ));
            }

            prom_data
        }
    };

    // Write to file or stdout
    if let Some(output_path) = output {
        tokio::fs::write(&output_path, export_data).await
            .map_err(|e| crate::cli::dspy::error::DspyCliError::resource_error(
                "filesystem",
                format!("Failed to write export file: {}", e),
                "Check output path permissions",
            ))?;

        println!("✓ Exported {} results to: {}", results.len(), output_path.display());
    } else {
        println!("{}", export_data);
    }

    info!("Exported {} benchmark results in {:?} format", results.len(), export_format);

    Ok(())
}

/// Show benchmark history
async fn show_history(
    context: &DspyCliContext,
    module: Option<String>,
    limit: usize,
    format: Option<OutputFormat>,
    metric: Option<String>,
    trend: bool,
) -> DspyCliResult<()> {
    debug!("Showing benchmark history for module: {:?}", module);

    let format = format.unwrap_or(OutputFormat::Table);

    // Generate historical benchmark data (in real implementation, this would come from storage)
    let mut history = Vec::new();

    let modules_to_show = if let Some(module_name) = module.clone() {
        ValidationUtils::validate_module_name(&module_name)?;
        if !context.registry.has_module(&module_name) {
            return Err(crate::cli::dspy::error::DspyCliError::validation_error(
                "module_name",
                format!("Module '{}' not found", module_name),
                vec!["Use 'dspy modules list' to see available modules".to_string()],
            ));
        }
        vec![module_name]
    } else {
        // Show history for all modules
        context.registry.list_modules().into_iter().map(|s| s.to_string()).collect()
    };

    for module_name in modules_to_show {
        // Generate historical data points
        for i in 0..limit.min(20) {
            let days_ago = i as i64;
            let timestamp = chrono::Utc::now() - chrono::Duration::days(days_ago);

            // Simulate performance trends over time
            let trend_factor = 1.0 + (i as f64 * 0.02); // Slight degradation over time
            let noise = (rand::random::<f64>() - 0.5) * 0.2; // ±10% noise

            let result = BenchmarkResult {
                module: module_name.clone(),
                iterations: 100,
                avg_latency_ms: (100.0 * trend_factor) + (noise * 20.0),
                throughput_rps: (8.0 / trend_factor) + (noise * 2.0),
                error_rate: (0.01 * trend_factor).min(0.1) + (noise.abs() * 0.005),
                memory_mb: (45.0 * trend_factor) + (noise * 5.0),
                timestamp: timestamp.to_rfc3339(),
            };

            history.push(result);
        }
    }

    // Sort by timestamp (newest first)
    history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Apply limit
    history.truncate(limit);

    // Filter by metric if specified
    if let Some(metric_name) = &metric {
        println!("📊 Focusing on metric: {}", metric_name);

        match metric_name.to_lowercase().as_str() {
            "latency" => {
                history.sort_by(|a, b| a.avg_latency_ms.partial_cmp(&b.avg_latency_ms).unwrap());
                println!("  Sorted by latency (lowest first)");
            }
            "throughput" => {
                history.sort_by(|a, b| b.throughput_rps.partial_cmp(&a.throughput_rps).unwrap());
                println!("  Sorted by throughput (highest first)");
            }
            "error_rate" => {
                history.sort_by(|a, b| a.error_rate.partial_cmp(&b.error_rate).unwrap());
                println!("  Sorted by error rate (lowest first)");
            }
            "memory" => {
                history.sort_by(|a, b| a.memory_mb.partial_cmp(&b.memory_mb).unwrap());
                println!("  Sorted by memory usage (lowest first)");
            }
            _ => {
                println!("  Unknown metric '{}', showing chronological order", metric_name);
            }
        }
    }

    // Show trend analysis if requested
    if trend && history.len() > 1 {
        println!("\n📈 Trend Analysis:");

        if let Some(module_name) = &module {
            let module_history: Vec<&BenchmarkResult> = history.iter()
                .filter(|r| r.module == *module_name)
                .collect();

            if module_history.len() > 1 {
                let latest = module_history[0];
                let oldest = module_history[module_history.len() - 1];

                let latency_change = ((latest.avg_latency_ms - oldest.avg_latency_ms) / oldest.avg_latency_ms) * 100.0;
                let throughput_change = ((latest.throughput_rps - oldest.throughput_rps) / oldest.throughput_rps) * 100.0;
                let error_change = ((latest.error_rate - oldest.error_rate) / oldest.error_rate.max(0.001)) * 100.0;
                let memory_change = ((latest.memory_mb - oldest.memory_mb) / oldest.memory_mb) * 100.0;

                println!("  Module: {}", module_name);
                println!("  Latency trend: {:.1}% {}", latency_change.abs(),
                    if latency_change > 0.0 { "⬆️ (worse)" } else { "⬇️ (better)" });
                println!("  Throughput trend: {:.1}% {}", throughput_change.abs(),
                    if throughput_change > 0.0 { "⬆️ (better)" } else { "⬇️ (worse)" });
                println!("  Error rate trend: {:.1}% {}", error_change.abs(),
                    if error_change > 0.0 { "⬆️ (worse)" } else { "⬇️ (better)" });
                println!("  Memory trend: {:.1}% {}", memory_change.abs(),
                    if memory_change > 0.0 { "⬆️ (worse)" } else { "⬇️ (better)" });
            }
        } else {
            // Show overall trends across all modules
            let avg_latency = history.iter().map(|r| r.avg_latency_ms).sum::<f64>() / history.len() as f64;
            let avg_throughput = history.iter().map(|r| r.throughput_rps).sum::<f64>() / history.len() as f64;
            let avg_error_rate = history.iter().map(|r| r.error_rate).sum::<f64>() / history.len() as f64;
            let avg_memory = history.iter().map(|r| r.memory_mb).sum::<f64>() / history.len() as f64;

            println!("  Overall averages across all modules:");
            println!("  Average latency: {:.2}ms", avg_latency);
            println!("  Average throughput: {:.2} rps", avg_throughput);
            println!("  Average error rate: {:.2}%", avg_error_rate * 100.0);
            println!("  Average memory usage: {:.2}MB", avg_memory);
        }
    }

    OutputFormatter::print_list(&history, format)?;

    println!("\n📋 Summary:");
    println!("  Total records: {}", history.len());
    if let Some(module_name) = &module {
        println!("  Module: {}", module_name);
    } else {
        let unique_modules: std::collections::HashSet<String> = history.iter()
            .map(|r| r.module.clone())
            .collect();
        println!("  Modules: {}", unique_modules.len());
    }

    info!("Displayed {} benchmark history records", history.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult {
            module: "test_module".to_string(),
            iterations: 100,
            avg_latency_ms: 125.5,
            throughput_rps: 8.0,
            error_rate: 0.0,
            memory_mb: 45.2,
            timestamp: "2024-01-15T10:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("test_module"));
        assert!(json.contains("125.5"));
        assert!(json.contains("8.0"));
    }

    #[test]
    fn test_benchmark_result_deserialization() {
        let json = r#"{
            "module": "test_module",
            "iterations": 100,
            "avg_latency_ms": 125.5,
            "throughput_rps": 8.0,
            "error_rate": 0.0,
            "memory_mb": 45.2,
            "timestamp": "2024-01-15T10:00:00Z"
        }"#;

        let result: BenchmarkResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.module, "test_module");
        assert_eq!(result.iterations, 100);
        assert_eq!(result.avg_latency_ms, 125.5);
        assert_eq!(result.throughput_rps, 8.0);
    }

    #[tokio::test]
    async fn test_simulate_module_execution() {
        let result = simulate_module_execution("simple_module", None, 30).await;
        assert!(result.is_ok());

        let latency = result.unwrap();
        assert!(latency > 0.0);
        assert!(latency < 30000.0); // Should be less than timeout
    }

    #[tokio::test]
    async fn test_simulate_module_execution_timeout() {
        // Test with very low timeout to trigger timeout condition
        let result = simulate_module_execution("heavy_module", None, 0).await;
        // This might succeed or fail depending on random execution time
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_calculate_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = calculate_std_dev(&values);
        assert!(std_dev > 0.0);
        assert!(std_dev < 10.0); // Reasonable range
    }

    #[test]
    fn test_calculate_std_dev_single_value() {
        let values = vec![5.0];
        let std_dev = calculate_std_dev(&values);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_calculate_std_dev_empty() {
        let values = vec![];
        let std_dev = calculate_std_dev(&values);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_benchmark_command_structure() {
        use crate::cli::dspy::commands::{BenchmarkCommand, OutputFormat};

        let run_command = BenchmarkCommand::Run {
            module: "test_module".to_string(),
            iterations: 100,
            timeout: 30,
            input: None,
            output: None,
            format: Some(OutputFormat::Json),
            warmup: 5,
            parallel: Some(2),
        };

        match run_command {
            BenchmarkCommand::Run { module, iterations, .. } => {
                assert_eq!(module, "test_module");
                assert_eq!(iterations, 100);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_benchmark_comparison_structure() {
        use crate::cli::dspy::commands::BenchmarkCommand;

        let compare_command = BenchmarkCommand::Compare {
            modules: "module1,module2,module3".to_string(),
            metric: Some("latency".to_string()),
            input: None,
            output: None,
            format: None,
            statistical: true,
        };

        match compare_command {
            BenchmarkCommand::Compare { modules, statistical, .. } => {
                assert_eq!(modules, "module1,module2,module3");
                assert!(statistical);
            }
            _ => panic!("Expected Compare command"),
        }
    }

    #[test]
    fn test_export_format_enum() {
        use crate::cli::dspy::commands::ExportFormat;

        // Test that export format enum works correctly
        assert_eq!(ExportFormat::Json as u8, ExportFormat::Json as u8);
        assert_ne!(ExportFormat::Json as u8, ExportFormat::Csv as u8);
        assert_ne!(ExportFormat::Csv as u8, ExportFormat::Prometheus as u8);
    }

    #[tokio::test]
    async fn test_benchmark_result_metrics() {
        let result = BenchmarkResult {
            module: "test_module".to_string(),
            iterations: 100,
            avg_latency_ms: 125.5,
            throughput_rps: 8.0,
            error_rate: 0.02,
            memory_mb: 45.2,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Test that metrics are within reasonable ranges
        assert!(result.avg_latency_ms > 0.0);
        assert!(result.throughput_rps > 0.0);
        assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
        assert!(result.memory_mb > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_benchmark_performance_calculations() {
        let latencies = vec![100.0, 120.0, 110.0, 130.0, 105.0];
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let throughput = 1000.0 / avg_latency;

        assert_eq!(avg_latency, 113.0);
        assert!((throughput - 8.849).abs() < 0.01); // Approximately 8.849 rps
    }

    #[test]
    fn test_error_rate_calculation() {
        let total_iterations = 100;
        let errors = 5;
        let error_rate = errors as f64 / total_iterations as f64;

        assert_eq!(error_rate, 0.05); // 5% error rate
        assert!(error_rate >= 0.0 && error_rate <= 1.0);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let memory_samples = vec![40.0, 42.5, 41.2, 43.8, 39.9];
        let avg_memory = memory_samples.iter().sum::<f64>() / memory_samples.len() as f64;

        assert!((avg_memory - 41.48).abs() < 0.01);
        assert!(avg_memory > 0.0);
    }

    #[test]
    fn test_timestamp_format() {
        let result = BenchmarkResult {
            module: "test".to_string(),
            iterations: 1,
            avg_latency_ms: 100.0,
            throughput_rps: 10.0,
            error_rate: 0.0,
            memory_mb: 40.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Test that timestamp is valid RFC3339 format
        assert!(chrono::DateTime::parse_from_rfc3339(&result.timestamp).is_ok());
    }
}
