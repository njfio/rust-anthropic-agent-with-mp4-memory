//! Performance benchmarking and optimization for DSPy framework
//!
//! This module provides comprehensive benchmarking capabilities for DSPy modules,
//! performance optimization tools, and system monitoring utilities.

use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub timeout_seconds: u64,
    pub memory_monitoring: bool,
    pub cpu_monitoring: bool,
    pub concurrent_requests: usize,
    pub enable_profiling: bool,
    pub output_format: BenchmarkOutputFormat,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 100,
            timeout_seconds: 300,
            memory_monitoring: true,
            cpu_monitoring: true,
            concurrent_requests: 1,
            enable_profiling: false,
            output_format: BenchmarkOutputFormat::Json,
        }
    }
}

/// Benchmark output formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkOutputFormat {
    Json,
    Csv,
    Table,
    Prometheus,
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub module_name: String,
    pub config: BenchmarkConfig,
    pub performance_metrics: PerformanceMetrics,
    pub resource_usage: ResourceUsage,
    pub error_analysis: ErrorAnalysis,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub average_latency_ms: f64,
    pub median_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub latency_distribution: Vec<LatencyBucket>,
}

/// Latency distribution bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBucket {
    pub upper_bound_ms: f64,
    pub count: usize,
    pub percentage: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub peak_cpu_percent: f64,
    pub average_cpu_percent: f64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub disk_reads: u64,
    pub disk_writes: u64,
    pub gc_collections: usize,
    pub gc_time_ms: f64,
}

/// Error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub error_types: HashMap<String, usize>,
    pub error_rate_by_time: Vec<ErrorRatePoint>,
    pub most_common_errors: Vec<ErrorFrequency>,
    pub error_patterns: Vec<String>,
}

/// Error rate at specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRatePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_rate: f64,
    pub error_count: usize,
}

/// Error frequency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorFrequency {
    pub error_type: String,
    pub count: usize,
    pub percentage: f64,
    pub sample_message: String,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub priority: OptimizationPriority,
    pub description: String,
    pub expected_improvement: String,
    pub implementation_effort: ImplementationEffort,
    pub code_example: Option<String>,
}

/// Categories of optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationCategory {
    Caching,
    Batching,
    Parallelization,
    MemoryOptimization,
    NetworkOptimization,
    AlgorithmOptimization,
    ConfigurationTuning,
    ResourcePooling,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImplementationEffort {
    Trivial,    // < 1 hour
    Easy,       // 1-4 hours
    Medium,     // 1-2 days
    Hard,       // 1-2 weeks
    Complex,    // > 2 weeks
}

/// Benchmark suite for DSPy modules
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResults>,
}

impl BenchmarkSuite {
    /// Create new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Benchmark a module with given inputs
    pub async fn benchmark_module<I, O, M>(
        &mut self,
        module: &M,
        test_inputs: Vec<I>,
    ) -> DspyResult<BenchmarkResults>
    where
        M: Module<Input = I, Output = O>,
        I: Clone + Send + Sync,
        O: Send + Sync,
    {
        info!("Starting benchmark for module: {}", module.name());

        if test_inputs.is_empty() {
            return Err(DspyError::configuration("test_inputs", "No test inputs provided"));
        }

        let start_time = Instant::now();
        let mut latencies = Vec::new();
        let mut errors = Vec::new();
        let mut resource_samples = Vec::new();

        // Warmup phase
        info!("Running warmup phase with {} iterations", self.config.warmup_iterations);
        for i in 0..self.config.warmup_iterations {
            let input = &test_inputs[i % test_inputs.len()];
            let _ = self.run_single_request(module, input.clone()).await;
        }

        // Benchmark phase
        info!("Running benchmark phase with {} iterations", self.config.benchmark_iterations);
        for i in 0..self.config.benchmark_iterations {
            let input = &test_inputs[i % test_inputs.len()];

            // Sample resource usage before request
            let pre_resources = self.sample_resource_usage().await;

            // Run request with timeout
            let request_start = Instant::now();
            let result = timeout(
                Duration::from_secs(self.config.timeout_seconds),
                self.run_single_request(module, input.clone())
            ).await;

            let latency = request_start.elapsed();
            latencies.push(latency.as_millis() as f64);

            // Sample resource usage after request
            let post_resources = self.sample_resource_usage().await;
            resource_samples.push((pre_resources, post_resources));

            // Handle result
            match result {
                Ok(Ok(_)) => {
                    // Success - no action needed
                }
                Ok(Err(e)) => {
                    errors.push(format!("{}", e));
                }
                Err(_) => {
                    errors.push("Request timeout".to_string());
                }
            }

            // Progress reporting
            if (i + 1) % (self.config.benchmark_iterations / 10).max(1) == 0 {
                debug!("Completed {}/{} benchmark iterations", i + 1, self.config.benchmark_iterations);
            }
        }

        let total_time = start_time.elapsed();

        // Analyze results
        let performance_metrics = self.analyze_performance(&latencies, &errors, total_time);
        let resource_usage = self.analyze_resource_usage(&resource_samples);
        let error_analysis = self.analyze_errors(&errors);
        let optimization_suggestions = self.generate_optimization_suggestions(
            &performance_metrics,
            &resource_usage,
            &error_analysis,
        );

        let results = BenchmarkResults {
            module_name: module.name().to_string(),
            config: self.config.clone(),
            performance_metrics,
            resource_usage,
            error_analysis,
            optimization_suggestions,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        self.results.push(results.clone());

        info!("Benchmark completed for module: {}", module.name());
        Ok(results)
    }

    /// Run a single request against the module
    async fn run_single_request<I, O, M>(
        &self,
        module: &M,
        input: I,
    ) -> DspyResult<O>
    where
        M: Module<Input = I, Output = O>,
    {
        module.forward(input).await
    }

    /// Sample current resource usage
    async fn sample_resource_usage(&self) -> ResourceSample {
        // In a real implementation, this would use system APIs to get actual resource usage
        ResourceSample {
            memory_mb: 100.0, // Mock value
            cpu_percent: 25.0, // Mock value
            timestamp: Instant::now(),
        }
    }

    /// Analyze performance metrics
    fn analyze_performance(
        &self,
        latencies: &[f64],
        errors: &[String],
        total_time: Duration,
    ) -> PerformanceMetrics {
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total_requests = self.config.benchmark_iterations;
        let successful_requests = total_requests - errors.len();
        let failed_requests = errors.len();

        let average_latency_ms = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let median_latency_ms = if !sorted_latencies.is_empty() {
            sorted_latencies[sorted_latencies.len() / 2]
        } else {
            0.0
        };

        let p95_latency_ms = if !sorted_latencies.is_empty() {
            sorted_latencies[(sorted_latencies.len() as f64 * 0.95) as usize]
        } else {
            0.0
        };

        let p99_latency_ms = if !sorted_latencies.is_empty() {
            sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize]
        } else {
            0.0
        };

        let min_latency_ms = sorted_latencies.first().copied().unwrap_or(0.0);
        let max_latency_ms = sorted_latencies.last().copied().unwrap_or(0.0);

        let throughput_rps = successful_requests as f64 / total_time.as_secs_f64();
        let error_rate = failed_requests as f64 / total_requests as f64;

        let latency_distribution = self.calculate_latency_distribution(&sorted_latencies);

        PerformanceMetrics {
            total_requests,
            successful_requests,
            failed_requests,
            average_latency_ms,
            median_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            min_latency_ms,
            max_latency_ms,
            throughput_rps,
            error_rate,
            latency_distribution,
        }
    }

    /// Calculate latency distribution
    fn calculate_latency_distribution(&self, sorted_latencies: &[f64]) -> Vec<LatencyBucket> {
        if sorted_latencies.is_empty() {
            return Vec::new();
        }

        let max_latency = sorted_latencies.last().copied().unwrap_or(0.0);
        let bucket_size = max_latency / 10.0; // 10 buckets
        let mut buckets = Vec::new();
        let mut previous_count = 0;

        for i in 1..=10 {
            let upper_bound = bucket_size * i as f64;
            let cumulative_count = sorted_latencies.iter()
                .filter(|&&latency| latency <= upper_bound)
                .count();

            // Calculate count for this specific bucket (not cumulative)
            let bucket_count = cumulative_count - previous_count;
            let percentage = bucket_count as f64 / sorted_latencies.len() as f64 * 100.0;

            buckets.push(LatencyBucket {
                upper_bound_ms: upper_bound,
                count: bucket_count,
                percentage,
            });

            previous_count = cumulative_count;
        }

        buckets
    }

    /// Analyze resource usage
    fn analyze_resource_usage(&self, samples: &[(ResourceSample, ResourceSample)]) -> ResourceUsage {
        if samples.is_empty() {
            return ResourceUsage {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                peak_cpu_percent: 0.0,
                average_cpu_percent: 0.0,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                disk_reads: 0,
                disk_writes: 0,
                gc_collections: 0,
                gc_time_ms: 0.0,
            };
        }

        let memory_values: Vec<f64> = samples.iter()
            .map(|(_, post)| post.memory_mb)
            .collect();
        let cpu_values: Vec<f64> = samples.iter()
            .map(|(_, post)| post.cpu_percent)
            .collect();

        ResourceUsage {
            peak_memory_mb: memory_values.iter().fold(0.0, |a, &b| a.max(b)),
            average_memory_mb: memory_values.iter().sum::<f64>() / memory_values.len() as f64,
            peak_cpu_percent: cpu_values.iter().fold(0.0, |a, &b| a.max(b)),
            average_cpu_percent: cpu_values.iter().sum::<f64>() / cpu_values.len() as f64,
            network_bytes_sent: 1024 * 100, // Mock value
            network_bytes_received: 1024 * 200, // Mock value
            disk_reads: 10, // Mock value
            disk_writes: 5, // Mock value
            gc_collections: 2, // Mock value
            gc_time_ms: 10.0, // Mock value
        }
    }

    /// Analyze errors
    fn analyze_errors(&self, errors: &[String]) -> ErrorAnalysis {
        let mut error_types = HashMap::new();

        for error in errors {
            let error_type = self.classify_error(error);
            *error_types.entry(error_type).or_insert(0) += 1;
        }

        let most_common_errors: Vec<ErrorFrequency> = error_types.iter()
            .map(|(error_type, &count)| {
                let percentage = count as f64 / errors.len() as f64 * 100.0;
                ErrorFrequency {
                    error_type: error_type.clone(),
                    count,
                    percentage,
                    sample_message: errors.iter()
                        .find(|e| self.classify_error(e) == *error_type)
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect();

        ErrorAnalysis {
            error_types,
            error_rate_by_time: Vec::new(), // Would be populated in real implementation
            most_common_errors,
            error_patterns: Vec::new(), // Would be populated in real implementation
        }
    }

    /// Classify error type
    fn classify_error(&self, error: &str) -> String {
        if error.contains("timeout") {
            "Timeout".to_string()
        } else if error.contains("network") || error.contains("connection") {
            "Network".to_string()
        } else if error.contains("memory") || error.contains("allocation") {
            "Memory".to_string()
        } else if error.contains("validation") || error.contains("invalid") {
            "Validation".to_string()
        } else {
            "Other".to_string()
        }
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        performance: &PerformanceMetrics,
        resources: &ResourceUsage,
        errors: &ErrorAnalysis,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // High latency suggestions
        if performance.average_latency_ms > 150.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Caching,
                priority: OptimizationPriority::High,
                description: "High average latency detected. Consider implementing caching.".to_string(),
                expected_improvement: "30-50% latency reduction".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                code_example: Some("module.enable_caching(CacheConfig::default())".to_string()),
            });
        }

        // High memory usage suggestions
        if resources.peak_memory_mb > 500.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::MemoryOptimization,
                priority: OptimizationPriority::Medium,
                description: "High memory usage detected. Consider memory optimization.".to_string(),
                expected_improvement: "20-40% memory reduction".to_string(),
                implementation_effort: ImplementationEffort::Hard,
                code_example: Some("Use streaming processing for large inputs".to_string()),
            });
        }

        // High error rate suggestions
        if performance.error_rate > 0.05 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::ConfigurationTuning,
                priority: OptimizationPriority::Critical,
                description: "High error rate detected. Review configuration and error handling.".to_string(),
                expected_improvement: "Reduce error rate to < 1%".to_string(),
                implementation_effort: ImplementationEffort::Easy,
                code_example: Some("Increase timeout and add retry logic".to_string()),
            });
        }

        // Low throughput suggestions
        if performance.throughput_rps < 1.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Parallelization,
                priority: OptimizationPriority::High,
                description: "Low throughput detected. Consider parallel processing.".to_string(),
                expected_improvement: "2-5x throughput increase".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                code_example: Some("Use batch processing or parallel execution".to_string()),
            });
        }

        suggestions
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &[BenchmarkResults] {
        &self.results
    }

    /// Export results in specified format
    pub fn export_results(&self, format: BenchmarkOutputFormat) -> DspyResult<String> {
        match format {
            BenchmarkOutputFormat::Json => {
                serde_json::to_string_pretty(&self.results)
                    .map_err(|e| DspyError::serialization("benchmark_results", &e.to_string()))
            }
            BenchmarkOutputFormat::Csv => {
                self.export_csv()
            }
            BenchmarkOutputFormat::Table => {
                Ok(self.export_table())
            }
            BenchmarkOutputFormat::Prometheus => {
                Ok(self.export_prometheus())
            }
        }
    }

    /// Export results as CSV
    fn export_csv(&self) -> DspyResult<String> {
        let mut csv = String::new();
        csv.push_str("module_name,avg_latency_ms,p95_latency_ms,throughput_rps,error_rate,peak_memory_mb\n");

        for result in &self.results {
            csv.push_str(&format!(
                "{},{:.2},{:.2},{:.2},{:.4},{:.2}\n",
                result.module_name,
                result.performance_metrics.average_latency_ms,
                result.performance_metrics.p95_latency_ms,
                result.performance_metrics.throughput_rps,
                result.performance_metrics.error_rate,
                result.resource_usage.peak_memory_mb
            ));
        }

        Ok(csv)
    }

    /// Export results as table
    fn export_table(&self) -> String {
        let mut table = String::new();
        table.push_str("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
        table.push_str("│ Module Name         │ Avg Lat(ms)│ P95 Lat(ms)│ RPS         │ Error Rate  │ Peak Mem(MB)│\n");
        table.push_str("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");

        for result in &self.results {
            table.push_str(&format!(
                "│ {:<19} │ {:>11.2} │ {:>11.2} │ {:>11.2} │ {:>11.4} │ {:>11.2} │\n",
                result.module_name,
                result.performance_metrics.average_latency_ms,
                result.performance_metrics.p95_latency_ms,
                result.performance_metrics.throughput_rps,
                result.performance_metrics.error_rate,
                result.resource_usage.peak_memory_mb
            ));
        }

        table.push_str("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");
        table
    }

    /// Export results in Prometheus format
    fn export_prometheus(&self) -> String {
        let mut prometheus = String::new();

        for result in &self.results {
            let labels = format!("module=\"{}\"", result.module_name);

            prometheus.push_str(&format!(
                "dspy_latency_average_ms{{{}}} {:.2}\n",
                labels, result.performance_metrics.average_latency_ms
            ));
            prometheus.push_str(&format!(
                "dspy_latency_p95_ms{{{}}} {:.2}\n",
                labels, result.performance_metrics.p95_latency_ms
            ));
            prometheus.push_str(&format!(
                "dspy_throughput_rps{{{}}} {:.2}\n",
                labels, result.performance_metrics.throughput_rps
            ));
            prometheus.push_str(&format!(
                "dspy_error_rate{{{}}} {:.4}\n",
                labels, result.performance_metrics.error_rate
            ));
            prometheus.push_str(&format!(
                "dspy_memory_peak_mb{{{}}} {:.2}\n",
                labels, result.resource_usage.peak_memory_mb
            ));
        }

        prometheus
    }
}

/// Resource usage sample
#[derive(Debug, Clone)]
struct ResourceSample {
    memory_mb: f64,
    cpu_percent: f64,
    timestamp: Instant,
}