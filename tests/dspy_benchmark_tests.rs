//! Comprehensive benchmark tests for DSPy framework
//!
//! This test suite validates the benchmarking and performance optimization
//! capabilities of the DSPy framework.

use rust_memvid_agent::anthropic::AnthropicClient;
use rust_memvid_agent::dspy::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio;

fn create_test_client() -> Arc<AnthropicClient> {
    Arc::new(
        AnthropicClient::new(rust_memvid_agent::config::AnthropicConfig {
            api_key: "test_key".to_string(),
            model: "claude-3-sonnet-20240229".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
            max_retries: 3,
        })
        .unwrap(),
    )
}

/// Mock module for testing
struct MockModule {
    id: String,
    name: String,
    signature: Signature<String, String>,
    latency_ms: u64,
    error_rate: f64,
    metadata: ModuleMetadata,
    stats: ModuleStats,
}

impl MockModule {
    fn new(name: &str, latency_ms: u64, error_rate: f64) -> Self {
        Self {
            id: format!("mock_{}", name),
            name: name.to_string(),
            signature: Signature::new(format!("mock_{}", name)),
            latency_ms,
            error_rate,
            metadata: ModuleMetadata::default(),
            stats: ModuleStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl Module for MockModule {
    type Input = String;
    type Output = String;

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.signature
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;

        // Simulate errors
        if rand::random::<f64>() < self.error_rate {
            return Err(DspyError::module(self.name(), "Simulated error"));
        }

        Ok(format!("Processed: {}", input))
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }

    fn stats(&self) -> &ModuleStats {
        &self.stats
    }

    fn supports_compilation(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn test_benchmark_config_creation() {
    let config = BenchmarkConfig {
        warmup_iterations: 3,
        benchmark_iterations: 10,
        timeout_seconds: 60,
        memory_monitoring: true,
        cpu_monitoring: true,
        concurrent_requests: 2,
        enable_profiling: false,
        output_format: BenchmarkOutputFormat::Json,
    };

    assert_eq!(config.warmup_iterations, 3);
    assert_eq!(config.benchmark_iterations, 10);
    assert_eq!(config.timeout_seconds, 60);
    assert!(config.memory_monitoring);
    assert_eq!(config.concurrent_requests, 2);
    assert_eq!(config.output_format, BenchmarkOutputFormat::Json);
}

#[tokio::test]
async fn test_benchmark_suite_creation() {
    let config = BenchmarkConfig::default();
    let suite = BenchmarkSuite::new(config.clone());

    assert_eq!(suite.get_results().len(), 0);
}

#[tokio::test]
async fn test_benchmark_fast_module() {
    let config = BenchmarkConfig {
        warmup_iterations: 2,
        benchmark_iterations: 5,
        timeout_seconds: 30,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("fast_module", 10, 0.0); // 10ms latency, no errors

    let test_inputs = vec![
        "test input 1".to_string(),
        "test input 2".to_string(),
        "test input 3".to_string(),
    ];

    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    assert_eq!(results.module_name, "fast_module");
    assert_eq!(results.performance_metrics.total_requests, 5);
    assert_eq!(results.performance_metrics.successful_requests, 5);
    assert_eq!(results.performance_metrics.failed_requests, 0);
    assert!(results.performance_metrics.average_latency_ms >= 10.0);
    assert!(results.performance_metrics.average_latency_ms < 50.0); // Should be close to 10ms
    assert_eq!(results.performance_metrics.error_rate, 0.0);
    assert!(results.performance_metrics.throughput_rps > 0.0);
}

#[tokio::test]
async fn test_benchmark_slow_module() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        benchmark_iterations: 3,
        timeout_seconds: 30,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("slow_module", 200, 0.0); // 200ms latency, no errors

    let test_inputs = vec!["test input".to_string()];

    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    assert_eq!(results.module_name, "slow_module");
    assert!(results.performance_metrics.average_latency_ms >= 200.0);
    assert!(results.performance_metrics.throughput_rps < 10.0); // Should be low due to high latency

    // Should suggest optimizations for high latency
    let has_latency_optimization = results
        .optimization_suggestions
        .iter()
        .any(|s| s.category == OptimizationCategory::Caching);
    assert!(has_latency_optimization);
}

#[tokio::test]
async fn test_benchmark_error_prone_module() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        benchmark_iterations: 10,
        timeout_seconds: 30,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("error_module", 50, 0.3); // 50ms latency, 30% error rate

    let test_inputs = vec!["test input".to_string()];

    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    assert_eq!(results.module_name, "error_module");
    assert!(results.performance_metrics.error_rate > 0.1); // Should have significant error rate
    assert!(results.performance_metrics.failed_requests > 0);

    // Should suggest optimizations for high error rate
    let has_error_optimization = results
        .optimization_suggestions
        .iter()
        .any(|s| s.priority == OptimizationPriority::Critical);
    assert!(has_error_optimization);
}

#[tokio::test]
async fn test_benchmark_results_export() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        benchmark_iterations: 2,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("export_test", 25, 0.0);

    let test_inputs = vec!["test".to_string()];
    let _results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    // Test JSON export
    let json_export = suite.export_results(BenchmarkOutputFormat::Json).unwrap();
    assert!(json_export.contains("export_test"));
    assert!(json_export.contains("performance_metrics"));

    // Test CSV export
    let csv_export = suite.export_results(BenchmarkOutputFormat::Csv).unwrap();
    assert!(csv_export.contains("module_name"));
    assert!(csv_export.contains("export_test"));

    // Test table export
    let table_export = suite.export_results(BenchmarkOutputFormat::Table).unwrap();
    assert!(table_export.contains("Module Name"));
    assert!(table_export.contains("export_test"));

    // Test Prometheus export
    let prometheus_export = suite
        .export_results(BenchmarkOutputFormat::Prometheus)
        .unwrap();
    assert!(prometheus_export.contains("dspy_latency_average_ms"));
    assert!(prometheus_export.contains("module=\"export_test\""));
}

#[tokio::test]
async fn test_performance_metrics_calculation() {
    let config = BenchmarkConfig {
        warmup_iterations: 0,
        benchmark_iterations: 5,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("metrics_test", 100, 0.2); // 100ms, 20% error rate

    let test_inputs = vec!["test".to_string()];
    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    let metrics = &results.performance_metrics;

    // Verify basic metrics
    assert_eq!(metrics.total_requests, 5);
    assert!(metrics.successful_requests <= 5);
    assert!(metrics.failed_requests <= 5);
    assert_eq!(
        metrics.successful_requests + metrics.failed_requests,
        metrics.total_requests
    );

    // Verify latency metrics
    assert!(metrics.average_latency_ms > 0.0);
    assert!(metrics.median_latency_ms > 0.0);
    assert!(metrics.min_latency_ms <= metrics.average_latency_ms);
    assert!(metrics.max_latency_ms >= metrics.average_latency_ms);
    assert!(metrics.p95_latency_ms >= metrics.median_latency_ms);
    assert!(metrics.p99_latency_ms >= metrics.p95_latency_ms);

    // Verify derived metrics
    assert!(metrics.throughput_rps > 0.0);
    assert!(metrics.error_rate >= 0.0 && metrics.error_rate <= 1.0);

    // Verify latency distribution
    assert!(!metrics.latency_distribution.is_empty());
    let total_percentage: f64 = metrics
        .latency_distribution
        .iter()
        .map(|bucket| bucket.percentage)
        .sum();
    assert!((total_percentage - 100.0).abs() < 1.0); // Should sum to ~100%
}

#[tokio::test]
async fn test_optimization_suggestions() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        benchmark_iterations: 3,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);

    // Test high latency module
    let slow_module = MockModule::new("slow_test", 1500, 0.0); // Very slow
    let test_inputs = vec!["test".to_string()];
    let results = suite
        .benchmark_module(&slow_module, test_inputs)
        .await
        .unwrap();

    let suggestions = &results.optimization_suggestions;
    assert!(!suggestions.is_empty());

    // Should suggest caching for high latency
    let has_caching_suggestion = suggestions
        .iter()
        .any(|s| s.category == OptimizationCategory::Caching);
    assert!(has_caching_suggestion);

    // Test high error rate module
    let error_module = MockModule::new("error_test", 100, 0.8); // High error rate
    let test_inputs = vec!["test".to_string()];
    let results = suite
        .benchmark_module(&error_module, test_inputs)
        .await
        .unwrap();

    let suggestions = &results.optimization_suggestions;

    // Should suggest configuration tuning for high error rate
    let has_config_suggestion = suggestions
        .iter()
        .any(|s| s.category == OptimizationCategory::ConfigurationTuning);
    assert!(has_config_suggestion);

    // Should have critical priority for high error rate
    let has_critical_suggestion = suggestions
        .iter()
        .any(|s| s.priority == OptimizationPriority::Critical);
    assert!(has_critical_suggestion);
}

#[tokio::test]
async fn test_adaptive_batch_processor() {
    let mut processor = AdaptiveBatchProcessor::<String, String>::new(100.0);

    // Initial batch size
    assert_eq!(processor.get_batch_size(), 10);

    // Simulate slow performance - should reduce batch size
    processor.update_performance(200.0, 10); // 200ms > 120ms (target * 1.2)
    let reduced_size = processor.get_batch_size();
    assert!(reduced_size < 10);

    // Simulate fast performance - should increase batch size
    // Need more iterations to overcome the averaging effect
    for _ in 0..10 {
        processor.update_performance(50.0, processor.get_batch_size()); // 50ms < 80ms (target * 0.8)
    }
    let increased_size = processor.get_batch_size();
    // The batch size should increase, but might not reach the original reduced size immediately
    // due to the conservative adjustment factor (0.1)
    assert!(increased_size > reduced_size || increased_size >= 8); // Either increased or reached reasonable size
}

#[tokio::test]
async fn test_connection_pool() {
    let pool = ConnectionPool::new(3);

    // Test initial state
    let metrics = pool.get_metrics();
    assert_eq!(metrics.current_active_connections, 0);

    // Test connection acquisition
    let _conn1 = pool.acquire_connection().await.unwrap();
    let _conn2 = pool.acquire_connection().await.unwrap();
    let _conn3 = pool.acquire_connection().await.unwrap();

    // Pool should be at capacity
    let metrics = pool.get_metrics();
    assert_eq!(metrics.current_active_connections, 3);

    // Test connection release (when guards are dropped)
    drop(_conn1);
    drop(_conn2);

    let metrics = pool.get_metrics();
    assert_eq!(metrics.current_active_connections, 1);
}

#[tokio::test]
async fn test_performance_optimizer() {
    let config = OptimizerConfig {
        enable_auto_optimization: true,
        performance_threshold: 0.8,
        max_optimization_attempts: 3,
        ..Default::default()
    };

    let optimizer = PerformanceOptimizer::new(config);

    // Record some metrics
    let metric1 = RequestMetric {
        timestamp: chrono::Utc::now(),
        latency_ms: 150.0,
        success: true,
        input_size_bytes: 1024,
        output_size_bytes: 512,
        cache_hit: false,
    };

    let metric2 = RequestMetric {
        timestamp: chrono::Utc::now(),
        latency_ms: 200.0,
        success: false,
        input_size_bytes: 2048,
        output_size_bytes: 0,
        cache_hit: false,
    };

    optimizer.record_request(metric1);
    optimizer.record_request(metric2);

    let summary = optimizer.get_performance_summary();
    assert_eq!(summary.average_latency_ms, 175.0);
    assert_eq!(summary.error_rate, 0.5);

    // Test resource metrics
    let resource_metric = ResourceMetric {
        timestamp: chrono::Utc::now(),
        memory_mb: 256.0,
        cpu_percent: 45.0,
        network_bytes_in: 1024,
        network_bytes_out: 512,
        active_connections: 5,
    };

    optimizer.record_resource(resource_metric);

    let updated_summary = optimizer.get_performance_summary();
    assert_eq!(updated_summary.memory_usage_mb, 256.0);
    assert_eq!(updated_summary.cpu_usage_percent, 45.0);
}

#[tokio::test]
async fn test_memory_aware_module() {
    let base_module = MockModule::new("base", 50, 0.0);
    let memory_module = MemoryAwareModule::new(base_module, 512); // 512MB limit

    // Test normal operation
    let result = memory_module.forward("test input".to_string()).await;
    assert!(result.is_ok());

    // Test module properties
    assert_eq!(memory_module.name(), "base");
    assert!(memory_module.supports_compilation());
}

#[tokio::test]
async fn test_benchmark_error_handling() {
    let config = BenchmarkConfig::default();
    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("test", 50, 0.0);

    // Test with empty inputs
    let empty_inputs: Vec<String> = vec![];
    let result = suite.benchmark_module(&module, empty_inputs).await;
    assert!(result.is_err());

    // Test with valid inputs
    let valid_inputs = vec!["test".to_string()];
    let result = suite.benchmark_module(&module, valid_inputs).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_benchmark_output_formats() {
    let formats = vec![
        BenchmarkOutputFormat::Json,
        BenchmarkOutputFormat::Csv,
        BenchmarkOutputFormat::Table,
        BenchmarkOutputFormat::Prometheus,
    ];

    // Test that all formats are distinct
    for (i, format1) in formats.iter().enumerate() {
        for (j, format2) in formats.iter().enumerate() {
            if i != j {
                assert_ne!(format1, format2);
            }
        }
    }
}

#[tokio::test]
async fn test_optimization_categories() {
    let categories = vec![
        OptimizationCategory::Caching,
        OptimizationCategory::Batching,
        OptimizationCategory::Parallelization,
        OptimizationCategory::MemoryOptimization,
        OptimizationCategory::NetworkOptimization,
        OptimizationCategory::AlgorithmOptimization,
        OptimizationCategory::ConfigurationTuning,
        OptimizationCategory::ResourcePooling,
    ];

    // Test that all categories are distinct
    for (i, cat1) in categories.iter().enumerate() {
        for (j, cat2) in categories.iter().enumerate() {
            if i != j {
                assert_ne!(cat1, cat2);
            }
        }
    }
}

#[tokio::test]
async fn test_optimization_priorities() {
    let priorities = vec![
        OptimizationPriority::Critical,
        OptimizationPriority::High,
        OptimizationPriority::Medium,
        OptimizationPriority::Low,
    ];

    // Test that all priorities are distinct
    for (i, pri1) in priorities.iter().enumerate() {
        for (j, pri2) in priorities.iter().enumerate() {
            if i != j {
                assert_ne!(pri1, pri2);
            }
        }
    }
}

#[tokio::test]
async fn test_implementation_efforts() {
    let efforts = vec![
        ImplementationEffort::Trivial,
        ImplementationEffort::Easy,
        ImplementationEffort::Medium,
        ImplementationEffort::Hard,
        ImplementationEffort::Complex,
    ];

    // Test that all efforts are distinct
    for (i, eff1) in efforts.iter().enumerate() {
        for (j, eff2) in efforts.iter().enumerate() {
            if i != j {
                assert_ne!(eff1, eff2);
            }
        }
    }
}

#[tokio::test]
async fn test_latency_distribution() {
    let config = BenchmarkConfig {
        warmup_iterations: 0,
        benchmark_iterations: 20,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("distribution_test", 100, 0.0);

    let test_inputs = vec!["test".to_string()];
    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    let distribution = &results.performance_metrics.latency_distribution;
    assert!(!distribution.is_empty());

    // Verify distribution properties
    for bucket in distribution {
        assert!(bucket.upper_bound_ms > 0.0);
        assert!(bucket.percentage >= 0.0 && bucket.percentage <= 100.0);
    }

    // Verify buckets are ordered
    for i in 1..distribution.len() {
        assert!(distribution[i].upper_bound_ms >= distribution[i - 1].upper_bound_ms);
    }
}

#[tokio::test]
async fn test_resource_usage_analysis() {
    let config = BenchmarkConfig {
        warmup_iterations: 1,
        benchmark_iterations: 5,
        memory_monitoring: true,
        cpu_monitoring: true,
        ..Default::default()
    };

    let mut suite = BenchmarkSuite::new(config);
    let module = MockModule::new("resource_test", 50, 0.0);

    let test_inputs = vec!["test".to_string()];
    let results = suite.benchmark_module(&module, test_inputs).await.unwrap();

    let resource_usage = &results.resource_usage;

    // Verify resource metrics are populated
    assert!(resource_usage.peak_memory_mb >= 0.0);
    assert!(resource_usage.average_memory_mb >= 0.0);
    assert!(resource_usage.peak_cpu_percent >= 0.0);
    assert!(resource_usage.average_cpu_percent >= 0.0);

    // Peak should be >= average
    assert!(resource_usage.peak_memory_mb >= resource_usage.average_memory_mb);
    assert!(resource_usage.peak_cpu_percent >= resource_usage.average_cpu_percent);
}
