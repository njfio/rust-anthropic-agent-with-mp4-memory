//! Performance optimization utilities for DSPy framework
//!
//! This module provides performance optimization tools, monitoring utilities,
//! and automatic performance tuning capabilities.

use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// Performance optimizer for DSPy modules
pub struct PerformanceOptimizer {
    config: OptimizerConfig,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
    optimization_history: Vec<OptimizationResult>,
}

/// Configuration for performance optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub enable_auto_optimization: bool,
    pub optimization_interval_seconds: u64,
    pub performance_threshold: f64,
    pub max_optimization_attempts: usize,
    pub enable_adaptive_batching: bool,
    pub enable_connection_pooling: bool,
    pub enable_request_coalescing: bool,
    pub memory_limit_mb: Option<usize>,
    pub cpu_limit_percent: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_auto_optimization: true,
            optimization_interval_seconds: 300, // 5 minutes
            performance_threshold: 0.8,
            max_optimization_attempts: 5,
            enable_adaptive_batching: true,
            enable_connection_pooling: true,
            enable_request_coalescing: true,
            memory_limit_mb: Some(1024), // 1GB
            cpu_limit_percent: Some(80.0),
        }
    }
}

/// Metrics collector for performance monitoring
pub struct MetricsCollector {
    request_metrics: Vec<RequestMetric>,
    resource_metrics: Vec<ResourceMetric>,
    error_metrics: Vec<ErrorMetric>,
    start_time: Instant,
}

/// Individual request metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub latency_ms: f64,
    pub success: bool,
    pub input_size_bytes: usize,
    pub output_size_bytes: usize,
    pub cache_hit: bool,
}

/// Resource usage metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub network_bytes_in: u64,
    pub network_bytes_out: u64,
    pub active_connections: usize,
}

/// Error metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_type: String,
    pub error_message: String,
    pub recovery_time_ms: Option<f64>,
}

/// Result of optimization attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub optimization_type: OptimizationType,
    pub before_metrics: PerformanceSummary,
    pub after_metrics: PerformanceSummary,
    pub improvement_percentage: f64,
    pub success: bool,
    pub details: String,
}

/// Types of optimizations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationType {
    CacheOptimization,
    BatchSizeOptimization,
    ConnectionPoolOptimization,
    MemoryOptimization,
    TimeoutOptimization,
    ConcurrencyOptimization,
    CompressionOptimization,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub average_latency_ms: f64,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Adaptive batch processor
pub struct AdaptiveBatchProcessor<I, O> {
    current_batch_size: usize,
    min_batch_size: usize,
    max_batch_size: usize,
    target_latency_ms: f64,
    recent_latencies: Vec<f64>,
    adjustment_factor: f64,
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O> AdaptiveBatchProcessor<I, O> {
    /// Create new adaptive batch processor
    pub fn new(target_latency_ms: f64) -> Self {
        Self {
            current_batch_size: 10,
            min_batch_size: 1,
            max_batch_size: 100,
            target_latency_ms,
            recent_latencies: Vec::new(),
            adjustment_factor: 0.1,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get current optimal batch size
    pub fn get_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Update batch size based on performance
    pub fn update_performance(&mut self, latency_ms: f64, batch_size: usize) {
        self.recent_latencies.push(latency_ms);

        // Keep only recent measurements
        if self.recent_latencies.len() > 10 {
            self.recent_latencies.remove(0);
        }

        let avg_latency =
            self.recent_latencies.iter().sum::<f64>() / self.recent_latencies.len() as f64;

        // Adjust batch size based on performance
        if avg_latency > self.target_latency_ms * 1.2 {
            // Too slow, reduce batch size
            self.current_batch_size =
                (self.current_batch_size as f64 * (1.0 - self.adjustment_factor)) as usize;
            self.current_batch_size = self.current_batch_size.max(self.min_batch_size);
        } else if avg_latency < self.target_latency_ms * 0.8 {
            // Fast enough, increase batch size
            self.current_batch_size =
                (self.current_batch_size as f64 * (1.0 + self.adjustment_factor)) as usize;
            self.current_batch_size = self.current_batch_size.min(self.max_batch_size);
        }

        debug!(
            "Adaptive batch size updated to {} (avg latency: {:.2}ms)",
            self.current_batch_size, avg_latency
        );
    }
}

/// Connection pool for managing API connections
pub struct ConnectionPool {
    max_connections: usize,
    active_connections: Arc<Semaphore>,
    connection_timeout: Duration,
    idle_timeout: Duration,
    metrics: Arc<Mutex<ConnectionPoolMetrics>>,
}

/// Connection pool metrics
#[derive(Debug, Clone, Default)]
pub struct ConnectionPoolMetrics {
    pub total_connections_created: usize,
    pub total_connections_destroyed: usize,
    pub current_active_connections: usize,
    pub current_idle_connections: usize,
    pub connection_wait_time_ms: f64,
    pub connection_reuse_rate: f64,
}

impl ConnectionPool {
    /// Create new connection pool
    pub fn new(max_connections: usize) -> Self {
        Self {
            max_connections,
            active_connections: Arc::new(Semaphore::new(max_connections)),
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            metrics: Arc::new(Mutex::new(ConnectionPoolMetrics::default())),
        }
    }

    /// Acquire connection from pool
    pub async fn acquire_connection(&self) -> DspyResult<ConnectionGuard<'_>> {
        let start_time = Instant::now();

        let permit = self
            .active_connections
            .acquire()
            .await
            .map_err(|_| DspyError::module("ConnectionPool", "Failed to acquire connection"))?;

        let wait_time = start_time.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.connection_wait_time_ms = wait_time.as_millis() as f64;
            metrics.current_active_connections += 1;
        }

        Ok(ConnectionGuard {
            _permit: permit,
            pool_metrics: self.metrics.clone(),
        })
    }

    /// Get connection pool metrics
    pub fn get_metrics(&self) -> ConnectionPoolMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Guard for connection pool
pub struct ConnectionGuard<'a> {
    _permit: tokio::sync::SemaphorePermit<'a>,
    pool_metrics: Arc<Mutex<ConnectionPoolMetrics>>,
}

impl<'a> Drop for ConnectionGuard<'a> {
    fn drop(&mut self) {
        let mut metrics = self.pool_metrics.lock().unwrap();
        metrics.current_active_connections -= 1;
    }
}

/// Request coalescing for combining similar requests
pub struct RequestCoalescer<I, O> {
    pending_requests: HashMap<String, Vec<tokio::sync::oneshot::Sender<DspyResult<O>>>>,
    coalescing_window_ms: u64,
    _phantom: std::marker::PhantomData<I>,
}

impl<I, O> RequestCoalescer<I, O>
where
    I: Clone + std::hash::Hash + Eq,
    O: Clone,
{
    /// Create new request coalescer
    pub fn new(coalescing_window_ms: u64) -> Self {
        Self {
            pending_requests: HashMap::new(),
            coalescing_window_ms,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Submit request for coalescing
    pub async fn submit_request<M>(&mut self, module: &M, input: I) -> DspyResult<O>
    where
        M: Module<Input = I, Output = O>,
        I: std::fmt::Debug,
    {
        let request_key = format!("{:?}", input);

        // Check if similar request is already pending
        if let Some(senders) = self.pending_requests.get_mut(&request_key) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            senders.push(tx);
            return rx
                .await
                .map_err(|_| DspyError::module("RequestCoalescer", "Request cancelled"))?;
        }

        // Start new coalesced request
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending_requests.insert(request_key.clone(), vec![tx]);

        // Wait for coalescing window
        tokio::time::sleep(Duration::from_millis(self.coalescing_window_ms)).await;

        // Execute request
        let result = module.forward(input).await;

        // Send result to all waiting requests
        if let Some(senders) = self.pending_requests.remove(&request_key) {
            for sender in senders {
                let _ = sender.send(result.clone());
            }
        }

        result
    }
}

/// Memory-aware module wrapper
pub struct MemoryAwareModule<M> {
    inner: M,
    memory_limit_mb: usize,
    current_memory_mb: Arc<Mutex<f64>>,
    memory_check_interval: Duration,
}

impl<M> MemoryAwareModule<M> {
    /// Create new memory-aware module
    pub fn new(inner: M, memory_limit_mb: usize) -> Self {
        Self {
            inner,
            memory_limit_mb,
            current_memory_mb: Arc::new(Mutex::new(0.0)),
            memory_check_interval: Duration::from_secs(10),
        }
    }

    /// Check current memory usage
    fn check_memory_usage(&self) -> DspyResult<()> {
        let current_memory = *self.current_memory_mb.lock().unwrap();

        if current_memory > self.memory_limit_mb as f64 {
            return Err(DspyError::module(
                "MemoryAwareModule",
                &format!(
                    "Memory limit exceeded: {:.1}MB > {}MB",
                    current_memory, self.memory_limit_mb
                ),
            ));
        }

        Ok(())
    }

    /// Update memory usage
    fn update_memory_usage(&self, memory_mb: f64) {
        *self.current_memory_mb.lock().unwrap() = memory_mb;
    }
}

#[async_trait]
impl<M, I, O> Module for MemoryAwareModule<M>
where
    M: Module<Input = I, Output = O>,
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    type Input = I;
    type Output = O;

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        self.inner.signature()
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        // Check memory before processing
        self.check_memory_usage()?;

        // Monitor memory during processing
        let start_memory = self.get_current_memory_usage();

        let result = self.inner.forward(input).await;

        let end_memory = self.get_current_memory_usage();
        self.update_memory_usage(end_memory);

        // Log memory usage change
        let memory_delta = end_memory - start_memory;
        if memory_delta > 10.0 {
            debug!(
                "Memory usage increased by {:.1}MB during processing",
                memory_delta
            );
        }

        result
    }

    fn metadata(&self) -> &ModuleMetadata {
        self.inner.metadata()
    }

    fn stats(&self) -> &ModuleStats {
        self.inner.stats()
    }

    fn supports_compilation(&self) -> bool {
        self.inner.supports_compilation()
    }
}

impl<M> MemoryAwareModule<M> {
    /// Get current memory usage (mock implementation)
    fn get_current_memory_usage(&self) -> f64 {
        // In a real implementation, this would use system APIs
        100.0 // Mock value in MB
    }
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new())),
            optimization_history: Vec::new(),
        }
    }

    /// Record request metric
    pub fn record_request(&self, metric: RequestMetric) {
        let mut collector = self.metrics_collector.lock().unwrap();
        collector.request_metrics.push(metric);

        // Keep only recent metrics
        if collector.request_metrics.len() > 10000 {
            collector.request_metrics.drain(0..1000);
        }
    }

    /// Record resource metric
    pub fn record_resource(&self, metric: ResourceMetric) {
        let mut collector = self.metrics_collector.lock().unwrap();
        collector.resource_metrics.push(metric);

        // Keep only recent metrics
        if collector.resource_metrics.len() > 1000 {
            collector.resource_metrics.drain(0..100);
        }
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let collector = self.metrics_collector.lock().unwrap();

        if collector.request_metrics.is_empty() {
            return PerformanceSummary {
                average_latency_ms: 0.0,
                throughput_rps: 0.0,
                error_rate: 0.0,
                memory_usage_mb: 0.0,
                cpu_usage_percent: 0.0,
            };
        }

        let total_requests = collector.request_metrics.len();
        let successful_requests = collector
            .request_metrics
            .iter()
            .filter(|m| m.success)
            .count();

        let average_latency_ms = collector
            .request_metrics
            .iter()
            .map(|m| m.latency_ms)
            .sum::<f64>()
            / total_requests as f64;

        let elapsed_seconds = collector.start_time.elapsed().as_secs_f64();
        let throughput_rps = total_requests as f64 / elapsed_seconds;

        let error_rate = (total_requests - successful_requests) as f64 / total_requests as f64;

        let memory_usage_mb = collector
            .resource_metrics
            .last()
            .map(|m| m.memory_mb)
            .unwrap_or(0.0);

        let cpu_usage_percent = collector
            .resource_metrics
            .last()
            .map(|m| m.cpu_percent)
            .unwrap_or(0.0);

        PerformanceSummary {
            average_latency_ms,
            throughput_rps,
            error_rate,
            memory_usage_mb,
            cpu_usage_percent,
        }
    }

    /// Check if optimization is needed
    pub fn should_optimize(&self) -> bool {
        if !self.config.enable_auto_optimization {
            return false;
        }

        let summary = self.get_performance_summary();

        // Check various performance indicators
        summary.average_latency_ms > 1000.0
            || summary.error_rate > 0.05
            || summary.memory_usage_mb > self.config.memory_limit_mb.unwrap_or(1024) as f64 * 0.9
            || summary.cpu_usage_percent > self.config.cpu_limit_percent.unwrap_or(80.0) * 0.9
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationResult] {
        &self.optimization_history
    }
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            request_metrics: Vec::new(),
            resource_metrics: Vec::new(),
            error_metrics: Vec::new(),
            start_time: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_types() {
        let types = vec![
            OptimizationType::CacheOptimization,
            OptimizationType::BatchSizeOptimization,
            OptimizationType::ConnectionPoolOptimization,
            OptimizationType::MemoryOptimization,
            OptimizationType::TimeoutOptimization,
            OptimizationType::ConcurrencyOptimization,
            OptimizationType::CompressionOptimization,
        ];

        // Test that all types are distinct
        for (i, type1) in types.iter().enumerate() {
            for (j, type2) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(type1, type2);
                }
            }
        }
    }

    #[test]
    fn test_adaptive_batch_processor() {
        let mut processor = AdaptiveBatchProcessor::<String, String>::new(100.0);

        // Initial batch size should be reasonable
        assert_eq!(processor.get_batch_size(), 10);

        // Test performance update with high latency (should reduce batch size)
        processor.update_performance(150.0, 10);
        let new_size = processor.get_batch_size();
        assert!(new_size <= 10);

        // Test performance update with low latency (should increase batch size)
        processor.update_performance(50.0, new_size);
        let final_size = processor.get_batch_size();
        assert!(final_size >= new_size);
    }

    #[test]
    fn test_connection_pool_creation() {
        let pool = ConnectionPool::new(5);
        let metrics = pool.get_metrics();

        assert_eq!(metrics.current_active_connections, 0);
        assert_eq!(metrics.total_connections_created, 0);
    }

    #[tokio::test]
    async fn test_connection_pool_acquire() {
        let pool = ConnectionPool::new(2);

        // Acquire first connection
        let _guard1 = pool.acquire_connection().await.unwrap();
        let metrics = pool.get_metrics();
        assert_eq!(metrics.current_active_connections, 1);

        // Acquire second connection
        let _guard2 = pool.acquire_connection().await.unwrap();
        let metrics = pool.get_metrics();
        assert_eq!(metrics.current_active_connections, 2);

        // Drop first connection
        drop(_guard1);
        let metrics = pool.get_metrics();
        assert_eq!(metrics.current_active_connections, 1);
    }

    #[test]
    fn test_request_coalescer_creation() {
        let coalescer = RequestCoalescer::<String, String>::new(100);
        assert_eq!(coalescer.coalescing_window_ms, 100);
    }

    #[test]
    fn test_memory_aware_module_creation() {
        use crate::anthropic::AnthropicClient;
        use crate::config::AnthropicConfig;
        use crate::dspy::predictor::Predict;
        use crate::dspy::signature::Signature;
        use std::sync::Arc;

        // Create a mock module
        let config = AnthropicConfig::default();
        let client = Arc::new(AnthropicClient::new(config).unwrap());
        let signature = Signature::<String, String>::new("test");
        let predict = Predict::new(signature, client);

        let memory_module = MemoryAwareModule::new(predict, 512);
        assert_eq!(memory_module.memory_limit_mb, 512);
    }

    #[test]
    fn test_performance_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        // Test initial performance summary
        let summary = optimizer.get_performance_summary();
        assert_eq!(summary.average_latency_ms, 0.0);
        assert_eq!(summary.throughput_rps, 0.0);
        assert_eq!(summary.error_rate, 0.0);
    }

    #[test]
    fn test_request_metric_recording() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let metric = RequestMetric {
            timestamp: chrono::Utc::now(),
            latency_ms: 100.0,
            success: true,
            input_size_bytes: 1024,
            output_size_bytes: 512,
            cache_hit: false,
        };

        optimizer.record_request(metric);

        let summary = optimizer.get_performance_summary();
        assert_eq!(summary.average_latency_ms, 100.0);
        assert!(summary.throughput_rps > 0.0);
        assert_eq!(summary.error_rate, 0.0);
    }

    #[test]
    fn test_resource_metric_recording() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        // First record a request metric to make the summary valid
        let request_metric = RequestMetric {
            timestamp: chrono::Utc::now(),
            latency_ms: 100.0,
            success: true,
            input_size_bytes: 1024,
            output_size_bytes: 512,
            cache_hit: false,
        };
        optimizer.record_request(request_metric);

        let resource_metric = ResourceMetric {
            timestamp: chrono::Utc::now(),
            memory_mb: 256.0,
            cpu_percent: 45.0,
            network_bytes_in: 1024,
            network_bytes_out: 512,
            active_connections: 3,
        };

        optimizer.record_resource(resource_metric);

        let summary = optimizer.get_performance_summary();
        assert_eq!(summary.memory_usage_mb, 256.0);
        assert_eq!(summary.cpu_usage_percent, 45.0);
    }

    #[test]
    fn test_performance_summary_with_errors() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        // Record successful request
        let success_metric = RequestMetric {
            timestamp: chrono::Utc::now(),
            latency_ms: 100.0,
            success: true,
            input_size_bytes: 1024,
            output_size_bytes: 512,
            cache_hit: false,
        };
        optimizer.record_request(success_metric);

        // Record failed request
        let error_metric = RequestMetric {
            timestamp: chrono::Utc::now(),
            latency_ms: 200.0,
            success: false,
            input_size_bytes: 1024,
            output_size_bytes: 0,
            cache_hit: false,
        };
        optimizer.record_request(error_metric);

        let summary = optimizer.get_performance_summary();
        assert_eq!(summary.average_latency_ms, 150.0); // (100 + 200) / 2
        assert_eq!(summary.error_rate, 0.5); // 1 error out of 2 requests
    }

    #[test]
    fn test_optimizer_config_defaults() {
        let config = OptimizerConfig::default();

        assert!(config.enable_auto_optimization);
        assert_eq!(config.optimization_interval_seconds, 300);
        assert_eq!(config.performance_threshold, 0.8);
        assert_eq!(config.max_optimization_attempts, 5);
        assert!(config.enable_adaptive_batching);
        assert!(config.enable_connection_pooling);
        assert!(config.enable_request_coalescing);
        assert_eq!(config.memory_limit_mb, Some(1024));
        assert_eq!(config.cpu_limit_percent, Some(80.0));
    }

    #[test]
    fn test_metrics_serialization() {
        let metric = RequestMetric {
            timestamp: chrono::Utc::now(),
            latency_ms: 123.45,
            success: true,
            input_size_bytes: 1024,
            output_size_bytes: 512,
            cache_hit: true,
        };

        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("123.45"));
        assert!(json.contains("true"));
        assert!(json.contains("1024"));

        let deserialized: RequestMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.latency_ms, 123.45);
        assert!(deserialized.success);
        assert_eq!(deserialized.input_size_bytes, 1024);
    }

    #[test]
    fn test_optimization_result_structure() {
        let before_metrics = PerformanceSummary {
            average_latency_ms: 200.0,
            throughput_rps: 5.0,
            error_rate: 0.1,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 80.0,
        };

        let after_metrics = PerformanceSummary {
            average_latency_ms: 150.0,
            throughput_rps: 6.67,
            error_rate: 0.05,
            memory_usage_mb: 400.0,
            cpu_usage_percent: 60.0,
        };

        let optimization_result = OptimizationResult {
            timestamp: chrono::Utc::now(),
            optimization_type: OptimizationType::CacheOptimization,
            before_metrics,
            after_metrics,
            improvement_percentage: 25.0,
            success: true,
            details: "Cache hit rate improved from 60% to 85%".to_string(),
        };

        assert_eq!(
            optimization_result.optimization_type,
            OptimizationType::CacheOptimization
        );
        assert!(optimization_result.success);
        assert_eq!(optimization_result.improvement_percentage, 25.0);
    }
}
