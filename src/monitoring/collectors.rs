// Metrics Collectors for Performance Monitoring
// Provides specialized collectors for different types of metrics

use super::{CollectorConfig, CollectorHealth, Metric, MetricType, MetricValue, MetricsCollector};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::debug;

/// Application metrics collector
pub struct ApplicationMetricsCollector {
    /// Collector configuration
    config: CollectorConfig,
    /// Application metrics storage
    metrics: Arc<RwLock<HashMap<String, ApplicationMetric>>>,
    /// Collection statistics
    stats: Arc<RwLock<CollectionStats>>,
}

/// HTTP metrics collector
pub struct HttpMetricsCollector {
    /// Collector configuration
    config: CollectorConfig,
    /// HTTP metrics storage
    metrics: Arc<RwLock<HttpMetrics>>,
    /// Collection statistics
    stats: Arc<RwLock<CollectionStats>>,
}

/// Database metrics collector
pub struct DatabaseMetricsCollector {
    /// Collector configuration
    config: CollectorConfig,
    /// Database metrics storage
    metrics: Arc<RwLock<DatabaseMetrics>>,
    /// Collection statistics
    stats: Arc<RwLock<CollectionStats>>,
}

/// Custom metrics collector
pub struct CustomMetricsCollector {
    /// Collector configuration
    config: CollectorConfig,
    /// Custom metrics storage
    metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
    /// Collection statistics
    stats: Arc<RwLock<CollectionStats>>,
}

/// Application metric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
    /// Metric description
    pub description: Option<String>,
}

/// HTTP metrics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests (2xx)
    pub successful_requests: u64,
    /// Client error requests (4xx)
    pub client_errors: u64,
    /// Server error requests (5xx)
    pub server_errors: u64,
    /// Average response time (milliseconds)
    pub avg_response_time: f64,
    /// Request rate (requests per second)
    pub request_rate: f64,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// Response time percentiles
    pub response_time_percentiles: ResponseTimePercentiles,
    /// Requests by endpoint
    pub requests_by_endpoint: HashMap<String, EndpointMetrics>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Response time percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimePercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Endpoint-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointMetrics {
    /// Total requests to this endpoint
    pub total_requests: u64,
    /// Average response time
    pub avg_response_time: f64,
    /// Error count
    pub error_count: u64,
    /// Last request timestamp
    pub last_request: DateTime<Utc>,
}

/// Database metrics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetrics {
    /// Total queries executed
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Average query time (milliseconds)
    pub avg_query_time: f64,
    /// Query rate (queries per second)
    pub query_rate: f64,
    /// Connection pool metrics
    pub connection_pool: ConnectionPoolMetrics,
    /// Slow queries count
    pub slow_queries: u64,
    /// Deadlocks count
    pub deadlocks: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Connection pool metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolMetrics {
    /// Active connections
    pub active_connections: u32,
    /// Idle connections
    pub idle_connections: u32,
    /// Maximum connections
    pub max_connections: u32,
    /// Connection wait time (milliseconds)
    pub avg_wait_time: f64,
}

/// Custom metric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric value
    pub value: MetricValue,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Collection timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric description
    pub description: Option<String>,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    /// Total collections performed
    pub total_collections: u64,
    /// Successful collections
    pub successful_collections: u64,
    /// Failed collections
    pub failed_collections: u64,
    /// Average collection time (milliseconds)
    pub avg_collection_time: f64,
    /// Last collection timestamp
    pub last_collection: Option<DateTime<Utc>>,
    /// Last error
    pub last_error: Option<String>,
}

impl Default for CollectionStats {
    fn default() -> Self {
        Self {
            total_collections: 0,
            successful_collections: 0,
            failed_collections: 0,
            avg_collection_time: 0.0,
            last_collection: None,
            last_error: None,
        }
    }
}

impl Default for HttpMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            client_errors: 0,
            server_errors: 0,
            avg_response_time: 0.0,
            request_rate: 0.0,
            error_rate: 0.0,
            response_time_percentiles: ResponseTimePercentiles {
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
            requests_by_endpoint: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for DatabaseMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            avg_query_time: 0.0,
            query_rate: 0.0,
            connection_pool: ConnectionPoolMetrics {
                active_connections: 0,
                idle_connections: 0,
                max_connections: 0,
                avg_wait_time: 0.0,
            },
            slow_queries: 0,
            deadlocks: 0,
            last_updated: Utc::now(),
        }
    }
}

impl ApplicationMetricsCollector {
    /// Create a new application metrics collector
    pub fn new(config: CollectorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CollectionStats::default())),
        }
    }

    /// Record an application metric
    pub async fn record_metric(
        &self,
        name: &str,
        value: f64,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.insert(
            name.to_string(),
            ApplicationMetric {
                name: name.to_string(),
                value,
                labels,
                last_updated: Utc::now(),
                description: None,
            },
        );
        Ok(())
    }

    /// Get application metrics
    pub async fn get_metrics(&self) -> HashMap<String, ApplicationMetric> {
        self.metrics.read().await.clone()
    }
}

#[async_trait::async_trait]
impl MetricsCollector for ApplicationMetricsCollector {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn collect(&self) -> Result<Vec<Metric>> {
        let start_time = Instant::now();
        let mut collected_metrics = Vec::new();

        let metrics = self.metrics.read().await;
        for (_, app_metric) in metrics.iter() {
            collected_metrics.push(Metric {
                name: app_metric.name.clone(),
                value: MetricValue::Gauge(app_metric.value),
                labels: app_metric.labels.clone(),
                timestamp: app_metric.last_updated,
                metric_type: MetricType::Gauge,
                help: app_metric.description.clone(),
            });
        }

        // Update collection statistics
        let collection_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_collections += 1;
            stats.successful_collections += 1;
            stats.last_collection = Some(Utc::now());
            stats.avg_collection_time =
                (stats.avg_collection_time + collection_time.as_millis() as f64) / 2.0;
        }

        debug!("Collected {} application metrics", collected_metrics.len());
        Ok(collected_metrics)
    }

    async fn health_check(&self) -> Result<CollectorHealth> {
        let stats = self.stats.read().await;
        Ok(CollectorHealth {
            is_healthy: stats.last_error.is_none(),
            last_collection: stats.last_collection,
            error_count: stats.failed_collections as u32,
            collection_duration_ms: Some(stats.avg_collection_time as u64),
        })
    }

    fn config(&self) -> CollectorConfig {
        self.config.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl HttpMetricsCollector {
    /// Create a new HTTP metrics collector
    pub fn new(config: CollectorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HttpMetrics::default())),
            stats: Arc::new(RwLock::new(CollectionStats::default())),
        }
    }

    /// Record an HTTP request
    pub async fn record_request(
        &self,
        endpoint: &str,
        status_code: u16,
        response_time: f64,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;

        // Update total requests
        metrics.total_requests += 1;

        // Update status code counters
        match status_code {
            200..=299 => metrics.successful_requests += 1,
            400..=499 => metrics.client_errors += 1,
            500..=599 => metrics.server_errors += 1,
            _ => {}
        }

        // Update response time
        metrics.avg_response_time = (metrics.avg_response_time + response_time) / 2.0;

        // Update error rate
        let total_errors = metrics.client_errors + metrics.server_errors;
        metrics.error_rate = if metrics.total_requests > 0 {
            (total_errors as f64 / metrics.total_requests as f64) * 100.0
        } else {
            0.0
        };

        // Update endpoint metrics
        let endpoint_metrics = metrics
            .requests_by_endpoint
            .entry(endpoint.to_string())
            .or_insert(EndpointMetrics {
                total_requests: 0,
                avg_response_time: 0.0,
                error_count: 0,
                last_request: Utc::now(),
            });

        endpoint_metrics.total_requests += 1;
        endpoint_metrics.avg_response_time =
            (endpoint_metrics.avg_response_time + response_time) / 2.0;
        endpoint_metrics.last_request = Utc::now();

        if status_code >= 400 {
            endpoint_metrics.error_count += 1;
        }

        metrics.last_updated = Utc::now();
        Ok(())
    }

    /// Get HTTP metrics
    pub async fn get_metrics(&self) -> HttpMetrics {
        self.metrics.read().await.clone()
    }
}

#[async_trait::async_trait]
impl MetricsCollector for HttpMetricsCollector {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn collect(&self) -> Result<Vec<Metric>> {
        let start_time = Instant::now();
        let mut collected_metrics = Vec::new();

        let metrics = self.metrics.read().await;

        // Total requests metric
        collected_metrics.push(Metric {
            name: "http_requests_total".to_string(),
            value: MetricValue::Counter(metrics.total_requests),
            labels: HashMap::new(),
            timestamp: metrics.last_updated,
            metric_type: MetricType::Counter,
            help: Some("Total HTTP requests".to_string()),
        });

        // Response time metric
        collected_metrics.push(Metric {
            name: "http_response_time_avg".to_string(),
            value: MetricValue::Gauge(metrics.avg_response_time),
            labels: HashMap::new(),
            timestamp: metrics.last_updated,
            metric_type: MetricType::Gauge,
            help: Some("Average HTTP response time in milliseconds".to_string()),
        });

        // Error rate metric
        collected_metrics.push(Metric {
            name: "http_error_rate".to_string(),
            value: MetricValue::Gauge(metrics.error_rate),
            labels: HashMap::new(),
            timestamp: metrics.last_updated,
            metric_type: MetricType::Gauge,
            help: Some("HTTP error rate percentage".to_string()),
        });

        // Request rate metric
        collected_metrics.push(Metric {
            name: "http_request_rate".to_string(),
            value: MetricValue::Gauge(metrics.request_rate),
            labels: HashMap::new(),
            timestamp: metrics.last_updated,
            metric_type: MetricType::Gauge,
            help: Some("HTTP request rate per second".to_string()),
        });

        // Update collection statistics
        let collection_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_collections += 1;
            stats.successful_collections += 1;
            stats.last_collection = Some(Utc::now());
            stats.avg_collection_time =
                (stats.avg_collection_time + collection_time.as_millis() as f64) / 2.0;
        }

        debug!("Collected {} HTTP metrics", collected_metrics.len());
        Ok(collected_metrics)
    }

    async fn health_check(&self) -> Result<CollectorHealth> {
        let stats = self.stats.read().await;
        Ok(CollectorHealth {
            is_healthy: stats.last_error.is_none(),
            last_collection: stats.last_collection,
            error_count: stats.failed_collections as u32,
            collection_duration_ms: Some(stats.avg_collection_time as u64),
        })
    }

    fn config(&self) -> CollectorConfig {
        self.config.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CustomMetricsCollector {
    /// Create a new custom metrics collector
    pub fn new(config: CollectorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CollectionStats::default())),
        }
    }

    /// Add a custom metric
    pub async fn add_metric(&self, metric: CustomMetric) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.insert(metric.name.clone(), metric);
        Ok(())
    }

    /// Remove a custom metric
    pub async fn remove_metric(&self, name: &str) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        if metrics.remove(name).is_some() {
            Ok(())
        } else {
            Err(AgentError::validation(format!(
                "Metric not found: {}",
                name
            )))
        }
    }

    /// Get custom metrics
    pub async fn get_metrics(&self) -> HashMap<String, CustomMetric> {
        self.metrics.read().await.clone()
    }
}

#[async_trait::async_trait]
impl MetricsCollector for CustomMetricsCollector {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn collect(&self) -> Result<Vec<Metric>> {
        let start_time = Instant::now();
        let mut collected_metrics = Vec::new();

        let metrics = self.metrics.read().await;
        for (_, custom_metric) in metrics.iter() {
            collected_metrics.push(Metric {
                name: custom_metric.name.clone(),
                value: custom_metric.value.clone(),
                labels: custom_metric.labels.clone(),
                timestamp: custom_metric.timestamp,
                metric_type: custom_metric.metric_type.clone(),
                help: custom_metric.description.clone(),
            });
        }

        // Update collection statistics
        let collection_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_collections += 1;
            stats.successful_collections += 1;
            stats.last_collection = Some(Utc::now());
            stats.avg_collection_time =
                (stats.avg_collection_time + collection_time.as_millis() as f64) / 2.0;
        }

        debug!("Collected {} custom metrics", collected_metrics.len());
        Ok(collected_metrics)
    }

    async fn health_check(&self) -> Result<CollectorHealth> {
        let stats = self.stats.read().await;
        Ok(CollectorHealth {
            is_healthy: stats.last_error.is_none(),
            last_collection: stats.last_collection,
            error_count: stats.failed_collections as u32,
            collection_duration_ms: Some(stats.avg_collection_time as u64),
        })
    }

    fn config(&self) -> CollectorConfig {
        self.config.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
