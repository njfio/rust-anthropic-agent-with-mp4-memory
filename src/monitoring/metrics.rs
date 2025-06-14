// Metrics Utilities and Helpers for Performance Monitoring
// Provides metric creation, aggregation, and analysis utilities

use super::{Metric, MetricType, MetricValue, HistogramData, HistogramBucket, SummaryData, Quantile};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;

/// Metrics registry for managing and aggregating metrics
pub struct MetricsRegistry {
    /// Registered metrics
    metrics: Arc<RwLock<HashMap<String, RegisteredMetric>>>,
    /// Metric aggregation rules
    aggregation_rules: Arc<RwLock<HashMap<String, AggregationRule>>>,
    /// Registry configuration
    config: RegistryConfig,
}

/// Registered metric with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredMetric {
    /// Metric definition
    pub metric: Metric,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Update count
    pub update_count: u64,
    /// Metric tags
    pub tags: HashMap<String, String>,
}

/// Aggregation rule for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationRule {
    /// Rule name
    pub name: String,
    /// Source metric pattern
    pub source_pattern: String,
    /// Aggregation function
    pub aggregation_function: AggregationFunction,
    /// Aggregation window
    pub window: Duration,
    /// Target metric name
    pub target_metric: String,
    /// Rule enabled
    pub enabled: bool,
}

/// Aggregation functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregationFunction {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Rate,
    Percentile(f64),
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum metrics to store
    pub max_metrics: usize,
    /// Metric retention period
    pub retention_period: Duration,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

/// Metric builder for creating metrics
pub struct MetricBuilder {
    name: String,
    metric_type: MetricType,
    help: Option<String>,
    labels: HashMap<String, String>,
    timestamp: Option<DateTime<Utc>>,
}

/// Histogram builder for creating histogram metrics
pub struct HistogramBuilder {
    buckets: Vec<f64>,
    values: Vec<f64>,
}

/// Summary builder for creating summary metrics
pub struct SummaryBuilder {
    values: Vec<f64>,
    quantiles: Vec<f64>,
}

/// Metric aggregator for combining metrics
pub struct MetricAggregator {
    /// Aggregation window
    window: Duration,
    /// Stored metrics for aggregation
    metrics_buffer: Arc<RwLock<Vec<Metric>>>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_metrics: 10000,
            retention_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60 * 60), // 1 hour
        }
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            aggregation_rules: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RegistryConfig::default())
    }

    /// Register a metric
    pub async fn register_metric(&self, metric: Metric, tags: HashMap<String, String>) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        // Check if we're at capacity
        if metrics.len() >= self.config.max_metrics {
            return Err(AgentError::validation("Metrics registry at capacity".to_string()));
        }

        let registered_metric = RegisteredMetric {
            metric: metric.clone(),
            registered_at: Utc::now(),
            last_updated: metric.timestamp,
            update_count: 1,
            tags,
        };

        metrics.insert(metric.name.clone(), registered_metric);
        debug!("Registered metric: {}", metric.name);
        Ok(())
    }

    /// Update an existing metric
    pub async fn update_metric(&self, name: &str, value: MetricValue) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        if let Some(registered_metric) = metrics.get_mut(name) {
            registered_metric.metric.value = value;
            registered_metric.metric.timestamp = Utc::now();
            registered_metric.last_updated = Utc::now();
            registered_metric.update_count += 1;
            debug!("Updated metric: {}", name);
            Ok(())
        } else {
            Err(AgentError::validation(format!("Metric not found: {}", name)))
        }
    }

    /// Get a metric by name
    pub async fn get_metric(&self, name: &str) -> Option<RegisteredMetric> {
        let metrics = self.metrics.read().await;
        metrics.get(name).cloned()
    }

    /// Get all metrics
    pub async fn get_all_metrics(&self) -> Vec<RegisteredMetric> {
        let metrics = self.metrics.read().await;
        metrics.values().cloned().collect()
    }

    /// Add an aggregation rule
    pub async fn add_aggregation_rule(&self, rule: AggregationRule) -> Result<()> {
        let mut rules = self.aggregation_rules.write().await;
        rules.insert(rule.name.clone(), rule.clone());
        debug!("Added aggregation rule: {}", rule.name);
        Ok(())
    }

    /// Remove an aggregation rule
    pub async fn remove_aggregation_rule(&self, name: &str) -> Result<()> {
        let mut rules = self.aggregation_rules.write().await;
        if rules.remove(name).is_some() {
            debug!("Removed aggregation rule: {}", name);
            Ok(())
        } else {
            Err(AgentError::validation(format!("Aggregation rule not found: {}", name)))
        }
    }

    /// Apply aggregation rules
    pub async fn apply_aggregations(&self) -> Result<Vec<Metric>> {
        let rules = self.aggregation_rules.read().await;
        let metrics = self.metrics.read().await;
        let mut aggregated_metrics = Vec::new();

        for rule in rules.values() {
            if !rule.enabled {
                continue;
            }

            // Find matching metrics
            let matching_metrics: Vec<&RegisteredMetric> = metrics
                .values()
                .filter(|m| self.matches_pattern(&m.metric.name, &rule.source_pattern))
                .collect();

            if matching_metrics.is_empty() {
                continue;
            }

            // Apply aggregation function
            if let Some(aggregated_metric) = self.aggregate_metrics(&matching_metrics, rule).await? {
                aggregated_metrics.push(aggregated_metric);
            }
        }

        debug!("Applied aggregations, created {} metrics", aggregated_metrics.len());
        Ok(aggregated_metrics)
    }

    /// Check if metric name matches pattern
    fn matches_pattern(&self, name: &str, pattern: &str) -> bool {
        // Simple pattern matching - could be enhanced with regex
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                name.starts_with(parts[0]) && name.ends_with(parts[1])
            } else {
                false
            }
        } else {
            name == pattern
        }
    }

    /// Aggregate metrics according to rule
    async fn aggregate_metrics(&self, metrics: &[&RegisteredMetric], rule: &AggregationRule) -> Result<Option<Metric>> {
        if metrics.is_empty() {
            return Ok(None);
        }

        let values: Vec<f64> = metrics
            .iter()
            .filter_map(|m| match &m.metric.value {
                MetricValue::Counter(v) => Some(*v as f64),
                MetricValue::Gauge(v) => Some(*v),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return Ok(None);
        }

        let aggregated_value = match rule.aggregation_function {
            AggregationFunction::Sum => values.iter().sum(),
            AggregationFunction::Average => values.iter().sum::<f64>() / values.len() as f64,
            AggregationFunction::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggregationFunction::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggregationFunction::Count => values.len() as f64,
            AggregationFunction::Rate => {
                // Simple rate calculation - could be enhanced
                values.iter().sum::<f64>() / rule.window.as_secs() as f64
            }
            AggregationFunction::Percentile(p) => {
                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let index = ((p / 100.0) * (sorted_values.len() - 1) as f64) as usize;
                sorted_values[index.min(sorted_values.len() - 1)]
            }
        };

        Ok(Some(Metric {
            name: rule.target_metric.clone(),
            value: MetricValue::Gauge(aggregated_value),
            labels: HashMap::new(),
            timestamp: Utc::now(),
            metric_type: MetricType::Gauge,
            help: Some(format!("Aggregated metric using {:?}", rule.aggregation_function)),
        }))
    }

    /// Cleanup old metrics
    pub async fn cleanup_old_metrics(&self) -> Result<usize> {
        let mut metrics = self.metrics.write().await;
        let cutoff_time = Utc::now() - chrono::Duration::from_std(self.config.retention_period)
            .map_err(|e| AgentError::validation(format!("Invalid retention period: {}", e)))?;

        let initial_count = metrics.len();
        metrics.retain(|_, metric| metric.last_updated > cutoff_time);
        let removed_count = initial_count - metrics.len();

        if removed_count > 0 {
            debug!("Cleaned up {} old metrics", removed_count);
        }

        Ok(removed_count)
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> RegistryStats {
        let metrics = self.metrics.read().await;
        let rules = self.aggregation_rules.read().await;

        RegistryStats {
            total_metrics: metrics.len(),
            total_rules: rules.len(),
            enabled_rules: rules.values().filter(|r| r.enabled).count(),
            oldest_metric: metrics.values()
                .map(|m| m.registered_at)
                .min(),
            newest_metric: metrics.values()
                .map(|m| m.registered_at)
                .max(),
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_metrics: usize,
    pub total_rules: usize,
    pub enabled_rules: usize,
    pub oldest_metric: Option<DateTime<Utc>>,
    pub newest_metric: Option<DateTime<Utc>>,
}

impl MetricBuilder {
    /// Create a new metric builder
    pub fn new(name: &str, metric_type: MetricType) -> Self {
        Self {
            name: name.to_string(),
            metric_type,
            help: None,
            labels: HashMap::new(),
            timestamp: None,
        }
    }

    /// Add help text
    pub fn help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    /// Add a label
    pub fn label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Add multiple labels
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels.extend(labels);
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Build a counter metric
    pub fn counter(self, value: u64) -> Metric {
        Metric {
            name: self.name,
            value: MetricValue::Counter(value),
            labels: self.labels,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            metric_type: MetricType::Counter,
            help: self.help,
        }
    }

    /// Build a gauge metric
    pub fn gauge(self, value: f64) -> Metric {
        Metric {
            name: self.name,
            value: MetricValue::Gauge(value),
            labels: self.labels,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            metric_type: MetricType::Gauge,
            help: self.help,
        }
    }

    /// Build a histogram metric
    pub fn histogram(self, histogram_data: HistogramData) -> Metric {
        Metric {
            name: self.name,
            value: MetricValue::Histogram(histogram_data),
            labels: self.labels,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            metric_type: MetricType::Histogram,
            help: self.help,
        }
    }

    /// Build a summary metric
    pub fn summary(self, summary_data: SummaryData) -> Metric {
        Metric {
            name: self.name,
            value: MetricValue::Summary(summary_data),
            labels: self.labels,
            timestamp: self.timestamp.unwrap_or_else(Utc::now),
            metric_type: MetricType::Summary,
            help: self.help,
        }
    }
}

impl HistogramBuilder {
    /// Create a new histogram builder
    pub fn new() -> Self {
        Self {
            buckets: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add bucket boundaries
    pub fn buckets(mut self, buckets: Vec<f64>) -> Self {
        self.buckets = buckets;
        self
    }

    /// Add values to histogram
    pub fn values(mut self, values: Vec<f64>) -> Self {
        self.values = values;
        self
    }

    /// Build histogram data
    pub fn build(self) -> HistogramData {
        let mut histogram_buckets = Vec::new();
        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for &upper_bound in &self.buckets {
            let count = sorted_values.iter().filter(|&&v| v <= upper_bound).count() as u64;
            histogram_buckets.push(HistogramBucket { upper_bound, count });
        }

        HistogramData {
            buckets: histogram_buckets,
            count: self.values.len() as u64,
            sum: self.values.iter().sum(),
        }
    }
}

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SummaryBuilder {
    /// Create a new summary builder
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            quantiles: vec![0.5, 0.9, 0.95, 0.99],
        }
    }

    /// Set values
    pub fn values(mut self, values: Vec<f64>) -> Self {
        self.values = values;
        self
    }

    /// Set quantiles to calculate
    pub fn quantiles(mut self, quantiles: Vec<f64>) -> Self {
        self.quantiles = quantiles;
        self
    }

    /// Build summary data
    pub fn build(self) -> SummaryData {
        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut quantiles = Vec::new();
        for &q in &self.quantiles {
            let value = if sorted_values.is_empty() {
                0.0
            } else {
                let index = ((q * (sorted_values.len() - 1) as f64) as usize).min(sorted_values.len() - 1);
                sorted_values.get(index).copied().unwrap_or(0.0)
            };
            quantiles.push(Quantile { quantile: q, value });
        }

        SummaryData {
            quantiles,
            count: self.values.len() as u64,
            sum: self.values.iter().sum(),
        }
    }
}

impl Default for SummaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
