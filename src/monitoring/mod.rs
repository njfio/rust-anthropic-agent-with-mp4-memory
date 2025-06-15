// Performance Monitoring System for Enterprise AI Agent
// Provides comprehensive metrics collection, alerting, and resource tracking

use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub mod alerts;
pub mod collectors;
pub mod exporters;
pub mod metrics;
pub mod resource_tracker;

#[cfg(test)]
mod tests;

/// Performance monitoring manager
pub struct PerformanceMonitor {
    /// Metrics collectors
    collectors: Arc<RwLock<HashMap<String, Box<dyn MetricsCollector>>>>,
    /// Alert manager
    alert_manager: Arc<alerts::AlertManager>,
    /// Resource tracker
    resource_tracker: Arc<resource_tracker::ResourceTracker>,
    /// Metrics exporters
    exporters: Arc<RwLock<Vec<Box<dyn MetricsExporter>>>>,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// System start time
    start_time: Instant,
    /// Performance statistics
    stats: Arc<RwLock<PerformanceStats>>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Collection interval in seconds
    pub collection_interval: u64,
    /// Metrics retention period in hours
    pub retention_hours: u64,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Export configuration
    pub export_config: ExportConfig,
    /// Sampling configuration
    pub sampling_config: SamplingConfig,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,
    /// Response time threshold (milliseconds)
    pub response_time_threshold: u64,
    /// Error rate threshold (percentage)
    pub error_rate_threshold: f64,
    /// Disk usage threshold (percentage)
    pub disk_threshold: f64,
    /// Network latency threshold (milliseconds)
    pub network_latency_threshold: u64,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Enable Prometheus export
    pub enable_prometheus: bool,
    /// Prometheus endpoint
    pub prometheus_endpoint: String,
    /// Enable JSON export
    pub enable_json: bool,
    /// JSON export path
    pub json_export_path: String,
    /// Export interval in seconds
    pub export_interval: u64,
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Metrics sampling rate (0.0 to 1.0)
    pub metrics_sampling_rate: f64,
    /// Trace sampling rate (0.0 to 1.0)
    pub trace_sampling_rate: f64,
    /// Enable adaptive sampling
    pub adaptive_sampling: bool,
    /// High-frequency metrics list
    pub high_frequency_metrics: Vec<String>,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total metrics collected
    pub total_metrics_collected: u64,
    /// Total alerts triggered
    pub total_alerts_triggered: u64,
    /// Average collection time (milliseconds)
    pub avg_collection_time_ms: f64,
    /// System uptime (seconds)
    pub uptime_seconds: u64,
    /// Current active collectors
    pub active_collectors: u32,
    /// Current active exporters
    pub active_exporters: u32,
    /// Last collection timestamp
    pub last_collection: Option<DateTime<Utc>>,
    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Performance trends data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// CPU usage trend (last 24 hours)
    pub cpu_trend: Vec<f64>,
    /// Memory usage trend (last 24 hours)
    pub memory_trend: Vec<f64>,
    /// Response time trend (last 24 hours)
    pub response_time_trend: Vec<f64>,
    /// Error rate trend (last 24 hours)
    pub error_rate_trend: Vec<f64>,
    /// Throughput trend (requests per minute)
    pub throughput_trend: Vec<f64>,
}

/// Comprehensive monitoring status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatus {
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Collector health status
    pub collector_health: Vec<(String, CollectorHealth)>,
    /// Exporter health status
    pub exporter_health: Vec<(String, ExporterHealth)>,
    /// Resource tracker status
    pub resource_status: crate::monitoring::resource_tracker::ResourceStats,
    /// System uptime
    pub system_uptime: Duration,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
}

/// Trait for metrics collectors
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Get collector name
    fn name(&self) -> &str;

    /// Collect metrics
    async fn collect(&self) -> Result<Vec<Metric>>;

    /// Get collector health status
    async fn health_check(&self) -> Result<CollectorHealth>;

    /// Get collector configuration
    fn config(&self) -> CollectorConfig;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Trait for metrics exporters
#[async_trait::async_trait]
pub trait MetricsExporter: Send + Sync {
    /// Get exporter name
    fn name(&self) -> &str;
    
    /// Export metrics
    async fn export(&self, metrics: &[Metric]) -> Result<()>;
    
    /// Get exporter health status
    async fn health_check(&self) -> Result<ExporterHealth>;
    
    /// Get exporter configuration
    fn config(&self) -> ExporterConfig;
}

/// Metric data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: MetricValue,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric type
    pub metric_type: MetricType,
    /// Help text
    pub help: Option<String>,
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(HistogramData),
    Summary(SummaryData),
}

/// Metric types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Histogram data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub buckets: Vec<HistogramBucket>,
    pub count: u64,
    pub sum: f64,
}

/// Histogram bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub upper_bound: f64,
    pub count: u64,
}

/// Summary data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryData {
    pub quantiles: Vec<Quantile>,
    pub count: u64,
    pub sum: f64,
}

/// Quantile data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantile {
    pub quantile: f64,
    pub value: f64,
}

/// Collector health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorHealth {
    pub is_healthy: bool,
    pub last_collection: Option<DateTime<Utc>>,
    pub error_count: u32,
    pub collection_duration_ms: Option<u64>,
}

/// Exporter health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterHealth {
    pub is_healthy: bool,
    pub last_export: Option<DateTime<Utc>>,
    pub error_count: u32,
    pub export_duration_ms: Option<u64>,
}

/// Collector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    pub name: String,
    pub enabled: bool,
    pub collection_interval: Duration,
    pub timeout: Duration,
    pub retry_count: u32,
}

/// Exporter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterConfig {
    pub name: String,
    pub enabled: bool,
    pub export_interval: Duration,
    pub timeout: Duration,
    pub retry_count: u32,
    pub endpoint: Option<String>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: 30,
            retention_hours: 24,
            enable_resource_monitoring: true,
            enable_profiling: false,
            alert_thresholds: AlertThresholds::default(),
            export_config: ExportConfig::default(),
            sampling_config: SamplingConfig::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            response_time_threshold: 1000,
            error_rate_threshold: 5.0,
            disk_threshold: 90.0,
            network_latency_threshold: 500,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            prometheus_endpoint: "0.0.0.0:9090".to_string(),
            enable_json: false,
            json_export_path: "/tmp/metrics.json".to_string(),
            export_interval: 60,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            metrics_sampling_rate: 1.0,
            trace_sampling_rate: 0.1,
            adaptive_sampling: false,
            high_frequency_metrics: vec![
                "cpu_usage".to_string(),
                "memory_usage".to_string(),
                "response_time".to_string(),
            ],
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_metrics_collected: 0,
            total_alerts_triggered: 0,
            avg_collection_time_ms: 0.0,
            uptime_seconds: 0,
            active_collectors: 0,
            active_exporters: 0,
            last_collection: None,
            trends: PerformanceTrends::default(),
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            cpu_trend: Vec::new(),
            memory_trend: Vec::new(),
            response_time_trend: Vec::new(),
            error_rate_trend: Vec::new(),
            throughput_trend: Vec::new(),
        }
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            collectors: Arc::new(RwLock::new(HashMap::new())),
            alert_manager: Arc::new(alerts::AlertManager::new(config.alert_thresholds.clone())),
            resource_tracker: Arc::new(resource_tracker::ResourceTracker::new()),
            exporters: Arc::new(RwLock::new(Vec::new())),
            config,
            start_time: Instant::now(),
            stats: Arc::new(RwLock::new(PerformanceStats::default())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MonitoringConfig::default())
    }

    /// Start the performance monitoring system
    pub async fn start(&self) -> Result<()> {
        info!("Starting performance monitoring system");

        // Start resource tracking if enabled
        if self.config.enable_resource_monitoring {
            self.resource_tracker.start().await?;
        }

        // Start alert manager
        self.alert_manager.start().await?;

        // Start collection loop
        self.start_collection_loop().await?;

        // Start export loop
        self.start_export_loop().await?;

        info!("Performance monitoring system started successfully");
        Ok(())
    }

    /// Stop the performance monitoring system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping performance monitoring system");

        // Stop resource tracking
        self.resource_tracker.stop().await?;

        // Stop alert manager
        self.alert_manager.stop().await?;

        info!("Performance monitoring system stopped");
        Ok(())
    }

    /// Add a metrics collector
    pub async fn add_collector(&self, collector: Box<dyn MetricsCollector>) -> Result<()> {
        let name = collector.name().to_string();
        let mut collectors = self.collectors.write().await;
        collectors.insert(name.clone(), collector);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.active_collectors = collectors.len() as u32;
        }

        info!("Added metrics collector: {}", name);
        Ok(())
    }

    /// Add application metrics collector with default configuration
    pub async fn add_application_collector(&self) -> Result<()> {
        use crate::monitoring::collectors::ApplicationMetricsCollector;

        let config = CollectorConfig {
            name: "application".to_string(),
            enabled: true,
            collection_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            retry_count: 3,
        };

        let collector = ApplicationMetricsCollector::new(config);
        self.add_collector(Box::new(collector)).await?;

        info!("Added application metrics collector");
        Ok(())
    }

    /// Add HTTP metrics collector with default configuration
    pub async fn add_http_collector(&self) -> Result<()> {
        use crate::monitoring::collectors::HttpMetricsCollector;

        let config = CollectorConfig {
            name: "http".to_string(),
            enabled: true,
            collection_interval: Duration::from_secs(15),
            timeout: Duration::from_secs(5),
            retry_count: 2,
        };

        let collector = HttpMetricsCollector::new(config);
        self.add_collector(Box::new(collector)).await?;

        info!("Added HTTP metrics collector");
        Ok(())
    }

    /// Add custom metrics collector with default configuration
    pub async fn add_custom_collector(&self) -> Result<()> {
        use crate::monitoring::collectors::CustomMetricsCollector;

        let config = CollectorConfig {
            name: "custom".to_string(),
            enabled: true,
            collection_interval: Duration::from_secs(60),
            timeout: Duration::from_secs(15),
            retry_count: 1,
        };

        let collector = CustomMetricsCollector::new(config);
        self.add_collector(Box::new(collector)).await?;

        info!("Added custom metrics collector");
        Ok(())
    }

    /// Initialize all default collectors
    pub async fn initialize_default_collectors(&self) -> Result<()> {
        info!("Initializing default metrics collectors");

        // Add application metrics collector
        if let Err(e) = self.add_application_collector().await {
            warn!("Failed to add application collector: {}", e);
        }

        // Add HTTP metrics collector
        if let Err(e) = self.add_http_collector().await {
            warn!("Failed to add HTTP collector: {}", e);
        }

        // Add custom metrics collector
        if let Err(e) = self.add_custom_collector().await {
            warn!("Failed to add custom collector: {}", e);
        }

        info!("Default metrics collectors initialized");
        Ok(())
    }

    /// Remove a metrics collector
    pub async fn remove_collector(&self, name: &str) -> Result<()> {
        let mut collectors = self.collectors.write().await;
        if collectors.remove(name).is_some() {
            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.active_collectors = collectors.len() as u32;
            }
            info!("Removed metrics collector: {}", name);
            Ok(())
        } else {
            Err(AgentError::validation(format!("Collector not found: {}", name)))
        }
    }

    /// Add a metrics exporter
    pub async fn add_exporter(&self, exporter: Box<dyn MetricsExporter>) -> Result<()> {
        let name = exporter.name().to_string();
        let mut exporters = self.exporters.write().await;
        exporters.push(exporter);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.active_exporters = exporters.len() as u32;
        }

        info!("Added metrics exporter: {}", name);
        Ok(())
    }

    /// Add Prometheus exporter with default configuration
    pub async fn add_prometheus_exporter(&self) -> Result<()> {
        use crate::monitoring::exporters::PrometheusExporter;

        let config = ExporterConfig {
            name: "prometheus".to_string(),
            enabled: true,
            export_interval: Duration::from_secs(self.config.export_config.export_interval),
            timeout: Duration::from_secs(10),
            retry_count: 3,
            endpoint: Some(self.config.export_config.prometheus_endpoint.clone()),
        };

        let exporter = PrometheusExporter::new(config, self.config.export_config.prometheus_endpoint.clone());
        self.add_exporter(Box::new(exporter)).await?;

        info!("Added Prometheus metrics exporter");
        Ok(())
    }

    /// Add console exporter with default configuration
    pub async fn add_console_exporter(&self) -> Result<()> {
        use crate::monitoring::exporters::ConsoleExporter;

        let config = ExporterConfig {
            name: "console".to_string(),
            enabled: true,
            export_interval: Duration::from_secs(60),
            timeout: Duration::from_secs(5),
            retry_count: 1,
            endpoint: None,
        };

        let exporter = ConsoleExporter::new(config, crate::monitoring::exporters::ConsoleFormat::Table);
        self.add_exporter(Box::new(exporter)).await?;

        info!("Added console metrics exporter");
        Ok(())
    }

    /// Add file exporter with default configuration
    pub async fn add_file_exporter(&self, file_path: &str) -> Result<()> {
        use crate::monitoring::exporters::CsvFileExporter;

        let config = ExporterConfig {
            name: "file".to_string(),
            enabled: true,
            export_interval: Duration::from_secs(120),
            timeout: Duration::from_secs(15),
            retry_count: 2,
            endpoint: Some(file_path.to_string()),
        };

        let exporter = CsvFileExporter::new(config);
        self.add_exporter(Box::new(exporter)).await?;

        info!("Added file metrics exporter: {}", file_path);
        Ok(())
    }

    /// Initialize all default exporters
    pub async fn initialize_default_exporters(&self) -> Result<()> {
        info!("Initializing default metrics exporters");

        // Add console exporter for development
        if let Err(e) = self.add_console_exporter().await {
            warn!("Failed to add console exporter: {}", e);
        }

        // Add Prometheus exporter if enabled
        if self.config.export_config.enable_prometheus {
            if let Err(e) = self.add_prometheus_exporter().await {
                warn!("Failed to add Prometheus exporter: {}", e);
            }
        }

        // Add file exporter if JSON export is enabled
        if self.config.export_config.enable_json {
            if let Err(e) = self.add_file_exporter(&self.config.export_config.json_export_path).await {
                warn!("Failed to add file exporter: {}", e);
            }
        }

        info!("Default metrics exporters initialized");
        Ok(())
    }

    /// Initialize complete monitoring system with all components
    pub async fn initialize_complete_monitoring(&self) -> Result<()> {
        info!("Initializing complete monitoring system");

        // Initialize all default collectors
        self.initialize_default_collectors().await?;

        // Initialize all default exporters
        self.initialize_default_exporters().await?;

        // Start the monitoring system
        self.start().await?;

        info!("Complete monitoring system initialized and started");
        Ok(())
    }

    /// Get comprehensive monitoring status
    pub async fn get_monitoring_status(&self) -> Result<MonitoringStatus> {
        let stats = self.stats.read().await;
        let collectors = self.collectors.read().await;
        let exporters = self.exporters.read().await;

        // Get collector health status
        let mut collector_health = Vec::new();
        for (name, collector) in collectors.iter() {
            match collector.health_check().await {
                Ok(health) => collector_health.push((name.clone(), health)),
                Err(e) => {
                    warn!("Failed to get health for collector {}: {}", name, e);
                }
            }
        }

        // Get exporter health status
        let mut exporter_health = Vec::new();
        for exporter in exporters.iter() {
            match exporter.health_check().await {
                Ok(health) => exporter_health.push((exporter.name().to_string(), health)),
                Err(e) => {
                    warn!("Failed to get health for exporter {}: {}", exporter.name(), e);
                }
            }
        }

        // Get resource tracker status
        let resource_status = self.resource_tracker.get_stats().await;

        Ok(MonitoringStatus {
            performance_stats: stats.clone(),
            collector_health,
            exporter_health,
            resource_status,
            system_uptime: self.start_time.elapsed(),
            monitoring_config: self.config.clone(),
        })
    }

    /// Record application metric through the monitoring system
    pub async fn record_application_metric(&self, name: &str, value: f64, labels: HashMap<String, String>) -> Result<()> {
        let collectors = self.collectors.read().await;
        if let Some(collector) = collectors.get("application") {
            if let Some(app_collector) = collector.as_any().downcast_ref::<crate::monitoring::collectors::ApplicationMetricsCollector>() {
                app_collector.record_metric(name, value, labels).await?;
                info!("Recorded application metric: {} = {}", name, value);
            }
        }
        Ok(())
    }

    /// Record HTTP request through the monitoring system
    pub async fn record_http_request(&self, endpoint: &str, status_code: u16, response_time: f64) -> Result<()> {
        let collectors = self.collectors.read().await;
        if let Some(collector) = collectors.get("http") {
            if let Some(http_collector) = collector.as_any().downcast_ref::<crate::monitoring::collectors::HttpMetricsCollector>() {
                http_collector.record_request(endpoint, status_code, response_time).await?;
                info!("Recorded HTTP request: {} {} {}ms", endpoint, status_code, response_time);
            }
        }
        Ok(())
    }

    /// Add custom metric through the monitoring system
    pub async fn add_custom_metric(&self, metric: crate::monitoring::collectors::CustomMetric) -> Result<()> {
        let collectors = self.collectors.read().await;
        if let Some(collector) = collectors.get("custom") {
            if let Some(custom_collector) = collector.as_any().downcast_ref::<crate::monitoring::collectors::CustomMetricsCollector>() {
                custom_collector.add_metric(metric.clone()).await?;
                info!("Added custom metric: {}", metric.name);
            }
        }
        Ok(())
    }

    /// Collect metrics from all collectors
    pub async fn collect_metrics(&self) -> Result<Vec<Metric>> {
        let start_time = Instant::now();
        let mut all_metrics = Vec::new();

        let collectors = self.collectors.read().await;
        for (name, collector) in collectors.iter() {
            match collector.collect().await {
                Ok(mut metrics) => {
                    debug!("Collected {} metrics from {}", metrics.len(), name);
                    all_metrics.append(&mut metrics);
                }
                Err(e) => {
                    warn!("Failed to collect metrics from {}: {}", name, e);
                }
            }
        }

        // Update collection statistics
        let collection_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_metrics_collected += all_metrics.len() as u64;
            stats.last_collection = Some(Utc::now());

            // Update average collection time
            let new_avg = (stats.avg_collection_time_ms + collection_time.as_millis() as f64) / 2.0;
            stats.avg_collection_time_ms = new_avg;
        }

        debug!("Collected {} total metrics in {:?}", all_metrics.len(), collection_time);
        Ok(all_metrics)
    }

    /// Export metrics using all exporters
    pub async fn export_metrics(&self, metrics: &[Metric]) -> Result<()> {
        let exporters = self.exporters.read().await;

        for exporter in exporters.iter() {
            match exporter.export(metrics).await {
                Ok(_) => {
                    debug!("Successfully exported {} metrics via {}", metrics.len(), exporter.name());
                }
                Err(e) => {
                    warn!("Failed to export metrics via {}: {}", exporter.name(), e);
                }
            }
        }

        Ok(())
    }

    /// Get performance statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        let mut stats = self.stats.read().await.clone();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();
        stats
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> Result<SystemHealth> {
        let mut collector_health = HashMap::new();
        let mut exporter_health = HashMap::new();

        // Check collector health
        let collectors = self.collectors.read().await;
        for (name, collector) in collectors.iter() {
            match collector.health_check().await {
                Ok(health) => {
                    collector_health.insert(name.clone(), health);
                }
                Err(e) => {
                    warn!("Health check failed for collector {}: {}", name, e);
                    collector_health.insert(name.clone(), CollectorHealth {
                        is_healthy: false,
                        last_collection: None,
                        error_count: 1,
                        collection_duration_ms: None,
                    });
                }
            }
        }

        // Check exporter health
        let exporters = self.exporters.read().await;
        for exporter in exporters.iter() {
            match exporter.health_check().await {
                Ok(health) => {
                    exporter_health.insert(exporter.name().to_string(), health);
                }
                Err(e) => {
                    warn!("Health check failed for exporter {}: {}", exporter.name(), e);
                    exporter_health.insert(exporter.name().to_string(), ExporterHealth {
                        is_healthy: false,
                        last_export: None,
                        error_count: 1,
                        export_duration_ms: None,
                    });
                }
            }
        }

        let overall_healthy = collector_health.values().all(|h| h.is_healthy) &&
                             exporter_health.values().all(|h| h.is_healthy);

        Ok(SystemHealth {
            overall_healthy,
            collector_health,
            exporter_health,
            resource_health: self.resource_tracker.get_health().await?,
            alert_manager_health: self.alert_manager.get_health().await?,
        })
    }

    /// Start the metrics collection loop
    async fn start_collection_loop(&self) -> Result<()> {
        let interval = Duration::from_secs(self.config.collection_interval);
        let monitor = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;

                match monitor.collect_metrics().await {
                    Ok(metrics) => {
                        // Check for alerts
                        if let Err(e) = monitor.alert_manager.check_metrics(&metrics).await {
                            error!("Alert checking failed: {}", e);
                        }

                        // Update trends
                        if let Err(e) = monitor.update_trends(&metrics).await {
                            error!("Trend update failed: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Metrics collection failed: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Start the metrics export loop
    async fn start_export_loop(&self) -> Result<()> {
        let interval = Duration::from_secs(self.config.export_config.export_interval);
        let monitor = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;

                match monitor.collect_metrics().await {
                    Ok(metrics) => {
                        if let Err(e) = monitor.export_metrics(&metrics).await {
                            error!("Metrics export failed: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Metrics collection for export failed: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Update performance trends
    async fn update_trends(&self, metrics: &[Metric]) -> Result<()> {
        let mut stats = self.stats.write().await;

        // Extract key metrics for trends
        for metric in metrics {
            match metric.name.as_str() {
                "cpu_usage" => {
                    if let MetricValue::Gauge(value) = metric.value {
                        stats.trends.cpu_trend.push(value);
                        if stats.trends.cpu_trend.len() > 1440 { // Keep 24 hours at 1-minute intervals
                            stats.trends.cpu_trend.remove(0);
                        }
                    }
                }
                "memory_usage" => {
                    if let MetricValue::Gauge(value) = metric.value {
                        stats.trends.memory_trend.push(value);
                        if stats.trends.memory_trend.len() > 1440 {
                            stats.trends.memory_trend.remove(0);
                        }
                    }
                }
                "response_time" => {
                    if let MetricValue::Gauge(value) = metric.value {
                        stats.trends.response_time_trend.push(value);
                        if stats.trends.response_time_trend.len() > 1440 {
                            stats.trends.response_time_trend.remove(0);
                        }
                    }
                }
                "error_rate" => {
                    if let MetricValue::Gauge(value) = metric.value {
                        stats.trends.error_rate_trend.push(value);
                        if stats.trends.error_rate_trend.len() > 1440 {
                            stats.trends.error_rate_trend.remove(0);
                        }
                    }
                }
                "throughput" => {
                    if let MetricValue::Gauge(value) = metric.value {
                        stats.trends.throughput_trend.push(value);
                        if stats.trends.throughput_trend.len() > 1440 {
                            stats.trends.throughput_trend.remove(0);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

impl Clone for PerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            collectors: Arc::clone(&self.collectors),
            alert_manager: Arc::clone(&self.alert_manager),
            resource_tracker: Arc::clone(&self.resource_tracker),
            exporters: Arc::clone(&self.exporters),
            config: self.config.clone(),
            start_time: self.start_time,
            stats: Arc::clone(&self.stats),
        }
    }
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_healthy: bool,
    pub collector_health: HashMap<String, CollectorHealth>,
    pub exporter_health: HashMap<String, ExporterHealth>,
    pub resource_health: resource_tracker::ResourceHealth,
    pub alert_manager_health: alerts::AlertManagerHealth,
}
