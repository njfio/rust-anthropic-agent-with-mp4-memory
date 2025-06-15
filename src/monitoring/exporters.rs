// Metrics Exporters for Performance Monitoring
// Provides various export formats and destinations for metrics

use super::{ExporterConfig, ExporterHealth, Metric, MetricValue, MetricsExporter};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::debug;

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    /// Exporter configuration
    config: ExporterConfig,
    /// Export statistics
    stats: Arc<RwLock<ExportStats>>,
    /// Prometheus endpoint
    endpoint: String,
}

/// JSON file exporter
pub struct JsonFileExporter {
    /// Exporter configuration
    config: ExporterConfig,
    /// Export statistics
    stats: Arc<RwLock<ExportStats>>,
    /// Output file path
    file_path: String,
}

/// CSV file exporter
pub struct CsvFileExporter {
    /// Exporter configuration
    config: ExporterConfig,
    /// Export statistics
    stats: Arc<RwLock<ExportStats>>,
    /// Output file path
    file_path: String,
}

impl CsvFileExporter {
    /// Create a new CSV file exporter
    pub fn new(config: ExporterConfig) -> Self {
        let file_path = config.endpoint.clone().unwrap_or_else(|| "/tmp/metrics.csv".to_string());
        Self {
            config,
            stats: Arc::new(RwLock::new(ExportStats::default())),
            file_path,
        }
    }
}

#[async_trait::async_trait]
impl MetricsExporter for CsvFileExporter {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn export(&self, metrics: &[Metric]) -> Result<()> {
        // Simple CSV export implementation
        let csv_content = self.format_metrics_as_csv(metrics);

        // Write to file (simplified implementation)
        std::fs::write(&self.file_path, csv_content)
            .map_err(|e| AgentError::config(format!("Failed to write CSV file: {}", e)))?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_exports += 1;
        stats.successful_exports += 1;
        stats.total_metrics_exported += metrics.len() as u64;

        Ok(())
    }

    async fn health_check(&self) -> Result<ExporterHealth> {
        let stats = self.stats.read().await;
        Ok(ExporterHealth {
            is_healthy: stats.last_error.is_none(),
            last_export: stats.last_export,
            error_count: stats.failed_exports as u32,
            export_duration_ms: Some(stats.avg_export_time as u64),
        })
    }

    fn config(&self) -> ExporterConfig {
        self.config.clone()
    }
}

impl CsvFileExporter {
    fn format_metrics_as_csv(&self, metrics: &[Metric]) -> String {
        let mut csv = String::from("name,value,timestamp,type\n");
        for metric in metrics {
            let value_str = match &metric.value {
                super::MetricValue::Counter(v) => v.to_string(),
                super::MetricValue::Gauge(v) => v.to_string(),
                super::MetricValue::Histogram(_) => "histogram".to_string(),
                super::MetricValue::Summary(_) => "summary".to_string(),
            };
            csv.push_str(&format!("{},{},{},{:?}\n",
                metric.name, value_str, metric.timestamp, metric.metric_type));
        }
        csv
    }
}

/// Console exporter for debugging
pub struct ConsoleExporter {
    /// Exporter configuration
    config: ExporterConfig,
    /// Export statistics
    stats: Arc<RwLock<ExportStats>>,
    /// Output format
    format: ConsoleFormat,
}

/// HTTP endpoint exporter
pub struct HttpExporter {
    /// Exporter configuration
    config: ExporterConfig,
    /// Export statistics
    stats: Arc<RwLock<ExportStats>>,
    /// Target endpoint URL
    endpoint_url: String,
    /// HTTP client
    client: reqwest::Client,
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    /// Total exports performed
    pub total_exports: u64,
    /// Successful exports
    pub successful_exports: u64,
    /// Failed exports
    pub failed_exports: u64,
    /// Average export time (milliseconds)
    pub avg_export_time: f64,
    /// Last export timestamp
    pub last_export: Option<DateTime<Utc>>,
    /// Last error
    pub last_error: Option<String>,
    /// Total metrics exported
    pub total_metrics_exported: u64,
}

/// Console output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsoleFormat {
    Table,
    Json,
    Plain,
}

/// Prometheus metric format
#[derive(Debug, Clone)]
pub struct PrometheusMetric {
    pub name: String,
    pub help: Option<String>,
    pub metric_type: String,
    pub samples: Vec<PrometheusSample>,
}

/// Prometheus sample
#[derive(Debug, Clone)]
pub struct PrometheusSample {
    pub labels: HashMap<String, String>,
    pub value: f64,
    pub timestamp: Option<i64>,
}

impl Default for ExportStats {
    fn default() -> Self {
        Self {
            total_exports: 0,
            successful_exports: 0,
            failed_exports: 0,
            avg_export_time: 0.0,
            last_export: None,
            last_error: None,
            total_metrics_exported: 0,
        }
    }
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(config: ExporterConfig, endpoint: String) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExportStats::default())),
            endpoint,
        }
    }

    /// Convert metrics to Prometheus format
    fn convert_to_prometheus_format(&self, metrics: &[Metric]) -> Vec<PrometheusMetric> {
        let mut prometheus_metrics = Vec::new();
        let mut grouped_metrics: HashMap<String, Vec<&Metric>> = HashMap::new();

        // Group metrics by name
        for metric in metrics {
            grouped_metrics.entry(metric.name.clone()).or_insert_with(Vec::new).push(metric);
        }

        // Convert each group to Prometheus format
        for (name, metric_group) in grouped_metrics {
            let first_metric = metric_group[0];
            let metric_type = match first_metric.metric_type {
                super::MetricType::Counter => "counter",
                super::MetricType::Gauge => "gauge",
                super::MetricType::Histogram => "histogram",
                super::MetricType::Summary => "summary",
            };

            let mut samples = Vec::new();
            for metric in metric_group {
                let value = match &metric.value {
                    MetricValue::Counter(v) => *v as f64,
                    MetricValue::Gauge(v) => *v,
                    MetricValue::Histogram(h) => h.sum,
                    MetricValue::Summary(s) => s.sum,
                };

                samples.push(PrometheusSample {
                    labels: metric.labels.clone(),
                    value,
                    timestamp: Some(metric.timestamp.timestamp_millis()),
                });
            }

            prometheus_metrics.push(PrometheusMetric {
                name: name.clone(),
                help: first_metric.help.clone(),
                metric_type: metric_type.to_string(),
                samples,
            });
        }

        prometheus_metrics
    }

    /// Format metrics as Prometheus text
    fn format_prometheus_text(&self, prometheus_metrics: &[PrometheusMetric]) -> String {
        let mut output = String::new();

        for metric in prometheus_metrics {
            // Add help comment
            if let Some(help) = &metric.help {
                output.push_str(&format!("# HELP {} {}\n", metric.name, help));
            }

            // Add type comment
            output.push_str(&format!("# TYPE {} {}\n", metric.name, metric.metric_type));

            // Add samples
            for sample in &metric.samples {
                let labels_str = if sample.labels.is_empty() {
                    String::new()
                } else {
                    let labels: Vec<String> = sample.labels
                        .iter()
                        .map(|(k, v)| format!("{}=\"{}\"", k, v))
                        .collect();
                    format!("{{{}}}", labels.join(","))
                };

                output.push_str(&format!("{}{} {}\n", metric.name, labels_str, sample.value));
            }

            output.push('\n');
        }

        output
    }
}

#[async_trait::async_trait]
impl MetricsExporter for PrometheusExporter {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn export(&self, metrics: &[Metric]) -> Result<()> {
        let start_time = Instant::now();

        // Convert to Prometheus format
        let prometheus_metrics = self.convert_to_prometheus_format(metrics);
        let prometheus_text = self.format_prometheus_text(&prometheus_metrics);

        // TODO: Implement actual HTTP server for Prometheus scraping
        // For now, we'll just log the metrics
        debug!("Prometheus metrics:\n{}", prometheus_text);

        // Update export statistics
        let export_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_exports += 1;
            stats.successful_exports += 1;
            stats.last_export = Some(Utc::now());
            stats.avg_export_time = (stats.avg_export_time + export_time.as_millis() as f64) / 2.0;
            stats.total_metrics_exported += metrics.len() as u64;
        }

        debug!("Exported {} metrics to Prometheus format", metrics.len());
        Ok(())
    }

    async fn health_check(&self) -> Result<ExporterHealth> {
        let stats = self.stats.read().await;
        Ok(ExporterHealth {
            is_healthy: stats.last_error.is_none(),
            last_export: stats.last_export,
            error_count: stats.failed_exports as u32,
            export_duration_ms: Some(stats.avg_export_time as u64),
        })
    }

    fn config(&self) -> ExporterConfig {
        self.config.clone()
    }
}

impl JsonFileExporter {
    /// Create a new JSON file exporter
    pub fn new(config: ExporterConfig, file_path: String) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExportStats::default())),
            file_path,
        }
    }
}

#[async_trait::async_trait]
impl MetricsExporter for JsonFileExporter {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn export(&self, metrics: &[Metric]) -> Result<()> {
        let start_time = Instant::now();

        // Serialize metrics to JSON
        let json_data = serde_json::to_string_pretty(metrics)
            .map_err(|e| AgentError::tool("json_exporter", &format!("JSON serialization failed: {}", e)))?;

        // Write to file
        let mut file = File::create(&self.file_path)
            .map_err(|e| AgentError::tool("json_exporter", &format!("Failed to create file: {}", e)))?;

        file.write_all(json_data.as_bytes())
            .map_err(|e| AgentError::tool("json_exporter", &format!("Failed to write file: {}", e)))?;

        // Update export statistics
        let export_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_exports += 1;
            stats.successful_exports += 1;
            stats.last_export = Some(Utc::now());
            stats.avg_export_time = (stats.avg_export_time + export_time.as_millis() as f64) / 2.0;
            stats.total_metrics_exported += metrics.len() as u64;
        }

        debug!("Exported {} metrics to JSON file: {}", metrics.len(), self.file_path);
        Ok(())
    }

    async fn health_check(&self) -> Result<ExporterHealth> {
        let stats = self.stats.read().await;
        
        // Check if file path is writable
        let path_healthy = Path::new(&self.file_path).parent()
            .map(|p| p.exists())
            .unwrap_or(false);

        Ok(ExporterHealth {
            is_healthy: stats.last_error.is_none() && path_healthy,
            last_export: stats.last_export,
            error_count: stats.failed_exports as u32,
            export_duration_ms: Some(stats.avg_export_time as u64),
        })
    }

    fn config(&self) -> ExporterConfig {
        self.config.clone()
    }
}

impl ConsoleExporter {
    /// Create a new console exporter
    pub fn new(config: ExporterConfig, format: ConsoleFormat) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExportStats::default())),
            format,
        }
    }

    /// Format metrics for console output
    fn format_metrics(&self, metrics: &[Metric]) -> String {
        match self.format {
            ConsoleFormat::Json => {
                serde_json::to_string_pretty(metrics).unwrap_or_else(|_| "JSON serialization failed".to_string())
            }
            ConsoleFormat::Table => {
                let mut output = String::new();
                output.push_str("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
                output.push_str("│                                    METRICS                                     │\n");
                output.push_str("├─────────────────────────────────────────────────────────────────────────────────┤\n");
                
                for metric in metrics {
                    let value_str = match &metric.value {
                        MetricValue::Counter(v) => format!("{}", v),
                        MetricValue::Gauge(v) => format!("{:.2}", v),
                        MetricValue::Histogram(h) => format!("sum={:.2}, count={}", h.sum, h.count),
                        MetricValue::Summary(s) => format!("sum={:.2}, count={}", s.sum, s.count),
                    };
                    
                    output.push_str(&format!("│ {:<30} │ {:<20} │ {:<25} │\n", 
                        metric.name, 
                        format!("{:?}", metric.metric_type),
                        value_str
                    ));
                }
                
                output.push_str("└─────────────────────────────────────────────────────────────────────────────────┘\n");
                output
            }
            ConsoleFormat::Plain => {
                let mut output = String::new();
                for metric in metrics {
                    let value_str = match &metric.value {
                        MetricValue::Counter(v) => format!("{}", v),
                        MetricValue::Gauge(v) => format!("{:.2}", v),
                        MetricValue::Histogram(h) => format!("sum={:.2}, count={}", h.sum, h.count),
                        MetricValue::Summary(s) => format!("sum={:.2}, count={}", s.sum, s.count),
                    };
                    
                    output.push_str(&format!("{}: {} ({})\n", metric.name, value_str, format!("{:?}", metric.metric_type)));
                }
                output
            }
        }
    }
}

#[async_trait::async_trait]
impl MetricsExporter for ConsoleExporter {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn export(&self, metrics: &[Metric]) -> Result<()> {
        let start_time = Instant::now();

        // Format and print metrics
        let formatted_output = self.format_metrics(metrics);
        println!("{}", formatted_output);

        // Update export statistics
        let export_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_exports += 1;
            stats.successful_exports += 1;
            stats.last_export = Some(Utc::now());
            stats.avg_export_time = (stats.avg_export_time + export_time.as_millis() as f64) / 2.0;
            stats.total_metrics_exported += metrics.len() as u64;
        }

        debug!("Exported {} metrics to console", metrics.len());
        Ok(())
    }

    async fn health_check(&self) -> Result<ExporterHealth> {
        let stats = self.stats.read().await;
        Ok(ExporterHealth {
            is_healthy: true, // Console is always available
            last_export: stats.last_export,
            error_count: stats.failed_exports as u32,
            export_duration_ms: Some(stats.avg_export_time as u64),
        })
    }

    fn config(&self) -> ExporterConfig {
        self.config.clone()
    }
}

impl HttpExporter {
    /// Create a new HTTP exporter
    pub fn new(config: ExporterConfig, endpoint_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            config,
            stats: Arc::new(RwLock::new(ExportStats::default())),
            endpoint_url,
            client,
        }
    }
}

#[async_trait::async_trait]
impl MetricsExporter for HttpExporter {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn export(&self, metrics: &[Metric]) -> Result<()> {
        let start_time = Instant::now();

        // Serialize metrics to JSON
        let json_data = serde_json::to_string(metrics)
            .map_err(|e| AgentError::tool("http_exporter", &format!("JSON serialization failed: {}", e)))?;

        // Send HTTP POST request
        let response = self.client
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .body(json_data)
            .send()
            .await
            .map_err(|e| AgentError::tool("http_exporter", &format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(AgentError::tool("http_exporter", 
                &format!("HTTP request failed with status: {}", response.status())));
        }

        // Update export statistics
        let export_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.total_exports += 1;
            stats.successful_exports += 1;
            stats.last_export = Some(Utc::now());
            stats.avg_export_time = (stats.avg_export_time + export_time.as_millis() as f64) / 2.0;
            stats.total_metrics_exported += metrics.len() as u64;
        }

        debug!("Exported {} metrics to HTTP endpoint: {}", metrics.len(), self.endpoint_url);
        Ok(())
    }

    async fn health_check(&self) -> Result<ExporterHealth> {
        let stats = self.stats.read().await;
        
        // Test endpoint connectivity
        let endpoint_healthy = self.client
            .head(&self.endpoint_url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        Ok(ExporterHealth {
            is_healthy: stats.last_error.is_none() && endpoint_healthy,
            last_export: stats.last_export,
            error_count: stats.failed_exports as u32,
            export_duration_ms: Some(stats.avg_export_time as u64),
        })
    }

    fn config(&self) -> ExporterConfig {
        self.config.clone()
    }
}
