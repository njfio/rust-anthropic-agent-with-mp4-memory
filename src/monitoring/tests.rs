// Comprehensive Tests for Performance Monitoring System
// Tests all components: monitoring, alerts, collectors, exporters, metrics, and resource tracking

use super::*;
use crate::monitoring::{alerts::*, collectors::*, exporters::*, metrics::*, resource_tracker::*};
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;
use tokio;

#[tokio::test]
async fn test_performance_monitor_creation() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    let stats = monitor.get_stats().await;
    assert_eq!(stats.total_metrics_collected, 0);
    assert_eq!(stats.active_collectors, 0);
    assert_eq!(stats.active_exporters, 0);
}

#[tokio::test]
async fn test_performance_monitor_with_defaults() {
    let monitor = PerformanceMonitor::with_defaults();

    let stats = monitor.get_stats().await;
    assert_eq!(stats.total_metrics_collected, 0);
    assert!(stats.uptime_seconds >= 0);
}

#[tokio::test]
async fn test_monitoring_config_default() {
    let config = MonitoringConfig::default();

    assert_eq!(config.collection_interval, 30);
    assert_eq!(config.retention_hours, 24);
    assert!(config.enable_resource_monitoring);
    assert!(!config.enable_profiling);
    assert_eq!(config.alert_thresholds.cpu_threshold, 80.0);
    assert_eq!(config.export_config.export_interval, 60);
}

#[tokio::test]
async fn test_alert_thresholds_default() {
    let thresholds = AlertThresholds::default();

    assert_eq!(thresholds.cpu_threshold, 80.0);
    assert_eq!(thresholds.memory_threshold, 85.0);
    assert_eq!(thresholds.response_time_threshold, 1000);
    assert_eq!(thresholds.error_rate_threshold, 5.0);
    assert_eq!(thresholds.disk_threshold, 90.0);
    assert_eq!(thresholds.network_latency_threshold, 500);
}

#[tokio::test]
async fn test_export_config_default() {
    let config = ExportConfig::default();

    assert!(config.enable_prometheus);
    assert_eq!(config.prometheus_endpoint, "0.0.0.0:9090");
    assert!(!config.enable_json);
    assert_eq!(config.json_export_path, "/tmp/metrics.json");
    assert_eq!(config.export_interval, 60);
}

#[tokio::test]
async fn test_sampling_config_default() {
    let config = SamplingConfig::default();

    assert_eq!(config.metrics_sampling_rate, 1.0);
    assert_eq!(config.trace_sampling_rate, 0.1);
    assert!(!config.adaptive_sampling);
    assert_eq!(config.high_frequency_metrics.len(), 3);
    assert!(config
        .high_frequency_metrics
        .contains(&"cpu_usage".to_string()));
}

#[tokio::test]
async fn test_performance_stats_default() {
    let stats = PerformanceStats::default();

    assert_eq!(stats.total_metrics_collected, 0);
    assert_eq!(stats.total_alerts_triggered, 0);
    assert_eq!(stats.avg_collection_time_ms, 0.0);
    assert_eq!(stats.active_collectors, 0);
    assert_eq!(stats.active_exporters, 0);
    assert!(stats.last_collection.is_none());
}

#[tokio::test]
async fn test_performance_trends_default() {
    let trends = PerformanceTrends::default();

    assert!(trends.cpu_trend.is_empty());
    assert!(trends.memory_trend.is_empty());
    assert!(trends.response_time_trend.is_empty());
    assert!(trends.error_rate_trend.is_empty());
    assert!(trends.throughput_trend.is_empty());
}

#[tokio::test]
async fn test_metric_creation() {
    let metric = Metric {
        name: "test_metric".to_string(),
        value: MetricValue::Gauge(42.0),
        labels: HashMap::new(),
        timestamp: Utc::now(),
        metric_type: MetricType::Gauge,
        help: Some("Test metric".to_string()),
    };

    assert_eq!(metric.name, "test_metric");
    assert!(matches!(metric.value, MetricValue::Gauge(42.0)));
    assert!(matches!(metric.metric_type, MetricType::Gauge));
    assert!(metric.help.is_some());
}

#[tokio::test]
async fn test_metric_value_types() {
    let counter = MetricValue::Counter(100);
    let gauge = MetricValue::Gauge(75.5);
    let histogram = MetricValue::Histogram(HistogramData {
        buckets: vec![HistogramBucket {
            upper_bound: 1.0,
            count: 10,
        }],
        count: 10,
        sum: 5.0,
    });
    let summary = MetricValue::Summary(SummaryData {
        quantiles: vec![Quantile {
            quantile: 0.5,
            value: 2.5,
        }],
        count: 20,
        sum: 50.0,
    });

    assert!(matches!(counter, MetricValue::Counter(100)));
    assert!(matches!(gauge, MetricValue::Gauge(75.5)));
    assert!(matches!(histogram, MetricValue::Histogram(_)));
    assert!(matches!(summary, MetricValue::Summary(_)));
}

#[tokio::test]
async fn test_metric_types() {
    assert_eq!(MetricType::Counter, MetricType::Counter);
    assert_eq!(MetricType::Gauge, MetricType::Gauge);
    assert_eq!(MetricType::Histogram, MetricType::Histogram);
    assert_eq!(MetricType::Summary, MetricType::Summary);
    assert_ne!(MetricType::Counter, MetricType::Gauge);
}

// Alert Manager Tests

#[tokio::test]
async fn test_alert_manager_creation() {
    let thresholds = AlertThresholds::default();
    let manager = AlertManager::new(thresholds);

    let stats = manager.get_stats().await;
    assert_eq!(stats.total_alerts, 0);
    assert_eq!(stats.active_alerts_count, 0);
    assert_eq!(stats.resolved_alerts_count, 0);
}

#[tokio::test]
async fn test_alert_creation() {
    let alert = Alert {
        id: "test_alert_1".to_string(),
        name: "Test Alert".to_string(),
        severity: AlertSeverity::Warning,
        message: "This is a test alert".to_string(),
        metric_name: "cpu_usage".to_string(),
        current_value: 85.0,
        threshold_value: 80.0,
        timestamp: Utc::now(),
        status: AlertStatus::Active,
        labels: HashMap::new(),
        resolved_at: None,
        resolution_reason: None,
    };

    assert_eq!(alert.id, "test_alert_1");
    assert_eq!(alert.severity, AlertSeverity::Warning);
    assert_eq!(alert.status, AlertStatus::Active);
    assert!(alert.resolved_at.is_none());
}

#[tokio::test]
async fn test_alert_severity_levels() {
    let critical = AlertSeverity::Critical;
    let warning = AlertSeverity::Warning;
    let info = AlertSeverity::Info;

    assert_eq!(critical, AlertSeverity::Critical);
    assert_eq!(warning, AlertSeverity::Warning);
    assert_eq!(info, AlertSeverity::Info);
    assert_ne!(critical, warning);
}

#[tokio::test]
async fn test_alert_status_types() {
    let active = AlertStatus::Active;
    let resolved = AlertStatus::Resolved;
    let suppressed = AlertStatus::Suppressed;

    assert_eq!(active, AlertStatus::Active);
    assert_eq!(resolved, AlertStatus::Resolved);
    assert_eq!(suppressed, AlertStatus::Suppressed);
    assert_ne!(active, resolved);
}

#[tokio::test]
async fn test_alert_rule_creation() {
    let rule = AlertRule {
        name: "high_cpu_rule".to_string(),
        metric_name: "cpu_usage".to_string(),
        threshold: 80.0,
        operator: ComparisonOperator::GreaterThan,
        severity: AlertSeverity::Warning,
        evaluation_window: 300,
        min_duration: 60,
        message_template: "CPU usage is {value}%".to_string(),
        labels: HashMap::new(),
        enabled: true,
    };

    assert_eq!(rule.name, "high_cpu_rule");
    assert_eq!(rule.threshold, 80.0);
    assert_eq!(rule.operator, ComparisonOperator::GreaterThan);
    assert!(rule.enabled);
}

#[tokio::test]
async fn test_comparison_operators() {
    assert_eq!(
        ComparisonOperator::GreaterThan,
        ComparisonOperator::GreaterThan
    );
    assert_eq!(ComparisonOperator::LessThan, ComparisonOperator::LessThan);
    assert_eq!(ComparisonOperator::Equal, ComparisonOperator::Equal);
    assert_ne!(
        ComparisonOperator::GreaterThan,
        ComparisonOperator::LessThan
    );
}

#[tokio::test]
async fn test_alert_stats_default() {
    let stats = AlertStats::default();

    assert_eq!(stats.total_alerts, 0);
    assert_eq!(stats.active_alerts_count, 0);
    assert_eq!(stats.resolved_alerts_count, 0);
    assert!(stats.alerts_by_severity.is_empty());
    assert_eq!(stats.avg_resolution_time_seconds, 0.0);
    assert!(stats.last_alert_time.is_none());
}

// Resource Tracker Tests

#[tokio::test]
async fn test_resource_tracker_creation() {
    let tracker = ResourceTracker::new();
    let stats = tracker.get_stats().await;

    assert_eq!(stats.cpu_usage, 0.0);
    assert_eq!(stats.memory_usage, 0.0);
    assert_eq!(stats.total_memory, 0);
    assert!(stats.disk_usage.is_empty());
    assert!(stats.network_stats.is_empty());
}

#[tokio::test]
async fn test_resource_tracker_with_config() {
    let config = ResourceConfig {
        enable_cpu_monitoring: true,
        enable_memory_monitoring: true,
        enable_disk_monitoring: false,
        enable_network_monitoring: false,
        enable_process_monitoring: false,
        monitoring_interval: 60,
        process_whitelist: vec!["test_process".to_string()],
        disk_paths: vec!["/tmp".to_string()],
        network_interfaces: vec!["eth0".to_string()],
    };

    let tracker = ResourceTracker::with_config(config.clone());
    assert_eq!(tracker.get_config().monitoring_interval, 60);
    assert!(!tracker.get_config().enable_disk_monitoring);
}

#[tokio::test]
async fn test_resource_config_default() {
    let config = ResourceConfig::default();

    assert!(config.enable_cpu_monitoring);
    assert!(config.enable_memory_monitoring);
    assert!(config.enable_disk_monitoring);
    assert!(config.enable_network_monitoring);
    assert!(!config.enable_process_monitoring);
    assert_eq!(config.monitoring_interval, 30);
    assert!(config.process_whitelist.is_empty());
    assert_eq!(config.disk_paths, vec!["/"]);
    assert!(config.network_interfaces.is_empty());
}

#[tokio::test]
async fn test_disk_usage_creation() {
    let disk_usage = DiskUsage {
        total_space: 1000000,
        used_space: 750000,
        available_space: 250000,
        usage_percentage: 75.0,
        mount_point: "/".to_string(),
        file_system: "ext4".to_string(),
    };

    assert_eq!(disk_usage.total_space, 1000000);
    assert_eq!(disk_usage.usage_percentage, 75.0);
    assert_eq!(disk_usage.mount_point, "/");
    assert_eq!(disk_usage.file_system, "ext4");
}

#[tokio::test]
async fn test_network_stats_creation() {
    let net_stats = NetworkStats {
        bytes_received: 1024,
        bytes_transmitted: 2048,
        packets_received: 10,
        packets_transmitted: 15,
        errors_received: 0,
        errors_transmitted: 0,
        interface_name: "eth0".to_string(),
    };

    assert_eq!(net_stats.bytes_received, 1024);
    assert_eq!(net_stats.bytes_transmitted, 2048);
    assert_eq!(net_stats.interface_name, "eth0");
}

#[tokio::test]
async fn test_process_stats_creation() {
    let process_stats = ProcessStats {
        pid: 1234,
        name: "test_process".to_string(),
        cpu_usage: 5.5,
        memory_usage: 1024000,
        virtual_memory: 2048000,
        status: "Running".to_string(),
        start_time: 1640995200,
        cmd: vec!["test_process".to_string(), "--arg1".to_string()],
    };

    assert_eq!(process_stats.pid, 1234);
    assert_eq!(process_stats.name, "test_process");
    assert_eq!(process_stats.cpu_usage, 5.5);
    assert_eq!(process_stats.cmd.len(), 2);
}

#[tokio::test]
async fn test_load_averages() {
    let load_averages = LoadAverages {
        one_minute: 1.5,
        five_minute: 1.2,
        fifteen_minute: 1.0,
    };

    assert_eq!(load_averages.one_minute, 1.5);
    assert_eq!(load_averages.five_minute, 1.2);
    assert_eq!(load_averages.fifteen_minute, 1.0);
}

// Metrics Collectors Tests

#[tokio::test]
async fn test_application_metrics_collector() {
    let config = CollectorConfig {
        name: "app_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = ApplicationMetricsCollector::new(config);
    assert_eq!(collector.name(), "app_collector");

    let metrics = collector.get_metrics().await;
    assert!(metrics.is_empty());
}

#[tokio::test]
async fn test_application_metrics_collector_record_metric() {
    let config = CollectorConfig {
        name: "app_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = ApplicationMetricsCollector::new(config);
    let mut labels = HashMap::new();
    labels.insert("service".to_string(), "test".to_string());

    collector
        .record_metric("test_metric", 42.0, labels)
        .await
        .unwrap();

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.len(), 1);
    assert!(metrics.contains_key("test_metric"));
}

#[tokio::test]
async fn test_http_metrics_collector() {
    let config = CollectorConfig {
        name: "http_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = HttpMetricsCollector::new(config);
    assert_eq!(collector.name(), "http_collector");

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.total_requests, 0);
    assert_eq!(metrics.successful_requests, 0);
}

#[tokio::test]
async fn test_http_metrics_collector_record_request() {
    let config = CollectorConfig {
        name: "http_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = HttpMetricsCollector::new(config);

    collector
        .record_request("/api/test", 200, 150.0)
        .await
        .unwrap();

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.total_requests, 1);
    assert_eq!(metrics.successful_requests, 1);
    assert_eq!(metrics.client_errors, 0);
    assert_eq!(metrics.server_errors, 0);
}

#[tokio::test]
async fn test_custom_metrics_collector() {
    let config = CollectorConfig {
        name: "custom_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = CustomMetricsCollector::new(config);
    assert_eq!(collector.name(), "custom_collector");

    let metrics = collector.get_metrics().await;
    assert!(metrics.is_empty());
}

#[tokio::test]
async fn test_custom_metrics_collector_add_metric() {
    let config = CollectorConfig {
        name: "custom_collector".to_string(),
        enabled: true,
        collection_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_count: 3,
    };

    let collector = CustomMetricsCollector::new(config);

    let custom_metric = CustomMetric {
        name: "custom_test".to_string(),
        metric_type: MetricType::Counter,
        value: MetricValue::Counter(100),
        labels: HashMap::new(),
        timestamp: Utc::now(),
        description: Some("Test custom metric".to_string()),
    };

    collector.add_metric(custom_metric).await.unwrap();

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.len(), 1);
    assert!(metrics.contains_key("custom_test"));
}

#[tokio::test]
async fn test_collection_stats_default() {
    let stats = CollectionStats::default();

    assert_eq!(stats.total_collections, 0);
    assert_eq!(stats.successful_collections, 0);
    assert_eq!(stats.failed_collections, 0);
    assert_eq!(stats.avg_collection_time, 0.0);
    assert!(stats.last_collection.is_none());
    assert!(stats.last_error.is_none());
}

// Metrics Exporters Tests

#[tokio::test]
async fn test_prometheus_exporter() {
    let config = ExporterConfig {
        name: "prometheus".to_string(),
        enabled: true,
        export_interval: Duration::from_secs(60),
        timeout: Duration::from_secs(30),
        retry_count: 3,
        endpoint: Some("localhost:9090".to_string()),
    };

    let exporter = PrometheusExporter::new(config, "localhost:9090".to_string());
    assert_eq!(exporter.name(), "prometheus");
}

#[tokio::test]
async fn test_json_file_exporter() {
    let config = ExporterConfig {
        name: "json_file".to_string(),
        enabled: true,
        export_interval: Duration::from_secs(60),
        timeout: Duration::from_secs(30),
        retry_count: 3,
        endpoint: None,
    };

    let exporter = JsonFileExporter::new(config, "/tmp/test_metrics.json".to_string());
    assert_eq!(exporter.name(), "json_file");
}

#[tokio::test]
async fn test_console_exporter() {
    let config = ExporterConfig {
        name: "console".to_string(),
        enabled: true,
        export_interval: Duration::from_secs(60),
        timeout: Duration::from_secs(30),
        retry_count: 3,
        endpoint: None,
    };

    let exporter = ConsoleExporter::new(config, ConsoleFormat::Json);
    assert_eq!(exporter.name(), "console");
}

#[tokio::test]
async fn test_console_format_types() {
    let table = ConsoleFormat::Table;
    let json = ConsoleFormat::Json;
    let plain = ConsoleFormat::Plain;

    assert!(matches!(table, ConsoleFormat::Table));
    assert!(matches!(json, ConsoleFormat::Json));
    assert!(matches!(plain, ConsoleFormat::Plain));
}

#[tokio::test]
async fn test_export_stats_default() {
    let stats = ExportStats::default();

    assert_eq!(stats.total_exports, 0);
    assert_eq!(stats.successful_exports, 0);
    assert_eq!(stats.failed_exports, 0);
    assert_eq!(stats.avg_export_time, 0.0);
    assert!(stats.last_export.is_none());
    assert!(stats.last_error.is_none());
    assert_eq!(stats.total_metrics_exported, 0);
}

// Metrics Registry Tests

#[tokio::test]
async fn test_metrics_registry_creation() {
    let config = RegistryConfig::default();
    let registry = MetricsRegistry::new(config);

    let stats = registry.get_stats().await;
    assert_eq!(stats.total_metrics, 0);
    assert_eq!(stats.total_rules, 0);
    assert_eq!(stats.enabled_rules, 0);
}

#[tokio::test]
async fn test_metrics_registry_with_defaults() {
    let registry = MetricsRegistry::with_defaults();

    let stats = registry.get_stats().await;
    assert_eq!(stats.total_metrics, 0);
}

#[tokio::test]
async fn test_registry_config_default() {
    let config = RegistryConfig::default();

    assert_eq!(config.max_metrics, 10000);
    assert_eq!(config.retention_period, Duration::from_secs(24 * 60 * 60));
    assert!(config.auto_cleanup);
    assert_eq!(config.cleanup_interval, Duration::from_secs(60 * 60));
}

#[tokio::test]
async fn test_aggregation_function_types() {
    let sum = AggregationFunction::Sum;
    let avg = AggregationFunction::Average;
    let min = AggregationFunction::Min;
    let max = AggregationFunction::Max;
    let count = AggregationFunction::Count;
    let rate = AggregationFunction::Rate;
    let percentile = AggregationFunction::Percentile(95.0);

    assert_eq!(sum, AggregationFunction::Sum);
    assert_eq!(avg, AggregationFunction::Average);
    assert_eq!(min, AggregationFunction::Min);
    assert_eq!(max, AggregationFunction::Max);
    assert_eq!(count, AggregationFunction::Count);
    assert_eq!(rate, AggregationFunction::Rate);
    assert_eq!(percentile, AggregationFunction::Percentile(95.0));
}

// Metric Builder Tests

#[tokio::test]
async fn test_metric_builder_counter() {
    let metric = MetricBuilder::new("test_counter", MetricType::Counter)
        .help("Test counter metric")
        .label("service", "test")
        .counter(42);

    assert_eq!(metric.name, "test_counter");
    assert!(matches!(metric.value, MetricValue::Counter(42)));
    assert_eq!(metric.metric_type, MetricType::Counter);
    assert_eq!(metric.help, Some("Test counter metric".to_string()));
    assert_eq!(metric.labels.get("service"), Some(&"test".to_string()));
}

#[tokio::test]
async fn test_metric_builder_gauge() {
    let metric = MetricBuilder::new("test_gauge", MetricType::Gauge)
        .help("Test gauge metric")
        .gauge(75.5);

    assert_eq!(metric.name, "test_gauge");
    assert!(matches!(metric.value, MetricValue::Gauge(75.5)));
    assert_eq!(metric.metric_type, MetricType::Gauge);
}

#[tokio::test]
async fn test_histogram_builder() {
    let histogram_data = HistogramBuilder::new()
        .buckets(vec![0.1, 0.5, 1.0, 5.0])
        .values(vec![0.2, 0.8, 1.5, 3.0])
        .build();

    assert_eq!(histogram_data.buckets.len(), 4);
    assert_eq!(histogram_data.count, 4);
    assert_eq!(histogram_data.sum, 5.5);
}

#[tokio::test]
async fn test_summary_builder() {
    let summary_data = SummaryBuilder::new()
        .values(vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .quantiles(vec![0.5, 0.9, 0.95, 0.99])
        .build();

    assert_eq!(summary_data.quantiles.len(), 4);
    assert_eq!(summary_data.count, 5);
    assert_eq!(summary_data.sum, 15.0);
}

#[tokio::test]
async fn test_histogram_builder_default() {
    let builder = HistogramBuilder::default();
    let histogram_data = builder.build();

    assert!(histogram_data.buckets.is_empty());
    assert_eq!(histogram_data.count, 0);
    assert_eq!(histogram_data.sum, 0.0);
}

#[tokio::test]
async fn test_summary_builder_default() {
    let builder = SummaryBuilder::default();
    let summary_data = builder.build();

    assert_eq!(summary_data.quantiles.len(), 4); // Default quantiles
    assert_eq!(summary_data.count, 0);
    assert_eq!(summary_data.sum, 0.0);
}
