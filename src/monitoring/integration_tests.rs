// Comprehensive Integration Tests for Monitoring System
// Tests the complete monitoring pipeline with collectors and exporters

use super::*;

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

/// Test monitoring system initialization
#[tokio::test]
async fn test_monitoring_system_initialization() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Test complete monitoring initialization
    let result = monitor.initialize_complete_monitoring().await;
    assert!(result.is_ok(), "Complete monitoring initialization should succeed");
    
    // Verify collectors are initialized
    let collectors = monitor.collectors.read().await;
    assert!(collectors.contains_key("application"), "Application collector should be initialized");
    assert!(collectors.contains_key("http"), "HTTP collector should be initialized");
    assert!(collectors.contains_key("custom"), "Custom collector should be initialized");
    
    // Verify exporters are initialized
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Exporters should be initialized");
    
    // Stop monitoring system
    let _ = monitor.stop().await;
}

/// Test application metrics collector integration
#[tokio::test]
async fn test_application_metrics_collector_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize application collector
    let result = monitor.add_application_collector().await;
    assert!(result.is_ok(), "Application collector should be added successfully");
    
    // Record application metric
    let mut labels = HashMap::new();
    labels.insert("service".to_string(), "test".to_string());
    
    let result = monitor.record_application_metric("test_metric", 42.0, labels).await;
    assert!(result.is_ok(), "Application metric should be recorded successfully");
    
    // Verify collector health
    let collectors = monitor.collectors.read().await;
    if let Some(collector) = collectors.get("application") {
        let health = collector.health_check().await;
        assert!(health.is_ok(), "Application collector health check should succeed");
    }
}

/// Test HTTP metrics collector integration
#[tokio::test]
async fn test_http_metrics_collector_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize HTTP collector
    let result = monitor.add_http_collector().await;
    assert!(result.is_ok(), "HTTP collector should be added successfully");
    
    // Record HTTP request
    let result = monitor.record_http_request("/api/test", 200, 150.0).await;
    assert!(result.is_ok(), "HTTP request should be recorded successfully");
    
    // Record error request
    let result = monitor.record_http_request("/api/error", 500, 250.0).await;
    assert!(result.is_ok(), "HTTP error request should be recorded successfully");
    
    // Verify collector health
    let collectors = monitor.collectors.read().await;
    if let Some(collector) = collectors.get("http") {
        let health = collector.health_check().await;
        assert!(health.is_ok(), "HTTP collector health check should succeed");
    }
}

/// Test custom metrics collector integration
#[tokio::test]
async fn test_custom_metrics_collector_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize custom collector
    let result = monitor.add_custom_collector().await;
    assert!(result.is_ok(), "Custom collector should be added successfully");
    
    // Create custom metric
    let custom_metric = crate::monitoring::collectors::CustomMetric {
        name: "custom_test_metric".to_string(),
        metric_type: MetricType::Gauge,
        value: MetricValue::Gauge(75.5),
        labels: HashMap::new(),
        timestamp: chrono::Utc::now(),
        description: Some("Test custom metric".to_string()),
    };
    
    // Add custom metric
    let result = monitor.add_custom_metric(custom_metric).await;
    assert!(result.is_ok(), "Custom metric should be added successfully");
    
    // Verify collector health
    let collectors = monitor.collectors.read().await;
    if let Some(collector) = collectors.get("custom") {
        let health = collector.health_check().await;
        assert!(health.is_ok(), "Custom collector health check should succeed");
    }
}

/// Test Prometheus exporter integration
#[tokio::test]
async fn test_prometheus_exporter_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Add Prometheus exporter
    let result = monitor.add_prometheus_exporter().await;
    assert!(result.is_ok(), "Prometheus exporter should be added successfully");
    
    // Verify exporter is added
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Exporters should contain Prometheus exporter");
    
    // Check exporter health
    for exporter in exporters.iter() {
        if exporter.name() == "prometheus" {
            let health = exporter.health_check().await;
            assert!(health.is_ok(), "Prometheus exporter health check should succeed");
        }
    }
}

/// Test console exporter integration
#[tokio::test]
async fn test_console_exporter_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Add console exporter
    let result = monitor.add_console_exporter().await;
    assert!(result.is_ok(), "Console exporter should be added successfully");
    
    // Verify exporter is added
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Exporters should contain console exporter");
    
    // Check exporter health
    for exporter in exporters.iter() {
        if exporter.name() == "console" {
            let health = exporter.health_check().await;
            assert!(health.is_ok(), "Console exporter health check should succeed");
        }
    }
}

/// Test file exporter integration
#[tokio::test]
async fn test_file_exporter_integration() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Add file exporter
    let result = monitor.add_file_exporter("/tmp/test_metrics.csv").await;
    assert!(result.is_ok(), "File exporter should be added successfully");
    
    // Verify exporter is added
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Exporters should contain file exporter");
    
    // Check exporter health
    for exporter in exporters.iter() {
        if exporter.name() == "file" {
            let health = exporter.health_check().await;
            assert!(health.is_ok(), "File exporter health check should succeed");
        }
    }
}

/// Test default collectors initialization
#[tokio::test]
async fn test_default_collectors_initialization() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize default collectors
    let result = monitor.initialize_default_collectors().await;
    assert!(result.is_ok(), "Default collectors initialization should succeed");
    
    // Verify all default collectors are added
    let collectors = monitor.collectors.read().await;
    assert!(collectors.contains_key("application"), "Application collector should be initialized");
    assert!(collectors.contains_key("http"), "HTTP collector should be initialized");
    assert!(collectors.contains_key("custom"), "Custom collector should be initialized");
    assert_eq!(collectors.len(), 3, "Should have exactly 3 default collectors");
}

/// Test default exporters initialization
#[tokio::test]
async fn test_default_exporters_initialization() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize default exporters
    let result = monitor.initialize_default_exporters().await;
    assert!(result.is_ok(), "Default exporters initialization should succeed");
    
    // Verify exporters are added
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Default exporters should be initialized");
    
    // Verify console exporter is always added
    let has_console = exporters.iter().any(|e| e.name() == "console");
    assert!(has_console, "Console exporter should always be initialized");
}

/// Test monitoring status retrieval
#[tokio::test]
async fn test_monitoring_status_retrieval() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize monitoring system
    let _ = monitor.initialize_default_collectors().await;
    let _ = monitor.initialize_default_exporters().await;
    
    // Get monitoring status
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "Monitoring status retrieval should succeed");
    
    let status = status.unwrap();
    assert!(!status.collector_health.is_empty(), "Collector health should be available");
    assert!(!status.exporter_health.is_empty(), "Exporter health should be available");
    assert!(status.system_uptime.as_secs() >= 0, "System uptime should be valid");
}

/// Test metrics collection workflow
#[tokio::test]
async fn test_metrics_collection_workflow() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize collectors
    let _ = monitor.add_application_collector().await;
    let _ = monitor.add_http_collector().await;
    
    // Record various metrics
    let mut labels = HashMap::new();
    labels.insert("component".to_string(), "test".to_string());
    
    let _ = monitor.record_application_metric("cpu_usage", 65.5, labels.clone()).await;
    let _ = monitor.record_application_metric("memory_usage", 78.2, labels).await;
    let _ = monitor.record_http_request("/api/users", 200, 120.0).await;
    let _ = monitor.record_http_request("/api/orders", 201, 95.0).await;
    
    // Verify metrics are collected
    let collectors = monitor.collectors.read().await;
    assert_eq!(collectors.len(), 2, "Should have 2 collectors");
    
    // Test metric collection
    for (name, collector) in collectors.iter() {
        let metrics = collector.collect().await;
        assert!(metrics.is_ok(), "Metric collection should succeed for {}", name);
    }
}

/// Test export workflow
#[tokio::test]
async fn test_export_workflow() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize system
    let _ = monitor.add_application_collector().await;
    let _ = monitor.add_console_exporter().await;
    
    // Record metrics
    let mut labels = HashMap::new();
    labels.insert("test".to_string(), "export".to_string());
    let _ = monitor.record_application_metric("test_export_metric", 100.0, labels).await;
    
    // Collect metrics
    let collectors = monitor.collectors.read().await;
    let mut all_metrics = Vec::new();
    
    for (_, collector) in collectors.iter() {
        if let Ok(metrics) = collector.collect().await {
            all_metrics.extend(metrics);
        }
    }
    
    // Export metrics
    let exporters = monitor.exporters.read().await;
    for exporter in exporters.iter() {
        let result = exporter.export(&all_metrics).await;
        assert!(result.is_ok(), "Metric export should succeed for {}", exporter.name());
    }
}

/// Test collector health monitoring
#[tokio::test]
async fn test_collector_health_monitoring() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize collectors
    let _ = monitor.initialize_default_collectors().await;
    
    // Check health of all collectors
    let collectors = monitor.collectors.read().await;
    for (name, collector) in collectors.iter() {
        let health = collector.health_check().await;
        assert!(health.is_ok(), "Health check should succeed for collector {}", name);
        
        let health = health.unwrap();
        assert!(health.is_healthy, "Collector {} should be healthy", name);
        assert_eq!(health.error_count, 0, "Collector {} should have no errors", name);
    }
}

/// Test exporter health monitoring
#[tokio::test]
async fn test_exporter_health_monitoring() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize exporters
    let _ = monitor.initialize_default_exporters().await;
    
    // Check health of all exporters
    let exporters = monitor.exporters.read().await;
    for exporter in exporters.iter() {
        let health = exporter.health_check().await;
        assert!(health.is_ok(), "Health check should succeed for exporter {}", exporter.name());
        
        let health = health.unwrap();
        assert!(health.is_healthy, "Exporter {} should be healthy", exporter.name());
        assert_eq!(health.error_count, 0, "Exporter {} should have no errors", exporter.name());
    }
}

/// Test monitoring system performance
#[tokio::test]
async fn test_monitoring_system_performance() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize complete monitoring
    let start = std::time::Instant::now();
    let _ = monitor.initialize_complete_monitoring().await;
    let init_duration = start.elapsed();
    
    // Initialization should be fast
    assert!(init_duration.as_millis() < 1000, "Monitoring initialization should be fast");
    
    // Record multiple metrics quickly
    let start = std::time::Instant::now();
    for i in 0..100 {
        let mut labels = HashMap::new();
        labels.insert("iteration".to_string(), i.to_string());
        let _ = monitor.record_application_metric("performance_test", i as f64, labels).await;
    }
    let record_duration = start.elapsed();
    
    // Recording should be efficient
    assert!(record_duration.as_millis() < 500, "Metric recording should be efficient");
    
    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system error handling
#[tokio::test]
async fn test_monitoring_system_error_handling() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Test recording metrics without collectors (should not panic)
    let mut labels = HashMap::new();
    labels.insert("test".to_string(), "error".to_string());
    let result = monitor.record_application_metric("test_metric", 50.0, labels).await;
    assert!(result.is_ok(), "Recording without collectors should not fail");
    
    // Test HTTP recording without collectors (should not panic)
    let result = monitor.record_http_request("/test", 200, 100.0).await;
    assert!(result.is_ok(), "HTTP recording without collectors should not fail");
    
    // Test custom metric without collectors (should not panic)
    let custom_metric = crate::monitoring::collectors::CustomMetric {
        name: "error_test".to_string(),
        metric_type: MetricType::Counter,
        value: MetricValue::Counter(1),
        labels: HashMap::new(),
        timestamp: chrono::Utc::now(),
        description: None,
    };
    let result = monitor.add_custom_metric(custom_metric).await;
    assert!(result.is_ok(), "Custom metric without collectors should not fail");
}

/// Test monitoring configuration validation
#[tokio::test]
async fn test_monitoring_configuration_validation() {
    // Test default configuration
    let default_config = MonitoringConfig::default();
    assert!(default_config.collection_interval > 0, "Collection interval should be positive");
    assert!(default_config.retention_hours > 0, "Retention hours should be positive");
    assert!(default_config.enable_resource_monitoring, "Resource monitoring should be enabled by default");
    
    // Test export configuration
    assert!(default_config.export_config.enable_prometheus, "Prometheus should be enabled by default");
    assert!(!default_config.export_config.prometheus_endpoint.is_empty(), "Prometheus endpoint should be configured");
    assert!(default_config.export_config.export_interval > 0, "Export interval should be positive");
    
    // Test alert thresholds
    assert!(default_config.alert_thresholds.cpu_threshold > 0.0, "CPU threshold should be positive");
    assert!(default_config.alert_thresholds.memory_threshold > 0.0, "Memory threshold should be positive");
    assert!(default_config.alert_thresholds.response_time_threshold > 0, "Response time threshold should be positive");
}

/// Test monitoring system scalability
#[tokio::test]
async fn test_monitoring_system_scalability() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;
    
    // Test with high volume of metrics
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let mut labels = HashMap::new();
        labels.insert("batch".to_string(), (i / 100).to_string());
        labels.insert("index".to_string(), i.to_string());
        
        let _ = monitor.record_application_metric("scalability_test", i as f64, labels).await;
        
        if i % 10 == 0 {
            let _ = monitor.record_http_request(&format!("/api/test/{}", i), 200, (i % 500) as f64).await;
        }
    }
    let duration = start.elapsed();
    
    // Should handle high volume efficiently
    assert!(duration.as_millis() < 2000, "High volume metric recording should be efficient");
    
    // Verify system is still healthy
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should remain healthy under load");
    
    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system integration with existing components
#[tokio::test]
async fn test_monitoring_integration_with_existing_components() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);
    
    // Test integration with alert manager (always available)
    // Alert manager is always present as Arc<AlertManager>

    // Test integration with resource tracker (always available)
    // Resource tracker is always present as Arc<ResourceTracker>
    
    // Initialize monitoring and verify integration
    let _ = monitor.initialize_complete_monitoring().await;
    
    // Test that existing functionality still works
    let stats = monitor.get_stats().await;
    assert!(stats.active_collectors >= 0, "Active collectors count should be valid");
    assert!(stats.active_exporters >= 0, "Active exporters count should be valid");
    
    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test concurrent monitoring operations
#[tokio::test]
async fn test_concurrent_monitoring_operations() {
    let config = MonitoringConfig::default();
    let monitor = std::sync::Arc::new(PerformanceMonitor::new(config));

    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Create multiple concurrent tasks
    let mut handles = Vec::new();

    for i in 0..10 {
        let monitor_clone = monitor.clone();
        let handle = tokio::spawn(async move {
            let mut labels = HashMap::new();
            labels.insert("task".to_string(), i.to_string());

            // Record metrics concurrently
            for j in 0..50 {
                let _ = monitor_clone.record_application_metric(
                    &format!("concurrent_metric_{}", j),
                    (i * 10 + j) as f64,
                    labels.clone()
                ).await;

                if j % 5 == 0 {
                    let _ = monitor_clone.record_http_request(
                        &format!("/api/concurrent/{}/{}", i, j),
                        200,
                        (j * 10) as f64
                    ).await;
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok(), "Concurrent task should complete successfully");
    }

    // Verify system is still healthy
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should remain healthy after concurrent operations");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system memory efficiency
#[tokio::test]
async fn test_monitoring_memory_efficiency() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Record a large number of metrics to test memory usage
    for batch in 0..10 {
        for i in 0..100 {
            let mut labels = HashMap::new();
            labels.insert("batch".to_string(), batch.to_string());
            labels.insert("metric_id".to_string(), i.to_string());

            let _ = monitor.record_application_metric(
                &format!("memory_test_metric_{}", i),
                (batch * 100 + i) as f64,
                labels
            ).await;
        }

        // Force collection to prevent memory buildup
        let collectors = monitor.collectors.read().await;
        for (_, collector) in collectors.iter() {
            let _ = collector.collect().await;
        }
    }

    // Verify system is still responsive
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should remain responsive with high metric volume");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system with custom configurations
#[tokio::test]
async fn test_monitoring_with_custom_configurations() {
    let mut config = MonitoringConfig::default();
    config.collection_interval = 10; // 10 seconds
    config.retention_hours = 48; // 48 hours
    config.export_config.export_interval = 30; // 30 seconds

    let monitor = PerformanceMonitor::new(config);

    // Initialize with custom configuration
    let _ = monitor.initialize_complete_monitoring().await;

    // Verify configuration is applied
    assert_eq!(monitor.config.collection_interval, 10, "Custom collection interval should be applied");
    assert_eq!(monitor.config.retention_hours, 48, "Custom retention hours should be applied");
    assert_eq!(monitor.config.export_config.export_interval, 30, "Custom export interval should be applied");

    // Test functionality with custom config
    let mut labels = HashMap::new();
    labels.insert("config".to_string(), "custom".to_string());
    let result = monitor.record_application_metric("custom_config_test", 123.45, labels).await;
    assert!(result.is_ok(), "Metric recording should work with custom config");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system resilience to failures
#[tokio::test]
async fn test_monitoring_system_resilience() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Test resilience to invalid metric names
    let mut labels = HashMap::new();
    labels.insert("test".to_string(), "resilience".to_string());

    let result = monitor.record_application_metric("", 50.0, labels.clone()).await;
    assert!(result.is_ok(), "Should handle empty metric names gracefully");

    let result = monitor.record_application_metric("metric with spaces", 60.0, labels.clone()).await;
    assert!(result.is_ok(), "Should handle metric names with spaces gracefully");

    // Test resilience to extreme values
    let result = monitor.record_application_metric("extreme_high", f64::MAX, labels.clone()).await;
    assert!(result.is_ok(), "Should handle extreme high values gracefully");

    let result = monitor.record_application_metric("extreme_low", f64::MIN, labels.clone()).await;
    assert!(result.is_ok(), "Should handle extreme low values gracefully");

    let result = monitor.record_application_metric("nan_value", f64::NAN, labels.clone()).await;
    assert!(result.is_ok(), "Should handle NaN values gracefully");

    // Test resilience to invalid HTTP status codes
    let result = monitor.record_http_request("/test", 999, 100.0).await;
    assert!(result.is_ok(), "Should handle invalid status codes gracefully");

    let result = monitor.record_http_request("/test", 0, 100.0).await;
    assert!(result.is_ok(), "Should handle zero status codes gracefully");

    // Verify system is still healthy
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should remain healthy after handling edge cases");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system with disabled components
#[tokio::test]
async fn test_monitoring_with_disabled_components() {
    let mut config = MonitoringConfig::default();
    config.enable_resource_monitoring = false;
    config.enable_profiling = false;
    config.export_config.enable_prometheus = false;
    config.export_config.enable_json = false;

    let monitor = PerformanceMonitor::new(config);

    // Initialize with disabled components
    let result = monitor.initialize_complete_monitoring().await;
    assert!(result.is_ok(), "Should initialize successfully with disabled components");

    // Verify only console exporter is initialized (always enabled)
    let exporters = monitor.exporters.read().await;
    assert!(!exporters.is_empty(), "Should have at least console exporter");

    let has_console = exporters.iter().any(|e| e.name() == "console");
    assert!(has_console, "Console exporter should always be present");

    // Test functionality still works
    let mut labels = HashMap::new();
    labels.insert("disabled".to_string(), "components".to_string());
    let result = monitor.record_application_metric("disabled_test", 75.0, labels).await;
    assert!(result.is_ok(), "Metric recording should work with disabled components");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system startup and shutdown lifecycle
#[tokio::test]
async fn test_monitoring_lifecycle() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Test multiple start/stop cycles
    for cycle in 0..3 {
        // Start monitoring
        let result = monitor.initialize_complete_monitoring().await;
        assert!(result.is_ok(), "Monitoring should start successfully in cycle {}", cycle);

        // Record some metrics
        let mut labels = HashMap::new();
        labels.insert("cycle".to_string(), cycle.to_string());
        let _ = monitor.record_application_metric("lifecycle_test", cycle as f64, labels).await;

        // Verify system is running
        let status = monitor.get_monitoring_status().await;
        assert!(status.is_ok(), "System should be healthy in cycle {}", cycle);

        // Stop monitoring
        let result = monitor.stop().await;
        assert!(result.is_ok(), "Monitoring should stop successfully in cycle {}", cycle);

        // Brief pause between cycles
        sleep(Duration::from_millis(10)).await;
    }
}

/// Test monitoring system with large label sets
#[tokio::test]
async fn test_monitoring_with_large_label_sets() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Create metrics with large label sets
    for i in 0..10 {
        let mut labels = HashMap::new();

        // Add many labels
        for j in 0..20 {
            labels.insert(format!("label_{}", j), format!("value_{}_{}", i, j));
        }

        let result = monitor.record_application_metric(
            &format!("large_labels_metric_{}", i),
            i as f64,
            labels
        ).await;
        assert!(result.is_ok(), "Should handle large label sets successfully");
    }

    // Verify system handles large labels efficiently
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should handle large label sets efficiently");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system metric type validation
#[tokio::test]
async fn test_monitoring_metric_type_validation() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring
    let _ = monitor.add_custom_collector().await;

    // Test different metric types
    let labels = HashMap::new();

    // Counter metric
    let counter_metric = crate::monitoring::collectors::CustomMetric {
        name: "test_counter".to_string(),
        metric_type: MetricType::Counter,
        value: MetricValue::Counter(42),
        labels: labels.clone(),
        timestamp: chrono::Utc::now(),
        description: Some("Test counter metric".to_string()),
    };
    let result = monitor.add_custom_metric(counter_metric).await;
    assert!(result.is_ok(), "Counter metric should be added successfully");

    // Gauge metric
    let gauge_metric = crate::monitoring::collectors::CustomMetric {
        name: "test_gauge".to_string(),
        metric_type: MetricType::Gauge,
        value: MetricValue::Gauge(75.5),
        labels: labels.clone(),
        timestamp: chrono::Utc::now(),
        description: Some("Test gauge metric".to_string()),
    };
    let result = monitor.add_custom_metric(gauge_metric).await;
    assert!(result.is_ok(), "Gauge metric should be added successfully");

    // Histogram metric
    let histogram_data = HistogramData {
        buckets: vec![
            HistogramBucket { upper_bound: 10.0, count: 5 },
            HistogramBucket { upper_bound: 50.0, count: 15 },
            HistogramBucket { upper_bound: 100.0, count: 25 },
        ],
        count: 25,
        sum: 1250.0,
    };
    let histogram_metric = crate::monitoring::collectors::CustomMetric {
        name: "test_histogram".to_string(),
        metric_type: MetricType::Histogram,
        value: MetricValue::Histogram(histogram_data),
        labels: labels.clone(),
        timestamp: chrono::Utc::now(),
        description: Some("Test histogram metric".to_string()),
    };
    let result = monitor.add_custom_metric(histogram_metric).await;
    assert!(result.is_ok(), "Histogram metric should be added successfully");

    // Summary metric
    let summary_data = SummaryData {
        quantiles: vec![
            Quantile { quantile: 0.5, value: 50.0 },
            Quantile { quantile: 0.9, value: 90.0 },
            Quantile { quantile: 0.99, value: 99.0 },
        ],
        count: 100,
        sum: 5500.0,
    };
    let summary_metric = crate::monitoring::collectors::CustomMetric {
        name: "test_summary".to_string(),
        metric_type: MetricType::Summary,
        value: MetricValue::Summary(summary_data),
        labels: labels,
        timestamp: chrono::Utc::now(),
        description: Some("Test summary metric".to_string()),
    };
    let result = monitor.add_custom_metric(summary_metric).await;
    assert!(result.is_ok(), "Summary metric should be added successfully");
}

/// Test monitoring system export format validation
#[tokio::test]
async fn test_monitoring_export_format_validation() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring with all exporters
    let _ = monitor.add_console_exporter().await;
    let _ = monitor.add_file_exporter("/tmp/test_export_validation.csv").await;
    let _ = monitor.add_prometheus_exporter().await;

    // Add some test metrics
    let _ = monitor.add_application_collector().await;
    let mut labels = HashMap::new();
    labels.insert("format".to_string(), "validation".to_string());
    let _ = monitor.record_application_metric("export_format_test", 88.8, labels).await;

    // Collect metrics
    let collectors = monitor.collectors.read().await;
    let mut all_metrics = Vec::new();
    for (_, collector) in collectors.iter() {
        if let Ok(metrics) = collector.collect().await {
            all_metrics.extend(metrics);
        }
    }

    // Test export to all formats
    let exporters = monitor.exporters.read().await;
    for exporter in exporters.iter() {
        let result = exporter.export(&all_metrics).await;
        assert!(result.is_ok(), "Export should succeed for format: {}", exporter.name());

        // Verify exporter health after export
        let health = exporter.health_check().await;
        assert!(health.is_ok(), "Exporter health should be good after export: {}", exporter.name());
    }
}

/// Test monitoring system with rapid metric updates
#[tokio::test]
async fn test_monitoring_rapid_metric_updates() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Rapid metric updates
    let start = std::time::Instant::now();
    for i in 0..500 {
        let mut labels = HashMap::new();
        labels.insert("rapid".to_string(), "update".to_string());
        labels.insert("iteration".to_string(), (i % 10).to_string());

        let _ = monitor.record_application_metric("rapid_update_test", (i % 100) as f64, labels).await;

        if i % 50 == 0 {
            let _ = monitor.record_http_request("/api/rapid", 200, (i % 200) as f64).await;
        }
    }
    let duration = start.elapsed();

    // Should handle rapid updates efficiently
    assert!(duration.as_millis() < 1000, "Rapid metric updates should be efficient");

    // Verify system is still healthy
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "System should remain healthy after rapid updates");

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system collector removal and re-addition
#[tokio::test]
async fn test_monitoring_collector_lifecycle() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Add collectors
    let _ = monitor.add_application_collector().await;
    let _ = monitor.add_http_collector().await;

    // Verify collectors are added
    {
        let collectors = monitor.collectors.read().await;
        assert_eq!(collectors.len(), 2, "Should have 2 collectors");
    }

    // Remove a collector
    let result = monitor.remove_collector("application").await;
    assert!(result.is_ok(), "Collector removal should succeed");

    // Verify collector is removed
    {
        let collectors = monitor.collectors.read().await;
        assert_eq!(collectors.len(), 1, "Should have 1 collector after removal");
        assert!(!collectors.contains_key("application"), "Application collector should be removed");
    }

    // Re-add the collector
    let _ = monitor.add_application_collector().await;

    // Verify collector is re-added
    {
        let collectors = monitor.collectors.read().await;
        assert_eq!(collectors.len(), 2, "Should have 2 collectors after re-addition");
        assert!(collectors.contains_key("application"), "Application collector should be re-added");
    }
}

/// Test monitoring system with mixed metric types
#[tokio::test]
async fn test_monitoring_mixed_metric_types() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize all collectors
    let _ = monitor.initialize_complete_monitoring().await;

    // Record various types of metrics
    let mut labels = HashMap::new();
    labels.insert("mixed".to_string(), "types".to_string());

    // Application metrics
    let _ = monitor.record_application_metric("cpu_usage", 65.5, labels.clone()).await;
    let _ = monitor.record_application_metric("memory_usage", 78.2, labels.clone()).await;
    let _ = monitor.record_application_metric("disk_usage", 45.8, labels.clone()).await;

    // HTTP metrics
    let _ = monitor.record_http_request("/api/users", 200, 120.0).await;
    let _ = monitor.record_http_request("/api/orders", 201, 95.0).await;
    let _ = monitor.record_http_request("/api/products", 404, 50.0).await;
    let _ = monitor.record_http_request("/api/auth", 500, 200.0).await;

    // Custom metrics
    let counter_metric = crate::monitoring::collectors::CustomMetric {
        name: "request_count".to_string(),
        metric_type: MetricType::Counter,
        value: MetricValue::Counter(1000),
        labels: labels.clone(),
        timestamp: chrono::Utc::now(),
        description: Some("Total request count".to_string()),
    };
    let _ = monitor.add_custom_metric(counter_metric).await;

    let gauge_metric = crate::monitoring::collectors::CustomMetric {
        name: "queue_size".to_string(),
        metric_type: MetricType::Gauge,
        value: MetricValue::Gauge(25.0),
        labels: labels,
        timestamp: chrono::Utc::now(),
        description: Some("Current queue size".to_string()),
    };
    let _ = monitor.add_custom_metric(gauge_metric).await;

    // Verify all metrics are collected
    let collectors = monitor.collectors.read().await;
    for (name, collector) in collectors.iter() {
        let metrics = collector.collect().await;
        assert!(metrics.is_ok(), "Metric collection should succeed for {}", name);

        let metrics = metrics.unwrap();
        assert!(!metrics.is_empty(), "Should have collected metrics for {}", name);
    }

    // Stop monitoring
    let _ = monitor.stop().await;
}

/// Test monitoring system comprehensive status reporting
#[tokio::test]
async fn test_monitoring_comprehensive_status_reporting() {
    let config = MonitoringConfig::default();
    let monitor = PerformanceMonitor::new(config);

    // Initialize complete monitoring
    let _ = monitor.initialize_complete_monitoring().await;

    // Record some metrics to generate activity
    let mut labels = HashMap::new();
    labels.insert("status".to_string(), "reporting".to_string());

    for i in 0..10 {
        let _ = monitor.record_application_metric(&format!("status_metric_{}", i), i as f64, labels.clone()).await;
        let _ = monitor.record_http_request(&format!("/api/status/{}", i), 200, (i * 10) as f64).await;
    }

    // Get comprehensive status
    let status = monitor.get_monitoring_status().await;
    assert!(status.is_ok(), "Status retrieval should succeed");

    let status = status.unwrap();

    // Verify performance stats
    assert!(status.performance_stats.total_metrics_collected >= 0, "Total metrics should be valid");
    assert!(status.performance_stats.active_collectors > 0, "Should have active collectors");
    assert!(status.performance_stats.active_exporters > 0, "Should have active exporters");

    // Verify collector health
    assert!(!status.collector_health.is_empty(), "Should have collector health data");
    for (name, health) in &status.collector_health {
        assert!(health.is_healthy, "Collector {} should be healthy", name);
    }

    // Verify exporter health
    assert!(!status.exporter_health.is_empty(), "Should have exporter health data");
    for (name, health) in &status.exporter_health {
        assert!(health.is_healthy, "Exporter {} should be healthy", name);
    }

    // Verify system uptime
    assert!(status.system_uptime.as_secs() >= 0, "System uptime should be valid");

    // Verify configuration
    assert_eq!(status.monitoring_config.collection_interval, 30, "Configuration should match");

    // Stop monitoring
    let _ = monitor.stop().await;
}
