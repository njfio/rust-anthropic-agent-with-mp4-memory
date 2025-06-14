// Resource Tracking System for Performance Monitoring
// Provides comprehensive system resource monitoring and tracking

use super::{Metric, MetricType, MetricValue};
use crate::utils::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{CpuExt, DiskExt, NetworkExt, System, SystemExt};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// Resource tracker for system monitoring
pub struct ResourceTracker {
    /// System information collector
    system: Arc<RwLock<System>>,
    /// Resource tracking configuration
    config: ResourceConfig,
    /// Resource statistics
    stats: Arc<RwLock<ResourceStats>>,
    /// Process tracking
    process_tracker: Arc<RwLock<ProcessTracker>>,
    /// Network tracking
    network_tracker: Arc<RwLock<NetworkTracker>>,
    /// Disk tracking
    disk_tracker: Arc<RwLock<DiskTracker>>,
    /// Start time
    start_time: Instant,
}

/// Resource tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Enable CPU monitoring
    pub enable_cpu_monitoring: bool,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    /// Enable disk monitoring
    pub enable_disk_monitoring: bool,
    /// Enable network monitoring
    pub enable_network_monitoring: bool,
    /// Enable process monitoring
    pub enable_process_monitoring: bool,
    /// Monitoring interval in seconds
    pub monitoring_interval: u64,
    /// Process monitoring whitelist (empty = all processes)
    pub process_whitelist: Vec<String>,
    /// Disk monitoring paths
    pub disk_paths: Vec<String>,
    /// Network interface whitelist (empty = all interfaces)
    pub network_interfaces: Vec<String>,
}

/// Resource statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage percentage
    pub memory_usage: f64,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Used memory in bytes
    pub used_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Disk usage statistics
    pub disk_usage: HashMap<String, DiskUsage>,
    /// Network statistics
    pub network_stats: HashMap<String, NetworkStats>,
    /// Process statistics
    pub process_stats: HashMap<String, ProcessStats>,
    /// System load averages
    pub load_averages: LoadAverages,
    /// System uptime in seconds
    pub system_uptime: u64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Disk usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    /// Total space in bytes
    pub total_space: u64,
    /// Used space in bytes
    pub used_space: u64,
    /// Available space in bytes
    pub available_space: u64,
    /// Usage percentage
    pub usage_percentage: f64,
    /// Mount point
    pub mount_point: String,
    /// File system type
    pub file_system: String,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes transmitted
    pub bytes_transmitted: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets transmitted
    pub packets_transmitted: u64,
    /// Errors received
    pub errors_received: u64,
    /// Errors transmitted
    pub errors_transmitted: u64,
    /// Interface name
    pub interface_name: String,
}

/// Process statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStats {
    /// Process ID
    pub pid: u32,
    /// Process name
    pub name: String,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Virtual memory usage in bytes
    pub virtual_memory: u64,
    /// Process status
    pub status: String,
    /// Start time
    pub start_time: u64,
    /// Command line
    pub cmd: Vec<String>,
}

/// System load averages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAverages {
    /// 1-minute load average
    pub one_minute: f64,
    /// 5-minute load average
    pub five_minute: f64,
    /// 15-minute load average
    pub fifteen_minute: f64,
}

/// Process tracker
#[derive(Debug, Clone)]
pub struct ProcessTracker {
    /// Tracked processes
    processes: HashMap<u32, ProcessStats>,
    /// Process whitelist
    whitelist: Vec<String>,
}

/// Network tracker
#[derive(Debug, Clone)]
pub struct NetworkTracker {
    /// Network interfaces
    interfaces: HashMap<String, NetworkStats>,
    /// Interface whitelist
    whitelist: Vec<String>,
}

/// Disk tracker
#[derive(Debug, Clone)]
pub struct DiskTracker {
    /// Disk usage information
    disks: HashMap<String, DiskUsage>,
    /// Monitored paths
    paths: Vec<String>,
}

/// Resource health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealth {
    pub is_healthy: bool,
    pub cpu_healthy: bool,
    pub memory_healthy: bool,
    pub disk_healthy: bool,
    pub network_healthy: bool,
    pub last_update: Option<DateTime<Utc>>,
    pub error_count: u32,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            enable_cpu_monitoring: true,
            enable_memory_monitoring: true,
            enable_disk_monitoring: true,
            enable_network_monitoring: true,
            enable_process_monitoring: false, // Disabled by default for performance
            monitoring_interval: 30,
            process_whitelist: Vec::new(),
            disk_paths: vec!["/".to_string()],
            network_interfaces: Vec::new(),
        }
    }
}

impl Default for ResourceStats {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            total_memory: 0,
            used_memory: 0,
            available_memory: 0,
            disk_usage: HashMap::new(),
            network_stats: HashMap::new(),
            process_stats: HashMap::new(),
            load_averages: LoadAverages {
                one_minute: 0.0,
                five_minute: 0.0,
                fifteen_minute: 0.0,
            },
            system_uptime: 0,
            last_update: Utc::now(),
        }
    }
}

impl ResourceTracker {
    /// Create a new resource tracker
    pub fn new() -> Self {
        Self::with_config(ResourceConfig::default())
    }

    /// Create a resource tracker with custom configuration
    pub fn with_config(config: ResourceConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            system: Arc::new(RwLock::new(system)),
            config,
            stats: Arc::new(RwLock::new(ResourceStats::default())),
            process_tracker: Arc::new(RwLock::new(ProcessTracker {
                processes: HashMap::new(),
                whitelist: Vec::new(),
            })),
            network_tracker: Arc::new(RwLock::new(NetworkTracker {
                interfaces: HashMap::new(),
                whitelist: Vec::new(),
            })),
            disk_tracker: Arc::new(RwLock::new(DiskTracker {
                disks: HashMap::new(),
                paths: vec!["/".to_string()],
            })),
            start_time: Instant::now(),
        }
    }

    /// Start resource tracking
    pub async fn start(&self) -> Result<()> {
        info!("Starting resource tracker");

        // Initialize trackers
        self.initialize_trackers().await?;

        // Start monitoring loop
        self.start_monitoring_loop().await?;

        info!("Resource tracker started successfully");
        Ok(())
    }

    /// Stop resource tracking
    pub async fn stop(&self) -> Result<()> {
        info!("Resource tracker stopped");
        Ok(())
    }

    /// Get current resource statistics
    pub async fn get_stats(&self) -> ResourceStats {
        self.stats.read().await.clone()
    }

    /// Get configuration (for testing)
    pub fn get_config(&self) -> &ResourceConfig {
        &self.config
    }

    /// Get resource health status
    pub async fn get_health(&self) -> Result<ResourceHealth> {
        let stats = self.stats.read().await;
        
        let cpu_healthy = stats.cpu_usage < 90.0;
        let memory_healthy = stats.memory_usage < 90.0;
        let disk_healthy = stats.disk_usage.values().all(|d| d.usage_percentage < 95.0);
        let network_healthy = true; // TODO: Add network health checks

        Ok(ResourceHealth {
            is_healthy: cpu_healthy && memory_healthy && disk_healthy && network_healthy,
            cpu_healthy,
            memory_healthy,
            disk_healthy,
            network_healthy,
            last_update: Some(stats.last_update),
            error_count: 0, // TODO: Track errors
        })
    }

    /// Collect resource metrics
    pub async fn collect_metrics(&self) -> Result<Vec<Metric>> {
        let mut metrics = Vec::new();
        let stats = self.get_stats().await;

        // CPU metrics
        if self.config.enable_cpu_monitoring {
            metrics.push(Metric {
                name: "cpu_usage".to_string(),
                value: MetricValue::Gauge(stats.cpu_usage),
                labels: HashMap::new(),
                timestamp: Utc::now(),
                metric_type: MetricType::Gauge,
                help: Some("CPU usage percentage".to_string()),
            });
        }

        // Memory metrics
        if self.config.enable_memory_monitoring {
            metrics.push(Metric {
                name: "memory_usage".to_string(),
                value: MetricValue::Gauge(stats.memory_usage),
                labels: HashMap::new(),
                timestamp: Utc::now(),
                metric_type: MetricType::Gauge,
                help: Some("Memory usage percentage".to_string()),
            });

            metrics.push(Metric {
                name: "memory_total".to_string(),
                value: MetricValue::Gauge(stats.total_memory as f64),
                labels: HashMap::new(),
                timestamp: Utc::now(),
                metric_type: MetricType::Gauge,
                help: Some("Total memory in bytes".to_string()),
            });

            metrics.push(Metric {
                name: "memory_used".to_string(),
                value: MetricValue::Gauge(stats.used_memory as f64),
                labels: HashMap::new(),
                timestamp: Utc::now(),
                metric_type: MetricType::Gauge,
                help: Some("Used memory in bytes".to_string()),
            });
        }

        // Disk metrics
        if self.config.enable_disk_monitoring {
            for (path, disk) in &stats.disk_usage {
                let mut labels = HashMap::new();
                labels.insert("mount_point".to_string(), path.clone());
                labels.insert("filesystem".to_string(), disk.file_system.clone());

                metrics.push(Metric {
                    name: "disk_usage".to_string(),
                    value: MetricValue::Gauge(disk.usage_percentage),
                    labels: labels.clone(),
                    timestamp: Utc::now(),
                    metric_type: MetricType::Gauge,
                    help: Some("Disk usage percentage".to_string()),
                });

                metrics.push(Metric {
                    name: "disk_total".to_string(),
                    value: MetricValue::Gauge(disk.total_space as f64),
                    labels: labels.clone(),
                    timestamp: Utc::now(),
                    metric_type: MetricType::Gauge,
                    help: Some("Total disk space in bytes".to_string()),
                });
            }
        }

        // Network metrics
        if self.config.enable_network_monitoring {
            for (interface, net_stats) in &stats.network_stats {
                let mut labels = HashMap::new();
                labels.insert("interface".to_string(), interface.clone());

                metrics.push(Metric {
                    name: "network_bytes_received".to_string(),
                    value: MetricValue::Counter(net_stats.bytes_received),
                    labels: labels.clone(),
                    timestamp: Utc::now(),
                    metric_type: MetricType::Counter,
                    help: Some("Network bytes received".to_string()),
                });

                metrics.push(Metric {
                    name: "network_bytes_transmitted".to_string(),
                    value: MetricValue::Counter(net_stats.bytes_transmitted),
                    labels: labels.clone(),
                    timestamp: Utc::now(),
                    metric_type: MetricType::Counter,
                    help: Some("Network bytes transmitted".to_string()),
                });
            }
        }

        // System uptime
        metrics.push(Metric {
            name: "system_uptime".to_string(),
            value: MetricValue::Gauge(stats.system_uptime as f64),
            labels: HashMap::new(),
            timestamp: Utc::now(),
            metric_type: MetricType::Gauge,
            help: Some("System uptime in seconds".to_string()),
        });

        debug!("Collected {} resource metrics", metrics.len());
        Ok(metrics)
    }

    /// Update resource statistics
    pub async fn update_stats(&self) -> Result<()> {
        let mut system = self.system.write().await;
        system.refresh_all();

        let mut stats = self.stats.write().await;

        // Update CPU usage
        if self.config.enable_cpu_monitoring {
            stats.cpu_usage = system.global_cpu_info().cpu_usage() as f64;
        }

        // Update memory usage
        if self.config.enable_memory_monitoring {
            stats.total_memory = system.total_memory();
            stats.used_memory = system.used_memory();
            stats.available_memory = system.available_memory();
            stats.memory_usage = if stats.total_memory > 0 {
                (stats.used_memory as f64 / stats.total_memory as f64) * 100.0
            } else {
                0.0
            };
        }

        // Update disk usage
        if self.config.enable_disk_monitoring {
            stats.disk_usage.clear();
            for disk in system.disks() {
                let mount_point = disk.mount_point().to_string_lossy().to_string();
                let total_space = disk.total_space();
                let available_space = disk.available_space();
                let used_space = total_space - available_space;
                let usage_percentage = if total_space > 0 {
                    (used_space as f64 / total_space as f64) * 100.0
                } else {
                    0.0
                };

                stats.disk_usage.insert(mount_point.clone(), DiskUsage {
                    total_space,
                    used_space,
                    available_space,
                    usage_percentage,
                    mount_point,
                    file_system: String::from_utf8_lossy(disk.file_system()).to_string(),
                });
            }
        }

        // Update network statistics
        if self.config.enable_network_monitoring {
            stats.network_stats.clear();
            for (interface_name, data) in system.networks() {
                stats.network_stats.insert(interface_name.clone(), NetworkStats {
                    bytes_received: data.received(),
                    bytes_transmitted: data.transmitted(),
                    packets_received: data.packets_received(),
                    packets_transmitted: data.packets_transmitted(),
                    errors_received: data.errors_on_received(),
                    errors_transmitted: data.errors_on_transmitted(),
                    interface_name: interface_name.clone(),
                });
            }
        }

        // Update system uptime
        stats.system_uptime = system.uptime();
        stats.last_update = Utc::now();

        debug!("Updated resource statistics");
        Ok(())
    }

    /// Initialize trackers
    async fn initialize_trackers(&self) -> Result<()> {
        // Initialize process tracker
        if self.config.enable_process_monitoring {
            let mut process_tracker = self.process_tracker.write().await;
            process_tracker.whitelist = self.config.process_whitelist.clone();
        }

        // Initialize network tracker
        if self.config.enable_network_monitoring {
            let mut network_tracker = self.network_tracker.write().await;
            network_tracker.whitelist = self.config.network_interfaces.clone();
        }

        // Initialize disk tracker
        if self.config.enable_disk_monitoring {
            let mut disk_tracker = self.disk_tracker.write().await;
            disk_tracker.paths = self.config.disk_paths.clone();
        }

        Ok(())
    }

    /// Start the monitoring loop
    async fn start_monitoring_loop(&self) -> Result<()> {
        let interval = Duration::from_secs(self.config.monitoring_interval);
        let tracker = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = tracker.update_stats().await {
                    error!("Failed to update resource statistics: {}", e);
                }
            }
        });

        Ok(())
    }
}

impl Clone for ResourceTracker {
    fn clone(&self) -> Self {
        Self {
            system: Arc::clone(&self.system),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
            process_tracker: Arc::clone(&self.process_tracker),
            network_tracker: Arc::clone(&self.network_tracker),
            disk_tracker: Arc::clone(&self.disk_tracker),
            start_time: self.start_time,
        }
    }
}
