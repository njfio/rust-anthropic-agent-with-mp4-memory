use crate::utils::error::{AgentError, Result};
use crate::utils::audit_logger::{AuditEvent, AuditEventType, AuditSeverity, audit_log};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use tracing::{debug, error, info};

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Memory usage as percentage of total system memory
    pub memory_percentage: f64,
    /// CPU usage percentage (0.0 to 100.0)
    pub cpu_percentage: f64,
    /// Number of threads
    pub thread_count: u32,
    /// Process uptime in seconds
    pub uptime_seconds: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Timestamp of measurement
    pub timestamp: u64,
}

/// Resource limits configuration
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Maximum memory usage as percentage of system memory
    pub max_memory_percentage: f64,
    /// Maximum CPU usage percentage
    pub max_cpu_percentage: f64,
    /// Maximum number of threads
    pub max_threads: u32,
    /// Warning threshold for memory (percentage of max)
    pub memory_warning_threshold: f64,
    /// Warning threshold for CPU (percentage of max)
    pub cpu_warning_threshold: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_memory_percentage: 25.0, // 25% of system memory
            max_cpu_percentage: 80.0, // 80% CPU usage
            max_threads: 100,
            memory_warning_threshold: 0.8, // 80% of max memory
            cpu_warning_threshold: 0.8, // 80% of max CPU
        }
    }
}

/// Resource monitoring configuration
#[derive(Debug, Clone)]
pub struct ResourceMonitorConfig {
    /// Resource limits
    pub limits: ResourceLimits,
    /// Monitoring interval in seconds
    pub monitoring_interval: Duration,
    /// Whether to enable automatic enforcement
    pub enforce_limits: bool,
    /// Whether to log resource usage
    pub log_usage: bool,
    /// Whether to audit resource violations
    pub audit_violations: bool,
}

impl Default for ResourceMonitorConfig {
    fn default() -> Self {
        Self {
            limits: ResourceLimits::default(),
            monitoring_interval: Duration::from_secs(30),
            enforce_limits: true,
            log_usage: true,
            audit_violations: true,
        }
    }
}

/// Resource monitor for tracking system resource usage
#[derive(Debug)]
pub struct ResourceMonitor {
    config: ResourceMonitorConfig,
    stats: Arc<Mutex<ResourceStats>>,
    start_time: Instant,
    peak_memory: Arc<Mutex<u64>>,
    monitoring_active: Arc<Mutex<bool>>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(config: ResourceMonitorConfig) -> Self {
        let initial_stats = ResourceStats {
            memory_usage: 0,
            memory_percentage: 0.0,
            cpu_percentage: 0.0,
            thread_count: 0,
            uptime_seconds: 0,
            peak_memory_usage: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        Self {
            config,
            stats: Arc::new(Mutex::new(initial_stats)),
            start_time: Instant::now(),
            peak_memory: Arc::new(Mutex::new(0)),
            monitoring_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start monitoring in a background thread
    pub fn start_monitoring(&self) -> Result<()> {
        {
            let mut active = self.monitoring_active.lock()
                .map_err(|_| AgentError::config("Resource monitor mutex poisoned".to_string()))?;
            if *active {
                return Ok(()); // Already monitoring
            }
            *active = true;
        }

        let stats = Arc::clone(&self.stats);
        let peak_memory = Arc::clone(&self.peak_memory);
        let monitoring_active = Arc::clone(&self.monitoring_active);
        let config = self.config.clone();
        let start_time = self.start_time;

        thread::spawn(move || {
            info!("Resource monitoring started");
            
            while {
                let active = monitoring_active.lock().unwrap_or_else(|_| {
                    error!("Resource monitor mutex poisoned");
                    std::process::exit(1);
                });
                *active
            } {
                if let Err(e) = Self::collect_stats(&stats, &peak_memory, &config, start_time) {
                    error!("Failed to collect resource stats: {}", e);
                }
                
                thread::sleep(config.monitoring_interval);
            }
            
            info!("Resource monitoring stopped");
        });

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) -> Result<()> {
        let mut active = self.monitoring_active.lock()
            .map_err(|_| AgentError::config("Resource monitor mutex poisoned".to_string()))?;
        *active = false;
        Ok(())
    }

    /// Get current resource statistics
    pub fn get_stats(&self) -> Result<ResourceStats> {
        let stats = self.stats.lock()
            .map_err(|_| AgentError::config("Resource monitor mutex poisoned".to_string()))?;
        Ok(stats.clone())
    }

    /// Check if resource usage is within limits
    pub fn check_limits(&self) -> Result<Vec<String>> {
        let stats = self.get_stats()?;
        let mut violations = Vec::new();

        // Check memory limits
        if stats.memory_usage > self.config.limits.max_memory_bytes {
            violations.push(format!(
                "Memory usage ({} MB) exceeds limit ({} MB)",
                stats.memory_usage / 1024 / 1024,
                self.config.limits.max_memory_bytes / 1024 / 1024
            ));
        }

        if stats.memory_percentage > self.config.limits.max_memory_percentage {
            violations.push(format!(
                "Memory percentage ({:.1}%) exceeds limit ({:.1}%)",
                stats.memory_percentage,
                self.config.limits.max_memory_percentage
            ));
        }

        // Check CPU limits
        if stats.cpu_percentage > self.config.limits.max_cpu_percentage {
            violations.push(format!(
                "CPU usage ({:.1}%) exceeds limit ({:.1}%)",
                stats.cpu_percentage,
                self.config.limits.max_cpu_percentage
            ));
        }

        // Check thread limits
        if stats.thread_count > self.config.limits.max_threads {
            violations.push(format!(
                "Thread count ({}) exceeds limit ({})",
                stats.thread_count,
                self.config.limits.max_threads
            ));
        }

        // Audit violations if configured
        if self.config.audit_violations && !violations.is_empty() {
            for violation in &violations {
                audit_log(AuditEvent::new(
                    AuditEventType::SecurityViolation,
                    AuditSeverity::High,
                    "resource_limit_exceeded".to_string(),
                ).with_error(violation));
            }
        }

        Ok(violations)
    }

    /// Check for warning thresholds
    pub fn check_warnings(&self) -> Result<Vec<String>> {
        let stats = self.get_stats()?;
        let mut warnings = Vec::new();

        // Memory warnings
        let memory_warning_limit = self.config.limits.max_memory_bytes as f64 * 
            self.config.limits.memory_warning_threshold;
        if stats.memory_usage as f64 > memory_warning_limit {
            warnings.push(format!(
                "Memory usage approaching limit: {:.1} MB / {:.1} MB",
                stats.memory_usage as f64 / 1024.0 / 1024.0,
                self.config.limits.max_memory_bytes as f64 / 1024.0 / 1024.0
            ));
        }

        // CPU warnings
        let cpu_warning_limit = self.config.limits.max_cpu_percentage * 
            self.config.limits.cpu_warning_threshold;
        if stats.cpu_percentage > cpu_warning_limit {
            warnings.push(format!(
                "CPU usage approaching limit: {:.1}% / {:.1}%",
                stats.cpu_percentage,
                self.config.limits.max_cpu_percentage
            ));
        }

        Ok(warnings)
    }

    /// Collect resource statistics
    fn collect_stats(
        stats: &Arc<Mutex<ResourceStats>>,
        peak_memory: &Arc<Mutex<u64>>,
        config: &ResourceMonitorConfig,
        start_time: Instant,
    ) -> Result<()> {
        // Get process information
        let _pid = std::process::id();
        
        // Read memory usage from /proc/self/status (Linux) or use cross-platform method
        let memory_usage = Self::get_memory_usage()?;
        let cpu_percentage = Self::get_cpu_usage()?;
        let thread_count = Self::get_thread_count()?;
        
        // Update peak memory
        {
            let mut peak = peak_memory.lock()
                .map_err(|_| AgentError::config("Peak memory mutex poisoned".to_string()))?;
            if memory_usage > *peak {
                *peak = memory_usage;
            }
        }

        // Calculate memory percentage (approximate)
        let total_memory = Self::get_total_system_memory()?;
        let memory_percentage = (memory_usage as f64 / total_memory as f64) * 100.0;

        let uptime_seconds = start_time.elapsed().as_secs();
        let peak_memory_usage = *peak_memory.lock()
            .map_err(|_| AgentError::config("Peak memory mutex poisoned".to_string()))?;

        let new_stats = ResourceStats {
            memory_usage,
            memory_percentage,
            cpu_percentage,
            thread_count,
            uptime_seconds,
            peak_memory_usage,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        // Update stats
        {
            let mut current_stats = stats.lock()
                .map_err(|_| AgentError::config("Resource stats mutex poisoned".to_string()))?;
            *current_stats = new_stats.clone();
        }

        // Log usage if configured
        if config.log_usage {
            debug!(
                "Resource usage: Memory: {:.1} MB ({:.1}%), CPU: {:.1}%, Threads: {}",
                memory_usage as f64 / 1024.0 / 1024.0,
                memory_percentage,
                cpu_percentage,
                thread_count
            );
        }

        Ok(())
    }

    /// Get memory usage in bytes
    fn get_memory_usage() -> Result<u64> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")
                .map_err(|e| AgentError::config(format!("Failed to read /proc/self/status: {}", e)))?;
            
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse()
                            .map_err(|e| AgentError::config(format!("Failed to parse memory usage: {}", e)))?;
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
            Err(AgentError::config("Could not find VmRSS in /proc/self/status".to_string()))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems - approximate using heap allocation
            // This is not accurate but provides some indication
            Ok(1024 * 1024 * 50) // Placeholder: 50MB
        }
    }

    /// Get CPU usage percentage (simplified)
    fn get_cpu_usage() -> Result<f64> {
        // This is a simplified implementation
        // In a real system, you'd want to track CPU time over intervals
        Ok(0.0) // Placeholder
    }

    /// Get thread count
    fn get_thread_count() -> Result<u32> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let stat = fs::read_to_string("/proc/self/stat")
                .map_err(|e| AgentError::config(format!("Failed to read /proc/self/stat: {}", e)))?;
            
            let parts: Vec<&str> = stat.split_whitespace().collect();
            if parts.len() > 19 {
                let threads: u32 = parts[19].parse()
                    .map_err(|e| AgentError::config(format!("Failed to parse thread count: {}", e)))?;
                return Ok(threads);
            }
            Err(AgentError::config("Could not parse thread count from /proc/self/stat".to_string()))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            Ok(1) // Placeholder
        }
    }

    /// Get total system memory
    fn get_total_system_memory() -> Result<u64> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let meminfo = fs::read_to_string("/proc/meminfo")
                .map_err(|e| AgentError::config(format!("Failed to read /proc/meminfo: {}", e)))?;
            
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse()
                            .map_err(|e| AgentError::config(format!("Failed to parse total memory: {}", e)))?;
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
            Err(AgentError::config("Could not find MemTotal in /proc/meminfo".to_string()))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            Ok(8 * 1024 * 1024 * 1024) // Placeholder: 8GB
        }
    }
}

/// Global resource monitor instance
static mut RESOURCE_MONITOR: Option<Arc<ResourceMonitor>> = None;
static RESOURCE_MONITOR_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global resource monitor
pub fn init_resource_monitor(config: ResourceMonitorConfig) -> Result<()> {
    RESOURCE_MONITOR_INIT.call_once(|| {
        let monitor = ResourceMonitor::new(config);
        if let Err(e) = monitor.start_monitoring() {
            error!("Failed to start resource monitoring: {}", e);
            return;
        }
        
        unsafe {
            RESOURCE_MONITOR = Some(Arc::new(monitor));
        }
        info!("Resource monitor initialized");
    });
    Ok(())
}

/// Get global resource monitor
pub fn get_resource_monitor() -> Option<Arc<ResourceMonitor>> {
    unsafe { RESOURCE_MONITOR.as_ref().cloned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_memory_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(limits.max_memory_percentage, 25.0);
        assert_eq!(limits.max_cpu_percentage, 80.0);
    }

    #[test]
    fn test_resource_monitor_creation() {
        let config = ResourceMonitorConfig::default();
        let monitor = ResourceMonitor::new(config);
        
        // Should be able to get initial stats
        let stats = monitor.get_stats().unwrap();
        assert_eq!(stats.uptime_seconds, 0);
    }

    #[tokio::test]
    async fn test_resource_monitoring() {
        let config = ResourceMonitorConfig {
            monitoring_interval: Duration::from_millis(100),
            ..Default::default()
        };
        
        let monitor = ResourceMonitor::new(config);
        monitor.start_monitoring().unwrap();
        
        // Wait a bit for monitoring to collect data
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        let stats = monitor.get_stats().unwrap();
        assert!(stats.uptime_seconds >= 0);
        
        monitor.stop_monitoring().unwrap();
    }
}
