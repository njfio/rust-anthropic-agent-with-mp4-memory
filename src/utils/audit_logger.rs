use crate::utils::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

/// Audit event severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Types of auditable operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditEventType {
    FileAccess,
    FileModification,
    CommandExecution,
    NetworkRequest,
    AuthenticationAttempt,
    ConfigurationChange,
    SecurityViolation,
    RateLimitExceeded,
    ToolExecution,
    MemoryOperation,
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: u64,
    pub event_type: AuditEventType,
    pub severity: AuditSeverity,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub operation: String,
    pub resource: Option<String>,
    pub source_ip: Option<String>,
    pub user_agent: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType, severity: AuditSeverity, operation: String) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type,
            severity,
            user_id: None,
            session_id: None,
            operation,
            resource: None,
            source_ip: None,
            user_agent: None,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Set resource being accessed
    pub fn with_resource<S: Into<String>>(mut self, resource: S) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Set user ID
    pub fn with_user_id<S: Into<String>>(mut self, user_id: S) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set session ID
    pub fn with_session_id<S: Into<String>>(mut self, session_id: S) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set source IP
    pub fn with_source_ip<S: Into<String>>(mut self, ip: S) -> Self {
        self.source_ip = Some(ip.into());
        self
    }

    /// Set success status
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Set error message
    pub fn with_error<S: Into<String>>(mut self, error: S) -> Self {
        self.error_message = Some(error.into());
        self.success = false;
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Audit logger configuration
#[derive(Debug, Clone)]
pub struct AuditLoggerConfig {
    pub log_file_path: PathBuf,
    pub max_file_size: u64,
    pub max_files: usize,
    pub buffer_size: usize,
    pub sync_interval_seconds: u64,
    pub minimum_severity: AuditSeverity,
}

impl Default for AuditLoggerConfig {
    fn default() -> Self {
        Self {
            log_file_path: PathBuf::from("audit.log"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 10,
            buffer_size: 1000,
            sync_interval_seconds: 60,
            minimum_severity: AuditSeverity::Low,
        }
    }
}

/// Thread-safe audit logger
#[derive(Debug)]
pub struct AuditLogger {
    config: AuditLoggerConfig,
    writer: Arc<Mutex<BufWriter<File>>>,
    event_count: Arc<Mutex<u64>>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditLoggerConfig) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.log_file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open log file in append mode
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_file_path)?;

        let writer = Arc::new(Mutex::new(BufWriter::new(file)));
        let event_count = Arc::new(Mutex::new(0));

        Ok(Self {
            config,
            writer,
            event_count,
        })
    }

    /// Log an audit event
    pub fn log_event(&self, event: AuditEvent) -> Result<()> {
        // Check severity filter
        if !self.should_log_severity(&event.severity) {
            return Ok(());
        }

        // Serialize event to JSON
        let json_line = serde_json::to_string(&event)
            .map_err(|e| AgentError::config(format!("Failed to serialize audit event: {}", e)))?;

        // Write to file
        {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| AgentError::config("Audit logger mutex poisoned".to_string()))?;

            writeln!(writer, "{}", json_line)?;

            // Increment event count
            let mut count = self
                .event_count
                .lock()
                .map_err(|_| AgentError::config("Event count mutex poisoned".to_string()))?;
            *count += 1;

            // Flush periodically or on high severity events
            if matches!(
                event.severity,
                AuditSeverity::High | AuditSeverity::Critical
            ) || *count % 100 == 0
            {
                writer.flush()?;
            }
        }

        // Log to tracing as well for immediate visibility
        match event.severity {
            AuditSeverity::Critical => error!("AUDIT: {}", event.operation),
            AuditSeverity::High => warn!("AUDIT: {}", event.operation),
            AuditSeverity::Medium | AuditSeverity::Low => info!("AUDIT: {}", event.operation),
        }

        // Check if log rotation is needed
        self.check_log_rotation()?;

        Ok(())
    }

    /// Check if we should log events of this severity
    fn should_log_severity(&self, severity: &AuditSeverity) -> bool {
        use AuditSeverity::*;
        match (&self.config.minimum_severity, severity) {
            (Low, _) => true,
            (Medium, Medium | High | Critical) => true,
            (High, High | Critical) => true,
            (Critical, Critical) => true,
            _ => false,
        }
    }

    /// Check if log rotation is needed
    fn check_log_rotation(&self) -> Result<()> {
        let metadata = std::fs::metadata(&self.config.log_file_path)?;
        if metadata.len() > self.config.max_file_size {
            self.rotate_logs()?;
        }
        Ok(())
    }

    /// Rotate log files
    fn rotate_logs(&self) -> Result<()> {
        // Flush current writer
        {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| AgentError::config("Audit logger mutex poisoned".to_string()))?;
            writer.flush()?;
        }

        // Rotate existing files
        for i in (1..self.config.max_files).rev() {
            let old_path = format!("{}.{}", self.config.log_file_path.display(), i);
            let new_path = format!("{}.{}", self.config.log_file_path.display(), i + 1);

            if Path::new(&old_path).exists() {
                if i + 1 >= self.config.max_files {
                    // Delete oldest file
                    let _ = std::fs::remove_file(&old_path);
                } else {
                    let _ = std::fs::rename(&old_path, &new_path);
                }
            }
        }

        // Move current log to .1
        let backup_path = format!("{}.1", self.config.log_file_path.display());
        let _ = std::fs::rename(&self.config.log_file_path, &backup_path);

        // Create new log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.log_file_path)?;

        // Replace writer
        {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| AgentError::config("Audit logger mutex poisoned".to_string()))?;
            *writer = BufWriter::new(file);
        }

        info!("Audit log rotated");
        Ok(())
    }

    /// Flush all pending writes
    pub fn flush(&self) -> Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| AgentError::config("Audit logger mutex poisoned".to_string()))?;
        writer.flush()?;
        Ok(())
    }

    /// Get audit statistics
    pub fn get_stats(&self) -> Result<AuditStats> {
        let count = *self
            .event_count
            .lock()
            .map_err(|_| AgentError::config("Event count mutex poisoned".to_string()))?;

        let file_size = std::fs::metadata(&self.config.log_file_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(AuditStats {
            total_events: count,
            current_file_size: file_size,
            max_file_size: self.config.max_file_size,
        })
    }
}

/// Audit statistics
#[derive(Debug, Clone, Serialize)]
pub struct AuditStats {
    pub total_events: u64,
    pub current_file_size: u64,
    pub max_file_size: u64,
}

/// Global audit logger instance
static mut AUDIT_LOGGER: Option<Arc<AuditLogger>> = None;
static AUDIT_LOGGER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global audit logger
pub fn init_audit_logger(config: AuditLoggerConfig) -> Result<()> {
    AUDIT_LOGGER_INIT.call_once(|| match AuditLogger::new(config) {
        Ok(logger) => {
            unsafe {
                AUDIT_LOGGER = Some(Arc::new(logger));
            }
            info!("Audit logger initialized");
        }
        Err(e) => {
            error!("Failed to initialize audit logger: {}", e);
        }
    });
    Ok(())
}

/// Get global audit logger
pub fn get_audit_logger() -> Option<Arc<AuditLogger>> {
    unsafe { AUDIT_LOGGER.as_ref().cloned() }
}

/// Convenience function to log audit events
pub fn audit_log(event: AuditEvent) {
    if let Some(logger) = get_audit_logger() {
        if let Err(e) = logger.log_event(event) {
            error!("Failed to log audit event: {}", e);
        }
    }
}

/// Macro for easy audit logging
#[macro_export]
macro_rules! audit {
    ($event_type:expr, $severity:expr, $operation:expr) => {
        $crate::utils::audit_logger::audit_log(
            $crate::utils::audit_logger::AuditEvent::new($event_type, $severity, $operation.to_string())
        );
    };
    ($event_type:expr, $severity:expr, $operation:expr, $($key:expr => $value:expr),*) => {
        {
            let mut event = $crate::utils::audit_logger::AuditEvent::new($event_type, $severity, $operation.to_string());
            $(
                event = event.with_metadata($key, $value);
            )*
            $crate::utils::audit_logger::audit_log(event);
        }
    };
}
