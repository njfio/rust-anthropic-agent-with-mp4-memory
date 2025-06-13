use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tracing::info;

use crate::utils::error::{AgentError, Result};
use super::{AuditConfig, AuditLogLevel, AuditStorageBackend, SecurityEvent};

/// Audit service trait
#[async_trait]
pub trait AuditService: Send + Sync {
    /// Log a security event
    async fn log_event(&self, event: SecurityEvent) -> Result<()>;
    
    /// Query audit logs
    async fn query_logs(&self, query: AuditQuery) -> Result<Vec<AuditLogEntry>>;
    
    /// Get audit statistics
    async fn get_statistics(&self, time_range: TimeRange) -> Result<AuditStatistics>;
    
    /// Archive old logs
    async fn archive_logs(&self, before_date: chrono::DateTime<chrono::Utc>) -> Result<u64>;
    
    /// Purge old logs
    async fn purge_logs(&self, before_date: chrono::DateTime<chrono::Utc>) -> Result<u64>;
    
    /// Get log retention policy
    async fn get_retention_policy(&self) -> Result<RetentionPolicy>;
    
    /// Update log retention policy
    async fn update_retention_policy(&self, policy: RetentionPolicy) -> Result<()>;
    
    /// Export logs
    async fn export_logs(&self, query: AuditQuery, format: ExportFormat) -> Result<Vec<u8>>;
    
    /// Validate log integrity
    async fn validate_integrity(&self) -> Result<IntegrityReport>;
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique log entry ID
    pub id: String,
    /// Timestamp of the event
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Log level
    pub level: AuditLogLevel,
    /// Security event
    pub event: SecurityEvent,
    /// Source of the event
    pub source: String,
    /// Session ID if applicable
    pub session_id: Option<String>,
    /// Request ID if applicable
    pub request_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Log entry hash for integrity
    pub hash: Option<String>,
}

/// Audit query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditQuery {
    /// Time range for the query
    pub time_range: TimeRange,
    /// Log levels to include
    pub levels: Vec<AuditLogLevel>,
    /// Event types to include
    pub event_types: Vec<String>,
    /// User IDs to filter by
    pub user_ids: Vec<String>,
    /// IP addresses to filter by
    pub ip_addresses: Vec<String>,
    /// Session IDs to filter by
    pub session_ids: Vec<String>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Sort order
    pub sort_order: SortOrder,
}

/// Time range for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time (inclusive)
    pub start: chrono::DateTime<chrono::Utc>,
    /// End time (exclusive)
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Sort order for query results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SortOrder {
    /// Ascending by timestamp
    TimestampAsc,
    /// Descending by timestamp
    TimestampDesc,
    /// Ascending by level
    LevelAsc,
    /// Descending by level
    LevelDesc,
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total number of log entries
    pub total_entries: u64,
    /// Entries by log level
    pub entries_by_level: HashMap<AuditLogLevel, u64>,
    /// Entries by event type
    pub entries_by_event_type: HashMap<String, u64>,
    /// Entries by user
    pub entries_by_user: HashMap<String, u64>,
    /// Entries by IP address
    pub entries_by_ip: HashMap<String, u64>,
    /// Time range covered
    pub time_range: TimeRange,
    /// Storage size in bytes
    pub storage_size_bytes: u64,
}

/// Log retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    /// Archive before deletion
    pub archive_before_deletion: bool,
    /// Compression for archived logs
    pub compress_archives: bool,
    /// Maximum storage size in bytes
    pub max_storage_bytes: Option<u64>,
    /// Auto-purge when storage limit reached
    pub auto_purge_on_limit: bool,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// XML format
    Xml,
    /// Plain text format
    Text,
}

/// Integrity report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    /// Whether integrity check passed
    pub valid: bool,
    /// Total entries checked
    pub total_entries: u64,
    /// Number of valid entries
    pub valid_entries: u64,
    /// Number of invalid entries
    pub invalid_entries: u64,
    /// List of invalid entry IDs
    pub invalid_entry_ids: Vec<String>,
    /// Check timestamp
    pub check_timestamp: chrono::DateTime<chrono::Utc>,
}

/// File-based audit service implementation
pub struct FileAuditService {
    /// Audit configuration
    config: AuditConfig,
    /// Log file path
    log_file_path: PathBuf,
    /// Log file writer
    log_writer: RwLock<Option<BufWriter<File>>>,
    /// In-memory log cache for queries
    log_cache: RwLock<Vec<AuditLogEntry>>,
    /// Retention policy
    retention_policy: RwLock<RetentionPolicy>,
}

impl FileAuditService {
    /// Create a new file audit service
    pub async fn new(config: AuditConfig, log_file_path: PathBuf) -> Result<Self> {
        // Ensure log directory exists
        if let Some(parent) = log_file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| AgentError::validation(format!("Failed to create log directory: {}", e)))?;
        }

        let retention_policy = RetentionPolicy {
            retention_days: config.retention_days,
            archive_before_deletion: true,
            compress_archives: true,
            max_storage_bytes: Some(1024 * 1024 * 1024), // 1GB default
            auto_purge_on_limit: true,
        };

        Ok(Self {
            config,
            log_file_path,
            log_writer: RwLock::new(None),
            log_cache: RwLock::new(Vec::new()),
            retention_policy: RwLock::new(retention_policy),
        })
    }

    /// Initialize the log writer
    async fn initialize_writer(&self) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)
            .await
            .map_err(|e| AgentError::validation(format!("Failed to open log file: {}", e)))?;

        let writer = BufWriter::new(file);
        let mut log_writer = self.log_writer.write().await;
        *log_writer = Some(writer);
        Ok(())
    }

    /// Write log entry to file
    async fn write_to_file(&self, entry: &AuditLogEntry) -> Result<()> {
        let mut log_writer = self.log_writer.write().await;
        
        if log_writer.is_none() {
            drop(log_writer);
            self.initialize_writer().await?;
            log_writer = self.log_writer.write().await;
        }

        if let Some(writer) = log_writer.as_mut() {
            let json_line = serde_json::to_string(entry)
                .map_err(|e| AgentError::validation(format!("Failed to serialize log entry: {}", e)))?;
            
            writer.write_all(json_line.as_bytes()).await
                .map_err(|e| AgentError::validation(format!("Failed to write to log file: {}", e)))?;
            writer.write_all(b"\n").await
                .map_err(|e| AgentError::validation(format!("Failed to write newline: {}", e)))?;
            writer.flush().await
                .map_err(|e| AgentError::validation(format!("Failed to flush log file: {}", e)))?;
        }

        Ok(())
    }

    /// Calculate hash for log entry
    fn calculate_hash(&self, entry: &AuditLogEntry) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        entry.id.hash(&mut hasher);
        entry.timestamp.hash(&mut hasher);
        entry.source.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }

    /// Check if event should be logged based on level
    fn should_log_event(&self, event: &SecurityEvent) -> bool {
        match self.config.log_level {
            AuditLogLevel::All => true,
            AuditLogLevel::Security => matches!(event, 
                SecurityEvent::AuthenticationAttempt { .. } |
                SecurityEvent::AuthorizationCheck { .. } |
                SecurityEvent::SessionCreated { .. } |
                SecurityEvent::SessionTerminated { .. } |
                SecurityEvent::PasswordChanged { .. } |
                SecurityEvent::PolicyViolation { .. } |
                SecurityEvent::SecurityIncident { .. }
            ),
            AuditLogLevel::Authentication => matches!(event,
                SecurityEvent::AuthenticationAttempt { .. } |
                SecurityEvent::PasswordChanged { .. }
            ),
            AuditLogLevel::Authorization => matches!(event,
                SecurityEvent::AuthorizationCheck { .. }
            ),
            AuditLogLevel::Critical => matches!(event,
                SecurityEvent::PolicyViolation { .. } |
                SecurityEvent::SecurityIncident { .. }
            ),
        }
    }

    /// Get event type string
    fn get_event_type(&self, event: &SecurityEvent) -> String {
        match event {
            SecurityEvent::AuthenticationAttempt { .. } => "authentication_attempt".to_string(),
            SecurityEvent::AuthorizationCheck { .. } => "authorization_check".to_string(),
            SecurityEvent::SessionCreated { .. } => "session_created".to_string(),
            SecurityEvent::SessionTerminated { .. } => "session_terminated".to_string(),
            SecurityEvent::PasswordChanged { .. } => "password_changed".to_string(),
            SecurityEvent::RateLimitExceeded { .. } => "rate_limit_exceeded".to_string(),
            SecurityEvent::PolicyViolation { .. } => "policy_violation".to_string(),
            SecurityEvent::DataAccess { .. } => "data_access".to_string(),
            SecurityEvent::ConfigurationChange { .. } => "configuration_change".to_string(),
            SecurityEvent::SecurityIncident { .. } => "security_incident".to_string(),
        }
    }
}

#[async_trait]
impl AuditService for FileAuditService {
    async fn log_event(&self, event: SecurityEvent) -> Result<()> {
        if !self.config.enabled || !self.should_log_event(&event) {
            return Ok(());
        }

        let entry_id = uuid::Uuid::new_v4().to_string();
        let mut entry = AuditLogEntry {
            id: entry_id,
            timestamp: chrono::Utc::now(),
            level: self.config.log_level.clone(),
            event,
            source: "rust-agent".to_string(),
            session_id: None,
            request_id: None,
            metadata: HashMap::new(),
            hash: None,
        };

        // Calculate hash for integrity
        entry.hash = Some(self.calculate_hash(&entry));

        // Write to file
        self.write_to_file(&entry).await?;

        // Add to cache
        let mut cache = self.log_cache.write().await;
        cache.push(entry);

        // Limit cache size
        if cache.len() > 10000 {
            cache.drain(0..1000);
        }

        Ok(())
    }

    async fn query_logs(&self, query: AuditQuery) -> Result<Vec<AuditLogEntry>> {
        let cache = self.log_cache.read().await;
        let mut results: Vec<AuditLogEntry> = cache
            .iter()
            .filter(|entry| {
                // Time range filter
                entry.timestamp >= query.time_range.start && entry.timestamp < query.time_range.end
            })
            .filter(|entry| {
                // Level filter
                query.levels.is_empty() || query.levels.contains(&entry.level)
            })
            .filter(|entry| {
                // Event type filter
                if query.event_types.is_empty() {
                    true
                } else {
                    let event_type = self.get_event_type(&entry.event);
                    query.event_types.contains(&event_type)
                }
            })
            .cloned()
            .collect();

        // Sort results
        match query.sort_order {
            SortOrder::TimestampAsc => results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp)),
            SortOrder::TimestampDesc => results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp)),
            SortOrder::LevelAsc => results.sort_by(|a, b| format!("{:?}", a.level).cmp(&format!("{:?}", b.level))),
            SortOrder::LevelDesc => results.sort_by(|a, b| format!("{:?}", b.level).cmp(&format!("{:?}", a.level))),
        }

        // Apply pagination
        if let Some(offset) = query.offset {
            if offset < results.len() {
                results = results.into_iter().skip(offset).collect();
            } else {
                results.clear();
            }
        }

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn get_statistics(&self, time_range: TimeRange) -> Result<AuditStatistics> {
        let cache = self.log_cache.read().await;
        let filtered_entries: Vec<&AuditLogEntry> = cache
            .iter()
            .filter(|entry| entry.timestamp >= time_range.start && entry.timestamp < time_range.end)
            .collect();

        let mut entries_by_level = HashMap::new();
        let mut entries_by_event_type = HashMap::new();
        let mut entries_by_user = HashMap::new();
        let mut entries_by_ip = HashMap::new();

        for entry in &filtered_entries {
            // Count by level
            *entries_by_level.entry(entry.level.clone()).or_insert(0) += 1;

            // Count by event type
            let event_type = self.get_event_type(&entry.event);
            *entries_by_event_type.entry(event_type).or_insert(0) += 1;

            // Count by user (extract from event)
            match &entry.event {
                SecurityEvent::AuthenticationAttempt { user_id, .. } |
                SecurityEvent::AuthorizationCheck { user_id, .. } |
                SecurityEvent::SessionCreated { user_id, .. } |
                SecurityEvent::SessionTerminated { user_id, .. } |
                SecurityEvent::PasswordChanged { user_id, .. } |
                SecurityEvent::DataAccess { user_id, .. } |
                SecurityEvent::ConfigurationChange { user_id, .. } => {
                    *entries_by_user.entry(user_id.clone()).or_insert(0) += 1;
                }
                SecurityEvent::RateLimitExceeded { user_id: Some(user_id), .. } => {
                    *entries_by_user.entry(user_id.clone()).or_insert(0) += 1;
                }
                _ => {}
            }

            // Count by IP address (extract from event)
            match &entry.event {
                SecurityEvent::AuthenticationAttempt { ip_address: Some(ip), .. } |
                SecurityEvent::SessionCreated { ip_address: Some(ip), .. } |
                SecurityEvent::PasswordChanged { ip_address: Some(ip), .. } |
                SecurityEvent::RateLimitExceeded { ip_address: Some(ip), .. } => {
                    *entries_by_ip.entry(ip.clone()).or_insert(0) += 1;
                }
                _ => {}
            }
        }

        // Calculate storage size (approximate)
        let storage_size_bytes = if let Ok(metadata) = tokio::fs::metadata(&self.log_file_path).await {
            metadata.len()
        } else {
            0
        };

        Ok(AuditStatistics {
            total_entries: filtered_entries.len() as u64,
            entries_by_level,
            entries_by_event_type,
            entries_by_user,
            entries_by_ip,
            time_range,
            storage_size_bytes,
        })
    }

    async fn archive_logs(&self, before_date: chrono::DateTime<chrono::Utc>) -> Result<u64> {
        // For now, just return 0 (not implemented)
        // In a real implementation, this would move old logs to archive storage
        info!("Archive logs before {} (not implemented)", before_date);
        Ok(0)
    }

    async fn purge_logs(&self, before_date: chrono::DateTime<chrono::Utc>) -> Result<u64> {
        let mut cache = self.log_cache.write().await;
        let initial_count = cache.len();
        
        cache.retain(|entry| entry.timestamp >= before_date);
        
        let purged_count = initial_count - cache.len();
        info!("Purged {} log entries before {}", purged_count, before_date);
        
        Ok(purged_count as u64)
    }

    async fn get_retention_policy(&self) -> Result<RetentionPolicy> {
        let policy = self.retention_policy.read().await;
        Ok(policy.clone())
    }

    async fn update_retention_policy(&self, policy: RetentionPolicy) -> Result<()> {
        let mut current_policy = self.retention_policy.write().await;
        *current_policy = policy;
        Ok(())
    }

    async fn export_logs(&self, query: AuditQuery, format: ExportFormat) -> Result<Vec<u8>> {
        let logs = self.query_logs(query).await?;
        
        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&logs)
                    .map_err(|e| AgentError::validation(format!("Failed to serialize to JSON: {}", e)))?;
                Ok(json.into_bytes())
            }
            ExportFormat::Csv => {
                // Simple CSV implementation
                let mut csv = String::new();
                csv.push_str("id,timestamp,level,event_type,source\n");
                
                for entry in logs {
                    let event_type = self.get_event_type(&entry.event);
                    csv.push_str(&format!(
                        "{},{},{:?},{},{}\n",
                        entry.id, entry.timestamp, entry.level, event_type, entry.source
                    ));
                }
                
                Ok(csv.into_bytes())
            }
            _ => Err(AgentError::validation("Export format not supported".to_string())),
        }
    }

    async fn validate_integrity(&self) -> Result<IntegrityReport> {
        let cache = self.log_cache.read().await;
        let mut valid_entries = 0;
        let mut invalid_entry_ids = Vec::new();

        for entry in cache.iter() {
            let calculated_hash = self.calculate_hash(entry);
            if let Some(stored_hash) = &entry.hash {
                if calculated_hash == *stored_hash {
                    valid_entries += 1;
                } else {
                    invalid_entry_ids.push(entry.id.clone());
                }
            } else {
                invalid_entry_ids.push(entry.id.clone());
            }
        }

        let total_entries = cache.len() as u64;
        let invalid_entries = invalid_entry_ids.len() as u64;

        Ok(IntegrityReport {
            valid: invalid_entries == 0,
            total_entries,
            valid_entries,
            invalid_entries,
            invalid_entry_ids,
            check_timestamp: chrono::Utc::now(),
        })
    }
}

/// Create an audit service
pub async fn create_audit_service(config: &AuditConfig) -> Result<Box<dyn AuditService>> {
    match config.storage_backend {
        AuditStorageBackend::File => {
            let log_path = PathBuf::from("logs/audit.log");
            let service = FileAuditService::new(config.clone(), log_path).await?;
            Ok(Box::new(service))
        }
        _ => Err(AgentError::validation("Audit storage backend not supported".to_string())),
    }
}
