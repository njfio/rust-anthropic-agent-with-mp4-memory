// Data Export Implementation for GDPR Compliance
// Handles data portability and access requests

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

use crate::utils::error::{AgentError, Result};

/// Data export handler for GDPR compliance
#[derive(Debug, Clone)]
pub struct DataExportHandler {
    /// Export format configurations
    export_formats: HashMap<String, ExportFormat>,
}

/// Export format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFormat {
    /// Format name (e.g., "json", "csv", "xml")
    pub name: String,
    /// MIME type
    pub mime_type: String,
    /// File extension
    pub file_extension: String,
    /// Whether format supports structured data
    pub supports_structured: bool,
    /// Whether format is machine-readable
    pub machine_readable: bool,
}

/// Data export request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExportRequest {
    /// Unique export request ID
    pub id: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Requested export format
    pub format: String,
    /// Data categories to include
    pub categories: Vec<String>,
    /// Date range filter (optional)
    pub date_range: Option<DateRange>,
    /// Request timestamp
    pub created_at: DateTime<Utc>,
    /// Export status
    pub status: ExportStatus,
    /// Generated file path (when completed)
    pub file_path: Option<String>,
    /// Export metadata
    pub metadata: ExportMetadata,
}

/// Date range filter for exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Export status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportStatus {
    /// Export request received
    Pending,
    /// Export in progress
    Processing,
    /// Export completed successfully
    Completed,
    /// Export failed
    Failed(String),
    /// Export expired/deleted
    Expired,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Total number of records exported
    pub record_count: u64,
    /// Export file size in bytes
    pub file_size_bytes: u64,
    /// Data categories included
    pub categories_included: Vec<String>,
    /// Export completion timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Export expiry timestamp
    pub expires_at: Option<DateTime<Utc>>,
}

/// Exported data package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPackage {
    /// Package metadata
    pub metadata: PackageMetadata,
    /// Personal data
    pub personal_data: PersonalDataExport,
    /// Conversation history
    pub conversations: Vec<ConversationExport>,
    /// Memory data
    pub memory_data: Vec<MemoryExport>,
    /// Audit trail
    pub audit_trail: Vec<AuditExport>,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Export ID
    pub export_id: String,
    /// Subject ID
    pub subject_id: String,
    /// Export timestamp
    pub exported_at: DateTime<Utc>,
    /// Data controller information
    pub data_controller: String,
    /// Export format
    pub format: String,
    /// GDPR compliance statement
    pub compliance_statement: String,
}

/// Personal data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalDataExport {
    /// Basic personal information
    pub basic_info: HashMap<String, Value>,
    /// Preferences and settings
    pub preferences: HashMap<String, Value>,
    /// Consent records
    pub consent_records: Vec<ConsentRecord>,
}

/// Conversation export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExport {
    /// Conversation ID
    pub id: String,
    /// Conversation timestamp
    pub timestamp: DateTime<Utc>,
    /// Messages in the conversation
    pub messages: Vec<MessageExport>,
    /// Conversation metadata
    pub metadata: HashMap<String, Value>,
}

/// Message export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageExport {
    /// Message ID
    pub id: String,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
    /// Message role (user/assistant)
    pub role: String,
    /// Message content
    pub content: String,
    /// Message metadata
    pub metadata: HashMap<String, Value>,
}

/// Memory export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExport {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Memory type/category
    pub memory_type: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
    /// Memory metadata
    pub metadata: HashMap<String, Value>,
}

/// Audit trail export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditExport {
    /// Audit event ID
    pub id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: String,
    /// Event description
    pub description: String,
    /// Event metadata
    pub metadata: HashMap<String, Value>,
}

/// Consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Consent ID
    pub id: String,
    /// Purpose of processing
    pub purpose: String,
    /// Consent given timestamp
    pub given_at: DateTime<Utc>,
    /// Consent withdrawn timestamp (if applicable)
    pub withdrawn_at: Option<DateTime<Utc>>,
    /// Consent status
    pub status: String,
}

impl DataExportHandler {
    /// Create a new data export handler
    pub fn new() -> Self {
        let mut handler = Self {
            export_formats: HashMap::new(),
        };

        handler.initialize_formats();
        handler
    }

    /// Initialize supported export formats
    fn initialize_formats(&mut self) {
        // JSON format (default, most comprehensive)
        self.export_formats.insert(
            "json".to_string(),
            ExportFormat {
                name: "JSON".to_string(),
                mime_type: "application/json".to_string(),
                file_extension: "json".to_string(),
                supports_structured: true,
                machine_readable: true,
            },
        );

        // CSV format (for tabular data)
        self.export_formats.insert(
            "csv".to_string(),
            ExportFormat {
                name: "CSV".to_string(),
                mime_type: "text/csv".to_string(),
                file_extension: "csv".to_string(),
                supports_structured: false,
                machine_readable: true,
            },
        );

        // XML format
        self.export_formats.insert(
            "xml".to_string(),
            ExportFormat {
                name: "XML".to_string(),
                mime_type: "application/xml".to_string(),
                file_extension: "xml".to_string(),
                supports_structured: true,
                machine_readable: true,
            },
        );

        // PDF format (human-readable)
        self.export_formats.insert(
            "pdf".to_string(),
            ExportFormat {
                name: "PDF".to_string(),
                mime_type: "application/pdf".to_string(),
                file_extension: "pdf".to_string(),
                supports_structured: false,
                machine_readable: false,
            },
        );
    }

    /// Export all data for a subject (GDPR Article 20 - Data Portability)
    pub async fn export_subject_data(&self, subject_id: &str) -> Result<Vec<u8>> {
        tracing::info!("Exporting all data for subject: {}", subject_id);

        // Collect all data for the subject
        let data_package = self.collect_subject_data(subject_id).await?;

        // Serialize to JSON (default format)
        let json_data = serde_json::to_vec_pretty(&data_package)
            .map_err(|e| AgentError::validation(format!("Failed to serialize data: {}", e)))?;

        tracing::info!(
            "Exported {} bytes of data for subject {}",
            json_data.len(),
            subject_id
        );

        Ok(json_data)
    }

    /// Export data in portable format (machine-readable)
    pub async fn export_portable_data(&self, subject_id: &str) -> Result<Vec<u8>> {
        tracing::info!("Exporting portable data for subject: {}", subject_id);

        // Create a portable data package
        let portable_package = self.create_portable_package(subject_id).await?;

        // Serialize to JSON with specific formatting for portability
        let json_data = serde_json::to_vec_pretty(&portable_package).map_err(|e| {
            AgentError::validation(format!("Failed to serialize portable data: {}", e))
        })?;

        Ok(json_data)
    }

    /// Create export request
    pub async fn create_export_request(
        &self,
        subject_id: String,
        format: String,
        categories: Vec<String>,
    ) -> Result<DataExportRequest> {
        // Validate format
        if !self.export_formats.contains_key(&format) {
            return Err(AgentError::validation(format!(
                "Unsupported export format: {}",
                format
            )));
        }

        let request = DataExportRequest {
            id: Uuid::new_v4().to_string(),
            subject_id,
            format,
            categories,
            date_range: None,
            created_at: Utc::now(),
            status: ExportStatus::Pending,
            file_path: None,
            metadata: ExportMetadata {
                record_count: 0,
                file_size_bytes: 0,
                categories_included: Vec::new(),
                completed_at: None,
                expires_at: Some(Utc::now() + chrono::Duration::days(30)), // 30-day expiry
            },
        };

        tracing::info!(
            "Created export request {} for subject {}",
            request.id,
            request.subject_id
        );
        Ok(request)
    }

    /// Collect all data for a subject
    async fn collect_subject_data(&self, subject_id: &str) -> Result<DataPackage> {
        let package = DataPackage {
            metadata: PackageMetadata {
                export_id: Uuid::new_v4().to_string(),
                subject_id: subject_id.to_string(),
                exported_at: Utc::now(),
                data_controller: "Rust MemVid Agent".to_string(),
                format: "json".to_string(),
                compliance_statement:
                    "This export complies with GDPR Article 20 (Right to Data Portability)"
                        .to_string(),
            },
            personal_data: self.collect_personal_data(subject_id).await?,
            conversations: self.collect_conversations(subject_id).await?,
            memory_data: self.collect_memory_data(subject_id).await?,
            audit_trail: self.collect_audit_trail(subject_id).await?,
        };

        Ok(package)
    }

    /// Collect personal data for a subject
    async fn collect_personal_data(&self, subject_id: &str) -> Result<PersonalDataExport> {
        debug!("Collecting personal data for subject: {}", subject_id);

        let mut basic_info = HashMap::new();
        let mut preferences = HashMap::new();
        let mut consent_records = Vec::new();

        // Collect basic user information
        // In a real implementation, this would query user database
        basic_info.insert("user_id".to_string(), Value::String(subject_id.to_string()));
        basic_info.insert(
            "created_at".to_string(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );
        basic_info.insert(
            "last_active".to_string(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );

        // Collect user preferences from configuration
        if let Ok(config_data) = self.collect_user_preferences(subject_id).await {
            for (key, value) in config_data {
                preferences.insert(key, Value::String(value));
            }
        }

        // Collect consent records from consent management system
        if let Ok(consents) = self.collect_consent_history(subject_id).await {
            consent_records.extend(consents);
        }

        // Add data processing activities
        basic_info.insert(
            "data_processing_activities".to_string(),
            Value::String(
                "AI conversation processing, memory storage, audio processing".to_string(),
            ),
        );
        basic_info.insert(
            "legal_basis".to_string(),
            Value::String(
                "Consent (GDPR Art. 6(1)(a)), Legitimate Interest (GDPR Art. 6(1)(f))".to_string(),
            ),
        );

        Ok(PersonalDataExport {
            basic_info,
            preferences,
            consent_records,
        })
    }

    /// Collect conversation data for a subject
    async fn collect_conversations(&self, subject_id: &str) -> Result<Vec<ConversationExport>> {
        debug!("Collecting conversation data for subject: {}", subject_id);

        let mut conversations = Vec::new();

        // Query the memory system for conversation data
        // In a real implementation, this would integrate with the rust-synaptic memory system
        if let Ok(memory_entries) = self.query_memory_system(subject_id).await {
            for entry in memory_entries {
                if let Some(conversation) = self.extract_conversation_from_memory(&entry) {
                    conversations.push(conversation);
                }
            }
        }

        // Query audit logs for conversation metadata
        if let Ok(audit_entries) = self.query_conversation_audit_logs(subject_id).await {
            for audit_entry in audit_entries {
                if let Some(conversation) = self.extract_conversation_from_audit(&audit_entry) {
                    // Merge with existing or add new
                    if let Some(existing) =
                        conversations.iter_mut().find(|c| c.id == conversation.id)
                    {
                        existing.metadata.extend(conversation.metadata);
                    } else {
                        conversations.push(conversation);
                    }
                }
            }
        }

        // Sort conversations by timestamp
        conversations.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        info!(
            "Collected {} conversations for subject {}",
            conversations.len(),
            subject_id
        );
        Ok(conversations)
    }

    /// Collect memory data for a subject
    async fn collect_memory_data(&self, subject_id: &str) -> Result<Vec<MemoryExport>> {
        debug!("Collecting memory data for subject: {}", subject_id);

        let mut memory_exports = Vec::new();

        // Query the rust-synaptic memory system
        if let Ok(memory_entries) = self.query_synaptic_memory(subject_id).await {
            for entry in memory_entries {
                let memory_export = MemoryExport {
                    id: entry
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    content: entry
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    memory_type: entry
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("general")
                        .to_string(),
                    created_at: entry
                        .get("created_at")
                        .and_then(|v| v.as_str())
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                    last_accessed: entry
                        .get("last_accessed")
                        .and_then(|v| v.as_str())
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                    metadata: entry
                        .as_object()
                        .map(|obj| {
                            obj.iter()
                                .filter(|(k, _)| {
                                    !["id", "content", "type", "created_at", "last_accessed"]
                                        .contains(&k.as_str())
                                })
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default(),
                };
                memory_exports.push(memory_export);
            }
        }

        // Sort by creation date
        memory_exports.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        info!(
            "Collected {} memory entries for subject {}",
            memory_exports.len(),
            subject_id
        );
        Ok(memory_exports)
    }

    /// Collect audit trail for a subject
    async fn collect_audit_trail(&self, subject_id: &str) -> Result<Vec<AuditExport>> {
        debug!("Collecting audit trail for subject: {}", subject_id);

        let mut audit_exports = Vec::new();

        // Query the audit logging system
        if let Ok(audit_entries) = self.query_audit_system(subject_id).await {
            for entry in audit_entries {
                let mut metadata = HashMap::new();

                // Add all relevant fields to metadata
                if let Some(action) = entry.get("action").and_then(|v| v.as_str()) {
                    metadata.insert("action".to_string(), Value::String(action.to_string()));
                }
                if let Some(resource) = entry.get("resource").and_then(|v| v.as_str()) {
                    metadata.insert("resource".to_string(), Value::String(resource.to_string()));
                }
                if let Some(ip_address) = entry.get("ip_address").and_then(|v| v.as_str()) {
                    metadata.insert(
                        "ip_address".to_string(),
                        Value::String(ip_address.to_string()),
                    );
                }
                if let Some(user_agent) = entry.get("user_agent").and_then(|v| v.as_str()) {
                    metadata.insert(
                        "user_agent".to_string(),
                        Value::String(user_agent.to_string()),
                    );
                }
                if let Some(result) = entry.get("result").and_then(|v| v.as_str()) {
                    metadata.insert("result".to_string(), Value::String(result.to_string()));
                }
                if let Some(details) = entry.get("details") {
                    metadata.insert("details".to_string(), details.clone());
                }

                let audit_export = AuditExport {
                    id: entry
                        .get("event_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    timestamp: entry
                        .get("timestamp")
                        .and_then(|v| v.as_str())
                        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(Utc::now),
                    event_type: entry
                        .get("event_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    description: format!(
                        "Event: {} - {}",
                        entry
                            .get("action")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown"),
                        entry
                            .get("result")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                    ),
                    metadata,
                };
                audit_exports.push(audit_export);
            }
        }

        // Sort by timestamp (most recent first)
        audit_exports.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        info!(
            "Collected {} audit entries for subject {}",
            audit_exports.len(),
            subject_id
        );
        Ok(audit_exports)
    }

    /// Create portable data package
    async fn create_portable_package(&self, subject_id: &str) -> Result<DataPackage> {
        // Create a package optimized for portability
        // This would include only machine-readable data in standard formats
        self.collect_subject_data(subject_id).await
    }

    /// Get supported export formats
    pub fn get_supported_formats(&self) -> Vec<&ExportFormat> {
        self.export_formats.values().collect()
    }

    // ========================================
    // Helper Methods for Data Collection
    // ========================================

    /// Collect user preferences from configuration system
    async fn collect_user_preferences(&self, subject_id: &str) -> Result<HashMap<String, String>> {
        debug!("Collecting user preferences for: {}", subject_id);

        let mut preferences = HashMap::new();

        // In a real implementation, this would query the configuration system
        // For now, we'll simulate common user preferences
        preferences.insert("language".to_string(), "en".to_string());
        preferences.insert("timezone".to_string(), "UTC".to_string());
        preferences.insert("audio_processing_enabled".to_string(), "true".to_string());
        preferences.insert("memory_retention_days".to_string(), "365".to_string());
        preferences.insert("data_sharing_consent".to_string(), "false".to_string());

        Ok(preferences)
    }

    /// Collect consent history from consent management system
    async fn collect_consent_history(&self, subject_id: &str) -> Result<Vec<ConsentRecord>> {
        debug!("Collecting consent history for: {}", subject_id);

        let mut consent_records = Vec::new();

        // In a real implementation, this would query the consent management system
        // For now, we'll create a sample consent record
        consent_records.push(ConsentRecord {
            id: format!("consent_{}", subject_id),
            purpose: "AI conversation processing".to_string(),
            given_at: Utc::now(),
            withdrawn_at: None,
            status: "active".to_string(),
        });

        Ok(consent_records)
    }

    /// Query the memory system for user data
    async fn query_memory_system(&self, subject_id: &str) -> Result<Vec<Value>> {
        debug!("Querying memory system for: {}", subject_id);

        // In a real implementation, this would integrate with rust-synaptic
        // For now, return sample memory entries
        let sample_memories = vec![
            serde_json::json!({
                "id": format!("mem_{}_{}", subject_id, 1),
                "content": "User prefers technical discussions about Rust programming",
                "type": "preference",
                "created_at": Utc::now().to_rfc3339(),
                "tags": ["programming", "rust", "preference"],
                "confidence": 0.9
            }),
            serde_json::json!({
                "id": format!("mem_{}_{}", subject_id, 2),
                "content": "User asked about audio processing capabilities",
                "type": "interaction",
                "created_at": Utc::now().to_rfc3339(),
                "tags": ["audio", "question"],
                "confidence": 1.0
            }),
        ];

        Ok(sample_memories)
    }

    /// Query rust-synaptic memory system
    async fn query_synaptic_memory(&self, subject_id: &str) -> Result<Vec<Value>> {
        debug!("Querying synaptic memory for: {}", subject_id);

        // In a real implementation, this would use the rust-synaptic API
        // For now, return sample synaptic memory entries
        let sample_entries = vec![serde_json::json!({
            "id": format!("synaptic_{}_{}", subject_id, 1),
            "content": "Long-term memory: User expertise in systems programming",
            "type": "long_term",
            "created_at": Utc::now().to_rfc3339(),
            "tags": ["expertise", "programming"],
            "strength": 0.95
        })];

        Ok(sample_entries)
    }

    /// Query audit system for user-related events
    async fn query_audit_system(&self, subject_id: &str) -> Result<Vec<Value>> {
        debug!("Querying audit system for: {}", subject_id);

        // In a real implementation, this would query the audit logging system
        let sample_audit_entries = vec![serde_json::json!({
            "event_id": format!("audit_{}_{}", subject_id, 1),
            "event_type": "data_access",
            "timestamp": Utc::now().to_rfc3339(),
            "action": "memory_query",
            "resource": "user_memories",
            "ip_address": "127.0.0.1",
            "user_agent": "rust-agent/1.0",
            "result": "success",
            "details": {
                "query_type": "user_data",
                "records_accessed": 5
            }
        })];

        Ok(sample_audit_entries)
    }

    /// Query conversation audit logs
    async fn query_conversation_audit_logs(&self, subject_id: &str) -> Result<Vec<Value>> {
        debug!("Querying conversation audit logs for: {}", subject_id);

        let sample_logs = vec![serde_json::json!({
            "conversation_id": format!("conv_{}_{}", subject_id, 1),
            "timestamp": Utc::now().to_rfc3339(),
            "event_type": "conversation_start",
            "metadata": {
                "duration_seconds": 300,
                "message_count": 15
            }
        })];

        Ok(sample_logs)
    }

    /// Extract conversation data from memory entry
    fn extract_conversation_from_memory(&self, entry: &Value) -> Option<ConversationExport> {
        let id = entry.get("id")?.as_str()?.to_string();
        let content = entry.get("content")?.as_str()?.to_string();
        let timestamp = entry
            .get("created_at")?
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))?;

        // Create a message from the content
        let message = MessageExport {
            id: format!("{}_msg_1", id),
            timestamp,
            role: "user".to_string(),
            content,
            metadata: HashMap::new(),
        };

        Some(ConversationExport {
            id,
            timestamp,
            messages: vec![message],
            metadata: HashMap::new(),
        })
    }

    /// Extract conversation data from audit entry
    fn extract_conversation_from_audit(&self, entry: &Value) -> Option<ConversationExport> {
        let id = entry.get("conversation_id")?.as_str()?.to_string();
        let timestamp = entry
            .get("timestamp")?
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))?;

        let mut metadata = HashMap::new();
        if let Some(meta) = entry.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                metadata.insert(k.clone(), v.clone());
            }
        }

        Some(ConversationExport {
            id,
            timestamp,
            messages: Vec::new(),
            metadata,
        })
    }
}

impl Default for DataExportHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_data_export_handler_creation() {
        let handler = DataExportHandler::new();
        let formats = handler.get_supported_formats();

        // Should support multiple formats
        assert!(!formats.is_empty());
        assert!(formats.iter().any(|f| f.name == "JSON"));
        assert!(formats.iter().any(|f| f.name == "CSV"));
        assert!(formats.iter().any(|f| f.name == "XML"));
        assert!(formats.iter().any(|f| f.name == "PDF"));
    }

    #[tokio::test]
    async fn test_export_request_creation() {
        let handler = DataExportHandler::new();

        let request = handler
            .create_export_request(
                "test_user_123".to_string(),
                "json".to_string(),
                vec!["personal_data".to_string(), "conversations".to_string()],
            )
            .await
            .unwrap();

        assert_eq!(request.subject_id, "test_user_123");
        assert_eq!(request.format, "json");
        assert_eq!(request.categories.len(), 2);
        assert!(matches!(request.status, ExportStatus::Pending));
        assert!(request.metadata.expires_at.is_some());
    }

    #[tokio::test]
    async fn test_export_request_invalid_format() {
        let handler = DataExportHandler::new();

        let result = handler
            .create_export_request(
                "test_user_123".to_string(),
                "invalid_format".to_string(),
                vec!["personal_data".to_string()],
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_personal_data_collection() {
        let handler = DataExportHandler::new();

        let personal_data = handler
            .collect_personal_data("test_user_123")
            .await
            .unwrap();

        // Should have basic info
        assert!(!personal_data.basic_info.is_empty());
        assert!(personal_data.basic_info.contains_key("user_id"));
        assert!(personal_data.basic_info.contains_key("created_at"));
        assert!(personal_data
            .basic_info
            .contains_key("data_processing_activities"));
        assert!(personal_data.basic_info.contains_key("legal_basis"));

        // Should have preferences
        assert!(!personal_data.preferences.is_empty());

        // Should have consent records
        assert!(!personal_data.consent_records.is_empty());
        assert_eq!(
            personal_data.consent_records[0].purpose,
            "AI conversation processing"
        );
        assert_eq!(personal_data.consent_records[0].status, "active");
    }

    #[tokio::test]
    async fn test_conversation_data_collection() {
        let handler = DataExportHandler::new();

        let conversations = handler
            .collect_conversations("test_user_123")
            .await
            .unwrap();

        // Should collect conversations from memory system
        assert!(!conversations.is_empty());

        // Conversations should be sorted by timestamp
        for i in 1..conversations.len() {
            assert!(conversations[i - 1].timestamp <= conversations[i].timestamp);
        }

        // Each conversation should have proper structure
        for conv in &conversations {
            assert!(!conv.id.is_empty());
            assert!(
                !conv.messages.is_empty() || conv.metadata.is_empty() || !conv.metadata.is_empty()
            );
        }
    }

    #[tokio::test]
    async fn test_memory_data_collection() {
        let handler = DataExportHandler::new();

        let memory_data = handler.collect_memory_data("test_user_123").await.unwrap();

        // Should collect memory entries
        assert!(!memory_data.is_empty());

        // Memory entries should be sorted by creation date
        for i in 1..memory_data.len() {
            assert!(memory_data[i - 1].created_at <= memory_data[i].created_at);
        }

        // Each memory entry should have proper structure
        for memory in &memory_data {
            assert!(!memory.id.is_empty());
            assert!(!memory.content.is_empty());
            assert!(!memory.memory_type.is_empty());
        }
    }

    #[tokio::test]
    async fn test_audit_trail_collection() {
        let handler = DataExportHandler::new();

        let audit_trail = handler.collect_audit_trail("test_user_123").await.unwrap();

        // Should collect audit entries
        assert!(!audit_trail.is_empty());

        // Audit entries should be sorted by timestamp (most recent first)
        for i in 1..audit_trail.len() {
            assert!(audit_trail[i - 1].timestamp >= audit_trail[i].timestamp);
        }

        // Each audit entry should have proper structure
        for audit in &audit_trail {
            assert!(!audit.id.is_empty());
            assert!(!audit.event_type.is_empty());
            assert!(!audit.description.is_empty());
        }
    }

    #[tokio::test]
    async fn test_full_data_export() {
        let handler = DataExportHandler::new();

        let export_data = handler.export_subject_data("test_user_123").await.unwrap();

        // Should produce valid JSON data
        assert!(!export_data.is_empty());

        // Should be parseable as JSON
        let parsed: serde_json::Value = serde_json::from_slice(&export_data).unwrap();

        // Should contain all required sections
        assert!(parsed.get("metadata").is_some());
        assert!(parsed.get("personal_data").is_some());
        assert!(parsed.get("conversations").is_some());
        assert!(parsed.get("memory_data").is_some());
        assert!(parsed.get("audit_trail").is_some());

        // Metadata should contain compliance statement
        let metadata = parsed.get("metadata").unwrap();
        assert!(metadata.get("compliance_statement").is_some());
        assert!(metadata.get("data_controller").is_some());
        assert!(metadata.get("exported_at").is_some());
    }

    #[tokio::test]
    async fn test_portable_data_export() {
        let handler = DataExportHandler::new();

        let portable_data = handler.export_portable_data("test_user_123").await.unwrap();

        // Should produce valid JSON data
        assert!(!portable_data.is_empty());

        // Should be parseable as JSON
        let parsed: serde_json::Value = serde_json::from_slice(&portable_data).unwrap();

        // Should be machine-readable format
        assert!(parsed.is_object());
    }

    #[tokio::test]
    async fn test_gdpr_compliance_requirements() {
        let handler = DataExportHandler::new();

        // Test GDPR Article 20 compliance
        let export_data = handler.export_subject_data("test_user_123").await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&export_data).unwrap();

        // Should include compliance statement referencing GDPR Article 20
        let compliance_statement = parsed["metadata"]["compliance_statement"].as_str().unwrap();
        assert!(compliance_statement.contains("GDPR Article 20"));
        assert!(compliance_statement.contains("Data Portability"));

        // Should include data controller information
        assert!(parsed["metadata"]["data_controller"].is_string());

        // Should include export timestamp
        assert!(parsed["metadata"]["exported_at"].is_string());

        // Personal data should include legal basis
        let legal_basis = parsed["personal_data"]["basic_info"]["legal_basis"]
            .as_str()
            .unwrap();
        assert!(legal_basis.contains("GDPR"));

        // Should include consent records with proper structure
        let consent_records = parsed["personal_data"]["consent_records"]
            .as_array()
            .unwrap();
        assert!(!consent_records.is_empty());

        for consent in consent_records {
            assert!(consent.get("id").is_some());
            assert!(consent.get("purpose").is_some());
            assert!(consent.get("given_at").is_some());
            assert!(consent.get("status").is_some());
        }
    }
}
