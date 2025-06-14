// Privacy Controls Implementation for GDPR Compliance

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::utils::error::{AgentError, Result};

/// Privacy configuration for the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Data minimization settings
    pub data_minimization: DataMinimizationConfig,
    /// Purpose limitation settings
    pub purpose_limitation: PurposeLimitationConfig,
    /// Storage limitation settings
    pub storage_limitation: StorageLimitationConfig,
    /// Accuracy requirements
    pub accuracy_requirements: AccuracyConfig,
    /// Security measures
    pub security_measures: SecurityConfig,
    /// Accountability measures
    pub accountability: AccountabilityConfig,
}

/// Data minimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMinimizationConfig {
    /// Whether to enable automatic data minimization
    pub enabled: bool,
    /// Fields to exclude from collection
    pub excluded_fields: Vec<String>,
    /// Maximum data collection scope
    pub max_collection_scope: String,
    /// Automatic anonymization threshold (days)
    pub anonymization_threshold_days: i64,
}

/// Purpose limitation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeLimitationConfig {
    /// Allowed processing purposes
    pub allowed_purposes: Vec<ProcessingPurpose>,
    /// Whether to enforce strict purpose checking
    pub strict_enforcement: bool,
    /// Purpose compatibility matrix
    pub compatibility_matrix: HashMap<String, Vec<String>>,
}

/// Storage limitation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLimitationConfig {
    /// Default retention period (days)
    pub default_retention_days: i64,
    /// Automatic deletion enabled
    pub auto_deletion_enabled: bool,
    /// Grace period before deletion (days)
    pub deletion_grace_period_days: i64,
    /// Archive before deletion
    pub archive_before_deletion: bool,
}

/// Accuracy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyConfig {
    /// Enable data quality checks
    pub quality_checks_enabled: bool,
    /// Automatic correction enabled
    pub auto_correction_enabled: bool,
    /// Data validation rules
    pub validation_rules: Vec<ValidationRule>,
    /// Update notification requirements
    pub update_notifications: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Encryption requirements
    pub encryption_required: bool,
    /// Access control enabled
    pub access_control_enabled: bool,
    /// Audit logging enabled
    pub audit_logging_enabled: bool,
    /// Data breach notification settings
    pub breach_notification: BreachNotificationConfig,
}

/// Accountability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountabilityConfig {
    /// Data protection impact assessment required
    pub dpia_required: bool,
    /// Privacy by design enabled
    pub privacy_by_design: bool,
    /// Regular compliance audits
    pub compliance_audits_enabled: bool,
    /// Documentation requirements
    pub documentation_required: bool,
}

/// Processing purpose definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPurpose {
    /// Purpose identifier
    pub id: String,
    /// Purpose name
    pub name: String,
    /// Purpose description
    pub description: String,
    /// Legal basis for processing
    pub legal_basis: String,
    /// Data categories involved
    pub data_categories: Vec<String>,
    /// Retention period for this purpose
    pub retention_days: i64,
    /// Whether consent is required
    pub requires_consent: bool,
}

/// Data validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Field to validate
    pub field: String,
    /// Validation type
    pub validation_type: ValidationType,
    /// Rule parameters
    pub parameters: HashMap<String, String>,
    /// Error message for validation failure
    pub error_message: String,
}

/// Types of validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    /// Required field validation
    Required,
    /// Format validation (regex)
    Format,
    /// Range validation
    Range,
    /// Length validation
    Length,
    /// Custom validation
    Custom,
}

/// Data breach notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachNotificationConfig {
    /// Enable automatic breach detection
    pub auto_detection_enabled: bool,
    /// Notification timeline (hours)
    pub notification_timeline_hours: u32,
    /// Authorities to notify
    pub authorities: Vec<String>,
    /// Notification templates
    pub templates: HashMap<String, String>,
}

/// Privacy control manager
#[derive(Debug, Clone)]
pub struct PrivacyControlManager {
    /// Current privacy configuration
    config: PrivacyConfig,
    /// Active processing purposes
    purposes: HashMap<String, ProcessingPurpose>,
    /// Privacy events log
    events: Vec<PrivacyEvent>,
}

/// Privacy event for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyEvent {
    /// Event ID
    pub id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: PrivacyEventType,
    /// Subject ID (if applicable)
    pub subject_id: Option<String>,
    /// Event description
    pub description: String,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Types of privacy events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyEventType {
    /// Data collection event
    DataCollection,
    /// Data processing event
    DataProcessing,
    /// Data sharing event
    DataSharing,
    /// Data deletion event
    DataDeletion,
    /// Consent given
    ConsentGiven,
    /// Consent withdrawn
    ConsentWithdrawn,
    /// Privacy violation detected
    PrivacyViolation,
    /// Data breach detected
    DataBreach,
}

impl PrivacyControlManager {
    /// Create a new privacy control manager
    pub fn new() -> Self {
        Self {
            config: PrivacyConfig::default(),
            purposes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Initialize with default GDPR-compliant configuration
    pub fn with_gdpr_defaults() -> Self {
        let mut manager = Self::new();
        manager.config = PrivacyConfig::gdpr_compliant();
        manager.initialize_default_purposes();
        manager
    }

    /// Check if data processing is allowed for a purpose
    pub fn is_processing_allowed(
        &self,
        purpose_id: &str,
        data_categories: &[String],
    ) -> Result<bool> {
        let purpose = self.purposes.get(purpose_id)
            .ok_or_else(|| AgentError::validation(
                format!("Unknown processing purpose: {}", purpose_id)
            ))?;

        // Check if all data categories are allowed for this purpose
        for category in data_categories {
            if !purpose.data_categories.contains(category) {
                tracing::warn!(
                    "Data category '{}' not allowed for purpose '{}'",
                    category,
                    purpose_id
                );
                return Ok(false);
            }
        }

        // Check purpose limitation compatibility
        if self.config.purpose_limitation.strict_enforcement {
            // Additional compatibility checks would go here
        }

        Ok(true)
    }

    /// Record a privacy event
    pub fn record_event(
        &mut self,
        event_type: PrivacyEventType,
        subject_id: Option<String>,
        description: String,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let event = PrivacyEvent {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: event_type.clone(),
            subject_id,
            description,
            metadata,
        };

        self.events.push(event);

        // Log for audit trail
        tracing::info!("Privacy event recorded: {:?}", event_type);
        
        Ok(())
    }

    /// Validate data collection against privacy rules
    pub fn validate_data_collection(
        &self,
        data_fields: &[String],
        purpose_id: &str,
    ) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            violations: Vec::new(),
            warnings: Vec::new(),
        };

        // Check data minimization
        if self.config.data_minimization.enabled {
            for field in data_fields {
                if self.config.data_minimization.excluded_fields.contains(field) {
                    result.valid = false;
                    result.violations.push(format!(
                        "Field '{}' is excluded by data minimization policy",
                        field
                    ));
                }
            }
        }

        // Check purpose limitation
        if let Some(purpose) = self.purposes.get(purpose_id) {
            // Validate that data collection aligns with stated purpose
            if purpose.requires_consent {
                result.warnings.push(
                    "This purpose requires explicit consent".to_string()
                );
            }
        } else {
            result.valid = false;
            result.violations.push(format!(
                "Unknown processing purpose: {}",
                purpose_id
            ));
        }

        Ok(result)
    }

    /// Initialize default processing purposes
    fn initialize_default_purposes(&mut self) {
        // Service provision purpose
        let service_purpose = ProcessingPurpose {
            id: "service_provision".to_string(),
            name: "Service Provision".to_string(),
            description: "Providing AI agent services to users".to_string(),
            legal_basis: "Performance of contract".to_string(),
            data_categories: vec![
                "conversation".to_string(),
                "preferences".to_string(),
            ],
            retention_days: 3 * 365, // 3 years
            requires_consent: false,
        };

        // Service improvement purpose
        let improvement_purpose = ProcessingPurpose {
            id: "service_improvement".to_string(),
            name: "Service Improvement".to_string(),
            description: "Improving AI agent capabilities and user experience".to_string(),
            legal_basis: "Legitimate interests".to_string(),
            data_categories: vec![
                "conversation".to_string(),
                "usage_analytics".to_string(),
            ],
            retention_days: 2 * 365, // 2 years
            requires_consent: true,
        };

        // Legal compliance purpose
        let compliance_purpose = ProcessingPurpose {
            id: "legal_compliance".to_string(),
            name: "Legal Compliance".to_string(),
            description: "Meeting legal and regulatory requirements".to_string(),
            legal_basis: "Legal obligation".to_string(),
            data_categories: vec![
                "audit_logs".to_string(),
                "personal".to_string(),
            ],
            retention_days: 7 * 365, // 7 years
            requires_consent: false,
        };

        self.purposes.insert(service_purpose.id.clone(), service_purpose);
        self.purposes.insert(improvement_purpose.id.clone(), improvement_purpose);
        self.purposes.insert(compliance_purpose.id.clone(), compliance_purpose);
    }

    /// Get privacy compliance report
    pub fn get_compliance_report(&self) -> PrivacyComplianceReport {
        let total_events = self.events.len();
        let violations = self.events.iter()
            .filter(|e| matches!(e.event_type, PrivacyEventType::PrivacyViolation))
            .count();
        let breaches = self.events.iter()
            .filter(|e| matches!(e.event_type, PrivacyEventType::DataBreach))
            .count();

        PrivacyComplianceReport {
            total_events,
            privacy_violations: violations,
            data_breaches: breaches,
            active_purposes: self.purposes.len(),
            compliance_score: self.calculate_compliance_score(),
            generated_at: Utc::now(),
        }
    }

    /// Get purposes count (for testing)
    pub fn purposes_count(&self) -> usize {
        self.purposes.len()
    }

    /// Get events count (for testing)
    pub fn events_count(&self) -> usize {
        self.events.len()
    }

    /// Get event by index (for testing)
    pub fn get_event(&self, index: usize) -> Option<&PrivacyEvent> {
        self.events.get(index)
    }

    /// Get privacy config (for testing)
    pub fn get_config(&self) -> &PrivacyConfig {
        &self.config
    }

    /// Calculate compliance score (0-100)
    fn calculate_compliance_score(&self) -> f64 {
        // Simple scoring algorithm
        let base_score = 100.0;
        let violation_penalty = self.events.iter()
            .filter(|e| matches!(e.event_type, PrivacyEventType::PrivacyViolation))
            .count() as f64 * 10.0;
        let breach_penalty = self.events.iter()
            .filter(|e| matches!(e.event_type, PrivacyEventType::DataBreach))
            .count() as f64 * 25.0;

        (base_score - violation_penalty - breach_penalty).max(0.0)
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
}

/// Privacy compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyComplianceReport {
    pub total_events: usize,
    pub privacy_violations: usize,
    pub data_breaches: usize,
    pub active_purposes: usize,
    pub compliance_score: f64,
    pub generated_at: DateTime<Utc>,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self::gdpr_compliant()
    }
}

impl PrivacyConfig {
    /// Create GDPR-compliant default configuration
    pub fn gdpr_compliant() -> Self {
        Self {
            data_minimization: DataMinimizationConfig {
                enabled: true,
                excluded_fields: vec![
                    "ssn".to_string(),
                    "credit_card".to_string(),
                    "password".to_string(),
                ],
                max_collection_scope: "necessary".to_string(),
                anonymization_threshold_days: 365,
            },
            purpose_limitation: PurposeLimitationConfig {
                allowed_purposes: Vec::new(),
                strict_enforcement: true,
                compatibility_matrix: HashMap::new(),
            },
            storage_limitation: StorageLimitationConfig {
                default_retention_days: 3 * 365, // 3 years
                auto_deletion_enabled: true,
                deletion_grace_period_days: 30,
                archive_before_deletion: true,
            },
            accuracy_requirements: AccuracyConfig {
                quality_checks_enabled: true,
                auto_correction_enabled: false,
                validation_rules: Vec::new(),
                update_notifications: true,
            },
            security_measures: SecurityConfig {
                encryption_required: true,
                access_control_enabled: true,
                audit_logging_enabled: true,
                breach_notification: BreachNotificationConfig {
                    auto_detection_enabled: true,
                    notification_timeline_hours: 72,
                    authorities: vec!["DPA".to_string()],
                    templates: HashMap::new(),
                },
            },
            accountability: AccountabilityConfig {
                dpia_required: true,
                privacy_by_design: true,
                compliance_audits_enabled: true,
                documentation_required: true,
            },
        }
    }
}

impl Default for PrivacyControlManager {
    fn default() -> Self {
        Self::new()
    }
}
