// Consent Management System for GDPR Compliance

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::utils::error::{AgentError, Result};

/// Consent management system
#[derive(Debug, Clone)]
pub struct ConsentManager {
    /// Active consent records
    consent_records: HashMap<String, ConsentRecord>,
    /// Consent purposes configuration
    purposes: HashMap<String, ConsentPurpose>,
    /// Consent history for audit trail
    consent_history: Vec<ConsentHistoryEntry>,
}

/// Individual consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique consent ID
    pub id: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Purpose of processing
    pub purpose_id: String,
    /// Consent status
    pub status: ConsentStatus,
    /// When consent was given
    pub given_at: DateTime<Utc>,
    /// When consent expires (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
    /// When consent was withdrawn (if applicable)
    pub withdrawn_at: Option<DateTime<Utc>>,
    /// Consent mechanism used
    pub mechanism: ConsentMechanism,
    /// Evidence of consent
    pub evidence: ConsentEvidence,
    /// Consent metadata
    pub metadata: HashMap<String, String>,
}

/// Consent status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStatus {
    /// Consent given and active
    Given,
    /// Consent withdrawn
    Withdrawn,
    /// Consent expired
    Expired,
    /// Consent pending (not yet given)
    Pending,
    /// Consent refused
    Refused,
}

/// Consent mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentMechanism {
    /// Explicit opt-in (checkbox, button, etc.)
    ExplicitOptIn,
    /// Implicit consent (continued use)
    ImplicitConsent,
    /// Verbal consent (recorded)
    VerbalConsent,
    /// Written consent (signed document)
    WrittenConsent,
    /// Digital signature
    DigitalSignature,
}

/// Evidence of consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentEvidence {
    /// Type of evidence
    pub evidence_type: String,
    /// Evidence data (e.g., IP address, timestamp, etc.)
    pub evidence_data: HashMap<String, String>,
    /// Digital signature (if applicable)
    pub signature: Option<String>,
    /// Witness information (if applicable)
    pub witness: Option<String>,
}

/// Consent purpose definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentPurpose {
    /// Purpose identifier
    pub id: String,
    /// Purpose name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Data categories involved
    pub data_categories: Vec<String>,
    /// Processing activities
    pub processing_activities: Vec<String>,
    /// Third parties involved
    pub third_parties: Vec<String>,
    /// Retention period
    pub retention_period: String,
    /// Whether consent is required
    pub consent_required: bool,
    /// Whether consent can be withdrawn
    pub withdrawable: bool,
    /// Legal basis if consent is not required
    pub alternative_legal_basis: Option<String>,
}

/// Consent history entry for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentHistoryEntry {
    /// Entry ID
    pub id: String,
    /// Consent record ID
    pub consent_id: String,
    /// Subject ID
    pub subject_id: String,
    /// Action performed
    pub action: ConsentAction,
    /// Timestamp of action
    pub timestamp: DateTime<Utc>,
    /// Previous status
    pub previous_status: Option<ConsentStatus>,
    /// New status
    pub new_status: ConsentStatus,
    /// Reason for change
    pub reason: Option<String>,
    /// User agent/context
    pub context: HashMap<String, String>,
}

/// Consent actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentAction {
    /// Consent given
    Given,
    /// Consent withdrawn
    Withdrawn,
    /// Consent renewed
    Renewed,
    /// Consent expired
    Expired,
    /// Consent refused
    Refused,
    /// Consent modified
    Modified,
}

/// Consent request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRequest {
    /// Request ID
    pub id: String,
    /// Subject ID
    pub subject_id: String,
    /// Purpose ID
    pub purpose_id: String,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Request context
    pub context: HashMap<String, String>,
    /// Response deadline
    pub deadline: Option<DateTime<Utc>>,
}

impl ConsentManager {
    /// Create a new consent manager
    pub fn new() -> Self {
        let mut manager = Self {
            consent_records: HashMap::new(),
            purposes: HashMap::new(),
            consent_history: Vec::new(),
        };
        
        manager.initialize_default_purposes();
        manager
    }

    /// Register a consent purpose
    pub fn register_purpose(&mut self, purpose: ConsentPurpose) -> Result<()> {
        self.validate_purpose(&purpose)?;
        
        tracing::info!("Registering consent purpose: {}", purpose.name);
        self.purposes.insert(purpose.id.clone(), purpose);
        
        Ok(())
    }

    /// Request consent from a data subject
    pub async fn request_consent(
        &mut self,
        subject_id: String,
        purpose_id: String,
        context: HashMap<String, String>,
    ) -> Result<ConsentRequest> {
        // Validate purpose exists
        if !self.purposes.contains_key(&purpose_id) {
            return Err(AgentError::validation(
                format!("Unknown consent purpose: {}", purpose_id)
            ));
        }

        let request = ConsentRequest {
            id: Uuid::new_v4().to_string(),
            subject_id,
            purpose_id,
            requested_at: Utc::now(),
            context,
            deadline: Some(Utc::now() + chrono::Duration::days(30)), // 30-day deadline
        };

        tracing::info!(
            "Consent requested for subject {} and purpose {}",
            request.subject_id,
            request.purpose_id
        );

        Ok(request)
    }

    /// Record consent given by a data subject
    pub async fn record_consent(
        &mut self,
        subject_id: String,
        purpose_id: String,
        mechanism: ConsentMechanism,
        evidence: ConsentEvidence,
        metadata: HashMap<String, String>,
    ) -> Result<ConsentRecord> {
        // Validate purpose
        let _purpose = self.purposes.get(&purpose_id)
            .ok_or_else(|| AgentError::validation(
                format!("Unknown consent purpose: {}", purpose_id)
            ))?;

        // Create consent record
        let consent_record = ConsentRecord {
            id: Uuid::new_v4().to_string(),
            subject_id: subject_id.clone(),
            purpose_id: purpose_id.clone(),
            status: ConsentStatus::Given,
            given_at: Utc::now(),
            expires_at: None, // Could be set based on purpose configuration
            withdrawn_at: None,
            mechanism,
            evidence,
            metadata,
        };

        // Record in history
        self.record_consent_history(
            consent_record.id.clone(),
            subject_id.clone(),
            ConsentAction::Given,
            None,
            ConsentStatus::Given,
            Some("Consent given by data subject".to_string()),
            HashMap::new(),
        );

        // Store consent record
        self.consent_records.insert(consent_record.id.clone(), consent_record.clone());

        tracing::info!(
            "Consent recorded for subject {} and purpose {}",
            subject_id,
            purpose_id
        );

        Ok(consent_record)
    }

    /// Withdraw consent
    pub async fn withdraw_consent(
        &mut self,
        consent_id: &str,
        reason: Option<String>,
    ) -> Result<()> {
        let (previous_status, subject_id) = {
            let consent_record = self.consent_records.get_mut(consent_id)
                .ok_or_else(|| AgentError::validation(
                    format!("Consent record not found: {}", consent_id)
                ))?;

            let previous_status = consent_record.status.clone();
            let subject_id = consent_record.subject_id.clone();
            consent_record.status = ConsentStatus::Withdrawn;
            consent_record.withdrawn_at = Some(Utc::now());

            (previous_status, subject_id)
        };

        // Record in history
        self.record_consent_history(
            consent_id.to_string(),
            subject_id.clone(),
            ConsentAction::Withdrawn,
            Some(previous_status),
            ConsentStatus::Withdrawn,
            reason,
            HashMap::new(),
        );

        tracing::info!(
            "Consent withdrawn for record {} by subject {}",
            consent_id,
            subject_id
        );

        Ok(())
    }

    /// Check if consent is valid for a purpose
    pub fn is_consent_valid(&self, subject_id: &str, purpose_id: &str) -> bool {
        // Find active consent for this subject and purpose
        for consent in self.consent_records.values() {
            if consent.subject_id == subject_id 
                && consent.purpose_id == purpose_id 
                && matches!(consent.status, ConsentStatus::Given) {
                
                // Check if consent has expired
                if let Some(expires_at) = consent.expires_at {
                    if Utc::now() > expires_at {
                        return false;
                    }
                }
                
                return true;
            }
        }
        
        false
    }

    /// Get all consents for a subject
    pub fn get_subject_consents(&self, subject_id: &str) -> Vec<&ConsentRecord> {
        self.consent_records
            .values()
            .filter(|consent| consent.subject_id == subject_id)
            .collect()
    }

    /// Get consent history for a subject
    pub fn get_consent_history(&self, subject_id: &str) -> Vec<&ConsentHistoryEntry> {
        self.consent_history
            .iter()
            .filter(|entry| entry.subject_id == subject_id)
            .collect()
    }

    /// Check for expired consents
    pub fn check_expired_consents(&mut self) -> Vec<String> {
        let now = Utc::now();
        let mut expired_ids = Vec::new();
        let mut history_entries = Vec::new();

        for (id, consent) in self.consent_records.iter_mut() {
            if let Some(expires_at) = consent.expires_at {
                if now > expires_at && matches!(consent.status, ConsentStatus::Given) {
                    consent.status = ConsentStatus::Expired;
                    expired_ids.push(id.clone());

                    // Prepare history entry for later recording
                    history_entries.push((
                        id.clone(),
                        consent.subject_id.clone(),
                        ConsentAction::Expired,
                        Some(ConsentStatus::Given),
                        ConsentStatus::Expired,
                        Some("Consent expired automatically".to_string()),
                        HashMap::new(),
                    ));
                }
            }
        }

        // Record history entries after the mutable iteration
        for (id, subject_id, action, prev_status, new_status, reason, context) in history_entries {
            self.record_consent_history(id, subject_id, action, prev_status, new_status, reason, context);
        }

        if !expired_ids.is_empty() {
            tracing::info!("Found {} expired consents", expired_ids.len());
        }

        expired_ids
    }

    /// Generate consent report
    pub fn generate_consent_report(&self) -> ConsentReport {
        let total_consents = self.consent_records.len();
        let active_consents = self.consent_records.values()
            .filter(|c| matches!(c.status, ConsentStatus::Given))
            .count();
        let withdrawn_consents = self.consent_records.values()
            .filter(|c| matches!(c.status, ConsentStatus::Withdrawn))
            .count();
        let expired_consents = self.consent_records.values()
            .filter(|c| matches!(c.status, ConsentStatus::Expired))
            .count();

        ConsentReport {
            total_consents,
            active_consents,
            withdrawn_consents,
            expired_consents,
            total_subjects: self.get_unique_subjects_count(),
            purposes_count: self.purposes.len(),
            generated_at: Utc::now(),
        }
    }

    /// Initialize default consent purposes
    fn initialize_default_purposes(&mut self) {
        let service_purpose = ConsentPurpose {
            id: "service_provision".to_string(),
            name: "Service Provision".to_string(),
            description: "Processing necessary to provide AI agent services".to_string(),
            data_categories: vec!["conversation".to_string(), "preferences".to_string()],
            processing_activities: vec!["conversation_processing".to_string()],
            third_parties: Vec::new(),
            retention_period: "3 years".to_string(),
            consent_required: false, // Contract performance
            withdrawable: false,
            alternative_legal_basis: Some("Performance of contract".to_string()),
        };

        let analytics_purpose = ConsentPurpose {
            id: "analytics".to_string(),
            name: "Analytics and Improvement".to_string(),
            description: "Analyzing usage patterns to improve our services".to_string(),
            data_categories: vec!["usage_data".to_string(), "performance_metrics".to_string()],
            processing_activities: vec!["analytics".to_string(), "service_improvement".to_string()],
            third_parties: Vec::new(),
            retention_period: "2 years".to_string(),
            consent_required: true,
            withdrawable: true,
            alternative_legal_basis: None,
        };

        let _ = self.register_purpose(service_purpose);
        let _ = self.register_purpose(analytics_purpose);
    }

    /// Validate a consent purpose
    fn validate_purpose(&self, purpose: &ConsentPurpose) -> Result<()> {
        if purpose.id.is_empty() {
            return Err(AgentError::validation("Purpose ID cannot be empty".to_string()));
        }

        if purpose.name.is_empty() {
            return Err(AgentError::validation("Purpose name cannot be empty".to_string()));
        }

        if purpose.description.is_empty() {
            return Err(AgentError::validation("Purpose description cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Record consent history entry
    fn record_consent_history(
        &mut self,
        consent_id: String,
        subject_id: String,
        action: ConsentAction,
        previous_status: Option<ConsentStatus>,
        new_status: ConsentStatus,
        reason: Option<String>,
        context: HashMap<String, String>,
    ) {
        let entry = ConsentHistoryEntry {
            id: Uuid::new_v4().to_string(),
            consent_id,
            subject_id,
            action,
            timestamp: Utc::now(),
            previous_status,
            new_status,
            reason,
            context,
        };

        self.consent_history.push(entry);
    }

    /// Get consent records count (for testing)
    pub fn consent_records_count(&self) -> usize {
        self.consent_records.len()
    }

    /// Get purposes count (for testing)
    pub fn purposes_count(&self) -> usize {
        self.purposes.len()
    }

    /// Get consent record by ID (for testing)
    pub fn get_consent_record(&self, consent_id: &str) -> Option<&ConsentRecord> {
        self.consent_records.get(consent_id)
    }

    /// Get count of unique subjects
    fn get_unique_subjects_count(&self) -> usize {
        let mut subjects = std::collections::HashSet::new();
        for consent in self.consent_records.values() {
            subjects.insert(&consent.subject_id);
        }
        subjects.len()
    }
}

/// Consent report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentReport {
    pub total_consents: usize,
    pub active_consents: usize,
    pub withdrawn_consents: usize,
    pub expired_consents: usize,
    pub total_subjects: usize,
    pub purposes_count: usize,
    pub generated_at: DateTime<Utc>,
}

impl Default for ConsentManager {
    fn default() -> Self {
        Self::new()
    }
}
