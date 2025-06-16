// Data Retention Policy Implementation for GDPR Compliance

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::utils::error::{AgentError, Result};

/// Data retention policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy name/identifier
    pub name: String,
    /// Data category this policy applies to
    pub data_category: String,
    /// Retention period in days
    pub retention_days: i64,
    /// Legal basis for retention
    pub legal_basis: String,
    /// Whether data should be automatically deleted
    pub auto_delete: bool,
    /// Grace period before deletion (days)
    pub grace_period_days: i64,
    /// Policy creation timestamp
    pub created_at: DateTime<Utc>,
    /// Policy last updated timestamp
    pub updated_at: DateTime<Utc>,
}

/// Data retention manager
#[derive(Debug, Clone)]
pub struct DataRetentionManager {
    /// Active retention policies
    policies: HashMap<String, RetentionPolicy>,
    /// Data items with retention tracking
    tracked_data: HashMap<String, DataRetentionRecord>,
}

/// Individual data retention record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionRecord {
    /// Unique data identifier
    pub data_id: String,
    /// Data category
    pub category: String,
    /// Subject identifier
    pub subject_id: String,
    /// Data creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Scheduled deletion date
    pub deletion_date: DateTime<Utc>,
    /// Whether deletion is on hold
    pub deletion_hold: bool,
    /// Reason for deletion hold
    pub hold_reason: Option<String>,
}

/// Retention policy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionViolation {
    /// Data identifier
    pub data_id: String,
    /// Policy that was violated
    pub policy_name: String,
    /// Violation type
    pub violation_type: ViolationType,
    /// When violation was detected
    pub detected_at: DateTime<Utc>,
    /// Severity level
    pub severity: ViolationSeverity,
}

/// Types of retention violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Data retained beyond policy limit
    OverRetention,
    /// Data deleted before minimum retention
    UnderRetention,
    /// Missing retention policy
    NoPolicy,
    /// Policy conflict
    PolicyConflict,
}

/// Severity levels for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl DataRetentionManager {
    /// Create a new data retention manager
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            tracked_data: HashMap::new(),
        }
    }

    /// Add a retention policy
    pub fn add_policy(&mut self, policy: RetentionPolicy) -> Result<()> {
        // Validate policy
        self.validate_policy(&policy)?;

        tracing::info!("Adding retention policy: {}", policy.name);
        self.policies.insert(policy.name.clone(), policy);

        Ok(())
    }

    /// Create default GDPR-compliant retention policies
    pub fn create_default_policies(&mut self) -> Result<()> {
        // Personal data - 6 years retention (typical for business records)
        let personal_data_policy = RetentionPolicy {
            name: "personal_data".to_string(),
            data_category: "personal".to_string(),
            retention_days: 6 * 365, // 6 years
            legal_basis: "Legitimate business interests".to_string(),
            auto_delete: true,
            grace_period_days: 30,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Conversation data - 3 years retention
        let conversation_policy = RetentionPolicy {
            name: "conversation_data".to_string(),
            data_category: "conversation".to_string(),
            retention_days: 3 * 365, // 3 years
            legal_basis: "Service improvement and support".to_string(),
            auto_delete: true,
            grace_period_days: 30,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Audit logs - 7 years retention (compliance requirement)
        let audit_log_policy = RetentionPolicy {
            name: "audit_logs".to_string(),
            data_category: "audit".to_string(),
            retention_days: 7 * 365, // 7 years
            legal_basis: "Legal compliance requirement".to_string(),
            auto_delete: false, // Manual review required
            grace_period_days: 90,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Sensitive data - 1 year retention (minimize exposure)
        let sensitive_data_policy = RetentionPolicy {
            name: "sensitive_data".to_string(),
            data_category: "sensitive".to_string(),
            retention_days: 365, // 1 year
            legal_basis: "Explicit consent".to_string(),
            auto_delete: true,
            grace_period_days: 7, // Short grace period
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.add_policy(personal_data_policy)?;
        self.add_policy(conversation_policy)?;
        self.add_policy(audit_log_policy)?;
        self.add_policy(sensitive_data_policy)?;

        tracing::info!("Created {} default retention policies", self.policies.len());
        Ok(())
    }

    /// Track a new data item for retention
    pub fn track_data(
        &mut self,
        data_id: String,
        category: String,
        subject_id: String,
    ) -> Result<()> {
        // Find applicable policy
        let policy = self.policies.get(&category).ok_or_else(|| {
            AgentError::validation(format!(
                "No retention policy found for category: {}",
                category
            ))
        })?;

        let now = Utc::now();
        let deletion_date = now + Duration::days(policy.retention_days);

        let record = DataRetentionRecord {
            data_id: data_id.clone(),
            category,
            subject_id,
            created_at: now,
            last_accessed: now,
            deletion_date,
            deletion_hold: false,
            hold_reason: None,
        };

        self.tracked_data.insert(data_id, record);
        Ok(())
    }

    /// Update last access time for data
    pub fn update_access_time(&mut self, data_id: &str) -> Result<()> {
        if let Some(record) = self.tracked_data.get_mut(data_id) {
            record.last_accessed = Utc::now();
            tracing::debug!("Updated access time for data: {}", data_id);
        }
        Ok(())
    }

    /// Place a hold on data deletion
    pub fn place_deletion_hold(&mut self, data_id: &str, reason: String) -> Result<()> {
        if let Some(record) = self.tracked_data.get_mut(data_id) {
            record.deletion_hold = true;
            record.hold_reason = Some(reason.clone());
            tracing::info!("Placed deletion hold on data {}: {}", data_id, reason);
        } else {
            return Err(AgentError::validation(format!(
                "Data record not found: {}",
                data_id
            )));
        }
        Ok(())
    }

    /// Remove deletion hold
    pub fn remove_deletion_hold(&mut self, data_id: &str) -> Result<()> {
        if let Some(record) = self.tracked_data.get_mut(data_id) {
            record.deletion_hold = false;
            record.hold_reason = None;
            tracing::info!("Removed deletion hold on data: {}", data_id);
        } else {
            return Err(AgentError::validation(format!(
                "Data record not found: {}",
                data_id
            )));
        }
        Ok(())
    }

    /// Get data items due for deletion
    pub fn get_items_due_for_deletion(&self) -> Vec<&DataRetentionRecord> {
        let now = Utc::now();
        self.tracked_data
            .values()
            .filter(|record| !record.deletion_hold && record.deletion_date <= now)
            .collect()
    }

    /// Get data items approaching deletion
    pub fn get_items_approaching_deletion(&self, days_ahead: i64) -> Vec<&DataRetentionRecord> {
        let threshold = Utc::now() + Duration::days(days_ahead);
        self.tracked_data
            .values()
            .filter(|record| !record.deletion_hold && record.deletion_date <= threshold)
            .collect()
    }

    /// Check for retention policy violations
    pub fn check_violations(&self) -> Vec<RetentionViolation> {
        let mut violations = Vec::new();
        let now = Utc::now();

        for record in self.tracked_data.values() {
            // Check for over-retention
            if let Some(policy) = self.policies.get(&record.category) {
                let max_retention_date = record.created_at + Duration::days(policy.retention_days);
                if now > max_retention_date && !record.deletion_hold {
                    violations.push(RetentionViolation {
                        data_id: record.data_id.clone(),
                        policy_name: policy.name.clone(),
                        violation_type: ViolationType::OverRetention,
                        detected_at: now,
                        severity: ViolationSeverity::High,
                    });
                }
            } else {
                // No policy found
                violations.push(RetentionViolation {
                    data_id: record.data_id.clone(),
                    policy_name: "unknown".to_string(),
                    violation_type: ViolationType::NoPolicy,
                    detected_at: now,
                    severity: ViolationSeverity::Critical,
                });
            }
        }

        violations
    }

    /// Generate retention report
    pub fn generate_retention_report(&self) -> RetentionReport {
        let total_items = self.tracked_data.len();
        let items_on_hold = self
            .tracked_data
            .values()
            .filter(|r| r.deletion_hold)
            .count();
        let items_due_deletion = self.get_items_due_for_deletion().len();
        let violations = self.check_violations();

        RetentionReport {
            total_tracked_items: total_items,
            items_on_hold,
            items_due_for_deletion: items_due_deletion,
            policy_violations: violations.len(),
            policies_count: self.policies.len(),
            generated_at: Utc::now(),
        }
    }

    /// Get policies count (for testing)
    pub fn policies_count(&self) -> usize {
        self.policies.len()
    }

    /// Check if policy exists (for testing)
    pub fn has_policy(&self, name: &str) -> bool {
        self.policies.contains_key(name)
    }

    /// Get tracked data count (for testing)
    pub fn tracked_data_count(&self) -> usize {
        self.tracked_data.len()
    }

    /// Check if data is tracked (for testing)
    pub fn has_tracked_data(&self, data_id: &str) -> bool {
        self.tracked_data.contains_key(data_id)
    }

    /// Get tracked data record (for testing)
    pub fn get_tracked_data(&self, data_id: &str) -> Option<&DataRetentionRecord> {
        self.tracked_data.get(data_id)
    }

    /// Validate a retention policy
    fn validate_policy(&self, policy: &RetentionPolicy) -> Result<()> {
        if policy.name.is_empty() {
            return Err(AgentError::validation(
                "Policy name cannot be empty".to_string(),
            ));
        }

        if policy.retention_days <= 0 {
            return Err(AgentError::validation(
                "Retention days must be positive".to_string(),
            ));
        }

        if policy.grace_period_days < 0 {
            return Err(AgentError::validation(
                "Grace period cannot be negative".to_string(),
            ));
        }

        Ok(())
    }
}

/// Data retention report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReport {
    pub total_tracked_items: usize,
    pub items_on_hold: usize,
    pub items_due_for_deletion: usize,
    pub policy_violations: usize,
    pub policies_count: usize,
    pub generated_at: DateTime<Utc>,
}

impl Default for DataRetentionManager {
    fn default() -> Self {
        Self::new()
    }
}
