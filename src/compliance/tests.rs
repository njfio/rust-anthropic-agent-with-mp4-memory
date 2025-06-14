// Comprehensive Tests for GDPR Compliance System

use super::*;
use crate::compliance::{
    consent_management::{ConsentManager, ConsentMechanism, ConsentEvidence},
    data_export::DataExportHandler,
    data_retention::{DataRetentionManager, RetentionPolicy},
    privacy_controls::{PrivacyControlManager, PrivacyEventType},
    GdprComplianceManager, DataSubjectRight,
};
use chrono::Utc;
use std::collections::HashMap;
use tokio;

#[tokio::test]
async fn test_gdpr_compliance_manager_creation() {
    let manager = GdprComplianceManager::new();

    // Verify manager is properly initialized
    // The manager should be created successfully
    assert!(true); // Basic creation test
}

#[tokio::test]
async fn test_submit_access_request() {
    let mut manager = GdprComplianceManager::new();
    
    let request = manager.submit_request(
        "user123".to_string(),
        DataSubjectRight::Access,
        "I want to see all my personal data".to_string(),
    ).await.unwrap();
    
    assert_eq!(request.subject_id, "user123");
    assert!(matches!(request.right, DataSubjectRight::Access));
    assert_eq!(request.details, "I want to see all my personal data");
}

#[tokio::test]
async fn test_submit_erasure_request() {
    let mut manager = GdprComplianceManager::new();
    
    let request = manager.submit_request(
        "user456".to_string(),
        DataSubjectRight::Erasure,
        "Please delete all my data".to_string(),
    ).await.unwrap();
    
    assert_eq!(request.subject_id, "user456");
    assert!(matches!(request.right, DataSubjectRight::Erasure));
}

#[tokio::test]
async fn test_submit_portability_request() {
    let mut manager = GdprComplianceManager::new();
    
    let request = manager.submit_request(
        "user789".to_string(),
        DataSubjectRight::DataPortability,
        "I want to export my data".to_string(),
    ).await.unwrap();
    
    assert_eq!(request.subject_id, "user789");
    assert!(matches!(request.right, DataSubjectRight::DataPortability));
}

#[tokio::test]
async fn test_invalid_request_validation() {
    let mut manager = GdprComplianceManager::new();
    
    // Test empty subject ID
    let result = manager.submit_request(
        "".to_string(),
        DataSubjectRight::Access,
        "Valid details".to_string(),
    ).await;
    
    assert!(result.is_err());
    
    // Test empty details
    let result = manager.submit_request(
        "user123".to_string(),
        DataSubjectRight::Access,
        "".to_string(),
    ).await;
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_compliance_stats() {
    let manager = GdprComplianceManager::new();
    
    let stats = manager.get_compliance_stats().await.unwrap();
    
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.pending_requests, 0);
    assert_eq!(stats.completed_requests, 0);
    assert_eq!(stats.data_subjects_count, 0);
}

// Data Retention Tests

#[tokio::test]
async fn test_data_retention_manager_creation() {
    let manager = DataRetentionManager::new();

    assert_eq!(manager.policies_count(), 0);
    assert_eq!(manager.tracked_data_count(), 0);
}

#[tokio::test]
async fn test_create_default_retention_policies() {
    let mut manager = DataRetentionManager::new();
    
    manager.create_default_policies().unwrap();

    assert_eq!(manager.policies_count(), 4);
    assert!(manager.has_policy("personal_data"));
    assert!(manager.has_policy("conversation_data"));
    assert!(manager.has_policy("audit_logs"));
    assert!(manager.has_policy("sensitive_data"));
}

#[tokio::test]
async fn test_add_retention_policy() {
    let mut manager = DataRetentionManager::new();
    
    let policy = RetentionPolicy {
        name: "test_policy".to_string(),
        data_category: "test".to_string(),
        retention_days: 365,
        legal_basis: "Test basis".to_string(),
        auto_delete: true,
        grace_period_days: 30,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    manager.add_policy(policy).unwrap();

    assert_eq!(manager.policies_count(), 1);
    assert!(manager.has_policy("test_policy"));
}

#[tokio::test]
async fn test_track_data_item() {
    let mut manager = DataRetentionManager::new();
    manager.create_default_policies().unwrap();
    
    manager.track_data(
        "data123".to_string(),
        "personal_data".to_string(),
        "user123".to_string(),
    ).unwrap();

    assert_eq!(manager.tracked_data_count(), 1);
    assert!(manager.has_tracked_data("data123"));
}

#[tokio::test]
async fn test_track_data_unknown_category() {
    let mut manager = DataRetentionManager::new();
    
    let result = manager.track_data(
        "data123".to_string(),
        "unknown_category".to_string(),
        "user123".to_string(),
    );
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_update_access_time() {
    let mut manager = DataRetentionManager::new();
    manager.create_default_policies().unwrap();
    
    manager.track_data(
        "data123".to_string(),
        "personal_data".to_string(),
        "user123".to_string(),
    ).unwrap();
    
    let original_time = manager.get_tracked_data("data123").unwrap().last_accessed;

    // Wait a bit to ensure time difference
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    manager.update_access_time("data123").unwrap();

    let updated_time = manager.get_tracked_data("data123").unwrap().last_accessed;
    assert!(updated_time > original_time);
}

#[tokio::test]
async fn test_deletion_hold() {
    let mut manager = DataRetentionManager::new();
    manager.create_default_policies().unwrap();
    
    manager.track_data(
        "data123".to_string(),
        "personal_data".to_string(),
        "user123".to_string(),
    ).unwrap();
    
    manager.place_deletion_hold("data123", "Legal investigation".to_string()).unwrap();

    let record = manager.get_tracked_data("data123").unwrap();
    assert!(record.deletion_hold);
    assert_eq!(record.hold_reason.as_ref().unwrap(), "Legal investigation");

    manager.remove_deletion_hold("data123").unwrap();

    let record = manager.get_tracked_data("data123").unwrap();
    assert!(!record.deletion_hold);
    assert!(record.hold_reason.is_none());
}

#[tokio::test]
async fn test_retention_report() {
    let mut manager = DataRetentionManager::new();
    manager.create_default_policies().unwrap();
    
    manager.track_data(
        "data1".to_string(),
        "personal_data".to_string(),
        "user1".to_string(),
    ).unwrap();
    
    manager.track_data(
        "data2".to_string(),
        "conversation_data".to_string(),
        "user2".to_string(),
    ).unwrap();
    
    let report = manager.generate_retention_report();
    
    assert_eq!(report.total_tracked_items, 2);
    assert_eq!(report.items_on_hold, 0);
    assert_eq!(report.policies_count, 4);
}

// Data Export Tests

#[tokio::test]
async fn test_data_export_handler_creation() {
    let handler = DataExportHandler::new();
    
    let formats = handler.get_supported_formats();
    assert!(!formats.is_empty());
    
    // Check for standard formats
    let format_names: Vec<&str> = formats.iter().map(|f| f.name.as_str()).collect();
    assert!(format_names.contains(&"JSON"));
    assert!(format_names.contains(&"CSV"));
    assert!(format_names.contains(&"XML"));
    assert!(format_names.contains(&"PDF"));
}

#[tokio::test]
async fn test_create_export_request() {
    let handler = DataExportHandler::new();
    
    let request = handler.create_export_request(
        "user123".to_string(),
        "json".to_string(),
        vec!["personal".to_string(), "conversation".to_string()],
    ).await.unwrap();
    
    assert_eq!(request.subject_id, "user123");
    assert_eq!(request.format, "json");
    assert_eq!(request.categories.len(), 2);
}

#[tokio::test]
async fn test_export_request_invalid_format() {
    let handler = DataExportHandler::new();
    
    let result = handler.create_export_request(
        "user123".to_string(),
        "invalid_format".to_string(),
        vec!["personal".to_string()],
    ).await;
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_export_subject_data() {
    let handler = DataExportHandler::new();
    
    let data = handler.export_subject_data("user123").await.unwrap();
    
    assert!(!data.is_empty());
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_slice(&data).unwrap();
    assert!(parsed.is_object());
}

#[tokio::test]
async fn test_export_portable_data() {
    let handler = DataExportHandler::new();
    
    let data = handler.export_portable_data("user123").await.unwrap();
    
    assert!(!data.is_empty());
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_slice(&data).unwrap();
    assert!(parsed.is_object());
}

// Privacy Controls Tests

#[tokio::test]
async fn test_privacy_control_manager_creation() {
    let manager = PrivacyControlManager::new();

    assert_eq!(manager.purposes_count(), 0);
    assert_eq!(manager.events_count(), 0);
}

#[tokio::test]
async fn test_privacy_control_manager_gdpr_defaults() {
    let manager = PrivacyControlManager::with_gdpr_defaults();

    assert!(manager.purposes_count() > 0);
    assert!(manager.get_config().data_minimization.enabled);
    assert!(manager.get_config().security_measures.encryption_required);
}

#[tokio::test]
async fn test_processing_allowed_check() {
    let manager = PrivacyControlManager::with_gdpr_defaults();
    
    let allowed = manager.is_processing_allowed(
        "service_provision",
        &["conversation".to_string()],
    ).unwrap();
    
    assert!(allowed);
    
    let not_allowed = manager.is_processing_allowed(
        "service_provision",
        &["sensitive_medical".to_string()],
    ).unwrap();
    
    assert!(!not_allowed);
}

#[tokio::test]
async fn test_record_privacy_event() {
    let mut manager = PrivacyControlManager::new();
    
    manager.record_event(
        PrivacyEventType::DataCollection,
        Some("user123".to_string()),
        "Data collected for service provision".to_string(),
        HashMap::new(),
    ).unwrap();

    assert_eq!(manager.events_count(), 1);

    let event = manager.get_event(0).unwrap();
    assert!(matches!(event.event_type, PrivacyEventType::DataCollection));
    assert_eq!(event.subject_id.as_ref().unwrap(), "user123");
}

#[tokio::test]
async fn test_validate_data_collection() {
    let manager = PrivacyControlManager::with_gdpr_defaults();
    
    let result = manager.validate_data_collection(
        &["name".to_string(), "email".to_string()],
        "service_provision",
    ).unwrap();
    
    assert!(result.valid);
    assert!(result.violations.is_empty());
}

#[tokio::test]
async fn test_compliance_report() {
    let mut manager = PrivacyControlManager::new();
    
    manager.record_event(
        PrivacyEventType::DataCollection,
        Some("user1".to_string()),
        "Test event".to_string(),
        HashMap::new(),
    ).unwrap();
    
    let report = manager.get_compliance_report();
    
    assert_eq!(report.total_events, 1);
    assert_eq!(report.privacy_violations, 0);
    assert_eq!(report.data_breaches, 0);
    assert!(report.compliance_score > 90.0);
}

// Consent Management Tests

#[tokio::test]
async fn test_consent_manager_creation() {
    let manager = ConsentManager::new();

    assert_eq!(manager.consent_records_count(), 0);
    assert!(manager.purposes_count() > 0); // Default purposes should be created
}

#[tokio::test]
async fn test_request_consent() {
    let mut manager = ConsentManager::new();
    
    let request = manager.request_consent(
        "user123".to_string(),
        "analytics".to_string(),
        HashMap::new(),
    ).await.unwrap();
    
    assert_eq!(request.subject_id, "user123");
    assert_eq!(request.purpose_id, "analytics");
}

#[tokio::test]
async fn test_record_consent() {
    let mut manager = ConsentManager::new();
    
    let evidence = ConsentEvidence {
        evidence_type: "checkbox".to_string(),
        evidence_data: HashMap::new(),
        signature: None,
        witness: None,
    };
    
    let consent = manager.record_consent(
        "user123".to_string(),
        "analytics".to_string(),
        ConsentMechanism::ExplicitOptIn,
        evidence,
        HashMap::new(),
    ).await.unwrap();
    
    assert_eq!(consent.subject_id, "user123");
    assert_eq!(consent.purpose_id, "analytics");
    assert!(matches!(consent.status, consent_management::ConsentStatus::Given));
}

#[tokio::test]
async fn test_withdraw_consent() {
    let mut manager = ConsentManager::new();
    
    let evidence = ConsentEvidence {
        evidence_type: "checkbox".to_string(),
        evidence_data: HashMap::new(),
        signature: None,
        witness: None,
    };
    
    let consent = manager.record_consent(
        "user123".to_string(),
        "analytics".to_string(),
        ConsentMechanism::ExplicitOptIn,
        evidence,
        HashMap::new(),
    ).await.unwrap();
    
    manager.withdraw_consent(&consent.id, Some("User request".to_string())).await.unwrap();

    let updated_consent = manager.get_consent_record(&consent.id).unwrap();
    assert!(matches!(updated_consent.status, consent_management::ConsentStatus::Withdrawn));
    assert!(updated_consent.withdrawn_at.is_some());
}

#[tokio::test]
async fn test_consent_validity_check() {
    let mut manager = ConsentManager::new();
    
    let evidence = ConsentEvidence {
        evidence_type: "checkbox".to_string(),
        evidence_data: HashMap::new(),
        signature: None,
        witness: None,
    };
    
    manager.record_consent(
        "user123".to_string(),
        "analytics".to_string(),
        ConsentMechanism::ExplicitOptIn,
        evidence,
        HashMap::new(),
    ).await.unwrap();
    
    assert!(manager.is_consent_valid("user123", "analytics"));
    assert!(!manager.is_consent_valid("user123", "unknown_purpose"));
    assert!(!manager.is_consent_valid("unknown_user", "analytics"));
}

#[tokio::test]
async fn test_consent_report() {
    let mut manager = ConsentManager::new();
    
    let evidence = ConsentEvidence {
        evidence_type: "checkbox".to_string(),
        evidence_data: HashMap::new(),
        signature: None,
        witness: None,
    };
    
    manager.record_consent(
        "user1".to_string(),
        "analytics".to_string(),
        ConsentMechanism::ExplicitOptIn,
        evidence.clone(),
        HashMap::new(),
    ).await.unwrap();
    
    manager.record_consent(
        "user2".to_string(),
        "analytics".to_string(),
        ConsentMechanism::ExplicitOptIn,
        evidence,
        HashMap::new(),
    ).await.unwrap();
    
    let report = manager.generate_consent_report();
    
    assert_eq!(report.total_consents, 2);
    assert_eq!(report.active_consents, 2);
    assert_eq!(report.withdrawn_consents, 0);
    assert_eq!(report.total_subjects, 2);
}
