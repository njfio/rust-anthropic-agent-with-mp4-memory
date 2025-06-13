use super::audit::*;
use super::authentication::*;
use super::authorization::{AuthorizationService, RbacAuthorizationService, Role};
use super::encryption::*;
use super::middleware::*;
use super::policy::{self, PolicyEngine, SecurityPolicy, SimplePolicyEngine};
use super::rate_limiting::*;
use super::session::*;
use super::*;
use std::collections::HashMap;

/// Create test security configuration
fn create_test_security_config() -> SecurityConfig {
    SecurityConfig {
        jwt: JwtConfig {
            secret: "test_secret_key_for_testing_only".to_string(),
            expiration_seconds: 3600,
            refresh_expiration_seconds: 86400,
            issuer: "test_issuer".to_string(),
            audience: "test_audience".to_string(),
            algorithm: JwtAlgorithm::HS256,
        },
        session: SessionConfig {
            timeout_seconds: 1800,
            max_concurrent_sessions: 3,
            persistent_sessions: false,
            storage_backend: SessionStorageBackend::Memory,
        },
        encryption: EncryptionConfig {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_derivation: KeyDerivationConfig {
                algorithm: KeyDerivationAlgorithm::Pbkdf2Sha256,
                iterations: 1000, // Lower for testing
                memory_cost: 1024,
                parallelism: 1,
                salt_length: 16,
            },
            encrypt_at_rest: true,
            encrypt_in_transit: true,
        },
        audit: AuditConfig {
            enabled: true,
            log_level: AuditLogLevel::All,
            storage_backend: AuditStorageBackend::File,
            retention_days: 30,
            encrypt_logs: false, // Disabled for testing
        },
        rate_limiting: RateLimitConfig {
            enabled: true,
            max_requests_per_minute: 100,
            max_requests_per_hour: 1000,
            max_requests_per_day: 10000,
            burst_allowance: 10,
            storage_backend: RateLimitStorageBackend::Memory,
        },
        password_policy: PasswordPolicyConfig {
            min_length: 8,
            max_length: 128,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special_chars: false, // Disabled for testing
            prevent_common_passwords: true,
            password_history_length: 3,
            password_expiration_days: Some(90),
        },
    }
}

/// Create test user registration data
fn create_test_user_data() -> UserRegistrationData {
    UserRegistrationData {
        username: "testuser".to_string(),
        email: "test@example.com".to_string(),
        display_name: "Test User".to_string(),
        password: "TestPassword123".to_string(),
        roles: vec!["user".to_string()],
        metadata: HashMap::new(),
    }
}

/// Create test security context
fn create_test_security_context() -> SecurityContext {
    SecurityContext::new("test_user_id".to_string(), "test_session_id".to_string())
        .with_ip_address("192.168.1.100".to_string())
        .with_user_agent("Test Agent".to_string())
}

#[tokio::test]
async fn test_security_config_default() {
    let config = SecurityConfig::default();

    assert_eq!(config.jwt.algorithm, JwtAlgorithm::HS256);
    assert_eq!(config.session.timeout_seconds, 3600);
    assert_eq!(config.encryption.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert!(config.audit.enabled);
    assert!(config.rate_limiting.enabled);
    assert_eq!(config.password_policy.min_length, 12);
}

#[tokio::test]
async fn test_security_context_creation() {
    let context = create_test_security_context();

    assert_eq!(context.user_id, "test_user_id");
    assert_eq!(context.session_id, "test_session_id");
    assert_eq!(context.ip_address, Some("192.168.1.100".to_string()));
    assert_eq!(context.user_agent, Some("Test Agent".to_string()));
    assert!(context.roles.is_empty());
    assert!(context.permissions.is_empty());
}

#[tokio::test]
async fn test_security_context_roles_and_permissions() {
    let mut context = create_test_security_context();

    context.add_role("admin".to_string());
    context.add_role("user".to_string());
    context.add_permission("read".to_string());
    context.add_permission("write".to_string());

    assert!(context.has_role("admin"));
    assert!(context.has_role("user"));
    assert!(!context.has_role("guest"));

    assert!(context.has_permission("read"));
    assert!(context.has_permission("write"));
    assert!(!context.has_permission("delete"));
}

#[tokio::test]
async fn test_jwt_authentication_service_creation() {
    let config = create_test_security_config();
    let _service = JwtAuthenticationService::new(config.jwt.clone(), config.password_policy);

    // Service should be created successfully
    // We can't access private fields, so just test that it was created
    assert_eq!(config.jwt.algorithm, JwtAlgorithm::HS256);
}

#[tokio::test]
async fn test_user_registration() {
    let config = create_test_security_config();
    let service = JwtAuthenticationService::new(config.jwt, config.password_policy);
    let user_data = create_test_user_data();

    let result = service.register_user(user_data.clone()).await;
    assert!(result.is_ok());

    let user = result.unwrap();
    assert_eq!(user.username, user_data.username);
    assert_eq!(user.email, user_data.email);
    assert_eq!(user.status, UserStatus::Active);
    assert!(!user.password_hash.is_empty());
}

#[tokio::test]
async fn test_password_hashing_and_verification() {
    let config = create_test_security_config();
    let service = JwtAuthenticationService::new(config.jwt, config.password_policy);
    let password = "TestPassword123";

    let hash_result = service.hash_password(password).await;
    assert!(hash_result.is_ok());

    let hash = hash_result.unwrap();
    assert!(!hash.is_empty());

    let verify_result = service.verify_password(password, &hash).await;
    assert!(verify_result.is_ok());
    assert!(verify_result.unwrap());

    let wrong_verify_result = service.verify_password("WrongPassword", &hash).await;
    assert!(wrong_verify_result.is_ok());
    assert!(!wrong_verify_result.unwrap());
}

#[tokio::test]
async fn test_password_policy_validation() {
    let config = create_test_security_config();
    let service = JwtAuthenticationService::new(config.jwt, config.password_policy);

    // Valid password
    let valid_result = service.validate_password_policy("ValidPass123").await;
    assert!(valid_result.is_ok());
    let validation = valid_result.unwrap();
    assert!(validation.valid);
    assert!(validation.errors.is_empty());

    // Invalid password (too short)
    let invalid_result = service.validate_password_policy("short").await;
    assert!(invalid_result.is_ok());
    let validation = invalid_result.unwrap();
    assert!(!validation.valid);
    assert!(!validation.errors.is_empty());
}

#[tokio::test]
async fn test_jwt_token_generation_and_validation() {
    let config = create_test_security_config();
    let service = JwtAuthenticationService::new(config.jwt, config.password_policy);
    let user_id = "test_user";
    let roles = vec!["user".to_string(), "admin".to_string()];

    // Generate token
    let token_result = service.generate_token(user_id, roles.clone()).await;
    assert!(token_result.is_ok());

    let token_pair = token_result.unwrap();
    assert!(!token_pair.access_token.is_empty());
    assert!(!token_pair.refresh_token.is_empty());
    assert_eq!(token_pair.token_type, "Bearer");

    // Validate token
    let validate_result = service.validate_token(&token_pair.access_token).await;
    if let Err(e) = &validate_result {
        println!("Token validation error: {:?}", e);
    }
    assert!(validate_result.is_ok());

    let claims = validate_result.unwrap();
    assert_eq!(claims.sub, user_id);
    assert_eq!(claims.roles, roles);
}

#[tokio::test]
async fn test_user_authentication() {
    let config = create_test_security_config();
    let service = JwtAuthenticationService::new(config.jwt, config.password_policy);
    let user_data = create_test_user_data();

    // Register user
    let _user = service.register_user(user_data.clone()).await.unwrap();

    // Authenticate with correct credentials
    let auth_result = service
        .authenticate(&user_data.username, &user_data.password)
        .await;
    assert!(auth_result.is_ok());

    let auth = auth_result.unwrap();
    assert!(auth.success);
    assert!(auth.user.is_some());
    assert!(auth.error.is_none());

    // Authenticate with wrong credentials
    let wrong_auth_result = service
        .authenticate(&user_data.username, "WrongPassword")
        .await;
    assert!(wrong_auth_result.is_ok());

    let wrong_auth = wrong_auth_result.unwrap();
    assert!(!wrong_auth.success);
    assert!(wrong_auth.user.is_none());
    assert!(wrong_auth.error.is_some());
}

#[tokio::test]
async fn test_rbac_authorization_service() {
    let service = RbacAuthorizationService::new();
    service.initialize_defaults().await.unwrap();

    // Test role creation
    let role = Role {
        name: "test_role".to_string(),
        description: "Test role".to_string(),
        permissions: vec!["read".to_string()],
        parent_roles: Vec::new(),
        metadata: HashMap::new(),
    };

    let create_result = service.create_role(role).await;
    assert!(create_result.is_ok());

    // Test role exists
    let exists_result = service.role_exists("test_role").await;
    assert!(exists_result.is_ok());
    assert!(exists_result.unwrap());
}

#[tokio::test]
async fn test_authorization_permission_check() {
    let service = RbacAuthorizationService::new();
    service.initialize_defaults().await.unwrap();

    let user_id = "test_user";
    let mut context = create_test_security_context();
    context.user_id = user_id.to_string();

    // Add user role
    service.add_user_role(user_id, "user").await.unwrap();

    // Check permission
    let check_result = service.check_permission(&context, "document", "read").await;
    assert!(check_result.is_ok());
    assert!(check_result.unwrap()); // Should have read permission

    // Check denied permission
    let denied_result = service
        .check_permission(&context, "document", "delete")
        .await;
    assert!(denied_result.is_ok());
    assert!(!denied_result.unwrap()); // Should not have delete permission
}

#[tokio::test]
async fn test_encryption_service() {
    let config = create_test_security_config();
    let service = RingEncryptionService::new(config.encryption);

    let key_id = "test_key";
    let data = b"Hello, World!";

    // Generate key
    let gen_result = service
        .generate_key(key_id, EncryptionAlgorithm::Aes256Gcm)
        .await;
    assert!(gen_result.is_ok());

    // Encrypt data
    let encrypt_result = service.encrypt(data, key_id).await;
    assert!(encrypt_result.is_ok());

    let encrypted = encrypt_result.unwrap();
    assert!(!encrypted.data.is_empty());
    assert!(!encrypted.nonce.is_empty());
    assert_eq!(encrypted.algorithm, EncryptionAlgorithm::Aes256Gcm);

    // Decrypt data
    let decrypt_result = service.decrypt(&encrypted, key_id).await;
    assert!(decrypt_result.is_ok());

    let decrypted = decrypt_result.unwrap();
    assert_eq!(decrypted, data);
}

#[tokio::test]
async fn test_encryption_string_operations() {
    let config = create_test_security_config();
    let service = RingEncryptionService::new(config.encryption);

    let key_id = "test_key";
    let data = "Hello, World!";

    // Generate key
    service
        .generate_key(key_id, EncryptionAlgorithm::Aes256Gcm)
        .await
        .unwrap();

    // Encrypt string
    let encrypt_result = service.encrypt_string(data, key_id).await;
    assert!(encrypt_result.is_ok());

    let encrypted = encrypt_result.unwrap();
    assert!(!encrypted.is_empty());

    // Decrypt string
    let decrypt_result = service.decrypt_string(&encrypted, key_id).await;
    assert!(decrypt_result.is_ok());

    let decrypted = decrypt_result.unwrap();
    assert_eq!(decrypted, data);
}

#[tokio::test]
async fn test_session_manager() {
    let config = create_test_security_config();
    let manager = MemorySessionManager::new(config.session);

    let user_id = "test_user";
    let ip_address = Some("192.168.1.100".to_string());

    // Create session
    let session_result = manager.create_session(user_id, ip_address.clone()).await;
    assert!(session_result.is_ok());

    let session_id = session_result.unwrap();
    assert!(!session_id.is_empty());

    // Get session
    let get_result = manager.get_session(&session_id).await;
    assert!(get_result.is_ok());

    let session = get_result.unwrap();
    assert!(session.is_some());

    let session = session.unwrap();
    assert_eq!(session.user_id, user_id);
    assert_eq!(session.ip_address, ip_address);
    assert_eq!(session.status, SessionStatus::Active);

    // Validate session
    let valid_result = manager.is_session_valid(&session_id).await;
    assert!(valid_result.is_ok());
    assert!(valid_result.unwrap());
}

#[tokio::test]
async fn test_session_data_operations() {
    let config = create_test_security_config();
    let manager = MemorySessionManager::new(config.session);

    let user_id = "test_user";
    let session_id = manager.create_session(user_id, None).await.unwrap();

    // Set session data
    let set_result = manager
        .set_session_data(&session_id, "key1", "value1".to_string())
        .await;
    assert!(set_result.is_ok());

    // Get session data
    let get_result = manager.get_session_data(&session_id, "key1").await;
    assert!(get_result.is_ok());

    let value = get_result.unwrap();
    assert_eq!(value, Some("value1".to_string()));

    // Remove session data
    let remove_result = manager.remove_session_data(&session_id, "key1").await;
    assert!(remove_result.is_ok());

    // Verify data is removed
    let get_after_remove = manager.get_session_data(&session_id, "key1").await;
    assert!(get_after_remove.is_ok());
    assert_eq!(get_after_remove.unwrap(), None);
}

#[tokio::test]
async fn test_audit_service() {
    let config = create_test_security_config();
    let temp_dir = tempfile::tempdir().unwrap();
    let log_path = temp_dir.path().join("test_audit.log");

    let service = FileAuditService::new(config.audit, log_path).await.unwrap();

    // Log security event
    let event = SecurityEvent::AuthenticationAttempt {
        user_id: "test_user".to_string(),
        success: true,
        method: "password".to_string(),
        ip_address: Some("192.168.1.100".to_string()),
    };

    let log_result = service.log_event(event).await;
    assert!(log_result.is_ok());

    // Query logs
    let query = AuditQuery {
        time_range: TimeRange {
            start: chrono::Utc::now() - chrono::Duration::hours(1),
            end: chrono::Utc::now() + chrono::Duration::hours(1),
        },
        levels: vec![AuditLogLevel::All],
        event_types: Vec::new(),
        user_ids: Vec::new(),
        ip_addresses: Vec::new(),
        session_ids: Vec::new(),
        limit: Some(10),
        offset: None,
        sort_order: SortOrder::TimestampDesc,
    };

    let query_result = service.query_logs(query).await;
    assert!(query_result.is_ok());

    let logs = query_result.unwrap();
    assert_eq!(logs.len(), 1);
    assert!(matches!(
        logs[0].event,
        SecurityEvent::AuthenticationAttempt { .. }
    ));
}

#[tokio::test]
async fn test_policy_engine() {
    let engine = SimplePolicyEngine::new();
    engine.initialize_defaults().await.unwrap();

    // Test policy creation
    let policy = SecurityPolicy {
        name: "test_policy".to_string(),
        description: "Test policy".to_string(),
        version: "1.0".to_string(),
        effect: policy::PolicyEffect::Allow,
        resources: vec!["test_resource".to_string()],
        actions: vec!["read".to_string()],
        conditions: Vec::new(),
        priority: 10,
        enabled: true,
        metadata: HashMap::new(),
    };

    let add_result = engine.add_policy(policy).await;
    assert!(add_result.is_ok());

    // Test policy evaluation
    let context = create_test_security_context();
    let eval_result = engine
        .evaluate_all_policies(&context, "test_resource", "read")
        .await;
    assert!(eval_result.is_ok());

    let decision = eval_result.unwrap();
    assert!(decision.granted);
    assert!(decision.deciding_policy.is_some());
}

#[tokio::test]
async fn test_security_manager_creation() {
    let config = create_test_security_config();
    let manager_result = SecurityManager::new(config).await;
    assert!(manager_result.is_ok());

    let manager = manager_result.unwrap();
    assert_eq!(manager.config().jwt.algorithm, JwtAlgorithm::HS256);
}

#[tokio::test]
async fn test_security_context_validation() {
    let config = create_test_security_config();
    let manager = SecurityManager::new(config).await.unwrap();

    let context = create_test_security_context();
    let validate_result = manager.validate_context(&context).await;

    // Should fail because session doesn't exist
    assert!(validate_result.is_ok());
    assert!(!validate_result.unwrap());
}

#[tokio::test]
async fn test_security_event_logging() {
    let config = create_test_security_config();
    let manager = SecurityManager::new(config).await.unwrap();

    let event = SecurityEvent::ConfigurationChange {
        user_id: "admin".to_string(),
        component: "security".to_string(),
        change: "Updated password policy".to_string(),
    };

    let log_result = manager.log_security_event(event).await;
    assert!(log_result.is_ok());
}

#[tokio::test]
async fn test_encryption_key_operations() {
    let config = create_test_security_config();
    let service = RingEncryptionService::new(config.encryption);

    let key_id = "test_key";

    // Check key doesn't exist
    let exists_result = service.key_exists(key_id).await;
    assert!(exists_result.is_ok());
    assert!(!exists_result.unwrap());

    // Generate key
    let gen_result = service
        .generate_key(key_id, EncryptionAlgorithm::Aes256Gcm)
        .await;
    assert!(gen_result.is_ok());

    // Check key exists
    let exists_after_gen = service.key_exists(key_id).await;
    assert!(exists_after_gen.is_ok());
    assert!(exists_after_gen.unwrap());

    // Get key metadata
    let metadata_result = service.get_key_metadata(key_id).await;
    assert!(metadata_result.is_ok());

    let metadata = metadata_result.unwrap();
    assert_eq!(metadata.key_id, key_id);
    assert_eq!(metadata.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert_eq!(metadata.status, KeyStatus::Active);
}

#[tokio::test]
async fn test_random_generation_and_hashing() {
    let config = create_test_security_config();
    let service = RingEncryptionService::new(config.encryption);

    // Generate random data
    let random_result = service.generate_random(32).await;
    assert!(random_result.is_ok());

    let random_data = random_result.unwrap();
    assert_eq!(random_data.len(), 32);

    // Hash data
    let hash_result = service.hash_data(&random_data, HashAlgorithm::Sha256).await;
    assert!(hash_result.is_ok());

    let hash = hash_result.unwrap();
    assert_eq!(hash.len(), 32); // SHA-256 produces 32-byte hash
}

#[tokio::test]
async fn test_session_cleanup() {
    let mut config = create_test_security_config();
    config.session.timeout_seconds = 1; // Very short timeout for testing

    let manager = MemorySessionManager::new(config.session);

    let user_id = "test_user";
    let session_id = manager.create_session(user_id, None).await.unwrap();

    // Wait for session to expire
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Cleanup expired sessions
    let cleanup_result = manager.cleanup_expired_sessions().await;
    assert!(cleanup_result.is_ok());

    let cleaned_count = cleanup_result.unwrap();
    assert_eq!(cleaned_count, 1);

    // Session should no longer be valid
    let valid_result = manager.is_session_valid(&session_id).await;
    assert!(valid_result.is_ok());
    assert!(!valid_result.unwrap());
}

#[tokio::test]
async fn test_rate_limiting_service() {
    let config = create_test_security_config();
    let max_requests = config.rate_limiting.max_requests_per_minute;
    let service = MemoryRateLimitService::new(config.rate_limiting);

    let key = "test_user";
    let limit_type = RateLimitType::PerMinute;

    // First request should be allowed
    let result1 = service.check_rate_limit(key, limit_type.clone()).await;
    assert!(result1.is_ok());
    assert!(result1.unwrap().allowed);

    // Record the request
    service
        .record_request(key, limit_type.clone())
        .await
        .unwrap();

    // Check usage
    let usage = service.get_usage(key, limit_type.clone()).await;
    assert!(usage.is_ok());
    let usage = usage.unwrap();
    assert_eq!(usage.count, 1);
    assert_eq!(usage.limit, max_requests);
}

#[tokio::test]
async fn test_rate_limiting_enforcement() {
    let mut config = create_test_security_config();
    config.rate_limiting.max_requests_per_minute = 2; // Very low limit for testing
    config.rate_limiting.burst_allowance = 0; // No burst allowance for strict testing

    let service = MemoryRateLimitService::new(config.rate_limiting);
    let key = "test_user";
    let limit_type = RateLimitType::PerMinute;

    // First two requests should be allowed
    for i in 1..=2 {
        let result = service
            .check_rate_limit(key, limit_type.clone())
            .await
            .unwrap();
        assert!(result.allowed, "Request {} should be allowed", i);
        service
            .record_request(key, limit_type.clone())
            .await
            .unwrap();
    }

    // Third request should be blocked (no burst allowance)
    let result = service
        .check_rate_limit(key, limit_type.clone())
        .await
        .unwrap();
    assert!(!result.allowed, "Request 3 should be blocked");
}

#[tokio::test]
async fn test_rate_limiting_statistics() {
    let config = create_test_security_config();
    let service = MemoryRateLimitService::new(config.rate_limiting);

    let key = "test_user";
    let limit_type = RateLimitType::PerMinute;

    // Make some requests
    for _ in 0..3 {
        service
            .check_rate_limit(key, limit_type.clone())
            .await
            .unwrap();
        service
            .record_request(key, limit_type.clone())
            .await
            .unwrap();
    }

    // Get statistics
    let stats = service.get_statistics().await;
    assert!(stats.is_ok());
    let stats = stats.unwrap();
    assert_eq!(stats.total_requests, 3);
    assert_eq!(stats.active_keys, 1);
}

#[tokio::test]
async fn test_rate_limiting_cleanup() {
    let config = create_test_security_config();
    let service = MemoryRateLimitService::new(config.rate_limiting);

    let key = "test_user";
    let limit_type = RateLimitType::PerMinute;

    // Record a request
    service
        .record_request(key, limit_type.clone())
        .await
        .unwrap();

    // Cleanup (should not remove anything yet as it's not expired)
    let cleaned = service.cleanup_expired().await.unwrap();
    assert_eq!(cleaned, 0);

    // Check that entry still exists
    let usage = service.get_usage(key, limit_type).await.unwrap();
    assert_eq!(usage.count, 1);
}

#[tokio::test]
async fn test_security_middleware_bypass_paths() {
    let config = create_test_security_config();
    let manager = std::sync::Arc::new(SecurityManager::new(config).await.unwrap());
    let middleware_config = SecurityMiddlewareConfig::default();
    let middleware = SecurityMiddleware::new(manager, middleware_config);

    let request = HttpRequest {
        method: "GET".to_string(),
        path: "/health".to_string(),
        headers: HashMap::new(),
        client_ip: Some("127.0.0.1".to_string()),
        user_agent: Some("Test Agent".to_string()),
        body_size: 0,
        timestamp: std::time::SystemTime::now(),
    };

    let result = middleware.process_request(request).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.allowed);
    assert!(result.context.is_none()); // No authentication required for bypass paths
}

#[tokio::test]
async fn test_security_middleware_authentication_required() {
    let config = create_test_security_config();
    let manager = std::sync::Arc::new(SecurityManager::new(config).await.unwrap());
    let middleware_config = SecurityMiddlewareConfig::default();
    let middleware = SecurityMiddleware::new(manager, middleware_config);

    let request = HttpRequest {
        method: "GET".to_string(),
        path: "/api/data".to_string(),
        headers: HashMap::new(),
        client_ip: Some("127.0.0.1".to_string()),
        user_agent: Some("Test Agent".to_string()),
        body_size: 0,
        timestamp: std::time::SystemTime::now(),
    };

    let result = middleware.process_request(request).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(!result.allowed);
    assert_eq!(
        result.denial_reason,
        Some("Authentication required".to_string())
    );
}

#[tokio::test]
async fn test_security_middleware_request_size_limit() {
    let config = create_test_security_config();
    let manager = std::sync::Arc::new(SecurityManager::new(config).await.unwrap());
    let mut middleware_config = SecurityMiddlewareConfig::default();
    middleware_config.max_request_size = Some(100); // Very small limit
    middleware_config.require_authentication = false; // Disable auth for this test
    let middleware = SecurityMiddleware::new(manager, middleware_config);

    let request = HttpRequest {
        method: "POST".to_string(),
        path: "/api/upload".to_string(),
        headers: HashMap::new(),
        client_ip: Some("127.0.0.1".to_string()),
        user_agent: Some("Test Agent".to_string()),
        body_size: 1000, // Exceeds limit
        timestamp: std::time::SystemTime::now(),
    };

    let result = middleware.process_request(request).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(!result.allowed);
    assert!(result
        .denial_reason
        .as_ref()
        .unwrap()
        .contains("Request body too large"));
}

#[tokio::test]
async fn test_security_middleware_header_validation() {
    let config = create_test_security_config();
    let manager = std::sync::Arc::new(SecurityManager::new(config).await.unwrap());
    let mut middleware_config = SecurityMiddlewareConfig::default();
    middleware_config.require_authentication = false; // Disable auth for this test
    let middleware = SecurityMiddleware::new(manager, middleware_config);

    let headers = HashMap::new();
    // Missing User-Agent header (required by default config)

    let request = HttpRequest {
        method: "GET".to_string(),
        path: "/api/data".to_string(),
        headers,
        client_ip: Some("127.0.0.1".to_string()),
        user_agent: None,
        body_size: 0,
        timestamp: std::time::SystemTime::now(),
    };

    let result = middleware.process_request(request).await;
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.allowed); // Still allowed but with warnings
    assert!(!result.warnings.is_empty());
    assert!(result.warnings[0].contains("Missing required header: User-Agent"));
}
