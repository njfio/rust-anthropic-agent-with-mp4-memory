use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::utils::error::Result;

pub mod audit;
/// Enterprise-grade security framework for the agent system
pub mod authentication;
pub mod authorization;
pub mod encryption;
pub mod middleware;
pub mod policy;
pub mod rate_limiting;
pub mod session;

#[cfg(test)]
mod tests;

/// Security configuration for the agent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// JWT configuration
    pub jwt: JwtConfig,
    /// Session management configuration
    pub session: SessionConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Audit logging configuration
    pub audit: AuditConfig,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// Password policy configuration
    pub password_policy: PasswordPolicyConfig,
}

/// JWT (JSON Web Token) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// Secret key for signing tokens
    pub secret: String,
    /// Token expiration time in seconds
    pub expiration_seconds: u64,
    /// Refresh token expiration time in seconds
    pub refresh_expiration_seconds: u64,
    /// Issuer identifier
    pub issuer: String,
    /// Audience identifier
    pub audience: String,
    /// Algorithm to use for signing
    pub algorithm: JwtAlgorithm,
}

/// Supported JWT algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JwtAlgorithm {
    /// HMAC using SHA-256
    HS256,
    /// HMAC using SHA-512
    HS512,
    /// RSA using SHA-256
    RS256,
    /// RSA using SHA-512
    RS512,
    /// ECDSA using P-256 and SHA-256
    ES256,
}

/// Session management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum concurrent sessions per user
    pub max_concurrent_sessions: u32,
    /// Enable session persistence
    pub persistent_sessions: bool,
    /// Session storage backend
    pub storage_backend: SessionStorageBackend,
}

/// Session storage backends
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStorageBackend {
    /// In-memory storage (not persistent)
    Memory,
    /// Redis storage
    Redis,
    /// Database storage
    Database,
    /// File-based storage
    File,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Data encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation configuration
    pub key_derivation: KeyDerivationConfig,
    /// Enable encryption at rest
    pub encrypt_at_rest: bool,
    /// Enable encryption in transit
    pub encrypt_in_transit: bool,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM
    Aes256Gcm,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
    /// AES-256-CBC
    Aes256Cbc,
}

/// Key derivation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    /// Algorithm for key derivation
    pub algorithm: KeyDerivationAlgorithm,
    /// Number of iterations
    pub iterations: u32,
    /// Memory cost (for Argon2)
    pub memory_cost: u32,
    /// Parallelism factor (for Argon2)
    pub parallelism: u32,
    /// Salt length in bytes
    pub salt_length: usize,
}

/// Key derivation algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyDerivationAlgorithm {
    /// Argon2id (recommended)
    Argon2id,
    /// Argon2i
    Argon2i,
    /// Argon2d
    Argon2d,
    /// PBKDF2 with SHA-256
    Pbkdf2Sha256,
    /// scrypt
    Scrypt,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log level for audit events
    pub log_level: AuditLogLevel,
    /// Audit log storage backend
    pub storage_backend: AuditStorageBackend,
    /// Log retention period in days
    pub retention_days: u32,
    /// Enable log encryption
    pub encrypt_logs: bool,
}

/// Audit log levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditLogLevel {
    /// Log all events
    All,
    /// Log security-related events only
    Security,
    /// Log authentication events only
    Authentication,
    /// Log authorization events only
    Authorization,
    /// Log critical events only
    Critical,
}

/// Audit log storage backends
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditStorageBackend {
    /// File-based storage
    File,
    /// Database storage
    Database,
    /// Syslog
    Syslog,
    /// External SIEM system
    Siem,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum requests per minute
    pub max_requests_per_minute: u32,
    /// Maximum requests per hour
    pub max_requests_per_hour: u32,
    /// Maximum requests per day
    pub max_requests_per_day: u32,
    /// Burst allowance
    pub burst_allowance: u32,
    /// Rate limit storage backend
    pub storage_backend: RateLimitStorageBackend,
}

/// Rate limit storage backends
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RateLimitStorageBackend {
    /// In-memory storage
    Memory,
    /// Redis storage
    Redis,
    /// Database storage
    Database,
}

/// Password policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicyConfig {
    /// Minimum password length
    pub min_length: usize,
    /// Maximum password length
    pub max_length: usize,
    /// Require uppercase letters
    pub require_uppercase: bool,
    /// Require lowercase letters
    pub require_lowercase: bool,
    /// Require numbers
    pub require_numbers: bool,
    /// Require special characters
    pub require_special_chars: bool,
    /// Prevent common passwords
    pub prevent_common_passwords: bool,
    /// Password history length
    pub password_history_length: usize,
    /// Password expiration days
    pub password_expiration_days: Option<u32>,
}

/// Security context for operations
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// User ID
    pub user_id: String,
    /// Session ID
    pub session_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Request IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Request timestamp
    pub timestamp: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Security event types for audit logging
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityEvent {
    /// User authentication attempt
    AuthenticationAttempt {
        user_id: String,
        success: bool,
        method: String,
        ip_address: Option<String>,
    },
    /// User authorization check
    AuthorizationCheck {
        user_id: String,
        resource: String,
        action: String,
        granted: bool,
    },
    /// Session creation
    SessionCreated {
        user_id: String,
        session_id: String,
        ip_address: Option<String>,
    },
    /// Session termination
    SessionTerminated {
        user_id: String,
        session_id: String,
        reason: String,
    },
    /// Password change
    PasswordChanged {
        user_id: String,
        ip_address: Option<String>,
    },
    /// Rate limit exceeded
    RateLimitExceeded {
        user_id: Option<String>,
        ip_address: Option<String>,
        limit_type: String,
    },
    /// Security policy violation
    PolicyViolation {
        user_id: Option<String>,
        policy: String,
        violation: String,
    },
    /// Data access
    DataAccess {
        user_id: String,
        resource: String,
        action: String,
        sensitive: bool,
    },
    /// Configuration change
    ConfigurationChange {
        user_id: String,
        component: String,
        change: String,
    },
    /// Security incident
    SecurityIncident {
        incident_type: String,
        severity: String,
        description: String,
    },
}

/// Security manager for the agent system
pub struct SecurityManager {
    /// Security configuration
    config: SecurityConfig,
    /// Authentication service
    auth_service: RwLock<Box<dyn authentication::AuthenticationService>>,
    /// Authorization service
    authz_service: RwLock<Box<dyn authorization::AuthorizationService>>,
    /// Encryption service
    encryption_service: RwLock<Box<dyn encryption::EncryptionService>>,
    /// Audit service
    audit_service: RwLock<Box<dyn audit::AuditService>>,
    /// Session manager
    session_manager: RwLock<Box<dyn session::SessionManager>>,
    /// Policy engine
    policy_engine: RwLock<Box<dyn policy::PolicyEngine>>,
    /// Rate limiting service
    rate_limit_service: RwLock<Box<dyn rate_limiting::RateLimitService>>,
}

impl Default for PasswordPolicyConfig {
    fn default() -> Self {
        Self {
            min_length: 12,
            max_length: 128,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special_chars: true,
            prevent_common_passwords: true,
            password_history_length: 5,
            password_expiration_days: Some(90),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            jwt: JwtConfig {
                secret: Uuid::new_v4().to_string(),
                expiration_seconds: 3600,              // 1 hour
                refresh_expiration_seconds: 86400 * 7, // 7 days
                issuer: "rust-agent".to_string(),
                audience: "rust-agent-api".to_string(),
                algorithm: JwtAlgorithm::HS256,
            },
            session: SessionConfig {
                timeout_seconds: 3600, // 1 hour
                max_concurrent_sessions: 5,
                persistent_sessions: false,
                storage_backend: SessionStorageBackend::Memory,
            },
            encryption: EncryptionConfig {
                algorithm: EncryptionAlgorithm::Aes256Gcm,
                key_derivation: KeyDerivationConfig {
                    algorithm: KeyDerivationAlgorithm::Argon2id,
                    iterations: 2,
                    memory_cost: 19456, // 19 MB
                    parallelism: 1,
                    salt_length: 16,
                },
                encrypt_at_rest: true,
                encrypt_in_transit: true,
            },
            audit: AuditConfig {
                enabled: true,
                log_level: AuditLogLevel::Security,
                storage_backend: AuditStorageBackend::File,
                retention_days: 90,
                encrypt_logs: true,
            },
            rate_limiting: RateLimitConfig {
                enabled: true,
                max_requests_per_minute: 60,
                max_requests_per_hour: 1000,
                max_requests_per_day: 10000,
                burst_allowance: 10,
                storage_backend: RateLimitStorageBackend::Memory,
            },
            password_policy: PasswordPolicyConfig {
                min_length: 12,
                max_length: 128,
                require_uppercase: true,
                require_lowercase: true,
                require_numbers: true,
                require_special_chars: true,
                prevent_common_passwords: true,
                password_history_length: 5,
                password_expiration_days: Some(90),
            },
        }
    }
}

impl SecurityContext {
    /// Create a new security context
    pub fn new(user_id: String, session_id: String) -> Self {
        Self {
            user_id,
            session_id,
            roles: Vec::new(),
            permissions: Vec::new(),
            ip_address: None,
            user_agent: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if the context has a specific role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(&role.to_string())
    }

    /// Check if the context has a specific permission
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string())
    }

    /// Add a role to the context
    pub fn add_role(&mut self, role: String) {
        if !self.roles.contains(&role) {
            self.roles.push(role);
        }
    }

    /// Add a permission to the context
    pub fn add_permission(&mut self, permission: String) {
        if !self.permissions.contains(&permission) {
            self.permissions.push(permission);
        }
    }

    /// Set IP address
    pub fn with_ip_address(mut self, ip_address: String) -> Self {
        self.ip_address = Some(ip_address);
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = Some(user_agent);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl SecurityManager {
    /// Create a new security manager
    pub async fn new(config: SecurityConfig) -> Result<Self> {
        let auth_service = authentication::create_authentication_service(&config.jwt).await?;
        let authz_service = authorization::create_authorization_service().await?;
        let encryption_service = encryption::create_encryption_service(&config.encryption).await?;
        let audit_service = audit::create_audit_service(&config.audit).await?;
        let session_manager = session::create_session_manager(&config.session).await?;
        let policy_engine = policy::create_policy_engine().await?;
        let rate_limit_service =
            rate_limiting::create_rate_limit_service(&config.rate_limiting).await?;

        Ok(Self {
            config,
            auth_service: RwLock::new(auth_service),
            authz_service: RwLock::new(authz_service),
            encryption_service: RwLock::new(encryption_service),
            audit_service: RwLock::new(audit_service),
            session_manager: RwLock::new(session_manager),
            policy_engine: RwLock::new(policy_engine),
            rate_limit_service: RwLock::new(rate_limit_service),
        })
    }

    /// Get the security configuration
    pub fn config(&self) -> &SecurityConfig {
        &self.config
    }

    /// Log a security event
    pub async fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        let audit_service = self.audit_service.read().await;
        audit_service.log_event(event).await
    }

    /// Validate a security context
    pub async fn validate_context(&self, context: &SecurityContext) -> Result<bool> {
        // Check session validity
        let session_manager = self.session_manager.read().await;
        if !session_manager
            .is_session_valid(&context.session_id)
            .await?
        {
            return Ok(false);
        }

        // Check rate limits
        if self.config.rate_limiting.enabled {
            let rate_limit_service = self.rate_limit_service.read().await;
            let key = format!("user:{}", context.user_id);
            let result = rate_limit_service
                .check_rate_limit(&key, rate_limiting::RateLimitType::PerMinute)
                .await?;
            if !result.allowed {
                return Ok(false);
            }
            rate_limit_service
                .record_request(&key, rate_limiting::RateLimitType::PerMinute)
                .await?;
        }

        // Log the validation attempt
        self.log_security_event(SecurityEvent::AuthorizationCheck {
            user_id: context.user_id.clone(),
            resource: "security_context".to_string(),
            action: "validate".to_string(),
            granted: true,
        })
        .await?;

        Ok(true)
    }

    /// Create a new security context for a user
    pub async fn create_context(
        &self,
        user_id: String,
        ip_address: Option<String>,
    ) -> Result<SecurityContext> {
        let session_manager = self.session_manager.read().await;
        let session_id = session_manager
            .create_session(&user_id, ip_address.clone())
            .await?;

        let mut context = SecurityContext::new(user_id.clone(), session_id.clone());
        if let Some(ip) = ip_address {
            context = context.with_ip_address(ip);
        }

        // Log session creation
        self.log_security_event(SecurityEvent::SessionCreated {
            user_id,
            session_id,
            ip_address: context.ip_address.clone(),
        })
        .await?;

        Ok(context)
    }

    /// Terminate a security context
    pub async fn terminate_context(&self, context: &SecurityContext, reason: String) -> Result<()> {
        let session_manager = self.session_manager.read().await;
        session_manager
            .terminate_session(&context.session_id)
            .await?;

        // Log session termination
        self.log_security_event(SecurityEvent::SessionTerminated {
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            reason,
        })
        .await?;

        Ok(())
    }

    /// Check if a user has permission to perform an action on a resource
    pub async fn check_permission(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &str,
    ) -> Result<bool> {
        let authz_service = self.authz_service.read().await;
        authz_service.check_permission(context, resource, action).await
    }

    /// Get authorization service (for advanced operations)
    pub async fn get_authorization_service(&self) -> tokio::sync::RwLockReadGuard<'_, Box<dyn authorization::AuthorizationService>> {
        self.authz_service.read().await
    }

    /// Encrypt sensitive data using the encryption service
    pub async fn encrypt_data(&self, data: &[u8], key_id: &str) -> Result<encryption::EncryptedData> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.encrypt(data, key_id).await
    }

    /// Decrypt sensitive data using the encryption service
    pub async fn decrypt_data(&self, encrypted_data: &encryption::EncryptedData, key_id: &str) -> Result<Vec<u8>> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.decrypt(encrypted_data, key_id).await
    }

    /// Encrypt a string using the encryption service
    pub async fn encrypt_string(&self, data: &str, key_id: &str) -> Result<String> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.encrypt_string(data, key_id).await
    }

    /// Decrypt a string using the encryption service
    pub async fn decrypt_string(&self, encrypted_data: &str, key_id: &str) -> Result<String> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.decrypt_string(encrypted_data, key_id).await
    }

    /// Generate an encryption key
    pub async fn generate_encryption_key(&self, key_id: &str, algorithm: EncryptionAlgorithm) -> Result<()> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.generate_key(key_id, algorithm).await
    }

    /// Check if an encryption key exists
    pub async fn encryption_key_exists(&self, key_id: &str) -> Result<bool> {
        let encryption_service = self.encryption_service.read().await;
        encryption_service.key_exists(key_id).await
    }

    /// Evaluate a security policy
    pub async fn evaluate_policy(&self, policy_name: &str, context: &SecurityContext) -> Result<policy::PolicyDecision> {
        let policy_engine = self.policy_engine.read().await;
        policy_engine.evaluate_policy(policy_name, context).await
    }

    /// Evaluate all applicable policies for a resource and action
    pub async fn evaluate_all_policies(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<policy::PolicyDecision> {
        let policy_engine = self.policy_engine.read().await;
        policy_engine.evaluate_all_policies(context, resource, action).await
    }

    /// Add a security policy
    pub async fn add_security_policy(&self, policy: policy::SecurityPolicy) -> Result<()> {
        let policy_engine = self.policy_engine.read().await;
        policy_engine.add_policy(policy).await
    }

    /// Check if a security policy exists
    pub async fn security_policy_exists(&self, policy_name: &str) -> Result<bool> {
        let policy_engine = self.policy_engine.read().await;
        policy_engine.policy_exists(policy_name).await
    }

    /// Get policy evaluation statistics
    pub async fn get_policy_statistics(&self) -> Result<policy::PolicyStatistics> {
        let policy_engine = self.policy_engine.read().await;
        policy_engine.get_evaluation_statistics().await
    }
}

impl std::fmt::Debug for SecurityManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityManager")
            .field("config", &self.config)
            .finish()
    }
}
