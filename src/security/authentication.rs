use async_trait::async_trait;
use jsonwebtoken::{encode, decode, Header, Validation, Algorithm, EncodingKey, DecodingKey};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, password_hash::SaltString};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::utils::error::{AgentError, Result};
use super::{JwtConfig, JwtAlgorithm, PasswordPolicyConfig};

/// Authentication service trait
#[async_trait]
pub trait AuthenticationService: Send + Sync {
    /// Authenticate a user with username and password
    async fn authenticate(&self, username: &str, password: &str) -> Result<AuthenticationResult>;
    
    /// Generate a JWT token for a user
    async fn generate_token(&self, user_id: &str, roles: Vec<String>) -> Result<TokenPair>;
    
    /// Validate a JWT token
    async fn validate_token(&self, token: &str) -> Result<TokenClaims>;
    
    /// Refresh a JWT token
    async fn refresh_token(&self, refresh_token: &str) -> Result<TokenPair>;
    
    /// Hash a password
    async fn hash_password(&self, password: &str) -> Result<String>;
    
    /// Verify a password against a hash
    async fn verify_password(&self, password: &str, hash: &str) -> Result<bool>;
    
    /// Validate password against policy
    async fn validate_password_policy(&self, password: &str) -> Result<PasswordValidationResult>;
    
    /// Register a new user
    async fn register_user(&self, user_data: UserRegistrationData) -> Result<User>;
    
    /// Get user by username
    async fn get_user(&self, username: &str) -> Result<Option<User>>;
    
    /// Update user password
    async fn update_password(&self, user_id: &str, old_password: &str, new_password: &str) -> Result<()>;
    
    /// Revoke a token
    async fn revoke_token(&self, token: &str) -> Result<()>;
    
    /// Check if token is revoked
    async fn is_token_revoked(&self, token: &str) -> Result<bool>;
}

/// Authentication result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResult {
    /// Whether authentication was successful
    pub success: bool,
    /// User information if successful
    pub user: Option<User>,
    /// Error message if failed
    pub error: Option<String>,
    /// Authentication method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Token pair (access token and refresh token)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    /// Access token
    pub access_token: String,
    /// Refresh token
    pub refresh_token: String,
    /// Token type (usually "Bearer")
    pub token_type: String,
    /// Expiration time in seconds
    pub expires_in: u64,
}

/// JWT token claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// Expiration time
    pub exp: u64,
    /// Not before time
    pub nbf: u64,
    /// Issued at time
    pub iat: u64,
    /// JWT ID
    pub jti: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Token type (access or refresh)
    pub token_type: String,
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User ID
    pub id: String,
    /// Username
    pub username: String,
    /// Email address
    pub email: String,
    /// Display name
    pub display_name: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Account status
    pub status: UserStatus,
    /// Password hash
    pub password_hash: String,
    /// Password history
    pub password_history: Vec<String>,
    /// Last login time
    pub last_login: Option<SystemTime>,
    /// Account creation time
    pub created_at: SystemTime,
    /// Account last updated time
    pub updated_at: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// User account status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserStatus {
    /// Account is active
    Active,
    /// Account is inactive
    Inactive,
    /// Account is suspended
    Suspended,
    /// Account is locked
    Locked,
    /// Account is pending verification
    PendingVerification,
}

/// User registration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRegistrationData {
    /// Username
    pub username: String,
    /// Email address
    pub email: String,
    /// Display name
    pub display_name: String,
    /// Password
    pub password: String,
    /// Initial roles
    pub roles: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Password validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordValidationResult {
    /// Whether password is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Password strength score (0-100)
    pub strength_score: u8,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// JWT-based authentication service implementation
pub struct JwtAuthenticationService {
    /// JWT configuration
    config: JwtConfig,
    /// Password policy configuration
    password_policy: PasswordPolicyConfig,
    /// User storage
    users: RwLock<HashMap<String, User>>,
    /// Revoked tokens
    revoked_tokens: RwLock<HashMap<String, SystemTime>>,
    /// Argon2 hasher
    argon2: Argon2<'static>,
}

impl JwtAuthenticationService {
    /// Create a new JWT authentication service
    pub fn new(config: JwtConfig, password_policy: PasswordPolicyConfig) -> Self {
        Self {
            config,
            password_policy,
            users: RwLock::new(HashMap::new()),
            revoked_tokens: RwLock::new(HashMap::new()),
            argon2: Argon2::default(),
        }
    }

    /// Get JWT algorithm
    fn get_algorithm(&self) -> Algorithm {
        match self.config.algorithm {
            JwtAlgorithm::HS256 => Algorithm::HS256,
            JwtAlgorithm::HS512 => Algorithm::HS512,
            JwtAlgorithm::RS256 => Algorithm::RS256,
            JwtAlgorithm::RS512 => Algorithm::RS512,
            JwtAlgorithm::ES256 => Algorithm::ES256,
        }
    }

    /// Create token claims
    fn create_claims(&self, user_id: &str, roles: Vec<String>, token_type: &str) -> TokenClaims {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let exp = if token_type == "refresh" {
            now + self.config.refresh_expiration_seconds
        } else {
            now + self.config.expiration_seconds
        };

        TokenClaims {
            sub: user_id.to_string(),
            iss: self.config.issuer.clone(),
            aud: self.config.audience.clone(),
            exp,
            nbf: now,
            iat: now,
            jti: Uuid::new_v4().to_string(),
            roles: roles.clone(),
            permissions: self.map_roles_to_permissions(&roles),
            token_type: token_type.to_string(),
        }
    }

    /// Validate password strength
    fn calculate_password_strength(&self, password: &str) -> u8 {
        let mut score = 0u8;
        
        // Length scoring
        if password.len() >= 8 { score += 20; }
        if password.len() >= 12 { score += 10; }
        if password.len() >= 16 { score += 10; }
        
        // Character variety scoring
        if password.chars().any(|c| c.is_uppercase()) { score += 15; }
        if password.chars().any(|c| c.is_lowercase()) { score += 15; }
        if password.chars().any(|c| c.is_numeric()) { score += 15; }
        if password.chars().any(|c| !c.is_alphanumeric()) { score += 15; }
        
        score.min(100)
    }

    /// Check if password is common
    fn is_common_password(&self, password: &str) -> bool {
        // Simple check against common passwords
        const COMMON_PASSWORDS: &[&str] = &[
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "1234567890", "abc123"
        ];

        COMMON_PASSWORDS.contains(&password.to_lowercase().as_str())
    }

    /// Map roles to permissions
    fn map_roles_to_permissions(&self, roles: &[String]) -> Vec<String> {
        let mut permissions = Vec::new();

        for role in roles {
            let role_permissions: Vec<String> = match role.as_str() {
                "admin" => vec![
                    "user:create", "user:read", "user:update", "user:delete",
                    "system:configure", "system:monitor", "system:backup",
                    "memory:read", "memory:write", "memory:delete",
                    "tools:execute", "tools:configure", "tools:manage",
                    "security:audit", "security:configure"
                ].into_iter().map(|s| s.to_string()).collect(),
                "user" => vec![
                    "user:read", "user:update_own",
                    "memory:read", "memory:write",
                    "tools:execute"
                ].into_iter().map(|s| s.to_string()).collect(),
                "readonly" => vec![
                    "user:read_own",
                    "memory:read",
                    "tools:read"
                ].into_iter().map(|s| s.to_string()).collect(),
                "operator" => vec![
                    "user:read", "user:update_own",
                    "system:monitor",
                    "memory:read", "memory:write",
                    "tools:execute", "tools:configure"
                ].into_iter().map(|s| s.to_string()).collect(),
                "developer" => vec![
                    "user:read", "user:update_own",
                    "system:monitor",
                    "memory:read", "memory:write",
                    "tools:execute", "tools:configure", "tools:manage",
                    "security:audit"
                ].into_iter().map(|s| s.to_string()).collect(),
                _ => {
                    // Unknown role gets minimal permissions
                    vec!["user:read_own".to_string()]
                }
            };

            permissions.extend(role_permissions);
        }

        // Remove duplicates and sort
        permissions.sort();
        permissions.dedup();
        permissions
    }
}

#[async_trait]
impl AuthenticationService for JwtAuthenticationService {
    async fn authenticate(&self, username: &str, password: &str) -> Result<AuthenticationResult> {
        let users = self.users.read().await;
        
        if let Some(user) = users.get(username) {
            if user.status != UserStatus::Active {
                return Ok(AuthenticationResult {
                    success: false,
                    user: None,
                    error: Some("Account is not active".to_string()),
                    method: "password".to_string(),
                    metadata: HashMap::new(),
                });
            }

            let password_valid = self.verify_password(password, &user.password_hash).await?;
            
            if password_valid {
                let mut user_clone = user.clone();
                user_clone.last_login = Some(SystemTime::now());
                
                Ok(AuthenticationResult {
                    success: true,
                    user: Some(user_clone),
                    error: None,
                    method: "password".to_string(),
                    metadata: HashMap::new(),
                })
            } else {
                Ok(AuthenticationResult {
                    success: false,
                    user: None,
                    error: Some("Invalid credentials".to_string()),
                    method: "password".to_string(),
                    metadata: HashMap::new(),
                })
            }
        } else {
            Ok(AuthenticationResult {
                success: false,
                user: None,
                error: Some("User not found".to_string()),
                method: "password".to_string(),
                metadata: HashMap::new(),
            })
        }
    }

    async fn generate_token(&self, user_id: &str, roles: Vec<String>) -> Result<TokenPair> {
        let access_claims = self.create_claims(user_id, roles.clone(), "access");
        let refresh_claims = self.create_claims(user_id, roles, "refresh");

        let header = Header::new(self.get_algorithm());
        let encoding_key = EncodingKey::from_secret(self.config.secret.as_ref());

        let access_token = encode(&header, &access_claims, &encoding_key)
            .map_err(|e| AgentError::validation(format!("Failed to generate access token: {}", e)))?;

        let refresh_token = encode(&header, &refresh_claims, &encoding_key)
            .map_err(|e| AgentError::validation(format!("Failed to generate refresh token: {}", e)))?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            token_type: "Bearer".to_string(),
            expires_in: self.config.expiration_seconds,
        })
    }

    async fn validate_token(&self, token: &str) -> Result<TokenClaims> {
        // Check if token is revoked
        if self.is_token_revoked(token).await? {
            return Err(AgentError::validation("Token has been revoked".to_string()));
        }

        let decoding_key = DecodingKey::from_secret(self.config.secret.as_ref());
        let mut validation = Validation::new(self.get_algorithm());

        // Set the expected issuer and audience
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_audience(&[&self.config.audience]);

        let token_data = decode::<TokenClaims>(token, &decoding_key, &validation)
            .map_err(|e| AgentError::validation(format!("Invalid token: {}", e)))?;

        Ok(token_data.claims)
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<TokenPair> {
        let claims = self.validate_token(refresh_token).await?;
        
        if claims.token_type != "refresh" {
            return Err(AgentError::validation("Invalid token type for refresh".to_string()));
        }

        // Revoke the old refresh token
        self.revoke_token(refresh_token).await?;

        // Generate new token pair
        self.generate_token(&claims.sub, claims.roles).await
    }

    async fn hash_password(&self, password: &str) -> Result<String> {
        let salt = SaltString::generate(&mut OsRng);
        let password_hash = self.argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| AgentError::validation(format!("Failed to hash password: {}", e)))?;
        
        Ok(password_hash.to_string())
    }

    async fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| AgentError::validation(format!("Invalid password hash: {}", e)))?;
        
        Ok(self.argon2.verify_password(password.as_bytes(), &parsed_hash).is_ok())
    }

    async fn validate_password_policy(&self, password: &str) -> Result<PasswordValidationResult> {
        let mut errors = Vec::new();
        let mut suggestions = Vec::new();

        // Length validation
        if password.len() < self.password_policy.min_length {
            errors.push(format!("Password must be at least {} characters long", self.password_policy.min_length));
            suggestions.push("Use a longer password".to_string());
        }

        if password.len() > self.password_policy.max_length {
            errors.push(format!("Password must be no more than {} characters long", self.password_policy.max_length));
        }

        // Character requirements
        if self.password_policy.require_uppercase && !password.chars().any(|c| c.is_uppercase()) {
            errors.push("Password must contain at least one uppercase letter".to_string());
            suggestions.push("Add uppercase letters".to_string());
        }

        if self.password_policy.require_lowercase && !password.chars().any(|c| c.is_lowercase()) {
            errors.push("Password must contain at least one lowercase letter".to_string());
            suggestions.push("Add lowercase letters".to_string());
        }

        if self.password_policy.require_numbers && !password.chars().any(|c| c.is_numeric()) {
            errors.push("Password must contain at least one number".to_string());
            suggestions.push("Add numbers".to_string());
        }

        if self.password_policy.require_special_chars && !password.chars().any(|c| !c.is_alphanumeric()) {
            errors.push("Password must contain at least one special character".to_string());
            suggestions.push("Add special characters like !@#$%".to_string());
        }

        // Common password check
        if self.password_policy.prevent_common_passwords && self.is_common_password(password) {
            errors.push("Password is too common".to_string());
            suggestions.push("Use a more unique password".to_string());
        }

        let strength_score = self.calculate_password_strength(password);
        let valid = errors.is_empty();

        Ok(PasswordValidationResult {
            valid,
            errors,
            strength_score,
            suggestions,
        })
    }

    async fn register_user(&self, user_data: UserRegistrationData) -> Result<User> {
        // Validate password policy
        let password_validation = self.validate_password_policy(&user_data.password).await?;
        if !password_validation.valid {
            return Err(AgentError::validation(format!(
                "Password policy validation failed: {}",
                password_validation.errors.join(", ")
            )));
        }

        // Check if user already exists
        let users = self.users.read().await;
        if users.contains_key(&user_data.username) {
            return Err(AgentError::validation("Username already exists".to_string()));
        }
        drop(users);

        // Hash password
        let password_hash = self.hash_password(&user_data.password).await?;

        // Create user
        let user = User {
            id: Uuid::new_v4().to_string(),
            username: user_data.username.clone(),
            email: user_data.email,
            display_name: user_data.display_name,
            roles: user_data.roles,
            permissions: Vec::new(),
            status: UserStatus::Active,
            password_hash,
            password_history: Vec::new(),
            last_login: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: user_data.metadata,
        };

        // Store user
        let mut users = self.users.write().await;
        users.insert(user_data.username, user.clone());

        Ok(user)
    }

    async fn get_user(&self, username: &str) -> Result<Option<User>> {
        let users = self.users.read().await;
        Ok(users.get(username).cloned())
    }

    async fn update_password(&self, user_id: &str, old_password: &str, new_password: &str) -> Result<()> {
        // Find user by ID
        let mut users = self.users.write().await;
        let user = users.values_mut()
            .find(|u| u.id == user_id)
            .ok_or_else(|| AgentError::validation("User not found".to_string()))?;

        // Verify old password
        if !self.verify_password(old_password, &user.password_hash).await? {
            return Err(AgentError::validation("Invalid current password".to_string()));
        }

        // Validate new password policy
        let password_validation = self.validate_password_policy(new_password).await?;
        if !password_validation.valid {
            return Err(AgentError::validation(format!(
                "New password policy validation failed: {}",
                password_validation.errors.join(", ")
            )));
        }

        // Check password history
        for old_hash in &user.password_history {
            if self.verify_password(new_password, old_hash).await? {
                return Err(AgentError::validation("Cannot reuse a recent password".to_string()));
            }
        }

        // Hash new password
        let new_hash = self.hash_password(new_password).await?;

        // Update password history
        user.password_history.push(user.password_hash.clone());
        if user.password_history.len() > self.password_policy.password_history_length {
            user.password_history.remove(0);
        }

        // Update password
        user.password_hash = new_hash;
        user.updated_at = SystemTime::now();

        Ok(())
    }

    async fn revoke_token(&self, token: &str) -> Result<()> {
        let mut revoked_tokens = self.revoked_tokens.write().await;
        revoked_tokens.insert(token.to_string(), SystemTime::now());
        Ok(())
    }

    async fn is_token_revoked(&self, token: &str) -> Result<bool> {
        let revoked_tokens = self.revoked_tokens.read().await;
        Ok(revoked_tokens.contains_key(token))
    }
}

/// Create an authentication service
pub async fn create_authentication_service(config: &JwtConfig) -> Result<Box<dyn AuthenticationService>> {
    let password_policy = super::PasswordPolicyConfig::default();
    Ok(Box::new(JwtAuthenticationService::new(config.clone(), password_policy)))
}
