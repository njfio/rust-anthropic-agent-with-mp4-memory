use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::{rate_limiting::RateLimitType, SecurityContext, SecurityEvent, SecurityManager};
use crate::utils::error::{AgentError, Result};

/// Security middleware for HTTP requests
pub struct SecurityMiddleware {
    /// Security manager
    security_manager: Arc<SecurityManager>,
    /// Middleware configuration
    config: SecurityMiddlewareConfig,
}

/// Security middleware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMiddlewareConfig {
    /// Whether to enforce authentication
    pub require_authentication: bool,
    /// Whether to enforce rate limiting
    pub enable_rate_limiting: bool,
    /// Whether to log all requests
    pub log_requests: bool,
    /// Paths that bypass security checks
    pub bypass_paths: Vec<String>,
    /// Required permissions for specific paths
    pub path_permissions: HashMap<String, Vec<String>>,
    /// Custom headers to validate
    pub required_headers: Vec<String>,
    /// Maximum request size in bytes
    pub max_request_size: Option<usize>,
}

/// HTTP request information
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// Request method
    pub method: String,
    /// Request path
    pub path: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Client IP address
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Request body size
    pub body_size: usize,
    /// Request timestamp
    pub timestamp: std::time::SystemTime,
}

/// Security check result
#[derive(Debug, Clone)]
pub struct SecurityCheckResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Security context if authenticated
    pub context: Option<SecurityContext>,
    /// Reason for denial if not allowed
    pub denial_reason: Option<String>,
    /// Rate limit information
    pub rate_limit_info: Option<RateLimitInfo>,
    /// Security warnings
    pub warnings: Vec<String>,
}

/// Rate limit information
#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    /// Current request count
    pub current_count: u32,
    /// Maximum allowed requests
    pub limit: u32,
    /// Time until reset in seconds
    pub reset_time_seconds: u64,
    /// Remaining requests
    pub remaining: u32,
}

impl Default for SecurityMiddlewareConfig {
    fn default() -> Self {
        Self {
            require_authentication: true,
            enable_rate_limiting: true,
            log_requests: true,
            bypass_paths: vec![
                "/health".to_string(),
                "/metrics".to_string(),
                "/login".to_string(),
                "/register".to_string(),
            ],
            path_permissions: HashMap::new(),
            required_headers: vec!["User-Agent".to_string()],
            max_request_size: Some(10 * 1024 * 1024), // 10MB
        }
    }
}

impl SecurityMiddleware {
    /// Create a new security middleware
    pub fn new(security_manager: Arc<SecurityManager>, config: SecurityMiddlewareConfig) -> Self {
        Self {
            security_manager,
            config,
        }
    }

    /// Check if a path should bypass security
    fn should_bypass_security(&self, path: &str) -> bool {
        self.config.bypass_paths.iter().any(|bypass_path| {
            if bypass_path.ends_with('*') {
                let prefix = &bypass_path[..bypass_path.len() - 1];
                path.starts_with(prefix)
            } else {
                path == bypass_path
            }
        })
    }

    /// Extract authentication token from headers
    fn extract_auth_token(&self, headers: &HashMap<String, String>) -> Option<String> {
        // Check Authorization header
        if let Some(auth_header) = headers.get("Authorization") {
            if auth_header.starts_with("Bearer ") {
                return Some(auth_header[7..].to_string());
            }
        }

        // Check X-API-Key header
        if let Some(api_key) = headers.get("X-API-Key") {
            return Some(api_key.clone());
        }

        None
    }

    /// Validate required headers
    fn validate_headers(&self, headers: &HashMap<String, String>) -> Vec<String> {
        let mut warnings = Vec::new();

        for required_header in &self.config.required_headers {
            if !headers.contains_key(required_header) {
                warnings.push(format!("Missing required header: {}", required_header));
            }
        }

        warnings
    }

    /// Check request size limits
    fn check_request_size(&self, request: &HttpRequest) -> Result<()> {
        if let Some(max_size) = self.config.max_request_size {
            if request.body_size > max_size {
                return Err(AgentError::validation(format!(
                    "Request body too large: {} bytes (max: {} bytes)",
                    request.body_size, max_size
                )));
            }
        }
        Ok(())
    }

    /// Get rate limit key for a request
    fn get_rate_limit_key(
        &self,
        request: &HttpRequest,
        context: Option<&SecurityContext>,
    ) -> String {
        if let Some(ctx) = context {
            format!("user:{}", ctx.user_id)
        } else if let Some(ip) = &request.client_ip {
            format!("ip:{}", ip)
        } else {
            "anonymous".to_string()
        }
    }

    /// Check rate limits
    async fn check_rate_limits(
        &self,
        request: &HttpRequest,
        context: Option<&SecurityContext>,
    ) -> Result<Option<RateLimitInfo>> {
        if !self.config.enable_rate_limiting {
            return Ok(None);
        }

        let rate_limit_service = self.security_manager.rate_limit_service.read().await;
        let key = self.get_rate_limit_key(request, context);

        let result = rate_limit_service
            .check_rate_limit(&key, RateLimitType::PerMinute)
            .await?;

        if result.allowed {
            rate_limit_service
                .record_request(&key, RateLimitType::PerMinute)
                .await?;
        }

        Ok(Some(RateLimitInfo {
            current_count: result.current_count,
            limit: result.limit,
            reset_time_seconds: result.reset_time_seconds,
            remaining: result.remaining,
        }))
    }

    /// Authenticate request
    async fn authenticate_request(&self, request: &HttpRequest) -> Result<Option<SecurityContext>> {
        if !self.config.require_authentication {
            return Ok(None);
        }

        let token = self.extract_auth_token(&request.headers);
        if let Some(token) = token {
            let auth_service = self.security_manager.auth_service.read().await;
            match auth_service.validate_token(&token).await {
                Ok(claims) => {
                    let mut context = SecurityContext::new(claims.sub, "middleware".to_string());
                    context.roles = claims.roles;
                    context.permissions = claims.permissions;
                    if let Some(ip) = &request.client_ip {
                        context = context.with_ip_address(ip.clone());
                    }
                    if let Some(ua) = &request.user_agent {
                        context = context.with_user_agent(ua.clone());
                    }
                    Ok(Some(context))
                }
                Err(_) => Ok(None), // Invalid token
            }
        } else {
            Ok(None) // No token provided
        }
    }

    /// Check path permissions
    async fn check_path_permissions(
        &self,
        request: &HttpRequest,
        context: Option<&SecurityContext>,
    ) -> Result<bool> {
        if let Some(required_permissions) = self.config.path_permissions.get(&request.path) {
            if let Some(ctx) = context {
                let authz_service = self.security_manager.authz_service.read().await;
                for permission in required_permissions {
                    if !authz_service
                        .check_permission(ctx, &request.path, permission)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            } else {
                Ok(false) // No context but permissions required
            }
        } else {
            Ok(true) // No specific permissions required
        }
    }

    /// Log security event
    async fn log_security_event(&self, request: &HttpRequest, result: &SecurityCheckResult) {
        if !self.config.log_requests {
            return;
        }

        let event = if result.allowed {
            SecurityEvent::DataAccess {
                user_id: result
                    .context
                    .as_ref()
                    .map(|c| c.user_id.clone())
                    .unwrap_or_else(|| "anonymous".to_string()),
                resource: request.path.clone(),
                action: request.method.clone(),
                sensitive: request.path.contains("/admin") || request.path.contains("/sensitive"),
            }
        } else {
            SecurityEvent::PolicyViolation {
                user_id: Some(
                    result
                        .context
                        .as_ref()
                        .map(|c| c.user_id.clone())
                        .unwrap_or_else(|| "anonymous".to_string()),
                ),
                policy: "security_middleware".to_string(),
                violation: result
                    .denial_reason
                    .clone()
                    .unwrap_or_else(|| "Access denied".to_string()),
            }
        };

        if let Err(e) = self.security_manager.log_security_event(event).await {
            eprintln!("Failed to log security event: {}", e);
        }
    }
}

impl SecurityMiddleware {
    /// Process an HTTP request through security checks
    pub async fn process_request(&self, request: HttpRequest) -> Result<SecurityCheckResult> {
        let mut warnings = Vec::new();

        // Check if path should bypass security
        if self.should_bypass_security(&request.path) {
            return Ok(SecurityCheckResult {
                allowed: true,
                context: None,
                denial_reason: None,
                rate_limit_info: None,
                warnings,
            });
        }

        // Validate request size
        if let Err(e) = self.check_request_size(&request) {
            let result = SecurityCheckResult {
                allowed: false,
                context: None,
                denial_reason: Some(e.to_string()),
                rate_limit_info: None,
                warnings,
            };
            self.log_security_event(&request, &result).await;
            return Ok(result);
        }

        // Validate headers
        warnings.extend(self.validate_headers(&request.headers));

        // Authenticate request
        let context = self.authenticate_request(&request).await?;

        // Check if authentication is required but not provided
        if self.config.require_authentication && context.is_none() {
            let result = SecurityCheckResult {
                allowed: false,
                context: None,
                denial_reason: Some("Authentication required".to_string()),
                rate_limit_info: None,
                warnings,
            };
            self.log_security_event(&request, &result).await;
            return Ok(result);
        }

        // Check rate limits
        let rate_limit_info = self.check_rate_limits(&request, context.as_ref()).await?;
        if let Some(ref rate_info) = rate_limit_info {
            if rate_info.remaining == 0 {
                let result = SecurityCheckResult {
                    allowed: false,
                    context,
                    denial_reason: Some("Rate limit exceeded".to_string()),
                    rate_limit_info,
                    warnings,
                };
                self.log_security_event(&request, &result).await;
                return Ok(result);
            }
        }

        // Check path permissions
        let has_permission = self
            .check_path_permissions(&request, context.as_ref())
            .await?;
        if !has_permission {
            let result = SecurityCheckResult {
                allowed: false,
                context,
                denial_reason: Some("Insufficient permissions".to_string()),
                rate_limit_info,
                warnings,
            };
            self.log_security_event(&request, &result).await;
            return Ok(result);
        }

        // All checks passed
        let result = SecurityCheckResult {
            allowed: true,
            context,
            denial_reason: None,
            rate_limit_info,
            warnings,
        };
        self.log_security_event(&request, &result).await;
        Ok(result)
    }
}
