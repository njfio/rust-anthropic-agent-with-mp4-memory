use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;

/// Security headers configuration
#[derive(Debug, Clone)]
pub struct SecurityHeadersConfig {
    /// Content Security Policy
    pub csp: Option<String>,
    /// X-Frame-Options
    pub frame_options: Option<String>,
    /// X-Content-Type-Options
    pub content_type_options: bool,
    /// X-XSS-Protection
    pub xss_protection: bool,
    /// Strict-Transport-Security
    pub hsts: Option<String>,
    /// Referrer-Policy
    pub referrer_policy: Option<String>,
    /// Permissions-Policy
    pub permissions_policy: Option<String>,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
}

impl Default for SecurityHeadersConfig {
    fn default() -> Self {
        Self {
            csp: Some("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https:; font-src 'self'; object-src 'none'; media-src 'self'; frame-src 'none';".to_string()),
            frame_options: Some("DENY".to_string()),
            content_type_options: true,
            xss_protection: true,
            hsts: Some("max-age=31536000; includeSubDomains; preload".to_string()),
            referrer_policy: Some("strict-origin-when-cross-origin".to_string()),
            permissions_policy: Some("geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), speaker=()".to_string()),
            custom_headers: HashMap::new(),
        }
    }
}

impl SecurityHeadersConfig {
    /// Create a new security headers configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Content Security Policy
    pub fn with_csp<S: Into<String>>(mut self, csp: S) -> Self {
        self.csp = Some(csp.into());
        self
    }

    /// Disable Content Security Policy
    pub fn without_csp(mut self) -> Self {
        self.csp = None;
        self
    }

    /// Set X-Frame-Options
    pub fn with_frame_options<S: Into<String>>(mut self, options: S) -> Self {
        self.frame_options = Some(options.into());
        self
    }

    /// Set HSTS header
    pub fn with_hsts<S: Into<String>>(mut self, hsts: S) -> Self {
        self.hsts = Some(hsts.into());
        self
    }

    /// Add custom header
    pub fn with_custom_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.custom_headers.insert(key.into(), value.into());
        self
    }

    /// Create a relaxed configuration for development
    pub fn relaxed() -> Self {
        Self {
            csp: Some("default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' data: https: http:; connect-src 'self' https: http: ws: wss:;".to_string()),
            frame_options: Some("SAMEORIGIN".to_string()),
            content_type_options: true,
            xss_protection: false, // Can interfere with development
            hsts: None, // Not needed for development
            referrer_policy: Some("no-referrer-when-downgrade".to_string()),
            permissions_policy: None,
            custom_headers: HashMap::new(),
        }
    }

    /// Create a strict configuration for production
    pub fn strict() -> Self {
        Self {
            csp: Some("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; media-src 'none'; frame-src 'none'; base-uri 'self'; form-action 'self';".to_string()),
            frame_options: Some("DENY".to_string()),
            content_type_options: true,
            xss_protection: true,
            hsts: Some("max-age=63072000; includeSubDomains; preload".to_string()),
            referrer_policy: Some("no-referrer".to_string()),
            permissions_policy: Some("geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), speaker=(), fullscreen=(), sync-xhr=()".to_string()),
            custom_headers: HashMap::new(),
        }
    }
}

/// Security headers builder
#[derive(Debug, Clone)]
pub struct SecurityHeaders {
    config: SecurityHeadersConfig,
}

impl SecurityHeaders {
    /// Create new security headers with default configuration
    pub fn new() -> Self {
        Self {
            config: SecurityHeadersConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: SecurityHeadersConfig) -> Self {
        Self { config }
    }

    /// Build HeaderMap for HTTP requests
    pub fn build_request_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        // Add User-Agent for identification
        if let Ok(value) = HeaderValue::from_str("rust-memvid-agent/1.0 (Security-Enhanced)") {
            headers.insert("User-Agent", value);
        }

        // Add security headers for outgoing requests
        if let Ok(value) = HeaderValue::from_str("no-cache, no-store, must-revalidate") {
            headers.insert("Cache-Control", value);
        }

        if let Ok(value) = HeaderValue::from_str("no-cache") {
            headers.insert("Pragma", value);
        }

        // Add custom headers
        for (key, value) in &self.config.custom_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }

        headers
    }

    /// Build HeaderMap for HTTP responses (if we serve HTTP)
    pub fn build_response_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        // Content Security Policy
        if let Some(csp) = &self.config.csp {
            if let Ok(value) = HeaderValue::from_str(csp) {
                headers.insert("Content-Security-Policy", value);
            }
        }

        // X-Frame-Options
        if let Some(frame_options) = &self.config.frame_options {
            if let Ok(value) = HeaderValue::from_str(frame_options) {
                headers.insert("X-Frame-Options", value);
            }
        }

        // X-Content-Type-Options
        if self.config.content_type_options {
            if let Ok(value) = HeaderValue::from_str("nosniff") {
                headers.insert("X-Content-Type-Options", value);
            }
        }

        // X-XSS-Protection
        if self.config.xss_protection {
            if let Ok(value) = HeaderValue::from_str("1; mode=block") {
                headers.insert("X-XSS-Protection", value);
            }
        }

        // Strict-Transport-Security
        if let Some(hsts) = &self.config.hsts {
            if let Ok(value) = HeaderValue::from_str(hsts) {
                headers.insert("Strict-Transport-Security", value);
            }
        }

        // Referrer-Policy
        if let Some(referrer_policy) = &self.config.referrer_policy {
            if let Ok(value) = HeaderValue::from_str(referrer_policy) {
                headers.insert("Referrer-Policy", value);
            }
        }

        // Permissions-Policy
        if let Some(permissions_policy) = &self.config.permissions_policy {
            if let Ok(value) = HeaderValue::from_str(permissions_policy) {
                headers.insert("Permissions-Policy", value);
            }
        }

        // Additional security headers
        if let Ok(value) = HeaderValue::from_str("no-cache, no-store, must-revalidate") {
            headers.insert("Cache-Control", value);
        }

        if let Ok(value) = HeaderValue::from_str("0") {
            headers.insert("Expires", value);
        }

        if let Ok(value) = HeaderValue::from_str("no-cache") {
            headers.insert("Pragma", value);
        }

        // Custom headers
        for (key, value) in &self.config.custom_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }

        headers
    }

    /// Validate response headers for security compliance
    pub fn validate_response_headers(&self, headers: &HeaderMap) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for missing security headers
        if !headers.contains_key("Content-Security-Policy") && self.config.csp.is_some() {
            issues.push("Missing Content-Security-Policy header".to_string());
        }

        if !headers.contains_key("X-Frame-Options") && self.config.frame_options.is_some() {
            issues.push("Missing X-Frame-Options header".to_string());
        }

        if !headers.contains_key("X-Content-Type-Options") && self.config.content_type_options {
            issues.push("Missing X-Content-Type-Options header".to_string());
        }

        if !headers.contains_key("Strict-Transport-Security") && self.config.hsts.is_some() {
            issues.push("Missing Strict-Transport-Security header".to_string());
        }

        // Check for insecure values
        if let Some(frame_options) = headers.get("X-Frame-Options") {
            if let Ok(value) = frame_options.to_str() {
                if value.to_lowercase() == "allowall" {
                    issues.push("Insecure X-Frame-Options value: ALLOWALL".to_string());
                }
            }
        }

        if let Some(csp) = headers.get("Content-Security-Policy") {
            if let Ok(value) = csp.to_str() {
                if value.contains("'unsafe-eval'") {
                    issues.push("CSP contains unsafe-eval directive".to_string());
                }
                if value.contains("*") && !value.contains("data:") {
                    issues.push("CSP contains overly permissive wildcard".to_string());
                }
            }
        }

        issues
    }
}

impl Default for SecurityHeaders {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_security_headers() {
        let headers = SecurityHeaders::new();
        let response_headers = headers.build_response_headers();
        
        assert!(response_headers.contains_key("Content-Security-Policy"));
        assert!(response_headers.contains_key("X-Frame-Options"));
        assert!(response_headers.contains_key("X-Content-Type-Options"));
    }

    #[test]
    fn test_strict_configuration() {
        let config = SecurityHeadersConfig::strict();
        let headers = SecurityHeaders::with_config(config);
        let response_headers = headers.build_response_headers();
        
        assert!(response_headers.contains_key("Strict-Transport-Security"));
        assert_eq!(
            response_headers.get("X-Frame-Options").unwrap().to_str().unwrap(),
            "DENY"
        );
    }

    #[test]
    fn test_header_validation() {
        let headers = SecurityHeaders::new();
        let mut test_headers = HeaderMap::new();
        
        let issues = headers.validate_response_headers(&test_headers);
        assert!(!issues.is_empty()); // Should have missing header issues
        
        // Add required headers
        test_headers.insert("Content-Security-Policy", HeaderValue::from_static("default-src 'self'"));
        test_headers.insert("X-Frame-Options", HeaderValue::from_static("DENY"));
        test_headers.insert("X-Content-Type-Options", HeaderValue::from_static("nosniff"));
        
        let issues = headers.validate_response_headers(&test_headers);
        assert!(issues.len() < 3); // Should have fewer issues now
    }

    #[test]
    fn test_custom_headers() {
        let config = SecurityHeadersConfig::new()
            .with_custom_header("X-Custom-Security", "enabled")
            .with_custom_header("X-API-Version", "1.0");
        
        let headers = SecurityHeaders::with_config(config);
        let request_headers = headers.build_request_headers();
        
        assert!(request_headers.contains_key("X-Custom-Security"));
        assert!(request_headers.contains_key("X-API-Version"));
    }
}
