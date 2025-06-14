// HTTP Client Tool for Enterprise AI Agent System
// Provides secure HTTP request capabilities with comprehensive validation

use crate::anthropic::models::ToolDefinition;
use crate::tools::{Tool, ToolResult};
use crate::utils::error::{AgentError, Result};
use crate::utils::validation::validate_url;
use async_trait::async_trait;
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// HTTP client tool for making secure web requests
#[derive(Debug, Clone)]
pub struct HttpClientTool {
    /// HTTP client with configured timeouts and security settings
    client: Client,
    /// Maximum response size in bytes
    max_response_size: usize,
    /// Allowed domains (if empty, all domains are allowed)
    allowed_domains: Vec<String>,
    /// Request timeout in seconds
    timeout_seconds: u64,
    /// Maximum redirects to follow
    max_redirects: u32,
    /// Whether to verify SSL certificates
    verify_ssl: bool,
}

/// HTTP request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    /// HTTP method (GET, POST, PUT, DELETE, etc.)
    pub method: String,
    /// Target URL
    pub url: String,
    /// Request headers
    pub headers: Option<HashMap<String, String>>,
    /// Request body (for POST, PUT, etc.)
    pub body: Option<String>,
    /// Query parameters
    pub query_params: Option<HashMap<String, String>>,
    /// Request timeout in seconds (overrides default)
    pub timeout: Option<u64>,
    /// Whether to follow redirects
    pub follow_redirects: Option<bool>,
}

/// HTTP response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponseData {
    /// HTTP status code
    pub status_code: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body as string
    pub body: String,
    /// Response size in bytes
    pub size_bytes: usize,
    /// Request duration in milliseconds
    pub duration_ms: u64,
    /// Final URL after redirects
    pub final_url: String,
}

/// HTTP client configuration
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Maximum response size in bytes (default: 10MB)
    pub max_response_size: usize,
    /// Allowed domains (empty = all allowed)
    pub allowed_domains: Vec<String>,
    /// Request timeout in seconds (default: 30)
    pub timeout_seconds: u64,
    /// Maximum redirects (default: 10)
    pub max_redirects: u32,
    /// Verify SSL certificates (default: true)
    pub verify_ssl: bool,
    /// User agent string
    pub user_agent: String,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            max_response_size: 10 * 1024 * 1024, // 10MB
            allowed_domains: Vec::new(),
            timeout_seconds: 30,
            max_redirects: 10,
            verify_ssl: true,
            user_agent: "RustMemVidAgent/1.0".to_string(),
        }
    }
}

impl HttpClientTool {
    /// Create a new HTTP client tool with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(HttpClientConfig::default())
    }

    /// Create a new HTTP client tool with custom configuration
    pub fn with_config(config: HttpClientConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects as usize))
            .danger_accept_invalid_certs(!config.verify_ssl)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| AgentError::tool("http_client", &format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            max_response_size: config.max_response_size,
            allowed_domains: config.allowed_domains,
            timeout_seconds: config.timeout_seconds,
            max_redirects: config.max_redirects,
            verify_ssl: config.verify_ssl,
        })
    }

    /// Create a secure HTTP client tool with restricted domains
    pub fn with_allowed_domains(domains: Vec<String>) -> Result<Self> {
        let config = HttpClientConfig {
            allowed_domains: domains,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Validate the HTTP request
    fn validate_request(&self, request: &HttpRequest) -> Result<()> {
        // Validate URL format and security
        validate_url(&request.url)?;

        // Check domain allowlist if configured
        if !self.allowed_domains.is_empty() {
            let url = reqwest::Url::parse(&request.url)
                .map_err(|e| AgentError::validation(format!("Invalid URL: {}", e)))?;
            
            if let Some(domain) = url.domain() {
                if !self.allowed_domains.iter().any(|allowed| domain.ends_with(allowed)) {
                    return Err(AgentError::validation(
                        format!("Domain '{}' is not in the allowed domains list", domain)
                    ));
                }
            }
        }

        // Validate HTTP method
        let method = request.method.to_uppercase();
        if !matches!(method.as_str(), "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "HEAD" | "OPTIONS") {
            return Err(AgentError::validation(
                format!("Unsupported HTTP method: {}", method)
            ));
        }

        // Validate headers
        if let Some(headers) = &request.headers {
            for (key, value) in headers {
                if key.is_empty() || value.is_empty() {
                    return Err(AgentError::validation("Header key and value cannot be empty".to_string()));
                }
                
                // Check for potentially dangerous headers
                let key_lower = key.to_lowercase();
                if matches!(key_lower.as_str(), "authorization" | "cookie" | "x-forwarded-for") {
                    warn!("Potentially sensitive header detected: {}", key);
                }
            }
        }

        Ok(())
    }

    /// Execute an HTTP request
    async fn execute_request(&self, request: HttpRequest) -> Result<HttpResponseData> {
        let start_time = std::time::Instant::now();
        
        // Parse method
        let method = Method::from_bytes(request.method.to_uppercase().as_bytes())
            .map_err(|e| AgentError::validation(format!("Invalid HTTP method: {}", e)))?;

        // Build request
        let mut req_builder = self.client.request(method, &request.url);

        // Add headers
        if let Some(headers) = &request.headers {
            for (key, value) in headers {
                req_builder = req_builder.header(key, value);
            }
        }

        // Add query parameters
        if let Some(query_params) = &request.query_params {
            req_builder = req_builder.query(query_params);
        }

        // Add body for applicable methods
        if let Some(body) = &request.body {
            req_builder = req_builder.body(body.clone());
        }

        // Apply custom timeout if specified
        if let Some(timeout) = request.timeout {
            req_builder = req_builder.timeout(Duration::from_secs(timeout));
        }

        debug!("Executing HTTP request: {} {}", request.method, request.url);

        // Execute request
        let response = req_builder.send().await
            .map_err(|e| AgentError::tool("http_client", &format!("Request failed: {}", e)))?;

        let status_code = response.status().as_u16();
        let final_url = response.url().to_string();
        
        // Extract headers
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                headers.insert(key.to_string(), value_str.to_string());
            }
        }

        // Check content length
        if let Some(content_length) = response.content_length() {
            if content_length as usize > self.max_response_size {
                return Err(AgentError::tool(
                    "http_client",
                    &format!("Response too large: {} bytes (max: {})", content_length, self.max_response_size)
                ));
            }
        }

        // Read response body with size limit
        let body_bytes = response.bytes().await
            .map_err(|e| AgentError::tool("http_client", &format!("Failed to read response body: {}", e)))?;

        if body_bytes.len() > self.max_response_size {
            return Err(AgentError::tool(
                "http_client",
                &format!("Response too large: {} bytes (max: {})", body_bytes.len(), self.max_response_size)
            ));
        }

        let body = String::from_utf8_lossy(&body_bytes).to_string();
        let size_bytes = body_bytes.len();
        let duration_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "HTTP request completed: {} {} -> {} ({} bytes, {}ms)",
            request.method, request.url, status_code, size_bytes, duration_ms
        );

        Ok(HttpResponseData {
            status_code,
            headers,
            body,
            size_bytes,
            duration_ms,
            final_url,
        })
    }

    /// Make a simple GET request
    pub async fn get(&self, url: &str) -> Result<HttpResponseData> {
        let request = HttpRequest {
            method: "GET".to_string(),
            url: url.to_string(),
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        self.validate_request(&request)?;
        self.execute_request(request).await
    }

    /// Make a POST request with JSON body
    pub async fn post_json(&self, url: &str, json_body: &Value) -> Result<HttpResponseData> {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        let request = HttpRequest {
            method: "POST".to_string(),
            url: url.to_string(),
            headers: Some(headers),
            body: Some(json_body.to_string()),
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        self.validate_request(&request)?;
        self.execute_request(request).await
    }

    /// Get tool configuration summary
    pub fn get_config_summary(&self) -> Value {
        json!({
            "max_response_size": self.max_response_size,
            "allowed_domains": self.allowed_domains,
            "timeout_seconds": self.timeout_seconds,
            "max_redirects": self.max_redirects,
            "verify_ssl": self.verify_ssl
        })
    }
}

#[async_trait]
impl Tool for HttpClientTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "custom".to_string(),
            name: "http_client".to_string(),
            description: Some("Make HTTP requests to web APIs and services with security validation".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
                    },
                    "url": {
                        "type": "string",
                        "description": "Target URL for the request"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs",
                        "additionalProperties": {"type": "string"}
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST, PUT, etc.)"
                    },
                    "query_params": {
                        "type": "object",
                        "description": "Optional query parameters as key-value pairs",
                        "additionalProperties": {"type": "string"}
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional request timeout in seconds"
                    },
                    "follow_redirects": {
                        "type": "boolean",
                        "description": "Whether to follow HTTP redirects"
                    }
                },
                "required": ["method", "url"]
            })),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    fn name(&self) -> &str {
        "http_client"
    }

    fn description(&self) -> Option<&str> {
        Some("Make HTTP requests to web APIs and services with security validation")
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let request: HttpRequest = serde_json::from_value(input)
            .map_err(|e| AgentError::invalid_input(format!("Invalid HTTP request format: {}", e)))?;

        // Validate request
        self.validate_request(&request)?;

        // Execute request
        match self.execute_request(request).await {
            Ok(response_data) => {
                let result_json = serde_json::to_value(&response_data)
                    .map_err(|e| AgentError::tool("http_client", &format!("Failed to serialize response: {}", e)))?;

                let mut metadata = HashMap::new();
                metadata.insert("response_data".to_string(), result_json);

                Ok(ToolResult {
                    content: format!(
                        "HTTP {} request completed successfully.\nStatus: {}\nSize: {} bytes\nDuration: {}ms\nResponse: {}",
                        response_data.status_code,
                        if response_data.status_code < 400 { "Success" } else { "Error" },
                        response_data.size_bytes,
                        response_data.duration_ms,
                        if response_data.body.len() > 1000 {
                            format!("{}... (truncated)", &response_data.body[..1000])
                        } else {
                            response_data.body.clone()
                        }
                    ),
                    is_error: response_data.status_code >= 400,
                    metadata: Some(metadata),
                })
            }
            Err(e) => Ok(ToolResult {
                content: format!("HTTP request failed: {}", e),
                is_error: true,
                metadata: None,
            })
        }
    }
}

impl Default for HttpClientTool {
    fn default() -> Self {
        Self::new().expect("Failed to create default HTTP client tool")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio;

    #[tokio::test]
    async fn test_http_client_creation() {
        let client = HttpClientTool::new().unwrap();
        assert_eq!(client.max_response_size, 10 * 1024 * 1024);
        assert_eq!(client.timeout_seconds, 30);
        assert!(client.verify_ssl);
    }

    #[tokio::test]
    async fn test_http_client_with_config() {
        let config = HttpClientConfig {
            max_response_size: 1024,
            timeout_seconds: 10,
            verify_ssl: false,
            allowed_domains: vec!["example.com".to_string()],
            ..Default::default()
        };

        let client = HttpClientTool::with_config(config).unwrap();
        assert_eq!(client.max_response_size, 1024);
        assert_eq!(client.timeout_seconds, 10);
        assert!(!client.verify_ssl);
        assert_eq!(client.allowed_domains.len(), 1);
    }

    #[tokio::test]
    async fn test_http_client_with_allowed_domains() {
        let domains = vec!["api.github.com".to_string(), "httpbin.org".to_string()];
        let client = HttpClientTool::with_allowed_domains(domains.clone()).unwrap();
        assert_eq!(client.allowed_domains, domains);
    }

    #[test]
    fn test_validate_request_valid() {
        let client = HttpClientTool::new().unwrap();
        let request = HttpRequest {
            method: "GET".to_string(),
            url: "https://httpbin.org/get".to_string(),
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        assert!(client.validate_request(&request).is_ok());
    }

    #[test]
    fn test_validate_request_invalid_url() {
        let client = HttpClientTool::new().unwrap();
        let request = HttpRequest {
            method: "GET".to_string(),
            url: "http://localhost:8080".to_string(), // Should be blocked by validation
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        assert!(client.validate_request(&request).is_err());
    }

    #[test]
    fn test_validate_request_invalid_method() {
        let client = HttpClientTool::new().unwrap();
        let request = HttpRequest {
            method: "INVALID".to_string(),
            url: "https://httpbin.org/get".to_string(),
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        assert!(client.validate_request(&request).is_err());
    }

    #[test]
    fn test_validate_request_domain_restriction() {
        let client = HttpClientTool::with_allowed_domains(vec!["example.com".to_string()]).unwrap();

        // Should pass for allowed domain
        let valid_request = HttpRequest {
            method: "GET".to_string(),
            url: "https://api.example.com/data".to_string(),
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };
        assert!(client.validate_request(&valid_request).is_ok());

        // Should fail for disallowed domain
        let invalid_request = HttpRequest {
            method: "GET".to_string(),
            url: "https://httpbin.org/get".to_string(),
            headers: None,
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };
        assert!(client.validate_request(&invalid_request).is_err());
    }

    #[test]
    fn test_validate_request_empty_headers() {
        let client = HttpClientTool::new().unwrap();
        let mut headers = HashMap::new();
        headers.insert("".to_string(), "value".to_string());

        let request = HttpRequest {
            method: "GET".to_string(),
            url: "https://httpbin.org/get".to_string(),
            headers: Some(headers),
            body: None,
            query_params: None,
            timeout: None,
            follow_redirects: None,
        };

        assert!(client.validate_request(&request).is_err());
    }

    #[tokio::test]
    async fn test_tool_interface() {
        let client = HttpClientTool::new().unwrap();

        assert_eq!(client.name(), "http_client");
        assert!(client.description().is_some());

        let definition = client.definition();
        assert_eq!(definition.name, "http_client");
        assert!(definition.input_schema.is_some());

        let schema = definition.input_schema.unwrap();
        assert!(schema["properties"]["method"].is_object());
        assert!(schema["properties"]["url"].is_object());
    }

    #[tokio::test]
    async fn test_tool_execute_invalid_input() {
        let client = HttpClientTool::new().unwrap();
        let invalid_input = json!({"invalid": "data"});

        let result = client.execute(invalid_input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid HTTP request format"));
    }

    #[tokio::test]
    async fn test_tool_execute_invalid_url() {
        let client = HttpClientTool::new().unwrap();
        let input = json!({
            "method": "GET",
            "url": "http://localhost:8080"
        });

        let result = client.execute(input).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_config_summary() {
        let client = HttpClientTool::new().unwrap();
        let summary = client.get_config_summary();

        assert!(summary.is_object());
        assert!(summary["max_response_size"].is_number());
        assert!(summary["timeout_seconds"].is_number());
        assert!(summary["verify_ssl"].is_boolean());
    }

    #[test]
    fn test_default_config() {
        let config = HttpClientConfig::default();
        assert_eq!(config.max_response_size, 10 * 1024 * 1024);
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.max_redirects, 10);
        assert!(config.verify_ssl);
        assert!(config.allowed_domains.is_empty());
        assert_eq!(config.user_agent, "RustMemVidAgent/1.0");
    }

    #[test]
    fn test_http_request_serialization() {
        let request = HttpRequest {
            method: "POST".to_string(),
            url: "https://httpbin.org/post".to_string(),
            headers: Some({
                let mut h = HashMap::new();
                h.insert("Content-Type".to_string(), "application/json".to_string());
                h
            }),
            body: Some(r#"{"test": "data"}"#.to_string()),
            query_params: Some({
                let mut q = HashMap::new();
                q.insert("param1".to_string(), "value1".to_string());
                q
            }),
            timeout: Some(60),
            follow_redirects: Some(true),
        };

        let json_value = serde_json::to_value(&request).unwrap();
        assert!(json_value.is_object());

        let deserialized: HttpRequest = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized.method, "POST");
        assert_eq!(deserialized.url, "https://httpbin.org/post");
    }

    #[test]
    fn test_http_response_data_serialization() {
        let response = HttpResponseData {
            status_code: 200,
            headers: {
                let mut h = HashMap::new();
                h.insert("content-type".to_string(), "application/json".to_string());
                h
            },
            body: r#"{"success": true}"#.to_string(),
            size_bytes: 17,
            duration_ms: 150,
            final_url: "https://httpbin.org/get".to_string(),
        };

        let json_value = serde_json::to_value(&response).unwrap();
        assert!(json_value.is_object());

        let deserialized: HttpResponseData = serde_json::from_value(json_value).unwrap();
        assert_eq!(deserialized.status_code, 200);
        assert_eq!(deserialized.size_bytes, 17);
    }

    // Integration tests (these would require actual network access)
    // Commented out to avoid network dependencies in unit tests

    /*
    #[tokio::test]
    async fn test_real_get_request() {
        let client = HttpClientTool::new().unwrap();
        let response = client.get("https://httpbin.org/get").await.unwrap();

        assert_eq!(response.status_code, 200);
        assert!(!response.body.is_empty());
        assert!(response.duration_ms > 0);
    }

    #[tokio::test]
    async fn test_real_post_json_request() {
        let client = HttpClientTool::new().unwrap();
        let json_data = json!({"test": "data", "number": 42});

        let response = client.post_json("https://httpbin.org/post", &json_data).await.unwrap();

        assert_eq!(response.status_code, 200);
        assert!(response.body.contains("test"));
        assert!(response.body.contains("data"));
    }
    */
}
