use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};
use crate::utils::audit_logger::{AuditEvent, AuditEventType, AuditSeverity, audit_log};
use crate::utils::security_headers::SecurityHeaders;

/// A secure shell command execution tool with allowlist-based filtering
#[derive(Debug, Clone)]
pub struct ShellCommandTool {
    /// Whether to use allowlist (secure) or blocklist (legacy) mode
    use_allowlist: bool,
    /// List of allowed commands (when use_allowlist is true)
    allowed_commands: Vec<String>,
    /// List of blocked commands (legacy mode only)
    blocked_commands: Vec<String>,
    /// Working directory for commands
    working_dir: Option<std::path::PathBuf>,
    /// Command timeout in seconds
    timeout_seconds: u64,
}

impl ShellCommandTool {
    /// Create a new secure shell command tool with allowlist (recommended)
    pub fn new() -> Self {
        Self {
            use_allowlist: true,
            // SECURITY: Default to safe, read-only commands only
            allowed_commands: vec![
                "ls".to_string(),
                "dir".to_string(),
                "pwd".to_string(),
                "echo".to_string(),
                "cat".to_string(),
                "head".to_string(),
                "tail".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "wc".to_string(),
                "sort".to_string(),
                "uniq".to_string(),
                "date".to_string(),
                "whoami".to_string(),
                "id".to_string(),
                "uname".to_string(),
                "which".to_string(),
                "type".to_string(),
            ],
            blocked_commands: vec![
                // Legacy blocklist for backward compatibility
                "rm".to_string(),
                "rmdir".to_string(),
                "del".to_string(),
                "format".to_string(),
                "sudo".to_string(),
                "su".to_string(),
                "chmod".to_string(),
                "chown".to_string(),
            ],
            working_dir: None,
            timeout_seconds: 30, // Default 30 second timeout
        }
    }

    /// Switch to legacy blocklist mode (less secure, use with caution!)
    pub fn with_blocklist_mode(mut self) -> Self {
        self.use_allowlist = false;
        self
    }

    /// Add an allowed command (only works in allowlist mode)
    pub fn allow_command<S: Into<String>>(mut self, command: S) -> Self {
        self.allowed_commands.push(command.into());
        self
    }

    /// Set working directory
    pub fn with_working_dir<P: Into<std::path::PathBuf>>(mut self, dir: P) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set command timeout in seconds
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Add a blocked command
    pub fn block_command<S: Into<String>>(mut self, command: S) -> Self {
        self.blocked_commands.push(command.into());
        self
    }

    /// Check if a command is safe to execute
    /// SECURITY: Uses allowlist by default for maximum security
    fn is_command_safe(&self, command: &str) -> bool {
        // Parse command to get the base command
        let command_parts: Vec<&str> = command.split_whitespace().collect();
        let base_command = match command_parts.first() {
            Some(cmd) => cmd,
            None => return false,
        };

        // Check for shell operators that could be used for command injection
        if command.contains(';') || command.contains('&') || command.contains('|') ||
           command.contains('`') || command.contains('$') || command.contains('>') ||
           command.contains('<') {
            return false;
        }

        if self.use_allowlist {
            // SECURITY: Allowlist mode - only explicitly allowed commands
            self.allowed_commands.iter().any(|allowed| base_command == allowed)
        } else {
            // Legacy blocklist mode - less secure
            !self.blocked_commands.iter().any(|blocked| base_command.contains(blocked))
        }
    }
}

impl Default for ShellCommandTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ShellCommandTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "shell_command",
            "Execute a shell command and return the output",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let command = extract_string_param(&input, "command")?;

        debug!("Executing shell command: {}", command);

        if !self.is_command_safe(&command) {
            // AUDIT: Log blocked command attempt
            audit_log(AuditEvent::new(
                AuditEventType::SecurityViolation,
                AuditSeverity::High,
                "blocked_command_execution".to_string(),
            ).with_resource(&command).with_success(false).with_error("Command blocked by security policy"));

            return Ok(ToolResult::error("Command blocked for security reasons"));
        }

        // AUDIT: Log command execution attempt
        audit_log(AuditEvent::new(
            AuditEventType::CommandExecution,
            AuditSeverity::Medium,
            "shell_command_execution".to_string(),
        ).with_resource(&command));

        let mut cmd = if cfg!(target_os = "windows") {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", &command]);
            cmd
        } else {
            let mut cmd = Command::new("sh");
            cmd.args(["-c", &command]);
            cmd
        };

        if let Some(working_dir) = &self.working_dir {
            cmd.current_dir(working_dir);
        }

        // Execute command with timeout
        let timeout_duration = Duration::from_secs(self.timeout_seconds);
        let output_future = tokio::task::spawn_blocking(move || cmd.output());

        match timeout(timeout_duration, output_future).await {
            Ok(Ok(Ok(output))) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let result = if output.status.success() {
                    if stdout.is_empty() && stderr.is_empty() {
                        "Command executed successfully (no output)".to_string()
                    } else if stderr.is_empty() {
                        stdout.to_string()
                    } else {
                        format!("STDOUT:\n{}\n\nSTDERR:\n{}", stdout, stderr)
                    }
                } else {
                    format!("Command failed with exit code: {:?}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                           output.status.code(), stdout, stderr)
                };

                info!("Shell command executed: {} (success: {})", command, output.status.success());
                Ok(ToolResult::success(result))
            }
            Ok(Ok(Err(e))) => {
                warn!("Failed to execute shell command: {}", e);
                Ok(ToolResult::error(format!("Failed to execute command: {}", e)))
            }
            Ok(Err(e)) => {
                warn!("Shell command task failed: {}", e);
                Ok(ToolResult::error(format!("Command execution task failed: {}", e)))
            }
            Err(_) => {
                warn!("Shell command timed out after {} seconds: {}", self.timeout_seconds, command);
                Ok(ToolResult::error(format!("Command timed out after {} seconds", self.timeout_seconds)))
            }
        }
    }

    fn name(&self) -> &str {
        "shell_command"
    }

    fn description(&self) -> Option<&str> {
        Some("Execute a shell command and return the output")
    }
}

/// A tool for making HTTP requests
#[derive(Debug, Clone)]
pub struct HttpRequestTool {
    /// HTTP client
    client: reqwest::Client,
    /// Allowed domains (None means all allowed)
    allowed_domains: Option<Vec<String>>,
    /// Request timeout in seconds
    timeout_seconds: u64,
    /// Security headers configuration
    security_headers: SecurityHeaders,
}

impl HttpRequestTool {
    /// Create a new HTTP request tool
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AgentError::config(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            allowed_domains: None,
            timeout_seconds: 30,
            security_headers: SecurityHeaders::new(),
        })
    }

    /// Set allowed domains
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Check if a URL is allowed
    /// SECURITY: Uses exact domain matching to prevent subdomain bypass attacks
    fn is_url_allowed(&self, url: &str) -> bool {
        if let Some(allowed) = &self.allowed_domains {
            if let Ok(parsed_url) = url::Url::parse(url) {
                if let Some(domain) = parsed_url.domain() {
                    // SECURITY FIX: Use exact matching or proper subdomain validation
                    return allowed.iter().any(|allowed_domain| {
                        // Exact match
                        domain == allowed_domain ||
                        // Subdomain match (must end with .allowed_domain)
                        (domain.ends_with(&format!(".{}", allowed_domain)) &&
                         allowed_domain.contains('.')) // Only allow subdomain matching for proper domains
                    });
                }
            }
            false
        } else {
            true
        }
    }
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "http_request",
            "Make an HTTP request and return the response",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to make the request to"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, DELETE)",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST/PUT)"
                    }
                },
                "required": ["url"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let url = extract_string_param(&input, "url")?;
        let method = input
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET");

        debug!("Making HTTP request: {} {}", method, url);

        if !self.is_url_allowed(&url) {
            // AUDIT: Log blocked URL attempt
            audit_log(AuditEvent::new(
                AuditEventType::SecurityViolation,
                AuditSeverity::High,
                "blocked_http_request".to_string(),
            ).with_resource(&url).with_success(false).with_error("URL not in allowed domains"));

            return Ok(ToolResult::error("URL not allowed"));
        }

        // AUDIT: Log HTTP request attempt
        audit_log(AuditEvent::new(
            AuditEventType::NetworkRequest,
            AuditSeverity::Medium,
            format!("http_request_{}", method.to_lowercase()),
        ).with_resource(&url));

        let mut request = match method.to_uppercase().as_str() {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "DELETE" => self.client.delete(&url),
            _ => return Ok(ToolResult::error("Unsupported HTTP method")),
        };

        // Add security headers
        let security_headers = self.security_headers.build_request_headers();
        for (name, value) in security_headers.iter() {
            request = request.header(name, value);
        }

        // Add headers if provided
        if let Some(headers) = input.get("headers").and_then(|v| v.as_object()) {
            for (key, value) in headers {
                if let Some(value_str) = value.as_str() {
                    request = request.header(key, value_str);
                }
            }
        }

        // Add body if provided
        if let Some(body) = input.get("body").and_then(|v| v.as_str()) {
            request = request.body(body.to_string());
        }

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let headers: HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();

                match response.text().await {
                    Ok(body) => {
                        let result = format!(
                            "Status: {}\nHeaders: {}\n\nBody:\n{}",
                            status,
                            serde_json::to_string_pretty(&headers).unwrap_or_default(),
                            body
                        );

                        info!("HTTP request completed: {} {} (status: {})", method, url, status);
                        Ok(ToolResult::success(result))
                    }
                    Err(e) => Ok(ToolResult::error(format!("Failed to read response body: {}", e))),
                }
            }
            Err(e) => {
                warn!("HTTP request failed: {}", e);
                Ok(ToolResult::error(format!("HTTP request failed: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> Option<&str> {
        Some("Make an HTTP request and return the response")
    }
}

/// A tool for generating UUIDs
#[derive(Debug, Clone, Default)]
pub struct UuidGeneratorTool;

#[async_trait]
impl Tool for UuidGeneratorTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "generate_uuid",
            "Generate a new UUID",
            json!({
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "description": "UUID version (v4 is default)",
                        "enum": ["v4"],
                        "default": "v4"
                    }
                },
                "required": []
            }),
        )
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolResult> {
        let uuid = uuid::Uuid::new_v4();
        debug!("Generated UUID: {}", uuid);
        Ok(ToolResult::success(uuid.to_string()))
    }

    fn name(&self) -> &str {
        "generate_uuid"
    }

    fn description(&self) -> Option<&str> {
        Some("Generate a new UUID")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_uuid_generator_tool() {
        let tool = UuidGeneratorTool::default();
        let result = tool.execute(json!({})).await.unwrap();
        
        assert!(!result.is_error);
        assert!(uuid::Uuid::parse_str(&result.content).is_ok());
    }

    #[tokio::test]
    async fn test_shell_command_tool_safe() {
        let tool = ShellCommandTool::new();
        let input = json!({"command": "echo hello"});
        
        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_shell_command_tool_blocked() {
        let tool = ShellCommandTool::new();
        let input = json!({"command": "rm -rf /"});
        
        let result = tool.execute(input).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("blocked"));
    }
}
