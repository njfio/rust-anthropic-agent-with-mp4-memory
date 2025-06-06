use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::process::Command;
use tracing::{debug, info, warn};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};

/// A simple shell command execution tool
#[derive(Debug, Clone)]
pub struct ShellCommandTool {
    /// Whether to allow potentially dangerous commands
    allow_dangerous: bool,
    /// List of blocked commands
    blocked_commands: Vec<String>,
    /// Working directory for commands
    working_dir: Option<std::path::PathBuf>,
}

impl ShellCommandTool {
    /// Create a new shell command tool
    pub fn new() -> Self {
        Self {
            allow_dangerous: false,
            blocked_commands: vec![
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
        }
    }

    /// Allow dangerous commands (use with caution!)
    pub fn with_dangerous_commands(mut self) -> Self {
        self.allow_dangerous = true;
        self
    }

    /// Set working directory
    pub fn with_working_dir<P: Into<std::path::PathBuf>>(mut self, dir: P) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Add a blocked command
    pub fn block_command<S: Into<String>>(mut self, command: S) -> Self {
        self.blocked_commands.push(command.into());
        self
    }

    /// Check if a command is safe to execute
    fn is_command_safe(&self, command: &str) -> bool {
        if self.allow_dangerous {
            return true;
        }

        let command_parts: Vec<&str> = command.split_whitespace().collect();
        if let Some(cmd) = command_parts.first() {
            !self.blocked_commands.iter().any(|blocked| cmd.contains(blocked))
        } else {
            false
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
            return Ok(ToolResult::error("Command blocked for security reasons"));
        }

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

        match cmd.output() {
            Ok(output) => {
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
            Err(e) => {
                warn!("Failed to execute shell command: {}", e);
                Ok(ToolResult::error(format!("Failed to execute command: {}", e)))
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
    fn is_url_allowed(&self, url: &str) -> bool {
        if let Some(allowed) = &self.allowed_domains {
            if let Ok(parsed_url) = url::Url::parse(url) {
                if let Some(domain) = parsed_url.domain() {
                    return allowed.iter().any(|allowed_domain| domain.contains(allowed_domain));
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
            return Ok(ToolResult::error("URL not allowed"));
        }

        let mut request = match method.to_uppercase().as_str() {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "PUT" => self.client.put(&url),
            "DELETE" => self.client.delete(&url),
            _ => return Ok(ToolResult::error("Unsupported HTTP method")),
        };

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
