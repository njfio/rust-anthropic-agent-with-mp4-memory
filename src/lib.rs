//! # Rust MemVid Agent
//!
//! A comprehensive AI agent system in Rust that integrates with Anthropic's Claude API
//! and uses JSON-based synaptic memory for persistent conversations and context.
//!
//! ## Features
//!
//! - **Full Anthropic API Integration**: Support for all latest tools including code execution,
//!   web search, and text editor tools
//! - **Synaptic JSON Memory**: Persistent memory using the synaptic library for conversation history
//!   and semantic search
//! - **Extensible Tool System**: Built-in tools and framework for custom tool development
//! - **Async/Await Architecture**: High-performance async operations throughout
//! - **Type Safety**: Comprehensive type system for API interactions and tool definitions
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rust_memvid_agent::{Agent, AgentConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize the agent with memory
//!     let config = AgentConfig::default()
//!         .with_anthropic_key("your-api-key")
//!         .with_memory_path("agent_memory.json");
//!     
//!     let mut agent = Agent::new(config).await?;
//!     
//!     // Start a conversation
//!     let response = agent.chat("Hello! Can you help me write some Rust code?").await?;
//!     println!("{}", response);
//!     
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod anthropic;
pub mod cli;
pub mod compliance;
pub mod config;
pub mod memory;
pub mod plugins;
pub mod security;
pub mod tools;
pub mod utils;

// Re-export main types for convenience
pub use agent::{Agent, AgentBuilder};
pub use config::{AgentConfig, AnthropicConfig, MemoryConfig};
pub use utils::error::{AgentError, Result};

// Re-export tool types
pub use tools::{Tool, ToolRegistry, ToolResult};

// Re-export memory types
pub use memory::{MemoryEntry, MemoryManager, MemoryStats, SearchResult};

/// Initialize the agent system with default logging
pub async fn init() -> Result<()> {
    utils::logging::init_logging()?;
    Ok(())
}

/// Initialize the agent system with custom logging configuration
pub async fn init_with_logging(level: tracing::Level) -> Result<()> {
    utils::logging::init_logging_with_level(level)?;
    Ok(())
}

#[cfg(test)]
mod security_tests {

    use crate::tools::custom_tools::ShellCommandTool;
    use crate::tools::local_file_ops::LocalTextEditorTool;
    use crate::tools::Tool;
    use crate::utils::rate_limiter::{RateLimitConfig, RateLimiter};
    use crate::utils::validation::{validate_command, validate_path, validate_url};
    use serde_json::json;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Test path traversal prevention in file operations
    #[tokio::test]
    async fn test_path_traversal_prevention() {
        let temp_dir = TempDir::new().unwrap();
        let tool =
            LocalTextEditorTool::new(temp_dir.path().to_path_buf()).with_max_file_size(1024 * 1024);

        // Test absolute path rejection
        let input = json!({
            "command": "view",
            "path": "/etc/passwd"
        });
        let result = tool.execute(input).await;
        // Should either fail during execution or return an error result
        match result {
            Ok(tool_result) => {
                assert!(tool_result.is_error);
                assert!(
                    tool_result.content.contains("Absolute paths not allowed")
                        || tool_result.content.contains("Invalid input")
                );
            }
            Err(_) => {
                // This is also acceptable - the validation caught it early
            }
        }

        // Test directory traversal rejection
        let input = json!({
            "command": "view",
            "path": "../../../etc/passwd"
        });
        let result = tool.execute(input).await;
        // Should either fail during execution or return an error result
        match result {
            Ok(tool_result) => {
                assert!(tool_result.is_error);
                assert!(
                    tool_result.content.contains("Path traversal not allowed")
                        || tool_result.content.contains("Invalid input")
                );
            }
            Err(_) => {
                // This is also acceptable - the validation caught it early
            }
        }
    }

    /// Test command injection prevention in shell tool
    #[tokio::test]
    async fn test_command_injection_prevention() {
        let tool = ShellCommandTool::new(); // Uses allowlist by default

        // Test command injection attempts
        let dangerous_commands = vec![
            "ls; rm -rf /",
            "echo hello && rm -rf /",
            "ls | grep secret",
            "$(whoami)",
            "ls `whoami`",
            "ls > /etc/passwd",
            "ls < /etc/passwd",
        ];

        for cmd in dangerous_commands {
            let input = json!({"command": cmd});
            let result = tool.execute(input).await.unwrap();
            assert!(result.is_error, "Command should be blocked: {}", cmd);
        }

        // Test allowed command
        let input = json!({"command": "echo hello"});
        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
    }

    /// Test input validation functions
    #[test]
    fn test_input_validation() {
        // Test path validation
        assert!(validate_path("src/main.rs").is_ok());
        assert!(validate_path("../etc/passwd").is_err());
        assert!(validate_path("/etc/passwd").is_err());
        assert!(validate_path("file\0name").is_err());
        assert!(validate_path("").is_err());

        // Test command validation
        assert!(validate_command("ls -la").is_ok());
        assert!(validate_command("echo hello").is_ok());
        assert!(validate_command("rm -rf /; echo done").is_err());
        assert!(validate_command("ls | grep test").is_err());
        assert!(validate_command("$(whoami)").is_err());
        assert!(validate_command("").is_err());

        // Test URL validation
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://api.github.com/repos").is_ok());
        assert!(validate_url("http://localhost:8080").is_err());
        assert!(validate_url("http://127.0.0.1").is_err());
        assert!(validate_url("http://192.168.1.1").is_err());
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("").is_err());
    }

    /// Test rate limiting functionality
    #[tokio::test]
    async fn test_rate_limiting() {
        let config = RateLimitConfig {
            max_requests: 3,
            window_duration: Duration::from_secs(1),
            per_tool_limiting: true,
        };
        let limiter = RateLimiter::new(config);

        // Test global rate limiting
        assert!(limiter.check_global_rate_limit().is_ok());
        assert!(limiter.check_global_rate_limit().is_ok());
        assert!(limiter.check_global_rate_limit().is_ok());
        assert!(limiter.check_global_rate_limit().is_err()); // Should fail

        // Reset and test tool-specific rate limiting
        limiter.reset().unwrap();

        // Tool limit is max_requests / 2 = 1 (minimum 1)
        assert!(limiter.check_tool_rate_limit("test_tool").is_ok());
        assert!(limiter.check_tool_rate_limit("test_tool").is_err()); // Should fail

        // Different tool should still work
        assert!(limiter.check_tool_rate_limit("other_tool").is_ok());
    }

    /// Test SSRF prevention in HTTP requests
    #[test]
    fn test_ssrf_prevention() {
        // Test localhost URLs
        assert!(validate_url("http://localhost:8080").is_err());
        assert!(validate_url("http://127.0.0.1:8080").is_err());
        assert!(validate_url("http://[::1]:8080").is_err());

        // Test private IP ranges
        assert!(validate_url("http://192.168.1.1").is_err());
        assert!(validate_url("http://10.0.0.1").is_err());
        assert!(validate_url("http://172.16.0.1").is_err());
        assert!(validate_url("http://169.254.1.1").is_err()); // Link-local

        // Test public URLs (should work)
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("https://8.8.8.8").is_ok()); // Google DNS
    }

    /// Test audit logging functionality
    #[tokio::test]
    async fn test_audit_logging() {
        use crate::utils::audit_logger::*;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.log");

        let config = AuditLoggerConfig {
            log_file_path: log_path.clone(),
            max_file_size: 1024,
            max_files: 3,
            buffer_size: 10,
            sync_interval_seconds: 1,
            minimum_severity: AuditSeverity::Low,
        };

        let logger = AuditLogger::new(config).unwrap();

        // Test logging an event
        let event = AuditEvent::new(
            AuditEventType::FileAccess,
            AuditSeverity::Medium,
            "test_file_access".to_string(),
        )
        .with_resource("test.txt");

        logger.log_event(event).unwrap();
        logger.flush().unwrap();

        // Verify log file was created and contains data
        assert!(log_path.exists());
        let log_content = std::fs::read_to_string(&log_path).unwrap();
        assert!(log_content.contains("test_file_access"));
        assert!(log_content.contains("test.txt"));
    }

    /// Test resource monitoring
    #[tokio::test]
    async fn test_resource_monitoring() {
        use crate::utils::resource_monitor::*;
        use std::time::Duration;

        let config = ResourceMonitorConfig {
            limits: ResourceLimits {
                max_memory_bytes: 1024 * 1024 * 1024, // 1GB
                max_memory_percentage: 50.0,
                max_cpu_percentage: 90.0,
                max_threads: 50,
                memory_warning_threshold: 0.8,
                cpu_warning_threshold: 0.8,
            },
            monitoring_interval: Duration::from_millis(100),
            enforce_limits: true,
            log_usage: false,
            audit_violations: false,
        };

        let monitor = ResourceMonitor::new(config);
        monitor.start_monitoring().unwrap();

        // Wait for monitoring to collect some data
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = monitor.get_stats().unwrap();
        // uptime_seconds is u64, so it's always >= 0
        assert!(stats.uptime_seconds < u64::MAX);
        assert!(stats.memory_usage > 0);

        monitor.stop_monitoring().unwrap();
    }

    /// Test penetration testing framework (placeholder until security module is enabled)
    #[tokio::test]
    async fn test_penetration_testing_placeholder() {
        // Placeholder test - will be implemented when security module is enabled
        assert!(true);
    }

    /// Test security headers functionality
    #[test]
    fn test_security_headers() {
        use crate::utils::security_headers::*;

        let headers = SecurityHeaders::new();
        let request_headers = headers.build_request_headers();
        let response_headers = headers.build_response_headers();

        // Verify request headers
        assert!(request_headers.contains_key("User-Agent"));
        assert!(request_headers.contains_key("Cache-Control"));

        // Verify response headers
        assert!(response_headers.contains_key("Content-Security-Policy"));
        assert!(response_headers.contains_key("X-Frame-Options"));
        assert!(response_headers.contains_key("X-Content-Type-Options"));

        // Test header validation
        let issues = headers.validate_response_headers(&response_headers);
        assert!(issues.is_empty()); // Should have no issues with our headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init() {
        init().await.unwrap();
    }
}
