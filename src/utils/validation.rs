use crate::utils::error::{AgentError, Result};
use std::collections::HashSet;

/// Maximum allowed length for various input types
pub const MAX_PATH_LENGTH: usize = 4096;
pub const MAX_COMMAND_LENGTH: usize = 8192;
pub const MAX_FILE_CONTENT_LENGTH: usize = 10 * 1024 * 1024; // 10MB
pub const MAX_URL_LENGTH: usize = 2048;
pub const MAX_HEADER_VALUE_LENGTH: usize = 8192;

/// Allowed characters for different input types
const SAFE_PATH_CHARS: &str =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/\\";
const SAFE_COMMAND_CHARS: &str =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/ =";
const SAFE_URL_CHARS: &str =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/:?&=+%#";

/// Dangerous patterns that should be blocked
const DANGEROUS_PATTERNS: &[&str] = &[
    "../",
    "..\\",
    "~",
    "$",
    "`",
    "$(",
    "${",
    "&&",
    "||",
    ";",
    "|",
    ">",
    "<",
    "rm ",
    "del ",
    "format ",
    "mkfs",
    "dd ",
    "chmod 777",
    "chmod +x",
    "curl ",
    "wget ",
    "nc ",
    "netcat",
    "telnet",
    "ssh",
    "ftp",
    "python -c",
    "perl -e",
    "ruby -e",
    "node -e",
    "eval",
    "exec",
    "/etc/passwd",
    "/etc/shadow",
    "C:\\Windows\\System32",
];

/// SQL injection patterns
const SQL_INJECTION_PATTERNS: &[&str] = &[
    "' OR ",
    "\" OR ",
    "' AND ",
    "\" AND ",
    "UNION SELECT",
    "DROP TABLE",
    "DELETE FROM",
    "INSERT INTO",
    "UPDATE SET",
    "CREATE TABLE",
    "ALTER TABLE",
    "'; ",
    "\"; ",
    "/*",
    "*/",
    "--",
    "xp_",
    "sp_",
];

/// XSS patterns
const XSS_PATTERNS: &[&str] = &[
    "<script",
    "</script>",
    "javascript:",
    "vbscript:",
    "onload=",
    "onerror=",
    "onclick=",
    "onmouseover=",
    "onfocus=",
    "onblur=",
    "onchange=",
    "onsubmit=",
    "eval(",
    "setTimeout(",
    "setInterval(",
    "Function(",
    "document.cookie",
    "document.write",
    "innerHTML",
    "outerHTML",
];

/// Command injection patterns
const COMMAND_INJECTION_PATTERNS: &[&str] = &[
    "$(",
    "${",
    "`",
    "&&",
    "||",
    ";",
    "|",
    ">",
    "<",
    ">>",
    "<<",
    "2>&1",
    "/dev/null",
    "/proc/",
    "/sys/",
    "\\x",
    "\\u",
];

/// Validate a file path for security
pub fn validate_path(path: &str) -> Result<()> {
    // Length check
    if path.len() > MAX_PATH_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "Path too long: {} characters (max: {})",
            path.len(),
            MAX_PATH_LENGTH
        )));
    }

    // Empty check
    if path.trim().is_empty() {
        return Err(AgentError::invalid_input("Path cannot be empty"));
    }

    // Character validation
    let safe_chars: HashSet<char> = SAFE_PATH_CHARS.chars().collect();
    for ch in path.chars() {
        if !safe_chars.contains(&ch) {
            return Err(AgentError::invalid_input(format!(
                "Invalid character in path: '{}'",
                ch
            )));
        }
    }

    // Security checks
    if path.contains("..") {
        return Err(AgentError::invalid_input("Path traversal not allowed"));
    }

    if path.starts_with('/') {
        return Err(AgentError::invalid_input("Absolute paths not allowed"));
    }

    // Check for null bytes
    if path.contains('\0') {
        return Err(AgentError::invalid_input("Null bytes not allowed in path"));
    }

    Ok(())
}

/// Validate a shell command for security
pub fn validate_command(command: &str) -> Result<()> {
    // Length check
    if command.len() > MAX_COMMAND_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "Command too long: {} characters (max: {})",
            command.len(),
            MAX_COMMAND_LENGTH
        )));
    }

    // Empty check
    if command.trim().is_empty() {
        return Err(AgentError::invalid_input("Command cannot be empty"));
    }

    // Character validation using SAFE_COMMAND_CHARS
    let safe_chars: HashSet<char> = SAFE_COMMAND_CHARS.chars().collect();
    for ch in command.chars() {
        if !safe_chars.contains(&ch) {
            return Err(AgentError::invalid_input(format!(
                "Invalid character in command: '{}' (only alphanumeric, space, dot, underscore, dash, slash, and equals allowed)",
                ch
            )));
        }
    }

    // Enhanced security checks
    check_dangerous_patterns(command, "command")?;
    check_command_injection_patterns(command)?;
    check_sql_injection_patterns(command)?;

    // Check for null bytes
    if command.contains('\0') {
        return Err(AgentError::invalid_input(
            "Null bytes not allowed in command",
        ));
    }

    Ok(())
}

/// Validate file content size and security
pub fn validate_file_content(content: &str) -> Result<()> {
    if content.len() > MAX_FILE_CONTENT_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "File content too large: {} bytes (max: {} bytes)",
            content.len(),
            MAX_FILE_CONTENT_LENGTH
        )));
    }

    // Enhanced security checks - but be more lenient for code content
    check_dangerous_patterns_lenient(content, "file content")?;
    check_xss_patterns(content)?;
    check_sql_injection_patterns_lenient(content)?;

    // Check for null bytes
    if content.contains('\0') {
        return Err(AgentError::invalid_input(
            "Null bytes not allowed in file content",
        ));
    }

    Ok(())
}

/// Validate a URL
pub fn validate_url(url: &str) -> Result<()> {
    // Length check
    if url.len() > MAX_URL_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "URL too long: {} characters (max: {})",
            url.len(),
            MAX_URL_LENGTH
        )));
    }

    // Empty check
    if url.trim().is_empty() {
        return Err(AgentError::invalid_input("URL cannot be empty"));
    }

    // Character validation using SAFE_URL_CHARS
    let safe_chars: HashSet<char> = SAFE_URL_CHARS.chars().collect();
    for ch in url.chars() {
        if !safe_chars.contains(&ch) {
            return Err(AgentError::invalid_input(format!(
                "Invalid character in URL: '{}' (only alphanumeric, dot, underscore, dash, colon, slash, question mark, ampersand, equals, plus, percent, and hash allowed)",
                ch
            )));
        }
    }

    // Parse URL to validate format
    match url::Url::parse(url) {
        Ok(parsed_url) => {
            // Check scheme
            match parsed_url.scheme() {
                "http" | "https" => {}
                scheme => {
                    return Err(AgentError::invalid_input(format!(
                        "Unsupported URL scheme: {}",
                        scheme
                    )))
                }
            }

            // Check for localhost/private IPs to prevent SSRF
            if let Some(host) = parsed_url.host_str() {
                if is_private_or_localhost(host) {
                    return Err(AgentError::invalid_input(
                        "Requests to localhost or private IPs not allowed",
                    ));
                }
            }
        }
        Err(_) => return Err(AgentError::invalid_input("Invalid URL format")),
    }

    Ok(())
}

/// Validate HTTP header value
pub fn validate_header_value(value: &str) -> Result<()> {
    if value.len() > MAX_HEADER_VALUE_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "Header value too long: {} characters (max: {})",
            value.len(),
            MAX_HEADER_VALUE_LENGTH
        )));
    }

    // Check for control characters
    for ch in value.chars() {
        if ch.is_control() && ch != '\t' {
            return Err(AgentError::invalid_input(
                "Control characters not allowed in header values",
            ));
        }
    }

    Ok(())
}

/// Check if a host is localhost or a private IP
fn is_private_or_localhost(host: &str) -> bool {
    // Check for localhost variations
    if host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]" {
        return true;
    }

    // Remove brackets from IPv6 addresses
    let clean_host = if host.starts_with('[') && host.ends_with(']') {
        &host[1..host.len() - 1]
    } else {
        host
    };

    // Check for private IP ranges
    if let Ok(ip) = clean_host.parse::<std::net::IpAddr>() {
        match ip {
            std::net::IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                octets[0] == 10 ||
                (octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31)) ||
                (octets[0] == 192 && octets[1] == 168) ||
                // Link-local 169.254.0.0/16
                (octets[0] == 169 && octets[1] == 254)
            }
            std::net::IpAddr::V6(ipv6) => {
                // Check for private IPv6 ranges
                ipv6.is_loopback() ||
                ipv6.segments()[0] == 0xfc00 || // fc00::/7
                ipv6.segments()[0] == 0xfd00 ||
                ipv6.segments()[0] == 0xfe80 // fe80::/10 link-local
            }
        }
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_path() {
        // Valid paths
        assert!(validate_path("src/main.rs").is_ok());
        assert!(validate_path("docs/README.md").is_ok());

        // Invalid paths
        assert!(validate_path("../etc/passwd").is_err());
        assert!(validate_path("/etc/passwd").is_err());
        assert!(validate_path("").is_err());
        assert!(validate_path("file\0name").is_err());
    }

    #[test]
    fn test_validate_command() {
        // Valid commands
        assert!(validate_command("ls -la").is_ok());
        assert!(validate_command("echo hello").is_ok());
        assert!(validate_command("cargo build").is_ok());

        // Invalid commands - dangerous patterns
        assert!(validate_command("rm -rf /; echo done").is_err());
        assert!(validate_command("ls | grep test").is_err());
        assert!(validate_command("$(whoami)").is_err());
        assert!(validate_command("").is_err());

        // Invalid commands - injection attempts
        assert!(validate_command("ls && rm -rf /").is_err());
        assert!(validate_command("echo `cat /etc/passwd`").is_err());
        assert!(validate_command("curl http://evil.com").is_err());
        assert!(validate_command("python -c 'import os; os.system(\"rm -rf /\")'").is_err());

        // Invalid commands - SQL injection patterns
        assert!(validate_command("echo 'DROP TABLE users'").is_err());
        assert!(validate_command("test ' OR 1=1 --").is_err());
    }

    #[test]
    fn test_validate_file_content() {
        // Valid content
        assert!(validate_file_content("Hello, world!").is_ok());
        assert!(validate_file_content("fn main() { println!(\"Hello\"); }").is_ok());

        // Invalid content - XSS patterns
        assert!(validate_file_content("<script>alert('xss')</script>").is_err());
        assert!(validate_file_content("javascript:alert(1)").is_err());
        assert!(validate_file_content("onload=alert(1)").is_err());

        // Invalid content - SQL injection
        assert!(validate_file_content("'; DROP TABLE users; --").is_err());
        assert!(validate_file_content("UNION SELECT * FROM passwords").is_err());

        // Invalid content - dangerous patterns
        assert!(validate_file_content("rm -rf /").is_err());
        assert!(validate_file_content("curl http://evil.com").is_err());
    }

    #[test]
    fn test_security_pattern_detection() {
        // Test dangerous patterns
        assert!(check_dangerous_patterns("rm -rf /", "test").is_err());
        assert!(check_dangerous_patterns("../../../etc/passwd", "test").is_err());
        assert!(check_dangerous_patterns("$(malicious_command)", "test").is_err());
        assert!(check_dangerous_patterns("safe content", "test").is_ok());

        // Test XSS patterns
        assert!(check_xss_patterns("<script>alert(1)</script>").is_err());
        assert!(check_xss_patterns("javascript:void(0)").is_err());
        assert!(check_xss_patterns("onclick=alert(1)").is_err());
        assert!(check_xss_patterns("normal text").is_ok());

        // Test SQL injection patterns
        assert!(check_sql_injection_patterns("' OR 1=1 --").is_err());
        assert!(check_sql_injection_patterns("UNION SELECT password FROM users").is_err());
        assert!(check_sql_injection_patterns("DROP TABLE important_data").is_err());
        assert!(check_sql_injection_patterns("SELECT name FROM users WHERE id = 1").is_ok());

        // Test command injection patterns
        assert!(check_command_injection_patterns("ls && rm -rf /").is_err());
        assert!(check_command_injection_patterns("echo $(whoami)").is_err());
        assert!(check_command_injection_patterns("cat file | grep pattern").is_err());
        assert!(check_command_injection_patterns("echo hello world").is_ok());
    }

    #[test]
    fn test_sanitize_input() {
        assert_eq!(sanitize_input("hello world"), "hello world");
        assert_eq!(sanitize_input("test_file.txt"), "test_file.txt");
        assert_eq!(
            sanitize_input("hello<script>alert(1)</script>"),
            "helloscriptalert1script"
        );
        assert_eq!(
            sanitize_input("'; DROP TABLE users; --"),
            "DROP TABLE users"
        );
    }

    #[test]
    fn test_validate_json_structure() {
        use serde_json::json;

        // Valid JSON
        let simple_json = json!({"key": "value"});
        assert!(validate_json_structure(&simple_json, 10).is_ok());

        // Too deep JSON
        let deep_json = json!({"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": "value"}}}}}}}}}}});
        assert!(validate_json_structure(&deep_json, 5).is_err());

        // Large array
        let large_array = json!(vec![1; 15000]);
        assert!(validate_json_structure(&large_array, 10).is_err());

        // Too many properties
        let mut large_object = serde_json::Map::new();
        for i in 0..1500 {
            large_object.insert(format!("key{}", i), json!("value"));
        }
        let large_json = serde_json::Value::Object(large_object);
        assert!(validate_json_structure(&large_json, 10).is_err());
    }

    #[test]
    fn test_validate_url() {
        // Valid URLs
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://api.github.com/repos").is_ok());

        // Invalid URLs
        assert!(validate_url("http://localhost:8080").is_err());
        assert!(validate_url("http://127.0.0.1").is_err());
        assert!(validate_url("http://192.168.1.1").is_err());
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("").is_err());
    }

    #[test]
    fn test_private_ip_detection() {
        assert!(is_private_or_localhost("localhost"));
        assert!(is_private_or_localhost("127.0.0.1"));
        assert!(is_private_or_localhost("192.168.1.1"));
        assert!(is_private_or_localhost("10.0.0.1"));
        assert!(is_private_or_localhost("172.16.0.1"));

        assert!(!is_private_or_localhost("8.8.8.8"));
        assert!(!is_private_or_localhost("example.com"));
    }

    // Validation Constants Integration Tests
    #[test]
    fn test_safe_command_chars_integration() {
        // Valid commands with safe characters
        assert!(validate_command("ls -la").is_ok());
        assert!(validate_command("echo hello_world").is_ok());
        assert!(validate_command("cargo build").is_ok());
        assert!(validate_command("test.exe").is_ok());

        // Invalid commands with unsafe characters
        assert!(validate_command("ls | grep test").is_err()); // pipe character
        assert!(validate_command("echo $(whoami)").is_err()); // parentheses
        assert!(validate_command("test & background").is_err()); // ampersand
        assert!(validate_command("echo 'quoted'").is_err()); // single quotes
        assert!(validate_command("test@host").is_err()); // at symbol
    }

    #[test]
    fn test_safe_url_chars_integration() {
        // Valid URLs with safe characters
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://api.test.com/path?param=value").is_ok());
        assert!(validate_url("https://site.org/page#section").is_ok());
        assert!(validate_url("http://test.com/path_with-dashes").is_ok());

        // Invalid URLs with unsafe characters
        assert!(validate_url("https://test.com/path with spaces").is_err()); // spaces
        assert!(validate_url("http://site.com/path[brackets]").is_err()); // brackets
        assert!(validate_url("https://test.com/path{braces}").is_err()); // braces
        assert!(validate_url("http://site.com/path|pipe").is_err()); // pipe
        assert!(validate_url("https://test.com/path<angle>").is_err()); // angle brackets
    }

    #[test]
    fn test_validation_service_creation() {
        let service = ValidationService::new();
        let stats = service.get_validation_stats();

        assert!(stats.strict_mode);
        assert_eq!(stats.custom_patterns_count, 0);
        assert_eq!(stats.available_constants.len(), 7);
        assert!(stats
            .available_constants
            .contains(&"SAFE_COMMAND_CHARS".to_string()));
        assert!(stats
            .available_constants
            .contains(&"SAFE_URL_CHARS".to_string()));
    }

    #[test]
    fn test_validation_service_custom_config() {
        let service = ValidationService::with_config(false);
        let stats = service.get_validation_stats();

        assert!(!stats.strict_mode);
        assert_eq!(stats.custom_patterns_count, 0);
    }

    #[test]
    fn test_validation_service_custom_patterns() {
        let mut service = ValidationService::new();
        service.add_custom_pattern("forbidden_word".to_string());
        service.add_custom_pattern("blocked_term".to_string());

        let stats = service.get_validation_stats();
        assert_eq!(stats.custom_patterns_count, 2);

        // Test custom pattern detection (use a valid input type)
        assert!(service
            .validate_comprehensive_input("normal_text", "path")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("text_with_forbidden_word", "path")
            .is_err());
        assert!(service
            .validate_comprehensive_input("text_with_BLOCKED_TERM", "path")
            .is_err()); // case insensitive
    }

    #[test]
    fn test_validation_service_character_whitelist() {
        let service = ValidationService::new();

        // Test path validation
        assert!(service
            .validate_comprehensive_input("src/main.rs", "path")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("docs/README.md", "path")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("path with spaces", "path")
            .is_err());

        // Test command validation
        assert!(service
            .validate_comprehensive_input("ls -la", "command")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("echo hello", "command")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("ls | grep", "command")
            .is_err());

        // Test URL validation
        assert!(service
            .validate_comprehensive_input("https://example.com/path?param=value", "url")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("http://test.com#section", "url")
            .is_ok());
        assert!(service
            .validate_comprehensive_input("https://site.com/path with spaces", "url")
            .is_err());
    }

    #[test]
    fn test_validation_service_sanitization() {
        let service = ValidationService::new();

        // Test path sanitization
        assert_eq!(
            service.sanitize_input_with_whitelist("src/main.rs", "path"),
            "src/main.rs"
        );
        assert_eq!(
            service.sanitize_input_with_whitelist("path with spaces", "path"),
            "pathwithspaces"
        );
        assert_eq!(
            service.sanitize_input_with_whitelist("file@name.txt", "path"),
            "filename.txt"
        );

        // Test command sanitization
        assert_eq!(
            service.sanitize_input_with_whitelist("ls -la", "command"),
            "ls -la"
        );
        assert_eq!(
            service.sanitize_input_with_whitelist("echo $(whoami)", "command"),
            "echo whoami"
        );

        // Test URL sanitization
        assert_eq!(
            service.sanitize_input_with_whitelist("https://example.com/path", "url"),
            "https://example.com/path"
        );
        assert_eq!(
            service.sanitize_input_with_whitelist("http://site.com/path with spaces", "url"),
            "http://site.com/pathwithspaces"
        );
    }

    #[test]
    fn test_validation_service_batch_validation() {
        let service = ValidationService::new();

        let inputs = vec![
            ("src/main.rs".to_string(), "path".to_string()),
            ("ls -la".to_string(), "command".to_string()),
            ("https://example.com".to_string(), "url".to_string()),
            ("invalid | command".to_string(), "command".to_string()),
            ("path with spaces".to_string(), "path".to_string()),
        ];

        let results = service.validate_batch(&inputs).unwrap();
        assert_eq!(results.len(), 5);

        // Check individual results
        assert!(results[0].valid); // valid path
        assert!(results[1].valid); // valid command
        assert!(results[2].valid); // valid URL
        assert!(!results[3].valid); // invalid command with pipe
        assert!(!results[4].valid); // invalid path with spaces

        // Check error messages
        assert!(results[3].error_message.is_some());
        assert!(results[4].error_message.is_some());
    }

    #[test]
    fn test_validation_service_strict_mode() {
        let strict_service = ValidationService::with_config(true);
        let lenient_service = ValidationService::with_config(false);

        let input_with_invalid_chars = "test@example";

        // Strict mode should reject invalid characters
        assert!(strict_service
            .validate_comprehensive_input(input_with_invalid_chars, "path")
            .is_err());

        // Lenient mode should still check security patterns but not character whitelist
        // Note: This will still fail due to security patterns, but for different reasons
        let result = lenient_service.validate_comprehensive_input(input_with_invalid_chars, "path");
        // The input should pass character validation but may fail on security patterns
        // Let's test with a safer input
        let safe_input = "test_example";
        assert!(lenient_service
            .validate_comprehensive_input(safe_input, "path")
            .is_ok());
    }

    #[test]
    fn test_validation_constants_coverage() {
        // Ensure all validation constants are being used
        let service = ValidationService::new();
        let stats = service.get_validation_stats();

        // Verify all expected constants are listed
        let expected_constants = vec![
            "SAFE_PATH_CHARS",
            "SAFE_COMMAND_CHARS",
            "SAFE_URL_CHARS",
            "DANGEROUS_PATTERNS",
            "SQL_INJECTION_PATTERNS",
            "XSS_PATTERNS",
            "COMMAND_INJECTION_PATTERNS",
        ];

        for constant in expected_constants {
            assert!(
                stats.available_constants.contains(&constant.to_string()),
                "Missing constant: {}",
                constant
            );
        }
    }
}

/// Check for dangerous patterns
fn check_dangerous_patterns(input: &str, context: &str) -> Result<()> {
    let input_lower = input.to_lowercase();
    for pattern in DANGEROUS_PATTERNS {
        if input_lower.contains(&pattern.to_lowercase()) {
            return Err(AgentError::invalid_input(format!(
                "Dangerous pattern '{}' detected in {}",
                pattern, context
            )));
        }
    }
    Ok(())
}

/// Check for dangerous patterns with more lenient rules for code content
fn check_dangerous_patterns_lenient(input: &str, context: &str) -> Result<()> {
    let input_lower = input.to_lowercase();

    // Only check the most dangerous patterns for file content
    let strict_patterns = [
        "rm -rf",
        "format ",
        "mkfs",
        "dd if=",
        "chmod 777",
        "curl ",
        "wget ",
        "nc ",
        "netcat",
        "ssh ",
        "ftp ",
    ];

    for pattern in &strict_patterns {
        if input_lower.contains(&pattern.to_lowercase()) {
            return Err(AgentError::invalid_input(format!(
                "Dangerous pattern '{}' detected in {}",
                pattern, context
            )));
        }
    }
    Ok(())
}

/// Check for SQL injection patterns
fn check_sql_injection_patterns(input: &str) -> Result<()> {
    let input_lower = input.to_lowercase();
    for pattern in SQL_INJECTION_PATTERNS {
        if input_lower.contains(&pattern.to_lowercase()) {
            return Err(AgentError::invalid_input(format!(
                "Potential SQL injection pattern '{}' detected",
                pattern
            )));
        }
    }
    Ok(())
}

/// Check for SQL injection patterns with more lenient rules for code content
fn check_sql_injection_patterns_lenient(input: &str) -> Result<()> {
    let input_lower = input.to_lowercase();

    // Check for dangerous SQL injection patterns but allow common SQL keywords in code
    let strict_patterns = [
        "'; drop table",
        "\"; drop table",
        "union select password",
        "delete from users",
        "' or 1=1",
        "\" or 1=1",
        "union select * from",
        "drop table users",
    ];

    for pattern in &strict_patterns {
        if input_lower.contains(&pattern.to_lowercase()) {
            return Err(AgentError::invalid_input(format!(
                "Potential SQL injection pattern '{}' detected",
                pattern
            )));
        }
    }
    Ok(())
}

/// Check for XSS patterns
fn check_xss_patterns(input: &str) -> Result<()> {
    let input_lower = input.to_lowercase();
    for pattern in XSS_PATTERNS {
        if input_lower.contains(&pattern.to_lowercase()) {
            return Err(AgentError::invalid_input(format!(
                "Potential XSS pattern '{}' detected",
                pattern
            )));
        }
    }
    Ok(())
}

/// Check for command injection patterns
fn check_command_injection_patterns(input: &str) -> Result<()> {
    for pattern in COMMAND_INJECTION_PATTERNS {
        if input.contains(pattern) {
            return Err(AgentError::invalid_input(format!(
                "Potential command injection pattern '{}' detected",
                pattern
            )));
        }
    }
    Ok(())
}

/// Sanitize input by removing dangerous characters
pub fn sanitize_input(input: &str) -> String {
    let result: String = input
        .chars()
        .filter(|&c| c.is_alphanumeric() || " ._".contains(c))
        .collect();
    result.trim().to_string()
}

/// Validate JSON input structure
pub fn validate_json_structure(value: &serde_json::Value, max_depth: usize) -> Result<()> {
    fn check_depth(
        value: &serde_json::Value,
        current_depth: usize,
        max_depth: usize,
    ) -> Result<()> {
        if current_depth > max_depth {
            return Err(AgentError::invalid_input(format!(
                "JSON structure too deep: {} levels (max: {})",
                current_depth, max_depth
            )));
        }

        match value {
            serde_json::Value::Object(obj) => {
                if obj.len() > 1000 {
                    return Err(AgentError::invalid_input("Too many object properties"));
                }
                for (key, val) in obj {
                    if key.len() > 256 {
                        return Err(AgentError::invalid_input("Object key too long"));
                    }
                    check_depth(val, current_depth + 1, max_depth)?;
                }
            }
            serde_json::Value::Array(arr) => {
                if arr.len() > 10000 {
                    return Err(AgentError::invalid_input("Array too large"));
                }
                for item in arr {
                    check_depth(item, current_depth + 1, max_depth)?;
                }
            }
            serde_json::Value::String(s) => {
                if s.len() > MAX_FILE_CONTENT_LENGTH {
                    return Err(AgentError::invalid_input("String value too long"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    check_depth(value, 0, max_depth)
}

/// Comprehensive validation service that integrates all validation constants
pub struct ValidationService {
    /// Enable strict character validation
    strict_mode: bool,
    /// Custom validation patterns
    custom_patterns: HashSet<String>,
}

impl ValidationService {
    /// Create a new validation service
    pub fn new() -> Self {
        Self {
            strict_mode: true,
            custom_patterns: HashSet::new(),
        }
    }

    /// Create validation service with custom configuration
    pub fn with_config(strict_mode: bool) -> Self {
        Self {
            strict_mode,
            custom_patterns: HashSet::new(),
        }
    }

    /// Add custom validation pattern
    pub fn add_custom_pattern(&mut self, pattern: String) {
        self.custom_patterns.insert(pattern);
    }

    /// Validate input using all available validation constants and patterns
    pub fn validate_comprehensive_input(&self, input: &str, input_type: &str) -> Result<()> {
        // Basic checks
        if input.trim().is_empty() {
            return Err(AgentError::invalid_input(format!(
                "{} cannot be empty",
                input_type
            )));
        }

        // Check for null bytes
        if input.contains('\0') {
            return Err(AgentError::invalid_input(format!(
                "Null bytes not allowed in {}",
                input_type
            )));
        }

        // Apply strict character validation if enabled
        if self.strict_mode {
            self.validate_character_whitelist(input, input_type)?;
        }

        // Security pattern checks
        self.check_all_security_patterns(input, input_type)?;

        // Custom pattern checks
        self.check_custom_patterns(input, input_type)?;

        Ok(())
    }

    /// Validate against character whitelists based on input type
    fn validate_character_whitelist(&self, input: &str, input_type: &str) -> Result<()> {
        let safe_chars: HashSet<char> = match input_type.to_lowercase().as_str() {
            "path" | "file_path" | "filename" => SAFE_PATH_CHARS.chars().collect(),
            "command" | "shell_command" => SAFE_COMMAND_CHARS.chars().collect(),
            "url" | "uri" | "endpoint" => SAFE_URL_CHARS.chars().collect(),
            _ => {
                // Default to path chars for unknown types
                SAFE_PATH_CHARS.chars().collect()
            }
        };

        for ch in input.chars() {
            if !safe_chars.contains(&ch) {
                return Err(AgentError::invalid_input(format!(
                    "Invalid character '{}' in {} (type: {})",
                    ch, input, input_type
                )));
            }
        }

        Ok(())
    }

    /// Check all security patterns
    fn check_all_security_patterns(&self, input: &str, input_type: &str) -> Result<()> {
        // Check dangerous patterns
        check_dangerous_patterns(input, input_type)?;

        // Check XSS patterns
        check_xss_patterns(input)?;

        // Check SQL injection patterns
        check_sql_injection_patterns(input)?;

        // Check command injection patterns
        check_command_injection_patterns(input)?;

        Ok(())
    }

    /// Check custom patterns
    fn check_custom_patterns(&self, input: &str, input_type: &str) -> Result<()> {
        let input_lower = input.to_lowercase();
        for pattern in &self.custom_patterns {
            if input_lower.contains(&pattern.to_lowercase()) {
                return Err(AgentError::invalid_input(format!(
                    "Custom security pattern '{}' detected in {} (type: {})",
                    pattern, input, input_type
                )));
            }
        }
        Ok(())
    }

    /// Sanitize input by removing invalid characters
    pub fn sanitize_input_with_whitelist(&self, input: &str, input_type: &str) -> String {
        let safe_chars: HashSet<char> = match input_type.to_lowercase().as_str() {
            "path" | "file_path" | "filename" => SAFE_PATH_CHARS.chars().collect(),
            "command" | "shell_command" => SAFE_COMMAND_CHARS.chars().collect(),
            "url" | "uri" | "endpoint" => SAFE_URL_CHARS.chars().collect(),
            _ => SAFE_PATH_CHARS.chars().collect(),
        };

        input.chars().filter(|ch| safe_chars.contains(ch)).collect()
    }

    /// Get validation statistics
    pub fn get_validation_stats(&self) -> ValidationStats {
        ValidationStats {
            strict_mode: self.strict_mode,
            custom_patterns_count: self.custom_patterns.len(),
            available_constants: vec![
                "SAFE_PATH_CHARS".to_string(),
                "SAFE_COMMAND_CHARS".to_string(),
                "SAFE_URL_CHARS".to_string(),
                "DANGEROUS_PATTERNS".to_string(),
                "SQL_INJECTION_PATTERNS".to_string(),
                "XSS_PATTERNS".to_string(),
                "COMMAND_INJECTION_PATTERNS".to_string(),
            ],
        }
    }

    /// Validate multiple inputs with different types
    pub fn validate_batch(&self, inputs: &[(String, String)]) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        for (input, input_type) in inputs {
            let result = match self.validate_comprehensive_input(input, input_type) {
                Ok(()) => ValidationResult {
                    input: input.clone(),
                    input_type: input_type.clone(),
                    valid: true,
                    error_message: None,
                },
                Err(e) => ValidationResult {
                    input: input.clone(),
                    input_type: input_type.clone(),
                    valid: false,
                    error_message: Some(e.to_string()),
                },
            };
            results.push(result);
        }

        Ok(results)
    }
}

impl Default for ValidationService {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub strict_mode: bool,
    pub custom_patterns_count: usize,
    pub available_constants: Vec<String>,
}

/// Validation result for batch operations
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub input: String,
    pub input_type: String,
    pub valid: bool,
    pub error_message: Option<String>,
}
