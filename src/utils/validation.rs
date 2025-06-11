use crate::utils::error::{AgentError, Result};
use std::collections::HashSet;

/// Maximum allowed length for various input types
pub const MAX_PATH_LENGTH: usize = 4096;
pub const MAX_COMMAND_LENGTH: usize = 8192;
pub const MAX_FILE_CONTENT_LENGTH: usize = 10 * 1024 * 1024; // 10MB
pub const MAX_URL_LENGTH: usize = 2048;
pub const MAX_HEADER_VALUE_LENGTH: usize = 8192;

/// Allowed characters for different input types
const SAFE_PATH_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/\\";
const SAFE_COMMAND_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/ =";

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

    // Check for dangerous shell operators
    let dangerous_operators = [";", "&", "|", "`", "$", ">", "<", "&&", "||", "$(", "${"];
    for op in &dangerous_operators {
        if command.contains(op) {
            return Err(AgentError::invalid_input(format!(
                "Dangerous shell operator not allowed: {}",
                op
            )));
        }
    }

    // Check for null bytes
    if command.contains('\0') {
        return Err(AgentError::invalid_input("Null bytes not allowed in command"));
    }

    Ok(())
}

/// Validate file content size
pub fn validate_file_content(content: &str) -> Result<()> {
    if content.len() > MAX_FILE_CONTENT_LENGTH {
        return Err(AgentError::invalid_input(format!(
            "File content too large: {} bytes (max: {} bytes)",
            content.len(),
            MAX_FILE_CONTENT_LENGTH
        )));
    }

    // Check for null bytes
    if content.contains('\0') {
        return Err(AgentError::invalid_input("Null bytes not allowed in file content"));
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

    // Parse URL to validate format
    match url::Url::parse(url) {
        Ok(parsed_url) => {
            // Check scheme
            match parsed_url.scheme() {
                "http" | "https" => {},
                scheme => return Err(AgentError::invalid_input(format!(
                    "Unsupported URL scheme: {}",
                    scheme
                ))),
            }

            // Check for localhost/private IPs to prevent SSRF
            if let Some(host) = parsed_url.host_str() {
                if is_private_or_localhost(host) {
                    return Err(AgentError::invalid_input(
                        "Requests to localhost or private IPs not allowed"
                    ));
                }
            }
        },
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
                "Control characters not allowed in header values"
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
        &host[1..host.len()-1]
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
            },
            std::net::IpAddr::V6(ipv6) => {
                // Check for private IPv6 ranges
                ipv6.is_loopback() ||
                ipv6.segments()[0] == 0xfc00 || // fc00::/7
                ipv6.segments()[0] == 0xfd00 ||
                ipv6.segments()[0] == 0xfe80    // fe80::/10 link-local
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
        
        // Invalid commands
        assert!(validate_command("rm -rf /; echo done").is_err());
        assert!(validate_command("ls | grep test").is_err());
        assert!(validate_command("$(whoami)").is_err());
        assert!(validate_command("").is_err());
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
}
