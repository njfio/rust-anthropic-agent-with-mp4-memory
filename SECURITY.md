# Security Policy

## Overview

The rust_memvid_agent project takes security seriously and implements comprehensive security measures to protect against common vulnerabilities and attacks. This document outlines our security practices, features, and vulnerability reporting process.

## Security Features

### 🔒 **Comprehensive Security Implementation**

Our security implementation includes multiple layers of protection:

#### **Audit Logging**
- **Structured JSON Logging**: All sensitive operations logged with timestamps, severity levels, and metadata
- **Automatic Log Rotation**: Configurable file size limits (100MB default) with retention policies
- **Severity Filtering**: Configurable minimum severity levels (Info, Low, Medium, High, Critical)
- **Real-time Monitoring**: Immediate logging of security violations and suspicious activities
- **Audit Trail**: Complete audit trail for compliance and forensic analysis

#### **Input Validation & Sanitization**
- **Path Traversal Protection**: Comprehensive validation prevents `../` attacks and absolute path access
- **Command Injection Prevention**: Allowlist-based command filtering with shell operator detection
- **SSRF Protection**: URL validation prevents requests to localhost and private IP ranges
- **Length Limits**: Configurable maximum lengths for paths (4096), commands (8192), and file content (10MB)
- **Character Validation**: Allowlist-based character validation for security-sensitive inputs
- **Null Byte Protection**: Detection and prevention of null byte injection attacks

#### **Rate Limiting**
- **Sliding Window Algorithm**: Advanced rate limiting with configurable time windows
- **Global Limits**: System-wide rate limiting (100 requests/minute default)
- **Per-Tool Limits**: Individual tool rate limiting (50 requests/minute default)
- **Violation Tracking**: Automatic detection and logging of rate limit violations
- **Configurable Thresholds**: Customizable limits based on deployment requirements

#### **Security Headers**
- **Content Security Policy (CSP)**: Configurable CSP with strict default policies
- **HTTP Strict Transport Security (HSTS)**: HSTS with preload support for HTTPS enforcement
- **X-Frame-Options**: Clickjacking protection with DENY/SAMEORIGIN options
- **X-Content-Type-Options**: MIME type sniffing prevention
- **X-XSS-Protection**: Cross-site scripting protection
- **Referrer-Policy**: Referrer information control
- **Permissions-Policy**: Feature policy restrictions

#### **Resource Monitoring**
- **Real-time Monitoring**: Continuous monitoring of memory, CPU, and thread usage
- **Configurable Limits**: Memory (2GB default), CPU (80% default), threads (100 default)
- **Warning Thresholds**: Early warning system at 80% of configured limits
- **Automatic Enforcement**: Optional automatic enforcement of resource limits
- **Violation Logging**: Automatic audit logging of resource limit violations

#### **Dependency Security**
- **Automated Vulnerability Scanning**: Daily cargo-audit scans for known vulnerabilities
- **License Compliance**: Automated license checking with configurable policies
- **Unmaintained Dependency Detection**: Identification of unmaintained dependencies
- **Security Advisory Monitoring**: Real-time monitoring of Rust security advisories

### 🛡️ **Automated Security Pipeline**

#### **Daily Security Audits**
- **Dependency Scanning**: cargo-audit for vulnerability detection
- **Static Analysis**: Clippy with security-focused lints
- **License Compliance**: cargo-deny for license and dependency policies
- **Outdated Dependencies**: Detection of outdated packages

#### **Advanced Security Scanning**
- **CodeQL Analysis**: GitHub's semantic code analysis for security vulnerabilities
- **Semgrep Scanning**: OWASP Top 10 and security pattern detection
- **Trivy Scanning**: Filesystem vulnerability scanning
- **OSSF Scorecard**: Security best practices assessment

#### **Penetration Testing**
- **Automated Testing**: Scheduled penetration testing with OWASP coverage
- **Input Validation Testing**: Comprehensive testing of all input validation mechanisms
- **Path Traversal Testing**: Automated testing of path traversal protections
- **Command Injection Testing**: Testing of command filtering and validation
- **Rate Limiting Testing**: Validation of rate limiting effectiveness

## Security Configuration

### Default Security Settings

The system ships with secure defaults:

```toml
[tools.security]
max_file_size = 10485760  # 10MB
max_path_length = 4096
max_command_length = 8192
allowed_domains = []  # Empty = all allowed

[tools.rate_limiting]
max_requests_per_minute = 100
per_tool_limiting = true
window_duration_seconds = 60

[audit]
log_file_path = "audit.log"
max_file_size = 104857600  # 100MB
max_files = 10
minimum_severity = "low"

[monitoring]
max_memory_bytes = 2147483648  # 2GB
max_memory_percentage = 25.0
max_cpu_percentage = 80.0
max_threads = 100
```

### Production Hardening

For production deployments, consider these additional security measures:

1. **Strict Security Headers**:
   ```rust
   let config = SecurityHeadersConfig::strict()
       .with_csp("default-src 'self'; script-src 'self'; style-src 'self'")
       .with_hsts("max-age=63072000; includeSubDomains; preload");
   ```

2. **Restrictive Rate Limits**:
   ```toml
   [tools.rate_limiting]
   max_requests_per_minute = 50
   per_tool_limiting = true
   ```

3. **Enhanced Monitoring**:
   ```toml
   [monitoring]
   max_memory_bytes = 1073741824  # 1GB
   max_cpu_percentage = 60.0
   monitoring_interval_seconds = 15
   ```

4. **Audit Configuration**:
   ```toml
   [audit]
   minimum_severity = "medium"
   sync_interval_seconds = 30
   ```

## Vulnerability Reporting

### Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. **Email**: Send details to [me@njf.io](mailto:me@njf.io) with subject "SECURITY: rust_memvid_agent vulnerability"
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 24 hours
- **Vulnerability Assessment**: Within 72 hours
- **Fix Development**: Within 7 days for critical issues, 14 days for others
- **Public Disclosure**: After fix is released and users have time to update

### Security Updates

Security updates are released as patch versions and include:
- Detailed security advisory
- Affected versions
- Mitigation steps
- Upgrade instructions

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**: Regularly run `cargo update` and `cargo audit`
2. **Review Configuration**: Use appropriate security settings for your environment
3. **Monitor Logs**: Regularly review audit logs for suspicious activity
4. **Resource Limits**: Configure appropriate resource limits for your deployment
5. **Network Security**: Use HTTPS and configure appropriate domain allowlists

### For Developers

1. **Input Validation**: Always validate and sanitize user inputs
2. **Principle of Least Privilege**: Grant minimal necessary permissions
3. **Security Testing**: Include security tests in your test suite
4. **Dependency Management**: Regularly audit and update dependencies
5. **Code Review**: Include security considerations in code reviews

## Compliance

This project implements security measures aligned with:

- **OWASP Top 10**: Protection against common web application vulnerabilities
- **NIST Cybersecurity Framework**: Comprehensive security controls
- **SOC 2 Type II**: Audit logging and monitoring capabilities
- **ISO 27001**: Information security management practices

## Security Testing

### Running Security Tests

```bash
# Run all security tests
cargo test security_tests --lib

# Run dependency vulnerability scan
cargo audit

# Run with security-focused clippy lints
cargo clippy -- -W clippy::suspicious -W clippy::security

# Run penetration tests (when available)
cargo test pentest --release
```

### Continuous Security

The project includes automated security testing in CI/CD:
- Daily dependency vulnerability scans
- Weekly penetration testing
- Automated security compliance checking
- Real-time security monitoring

## Contact

For security-related questions or concerns:
- **Security Issues**: [me@njf.io](mailto:me@njf.io)
- **General Questions**: Create a GitHub issue (non-security related only)
- **Documentation**: Refer to this SECURITY.md file

---

**Last Updated**: December 2024
**Security Policy Version**: 1.0
