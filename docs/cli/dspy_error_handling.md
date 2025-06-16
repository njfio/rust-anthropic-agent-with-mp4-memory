# DSPy CLI Error Handling Specification

## Error Handling Philosophy

The DSPy CLI follows a comprehensive error handling strategy that prioritizes:
1. **User Experience**: Clear, actionable error messages
2. **Developer Experience**: Detailed error context for debugging
3. **System Reliability**: Graceful degradation and recovery
4. **Operational Excellence**: Structured logging and monitoring

## Error Categories and Hierarchy

### Primary Error Categories

```rust
#[derive(thiserror::Error, Debug)]
pub enum DspyCliError {
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        suggestion: Option<String>,
        config_path: Option<PathBuf>,
        line_number: Option<usize>,
    },

    #[error("Validation error in {field}: {message}")]
    Validation {
        field: String,
        message: String,
        expected: Option<String>,
        actual: Option<String>,
        suggestions: Vec<String>,
    },

    #[error("Execution error in {operation}: {message}")]
    Execution {
        operation: String,
        message: String,
        stage: Option<String>,
        retry_possible: bool,
        context: HashMap<String, String>,
    },

    #[error("Network error: {message}")]
    Network {
        message: String,
        error_code: Option<u16>,
        retry_after: Option<Duration>,
        endpoint: Option<String>,
    },

    #[error("Resource error - {resource}: {message}")]
    Resource {
        resource: String,
        message: String,
        current_usage: Option<String>,
        limit: Option<String>,
        suggestion: Option<String>,
    },

    #[error("Permission error: {message}")]
    Permission {
        message: String,
        path: Option<PathBuf>,
        required_permission: Option<String>,
        current_user: Option<String>,
    },

    #[error("Timeout error: {operation} timed out after {duration:?}")]
    Timeout {
        operation: String,
        duration: Duration,
        suggestion: Option<String>,
    },

    #[error("User cancelled operation: {operation}")]
    UserCancelled {
        operation: String,
        partial_completion: bool,
        cleanup_required: bool,
    },

    #[error("Dependency error: {dependency} - {message}")]
    Dependency {
        dependency: String,
        message: String,
        version_required: Option<String>,
        version_found: Option<String>,
        install_suggestion: Option<String>,
    },

    #[error("Internal error: {message}")]
    Internal {
        message: String,
        error_id: String,
        context: HashMap<String, String>,
    },
}
```

## Error Context Enhancement

### Context Information Structure

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ErrorContext {
    pub command: String,
    pub subcommand: Option<String>,
    pub arguments: Vec<String>,
    pub working_directory: PathBuf,
    pub config_file: Option<PathBuf>,
    pub environment: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user: Option<String>,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub rust_version: String,
    pub cli_version: String,
    pub available_memory: u64,
    pub available_disk_space: u64,
}
```

### Error Enhancement Traits

```rust
pub trait ErrorEnhancement {
    fn with_suggestion(self, suggestion: impl Into<String>) -> Self;
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> Self;
    fn with_retry_info(self, retry_possible: bool, retry_after: Option<Duration>) -> Self;
    fn with_help_url(self, url: impl Into<String>) -> Self;
}
```

## Error Response Patterns

### User-Facing Error Messages

#### Configuration Errors
```
Error: Configuration error: Invalid optimization strategy 'invalid_strategy'

Suggestion: Available strategies are: mipro_v2, bootstrap_finetune, multi_objective
           Update your configuration file: ~/.config/memvid-agent/dspy.toml
           
Example:
  [dspy.optimization]
  default_strategy = "mipro_v2"

Help: https://docs.memvid-agent.com/dspy/configuration#optimization-strategies
```

#### Validation Errors
```
Error: Validation error in signature: Missing required field 'output_fields'

Expected: Signature must contain both 'input_fields' and 'output_fields'
Actual:   Found only 'input_fields'

Suggestions:
  • Add output_fields section to your signature
  • Use 'dspy dev validate --fix signature.json' to auto-fix
  • Check the signature template with 'dspy modules templates'

File: /path/to/signature.json (line 15)
```

#### Execution Errors
```
Error: Execution error in module optimization: Module 'qa_system' failed during iteration 23

Stage: Parameter tuning
Retry: Possible (2 attempts remaining)

Context:
  • Strategy: mipro_v2
  • Examples: 150 training examples
  • Current accuracy: 0.72
  • Target accuracy: 0.85

Suggestion: Try reducing the learning rate or increasing the number of examples
           Use 'dspy optimize run qa_system --resume' to continue from checkpoint

Debug: Run with --verbose for detailed execution logs
```

#### Network Errors
```
Error: Network error: Failed to connect to Anthropic API

Details:
  • Endpoint: https://api.anthropic.com/v1/messages
  • Status: Connection timeout
  • Retry after: 30 seconds

Suggestions:
  • Check your internet connection
  • Verify your API key is valid
  • Check Anthropic service status: https://status.anthropic.com
  • Try again with increased timeout: --timeout 60

Environment:
  • API Key: Set (sk-ant-***...***abc)
  • Timeout: 30 seconds
  • Retries: 3 attempts made
```

### Machine-Readable Error Format

```json
{
  "error": {
    "type": "ValidationError",
    "code": "DSPY_VAL_001",
    "message": "Missing required field 'output_fields'",
    "severity": "error",
    "category": "validation",
    "context": {
      "field": "signature",
      "file": "/path/to/signature.json",
      "line": 15,
      "column": 3
    },
    "suggestions": [
      {
        "type": "fix",
        "description": "Add output_fields section",
        "command": "dspy dev validate --fix signature.json"
      },
      {
        "type": "template",
        "description": "Use signature template",
        "command": "dspy modules templates"
      }
    ],
    "help_url": "https://docs.memvid-agent.com/dspy/signatures#output-fields",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## Exit Code Standards

### Standard Exit Codes

```rust
pub enum ExitCode {
    Success = 0,
    GeneralError = 1,
    ConfigurationError = 2,
    ValidationError = 3,
    NetworkError = 4,
    ResourceError = 5,
    PermissionError = 6,
    TimeoutError = 7,
    UserCancelled = 8,
    DependencyError = 9,
    InternalError = 10,
}
```

### Exit Code Usage Guidelines

- **0 (Success)**: Operation completed successfully
- **1 (General Error)**: Unspecified error or multiple error types
- **2 (Configuration Error)**: Invalid configuration, missing config files
- **3 (Validation Error)**: Invalid input data, malformed files
- **4 (Network Error)**: API failures, connectivity issues
- **5 (Resource Error)**: Insufficient memory, disk space, CPU
- **6 (Permission Error)**: File access denied, insufficient privileges
- **7 (Timeout Error)**: Operation exceeded time limit
- **8 (User Cancelled)**: User interrupted operation (Ctrl+C)
- **9 (Dependency Error)**: Missing dependencies, version conflicts
- **10 (Internal Error)**: Unexpected errors, bugs

## Error Recovery Strategies

### Automatic Recovery

```rust
pub struct RecoveryStrategy {
    pub retry_count: usize,
    pub retry_delay: Duration,
    pub exponential_backoff: bool,
    pub fallback_action: Option<FallbackAction>,
    pub cleanup_required: bool,
}

pub enum FallbackAction {
    UseDefault,
    SkipOperation,
    PromptUser,
    SaveState,
}
```

### Recovery Scenarios

#### Network Failures
1. **Immediate Retry**: For transient network issues
2. **Exponential Backoff**: For rate limiting or server overload
3. **Fallback to Cache**: Use cached results when available
4. **Offline Mode**: Continue with local operations only

#### Resource Exhaustion
1. **Cleanup Temporary Files**: Free disk space
2. **Reduce Memory Usage**: Use streaming processing
3. **Pause Operations**: Wait for resources to become available
4. **Graceful Degradation**: Reduce operation complexity

#### Configuration Issues
1. **Use Defaults**: Fall back to built-in defaults
2. **Auto-Fix**: Automatically correct common issues
3. **Interactive Repair**: Guide user through configuration fix
4. **Reset Configuration**: Restore to known good state

## Error Reporting and Monitoring

### Error Metrics Collection

```rust
pub struct ErrorMetrics {
    pub error_count: u64,
    pub error_rate: f64,
    pub error_types: HashMap<String, u64>,
    pub recovery_success_rate: f64,
    pub user_resolution_time: Duration,
}
```

### Error Reporting Channels

1. **Console Output**: Immediate user feedback
2. **Log Files**: Detailed error information for debugging
3. **Metrics System**: Aggregated error statistics
4. **Crash Reports**: Automatic error reporting (opt-in)
5. **User Feedback**: Error experience surveys

### Privacy and Security

- **Data Sanitization**: Remove sensitive information from error reports
- **User Consent**: Explicit opt-in for error reporting
- **Local Processing**: Process errors locally when possible
- **Encryption**: Encrypt error reports in transit and at rest

## Testing Error Scenarios

### Error Simulation Framework

```rust
pub struct ErrorSimulator {
    pub network_failures: bool,
    pub resource_exhaustion: bool,
    pub permission_errors: bool,
    pub timeout_scenarios: bool,
    pub corruption_scenarios: bool,
}
```

### Test Categories

1. **Unit Tests**: Individual error handling functions
2. **Integration Tests**: End-to-end error scenarios
3. **Chaos Testing**: Random error injection
4. **User Experience Tests**: Error message clarity and helpfulness
5. **Recovery Tests**: Automatic recovery mechanism validation

## Error Documentation

### Error Code Registry

Each error type has a unique code and documentation:

- **DSPY_CFG_001**: Invalid configuration file format
- **DSPY_VAL_001**: Missing required signature field
- **DSPY_EXE_001**: Module execution failure
- **DSPY_NET_001**: API connection timeout
- **DSPY_RES_001**: Insufficient memory for operation

### User Documentation

- **Error Reference**: Complete list of error codes and solutions
- **Troubleshooting Guide**: Common issues and resolutions
- **FAQ**: Frequently asked questions about errors
- **Community Support**: Forums and chat for error assistance
