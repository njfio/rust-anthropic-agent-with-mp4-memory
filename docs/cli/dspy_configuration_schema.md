# DSPy CLI Configuration Schema

## Configuration File Structure

The DSPy CLI uses TOML configuration files with a hierarchical structure supporting multiple configuration sources and environment variable overrides.

## Complete Configuration Schema

```toml
# DSPy CLI Configuration Schema
# File: dspy.toml

[dspy]
# Global DSPy Framework Settings
version = "1.0.0"                    # Configuration schema version
default_strategy = "mipro_v2"        # Default optimization strategy
enable_caching = true                # Enable global caching
cache_ttl_seconds = 3600             # Cache time-to-live (1 hour)
max_concurrent_operations = 4        # Maximum parallel operations
log_level = "info"                   # Logging level (debug|info|warn|error)
timeout_seconds = 300                # Default operation timeout (5 minutes)
auto_save_results = true             # Automatically save operation results
backup_before_changes = true         # Create backups before destructive operations

[dspy.paths]
# Directory and File Path Configuration
base_dir = "~/.config/memvid-agent/dspy"  # Base directory for DSPy data
registry_path = "${base_dir}/modules"      # Module registry directory
templates_path = "${base_dir}/templates"   # Module templates directory
cache_path = "${base_dir}/cache"           # Cache directory
logs_path = "${base_dir}/logs"             # Log files directory
backups_path = "${base_dir}/backups"       # Backup files directory
temp_path = "${base_dir}/temp"             # Temporary files directory

[dspy.modules]
# Module Management Configuration
auto_validate = true                 # Validate modules on creation/modification
template_auto_update = false        # Auto-update templates from registry
default_template = "predict"        # Default module template
signature_validation = "strict"     # Validation level (strict|normal|lenient)
metadata_required = ["name", "description", "version"]  # Required metadata fields
max_module_size_mb = 100            # Maximum module size in MB
compression_enabled = true          # Enable module compression

[dspy.modules.templates]
# Module Template Configuration
predict = { description = "Basic prediction module", inputs = ["text"], outputs = ["response"] }
chain_of_thought = { description = "Chain of thought reasoning", inputs = ["question"], outputs = ["answer", "reasoning"] }
rag = { description = "Retrieval-augmented generation", inputs = ["query"], outputs = ["answer", "sources"] }
react = { description = "ReAct agent module", inputs = ["task"], outputs = ["result", "actions"] }

[dspy.benchmark]
# Benchmarking Configuration
default_iterations = 100            # Default number of benchmark iterations
warmup_iterations = 5               # Warmup iterations before measurement
timeout_seconds = 300               # Benchmark timeout (5 minutes)
output_format = "table"             # Default output format (table|json|csv)
save_results = true                 # Save benchmark results to disk
results_retention_days = 30         # Keep results for 30 days
parallel_execution = true           # Enable parallel benchmark execution
max_parallel_benchmarks = 2         # Maximum concurrent benchmarks
memory_monitoring = true            # Monitor memory usage during benchmarks
cpu_monitoring = true               # Monitor CPU usage during benchmarks

[dspy.benchmark.metrics]
# Benchmark Metrics Configuration
default_metrics = ["latency", "throughput", "accuracy"]  # Default metrics to collect
latency_percentiles = [50, 90, 95, 99]  # Latency percentiles to calculate
accuracy_threshold = 0.8            # Minimum accuracy threshold
performance_baseline = "previous"   # Baseline for performance comparison

[dspy.optimization]
# Optimization Configuration
max_iterations = 50                 # Maximum optimization iterations
convergence_threshold = 0.01        # Convergence threshold for optimization
save_history = true                 # Save optimization history
history_retention_days = 90         # Keep optimization history for 90 days
auto_apply_best = false             # Automatically apply best optimization result
validation_split = 0.2             # Validation set split ratio
early_stopping_patience = 5        # Early stopping patience (iterations)
checkpoint_frequency = 10          # Save checkpoint every N iterations

[dspy.optimization.strategies]
# Optimization Strategy Configuration
mipro_v2 = { max_candidates = 50, max_bootstrapped_demos = 20, max_labeled_demos = 10, num_trials = 100 }
bootstrap_finetune = { learning_rate = 0.001, batch_size = 16, epochs = 10, warmup_steps = 100 }
multi_objective = { objectives = ["accuracy", "latency"], weights = [0.7, 0.3] }

[dspy.pipeline]
# Pipeline Configuration
execution_timeout = 600             # Pipeline execution timeout (10 minutes)
parallel_execution = true           # Enable parallel pipeline execution
max_parallel_stages = 3             # Maximum parallel pipeline stages
save_logs = true                    # Save pipeline execution logs
log_retention_days = 14             # Keep pipeline logs for 14 days
checkpoint_enabled = true           # Enable pipeline checkpointing
auto_retry_failed_stages = true     # Retry failed pipeline stages
max_retries = 3                     # Maximum retry attempts

[dspy.pipeline.monitoring]
# Pipeline Monitoring Configuration
enable_metrics = true               # Enable pipeline metrics collection
metrics_interval_seconds = 30      # Metrics collection interval
alert_on_failure = true             # Alert on pipeline failures
alert_on_slow_execution = true      # Alert on slow pipeline execution
slow_execution_threshold = 300     # Slow execution threshold (5 minutes)

[dspy.dev]
# Development Tools Configuration
auto_format_code = true             # Auto-format generated code
include_documentation = true        # Include documentation in generated code
validation_on_save = true          # Validate on file save
debug_mode = false                  # Enable debug mode by default
test_coverage_threshold = 0.8      # Minimum test coverage threshold
generate_examples = true           # Generate usage examples

[dspy.dev.templates]
# Development Template Configuration
code_style = "rust_standard"        # Code style for generated templates
include_tests = true                # Include test templates
include_benchmarks = false         # Include benchmark templates
license = "MIT"                     # Default license for generated code

[dspy.security]
# Security Configuration
validate_inputs = true              # Validate all user inputs
sanitize_file_paths = true          # Sanitize file paths to prevent traversal
max_file_size_mb = 50               # Maximum file size for uploads/processing
allowed_file_extensions = [".json", ".yaml", ".toml", ".txt"]  # Allowed file extensions
rate_limiting = true                # Enable rate limiting for operations
max_operations_per_minute = 60     # Maximum operations per minute

[dspy.api]
# API Configuration
anthropic_timeout_seconds = 30      # Anthropic API timeout
max_retries = 3                     # Maximum API retry attempts
retry_delay_seconds = 1             # Delay between retries
connection_pool_size = 10           # HTTP connection pool size
enable_compression = true           # Enable HTTP compression
user_agent = "memvid-agent-dspy/1.0"  # User agent string

[dspy.output]
# Output Configuration
default_format = "auto"             # Default output format (auto|table|json|yaml|csv)
color_output = "auto"               # Colored output (auto|always|never)
progress_indicators = "auto"        # Progress indicators (auto|always|never)
table_max_width = 120               # Maximum table width for terminal output
json_pretty_print = true           # Pretty print JSON output
include_timestamps = true          # Include timestamps in output
include_metadata = false           # Include metadata in output

[dspy.logging]
# Logging Configuration
level = "info"                      # Log level (debug|info|warn|error)
format = "structured"               # Log format (structured|plain)
output = "file"                     # Log output (file|console|both)
file_path = "${dspy.paths.logs_path}/dspy.log"  # Log file path
max_file_size_mb = 10               # Maximum log file size
max_files = 5                       # Maximum number of log files to keep
include_source_location = false    # Include source code location in logs

[dspy.cache]
# Caching Configuration
enabled = true                      # Enable caching
type = "disk"                       # Cache type (disk|memory|redis)
max_size_mb = 500                   # Maximum cache size in MB
ttl_seconds = 3600                  # Default cache TTL (1 hour)
compression = true                  # Enable cache compression
cleanup_interval_seconds = 300     # Cache cleanup interval (5 minutes)
eviction_policy = "lru"             # Cache eviction policy (lru|lfu|fifo)

[dspy.performance]
# Performance Configuration
enable_monitoring = true            # Enable performance monitoring
metrics_collection = true          # Collect performance metrics
profiling_enabled = false          # Enable performance profiling
memory_limit_mb = 1024              # Memory limit for operations (1GB)
cpu_limit_percent = 80.0            # CPU usage limit percentage
disk_space_threshold_mb = 1024      # Minimum free disk space (1GB)

[dspy.notifications]
# Notification Configuration
enabled = false                     # Enable notifications
email_enabled = false               # Enable email notifications
webhook_enabled = false             # Enable webhook notifications
desktop_enabled = true              # Enable desktop notifications
notify_on_completion = true         # Notify on operation completion
notify_on_error = true              # Notify on errors
notify_on_optimization_complete = true  # Notify on optimization completion
```

## Environment Variable Overrides

All configuration values can be overridden using environment variables with the `DSPY_` prefix:

```bash
# Global settings
export DSPY_DEFAULT_STRATEGY="bootstrap_finetune"
export DSPY_ENABLE_CACHING="false"
export DSPY_LOG_LEVEL="debug"
export DSPY_TIMEOUT_SECONDS="600"

# Path settings
export DSPY_BASE_DIR="/custom/dspy/path"
export DSPY_REGISTRY_PATH="/custom/modules"
export DSPY_CACHE_PATH="/tmp/dspy_cache"

# Module settings
export DSPY_MODULES_AUTO_VALIDATE="false"
export DSPY_MODULES_DEFAULT_TEMPLATE="chain_of_thought"

# Benchmark settings
export DSPY_BENCHMARK_DEFAULT_ITERATIONS="200"
export DSPY_BENCHMARK_OUTPUT_FORMAT="json"
export DSPY_BENCHMARK_PARALLEL_EXECUTION="false"

# Optimization settings
export DSPY_OPTIMIZATION_MAX_ITERATIONS="100"
export DSPY_OPTIMIZATION_AUTO_APPLY_BEST="true"

# Security settings
export DSPY_SECURITY_MAX_FILE_SIZE_MB="100"
export DSPY_SECURITY_RATE_LIMITING="false"

# API settings
export DSPY_API_ANTHROPIC_TIMEOUT_SECONDS="60"
export DSPY_API_MAX_RETRIES="5"
```

## Configuration Validation

The configuration system includes comprehensive validation:

### Required Fields
- `dspy.version`: Configuration schema version
- `dspy.paths.base_dir`: Base directory for DSPy data

### Validation Rules
- All timeout values must be positive integers
- File size limits must be positive numbers
- Percentages must be between 0.0 and 1.0
- Directory paths must be valid and accessible
- Strategy names must match available strategies

### Default Value Resolution
1. Explicit configuration value
2. Environment variable override
3. Built-in default value
4. Error if no default available

## Configuration Migration

The system supports automatic configuration migration between schema versions:

```toml
[migration]
from_version = "0.9.0"
to_version = "1.0.0"
backup_original = true
migration_log = "${dspy.paths.logs_path}/migration.log"
```

Migration rules handle:
- Renamed configuration keys
- Changed value formats
- Deprecated settings removal
- New required settings addition
