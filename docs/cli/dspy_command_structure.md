# DSPy CLI Command Structure

## Command Hierarchy Diagram

```
memvid-agent dspy
│
├── modules                           # Module Lifecycle Management
│   ├── list                         # List all available modules
│   │   ├── --format [table|json]    # Output format
│   │   ├── --filter <pattern>       # Filter by name pattern
│   │   └── --sort [name|created|modified]
│   │
│   ├── create <name>                # Create new module
│   │   ├── --template <type>        # Module template (predict|cot|rag)
│   │   ├── --signature <file>       # Signature definition file
│   │   ├── --description <text>     # Module description
│   │   └── --force                  # Overwrite existing module
│   │
│   ├── show <name>                  # Display module details
│   │   ├── --format [table|json|yaml]
│   │   ├── --include-stats          # Include performance statistics
│   │   └── --include-history        # Include optimization history
│   │
│   ├── delete <name>                # Remove module
│   │   ├── --force                  # Skip confirmation
│   │   └── --backup                 # Create backup before deletion
│   │
│   ├── validate <file>              # Validate module/signature
│   │   ├── --strict                 # Strict validation mode
│   │   ├── --fix                    # Auto-fix common issues
│   │   └── --output <file>          # Save validation report
│   │
│   └── templates                    # List available templates
│       ├── --format [table|json]
│       └── --category <type>        # Filter by template category
│
├── benchmark                        # Performance Testing & Analysis
│   ├── run <module>                 # Execute benchmark
│   │   ├── --iterations <n>         # Number of test iterations
│   │   ├── --timeout <seconds>      # Timeout per iteration
│   │   ├── --input <file>           # Test input data file
│   │   ├── --output <file>          # Save results to file
│   │   ├── --format [json|csv|table]
│   │   ├── --warmup <n>             # Warmup iterations
│   │   └── --parallel <n>           # Parallel execution threads
│   │
│   ├── compare <modules...>         # Compare multiple modules
│   │   ├── --metric <name>          # Comparison metric
│   │   ├── --input <file>           # Common test input
│   │   ├── --output <file>          # Comparison report
│   │   ├── --format [table|json|chart]
│   │   └── --statistical            # Include statistical analysis
│   │
│   ├── export [module]              # Export benchmark results
│   │   ├── --format [json|csv|prometheus]
│   │   ├── --filter <pattern>       # Filter results
│   │   ├── --since <date>           # Results since date
│   │   └── --output <file>          # Output file
│   │
│   ├── history [module]             # Show benchmark history
│   │   ├── --limit <n>              # Number of results
│   │   ├── --format [table|json|chart]
│   │   ├── --metric <name>          # Focus on specific metric
│   │   └── --trend                  # Show performance trends
│   │
│   └── config                       # Manage benchmark configurations
│       ├── show                     # Show current configuration
│       ├── set <key> <value>        # Set configuration value
│       ├── reset                    # Reset to defaults
│       └── validate                 # Validate configuration
│
├── optimize                         # Module Optimization Workflows
│   ├── run <module>                 # Start optimization
│   │   ├── --strategy <name>        # Optimization strategy
│   │   ├── --examples <file>        # Training examples file
│   │   ├── --iterations <n>         # Maximum iterations
│   │   ├── --target-metric <name>   # Target optimization metric
│   │   ├── --threshold <value>      # Convergence threshold
│   │   ├── --output <file>          # Save optimization results
│   │   └── --resume <id>            # Resume previous optimization
│   │
│   ├── strategies                   # List optimization strategies
│   │   ├── --format [table|json]
│   │   ├── --category <type>        # Filter by strategy type
│   │   └── --describe <name>        # Detailed strategy description
│   │
│   ├── history <module>             # Show optimization history
│   │   ├── --limit <n>              # Number of results
│   │   ├── --format [table|json]
│   │   ├── --strategy <name>        # Filter by strategy
│   │   └── --successful-only        # Show only successful runs
│   │
│   ├── apply <module> <result-id>   # Apply optimization result
│   │   ├── --backup                 # Backup current module
│   │   ├── --force                  # Skip confirmation
│   │   └── --validate               # Validate after application
│   │
│   └── examples <module>            # Manage training examples
│       ├── list                     # List examples
│       ├── add <file>               # Add examples from file
│       ├── remove <id>              # Remove specific example
│       ├── validate                 # Validate examples
│       └── export <file>            # Export examples to file
│
├── pipeline                         # Pipeline Creation & Execution
│   ├── create <name>                # Create new pipeline
│   │   ├── --template <type>        # Pipeline template
│   │   ├── --modules <list>         # Comma-separated module list
│   │   ├── --config <file>          # Pipeline configuration file
│   │   ├── --description <text>     # Pipeline description
│   │   └── --force                  # Overwrite existing pipeline
│   │
│   ├── run <name>                   # Execute pipeline
│   │   ├── --input <file>           # Input data file
│   │   ├── --output <file>          # Output file
│   │   ├── --format [json|yaml]     # Output format
│   │   ├── --timeout <seconds>      # Execution timeout
│   │   ├── --parallel               # Enable parallel execution
│   │   └── --monitor                # Real-time monitoring
│   │
│   ├── list                         # List all pipelines
│   │   ├── --format [table|json]
│   │   ├── --filter <pattern>       # Filter by name pattern
│   │   └── --sort [name|created|last-run]
│   │
│   ├── show <name>                  # Display pipeline details
│   │   ├── --format [table|json|yaml]
│   │   ├── --include-modules        # Include module details
│   │   └── --include-stats          # Include execution statistics
│   │
│   ├── stats <name>                 # Show pipeline performance
│   │   ├── --format [table|json|chart]
│   │   ├── --metric <name>          # Focus on specific metric
│   │   ├── --since <date>           # Statistics since date
│   │   └── --trend                  # Show performance trends
│   │
│   └── templates                    # List pipeline templates
│       ├── --format [table|json]
│       ├── --category <type>        # Filter by template category
│       └── --describe <name>        # Detailed template description
│
└── dev                              # Development Tools & Utilities
    ├── validate <signature>         # Validate signature files
    │   ├── --strict                 # Strict validation mode
    │   ├── --format [table|json]    # Validation report format
    │   ├── --fix                    # Auto-fix common issues
    │   └── --schema <file>          # Custom validation schema
    │
    ├── test <module>                # Run module tests
    │   ├── --test-cases <file>      # Test cases file
    │   ├── --coverage               # Generate coverage report
    │   ├── --format [table|json]    # Test report format
    │   ├── --parallel               # Parallel test execution
    │   └── --output <file>          # Save test results
    │
    ├── debug <module>               # Interactive debugging
    │   ├── --input <data>           # Debug input data
    │   ├── --breakpoint <step>      # Set breakpoint at step
    │   ├── --trace                  # Enable execution tracing
    │   └── --output <file>          # Save debug session
    │
    ├── generate <template>          # Generate code templates
    │   ├── --name <name>            # Generated item name
    │   ├── --output <dir>           # Output directory
    │   ├── --parameters <file>      # Template parameters file
    │   └── --force                  # Overwrite existing files
    │
    └── inspect <module>             # Inspect module internals
        ├── --format [table|json|tree]
        ├── --depth <n>              # Inspection depth
        ├── --include-cache          # Include cache information
        └── --include-metrics        # Include performance metrics
```

## Global Flags

All DSPy commands support these global flags:

```
Global Options:
  -h, --help                    Show help information
  -V, --version                 Show version information
  -v, --verbose                 Enable verbose output
  -q, --quiet                   Suppress non-error output
  --config <file>               Use custom configuration file
  --no-cache                    Disable caching
  --timeout <seconds>           Global operation timeout
  --format <format>             Override default output format
  --color [auto|always|never]   Control colored output
  --progress [auto|always|never] Control progress indicators
```

## Output Formats

### Supported Formats
- **table**: Human-readable tabular format (default for terminals)
- **json**: Machine-readable JSON format
- **yaml**: Human-readable YAML format
- **csv**: Comma-separated values for data analysis
- **chart**: ASCII charts for performance data
- **prometheus**: Prometheus metrics format

### Format Selection Priority
1. Command-line `--format` flag
2. Environment variable `DSPY_OUTPUT_FORMAT`
3. Configuration file setting
4. Auto-detection based on output destination (pipe vs terminal)

## Exit Codes

```
0   Success
1   General error
2   Configuration error
3   Validation error
4   Network error
5   Resource error (disk, memory)
6   Permission error
7   Timeout error
8   User cancellation
9   Dependency error
10  Internal error
```

## Environment Variables

```
DSPY_CONFIG_PATH          # Override default config path
DSPY_OUTPUT_FORMAT        # Default output format
DSPY_CACHE_DISABLE        # Disable caching (true/false)
DSPY_LOG_LEVEL           # Logging level (debug/info/warn/error)
DSPY_TIMEOUT             # Default timeout in seconds
DSPY_PARALLEL_LIMIT      # Maximum parallel operations
DSPY_REGISTRY_PATH       # Module registry path
DSPY_TEMPLATES_PATH      # Templates directory path
```

## Command Aliases

Common command aliases for improved usability:

```
dspy m     → dspy modules
dspy b     → dspy benchmark
dspy o     → dspy optimize
dspy p     → dspy pipeline
dspy d     → dspy dev

dspy m ls  → dspy modules list
dspy m new → dspy modules create
dspy b run → dspy benchmark run
dspy o run → dspy optimize run
dspy p run → dspy pipeline run
```
