//! DSPy CLI Commands
//!
//! This module defines the command structure and routing for DSPy CLI operations.

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

// Re-export command implementations
pub mod modules;
pub mod benchmark;
pub mod optimize;
pub mod pipeline;
pub mod dev;

/// DSPy CLI commands following noun-verb hierarchy
#[derive(Debug, Clone, Parser)]
#[command(name = "dspy")]
#[command(about = "DSPy framework operations")]
#[command(version)]
pub struct DspyCommands {
    #[command(subcommand)]
    pub command: DspySubcommand,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress non-error output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Override default output format
    #[arg(long, global = true, value_enum)]
    pub format: Option<OutputFormat>,

    /// Disable caching for this operation
    #[arg(long, global = true)]
    pub no_cache: bool,

    /// Operation timeout in seconds
    #[arg(long, global = true)]
    pub timeout: Option<u64>,

    /// Custom configuration file path
    #[arg(long, global = true)]
    pub config: Option<std::path::PathBuf>,
}

/// DSPy subcommands organized by functional area
#[derive(Debug, Clone, Subcommand)]
pub enum DspySubcommand {
    /// Module lifecycle management
    #[command(name = "modules", alias = "m")]
    Modules {
        #[command(subcommand)]
        command: ModulesCommand,
    },

    /// Performance testing and analysis
    #[command(name = "benchmark", alias = "b")]
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },

    /// Module optimization workflows
    #[command(name = "optimize", alias = "o")]
    Optimize {
        #[command(subcommand)]
        command: OptimizeCommand,
    },

    /// Pipeline creation and execution
    #[command(name = "pipeline", alias = "p")]
    Pipeline {
        #[command(subcommand)]
        command: PipelineCommand,
    },

    /// Development tools and utilities
    #[command(name = "dev", alias = "d")]
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

/// Module management commands
#[derive(Debug, Clone, Subcommand)]
pub enum ModulesCommand {
    /// List all available modules
    #[command(name = "list", alias = "ls")]
    List {
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Filter by name pattern
        #[arg(long)]
        filter: Option<String>,
        /// Sort order
        #[arg(long, value_enum)]
        sort: Option<SortOrder>,
    },

    /// Create new module from template
    #[command(name = "create", alias = "new")]
    Create {
        /// Module name
        name: String,
        /// Module template type
        #[arg(long, value_enum)]
        template: Option<ModuleTemplate>,
        /// Signature definition file
        #[arg(long)]
        signature: Option<std::path::PathBuf>,
        /// Module description
        #[arg(long)]
        description: Option<String>,
        /// Overwrite existing module
        #[arg(long)]
        force: bool,
    },

    /// Display module details and metadata
    #[command(name = "show")]
    Show {
        /// Module name
        name: String,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Include performance statistics
        #[arg(long)]
        include_stats: bool,
        /// Include optimization history
        #[arg(long)]
        include_history: bool,
    },

    /// Remove module from registry
    #[command(name = "delete", alias = "rm")]
    Delete {
        /// Module name
        name: String,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
        /// Create backup before deletion
        #[arg(long)]
        backup: bool,
    },

    /// Validate module signature/configuration
    #[command(name = "validate")]
    Validate {
        /// File to validate
        file: std::path::PathBuf,
        /// Strict validation mode
        #[arg(long)]
        strict: bool,
        /// Auto-fix common issues
        #[arg(long)]
        fix: bool,
        /// Save validation report
        #[arg(long)]
        output: Option<std::path::PathBuf>,
    },

    /// List available module templates
    #[command(name = "templates")]
    Templates {
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Filter by template category
        #[arg(long)]
        category: Option<String>,
    },
}

/// Benchmarking commands
#[derive(Debug, Clone, Subcommand)]
pub enum BenchmarkCommand {
    /// Execute benchmark on module
    #[command(name = "run")]
    Run {
        /// Module to benchmark
        module: String,
        /// Number of test iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        /// Timeout per iteration in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,
        /// Test input data file
        #[arg(short, long)]
        input: Option<std::path::PathBuf>,
        /// Save results to file
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Warmup iterations
        #[arg(long, default_value = "5")]
        warmup: usize,
        /// Parallel execution threads
        #[arg(long)]
        parallel: Option<usize>,
    },

    /// Compare multiple modules
    #[command(name = "compare")]
    Compare {
        /// Modules to compare (comma-separated)
        modules: String,
        /// Comparison metric
        #[arg(short, long)]
        metric: Option<String>,
        /// Common test input file
        #[arg(short, long)]
        input: Option<std::path::PathBuf>,
        /// Comparison report output
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Include statistical analysis
        #[arg(long)]
        statistical: bool,
    },

    /// Export benchmark results
    #[command(name = "export")]
    Export {
        /// Module name (optional, exports all if not specified)
        module: Option<String>,
        /// Export format
        #[arg(long, value_enum)]
        format: Option<ExportFormat>,
        /// Filter results by pattern
        #[arg(long)]
        filter: Option<String>,
        /// Results since date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
        /// Output file
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Show benchmark history
    #[command(name = "history")]
    History {
        /// Module name (optional, shows all if not specified)
        module: Option<String>,
        /// Number of results to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Focus on specific metric
        #[arg(short, long)]
        metric: Option<String>,
        /// Show performance trends
        #[arg(long)]
        trend: bool,
    },
}

/// Optimization commands
#[derive(Debug, Clone, Subcommand)]
pub enum OptimizeCommand {
    /// Start module optimization
    #[command(name = "run")]
    Run {
        /// Module to optimize
        module: String,
        /// Optimization strategy
        #[arg(short, long)]
        strategy: Option<String>,
        /// Training examples file
        #[arg(short, long)]
        examples: Option<std::path::PathBuf>,
        /// Maximum iterations
        #[arg(short, long, default_value = "50")]
        iterations: usize,
        /// Target optimization metric
        #[arg(long)]
        target_metric: Option<String>,
        /// Convergence threshold
        #[arg(long)]
        threshold: Option<f64>,
        /// Save optimization results
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Resume previous optimization
        #[arg(long)]
        resume: Option<String>,
    },

    /// List available optimization strategies
    #[command(name = "strategies")]
    Strategies {
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Filter by strategy category
        #[arg(long)]
        category: Option<String>,
        /// Show detailed strategy description
        #[arg(long)]
        describe: Option<String>,
    },

    /// Show optimization history
    #[command(name = "history")]
    History {
        /// Module name
        module: String,
        /// Number of results to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Filter by strategy
        #[arg(short, long)]
        strategy: Option<String>,
        /// Show only successful optimizations
        #[arg(long)]
        successful_only: bool,
    },

    /// Apply optimization result
    #[command(name = "apply")]
    Apply {
        /// Module name
        module: String,
        /// Optimization result ID
        result_id: String,
        /// Create backup before applying
        #[arg(long)]
        backup: bool,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
        /// Validate after application
        #[arg(long)]
        validate: bool,
    },
}

/// Pipeline management commands
#[derive(Debug, Clone, Subcommand)]
pub enum PipelineCommand {
    /// Create new pipeline
    #[command(name = "create", alias = "new")]
    Create {
        /// Pipeline name
        name: String,
        /// Pipeline template
        #[arg(long)]
        template: Option<String>,
        /// Comma-separated module list
        #[arg(long)]
        modules: Option<String>,
        /// Pipeline configuration file
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        /// Pipeline description
        #[arg(long)]
        description: Option<String>,
        /// Overwrite existing pipeline
        #[arg(long)]
        force: bool,
    },

    /// Execute pipeline
    #[command(name = "run")]
    Run {
        /// Pipeline name
        name: String,
        /// Input data file
        #[arg(short, long)]
        input: Option<std::path::PathBuf>,
        /// Output file
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Execution timeout in seconds
        #[arg(long, default_value = "600")]
        timeout: u64,
        /// Enable parallel execution
        #[arg(long)]
        parallel: bool,
        /// Real-time monitoring
        #[arg(long)]
        monitor: bool,
    },

    /// List all pipelines
    #[command(name = "list", alias = "ls")]
    List {
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Filter by name pattern
        #[arg(long)]
        filter: Option<String>,
        /// Sort order
        #[arg(long, value_enum)]
        sort: Option<SortOrder>,
    },

    /// Display pipeline details
    #[command(name = "show")]
    Show {
        /// Pipeline name
        name: String,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Include module details
        #[arg(long)]
        include_modules: bool,
        /// Include execution statistics
        #[arg(long)]
        include_stats: bool,
    },

    /// Show pipeline performance statistics
    #[command(name = "stats")]
    Stats {
        /// Pipeline name
        name: String,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Focus on specific metric
        #[arg(short, long)]
        metric: Option<String>,
        /// Statistics since date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
        /// Show performance trends
        #[arg(long)]
        trend: bool,
    },
}

/// Development tools commands
#[derive(Debug, Clone, Subcommand)]
pub enum DevCommand {
    /// Validate signature files
    #[command(name = "validate")]
    Validate {
        /// Signature file to validate
        signature: std::path::PathBuf,
        /// Strict validation mode
        #[arg(long)]
        strict: bool,
        /// Validation report format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Auto-fix common issues
        #[arg(long)]
        fix: bool,
        /// Custom validation schema
        #[arg(long)]
        schema: Option<std::path::PathBuf>,
    },

    /// Run module tests
    #[command(name = "test")]
    Test {
        /// Module to test
        module: String,
        /// Test cases file
        #[arg(long)]
        test_cases: Option<std::path::PathBuf>,
        /// Generate coverage report
        #[arg(long)]
        coverage: bool,
        /// Test report format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Parallel test execution
        #[arg(long)]
        parallel: bool,
        /// Save test results
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Interactive module debugging
    #[command(name = "debug")]
    Debug {
        /// Module to debug
        module: String,
        /// Debug input data
        #[arg(short, long)]
        input: Option<String>,
        /// Set breakpoint at step
        #[arg(long)]
        breakpoint: Option<String>,
        /// Enable execution tracing
        #[arg(long)]
        trace: bool,
        /// Save debug session
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Generate code templates
    #[command(name = "generate", alias = "gen")]
    Generate {
        /// Template type to generate
        template: String,
        /// Generated item name
        #[arg(short, long)]
        name: Option<String>,
        /// Output directory
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Template parameters file
        #[arg(long)]
        parameters: Option<std::path::PathBuf>,
        /// Overwrite existing files
        #[arg(long)]
        force: bool,
    },

    /// Inspect module internals
    #[command(name = "inspect")]
    Inspect {
        /// Module to inspect
        module: String,
        /// Output format
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
        /// Inspection depth
        #[arg(short, long)]
        depth: Option<usize>,
        /// Include cache information
        #[arg(long)]
        include_cache: bool,
        /// Include performance metrics
        #[arg(long)]
        include_metrics: bool,
    },
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum OutputFormat {
    #[value(name = "table")]
    Table,
    #[value(name = "json")]
    Json,
    #[value(name = "yaml")]
    Yaml,
    #[value(name = "csv")]
    Csv,
    #[value(name = "chart")]
    Chart,
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum ExportFormat {
    #[value(name = "json")]
    Json,
    #[value(name = "csv")]
    Csv,
    #[value(name = "prometheus")]
    Prometheus,
}

/// Sort order options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum SortOrder {
    #[value(name = "name")]
    Name,
    #[value(name = "created")]
    Created,
    #[value(name = "modified")]
    Modified,
    #[value(name = "last-run")]
    LastRun,
}

/// Module template types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum ModuleTemplate {
    #[value(name = "predict")]
    Predict,
    #[value(name = "cot")]
    ChainOfThought,
    #[value(name = "rag")]
    RAG,
    #[value(name = "react")]
    ReAct,
    #[value(name = "pot")]
    ProgramOfThought,
}
