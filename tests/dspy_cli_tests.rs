//! DSPy CLI Integration Tests
//!
//! This module contains integration tests for the DSPy CLI functionality.

use rust_memvid_agent::cli::dspy::{
    commands::{DspyCommands, DspySubcommand, ModulesCommand, OutputFormat},
    DspyCliConfig, DspyCliError, DspyCliResult, DspyConfigManager,
};
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::test]
async fn test_dspy_config_manager_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("dspy.toml");

    let manager = DspyConfigManager::with_path(config_path.clone());
    assert_eq!(manager.config_path(), config_path);
}

#[tokio::test]
async fn test_dspy_config_default_validation() {
    let config = DspyCliConfig::default();
    let result = config.validate();
    assert!(result.is_ok(), "Default configuration should be valid");
}

#[tokio::test]
async fn test_dspy_config_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_dspy.toml");

    let manager = DspyConfigManager::with_path(config_path);
    let original_config = DspyCliConfig::default();

    // Save configuration
    let save_result = manager.save_config(&original_config).await;
    assert!(save_result.is_ok(), "Should be able to save configuration");

    // Load configuration
    let loaded_config = manager.load_config().await;
    assert!(
        loaded_config.is_ok(),
        "Should be able to load configuration"
    );

    let loaded_config = loaded_config.unwrap();
    assert_eq!(original_config.version, loaded_config.version);
    assert_eq!(
        original_config.dspy.default_strategy,
        loaded_config.dspy.default_strategy
    );
}

#[tokio::test]
async fn test_dspy_config_ensure_exists() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("auto_dspy.toml");

    let manager = DspyConfigManager::with_path(config_path.clone());

    // Config should not exist initially
    assert!(!config_path.exists());

    // Ensure config exists should create it
    let result = manager.ensure_config_exists().await;
    assert!(result.is_ok(), "Should be able to ensure config exists");
    assert!(config_path.exists(), "Config file should be created");
}

#[test]
fn test_dspy_cli_error_creation() {
    let error = DspyCliError::config_error("Test message", "Test suggestion");
    assert!(matches!(error, DspyCliError::Config { .. }));

    let user_message = error.user_message();
    assert!(user_message.contains("Test message"));
    assert!(user_message.contains("Test suggestion"));
}

#[test]
fn test_dspy_cli_error_context() {
    let error =
        DspyCliError::execution_error("test_op", "test message", true).with_context("key", "value");

    if let DspyCliError::Execution { context, .. } = error {
        assert_eq!(context.get("key"), Some(&"value".to_string()));
    } else {
        panic!("Expected execution error");
    }
}

#[test]
fn test_dspy_cli_error_info() {
    let error = DspyCliError::validation_error(
        "test_field",
        "test message",
        vec!["suggestion 1".to_string()],
    );

    let error_info = error.error_info();
    assert_eq!(error_info.error_type, "ValidationError");
    assert_eq!(error_info.error_code, "DSPY_VAL_001");
    assert_eq!(error_info.category, "validation");
}

#[test]
fn test_output_format_enum() {
    // Test that output format enum works correctly
    assert_eq!(OutputFormat::Table as u8, OutputFormat::Table as u8);
    assert_ne!(OutputFormat::Table as u8, OutputFormat::Json as u8);
}

#[test]
fn test_modules_command_structure() {
    // Test that modules command structure is properly defined
    let list_command = ModulesCommand::List {
        format: Some(OutputFormat::Json),
        filter: Some("test".to_string()),
        sort: None,
    };

    match list_command {
        ModulesCommand::List { format, filter, .. } => {
            assert_eq!(format, Some(OutputFormat::Json));
            assert_eq!(filter, Some("test".to_string()));
        }
        _ => panic!("Expected List command"),
    }
}

#[test]
fn test_dspy_commands_structure() {
    // Test that the main DSPy commands structure is properly defined
    let modules_command = DspySubcommand::Modules {
        command: ModulesCommand::List {
            format: None,
            filter: None,
            sort: None,
        },
    };

    match modules_command {
        DspySubcommand::Modules { .. } => {
            // Success - structure is correct
        }
        _ => panic!("Expected Modules subcommand"),
    }
}

#[tokio::test]
async fn test_config_to_dspy_config_conversion() {
    let cli_config = DspyCliConfig::default();
    let dspy_config = cli_config.to_dspy_config();

    // Test that conversion preserves important settings
    assert_eq!(dspy_config.enable_module_caching, cli_config.cache.enabled);
    assert_eq!(dspy_config.cache_ttl_seconds, cli_config.cache.ttl_seconds);
    assert_eq!(
        dspy_config.max_optimization_iterations,
        cli_config.optimization.max_iterations as u32
    );
}

#[test]
fn test_validation_utils_module_name() {
    use rust_memvid_agent::cli::dspy::utils::ValidationUtils;

    // Valid names
    assert!(ValidationUtils::validate_module_name("valid_name").is_ok());
    assert!(ValidationUtils::validate_module_name("valid-name").is_ok());
    assert!(ValidationUtils::validate_module_name("ValidName123").is_ok());

    // Invalid names
    assert!(ValidationUtils::validate_module_name("").is_err());
    assert!(ValidationUtils::validate_module_name("invalid name").is_err());
    assert!(ValidationUtils::validate_module_name("invalid@name").is_err());

    // Too long name
    let long_name = "a".repeat(65);
    assert!(ValidationUtils::validate_module_name(&long_name).is_err());
}

#[test]
fn test_validation_utils_timeout() {
    use rust_memvid_agent::cli::dspy::utils::ValidationUtils;

    // Valid timeouts
    assert!(ValidationUtils::validate_timeout(1).is_ok());
    assert!(ValidationUtils::validate_timeout(300).is_ok());
    assert!(ValidationUtils::validate_timeout(86400).is_ok());

    // Invalid timeouts
    assert!(ValidationUtils::validate_timeout(0).is_err());
    assert!(ValidationUtils::validate_timeout(86401).is_err());
}

#[test]
fn test_validation_utils_iterations() {
    use rust_memvid_agent::cli::dspy::utils::ValidationUtils;

    // Valid iterations
    assert!(ValidationUtils::validate_iterations(1).is_ok());
    assert!(ValidationUtils::validate_iterations(100).is_ok());
    assert!(ValidationUtils::validate_iterations(10000).is_ok());

    // Invalid iterations
    assert!(ValidationUtils::validate_iterations(0).is_err());
    assert!(ValidationUtils::validate_iterations(10001).is_err());
}

#[test]
fn test_string_utils_functions() {
    use rust_memvid_agent::cli::dspy::utils::StringUtils;

    // Test truncate
    assert_eq!(StringUtils::truncate("hello", 10), "hello");
    assert_eq!(StringUtils::truncate("hello world", 8), "hello...");
    assert_eq!(StringUtils::truncate("hi", 5), "hi");

    // Test format_file_size
    assert_eq!(StringUtils::format_file_size(1024), "1.00 KB");
    assert_eq!(StringUtils::format_file_size(1048576), "1.00 MB");
    assert_eq!(StringUtils::format_file_size(500), "500 B");

    // Test parse_comma_separated
    assert_eq!(
        StringUtils::parse_comma_separated("a,b,c"),
        vec!["a", "b", "c"]
    );
    assert_eq!(
        StringUtils::parse_comma_separated("a, b , c "),
        vec!["a", "b", "c"]
    );
    assert_eq!(StringUtils::parse_comma_separated(""), Vec::<String>::new());
}

#[tokio::test]
async fn test_file_utils_ensure_dir_exists() {
    use rust_memvid_agent::cli::dspy::utils::FileUtils;

    let temp_dir = TempDir::new().unwrap();
    let test_dir = temp_dir.path().join("test_subdir");

    // Directory should not exist initially
    assert!(!test_dir.exists());

    // Ensure directory exists
    let result = FileUtils::ensure_dir_exists(&test_dir).await;
    assert!(result.is_ok(), "Should be able to create directory");
    assert!(test_dir.exists(), "Directory should exist after creation");

    // Calling again should not fail
    let result = FileUtils::ensure_dir_exists(&test_dir).await;
    assert!(
        result.is_ok(),
        "Should handle existing directory gracefully"
    );
}

#[tokio::test]
async fn test_file_utils_file_size() {
    use rust_memvid_agent::cli::dspy::utils::FileUtils;
    use tokio::fs;

    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.txt");

    // Create a test file with known content
    let content = "Hello, World!";
    fs::write(&test_file, content).await.unwrap();

    // Get file size
    let size = FileUtils::get_file_size(&test_file).await;
    assert!(size.is_ok(), "Should be able to get file size");
    assert_eq!(size.unwrap(), content.len() as u64);
}

#[test]
fn test_exit_code_conversion() {
    use rust_memvid_agent::cli::dspy::error::ExitCode;

    let config_error = DspyCliError::config_error("test", "test");
    let exit_code: ExitCode = config_error.into();
    assert_eq!(exit_code, ExitCode::ConfigurationError);

    let validation_error = DspyCliError::validation_error("field", "message", vec![]);
    let exit_code: ExitCode = validation_error.into();
    assert_eq!(exit_code, ExitCode::ValidationError);
}

#[test]
fn test_error_severity_and_category() {
    use rust_memvid_agent::cli::dspy::error::ErrorSeverity;

    let error = DspyCliError::internal_error("test");
    assert_eq!(error.severity(), ErrorSeverity::Critical);
    assert_eq!(error.category(), "internal");
    assert!(!error.is_retry_possible());

    let network_error = DspyCliError::network_error("test", None);
    assert_eq!(network_error.severity(), ErrorSeverity::Warning);
    assert_eq!(network_error.category(), "network");
    assert!(network_error.is_retry_possible());
}
