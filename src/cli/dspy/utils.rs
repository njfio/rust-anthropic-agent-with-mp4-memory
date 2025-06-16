//! DSPy CLI Utilities
//!
//! This module provides shared utilities and helper functions for DSPy CLI operations.

use crate::cli::dspy::commands::OutputFormat;
use crate::cli::dspy::error::{DspyCliError, DspyCliResult};
use serde::Serialize;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tabled::{Table, Tabled};
use tracing::{debug, info};

/// Output formatter for different formats
pub struct OutputFormatter;

impl OutputFormatter {
    /// Format and print output based on the specified format
    pub fn print<T>(data: &T, format: OutputFormat) -> DspyCliResult<()>
    where
        T: Serialize + Tabled,
    {
        match format {
            OutputFormat::Table => Self::print_table(data),
            OutputFormat::Json => Self::print_json(data),
            OutputFormat::Yaml => Self::print_yaml(data),
            OutputFormat::Csv => Self::print_csv(data),
            OutputFormat::Chart => Self::print_chart(data),
        }
    }

    /// Print data as a table
    fn print_table<T: Tabled>(data: &T) -> DspyCliResult<()> {
        let table = Table::new([data]).to_string();
        println!("{}", table);
        Ok(())
    }

    /// Print data as JSON
    fn print_json<T: Serialize>(data: &T) -> DspyCliResult<()> {
        let json = serde_json::to_string_pretty(data).map_err(|e| {
            DspyCliError::internal_error(format!("JSON serialization failed: {}", e))
        })?;
        println!("{}", json);
        Ok(())
    }

    /// Print data as YAML
    fn print_yaml<T: Serialize>(data: &T) -> DspyCliResult<()> {
        let yaml = serde_yaml::to_string(data).map_err(|e| {
            DspyCliError::internal_error(format!("YAML serialization failed: {}", e))
        })?;
        println!("{}", yaml);
        Ok(())
    }

    /// Print data as CSV
    fn print_csv<T: Serialize>(data: &T) -> DspyCliResult<()> {
        let mut writer = csv::Writer::from_writer(io::stdout());
        writer.serialize(data).map_err(|e| {
            DspyCliError::internal_error(format!("CSV serialization failed: {}", e))
        })?;
        writer
            .flush()
            .map_err(|e| DspyCliError::internal_error(format!("CSV flush failed: {}", e)))?;
        Ok(())
    }

    /// Print data as ASCII chart (placeholder implementation)
    fn print_chart<T: Serialize>(_data: &T) -> DspyCliResult<()> {
        println!("Chart output not yet implemented");
        Ok(())
    }

    /// Print a list of items with the specified format
    pub fn print_list<T>(items: &[T], format: OutputFormat) -> DspyCliResult<()>
    where
        T: Serialize + Tabled,
    {
        if items.is_empty() {
            println!("No items found.");
            return Ok(());
        }

        match format {
            OutputFormat::Table => {
                let table = Table::new(items).to_string();
                println!("{}", table);
            }
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(items).map_err(|e| {
                    DspyCliError::internal_error(format!("JSON serialization failed: {}", e))
                })?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(items).map_err(|e| {
                    DspyCliError::internal_error(format!("YAML serialization failed: {}", e))
                })?;
                println!("{}", yaml);
            }
            OutputFormat::Csv => {
                let mut writer = csv::Writer::from_writer(io::stdout());
                for item in items {
                    writer.serialize(item).map_err(|e| {
                        DspyCliError::internal_error(format!("CSV serialization failed: {}", e))
                    })?;
                }
                writer.flush().map_err(|e| {
                    DspyCliError::internal_error(format!("CSV flush failed: {}", e))
                })?;
            }
            OutputFormat::Chart => {
                println!("Chart output for lists not yet implemented");
            }
        }
        Ok(())
    }
}

/// Progress indicator for long-running operations
pub struct ProgressIndicator {
    operation: String,
    total_steps: Option<usize>,
    current_step: usize,
    show_progress: bool,
}

impl ProgressIndicator {
    /// Create a new progress indicator
    pub fn new(operation: impl Into<String>, total_steps: Option<usize>) -> Self {
        Self {
            operation: operation.into(),
            total_steps,
            current_step: 0,
            show_progress: atty::is(atty::Stream::Stdout),
        }
    }

    /// Update progress with a message
    pub fn update(&mut self, message: impl Into<String>) {
        if !self.show_progress {
            return;
        }

        self.current_step += 1;
        let progress_info = if let Some(total) = self.total_steps {
            format!("[{}/{}]", self.current_step, total)
        } else {
            format!("[{}]", self.current_step)
        };

        print!("\r{} {} {}", self.operation, progress_info, message.into());
        io::stdout().flush().unwrap_or(());
    }

    /// Finish the progress indicator
    pub fn finish(&self, message: impl Into<String>) {
        if self.show_progress {
            println!("\r{} ✓ {}", self.operation, message.into());
        } else {
            println!("{}: {}", self.operation, message.into());
        }
    }

    /// Finish with error
    pub fn finish_with_error(&self, error: impl Into<String>) {
        if self.show_progress {
            println!("\r{} ✗ {}", self.operation, error.into());
        } else {
            eprintln!("{}: Error - {}", self.operation, error.into());
        }
    }
}

/// File system utilities
pub struct FileUtils;

impl FileUtils {
    /// Ensure a directory exists, creating it if necessary
    pub async fn ensure_dir_exists(path: &Path) -> DspyCliResult<()> {
        if !path.exists() {
            debug!("Creating directory: {}", path.display());
            tokio::fs::create_dir_all(path).await.map_err(|e| {
                DspyCliError::resource_error(
                    "filesystem",
                    format!("Failed to create directory {}: {}", path.display(), e),
                    "Check directory permissions and available disk space",
                )
            })?;
            info!("Created directory: {}", path.display());
        }
        Ok(())
    }

    /// Check if a file has a valid extension
    pub fn has_valid_extension(path: &Path, allowed_extensions: &[String]) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            let ext_with_dot = format!(".{}", extension);
            allowed_extensions.contains(&ext_with_dot)
        } else {
            false
        }
    }

    /// Get file size in bytes
    pub async fn get_file_size(path: &Path) -> DspyCliResult<u64> {
        let metadata = tokio::fs::metadata(path).await.map_err(|e| {
            DspyCliError::resource_error(
                "filesystem",
                format!("Failed to get file metadata for {}: {}", path.display(), e),
                "Check if file exists and is accessible",
            )
        })?;
        Ok(metadata.len())
    }

    /// Check if file size is within limit
    pub async fn check_file_size_limit(path: &Path, max_size_mb: usize) -> DspyCliResult<()> {
        let size_bytes = Self::get_file_size(path).await?;
        let max_size_bytes = max_size_mb * 1024 * 1024;

        if size_bytes > max_size_bytes as u64 {
            return Err(DspyCliError::resource_error(
                "file_size",
                format!("File {} exceeds size limit", path.display()),
                format!(
                    "File size: {} MB, Limit: {} MB",
                    size_bytes / 1024 / 1024,
                    max_size_mb
                ),
            ));
        }
        Ok(())
    }

    /// Create a backup of a file
    pub async fn create_backup(path: &Path, backup_dir: &Path) -> DspyCliResult<PathBuf> {
        Self::ensure_dir_exists(backup_dir).await?;

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = path.file_name().ok_or_else(|| {
            DspyCliError::validation_error(
                "file_path",
                "Invalid file path",
                vec!["Provide a valid file path".to_string()],
            )
        })?;

        let backup_filename = format!("{}_{}", timestamp, filename.to_string_lossy());
        let backup_path = backup_dir.join(backup_filename);

        tokio::fs::copy(path, &backup_path).await.map_err(|e| {
            DspyCliError::resource_error(
                "filesystem",
                format!("Failed to create backup: {}", e),
                "Check disk space and permissions",
            )
        })?;

        info!("Created backup: {}", backup_path.display());
        Ok(backup_path)
    }
}

/// Input validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate a module name
    pub fn validate_module_name(name: &str) -> DspyCliResult<()> {
        if name.is_empty() {
            return Err(DspyCliError::validation_error(
                "module_name",
                "Module name cannot be empty",
                vec!["Provide a non-empty module name".to_string()],
            ));
        }

        if name.len() > 64 {
            return Err(DspyCliError::validation_error(
                "module_name",
                "Module name too long",
                vec!["Module name must be 64 characters or less".to_string()],
            ));
        }

        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(DspyCliError::validation_error(
                "module_name",
                "Invalid characters in module name",
                vec!["Use only alphanumeric characters, underscores, and hyphens".to_string()],
            ));
        }

        Ok(())
    }

    /// Validate a file path
    pub fn validate_file_path(path: &Path, must_exist: bool) -> DspyCliResult<()> {
        if must_exist && !path.exists() {
            return Err(DspyCliError::validation_error(
                "file_path",
                format!("File does not exist: {}", path.display()),
                vec!["Check the file path and ensure the file exists".to_string()],
            ));
        }

        if path.is_dir() {
            return Err(DspyCliError::validation_error(
                "file_path",
                format!("Path is a directory, not a file: {}", path.display()),
                vec!["Provide a file path, not a directory path".to_string()],
            ));
        }

        Ok(())
    }

    /// Validate timeout value
    pub fn validate_timeout(timeout_seconds: u64) -> DspyCliResult<()> {
        if timeout_seconds == 0 {
            return Err(DspyCliError::validation_error(
                "timeout",
                "Timeout must be greater than 0",
                vec!["Provide a positive timeout value in seconds".to_string()],
            ));
        }

        if timeout_seconds > 86400 {
            return Err(DspyCliError::validation_error(
                "timeout",
                "Timeout too large (max 24 hours)",
                vec!["Provide a timeout value less than 86400 seconds (24 hours)".to_string()],
            ));
        }

        Ok(())
    }

    /// Validate iteration count
    pub fn validate_iterations(iterations: usize) -> DspyCliResult<()> {
        if iterations == 0 {
            return Err(DspyCliError::validation_error(
                "iterations",
                "Iterations must be greater than 0",
                vec!["Provide a positive number of iterations".to_string()],
            ));
        }

        if iterations > 10000 {
            return Err(DspyCliError::validation_error(
                "iterations",
                "Too many iterations (max 10000)",
                vec!["Provide a reasonable number of iterations (≤ 10000)".to_string()],
            ));
        }

        Ok(())
    }
}

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Truncate a string to a maximum length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    /// Format duration in human-readable format
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else {
            format!("{}ms", millis)
        }
    }

    /// Format file size in human-readable format
    pub fn format_file_size(size_bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = size_bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", size_bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Parse comma-separated values
    pub fn parse_comma_separated(input: &str) -> Vec<String> {
        input
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// User interaction utilities
pub struct InteractionUtils;

impl InteractionUtils {
    /// Prompt user for confirmation
    pub fn confirm(message: &str, default: bool) -> DspyCliResult<bool> {
        let default_char = if default { 'Y' } else { 'N' };
        let prompt = format!("{} [{}]: ", message, if default { "Y/n" } else { "y/N" });

        print!("{}", prompt);
        io::stdout()
            .flush()
            .map_err(|e| DspyCliError::internal_error(format!("Failed to flush stdout: {}", e)))?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| DspyCliError::internal_error(format!("Failed to read input: {}", e)))?;

        let input = input.trim().to_lowercase();
        match input.as_str() {
            "y" | "yes" => Ok(true),
            "n" | "no" => Ok(false),
            "" => Ok(default),
            _ => {
                println!("Please enter 'y' or 'n'");
                Self::confirm(message, default)
            }
        }
    }

    /// Prompt user for input with validation
    pub fn prompt_input(
        message: &str,
        validator: Option<fn(&str) -> bool>,
    ) -> DspyCliResult<String> {
        print!("{}: ", message);
        io::stdout()
            .flush()
            .map_err(|e| DspyCliError::internal_error(format!("Failed to flush stdout: {}", e)))?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| DspyCliError::internal_error(format!("Failed to read input: {}", e)))?;

        let input = input.trim().to_string();

        if let Some(validator) = validator {
            if !validator(&input) {
                println!("Invalid input. Please try again.");
                return Self::prompt_input(message, Some(validator));
            }
        }

        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_string_utils_truncate() {
        assert_eq!(StringUtils::truncate("hello", 10), "hello");
        assert_eq!(StringUtils::truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_string_utils_format_duration() {
        let duration = std::time::Duration::from_millis(1500);
        assert_eq!(StringUtils::format_duration(duration), "1.500s");

        let duration = std::time::Duration::from_secs(3661);
        assert_eq!(StringUtils::format_duration(duration), "1h 1m 1s");
    }

    #[test]
    fn test_string_utils_format_file_size() {
        assert_eq!(StringUtils::format_file_size(1024), "1.00 KB");
        assert_eq!(StringUtils::format_file_size(1048576), "1.00 MB");
        assert_eq!(StringUtils::format_file_size(500), "500 B");
    }

    #[test]
    fn test_string_utils_parse_comma_separated() {
        let result = StringUtils::parse_comma_separated("a,b,c");
        assert_eq!(result, vec!["a", "b", "c"]);

        let result = StringUtils::parse_comma_separated("a, b , c ");
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_validation_utils_module_name() {
        assert!(ValidationUtils::validate_module_name("valid_name").is_ok());
        assert!(ValidationUtils::validate_module_name("valid-name").is_ok());
        assert!(ValidationUtils::validate_module_name("").is_err());
        assert!(ValidationUtils::validate_module_name("invalid name").is_err());
    }

    #[test]
    fn test_validation_utils_timeout() {
        assert!(ValidationUtils::validate_timeout(30).is_ok());
        assert!(ValidationUtils::validate_timeout(0).is_err());
        assert!(ValidationUtils::validate_timeout(100000).is_err());
    }

    #[tokio::test]
    async fn test_file_utils_ensure_dir_exists() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("test_dir");

        assert!(!test_dir.exists());
        FileUtils::ensure_dir_exists(&test_dir).await.unwrap();
        assert!(test_dir.exists());
    }
}
