use async_trait::async_trait;
use base64::Engine;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};

/// Tool for reading files
#[derive(Debug, Clone)]
pub struct FileReadTool {
    /// Working directory for file operations
    working_dir: PathBuf,
    /// Maximum file size to read (in bytes)
    max_file_size: usize,
    /// Allowed file extensions (None means all allowed)
    allowed_extensions: Option<Vec<String>>,
}

impl FileReadTool {
    /// Create a new file read tool
    pub fn new<P: Into<PathBuf>>(working_dir: P) -> Self {
        Self {
            working_dir: working_dir.into(),
            max_file_size: 10 * 1024 * 1024, // 10MB default
            allowed_extensions: None,
        }
    }

    /// Set maximum file size
    pub fn with_max_file_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
        self
    }

    /// Set allowed file extensions
    pub fn with_allowed_extensions(mut self, extensions: Vec<String>) -> Self {
        self.allowed_extensions = Some(extensions);
        self
    }

    /// Resolve a path relative to the working directory
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let path = Path::new(path);
        
        // Prevent directory traversal attacks
        if path.components().any(|comp| matches!(comp, std::path::Component::ParentDir)) {
            return Err(AgentError::invalid_input(
                "Path traversal not allowed (..)",
            ));
        }

        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir.join(path)
        };

        Ok(resolved)
    }

    /// Check if file extension is allowed
    fn is_extension_allowed(&self, path: &Path) -> bool {
        if let Some(allowed) = &self.allowed_extensions {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                allowed.iter().any(|a| a.eq_ignore_ascii_case(ext))
            } else {
                false
            }
        } else {
            true
        }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "file_read",
            "Read the contents of a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(&input, "path")?;
        let resolved_path = self.resolve_path(&path_str)?;

        debug!("Reading file: {:?}", resolved_path);

        if !resolved_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        if !resolved_path.is_file() {
            return Ok(ToolResult::error("Path is not a file"));
        }

        if !self.is_extension_allowed(&resolved_path) {
            return Ok(ToolResult::error("File extension not allowed"));
        }

        let metadata = fs::metadata(&resolved_path)?;
        if metadata.len() > self.max_file_size as u64 {
            return Ok(ToolResult::error(format!(
                "File too large ({} bytes). Maximum size is {} bytes.",
                metadata.len(),
                self.max_file_size
            )));
        }

        match fs::read_to_string(&resolved_path) {
            Ok(content) => {
                info!("Successfully read file: {:?} ({} chars)", resolved_path, content.len());
                Ok(ToolResult::success(content))
            }
            Err(e) => {
                warn!("Failed to read file as UTF-8: {:?}, error: {}", resolved_path, e);
                // Try reading as bytes and convert to base64 for binary files
                match fs::read(&resolved_path) {
                    Ok(bytes) => {
                        let base64_content = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        Ok(ToolResult::success(format!(
                            "Binary file content (base64 encoded):\n{}",
                            base64_content
                        )))
                    }
                    Err(e) => Ok(ToolResult::error(format!("Failed to read file: {}", e))),
                }
            }
        }
    }

    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> Option<&str> {
        Some("Read the contents of a file")
    }
}

/// Tool for writing files
#[derive(Debug, Clone)]
pub struct FileWriteTool {
    /// Working directory for file operations
    working_dir: PathBuf,
    /// Whether to allow overwriting existing files
    allow_overwrite: bool,
    /// Allowed file extensions (None means all allowed)
    allowed_extensions: Option<Vec<String>>,
}

impl FileWriteTool {
    /// Create a new file write tool
    pub fn new<P: Into<PathBuf>>(working_dir: P) -> Self {
        Self {
            working_dir: working_dir.into(),
            allow_overwrite: false,
            allowed_extensions: None,
        }
    }

    /// Set whether to allow overwriting existing files
    pub fn with_overwrite(mut self, allow_overwrite: bool) -> Self {
        self.allow_overwrite = allow_overwrite;
        self
    }

    /// Set allowed file extensions
    pub fn with_allowed_extensions(mut self, extensions: Vec<String>) -> Self {
        self.allowed_extensions = Some(extensions);
        self
    }

    /// Resolve a path relative to the working directory
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let path = Path::new(path);
        
        // Prevent directory traversal attacks
        if path.components().any(|comp| matches!(comp, std::path::Component::ParentDir)) {
            return Err(AgentError::invalid_input(
                "Path traversal not allowed (..)",
            ));
        }

        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir.join(path)
        };

        Ok(resolved)
    }

    /// Check if file extension is allowed
    fn is_extension_allowed(&self, path: &Path) -> bool {
        if let Some(allowed) = &self.allowed_extensions {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                allowed.iter().any(|a| a.eq_ignore_ascii_case(ext))
            } else {
                false
            }
        } else {
            true
        }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "file_write",
            "Write content to a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(&input, "path")?;
        let content = extract_string_param(&input, "content")?;
        let resolved_path = self.resolve_path(&path_str)?;

        debug!("Writing file: {:?} ({} chars)", resolved_path, content.len());

        if !self.is_extension_allowed(&resolved_path) {
            return Ok(ToolResult::error("File extension not allowed"));
        }

        if resolved_path.exists() && !self.allow_overwrite {
            return Ok(ToolResult::error("File already exists and overwrite is not allowed"));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = resolved_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Ok(ToolResult::error(format!("Failed to create parent directories: {}", e)));
            }
        }

        match fs::write(&resolved_path, content) {
            Ok(()) => {
                info!("Successfully wrote file: {:?}", resolved_path);
                Ok(ToolResult::success(format!("Successfully wrote file: {}", path_str)))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
    }

    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> Option<&str> {
        Some("Write content to a file")
    }
}

/// Tool for listing directory contents
#[derive(Debug, Clone)]
pub struct DirectoryListTool {
    /// Working directory for file operations
    working_dir: PathBuf,
}

impl DirectoryListTool {
    /// Create a new directory list tool
    pub fn new<P: Into<PathBuf>>(working_dir: P) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }

    /// Resolve a path relative to the working directory
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let path = Path::new(path);
        
        // Prevent directory traversal attacks
        if path.components().any(|comp| matches!(comp, std::path::Component::ParentDir)) {
            return Err(AgentError::invalid_input(
                "Path traversal not allowed (..)",
            ));
        }

        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir.join(path)
        };

        Ok(resolved)
    }
}

#[async_trait]
impl Tool for DirectoryListTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "directory_list",
            "List the contents of a directory",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list (default: current directory)",
                        "default": "."
                    }
                },
                "required": []
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let resolved_path = self.resolve_path(path_str)?;

        debug!("Listing directory: {:?}", resolved_path);

        if !resolved_path.exists() {
            return Ok(ToolResult::error("Directory not found"));
        }

        if !resolved_path.is_dir() {
            return Ok(ToolResult::error("Path is not a directory"));
        }

        match fs::read_dir(&resolved_path) {
            Ok(entries) => {
                let mut items = Vec::new();
                
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let file_name = entry.file_name().to_string_lossy().to_string();
                            let file_type = if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                                "directory"
                            } else {
                                "file"
                            };
                            
                            let size = if file_type == "file" {
                                entry.metadata()
                                    .map(|m| m.len())
                                    .unwrap_or(0)
                            } else {
                                0
                            };
                            
                            items.push(format!("{} ({}, {} bytes)", file_name, file_type, size));
                        }
                        Err(e) => {
                            warn!("Error reading directory entry: {}", e);
                        }
                    }
                }

                items.sort();
                let content = if items.is_empty() {
                    "Directory is empty".to_string()
                } else {
                    format!("Directory contents ({} items):\n{}", items.len(), items.join("\n"))
                };

                info!("Listed directory: {:?} ({} items)", resolved_path, items.len());
                Ok(ToolResult::success(content))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to read directory: {}", e))),
        }
    }

    fn name(&self) -> &str {
        "directory_list"
    }

    fn description(&self) -> Option<&str> {
        Some("List the contents of a directory")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_read_tool() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Hello, world!").unwrap();

        let tool = FileReadTool::new(temp_dir.path());
        let input = json!({"path": "test.txt"});
        
        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_file_write_tool() {
        let temp_dir = tempdir().unwrap();
        let tool = FileWriteTool::new(temp_dir.path()).with_overwrite(true);
        
        let input = json!({
            "path": "new_file.txt",
            "content": "Test content"
        });
        
        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        
        let written_content = fs::read_to_string(temp_dir.path().join("new_file.txt")).unwrap();
        assert_eq!(written_content, "Test content");
    }

    #[tokio::test]
    async fn test_directory_list_tool() {
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(temp_dir.path().join("file2.txt"), "content2").unwrap();
        fs::create_dir(temp_dir.path().join("subdir")).unwrap();

        let tool = DirectoryListTool::new(temp_dir.path());
        let input = json!({"path": "."});
        
        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("file1.txt"));
        assert!(result.content.contains("file2.txt"));
        assert!(result.content.contains("subdir"));
    }
}
