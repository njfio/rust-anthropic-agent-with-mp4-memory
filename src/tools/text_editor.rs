use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_optional_int_param, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};

/// Text editor tool implementation for Anthropic's text editor
#[derive(Debug, Clone)]
pub struct TextEditorTool {
    /// Working directory for file operations
    working_dir: PathBuf,
    /// Whether to create backups before editing
    create_backups: bool,
    /// Maximum file size to read (in bytes)
    max_file_size: usize,
}

impl TextEditorTool {
    /// Create a new text editor tool
    pub fn new<P: Into<PathBuf>>(working_dir: P) -> Self {
        Self {
            working_dir: working_dir.into(),
            create_backups: true,
            max_file_size: 10 * 1024 * 1024, // 10MB default
        }
    }

    /// Set whether to create backups
    pub fn with_backups(mut self, create_backups: bool) -> Self {
        self.create_backups = create_backups;
        self
    }

    /// Set maximum file size
    pub fn with_max_file_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
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

    /// Create a backup of a file
    fn create_backup(&self, file_path: &Path) -> Result<()> {
        if !self.create_backups || !file_path.exists() {
            return Ok(());
        }

        let backup_path = file_path.with_extension(
            format!("{}.backup", 
                file_path.extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("txt")
            )
        );

        fs::copy(file_path, &backup_path)?;
        debug!("Created backup: {:?}", backup_path);
        Ok(())
    }

    /// Handle the view command
    async fn handle_view(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let resolved_path = self.resolve_path(&path_str)?;

        if resolved_path.is_dir() {
            self.view_directory(&resolved_path).await
        } else {
            self.view_file(&resolved_path, input).await
        }
    }

    /// View a directory
    async fn view_directory(&self, dir_path: &Path) -> Result<ToolResult> {
        if !dir_path.exists() {
            return Ok(ToolResult::error("Directory not found"));
        }

        let mut entries = Vec::new();
        let read_dir = fs::read_dir(dir_path)?;

        for entry in read_dir {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().to_string();
            let file_type = if entry.file_type()?.is_dir() {
                "directory"
            } else {
                "file"
            };
            entries.push(format!("{} ({})", file_name, file_type));
        }

        entries.sort();
        let content = if entries.is_empty() {
            "Directory is empty".to_string()
        } else {
            format!("Directory contents:\n{}", entries.join("\n"))
        };

        Ok(ToolResult::success(content))
    }

    /// View a file
    async fn view_file(&self, file_path: &Path, input: &serde_json::Value) -> Result<ToolResult> {
        if !file_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        let metadata = fs::metadata(file_path)?;
        if metadata.len() > self.max_file_size as u64 {
            return Ok(ToolResult::error(format!(
                "File too large ({} bytes). Maximum size is {} bytes.",
                metadata.len(),
                self.max_file_size
            )));
        }

        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Handle view_range parameter
        if let Some(view_range) = input.get("view_range").and_then(|v| v.as_array()) {
            if view_range.len() == 2 {
                let start = view_range[0].as_i64().unwrap_or(1) as usize;
                let end = if view_range[1].as_i64().unwrap_or(-1) == -1 {
                    lines.len()
                } else {
                    view_range[1].as_i64().unwrap_or(1) as usize
                };

                if start > 0 && start <= lines.len() {
                    let start_idx = start - 1; // Convert to 0-based indexing
                    let end_idx = std::cmp::min(end, lines.len());
                    
                    let selected_lines: Vec<String> = lines[start_idx..end_idx]
                        .iter()
                        .enumerate()
                        .map(|(i, line)| format!("{}: {}", start_idx + i + 1, line))
                        .collect();
                    
                    return Ok(ToolResult::success(selected_lines.join("\n")));
                }
            }
        }

        // Return full file with line numbers
        let numbered_lines: Vec<String> = lines
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}: {}", i + 1, line))
            .collect();

        Ok(ToolResult::success(numbered_lines.join("\n")))
    }

    /// Handle the str_replace command
    async fn handle_str_replace(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let old_str = extract_string_param(input, "old_str")?;
        let new_str = extract_string_param(input, "new_str")?;

        let resolved_path = self.resolve_path(&path_str)?;

        if !resolved_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        self.create_backup(&resolved_path)?;

        let content = fs::read_to_string(&resolved_path)?;
        let matches = content.matches(&old_str).count();

        match matches {
            0 => Ok(ToolResult::error("No match found for replacement")),
            1 => {
                let new_content = content.replace(&old_str, &new_str);
                fs::write(&resolved_path, new_content)?;
                info!("Successfully replaced text in {:?}", resolved_path);
                Ok(ToolResult::success("Successfully replaced text at exactly one location"))
            }
            n => Ok(ToolResult::error(format!(
                "Found {} matches for replacement text. Please provide more context to make a unique match.",
                n
            ))),
        }
    }

    /// Handle the create command
    async fn handle_create(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let file_text = extract_string_param(input, "file_text")?;

        let resolved_path = self.resolve_path(&path_str)?;

        if resolved_path.exists() {
            return Ok(ToolResult::error("File already exists"));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = resolved_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&resolved_path, file_text)?;
        info!("Created file: {:?}", resolved_path);
        Ok(ToolResult::success(format!("Successfully created file: {}", path_str)))
    }

    /// Handle the insert command
    async fn handle_insert(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let insert_line = extract_optional_int_param(input, "insert_line").unwrap_or(0) as usize;
        let new_str = extract_string_param(input, "new_str")?;

        let resolved_path = self.resolve_path(&path_str)?;

        if !resolved_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        self.create_backup(&resolved_path)?;

        let content = fs::read_to_string(&resolved_path)?;
        let mut lines: Vec<&str> = content.lines().collect();

        if insert_line > lines.len() {
            return Ok(ToolResult::error(format!(
                "Insert line {} is beyond file length ({})",
                insert_line,
                lines.len()
            )));
        }

        // Insert the new text
        let new_lines: Vec<&str> = new_str.lines().collect();
        for (i, line) in new_lines.iter().enumerate() {
            lines.insert(insert_line + i, line);
        }

        let new_content = lines.join("\n");
        fs::write(&resolved_path, new_content)?;
        info!("Inserted text at line {} in {:?}", insert_line, resolved_path);
        Ok(ToolResult::success(format!(
            "Successfully inserted text at line {}",
            insert_line
        )))
    }

    /// Handle the undo_edit command (for older models)
    async fn handle_undo_edit(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let resolved_path = self.resolve_path(&path_str)?;

        let backup_path = resolved_path.with_extension(
            format!("{}.backup", 
                resolved_path.extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("txt")
            )
        );

        if !backup_path.exists() {
            return Ok(ToolResult::error("No backup found to restore"));
        }

        fs::copy(&backup_path, &resolved_path)?;
        info!("Restored file from backup: {:?}", resolved_path);
        Ok(ToolResult::success("Successfully undid the last edit"))
    }
}

#[async_trait]
impl Tool for TextEditorTool {
    fn definition(&self) -> ToolDefinition {
        // For Anthropic's text editor tool, we use the specific type
        // Built-in tools don't accept custom descriptions or schemas
        ToolDefinition {
            tool_type: "text_editor_20250429".to_string(),
            name: "str_replace_based_edit_tool".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let command = extract_string_param(&input, "command")?;

        debug!("Executing text editor command: {}", command);

        match command.as_str() {
            "view" => self.handle_view(&input).await,
            "str_replace" => self.handle_str_replace(&input).await,
            "create" => self.handle_create(&input).await,
            "insert" => self.handle_insert(&input).await,
            "undo_edit" => self.handle_undo_edit(&input).await,
            _ => Ok(ToolResult::error(format!("Unknown command: {}", command))),
        }
    }

    fn name(&self) -> &str {
        "str_replace_based_edit_tool"
    }

    fn description(&self) -> Option<&str> {
        Some("Text editor tool for viewing and editing files")
    }
}
