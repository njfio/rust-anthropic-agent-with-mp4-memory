use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};

/// Local implementation of Anthropic's str_replace_based_edit_tool
/// This tool runs locally and can actually modify files on the user's machine
/// It implements the exact same interface as Anthropic's text editor tool
#[derive(Debug, Clone)]
pub struct LocalTextEditorTool {
    /// Working directory for file operations
    working_dir: PathBuf,
    /// Maximum file size to read (in bytes)
    max_file_size: usize,
}

impl LocalTextEditorTool {
    /// Create a new local text editor tool
    pub fn new<P: Into<PathBuf>>(working_dir: P) -> Self {
        Self {
            working_dir: working_dir.into(),
            max_file_size: 10 * 1024 * 1024, // 10MB default
        }
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

    /// Handle the view command
    async fn handle_view(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;
        let resolved_path = self.resolve_path(&path_str)?;

        if !resolved_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        if resolved_path.is_dir() {
            // List directory contents
            let entries = fs::read_dir(&resolved_path)?;
            let mut items = Vec::new();
            
            for entry in entries {
                let entry = entry?;
                let file_name = entry.file_name().to_string_lossy().to_string();
                let file_type = if entry.file_type()?.is_dir() {
                    "directory"
                } else {
                    "file"
                };
                items.push(format!("{} ({})", file_name, file_type));
            }

            items.sort();
            let content = if items.is_empty() {
                "Directory is empty".to_string()
            } else {
                format!("Directory contents:\n{}", items.join("\n"))
            };

            Ok(ToolResult::success(content))
        } else {
            // Read file contents
            let metadata = fs::metadata(&resolved_path)?;
            if metadata.len() > self.max_file_size as u64 {
                return Ok(ToolResult::error(format!(
                    "File too large ({} bytes). Maximum size is {} bytes.",
                    metadata.len(),
                    self.max_file_size
                )));
            }

            let content = fs::read_to_string(&resolved_path)?;
            let lines: Vec<&str> = content.lines().collect();

            // Return file with line numbers
            let numbered_lines: Vec<String> = lines
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{}: {}", i + 1, line))
                .collect();

            Ok(ToolResult::success(numbered_lines.join("\n")))
        }
    }

    /// Handle the create command (matches Anthropic's interface)
    async fn handle_create(&self, input: &serde_json::Value) -> Result<ToolResult> {
        info!("📝 CREATE operation:");
        info!("🔍 ALL PARAMETERS: {}", serde_json::to_string_pretty(input).unwrap_or_else(|_| "Invalid JSON".to_string()));

        // List all available keys in the input
        if let Some(obj) = input.as_object() {
            let keys: Vec<&String> = obj.keys().collect();
            info!("🔑 Available parameter keys: {:?}", keys);
        }

        // Try to get path parameter with better error handling
        let path_str = match extract_string_param(input, "path") {
            Ok(path) => {
                info!("📁 Path: {}", path);
                path
            },
            Err(_) => {
                info!("❌ Missing path parameter");
                return Ok(ToolResult::error(
                    "Missing required parameter 'path' for create command. Please provide the file path."
                ));
            }
        };

        // Try multiple possible parameter names for file content
        let file_text = if let Ok(text) = extract_string_param(input, "file_text") {
            info!("📄 Found file_text parameter: {} chars", text.len());
            text
        } else if let Ok(text) = extract_string_param(input, "content") {
            info!("📄 Found content parameter: {} chars", text.len());
            text
        } else if let Ok(text) = extract_string_param(input, "text") {
            info!("📄 Found text parameter: {} chars", text.len());
            text
        } else {
            info!("❌ Missing file content parameter - tried file_text, content, text");
            return Ok(ToolResult::error(
                "Missing required parameter for file content. Please provide 'file_text', 'content', or 'text' parameter with the content to write to the file."
            ));
        };

        let resolved_path = self.resolve_path(&path_str)?;
        info!("📍 Resolved path: {:?}", resolved_path);

        if resolved_path.exists() {
            info!("❌ File already exists");
            return Ok(ToolResult::error(format!("File already exists: {}", path_str)));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = resolved_path.parent() {
            info!("📂 Creating parent directories: {:?}", parent);
            fs::create_dir_all(parent)?;
        }

        fs::write(&resolved_path, file_text)?;
        info!("✅ Successfully created file: {:?}", resolved_path);
        info!("💾 File written successfully!");
        Ok(ToolResult::success(format!("File created successfully at {}", path_str)))
    }

    /// Handle the str_replace command
    async fn handle_str_replace(&self, input: &serde_json::Value) -> Result<ToolResult> {
        info!("🔄 STR_REPLACE operation:");
        info!("🔍 ALL PARAMETERS: {}", serde_json::to_string_pretty(input).unwrap_or_else(|_| "Invalid JSON".to_string()));

        // List all available keys in the input
        if let Some(obj) = input.as_object() {
            let keys: Vec<&String> = obj.keys().collect();
            info!("🔑 Available parameter keys: {:?}", keys);
        }

        let path_str = match extract_string_param(input, "path") {
            Ok(path) => {
                info!("📁 Path: {}", path);
                path
            },
            Err(_) => {
                info!("❌ Missing path parameter");
                return Ok(ToolResult::error(
                    "Missing required parameter 'path' for str_replace command. Please provide the file path."
                ));
            }
        };

        let old_str = match extract_string_param(input, "old_str") {
            Ok(old) => {
                info!("🔍 Old text: {:?} ({} chars)", old, old.len());
                old
            },
            Err(_) => {
                info!("❌ Missing old_str parameter");
                return Ok(ToolResult::error(
                    "Missing required parameter 'old_str' for str_replace command. Please provide the text to find and replace."
                ));
            }
        };

        let new_str = match extract_string_param(input, "new_str") {
            Ok(new) => {
                info!("✏️  New text: {:?} ({} chars)", new, new.len());
                new
            },
            Err(_) => {
                info!("❌ Missing new_str parameter");
                return Ok(ToolResult::error(
                    "Missing required parameter 'new_str' for str_replace command. Please provide the replacement text. Use empty string \"\" if you want to delete the old text."
                ));
            }
        };

        let resolved_path = self.resolve_path(&path_str)?;
        info!("📍 Resolved path: {:?}", resolved_path);

        if !resolved_path.exists() {
            return Ok(ToolResult::error("File not found"));
        }

        let content = fs::read_to_string(&resolved_path)?;
        let matches = content.matches(&old_str).count();

        match matches {
            0 => {
                info!("❌ No match found for replacement");
                Ok(ToolResult::error("No match found for replacement"))
            },
            1 => {
                let new_content = content.replace(&old_str, &new_str);
                fs::write(&resolved_path, new_content)?;
                info!("✅ Successfully replaced text in {:?}", resolved_path);
                info!("💾 File written successfully!");
                Ok(ToolResult::success("Successfully replaced text at exactly one location"))
            }
            n => {
                info!("⚠️  Found {} matches for replacement text", n);
                Ok(ToolResult::error(format!(
                    "Found {} matches for replacement text. Please provide more context to make a unique match.",
                    n
                )))
            },
        }
    }
}

#[async_trait]
impl Tool for LocalTextEditorTool {
    fn definition(&self) -> ToolDefinition {
        // Local text editor tool with custom implementation
        create_tool_definition(
            "str_replace_based_edit_tool",
            "Local text editor for viewing and editing files. Commands: view (path, optional view_range), create (path, file_text), str_replace (path, old_str, new_str)",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "str_replace", "create"],
                        "description": "The operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to file"
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Content for create command (required for create)"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Text to find for str_replace command (required for str_replace)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Text to replace with for str_replace command (required for str_replace)"
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Line range to view [start, end]"
                    }
                },
                "required": ["command", "path"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        // Debug: Log the full input to see what parameters we're receiving
        info!("🔧 LOCAL TEXT EDITOR TOOL CALLED");
        info!("📥 Input received: {}", serde_json::to_string_pretty(&input).unwrap_or_else(|_| "Invalid JSON".to_string()));

        let command = extract_string_param(&input, "command")?;

        info!("⚡ Executing local file operation: {}", command);

        match command.as_str() {
            "view" => self.handle_view(&input).await,
            "create" => self.handle_create(&input).await,
            "str_replace" => self.handle_str_replace(&input).await,
            _ => Ok(ToolResult::error(format!("Unknown command: {}", command))),
        }
    }

    fn name(&self) -> &str {
        "str_replace_based_edit_tool"
    }

    fn description(&self) -> Option<&str> {
        Some("Tool for viewing and editing files using str_replace operations (local implementation)")
    }
}
