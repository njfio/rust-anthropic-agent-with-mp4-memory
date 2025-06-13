use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_string_param, Tool, ToolResult};
use crate::utils::audit_logger::{audit_log, AuditEvent, AuditEventType, AuditSeverity};
use crate::utils::error::{AgentError, Result};
use crate::utils::validation::{validate_file_content, validate_path};

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
    /// SECURITY: All paths must be relative to working directory to prevent path traversal
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let path = Path::new(path);

        // SECURITY FIX: Reject absolute paths to prevent path traversal
        if path.is_absolute() {
            return Err(AgentError::invalid_input(
                "Absolute paths not allowed for security reasons. Use relative paths only.",
            ));
        }

        // Prevent directory traversal attacks with .. components
        if path
            .components()
            .any(|comp| matches!(comp, std::path::Component::ParentDir))
        {
            return Err(AgentError::invalid_input(
                "Path traversal not allowed (..). Use relative paths within working directory only.",
            ));
        }

        // All paths are now relative to working directory
        let resolved = self.working_dir.join(path);

        // Additional security check: ensure resolved path is still within working directory
        let canonical_working_dir = self
            .working_dir
            .canonicalize()
            .map_err(|_| AgentError::invalid_input("Invalid working directory"))?;

        if let Ok(canonical_resolved) = resolved.canonicalize() {
            if !canonical_resolved.starts_with(&canonical_working_dir) {
                return Err(AgentError::invalid_input(
                    "Path resolves outside working directory",
                ));
            }
        }

        Ok(resolved)
    }

    /// Handle the view command
    async fn handle_view(&self, input: &serde_json::Value) -> Result<ToolResult> {
        let path_str = extract_string_param(input, "path")?;

        // SECURITY: Validate path input
        validate_path(&path_str)?;

        let resolved_path = self.resolve_path(&path_str)?;

        // AUDIT: Log file access attempt
        audit_log(
            AuditEvent::new(
                AuditEventType::FileAccess,
                AuditSeverity::Low,
                "file_view".to_string(),
            )
            .with_resource(&path_str),
        );

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
        info!("📝 CREATE operation");
        // SECURITY: Move sensitive parameter logging to DEBUG level
        debug!(
            "🔍 ALL PARAMETERS: {}",
            serde_json::to_string_pretty(input).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        // List all available keys in the input
        if let Some(obj) = input.as_object() {
            let keys: Vec<&String> = obj.keys().collect();
            debug!("🔑 Available parameter keys: {:?}", keys);
        }

        // Try to get path parameter with better error handling
        let path_str = match extract_string_param(input, "path") {
            Ok(path) => {
                debug!("📁 Path: {}", path); // SECURITY: Move to DEBUG level
                                             // SECURITY: Validate path input
                validate_path(&path)?;
                path
            }
            Err(_) => {
                debug!("❌ Missing path parameter"); // SECURITY: Move to DEBUG level
                return Ok(ToolResult::error(
                    "Missing required parameter 'path' for create command. Please provide the file path."
                ));
            }
        };

        // Try multiple possible parameter names for file content
        let file_text = if let Ok(text) = extract_string_param(input, "file_text") {
            debug!("📄 Found file_text parameter: {} chars", text.len()); // SECURITY: Move to DEBUG level
                                                                          // SECURITY: Validate file content
            validate_file_content(&text)?;
            text
        } else if let Ok(text) = extract_string_param(input, "content") {
            debug!("📄 Found content parameter: {} chars", text.len()); // SECURITY: Move to DEBUG level
                                                                        // SECURITY: Validate file content
            validate_file_content(&text)?;
            text
        } else if let Ok(text) = extract_string_param(input, "text") {
            debug!("📄 Found text parameter: {} chars", text.len()); // SECURITY: Move to DEBUG level
                                                                     // SECURITY: Validate file content
            validate_file_content(&text)?;
            text
        } else {
            debug!("❌ Missing file content parameter - tried file_text, content, text"); // SECURITY: Move to DEBUG level
            return Ok(ToolResult::error(
                "Missing required parameter for file content. Please provide 'file_text', 'content', or 'text' parameter with the content to write to the file."
            ));
        };

        let resolved_path = self.resolve_path(&path_str)?;
        debug!("📍 Resolved path: {:?}", resolved_path); // SECURITY: Move to DEBUG level

        if resolved_path.exists() {
            debug!("❌ File already exists"); // SECURITY: Move to DEBUG level
            return Ok(ToolResult::error(format!(
                "File already exists: {}",
                path_str
            )));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = resolved_path.parent() {
            debug!("📂 Creating parent directories: {:?}", parent); // SECURITY: Move to DEBUG level
            fs::create_dir_all(parent)?;
        }

        fs::write(&resolved_path, file_text)?;
        info!("✅ Successfully created file"); // SECURITY: Remove path from INFO log
        debug!("💾 File written successfully to: {:?}", resolved_path); // SECURITY: Move detailed info to DEBUG

        // AUDIT: Log successful file creation
        audit_log(
            AuditEvent::new(
                AuditEventType::FileModification,
                AuditSeverity::Medium,
                "file_create".to_string(),
            )
            .with_resource(&path_str)
            .with_success(true),
        );

        Ok(ToolResult::success(format!(
            "File created successfully at {}",
            path_str
        )))
    }

    /// Handle the str_replace command
    async fn handle_str_replace(&self, input: &serde_json::Value) -> Result<ToolResult> {
        info!("🔄 STR_REPLACE operation");
        // SECURITY: Move sensitive parameter logging to DEBUG level
        debug!(
            "🔍 ALL PARAMETERS: {}",
            serde_json::to_string_pretty(input).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        // List all available keys in the input
        if let Some(obj) = input.as_object() {
            let keys: Vec<&String> = obj.keys().collect();
            debug!("🔑 Available parameter keys: {:?}", keys);
        }

        let path_str = match extract_string_param(input, "path") {
            Ok(path) => {
                info!("📁 Path: {}", path);
                path
            }
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
            }
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
            }
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
            }
            1 => {
                let new_content = content.replace(&old_str, &new_str);
                fs::write(&resolved_path, new_content)?;
                info!("✅ Successfully replaced text in {:?}", resolved_path);
                info!("💾 File written successfully!");
                Ok(ToolResult::success(
                    "Successfully replaced text at exactly one location",
                ))
            }
            n => {
                info!("⚠️  Found {} matches for replacement text", n);
                Ok(ToolResult::error(format!(
                    "Found {} matches for replacement text. Please provide more context to make a unique match.",
                    n
                )))
            }
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
        // SECURITY: Move sensitive logging to DEBUG level
        info!("🔧 LOCAL TEXT EDITOR TOOL CALLED");
        debug!(
            "📥 Input received: {}",
            serde_json::to_string_pretty(&input).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

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
