use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::anthropic::models::ToolDefinition;

/// Types of tools supported by Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    /// Custom user-defined tool
    Custom,
    /// Text editor tool (client-side)
    TextEditor20250429,
    TextEditor20250124,
    TextEditor20241022,
    /// Code execution tool (server-side)
    CodeExecution20250522,
    /// Web search tool (server-side)
    WebSearch20250305,
}

/// Anthropic tool definition
#[derive(Debug, Clone)]
pub struct AnthropicTool {
    pub tool_type: ToolType,
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<serde_json::Value>,
    pub max_uses: Option<u32>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
}

impl AnthropicTool {
    /// Create a new text editor tool for Claude 4
    pub fn text_editor_claude4() -> Self {
        Self {
            tool_type: ToolType::TextEditor20250429,
            name: "str_replace_based_edit_tool".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a new text editor tool for Claude 3.7
    pub fn text_editor_claude37() -> Self {
        Self {
            tool_type: ToolType::TextEditor20250124,
            name: "str_replace_editor".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a new text editor tool for Claude 3.5
    pub fn text_editor_claude35() -> Self {
        Self {
            tool_type: ToolType::TextEditor20241022,
            name: "str_replace_editor".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a new code execution tool
    pub fn code_execution() -> Self {
        Self {
            tool_type: ToolType::CodeExecution20250522,
            name: "code_execution".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a new web search tool
    pub fn web_search() -> Self {
        Self {
            tool_type: ToolType::WebSearch20250305,
            name: "web_search".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None, // Schema is built into the model
            max_uses: Some(5), // Default limit
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a web search tool with domain filtering
    pub fn web_search_with_domains(
        max_uses: Option<u32>,
        allowed_domains: Option<Vec<String>>,
        blocked_domains: Option<Vec<String>>,
    ) -> Self {
        Self {
            tool_type: ToolType::WebSearch20250305,
            name: "web_search".to_string(),
            description: None, // Built-in tools don't accept descriptions
            input_schema: None,
            max_uses,
            allowed_domains,
            blocked_domains,
        }
    }

    /// Create a custom tool
    pub fn custom<S: Into<String>>(
        name: S,
        description: S,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: ToolType::Custom,
            name: name.into(),
            description: Some(description.into()),
            input_schema: Some(input_schema),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a memory search tool
    pub fn memory_search() -> Self {
        Self::custom(
            "memory_search",
            "Search through the agent's memory for relevant information",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant memories"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
        )
    }

    /// Create a memory save tool
    pub fn memory_save() -> Self {
        Self::custom(
            "memory_save",
            "Save important information to the agent's memory",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save to memory"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to categorize the memory"
                    }
                },
                "required": ["content"]
            }),
        )
    }

    /// Create a file read tool
    pub fn file_read() -> Self {
        Self::custom(
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

    /// Create a file write tool
    pub fn file_write() -> Self {
        Self::custom(
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

    /// Convert to API tool definition
    pub fn to_definition(&self) -> ToolDefinition {
        let tool_type = match self.tool_type {
            ToolType::Custom => "function".to_string(),
            ToolType::TextEditor20250429 => "text_editor_20250429".to_string(),
            ToolType::TextEditor20250124 => "text_editor_20250124".to_string(),
            ToolType::TextEditor20241022 => "text_editor_20241022".to_string(),
            ToolType::CodeExecution20250522 => "code_execution_20250522".to_string(),
            ToolType::WebSearch20250305 => "web_search_20250305".to_string(),
        };

        ToolDefinition {
            tool_type,
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            max_uses: self.max_uses,
            allowed_domains: self.allowed_domains.clone(),
            blocked_domains: self.blocked_domains.clone(),
        }
    }

    /// Check if this is a server-side tool
    pub fn is_server_tool(&self) -> bool {
        matches!(
            self.tool_type,
            ToolType::CodeExecution20250522 | ToolType::WebSearch20250305
        )
    }

    /// Check if this is a client-side tool
    pub fn is_client_tool(&self) -> bool {
        !self.is_server_tool()
    }

    /// Get the appropriate text editor tool for a model
    pub fn text_editor_for_model(model: &str) -> Self {
        if model.contains("claude-opus-4") || model.contains("claude-sonnet-4") {
            Self::text_editor_claude4()
        } else if model.contains("claude-3-7-sonnet") {
            Self::text_editor_claude37()
        } else {
            Self::text_editor_claude35()
        }
    }
}
