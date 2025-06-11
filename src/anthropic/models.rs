use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Role of a message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Content block within a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    CodeExecutionToolResult {
        tool_use_id: String,
        content: CodeExecutionResult,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: Vec<WebSearchResult>,
    },
}

/// Code execution result from server-side tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub stdout: String,
    pub stderr: String,
    pub return_code: i32,
}

/// Web search result from server-side tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub url: String,
    pub title: String,
    pub encrypted_content: String,
    pub page_age: String,
}

/// A message in the conversation (internal representation with metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// A message for API requests (without metadata fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl ApiMessage {
    /// Create a new user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
        }
    }

    /// Create a new assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
        }
    }
}

/// Tool definition for the API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

/// Request to the Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Response from the Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Container>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,
}

/// Server tool usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerToolUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution_time: Option<f64>,
}

/// Container information for code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container {
    pub id: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// Tool use information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_use: ToolUse,
    pub result: ToolResult,
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            id: Some(Uuid::new_v4().to_string()),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Convert to API message (without metadata)
    pub fn to_api_message(&self) -> ApiMessage {
        ApiMessage {
            role: self.role.clone(),
            content: self.content.clone(),
        }
    }

    /// Create a new assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            id: Some(Uuid::new_v4().to_string()),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Create a new system message
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            id: Some(Uuid::new_v4().to_string()),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Add a tool result to the message
    pub fn with_tool_result(mut self, tool_use_id: String, content: String, is_error: bool) -> Self {
        self.content.push(ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error: Some(is_error),
        });
        self
    }

    /// Get the text content of the message
    pub fn get_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if the message has no content blocks
    pub fn has_empty_content(&self) -> bool {
        self.content.is_empty()
    }

    /// Check if the message contains tool uses
    pub fn has_tool_uses(&self) -> bool {
        self.content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolUse { .. } | ContentBlock::ServerToolUse { .. }
            )
        })
    }

    /// Get all tool uses in the message
    pub fn get_tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| {
                matches!(
                    block,
                    ContentBlock::ToolUse { .. } | ContentBlock::ServerToolUse { .. }
                )
            })
            .collect()
    }

    /// Check if the message has no meaningful content
    pub fn has_empty_content(&self) -> bool {

        if self.content.is_empty() {
            return true;
        }

        self.content.iter().all(|block| match block {
            ContentBlock::Text { text } => text.trim().is_empty(),
            ContentBlock::ToolResult { content, .. } => content.trim().is_empty(),

        self.content.iter().all(|block| match block {
            ContentBlock::Text { text } => text.trim().is_empty(),

            _ => false,
        })
    }
}
