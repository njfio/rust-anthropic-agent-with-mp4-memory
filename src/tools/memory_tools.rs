use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::anthropic::models::ToolDefinition;
use crate::memory::MemoryManager;
use crate::tools::{
    create_tool_definition, extract_optional_int_param, extract_optional_string_param,
    extract_string_param, Tool, ToolResult,
};
use crate::utils::error::{AgentError, Result};

/// Tool for searching through the agent's memory
#[derive(Debug, Clone)]
pub struct MemorySearchTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl MemorySearchTool {
    /// Create a new memory search tool
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
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
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let query = extract_string_param(&input, "query")?;
        let limit = extract_optional_int_param(&input, "limit").unwrap_or(5) as usize;

        debug!("Searching memory for: '{}' (limit: {})", query, limit);

        let memory_manager = self.memory_manager.lock().await;
        let search_results = memory_manager.search_memory(&query, limit).await?;

        if search_results.is_empty() {
            return Ok(ToolResult::success("No relevant memories found."));
        }

        let mut result_text = format!("Found {} relevant memories:\n\n", search_results.len());
        
        for (i, entry) in search_results.iter().enumerate() {
            result_text.push_str(&format!(
                "{}. [{}] {}\n   Tags: {}\n   Created: {}\n\n",
                i + 1,
                &entry.entry_type,
                if entry.content.len() > 200 {
                    format!("{}...", &entry.content[..200])
                } else {
                    entry.content.clone()
                },
                "no tags", // Simple memory doesn't have tags
                entry.created_at.format("%Y-%m-%d %H:%M:%S")
            ));
        }

        info!("Memory search completed: {} results", search_results.len());
        Ok(ToolResult::success(result_text))
    }

    fn name(&self) -> &str {
        "memory_search"
    }

    fn description(&self) -> Option<&str> {
        Some("Search through the agent's memory for relevant information")
    }
}

/// Tool for saving information to the agent's memory
#[derive(Debug, Clone)]
pub struct MemorySaveTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl MemorySaveTool {
    /// Create a new memory save tool
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for MemorySaveTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "memory_save",
            "Save important information to the agent's memory",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save to memory"
                    },
                    "entry_type": {
                        "type": "string",
                        "description": "Type of memory entry",
                        "enum": ["note", "code", "document", "fact", "conversation"],
                        "default": "note"
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

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let content = extract_string_param(&input, "content")?;
        let entry_type_str = extract_optional_string_param(&input, "entry_type")
            .unwrap_or_else(|| "note".to_string());
        
        let entry_type = entry_type_str;

        let _tags: Vec<String> = input
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        debug!("Saving memory entry: {} chars, type: {:?}", content.len(), entry_type);

        let mut memory_manager = self.memory_manager.lock().await;
        let metadata = std::collections::HashMap::new();
        let memory_id = memory_manager.save_memory(content, entry_type, metadata).await?;

        info!("Saved memory entry with ID: {}", memory_id);
        Ok(ToolResult::success(format!(
            "Successfully saved memory entry with ID: {}",
            memory_id
        )))
    }

    fn name(&self) -> &str {
        "memory_save"
    }

    fn description(&self) -> Option<&str> {
        Some("Save important information to the agent's memory")
    }
}

/// Tool for getting memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatsTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl MemoryStatsTool {
    /// Create a new memory stats tool
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for MemoryStatsTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "memory_stats",
            "Get statistics about the agent's memory system",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        )
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolResult> {
        debug!("Getting memory statistics");

        let memory_manager = self.memory_manager.lock().await;
        let stats = memory_manager.get_stats().await?;

        let stats_text = format!(
            "Memory System Statistics:\n\
            • Total chunks: {}\n\
            • Total conversations: {}\n\
            • Total memories: {}\n\
            • Memory file size: {:.2} MB\n\
            • Index file size: {:.2} KB",
            stats.total_chunks,
            stats.total_conversations,
            stats.total_memories,
            stats.memory_file_size as f64 / 1024.0 / 1024.0,
            stats.index_file_size as f64 / 1024.0
        );

        Ok(ToolResult::success(stats_text))
    }

    fn name(&self) -> &str {
        "memory_stats"
    }

    fn description(&self) -> Option<&str> {
        Some("Get statistics about the agent's memory system")
    }
}

/// Tool for searching conversations
#[derive(Debug, Clone)]
pub struct ConversationSearchTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl ConversationSearchTool {
    /// Create a new conversation search tool
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for ConversationSearchTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "conversation_search",
            "Search through past conversations",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant conversations"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of conversations to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let query = extract_string_param(&input, "query")?;
        let limit = extract_optional_int_param(&input, "limit").unwrap_or(5) as usize;

        debug!("Searching conversations for: '{}' (limit: {})", query, limit);

        let memory_manager = self.memory_manager.lock().await;
        let conversations = memory_manager.search_conversations(&query, limit).await?;

        if conversations.is_empty() {
            return Ok(ToolResult::success("No relevant conversations found."));
        }

        let mut result_text = format!("Found {} relevant conversations:\n\n", conversations.len());
        
        for (i, conversation) in conversations.iter().enumerate() {
            result_text.push_str(&format!(
                "{}. {}\n   Created: {}\n   Messages: {}\n\n",
                i + 1,
                format!("{} messages", conversation.messages.len()),
                conversation.created_at.format("%Y-%m-%d %H:%M:%S"),
                conversation.messages.len()
            ));
        }

        info!("Conversation search completed: {} results", conversations.len());
        Ok(ToolResult::success(result_text))
    }

    fn name(&self) -> &str {
        "conversation_search"
    }

    fn description(&self) -> Option<&str> {
        Some("Search through past conversations")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_memory_search_tool() {
        let temp_dir = tempdir().unwrap();
        let config = MemoryConfig {
            memory_path: temp_dir.path().join("test.mp4"),
            index_path: temp_dir.path().join("test.json"),
            ..Default::default()
        };

        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(config).await.unwrap()));
        let tool = MemorySearchTool::new(memory_manager);

        let input = json!({
            "query": "test query",
            "limit": 5
        });

        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_memory_save_tool() {
        let temp_dir = tempdir().unwrap();
        let config = MemoryConfig {
            memory_path: temp_dir.path().join("test.mp4"),
            index_path: temp_dir.path().join("test.json"),
            ..Default::default()
        };

        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(config).await.unwrap()));
        let tool = MemorySaveTool::new(memory_manager);

        let input = json!({
            "content": "This is a test memory",
            "entry_type": "note",
            "tags": ["test", "example"]
        });

        let result = tool.execute(input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Successfully saved"));
    }
}
