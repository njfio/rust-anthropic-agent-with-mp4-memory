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
use crate::utils::error::Result;

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

    /// Format search results with enhanced visual presentation
    fn format_search_results(query: &str, results: &[crate::memory::MemoryEntry]) -> String {
        let mut output = String::new();

        // Header with search summary
        output.push_str("╭─────────────────────────────────────────────────────────────────╮\n");
        output.push_str(&format!("│ 🔍 Memory Search Results                                        │\n"));
        output.push_str("├─────────────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!("│ Query: \"{}\"{}│\n",
            Self::truncate_text(query, 45),
            " ".repeat(45_i32.saturating_sub(query.len() as i32).max(0) as usize + 8)
        ));
        output.push_str(&format!("│ Found: {} result{}{}│\n",
            results.len(),
            if results.len() == 1 { "" } else { "s" },
            " ".repeat(50_i32.saturating_sub(format!("{} result{}", results.len(), if results.len() == 1 { "" } else { "s" }).len() as i32).max(0) as usize)
        ));
        output.push_str("╰─────────────────────────────────────────────────────────────────╯\n\n");

        // Results
        for (i, entry) in results.iter().enumerate() {
            let entry_number = i + 1;
            let content_preview = Self::format_content_preview(&entry.content);
            let entry_type_display = Self::format_entry_type(&entry.entry_type);
            let time_display = Self::format_time_ago(&entry.created_at);

            output.push_str(&format!("┌─ Result {} {}\n", entry_number, "─".repeat(55)));
            output.push_str(&format!("│ 📝 Type: {}\n", entry_type_display));
            output.push_str(&format!("│ 🕒 Created: {}\n", time_display));
            output.push_str(&format!("│ 📊 Size: {} characters\n", entry.content.len()));
            output.push_str("├─ Content ─────────────────────────────────────────────────────\n");

            // Format content with proper line wrapping
            for line in content_preview.lines() {
                if line.trim().is_empty() {
                    output.push_str("│\n");
                } else {
                    let wrapped_lines = Self::wrap_text(line, 60);
                    for wrapped_line in wrapped_lines {
                        output.push_str(&format!("│ {}\n", wrapped_line));
                    }
                }
            }

            if entry.content.len() > 300 {
                output.push_str("│ ⋯ (content truncated)\n");
            }

            output.push_str("└───────────────────────────────────────────────────────────────\n");

            if i < results.len() - 1 {
                output.push_str("\n");
            }
        }

        output.push_str("\n💡 Tip: Use more specific search terms to refine results\n");
        output
    }

    /// Format the no results message
    fn format_no_results(query: &str) -> String {
        format!(
            "╭─────────────────────────────────────────────────────────────────╮\n\
             │ 🔍 Memory Search Results                                        │\n\
             ├─────────────────────────────────────────────────────────────────┤\n\
             │ Query: \"{}\"{}│\n\
             │ Found: 0 results{}│\n\
             ╰─────────────────────────────────────────────────────────────────╯\n\n\
             🤔 No memories found matching your search.\n\n\
             💡 Try:\n\
             • Using different keywords\n\
             • Searching for partial words\n\
             • Using broader terms\n\
             • Checking if the information was saved to memory",
            Self::truncate_text(query, 45),
            " ".repeat(45_i32.saturating_sub(query.len() as i32).max(0) as usize + 8),
            " ".repeat(50)
        )
    }

    /// Format content preview with smart truncation
    fn format_content_preview(content: &str) -> String {
        const MAX_PREVIEW_CHARS: usize = 300;

        if content.len() <= MAX_PREVIEW_CHARS {
            content.to_string()
        } else {
            // Try to break at a sentence or word boundary
            let truncated = &content[..MAX_PREVIEW_CHARS];
            if let Some(last_sentence) = truncated.rfind('.') {
                if last_sentence > MAX_PREVIEW_CHARS / 2 {
                    return format!("{}.", &truncated[..last_sentence]);
                }
            }
            if let Some(last_word) = truncated.rfind(' ') {
                if last_word > MAX_PREVIEW_CHARS / 2 {
                    return format!("{}...", &truncated[..last_word]);
                }
            }
            format!("{}...", truncated)
        }
    }

    /// Format entry type with appropriate emoji
    fn format_entry_type(entry_type: &str) -> String {
        match entry_type.to_lowercase().as_str() {
            "note" => "📝 Note".to_string(),
            "code" => "💻 Code".to_string(),
            "document" => "📄 Document".to_string(),
            "fact" => "💡 Fact".to_string(),
            "conversation" => "💬 Conversation".to_string(),
            _ => format!("📋 {}", entry_type),
        }
    }

    /// Format time in a human-readable "time ago" format
    fn format_time_ago(created_at: &chrono::DateTime<chrono::Utc>) -> String {
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(*created_at);

        if duration.num_days() > 0 {
            format!("{} days ago ({})", duration.num_days(), created_at.format("%Y-%m-%d %H:%M"))
        } else if duration.num_hours() > 0 {
            format!("{} hours ago ({})", duration.num_hours(), created_at.format("%H:%M"))
        } else if duration.num_minutes() > 0 {
            format!("{} minutes ago", duration.num_minutes())
        } else {
            "Just now".to_string()
        }
    }

    /// Truncate text to specified length with ellipsis
    fn truncate_text(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len.saturating_sub(3)])
        }
    }

    /// Wrap text to specified width
    fn wrap_text(text: &str, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + word.len() + 1 <= width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        if lines.is_empty() {
            lines.push(String::new());
        }

        lines
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
            return Ok(ToolResult::success(Self::format_no_results(&query)));
        }

        let result_text = Self::format_search_results(&query, &search_results);

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

    /// Format memory statistics with enhanced visual presentation
    fn format_memory_stats(stats: &crate::memory::MemoryStats) -> String {
        let memory_size_mb = stats.memory_file_size as f64 / 1024.0 / 1024.0;
        let index_size_kb = stats.index_file_size as f64 / 1024.0;

        format!(
            "╭─────────────────────────────────────────────────────────────────╮\n\
             │ 📊 Memory System Statistics                                     │\n\
             ├─────────────────────────────────────────────────────────────────┤\n\
             │                                                                 │\n\
             │ 📝 Content Statistics:                                          │\n\
             │   • Total chunks: {:>8}                                      │\n\
             │   • Total conversations: {:>8}                               │\n\
             │   • Total memories: {:>8}                                    │\n\
             │                                                                 │\n\
             │ 💾 Storage Information:                                         │\n\
             │   • Memory file size: {:>8.2} MB                             │\n\
             │   • Index file size: {:>8.2} KB                              │\n\
             │                                                                 │\n\
             │ 🎯 System Health:                                               │\n\
             │   • Status: {}                                        │\n\
             │   • Performance: {}                                   │\n\
             ╰─────────────────────────────────────────────────────────────────╯\n\n\
             💡 Memory system is actively storing and indexing your interactions",
            stats.total_chunks,
            stats.total_conversations,
            stats.total_memories,
            memory_size_mb,
            index_size_kb,
            if stats.total_chunks > 0 { "🟢 Active" } else { "🟡 Empty" },
            if memory_size_mb < 100.0 { "🟢 Optimal" } else if memory_size_mb < 500.0 { "🟡 Good" } else { "🟠 Large" }
        )
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

        let stats_text = Self::format_memory_stats(&stats);

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

    /// Format conversation search results with enhanced visual presentation
    fn format_conversation_results(query: &str, conversations: &[crate::memory::Conversation]) -> String {
        let mut output = String::new();

        // Header with search summary
        output.push_str("╭─────────────────────────────────────────────────────────────────╮\n");
        output.push_str(&format!("│ 💬 Conversation Search Results                                 │\n"));
        output.push_str("├─────────────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!("│ Query: \"{}\"{}│\n",
            Self::truncate_text(query, 45),
            " ".repeat(45_i32.saturating_sub(query.len() as i32).max(0) as usize + 8)
        ));
        output.push_str(&format!("│ Found: {} conversation{}{}│\n",
            conversations.len(),
            if conversations.len() == 1 { "" } else { "s" },
            " ".repeat(44_i32.saturating_sub(format!("{} conversation{}", conversations.len(), if conversations.len() == 1 { "" } else { "s" }).len() as i32).max(0) as usize)
        ));
        output.push_str("╰─────────────────────────────────────────────────────────────────╯\n\n");

        // Results
        for (i, conversation) in conversations.iter().enumerate() {
            let conversation_number = i + 1;
            let time_display = Self::format_time_ago(&conversation.created_at);
            let message_count = conversation.messages.len();

            output.push_str(&format!("┌─ Conversation {} {}\n", conversation_number, "─".repeat(50)));
            output.push_str(&format!("│ 🕒 Created: {}\n", time_display));
            output.push_str(&format!("│ 💬 Messages: {}\n", message_count));
            output.push_str(&format!("│ 🆔 ID: {}\n", conversation.id));

            if let Some(title) = &conversation.title {
                output.push_str(&format!("│ 📝 Title: {}\n", Self::truncate_text(title, 50)));
            }

            // Show preview of first few messages
            if !conversation.messages.is_empty() {
                output.push_str("├─ Message Preview ────────────────────────────────────────────\n");

                for (msg_idx, message) in conversation.messages.iter().take(2).enumerate() {
                    let (role_emoji, role_name) = match message.role {
                        crate::anthropic::models::MessageRole::User => ("👤", "User"),
                        crate::anthropic::models::MessageRole::Assistant => ("🤖", "Assistant"),
                        crate::anthropic::models::MessageRole::System => ("💭", "System"),
                    };

                    output.push_str(&format!("│ {} {}: ", role_emoji, role_name));

                    // Extract text content from message
                    let content_text = Self::extract_message_text(&message.content);
                    let preview = Self::truncate_text(&content_text, 45);
                    output.push_str(&format!("{}\n", preview));

                    if msg_idx == 0 && conversation.messages.len() > 2 {
                        output.push_str("│   ⋯\n");
                    }
                }

                if conversation.messages.len() > 2 {
                    output.push_str(&format!("│ (and {} more messages)\n", conversation.messages.len() - 2));
                }
            }

            output.push_str("└───────────────────────────────────────────────────────────────\n");

            if i < conversations.len() - 1 {
                output.push_str("\n");
            }
        }

        output.push_str("\n💡 Tip: Use specific keywords from conversations to find them faster\n");
        output
    }

    /// Format the no conversation results message
    fn format_no_conversation_results(query: &str) -> String {
        format!(
            "╭─────────────────────────────────────────────────────────────────╮\n\
             │ 💬 Conversation Search Results                                 │\n\
             ├─────────────────────────────────────────────────────────────────┤\n\
             │ Query: \"{}\"{}│\n\
             │ Found: 0 conversations{}│\n\
             ╰─────────────────────────────────────────────────────────────────╯\n\n\
             🤔 No conversations found matching your search.\n\n\
             💡 Try:\n\
             • Using keywords from past discussions\n\
             • Searching for topics you've talked about\n\
             • Using broader search terms\n\
             • Checking if conversations were saved",
            Self::truncate_text(query, 45),
            " ".repeat(45_i32.saturating_sub(query.len() as i32).max(0) as usize + 8),
            " ".repeat(44)
        )
    }

    /// Extract text content from message content blocks
    fn extract_message_text(content: &[crate::anthropic::models::ContentBlock]) -> String {
        content
            .iter()
            .filter_map(|block| match block {
                crate::anthropic::models::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Format time in a human-readable "time ago" format
    fn format_time_ago(created_at: &chrono::DateTime<chrono::Utc>) -> String {
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(*created_at);

        if duration.num_days() > 0 {
            format!("{} days ago ({})", duration.num_days(), created_at.format("%Y-%m-%d %H:%M"))
        } else if duration.num_hours() > 0 {
            format!("{} hours ago ({})", duration.num_hours(), created_at.format("%H:%M"))
        } else if duration.num_minutes() > 0 {
            format!("{} minutes ago", duration.num_minutes())
        } else {
            "Just now".to_string()
        }
    }

    /// Truncate text to specified length with ellipsis
    fn truncate_text(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len.saturating_sub(3)])
        }
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
            return Ok(ToolResult::success(Self::format_no_conversation_results(&query)));
        }

        let result_text = Self::format_conversation_results(&query, &conversations);

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
            memory_path: temp_dir.path().join("test.json"),
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
            memory_path: temp_dir.path().join("test.json"),
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
