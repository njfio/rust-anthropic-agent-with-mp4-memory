use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::anthropic::models::ToolDefinition;
use crate::memory::MemoryManager;
use crate::tools::{
    create_tool_definition, extract_optional_bool_param, extract_optional_int_param,
    extract_optional_string_param, extract_string_param, Tool, ToolResult,
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

        // Handle the case where counts might be 0 (fast stats mode)
        let conversations_display = if stats.total_conversations == 0 {
            "~".to_string() // Indicate approximate/unknown
        } else {
            stats.total_conversations.to_string()
        };

        let memories_display = if stats.total_memories == 0 {
            "~".to_string() // Indicate approximate/unknown
        } else {
            stats.total_memories.to_string()
        };

        let mode_info = if stats.total_conversations == 0 && stats.total_memories == 0 {
            "\n             ℹ️  Fast mode: ~ indicates approximate values. Use 'detailed: true' for exact counts."
        } else {
            "\n             ✅ Detailed mode: Exact counts provided."
        };

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
             ╰─────────────────────────────────────────────────────────────────╯{}\n\n\
             💡 Memory system is actively storing and indexing your interactions",
            stats.total_chunks,
            conversations_display,
            memories_display,
            memory_size_mb,
            index_size_kb,
            if stats.total_chunks > 0 { "🟢 Active" } else { "🟡 Empty" },
            if memory_size_mb < 100.0 { "🟢 Optimal" } else if memory_size_mb < 500.0 { "🟡 Good" } else { "🟠 Large" },
            mode_info
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
                "properties": {
                    "detailed": {
                        "type": "boolean",
                        "description": "Whether to include detailed counts (slower but more accurate)",
                        "default": false
                    }
                },
                "required": []
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        debug!("Getting memory statistics");

        // Check if detailed stats are requested
        let detailed = extract_optional_bool_param(&input, "detailed").unwrap_or(false);

        let memory_manager = self.memory_manager.lock().await;

        // Use fast stats by default to avoid timeouts
        let stats = if detailed {
            debug!("Getting detailed memory statistics (may be slower)");
            memory_manager.get_stats().await?
        } else {
            debug!("Getting fast memory statistics");
            memory_manager.get_fast_stats().await?
        };

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

/// Tool for semantic memory search using rust-synaptic's embeddings
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone)]
pub struct SemanticMemorySearchTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

#[cfg(feature = "embeddings")]
impl SemanticMemorySearchTool {
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[cfg(feature = "embeddings")]
#[async_trait]
impl Tool for SemanticMemorySearchTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "semantic_memory_search",
            "Search memory using semantic similarity (meaning-based search)",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic query to search for (searches by meaning, not just keywords)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
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
        let limit = extract_optional_int_param(&input, "limit")?.unwrap_or(5) as usize;

        debug!("Performing semantic search for: {} (limit: {})", query, limit);

        let memory_manager = self.memory_manager.lock().await;
        let results = memory_manager.semantic_search(&query, limit).await?;

        if results.is_empty() {
            return Ok(ToolResult::success("No semantically similar memories found."));
        }

        let mut response = format!("🧠 Found {} semantically similar memories:\n\n", results.len());
        for (i, result) in results.iter().enumerate() {
            response.push_str(&format!(
                "{}. **Semantic Score: {:.3}**\n{}\n\n",
                i + 1,
                result.score,
                result.content
            ));
        }

        Ok(ToolResult::success(response))
    }

    fn name(&self) -> &str {
        "semantic_memory_search"
    }

    fn description(&self) -> Option<&str> {
        Some("Search memory using semantic similarity (meaning-based search)")
    }
}

/// Tool for finding related memories using knowledge graph
#[derive(Debug, Clone)]
pub struct RelatedMemoryTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl RelatedMemoryTool {
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for RelatedMemoryTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "find_related_memories",
            "Find memories related to a specific memory using knowledge graph relationships",
            json!({
                "type": "object",
                "properties": {
                    "memory_key": {
                        "type": "string",
                        "description": "The key/ID of the memory to find relationships for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of related memories to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 15
                    }
                },
                "required": ["memory_key"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let memory_key = extract_string_param(&input, "memory_key")?;
        let limit = extract_optional_int_param(&input, "limit").unwrap_or(5) as usize;

        debug!("Finding related memories for: {} (limit: {})", memory_key, limit);

        let memory_manager = self.memory_manager.lock().await;
        let results = memory_manager.find_related_memories(&memory_key, limit).await?;

        if results.is_empty() {
            return Ok(ToolResult::success("No related memories found."));
        }

        let mut response = format!("🔗 Found {} related memories:\n\n", results.len());
        for (i, result) in results.iter().enumerate() {
            response.push_str(&format!(
                "{}. **Relationship Strength: {:.3}**\n{}\n\n",
                i + 1,
                result.score,
                result.content
            ));
        }

        Ok(ToolResult::success(response))
    }

    fn name(&self) -> &str {
        "find_related_memories"
    }

    fn description(&self) -> Option<&str> {
        Some("Find memories related to a specific memory using knowledge graph relationships")
    }
}

/// Tool for getting advanced memory analytics
#[cfg(feature = "analytics")]
#[derive(Debug, Clone)]
pub struct MemoryAnalyticsTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

#[cfg(feature = "analytics")]
impl MemoryAnalyticsTool {
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[cfg(feature = "analytics")]
#[async_trait]
impl Tool for MemoryAnalyticsTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "memory_analytics",
            "Get advanced analytics about memory usage patterns and performance",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        )
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolResult> {
        debug!("Getting advanced memory analytics");

        let memory_manager = self.memory_manager.lock().await;
        let analytics = memory_manager.get_analytics().await?;

        let response = format!(
            "📊 **Advanced Memory Analytics**\n\n\
            🧠 **Knowledge Graph:**\n\
            - Nodes: {}\n\
            - Relationships: {}\n\
            - Graph Density: {:.3}\n\n\
            ⚡ **Performance:**\n\
            - Memory Efficiency: {:.1}%\n\n\
            📈 **Usage Patterns:**\n\
            - Access Patterns: Available\n\
            - Temporal Patterns: Available\n\
            - Search Performance: Available\n\n\
            💡 The knowledge graph shows how memories are interconnected,\n\
            helping the AI understand relationships between different pieces of information.",
            analytics.node_count,
            analytics.relationship_count,
            analytics.graph_density,
            analytics.memory_efficiency * 100.0
        );

        Ok(ToolResult::success(response))
    }

    fn name(&self) -> &str {
        "memory_analytics"
    }

    fn description(&self) -> Option<&str> {
        Some("Get advanced analytics about memory usage patterns and performance")
    }
}

/// Tool for getting cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatsTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl CacheStatsTool {
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for CacheStatsTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "cache_stats",
            "Get cache performance statistics and memory optimization metrics",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        )
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolResult> {
        debug!("Getting cache performance statistics");

        let memory_manager = self.memory_manager.lock().await;
        let cache_stats = memory_manager.get_cache_stats().await?;

        let response = format!(
            "🚀 **Cache Performance Statistics**\n\n\
            📊 **Cache Utilization:**\n\
            - Search Results: {}/{} entries\n\
            - Conversations: {}/{} entries\n\
            - Memory Entries: {}/{} entries\n\
            - Total Cached: {}/{} entries\n\n\
            ⚡ **Performance:**\n\
            - Cache Hit Rate: {:.1}%\n\
            - Memory Efficiency: {:.1}%\n\n\
            💡 The cache improves performance by storing frequently accessed data in memory,\n\
            reducing the need to query the persistent storage layer.",
            cache_stats.search_entries, cache_stats.max_entries,
            cache_stats.conversation_entries, cache_stats.max_entries,
            cache_stats.memory_entries, cache_stats.max_entries,
            cache_stats.total_entries, cache_stats.max_entries,
            cache_stats.hit_rate * 100.0,
            ((cache_stats.max_entries - cache_stats.total_entries) as f64 / cache_stats.max_entries as f64) * 100.0
        );

        Ok(ToolResult::success(response))
    }

    fn name(&self) -> &str {
        "cache_stats"
    }

    fn description(&self) -> Option<&str> {
        Some("Get cache performance statistics and memory optimization metrics")
    }
}

/// Tool for batch memory operations
#[derive(Debug, Clone)]
pub struct BatchMemoryTool {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl BatchMemoryTool {
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for BatchMemoryTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "batch_memory_operations",
            "Perform batch memory operations for improved performance",
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["batch_search", "batch_store"],
                        "description": "Type of batch operation to perform"
                    },
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer", "default": 5}
                            },
                            "required": ["query"]
                        },
                        "description": "Array of search queries for batch_search operation"
                    },
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "entry_type": {"type": "string"},
                                "key_prefix": {"type": "string", "default": ""},
                                "metadata": {"type": "object", "default": {}}
                            },
                            "required": ["content", "entry_type"]
                        },
                        "description": "Array of memory entries for batch_store operation"
                    }
                },
                "required": ["operation"]
            }),
        )
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        let operation = extract_string_param(&input, "operation")?;

        match operation.as_str() {
            "batch_search" => {
                let queries_json = input.get("queries")
                    .ok_or_else(|| AgentError::tool("batch_memory_operations", "Missing queries parameter for batch_search"))?;

                let queries: Vec<(String, usize)> = queries_json.as_array()
                    .ok_or_else(|| AgentError::tool("batch_memory_operations", "Queries must be an array"))?
                    .iter()
                    .map(|q| {
                        let query = q.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let limit = q.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                        (query, limit)
                    })
                    .collect();

                debug!("Performing batch search for {} queries", queries.len());

                let memory_manager = self.memory_manager.lock().await;
                let results = memory_manager.batch_search(queries).await?;

                let mut response = format!("🔍 **Batch Search Results** ({} queries processed)\n\n", results.len());
                for (i, query_results) in results.iter().enumerate() {
                    response.push_str(&format!("**Query {}:** {} results\n", i + 1, query_results.len()));
                    for (j, result) in query_results.iter().take(3).enumerate() {
                        response.push_str(&format!("  {}. Score: {:.3} - {}\n", j + 1, result.score,
                            result.content.chars().take(100).collect::<String>()));
                    }
                    if query_results.len() > 3 {
                        response.push_str(&format!("  ... and {} more results\n", query_results.len() - 3));
                    }
                    response.push('\n');
                }

                Ok(ToolResult::success(response))
            },
            "batch_store" => {
                let entries_json = input.get("entries")
                    .ok_or_else(|| AgentError::tool("batch_memory_operations", "Missing entries parameter for batch_store"))?;

                let entries: Vec<(String, String, String, std::collections::HashMap<String, String>)> = entries_json.as_array()
                    .ok_or_else(|| AgentError::tool("batch_memory_operations", "Entries must be an array"))?
                    .iter()
                    .map(|e| {
                        let content = e.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let entry_type = e.get("entry_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let key_prefix = e.get("key_prefix").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let metadata = std::collections::HashMap::new(); // Simplified for now
                        (content, entry_type, key_prefix, metadata)
                    })
                    .collect();

                debug!("Performing batch store for {} entries", entries.len());

                let mut memory_manager = self.memory_manager.lock().await;
                let memory_ids = memory_manager.batch_store_memories(entries).await?;

                let response = format!(
                    "💾 **Batch Store Complete**\n\n\
                    Successfully stored {} memory entries.\n\
                    Memory IDs: {}\n\n\
                    ⚡ Batch operations are more efficient than individual stores.",
                    memory_ids.len(),
                    memory_ids.join(", ")
                );

                Ok(ToolResult::success(response))
            },
            _ => Err(AgentError::tool("batch_memory_operations", &format!("Unknown batch operation: {}", operation)))
        }
    }

    fn name(&self) -> &str {
        "batch_memory_operations"
    }

    fn description(&self) -> Option<&str> {
        Some("Perform batch memory operations for improved performance")
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
