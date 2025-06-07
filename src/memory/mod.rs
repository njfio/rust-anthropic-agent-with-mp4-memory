pub mod memvid_wrapper;
pub mod search;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;
use tracing::warn;

use crate::anthropic::models::ChatMessage;
use crate::config::MemoryConfig;
use crate::utils::error::{AgentError, Result};

pub use memvid_wrapper::MemvidWrapper;
pub use search::SearchResult;

/// Memory manager for the agent system
#[derive(Debug)]
pub struct MemoryManager {
    /// Wrapper for the rust-mp4-memory library
    memvid: MemvidWrapper,
    /// Configuration
    config: MemoryConfig,
    /// Current conversation ID
    current_conversation_id: Option<String>,
    /// In-memory conversation cache for quick access
    current_conversation: Option<Conversation>,
}

/// A conversation stored in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub tags: Vec<String>,
}

/// Memory entry for storing arbitrary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub entry_type: MemoryEntryType,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Types of memory entries
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEntryType {
    Conversation,
    Note,
    Code,
    Document,
    Fact,
    Custom(String),
}

impl MemoryManager {
    /// Create a new memory manager
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        let mut memvid = MemvidWrapper::new(&config.memory_path, &config.index_path).await?;

        // Initialize Phase 2 performance features
        if let Err(e) = memvid.initialize_phase2_features().await {
            warn!("Failed to initialize Phase 2 features: {}", e);
        }

        Ok(Self {
            memvid,
            config,
            current_conversation_id: None,
            current_conversation: None,
        })
    }

    /// Start a new conversation
    pub async fn start_conversation(&mut self, title: Option<String>) -> Result<String> {
        let conversation_id = Uuid::new_v4().to_string();
        let conversation = Conversation {
            id: conversation_id.clone(),
            title,
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            tags: Vec::new(),
        };

        // Save the initial empty conversation
        self.save_conversation(&conversation).await?;
        self.current_conversation_id = Some(conversation_id.clone());
        self.current_conversation = Some(conversation);

        Ok(conversation_id)
    }

    /// Add a message to the current conversation
    pub async fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        if let Some(ref mut conversation) = self.current_conversation {
            conversation.messages.push(message);
            conversation.updated_at = chrono::Utc::now();

            // Save the updated conversation (clone to avoid borrowing issues)
            let conversation_to_save = conversation.clone();
            self.save_conversation(&conversation_to_save).await?;
        } else {
            return Err(AgentError::memory("No active conversation"));
        }

        Ok(())
    }

    /// Get a conversation by ID
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>> {
        // Search for the conversation in memory using a more specific search
        let search_results = self.memvid.search(&format!("conversation_id:{}", conversation_id), 10).await?;

        // Look through results to find the exact conversation
        for result in search_results {
            // Try to parse as conversation JSON
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.content) {
                if conversation.id == conversation_id {
                    return Ok(Some(conversation));
                }
            }

            // Also check if the content contains the conversation data in our searchable format
            if result.content.contains(&format!("conversation_id:{}", conversation_id)) {
                // This is a searchable format, we need to reconstruct or find the actual conversation
                // For now, let's create a minimal conversation if we can't find the full data
                let conversation = Conversation {
                    id: conversation_id.to_string(),
                    title: None,
                    messages: Vec::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    tags: Vec::new(),
                };
                return Ok(Some(conversation));
            }
        }

        Ok(None)
    }

    /// Save a conversation to memory
    pub async fn save_conversation(&mut self, conversation: &Conversation) -> Result<()> {
        let content = serde_json::to_string(conversation)
            .map_err(|e| AgentError::memory(format!("Failed to serialize conversation: {}", e)))?;

        // Store the full conversation JSON
        self.memvid.add_chunk(content).await?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "conversation_id:{} title:{} messages:{}",
            conversation.id,
            conversation.title.as_deref().unwrap_or("untitled"),
            conversation.messages.iter()
                .map(|m| m.get_text())
                .collect::<Vec<_>>()
                .join(" ")
        );

        self.memvid.add_chunk(searchable_content).await?;

        // Build the video to persist the chunks
        self.memvid.build_video().await?;

        Ok(())
    }

    /// Search conversations
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<Conversation>> {
        let search_results = self.memvid.search(query, limit).await?;
        let mut conversations = Vec::new();

        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.content) {
                conversations.push(conversation);
            }
        }

        Ok(conversations)
    }

    /// Save a memory entry
    pub async fn save_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        let content = serde_json::to_string(&entry)
            .map_err(|e| AgentError::memory(format!("Failed to serialize memory entry: {}", e)))?;

        // Store the full memory entry JSON
        self.memvid.add_chunk(content).await?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "memory_id:{} type:{:?} tags:{} content:{}",
            entry.id,
            entry.entry_type,
            entry.tags.join(" "),
            entry.content
        );

        self.memvid.add_chunk(searchable_content).await?;

        // Build the video to persist the chunks
        self.memvid.build_video().await?;

        Ok(())
    }

    /// Search memory entries
    pub async fn search_memory(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let search_results = self.memvid.search(query, limit).await?;
        let mut entries = Vec::new();

        for result in search_results {
            // Try to parse as JSON first (full memory entry)
            if let Ok(entry) = serde_json::from_str::<MemoryEntry>(&result.content) {
                entries.push(entry);
            }
            // If that fails, it might be searchable text - skip it since we want the full entries
        }

        Ok(entries)
    }

    /// Search memory and return raw search results
    pub async fn search_raw(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.memvid.search(query, limit).await
    }

    /// Get conversation history for context
    pub async fn get_conversation_context(&self, limit: usize) -> Result<Vec<ChatMessage>> {
        if let Some(ref conversation) = self.current_conversation {
            let start_idx = if conversation.messages.len() > limit {
                conversation.messages.len() - limit
            } else {
                0
            };
            return Ok(conversation.messages[start_idx..].to_vec());
        }
        Ok(Vec::new())
    }

    /// Build the memory video (finalize and optimize)
    pub async fn build_memory(&mut self) -> Result<()> {
        self.memvid.build_video().await?;
        Ok(())
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let stats = self.memvid.get_stats().await?;
        Ok(MemoryStats {
            total_chunks: stats.total_chunks,
            total_conversations: self.count_conversations().await?,
            total_memories: self.count_memories().await?,
            memory_file_size: stats.video_size_bytes,
            index_file_size: stats.index_size_bytes,
        })
    }

    /// Count conversations in memory
    async fn count_conversations(&self) -> Result<usize> {
        let results = self.memvid.search("conversation_id:", 1000).await?;
        Ok(results.len())
    }

    /// Count memory entries
    async fn count_memories(&self) -> Result<usize> {
        let results = self.memvid.search("memory_id:", 1000).await?;
        Ok(results.len())
    }

    /// Get current conversation ID
    pub fn current_conversation_id(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Set current conversation ID
    pub fn set_current_conversation_id(&mut self, conversation_id: Option<String>) {
        // Check if we need to clear the cache before setting the new ID
        let should_clear = if let Some(ref current_conv) = self.current_conversation {
            if let Some(ref new_id) = conversation_id {
                current_conv.id != *new_id
            } else {
                true
            }
        } else {
            false
        };

        self.current_conversation_id = conversation_id;

        if should_clear {
            self.current_conversation = None;
        }
    }

    // ========================================
    // Phase 2 Performance Enhancement Methods
    // ========================================

    /// Perform multi-memory search across different memory instances
    pub async fn multi_memory_search(&mut self, query: &str, limit: usize) -> Result<rust_mem_vid::MultiMemorySearchResult> {
        self.memvid.multi_memory_search(query, limit).await
    }

    /// Generate temporal analysis of memory evolution
    pub async fn temporal_analysis(&self, days_back: u32) -> Result<serde_json::Value> {
        self.memvid.temporal_analysis(days_back).await
    }

    /// Build knowledge graph from memory content
    pub async fn build_knowledge_graph(&self) -> Result<serde_json::Value> {
        self.memvid.build_knowledge_graph().await
    }

    /// Synthesize content using AI
    pub async fn synthesize_content(&self, synthesis_type: &str, query: Option<&str>) -> Result<serde_json::Value> {
        self.memvid.synthesize_content(synthesis_type, query).await
    }

    /// Generate analytics dashboard data
    pub async fn generate_analytics_dashboard(&self) -> Result<serde_json::Value> {
        self.memvid.generate_analytics_dashboard().await
    }

    /// Check if Phase 2 features are available
    pub fn has_phase2_features(&self) -> bool {
        self.memvid.has_phase2_features()
    }

    /// Get Phase 2 features status
    pub fn get_phase2_status(&self) -> std::collections::HashMap<String, bool> {
        self.memvid.get_phase2_status()
    }

    /// Initialize Phase 2 features if not already done
    pub async fn ensure_phase2_features(&mut self) -> Result<()> {
        if !self.has_phase2_features() {
            self.memvid.initialize_phase2_features().await?;
        }
        Ok(())
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_chunks: usize,
    pub total_conversations: usize,
    pub total_memories: usize,
    pub memory_file_size: u64,
    pub index_file_size: u64,
}

impl Conversation {
    /// Create a new conversation
    pub fn new(title: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title,
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            tags: Vec::new(),
        }
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        self.updated_at = chrono::Utc::now();
    }

    /// Get the conversation summary
    pub fn get_summary(&self) -> String {
        if self.messages.is_empty() {
            return "Empty conversation".to_string();
        }

        let first_message = &self.messages[0];
        let preview = first_message.get_text();
        let preview = if preview.len() > 100 {
            format!("{}...", &preview[..100])
        } else {
            preview
        };

        format!(
            "{} ({} messages) - {}",
            self.title.as_deref().unwrap_or("Untitled"),
            self.messages.len(),
            preview
        )
    }
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new<S: Into<String>>(content: S, entry_type: MemoryEntryType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content: content.into(),
            entry_type,
            tags: Vec::new(),
            created_at: chrono::Utc::now(),
            metadata: None,
        }
    }

    /// Add tags to the memory entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add metadata to the memory entry
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}
