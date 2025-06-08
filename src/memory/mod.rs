pub mod simple_memory;
pub mod synaptic;
pub mod search;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::anthropic::models::ChatMessage;
use crate::config::MemoryConfig;
use crate::utils::error::{AgentError, Result};

pub use simple_memory::SimpleMemory;
pub use search::SearchResult;

/// Memory manager for the agent system
#[derive(Debug)]
pub struct MemoryManager {
    /// Simple JSON-based memory storage
    memory: SimpleMemory,
    /// Configuration
    config: MemoryConfig,
    /// Current conversation ID
    current_conversation_id: Option<String>,
    /// Current conversation
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
}

/// A memory entry stored in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub entry_type: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        let memory = SimpleMemory::new(&config.memory_path, &config.index_path).await?;

        Ok(Self {
            memory,
            config,
            current_conversation_id: None,
            current_conversation: None,
        })
    }

    /// Start a new conversation
    pub fn start_conversation(&mut self, title: Option<String>) -> String {
        let conversation_id = Uuid::new_v4().to_string();
        let conversation = Conversation {
            id: conversation_id.clone(),
            title,
            messages: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        self.current_conversation_id = Some(conversation_id.clone());
        self.current_conversation = Some(conversation);

        conversation_id
    }

    /// Add a message to the current conversation
    pub async fn add_message(&mut self, message: ChatMessage) -> Result<()> {
        let conversation_to_save = if let Some(ref mut conversation) = self.current_conversation {
            conversation.messages.push(message);
            conversation.updated_at = chrono::Utc::now();
            conversation.clone()
        } else {
            return Err(AgentError::memory("No active conversation".to_string()));
        };

        // Save the conversation to memory
        self.save_conversation(conversation_to_save).await?;
        Ok(())
    }

    /// Get a conversation by ID
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>> {
        // Search for the conversation in memory using a more specific search
        let search_results = self.memory.search(&format!("conversation_id:{}", conversation_id), 10).await?;

        // Look through results to find the exact conversation
        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.content) {
                if conversation.id == conversation_id {
                    return Ok(Some(conversation));
                }
            }
        }

        Ok(None)
    }

    /// Save a conversation to memory
    async fn save_conversation(&mut self, conversation: Conversation) -> Result<()> {
        // Serialize the conversation to JSON
        let content = serde_json::to_string(&conversation)
            .map_err(|e| AgentError::memory(format!("Failed to serialize conversation: {}", e)))?;

        // Store the full conversation JSON
        self.memory.add_chunk(content).await?;

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

        self.memory.add_chunk(searchable_content).await?;
        Ok(())
    }

    /// Search conversations
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<Conversation>> {
        let search_results = self.memory.search(query, limit).await?;
        let mut conversations = Vec::new();

        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.content) {
                conversations.push(conversation);
            }
        }

        Ok(conversations)
    }

    /// Save a memory entry
    pub async fn save_memory(&mut self, content: String, entry_type: String, metadata: std::collections::HashMap<String, String>) -> Result<String> {
        let memory_id = Uuid::new_v4().to_string();
        let entry = MemoryEntry {
            id: memory_id.clone(),
            content: content.clone(),
            entry_type: entry_type.clone(),
            metadata,
            created_at: chrono::Utc::now(),
        };

        // Serialize the memory entry to JSON
        let content = serde_json::to_string(&entry)
            .map_err(|e| AgentError::memory(format!("Failed to serialize memory entry: {}", e)))?;

        // Store the full memory entry JSON
        self.memory.add_chunk(content).await?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "memory_id:{} type:{} content:{}",
            entry.id,
            entry.entry_type,
            entry.content
        );

        self.memory.add_chunk(searchable_content).await?;
        Ok(memory_id)
    }

    /// Search memory entries
    pub async fn search_memory(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let search_results = self.memory.search(query, limit).await?;
        let mut entries = Vec::new();

        for result in search_results {
            if let Ok(entry) = serde_json::from_str::<MemoryEntry>(&result.content) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Search memory and return raw search results
    pub async fn search_raw(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.memory.search(query, limit).await
    }

    /// Get current conversation
    pub fn get_current_conversation(&self) -> Option<&Conversation> {
        self.current_conversation.as_ref()
    }

    /// Get current conversation ID
    pub fn get_current_conversation_id(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let stats = self.memory.get_stats().await?;
        Ok(MemoryStats {
            total_chunks: stats.total_chunks,
            total_conversations: self.count_conversations().await?,
            total_memories: self.count_memories().await?,
            memory_file_size: stats.file_size_bytes,
            index_file_size: 0, // No separate index file for JSON storage
        })
    }

    /// Count conversations in memory
    async fn count_conversations(&self) -> Result<usize> {
        let results = self.memory.search("conversation_id:", 1000).await?;
        Ok(results.len())
    }

    /// Count memory entries
    async fn count_memories(&self) -> Result<usize> {
        let results = self.memory.search("memory_id:", 1000).await?;
        Ok(results.len())
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_chunks: usize,
    pub total_conversations: usize,
    pub total_memories: usize,
    pub memory_file_size: u64,
    pub index_file_size: u64,
}
