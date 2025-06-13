// Using rust-synaptic directly as the memory system
pub mod compression;
pub mod search;
pub mod search_algorithms;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::anthropic::models::ChatMessage;
use crate::config::MemoryConfig;
use crate::utils::error::{AgentError, Result};

pub use search::SearchResult;

// Re-export synaptic types for convenience
pub use synaptic::{
    AgentMemory, MemoryConfig as SynapticConfig, MemoryStats as SynapticStats, StorageBackend,
};

/// Memory manager for the agent system using rust-synaptic
pub struct MemoryManager {
    /// Synaptic AI memory system
    memory: Arc<Mutex<AgentMemory>>,
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
    /// Create a new memory manager with synaptic
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        // Convert our config to synaptic config
        let synaptic_config = SynapticConfig {
            storage_backend: StorageBackend::File {
                path: config.memory_path.to_string_lossy().to_string(),
            },
            session_id: Some(Uuid::new_v4()),
            checkpoint_interval: 100,
            max_short_term_memories: 10000,
            max_long_term_memories: 100000,
            similarity_threshold: 0.7,
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            #[cfg(feature = "embeddings")]
            enable_embeddings: false, // Disable embeddings to avoid candle-core conflicts
            #[cfg(feature = "distributed")]
            enable_distributed: false, // Start with distributed disabled for simplicity
            #[cfg(feature = "distributed")]
            distributed_config: None,
            enable_integrations: false, // Disable integrations to avoid external dependencies
            integrations_config: None,
            enable_security: false, // Start with security disabled for simplicity
            security_config: None,
            #[cfg(feature = "multimodal")]
            enable_multimodal: false,
            #[cfg(feature = "multimodal")]
            multimodal_config: None,
            #[cfg(feature = "cross-platform")]
            enable_cross_platform: false,
            #[cfg(feature = "cross-platform")]
            cross_platform_config: None,
        };

        let memory = AgentMemory::new(synaptic_config)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to create synaptic memory: {}", e)))?;

        Ok(Self {
            memory: Arc::new(Mutex::new(memory)),
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

        // Save the conversation to synaptic memory
        self.save_conversation(conversation_to_save).await?;
        Ok(())
    }

    /// Get a conversation by ID
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>> {
        // Search for the conversation in synaptic memory
        let memory = self.memory.lock().await;
        let search_results = memory
            .search(&format!("conversation_id:{}", conversation_id), 10)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to search conversations: {}", e)))?;

        // Look through results to find the exact conversation
        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.entry.value) {
                if conversation.id == conversation_id {
                    return Ok(Some(conversation));
                }
            }
        }

        Ok(None)
    }

    /// Save a conversation to synaptic memory
    async fn save_conversation(&mut self, conversation: Conversation) -> Result<()> {
        // Serialize the conversation to JSON
        let content = serde_json::to_string(&conversation)
            .map_err(|e| AgentError::memory(format!("Failed to serialize conversation: {}", e)))?;

        // Store the full conversation JSON in synaptic memory
        let mut memory = self.memory.lock().await;
        memory
            .store(&format!("conversation:{}", conversation.id), &content)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to store conversation: {}", e)))?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "conversation_id:{} title:{} messages:{}",
            conversation.id,
            conversation.title.as_deref().unwrap_or("untitled"),
            conversation
                .messages
                .iter()
                .map(|m| m.get_text())
                .collect::<Vec<_>>()
                .join(" ")
        );

        memory
            .store(
                &format!("conversation_search:{}", conversation.id),
                &searchable_content,
            )
            .await
            .map_err(|e| {
                AgentError::memory(format!("Failed to store conversation search data: {}", e))
            })?;

        Ok(())
    }

    /// Search conversations
    pub async fn search_conversations(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Conversation>> {
        let memory = self.memory.lock().await;
        let search_results = memory
            .search(query, limit)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to search conversations: {}", e)))?;
        let mut conversations = Vec::new();

        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.entry.value) {
                conversations.push(conversation);
            }
        }

        Ok(conversations)
    }

    /// Save a memory entry
    pub async fn save_memory(
        &mut self,
        content: String,
        entry_type: String,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<String> {
        let memory_id = Uuid::new_v4().to_string();
        let entry = MemoryEntry {
            id: memory_id.clone(),
            content: content.clone(),
            entry_type: entry_type.clone(),
            metadata,
            created_at: chrono::Utc::now(),
        };

        // Serialize the memory entry to JSON
        let serialized_content = serde_json::to_string(&entry)
            .map_err(|e| AgentError::memory(format!("Failed to serialize memory entry: {}", e)))?;

        // Store the full memory entry JSON in synaptic memory
        let mut memory = self.memory.lock().await;
        memory
            .store(&format!("memory:{}", memory_id), &serialized_content)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to store memory entry: {}", e)))?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "memory_id:{} type:{} content:{}",
            entry.id, entry.entry_type, entry.content
        );

        memory
            .store(&format!("memory_search:{}", memory_id), &searchable_content)
            .await
            .map_err(|e| {
                AgentError::memory(format!("Failed to store memory search data: {}", e))
            })?;

        Ok(memory_id)
    }

    /// Search memory entries
    pub async fn search_memory(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let memory = self.memory.lock().await;
        let search_results = memory
            .search(query, limit)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to search memory entries: {}", e)))?;
        let mut entries = Vec::new();

        for result in search_results {
            if let Ok(entry) = serde_json::from_str::<MemoryEntry>(&result.entry.value) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Search memory and return raw search results
    pub async fn search_raw(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let memory = self.memory.lock().await;
        let synaptic_results = memory
            .search(query, limit)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to search memory: {}", e)))?;

        // Convert synaptic results to our SearchResult format
        let results = synaptic_results
            .into_iter()
            .enumerate()
            .map(|(i, result)| SearchResult {
                content: result.entry.value,
                score: result.relevance_score as f32,
                chunk_id: i,
                metadata: None,
            })
            .collect();

        Ok(results)
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
        let memory = self.memory.lock().await;
        let synaptic_stats = memory.stats();

        Ok(MemoryStats {
            total_chunks: synaptic_stats.short_term_count + synaptic_stats.long_term_count,
            total_conversations: self.count_conversations().await?,
            total_memories: self.count_memories().await?,
            memory_file_size: synaptic_stats.total_size as u64,
            index_file_size: 0, // Synaptic handles indexing internally
        })
    }

    /// Count conversations in memory
    async fn count_conversations(&self) -> Result<usize> {
        let memory = self.memory.lock().await;
        let results = memory
            .search("conversation_id:", 1000)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to count conversations: {}", e)))?;
        Ok(results.len())
    }

    /// Count memory entries
    async fn count_memories(&self) -> Result<usize> {
        let memory = self.memory.lock().await;
        let results = memory
            .search("memory_id:", 1000)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to count memories: {}", e)))?;
        Ok(results.len())
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("config", &self.config)
            .field("current_conversation_id", &self.current_conversation_id)
            .field("current_conversation", &self.current_conversation)
            .field("memory", &"<AgentMemory>")
            .finish()
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
