//! # Synaptic Memory Integration
//!
//! This module integrates the state-of-the-art rust-synaptic memory system
//! into the rust_memvid_agent, providing advanced AI memory capabilities
//! including knowledge graphs, semantic search, and temporal intelligence.

use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use synaptic::{AgentMemory, MemoryConfig, MemoryStats, StorageBackend};
use crate::utils::error::{AgentError, Result};

pub mod bridge;
pub mod config;
pub mod tools;

/// Synaptic memory manager that wraps the rust-synaptic AgentMemory
/// and provides integration with the rust_memvid_agent system
#[derive(Debug)]
pub struct SynapticMemoryManager {
    /// The underlying synaptic memory system
    memory: Arc<Mutex<AgentMemory>>,
    /// Configuration for the synaptic memory system
    config: SynapticConfig,
    /// Session ID for this memory instance
    session_id: Uuid,
    /// Creation timestamp
    created_at: DateTime<Utc>,
}

/// Configuration for the Synaptic memory integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticConfig {
    /// Enable knowledge graph functionality
    pub enable_knowledge_graph: bool,
    /// Enable temporal tracking
    pub enable_temporal_tracking: bool,
    /// Enable advanced memory management
    pub enable_advanced_management: bool,
    /// Enable vector embeddings for semantic search
    pub enable_embeddings: bool,
    /// Enable distributed memory features
    pub enable_distributed: bool,
    /// Enable external integrations (PostgreSQL, Redis, etc.)
    pub enable_integrations: bool,
    /// Storage backend configuration
    pub storage_backend: SynapticStorageBackend,
    /// Maximum number of short-term memories
    pub max_short_term_memories: usize,
    /// Maximum number of long-term memories
    pub max_long_term_memories: usize,
    /// Similarity threshold for memory matching
    pub similarity_threshold: f64,
    /// Checkpoint interval for memory persistence
    pub checkpoint_interval: usize,
}

/// Storage backend options for Synaptic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynapticStorageBackend {
    /// In-memory storage (fastest, not persistent)
    Memory,
    /// File-based storage (persistent, good for development)
    File { path: String },
    /// SQL database storage (persistent, production-ready)
    #[cfg(feature = "sql-storage")]
    Sql { connection_string: String },
}

impl Default for SynapticConfig {
    fn default() -> Self {
        Self {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            enable_embeddings: true,
            enable_distributed: false, // Disabled by default for simplicity
            enable_integrations: false, // Disabled by default to avoid external dependencies
            storage_backend: SynapticStorageBackend::File {
                path: "synaptic_memory.db".to_string(),
            },
            max_short_term_memories: 1000,
            max_long_term_memories: 10000,
            similarity_threshold: 0.7,
            checkpoint_interval: 100,
        }
    }
}

impl SynapticMemoryManager {
    /// Create a new Synaptic memory manager
    pub async fn new(config: SynapticConfig) -> Result<Self> {
        let session_id = Uuid::new_v4();
        
        // Convert our config to synaptic's MemoryConfig
        let synaptic_config = MemoryConfig {
            storage_backend: match &config.storage_backend {
                SynapticStorageBackend::Memory => StorageBackend::Memory,
                SynapticStorageBackend::File { path } => StorageBackend::File { 
                    path: path.clone() 
                },
                #[cfg(feature = "sql-storage")]
                SynapticStorageBackend::Sql { connection_string } => StorageBackend::Sql { 
                    connection_string: connection_string.clone() 
                },
            },
            session_id: Some(session_id),
            checkpoint_interval: config.checkpoint_interval,
            max_short_term_memories: config.max_short_term_memories,
            max_long_term_memories: config.max_long_term_memories,
            similarity_threshold: config.similarity_threshold,
            enable_knowledge_graph: config.enable_knowledge_graph,
            enable_temporal_tracking: config.enable_temporal_tracking,
            enable_advanced_management: config.enable_advanced_management,
            #[cfg(feature = "embeddings")]
            enable_embeddings: config.enable_embeddings,
            #[cfg(feature = "distributed")]
            enable_distributed: config.enable_distributed,
            #[cfg(feature = "distributed")]
            distributed_config: None, // TODO: Add distributed config if needed
            enable_integrations: config.enable_integrations,
            integrations_config: None, // TODO: Add integrations config if needed
            ..Default::default()
        };

        // Create the synaptic memory system
        let memory = AgentMemory::new(synaptic_config)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to create synaptic memory: {}", e)))?;

        Ok(Self {
            memory: Arc::new(Mutex::new(memory)),
            config,
            session_id,
            created_at: Utc::now(),
        })
    }

    /// Store a memory entry
    pub async fn store(&self, key: &str, content: &str) -> Result<()> {
        let mut memory = self.memory.lock().await;
        memory.store(key, content)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to store memory: {}", e)))
    }

    /// Retrieve a memory entry by key
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        let mut memory = self.memory.lock().await;
        match memory.retrieve(key).await {
            Ok(Some(entry)) => Ok(Some(entry.content)),
            Ok(None) => Ok(None),
            Err(e) => Err(AgentError::memory(format!("Failed to retrieve memory: {}", e))),
        }
    }

    /// Search memories by content
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemorySearchResult>> {
        let memory = self.memory.lock().await;
        let fragments = memory.search(query, limit)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to search memories: {}", e)))?;

        Ok(fragments.into_iter().map(|fragment| MemorySearchResult {
            key: fragment.key,
            content: fragment.content,
            relevance_score: fragment.relevance_score,
            timestamp: fragment.timestamp,
        }).collect())
    }

    /// Perform semantic search using embeddings (if enabled)
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search(&self, query: &str, limit: Option<usize>) -> Result<Vec<SemanticSearchResult>> {
        let mut memory = self.memory.lock().await;
        let similar_memories = memory.semantic_search(query, limit)
            .map_err(|e| AgentError::memory(format!("Failed to perform semantic search: {}", e)))?;

        Ok(similar_memories.into_iter().map(|sim_mem| SemanticSearchResult {
            key: sim_mem.key,
            content: sim_mem.content,
            similarity_score: sim_mem.similarity_score,
            timestamp: sim_mem.timestamp,
        }).collect())
    }

    /// Find related memories using the knowledge graph
    pub async fn find_related(&self, memory_key: &str, max_depth: usize) -> Result<Vec<RelatedMemoryResult>> {
        let memory = self.memory.lock().await;
        let related = memory.find_related_memories(memory_key, max_depth)
            .await
            .map_err(|e| AgentError::memory(format!("Failed to find related memories: {}", e)))?;

        Ok(related.into_iter().map(|rel| RelatedMemoryResult {
            key: rel.memory_key,
            content: rel.content.unwrap_or_default(),
            relationship_type: format!("{:?}", rel.relationship_type),
            distance: rel.distance,
        }).collect())
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<SynapticMemoryStats> {
        let memory = self.memory.lock().await;
        let stats = memory.stats();
        
        Ok(SynapticMemoryStats {
            short_term_count: stats.short_term_count,
            long_term_count: stats.long_term_count,
            total_size: stats.total_size,
            session_id: self.session_id,
            created_at: self.created_at,
            knowledge_graph_stats: memory.knowledge_graph_stats(),
            #[cfg(feature = "embeddings")]
            embedding_stats: memory.embedding_stats(),
        })
    }

    /// Create a checkpoint of the current memory state
    pub async fn checkpoint(&self) -> Result<Uuid> {
        let memory = self.memory.lock().await;
        memory.checkpoint()
            .await
            .map_err(|e| AgentError::memory(format!("Failed to create checkpoint: {}", e)))
    }

    /// Clear all memories (use with caution)
    pub async fn clear(&self) -> Result<()> {
        let mut memory = self.memory.lock().await;
        memory.clear()
            .await
            .map_err(|e| AgentError::memory(format!("Failed to clear memories: {}", e)))
    }

    /// Check if a memory exists
    pub async fn has_memory(&self, key: &str) -> bool {
        let memory = self.memory.lock().await;
        memory.has_memory(key)
    }

    /// Get the session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get the configuration
    pub fn config(&self) -> &SynapticConfig {
        &self.config
    }
}

/// Result from a memory search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub key: String,
    pub content: String,
    pub relevance_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Result from a semantic search operation
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchResult {
    pub key: String,
    pub content: String,
    pub similarity_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Result from finding related memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedMemoryResult {
    pub key: String,
    pub content: String,
    pub relationship_type: String,
    pub distance: usize,
}

/// Statistics for the Synaptic memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticMemoryStats {
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub total_size: usize,
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub knowledge_graph_stats: Option<synaptic::memory::knowledge_graph::GraphStats>,
    #[cfg(feature = "embeddings")]
    pub embedding_stats: Option<synaptic::memory::embeddings::EmbeddingStats>,
}
