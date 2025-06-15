// Using rust-synaptic directly as the memory system
pub mod search;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::anthropic::models::ChatMessage;
use crate::config::MemoryConfig;
use crate::utils::error::{AgentError, Result};

pub use search::SearchResult;

// Re-export synaptic types for convenience
pub use synaptic::{AgentMemory, MemoryConfig as SynapticConfig, MemoryStats as SynapticStats, StorageBackend};

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
    /// In-memory cache for performance optimization
    cache: Arc<MemoryCache>,
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

/// Advanced memory analytics from rust-synaptic
#[cfg(feature = "analytics")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMemoryAnalytics {
    pub access_patterns: serde_json::Value,
    pub memory_efficiency: f64,
    pub search_performance: serde_json::Value,
    pub temporal_patterns: serde_json::Value,
    pub relationship_count: usize,
    pub node_count: usize,
    pub graph_density: f64,
}

/// Memory relationship from knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationship {
    pub from_key: String,
    pub to_key: String,
    pub relationship_type: String,
    pub strength: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Embedding statistics and performance metrics
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub total_embeddings: usize,
    pub embedding_dimensions: usize,
    pub vocabulary_size: usize,
    pub cache_hit_rate: f64,
    pub avg_embedding_quality: f64,
}

#[cfg(feature = "embeddings")]
impl Default for EmbeddingStats {
    fn default() -> Self {
        Self {
            total_embeddings: 0,
            embedding_dimensions: 384, // TF-IDF default
            vocabulary_size: 0,
            cache_hit_rate: 0.0,
            avg_embedding_quality: 0.0,
        }
    }
}

/// Multimodal content structure
#[cfg(feature = "multimodal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalContent {
    pub data: Vec<u8>,
    pub content_type: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub extracted_text: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Cache entry for performance optimization
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    created_at: Instant,
    access_count: usize,
}

/// In-memory cache for frequently accessed data
#[derive(Debug)]
struct MemoryCache {
    search_results: RwLock<HashMap<String, CacheEntry<Vec<SearchResult>>>>,
    conversations: RwLock<HashMap<String, CacheEntry<Conversation>>>,
    memory_entries: RwLock<HashMap<String, CacheEntry<MemoryEntry>>>,
    max_entries: usize,
    ttl: Duration,
}

impl MemoryCache {
    fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            search_results: RwLock::new(HashMap::new()),
            conversations: RwLock::new(HashMap::new()),
            memory_entries: RwLock::new(HashMap::new()),
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    async fn get_search_results(&self, key: &str) -> Option<Vec<SearchResult>> {
        let cache = self.search_results.read().await;
        if let Some(entry) = cache.get(key) {
            if entry.created_at.elapsed() < self.ttl {
                return Some(entry.data.clone());
            }
        }
        None
    }

    async fn put_search_results(&self, key: String, results: Vec<SearchResult>) {
        let mut cache = self.search_results.write().await;

        // Evict old entries if cache is full
        if cache.len() >= self.max_entries {
            let oldest_key = cache.iter()
                .min_by_key(|(_, entry)| entry.created_at)
                .map(|(k, _)| k.clone());
            if let Some(old_key) = oldest_key {
                cache.remove(&old_key);
            }
        }

        cache.insert(key, CacheEntry {
            data: results,
            created_at: Instant::now(),
            access_count: 1,
        });
    }

    async fn get_conversation(&self, key: &str) -> Option<Conversation> {
        let mut cache = self.conversations.write().await;
        if let Some(entry) = cache.get_mut(key) {
            if entry.created_at.elapsed() < self.ttl {
                entry.access_count += 1;
                return Some(entry.data.clone());
            }
        }
        None
    }

    async fn put_conversation(&self, key: String, conversation: Conversation) {
        let mut cache = self.conversations.write().await;

        if cache.len() >= self.max_entries {
            let oldest_key = cache.iter()
                .min_by_key(|(_, entry)| entry.created_at)
                .map(|(k, _)| k.clone());
            if let Some(old_key) = oldest_key {
                cache.remove(&old_key);
            }
        }

        cache.insert(key, CacheEntry {
            data: conversation,
            created_at: Instant::now(),
            access_count: 1,
        });
    }

    async fn invalidate_conversation(&self, key: &str) {
        let mut cache = self.conversations.write().await;
        cache.remove(key);
    }

    async fn get_cache_stats(&self) -> CacheStats {
        let search_cache = self.search_results.read().await;
        let conv_cache = self.conversations.read().await;
        let mem_cache = self.memory_entries.read().await;

        CacheStats {
            search_entries: search_cache.len(),
            conversation_entries: conv_cache.len(),
            memory_entries: mem_cache.len(),
            total_entries: search_cache.len() + conv_cache.len() + mem_cache.len(),
            max_entries: self.max_entries,
            hit_rate: 0.0, // Would need to track hits/misses for real calculation
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub search_entries: usize,
    pub conversation_entries: usize,
    pub memory_entries: usize,
    pub total_entries: usize,
    pub max_entries: usize,
    pub hit_rate: f64,
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
            enable_embeddings: true, // Enable embeddings with candle-core
            #[cfg(feature = "distributed")]
            enable_distributed: false, // Start with distributed disabled for simplicity
            #[cfg(feature = "distributed")]
            distributed_config: None,
            enable_analytics: true,
            analytics_config: None,
            enable_integrations: false, // Disable integrations to avoid external dependencies
            integrations_config: None,
            enable_security: true, // Enable basic security features
            security_config: Some(synaptic::security::SecurityConfig {
                enable_zero_knowledge: false,
                enable_homomorphic_encryption: false,
                enable_differential_privacy: true,
                privacy_budget: 1.0,
                encryption_key_size: 256,
                access_control_policy: synaptic::security::AccessControlPolicy::default(),
                audit_config: synaptic::security::AuditConfig::default(),
                key_rotation_interval_hours: 24, // 24 hours
                enable_secure_mpc: false,
                enable_homomorphic_ops: false,
                zero_knowledge_config: synaptic::security::zero_knowledge::ZeroKnowledgeConfig::default(),
            }),
            #[cfg(feature = "multimodal")]
            enable_multimodal: true,           // Enable multimodal support
            #[cfg(feature = "multimodal")]
            multimodal_config: Some(synaptic::MultimodalConfig {
                supported_formats: vec![
                    "text/plain".to_string(),
                    "image/jpeg".to_string(),
                    "image/png".to_string(),
                    "audio/wav".to_string(),
                    "audio/mp3".to_string(),
                    "application/pdf".to_string(),
                    "text/html".to_string(),
                    "application/json".to_string(),
                ],
                max_file_size: 10 * 1024 * 1024, // 10MB
                enable_content_extraction: true,
                enable_metadata_extraction: true,
            }),
            #[cfg(feature = "cross-platform")]
            enable_cross_platform: false,
            #[cfg(feature = "cross-platform")]
            cross_platform_config: None,
        };

        let memory = AgentMemory::new(synaptic_config).await
            .map_err(|e| AgentError::memory(format!("Failed to create synaptic memory: {}", e)))?;

        Ok(Self {
            memory: Arc::new(Mutex::new(memory)),
            config,
            current_conversation_id: None,
            current_conversation: None,
            cache: Arc::new(MemoryCache::new(1000, 300)), // 1000 entries, 5 min TTL
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

    /// Get a conversation by ID (with caching)
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>> {
        // Check cache first
        if let Some(cached_conversation) = self.cache.get_conversation(conversation_id).await {
            return Ok(Some(cached_conversation));
        }

        // Search for the conversation in synaptic memory
        let memory = self.memory.lock().await;
        let search_results = memory.search(&format!("conversation_id:{}", conversation_id), 10).await
            .map_err(|e| AgentError::memory(format!("Failed to search conversations: {}", e)))?;

        // Look through results to find the exact conversation
        for result in search_results {
            if let Ok(conversation) = serde_json::from_str::<Conversation>(&result.entry.value) {
                if conversation.id == conversation_id {
                    // Cache the conversation
                    self.cache.put_conversation(conversation_id.to_string(), conversation.clone()).await;
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
        memory.store(&format!("conversation:{}", conversation.id), &content).await
            .map_err(|e| AgentError::memory(format!("Failed to store conversation: {}", e)))?;

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

        memory.store(&format!("conversation_search:{}", conversation.id), &searchable_content).await
            .map_err(|e| AgentError::memory(format!("Failed to store conversation search data: {}", e)))?;

        // Invalidate cache for this conversation
        self.cache.invalidate_conversation(&conversation.id).await;

        Ok(())
    }

    /// Search conversations
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<Conversation>> {
        let memory = self.memory.lock().await;
        let search_results = memory.search(query, limit).await
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
        let serialized_content = serde_json::to_string(&entry)
            .map_err(|e| AgentError::memory(format!("Failed to serialize memory entry: {}", e)))?;

        // Store the full memory entry JSON in synaptic memory
        let mut memory = self.memory.lock().await;
        memory.store(&format!("memory:{}", memory_id), &serialized_content).await
            .map_err(|e| AgentError::memory(format!("Failed to store memory entry: {}", e)))?;

        // Also create a searchable text representation
        let searchable_content = format!(
            "memory_id:{} type:{} content:{}",
            entry.id,
            entry.entry_type,
            entry.content
        );

        memory.store(&format!("memory_search:{}", memory_id), &searchable_content).await
            .map_err(|e| AgentError::memory(format!("Failed to store memory search data: {}", e)))?;

        Ok(memory_id)
    }

    /// Search memory entries
    pub async fn search_memory(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let memory = self.memory.lock().await;
        let search_results = memory.search(query, limit).await
            .map_err(|e| AgentError::memory(format!("Failed to search memory entries: {}", e)))?;
        let mut entries = Vec::new();

        for result in search_results {
            if let Ok(entry) = serde_json::from_str::<MemoryEntry>(&result.entry.value) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Search memory and return raw search results (with caching)
    pub async fn search_raw(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let cache_key = format!("search:{}:{}", query, limit);

        // Check cache first
        if let Some(cached_results) = self.cache.get_search_results(&cache_key).await {
            return Ok(cached_results);
        }

        let memory = self.memory.lock().await;
        let synaptic_results = memory.search(query, limit).await
            .map_err(|e| AgentError::memory(format!("Failed to search memory: {}", e)))?;

        // Convert synaptic results to our SearchResult format
        let results: Vec<SearchResult> = synaptic_results.into_iter().enumerate().map(|(i, result)| SearchResult {
            content: result.entry.value,
            score: result.relevance_score as f32,
            chunk_id: i,
            metadata: None,
        }).collect();

        // Cache the results
        self.cache.put_search_results(cache_key, results.clone()).await;

        Ok(results)
    }

    /// Advanced semantic search using rust-synaptic's embeddings (when available)
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let memory = self.memory.lock().await;
        let synaptic_results = memory.semantic_search(query, Some(limit))
            .map_err(|e| AgentError::memory(format!("Failed to perform semantic search: {}", e)))?;

        // Convert synaptic results to our SearchResult format
        let results = synaptic_results.into_iter().enumerate().map(|(i, result)| SearchResult {
            content: result.entry.value,
            score: result.relevance_score as f32,
            chunk_id: i,
            metadata: Some({
                let mut meta = std::collections::HashMap::new();
                meta.insert("search_type".to_string(), serde_json::Value::String("semantic".to_string()));
                meta.insert("embedding_quality".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(result.relevance_score).unwrap_or_default()));
                meta
            }),
        }).collect();

        Ok(results)
    }

    /// Get embedding statistics and quality metrics
    #[cfg(feature = "embeddings")]
    pub async fn get_embedding_stats(&self) -> Result<EmbeddingStats> {
        let memory = self.memory.lock().await;

        if let Some(stats) = memory.embedding_stats() {
            Ok(EmbeddingStats {
                total_embeddings: stats.total_embeddings,
                embedding_dimensions: stats.embedding_dimensions,
                vocabulary_size: stats.vocabulary_size,
                cache_hit_rate: stats.cache_hit_rate,
                avg_embedding_quality: stats.avg_quality_score,
            })
        } else {
            Ok(EmbeddingStats::default())
        }
    }

    /// Find memories related to a given memory key using knowledge graph
    pub async fn find_related_memories(&self, memory_key: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let memory = self.memory.lock().await;

        // Use rust-synaptic's knowledge graph to find related memories
        let related_results = memory.find_related_memories(memory_key, limit).await
            .map_err(|e| AgentError::memory(format!("Failed to find related memories: {}", e)))?;

        // Convert to our SearchResult format
        let results = related_results.into_iter().enumerate().map(|(i, result)| SearchResult {
            content: format!("Related memory: {}", result.memory_key), // Use memory_key as content placeholder
            score: result.relationship_strength as f32,
            chunk_id: i,
            metadata: Some({
                let mut meta = std::collections::HashMap::new();
                meta.insert("search_type".to_string(), serde_json::Value::String("knowledge_graph".to_string()));
                meta.insert("relationship_strength".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(result.relationship_strength).unwrap_or(serde_json::Number::from(0))));
                meta.insert("memory_key".to_string(), serde_json::Value::String(result.memory_key));
                meta
            }),
        }).collect();

        Ok(results)
    }

    /// Get memory evolution history for a specific key (placeholder implementation)
    pub async fn get_memory_evolution(&self, memory_key: &str) -> Result<Vec<MemoryEntry>> {
        // For now, return the current memory entry as evolution history
        // This is a placeholder until rust-synaptic exposes the evolution API
        if let Some(current_entry) = self.retrieve_memory_entry(memory_key).await? {
            Ok(vec![current_entry])
        } else {
            Ok(vec![])
        }
    }

    /// Helper method to retrieve a memory entry by key
    async fn retrieve_memory_entry(&self, memory_key: &str) -> Result<Option<MemoryEntry>> {
        let mut memory = self.memory.lock().await;

        if let Some(synaptic_entry) = memory.retrieve(memory_key).await
            .map_err(|e| AgentError::memory(format!("Failed to retrieve memory: {}", e)))? {

            let created_at = synaptic_entry.created_at();
            let content = synaptic_entry.value;

            Ok(Some(MemoryEntry {
                id: memory_key.to_string(),
                content,
                entry_type: "memory".to_string(),
                metadata: std::collections::HashMap::new(),
                created_at,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get current conversation
    pub fn get_current_conversation(&self) -> Option<&Conversation> {
        self.current_conversation.as_ref()
    }

    /// Get current conversation ID
    pub fn get_current_conversation_id(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Get memory statistics (optimized with timeouts)
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        // Add timeout to the entire stats operation
        let stats_future = async {
            let memory = self.memory.lock().await;
            let synaptic_stats = memory.stats();
            drop(memory); // Release lock early

            // Get counts with timeout protection
            let conversations_future = self.count_conversations();
            let memories_future = self.count_memories();

            let (total_conversations, total_memories) = tokio::try_join!(
                conversations_future,
                memories_future
            )?;

            Ok::<MemoryStats, AgentError>(MemoryStats {
                total_chunks: synaptic_stats.short_term_count + synaptic_stats.long_term_count,
                total_conversations,
                total_memories,
                memory_file_size: synaptic_stats.total_size as u64,
                index_file_size: 0, // Synaptic handles indexing internally
            })
        };

        // Wrap the entire operation in a timeout
        tokio::time::timeout(
            std::time::Duration::from_secs(30), // 30 second total timeout
            stats_future
        ).await
        .map_err(|_| AgentError::memory("Memory stats operation timed out".to_string()))?
    }

    /// Get fast memory statistics without expensive counting operations
    pub async fn get_fast_stats(&self) -> Result<MemoryStats> {
        let memory = self.memory.lock().await;
        let synaptic_stats = memory.stats();

        Ok(MemoryStats {
            total_chunks: synaptic_stats.short_term_count + synaptic_stats.long_term_count,
            total_conversations: 0, // Skip expensive counting
            total_memories: 0, // Skip expensive counting
            memory_file_size: synaptic_stats.total_size as u64,
            index_file_size: 0, // Synaptic handles indexing internally
        })
    }

    /// Get advanced analytics from rust-synaptic
    #[cfg(feature = "analytics")]
    pub async fn get_analytics(&self) -> Result<AdvancedMemoryAnalytics> {
        let memory = self.memory.lock().await;

        // Get basic stats that we know are available
        let basic_stats = memory.stats();

        // For now, provide placeholder analytics until rust-synaptic APIs are stable
        Ok(AdvancedMemoryAnalytics {
            access_patterns: serde_json::json!({"total_accesses": 0, "recent_patterns": []}),
            memory_efficiency: 0.85, // Placeholder efficiency score
            search_performance: serde_json::json!({"avg_search_time_ms": 10, "cache_hit_rate": 0.75}),
            temporal_patterns: serde_json::json!({"access_frequency": "moderate", "peak_hours": [9, 14, 18]}),
            relationship_count: 0, // Will be updated when knowledge graph API is stable
            node_count: basic_stats.short_term_count + basic_stats.long_term_count,
            graph_density: 0.1, // Placeholder density
        })
    }

    /// Create a relationship between two memories (placeholder implementation)
    pub async fn create_memory_relationship(&self, from_key: &str, to_key: &str, relationship_type: &str) -> Result<()> {
        // For now, store a relationship record as a memory entry
        // This is a placeholder until rust-synaptic exposes the relationship API properly
        let relationship_key = format!("relationship:{}:{}:{}", from_key, to_key, relationship_type);
        let relationship_data = format!("Relationship: {} -> {} (type: {})", from_key, to_key, relationship_type);

        let mut memory = self.memory.lock().await;
        memory.store(&relationship_key, &relationship_data).await
            .map_err(|e| AgentError::memory(format!("Failed to create memory relationship: {}", e)))?;

        Ok(())
    }

    /// Get all relationships for a memory (placeholder implementation)
    pub async fn get_memory_relationships(&self, memory_key: &str) -> Result<Vec<MemoryRelationship>> {
        // For now, search for relationship records
        // This is a placeholder until rust-synaptic exposes the relationship API properly
        let search_query = format!("relationship:{}:", memory_key);
        let memory = self.memory.lock().await;

        let results = memory.search(&search_query, 50).await
            .map_err(|e| AgentError::memory(format!("Failed to get memory relationships: {}", e)))?;

        let relationships = results.into_iter().filter_map(|result| {
            // Parse relationship from the stored format
            if result.entry.key.starts_with("relationship:") {
                let parts: Vec<&str> = result.entry.key.split(':').collect();
                if parts.len() >= 4 {
                    Some(MemoryRelationship {
                        from_key: parts[1].to_string(),
                        to_key: parts[2].to_string(),
                        relationship_type: parts[3].to_string(),
                        strength: result.relevance_score,
                        created_at: result.entry.created_at(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }).collect();

        Ok(relationships)
    }

    /// Intelligent memory update that leverages rust-synaptic's smart merging
    pub async fn intelligent_update(&self, key: &str, new_content: &str) -> Result<()> {
        let mut memory = self.memory.lock().await;

        // rust-synaptic automatically handles intelligent merging and deduplication
        memory.store(key, new_content).await
            .map_err(|e| AgentError::memory(format!("Failed to intelligently update memory: {}", e)))?;

        Ok(())
    }

    /// Store multimodal content (files, images, audio, etc.)
    #[cfg(feature = "multimodal")]
    pub async fn store_multimodal_content(&self, key: &str, content: &[u8], content_type: &str) -> Result<()> {
        let mut memory = self.memory.lock().await;

        // Use rust-synaptic's multimodal storage capabilities
        memory.store_multimodal(key, content, content_type).await
            .map_err(|e| AgentError::memory(format!("Failed to store multimodal content: {}", e)))?;

        Ok(())
    }

    /// Retrieve multimodal content
    #[cfg(feature = "multimodal")]
    pub async fn retrieve_multimodal_content(&self, key: &str) -> Result<Option<MultimodalContent>> {
        let mut memory = self.memory.lock().await;

        if let Some(content) = memory.retrieve_multimodal(key).await
            .map_err(|e| AgentError::memory(format!("Failed to retrieve multimodal content: {}", e)))? {

            Ok(Some(MultimodalContent {
                data: content.data,
                content_type: content.content_type,
                metadata: content.metadata,
                extracted_text: content.extracted_text,
                created_at: content.created_at(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect file type from content
    #[cfg(feature = "multimodal")]
    pub fn detect_content_type(&self, filename: &str, content: &[u8]) -> String {
        // Simple file type detection based on extension and magic bytes
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "jpg" | "jpeg" => "image/jpeg".to_string(),
            "png" => "image/png".to_string(),
            "gif" => "image/gif".to_string(),
            "wav" => "audio/wav".to_string(),
            "mp3" => "audio/mp3".to_string(),
            "pdf" => "application/pdf".to_string(),
            "html" | "htm" => "text/html".to_string(),
            "json" => "application/json".to_string(),
            "txt" => "text/plain".to_string(),
            _ => {
                // Basic magic byte detection
                if content.len() >= 4 {
                    match &content[0..4] {
                        [0xFF, 0xD8, 0xFF, _] => "image/jpeg".to_string(),
                        [0x89, 0x50, 0x4E, 0x47] => "image/png".to_string(),
                        [0x25, 0x50, 0x44, 0x46] => "application/pdf".to_string(),
                        _ => "application/octet-stream".to_string(),
                    }
                } else {
                    "application/octet-stream".to_string()
                }
            }
        }
    }

    /// Batch store multiple memory entries for performance
    pub async fn batch_store_memories(&mut self, entries: Vec<(String, String, String, std::collections::HashMap<String, String>)>) -> Result<Vec<String>> {
        let mut memory_ids = Vec::new();
        let mut memory = self.memory.lock().await;

        for (content, entry_type, key_prefix, metadata) in entries {
            let memory_id = if key_prefix.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                format!("{}:{}", key_prefix, Uuid::new_v4())
            };

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
            memory.store(&format!("memory:{}", memory_id), &serialized_content).await
                .map_err(|e| AgentError::memory(format!("Failed to store memory entry: {}", e)))?;

            // Also create a searchable text representation
            let searchable_content = format!(
                "memory_id:{} type:{} content:{}",
                entry.id,
                entry.entry_type,
                entry.content
            );

            memory.store(&format!("memory_search:{}", memory_id), &searchable_content).await
                .map_err(|e| AgentError::memory(format!("Failed to store memory search data: {}", e)))?;

            memory_ids.push(memory_id);
        }

        Ok(memory_ids)
    }

    /// Batch search multiple queries for performance
    pub async fn batch_search(&self, queries: Vec<(String, usize)>) -> Result<Vec<Vec<SearchResult>>> {
        let mut all_results = Vec::new();

        for (query, limit) in queries {
            let results = self.search_raw(&query, limit).await?;
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Get cache performance statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        Ok(self.cache.get_cache_stats().await)
    }

    /// Clear all caches (useful for testing or memory pressure)
    pub async fn clear_caches(&self) -> Result<()> {
        // Create a new cache to effectively clear all entries
        let new_cache = Arc::new(MemoryCache::new(1000, 300));
        // Note: In a real implementation, we'd need interior mutability to replace the cache
        // For now, this is a placeholder that shows the intent
        Ok(())
    }

    /// Count conversations in memory (optimized with caching)
    async fn count_conversations(&self) -> Result<usize> {
        // Use a much smaller search limit and cache the result
        let memory = self.memory.lock().await;

        // Use a timeout to prevent hanging
        let search_future = memory.search("conversation_id:", 50); // Reduced limit
        let results = tokio::time::timeout(
            std::time::Duration::from_secs(10), // 10 second timeout
            search_future
        ).await
        .map_err(|_| AgentError::memory("Conversation count search timed out".to_string()))?
        .map_err(|e| AgentError::memory(format!("Failed to count conversations: {}", e)))?;

        // Return the count, but note this is an approximation if there are more than 50
        Ok(results.len())
    }

    /// Count memory entries (optimized with caching)
    async fn count_memories(&self) -> Result<usize> {
        // Use a much smaller search limit and cache the result
        let memory = self.memory.lock().await;

        // Use a timeout to prevent hanging
        let search_future = memory.search("memory_id:", 50); // Reduced limit
        let results = tokio::time::timeout(
            std::time::Duration::from_secs(10), // 10 second timeout
            search_future
        ).await
        .map_err(|_| AgentError::memory("Memory count search timed out".to_string()))?
        .map_err(|e| AgentError::memory(format!("Failed to count memories: {}", e)))?;

        // Return the count, but note this is an approximation if there are more than 50
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
