//! Agent-specific memory management system
//!
//! This module provides high-level memory management functionality specifically
//! designed for AI agents, including conversation context tracking, episodic memory,
//! working memory, and long-term memory consolidation patterns.

use crate::memory::{Conversation, MemoryEntry, MemoryManager, SearchResult};
use crate::utils::error::Result;
use crate::utils::resource_monitor::ResourceMonitor;
use crate::caching::CacheManager;
use crate::security::SecurityManager;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::debug;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export memory types for convenience
pub use crate::memory::{Conversation as BaseConversation, MemoryEntry as BaseMemoryEntry, MemoryManager as BaseMemoryManager, SearchResult as BaseSearchResult};

/// Configuration for agent memory management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemoryConfig {
    /// Maximum size of working memory (number of entries)
    pub working_memory_size: usize,
    /// Duration to keep items in working memory
    pub working_memory_ttl: Duration,
    /// Maximum conversation context length
    pub max_context_length: usize,
    /// Episodic memory retention period
    pub episodic_retention_days: u64,
    /// Long-term memory consolidation threshold
    pub consolidation_threshold: usize,
    /// Memory importance scoring weights
    pub importance_weights: ImportanceWeights,
    /// Enable automatic memory consolidation
    pub auto_consolidation: bool,
    /// Consolidation interval
    pub consolidation_interval: Duration,
}

/// Weights for calculating memory importance scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceWeights {
    /// Weight for recency (newer = more important)
    pub recency: f64,
    /// Weight for frequency of access
    pub frequency: f64,
    /// Weight for emotional significance
    pub emotional: f64,
    /// Weight for task relevance
    pub relevance: f64,
    /// Weight for user interaction quality
    pub interaction_quality: f64,
}

/// Types of memory in the agent system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    /// Working memory for immediate task execution
    Working,
    /// Episodic memory for specific events and interactions
    Episodic,
    /// Long-term memory for consolidated knowledge
    LongTerm,
    /// Conversation context for dialogue management
    Conversational,
}

/// Memory entry with agent-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemoryEntry {
    /// Unique identifier
    pub id: String,
    /// Memory type classification
    pub memory_type: MemoryType,
    /// Content of the memory
    pub content: String,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when created
    pub created_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Access frequency counter
    pub access_count: u64,
    /// Importance score (0.0 to 1.0)
    pub importance_score: f64,
    /// Emotional significance (-1.0 to 1.0)
    pub emotional_score: f64,
    /// Task relevance score (0.0 to 1.0)
    pub relevance_score: f64,
    /// Associated conversation ID
    pub conversation_id: Option<String>,
    /// User ID associated with this memory
    pub user_id: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Working memory for immediate task execution
#[derive(Debug)]
struct WorkingMemory {
    /// Current active entries
    entries: VecDeque<AgentMemoryEntry>,
    /// Maximum capacity
    max_size: usize,
    /// Time-to-live for entries
    ttl: Duration,
    /// Last cleanup time
    last_cleanup: Instant,
}

/// Conversation context tracker
#[derive(Debug)]
struct ConversationContext {
    /// Current conversation entries
    entries: VecDeque<AgentMemoryEntry>,
    /// Maximum context length
    max_length: usize,
    /// Current conversation ID
    conversation_id: Option<String>,
    /// Conversation start time
    start_time: Option<SystemTime>,
    /// Conversation metadata
    metadata: HashMap<String, String>,
}

/// Memory consolidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Number of entries consolidated
    pub entries_consolidated: u64,
    /// Number of entries archived
    pub entries_archived: u64,
    /// Number of entries deleted
    pub entries_deleted: u64,
    /// Last consolidation time
    pub last_consolidation: SystemTime,
    /// Average importance score of consolidated entries
    pub avg_importance_score: f64,
    /// Memory usage before consolidation (bytes)
    pub memory_before: u64,
    /// Memory usage after consolidation (bytes)
    pub memory_after: u64,
}

/// Agent-specific memory manager
pub struct AgentMemoryManager {
    /// Configuration
    config: AgentMemoryConfig,
    /// Base memory manager (synaptic integration)
    base_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
    /// Working memory for immediate tasks
    working_memory: Arc<RwLock<WorkingMemory>>,
    /// Conversation context tracker
    conversation_context: Arc<RwLock<ConversationContext>>,
    /// Episodic memory cache
    episodic_cache: Arc<RwLock<HashMap<String, AgentMemoryEntry>>>,
    /// Cache manager for performance
    cache_manager: Option<Arc<CacheManager>>,
    /// Resource monitor for memory usage tracking
    resource_monitor: Option<Arc<ResourceMonitor>>,
    /// Security manager for access control
    security_manager: Option<Arc<SecurityManager>>,
    /// Consolidation statistics
    consolidation_stats: Arc<RwLock<ConsolidationStats>>,
    /// Last consolidation time
    last_consolidation: Arc<RwLock<Instant>>,
}

impl Default for AgentMemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 100,
            working_memory_ttl: Duration::from_secs(3600), // 1 hour
            max_context_length: 50,
            episodic_retention_days: 30,
            consolidation_threshold: 1000,
            importance_weights: ImportanceWeights::default(),
            auto_consolidation: true,
            consolidation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            recency: 0.3,
            frequency: 0.2,
            emotional: 0.2,
            relevance: 0.2,
            interaction_quality: 0.1,
        }
    }
}

impl Default for ConsolidationStats {
    fn default() -> Self {
        Self {
            entries_consolidated: 0,
            entries_archived: 0,
            entries_deleted: 0,
            last_consolidation: SystemTime::now(),
            avg_importance_score: 0.0,
            memory_before: 0,
            memory_after: 0,
        }
    }
}

impl WorkingMemory {
    fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size),
            max_size,
            ttl,
            last_cleanup: Instant::now(),
        }
    }

    fn add_entry(&mut self, entry: AgentMemoryEntry) {
        // Remove oldest entries if at capacity
        while self.entries.len() >= self.max_size {
            if let Some(removed) = self.entries.pop_front() {
                debug!("Removed oldest working memory entry: {}", removed.id);
            }
        }

        self.entries.push_back(entry);
        debug!("Added entry to working memory, current size: {}", self.entries.len());
    }

    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_cleanup) < Duration::from_secs(60) {
            return; // Only cleanup every minute
        }

        let cutoff = SystemTime::now() - self.ttl;
        let initial_len = self.entries.len();

        self.entries.retain(|entry| entry.created_at > cutoff);

        let removed_count = initial_len - self.entries.len();
        if removed_count > 0 {
            debug!("Cleaned up {} expired working memory entries", removed_count);
        }

        self.last_cleanup = now;
    }

    fn get_entries(&self) -> Vec<AgentMemoryEntry> {
        self.entries.iter().cloned().collect()
    }

    fn find_by_id(&self, id: &str) -> Option<&AgentMemoryEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }
}

impl ConversationContext {
    fn new(max_length: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_length),
            max_length,
            conversation_id: None,
            start_time: None,
            metadata: HashMap::new(),
        }
    }

    fn start_conversation(&mut self, conversation_id: String, metadata: HashMap<String, String>) {
        self.conversation_id = Some(conversation_id);
        self.start_time = Some(SystemTime::now());
        self.metadata = metadata;
        self.entries.clear();
        debug!("Started new conversation: {:?}", self.conversation_id);
    }

    fn add_entry(&mut self, mut entry: AgentMemoryEntry) {
        // Set conversation ID
        entry.conversation_id = self.conversation_id.clone();

        // Remove oldest entries if at capacity
        while self.entries.len() >= self.max_length {
            if let Some(removed) = self.entries.pop_front() {
                debug!("Removed oldest conversation entry: {}", removed.id);
            }
        }

        self.entries.push_back(entry);
        debug!("Added entry to conversation context, current size: {}", self.entries.len());
    }

    fn get_context(&self) -> Vec<AgentMemoryEntry> {
        self.entries.iter().cloned().collect()
    }

    fn get_conversation_summary(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        let total_entries = self.entries.len();
        let start_time = self.start_time?;
        let duration = SystemTime::now().duration_since(start_time).ok()?;

        Some(format!(
            "Conversation {} with {} entries over {:?}",
            self.conversation_id.as_ref().unwrap_or(&"unknown".to_string()),
            total_entries,
            duration
        ))
    }
}

impl AgentMemoryEntry {
    /// Create a new agent memory entry
    pub fn new(
        memory_type: MemoryType,
        content: String,
        metadata: HashMap<String, String>,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id: Uuid::new_v4().to_string(),
            memory_type,
            content,
            metadata,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance_score: 0.5, // Default neutral importance
            emotional_score: 0.0,  // Default neutral emotion
            relevance_score: 0.5,  // Default neutral relevance
            conversation_id: None,
            user_id: None,
            tags: Vec::new(),
        }
    }

    /// Update access tracking
    pub fn mark_accessed(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }

    /// Calculate importance score based on weights
    pub fn calculate_importance(&mut self, weights: &ImportanceWeights) -> f64 {
        let now = SystemTime::now();

        // Recency score (0.0 to 1.0, newer is higher)
        let recency_score = if let Ok(duration) = now.duration_since(self.created_at) {
            let hours = duration.as_secs() as f64 / 3600.0;
            (1.0 / (1.0 + hours / 24.0)).max(0.0).min(1.0) // Decay over days
        } else {
            0.0
        };

        // Frequency score (normalized by access count)
        let frequency_score = (self.access_count as f64 / 10.0).min(1.0);

        // Emotional score (absolute value, strong emotions are important)
        let emotional_score = self.emotional_score.abs();

        // Relevance score (as provided)
        let relevance_score = self.relevance_score;

        // Interaction quality score (based on content length and metadata richness)
        let interaction_quality_score = {
            let content_quality = (self.content.len() as f64 / 1000.0).min(1.0);
            let metadata_quality = (self.metadata.len() as f64 / 10.0).min(1.0);
            (content_quality + metadata_quality) / 2.0
        };

        // Calculate weighted importance
        let importance = weights.recency * recency_score
            + weights.frequency * frequency_score
            + weights.emotional * emotional_score
            + weights.relevance * relevance_score
            + weights.interaction_quality * interaction_quality_score;

        self.importance_score = importance.max(0.0).min(1.0);
        self.importance_score
    }
}

impl AgentMemoryManager {
    /// Create a new agent memory manager
    pub fn new(
        config: AgentMemoryConfig,
        base_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
    ) -> Self {
        let working_memory = Arc::new(RwLock::new(WorkingMemory::new(
            config.working_memory_size,
            config.working_memory_ttl,
        )));

        let conversation_context = Arc::new(RwLock::new(ConversationContext::new(
            config.max_context_length,
        )));

        Self {
            config,
            base_manager,
            working_memory,
            conversation_context,
            episodic_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_manager: None,
            resource_monitor: None,
            security_manager: None,
            consolidation_stats: Arc::new(RwLock::new(ConsolidationStats::default())),
            last_consolidation: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Set cache manager for performance optimization
    pub async fn set_cache_manager(&mut self, cache_manager: Arc<CacheManager>) {
        self.cache_manager = Some(cache_manager);
        debug!("Cache manager set for agent memory");
    }

    /// Set resource monitor for memory usage tracking
    pub async fn set_resource_monitor(&mut self, resource_monitor: Arc<ResourceMonitor>) {
        self.resource_monitor = Some(resource_monitor);
        debug!("Resource monitor set for agent memory");
    }

    /// Set security manager for access control
    pub async fn set_security_manager(&mut self, security_manager: Arc<SecurityManager>) {
        self.security_manager = Some(security_manager);
        debug!("Security manager set for agent memory");
    }

    /// Store a memory entry in the appropriate memory type
    pub async fn store_memory(
        &self,
        memory_type: MemoryType,
        content: String,
        metadata: HashMap<String, String>,
        user_id: Option<String>,
        tags: Vec<String>,
    ) -> Result<String> {
        let mut entry = AgentMemoryEntry::new(memory_type.clone(), content, metadata);
        entry.user_id = user_id;
        entry.tags = tags;

        // Calculate initial importance score
        entry.calculate_importance(&self.config.importance_weights);

        match memory_type {
            MemoryType::Working => {
                let mut working = self.working_memory.write().await;
                working.cleanup_expired();
                working.add_entry(entry.clone());
                debug!("Stored entry in working memory: {}", entry.id);
            }
            MemoryType::Conversational => {
                let mut context = self.conversation_context.write().await;
                context.add_entry(entry.clone());
                debug!("Stored entry in conversation context: {}", entry.id);
            }
            MemoryType::Episodic => {
                // Store in episodic cache and base memory manager
                let mut cache = self.episodic_cache.write().await;
                cache.insert(entry.id.clone(), entry.clone());

                // Store in base memory manager using save_memory
                let mut base_manager = self.base_manager.lock().await;
                base_manager.save_memory(
                    entry.content.clone(),
                    format!("{:?}", entry.memory_type),
                    entry.metadata.clone(),
                ).await?;
                debug!("Stored entry in episodic memory: {}", entry.id);
            }
            MemoryType::LongTerm => {
                // Store directly in base memory manager for long-term persistence
                let mut base_manager = self.base_manager.lock().await;
                base_manager.save_memory(
                    entry.content.clone(),
                    format!("{:?}", entry.memory_type),
                    entry.metadata.clone(),
                ).await?;
                debug!("Stored entry in long-term memory: {}", entry.id);
            }
        }

        Ok(entry.id)
    }

    /// Retrieve memory entries by type
    pub async fn get_memories(&self, memory_type: MemoryType, limit: Option<usize>) -> Result<Vec<AgentMemoryEntry>> {
        match memory_type {
            MemoryType::Working => {
                let mut working = self.working_memory.write().await;
                working.cleanup_expired();
                let entries = working.get_entries();
                Ok(if let Some(limit) = limit {
                    entries.into_iter().take(limit).collect()
                } else {
                    entries
                })
            }
            MemoryType::Conversational => {
                let context = self.conversation_context.read().await;
                let entries = context.get_context();
                Ok(if let Some(limit) = limit {
                    entries.into_iter().take(limit).collect()
                } else {
                    entries
                })
            }
            MemoryType::Episodic => {
                let cache = self.episodic_cache.read().await;
                let mut entries: Vec<_> = cache.values().cloned().collect();

                // Sort by importance and recency
                entries.sort_by(|a, b| {
                    b.importance_score.partial_cmp(&a.importance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.created_at.cmp(&a.created_at))
                });

                Ok(if let Some(limit) = limit {
                    entries.into_iter().take(limit).collect()
                } else {
                    entries
                })
            }
            MemoryType::LongTerm => {
                // Query base memory manager for long-term memories
                let base_manager = self.base_manager.lock().await;
                let search_results = base_manager.search_memory("", 100).await?;
                let entries = self.convert_from_memory_entries(search_results, MemoryType::LongTerm).await?;

                Ok(if let Some(limit) = limit {
                    entries.into_iter().take(limit).collect()
                } else {
                    entries
                })
            }
        }
    }

    /// Search memories across all types
    pub async fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<AgentMemoryEntry>> {
        let mut all_results = Vec::new();

        // Search working memory
        let working = self.working_memory.read().await;
        for entry in working.get_entries() {
            if entry.content.to_lowercase().contains(&query.to_lowercase()) {
                all_results.push(entry);
            }
        }

        // Search conversation context
        let context = self.conversation_context.read().await;
        for entry in context.get_context() {
            if entry.content.to_lowercase().contains(&query.to_lowercase()) {
                all_results.push(entry);
            }
        }

        // Search episodic cache
        let cache = self.episodic_cache.read().await;
        for entry in cache.values() {
            if entry.content.to_lowercase().contains(&query.to_lowercase()) {
                all_results.push(entry.clone());
            }
        }

        // Search base memory manager for long-term memories
        let base_manager = self.base_manager.lock().await;
        let search_results = base_manager.search_memory(query, limit).await?;
        let long_term_entries = self.convert_from_memory_entries(search_results, MemoryType::LongTerm).await?;
        all_results.extend(long_term_entries);

        // Sort by importance score and limit results
        all_results.sort_by(|a, b| {
            b.importance_score.partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results.into_iter().take(limit).collect())
    }

    /// Start a new conversation context
    pub async fn start_conversation(
        &self,
        conversation_id: String,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let mut context = self.conversation_context.write().await;
        context.start_conversation(conversation_id, metadata);
        debug!("Started new conversation context");
        Ok(())
    }

    /// Get current conversation context
    pub async fn get_conversation_context(&self) -> Result<Vec<AgentMemoryEntry>> {
        let context = self.conversation_context.read().await;
        Ok(context.get_context())
    }

    /// Get conversation summary
    pub async fn get_conversation_summary(&self) -> Result<Option<String>> {
        let context = self.conversation_context.read().await;
        Ok(context.get_conversation_summary())
    }

    /// Add entry to current conversation
    pub async fn add_to_conversation(
        &self,
        content: String,
        metadata: HashMap<String, String>,
        user_id: Option<String>,
        tags: Vec<String>,
    ) -> Result<String> {
        self.store_memory(MemoryType::Conversational, content, metadata, user_id, tags).await
    }

    /// Perform memory consolidation
    pub async fn consolidate_memories(&self) -> Result<ConsolidationStats> {
        let mut stats = ConsolidationStats::default();
        let start_time = SystemTime::now();

        // Get current memory usage
        stats.memory_before = self.estimate_memory_usage().await?;

        // Consolidate episodic memories based on importance
        let mut episodic_cache = self.episodic_cache.write().await;
        let mut entries_to_consolidate = Vec::new();
        let mut entries_to_archive = Vec::new();
        let mut entries_to_delete = Vec::new();

        let cutoff_date = SystemTime::now() - Duration::from_secs(self.config.episodic_retention_days * 24 * 3600);

        for (id, entry) in episodic_cache.iter() {
            if entry.created_at < cutoff_date {
                if entry.importance_score > 0.7 {
                    // High importance: consolidate to long-term
                    entries_to_consolidate.push(entry.clone());
                } else if entry.importance_score > 0.3 {
                    // Medium importance: archive
                    entries_to_archive.push(id.clone());
                } else {
                    // Low importance: delete
                    entries_to_delete.push(id.clone());
                }
            }
        }

        // Process consolidation
        for entry in entries_to_consolidate {
            let mut long_term_entry = entry.clone();
            long_term_entry.memory_type = MemoryType::LongTerm;

            let mut base_manager = self.base_manager.lock().await;
            base_manager.save_memory(
                long_term_entry.content.clone(),
                format!("{:?}", long_term_entry.memory_type),
                long_term_entry.metadata.clone(),
            ).await?;
            drop(base_manager); // Release lock early

            episodic_cache.remove(&entry.id);
            stats.entries_consolidated += 1;
        }

        // Archive medium importance entries (keep in episodic but mark as archived)
        for id in entries_to_archive {
            if let Some(entry) = episodic_cache.get_mut(&id) {
                entry.metadata.insert("archived".to_string(), "true".to_string());
                stats.entries_archived += 1;
            }
        }

        // Delete low importance entries
        for id in entries_to_delete {
            episodic_cache.remove(&id);
            stats.entries_deleted += 1;
        }

        // Calculate average importance score
        if !episodic_cache.is_empty() {
            let total_importance: f64 = episodic_cache.values().map(|e| e.importance_score).sum();
            stats.avg_importance_score = total_importance / episodic_cache.len() as f64;
        }

        // Update consolidation time
        stats.last_consolidation = start_time;
        stats.memory_after = self.estimate_memory_usage().await?;

        // Update internal stats
        let mut internal_stats = self.consolidation_stats.write().await;
        *internal_stats = stats.clone();

        let mut last_consolidation = self.last_consolidation.write().await;
        *last_consolidation = Instant::now();

        debug!(
            "Memory consolidation completed: {} consolidated, {} archived, {} deleted",
            stats.entries_consolidated, stats.entries_archived, stats.entries_deleted
        );

        Ok(stats)
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> Result<HashMap<String, u64>> {
        let mut stats = HashMap::new();

        // Working memory stats
        let working = self.working_memory.read().await;
        stats.insert("working_memory_entries".to_string(), working.entries.len() as u64);

        // Conversation context stats
        let context = self.conversation_context.read().await;
        stats.insert("conversation_entries".to_string(), context.entries.len() as u64);

        // Episodic memory stats
        let episodic = self.episodic_cache.read().await;
        stats.insert("episodic_entries".to_string(), episodic.len() as u64);

        // Memory usage estimate
        let memory_usage = self.estimate_memory_usage().await?;
        stats.insert("estimated_memory_bytes".to_string(), memory_usage);

        // Consolidation stats
        let consolidation_stats = self.consolidation_stats.read().await;
        stats.insert("entries_consolidated".to_string(), consolidation_stats.entries_consolidated);
        stats.insert("entries_archived".to_string(), consolidation_stats.entries_archived);
        stats.insert("entries_deleted".to_string(), consolidation_stats.entries_deleted);

        Ok(stats)
    }

    /// Check if automatic consolidation should run
    pub async fn should_consolidate(&self) -> Result<bool> {
        if !self.config.auto_consolidation {
            return Ok(false);
        }

        let last_consolidation = self.last_consolidation.read().await;
        let time_since_last = last_consolidation.elapsed();

        if time_since_last < self.config.consolidation_interval {
            return Ok(false);
        }

        // Check if we have enough entries to warrant consolidation
        let episodic = self.episodic_cache.read().await;
        Ok(episodic.len() >= self.config.consolidation_threshold)
    }

    /// Run automatic consolidation if needed
    pub async fn auto_consolidate(&self) -> Result<Option<ConsolidationStats>> {
        if self.should_consolidate().await? {
            Ok(Some(self.consolidate_memories().await?))
        } else {
            Ok(None)
        }
    }

    /// Convert MemoryEntry to AgentMemoryEntry
    async fn convert_from_memory_entries(
        &self,
        memory_entries: Vec<BaseMemoryEntry>,
        memory_type: MemoryType,
    ) -> Result<Vec<AgentMemoryEntry>> {
        let mut entries = Vec::new();

        for base_entry in memory_entries {
            let mut entry = AgentMemoryEntry {
                id: base_entry.id,
                memory_type: memory_type.clone(),
                content: base_entry.content,
                metadata: base_entry.metadata.clone(),
                created_at: base_entry.created_at.into(),
                last_accessed: SystemTime::now(),
                access_count: base_entry.metadata
                    .get("access_count")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                importance_score: base_entry.metadata
                    .get("importance_score")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5),
                emotional_score: base_entry.metadata
                    .get("emotional_score")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                relevance_score: base_entry.metadata
                    .get("relevance_score")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5),
                conversation_id: base_entry.metadata.get("conversation_id").cloned(),
                user_id: base_entry.metadata.get("user_id").cloned(),
                tags: base_entry.metadata
                    .get("tags")
                    .map(|s| s.split(',').map(|t| t.to_string()).collect())
                    .unwrap_or_default(),
            };

            // Mark as accessed
            entry.mark_accessed();
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Estimate memory usage in bytes
    async fn estimate_memory_usage(&self) -> Result<u64> {
        let mut total_bytes = 0u64;

        // Working memory
        let working = self.working_memory.read().await;
        for entry in &working.entries {
            total_bytes += self.estimate_entry_size(entry);
        }

        // Conversation context
        let context = self.conversation_context.read().await;
        for entry in &context.entries {
            total_bytes += self.estimate_entry_size(entry);
        }

        // Episodic cache
        let episodic = self.episodic_cache.read().await;
        for entry in episodic.values() {
            total_bytes += self.estimate_entry_size(entry);
        }

        Ok(total_bytes)
    }

    /// Estimate size of a single entry in bytes
    fn estimate_entry_size(&self, entry: &AgentMemoryEntry) -> u64 {
        let mut size = 0u64;

        // Basic fields
        size += entry.id.len() as u64;
        size += entry.content.len() as u64;
        size += 8 * 8; // timestamps and scores (8 fields * 8 bytes each)

        // Metadata
        for (key, value) in &entry.metadata {
            size += key.len() as u64 + value.len() as u64;
        }

        // Tags
        for tag in &entry.tags {
            size += tag.len() as u64;
        }

        // Optional fields
        if let Some(conversation_id) = &entry.conversation_id {
            size += conversation_id.len() as u64;
        }

        if let Some(user_id) = &entry.user_id {
            size += user_id.len() as u64;
        }

        size
    }
}

/// Create a new agent memory manager with default configuration
pub fn create_agent_memory_manager(base_manager: Arc<tokio::sync::Mutex<MemoryManager>>) -> AgentMemoryManager {
    AgentMemoryManager::new(AgentMemoryConfig::default(), base_manager)
}

/// Create agent memory manager with custom configuration
pub fn create_agent_memory_manager_with_config(
    config: AgentMemoryConfig,
    base_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
) -> AgentMemoryManager {
    AgentMemoryManager::new(config, base_manager)
}

impl std::fmt::Debug for AgentMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentMemoryManager")
            .field("config", &self.config)
            .field("base_manager", &"<MemoryManager>")
            .field("working_memory", &"<WorkingMemory>")
            .field("conversation_context", &"<ConversationContext>")
            .field("episodic_cache", &"<EpisodicCache>")
            .field("cache_manager", &self.cache_manager.is_some())
            .field("resource_monitor", &self.resource_monitor.is_some())
            .field("security_manager", &self.security_manager.is_some())
            .finish()
    }
}
