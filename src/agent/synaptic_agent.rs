//! # Synaptic Agent - Full Distributed Power
//!
//! This module provides an enhanced agent implementation that uses the
//! rust-synaptic memory system with full distributed power capabilities.

use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentConfig};
use crate::agent::conversation_manager::ConversationManager;
use crate::agent::tool_orchestrator::ToolOrchestrator;
use crate::anthropic::client::AnthropicClient;
use crate::anthropic::models::{ChatMessage, MessageRole, ContentBlock};
use crate::memory::synaptic::{
    SynapticMemoryManager, 
    SynapticConfig, 
    MemorySearchResult,
    bridge::{MemoryBridge, BridgeConfig, MemorySystem}
};
use crate::memory::synaptic::config::{
    FullPowerSynapticConfig,
    CoreMemoryConfig,
    DistributedConfig,
    IntegrationsConfig,
    EmbeddingsConfig,
    AnalyticsConfig,
    RealtimeConfig,
    EmbeddingModel,
    ConsensusAlgorithm
};
use crate::memory::synaptic::tools::{
    SynapticStoreTool,
    SynapticSearchTool,
    SynapticSemanticSearchTool,
    SynapticRelatedMemoriesTool,
    SynapticStatsTool,
    SynapticCheckpointTool
};
use crate::utils::error::{AgentError, Result};

/// Enhanced agent with full distributed synaptic memory power
#[derive(Debug)]
pub struct SynapticAgent {
    /// Core agent functionality
    base_agent: Agent,
    /// Synaptic memory bridge for advanced memory operations
    memory_bridge: Arc<Mutex<MemoryBridge>>,
    /// Full power configuration
    synaptic_config: FullPowerSynapticConfig,
    /// Agent session ID
    session_id: Uuid,
    /// Creation timestamp
    created_at: DateTime<Utc>,
}

/// Configuration for the Synaptic Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticAgentConfig {
    /// Base agent configuration
    pub base_config: AgentConfig,
    /// Full power synaptic configuration
    pub synaptic_config: FullPowerSynapticConfig,
    /// Memory bridge configuration
    pub bridge_config: BridgeConfig,
    /// Enable migration from simple memory
    pub enable_migration: bool,
    /// Enable all distributed features
    pub enable_full_power: bool,
}

impl Default for SynapticAgentConfig {
    fn default() -> Self {
        Self {
            base_config: AgentConfig::default(),
            synaptic_config: FullPowerSynapticConfig::default(),
            bridge_config: BridgeConfig {
                primary_system: MemorySystem::Synaptic,
                enable_dual_write: false,
                enable_fallback_reads: true,
                enable_migration: false,
                migration_batch_size: 1000,
                enable_distributed_consensus: true,
                enable_realtime_sync: true,
                enable_analytics: true,
            },
            enable_migration: false,
            enable_full_power: true,
        }
    }
}

impl SynapticAgent {
    /// Create a new Synaptic Agent with full distributed power
    pub async fn new(config: SynapticAgentConfig) -> Result<Self> {
        let session_id = Uuid::new_v4();
        
        // Create the base agent
        let base_agent = Agent::new(config.base_config.clone()).await?;
        
        // Convert synaptic config to the format expected by SynapticMemoryManager
        let synaptic_memory_config = SynapticConfig {
            enable_knowledge_graph: config.synaptic_config.core.enable_knowledge_graph,
            enable_temporal_tracking: config.synaptic_config.core.enable_temporal_tracking,
            enable_advanced_management: config.synaptic_config.core.enable_advanced_management,
            enable_embeddings: config.synaptic_config.embeddings.enable_embeddings,
            enable_distributed: config.synaptic_config.distributed.enable_consensus,
            enable_integrations: !config.synaptic_config.integrations.postgresql.is_none() ||
                                !config.synaptic_config.integrations.redis.is_none() ||
                                !config.synaptic_config.integrations.elasticsearch.is_none(),
            storage_backend: crate::memory::synaptic::SynapticStorageBackend::File {
                path: "synaptic_full_power.db".to_string(),
            },
            max_short_term_memories: config.synaptic_config.core.max_short_term_memories,
            max_long_term_memories: config.synaptic_config.core.max_long_term_memories,
            similarity_threshold: config.synaptic_config.core.similarity_threshold,
            checkpoint_interval: config.synaptic_config.core.checkpoint_interval,
        };

        // Create the memory bridge with full power
        let memory_bridge = if config.enable_migration {
            // Create bridge with both systems for migration
            MemoryBridge::with_both_memories_full_power(
                &config.base_config.memory_path,
                &config.base_config.index_path,
                synaptic_memory_config,
                config.bridge_config,
            ).await?
        } else {
            // Create bridge with synaptic only for maximum performance
            MemoryBridge::with_full_power_synaptic(
                synaptic_memory_config,
                config.bridge_config,
            ).await?
        };

        Ok(Self {
            base_agent,
            memory_bridge: Arc::new(Mutex::new(memory_bridge)),
            synaptic_config: config.synaptic_config,
            session_id,
            created_at: Utc::now(),
        })
    }

    /// Create a new Synaptic Agent with default full power configuration
    pub async fn with_full_power_defaults() -> Result<Self> {
        let mut config = SynapticAgentConfig::default();
        
        // Enable all full power features
        config.enable_full_power = true;
        config.synaptic_config.distributed.enable_consensus = true;
        config.synaptic_config.distributed.enable_failover = true;
        config.synaptic_config.distributed.enable_load_balancing = true;
        config.synaptic_config.embeddings.enable_embeddings = true;
        config.synaptic_config.embeddings.enable_semantic_search = true;
        config.synaptic_config.embeddings.enable_ai_consolidation = true;
        config.synaptic_config.analytics.enable_performance_analytics = true;
        config.synaptic_config.analytics.enable_distributed_metrics = true;
        config.synaptic_config.realtime.enable_realtime_sync = true;
        config.synaptic_config.realtime.enable_memory_streaming = true;

        Self::new(config).await
    }

    /// Process a chat message with full synaptic power
    pub async fn chat_with_synaptic_power(&mut self, message: &str) -> Result<String> {
        // Add the user message to conversation
        let user_message = ChatMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text { text: message.to_string() }],
            id: Some(Uuid::new_v4().to_string()),
            timestamp: Some(Utc::now()),
        };

        self.base_agent.conversation_manager.add_message(user_message).await?;

        // Store the message in synaptic memory for future reference
        let memory_key = format!("user_message_{}", Uuid::new_v4());
        self.memory_bridge.lock().await.add_chunk(message.to_string()).await?;

        // Search for relevant memories using full power features
        let relevant_memories = self.search_relevant_memories(message, 5).await?;
        
        // Add memory context to the conversation if relevant memories found
        if !relevant_memories.is_empty() {
            let memory_context = self.format_memory_context(&relevant_memories);
            let context_message = ChatMessage {
                role: MessageRole::User,
                content: vec![ContentBlock::Text { 
                    text: format!("Relevant memories from synaptic system:\n{}", memory_context)
                }],
                id: Some(Uuid::new_v4().to_string()),
                timestamp: Some(Utc::now()),
            };
            self.base_agent.conversation_manager.add_message(context_message).await?;
        }

        // Get response from the base agent
        let response = self.base_agent.process_turn().await?;

        // Store the response in synaptic memory
        if let Some(text_content) = self.extract_text_from_response(&response) {
            let response_key = format!("agent_response_{}", Uuid::new_v4());
            self.memory_bridge.lock().await.add_chunk(text_content.clone()).await?;
            
            Ok(text_content)
        } else {
            Ok("I processed your request, but couldn't generate a text response.".to_string())
        }
    }

    /// Search for relevant memories using full synaptic power
    async fn search_relevant_memories(&self, query: &str, limit: usize) -> Result<Vec<MemorySearchResult>> {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.search_full_power(query, limit).await
    }

    /// Perform semantic search using embeddings
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search(&self, query: &str, limit: Option<usize>) -> Result<Vec<crate::memory::synaptic::SemanticSearchResult>> {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.semantic_search_full_power(query, limit).await
    }

    /// Find related memories using knowledge graph
    pub async fn find_related_memories(&self, memory_key: &str, max_depth: usize) -> Result<Vec<crate::memory::synaptic::RelatedMemoryResult>> {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.find_related_full_power(memory_key, max_depth).await
    }

    /// Get comprehensive memory statistics
    pub async fn get_memory_stats(&self) -> Result<crate::memory::synaptic::bridge::FullPowerStats> {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.get_full_power_stats().await
    }

    /// Create a memory checkpoint
    pub async fn create_checkpoint(&self) -> Result<Uuid> {
        // This would need to be implemented based on the actual synaptic memory manager
        // For now, return a placeholder
        Ok(Uuid::new_v4())
    }

    /// Migrate from simple memory to synaptic (if enabled)
    pub async fn migrate_to_synaptic(&self) -> Result<crate::memory::synaptic::bridge::MigrationResult> {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.migrate_to_full_power_synaptic().await
    }

    /// Check if full power features are available
    pub async fn has_full_power(&self) -> bool {
        let memory_bridge = self.memory_bridge.lock().await;
        memory_bridge.has_full_power()
    }

    /// Get the session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get the synaptic configuration
    pub fn synaptic_config(&self) -> &FullPowerSynapticConfig {
        &self.synaptic_config
    }

    /// Format memory context for inclusion in conversation
    fn format_memory_context(&self, memories: &[MemorySearchResult]) -> String {
        memories.iter()
            .enumerate()
            .map(|(i, memory)| {
                format!("{}. [Score: {:.2}] {}", 
                    i + 1, 
                    memory.relevance_score, 
                    memory.content.chars().take(200).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract text content from agent response
    fn extract_text_from_response(&self, content: &[ContentBlock]) -> Option<String> {
        content.iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
            .into()
    }
}

/// Statistics for the Synaptic Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticAgentStats {
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub memory_stats: crate::memory::synaptic::bridge::FullPowerStats,
    pub full_power_enabled: bool,
    pub distributed_features: Vec<String>,
    pub integration_features: Vec<String>,
}
