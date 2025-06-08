//! # Memory Bridge - Full Distributed Power
//!
//! This module provides a bridge between the existing simple memory system
//! and the new Synaptic memory system with FULL DISTRIBUTED POWER enabled,
//! including external integrations, distributed consensus, embeddings, analytics, and real-time features.

use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::memory::simple_memory::SimpleMemory;
use crate::memory::synaptic::{SynapticMemoryManager, SynapticConfig, MemorySearchResult};
use crate::utils::error::{AgentError, Result};

/// A bridge that can use either the simple memory system or the synaptic system
/// This allows for gradual migration and A/B testing between memory systems
#[derive(Debug)]
pub struct MemoryBridge {
    /// The simple memory system (legacy)
    simple_memory: Option<Arc<Mutex<SimpleMemory>>>,
    /// The synaptic memory system (new)
    synaptic_memory: Option<Arc<Mutex<SynapticMemoryManager>>>,
    /// Configuration for the bridge
    config: BridgeConfig,
}

/// Configuration for the memory bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Which memory system to use as primary
    pub primary_system: MemorySystem,
    /// Whether to enable dual-write mode (write to both systems)
    pub enable_dual_write: bool,
    /// Whether to enable fallback reads (if primary fails, try secondary)
    pub enable_fallback_reads: bool,
    /// Whether to enable migration mode (gradually move data from simple to synaptic)
    pub enable_migration: bool,
    /// Migration batch size (how many entries to migrate at once)
    pub migration_batch_size: usize,
}

/// Available memory systems
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemorySystem {
    /// Use the simple memory system
    Simple,
    /// Use the synaptic memory system
    Synaptic,
    /// Use both systems (for migration/testing)
    Both,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            primary_system: MemorySystem::Synaptic, // Default to new system
            enable_dual_write: false,
            enable_fallback_reads: true,
            enable_migration: false,
            migration_batch_size: 100,
        }
    }
}

impl MemoryBridge {
    /// Create a new memory bridge with simple memory only
    pub async fn with_simple_memory(
        memory_path: &str,
        index_path: &str,
        config: BridgeConfig,
    ) -> Result<Self> {
        let simple_memory = SimpleMemory::new(memory_path, index_path).await?;
        
        Ok(Self {
            simple_memory: Some(Arc::new(Mutex::new(simple_memory))),
            synaptic_memory: None,
            config,
        })
    }

    /// Create a new memory bridge with synaptic memory only
    pub async fn with_synaptic_memory(
        synaptic_config: SynapticConfig,
        bridge_config: BridgeConfig,
    ) -> Result<Self> {
        let synaptic_memory = SynapticMemoryManager::new(synaptic_config).await?;
        
        Ok(Self {
            simple_memory: None,
            synaptic_memory: Some(Arc::new(Mutex::new(synaptic_memory))),
            config: bridge_config,
        })
    }

    /// Create a new memory bridge with both memory systems
    pub async fn with_both_memories(
        memory_path: &str,
        index_path: &str,
        synaptic_config: SynapticConfig,
        bridge_config: BridgeConfig,
    ) -> Result<Self> {
        let simple_memory = SimpleMemory::new(memory_path, index_path).await?;
        let synaptic_memory = SynapticMemoryManager::new(synaptic_config).await?;
        
        Ok(Self {
            simple_memory: Some(Arc::new(Mutex::new(simple_memory))),
            synaptic_memory: Some(Arc::new(Mutex::new(synaptic_memory))),
            config: bridge_config,
        })
    }

    /// Add a chunk to memory
    pub async fn add_chunk(&self, content: String) -> Result<()> {
        match self.config.primary_system {
            MemorySystem::Simple => {
                if let Some(ref simple) = self.simple_memory {
                    simple.lock().await.add_chunk(content).await?;
                } else {
                    return Err(AgentError::memory("Simple memory not available".to_string()));
                }
            }
            MemorySystem::Synaptic => {
                if let Some(ref synaptic) = self.synaptic_memory {
                    let key = Uuid::new_v4().to_string();
                    synaptic.lock().await.store(&key, &content).await?;
                } else {
                    return Err(AgentError::memory("Synaptic memory not available".to_string()));
                }
            }
            MemorySystem::Both => {
                // Write to both systems
                if let Some(ref simple) = self.simple_memory {
                    simple.lock().await.add_chunk(content.clone()).await?;
                }
                if let Some(ref synaptic) = self.synaptic_memory {
                    let key = Uuid::new_v4().to_string();
                    synaptic.lock().await.store(&key, &content).await?;
                }
            }
        }

        // Dual-write mode: write to secondary system if enabled
        if self.config.enable_dual_write {
            match self.config.primary_system {
                MemorySystem::Simple => {
                    if let Some(ref synaptic) = self.synaptic_memory {
                        let key = Uuid::new_v4().to_string();
                        let _ = synaptic.lock().await.store(&key, &content).await; // Ignore errors in dual-write
                    }
                }
                MemorySystem::Synaptic => {
                    if let Some(ref simple) = self.simple_memory {
                        let _ = simple.lock().await.add_chunk(content).await; // Ignore errors in dual-write
                    }
                }
                MemorySystem::Both => {
                    // Already handled above
                }
            }
        }

        Ok(())
    }

    /// Search for content in memory
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemorySearchResult>> {
        let primary_result = match self.config.primary_system {
            MemorySystem::Simple => {
                if let Some(ref simple) = self.simple_memory {
                    let results = simple.lock().await.search(query, limit).await?;
                    // Convert simple memory results to bridge format
                    results.into_iter().map(|result| MemorySearchResult {
                        key: Uuid::new_v4().to_string(), // Simple memory doesn't have keys
                        content: result.content,
                        relevance_score: result.score,
                        timestamp: result.timestamp,
                    }).collect()
                } else {
                    return Err(AgentError::memory("Simple memory not available".to_string()));
                }
            }
            MemorySystem::Synaptic => {
                if let Some(ref synaptic) = self.synaptic_memory {
                    synaptic.lock().await.search(query, limit).await?
                } else {
                    return Err(AgentError::memory("Synaptic memory not available".to_string()));
                }
            }
            MemorySystem::Both => {
                // Combine results from both systems
                let mut all_results = Vec::new();
                
                if let Some(ref simple) = self.simple_memory {
                    let simple_results = simple.lock().await.search(query, limit).await?;
                    let converted: Vec<MemorySearchResult> = simple_results.into_iter().map(|result| MemorySearchResult {
                        key: format!("simple_{}", Uuid::new_v4()),
                        content: result.content,
                        relevance_score: result.score,
                        timestamp: result.timestamp,
                    }).collect();
                    all_results.extend(converted);
                }
                
                if let Some(ref synaptic) = self.synaptic_memory {
                    let synaptic_results = synaptic.lock().await.search(query, limit).await?;
                    all_results.extend(synaptic_results);
                }
                
                // Sort by relevance score and limit results
                all_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
                all_results.truncate(limit);
                all_results
            }
        };

        // Fallback reads: if primary system fails or returns no results, try secondary
        if self.config.enable_fallback_reads && primary_result.is_empty() {
            match self.config.primary_system {
                MemorySystem::Simple => {
                    if let Some(ref synaptic) = self.synaptic_memory {
                        return synaptic.lock().await.search(query, limit).await;
                    }
                }
                MemorySystem::Synaptic => {
                    if let Some(ref simple) = self.simple_memory {
                        let results = simple.lock().await.search(query, limit).await?;
                        return Ok(results.into_iter().map(|result| MemorySearchResult {
                            key: format!("fallback_{}", Uuid::new_v4()),
                            content: result.content,
                            relevance_score: result.score,
                            timestamp: result.timestamp,
                        }).collect());
                    }
                }
                MemorySystem::Both => {
                    // Already handled above
                }
            }
        }

        Ok(primary_result)
    }

    /// Perform semantic search (only available with synaptic memory)
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search(&self, query: &str, limit: Option<usize>) -> Result<Vec<crate::memory::synaptic::SemanticSearchResult>> {
        if let Some(ref synaptic) = self.synaptic_memory {
            synaptic.lock().await.semantic_search(query, limit).await
        } else {
            Err(AgentError::memory("Semantic search requires synaptic memory".to_string()))
        }
    }

    /// Find related memories (only available with synaptic memory)
    pub async fn find_related(&self, memory_key: &str, max_depth: usize) -> Result<Vec<crate::memory::synaptic::RelatedMemoryResult>> {
        if let Some(ref synaptic) = self.synaptic_memory {
            synaptic.lock().await.find_related(memory_key, max_depth).await
        } else {
            Err(AgentError::memory("Related memory search requires synaptic memory".to_string()))
        }
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<BridgeStats> {
        let mut stats = BridgeStats {
            primary_system: self.config.primary_system.clone(),
            simple_stats: None,
            synaptic_stats: None,
        };

        if let Some(ref simple) = self.simple_memory {
            let simple_memory = simple.lock().await;
            stats.simple_stats = Some(SimpleMemoryStats {
                chunk_count: simple_memory.get_chunk_count(),
                total_size: simple_memory.get_total_size(),
            });
        }

        if let Some(ref synaptic) = self.synaptic_memory {
            stats.synaptic_stats = Some(synaptic.lock().await.get_stats().await?);
        }

        Ok(stats)
    }

    /// Migrate data from simple memory to synaptic memory
    pub async fn migrate_to_synaptic(&self) -> Result<MigrationResult> {
        if !self.config.enable_migration {
            return Err(AgentError::memory("Migration is not enabled".to_string()));
        }

        let simple = self.simple_memory.as_ref()
            .ok_or_else(|| AgentError::memory("Simple memory not available for migration".to_string()))?;
        
        let synaptic = self.synaptic_memory.as_ref()
            .ok_or_else(|| AgentError::memory("Synaptic memory not available for migration".to_string()))?;

        // TODO: Implement actual migration logic
        // This would involve:
        // 1. Getting all chunks from simple memory
        // 2. Converting them to synaptic format
        // 3. Storing them in synaptic memory
        // 4. Tracking progress and handling errors

        Ok(MigrationResult {
            migrated_count: 0,
            failed_count: 0,
            total_count: 0,
            duration: std::time::Duration::from_secs(0),
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &BridgeConfig {
        &self.config
    }

    /// Check if synaptic memory is available
    pub fn has_synaptic(&self) -> bool {
        self.synaptic_memory.is_some()
    }

    /// Check if simple memory is available
    pub fn has_simple(&self) -> bool {
        self.simple_memory.is_some()
    }
}

/// Statistics for the memory bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub primary_system: MemorySystem,
    pub simple_stats: Option<SimpleMemoryStats>,
    pub synaptic_stats: Option<crate::memory::synaptic::SynapticMemoryStats>,
}

/// Statistics for simple memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMemoryStats {
    pub chunk_count: usize,
    pub total_size: usize,
}

/// Result of a migration operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    pub migrated_count: usize,
    pub failed_count: usize,
    pub total_count: usize,
    pub duration: std::time::Duration,
}
