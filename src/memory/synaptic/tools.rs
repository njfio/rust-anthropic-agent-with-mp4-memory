//! # Synaptic Memory Tools - Full Distributed Power
//!
//! This module provides tools for interacting with the synaptic memory system
//! through the agent's tool interface, enabling full distributed power features.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::tools::{Tool, ToolResult, create_tool_definition};
use crate::anthropic::models::ToolDefinition;
use crate::memory::synaptic::SynapticMemoryManager;
use crate::utils::error::{AgentError, Result};

/// Tool for storing memories in the synaptic system
pub struct SynapticStoreTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

/// Tool for searching memories in the synaptic system
pub struct SynapticSearchTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

/// Tool for semantic search using embeddings
pub struct SynapticSemanticSearchTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

/// Tool for finding related memories using knowledge graph
pub struct SynapticRelatedMemoriesTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

/// Tool for getting memory statistics and analytics
pub struct SynapticStatsTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

/// Tool for creating memory checkpoints
pub struct SynapticCheckpointTool {
    memory_manager: Arc<Mutex<SynapticMemoryManager>>,
}

impl SynapticStoreTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

impl SynapticSearchTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

impl SynapticSemanticSearchTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

impl SynapticRelatedMemoriesTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

impl SynapticStatsTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

impl SynapticCheckpointTool {
    pub fn new(memory_manager: Arc<Mutex<SynapticMemoryManager>>) -> Self {
        Self { memory_manager }
    }
}

#[async_trait]
impl Tool for SynapticStoreTool {
    fn name(&self) -> &str {
        "synaptic_store_memory"
    }

    fn description(&self) -> Option<&str> {
        Some("Store a memory in the synaptic memory system with full distributed power. Supports knowledge graph relationships, temporal tracking, and distributed consensus.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_store",
            "Store a memory in the synaptic memory system with full distributed power. Supports knowledge graph relationships, temporal tracking, and distributed consensus.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique key for the memory entry"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to store in memory"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorizing the memory"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for the memory entry"
                    }
                },
                "required": ["key", "content"]
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let key = input["key"].as_str()
            .ok_or_else(|| AgentError::tool("synaptic_store", "Missing 'key' parameter"))?;
        let content = input["content"].as_str()
            .ok_or_else(|| AgentError::tool("synaptic_store", "Missing 'content' parameter"))?;

        let memory_manager = self.memory_manager.lock().await;
        memory_manager.store(key, content).await?;

        Ok(ToolResult::success(format!(
            "Successfully stored memory with key '{}' in synaptic memory system",
            key
        )))
    }
}

#[async_trait]
impl Tool for SynapticSearchTool {
    fn name(&self) -> &str {
        "synaptic_search_memory"
    }

    fn description(&self) -> Option<&str> {
        Some("Search for memories in the synaptic memory system using advanced algorithms including knowledge graph traversal and temporal analysis.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_search_memory",
            "Search for memories in the synaptic memory system using advanced algorithms including knowledge graph traversal and temporal analysis.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "include_temporal": {
                        "type": "boolean",
                        "description": "Include temporal analysis in search",
                        "default": true
                    },
                    "include_knowledge_graph": {
                        "type": "boolean",
                        "description": "Include knowledge graph relationships in search",
                        "default": true
                    }
                },
                "required": ["query"]
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let query = input["query"].as_str()
            .ok_or_else(|| AgentError::tool("synaptic_search_memory", "Missing 'query' parameter"))?;
        let limit = input["limit"].as_u64().unwrap_or(10) as usize;

        let memory_manager = self.memory_manager.lock().await;
        let results = memory_manager.search(query, limit).await?;

        let search_results: Vec<Value> = results.into_iter().map(|result| {
            serde_json::json!({
                "key": result.key,
                "content": result.content,
                "relevance_score": result.relevance_score,
                "timestamp": result.timestamp
            })
        }).collect();

        Ok(ToolResult::success(
            format!("Found {} memories matching query '{}': {}", search_results.len(), query, serde_json::to_string(&serde_json::json!({
                "results": search_results,
                "query": query,
                "total_found": search_results.len()
            })).unwrap_or_default())
        ))
    }
}

#[async_trait]
impl Tool for SynapticSemanticSearchTool {
    fn name(&self) -> &str {
        "synaptic_semantic_search"
    }

    fn description(&self) -> Option<&str> {
        Some("Perform semantic search using vector embeddings in the synaptic memory system. Finds memories based on semantic similarity rather than keyword matching.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_semantic_search",
            "Perform semantic search using vector embeddings in the synaptic memory system. Finds memories based on semantic similarity rather than keyword matching.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0 to 1.0)",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let query = input["query"].as_str()
            .ok_or_else(|| AgentError::tool("synaptic_semantic_search", "Missing 'query' parameter"))?;
        let limit = input["limit"].as_u64().map(|l| l as usize);

        #[cfg(feature = "embeddings")]
        {
            let memory_manager = self.memory_manager.lock().await;
            let results = memory_manager.semantic_search(query, limit).await?;

            let search_results: Vec<Value> = results.into_iter().map(|result| {
                serde_json::json!({
                    "key": result.key,
                    "content": result.content,
                    "similarity_score": result.similarity_score,
                    "timestamp": result.timestamp
                })
            }).collect();

            Ok(ToolResult::success(
                format!("Found {} semantically similar memories for query '{}': {}", search_results.len(), query, serde_json::to_string(&serde_json::json!({
                    "results": search_results,
                    "query": query,
                    "search_type": "semantic",
                    "total_found": search_results.len()
                })).unwrap_or_default())
            ))
        }

        #[cfg(not(feature = "embeddings"))]
        {
            Ok(ToolResult::error("Semantic search requires embeddings feature to be enabled".to_string()))
        }
    }
}

#[async_trait]
impl Tool for SynapticRelatedMemoriesTool {
    fn name(&self) -> &str {
        "synaptic_find_related"
    }

    fn description(&self) -> Option<&str> {
        Some("Find memories related to a specific memory using the knowledge graph. Discovers connections and relationships between memories.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_find_related",
            "Find memories related to a specific memory using the knowledge graph. Discovers connections and relationships between memories.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "memory_key": {
                        "type": "string",
                        "description": "Key of the memory to find related memories for"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse in the knowledge graph",
                        "default": 3
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of relationships to follow (optional)"
                    }
                },
                "required": ["memory_key"]
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let memory_key = input["memory_key"].as_str()
            .ok_or_else(|| AgentError::tool("synaptic_find_related", "Missing 'memory_key' parameter"))?;
        let max_depth = input["max_depth"].as_u64().unwrap_or(3) as usize;

        let memory_manager = self.memory_manager.lock().await;
        let results = memory_manager.find_related(memory_key, max_depth).await?;

        let related_results: Vec<Value> = results.into_iter().map(|result| {
            serde_json::json!({
                "key": result.key,
                "content": result.content,
                "relationship_type": result.relationship_type,
                "distance": result.distance
            })
        }).collect();

        Ok(ToolResult::success(
            format!("Found {} related memories for key '{}': {}", related_results.len(), memory_key, serde_json::to_string(&serde_json::json!({
                "results": related_results,
                "source_key": memory_key,
                "max_depth": max_depth,
                "total_found": related_results.len()
            })).unwrap_or_default())
        ))
    }
}

#[async_trait]
impl Tool for SynapticStatsTool {
    fn name(&self) -> &str {
        "synaptic_memory_stats"
    }

    fn description(&self) -> Option<&str> {
        Some("Get comprehensive statistics and analytics for the synaptic memory system, including distributed system metrics.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_memory_stats",
            "Get comprehensive statistics and analytics for the synaptic memory system, including distributed system metrics.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "include_detailed": {
                        "type": "boolean",
                        "description": "Include detailed analytics and metrics",
                        "default": true
                    },
                    "include_distributed": {
                        "type": "boolean",
                        "description": "Include distributed system metrics",
                        "default": true
                    }
                }
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let memory_manager = self.memory_manager.lock().await;
        let stats = memory_manager.get_stats().await?;

        Ok(ToolResult::success(
            format!("Retrieved synaptic memory statistics: {}", serde_json::to_string(&serde_json::to_value(stats).map_err(|e| AgentError::tool("synaptic_memory_stats", &format!("Failed to serialize stats: {}", e)))?).unwrap_or_default())
        ))
    }
}

#[async_trait]
impl Tool for SynapticCheckpointTool {
    fn name(&self) -> &str {
        "synaptic_create_checkpoint"
    }

    fn description(&self) -> Option<&str> {
        Some("Create a checkpoint of the current synaptic memory state for backup and recovery purposes.")
    }

    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "synaptic_create_checkpoint",
            "Create a checkpoint of the current synaptic memory state for backup and recovery purposes.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Optional description for the checkpoint"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include metadata in the checkpoint",
                        "default": true
                    }
                }
            })
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let description = input["description"].as_str().unwrap_or("Manual checkpoint");

        let memory_manager = self.memory_manager.lock().await;
        let checkpoint_id = memory_manager.checkpoint().await?;

        Ok(ToolResult::success(
            format!("Created checkpoint with ID: {}: {}", checkpoint_id, serde_json::to_string(&serde_json::json!({
                "checkpoint_id": checkpoint_id,
                "description": description,
                "created_at": chrono::Utc::now()
            })).unwrap_or_default())
        ))
    }
}
