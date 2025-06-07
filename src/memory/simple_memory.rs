use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use std::collections::HashMap;

use crate::memory::SearchResult;
use crate::utils::error::{AgentError, Result};

/// Simple memory chunk for JSON storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub id: usize,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: Option<String>,
}

/// Simple JSON-based memory storage (fast replacement for memvid)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryData {
    pub chunks: Vec<MemoryChunk>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Fast JSON-based memory wrapper (replaces slow memvid system)
#[derive(Debug)]
pub struct SimpleMemory {
    memory_path: PathBuf,
    data: MemoryData,
    next_id: usize,
}

impl SimpleMemory {
    /// Initialize a new fast JSON-based memory wrapper
    pub async fn new(memory_path: &Path, _index_path: &Path) -> Result<Self> {
        let memory_path = memory_path.to_path_buf();
        
        debug!("Initializing fast JSON memory at: {:?}", memory_path);

        // Load existing data if file exists, or create new if incompatible format
        let data = if memory_path.exists() {
            debug!("Loading existing memory data");
            let content = fs::read_to_string(&memory_path)
                .map_err(|e| AgentError::memory(format!("Failed to read memory file: {}", e)))?;

            // Try to parse as new format, fallback to creating new if incompatible
            match serde_json::from_str::<MemoryData>(&content) {
                Ok(data) => {
                    debug!("Successfully loaded existing JSON memory data with {} chunks", data.chunks.len());
                    data
                }
                Err(e) => {
                    debug!("Existing memory file has incompatible format ({}), creating new JSON memory", e);
                    // Backup the old file
                    let backup_path = memory_path.with_extension("mp4.backup");
                    if let Err(backup_err) = fs::copy(&memory_path, &backup_path) {
                        debug!("Failed to backup old memory file: {}", backup_err);
                    } else {
                        debug!("Backed up old memory file to: {:?}", backup_path);
                    }

                    MemoryData {
                        chunks: Vec::new(),
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                    }
                }
            }
        } else {
            debug!("Creating new memory data");
            MemoryData {
                chunks: Vec::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            }
        };

        let next_id = data.chunks.iter().map(|c| c.id).max().unwrap_or(0) + 1;

        info!("Loaded {} chunks from memory", data.chunks.len());

        Ok(Self {
            memory_path,
            data,
            next_id,
        })
    }

    /// Add a text chunk to memory
    pub async fn add_chunk<S: Into<String>>(&mut self, chunk: S) -> Result<()> {
        let content = chunk.into();
        debug!("Adding chunk to memory: {} chars", content.len());
        
        let chunk = MemoryChunk {
            id: self.next_id,
            content,
            timestamp: chrono::Utc::now(),
            source: Some("agent".to_string()),
        };
        
        self.data.chunks.push(chunk);
        self.next_id += 1;
        self.data.updated_at = chrono::Utc::now();
        
        // Save immediately for persistence
        self.save().await?;
        
        Ok(())
    }

    /// Add multiple chunks to memory
    pub async fn add_chunks(&mut self, chunks: Vec<String>) -> Result<()> {
        debug!("Adding {} chunks to memory", chunks.len());
        
        for content in chunks {
            let chunk = MemoryChunk {
                id: self.next_id,
                content,
                timestamp: chrono::Utc::now(),
                source: Some("agent".to_string()),
            };
            
            self.data.chunks.push(chunk);
            self.next_id += 1;
        }
        
        self.data.updated_at = chrono::Utc::now();
        
        // Save after adding all chunks
        self.save().await?;
        
        Ok(())
    }

    /// Save memory data to file
    async fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.data)
            .map_err(|e| AgentError::memory(format!("Failed to serialize memory data: {}", e)))?;
        
        fs::write(&self.memory_path, json)
            .map_err(|e| AgentError::memory(format!("Failed to write memory file: {}", e)))?;
        
        Ok(())
    }

    /// Build the memory (no-op for JSON storage)
    pub async fn build_video(&mut self) -> Result<()> {
        debug!("Build video called - no-op for JSON memory");
        Ok(())
    }

    /// Search through memory using simple text matching
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Searching memory for: '{}' (limit: {})", query, limit);

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for chunk in &self.data.chunks {
            if chunk.content.to_lowercase().contains(&query_lower) {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("timestamp".to_string(), serde_json::Value::String(chunk.timestamp.to_rfc3339()));
                if let Some(ref source) = chunk.source {
                    metadata.insert("source".to_string(), serde_json::Value::String(source.clone()));
                }

                results.push(SearchResult {
                    content: chunk.content.clone(),
                    score: 1.0, // Simple scoring
                    chunk_id: chunk.id,
                    metadata: Some(metadata),
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        // Sort by chunk ID (newest first)
        results.sort_by(|a, b| b.chunk_id.cmp(&a.chunk_id));

        debug!("Found {} search results", results.len());
        Ok(results)
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let file_size = if self.memory_path.exists() {
            std::fs::metadata(&self.memory_path)?.len()
        } else {
            0
        };

        Ok(MemoryStats {
            total_chunks: self.data.chunks.len(),
            file_size_bytes: file_size,
        })
    }

    /// Check if memory is built (always true for JSON)
    pub fn is_built(&self) -> bool {
        true
    }

    /// Get the number of chunks
    pub fn chunk_count(&self) -> usize {
        self.data.chunks.len()
    }

    /// Clear all chunks
    pub fn clear_chunks(&mut self) {
        self.data.chunks.clear();
        self.data.updated_at = chrono::Utc::now();
    }
}

/// Simple memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_chunks: usize,
    pub file_size_bytes: u64,
}
