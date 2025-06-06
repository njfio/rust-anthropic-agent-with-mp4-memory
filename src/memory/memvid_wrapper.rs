use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::memory::SearchResult;
use crate::utils::error::{AgentError, Result};

/// Wrapper around the rust-mp4-memory library
#[derive(Debug)]
pub struct MemvidWrapper {
    memory_path: PathBuf,
    index_path: PathBuf,
    chunks: Vec<String>,
    is_built: bool,
}

/// Statistics about the memory system
#[derive(Debug, Clone)]
pub struct MemvidStats {
    pub total_chunks: usize,
    pub video_size_bytes: u64,
    pub index_size_bytes: u64,
}

impl MemvidWrapper {
    /// Create a new MemVid wrapper
    pub async fn new<P: AsRef<Path>>(memory_path: P, index_path: P) -> Result<Self> {
        let memory_path = memory_path.as_ref().to_path_buf();
        let index_path = index_path.as_ref().to_path_buf();

        // Initialize rust_mem_vid if not already done
        if let Err(e) = rust_mem_vid::init().await {
            warn!("Failed to initialize rust_mem_vid: {}", e);
        }

        let mut wrapper = Self {
            memory_path,
            index_path,
            chunks: Vec::new(),
            is_built: false,
        };

        // Load existing memory if it exists
        wrapper.load_existing().await?;

        Ok(wrapper)
    }

    /// Load existing memory if files exist
    async fn load_existing(&mut self) -> Result<()> {
        if self.memory_path.exists() && self.index_path.exists() {
            info!("Loading existing memory from {:?}", self.memory_path);

            // Load chunks from the index file
            let content = std::fs::read_to_string(&self.index_path)?;
            if let Ok(chunks) = serde_json::from_str::<Vec<String>>(&content) {
                self.chunks = chunks;
                info!("Loaded {} chunks from existing memory", self.chunks.len());
            } else {
                warn!("Failed to parse existing memory index, starting fresh");
            }

            self.is_built = true;
        } else {
            debug!("No existing memory found, starting fresh");
        }
        Ok(())
    }

    /// Add a text chunk to memory
    pub async fn add_chunk<S: Into<String>>(&mut self, chunk: S) -> Result<()> {
        let chunk = chunk.into();
        debug!("Adding chunk to memory: {} chars", chunk.len());
        
        self.chunks.push(chunk);
        self.is_built = false; // Mark as needing rebuild
        
        Ok(())
    }

    /// Add multiple chunks to memory
    pub async fn add_chunks(&mut self, chunks: Vec<String>) -> Result<()> {
        debug!("Adding {} chunks to memory", chunks.len());
        
        self.chunks.extend(chunks);
        self.is_built = false;
        
        Ok(())
    }

    /// Build the memory video (encode chunks into MP4)
    pub async fn build_video(&mut self) -> Result<()> {
        if self.chunks.is_empty() {
            warn!("No chunks to build into video");
            return Ok(());
        }

        info!("Building memory video with {} chunks", self.chunks.len());

        // For now, we'll simulate the video building process
        // In a real implementation, this would use the rust-mp4-memory library
        // to encode the chunks into a video file
        
        // TODO: Implement actual video building using rust_mem_vid
        // let mut encoder = rust_mem_vid::MemvidEncoder::new().await?;
        // encoder.add_chunks(self.chunks.clone()).await?;
        // encoder.build_video(&self.memory_path, &self.index_path).await?;

        // For now, we'll just save the chunks as JSON for testing
        let chunks_json = serde_json::to_string_pretty(&self.chunks)
            .map_err(|e| AgentError::memory(format!("Failed to serialize chunks: {}", e)))?;
        
        std::fs::write(&self.index_path, chunks_json)?;
        
        // Create a dummy video file
        std::fs::write(&self.memory_path, b"dummy video content")?;

        self.is_built = true;
        info!("Memory video built successfully");

        Ok(())
    }

    /// Search through memory
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Searching memory for: '{}' (limit: {})", query, limit);

        if !self.is_built && !self.memory_path.exists() {
            return Ok(Vec::new());
        }

        // For now, we'll implement a simple text search
        // In a real implementation, this would use the rust-mp4-memory library
        // for semantic search
        
        // TODO: Implement actual semantic search using rust_mem_vid
        // let retriever = rust_mem_vid::MemvidRetriever::new(&self.memory_path, &self.index_path).await?;
        // let results = retriever.search(query, limit).await?;

        // Simple text search implementation for testing
        let mut results = Vec::new();
        
        // Load chunks from index file if it exists
        let chunks = if self.index_path.exists() {
            let content = std::fs::read_to_string(&self.index_path)?;
            serde_json::from_str::<Vec<String>>(&content)
                .unwrap_or_else(|_| self.chunks.clone())
        } else {
            self.chunks.clone()
        };

        let query_lower = query.to_lowercase();
        
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.to_lowercase().contains(&query_lower) {
                results.push(SearchResult {
                    content: chunk.clone(),
                    score: 1.0 - (i as f32 / chunks.len() as f32), // Simple scoring
                    chunk_id: i,
                    metadata: None,
                });
                
                if results.len() >= limit {
                    break;
                }
            }
        }

        // Sort by score (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        debug!("Found {} search results", results.len());
        Ok(results)
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemvidStats> {
        let video_size = if self.memory_path.exists() {
            std::fs::metadata(&self.memory_path)?.len()
        } else {
            0
        };

        let index_size = if self.index_path.exists() {
            std::fs::metadata(&self.index_path)?.len()
        } else {
            0
        };

        Ok(MemvidStats {
            total_chunks: self.chunks.len(),
            video_size_bytes: video_size,
            index_size_bytes: index_size,
        })
    }

    /// Extract text from a specific frame (for debugging)
    pub async fn extract_frame(&self, frame_number: usize) -> Result<Option<String>> {
        // TODO: Implement frame extraction using rust_mem_vid
        // let retriever = rust_mem_vid::MemvidRetriever::new(&self.memory_path, &self.index_path).await?;
        // let frame_content = retriever.extract_frame(frame_number).await?;
        
        // For now, return a chunk if it exists
        if frame_number < self.chunks.len() {
            Ok(Some(self.chunks[frame_number].clone()))
        } else {
            Ok(None)
        }
    }

    /// Get information about the memory
    pub async fn get_info(&self) -> Result<MemoryInfo> {
        let stats = self.get_stats().await?;
        
        Ok(MemoryInfo {
            memory_path: self.memory_path.clone(),
            index_path: self.index_path.clone(),
            is_built: self.is_built,
            stats,
        })
    }

    /// Clear all chunks (but keep files)
    pub fn clear_chunks(&mut self) {
        self.chunks.clear();
        self.is_built = false;
    }

    /// Check if memory is built
    pub fn is_built(&self) -> bool {
        self.is_built
    }

    /// Get the number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

/// Information about the memory system
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub memory_path: PathBuf,
    pub index_path: PathBuf,
    pub is_built: bool,
    pub stats: MemvidStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_memvid_wrapper_creation() {
        let temp_dir = tempdir().unwrap();
        let memory_path = temp_dir.path().join("test_memory.mp4");
        let index_path = temp_dir.path().join("test_index.json");

        let wrapper = MemvidWrapper::new(&memory_path, &index_path).await;
        assert!(wrapper.is_ok());
    }

    #[tokio::test]
    async fn test_add_and_search_chunks() {
        let temp_dir = tempdir().unwrap();
        let memory_path = temp_dir.path().join("test_memory.mp4");
        let index_path = temp_dir.path().join("test_index.json");

        let mut wrapper = MemvidWrapper::new(&memory_path, &index_path).await.unwrap();
        
        wrapper.add_chunk("This is a test chunk about AI").await.unwrap();
        wrapper.add_chunk("Another chunk about machine learning").await.unwrap();
        wrapper.build_video().await.unwrap();

        let results = wrapper.search("AI", 10).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("AI"));
    }
}
