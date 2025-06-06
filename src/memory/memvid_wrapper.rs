use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::memory::SearchResult;
use crate::utils::error::{AgentError, Result};

/// Wrapper around the rust-mp4-memory library
pub struct MemvidWrapper {
    memory_path: PathBuf,
    index_path: PathBuf,
    encoder: Option<rust_mem_vid::MemvidEncoder>,
    retriever: Option<rust_mem_vid::MemvidRetriever>,
    chunks: Vec<String>,
    is_built: bool,
}

impl std::fmt::Debug for MemvidWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemvidWrapper")
            .field("memory_path", &self.memory_path)
            .field("index_path", &self.index_path)
            .field("chunks", &self.chunks.len())
            .field("is_built", &self.is_built)
            .field("has_encoder", &self.encoder.is_some())
            .field("has_retriever", &self.retriever.is_some())
            .finish()
    }
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
        // The index_path is actually the base path - the library will add .metadata and .vector extensions
        let index_path = index_path.as_ref().to_path_buf();

        // Initialize rust_mem_vid if not already done
        if let Err(e) = rust_mem_vid::init().await {
            warn!("Failed to initialize rust_mem_vid: {}", e);
        }

        // Create encoder for building videos
        let encoder = rust_mem_vid::MemvidEncoder::new().await
            .map_err(|e| AgentError::memory(format!("Failed to create MemvidEncoder: {}", e)))?;

        // Try to create retriever if files exist
        // Check for the actual files that the library creates: .metadata and .vector
        let metadata_path = index_path.with_extension("metadata");
        let vector_path = index_path.with_extension("vector");

        let retriever = if memory_path.exists() && metadata_path.exists() && vector_path.exists() {
            let memory_path_str = memory_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
            let index_path_str = index_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

            Some(rust_mem_vid::MemvidRetriever::new(memory_path_str, index_path_str).await
                .map_err(|e| AgentError::memory(format!("Failed to create MemvidRetriever: {}", e)))?)
        } else {
            None
        };

        let mut wrapper = Self {
            memory_path,
            index_path,
            encoder: Some(encoder),
            retriever,
            chunks: Vec::new(),
            is_built: false,
        };

        // Load existing memory if it exists
        wrapper.load_existing().await?;

        Ok(wrapper)
    }

    /// Load existing memory if files exist
    async fn load_existing(&mut self) -> Result<()> {
        let metadata_path = self.index_path.with_extension("metadata");
        let vector_path = self.index_path.with_extension("vector");

        if self.memory_path.exists() && metadata_path.exists() && vector_path.exists() {
            info!("Loading existing memory from {:?}", self.memory_path);

            // If we have a retriever, we can get info about the existing memory
            if let Some(ref retriever) = self.retriever {
                // Get info about the existing video
                let video_info = retriever.get_video_info();
                info!("Loaded existing memory with {} frames", video_info.total_frames);
                self.is_built = true;
            } else {
                // Fall back to loading from index file
                self.load_from_index_file().await?;
            }
        } else {
            debug!("No existing memory found, starting fresh");
        }
        Ok(())
    }

    /// Load chunks from the index file (fallback method)
    async fn load_from_index_file(&mut self) -> Result<()> {
        let metadata_path = self.index_path.with_extension("metadata");

        if metadata_path.exists() {
            // Try to load the metadata file to get chunk count
            match std::fs::read_to_string(&metadata_path) {
                Ok(_content) => {
                    // The metadata file contains structured data, not just a simple array
                    // For now, just mark as built since we know the files exist
                    info!("Found existing metadata file, marking as built");
                    self.is_built = true;
                }
                Err(e) => {
                    warn!("Failed to read metadata file: {}", e);
                }
            }
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

        // Use the actual rust-mp4-memory library to build the video
        if let Some(ref mut encoder) = self.encoder {
            // Add all chunks to the encoder
            encoder.add_chunks(self.chunks.clone()).await
                .map_err(|e| AgentError::memory(format!("Failed to add chunks to encoder: {}", e)))?;

            // Build the video
            let memory_path_str = self.memory_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
            let index_path_str = self.index_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

            let _stats = encoder.build_video(memory_path_str, index_path_str).await
                .map_err(|e| AgentError::memory(format!("Failed to build video: {}", e)))?;

            // Wait a moment for files to be fully written and verify they exist
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Check if files exist before creating retriever
            if !self.memory_path.exists() {
                return Err(AgentError::memory(format!("Video file not found after build: {:?}", self.memory_path)));
            }

            let metadata_path = self.index_path.with_extension("metadata");
            let vector_path = self.index_path.with_extension("vector");

            if !metadata_path.exists() {
                return Err(AgentError::memory(format!("Metadata file not found after build: {:?}", metadata_path)));
            }
            if !vector_path.exists() {
                return Err(AgentError::memory(format!("Vector file not found after build: {:?}", vector_path)));
            }

            // Create a new retriever for the built video
            self.retriever = Some(rust_mem_vid::MemvidRetriever::new(memory_path_str, index_path_str).await
                .map_err(|e| AgentError::memory(format!("Failed to create retriever after build: {}", e)))?);

            self.is_built = true;
            info!("Memory video built successfully");
        } else {
            return Err(AgentError::memory("No encoder available for building video".to_string()));
        }

        Ok(())
    }

    /// Search through memory
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Searching memory for: '{}' (limit: {})", query, limit);

        if !self.is_built && !self.memory_path.exists() {
            return Ok(Vec::new());
        }

        // Use the actual rust-mp4-memory library for semantic search
        if let Some(ref retriever) = self.retriever {
            let search_results = retriever.search(query, limit).await
                .map_err(|e| AgentError::memory(format!("Failed to search memory: {}", e)))?;

            // Convert Vec<String> to our SearchResult format
            let results: Vec<SearchResult> = search_results.into_iter().enumerate().map(|(i, content)| SearchResult {
                content,
                score: 1.0 - (i as f32 / limit as f32), // Simple scoring based on order
                chunk_id: i,
                metadata: None,
            }).collect();

            debug!("Found {} search results", results.len());
            Ok(results)
        } else {
            // Fall back to simple text search if no retriever available
            debug!("No retriever available, falling back to simple text search");
            self.simple_text_search(query, limit).await
        }
    }

    /// Simple text search fallback
    async fn simple_text_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // For simple text search, just use the in-memory chunks
        // The new library format is more complex and requires proper parsing
        let chunks = self.chunks.clone();

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

        debug!("Found {} search results (simple text search)", results.len());
        Ok(results)
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> Result<MemvidStats> {
        if let Some(ref retriever) = self.retriever {
            // Get stats from the actual video
            let video_info = retriever.get_video_info();
            let video_size = if self.memory_path.exists() {
                std::fs::metadata(&self.memory_path)?.len()
            } else {
                0
            };

            Ok(MemvidStats {
                total_chunks: video_info.total_frames as usize, // Use total_frames as proxy for chunks
                video_size_bytes: video_size,
                index_size_bytes: if self.index_path.exists() {
                    std::fs::metadata(&self.index_path)?.len()
                } else {
                    0
                },
            })
        } else {
            // Fall back to file system stats
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
    }

    /// Extract text from a specific frame (for debugging)
    pub async fn extract_frame(&mut self, frame_number: usize) -> Result<Option<String>> {
        if let Some(ref mut retriever) = self.retriever {
            // Use the actual rust-mp4-memory library to extract frame
            match retriever.extract_frame(frame_number as u32).await {
                Ok(content) => Ok(Some(content)),
                Err(e) => {
                    debug!("Failed to extract frame {}: {}", frame_number, e);
                    Ok(None)
                }
            }
        } else {
            // Fall back to returning a chunk if it exists
            if frame_number < self.chunks.len() {
                Ok(Some(self.chunks[frame_number].clone()))
            } else {
                Ok(None)
            }
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
