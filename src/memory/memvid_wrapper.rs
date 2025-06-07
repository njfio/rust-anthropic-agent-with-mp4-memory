use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::memory::SearchResult;
use crate::utils::error::{AgentError, Result};

// Import Phase 2 types from the updated memvid library
use rust_mem_vid::multi_memory::{AggregatedSearchResult, SearchMetadata};
use rust_mem_vid::retriever::RetrievalResult;

/// Enhanced wrapper around the rust-mp4-memory library with Phase 2 performance features
pub struct MemvidWrapper {
    memory_path: PathBuf,
    index_path: PathBuf,
    encoder: Option<rust_mem_vid::MemvidEncoder>,
    retriever: Option<rust_mem_vid::MemvidRetriever>,
    chunks: Vec<String>,
    is_built: bool,
    // New Phase 2 performance features
    multi_memory_engine: Option<rust_mem_vid::MultiMemoryEngine>,
    temporal_analysis_engine: Option<rust_mem_vid::TemporalAnalysisEngine>,
    knowledge_graph_builder: Option<rust_mem_vid::KnowledgeGraphBuilder>,
    content_synthesizer: Option<rust_mem_vid::ContentSynthesizer>,
    analytics_dashboard: Option<rust_mem_vid::AnalyticsDashboard>,
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
            .field("has_multi_memory", &self.multi_memory_engine.is_some())
            .field("has_temporal_analysis", &self.temporal_analysis_engine.is_some())
            .field("has_knowledge_graph", &self.knowledge_graph_builder.is_some())
            .field("has_content_synthesizer", &self.content_synthesizer.is_some())
            .field("has_analytics_dashboard", &self.analytics_dashboard.is_some())
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
            // Initialize Phase 2 performance features as None initially
            multi_memory_engine: None,
            temporal_analysis_engine: None,
            knowledge_graph_builder: None,
            content_synthesizer: None,
            analytics_dashboard: None,
        };

        // Load existing memory if it exists
        wrapper.load_existing().await?;

        Ok(wrapper)
    }

    /// Load existing memory if files exist using new incremental features
    async fn load_existing(&mut self) -> Result<()> {
        let metadata_path = self.index_path.with_extension("metadata");
        let vector_path = self.index_path.with_extension("vector");

        if self.memory_path.exists() && metadata_path.exists() && vector_path.exists() {
            info!("Loading existing memory from {:?} using incremental features", self.memory_path);

            let memory_path_str = self.memory_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
            let index_path_str = self.index_path.to_str()
                .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

            // Load existing video into encoder for true incremental building
            match rust_mem_vid::MemvidEncoder::load_existing(memory_path_str, index_path_str).await {
                Ok(encoder) => {
                    info!("Successfully loaded existing memory with {} chunks for incremental building",
                          encoder.chunks().len());

                    // Extract existing chunks for in-memory operations
                    self.chunks = encoder.chunks().iter()
                        .map(|chunk| chunk.content.clone())
                        .collect();

                    // Replace encoder with loaded one
                    self.encoder = Some(encoder);

                    // Create retriever for searching
                    match rust_mem_vid::MemvidRetriever::new(memory_path_str, index_path_str).await {
                        Ok(retriever) => {
                            self.retriever = Some(retriever);
                            self.is_built = true;
                            info!("Incremental memory system ready with {} existing chunks", self.chunks.len());
                        }
                        Err(e) => {
                            warn!("Failed to create retriever: {}, search will be limited", e);
                            self.is_built = true; // Still mark as built since we have the encoder
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to load existing memory for incremental building: {}, falling back to basic loading", e);
                    self.load_from_index_file().await?;
                }
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

    /// Build the memory video using incremental append (true incremental building)
    pub async fn build_video(&mut self) -> Result<()> {
        if self.chunks.is_empty() {
            warn!("No chunks to build into video");
            return Ok(());
        }

        let memory_path_str = self.memory_path.to_str()
            .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
        let index_path_str = self.index_path.to_str()
            .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

        if let Some(ref mut encoder) = self.encoder {
            // Check if this is an incremental update or full build
            let existing_chunk_count = encoder.chunks().len();
            let new_chunk_count = self.chunks.len();

            if existing_chunk_count > 0 && new_chunk_count > existing_chunk_count {
                // Incremental append: only add new chunks
                let new_chunks: Vec<String> = self.chunks[existing_chunk_count..].to_vec();
                info!("Performing incremental append: {} new chunks to existing {} chunks",
                      new_chunks.len(), existing_chunk_count);

                // Convert to TextChunk format for append_to_video
                let text_chunks: Vec<rust_mem_vid::text::TextChunk> = new_chunks.into_iter().enumerate().map(|(i, content)| {
                    rust_mem_vid::text::TextChunk {
                        content,
                        metadata: rust_mem_vid::text::ChunkMetadata {
                            id: existing_chunk_count + i,
                            source: Some("agent_memory".to_string()),
                            page: None,
                            char_offset: 0,
                            length: 0, // Will be calculated by the library
                            frame: (existing_chunk_count + i) as u32,
                            extra: std::collections::HashMap::new(),
                        },
                    }
                }).collect();

                let _stats = encoder.append_to_video(memory_path_str, index_path_str, text_chunks).await
                    .map_err(|e| AgentError::memory(format!("Failed to append to video: {}", e)))?;

                info!("Incremental append completed successfully");
            } else {
                // Full build: add all chunks and build from scratch
                info!("Performing full build with {} chunks", self.chunks.len());

                encoder.add_chunks(self.chunks.clone()).await
                    .map_err(|e| AgentError::memory(format!("Failed to add chunks to encoder: {}", e)))?;

                let _stats = encoder.build_video(memory_path_str, index_path_str).await
                    .map_err(|e| AgentError::memory(format!("Failed to build video: {}", e)))?;

                info!("Full build completed successfully");
            }

            // Wait a moment for files to be fully written and verify they exist
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Verify files exist
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

            // Create/update retriever for the built video
            self.retriever = Some(rust_mem_vid::MemvidRetriever::new(memory_path_str, index_path_str).await
                .map_err(|e| AgentError::memory(format!("Failed to create retriever after build: {}", e)))?);

            self.is_built = true;
            info!("Memory video built successfully with incremental features");
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

    /// Compare this memory with another memory video using the new diff engine
    pub async fn compare_with_memory(&self, other_video: &str, other_index: &str) -> Result<serde_json::Value> {
        let memory_path_str = self.memory_path.to_str()
            .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
        let index_path_str = self.index_path.to_str()
            .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

        if !self.memory_path.exists() {
            return Err(AgentError::memory("Current memory video does not exist".to_string()));
        }

        info!("Comparing current memory with: {}", other_video);

        // Create memory diff engine
        let config = rust_mem_vid::Config::default();
        let diff_engine = rust_mem_vid::MemoryDiffEngine::new(config)
            .with_similarity_threshold(0.8)
            .with_semantic_analysis(true);

        // Perform the comparison
        match diff_engine.compare_memories(memory_path_str, index_path_str, other_video, other_index).await {
            Ok(diff) => {
                info!("Memory comparison completed: {} added, {} removed, {} modified",
                      diff.summary.added_count, diff.summary.removed_count, diff.summary.modified_count);

                Ok(serde_json::json!({
                    "comparison_type": "memory_diff",
                    "status": "success",
                    "source": "memvid_memory_diff_engine",
                    "diff_summary": {
                        "old_memory": diff.old_memory,
                        "new_memory": diff.new_memory,
                        "timestamp": diff.timestamp,
                        "total_old_chunks": diff.summary.total_old_chunks,
                        "total_new_chunks": diff.summary.total_new_chunks,
                        "added_count": diff.summary.added_count,
                        "removed_count": diff.summary.removed_count,
                        "modified_count": diff.summary.modified_count,
                        "unchanged_count": diff.summary.unchanged_count,
                        "similarity_score": diff.summary.similarity_score,
                        "content_growth_ratio": diff.summary.content_growth_ratio
                    },
                    "changes": {
                        "added_chunks": diff.added_chunks.len(),
                        "removed_chunks": diff.removed_chunks.len(),
                        "modified_chunks": diff.modified_chunks.len(),
                        "semantic_changes": diff.semantic_changes.len()
                    },
                    "detailed_diff": {
                        "added_chunks": diff.added_chunks.iter().take(5).map(|chunk| {
                            serde_json::json!({
                                "chunk_id": chunk.chunk_id,
                                "content_preview": if chunk.content.len() > 100 {
                                    format!("{}...", &chunk.content[..100])
                                } else {
                                    chunk.content.clone()
                                },
                                "source": chunk.source,
                                "frame_number": chunk.frame_number
                            })
                        }).collect::<Vec<_>>(),
                        "removed_chunks": diff.removed_chunks.iter().take(5).map(|chunk| {
                            serde_json::json!({
                                "chunk_id": chunk.chunk_id,
                                "content_preview": if chunk.content.len() > 100 {
                                    format!("{}...", &chunk.content[..100])
                                } else {
                                    chunk.content.clone()
                                },
                                "source": chunk.source,
                                "frame_number": chunk.frame_number
                            })
                        }).collect::<Vec<_>>(),
                        "modified_chunks": diff.modified_chunks.iter().take(5).map(|modification| {
                            serde_json::json!({
                                "old_chunk_id": modification.old_chunk.chunk_id,
                                "new_chunk_id": modification.new_chunk.chunk_id,
                                "similarity_score": modification.similarity_score,
                                "change_type": modification.change_type,
                                "text_diff": {
                                    "added_words": modification.text_diff.added_text.len(),
                                    "removed_words": modification.text_diff.removed_text.len(),
                                    "common_words": modification.text_diff.common_text.len(),
                                    "edit_distance": modification.text_diff.edit_distance
                                }
                            })
                        }).collect::<Vec<_>>()
                    }
                }))
            }
            Err(e) => {
                warn!("Memory comparison failed: {}", e);
                Err(AgentError::memory(format!("Failed to compare memories: {}", e)))
            }
        }
    }

    // ========================================
    // Phase 2 Performance Enhancement Methods
    // ========================================

    /// Initialize Phase 2 performance features with the latest memvid library
    pub async fn initialize_phase2_features(&mut self) -> Result<()> {
        info!("Initializing Phase 2 performance features with latest memvid library");

        // Create a default config for the Phase 2 features
        let config = rust_mem_vid::Config::default();

        // Initialize Multi-Memory Engine for cross-memory operations
        if self.multi_memory_engine.is_none() {
            let engine = rust_mem_vid::MultiMemoryEngine::new(config.clone());
            self.multi_memory_engine = Some(engine);
            debug!("Multi-memory engine initialized with latest performance improvements");
        }

        // Initialize Temporal Analysis Engine for memory evolution tracking
        if self.temporal_analysis_engine.is_none() {
            let engine = rust_mem_vid::TemporalAnalysisEngine::new(config.clone());
            self.temporal_analysis_engine = Some(engine);
            debug!("Temporal analysis engine initialized with enhanced capabilities");
        }

        // Initialize Knowledge Graph Builder for concept relationships
        if self.knowledge_graph_builder.is_none() {
            let builder = rust_mem_vid::KnowledgeGraphBuilder::new(config.clone());
            self.knowledge_graph_builder = Some(builder);
            debug!("Knowledge graph builder initialized with advanced AI features");
        }

        // Initialize Content Synthesizer for AI-powered content generation
        if self.content_synthesizer.is_none() {
            let synthesizer = rust_mem_vid::ContentSynthesizer::new(config.clone());
            self.content_synthesizer = Some(synthesizer);
            debug!("Content synthesizer initialized with AI intelligence");
        }

        // Initialize Analytics Dashboard for comprehensive metrics
        if self.analytics_dashboard.is_none() {
            let dashboard = rust_mem_vid::AnalyticsDashboard::new(config);
            self.analytics_dashboard = Some(dashboard);
            debug!("Analytics dashboard initialized with comprehensive metrics");
        }

        info!("Phase 2 performance features initialization completed with latest memvid library");
        Ok(())
    }

    /// Perform multi-memory search across different memory instances
    pub async fn multi_memory_search(&mut self, query: &str, limit: usize) -> Result<rust_mem_vid::MultiMemorySearchResult> {
        if let Some(ref mut engine) = self.multi_memory_engine {
            debug!("Performing multi-memory search for: '{}'", query);

            // Add current memory to the search if it exists
            if let Some(ref _retriever) = self.retriever {
                let memory_path_str = self.memory_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
                let index_path_str = self.index_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

                // Register current memory with the multi-memory engine
                // Note: API requires memory_id, tags, and description parameters
                if let Err(e) = engine.add_memory(
                    memory_path_str,
                    index_path_str,
                    "current_memory",
                    vec!["current".to_string()],
                    Some("Current agent memory".to_string())
                ).await {
                    warn!("Failed to add current memory to multi-memory engine: {}", e);
                }
            }

            // Perform the search with required parameters
            let results = engine.search_all(query, limit, true, false).await
                .map_err(|e| AgentError::memory(format!("Multi-memory search failed: {}", e)))?;

            debug!("Multi-memory search found {} total results", results.total_results);
            Ok(results)
        } else {
            // Fall back to regular search if multi-memory engine not available
            debug!("Multi-memory engine not available, falling back to regular search");
            let regular_results = self.search(query, limit).await?;

            // Create a simplified fallback MultiMemorySearchResult
            let mut results_by_memory = std::collections::HashMap::new();

            // Create simplified retrieval results
            let retrieval_results: Vec<RetrievalResult> = regular_results.iter().map(|r| {
                RetrievalResult {
                    chunk_id: r.chunk_id,
                    similarity: r.score,
                    text: r.content.clone(),
                    metadata: rust_mem_vid::text::ChunkMetadata {
                        id: r.chunk_id,
                        source: Some("current_memory".to_string()),
                        page: None,
                        char_offset: 0,
                        length: r.content.len(),
                        frame: r.chunk_id as u32,
                        extra: std::collections::HashMap::new(),
                    },
                    frame_number: r.chunk_id as u32,
                }
            }).collect();

            results_by_memory.insert("current".to_string(), retrieval_results);

            // Create simplified aggregated results
            let aggregated_results: Vec<AggregatedSearchResult> = regular_results.iter().map(|result| {
                AggregatedSearchResult {
                    text: result.content.clone(),
                    similarity: result.score as f64,
                    source_memory: "current".to_string(),
                    chunk_id: result.chunk_id,
                    frame_number: result.chunk_id as u32,
                    related_results: Vec::new(),
                    temporal_context: None,
                }
            }).collect();

            let multi_result = rust_mem_vid::MultiMemorySearchResult {
                query: query.to_string(),
                total_results: regular_results.len(),
                results_by_memory,
                aggregated_results,
                cross_memory_correlations: Vec::new(),
                search_metadata: SearchMetadata {
                    search_time_ms: 50,
                    memories_searched: 1,
                    total_chunks_searched: regular_results.len(),
                    correlation_analysis_enabled: false,
                    temporal_analysis_enabled: false,
                },
            };

            Ok(multi_result)
        }
    }

    /// Generate temporal analysis of memory evolution using the latest memvid library
    pub async fn temporal_analysis(&self, days_back: u32) -> Result<serde_json::Value> {
        debug!("Performing temporal analysis for {} days back using latest memvid library", days_back);

        if let Some(ref engine) = self.temporal_analysis_engine {
            // Try to use the actual temporal analysis engine
            if let Some(ref _retriever) = self.retriever {
                let memory_path_str = self.memory_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;
                let index_path_str = self.index_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

                // Create a snapshot for current memory state
                match engine.create_snapshot(memory_path_str, index_path_str, None, Vec::new()).await {
                    Ok(snapshot) => {
                        debug!("Temporal analysis snapshot created successfully");

                        return Ok(serde_json::json!({
                            "analysis_type": "temporal_analysis",
                            "days_back": days_back,
                            "status": "success",
                            "source": "memvid_temporal_analysis_engine",
                            "snapshot_analysis": {
                                "timestamp": snapshot.timestamp,
                                "video_path": snapshot.video_path,
                                "index_path": snapshot.index_path,
                                "total_chunks": snapshot.metadata.total_chunks,
                                "total_characters": snapshot.metadata.total_characters,
                                "unique_sources": snapshot.metadata.unique_sources,
                                "file_size_bytes": snapshot.metadata.file_size_bytes,
                                "creation_duration_seconds": snapshot.metadata.creation_duration_seconds,
                                "content_hash": snapshot.metadata.content_hash,
                                "tags": snapshot.tags,
                                "description": snapshot.description,
                                "message": "Single snapshot created - timeline analysis requires multiple snapshots over time"
                            }
                        }));
                    }
                    Err(e) => {
                        warn!("Temporal analysis engine failed: {}, falling back to basic stats", e);
                    }
                }
            }
        }

        // Fallback to basic stats if temporal analysis engine is not available
        let stats = self.get_stats().await?;

        Ok(serde_json::json!({
            "analysis_type": "temporal_analysis",
            "days_back": days_back,
            "status": "fallback_implementation",
            "message": "Using basic stats - temporal analysis engine not available",
            "basic_stats": {
                "total_chunks": stats.total_chunks,
                "video_size_bytes": stats.video_size_bytes,
                "index_size_bytes": stats.index_size_bytes,
                "estimated_growth": "Unable to calculate without temporal engine"
            }
        }))
    }

    /// Build knowledge graph from memory content using the latest memvid library
    pub async fn build_knowledge_graph(&self) -> Result<serde_json::Value> {
        debug!("Building knowledge graph from memory content using latest memvid library");

        if let Some(ref builder) = self.knowledge_graph_builder {
            // Try to use the actual knowledge graph builder
            if let Some(ref _retriever) = self.retriever {
                let memory_path_str = self.memory_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid memory path".to_string()))?;

                // The API expects (memory_path, index_path) tuples
                let index_path_str = self.index_path.to_str()
                    .ok_or_else(|| AgentError::memory("Invalid index path".to_string()))?;

                match builder.build_from_memories(&[(memory_path_str.to_string(), index_path_str.to_string())]).await {
                    Ok(graph) => {
                        debug!("Knowledge graph built with {} nodes and {} relationships",
                               graph.nodes.len(), graph.relationships.len());

                        return Ok(serde_json::json!({
                            "analysis_type": "knowledge_graph",
                            "status": "success",
                            "source": "memvid_knowledge_graph_builder",
                            "knowledge_graph": {
                                "nodes": graph.nodes.iter().map(|(id, node)| {
                                    serde_json::json!({
                                        "id": id,
                                        "name": node.name,
                                        "concept_type": node.concept_type,
                                        "importance_score": node.importance_score,
                                        "frequency": node.frequency,
                                        "related_chunks": node.related_chunks,
                                        "has_embedding": node.embedding.is_some(),
                                        "metadata": node.metadata
                                    })
                                }).collect::<Vec<_>>(),
                                "relationships": graph.relationships.iter().map(|(id, rel)| {
                                    serde_json::json!({
                                        "id": id,
                                        "source_concept": rel.source_concept,
                                        "target_concept": rel.target_concept,
                                        "relationship_type": rel.relationship_type,
                                        "strength": rel.strength,
                                        "evidence_chunks": rel.evidence_chunks,
                                        "confidence": rel.confidence,
                                        "temporal_pattern": rel.temporal_pattern
                                    })
                                }).collect::<Vec<_>>(),
                                "metadata": {
                                    "total_nodes": graph.nodes.len(),
                                    "total_relationships": graph.relationships.len(),
                                    "created_at": graph.metadata.created_at,
                                    "last_updated": graph.metadata.last_updated,
                                    "source_memories": graph.metadata.source_memories,
                                    "generation_algorithm": graph.metadata.generation_algorithm,
                                    "confidence_threshold": graph.metadata.confidence_threshold,
                                    "communities": graph.communities.len(),
                                    "temporal_snapshots": graph.temporal_evolution.len()
                                }
                            }
                        }));
                    }
                    Err(e) => {
                        warn!("Knowledge graph builder failed: {}, falling back to basic stats", e);
                    }
                }
            }
        }

        // Fallback to basic stats if knowledge graph builder is not available
        let stats = self.get_stats().await?;

        Ok(serde_json::json!({
            "analysis_type": "knowledge_graph",
            "status": "fallback_implementation",
            "message": "Using basic stats - knowledge graph builder not available",
            "basic_stats": {
                "total_chunks": stats.total_chunks,
                "estimated_concepts": stats.total_chunks / 10,
                "estimated_relationships": stats.total_chunks / 20
            }
        }))
    }

    /// Synthesize content using AI (placeholder for Phase 2)
    pub async fn synthesize_content(&self, synthesis_type: &str, query: Option<&str>) -> Result<serde_json::Value> {
        debug!("Synthesizing content with type: {} (placeholder implementation)", synthesis_type);

        // For now, return a placeholder response until the API stabilizes
        let stats = self.get_stats().await?;

        Ok(serde_json::json!({
            "analysis_type": "content_synthesis",
            "synthesis_type": synthesis_type,
            "query": query,
            "status": "placeholder_implementation",
            "message": "Phase 2 content synthesis features are being integrated",
            "basic_stats": {
                "source_chunks": stats.total_chunks,
                "estimated_output_length": stats.total_chunks * 50
            }
        }))
    }

    /// Generate analytics dashboard data (placeholder for Phase 2)
    pub async fn generate_analytics_dashboard(&self) -> Result<serde_json::Value> {
        debug!("Generating analytics dashboard (placeholder implementation)");

        // For now, return a placeholder response until the API stabilizes
        let stats = self.get_stats().await?;

        Ok(serde_json::json!({
            "analysis_type": "analytics_dashboard",
            "status": "placeholder_implementation",
            "message": "Phase 2 analytics dashboard features are being integrated",
            "basic_metrics": {
                "total_chunks": stats.total_chunks,
                "video_size_bytes": stats.video_size_bytes,
                "index_size_bytes": stats.index_size_bytes,
                "memory_efficiency": if stats.video_size_bytes > 0 {
                    stats.total_chunks as f64 / (stats.video_size_bytes as f64 / 1024.0)
                } else {
                    0.0
                }
            }
        }))
    }

    /// Check if Phase 2 features are available
    pub fn has_phase2_features(&self) -> bool {
        self.multi_memory_engine.is_some() ||
        self.temporal_analysis_engine.is_some() ||
        self.knowledge_graph_builder.is_some() ||
        self.content_synthesizer.is_some() ||
        self.analytics_dashboard.is_some()
    }

    /// Get Phase 2 features status
    pub fn get_phase2_status(&self) -> std::collections::HashMap<String, bool> {
        let mut status = std::collections::HashMap::new();
        status.insert("multi_memory_engine".to_string(), self.multi_memory_engine.is_some());
        status.insert("temporal_analysis_engine".to_string(), self.temporal_analysis_engine.is_some());
        status.insert("knowledge_graph_builder".to_string(), self.knowledge_graph_builder.is_some());
        status.insert("content_synthesizer".to_string(), self.content_synthesizer.is_some());
        status.insert("analytics_dashboard".to_string(), self.analytics_dashboard.is_some());
        status
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
