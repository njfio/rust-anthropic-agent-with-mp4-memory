use crate::utils::error::{AgentError, Result};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

/// Streaming response chunk types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamChunk {
    /// Text content chunk
    Text(String),
    /// Tool execution start
    ToolStart {
        tool_name: String,
        parameters: serde_json::Value,
    },
    /// Tool execution result
    ToolResult {
        tool_name: String,
        result: serde_json::Value,
    },
    /// Tool execution error
    ToolError { tool_name: String, error: String },
    /// Thinking/reasoning chunk
    Thinking(String),
    /// Memory operation
    Memory {
        operation: String,
        details: serde_json::Value,
    },
    /// Progress indicator
    Progress {
        current: usize,
        total: usize,
        message: String,
    },
    /// Error chunk
    Error(String),
    /// End of stream marker
    Done,
}

/// Streaming response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub request_id: String,
    pub timestamp: u64,
    pub chunk_index: usize,
    pub total_chunks: Option<usize>,
    pub processing_time_ms: u64,
}

/// Complete streaming response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponse {
    pub metadata: StreamMetadata,
    pub chunk: StreamChunk,
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for the streaming channel
    pub buffer_size: usize,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// Minimum delay between chunks (for rate limiting)
    pub min_chunk_delay: Duration,
    /// Maximum total streaming time
    pub max_stream_duration: Duration,
    /// Enable chunk compression
    pub enable_compression: bool,
    /// Enable progress tracking
    pub enable_progress: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_chunk_size: 1024,
            min_chunk_delay: Duration::from_millis(10),
            max_stream_duration: Duration::from_secs(300), // 5 minutes
            enable_compression: false,
            enable_progress: true,
        }
    }
}

/// Streaming response builder
pub struct StreamBuilder {
    config: StreamConfig,
    request_id: String,
    start_time: Instant,
    chunk_index: usize,
    sender: mpsc::Sender<StreamResponse>,
}

impl StreamBuilder {
    /// Create a new stream builder
    pub fn new(config: StreamConfig, request_id: String) -> (Self, mpsc::Receiver<StreamResponse>) {
        let (sender, receiver) = mpsc::channel(config.buffer_size);

        let builder = Self {
            config,
            request_id,
            start_time: Instant::now(),
            chunk_index: 0,
            sender,
        };

        (builder, receiver)
    }

    /// Send a text chunk
    pub async fn send_text(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        // Split large text into smaller chunks
        let chunks = self.split_text(text);

        for chunk_text in chunks {
            self.send_chunk(StreamChunk::Text(chunk_text)).await?;
        }

        Ok(())
    }

    /// Send a tool start notification
    pub async fn send_tool_start(
        &mut self,
        tool_name: &str,
        parameters: serde_json::Value,
    ) -> Result<()> {
        self.send_chunk(StreamChunk::ToolStart {
            tool_name: tool_name.to_string(),
            parameters,
        })
        .await
    }

    /// Send a tool result
    pub async fn send_tool_result(
        &mut self,
        tool_name: &str,
        result: serde_json::Value,
    ) -> Result<()> {
        self.send_chunk(StreamChunk::ToolResult {
            tool_name: tool_name.to_string(),
            result,
        })
        .await
    }

    /// Send a tool error
    pub async fn send_tool_error(&mut self, tool_name: &str, error: &str) -> Result<()> {
        self.send_chunk(StreamChunk::ToolError {
            tool_name: tool_name.to_string(),
            error: error.to_string(),
        })
        .await
    }

    /// Send thinking/reasoning content
    pub async fn send_thinking(&mut self, content: &str) -> Result<()> {
        if content.is_empty() {
            return Ok(());
        }

        let chunks = self.split_text(content);
        for chunk_text in chunks {
            self.send_chunk(StreamChunk::Thinking(chunk_text)).await?;
        }

        Ok(())
    }

    /// Send memory operation notification
    pub async fn send_memory_operation(
        &mut self,
        operation: &str,
        details: serde_json::Value,
    ) -> Result<()> {
        self.send_chunk(StreamChunk::Memory {
            operation: operation.to_string(),
            details,
        })
        .await
    }

    /// Send progress update
    pub async fn send_progress(
        &mut self,
        current: usize,
        total: usize,
        message: &str,
    ) -> Result<()> {
        if !self.config.enable_progress {
            return Ok(());
        }

        self.send_chunk(StreamChunk::Progress {
            current,
            total,
            message: message.to_string(),
        })
        .await
    }

    /// Send error chunk
    pub async fn send_error(&mut self, error: &str) -> Result<()> {
        self.send_chunk(StreamChunk::Error(error.to_string())).await
    }

    /// Send end of stream marker
    pub async fn send_done(&mut self) -> Result<()> {
        self.send_chunk(StreamChunk::Done).await
    }

    /// Send a raw chunk
    async fn send_chunk(&mut self, chunk: StreamChunk) -> Result<()> {
        // Check if we've exceeded the maximum streaming duration
        if self.start_time.elapsed() > self.config.max_stream_duration {
            return Err(AgentError::tool(
                "streaming",
                "Maximum streaming duration exceeded",
            ));
        }

        // Apply rate limiting
        if self.chunk_index > 0 && !self.config.min_chunk_delay.is_zero() {
            tokio::time::sleep(self.config.min_chunk_delay).await;
        }

        let metadata = StreamMetadata {
            request_id: self.request_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            chunk_index: self.chunk_index,
            total_chunks: None, // We don't know the total in advance
            processing_time_ms: self.start_time.elapsed().as_millis() as u64,
        };

        let response = StreamResponse { metadata, chunk };

        self.sender.send(response).await.map_err(|_| {
            AgentError::tool("streaming", "Failed to send chunk - receiver dropped")
        })?;

        self.chunk_index += 1;
        Ok(())
    }

    /// Split text into appropriately sized chunks
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.len() <= self.config.max_chunk_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for word in text.split_whitespace() {
            if current_chunk.len() + word.len() + 1 > self.config.max_chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamStats {
        StreamStats {
            chunks_sent: self.chunk_index,
            elapsed_time: self.start_time.elapsed(),
            average_chunk_time: if self.chunk_index > 0 {
                self.start_time.elapsed() / self.chunk_index as u32
            } else {
                Duration::ZERO
            },
        }
    }
}

/// Streaming statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub chunks_sent: usize,
    pub elapsed_time: Duration,
    pub average_chunk_time: Duration,
}

/// Stream wrapper for easier consumption
pub struct ResponseStream {
    receiver: mpsc::Receiver<StreamResponse>,
    config: StreamConfig,
}

impl ResponseStream {
    /// Create a new response stream
    pub fn new(receiver: mpsc::Receiver<StreamResponse>, config: StreamConfig) -> Self {
        Self { receiver, config }
    }

    /// Convert to a tokio stream
    pub fn into_stream(self) -> BoxStream<'static, Result<StreamResponse>> {
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(self.receiver).map(Ok))
    }

    /// Collect all chunks into a single response
    pub async fn collect_all(mut self) -> Result<Vec<StreamResponse>> {
        let mut responses = Vec::new();

        while let Some(response) = self.receiver.recv().await {
            let is_done = matches!(response.chunk, StreamChunk::Done);
            responses.push(response);

            if is_done {
                break;
            }
        }

        Ok(responses)
    }

    /// Get only text chunks as a single string
    pub async fn collect_text(mut self) -> Result<String> {
        let mut text = String::new();

        while let Some(response) = self.receiver.recv().await {
            match response.chunk {
                StreamChunk::Text(chunk_text) => {
                    text.push_str(&chunk_text);
                }
                StreamChunk::Done => break,
                _ => {} // Ignore non-text chunks
            }
        }

        Ok(text)
    }

    /// Process stream with a callback function
    pub async fn process_with<F, Fut>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(StreamResponse) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        while let Some(response) = self.receiver.recv().await {
            let is_done = matches!(response.chunk, StreamChunk::Done);
            callback(response).await?;

            if is_done {
                break;
            }
        }

        Ok(())
    }
}

/// Utility functions for streaming
pub mod utils {
    use super::*;

    /// Create a simple text stream
    pub async fn create_text_stream(
        text: &str,
        config: StreamConfig,
        request_id: String,
    ) -> Result<ResponseStream> {
        let (mut builder, receiver) = StreamBuilder::new(config.clone(), request_id);

        // Send text in background
        let text = text.to_string();
        tokio::spawn(async move {
            if let Err(e) = builder.send_text(&text).await {
                let _ = builder.send_error(&e.to_string()).await;
            }
            let _ = builder.send_done().await;
        });

        Ok(ResponseStream::new(receiver, config))
    }

    /// Create a progress stream for long operations
    pub async fn create_progress_stream(
        _total_steps: usize,
        config: StreamConfig,
        request_id: String,
    ) -> Result<(StreamBuilder, ResponseStream)> {
        let (builder, receiver) = StreamBuilder::new(config.clone(), request_id);
        let stream = ResponseStream::new(receiver, config);
        Ok((builder, stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_stream_builder_text_chunks() {
        let config = StreamConfig::default();
        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-123".to_string());

        // Send text in background
        tokio::spawn(async move {
            builder.send_text("Hello world").await.unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect responses
        let mut responses = Vec::new();
        while let Some(response) = receiver.recv().await {
            let is_done = matches!(response.chunk, StreamChunk::Done);
            responses.push(response);
            if is_done {
                break;
            }
        }

        assert_eq!(responses.len(), 2); // Text + Done

        if let StreamChunk::Text(text) = &responses[0].chunk {
            assert_eq!(text, "Hello world");
        } else {
            panic!("Expected text chunk");
        }

        assert!(matches!(responses[1].chunk, StreamChunk::Done));
        assert_eq!(responses[0].metadata.request_id, "test-123");
    }

    #[tokio::test]
    async fn test_stream_builder_large_text_splitting() {
        let mut config = StreamConfig::default();
        config.max_chunk_size = 10; // Small chunks for testing

        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-456".to_string());

        let large_text = "This is a very long text that should be split into multiple chunks";

        // Send text in background
        tokio::spawn(async move {
            builder.send_text(large_text).await.unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect text chunks
        let mut text_chunks = Vec::new();
        while let Some(response) = receiver.recv().await {
            match response.chunk {
                StreamChunk::Text(text) => text_chunks.push(text),
                StreamChunk::Done => break,
                _ => {}
            }
        }

        assert!(
            text_chunks.len() > 1,
            "Text should be split into multiple chunks"
        );

        // Verify all chunks are within size limit
        for chunk in &text_chunks {
            assert!(chunk.len() <= 10, "Chunk size should not exceed limit");
        }

        // Verify concatenated text matches original
        let reconstructed = text_chunks.join(" ");
        assert!(reconstructed.contains("This is a very long text"));
    }

    #[tokio::test]
    async fn test_stream_builder_tool_operations() {
        let config = StreamConfig::default();
        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-789".to_string());

        let params = json!({"query": "test"});
        let result = json!({"status": "success"});

        // Send tool operations in background
        tokio::spawn(async move {
            builder.send_tool_start("search", params).await.unwrap();
            builder.send_tool_result("search", result).await.unwrap();
            builder
                .send_tool_error("search", "Test error")
                .await
                .unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect responses
        let mut responses = Vec::new();
        while let Some(response) = receiver.recv().await {
            let is_done = matches!(response.chunk, StreamChunk::Done);
            responses.push(response);
            if is_done {
                break;
            }
        }

        assert_eq!(responses.len(), 4); // ToolStart + ToolResult + ToolError + Done

        // Verify tool start
        if let StreamChunk::ToolStart {
            tool_name,
            parameters,
        } = &responses[0].chunk
        {
            assert_eq!(tool_name, "search");
            assert_eq!(parameters["query"], "test");
        } else {
            panic!("Expected tool start chunk");
        }

        // Verify tool result
        if let StreamChunk::ToolResult { tool_name, result } = &responses[1].chunk {
            assert_eq!(tool_name, "search");
            assert_eq!(result["status"], "success");
        } else {
            panic!("Expected tool result chunk");
        }

        // Verify tool error
        if let StreamChunk::ToolError { tool_name, error } = &responses[2].chunk {
            assert_eq!(tool_name, "search");
            assert_eq!(error, "Test error");
        } else {
            panic!("Expected tool error chunk");
        }
    }

    #[tokio::test]
    async fn test_stream_builder_progress_tracking() {
        let mut config = StreamConfig::default();
        config.enable_progress = true;

        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-progress".to_string());

        // Send progress updates in background
        tokio::spawn(async move {
            builder.send_progress(1, 5, "Starting").await.unwrap();
            builder.send_progress(3, 5, "Halfway").await.unwrap();
            builder.send_progress(5, 5, "Complete").await.unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect progress responses
        let mut progress_responses = Vec::new();
        while let Some(response) = receiver.recv().await {
            match response.chunk {
                StreamChunk::Progress {
                    current,
                    total,
                    message,
                } => {
                    progress_responses.push((current, total, message));
                }
                StreamChunk::Done => break,
                _ => {}
            }
        }

        assert_eq!(progress_responses.len(), 3);
        assert_eq!(progress_responses[0], (1, 5, "Starting".to_string()));
        assert_eq!(progress_responses[1], (3, 5, "Halfway".to_string()));
        assert_eq!(progress_responses[2], (5, 5, "Complete".to_string()));
    }

    #[tokio::test]
    async fn test_stream_builder_thinking_chunks() {
        let config = StreamConfig::default();
        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-thinking".to_string());

        // Send thinking content in background
        tokio::spawn(async move {
            builder
                .send_thinking("Let me think about this...")
                .await
                .unwrap();
            builder
                .send_thinking("I need to consider multiple factors")
                .await
                .unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect thinking chunks
        let mut thinking_chunks = Vec::new();
        while let Some(response) = receiver.recv().await {
            match response.chunk {
                StreamChunk::Thinking(content) => thinking_chunks.push(content),
                StreamChunk::Done => break,
                _ => {}
            }
        }

        assert_eq!(thinking_chunks.len(), 2);
        assert_eq!(thinking_chunks[0], "Let me think about this...");
        assert_eq!(thinking_chunks[1], "I need to consider multiple factors");
    }

    #[tokio::test]
    async fn test_stream_builder_memory_operations() {
        let config = StreamConfig::default();
        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-memory".to_string());

        let details = json!({"key": "value", "count": 42});

        // Send memory operation in background
        tokio::spawn(async move {
            builder
                .send_memory_operation("store", details)
                .await
                .unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect memory operations
        let mut memory_ops = Vec::new();
        while let Some(response) = receiver.recv().await {
            match response.chunk {
                StreamChunk::Memory { operation, details } => {
                    memory_ops.push((operation, details));
                }
                StreamChunk::Done => break,
                _ => {}
            }
        }

        assert_eq!(memory_ops.len(), 1);
        assert_eq!(memory_ops[0].0, "store");
        assert_eq!(memory_ops[0].1["key"], "value");
        assert_eq!(memory_ops[0].1["count"], 42);
    }

    #[tokio::test]
    async fn test_response_stream_collect_text() {
        let config = StreamConfig::default();
        let (mut builder, receiver) =
            StreamBuilder::new(config.clone(), "test-collect".to_string());

        // Send mixed content in background
        tokio::spawn(async move {
            builder.send_text("Hello ").await.unwrap();
            builder.send_tool_start("test", json!({})).await.unwrap();
            builder.send_text("world").await.unwrap();
            builder.send_thinking("Some thinking").await.unwrap();
            builder.send_text("!").await.unwrap();
            builder.send_done().await.unwrap();
        });

        let stream = ResponseStream::new(receiver, config);
        let text = stream.collect_text().await.unwrap();

        assert_eq!(text, "Hello world!");
    }

    #[tokio::test]
    async fn test_response_stream_collect_all() {
        let config = StreamConfig::default();
        let (mut builder, receiver) = StreamBuilder::new(config.clone(), "test-all".to_string());

        // Send content in background
        tokio::spawn(async move {
            builder.send_text("Test").await.unwrap();
            builder.send_error("Test error").await.unwrap();
            builder.send_done().await.unwrap();
        });

        let stream = ResponseStream::new(receiver, config);
        let responses = stream.collect_all().await.unwrap();

        assert_eq!(responses.len(), 3); // Text + Error + Done
        assert!(matches!(responses[0].chunk, StreamChunk::Text(_)));
        assert!(matches!(responses[1].chunk, StreamChunk::Error(_)));
        assert!(matches!(responses[2].chunk, StreamChunk::Done));
    }

    #[tokio::test]
    async fn test_stream_config_rate_limiting() {
        let mut config = StreamConfig::default();
        config.min_chunk_delay = Duration::from_millis(50);

        let (mut builder, mut receiver) = StreamBuilder::new(config, "test-rate".to_string());

        let start_time = Instant::now();

        // Send multiple chunks in background
        tokio::spawn(async move {
            builder.send_text("Chunk 1").await.unwrap();
            builder.send_text("Chunk 2").await.unwrap();
            builder.send_text("Chunk 3").await.unwrap();
            builder.send_done().await.unwrap();
        });

        // Collect all responses
        let mut count = 0;
        while let Some(_) = receiver.recv().await {
            count += 1;
            if count == 4 {
                // 3 text chunks + done
                break;
            }
        }

        let elapsed = start_time.elapsed();

        // Should take at least 150ms (3 delays of 50ms each)
        assert!(
            elapsed >= Duration::from_millis(100),
            "Rate limiting should add delays"
        );
    }

    #[tokio::test]
    async fn test_stream_builder_stats() {
        let config = StreamConfig::default();
        let (mut builder, _receiver) = StreamBuilder::new(config, "test-stats".to_string());

        // Send some chunks
        builder.send_text("Test 1").await.unwrap();
        builder.send_text("Test 2").await.unwrap();

        let stats = builder.get_stats();
        assert_eq!(stats.chunks_sent, 2);
        assert!(stats.elapsed_time > Duration::ZERO);
        assert!(stats.average_chunk_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_utils_create_text_stream() {
        let config = StreamConfig::default();
        let stream =
            utils::create_text_stream("Hello streaming world", config, "test-utils".to_string())
                .await
                .unwrap();

        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "Hello streaming world");
    }

    #[tokio::test]
    async fn test_stream_timeout_protection() {
        let mut config = StreamConfig::default();
        config.max_stream_duration = Duration::from_millis(100);

        let (mut builder, _receiver) = StreamBuilder::new(config, "test-timeout".to_string());

        // Wait longer than the timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // This should fail due to timeout
        let result = builder.send_text("This should fail").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("duration exceeded"));
    }
}
