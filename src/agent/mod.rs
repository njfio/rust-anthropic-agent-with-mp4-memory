pub mod conversation;
pub mod memory_manager;
pub mod tool_orchestrator;

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::anthropic::{AnthropicClient, ChatMessage, ChatRequest, MessageRole};
use crate::config::AgentConfig;
use crate::memory::MemoryManager;
use crate::utils::error::{AgentError, Result};

pub use conversation::ConversationManager;
pub use tool_orchestrator::ToolOrchestrator;

/// Main agent that orchestrates conversations, tools, and memory
#[derive(Debug)]
pub struct Agent {
    /// Configuration
    config: AgentConfig,
    /// Anthropic API client
    anthropic_client: AnthropicClient,
    /// Memory manager
    memory_manager: Arc<Mutex<MemoryManager>>,
    /// Tool orchestrator
    tool_orchestrator: ToolOrchestrator,
    /// Conversation manager
    conversation_manager: ConversationManager,
    /// Current conversation ID
    current_conversation_id: Option<String>,
}

/// Builder for creating agents with custom configurations
pub struct AgentBuilder {
    config: AgentConfig,
    custom_tools: Vec<Box<dyn crate::tools::Tool>>,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("config", &self.config)
            .field("custom_tool_count", &self.custom_tools.len())
            .finish()
    }
}

impl Agent {
    /// Create a new agent with the given configuration
    pub async fn new(config: AgentConfig) -> Result<Self> {
        config.validate()?;

        info!("Initializing agent with model: {}", config.anthropic.model);

        // Create Anthropic client
        let anthropic_client = AnthropicClient::new(config.anthropic.clone())?;

        // Test API connection
        anthropic_client.test_connection().await?;

        // Create memory manager
        let memory_manager = Arc::new(Mutex::new(
            MemoryManager::new(config.memory.clone()).await?,
        ));

        // Create tool orchestrator
        let mut tool_orchestrator = ToolOrchestrator::new(memory_manager.clone());

        // Register built-in tools based on configuration
        tool_orchestrator.register_builtin_tools(&config).await?;

        // Create conversation manager
        let conversation_manager = ConversationManager::new(memory_manager.clone());

        Ok(Self {
            config,
            anthropic_client,
            memory_manager,
            tool_orchestrator,
            conversation_manager,
            current_conversation_id: None,
        })
    }

    /// Start a new conversation
    pub async fn start_conversation(&mut self, title: Option<String>) -> Result<String> {
        let conversation_id = self.conversation_manager.start_conversation(title).await?;
        self.current_conversation_id = Some(conversation_id.clone());

        info!("Started new conversation: {}", conversation_id);
        Ok(conversation_id)
    }

    /// Send a message and get a response
    pub async fn chat<S: Into<String>>(&mut self, message: S) -> Result<String> {
        let message = message.into();
        debug!("Processing chat message: {} chars", message.len());

        // Ensure we have an active conversation
        if self.current_conversation_id.is_none() {
            self.start_conversation(None).await?;
        }

        // Add user message to conversation
        let user_message = ChatMessage::user(message);
        self.conversation_manager.add_message(user_message.clone()).await?;

        // Get conversation history for context
        let history = self.conversation_manager.get_recent_history(
            self.config.agent.max_history_length
        ).await?;

        // Create chat request
        let mut request = ChatRequest {
            model: self.config.anthropic.model.clone(),
            max_tokens: self.config.anthropic.max_tokens,
            messages: history.iter().map(|msg| msg.to_api_message()).collect(),
            system: self.config.agent.system_prompt.clone(),
            tools: Some(self.tool_orchestrator.get_tool_definitions()),
            tool_choice: None,
            temperature: Some(self.config.anthropic.temperature),
            stream: Some(self.config.agent.enable_streaming),
        };

        // Send request to Anthropic
        let mut response = self.anthropic_client.chat(request.clone()).await?;

        // Process tool calls if any
        let mut tool_iterations = 0;
        const MAX_TOOL_ITERATIONS: usize = 10;

        while response.content.iter().any(|block| {
            matches!(block, crate::anthropic::models::ContentBlock::ToolUse { .. })
        }) && tool_iterations < MAX_TOOL_ITERATIONS {
            tool_iterations += 1;
            debug!("Processing tool calls (iteration {})", tool_iterations);

            // Add assistant message with tool calls
            let assistant_message = ChatMessage {
                role: MessageRole::Assistant,
                content: response.content.clone(),
                id: Some(response.id.clone()),
                timestamp: Some(chrono::Utc::now()),
            };
            self.conversation_manager.add_message(assistant_message).await?;

            // Execute tools and collect results
            let tool_results = self.tool_orchestrator.execute_tools(&response.content).await?;

            // Only create user message with tool results if there are actual results
            if !tool_results.is_empty() {
                let tool_result_message = ChatMessage {
                    role: MessageRole::User,
                    content: tool_results,
                    id: Some(Uuid::new_v4().to_string()),
                    timestamp: Some(chrono::Utc::now()),
                };
                self.conversation_manager.add_message(tool_result_message).await?;
            }

            // Get updated history and make another request
            let updated_history = self.conversation_manager.get_recent_history(
                self.config.agent.max_history_length
            ).await?;

            request.messages = updated_history.iter().map(|msg| msg.to_api_message()).collect();
            response = self.anthropic_client.chat(request.clone()).await?;
        }

        if tool_iterations >= MAX_TOOL_ITERATIONS {
            warn!("Maximum tool iterations reached, stopping tool execution");
        }

        // Add final assistant response
        let final_message = ChatMessage {
            role: MessageRole::Assistant,
            content: response.content.clone(),
            id: Some(response.id),
            timestamp: Some(chrono::Utc::now()),
        };
        self.conversation_manager.add_message(final_message.clone()).await?;

        // Extract text response
        let response_text = final_message.get_text();
        
        info!("Chat response generated: {} chars", response_text.len());
        Ok(response_text)
    }

    /// Get conversation history
    pub async fn get_conversation_history(&self) -> Result<Vec<ChatMessage>> {
        self.conversation_manager.get_recent_history(usize::MAX).await
    }

    /// Search memory
    pub async fn search_memory<S: Into<String>>(&self, query: S, limit: usize) -> Result<Vec<crate::memory::SearchResult>> {
        let memory_manager = self.memory_manager.lock().await;
        memory_manager.search_raw(&query.into(), limit).await
    }

    /// Save information to memory
    pub async fn save_to_memory<S: Into<String>>(&mut self, content: S, entry_type: crate::memory::MemoryEntryType) -> Result<()> {
        let entry = crate::memory::MemoryEntry::new(content, entry_type);
        let mut memory_manager = self.memory_manager.lock().await;
        memory_manager.save_memory(entry).await
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> Result<crate::memory::MemoryStats> {
        let memory_manager = self.memory_manager.lock().await;
        memory_manager.get_stats().await
    }

    /// Register a custom tool
    pub fn register_tool<T: crate::tools::Tool + 'static>(&mut self, tool: T) {
        self.tool_orchestrator.register_tool(tool);
    }

    /// Get available tools
    pub fn get_available_tools(&self) -> Vec<String> {
        self.tool_orchestrator.get_tool_names()
    }

    /// Execute a specific tool directly
    pub async fn execute_tool(&self, tool_name: &str, input: serde_json::Value) -> Result<crate::tools::ToolResult> {
        self.tool_orchestrator.execute_tool_direct(tool_name, input).await
    }

    /// Update configuration
    pub async fn update_config(&mut self, config: AgentConfig) -> Result<()> {
        config.validate()?;
        
        // Update Anthropic client if needed
        if config.anthropic != self.config.anthropic {
            self.anthropic_client.update_config(config.anthropic.clone())?;
        }

        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get current conversation ID
    pub fn current_conversation_id(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Switch to a different conversation
    pub async fn switch_conversation(&mut self, conversation_id: String) -> Result<()> {
        // Verify the conversation exists
        let memory_manager = self.memory_manager.lock().await;
        if memory_manager.get_conversation(&conversation_id).await?.is_some() {
            self.current_conversation_id = Some(conversation_id);
            Ok(())
        } else {
            Err(AgentError::invalid_input("Conversation not found"))
        }
    }

    /// Build and finalize memory
    pub async fn finalize_memory(&mut self) -> Result<()> {
        let mut memory_manager = self.memory_manager.lock().await;
        memory_manager.build_memory().await
    }
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            custom_tools: Vec::new(),
        }
    }

    /// Set the configuration
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the Anthropic API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.anthropic.api_key = api_key.into();
        self
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.anthropic.model = model.into();
        self
    }

    /// Set the memory path
    pub fn with_memory_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        let path = path.into();
        self.config.memory.memory_path = path.clone();
        self.config.memory.index_path = path.with_extension("json");
        self
    }

    /// Add a custom tool
    pub fn with_tool<T: crate::tools::Tool + 'static>(mut self, tool: T) -> Self {
        self.custom_tools.push(Box::new(tool));
        self
    }

    /// Build the agent
    pub async fn build(self) -> Result<Agent> {
        let mut agent = Agent::new(self.config).await?;
        
        // Register custom tools
        for tool in self.custom_tools {
            agent.tool_orchestrator.register_boxed_tool(tool);
        }

        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
