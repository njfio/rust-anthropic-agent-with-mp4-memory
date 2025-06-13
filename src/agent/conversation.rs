use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::anthropic::models::ChatMessage;
use crate::memory::MemoryManager;
use crate::utils::error::Result;

/// Manages conversations and their persistence
#[derive(Debug)]
pub struct ConversationManager {
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl ConversationManager {
    /// Create a new conversation manager
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self { memory_manager }
    }

    /// Start a new conversation
    pub async fn start_conversation(&self, title: Option<String>) -> Result<String> {
        let mut memory_manager = self.memory_manager.lock().await;
        let conversation_id = memory_manager.start_conversation(title);

        info!("Started conversation: {}", conversation_id);
        Ok(conversation_id)
    }

    /// Add a message to the current conversation
    pub async fn add_message(&self, message: ChatMessage) -> Result<()> {
        let mut memory_manager = self.memory_manager.lock().await;
        memory_manager.add_message(message).await?;
        Ok(())
    }

    /// Get recent conversation history
    pub async fn get_recent_history(&self, _limit: usize) -> Result<Vec<ChatMessage>> {
        let memory_manager = self.memory_manager.lock().await;

        // Get current conversation and return its messages
        if let Some(conversation) = memory_manager.get_current_conversation() {
            debug!("Retrieved {} messages from conversation history", conversation.messages.len());
            Ok(conversation.messages.clone())
        } else {
            debug!("No current conversation found");
            Ok(Vec::new())
        }
    }

    /// Get a specific conversation
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<crate::memory::Conversation>> {
        let memory_manager = self.memory_manager.lock().await;
        memory_manager.get_conversation(conversation_id).await
    }

    /// Search conversations
    pub async fn search_conversations(&self, query: &str, limit: usize) -> Result<Vec<crate::memory::Conversation>> {
        let memory_manager = self.memory_manager.lock().await;
        memory_manager.search_conversations(query, limit).await
    }

    /// Get current conversation ID
    pub async fn current_conversation_id(&self) -> Option<String> {
        let memory_manager = self.memory_manager.lock().await;
        memory_manager.get_current_conversation_id().map(|s| s.to_string())
    }

    /// Set current conversation ID (not implemented for simple memory)
    pub async fn set_current_conversation_id(&self, _conversation_id: Option<String>) {
        // Not implemented for simple memory system
        // Current conversation is managed internally
    }
}
