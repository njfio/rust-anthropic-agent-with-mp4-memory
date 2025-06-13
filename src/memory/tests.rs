#[cfg(test)]
mod tests {
    use crate::memory::MemoryManager;
    use crate::config::MemoryConfig;
    use crate::anthropic::models::ChatMessage;
    use tempfile::TempDir;
    use std::collections::HashMap;
    use std::time::Duration;

    fn create_test_memory_config() -> MemoryConfig {
        let temp_dir = TempDir::new().unwrap();
        MemoryConfig {
            memory_path: temp_dir.path().join("test_memory.json"),
            index_path: temp_dir.path().join("test_memory"),
            auto_save: true,
            max_conversations: 100,
            enable_search: true,
            search_limit: 10,
        }
    }

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await;
        assert!(memory_manager.is_ok());
    }

    #[tokio::test]
    async fn test_conversation_management() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Test starting a new conversation
        let conversation_id = memory_manager.start_conversation(Some("Test Conversation".to_string()));
        assert!(!conversation_id.is_empty());

        // Test adding messages to conversation
        let user_message = ChatMessage::user("Hello, how are you?");
        let assistant_message = ChatMessage::assistant("I'm doing well, thank you!");

        memory_manager.add_message(user_message.clone()).await.unwrap();
        memory_manager.add_message(assistant_message.clone()).await.unwrap();

        // Test getting current conversation
        let current_conversation = memory_manager.get_current_conversation().unwrap();
        assert_eq!(current_conversation.messages.len(), 2);
        assert_eq!(current_conversation.messages[0].get_text(), "Hello, how are you?");
        assert_eq!(current_conversation.messages[1].get_text(), "I'm doing well, thank you!");
    }

    #[tokio::test]
    async fn test_memory_entry_storage_and_retrieval() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Test storing a memory entry
        let entry_content = "This is an important fact to remember";
        let entry_type = "fact";
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "testing".to_string());
        metadata.insert("priority".to_string(), "high".to_string());

        let entry_id = memory_manager.save_memory(
            entry_content.to_string(),
            entry_type.to_string(),
            metadata.clone()
        ).await.unwrap();

        assert!(!entry_id.is_empty());

        // Test searching for the memory entry
        let search_results = memory_manager.search_memory("important fact", 5).await.unwrap();
        assert!(!search_results.is_empty());

        // The search should find our entry
        let found = search_results.iter().any(|result|
            result.content.contains("important fact")
        );
        assert!(found);
    }

    #[tokio::test]
    async fn test_memory_search_functionality() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Store multiple memory entries with different content
        let entries = vec![
            ("Rust is a systems programming language", "fact"),
            ("Python is great for data science", "fact"),
            ("JavaScript runs in browsers", "fact"),
            ("Machine learning requires large datasets", "insight"),
        ];

        for (content, entry_type) in entries {
            memory_manager.save_memory(
                content.to_string(),
                entry_type.to_string(),
                HashMap::new()
            ).await.unwrap();
        }

        // Test searching for programming languages
        let results = memory_manager.search_memory("programming language", 5).await.unwrap();
        assert!(!results.is_empty());

        // Test searching for specific language
        let rust_results = memory_manager.search_memory("Rust systems", 5).await.unwrap();
        assert!(!rust_results.is_empty());

        // Test searching for machine learning
        let ml_results = memory_manager.search_memory("machine learning", 5).await.unwrap();
        assert!(!ml_results.is_empty());
    }

    #[tokio::test]
    async fn test_conversation_persistence() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config.clone()).await.unwrap();

        // Start a conversation and add messages
        let conversation_id = memory_manager.start_conversation(Some("Persistent Test".to_string()));

        let message = ChatMessage::user("This message should persist");
        memory_manager.add_message(message.clone()).await.unwrap();

        // Get the current conversation to verify it was saved
        let current_conversation = memory_manager.get_current_conversation().unwrap();
        assert_eq!(current_conversation.messages.len(), 1);
        assert_eq!(current_conversation.messages[0].get_text(), "This message should persist");

        // Test that we can retrieve the conversation by ID
        let retrieved_conversation = memory_manager.get_conversation(&conversation_id).await.unwrap();
        assert!(retrieved_conversation.is_some());
        let retrieved_conversation = retrieved_conversation.unwrap();
        assert_eq!(retrieved_conversation.id, conversation_id);
        assert_eq!(retrieved_conversation.title, Some("Persistent Test".to_string()));
    }

    #[tokio::test]
    async fn test_memory_statistics() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();

        // Test that we can get stats without hanging
        let stats = memory_manager.get_stats().await.unwrap();

        // Basic sanity checks - these should always be valid
        assert!(stats.total_chunks < 1000000); // Reasonable upper bound
        assert!(stats.total_conversations < 1000000); // Reasonable upper bound
        assert!(stats.total_memories < 1000000); // Reasonable upper bound

        // File sizes should be reasonable
        assert!(stats.memory_file_size < 1_000_000_000); // Less than 1GB
        assert!(stats.index_file_size < 1_000_000_000); // Less than 1GB
    }

    #[tokio::test]
    async fn test_conversation_title_and_metadata() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Test conversation with title
        let conversation_id = memory_manager.start_conversation(Some("Important Discussion".to_string()));

        // Add some messages
        let messages = vec![
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("Rust is a systems programming language."),
        ];

        for message in messages {
            memory_manager.add_message(message).await.unwrap();
        }

        // Verify the conversation has the correct title and messages
        let current_conversation = memory_manager.get_current_conversation().unwrap();
        assert_eq!(current_conversation.title, Some("Important Discussion".to_string()));
        assert_eq!(current_conversation.messages.len(), 2);
        assert_eq!(current_conversation.id, conversation_id);
    }

    #[tokio::test]
    async fn test_memory_entry_types_and_metadata() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Test different entry types
        let entry_types = vec!["fact", "insight", "preference", "goal", "context"];
        
        for entry_type in entry_types {
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), entry_type.to_string());
            metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());

            let entry_id = memory_manager.save_memory(
                format!("This is a {} entry", entry_type),
                entry_type.to_string(),
                metadata
            ).await.unwrap();

            assert!(!entry_id.is_empty());
        }

        // Search should find entries of different types
        let results = memory_manager.search_memory("entry", 10).await.unwrap();
        assert!(results.len() >= 5);
    }

    #[tokio::test]
    async fn test_search_result_structure() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        // Store a memory entry
        memory_manager.save_memory(
            "Rust provides memory safety without garbage collection".to_string(),
            "fact".to_string(),
            HashMap::new()
        ).await.unwrap();

        // Search for it
        let results = memory_manager.search_memory("memory safety", 5).await.unwrap();
        assert!(!results.is_empty());

        let result = &results[0];
        assert!(!result.content.is_empty());
        assert!(!result.id.is_empty());
        assert_eq!(result.entry_type, "fact");
    }

    #[tokio::test]
    async fn test_conversation_message_handling() {
        let config = create_test_memory_config();
        let mut memory_manager = MemoryManager::new(config).await.unwrap();

        memory_manager.start_conversation(Some("Message Test".to_string()));

        // Add multiple messages
        for i in 0..5 {
            let message = if i % 2 == 0 {
                ChatMessage::user(format!("User message {}", i))
            } else {
                ChatMessage::assistant(format!("Assistant message {}", i))
            };
            memory_manager.add_message(message).await.unwrap();
        }

        // Test getting current conversation
        let current_conversation = memory_manager.get_current_conversation().unwrap();
        assert_eq!(current_conversation.messages.len(), 5);

        // Verify message order and content
        assert!(current_conversation.messages[0].get_text().contains("User message 0"));
        assert!(current_conversation.messages[1].get_text().contains("Assistant message 1"));
        assert!(current_conversation.messages[4].get_text().contains("User message 4"));
    }
}
