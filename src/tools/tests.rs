#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Tool, ToolRegistry, ToolResult};
    use crate::tools::memory_tools::{MemorySearchTool, MemorySaveTool, MemoryStatsTool};
    use crate::tools::custom_tools::{UuidGeneratorTool, ShellCommandTool};
    use crate::memory::MemoryManager;
    use crate::config::MemoryConfig;
    use serde_json::{json, Value};
    use tempfile::TempDir;
    use std::collections::HashMap;

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
    async fn test_tool_registry_creation() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.list_tools().len(), 0);
    }

    #[tokio::test]
    async fn test_tool_registry_registration() {
        let mut registry = ToolRegistry::new();
        let uuid_tool = UuidGeneratorTool::new();
        
        registry.register_tool(Box::new(uuid_tool));
        let tools = registry.list_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "uuid_generator");
    }

    #[tokio::test]
    async fn test_uuid_generator_tool() {
        let uuid_tool = UuidGeneratorTool::new();
        
        // Test tool metadata
        assert_eq!(uuid_tool.name(), "uuid_generator");
        assert!(!uuid_tool.description().is_empty());
        
        // Test UUID generation
        let params = json!({});
        let result = uuid_tool.execute(params).await;
        assert!(result.is_ok());
        
        let tool_result = result.unwrap();
        match tool_result {
            ToolResult::Success { content, .. } => {
                // Should contain a valid UUID
                assert!(content.contains("-"));
                assert_eq!(content.len(), 36); // Standard UUID length
            }
            _ => panic!("Expected successful UUID generation"),
        }
    }

    #[tokio::test]
    async fn test_shell_command_tool_safe_command() {
        let shell_tool = ShellCommandTool::new();
        
        // Test tool metadata
        assert_eq!(shell_tool.name(), "shell_command");
        assert!(!shell_tool.description().is_empty());
        
        // Test safe command execution
        let params = json!({
            "command": "echo 'Hello, World!'"
        });
        
        let result = shell_tool.execute(params).await;
        assert!(result.is_ok());
        
        let tool_result = result.unwrap();
        match tool_result {
            ToolResult::Success { content, .. } => {
                assert!(content.contains("Hello, World!"));
            }
            _ => panic!("Expected successful command execution"),
        }
    }

    #[tokio::test]
    async fn test_shell_command_tool_invalid_params() {
        let shell_tool = ShellCommandTool::new();
        
        // Test with missing command parameter
        let params = json!({});
        let result = shell_tool.execute(params).await;
        assert!(result.is_err());
        
        // Test with invalid command parameter type
        let params = json!({
            "command": 123
        });
        let result = shell_tool.execute(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_search_tool() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();
        let search_tool = MemorySearchTool::new(memory_manager.clone());
        
        // First, add some data to search
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "test".to_string());
        memory_manager.save_memory(
            "Rust is a systems programming language".to_string(),
            "fact".to_string(),
            metadata
        ).await.unwrap();
        
        // Test tool metadata
        assert_eq!(search_tool.name(), "memory_search");
        assert!(!search_tool.description().is_empty());
        
        // Test memory search
        let params = json!({
            "query": "Rust programming",
            "limit": 5
        });
        
        let result = search_tool.execute(params).await;
        assert!(result.is_ok());
        
        let tool_result = result.unwrap();
        match tool_result {
            ToolResult::Success { content, .. } => {
                // Should find the memory entry we added
                assert!(content.contains("Rust") || content.contains("programming"));
            }
            _ => panic!("Expected successful memory search"),
        }
    }

    #[tokio::test]
    async fn test_memory_save_tool() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();
        let save_tool = MemorySaveTool::new(memory_manager.clone());
        
        // Test tool metadata
        assert_eq!(save_tool.name(), "memory_save");
        assert!(!save_tool.description().is_empty());
        
        // Test memory save
        let params = json!({
            "content": "Python is great for data science",
            "entry_type": "fact",
            "metadata": {
                "category": "programming",
                "language": "python"
            }
        });
        
        let result = save_tool.execute(params).await;
        assert!(result.is_ok());
        
        let tool_result = result.unwrap();
        match tool_result {
            ToolResult::Success { content, .. } => {
                // Should return the entry ID
                assert!(!content.is_empty());
                assert!(content.contains("saved") || content.contains("stored"));
            }
            _ => panic!("Expected successful memory save"),
        }
    }

    #[tokio::test]
    async fn test_memory_stats_tool() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();
        let stats_tool = MemoryStatsTool::new(memory_manager.clone());
        
        // Test tool metadata
        assert_eq!(stats_tool.name(), "memory_stats");
        assert!(!stats_tool.description().is_empty());
        
        // Test memory stats
        let params = json!({});
        let result = stats_tool.execute(params).await;
        assert!(result.is_ok());
        
        let tool_result = result.unwrap();
        match tool_result {
            ToolResult::Success { content, .. } => {
                // Should contain statistics information
                assert!(content.contains("chunks") || content.contains("entries") || content.contains("stats"));
            }
            _ => panic!("Expected successful memory stats"),
        }
    }

    #[tokio::test]
    async fn test_memory_search_tool_invalid_params() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();
        let search_tool = MemorySearchTool::new(memory_manager);
        
        // Test with missing query parameter
        let params = json!({});
        let result = search_tool.execute(params).await;
        assert!(result.is_err());
        
        // Test with invalid query parameter type
        let params = json!({
            "query": 123
        });
        let result = search_tool.execute(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_save_tool_invalid_params() {
        let config = create_test_memory_config();
        let memory_manager = MemoryManager::new(config).await.unwrap();
        let save_tool = MemorySaveTool::new(memory_manager);
        
        // Test with missing content parameter
        let params = json!({
            "entry_type": "fact"
        });
        let result = save_tool.execute(params).await;
        assert!(result.is_err());
        
        // Test with missing entry_type parameter
        let params = json!({
            "content": "Some content"
        });
        let result = save_tool.execute(params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_registry_execution() {
        let mut registry = ToolRegistry::new();
        let uuid_tool = UuidGeneratorTool::new();
        
        registry.register_tool(Box::new(uuid_tool));
        
        // Test executing tool by name
        let params = json!({});
        let result = registry.execute_tool("uuid_generator", params).await;
        assert!(result.is_ok());
        
        // Test executing non-existent tool
        let params = json!({});
        let result = registry.execute_tool("non_existent_tool", params).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_registry_multiple_tools() {
        let mut registry = ToolRegistry::new();
        
        // Register multiple tools
        registry.register_tool(Box::new(UuidGeneratorTool::new()));
        registry.register_tool(Box::new(ShellCommandTool::new()));
        
        let tools = registry.list_tools();
        assert_eq!(tools.len(), 2);
        
        // Verify both tools are registered
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(tool_names.contains(&"uuid_generator"));
        assert!(tool_names.contains(&"shell_command"));
    }

    #[tokio::test]
    async fn test_tool_result_types() {
        // Test Success result
        let success_result = ToolResult::Success {
            content: "Test content".to_string(),
            metadata: Some(json!({"key": "value"})),
        };
        
        match success_result {
            ToolResult::Success { content, metadata } => {
                assert_eq!(content, "Test content");
                assert!(metadata.is_some());
            }
            _ => panic!("Expected Success result"),
        }
        
        // Test Error result
        let error_result = ToolResult::Error {
            message: "Test error".to_string(),
            details: Some("Error details".to_string()),
        };
        
        match error_result {
            ToolResult::Error { message, details } => {
                assert_eq!(message, "Test error");
                assert_eq!(details, Some("Error details".to_string()));
            }
            _ => panic!("Expected Error result"),
        }
    }

    #[tokio::test]
    async fn test_tool_parameter_validation() {
        let uuid_tool = UuidGeneratorTool::new();
        
        // UUID tool should work with empty parameters
        let result = uuid_tool.execute(json!({})).await;
        assert!(result.is_ok());
        
        // UUID tool should work with extra parameters (should ignore them)
        let result = uuid_tool.execute(json!({"extra": "param"})).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_concurrent_execution() {
        let uuid_tool = UuidGeneratorTool::new();
        
        // Execute multiple UUID generations concurrently
        let mut handles = vec![];
        for _ in 0..5 {
            let tool = UuidGeneratorTool::new();
            let handle = tokio::spawn(async move {
                tool.execute(json!({})).await
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}
