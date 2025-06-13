#[cfg(test)]
mod tests {
    use crate::{Agent, AgentBuilder};
    use crate::config::{AgentConfig, AnthropicConfig, MemoryConfig, ToolConfig};
    use crate::config::settings::AgentSettings;
    use tempfile::TempDir;
    use std::path::PathBuf;

    fn create_test_config() -> AgentConfig {
        let temp_dir = TempDir::new().unwrap();
        AgentConfig {
            anthropic: AnthropicConfig {
                api_key: "test-key".to_string(),
                base_url: "http://localhost:8080".to_string(),
                model: "claude-test".to_string(),
                max_tokens: 1000,
                temperature: 0.7,
                timeout_seconds: 30,
                max_retries: 3,
            },
            memory: MemoryConfig {
                memory_path: temp_dir.path().join("test_memory.json"),
                index_path: temp_dir.path().join("test_memory"),
                auto_save: true,
                max_conversations: 100,
                enable_search: true,
                search_limit: 10,
            },
            tools: ToolConfig {
                enable_text_editor: true,
                enable_local_file_ops: false, // Disable for testing
                enable_memory_tools: true,
                enable_code_execution: false, // Disable for testing
                enable_web_search: false, // Disable for testing
                enable_code_analysis: false, // Disable for testing
                custom_tools: std::collections::HashMap::new(),
                tool_timeout_seconds: 30,
            },
            agent: AgentSettings {
                name: "TestAgent".to_string(),
                system_prompt: Some("Test system prompt".to_string()),
                persist_conversations: true,
                max_history_length: 10,
                enable_streaming: false,
                max_tool_iterations: 5,
                enable_human_in_loop: false,
                human_input_prompt: "Test prompt".to_string(),
                human_input_after_iterations: None,
            },
        }
    }

    #[tokio::test]
    async fn test_agent_config_validation() {
        let mut config = create_test_config();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Empty API key should fail
        config.anthropic.api_key = "".to_string();
        assert!(config.validate().is_err());
        
        // Invalid temperature should fail
        config.anthropic.api_key = "test-key".to_string();
        config.anthropic.temperature = -1.0;
        assert!(config.validate().is_err());
        
        config.anthropic.temperature = 3.0;
        assert!(config.validate().is_err());
        
        // Zero max_tokens should fail
        config.anthropic.temperature = 0.7;
        config.anthropic.max_tokens = 0;
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_agent_builder() {
        let config = create_test_config();

        let builder = AgentBuilder::new()
            .with_config(config.clone())
            .with_api_key("test-key")
            .with_model("claude-test")
            .with_memory_path("test_memory.json");

        assert_eq!(builder.config.anthropic.api_key, "test-key");
        assert_eq!(builder.config.anthropic.model, "claude-test");
        assert_eq!(builder.config.memory.memory_path, PathBuf::from("test_memory.json"));
    }

    #[tokio::test]
    async fn test_agent_configuration_methods() {
        let config = create_test_config();
        
        // Test with_anthropic_key
        let config = config.with_anthropic_key("new-key");
        assert_eq!(config.anthropic.api_key, "new-key");
        
        // Test with_model
        let config = config.with_model("new-model");
        assert_eq!(config.anthropic.model, "new-model");
        
        // Test with_system_prompt
        let config = config.with_system_prompt("New prompt");
        assert_eq!(config.agent.system_prompt, Some("New prompt".to_string()));
        
        // Test with_memory_path
        let config = config.with_memory_path("new_memory.json");
        assert_eq!(config.memory.memory_path, PathBuf::from("new_memory.json"));
    }

    #[tokio::test]
    async fn test_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let config = create_test_config();
        
        // Test saving config
        assert!(config.save_to_file(&config_path).is_ok());
        assert!(config_path.exists());
        
        // Test loading config
        let loaded_config = AgentConfig::from_file(&config_path);
        assert!(loaded_config.is_ok());
        
        let loaded_config = loaded_config.unwrap();
        assert_eq!(loaded_config.anthropic.api_key, config.anthropic.api_key);
        assert_eq!(loaded_config.anthropic.model, config.anthropic.model);
        assert_eq!(loaded_config.agent.name, config.agent.name);
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = AgentConfig::default();
        
        // Test default values
        assert_eq!(config.anthropic.base_url, "https://api.anthropic.com");
        assert_eq!(config.anthropic.model, "claude-sonnet-4-20250514");
        assert_eq!(config.anthropic.max_tokens, 8192);
        assert_eq!(config.anthropic.temperature, 0.7);
        assert_eq!(config.anthropic.timeout_seconds, 300);
        assert_eq!(config.anthropic.max_retries, 3);
        
        assert_eq!(config.memory.memory_path, PathBuf::from("agent_memory.json"));
        assert_eq!(config.memory.auto_save, true);
        assert_eq!(config.memory.max_conversations, 1000);
        assert_eq!(config.memory.enable_search, true);
        assert_eq!(config.memory.search_limit, 10);
        
        assert_eq!(config.tools.enable_text_editor, true);
        assert_eq!(config.tools.enable_memory_tools, true);
        assert_eq!(config.tools.tool_timeout_seconds, 30);
        
        assert_eq!(config.agent.name, "MemVidAgent");
        assert_eq!(config.agent.persist_conversations, true);
        assert_eq!(config.agent.max_history_length, 50);
        assert_eq!(config.agent.enable_streaming, false);
        assert_eq!(config.agent.max_tool_iterations, 50);
        assert_eq!(config.agent.enable_human_in_loop, false);
    }

    #[test]
    fn test_agent_settings_default_system_prompt() {
        let prompt = AgentSettings::default_system_prompt();
        
        // Test that the prompt contains key elements
        assert!(prompt.contains("MemVidAgent"));
        assert!(prompt.contains("CRITICAL FILE CREATION RULE"));
        assert!(prompt.contains("local_file_editor"));
        assert!(prompt.contains("str_replace_based_edit_tool"));
        assert!(prompt.contains("memory_search"));
        assert!(prompt.len() > 1000); // Should be a substantial prompt
    }

    #[test]
    fn test_anthropic_config_environment_variable() {
        // Test that API key is read from environment variable
        std::env::set_var("ANTHROPIC_API_KEY", "env-test-key");
        let config = AnthropicConfig::default();
        assert_eq!(config.api_key, "env-test-key");
        
        // Clean up
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn test_memory_config_path_handling() {
        let config = MemoryConfig::default();
        
        // Test default paths
        assert_eq!(config.memory_path, PathBuf::from("agent_memory.json"));
        assert_eq!(config.index_path, PathBuf::from("agent_memory"));
        
        // Test that index path is derived from memory path
        let agent_config = AgentConfig::default().with_memory_path("custom/path/memory.json");
        assert_eq!(agent_config.memory.memory_path, PathBuf::from("custom/path/memory.json"));
        assert_eq!(agent_config.memory.index_path, PathBuf::from("custom/path/memory"));
    }

    #[test]
    fn test_tool_config_defaults() {
        let config = ToolConfig::default();
        
        // Test that safe tools are enabled by default
        assert!(config.enable_text_editor);
        assert!(config.enable_memory_tools);
        assert!(config.enable_local_file_ops);
        
        // Test that potentially dangerous tools are enabled (but can be disabled in production)
        assert!(config.enable_code_execution);
        assert!(config.enable_web_search);
        assert!(config.enable_code_analysis);
        
        assert_eq!(config.tool_timeout_seconds, 30);
        assert!(config.custom_tools.is_empty());
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = AgentConfig::default()
            .with_anthropic_key("test-key")
            .with_model("test-model")
            .with_memory_path("test-memory.json")
            .with_system_prompt("Test prompt");
        
        assert_eq!(config.anthropic.api_key, "test-key");
        assert_eq!(config.anthropic.model, "test-model");
        assert_eq!(config.memory.memory_path, PathBuf::from("test-memory.json"));
        assert_eq!(config.agent.system_prompt, Some("Test prompt".to_string()));
    }
}
