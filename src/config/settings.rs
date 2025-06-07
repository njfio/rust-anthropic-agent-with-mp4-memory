use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::utils::error::{AgentError, Result};

/// Main configuration for the agent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Anthropic API configuration
    pub anthropic: AnthropicConfig,
    /// Memory system configuration
    pub memory: MemoryConfig,
    /// Tool system configuration
    pub tools: ToolConfig,
    /// General agent settings
    pub agent: AgentSettings,
}

/// Anthropic API configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicConfig {
    /// API key for Anthropic
    pub api_key: String,
    /// Base URL for the API (defaults to official endpoint)
    pub base_url: String,
    /// Default model to use
    pub model: String,
    /// Maximum tokens for responses
    pub max_tokens: u32,
    /// Temperature for response generation
    pub temperature: f32,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
}

/// Memory system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Path to the memory file (MP4)
    pub memory_path: PathBuf,
    /// Path to the memory index file
    pub index_path: PathBuf,
    /// Enable automatic memory saving
    pub auto_save: bool,
    /// Maximum number of conversations to keep in memory
    pub max_conversations: usize,
    /// Enable semantic search
    pub enable_search: bool,
    /// Search result limit
    pub search_limit: usize,
}

/// Tool system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Enable text editor tool
    pub enable_text_editor: bool,
    /// Enable memory tools
    pub enable_memory_tools: bool,
    /// Enable file system tools
    pub enable_file_tools: bool,
    /// Enable code execution tool
    pub enable_code_execution: bool,
    /// Enable web search tool
    pub enable_web_search: bool,
    /// Enable code analysis tool
    pub enable_code_analysis: bool,
    /// Custom tool configurations
    pub custom_tools: HashMap<String, serde_json::Value>,
    /// Tool timeout in seconds
    pub tool_timeout_seconds: u64,
}

/// General agent settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSettings {
    /// Agent name/identifier
    pub name: String,
    /// System prompt for the agent
    pub system_prompt: Option<String>,
    /// Enable conversation persistence
    pub persist_conversations: bool,
    /// Maximum conversation history length
    pub max_history_length: usize,
    /// Enable streaming responses
    pub enable_streaming: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            anthropic: AnthropicConfig::default(),
            memory: MemoryConfig::default(),
            tools: ToolConfig::default(),
            agent: AgentSettings::default(),
        }
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-opus-4-20250514".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
            timeout_seconds: 120,
            max_retries: 3,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_path: PathBuf::from("agent_memory.mp4"),
            index_path: PathBuf::from("agent_memory"), // Base path - library will add .metadata and .vector extensions
            auto_save: true,
            max_conversations: 1000,
            enable_search: true,
            search_limit: 10,
        }
    }
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            enable_text_editor: false,
            enable_memory_tools: true,
            enable_file_tools: true,
            enable_code_execution: true,
            enable_web_search: true,
            enable_code_analysis: true,
            custom_tools: HashMap::new(),
            tool_timeout_seconds: 30,
        }
    }
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            name: "MemVidAgent".to_string(),
            system_prompt: None,
            persist_conversations: true,
            max_history_length: 50,
            enable_streaming: false,
        }
    }
}

impl AgentConfig {
    /// Create a new configuration with the given Anthropic API key
    pub fn with_anthropic_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.anthropic.api_key = api_key.into();
        self
    }

    /// Set the memory path
    pub fn with_memory_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        let path = path.into();
        self.memory.memory_path = path.clone();
        // Use base path without extension - rust-mp4-memory will add .metadata and .vector
        if let Some(stem) = path.file_stem() {
            self.memory.index_path = path.with_file_name(stem);
        } else {
            self.memory.index_path = path.clone();
        }
        self
    }

    /// Set the model to use
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.anthropic.model = model.into();
        self
    }

    /// Set the system prompt
    pub fn with_system_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.agent.system_prompt = Some(prompt.into());
        self
    }

    /// Load configuration from a TOML file
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.into())?;
        let config: AgentConfig = toml::from_str(&content)
            .map_err(|e| AgentError::config(format!("Failed to parse config file: {}", e)))?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| AgentError::config(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path.into(), content)?;
        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.anthropic.api_key.is_empty() {
            return Err(AgentError::config(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or provide in config.",
            ));
        }

        if self.anthropic.max_tokens == 0 {
            return Err(AgentError::config("max_tokens must be greater than 0"));
        }

        if self.anthropic.temperature < 0.0 || self.anthropic.temperature > 2.0 {
            return Err(AgentError::config(
                "temperature must be between 0.0 and 2.0",
            ));
        }

        Ok(())
    }
}
