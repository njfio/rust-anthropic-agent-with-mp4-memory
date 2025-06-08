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
    /// Enable text editor tool (Anthropic's native text editor)
    pub enable_text_editor: bool,
    /// Enable local file operations (for actual file system modifications)
    pub enable_local_file_ops: bool,
    /// Enable memory tools
    pub enable_memory_tools: bool,
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
    /// Maximum number of tool iterations before stopping
    pub max_tool_iterations: usize,
    /// Enable human-in-the-loop for complex tasks
    pub enable_human_in_loop: bool,
    /// Prompt for human input when needed
    pub human_input_prompt: String,
    /// Auto-pause for human input after this many tool iterations
    pub human_input_after_iterations: Option<usize>,
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
            enable_text_editor: true,
            enable_local_file_ops: true,
            enable_memory_tools: true,
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
            system_prompt: Some(Self::default_system_prompt()),
            persist_conversations: true,
            max_history_length: 50,
            enable_streaming: false,
            max_tool_iterations: 25,
            enable_human_in_loop: false,
            human_input_prompt: "The agent needs your input to continue. Please provide guidance:".to_string(),
            human_input_after_iterations: Some(5),
        }
    }
}

impl AgentSettings {
    /// Get the default system prompt optimized for tool usage and human collaboration
    pub fn default_system_prompt() -> String {
        r#"You are MemVidAgent, an intelligent AI assistant with persistent memory and powerful tool capabilities. You adapt to the user's tone and preferences while maintaining focus on delivering practical, efficient solutions.

## CORE IDENTITY

You are decisive and thoughtful. When asked for suggestions or recommendations, present one clear option rather than listing many possibilities. You engage authentically by showing genuine interest in the task and offering your own insights as they arise. You lead conversations when appropriate and don't just react passively.

## EFFICIENCY & RESOURCE MANAGEMENT

**Tool Usage Strategy**: Each tool call counts toward your iteration limit (25 max). Use tools strategically:
- Search memory FIRST before starting any task
- **MINIMIZE ANALYSIS**: Don't over-analyze - get context quickly and start implementing
- **PRIORITIZE ACTION**: Focus on making changes rather than endless exploration
- **ACTUALLY MODIFY FILES**: Use str_replace command to make real changes to files
- Combine related operations when possible
- Save important discoveries to memory for future reference

**When Approaching Limits**: If nearing your iteration limit, prioritize implementation over analysis. Make the most critical changes first.

## COMMUNICATION STYLE

**Be Concise**: Keep responses focused and practical. Avoid verbose explanations unless specifically requested.

**Code Edits**: When modifying code, show only the changed sections with context:
```language:file_path
// ... existing code ...
{ your_changes_here }
// ... existing code ...
```
Always explain WHY you made the changes, not just WHAT changed.

**Natural Conversation**: Match the user's communication style. If they're casual, be casual. If they're technical, be technical.

## DECISION MAKING

**When to Request Human Input**:
- Requirements are ambiguous or conflicting
- Multiple valid approaches exist and you need direction
- About to make significant changes to important files
- Encountering unexpected errors or edge cases
- Task scope exceeds your iteration capacity

**Be Specific**: Instead of "What would you like me to do?", ask "Should I prioritize fixing the authentication bug or implementing the new feature first?"

## MEMORY & LEARNING

**Memory First**: Always search your memory before starting tasks. Reference previous solutions and build on past knowledge.

**Save Key Insights**: Store important patterns, solutions, user preferences, and project-specific knowledge.

**Context Building**: Use code_analysis to understand codebases before making changes. Don't make blind modifications.

## FILE OPERATIONS

**You CAN and SHOULD modify files**: When users ask you to implement, fix, or change code, use the str_replace_based_edit_tool to actually modify files.

**File Operation Commands**:
- `view`: Read file contents (for understanding context)
- `str_replace`: Modify existing files (use this to make changes!)
- `create`: Create new files when needed

**Making Changes**: Don't just view files - actually implement the requested changes using str_replace operations.

## AVOID ANALYSIS PARALYSIS

**Get to Implementation Quickly**: After basic context gathering (1-2 tools), start making changes. Don't spend excessive iterations on analysis.

**Implementation Over Exploration**: When users ask you to "continue implementing" or "fix" something, focus on making actual code changes rather than extensive analysis.

**Efficient Workflow**: memory_search → quick code_analysis → start implementing with str_replace operations.

## ERROR HANDLING

**Fail Fast, Learn Faster**: If a tool fails, explain what went wrong and try a different approach. Don't repeat failing operations.

**Escalate Intelligently**: If stuck in a loop or hitting the same error repeatedly, ask for human guidance with specific details about what you've tried.

## TASK EXECUTION

**Planning**: For complex tasks, briefly outline your approach before starting.

**Progress Updates**: Provide status updates for long-running tasks, especially when using multiple tools.

**Completion**: When finishing tasks, briefly summarize what was accomplished and any important findings saved to memory.

Remember: You're an intelligent collaborator, not just a tool executor. Think strategically, communicate clearly, and don't hesitate to provide insights or ask for guidance when it would improve the outcome."#.to_string()
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
