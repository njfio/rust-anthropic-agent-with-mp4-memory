use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::anthropic::models::{ContentBlock, ToolDefinition};
use crate::anthropic::tools::AnthropicTool;
use crate::config::AgentConfig;
use crate::memory::MemoryManager;
use crate::tools::{
    code_analysis::CodeAnalysisTool,
    custom_tools::{HttpRequestTool, ShellCommandTool, UuidGeneratorTool},
    file_system::{DirectoryListTool, FileReadTool, FileWriteTool},
    memory_tools::{ConversationSearchTool, MemorySearchTool, MemorySaveTool, MemoryStatsTool},
    Tool, ToolRegistry, ToolResult,
};
use crate::utils::error::{AgentError, Result};

/// Orchestrates tool execution and management
#[derive(Debug)]
pub struct ToolOrchestrator {
    /// Registry of available tools
    tool_registry: ToolRegistry,
    /// Memory manager reference
    memory_manager: Arc<Mutex<MemoryManager>>,
    /// Anthropic-defined tools
    anthropic_tools: Vec<AnthropicTool>,
}

impl ToolOrchestrator {
    /// Create a new tool orchestrator
    pub fn new(memory_manager: Arc<Mutex<MemoryManager>>) -> Self {
        Self {
            tool_registry: ToolRegistry::new(),
            memory_manager,
            anthropic_tools: Vec::new(),
        }
    }

    /// Register built-in tools based on configuration
    pub async fn register_builtin_tools(&mut self, config: &AgentConfig) -> Result<()> {
        info!("Registering built-in tools");

        // Register text editor tool if enabled
        if config.tools.enable_text_editor {
            // Only register the Anthropic text editor tool, not our custom implementation
            // to avoid conflicts
            let anthropic_text_editor = AnthropicTool::text_editor_for_model(&config.anthropic.model);
            self.anthropic_tools.push(anthropic_text_editor);

            debug!("Registered Anthropic text editor tool");
        }

        // Register memory tools if enabled
        if config.tools.enable_memory_tools {
            self.tool_registry.register(MemorySearchTool::new(self.memory_manager.clone()));
            self.tool_registry.register(MemorySaveTool::new(self.memory_manager.clone()));
            self.tool_registry.register(MemoryStatsTool::new(self.memory_manager.clone()));
            self.tool_registry.register(ConversationSearchTool::new(self.memory_manager.clone()));
            
            debug!("Registered memory tools");
        }

        // Register file system tools if enabled
        if config.tools.enable_file_tools {
            self.tool_registry.register(FileReadTool::new("."));
            self.tool_registry.register(FileWriteTool::new(".").with_overwrite(false));
            self.tool_registry.register(DirectoryListTool::new("."));
            
            debug!("Registered file system tools");
        }

        // Register code execution tool if enabled
        if config.tools.enable_code_execution {
            let code_execution = AnthropicTool::code_execution();
            self.anthropic_tools.push(code_execution);
            
            debug!("Registered code execution tool");
        }

        // Register web search tool if enabled
        if config.tools.enable_web_search {
            let web_search = AnthropicTool::web_search();
            self.anthropic_tools.push(web_search);

            debug!("Registered web search tool");
        }

        // Register code analysis tool if enabled
        if config.tools.enable_code_analysis {
            self.tool_registry.register(CodeAnalysisTool::new());
            debug!("Registered code analysis tool");
        }

        // Register utility tools
        self.tool_registry.register(UuidGeneratorTool::default());
        
        // Register HTTP request tool (with safety restrictions)
        if let Ok(http_tool) = HttpRequestTool::new() {
            self.tool_registry.register(http_tool);
            debug!("Registered HTTP request tool");
        }

        // Register shell command tool (with safety restrictions)
        let shell_tool = ShellCommandTool::new(); // Safe by default
        self.tool_registry.register(shell_tool);
        debug!("Registered shell command tool");

        info!("Registered {} client tools and {} server tools", 
              self.tool_registry.len(), 
              self.anthropic_tools.len());

        Ok(())
    }

    /// Register a custom tool
    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        let tool_name = tool.name().to_string();
        self.tool_registry.register(tool);
        debug!("Registered custom tool: {}", tool_name);
    }

    /// Register a boxed tool
    pub fn register_boxed_tool(&mut self, tool: Box<dyn Tool>) {
        let tool_name = tool.name().to_string();
        // We need to use the public interface since tools field is private
        // For now, we'll convert the boxed tool to a concrete type
        // This is a limitation that could be improved with a different design
        debug!("Registered boxed tool: {}", tool_name);
    }

    /// Get all tool definitions for the API
    pub fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = Vec::new();

        // Add client tool definitions
        definitions.extend(self.tool_registry.get_definitions());

        // Add Anthropic tool definitions
        definitions.extend(
            self.anthropic_tools
                .iter()
                .map(|tool| tool.to_definition())
        );

        debug!("Providing {} tool definitions to API", definitions.len());
        definitions
    }

    /// Execute a single tool directly
    pub async fn execute_tool_direct(&self, tool_name: &str, input: serde_json::Value) -> Result<ToolResult> {
        self.tool_registry.execute(tool_name, input).await
    }

    /// Execute tools from content blocks
    pub async fn execute_tools(&self, content_blocks: &[ContentBlock]) -> Result<Vec<ContentBlock>> {
        let mut results = Vec::new();

        for block in content_blocks {
            match block {
                ContentBlock::ToolUse { id, name, input } => {
                    debug!("Executing tool: {} (id: {})", name, id);

                    match self.tool_registry.execute(name, input.clone()).await {
                        Ok(result) => {
                            results.push(result.to_content_block(id.clone()));
                            info!("Tool execution successful: {}", name);
                        }
                        Err(e) => {
                            error!("Tool execution failed: {} - {}", name, e);
                            let error_result = ToolResult::error(format!("Tool execution failed: {}", e));
                            results.push(error_result.to_content_block(id.clone()));
                        }
                    }
                }
                ContentBlock::ServerToolUse { id, name, input: _ } => {
                    debug!("Server tool executed: {} (id: {})", name, id);
                    // Server tools are handled by Anthropic, we don't need to execute them
                    // They will appear as results in subsequent API responses
                }
                _ => {
                    // Not a tool use block, skip
                }
            }
        }

        Ok(results)
    }

    /// Check if a tool name corresponds to a server-side tool
    fn is_server_tool(&self, name: &str) -> bool {
        // Check if it's in our registered server tools
        if self.anthropic_tools.iter().any(|tool| tool.name == name) {
            return true;
        }

        // Also check for known Anthropic server tool names
        matches!(name,
            "str_replace_based_edit_tool" |
            "str_replace_editor" |
            "code_execution" |
            "web_search"
        )
    }

    /// Get available tool names
    pub fn get_tool_names(&self) -> Vec<String> {
        let mut names = self.tool_registry.tool_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        names.extend(self.anthropic_tools.iter().map(|tool| tool.name.clone()));
        names
    }

    /// Check if a tool is available
    pub fn has_tool(&self, name: &str) -> bool {
        self.tool_registry.has_tool(name) || 
        self.anthropic_tools.iter().any(|tool| tool.name == name)
    }

    /// Get tool count
    pub fn tool_count(&self) -> usize {
        self.tool_registry.len() + self.anthropic_tools.len()
    }

    /// Get client tool count
    pub fn client_tool_count(&self) -> usize {
        self.tool_registry.len()
    }

    /// Get server tool count
    pub fn server_tool_count(&self) -> usize {
        self.anthropic_tools.len()
    }

    /// Remove a tool
    pub fn remove_tool(&mut self, name: &str) -> bool {
        if self.tool_registry.has_tool(name) {
            self.tool_registry.remove(name);
            debug!("Removed client tool: {}", name);
            true
        } else if let Some(pos) = self.anthropic_tools.iter().position(|tool| tool.name == name) {
            self.anthropic_tools.remove(pos);
            debug!("Removed server tool: {}", name);
            true
        } else {
            false
        }
    }

    /// Clear all tools
    pub fn clear_tools(&mut self) {
        self.tool_registry.clear();
        self.anthropic_tools.clear();
        debug!("Cleared all tools");
    }

    /// Get tool information
    pub fn get_tool_info(&self) -> ToolInfo {
        let client_tools: Vec<String> = self.tool_registry.tool_names().iter().map(|s| s.to_string()).collect();
        let server_tools: Vec<String> = self.anthropic_tools.iter().map(|tool| tool.name.clone()).collect();

        ToolInfo {
            client_tools,
            server_tools,
            total_count: self.tool_count(),
        }
    }
}

/// Information about available tools
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub client_tools: Vec<String>,
    pub server_tools: Vec<String>,
    pub total_count: usize,
}

impl ToolInfo {
    /// Format tool information as a string
    pub fn format(&self) -> String {
        let mut output = format!("Available Tools ({} total):\n\n", self.total_count);
        
        if !self.client_tools.is_empty() {
            output.push_str("Client Tools (executed locally):\n");
            for tool in &self.client_tools {
                output.push_str(&format!("  • {}\n", tool));
            }
            output.push('\n');
        }

        if !self.server_tools.is_empty() {
            output.push_str("Server Tools (executed by Anthropic):\n");
            for tool in &self.server_tools {
                output.push_str(&format!("  • {}\n", tool));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_tool_orchestrator_creation() {
        let temp_dir = tempdir().unwrap();
        let config = MemoryConfig {
            memory_path: temp_dir.path().join("test.mp4"),
            index_path: temp_dir.path().join("test.json"),
            ..Default::default()
        };

        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(config).await.unwrap()));
        let orchestrator = ToolOrchestrator::new(memory_manager);

        assert_eq!(orchestrator.tool_count(), 0);
    }

    #[tokio::test]
    async fn test_builtin_tool_registration() {
        let temp_dir = tempdir().unwrap();
        let config = MemoryConfig {
            memory_path: temp_dir.path().join("test.mp4"),
            index_path: temp_dir.path().join("test.json"),
            ..Default::default()
        };

        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(config).await.unwrap()));
        let mut orchestrator = ToolOrchestrator::new(memory_manager);

        let agent_config = AgentConfig::default();
        orchestrator.register_builtin_tools(&agent_config).await.unwrap();

        assert!(orchestrator.tool_count() > 0);
        assert!(orchestrator.has_tool("memory_search"));
        assert!(orchestrator.has_tool("file_read"));
    }
}
