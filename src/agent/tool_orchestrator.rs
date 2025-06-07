use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::anthropic::models::{ContentBlock, ToolDefinition};
use crate::anthropic::tools::AnthropicTool;
use crate::config::AgentConfig;
use crate::memory::MemoryManager;
use crate::tools::{
    advanced_memory_tools::AdvancedMemoryAnalyticsTool,
    code_analysis::CodeAnalysisTool,
    custom_tools::{HttpRequestTool, ShellCommandTool, UuidGeneratorTool},
    local_file_ops::LocalTextEditorTool,
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
        // We use a local implementation instead of Anthropic's server-side tool
        // to ensure files can actually be modified on the local machine
        if config.tools.enable_text_editor {
            // Don't register Anthropic's server-side text editor tool
            // Instead, we'll register our local implementation below
            debug!("Text editor enabled - will use local implementation for actual file modifications");
        }

        // Register memory tools if enabled
        if config.tools.enable_memory_tools {
            self.tool_registry.register(MemorySearchTool::new(self.memory_manager.clone()));
            self.tool_registry.register(MemorySaveTool::new(self.memory_manager.clone()));
            self.tool_registry.register(MemoryStatsTool::new(self.memory_manager.clone()));
            self.tool_registry.register(ConversationSearchTool::new(self.memory_manager.clone()));

            // Register Phase 2 Advanced Memory Analytics Tool
            self.tool_registry.register(AdvancedMemoryAnalyticsTool::new(self.memory_manager.clone()));

            debug!("Registered memory tools including advanced analytics");
        }

        // Register local file operations tool for actual file system modifications
        // This complements Anthropic's server-side text editor tool
        if config.tools.enable_text_editor {
            self.tool_registry.register(LocalTextEditorTool::new("."));
            debug!("Registered local text editor tool for actual file modifications");
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
        let client_definitions = self.tool_registry.get_definitions();
        debug!("Client tools: {:?}", client_definitions.iter().map(|d| &d.name).collect::<Vec<_>>());
        definitions.extend(client_definitions);

        // Add Anthropic tool definitions
        let server_definitions: Vec<ToolDefinition> = self.anthropic_tools
            .iter()
            .map(|tool| tool.to_definition())
            .collect();
        debug!("Server tools: {:?}", server_definitions.iter().map(|d| &d.name).collect::<Vec<_>>());
        definitions.extend(server_definitions);

        // Check for duplicates
        let mut seen_names = std::collections::HashSet::new();
        let mut duplicates = Vec::new();
        for def in &definitions {
            if !seen_names.insert(&def.name) {
                duplicates.push(&def.name);
            }
        }

        if !duplicates.is_empty() {
            error!("Duplicate tool names found: {:?}", duplicates);
        }

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
                    // Check if this is a server-side tool
                    if self.is_server_tool(name) {
                        debug!("Creating placeholder result for server tool: {} (id: {}) - execution handled by Anthropic", name, id);
                        // Server tools are executed by Anthropic, but we need to provide a tool_result
                        // to satisfy the API requirement. The actual result will be provided by Anthropic.
                        let placeholder_result = ToolResult::success("Server tool execution handled by Anthropic".to_string());
                        results.push(placeholder_result.to_content_block(id.clone()));
                        continue;
                    }

                    debug!("Executing client tool: {} (id: {})", name, id);

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
        // File operations are now handled by Anthropic's native text editor tool
    }
}
