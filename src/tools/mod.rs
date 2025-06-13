pub mod advanced_memory_tools;
pub mod code_analysis;
pub mod custom_tools;
pub mod local_file_ops;
pub mod memory_tools;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::anthropic::models::{ContentBlock, ToolDefinition};
use crate::utils::error::{AgentError, Result};

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success<S: Into<String>>(content: S) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: None,
        }
    }

    /// Create an error tool result
    pub fn error<S: Into<String>>(content: S) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            metadata: None,
        }
    }

    /// Add metadata to the result
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Convert to a content block for the API
    pub fn to_content_block(&self, tool_use_id: String) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id,
            content: self.content.clone(),
            is_error: Some(self.is_error),
        }
    }
}

/// Trait for implementing tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition for the API
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given input
    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult>;

    /// Get the tool name
    fn name(&self) -> &str;

    /// Get the tool description
    fn description(&self) -> Option<&str>;

    /// Check if this tool can handle the given tool use
    fn can_handle(&self, tool_name: &str) -> bool {
        tool_name == self.name()
    }

    /// Validate the input before execution
    fn validate_input(&self, _input: &serde_json::Value) -> Result<()> {
        Ok(())
    }
}

/// Registry for managing tools
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.tools.len())
            .field("tool_names", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Box::new(tool));
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Get all tool definitions
    pub fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Execute a tool
    pub async fn execute(&self, tool_name: &str, input: serde_json::Value) -> Result<ToolResult> {
        let tool = self
            .get(tool_name)
            .ok_or_else(|| AgentError::tool(tool_name, "Tool not found"))?;

        tool.validate_input(&input)?;
        tool.execute(input).await
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tool is registered
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Remove a tool
    pub fn remove(&mut self, name: &str) -> Option<Box<dyn Tool>> {
        self.tools.remove(name)
    }

    /// Clear all tools
    pub fn clear(&mut self) {
        self.tools.clear();
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Helper function to create a tool definition
pub fn create_tool_definition(
    name: &str,
    description: &str,
    input_schema: serde_json::Value,
) -> ToolDefinition {
    ToolDefinition {
        tool_type: "custom".to_string(), // Use "custom" for user-defined tools
        name: name.to_string(),
        description: Some(description.to_string()),
        input_schema: Some(input_schema),
        max_uses: None,
        allowed_domains: None,
        blocked_domains: None,
    }
}

/// Helper function to extract string parameter from tool input
pub fn extract_string_param(input: &serde_json::Value, param_name: &str) -> Result<String> {
    input
        .get(param_name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            AgentError::invalid_input(format!("Missing or invalid parameter: {}", param_name))
        })
}

/// Helper function to extract optional string parameter from tool input
pub fn extract_optional_string_param(
    input: &serde_json::Value,
    param_name: &str,
) -> Option<String> {
    input
        .get(param_name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Helper function to extract integer parameter from tool input
pub fn extract_int_param(input: &serde_json::Value, param_name: &str) -> Result<i64> {
    input
        .get(param_name)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| {
            AgentError::invalid_input(format!("Missing or invalid parameter: {}", param_name))
        })
}

/// Helper function to extract optional integer parameter from tool input
pub fn extract_optional_int_param(input: &serde_json::Value, param_name: &str) -> Option<i64> {
    input.get(param_name).and_then(|v| v.as_i64())
}

/// Helper function to extract boolean parameter from tool input
pub fn extract_bool_param(input: &serde_json::Value, param_name: &str) -> Result<bool> {
    input
        .get(param_name)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| {
            AgentError::invalid_input(format!("Missing or invalid parameter: {}", param_name))
        })
}

/// Helper function to extract optional boolean parameter from tool input
pub fn extract_optional_bool_param(input: &serde_json::Value, param_name: &str) -> Option<bool> {
    input.get(param_name).and_then(|v| v.as_bool())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_result_creation() {
        let success = ToolResult::success("Test content");
        assert!(!success.is_error);
        assert_eq!(success.content, "Test content");

        let error = ToolResult::error("Error message");
        assert!(error.is_error);
        assert_eq!(error.content, "Error message");
    }

    #[test]
    fn test_parameter_extraction() {
        let input = json!({
            "string_param": "test",
            "int_param": 42,
            "bool_param": true
        });

        assert_eq!(
            extract_string_param(&input, "string_param").unwrap(),
            "test"
        );
        assert_eq!(extract_int_param(&input, "int_param").unwrap(), 42);
        assert_eq!(extract_bool_param(&input, "bool_param").unwrap(), true);

        assert!(extract_string_param(&input, "missing").is_err());
        assert!(extract_optional_string_param(&input, "missing").is_none());
    }

    // Integration tests for tool system
    use crate::tools::custom_tools::UuidGeneratorTool;

    #[tokio::test]
    async fn test_tool_registry_basic_operations() {
        let mut registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());

        let uuid_tool = UuidGeneratorTool;
        registry.register(uuid_tool);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.has_tool("generate_uuid"));

        let tool_names = registry.tool_names();
        assert_eq!(tool_names.len(), 1);
        assert_eq!(tool_names[0], "generate_uuid");
    }

    #[tokio::test]
    async fn test_uuid_generator_tool_execution() {
        let uuid_tool = UuidGeneratorTool;

        assert_eq!(uuid_tool.name(), "generate_uuid");
        assert!(uuid_tool.description().is_some());

        let result = uuid_tool.execute(json!({})).await;
        assert!(result.is_ok());

        let tool_result = result.unwrap();
        assert!(!tool_result.is_error);
        assert_eq!(tool_result.content.len(), 36); // Standard UUID length
    }

    #[tokio::test]
    async fn test_tool_registry_execution() {
        let mut registry = ToolRegistry::new();
        let uuid_tool = UuidGeneratorTool;
        registry.register(uuid_tool);

        // Test executing tool by name
        let result = registry.execute("generate_uuid", json!({})).await;
        assert!(result.is_ok());

        let tool_result = result.unwrap();
        assert!(!tool_result.is_error);
        assert_eq!(tool_result.content.len(), 36);

        // Test executing non-existent tool
        let result = registry.execute("non_existent_tool", json!({})).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_registry_management() {
        let mut registry = ToolRegistry::new();

        // Test initial state
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        // Register a tool
        registry.register(UuidGeneratorTool);
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.has_tool("generate_uuid"));

        // Test tool definitions
        let definitions = registry.get_definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "generate_uuid");

        // Test removing tool
        let removed = registry.remove("generate_uuid");
        assert!(removed.is_some());
        assert!(registry.is_empty());

        // Test clearing
        registry.register(UuidGeneratorTool);
        registry.clear();
        assert!(registry.is_empty());
    }
}
