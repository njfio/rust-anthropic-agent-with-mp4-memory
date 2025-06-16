//! Integration between DSPy modules and the Tool trait
//!
//! This module provides a bridge that allows DSPy modules to be used as tools
//! and enables tool composition with DSPy optimization capabilities.

use crate::anthropic::models::ToolDefinition;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::Module,
    signature::Signature,
};
use crate::tools::{Tool, ToolResult};
use crate::utils::error::{AgentError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Wrapper that makes a DSPy module usable as a Tool
pub struct DspyModuleTool<I, O> {
    /// The wrapped DSPy module
    module: Arc<dyn Module<Input = I, Output = O>>,
    /// Tool definition for the API
    tool_definition: ToolDefinition,
    /// Input schema for validation
    input_schema: serde_json::Value,
    /// Performance metrics
    metrics: ToolMetrics,
}

/// Performance metrics for DSPy tools
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetrics {
    /// Total number of executions
    pub execution_count: u64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// Average execution time in milliseconds
    pub average_execution_time_ms: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Last execution timestamp
    pub last_execution_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ToolMetrics {
    /// Record a successful execution
    pub fn record_success(&mut self, execution_time_ms: f64) {
        self.execution_count += 1;
        self.successful_executions += 1;
        self.total_execution_time_ms += execution_time_ms;
        self.average_execution_time_ms = self.total_execution_time_ms / self.execution_count as f64;
        self.success_rate = self.successful_executions as f64 / self.execution_count as f64;
        self.last_execution_at = Some(chrono::Utc::now());
    }

    /// Record a failed execution
    pub fn record_failure(&mut self, execution_time_ms: f64) {
        self.execution_count += 1;
        self.failed_executions += 1;
        self.total_execution_time_ms += execution_time_ms;
        self.average_execution_time_ms = self.total_execution_time_ms / self.execution_count as f64;
        self.success_rate = self.successful_executions as f64 / self.execution_count as f64;
        self.last_execution_at = Some(chrono::Utc::now());
    }
}

impl<I, O> DspyModuleTool<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    /// Create a new DSPy module tool
    pub fn new(
        module: Arc<dyn Module<Input = I, Output = O>>,
        tool_name: String,
        description: String,
        input_schema: serde_json::Value,
    ) -> Self {
        let tool_definition = ToolDefinition {
            tool_type: "function".to_string(),
            name: tool_name,
            description: Some(description),
            input_schema: Some(input_schema.clone()),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        };

        Self {
            module,
            tool_definition,
            input_schema,
            metrics: ToolMetrics::default(),
        }
    }

    /// Create from a DSPy module with automatic schema generation
    pub fn from_module(
        module: Arc<dyn Module<Input = I, Output = O>>,
        tool_name: Option<String>,
        description: Option<String>,
    ) -> DspyResult<Self> {
        let name = tool_name.unwrap_or_else(|| module.name().to_string());
        let desc = description.unwrap_or_else(|| {
            format!("DSPy module: {}", module.name())
        });

        // Generate input schema from signature
        let signature = module.signature();
        let input_schema = Self::generate_input_schema(signature)?;

        Ok(Self::new(module, name, desc, input_schema))
    }

    /// Generate JSON schema from DSPy signature
    fn generate_input_schema(signature: &Signature<I, O>) -> DspyResult<serde_json::Value> {
        // Simplified schema generation - in practice, this would use the signature's field definitions
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "object",
                    "description": format!("Input for {}", signature.name)
                }
            },
            "required": ["input"]
        });

        Ok(schema)
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &ToolMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ToolMetrics::default();
    }

    /// Convert input JSON to typed input
    fn convert_input(&self, input: &Value) -> DspyResult<I> {
        // Extract the actual input from the JSON structure
        let input_value = input.get("input")
            .ok_or_else(|| DspyError::invalid_input("Missing 'input' field in tool input".to_string()))?;

        serde_json::from_value(input_value.clone())
            .map_err(|e| DspyError::invalid_input(format!("Failed to deserialize input: {}", e)))
    }

    /// Convert typed output to JSON
    fn convert_output(&self, output: &O) -> DspyResult<String> {
        serde_json::to_string_pretty(output)
            .map_err(|e| DspyError::serialization("output", &format!("Failed to serialize output: {}", e)))
    }
}

#[async_trait]
impl<I, O> Tool for DspyModuleTool<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.tool_definition.clone()
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let start_time = std::time::Instant::now();

        // Convert input
        let typed_input = match self.convert_input(&input) {
            Ok(input) => input,
            Err(e) => {
                let execution_time = start_time.elapsed().as_millis() as f64;
                // Note: We can't mutate self here due to &self, so we skip metrics recording
                error!("DSPy tool input conversion failed: {}", e);
                return Ok(ToolResult::error(format!("Input conversion failed: {}", e)));
            }
        };

        // Execute the DSPy module
        let result = match self.module.forward(typed_input).await {
            Ok(output) => {
                match self.convert_output(&output) {
                    Ok(output_str) => {
                        let execution_time = start_time.elapsed().as_millis() as f64;
                        debug!("DSPy tool executed successfully in {:.2}ms", execution_time);
                        ToolResult::success(output_str)
                    }
                    Err(e) => {
                        error!("DSPy tool output conversion failed: {}", e);
                        ToolResult::error(format!("Output conversion failed: {}", e))
                    }
                }
            }
            Err(e) => {
                error!("DSPy tool execution failed: {}", e);
                ToolResult::error(format!("Module execution failed: {}", e))
            }
        };

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.tool_definition.name
    }

    fn description(&self) -> Option<&str> {
        self.tool_definition.description.as_deref()
    }

    fn validate_input(&self, input: &Value) -> Result<()> {
        // Validate against the input schema
        if !input.is_object() {
            return Err(AgentError::invalid_input("Input must be an object"));
        }

        if !input.get("input").is_some() {
            return Err(AgentError::invalid_input("Missing required 'input' field"));
        }

        Ok(())
    }
}

/// Builder for creating DSPy module tools
pub struct DspyToolBuilder<I, O> {
    module: Option<Arc<dyn Module<Input = I, Output = O>>>,
    tool_name: Option<String>,
    description: Option<String>,
    input_schema: Option<serde_json::Value>,
}

impl<I, O> DspyToolBuilder<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            module: None,
            tool_name: None,
            description: None,
            input_schema: None,
        }
    }

    /// Set the DSPy module
    pub fn with_module(mut self, module: Arc<dyn Module<Input = I, Output = O>>) -> Self {
        self.module = Some(module);
        self
    }

    /// Set the tool name
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Set the tool description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the input schema
    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Build the DSPy tool
    pub fn build(self) -> DspyResult<DspyModuleTool<I, O>> {
        let module = self.module
            .ok_or_else(|| DspyError::configuration("module", "Module is required"))?;

        match (self.tool_name, self.description, self.input_schema) {
            (Some(name), Some(description), Some(schema)) => {
                Ok(DspyModuleTool::new(module, name, description, schema))
            }
            (tool_name, description, _) => {
                DspyModuleTool::from_module(module, tool_name, description)
            }
        }
    }
}

impl<I, O> Default for DspyToolBuilder<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Registry for managing DSPy tools
#[derive(Default)]
pub struct DspyToolRegistry {
    /// Registered tools by name
    tools: HashMap<String, Box<dyn Tool>>,
    /// Tool metadata
    metadata: HashMap<String, DspyToolMetadata>,
}

impl std::fmt::Debug for DspyToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DspyToolRegistry")
            .field("tools_count", &self.tools.len())
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Metadata for DSPy tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyToolMetadata {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: Option<String>,
    /// Module ID that backs this tool
    pub module_id: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Performance metrics
    pub metrics: ToolMetrics,
}

impl DspyToolRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a DSPy tool
    pub fn register_dspy_tool<I, O>(
        &mut self,
        tool: DspyModuleTool<I, O>,
        module_id: String,
    ) -> Result<()>
    where
        I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
        O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    {
        let tool_name = tool.name().to_string();
        let metadata = DspyToolMetadata {
            name: tool_name.clone(),
            description: tool.description().map(|s| s.to_string()),
            module_id,
            created_at: chrono::Utc::now(),
            metrics: tool.metrics().clone(),
        };

        self.tools.insert(tool_name.clone(), Box::new(tool));
        self.metadata.insert(tool_name.clone(), metadata);

        info!("Registered DSPy tool: {}", tool_name);
        Ok(())
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Remove a tool
    pub fn remove_tool(&mut self, name: &str) -> Option<Box<dyn Tool>> {
        self.metadata.remove(name);
        self.tools.remove(name)
    }

    /// List all tool names
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get tool metadata
    pub fn get_metadata(&self, name: &str) -> Option<&DspyToolMetadata> {
        self.metadata.get(name)
    }

    /// Get all metadata
    pub fn list_metadata(&self) -> Vec<&DspyToolMetadata> {
        self.metadata.values().collect()
    }

    /// Clear all tools
    pub fn clear(&mut self) {
        self.tools.clear();
        self.metadata.clear();
    }

    /// Get registry statistics
    pub fn stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_tools".to_string(), serde_json::Value::Number(self.tools.len().into()));
        
        let total_executions: u64 = self.metadata.values().map(|m| m.metrics.execution_count).sum();
        stats.insert("total_executions".to_string(), serde_json::Value::Number(total_executions.into()));
        
        let avg_success_rate: f64 = if !self.metadata.is_empty() {
            self.metadata.values().map(|m| m.metrics.success_rate).sum::<f64>() / self.metadata.len() as f64
        } else {
            0.0
        };
        stats.insert("average_success_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(avg_success_rate).unwrap_or_else(|| serde_json::Number::from(0))));

        stats
    }
}
