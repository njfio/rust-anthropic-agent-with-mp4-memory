//! ReAct (Reasoning and Acting) module
//!
//! This module implements the ReAct pattern, which interleaves reasoning and acting
//! to solve complex tasks that require both thinking and tool usage.

use super::{
    ModuleInfo, ReasoningMetrics, ReasoningModule, ReasoningStep, SpecializedModuleConfig,
};
use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use crate::tools::{Tool, ToolResult};
use crate::utils::error::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for ReAct module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActConfig {
    /// Base configuration
    pub base: SpecializedModuleConfig,
    /// Available tools for actions
    pub available_tools: Vec<String>,
    /// Maximum number of thought-action cycles
    pub max_cycles: usize,
    /// Whether to validate actions before execution
    pub validate_actions: bool,
    /// Timeout for tool execution in seconds
    pub tool_timeout_seconds: u64,
    /// Whether to continue on tool errors
    pub continue_on_tool_error: bool,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self {
            base: SpecializedModuleConfig::default(),
            available_tools: Vec::new(),
            max_cycles: 5,
            validate_actions: true,
            tool_timeout_seconds: 30,
            continue_on_tool_error: false,
        }
    }
}

/// A single step in the ReAct process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    /// Step number
    pub step_number: usize,
    /// Thought process
    pub thought: String,
    /// Action to take
    pub action: Option<ReActAction>,
    /// Observation from action
    pub observation: Option<String>,
    /// Confidence in this step
    pub confidence: f64,
    /// Execution time for this step
    pub execution_time_ms: f64,
}

/// An action in the ReAct process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActAction {
    /// Tool name to use
    pub tool: String,
    /// Input for the tool
    pub input: serde_json::Value,
    /// Expected output type
    pub expected_output: Option<String>,
}

/// ReAct reasoning and acting module
pub struct ReAct<I, O> {
    /// Module ID
    id: String,
    /// Module name
    name: String,
    /// Module signature
    signature: Signature<I, O>,
    /// Configuration
    config: ReActConfig,
    /// Anthropic client
    anthropic_client: Arc<AnthropicClient>,
    /// Available tools
    tools: HashMap<String, Box<dyn Tool>>,
    /// Module metadata
    metadata: ModuleMetadata,
    /// Module statistics
    stats: Arc<RwLock<ModuleStats>>,
    /// Performance metrics
    metrics: Arc<RwLock<ReasoningMetrics>>,
    /// Last ReAct steps
    last_react_steps: Arc<RwLock<Vec<ReActStep>>>,
    /// Last confidence score
    last_confidence: Arc<RwLock<f64>>,
}

impl<I, O> std::fmt::Debug for ReAct<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReAct")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("config", &self.config)
            .field("tools_count", &self.tools.len())
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<I, O> ReAct<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    /// Create a new ReAct module
    pub fn new(signature: Signature<I, O>, anthropic_client: Arc<AnthropicClient>) -> Self {
        let id = Uuid::new_v4().to_string();
        let name = format!("ReAct_{}", &id[..8]);

        Self {
            id,
            name,
            signature,
            config: ReActConfig::default(),
            anthropic_client,
            tools: HashMap::new(),
            metadata: ModuleMetadata::default(),
            stats: Arc::new(RwLock::new(ModuleStats::default())),
            metrics: Arc::new(RwLock::new(ReasoningMetrics::default())),
            last_react_steps: Arc::new(RwLock::new(Vec::new())),
            last_confidence: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ReActConfig,
    ) -> Self {
        let mut module = Self::new(signature, anthropic_client);
        module.config = config;
        module
    }

    /// Add a tool to the module
    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        let tool_name = tool.name().to_string();
        self.tools.insert(tool_name.clone(), Box::new(tool));
        if !self.config.available_tools.contains(&tool_name) {
            self.config.available_tools.push(tool_name);
        }
    }

    /// Get available tools
    pub fn get_available_tools(&self) -> &[String] {
        &self.config.available_tools
    }

    /// Generate ReAct prompt
    fn generate_react_prompt(&self, input: &I, previous_steps: &[ReActStep]) -> DspyResult<String> {
        let input_str = serde_json::to_string_pretty(input)
            .map_err(|e| DspyError::serialization("input", &e.to_string()))?;

        let mut prompt =
            format!("You are solving a task using the ReAct (Reasoning and Acting) approach.\n\n");

        prompt.push_str(&format!("Task: {}\n\n", input_str));

        if !self.config.available_tools.is_empty() {
            prompt.push_str("Available tools:\n");
            for tool in &self.config.available_tools {
                prompt.push_str(&format!("- {}\n", tool));
            }
            prompt.push('\n');
        }

        prompt.push_str(
            "Please follow this format:\n\
            Thought: [your reasoning about what to do next]\n\
            Action: [tool_name] [input_json]\n\
            Observation: [result from the action]\n\n",
        );

        // Add previous steps if any
        if !previous_steps.is_empty() {
            prompt.push_str("Previous steps:\n");
            for step in previous_steps {
                prompt.push_str(&format!("Thought: {}\n", step.thought));
                if let Some(action) = &step.action {
                    prompt.push_str(&format!(
                        "Action: {} {}\n",
                        action.tool,
                        serde_json::to_string(&action.input).unwrap_or_default()
                    ));
                }
                if let Some(observation) = &step.observation {
                    prompt.push_str(&format!("Observation: {}\n", observation));
                }
                prompt.push('\n');
            }
        }

        prompt.push_str("Now continue with the next step:\nThought:");

        Ok(prompt)
    }

    /// Parse ReAct response
    fn parse_react_response(&self, response: &str, step_number: usize) -> DspyResult<ReActStep> {
        let lines: Vec<&str> = response.lines().collect();
        let mut thought = String::new();
        let mut action = None;
        let mut observation = None;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("Thought:") {
                thought = trimmed[8..].trim().to_string();
            } else if trimmed.starts_with("Action:") {
                let action_str = trimmed[7..].trim();
                action = self.parse_action(action_str)?;
            } else if trimmed.starts_with("Observation:") {
                observation = Some(trimmed[12..].trim().to_string());
            }
        }

        if thought.is_empty() {
            return Err(DspyError::module(
                self.name(),
                "No thought found in response",
            ));
        }

        Ok(ReActStep {
            step_number,
            thought,
            action,
            observation,
            confidence: 0.8, // Default confidence
            execution_time_ms: 0.0,
        })
    }

    /// Parse action from text
    fn parse_action(&self, action_str: &str) -> DspyResult<Option<ReActAction>> {
        let parts: Vec<&str> = action_str.splitn(2, ' ').collect();
        if parts.len() < 2 {
            return Ok(None);
        }

        let tool_name = parts[0].to_string();
        let input_str = parts[1];

        // Validate tool exists
        if !self.config.available_tools.contains(&tool_name) {
            return Err(DspyError::module(
                &self.name,
                &format!("Unknown tool: {}", tool_name),
            ));
        }

        // Parse input JSON
        let input = serde_json::from_str(input_str).map_err(|e| {
            DspyError::module(&self.name, &format!("Invalid action input JSON: {}", e))
        })?;

        Ok(Some(ReActAction {
            tool: tool_name,
            input,
            expected_output: None,
        }))
    }

    /// Execute an action
    async fn execute_action(&self, action: &ReActAction) -> DspyResult<String> {
        let tool = self.tools.get(&action.tool).ok_or_else(|| {
            DspyError::module(&self.name, &format!("Tool not found: {}", action.tool))
        })?;

        // Validate input if configured
        if self.config.validate_actions {
            tool.validate_input(&action.input).map_err(|e| {
                DspyError::module(&self.name, &format!("Action validation failed: {}", e))
            })?;
        }

        // Execute with timeout
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.tool_timeout_seconds),
            tool.execute(action.input.clone()),
        )
        .await;

        match result {
            Ok(Ok(tool_result)) => {
                if tool_result.is_error && !self.config.continue_on_tool_error {
                    return Err(DspyError::module(
                        &self.name,
                        &format!("Tool execution failed: {}", tool_result.content),
                    ));
                }
                Ok(tool_result.content)
            }
            Ok(Err(e)) => {
                if self.config.continue_on_tool_error {
                    Ok(format!("Tool error: {}", e))
                } else {
                    Err(DspyError::module(
                        &self.name,
                        &format!("Tool execution error: {}", e),
                    ))
                }
            }
            Err(_) => {
                let error_msg = format!(
                    "Tool execution timeout after {} seconds",
                    self.config.tool_timeout_seconds
                );
                if self.config.continue_on_tool_error {
                    Ok(error_msg)
                } else {
                    Err(DspyError::module(&self.name, &error_msg))
                }
            }
        }
    }

    /// Check if task is complete
    fn is_task_complete(&self, steps: &[ReActStep], _input: &I) -> bool {
        // Simple heuristic: task is complete if the last thought indicates completion
        if let Some(last_step) = steps.last() {
            let thought_lower = last_step.thought.to_lowercase();
            return thought_lower.contains("complete")
                || thought_lower.contains("finished")
                || thought_lower.contains("done")
                || thought_lower.contains("final answer");
        }
        false
    }

    /// Extract final answer from ReAct steps
    fn extract_final_answer(&self, steps: &[ReActStep]) -> DspyResult<O> {
        // Look for answer in the last few steps
        for step in steps.iter().rev().take(3) {
            // Check thought for answer
            if let Ok(answer) = serde_json::from_str::<O>(&step.thought) {
                return Ok(answer);
            }

            // Check observation for answer
            if let Some(observation) = &step.observation {
                if let Ok(answer) = serde_json::from_str::<O>(observation) {
                    return Ok(answer);
                }
            }
        }

        // Fallback: try to construct answer from the final thought
        if let Some(last_step) = steps.last() {
            if let Ok(answer) = serde_json::from_str::<O>(&format!("\"{}\"", last_step.thought)) {
                return Ok(answer);
            }
        }

        Err(DspyError::module(
            self.name(),
            "Could not extract final answer from ReAct steps",
        ))
    }

    /// Calculate confidence based on ReAct execution
    fn calculate_confidence(&self, steps: &[ReActStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let mut total_confidence = 0.0;
        let mut successful_actions = 0;
        let mut total_actions = 0;

        for step in steps {
            total_confidence += step.confidence;

            if step.action.is_some() {
                total_actions += 1;
                if step.observation.is_some() {
                    successful_actions += 1;
                }
            }
        }

        let avg_step_confidence = total_confidence / steps.len() as f64;
        let action_success_rate = if total_actions > 0 {
            successful_actions as f64 / total_actions as f64
        } else {
            1.0 // No actions means no failures
        };

        // Combine step confidence and action success rate
        (avg_step_confidence * 0.7) + (action_success_rate * 0.3)
    }
}

#[async_trait]
impl<I, O> Module for ReAct<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    type Input = I;
    type Output = O;

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.signature
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        let start_time = std::time::Instant::now();
        let mut steps = Vec::new();

        for cycle in 1..=self.config.max_cycles {
            // Generate prompt for current cycle
            let prompt = self.generate_react_prompt(&input, &steps)?;

            // TODO: Call Anthropic API with the prompt
            // For now, simulate a response
            let response = format!(
                "Thought: I need to analyze this input and determine the best approach.\n\
                Action: analyze {}\n\
                Observation: Analysis complete, proceeding with solution.",
                serde_json::to_string(&input).unwrap_or_default()
            );

            // Parse the response
            let mut step = self.parse_react_response(&response, cycle)?;
            let step_start = std::time::Instant::now();

            // Execute action if present
            if let Some(action) = &step.action {
                match self.execute_action(action).await {
                    Ok(observation) => {
                        step.observation = Some(observation);
                    }
                    Err(e) => {
                        if !self.config.continue_on_tool_error {
                            return Err(e);
                        }
                        step.observation = Some(format!("Error: {}", e));
                    }
                }
            }

            step.execution_time_ms = step_start.elapsed().as_millis() as f64;
            steps.push(step);

            // Check if task is complete
            if self.is_task_complete(&steps, &input) {
                break;
            }
        }

        // Calculate final confidence and extract answer
        let confidence = self.calculate_confidence(&steps);
        let answer = self.extract_final_answer(&steps)?;

        // Update state
        *self.last_react_steps.write().await = steps.clone();
        *self.last_confidence.write().await = confidence;

        // Update metrics
        let execution_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.record_success(steps.len(), execution_time, confidence);

        info!(
            "ReAct completed successfully with {} steps and confidence {:.3}",
            steps.len(),
            confidence
        );

        Ok(answer)
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }

    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }

    fn supports_compilation(&self) -> bool {
        true
    }
}

#[async_trait]
impl<I, O> ReasoningModule<I, O> for ReAct<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    fn get_reasoning_steps(&self) -> Vec<ReasoningStep> {
        // Convert ReAct steps to reasoning steps
        Vec::new() // Simplified for now
    }

    fn get_confidence(&self) -> f64 {
        0.0 // Simplified for now
    }

    fn get_performance_metrics(&self) -> ReasoningMetrics {
        ReasoningMetrics::default() // Simplified for now
    }

    async fn reset_state(&mut self) -> DspyResult<()> {
        *self.last_react_steps.write().await = Vec::new();
        *self.last_confidence.write().await = 0.0;
        Ok(())
    }

    fn configure_reasoning(&mut self, config: SpecializedModuleConfig) -> DspyResult<()> {
        self.config.base = config;
        Ok(())
    }
}

impl ModuleInfo for ReAct<(), ()> {
    fn name(&self) -> &str {
        "ReAct"
    }

    fn description(&self) -> Option<&str> {
        Some("ReAct (Reasoning and Acting) module that interleaves reasoning and tool usage to solve complex tasks")
    }

    fn module_type(&self) -> &str {
        "reasoning_acting"
    }

    fn reasoning_patterns(&self) -> Vec<String> {
        vec![
            "reasoning_acting".to_string(),
            "tool_usage".to_string(),
            "iterative_problem_solving".to_string(),
        ]
    }

    fn supports_capability(&self, capability: &str) -> bool {
        matches!(
            capability,
            "reasoning" | "acting" | "tool_usage" | "iterative_solving" | "multi_step"
        )
    }
}
