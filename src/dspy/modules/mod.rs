//! Specialized DSPy modules for advanced reasoning and problem-solving
//!
//! This module provides a collection of specialized DSPy modules that implement
//! various reasoning patterns and problem-solving approaches commonly used in
//! AI applications.

pub mod chain_of_thought;
pub mod program_of_thought;
pub mod rag;
pub mod react;
pub mod self_improving;

pub use chain_of_thought::{ChainOfThought, ChainOfThoughtConfig};
pub use program_of_thought::{
    CodeExecutionResult, ProgramOfThought, ProgramOfThoughtConfig, ProgrammingLanguage,
    SecurityRestrictions,
};
pub use rag::{RAGConfig, RAGResult, RetrievalStrategy, RetrievedDocument, RAG};
pub use react::{ReAct, ReActAction, ReActConfig, ReActStep};
pub use self_improving::{
    FeedbackEntry, FeedbackSettings, FeedbackType, ImprovementMetrics, ImprovementRecord,
    ImprovementStrategy, SelfImproving, SelfImprovingConfig,
};

use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Common configuration for specialized modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedModuleConfig {
    /// Maximum number of reasoning steps
    pub max_steps: usize,
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum tokens per step
    pub max_tokens_per_step: usize,
    /// Enable verbose logging
    pub verbose: bool,
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for SpecializedModuleConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            temperature: 0.7,
            max_tokens_per_step: 500,
            verbose: false,
            custom_params: HashMap::new(),
        }
    }
}

/// Base trait for specialized modules with reasoning capabilities
#[async_trait]
pub trait ReasoningModule<I, O>: Module<Input = I, Output = O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Get the reasoning steps taken during the last execution
    fn get_reasoning_steps(&self) -> Vec<ReasoningStep>;

    /// Get the confidence score for the last result
    fn get_confidence(&self) -> f64;

    /// Get performance metrics for the module
    fn get_performance_metrics(&self) -> ReasoningMetrics;

    /// Reset the module's internal state
    async fn reset_state(&mut self) -> DspyResult<()>;

    /// Configure the reasoning parameters
    fn configure_reasoning(&mut self, config: SpecializedModuleConfig) -> DspyResult<()>;
}

/// A single step in the reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step_number: usize,
    /// Type of reasoning step
    pub step_type: String,
    /// Input to this step
    pub input: String,
    /// Output from this step
    pub output: String,
    /// Confidence score for this step
    pub confidence: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Performance metrics for reasoning modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningMetrics {
    /// Total number of executions
    pub total_executions: u64,
    /// Average number of reasoning steps
    pub avg_reasoning_steps: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Average confidence score
    pub avg_confidence: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Last execution timestamp
    pub last_execution_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for ReasoningMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            avg_reasoning_steps: 0.0,
            avg_execution_time_ms: 0.0,
            avg_confidence: 0.0,
            success_rate: 0.0,
            successful_executions: 0,
            failed_executions: 0,
            last_execution_at: None,
            custom_metrics: HashMap::new(),
        }
    }
}

impl ReasoningMetrics {
    /// Record a successful execution
    pub fn record_success(
        &mut self,
        reasoning_steps: usize,
        execution_time_ms: f64,
        confidence: f64,
    ) {
        self.total_executions += 1;
        self.successful_executions += 1;

        // Update averages
        let total = self.total_executions as f64;
        self.avg_reasoning_steps =
            (self.avg_reasoning_steps * (total - 1.0) + reasoning_steps as f64) / total;
        self.avg_execution_time_ms =
            (self.avg_execution_time_ms * (total - 1.0) + execution_time_ms) / total;
        self.avg_confidence = (self.avg_confidence * (total - 1.0) + confidence) / total;

        self.success_rate = self.successful_executions as f64 / total;
        self.last_execution_at = Some(chrono::Utc::now());
    }

    /// Record a failed execution
    pub fn record_failure(&mut self, execution_time_ms: f64) {
        self.total_executions += 1;
        self.failed_executions += 1;

        // Update averages
        let total = self.total_executions as f64;
        self.avg_execution_time_ms =
            (self.avg_execution_time_ms * (total - 1.0) + execution_time_ms) / total;

        self.success_rate = self.successful_executions as f64 / total;
        self.last_execution_at = Some(chrono::Utc::now());
    }

    /// Add a custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Get a custom metric
    pub fn get_custom_metric(&self, name: &str) -> Option<f64> {
        self.custom_metrics.get(name).copied()
    }
}

/// Registry for specialized modules
#[derive(Debug, Default)]
pub struct SpecializedModuleRegistry {
    /// Registered modules by name
    modules: HashMap<String, Box<dyn ModuleInfo>>,
    /// Module performance metrics
    metrics: HashMap<String, ReasoningMetrics>,
}

/// Information about a specialized module
pub trait ModuleInfo: Send + Sync + std::fmt::Debug {
    /// Get module name
    fn name(&self) -> &str;
    /// Get module description
    fn description(&self) -> Option<&str>;
    /// Get module type
    fn module_type(&self) -> &str;
    /// Get supported reasoning patterns
    fn reasoning_patterns(&self) -> Vec<String>;
    /// Check if module supports a specific capability
    fn supports_capability(&self, capability: &str) -> bool;
}

impl SpecializedModuleRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a specialized module
    pub fn register_module<T: ModuleInfo + 'static>(&mut self, module: T) {
        let name = module.name().to_string();
        self.modules.insert(name.clone(), Box::new(module));
        self.metrics
            .insert(name.clone(), ReasoningMetrics::default());
        info!("Registered specialized module: {}", name);
    }

    /// Get module information
    pub fn get_module_info(&self, name: &str) -> Option<&dyn ModuleInfo> {
        self.modules.get(name).map(|m| m.as_ref())
    }

    /// Get module metrics
    pub fn get_module_metrics(&self, name: &str) -> Option<&ReasoningMetrics> {
        self.metrics.get(name)
    }

    /// Update module metrics
    pub fn update_module_metrics(&mut self, name: &str, metrics: ReasoningMetrics) {
        self.metrics.insert(name.to_string(), metrics);
    }

    /// List all registered modules
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Find modules by capability
    pub fn find_modules_by_capability(&self, capability: &str) -> Vec<&str> {
        self.modules
            .iter()
            .filter(|(_, module)| module.supports_capability(capability))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get registry statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert(
            "total_modules".to_string(),
            serde_json::Value::Number(self.modules.len().into()),
        );

        let total_executions: u64 = self.metrics.values().map(|m| m.total_executions).sum();
        stats.insert(
            "total_executions".to_string(),
            serde_json::Value::Number(total_executions.into()),
        );

        let avg_success_rate = if !self.metrics.is_empty() {
            self.metrics.values().map(|m| m.success_rate).sum::<f64>() / self.metrics.len() as f64
        } else {
            0.0
        };
        stats.insert(
            "average_success_rate".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(avg_success_rate)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );

        // Module type distribution
        let mut type_counts = HashMap::new();
        for module in self.modules.values() {
            *type_counts.entry(module.module_type()).or_insert(0) += 1;
        }
        stats.insert(
            "module_types".to_string(),
            serde_json::to_value(type_counts).unwrap_or_default(),
        );

        stats
    }

    /// Clear all modules
    pub fn clear(&mut self) {
        self.modules.clear();
        self.metrics.clear();
    }
}

/// Utility functions for specialized modules
pub mod utils {
    use super::*;

    /// Parse reasoning steps from text
    pub fn parse_reasoning_steps(text: &str) -> Vec<ReasoningStep> {
        let mut steps = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("Step") || line.trim().starts_with("Thought") {
                let step = ReasoningStep {
                    step_number: i + 1,
                    step_type: "reasoning".to_string(),
                    input: line.trim().to_string(),
                    output: lines.get(i + 1).unwrap_or(&"").trim().to_string(),
                    confidence: 0.8, // Default confidence
                    execution_time_ms: 0.0,
                    metadata: HashMap::new(),
                };
                steps.push(step);
            }
        }

        steps
    }

    /// Calculate confidence score based on reasoning quality
    pub fn calculate_confidence(reasoning_steps: &[ReasoningStep]) -> f64 {
        if reasoning_steps.is_empty() {
            return 0.0;
        }

        let avg_confidence: f64 = reasoning_steps.iter().map(|s| s.confidence).sum::<f64>()
            / reasoning_steps.len() as f64;

        // Adjust based on number of steps (more steps can indicate more thorough reasoning)
        let step_bonus = (reasoning_steps.len() as f64 * 0.05).min(0.2);

        (avg_confidence + step_bonus).min(1.0)
    }

    /// Validate reasoning chain for consistency
    pub fn validate_reasoning_chain(steps: &[ReasoningStep]) -> bool {
        if steps.is_empty() {
            return false;
        }

        // Check that steps are sequential
        for (i, step) in steps.iter().enumerate() {
            if step.step_number != i + 1 {
                return false;
            }
        }

        // Check that each step has meaningful content
        for step in steps {
            if step.input.trim().is_empty() || step.output.trim().is_empty() {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialized_module_config_default() {
        let config = SpecializedModuleConfig::default();
        assert_eq!(config.max_steps, 10);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens_per_step, 500);
        assert!(!config.verbose);
        assert!(config.custom_params.is_empty());
    }

    #[test]
    fn test_reasoning_metrics_record_success() {
        let mut metrics = ReasoningMetrics::default();

        metrics.record_success(3, 100.0, 0.9);
        assert_eq!(metrics.total_executions, 1);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.failed_executions, 0);
        assert_eq!(metrics.avg_reasoning_steps, 3.0);
        assert_eq!(metrics.avg_execution_time_ms, 100.0);
        assert_eq!(metrics.avg_confidence, 0.9);
        assert_eq!(metrics.success_rate, 1.0);

        metrics.record_success(5, 200.0, 0.8);
        assert_eq!(metrics.total_executions, 2);
        assert_eq!(metrics.successful_executions, 2);
        assert_eq!(metrics.avg_reasoning_steps, 4.0);
        assert_eq!(metrics.avg_execution_time_ms, 150.0);
        assert!((metrics.avg_confidence - 0.85).abs() < 0.001);
        assert_eq!(metrics.success_rate, 1.0);
    }

    #[test]
    fn test_reasoning_metrics_record_failure() {
        let mut metrics = ReasoningMetrics::default();

        metrics.record_success(3, 100.0, 0.9);
        metrics.record_failure(50.0);

        assert_eq!(metrics.total_executions, 2);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.failed_executions, 1);
        assert_eq!(metrics.success_rate, 0.5);
        assert_eq!(metrics.avg_execution_time_ms, 75.0);
    }

    #[test]
    fn test_reasoning_metrics_custom_metrics() {
        let mut metrics = ReasoningMetrics::default();

        metrics.add_custom_metric("accuracy".to_string(), 0.95);
        metrics.add_custom_metric("precision".to_string(), 0.88);

        assert_eq!(metrics.get_custom_metric("accuracy"), Some(0.95));
        assert_eq!(metrics.get_custom_metric("precision"), Some(0.88));
        assert_eq!(metrics.get_custom_metric("recall"), None);
    }

    #[test]
    fn test_specialized_module_registry() {
        let mut registry = SpecializedModuleRegistry::new();
        assert!(registry.list_modules().is_empty());

        let stats = registry.get_statistics();
        assert_eq!(
            stats.get("total_modules").unwrap(),
            &serde_json::Value::Number(0.into())
        );
        assert_eq!(
            stats.get("total_executions").unwrap(),
            &serde_json::Value::Number(0.into())
        );
    }

    #[test]
    fn test_utils_parse_reasoning_steps() {
        let text = "Step 1: Analyze the problem\nThis is the analysis\nStep 2: Find solution\nHere is the solution";
        let steps = utils::parse_reasoning_steps(text);

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].step_number, 1);
        assert!(steps[0].input.contains("Step 1"));
        assert_eq!(steps[1].step_number, 3);
        assert!(steps[1].input.contains("Step 2"));
    }

    #[test]
    fn test_utils_calculate_confidence() {
        let steps = vec![
            ReasoningStep {
                step_number: 1,
                step_type: "reasoning".to_string(),
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.8,
                execution_time_ms: 100.0,
                metadata: HashMap::new(),
            },
            ReasoningStep {
                step_number: 2,
                step_type: "reasoning".to_string(),
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
                execution_time_ms: 100.0,
                metadata: HashMap::new(),
            },
        ];

        let confidence = utils::calculate_confidence(&steps);
        assert!(confidence > 0.85); // Should be average + step bonus
        assert!(confidence <= 1.0);
    }

    #[test]
    fn test_utils_validate_reasoning_chain() {
        let valid_steps = vec![
            ReasoningStep {
                step_number: 1,
                step_type: "reasoning".to_string(),
                input: "valid input".to_string(),
                output: "valid output".to_string(),
                confidence: 0.8,
                execution_time_ms: 100.0,
                metadata: HashMap::new(),
            },
            ReasoningStep {
                step_number: 2,
                step_type: "reasoning".to_string(),
                input: "valid input 2".to_string(),
                output: "valid output 2".to_string(),
                confidence: 0.9,
                execution_time_ms: 100.0,
                metadata: HashMap::new(),
            },
        ];

        assert!(utils::validate_reasoning_chain(&valid_steps));

        // Test invalid chain (empty input)
        let invalid_steps = vec![ReasoningStep {
            step_number: 1,
            step_type: "reasoning".to_string(),
            input: "".to_string(),
            output: "valid output".to_string(),
            confidence: 0.8,
            execution_time_ms: 100.0,
            metadata: HashMap::new(),
        }];

        assert!(!utils::validate_reasoning_chain(&invalid_steps));

        // Test empty chain
        assert!(!utils::validate_reasoning_chain(&[]));
    }
}
