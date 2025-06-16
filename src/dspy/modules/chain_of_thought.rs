//! Chain of Thought (CoT) reasoning module
//!
//! This module implements the Chain of Thought reasoning pattern, which encourages
//! the model to break down complex problems into step-by-step reasoning chains.

use super::{
    ModuleInfo, ReasoningMetrics, ReasoningModule, ReasoningStep, SpecializedModuleConfig,
};
use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for Chain of Thought module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfThoughtConfig {
    /// Base configuration
    pub base: SpecializedModuleConfig,
    /// Reasoning prompt template
    pub reasoning_template: String,
    /// Whether to include step numbers
    pub include_step_numbers: bool,
    /// Whether to validate reasoning chain
    pub validate_chain: bool,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum retries for low confidence
    pub max_retries: usize,
}

impl Default for ChainOfThoughtConfig {
    fn default() -> Self {
        Self {
            base: SpecializedModuleConfig::default(),
            reasoning_template:
                "Let's think step by step:\n\n{reasoning}\n\nTherefore, the answer is: {answer}"
                    .to_string(),
            include_step_numbers: true,
            validate_chain: true,
            min_confidence: 0.7,
            max_retries: 2,
        }
    }
}

/// Chain of Thought reasoning module
#[derive(Debug)]
pub struct ChainOfThought<I, O> {
    /// Module ID
    id: String,
    /// Module name
    name: String,
    /// Module signature
    signature: Signature<I, O>,
    /// Configuration
    config: ChainOfThoughtConfig,
    /// Anthropic client
    anthropic_client: Arc<AnthropicClient>,
    /// Module metadata
    metadata: ModuleMetadata,
    /// Module statistics
    stats: Arc<RwLock<ModuleStats>>,
    /// Performance metrics
    metrics: Arc<RwLock<ReasoningMetrics>>,
    /// Last reasoning steps
    last_reasoning_steps: Arc<RwLock<Vec<ReasoningStep>>>,
    /// Last confidence score
    last_confidence: Arc<RwLock<f64>>,
}

impl<I, O> ChainOfThought<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    /// Create a new Chain of Thought module
    pub fn new(signature: Signature<I, O>, anthropic_client: Arc<AnthropicClient>) -> Self {
        let id = Uuid::new_v4().to_string();
        let name = format!("ChainOfThought_{}", &id[..8]);

        Self {
            id,
            name,
            signature,
            config: ChainOfThoughtConfig::default(),
            anthropic_client,
            metadata: ModuleMetadata::default(),
            stats: Arc::new(RwLock::new(ModuleStats::default())),
            metrics: Arc::new(RwLock::new(ReasoningMetrics::default())),
            last_reasoning_steps: Arc::new(RwLock::new(Vec::new())),
            last_confidence: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ChainOfThoughtConfig,
    ) -> Self {
        let mut module = Self::new(signature, anthropic_client);
        module.config = config;
        module
    }

    /// Get the current configuration
    pub fn config(&self) -> &ChainOfThoughtConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ChainOfThoughtConfig) {
        self.config = config;
    }

    /// Generate reasoning prompt
    fn generate_reasoning_prompt(&self, input: &I) -> DspyResult<String> {
        let input_str = serde_json::to_string_pretty(input)
            .map_err(|e| DspyError::serialization("input", &e.to_string()))?;

        let mut prompt = format!(
            "Given the following input, please provide a step-by-step reasoning process:\n\nInput: {}\n\n",
            input_str
        );

        if self.config.include_step_numbers {
            prompt.push_str("Please number each step clearly (Step 1:, Step 2:, etc.).\n\n");
        }

        prompt.push_str(&self.config.reasoning_template);

        Ok(prompt)
    }

    /// Parse reasoning from response
    fn parse_reasoning(&self, response: &str) -> Vec<ReasoningStep> {
        let mut steps = Vec::new();
        let lines: Vec<&str> = response.lines().collect();
        let mut current_step = 1;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Look for step indicators
            if trimmed.starts_with(&format!("Step {}:", current_step))
                || trimmed.starts_with(&format!("{}.", current_step))
                || (trimmed.starts_with("Step") && trimmed.contains(&current_step.to_string()))
            {
                let input = trimmed.to_string();
                let output = lines
                    .get(i + 1)
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default();

                let step = ReasoningStep {
                    step_number: current_step,
                    step_type: "chain_of_thought".to_string(),
                    input,
                    output,
                    confidence: 0.8, // Default confidence, can be improved with analysis
                    execution_time_ms: 0.0,
                    metadata: HashMap::new(),
                };

                steps.push(step);
                current_step += 1;
            }
        }

        // If no numbered steps found, try to extract reasoning from paragraphs
        if steps.is_empty() {
            let paragraphs: Vec<&str> = response.split("\n\n").collect();
            for (i, paragraph) in paragraphs.iter().enumerate() {
                if !paragraph.trim().is_empty() && paragraph.len() > 20 {
                    let step = ReasoningStep {
                        step_number: i + 1,
                        step_type: "chain_of_thought".to_string(),
                        input: format!("Reasoning paragraph {}", i + 1),
                        output: paragraph.trim().to_string(),
                        confidence: 0.7,
                        execution_time_ms: 0.0,
                        metadata: HashMap::new(),
                    };
                    steps.push(step);
                }
            }
        }

        steps
    }

    /// Extract final answer from reasoning
    fn extract_answer(&self, response: &str, reasoning_steps: &[ReasoningStep]) -> DspyResult<O> {
        // Look for answer indicators
        let answer_indicators = [
            "Therefore, the answer is:",
            "The answer is:",
            "Answer:",
            "Final answer:",
            "Conclusion:",
        ];

        for indicator in &answer_indicators {
            if let Some(pos) = response.find(indicator) {
                let answer_text = &response[pos + indicator.len()..].trim();

                // Try to parse as JSON first
                if let Ok(answer) = serde_json::from_str::<O>(answer_text) {
                    return Ok(answer);
                }

                // If JSON parsing fails, try to extract from the text
                if let Ok(answer) = serde_json::from_str::<O>(&format!("\"{}\"", answer_text)) {
                    return Ok(answer);
                }
            }
        }

        // If no explicit answer found, try to use the last reasoning step
        if let Some(last_step) = reasoning_steps.last() {
            if let Ok(answer) = serde_json::from_str::<O>(&last_step.output) {
                return Ok(answer);
            }

            if let Ok(answer) = serde_json::from_str::<O>(&format!("\"{}\"", last_step.output)) {
                return Ok(answer);
            }
        }

        // Fallback: try to parse the entire response
        if let Ok(answer) = serde_json::from_str::<O>(response) {
            return Ok(answer);
        }

        Err(DspyError::module(
            self.name(),
            "Could not extract valid answer from reasoning chain",
        ))
    }

    /// Validate reasoning chain quality
    fn validate_reasoning_quality(&self, steps: &[ReasoningStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let mut quality_score = 0.0;
        let mut factors = 0;

        // Factor 1: Number of steps (more steps can indicate thorough reasoning)
        let step_score = (steps.len() as f64 / self.config.base.max_steps as f64).min(1.0);
        quality_score += step_score * 0.3;
        factors += 1;

        // Factor 2: Average step length (longer steps might indicate more detailed reasoning)
        let avg_length =
            steps.iter().map(|s| s.output.len()).sum::<usize>() as f64 / steps.len() as f64;
        let length_score = (avg_length / 100.0).min(1.0); // Normalize to 100 chars
        quality_score += length_score * 0.2;
        factors += 1;

        // Factor 3: Sequential consistency
        let mut sequential_score = 1.0;
        for (i, step) in steps.iter().enumerate() {
            if step.step_number != i + 1 {
                sequential_score *= 0.8;
            }
        }
        quality_score += sequential_score * 0.2;
        factors += 1;

        // Factor 4: Content quality (basic heuristics)
        let mut content_score = 0.0;
        for step in steps {
            let mut step_score = 0.5f64; // Base score

            // Check for reasoning keywords
            let reasoning_keywords = [
                "because",
                "therefore",
                "since",
                "thus",
                "hence",
                "so",
                "given",
            ];
            for keyword in &reasoning_keywords {
                if step.output.to_lowercase().contains(keyword) {
                    step_score += 0.1;
                }
            }

            content_score += step_score.min(1.0f64);
        }
        content_score /= steps.len() as f64;
        quality_score += content_score * 0.3;
        factors += 1;

        quality_score / factors as f64
    }
}

#[async_trait]
impl<I, O> Module for ChainOfThought<I, O>
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

        // Generate reasoning prompt
        let prompt = self.generate_reasoning_prompt(&input)?;

        // Execute reasoning with retries
        let mut attempts = 0;
        let mut best_result = None;
        let mut best_confidence = 0.0;

        while attempts <= self.config.max_retries {
            attempts += 1;

            // TODO: Call Anthropic API with the prompt
            // For now, simulate a response
            let response = format!(
                "Step 1: Analyze the input\nI need to carefully examine the given input.\n\nStep 2: Apply reasoning\nBased on the analysis, I can determine the appropriate response.\n\nStep 3: Formulate answer\nConsidering all factors, the answer is clear.\n\nTherefore, the answer is: {{\"result\": \"processed\"}}"
            );

            // Parse reasoning steps
            let reasoning_steps = self.parse_reasoning(&response);

            // Validate reasoning quality
            let confidence = self.validate_reasoning_quality(&reasoning_steps);

            if confidence >= self.config.min_confidence || attempts > self.config.max_retries {
                // Extract answer
                match self.extract_answer(&response, &reasoning_steps) {
                    Ok(answer) => {
                        // Update state
                        *self.last_reasoning_steps.write().await = reasoning_steps.clone();
                        *self.last_confidence.write().await = confidence;

                        // Update metrics
                        let execution_time = start_time.elapsed().as_millis() as f64;
                        let mut metrics = self.metrics.write().await;
                        metrics.record_success(reasoning_steps.len(), execution_time, confidence);

                        info!(
                            "Chain of Thought completed successfully in {} attempts with confidence {:.3}",
                            attempts, confidence
                        );

                        return Ok(answer);
                    }
                    Err(e) => {
                        if confidence > best_confidence {
                            best_confidence = confidence;
                            best_result = Some(e);
                        }
                    }
                }
            }

            if attempts <= self.config.max_retries {
                debug!("Retrying Chain of Thought reasoning (attempt {})", attempts);
            }
        }

        // If all attempts failed, record failure and return error
        let execution_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.record_failure(execution_time);

        Err(best_result.unwrap_or_else(|| {
            DspyError::module(
                self.name(),
                "Failed to generate valid reasoning chain after all retries",
            )
        }))
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }

    fn stats(&self) -> &ModuleStats {
        // Note: This is a simplified implementation
        // In a real implementation, we'd need to handle the async nature properly
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }

    fn supports_compilation(&self) -> bool {
        true
    }

    async fn compile(&mut self, examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        info!(
            "Compiling Chain of Thought module with {} examples",
            examples.len()
        );

        // Analyze examples to improve reasoning template
        if !examples.is_empty() {
            // TODO: Implement example-based optimization
            // This could involve analyzing successful reasoning patterns
            // and updating the reasoning template accordingly
        }

        Ok(())
    }
}

#[async_trait]
impl<I, O> ReasoningModule<I, O> for ChainOfThought<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    fn get_reasoning_steps(&self) -> Vec<ReasoningStep> {
        // Note: This is a simplified synchronous version
        // In practice, you'd want to handle this differently
        Vec::new()
    }

    fn get_confidence(&self) -> f64 {
        // Note: This is a simplified synchronous version
        0.0
    }

    fn get_performance_metrics(&self) -> ReasoningMetrics {
        // Note: This is a simplified synchronous version
        ReasoningMetrics::default()
    }

    async fn reset_state(&mut self) -> DspyResult<()> {
        *self.last_reasoning_steps.write().await = Vec::new();
        *self.last_confidence.write().await = 0.0;
        Ok(())
    }

    fn configure_reasoning(&mut self, config: SpecializedModuleConfig) -> DspyResult<()> {
        self.config.base = config;
        Ok(())
    }
}

impl ModuleInfo for ChainOfThought<(), ()> {
    fn name(&self) -> &str {
        "ChainOfThought"
    }

    fn description(&self) -> Option<&str> {
        Some("Chain of Thought reasoning module that breaks down complex problems into step-by-step reasoning chains")
    }

    fn module_type(&self) -> &str {
        "reasoning"
    }

    fn reasoning_patterns(&self) -> Vec<String> {
        vec![
            "step_by_step".to_string(),
            "sequential_reasoning".to_string(),
            "logical_chain".to_string(),
        ]
    }

    fn supports_capability(&self, capability: &str) -> bool {
        matches!(
            capability,
            "reasoning" | "step_by_step" | "logical_thinking" | "problem_decomposition"
        )
    }
}
