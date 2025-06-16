//! Self-Improving module
//!
//! This module implements self-improvement capabilities, allowing the module
//! to learn from its mistakes and improve its performance over time.

use super::{
    ModuleInfo, ReasoningMetrics, ReasoningModule, ReasoningStep, SpecializedModuleConfig,
};
use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    examples::{Example, ExampleSet},
    module::{Module, ModuleMetadata, ModuleStats},
    optimization::{OptimizationMetrics, OptimizationStrategy, Optimizer},
    signature::Signature,
    teleprompter::{Teleprompter, TeleprompterConfig},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for Self-Improving module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfImprovingConfig {
    /// Base configuration
    pub base: SpecializedModuleConfig,
    /// Minimum number of examples before improvement
    pub min_examples_for_improvement: usize,
    /// Improvement trigger threshold (error rate)
    pub improvement_threshold: f64,
    /// Maximum number of improvement iterations
    pub max_improvement_iterations: usize,
    /// Learning rate for improvements
    pub learning_rate: f64,
    /// Whether to use automatic improvement
    pub auto_improve: bool,
    /// Improvement strategy
    pub improvement_strategy: ImprovementStrategy,
    /// Feedback collection settings
    pub feedback_settings: FeedbackSettings,
}

impl Default for SelfImprovingConfig {
    fn default() -> Self {
        Self {
            base: SpecializedModuleConfig::default(),
            min_examples_for_improvement: 10,
            improvement_threshold: 0.3, // Improve if error rate > 30%
            max_improvement_iterations: 5,
            learning_rate: 0.1,
            auto_improve: true,
            improvement_strategy: ImprovementStrategy::GradualOptimization,
            feedback_settings: FeedbackSettings::default(),
        }
    }
}

/// Strategy for self-improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementStrategy {
    /// Gradual optimization based on feedback
    GradualOptimization,
    /// Reinforcement learning approach
    ReinforcementLearning,
    /// Meta-learning approach
    MetaLearning,
    /// Ensemble-based improvement
    EnsembleImprovement,
}

/// Settings for feedback collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSettings {
    /// Collect implicit feedback from performance
    pub collect_implicit_feedback: bool,
    /// Collect explicit feedback from users
    pub collect_explicit_feedback: bool,
    /// Weight for positive feedback
    pub positive_feedback_weight: f64,
    /// Weight for negative feedback
    pub negative_feedback_weight: f64,
    /// Maximum feedback history to keep
    pub max_feedback_history: usize,
}

impl Default for FeedbackSettings {
    fn default() -> Self {
        Self {
            collect_implicit_feedback: true,
            collect_explicit_feedback: false,
            positive_feedback_weight: 1.0,
            negative_feedback_weight: 2.0, // Negative feedback has more impact
            max_feedback_history: 1000,
        }
    }
}

/// Feedback entry for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    /// Unique feedback ID
    pub id: String,
    /// Input that was processed
    pub input: serde_json::Value,
    /// Output that was generated
    pub output: serde_json::Value,
    /// Expected/correct output
    pub expected_output: Option<serde_json::Value>,
    /// Feedback score (-1.0 to 1.0)
    pub score: f64,
    /// Feedback type
    pub feedback_type: FeedbackType,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Type of feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Implicit feedback from performance metrics
    Implicit,
    /// Explicit feedback from users
    Explicit,
    /// Automated feedback from validation
    Automated,
}

/// Metrics for improvement tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    /// Number of improvement iterations performed
    pub improvement_iterations: usize,
    /// Performance before improvement
    pub initial_performance: f64,
    /// Current performance
    pub current_performance: f64,
    /// Performance improvement percentage
    pub improvement_percentage: f64,
    /// Number of feedback entries processed
    pub feedback_entries_processed: usize,
    /// Last improvement timestamp
    pub last_improvement_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Improvement history
    pub improvement_history: Vec<ImprovementRecord>,
}

impl Default for ImprovementMetrics {
    fn default() -> Self {
        Self {
            improvement_iterations: 0,
            initial_performance: 0.0,
            current_performance: 0.0,
            improvement_percentage: 0.0,
            feedback_entries_processed: 0,
            last_improvement_at: None,
            improvement_history: Vec::new(),
        }
    }
}

/// Record of a single improvement iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementRecord {
    /// Iteration number
    pub iteration: usize,
    /// Performance before this iteration
    pub performance_before: f64,
    /// Performance after this iteration
    pub performance_after: f64,
    /// Strategy used for improvement
    pub strategy_used: String,
    /// Number of examples used
    pub examples_used: usize,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Self-Improving module
pub struct SelfImproving<I, O> {
    /// Module ID
    id: String,
    /// Module name
    name: String,
    /// Module signature
    signature: Signature<I, O>,
    /// Configuration
    config: SelfImprovingConfig,
    /// Anthropic client
    anthropic_client: Arc<AnthropicClient>,
    /// Base module to improve
    base_module: Arc<RwLock<dyn Module<Input = I, Output = O>>>,
    /// Module metadata
    metadata: ModuleMetadata,
    /// Module statistics
    stats: Arc<RwLock<ModuleStats>>,
    /// Performance metrics
    metrics: Arc<RwLock<ReasoningMetrics>>,
    /// Improvement metrics
    improvement_metrics: Arc<RwLock<ImprovementMetrics>>,
    /// Feedback history
    feedback_history: Arc<RwLock<Vec<FeedbackEntry>>>,
    /// Training examples collected
    training_examples: Arc<RwLock<ExampleSet<I, O>>>,
    /// Last reasoning steps
    last_reasoning_steps: Arc<RwLock<Vec<ReasoningStep>>>,
    /// Last confidence score
    last_confidence: Arc<RwLock<f64>>,
}

impl<I, O> std::fmt::Debug for SelfImproving<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfImproving")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("config", &self.config)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<I, O> SelfImproving<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    /// Create a new Self-Improving module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        base_module: Arc<RwLock<dyn Module<Input = I, Output = O>>>,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let name = format!("SelfImproving_{}", &id[..8]);

        Self {
            id,
            name,
            signature,
            config: SelfImprovingConfig::default(),
            anthropic_client,
            base_module,
            metadata: ModuleMetadata::default(),
            stats: Arc::new(RwLock::new(ModuleStats::default())),
            metrics: Arc::new(RwLock::new(ReasoningMetrics::default())),
            improvement_metrics: Arc::new(RwLock::new(ImprovementMetrics::default())),
            feedback_history: Arc::new(RwLock::new(Vec::new())),
            training_examples: Arc::new(RwLock::new(ExampleSet::new())),
            last_reasoning_steps: Arc::new(RwLock::new(Vec::new())),
            last_confidence: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        base_module: Arc<RwLock<dyn Module<Input = I, Output = O>>>,
        config: SelfImprovingConfig,
    ) -> Self {
        let mut module = Self::new(signature, anthropic_client, base_module);
        module.config = config;
        module
    }

    /// Add feedback for learning
    pub async fn add_feedback(
        &self,
        input: I,
        output: O,
        expected_output: Option<O>,
        score: f64,
        feedback_type: FeedbackType,
    ) -> DspyResult<()> {
        let feedback = FeedbackEntry {
            id: Uuid::new_v4().to_string(),
            input: serde_json::to_value(&input)
                .map_err(|e| DspyError::serialization("input", &e.to_string()))?,
            output: serde_json::to_value(&output)
                .map_err(|e| DspyError::serialization("output", &e.to_string()))?,
            expected_output: if let Some(expected) = expected_output {
                Some(
                    serde_json::to_value(&expected)
                        .map_err(|e| DspyError::serialization("expected_output", &e.to_string()))?,
                )
            } else {
                None
            },
            score: score.clamp(-1.0, 1.0),
            feedback_type,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        // Add to feedback history
        let mut history = self.feedback_history.write().await;
        history.push(feedback.clone());

        // Limit history size
        if history.len() > self.config.feedback_settings.max_feedback_history {
            history.remove(0);
        }

        // Add to training examples if we have expected output
        if let Some(expected) = &feedback.expected_output {
            let example = Example::new(
                input,
                serde_json::from_value(expected.clone())
                    .map_err(|e| DspyError::serialization("expected_output", &e.to_string()))?,
            );

            let mut examples = self.training_examples.write().await;
            examples.add_example(example);
        }

        // Update improvement metrics
        let mut improvement_metrics = self.improvement_metrics.write().await;
        improvement_metrics.feedback_entries_processed += 1;

        // Trigger improvement if conditions are met
        if self.config.auto_improve && self.should_trigger_improvement().await {
            drop(history);
            drop(improvement_metrics);
            self.trigger_improvement().await?;
        }

        Ok(())
    }

    /// Check if improvement should be triggered
    async fn should_trigger_improvement(&self) -> bool {
        let history = self.feedback_history.read().await;

        if history.len() < self.config.min_examples_for_improvement {
            return false;
        }

        // Calculate recent error rate
        let recent_feedback: Vec<_> = history
            .iter()
            .rev()
            .take(self.config.min_examples_for_improvement)
            .collect();

        let error_rate = recent_feedback.iter().filter(|f| f.score < 0.0).count() as f64
            / recent_feedback.len() as f64;

        error_rate > self.config.improvement_threshold
    }

    /// Trigger improvement process
    pub async fn trigger_improvement(&self) -> DspyResult<ImprovementMetrics> {
        info!("Triggering self-improvement process");

        let mut improvement_metrics = self.improvement_metrics.write().await;
        let current_iteration = improvement_metrics.improvement_iterations;

        if current_iteration >= self.config.max_improvement_iterations {
            warn!("Maximum improvement iterations reached");
            return Ok(improvement_metrics.clone());
        }

        // Get current performance
        let performance_before = self.calculate_current_performance().await;

        // Perform improvement based on strategy
        let improvement_result = match self.config.improvement_strategy {
            ImprovementStrategy::GradualOptimization => self.perform_gradual_optimization().await?,
            ImprovementStrategy::ReinforcementLearning => {
                self.perform_reinforcement_learning().await?
            }
            ImprovementStrategy::MetaLearning => self.perform_meta_learning().await?,
            ImprovementStrategy::EnsembleImprovement => self.perform_ensemble_improvement().await?,
        };

        // Get performance after improvement
        let performance_after = self.calculate_current_performance().await;

        // Record improvement
        let improvement_record = ImprovementRecord {
            iteration: current_iteration + 1,
            performance_before,
            performance_after,
            strategy_used: format!("{:?}", self.config.improvement_strategy),
            examples_used: improvement_result.examples_used,
            timestamp: chrono::Utc::now(),
        };

        improvement_metrics.improvement_iterations += 1;
        improvement_metrics.current_performance = performance_after;
        improvement_metrics.improvement_percentage = ((performance_after
            - improvement_metrics.initial_performance)
            / improvement_metrics.initial_performance)
            * 100.0;
        improvement_metrics.last_improvement_at = Some(chrono::Utc::now());
        improvement_metrics
            .improvement_history
            .push(improvement_record);

        if improvement_metrics.initial_performance == 0.0 {
            improvement_metrics.initial_performance = performance_before;
        }

        info!(
            "Self-improvement completed. Performance: {:.3} -> {:.3} ({:+.1}%)",
            performance_before,
            performance_after,
            ((performance_after - performance_before) / performance_before) * 100.0
        );

        Ok(improvement_metrics.clone())
    }

    /// Calculate current performance based on recent feedback
    async fn calculate_current_performance(&self) -> f64 {
        let history = self.feedback_history.read().await;

        if history.is_empty() {
            return 0.5; // Neutral performance
        }

        let recent_feedback: Vec<_> = history
            .iter()
            .rev()
            .take(20) // Use last 20 feedback entries
            .collect();

        let avg_score =
            recent_feedback.iter().map(|f| f.score).sum::<f64>() / recent_feedback.len() as f64;

        // Convert from [-1, 1] to [0, 1]
        (avg_score + 1.0) / 2.0
    }

    /// Perform gradual optimization improvement
    async fn perform_gradual_optimization(&self) -> DspyResult<ImprovementResult> {
        let examples = self.training_examples.read().await;
        let examples_count = examples.len();

        if examples_count < self.config.min_examples_for_improvement {
            return Ok(ImprovementResult {
                success: false,
                examples_used: examples_count,
                improvement_score: 0.0,
            });
        }

        // Create teleprompter for optimization
        let teleprompter_config = TeleprompterConfig {
            strategy: OptimizationStrategy::bootstrap(5, 10),
            max_iterations: 10,
            convergence_threshold: 0.01,
            min_improvement: 0.001,
            early_stopping_patience: 3,
            validation_split: 0.2,
            use_cross_validation: false,
            cv_folds: 5,
            random_seed: None,
            verbose: false,
            custom_params: HashMap::new(),
        };

        let mut teleprompter = Teleprompter::with_config(teleprompter_config);

        // Optimize the base module
        let mut base_module = self.base_module.write().await;
        let optimization_result = teleprompter
            .optimize(&mut *base_module, examples.clone())
            .await?;

        Ok(ImprovementResult {
            success: true,
            examples_used: examples_count,
            improvement_score: optimization_result.metrics.best_score,
        })
    }

    /// Perform reinforcement learning improvement
    async fn perform_reinforcement_learning(&self) -> DspyResult<ImprovementResult> {
        // TODO: Implement reinforcement learning approach
        // For now, return a placeholder result
        Ok(ImprovementResult {
            success: true,
            examples_used: 0,
            improvement_score: 0.1,
        })
    }

    /// Perform meta-learning improvement
    async fn perform_meta_learning(&self) -> DspyResult<ImprovementResult> {
        // TODO: Implement meta-learning approach
        // For now, return a placeholder result
        Ok(ImprovementResult {
            success: true,
            examples_used: 0,
            improvement_score: 0.1,
        })
    }

    /// Perform ensemble improvement
    async fn perform_ensemble_improvement(&self) -> DspyResult<ImprovementResult> {
        // TODO: Implement ensemble-based improvement
        // For now, return a placeholder result
        Ok(ImprovementResult {
            success: true,
            examples_used: 0,
            improvement_score: 0.1,
        })
    }

    /// Get improvement metrics
    pub async fn get_improvement_metrics(&self) -> ImprovementMetrics {
        self.improvement_metrics.read().await.clone()
    }

    /// Get feedback history
    pub async fn get_feedback_history(&self) -> Vec<FeedbackEntry> {
        self.feedback_history.read().await.clone()
    }

    /// Reset improvement state
    pub async fn reset_improvement_state(&self) -> DspyResult<()> {
        *self.improvement_metrics.write().await = ImprovementMetrics::default();
        self.feedback_history.write().await.clear();
        *self.training_examples.write().await = ExampleSet::new();
        Ok(())
    }
}

/// Result of an improvement operation
#[derive(Debug, Clone)]
struct ImprovementResult {
    /// Whether improvement was successful
    success: bool,
    /// Number of examples used for improvement
    examples_used: usize,
    /// Improvement score achieved
    improvement_score: f64,
}

#[async_trait]
impl<I, O> Module for SelfImproving<I, O>
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

        // Forward to base module
        let base_module = self.base_module.read().await;
        let result = base_module.forward(input.clone()).await;

        match &result {
            Ok(output) => {
                // Collect implicit feedback based on execution success
                if self.config.feedback_settings.collect_implicit_feedback {
                    let _ = self
                        .add_feedback(
                            input,
                            output.clone(),
                            None,
                            0.5, // Neutral score for successful execution
                            FeedbackType::Implicit,
                        )
                        .await;
                }

                // Update metrics
                let execution_time = start_time.elapsed().as_millis() as f64;
                let mut metrics = self.metrics.write().await;
                metrics.record_success(1, execution_time, 0.8);
            }
            Err(_) => {
                // Collect negative implicit feedback for failures
                if self.config.feedback_settings.collect_implicit_feedback {
                    // We can't add feedback without a valid output, so we skip this for failures
                    // In a real implementation, you might want to track failure patterns differently
                }

                // Update metrics
                let execution_time = start_time.elapsed().as_millis() as f64;
                let mut metrics = self.metrics.write().await;
                metrics.record_failure(execution_time);
            }
        }

        result
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
impl<I, O> ReasoningModule<I, O> for SelfImproving<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    fn get_reasoning_steps(&self) -> Vec<ReasoningStep> {
        Vec::new() // Simplified for now
    }

    fn get_confidence(&self) -> f64 {
        0.0 // Simplified for now
    }

    fn get_performance_metrics(&self) -> ReasoningMetrics {
        ReasoningMetrics::default() // Simplified for now
    }

    async fn reset_state(&mut self) -> DspyResult<()> {
        *self.last_reasoning_steps.write().await = Vec::new();
        *self.last_confidence.write().await = 0.0;
        self.reset_improvement_state().await
    }

    fn configure_reasoning(&mut self, config: SpecializedModuleConfig) -> DspyResult<()> {
        self.config.base = config;
        Ok(())
    }
}

impl ModuleInfo for SelfImproving<(), ()> {
    fn name(&self) -> &str {
        "SelfImproving"
    }

    fn description(&self) -> Option<&str> {
        Some("Self-improving module that learns from feedback and optimizes its performance over time")
    }

    fn module_type(&self) -> &str {
        "self_improving"
    }

    fn reasoning_patterns(&self) -> Vec<String> {
        vec![
            "self_improvement".to_string(),
            "adaptive_learning".to_string(),
            "feedback_driven".to_string(),
        ]
    }

    fn supports_capability(&self, capability: &str) -> bool {
        matches!(
            capability,
            "self_improvement"
                | "adaptive_learning"
                | "feedback_processing"
                | "continuous_learning"
        )
    }
}
