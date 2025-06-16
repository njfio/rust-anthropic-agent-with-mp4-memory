//! Teleprompter system for DSPy module optimization
//!
//! Teleprompters are responsible for optimizing DSPy modules by improving their
//! prompts and selecting better examples for training.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::{Example, ExampleSet};
use crate::dspy::module::Module;
use crate::dspy::optimization::{OptimizationMetrics, OptimizationStrategy, Optimizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Configuration for teleprompter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleprompterConfig {
    /// Optimization strategy to use
    pub strategy: OptimizationStrategy,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold for stopping optimization
    pub convergence_threshold: f64,
    /// Minimum improvement threshold to continue optimization
    pub min_improvement: f64,
    /// Early stopping patience (iterations without improvement)
    pub early_stopping_patience: usize,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f64,
    /// Whether to use cross-validation
    pub use_cross_validation: bool,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Verbose logging
    pub verbose: bool,
    /// Custom optimization parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for TeleprompterConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::random_sampling(10),
            max_iterations: 50,
            convergence_threshold: 0.001,
            min_improvement: 0.001,
            early_stopping_patience: 5,
            validation_split: 0.2,
            use_cross_validation: false,
            cv_folds: 5,
            random_seed: None,
            verbose: false,
            custom_params: HashMap::new(),
        }
    }
}

/// Result of teleprompter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Whether optimization was successful
    pub success: bool,
    /// Final optimization score
    pub final_score: f64,
    /// Improvement over baseline
    pub improvement: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Time taken for optimization (seconds)
    pub optimization_time: f64,
    /// Best examples found
    pub best_examples_count: usize,
    /// Optimization metrics
    pub metrics: OptimizationMetrics,
    /// Error message if optimization failed
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl OptimizationResult {
    /// Create a successful result
    pub fn success(
        final_score: f64,
        improvement: f64,
        iterations: usize,
        optimization_time: f64,
        best_examples_count: usize,
        metrics: OptimizationMetrics,
    ) -> Self {
        Self {
            success: true,
            final_score,
            improvement,
            iterations,
            optimization_time,
            best_examples_count,
            metrics,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed result
    pub fn failure(error_message: String, metrics: OptimizationMetrics) -> Self {
        Self {
            success: false,
            final_score: 0.0,
            improvement: 0.0,
            iterations: 0,
            optimization_time: 0.0,
            best_examples_count: 0,
            metrics,
            error_message: Some(error_message),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(
                f,
                "OptimizationResult[SUCCESS: score={:.3}, improvement={:.3}, iterations={}]",
                self.final_score, self.improvement, self.iterations
            )
        } else {
            write!(
                f,
                "OptimizationResult[FAILED: {}]",
                self.error_message.as_deref().unwrap_or("Unknown error")
            )
        }
    }
}

/// Teleprompter for optimizing DSPy modules
pub struct Teleprompter {
    /// Optimization strategy
    optimizer: Optimizer,
    /// Configuration
    config: TeleprompterConfig,
    /// Current best examples
    best_examples: Option<ExampleSet<serde_json::Value, serde_json::Value>>,
    /// Optimization history
    optimization_history: Vec<OptimizationResult>,
}

impl Teleprompter {
    /// Create a new teleprompter
    pub fn new(strategy: OptimizationStrategy) -> Self {
        let mut config = TeleprompterConfig::default();
        config.strategy = strategy.clone();
        Self {
            optimizer: Optimizer::new(strategy),
            config,
            best_examples: None,
            optimization_history: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TeleprompterConfig) -> Self {
        Self {
            optimizer: Optimizer::new(config.strategy.clone()),
            config,
            best_examples: None,
            optimization_history: Vec::new(),
        }
    }

    /// Optimize a module using the provided examples
    pub async fn optimize<I, O>(
        &mut self,
        module: &mut dyn Module<Input = I, Output = O>,
        examples: ExampleSet<I, O>,
    ) -> DspyResult<OptimizationResult>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
    {
        info!(
            "Starting teleprompter optimization with {} examples",
            examples.len()
        );

        let start_time = Instant::now();

        // Validate examples
        if examples.is_empty() {
            return Ok(OptimizationResult::failure(
                "No examples provided for optimization".to_string(),
                self.optimizer.metrics().clone(),
            ));
        }

        // Split examples into training and validation sets
        let (train_examples, val_examples) = if self.config.validation_split > 0.0 {
            examples.split(1.0 - self.config.validation_split)?
        } else {
            (examples, ExampleSet::new())
        };

        debug!(
            "Split examples: {} training, {} validation",
            train_examples.len(),
            val_examples.len()
        );

        // Initialize baseline score
        let baseline_score = if !val_examples.is_empty() {
            self.evaluate_module(module, &val_examples).await?
        } else {
            self.evaluate_module(module, &train_examples).await?
        };

        info!("Baseline score: {:.3}", baseline_score);
        self.optimizer.metrics_mut().record_score(baseline_score);

        let mut best_score = baseline_score;
        let mut best_iteration = 0;
        let mut iterations_without_improvement = 0;

        // Optimization loop
        for iteration in 0..self.config.max_iterations {
            debug!("Optimization iteration {}", iteration + 1);

            // Select examples using current strategy
            let selected_examples = self.optimizer.select_examples(&train_examples)?;

            if selected_examples.is_empty() {
                warn!("No examples selected in iteration {}", iteration + 1);
                continue;
            }

            // Compile module with selected examples
            let compilation_examples: Vec<(I, O)> = selected_examples
                .examples()
                .iter()
                .map(|e| (e.input.clone(), e.output.clone()))
                .collect();

            if module.supports_compilation() {
                module.compile(compilation_examples).await?;
            }

            // Evaluate on validation set
            let current_score = if !val_examples.is_empty() {
                self.evaluate_module(module, &val_examples).await?
            } else {
                self.evaluate_module(module, &train_examples).await?
            };

            self.optimizer.metrics_mut().record_score(current_score);

            if self.config.verbose {
                info!(
                    "Iteration {}: score={:.3} (best={:.3})",
                    iteration + 1,
                    current_score,
                    best_score
                );
            }

            // Check for improvement
            if current_score > best_score + self.config.min_improvement {
                best_score = current_score;
                best_iteration = iteration;
                iterations_without_improvement = 0;

                // Store best examples (convert to generic format)
                self.best_examples = Some(self.convert_examples_to_generic(&selected_examples)?);
            } else {
                iterations_without_improvement += 1;
            }

            // Early stopping check
            if iterations_without_improvement >= self.config.early_stopping_patience {
                info!(
                    "Early stopping at iteration {} (no improvement for {} iterations)",
                    iteration + 1,
                    iterations_without_improvement
                );
                break;
            }
        }

        let optimization_time = start_time.elapsed().as_secs_f64();
        let improvement = best_score - baseline_score;

        let result = OptimizationResult::success(
            best_score,
            improvement,
            best_iteration + 1,
            optimization_time,
            self.best_examples
                .as_ref()
                .map(|e| e.len())
                .unwrap_or(0),
            self.optimizer.metrics().clone(),
        );

        self.optimization_history.push(result.clone());

        info!(
            "Optimization complete: {:.3} -> {:.3} (improvement: {:.3})",
            baseline_score, best_score, improvement
        );

        Ok(result)
    }

    /// Evaluate a module on a set of examples
    async fn evaluate_module<I, O>(
        &self,
        module: &dyn Module<Input = I, Output = O>,
        examples: &ExampleSet<I, O>,
    ) -> DspyResult<f64>
    where
        I: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
        O: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        let mut total_score = 0.0;
        let mut count = 0;

        for example in examples.examples() {
            match module.forward(example.input.clone()).await {
                Ok(_output) => {
                    // Simplified scoring: use example quality score
                    // In a real implementation, this would compare output with expected
                    total_score += example.quality_score;
                    count += 1;
                }
                Err(e) => {
                    warn!("Module evaluation failed for example {}: {}", example.id, e);
                    // Penalize failures
                    total_score += 0.0;
                    count += 1;
                }
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_score / count as f64)
        }
    }

    /// Convert examples to generic format for storage
    fn convert_examples_to_generic<I, O>(
        &self,
        examples: &ExampleSet<I, O>,
    ) -> DspyResult<ExampleSet<serde_json::Value, serde_json::Value>>
    where
        I: Serialize,
        O: Serialize,
    {
        let mut generic_examples = Vec::new();

        for example in examples.examples() {
            let generic_input = serde_json::to_value(&example.input)
                .map_err(|e| DspyError::serialization("input", &e.to_string()))?;
            let generic_output = serde_json::to_value(&example.output)
                .map_err(|e| DspyError::serialization("output", &e.to_string()))?;

            let generic_example = Example::new(generic_input, generic_output)
                .with_quality_score(example.quality_score);

            generic_examples.push(generic_example);
        }

        Ok(ExampleSet::from_examples(generic_examples))
    }

    /// Get the best examples found during optimization
    pub fn best_examples(&self) -> Option<&ExampleSet<serde_json::Value, serde_json::Value>> {
        self.best_examples.as_ref()
    }

    /// Get optimization history
    pub fn optimization_history(&self) -> &[OptimizationResult] {
        &self.optimization_history
    }

    /// Get current configuration
    pub fn config(&self) -> &TeleprompterConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: TeleprompterConfig) {
        self.config = config;
    }

    /// Get current optimizer
    pub fn optimizer(&self) -> &Optimizer {
        &self.optimizer
    }

    /// Get mutable optimizer
    pub fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    /// Reset teleprompter state
    pub fn reset(&mut self) {
        self.optimizer.reset_metrics();
        self.best_examples = None;
        self.optimization_history.clear();
        info!("Teleprompter state reset");
    }

    /// Create a simple teleprompter with random sampling
    pub fn simple(sample_size: usize) -> Self {
        Self::new(OptimizationStrategy::random_sampling(sample_size))
    }

    /// Create a quality-focused teleprompter
    pub fn quality_focused(min_quality: f64) -> Self {
        Self::new(OptimizationStrategy::quality_based(min_quality))
    }

    /// Create a bootstrap teleprompter
    pub fn bootstrap(num_samples: usize, sample_size: usize) -> Self {
        Self::new(OptimizationStrategy::bootstrap(num_samples, sample_size))
    }
}

impl fmt::Display for Teleprompter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Teleprompter[{}, {} optimizations]",
            self.optimizer.strategy(),
            self.optimization_history.len()
        )
    }
}
