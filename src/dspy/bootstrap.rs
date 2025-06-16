//! Bootstrap teleprompter implementation for DSPy
//!
//! This module provides the BootstrapFewShot teleprompter, which combines labeled
//! training examples with algorithmically generated demonstrations to optimize
//! prompt construction for language model predictors.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::{Example, ExampleSet};
use crate::dspy::module::Module;
use crate::dspy::teleprompter::{OptimizationResult, Teleprompter};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Configuration for Bootstrap teleprompter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Maximum number of labeled examples to use
    pub max_labeled_demos: usize,
    /// Maximum number of bootstrapped examples to generate
    pub max_bootstrapped_demos: usize,
    /// Maximum number of bootstrap rounds to attempt
    pub max_rounds: usize,
    /// Maximum number of errors allowed during bootstrapping
    pub max_errors: usize,
    /// Minimum confidence threshold for accepting bootstrapped examples
    pub min_confidence: f64,
    /// Whether to use teacher forcing during bootstrap generation
    pub use_teacher_forcing: bool,
    /// Random seed for reproducible bootstrapping
    pub random_seed: Option<u64>,
    /// Validation strictness level
    pub validation_strictness: ValidationStrictness,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            max_labeled_demos: 16,
            max_bootstrapped_demos: 4,
            max_rounds: 1,
            max_errors: 5,
            min_confidence: 0.7,
            use_teacher_forcing: true,
            random_seed: None,
            validation_strictness: ValidationStrictness::Medium,
        }
    }
}

/// Validation strictness levels for bootstrap examples
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ValidationStrictness {
    /// Minimal validation - only check basic format
    Low,
    /// Standard validation - check format and basic quality
    Medium,
    /// Strict validation - comprehensive quality checks
    High,
    /// Custom validation - use provided metric function
    Custom,
}

/// Metric function type for validating bootstrap examples
pub type MetricFunction<I, O> = Arc<dyn Fn(&Example<I, O>, &O) -> DspyResult<bool> + Send + Sync>;

/// Bootstrap teleprompter for few-shot learning optimization
pub struct BootstrapFewShot<I, O> {
    /// Bootstrap configuration
    pub config: BootstrapConfig,
    /// Validation metric function
    metric: Option<MetricFunction<I, O>>,
    /// Base teleprompter for optimization
    base_teleprompter: Teleprompter,
    /// Generated bootstrap examples
    pub bootstrap_examples: Vec<Example<I, O>>,
    /// Bootstrap statistics
    pub stats: BootstrapStats,
}

/// Statistics for bootstrap generation process
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BootstrapStats {
    /// Total bootstrap attempts
    pub total_attempts: usize,
    /// Successful bootstrap generations
    pub successful_generations: usize,
    /// Failed bootstrap attempts
    pub failed_attempts: usize,
    /// Examples that passed validation
    pub validated_examples: usize,
    /// Examples that failed validation
    pub validation_failures: usize,
    /// Average confidence score of generated examples
    pub average_confidence: f64,
    /// Time spent on bootstrap generation
    pub generation_time_seconds: f64,
    /// Bootstrap rounds completed
    pub rounds_completed: usize,
}

impl<I, O> BootstrapFewShot<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    /// Create a new Bootstrap teleprompter
    pub fn new() -> Self {
        Self {
            config: BootstrapConfig::default(),
            metric: None,
            base_teleprompter: Teleprompter::simple(10),
            bootstrap_examples: Vec::new(),
            stats: BootstrapStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BootstrapConfig) -> Self {
        Self {
            config,
            metric: None,
            base_teleprompter: Teleprompter::simple(10),
            bootstrap_examples: Vec::new(),
            stats: BootstrapStats::default(),
        }
    }

    /// Set the validation metric function
    pub fn with_metric(mut self, metric: MetricFunction<I, O>) -> Self {
        self.metric = Some(metric);
        self.config.validation_strictness = ValidationStrictness::Custom;
        self
    }

    /// Set the base teleprompter for optimization
    pub fn with_teleprompter(mut self, teleprompter: Teleprompter) -> Self {
        self.base_teleprompter = teleprompter;
        self
    }

    /// Compile a module using bootstrap few-shot optimization
    pub async fn compile<M>(
        &mut self,
        module: &mut M,
        trainset: ExampleSet<I, O>,
    ) -> DspyResult<OptimizationResult>
    where
        I: serde::Serialize + for<'de> serde::Deserialize<'de>,
        O: serde::Serialize + for<'de> serde::Deserialize<'de>,
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        info!(
            "Starting Bootstrap few-shot compilation with {} training examples",
            trainset.len()
        );

        let start_time = Instant::now();

        // Validate configuration
        self.validate_config()?;

        // Split examples into labeled and unlabeled sets
        let (labeled_examples, unlabeled_examples) = self.split_examples(trainset)?;

        // Generate bootstrap examples
        self.generate_bootstrap_examples(module, &unlabeled_examples).await?;

        // Combine labeled and bootstrap examples
        let combined_examples = self.combine_examples(&labeled_examples)?;

        // Optimize module using combined examples
        let optimization_result = self.base_teleprompter.optimize(module, combined_examples).await?;

        // Update statistics
        self.stats.generation_time_seconds = start_time.elapsed().as_secs_f64();

        info!(
            "Bootstrap compilation completed: {} labeled + {} bootstrapped examples",
            labeled_examples.len(),
            self.bootstrap_examples.len()
        );

        Ok(optimization_result)
    }

    /// Generate bootstrap examples using the module
    async fn generate_bootstrap_examples<M>(
        &mut self,
        module: &M,
        unlabeled_examples: &ExampleSet<I, O>,
    ) -> DspyResult<()>
    where
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        debug!("Generating bootstrap examples");

        let mut attempts = 0;
        let mut successful = 0;
        let mut errors = 0;

        for round in 0..self.config.max_rounds {
            debug!("Bootstrap round {}", round + 1);

            for example in unlabeled_examples.examples() {
                if successful >= self.config.max_bootstrapped_demos {
                    break;
                }

                if errors >= self.config.max_errors {
                    warn!("Maximum errors reached, stopping bootstrap generation");
                    break;
                }

                attempts += 1;

                match self.generate_single_example(module, example).await {
                    Ok(bootstrap_example) => {
                        if self.validate_example(&bootstrap_example, example).await? {
                            self.bootstrap_examples.push(bootstrap_example);
                            successful += 1;
                            self.stats.validated_examples += 1;
                        } else {
                            self.stats.validation_failures += 1;
                        }
                    }
                    Err(e) => {
                        warn!("Bootstrap generation failed: {}", e);
                        errors += 1;
                        self.stats.failed_attempts += 1;
                    }
                }
            }

            self.stats.rounds_completed = round + 1;

            if successful >= self.config.max_bootstrapped_demos {
                break;
            }
        }

        self.stats.total_attempts = attempts;
        self.stats.successful_generations = successful;

        info!(
            "Bootstrap generation complete: {}/{} successful",
            successful, attempts
        );

        Ok(())
    }

    /// Generate a single bootstrap example
    async fn generate_single_example<M>(
        &self,
        module: &M,
        example: &Example<I, O>,
    ) -> DspyResult<Example<I, O>>
    where
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        // Generate output using the module
        let generated_output = module.forward(example.input.clone()).await?;

        // Create bootstrap example
        let quality_score = self.estimate_quality_score(example, &generated_output);
        let bootstrap_example = Example::new(example.input.clone(), generated_output)
            .with_metadata("bootstrap_source".to_string(), serde_json::json!("generated"))
            .with_metadata("original_id".to_string(), serde_json::json!(example.id))
            .with_quality_score(quality_score);

        Ok(bootstrap_example)
    }

    /// Validate a bootstrap example
    pub async fn validate_example(
        &self,
        bootstrap_example: &Example<I, O>,
        original_example: &Example<I, O>,
    ) -> DspyResult<bool> {
        // Use custom metric if provided
        if let Some(metric) = &self.metric {
            return metric(original_example, &bootstrap_example.output);
        }

        // Use built-in validation based on strictness
        match self.config.validation_strictness {
            ValidationStrictness::Low => Ok(true), // Accept all
            ValidationStrictness::Medium => {
                Ok(bootstrap_example.quality_score >= self.config.min_confidence)
            }
            ValidationStrictness::High => {
                Ok(bootstrap_example.quality_score >= self.config.min_confidence
                    && bootstrap_example.quality_score > original_example.quality_score)
            }
            ValidationStrictness::Custom => {
                // Should not reach here if metric is properly set
                Err(DspyError::configuration(
                    "validation_strictness",
                    "Custom validation requires a metric function",
                ))
            }
        }
    }

    /// Estimate quality score for generated output
    pub fn estimate_quality_score(&self, _original: &Example<I, O>, _generated: &O) -> f64 {
        // Simplified quality estimation
        // In a real implementation, this would use more sophisticated metrics
        self.config.min_confidence + 0.1
    }

    /// Split examples into labeled and unlabeled sets
    pub fn split_examples(
        &self,
        trainset: ExampleSet<I, O>,
    ) -> DspyResult<(ExampleSet<I, O>, ExampleSet<I, O>)> {
        let labeled_examples: Vec<Example<I, O>> = trainset
            .examples()
            .iter()
            .take(self.config.max_labeled_demos)
            .cloned()
            .collect();

        let unlabeled_examples: Vec<Example<I, O>> = trainset
            .examples()
            .iter()
            .skip(self.config.max_labeled_demos)
            .cloned()
            .collect();

        Ok((
            ExampleSet::from_examples(labeled_examples),
            ExampleSet::from_examples(unlabeled_examples),
        ))
    }

    /// Combine labeled and bootstrap examples
    pub fn combine_examples(&self, labeled_examples: &ExampleSet<I, O>) -> DspyResult<ExampleSet<I, O>> {
        let mut combined = labeled_examples.clone();

        for bootstrap_example in &self.bootstrap_examples {
            combined.add_example(bootstrap_example.clone());
        }

        Ok(combined)
    }

    /// Validate configuration
    pub fn validate_config(&self) -> DspyResult<()> {
        if self.config.max_labeled_demos == 0 && self.config.max_bootstrapped_demos == 0 {
            return Err(DspyError::configuration(
                "demos",
                "At least one of max_labeled_demos or max_bootstrapped_demos must be > 0",
            ));
        }

        if self.config.max_rounds == 0 {
            return Err(DspyError::configuration(
                "max_rounds",
                "Must be greater than 0",
            ));
        }

        if !(0.0..=1.0).contains(&self.config.min_confidence) {
            return Err(DspyError::configuration(
                "min_confidence",
                "Must be between 0.0 and 1.0",
            ));
        }

        Ok(())
    }

    /// Get bootstrap statistics
    pub fn stats(&self) -> &BootstrapStats {
        &self.stats
    }

    /// Get generated bootstrap examples
    pub fn bootstrap_examples(&self) -> &[Example<I, O>] {
        &self.bootstrap_examples
    }

    /// Get configuration
    pub fn config(&self) -> &BootstrapConfig {
        &self.config
    }

    /// Reset bootstrap state
    pub fn reset(&mut self) {
        self.bootstrap_examples.clear();
        self.stats = BootstrapStats::default();
    }
}

impl<I, O> Default for BootstrapFewShot<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I, O> fmt::Display for BootstrapFewShot<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BootstrapFewShot[labeled: {}, bootstrap: {}, rounds: {}]",
            self.config.max_labeled_demos,
            self.config.max_bootstrapped_demos,
            self.config.max_rounds
        )
    }
}

impl fmt::Display for BootstrapStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BootstrapStats[attempts: {}, success: {}, validated: {}, avg_confidence: {:.2}]",
            self.total_attempts, self.successful_generations, self.validated_examples, self.average_confidence
        )
    }
}
