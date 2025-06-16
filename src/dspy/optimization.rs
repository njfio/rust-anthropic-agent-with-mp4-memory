//! Optimization strategies for DSPy modules
//!
//! This module provides various optimization strategies used by teleprompters
//! to improve module performance through prompt optimization and example selection.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::{Example, ExampleSet};
use crate::dspy::module::Module;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::{debug, info, warn};

/// Optimization strategy for improving module performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Random sampling of examples
    RandomSampling {
        /// Number of examples to sample
        sample_size: usize,
        /// Random seed for reproducibility
        seed: Option<u64>,
    },
    /// Bootstrap sampling with replacement
    Bootstrap {
        /// Number of bootstrap samples
        num_samples: usize,
        /// Size of each bootstrap sample
        sample_size: usize,
    },
    /// Quality-based selection
    QualityBased {
        /// Minimum quality threshold
        min_quality: f64,
        /// Maximum number of examples
        max_examples: Option<usize>,
    },
    /// Diversity-based selection
    DiversityBased {
        /// Number of clusters for diversity
        num_clusters: usize,
        /// Examples per cluster
        examples_per_cluster: usize,
    },
    /// Gradient-based optimization (simplified)
    GradientBased {
        /// Learning rate
        learning_rate: f64,
        /// Number of iterations
        iterations: usize,
        /// Batch size
        batch_size: usize,
    },
    /// Evolutionary optimization
    Evolutionary {
        /// Population size
        population_size: usize,
        /// Number of generations
        generations: usize,
        /// Mutation rate
        mutation_rate: f64,
        /// Selection pressure
        selection_pressure: f64,
    },
    /// MIPROv2 multi-stage optimization
    MIPROv2 {
        /// Number of bootstrap samples
        num_bootstrap: usize,
        /// Number of proposal candidates per iteration
        num_candidates: usize,
        /// Mini-batch size for evaluation
        mini_batch_size: usize,
        /// Maximum optimization iterations
        max_iterations: usize,
        /// Surrogate model confidence threshold
        confidence_threshold: f64,
    },
    /// Bootstrap fine-tuning
    BootstrapFinetune {
        /// Learning rate for fine-tuning
        learning_rate: f64,
        /// Number of fine-tuning epochs
        num_epochs: usize,
        /// Batch size for training
        batch_size: usize,
        /// Regularization strength
        regularization: f64,
        /// Early stopping patience
        early_stopping_patience: usize,
    },
    /// Multi-objective optimization
    MultiObjective {
        /// Objective weights for scalarization
        objective_weights: Vec<f64>,
        /// Whether to use Pareto optimization
        use_pareto: bool,
        /// Maximum Pareto front size
        max_pareto_size: usize,
        /// Convergence tolerance
        tolerance: f64,
    },

    /// Hyperparameter tuning
    HyperparameterTuning {
        /// Parameter ranges for tuning
        param_ranges: HashMap<String, (f64, f64)>,
        /// Search method (grid, random, bayesian)
        search_method: String,
        /// Maximum number of evaluations
        max_evaluations: usize,
    },
}

impl OptimizationStrategy {
    /// Create a default random sampling strategy
    pub fn random_sampling(sample_size: usize) -> Self {
        Self::RandomSampling {
            sample_size,
            seed: None,
        }
    }

    /// Create a quality-based strategy
    pub fn quality_based(min_quality: f64) -> Self {
        Self::QualityBased {
            min_quality,
            max_examples: None,
        }
    }

    /// Create a bootstrap strategy
    pub fn bootstrap(num_samples: usize, sample_size: usize) -> Self {
        Self::Bootstrap {
            num_samples,
            sample_size,
        }
    }

    /// Create a diversity-based strategy
    pub fn diversity_based(num_clusters: usize, examples_per_cluster: usize) -> Self {
        Self::DiversityBased {
            num_clusters,
            examples_per_cluster,
        }
    }

    /// Create a MIPROv2 strategy
    pub fn mipro_v2(
        num_bootstrap: usize,
        num_candidates: usize,
        mini_batch_size: usize,
        max_iterations: usize,
    ) -> Self {
        Self::MIPROv2 {
            num_bootstrap,
            num_candidates,
            mini_batch_size,
            max_iterations,
            confidence_threshold: 0.8,
        }
    }

    /// Create a bootstrap fine-tune strategy
    pub fn bootstrap_finetune(learning_rate: f64, num_epochs: usize, batch_size: usize) -> Self {
        Self::BootstrapFinetune {
            learning_rate,
            num_epochs,
            batch_size,
            regularization: 0.01,
            early_stopping_patience: 5,
        }
    }

    /// Create a multi-objective strategy
    pub fn multi_objective(objective_weights: Vec<f64>) -> Self {
        Self::MultiObjective {
            objective_weights,
            use_pareto: true,
            max_pareto_size: 100,
            tolerance: 1e-6,
        }
    }

    /// Get strategy name
    pub fn name(&self) -> &'static str {
        match self {
            Self::RandomSampling { .. } => "RandomSampling",
            Self::Bootstrap { .. } => "Bootstrap",
            Self::QualityBased { .. } => "QualityBased",
            Self::DiversityBased { .. } => "DiversityBased",
            Self::GradientBased { .. } => "GradientBased",
            Self::Evolutionary { .. } => "Evolutionary",
            Self::MIPROv2 { .. } => "MIPROv2",
            Self::BootstrapFinetune { .. } => "BootstrapFinetune",
            Self::MultiObjective { .. } => "MultiObjective",
            Self::HyperparameterTuning { .. } => "HyperparameterTuning",
        }
    }
}

impl fmt::Display for OptimizationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSampling { sample_size, .. } => {
                write!(f, "RandomSampling(size={})", sample_size)
            }
            Self::Bootstrap {
                num_samples,
                sample_size,
            } => write!(
                f,
                "Bootstrap(samples={}, size={})",
                num_samples, sample_size
            ),
            Self::QualityBased {
                min_quality,
                max_examples,
            } => write!(
                f,
                "QualityBased(min_quality={:.2}, max={})",
                min_quality,
                max_examples
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "∞".to_string())
            ),
            Self::DiversityBased {
                num_clusters,
                examples_per_cluster,
            } => write!(
                f,
                "DiversityBased(clusters={}, per_cluster={})",
                num_clusters, examples_per_cluster
            ),
            Self::GradientBased {
                learning_rate,
                iterations,
                batch_size,
            } => write!(
                f,
                "GradientBased(lr={:.3}, iter={}, batch={})",
                learning_rate, iterations, batch_size
            ),
            Self::Evolutionary {
                population_size,
                generations,
                mutation_rate,
                selection_pressure,
            } => write!(
                f,
                "Evolutionary(pop={}, gen={}, mut={:.2}, sel={:.2})",
                population_size, generations, mutation_rate, selection_pressure
            ),
            Self::MIPROv2 {
                num_bootstrap,
                num_candidates,
                mini_batch_size,
                max_iterations,
                confidence_threshold,
            } => write!(
                f,
                "MIPROv2(bootstrap={}, candidates={}, batch={}, iter={}, conf={:.2})",
                num_bootstrap,
                num_candidates,
                mini_batch_size,
                max_iterations,
                confidence_threshold
            ),
            Self::BootstrapFinetune {
                learning_rate,
                num_epochs,
                batch_size,
                regularization,
                early_stopping_patience,
            } => write!(
                f,
                "BootstrapFinetune(lr={:.4}, epochs={}, batch={}, reg={:.3}, patience={})",
                learning_rate, num_epochs, batch_size, regularization, early_stopping_patience
            ),
            Self::MultiObjective {
                objective_weights,
                use_pareto,
                max_pareto_size,
                tolerance,
            } => write!(
                f,
                "MultiObjective(weights={:?}, pareto={}, max_size={}, tol={:.2e})",
                objective_weights, use_pareto, max_pareto_size, tolerance
            ),
            Self::HyperparameterTuning {
                param_ranges,
                search_method,
                max_evaluations,
            } => write!(
                f,
                "HyperparameterTuning(params={}, method={}, max_evals={})",
                param_ranges.len(),
                search_method,
                max_evaluations
            ),
        }
    }
}

/// Optimization metrics for tracking performance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationMetrics {
    /// Number of optimization iterations performed
    pub iterations: usize,
    /// Best score achieved
    pub best_score: f64,
    /// Current score
    pub current_score: f64,
    /// Score history
    pub score_history: Vec<f64>,
    /// Time spent optimizing (in seconds)
    pub optimization_time: f64,
    /// Time spent optimizing (in milliseconds) - for compatibility
    pub optimization_time_ms: f64,
    /// Number of examples used
    pub examples_used: usize,
    /// Strategy used
    pub strategy_name: String,
    /// Additional metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Pareto front for multi-objective optimization
    pub pareto_front: Vec<Vec<f64>>,
    /// Hyperparameters for tracking
    pub hyperparameters: HashMap<String, f64>,
}

impl OptimizationMetrics {
    /// Create new metrics
    pub fn new(strategy_name: String) -> Self {
        Self {
            strategy_name,
            ..Default::default()
        }
    }

    /// Record a score
    pub fn record_score(&mut self, score: f64) {
        self.current_score = score;
        self.score_history.push(score);
        if score > self.best_score {
            self.best_score = score;
        }
        self.iterations += 1;
    }

    /// Add custom metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Get a custom metric
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.custom_metrics.get(name).copied()
    }

    /// Get improvement over baseline
    pub fn improvement(&self) -> f64 {
        if self.score_history.is_empty() {
            0.0
        } else {
            self.best_score - self.score_history[0]
        }
    }

    /// Check if optimization is converging
    pub fn is_converging(&self, window_size: usize, threshold: f64) -> bool {
        if self.score_history.len() < window_size {
            return false;
        }

        let recent_scores = &self.score_history[self.score_history.len() - window_size..];
        let variance = {
            let mean = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
            recent_scores
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_scores.len() as f64
        };

        variance < threshold
    }

    /// Add a point to the Pareto front
    pub fn add_pareto_point(&mut self, point: Vec<f64>) {
        self.pareto_front.push(point);
    }

    /// Get the size of the Pareto front
    pub fn pareto_front_size(&self) -> usize {
        self.pareto_front.len()
    }

    /// Check if one point dominates another
    pub fn dominates(&self, point1: &[f64], point2: &[f64]) -> bool {
        if point1.len() != point2.len() {
            return false;
        }

        let mut at_least_one_better = false;
        for (a, b) in point1.iter().zip(point2.iter()) {
            if a < b {
                return false; // point1 is worse in this objective
            }
            if a > b {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Set a hyperparameter
    pub fn set_hyperparameter(&mut self, name: &str, value: f64) {
        self.hyperparameters.insert(name.to_string(), value);
    }

    /// Get a hyperparameter
    pub fn get_hyperparameter(&self, name: &str) -> Option<f64> {
        self.hyperparameters.get(name).copied()
    }
}

impl fmt::Display for OptimizationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OptimizationMetrics[{} iterations, best: {:.3}, current: {:.3}, improvement: {:.3}]",
            self.iterations,
            self.best_score,
            self.current_score,
            self.improvement()
        )
    }
}

/// Optimizer for applying optimization strategies
pub struct Optimizer {
    /// Current strategy
    strategy: OptimizationStrategy,
    /// Optimization metrics
    metrics: OptimizationMetrics,
    /// Configuration parameters
    config: OptimizerConfig,
}

/// Configuration for the optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Minimum improvement threshold
    pub min_improvement: f64,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            early_stopping_patience: 10,
            min_improvement: 1e-4,
            verbose: false,
        }
    }
}

impl Optimizer {
    /// Create a new optimizer
    pub fn new(strategy: OptimizationStrategy) -> Self {
        let metrics = OptimizationMetrics::new(strategy.name().to_string());
        Self {
            strategy,
            metrics,
            config: OptimizerConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(strategy: OptimizationStrategy, config: OptimizerConfig) -> Self {
        let metrics = OptimizationMetrics::new(strategy.name().to_string());
        Self {
            strategy,
            metrics,
            config,
        }
    }

    /// Select examples using the current strategy
    pub fn select_examples<I, O>(
        &mut self,
        examples: &ExampleSet<I, O>,
    ) -> DspyResult<ExampleSet<I, O>>
    where
        I: Clone,
        O: Clone,
    {
        debug!("Selecting examples using strategy: {}", self.strategy);

        let selected = match &self.strategy {
            OptimizationStrategy::RandomSampling { sample_size, seed } => {
                self.random_sampling(examples, *sample_size, *seed)?
            }
            OptimizationStrategy::Bootstrap {
                num_samples,
                sample_size,
            } => self.bootstrap_sampling(examples, *num_samples, *sample_size)?,
            OptimizationStrategy::QualityBased {
                min_quality,
                max_examples,
            } => self.quality_based_selection(examples, *min_quality, *max_examples)?,
            OptimizationStrategy::DiversityBased {
                num_clusters,
                examples_per_cluster,
            } => self.diversity_based_selection(examples, *num_clusters, *examples_per_cluster)?,
            OptimizationStrategy::GradientBased { .. } => {
                warn!(
                    "Gradient-based selection not yet implemented, falling back to quality-based"
                );
                self.quality_based_selection(examples, 0.5, None)?
            }
            OptimizationStrategy::Evolutionary { .. } => {
                warn!("Evolutionary selection not yet implemented, falling back to quality-based");
                self.quality_based_selection(examples, 0.5, None)?
            }
            OptimizationStrategy::MIPROv2 { num_bootstrap, .. } => {
                info!("Using MIPROv2 bootstrap sampling");
                self.bootstrap_sampling(examples, 1, *num_bootstrap)?
            }
            OptimizationStrategy::BootstrapFinetune { batch_size, .. } => {
                info!("Using BootstrapFinetune sampling");
                self.random_sampling(examples, *batch_size, None)?
            }
            OptimizationStrategy::MultiObjective { .. } => {
                info!("Using multi-objective sampling (quality-based)");
                self.quality_based_selection(examples, 0.7, None)?
            }
            OptimizationStrategy::HyperparameterTuning { .. } => {
                info!("Using hyperparameter tuning sampling (random)");
                self.random_sampling(examples, 20, None)?
            }
        };

        self.metrics.examples_used = selected.len();
        info!(
            "Selected {} examples using {}",
            selected.len(),
            self.strategy.name()
        );

        Ok(selected)
    }

    /// Random sampling implementation
    fn random_sampling<I, O>(
        &self,
        examples: &ExampleSet<I, O>,
        sample_size: usize,
        seed: Option<u64>,
    ) -> DspyResult<ExampleSet<I, O>>
    where
        I: Clone,
        O: Clone,
    {
        use rand::seq::SliceRandom;
        use rand::{rngs::StdRng, SeedableRng};

        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let available = examples.examples();
        let sample_size = sample_size.min(available.len());

        let sampled: Vec<Example<I, O>> = available
            .choose_multiple(&mut rng, sample_size)
            .cloned()
            .collect();

        Ok(ExampleSet::from_examples(sampled))
    }

    /// Bootstrap sampling implementation
    fn bootstrap_sampling<I, O>(
        &self,
        examples: &ExampleSet<I, O>,
        num_samples: usize,
        sample_size: usize,
    ) -> DspyResult<ExampleSet<I, O>>
    where
        I: Clone,
        O: Clone,
    {
        use rand::seq::SliceRandom;

        let mut rng = rand::thread_rng();
        let available = examples.examples();
        let mut all_sampled = Vec::new();

        for _ in 0..num_samples {
            let sample_size = sample_size.min(available.len());
            let sampled: Vec<Example<I, O>> = available
                .choose_multiple(&mut rng, sample_size)
                .cloned()
                .collect();
            all_sampled.extend(sampled);
        }

        Ok(ExampleSet::from_examples(all_sampled))
    }

    /// Quality-based selection implementation
    fn quality_based_selection<I, O>(
        &self,
        examples: &ExampleSet<I, O>,
        min_quality: f64,
        max_examples: Option<usize>,
    ) -> DspyResult<ExampleSet<I, O>>
    where
        I: Clone,
        O: Clone,
    {
        let mut filtered: Vec<Example<I, O>> = examples
            .examples()
            .iter()
            .filter(|e| e.quality_score >= min_quality)
            .cloned()
            .collect();

        // Sort by quality score (descending)
        filtered.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());

        // Limit to max_examples if specified
        if let Some(max) = max_examples {
            filtered.truncate(max);
        }

        Ok(ExampleSet::from_examples(filtered))
    }

    /// Diversity-based selection implementation (simplified)
    fn diversity_based_selection<I, O>(
        &self,
        examples: &ExampleSet<I, O>,
        num_clusters: usize,
        examples_per_cluster: usize,
    ) -> DspyResult<ExampleSet<I, O>>
    where
        I: Clone,
        O: Clone,
    {
        // Simplified implementation: randomly distribute examples into clusters
        // In a real implementation, this would use proper clustering algorithms
        use rand::seq::SliceRandom;

        let mut rng = rand::thread_rng();
        let available = examples.examples();
        let mut selected = Vec::new();

        let cluster_size = available.len() / num_clusters.max(1);

        for cluster_idx in 0..num_clusters {
            let start = cluster_idx * cluster_size;
            let end = ((cluster_idx + 1) * cluster_size).min(available.len());

            if start < available.len() {
                let cluster_examples = &available[start..end];
                let sample_size = examples_per_cluster.min(cluster_examples.len());

                let sampled: Vec<Example<I, O>> = cluster_examples
                    .choose_multiple(&mut rng, sample_size)
                    .cloned()
                    .collect();

                selected.extend(sampled);
            }
        }

        Ok(ExampleSet::from_examples(selected))
    }

    /// Get current metrics
    pub fn metrics(&self) -> &OptimizationMetrics {
        &self.metrics
    }

    /// Get mutable metrics
    pub fn metrics_mut(&mut self) -> &mut OptimizationMetrics {
        &mut self.metrics
    }

    /// Get current strategy
    pub fn strategy(&self) -> &OptimizationStrategy {
        &self.strategy
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: OptimizationStrategy) {
        self.strategy = strategy;
        self.metrics.strategy_name = self.strategy.name().to_string();
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = OptimizationMetrics::new(self.strategy.name().to_string());
    }
}

impl fmt::Display for Optimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Optimizer[{}]", self.strategy)
    }
}
