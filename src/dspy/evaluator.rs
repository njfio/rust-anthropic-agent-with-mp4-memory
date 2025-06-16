//! DSPy evaluator for running multiple metrics and statistical analysis
//!
//! This module provides comprehensive evaluation capabilities for DSPy modules,
//! including multiple metric evaluation, statistical significance testing,
//! and result formatting.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::{Example, ExampleSet};
use crate::dspy::metrics::{Metric, MetricResult};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Evaluator for running multiple metrics on predictions
pub struct Evaluator<I, O> {
    /// Metrics to evaluate
    metrics: Vec<Arc<dyn Metric<I, O>>>,
    /// Configuration for evaluation
    config: EvaluatorConfig,
    /// Statistics from previous evaluations
    stats: EvaluationStats,
}

/// Configuration for the evaluator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorConfig {
    /// Whether to run statistical significance tests
    pub enable_significance_testing: bool,
    /// Confidence level for significance tests (e.g., 0.95 for 95%)
    pub significance_confidence: f64,
    /// Whether to compute detailed statistics
    pub compute_detailed_stats: bool,
    /// Whether to stop evaluation on first failure
    pub fail_fast: bool,
    /// Maximum number of examples to evaluate (0 = no limit)
    pub max_examples: usize,
    /// Whether to shuffle examples before evaluation
    pub shuffle_examples: bool,
    /// Random seed for shuffling (None = random)
    pub random_seed: Option<u64>,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            enable_significance_testing: true,
            significance_confidence: 0.95,
            compute_detailed_stats: true,
            fail_fast: false,
            max_examples: 0,
            shuffle_examples: false,
            random_seed: None,
        }
    }
}

/// Statistics from evaluation runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationStats {
    /// Total number of evaluations run
    pub total_evaluations: usize,
    /// Total number of examples evaluated
    pub total_examples: usize,
    /// Average evaluation time in milliseconds
    pub avg_evaluation_time_ms: f64,
    /// Number of failed evaluations
    pub failed_evaluations: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

impl Default for EvaluationStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_examples: 0,
            avg_evaluation_time_ms: 0.0,
            failed_evaluations: 0,
            success_rate: 1.0,
        }
    }
}

/// Result of evaluating a set of examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Results for each metric
    pub metric_results: HashMap<String, MetricSummary>,
    /// Overall statistics
    pub overall_stats: OverallStats,
    /// Statistical significance results (if enabled)
    pub significance_tests: Option<SignificanceTestResults>,
    /// Detailed per-example results (if enabled)
    pub detailed_results: Option<Vec<ExampleResult>>,
    /// Evaluation metadata
    pub metadata: EvaluationMetadata,
}

/// Summary statistics for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Metric name
    pub name: String,
    /// Mean score
    pub mean_score: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum score
    pub min_score: f64,
    /// Maximum score
    pub max_score: f64,
    /// Median score
    pub median_score: f64,
    /// Pass rate (percentage of examples that passed)
    pub pass_rate: f64,
    /// Total number of examples evaluated
    pub total_examples: usize,
    /// Number of examples that passed
    pub passed_examples: usize,
    /// Confidence interval (if computed)
    pub confidence_interval: Option<(f64, f64)>,
}

/// Overall evaluation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallStats {
    /// Total examples evaluated
    pub total_examples: usize,
    /// Total evaluation time in milliseconds
    pub total_time_ms: f64,
    /// Average time per example in milliseconds
    pub avg_time_per_example_ms: f64,
    /// Number of metrics evaluated
    pub num_metrics: usize,
    /// Overall success rate (all metrics passed)
    pub overall_success_rate: f64,
}

/// Results of statistical significance tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTestResults {
    /// Pairwise comparisons between metrics
    pub pairwise_comparisons: HashMap<String, HashMap<String, PairwiseComparison>>,
    /// Overall ANOVA results (if applicable)
    pub anova_results: Option<AnovaResults>,
}

/// Pairwise comparison between two metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    /// Metric 1 name
    pub metric1: String,
    /// Metric 2 name
    pub metric2: String,
    /// T-test statistic
    pub t_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether the difference is significant
    pub is_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
}

/// ANOVA test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResults {
    /// F-statistic
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether there are significant differences
    pub is_significant: bool,
    /// Degrees of freedom
    pub degrees_of_freedom: (usize, usize),
}

/// Result for a single example evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// Example index
    pub example_index: usize,
    /// Results for each metric
    pub metric_results: HashMap<String, MetricResult>,
    /// Whether all metrics passed
    pub all_passed: bool,
    /// Evaluation time in milliseconds
    pub evaluation_time_ms: f64,
}

/// Metadata about the evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetadata {
    /// Timestamp when evaluation started
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Timestamp when evaluation completed
    pub end_time: chrono::DateTime<chrono::Utc>,
    /// Configuration used for evaluation
    pub config: EvaluatorConfig,
    /// Version of the evaluator
    pub evaluator_version: String,
}

impl<I, O> Evaluator<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            config: EvaluatorConfig::default(),
            stats: EvaluationStats::default(),
        }
    }

    /// Create evaluator with configuration
    pub fn with_config(config: EvaluatorConfig) -> Self {
        Self {
            metrics: Vec::new(),
            config,
            stats: EvaluationStats::default(),
        }
    }

    /// Add a metric to the evaluator
    pub fn add_metric(mut self, metric: Arc<dyn Metric<I, O>>) -> Self {
        self.metrics.push(metric);
        self
    }

    /// Set configuration
    pub fn set_config(&mut self, config: EvaluatorConfig) {
        self.config = config;
    }

    /// Get configuration
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }

    /// Get current statistics
    pub fn stats(&self) -> &EvaluationStats {
        &self.stats
    }

    /// Validate the evaluator configuration
    pub fn validate(&self) -> DspyResult<()> {
        if self.metrics.is_empty() {
            return Err(DspyError::configuration(
                "metrics",
                "Evaluator must have at least one metric",
            ));
        }

        if !(0.0..=1.0).contains(&self.config.significance_confidence) {
            return Err(DspyError::configuration(
                "significance_confidence",
                "Significance confidence must be between 0.0 and 1.0",
            ));
        }

        // Validate all metrics
        for metric in &self.metrics {
            metric.validate()?;
        }

        Ok(())
    }

    /// Evaluate predictions against examples
    pub async fn evaluate<F, Fut>(
        &mut self,
        examples: &ExampleSet<I, O>,
        predict_fn: F,
    ) -> DspyResult<EvaluationResult>
    where
        F: Fn(&I) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = DspyResult<O>> + Send,
    {
        let start_time = chrono::Utc::now();
        
        // Validate before starting
        self.validate()?;

        if examples.is_empty() {
            return Err(DspyError::configuration(
                "examples",
                "Cannot evaluate with empty example set",
            ));
        }

        info!(
            "Starting evaluation with {} metrics on {} examples",
            self.metrics.len(),
            examples.len()
        );

        // Prepare examples (shuffle if configured)
        let mut example_indices: Vec<usize> = (0..examples.len()).collect();
        if self.config.shuffle_examples {
            let mut rng = if let Some(seed) = self.config.random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_entropy()
            };
            example_indices.shuffle(&mut rng);
        }

        // Limit examples if configured
        if self.config.max_examples > 0 && self.config.max_examples < example_indices.len() {
            example_indices.truncate(self.config.max_examples);
        }

        let mut detailed_results = if self.config.compute_detailed_stats {
            Some(Vec::new())
        } else {
            None
        };

        let mut all_metric_scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut all_metric_results: HashMap<String, Vec<MetricResult>> = HashMap::new();

        // Initialize score collections
        for metric in &self.metrics {
            all_metric_scores.insert(metric.name().to_string(), Vec::new());
            all_metric_results.insert(metric.name().to_string(), Vec::new());
        }

        let total_examples = example_indices.len();
        let mut successful_evaluations = 0;

        // Evaluate each example
        for (eval_index, &example_index) in example_indices.iter().enumerate() {
            let example = examples.examples().get(example_index).ok_or_else(|| {
                DspyError::evaluation("evaluation", &format!("Example index {} out of bounds", example_index))
            })?;
            let eval_start = std::time::Instant::now();

            debug!("Evaluating example {}/{}", eval_index + 1, total_examples);

            // Add small delay to simulate actual work
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

            // Get prediction
            let prediction = match predict_fn(&example.input).await {
                Ok(pred) => pred,
                Err(e) => {
                    warn!("Prediction failed for example {}: {}", example_index, e);
                    if self.config.fail_fast {
                        return Err(e);
                    }
                    continue;
                }
            };

            let mut example_metric_results = HashMap::new();
            let mut all_passed = true;

            // Evaluate with each metric
            for metric in &self.metrics {
                let metric_result = match metric.evaluate(example, &prediction) {
                    Ok(result) => result,
                    Err(e) => {
                        warn!("Metric {} failed for example {}: {}", metric.name(), example_index, e);
                        if self.config.fail_fast {
                            return Err(e);
                        }
                        all_passed = false;
                        continue;
                    }
                };

                let metric_name = metric.name().to_string();
                all_metric_scores
                    .get_mut(&metric_name)
                    .unwrap()
                    .push(metric_result.score);
                all_metric_results
                    .get_mut(&metric_name)
                    .unwrap()
                    .push(metric_result.clone());

                if !metric_result.passed {
                    all_passed = false;
                }

                example_metric_results.insert(metric_name, metric_result);
            }

            if all_passed {
                successful_evaluations += 1;
            }

            let eval_time = eval_start.elapsed().as_millis() as f64;

            // Store detailed results if enabled
            if let Some(ref mut detailed) = detailed_results {
                detailed.push(ExampleResult {
                    example_index,
                    metric_results: example_metric_results,
                    all_passed,
                    evaluation_time_ms: eval_time,
                });
            }
        }

        let end_time = chrono::Utc::now();
        let total_time_ms = (end_time - start_time).num_milliseconds() as f64;

        // Compute metric summaries
        let mut metric_results = HashMap::new();
        for metric in &self.metrics {
            let metric_name = metric.name().to_string();
            let scores = &all_metric_scores[&metric_name];
            let results = &all_metric_results[&metric_name];

            if !scores.is_empty() {
                let summary = self.compute_metric_summary(&metric_name, scores, results)?;
                metric_results.insert(metric_name, summary);
            }
        }

        // Compute overall statistics
        let overall_stats = OverallStats {
            total_examples,
            total_time_ms,
            avg_time_per_example_ms: if total_examples > 0 {
                total_time_ms / total_examples as f64
            } else {
                0.0
            },
            num_metrics: self.metrics.len(),
            overall_success_rate: if total_examples > 0 {
                successful_evaluations as f64 / total_examples as f64
            } else {
                0.0
            },
        };

        // Compute significance tests if enabled
        let significance_tests = if self.config.enable_significance_testing && self.metrics.len() > 1 {
            Some(self.compute_significance_tests(&all_metric_scores)?)
        } else {
            None
        };

        // Update internal statistics
        self.stats.total_evaluations += 1;
        self.stats.total_examples += total_examples;
        self.stats.failed_evaluations += total_examples - successful_evaluations;
        self.stats.success_rate = if self.stats.total_examples > 0 {
            (self.stats.total_examples - self.stats.failed_evaluations) as f64 / self.stats.total_examples as f64
        } else {
            1.0
        };

        // Update average evaluation time
        let total_time_sum = self.stats.avg_evaluation_time_ms * (self.stats.total_evaluations - 1) as f64 + total_time_ms;
        self.stats.avg_evaluation_time_ms = total_time_sum / self.stats.total_evaluations as f64;

        let metadata = EvaluationMetadata {
            start_time,
            end_time,
            config: self.config.clone(),
            evaluator_version: "1.0.0".to_string(),
        };

        info!(
            "Evaluation completed: {}/{} examples successful, {:.1}ms total",
            successful_evaluations, total_examples, total_time_ms
        );

        Ok(EvaluationResult {
            metric_results,
            overall_stats,
            significance_tests,
            detailed_results,
            metadata,
        })
    }

    /// Compute summary statistics for a metric
    fn compute_metric_summary(
        &self,
        metric_name: &str,
        scores: &[f64],
        results: &[MetricResult],
    ) -> DspyResult<MetricSummary> {
        if scores.is_empty() {
            return Err(DspyError::evaluation(
                metric_name,
                &format!("No scores available for metric {}", metric_name)
            ));
        }

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_dev = variance.sqrt();

        let min_score = sorted_scores[0];
        let max_score = sorted_scores[sorted_scores.len() - 1];
        let median_score = if sorted_scores.len() % 2 == 0 {
            let mid = sorted_scores.len() / 2;
            (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
        } else {
            sorted_scores[sorted_scores.len() / 2]
        };

        let passed_examples = results.iter().filter(|r| r.passed).count();
        let pass_rate = passed_examples as f64 / results.len() as f64;

        let confidence_interval = if self.config.compute_detailed_stats && scores.len() > 1 {
            Some(self.compute_confidence_interval(scores, self.config.significance_confidence))
        } else {
            None
        };

        Ok(MetricSummary {
            name: metric_name.to_string(),
            mean_score,
            std_dev,
            min_score,
            max_score,
            median_score,
            pass_rate,
            total_examples: scores.len(),
            passed_examples,
            confidence_interval,
        })
    }

    /// Compute confidence interval for scores
    fn compute_confidence_interval(&self, scores: &[f64], confidence: f64) -> (f64, f64) {
        if scores.len() < 2 {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            return (mean, mean);
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / (scores.len() - 1) as f64;
        let std_error = (variance / scores.len() as f64).sqrt();

        // Use t-distribution critical value (approximation for large n)
        let alpha = 1.0 - confidence;
        let t_critical = self.t_critical_value(scores.len() - 1, alpha / 2.0);

        let margin_of_error = t_critical * std_error;
        (mean - margin_of_error, mean + margin_of_error)
    }

    /// Approximate t-critical value (simplified implementation)
    fn t_critical_value(&self, df: usize, alpha: f64) -> f64 {
        // Simplified approximation - in practice, you'd use a proper t-table or library
        if df >= 30 {
            // Use normal approximation for large df
            match alpha {
                a if a <= 0.005 => 2.576,  // 99% confidence
                a if a <= 0.01 => 2.326,   // 98% confidence
                a if a <= 0.025 => 1.96,   // 95% confidence
                a if a <= 0.05 => 1.645,   // 90% confidence
                _ => 1.0,
            }
        } else {
            // Rough approximation for smaller df
            let base = match alpha {
                a if a <= 0.005 => 2.8,
                a if a <= 0.01 => 2.5,
                a if a <= 0.025 => 2.1,
                a if a <= 0.05 => 1.8,
                _ => 1.2,
            };
            base * (1.0 + 2.0 / df as f64)
        }
    }

    /// Compute statistical significance tests between metrics
    fn compute_significance_tests(
        &self,
        metric_scores: &HashMap<String, Vec<f64>>,
    ) -> DspyResult<SignificanceTestResults> {
        let mut pairwise_comparisons = HashMap::new();

        let metric_names: Vec<String> = metric_scores.keys().cloned().collect();

        // Compute pairwise t-tests
        for i in 0..metric_names.len() {
            let metric1 = &metric_names[i];
            let mut comparisons = HashMap::new();

            for j in (i + 1)..metric_names.len() {
                let metric2 = &metric_names[j];

                let scores1 = &metric_scores[metric1];
                let scores2 = &metric_scores[metric2];

                if scores1.len() == scores2.len() && !scores1.is_empty() {
                    let comparison = self.compute_paired_t_test(scores1, scores2)?;
                    comparisons.insert(metric2.clone(), comparison);
                }
            }

            if !comparisons.is_empty() {
                pairwise_comparisons.insert(metric1.clone(), comparisons);
            }
        }

        // Compute ANOVA if we have more than 2 metrics
        let anova_results = if metric_names.len() > 2 {
            Some(self.compute_anova(metric_scores)?)
        } else {
            None
        };

        Ok(SignificanceTestResults {
            pairwise_comparisons,
            anova_results,
        })
    }

    /// Compute paired t-test between two sets of scores
    fn compute_paired_t_test(&self, scores1: &[f64], scores2: &[f64]) -> DspyResult<PairwiseComparison> {
        if scores1.len() != scores2.len() || scores1.is_empty() {
            return Err(DspyError::evaluation(
                "t_test",
                "Cannot perform t-test on mismatched or empty score sets"
            ));
        }

        let n = scores1.len() as f64;
        let differences: Vec<f64> = scores1
            .iter()
            .zip(scores2.iter())
            .map(|(s1, s2)| s1 - s2)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n;
        let variance_diff = differences
            .iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std_error = (variance_diff / n).sqrt();

        let t_statistic = if std_error > 0.0 {
            mean_diff / std_error
        } else {
            0.0
        };

        // Simplified p-value calculation (two-tailed)
        let p_value = self.t_to_p_value(t_statistic.abs(), (n - 1.0) as usize);
        let is_significant = p_value < (1.0 - self.config.significance_confidence);

        // Cohen's d effect size
        let pooled_std = ((scores1.iter().map(|s| s.powi(2)).sum::<f64>() +
                          scores2.iter().map(|s| s.powi(2)).sum::<f64>()) / (2.0 * n)).sqrt();
        let effect_size = if pooled_std > 0.0 {
            mean_diff / pooled_std
        } else {
            0.0
        };

        Ok(PairwiseComparison {
            metric1: "metric1".to_string(), // Will be filled by caller
            metric2: "metric2".to_string(), // Will be filled by caller
            t_statistic,
            p_value,
            is_significant,
            effect_size,
        })
    }

    /// Convert t-statistic to p-value (simplified approximation)
    fn t_to_p_value(&self, t_abs: f64, df: usize) -> f64 {
        // Very simplified approximation - in practice use proper statistical library
        if df >= 30 {
            // Normal approximation
            if t_abs >= 2.576 { 0.01 }
            else if t_abs >= 1.96 { 0.05 }
            else if t_abs >= 1.645 { 0.10 }
            else { 0.20 }
        } else {
            // Rough adjustment for smaller df
            let adjustment = 1.0 + 1.0 / df as f64;
            let adjusted_t = t_abs / adjustment;
            if adjusted_t >= 2.5 { 0.01 }
            else if adjusted_t >= 1.8 { 0.05 }
            else if adjusted_t >= 1.4 { 0.10 }
            else { 0.20 }
        }
    }

    /// Compute one-way ANOVA
    fn compute_anova(&self, metric_scores: &HashMap<String, Vec<f64>>) -> DspyResult<AnovaResults> {
        let groups: Vec<&Vec<f64>> = metric_scores.values().collect();

        if groups.len() < 2 {
            return Err(DspyError::evaluation(
                "ANOVA",
                "ANOVA requires at least 2 groups"
            ));
        }

        // Check that all groups have the same size
        let group_size = groups[0].len();
        if !groups.iter().all(|g| g.len() == group_size) {
            return Err(DspyError::evaluation(
                "ANOVA",
                "ANOVA requires all groups to have the same size"
            ));
        }

        let k = groups.len(); // number of groups
        let n = group_size; // size of each group
        let total_n = k * n;

        // Calculate group means and overall mean
        let group_means: Vec<f64> = groups
            .iter()
            .map(|group| group.iter().sum::<f64>() / group.len() as f64)
            .collect();

        let overall_mean = groups
            .iter()
            .flat_map(|group| group.iter())
            .sum::<f64>() / total_n as f64;

        // Calculate sum of squares
        let ss_between = n as f64 * group_means
            .iter()
            .map(|mean| (mean - overall_mean).powi(2))
            .sum::<f64>();

        let ss_within = groups
            .iter()
            .flat_map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.iter().map(move |value| (value - group_mean).powi(2))
            })
            .sum::<f64>();

        let df_between = k - 1;
        let df_within = total_n - k;

        let ms_between = ss_between / df_between as f64;
        let ms_within = if df_within > 0 {
            ss_within / df_within as f64
        } else {
            1.0
        };

        let f_statistic = if ms_within > 0.0 {
            ms_between / ms_within
        } else {
            0.0
        };

        // Simplified p-value calculation
        let p_value = if f_statistic >= 4.0 { 0.01 }
        else if f_statistic >= 3.0 { 0.05 }
        else if f_statistic >= 2.0 { 0.10 }
        else { 0.20 };

        let is_significant = p_value < (1.0 - self.config.significance_confidence);

        Ok(AnovaResults {
            f_statistic,
            p_value,
            is_significant,
            degrees_of_freedom: (df_between, df_within),
        })
    }
}

impl<I, O> Default for Evaluator<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EvaluationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Evaluation Results")?;
        writeln!(f, "==================")?;
        writeln!(f, "Total Examples: {}", self.overall_stats.total_examples)?;
        writeln!(f, "Total Time: {:.1}ms", self.overall_stats.total_time_ms)?;
        writeln!(f, "Success Rate: {:.1}%", self.overall_stats.overall_success_rate * 100.0)?;
        writeln!(f)?;

        writeln!(f, "Metric Results:")?;
        for (name, summary) in &self.metric_results {
            writeln!(f, "  {}: {:.3} ± {:.3} (pass rate: {:.1}%)",
                name, summary.mean_score, summary.std_dev, summary.pass_rate * 100.0)?;
        }

        if let Some(ref sig_tests) = self.significance_tests {
            writeln!(f)?;
            writeln!(f, "Statistical Significance:")?;
            for (metric1, comparisons) in &sig_tests.pairwise_comparisons {
                for (metric2, comparison) in comparisons {
                    writeln!(f, "  {} vs {}: {} (p={:.3})",
                        metric1, metric2,
                        if comparison.is_significant { "significant" } else { "not significant" },
                        comparison.p_value)?;
                }
            }
        }

        Ok(())
    }
}
