//! DSPy metrics framework for evaluating module outputs
//!
//! This module provides a comprehensive set of evaluation metrics for DSPy modules,
//! including exact match, semantic similarity, F1 score, and custom metrics.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::Example;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Core trait for evaluation metrics
pub trait Metric<I, O>: Send + Sync
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    /// Evaluate a prediction against an example
    fn evaluate(&self, example: &Example<I, O>, prediction: &O) -> DspyResult<MetricResult>;

    /// Get the metric name
    fn name(&self) -> &str;

    /// Get the metric description
    fn description(&self) -> &str {
        "No description provided"
    }

    /// Check if higher scores are better (default: true)
    fn higher_is_better(&self) -> bool {
        true
    }

    /// Get the expected score range
    fn score_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Validate the metric configuration
    fn validate(&self) -> DspyResult<()> {
        Ok(())
    }
}

/// Result of a metric evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// The metric score
    pub score: f64,
    /// Whether the evaluation passed (for binary metrics)
    pub passed: bool,
    /// Additional details about the evaluation
    pub details: HashMap<String, serde_json::Value>,
    /// Confidence in the result (0.0 to 1.0)
    pub confidence: f64,
}

impl MetricResult {
    /// Create a new metric result
    pub fn new(score: f64) -> Self {
        Self {
            score,
            passed: score > 0.5, // Default threshold
            details: HashMap::new(),
            confidence: 1.0,
        }
    }

    /// Create with pass/fail status
    pub fn with_passed(mut self, passed: bool) -> Self {
        self.passed = passed;
        self
    }

    /// Add detail information
    pub fn with_detail<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Create a passing result
    pub fn pass(score: f64) -> Self {
        Self::new(score).with_passed(true)
    }

    /// Create a failing result
    pub fn fail(score: f64) -> Self {
        Self::new(score).with_passed(false)
    }
}

/// Exact match metric - checks if outputs are exactly equal
pub struct ExactMatch {
    /// Whether to ignore case differences
    pub case_sensitive: bool,
    /// Whether to trim whitespace
    pub trim_whitespace: bool,
    /// Whether to normalize punctuation
    pub normalize_punctuation: bool,
}

impl Default for ExactMatch {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            trim_whitespace: true,
            normalize_punctuation: false,
        }
    }
}

impl ExactMatch {
    /// Create a new exact match metric
    pub fn new() -> Self {
        Self::default()
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Set whitespace trimming
    pub fn trim_whitespace(mut self, trim_whitespace: bool) -> Self {
        self.trim_whitespace = trim_whitespace;
        self
    }

    /// Set punctuation normalization
    pub fn normalize_punctuation(mut self, normalize_punctuation: bool) -> Self {
        self.normalize_punctuation = normalize_punctuation;
        self
    }

    /// Normalize text according to configuration
    fn normalize_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        if self.trim_whitespace {
            result = result.trim().to_string();
        }

        if !self.case_sensitive {
            result = result.to_lowercase();
        }

        if self.normalize_punctuation {
            // Remove common punctuation
            result = result
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        result
    }
}

impl<I, O> Metric<I, O> for ExactMatch
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync + fmt::Display,
{
    fn evaluate(&self, example: &Example<I, O>, prediction: &O) -> DspyResult<MetricResult> {
        let expected = self.normalize_text(&example.output.to_string());
        let predicted = self.normalize_text(&prediction.to_string());

        let matches = expected == predicted;
        let score = if matches { 1.0 } else { 0.0 };

        debug!(
            "ExactMatch evaluation: expected='{}', predicted='{}', matches={}",
            expected, predicted, matches
        );

        Ok(MetricResult::new(score)
            .with_passed(matches)
            .with_detail("expected", expected)
            .with_detail("predicted", predicted)
            .with_detail("normalized", true))
    }

    fn name(&self) -> &str {
        "ExactMatch"
    }

    fn description(&self) -> &str {
        "Checks if the predicted output exactly matches the expected output"
    }
}

/// Semantic similarity metric using text comparison
pub struct SemanticSimilarity {
    /// Minimum similarity threshold for passing
    pub threshold: f64,
    /// Similarity algorithm to use
    pub algorithm: SimilarityAlgorithm,
}

/// Algorithms for computing semantic similarity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityAlgorithm {
    /// Jaccard similarity (intersection over union of words)
    Jaccard,
    /// Cosine similarity of word frequency vectors
    Cosine,
    /// Levenshtein distance normalized
    Levenshtein,
    /// Simple word overlap ratio
    WordOverlap,
}

impl Default for SemanticSimilarity {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            algorithm: SimilarityAlgorithm::Jaccard,
        }
    }
}

impl SemanticSimilarity {
    /// Create a new semantic similarity metric
    pub fn new() -> Self {
        Self::default()
    }

    /// Set similarity threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set similarity algorithm
    pub fn with_algorithm(mut self, algorithm: SimilarityAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Compute Jaccard similarity
    fn jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            1.0 // Both empty
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Compute word overlap ratio
    fn word_overlap(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let total = words1.len() + words2.len();

        if total == 0 {
            1.0
        } else {
            (2 * intersection) as f64 / total as f64
        }
    }

    /// Compute Levenshtein distance normalized
    fn levenshtein_similarity(&self, text1: &str, text2: &str) -> f64 {
        let distance = self.levenshtein_distance(text1, text2);
        let max_len = text1.len().max(text2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Compute Levenshtein distance
    fn levenshtein_distance(&self, text1: &str, text2: &str) -> usize {
        let chars1: Vec<char> = text1.chars().collect();
        let chars2: Vec<char> = text2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Compute cosine similarity of word frequency vectors
    fn cosine_similarity(&self, text1: &str, text2: &str) -> f64 {
        let mut word_counts1: HashMap<&str, usize> = HashMap::new();
        let mut word_counts2: HashMap<&str, usize> = HashMap::new();

        for word in text1.split_whitespace() {
            *word_counts1.entry(word).or_insert(0) += 1;
        }
        for word in text2.split_whitespace() {
            *word_counts2.entry(word).or_insert(0) += 1;
        }

        let mut all_words: std::collections::HashSet<&str> = std::collections::HashSet::new();
        all_words.extend(word_counts1.keys());
        all_words.extend(word_counts2.keys());

        if all_words.is_empty() {
            return 1.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for word in all_words {
            let count1 = *word_counts1.get(word).unwrap_or(&0) as f64;
            let count2 = *word_counts2.get(word).unwrap_or(&0) as f64;

            dot_product += count1 * count2;
            norm1 += count1 * count1;
            norm2 += count2 * count2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        }
    }

    /// Compute similarity using the configured algorithm
    fn compute_similarity(&self, text1: &str, text2: &str) -> f64 {
        match self.algorithm {
            SimilarityAlgorithm::Jaccard => self.jaccard_similarity(text1, text2),
            SimilarityAlgorithm::Cosine => self.cosine_similarity(text1, text2),
            SimilarityAlgorithm::Levenshtein => self.levenshtein_similarity(text1, text2),
            SimilarityAlgorithm::WordOverlap => self.word_overlap(text1, text2),
        }
    }
}

impl<I, O> Metric<I, O> for SemanticSimilarity
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync + fmt::Display,
{
    fn evaluate(&self, example: &Example<I, O>, prediction: &O) -> DspyResult<MetricResult> {
        let expected = example.output.to_string().to_lowercase();
        let predicted = prediction.to_string().to_lowercase();

        let similarity = self.compute_similarity(&expected, &predicted);
        let passed = similarity >= self.threshold;

        debug!(
            "SemanticSimilarity evaluation: similarity={:.3}, threshold={:.3}, passed={}",
            similarity, self.threshold, passed
        );

        Ok(MetricResult::new(similarity)
            .with_passed(passed)
            .with_detail("algorithm", format!("{:?}", self.algorithm))
            .with_detail("threshold", self.threshold)
            .with_detail("expected", expected)
            .with_detail("predicted", predicted))
    }

    fn name(&self) -> &str {
        "SemanticSimilarity"
    }

    fn description(&self) -> &str {
        "Measures semantic similarity between expected and predicted outputs"
    }

    fn validate(&self) -> DspyResult<()> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(DspyError::configuration(
                "threshold",
                "Threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

/// F1 Score metric combining precision and recall
pub struct F1Score {
    /// How to extract tokens for comparison
    pub tokenization: TokenizationStrategy,
    /// Whether to use micro or macro averaging
    pub averaging: AveragingStrategy,
    /// Minimum F1 score threshold for passing
    pub threshold: f64,
}

/// Strategies for tokenizing text for F1 calculation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TokenizationStrategy {
    /// Split by whitespace
    Whitespace,
    /// Split by words (alphanumeric sequences)
    Words,
    /// Split by characters
    Characters,
    /// Split by sentences
    Sentences,
}

/// Strategies for averaging F1 scores
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AveragingStrategy {
    /// Micro-averaging (global precision and recall)
    Micro,
    /// Macro-averaging (average of individual F1 scores)
    Macro,
    /// Weighted averaging
    Weighted,
}

impl Default for F1Score {
    fn default() -> Self {
        Self {
            tokenization: TokenizationStrategy::Words,
            averaging: AveragingStrategy::Micro,
            threshold: 0.5,
        }
    }
}

impl F1Score {
    /// Create a new F1 score metric
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tokenization strategy
    pub fn with_tokenization(mut self, tokenization: TokenizationStrategy) -> Self {
        self.tokenization = tokenization;
        self
    }

    /// Set averaging strategy
    pub fn with_averaging(mut self, averaging: AveragingStrategy) -> Self {
        self.averaging = averaging;
        self
    }

    /// Set threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Tokenize text according to strategy
    fn tokenize(&self, text: &str) -> Vec<String> {
        match self.tokenization {
            TokenizationStrategy::Whitespace => {
                text.split_whitespace().map(|s| s.to_string()).collect()
            }
            TokenizationStrategy::Words => text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_lowercase())
                .collect(),
            TokenizationStrategy::Characters => text.chars().map(|c| c.to_string()).collect(),
            TokenizationStrategy::Sentences => text
                .split(&['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().to_string())
                .collect(),
        }
    }

    /// Calculate precision, recall, and F1 score
    fn calculate_f1(
        &self,
        expected_tokens: &[String],
        predicted_tokens: &[String],
    ) -> (f64, f64, f64) {
        let expected_set: std::collections::HashSet<_> = expected_tokens.iter().collect();
        let predicted_set: std::collections::HashSet<_> = predicted_tokens.iter().collect();

        let true_positives = expected_set.intersection(&predicted_set).count() as f64;
        let false_positives = predicted_set.difference(&expected_set).count() as f64;
        let false_negatives = expected_set.difference(&predicted_set).count() as f64;

        let precision = if true_positives + false_positives > 0.0 {
            true_positives / (true_positives + false_positives)
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0.0 {
            true_positives / (true_positives + false_negatives)
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }
}

impl<I, O> Metric<I, O> for F1Score
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync + fmt::Display,
{
    fn evaluate(&self, example: &Example<I, O>, prediction: &O) -> DspyResult<MetricResult> {
        let expected_text = example.output.to_string();
        let predicted_text = prediction.to_string();

        let expected_tokens = self.tokenize(&expected_text);
        let predicted_tokens = self.tokenize(&predicted_text);

        let (precision, recall, f1) = self.calculate_f1(&expected_tokens, &predicted_tokens);
        let passed = f1 >= self.threshold;

        debug!(
            "F1Score evaluation: precision={:.3}, recall={:.3}, f1={:.3}, threshold={:.3}, passed={}",
            precision, recall, f1, self.threshold, passed
        );

        Ok(MetricResult::new(f1)
            .with_passed(passed)
            .with_detail("precision", precision)
            .with_detail("recall", recall)
            .with_detail("tokenization", format!("{:?}", self.tokenization))
            .with_detail("averaging", format!("{:?}", self.averaging))
            .with_detail("expected_tokens", expected_tokens.len())
            .with_detail("predicted_tokens", predicted_tokens.len()))
    }

    fn name(&self) -> &str {
        "F1Score"
    }

    fn description(&self) -> &str {
        "Calculates F1 score (harmonic mean of precision and recall)"
    }

    fn validate(&self) -> DspyResult<()> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(DspyError::configuration(
                "threshold",
                "Threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

/// Custom metric that combines multiple metrics
pub struct CompositeMetric<I, O> {
    /// Name of the composite metric
    name: String,
    /// Description of the composite metric
    description: String,
    /// Component metrics with their weights
    metrics: Vec<(Arc<dyn Metric<I, O>>, f64)>,
    /// How to combine the metrics
    combination_strategy: CombinationStrategy,
}

/// Strategies for combining multiple metrics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Weighted average of scores
    WeightedAverage,
    /// Minimum score across all metrics
    Minimum,
    /// Maximum score across all metrics
    Maximum,
    /// All metrics must pass
    AllPass,
    /// At least one metric must pass
    AnyPass,
}

impl<I, O> CompositeMetric<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    /// Create a new composite metric
    pub fn new<S: Into<String>>(name: S, description: S) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            metrics: Vec::new(),
            combination_strategy: CombinationStrategy::WeightedAverage,
        }
    }

    /// Add a metric with weight
    pub fn add_metric(mut self, metric: Arc<dyn Metric<I, O>>, weight: f64) -> Self {
        self.metrics.push((metric, weight));
        self
    }

    /// Set combination strategy
    pub fn with_strategy(mut self, strategy: CombinationStrategy) -> Self {
        self.combination_strategy = strategy;
        self
    }

    /// Normalize weights to sum to 1.0
    fn normalize_weights(&self) -> Vec<f64> {
        let total_weight: f64 = self.metrics.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            self.metrics.iter().map(|(_, w)| w / total_weight).collect()
        } else {
            vec![1.0 / self.metrics.len() as f64; self.metrics.len()]
        }
    }
}

impl<I, O> Metric<I, O> for CompositeMetric<I, O>
where
    I: Clone + Send + Sync,
    O: Clone + Send + Sync,
{
    fn evaluate(&self, example: &Example<I, O>, prediction: &O) -> DspyResult<MetricResult> {
        if self.metrics.is_empty() {
            return Err(DspyError::configuration(
                "metrics",
                "Composite metric must have at least one component metric",
            ));
        }

        let mut results = Vec::new();
        let mut all_details = HashMap::new();

        // Evaluate all component metrics
        for (metric, _) in &self.metrics {
            let result = metric.evaluate(example, prediction)?;
            results.push(result.clone());

            // Collect details with metric name prefix
            for (key, value) in result.details {
                all_details.insert(format!("{}_{}", metric.name(), key), value);
            }
        }

        // Combine results according to strategy
        let (final_score, passed) = match self.combination_strategy {
            CombinationStrategy::WeightedAverage => {
                let weights = self.normalize_weights();
                let weighted_score: f64 = results
                    .iter()
                    .zip(weights.iter())
                    .map(|(result, weight)| result.score * weight)
                    .sum();
                let all_passed = results.iter().all(|r| r.passed);
                (weighted_score, all_passed)
            }
            CombinationStrategy::Minimum => {
                let min_score = results
                    .iter()
                    .map(|r| r.score)
                    .fold(f64::INFINITY, f64::min);
                let all_passed = results.iter().all(|r| r.passed);
                (min_score, all_passed)
            }
            CombinationStrategy::Maximum => {
                let max_score = results
                    .iter()
                    .map(|r| r.score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let any_passed = results.iter().any(|r| r.passed);
                (max_score, any_passed)
            }
            CombinationStrategy::AllPass => {
                let all_passed = results.iter().all(|r| r.passed);
                let avg_score: f64 =
                    results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
                (avg_score, all_passed)
            }
            CombinationStrategy::AnyPass => {
                let any_passed = results.iter().any(|r| r.passed);
                let avg_score: f64 =
                    results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
                (avg_score, any_passed)
            }
        };

        // Add component scores to details
        for (i, (metric, _)) in self.metrics.iter().enumerate() {
            all_details.insert(
                format!("{}_score", metric.name()),
                serde_json::json!(results[i].score),
            );
            all_details.insert(
                format!("{}_passed", metric.name()),
                serde_json::json!(results[i].passed),
            );
        }

        all_details.insert(
            "combination_strategy".to_string(),
            serde_json::json!(format!("{:?}", self.combination_strategy)),
        );

        let mut result = MetricResult::new(final_score)
            .with_passed(passed)
            .with_detail("component_count", self.metrics.len())
            .with_detail("strategy", format!("{:?}", self.combination_strategy));

        // Add all collected details
        for (key, value) in all_details {
            result.details.insert(key, value);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn validate(&self) -> DspyResult<()> {
        if self.metrics.is_empty() {
            return Err(DspyError::configuration(
                "metrics",
                "Composite metric must have at least one component metric",
            ));
        }

        // Validate all component metrics
        for (metric, weight) in &self.metrics {
            metric.validate()?;
            if *weight < 0.0 {
                return Err(DspyError::configuration(
                    "weight",
                    "Metric weights must be non-negative",
                ));
            }
        }

        Ok(())
    }
}
