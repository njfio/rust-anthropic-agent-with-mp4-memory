//! Example management system for DSPy optimization
//!
//! This module provides functionality for managing training examples used in
//! DSPy module optimization and teleprompter training.

use crate::dspy::error::{DspyError, DspyResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// A training example consisting of input and expected output
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Example<I, O> {
    /// Unique identifier for the example
    pub id: String,
    /// Input data
    pub input: I,
    /// Expected output
    pub output: O,
    /// Optional metadata for the example
    pub metadata: HashMap<String, serde_json::Value>,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Whether this example has been validated
    pub validated: bool,
}

impl<I, O> Example<I, O> {
    /// Create a new example
    pub fn new(input: I, output: O) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            input,
            output,
            metadata: HashMap::new(),
            quality_score: 1.0,
            validated: false,
        }
    }

    /// Create a new example with metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set the quality score
    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Mark as validated
    pub fn validated(mut self) -> Self {
        self.validated = true;
        self
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

impl<I, O> fmt::Display for Example<I, O>
where
    I: fmt::Debug,
    O: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Example[{}] (quality: {:.2}, validated: {})",
            &self.id[..8],
            self.quality_score,
            self.validated
        )
    }
}

/// Collection of examples with management capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleSet<I, O> {
    /// All examples in the set
    examples: Vec<Example<I, O>>,
    /// Set metadata
    metadata: HashMap<String, serde_json::Value>,
    /// Validation statistics
    validation_stats: ValidationStats,
}

/// Statistics about example validation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationStats {
    /// Total number of examples
    pub total_count: usize,
    /// Number of validated examples
    pub validated_count: usize,
    /// Number of high-quality examples (score >= 0.8)
    pub high_quality_count: usize,
    /// Average quality score
    pub average_quality: f64,
    /// Minimum quality score
    pub min_quality: f64,
    /// Maximum quality score
    pub max_quality: f64,
}

impl<I, O> ExampleSet<I, O> {
    /// Create a new empty example set
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            metadata: HashMap::new(),
            validation_stats: ValidationStats::default(),
        }
    }

    /// Create from a vector of examples
    pub fn from_examples(examples: Vec<Example<I, O>>) -> Self {
        let mut set = Self::new();
        for example in examples {
            set.add_example(example);
        }
        set
    }

    /// Add an example to the set
    pub fn add_example(&mut self, example: Example<I, O>) {
        debug!("Adding example {} to set", example.id);
        self.examples.push(example);
        self.update_stats();
    }

    /// Add multiple examples
    pub fn add_examples(&mut self, examples: Vec<Example<I, O>>) {
        for example in examples {
            self.examples.push(example);
        }
        self.update_stats();
        info!("Added {} examples to set", self.examples.len());
    }

    /// Get all examples
    pub fn examples(&self) -> &[Example<I, O>] {
        &self.examples
    }

    /// Get examples by quality threshold
    pub fn examples_by_quality(&self, min_quality: f64) -> Vec<&Example<I, O>> {
        self.examples
            .iter()
            .filter(|e| e.quality_score >= min_quality)
            .collect()
    }

    /// Get validated examples only
    pub fn validated_examples(&self) -> Vec<&Example<I, O>> {
        self.examples.iter().filter(|e| e.validated).collect()
    }

    /// Get example by ID
    pub fn get_example(&self, id: &str) -> Option<&Example<I, O>> {
        self.examples.iter().find(|e| e.id == id)
    }

    /// Remove example by ID
    pub fn remove_example(&mut self, id: &str) -> Option<Example<I, O>> {
        if let Some(pos) = self.examples.iter().position(|e| e.id == id) {
            let example = self.examples.remove(pos);
            self.update_stats();
            debug!("Removed example {} from set", id);
            Some(example)
        } else {
            None
        }
    }

    /// Split into training and validation sets
    pub fn split(&self, train_ratio: f64) -> DspyResult<(ExampleSet<I, O>, ExampleSet<I, O>)>
    where
        I: Clone,
        O: Clone,
    {
        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(DspyError::configuration(
                "train_ratio",
                "Must be between 0.0 and 1.0",
            ));
        }

        let total = self.examples.len();
        let train_size = (total as f64 * train_ratio).round() as usize;

        let mut train_examples = Vec::new();
        let mut val_examples = Vec::new();

        for (i, example) in self.examples.iter().enumerate() {
            if i < train_size {
                train_examples.push(example.clone());
            } else {
                val_examples.push(example.clone());
            }
        }

        Ok((
            ExampleSet::from_examples(train_examples),
            ExampleSet::from_examples(val_examples),
        ))
    }

    /// Shuffle examples randomly
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.examples.shuffle(&mut rng);
        debug!("Shuffled {} examples", self.examples.len());
    }

    /// Filter examples by predicate
    pub fn filter<F>(&self, predicate: F) -> ExampleSet<I, O>
    where
        F: Fn(&Example<I, O>) -> bool,
        I: Clone,
        O: Clone,
    {
        let filtered: Vec<Example<I, O>> = self
            .examples
            .iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();

        ExampleSet::from_examples(filtered)
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ValidationStats {
        &self.validation_stats
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        let total = self.examples.len();
        let validated = self.examples.iter().filter(|e| e.validated).count();
        let high_quality = self
            .examples
            .iter()
            .filter(|e| e.quality_score >= 0.8)
            .count();

        let qualities: Vec<f64> = self.examples.iter().map(|e| e.quality_score).collect();
        let avg_quality = if qualities.is_empty() {
            0.0
        } else {
            qualities.iter().sum::<f64>() / qualities.len() as f64
        };

        let min_quality = qualities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_quality = qualities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        self.validation_stats = ValidationStats {
            total_count: total,
            validated_count: validated,
            high_quality_count: high_quality,
            average_quality: avg_quality,
            min_quality: if min_quality.is_finite() {
                min_quality
            } else {
                0.0
            },
            max_quality: if max_quality.is_finite() {
                max_quality
            } else {
                0.0
            },
        };
    }

    /// Validate all examples using a validation function
    pub fn validate_examples<F>(&mut self, validator: F) -> DspyResult<()>
    where
        F: Fn(&Example<I, O>) -> DspyResult<bool>,
    {
        let mut validated_count = 0;
        let mut failed_count = 0;

        for example in &mut self.examples {
            match validator(example) {
                Ok(is_valid) => {
                    if is_valid {
                        example.validated = true;
                        validated_count += 1;
                    } else {
                        warn!("Example {} failed validation", example.id);
                        failed_count += 1;
                    }
                }
                Err(e) => {
                    warn!("Validation error for example {}: {}", example.id, e);
                    failed_count += 1;
                }
            }
        }

        self.update_stats();
        info!(
            "Validation complete: {} validated, {} failed",
            validated_count, failed_count
        );

        Ok(())
    }

    /// Set metadata for the example set
    pub fn set_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Check if the example set is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }
}

impl<I, O> Default for ExampleSet<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I, O> fmt::Display for ExampleSet<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ExampleSet[{} examples, {:.1}% validated, avg quality: {:.2}]",
            self.validation_stats.total_count,
            if self.validation_stats.total_count > 0 {
                (self.validation_stats.validated_count as f64
                    / self.validation_stats.total_count as f64)
                    * 100.0
            } else {
                0.0
            },
            self.validation_stats.average_quality
        )
    }
}
