//! Multi-Modal Evaluation Metrics for DSPy
//!
//! This module provides specialized evaluation metrics for multi-modal DSPy modules,
//! including image-text alignment, audio-text coherence, and cross-modal consistency.

use crate::dspy::{
    error::{DspyError, DspyResult},
    examples::Example,
    metrics::{Metric, MetricResult},
    multimodal::{MediaType, MultiModalInput, MultiModalOutput},
    vision::{VisionInput, VisionOutput},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-modal evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiModalMetric {
    ImageTextAlignment,
    AudioTextCoherence,
    CrossModalConsistency,
    VisualQuestionAccuracy,
    ImageCaptionQuality,
    MultiModalFaithfulness,
}

/// Image-text alignment metric
#[derive(Debug, Clone)]
pub struct ImageTextAlignmentMetric {
    pub threshold: f64,
    pub weight_semantic: f64,
    pub weight_visual: f64,
}

impl Default for ImageTextAlignmentMetric {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            weight_semantic: 0.6,
            weight_visual: 0.4,
        }
    }
}

impl ImageTextAlignmentMetric {
    /// Create new image-text alignment metric
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Set weights for semantic and visual components
    pub fn with_weights(mut self, semantic: f64, visual: f64) -> Self {
        let total = semantic + visual;
        self.weight_semantic = semantic / total;
        self.weight_visual = visual / total;
        self
    }

    /// Calculate semantic alignment score
    fn calculate_semantic_alignment(&self, text: &str, image_description: &str) -> f64 {
        // In a real implementation, this would use semantic similarity models
        // For now, we'll use a simple keyword overlap approach
        let text_lower = text.to_lowercase();
        let desc_lower = image_description.to_lowercase();
        let text_words: Vec<&str> = text_lower.split_whitespace().collect();
        let desc_words: Vec<&str> = desc_lower.split_whitespace().collect();

        let common_words = text_words
            .iter()
            .filter(|word| desc_words.contains(word))
            .count();

        let total_words = text_words.len().max(desc_words.len());
        if total_words == 0 {
            0.0
        } else {
            common_words as f64 / total_words as f64
        }
    }

    /// Calculate visual alignment score
    fn calculate_visual_alignment(
        &self,
        _text: &str,
        _image_features: &HashMap<String, f64>,
    ) -> f64 {
        // In a real implementation, this would analyze visual features
        // For now, return a placeholder score
        0.75
    }
}

impl Metric<MultiModalInput, MultiModalOutput> for ImageTextAlignmentMetric {
    fn name(&self) -> &str {
        "ImageTextAlignment"
    }

    fn evaluate(
        &self,
        example: &Example<MultiModalInput, MultiModalOutput>,
        prediction: &MultiModalOutput,
    ) -> DspyResult<MetricResult> {
        let input = &example.input;
        let output = prediction;
        // Check if input contains images
        if !input.has_media_type(&MediaType::Image) {
            return Err(DspyError::invalid_input(
                "ImageTextAlignment metric requires image input",
            ));
        }

        // Get image analysis from output
        let image_description = output
            .media_analysis
            .values()
            .find_map(|analysis| analysis.get("scene_description").and_then(|v| v.as_str()))
            .unwrap_or("");

        // Calculate semantic alignment
        let semantic_score = self.calculate_semantic_alignment(&input.text, image_description);

        // Calculate visual alignment (placeholder)
        let visual_features = HashMap::new(); // Would extract from image
        let visual_score = self.calculate_visual_alignment(&input.text, &visual_features);

        // Combine scores
        let overall_score =
            (semantic_score * self.weight_semantic) + (visual_score * self.weight_visual);

        let passed = overall_score >= self.threshold;

        let mut details = HashMap::new();
        details.insert(
            "semantic_score".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(semantic_score).unwrap()),
        );
        details.insert(
            "visual_score".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(visual_score).unwrap()),
        );
        details.insert(
            "threshold".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.threshold).unwrap()),
        );

        Ok(MetricResult {
            score: overall_score,
            passed,
            confidence: overall_score, // Use score as confidence
            details,
        })
    }
}

/// Visual question answering accuracy metric
#[derive(Debug, Clone)]
pub struct VisualQuestionAccuracyMetric {
    pub threshold: f64,
    pub exact_match_weight: f64,
    pub semantic_weight: f64,
}

impl Default for VisualQuestionAccuracyMetric {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            exact_match_weight: 0.3,
            semantic_weight: 0.7,
        }
    }
}

impl VisualQuestionAccuracyMetric {
    /// Create new VQA accuracy metric
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Calculate answer accuracy
    fn calculate_accuracy(&self, predicted: &str, expected: &str) -> f64 {
        // Exact match component
        let exact_match = if predicted.trim().to_lowercase() == expected.trim().to_lowercase() {
            1.0
        } else {
            0.0
        };

        // Semantic similarity component (simplified)
        let pred_lower = predicted.to_lowercase();
        let exp_lower = expected.to_lowercase();
        let pred_words: Vec<&str> = pred_lower.split_whitespace().collect();
        let exp_words: Vec<&str> = exp_lower.split_whitespace().collect();

        let common_words = pred_words
            .iter()
            .filter(|word| exp_words.contains(word))
            .count();

        let total_words = pred_words.len().max(exp_words.len());
        let semantic_sim = if total_words == 0 {
            0.0
        } else {
            common_words as f64 / total_words as f64
        };

        (exact_match * self.exact_match_weight) + (semantic_sim * self.semantic_weight)
    }
}

impl Metric<VisionInput, VisionOutput> for VisualQuestionAccuracyMetric {
    fn name(&self) -> &str {
        "VisualQuestionAccuracy"
    }

    fn evaluate(
        &self,
        example: &Example<VisionInput, VisionOutput>,
        prediction: &VisionOutput,
    ) -> DspyResult<MetricResult> {
        let input = &example.input;
        let output = prediction;
        // For evaluation, we need expected answer in metadata
        let expected_answer = input
            .metadata
            .get("expected_answer")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                DspyError::invalid_input("Expected answer not found in input metadata")
            })?;

        let accuracy = self.calculate_accuracy(&output.analysis, expected_answer);
        let passed = accuracy >= self.threshold;

        let mut details = HashMap::new();
        details.insert(
            "accuracy".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(accuracy).unwrap()),
        );
        details.insert(
            "threshold".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.threshold).unwrap()),
        );
        details.insert(
            "predicted_answer".to_string(),
            serde_json::Value::String(output.analysis.clone()),
        );
        details.insert(
            "expected_answer".to_string(),
            serde_json::Value::String(expected_answer.to_string()),
        );

        Ok(MetricResult {
            score: accuracy,
            passed,
            confidence: accuracy, // Use accuracy as confidence
            details,
        })
    }
}

/// Cross-modal consistency metric
#[derive(Debug, Clone)]
pub struct CrossModalConsistencyMetric {
    pub threshold: f64,
    pub modality_weights: HashMap<MediaType, f64>,
}

impl Default for CrossModalConsistencyMetric {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(MediaType::Image, 0.4);
        weights.insert(MediaType::Audio, 0.3);
        weights.insert(MediaType::Video, 0.3);

        Self {
            threshold: 0.75,
            modality_weights: weights,
        }
    }
}

impl CrossModalConsistencyMetric {
    /// Create new cross-modal consistency metric
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Calculate consistency across modalities
    fn calculate_consistency(&self, input: &MultiModalInput, output: &MultiModalOutput) -> f64 {
        let mut consistency_scores = Vec::new();
        let mut total_weight = 0.0;

        // Check consistency between text and each media type
        for media in &input.media {
            if let Some(weight) = self.modality_weights.get(&media.media_type) {
                let consistency = self.calculate_modality_consistency(&input.text, media, output);
                consistency_scores.push(consistency * weight);
                total_weight += weight;
            }
        }

        if total_weight == 0.0 {
            0.0
        } else {
            consistency_scores.iter().sum::<f64>() / total_weight
        }
    }

    /// Calculate consistency for a specific modality
    fn calculate_modality_consistency(
        &self,
        text: &str,
        _media: &crate::dspy::multimodal::MediaContent,
        output: &MultiModalOutput,
    ) -> f64 {
        // In a real implementation, this would analyze cross-modal features
        // For now, check if output mentions relevant concepts
        let text_lower = text.to_lowercase();
        let output_lower = output.text.to_lowercase();

        // Simple keyword overlap
        let text_words: Vec<&str> = text_lower.split_whitespace().collect();
        let output_words: Vec<&str> = output_lower.split_whitespace().collect();

        let common_words = text_words
            .iter()
            .filter(|word| output_words.contains(word))
            .count();

        let total_words = text_words.len().max(output_words.len());
        if total_words == 0 {
            0.0
        } else {
            common_words as f64 / total_words as f64
        }
    }
}

impl Metric<MultiModalInput, MultiModalOutput> for CrossModalConsistencyMetric {
    fn name(&self) -> &str {
        "CrossModalConsistency"
    }

    fn evaluate(
        &self,
        example: &Example<MultiModalInput, MultiModalOutput>,
        prediction: &MultiModalOutput,
    ) -> DspyResult<MetricResult> {
        let input = &example.input;
        let output = prediction;
        if input.media.is_empty() {
            return Err(DspyError::invalid_input(
                "CrossModalConsistency metric requires media input",
            ));
        }

        let consistency = self.calculate_consistency(input, output);
        let passed = consistency >= self.threshold;

        let mut details = HashMap::new();
        details.insert(
            "consistency_score".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(consistency).unwrap()),
        );
        details.insert(
            "threshold".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.threshold).unwrap()),
        );
        details.insert(
            "media_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(input.media.len())),
        );

        Ok(MetricResult {
            score: consistency,
            passed,
            confidence: consistency, // Use consistency as confidence
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspy::multimodal::{MediaContent, MultiModalInput, MultiModalOutput};

    fn create_test_image() -> MediaContent {
        MediaContent::from_bytes(b"test image data".to_vec(), "image/jpeg".to_string()).unwrap()
    }

    #[test]
    fn test_image_text_alignment_metric() {
        let metric = ImageTextAlignmentMetric::new(0.7);

        let image = create_test_image();
        let input = MultiModalInput::new("A cat sitting on a chair".to_string()).add_media(image);

        let mut output = MultiModalOutput::new("I see a cat on furniture".to_string(), 0.9);
        output = output.add_media_analysis(
            "image_0".to_string(),
            serde_json::json!({"scene_description": "A cat sitting on a chair"}),
        );

        let example = Example::new(input, output.clone());
        let result = metric.evaluate(&example, &output).unwrap();
        assert!(result.score > 0.0);
        assert!(result.details.contains_key("semantic_score"));
    }

    #[test]
    fn test_visual_question_accuracy_metric() {
        let metric = VisualQuestionAccuracyMetric::new(0.8);

        let image = create_test_image();
        let mut input = VisionInput::new(image, "What color is the cat?".to_string()).unwrap();
        input = input.with_metadata(
            "expected_answer".to_string(),
            serde_json::Value::String("black".to_string()),
        );

        let output = VisionOutput::new("The cat is black".to_string(), 0.9);

        let example = Example::new(input, output.clone());
        let result = metric.evaluate(&example, &output).unwrap();
        assert!(result.score > 0.0);
        assert!(result.details.contains_key("accuracy"));
    }

    #[test]
    fn test_cross_modal_consistency_metric() {
        let metric = CrossModalConsistencyMetric::new(0.75);

        let image = create_test_image();
        let input = MultiModalInput::new("Show me a cat".to_string()).add_media(image);

        let output = MultiModalOutput::new("Here is a cat in the image".to_string(), 0.9);

        let example = Example::new(input, output.clone());
        let result = metric.evaluate(&example, &output).unwrap();
        assert!(result.score > 0.0);
        assert!(result.details.contains_key("consistency_score"));
    }

    #[test]
    fn test_metric_names() {
        let alignment_metric = ImageTextAlignmentMetric::default();
        assert_eq!(alignment_metric.name(), "ImageTextAlignment");

        let vqa_metric = VisualQuestionAccuracyMetric::default();
        assert_eq!(vqa_metric.name(), "VisualQuestionAccuracy");

        let consistency_metric = CrossModalConsistencyMetric::default();
        assert_eq!(consistency_metric.name(), "CrossModalConsistency");
    }
}
