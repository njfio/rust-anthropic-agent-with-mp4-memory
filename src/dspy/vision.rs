//! Vision-language model integration for DSPy framework
//!
//! This module provides specialized support for vision-language models,
//! image analysis, and visual reasoning capabilities.

use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    multimodal::{MediaContent, MediaType, MultiModalInput, MultiModalOutput},
    signature::Signature,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Vision-specific input for image analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionInput {
    pub image: MediaContent,
    pub query: String,
    pub context: Option<String>,
    pub analysis_type: VisionAnalysisType,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VisionInput {
    /// Create new vision input
    pub fn new(image: MediaContent, query: String) -> DspyResult<Self> {
        if image.media_type != MediaType::Image {
            return Err(DspyError::module("VisionInput", "Media must be an image"));
        }
        
        Ok(Self {
            image,
            query,
            context: None,
            analysis_type: VisionAnalysisType::General,
            metadata: HashMap::new(),
        })
    }
    
    /// Set analysis type
    pub fn with_analysis_type(mut self, analysis_type: VisionAnalysisType) -> Self {
        self.analysis_type = analysis_type;
        self
    }
    
    /// Add context
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Types of vision analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VisionAnalysisType {
    General,
    ObjectDetection,
    SceneDescription,
    TextExtraction,
    FaceAnalysis,
    MedicalImaging,
    DocumentAnalysis,
    ArtAnalysis,
    TechnicalDiagram,
}

impl VisionAnalysisType {
    /// Get analysis prompt template
    pub fn get_prompt_template(&self) -> &'static str {
        match self {
            VisionAnalysisType::General => "Analyze this image and answer the following question: {query}",
            VisionAnalysisType::ObjectDetection => "Identify and list all objects visible in this image. Question: {query}",
            VisionAnalysisType::SceneDescription => "Provide a detailed description of the scene in this image. Focus on: {query}",
            VisionAnalysisType::TextExtraction => "Extract and transcribe all text visible in this image. Additional instructions: {query}",
            VisionAnalysisType::FaceAnalysis => "Analyze faces in this image (if any). Consider: {query}",
            VisionAnalysisType::MedicalImaging => "Analyze this medical image. Clinical question: {query}",
            VisionAnalysisType::DocumentAnalysis => "Analyze this document image. Focus on: {query}",
            VisionAnalysisType::ArtAnalysis => "Analyze this artwork or artistic image. Consider: {query}",
            VisionAnalysisType::TechnicalDiagram => "Analyze this technical diagram or schematic. Question: {query}",
        }
    }
}

/// Vision analysis output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionOutput {
    pub analysis: String,
    pub confidence: f64,
    pub detected_objects: Vec<DetectedObject>,
    pub extracted_text: Option<String>,
    pub scene_attributes: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VisionOutput {
    /// Create new vision output
    pub fn new(analysis: String, confidence: f64) -> Self {
        Self {
            analysis,
            confidence,
            detected_objects: Vec::new(),
            extracted_text: None,
            scene_attributes: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add detected object
    pub fn add_detected_object(mut self, object: DetectedObject) -> Self {
        self.detected_objects.push(object);
        self
    }
    
    /// Set extracted text
    pub fn with_extracted_text(mut self, text: String) -> Self {
        self.extracted_text = Some(text);
        self
    }
    
    /// Add scene attribute
    pub fn add_scene_attribute(mut self, key: String, value: serde_json::Value) -> Self {
        self.scene_attributes.insert(key, value);
        self
    }
    
    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Detected object in image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f64,
    pub bounding_box: Option<BoundingBox>,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Bounding box coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Configuration for vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub model: String,
    pub max_image_size_mb: usize,
    pub supported_formats: Vec<String>,
    pub enable_object_detection: bool,
    pub enable_text_extraction: bool,
    pub enable_scene_analysis: bool,
    pub confidence_threshold: f64,
    pub max_objects: usize,
    pub preprocessing_enabled: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            model: "claude-3-vision".to_string(),
            max_image_size_mb: 5,
            supported_formats: vec![
                "image/jpeg".to_string(),
                "image/png".to_string(),
                "image/gif".to_string(),
                "image/webp".to_string(),
            ],
            enable_object_detection: true,
            enable_text_extraction: true,
            enable_scene_analysis: true,
            confidence_threshold: 0.5,
            max_objects: 50,
            preprocessing_enabled: true,
        }
    }
}

/// Vision-language model for image analysis
pub struct VisionLanguageModel {
    id: String,
    name: String,
    signature: Signature<VisionInput, VisionOutput>,
    anthropic_client: Arc<AnthropicClient>,
    config: VisionConfig,
    metadata: ModuleMetadata,
    stats: ModuleStats,
}

impl VisionLanguageModel {
    /// Create new vision-language model
    pub fn new(
        signature: Signature<VisionInput, VisionOutput>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self {
        let id = format!("VisionLanguageModel_{}", uuid::Uuid::new_v4());
        let name = format!("VisionLanguageModel_{}", signature.name);
        
        Self {
            id,
            name,
            signature,
            anthropic_client,
            config: VisionConfig::default(),
            metadata: ModuleMetadata::default(),
            stats: ModuleStats::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<VisionInput, VisionOutput>,
        anthropic_client: Arc<AnthropicClient>,
        config: VisionConfig,
    ) -> Self {
        let mut model = Self::new(signature, anthropic_client);
        model.config = config;
        model
    }
    
    /// Get configuration
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: VisionConfig) {
        self.config = config;
    }
    
    /// Validate image input
    async fn validate_image(&self, image: &MediaContent) -> DspyResult<()> {
        // Check media type
        if image.media_type != MediaType::Image {
            return Err(DspyError::module(self.name(), "Input must be an image"));
        }
        
        // Check format support
        if !self.config.supported_formats.contains(&image.mime_type) {
            return Err(DspyError::module(
                self.name(),
                &format!("Unsupported image format: {}", image.mime_type)
            ));
        }
        
        // Check file size
        let size_mb = image.data.len() as f64 / (1024.0 * 1024.0);
        if size_mb > self.config.max_image_size_mb as f64 {
            return Err(DspyError::module(
                self.name(),
                &format!("Image size {:.1}MB exceeds limit of {}MB", size_mb, self.config.max_image_size_mb)
            ));
        }
        
        Ok(())
    }
    
    /// Preprocess image if needed
    async fn preprocess_image(&self, image: &mut MediaContent) -> DspyResult<()> {
        if !self.config.preprocessing_enabled {
            return Ok(());
        }
        
        // In a real implementation, you would:
        // - Resize image if too large
        // - Normalize format
        // - Enhance quality if needed
        // - Apply filters for better analysis
        
        debug!("Preprocessing image: {} bytes", image.data.len());
        image.metadata.insert(
            "preprocessed".to_string(),
            serde_json::Value::Bool(true)
        );
        
        Ok(())
    }
    
    /// Analyze image with vision model
    async fn analyze_image(&self, input: &VisionInput) -> DspyResult<VisionOutput> {
        // Create prompt based on analysis type
        let prompt_template = input.analysis_type.get_prompt_template();
        let prompt = prompt_template.replace("{query}", &input.query);
        
        // Add context if provided
        let full_prompt = if let Some(context) = &input.context {
            format!("{}\n\nContext: {}", prompt, context)
        } else {
            prompt
        };
        
        // In a real implementation, you would call the vision API here
        // For now, we'll create a mock response based on analysis type
        let (analysis, confidence) = self.generate_mock_analysis(&input.analysis_type, &input.query);
        
        let mut output = VisionOutput::new(analysis, confidence);
        
        // Add mock detected objects for object detection
        if input.analysis_type == VisionAnalysisType::ObjectDetection && self.config.enable_object_detection {
            output = output.add_detected_object(DetectedObject {
                label: "person".to_string(),
                confidence: 0.9,
                bounding_box: Some(BoundingBox { x: 100.0, y: 50.0, width: 200.0, height: 300.0 }),
                attributes: HashMap::new(),
            });
        }
        
        // Add mock extracted text for text extraction
        if input.analysis_type == VisionAnalysisType::TextExtraction && self.config.enable_text_extraction {
            output = output.with_extracted_text("Sample extracted text from image".to_string());
        }
        
        // Add scene attributes for scene analysis
        if self.config.enable_scene_analysis {
            output = output.add_scene_attribute("lighting".to_string(), serde_json::Value::String("natural".to_string()));
            output = output.add_scene_attribute("setting".to_string(), serde_json::Value::String("indoor".to_string()));
        }
        
        // Add processing metadata
        output = output.add_metadata("model".to_string(), serde_json::Value::String(self.config.model.clone()));
        output = output.add_metadata("analysis_type".to_string(), serde_json::to_value(&input.analysis_type)?);
        output = output.add_metadata("image_size".to_string(), serde_json::Value::Number(input.image.data.len().into()));
        
        Ok(output)
    }
    
    /// Generate mock analysis for testing
    fn generate_mock_analysis(&self, analysis_type: &VisionAnalysisType, query: &str) -> (String, f64) {
        match analysis_type {
            VisionAnalysisType::General => (
                format!("General analysis of the image in response to: {}", query),
                0.85
            ),
            VisionAnalysisType::ObjectDetection => (
                "Detected objects: person, chair, table, window".to_string(),
                0.9
            ),
            VisionAnalysisType::SceneDescription => (
                "The scene shows an indoor setting with natural lighting, featuring furniture and a person".to_string(),
                0.88
            ),
            VisionAnalysisType::TextExtraction => (
                "Extracted text from the image has been processed".to_string(),
                0.92
            ),
            VisionAnalysisType::FaceAnalysis => (
                "Face analysis completed with privacy considerations".to_string(),
                0.8
            ),
            VisionAnalysisType::MedicalImaging => (
                "Medical image analysis requires specialized expertise".to_string(),
                0.75
            ),
            VisionAnalysisType::DocumentAnalysis => (
                "Document structure and content have been analyzed".to_string(),
                0.9
            ),
            VisionAnalysisType::ArtAnalysis => (
                "Artistic elements, style, and composition have been analyzed".to_string(),
                0.85
            ),
            VisionAnalysisType::TechnicalDiagram => (
                "Technical diagram components and relationships have been identified".to_string(),
                0.87
            ),
        }
    }
}

#[async_trait]
impl Module for VisionLanguageModel {
    type Input = VisionInput;
    type Output = VisionOutput;
    
    fn id(&self) -> &str {
        &self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.signature
    }
    
    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        if input.query.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Query cannot be empty"));
        }
        
        self.validate_image(&input.image).await?;
        
        Ok(())
    }
    
    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        if output.analysis.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Analysis cannot be empty"));
        }
        
        if output.confidence < 0.0 || output.confidence > 1.0 {
            return Err(DspyError::module(self.name(), "Confidence must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
    
    async fn forward(&self, mut input: Self::Input) -> DspyResult<Self::Output> {
        info!("Processing vision input with analysis type: {:?}", input.analysis_type);
        
        // Validate input
        self.validate_input(&input).await?;
        
        // Preprocess image
        self.preprocess_image(&mut input.image).await?;
        
        // Analyze image
        let output = self.analyze_image(&input).await?;
        
        // Validate output
        self.validate_output(&output).await?;
        
        info!("Vision analysis completed successfully");
        Ok(output)
    }
    
    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }
    
    fn stats(&self) -> &ModuleStats {
        &self.stats
    }
    
    fn supports_compilation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspy::multimodal::MediaContent;
    
    fn create_test_client() -> Arc<AnthropicClient> {
        Arc::new(AnthropicClient::new(crate::config::AnthropicConfig {
            api_key: "test_key".to_string(),
            model: "claude-3-sonnet-20240229".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
            max_retries: 3,
        }).unwrap())
    }
    
    fn create_test_image() -> MediaContent {
        MediaContent::from_bytes(
            b"test image data".to_vec(),
            "image/jpeg".to_string()
        ).unwrap()
    }
    
    #[tokio::test]
    async fn test_vision_input_creation() {
        let image = create_test_image();
        let input = VisionInput::new(image, "What do you see?".to_string()).unwrap();
        
        assert_eq!(input.query, "What do you see?");
        assert_eq!(input.analysis_type, VisionAnalysisType::General);
    }
    
    #[tokio::test]
    async fn test_vision_analysis_types() {
        let template = VisionAnalysisType::ObjectDetection.get_prompt_template();
        assert!(template.contains("objects"));
        
        let template = VisionAnalysisType::TextExtraction.get_prompt_template();
        assert!(template.contains("text"));
    }
    
    #[tokio::test]
    async fn test_vision_model_creation() {
        let client = create_test_client();
        let signature = Signature::new("test_vision".to_string());
        let model = VisionLanguageModel::new(signature, client);
        
        assert!(model.name().starts_with("VisionLanguageModel_"));
        assert!(model.supports_compilation());
    }
    
    #[tokio::test]
    async fn test_image_validation() {
        let client = create_test_client();
        let signature = Signature::new("test_vision".to_string());
        let model = VisionLanguageModel::new(signature, client);
        
        let valid_image = create_test_image();
        assert!(model.validate_image(&valid_image).await.is_ok());
        
        // Test unsupported format
        let invalid_image = MediaContent::from_bytes(
            b"test".to_vec(),
            "image/bmp".to_string()
        ).unwrap();
        assert!(model.validate_image(&invalid_image).await.is_err());
    }
    
    #[tokio::test]
    async fn test_vision_output_creation() {
        let output = VisionOutput::new("Test analysis".to_string(), 0.9)
            .add_detected_object(DetectedObject {
                label: "test".to_string(),
                confidence: 0.8,
                bounding_box: None,
                attributes: HashMap::new(),
            })
            .with_extracted_text("Test text".to_string())
            .add_scene_attribute("lighting".to_string(), serde_json::Value::String("bright".to_string()));
        
        assert_eq!(output.analysis, "Test analysis");
        assert_eq!(output.confidence, 0.9);
        assert_eq!(output.detected_objects.len(), 1);
        assert_eq!(output.extracted_text, Some("Test text".to_string()));
        assert_eq!(output.scene_attributes.len(), 1);
    }
}
