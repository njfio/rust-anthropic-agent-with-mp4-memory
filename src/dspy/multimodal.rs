//! Multi-modal support for DSPy framework
//!
//! This module provides support for multi-modal inputs including images, audio,
//! and combined text-media processing capabilities.

use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use async_trait::async_trait;
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, error, info, warn};

/// Supported media types for multi-modal processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MediaType {
    Image,
    Audio,
    Video,
    Document,
}

/// Media content representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaContent {
    pub media_type: MediaType,
    pub data: Vec<u8>,
    pub mime_type: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MediaContent {
    /// Create new media content from file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> DspyResult<Self> {
        let path = path.as_ref();
        let data = fs::read(path)
            .await
            .map_err(|e| DspyError::io(format!("Failed to read file {}: {}", path.display(), e)))?;

        let mime_type = Self::detect_mime_type(path)?;
        let media_type = Self::media_type_from_mime(&mime_type)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "file_path".to_string(),
            serde_json::Value::String(path.display().to_string()),
        );
        metadata.insert(
            "file_size".to_string(),
            serde_json::Value::Number(data.len().into()),
        );

        Ok(Self {
            media_type,
            data,
            mime_type,
            metadata,
        })
    }

    /// Create media content from bytes
    pub fn from_bytes(data: Vec<u8>, mime_type: String) -> DspyResult<Self> {
        let media_type = Self::media_type_from_mime(&mime_type)?;
        let mut metadata = HashMap::new();
        metadata.insert(
            "data_size".to_string(),
            serde_json::Value::Number(data.len().into()),
        );

        Ok(Self {
            media_type,
            data,
            mime_type,
            metadata,
        })
    }

    /// Convert to base64 string
    pub fn to_base64(&self) -> String {
        general_purpose::STANDARD.encode(&self.data)
    }

    /// Get data URL representation
    pub fn to_data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.to_base64())
    }

    /// Detect MIME type from file extension
    fn detect_mime_type(path: &Path) -> DspyResult<String> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                DspyError::configuration("file_extension", "Could not determine file extension")
            })?;

        let mime_type = match extension.to_lowercase().as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "mp3" => "audio/mpeg",
            "wav" => "audio/wav",
            "ogg" => "audio/ogg",
            "mp4" => "video/mp4",
            "webm" => "video/webm",
            "pdf" => "application/pdf",
            "txt" => "text/plain",
            _ => {
                return Err(DspyError::configuration(
                    "mime_type",
                    &format!("Unsupported file extension: {}", extension),
                ))
            }
        };

        Ok(mime_type.to_string())
    }

    /// Convert MIME type to media type
    fn media_type_from_mime(mime_type: &str) -> DspyResult<MediaType> {
        let media_type = if mime_type.starts_with("image/") {
            MediaType::Image
        } else if mime_type.starts_with("audio/") {
            MediaType::Audio
        } else if mime_type.starts_with("video/") {
            MediaType::Video
        } else if mime_type.starts_with("application/") || mime_type.starts_with("text/") {
            MediaType::Document
        } else {
            return Err(DspyError::configuration(
                "media_type",
                &format!("Unsupported MIME type: {}", mime_type),
            ));
        };

        Ok(media_type)
    }
}

/// Multi-modal input combining text and media
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalInput {
    pub text: String,
    pub media: Vec<MediaContent>,
    pub context: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MultiModalInput {
    /// Create new multi-modal input
    pub fn new(text: String) -> Self {
        Self {
            text,
            media: Vec::new(),
            context: None,
            metadata: HashMap::new(),
        }
    }

    /// Add media content
    pub fn add_media(mut self, media: MediaContent) -> Self {
        self.media.push(media);
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

    /// Check if input contains specific media type
    pub fn has_media_type(&self, media_type: &MediaType) -> bool {
        self.media.iter().any(|m| &m.media_type == media_type)
    }

    /// Get media by type
    pub fn get_media_by_type(&self, media_type: &MediaType) -> Vec<&MediaContent> {
        self.media
            .iter()
            .filter(|m| &m.media_type == media_type)
            .collect()
    }
}

/// Multi-modal output with enhanced response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalOutput {
    pub text: String,
    pub confidence: f64,
    pub media_analysis: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MultiModalOutput {
    /// Create new multi-modal output
    pub fn new(text: String, confidence: f64) -> Self {
        Self {
            text,
            confidence,
            media_analysis: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add media analysis result
    pub fn add_media_analysis(mut self, key: String, analysis: serde_json::Value) -> Self {
        self.media_analysis.insert(key, analysis);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Configuration for multi-modal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    pub max_media_size_mb: usize,
    pub supported_image_formats: Vec<String>,
    pub supported_audio_formats: Vec<String>,
    pub enable_media_preprocessing: bool,
    pub image_max_dimension: Option<usize>,
    pub audio_max_duration_seconds: Option<f64>,
    pub enable_media_analysis: bool,
    pub vision_model: Option<String>,
    pub audio_model: Option<String>,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            max_media_size_mb: 10,
            supported_image_formats: vec![
                "image/jpeg".to_string(),
                "image/png".to_string(),
                "image/gif".to_string(),
                "image/webp".to_string(),
            ],
            supported_audio_formats: vec![
                "audio/mpeg".to_string(),
                "audio/wav".to_string(),
                "audio/ogg".to_string(),
            ],
            enable_media_preprocessing: true,
            image_max_dimension: Some(2048),
            audio_max_duration_seconds: Some(300.0), // 5 minutes
            enable_media_analysis: true,
            vision_model: Some("claude-3-vision".to_string()),
            audio_model: Some("whisper-1".to_string()),
        }
    }
}

/// Multi-modal prediction module
pub struct MultiModalPredict {
    id: String,
    name: String,
    signature: Signature<MultiModalInput, MultiModalOutput>,
    anthropic_client: Arc<AnthropicClient>,
    config: MultiModalConfig,
    metadata: ModuleMetadata,
    stats: ModuleStats,
}

impl MultiModalPredict {
    /// Create new multi-modal predict module
    pub fn new(
        signature: Signature<MultiModalInput, MultiModalOutput>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self {
        let id = format!("MultiModalPredict_{}", uuid::Uuid::new_v4());
        let name = format!("MultiModalPredict_{}", signature.name);

        Self {
            id,
            name,
            signature,
            anthropic_client,
            config: MultiModalConfig::default(),
            metadata: ModuleMetadata::default(),
            stats: ModuleStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<MultiModalInput, MultiModalOutput>,
        anthropic_client: Arc<AnthropicClient>,
        config: MultiModalConfig,
    ) -> Self {
        let mut module = Self::new(signature, anthropic_client);
        module.config = config;
        module
    }

    /// Get configuration
    pub fn config(&self) -> &MultiModalConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: MultiModalConfig) {
        self.config = config;
    }

    /// Validate media content
    async fn validate_media(&self, media: &MediaContent) -> DspyResult<()> {
        // Check file size
        let size_mb = media.data.len() as f64 / (1024.0 * 1024.0);
        if size_mb > self.config.max_media_size_mb as f64 {
            return Err(DspyError::module(
                &self.name,
                &format!(
                    "Media size {:.1}MB exceeds limit of {}MB",
                    size_mb, self.config.max_media_size_mb
                ),
            ));
        }

        // Check supported formats
        let supported_formats = match media.media_type {
            MediaType::Image => &self.config.supported_image_formats,
            MediaType::Audio => &self.config.supported_audio_formats,
            _ => return Ok(()), // Other types not restricted
        };

        if !supported_formats.contains(&media.mime_type) {
            return Err(DspyError::module(
                &self.name,
                &format!("Unsupported media format: {}", media.mime_type),
            ));
        }

        Ok(())
    }

    /// Preprocess media content
    async fn preprocess_media(&self, media: &mut MediaContent) -> DspyResult<()> {
        if !self.config.enable_media_preprocessing {
            return Ok(());
        }

        match media.media_type {
            MediaType::Image => {
                if let Some(max_dim) = self.config.image_max_dimension {
                    // In a real implementation, you would resize the image here
                    debug!("Image preprocessing: max dimension {}", max_dim);
                    media
                        .metadata
                        .insert("preprocessed".to_string(), serde_json::Value::Bool(true));
                }
            }
            MediaType::Audio => {
                if let Some(max_duration) = self.config.audio_max_duration_seconds {
                    // In a real implementation, you would trim the audio here
                    debug!("Audio preprocessing: max duration {}s", max_duration);
                    media
                        .metadata
                        .insert("preprocessed".to_string(), serde_json::Value::Bool(true));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Analyze media content
    async fn analyze_media(&self, media: &MediaContent) -> DspyResult<serde_json::Value> {
        if !self.config.enable_media_analysis {
            return Ok(serde_json::Value::Null);
        }

        let analysis = match media.media_type {
            MediaType::Image => {
                // In a real implementation, you would call vision API here
                serde_json::json!({
                    "type": "image_analysis",
                    "detected_objects": [],
                    "scene_description": "Image analysis placeholder",
                    "confidence": 0.8
                })
            }
            MediaType::Audio => {
                // In a real implementation, you would call audio analysis API here
                serde_json::json!({
                    "type": "audio_analysis",
                    "transcription": "Audio transcription placeholder",
                    "language": "en",
                    "confidence": 0.9
                })
            }
            _ => serde_json::Value::Null,
        };

        Ok(analysis)
    }

    /// Process multi-modal input
    async fn process_multimodal(&self, input: MultiModalInput) -> DspyResult<MultiModalOutput> {
        let mut media_analyses = HashMap::new();

        // Process each media item
        for (i, media) in input.media.iter().enumerate() {
            let analysis = self.analyze_media(media).await?;
            if !analysis.is_null() {
                media_analyses.insert(format!("media_{}", i), analysis);
            }
        }

        // Create prompt with media context
        let mut prompt_parts = vec![input.text.clone()];

        if let Some(context) = &input.context {
            prompt_parts.push(format!("Context: {}", context));
        }

        // Add media descriptions
        for (key, analysis) in &media_analyses {
            if let Some(description) = analysis.get("scene_description").and_then(|v| v.as_str()) {
                prompt_parts.push(format!("Image analysis for {}: {}", key, description));
            }
            if let Some(transcription) = analysis.get("transcription").and_then(|v| v.as_str()) {
                prompt_parts.push(format!(
                    "Audio transcription for {}: {}",
                    key, transcription
                ));
            }
        }

        let combined_prompt = prompt_parts.join("\n\n");

        // In a real implementation, you would call the Anthropic API here
        // For now, we'll create a mock response
        let response_text = format!("Multi-modal response to: {}", combined_prompt);
        let confidence = 0.85;

        let mut output = MultiModalOutput::new(response_text, confidence);

        // Add media analysis to output
        for (key, analysis) in media_analyses {
            output = output.add_media_analysis(key, analysis);
        }

        // Add processing metadata
        output = output.add_metadata(
            "media_count".to_string(),
            serde_json::Value::Number(input.media.len().into()),
        );
        output = output.add_metadata(
            "has_images".to_string(),
            serde_json::Value::Bool(input.has_media_type(&MediaType::Image)),
        );
        output = output.add_metadata(
            "has_audio".to_string(),
            serde_json::Value::Bool(input.has_media_type(&MediaType::Audio)),
        );

        Ok(output)
    }
}

#[async_trait]
impl Module for MultiModalPredict {
    type Input = MultiModalInput;
    type Output = MultiModalOutput;

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
        if input.text.trim().is_empty() && input.media.is_empty() {
            return Err(DspyError::module(
                self.name(),
                "Input must contain either text or media",
            ));
        }

        // Validate each media item
        for media in &input.media {
            self.validate_media(media).await?;
        }

        Ok(())
    }

    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        if output.text.trim().is_empty() {
            return Err(DspyError::module(
                self.name(),
                "Output text cannot be empty",
            ));
        }

        if output.confidence < 0.0 || output.confidence > 1.0 {
            return Err(DspyError::module(
                self.name(),
                "Confidence must be between 0.0 and 1.0",
            ));
        }

        Ok(())
    }

    async fn forward(&self, mut input: Self::Input) -> DspyResult<Self::Output> {
        info!(
            "Processing multi-modal input with {} media items",
            input.media.len()
        );

        // Validate input
        self.validate_input(&input).await?;

        // Preprocess media
        for media in &mut input.media {
            self.preprocess_media(media).await?;
        }

        // Process the multi-modal input
        let output = self.process_multimodal(input).await?;

        // Validate output
        self.validate_output(&output).await?;

        info!("Multi-modal processing completed successfully");
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
    use tokio;

    fn create_test_client() -> Arc<AnthropicClient> {
        Arc::new(
            AnthropicClient::new(crate::config::AnthropicConfig {
                api_key: "test_key".to_string(),
                model: "claude-3-sonnet-20240229".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                max_tokens: 1000,
                temperature: 0.7,
                timeout_seconds: 30,
                max_retries: 3,
            })
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn test_media_content_creation() {
        let data = b"test image data".to_vec();
        let media = MediaContent::from_bytes(data, "image/jpeg".to_string()).unwrap();

        assert_eq!(media.media_type, MediaType::Image);
        assert_eq!(media.mime_type, "image/jpeg");
        assert!(!media.to_base64().is_empty());
        assert!(media.to_data_url().starts_with("data:image/jpeg;base64,"));
    }

    #[tokio::test]
    async fn test_multimodal_input_creation() {
        let input = MultiModalInput::new("Test text".to_string())
            .with_context("Test context".to_string())
            .with_metadata(
                "key".to_string(),
                serde_json::Value::String("value".to_string()),
            );

        assert_eq!(input.text, "Test text");
        assert_eq!(input.context, Some("Test context".to_string()));
        assert_eq!(
            input.metadata.get("key").unwrap(),
            &serde_json::Value::String("value".to_string())
        );
    }

    #[tokio::test]
    async fn test_multimodal_predict_creation() {
        let client = create_test_client();
        let signature = Signature::new("test_multimodal".to_string());
        let module = MultiModalPredict::new(signature, client);

        assert!(module.name().starts_with("MultiModalPredict_"));
        assert!(module.supports_compilation());
    }

    #[tokio::test]
    async fn test_media_validation() {
        let client = create_test_client();
        let signature = Signature::new("test_multimodal".to_string());
        let module = MultiModalPredict::new(signature, client);

        // Test valid media
        let valid_media =
            MediaContent::from_bytes(b"small image".to_vec(), "image/jpeg".to_string()).unwrap();

        assert!(module.validate_media(&valid_media).await.is_ok());

        // Test oversized media
        let large_data = vec![0u8; 20 * 1024 * 1024]; // 20MB
        let large_media = MediaContent::from_bytes(large_data, "image/jpeg".to_string()).unwrap();

        assert!(module.validate_media(&large_media).await.is_err());
    }

    #[tokio::test]
    async fn test_multimodal_processing() {
        let client = create_test_client();
        let signature = Signature::new("test_multimodal".to_string());
        let module = MultiModalPredict::new(signature, client);

        let media = MediaContent::from_bytes(b"test image data".to_vec(), "image/jpeg".to_string())
            .unwrap();

        let input = MultiModalInput::new("Describe this image".to_string()).add_media(media);

        // Note: This would require actual API integration to fully test
        // For now, we test the structure
        assert!(input.has_media_type(&MediaType::Image));
        assert_eq!(input.get_media_by_type(&MediaType::Image).len(), 1);
    }
}
