//! Comprehensive tests for DSPy multi-modal functionality
//!
//! This test suite validates the multi-modal capabilities including
//! image processing, vision-language models, and media handling.

use rust_memvid_agent::anthropic::AnthropicClient;
use rust_memvid_agent::dspy::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

fn create_test_client() -> Arc<AnthropicClient> {
    Arc::new(
        AnthropicClient::new(rust_memvid_agent::config::AnthropicConfig {
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

fn create_test_image() -> MediaContent {
    MediaContent::from_bytes(b"fake_jpeg_data".to_vec(), "image/jpeg".to_string()).unwrap()
}

fn create_test_audio() -> MediaContent {
    MediaContent::from_bytes(b"fake_audio_data".to_vec(), "audio/mpeg".to_string()).unwrap()
}

#[tokio::test]
async fn test_media_content_creation() {
    // Test image creation
    let image_data = b"test image data".to_vec();
    let image = MediaContent::from_bytes(image_data.clone(), "image/jpeg".to_string()).unwrap();

    assert_eq!(image.media_type, MediaType::Image);
    assert_eq!(image.mime_type, "image/jpeg");
    assert_eq!(image.data, image_data);
    assert!(!image.to_base64().is_empty());
    assert!(image.to_data_url().starts_with("data:image/jpeg;base64,"));

    // Test audio creation
    let audio_data = b"test audio data".to_vec();
    let audio = MediaContent::from_bytes(audio_data.clone(), "audio/mpeg".to_string()).unwrap();

    assert_eq!(audio.media_type, MediaType::Audio);
    assert_eq!(audio.mime_type, "audio/mpeg");
    assert_eq!(audio.data, audio_data);

    // Test unsupported format
    let result = MediaContent::from_bytes(b"test".to_vec(), "unsupported/format".to_string());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multimodal_input_creation() {
    let image = create_test_image();
    let audio = create_test_audio();

    let input = MultiModalInput::new("Analyze this content".to_string())
        .add_media(image.clone())
        .add_media(audio.clone())
        .with_context("Test context".to_string())
        .with_metadata(
            "test_key".to_string(),
            serde_json::Value::String("test_value".to_string()),
        );

    assert_eq!(input.text, "Analyze this content");
    assert_eq!(input.media.len(), 2);
    assert_eq!(input.context, Some("Test context".to_string()));
    assert!(input.has_media_type(&MediaType::Image));
    assert!(input.has_media_type(&MediaType::Audio));
    assert!(!input.has_media_type(&MediaType::Video));

    let images = input.get_media_by_type(&MediaType::Image);
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].mime_type, "image/jpeg");

    let audio_items = input.get_media_by_type(&MediaType::Audio);
    assert_eq!(audio_items.len(), 1);
    assert_eq!(audio_items[0].mime_type, "audio/mpeg");
}

#[tokio::test]
async fn test_multimodal_output_creation() {
    let output = MultiModalOutput::new("Analysis result".to_string(), 0.9)
        .add_media_analysis(
            "image_0".to_string(),
            serde_json::json!({
                "objects": ["person", "car"],
                "scene": "street"
            }),
        )
        .add_metadata(
            "processing_time".to_string(),
            serde_json::Value::Number(150.into()),
        );

    assert_eq!(output.text, "Analysis result");
    assert_eq!(output.confidence, 0.9);
    assert_eq!(output.media_analysis.len(), 1);
    assert_eq!(output.metadata.len(), 1);

    let image_analysis = output.media_analysis.get("image_0").unwrap();
    assert!(image_analysis.get("objects").is_some());
    assert!(image_analysis.get("scene").is_some());
}

#[tokio::test]
async fn test_multimodal_predict_creation() {
    let client = create_test_client();
    let signature =
        Signature::<MultiModalInput, MultiModalOutput>::new("test_multimodal".to_string())
            .with_description("Test multi-modal processing");

    let module = MultiModalPredict::new(signature, client);

    assert!(module.name().starts_with("MultiModalPredict_"));
    assert!(module.supports_compilation());
    assert_eq!(module.config().max_media_size_mb, 10);
    assert!(module.config().enable_media_preprocessing);
    assert!(module.config().enable_media_analysis);
}

#[tokio::test]
async fn test_multimodal_predict_with_config() {
    let client = create_test_client();
    let signature =
        Signature::<MultiModalInput, MultiModalOutput>::new("test_multimodal".to_string());

    let config = MultiModalConfig {
        max_media_size_mb: 5,
        enable_media_preprocessing: false,
        enable_media_analysis: false,
        image_max_dimension: Some(1024),
        audio_max_duration_seconds: Some(60.0),
        ..Default::default()
    };

    let module = MultiModalPredict::with_config(signature, client, config);

    assert_eq!(module.config().max_media_size_mb, 5);
    assert!(!module.config().enable_media_preprocessing);
    assert!(!module.config().enable_media_analysis);
    assert_eq!(module.config().image_max_dimension, Some(1024));
    assert_eq!(module.config().audio_max_duration_seconds, Some(60.0));
}

#[tokio::test]
async fn test_multimodal_input_validation() {
    let client = create_test_client();
    let signature =
        Signature::<MultiModalInput, MultiModalOutput>::new("test_multimodal".to_string());
    let module = MultiModalPredict::new(signature, client);

    // Test valid input
    let image = create_test_image();
    let valid_input = MultiModalInput::new("Test query".to_string()).add_media(image);

    let result = module.validate_input(&valid_input).await;
    assert!(result.is_ok());

    // Test empty input
    let empty_input = MultiModalInput::new("".to_string());
    let result = module.validate_input(&empty_input).await;
    assert!(result.is_err());

    // Test oversized media
    let large_data = vec![0u8; 20 * 1024 * 1024]; // 20MB
    let large_image = MediaContent::from_bytes(large_data, "image/jpeg".to_string()).unwrap();
    let oversized_input = MultiModalInput::new("Test".to_string()).add_media(large_image);

    let result = module.validate_input(&oversized_input).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_vision_input_creation() {
    let image = create_test_image();

    let input = VisionInput::new(image, "What do you see?".to_string())
        .unwrap()
        .with_analysis_type(VisionAnalysisType::ObjectDetection)
        .with_context("Analyze for safety".to_string())
        .with_metadata(
            "priority".to_string(),
            serde_json::Value::String("high".to_string()),
        );

    assert_eq!(input.query, "What do you see?");
    assert_eq!(input.analysis_type, VisionAnalysisType::ObjectDetection);
    assert_eq!(input.context, Some("Analyze for safety".to_string()));
    assert_eq!(input.metadata.len(), 1);

    // Test invalid media type
    let audio = create_test_audio();
    let result = VisionInput::new(audio, "Test".to_string());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_vision_analysis_types() {
    let types = vec![
        VisionAnalysisType::General,
        VisionAnalysisType::ObjectDetection,
        VisionAnalysisType::SceneDescription,
        VisionAnalysisType::TextExtraction,
        VisionAnalysisType::FaceAnalysis,
        VisionAnalysisType::MedicalImaging,
        VisionAnalysisType::DocumentAnalysis,
        VisionAnalysisType::ArtAnalysis,
        VisionAnalysisType::TechnicalDiagram,
    ];

    for analysis_type in types {
        let template = analysis_type.get_prompt_template();
        assert!(!template.is_empty());
        assert!(template.contains("{query}"));
    }
}

#[tokio::test]
async fn test_vision_output_creation() {
    let bbox = BoundingBox {
        x: 10.0,
        y: 20.0,
        width: 100.0,
        height: 150.0,
    };

    let detected_object = DetectedObject {
        label: "person".to_string(),
        confidence: 0.95,
        bounding_box: Some(bbox),
        attributes: HashMap::new(),
    };

    let output = VisionOutput::new("I see a person in the image".to_string(), 0.9)
        .add_detected_object(detected_object)
        .with_extracted_text("Some text in the image".to_string())
        .add_scene_attribute(
            "lighting".to_string(),
            serde_json::Value::String("bright".to_string()),
        )
        .add_metadata(
            "model".to_string(),
            serde_json::Value::String("claude-3-vision".to_string()),
        );

    assert_eq!(output.analysis, "I see a person in the image");
    assert_eq!(output.confidence, 0.9);
    assert_eq!(output.detected_objects.len(), 1);
    assert_eq!(
        output.extracted_text,
        Some("Some text in the image".to_string())
    );
    assert_eq!(output.scene_attributes.len(), 1);
    assert_eq!(output.metadata.len(), 1);

    let person = &output.detected_objects[0];
    assert_eq!(person.label, "person");
    assert_eq!(person.confidence, 0.95);
    assert!(person.bounding_box.is_some());
}

#[tokio::test]
async fn test_vision_language_model_creation() {
    let client = create_test_client();
    let signature = Signature::<VisionInput, VisionOutput>::new("test_vision".to_string())
        .with_description("Test vision analysis");

    let model = VisionLanguageModel::new(signature, client);

    assert!(model.name().starts_with("VisionLanguageModel_"));
    assert!(model.supports_compilation());
    assert_eq!(model.config().model, "claude-3-vision");
    assert_eq!(model.config().max_image_size_mb, 5);
    assert!(model.config().enable_object_detection);
    assert!(model.config().enable_text_extraction);
    assert!(model.config().enable_scene_analysis);
}

#[tokio::test]
async fn test_vision_language_model_with_config() {
    let client = create_test_client();
    let signature = Signature::<VisionInput, VisionOutput>::new("test_vision".to_string());

    let config = VisionConfig {
        model: "custom-vision-model".to_string(),
        max_image_size_mb: 2,
        enable_object_detection: false,
        enable_text_extraction: true,
        enable_scene_analysis: false,
        confidence_threshold: 0.8,
        max_objects: 20,
        preprocessing_enabled: false,
        ..Default::default()
    };

    let model = VisionLanguageModel::with_config(signature, client, config);

    assert_eq!(model.config().model, "custom-vision-model");
    assert_eq!(model.config().max_image_size_mb, 2);
    assert!(!model.config().enable_object_detection);
    assert!(model.config().enable_text_extraction);
    assert!(!model.config().enable_scene_analysis);
    assert_eq!(model.config().confidence_threshold, 0.8);
    assert_eq!(model.config().max_objects, 20);
    assert!(!model.config().preprocessing_enabled);
}

#[tokio::test]
async fn test_vision_input_validation() {
    let client = create_test_client();
    let signature = Signature::<VisionInput, VisionOutput>::new("test_vision".to_string());
    let model = VisionLanguageModel::new(signature, client);

    // Test valid input
    let image = create_test_image();
    let valid_input = VisionInput::new(image, "Describe this image".to_string()).unwrap();

    let result = model.validate_input(&valid_input).await;
    assert!(result.is_ok());

    // Test empty query
    let image = create_test_image();
    let mut empty_query_input = VisionInput::new(image, "".to_string()).unwrap();

    let result = model.validate_input(&empty_query_input).await;
    assert!(result.is_err());

    // Test unsupported format
    let unsupported_image =
        MediaContent::from_bytes(b"test".to_vec(), "image/bmp".to_string()).unwrap();
    let mut unsupported_input = VisionInput::new(unsupported_image, "Test".to_string()).unwrap();

    let result = model.validate_input(&unsupported_input).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multimodal_integration() {
    // Test that multi-modal and vision modules work together
    let client = create_test_client();

    // Create multi-modal module
    let mm_signature =
        Signature::<MultiModalInput, MultiModalOutput>::new("multimodal_test".to_string());
    let mm_module = MultiModalPredict::new(mm_signature, client.clone());

    // Create vision module
    let vision_signature = Signature::<VisionInput, VisionOutput>::new("vision_test".to_string());
    let vision_module = VisionLanguageModel::new(vision_signature, client);

    // Test that both modules can handle the same image
    let image = create_test_image();

    // Multi-modal input
    let mm_input = MultiModalInput::new("Analyze this image".to_string()).add_media(image.clone());

    assert!(mm_module.validate_input(&mm_input).await.is_ok());

    // Vision input
    let vision_input = VisionInput::new(image, "Analyze this image".to_string()).unwrap();

    assert!(vision_module.validate_input(&vision_input).await.is_ok());
}

#[tokio::test]
async fn test_media_type_detection() {
    // Test various media types
    let test_cases = vec![
        ("image/jpeg", MediaType::Image),
        ("image/png", MediaType::Image),
        ("audio/mpeg", MediaType::Audio),
        ("audio/wav", MediaType::Audio),
        ("video/mp4", MediaType::Video),
        ("application/pdf", MediaType::Document),
        ("text/plain", MediaType::Document),
    ];

    for (mime_type, expected_type) in test_cases {
        let media = MediaContent::from_bytes(b"test data".to_vec(), mime_type.to_string()).unwrap();

        assert_eq!(media.media_type, expected_type);
        assert_eq!(media.mime_type, mime_type);
    }
}

#[tokio::test]
async fn test_multimodal_error_handling() {
    let client = create_test_client();
    let signature = Signature::<MultiModalInput, MultiModalOutput>::new("error_test".to_string());
    let module = MultiModalPredict::new(signature, client);

    // Test various error conditions
    let test_cases = vec![
        // Empty text and no media
        (
            MultiModalInput::new("".to_string()),
            "Input must contain either text or media",
        ),
        // Oversized media
        (
            MultiModalInput::new("Test".to_string()).add_media(
                MediaContent::from_bytes(vec![0u8; 20 * 1024 * 1024], "image/jpeg".to_string())
                    .unwrap(),
            ),
            "exceeds limit",
        ),
        // Unsupported format
        (
            MultiModalInput::new("Test".to_string()).add_media(
                MediaContent::from_bytes(b"test".to_vec(), "image/bmp".to_string()).unwrap(),
            ),
            "Unsupported media format",
        ),
    ];

    for (input, expected_error) in test_cases {
        let result = module.validate_input(&input).await;
        assert!(result.is_err());
        let error_msg = format!("{}", result.unwrap_err());
        assert!(
            error_msg.contains(expected_error),
            "Expected '{}' in error: {}",
            expected_error,
            error_msg
        );
    }
}

#[tokio::test]
async fn test_vision_error_handling() {
    let client = create_test_client();
    let signature = Signature::<VisionInput, VisionOutput>::new("vision_error_test".to_string());
    let model = VisionLanguageModel::new(signature, client);

    // Test empty query
    let image = create_test_image();
    let empty_query = VisionInput::new(image, "".to_string()).unwrap();

    let result = model.validate_input(&empty_query).await;
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Query cannot be empty"));

    // Test invalid output
    let invalid_output = VisionOutput::new("".to_string(), 0.5);
    let result = model.validate_output(&invalid_output).await;
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Analysis cannot be empty"));

    let invalid_confidence = VisionOutput::new("Valid analysis".to_string(), 1.5);
    let result = model.validate_output(&invalid_confidence).await;
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("Confidence must be between 0.0 and 1.0"));
}
