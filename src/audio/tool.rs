// Audio Processing Tool Implementation
// Provides Tool trait integration for the audio processing system

use super::{
    AudioConfig, AudioProcessor, AudioQuality, EffectsConfig,
    codecs::{AudioCodec, AudioData},
    effects::{AudioEffects, create_voice_effects_processor, create_music_effects_processor},
    metadata::{MetadataExtractor, AudioMetadata},
    synthesis::SynthesisService,
    transcription::TranscriptionService,
};
use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_optional_string_param, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};
use crate::utils::validation;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Comprehensive audio analysis result
#[derive(Debug, Clone)]
pub struct AudioAnalysisResult {
    /// Extracted metadata
    pub metadata: super::metadata::AudioMetadata,
    /// Audio effects analysis
    pub effects_analysis: super::effects::AudioAnalysis,
    /// Estimated synthesis duration if text content available
    pub estimated_synthesis_duration: Option<f64>,
    /// Source file path
    pub file_path: String,
    /// Analysis timestamp
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Processed audio result with comprehensive information
#[derive(Debug, Clone)]
pub struct ProcessedAudioResult {
    /// Initial metadata before processing
    pub initial_metadata: super::metadata::AudioMetadata,
    /// Final metadata after processing
    pub final_metadata: super::metadata::AudioMetadata,
    /// Processed audio data
    pub processed_audio: super::codecs::AudioData,
    /// Effects analysis of processed audio
    pub effects_analysis: super::effects::AudioAnalysis,
    /// Whether effects were applied
    pub effects_applied: bool,
    /// Processing timestamp
    pub processing_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Audio processing tool for comprehensive audio operations
#[derive(Debug)]
pub struct AudioProcessingTool {
    /// Audio processor instance
    processor: Arc<Mutex<AudioProcessor>>,
    /// Audio effects processor
    effects: Arc<Mutex<AudioEffects>>,
    /// Metadata extractor
    metadata_extractor: MetadataExtractor,
    /// Transcription service (optional)
    transcription_service: Option<Arc<Mutex<TranscriptionService>>>,
    /// Synthesis service (optional)
    synthesis_service: Option<Arc<Mutex<SynthesisService>>>,
    // Streaming manager disabled for now due to thread safety issues
}

impl AudioProcessingTool {
    /// Create a new audio processing tool
    pub fn new(config: AudioConfig) -> Result<Self> {
        let processor = AudioProcessor::new(config.clone());
        let effects = AudioEffects::new(config.effects.clone());
        let metadata_extractor = MetadataExtractor;

        // Initialize transcription service if configured
        let transcription_service = if !config.transcription.api_key.is_none() {
            match TranscriptionService::new(config.transcription.clone()) {
                Ok(service) => Some(Arc::new(Mutex::new(service))),
                Err(e) => {
                    warn!("Failed to initialize transcription service: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize synthesis service if configured
        let synthesis_service = if !config.synthesis.api_key.is_none() {
            match SynthesisService::new(config.synthesis.clone()) {
                Ok(service) => Some(Arc::new(Mutex::new(service))),
                Err(e) => {
                    warn!("Failed to initialize synthesis service: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize streaming manager (disabled for now due to thread safety issues)
        let _streaming_manager: Option<()> = None;

        Ok(Self {
            processor: Arc::new(Mutex::new(processor)),
            effects: Arc::new(Mutex::new(effects)),
            metadata_extractor,
            transcription_service,
            synthesis_service,
        })
    }

    /// Set cache manager for the audio processor
    pub async fn set_cache_manager(&self, cache_manager: Arc<crate::caching::CacheManager>) {
        // Note: This would need to be implemented properly in AudioProcessor
        // For now, we'll skip this functionality
        warn!("Cache manager setting not implemented for audio processor");
    }

    /// Set resource monitor for the audio processor
    pub async fn set_resource_monitor(&self, resource_monitor: Arc<crate::utils::resource_monitor::ResourceMonitor>) {
        // Note: This would need to be implemented properly in AudioProcessor
        // For now, we'll skip this functionality
        warn!("Resource monitor setting not implemented for audio processor");
    }

    /// Process audio file
    async fn process_audio_file(&self, input: &Value) -> Result<ToolResult> {
        let file_path = extract_string_param(input, "file_path")?;
        let action = extract_string_param(input, "action")?;
        let output_path = extract_optional_string_param(input, "output_path");

        // Validate file path
        validation::validate_path(&file_path)?;
        let path = Path::new(&file_path);

        if !path.exists() {
            return Err(AgentError::invalid_input(format!("File not found: {}", file_path)));
        }

        // Get file size for validation
        let file_size = std::fs::metadata(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to get file metadata: {}", e)))?
            .len() as usize;

        // Validate audio file
        let processor = self.processor.lock().await;
        processor.validate_audio_file(path, file_size)?;
        drop(processor);

        match action.as_str() {
            "decode" => self.decode_audio_file(path).await,
            "metadata" => self.extract_metadata(path).await,
            "analyze" => self.analyze_audio_file(path).await,
            "effects" => self.apply_effects_to_file(path, input, output_path).await,
            "transcribe" => self.transcribe_audio_file(path).await,
            "convert" => self.convert_audio_file(path, input, output_path).await,
            _ => Err(AgentError::invalid_input(format!("Unsupported action: {}", action))),
        }
    }

    /// Decode audio file
    async fn decode_audio_file(&self, path: &Path) -> Result<ToolResult> {
        info!("Decoding audio file: {}", path.display());

        let audio = AudioCodec::decode_file(path)?;
        
        let result = json!({
            "action": "decode",
            "file_path": path.display().to_string(),
            "format": audio.format,
            "duration_seconds": audio.duration,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "sample_count": audio.samples.len(),
            "frames": audio.frames(),
        });

        Ok(ToolResult::success(format!(
            "Successfully decoded audio file:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Extract metadata from audio file
    async fn extract_metadata(&self, path: &Path) -> Result<ToolResult> {
        info!("Extracting metadata from: {}", path.display());

        let metadata = MetadataExtractor::extract_from_file(path)?;
        let summary = MetadataExtractor::get_metadata_summary(&metadata);
        let validation_issues = MetadataExtractor::validate_metadata(&metadata);

        let result = json!({
            "action": "metadata",
            "file_path": path.display().to_string(),
            "metadata": metadata,
            "summary": summary,
            "validation_issues": validation_issues,
            "has_complete_metadata": MetadataExtractor::has_complete_metadata(&metadata),
        });

        Ok(ToolResult::success(format!(
            "Metadata extraction completed:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Analyze audio file
    async fn analyze_audio_file(&self, path: &Path) -> Result<ToolResult> {
        info!("Analyzing audio file: {}", path.display());

        // Decode audio
        let audio = AudioCodec::decode_file(path)?;
        
        // Analyze with effects processor
        let effects = self.effects.lock().await;
        let analysis = effects.analyze_audio(&audio);
        drop(effects);

        // Extract metadata
        let metadata = MetadataExtractor::extract_from_file(path)?;

        let result = json!({
            "action": "analyze",
            "file_path": path.display().to_string(),
            "audio_analysis": {
                "peak_amplitude": analysis.peak_amplitude,
                "rms_level": analysis.rms_level,
                "peak_db": analysis.peak_db,
                "rms_db": analysis.rms_db,
                "dynamic_range_db": analysis.dynamic_range_db,
                "zero_crossing_rate": analysis.zero_crossing_rate,
                "duration_seconds": analysis.duration_seconds,
                "sample_count": analysis.sample_count,
            },
            "technical_info": {
                "format": audio.format,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "frames": audio.frames(),
            },
            "metadata_summary": MetadataExtractor::get_metadata_summary(&metadata),
        });

        Ok(ToolResult::success(format!(
            "Audio analysis completed:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Apply effects to audio file
    async fn apply_effects_to_file(&self, path: &Path, input: &Value, output_path: Option<String>) -> Result<ToolResult> {
        info!("Applying effects to audio file: {}", path.display());

        // Decode audio
        let audio = AudioCodec::decode_file(path)?;

        // Determine effects preset
        let preset = extract_optional_string_param(input, "preset").unwrap_or_default();
        let mut effects = match preset.as_str() {
            "voice" => create_voice_effects_processor(),
            "music" => create_music_effects_processor(),
            _ => {
                // Use default effects configuration
                super::effects::create_default_effects_processor()
            }
        };

        // Apply effects
        let processed_audio = effects.process(&audio)?;

        // Save to output file if specified
        if let Some(ref output_path) = output_path {
            validation::validate_path(&output_path)?;
            let output_path = Path::new(&output_path);
            
            let quality = AudioQuality::default();
            AudioCodec::encode_file(&processed_audio, output_path, &quality)?;
            
            info!("Processed audio saved to: {}", output_path.display());
        }

        let result = json!({
            "action": "effects",
            "input_file": path.display().to_string(),
            "output_file": output_path,
            "preset": preset,
            "original_duration": audio.duration,
            "processed_duration": processed_audio.duration,
            "original_samples": audio.samples.len(),
            "processed_samples": processed_audio.samples.len(),
        });

        Ok(ToolResult::success(format!(
            "Audio effects applied successfully:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Transcribe audio file
    async fn transcribe_audio_file(&self, path: &Path) -> Result<ToolResult> {
        info!("Transcribing audio file: {}", path.display());

        let transcription_service = self.transcription_service.as_ref()
            .ok_or_else(|| AgentError::tool("audio_processing".to_string(), "Transcription service not configured".to_string()))?;

        // Read file as bytes for transcription
        let audio_bytes = std::fs::read(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to read audio file: {}", e)))?;

        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        // Perform transcription
        let service = transcription_service.lock().await;
        let transcription = service.transcribe_bytes(&audio_bytes, extension).await?;
        drop(service);

        let result = json!({
            "action": "transcribe",
            "file_path": path.display().to_string(),
            "transcription": {
                "text": transcription.text,
                "language": transcription.language,
                "confidence": transcription.confidence,
                "processing_duration_ms": transcription.processing_duration_ms,
                "word_count": transcription.text.split_whitespace().count(),
                "has_timestamps": transcription.words.is_some(),
                "segment_count": transcription.segments.as_ref().map(|s| s.len()).unwrap_or(0),
            }
        });

        Ok(ToolResult::success(format!(
            "Audio transcription completed:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Convert audio file format
    async fn convert_audio_file(&self, path: &Path, input: &Value, output_path: Option<String>) -> Result<ToolResult> {
        info!("Converting audio file: {}", path.display());

        let output_path = output_path.ok_or_else(|| 
            AgentError::invalid_input("Output path required for conversion".to_string()))?;

        validation::validate_path(&output_path)?;
        let output_path = Path::new(&output_path);

        // Decode input audio
        let audio = AudioCodec::decode_file(path)?;

        // Get quality settings
        let sample_rate = extract_optional_string_param(input, "sample_rate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(audio.sample_rate);

        let channels = extract_optional_string_param(input, "channels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(audio.channels);

        let bit_depth = extract_optional_string_param(input, "bit_depth")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        let quality = AudioQuality {
            sample_rate,
            channels,
            bit_depth,
            bitrate: Some(128000), // Default bitrate
        };

        // Store original format before moving audio
        let original_format = audio.format;

        // Resample if necessary
        let converted_audio = if audio.sample_rate != sample_rate {
            audio.resample(sample_rate)?
        } else {
            audio
        };

        // Encode to output format
        AudioCodec::encode_file(&converted_audio, output_path, &quality)?;

        let result = json!({
            "action": "convert",
            "input_file": path.display().to_string(),
            "output_file": output_path.display().to_string(),
            "original_format": original_format,
            "target_quality": quality,
            "conversion_successful": true,
        });

        Ok(ToolResult::success(format!(
            "Audio conversion completed:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Synthesize text to speech
    async fn synthesize_text(&self, input: &Value) -> Result<ToolResult> {
        let text = extract_string_param(input, "text")?;
        let output_path = extract_optional_string_param(input, "output_path");

        let synthesis_service = self.synthesis_service.as_ref()
            .ok_or_else(|| AgentError::tool("audio_processing".to_string(), "Synthesis service not configured".to_string()))?;

        info!("Synthesizing text to speech: {} characters", text.len());

        // Perform synthesis
        let service = synthesis_service.lock().await;
        let synthesis = service.synthesize(&text).await?;
        drop(service);

        // Save to file if specified
        let output_file_path = if let Some(ref output_path) = output_path {
            validation::validate_path(output_path)?;
            let output_path = Path::new(output_path);

            let quality = AudioQuality::default();
            AudioCodec::encode_file(&synthesis.audio, output_path, &quality)?;

            info!("Synthesized audio saved to: {}", output_path.display());
            Some(output_path.display().to_string())
        } else {
            None
        };

        let result = json!({
            "action": "synthesize",
            "text": text,
            "output_file": output_file_path,
            "synthesis": {
                "voice": synthesis.voice,
                "format": synthesis.format,
                "duration_seconds": synthesis.audio.duration,
                "sample_rate": synthesis.audio.sample_rate,
                "channels": synthesis.audio.channels,
                "processing_duration_ms": synthesis.processing_duration_ms,
            }
        });

        Ok(ToolResult::success(format!(
            "Text-to-speech synthesis completed:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Get audio processing statistics
    async fn get_statistics(&self) -> Result<ToolResult> {
        let processor = self.processor.lock().await;
        let stats = processor.get_stats().await;
        drop(processor);

        let result = json!({
            "action": "statistics",
            "stats": stats,
        });

        Ok(ToolResult::success(format!(
            "Audio processing statistics:\n{}",
            serde_json::to_string_pretty(&result)?
        )))
    }

    /// Extract comprehensive metadata using the instance metadata extractor
    pub async fn extract_comprehensive_metadata(&self, file_path: &str) -> Result<AudioMetadata> {
        // Validate file path
        crate::utils::validation::validate_path(file_path)?;

        // Use the instance metadata extractor
        let metadata = self.metadata_extractor.extract_metadata(file_path).await?;

        info!("Extracted comprehensive metadata from: {} (duration: {:?}s, format: {:?})",
              file_path, metadata.duration_seconds, metadata.format);

        Ok(metadata)
    }

    /// Extract metadata from audio data buffer using instance extractor
    pub async fn extract_metadata_from_buffer(&self, audio_data: &AudioData) -> Result<AudioMetadata> {
        // Use the instance metadata extractor for buffer analysis
        let metadata = self.metadata_extractor.extract_from_buffer(audio_data).await?;

        info!("Extracted metadata from audio buffer (samples: {}, channels: {}, sample_rate: {})",
              audio_data.samples.len(), audio_data.channels, audio_data.sample_rate);

        Ok(metadata)
    }

    /// Get audio file duration using instance metadata extractor
    pub async fn get_audio_duration(&self, file_path: &str) -> Result<f64> {
        let metadata = self.extract_comprehensive_metadata(file_path).await?;
        Ok(metadata.duration_seconds)
    }

    /// Estimate synthesis duration for text using synthesis service
    pub async fn estimate_synthesis_duration(&self, text: &str, speed: Option<f32>) -> Result<f64> {
        let synthesis_service = self.synthesis_service.as_ref()
            .ok_or_else(|| AgentError::tool("audio_processing".to_string(), "Synthesis service not configured".to_string()))?;

        let service = synthesis_service.lock().await;
        let duration = service.estimate_audio_duration(text, speed.unwrap_or(1.0));
        drop(service);

        info!("Estimated synthesis duration for {} characters: {:.2}s (speed: {:.1}x)",
              text.len(), duration, speed.unwrap_or(1.0));

        Ok(duration)
    }

    /// Analyze audio file comprehensively using all available tools
    pub async fn analyze_audio_comprehensive(&self, file_path: &str) -> Result<AudioAnalysisResult> {
        info!("Starting comprehensive audio analysis for: {}", file_path);

        // Extract metadata using instance extractor
        let metadata = self.extract_comprehensive_metadata(file_path).await?;

        // Load audio data for effects analysis
        let audio_data = AudioCodec::decode_file(std::path::Path::new(file_path))?;

        // Analyze audio characteristics using effects processor
        let effects_analysis = {
            let effects = self.effects.lock().await;
            effects.analyze_audio_characteristics(&audio_data).await?
        };

        // Estimate synthesis duration if text content is available
        let estimated_synthesis_duration = if let Some(ref synthesis_service) = self.synthesis_service {
            if let Some(text_content) = metadata.title.as_ref().or(metadata.comment.as_ref()) {
                let service = synthesis_service.lock().await;
                Some(service.estimate_audio_duration(text_content, 1.0))
            } else {
                None
            }
        } else {
            None
        };

        let result = AudioAnalysisResult {
            metadata,
            effects_analysis,
            estimated_synthesis_duration,
            file_path: file_path.to_string(),
            analysis_timestamp: chrono::Utc::now(),
        };

        info!("Completed comprehensive audio analysis for: {}", file_path);
        Ok(result)
    }

    /// Process audio with comprehensive pipeline using all components
    pub async fn process_audio_comprehensive(&self, file_path: &str, effects_config: Option<EffectsConfig>) -> Result<ProcessedAudioResult> {
        info!("Starting comprehensive audio processing pipeline for: {}", file_path);

        // Step 1: Extract initial metadata using instance extractor
        let initial_metadata = self.extract_comprehensive_metadata(file_path).await?;

        // Step 2: Load audio data
        let mut audio_data = AudioCodec::decode_file(std::path::Path::new(file_path))?;

        // Step 3: Apply effects if configured
        let effects_applied = if let Some(config) = effects_config {
            let effects = self.effects.lock().await;
            effects.apply_effects(&mut audio_data, &config).await?;
            true
        } else {
            false
        };

        // Step 4: Extract metadata from processed audio using instance extractor
        let final_metadata = self.extract_metadata_from_buffer(&audio_data).await?;

        // Step 5: Analyze processed audio
        let effects_analysis = {
            let effects = self.effects.lock().await;
            effects.analyze_audio_characteristics(&audio_data).await?
        };

        let result = ProcessedAudioResult {
            initial_metadata,
            final_metadata,
            processed_audio: audio_data,
            effects_analysis,
            effects_applied,
            processing_timestamp: chrono::Utc::now(),
        };

        info!("Completed comprehensive audio processing pipeline for: {} (effects applied: {})",
              file_path, effects_applied);

        Ok(result)
    }

    /// Batch process multiple audio files using comprehensive analysis
    pub async fn batch_process_audio(&self, file_paths: &[String]) -> Result<Vec<AudioAnalysisResult>> {
        let mut results = Vec::new();

        info!("Starting batch processing of {} audio files", file_paths.len());

        for (index, file_path) in file_paths.iter().enumerate() {
            match self.analyze_audio_comprehensive(file_path).await {
                Ok(result) => {
                    results.push(result);
                    info!("Successfully processed audio file {}/{}: {}",
                          index + 1, file_paths.len(), file_path);
                }
                Err(e) => {
                    warn!("Failed to process audio file {}/{} ({}): {}",
                          index + 1, file_paths.len(), file_path, e);
                    // Continue processing other files
                }
            }
        }

        info!("Batch processing completed: {} successful out of {} files",
              results.len(), file_paths.len());

        Ok(results)
    }
}

#[async_trait]
impl Tool for AudioProcessingTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "audio_processing",
            "Comprehensive audio processing tool supporting format conversion, transcription, synthesis, effects, and metadata extraction",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "process_file", "synthesize", "statistics",
                            "decode", "metadata", "analyze", "effects", "transcribe", "convert"
                        ],
                        "description": "Action to perform"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to audio file (required for file operations)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize (required for synthesis)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional)"
                    },
                    "preset": {
                        "type": "string",
                        "enum": ["voice", "music", "custom"],
                        "description": "Effects preset to apply"
                    },
                    "sample_rate": {
                        "type": "string",
                        "description": "Target sample rate for conversion"
                    },
                    "channels": {
                        "type": "string",
                        "description": "Target channel count for conversion"
                    },
                    "bit_depth": {
                        "type": "string",
                        "description": "Target bit depth for conversion"
                    }
                },
                "required": ["action"]
            }),
        )
    }

    async fn execute(&self, input: Value) -> Result<ToolResult> {
        let action = extract_string_param(&input, "action")?;

        debug!("Executing audio processing action: {}", action);

        match action.as_str() {
            "process_file" | "decode" | "metadata" | "analyze" | "effects" | "transcribe" | "convert" => {
                self.process_audio_file(&input).await
            }
            "synthesize" => self.synthesize_text(&input).await,
            "statistics" => self.get_statistics().await,
            _ => Err(AgentError::invalid_input(format!("Unsupported action: {}", action))),
        }
    }

    fn name(&self) -> &str {
        "audio_processing"
    }

    fn description(&self) -> Option<&str> {
        Some("Comprehensive audio processing tool supporting format conversion, transcription, synthesis, effects, and metadata extraction")
    }

    fn validate_input(&self, input: &Value) -> Result<()> {
        let action = extract_string_param(input, "action")?;

        match action.as_str() {
            "process_file" | "decode" | "metadata" | "analyze" | "effects" | "transcribe" | "convert" => {
                extract_string_param(input, "file_path")?;
            }
            "synthesize" => {
                extract_string_param(input, "text")?;
            }
            "statistics" => {
                // No additional validation needed
            }
            _ => return Err(AgentError::invalid_input(format!("Unsupported action: {}", action))),
        }

        Ok(())
    }
}

/// Create an audio processing tool with default configuration
pub fn create_default_audio_tool() -> Result<AudioProcessingTool> {
    let config = AudioConfig::default();
    AudioProcessingTool::new(config)
}

/// Create an audio processing tool with OpenAI services
pub fn create_openai_audio_tool(api_key: String) -> Result<AudioProcessingTool> {
    let mut config = AudioConfig::default();
    
    // Configure transcription
    config.transcription.provider = "openai".to_string();
    config.transcription.api_key = Some(api_key.clone());
    config.transcription.model = "whisper-1".to_string();
    
    // Configure synthesis
    config.synthesis.provider = "openai".to_string();
    config.synthesis.api_key = Some(api_key);
    config.synthesis.voice = "alloy".to_string();
    
    AudioProcessingTool::new(config)
}
