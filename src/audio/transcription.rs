// Audio Transcription Service Integration
// Provides speech-to-text functionality using Whisper API and other services

use super::{TranscriptionConfig, codecs::AudioData};
use crate::utils::error::{AgentError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Detected language (if available)
    pub language: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// Processing duration in milliseconds
    pub processing_duration_ms: u64,
    /// Word-level timestamps (if available)
    pub words: Option<Vec<WordTimestamp>>,
    /// Segments with timestamps (if available)
    pub segments: Option<Vec<TranscriptionSegment>>,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Confidence score for this word
    pub confidence: Option<f32>,
}

/// Transcription segment with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Average confidence for this segment
    pub avg_confidence: Option<f32>,
    /// Words in this segment
    pub words: Option<Vec<WordTimestamp>>,
}

/// Transcription service provider
#[derive(Debug, Clone)]
pub enum TranscriptionProvider {
    /// OpenAI Whisper API
    OpenAIWhisper,
    /// Azure Speech Services
    AzureSpeech,
    /// Google Cloud Speech-to-Text
    GoogleSpeech,
    /// Local Whisper implementation
    LocalWhisper,
}

impl TranscriptionProvider {
    /// Get provider from string
    pub fn from_string(provider: &str) -> Option<Self> {
        match provider.to_lowercase().as_str() {
            "whisper" | "openai" | "openai-whisper" => Some(TranscriptionProvider::OpenAIWhisper),
            "azure" | "azure-speech" => Some(TranscriptionProvider::AzureSpeech),
            "google" | "google-speech" => Some(TranscriptionProvider::GoogleSpeech),
            "local" | "local-whisper" => Some(TranscriptionProvider::LocalWhisper),
            _ => None,
        }
    }
}

/// Transcription service for speech-to-text conversion
#[derive(Debug)]
pub struct TranscriptionService {
    /// Service configuration
    config: TranscriptionConfig,
    /// HTTP client for API requests
    client: Client,
    /// Service provider
    provider: TranscriptionProvider,
}

impl TranscriptionService {
    /// Create a new transcription service
    pub fn new(config: TranscriptionConfig) -> Result<Self> {
        let provider = TranscriptionProvider::from_string(&config.provider)
            .ok_or_else(|| AgentError::invalid_input(format!("Unsupported transcription provider: {}", config.provider)))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            provider,
        })
    }

    /// Transcribe audio data to text
    pub async fn transcribe(&self, audio: &AudioData) -> Result<TranscriptionResult> {
        let start_time = Instant::now();
        info!("Starting transcription with provider: {:?}", self.provider);

        // Validate audio data
        self.validate_audio(audio)?;

        // Perform transcription based on provider
        let result = match self.provider {
            TranscriptionProvider::OpenAIWhisper => self.transcribe_with_openai(audio).await,
            TranscriptionProvider::AzureSpeech => self.transcribe_with_azure(audio).await,
            TranscriptionProvider::GoogleSpeech => self.transcribe_with_google(audio).await,
            TranscriptionProvider::LocalWhisper => self.transcribe_with_local_whisper(audio).await,
        };

        match result {
            Ok(mut transcription) => {
                transcription.processing_duration_ms = start_time.elapsed().as_millis() as u64;
                info!("Transcription completed in {}ms", transcription.processing_duration_ms);
                Ok(transcription)
            }
            Err(e) => {
                error!("Transcription failed: {}", e);
                Err(e)
            }
        }
    }

    /// Transcribe audio file from bytes
    pub async fn transcribe_bytes(&self, audio_bytes: &[u8], format: &str) -> Result<TranscriptionResult> {
        let start_time = Instant::now();
        info!("Starting transcription from bytes, format: {}", format);

        let result = match self.provider {
            TranscriptionProvider::OpenAIWhisper => self.transcribe_bytes_with_openai(audio_bytes, format).await,
            TranscriptionProvider::AzureSpeech => self.transcribe_bytes_with_azure(audio_bytes, format).await,
            TranscriptionProvider::GoogleSpeech => self.transcribe_bytes_with_google(audio_bytes, format).await,
            TranscriptionProvider::LocalWhisper => {
                // For local whisper, we need to decode the audio first
                let audio_format = super::AudioFormat::from_extension(format)
                    .ok_or_else(|| AgentError::invalid_input(format!("Unsupported format: {}", format)))?;
                let audio = super::codecs::AudioCodec::decode_bytes(audio_bytes, audio_format)?;
                self.transcribe_with_local_whisper(&audio).await
            }
        };

        match result {
            Ok(mut transcription) => {
                transcription.processing_duration_ms = start_time.elapsed().as_millis() as u64;
                info!("Transcription completed in {}ms", transcription.processing_duration_ms);
                Ok(transcription)
            }
            Err(e) => {
                error!("Transcription failed: {}", e);
                Err(e)
            }
        }
    }

    /// Validate audio data for transcription
    fn validate_audio(&self, audio: &AudioData) -> Result<()> {
        // Check duration (most services have limits)
        if audio.duration > 600.0 {
            return Err(AgentError::invalid_input(
                "Audio too long for transcription (max 10 minutes)".to_string(),
            ));
        }

        // Check if audio has content
        if audio.samples.is_empty() {
            return Err(AgentError::invalid_input("Audio data is empty".to_string()));
        }

        // Check for silence (all samples near zero)
        let max_amplitude = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max_amplitude < 0.001 {
            warn!("Audio appears to be silent (max amplitude: {})", max_amplitude);
        }

        Ok(())
    }

    /// Transcribe using OpenAI Whisper API
    async fn transcribe_with_openai(&self, audio: &AudioData) -> Result<TranscriptionResult> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| AgentError::authentication("OpenAI API key not configured".to_string()))?;

        // Convert audio to WAV format for API
        let wav_data = super::codecs::AudioCodec::encode_bytes(
            audio,
            super::AudioFormat::Wav,
            &super::AudioQuality::voice(), // Use voice quality for transcription
        )?;

        // Prepare multipart form
        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(wav_data)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to create form part: {}", e)))?)
            .text("model", self.config.model.clone())
            .text("response_format", "verbose_json");

        let form = if let Some(ref language) = self.config.language {
            form.text("language", language.clone())
        } else {
            form
        };

        // Make API request
        let endpoint = self.config.endpoint.as_deref().unwrap_or("https://api.openai.com/v1/audio/transcriptions");
        
        let response = timeout(
            Duration::from_secs(self.config.timeout),
            self.client
                .post(endpoint)
                .header("Authorization", format!("Bearer {}", api_key))
                .multipart(form)
                .send()
        ).await
        .map_err(|_| AgentError::tool("transcription".to_string(), "Request timeout".to_string()))?
        .map_err(|e| AgentError::tool("transcription".to_string(), format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::tool("transcription".to_string(), format!("API error: {}", error_text)));
        }

        // Parse response
        let response_text = response.text().await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to read response: {}", e)))?;

        let whisper_response: WhisperResponse = serde_json::from_str(&response_text)
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to parse response: {}", e)))?;

        Ok(TranscriptionResult {
            text: whisper_response.text,
            language: whisper_response.language,
            confidence: None, // OpenAI doesn't provide overall confidence
            processing_duration_ms: 0, // Will be set by caller
            words: whisper_response.words,
            segments: whisper_response.segments,
        })
    }

    /// Transcribe bytes using OpenAI Whisper API
    async fn transcribe_bytes_with_openai(&self, audio_bytes: &[u8], format: &str) -> Result<TranscriptionResult> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| AgentError::authentication("OpenAI API key not configured".to_string()))?;

        // Prepare multipart form
        let mime_type = match format {
            "wav" => "audio/wav",
            "mp3" => "audio/mpeg",
            "flac" => "audio/flac",
            "ogg" => "audio/ogg",
            "m4a" => "audio/mp4",
            _ => "audio/wav",
        };

        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(audio_bytes.to_vec())
                .file_name(format!("audio.{}", format))
                .mime_str(mime_type)
                .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to create form part: {}", e)))?)
            .text("model", self.config.model.clone())
            .text("response_format", "verbose_json");

        let form = if let Some(ref language) = self.config.language {
            form.text("language", language.clone())
        } else {
            form
        };

        // Make API request
        let endpoint = self.config.endpoint.as_deref().unwrap_or("https://api.openai.com/v1/audio/transcriptions");
        
        let response = timeout(
            Duration::from_secs(self.config.timeout),
            self.client
                .post(endpoint)
                .header("Authorization", format!("Bearer {}", api_key))
                .multipart(form)
                .send()
        ).await
        .map_err(|_| AgentError::tool("transcription".to_string(), "Request timeout".to_string()))?
        .map_err(|e| AgentError::tool("transcription".to_string(), format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::tool("transcription".to_string(), format!("API error: {}", error_text)));
        }

        // Parse response
        let response_text = response.text().await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to read response: {}", e)))?;

        let whisper_response: WhisperResponse = serde_json::from_str(&response_text)
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to parse response: {}", e)))?;

        Ok(TranscriptionResult {
            text: whisper_response.text,
            language: whisper_response.language,
            confidence: None,
            processing_duration_ms: 0,
            words: whisper_response.words,
            segments: whisper_response.segments,
        })
    }

    /// Transcribe using Azure Speech Services (placeholder)
    async fn transcribe_with_azure(&self, _audio: &AudioData) -> Result<TranscriptionResult> {
        Err(AgentError::tool("transcription".to_string(), "Azure Speech Services not implemented yet".to_string()))
    }

    /// Transcribe bytes using Azure Speech Services (placeholder)
    async fn transcribe_bytes_with_azure(&self, _audio_bytes: &[u8], _format: &str) -> Result<TranscriptionResult> {
        Err(AgentError::tool("transcription".to_string(), "Azure Speech Services not implemented yet".to_string()))
    }

    /// Transcribe using Google Cloud Speech-to-Text (placeholder)
    async fn transcribe_with_google(&self, _audio: &AudioData) -> Result<TranscriptionResult> {
        Err(AgentError::tool("transcription".to_string(), "Google Cloud Speech-to-Text not implemented yet".to_string()))
    }

    /// Transcribe bytes using Google Cloud Speech-to-Text (placeholder)
    async fn transcribe_bytes_with_google(&self, _audio_bytes: &[u8], _format: &str) -> Result<TranscriptionResult> {
        Err(AgentError::tool("transcription".to_string(), "Google Cloud Speech-to-Text not implemented yet".to_string()))
    }

    /// Transcribe using local Whisper implementation (placeholder)
    async fn transcribe_with_local_whisper(&self, _audio: &AudioData) -> Result<TranscriptionResult> {
        Err(AgentError::tool("transcription".to_string(), "Local Whisper implementation not available yet".to_string()))
    }

    /// Get supported languages for the current provider
    pub fn get_supported_languages(&self) -> Vec<String> {
        match self.provider {
            TranscriptionProvider::OpenAIWhisper => vec![
                "en".to_string(), "es".to_string(), "fr".to_string(), "de".to_string(),
                "it".to_string(), "pt".to_string(), "ru".to_string(), "ja".to_string(),
                "ko".to_string(), "zh".to_string(), "ar".to_string(), "hi".to_string(),
                // Add more languages as supported by Whisper
            ],
            _ => vec!["en".to_string()], // Default to English for other providers
        }
    }

    /// Check if a language is supported
    pub fn is_language_supported(&self, language: &str) -> bool {
        self.get_supported_languages().contains(&language.to_string())
    }
}

/// OpenAI Whisper API response structure
#[derive(Debug, Deserialize)]
struct WhisperResponse {
    text: String,
    language: Option<String>,
    duration: Option<f64>,
    words: Option<Vec<WordTimestamp>>,
    segments: Option<Vec<TranscriptionSegment>>,
}

/// Create a transcription service with default configuration
pub fn create_default_transcription_service() -> Result<TranscriptionService> {
    let config = TranscriptionConfig::default();
    TranscriptionService::new(config)
}

/// Create a transcription service with OpenAI Whisper
pub fn create_openai_transcription_service(api_key: String) -> Result<TranscriptionService> {
    let mut config = TranscriptionConfig::default();
    config.provider = "openai".to_string();
    config.api_key = Some(api_key);
    config.model = "whisper-1".to_string();
    TranscriptionService::new(config)
}

/// Utility function to detect speech in audio
pub fn detect_speech_activity(audio: &AudioData, threshold: f32) -> Vec<(f64, f64)> {
    let mut speech_segments = Vec::new();
    let mut in_speech = false;
    let mut speech_start = 0.0;
    
    let frame_duration = 1.0 / audio.sample_rate as f64;
    let window_size = (audio.sample_rate as f64 * 0.02) as usize; // 20ms windows
    
    for (i, window) in audio.samples.chunks(window_size).enumerate() {
        let energy = window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32;
        let time = i as f64 * window_size as f64 * frame_duration;
        
        if energy > threshold && !in_speech {
            // Speech started
            in_speech = true;
            speech_start = time;
        } else if energy <= threshold && in_speech {
            // Speech ended
            in_speech = false;
            speech_segments.push((speech_start, time));
        }
    }
    
    // Handle case where speech continues to end of audio
    if in_speech {
        speech_segments.push((speech_start, audio.duration));
    }
    
    speech_segments
}
