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

    /// Transcribe using Azure Speech Services
    async fn transcribe_with_azure(&self, audio: &AudioData) -> Result<TranscriptionResult> {
        debug!("Starting Azure Speech Services transcription");

        let subscription_key = std::env::var("AZURE_SPEECH_KEY")
            .map_err(|_| AgentError::tool("transcription".to_string(), "AZURE_SPEECH_KEY environment variable not set".to_string()))?;

        let region = std::env::var("AZURE_SPEECH_REGION")
            .unwrap_or_else(|_| "eastus".to_string());

        // Get access token
        let token = self.get_azure_access_token(&subscription_key, &region).await?;

        // Convert audio to required format (WAV, 16kHz, mono)
        let audio_bytes = self.convert_audio_for_azure(audio).await?;

        // Create transcription request
        let client = reqwest::Client::new();
        let endpoint = format!("https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1", region);

        let response = client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "audio/wav; codecs=audio/pcm; samplerate=16000")
            .header("Accept", "application/json")
            .query(&[
                ("language", "en-US"),
                ("format", "detailed"),
                ("profanity", "masked")
            ])
            .body(audio_bytes)
            .send()
            .await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Azure API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool("transcription".to_string(), format!("Azure API error: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to parse Azure response: {}", e)))?;

        // Parse Azure response
        let text = response_json
            .get("DisplayText")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let confidence = response_json
            .get("NBest")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("Confidence"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        info!("Azure transcription completed: {} characters, confidence: {:.2}", text.len(), confidence);

        Ok(TranscriptionResult {
            text,
            confidence: Some(confidence as f32),
            language: Some("en-US".to_string()),
            processing_duration_ms: 0, // Will be set by caller
            words: None, // Azure detailed format could provide word-level timestamps
            segments: Some(Vec::new()), // Azure detailed format could provide segments
        })
    }

    /// Transcribe bytes using Azure Speech Services
    async fn transcribe_bytes_with_azure(&self, audio_bytes: &[u8], format: &str) -> Result<TranscriptionResult> {
        debug!("Starting Azure Speech Services transcription from bytes, format: {}", format);

        let subscription_key = std::env::var("AZURE_SPEECH_KEY")
            .map_err(|_| AgentError::tool("transcription".to_string(), "AZURE_SPEECH_KEY environment variable not set".to_string()))?;

        let region = std::env::var("AZURE_SPEECH_REGION")
            .unwrap_or_else(|_| "eastus".to_string());

        // Get access token
        let token = self.get_azure_access_token(&subscription_key, &region).await?;

        // Convert audio bytes to required format if needed
        let processed_bytes = if format.to_lowercase().contains("wav") && format.contains("16000") {
            audio_bytes.to_vec()
        } else {
            self.convert_audio_bytes_for_azure(audio_bytes, format).await?
        };

        // Create transcription request
        let client = reqwest::Client::new();
        let endpoint = format!("https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1", region);

        let response = client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "audio/wav; codecs=audio/pcm; samplerate=16000")
            .header("Accept", "application/json")
            .query(&[
                ("language", "en-US"),
                ("format", "detailed"),
                ("profanity", "masked")
            ])
            .body(processed_bytes)
            .send()
            .await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Azure API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool("transcription".to_string(), format!("Azure API error: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to parse Azure response: {}", e)))?;

        // Parse Azure response
        let text = response_json
            .get("DisplayText")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let confidence = response_json
            .get("NBest")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("Confidence"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        info!("Azure transcription from bytes completed: {} characters, confidence: {:.2}", text.len(), confidence);

        Ok(TranscriptionResult {
            text,
            confidence: Some(confidence as f32),
            language: Some("en-US".to_string()),
            processing_duration_ms: 0, // Will be set by caller
            words: None, // Azure detailed format could provide word-level timestamps
            segments: Some(Vec::new()), // Azure detailed format could provide segments
        })
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

    // ========================================
    // Azure Speech Services Helper Methods
    // ========================================

    /// Get Azure access token for Speech Services
    async fn get_azure_access_token(&self, subscription_key: &str, region: &str) -> Result<String> {
        let token_endpoint = format!("https://{}.api.cognitive.microsoft.com/sts/v1.0/issuetoken", region);

        let response = self.client
            .post(&token_endpoint)
            .header("Ocp-Apim-Subscription-Key", subscription_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to get Azure token: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool("transcription".to_string(), format!("Azure token request failed: {}", error_text)));
        }

        let token = response.text().await
            .map_err(|e| AgentError::tool("transcription".to_string(), format!("Failed to read Azure token: {}", e)))?;

        Ok(token)
    }

    /// Convert AudioData to Azure-compatible format (WAV, 16kHz, mono)
    async fn convert_audio_for_azure(&self, audio: &AudioData) -> Result<Vec<u8>> {
        // Convert to the required format: WAV, 16kHz, mono
        let mut processed_audio = audio.clone();

        // Convert to mono if needed
        if processed_audio.channels > 1 {
            processed_audio = processed_audio.to_mono();
        }

        // Resample to 16kHz if needed
        if processed_audio.sample_rate != 16000 {
            processed_audio = processed_audio.resample(16000)?;
        }

        // Encode to WAV format
        let quality = super::AudioQuality {
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
            bitrate: None,
        };

        let wav_bytes = super::codecs::AudioCodec::encode_bytes(&processed_audio, super::AudioFormat::Wav, &quality)?;
        Ok(wav_bytes)
    }

    /// Convert audio bytes to Azure-compatible format
    async fn convert_audio_bytes_for_azure(&self, audio_bytes: &[u8], format: &str) -> Result<Vec<u8>> {
        // For now, assume the audio is already in the correct format
        // In a real implementation, you would use audio processing libraries
        // to convert the audio format based on the input format string

        if format.to_lowercase().contains("wav") && format.contains("16000") {
            Ok(audio_bytes.to_vec())
        } else {
            // Placeholder for audio conversion
            // In production, implement actual audio format conversion
            warn!("Audio format conversion not fully implemented for format: {}. Attempting to use original data.", format);
            Ok(audio_bytes.to_vec())
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::codecs::AudioData;
    use crate::audio::AudioFormat;

    fn create_test_audio() -> AudioData {
        // Create 1 second of test audio at 16kHz mono
        let sample_rate = 16000;
        let duration = 1.0;
        let samples_count = (sample_rate as f64 * duration) as usize;

        // Generate a simple sine wave
        let mut samples = Vec::with_capacity(samples_count);
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let frequency = 440.0; // A4 note
            let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * 0.5;
            samples.push(sample);
        }

        AudioData::new(samples, sample_rate, 1, AudioFormat::Wav)
    }

    #[tokio::test]
    async fn test_transcription_service_creation() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config);
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_audio_validation() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config).unwrap();

        // Test valid audio
        let audio = create_test_audio();
        assert!(service.validate_audio(&audio).is_ok());

        // Test empty audio
        let empty_audio = AudioData::new(Vec::new(), 16000, 1, AudioFormat::Wav);
        assert!(service.validate_audio(&empty_audio).is_err());

        // Test audio that's too long
        let long_samples = vec![0.0f32; 16000 * 700]; // 700 seconds
        let long_audio = AudioData::new(long_samples, 16000, 1, AudioFormat::Wav);
        assert!(service.validate_audio(&long_audio).is_err());
    }

    #[tokio::test]
    async fn test_azure_audio_conversion() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config).unwrap();

        // Test audio that needs conversion (stereo, 44.1kHz)
        let samples = vec![0.1f32; 44100 * 2]; // 1 second stereo
        let audio = AudioData::new(samples, 44100, 2, AudioFormat::Wav);

        let result = service.convert_audio_for_azure(&audio).await;
        assert!(result.is_ok());

        let wav_bytes = result.unwrap();
        assert!(!wav_bytes.is_empty());

        // The result should be a valid WAV file
        assert!(wav_bytes.starts_with(b"RIFF"));
    }

    #[tokio::test]
    async fn test_azure_audio_conversion_already_correct_format() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config).unwrap();

        // Test audio that's already in correct format (mono, 16kHz)
        let audio = create_test_audio();

        let result = service.convert_audio_for_azure(&audio).await;
        assert!(result.is_ok());

        let wav_bytes = result.unwrap();
        assert!(!wav_bytes.is_empty());
        assert!(wav_bytes.starts_with(b"RIFF"));
    }

    #[tokio::test]
    async fn test_convert_audio_bytes_for_azure() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config).unwrap();

        // Test with WAV format that contains "16000"
        let test_bytes = b"test audio data";
        let result = service.convert_audio_bytes_for_azure(test_bytes, "wav_16000").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_bytes.to_vec());

        // Test with other format
        let result = service.convert_audio_bytes_for_azure(test_bytes, "mp3").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_bytes.to_vec());
    }

    #[tokio::test]
    async fn test_supported_languages() {
        let config = TranscriptionConfig::default();
        let service = TranscriptionService::new(config).unwrap();

        let languages = service.get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en".to_string()));

        assert!(service.is_language_supported("en"));
        assert!(!service.is_language_supported("xyz"));
    }

    #[tokio::test]
    async fn test_transcription_provider_from_string() {
        assert!(matches!(
            TranscriptionProvider::from_string("openai"),
            Some(TranscriptionProvider::OpenAIWhisper)
        ));
        assert!(matches!(
            TranscriptionProvider::from_string("azure"),
            Some(TranscriptionProvider::AzureSpeech)
        ));
        assert!(matches!(
            TranscriptionProvider::from_string("google"),
            Some(TranscriptionProvider::GoogleSpeech)
        ));
        assert!(matches!(
            TranscriptionProvider::from_string("local"),
            Some(TranscriptionProvider::LocalWhisper)
        ));
        assert!(TranscriptionProvider::from_string("invalid").is_none());
    }

    #[tokio::test]
    async fn test_azure_transcription_missing_env_vars() {
        // Test Azure transcription without environment variables
        let mut config = TranscriptionConfig::default();
        config.provider = "azure".to_string();
        let service = TranscriptionService::new(config).unwrap();

        let audio = create_test_audio();

        // This should fail because AZURE_SPEECH_KEY is not set
        let result = service.transcribe(&audio).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("AZURE_SPEECH_KEY"));
    }

    #[tokio::test]
    async fn test_azure_transcription_bytes_missing_env_vars() {
        // Test Azure transcription from bytes without environment variables
        let mut config = TranscriptionConfig::default();
        config.provider = "azure".to_string();
        let service = TranscriptionService::new(config).unwrap();

        let test_bytes = b"test audio data";

        // This should fail because AZURE_SPEECH_KEY is not set
        let result = service.transcribe_bytes(test_bytes, "wav").await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("AZURE_SPEECH_KEY"));
    }

    #[test]
    fn test_speech_activity_detection() {
        // Create test audio with speech activity
        let sample_rate = 16000;
        let duration = 2.0;
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::with_capacity(samples_count);

        // First half: silence
        for _ in 0..(samples_count / 2) {
            samples.push(0.01f32); // Very low amplitude
        }

        // Second half: speech (higher amplitude)
        for i in (samples_count / 2)..samples_count {
            let t = i as f64 / sample_rate as f64;
            let frequency = 440.0;
            let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * 0.8;
            samples.push(sample);
        }

        let audio = AudioData::new(samples, sample_rate, 1, AudioFormat::Wav);
        let segments = detect_speech_activity(&audio, 0.1);

        // Should detect speech in the second half
        assert!(!segments.is_empty());
        assert!(segments[0].0 >= 0.9); // Speech starts around 1 second
        assert!(segments[0].1 <= 2.1); // Speech ends around 2 seconds
    }

    #[tokio::test]
    async fn test_create_transcription_services() {
        // Test default service creation
        let service = create_default_transcription_service();
        assert!(service.is_ok());

        // Test OpenAI service creation
        let service = create_openai_transcription_service("test-key".to_string());
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_google_transcription_placeholder() {
        let mut config = TranscriptionConfig::default();
        config.provider = "google".to_string();
        let service = TranscriptionService::new(config).unwrap();

        let audio = create_test_audio();
        let result = service.transcribe(&audio).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Google Cloud Speech-to-Text not implemented yet"));
    }

    #[tokio::test]
    async fn test_local_whisper_placeholder() {
        let mut config = TranscriptionConfig::default();
        config.provider = "local".to_string();
        let service = TranscriptionService::new(config).unwrap();

        let audio = create_test_audio();
        let result = service.transcribe(&audio).await;
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Local Whisper implementation not available yet"));
    }
}
