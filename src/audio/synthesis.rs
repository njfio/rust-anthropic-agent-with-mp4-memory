// Text-to-Speech Synthesis Service
// Provides speech synthesis functionality using various TTS providers

use super::{codecs::AudioData, AudioFormat, SynthesisConfig};
use crate::utils::error::{AgentError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Speech synthesis result
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Generated audio data
    pub audio: AudioData,
    /// Processing duration in milliseconds
    pub processing_duration_ms: u64,
    /// Voice used for synthesis
    pub voice: String,
    /// Text that was synthesized
    pub text: String,
    /// Audio format of the result
    pub format: AudioFormat,
}

/// Voice information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Voice identifier
    pub id: String,
    /// Human-readable voice name
    pub name: String,
    /// Language code (e.g., "en-US", "es-ES")
    pub language: String,
    /// Gender of the voice
    pub gender: VoiceGender,
    /// Voice quality/type
    pub quality: VoiceQuality,
    /// Whether this voice supports SSML
    pub supports_ssml: bool,
    /// Sample rate for this voice
    pub sample_rate: u32,
}

/// Voice gender enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
    Unknown,
}

/// Voice quality enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceQuality {
    Standard,
    Premium,
    Neural,
    Custom,
}

/// TTS service provider
#[derive(Debug, Clone)]
pub enum SynthesisProvider {
    /// OpenAI TTS API
    OpenAI,
    /// Azure Cognitive Services Speech
    Azure,
    /// Google Cloud Text-to-Speech
    Google,
    /// Amazon Polly
    Amazon,
    /// Local TTS engine
    Local,
}

impl SynthesisProvider {
    /// Get provider from string
    pub fn from_string(provider: &str) -> Option<Self> {
        match provider.to_lowercase().as_str() {
            "openai" => Some(SynthesisProvider::OpenAI),
            "azure" => Some(SynthesisProvider::Azure),
            "google" => Some(SynthesisProvider::Google),
            "amazon" | "polly" => Some(SynthesisProvider::Amazon),
            "local" => Some(SynthesisProvider::Local),
            _ => None,
        }
    }
}

/// Text-to-speech synthesis service
#[derive(Debug)]
pub struct SynthesisService {
    /// Service configuration
    config: SynthesisConfig,
    /// HTTP client for API requests
    client: Client,
    /// Service provider
    provider: SynthesisProvider,
}

impl SynthesisService {
    /// Create a new synthesis service
    pub fn new(config: SynthesisConfig) -> Result<Self> {
        let provider = SynthesisProvider::from_string(&config.provider).ok_or_else(|| {
            AgentError::invalid_input(format!(
                "Unsupported synthesis provider: {}",
                config.provider
            ))
        })?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Failed to create HTTP client: {}", e),
                )
            })?;

        Ok(Self {
            config,
            client,
            provider,
        })
    }

    /// Synthesize text to speech
    pub async fn synthesize(&self, text: &str) -> Result<SynthesisResult> {
        let start_time = Instant::now();
        info!(
            "Starting speech synthesis with provider: {:?}",
            self.provider
        );

        // Validate input text
        self.validate_text(text)?;

        // Perform synthesis based on provider
        let result = match self.provider {
            SynthesisProvider::OpenAI => self.synthesize_with_openai(text).await,
            SynthesisProvider::Azure => self.synthesize_with_azure(text).await,
            SynthesisProvider::Google => self.synthesize_with_google(text).await,
            SynthesisProvider::Amazon => self.synthesize_with_amazon(text).await,
            SynthesisProvider::Local => self.synthesize_with_local(text).await,
        };

        match result {
            Ok(mut synthesis) => {
                synthesis.processing_duration_ms = start_time.elapsed().as_millis() as u64;
                info!(
                    "Speech synthesis completed in {}ms",
                    synthesis.processing_duration_ms
                );
                Ok(synthesis)
            }
            Err(e) => {
                error!("Speech synthesis failed: {}", e);
                Err(e)
            }
        }
    }

    /// Get available voices for the current provider
    pub async fn get_voices(&self) -> Result<Vec<VoiceInfo>> {
        match self.provider {
            SynthesisProvider::OpenAI => self.get_openai_voices().await,
            SynthesisProvider::Azure => self.get_azure_voices().await,
            SynthesisProvider::Google => self.get_google_voices().await,
            SynthesisProvider::Amazon => self.get_amazon_voices().await,
            SynthesisProvider::Local => self.get_local_voices().await,
        }
    }

    /// Validate input text for synthesis
    fn validate_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Err(AgentError::invalid_input(
                "Text cannot be empty".to_string(),
            ));
        }

        if text.len() > 4000 {
            return Err(AgentError::invalid_input(
                "Text too long for synthesis (max 4000 characters)".to_string(),
            ));
        }

        // Check for potentially problematic characters
        if text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t')
        {
            warn!("Text contains control characters that may affect synthesis");
        }

        Ok(())
    }

    /// Synthesize using OpenAI TTS API
    async fn synthesize_with_openai(&self, text: &str) -> Result<SynthesisResult> {
        let api_key = self.config.api_key.as_ref().ok_or_else(|| {
            AgentError::authentication("OpenAI API key not configured".to_string())
        })?;

        // Prepare request body
        let request_body = serde_json::json!({
            "model": "tts-1",
            "input": text,
            "voice": self.config.voice,
            "response_format": match self.config.output_format {
                AudioFormat::Mp3 => "mp3",
                AudioFormat::Wav => "wav",
                AudioFormat::Flac => "flac",
                AudioFormat::Aac => "aac",
                _ => "mp3",
            },
            "speed": self.config.speed,
        });

        // Make API request
        let endpoint = self
            .config
            .endpoint
            .as_deref()
            .unwrap_or("https://api.openai.com/v1/audio/speech");

        let response = timeout(
            Duration::from_secs(self.config.timeout),
            self.client
                .post(endpoint)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send(),
        )
        .await
        .map_err(|_| AgentError::tool("synthesis".to_string(), "Request timeout".to_string()))?
        .map_err(|e| AgentError::tool("synthesis".to_string(), format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("API error: {}", error_text),
            ));
        }

        // Get audio bytes
        let audio_bytes = response.bytes().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read audio data: {}", e),
            )
        })?;

        // Decode audio data
        let audio =
            super::codecs::AudioCodec::decode_bytes(&audio_bytes, self.config.output_format)?;

        Ok(SynthesisResult {
            audio,
            processing_duration_ms: 0, // Will be set by caller
            voice: self.config.voice.clone(),
            text: text.to_string(),
            format: self.config.output_format,
        })
    }

    /// Get OpenAI voices
    async fn get_openai_voices(&self) -> Result<Vec<VoiceInfo>> {
        // OpenAI TTS voices (as of 2024)
        Ok(vec![
            VoiceInfo {
                id: "alloy".to_string(),
                name: "Alloy".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "fable".to_string(),
                name: "Fable".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "onyx".to_string(),
                name: "Onyx".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
            VoiceInfo {
                id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                quality: VoiceQuality::Neural,
                supports_ssml: false,
                sample_rate: 24000,
            },
        ])
    }

    /// Synthesize using Azure Cognitive Services
    async fn synthesize_with_azure(&self, text: &str) -> Result<SynthesisResult> {
        debug!("Starting Azure Cognitive Services synthesis");

        let subscription_key = std::env::var("AZURE_SPEECH_KEY").map_err(|_| {
            AgentError::tool(
                "synthesis".to_string(),
                "AZURE_SPEECH_KEY environment variable not set".to_string(),
            )
        })?;

        let region = std::env::var("AZURE_SPEECH_REGION").unwrap_or_else(|_| "eastus".to_string());

        // Get access token
        let token = self
            .get_azure_access_token(&subscription_key, &region)
            .await?;

        // Create SSML for Azure TTS
        let ssml = self.create_azure_ssml(text, &self.config.voice).await?;

        // Create synthesis request
        let endpoint = format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/v1",
            region
        );

        let response = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/ssml+xml")
            .header(
                "X-Microsoft-OutputFormat",
                "audio-16khz-32kbitrate-mono-mp3",
            )
            .header("User-Agent", "rust-agent/1.0")
            .body(ssml)
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Azure API request failed: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure API error: {}", error_text),
            ));
        }

        let audio_bytes = response.bytes().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read Azure response: {}", e),
            )
        })?;

        info!(
            "Azure synthesis completed: {} bytes generated",
            audio_bytes.len()
        );

        // Decode the audio bytes to AudioData
        let audio = super::codecs::AudioCodec::decode_bytes(&audio_bytes, super::AudioFormat::Mp3)?;

        Ok(SynthesisResult {
            audio,
            processing_duration_ms: 0, // Will be set by caller
            voice: self.config.voice.clone(),
            text: text.to_string(),
            format: super::AudioFormat::Mp3,
        })
    }

    /// Get Azure voices
    async fn get_azure_voices(&self) -> Result<Vec<VoiceInfo>> {
        debug!("Getting Azure voices list");

        let subscription_key = std::env::var("AZURE_SPEECH_KEY").map_err(|_| {
            AgentError::tool(
                "synthesis".to_string(),
                "AZURE_SPEECH_KEY environment variable not set".to_string(),
            )
        })?;

        let region = std::env::var("AZURE_SPEECH_REGION").unwrap_or_else(|_| "eastus".to_string());

        // Get access token
        let token = self
            .get_azure_access_token(&subscription_key, &region)
            .await?;

        // Get voices list
        let endpoint = format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/voices/list",
            region
        );

        let response = self
            .client
            .get(&endpoint)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Azure API request failed: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure API error: {}", error_text),
            ));
        }

        let voices_json: serde_json::Value = response.json().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to parse Azure response: {}", e),
            )
        })?;

        let mut voices = Vec::new();

        if let Some(voices_array) = voices_json.as_array() {
            for voice in voices_array {
                if let (Some(name), Some(display_name), Some(locale)) = (
                    voice.get("ShortName").and_then(|v| v.as_str()),
                    voice.get("DisplayName").and_then(|v| v.as_str()),
                    voice.get("Locale").and_then(|v| v.as_str()),
                ) {
                    let gender_str = voice
                        .get("Gender")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown");

                    let gender = match gender_str.to_lowercase().as_str() {
                        "male" => VoiceGender::Male,
                        "female" => VoiceGender::Female,
                        "neutral" => VoiceGender::Neutral,
                        _ => VoiceGender::Unknown,
                    };

                    voices.push(VoiceInfo {
                        id: name.to_string(),
                        name: display_name.to_string(),
                        language: locale.to_string(),
                        gender,
                        quality: VoiceQuality::Neural, // Azure voices are typically neural
                        supports_ssml: true,           // Azure supports SSML
                        sample_rate: 16000,            // Default sample rate for Azure
                    });
                }
            }
        }

        info!("Retrieved {} Azure voices", voices.len());
        Ok(voices)
    }

    /// Synthesize using Google Cloud Text-to-Speech (placeholder)
    async fn synthesize_with_google(&self, _text: &str) -> Result<SynthesisResult> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Google Cloud Text-to-Speech not implemented yet".to_string(),
        ))
    }

    /// Get Google voices (placeholder)
    async fn get_google_voices(&self) -> Result<Vec<VoiceInfo>> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Google Cloud Text-to-Speech not implemented yet".to_string(),
        ))
    }

    /// Synthesize using Amazon Polly (placeholder)
    async fn synthesize_with_amazon(&self, _text: &str) -> Result<SynthesisResult> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Amazon Polly not implemented yet".to_string(),
        ))
    }

    /// Get Amazon voices (placeholder)
    async fn get_amazon_voices(&self) -> Result<Vec<VoiceInfo>> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Amazon Polly not implemented yet".to_string(),
        ))
    }

    /// Synthesize using local TTS engine (placeholder)
    async fn synthesize_with_local(&self, _text: &str) -> Result<SynthesisResult> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Local TTS engine not implemented yet".to_string(),
        ))
    }

    /// Get local voices (placeholder)
    async fn get_local_voices(&self) -> Result<Vec<VoiceInfo>> {
        Err(AgentError::tool(
            "synthesis".to_string(),
            "Local TTS engine not implemented yet".to_string(),
        ))
    }

    /// Check if a voice is available
    pub async fn is_voice_available(&self, voice_id: &str) -> Result<bool> {
        let voices = self.get_voices().await?;
        Ok(voices.iter().any(|v| v.id == voice_id))
    }

    /// Get voice information by ID
    pub async fn get_voice_info(&self, voice_id: &str) -> Result<Option<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices.into_iter().find(|v| v.id == voice_id))
    }

    /// Get voices by language
    pub async fn get_voices_by_language(&self, language: &str) -> Result<Vec<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices
            .into_iter()
            .filter(|v| v.language.starts_with(language))
            .collect())
    }

    /// Get voices by gender
    pub async fn get_voices_by_gender(&self, gender: VoiceGender) -> Result<Vec<VoiceInfo>> {
        let voices = self.get_voices().await?;
        Ok(voices
            .into_iter()
            .filter(|v| std::mem::discriminant(&v.gender) == std::mem::discriminant(&gender))
            .collect())
    }

    // ========================================
    // Azure Speech Services Helper Methods
    // ========================================

    /// Get Azure access token for Speech Services
    async fn get_azure_access_token(&self, subscription_key: &str, region: &str) -> Result<String> {
        let token_endpoint = format!(
            "https://{}.api.cognitive.microsoft.com/sts/v1.0/issuetoken",
            region
        );

        let response = self
            .client
            .post(&token_endpoint)
            .header("Ocp-Apim-Subscription-Key", subscription_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send()
            .await
            .map_err(|e| {
                AgentError::tool(
                    "synthesis".to_string(),
                    format!("Failed to get Azure token: {}", e),
                )
            })?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::tool(
                "synthesis".to_string(),
                format!("Azure token request failed: {}", error_text),
            ));
        }

        let token = response.text().await.map_err(|e| {
            AgentError::tool(
                "synthesis".to_string(),
                format!("Failed to read Azure token: {}", e),
            )
        })?;

        Ok(token)
    }

    /// Create SSML for Azure TTS
    async fn create_azure_ssml(&self, text: &str, voice: &str) -> Result<String> {
        // Create SSML with proper voice and language settings
        let ssml = format!(
            r#"<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{}">
                    <prosody rate="{:.1}">
                        {}
                    </prosody>
                </voice>
            </speak>"#,
            voice,
            self.config.speed,
            self.escape_ssml_text(text)
        );

        Ok(ssml)
    }

    /// Escape text for SSML
    fn escape_ssml_text(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Estimate audio duration based on text and speech rate
    pub fn estimate_audio_duration(&self, text: &str, speed: f32) -> f64 {
        // Average speaking rate is about 150 words per minute
        let base_wpm = 150.0f64;
        let adjusted_wpm = base_wpm * speed as f64;
        let word_count = text.split_whitespace().count() as f64;
        let duration_minutes = word_count / adjusted_wpm;
        duration_minutes * 60.0 // Convert to seconds
    }
}

/// Create a synthesis service with default configuration
pub fn create_default_synthesis_service() -> Result<SynthesisService> {
    let config = SynthesisConfig::default();
    SynthesisService::new(config)
}

/// Create a synthesis service with OpenAI TTS
pub fn create_openai_synthesis_service(api_key: String) -> Result<SynthesisService> {
    let mut config = SynthesisConfig::default();
    config.provider = "openai".to_string();
    config.api_key = Some(api_key);
    config.voice = "alloy".to_string();
    SynthesisService::new(config)
}

/// Utility function to estimate speech duration
pub fn estimate_speech_duration(text: &str, words_per_minute: f32) -> Duration {
    let word_count = text.split_whitespace().count() as f32;
    let minutes = word_count / words_per_minute;
    Duration::from_secs_f32(minutes * 60.0)
}

/// Utility function to split long text into chunks
pub fn split_text_for_synthesis(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for sentence in text.split('.') {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        let sentence_with_period = format!("{}.", sentence);

        if current_chunk.len() + sentence_with_period.len() > max_chunk_size {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }

            // If single sentence is too long, split by words
            if sentence_with_period.len() > max_chunk_size {
                let words: Vec<&str> = sentence_with_period.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    if word_chunk.len() + word.len() + 1 > max_chunk_size && !word_chunk.is_empty()
                    {
                        chunks.push(word_chunk.trim().to_string());
                        word_chunk.clear();
                    }

                    if !word_chunk.is_empty() {
                        word_chunk.push(' ');
                    }
                    word_chunk.push_str(word);
                }

                if !word_chunk.is_empty() {
                    current_chunk = word_chunk;
                }
            } else {
                current_chunk = sentence_with_period;
            }
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&sentence_with_period);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}
