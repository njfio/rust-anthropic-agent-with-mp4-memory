// Advanced Audio Processing System for Enterprise AI Agent
// Provides comprehensive audio format support, real-time streaming, transcription, and synthesis

use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub mod codecs;
pub mod effects;
pub mod metadata;
pub mod streaming;
pub mod synthesis;
pub mod transcription;
pub mod tool;

#[cfg(test)]
mod tests;

/// Audio format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Aac,
    M4a,
}

impl AudioFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Flac => "flac",
            AudioFormat::Ogg => "ogg",
            AudioFormat::Aac => "aac",
            AudioFormat::M4a => "m4a",
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(AudioFormat::Wav),
            "mp3" => Some(AudioFormat::Mp3),
            "flac" => Some(AudioFormat::Flac),
            "ogg" => Some(AudioFormat::Ogg),
            "aac" => Some(AudioFormat::Aac),
            "m4a" => Some(AudioFormat::M4a),
            _ => None,
        }
    }

    /// Check if format supports metadata
    pub fn supports_metadata(&self) -> bool {
        matches!(self, AudioFormat::Mp3 | AudioFormat::Flac | AudioFormat::M4a)
    }
}

/// Audio quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQuality {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
    pub bitrate: Option<u32>, // For compressed formats
}

impl Default for AudioQuality {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
            bitrate: Some(128000), // 128 kbps
        }
    }
}

impl AudioQuality {
    /// High quality preset (48kHz, 24-bit, stereo)
    pub fn high() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
            bitrate: Some(320000), // 320 kbps
        }
    }

    /// Low quality preset (22kHz, 16-bit, mono)
    pub fn low() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bit_depth: 16,
            bitrate: Some(64000), // 64 kbps
        }
    }

    /// Voice quality preset (16kHz, 16-bit, mono)
    pub fn voice() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
            bitrate: Some(32000), // 32 kbps
        }
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Maximum file size in bytes (default: 100MB)
    pub max_file_size: usize,
    /// Maximum processing duration in seconds (default: 300s = 5 minutes)
    pub max_processing_duration: u64,
    /// Default audio quality settings
    pub default_quality: AudioQuality,
    /// Enable caching of processed audio
    pub enable_caching: bool,
    /// Cache TTL for processed audio (default: 1 hour)
    pub cache_ttl: Duration,
    /// Enable resource monitoring
    pub enable_monitoring: bool,
    /// Transcription service configuration
    pub transcription: TranscriptionConfig,
    /// Text-to-speech configuration
    pub synthesis: SynthesisConfig,
    /// Audio effects configuration
    pub effects: EffectsConfig,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_processing_duration: 300,     // 5 minutes
            default_quality: AudioQuality::default(),
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            enable_monitoring: true,
            transcription: TranscriptionConfig::default(),
            synthesis: SynthesisConfig::default(),
            effects: EffectsConfig::default(),
        }
    }
}

/// Transcription service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Service provider (whisper, azure, google, etc.)
    pub provider: String,
    /// API endpoint URL
    pub endpoint: Option<String>,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Model to use (e.g., "whisper-1", "base", "small", etc.)
    pub model: String,
    /// Language code (e.g., "en", "es", "fr")
    pub language: Option<String>,
    /// Enable automatic language detection
    pub auto_detect_language: bool,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            provider: "whisper".to_string(),
            endpoint: None,
            api_key: None,
            model: "base".to_string(),
            language: Some("en".to_string()),
            auto_detect_language: true,
            timeout: 60,
            max_retries: 3,
        }
    }
}

/// Text-to-speech configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// TTS provider (openai, azure, google, etc.)
    pub provider: String,
    /// API endpoint URL
    pub endpoint: Option<String>,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Voice model to use
    pub voice: String,
    /// Speech speed (0.25 to 4.0)
    pub speed: f32,
    /// Audio format for output
    pub output_format: AudioFormat,
    /// Request timeout in seconds
    pub timeout: u64,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            endpoint: None,
            api_key: None,
            voice: "alloy".to_string(),
            speed: 1.0,
            output_format: AudioFormat::Mp3,
            timeout: 60,
        }
    }
}

/// Audio effects configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsConfig {
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Noise reduction strength (0.0 to 1.0)
    pub noise_reduction_strength: f32,
    /// Enable audio normalization
    pub enable_normalization: bool,
    /// Target normalization level in dB
    pub normalization_target: f32,
    /// Enable high-pass filter
    pub enable_highpass_filter: bool,
    /// High-pass filter cutoff frequency in Hz
    pub highpass_cutoff: f32,
    /// Enable low-pass filter
    pub enable_lowpass_filter: bool,
    /// Low-pass filter cutoff frequency in Hz
    pub lowpass_cutoff: f32,
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            enable_noise_reduction: false,
            noise_reduction_strength: 0.5,
            enable_normalization: false,
            normalization_target: -23.0, // LUFS standard
            enable_highpass_filter: false,
            highpass_cutoff: 80.0,
            enable_lowpass_filter: false,
            lowpass_cutoff: 8000.0,
        }
    }
}

/// Audio processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStats {
    /// Total files processed
    pub files_processed: u64,
    /// Total processing time in milliseconds
    pub total_processing_time: u64,
    /// Average processing time per file
    pub average_processing_time: f64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Number of transcription requests
    pub transcription_requests: u64,
    /// Number of synthesis requests
    pub synthesis_requests: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for AudioStats {
    fn default() -> Self {
        Self {
            files_processed: 0,
            total_processing_time: 0,
            average_processing_time: 0.0,
            bytes_processed: 0,
            transcription_requests: 0,
            synthesis_requests: 0,
            failed_operations: 0,
            cache_hit_rate: 0.0,
            last_updated: Utc::now(),
        }
    }
}

/// Main audio processing manager
pub struct AudioProcessor {
    /// Configuration
    config: AudioConfig,
    /// Processing statistics
    stats: Arc<RwLock<AudioStats>>,
    /// Cache manager reference
    cache_manager: Option<Arc<crate::caching::CacheManager>>,
    /// Resource monitor reference
    resource_monitor: Option<Arc<crate::utils::resource_monitor::ResourceMonitor>>,
}

impl std::fmt::Debug for AudioProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioProcessor")
            .field("config", &self.config)
            .field("stats", &"RwLock<AudioStats>")
            .field("cache_manager", &self.cache_manager.as_ref().map(|_| "Some(CacheManager)"))
            .field("resource_monitor", &self.resource_monitor.as_ref().map(|_| "Some(ResourceMonitor)"))
            .finish()
    }
}

impl AudioProcessor {
    /// Create a new audio processor
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(AudioStats::default())),
            cache_manager: None,
            resource_monitor: None,
        }
    }

    /// Set cache manager for caching processed audio
    pub fn with_cache_manager(mut self, cache_manager: Arc<crate::caching::CacheManager>) -> Self {
        self.cache_manager = Some(cache_manager);
        self
    }

    /// Set resource monitor for tracking processing resources
    pub fn with_resource_monitor(
        mut self,
        resource_monitor: Arc<crate::utils::resource_monitor::ResourceMonitor>,
    ) -> Self {
        self.resource_monitor = Some(resource_monitor);
        self
    }

    /// Get current processing statistics
    pub async fn get_stats(&self) -> AudioStats {
        self.stats.read().await.clone()
    }

    /// Reset processing statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = AudioStats::default();
    }

    /// Validate audio file before processing
    pub fn validate_audio_file(&self, file_path: &Path, file_size: usize) -> Result<()> {
        // Check file size
        if file_size > self.config.max_file_size {
            return Err(AgentError::invalid_input(format!(
                "Audio file too large: {} bytes (max: {} bytes)",
                file_size, self.config.max_file_size
            )));
        }

        // Check file extension
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        if AudioFormat::from_extension(extension).is_none() {
            return Err(AgentError::invalid_input(format!(
                "Unsupported audio format: {}",
                extension
            )));
        }

        // Additional security checks
        crate::utils::validation::validate_path(file_path.to_str().unwrap())?;

        Ok(())
    }
}
