// Audio Metadata Extraction and Management
// Provides comprehensive metadata reading and writing for various audio formats

use super::AudioFormat;
use crate::utils::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, StandardTagKey, Tag, Value};
use symphonia::core::probe::Hint;
use tracing::{debug, error, info, warn};

/// Audio metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Basic audio properties
    pub format: AudioFormat,
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: Option<u16>,
    pub bitrate: Option<u32>,
    pub file_size: u64,

    /// Standard tags
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub album_artist: Option<String>,
    pub track_number: Option<u32>,
    pub track_total: Option<u32>,
    pub disc_number: Option<u32>,
    pub disc_total: Option<u32>,
    pub date: Option<String>,
    pub year: Option<u32>,
    pub genre: Option<String>,
    pub composer: Option<String>,
    pub comment: Option<String>,

    /// Technical metadata
    pub encoder: Option<String>,
    pub encoding_settings: Option<String>,
    pub copyright: Option<String>,
    pub isrc: Option<String>,
    pub musicbrainz_track_id: Option<String>,
    pub musicbrainz_album_id: Option<String>,
    pub musicbrainz_artist_id: Option<String>,

    /// Custom tags
    pub custom_tags: HashMap<String, String>,

    /// Embedded artwork information
    pub artwork: Option<ArtworkInfo>,

    /// Technical analysis
    pub peak_amplitude: Option<f32>,
    pub rms_level: Option<f32>,
    pub dynamic_range: Option<f32>,
    pub replay_gain_track: Option<f32>,
    pub replay_gain_album: Option<f32>,
}

impl Default for AudioMetadata {
    fn default() -> Self {
        Self {
            format: AudioFormat::Wav,
            duration_seconds: 0.0,
            sample_rate: 44100,
            channels: 2,
            bit_depth: None,
            bitrate: None,
            file_size: 0,
            title: None,
            artist: None,
            album: None,
            album_artist: None,
            track_number: None,
            track_total: None,
            disc_number: None,
            disc_total: None,
            date: None,
            year: None,
            genre: None,
            composer: None,
            comment: None,
            encoder: None,
            encoding_settings: None,
            copyright: None,
            isrc: None,
            musicbrainz_track_id: None,
            musicbrainz_album_id: None,
            musicbrainz_artist_id: None,
            custom_tags: HashMap::new(),
            artwork: None,
            peak_amplitude: None,
            rms_level: None,
            dynamic_range: None,
            replay_gain_track: None,
            replay_gain_album: None,
        }
    }
}

/// Embedded artwork information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtworkInfo {
    /// MIME type of the artwork
    pub mime_type: String,
    /// Artwork data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: Option<u32>,
    /// Height in pixels
    pub height: Option<u32>,
    /// Color depth
    pub color_depth: Option<u8>,
    /// Artwork description
    pub description: Option<String>,
}

/// Metadata extraction service
#[derive(Debug)]
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract metadata from audio file using instance method
    pub async fn extract_metadata(&self, file_path: &str) -> Result<AudioMetadata> {
        Self::extract_from_file(file_path)
    }

    /// Extract metadata from audio data buffer using instance method
    pub async fn extract_from_buffer(
        &self,
        audio_data: &super::codecs::AudioData,
    ) -> Result<AudioMetadata> {
        // Create metadata from audio data buffer
        let mut metadata = AudioMetadata::default();

        // Extract technical information from audio data
        metadata.duration_seconds = audio_data.duration;
        metadata.sample_rate = audio_data.sample_rate as u32;
        metadata.channels = audio_data.channels as u16;
        metadata.format = audio_data.format.clone();

        // Calculate additional technical metadata
        let _total_samples = audio_data.samples.len();
        let bits_per_sample = 16; // Default assumption
        let byte_rate =
            audio_data.sample_rate as u32 * audio_data.channels as u32 * (bits_per_sample / 8);
        metadata.bitrate = Some(byte_rate);

        // Set extraction timestamp (using comment as a proxy for extraction info)
        metadata.comment = Some(format!(
            "Extracted from buffer at {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
        ));

        debug!(
            "Extracted metadata from audio buffer: {}s, {}Hz, {} channels",
            audio_data.duration, audio_data.sample_rate, audio_data.channels
        );

        Ok(metadata)
    }

    /// Extract metadata from audio file
    pub fn extract_from_file<P: AsRef<Path>>(path: P) -> Result<AudioMetadata> {
        let path = path.as_ref();
        debug!("Extracting metadata from: {}", path.display());

        // Get file size
        let file_size = std::fs::metadata(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to get file metadata: {}", e)))?
            .len();

        // Detect format from extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        let format = AudioFormat::from_extension(extension).ok_or_else(|| {
            AgentError::invalid_input(format!("Unsupported format: {}", extension))
        })?;

        // Extract metadata based on format
        match format {
            AudioFormat::Wav => Self::extract_wav_metadata(path, file_size),
            AudioFormat::Mp3
            | AudioFormat::Flac
            | AudioFormat::Ogg
            | AudioFormat::Aac
            | AudioFormat::M4a => Self::extract_metadata_with_symphonia(path, format, file_size),
        }
    }

    /// Extract metadata from WAV file
    fn extract_wav_metadata<P: AsRef<Path>>(path: P, file_size: u64) -> Result<AudioMetadata> {
        let reader = hound::WavReader::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open WAV file: {}", e)))?;

        let spec = reader.spec();
        let duration = reader.duration() as f64 / spec.sample_rate as f64;

        let metadata = AudioMetadata {
            format: AudioFormat::Wav,
            duration_seconds: duration,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bit_depth: Some(spec.bits_per_sample),
            bitrate: Some(spec.sample_rate * spec.channels as u32 * spec.bits_per_sample as u32),
            file_size,
            ..Default::default()
        };

        // WAV files can have INFO chunks with metadata, but hound doesn't support them
        // For now, we'll just return the basic technical metadata
        info!(
            "Extracted WAV metadata: {}s, {}Hz, {} channels, {} bits",
            duration, spec.sample_rate, spec.channels, spec.bits_per_sample
        );

        Ok(metadata)
    }

    /// Extract metadata using Symphonia
    fn extract_metadata_with_symphonia<P: AsRef<Path>>(
        path: P,
        format: AudioFormat,
        file_size: u64,
    ) -> Result<AudioMetadata> {
        let file = File::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open file: {}", e)))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(format.extension());

        let meta_opts = MetadataOptions::default();
        let fmt_opts = symphonia::core::formats::FormatOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to probe format: {}", e)))?;

        let mut format_reader = probed.format;
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| AgentError::invalid_input("No supported audio tracks found"))?;

        // Extract basic technical metadata
        let mut metadata = AudioMetadata {
            format,
            sample_rate: track.codec_params.sample_rate.unwrap_or(44100),
            channels: track
                .codec_params
                .channels
                .map(|c| c.count() as u16)
                .unwrap_or(2),
            bit_depth: track.codec_params.bits_per_sample.map(|b| b as u16),
            bitrate: None, // Bitrate not available in codec params
            file_size,
            ..Default::default()
        };

        // Calculate duration if available
        if let (Some(n_frames), Some(sample_rate)) =
            (track.codec_params.n_frames, track.codec_params.sample_rate)
        {
            metadata.duration_seconds = n_frames as f64 / sample_rate as f64;
        }

        // Extract metadata tags
        if let Some(metadata_rev) = format_reader.metadata().current() {
            for tag in metadata_rev.tags() {
                Self::process_tag(&mut metadata, tag);
            }

            // Extract artwork
            for visual in metadata_rev.visuals() {
                if metadata.artwork.is_none() {
                    metadata.artwork = Some(ArtworkInfo {
                        mime_type: visual.media_type.clone(),
                        data: visual.data.to_vec(),
                        width: visual.dimensions.map(|d| d.width),
                        height: visual.dimensions.map(|d| d.height),
                        color_depth: visual.bits_per_pixel.map(|b| b.get() as u8),
                        description: visual.usage.as_ref().map(|u| format!("{:?}", u)),
                    });
                }
            }
        }

        info!(
            "Extracted metadata: {}s, {}Hz, {} channels",
            metadata.duration_seconds, metadata.sample_rate, metadata.channels
        );

        Ok(metadata)
    }

    /// Process a metadata tag
    fn process_tag(metadata: &mut AudioMetadata, tag: &Tag) {
        if let Some(std_key) = tag.std_key {
            match std_key {
                StandardTagKey::TrackTitle => {
                    metadata.title = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::Artist => {
                    metadata.artist = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::Album => {
                    metadata.album = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::AlbumArtist => {
                    metadata.album_artist = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::TrackNumber => {
                    metadata.track_number = Self::extract_uint_value(&tag.value);
                }
                StandardTagKey::TrackTotal => {
                    metadata.track_total = Self::extract_uint_value(&tag.value);
                }
                StandardTagKey::DiscNumber => {
                    metadata.disc_number = Self::extract_uint_value(&tag.value);
                }
                StandardTagKey::DiscTotal => {
                    metadata.disc_total = Self::extract_uint_value(&tag.value);
                }
                StandardTagKey::Date => {
                    metadata.date = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::OriginalDate => {
                    if metadata.date.is_none() {
                        metadata.date = Self::extract_string_value(&tag.value);
                    }
                }
                StandardTagKey::Genre => {
                    metadata.genre = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::Composer => {
                    metadata.composer = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::Comment => {
                    metadata.comment = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::Encoder => {
                    metadata.encoder = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::EncodedBy => {
                    if metadata.encoder.is_none() {
                        metadata.encoder = Self::extract_string_value(&tag.value);
                    }
                }
                StandardTagKey::Copyright => {
                    metadata.copyright = Self::extract_string_value(&tag.value);
                }
                // StandardTagKey::Isrc => {
                //     metadata.isrc = Self::extract_string_value(&tag.value);
                // }
                StandardTagKey::MusicBrainzTrackId => {
                    metadata.musicbrainz_track_id = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::MusicBrainzAlbumId => {
                    metadata.musicbrainz_album_id = Self::extract_string_value(&tag.value);
                }
                StandardTagKey::MusicBrainzArtistId => {
                    metadata.musicbrainz_artist_id = Self::extract_string_value(&tag.value);
                }
                _ => {
                    // Store other standard tags as custom tags
                    if let Some(value) = Self::extract_string_value(&tag.value) {
                        metadata.custom_tags.insert(format!("{:?}", std_key), value);
                    }
                }
            }
        } else {
            // Store non-standard tags as custom tags
            let key = &tag.key;
            if let Some(value) = Self::extract_string_value(&tag.value) {
                metadata.custom_tags.insert(key.clone(), value);
            }
        }

        // Extract year from date if available
        if let Some(ref date) = metadata.date {
            if let Ok(year) = date[..4.min(date.len())].parse::<u32>() {
                metadata.year = Some(year);
            }
        }
    }

    /// Extract string value from metadata value
    fn extract_string_value(value: &Value) -> Option<String> {
        match value {
            Value::String(s) => Some(s.clone()),
            Value::Binary(b) => String::from_utf8(b.to_vec()).ok(),
            Value::UnsignedInt(u) => Some(u.to_string()),
            Value::SignedInt(i) => Some(i.to_string()),
            Value::Float(f) => Some(f.to_string()),
            Value::Flag => Some("true".to_string()),
            Value::Boolean(b) => Some(b.to_string()),
        }
    }

    /// Extract unsigned integer value from metadata value
    fn extract_uint_value(value: &Value) -> Option<u32> {
        match value {
            Value::UnsignedInt(u) => Some(*u as u32),
            Value::SignedInt(i) => Some(*i as u32),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Get a summary of the metadata
    pub fn get_metadata_summary(metadata: &AudioMetadata) -> String {
        let mut summary = Vec::new();

        // Basic info
        summary.push(format!("Format: {:?}", metadata.format));
        summary.push(format!("Duration: {:.2}s", metadata.duration_seconds));
        summary.push(format!("Sample Rate: {}Hz", metadata.sample_rate));
        summary.push(format!("Channels: {}", metadata.channels));

        if let Some(bit_depth) = metadata.bit_depth {
            summary.push(format!("Bit Depth: {} bits", bit_depth));
        }

        if let Some(bitrate) = metadata.bitrate {
            summary.push(format!("Bitrate: {} bps", bitrate));
        }

        summary.push(format!("File Size: {} bytes", metadata.file_size));

        // Tags
        if let Some(ref title) = metadata.title {
            summary.push(format!("Title: {}", title));
        }

        if let Some(ref artist) = metadata.artist {
            summary.push(format!("Artist: {}", artist));
        }

        if let Some(ref album) = metadata.album {
            summary.push(format!("Album: {}", album));
        }

        if let Some(year) = metadata.year {
            summary.push(format!("Year: {}", year));
        }

        if let Some(ref genre) = metadata.genre {
            summary.push(format!("Genre: {}", genre));
        }

        if metadata.artwork.is_some() {
            summary.push("Has embedded artwork".to_string());
        }

        if !metadata.custom_tags.is_empty() {
            summary.push(format!("Custom tags: {}", metadata.custom_tags.len()));
        }

        summary.join("\n")
    }

    /// Check if metadata contains specific information
    pub fn has_complete_metadata(metadata: &AudioMetadata) -> bool {
        metadata.title.is_some()
            && metadata.artist.is_some()
            && metadata.album.is_some()
            && metadata.year.is_some()
    }

    /// Validate metadata for completeness
    pub fn validate_metadata(metadata: &AudioMetadata) -> Vec<String> {
        let mut issues = Vec::new();

        if metadata.title.is_none() {
            issues.push("Missing title".to_string());
        }

        if metadata.artist.is_none() {
            issues.push("Missing artist".to_string());
        }

        if metadata.album.is_none() {
            issues.push("Missing album".to_string());
        }

        if metadata.year.is_none() && metadata.date.is_none() {
            issues.push("Missing date/year".to_string());
        }

        if metadata.duration_seconds <= 0.0 {
            issues.push("Invalid duration".to_string());
        }

        if metadata.sample_rate == 0 {
            issues.push("Invalid sample rate".to_string());
        }

        if metadata.channels == 0 {
            issues.push("Invalid channel count".to_string());
        }

        issues
    }
}

/// Create a metadata extractor
pub fn create_metadata_extractor() -> MetadataExtractor {
    MetadataExtractor
}

/// Extract basic metadata from audio data
pub fn extract_basic_metadata_from_audio(audio: &super::codecs::AudioData) -> AudioMetadata {
    AudioMetadata {
        format: audio.format,
        duration_seconds: audio.duration,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        bit_depth: Some(32), // Assuming f32 samples
        bitrate: Some(audio.sample_rate * audio.channels as u32 * 32),
        file_size: (audio.samples.len() * 4) as u64, // f32 = 4 bytes
        ..Default::default()
    }
}
