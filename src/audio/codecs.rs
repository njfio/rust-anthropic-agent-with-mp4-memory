// Audio Codec Support for Multiple Formats
// Provides encoding/decoding for WAV, MP3, FLAC, OGG with proper error handling

use super::{AudioFormat, AudioQuality};
use crate::utils::error::{AgentError, Result};
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use symphonia::core::audio::{AudioBuffer, AudioBufferRef, SampleBuffer, SignalSpec};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{debug, error, info, warn};

/// Audio sample data
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw audio samples (interleaved for multi-channel)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: f64,
    /// Original format
    pub format: AudioFormat,
}

impl AudioData {
    /// Create new audio data
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16, format: AudioFormat) -> Self {
        let duration = samples.len() as f64 / (sample_rate as f64 * channels as f64);
        Self {
            samples,
            sample_rate,
            channels,
            duration,
            format,
        }
    }

    /// Get number of frames (samples per channel)
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Convert to mono by averaging channels
    pub fn to_mono(&self) -> AudioData {
        if self.channels == 1 {
            return self.clone();
        }

        let mut mono_samples = Vec::with_capacity(self.frames());
        for frame in self.samples.chunks(self.channels as usize) {
            let sum: f32 = frame.iter().sum();
            mono_samples.push(sum / self.channels as f32);
        }

        AudioData::new(mono_samples, self.sample_rate, 1, self.format)
    }

    /// Resample to target sample rate using linear interpolation
    pub fn resample(&self, target_sample_rate: u32) -> Result<AudioData> {
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }

        let ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let target_frames = (self.frames() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(target_frames * self.channels as usize);

        for target_frame in 0..target_frames {
            let source_frame = target_frame as f64 / ratio;
            let source_frame_floor = source_frame.floor() as usize;
            let source_frame_ceil = (source_frame_floor + 1).min(self.frames() - 1);
            let fraction = source_frame - source_frame_floor as f64;

            for channel in 0..self.channels as usize {
                let sample_floor =
                    self.samples[source_frame_floor * self.channels as usize + channel];
                let sample_ceil =
                    self.samples[source_frame_ceil * self.channels as usize + channel];
                let interpolated = sample_floor + (sample_ceil - sample_floor) * fraction as f32;
                resampled.push(interpolated);
            }
        }

        Ok(AudioData::new(
            resampled,
            target_sample_rate,
            self.channels,
            self.format,
        ))
    }

    /// Normalize audio to target peak level
    pub fn normalize(&self, target_peak: f32) -> AudioData {
        let current_peak = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if current_peak == 0.0 {
            return self.clone();
        }

        let gain = target_peak / current_peak;
        let normalized_samples: Vec<f32> = self.samples.iter().map(|s| s * gain).collect();

        AudioData::new(
            normalized_samples,
            self.sample_rate,
            self.channels,
            self.format,
        )
    }
}

/// Audio codec for reading and writing various formats
pub struct AudioCodec;

impl AudioCodec {
    /// Decode audio file to AudioData
    pub fn decode_file<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let path = path.as_ref();
        debug!("Decoding audio file: {}", path.display());

        // Detect format from extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        let format = AudioFormat::from_extension(extension).ok_or_else(|| {
            AgentError::invalid_input(format!("Unsupported format: {}", extension))
        })?;

        match format {
            AudioFormat::Wav => Self::decode_wav(path),
            AudioFormat::Mp3 => Self::decode_with_symphonia(path, format),
            AudioFormat::Flac => Self::decode_with_symphonia(path, format),
            AudioFormat::Ogg => Self::decode_with_symphonia(path, format),
            AudioFormat::Aac | AudioFormat::M4a => Self::decode_with_symphonia(path, format),
        }
    }

    /// Decode audio from byte buffer
    pub fn decode_bytes(data: &[u8], format: AudioFormat) -> Result<AudioData> {
        debug!(
            "Decoding audio from {} bytes, format: {:?}",
            data.len(),
            format
        );

        match format {
            AudioFormat::Wav => Self::decode_wav_bytes(data),
            AudioFormat::Mp3 => Self::decode_mp3_bytes(data),
            _ => Self::decode_bytes_with_symphonia(data, format),
        }
    }

    /// Encode AudioData to file
    pub fn encode_file<P: AsRef<Path>>(
        audio: &AudioData,
        path: P,
        quality: &AudioQuality,
    ) -> Result<()> {
        let path = path.as_ref();
        debug!("Encoding audio to file: {}", path.display());

        // Detect format from extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AgentError::invalid_input("Invalid file extension"))?;

        let format = AudioFormat::from_extension(extension).ok_or_else(|| {
            AgentError::invalid_input(format!("Unsupported format: {}", extension))
        })?;

        match format {
            AudioFormat::Wav => Self::encode_wav(audio, path, quality),
            _ => Err(AgentError::invalid_input(format!(
                "Encoding not supported for format: {:?}",
                format
            ))),
        }
    }

    /// Encode AudioData to bytes
    pub fn encode_bytes(
        audio: &AudioData,
        format: AudioFormat,
        quality: &AudioQuality,
    ) -> Result<Vec<u8>> {
        debug!("Encoding audio to bytes, format: {:?}", format);

        match format {
            AudioFormat::Wav => Self::encode_wav_bytes(audio, quality),
            _ => Err(AgentError::invalid_input(format!(
                "Encoding not supported for format: {:?}",
                format
            ))),
        }
    }

    /// Decode WAV file using hound
    fn decode_wav<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open WAV file: {}", e)))?;

        let spec = reader.spec();
        let samples: Result<Vec<f32>> = match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            24 => reader
                .samples::<i32>()
                .map(|s| s.map(|sample| sample as f32 / (1 << 23) as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            32 => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| AgentError::invalid_input(format!("Failed to read samples: {}", e))),
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    spec.bits_per_sample
                )))
            }
        };

        let samples = samples?;
        Ok(AudioData::new(
            samples,
            spec.sample_rate,
            spec.channels,
            AudioFormat::Wav,
        ))
    }

    /// Decode WAV from bytes
    fn decode_wav_bytes(data: &[u8]) -> Result<AudioData> {
        let cursor = Cursor::new(data);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| AgentError::invalid_input(format!("Failed to read WAV data: {}", e)))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
                .collect(),
            24 => reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / (1 << 23) as f32)
                .collect(),
            32 => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    spec.bits_per_sample
                )))
            }
        };

        Ok(AudioData::new(
            samples,
            spec.sample_rate,
            spec.channels,
            AudioFormat::Wav,
        ))
    }

    /// Decode MP3 from bytes using minimp3
    fn decode_mp3_bytes(data: &[u8]) -> Result<AudioData> {
        let mut decoder = minimp3::Decoder::new(data);
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    if sample_rate == 0 {
                        sample_rate = frame.sample_rate as u32;
                        channels = frame.channels as u16;
                    }
                    // Convert i16 samples to f32
                    for sample in frame.data {
                        samples.push(sample as f32 / i16::MAX as f32);
                    }
                }
                Err(minimp3::Error::Eof) => break,
                Err(e) => {
                    return Err(AgentError::invalid_input(format!(
                        "MP3 decode error: {:?}",
                        e
                    )))
                }
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data found in MP3"));
        }

        Ok(AudioData::new(
            samples,
            sample_rate,
            channels,
            AudioFormat::Mp3,
        ))
    }

    /// Decode using Symphonia for various formats
    fn decode_with_symphonia<P: AsRef<Path>>(
        path: P,
        audio_format: AudioFormat,
    ) -> Result<AudioData> {
        let file = File::open(path)
            .map_err(|e| AgentError::invalid_input(format!("Failed to open file: {}", e)))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(audio_format.extension());

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to probe format: {}", e)))?;

        let mut format = probed.format;
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| AgentError::invalid_input("No supported audio tracks found"))?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => {
                    // Reset decoder and continue
                    decoder.reset();
                    continue;
                }
                Err(SymphoniaError::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => return Err(AgentError::invalid_input(format!("Decode error: {}", e))),
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    if sample_rate == 0 {
                        let spec = *decoded.spec();
                        sample_rate = spec.rate;
                        channels = spec.channels.count() as u16;
                    }

                    // Convert to f32 samples
                    let mut sample_buf =
                        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    sample_buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(sample_buf.samples());
                }
                Err(SymphoniaError::IoError(_)) => break,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(e) => return Err(AgentError::invalid_input(format!("Decode error: {}", e))),
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data decoded"));
        }

        Ok(AudioData::new(samples, sample_rate, channels, audio_format))
    }

    /// Decode bytes using Symphonia
    fn decode_bytes_with_symphonia(data: &[u8], audio_format: AudioFormat) -> Result<AudioData> {
        let data_vec = data.to_vec();
        let cursor = Cursor::new(data_vec);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(audio_format.extension());

        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to probe format: {}", e)))?;

        let mut format_reader = probed.format;
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| AgentError::invalid_input("No supported audio tracks found"))?;

        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| AgentError::invalid_input(format!("Failed to create decoder: {}", e)))?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        loop {
            let packet = match format_reader.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => {
                    decoder.reset();
                    continue;
                }
                Err(SymphoniaError::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(_) => break,
            };

            if packet.track_id() != track_id {
                continue;
            }

            if let Ok(decoded) = decoder.decode(&packet) {
                if sample_rate == 0 {
                    let spec = *decoded.spec();
                    sample_rate = spec.rate;
                    channels = spec.channels.count() as u16;
                }

                let mut sample_buf =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sample_buf.samples());
            }
        }

        if samples.is_empty() {
            return Err(AgentError::invalid_input("No audio data decoded"));
        }

        Ok(AudioData::new(samples, sample_rate, channels, audio_format))
    }

    /// Encode WAV file using hound
    fn encode_wav<P: AsRef<Path>>(
        audio: &AudioData,
        path: P,
        quality: &AudioQuality,
    ) -> Result<()> {
        let spec = hound::WavSpec {
            channels: quality.channels,
            sample_rate: quality.sample_rate,
            bits_per_sample: quality.bit_depth,
            sample_format: if quality.bit_depth == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
            AgentError::invalid_input(format!("Failed to create WAV writer: {}", e))
        })?;

        // Resample if necessary
        let resampled_audio = if audio.sample_rate != quality.sample_rate {
            audio.resample(quality.sample_rate)?
        } else {
            audio.clone()
        };

        // Write samples based on bit depth
        match quality.bit_depth {
            16 => {
                for sample in &resampled_audio.samples {
                    let sample_i16 = (*sample * i16::MAX as f32) as i16;
                    writer.write_sample(sample_i16).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            24 => {
                for sample in &resampled_audio.samples {
                    let sample_i32 = (*sample * (1 << 23) as f32) as i32;
                    writer.write_sample(sample_i32).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            32 => {
                for sample in &resampled_audio.samples {
                    writer.write_sample(*sample).map_err(|e| {
                        AgentError::invalid_input(format!("Failed to write sample: {}", e))
                    })?;
                }
            }
            _ => {
                return Err(AgentError::invalid_input(format!(
                    "Unsupported bit depth: {}",
                    quality.bit_depth
                )))
            }
        }

        writer.finalize().map_err(|e| {
            AgentError::invalid_input(format!("Failed to finalize WAV file: {}", e))
        })?;

        Ok(())
    }

    /// Encode WAV to bytes
    fn encode_wav_bytes(audio: &AudioData, quality: &AudioQuality) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: quality.channels,
            sample_rate: quality.sample_rate,
            bits_per_sample: quality.bit_depth,
            sample_format: if quality.bit_depth == 32 {
                hound::SampleFormat::Float
            } else {
                hound::SampleFormat::Int
            },
        };

        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = hound::WavWriter::new(cursor, spec).map_err(|e| {
                AgentError::invalid_input(format!("Failed to create WAV writer: {}", e))
            })?;

            // Resample if necessary
            let resampled_audio = if audio.sample_rate != quality.sample_rate {
                audio.resample(quality.sample_rate)?
            } else {
                audio.clone()
            };

            // Write samples based on bit depth
            match quality.bit_depth {
                16 => {
                    for sample in &resampled_audio.samples {
                        let sample_i16 = (*sample * i16::MAX as f32) as i16;
                        writer.write_sample(sample_i16).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                24 => {
                    for sample in &resampled_audio.samples {
                        let sample_i32 = (*sample * (1 << 23) as f32) as i32;
                        writer.write_sample(sample_i32).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                32 => {
                    for sample in &resampled_audio.samples {
                        writer.write_sample(*sample).map_err(|e| {
                            AgentError::invalid_input(format!("Failed to write sample: {}", e))
                        })?;
                    }
                }
                _ => {
                    return Err(AgentError::invalid_input(format!(
                        "Unsupported bit depth: {}",
                        quality.bit_depth
                    )))
                }
            }

            writer
                .finalize()
                .map_err(|e| AgentError::invalid_input(format!("Failed to finalize WAV: {}", e)))?;
        }

        Ok(buffer)
    }
}
