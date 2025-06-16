// Audio Effects Processing
// Provides noise reduction, normalization, filtering, and other audio effects

use super::{codecs::AudioData, EffectsConfig};
use crate::utils::error::{AgentError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use tracing::{debug, info, warn};

/// Audio effects processor
pub struct AudioEffects {
    /// Effects configuration
    config: EffectsConfig,
    /// FFT planner for frequency domain processing
    fft_planner: FftPlanner<f32>,
}

impl std::fmt::Debug for AudioEffects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEffects")
            .field("config", &self.config)
            .field("fft_planner", &"FftPlanner<f32>")
            .finish()
    }
}

impl AudioEffects {
    /// Create a new audio effects processor
    pub fn new(config: EffectsConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Apply all configured effects to audio data
    pub fn process(&mut self, audio: &AudioData) -> Result<AudioData> {
        let mut processed = audio.clone();

        info!(
            "Applying audio effects to {} samples",
            processed.samples.len()
        );

        // Apply effects in order
        if self.config.enable_noise_reduction {
            processed = self.apply_noise_reduction(&processed)?;
        }

        if self.config.enable_highpass_filter {
            processed = self.apply_highpass_filter(&processed, self.config.highpass_cutoff)?;
        }

        if self.config.enable_lowpass_filter {
            processed = self.apply_lowpass_filter(&processed, self.config.lowpass_cutoff)?;
        }

        if self.config.enable_normalization {
            processed = self.apply_normalization(&processed, self.config.normalization_target)?;
        }

        info!("Audio effects processing completed");
        Ok(processed)
    }

    /// Apply noise reduction using spectral subtraction
    pub fn apply_noise_reduction(&mut self, audio: &AudioData) -> Result<AudioData> {
        debug!(
            "Applying noise reduction with strength: {}",
            self.config.noise_reduction_strength
        );

        if audio.samples.is_empty() {
            return Ok(audio.clone());
        }

        // Convert to mono for processing if stereo
        let mono_audio = if audio.channels > 1 {
            audio.to_mono()
        } else {
            audio.clone()
        };

        // Use FFT for frequency domain noise reduction
        let frame_size = 2048;
        let hop_size = frame_size / 4;
        let mut processed_samples = Vec::new();

        // Estimate noise from first 0.5 seconds
        let noise_frames = (mono_audio.sample_rate as f32 * 0.5) as usize;
        let noise_spectrum = self.estimate_noise_spectrum(
            &mono_audio.samples[..noise_frames.min(mono_audio.samples.len())],
            frame_size,
        )?;

        // Process audio in overlapping frames
        for start in (0..mono_audio.samples.len()).step_by(hop_size) {
            let end = (start + frame_size).min(mono_audio.samples.len());
            let frame = &mono_audio.samples[start..end];

            if frame.len() < frame_size {
                // Pad with zeros for last frame
                let mut padded_frame = vec![0.0; frame_size];
                padded_frame[..frame.len()].copy_from_slice(frame);
                let denoised_frame = self.denoise_frame(&padded_frame, &noise_spectrum)?;
                processed_samples.extend_from_slice(&denoised_frame[..frame.len()]);
            } else {
                let denoised_frame = self.denoise_frame(frame, &noise_spectrum)?;
                let start_idx = if start == 0 { 0 } else { hop_size };
                let end_idx = if end == mono_audio.samples.len() {
                    frame_size
                } else {
                    hop_size * 3
                };
                processed_samples.extend_from_slice(&denoised_frame[start_idx..end_idx]);
            }
        }

        // Convert back to original channel configuration
        let final_samples = if audio.channels > 1 {
            // Duplicate mono to stereo
            let mut stereo_samples =
                Vec::with_capacity(processed_samples.len() * audio.channels as usize);
            for sample in processed_samples {
                for _ in 0..audio.channels {
                    stereo_samples.push(sample);
                }
            }
            stereo_samples
        } else {
            processed_samples
        };

        Ok(AudioData::new(
            final_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Estimate noise spectrum from audio samples
    fn estimate_noise_spectrum(&mut self, samples: &[f32], frame_size: usize) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(vec![0.0; frame_size / 2 + 1]);
        }

        let mut noise_spectrum = vec![0.0; frame_size / 2 + 1];
        let mut frame_count = 0;

        for start in (0..samples.len()).step_by(frame_size) {
            let end = (start + frame_size).min(samples.len());
            if end - start < frame_size / 2 {
                break;
            }

            let mut frame = vec![0.0; frame_size];
            frame[..end - start].copy_from_slice(&samples[start..end]);

            // Apply window function
            self.apply_hann_window(&mut frame);

            // Convert to complex for FFT
            let mut complex_frame: Vec<Complex<f32>> =
                frame.iter().map(|&x| Complex::new(x, 0.0)).collect();

            // Perform FFT
            let fft = self.fft_planner.plan_fft_forward(frame_size);
            fft.process(&mut complex_frame);

            // Accumulate magnitude spectrum
            for (i, &complex_val) in complex_frame.iter().enumerate().take(frame_size / 2 + 1) {
                noise_spectrum[i] += complex_val.norm();
            }

            frame_count += 1;
        }

        // Average the spectrum
        if frame_count > 0 {
            for magnitude in &mut noise_spectrum {
                *magnitude /= frame_count as f32;
            }
        }

        Ok(noise_spectrum)
    }

    /// Denoise a single frame using spectral subtraction
    fn denoise_frame(&mut self, frame: &[f32], noise_spectrum: &[f32]) -> Result<Vec<f32>> {
        let frame_size = frame.len();
        let mut windowed_frame = frame.to_vec();

        // Apply window function
        self.apply_hann_window(&mut windowed_frame);

        // Convert to complex for FFT
        let mut complex_frame: Vec<Complex<f32>> = windowed_frame
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(frame_size);
        fft.process(&mut complex_frame);

        // Apply spectral subtraction
        for (i, complex_val) in complex_frame
            .iter_mut()
            .enumerate()
            .take(frame_size / 2 + 1)
        {
            let magnitude = complex_val.norm();
            let phase = complex_val.arg();

            let noise_mag = if i < noise_spectrum.len() {
                noise_spectrum[i]
            } else {
                0.0
            };

            // Spectral subtraction with over-subtraction factor
            let alpha = self.config.noise_reduction_strength * 2.0;
            let beta = 0.01; // Spectral floor
            let subtracted_mag = magnitude - alpha * noise_mag;
            let final_mag = subtracted_mag.max(beta * magnitude);

            *complex_val = Complex::from_polar(final_mag, phase);
        }

        // Mirror the spectrum for negative frequencies
        for i in 1..frame_size / 2 {
            complex_frame[frame_size - i] = complex_frame[i].conj();
        }

        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(frame_size);
        ifft.process(&mut complex_frame);

        // Extract real part and normalize
        let mut result: Vec<f32> = complex_frame
            .iter()
            .map(|c| c.re / frame_size as f32)
            .collect();

        // Apply window again for overlap-add
        self.apply_hann_window(&mut result);

        Ok(result)
    }

    /// Apply Hann window function
    fn apply_hann_window(&self, samples: &mut [f32]) {
        let n = samples.len();
        for (i, sample) in samples.iter_mut().enumerate() {
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            *sample *= window_val;
        }
    }

    /// Apply high-pass filter
    pub fn apply_highpass_filter(&self, audio: &AudioData, cutoff_freq: f32) -> Result<AudioData> {
        debug!("Applying high-pass filter with cutoff: {} Hz", cutoff_freq);

        let mut filtered_samples = audio.samples.clone();
        let sample_rate = audio.sample_rate as f32;

        // Simple first-order high-pass filter
        let rc = 1.0 / (2.0 * PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        let mut prev_input = 0.0;
        let mut prev_output = 0.0;

        for sample in &mut filtered_samples {
            let current_input = *sample;
            let output = alpha * (prev_output + current_input - prev_input);
            *sample = output;
            prev_input = current_input;
            prev_output = output;
        }

        Ok(AudioData::new(
            filtered_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Apply low-pass filter
    pub fn apply_lowpass_filter(&self, audio: &AudioData, cutoff_freq: f32) -> Result<AudioData> {
        debug!("Applying low-pass filter with cutoff: {} Hz", cutoff_freq);

        let mut filtered_samples = audio.samples.clone();
        let sample_rate = audio.sample_rate as f32;

        // Simple first-order low-pass filter
        let rc = 1.0 / (2.0 * PI * cutoff_freq);
        let dt = 1.0 / sample_rate;
        let alpha = dt / (rc + dt);

        let mut prev_output = 0.0;

        for sample in &mut filtered_samples {
            let output = prev_output + alpha * (*sample - prev_output);
            *sample = output;
            prev_output = output;
        }

        Ok(AudioData::new(
            filtered_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Apply normalization to target level
    pub fn apply_normalization(&self, audio: &AudioData, target_db: f32) -> Result<AudioData> {
        debug!("Applying normalization to target: {} dB", target_db);

        if audio.samples.is_empty() {
            return Ok(audio.clone());
        }

        // Calculate RMS level
        let rms = self.calculate_rms(&audio.samples);
        if rms == 0.0 {
            warn!("Audio RMS is zero, skipping normalization");
            return Ok(audio.clone());
        }

        // Convert RMS to dB
        let current_db = 20.0 * rms.log10();

        // Calculate gain needed
        let gain_db = target_db - current_db;
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        debug!(
            "Current level: {:.2} dB, Target: {:.2} dB, Gain: {:.2} dB",
            current_db, target_db, gain_db
        );

        // Apply gain
        let normalized_samples: Vec<f32> = audio.samples.iter().map(|&s| s * gain_linear).collect();

        // Check for clipping
        let peak = normalized_samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        if peak > 1.0 {
            warn!(
                "Normalization would cause clipping (peak: {:.2}), applying limiter",
                peak
            );
            let limiter_gain = 0.95 / peak;
            let limited_samples: Vec<f32> = normalized_samples
                .iter()
                .map(|&s| s * limiter_gain)
                .collect();

            Ok(AudioData::new(
                limited_samples,
                audio.sample_rate,
                audio.channels,
                audio.format,
            ))
        } else {
            Ok(AudioData::new(
                normalized_samples,
                audio.sample_rate,
                audio.channels,
                audio.format,
            ))
        }
    }

    /// Calculate RMS (Root Mean Square) level
    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Apply gain to audio
    pub fn apply_gain(&self, audio: &AudioData, gain_db: f32) -> Result<AudioData> {
        debug!("Applying gain: {} dB", gain_db);

        let gain_linear = 10.0_f32.powf(gain_db / 20.0);
        let gained_samples: Vec<f32> = audio.samples.iter().map(|&s| s * gain_linear).collect();

        Ok(AudioData::new(
            gained_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Apply fade in effect
    pub fn apply_fade_in(&self, audio: &AudioData, duration_seconds: f32) -> Result<AudioData> {
        debug!("Applying fade in: {} seconds", duration_seconds);

        let fade_samples = (duration_seconds * audio.sample_rate as f32) as usize;
        let mut faded_samples = audio.samples.clone();

        for (i, sample) in faded_samples.iter_mut().enumerate().take(fade_samples) {
            let fade_factor = i as f32 / fade_samples as f32;
            *sample *= fade_factor;
        }

        Ok(AudioData::new(
            faded_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Apply fade out effect
    pub fn apply_fade_out(&self, audio: &AudioData, duration_seconds: f32) -> Result<AudioData> {
        debug!("Applying fade out: {} seconds", duration_seconds);

        let fade_samples = (duration_seconds * audio.sample_rate as f32) as usize;
        let mut faded_samples = audio.samples.clone();
        let start_fade = faded_samples.len().saturating_sub(fade_samples);

        for (i, sample) in faded_samples.iter_mut().enumerate().skip(start_fade) {
            let fade_progress = (i - start_fade) as f32 / fade_samples as f32;
            let fade_factor = 1.0 - fade_progress;
            *sample *= fade_factor;
        }

        Ok(AudioData::new(
            faded_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Trim silence from beginning and end
    pub fn trim_silence(&self, audio: &AudioData, threshold: f32) -> Result<AudioData> {
        debug!("Trimming silence with threshold: {}", threshold);

        if audio.samples.is_empty() {
            return Ok(audio.clone());
        }

        // Find first non-silent sample
        let start = audio
            .samples
            .iter()
            .position(|&s| s.abs() > threshold)
            .unwrap_or(0);

        // Find last non-silent sample
        let end = audio
            .samples
            .iter()
            .rposition(|&s| s.abs() > threshold)
            .unwrap_or(audio.samples.len() - 1)
            + 1;

        if start >= end {
            // All silence, return empty audio
            return Ok(AudioData::new(
                vec![],
                audio.sample_rate,
                audio.channels,
                audio.format,
            ));
        }

        let trimmed_samples = audio.samples[start..end].to_vec();

        Ok(AudioData::new(
            trimmed_samples,
            audio.sample_rate,
            audio.channels,
            audio.format,
        ))
    }

    /// Get audio analysis information
    pub fn analyze_audio(&self, audio: &AudioData) -> AudioAnalysis {
        let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms = self.calculate_rms(&audio.samples);
        let dynamic_range = if rms > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };

        // Calculate zero crossing rate
        let mut zero_crossings = 0;
        for window in audio.samples.windows(2) {
            if (window[0] >= 0.0) != (window[1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zero_crossing_rate = zero_crossings as f32 / audio.samples.len() as f32;

        AudioAnalysis {
            peak_amplitude: peak,
            rms_level: rms,
            peak_db: if peak > 0.0 {
                20.0 * peak.log10()
            } else {
                -std::f32::INFINITY
            },
            rms_db: if rms > 0.0 {
                20.0 * rms.log10()
            } else {
                -std::f32::INFINITY
            },
            dynamic_range_db: dynamic_range,
            zero_crossing_rate,
            duration_seconds: audio.duration,
            sample_count: audio.samples.len(),
        }
    }

    /// Analyze audio characteristics with comprehensive analysis
    pub async fn analyze_audio_characteristics(
        &self,
        audio_data: &super::codecs::AudioData,
    ) -> Result<AudioAnalysis> {
        // Use the existing analyze_audio method
        Ok(self.analyze_audio(audio_data))
    }

    /// Apply effects to audio data with configuration
    pub async fn apply_effects(
        &self,
        audio_data: &mut super::codecs::AudioData,
        config: &EffectsConfig,
    ) -> Result<()> {
        // Create a temporary effects processor with the provided config
        let mut temp_effects = AudioEffects::new(config.clone());

        // Process the audio with the temporary effects processor
        let processed_audio = temp_effects.process(audio_data)?;

        // Update the original audio data with processed results
        audio_data.samples = processed_audio.samples;
        audio_data.duration = processed_audio.duration;

        debug!(
            "Applied effects to audio data: {} samples processed",
            audio_data.samples.len()
        );

        Ok(())
    }

    /// Get current effects configuration
    pub fn get_config(&self) -> &EffectsConfig {
        &self.config
    }

    /// Update effects configuration
    pub fn set_config(&mut self, config: EffectsConfig) {
        self.config = config;
    }

    /// Check if effects are enabled
    pub fn has_effects_enabled(&self) -> bool {
        self.config.enable_noise_reduction
            || self.config.enable_normalization
            || self.config.enable_highpass_filter
            || self.config.enable_lowpass_filter
    }
}

/// Audio analysis results
#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    /// Peak amplitude (0.0 to 1.0)
    pub peak_amplitude: f32,
    /// RMS level (0.0 to 1.0)
    pub rms_level: f32,
    /// Peak level in dB
    pub peak_db: f32,
    /// RMS level in dB
    pub rms_db: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Total sample count
    pub sample_count: usize,
}

/// Create an audio effects processor with default configuration
pub fn create_default_effects_processor() -> AudioEffects {
    AudioEffects::new(EffectsConfig::default())
}

/// Create an audio effects processor for voice processing
pub fn create_voice_effects_processor() -> AudioEffects {
    let config = EffectsConfig {
        enable_noise_reduction: true,
        noise_reduction_strength: 0.7,
        enable_normalization: true,
        normalization_target: -20.0,
        enable_highpass_filter: true,
        highpass_cutoff: 80.0,
        enable_lowpass_filter: true,
        lowpass_cutoff: 8000.0,
    };
    AudioEffects::new(config)
}

/// Create an audio effects processor for music processing
pub fn create_music_effects_processor() -> AudioEffects {
    let config = EffectsConfig {
        enable_noise_reduction: false,
        noise_reduction_strength: 0.3,
        enable_normalization: true,
        normalization_target: -23.0, // LUFS standard
        enable_highpass_filter: false,
        highpass_cutoff: 20.0,
        enable_lowpass_filter: false,
        lowpass_cutoff: 20000.0,
    };
    AudioEffects::new(config)
}
