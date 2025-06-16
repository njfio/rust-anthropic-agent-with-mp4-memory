// Comprehensive Test Suite for Audio Processing System
// Tests all audio processing features with 25+ test functions

#[cfg(test)]
mod tests {
    use crate::audio::{
        codecs::{AudioCodec, AudioData},
        effects::{create_music_effects_processor, create_voice_effects_processor, AudioEffects},
        metadata::{create_metadata_extractor, AudioMetadata, MetadataExtractor},
        streaming::{AudioStreamManager, StreamSampleFormat, StreamingConfig},
        tool::create_default_audio_tool,
        AudioConfig, AudioFormat, AudioProcessor, AudioQuality, EffectsConfig, SynthesisConfig,
        TranscriptionConfig,
    };
    use crate::tools::Tool;
    use serde_json::json;
    use tempfile::NamedTempFile;

    // Helper function to create test audio data
    fn create_test_audio() -> AudioData {
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let samples_per_channel = (sample_rate as f64 * duration) as usize;
        let mut samples = Vec::new();

        // Generate a 440Hz sine wave (A4 note)
        for i in 0..samples_per_channel {
            let t = i as f64 / sample_rate as f64;
            let sample = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5;
            samples.push(sample); // Left channel
            samples.push(sample); // Right channel (stereo)
        }

        AudioData::new(samples, sample_rate, 2, AudioFormat::Wav)
    }

    // Helper function to create silent audio
    fn create_silent_audio(duration_seconds: f64) -> AudioData {
        let sample_rate = 44100;
        let samples_per_channel = (sample_rate as f64 * duration_seconds) as usize;
        let samples = vec![0.0; samples_per_channel * 2]; // Stereo silence

        AudioData::new(samples, sample_rate, 2, AudioFormat::Wav)
    }

    // Helper function to create noise audio
    fn create_noise_audio() -> AudioData {
        let sample_rate = 44100;
        let duration = 0.5; // 0.5 seconds
        let samples_per_channel = (sample_rate as f64 * duration) as usize;
        let mut samples = Vec::new();

        // Generate white noise
        for _ in 0..samples_per_channel * 2 {
            let sample = (rand::random::<f32>() - 0.5) * 0.1; // Low amplitude noise
            samples.push(sample);
        }

        AudioData::new(samples, sample_rate, 2, AudioFormat::Wav)
    }

    #[test]
    fn test_audio_format_detection() {
        assert_eq!(AudioFormat::from_extension("wav"), Some(AudioFormat::Wav));
        assert_eq!(AudioFormat::from_extension("mp3"), Some(AudioFormat::Mp3));
        assert_eq!(AudioFormat::from_extension("flac"), Some(AudioFormat::Flac));
        assert_eq!(AudioFormat::from_extension("ogg"), Some(AudioFormat::Ogg));
        assert_eq!(AudioFormat::from_extension("aac"), Some(AudioFormat::Aac));
        assert_eq!(AudioFormat::from_extension("m4a"), Some(AudioFormat::M4a));
        assert_eq!(AudioFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_audio_format_extensions() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Ogg.extension(), "ogg");
        assert_eq!(AudioFormat::Aac.extension(), "aac");
        assert_eq!(AudioFormat::M4a.extension(), "m4a");
    }

    #[test]
    fn test_audio_format_metadata_support() {
        assert!(!AudioFormat::Wav.supports_metadata());
        assert!(AudioFormat::Mp3.supports_metadata());
        assert!(AudioFormat::Flac.supports_metadata());
        assert!(!AudioFormat::Ogg.supports_metadata());
        assert!(!AudioFormat::Aac.supports_metadata());
        assert!(AudioFormat::M4a.supports_metadata());
    }

    #[test]
    fn test_audio_quality_presets() {
        let default = AudioQuality::default();
        assert_eq!(default.sample_rate, 44100);
        assert_eq!(default.channels, 2);
        assert_eq!(default.bit_depth, 16);

        let high = AudioQuality::high();
        assert_eq!(high.sample_rate, 48000);
        assert_eq!(high.channels, 2);
        assert_eq!(high.bit_depth, 24);

        let low = AudioQuality::low();
        assert_eq!(low.sample_rate, 22050);
        assert_eq!(low.channels, 1);
        assert_eq!(low.bit_depth, 16);

        let voice = AudioQuality::voice();
        assert_eq!(voice.sample_rate, 16000);
        assert_eq!(voice.channels, 1);
        assert_eq!(voice.bit_depth, 16);
    }

    #[test]
    fn test_audio_config_defaults() {
        let config = AudioConfig::default();
        assert_eq!(config.max_file_size, 100 * 1024 * 1024);
        assert_eq!(config.max_processing_duration, 300);
        assert!(config.enable_caching);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_audio_data_creation() {
        let audio = create_test_audio();
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.format, AudioFormat::Wav);
        assert!((audio.duration - 1.0).abs() < 0.01); // Approximately 1 second
        assert_eq!(audio.frames(), 44100); // 1 second at 44.1kHz
    }

    #[test]
    fn test_audio_data_mono_conversion() {
        let stereo_audio = create_test_audio();
        let mono_audio = stereo_audio.to_mono();

        assert_eq!(mono_audio.channels, 1);
        assert_eq!(mono_audio.sample_rate, stereo_audio.sample_rate);
        assert_eq!(mono_audio.frames(), stereo_audio.frames());
        assert_eq!(mono_audio.samples.len(), stereo_audio.samples.len() / 2);
    }

    #[test]
    fn test_audio_data_resampling() {
        let audio = create_test_audio();
        let resampled = audio.resample(22050).unwrap();

        assert_eq!(resampled.sample_rate, 22050);
        assert_eq!(resampled.channels, audio.channels);
        assert_eq!(resampled.frames(), 22050); // Half the original frames
    }

    #[test]
    fn test_audio_data_normalization() {
        let audio = create_test_audio();
        let normalized = audio.normalize(0.8);

        let max_amplitude = normalized
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((max_amplitude - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_wav_codec_encode_decode() {
        let original_audio = create_test_audio();
        let quality = AudioQuality::default();

        // Encode to bytes
        let wav_bytes =
            AudioCodec::encode_bytes(&original_audio, AudioFormat::Wav, &quality).unwrap();
        assert!(!wav_bytes.is_empty());

        // Decode from bytes
        let decoded_audio = AudioCodec::decode_bytes(&wav_bytes, AudioFormat::Wav).unwrap();

        assert_eq!(decoded_audio.sample_rate, original_audio.sample_rate);
        assert_eq!(decoded_audio.channels, original_audio.channels);
        assert_eq!(decoded_audio.format, AudioFormat::Wav);
    }

    #[test]
    fn test_wav_codec_file_operations() {
        let audio = create_test_audio();
        let quality = AudioQuality::default();

        // Create temporary file with .wav extension
        let temp_file = NamedTempFile::with_suffix(".wav").unwrap();
        let temp_path = temp_file.path();

        // Encode to file
        AudioCodec::encode_file(&audio, temp_path, &quality).unwrap();

        // Verify file exists and has content
        assert!(temp_path.exists());
        let file_size = std::fs::metadata(temp_path).unwrap().len();
        assert!(file_size > 0);

        // Decode from file
        let decoded_audio = AudioCodec::decode_file(temp_path).unwrap();

        assert_eq!(decoded_audio.sample_rate, audio.sample_rate);
        assert_eq!(decoded_audio.channels, audio.channels);
        assert_eq!(decoded_audio.format, AudioFormat::Wav);
    }

    #[test]
    fn test_audio_effects_creation() {
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        // Test preset processors
        let voice_effects = create_voice_effects_processor();
        let music_effects = create_music_effects_processor();

        // Verify they were created successfully (no panics)
        assert!(true);
    }

    #[test]
    fn test_audio_effects_gain() {
        let audio = create_test_audio();
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let gained_audio = effects.apply_gain(&audio, 6.0).unwrap(); // +6dB

        // Check that gain was applied (samples should be roughly doubled)
        let original_max = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let gained_max = gained_audio
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(gained_max > original_max * 1.5); // Should be roughly doubled
    }

    #[test]
    fn test_audio_effects_normalization() {
        let audio = create_test_audio();
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let normalized_audio = effects.apply_normalization(&audio, -20.0).unwrap();

        // Verify normalization was applied
        assert_eq!(normalized_audio.sample_rate, audio.sample_rate);
        assert_eq!(normalized_audio.channels, audio.channels);
        assert_eq!(normalized_audio.samples.len(), audio.samples.len());
    }

    #[test]
    fn test_audio_effects_fade_in() {
        let audio = create_test_audio();
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let faded_audio = effects.apply_fade_in(&audio, 0.1).unwrap(); // 100ms fade

        // Check that the beginning is quieter than the end
        let start_sample = faded_audio.samples[0].abs();
        let end_sample = faded_audio.samples[faded_audio.samples.len() - 1].abs();

        assert!(start_sample < end_sample);
    }

    #[test]
    fn test_audio_effects_fade_out() {
        let audio = create_test_audio();
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let faded_audio = effects.apply_fade_out(&audio, 0.1).unwrap(); // 100ms fade

        // Check that fade out was applied by comparing RMS levels
        let start_range_end = 1000.min(faded_audio.samples.len());
        let start_rms = (faded_audio.samples[..start_range_end]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            / start_range_end as f32)
            .sqrt();

        let end_range_start = faded_audio.samples.len().saturating_sub(1000);
        let end_rms = (faded_audio.samples[end_range_start..]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            / (faded_audio.samples.len() - end_range_start) as f32)
            .sqrt();

        // The fade out should make the end significantly quieter
        assert!(
            end_rms < start_rms * 0.5,
            "End RMS: {}, Start RMS: {}",
            end_rms,
            start_rms
        );
    }

    #[test]
    fn test_audio_effects_trim_silence() {
        // Create audio with silence at beginning and end
        let mut samples = vec![0.0; 1000]; // Silence
        samples.extend(create_test_audio().samples); // Audio content
        samples.extend(vec![0.0; 1000]); // More silence

        let audio_with_silence = AudioData::new(samples, 44100, 2, AudioFormat::Wav);

        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let trimmed_audio = effects.trim_silence(&audio_with_silence, 0.01).unwrap();

        // Trimmed audio should be shorter
        assert!(trimmed_audio.samples.len() < audio_with_silence.samples.len());
    }

    #[test]
    fn test_audio_effects_analysis() {
        let audio = create_test_audio();
        let config = EffectsConfig::default();
        let effects = AudioEffects::new(config);

        let analysis = effects.analyze_audio(&audio);

        assert!(analysis.peak_amplitude > 0.0);
        assert!(analysis.rms_level > 0.0);
        assert!(analysis.peak_db > -100.0);
        assert!(analysis.rms_db > -100.0);
        assert!(analysis.dynamic_range_db >= 0.0);
        assert!(analysis.zero_crossing_rate >= 0.0);
        assert_eq!(analysis.duration_seconds, audio.duration);
        assert_eq!(analysis.sample_count, audio.samples.len());
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.buffer_size, 1024);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.sample_format, StreamSampleFormat::F32);
        assert_eq!(config.max_latency_ms, 50);
        assert!(!config.enable_agc);
        assert_eq!(config.input_gain, 1.0);
        assert_eq!(config.output_gain, 1.0);
    }

    #[test]
    fn test_streaming_sample_format() {
        assert_eq!(StreamSampleFormat::I16.sample_size(), 2);
        assert_eq!(StreamSampleFormat::U16.sample_size(), 2);
        assert_eq!(StreamSampleFormat::F32.sample_size(), 4);
    }

    #[tokio::test]
    async fn test_audio_stream_manager_creation() {
        let config = StreamingConfig::default();
        let result = AudioStreamManager::new(config);

        assert!(result.is_ok());
        let manager = result.unwrap();
        assert!(!manager.is_input_active());
        assert!(!manager.is_output_active());
    }

    #[tokio::test]
    async fn test_audio_stream_manager_device_listing() {
        let config = StreamingConfig::default();
        let manager = AudioStreamManager::new(config).unwrap();

        // This might fail on systems without audio devices, so we'll just check it doesn't panic
        let result = manager.list_devices();
        // Don't assert on the result as it depends on system audio configuration
    }

    #[tokio::test]
    async fn test_audio_stream_manager_latency_calculation() {
        let config = StreamingConfig::default();
        let manager = AudioStreamManager::new(config).unwrap();

        let latency = manager.get_latency_ms().await;
        assert!(latency > 0.0);
        assert!(latency < 1000.0); // Should be reasonable
    }

    #[test]
    fn test_transcription_config_defaults() {
        let config = TranscriptionConfig::default();
        assert_eq!(config.provider, "whisper");
        assert_eq!(config.model, "base");
        assert_eq!(config.language, Some("en".to_string()));
        assert!(config.auto_detect_language);
        assert_eq!(config.timeout, 60);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_synthesis_config_defaults() {
        let config = SynthesisConfig::default();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.voice, "alloy");
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.output_format, AudioFormat::Mp3);
        assert_eq!(config.timeout, 60);
    }

    #[test]
    fn test_metadata_extractor_creation() {
        let extractor = create_metadata_extractor();
        // Just verify it can be created without panicking
        assert!(true);
    }

    #[test]
    fn test_metadata_validation() {
        let mut metadata = AudioMetadata::default();
        let issues = MetadataExtractor::validate_metadata(&metadata);

        // Should have several validation issues for default metadata
        assert!(!issues.is_empty());
        assert!(issues.contains(&"Missing title".to_string()));
        assert!(issues.contains(&"Missing artist".to_string()));

        // Fix some issues
        metadata.title = Some("Test Title".to_string());
        metadata.artist = Some("Test Artist".to_string());
        metadata.album = Some("Test Album".to_string());
        metadata.year = Some(2024);
        metadata.duration_seconds = 180.0;
        metadata.sample_rate = 44100;
        metadata.channels = 2;

        let issues_after = MetadataExtractor::validate_metadata(&metadata);
        assert!(issues_after.is_empty());
    }

    #[test]
    fn test_metadata_completeness_check() {
        let mut metadata = AudioMetadata::default();
        assert!(!MetadataExtractor::has_complete_metadata(&metadata));

        metadata.title = Some("Test Title".to_string());
        metadata.artist = Some("Test Artist".to_string());
        metadata.album = Some("Test Album".to_string());
        metadata.year = Some(2024);

        assert!(MetadataExtractor::has_complete_metadata(&metadata));
    }

    #[tokio::test]
    async fn test_audio_processor_creation() {
        let config = AudioConfig::default();
        let processor = AudioProcessor::new(config);

        let stats = processor.get_stats().await;
        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.total_processing_time, 0);
        assert_eq!(stats.bytes_processed, 0);
    }

    #[tokio::test]
    async fn test_audio_processor_validation() {
        let config = AudioConfig::default();
        let processor = AudioProcessor::new(config);

        // Test file size validation with a relative path
        let test_file_path = std::path::Path::new("test_audio.wav");

        // Test normal file size
        let result = processor.validate_audio_file(test_file_path, 1000);
        assert!(result.is_ok(), "Validation failed: {:?}", result);

        // Test oversized file
        let result = processor.validate_audio_file(test_file_path, 200 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audio_tool_creation() {
        let result = create_default_audio_tool();
        assert!(result.is_ok());

        let tool = result.unwrap();
        assert_eq!(tool.name(), "audio_processing");
        assert!(tool.description().is_some());
    }

    #[tokio::test]
    async fn test_audio_tool_validation() {
        let tool = create_default_audio_tool().unwrap();

        // Test valid input
        let valid_input = json!({
            "action": "statistics"
        });
        assert!(tool.validate_input(&valid_input).is_ok());

        // Test invalid action
        let invalid_input = json!({
            "action": "invalid_action"
        });
        assert!(tool.validate_input(&invalid_input).is_err());

        // Test missing required parameter
        let missing_param = json!({
            "action": "decode"
            // Missing file_path
        });
        assert!(tool.validate_input(&missing_param).is_err());
    }

    #[tokio::test]
    async fn test_audio_tool_statistics() {
        let tool = create_default_audio_tool().unwrap();

        let input = json!({
            "action": "statistics"
        });

        let result = tool.execute(input).await;
        assert!(result.is_ok());

        let tool_result = result.unwrap();
        assert!(!tool_result.is_error);
        assert!(tool_result.content.contains("statistics"));
    }

    #[test]
    fn test_audio_tool_definition() {
        let tool = create_default_audio_tool().unwrap();
        let definition = tool.definition();

        assert_eq!(definition.name, "audio_processing");
        assert!(definition
            .description
            .unwrap_or_default()
            .contains("audio processing"));
        assert!(definition
            .input_schema
            .as_ref()
            .map(|s| s.is_object())
            .unwrap_or(false));
    }
}
