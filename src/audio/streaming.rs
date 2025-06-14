// Real-time Audio Streaming with Buffering and Latency Management
// Provides cross-platform audio I/O using CPAL with proper async integration

use super::codecs::AudioData;
use crate::utils::error::{AgentError, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Sample, SampleFormat, SampleRate, Stream, StreamConfig, SupportedStreamConfig};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

/// Audio streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size in frames
    pub buffer_size: u32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Sample format
    pub sample_format: StreamSampleFormat,
    /// Maximum latency in milliseconds
    pub max_latency_ms: u32,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// Input gain multiplier
    pub input_gain: f32,
    /// Output gain multiplier
    pub output_gain: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            sample_rate: 44100,
            channels: 2,
            sample_format: StreamSampleFormat::F32,
            max_latency_ms: 50,
            enable_agc: false,
            input_gain: 1.0,
            output_gain: 1.0,
        }
    }
}

/// Sample format for streaming
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamSampleFormat {
    I16,
    U16,
    F32,
}

impl StreamSampleFormat {
    /// Convert to CPAL sample format
    pub fn to_cpal_format(&self) -> SampleFormat {
        match self {
            StreamSampleFormat::I16 => SampleFormat::I16,
            StreamSampleFormat::U16 => SampleFormat::U16,
            StreamSampleFormat::F32 => SampleFormat::F32,
        }
    }

    /// Get sample size in bytes
    pub fn sample_size(&self) -> usize {
        match self {
            StreamSampleFormat::I16 | StreamSampleFormat::U16 => 2,
            StreamSampleFormat::F32 => 4,
        }
    }
}

/// Audio streaming statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Buffer underruns
    pub underruns: u32,
    /// Buffer overruns
    pub overruns: u32,
    /// Average latency in milliseconds
    pub average_latency_ms: f32,
    /// Peak latency in milliseconds
    pub peak_latency_ms: f32,
    /// Stream uptime in seconds
    pub uptime_seconds: u64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// Device name
    pub name: String,
    /// Whether this is the default device
    pub is_default: bool,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Supported channel counts
    pub supported_channels: Vec<u16>,
    /// Supported sample formats
    pub supported_formats: Vec<StreamSampleFormat>,
    /// Maximum buffer size
    pub max_buffer_size: u32,
    /// Minimum buffer size
    pub min_buffer_size: u32,
}

/// Audio stream manager for real-time processing
pub struct AudioStreamManager {
    /// CPAL host
    host: Host,
    /// Current input device
    input_device: Option<Device>,
    /// Current output device
    output_device: Option<Device>,
    /// Active input stream
    input_stream: Option<Stream>,
    /// Active output stream
    output_stream: Option<Stream>,
    /// Streaming configuration
    config: StreamingConfig,
    /// Streaming statistics
    stats: Arc<RwLock<StreamingStats>>,
    /// Audio data sender for input
    input_sender: Option<broadcast::Sender<Vec<f32>>>,
    /// Audio data receiver for output
    output_receiver: Option<Arc<Mutex<Receiver<Vec<f32>>>>>,
    /// Stream start time
    start_time: Option<Instant>,
}

impl AudioStreamManager {
    /// Create a new audio stream manager
    pub fn new(config: StreamingConfig) -> Result<Self> {
        let host = cpal::default_host();
        
        Ok(Self {
            host,
            input_device: None,
            output_device: None,
            input_stream: None,
            output_stream: None,
            config,
            stats: Arc::new(RwLock::new(StreamingStats::default())),
            input_sender: None,
            output_receiver: None,
            start_time: None,
        })
    }

    /// List available audio devices
    pub fn list_devices(&self) -> Result<(Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>)> {
        let input_devices = self.host.input_devices()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to enumerate input devices: {}", e)))?;

        let output_devices = self.host.output_devices()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to enumerate output devices: {}", e)))?;

        let default_input = self.host.default_input_device();
        let default_output = self.host.default_output_device();

        let mut input_infos = Vec::new();
        for device in input_devices {
            if let Ok(info) = self.get_device_info(&device, &default_input) {
                input_infos.push(info);
            }
        }

        let mut output_infos = Vec::new();
        for device in output_devices {
            if let Ok(info) = self.get_device_info(&device, &default_output) {
                output_infos.push(info);
            }
        }

        Ok((input_infos, output_infos))
    }

    /// Get device information
    fn get_device_info(&self, device: &Device, default_device: &Option<Device>) -> Result<AudioDeviceInfo> {
        let name = device.name()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to get device name: {}", e)))?;

        let is_default = default_device.as_ref()
            .map(|d| d.name().unwrap_or_default() == name)
            .unwrap_or(false);

        // Get supported configurations (simplified approach)
        let supported_configs = device.supported_input_configs()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to get supported configs: {}", e)))?;

        let mut supported_sample_rates = Vec::new();
        let mut supported_channels = Vec::new();
        let mut supported_formats = Vec::new();
        let mut max_buffer_size = 0;
        let mut min_buffer_size = u32::MAX;

        for config in supported_configs {
            // Sample rates
            if !supported_sample_rates.contains(&config.min_sample_rate().0) {
                supported_sample_rates.push(config.min_sample_rate().0);
            }
            if !supported_sample_rates.contains(&config.max_sample_rate().0) {
                supported_sample_rates.push(config.max_sample_rate().0);
            }

            // Channels
            if !supported_channels.contains(&config.channels()) {
                supported_channels.push(config.channels());
            }

            // Sample formats
            let format = match config.sample_format() {
                SampleFormat::I16 => StreamSampleFormat::I16,
                SampleFormat::U16 => StreamSampleFormat::U16,
                SampleFormat::F32 => StreamSampleFormat::F32,
                _ => continue,
            };
            if !supported_formats.contains(&format) {
                supported_formats.push(format);
            }

            // Buffer sizes (CPAL doesn't expose min/max buffer sizes in this way)
            // Use reasonable defaults
            max_buffer_size = max_buffer_size.max(8192);
            min_buffer_size = min_buffer_size.min(64);
        }

        Ok(AudioDeviceInfo {
            name,
            is_default,
            supported_sample_rates,
            supported_channels,
            supported_formats,
            max_buffer_size,
            min_buffer_size: if min_buffer_size == u32::MAX { 64 } else { min_buffer_size },
        })
    }

    /// Initialize input stream for recording
    pub async fn start_input_stream(&mut self) -> Result<broadcast::Receiver<Vec<f32>>> {
        info!("Starting audio input stream");

        // Get default input device
        let device = self.host.default_input_device()
            .ok_or_else(|| AgentError::tool("audio_streaming".to_string(), "No default input device available".to_string()))?;

        // Get supported configuration
        let supported_config = device.default_input_config()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to get default input config: {}", e)))?;

        // Create stream configuration
        let stream_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
        };

        // Create broadcast channel for audio data
        let (sender, receiver) = broadcast::channel(1000);
        let sender_clone = sender.clone();
        let stats = Arc::clone(&self.stats);
        let start_time = Instant::now();

        // Create input stream based on sample format
        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => {
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let audio_data = data.to_vec();
                        if let Err(e) = sender_clone.send(audio_data) {
                            warn!("Failed to send audio data: {}", e);
                        }
                        
                        // Update stats (simplified to avoid lifetime issues)
                        // Note: In a real implementation, you'd use a different approach
                        // to update stats without borrowing data across async boundaries
                    },
                    |err| error!("Input stream error: {}", err),
                    None,
                )
            }
            SampleFormat::I16 => {
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let audio_data: Vec<f32> = data.iter()
                            .map(|&sample| sample as f32 / i16::MAX as f32)
                            .collect();
                        
                        if let Err(e) = sender_clone.send(audio_data) {
                            warn!("Failed to send audio data: {}", e);
                        }
                        
                        // Update stats (simplified to avoid lifetime issues)
                    },
                    |err| error!("Input stream error: {}", err),
                    None,
                )
            }
            SampleFormat::U16 => {
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        let audio_data: Vec<f32> = data.iter()
                            .map(|&sample| (sample as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0))
                            .collect();
                        
                        if let Err(e) = sender_clone.send(audio_data) {
                            warn!("Failed to send audio data: {}", e);
                        }
                        
                        // Update stats (simplified to avoid lifetime issues)
                    },
                    |err| error!("Input stream error: {}", err),
                    None,
                )
            }
            _ => return Err(AgentError::tool("audio_streaming".to_string(), "Unsupported sample format".to_string())),
        }.map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to build input stream: {}", e)))?;

        // Start the stream
        stream.play()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to start input stream: {}", e)))?;

        self.input_device = Some(device);
        self.input_stream = Some(stream);
        self.input_sender = Some(sender);
        self.start_time = Some(start_time);

        info!("Audio input stream started successfully");
        Ok(receiver)
    }

    /// Initialize output stream for playback
    pub async fn start_output_stream(&mut self) -> Result<Sender<Vec<f32>>> {
        info!("Starting audio output stream");

        // Get default output device
        let device = self.host.default_output_device()
            .ok_or_else(|| AgentError::tool("audio_streaming".to_string(), "No default output device available".to_string()))?;

        // Get supported configuration
        let supported_config = device.default_output_config()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to get default output config: {}", e)))?;

        // Create stream configuration
        let stream_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
        };

        // Create channel for audio data
        let (sender, receiver) = mpsc::channel::<Vec<f32>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let receiver_clone = Arc::clone(&receiver);
        let stats = Arc::clone(&self.stats);
        let start_time = self.start_time.unwrap_or_else(Instant::now);

        // Create output stream based on sample format
        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => {
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        if let Ok(receiver) = receiver_clone.lock() {
                            if let Ok(audio_data) = receiver.try_recv() {
                                let len = data.len().min(audio_data.len());
                                data[..len].copy_from_slice(&audio_data[..len]);
                                
                                // Update stats
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.frames_processed += len as u64;
                                        s.uptime_seconds = start_time.elapsed().as_secs();
                                    }
                                });
                            } else {
                                // Fill with silence if no data available
                                data.fill(0.0);
                                
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.underruns += 1;
                                    }
                                });
                            }
                        }
                    },
                    |err| error!("Output stream error: {}", err),
                    None,
                )
            }
            SampleFormat::I16 => {
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        if let Ok(receiver) = receiver_clone.lock() {
                            if let Ok(audio_data) = receiver.try_recv() {
                                let len = data.len().min(audio_data.len());
                                for (i, &sample) in audio_data[..len].iter().enumerate() {
                                    data[i] = (sample * i16::MAX as f32) as i16;
                                }
                                
                                // Update stats
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.frames_processed += len as u64;
                                        s.uptime_seconds = start_time.elapsed().as_secs();
                                    }
                                });
                            } else {
                                // Fill with silence if no data available
                                data.fill(0);
                                
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.underruns += 1;
                                    }
                                });
                            }
                        }
                    },
                    |err| error!("Output stream error: {}", err),
                    None,
                )
            }
            SampleFormat::U16 => {
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        if let Ok(receiver) = receiver_clone.lock() {
                            if let Ok(audio_data) = receiver.try_recv() {
                                let len = data.len().min(audio_data.len());
                                for (i, &sample) in audio_data[..len].iter().enumerate() {
                                    data[i] = ((sample + 1.0) * (u16::MAX as f32 / 2.0)) as u16;
                                }
                                
                                // Update stats
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.frames_processed += len as u64;
                                        s.uptime_seconds = start_time.elapsed().as_secs();
                                    }
                                });
                            } else {
                                // Fill with silence if no data available
                                data.fill(u16::MAX / 2);
                                
                                tokio::spawn({
                                    let stats = Arc::clone(&stats);
                                    async move {
                                        let mut s = stats.write().await;
                                        s.underruns += 1;
                                    }
                                });
                            }
                        }
                    },
                    |err| error!("Output stream error: {}", err),
                    None,
                )
            }
            _ => return Err(AgentError::tool("audio_streaming".to_string(), "Unsupported sample format".to_string())),
        }.map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to build output stream: {}", e)))?;

        // Start the stream
        stream.play()
            .map_err(|e| AgentError::tool("audio_streaming".to_string(), format!("Failed to start output stream: {}", e)))?;

        self.output_device = Some(device);
        self.output_stream = Some(stream);
        self.output_receiver = Some(receiver);
        if self.start_time.is_none() {
            self.start_time = Some(start_time);
        }

        info!("Audio output stream started successfully");
        Ok(sender)
    }

    /// Stop input stream
    pub async fn stop_input_stream(&mut self) -> Result<()> {
        if let Some(stream) = self.input_stream.take() {
            drop(stream);
            info!("Audio input stream stopped");
        }
        self.input_device = None;
        self.input_sender = None;
        Ok(())
    }

    /// Stop output stream
    pub async fn stop_output_stream(&mut self) -> Result<()> {
        if let Some(stream) = self.output_stream.take() {
            drop(stream);
            info!("Audio output stream stopped");
        }
        self.output_device = None;
        self.output_receiver = None;
        Ok(())
    }

    /// Stop all streams
    pub async fn stop_all_streams(&mut self) -> Result<()> {
        self.stop_input_stream().await?;
        self.stop_output_stream().await?;
        self.start_time = None;
        Ok(())
    }

    /// Get current streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.stats.read().await.clone()
    }

    /// Reset streaming statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = StreamingStats::default();
    }

    /// Check if input stream is active
    pub fn is_input_active(&self) -> bool {
        self.input_stream.is_some()
    }

    /// Check if output stream is active
    pub fn is_output_active(&self) -> bool {
        self.output_stream.is_some()
    }

    /// Get current latency estimate in milliseconds
    pub async fn get_latency_ms(&self) -> f32 {
        let buffer_frames = self.config.buffer_size as f32;
        let sample_rate = self.config.sample_rate as f32;
        (buffer_frames / sample_rate) * 1000.0
    }
}
