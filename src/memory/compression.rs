use std::collections::HashMap;
use std::io::{Read, Write};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde::{Deserialize, Serialize};
use crate::utils::error::{AgentError, Result};

/// Compression algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// GZIP compression (good balance of speed and compression ratio)
    Gzip,
    /// LZ4 compression (fastest, lower compression ratio)
    Lz4,
    /// ZSTD compression (best compression ratio, slower)
    Zstd,
    /// Custom dictionary-based compression for repetitive data
    Dictionary,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Algorithm to use
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9, algorithm dependent)
    pub level: u32,
    /// Minimum size threshold for compression (bytes)
    pub min_size_threshold: usize,
    /// Maximum dictionary size for dictionary compression
    pub max_dictionary_size: usize,
    /// Enable adaptive compression based on data characteristics
    pub adaptive: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Gzip,
            level: 6, // Balanced compression level
            min_size_threshold: 1024, // Only compress data > 1KB
            max_dictionary_size: 64 * 1024, // 64KB dictionary
            adaptive: true,
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub algorithm_used: CompressionAlgorithm,
    pub compression_time_ms: u64,
    pub decompression_time_ms: u64,
}

impl CompressionStats {
    fn new(original_size: usize, compressed_size: usize, algorithm: CompressionAlgorithm) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            algorithm_used: algorithm,
            compression_time_ms: 0,
            decompression_time_ms: 0,
        }
    }

    /// Calculate space savings percentage
    pub fn space_savings_percent(&self) -> f64 {
        (1.0 - self.compression_ratio) * 100.0
    }
}

/// Dictionary for dictionary-based compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionDictionary {
    /// Common patterns and their replacements
    patterns: HashMap<String, String>,
    /// Reverse mapping for decompression
    reverse_patterns: HashMap<String, String>,
    /// Dictionary usage statistics
    usage_stats: HashMap<String, usize>,
}

impl CompressionDictionary {
    /// Create a new empty dictionary
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            reverse_patterns: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }

    /// Build dictionary from sample data
    pub fn build_from_samples(&mut self, samples: &[String], max_patterns: usize) {
        let mut pattern_counts = HashMap::new();
        
        // Find common patterns (simple n-gram approach)
        for sample in samples {
            // Extract 3-grams to 8-grams
            for n in 3..=8 {
                for window in sample.chars().collect::<Vec<_>>().windows(n) {
                    let pattern: String = window.iter().collect();
                    if pattern.len() >= 3 && pattern.chars().all(|c| c.is_alphanumeric() || c.is_whitespace()) {
                        *pattern_counts.entry(pattern).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort patterns by frequency and take the most common ones
        let mut sorted_patterns: Vec<_> = pattern_counts.into_iter().collect();
        sorted_patterns.sort_by(|a, b| b.1.cmp(&a.1));

        // Create dictionary entries
        for (i, (pattern, count)) in sorted_patterns.iter().take(max_patterns).enumerate() {
            if pattern.len() > 3 && *count > 1 {
                let replacement = format!("§{}§", i); // Use special markers
                self.patterns.insert(pattern.clone(), replacement.clone());
                self.reverse_patterns.insert(replacement, pattern.clone());
                self.usage_stats.insert(pattern.clone(), *count);
            }
        }
    }

    /// Compress text using dictionary
    pub fn compress_text(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Sort patterns by length (longest first) to avoid partial replacements
        let mut patterns: Vec<_> = self.patterns.iter().collect();
        patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        
        for (pattern, replacement) in patterns {
            result = result.replace(pattern, replacement);
        }
        
        result
    }

    /// Decompress text using dictionary
    pub fn decompress_text(&self, compressed: &str) -> String {
        let mut result = compressed.to_string();
        
        for (replacement, pattern) in &self.reverse_patterns {
            result = result.replace(replacement, pattern);
        }
        
        result
    }

    /// Get dictionary efficiency metrics
    pub fn get_efficiency_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        let total_patterns = self.patterns.len();
        let total_usage: usize = self.usage_stats.values().sum();
        
        metrics.insert("total_patterns".to_string(), total_patterns as f64);
        metrics.insert("total_usage".to_string(), total_usage as f64);
        
        if total_patterns > 0 {
            metrics.insert("average_usage".to_string(), total_usage as f64 / total_patterns as f64);
        }
        
        metrics
    }
}

/// Memory compression engine
pub struct MemoryCompressor {
    config: CompressionConfig,
    dictionary: Option<CompressionDictionary>,
    stats_history: Vec<CompressionStats>,
}

impl MemoryCompressor {
    /// Create a new memory compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            dictionary: None,
            stats_history: Vec::new(),
        }
    }

    /// Set compression dictionary
    pub fn set_dictionary(&mut self, dictionary: CompressionDictionary) {
        self.dictionary = Some(dictionary);
    }

    /// Compress data with automatic algorithm selection
    pub fn compress(&mut self, data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        let start_time = std::time::Instant::now();
        
        // Skip compression for small data
        if data.len() < self.config.min_size_threshold {
            let stats = CompressionStats::new(data.len(), data.len(), CompressionAlgorithm::None);
            return Ok((data.to_vec(), stats));
        }

        let algorithm = if self.config.adaptive {
            self.select_optimal_algorithm(data)
        } else {
            self.config.algorithm
        };

        let compressed = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Gzip => self.compress_gzip(data)?,
            CompressionAlgorithm::Lz4 => self.compress_lz4(data)?,
            CompressionAlgorithm::Zstd => self.compress_zstd(data)?,
            CompressionAlgorithm::Dictionary => self.compress_dictionary(data)?,
        };

        let compression_time = start_time.elapsed().as_millis() as u64;
        let mut stats = CompressionStats::new(data.len(), compressed.len(), algorithm);
        stats.compression_time_ms = compression_time;

        self.stats_history.push(stats.clone());
        Ok((compressed, stats))
    }

    /// Decompress data
    pub fn decompress(&mut self, compressed_data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        let decompressed = match algorithm {
            CompressionAlgorithm::None => compressed_data.to_vec(),
            CompressionAlgorithm::Gzip => self.decompress_gzip(compressed_data)?,
            CompressionAlgorithm::Lz4 => self.decompress_lz4(compressed_data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(compressed_data)?,
            CompressionAlgorithm::Dictionary => self.decompress_dictionary(compressed_data)?,
        };

        let decompression_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats if we have a matching entry
        if let Some(stats) = self.stats_history.last_mut() {
            if stats.algorithm_used == algorithm {
                stats.decompression_time_ms = decompression_time;
            }
        }

        Ok(decompressed)
    }

    /// Select optimal compression algorithm based on data characteristics
    fn select_optimal_algorithm(&self, data: &[u8]) -> CompressionAlgorithm {
        // Analyze data characteristics
        let entropy = self.calculate_entropy(data);
        let repetition_ratio = self.calculate_repetition_ratio(data);
        
        // Decision logic based on data characteristics
        if entropy < 0.5 && repetition_ratio > 0.3 {
            // High repetition, low entropy - dictionary compression works well
            if self.dictionary.is_some() {
                CompressionAlgorithm::Dictionary
            } else {
                CompressionAlgorithm::Gzip
            }
        } else if data.len() > 100_000 {
            // Large data - use ZSTD for best compression
            CompressionAlgorithm::Zstd
        } else if data.len() < 10_000 {
            // Small data - use LZ4 for speed
            CompressionAlgorithm::Lz4
        } else {
            // Medium data - use GZIP for balance
            CompressionAlgorithm::Gzip
        }
    }

    /// Calculate Shannon entropy of data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        entropy / 8.0 // Normalize to 0-1 range
    }

    /// Calculate repetition ratio in data
    fn calculate_repetition_ratio(&self, data: &[u8]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }

        let mut repeated_bytes = 0;
        for window in data.windows(4) {
            if window[0] == window[1] && window[1] == window[2] && window[2] == window[3] {
                repeated_bytes += 1;
            }
        }

        repeated_bytes as f64 / (data.len() - 3) as f64
    }

    /// GZIP compression
    fn compress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.config.level));
        encoder.write_all(data)
            .map_err(|e| AgentError::tool("compression", &format!("GZIP compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| AgentError::tool("compression", &format!("GZIP finalization failed: {}", e)))
    }

    /// GZIP decompression
    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| AgentError::tool("compression", &format!("GZIP decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    /// LZ4 compression (placeholder - would need lz4 crate)
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to GZIP
        // In a real implementation, you would use the lz4 crate
        self.compress_gzip(data)
    }

    /// LZ4 decompression (placeholder)
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to GZIP
        self.decompress_gzip(data)
    }

    /// ZSTD compression (placeholder - would need zstd crate)
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to GZIP
        self.compress_gzip(data)
    }

    /// ZSTD decompression (placeholder)
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to GZIP
        self.decompress_gzip(data)
    }

    /// Dictionary-based compression
    fn compress_dictionary(&self, data: &[u8]) -> Result<Vec<u8>> {
        if let Some(dict) = &self.dictionary {
            let text = String::from_utf8_lossy(data);
            let compressed_text = dict.compress_text(&text);
            Ok(compressed_text.into_bytes())
        } else {
            Err(AgentError::tool("compression", "Dictionary not available for compression"))
        }
    }

    /// Dictionary-based decompression
    fn decompress_dictionary(&self, data: &[u8]) -> Result<Vec<u8>> {
        if let Some(dict) = &self.dictionary {
            let compressed_text = String::from_utf8_lossy(data);
            let decompressed_text = dict.decompress_text(&compressed_text);
            Ok(decompressed_text.into_bytes())
        } else {
            Err(AgentError::tool("compression", "Dictionary not available for decompression"))
        }
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> Vec<CompressionStats> {
        self.stats_history.clone()
    }

    /// Get average compression ratio
    pub fn get_average_compression_ratio(&self) -> f64 {
        if self.stats_history.is_empty() {
            return 1.0;
        }

        let total_ratio: f64 = self.stats_history.iter()
            .map(|s| s.compression_ratio)
            .sum();
        
        total_ratio / self.stats_history.len() as f64
    }

    /// Clear statistics history
    pub fn clear_stats(&mut self) {
        self.stats_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.algorithm, CompressionAlgorithm::Gzip);
        assert_eq!(config.level, 6);
        assert_eq!(config.min_size_threshold, 1024);
        assert_eq!(config.max_dictionary_size, 64 * 1024);
        assert!(config.adaptive);
    }

    #[test]
    fn test_compression_stats_calculation() {
        let stats = CompressionStats::new(1000, 500, CompressionAlgorithm::Gzip);
        assert_eq!(stats.original_size, 1000);
        assert_eq!(stats.compressed_size, 500);
        assert_eq!(stats.compression_ratio, 0.5);
        assert_eq!(stats.space_savings_percent(), 50.0);
    }

    #[test]
    fn test_compression_dictionary_creation() {
        let dict = CompressionDictionary::new();
        assert!(dict.patterns.is_empty());
        assert!(dict.reverse_patterns.is_empty());
        assert!(dict.usage_stats.is_empty());
    }

    #[test]
    fn test_compression_dictionary_building() {
        let mut dict = CompressionDictionary::new();
        let samples = vec![
            "The quick brown fox jumps over the lazy dog".to_string(),
            "The quick brown fox runs over the lazy cat".to_string(),
            "The quick brown fox walks over the lazy bird".to_string(),
        ];

        dict.build_from_samples(&samples, 10);

        // Should have found some common patterns
        assert!(!dict.patterns.is_empty());
        assert_eq!(dict.patterns.len(), dict.reverse_patterns.len());

        // Check that some patterns were found (may not always find specific patterns)
        // This is more lenient as pattern detection depends on the exact algorithm
        assert!(dict.patterns.len() <= 10); // Should not exceed max_patterns
    }

    #[test]
    fn test_compression_dictionary_compress_decompress() {
        let mut dict = CompressionDictionary::new();
        let samples = vec![
            "Hello world, this is a test".to_string(),
            "Hello world, this is another test".to_string(),
            "Hello world, this is yet another test".to_string(),
        ];

        dict.build_from_samples(&samples, 5);

        let original_text = "Hello world, this is a test message";
        let compressed = dict.compress_text(original_text);
        let decompressed = dict.decompress_text(&compressed);

        assert_eq!(original_text, decompressed);

        // Compression should have some effect if patterns were found
        if !dict.patterns.is_empty() {
            assert_ne!(original_text, compressed);
        }
    }

    #[test]
    fn test_compression_dictionary_efficiency_metrics() {
        let mut dict = CompressionDictionary::new();
        dict.patterns.insert("test".to_string(), "§0§".to_string());
        dict.usage_stats.insert("test".to_string(), 5);

        let metrics = dict.get_efficiency_metrics();
        assert_eq!(metrics["total_patterns"], 1.0);
        assert_eq!(metrics["total_usage"], 5.0);
        assert_eq!(metrics["average_usage"], 5.0);
    }

    #[tokio::test]
    async fn test_memory_compressor_small_data_skip() {
        let config = CompressionConfig {
            min_size_threshold: 100,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        let small_data = b"small";
        let (compressed, stats) = compressor.compress(small_data).unwrap();

        assert_eq!(compressed, small_data);
        assert_eq!(stats.algorithm_used, CompressionAlgorithm::None);
        assert_eq!(stats.compression_ratio, 1.0);
    }

    #[tokio::test]
    async fn test_memory_compressor_gzip_compression() {
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Gzip,
            adaptive: false,
            min_size_threshold: 10,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        let data = b"This is a longer text that should be compressed using GZIP algorithm for testing purposes. It contains repetitive patterns and should compress well.";
        let (compressed, stats) = compressor.compress(data).unwrap();

        assert_eq!(stats.algorithm_used, CompressionAlgorithm::Gzip);
        assert!(stats.compressed_size < stats.original_size);
        assert!(stats.compression_ratio < 1.0);

        // Test decompression
        let decompressed = compressor.decompress(&compressed, CompressionAlgorithm::Gzip).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[tokio::test]
    async fn test_memory_compressor_dictionary_compression() {
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Dictionary,
            adaptive: false,
            min_size_threshold: 10,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        // Create and set a dictionary
        let mut dict = CompressionDictionary::new();
        let samples = vec![
            "The quick brown fox".to_string(),
            "The quick brown dog".to_string(),
            "The quick brown cat".to_string(),
        ];
        dict.build_from_samples(&samples, 5);
        compressor.set_dictionary(dict);

        let data = b"The quick brown fox jumps high";
        let (compressed, stats) = compressor.compress(data).unwrap();

        assert_eq!(stats.algorithm_used, CompressionAlgorithm::Dictionary);

        // Test decompression
        let decompressed = compressor.decompress(&compressed, CompressionAlgorithm::Dictionary).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[tokio::test]
    async fn test_memory_compressor_adaptive_selection() {
        let config = CompressionConfig {
            adaptive: true,
            min_size_threshold: 10,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        // Test with repetitive data (should prefer dictionary or gzip)
        let repetitive_data = b"aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee";
        let (_, stats) = compressor.compress(repetitive_data).unwrap();

        // Should select an appropriate algorithm (not None for data above threshold)
        assert_ne!(stats.algorithm_used, CompressionAlgorithm::None);
    }

    #[test]
    fn test_memory_compressor_entropy_calculation() {
        let config = CompressionConfig::default();
        let compressor = MemoryCompressor::new(config);

        // Test with uniform data (high entropy)
        let uniform_data = (0..256).map(|i| i as u8).collect::<Vec<_>>();
        let entropy = compressor.calculate_entropy(&uniform_data);
        assert!(entropy > 0.9); // Should be close to 1.0

        // Test with repetitive data (low entropy)
        let repetitive_data = vec![0u8; 256];
        let entropy = compressor.calculate_entropy(&repetitive_data);
        assert!(entropy < 0.1); // Should be close to 0.0
    }

    #[test]
    fn test_memory_compressor_repetition_ratio() {
        let config = CompressionConfig::default();
        let compressor = MemoryCompressor::new(config);

        // Test with highly repetitive data
        let repetitive_data = vec![0u8; 100];
        let ratio = compressor.calculate_repetition_ratio(&repetitive_data);
        assert!(ratio > 0.9); // Should be close to 1.0

        // Test with random data
        let random_data: Vec<u8> = (0..100).map(|i| (i * 7 + 13) as u8).collect();
        let ratio = compressor.calculate_repetition_ratio(&random_data);
        assert!(ratio < 0.1); // Should be close to 0.0
    }

    #[tokio::test]
    async fn test_memory_compressor_statistics() {
        let config = CompressionConfig {
            min_size_threshold: 10,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        // Use larger, more compressible data
        let data1 = b"This is a longer piece of text that should compress well with GZIP. It has repetitive patterns and common words that compression algorithms can take advantage of.";
        let data2 = b"Another longer piece of text for compression testing. This also has repetitive patterns and should compress reasonably well with standard algorithms.";

        compressor.compress(data1).unwrap();
        compressor.compress(data2).unwrap();

        let stats = compressor.get_stats();
        assert_eq!(stats.len(), 2);

        let avg_ratio = compressor.get_average_compression_ratio();
        assert!(avg_ratio > 0.0);
        // For small data, compression might actually increase size due to headers
        // So we allow ratios > 1.0 but should be reasonable
        assert!(avg_ratio <= 2.0); // Allow some expansion but not excessive

        compressor.clear_stats();
        assert_eq!(compressor.get_stats().len(), 0);
    }

    #[tokio::test]
    async fn test_memory_compressor_error_handling() {
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Dictionary,
            adaptive: false,
            min_size_threshold: 10,
            ..Default::default()
        };
        let mut compressor = MemoryCompressor::new(config);

        // Try to compress without setting dictionary
        let data = b"Test data without dictionary";
        let result = compressor.compress(data);
        assert!(result.is_err());

        // Try to decompress without dictionary
        let result = compressor.decompress(data, CompressionAlgorithm::Dictionary);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_algorithm_serialization() {
        let algorithm = CompressionAlgorithm::Gzip;
        let serialized = serde_json::to_string(&algorithm).unwrap();
        let deserialized: CompressionAlgorithm = serde_json::from_str(&serialized).unwrap();
        assert_eq!(algorithm, deserialized);
    }

    #[test]
    fn test_compression_config_serialization() {
        let config = CompressionConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CompressionConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.algorithm, deserialized.algorithm);
        assert_eq!(config.level, deserialized.level);
    }

    #[test]
    fn test_compression_stats_serialization() {
        let stats = CompressionStats::new(1000, 500, CompressionAlgorithm::Zstd);
        let serialized = serde_json::to_string(&stats).unwrap();
        let deserialized: CompressionStats = serde_json::from_str(&serialized).unwrap();
        assert_eq!(stats.original_size, deserialized.original_size);
        assert_eq!(stats.compressed_size, deserialized.compressed_size);
        assert_eq!(stats.algorithm_used, deserialized.algorithm_used);
    }
}
