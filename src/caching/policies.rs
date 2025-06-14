// Cache Policies and Strategies
// Provides configurable caching policies for different data types and access patterns

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Cache policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Policy name
    pub name: String,
    /// Policy type
    pub policy_type: PolicyType,
    /// Time-to-live configuration
    pub ttl_config: TtlConfig,
    /// Eviction configuration
    pub eviction_config: EvictionConfig,
    /// Compression configuration
    pub compression_config: CompressionConfig,
    /// Replication configuration
    pub replication_config: ReplicationConfig,
    /// Policy priority (higher = more important)
    pub priority: u8,
    /// Whether policy is enabled
    pub enabled: bool,
    /// Policy metadata
    pub metadata: HashMap<String, String>,
    /// Policy creation time
    pub created_at: DateTime<Utc>,
}

/// Types of cache policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PolicyType {
    /// High-frequency access data (AI model weights, embeddings)
    HighFrequency,
    /// Medium-frequency access data (user sessions, API responses)
    MediumFrequency,
    /// Low-frequency access data (configuration, static content)
    LowFrequency,
    /// Temporary data (computation results, intermediate values)
    Temporary,
    /// Critical data (authentication tokens, security keys)
    Critical,
    /// Large objects (files, images, documents)
    LargeObject,
    /// Real-time data (live metrics, streaming data)
    RealTime,
    /// Custom policy
    Custom,
}

/// TTL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtlConfig {
    /// Default TTL in seconds
    pub default_ttl: u64,
    /// Minimum TTL in seconds
    pub min_ttl: u64,
    /// Maximum TTL in seconds
    pub max_ttl: u64,
    /// TTL strategy
    pub strategy: TtlStrategy,
    /// Enable sliding expiration
    pub sliding_expiration: bool,
    /// Refresh threshold (refresh when TTL < threshold)
    pub refresh_threshold: f64,
}

/// TTL strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TtlStrategy {
    /// Fixed TTL
    Fixed,
    /// Adaptive TTL based on access patterns
    Adaptive,
    /// Exponential backoff TTL
    ExponentialBackoff,
    /// Random TTL within range
    Random,
    /// Access-based TTL (extends on access)
    AccessBased,
}

/// Eviction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionConfig {
    /// Eviction strategy
    pub strategy: EvictionStrategy,
    /// Memory threshold for eviction (percentage)
    pub memory_threshold: f64,
    /// Entry count threshold
    pub entry_threshold: usize,
    /// Eviction batch size
    pub batch_size: usize,
    /// Protect from eviction
    pub protected: bool,
}

/// Eviction strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based (oldest first)
    TimeBase,
    /// Size-based (largest first)
    SizeBased,
    /// Priority-based
    PriorityBased,
    /// Random eviction
    Random,
    /// No eviction (protected)
    NoEviction,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression threshold in bytes
    pub threshold: usize,
    /// Compression level (1-9)
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstandard compression (balanced)
    Zstd,
    /// Brotli compression (high ratio)
    Brotli,
}

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Enable replication across tiers
    pub enabled: bool,
    /// Replication strategy
    pub strategy: ReplicationStrategy,
    /// Number of replicas
    pub replica_count: u8,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Replication strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication
    Synchronous,
    /// Asynchronous replication
    Asynchronous,
    /// Write-through replication
    WriteThrough,
    /// Write-behind replication
    WriteBehind,
}

/// Consistency levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency
    Strong,
    /// Eventual consistency
    Eventual,
    /// Weak consistency
    Weak,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            policy_type: PolicyType::MediumFrequency,
            ttl_config: TtlConfig::default(),
            eviction_config: EvictionConfig::default(),
            compression_config: CompressionConfig::default(),
            replication_config: ReplicationConfig::default(),
            priority: 5,
            enabled: true,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }
}

impl Default for TtlConfig {
    fn default() -> Self {
        Self {
            default_ttl: 3600, // 1 hour
            min_ttl: 60,       // 1 minute
            max_ttl: 86400,    // 24 hours
            strategy: TtlStrategy::Fixed,
            sliding_expiration: false,
            refresh_threshold: 0.1, // Refresh when 10% of TTL remains
        }
    }
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            strategy: EvictionStrategy::LRU,
            memory_threshold: 80.0, // 80% memory usage
            entry_threshold: 10000,
            batch_size: 100,
            protected: false,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            threshold: 1024, // 1KB
            level: 6,        // Balanced compression
        }
    }
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: ReplicationStrategy::Asynchronous,
            replica_count: 1,
            consistency_level: ConsistencyLevel::Eventual,
        }
    }
}

impl CachePolicy {
    /// Create a high-frequency policy for AI model data
    pub fn high_frequency() -> Self {
        Self {
            name: "high_frequency".to_string(),
            policy_type: PolicyType::HighFrequency,
            ttl_config: TtlConfig {
                default_ttl: 7200, // 2 hours
                min_ttl: 300,      // 5 minutes
                max_ttl: 86400,    // 24 hours
                strategy: TtlStrategy::AccessBased,
                sliding_expiration: true,
                refresh_threshold: 0.2,
            },
            eviction_config: EvictionConfig {
                strategy: EvictionStrategy::LFU,
                memory_threshold: 90.0,
                entry_threshold: 5000,
                batch_size: 50,
                protected: true,
            },
            compression_config: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Lz4,
                threshold: 512,
                level: 3, // Fast compression
            },
            replication_config: ReplicationConfig {
                enabled: true,
                strategy: ReplicationStrategy::WriteThrough,
                replica_count: 2,
                consistency_level: ConsistencyLevel::Strong,
            },
            priority: 9,
            enabled: true,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Create a temporary policy for computation results
    pub fn temporary() -> Self {
        Self {
            name: "temporary".to_string(),
            policy_type: PolicyType::Temporary,
            ttl_config: TtlConfig {
                default_ttl: 300,  // 5 minutes
                min_ttl: 60,       // 1 minute
                max_ttl: 1800,     // 30 minutes
                strategy: TtlStrategy::Fixed,
                sliding_expiration: false,
                refresh_threshold: 0.0,
            },
            eviction_config: EvictionConfig {
                strategy: EvictionStrategy::TimeBase,
                memory_threshold: 70.0,
                entry_threshold: 1000,
                batch_size: 200,
                protected: false,
            },
            compression_config: CompressionConfig {
                enabled: false,
                algorithm: CompressionAlgorithm::None,
                threshold: 0,
                level: 0,
            },
            replication_config: ReplicationConfig {
                enabled: false,
                strategy: ReplicationStrategy::Asynchronous,
                replica_count: 0,
                consistency_level: ConsistencyLevel::Weak,
            },
            priority: 2,
            enabled: true,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Create a critical policy for security-sensitive data
    pub fn critical() -> Self {
        Self {
            name: "critical".to_string(),
            policy_type: PolicyType::Critical,
            ttl_config: TtlConfig {
                default_ttl: 1800, // 30 minutes
                min_ttl: 300,      // 5 minutes
                max_ttl: 3600,     // 1 hour
                strategy: TtlStrategy::Fixed,
                sliding_expiration: false,
                refresh_threshold: 0.5,
            },
            eviction_config: EvictionConfig {
                strategy: EvictionStrategy::NoEviction,
                memory_threshold: 95.0,
                entry_threshold: usize::MAX,
                batch_size: 1,
                protected: true,
            },
            compression_config: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Zstd,
                threshold: 256,
                level: 9, // Maximum compression
            },
            replication_config: ReplicationConfig {
                enabled: true,
                strategy: ReplicationStrategy::Synchronous,
                replica_count: 3,
                consistency_level: ConsistencyLevel::Strong,
            },
            priority: 10,
            enabled: true,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Create a large object policy for files and documents
    pub fn large_object() -> Self {
        Self {
            name: "large_object".to_string(),
            policy_type: PolicyType::LargeObject,
            ttl_config: TtlConfig {
                default_ttl: 86400, // 24 hours
                min_ttl: 3600,      // 1 hour
                max_ttl: 604800,    // 7 days
                strategy: TtlStrategy::Adaptive,
                sliding_expiration: true,
                refresh_threshold: 0.1,
            },
            eviction_config: EvictionConfig {
                strategy: EvictionStrategy::SizeBased,
                memory_threshold: 60.0,
                entry_threshold: 100,
                batch_size: 10,
                protected: false,
            },
            compression_config: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Zstd,
                threshold: 4096, // 4KB
                level: 6,
            },
            replication_config: ReplicationConfig {
                enabled: false,
                strategy: ReplicationStrategy::WriteBehind,
                replica_count: 1,
                consistency_level: ConsistencyLevel::Eventual,
            },
            priority: 3,
            enabled: true,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Calculate effective TTL based on policy and access patterns
    pub fn calculate_ttl(&self, access_count: u64, last_access: DateTime<Utc>) -> u64 {
        match self.ttl_config.strategy {
            TtlStrategy::Fixed => self.ttl_config.default_ttl,
            TtlStrategy::Adaptive => {
                // Increase TTL for frequently accessed items
                let base_ttl = self.ttl_config.default_ttl as f64;
                let access_factor = (access_count as f64).log10().max(1.0);
                let adaptive_ttl = (base_ttl * access_factor) as u64;
                adaptive_ttl.clamp(self.ttl_config.min_ttl, self.ttl_config.max_ttl)
            }
            TtlStrategy::ExponentialBackoff => {
                // Decrease TTL for items that haven't been accessed recently
                let hours_since_access = Utc::now()
                    .signed_duration_since(last_access)
                    .num_hours() as f64;
                let backoff_factor = 2.0_f64.powf(-hours_since_access / 24.0);
                let backoff_ttl = (self.ttl_config.default_ttl as f64 * backoff_factor) as u64;
                backoff_ttl.clamp(self.ttl_config.min_ttl, self.ttl_config.max_ttl)
            }
            TtlStrategy::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen_range(self.ttl_config.min_ttl..=self.ttl_config.max_ttl)
            }
            TtlStrategy::AccessBased => {
                if self.ttl_config.sliding_expiration {
                    // Reset TTL on access
                    self.ttl_config.default_ttl
                } else {
                    // Extend TTL based on access frequency
                    let extension = (access_count * 60).min(self.ttl_config.max_ttl / 2);
                    (self.ttl_config.default_ttl + extension)
                        .clamp(self.ttl_config.min_ttl, self.ttl_config.max_ttl)
                }
            }
        }
    }

    /// Check if entry should be refreshed based on remaining TTL
    pub fn should_refresh(&self, remaining_ttl: u64, original_ttl: u64) -> bool {
        if original_ttl == 0 {
            return false;
        }
        
        let remaining_ratio = remaining_ttl as f64 / original_ttl as f64;
        remaining_ratio < self.ttl_config.refresh_threshold
    }

    /// Get compression settings for entry size
    pub fn get_compression_settings(&self, entry_size: usize) -> (bool, CompressionAlgorithm, u8) {
        if !self.compression_config.enabled || entry_size < self.compression_config.threshold {
            return (false, CompressionAlgorithm::None, 0);
        }
        
        (
            true,
            self.compression_config.algorithm.clone(),
            self.compression_config.level,
        )
    }

    /// Check if entry is protected from eviction
    pub fn is_protected(&self) -> bool {
        self.eviction_config.protected || self.eviction_config.strategy == EvictionStrategy::NoEviction
    }

    /// Get eviction priority (higher = more likely to be evicted)
    pub fn get_eviction_priority(&self, access_count: u64, last_access: DateTime<Utc>, entry_size: usize) -> u64 {
        if self.is_protected() {
            return 0; // Never evict protected entries
        }
        
        match self.eviction_config.strategy {
            EvictionStrategy::LRU => {
                Utc::now().signed_duration_since(last_access).num_seconds() as u64
            }
            EvictionStrategy::LFU => {
                u64::MAX - access_count // Invert so lower access count = higher priority
            }
            EvictionStrategy::TimeBase => {
                Utc::now().signed_duration_since(last_access).num_seconds() as u64
            }
            EvictionStrategy::SizeBased => {
                entry_size as u64
            }
            EvictionStrategy::PriorityBased => {
                (10 - self.priority) as u64 // Invert priority
            }
            EvictionStrategy::Random => {
                use rand::Rng;
                rand::thread_rng().gen_range(1..=1000)
            }
            EvictionStrategy::NoEviction => 0,
        }
    }
}
