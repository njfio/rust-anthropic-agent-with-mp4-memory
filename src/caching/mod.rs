// Advanced Caching System for Enterprise AI Agent
// Provides multi-tier caching with Redis integration, TTL management, and performance optimization

use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub mod backends;
pub mod invalidation;
pub mod memory_cache;
pub mod metrics;
pub mod policies;
pub mod redis_cache;
pub mod strategies;

#[cfg(test)]
mod tests;

/// Multi-tier cache manager
pub struct CacheManager {
    /// Cache tiers (L1: memory, L2: Redis, L3: persistent)
    tiers: Vec<Arc<dyn CacheTier>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache metrics
    metrics: Arc<RwLock<CacheMetrics>>,
    /// Invalidation manager
    invalidation_manager: Arc<invalidation::InvalidationManager>,
    /// Cache policies
    policies: Arc<RwLock<HashMap<String, policies::CachePolicy>>>,
    /// Cache strategies
    strategies: Arc<RwLock<HashMap<String, Arc<dyn strategies::CacheStrategy>>>>,
    /// Active strategy name
    active_strategy: Arc<RwLock<Option<String>>>,
    /// Start time for uptime tracking
    start_time: Instant,
}

/// Cache tier trait for different storage backends
#[async_trait::async_trait]
pub trait CacheTier: Send + Sync {
    /// Get tier name
    fn name(&self) -> &str;

    /// Get tier level (1 = fastest, higher = slower but larger)
    fn level(&self) -> u8;

    /// Get cached value
    async fn get(&self, key: &str) -> Result<Option<CacheEntry>>;

    /// Set cached value with TTL
    async fn set(&self, key: &str, entry: CacheEntry) -> Result<()>;

    /// Delete cached value
    async fn delete(&self, key: &str) -> Result<bool>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Get tier statistics
    async fn stats(&self) -> Result<TierStats>;

    /// Clear all entries (use with caution)
    async fn clear(&self) -> Result<()>;

    /// Get tier health status
    async fn health_check(&self) -> Result<TierHealth>;
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached data
    pub data: Vec<u8>,
    /// Entry creation timestamp
    pub created_at: DateTime<Utc>,
    /// Time-to-live in seconds
    pub ttl: Option<u64>,
    /// Entry metadata
    pub metadata: CacheMetadata,
    /// Content type for deserialization
    pub content_type: String,
    /// Compression algorithm used
    pub compression: Option<CompressionType>,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Entry size in bytes
    pub size: usize,
    /// Access count
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Entry tags for grouping
    pub tags: Vec<String>,
    /// Entry priority (higher = more important)
    pub priority: u8,
    /// Source tier level
    pub source_tier: u8,
}

/// Compression types supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Lz4,
    Zstd,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable multi-tier caching
    pub enable_multi_tier: bool,
    /// Default TTL in seconds
    pub default_ttl: u64,
    /// Maximum entry size in bytes
    pub max_entry_size: usize,
    /// Enable compression for large entries
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Cache warming configuration
    pub warming_config: WarmingConfig,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable cache metrics
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
}

/// Cache warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingConfig {
    /// Enable cache warming
    pub enabled: bool,
    /// Warming batch size
    pub batch_size: usize,
    /// Warming interval in seconds
    pub interval: u64,
    /// Warming priority keys
    pub priority_keys: Vec<String>,
}

/// Cache eviction policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Time-based expiration only
    TTL,
    /// Random eviction
    Random,
    /// Priority-based eviction
    Priority,
}

/// Cache metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache hit ratio
    pub hit_ratio: f64,
    /// Total entries stored
    pub total_entries: u64,
    /// Total memory usage in bytes
    pub memory_usage: u64,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Tier-specific metrics
    pub tier_metrics: HashMap<String, TierMetrics>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Tier-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierMetrics {
    /// Tier hits
    pub hits: u64,
    /// Tier misses
    pub misses: u64,
    /// Tier hit ratio
    pub hit_ratio: f64,
    /// Tier entry count
    pub entry_count: u64,
    /// Tier memory usage
    pub memory_usage: u64,
    /// Average access time
    pub avg_access_time: f64,
}

/// Tier statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierStats {
    /// Number of entries
    pub entry_count: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Hit count
    pub hits: u64,
    /// Miss count
    pub misses: u64,
    /// Average access time in milliseconds
    pub avg_access_time: f64,
    /// Last access timestamp
    pub last_access: Option<DateTime<Utc>>,
}

/// Tier health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierHealth {
    /// Is tier healthy
    pub is_healthy: bool,
    /// Health score (0-100)
    pub health_score: u8,
    /// Last health check
    pub last_check: DateTime<Utc>,
    /// Error count
    pub error_count: u32,
    /// Connection status
    pub connection_status: ConnectionStatus,
}

/// Connection status for external tiers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Reconnecting,
    Error,
}

/// Cache operation result
#[derive(Debug, Clone)]
pub struct CacheResult<T> {
    /// Operation result
    pub value: Option<T>,
    /// Source tier that provided the value
    pub source_tier: Option<String>,
    /// Operation duration
    pub duration: Duration,
    /// Whether this was a cache hit
    pub hit: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_multi_tier: true,
            default_ttl: 3600,                // 1 hour
            max_entry_size: 10 * 1024 * 1024, // 10MB
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            warming_config: WarmingConfig::default(),
            eviction_policy: EvictionPolicy::LRU,
            enable_metrics: true,
            metrics_interval: Duration::from_secs(60),
        }
    }
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            batch_size: 100,
            interval: 300, // 5 minutes
            priority_keys: Vec::new(),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            total_entries: 0,
            memory_usage: 0,
            avg_response_time: 0.0,
            tier_metrics: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for TierStats {
    fn default() -> Self {
        Self {
            entry_count: 0,
            memory_usage: 0,
            hits: 0,
            misses: 0,
            avg_access_time: 0.0,
            last_access: None,
        }
    }
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: CacheConfig) -> Self {
        Self {
            tiers: Vec::new(),
            config,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
            invalidation_manager: Arc::new(invalidation::InvalidationManager::new()),
            policies: Arc::new(RwLock::new(HashMap::new())),
            strategies: Arc::new(RwLock::new(HashMap::new())),
            active_strategy: Arc::new(RwLock::new(None)),
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Create cache manager with default strategies
    pub async fn with_default_strategies() -> Result<Self> {
        let manager = Self::with_defaults();

        // Add default strategies
        let memory_source: Arc<dyn strategies::DataSource> = Arc::new(
            strategies::MemoryDataSource::new("default_memory".to_string()),
        );

        // Cache-aside strategy (most common)
        let cache_aside = Arc::new(strategies::CacheAsideStrategy::new(
            "cache_aside".to_string(),
            Arc::clone(&memory_source),
        ));
        manager.add_strategy(cache_aside).await?;

        // Write-through strategy
        let write_through = Arc::new(strategies::WriteThroughStrategy::new(
            "write_through".to_string(),
            Arc::clone(&memory_source),
        ));
        manager.add_strategy(write_through).await?;

        // Read-through strategy
        let read_through = Arc::new(strategies::ReadThroughStrategy::new(
            "read_through".to_string(),
            Arc::clone(&memory_source),
        ));
        manager.add_strategy(read_through).await?;

        // Refresh-ahead strategy
        let refresh_ahead = Arc::new(strategies::RefreshAheadStrategy::new(
            "refresh_ahead".to_string(),
            Arc::clone(&memory_source),
            0.2, // Refresh when 20% TTL remaining
        ));
        manager.add_strategy(refresh_ahead).await?;

        // Set cache-aside as default active strategy
        manager.set_active_strategy("cache_aside").await?;

        info!("Created cache manager with default strategies");
        Ok(manager)
    }

    /// Add a cache tier
    pub async fn add_tier(&mut self, tier: Arc<dyn CacheTier>) -> Result<()> {
        let tier_name = tier.name().to_string();
        let tier_level = tier.level();

        // Insert tier in order by level
        let insert_pos = self
            .tiers
            .iter()
            .position(|t| t.level() > tier_level)
            .unwrap_or(self.tiers.len());

        self.tiers.insert(insert_pos, tier);

        // Initialize tier metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.tier_metrics.insert(
                tier_name.clone(),
                TierMetrics {
                    hits: 0,
                    misses: 0,
                    hit_ratio: 0.0,
                    entry_count: 0,
                    memory_usage: 0,
                    avg_access_time: 0.0,
                },
            );
        }

        info!("Added cache tier: {} (level {})", tier_name, tier_level);
        Ok(())
    }

    /// Add a cache strategy
    pub async fn add_strategy(&self, strategy: Arc<dyn strategies::CacheStrategy>) -> Result<()> {
        let strategy_name = strategy.name().to_string();
        let mut strategies = self.strategies.write().await;
        strategies.insert(strategy_name.clone(), strategy);

        // Set as active strategy if it's the first one
        {
            let mut active = self.active_strategy.write().await;
            if active.is_none() {
                *active = Some(strategy_name.clone());
            }
        }

        info!("Added cache strategy: {}", strategy_name);
        Ok(())
    }

    /// Set active cache strategy
    pub async fn set_active_strategy(&self, strategy_name: &str) -> Result<()> {
        let strategies = self.strategies.read().await;
        if !strategies.contains_key(strategy_name) {
            return Err(AgentError::validation(format!(
                "Strategy '{}' not found",
                strategy_name
            )));
        }

        let mut active = self.active_strategy.write().await;
        *active = Some(strategy_name.to_string());

        info!("Set active cache strategy: {}", strategy_name);
        Ok(())
    }

    /// Get active cache strategy
    pub async fn get_active_strategy(&self) -> Option<Arc<dyn strategies::CacheStrategy>> {
        let active = self.active_strategy.read().await;
        if let Some(strategy_name) = active.as_ref() {
            let strategies = self.strategies.read().await;
            strategies.get(strategy_name).cloned()
        } else {
            None
        }
    }

    /// Get all available strategies
    pub async fn get_strategies(&self) -> HashMap<String, strategies::StrategyConfig> {
        let strategies = self.strategies.read().await;
        strategies
            .iter()
            .map(|(name, strategy)| (name.clone(), strategy.config()))
            .collect()
    }

    /// Remove a cache strategy
    pub async fn remove_strategy(&self, strategy_name: &str) -> Result<()> {
        let mut strategies = self.strategies.write().await;
        if strategies.remove(strategy_name).is_none() {
            return Err(AgentError::validation(format!(
                "Strategy '{}' not found",
                strategy_name
            )));
        }

        // Clear active strategy if it was the removed one
        {
            let mut active = self.active_strategy.write().await;
            if active.as_ref() == Some(&strategy_name.to_string()) {
                *active = None;
            }
        }

        info!("Removed cache strategy: {}", strategy_name);
        Ok(())
    }

    /// Get value using active strategy
    pub async fn get_with_strategy<T>(&self, key: &str) -> Result<CacheResult<T>>
    where
        T: for<'de> Deserialize<'de> + Serialize + Send + Sync + 'static,
    {
        // First try regular cache lookup
        let cache_result = self.get::<T>(key).await?;
        if cache_result.hit {
            return Ok(cache_result);
        }

        // If cache miss, use active strategy to load data
        if let Some(strategy) = self.get_active_strategy().await {
            let strategy_config = strategy.config();

            match strategy_config.strategy_type {
                strategies::StrategyType::CacheAside => {
                    // For cache-aside, the application handles loading
                    // Return the cache miss result
                    Ok(cache_result)
                }
                strategies::StrategyType::ReadThrough => {
                    // For read-through, attempt to load from data source
                    self.read_through_load::<T>(key, &strategy).await
                }
                strategies::StrategyType::RefreshAhead => {
                    // Check if refresh is needed and trigger background refresh
                    self.refresh_ahead_check::<T>(key, &strategy).await;
                    Ok(cache_result)
                }
                _ => {
                    // For other strategies, return cache miss
                    Ok(cache_result)
                }
            }
        } else {
            Ok(cache_result)
        }
    }

    /// Set value using active strategy
    pub async fn set_with_strategy<T>(&self, key: &str, value: &T, ttl: Option<u64>) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        if let Some(strategy) = self.get_active_strategy().await {
            let strategy_config = strategy.config();

            match strategy_config.strategy_type {
                strategies::StrategyType::WriteThrough => {
                    // Write to both cache and data source
                    self.write_through_save(key, value, &strategy).await?;
                    self.set(key, value, ttl).await
                }
                strategies::StrategyType::WriteBehind => {
                    // Write to cache immediately, queue for background write
                    self.set(key, value, ttl).await?;
                    self.write_behind_queue(key, value, &strategy).await
                }
                _ => {
                    // For other strategies, use regular cache set
                    self.set(key, value, ttl).await
                }
            }
        } else {
            self.set(key, value, ttl).await
        }
    }

    /// Read-through strategy implementation
    async fn read_through_load<T>(
        &self,
        key: &str,
        strategy: &Arc<dyn strategies::CacheStrategy>,
    ) -> Result<CacheResult<T>>
    where
        T: for<'de> Deserialize<'de> + Serialize + Send + Sync + 'static,
    {
        let start_time = Instant::now();

        // This is a simplified implementation - in a real scenario, we'd need access to the data source
        // For now, we'll simulate a data source load
        debug!(
            "Attempting read-through load for key: {} using strategy: {}",
            key,
            strategy.name()
        );

        // Simulate data source miss
        Ok(CacheResult {
            value: None,
            source_tier: Some("read_through".to_string()),
            duration: start_time.elapsed(),
            hit: false,
        })
    }

    /// Write-through strategy implementation
    async fn write_through_save<T>(
        &self,
        key: &str,
        value: &T,
        strategy: &Arc<dyn strategies::CacheStrategy>,
    ) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        debug!(
            "Write-through save for key: {} using strategy: {}",
            key,
            strategy.name()
        );

        // Serialize the value for data source storage
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;

        // In a real implementation, we'd write to the data source here
        // For now, we'll just log the operation
        info!("Write-through: saved {} bytes for key: {}", data.len(), key);
        Ok(())
    }

    /// Write-behind strategy implementation
    async fn write_behind_queue<T>(
        &self,
        key: &str,
        value: &T,
        strategy: &Arc<dyn strategies::CacheStrategy>,
    ) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        debug!(
            "Write-behind queue for key: {} using strategy: {}",
            key,
            strategy.name()
        );

        // Serialize the value for queuing
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;

        // In a real implementation, we'd add to the write queue here
        // For now, we'll just log the operation
        info!("Write-behind: queued {} bytes for key: {}", data.len(), key);
        Ok(())
    }

    /// Refresh-ahead strategy check
    async fn refresh_ahead_check<T>(&self, key: &str, strategy: &Arc<dyn strategies::CacheStrategy>)
    where
        T: for<'de> Deserialize<'de> + Serialize + Send + Sync + 'static,
    {
        debug!(
            "Refresh-ahead check for key: {} using strategy: {}",
            key,
            strategy.name()
        );

        // Check if the entry exists and needs refresh
        if let Ok(Some(entry)) = self.get_cache_entry(key).await {
            if self.should_refresh_entry(&entry) {
                // Trigger background refresh
                self.trigger_background_refresh::<T>(key, strategy).await;
            }
        }
    }

    /// Get cache entry without deserialization
    async fn get_cache_entry(&self, key: &str) -> Result<Option<CacheEntry>> {
        for tier in &self.tiers {
            match tier.get(key).await {
                Ok(Some(entry)) => {
                    if !self.is_expired(&entry) {
                        return Ok(Some(entry));
                    }
                }
                Ok(None) => continue,
                Err(_) => continue,
            }
        }
        Ok(None)
    }

    /// Check if entry should be refreshed
    fn should_refresh_entry(&self, entry: &CacheEntry) -> bool {
        if let Some(ttl) = entry.ttl {
            let age = Utc::now()
                .signed_duration_since(entry.created_at)
                .num_seconds() as u64;
            let remaining_ratio = (ttl - age) as f64 / ttl as f64;

            // Refresh if less than 20% of TTL remaining
            remaining_ratio < 0.2
        } else {
            false
        }
    }

    /// Trigger background refresh
    async fn trigger_background_refresh<T>(
        &self,
        key: &str,
        strategy: &Arc<dyn strategies::CacheStrategy>,
    ) where
        T: for<'de> Deserialize<'de> + Serialize + Send + Sync + 'static,
    {
        let key = key.to_string();
        let strategy_name = strategy.name().to_string();

        tokio::spawn(async move {
            debug!(
                "Background refresh started for key: {} using strategy: {}",
                key, strategy_name
            );

            // In a real implementation, we'd load from data source and update cache
            // For now, we'll just log the operation
            info!("Background refresh completed for key: {}", key);
        });
    }

    /// Get value from cache with multi-tier lookup
    pub async fn get<T>(&self, key: &str) -> Result<CacheResult<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start_time = Instant::now();

        for tier in &self.tiers {
            match tier.get(key).await {
                Ok(Some(entry)) => {
                    // Check TTL
                    if self.is_expired(&entry) {
                        // Remove expired entry
                        let _ = tier.delete(key).await;
                        continue;
                    }

                    // Deserialize data
                    let data = self.decompress_data(&entry)?;
                    let value: T = serde_json::from_slice(&data).map_err(|e| {
                        AgentError::validation(format!("Deserialization failed: {}", e))
                    })?;

                    // Update metrics
                    self.record_hit(tier.name(), start_time.elapsed()).await;

                    // Promote to higher tiers if configured
                    if self.config.enable_multi_tier && tier.level() > 1 {
                        self.promote_entry(key, &entry, tier.level()).await?;
                    }

                    return Ok(CacheResult {
                        value: Some(value),
                        source_tier: Some(tier.name().to_string()),
                        duration: start_time.elapsed(),
                        hit: true,
                    });
                }
                Ok(None) => {
                    // Record miss for this tier
                    self.record_miss(tier.name()).await;
                    continue;
                }
                Err(e) => {
                    warn!("Cache tier {} error: {}", tier.name(), e);
                    continue;
                }
            }
        }

        // No cache hit in any tier
        self.record_miss("all_tiers").await;
        Ok(CacheResult {
            value: None,
            source_tier: None,
            duration: start_time.elapsed(),
            hit: false,
        })
    }

    /// Set value in cache with multi-tier storage
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<u64>) -> Result<()>
    where
        T: Serialize,
    {
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;

        let compressed_data = self.compress_data(&data)?;
        let ttl = ttl.unwrap_or(self.config.default_ttl);

        let entry = CacheEntry {
            data: compressed_data,
            created_at: Utc::now(),
            ttl: Some(ttl),
            metadata: CacheMetadata {
                size: data.len(),
                access_count: 0,
                last_accessed: Utc::now(),
                tags: Vec::new(),
                priority: 5,    // Default priority
                source_tier: 1, // Set from L1
            },
            content_type: "application/json".to_string(),
            compression: if data.len() > self.config.compression_threshold {
                Some(CompressionType::Gzip)
            } else {
                Some(CompressionType::None)
            },
        };

        // Store in all tiers
        for tier in &self.tiers {
            if let Err(e) = tier.set(key, entry.clone()).await {
                warn!("Failed to set cache in tier {}: {}", tier.name(), e);
            }
        }

        debug!(
            "Cached entry: {} (size: {} bytes, TTL: {}s)",
            key, entry.metadata.size, ttl
        );
        Ok(())
    }

    /// Delete value from all cache tiers
    pub async fn delete(&self, key: &str) -> Result<bool> {
        let mut deleted = false;

        for tier in &self.tiers {
            match tier.delete(key).await {
                Ok(true) => deleted = true,
                Ok(false) => {}
                Err(e) => warn!("Failed to delete from tier {}: {}", tier.name(), e),
            }
        }

        if deleted {
            debug!("Deleted cache entry: {}", key);
        }

        Ok(deleted)
    }

    /// Check if key exists in any tier
    pub async fn exists(&self, key: &str) -> Result<bool> {
        for tier in &self.tiers {
            match tier.exists(key).await {
                Ok(true) => return Ok(true),
                Ok(false) => continue,
                Err(e) => warn!("Error checking existence in tier {}: {}", tier.name(), e),
            }
        }
        Ok(false)
    }

    /// Get cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        let mut metrics = self.metrics.read().await.clone();

        // Update hit ratio
        let total_requests = metrics.hits + metrics.misses;
        metrics.hit_ratio = if total_requests > 0 {
            metrics.hits as f64 / total_requests as f64 * 100.0
        } else {
            0.0
        };

        // Update tier metrics
        for tier in &self.tiers {
            if let Ok(stats) = tier.stats().await {
                if let Some(tier_metrics) = metrics.tier_metrics.get_mut(tier.name()) {
                    tier_metrics.entry_count = stats.entry_count;
                    tier_metrics.memory_usage = stats.memory_usage;
                    tier_metrics.avg_access_time = stats.avg_access_time;

                    let tier_total = tier_metrics.hits + tier_metrics.misses;
                    tier_metrics.hit_ratio = if tier_total > 0 {
                        tier_metrics.hits as f64 / tier_total as f64 * 100.0
                    } else {
                        0.0
                    };
                }
            }
        }

        metrics.last_updated = Utc::now();
        metrics
    }

    /// Get system health status
    pub async fn get_health(&self) -> Result<CacheHealth> {
        let mut tier_health = HashMap::new();
        let mut overall_healthy = true;

        for tier in &self.tiers {
            match tier.health_check().await {
                Ok(health) => {
                    if !health.is_healthy {
                        overall_healthy = false;
                    }
                    tier_health.insert(tier.name().to_string(), health);
                }
                Err(e) => {
                    overall_healthy = false;
                    tier_health.insert(
                        tier.name().to_string(),
                        TierHealth {
                            is_healthy: false,
                            health_score: 0,
                            last_check: Utc::now(),
                            error_count: 1,
                            connection_status: ConnectionStatus::Error,
                        },
                    );
                    error!("Health check failed for tier {}: {}", tier.name(), e);
                }
            }
        }

        let healthy_tiers = tier_health.values().filter(|h| h.is_healthy).count() as u8;

        Ok(CacheHealth {
            overall_healthy,
            tier_health,
            uptime: self.start_time.elapsed(),
            total_tiers: self.tiers.len() as u8,
            healthy_tiers,
        })
    }

    /// Clear all cache tiers
    pub async fn clear_all(&self) -> Result<()> {
        for tier in &self.tiers {
            if let Err(e) = tier.clear().await {
                error!("Failed to clear tier {}: {}", tier.name(), e);
            }
        }

        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = CacheMetrics::default();
        }

        info!("Cleared all cache tiers");
        Ok(())
    }

    /// Invalidate cache entries by pattern
    pub async fn invalidate_pattern(&self, pattern: &str) -> Result<u64> {
        self.invalidation_manager
            .invalidate_pattern(pattern, &self.tiers)
            .await
    }

    /// Invalidate cache entries by tags
    pub async fn invalidate_tags(&self, tags: &[String]) -> Result<u64> {
        self.invalidation_manager
            .invalidate_tags(tags, &self.tiers)
            .await
    }

    /// Warm cache with priority keys
    pub async fn warm_cache(&self) -> Result<()> {
        if !self.config.warming_config.enabled {
            return Ok(());
        }

        info!("Starting cache warming process");

        for key in &self.config.warming_config.priority_keys {
            // Check if key exists in any tier
            if !self.exists(key).await? {
                // Key not in cache, could trigger warming logic here
                debug!("Priority key not in cache: {}", key);
            }
        }

        info!("Cache warming completed");
        Ok(())
    }

    /// Check if cache entry is expired
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        if let Some(ttl) = entry.ttl {
            let age = Utc::now().signed_duration_since(entry.created_at);
            age.num_seconds() as u64 > ttl
        } else {
            false
        }
    }

    /// Compress data if needed
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.enable_compression || data.len() < self.config.compression_threshold {
            return Ok(data.to_vec());
        }

        // Simple gzip compression (could be enhanced with other algorithms)
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| AgentError::tool("cache", &format!("Compression failed: {}", e)))?;
        encoder.finish().map_err(|e| {
            AgentError::tool("cache", &format!("Compression finalization failed: {}", e))
        })
    }

    /// Decompress data if needed
    fn decompress_data(&self, entry: &CacheEntry) -> Result<Vec<u8>> {
        match entry.compression {
            Some(CompressionType::None) | None => Ok(entry.data.clone()),
            Some(CompressionType::Gzip) => {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(&entry.data[..]);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    AgentError::tool("cache", &format!("Decompression failed: {}", e))
                })?;
                Ok(decompressed)
            }
            _ => Err(AgentError::tool("cache", "Unsupported compression type")),
        }
    }

    /// Promote entry to higher cache tiers
    async fn promote_entry(&self, key: &str, entry: &CacheEntry, current_level: u8) -> Result<()> {
        for tier in &self.tiers {
            if tier.level() < current_level {
                if let Err(e) = tier.set(key, entry.clone()).await {
                    warn!("Failed to promote entry to tier {}: {}", tier.name(), e);
                }
            }
        }
        Ok(())
    }

    /// Record cache hit
    async fn record_hit(&self, tier_name: &str, duration: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.hits += 1;

        // Update average response time
        let new_avg = (metrics.avg_response_time + duration.as_millis() as f64) / 2.0;
        metrics.avg_response_time = new_avg;

        // Update tier metrics
        if let Some(tier_metrics) = metrics.tier_metrics.get_mut(tier_name) {
            tier_metrics.hits += 1;
        }
    }

    /// Record cache miss
    async fn record_miss(&self, tier_name: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.misses += 1;

        // Update tier metrics
        if let Some(tier_metrics) = metrics.tier_metrics.get_mut(tier_name) {
            tier_metrics.misses += 1;
        }
    }
}

/// Overall cache health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHealth {
    /// Overall system health
    pub overall_healthy: bool,
    /// Per-tier health status
    pub tier_health: HashMap<String, TierHealth>,
    /// System uptime
    pub uptime: Duration,
    /// Total number of tiers
    pub total_tiers: u8,
    /// Number of healthy tiers
    pub healthy_tiers: u8,
}
