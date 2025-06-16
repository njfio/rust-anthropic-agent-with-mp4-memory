//! DSPy caching system for compiled modules and optimization results
//!
//! This module provides persistent and in-memory caching for DSPy compilation
//! results, with TTL support, size limits, and performance monitoring.

use crate::dspy::error::{DspyError, DspyResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: usize,
    /// Entry size in bytes (estimated)
    pub size_bytes: usize,
    /// Entry tags for categorization
    pub tags: Vec<String>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Time-to-live in seconds
    pub ttl_seconds: u64,
    /// Enable persistent storage
    pub persistent: bool,
    /// Cache directory for persistent storage
    pub cache_dir: PathBuf,
    /// Enable compression for stored entries
    pub enable_compression: bool,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            ttl_seconds: 86400,                 // 24 hours
            persistent: true,
            cache_dir: PathBuf::from("./dspy_cache"),
            enable_compression: true,
            max_entries: 10000,
            cleanup_interval_seconds: 3600, // 1 hour
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Current cache size in bytes
    pub size_bytes: usize,
    /// Number of entries
    pub entry_count: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Number of expired entries removed
    pub expirations: usize,
}

/// Generic cache implementation with TTL and size limits
pub struct Cache<T> {
    /// Cache configuration
    config: CacheConfig,
    /// In-memory cache storage
    storage: Arc<RwLock<HashMap<String, CacheEntry<T>>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Last cleanup time
    last_cleanup: Arc<RwLock<u64>>,
}

impl<T> Cache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    /// Create a new cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        // Create cache directory if it doesn't exist
        if config.persistent {
            if let Err(e) = fs::create_dir_all(&config.cache_dir) {
                warn!("Failed to create cache directory: {}", e);
            }
        }

        Self {
            config,
            storage: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            last_cleanup: Arc::new(RwLock::new(0)),
        }
    }

    /// Get an entry from the cache
    pub async fn get(&self, key: &str) -> DspyResult<Option<T>> {
        // Check if cleanup is needed
        self.maybe_cleanup().await?;

        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = storage.get_mut(key) {
            // Check if entry is expired
            let now = self.current_timestamp();
            if now - entry.created_at > self.config.ttl_seconds {
                storage.remove(key);
                stats.expirations += 1;
                stats.misses += 1;
                return Ok(None);
            }

            // Update access statistics
            entry.last_accessed = now;
            entry.access_count += 1;
            stats.hits += 1;

            debug!("Cache hit for key: {}", key);
            Ok(Some(entry.data.clone()))
        } else {
            stats.misses += 1;
            debug!("Cache miss for key: {}", key);

            // Try to load from persistent storage
            if self.config.persistent {
                if let Ok(Some(data)) = self.load_from_disk(key).await {
                    // Add to in-memory cache
                    let now = self.current_timestamp();
                    let entry = CacheEntry {
                        data: data.clone(),
                        created_at: now,
                        last_accessed: now,
                        access_count: 1,
                        size_bytes: self.estimate_size(&data),
                        tags: Vec::new(),
                    };

                    storage.insert(key.to_string(), entry);
                    stats.entry_count += 1;
                    stats.hits += 1;

                    return Ok(Some(data));
                }
            }

            Ok(None)
        }
    }

    /// Put an entry into the cache
    pub async fn put(&self, key: &str, data: T) -> DspyResult<()> {
        self.put_with_tags(key, data, Vec::new()).await
    }

    /// Put an entry into the cache with tags
    pub async fn put_with_tags(&self, key: &str, data: T, tags: Vec<String>) -> DspyResult<()> {
        let now = self.current_timestamp();
        let size_bytes = self.estimate_size(&data);

        let entry = CacheEntry {
            data: data.clone(),
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            tags,
        };

        {
            let mut storage = self.storage.write().await;
            let mut stats = self.stats.write().await;

            // Check if we need to evict entries
            self.maybe_evict(&mut storage, &mut stats, size_bytes)
                .await?;

            // Insert the new entry
            let is_new = !storage.contains_key(key);
            storage.insert(key.to_string(), entry);

            if is_new {
                stats.entry_count += 1;
                stats.size_bytes += size_bytes;
            }
        }

        // Save to persistent storage if enabled
        if self.config.persistent {
            self.save_to_disk(key, &data).await?;
        }

        debug!("Cached entry for key: {} (size: {} bytes)", key, size_bytes);
        Ok(())
    }

    /// Remove an entry from the cache
    pub async fn remove(&self, key: &str) -> DspyResult<bool> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = storage.remove(key) {
            stats.entry_count -= 1;
            stats.size_bytes = stats.size_bytes.saturating_sub(entry.size_bytes);

            // Remove from persistent storage
            if self.config.persistent {
                let file_path = self.get_cache_file_path(key);
                if file_path.exists() {
                    if let Err(e) = fs::remove_file(&file_path) {
                        warn!("Failed to remove cache file: {}", e);
                    }
                }
            }

            debug!("Removed cache entry for key: {}", key);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> DspyResult<()> {
        {
            let mut storage = self.storage.write().await;
            let mut stats = self.stats.write().await;

            storage.clear();
            stats.entry_count = 0;
            stats.size_bytes = 0;
        }

        // Clear persistent storage
        if self.config.persistent {
            if let Err(e) = fs::remove_dir_all(&self.config.cache_dir) {
                warn!("Failed to clear cache directory: {}", e);
            }
            if let Err(e) = fs::create_dir_all(&self.config.cache_dir) {
                warn!("Failed to recreate cache directory: {}", e);
            }
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }

    /// Check if cache contains a key
    pub async fn contains(&self, key: &str) -> bool {
        let storage = self.storage.read().await;
        storage.contains_key(key)
    }

    /// Get all cache keys
    pub async fn keys(&self) -> Vec<String> {
        let storage = self.storage.read().await;
        storage.keys().cloned().collect()
    }

    /// Get entries by tag
    pub async fn get_by_tag(&self, tag: &str) -> Vec<String> {
        let storage = self.storage.read().await;
        storage
            .iter()
            .filter(|(_, entry)| entry.tags.contains(&tag.to_string()))
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Cleanup expired entries
    async fn maybe_cleanup(&self) -> DspyResult<()> {
        let now = self.current_timestamp();
        let mut last_cleanup = self.last_cleanup.write().await;

        if now - *last_cleanup > self.config.cleanup_interval_seconds {
            *last_cleanup = now;
            drop(last_cleanup);

            self.cleanup_expired().await?;
        }

        Ok(())
    }

    /// Remove expired entries
    async fn cleanup_expired(&self) -> DspyResult<()> {
        let now = self.current_timestamp();
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        let mut expired_keys = Vec::new();
        for (key, entry) in storage.iter() {
            if now - entry.created_at > self.config.ttl_seconds {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            if let Some(entry) = storage.remove(&key) {
                stats.entry_count -= 1;
                stats.size_bytes = stats.size_bytes.saturating_sub(entry.size_bytes);
                stats.expirations += 1;
            }
        }

        if stats.expirations > 0 {
            debug!("Cleaned up {} expired cache entries", stats.expirations);
        }

        Ok(())
    }

    /// Evict entries if cache is too large
    async fn maybe_evict(
        &self,
        storage: &mut HashMap<String, CacheEntry<T>>,
        stats: &mut CacheStats,
        new_entry_size: usize,
    ) -> DspyResult<()> {
        // Check size limit
        if stats.size_bytes + new_entry_size > self.config.max_size_bytes {
            self.evict_lru(storage, stats, new_entry_size).await?;
        }

        // Check entry count limit
        if stats.entry_count >= self.config.max_entries {
            self.evict_lru(storage, stats, 0).await?;
        }

        Ok(())
    }

    /// Evict least recently used entries
    async fn evict_lru(
        &self,
        storage: &mut HashMap<String, CacheEntry<T>>,
        stats: &mut CacheStats,
        needed_space: usize,
    ) -> DspyResult<()> {
        let mut entries: Vec<(String, u64)> = storage
            .iter()
            .map(|(key, entry)| (key.clone(), entry.last_accessed))
            .collect();

        // Sort by last accessed time (oldest first)
        entries.sort_by_key(|(_, last_accessed)| *last_accessed);

        let mut freed_space = 0;
        let mut evicted_count = 0;

        for (key, _) in entries {
            if freed_space >= needed_space && stats.entry_count < self.config.max_entries {
                break;
            }

            if let Some(entry) = storage.remove(&key) {
                freed_space += entry.size_bytes;
                stats.size_bytes = stats.size_bytes.saturating_sub(entry.size_bytes);
                stats.entry_count -= 1;
                stats.evictions += 1;
                evicted_count += 1;
            }
        }

        if evicted_count > 0 {
            debug!(
                "Evicted {} cache entries (freed {} bytes)",
                evicted_count, freed_space
            );
        }

        Ok(())
    }

    /// Load entry from persistent storage
    async fn load_from_disk(&self, key: &str) -> DspyResult<Option<T>> {
        let file_path = self.get_cache_file_path(key);

        if !file_path.exists() {
            return Ok(None);
        }

        match fs::read_to_string(&file_path) {
            Ok(content) => match serde_json::from_str::<T>(&content) {
                Ok(data) => Ok(Some(data)),
                Err(e) => {
                    warn!(
                        "Failed to deserialize cache file {}: {}",
                        file_path.display(),
                        e
                    );
                    Ok(None)
                }
            },
            Err(e) => {
                warn!("Failed to read cache file {}: {}", file_path.display(), e);
                Ok(None)
            }
        }
    }

    /// Save entry to persistent storage
    async fn save_to_disk(&self, key: &str, data: &T) -> DspyResult<()> {
        let file_path = self.get_cache_file_path(key);

        if let Some(parent) = file_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(DspyError::cache(
                    "disk_write",
                    &format!("Failed to create directory: {}", e),
                ));
            }
        }

        let content = serde_json::to_string(data).map_err(|e| {
            DspyError::cache("serialization", &format!("Failed to serialize: {}", e))
        })?;

        fs::write(&file_path, content)
            .map_err(|e| DspyError::cache("disk_write", &format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Get file path for cache key
    fn get_cache_file_path(&self, key: &str) -> PathBuf {
        // Use a hash of the key to avoid filesystem issues
        let hash = format!("{:x}", md5::compute(key.as_bytes()));
        self.config.cache_dir.join(format!("{}.json", hash))
    }

    /// Estimate size of data in bytes
    fn estimate_size(&self, data: &T) -> usize {
        // Simple estimation based on JSON serialization
        match serde_json::to_string(data) {
            Ok(json) => json.len(),
            Err(_) => 1024, // Default estimate
        }
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

impl<T> Default for Cache<T>
where
    T: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}
