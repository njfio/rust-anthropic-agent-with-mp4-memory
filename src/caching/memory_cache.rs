// In-Memory Cache Implementation (L1 Cache Tier)
// Provides high-performance local caching with LRU eviction and TTL support

use super::{CacheEntry, CacheTier, ConnectionStatus, TierHealth, TierStats};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// In-memory cache tier (L1)
pub struct MemoryCache {
    /// Cache name
    name: String,
    /// Cache storage
    storage: Arc<RwLock<CacheStorage>>,
    /// Cache configuration
    config: MemoryCacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<MemoryCacheStats>>,
    /// Start time for uptime tracking
    start_time: Instant,
}

/// Memory cache storage
#[derive(Debug)]
struct CacheStorage {
    /// Key-value storage
    entries: HashMap<String, StoredEntry>,
    /// LRU access order
    access_order: VecDeque<String>,
    /// Current memory usage in bytes
    memory_usage: usize,
}

/// Stored cache entry with access tracking
#[derive(Debug, Clone)]
struct StoredEntry {
    /// Cache entry data
    entry: CacheEntry,
    /// Last access time for LRU
    last_accessed: Instant,
    /// Access count for LFU
    access_count: u64,
}

/// Memory cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Enable TTL checking
    pub enable_ttl: bool,
    /// TTL check interval in seconds
    pub ttl_check_interval: u64,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable access tracking
    pub track_access: bool,
}

/// Eviction policies for memory cache
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Random eviction
    Random,
}

/// Memory cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCacheStats {
    /// Total hits
    pub hits: u64,
    /// Total misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total expired entries
    pub expired_entries: u64,
    /// Current entry count
    pub entry_count: usize,
    /// Current memory usage
    pub memory_usage: usize,
    /// Average access time in nanoseconds
    pub avg_access_time: u64,
    /// Last cleanup time
    pub last_cleanup: Option<DateTime<Utc>>,
}

impl Default for MemoryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory: 100 * 1024 * 1024, // 100MB
            enable_ttl: true,
            ttl_check_interval: 60, // 1 minute
            eviction_policy: EvictionPolicy::LRU,
            track_access: true,
        }
    }
}

impl Default for MemoryCacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            expired_entries: 0,
            entry_count: 0,
            memory_usage: 0,
            avg_access_time: 0,
            last_cleanup: None,
        }
    }
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new(name: String, config: MemoryCacheConfig) -> Self {
        Self {
            name,
            storage: Arc::new(RwLock::new(CacheStorage {
                entries: HashMap::new(),
                access_order: VecDeque::new(),
                memory_usage: 0,
            })),
            config,
            stats: Arc::new(RwLock::new(MemoryCacheStats::default())),
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(name: String) -> Self {
        Self::new(name, MemoryCacheConfig::default())
    }

    /// Start background cleanup task
    pub async fn start_cleanup_task(&self) -> Result<()> {
        if !self.config.enable_ttl {
            return Ok(());
        }

        let storage = Arc::clone(&self.storage);
        let stats = Arc::clone(&self.stats);
        let interval = self.config.ttl_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(
                tokio::time::Duration::from_secs(interval)
            );

            loop {
                interval_timer.tick().await;
                
                let expired_keys = {
                    let storage_guard = storage.read().await;
                    let now = Instant::now();
                    
                    storage_guard.entries
                        .iter()
                        .filter_map(|(key, stored_entry)| {
                            if Self::is_entry_expired(&stored_entry.entry, now) {
                                Some(key.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                };

                if !expired_keys.is_empty() {
                    let mut storage_guard = storage.write().await;
                    let mut stats_guard = stats.write().await;
                    
                    for key in expired_keys {
                        if let Some(stored_entry) = storage_guard.entries.remove(&key) {
                            storage_guard.memory_usage = storage_guard.memory_usage
                                .saturating_sub(stored_entry.entry.metadata.size);
                            stats_guard.expired_entries += 1;
                            
                            // Remove from access order
                            if let Some(pos) = storage_guard.access_order.iter().position(|k| k == &key) {
                                storage_guard.access_order.remove(pos);
                            }
                        }
                    }
                    
                    stats_guard.entry_count = storage_guard.entries.len();
                    stats_guard.memory_usage = storage_guard.memory_usage;
                    stats_guard.last_cleanup = Some(Utc::now());
                }
            }
        });

        debug!("Started TTL cleanup task for memory cache: {}", self.name);
        Ok(())
    }

    /// Check if entry is expired
    fn is_entry_expired(entry: &CacheEntry, now: Instant) -> bool {
        if let Some(ttl) = entry.ttl {
            let created_instant = now - std::time::Duration::from_secs(
                Utc::now().signed_duration_since(entry.created_at).num_seconds() as u64
            );
            now.duration_since(created_instant).as_secs() > ttl
        } else {
            false
        }
    }

    /// Evict entries based on policy
    async fn evict_if_needed(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        // Check if eviction is needed
        let needs_eviction = storage.entries.len() >= self.config.max_entries
            || storage.memory_usage >= self.config.max_memory;

        if !needs_eviction {
            return Ok(());
        }

        let evict_count = std::cmp::max(
            storage.entries.len().saturating_sub(self.config.max_entries * 9 / 10),
            if storage.memory_usage >= self.config.max_memory { 1 } else { 0 }
        );

        let keys_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                storage.access_order.iter()
                    .take(evict_count)
                    .cloned()
                    .collect::<Vec<_>>()
            }
            EvictionPolicy::LFU => {
                let mut entries: Vec<_> = storage.entries.iter().collect();
                entries.sort_by_key(|(_, stored)| stored.access_count);
                entries.into_iter()
                    .take(evict_count)
                    .map(|(key, _)| key.clone())
                    .collect()
            }
            EvictionPolicy::FIFO => {
                storage.access_order.iter()
                    .take(evict_count)
                    .cloned()
                    .collect::<Vec<_>>()
            }
            EvictionPolicy::Random => {
                use rand::seq::SliceRandom;
                let mut keys: Vec<_> = storage.entries.keys().cloned().collect();
                keys.shuffle(&mut rand::thread_rng());
                keys.into_iter().take(evict_count).collect()
            }
        };

        // Perform eviction
        for key in keys_to_evict {
            if let Some(stored_entry) = storage.entries.remove(&key) {
                storage.memory_usage = storage.memory_usage
                    .saturating_sub(stored_entry.entry.metadata.size);
                stats.evictions += 1;
                
                // Remove from access order
                if let Some(pos) = storage.access_order.iter().position(|k| k == &key) {
                    storage.access_order.remove(pos);
                }
            }
        }

        stats.entry_count = storage.entries.len();
        stats.memory_usage = storage.memory_usage;

        debug!("Evicted {} entries from memory cache: {}", stats.evictions, self.name);
        Ok(())
    }

    /// Update access tracking
    fn update_access_tracking(&self, storage: &mut CacheStorage, key: &str, stored_entry: &mut StoredEntry) {
        if !self.config.track_access {
            return;
        }

        stored_entry.last_accessed = Instant::now();
        stored_entry.access_count += 1;

        // Update LRU order
        if let Some(pos) = storage.access_order.iter().position(|k| k == key) {
            storage.access_order.remove(pos);
        }
        storage.access_order.push_back(key.to_string());
    }
}

#[async_trait::async_trait]
impl CacheTier for MemoryCache {
    fn name(&self) -> &str {
        &self.name
    }

    fn level(&self) -> u8 {
        1 // L1 cache (fastest)
    }

    async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        let start_time = Instant::now();
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        // Check if entry exists
        if let Some(stored_entry) = storage.entries.get(key) {
            // Check if expired
            if Self::is_entry_expired(&stored_entry.entry, start_time) {
                // Entry is expired, remove it
                let key_to_remove = key.to_string();
                if let Some(removed_entry) = storage.entries.remove(&key_to_remove) {
                    storage.memory_usage = storage.memory_usage
                        .saturating_sub(removed_entry.entry.metadata.size);
                    stats.expired_entries += 1;

                    // Remove from access order
                    if let Some(pos) = storage.access_order.iter().position(|k| k == &key_to_remove) {
                        storage.access_order.remove(pos);
                    }
                }
                stats.misses += 1;
                return Ok(None);
            }

            // Entry is valid, clone it first
            let entry = stored_entry.entry.clone();

            // Update access tracking manually to avoid borrow checker issues
            if self.config.track_access {
                if let Some(stored_entry_mut) = storage.entries.get_mut(key) {
                    stored_entry_mut.last_accessed = std::time::Instant::now();
                    stored_entry_mut.access_count += 1;
                }

                // Update LRU order
                if let Some(pos) = storage.access_order.iter().position(|k| k == key) {
                    storage.access_order.remove(pos);
                }
                storage.access_order.push_back(key.to_string());
            }

            // Update statistics
            stats.hits += 1;
            let access_time = start_time.elapsed().as_nanos() as u64;
            stats.avg_access_time = (stats.avg_access_time + access_time) / 2;

            Ok(Some(entry))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    async fn set(&self, key: &str, entry: CacheEntry) -> Result<()> {
        // Check entry size
        if entry.metadata.size > self.config.max_memory {
            return Err(AgentError::validation("Entry too large for memory cache".to_string()));
        }

        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        // Remove existing entry if present
        if let Some(existing) = storage.entries.remove(key) {
            storage.memory_usage = storage.memory_usage
                .saturating_sub(existing.entry.metadata.size);
            
            // Remove from access order
            if let Some(pos) = storage.access_order.iter().position(|k| k == key) {
                storage.access_order.remove(pos);
            }
        }

        // Create stored entry
        let stored_entry = StoredEntry {
            entry: entry.clone(),
            last_accessed: Instant::now(),
            access_count: 0,
        };

        // Add to storage
        storage.entries.insert(key.to_string(), stored_entry);
        storage.memory_usage += entry.metadata.size;
        storage.access_order.push_back(key.to_string());

        // Update statistics
        stats.entry_count = storage.entries.len();
        stats.memory_usage = storage.memory_usage;

        // Release locks before eviction
        drop(storage);
        drop(stats);

        // Check if eviction is needed
        self.evict_if_needed().await?;

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        if let Some(stored_entry) = storage.entries.remove(key) {
            storage.memory_usage = storage.memory_usage
                .saturating_sub(stored_entry.entry.metadata.size);
            
            // Remove from access order
            if let Some(pos) = storage.access_order.iter().position(|k| k == key) {
                storage.access_order.remove(pos);
            }

            stats.entry_count = storage.entries.len();
            stats.memory_usage = storage.memory_usage;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let storage = self.storage.read().await;
        Ok(storage.entries.contains_key(key))
    }

    async fn stats(&self) -> Result<TierStats> {
        let storage = self.storage.read().await;
        let stats = self.stats.read().await;

        Ok(TierStats {
            entry_count: storage.entries.len() as u64,
            memory_usage: storage.memory_usage as u64,
            hits: stats.hits,
            misses: stats.misses,
            avg_access_time: stats.avg_access_time as f64 / 1_000_000.0, // Convert to milliseconds
            last_access: stats.last_cleanup,
        })
    }

    async fn clear(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        let mut stats = self.stats.write().await;

        storage.entries.clear();
        storage.access_order.clear();
        storage.memory_usage = 0;

        stats.entry_count = 0;
        stats.memory_usage = 0;

        Ok(())
    }

    async fn health_check(&self) -> Result<TierHealth> {
        let storage = self.storage.read().await;
        let _stats = self.stats.read().await;

        let memory_usage_ratio = storage.memory_usage as f64 / self.config.max_memory as f64;
        let entry_usage_ratio = storage.entries.len() as f64 / self.config.max_entries as f64;
        
        let health_score = ((1.0 - memory_usage_ratio.max(entry_usage_ratio)) * 100.0) as u8;
        let is_healthy = health_score > 20; // Healthy if less than 80% capacity

        Ok(TierHealth {
            is_healthy,
            health_score,
            last_check: Utc::now(),
            error_count: 0, // Memory cache doesn't have connection errors
            connection_status: ConnectionStatus::Connected,
        })
    }
}
