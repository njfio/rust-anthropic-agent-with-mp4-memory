// Basic Cache Manager Tests
// Testing core cache functionality in small chunks

use super::{CacheConfig, CacheEntry, CacheManager, CacheMetadata, CompressionType, ConnectionStatus};
use crate::caching::memory_cache::{MemoryCache, MemoryCacheConfig, EvictionPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestData {
    id: u32,
    name: String,
    value: f64,
}

impl TestData {
    fn new(id: u32, name: &str, value: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            value,
        }
    }
}

// Basic Cache Manager Tests
#[tokio::test]
async fn test_cache_manager_creation() {
    let config = CacheConfig::default();
    let manager = CacheManager::new(config);

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
    assert_eq!(metrics.total_entries, 0);
}

#[tokio::test]
async fn test_cache_manager_with_defaults() {
    let manager = CacheManager::with_defaults();

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
}

#[tokio::test]
async fn test_cache_config_default() {
    let config = CacheConfig::default();

    assert!(config.enable_multi_tier);
    assert_eq!(config.default_ttl, 3600);
    assert_eq!(config.max_entry_size, 10 * 1024 * 1024);
    assert!(config.enable_compression);
    assert_eq!(config.compression_threshold, 1024);
    assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
}

#[tokio::test]
async fn test_cache_entry_creation() {
    let test_data = TestData::new(1, "test", 42.0);
    let data = serde_json::to_vec(&test_data).unwrap();

    let entry = CacheEntry {
        data: data.clone(),
        created_at: chrono::Utc::now(),
        ttl: Some(3600),
        metadata: CacheMetadata {
            size: data.len(),
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            tags: vec!["test".to_string()],
            priority: 5,
            source_tier: 1,
        },
        content_type: "application/json".to_string(),
        compression: Some(CompressionType::None),
    };

    assert_eq!(entry.metadata.size, data.len());
    assert_eq!(entry.ttl, Some(3600));
    assert_eq!(entry.metadata.tags, vec!["test"]);
}

#[tokio::test]
async fn test_compression_types() {
    assert_eq!(CompressionType::None, CompressionType::None);
    assert_eq!(CompressionType::Gzip, CompressionType::Gzip);
    assert_eq!(CompressionType::Lz4, CompressionType::Lz4);
    assert_eq!(CompressionType::Zstd, CompressionType::Zstd);
    assert_ne!(CompressionType::None, CompressionType::Gzip);
}

#[tokio::test]
async fn test_eviction_policies() {
    assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
    assert_eq!(EvictionPolicy::LFU, EvictionPolicy::LFU);
    assert_eq!(EvictionPolicy::TTL, EvictionPolicy::TTL);
    assert_eq!(EvictionPolicy::Random, EvictionPolicy::Random);
    assert_eq!(EvictionPolicy::Priority, EvictionPolicy::Priority);
    assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
}

// Memory Cache Tests
#[tokio::test]
async fn test_memory_cache_creation() {
    let config = MemoryCacheConfig::default();
    let cache = MemoryCache::new("test_cache".to_string(), config);

    assert_eq!(cache.name(), "test_cache");
    assert_eq!(cache.level(), 1);
}

#[tokio::test]
async fn test_memory_cache_with_defaults() {
    let cache = MemoryCache::with_defaults("test_cache".to_string());

    assert_eq!(cache.name(), "test_cache");
    assert_eq!(cache.level(), 1);
}

#[tokio::test]
async fn test_memory_cache_config_default() {
    let config = MemoryCacheConfig::default();

    assert_eq!(config.max_entries, 10000);
    assert_eq!(config.max_memory, 100 * 1024 * 1024);
    assert!(config.enable_ttl);
    assert_eq!(config.ttl_check_interval, 60);
    assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
    assert!(config.track_access);
}

#[tokio::test]
async fn test_memory_cache_basic_operations() {
    let config = MemoryCacheConfig::default();
    let cache = MemoryCache::new("test_cache".to_string(), config);

    let test_data = TestData::new(1, "test", 42.0);
    let data = serde_json::to_vec(&test_data).unwrap();

    let entry = CacheEntry {
        data: data.clone(),
        created_at: chrono::Utc::now(),
        ttl: Some(3600),
        metadata: CacheMetadata {
            size: data.len(),
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            tags: vec![],
            priority: 5,
            source_tier: 1,
        },
        content_type: "application/json".to_string(),
        compression: Some(CompressionType::None),
    };

    // Test set
    cache.set("test_key", entry.clone()).await.unwrap();

    // Test exists
    assert!(cache.exists("test_key").await.unwrap());
    assert!(!cache.exists("nonexistent_key").await.unwrap());

    // Test get
    let retrieved = cache.get("test_key").await.unwrap();
    assert!(retrieved.is_some());

    // Test delete
    assert!(cache.delete("test_key").await.unwrap());
    assert!(!cache.delete("nonexistent_key").await.unwrap());
}

#[tokio::test]
async fn test_memory_cache_stats() {
    let config = MemoryCacheConfig::default();
    let cache = MemoryCache::new("test_cache".to_string(), config);

    let stats = cache.stats().await.unwrap();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.memory_usage, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[tokio::test]
async fn test_memory_cache_health_check() {
    let config = MemoryCacheConfig::default();
    let cache = MemoryCache::new("test_cache".to_string(), config);

    let health = cache.health_check().await.unwrap();
    assert!(health.is_healthy);
    assert_eq!(health.connection_status, ConnectionStatus::Connected);
    assert_eq!(health.error_count, 0);
}

#[tokio::test]
async fn test_memory_cache_clear() {
    let config = MemoryCacheConfig::default();
    let cache = MemoryCache::new("test_cache".to_string(), config);

    let test_data = TestData::new(1, "test", 42.0);
    let data = serde_json::to_vec(&test_data).unwrap();

    let entry = CacheEntry {
        data: data.clone(),
        created_at: chrono::Utc::now(),
        ttl: Some(3600),
        metadata: CacheMetadata {
            size: data.len(),
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            tags: vec![],
            priority: 5,
            source_tier: 1,
        },
        content_type: "application/json".to_string(),
        compression: Some(CompressionType::None),
    };

    cache.set("test_key", entry).await.unwrap();
    assert!(cache.exists("test_key").await.unwrap());

    cache.clear().await.unwrap();
    assert!(!cache.exists("test_key").await.unwrap());
}