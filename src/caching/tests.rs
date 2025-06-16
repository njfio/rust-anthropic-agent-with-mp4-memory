// Basic Cache Manager Tests
// Testing core cache functionality in small chunks

use super::{
    CacheConfig, CacheEntry, CacheManager, CacheMetadata, CacheResult, CacheTier, CompressionType,
    ConnectionStatus,
};
use crate::caching::memory_cache::{EvictionPolicy, MemoryCache, MemoryCacheConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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
    assert_eq!(EvictionPolicy::FIFO, EvictionPolicy::FIFO);
    assert_eq!(EvictionPolicy::Random, EvictionPolicy::Random);
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

// Backend Tests
#[tokio::test]
async fn test_in_memory_data_source_creation() {
    use crate::caching::backends::InMemoryDataSource;
    use crate::caching::strategies::DataSource;

    let source = InMemoryDataSource::new();
    assert_eq!(source.count().await, 0);
    assert!(source.health_check().await.unwrap());
}

#[tokio::test]
async fn test_in_memory_data_source_with_latency() {
    use crate::caching::backends::InMemoryDataSource;

    let source = InMemoryDataSource::with_latency(10);
    assert_eq!(source.count().await, 0);
}

#[tokio::test]
async fn test_in_memory_data_source_operations() {
    use crate::caching::backends::InMemoryDataSource;
    use crate::caching::strategies::DataSource;

    let source = InMemoryDataSource::new();
    let test_data = TestData::new(1, "test", 42.0);

    // Test save
    let data = serde_json::to_vec(&test_data).unwrap();
    source.save_bytes("test_key", &data).await.unwrap();
    assert_eq!(source.count().await, 1);

    // Test load
    let loaded_data = source.load_bytes("test_key").await.unwrap();
    assert!(loaded_data.is_some());
    let loaded: TestData = serde_json::from_slice(&loaded_data.unwrap()).unwrap();
    assert_eq!(loaded, test_data);

    // Test load non-existent
    let missing = source.load_bytes("missing_key").await.unwrap();
    assert!(missing.is_none());

    // Test delete
    assert!(source.delete("test_key").await.unwrap());
    assert!(!source.delete("missing_key").await.unwrap());
    assert_eq!(source.count().await, 0);
}

#[tokio::test]
async fn test_mock_data_source_creation() {
    use crate::caching::backends::MockDataSource;
    use crate::caching::strategies::DataSource;

    let source = MockDataSource::new();
    assert!(source.health_check().await.unwrap());
}

#[tokio::test]
async fn test_mock_data_source_with_failure_rate() {
    use crate::caching::backends::MockDataSource;
    use crate::caching::strategies::DataSource;

    let source = MockDataSource::with_failure_rate(0.5);
    assert!(source.health_check().await.unwrap());
}

#[tokio::test]
async fn test_mock_data_source_health_control() {
    use crate::caching::backends::MockDataSource;
    use crate::caching::strategies::DataSource;

    let source = MockDataSource::new();
    assert!(source.health_check().await.unwrap());

    source.set_healthy(false).await;
    assert!(!source.health_check().await.unwrap());

    source.set_healthy(true).await;
    assert!(source.health_check().await.unwrap());
}

// Invalidation Tests
#[tokio::test]
async fn test_invalidation_manager_creation() {
    use crate::caching::invalidation::InvalidationManager;

    let manager = InvalidationManager::new();
    let stats = manager.get_stats().await;

    assert_eq!(stats.total_invalidations, 0);
    assert_eq!(stats.total_keys_invalidated, 0);
    assert_eq!(stats.failed_invalidations, 0);
}

#[tokio::test]
async fn test_invalidation_rule_creation() {
    use crate::caching::invalidation::{InvalidationRule, InvalidationType};
    use std::collections::HashMap;

    let rule = InvalidationRule {
        name: "test_rule".to_string(),
        rule_type: InvalidationType::Pattern,
        pattern: "test_*".to_string(),
        priority: 5,
        enabled: true,
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
    };

    assert_eq!(rule.name, "test_rule");
    assert_eq!(rule.rule_type, InvalidationType::Pattern);
    assert_eq!(rule.pattern, "test_*");
    assert!(rule.enabled);
}

#[tokio::test]
async fn test_invalidation_types() {
    use crate::caching::invalidation::InvalidationType;

    assert_eq!(InvalidationType::Pattern, InvalidationType::Pattern);
    assert_eq!(InvalidationType::Tag, InvalidationType::Tag);
    assert_eq!(InvalidationType::Time, InvalidationType::Time);
    assert_eq!(InvalidationType::Dependency, InvalidationType::Dependency);
    assert_eq!(InvalidationType::Manual, InvalidationType::Manual);
    assert_eq!(InvalidationType::Event, InvalidationType::Event);
    assert_ne!(InvalidationType::Pattern, InvalidationType::Tag);
}

#[tokio::test]
async fn test_dependency_types() {
    use crate::caching::invalidation::DependencyType;

    assert_eq!(DependencyType::Strong, DependencyType::Strong);
    assert_eq!(DependencyType::Weak, DependencyType::Weak);
    assert_eq!(DependencyType::Conditional, DependencyType::Conditional);
    assert_ne!(DependencyType::Strong, DependencyType::Weak);
}

// Policy Tests
#[tokio::test]
async fn test_cache_policy_default() {
    use crate::caching::policies::{CachePolicy, PolicyType};

    let policy = CachePolicy::default();

    assert_eq!(policy.name, "default");
    assert_eq!(policy.policy_type, PolicyType::MediumFrequency);
    assert_eq!(policy.priority, 5);
    assert!(policy.enabled);
    assert_eq!(policy.ttl_config.default_ttl, 3600);
}

#[tokio::test]
async fn test_cache_policy_high_frequency() {
    use crate::caching::policies::{CachePolicy, PolicyType};

    let policy = CachePolicy::high_frequency();

    assert_eq!(policy.name, "high_frequency");
    assert_eq!(policy.policy_type, PolicyType::HighFrequency);
    assert_eq!(policy.priority, 9);
    assert!(policy.enabled);
    assert_eq!(policy.ttl_config.default_ttl, 7200);
}

#[tokio::test]
async fn test_cache_policy_temporary() {
    use crate::caching::policies::{CachePolicy, PolicyType};

    let policy = CachePolicy::temporary();

    assert_eq!(policy.name, "temporary");
    assert_eq!(policy.policy_type, PolicyType::Temporary);
    assert_eq!(policy.priority, 2);
    assert!(policy.enabled);
    assert_eq!(policy.ttl_config.default_ttl, 300);
}

#[tokio::test]
async fn test_cache_policy_critical() {
    use crate::caching::policies::{CachePolicy, PolicyType};

    let policy = CachePolicy::critical();

    assert_eq!(policy.name, "critical");
    assert_eq!(policy.policy_type, PolicyType::Critical);
    assert_eq!(policy.priority, 10);
    assert!(policy.enabled);
    assert_eq!(policy.ttl_config.default_ttl, 1800);
}

#[tokio::test]
async fn test_policy_types() {
    use crate::caching::policies::PolicyType;

    assert_eq!(PolicyType::HighFrequency, PolicyType::HighFrequency);
    assert_eq!(PolicyType::MediumFrequency, PolicyType::MediumFrequency);
    assert_eq!(PolicyType::LowFrequency, PolicyType::LowFrequency);
    assert_eq!(PolicyType::Temporary, PolicyType::Temporary);
    assert_eq!(PolicyType::Critical, PolicyType::Critical);
    assert_eq!(PolicyType::LargeObject, PolicyType::LargeObject);
    assert_eq!(PolicyType::RealTime, PolicyType::RealTime);
    assert_eq!(PolicyType::Custom, PolicyType::Custom);
    assert_ne!(PolicyType::HighFrequency, PolicyType::LowFrequency);
}

// Metrics Tests
#[tokio::test]
async fn test_cache_metrics_collector_creation() {
    use crate::caching::metrics::{CacheMetricsCollector, MetricsConfig};

    let config = MetricsConfig::default();
    let collector = CacheMetricsCollector::new(config);

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.overall_stats.total_hits, 0);
    assert_eq!(metrics.overall_stats.total_misses, 0);
    assert_eq!(metrics.overall_stats.hit_ratio, 0.0);
}

#[tokio::test]
async fn test_cache_metrics_collector_with_defaults() {
    use crate::caching::metrics::CacheMetricsCollector;

    let collector = CacheMetricsCollector::with_defaults();

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.overall_stats.total_operations, 0);
    assert_eq!(metrics.overall_stats.total_entries, 0);
}

#[tokio::test]
async fn test_metrics_config_default() {
    use crate::caching::metrics::MetricsConfig;

    let config = MetricsConfig::default();

    assert!(config.enable_detailed_metrics);
    assert_eq!(config.collection_interval, 60);
    assert_eq!(config.history_retention_hours, 24);
    assert!(config.enable_performance_analysis);
    assert!(config.enable_cost_analysis);
    assert_eq!(config.latency_sample_rate, 0.1);
}

#[tokio::test]
async fn test_cache_metrics_record_operations() {
    use crate::caching::metrics::CacheMetricsCollector;
    use std::time::Duration;

    let collector = CacheMetricsCollector::with_defaults();

    // Record some operations
    collector
        .record_operation("get", Duration::from_millis(10), true)
        .await;
    collector
        .record_operation("set", Duration::from_millis(5), true)
        .await;
    collector
        .record_operation("get", Duration::from_millis(15), false)
        .await;

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.operation_stats.get_operations, 2);
    assert_eq!(metrics.operation_stats.set_operations, 1);
    assert_eq!(metrics.operation_stats.failed_operations, 1);
    assert_eq!(metrics.overall_stats.total_operations, 3);
}

#[tokio::test]
async fn test_cache_metrics_hit_miss_tracking() {
    use crate::caching::metrics::CacheMetricsCollector;

    let collector = CacheMetricsCollector::with_defaults();

    // Record hits and misses
    collector.record_hit_miss("tier1", true).await;
    collector.record_hit_miss("tier1", true).await;
    collector.record_hit_miss("tier1", false).await;
    collector.record_hit_miss("tier2", true).await;

    let metrics = collector.get_metrics().await;
    assert_eq!(metrics.overall_stats.total_hits, 3);
    assert_eq!(metrics.overall_stats.total_misses, 1);
    assert_eq!(metrics.overall_stats.hit_ratio, 75.0);

    // Check tier-specific metrics
    assert!(metrics.tier_stats.contains_key("tier1"));
    assert!(metrics.tier_stats.contains_key("tier2"));

    let tier1_stats = &metrics.tier_stats["tier1"];
    assert_eq!(tier1_stats.hits, 2);
    assert_eq!(tier1_stats.misses, 1);
    assert!((tier1_stats.hit_ratio - 66.66666666666667).abs() < 0.0001);
}

// Strategy Tests
#[tokio::test]
async fn test_strategy_types() {
    use crate::caching::strategies::StrategyType;

    assert_eq!(StrategyType::WriteThrough, StrategyType::WriteThrough);
    assert_eq!(StrategyType::WriteBehind, StrategyType::WriteBehind);
    assert_eq!(StrategyType::CacheAside, StrategyType::CacheAside);
    assert_eq!(StrategyType::ReadThrough, StrategyType::ReadThrough);
    assert_eq!(StrategyType::RefreshAhead, StrategyType::RefreshAhead);
    assert_eq!(StrategyType::CircuitBreaker, StrategyType::CircuitBreaker);
    assert_eq!(StrategyType::Bulkhead, StrategyType::Bulkhead);
    assert_ne!(StrategyType::WriteThrough, StrategyType::WriteBehind);
}

#[tokio::test]
async fn test_write_operation_types() {
    use crate::caching::strategies::WriteOperationType;

    assert_eq!(WriteOperationType::Insert, WriteOperationType::Insert);
    assert_eq!(WriteOperationType::Update, WriteOperationType::Update);
    assert_eq!(WriteOperationType::Delete, WriteOperationType::Delete);
    assert_ne!(WriteOperationType::Insert, WriteOperationType::Update);
}

#[tokio::test]
async fn test_circuit_breaker_states() {
    use crate::caching::strategies::CircuitState;

    assert_eq!(CircuitState::Closed, CircuitState::Closed);
    assert_eq!(CircuitState::Open, CircuitState::Open);
    assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
    assert_ne!(CircuitState::Closed, CircuitState::Open);
}

// Strategy tests temporarily disabled due to trait refactoring
// TODO: Re-enable after completing CacheStrategy trait dyn compatibility

// Integration Tests
#[tokio::test]
async fn test_cache_manager_with_memory_tier() {
    let mut manager = CacheManager::with_defaults();
    let memory_cache = Arc::new(MemoryCache::with_defaults("memory_l1".to_string()));

    manager.add_tier(memory_cache).await.unwrap();

    let test_data = TestData::new(1, "integration_test", 99.9);

    // Test set operation
    manager
        .set("integration_key", &test_data, Some(3600))
        .await
        .unwrap();

    // Test get operation
    let result: crate::caching::CacheResult<TestData> =
        manager.get("integration_key").await.unwrap();
    assert!(result.hit);
    assert!(result.value.is_some());
    assert_eq!(result.value.unwrap(), test_data);
    assert_eq!(result.source_tier, Some("memory_l1".to_string()));

    // Test exists
    assert!(manager.exists("integration_key").await.unwrap());
    assert!(!manager.exists("nonexistent_key").await.unwrap());

    // Test delete
    assert!(manager.delete("integration_key").await.unwrap());
    assert!(!manager.exists("integration_key").await.unwrap());
}

#[tokio::test]
async fn test_cache_manager_metrics_integration() {
    let mut manager = CacheManager::with_defaults();
    let memory_cache = Arc::new(MemoryCache::with_defaults("memory_l1".to_string()));

    manager.add_tier(memory_cache).await.unwrap();

    let test_data = TestData::new(1, "metrics_test", 42.0);

    // Perform operations to generate metrics
    manager
        .set("metrics_key", &test_data, Some(3600))
        .await
        .unwrap();
    let _result: crate::caching::CacheResult<TestData> = manager.get("metrics_key").await.unwrap();
    let _miss_result: crate::caching::CacheResult<TestData> =
        manager.get("missing_key").await.unwrap();

    let metrics = manager.get_metrics().await;
    assert!(metrics.hits > 0);
    assert!(metrics.misses > 0);
    assert!(metrics.hit_ratio > 0.0);
    assert!(metrics.tier_metrics.contains_key("memory_l1"));
}

#[tokio::test]
async fn test_cache_manager_health_check() {
    let mut manager = CacheManager::with_defaults();
    let memory_cache = Arc::new(MemoryCache::with_defaults("memory_l1".to_string()));

    manager.add_tier(memory_cache).await.unwrap();

    let health = manager.get_health().await.unwrap();
    assert!(health.overall_healthy);
    assert_eq!(health.total_tiers, 1);
    assert_eq!(health.healthy_tiers, 1);
    assert!(health.tier_health.contains_key("memory_l1"));
}

#[tokio::test]
async fn test_cache_manager_clear_all() {
    let mut manager = CacheManager::with_defaults();
    let memory_cache = Arc::new(MemoryCache::with_defaults("memory_l1".to_string()));

    manager.add_tier(memory_cache).await.unwrap();

    let test_data = TestData::new(1, "clear_test", 123.0);
    manager
        .set("clear_key", &test_data, Some(3600))
        .await
        .unwrap();

    assert!(manager.exists("clear_key").await.unwrap());

    manager.clear_all().await.unwrap();

    assert!(!manager.exists("clear_key").await.unwrap());

    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.hits, 0);
    assert_eq!(metrics.misses, 0);
}

// Cache Strategy Integration Tests
#[tokio::test]
async fn test_cache_manager_with_strategies() {
    let manager = CacheManager::with_default_strategies().await.unwrap();

    // Verify strategies were added
    let strategies = manager.get_strategies().await;
    assert!(strategies.contains_key("cache_aside"));
    assert!(strategies.contains_key("write_through"));
    assert!(strategies.contains_key("read_through"));
    assert!(strategies.contains_key("refresh_ahead"));

    // Verify active strategy
    let active = manager.get_active_strategy().await;
    assert!(active.is_some());
    assert_eq!(active.unwrap().name(), "cache_aside");
}

#[tokio::test]
async fn test_strategy_management() {
    let mut manager = CacheManager::with_defaults();

    // Add a strategy
    let source = Arc::new(crate::caching::strategies::MemoryDataSource::new(
        "test".to_string(),
    ));
    let strategy = Arc::new(crate::caching::strategies::CacheAsideStrategy::new(
        "test_strategy".to_string(),
        source,
    ));

    manager.add_strategy(strategy).await.unwrap();

    // Verify strategy was added
    let strategies = manager.get_strategies().await;
    assert!(strategies.contains_key("test_strategy"));

    // Set as active
    manager.set_active_strategy("test_strategy").await.unwrap();
    let active = manager.get_active_strategy().await;
    assert!(active.is_some());
    assert_eq!(active.unwrap().name(), "test_strategy");

    // Remove strategy
    manager.remove_strategy("test_strategy").await.unwrap();
    let strategies = manager.get_strategies().await;
    assert!(!strategies.contains_key("test_strategy"));

    // Active strategy should be cleared
    let active = manager.get_active_strategy().await;
    assert!(active.is_none());
}

#[tokio::test]
async fn test_strategy_aware_operations() {
    let manager = CacheManager::with_default_strategies().await.unwrap();

    // Add memory cache tier for testing
    let memory_cache = Arc::new(crate::caching::memory_cache::MemoryCache::new(
        "test_memory".to_string(),
        crate::caching::memory_cache::MemoryCacheConfig::default(),
    ));
    let mut manager = manager;
    manager.add_tier(memory_cache).await.unwrap();

    // Test strategy-aware get (cache miss)
    let result: CacheResult<TestData> = manager.get_with_strategy("test_key").await.unwrap();
    assert!(!result.hit);
    assert!(result.value.is_none());

    // Test strategy-aware set
    let test_value = TestData::new(1, "strategy_test", 42.0);
    manager
        .set_with_strategy("test_key", &test_value, Some(3600))
        .await
        .unwrap();

    // Verify value was cached
    let result: CacheResult<TestData> = manager.get("test_key").await.unwrap();
    assert!(result.hit);
    assert_eq!(result.value.unwrap(), test_value);
}

#[tokio::test]
async fn test_strategy_error_handling() {
    let mut manager = CacheManager::with_defaults();

    // Try to set non-existent strategy as active
    let result = manager.set_active_strategy("non_existent").await;
    assert!(result.is_err());

    // Try to remove non-existent strategy
    let result = manager.remove_strategy("non_existent").await;
    assert!(result.is_err());
}
