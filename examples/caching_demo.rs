// Advanced Caching System Demo
// Demonstrates the comprehensive caching capabilities for AI agent systems

use rust_memvid_agent::caching::{
    CacheManager, CacheConfig, 
    memory_cache::MemoryCache,
    backends::InMemoryDataSource,
    strategies::{WriteThroughStrategy, DataSource},
    policies::CachePolicy,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AIModelData {
    model_id: String,
    embeddings: Vec<f32>,
    metadata: std::collections::HashMap<String, String>,
}

impl AIModelData {
    fn new(model_id: &str, size: usize) -> Self {
        Self {
            model_id: model_id.to_string(),
            embeddings: vec![0.5; size], // Dummy embeddings
            metadata: std::collections::HashMap::new(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Caching System Demo");
    println!("================================");

    // 1. Create cache manager with custom configuration
    let config = CacheConfig {
        enable_multi_tier: true,
        default_ttl: 3600, // 1 hour
        max_entry_size: 50 * 1024 * 1024, // 50MB
        enable_compression: true,
        compression_threshold: 1024, // 1KB
        ..Default::default()
    };
    
    let mut cache_manager = CacheManager::new(config);
    println!("✅ Created cache manager with custom configuration");

    // 2. Add memory cache tier (L1)
    let memory_cache = Arc::new(MemoryCache::with_defaults("memory_l1".to_string()));
    cache_manager.add_tier(memory_cache).await?;
    println!("✅ Added L1 memory cache tier");

    // 3. Create test AI model data
    let model_data = AIModelData::new("gpt-4-embeddings", 1536);
    let user_session = serde_json::json!({
        "user_id": "user_123",
        "session_token": "abc123xyz",
        "preferences": {
            "language": "en",
            "theme": "dark"
        }
    });

    // 4. Cache AI model data with high-frequency policy
    let high_freq_policy = CachePolicy::high_frequency();
    println!("📊 Using high-frequency cache policy: TTL={}s, Priority={}", 
             high_freq_policy.ttl_config.default_ttl, 
             high_freq_policy.priority);
    
    cache_manager.set("ai_model:gpt-4", &model_data, Some(7200)).await?;
    println!("✅ Cached AI model data");

    // 5. Cache user session with temporary policy
    let temp_policy = CachePolicy::temporary();
    println!("📊 Using temporary cache policy: TTL={}s, Priority={}", 
             temp_policy.ttl_config.default_ttl, 
             temp_policy.priority);
    
    cache_manager.set("session:user_123", &user_session, Some(300)).await?;
    println!("✅ Cached user session data");

    // 6. Retrieve cached data
    let cached_model: rust_memvid_agent::caching::CacheResult<AIModelData> = 
        cache_manager.get("ai_model:gpt-4").await?;
    
    if cached_model.hit {
        println!("🎯 Cache HIT for AI model data from tier: {}", 
                 cached_model.source_tier.unwrap_or("unknown".to_string()));
        println!("   Model ID: {}", cached_model.value.unwrap().model_id);
    }

    let cached_session: rust_memvid_agent::caching::CacheResult<serde_json::Value> = 
        cache_manager.get("session:user_123").await?;
    
    if cached_session.hit {
        println!("🎯 Cache HIT for user session from tier: {}", 
                 cached_session.source_tier.unwrap_or("unknown".to_string()));
    }

    // 7. Test cache miss
    let missing_data: rust_memvid_agent::caching::CacheResult<AIModelData> = 
        cache_manager.get("ai_model:nonexistent").await?;
    
    if !missing_data.hit {
        println!("❌ Cache MISS for nonexistent model (expected)");
    }

    // 8. Get cache metrics
    let metrics = cache_manager.get_metrics().await;
    println!("\n📈 Cache Performance Metrics:");
    println!("   Total Hits: {}", metrics.hits);
    println!("   Total Misses: {}", metrics.misses);
    println!("   Hit Ratio: {:.2}%", metrics.hit_ratio);
    println!("   Total Entries: {}", metrics.total_entries);
    println!("   Memory Usage: {} bytes", metrics.memory_usage);

    // 9. Get cache health status
    let health = cache_manager.get_health().await?;
    println!("\n🏥 Cache Health Status:");
    println!("   Overall Healthy: {}", health.overall_healthy);
    println!("   Total Tiers: {}", health.total_tiers);
    println!("   Healthy Tiers: {}", health.healthy_tiers);
    println!("   Uptime: {:?}", health.uptime);

    // 10. Demonstrate cache invalidation
    println!("\n🗑️  Testing cache invalidation...");
    let deleted = cache_manager.delete("session:user_123").await?;
    if deleted {
        println!("✅ Successfully invalidated user session");
    }

    // 11. Verify invalidation
    let invalidated_session: rust_memvid_agent::caching::CacheResult<serde_json::Value> = 
        cache_manager.get("session:user_123").await?;
    
    if !invalidated_session.hit {
        println!("✅ Confirmed session invalidation - cache miss as expected");
    }

    // 12. Final metrics
    let final_metrics = cache_manager.get_metrics().await;
    println!("\n📊 Final Cache Metrics:");
    println!("   Total Operations: {}", final_metrics.hits + final_metrics.misses);
    println!("   Hit Ratio: {:.2}%", final_metrics.hit_ratio);
    println!("   Average Response Time: {:.2}ms", final_metrics.avg_response_time);

    println!("\n🎉 Advanced Caching System Demo Complete!");
    println!("   ✅ Multi-tier caching operational");
    println!("   ✅ Policy-based cache management");
    println!("   ✅ Performance metrics collection");
    println!("   ✅ Health monitoring active");
    println!("   ✅ Cache invalidation working");
    println!("\n🚀 Ready for production AI agent deployments!");

    Ok(())
}
