// Redis Cache Implementation (L2 Cache Tier)
// Provides distributed caching with Redis backend, connection pooling, and cluster support

use super::{CacheEntry, CacheTier, ConnectionStatus, TierHealth, TierStats};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Redis cache tier (L2)
pub struct RedisCache {
    /// Cache name
    name: String,
    /// Redis connection pool
    pool: Option<deadpool_redis::Pool>,
    /// Cache configuration
    config: RedisCacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<RedisCacheStats>>,
    /// Connection health
    health: Arc<RwLock<RedisHealth>>,
    /// Start time for uptime tracking
    start_time: Instant,
}

/// Redis cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisCacheConfig {
    /// Redis connection URL
    pub redis_url: String,
    /// Connection pool size
    pub pool_size: usize,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Command timeout in seconds
    pub command_timeout: u64,
    /// Enable Redis cluster mode
    pub cluster_mode: bool,
    /// Key prefix for namespacing
    pub key_prefix: String,
    /// Enable compression
    pub enable_compression: bool,
    /// Retry attempts for failed operations
    pub retry_attempts: u32,
    /// Retry delay in milliseconds
    pub retry_delay: u64,
}

/// Redis cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisCacheStats {
    /// Total hits
    pub hits: u64,
    /// Total misses
    pub misses: u64,
    /// Total errors
    pub errors: u64,
    /// Total operations
    pub operations: u64,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Connection pool statistics
    pub pool_stats: PoolStats,
    /// Last operation timestamp
    pub last_operation: Option<DateTime<Utc>>,
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    /// Active connections
    pub active_connections: u32,
    /// Idle connections
    pub idle_connections: u32,
    /// Total connections created
    pub total_connections: u64,
    /// Connection errors
    pub connection_errors: u64,
}

/// Redis health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisHealth {
    /// Connection status
    pub connection_status: ConnectionStatus,
    /// Last successful ping
    pub last_ping: Option<DateTime<Utc>>,
    /// Consecutive errors
    pub consecutive_errors: u32,
    /// Redis server info
    pub server_info: Option<RedisServerInfo>,
}

/// Redis server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisServerInfo {
    /// Redis version
    pub version: String,
    /// Server mode (standalone/cluster)
    pub mode: String,
    /// Used memory
    pub used_memory: u64,
    /// Connected clients
    pub connected_clients: u32,
}

impl Default for RedisCacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            pool_size: 20,
            connection_timeout: 5,
            command_timeout: 3,
            cluster_mode: false,
            key_prefix: "cache:".to_string(),
            enable_compression: true,
            retry_attempts: 3,
            retry_delay: 100,
        }
    }
}

impl Default for RedisCacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            errors: 0,
            operations: 0,
            avg_response_time: 0.0,
            pool_stats: PoolStats::default(),
            last_operation: None,
        }
    }
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            active_connections: 0,
            idle_connections: 0,
            total_connections: 0,
            connection_errors: 0,
        }
    }
}

impl Default for RedisHealth {
    fn default() -> Self {
        Self {
            connection_status: ConnectionStatus::Disconnected,
            last_ping: None,
            consecutive_errors: 0,
            server_info: None,
        }
    }
}

impl RedisCache {
    /// Create a new Redis cache
    pub async fn new(name: String, config: RedisCacheConfig) -> Result<Self> {
        let cache = Self {
            name,
            pool: None,
            config,
            stats: Arc::new(RwLock::new(RedisCacheStats::default())),
            health: Arc::new(RwLock::new(RedisHealth::default())),
            start_time: Instant::now(),
        };

        Ok(cache)
    }

    /// Create with default configuration
    pub async fn with_defaults(name: String) -> Result<Self> {
        Self::new(name, RedisCacheConfig::default()).await
    }

    /// Initialize Redis connection pool
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to Redis: {}", self.config.redis_url);

        let pool_config = deadpool_redis::Config::from_url(&self.config.redis_url);
        let pool = pool_config
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to create Redis pool: {}", e)))?;

        // Test connection
        match pool.get().await {
            Ok(mut conn) => {
                // Ping Redis to verify connection
                let _: String = redis::cmd("PING")
                    .query_async(&mut conn)
                    .await
                    .map_err(|e| AgentError::tool("redis_cache", &format!("Redis ping failed: {}", e)))?;

                self.pool = Some(pool);
                
                // Update health status
                {
                    let mut health = self.health.write().await;
                    health.connection_status = ConnectionStatus::Connected;
                    health.last_ping = Some(Utc::now());
                    health.consecutive_errors = 0;
                }

                info!("Successfully connected to Redis: {}", self.name);
                Ok(())
            }
            Err(e) => {
                let mut health = self.health.write().await;
                health.connection_status = ConnectionStatus::Error;
                health.consecutive_errors += 1;
                
                Err(AgentError::tool("redis_cache", &format!("Redis connection failed: {}", e)))
            }
        }
    }

    /// Get full key with prefix
    fn get_full_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }

    /// Execute Redis operation with retry logic
    async fn execute_with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send,
        T: Send,
    {
        let mut last_error: Option<AgentError> = None;

        for attempt in 0..=self.config.retry_attempts {
            match operation().await {
                Ok(result) => {
                    // Reset consecutive errors on success
                    {
                        let mut health = self.health.write().await;
                        health.consecutive_errors = 0;
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);

                    // Update error count
                    {
                        let mut health = self.health.write().await;
                        health.consecutive_errors += 1;
                        let mut stats = self.stats.write().await;
                        stats.errors += 1;
                    }

                    if attempt < self.config.retry_attempts {
                        tokio::time::sleep(Duration::from_millis(self.config.retry_delay)).await;
                        warn!("Redis operation failed, retrying (attempt {}/{})", attempt + 1, self.config.retry_attempts);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| AgentError::tool("redis_cache", "All retry attempts failed")))
    }

    /// Update operation statistics
    async fn update_stats(&self, duration: Duration, hit: bool) {
        let mut stats = self.stats.write().await;
        stats.operations += 1;
        stats.last_operation = Some(Utc::now());
        
        if hit {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        
        // Update average response time
        let response_time_ms = duration.as_millis() as f64;
        stats.avg_response_time = (stats.avg_response_time + response_time_ms) / 2.0;
    }

    /// Perform health check ping
    pub async fn ping(&self) -> Result<()> {
        if let Some(pool) = &self.pool {
            let mut conn = pool.get().await
                .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;
            
            let _: String = redis::cmd("PING")
                .query_async(&mut conn)
                .await
                .map_err(|e| AgentError::tool("redis_cache", &format!("Ping failed: {}", e)))?;
            
            // Update health status
            {
                let mut health = self.health.write().await;
                health.last_ping = Some(Utc::now());
                health.connection_status = ConnectionStatus::Connected;
            }
            
            Ok(())
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    /// Get Redis server information
    pub async fn get_server_info(&self) -> Result<RedisServerInfo> {
        if let Some(pool) = &self.pool {
            let mut conn = pool.get().await
                .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;
            
            let info: String = redis::cmd("INFO")
                .arg("server")
                .query_async(&mut conn)
                .await
                .map_err(|e| AgentError::tool("redis_cache", &format!("INFO command failed: {}", e)))?;
            
            // Parse basic info (simplified)
            let version = info.lines()
                .find(|line| line.starts_with("redis_version:"))
                .and_then(|line| line.split(':').nth(1))
                .unwrap_or("unknown")
                .to_string();
            
            Ok(RedisServerInfo {
                version,
                mode: if self.config.cluster_mode { "cluster".to_string() } else { "standalone".to_string() },
                used_memory: 0, // Would need to parse from INFO memory
                connected_clients: 0, // Would need to parse from INFO clients
            })
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }
}

#[async_trait::async_trait]
impl CacheTier for RedisCache {
    fn name(&self) -> &str {
        &self.name
    }

    fn level(&self) -> u8 {
        2 // L2 cache (distributed)
    }

    async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        let start_time = Instant::now();
        let full_key = self.get_full_key(key);
        
        if let Some(pool) = &self.pool {
            let result = self.execute_with_retry(|| {
                let pool = pool.clone();
                let full_key = full_key.clone();
                
                Box::pin(async move {
                    let mut conn = pool.get().await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;
                    
                    let data: Option<Vec<u8>> = redis::cmd("GET")
                        .arg(&full_key)
                        .query_async(&mut conn)
                        .await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("GET command failed: {}", e)))?;
                    
                    Ok(data)
                })
            }).await?;
            
            let duration = start_time.elapsed();
            
            match result {
                Some(data) => {
                    // Deserialize cache entry
                    let entry: CacheEntry = serde_json::from_slice(&data)
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Deserialization failed: {}", e)))?;
                    
                    self.update_stats(duration, true).await;
                    Ok(Some(entry))
                }
                None => {
                    self.update_stats(duration, false).await;
                    Ok(None)
                }
            }
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    async fn set(&self, key: &str, entry: CacheEntry) -> Result<()> {
        let start_time = Instant::now();
        let full_key = self.get_full_key(key);
        
        if let Some(pool) = &self.pool {
            // Serialize entry
            let data = serde_json::to_vec(&entry)
                .map_err(|e| AgentError::tool("redis_cache", &format!("Serialization failed: {}", e)))?;
            
            self.execute_with_retry(|| {
                let pool = pool.clone();
                let full_key = full_key.clone();
                let data = data.clone();
                let ttl = entry.ttl;

                Box::pin(async move {
                    let mut conn = pool.get().await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;

                    if let Some(ttl_seconds) = ttl {
                        let _: () = redis::cmd("SETEX")
                            .arg(&full_key)
                            .arg(ttl_seconds)
                            .arg(&data)
                            .query_async(&mut conn)
                            .await
                            .map_err(|e| AgentError::tool("redis_cache", &format!("SETEX command failed: {}", e)))?;
                    } else {
                        let _: () = redis::cmd("SET")
                            .arg(&full_key)
                            .arg(&data)
                            .query_async(&mut conn)
                            .await
                            .map_err(|e| AgentError::tool("redis_cache", &format!("SET command failed: {}", e)))?;
                    }

                    Ok::<(), AgentError>(())
                })
            }).await?;
            
            let duration = start_time.elapsed();
            self.update_stats(duration, false).await; // Set operations don't count as hits
            
            debug!("Set Redis cache entry: {} (TTL: {:?})", key, entry.ttl);
            Ok(())
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let full_key = self.get_full_key(key);
        
        if let Some(pool) = &self.pool {
            let result = self.execute_with_retry(|| {
                let pool = pool.clone();
                let full_key = full_key.clone();
                
                Box::pin(async move {
                    let mut conn = pool.get().await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;
                    
                    let deleted: u32 = redis::cmd("DEL")
                        .arg(&full_key)
                        .query_async(&mut conn)
                        .await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("DEL command failed: {}", e)))?;
                    
                    Ok(deleted > 0)
                })
            }).await?;
            
            Ok(result)
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let full_key = self.get_full_key(key);
        
        if let Some(pool) = &self.pool {
            let result = self.execute_with_retry(|| {
                let pool = pool.clone();
                let full_key = full_key.clone();
                
                Box::pin(async move {
                    let mut conn = pool.get().await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;
                    
                    let exists: bool = redis::cmd("EXISTS")
                        .arg(&full_key)
                        .query_async(&mut conn)
                        .await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("EXISTS command failed: {}", e)))?;
                    
                    Ok(exists)
                })
            }).await?;
            
            Ok(result)
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    async fn stats(&self) -> Result<TierStats> {
        let stats = self.stats.read().await;
        
        Ok(TierStats {
            entry_count: 0, // Redis doesn't easily provide this without scanning
            memory_usage: 0, // Would need INFO memory command
            hits: stats.hits,
            misses: stats.misses,
            avg_access_time: stats.avg_response_time,
            last_access: stats.last_operation,
        })
    }

    async fn clear(&self) -> Result<()> {
        if let Some(pool) = &self.pool {
            self.execute_with_retry(|| {
                let pool = pool.clone();

                Box::pin(async move {
                    let mut conn = pool.get().await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("Failed to get connection: {}", e)))?;

                    let _: String = redis::cmd("FLUSHDB")
                        .query_async(&mut conn)
                        .await
                        .map_err(|e| AgentError::tool("redis_cache", &format!("FLUSHDB command failed: {}", e)))?;

                    Ok::<(), AgentError>(())
                })
            }).await?;
            
            info!("Cleared Redis cache: {}", self.name);
            Ok(())
        } else {
            Err(AgentError::tool("redis_cache", "Redis not connected"))
        }
    }

    async fn health_check(&self) -> Result<TierHealth> {
        let health = self.health.read().await;
        let stats = self.stats.read().await;
        
        // Determine health score based on error rate and connection status
        let error_rate = if stats.operations > 0 {
            stats.errors as f64 / stats.operations as f64
        } else {
            0.0
        };
        
        let health_score = match health.connection_status {
            ConnectionStatus::Connected => {
                if error_rate < 0.01 { 100 } // Less than 1% error rate
                else if error_rate < 0.05 { 80 } // Less than 5% error rate
                else if error_rate < 0.1 { 60 } // Less than 10% error rate
                else { 30 }
            }
            ConnectionStatus::Reconnecting => 50,
            ConnectionStatus::Disconnected => 20,
            ConnectionStatus::Error => 0,
        };
        
        Ok(TierHealth {
            is_healthy: health_score > 50,
            health_score: health_score as u8,
            last_check: Utc::now(),
            error_count: health.consecutive_errors,
            connection_status: health.connection_status.clone(),
        })
    }
}
