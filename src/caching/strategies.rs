// Cache Strategies and Patterns
// Provides advanced caching strategies for different use cases and access patterns

use super::{CacheEntry, CacheManager, CacheResult};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache strategy trait
pub trait CacheStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;

    /// Strategy configuration
    fn config(&self) -> StrategyConfig;
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Strategy parameters
    pub parameters: HashMap<String, String>,
    /// Enable strategy
    pub enabled: bool,
}

/// Types of cache strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StrategyType {
    /// Write-through caching
    WriteThrough,
    /// Write-behind (write-back) caching
    WriteBehind,
    /// Cache-aside (lazy loading)
    CacheAside,
    /// Read-through caching
    ReadThrough,
    /// Refresh-ahead caching
    RefreshAhead,
    /// Circuit breaker pattern
    CircuitBreaker,
    /// Bulkhead pattern
    Bulkhead,
}

/// Write-through cache strategy
pub struct WriteThroughStrategy {
    name: String,
    config: StrategyConfig,
    data_source: Arc<dyn DataSource>,
}

/// Write-behind cache strategy
pub struct WriteBehindStrategy {
    name: String,
    config: StrategyConfig,
    data_source: Arc<dyn DataSource>,
    write_queue: Arc<RwLock<Vec<WriteOperation>>>,
    batch_size: usize,
    flush_interval: Duration,
}

/// Cache-aside strategy
pub struct CacheAsideStrategy {
    name: String,
    config: StrategyConfig,
    data_source: Arc<dyn DataSource>,
}

/// Read-through strategy
pub struct ReadThroughStrategy {
    name: String,
    config: StrategyConfig,
    data_source: Arc<dyn DataSource>,
}

/// Refresh-ahead strategy
pub struct RefreshAheadStrategy {
    name: String,
    config: StrategyConfig,
    data_source: Arc<dyn DataSource>,
    refresh_threshold: f64,
    background_refresh: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
}

/// Circuit breaker strategy
pub struct CircuitBreakerStrategy {
    name: String,
    config: StrategyConfig,
    inner_strategy: Arc<dyn CacheStrategy>,
    circuit_breaker: Arc<RwLock<CircuitBreakerState>>,
}

/// Data source trait for strategies
#[async_trait::async_trait]
pub trait DataSource: Send + Sync {
    /// Load data from source (returns JSON bytes)
    async fn load_bytes(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Save data to source (accepts JSON bytes)
    async fn save_bytes(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Delete data from source
    async fn delete(&self, key: &str) -> Result<bool>;

    /// Check if source is healthy
    async fn health_check(&self) -> Result<bool>;
}

/// Write operation for write-behind strategy
#[derive(Debug, Clone)]
pub struct WriteOperation {
    pub key: String,
    pub data: Vec<u8>,
    pub operation_type: WriteOperationType,
    pub timestamp: DateTime<Utc>,
}

/// Types of write operations
#[derive(Debug, Clone, PartialEq)]
pub enum WriteOperationType {
    Insert,
    Update,
    Delete,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub state: CircuitState,
    pub failure_count: u32,
    pub last_failure_time: Option<DateTime<Utc>>,
    pub success_count: u32,
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub success_threshold: u32,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl WriteThroughStrategy {
    /// Create a new write-through strategy
    pub fn new(name: String, data_source: Arc<dyn DataSource>) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::WriteThrough,
                parameters: HashMap::new(),
                enabled: true,
            },
            data_source,
        }
    }
}

impl CacheStrategy for WriteThroughStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> StrategyConfig {
        self.config.clone()
    }
}

impl WriteBehindStrategy {
    /// Create a new write-behind strategy
    pub fn new(name: String, data_source: Arc<dyn DataSource>, batch_size: usize, flush_interval: Duration) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::WriteBehind,
                parameters: HashMap::new(),
                enabled: true,
            },
            data_source,
            write_queue: Arc::new(RwLock::new(Vec::new())),
            batch_size,
            flush_interval,
        }
    }

    /// Start background flush task
    pub async fn start_flush_task(&self) -> tokio::task::JoinHandle<()> {
        let write_queue = Arc::clone(&self.write_queue);
        let data_source = Arc::clone(&self.data_source);
        let batch_size = self.batch_size;
        let flush_interval = self.flush_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);

            loop {
                interval.tick().await;

                let operations = {
                    let mut queue = write_queue.write().await;
                    if queue.is_empty() {
                        continue;
                    }

                    let drain_count = queue.len().min(batch_size);
                    queue.drain(0..drain_count).collect::<Vec<_>>()
                };

                let operation_count = operations.len();

                for operation in operations {
                    match operation.operation_type {
                        WriteOperationType::Insert | WriteOperationType::Update => {
                            if let Err(e) = data_source.save_bytes(&operation.key, &operation.data).await {
                                warn!("Failed to flush write operation for key {}: {}", operation.key, e);
                                // Could implement retry logic here
                            }
                        }
                        WriteOperationType::Delete => {
                            if let Err(e) = data_source.delete(&operation.key).await {
                                warn!("Failed to flush delete operation for key {}: {}", operation.key, e);
                            }
                        }
                    }
                }

                if operation_count > 0 {
                    debug!("Flushed {} write operations", operation_count);
                }
            }
        })
    }
}

impl CacheStrategy for WriteBehindStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> StrategyConfig {
        self.config.clone()
    }
}

impl RefreshAheadStrategy {
    /// Create a new refresh-ahead strategy
    pub fn new(name: String, data_source: Arc<dyn DataSource>, refresh_threshold: f64) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::RefreshAhead,
                parameters: HashMap::new(),
                enabled: true,
            },
            data_source,
            refresh_threshold,
            background_refresh: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if entry should be refreshed
    async fn should_refresh(&self, key: &str, entry: &CacheEntry) -> bool {
        if let Some(ttl) = entry.ttl {
            let age = Utc::now().signed_duration_since(entry.created_at).num_seconds() as u64;
            let remaining_ratio = (ttl - age) as f64 / ttl as f64;
            
            if remaining_ratio < self.refresh_threshold {
                // Check if already refreshing
                let background_refresh = self.background_refresh.read().await;
                return !background_refresh.contains_key(key);
            }
        }
        false
    }

    /// Start background refresh
    async fn start_background_refresh<T>(&self, key: String, cache_manager: Arc<CacheManager>)
    where
        T: for<'de> Deserialize<'de> + Serialize + Send + Sync + 'static,
    {
        // Mark as refreshing
        {
            let mut background_refresh = self.background_refresh.write().await;
            background_refresh.insert(key.clone(), Utc::now());
        }

        let data_source = Arc::clone(&self.data_source);
        let background_refresh = Arc::clone(&self.background_refresh);

        tokio::spawn(async move {
            match data_source.load_bytes(&key).await {
                Ok(Some(data)) => {
                    // Deserialize the data
                    match serde_json::from_slice::<T>(&data) {
                        Ok(value) => {
                            if let Err(e) = cache_manager.set(&key, &value, None).await {
                                warn!("Failed to refresh cache for key {}: {}", key, e);
                            } else {
                                debug!("Background refresh completed for key: {}", key);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to deserialize data during background refresh for key {}: {}", key, e);
                        }
                    }
                }
                Ok(None) => {
                    debug!("No data found during background refresh for key: {}", key);
                }
                Err(e) => {
                    warn!("Background refresh failed for key {}: {}", key, e);
                }
            }

            // Remove from refreshing set
            let mut refresh_guard = background_refresh.write().await;
            refresh_guard.remove(&key);
        });
    }
}

impl CacheStrategy for RefreshAheadStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> StrategyConfig {
        self.config.clone()
    }
}

impl CircuitBreakerStrategy {
    /// Create a new circuit breaker strategy
    pub fn new(
        name: String,
        inner_strategy: Arc<dyn CacheStrategy>,
        failure_threshold: u32,
        recovery_timeout: Duration,
        success_threshold: u32,
    ) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::CircuitBreaker,
                parameters: HashMap::new(),
                enabled: true,
            },
            inner_strategy,
            circuit_breaker: Arc::new(RwLock::new(CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                success_count: 0,
                failure_threshold,
                recovery_timeout,
                success_threshold,
            })),
        }
    }

    /// Check if circuit should be opened
    async fn should_open_circuit(&self) -> bool {
        let breaker = self.circuit_breaker.read().await;
        breaker.failure_count >= breaker.failure_threshold
    }

    /// Check if circuit should transition to half-open
    async fn should_attempt_reset(&self) -> bool {
        let breaker = self.circuit_breaker.read().await;
        if let Some(last_failure) = breaker.last_failure_time {
            Utc::now().signed_duration_since(last_failure) > chrono::Duration::from_std(breaker.recovery_timeout).unwrap()
        } else {
            false
        }
    }

    /// Record operation result
    async fn record_result(&self, success: bool) {
        let mut breaker = self.circuit_breaker.write().await;
        
        if success {
            breaker.success_count += 1;
            breaker.failure_count = 0;
            
            // Transition from half-open to closed if enough successes
            if breaker.state == CircuitState::HalfOpen && breaker.success_count >= breaker.success_threshold {
                breaker.state = CircuitState::Closed;
                breaker.success_count = 0;
                info!("Circuit breaker closed for strategy: {}", self.name);
            }
        } else {
            breaker.failure_count += 1;
            breaker.success_count = 0;
            breaker.last_failure_time = Some(Utc::now());
            
            // Open circuit if failure threshold reached
            if breaker.failure_count >= breaker.failure_threshold {
                breaker.state = CircuitState::Open;
                warn!("Circuit breaker opened for strategy: {}", self.name);
            }
        }
    }
}

impl CacheStrategy for CircuitBreakerStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> StrategyConfig {
        self.config.clone()
    }
}
