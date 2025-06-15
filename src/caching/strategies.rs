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

impl CacheAsideStrategy {
    /// Create a new cache-aside strategy
    pub fn new(name: String, data_source: Arc<dyn DataSource>) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::CacheAside,
                parameters: HashMap::new(),
                enabled: true,
            },
            data_source,
        }
    }
}

impl CacheStrategy for CacheAsideStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> StrategyConfig {
        self.config.clone()
    }
}

impl ReadThroughStrategy {
    /// Create a new read-through strategy
    pub fn new(name: String, data_source: Arc<dyn DataSource>) -> Self {
        Self {
            name: name.clone(),
            config: StrategyConfig {
                name,
                strategy_type: StrategyType::ReadThrough,
                parameters: HashMap::new(),
                enabled: true,
            },
            data_source,
        }
    }
}

impl CacheStrategy for ReadThroughStrategy {
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

/// Simple in-memory data source for testing and development
pub struct MemoryDataSource {
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    name: String,
}

impl MemoryDataSource {
    /// Create a new memory data source
    pub fn new(name: String) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            name,
        }
    }
}

#[async_trait::async_trait]
impl DataSource for MemoryDataSource {
    async fn load_bytes(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let data = self.data.read().await;
        Ok(data.get(key).cloned())
    }

    async fn save_bytes(&self, key: &str, data: &[u8]) -> Result<()> {
        let mut storage = self.data.write().await;
        storage.insert(key.to_string(), data.to_vec());
        debug!("Saved data to memory source {}: {} bytes", self.name, data.len());
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut data = self.data.write().await;
        Ok(data.remove(key).is_some())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true) // Memory source is always healthy
    }
}

/// Database data source implementation
pub struct DatabaseDataSource {
    /// Connection string
    connection_string: String,
    /// Table name
    table_name: String,
    /// Key column name
    key_column: String,
    /// Value column name
    value_column: String,
    /// Connection pool
    pool: Arc<RwLock<Option<String>>>, // Simplified for now
}

impl DatabaseDataSource {
    /// Create a new database data source
    pub fn new(
        connection_string: String,
        table_name: String,
        key_column: String,
        value_column: String,
    ) -> Self {
        Self {
            connection_string,
            table_name,
            key_column,
            value_column,
            pool: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize database connection
    pub async fn initialize(&self) -> Result<()> {
        // In a real implementation, this would establish a database connection
        let mut pool = self.pool.write().await;
        *pool = Some(self.connection_string.clone());
        info!("Initialized database data source: {}", self.table_name);
        Ok(())
    }
}

#[async_trait::async_trait]
impl DataSource for DatabaseDataSource {
    async fn load_bytes(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let pool = self.pool.read().await;
        if pool.is_none() {
            return Err(AgentError::validation("Database not initialized".to_string()));
        }

        // Simulate database query
        debug!("Loading from database: {} WHERE {} = '{}'", self.table_name, self.key_column, key);

        // In a real implementation, this would execute:
        // SELECT {value_column} FROM {table_name} WHERE {key_column} = ?
        // For now, return None to simulate cache miss
        Ok(None)
    }

    async fn save_bytes(&self, key: &str, data: &[u8]) -> Result<()> {
        let pool = self.pool.read().await;
        if pool.is_none() {
            return Err(AgentError::validation("Database not initialized".to_string()));
        }

        // Simulate database insert/update
        debug!("Saving to database: {} SET {} = ? WHERE {} = '{}'",
               self.table_name, self.value_column, self.key_column, key);

        // In a real implementation, this would execute:
        // INSERT INTO {table_name} ({key_column}, {value_column}) VALUES (?, ?)
        // ON DUPLICATE KEY UPDATE {value_column} = VALUES({value_column})
        info!("Saved {} bytes to database for key: {}", data.len(), key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let pool = self.pool.read().await;
        if pool.is_none() {
            return Err(AgentError::validation("Database not initialized".to_string()));
        }

        // Simulate database delete
        debug!("Deleting from database: {} WHERE {} = '{}'", self.table_name, self.key_column, key);

        // In a real implementation, this would execute:
        // DELETE FROM {table_name} WHERE {key_column} = ?
        Ok(true) // Assume successful deletion
    }

    async fn health_check(&self) -> Result<bool> {
        let pool = self.pool.read().await;
        Ok(pool.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_memory_data_source() {
        let source = MemoryDataSource::new("test".to_string());

        // Test save and load
        let test_data = b"test data";
        source.save_bytes("key1", test_data).await.unwrap();

        let loaded = source.load_bytes("key1").await.unwrap();
        assert_eq!(loaded, Some(test_data.to_vec()));

        // Test non-existent key
        let missing = source.load_bytes("missing").await.unwrap();
        assert_eq!(missing, None);

        // Test delete
        let deleted = source.delete("key1").await.unwrap();
        assert!(deleted);

        let after_delete = source.load_bytes("key1").await.unwrap();
        assert_eq!(after_delete, None);

        // Test health check
        assert!(source.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_write_through_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let strategy = WriteThroughStrategy::new("test_strategy".to_string(), source);

        assert_eq!(strategy.name(), "test_strategy");
        assert_eq!(strategy.config().strategy_type, StrategyType::WriteThrough);
        assert!(strategy.config().enabled);
    }

    #[tokio::test]
    async fn test_cache_aside_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let strategy = CacheAsideStrategy::new("cache_aside_test".to_string(), source);

        assert_eq!(strategy.name(), "cache_aside_test");
        assert_eq!(strategy.config().strategy_type, StrategyType::CacheAside);
        assert!(strategy.config().enabled);
    }

    #[tokio::test]
    async fn test_read_through_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let strategy = ReadThroughStrategy::new("read_through_test".to_string(), source);

        assert_eq!(strategy.name(), "read_through_test");
        assert_eq!(strategy.config().strategy_type, StrategyType::ReadThrough);
        assert!(strategy.config().enabled);
    }

    #[tokio::test]
    async fn test_refresh_ahead_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let strategy = RefreshAheadStrategy::new("refresh_test".to_string(), source, 0.3);

        assert_eq!(strategy.name(), "refresh_test");
        assert_eq!(strategy.config().strategy_type, StrategyType::RefreshAhead);
        assert!(strategy.config().enabled);
        assert_eq!(strategy.refresh_threshold, 0.3);
    }

    #[tokio::test]
    async fn test_write_behind_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let strategy = WriteBehindStrategy::new(
            "write_behind_test".to_string(),
            source,
            10, // batch size
            Duration::from_millis(100), // flush interval
        );

        assert_eq!(strategy.name(), "write_behind_test");
        assert_eq!(strategy.config().strategy_type, StrategyType::WriteBehind);
        assert!(strategy.config().enabled);
        assert_eq!(strategy.batch_size, 10);
        assert_eq!(strategy.flush_interval, Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_circuit_breaker_strategy() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let inner_strategy = Arc::new(CacheAsideStrategy::new("inner".to_string(), source));

        let circuit_breaker = CircuitBreakerStrategy::new(
            "circuit_test".to_string(),
            inner_strategy,
            3, // failure threshold
            Duration::from_secs(5), // recovery timeout
            2, // success threshold
        );

        assert_eq!(circuit_breaker.name(), "circuit_test");
        assert_eq!(circuit_breaker.config().strategy_type, StrategyType::CircuitBreaker);

        // Test circuit breaker state
        let breaker_state = circuit_breaker.circuit_breaker.read().await;
        assert_eq!(breaker_state.state, CircuitState::Closed);
        assert_eq!(breaker_state.failure_threshold, 3);
        assert_eq!(breaker_state.success_threshold, 2);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_handling() {
        let source = Arc::new(MemoryDataSource::new("test".to_string()));
        let inner_strategy = Arc::new(CacheAsideStrategy::new("inner".to_string(), source));

        let circuit_breaker = CircuitBreakerStrategy::new(
            "circuit_test".to_string(),
            inner_strategy,
            2, // failure threshold
            Duration::from_millis(100), // recovery timeout
            1, // success threshold
        );

        // Record failures
        circuit_breaker.record_result(false).await;
        circuit_breaker.record_result(false).await;

        // Circuit should be open now
        let breaker_state = circuit_breaker.circuit_breaker.read().await;
        assert_eq!(breaker_state.state, CircuitState::Open);
        assert_eq!(breaker_state.failure_count, 2);

        drop(breaker_state);

        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;

        // Should be able to attempt reset
        assert!(circuit_breaker.should_attempt_reset().await);
    }

    #[tokio::test]
    async fn test_database_data_source() {
        let source = DatabaseDataSource::new(
            "test://connection".to_string(),
            "cache_table".to_string(),
            "cache_key".to_string(),
            "cache_value".to_string(),
        );

        // Initialize the source
        source.initialize().await.unwrap();

        // Test health check
        assert!(source.health_check().await.unwrap());

        // Test operations (these are simulated)
        let result = source.load_bytes("test_key").await.unwrap();
        assert_eq!(result, None); // Simulated miss

        source.save_bytes("test_key", b"test_data").await.unwrap();

        let deleted = source.delete("test_key").await.unwrap();
        assert!(deleted); // Simulated success
    }
}
