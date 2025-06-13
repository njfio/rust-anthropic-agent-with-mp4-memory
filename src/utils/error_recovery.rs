use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::sleep;

use crate::utils::error::{AgentError, Result};

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    ExponentialBackoff {
        max_retries: usize,
        base_delay: Duration,
        max_delay: Duration,
    },
    /// Retry with fixed intervals
    FixedInterval {
        max_retries: usize,
        interval: Duration,
    },
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: usize,
        recovery_timeout: Duration,
    },
    /// Fallback to alternative implementation
    Fallback {
        fallback_fn: String, // Function name for fallback
    },
    /// Graceful degradation
    GracefulDegradation {
        reduced_functionality: bool,
    },
    /// No recovery - fail fast
    FailFast,
}

/// Error recovery configuration
#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfig {
    /// Default strategy for unknown errors
    pub default_strategy: RecoveryStrategy,
    /// Strategy mapping for specific error types
    pub error_strategies: HashMap<String, RecoveryStrategy>,
    /// Maximum total recovery time
    pub max_recovery_time: Duration,
    /// Enable recovery logging
    pub enable_logging: bool,
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        let mut error_strategies = HashMap::new();
        
        // Configure strategies for different error types
        error_strategies.insert("network".to_string(), RecoveryStrategy::ExponentialBackoff {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        });
        
        error_strategies.insert("rate_limit".to_string(), RecoveryStrategy::FixedInterval {
            max_retries: 5,
            interval: Duration::from_secs(1),
        });
        
        error_strategies.insert("authentication".to_string(), RecoveryStrategy::FailFast);
        
        error_strategies.insert("memory".to_string(), RecoveryStrategy::GracefulDegradation {
            reduced_functionality: true,
        });
        
        Self {
            default_strategy: RecoveryStrategy::ExponentialBackoff {
                max_retries: 2,
                base_delay: Duration::from_millis(50),
                max_delay: Duration::from_secs(2),
            },
            error_strategies,
            max_recovery_time: Duration::from_secs(30),
            enable_logging: true,
        }
    }
}

/// Recovery attempt information
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    pub attempt_number: usize,
    pub strategy: RecoveryStrategy,
    pub error: String,
    pub timestamp: Instant,
    pub delay: Option<Duration>,
}

/// Error recovery manager
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    config: ErrorRecoveryConfig,
    recovery_history: Arc<Mutex<HashMap<String, Vec<RecoveryAttempt>>>>,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        Self {
            config,
            recovery_history: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Execute an operation with error recovery
    pub async fn execute_with_recovery<F, Fut, T>(
        &self,
        operation_id: &str,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let start_time = Instant::now();
        let error_type = self.classify_error_type(operation_id);
        let strategy = self.get_strategy_for_error(&error_type);
        
        match strategy {
            RecoveryStrategy::ExponentialBackoff { max_retries, base_delay, max_delay } => {
                self.execute_with_exponential_backoff(
                    operation_id, operation, max_retries, base_delay, max_delay, start_time
                ).await
            },
            RecoveryStrategy::FixedInterval { max_retries, interval } => {
                self.execute_with_fixed_interval(
                    operation_id, operation, max_retries, interval, start_time
                ).await
            },
            RecoveryStrategy::FailFast => {
                // Execute once, fail immediately on error
                operation().await
            },
            _ => {
                // For other strategies, use default exponential backoff
                self.execute_with_exponential_backoff(
                    operation_id, operation, 2, Duration::from_millis(50), Duration::from_secs(2), start_time
                ).await
            }
        }
    }

    /// Execute with exponential backoff
    async fn execute_with_exponential_backoff<F, Fut, T>(
        &self,
        operation_id: &str,
        operation: F,
        max_retries: usize,
        base_delay: Duration,
        max_delay: Duration,
        start_time: Instant,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let mut attempt = 0;
        let mut delay = base_delay;
        
        loop {
            // Check if we've exceeded the maximum recovery time
            if start_time.elapsed() > self.config.max_recovery_time {
                return Err(AgentError::tool("error_recovery", &format!(
                    "Recovery timeout exceeded for operation: {}", operation_id
                )));
            }
            
            match operation().await {
                Ok(result) => {
                    if attempt > 0 && self.config.enable_logging {
                        tracing::info!("Operation {} succeeded after {} attempts", operation_id, attempt + 1);
                    }
                    return Ok(result);
                },
                Err(e) => {
                    attempt += 1;
                    
                    // Record recovery attempt
                    self.record_recovery_attempt(operation_id, RecoveryAttempt {
                        attempt_number: attempt,
                        strategy: RecoveryStrategy::ExponentialBackoff {
                            max_retries,
                            base_delay,
                            max_delay,
                        },
                        error: e.to_string(),
                        timestamp: Instant::now(),
                        delay: Some(delay),
                    });
                    
                    if attempt >= max_retries {
                        return Err(AgentError::tool("error_recovery", &format!(
                            "Operation {} failed after {} attempts: {}", operation_id, max_retries, e
                        )));
                    }
                    
                    if self.config.enable_logging {
                        tracing::warn!("Operation {} failed (attempt {}), retrying in {:?}: {}", 
                                     operation_id, attempt, delay, e);
                    }
                    
                    sleep(delay).await;
                    delay = std::cmp::min(delay * 2, max_delay);
                }
            }
        }
    }

    /// Execute with fixed interval
    async fn execute_with_fixed_interval<F, Fut, T>(
        &self,
        operation_id: &str,
        operation: F,
        max_retries: usize,
        interval: Duration,
        start_time: Instant,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let mut attempt = 0;
        
        loop {
            // Check if we've exceeded the maximum recovery time
            if start_time.elapsed() > self.config.max_recovery_time {
                return Err(AgentError::tool("error_recovery", &format!(
                    "Recovery timeout exceeded for operation: {}", operation_id
                )));
            }
            
            match operation().await {
                Ok(result) => {
                    if attempt > 0 && self.config.enable_logging {
                        tracing::info!("Operation {} succeeded after {} attempts", operation_id, attempt + 1);
                    }
                    return Ok(result);
                },
                Err(e) => {
                    attempt += 1;
                    
                    // Record recovery attempt
                    self.record_recovery_attempt(operation_id, RecoveryAttempt {
                        attempt_number: attempt,
                        strategy: RecoveryStrategy::FixedInterval { max_retries, interval },
                        error: e.to_string(),
                        timestamp: Instant::now(),
                        delay: Some(interval),
                    });
                    
                    if attempt >= max_retries {
                        return Err(AgentError::tool("error_recovery", &format!(
                            "Operation {} failed after {} attempts: {}", operation_id, max_retries, e
                        )));
                    }
                    
                    if self.config.enable_logging {
                        tracing::warn!("Operation {} failed (attempt {}), retrying in {:?}: {}", 
                                     operation_id, attempt, interval, e);
                    }
                    
                    sleep(interval).await;
                }
            }
        }
    }

    /// Classify error type based on operation ID and context
    fn classify_error_type(&self, operation_id: &str) -> String {
        if operation_id.contains("network") || operation_id.contains("http") || operation_id.contains("api") {
            "network".to_string()
        } else if operation_id.contains("auth") || operation_id.contains("login") {
            "authentication".to_string()
        } else if operation_id.contains("rate") || operation_id.contains("limit") {
            "rate_limit".to_string()
        } else if operation_id.contains("memory") || operation_id.contains("alloc") {
            "memory".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Get recovery strategy for error type
    fn get_strategy_for_error(&self, error_type: &str) -> RecoveryStrategy {
        self.config.error_strategies
            .get(error_type)
            .cloned()
            .unwrap_or_else(|| self.config.default_strategy.clone())
    }

    /// Record a recovery attempt
    fn record_recovery_attempt(&self, operation_id: &str, attempt: RecoveryAttempt) {
        if let Ok(mut history) = self.recovery_history.lock() {
            history.entry(operation_id.to_string())
                .or_insert_with(Vec::new)
                .push(attempt);
        }
    }

    /// Get recovery statistics for an operation
    pub fn get_recovery_stats(&self, operation_id: &str) -> Option<RecoveryStats> {
        if let Ok(history) = self.recovery_history.lock() {
            if let Some(attempts) = history.get(operation_id) {
                return Some(RecoveryStats::from_attempts(attempts));
            }
        }
        None
    }

    /// Get all recovery statistics
    pub fn get_all_recovery_stats(&self) -> HashMap<String, RecoveryStats> {
        let mut stats = HashMap::new();
        if let Ok(history) = self.recovery_history.lock() {
            for (operation_id, attempts) in history.iter() {
                stats.insert(operation_id.clone(), RecoveryStats::from_attempts(attempts));
            }
        }
        stats
    }

    /// Clear recovery history
    pub fn clear_history(&self) {
        if let Ok(mut history) = self.recovery_history.lock() {
            history.clear();
        }
    }
}

/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_attempts: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub average_attempts_to_success: f64,
    pub total_recovery_time: Duration,
    pub most_common_error: Option<String>,
}

impl RecoveryStats {
    fn from_attempts(attempts: &[RecoveryAttempt]) -> Self {
        let total_attempts = attempts.len();
        let mut successful_recoveries = 0;
        let mut failed_recoveries = 0;
        let mut total_recovery_time = Duration::new(0, 0);
        let mut error_counts = HashMap::new();
        
        // Group attempts by operation (assuming consecutive attempts are for the same operation)
        let mut current_operation_attempts = 0;
        let mut operation_success_attempts = Vec::new();
        
        for (i, attempt) in attempts.iter().enumerate() {
            current_operation_attempts += 1;
            
            // Count error types
            *error_counts.entry(attempt.error.clone()).or_insert(0) += 1;
            
            // Check if this is the last attempt or if the next attempt is for a new operation
            let is_last_attempt = i == attempts.len() - 1;
            let is_operation_end = is_last_attempt || 
                (i + 1 < attempts.len() && attempts[i + 1].attempt_number == 1);
            
            if is_operation_end {
                if attempt.error.contains("succeeded") || attempt.attempt_number == 1 {
                    successful_recoveries += 1;
                    operation_success_attempts.push(current_operation_attempts);
                } else {
                    failed_recoveries += 1;
                }
                current_operation_attempts = 0;
            }
            
            if let Some(delay) = attempt.delay {
                total_recovery_time += delay;
            }
        }
        
        let average_attempts_to_success = if !operation_success_attempts.is_empty() {
            operation_success_attempts.iter().sum::<usize>() as f64 / operation_success_attempts.len() as f64
        } else {
            0.0
        };
        
        let most_common_error = error_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(error, _)| error);
        
        Self {
            total_attempts,
            successful_recoveries,
            failed_recoveries,
            average_attempts_to_success,
            total_recovery_time,
            most_common_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_exponential_backoff_success() {
        let mut config = ErrorRecoveryConfig::default();
        // Ensure we have enough retries for the test
        config.default_strategy = RecoveryStrategy::ExponentialBackoff {
            max_retries: 5,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };
        let manager = ErrorRecoveryManager::new(config);

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let operation = move || {
            let count = call_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if count < 2 {
                    Err(AgentError::tool("test", "Temporary failure"))
                } else {
                    Ok("Success")
                }
            })
        };

        let result = manager.execute_with_recovery("test_operation", operation).await;
        assert!(result.is_ok(), "Expected success but got error: {:?}", result.err());
        assert_eq!(result.unwrap(), "Success");

        // Should have made exactly 3 attempts (fails on 0, 1, succeeds on 2)
        let final_count = call_count.load(Ordering::SeqCst);
        assert_eq!(final_count, 3, "Expected exactly 3 attempts, got {}", final_count);
    }

    #[tokio::test]
    async fn test_exponential_backoff_failure() {
        let mut config = ErrorRecoveryConfig::default();
        config.default_strategy = RecoveryStrategy::ExponentialBackoff {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };

        let manager = ErrorRecoveryManager::new(config);

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let operation = move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Err::<&str, AgentError>(AgentError::tool("test", "Persistent failure"))
            })
        };

        let result = manager.execute_with_recovery("test_operation", operation).await;
        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_fixed_interval_recovery() {
        let mut config = ErrorRecoveryConfig::default();
        config.error_strategies.insert("test".to_string(), RecoveryStrategy::FixedInterval {
            max_retries: 3,
            interval: Duration::from_millis(1),
        });

        let manager = ErrorRecoveryManager::new(config);

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let operation = move || {
            let count = call_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if count < 1 {
                    Err(AgentError::tool("test", "Temporary failure"))
                } else {
                    Ok("Success")
                }
            })
        };

        let result = manager.execute_with_recovery("test_operation", operation).await;
        assert!(result.is_ok());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_fail_fast_strategy() {
        let mut config = ErrorRecoveryConfig::default();
        config.error_strategies.insert("auth".to_string(), RecoveryStrategy::FailFast);

        let manager = ErrorRecoveryManager::new(config);

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let operation = move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Err::<&str, AgentError>(AgentError::authentication("Authentication failure"))
            })
        };

        let result = manager.execute_with_recovery("auth_operation", operation).await;
        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_error_classification() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        assert_eq!(manager.classify_error_type("network_request"), "network");
        assert_eq!(manager.classify_error_type("http_call"), "network");
        assert_eq!(manager.classify_error_type("api_request"), "network");
        assert_eq!(manager.classify_error_type("auth_login"), "authentication");
        assert_eq!(manager.classify_error_type("rate_limit_check"), "rate_limit");
        assert_eq!(manager.classify_error_type("memory_allocation"), "memory");
        assert_eq!(manager.classify_error_type("unknown_operation"), "unknown");
    }

    #[tokio::test]
    async fn test_recovery_stats() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        // Record some recovery attempts
        manager.record_recovery_attempt("test_op", RecoveryAttempt {
            attempt_number: 1,
            strategy: RecoveryStrategy::FailFast,
            error: "First error".to_string(),
            timestamp: Instant::now(),
            delay: Some(Duration::from_millis(100)),
        });

        manager.record_recovery_attempt("test_op", RecoveryAttempt {
            attempt_number: 2,
            strategy: RecoveryStrategy::FailFast,
            error: "Second error".to_string(),
            timestamp: Instant::now(),
            delay: Some(Duration::from_millis(200)),
        });

        let stats = manager.get_recovery_stats("test_op");
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.total_attempts, 2);
        assert!(stats.total_recovery_time >= Duration::from_millis(300));
    }

    #[tokio::test]
    async fn test_recovery_timeout() {
        let mut config = ErrorRecoveryConfig::default();
        config.max_recovery_time = Duration::from_millis(10);
        config.default_strategy = RecoveryStrategy::ExponentialBackoff {
            max_retries: 10,
            base_delay: Duration::from_millis(5),
            max_delay: Duration::from_millis(50),
        };

        let manager = ErrorRecoveryManager::new(config);

        let operation = || {
            Box::pin(async move {
                Err::<&str, AgentError>(AgentError::tool("test", "Always fails"))
            })
        };

        let result = manager.execute_with_recovery("timeout_test", operation).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout"));
    }

    #[test]
    fn test_recovery_strategy_equality() {
        let strategy1 = RecoveryStrategy::ExponentialBackoff {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        };

        let strategy2 = RecoveryStrategy::ExponentialBackoff {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        };

        assert_eq!(strategy1, strategy2);

        let strategy3 = RecoveryStrategy::FailFast;
        assert_ne!(strategy1, strategy3);
    }

    #[test]
    fn test_default_config() {
        let config = ErrorRecoveryConfig::default();

        assert!(config.error_strategies.contains_key("network"));
        assert!(config.error_strategies.contains_key("rate_limit"));
        assert!(config.error_strategies.contains_key("authentication"));
        assert!(config.error_strategies.contains_key("memory"));

        assert_eq!(config.error_strategies["authentication"], RecoveryStrategy::FailFast);
        assert!(config.enable_logging);
        assert_eq!(config.max_recovery_time, Duration::from_secs(30));
    }
}
