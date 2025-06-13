use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    /// Circuit is closed, requests are allowed through
    Closed,
    /// Circuit is open, requests are blocked
    Open,
    /// Circuit is half-open, limited requests are allowed to test recovery
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit
    pub failure_threshold: usize,
    /// Time to wait before transitioning from Open to HalfOpen
    pub recovery_timeout: Duration,
    /// Number of successful requests needed to close the circuit from HalfOpen
    pub success_threshold: usize,
    /// Time window for counting failures
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 3,
            failure_window: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    last_failure_time: AtomicU64,
    last_state_change: AtomicU64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_time: AtomicU64::new(0),
            last_state_change: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
        }
    }

    /// Create a circuit breaker with default configuration
    pub fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Check if a request should be allowed through the circuit breaker
    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let last_change = self.last_state_change.load(Ordering::Relaxed);

                if now - last_change >= self.config.recovery_timeout.as_secs() {
                    drop(state);
                    self.transition_to_half_open().await;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful execution
    pub async fn record_success(&self) {
        let current_state = {
            let state = self.state.read().await;
            *state
        };

        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.config.success_threshold {
                    self.transition_to_closed().await;
                }
            }
            CircuitState::Open => {
                // Ignore successes when open (shouldn't happen)
            }
        }

        debug!(
            "Circuit breaker recorded success, state: {:?}",
            current_state
        );
    }

    /// Record a failed execution
    pub async fn record_failure(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_failure_time.store(now, Ordering::Relaxed);

        let current_state = {
            let state = self.state.read().await;
            *state
        };

        match current_state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failure_count >= self.config.failure_threshold {
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state should open the circuit
                self.transition_to_open().await;
            }
            CircuitState::Open => {
                // Already open, just increment counter
                self.failure_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        warn!(
            "Circuit breaker recorded failure, state: {:?}",
            current_state
        );
    }

    /// Get the current state of the circuit breaker
    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Get circuit breaker statistics
    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = *self.state.read().await;
        CircuitBreakerStats {
            state,
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            last_failure_time: self.last_failure_time.load(Ordering::Relaxed),
            last_state_change: self.last_state_change.load(Ordering::Relaxed),
        }
    }

    /// Reset the circuit breaker to closed state
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.last_state_change.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
        debug!("Circuit breaker reset to closed state");
    }

    /// Transition to open state
    async fn transition_to_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Open;
        self.success_count.store(0, Ordering::Relaxed);
        self.last_state_change.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
        error!("Circuit breaker opened due to failures");
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
        self.last_state_change.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
        warn!("Circuit breaker transitioned to half-open for testing");
    }

    /// Transition to closed state
    async fn transition_to_closed(&self) {
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.last_state_change.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
        debug!("Circuit breaker closed after successful recovery");
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_failure_time: u64,
    pub last_state_change: u64,
}

/// Error type for circuit breaker
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open, request blocked")]
    CircuitOpen,
    #[error("Circuit breaker execution failed: {0}")]
    ExecutionFailed(String),
}

/// Execute a function with circuit breaker protection
pub async fn execute_with_circuit_breaker<F, T, E>(
    circuit_breaker: &CircuitBreaker,
    operation: F,
) -> Result<T, CircuitBreakerError>
where
    F: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    if !circuit_breaker.can_execute().await {
        return Err(CircuitBreakerError::CircuitOpen);
    }

    match operation.await {
        Ok(result) => {
            circuit_breaker.record_success().await;
            Ok(result)
        }
        Err(error) => {
            circuit_breaker.record_failure().await;
            Err(CircuitBreakerError::ExecutionFailed(error.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_circuit_breaker_closed_state() {
        let circuit_breaker = CircuitBreaker::default();

        assert_eq!(circuit_breaker.get_state().await, CircuitState::Closed);
        assert!(circuit_breaker.can_execute().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // Record failures
        for _ in 0..3 {
            circuit_breaker.record_failure().await;
        }

        assert_eq!(circuit_breaker.get_state().await, CircuitState::Open);
        assert!(!circuit_breaker.can_execute().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // Open the circuit
        circuit_breaker.record_failure().await;
        circuit_breaker.record_failure().await;
        assert_eq!(circuit_breaker.get_state().await, CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;

        // Should allow execution (transitioning to half-open)
        assert!(circuit_breaker.can_execute().await);
        assert_eq!(circuit_breaker.get_state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_closes_on_success() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // Open the circuit
        circuit_breaker.record_failure().await;
        circuit_breaker.record_failure().await;

        // Wait and transition to half-open
        sleep(Duration::from_millis(150)).await;
        assert!(circuit_breaker.can_execute().await);

        // Record successes to close the circuit
        circuit_breaker.record_success().await;
        circuit_breaker.record_success().await;

        assert_eq!(circuit_breaker.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_execute_with_circuit_breaker() {
        let circuit_breaker = CircuitBreaker::default();

        // Test successful execution
        let result =
            execute_with_circuit_breaker(&circuit_breaker, async { Ok::<i32, &str>(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Test failed execution
        let result = execute_with_circuit_breaker(&circuit_breaker, async {
            Err::<i32, &str>("test error")
        })
        .await;

        assert!(result.is_err());
    }
}
