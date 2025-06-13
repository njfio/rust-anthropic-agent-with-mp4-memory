use crate::utils::error::{AgentError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of requests per window
    pub max_requests: u32,
    /// Time window duration
    pub window_duration: Duration,
    /// Whether to use per-tool rate limiting
    pub per_tool_limiting: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_duration: Duration::from_secs(60), // 1 minute
            per_tool_limiting: true,
        }
    }
}

/// Rate limiter implementation using sliding window
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    // Global rate limiting
    global_requests: Arc<Mutex<Vec<Instant>>>,
    // Per-tool rate limiting
    tool_requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            global_requests: Arc::new(Mutex::new(Vec::new())),
            tool_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if a request is allowed (global rate limiting)
    pub fn check_global_rate_limit(&self) -> Result<()> {
        let now = Instant::now();
        let mut requests = self
            .global_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;

        // Remove old requests outside the window
        requests
            .retain(|&request_time| now.duration_since(request_time) < self.config.window_duration);

        // Check if we're within the limit
        if requests.len() >= self.config.max_requests as usize {
            return Err(AgentError::rate_limit(format!(
                "Global rate limit exceeded: {} requests per {} seconds",
                self.config.max_requests,
                self.config.window_duration.as_secs()
            )));
        }

        // Add current request
        requests.push(now);
        Ok(())
    }

    /// Check if a request is allowed for a specific tool
    pub fn check_tool_rate_limit(&self, tool_name: &str) -> Result<()> {
        if !self.config.per_tool_limiting {
            return Ok(());
        }

        let now = Instant::now();
        let mut tool_requests = self
            .tool_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;

        // Get or create request history for this tool
        let requests = tool_requests
            .entry(tool_name.to_string())
            .or_insert_with(Vec::new);

        // Remove old requests outside the window
        requests
            .retain(|&request_time| now.duration_since(request_time) < self.config.window_duration);

        // Check if we're within the limit (per-tool limit is half of global limit)
        let tool_limit = (self.config.max_requests / 2).max(1);
        if requests.len() >= tool_limit as usize {
            return Err(AgentError::rate_limit(format!(
                "Tool '{}' rate limit exceeded: {} requests per {} seconds",
                tool_name,
                tool_limit,
                self.config.window_duration.as_secs()
            )));
        }

        // Add current request
        requests.push(now);
        Ok(())
    }

    /// Check both global and tool-specific rate limits
    pub fn check_rate_limit(&self, tool_name: &str) -> Result<()> {
        // Check global rate limit first
        self.check_global_rate_limit()?;

        // Then check tool-specific rate limit
        self.check_tool_rate_limit(tool_name)?;

        Ok(())
    }

    /// Get current rate limit status
    pub fn get_rate_limit_status(&self) -> Result<RateLimitStatus> {
        let now = Instant::now();

        // Global status
        let global_requests = self
            .global_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;
        let global_count = global_requests
            .iter()
            .filter(|&&request_time| now.duration_since(request_time) < self.config.window_duration)
            .count();

        // Tool status
        let tool_requests = self
            .tool_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;
        let mut tool_counts = HashMap::new();

        for (tool_name, requests) in tool_requests.iter() {
            let count = requests
                .iter()
                .filter(|&&request_time| {
                    now.duration_since(request_time) < self.config.window_duration
                })
                .count();
            tool_counts.insert(tool_name.clone(), count);
        }

        Ok(RateLimitStatus {
            global_requests: global_count,
            global_limit: self.config.max_requests as usize,
            tool_requests: tool_counts,
            tool_limit: (self.config.max_requests / 2).max(1) as usize,
            window_duration: self.config.window_duration,
        })
    }

    /// Reset rate limiting (for testing or admin purposes)
    pub fn reset(&self) -> Result<()> {
        let mut global_requests = self
            .global_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;
        global_requests.clear();

        let mut tool_requests = self
            .tool_requests
            .lock()
            .map_err(|_| AgentError::config("Rate limiter mutex poisoned".to_string()))?;
        tool_requests.clear();

        Ok(())
    }
}

/// Rate limit status information
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub global_requests: usize,
    pub global_limit: usize,
    pub tool_requests: HashMap<String, usize>,
    pub tool_limit: usize,
    pub window_duration: Duration,
}

impl RateLimitStatus {
    /// Check if global rate limit is close to being exceeded
    pub fn is_global_limit_close(&self) -> bool {
        self.global_requests as f64 / self.global_limit as f64 > 0.8
    }

    /// Check if any tool rate limit is close to being exceeded
    pub fn is_any_tool_limit_close(&self) -> bool {
        self.tool_requests
            .values()
            .any(|&count| count as f64 / self.tool_limit as f64 > 0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_global_rate_limiting() {
        let config = RateLimitConfig {
            max_requests: 3,
            window_duration: Duration::from_secs(1),
            per_tool_limiting: false,
        };
        let limiter = RateLimiter::new(config);

        // First 3 requests should succeed
        assert!(limiter.check_global_rate_limit().is_ok());
        assert!(limiter.check_global_rate_limit().is_ok());
        assert!(limiter.check_global_rate_limit().is_ok());

        // 4th request should fail
        assert!(limiter.check_global_rate_limit().is_err());

        // Wait for window to reset
        thread::sleep(Duration::from_millis(1100));

        // Should work again
        assert!(limiter.check_global_rate_limit().is_ok());
    }

    #[test]
    fn test_tool_rate_limiting() {
        let config = RateLimitConfig {
            max_requests: 4,
            window_duration: Duration::from_secs(1),
            per_tool_limiting: true,
        };
        let limiter = RateLimiter::new(config);

        // Tool limit is max_requests / 2 = 2
        assert!(limiter.check_tool_rate_limit("test_tool").is_ok());
        assert!(limiter.check_tool_rate_limit("test_tool").is_ok());

        // 3rd request should fail
        assert!(limiter.check_tool_rate_limit("test_tool").is_err());

        // Different tool should still work
        assert!(limiter.check_tool_rate_limit("other_tool").is_ok());
    }

    #[test]
    fn test_rate_limit_status() {
        let config = RateLimitConfig {
            max_requests: 10,
            window_duration: Duration::from_secs(60),
            per_tool_limiting: true,
        };
        let limiter = RateLimiter::new(config);

        // Make some requests
        let _ = limiter.check_rate_limit("tool1");
        let _ = limiter.check_rate_limit("tool1");
        let _ = limiter.check_rate_limit("tool2");

        let status = limiter.get_rate_limit_status().unwrap();
        assert_eq!(status.global_requests, 3);
        assert_eq!(status.tool_requests.get("tool1"), Some(&2));
        assert_eq!(status.tool_requests.get("tool2"), Some(&1));
    }
}
