use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use super::{RateLimitConfig, RateLimitStorageBackend};
use crate::utils::error::{AgentError, Result};

/// Rate limiting service trait
#[async_trait]
pub trait RateLimitService: Send + Sync {
    /// Check if a request is allowed
    async fn check_rate_limit(
        &self,
        key: &str,
        limit_type: RateLimitType,
    ) -> Result<RateLimitResult>;

    /// Record a request
    async fn record_request(&self, key: &str, limit_type: RateLimitType) -> Result<()>;

    /// Get current usage for a key
    async fn get_usage(&self, key: &str, limit_type: RateLimitType) -> Result<RateLimitUsage>;

    /// Reset rate limit for a key
    async fn reset_limit(&self, key: &str, limit_type: RateLimitType) -> Result<()>;

    /// Get rate limit statistics
    async fn get_statistics(&self) -> Result<RateLimitStatistics>;

    /// Clean up expired entries
    async fn cleanup_expired(&self) -> Result<u32>;
}

/// Rate limit types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RateLimitType {
    /// Requests per minute
    PerMinute,
    /// Requests per hour
    PerHour,
    /// Requests per day
    PerDay,
    /// Custom rate limit
    Custom(String),
}

/// Rate limit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Current request count
    pub current_count: u32,
    /// Maximum allowed requests
    pub limit: u32,
    /// Time until reset (in seconds)
    pub reset_time_seconds: u64,
    /// Remaining requests
    pub remaining: u32,
}

/// Rate limit usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitUsage {
    /// Current request count
    pub count: u32,
    /// Maximum allowed requests
    pub limit: u32,
    /// Window start time
    pub window_start: SystemTime,
    /// Window duration in seconds
    pub window_duration_seconds: u64,
    /// Time until reset
    pub reset_time: SystemTime,
}

/// Rate limit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatistics {
    /// Total requests processed
    pub total_requests: u64,
    /// Total requests blocked
    pub blocked_requests: u64,
    /// Active rate limit keys
    pub active_keys: u32,
    /// Requests by type
    pub requests_by_type: HashMap<RateLimitType, u64>,
    /// Top rate limited keys
    pub top_limited_keys: Vec<(String, u32)>,
}

/// Rate limit entry
#[derive(Debug, Clone)]
struct RateLimitEntry {
    /// Request count in current window
    count: u32,
    /// Window start time
    window_start: SystemTime,
    /// Last request time
    last_request: SystemTime,
}

/// In-memory rate limiting service
pub struct MemoryRateLimitService {
    /// Rate limit configuration
    config: RateLimitConfig,
    /// Rate limit entries by key and type
    entries: RwLock<HashMap<(String, RateLimitType), RateLimitEntry>>,
    /// Statistics
    statistics: RwLock<RateLimitStatistics>,
}

impl MemoryRateLimitService {
    /// Create a new memory rate limit service
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            statistics: RwLock::new(RateLimitStatistics {
                total_requests: 0,
                blocked_requests: 0,
                active_keys: 0,
                requests_by_type: HashMap::new(),
                top_limited_keys: Vec::new(),
            }),
        }
    }

    /// Get rate limit for a specific type
    fn get_limit(&self, limit_type: &RateLimitType) -> u32 {
        match limit_type {
            RateLimitType::PerMinute => self.config.max_requests_per_minute,
            RateLimitType::PerHour => self.config.max_requests_per_hour,
            RateLimitType::PerDay => self.config.max_requests_per_day,
            RateLimitType::Custom(_) => self.config.max_requests_per_minute, // Default fallback
        }
    }

    /// Get window duration for a specific type
    fn get_window_duration(&self, limit_type: &RateLimitType) -> Duration {
        match limit_type {
            RateLimitType::PerMinute => Duration::from_secs(60),
            RateLimitType::PerHour => Duration::from_secs(3600),
            RateLimitType::PerDay => Duration::from_secs(86400),
            RateLimitType::Custom(_) => Duration::from_secs(60), // Default fallback
        }
    }

    /// Check if window has expired
    fn is_window_expired(&self, entry: &RateLimitEntry, limit_type: &RateLimitType) -> bool {
        let window_duration = self.get_window_duration(limit_type);
        SystemTime::now()
            .duration_since(entry.window_start)
            .unwrap_or(Duration::ZERO)
            >= window_duration
    }

    /// Calculate reset time
    fn calculate_reset_time(
        &self,
        entry: &RateLimitEntry,
        limit_type: &RateLimitType,
    ) -> SystemTime {
        let window_duration = self.get_window_duration(limit_type);
        entry.window_start + window_duration
    }

    /// Update statistics
    async fn update_statistics(&self, limit_type: &RateLimitType, blocked: bool) {
        let mut stats = self.statistics.write().await;
        stats.total_requests += 1;
        if blocked {
            stats.blocked_requests += 1;
        }
        *stats
            .requests_by_type
            .entry(limit_type.clone())
            .or_insert(0) += 1;
    }
}

#[async_trait]
impl RateLimitService for MemoryRateLimitService {
    async fn check_rate_limit(
        &self,
        key: &str,
        limit_type: RateLimitType,
    ) -> Result<RateLimitResult> {
        if !self.config.enabled {
            return Ok(RateLimitResult {
                allowed: true,
                current_count: 0,
                limit: self.get_limit(&limit_type),
                reset_time_seconds: 0,
                remaining: self.get_limit(&limit_type),
            });
        }

        let limit = self.get_limit(&limit_type);
        let entry_key = (key.to_string(), limit_type.clone());

        let result = {
            let entries = self.entries.read().await;

            if let Some(entry) = entries.get(&entry_key) {
                if self.is_window_expired(entry, &limit_type) {
                    // Window expired, allow request
                    Some(RateLimitResult {
                        allowed: true,
                        current_count: 0,
                        limit,
                        reset_time_seconds: self.get_window_duration(&limit_type).as_secs(),
                        remaining: limit,
                    })
                } else {
                    // Check if within limit
                    let allowed =
                        entry.count < limit || entry.count < limit + self.config.burst_allowance;
                    let reset_time = self.calculate_reset_time(entry, &limit_type);
                    let reset_seconds = reset_time
                        .duration_since(SystemTime::now())
                        .unwrap_or(Duration::ZERO)
                        .as_secs();

                    Some(RateLimitResult {
                        allowed,
                        current_count: entry.count,
                        limit,
                        reset_time_seconds: reset_seconds,
                        remaining: limit.saturating_sub(entry.count),
                    })
                }
            } else {
                // No entry exists, allow request
                None
            }
        };

        if let Some(result) = result {
            self.update_statistics(&limit_type, !result.allowed).await;
            Ok(result)
        } else {
            self.update_statistics(&limit_type, false).await;
            Ok(RateLimitResult {
                allowed: true,
                current_count: 0,
                limit,
                reset_time_seconds: self.get_window_duration(&limit_type).as_secs(),
                remaining: limit,
            })
        }
    }

    async fn record_request(&self, key: &str, limit_type: RateLimitType) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut entries = self.entries.write().await;
        let entry_key = (key.to_string(), limit_type.clone());
        let now = SystemTime::now();

        match entries.get_mut(&entry_key) {
            Some(entry) => {
                if self.is_window_expired(entry, &limit_type) {
                    // Reset window
                    entry.count = 1;
                    entry.window_start = now;
                    entry.last_request = now;
                } else {
                    // Increment count
                    entry.count += 1;
                    entry.last_request = now;
                }
            }
            None => {
                // Create new entry
                entries.insert(
                    entry_key,
                    RateLimitEntry {
                        count: 1,
                        window_start: now,
                        last_request: now,
                    },
                );
            }
        }

        Ok(())
    }

    async fn get_usage(&self, key: &str, limit_type: RateLimitType) -> Result<RateLimitUsage> {
        let entries = self.entries.read().await;
        let entry_key = (key.to_string(), limit_type.clone());
        let limit = self.get_limit(&limit_type);
        let window_duration = self.get_window_duration(&limit_type);

        if let Some(entry) = entries.get(&entry_key) {
            let reset_time = self.calculate_reset_time(entry, &limit_type);

            Ok(RateLimitUsage {
                count: entry.count,
                limit,
                window_start: entry.window_start,
                window_duration_seconds: window_duration.as_secs(),
                reset_time,
            })
        } else {
            let now = SystemTime::now();
            Ok(RateLimitUsage {
                count: 0,
                limit,
                window_start: now,
                window_duration_seconds: window_duration.as_secs(),
                reset_time: now + window_duration,
            })
        }
    }

    async fn reset_limit(&self, key: &str, limit_type: RateLimitType) -> Result<()> {
        let mut entries = self.entries.write().await;
        let entry_key = (key.to_string(), limit_type);
        entries.remove(&entry_key);
        Ok(())
    }

    async fn get_statistics(&self) -> Result<RateLimitStatistics> {
        let entries = self.entries.read().await;
        let mut stats = self.statistics.read().await.clone();

        stats.active_keys = entries.len() as u32;

        // Calculate top limited keys
        let mut key_counts: HashMap<String, u32> = HashMap::new();
        for ((key, _), entry) in entries.iter() {
            *key_counts.entry(key.clone()).or_insert(0) += entry.count;
        }

        let mut top_keys: Vec<(String, u32)> = key_counts.into_iter().collect();
        top_keys.sort_by(|a, b| b.1.cmp(&a.1));
        top_keys.truncate(10); // Top 10
        stats.top_limited_keys = top_keys;

        Ok(stats)
    }

    async fn cleanup_expired(&self) -> Result<u32> {
        let mut entries = self.entries.write().await;
        let mut removed_count = 0;

        let expired_keys: Vec<_> = entries
            .iter()
            .filter(|((_, limit_type), entry)| self.is_window_expired(entry, limit_type))
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            entries.remove(&key);
            removed_count += 1;
        }

        Ok(removed_count)
    }
}

/// Create a rate limiting service
pub async fn create_rate_limit_service(
    config: &RateLimitConfig,
) -> Result<Box<dyn RateLimitService>> {
    match config.storage_backend {
        RateLimitStorageBackend::Memory => {
            Ok(Box::new(MemoryRateLimitService::new(config.clone())))
        }
        _ => Err(AgentError::validation(
            "Rate limit storage backend not supported".to_string(),
        )),
    }
}
