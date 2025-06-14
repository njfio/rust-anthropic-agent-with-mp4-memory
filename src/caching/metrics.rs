// Cache Metrics and Performance Monitoring
// Provides comprehensive metrics collection and analysis for cache performance optimization

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Cache metrics collector
pub struct CacheMetricsCollector {
    /// Metrics storage
    metrics: Arc<RwLock<CacheMetricsData>>,
    /// Performance history
    history: Arc<RwLock<PerformanceHistory>>,
    /// Metrics configuration
    config: MetricsConfig,
    /// Start time for uptime tracking
    start_time: Instant,
}

/// Cache metrics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetricsData {
    /// Overall cache statistics
    pub overall_stats: OverallStats,
    /// Per-tier statistics
    pub tier_stats: HashMap<String, TierMetricsData>,
    /// Per-policy statistics
    pub policy_stats: HashMap<String, PolicyMetricsData>,
    /// Operation statistics
    pub operation_stats: OperationStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Overall cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallStats {
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Overall hit ratio
    pub hit_ratio: f64,
    /// Total operations
    pub total_operations: u64,
    /// Total entries across all tiers
    pub total_entries: u64,
    /// Total memory usage in bytes
    pub total_memory_usage: u64,
    /// Cache uptime in seconds
    pub uptime_seconds: u64,
}

/// Tier-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierMetricsData {
    /// Tier name
    pub tier_name: String,
    /// Tier level
    pub tier_level: u8,
    /// Tier hits
    pub hits: u64,
    /// Tier misses
    pub misses: u64,
    /// Tier hit ratio
    pub hit_ratio: f64,
    /// Entry count
    pub entry_count: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Error count
    pub error_count: u64,
    /// Health score (0-100)
    pub health_score: u8,
}

/// Policy-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetricsData {
    /// Policy name
    pub policy_name: String,
    /// Entries using this policy
    pub entry_count: u64,
    /// Total hits for this policy
    pub hits: u64,
    /// Total misses for this policy
    pub misses: u64,
    /// Hit ratio for this policy
    pub hit_ratio: f64,
    /// Average TTL for this policy
    pub avg_ttl: f64,
    /// Eviction count
    pub evictions: u64,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    /// Get operations
    pub get_operations: u64,
    /// Set operations
    pub set_operations: u64,
    /// Delete operations
    pub delete_operations: u64,
    /// Clear operations
    pub clear_operations: u64,
    /// Invalidation operations
    pub invalidation_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average operation time in milliseconds
    pub avg_operation_time: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Latency percentiles in milliseconds
    pub latency_percentiles: LatencyPercentiles,
    /// Memory efficiency (hit ratio per MB)
    pub memory_efficiency: f64,
    /// Cache effectiveness score (0-100)
    pub effectiveness_score: u8,
    /// Cost savings ratio
    pub cost_savings_ratio: f64,
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p99_9: f64,
}

/// Performance history
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Hit ratio history (last 24 hours)
    pub hit_ratio_history: VecDeque<(DateTime<Utc>, f64)>,
    /// Throughput history
    pub throughput_history: VecDeque<(DateTime<Utc>, f64)>,
    /// Latency history
    pub latency_history: VecDeque<(DateTime<Utc>, f64)>,
    /// Memory usage history
    pub memory_usage_history: VecDeque<(DateTime<Utc>, u64)>,
    /// Error rate history
    pub error_rate_history: VecDeque<(DateTime<Utc>, f64)>,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable detailed metrics collection
    pub enable_detailed_metrics: bool,
    /// Metrics collection interval in seconds
    pub collection_interval: u64,
    /// History retention period in hours
    pub history_retention_hours: u64,
    /// Enable performance analysis
    pub enable_performance_analysis: bool,
    /// Enable cost analysis
    pub enable_cost_analysis: bool,
    /// Latency tracking sample rate (0.0-1.0)
    pub latency_sample_rate: f64,
}

/// Cache performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report generation time
    pub generated_at: DateTime<Utc>,
    /// Report period
    pub period: ReportPeriod,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Tier analysis
    pub tier_analysis: Vec<TierAnalysis>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Report period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportPeriod {
    LastHour,
    Last24Hours,
    LastWeek,
    LastMonth,
    Custom { start: DateTime<Utc>, end: DateTime<Utc> },
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Average hit ratio
    pub avg_hit_ratio: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Average latency
    pub avg_latency: f64,
    /// Total cost savings
    pub total_cost_savings: f64,
    /// Efficiency score
    pub efficiency_score: u8,
}

/// Tier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierAnalysis {
    /// Tier name
    pub tier_name: String,
    /// Performance score
    pub performance_score: u8,
    /// Utilization percentage
    pub utilization: f64,
    /// Effectiveness rating
    pub effectiveness: EffectivenessRating,
    /// Issues identified
    pub issues: Vec<String>,
}

/// Effectiveness rating
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffectivenessRating {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Hit ratio trend
    pub hit_ratio_trend: TrendDirection,
    /// Latency trend
    pub latency_trend: TrendDirection,
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Memory usage trend
    pub memory_usage_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_detailed_metrics: true,
            collection_interval: 60, // 1 minute
            history_retention_hours: 24,
            enable_performance_analysis: true,
            enable_cost_analysis: true,
            latency_sample_rate: 0.1, // 10% sampling
        }
    }
}

impl Default for CacheMetricsData {
    fn default() -> Self {
        Self {
            overall_stats: OverallStats::default(),
            tier_stats: HashMap::new(),
            policy_stats: HashMap::new(),
            operation_stats: OperationStats::default(),
            performance_metrics: PerformanceMetrics::default(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for OverallStats {
    fn default() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            hit_ratio: 0.0,
            total_operations: 0,
            total_entries: 0,
            total_memory_usage: 0,
            uptime_seconds: 0,
        }
    }
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            get_operations: 0,
            set_operations: 0,
            delete_operations: 0,
            clear_operations: 0,
            invalidation_operations: 0,
            failed_operations: 0,
            avg_operation_time: 0.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency_percentiles: LatencyPercentiles::default(),
            memory_efficiency: 0.0,
            effectiveness_score: 0,
            cost_savings_ratio: 0.0,
        }
    }
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p99_9: 0.0,
        }
    }
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self {
            hit_ratio_history: VecDeque::new(),
            throughput_history: VecDeque::new(),
            latency_history: VecDeque::new(),
            memory_usage_history: VecDeque::new(),
            error_rate_history: VecDeque::new(),
        }
    }
}

impl CacheMetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(CacheMetricsData::default())),
            history: Arc::new(RwLock::new(PerformanceHistory::default())),
            config,
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MetricsConfig::default())
    }

    /// Record a cache operation
    pub async fn record_operation(&self, operation_type: &str, duration: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        // Update operation stats
        match operation_type {
            "get" => metrics.operation_stats.get_operations += 1,
            "set" => metrics.operation_stats.set_operations += 1,
            "delete" => metrics.operation_stats.delete_operations += 1,
            "clear" => metrics.operation_stats.clear_operations += 1,
            "invalidate" => metrics.operation_stats.invalidation_operations += 1,
            _ => {}
        }
        
        if !success {
            metrics.operation_stats.failed_operations += 1;
        }
        
        // Update average operation time
        let duration_ms = duration.as_millis() as f64;
        metrics.operation_stats.avg_operation_time = 
            (metrics.operation_stats.avg_operation_time + duration_ms) / 2.0;
        
        metrics.overall_stats.total_operations += 1;
        metrics.last_updated = Utc::now();
    }

    /// Record a cache hit or miss
    pub async fn record_hit_miss(&self, tier_name: &str, hit: bool) {
        let mut metrics = self.metrics.write().await;
        
        // Update overall stats
        if hit {
            metrics.overall_stats.total_hits += 1;
        } else {
            metrics.overall_stats.total_misses += 1;
        }
        
        // Calculate hit ratio
        let total_requests = metrics.overall_stats.total_hits + metrics.overall_stats.total_misses;
        if total_requests > 0 {
            metrics.overall_stats.hit_ratio = 
                metrics.overall_stats.total_hits as f64 / total_requests as f64 * 100.0;
        }
        
        // Update tier stats
        let tier_stats = metrics.tier_stats.entry(tier_name.to_string()).or_insert(TierMetricsData {
            tier_name: tier_name.to_string(),
            tier_level: 0,
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            entry_count: 0,
            memory_usage: 0,
            avg_response_time: 0.0,
            error_count: 0,
            health_score: 100,
        });
        
        if hit {
            tier_stats.hits += 1;
        } else {
            tier_stats.misses += 1;
        }
        
        // Calculate tier hit ratio
        let tier_total = tier_stats.hits + tier_stats.misses;
        if tier_total > 0 {
            tier_stats.hit_ratio = tier_stats.hits as f64 / tier_total as f64 * 100.0;
        }
    }

    /// Update tier statistics
    pub async fn update_tier_stats(&self, tier_name: &str, entry_count: u64, memory_usage: u64, avg_response_time: f64) {
        let mut metrics = self.metrics.write().await;
        
        if let Some(tier_stats) = metrics.tier_stats.get_mut(tier_name) {
            tier_stats.entry_count = entry_count;
            tier_stats.memory_usage = memory_usage;
            tier_stats.avg_response_time = avg_response_time;
        }
    }

    /// Calculate performance metrics
    pub async fn calculate_performance_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        
        // Calculate throughput (operations per second)
        let uptime = self.start_time.elapsed().as_secs() as f64;
        if uptime > 0.0 {
            metrics.performance_metrics.throughput = 
                metrics.overall_stats.total_operations as f64 / uptime;
        }
        
        // Calculate memory efficiency
        if metrics.overall_stats.total_memory_usage > 0 {
            metrics.performance_metrics.memory_efficiency = 
                metrics.overall_stats.hit_ratio / (metrics.overall_stats.total_memory_usage as f64 / 1_000_000.0);
        }
        
        // Calculate effectiveness score
        let hit_ratio_score = (metrics.overall_stats.hit_ratio / 100.0 * 40.0) as u8;
        let throughput_score = (metrics.performance_metrics.throughput.min(1000.0) / 1000.0 * 30.0) as u8;
        let latency_score = if metrics.operation_stats.avg_operation_time > 0.0 {
            ((100.0 - metrics.operation_stats.avg_operation_time.min(100.0)) / 100.0 * 30.0) as u8
        } else {
            30
        };
        
        metrics.performance_metrics.effectiveness_score = hit_ratio_score + throughput_score + latency_score;
        
        // Update uptime
        metrics.overall_stats.uptime_seconds = uptime as u64;
    }

    /// Add data point to history
    pub async fn add_to_history(&self) {
        let metrics = self.metrics.read().await;
        let mut history = self.history.write().await;
        let now = Utc::now();
        
        // Add current metrics to history
        history.hit_ratio_history.push_back((now, metrics.overall_stats.hit_ratio));
        history.throughput_history.push_back((now, metrics.performance_metrics.throughput));
        history.latency_history.push_back((now, metrics.operation_stats.avg_operation_time));
        history.memory_usage_history.push_back((now, metrics.overall_stats.total_memory_usage));
        
        // Calculate error rate
        let error_rate = if metrics.overall_stats.total_operations > 0 {
            metrics.operation_stats.failed_operations as f64 / metrics.overall_stats.total_operations as f64 * 100.0
        } else {
            0.0
        };
        history.error_rate_history.push_back((now, error_rate));
        
        // Trim history to retention period
        let retention_cutoff = now - chrono::Duration::hours(self.config.history_retention_hours as i64);
        
        history.hit_ratio_history.retain(|(timestamp, _)| *timestamp > retention_cutoff);
        history.throughput_history.retain(|(timestamp, _)| *timestamp > retention_cutoff);
        history.latency_history.retain(|(timestamp, _)| *timestamp > retention_cutoff);
        history.memory_usage_history.retain(|(timestamp, _)| *timestamp > retention_cutoff);
        history.error_rate_history.retain(|(timestamp, _)| *timestamp > retention_cutoff);
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> CacheMetricsData {
        self.metrics.read().await.clone()
    }

    /// Get performance history
    pub async fn get_history(&self) -> PerformanceHistory {
        self.history.read().await.clone()
    }

    /// Generate performance report
    pub async fn generate_report(&self, period: ReportPeriod) -> PerformanceReport {
        let metrics = self.get_metrics().await;
        let history = self.get_history().await;
        
        // Calculate summary
        let summary = ReportSummary {
            avg_hit_ratio: metrics.overall_stats.hit_ratio,
            peak_throughput: history.throughput_history.iter()
                .map(|(_, throughput)| *throughput)
                .fold(0.0, f64::max),
            avg_latency: metrics.operation_stats.avg_operation_time,
            total_cost_savings: metrics.performance_metrics.cost_savings_ratio * 100.0,
            efficiency_score: metrics.performance_metrics.effectiveness_score,
        };
        
        // Analyze tiers
        let tier_analysis: Vec<TierAnalysis> = metrics.tier_stats.values()
            .map(|tier| {
                let performance_score = ((tier.hit_ratio + (100.0 - tier.avg_response_time.min(100.0))) / 2.0) as u8;
                let utilization = if tier.memory_usage > 0 { 75.0 } else { 0.0 }; // Simplified
                let effectiveness = match performance_score {
                    90..=100 => EffectivenessRating::Excellent,
                    80..=89 => EffectivenessRating::Good,
                    70..=79 => EffectivenessRating::Fair,
                    50..=69 => EffectivenessRating::Poor,
                    _ => EffectivenessRating::Critical,
                };
                
                TierAnalysis {
                    tier_name: tier.tier_name.clone(),
                    performance_score,
                    utilization,
                    effectiveness,
                    issues: Vec::new(), // Would be populated with actual analysis
                }
            })
            .collect();
        
        // Calculate trends (simplified)
        let trends = PerformanceTrends {
            hit_ratio_trend: TrendDirection::Stable,
            latency_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Stable,
            memory_usage_trend: TrendDirection::Stable,
        };
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        if metrics.overall_stats.hit_ratio < 80.0 {
            recommendations.push("Consider increasing cache TTL or adjusting eviction policies".to_string());
        }
        if metrics.operation_stats.avg_operation_time > 50.0 {
            recommendations.push("High latency detected - consider optimizing cache tier configuration".to_string());
        }
        
        PerformanceReport {
            generated_at: Utc::now(),
            period,
            summary,
            tier_analysis,
            trends,
            recommendations,
        }
    }

    /// Start metrics collection background task
    pub async fn start_collection_task(&self) -> tokio::task::JoinHandle<()> {
        let collector = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_secs(collector.config.collection_interval)
            );
            
            loop {
                interval.tick().await;
                
                collector.calculate_performance_metrics().await;
                collector.add_to_history().await;
                
                debug!("Cache metrics updated");
            }
        })
    }
}

impl Clone for CacheMetricsCollector {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
            history: Arc::clone(&self.history),
            config: self.config.clone(),
            start_time: self.start_time,
        }
    }
}
