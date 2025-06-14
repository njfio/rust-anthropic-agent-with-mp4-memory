// Alert Management System for Performance Monitoring
// Provides threshold-based alerting, escalation policies, and notification channels

use super::{AlertThresholds, Metric, MetricValue};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Alert manager for handling performance alerts
pub struct AlertManager {
    /// Alert configuration
    thresholds: AlertThresholds,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    /// Alert history
    alert_history: Arc<RwLock<Vec<Alert>>>,
    /// Notification channels
    notification_channels: Arc<RwLock<Vec<Box<dyn NotificationChannel>>>>,
    /// Alert rules
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    /// Alert statistics
    stats: Arc<RwLock<AlertStats>>,
}

/// Alert data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert name
    pub name: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric that triggered the alert
    pub metric_name: String,
    /// Current metric value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Alert status
    pub status: AlertStatus,
    /// Labels associated with the alert
    pub labels: HashMap<String, String>,
    /// Resolution timestamp
    pub resolved_at: Option<DateTime<Utc>>,
    /// Resolution reason
    pub resolution_reason: Option<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Alert status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Resolved,
    Suppressed,
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric name to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Evaluation window (seconds)
    pub evaluation_window: u64,
    /// Minimum duration before alerting (seconds)
    pub min_duration: u64,
    /// Alert message template
    pub message_template: String,
    /// Labels to add to alerts
    pub labels: HashMap<String, String>,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Comparison operators for alert rules
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    /// Total alerts triggered
    pub total_alerts: u64,
    /// Active alerts count
    pub active_alerts_count: u32,
    /// Resolved alerts count
    pub resolved_alerts_count: u64,
    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, u32>,
    /// Average resolution time (seconds)
    pub avg_resolution_time_seconds: f64,
    /// Last alert timestamp
    pub last_alert_time: Option<DateTime<Utc>>,
}

/// Alert manager health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerHealth {
    pub is_healthy: bool,
    pub active_rules_count: u32,
    pub notification_channels_count: u32,
    pub last_evaluation: Option<DateTime<Utc>>,
    pub error_count: u32,
}

/// Trait for notification channels
#[async_trait::async_trait]
pub trait NotificationChannel: Send + Sync {
    /// Get channel name
    fn name(&self) -> &str;
    
    /// Send alert notification
    async fn send_alert(&self, alert: &Alert) -> Result<()>;
    
    /// Test the notification channel
    async fn test(&self) -> Result<()>;
    
    /// Get channel configuration
    fn config(&self) -> NotificationConfig;
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub name: String,
    pub enabled: bool,
    pub timeout: Duration,
    pub retry_count: u32,
    pub rate_limit: Option<u32>, // Max notifications per minute
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(thresholds: AlertThresholds) -> Self {
        Self {
            thresholds,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            notification_channels: Arc::new(RwLock::new(Vec::new())),
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AlertStats::default())),
        }
    }

    /// Start the alert manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting alert manager");
        
        // Initialize default alert rules
        self.initialize_default_rules().await?;
        
        info!("Alert manager started successfully");
        Ok(())
    }

    /// Stop the alert manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping alert manager");
        
        // Resolve all active alerts
        let mut active_alerts = self.active_alerts.write().await;
        for alert in active_alerts.values_mut() {
            alert.status = AlertStatus::Resolved;
            alert.resolved_at = Some(Utc::now());
            alert.resolution_reason = Some("System shutdown".to_string());
        }
        
        info!("Alert manager stopped");
        Ok(())
    }

    /// Add a notification channel
    pub async fn add_notification_channel(&self, channel: Box<dyn NotificationChannel>) -> Result<()> {
        let name = channel.name().to_string();
        let mut channels = self.notification_channels.write().await;
        channels.push(channel);
        
        info!("Added notification channel: {}", name);
        Ok(())
    }

    /// Add an alert rule
    pub async fn add_alert_rule(&self, rule: AlertRule) -> Result<()> {
        let name = rule.name.clone();
        let mut rules = self.alert_rules.write().await;
        rules.insert(name.clone(), rule);
        
        info!("Added alert rule: {}", name);
        Ok(())
    }

    /// Check metrics against alert rules
    pub async fn check_metrics(&self, metrics: &[Metric]) -> Result<()> {
        let rules = self.alert_rules.read().await;
        
        for metric in metrics {
            // Check built-in thresholds
            self.check_builtin_thresholds(metric).await?;
            
            // Check custom rules
            for rule in rules.values() {
                if rule.enabled && rule.metric_name == metric.name {
                    self.evaluate_rule(rule, metric).await?;
                }
            }
        }
        
        Ok(())
    }

    /// Check built-in threshold alerts
    async fn check_builtin_thresholds(&self, metric: &Metric) -> Result<()> {
        let value = match &metric.value {
            MetricValue::Gauge(v) => *v,
            MetricValue::Counter(v) => *v as f64,
            _ => return Ok(()), // Skip non-numeric metrics
        };

        let (threshold, severity) = match metric.name.as_str() {
            "cpu_usage" => (self.thresholds.cpu_threshold, AlertSeverity::Warning),
            "memory_usage" => (self.thresholds.memory_threshold, AlertSeverity::Warning),
            "disk_usage" => (self.thresholds.disk_threshold, AlertSeverity::Critical),
            "error_rate" => (self.thresholds.error_rate_threshold, AlertSeverity::Warning),
            _ => return Ok(()),
        };

        if value > threshold {
            let alert_id = format!("builtin_{}_{}", metric.name, Utc::now().timestamp());
            let alert = Alert {
                id: alert_id.clone(),
                name: format!("High {}", metric.name),
                severity,
                message: format!("{} is {}%, exceeding threshold of {}%", metric.name, value, threshold),
                metric_name: metric.name.clone(),
                current_value: value,
                threshold_value: threshold,
                timestamp: Utc::now(),
                status: AlertStatus::Active,
                labels: metric.labels.clone(),
                resolved_at: None,
                resolution_reason: None,
            };

            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Evaluate a custom alert rule
    async fn evaluate_rule(&self, rule: &AlertRule, metric: &Metric) -> Result<()> {
        let value = match &metric.value {
            MetricValue::Gauge(v) => *v,
            MetricValue::Counter(v) => *v as f64,
            _ => return Ok(()),
        };

        let should_alert = match rule.operator {
            ComparisonOperator::GreaterThan => value > rule.threshold,
            ComparisonOperator::LessThan => value < rule.threshold,
            ComparisonOperator::GreaterThanOrEqual => value >= rule.threshold,
            ComparisonOperator::LessThanOrEqual => value <= rule.threshold,
            ComparisonOperator::Equal => (value - rule.threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - rule.threshold).abs() >= f64::EPSILON,
        };

        if should_alert {
            let alert_id = format!("rule_{}_{}", rule.name, Utc::now().timestamp());
            let mut labels = metric.labels.clone();
            labels.extend(rule.labels.clone());

            let alert = Alert {
                id: alert_id.clone(),
                name: rule.name.clone(),
                severity: rule.severity.clone(),
                message: rule.message_template.replace("{value}", &value.to_string())
                    .replace("{threshold}", &rule.threshold.to_string()),
                metric_name: metric.name.clone(),
                current_value: value,
                threshold_value: rule.threshold,
                timestamp: Utc::now(),
                status: AlertStatus::Active,
                labels,
                resolved_at: None,
                resolution_reason: None,
            };

            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Trigger an alert
    async fn trigger_alert(&self, alert: Alert) -> Result<()> {
        let alert_id = alert.id.clone();
        
        // Check if alert is already active
        {
            let active_alerts = self.active_alerts.read().await;
            if active_alerts.contains_key(&alert_id) {
                return Ok(()); // Alert already active
            }
        }

        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert_id.clone(), alert.clone());
        }

        // Add to history
        {
            let mut history = self.alert_history.write().await;
            history.push(alert.clone());
            
            // Keep only last 1000 alerts in history
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_alerts += 1;
            stats.active_alerts_count += 1;
            stats.last_alert_time = Some(alert.timestamp);
            
            let severity_count = stats.alerts_by_severity.entry(alert.severity.clone()).or_insert(0);
            *severity_count += 1;
        }

        // Send notifications
        self.send_notifications(&alert).await?;

        warn!("Alert triggered: {} - {}", alert.name, alert.message);
        Ok(())
    }

    /// Send alert notifications
    async fn send_notifications(&self, alert: &Alert) -> Result<()> {
        let channels = self.notification_channels.read().await;
        
        for channel in channels.iter() {
            match channel.send_alert(alert).await {
                Ok(_) => {
                    debug!("Alert notification sent via {}", channel.name());
                }
                Err(e) => {
                    error!("Failed to send alert via {}: {}", channel.name(), e);
                }
            }
        }
        
        Ok(())
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: &str, reason: Option<String>) -> Result<()> {
        let mut active_alerts = self.active_alerts.write().await;
        
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Resolved;
            alert.resolved_at = Some(Utc::now());
            alert.resolution_reason = reason;
            
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.active_alerts_count = stats.active_alerts_count.saturating_sub(1);
                stats.resolved_alerts_count += 1;
                
                // Update average resolution time
                if let Some(resolved_at) = alert.resolved_at {
                    let resolution_time = (resolved_at - alert.timestamp).num_seconds() as f64;
                    stats.avg_resolution_time_seconds = 
                        (stats.avg_resolution_time_seconds + resolution_time) / 2.0;
                }
            }
            
            info!("Alert resolved: {}", alert_id);
            Ok(())
        } else {
            Err(AgentError::validation(format!("Alert not found: {}", alert_id)))
        }
    }

    /// Get alert statistics
    pub async fn get_stats(&self) -> AlertStats {
        self.stats.read().await.clone()
    }

    /// Get health status
    pub async fn get_health(&self) -> Result<AlertManagerHealth> {
        let rules = self.alert_rules.read().await;
        let channels = self.notification_channels.read().await;
        let stats = self.stats.read().await;

        Ok(AlertManagerHealth {
            is_healthy: true, // TODO: Add actual health checks
            active_rules_count: rules.len() as u32,
            notification_channels_count: channels.len() as u32,
            last_evaluation: stats.last_alert_time,
            error_count: 0, // TODO: Track errors
        })
    }

    /// Initialize default alert rules
    async fn initialize_default_rules(&self) -> Result<()> {
        let default_rules = vec![
            AlertRule {
                name: "high_cpu_usage".to_string(),
                metric_name: "cpu_usage".to_string(),
                threshold: self.thresholds.cpu_threshold,
                operator: ComparisonOperator::GreaterThan,
                severity: AlertSeverity::Warning,
                evaluation_window: 300,
                min_duration: 60,
                message_template: "CPU usage is {value}%, exceeding threshold of {threshold}%".to_string(),
                labels: HashMap::new(),
                enabled: true,
            },
            AlertRule {
                name: "high_memory_usage".to_string(),
                metric_name: "memory_usage".to_string(),
                threshold: self.thresholds.memory_threshold,
                operator: ComparisonOperator::GreaterThan,
                severity: AlertSeverity::Warning,
                evaluation_window: 300,
                min_duration: 60,
                message_template: "Memory usage is {value}%, exceeding threshold of {threshold}%".to_string(),
                labels: HashMap::new(),
                enabled: true,
            },
        ];

        for rule in default_rules {
            self.add_alert_rule(rule).await?;
        }

        Ok(())
    }
}

impl Default for AlertStats {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            active_alerts_count: 0,
            resolved_alerts_count: 0,
            alerts_by_severity: HashMap::new(),
            avg_resolution_time_seconds: 0.0,
            last_alert_time: None,
        }
    }
}
