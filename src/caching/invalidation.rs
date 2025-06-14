// Cache Invalidation System
// Provides intelligent cache invalidation strategies including pattern-based, tag-based, and time-based invalidation

use super::{CacheTier};
use crate::utils::error::{AgentError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache invalidation manager
pub struct InvalidationManager {
    /// Invalidation rules
    rules: Arc<RwLock<HashMap<String, InvalidationRule>>>,
    /// Tag mappings (tag -> set of keys)
    tag_mappings: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Dependency mappings (key -> set of dependent keys)
    dependency_mappings: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    /// Invalidation statistics
    stats: Arc<RwLock<InvalidationStats>>,
}

/// Invalidation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: InvalidationType,
    /// Rule pattern or condition
    pub pattern: String,
    /// Rule priority (higher = more important)
    pub priority: u8,
    /// Whether rule is enabled
    pub enabled: bool,
    /// Rule creation time
    pub created_at: DateTime<Utc>,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Types of cache invalidation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InvalidationType {
    /// Pattern-based invalidation (glob patterns)
    Pattern,
    /// Tag-based invalidation
    Tag,
    /// Time-based invalidation (TTL)
    Time,
    /// Dependency-based invalidation
    Dependency,
    /// Manual invalidation
    Manual,
    /// Event-driven invalidation
    Event,
}

/// Invalidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationStats {
    /// Total invalidations performed
    pub total_invalidations: u64,
    /// Invalidations by type
    pub invalidations_by_type: HashMap<InvalidationType, u64>,
    /// Total keys invalidated
    pub total_keys_invalidated: u64,
    /// Average invalidation time in milliseconds
    pub avg_invalidation_time: f64,
    /// Last invalidation timestamp
    pub last_invalidation: Option<DateTime<Utc>>,
    /// Failed invalidations
    pub failed_invalidations: u64,
}

/// Invalidation result
#[derive(Debug, Clone)]
pub struct InvalidationResult {
    /// Number of keys invalidated
    pub keys_invalidated: u64,
    /// Tiers affected
    pub tiers_affected: Vec<String>,
    /// Invalidation duration
    pub duration: std::time::Duration,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Cache dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheDependency {
    /// Source key
    pub source_key: String,
    /// Dependent keys
    pub dependent_keys: HashSet<String>,
    /// Dependency type
    pub dependency_type: DependencyType,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Types of cache dependencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Strong dependency (invalidate immediately)
    Strong,
    /// Weak dependency (invalidate eventually)
    Weak,
    /// Conditional dependency (invalidate based on condition)
    Conditional,
}

impl Default for InvalidationStats {
    fn default() -> Self {
        Self {
            total_invalidations: 0,
            invalidations_by_type: HashMap::new(),
            total_keys_invalidated: 0,
            avg_invalidation_time: 0.0,
            last_invalidation: None,
            failed_invalidations: 0,
        }
    }
}

impl InvalidationManager {
    /// Create a new invalidation manager
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(HashMap::new())),
            tag_mappings: Arc::new(RwLock::new(HashMap::new())),
            dependency_mappings: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(InvalidationStats::default())),
        }
    }

    /// Add an invalidation rule
    pub async fn add_rule(&self, rule: InvalidationRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.name.clone(), rule.clone());
        
        info!("Added invalidation rule: {} (type: {:?})", rule.name, rule.rule_type);
        Ok(())
    }

    /// Remove an invalidation rule
    pub async fn remove_rule(&self, rule_name: &str) -> Result<bool> {
        let mut rules = self.rules.write().await;
        let removed = rules.remove(rule_name).is_some();
        
        if removed {
            info!("Removed invalidation rule: {}", rule_name);
        }
        
        Ok(removed)
    }

    /// Register a tag mapping
    pub async fn register_tag(&self, key: &str, tags: Vec<String>) -> Result<()> {
        let mut tag_mappings = self.tag_mappings.write().await;
        
        for tag in tags {
            tag_mappings.entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(key.to_string());
        }
        
        debug!("Registered tags for key: {}", key);
        Ok(())
    }

    /// Unregister tag mappings for a key
    pub async fn unregister_tags(&self, key: &str) -> Result<()> {
        let mut tag_mappings = self.tag_mappings.write().await;
        
        // Remove key from all tag mappings
        for (_, keys) in tag_mappings.iter_mut() {
            keys.remove(key);
        }
        
        // Clean up empty tag mappings
        tag_mappings.retain(|_, keys| !keys.is_empty());
        
        debug!("Unregistered tags for key: {}", key);
        Ok(())
    }

    /// Add a cache dependency
    pub async fn add_dependency(&self, source_key: &str, dependent_keys: Vec<String>, dependency_type: DependencyType) -> Result<()> {
        let mut dependency_mappings = self.dependency_mappings.write().await;
        
        dependency_mappings.entry(source_key.to_string())
            .or_insert_with(HashSet::new)
            .extend(dependent_keys.iter().cloned());
        
        debug!("Added dependency: {} -> {:?} (type: {:?})", source_key, dependent_keys, dependency_type);
        Ok(())
    }

    /// Remove a cache dependency
    pub async fn remove_dependency(&self, source_key: &str) -> Result<bool> {
        let mut dependency_mappings = self.dependency_mappings.write().await;
        let removed = dependency_mappings.remove(source_key).is_some();
        
        if removed {
            debug!("Removed dependency for key: {}", source_key);
        }
        
        Ok(removed)
    }

    /// Invalidate cache entries by pattern
    pub async fn invalidate_pattern(&self, pattern: &str, tiers: &[Arc<dyn CacheTier>]) -> Result<u64> {
        let start_time = std::time::Instant::now();
        let mut total_invalidated = 0u64;
        let mut errors = Vec::new();
        let mut tiers_affected = Vec::new();

        info!("Starting pattern invalidation: {}", pattern);

        // For each tier, we need to find matching keys
        // Note: This is a simplified implementation. In practice, you'd need
        // tier-specific pattern matching (e.g., Redis SCAN with pattern)
        for tier in tiers {
            match self.invalidate_pattern_in_tier(pattern, tier).await {
                Ok(count) => {
                    total_invalidated += count;
                    if count > 0 {
                        tiers_affected.push(tier.name().to_string());
                    }
                }
                Err(e) => {
                    errors.push(format!("Tier {}: {}", tier.name(), e));
                    warn!("Pattern invalidation failed in tier {}: {}", tier.name(), e);
                }
            }
        }

        let duration = start_time.elapsed();
        
        // Update statistics
        self.update_invalidation_stats(InvalidationType::Pattern, total_invalidated, duration, errors.is_empty()).await;

        if !errors.is_empty() {
            warn!("Pattern invalidation completed with errors: {:?}", errors);
        } else {
            info!("Pattern invalidation completed: {} keys invalidated in {:?}", total_invalidated, duration);
        }

        Ok(total_invalidated)
    }

    /// Invalidate cache entries by tags
    pub async fn invalidate_tags(&self, tags: &[String], tiers: &[Arc<dyn CacheTier>]) -> Result<u64> {
        let start_time = std::time::Instant::now();
        let mut total_invalidated = 0u64;
        let mut errors = Vec::new();

        info!("Starting tag invalidation: {:?}", tags);

        // Get all keys associated with the tags
        let keys_to_invalidate = {
            let tag_mappings = self.tag_mappings.read().await;
            let mut keys = HashSet::new();
            
            for tag in tags {
                if let Some(tag_keys) = tag_mappings.get(tag) {
                    keys.extend(tag_keys.iter().cloned());
                }
            }
            
            keys
        };

        // Invalidate keys in all tiers
        for key in &keys_to_invalidate {
            for tier in tiers {
                match tier.delete(key).await {
                    Ok(true) => total_invalidated += 1,
                    Ok(false) => {}, // Key didn't exist
                    Err(e) => {
                        errors.push(format!("Tier {} key {}: {}", tier.name(), key, e));
                    }
                }
            }
        }

        // Remove tag mappings for invalidated keys
        {
            let mut tag_mappings = self.tag_mappings.write().await;
            for tag in tags {
                if let Some(tag_keys) = tag_mappings.get_mut(tag) {
                    for key in &keys_to_invalidate {
                        tag_keys.remove(key);
                    }
                }
            }
            
            // Clean up empty tag mappings
            for tag in tags {
                if let Some(tag_keys) = tag_mappings.get(tag) {
                    if tag_keys.is_empty() {
                        tag_mappings.remove(tag);
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        
        // Update statistics
        self.update_invalidation_stats(InvalidationType::Tag, total_invalidated, duration, errors.is_empty()).await;

        if !errors.is_empty() {
            warn!("Tag invalidation completed with errors: {:?}", errors);
        } else {
            info!("Tag invalidation completed: {} keys invalidated in {:?}", total_invalidated, duration);
        }

        Ok(total_invalidated)
    }

    /// Invalidate cache entries by dependency
    pub async fn invalidate_dependencies(&self, source_key: &str, tiers: &[Arc<dyn CacheTier>]) -> Result<u64> {
        let start_time = std::time::Instant::now();
        let mut total_invalidated = 0u64;

        info!("Starting dependency invalidation for key: {}", source_key);

        // Get dependent keys
        let dependent_keys = {
            let dependency_mappings = self.dependency_mappings.read().await;
            dependency_mappings.get(source_key)
                .cloned()
                .unwrap_or_default()
        };

        // Invalidate dependent keys
        for key in &dependent_keys {
            for tier in tiers {
                match tier.delete(key).await {
                    Ok(true) => total_invalidated += 1,
                    Ok(false) => {}, // Key didn't exist
                    Err(e) => {
                        warn!("Failed to invalidate dependent key {} in tier {}: {}", key, tier.name(), e);
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        
        // Update statistics
        self.update_invalidation_stats(InvalidationType::Dependency, total_invalidated, duration, true).await;

        info!("Dependency invalidation completed: {} keys invalidated in {:?}", total_invalidated, duration);
        Ok(total_invalidated)
    }

    /// Invalidate a specific key manually
    pub async fn invalidate_key(&self, key: &str, tiers: &[Arc<dyn CacheTier>]) -> Result<bool> {
        let start_time = std::time::Instant::now();
        let mut invalidated = false;

        for tier in tiers {
            match tier.delete(key).await {
                Ok(true) => invalidated = true,
                Ok(false) => {}, // Key didn't exist in this tier
                Err(e) => {
                    warn!("Failed to invalidate key {} in tier {}: {}", key, tier.name(), e);
                }
            }
        }

        // Also invalidate dependencies
        self.invalidate_dependencies(key, tiers).await?;

        // Remove from tag mappings
        self.unregister_tags(key).await?;

        let duration = start_time.elapsed();
        
        // Update statistics
        self.update_invalidation_stats(InvalidationType::Manual, if invalidated { 1 } else { 0 }, duration, true).await;

        debug!("Manual invalidation of key {} completed in {:?}", key, duration);
        Ok(invalidated)
    }

    /// Get invalidation statistics
    pub async fn get_stats(&self) -> InvalidationStats {
        self.stats.read().await.clone()
    }

    /// Clear all invalidation data
    pub async fn clear_all(&self) -> Result<()> {
        {
            let mut rules = self.rules.write().await;
            rules.clear();
        }
        
        {
            let mut tag_mappings = self.tag_mappings.write().await;
            tag_mappings.clear();
        }
        
        {
            let mut dependency_mappings = self.dependency_mappings.write().await;
            dependency_mappings.clear();
        }
        
        {
            let mut stats = self.stats.write().await;
            *stats = InvalidationStats::default();
        }

        info!("Cleared all invalidation data");
        Ok(())
    }

    /// Invalidate pattern in a specific tier (simplified implementation)
    async fn invalidate_pattern_in_tier(&self, pattern: &str, tier: &Arc<dyn CacheTier>) -> Result<u64> {
        // This is a simplified implementation. In practice, you would:
        // 1. For Redis: Use SCAN command with pattern matching
        // 2. For Memory cache: Iterate through keys and match pattern
        // 3. For other backends: Use backend-specific pattern matching
        
        // For now, we'll return 0 as this requires tier-specific implementation
        debug!("Pattern invalidation in tier {} not fully implemented for pattern: {}", tier.name(), pattern);
        Ok(0)
    }

    /// Update invalidation statistics
    async fn update_invalidation_stats(&self, invalidation_type: InvalidationType, keys_invalidated: u64, duration: std::time::Duration, success: bool) {
        let mut stats = self.stats.write().await;
        
        stats.total_invalidations += 1;
        stats.total_keys_invalidated += keys_invalidated;
        stats.last_invalidation = Some(Utc::now());
        
        if !success {
            stats.failed_invalidations += 1;
        }
        
        // Update average invalidation time
        let duration_ms = duration.as_millis() as f64;
        stats.avg_invalidation_time = (stats.avg_invalidation_time + duration_ms) / 2.0;
        
        // Update by-type statistics
        let type_count = stats.invalidations_by_type.entry(invalidation_type).or_insert(0);
        *type_count += 1;
    }
}

impl Default for InvalidationManager {
    fn default() -> Self {
        Self::new()
    }
}
