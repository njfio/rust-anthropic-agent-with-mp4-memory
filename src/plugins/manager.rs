use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use serde_json::Value;

use crate::utils::error::{AgentError, Result};
use super::{
    Plugin, PluginMetadata, PluginConfig, PluginContext, PluginResult, 
    PluginStats, PluginEvent, PluginHealthStatus, PluginPermissions
};
use super::registry::PluginRegistry;
use super::loader::PluginLoader;

/// Advanced plugin manager with lifecycle management, security, and monitoring
pub struct AdvancedPluginManager {
    /// Loaded plugins
    plugins: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
    /// Plugin configurations
    configs: Arc<RwLock<HashMap<String, PluginConfig>>>,
    /// Plugin statistics
    stats: Arc<RwLock<HashMap<String, PluginStats>>>,
    /// Plugin health status
    health_status: Arc<RwLock<HashMap<String, PluginHealthStatus>>>,
    /// Plugin registry
    registry: Arc<RwLock<PluginRegistry>>,
    /// Plugin loader
    loader: PluginLoader,
    /// Base plugin directory
    plugin_dir: PathBuf,
    /// Plugin data directory
    data_dir: PathBuf,
    /// Security manager
    security_manager: PluginSecurityManager,
    /// Resource monitor
    resource_monitor: PluginResourceMonitor,
}

/// Plugin security manager for enforcing permissions and sandboxing
#[derive(Debug, Clone)]
pub struct PluginSecurityManager {
    /// Global security settings
    global_permissions: PluginPermissions,
    /// Plugin-specific permission overrides
    plugin_permissions: HashMap<String, PluginPermissions>,
    /// Blocked plugins
    blocked_plugins: Vec<String>,
}

/// Plugin resource monitor for tracking resource usage
#[derive(Debug, Clone)]
pub struct PluginResourceMonitor {
    /// Memory usage limits
    memory_limits: HashMap<String, u64>,
    /// CPU usage limits
    cpu_limits: HashMap<String, f64>,
    /// Execution time limits
    time_limits: HashMap<String, u64>,
    /// Current resource usage
    current_usage: HashMap<String, ResourceUsage>,
}

/// Current resource usage for a plugin
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU usage percentage
    pub cpu_percent: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of active executions
    pub active_executions: u32,
}

impl AdvancedPluginManager {
    /// Create a new advanced plugin manager
    pub fn new(plugin_dir: PathBuf, data_dir: PathBuf) -> Self {
        let registry_path = data_dir.join("plugin_registry.toml");
        
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            health_status: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(PluginRegistry::with_storage(registry_path))),
            loader: PluginLoader::new(),
            plugin_dir,
            data_dir,
            security_manager: PluginSecurityManager::new(),
            resource_monitor: PluginResourceMonitor::new(),
        }
    }

    /// Initialize the plugin manager
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing advanced plugin manager");

        // Load registry from storage
        self.registry.write().await.load_registry().await?;

        // Discover plugins in the plugin directory
        let discovery_result = self.registry.read().await.discover_plugins(self.plugin_dir.clone()).await?;
        
        info!("Discovered {} plugins", discovery_result.plugins.len());
        
        for error in &discovery_result.errors {
            warn!("Plugin discovery error: {}", error);
        }

        // Auto-load discovered plugins
        for metadata in discovery_result.plugins {
            let plugin_id = metadata.id.clone();
            if let Err(e) = self.auto_load_plugin(metadata).await {
                error!("Failed to auto-load plugin {}: {}", plugin_id, e);
            }
        }

        Ok(())
    }

    /// Auto-load a discovered plugin
    async fn auto_load_plugin(&self, metadata: PluginMetadata) -> Result<()> {
        let plugin_id = metadata.id.clone();
        let plugin_path = self.plugin_dir.join(&plugin_id);

        // Create default configuration
        let config = PluginConfig {
            enabled: true,
            config: Value::Null,
            priority: 0,
            timeout_seconds: 30,
            max_memory_mb: 100,
            permissions: PluginPermissions::default(),
        };

        // Register in registry
        self.registry.write().await.register_plugin(metadata, config.clone()).await?;

        // Load the plugin
        self.load_plugin_from_path(plugin_path, Some(config)).await?;

        Ok(())
    }

    /// Load a plugin from a path with optional configuration
    pub async fn load_plugin_from_path(&self, plugin_path: PathBuf, config: Option<PluginConfig>) -> Result<String> {
        let mut plugin = self.loader.load_from_path(plugin_path).await?;
        let plugin_id = plugin.metadata().id.clone();

        // Security check
        if self.security_manager.is_blocked(&plugin_id) {
            return Err(AgentError::plugin(format!("Plugin is blocked: {}", plugin_id)));
        }

        // Get or create configuration
        let plugin_config = config.unwrap_or_else(|| PluginConfig {
            enabled: true,
            config: Value::Null,
            priority: 0,
            timeout_seconds: 30,
            max_memory_mb: 100,
            permissions: PluginPermissions::default(),
        });

        // Validate permissions
        self.security_manager.validate_permissions(&plugin_id, &plugin_config.permissions)?;

        // Create plugin context
        let context = PluginContext {
            plugin_id: plugin_id.clone(),
            execution_id: uuid::Uuid::new_v4().to_string(),
            config: plugin_config.clone(),
            services: HashMap::new(),
            data_dir: self.data_dir.join(&plugin_id),
            temp_dir: std::env::temp_dir().join("agent_plugins").join(&plugin_id),
        };

        // Initialize plugin
        plugin.initialize(&context).await?;

        // Send loading event
        plugin.on_event(PluginEvent::Loaded, &context).await?;

        // Initialize statistics
        let stats = PluginStats {
            plugin_id: plugin_id.clone(),
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            avg_execution_time_ms: 0.0,
            total_execution_time_ms: 0,
            avg_memory_usage_bytes: 0.0,
            peak_memory_usage_bytes: 0,
            last_execution: None,
            uptime_seconds: 0,
        };

        // Store plugin, config, and stats
        self.plugins.write().await.insert(plugin_id.clone(), plugin);
        self.configs.write().await.insert(plugin_id.clone(), plugin_config);
        self.stats.write().await.insert(plugin_id.clone(), stats);

        // Initialize health status
        let health = PluginHealthStatus {
            healthy: true,
            message: "Plugin loaded successfully".to_string(),
            details: Value::Null,
        };
        self.health_status.write().await.insert(plugin_id.clone(), health);

        info!("Successfully loaded plugin: {}", plugin_id);
        Ok(plugin_id)
    }

    /// Execute a plugin with security and resource monitoring
    pub async fn execute_plugin_secure(&self, plugin_id: &str, input: Value) -> Result<PluginResult> {
        let start_time = std::time::Instant::now();

        // Security checks
        if self.security_manager.is_blocked(plugin_id) {
            return Err(AgentError::plugin(format!("Plugin is blocked: {}", plugin_id)));
        }

        // Resource checks
        if !self.resource_monitor.can_execute(plugin_id).await {
            return Err(AgentError::plugin(format!("Plugin resource limits exceeded: {}", plugin_id)));
        }

        // Get plugin and config
        let plugins = self.plugins.read().await;
        let plugin = plugins.get(plugin_id)
            .ok_or_else(|| AgentError::plugin(format!("Plugin not found: {}", plugin_id)))?;

        let configs = self.configs.read().await;
        let config = configs.get(plugin_id)
            .ok_or_else(|| AgentError::plugin(format!("Plugin config not found: {}", plugin_id)))?;

        // Check if plugin is enabled
        if !config.enabled {
            return Err(AgentError::plugin(format!("Plugin is disabled: {}", plugin_id)));
        }

        // Create execution context
        let context = PluginContext {
            plugin_id: plugin_id.to_string(),
            execution_id: uuid::Uuid::new_v4().to_string(),
            config: config.clone(),
            services: HashMap::new(),
            data_dir: self.data_dir.join(plugin_id),
            temp_dir: std::env::temp_dir().join("agent_plugins").join(plugin_id),
        };

        // Send executing event
        plugin.on_event(PluginEvent::Executing, &context).await?;

        // Execute with timeout and resource monitoring
        let result = self.execute_with_monitoring(plugin.as_ref(), input, &context).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update statistics and health
        match &result {
            Ok(plugin_result) => {
                self.update_stats(plugin_id, true, execution_time, plugin_result.memory_usage_bytes).await;
                self.update_health_status(plugin_id, true, "Execution successful".to_string()).await;
                plugin.on_event(PluginEvent::ExecutionCompleted, &context).await?;
            }
            Err(e) => {
                self.update_stats(plugin_id, false, execution_time, 0).await;
                self.update_health_status(plugin_id, false, e.to_string()).await;
                plugin.on_event(PluginEvent::Error(e.to_string()), &context).await?;
            }
        }

        result
    }

    /// Execute plugin with resource monitoring
    async fn execute_with_monitoring(&self, plugin: &dyn Plugin, input: Value, context: &PluginContext) -> Result<PluginResult> {
        let plugin_id = &context.plugin_id;
        
        // Start resource monitoring
        self.resource_monitor.start_monitoring(plugin_id).await;

        // Execute with timeout
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(context.config.timeout_seconds),
            plugin.execute(input, context)
        ).await;

        // Stop resource monitoring
        self.resource_monitor.stop_monitoring(plugin_id).await;

        match result {
            Ok(Ok(plugin_result)) => Ok(plugin_result),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(AgentError::plugin(format!("Plugin execution timed out: {}", plugin_id))),
        }
    }

    /// Update plugin statistics
    async fn update_stats(&self, plugin_id: &str, success: bool, execution_time_ms: u64, memory_usage: u64) {
        let mut stats = self.stats.write().await;
        if let Some(plugin_stats) = stats.get_mut(plugin_id) {
            plugin_stats.total_executions += 1;
            if success {
                plugin_stats.successful_executions += 1;
            } else {
                plugin_stats.failed_executions += 1;
            }
            
            plugin_stats.total_execution_time_ms += execution_time_ms;
            plugin_stats.avg_execution_time_ms = 
                plugin_stats.total_execution_time_ms as f64 / plugin_stats.total_executions as f64;
            
            if memory_usage > plugin_stats.peak_memory_usage_bytes {
                plugin_stats.peak_memory_usage_bytes = memory_usage;
            }
            
            plugin_stats.last_execution = Some(chrono::Utc::now());
        }
    }

    /// Update plugin health status
    async fn update_health_status(&self, plugin_id: &str, healthy: bool, message: String) {
        let mut health_status = self.health_status.write().await;
        if let Some(status) = health_status.get_mut(plugin_id) {
            status.healthy = healthy;
            status.message = message;
        }
    }

    /// Get plugin health status
    pub async fn get_health_status(&self, plugin_id: &str) -> Option<PluginHealthStatus> {
        self.health_status.read().await.get(plugin_id).cloned()
    }

    /// Perform health check on all plugins
    pub async fn health_check_all(&self) -> HashMap<String, PluginHealthStatus> {
        let mut results = HashMap::new();
        let plugins = self.plugins.read().await;
        
        for (plugin_id, plugin) in plugins.iter() {
            let configs = self.configs.read().await;
            if let Some(config) = configs.get(plugin_id) {
                let context = PluginContext {
                    plugin_id: plugin_id.clone(),
                    execution_id: uuid::Uuid::new_v4().to_string(),
                    config: config.clone(),
                    services: HashMap::new(),
                    data_dir: self.data_dir.join(plugin_id),
                    temp_dir: std::env::temp_dir().join("agent_plugins").join(plugin_id),
                };

                match plugin.health_check(&context).await {
                    Ok(status) => {
                        results.insert(plugin_id.clone(), status);
                    }
                    Err(e) => {
                        results.insert(plugin_id.clone(), PluginHealthStatus {
                            healthy: false,
                            message: format!("Health check failed: {}", e),
                            details: Value::Null,
                        });
                    }
                }
            }
        }

        results
    }

    /// Get comprehensive plugin statistics
    pub async fn get_all_stats(&self) -> HashMap<String, PluginStats> {
        self.stats.read().await.clone()
    }

    /// List all loaded plugins with their status
    pub async fn list_plugins_detailed(&self) -> Vec<PluginInfo> {
        let mut plugin_info = Vec::new();
        let plugins = self.plugins.read().await;
        let configs = self.configs.read().await;
        let stats = self.stats.read().await;
        let health = self.health_status.read().await;

        for (plugin_id, plugin) in plugins.iter() {
            let info = PluginInfo {
                metadata: plugin.metadata().clone(),
                config: configs.get(plugin_id).cloned(),
                stats: stats.get(plugin_id).cloned(),
                health: health.get(plugin_id).cloned(),
            };
            plugin_info.push(info);
        }

        plugin_info
    }

    /// Unload a plugin safely
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        // Get plugin for cleanup
        let plugins = self.plugins.read().await;
        if let Some(plugin) = plugins.get(plugin_id) {
            let configs = self.configs.read().await;
            if let Some(config) = configs.get(plugin_id) {
                let context = PluginContext {
                    plugin_id: plugin_id.to_string(),
                    execution_id: uuid::Uuid::new_v4().to_string(),
                    config: config.clone(),
                    services: HashMap::new(),
                    data_dir: self.data_dir.join(plugin_id),
                    temp_dir: std::env::temp_dir().join("agent_plugins").join(plugin_id),
                };

                // Send unloading event
                plugin.on_event(PluginEvent::Unloading, &context).await?;
                
                // Cleanup plugin resources
                // Note: We can't call cleanup on the plugin here because we have a read lock
                // In a real implementation, we'd need to handle this differently
            }
        }
        drop(plugins);

        // Remove from all collections
        self.plugins.write().await.remove(plugin_id);
        self.configs.write().await.remove(plugin_id);
        self.stats.write().await.remove(plugin_id);
        self.health_status.write().await.remove(plugin_id);

        info!("Unloaded plugin: {}", plugin_id);
        Ok(())
    }
}

/// Detailed plugin information
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub metadata: PluginMetadata,
    pub config: Option<PluginConfig>,
    pub stats: Option<PluginStats>,
    pub health: Option<PluginHealthStatus>,
}

impl PluginSecurityManager {
    pub fn new() -> Self {
        Self {
            global_permissions: PluginPermissions::default(),
            plugin_permissions: HashMap::new(),
            blocked_plugins: Vec::new(),
        }
    }

    pub fn is_blocked(&self, plugin_id: &str) -> bool {
        self.blocked_plugins.contains(&plugin_id.to_string())
    }

    pub fn validate_permissions(&self, plugin_id: &str, permissions: &PluginPermissions) -> Result<()> {
        // Check against global permissions
        if permissions.file_write && !self.global_permissions.file_write {
            return Err(AgentError::plugin(format!("Plugin {} not allowed to write files", plugin_id)));
        }

        if permissions.execute_commands && !self.global_permissions.execute_commands {
            return Err(AgentError::plugin(format!("Plugin {} not allowed to execute commands", plugin_id)));
        }

        if permissions.network_access && !self.global_permissions.network_access {
            return Err(AgentError::plugin(format!("Plugin {} not allowed network access", plugin_id)));
        }

        Ok(())
    }
}

impl PluginResourceMonitor {
    pub fn new() -> Self {
        Self {
            memory_limits: HashMap::new(),
            cpu_limits: HashMap::new(),
            time_limits: HashMap::new(),
            current_usage: HashMap::new(),
        }
    }

    pub async fn can_execute(&self, plugin_id: &str) -> bool {
        // Check if plugin is within resource limits
        if let Some(usage) = self.current_usage.get(plugin_id) {
            if let Some(memory_limit) = self.memory_limits.get(plugin_id) {
                if usage.memory_bytes > *memory_limit {
                    return false;
                }
            }
        }
        true
    }

    pub async fn start_monitoring(&self, plugin_id: &str) {
        // Start monitoring plugin resource usage
        // In a real implementation, this would start actual monitoring
        debug!("Started monitoring plugin: {}", plugin_id);
    }

    pub async fn stop_monitoring(&self, plugin_id: &str) {
        // Stop monitoring plugin resource usage
        debug!("Stopped monitoring plugin: {}", plugin_id);
    }
}
