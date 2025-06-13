use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::utils::error::{AgentError, Result};

pub mod registry;
pub mod loader;
pub mod manager;

#[cfg(test)]
mod tests;

/// Plugin metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Unique plugin identifier
    pub id: String,
    /// Human-readable plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin license
    pub license: String,
    /// Minimum agent version required
    pub min_agent_version: String,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
    /// Plugin capabilities/features
    pub capabilities: Vec<String>,
    /// Plugin configuration schema
    pub config_schema: Option<Value>,
    /// Plugin entry point
    pub entry_point: String,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Whether the plugin is enabled
    pub enabled: bool,
    /// Plugin-specific configuration
    pub config: Value,
    /// Plugin priority (higher = loaded first)
    pub priority: i32,
    /// Plugin timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Plugin permissions
    pub permissions: PluginPermissions,
}

/// Plugin permissions system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginPermissions {
    /// Can read files
    pub file_read: bool,
    /// Can write files
    pub file_write: bool,
    /// Can execute commands
    pub execute_commands: bool,
    /// Can access network
    pub network_access: bool,
    /// Can access memory system
    pub memory_access: bool,
    /// Can register tools
    pub register_tools: bool,
    /// Can access environment variables
    pub env_access: bool,
    /// Allowed file paths (if file access is enabled)
    pub allowed_paths: Vec<PathBuf>,
    /// Allowed network hosts (if network access is enabled)
    pub allowed_hosts: Vec<String>,
}

impl Default for PluginPermissions {
    fn default() -> Self {
        Self {
            file_read: false,
            file_write: false,
            execute_commands: false,
            network_access: false,
            memory_access: false,
            register_tools: false,
            env_access: false,
            allowed_paths: Vec::new(),
            allowed_hosts: Vec::new(),
        }
    }
}

/// Plugin execution context
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Plugin ID
    pub plugin_id: String,
    /// Execution ID for this invocation
    pub execution_id: String,
    /// Plugin configuration
    pub config: PluginConfig,
    /// Available services
    pub services: HashMap<String, Value>,
    /// Plugin data directory
    pub data_dir: PathBuf,
    /// Plugin temporary directory
    pub temp_dir: PathBuf,
}

/// Plugin execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResult {
    /// Whether execution was successful
    pub success: bool,
    /// Result data
    pub data: Value,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Plugin logs
    pub logs: Vec<PluginLogEntry>,
}

/// Plugin log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginLogEntry {
    /// Log level
    pub level: String,
    /// Log message
    pub message: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Source location
    pub source: Option<String>,
}

/// Plugin lifecycle events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// Plugin is being loaded
    Loading,
    /// Plugin has been loaded successfully
    Loaded,
    /// Plugin is being initialized
    Initializing,
    /// Plugin has been initialized successfully
    Initialized,
    /// Plugin is being executed
    Executing,
    /// Plugin execution completed
    ExecutionCompleted,
    /// Plugin is being unloaded
    Unloading,
    /// Plugin has been unloaded
    Unloaded,
    /// Plugin encountered an error
    Error(String),
}

/// Plugin trait that all plugins must implement
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin with configuration
    async fn initialize(&mut self, context: &PluginContext) -> Result<()>;

    /// Execute the plugin with given input
    async fn execute(&self, input: Value, context: &PluginContext) -> Result<PluginResult>;

    /// Handle plugin events
    async fn on_event(&self, _event: PluginEvent, _context: &PluginContext) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Cleanup plugin resources
    async fn cleanup(&mut self, _context: &PluginContext) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Validate plugin configuration
    fn validate_config(&self, _config: &Value) -> Result<()> {
        // Default implementation accepts any config
        Ok(())
    }

    /// Get plugin health status
    async fn health_check(&self, _context: &PluginContext) -> Result<PluginHealthStatus> {
        Ok(PluginHealthStatus {
            healthy: true,
            message: "Plugin is healthy".to_string(),
            details: Value::Null,
        })
    }
}

/// Plugin health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHealthStatus {
    /// Whether the plugin is healthy
    pub healthy: bool,
    /// Health status message
    pub message: String,
    /// Additional health details
    pub details: Value,
}

/// Plugin execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStats {
    /// Plugin ID
    pub plugin_id: String,
    /// Total executions
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Average memory usage in bytes
    pub avg_memory_usage_bytes: f64,
    /// Peak memory usage in bytes
    pub peak_memory_usage_bytes: u64,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
    /// Plugin uptime in seconds
    pub uptime_seconds: u64,
}

/// Plugin manager for loading, managing, and executing plugins
pub struct PluginManager {
    /// Loaded plugins
    plugins: Arc<RwLock<HashMap<String, Box<dyn Plugin>>>>,
    /// Plugin configurations
    configs: Arc<RwLock<HashMap<String, PluginConfig>>>,
    /// Plugin statistics
    stats: Arc<RwLock<HashMap<String, PluginStats>>>,
    /// Plugin registry
    registry: registry::PluginRegistry,
    /// Plugin loader
    loader: loader::PluginLoader,
    /// Base plugin directory
    plugin_dir: PathBuf,
    /// Plugin data directory
    data_dir: PathBuf,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(plugin_dir: PathBuf, data_dir: PathBuf) -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
            registry: registry::PluginRegistry::new(),
            loader: loader::PluginLoader::new(),
            plugin_dir,
            data_dir,
        }
    }

    /// Load a plugin from a directory
    pub async fn load_plugin(&self, plugin_path: PathBuf) -> Result<String> {
        let plugin = self.loader.load_from_path(plugin_path).await?;
        let plugin_id = plugin.metadata().id.clone();

        // Initialize plugin statistics
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

        // Store plugin and stats
        self.plugins.write().await.insert(plugin_id.clone(), plugin);
        self.stats.write().await.insert(plugin_id.clone(), stats);

        Ok(plugin_id)
    }

    /// Execute a plugin
    pub async fn execute_plugin(&self, plugin_id: &str, input: Value) -> Result<PluginResult> {
        let start_time = std::time::Instant::now();
        
        // Get plugin and config
        let plugins = self.plugins.read().await;
        let plugin = plugins.get(plugin_id)
            .ok_or_else(|| AgentError::plugin(format!("Plugin not found: {}", plugin_id)))?;

        let configs = self.configs.read().await;
        let config = configs.get(plugin_id)
            .cloned()
            .unwrap_or_else(|| PluginConfig {
                enabled: true,
                config: Value::Null,
                priority: 0,
                timeout_seconds: 30,
                max_memory_mb: 100,
                permissions: PluginPermissions::default(),
            });

        // Check if plugin is enabled
        if !config.enabled {
            return Err(AgentError::plugin(format!("Plugin is disabled: {}", plugin_id)));
        }

        // Create execution context
        let context = PluginContext {
            plugin_id: plugin_id.to_string(),
            execution_id: Uuid::new_v4().to_string(),
            config,
            services: HashMap::new(),
            data_dir: self.data_dir.join(plugin_id),
            temp_dir: std::env::temp_dir().join("agent_plugins").join(plugin_id),
        };

        // Execute plugin with timeout
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(context.config.timeout_seconds),
            plugin.execute(input, &context)
        ).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        // Handle result and update statistics
        match result {
            Ok(Ok(mut plugin_result)) => {
                plugin_result.execution_time_ms = execution_time;
                self.update_stats(plugin_id, true, execution_time, plugin_result.memory_usage_bytes).await;
                Ok(plugin_result)
            }
            Ok(Err(e)) => {
                self.update_stats(plugin_id, false, execution_time, 0).await;
                Err(e)
            }
            Err(_) => {
                self.update_stats(plugin_id, false, execution_time, 0).await;
                Err(AgentError::plugin(format!("Plugin execution timed out: {}", plugin_id)))
            }
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

    /// Get plugin statistics
    pub async fn get_stats(&self, plugin_id: &str) -> Option<PluginStats> {
        self.stats.read().await.get(plugin_id).cloned()
    }

    /// List all loaded plugins
    pub async fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().await.keys().cloned().collect()
    }

    /// Unload a plugin
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;
        let mut configs = self.configs.write().await;
        let mut stats = self.stats.write().await;

        plugins.remove(plugin_id);
        configs.remove(plugin_id);
        stats.remove(plugin_id);

        Ok(())
    }
}
