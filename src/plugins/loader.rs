use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokio::fs;
use tracing::{debug, info};

use crate::utils::error::{AgentError, Result};
use super::{Plugin, PluginMetadata};

/// Plugin loader for loading plugins from various sources
pub struct PluginLoader {
    /// Loaded plugin factories
    factories: HashMap<String, Box<dyn PluginFactory>>,
    /// Plugin loading cache
    cache: HashMap<PathBuf, Box<dyn Plugin>>,
}

/// Plugin factory trait for creating plugin instances
pub trait PluginFactory: Send + Sync {
    /// Create a plugin instance from a path
    fn create_plugin(&self, plugin_path: PathBuf) -> Result<Box<dyn Plugin>>;
    
    /// Check if this factory can handle the given plugin type
    fn can_handle(&self, plugin_type: &str) -> bool;
    
    /// Get supported plugin types
    fn supported_types(&self) -> Vec<String>;
}

/// Native Rust plugin factory
#[derive(Debug)]
pub struct RustPluginFactory;

impl PluginFactory for RustPluginFactory {
    fn create_plugin(&self, plugin_path: PathBuf) -> Result<Box<dyn Plugin>> {
        // For now, create a basic plugin wrapper
        // In a real implementation, this would use dynamic loading
        let metadata = self.load_metadata(&plugin_path)?;
        Ok(Box::new(BasicPlugin::new(metadata)))
    }

    fn can_handle(&self, plugin_type: &str) -> bool {
        plugin_type == "rust" || plugin_type == "native"
    }

    fn supported_types(&self) -> Vec<String> {
        vec!["rust".to_string(), "native".to_string()]
    }
}

impl RustPluginFactory {
    fn load_metadata(&self, plugin_path: &PathBuf) -> Result<PluginMetadata> {
        let manifest_path = plugin_path.join("plugin.toml");
        if !manifest_path.exists() {
            return Err(AgentError::plugin(format!("Plugin manifest not found: {}", manifest_path.display())));
        }

        let content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| AgentError::plugin(format!("Failed to read manifest: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| AgentError::plugin(format!("Failed to parse manifest: {}", e)))
    }
}

/// Script plugin factory for script-based plugins
#[derive(Debug)]
pub struct ScriptPluginFactory;

impl PluginFactory for ScriptPluginFactory {
    fn create_plugin(&self, plugin_path: PathBuf) -> Result<Box<dyn Plugin>> {
        let metadata = self.load_metadata(&plugin_path)?;
        Ok(Box::new(ScriptPlugin::new(metadata, plugin_path)))
    }

    fn can_handle(&self, plugin_type: &str) -> bool {
        matches!(plugin_type, "python" | "javascript" | "shell" | "script")
    }

    fn supported_types(&self) -> Vec<String> {
        vec!["python".to_string(), "javascript".to_string(), "shell".to_string(), "script".to_string()]
    }
}

impl ScriptPluginFactory {
    fn load_metadata(&self, plugin_path: &PathBuf) -> Result<PluginMetadata> {
        let manifest_path = plugin_path.join("plugin.toml");
        if !manifest_path.exists() {
            return Err(AgentError::plugin(format!("Plugin manifest not found: {}", manifest_path.display())));
        }

        let content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| AgentError::plugin(format!("Failed to read manifest: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| AgentError::plugin(format!("Failed to parse manifest: {}", e)))
    }
}

/// Basic plugin implementation for testing and simple plugins
#[derive(Debug)]
pub struct BasicPlugin {
    metadata: PluginMetadata,
}

impl BasicPlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self { metadata }
    }
}

#[async_trait::async_trait]
impl Plugin for BasicPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, context: &super::PluginContext) -> Result<()> {
        info!("Initializing basic plugin: {}", self.metadata.id);
        
        // Create plugin data directory
        fs::create_dir_all(&context.data_dir).await
            .map_err(|e| AgentError::plugin(format!("Failed to create data directory: {}", e)))?;
        
        Ok(())
    }

    async fn execute(&self, input: serde_json::Value, _context: &super::PluginContext) -> Result<super::PluginResult> {
        let start_time = std::time::Instant::now();
        
        info!("Executing basic plugin: {}", self.metadata.id);
        
        // Basic plugin just echoes the input
        let result = super::PluginResult {
            success: true,
            data: serde_json::json!({
                "plugin_id": self.metadata.id,
                "input": input,
                "message": "Basic plugin executed successfully"
            }),
            error: None,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            memory_usage_bytes: 1024, // Minimal memory usage
            logs: vec![
                super::PluginLogEntry {
                    level: "info".to_string(),
                    message: format!("Basic plugin {} executed", self.metadata.id),
                    timestamp: chrono::Utc::now(),
                    source: Some(self.metadata.id.clone()),
                }
            ],
        };

        Ok(result)
    }
}

/// Script-based plugin implementation
#[derive(Debug)]
pub struct ScriptPlugin {
    metadata: PluginMetadata,
    plugin_path: PathBuf,
}

impl ScriptPlugin {
    pub fn new(metadata: PluginMetadata, plugin_path: PathBuf) -> Self {
        Self { metadata, plugin_path }
    }

    fn get_interpreter(&self) -> Result<String> {
        let entry_point = &self.metadata.entry_point;
        
        if entry_point.ends_with(".py") {
            Ok("python3".to_string())
        } else if entry_point.ends_with(".js") {
            Ok("node".to_string())
        } else if entry_point.ends_with(".sh") {
            Ok("bash".to_string())
        } else {
            Err(AgentError::plugin(format!("Unsupported script type: {}", entry_point)))
        }
    }
}

#[async_trait::async_trait]
impl Plugin for ScriptPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, context: &super::PluginContext) -> Result<()> {
        info!("Initializing script plugin: {}", self.metadata.id);
        
        // Create plugin data directory
        fs::create_dir_all(&context.data_dir).await
            .map_err(|e| AgentError::plugin(format!("Failed to create data directory: {}", e)))?;
        
        // Verify script exists
        let script_path = self.plugin_path.join(&self.metadata.entry_point);
        if !script_path.exists() {
            return Err(AgentError::plugin(format!("Script not found: {}", script_path.display())));
        }

        Ok(())
    }

    async fn execute(&self, input: serde_json::Value, context: &super::PluginContext) -> Result<super::PluginResult> {
        let start_time = std::time::Instant::now();
        
        info!("Executing script plugin: {}", self.metadata.id);
        
        let interpreter = self.get_interpreter()?;
        let script_path = self.plugin_path.join(&self.metadata.entry_point);
        
        // Prepare input as JSON string
        let input_json = serde_json::to_string(&input)
            .map_err(|e| AgentError::plugin(format!("Failed to serialize input: {}", e)))?;

        // Execute script
        let output = Command::new(&interpreter)
            .arg(&script_path)
            .arg(&input_json)
            .env("PLUGIN_ID", &self.metadata.id)
            .env("PLUGIN_DATA_DIR", &context.data_dir)
            .env("PLUGIN_TEMP_DIR", &context.temp_dir)
            .output()
            .map_err(|e| AgentError::plugin(format!("Failed to execute script: {}", e)))?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let result_data: serde_json::Value = serde_json::from_str(&stdout)
                .unwrap_or_else(|_| serde_json::json!({ "output": stdout }));

            Ok(super::PluginResult {
                success: true,
                data: result_data,
                error: None,
                execution_time_ms: execution_time,
                memory_usage_bytes: output.stdout.len() as u64 + output.stderr.len() as u64,
                logs: vec![
                    super::PluginLogEntry {
                        level: "info".to_string(),
                        message: format!("Script plugin {} executed successfully", self.metadata.id),
                        timestamp: chrono::Utc::now(),
                        source: Some(self.metadata.id.clone()),
                    }
                ],
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Ok(super::PluginResult {
                success: false,
                data: serde_json::Value::Null,
                error: Some(stderr.to_string()),
                execution_time_ms: execution_time,
                memory_usage_bytes: output.stderr.len() as u64,
                logs: vec![
                    super::PluginLogEntry {
                        level: "error".to_string(),
                        message: format!("Script plugin {} failed: {}", self.metadata.id, stderr),
                        timestamp: chrono::Utc::now(),
                        source: Some(self.metadata.id.clone()),
                    }
                ],
            })
        }
    }
}

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new() -> Self {
        let mut loader = Self {
            factories: HashMap::new(),
            cache: HashMap::new(),
        };

        // Register default factories
        loader.register_factory("rust", Box::new(RustPluginFactory));
        loader.register_factory("script", Box::new(ScriptPluginFactory));

        loader
    }

    /// Register a plugin factory
    pub fn register_factory(&mut self, plugin_type: &str, factory: Box<dyn PluginFactory>) {
        self.factories.insert(plugin_type.to_string(), factory);
    }

    /// Load a plugin from a path
    pub async fn load_from_path(&self, plugin_path: PathBuf) -> Result<Box<dyn Plugin>> {
        // Check cache first
        if let Some(_cached_plugin) = self.cache.get(&plugin_path) {
            debug!("Loading plugin from cache: {}", plugin_path.display());
            // Note: In a real implementation, we'd need to clone the plugin properly
            // For now, we'll reload it
        }

        info!("Loading plugin from: {}", plugin_path.display());

        // Determine plugin type
        let plugin_type = self.detect_plugin_type(&plugin_path).await?;
        
        // Find appropriate factory
        let factory = self.factories.values()
            .find(|f| f.can_handle(&plugin_type))
            .ok_or_else(|| AgentError::plugin(format!("No factory found for plugin type: {}", plugin_type)))?;

        // Create plugin instance
        let plugin = factory.create_plugin(plugin_path.clone())?;

        info!("Successfully loaded plugin: {}", plugin.metadata().id);
        Ok(plugin)
    }

    /// Detect plugin type from directory structure
    async fn detect_plugin_type(&self, plugin_path: &PathBuf) -> Result<String> {
        let manifest_path = plugin_path.join("plugin.toml");
        
        if manifest_path.exists() {
            let content = fs::read_to_string(&manifest_path).await
                .map_err(|e| AgentError::plugin(format!("Failed to read manifest: {}", e)))?;
            
            // Try to extract plugin type from manifest
            if let Ok(manifest) = toml::from_str::<serde_json::Value>(&content) {
                if let Some(plugin_type) = manifest.get("type").and_then(|v| v.as_str()) {
                    return Ok(plugin_type.to_string());
                }
            }
        }

        // Fallback: detect by file extensions
        if plugin_path.join("main.py").exists() || plugin_path.join("plugin.py").exists() {
            Ok("python".to_string())
        } else if plugin_path.join("main.js").exists() || plugin_path.join("plugin.js").exists() {
            Ok("javascript".to_string())
        } else if plugin_path.join("main.sh").exists() || plugin_path.join("plugin.sh").exists() {
            Ok("shell".to_string())
        } else if plugin_path.join("Cargo.toml").exists() {
            Ok("rust".to_string())
        } else {
            Err(AgentError::plugin(format!("Cannot detect plugin type for: {}", plugin_path.display())))
        }
    }

    /// Get supported plugin types
    pub fn supported_types(&self) -> Vec<String> {
        self.factories.values()
            .flat_map(|f| f.supported_types())
            .collect()
    }

    /// Clear plugin cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}
