use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, info};

use crate::utils::error::{AgentError, Result};
use super::{PluginMetadata, PluginConfig};

/// Plugin registry for managing plugin metadata and discovery
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    /// Registered plugins metadata
    plugins: HashMap<String, PluginMetadata>,
    /// Plugin configurations
    configs: HashMap<String, PluginConfig>,
    /// Registry file path
    registry_path: Option<PathBuf>,
}

/// Plugin registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistryEntry {
    /// Plugin metadata
    pub metadata: PluginMetadata,
    /// Plugin configuration
    pub config: PluginConfig,
    /// Plugin installation path
    pub install_path: PathBuf,
    /// Installation timestamp
    pub installed_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Plugin status
    pub status: PluginStatus,
}

/// Plugin status in the registry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PluginStatus {
    /// Plugin is installed and ready
    Installed,
    /// Plugin is currently loading
    Loading,
    /// Plugin is loaded and active
    Active,
    /// Plugin is disabled
    Disabled,
    /// Plugin has errors
    Error(String),
    /// Plugin is being updated
    Updating,
    /// Plugin is being uninstalled
    Uninstalling,
}

/// Plugin discovery result
#[derive(Debug, Clone)]
pub struct PluginDiscoveryResult {
    /// Found plugins
    pub plugins: Vec<PluginMetadata>,
    /// Discovery errors
    pub errors: Vec<String>,
    /// Scanned directories
    pub scanned_dirs: Vec<PathBuf>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            configs: HashMap::new(),
            registry_path: None,
        }
    }

    /// Create a plugin registry with persistent storage
    pub fn with_storage(registry_path: PathBuf) -> Self {
        Self {
            plugins: HashMap::new(),
            configs: HashMap::new(),
            registry_path: Some(registry_path),
        }
    }

    /// Register a plugin in the registry
    pub async fn register_plugin(&mut self, metadata: PluginMetadata, config: PluginConfig) -> Result<()> {
        let plugin_id = metadata.id.clone();
        
        // Validate plugin metadata
        self.validate_plugin_metadata(&metadata)?;
        
        // Check for conflicts
        if self.plugins.contains_key(&plugin_id) {
            return Err(AgentError::plugin(format!("Plugin already registered: {}", plugin_id)));
        }

        // Register the plugin
        self.plugins.insert(plugin_id.clone(), metadata);
        self.configs.insert(plugin_id.clone(), config);

        info!("Registered plugin: {}", plugin_id);

        // Persist to storage if configured
        if self.registry_path.is_some() {
            self.save_registry().await?;
        }

        Ok(())
    }

    /// Unregister a plugin from the registry
    pub async fn unregister_plugin(&mut self, plugin_id: &str) -> Result<()> {
        if !self.plugins.contains_key(plugin_id) {
            return Err(AgentError::plugin(format!("Plugin not found: {}", plugin_id)));
        }

        self.plugins.remove(plugin_id);
        self.configs.remove(plugin_id);

        info!("Unregistered plugin: {}", plugin_id);

        // Persist to storage if configured
        if self.registry_path.is_some() {
            self.save_registry().await?;
        }

        Ok(())
    }

    /// Get plugin metadata by ID
    pub fn get_plugin(&self, plugin_id: &str) -> Option<&PluginMetadata> {
        self.plugins.get(plugin_id)
    }

    /// Get plugin configuration by ID
    pub fn get_config(&self, plugin_id: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_id)
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.plugins.values().collect()
    }

    /// Search plugins by capability
    pub fn search_by_capability(&self, capability: &str) -> Vec<&PluginMetadata> {
        self.plugins
            .values()
            .filter(|plugin| plugin.capabilities.contains(&capability.to_string()))
            .collect()
    }

    /// Search plugins by name or description
    pub fn search_plugins(&self, query: &str) -> Vec<&PluginMetadata> {
        let query_lower = query.to_lowercase();
        self.plugins
            .values()
            .filter(|plugin| {
                plugin.name.to_lowercase().contains(&query_lower) ||
                plugin.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Discover plugins in a directory
    pub async fn discover_plugins(&self, search_dir: PathBuf) -> Result<PluginDiscoveryResult> {
        let mut result = PluginDiscoveryResult {
            plugins: Vec::new(),
            errors: Vec::new(),
            scanned_dirs: vec![search_dir.clone()],
        };

        if !search_dir.exists() {
            result.errors.push(format!("Directory does not exist: {}", search_dir.display()));
            return Ok(result);
        }

        debug!("Discovering plugins in: {}", search_dir.display());

        let mut entries = fs::read_dir(&search_dir).await
            .map_err(|e| AgentError::plugin(format!("Failed to read directory {}: {}", search_dir.display(), e)))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| AgentError::plugin(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.is_dir() {
                match self.discover_plugin_in_dir(path.clone()).await {
                    Ok(Some(metadata)) => {
                        result.plugins.push(metadata);
                    }
                    Ok(None) => {
                        // No plugin found in this directory
                    }
                    Err(e) => {
                        result.errors.push(format!("Error in {}: {}", path.display(), e));
                    }
                }
            }
        }

        info!("Discovered {} plugins in {}", result.plugins.len(), search_dir.display());
        Ok(result)
    }

    /// Discover a plugin in a specific directory
    async fn discover_plugin_in_dir(&self, plugin_dir: PathBuf) -> Result<Option<PluginMetadata>> {
        let manifest_path = plugin_dir.join("plugin.toml");
        
        if !manifest_path.exists() {
            // Try alternative manifest names
            let alt_manifest = plugin_dir.join("manifest.toml");
            if !alt_manifest.exists() {
                return Ok(None);
            }
        }

        let manifest_content = fs::read_to_string(&manifest_path).await
            .map_err(|e| AgentError::plugin(format!("Failed to read manifest {}: {}", manifest_path.display(), e)))?;

        let metadata: PluginMetadata = toml::from_str(&manifest_content)
            .map_err(|e| AgentError::plugin(format!("Failed to parse manifest {}: {}", manifest_path.display(), e)))?;

        // Validate the discovered plugin
        self.validate_plugin_metadata(&metadata)?;

        Ok(Some(metadata))
    }

    /// Validate plugin metadata
    fn validate_plugin_metadata(&self, metadata: &PluginMetadata) -> Result<()> {
        if metadata.id.is_empty() {
            return Err(AgentError::plugin("Plugin ID cannot be empty".to_string()));
        }

        if metadata.name.is_empty() {
            return Err(AgentError::plugin("Plugin name cannot be empty".to_string()));
        }

        if metadata.version.is_empty() {
            return Err(AgentError::plugin("Plugin version cannot be empty".to_string()));
        }

        if metadata.entry_point.is_empty() {
            return Err(AgentError::plugin("Plugin entry point cannot be empty".to_string()));
        }

        // Validate version format (basic semver check)
        if !self.is_valid_version(&metadata.version) {
            return Err(AgentError::plugin(format!("Invalid version format: {}", metadata.version)));
        }

        Ok(())
    }

    /// Check if a version string is valid (basic semver validation)
    fn is_valid_version(&self, version: &str) -> bool {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return false;
        }

        parts.iter().all(|part| part.parse::<u32>().is_ok())
    }

    /// Load registry from storage
    pub async fn load_registry(&mut self) -> Result<()> {
        if let Some(ref registry_path) = self.registry_path {
            if registry_path.exists() {
                let content = fs::read_to_string(registry_path).await
                    .map_err(|e| AgentError::plugin(format!("Failed to read registry file: {}", e)))?;

                let registry_data: HashMap<String, PluginRegistryEntry> = toml::from_str(&content)
                    .map_err(|e| AgentError::plugin(format!("Failed to parse registry file: {}", e)))?;

                for (plugin_id, entry) in registry_data {
                    self.plugins.insert(plugin_id.clone(), entry.metadata);
                    self.configs.insert(plugin_id, entry.config);
                }

                info!("Loaded {} plugins from registry", self.plugins.len());
            }
        }
        Ok(())
    }

    /// Save registry to storage
    pub async fn save_registry(&self) -> Result<()> {
        if let Some(ref registry_path) = self.registry_path {
            let mut registry_data = HashMap::new();

            for (plugin_id, metadata) in &self.plugins {
                if let Some(config) = self.configs.get(plugin_id) {
                    let entry = PluginRegistryEntry {
                        metadata: metadata.clone(),
                        config: config.clone(),
                        install_path: PathBuf::from(""), // TODO: Track actual install paths
                        installed_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                        status: PluginStatus::Installed,
                    };
                    registry_data.insert(plugin_id.clone(), entry);
                }
            }

            let content = toml::to_string_pretty(&registry_data)
                .map_err(|e| AgentError::plugin(format!("Failed to serialize registry: {}", e)))?;

            // Ensure parent directory exists
            if let Some(parent) = registry_path.parent() {
                fs::create_dir_all(parent).await
                    .map_err(|e| AgentError::plugin(format!("Failed to create registry directory: {}", e)))?;
            }

            fs::write(registry_path, content).await
                .map_err(|e| AgentError::plugin(format!("Failed to write registry file: {}", e)))?;

            debug!("Saved registry with {} plugins", registry_data.len());
        }
        Ok(())
    }

    /// Check for plugin dependency conflicts
    pub fn check_dependencies(&self, plugin_id: &str) -> Result<Vec<String>> {
        let mut missing_deps = Vec::new();

        if let Some(metadata) = self.plugins.get(plugin_id) {
            for dep in &metadata.dependencies {
                if !self.plugins.contains_key(dep) {
                    missing_deps.push(dep.clone());
                }
            }
        }

        Ok(missing_deps)
    }

    /// Get plugin dependency graph
    pub fn get_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();

        for (plugin_id, metadata) in &self.plugins {
            graph.insert(plugin_id.clone(), metadata.dependencies.clone());
        }

        graph
    }

    /// Validate plugin compatibility
    pub fn validate_compatibility(&self, plugin_id: &str, agent_version: &str) -> Result<bool> {
        if let Some(metadata) = self.plugins.get(plugin_id) {
            // Simple version comparison (in real implementation, use proper semver)
            let min_version = &metadata.min_agent_version;
            Ok(agent_version >= min_version.as_str())
        } else {
            Err(AgentError::plugin(format!("Plugin not found: {}", plugin_id)))
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
