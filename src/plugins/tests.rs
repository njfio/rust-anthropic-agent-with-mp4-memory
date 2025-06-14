use super::loader::PluginLoader;
use super::registry::PluginRegistry;
use super::*;
use serde_json::json;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

/// Create a test plugin metadata
fn create_test_metadata() -> PluginMetadata {
    PluginMetadata {
        id: "test_plugin".to_string(),
        name: "Test Plugin".to_string(),
        version: "1.0.0".to_string(),
        description: "A test plugin for unit testing".to_string(),
        author: "Test Author".to_string(),
        license: "MIT".to_string(),
        min_agent_version: "0.1.0".to_string(),
        dependencies: vec!["dep1".to_string(), "dep2".to_string()],
        capabilities: vec!["test".to_string(), "example".to_string()],
        config_schema: Some(json!({"type": "object"})),
        entry_point: "main.py".to_string(),
    }
}

/// Create a test plugin configuration
fn create_test_config() -> PluginConfig {
    PluginConfig {
        enabled: true,
        config: json!({"test_setting": "value"}),
        priority: 10,
        timeout_seconds: 30,
        max_memory_mb: 100,
        permissions: PluginPermissions {
            file_read: true,
            file_write: false,
            execute_commands: false,
            network_access: true,
            memory_access: true,
            register_tools: false,
            env_access: false,
            allowed_paths: vec![PathBuf::from("/tmp")],
            allowed_hosts: vec!["example.com".to_string()],
        },
    }
}

/// Create a test plugin directory structure
async fn create_test_plugin_dir(temp_dir: &TempDir, plugin_id: &str) -> Result<PathBuf> {
    let plugin_dir = temp_dir.path().join(plugin_id);
    fs::create_dir_all(&plugin_dir).await?;

    // Create plugin manifest
    let manifest = r#"
id = "test_plugin"
name = "Test Plugin"
version = "1.0.0"
description = "A test plugin"
author = "Test Author"
license = "MIT"
min_agent_version = "0.1.0"
dependencies = ["dep1", "dep2"]
capabilities = ["test", "example"]
entry_point = "main.py"
type = "python"
"#;
    fs::write(plugin_dir.join("plugin.toml"), manifest).await?;

    // Create a simple Python script
    let script = r#"#!/usr/bin/env python3
import sys
import json

def main():
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
        result = {
            "success": True,
            "message": "Test plugin executed successfully",
            "input": input_data
        }
        print(json.dumps(result))
    else:
        print(json.dumps({"error": "No input provided"}))

if __name__ == "__main__":
    main()
"#;
    fs::write(plugin_dir.join("main.py"), script).await?;

    Ok(plugin_dir)
}

#[tokio::test]
async fn test_plugin_metadata_creation() {
    let metadata = create_test_metadata();

    assert_eq!(metadata.id, "test_plugin");
    assert_eq!(metadata.name, "Test Plugin");
    assert_eq!(metadata.version, "1.0.0");
    assert_eq!(metadata.dependencies.len(), 2);
    assert_eq!(metadata.capabilities.len(), 2);
    assert!(metadata.config_schema.is_some());
}

#[tokio::test]
async fn test_plugin_config_creation() {
    let config = create_test_config();

    assert!(config.enabled);
    assert_eq!(config.priority, 10);
    assert_eq!(config.timeout_seconds, 30);
    assert_eq!(config.max_memory_mb, 100);
    assert!(config.permissions.file_read);
    assert!(!config.permissions.file_write);
    assert!(config.permissions.network_access);
}

#[tokio::test]
async fn test_plugin_permissions_default() {
    let permissions = PluginPermissions::default();

    assert!(!permissions.file_read);
    assert!(!permissions.file_write);
    assert!(!permissions.execute_commands);
    assert!(!permissions.network_access);
    assert!(!permissions.memory_access);
    assert!(!permissions.register_tools);
    assert!(!permissions.env_access);
    assert!(permissions.allowed_paths.is_empty());
    assert!(permissions.allowed_hosts.is_empty());
}

#[tokio::test]
async fn test_plugin_context_creation() {
    let config = create_test_config();
    let context = PluginContext {
        plugin_id: "test_plugin".to_string(),
        execution_id: "exec_123".to_string(),
        config: config.clone(),
        services: HashMap::new(),
        data_dir: PathBuf::from("/tmp/data"),
        temp_dir: PathBuf::from("/tmp/temp"),
    };

    assert_eq!(context.plugin_id, "test_plugin");
    assert_eq!(context.execution_id, "exec_123");
    assert_eq!(context.config.priority, config.priority);
}

#[tokio::test]
async fn test_plugin_result_creation() {
    let result = PluginResult {
        success: true,
        data: json!({"message": "test"}),
        error: None,
        execution_time_ms: 100,
        memory_usage_bytes: 1024,
        logs: vec![PluginLogEntry {
            level: "info".to_string(),
            message: "Test log".to_string(),
            timestamp: chrono::Utc::now(),
            source: Some("test_plugin".to_string()),
        }],
    };

    assert!(result.success);
    assert!(result.error.is_none());
    assert_eq!(result.execution_time_ms, 100);
    assert_eq!(result.memory_usage_bytes, 1024);
    assert_eq!(result.logs.len(), 1);
}

#[tokio::test]
async fn test_plugin_stats_initialization() {
    let stats = PluginStats {
        plugin_id: "test_plugin".to_string(),
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

    assert_eq!(stats.plugin_id, "test_plugin");
    assert_eq!(stats.total_executions, 0);
    assert_eq!(stats.successful_executions, 0);
    assert_eq!(stats.failed_executions, 0);
    assert!(stats.last_execution.is_none());
}

#[tokio::test]
async fn test_plugin_health_status() {
    let health = PluginHealthStatus {
        healthy: true,
        message: "Plugin is healthy".to_string(),
        details: json!({"uptime": 3600}),
    };

    assert!(health.healthy);
    assert_eq!(health.message, "Plugin is healthy");
    assert!(health.details.is_object());
}

#[tokio::test]
async fn test_plugin_manager_creation() {
    let temp_dir = TempDir::new().unwrap();
    let plugin_dir = temp_dir.path().join("plugins");
    let data_dir = temp_dir.path().join("data");

    let manager = PluginManager::new(plugin_dir.clone(), data_dir.clone());

    // Basic checks that manager was created
    assert_eq!(manager.plugin_dir, plugin_dir);
    assert_eq!(manager.data_dir, data_dir);
}

#[tokio::test]
async fn test_plugin_loader_creation() {
    let loader = PluginLoader::new();
    let supported_types = loader.supported_types();

    assert!(!supported_types.is_empty());
    assert!(supported_types.contains(&"rust".to_string()));
    assert!(supported_types.contains(&"python".to_string()));
}

#[tokio::test]
async fn test_plugin_registry_creation() {
    let registry = PluginRegistry::new();
    let plugins = registry.list_plugins();

    assert!(plugins.is_empty());
}

#[tokio::test]
async fn test_plugin_registry_with_storage() {
    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path().join("registry.toml");

    let registry = PluginRegistry::with_storage(registry_path);
    let plugins = registry.list_plugins();

    assert!(plugins.is_empty());
}

#[tokio::test]
async fn test_plugin_registry_register_plugin() {
    let mut registry = PluginRegistry::new();
    let metadata = create_test_metadata();
    let config = create_test_config();

    let result = registry
        .register_plugin(metadata.clone(), config, None)
        .await;
    assert!(result.is_ok());

    let retrieved = registry.get_plugin(&metadata.id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, metadata.id);
}

#[tokio::test]
async fn test_plugin_registry_duplicate_registration() {
    let mut registry = PluginRegistry::new();
    let metadata = create_test_metadata();
    let config = create_test_config();

    // First registration should succeed
    let result1 = registry
        .register_plugin(metadata.clone(), config.clone(), None)
        .await;
    assert!(result1.is_ok());

    // Second registration should fail
    let result2 = registry.register_plugin(metadata, config, None).await;
    assert!(result2.is_err());
}

#[tokio::test]
async fn test_plugin_registry_unregister_plugin() {
    let mut registry = PluginRegistry::new();
    let metadata = create_test_metadata();
    let config = create_test_config();

    // Register plugin
    registry
        .register_plugin(metadata.clone(), config, None)
        .await
        .unwrap();
    assert!(registry.get_plugin(&metadata.id).is_some());

    // Unregister plugin
    let result = registry.unregister_plugin(&metadata.id).await;
    assert!(result.is_ok());
    assert!(registry.get_plugin(&metadata.id).is_none());
}

#[tokio::test]
async fn test_plugin_registry_search_by_capability() {
    let mut registry = PluginRegistry::new();
    let metadata = create_test_metadata();
    let config = create_test_config();

    registry
        .register_plugin(metadata, config, None)
        .await
        .unwrap();

    let results = registry.search_by_capability("test");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test_plugin");

    let no_results = registry.search_by_capability("nonexistent");
    assert!(no_results.is_empty());
}

#[tokio::test]
async fn test_plugin_registry_search_plugins() {
    let mut registry = PluginRegistry::new();
    let metadata = create_test_metadata();
    let config = create_test_config();

    registry
        .register_plugin(metadata, config, None)
        .await
        .unwrap();

    let results = registry.search_plugins("Test");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test_plugin");

    let results2 = registry.search_plugins("unit testing");
    assert_eq!(results2.len(), 1);

    let no_results = registry.search_plugins("nonexistent");
    assert!(no_results.is_empty());
}

#[tokio::test]
async fn test_plugin_discovery() {
    let temp_dir = TempDir::new().unwrap();
    let registry = PluginRegistry::new();

    // Create test plugin directory
    create_test_plugin_dir(&temp_dir, "test_plugin")
        .await
        .unwrap();

    let result = registry
        .discover_plugins(temp_dir.path().to_path_buf())
        .await;
    assert!(result.is_ok());

    let discovery_result = result.unwrap();
    assert_eq!(discovery_result.plugins.len(), 1);
    assert_eq!(discovery_result.plugins[0].id, "test_plugin");
    assert_eq!(discovery_result.scanned_dirs.len(), 1);
}

#[tokio::test]
async fn test_plugin_discovery_empty_directory() {
    let temp_dir = TempDir::new().unwrap();
    let registry = PluginRegistry::new();

    let result = registry
        .discover_plugins(temp_dir.path().to_path_buf())
        .await;
    assert!(result.is_ok());

    let discovery_result = result.unwrap();
    assert!(discovery_result.plugins.is_empty());
    assert_eq!(discovery_result.scanned_dirs.len(), 1);
}

#[tokio::test]
async fn test_plugin_discovery_nonexistent_directory() {
    let registry = PluginRegistry::new();
    let nonexistent_path = PathBuf::from("/nonexistent/path");

    let result = registry.discover_plugins(nonexistent_path).await;
    assert!(result.is_ok());

    let discovery_result = result.unwrap();
    assert!(discovery_result.plugins.is_empty());
    assert_eq!(discovery_result.errors.len(), 1);
}

#[tokio::test]
async fn test_basic_plugin_execution() {
    let metadata = create_test_metadata();
    let plugin = loader::BasicPlugin::new(metadata);

    let config = create_test_config();
    let context = PluginContext {
        plugin_id: "test_plugin".to_string(),
        execution_id: "exec_123".to_string(),
        config,
        services: HashMap::new(),
        data_dir: PathBuf::from("/tmp/data"),
        temp_dir: PathBuf::from("/tmp/temp"),
    };

    let input = json!({"test": "data"});
    let result = plugin.execute(input.clone(), &context).await;

    assert!(result.is_ok());
    let plugin_result = result.unwrap();
    assert!(plugin_result.success);
    assert!(plugin_result.error.is_none());
    assert_eq!(plugin_result.data["input"], input);
}

#[tokio::test]
async fn test_plugin_security_manager() {
    let security_manager = manager::PluginSecurityManager::new();

    assert!(!security_manager.is_blocked("test_plugin"));

    let permissions = PluginPermissions::default();
    let result = security_manager.validate_permissions("test_plugin", &permissions);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_plugin_resource_monitor() {
    let resource_monitor = manager::PluginResourceMonitor::new();

    let can_execute = resource_monitor.can_execute("test_plugin").await;
    assert!(can_execute);
}
