//! DSPy integration for the Agent system
//!
//! This module provides seamless integration between the Agent system and DSPy modules,
//! enabling agents to create, use, and optimize DSPy modules for various tasks.

use crate::agent::Agent;
use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    examples::{Example, ExampleSet},
    module::Module,
    optimization::{OptimizationMetrics, OptimizationStrategy, Optimizer},
    predictor::{Predict, PredictConfig},
    signature::Signature,
    teleprompter::{Teleprompter, TeleprompterConfig},
};
use crate::security::{SecurityContext, SecurityEvent, SecurityManager};
use crate::utils::error::{AgentError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// DSPy module registry for managing agent-created modules
#[derive(Default)]
pub struct DspyModuleRegistry {
    /// Registered modules by ID
    modules: HashMap<String, Box<dyn ModuleInfo>>,
    /// Module metadata
    metadata: HashMap<String, DspyModuleMetadata>,
    /// Security policies for modules
    security_policies: HashMap<String, Vec<String>>,
}

impl std::fmt::Debug for DspyModuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DspyModuleRegistry")
            .field("modules_count", &self.modules.len())
            .field("metadata", &self.metadata)
            .field("security_policies", &self.security_policies)
            .finish()
    }
}

/// Information about a registered DSPy module
pub trait ModuleInfo: Send + Sync {
    /// Get module ID
    fn id(&self) -> &str;
    /// Get module name
    fn name(&self) -> &str;
    /// Get module description
    fn description(&self) -> Option<&str>;
    /// Get module type
    fn module_type(&self) -> &str;
    /// Check if module supports compilation
    fn supports_compilation(&self) -> bool;
    /// Get module creation timestamp
    fn created_at(&self) -> chrono::DateTime<chrono::Utc>;
}

/// Metadata for DSPy modules created by agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyModuleMetadata {
    /// Module ID
    pub id: String,
    /// Module name
    pub name: String,
    /// Module description
    pub description: Option<String>,
    /// Module type (e.g., "Predict", "Chain", "Parallel")
    pub module_type: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last used timestamp
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Usage count
    pub usage_count: u64,
    /// Compilation status
    pub is_compiled: bool,
    /// Optimization metrics
    pub optimization_metrics: Option<OptimizationMetrics>,
    /// Security context when created
    pub security_context: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Default for DspyModuleMetadata {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Unnamed Module".to_string(),
            description: None,
            module_type: "Unknown".to_string(),
            created_at: chrono::Utc::now(),
            last_used_at: None,
            usage_count: 0,
            is_compiled: false,
            optimization_metrics: None,
            security_context: None,
            tags: Vec::new(),
        }
    }
}

/// Configuration for DSPy integration with agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyAgentConfig {
    /// Enable automatic module optimization
    pub auto_optimize: bool,
    /// Maximum number of modules to keep in registry
    pub max_modules: usize,
    /// Default optimization strategy
    pub default_optimization_strategy: OptimizationStrategy,
    /// Enable security validation for modules
    pub enable_security_validation: bool,
    /// Enable audit logging for module operations
    pub enable_audit_logging: bool,
    /// Module cache TTL in seconds
    pub module_cache_ttl: u64,
}

impl Default for DspyAgentConfig {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            max_modules: 100,
            default_optimization_strategy: OptimizationStrategy::bootstrap(5, 10),
            enable_security_validation: true,
            enable_audit_logging: true,
            module_cache_ttl: 3600, // 1 hour
        }
    }
}

/// DSPy integration extension for Agent
#[derive(Debug)]
pub struct DspyAgentExtension {
    /// Module registry
    registry: Arc<RwLock<DspyModuleRegistry>>,
    /// Configuration
    config: DspyAgentConfig,
    /// Anthropic client for module creation
    anthropic_client: AnthropicClient,
    /// Security manager reference
    security_manager: Option<Arc<SecurityManager>>,
}

impl DspyAgentExtension {
    /// Create a new DSPy agent extension
    pub fn new(
        config: DspyAgentConfig,
        anthropic_client: AnthropicClient,
        security_manager: Option<Arc<SecurityManager>>,
    ) -> Self {
        Self {
            registry: Arc::new(RwLock::new(DspyModuleRegistry::default())),
            config,
            anthropic_client,
            security_manager,
        }
    }

    /// Create a DSPy Predict module from a signature
    pub async fn create_predict_module<I, O>(
        &self,
        signature: Signature<I, O>,
        config: Option<PredictConfig>,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<Predict<I, O>>
    where
        I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone + 'static,
        O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone + 'static,
    {
        // Security validation
        if self.config.enable_security_validation {
            self.validate_module_creation_security(security_context)
                .await?;
        }

        // Create the predict module
        let predict_config = config.unwrap_or_default();
        let predict_module = Predict::with_config(
            signature,
            Arc::new(self.anthropic_client.clone()),
            predict_config,
        );

        // Register the module
        let module_id = predict_module.id().to_string();
        let metadata = DspyModuleMetadata {
            id: module_id.clone(),
            name: predict_module.name().to_string(),
            description: Some("Predict module created by agent".to_string()),
            module_type: "Predict".to_string(),
            created_at: chrono::Utc::now(),
            security_context: security_context.map(|ctx| format!("{:?}", ctx)),
            ..Default::default()
        };

        // Store in registry
        let mut registry = self.registry.write().await;
        registry.metadata.insert(module_id.clone(), metadata);

        // Audit logging
        if self.config.enable_audit_logging {
            self.log_module_creation(&module_id, "Predict", security_context)
                .await;
        }

        info!("Created DSPy Predict module: {}", module_id);
        Ok(predict_module)
    }

    /// Use a DSPy module with input validation and security checks
    pub async fn use_module<I, O>(
        &self,
        module: &dyn Module<Input = I, Output = O>,
        input: I,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<O>
    where
        I: Serialize + for<'de> Deserialize<'de> + Send + Sync,
        O: Serialize + for<'de> Deserialize<'de> + Send + Sync,
    {
        // Security validation
        if self.config.enable_security_validation {
            self.validate_module_usage_security(module.id(), security_context)
                .await?;
        }

        // Input validation
        module.validate_input(&input).await?;

        // Execute the module
        let start_time = std::time::Instant::now();
        let result = module.forward(input).await?;
        let execution_time = start_time.elapsed();

        // Output validation
        module.validate_output(&result).await?;

        // Update usage statistics
        self.update_module_usage_stats(module.id(), execution_time)
            .await;

        // Audit logging
        if self.config.enable_audit_logging {
            self.log_module_usage(module.id(), execution_time, security_context)
                .await;
        }

        debug!(
            "Used DSPy module {} in {:.2}ms",
            module.id(),
            execution_time.as_millis()
        );

        Ok(result)
    }

    /// Optimize a module using teleprompter
    pub async fn optimize_module<I, O>(
        &self,
        module: &mut dyn Module<Input = I, Output = O>,
        examples: ExampleSet<I, O>,
        strategy: Option<OptimizationStrategy>,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<OptimizationMetrics>
    where
        I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
        O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    {
        // Security validation
        if self.config.enable_security_validation {
            self.validate_module_optimization_security(module.id(), security_context)
                .await?;
        }

        // Use provided strategy or default
        let optimization_strategy =
            strategy.unwrap_or_else(|| self.config.default_optimization_strategy.clone());

        // Create teleprompter
        let teleprompter_config = TeleprompterConfig {
            strategy: optimization_strategy,
            max_iterations: 50,
            convergence_threshold: 0.01,
            min_improvement: 0.001,
            early_stopping_patience: 5,
            validation_split: 0.2,
            use_cross_validation: false,
            cv_folds: 5,
            random_seed: None,
            verbose: true,
            custom_params: HashMap::new(),
        };

        let mut teleprompter = Teleprompter::with_config(teleprompter_config);

        // Optimize the module
        info!("Starting optimization for module: {}", module.id());
        let optimization_result = teleprompter.optimize(module, examples).await?;

        // Update module metadata
        self.update_module_optimization_metrics(module.id(), optimization_result.metrics.clone())
            .await;

        // Audit logging
        if self.config.enable_audit_logging {
            self.log_module_optimization(
                module.id(),
                &optimization_result.metrics,
                security_context,
            )
            .await;
        }

        info!(
            "Completed optimization for module {} with score: {:.3}",
            module.id(),
            optimization_result.metrics.best_score
        );

        Ok(optimization_result.metrics)
    }

    /// Get module registry statistics
    pub async fn get_registry_stats(&self) -> HashMap<String, serde_json::Value> {
        let registry = self.registry.read().await;
        let mut stats = HashMap::new();

        stats.insert(
            "total_modules".to_string(),
            serde_json::Value::Number(registry.metadata.len().into()),
        );

        let compiled_count = registry.metadata.values().filter(|m| m.is_compiled).count();
        stats.insert(
            "compiled_modules".to_string(),
            serde_json::Value::Number(compiled_count.into()),
        );

        let total_usage: u64 = registry.metadata.values().map(|m| m.usage_count).sum();
        stats.insert(
            "total_usage_count".to_string(),
            serde_json::Value::Number(total_usage.into()),
        );

        // Module types distribution
        let mut type_counts = HashMap::new();
        for metadata in registry.metadata.values() {
            *type_counts.entry(&metadata.module_type).or_insert(0) += 1;
        }
        stats.insert(
            "module_types".to_string(),
            serde_json::to_value(type_counts).unwrap_or_default(),
        );

        stats
    }

    /// List all registered modules
    pub async fn list_modules(&self) -> Vec<DspyModuleMetadata> {
        let registry = self.registry.read().await;
        registry.metadata.values().cloned().collect()
    }

    /// Remove a module from the registry
    pub async fn remove_module(
        &self,
        module_id: &str,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<()> {
        // Security validation
        if self.config.enable_security_validation {
            self.validate_module_removal_security(module_id, security_context)
                .await?;
        }

        let mut registry = self.registry.write().await;
        registry.modules.remove(module_id);
        registry.metadata.remove(module_id);
        registry.security_policies.remove(module_id);

        // Audit logging
        if self.config.enable_audit_logging {
            self.log_module_removal(module_id, security_context).await;
        }

        info!("Removed DSPy module: {}", module_id);
        Ok(())
    }

    /// Security validation for module creation
    async fn validate_module_creation_security(
        &self,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<()> {
        if let Some(security_manager) = &self.security_manager {
            if let Some(context) = security_context {
                // Check permission for module creation
                let has_permission = security_manager
                    .check_permission(context, "dspy_modules", "create")
                    .await
                    .map_err(|e| DspyError::configuration("security", &e.to_string()))?;

                if !has_permission {
                    return Err(DspyError::configuration(
                        "security",
                        "Module creation not authorized",
                    ));
                }

                // Log the security event
                let event = SecurityEvent::DataAccess {
                    user_id: context.user_id.clone(),
                    resource: "dspy_modules".to_string(),
                    action: "create".to_string(),
                    sensitive: false,
                };

                security_manager
                    .log_security_event(event)
                    .await
                    .map_err(|e| DspyError::configuration("audit", &e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Security validation for module usage
    async fn validate_module_usage_security(
        &self,
        module_id: &str,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<()> {
        if let Some(security_manager) = &self.security_manager {
            if let Some(context) = security_context {
                // Check permission for module usage
                let resource = format!("dspy_modules/{}", module_id);
                let has_permission = security_manager
                    .check_permission(context, &resource, "execute")
                    .await
                    .map_err(|e| DspyError::configuration("security", &e.to_string()))?;

                if !has_permission {
                    return Err(DspyError::configuration(
                        "security",
                        "Module usage not authorized",
                    ));
                }

                // Log the security event
                let event = SecurityEvent::DataAccess {
                    user_id: context.user_id.clone(),
                    resource,
                    action: "execute".to_string(),
                    sensitive: false,
                };

                security_manager
                    .log_security_event(event)
                    .await
                    .map_err(|e| DspyError::configuration("audit", &e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Security validation for module optimization
    async fn validate_module_optimization_security(
        &self,
        module_id: &str,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<()> {
        if let Some(security_manager) = &self.security_manager {
            if let Some(context) = security_context {
                // Check permission for module optimization
                let resource = format!("dspy_modules/{}", module_id);
                let has_permission = security_manager
                    .check_permission(context, &resource, "optimize")
                    .await
                    .map_err(|e| DspyError::configuration("security", &e.to_string()))?;

                if !has_permission {
                    return Err(DspyError::configuration(
                        "security",
                        "Module optimization not authorized",
                    ));
                }

                // Log the security event
                let event = SecurityEvent::DataAccess {
                    user_id: context.user_id.clone(),
                    resource,
                    action: "optimize".to_string(),
                    sensitive: true, // Optimization is considered sensitive
                };

                security_manager
                    .log_security_event(event)
                    .await
                    .map_err(|e| DspyError::configuration("audit", &e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Security validation for module removal
    async fn validate_module_removal_security(
        &self,
        module_id: &str,
        security_context: Option<&SecurityContext>,
    ) -> DspyResult<()> {
        if let Some(security_manager) = &self.security_manager {
            if let Some(context) = security_context {
                // Check permission for module removal
                let resource = format!("dspy_modules/{}", module_id);
                let has_permission = security_manager
                    .check_permission(context, &resource, "delete")
                    .await
                    .map_err(|e| DspyError::configuration("security", &e.to_string()))?;

                if !has_permission {
                    return Err(DspyError::configuration(
                        "security",
                        "Module removal not authorized",
                    ));
                }

                // Log the security event
                let event = SecurityEvent::DataAccess {
                    user_id: context.user_id.clone(),
                    resource,
                    action: "delete".to_string(),
                    sensitive: true, // Deletion is considered sensitive
                };

                security_manager
                    .log_security_event(event)
                    .await
                    .map_err(|e| DspyError::configuration("audit", &e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Update module usage statistics
    async fn update_module_usage_stats(
        &self,
        module_id: &str,
        execution_time: std::time::Duration,
    ) {
        let mut registry = self.registry.write().await;
        if let Some(metadata) = registry.metadata.get_mut(module_id) {
            metadata.usage_count += 1;
            metadata.last_used_at = Some(chrono::Utc::now());
        }
    }

    /// Update module optimization metrics
    async fn update_module_optimization_metrics(
        &self,
        module_id: &str,
        metrics: OptimizationMetrics,
    ) {
        let mut registry = self.registry.write().await;
        if let Some(metadata) = registry.metadata.get_mut(module_id) {
            metadata.is_compiled = true;
            metadata.optimization_metrics = Some(metrics);
        }
    }

    /// Log module creation for audit
    async fn log_module_creation(
        &self,
        module_id: &str,
        module_type: &str,
        security_context: Option<&SecurityContext>,
    ) {
        info!(
            "AUDIT: DSPy module created - ID: {}, Type: {}, User: {:?}",
            module_id,
            module_type,
            security_context.map(|ctx| &ctx.user_id)
        );
    }

    /// Log module usage for audit
    async fn log_module_usage(
        &self,
        module_id: &str,
        execution_time: std::time::Duration,
        security_context: Option<&SecurityContext>,
    ) {
        info!(
            "AUDIT: DSPy module used - ID: {}, Duration: {:.2}ms, User: {:?}",
            module_id,
            execution_time.as_millis(),
            security_context.map(|ctx| &ctx.user_id)
        );
    }

    /// Log module optimization for audit
    async fn log_module_optimization(
        &self,
        module_id: &str,
        metrics: &OptimizationMetrics,
        security_context: Option<&SecurityContext>,
    ) {
        info!(
            "AUDIT: DSPy module optimized - ID: {}, Score: {:.3}, User: {:?}",
            module_id,
            metrics.best_score,
            security_context.map(|ctx| &ctx.user_id)
        );
    }

    /// Log module removal for audit
    async fn log_module_removal(
        &self,
        module_id: &str,
        security_context: Option<&SecurityContext>,
    ) {
        info!(
            "AUDIT: DSPy module removed - ID: {}, User: {:?}",
            module_id,
            security_context.map(|ctx| &ctx.user_id)
        );
    }
}
