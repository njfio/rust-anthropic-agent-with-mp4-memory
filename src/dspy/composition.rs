//! Advanced composition patterns for DSPy modules
//!
//! This module provides advanced composition patterns including parallel execution,
//! conditional routing, and complex pipeline building capabilities.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::module::{BaseModule, Module, ModuleMetadata, ModuleStats};
use crate::dspy::signature::{FieldType, Signature, SignatureBuilder};
use async_trait::async_trait;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;

use tracing::{debug, info, warn};

/// Parallel execution module that runs multiple modules concurrently
pub struct Parallel<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + Clone,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Base module functionality
    base: BaseModule<I, Vec<O>>,
    /// Modules to execute in parallel
    modules: Vec<Arc<dyn Module<Input = I, Output = O>>>,
    /// Phantom data for type safety
    _phantom: PhantomData<(I, O)>,
}

impl<I, O> Parallel<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + Clone,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new parallel execution module
    pub fn new(name: String, modules: Vec<Arc<dyn Module<Input = I, Output = O>>>) -> Self {
        if modules.is_empty() {
            panic!("Parallel module must have at least one module");
        }

        let module_names: Vec<String> = modules.iter().map(|m| m.name().to_string()).collect();
        let signature = SignatureBuilder::new(&name)
            .description(&format!(
                "Parallel execution of: {}",
                module_names.join(", ")
            ))
            .input_field("input", "Parallel input", FieldType::Object(vec![]))
            .output_field(
                "outputs",
                "Parallel outputs",
                FieldType::Array(Box::new(FieldType::Object(vec![]))),
            )
            .build();

        let base = BaseModule::new(name, signature);

        Self {
            base,
            modules,
            _phantom: PhantomData,
        }
    }

    /// Create with metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.base = self.base.with_metadata(metadata);
        self
    }

    /// Get the modules being executed in parallel
    pub fn modules(&self) -> &[Arc<dyn Module<Input = I, Output = O>>] {
        &self.modules
    }

    /// Add another module to the parallel execution
    pub fn add_module(mut self, module: Arc<dyn Module<Input = I, Output = O>>) -> Self {
        self.modules.push(module);
        self
    }
}

#[async_trait]
impl<I, O> Module for Parallel<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + Clone,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    type Input = I;
    type Output = Vec<O>;

    fn id(&self) -> &str {
        &self.base.id
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.base.signature
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        debug!(
            "Parallel[{}] starting execution with {} modules",
            self.name(),
            self.modules.len()
        );

        let start_time = std::time::Instant::now();

        // Create futures for all modules
        let futures: Vec<_> = self
            .modules
            .iter()
            .enumerate()
            .map(|(idx, module)| {
                let input_clone = input.clone();
                let module_name = module.name().to_string();
                async move {
                    debug!(
                        "Parallel[{}] executing module {}: {}",
                        self.name(),
                        idx,
                        module_name
                    );
                    module
                        .forward(input_clone)
                        .await
                        .map_err(|e| DspyError::Module {
                            module_name: module_name.clone(),
                            message: format!("Parallel execution failed: {}", e),
                        })
                }
            })
            .collect();

        // Execute all modules concurrently
        let results = try_join_all(futures).await?;

        // Update stats manually
        let execution_time = start_time.elapsed();
        let mut stats = self.base.stats.write().await;
        stats.record_success(execution_time);

        info!(
            "Parallel[{}] completed successfully with {} results",
            self.name(),
            results.len()
        );

        Ok(results)
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.base.metadata
    }

    fn supports_compilation(&self) -> bool {
        self.modules.iter().any(|m| m.supports_compilation())
    }

    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        if !self.supports_compilation() {
            return Ok(());
        }

        info!("Parallel[{}] starting compilation", self.name());

        // For parallel compilation, we would need to decompose examples
        // This is a complex problem that would require sophisticated example splitting
        warn!(
            "Parallel[{}] compilation not fully implemented - requires example decomposition",
            self.name()
        );

        // Record compilation in stats
        let mut stats = self.base.stats.write().await;
        stats.record_compilation();

        info!("Parallel[{}] compilation completed", self.name());
        Ok(())
    }

    fn is_compiled(&self) -> bool {
        self.modules.iter().all(|m| m.is_compiled())
    }

    async fn reset(&mut self) -> DspyResult<()> {
        info!("Parallel[{}] resetting compilation state", self.name());

        let mut stats = self.base.stats.write().await;
        stats.reset_compilation();

        Ok(())
    }

    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }
}

/// Conditional routing module that selects which module to execute based on input
pub struct Conditional<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Base module functionality
    base: BaseModule<I, O>,
    /// Condition function to determine which module to use
    condition: Arc<dyn Fn(&I) -> usize + Send + Sync>,
    /// Modules to choose from
    modules: Vec<Arc<dyn Module<Input = I, Output = O>>>,
    /// Default module index if condition fails
    default_module: usize,
    /// Phantom data for type safety
    _phantom: PhantomData<(I, O)>,
}

impl<I, O> Conditional<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new conditional routing module
    pub fn new(
        name: String,
        condition: Arc<dyn Fn(&I) -> usize + Send + Sync>,
        modules: Vec<Arc<dyn Module<Input = I, Output = O>>>,
        default_module: usize,
    ) -> DspyResult<Self> {
        if modules.is_empty() {
            return Err(DspyError::configuration(
                "modules",
                "Must have at least one module",
            ));
        }

        if default_module >= modules.len() {
            return Err(DspyError::configuration(
                "default_module",
                "Default module index out of bounds",
            ));
        }

        let module_names: Vec<String> = modules.iter().map(|m| m.name().to_string()).collect();
        let signature = SignatureBuilder::new(&name)
            .description(&format!(
                "Conditional routing among: {}",
                module_names.join(", ")
            ))
            .input_field("input", "Conditional input", FieldType::Object(vec![]))
            .output_field("output", "Conditional output", FieldType::Object(vec![]))
            .build();

        let base = BaseModule::new(name, signature);

        Ok(Self {
            base,
            condition,
            modules,
            default_module,
            _phantom: PhantomData,
        })
    }

    /// Create with metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.base = self.base.with_metadata(metadata);
        self
    }

    /// Get the modules available for routing
    pub fn modules(&self) -> &[Arc<dyn Module<Input = I, Output = O>>] {
        &self.modules
    }
}

#[async_trait]
impl<I, O> Module for Conditional<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    type Input = I;
    type Output = O;

    fn id(&self) -> &str {
        &self.base.id
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.base.signature
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        debug!("Conditional[{}] evaluating routing condition", self.name());

        let start_time = std::time::Instant::now();

        // Evaluate condition to select module
        let selected_index = (self.condition)(&input);
        let module_index = if selected_index < self.modules.len() {
            selected_index
        } else {
            warn!(
                "Conditional[{}] condition returned invalid index {}, using default {}",
                self.name(),
                selected_index,
                self.default_module
            );
            self.default_module
        };

        let selected_module = &self.modules[module_index];
        debug!(
            "Conditional[{}] routing to module {}: {}",
            self.name(),
            module_index,
            selected_module.name()
        );

        // Execute selected module
        let result = selected_module
            .forward(input)
            .await
            .map_err(|e| DspyError::Module {
                module_name: selected_module.name().to_string(),
                message: format!("Conditional execution failed: {}", e),
            })?;

        // Update stats manually
        let execution_time = start_time.elapsed();
        let mut stats = self.base.stats.write().await;
        stats.record_success(execution_time);

        info!(
            "Conditional[{}] completed successfully using module: {}",
            self.name(),
            selected_module.name()
        );

        Ok(result)
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.base.metadata
    }

    fn supports_compilation(&self) -> bool {
        self.modules.iter().any(|m| m.supports_compilation())
    }

    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        if !self.supports_compilation() {
            return Ok(());
        }

        info!("Conditional[{}] starting compilation", self.name());

        // For conditional compilation, we would need to route examples to appropriate modules
        warn!(
            "Conditional[{}] compilation not fully implemented - requires example routing",
            self.name()
        );

        let mut stats = self.base.stats.write().await;
        stats.record_compilation();

        info!("Conditional[{}] compilation completed", self.name());
        Ok(())
    }

    fn is_compiled(&self) -> bool {
        self.modules.iter().all(|m| m.is_compiled())
    }

    async fn reset(&mut self) -> DspyResult<()> {
        info!("Conditional[{}] resetting compilation state", self.name());

        let mut stats = self.base.stats.write().await;
        stats.reset_compilation();

        Ok(())
    }

    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }
}
