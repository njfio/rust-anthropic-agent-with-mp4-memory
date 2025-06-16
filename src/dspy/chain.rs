//! Chain module for sequential DSPy module composition
//!
//! This module provides the Chain struct for composing DSPy modules in sequence,
//! where the output of one module becomes the input to the next module.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::module::{BaseModule, Module, ModuleMetadata, ModuleStats};
use crate::dspy::signature::{FieldType, Signature, SignatureBuilder};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;

use tracing::{debug, info};

/// Chain module for sequential execution of two modules
pub struct Chain<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Base module functionality
    base: BaseModule<I, O>,
    /// First module in the chain
    first_module: Arc<dyn Module<Input = I, Output = M>>,
    /// Second module in the chain
    second_module: Arc<dyn Module<Input = M, Output = O>>,
    /// Phantom data for type safety
    _phantom: PhantomData<(I, M, O)>,
}

impl<I, M, O> Chain<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    /// Create a new chain module
    pub fn new(
        name: String,
        first_module: Arc<dyn Module<Input = I, Output = M>>,
        second_module: Arc<dyn Module<Input = M, Output = O>>,
    ) -> Self {
        // Create a combined signature for the chain
        let signature = SignatureBuilder::new(&name)
            .description(&format!(
                "Chain of {} -> {}",
                first_module.name(),
                second_module.name()
            ))
            .input_field("input", "Chain input", FieldType::Object(vec![]))
            .output_field("output", "Chain output", FieldType::Object(vec![]))
            .build();

        let base = BaseModule::new(name, signature);

        Self {
            base,
            first_module,
            second_module,
            _phantom: PhantomData,
        }
    }

    /// Create a new chain with metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.base = self.base.with_metadata(metadata);
        self
    }

    /// Get the first module in the chain
    pub fn first_module(&self) -> &Arc<dyn Module<Input = I, Output = M>> {
        &self.first_module
    }

    /// Get the second module in the chain
    pub fn second_module(&self) -> &Arc<dyn Module<Input = M, Output = O>> {
        &self.second_module
    }

    /// Chain another module to create a longer chain
    pub fn chain<N>(self, next_module: Arc<dyn Module<Input = O, Output = N>>) -> Chain<I, O, N>
    where
        N: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    {
        let chain_name = format!("{}_extended", self.base.name);
        Chain::new(chain_name, Arc::new(ChainWrapper::new(self)), next_module)
    }
}

#[async_trait]
impl<I, M, O> Module for Chain<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
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
        debug!(
            "Chain[{}] starting execution: {} -> {}",
            self.name(),
            self.first_module.name(),
            self.second_module.name()
        );

        let start_time = std::time::Instant::now();

        // Execute first module
        debug!(
            "Chain[{}] executing first module: {}",
            self.name(),
            self.first_module.name()
        );
        let intermediate_result =
            self.first_module
                .forward(input)
                .await
                .map_err(|e| DspyError::ChainExecution {
                    module_name: self.first_module.name().to_string(),
                    stage: "first".to_string(),
                    source: Box::new(e),
                })?;

        // Execute second module with intermediate result
        debug!(
            "Chain[{}] executing second module: {}",
            self.name(),
            self.second_module.name()
        );
        let final_result = self
            .second_module
            .forward(intermediate_result)
            .await
            .map_err(|e| DspyError::ChainExecution {
                module_name: self.second_module.name().to_string(),
                stage: "second".to_string(),
                source: Box::new(e),
            })?;

        // Update stats manually
        let execution_time = start_time.elapsed();
        let mut stats = self.base.stats.write().await;
        stats.record_success(execution_time);

        info!(
            "Chain[{}] completed successfully: {} -> {}",
            self.name(),
            self.first_module.name(),
            self.second_module.name()
        );

        Ok(final_result)
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.base.metadata
    }

    fn supports_compilation(&self) -> bool {
        self.first_module.supports_compilation() || self.second_module.supports_compilation()
    }

    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        if !self.supports_compilation() {
            return Ok(());
        }

        info!("Chain[{}] starting compilation", self.name());

        // For chain compilation, we need to split examples into intermediate stages
        // This is a simplified approach - in practice, we'd need more sophisticated
        // example decomposition based on the intermediate type M

        // For now, we'll attempt to compile each module independently
        // This is a limitation that would be addressed in a full implementation

        if self.first_module.supports_compilation() {
            // We can't easily compile the first module without intermediate examples
            debug!("Chain[{}] first module supports compilation but intermediate examples not available", self.name());
        }

        if self.second_module.supports_compilation() {
            // We can't easily compile the second module without intermediate examples
            debug!("Chain[{}] second module supports compilation but intermediate examples not available", self.name());
        }

        // Record compilation in stats
        let mut stats = self.base.stats.write().await;
        stats.record_compilation();

        info!("Chain[{}] compilation completed", self.name());
        Ok(())
    }

    fn is_compiled(&self) -> bool {
        // Chain is compiled if both modules are compiled
        self.first_module.is_compiled() && self.second_module.is_compiled()
    }

    async fn reset(&mut self) -> DspyResult<()> {
        info!("Chain[{}] resetting compilation state", self.name());

        // Reset stats
        let mut stats = self.base.stats.write().await;
        stats.reset_compilation();

        Ok(())
    }

    fn stats(&self) -> &ModuleStats {
        // Return a reference to the default stats since we can't return a reference
        // to the async-protected stats. In practice, this would need a different approach.
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }
}

/// Wrapper to make a Chain usable as a module in another chain
struct ChainWrapper<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    chain: Chain<I, M, O>,
}

impl<I, M, O> ChainWrapper<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    fn new(chain: Chain<I, M, O>) -> Self {
        Self { chain }
    }
}

#[async_trait]
impl<I, M, O> Module for ChainWrapper<I, M, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    M: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    type Input = I;
    type Output = O;

    fn id(&self) -> &str {
        self.chain.id()
    }

    fn name(&self) -> &str {
        self.chain.name()
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        self.chain.signature()
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        self.chain.forward(input).await
    }

    fn metadata(&self) -> &ModuleMetadata {
        self.chain.metadata()
    }

    fn supports_compilation(&self) -> bool {
        self.chain.supports_compilation()
    }

    fn is_compiled(&self) -> bool {
        self.chain.is_compiled()
    }

    fn stats(&self) -> &ModuleStats {
        self.chain.stats()
    }
}
