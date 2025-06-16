# DSPy API Reference

This document provides comprehensive API documentation for the DSPy framework.

## Core Modules

### Signature<I, O>

Defines the input-output interface for DSPy modules.

```rust
pub struct Signature<I, O> {
    // Private fields
}

impl<I, O> Signature<I, O> {
    /// Create a new signature
    pub fn new(name: String) -> Self
    
    /// Set description
    pub fn with_description(mut self, description: &str) -> Self
    
    /// Add input field
    pub fn add_input_field(mut self, field: Field) -> Self
    
    /// Add output field  
    pub fn add_output_field(mut self, field: Field) -> Self
    
    /// Get signature name
    pub fn name(&self) -> &str
    
    /// Get description
    pub fn description(&self) -> Option<&str>
    
    /// Get input fields
    pub fn input_fields(&self) -> &[Field]
    
    /// Get output fields
    pub fn output_fields(&self) -> &[Field]
}
```

### Module Trait

The core trait that all DSPy modules implement.

```rust
#[async_trait]
pub trait Module: Send + Sync {
    type Input: Serialize + for<'de> Deserialize<'de> + Send + Sync;
    type Output: Serialize + for<'de> Deserialize<'de> + Send + Sync;

    /// Module identifier
    fn id(&self) -> &str;
    
    /// Module name
    fn name(&self) -> &str;
    
    /// Module signature
    fn signature(&self) -> &Signature<Self::Input, Self::Output>;
    
    /// Forward pass through the module
    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output>;
    
    /// Validate input
    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        Ok(())
    }
    
    /// Validate output
    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        Ok(())
    }
    
    /// Get module metadata
    fn metadata(&self) -> &ModuleMetadata;
    
    /// Get module statistics
    fn stats(&self) -> &ModuleStats;
    
    /// Check if module supports compilation
    fn supports_compilation(&self) -> bool {
        false
    }
    
    /// Compile the module with examples
    async fn compile(&mut self, examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        Ok(())
    }
}
```

## Basic Modules

### Predict<I, O>

The fundamental prediction module.

```rust
pub struct Predict<I, O> {
    // Private fields
}

impl<I, O> Predict<I, O> {
    /// Create new Predict module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: PredictConfig,
    ) -> Self
    
    /// Get current configuration
    pub fn config(&self) -> &PredictConfig
    
    /// Update configuration
    pub fn set_config(&mut self, config: PredictConfig)
    
    /// Enable caching
    pub fn enable_caching(&mut self, cache_config: CacheConfig)
    
    /// Process batch of inputs
    pub async fn forward_batch(&self, inputs: Vec<I>) -> DspyResult<Vec<O>>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictConfig {
    pub temperature: f64,
    pub max_tokens: usize,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub stop_sequences: Vec<String>,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub max_retries: usize,
    pub retry_delay_ms: u64,
    pub timeout_seconds: u64,
    pub enable_streaming: bool,
    pub custom_headers: HashMap<String, String>,
}
```

### Chain<I, O>

Sequential module composition.

```rust
pub struct Chain<I, O> {
    // Private fields
}

impl<I, O> Chain<I, O> {
    /// Create new chain
    pub fn new() -> Self
    
    /// Add module to chain
    pub fn add_module<M>(mut self, module: M) -> Self
    where
        M: Module + 'static
    
    /// Add module with transformation
    pub fn add_module_with_transform<M, F>(
        mut self,
        module: M,
        transform: F,
    ) -> Self
    where
        M: Module + 'static,
        F: Fn(PreviousOutput) -> M::Input + Send + Sync + 'static
    
    /// Get number of modules
    pub fn len(&self) -> usize
    
    /// Check if chain is empty
    pub fn is_empty(&self) -> bool
    
    /// Get module at index
    pub fn get_module(&self, index: usize) -> Option<&dyn Module>
}
```

### Parallel<I, O>

Concurrent module execution.

```rust
pub struct Parallel<I, O> {
    // Private fields
}

impl<I, O> Parallel<I, O> {
    /// Create new parallel module
    pub fn new() -> Self
    
    /// Add named module
    pub fn add_module<M>(mut self, name: &str, module: M) -> Self
    where
        M: Module<Input = I> + 'static
    
    /// Set execution strategy
    pub fn with_strategy(mut self, strategy: ParallelStrategy) -> Self
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self
    
    /// Get module names
    pub fn module_names(&self) -> Vec<&str>
    
    /// Get module by name
    pub fn get_module(&self, name: &str) -> Option<&dyn Module<Input = I>>
}

#[derive(Debug, Clone)]
pub enum ParallelStrategy {
    /// Wait for all modules to complete
    WaitAll,
    /// Return when first module completes
    FirstComplete,
    /// Return when majority completes
    Majority,
    /// Custom strategy with minimum completion count
    MinimumCount(usize),
}
```

## Specialized Modules

### ChainOfThought<I, O>

Step-by-step reasoning module.

```rust
pub struct ChainOfThought<I, O> {
    // Private fields
}

impl<I, O> ChainOfThought<I, O> {
    /// Create new Chain of Thought module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ChainOfThoughtConfig,
    ) -> Self
    
    /// Get configuration
    pub fn config(&self) -> &ChainOfThoughtConfig
    
    /// Update configuration
    pub fn set_config(&mut self, config: ChainOfThoughtConfig)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfThoughtConfig {
    pub base: SpecializedModuleConfig,
    pub reasoning_template: String,
    pub include_step_numbers: bool,
    pub validate_chain: bool,
    pub min_confidence: f64,
    pub max_retries: usize,
}
```

### ReAct<I, O>

Reasoning and Acting module.

```rust
pub struct ReAct<I, O> {
    // Private fields
}

impl<I, O> ReAct<I, O> {
    /// Create new ReAct module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ReActConfig,
    ) -> Self
    
    /// Add tool to module
    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T)
    
    /// Get available tools
    pub fn get_available_tools(&self) -> &[String]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActConfig {
    pub base: SpecializedModuleConfig,
    pub available_tools: Vec<String>,
    pub max_cycles: usize,
    pub validate_actions: bool,
    pub tool_timeout_seconds: u64,
    pub continue_on_tool_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    pub step_number: usize,
    pub thought: String,
    pub action: Option<ReActAction>,
    pub observation: Option<String>,
    pub confidence: f64,
    pub execution_time_ms: f64,
}
```

### RAG<I, O>

Retrieval-Augmented Generation module.

```rust
pub struct RAG<I, O> {
    // Private fields
}

impl<I, O> RAG<I, O> {
    /// Create new RAG module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        memory_manager: Arc<Mutex<MemoryManager>>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        memory_manager: Arc<Mutex<MemoryManager>>,
        config: RAGConfig,
    ) -> Self
    
    /// Get configuration
    pub fn config(&self) -> &RAGConfig
    
    /// Update configuration
    pub fn set_config(&mut self, config: RAGConfig)
    
    /// Get last RAG result
    pub async fn get_last_rag_result(&self) -> Option<RAGResult<O>>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGConfig {
    pub base: SpecializedModuleConfig,
    pub num_documents: usize,
    pub min_relevance_score: f64,
    pub max_context_length: usize,
    pub enable_reranking: bool,
    pub enable_query_expansion: bool,
    pub retrieval_strategy: RetrievalStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    Semantic,
    Keyword,
    Hybrid,
    DensePassage,
}
```

### ProgramOfThought<I, O>

Code generation and execution module.

```rust
pub struct ProgramOfThought<I, O> {
    // Private fields
}

impl<I, O> ProgramOfThought<I, O> {
    /// Create new Program of Thought module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ProgramOfThoughtConfig,
    ) -> Self
    
    /// Get last generated code
    pub async fn get_last_generated_code(&self) -> Option<String>
    
    /// Get last execution result
    pub async fn get_last_execution_result(&self) -> Option<CodeExecutionResult>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramOfThoughtConfig {
    pub base: SpecializedModuleConfig,
    pub language: ProgrammingLanguage,
    pub execute_code: bool,
    pub execution_timeout_seconds: u64,
    pub max_code_length: usize,
    pub validate_syntax: bool,
    pub allowed_imports: Vec<String>,
    pub security_restrictions: SecurityRestrictions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgrammingLanguage {
    Python,
    JavaScript,
    Rust,
    Go,
}
```

### SelfImproving<I, O>

Adaptive learning module.

```rust
pub struct SelfImproving<I, O> {
    // Private fields
}

impl<I, O> SelfImproving<I, O> {
    /// Create new Self-Improving module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        base_module: Arc<RwLock<dyn Module<Input = I, Output = O>>>,
    ) -> Self
    
    /// Create with configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        base_module: Arc<RwLock<dyn Module<Input = I, Output = O>>>,
        config: SelfImprovingConfig,
    ) -> Self
    
    /// Add feedback for learning
    pub async fn add_feedback(
        &self,
        input: I,
        output: O,
        expected_output: Option<O>,
        score: f64,
        feedback_type: FeedbackType,
    ) -> DspyResult<()>
    
    /// Trigger improvement process
    pub async fn trigger_improvement(&self) -> DspyResult<ImprovementMetrics>
    
    /// Get improvement metrics
    pub async fn get_improvement_metrics(&self) -> ImprovementMetrics
    
    /// Get feedback history
    pub async fn get_feedback_history(&self) -> Vec<FeedbackEntry>
    
    /// Reset improvement state
    pub async fn reset_improvement_state(&self) -> DspyResult<()>
}
```

## Optimization and Compilation

### Teleprompter

Module optimization framework.

```rust
pub struct Teleprompter {
    // Private fields
}

impl Teleprompter {
    /// Create new teleprompter
    pub fn new(strategy: OptimizationStrategy) -> Self
    
    /// Optimize a module
    pub async fn optimize<I, O, M>(
        &mut self,
        module: &mut M,
        examples: ExampleSet<I, O>,
    ) -> DspyResult<OptimizationResult>
    where
        M: Module<Input = I, Output = O> + Send + Sync,
        I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
        O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    
    /// Get optimization strategy
    pub fn strategy(&self) -> &OptimizationStrategy
    
    /// Set optimization strategy
    pub fn set_strategy(&mut self, strategy: OptimizationStrategy)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Bootstrap { num_examples: usize, max_iterations: usize },
    RandomSampling { num_samples: usize, seed: Option<u64> },
    MIPROv2 { config: MIPROv2Config },
    BootstrapFinetune { config: BootstrapFinetuneConfig },
    MultiObjective { config: MultiObjectiveConfig },
}
```

### Example Management

```rust
pub struct Example<I, O> {
    // Private fields
}

impl<I, O> Example<I, O> {
    /// Create new example
    pub fn new(input: I, output: O) -> Self
    
    /// Create with metadata
    pub fn with_metadata(input: I, output: O, metadata: HashMap<String, serde_json::Value>) -> Self
    
    /// Get input
    pub fn input(&self) -> &I
    
    /// Get output
    pub fn output(&self) -> &O
    
    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value>
}

pub struct ExampleSet<I, O> {
    // Private fields
}

impl<I, O> ExampleSet<I, O> {
    /// Create new example set
    pub fn new() -> Self
    
    /// Add example
    pub fn add_example(&mut self, example: Example<I, O>)
    
    /// Add multiple examples
    pub fn add_examples(&mut self, examples: Vec<Example<I, O>>)
    
    /// Get example by index
    pub fn get(&self, index: usize) -> Option<&Example<I, O>>
    
    /// Get number of examples
    pub fn len(&self) -> usize
    
    /// Check if empty
    pub fn is_empty(&self) -> bool
    
    /// Split into train/test sets
    pub fn split(&self, train_ratio: f64) -> (ExampleSet<I, O>, ExampleSet<I, O>)
    
    /// Shuffle examples
    pub fn shuffle(&mut self, seed: Option<u64>)
    
    /// Clear all examples
    pub fn clear(&mut self)
}
```

## Evaluation

### Metrics

```rust
#[async_trait]
pub trait Metric<T>: Send + Sync {
    /// Evaluate metric
    async fn evaluate(&self, predicted: &T, expected: &T) -> DspyResult<MetricResult>;
    
    /// Get metric name
    fn name(&self) -> &str;
    
    /// Get metric description
    fn description(&self) -> Option<&str> { None }
    
    /// Check if higher scores are better
    fn higher_is_better(&self) -> bool { true }
}

pub struct ExactMatch;
pub struct SemanticSimilarity { /* fields */ }
pub struct F1Score;
pub struct CompositeMetric { /* fields */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub score: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

### Evaluator

```rust
pub struct Evaluator<I, O> {
    // Private fields
}

impl<I, O> Evaluator<I, O> {
    /// Create new evaluator
    pub fn new() -> Self
    
    /// Add metric
    pub fn add_metric<T>(mut self, name: &str, metric: Box<dyn Metric<T>>) -> Self
    
    /// Evaluate module
    pub async fn evaluate<M>(
        &self,
        module: &M,
        examples: &ExampleSet<I, O>,
    ) -> DspyResult<EvaluationResults>
    where
        M: Module<Input = I, Output = O>,
        I: Clone,
        O: Clone,
    
    /// Cross-validation evaluation
    pub async fn cross_validate<M>(
        &self,
        module: &mut M,
        examples: &ExampleSet<I, O>,
        folds: usize,
    ) -> DspyResult<CrossValidationResults>
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum DspyError {
    #[error("Signature error: {message}")]
    Signature { message: String },
    
    #[error("Module error in {module_name}: {message}")]
    Module { module_name: String, message: String },
    
    #[error("Optimization error with {strategy}: {message}")]
    Optimization { strategy: String, message: String },
    
    #[error("Evaluation error for {metric_name}: {message}")]
    Evaluation { metric_name: String, message: String },
    
    #[error("Compilation error: {message}")]
    Compilation { message: String },
    
    #[error("Serialization error in {context}: {message}")]
    Serialization { context: String, message: String },
    
    #[error("Configuration error for {parameter}: {message}")]
    Configuration { parameter: String, message: String },
    
    #[error("Cache error: {message}")]
    Cache { message: String },
    
    #[error("Anthropic API error: {0}")]
    Anthropic(#[from] crate::anthropic::AnthropicError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type DspyResult<T> = Result<T, DspyError>;
```

## Agent Integration

### DSPy Agent Extensions

```rust
impl Agent {
    /// Enable DSPy integration
    pub fn enable_dspy_integration(&mut self, config: Option<DspyAgentConfig>) -> Result<()>
    
    /// Create DSPy module
    pub async fn as_dspy_module<I, O>(
        &self,
        signature: Signature<I, O>,
        config: Option<PredictConfig>,
    ) -> DspyResult<Predict<I, O>>
    
    /// Use DSPy module
    pub async fn use_dspy_module<I, O>(
        &self,
        module: &dyn Module<Input = I, Output = O>,
        input: I,
    ) -> DspyResult<O>
    
    /// Optimize DSPy module
    pub async fn optimize_dspy_module<I, O>(
        &self,
        module: &mut dyn Module<Input = I, Output = O>,
        examples: ExampleSet<I, O>,
        strategy: Option<OptimizationStrategy>,
    ) -> DspyResult<OptimizationMetrics>
    
    /// Get DSPy registry statistics
    pub async fn get_dspy_registry_stats(&self) -> Result<HashMap<String, serde_json::Value>>
    
    /// List DSPy modules
    pub async fn list_dspy_modules(&self) -> Result<Vec<DspyModuleMetadata>>
    
    /// Remove DSPy module
    pub async fn remove_dspy_module(&self, module_id: &str) -> DspyResult<()>
}
```

## Tool Integration

### DSPy Module Tool

```rust
pub struct DspyModuleTool<I, O> {
    // Private fields
}

impl<I, O> DspyModuleTool<I, O> {
    /// Create new DSPy module tool
    pub fn new(
        module: Arc<dyn Module<Input = I, Output = O>>,
        tool_name: String,
        description: String,
        input_schema: serde_json::Value,
    ) -> Self
    
    /// Create from module with automatic schema generation
    pub fn from_module(
        module: Arc<dyn Module<Input = I, Output = O>>,
        tool_name: Option<String>,
        description: Option<String>,
    ) -> DspyResult<Self>
    
    /// Get performance metrics
    pub fn metrics(&self) -> &ToolMetrics
    
    /// Reset performance metrics
    pub fn reset_metrics(&mut self)
}

pub struct DspyToolBuilder<I, O> {
    // Private fields
}

impl<I, O> DspyToolBuilder<I, O> {
    /// Create new builder
    pub fn new() -> Self
    
    /// Set DSPy module
    pub fn with_module(mut self, module: Arc<dyn Module<Input = I, Output = O>>) -> Self
    
    /// Set tool name
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self
    
    /// Set tool description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self
    
    /// Set input schema
    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self
    
    /// Build the DSPy tool
    pub fn build(self) -> DspyResult<DspyModuleTool<I, O>>
}
```

This API reference provides comprehensive documentation for all public interfaces in the DSPy framework. For more detailed examples and usage patterns, see the [Examples](../examples/) directory.
