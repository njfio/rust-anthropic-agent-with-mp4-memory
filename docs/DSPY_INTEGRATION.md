# DSPy Integration for Rust MemVid Agent

## Overview

[DSPy](https://github.com/stanfordnlp/dspy) is a framework for programming foundation models using techniques like prompt optimization, self-improvement, and reasoning. This document outlines how to integrate DSPy's core concepts into the Rust MemVid Agent framework.

## Core DSPy Concepts

DSPy introduces several powerful concepts:

1. **Modules**: Composable units that define specific LLM tasks
2. **Signatures**: Type definitions for inputs and outputs
3. **Teleprompters**: Automatic prompt optimization
4. **Tracing and Compilation**: Runtime optimization of prompts
5. **Metrics and Evaluation**: Systematic evaluation of LLM outputs

## Implementation Strategy

### 1. Rust Module Structure

```rust
rust_memvid_agent/
├── src/
│   ├── dspy/
│   │   ├── mod.rs                 // Module exports
│   │   ├── signature.rs           // Input/output type definitions
│   │   ├── module.rs              // Base module trait and implementations
│   │   ├── teleprompter.rs        // Prompt optimization
│   │   ├── compiler.rs            // Runtime optimization
│   │   ├── metrics.rs             // Evaluation metrics
│   │   ├── predictor.rs           // Core prediction modules
│   │   └── chain.rs               // Module composition
```

### 2. Core Components

#### Signatures

```rust
/// Defines the input and output types for a DSPy module
pub struct Signature<I, O> {
    pub input_fields: Vec<Field>,
    pub output_fields: Vec<Field>,
    _input_type: PhantomData<I>,
    _output_type: PhantomData<O>,
}

/// Field definition with name, description, and type
pub struct Field {
    pub name: String,
    pub description: String,
    pub field_type: FieldType,
}

/// Supported field types
pub enum FieldType {
    String,
    Number,
    Boolean,
    Array(Box<FieldType>),
    Object(Vec<Field>),
}
```

#### Modules

```rust
/// Base trait for all DSPy modules
#[async_trait]
pub trait Module {
    type Input;
    type Output;
    
    /// Get the signature of this module
    fn signature(&self) -> Signature<Self::Input, Self::Output>;
    
    /// Forward pass through the module
    async fn forward(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Compile the module with a teleprompter
    async fn compile(&mut self, teleprompter: &Teleprompter, examples: Vec<(Self::Input, Self::Output)>) -> Result<()>;
}
```

#### Predictor Modules

```rust
/// LM-based prediction module
pub struct Predict<I, O> {
    signature: Signature<I, O>,
    prompt_template: String,
    anthropic_client: AnthropicClient,
    compiled_prompt: Option<String>,
}

#[async_trait]
impl<I, O> Module for Predict<I, O> 
where
    I: Serialize + DeserializeOwned + Send + Sync,
    O: Serialize + DeserializeOwned + Send + Sync,
{
    type Input = I;
    type Output = O;
    
    fn signature(&self) -> Signature<I, O> {
        self.signature.clone()
    }
    
    async fn forward(&self, input: I) -> Result<O> {
        // Format prompt using template and input
        // Call Anthropic API
        // Parse response into output type
    }
    
    async fn compile(&mut self, teleprompter: &Teleprompter, examples: Vec<(I, O)>) -> Result<()> {
        // Use teleprompter to optimize the prompt template
        self.compiled_prompt = Some(teleprompter.optimize(self, examples).await?);
        Ok(())
    }
}
```

### 3. Module Composition

```rust
/// Chain multiple modules together
pub struct Chain<I, O> {
    modules: Vec<Box<dyn Module<Output = O>>>,
    _input_type: PhantomData<I>,
}

impl<I, O> Chain<I, O> {
    pub fn new(modules: Vec<Box<dyn Module<Output = O>>>) -> Self {
        Self {
            modules,
            _input_type: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O> Module for Chain<I, O>
where
    I: Serialize + DeserializeOwned + Send + Sync,
    O: Serialize + DeserializeOwned + Send + Sync,
{
    type Input = I;
    type Output = O;
    
    async fn forward(&self, input: I) -> Result<O> {
        // Pass input through each module in sequence
        // Return final output
    }
}
```

### 4. Teleprompters

```rust
/// Optimizes prompts based on examples
pub struct Teleprompter {
    anthropic_client: AnthropicClient,
    optimization_strategy: OptimizationStrategy,
}

impl Teleprompter {
    pub async fn optimize<M: Module>(&self, module: &M, examples: Vec<(M::Input, M::Output)>) -> Result<String> {
        // Use examples to optimize the prompt template
        // Apply optimization strategy
        // Return optimized prompt
    }
}

pub enum OptimizationStrategy {
    BootstrapOfThoughts,
    FewShotExamples,
    ChainOfThought,
    ReAct,
    Custom(Box<dyn Fn(&str, &Vec<(String, String)>) -> String + Send + Sync>),
}
```

### 5. Metrics and Evaluation

```rust
/// Evaluates module outputs against expected results
pub struct Evaluator<I, O> {
    metrics: Vec<Box<dyn Metric<I, O>>>,
}

impl<I, O> Evaluator<I, O> {
    pub async fn evaluate(&self, module: &impl Module<Input = I, Output = O>, examples: Vec<(I, O)>) -> Result<EvaluationResults> {
        // Run module on examples
        // Apply metrics
        // Return results
    }
}

#[async_trait]
pub trait Metric<I, O> {
    async fn compute(&self, input: &I, predicted: &O, expected: &O) -> f64;
    fn name(&self) -> &str;
}
```

## Integration with Agent System

### 1. Agent Extension

```rust
impl Agent {
    /// Create a DSPy module from this agent
    pub fn as_dspy_module<I, O>(&self, signature: Signature<I, O>) -> Predict<I, O>
    where
        I: Serialize + DeserializeOwned + Send + Sync,
        O: Serialize + DeserializeOwned + Send + Sync,
    {
        Predict::new(signature, self.anthropic_client.clone())
    }
    
    /// Use a compiled DSPy module for a specific task
    pub async fn use_dspy_module<I, O>(&self, module: &impl Module<Input = I, Output = O>, input: I) -> Result<O>
    where
        I: Serialize + DeserializeOwned + Send + Sync,
        O: Serialize + DeserializeOwned + Send + Sync,
    {
        module.forward(input).await
    }
}
```

### 2. Example Usage

```rust
// Define a signature for a task
let signature = Signature::<QuestionInput, ReasonedAnswer>::new()
    .with_input_field("question", "The question to answer")
    .with_output_field("reasoning", "Step-by-step reasoning")
    .with_output_field("answer", "The final answer");

// Create a module
let mut qa_module = Predict::new(signature, agent.anthropic_client.clone());

// Compile with examples
let examples = vec![
    (
        QuestionInput { question: "What is the capital of France?".to_string() },
        ReasonedAnswer {
            reasoning: "France is a country in Europe. Its capital city is Paris.".to_string(),
            answer: "Paris".to_string(),
        }
    ),
    // More examples...
];

let teleprompter = Teleprompter::new(agent.anthropic_client.clone(), OptimizationStrategy::ChainOfThought);
qa_module.compile(&teleprompter, examples).await?;

// Use the module
let result = agent.use_dspy_module(&qa_module, QuestionInput {
    question: "What is the capital of Italy?".to_string(),
}).await?;

println!("Reasoning: {}", result.reasoning);
println!("Answer: {}", result.answer);
```

## Advanced Features

### 1. Multi-step Reasoning

```rust
// Define a chain of thought module
let cot_module = ChainOfThought::new(
    agent.as_dspy_module(question_signature),
    agent.as_dspy_module(reasoning_signature),
    agent.as_dspy_module(answer_signature),
);

// Compile with examples
cot_module.compile(&teleprompter, examples).await?;

// Use for complex reasoning
let result = agent.use_dspy_module(&cot_module, complex_question).await?;
```

### 2. Self-improvement

```rust
// Create a self-improving module
let mut self_improving_module = SelfImproving::new(
    agent.as_dspy_module(base_signature),
    agent.as_dspy_module(critique_signature),
    agent.as_dspy_module(improvement_signature),
);

// Train through iterations
for _ in 0..5 {
    self_improving_module.improve_on(training_examples).await?;
}

// Use the improved module
let result = agent.use_dspy_module(&self_improving_module, input).await?;
```

### 3. Retrieval-Augmented Generation

```rust
// Create a RAG module
let rag_module = RAG::new(
    agent.memory_manager.clone(),
    agent.as_dspy_module(query_signature),
    agent.as_dspy_module(generation_signature),
);

// Use for knowledge-intensive tasks
let result = agent.use_dspy_module(&rag_module, question).await?;
```

## Implementation Roadmap

1. **Phase 1: Core Framework**
   - Implement `Signature` and basic `Module` trait
   - Create `Predict` module that works with Anthropic API
   - Develop simple module composition with `Chain`

2. **Phase 2: Optimization**
   - Implement `Teleprompter` with basic strategies
   - Add compilation support to modules
   - Create evaluation metrics

3. **Phase 3: Advanced Modules**
   - Implement specialized modules (ChainOfThought, RAG, etc.)
   - Add self-improvement capabilities
   - Develop complex reasoning patterns

4. **Phase 4: Integration**
   - Integrate with existing Agent memory system
   - Add tool support within DSPy modules
   - Create high-level API for agent developers

## Conclusion

By integrating DSPy concepts into the Rust MemVid Agent, we can create a powerful framework for prompt engineering, optimization, and evaluation. This approach maintains the performance benefits of Rust while leveraging the advanced LLM programming techniques from DSPy.

The implementation will enable:
- Systematic prompt engineering
- Automatic prompt optimization
- Complex reasoning patterns
- Rigorous evaluation of LLM outputs
- Composition of specialized modules

This integration represents a significant enhancement to the agent's capabilities, particularly for complex reasoning tasks and specialized domain applications.