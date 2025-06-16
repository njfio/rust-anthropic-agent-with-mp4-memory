# DSPy Framework Documentation

Welcome to the DSPy (Declarative Self-improving Python) framework documentation. DSPy is a powerful framework for building, optimizing, and deploying language model applications with type safety, composability, and automatic optimization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Security](#security)

## Quick Start

### Installation

Add DSPy to your Rust project:

```toml
[dependencies]
rust_memvid_agent = { path = ".", features = ["dspy"] }
```

### Basic Usage

```rust
use rust_memvid_agent::dspy::*;
use rust_memvid_agent::anthropic::AnthropicClient;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create Anthropic client
    let client = Arc::new(AnthropicClient::new("your-api-key".to_string(), None)?);
    
    // Define a signature for question answering
    let signature = Signature::<String, String>::new("question_answering".to_string())
        .with_description("Answer questions based on context");
    
    // Create a Predict module
    let qa_module = Predict::new(signature, client);
    
    // Use the module
    let question = "What is the capital of France?".to_string();
    let answer = qa_module.forward(question).await?;
    
    println!("Answer: {}", answer);
    Ok(())
}
```

## Core Concepts

### 1. Signatures

Signatures define the input-output interface of your modules:

```rust
// Simple string-to-string signature
let signature = Signature::<String, String>::new("summarization".to_string())
    .with_description("Summarize the given text");

// Complex structured signature
#[derive(Serialize, Deserialize)]
struct QuestionInput {
    context: String,
    question: String,
}

#[derive(Serialize, Deserialize)]
struct AnswerOutput {
    answer: String,
    confidence: f64,
}

let signature = Signature::<QuestionInput, AnswerOutput>::new("qa_with_confidence".to_string())
    .with_description("Answer questions with confidence scores");
```

### 2. Modules

Modules are the building blocks of DSPy applications:

#### Predict Module
The basic module for language model predictions:

```rust
let predict = Predict::new(signature, anthropic_client);
let result = predict.forward(input).await?;
```

#### Chain Module
For sequential processing:

```rust
let chain = Chain::new()
    .add_module(summarizer)
    .add_module(classifier);
let result = chain.forward(input).await?;
```

#### Parallel Module
For concurrent processing:

```rust
let parallel = Parallel::new()
    .add_module("sentiment", sentiment_analyzer)
    .add_module("topic", topic_classifier);
let results = parallel.forward(input).await?;
```

### 3. Specialized Modules

#### Chain of Thought
For step-by-step reasoning:

```rust
let cot = ChainOfThought::new(signature, client)
    .with_config(ChainOfThoughtConfig {
        max_steps: 5,
        include_step_numbers: true,
        validate_chain: true,
        ..Default::default()
    });
```

#### ReAct (Reasoning and Acting)
For tool-using applications:

```rust
let mut react = ReAct::new(signature, client);
react.add_tool(search_tool);
react.add_tool(calculator_tool);
let result = react.forward(input).await?;
```

#### RAG (Retrieval-Augmented Generation)
For knowledge-grounded responses:

```rust
let rag = RAG::new(signature, client, memory_manager)
    .with_config(RAGConfig {
        num_documents: 5,
        min_relevance_score: 0.7,
        enable_reranking: true,
        ..Default::default()
    });
```

#### Program of Thought
For computational reasoning:

```rust
let pot = ProgramOfThought::new(signature, client)
    .with_config(ProgramOfThoughtConfig {
        language: ProgrammingLanguage::Python,
        execute_code: true,
        max_code_length: 1000,
        ..Default::default()
    });
```

#### Self-Improving
For adaptive learning:

```rust
let self_improving = SelfImproving::new(signature, client, base_module)
    .with_config(SelfImprovingConfig {
        auto_improve: true,
        improvement_threshold: 0.3,
        max_improvement_iterations: 10,
        ..Default::default()
    });

// Add feedback for learning
self_improving.add_feedback(
    input,
    output,
    Some(expected_output),
    0.8, // score
    FeedbackType::Explicit,
).await?;
```

### 4. Optimization and Compilation

#### Basic Optimization
```rust
// Create examples
let mut examples = ExampleSet::new();
examples.add_example(Example::new(input1, output1));
examples.add_example(Example::new(input2, output2));

// Create teleprompter
let teleprompter = Teleprompter::new(OptimizationStrategy::bootstrap(5, 10));

// Optimize module
let result = teleprompter.optimize(&mut module, examples).await?;
println!("Optimization score: {:.3}", result.metrics.best_score);
```

#### Advanced Optimization
```rust
// MIPROv2 optimization
let strategy = OptimizationStrategy::mipro_v2(MIPROv2Config {
    num_bootstrap: 100,
    num_candidates: 50,
    max_iterations: 20,
    confidence_threshold: 0.9,
    ..Default::default()
});

// Multi-objective optimization
let strategy = OptimizationStrategy::multi_objective(MultiObjectiveConfig {
    objective_weights: vec![0.6, 0.4], // accuracy, efficiency
    use_pareto_optimization: true,
    max_pareto_points: 50,
    ..Default::default()
});
```

### 5. Evaluation

#### Basic Metrics
```rust
// Exact match
let exact_match = ExactMatch::new();
let score = exact_match.evaluate(&predicted, &expected).await?;

// Semantic similarity
let semantic_sim = SemanticSimilarity::new(client.clone());
let score = semantic_sim.evaluate(&predicted, &expected).await?;

// F1 Score
let f1 = F1Score::new();
let score = f1.evaluate(&predicted_tokens, &expected_tokens).await?;
```

#### Composite Metrics
```rust
let composite = CompositeMetric::new()
    .add_metric("accuracy", Box::new(ExactMatch::new()), 0.5)
    .add_metric("similarity", Box::new(SemanticSimilarity::new(client)), 0.3)
    .add_metric("f1", Box::new(F1Score::new()), 0.2);

let score = composite.evaluate(&predicted, &expected).await?;
```

#### Evaluation Framework
```rust
let evaluator = Evaluator::new()
    .add_metric("accuracy", Box::new(ExactMatch::new()))
    .add_metric("similarity", Box::new(SemanticSimilarity::new(client)));

let results = evaluator.evaluate(&module, &test_examples).await?;
println!("Results: {:#?}", results);
```

### 6. Agent Integration

#### Enable DSPy in Agent
```rust
let mut agent = Agent::new(config, client, memory_manager, tool_orchestrator)?;
agent.enable_dspy_integration(Some(DspyAgentConfig::default()))?;
```

#### Create and Use DSPy Modules
```rust
// Create a module
let signature = Signature::<String, String>::new("summarization".to_string());
let module = agent.as_dspy_module(signature, None).await?;

// Use the module
let result = agent.use_dspy_module(&*module, input).await?;

// Optimize the module
let metrics = agent.optimize_dspy_module(&mut *module, examples, None).await?;
```

### 7. Tool Integration

#### DSPy Module as Tool
```rust
let dspy_tool = DspyModuleTool::from_module(
    Arc::new(module),
    Some("summarizer".to_string()),
    Some("Summarizes text input".to_string()),
)?;

// Register with tool orchestrator
tool_orchestrator.register_tool(Box::new(dspy_tool))?;
```

#### Tool Registry
```rust
let mut registry = DspyToolRegistry::new();
registry.register_dspy_tool(dspy_tool, "module_123".to_string())?;

let stats = registry.stats();
println!("Registry stats: {:#?}", stats);
```

## Performance Optimization

### Caching
```rust
// Enable caching
let cache_config = CacheConfig {
    enabled: true,
    ttl_seconds: 3600,
    max_entries: 1000,
    ..Default::default()
};

let mut module = Predict::new(signature, client);
module.enable_caching(cache_config);
```

### Batching
```rust
// Process multiple inputs efficiently
let inputs = vec![input1, input2, input3];
let results = module.forward_batch(inputs).await?;
```

### Memory Management
```rust
// Configure memory limits
let config = PredictConfig {
    max_memory_mb: 512,
    enable_memory_monitoring: true,
    ..Default::default()
};
```

## Error Handling

DSPy uses a comprehensive error system:

```rust
use rust_memvid_agent::dspy::error::{DspyError, DspyResult};

match module.forward(input).await {
    Ok(result) => println!("Success: {}", result),
    Err(DspyError::Module { module_name, message }) => {
        eprintln!("Module error in {}: {}", module_name, message);
    }
    Err(DspyError::Optimization { strategy, message }) => {
        eprintln!("Optimization error with {}: {}", strategy, message);
    }
    Err(DspyError::Configuration { parameter, message }) => {
        eprintln!("Configuration error for {}: {}", parameter, message);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Security Best Practices

### 1. API Key Management
```rust
// Use environment variables
let api_key = std::env::var("ANTHROPIC_API_KEY")
    .expect("ANTHROPIC_API_KEY must be set");

// Or use secure configuration
let client = AnthropicClient::new(api_key, Some(client_config))?;
```

### 2. Input Validation
```rust
// Always validate inputs
impl Module for MyModule {
    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        if input.is_empty() {
            return Err(DspyError::module(self.name(), "Input cannot be empty"));
        }
        Ok(())
    }
}
```

### 3. Security Context
```rust
// Use security context for operations
let security_context = SecurityContext {
    user_id: "user123".to_string(),
    session_id: "session456".to_string(),
    roles: vec!["user".to_string()],
    permissions: vec!["dspy:execute".to_string()],
    // ...
};

agent.set_security_context(Some(security_context));
```

### 4. Code Execution Security
```rust
// Configure secure code execution
let pot_config = ProgramOfThoughtConfig {
    execute_code: false, // Disable by default
    security_restrictions: SecurityRestrictions {
        disallow_file_operations: true,
        disallow_network_operations: true,
        disallow_subprocess: true,
        max_memory_mb: 100,
        blacklisted_functions: vec!["exec".to_string(), "eval".to_string()],
    },
    ..Default::default()
};
```

## Next Steps

- Check out the [Examples](examples/) directory for complete working examples
- Read the [API Reference](api/) for detailed documentation
- See [Best Practices](best_practices.md) for advanced usage patterns
- Review [Troubleshooting](troubleshooting.md) for common issues

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the DSPy framework.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
