//! Advanced DSPy Usage Examples
//!
//! This file demonstrates advanced DSPy concepts including composition,
//! optimization, specialized modules, and agent integration.

use rust_memvid_agent::anthropic::AnthropicClient;
use rust_memvid_agent::dspy::*;
use rust_memvid_agent::agent::{Agent, DspyAgentConfig};
use rust_memvid_agent::memory::MemoryManager;
use rust_memvid_agent::tools::{Tool, ToolResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use async_trait::async_trait;

/// Example 1: Module Composition with Chain
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocumentInput {
    text: String,
    max_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SummaryOutput {
    summary: String,
    key_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationOutput {
    category: String,
    confidence: f64,
}

async fn example_chain_composition() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create summarization module
    let summary_sig = Signature::<DocumentInput, SummaryOutput>::new("summarization".to_string())
        .with_description("Summarize documents with key points");
    let summarizer = Predict::new(summary_sig, client.clone());

    // Create classification module
    let class_sig = Signature::<SummaryOutput, ClassificationOutput>::new("classification".to_string())
        .with_description("Classify document summaries");
    let classifier = Predict::new(class_sig, client);

    // Create chain
    let chain = Chain::new()
        .add_module(summarizer)
        .add_module(classifier);

    let input = DocumentInput {
        text: "Artificial intelligence is transforming healthcare through machine learning algorithms that can diagnose diseases, predict patient outcomes, and personalize treatment plans. Recent advances in deep learning have enabled AI systems to analyze medical images with accuracy comparable to human radiologists.".to_string(),
        max_length: 100,
    };

    let result = chain.forward(input).await?;

    println!("Chain Composition Result:");
    println!("Category: {}", result.category);
    println!("Confidence: {:.2}", result.confidence);

    Ok(())
}

/// Example 2: Parallel Processing
async fn example_parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create multiple analysis modules
    let sentiment_sig = Signature::<String, String>::new("sentiment".to_string())
        .with_description("Analyze sentiment");
    let sentiment_module = Predict::new(sentiment_sig, client.clone());

    let topic_sig = Signature::<String, String>::new("topic".to_string())
        .with_description("Extract main topic");
    let topic_module = Predict::new(topic_sig, client.clone());

    let language_sig = Signature::<String, String>::new("language".to_string())
        .with_description("Detect language");
    let language_module = Predict::new(language_sig, client);

    // Create parallel module
    let parallel = Parallel::new()
        .add_module("sentiment", sentiment_module)
        .add_module("topic", topic_module)
        .add_module("language", language_module)
        .with_strategy(ParallelStrategy::WaitAll);

    let text = "I absolutely love this new technology! It's revolutionizing how we work.".to_string();
    let results = parallel.forward(text).await?;

    println!("Parallel Processing Results:");
    for (name, result) in results {
        println!("{}: {}", name, result);
    }

    Ok(())
}

/// Example 3: Chain of Thought Reasoning
async fn example_chain_of_thought() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("math_reasoning".to_string())
        .with_description("Solve math problems with step-by-step reasoning");

    let config = ChainOfThoughtConfig {
        base: SpecializedModuleConfig {
            max_steps: 5,
            temperature: 0.1,
            verbose: true,
            ..Default::default()
        },
        include_step_numbers: true,
        validate_chain: true,
        min_confidence: 0.8,
        max_retries: 2,
        ..Default::default()
    };

    let cot_module = ChainOfThought::with_config(signature, client, config);

    let problem = "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?".to_string();

    let answer = cot_module.forward(problem).await?;

    println!("Chain of Thought Result:");
    println!("Answer: {}", answer);

    // Get reasoning steps
    let steps = cot_module.get_reasoning_steps();
    println!("Reasoning steps: {} steps taken", steps.len());

    Ok(())
}

/// Example 4: ReAct with Tools
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn definition(&self) -> rust_memvid_agent::anthropic::models::ToolDefinition {
        rust_memvid_agent::anthropic::models::ToolDefinition {
            tool_type: "function".to_string(),
            name: "calculator".to_string(),
            description: Some("Perform basic arithmetic calculations".to_string()),
            input_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            })),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    async fn execute(&self, input: serde_json::Value) -> rust_memvid_agent::utils::error::Result<ToolResult> {
        let expression = input.get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| rust_memvid_agent::utils::error::AgentError::invalid_input("Missing expression"))?;

        // Simple calculator (in practice, use a proper math parser)
        let result = match expression {
            "2 + 2" => "4",
            "10 * 5" => "50",
            "100 / 4" => "25",
            _ => "Cannot compute",
        };

        Ok(ToolResult::success(result.to_string()))
    }

    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> Option<&str> {
        Some("Perform basic arithmetic calculations")
    }

    fn validate_input(&self, input: &serde_json::Value) -> rust_memvid_agent::utils::error::Result<()> {
        if !input.get("expression").is_some() {
            return Err(rust_memvid_agent::utils::error::AgentError::invalid_input("Missing expression"));
        }
        Ok(())
    }
}

async fn example_react_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("math_solver".to_string())
        .with_description("Solve math problems using available tools");

    let config = ReActConfig {
        base: SpecializedModuleConfig {
            max_steps: 3,
            temperature: 0.1,
            ..Default::default()
        },
        max_cycles: 5,
        validate_actions: true,
        tool_timeout_seconds: 10,
        continue_on_tool_error: false,
        ..Default::default()
    };

    let mut react_module = ReAct::with_config(signature, client, config);
    react_module.add_tool(CalculatorTool);

    let problem = "What is 2 + 2 multiplied by 5?".to_string();
    let answer = react_module.forward(problem).await?;

    println!("ReAct with Tools Result:");
    println!("Answer: {}", answer);

    Ok(())
}

/// Example 5: RAG (Retrieval-Augmented Generation)
async fn example_rag() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create memory manager
    let memory_manager = Arc::new(Mutex::new(MemoryManager::new().await?));

    // Add some documents to memory
    {
        let mut mm = memory_manager.lock().await;
        mm.store_memory(
            "doc1".to_string(),
            "Paris is the capital of France and its largest city.".to_string(),
            HashMap::new(),
        ).await?;
        mm.store_memory(
            "doc2".to_string(),
            "London is the capital of the United Kingdom.".to_string(),
            HashMap::new(),
        ).await?;
        mm.store_memory(
            "doc3".to_string(),
            "Tokyo is the capital of Japan and one of the world's most populous cities.".to_string(),
            HashMap::new(),
        ).await?;
    }

    let signature = Signature::<String, String>::new("knowledge_qa".to_string())
        .with_description("Answer questions using retrieved knowledge");

    let config = RAGConfig {
        base: SpecializedModuleConfig::default(),
        num_documents: 3,
        min_relevance_score: 0.1,
        max_context_length: 1000,
        enable_reranking: true,
        enable_query_expansion: false,
        retrieval_strategy: RetrievalStrategy::Semantic,
    };

    let rag_module = RAG::with_config(signature, client, memory_manager, config);

    let question = "What is the capital of France?".to_string();
    let answer = rag_module.forward(question).await?;

    println!("RAG Result:");
    println!("Answer: {}", answer);

    // Get RAG details
    if let Some(rag_result) = rag_module.get_last_rag_result().await {
        println!("Retrieved {} documents", rag_result.retrieved_documents.len());
        println!("Retrieval confidence: {:.2}", rag_result.retrieval_confidence);
        println!("Generation confidence: {:.2}", rag_result.generation_confidence);
    }

    Ok(())
}

/// Example 6: Module Optimization
async fn example_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("sentiment_classifier".to_string())
        .with_description("Classify sentiment as positive, negative, or neutral");

    let mut module = Predict::new(signature, client);

    // Create training examples
    let mut examples = ExampleSet::new();
    examples.add_example(Example::new("I love this!".to_string(), "positive".to_string()));
    examples.add_example(Example::new("This is terrible.".to_string(), "negative".to_string()));
    examples.add_example(Example::new("It's okay.".to_string(), "neutral".to_string()));
    examples.add_example(Example::new("Amazing product!".to_string(), "positive".to_string()));
    examples.add_example(Example::new("Worst experience ever.".to_string(), "negative".to_string()));

    // Create teleprompter with bootstrap strategy
    let strategy = OptimizationStrategy::bootstrap(3, 5);
    let mut teleprompter = Teleprompter::new(strategy);

    // Optimize the module
    println!("Starting optimization...");
    let result = teleprompter.optimize(&mut module, examples).await?;

    println!("Optimization Results:");
    println!("Best score: {:.3}", result.metrics.best_score);
    println!("Iterations: {}", result.metrics.iterations);
    println!("Total examples: {}", result.metrics.total_examples);

    // Test optimized module
    let test_input = "This product exceeded my expectations!".to_string();
    let prediction = module.forward(test_input).await?;
    println!("Optimized prediction: {}", prediction);

    Ok(())
}

/// Example 7: Agent Integration
async fn example_agent_integration() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create agent components
    let memory_manager = MemoryManager::new().await?;
    let tool_orchestrator = rust_memvid_agent::agent::ToolOrchestrator::new();
    let conversation_manager = rust_memvid_agent::agent::ConversationManager::new(memory_manager.clone());

    // Create agent
    let config = rust_memvid_agent::config::AgentConfig::default();
    let mut agent = Agent::new(
        config,
        client,
        memory_manager,
        tool_orchestrator,
        conversation_manager,
        None,
    )?;

    // Enable DSPy integration
    let dspy_config = DspyAgentConfig {
        auto_optimize: true,
        max_modules: 50,
        enable_security_validation: false, // Simplified for example
        enable_audit_logging: true,
        ..Default::default()
    };
    agent.enable_dspy_integration(Some(dspy_config))?;

    // Create DSPy module through agent
    let signature = Signature::<String, String>::new("agent_qa".to_string())
        .with_description("Answer questions through agent");

    let module = agent.as_dspy_module(signature, None).await?;

    // Use module through agent
    let question = "What is machine learning?".to_string();
    let answer = agent.use_dspy_module(&*module, question).await?;

    println!("Agent DSPy Integration Result:");
    println!("Answer: {}", answer);

    // Get registry stats
    let stats = agent.get_dspy_registry_stats().await?;
    println!("Registry stats: {:#?}", stats);

    Ok(())
}

/// Example 8: Self-Improving Module
async fn example_self_improving() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create base module
    let base_signature = Signature::<String, String>::new("base_classifier".to_string())
        .with_description("Classify text");
    let base_module = Arc::new(RwLock::new(Predict::new(base_signature, client.clone())));

    // Create self-improving wrapper
    let signature = Signature::<String, String>::new("self_improving_classifier".to_string())
        .with_description("Self-improving text classifier");

    let config = SelfImprovingConfig {
        min_examples_for_improvement: 3,
        improvement_threshold: 0.5,
        max_improvement_iterations: 3,
        auto_improve: true,
        improvement_strategy: ImprovementStrategy::GradualOptimization,
        ..Default::default()
    };

    let self_improving = SelfImproving::with_config(signature, client, base_module, config);

    // Simulate usage with feedback
    let inputs_outputs = vec![
        ("Great product!".to_string(), "positive".to_string(), Some("positive".to_string()), 1.0),
        ("Terrible service.".to_string(), "negative".to_string(), Some("negative".to_string()), 1.0),
        ("It's okay.".to_string(), "positive".to_string(), Some("neutral".to_string()), -0.5), // Wrong prediction
        ("Amazing!".to_string(), "neutral".to_string(), Some("positive".to_string()), -0.8), // Wrong prediction
    ];

    for (input, output, expected, score) in inputs_outputs {
        // Use module
        let result = self_improving.forward(input.clone()).await?;
        println!("Prediction: {} -> {}", input, result);

        // Add feedback
        self_improving.add_feedback(
            input,
            output,
            expected,
            score,
            FeedbackType::Explicit,
        ).await?;
    }

    // Check improvement metrics
    let metrics = self_improving.get_improvement_metrics().await;
    println!("Self-Improvement Metrics:");
    println!("Improvement iterations: {}", metrics.improvement_iterations);
    println!("Current performance: {:.3}", metrics.current_performance);
    println!("Improvement percentage: {:.1}%", metrics.improvement_percentage);

    Ok(())
}

/// Run all advanced examples
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DSPy Advanced Usage Examples ===\n");

    println!("1. Chain Composition:");
    if let Err(e) = example_chain_composition().await {
        eprintln!("Error in chain composition: {}", e);
    }
    println!();

    println!("2. Parallel Processing:");
    if let Err(e) = example_parallel_processing().await {
        eprintln!("Error in parallel processing: {}", e);
    }
    println!();

    println!("3. Chain of Thought:");
    if let Err(e) = example_chain_of_thought().await {
        eprintln!("Error in chain of thought: {}", e);
    }
    println!();

    println!("4. ReAct with Tools:");
    if let Err(e) = example_react_with_tools().await {
        eprintln!("Error in ReAct: {}", e);
    }
    println!();

    println!("5. RAG (Retrieval-Augmented Generation):");
    if let Err(e) = example_rag().await {
        eprintln!("Error in RAG: {}", e);
    }
    println!();

    println!("6. Module Optimization:");
    if let Err(e) = example_optimization().await {
        eprintln!("Error in optimization: {}", e);
    }
    println!();

    println!("7. Agent Integration:");
    if let Err(e) = example_agent_integration().await {
        eprintln!("Error in agent integration: {}", e);
    }
    println!();

    println!("8. Self-Improving Module:");
    if let Err(e) = example_self_improving().await {
        eprintln!("Error in self-improving: {}", e);
    }

    println!("\n=== All advanced examples completed ===");
    Ok(())
}
