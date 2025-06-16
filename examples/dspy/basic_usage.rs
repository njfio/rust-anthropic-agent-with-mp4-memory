//! Basic DSPy Usage Examples
//!
//! This file demonstrates fundamental DSPy concepts and usage patterns.

use rust_memvid_agent::anthropic::AnthropicClient;
use rust_memvid_agent::dspy::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use async_trait::async_trait;

/// Example 1: Simple Question Answering
#[tokio::main]
async fn example_simple_qa() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Anthropic client
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create a signature for question answering
    let signature = Signature::<String, String>::new("question_answering".to_string())
        .with_description("Answer questions accurately and concisely");

    // Create a Predict module
    let qa_module = Predict::new(signature, client);

    // Use the module
    let question = "What is the capital of France?".to_string();
    let answer = qa_module.forward(question).await?;

    println!("Question: What is the capital of France?");
    println!("Answer: {}", answer);

    Ok(())
}

/// Example 2: Structured Input/Output
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionInput {
    context: String,
    question: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnswerOutput {
    answer: String,
    confidence: f64,
    reasoning: String,
}

async fn example_structured_qa() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create signature with structured types
    let signature = Signature::<QuestionInput, AnswerOutput>::new("structured_qa".to_string())
        .with_description("Answer questions based on context with confidence and reasoning");

    let qa_module = Predict::new(signature, client);

    let input = QuestionInput {
        context: "Paris is the capital and most populous city of France. It is located in the north-central part of the country.".to_string(),
        question: "What is the capital of France?".to_string(),
    };

    let output = qa_module.forward(input).await?;

    println!("Structured QA Result:");
    println!("Answer: {}", output.answer);
    println!("Confidence: {:.2}", output.confidence);
    println!("Reasoning: {}", output.reasoning);

    Ok(())
}

/// Example 3: Module Configuration
async fn example_module_configuration() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("creative_writing".to_string())
        .with_description("Generate creative content");

    // Configure the module for creative tasks
    let config = PredictConfig {
        temperature: 0.9,
        max_tokens: 500,
        top_p: 0.95,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        enable_caching: true,
        cache_ttl_seconds: 3600,
        ..Default::default()
    };

    let creative_module = Predict::with_config(signature, client, config);

    let prompt = "Write a short story about a robot learning to paint.".to_string();
    let story = creative_module.forward(prompt).await?;

    println!("Generated Story:");
    println!("{}", story);

    Ok(())
}

/// Example 4: Error Handling
async fn example_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("test_module".to_string());
    let module = Predict::new(signature, client);

    // Demonstrate error handling
    let result = module.forward("".to_string()).await;

    match result {
        Ok(output) => println!("Success: {}", output),
        Err(DspyError::Module { module_name, message }) => {
            eprintln!("Module error in {}: {}", module_name, message);
        }
        Err(DspyError::Anthropic(e)) => {
            eprintln!("Anthropic API error: {}", e);
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }

    Ok(())
}

/// Example 5: Batch Processing
async fn example_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("sentiment_analysis".to_string())
        .with_description("Analyze sentiment of text (positive, negative, neutral)");

    let sentiment_module = Predict::new(signature, client);

    let texts = vec![
        "I love this product!".to_string(),
        "This is terrible.".to_string(),
        "It's okay, nothing special.".to_string(),
        "Amazing quality and service!".to_string(),
    ];

    // Process batch
    let results = sentiment_module.forward_batch(texts.clone()).await?;

    println!("Sentiment Analysis Results:");
    for (text, sentiment) in texts.iter().zip(results.iter()) {
        println!("Text: \"{}\" -> Sentiment: {}", text, sentiment);
    }

    Ok(())
}

/// Example 6: Module Validation
struct ValidatedModule {
    inner: Predict<String, String>,
}

impl ValidatedModule {
    fn new(signature: Signature<String, String>, client: Arc<AnthropicClient>) -> Self {
        Self {
            inner: Predict::new(signature, client),
        }
    }
}

#[async_trait::async_trait]
impl Module for ValidatedModule {
    type Input = String;
    type Output = String;

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        self.inner.signature()
    }

    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        if input.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Input cannot be empty"));
        }
        if input.len() > 1000 {
            return Err(DspyError::module(self.name(), "Input too long (max 1000 characters)"));
        }
        Ok(())
    }

    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        if output.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Output cannot be empty"));
        }
        Ok(())
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        self.validate_input(&input).await?;
        let output = self.inner.forward(input).await?;
        self.validate_output(&output).await?;
        Ok(output)
    }

    fn metadata(&self) -> &ModuleMetadata {
        self.inner.metadata()
    }

    fn stats(&self) -> &ModuleStats {
        self.inner.stats()
    }

    fn supports_compilation(&self) -> bool {
        self.inner.supports_compilation()
    }
}

async fn example_module_validation() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("validated_qa".to_string())
        .with_description("Answer questions with input validation");

    let validated_module = ValidatedModule::new(signature, client);

    // Test with valid input
    match validated_module.forward("What is 2 + 2?".to_string()).await {
        Ok(answer) => println!("Valid input result: {}", answer),
        Err(e) => eprintln!("Error: {}", e),
    }

    // Test with invalid input (empty)
    match validated_module.forward("".to_string()).await {
        Ok(answer) => println!("Empty input result: {}", answer),
        Err(e) => eprintln!("Expected error for empty input: {}", e),
    }

    Ok(())
}

/// Example 7: Caching
async fn example_caching() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    let signature = Signature::<String, String>::new("cached_qa".to_string())
        .with_description("Question answering with caching");

    let mut qa_module = Predict::new(signature, client);

    // Enable caching
    let cache_config = CacheConfig {
        enabled: true,
        ttl_seconds: 3600, // 1 hour
        max_entries: 1000,
        ..Default::default()
    };
    qa_module.enable_caching(cache_config);

    let question = "What is the speed of light?".to_string();

    // First call - will hit the API
    let start = std::time::Instant::now();
    let answer1 = qa_module.forward(question.clone()).await?;
    let duration1 = start.elapsed();

    println!("First call (API): {} in {:?}", answer1, duration1);

    // Second call - should hit the cache
    let start = std::time::Instant::now();
    let answer2 = qa_module.forward(question.clone()).await?;
    let duration2 = start.elapsed();

    println!("Second call (cache): {} in {:?}", answer2, duration2);
    println!("Cache speedup: {:.2}x", duration1.as_millis() as f64 / duration2.as_millis() as f64);

    Ok(())
}

/// Example 8: Custom Signature Fields
async fn example_custom_signature() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Arc::new(AnthropicClient::new(api_key, None)?);

    // Create signature with custom fields
    let signature = Signature::<String, String>::new("custom_summarization".to_string())
        .with_description("Summarize text with specific requirements")
        .add_input_field(Field::new("text", FieldType::Text)
            .with_description("The text to summarize")
            .with_required(true))
        .add_output_field(Field::new("summary", FieldType::Text)
            .with_description("A concise summary")
            .with_required(true))
        .add_output_field(Field::new("key_points", FieldType::List)
            .with_description("Key points from the text")
            .with_required(false));

    let summarizer = Predict::new(signature, client);

    let long_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.".to_string();

    let summary = summarizer.forward(long_text).await?;

    println!("Custom Signature Summary:");
    println!("{}", summary);

    Ok(())
}

/// Run all examples
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DSPy Basic Usage Examples ===\n");

    println!("1. Simple Question Answering:");
    if let Err(e) = example_simple_qa().await {
        eprintln!("Error in simple QA: {}", e);
    }
    println!();

    println!("2. Structured Input/Output:");
    if let Err(e) = example_structured_qa().await {
        eprintln!("Error in structured QA: {}", e);
    }
    println!();

    println!("3. Module Configuration:");
    if let Err(e) = example_module_configuration().await {
        eprintln!("Error in module configuration: {}", e);
    }
    println!();

    println!("4. Error Handling:");
    if let Err(e) = example_error_handling().await {
        eprintln!("Error in error handling example: {}", e);
    }
    println!();

    println!("5. Batch Processing:");
    if let Err(e) = example_batch_processing().await {
        eprintln!("Error in batch processing: {}", e);
    }
    println!();

    println!("6. Module Validation:");
    if let Err(e) = example_module_validation().await {
        eprintln!("Error in module validation: {}", e);
    }
    println!();

    println!("7. Caching:");
    if let Err(e) = example_caching().await {
        eprintln!("Error in caching example: {}", e);
    }
    println!();

    println!("8. Custom Signature:");
    if let Err(e) = example_custom_signature().await {
        eprintln!("Error in custom signature: {}", e);
    }

    println!("\n=== All examples completed ===");
    Ok(())
}
