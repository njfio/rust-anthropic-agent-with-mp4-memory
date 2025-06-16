# DSPy Best Practices

This guide provides best practices for building robust, efficient, and maintainable DSPy applications.

## Table of Contents

1. [Module Design](#module-design)
2. [Error Handling](#error-handling)
3. [Performance Optimization](#performance-optimization)
4. [Security](#security)
5. [Testing](#testing)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Deployment](#deployment)

## Module Design

### 1. Use Strong Types

Always use strongly typed signatures for better type safety and documentation:

```rust
// Good: Strongly typed
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionInput {
    context: String,
    question: String,
    max_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnswerOutput {
    answer: String,
    confidence: f64,
    sources: Vec<String>,
}

let signature = Signature::<QuestionInput, AnswerOutput>::new("qa".to_string());

// Avoid: Weak typing
let signature = Signature::<String, String>::new("qa".to_string());
```

### 2. Implement Proper Validation

Always validate inputs and outputs:

```rust
#[async_trait]
impl Module for MyModule {
    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        if input.question.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Question cannot be empty"));
        }
        if input.question.len() > 1000 {
            return Err(DspyError::module(self.name(), "Question too long"));
        }
        Ok(())
    }

    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        if output.answer.trim().is_empty() {
            return Err(DspyError::module(self.name(), "Answer cannot be empty"));
        }
        if output.confidence < 0.0 || output.confidence > 1.0 {
            return Err(DspyError::module(self.name(), "Invalid confidence score"));
        }
        Ok(())
    }
}
```

### 3. Use Descriptive Signatures

Provide clear descriptions for your signatures:

```rust
let signature = Signature::<QuestionInput, AnswerOutput>::new("question_answering".to_string())
    .with_description("Answer questions based on provided context with confidence scores")
    .add_input_field(Field::new("context", FieldType::Text)
        .with_description("Background information for answering the question")
        .with_required(true))
    .add_input_field(Field::new("question", FieldType::Text)
        .with_description("The question to be answered")
        .with_required(true))
    .add_output_field(Field::new("answer", FieldType::Text)
        .with_description("The answer to the question")
        .with_required(true))
    .add_output_field(Field::new("confidence", FieldType::Number)
        .with_description("Confidence score between 0 and 1")
        .with_required(true));
```

### 4. Compose Modules Effectively

Use composition to build complex workflows:

```rust
// Create specialized modules
let summarizer = Predict::new(summary_signature, client.clone());
let classifier = Predict::new(classification_signature, client.clone());
let sentiment_analyzer = Predict::new(sentiment_signature, client);

// Compose them
let document_processor = Chain::new()
    .add_module(summarizer)
    .add_module(classifier);

let multi_analyzer = Parallel::new()
    .add_module("classification", document_processor)
    .add_module("sentiment", sentiment_analyzer)
    .with_strategy(ParallelStrategy::WaitAll);
```

## Error Handling

### 1. Use Comprehensive Error Handling

Handle all possible error cases:

```rust
async fn process_document(module: &dyn Module<Input = String, Output = String>, text: String) -> Result<String, Box<dyn std::error::Error>> {
    match module.forward(text).await {
        Ok(result) => Ok(result),
        Err(DspyError::Module { module_name, message }) => {
            eprintln!("Module {} failed: {}", module_name, message);
            Err(format!("Processing failed in {}", module_name).into())
        }
        Err(DspyError::Anthropic(e)) => {
            eprintln!("API error: {}", e);
            Err("External API error".into())
        }
        Err(DspyError::Configuration { parameter, message }) => {
            eprintln!("Configuration error for {}: {}", parameter, message);
            Err("Configuration error".into())
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            Err("Unexpected error".into())
        }
    }
}
```

### 2. Implement Retry Logic

Add retry logic for transient failures:

```rust
async fn robust_forward<I, O>(
    module: &dyn Module<Input = I, Output = O>,
    input: I,
    max_retries: usize,
) -> DspyResult<O>
where
    I: Clone,
{
    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match module.forward(input.clone()).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    let delay = std::time::Duration::from_millis(100 * (1 << attempt));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
    
    Err(last_error.unwrap())
}
```

### 3. Use Circuit Breaker Pattern

Implement circuit breakers for external dependencies:

```rust
struct CircuitBreakerModule<I, O> {
    inner: Box<dyn Module<Input = I, Output = O>>,
    failure_count: Arc<AtomicUsize>,
    last_failure: Arc<Mutex<Option<Instant>>>,
    failure_threshold: usize,
    recovery_timeout: Duration,
}

impl<I, O> CircuitBreakerModule<I, O> {
    async fn is_circuit_open(&self) -> bool {
        let failure_count = self.failure_count.load(Ordering::Relaxed);
        if failure_count >= self.failure_threshold {
            if let Some(last_failure) = *self.last_failure.lock().await {
                return last_failure.elapsed() < self.recovery_timeout;
            }
        }
        false
    }
}
```

## Performance Optimization

### 1. Enable Caching

Use caching for frequently accessed data:

```rust
let cache_config = CacheConfig {
    enabled: true,
    ttl_seconds: 3600, // 1 hour
    max_entries: 10000,
    eviction_policy: EvictionPolicy::LRU,
};

let mut module = Predict::new(signature, client);
module.enable_caching(cache_config);
```

### 2. Use Batch Processing

Process multiple inputs together when possible:

```rust
// Instead of individual calls
for input in inputs {
    let result = module.forward(input).await?;
    results.push(result);
}

// Use batch processing
let results = module.forward_batch(inputs).await?;
```

### 3. Optimize Module Configuration

Tune module parameters for your use case:

```rust
let config = PredictConfig {
    temperature: 0.1,        // Lower for deterministic tasks
    max_tokens: 100,         // Limit for shorter responses
    top_p: 0.9,             // Nucleus sampling
    timeout_seconds: 30,     // Reasonable timeout
    enable_streaming: true,  // For real-time applications
    ..Default::default()
};
```

### 4. Use Connection Pooling

Reuse client connections:

```rust
// Create a shared client
let client = Arc::new(AnthropicClient::new(api_key, Some(client_config))?);

// Share across modules
let module1 = Predict::new(signature1, client.clone());
let module2 = Predict::new(signature2, client.clone());
```

## Security

### 1. Validate All Inputs

Never trust user input:

```rust
fn validate_user_input(input: &str) -> Result<(), &'static str> {
    if input.len() > MAX_INPUT_LENGTH {
        return Err("Input too long");
    }
    
    if contains_malicious_patterns(input) {
        return Err("Input contains prohibited content");
    }
    
    Ok(())
}
```

### 2. Use Security Context

Always use security context for operations:

```rust
let security_context = SecurityContext {
    user_id: user.id.clone(),
    session_id: session.id.clone(),
    roles: user.roles.clone(),
    permissions: user.permissions.clone(),
    ip_address: Some(request.ip()),
    user_agent: request.headers().get("user-agent").map(|h| h.to_string()),
    timestamp: SystemTime::now(),
    metadata: HashMap::new(),
};

agent.set_security_context(Some(security_context));
```

### 3. Secure Code Execution

When using Program of Thought, ensure secure execution:

```rust
let pot_config = ProgramOfThoughtConfig {
    execute_code: false, // Disable by default
    security_restrictions: SecurityRestrictions {
        disallow_file_operations: true,
        disallow_network_operations: true,
        disallow_subprocess: true,
        max_memory_mb: 100,
        blacklisted_functions: vec![
            "exec".to_string(),
            "eval".to_string(),
            "open".to_string(),
            "__import__".to_string(),
        ],
    },
    execution_timeout_seconds: 10,
    max_code_length: 1000,
    ..Default::default()
};
```

### 4. Sanitize Outputs

Clean outputs before returning to users:

```rust
fn sanitize_output(output: &str) -> String {
    output
        .replace("<script>", "&lt;script&gt;")
        .replace("</script>", "&lt;/script&gt;")
        .trim()
        .to_string()
}
```

## Testing

### 1. Unit Test Modules

Test individual modules thoroughly:

```rust
#[tokio::test]
async fn test_qa_module() {
    let client = create_test_client();
    let signature = create_test_signature();
    let module = Predict::new(signature, client);
    
    let input = QuestionInput {
        context: "Paris is the capital of France.".to_string(),
        question: "What is the capital of France?".to_string(),
        max_length: None,
    };
    
    let result = module.forward(input).await.unwrap();
    assert!(result.answer.to_lowercase().contains("paris"));
    assert!(result.confidence > 0.5);
}
```

### 2. Integration Testing

Test module compositions:

```rust
#[tokio::test]
async fn test_document_processing_chain() {
    let chain = create_document_processing_chain().await;
    
    let input = DocumentInput {
        text: "Long document text...".to_string(),
        max_length: 100,
    };
    
    let result = chain.forward(input).await.unwrap();
    assert!(!result.category.is_empty());
    assert!(result.confidence > 0.0);
}
```

### 3. Property-Based Testing

Use property-based testing for robustness:

```rust
#[tokio::test]
async fn test_module_properties() {
    let module = create_test_module().await;
    
    // Test with various inputs
    for _ in 0..100 {
        let input = generate_random_valid_input();
        let result = module.forward(input).await;
        
        match result {
            Ok(output) => {
                // Verify output properties
                assert!(!output.answer.is_empty());
                assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
            }
            Err(e) => {
                // Verify error is expected
                assert!(matches!(e, DspyError::Module { .. }));
            }
        }
    }
}
```

### 4. Performance Testing

Test performance characteristics:

```rust
#[tokio::test]
async fn test_module_performance() {
    let module = create_test_module().await;
    let input = create_test_input();
    
    let start = Instant::now();
    let _result = module.forward(input).await.unwrap();
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_secs(5), "Module too slow: {:?}", duration);
}
```

## Monitoring and Observability

### 1. Add Structured Logging

Use structured logging throughout:

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(self, input))]
async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
    info!(module_name = %self.name(), "Starting forward pass");
    
    let start = Instant::now();
    let result = self.inner_forward(input).await;
    let duration = start.elapsed();
    
    match &result {
        Ok(_) => info!(
            module_name = %self.name(),
            duration_ms = duration.as_millis(),
            "Forward pass completed successfully"
        ),
        Err(e) => error!(
            module_name = %self.name(),
            duration_ms = duration.as_millis(),
            error = %e,
            "Forward pass failed"
        ),
    }
    
    result
}
```

### 2. Collect Metrics

Track important metrics:

```rust
use prometheus::{Counter, Histogram, register_counter, register_histogram};

lazy_static! {
    static ref MODULE_REQUESTS: Counter = register_counter!(
        "dspy_module_requests_total",
        "Total number of module requests"
    ).unwrap();
    
    static ref MODULE_DURATION: Histogram = register_histogram!(
        "dspy_module_duration_seconds",
        "Module execution duration"
    ).unwrap();
}

async fn forward_with_metrics(&self, input: Self::Input) -> DspyResult<Self::Output> {
    MODULE_REQUESTS.inc();
    let timer = MODULE_DURATION.start_timer();
    
    let result = self.forward(input).await;
    timer.observe_duration();
    
    result
}
```

### 3. Health Checks

Implement health checks:

```rust
#[async_trait]
pub trait HealthCheck {
    async fn health_check(&self) -> HealthStatus;
}

#[derive(Debug)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

impl HealthCheck for Predict<String, String> {
    async fn health_check(&self) -> HealthStatus {
        match self.forward("health check".to_string()).await {
            Ok(_) => HealthStatus::Healthy,
            Err(e) => HealthStatus::Unhealthy { 
                reason: format!("Module failed: {}", e) 
            },
        }
    }
}
```

## Deployment

### 1. Environment Configuration

Use environment-based configuration:

```rust
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub anthropic_api_key: String,
    pub cache_redis_url: Option<String>,
    pub log_level: String,
    pub metrics_port: u16,
    pub health_check_interval: Duration,
}

impl DeploymentConfig {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY")?,
            cache_redis_url: std::env::var("REDIS_URL").ok(),
            log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            metrics_port: std::env::var("METRICS_PORT")
                .unwrap_or_else(|_| "9090".to_string())
                .parse()?,
            health_check_interval: Duration::from_secs(
                std::env::var("HEALTH_CHECK_INTERVAL_SECONDS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()?
            ),
        })
    }
}
```

### 2. Graceful Shutdown

Implement graceful shutdown:

```rust
async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    
    // Setup signal handler
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        let _ = shutdown_tx.send(());
    });
    
    // Run server
    let server = tokio::spawn(async move {
        // Server logic here
        loop {
            tokio::select! {
                _ = shutdown_rx => {
                    info!("Shutdown signal received, stopping server");
                    break;
                }
                // Handle requests
            }
        }
    });
    
    server.await?;
    Ok(())
}
```

### 3. Resource Management

Monitor and limit resource usage:

```rust
pub struct ResourceLimits {
    pub max_memory_mb: usize,
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
}

pub struct ResourceMonitor {
    limits: ResourceLimits,
    current_requests: Arc<AtomicUsize>,
}

impl ResourceMonitor {
    pub async fn check_resources(&self) -> Result<(), &'static str> {
        let current_requests = self.current_requests.load(Ordering::Relaxed);
        if current_requests >= self.limits.max_concurrent_requests {
            return Err("Too many concurrent requests");
        }
        
        let memory_usage = get_memory_usage().await;
        if memory_usage > self.limits.max_memory_mb {
            return Err("Memory limit exceeded");
        }
        
        Ok(())
    }
}
```

Following these best practices will help you build robust, scalable, and maintainable DSPy applications that perform well in production environments.

## Additional Resources

- [API Reference](api/README.md) - Complete API documentation
- [Examples](../examples/) - Working code examples
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Security Guide](security.md) - Security best practices
