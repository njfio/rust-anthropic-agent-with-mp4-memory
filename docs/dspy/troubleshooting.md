# DSPy Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with the DSPy framework.

## Table of Contents

1. [Compilation Errors](#compilation-errors)
2. [Runtime Errors](#runtime-errors)
3. [Performance Issues](#performance-issues)
4. [API Integration Issues](#api-integration-issues)
5. [Module Optimization Problems](#module-optimization-problems)
6. [Memory and Resource Issues](#memory-and-resource-issues)
7. [Security and Permissions](#security-and-permissions)

## Compilation Errors

### Error: "trait bound not satisfied"

**Problem**: Type constraints not met for generic parameters.

```rust
error[E0277]: the trait bound `MyType: Serialize` is not satisfied
```

**Solution**: Ensure your types implement required traits:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MyType {
    field: String,
}
```

### Error: "cannot find type in this scope"

**Problem**: Missing imports or incorrect module paths.

**Solution**: Add proper imports:

```rust
use rust_memvid_agent::dspy::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
```

### Error: "async trait method not found"

**Problem**: Missing `async_trait` import or incorrect implementation.

**Solution**: Add the async_trait import and annotation:

```rust
use async_trait::async_trait;

#[async_trait]
impl Module for MyModule {
    // Implementation
}
```

## Runtime Errors

### DspyError::Module

**Problem**: Module execution failed.

**Common Causes**:
- Invalid input format
- API rate limiting
- Network connectivity issues
- Module configuration errors

**Debugging Steps**:

1. **Check input validation**:
```rust
async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
    println!("Validating input: {:?}", input);
    // Add validation logic
    Ok(())
}
```

2. **Add detailed logging**:
```rust
use tracing::{info, error, debug};

async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
    debug!("Module {} processing input", self.name());
    
    match self.process(input).await {
        Ok(output) => {
            info!("Module {} completed successfully", self.name());
            Ok(output)
        }
        Err(e) => {
            error!("Module {} failed: {}", self.name(), e);
            Err(e)
        }
    }
}
```

3. **Test with minimal input**:
```rust
#[tokio::test]
async fn test_minimal_input() {
    let module = create_test_module();
    let minimal_input = create_minimal_valid_input();
    
    let result = module.forward(minimal_input).await;
    assert!(result.is_ok(), "Module failed with minimal input: {:?}", result);
}
```

### DspyError::Anthropic

**Problem**: Anthropic API errors.

**Common Solutions**:

1. **Check API key**:
```rust
let api_key = std::env::var("ANTHROPIC_API_KEY")
    .expect("ANTHROPIC_API_KEY environment variable must be set");
```

2. **Handle rate limiting**:
```rust
async fn forward_with_retry(&self, input: Self::Input) -> DspyResult<Self::Output> {
    let mut retries = 0;
    const MAX_RETRIES: usize = 3;
    
    loop {
        match self.forward(input.clone()).await {
            Ok(result) => return Ok(result),
            Err(DspyError::Anthropic(e)) if retries < MAX_RETRIES => {
                retries += 1;
                let delay = Duration::from_millis(1000 * retries as u64);
                tokio::time::sleep(delay).await;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
```

3. **Check request size**:
```rust
fn validate_request_size(input: &str) -> Result<(), &'static str> {
    const MAX_TOKENS: usize = 100000; // Anthropic's limit
    
    if input.len() > MAX_TOKENS * 4 { // Rough estimate: 4 chars per token
        return Err("Input too large for API");
    }
    Ok(())
}
```

### DspyError::Serialization

**Problem**: JSON serialization/deserialization errors.

**Solutions**:

1. **Check field names match**:
```rust
#[derive(Serialize, Deserialize)]
struct MyStruct {
    #[serde(rename = "field_name")]
    field: String,
}
```

2. **Handle optional fields**:
```rust
#[derive(Serialize, Deserialize)]
struct MyStruct {
    required_field: String,
    #[serde(default)]
    optional_field: Option<String>,
}
```

3. **Add custom serialization**:
```rust
#[derive(Serialize, Deserialize)]
struct MyStruct {
    #[serde(serialize_with = "serialize_custom")]
    custom_field: CustomType,
}

fn serialize_custom<S>(value: &CustomType, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // Custom serialization logic
}
```

## Performance Issues

### Slow Module Execution

**Diagnosis**:

1. **Add timing measurements**:
```rust
async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
    let start = std::time::Instant::now();
    let result = self.inner_forward(input).await;
    let duration = start.elapsed();
    
    if duration > Duration::from_secs(5) {
        warn!("Slow execution: {:?}", duration);
    }
    
    result
}
```

2. **Profile API calls**:
```rust
async fn call_api(&self, request: &str) -> Result<String, ApiError> {
    let start = Instant::now();
    let response = self.client.call(request).await?;
    let api_duration = start.elapsed();
    
    info!("API call took {:?}", api_duration);
    Ok(response)
}
```

**Solutions**:

1. **Enable caching**:
```rust
let cache_config = CacheConfig {
    enabled: true,
    ttl_seconds: 3600,
    max_entries: 10000,
};
module.enable_caching(cache_config);
```

2. **Use batch processing**:
```rust
// Instead of individual calls
let results = module.forward_batch(inputs).await?;
```

3. **Optimize configuration**:
```rust
let config = PredictConfig {
    max_tokens: 100,        // Reduce for shorter responses
    temperature: 0.1,       // Lower for faster, more deterministic responses
    timeout_seconds: 30,    // Set reasonable timeout
    ..Default::default()
};
```

### Memory Usage Issues

**Diagnosis**:
```rust
fn check_memory_usage() {
    let usage = get_memory_usage();
    if usage > MEMORY_THRESHOLD {
        warn!("High memory usage: {} MB", usage);
    }
}
```

**Solutions**:

1. **Limit concurrent operations**:
```rust
let semaphore = Arc::new(Semaphore::new(10)); // Max 10 concurrent operations

async fn process_with_limit(&self, input: Input) -> Result<Output> {
    let _permit = self.semaphore.acquire().await?;
    self.process(input).await
}
```

2. **Clear caches periodically**:
```rust
async fn cleanup_caches(&mut self) {
    if self.cache.len() > MAX_CACHE_SIZE {
        self.cache.clear();
    }
}
```

## API Integration Issues

### Connection Timeouts

**Problem**: API calls timing out.

**Solutions**:

1. **Increase timeout**:
```rust
let config = PredictConfig {
    timeout_seconds: 60, // Increase from default
    ..Default::default()
};
```

2. **Implement connection pooling**:
```rust
let client_config = AnthropicClientConfig {
    max_connections: 10,
    connection_timeout: Duration::from_secs(30),
    request_timeout: Duration::from_secs(60),
};
```

### Authentication Errors

**Problem**: API authentication failing.

**Debugging**:

1. **Verify API key format**:
```rust
fn validate_api_key(key: &str) -> Result<(), &'static str> {
    if !key.starts_with("sk-") {
        return Err("Invalid API key format");
    }
    if key.len() < 20 {
        return Err("API key too short");
    }
    Ok(())
}
```

2. **Test API key separately**:
```rust
#[tokio::test]
async fn test_api_key() {
    let client = AnthropicClient::new(api_key, None).unwrap();
    let result = client.test_connection().await;
    assert!(result.is_ok(), "API key test failed: {:?}", result);
}
```

## Module Optimization Problems

### Poor Optimization Results

**Problem**: Teleprompter optimization not improving performance.

**Diagnosis**:

1. **Check example quality**:
```rust
fn validate_examples<I, O>(examples: &ExampleSet<I, O>) -> Result<(), String> {
    if examples.len() < 10 {
        return Err("Need at least 10 examples for optimization".to_string());
    }
    
    // Check for duplicate examples
    let mut seen = HashSet::new();
    for example in examples.iter() {
        let key = format!("{:?}", example.input());
        if !seen.insert(key) {
            return Err("Duplicate examples found".to_string());
        }
    }
    
    Ok(())
}
```

2. **Monitor optimization progress**:
```rust
let mut teleprompter = Teleprompter::new(strategy);
teleprompter.set_progress_callback(|iteration, score| {
    println!("Iteration {}: score = {:.3}", iteration, score);
});
```

**Solutions**:

1. **Improve example diversity**:
```rust
fn create_diverse_examples() -> ExampleSet<String, String> {
    let mut examples = ExampleSet::new();
    
    // Add examples covering different scenarios
    examples.add_example(Example::new("short question".to_string(), "short answer".to_string()));
    examples.add_example(Example::new("long detailed question...".to_string(), "detailed answer...".to_string()));
    examples.add_example(Example::new("edge case question".to_string(), "edge case answer".to_string()));
    
    examples
}
```

2. **Try different optimization strategies**:
```rust
// Try bootstrap optimization
let bootstrap_strategy = OptimizationStrategy::bootstrap(10, 20);

// Try random sampling
let random_strategy = OptimizationStrategy::random_sampling(50, Some(42));

// Try advanced strategies
let mipro_strategy = OptimizationStrategy::mipro_v2(MIPROv2Config::default());
```

### Optimization Timeout

**Problem**: Optimization taking too long.

**Solutions**:

1. **Reduce optimization scope**:
```rust
let config = TeleprompterConfig {
    max_iterations: 10,     // Reduce from default
    early_stopping_patience: 3,
    convergence_threshold: 0.05, // Less strict
    ..Default::default()
};
```

2. **Use smaller example sets**:
```rust
let (train_examples, _) = examples.split(0.5); // Use only 50% for optimization
```

## Memory and Resource Issues

### Out of Memory Errors

**Problem**: Application running out of memory.

**Solutions**:

1. **Monitor memory usage**:
```rust
async fn monitor_memory() {
    loop {
        let usage = get_memory_usage();
        if usage > MEMORY_WARNING_THRESHOLD {
            warn!("High memory usage: {} MB", usage);
            
            if usage > MEMORY_CRITICAL_THRESHOLD {
                // Trigger cleanup
                cleanup_resources().await;
            }
        }
        
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}
```

2. **Implement resource limits**:
```rust
let config = PredictConfig {
    max_memory_mb: 512,
    enable_memory_monitoring: true,
    ..Default::default()
};
```

3. **Use streaming for large inputs**:
```rust
async fn process_large_input(&self, input: LargeInput) -> DspyResult<Output> {
    let chunks = input.into_chunks(CHUNK_SIZE);
    let mut results = Vec::new();
    
    for chunk in chunks {
        let result = self.process_chunk(chunk).await?;
        results.push(result);
        
        // Allow garbage collection between chunks
        tokio::task::yield_now().await;
    }
    
    self.combine_results(results)
}
```

### Resource Leaks

**Problem**: Resources not being properly cleaned up.

**Solutions**:

1. **Use RAII patterns**:
```rust
struct ResourceGuard {
    resource: Resource,
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        self.resource.cleanup();
    }
}
```

2. **Implement proper cleanup**:
```rust
async fn cleanup(&mut self) {
    self.cache.clear();
    self.connections.close_all().await;
    self.background_tasks.abort_all();
}
```

## Security and Permissions

### Permission Denied Errors

**Problem**: Security context rejecting operations.

**Solutions**:

1. **Check permissions**:
```rust
fn check_permissions(context: &SecurityContext, operation: &str) -> Result<(), SecurityError> {
    let required_permission = format!("dspy:{}", operation);
    
    if !context.permissions.contains(&required_permission) {
        return Err(SecurityError::InsufficientPermissions {
            required: required_permission,
            available: context.permissions.clone(),
        });
    }
    
    Ok(())
}
```

2. **Debug security context**:
```rust
fn debug_security_context(context: &SecurityContext) {
    debug!("User ID: {}", context.user_id);
    debug!("Roles: {:?}", context.roles);
    debug!("Permissions: {:?}", context.permissions);
}
```

### Code Execution Security

**Problem**: Program of Thought security restrictions.

**Solutions**:

1. **Review security settings**:
```rust
let security_config = SecurityRestrictions {
    disallow_file_operations: true,
    disallow_network_operations: true,
    disallow_subprocess: true,
    max_memory_mb: 100,
    blacklisted_functions: vec![
        "exec".to_string(),
        "eval".to_string(),
        "open".to_string(),
    ],
};
```

2. **Test in sandbox**:
```rust
#[tokio::test]
async fn test_code_security() {
    let pot = ProgramOfThought::new(signature, client);
    
    // Test that dangerous code is rejected
    let dangerous_code = "import os; os.system('rm -rf /')";
    let result = pot.validate_code(dangerous_code);
    assert!(result.is_err());
}
```

## Getting Help

If you're still experiencing issues:

1. **Enable debug logging**:
```rust
std::env::set_var("RUST_LOG", "debug");
tracing_subscriber::init();
```

2. **Create minimal reproduction**:
```rust
// Create the smallest possible example that reproduces the issue
```

3. **Check the documentation**:
   - [API Reference](api/README.md)
   - [Best Practices](best_practices.md)
   - [Examples](../examples/)

4. **Report issues** with:
   - Rust version
   - DSPy version
   - Complete error messages
   - Minimal reproduction code
   - Environment details
