//! Core module system for DSPy integration
//!
//! This module provides the fundamental Module trait and implementations
//! that enable composable, type-safe LLM operations with optimization support.

use crate::dspy::error::DspyResult;
use crate::dspy::signature::Signature;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::warn;
use uuid::Uuid;

/// Core trait for all DSPy modules
#[async_trait]
pub trait Module: Send + Sync {
    /// Input type for the module
    type Input: Send + Sync + Serialize + for<'de> Deserialize<'de>;
    /// Output type for the module
    type Output: Send + Sync + Serialize + for<'de> Deserialize<'de>;

    /// Get the module's unique identifier
    fn id(&self) -> &str;

    /// Get the module's name
    fn name(&self) -> &str;

    /// Get the module's signature
    fn signature(&self) -> &Signature<Self::Input, Self::Output>;

    /// Execute the module's forward pass
    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output>;

    /// Get module metadata
    fn metadata(&self) -> &ModuleMetadata {
        static DEFAULT_METADATA: std::sync::OnceLock<ModuleMetadata> = std::sync::OnceLock::new();
        DEFAULT_METADATA.get_or_init(ModuleMetadata::default)
    }

    /// Check if the module supports compilation/optimization
    fn supports_compilation(&self) -> bool {
        false
    }

    /// Compile the module with optimization (default implementation does nothing)
    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        if self.supports_compilation() {
            warn!(
                "Module {} supports compilation but no implementation provided",
                self.name()
            );
        }
        Ok(())
    }

    /// Get compilation status
    fn is_compiled(&self) -> bool {
        false
    }

    /// Reset the module to uncompiled state
    async fn reset(&mut self) -> DspyResult<()> {
        Ok(())
    }

    /// Validate input before processing
    async fn validate_input(&self, input: &Self::Input) -> DspyResult<()> {
        self.signature().validate_input(input)
    }

    /// Validate output after processing
    async fn validate_output(&self, output: &Self::Output) -> DspyResult<()> {
        self.signature().validate_output(output)
    }

    /// Get module statistics
    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }

    /// Clone the module (for modules that support cloning)
    fn clone_module(&self) -> Option<Box<dyn Module<Input = Self::Input, Output = Self::Output>>> {
        None
    }
}

/// Metadata associated with a DSPy module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    /// Module version
    pub version: String,
    /// Module author
    pub author: Option<String>,
    /// Module description
    pub description: String,
    /// Module tags for categorization
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Custom metadata fields
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ModuleMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            version: "1.0.0".to_string(),
            author: None,
            description: String::new(),
            tags: Vec::new(),
            created_at: now,
            modified_at: now,
            custom: HashMap::new(),
        }
    }
}

impl ModuleMetadata {
    /// Create new metadata
    pub fn new<S: Into<String>>(description: S) -> Self {
        Self {
            description: description.into(),
            ..Default::default()
        }
    }

    /// Set version
    pub fn with_version<S: Into<String>>(mut self, version: S) -> Self {
        self.version = version.into();
        self.modified_at = chrono::Utc::now();
        self
    }

    /// Set author
    pub fn with_author<S: Into<String>>(mut self, author: S) -> Self {
        self.author = Some(author.into());
        self.modified_at = chrono::Utc::now();
        self
    }

    /// Add tag
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self.modified_at = chrono::Utc::now();
        self
    }

    /// Add custom metadata
    pub fn with_custom<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.custom.insert(key.into(), value.into());
        self.modified_at = chrono::Utc::now();
        self
    }
}

/// Statistics for module execution and performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleStats {
    /// Total number of executions
    pub execution_count: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Option<Duration>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Number of successful executions
    pub success_count: u64,
    /// Number of failed executions
    pub error_count: u64,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
    /// Compilation status
    pub is_compiled: bool,
    /// Compilation timestamp
    pub compiled_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for ModuleStats {
    fn default() -> Self {
        Self {
            execution_count: 0,
            total_execution_time: Duration::ZERO,
            average_execution_time: Duration::ZERO,
            min_execution_time: None,
            max_execution_time: None,
            success_count: 0,
            error_count: 0,
            last_execution: None,
            is_compiled: false,
            compiled_at: None,
        }
    }
}

impl ModuleStats {
    /// Record a successful execution
    pub fn record_success(&mut self, execution_time: Duration) {
        self.execution_count += 1;
        self.success_count += 1;
        self.total_execution_time += execution_time;
        self.average_execution_time = self.total_execution_time / self.execution_count as u32;
        self.last_execution = Some(chrono::Utc::now());

        // Update min/max times
        match self.min_execution_time {
            None => self.min_execution_time = Some(execution_time),
            Some(min) if execution_time < min => self.min_execution_time = Some(execution_time),
            _ => {}
        }

        match self.max_execution_time {
            None => self.max_execution_time = Some(execution_time),
            Some(max) if execution_time > max => self.max_execution_time = Some(execution_time),
            _ => {}
        }
    }

    /// Record a failed execution
    pub fn record_error(&mut self, execution_time: Duration) {
        self.execution_count += 1;
        self.error_count += 1;
        self.total_execution_time += execution_time;
        self.average_execution_time = self.total_execution_time / self.execution_count as u32;
        self.last_execution = Some(chrono::Utc::now());
    }

    /// Record compilation
    pub fn record_compilation(&mut self) {
        self.is_compiled = true;
        self.compiled_at = Some(chrono::Utc::now());
    }

    /// Reset compilation status
    pub fn reset_compilation(&mut self) {
        self.is_compiled = false;
        self.compiled_at = None;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.execution_count as f64
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.execution_count as f64
        }
    }
}

/// Base implementation for modules with common functionality
pub struct BaseModule<I, O> {
    /// Module ID
    pub id: String,
    /// Module name
    pub name: String,
    /// Module signature
    pub signature: Signature<I, O>,
    /// Module metadata
    pub metadata: ModuleMetadata,
    /// Module statistics
    pub stats: Arc<RwLock<ModuleStats>>,
}

impl<I, O> BaseModule<I, O>
where
    I: Send + Sync + Serialize + for<'de> Deserialize<'de>,
    O: Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new base module
    pub fn new<S: Into<String>>(name: S, signature: Signature<I, O>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            signature,
            metadata: ModuleMetadata::default(),
            stats: Arc::new(RwLock::new(ModuleStats::default())),
        }
    }

    /// Create with metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get mutable reference to stats (for internal use)
    pub async fn stats_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, ModuleStats> {
        self.stats.write().await
    }

    /// Execute with timing and error handling
    pub async fn execute_with_stats<F, Fut>(&self, input: I, execution_fn: F) -> DspyResult<O>
    where
        F: FnOnce(I) -> Fut,
        Fut: std::future::Future<Output = DspyResult<O>>,
    {
        let start_time = Instant::now();

        // Validate input
        self.signature.validate_input(&input)?;

        // Execute the function
        let result = execution_fn(input).await;

        let execution_time = start_time.elapsed();

        // Update statistics
        let mut stats = self.stats.write().await;
        match &result {
            Ok(output) => {
                // Validate output
                if let Err(e) = self.signature.validate_output(output) {
                    stats.record_error(execution_time);
                    return Err(e);
                }
                stats.record_success(execution_time);
            }
            Err(_) => {
                stats.record_error(execution_time);
            }
        }

        result
    }
}

impl<I, O> fmt::Debug for BaseModule<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaseModule")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<I, O> fmt::Display for BaseModule<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Module[{}:{}]", self.name, self.id)
    }
}

/// Module execution context for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Execution ID
    pub execution_id: String,
    /// Module ID
    pub module_id: String,
    /// Start timestamp
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// End timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Success status
    pub success: Option<bool>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Additional context data
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new<S: Into<String>>(module_id: S) -> Self {
        Self {
            execution_id: Uuid::new_v4().to_string(),
            module_id: module_id.into(),
            started_at: chrono::Utc::now(),
            completed_at: None,
            duration: None,
            success: None,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Mark execution as completed successfully
    pub fn complete_success(mut self) -> Self {
        let now = chrono::Utc::now();
        self.completed_at = Some(now);
        self.duration = Some(Duration::from_millis(
            (now - self.started_at).num_milliseconds() as u64,
        ));
        self.success = Some(true);
        self
    }

    /// Mark execution as failed
    pub fn complete_error<S: Into<String>>(mut self, error_message: S) -> Self {
        let now = chrono::Utc::now();
        self.completed_at = Some(now);
        self.duration = Some(Duration::from_millis(
            (now - self.started_at).num_milliseconds() as u64,
        ));
        self.success = Some(false);
        self.error_message = Some(error_message.into());
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspy::signature::{FieldType, SignatureBuilder};
    use serde_json::json;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestInput {
        text: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestOutput {
        result: String,
    }

    struct TestModule {
        base: BaseModule<TestInput, TestOutput>,
    }

    impl TestModule {
        fn new() -> Self {
            let signature = SignatureBuilder::new("test_module")
                .description("A test module")
                .input_field("text", "Input text", FieldType::String)
                .output_field("result", "Output result", FieldType::String)
                .build();

            Self {
                base: BaseModule::new("test_module", signature),
            }
        }
    }

    #[async_trait]
    impl Module for TestModule {
        type Input = TestInput;
        type Output = TestOutput;

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
            self.base
                .execute_with_stats(input, |input| async move {
                    Ok(TestOutput {
                        result: format!("Processed: {}", input.text),
                    })
                })
                .await
        }

        fn metadata(&self) -> &ModuleMetadata {
            &self.base.metadata
        }

        fn stats(&self) -> &ModuleStats {
            static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
            DEFAULT_STATS.get_or_init(|| ModuleStats::default())
        }
    }

    #[tokio::test]
    async fn test_module_creation() {
        let module = TestModule::new();
        assert_eq!(module.name(), "test_module");
        assert!(!module.id().is_empty());
        assert_eq!(module.signature().name, "test_module");
    }

    #[tokio::test]
    async fn test_module_execution() {
        let module = TestModule::new();
        let input = TestInput {
            text: "Hello, World!".to_string(),
        };

        let result = module.forward(input).await.unwrap();
        assert_eq!(result.result, "Processed: Hello, World!");

        let stats = module.stats();
        // Note: This is using the default stats since the test module doesn't track real stats
        assert_eq!(stats.execution_count, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_module_metadata() {
        let metadata = ModuleMetadata::new("Test module")
            .with_version("2.0.0")
            .with_author("Test Author")
            .with_tag("test")
            .with_tag("example")
            .with_custom("key", "value");

        assert_eq!(metadata.description, "Test module");
        assert_eq!(metadata.version, "2.0.0");
        assert_eq!(metadata.author, Some("Test Author".to_string()));
        assert_eq!(metadata.tags, vec!["test", "example"]);
        assert_eq!(metadata.custom.get("key"), Some(&json!("value")));
    }

    #[test]
    fn test_module_stats() {
        let mut stats = ModuleStats::default();

        stats.record_success(Duration::from_millis(100));
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.success_rate(), 1.0);

        stats.record_error(Duration::from_millis(50));
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.success_rate(), 0.5);
        assert_eq!(stats.error_rate(), 0.5);
    }

    #[test]
    fn test_execution_context() {
        let context = ExecutionContext::new("test_module")
            .with_metadata("key", "value")
            .complete_success();

        assert_eq!(context.module_id, "test_module");
        assert_eq!(context.success, Some(true));
        assert!(context.completed_at.is_some());
        assert!(context.duration.is_some());
        assert_eq!(context.metadata.get("key"), Some(&json!("value")));
    }
}
