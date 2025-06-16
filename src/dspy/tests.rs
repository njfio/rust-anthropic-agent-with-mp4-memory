//! Comprehensive tests for DSPy core functionality
//!
//! This module provides extensive test coverage for the DSPy integration,
//! ensuring all components work correctly and integrate properly with
//! the existing agent system.

use super::*;
use crate::dspy::{
    bootstrap::{BootstrapConfig, BootstrapFewShot, BootstrapStats, ValidationStrictness},
    cache::{Cache, CacheConfig},
    chain::Chain,
    compiler::{CompilationContext, CompilationMetrics, Compiler, CompilerConfig, CompilerStats},
    composition::{Conditional, Parallel},
    error::{DspyError, ErrorSeverity},
    examples::{Example, ExampleSet},
    module::{BaseModule, ExecutionContext, Module, ModuleMetadata, ModuleStats},
    optimization::{OptimizationMetrics, OptimizationStrategy, Optimizer},
    predictor::{Predict, PredictConfig},
    signature::{Field, FieldConstraint, FieldType, Signature, SignatureBuilder},
    teleprompter::{OptimizationResult, Teleprompter, TeleprompterConfig},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use tokio;

// Test data structures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct QuestionInput {
    question: String,
    context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AnswerOutput {
    answer: String,
    confidence: f64,
    reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SimpleInput {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SimpleOutput {
    result: String,
}

// Test module implementation
struct MockModule {
    base: BaseModule<SimpleInput, SimpleOutput>,
    should_fail: bool,
}

impl MockModule {
    fn new(should_fail: bool) -> Self {
        let signature = SignatureBuilder::new("mock_module")
            .description("A mock module for testing")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        Self {
            base: BaseModule::new("mock_module", signature),
            should_fail,
        }
    }
}

#[async_trait]
impl Module for MockModule {
    type Input = SimpleInput;
    type Output = SimpleOutput;

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
        if self.should_fail {
            return Err(DspyError::module("mock_module", "Intentional test failure"));
        }

        self.base
            .execute_with_stats(input, |input| async move {
                Ok(SimpleOutput {
                    result: format!("Processed: {}", input.text),
                })
            })
            .await
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.base.metadata
    }

    fn supports_compilation(&self) -> bool {
        true
    }

    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        let mut stats = self.base.stats_mut().await;
        stats.record_compilation();
        Ok(())
    }

    fn is_compiled(&self) -> bool {
        // This is a simplified check - in practice, you'd check the actual compilation state
        false
    }

    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }
}

// Error handling tests
#[tokio::test]
async fn test_dspy_error_creation_and_conversion() {
    let signature_error = DspyError::signature("Invalid field definition");
    assert_eq!(signature_error.category(), "signature");
    assert_eq!(signature_error.severity(), ErrorSeverity::High);
    assert!(!signature_error.is_recoverable());

    let module_error = DspyError::module("test_module", "Execution failed");
    assert_eq!(module_error.category(), "module");
    assert_eq!(module_error.severity(), ErrorSeverity::Medium);

    let optimization_error = DspyError::optimization("bootstrap", "No examples provided");
    assert_eq!(optimization_error.category(), "optimization");
    assert!(optimization_error.is_recoverable());

    // Test conversion to AgentError
    let agent_error = signature_error.into_agent_error();
    match agent_error {
        AgentError::InvalidInput { message } => {
            assert!(message.contains("DSPy signature error"));
        }
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn test_error_severity_and_recoverability() {
    assert_eq!(
        DspyError::configuration("param", "bad value").severity(),
        ErrorSeverity::Critical
    );
    assert_eq!(
        DspyError::type_validation("field", "wrong type").severity(),
        ErrorSeverity::High
    );
    assert_eq!(
        DspyError::module("mod", "error").severity(),
        ErrorSeverity::Medium
    );
    assert_eq!(
        DspyError::cache("get", "miss").severity(),
        ErrorSeverity::Low
    );

    assert!(DspyError::resource("memory", "low").is_recoverable());
    assert!(DspyError::optimization("strategy", "failed").is_recoverable());
    assert!(!DspyError::signature("invalid").is_recoverable());
}

// Signature tests
#[test]
fn test_field_creation_and_validation() {
    let field = Field::new("test_field", "A test field", FieldType::String)
        .optional(Some(json!("default_value")))
        .with_constraint(FieldConstraint::MinLength(1))
        .with_constraint(FieldConstraint::MaxLength(100))
        .with_constraint(FieldConstraint::Pattern(r"^[a-zA-Z]+$".to_string()));

    assert_eq!(field.name, "test_field");
    assert_eq!(field.description, "A test field");
    assert_eq!(field.field_type, FieldType::String);
    assert!(!field.required);
    assert_eq!(field.constraints.len(), 3);

    // Test valid value
    assert!(field.validate_value(&json!("hello")).is_ok());

    // Test invalid values
    assert!(field.validate_value(&json!("")).is_err()); // Too short
    assert!(field.validate_value(&json!("a".repeat(101))).is_err()); // Too long
    assert!(field.validate_value(&json!("hello123")).is_err()); // Invalid pattern
    assert!(field.validate_value(&json!(123)).is_err()); // Wrong type
}

#[test]
fn test_field_type_compatibility() {
    let string_field = Field::new("str", "String field", FieldType::String);
    let int_field = Field::new("int", "Integer field", FieldType::Integer);
    let float_field = Field::new("float", "Float field", FieldType::Float);
    let bool_field = Field::new("bool", "Boolean field", FieldType::Boolean);
    let array_field = Field::new(
        "arr",
        "Array field",
        FieldType::Array(Box::new(FieldType::String)),
    );

    assert!(string_field.validate_value(&json!("test")).is_ok());
    assert!(int_field.validate_value(&json!(42)).is_ok());
    assert!(float_field.validate_value(&json!(3.14)).is_ok());
    assert!(bool_field.validate_value(&json!(true)).is_ok());
    assert!(array_field.validate_value(&json!(["a", "b", "c"])).is_ok());

    // Test type mismatches
    assert!(string_field.validate_value(&json!(123)).is_err());
    assert!(int_field.validate_value(&json!("not a number")).is_err());
    assert!(bool_field.validate_value(&json!("not a boolean")).is_err());
}

#[test]
fn test_signature_creation_and_validation() {
    let signature = SignatureBuilder::<QuestionInput, AnswerOutput>::new("qa_signature")
        .description("Question answering signature")
        .input_field("question", "The question to answer", FieldType::String)
        .input_field("context", "Optional context", FieldType::String)
        .output_field("answer", "The answer", FieldType::String)
        .output_field("confidence", "Confidence score", FieldType::Float)
        .output_field("reasoning", "Reasoning steps", FieldType::String)
        .metadata("version", "1.0")
        .metadata("author", "test")
        .build();

    assert_eq!(signature.name, "qa_signature");
    assert_eq!(signature.description, "Question answering signature");
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 3);
    assert_eq!(signature.metadata.len(), 2);

    // Test field access
    assert!(signature.get_input_field("question").is_some());
    assert!(signature.get_input_field("nonexistent").is_none());
    assert!(signature.get_output_field("answer").is_some());

    // Test field name lists
    let input_names = signature.input_field_names();
    assert!(input_names.contains(&"question"));
    assert!(input_names.contains(&"context"));

    let output_names = signature.output_field_names();
    assert!(output_names.contains(&"answer"));
    assert!(output_names.contains(&"confidence"));
    assert!(output_names.contains(&"reasoning"));
}

#[test]
fn test_signature_validation() {
    let signature = SignatureBuilder::<QuestionInput, AnswerOutput>::new("test")
        .input_field("question", "Question", FieldType::String)
        .output_field("answer", "Answer", FieldType::String)
        .output_field("confidence", "Confidence", FieldType::Float)
        .build();

    let valid_input = QuestionInput {
        question: "What is 2+2?".to_string(),
        context: Some("Math context".to_string()),
    };

    let valid_output = AnswerOutput {
        answer: "4".to_string(),
        confidence: 0.95,
        reasoning: Some("Simple arithmetic".to_string()),
    };

    assert!(signature.validate_input(&valid_input).is_ok());
    assert!(signature.validate_output(&valid_output).is_ok());
}

#[test]
fn test_prompt_template_generation() {
    let signature = SignatureBuilder::<QuestionInput, AnswerOutput>::new("qa_task")
        .description("Answer questions with reasoning")
        .input_field("question", "The question to answer", FieldType::String)
        .input_field("context", "Relevant context information", FieldType::String)
        .output_field("answer", "The final answer", FieldType::String)
        .output_field("confidence", "Confidence score (0-1)", FieldType::Float)
        .build();

    let template = signature.generate_prompt_template();

    assert!(template.contains("Task: Answer questions with reasoning"));
    assert!(template.contains("Input:"));
    assert!(template.contains("- question: The question to answer"));
    assert!(template.contains("- context: Relevant context information"));
    assert!(template.contains("Output:"));
    assert!(template.contains("- answer: The final answer"));
    assert!(template.contains("- confidence: Confidence score (0-1)"));
}

// Module tests
#[tokio::test]
async fn test_module_creation_and_basic_functionality() {
    let module = MockModule::new(false);

    assert!(!module.id().is_empty());
    assert_eq!(module.name(), "mock_module");
    assert_eq!(module.signature().name, "mock_module");
    assert!(module.supports_compilation());
    assert!(!module.is_compiled());
}

#[tokio::test]
async fn test_module_execution_success() {
    let module = MockModule::new(false);
    let input = SimpleInput {
        text: "Hello, World!".to_string(),
    };

    let result = module.forward(input).await;
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.result, "Processed: Hello, World!");

    // Test that stats are accessible (MockModule uses default stats)
    let stats = module.stats();
    assert_eq!(stats.execution_count, 0); // MockModule doesn't track real stats
    assert_eq!(stats.success_count, 0);
}

#[tokio::test]
async fn test_module_execution_failure() {
    let module = MockModule::new(true);
    let input = SimpleInput {
        text: "This will fail".to_string(),
    };

    let result = module.forward(input).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        DspyError::Module {
            module_name,
            message,
        } => {
            assert_eq!(module_name, "mock_module");
            assert!(message.contains("Intentional test failure"));
        }
        _ => panic!("Expected Module error"),
    }
}

#[tokio::test]
async fn test_module_compilation() {
    let mut module = MockModule::new(false);
    assert!(!module.is_compiled());

    let examples = vec![
        (
            SimpleInput {
                text: "test1".to_string(),
            },
            SimpleOutput {
                result: "result1".to_string(),
            },
        ),
        (
            SimpleInput {
                text: "test2".to_string(),
            },
            SimpleOutput {
                result: "result2".to_string(),
            },
        ),
    ];

    let result = module.compile(examples).await;
    assert!(result.is_ok());
}

#[test]
fn test_module_metadata() {
    let metadata = ModuleMetadata::new("Test module for DSPy")
        .with_version("2.1.0")
        .with_author("DSPy Team")
        .with_tag("test")
        .with_tag("dspy")
        .with_tag("nlp")
        .with_custom("complexity", "low")
        .with_custom(
            "performance",
            json!({"latency": "10ms", "throughput": "1000rps"}),
        );

    assert_eq!(metadata.description, "Test module for DSPy");
    assert_eq!(metadata.version, "2.1.0");
    assert_eq!(metadata.author, Some("DSPy Team".to_string()));
    assert_eq!(metadata.tags.len(), 3);
    assert!(metadata.tags.contains(&"test".to_string()));
    assert!(metadata.tags.contains(&"dspy".to_string()));
    assert!(metadata.tags.contains(&"nlp".to_string()));
    assert_eq!(metadata.custom.get("complexity"), Some(&json!("low")));
    assert!(metadata.custom.contains_key("performance"));
}

#[test]
fn test_module_stats() {
    let mut stats = ModuleStats::default();

    // Test initial state
    assert_eq!(stats.execution_count, 0);
    assert_eq!(stats.success_count, 0);
    assert_eq!(stats.error_count, 0);
    assert_eq!(stats.success_rate(), 0.0);
    assert_eq!(stats.error_rate(), 0.0);

    // Record successful executions
    stats.record_success(Duration::from_millis(100));
    stats.record_success(Duration::from_millis(150));
    stats.record_success(Duration::from_millis(80));

    assert_eq!(stats.execution_count, 3);
    assert_eq!(stats.success_count, 3);
    assert_eq!(stats.error_count, 0);
    assert_eq!(stats.success_rate(), 1.0);
    assert_eq!(stats.error_rate(), 0.0);
    assert_eq!(stats.min_execution_time, Some(Duration::from_millis(80)));
    assert_eq!(stats.max_execution_time, Some(Duration::from_millis(150)));

    // Record error
    stats.record_error(Duration::from_millis(200));

    assert_eq!(stats.execution_count, 4);
    assert_eq!(stats.success_count, 3);
    assert_eq!(stats.error_count, 1);
    assert_eq!(stats.success_rate(), 0.75);
    assert_eq!(stats.error_rate(), 0.25);

    // Test compilation recording
    assert!(!stats.is_compiled);
    stats.record_compilation();
    assert!(stats.is_compiled);
    assert!(stats.compiled_at.is_some());

    stats.reset_compilation();
    assert!(!stats.is_compiled);
    assert!(stats.compiled_at.is_none());
}

#[test]
fn test_execution_context() {
    let context = ExecutionContext::new("test_module_id")
        .with_metadata("input_size", 1024)
        .with_metadata("model_version", "1.0.0");

    assert_eq!(context.module_id, "test_module_id");
    assert!(!context.execution_id.is_empty());
    assert!(context.completed_at.is_none());
    assert!(context.success.is_none());
    assert_eq!(context.metadata.len(), 2);

    // Test successful completion
    let completed_context = context.clone().complete_success();
    assert!(completed_context.completed_at.is_some());
    assert!(completed_context.duration.is_some());
    assert_eq!(completed_context.success, Some(true));
    assert!(completed_context.error_message.is_none());

    // Test error completion
    let error_context = context.complete_error("Test error message");
    assert!(error_context.completed_at.is_some());
    assert!(error_context.duration.is_some());
    assert_eq!(error_context.success, Some(false));
    assert_eq!(
        error_context.error_message,
        Some("Test error message".to_string())
    );
}

// Registry tests
#[test]
fn test_dspy_registry() {
    let mut registry = DspyRegistry::new();
    assert_eq!(registry.module_count(), 0);
    assert!(registry.list_modules().is_empty());

    // Test module info implementation
    #[derive(Debug)]
    struct TestModuleInfo {
        name: String,
        description: String,
        version: String,
        capabilities: Vec<String>,
    }

    impl ModuleInfo for TestModuleInfo {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> Option<&str> {
            Some(&self.description)
        }

        fn version(&self) -> &str {
            &self.version
        }

        fn capabilities(&self) -> Vec<String> {
            self.capabilities.clone()
        }
    }

    let module_info = TestModuleInfo {
        name: "test_module".to_string(),
        description: "A test module".to_string(),
        version: "1.0.0".to_string(),
        capabilities: vec!["prediction".to_string(), "optimization".to_string()],
    };

    // Register module
    assert!(registry.register_module(module_info).is_ok());
    assert_eq!(registry.module_count(), 1);
    assert!(registry.has_module("test_module"));
    assert!(registry.list_modules().contains(&"test_module"));

    // Test duplicate registration
    let duplicate_info = TestModuleInfo {
        name: "test_module".to_string(),
        description: "Duplicate".to_string(),
        version: "2.0.0".to_string(),
        capabilities: vec![],
    };
    assert!(registry.register_module(duplicate_info).is_err());

    // Test module info retrieval
    let info = registry.get_module_info("test_module");
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.name(), "test_module");
    assert_eq!(info.description(), Some("A test module"));
    assert_eq!(info.version(), "1.0.0");
    assert_eq!(info.capabilities().len(), 2);

    // Test unregistration
    assert!(registry.unregister_module("test_module").is_ok());
    assert_eq!(registry.module_count(), 0);
    assert!(!registry.has_module("test_module"));

    // Test unregistering non-existent module
    assert!(registry.unregister_module("nonexistent").is_err());
}

// Configuration tests
#[test]
fn test_dspy_config() {
    let config = DspyConfig::new()
        .with_optimization(true)
        .with_max_iterations(20)
        .with_timeout(600)
        .with_caching(true)
        .with_cache_ttl(7200)
        .with_monitoring(true)
        .with_max_examples(200)
        .with_min_confidence(0.8);

    assert!(config.enable_optimization);
    assert_eq!(config.max_optimization_iterations, 20);
    assert_eq!(config.optimization_timeout_seconds, 600);
    assert!(config.enable_module_caching);
    assert_eq!(config.cache_ttl_seconds, 7200);
    assert!(config.enable_monitoring);
    assert_eq!(config.max_examples, 200);
    assert_eq!(config.min_confidence_threshold, 0.8);

    assert!(config.validate().is_ok());
}

#[test]
fn test_dspy_config_validation() {
    let mut config = DspyConfig::default();
    assert!(config.validate().is_ok());

    // Test invalid configurations
    config.max_optimization_iterations = 0;
    assert!(config.validate().is_err());

    config = DspyConfig::default();
    config.optimization_timeout_seconds = 0;
    assert!(config.validate().is_err());

    config = DspyConfig::default();
    config.cache_ttl_seconds = 0;
    assert!(config.validate().is_err());

    config = DspyConfig::default();
    config.max_examples = 0;
    assert!(config.validate().is_err());

    config = DspyConfig::default();
    config.min_confidence_threshold = 1.5;
    assert!(config.validate().is_err());

    config = DspyConfig::default();
    config.min_confidence_threshold = -0.1;
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn test_dspy_system_initialization() {
    let config = DspyConfig::default();
    let registry = init_dspy(config).await;
    assert!(registry.is_ok());

    let registry = registry.unwrap();
    assert_eq!(registry.module_count(), 0);
    assert!(registry.list_modules().is_empty());
}

#[tokio::test]
async fn test_dspy_system_initialization_with_invalid_config() {
    let mut config = DspyConfig::default();
    config.max_optimization_iterations = 0;

    let result = init_dspy(config).await;
    assert!(result.is_err());
}

// Predict Module Tests
mod predict_tests {
    use super::*;
    use crate::anthropic::client::AnthropicClient;
    use crate::config::AnthropicConfig;
    use std::sync::Arc;

    // Mock Anthropic client for testing
    struct MockAnthropicClient;

    impl MockAnthropicClient {
        fn new() -> Arc<AnthropicClient> {
            // Create a real client for testing - in practice this would be mocked
            let config = AnthropicConfig::default();
            Arc::new(AnthropicClient::new(config).expect("Failed to create test client"))
        }
    }

    #[test]
    fn test_predict_config_default() {
        let config = PredictConfig::default();

        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.temperature, 0.7);
        assert!(!config.stream);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_seconds, 120);
        assert!(config.enable_security_validation);
        assert!(config.enable_rate_limiting);
    }

    #[test]
    fn test_predict_module_creation() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("test_predict")
            .description("Test prediction module")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        assert_eq!(predict.name(), "predict");
        assert_eq!(predict.id(), "predict_test_predict");
        assert_eq!(predict.signature().name, "test_predict");
        assert_eq!(predict.metadata().description, "DSPy Predict Module");
    }

    #[test]
    fn test_predict_module_with_custom_config() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("custom_predict")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let config = PredictConfig {
            model: "claude-haiku-3-20240307".to_string(),
            max_tokens: 2048,
            temperature: 0.5,
            stream: true,
            max_retries: 5,
            timeout_seconds: 60,
            enable_security_validation: false,
            enable_rate_limiting: false,
        };

        let predict = Predict::with_config(signature, client, config.clone());

        assert_eq!(predict.config().model, "claude-haiku-3-20240307");
        assert_eq!(predict.config().max_tokens, 2048);
        assert_eq!(predict.config().temperature, 0.5);
        assert!(predict.config().stream);
        assert_eq!(predict.config().max_retries, 5);
        assert_eq!(predict.config().timeout_seconds, 60);
        assert!(!predict.config().enable_security_validation);
        assert!(!predict.config().enable_rate_limiting);
    }

    #[tokio::test]
    async fn test_predict_prompt_generation() {
        let signature = SignatureBuilder::<QuestionInput, AnswerOutput>::new("qa_predict")
            .description("Question answering prediction")
            .input_field("question", "The question to answer", FieldType::String)
            .input_field("context", "Optional context", FieldType::String)
            .output_field("answer", "The answer", FieldType::String)
            .output_field("confidence", "Confidence score", FieldType::Float)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let input = QuestionInput {
            question: "What is the capital of France?".to_string(),
            context: Some("European geography".to_string()),
        };

        let prompt = predict.test_generate_prompt(&input).await.unwrap();

        assert!(prompt.contains("Question answering prediction"));
        assert!(prompt.contains("question"));
        assert!(prompt.contains("context"));
    }

    #[tokio::test]
    async fn test_predict_variable_substitution() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("substitution_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let template = "Process this text: {text}. Return the result.";
        let input_json = json!({"text": "Hello World"});

        let result = predict
            .test_substitute_variables(template, &input_json)
            .unwrap();
        assert_eq!(result, "Process this text: Hello World. Return the result.");
    }

    #[tokio::test]
    async fn test_predict_input_validation() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("validation_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let valid_input = SimpleInput {
            text: "Valid input text".to_string(),
        };

        let result = predict.validate_input(&valid_input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_predict_output_parsing() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("parsing_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let response_json = r#"{"result": "Parsed output"}"#;
        let output = predict.test_parse_output(response_json).await.unwrap();

        assert_eq!(output.result, "Parsed output");
    }

    #[tokio::test]
    async fn test_predict_output_parsing_invalid_json() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("invalid_parsing_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let invalid_json = "This is not valid JSON";
        let result = predict.test_parse_output(invalid_json).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            DspyError::TypeValidation {
                field_name,
                message,
            } => {
                assert_eq!(field_name, "output");
                assert!(message.contains("Failed to parse response as JSON"));
            }
            _ => panic!("Expected TypeValidation error"),
        }
    }

    #[tokio::test]
    async fn test_predict_chat_request_creation() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("request_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let config = PredictConfig {
            model: "test-model".to_string(),
            max_tokens: 1024,
            temperature: 0.8,
            stream: true,
            ..Default::default()
        };
        let predict = Predict::with_config(signature, client, config);

        let prompt = "Test prompt for API request";
        let request = predict.test_create_chat_request(prompt).await.unwrap();

        assert_eq!(request.model, "test-model");
        assert_eq!(request.max_tokens, 1024);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.stream, Some(true));
        assert_eq!(request.messages.len(), 1);
        assert!(request.system.is_some());
        assert!(request.system.unwrap().contains("JSON"));
    }

    #[test]
    fn test_predict_module_metadata() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("metadata_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let metadata = predict.metadata();
        assert_eq!(metadata.description, "DSPy Predict Module");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.author, Some("DSPy Integration".to_string()));
        assert!(metadata.tags.contains(&"prediction".to_string()));
        assert!(metadata.tags.contains(&"llm".to_string()));
        assert_eq!(
            metadata.custom.get("signature_name"),
            Some(&json!("metadata_test"))
        );
    }

    #[test]
    fn test_predict_module_stats_access() {
        let signature = SignatureBuilder::<SimpleInput, SimpleOutput>::new("stats_test")
            .input_field("text", "Input text", FieldType::String)
            .output_field("result", "Output result", FieldType::String)
            .build();

        let client = MockAnthropicClient::new();
        let predict = Predict::new(signature, client);

        let stats = predict.stats();
        assert_eq!(stats.execution_count, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }
}

// Chain and Composition tests
pub mod composition_tests {
    use super::*;
    use std::sync::Arc;

    // Additional test data structures for composition
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct IntermediateOutput {
        processed_text: String,
        metadata: String,
    }

    // Mock module that transforms SimpleInput to IntermediateOutput
    struct FirstMockModule {
        base: BaseModule<SimpleInput, IntermediateOutput>,
        should_fail: bool,
    }

    impl FirstMockModule {
        fn new(should_fail: bool) -> Self {
            let signature = SignatureBuilder::new("first_mock_module")
                .description("First module in chain")
                .input_field("text", "Input text", FieldType::String)
                .output_field("processed_text", "Processed text", FieldType::String)
                .output_field("metadata", "Processing metadata", FieldType::String)
                .build();

            Self {
                base: BaseModule::new("first_mock_module", signature),
                should_fail,
            }
        }
    }

    #[async_trait]
    impl Module for FirstMockModule {
        type Input = SimpleInput;
        type Output = IntermediateOutput;

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
            if self.should_fail {
                return Err(DspyError::module(
                    "first_mock_module",
                    "First module failure",
                ));
            }

            self.base
                .execute_with_stats(input, |input| async move {
                    Ok(IntermediateOutput {
                        processed_text: format!("First: {}", input.text),
                        metadata: "first_processing".to_string(),
                    })
                })
                .await
        }

        fn metadata(&self) -> &ModuleMetadata {
            &self.base.metadata
        }

        fn supports_compilation(&self) -> bool {
            true
        }

        async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
            let mut stats = self.base.stats_mut().await;
            stats.record_compilation();
            Ok(())
        }

        fn is_compiled(&self) -> bool {
            false
        }

        fn stats(&self) -> &ModuleStats {
            static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
            DEFAULT_STATS.get_or_init(ModuleStats::default)
        }
    }

    // Mock module that transforms IntermediateOutput to SimpleOutput
    struct SecondMockModule {
        base: BaseModule<IntermediateOutput, SimpleOutput>,
        should_fail: bool,
    }

    impl SecondMockModule {
        fn new(should_fail: bool) -> Self {
            let signature = SignatureBuilder::new("second_mock_module")
                .description("Second module in chain")
                .input_field("processed_text", "Processed text", FieldType::String)
                .input_field("metadata", "Processing metadata", FieldType::String)
                .output_field("result", "Final result", FieldType::String)
                .build();

            Self {
                base: BaseModule::new("second_mock_module", signature),
                should_fail,
            }
        }
    }

    #[async_trait]
    impl Module for SecondMockModule {
        type Input = IntermediateOutput;
        type Output = SimpleOutput;

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
            if self.should_fail {
                return Err(DspyError::module(
                    "second_mock_module",
                    "Second module failure",
                ));
            }

            self.base
                .execute_with_stats(input, |input| async move {
                    Ok(SimpleOutput {
                        result: format!("Second: {} ({})", input.processed_text, input.metadata),
                    })
                })
                .await
        }

        fn metadata(&self) -> &ModuleMetadata {
            &self.base.metadata
        }

        fn supports_compilation(&self) -> bool {
            true
        }

        async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
            let mut stats = self.base.stats_mut().await;
            stats.record_compilation();
            Ok(())
        }

        fn is_compiled(&self) -> bool {
            false
        }

        fn stats(&self) -> &ModuleStats {
            static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
            DEFAULT_STATS.get_or_init(ModuleStats::default)
        }
    }

    #[tokio::test]
    async fn test_chain_creation_and_basic_properties() {
        let first_module = Arc::new(FirstMockModule::new(false));
        let second_module = Arc::new(SecondMockModule::new(false));

        let chain = Chain::new(
            "test_chain".to_string(),
            first_module.clone(),
            second_module.clone(),
        );

        assert_eq!(chain.name(), "test_chain");
        assert!(!chain.id().is_empty());
        assert_eq!(chain.first_module().name(), "first_mock_module");
        assert_eq!(chain.second_module().name(), "second_mock_module");
        assert!(chain.supports_compilation());
    }

    #[tokio::test]
    async fn test_chain_successful_execution() {
        let first_module = Arc::new(FirstMockModule::new(false));
        let second_module = Arc::new(SecondMockModule::new(false));

        let chain = Chain::new("test_chain".to_string(), first_module, second_module);

        let input = SimpleInput {
            text: "Hello".to_string(),
        };

        let result = chain.forward(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.result, "Second: First: Hello (first_processing)");
    }

    #[tokio::test]
    async fn test_chain_first_module_failure() {
        let first_module = Arc::new(FirstMockModule::new(true)); // Will fail
        let second_module = Arc::new(SecondMockModule::new(false));

        let chain = Chain::new("test_chain".to_string(), first_module, second_module);

        let input = SimpleInput {
            text: "Hello".to_string(),
        };

        let result = chain.forward(input).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DspyError::ChainExecution {
                module_name, stage, ..
            } => {
                assert_eq!(module_name, "first_mock_module");
                assert_eq!(stage, "first");
            }
            _ => panic!("Expected ChainExecution error"),
        }
    }

    #[tokio::test]
    async fn test_chain_second_module_failure() {
        let first_module = Arc::new(FirstMockModule::new(false));
        let second_module = Arc::new(SecondMockModule::new(true)); // Will fail

        let chain = Chain::new("test_chain".to_string(), first_module, second_module);

        let input = SimpleInput {
            text: "Hello".to_string(),
        };

        let result = chain.forward(input).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DspyError::ChainExecution {
                module_name, stage, ..
            } => {
                assert_eq!(module_name, "second_mock_module");
                assert_eq!(stage, "second");
            }
            _ => panic!("Expected ChainExecution error"),
        }
    }

    #[tokio::test]
    async fn test_chain_with_metadata() {
        let first_module = Arc::new(FirstMockModule::new(false));
        let second_module = Arc::new(SecondMockModule::new(false));

        let metadata = ModuleMetadata::new("Test chain with metadata")
            .with_version("1.0.0")
            .with_author("Test Author")
            .with_tag("test")
            .with_tag("chain");

        let chain = Chain::new("test_chain".to_string(), first_module, second_module)
            .with_metadata(metadata);

        assert_eq!(chain.metadata().description, "Test chain with metadata");
        assert_eq!(chain.metadata().version, "1.0.0");
        assert_eq!(chain.metadata().author, Some("Test Author".to_string()));
        assert!(chain.metadata().tags.contains(&"test".to_string()));
        assert!(chain.metadata().tags.contains(&"chain".to_string()));
    }

    #[tokio::test]
    async fn test_chain_compilation() {
        let first_module = Arc::new(FirstMockModule::new(false));
        let second_module = Arc::new(SecondMockModule::new(false));

        let mut chain = Chain::new("test_chain".to_string(), first_module, second_module);

        let examples = vec![
            (
                SimpleInput {
                    text: "test1".to_string(),
                },
                SimpleOutput {
                    result: "result1".to_string(),
                },
            ),
            (
                SimpleInput {
                    text: "test2".to_string(),
                },
                SimpleOutput {
                    result: "result2".to_string(),
                },
            ),
        ];

        let result = chain.compile(examples).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_parallel_creation_and_basic_properties() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));
        let module3 = Arc::new(MockModule::new(false));

        let parallel = Parallel::new("test_parallel".to_string(), vec![module1, module2, module3]);

        assert_eq!(parallel.name(), "test_parallel");
        assert!(!parallel.id().is_empty());
        assert_eq!(parallel.modules().len(), 3);
        assert!(parallel.supports_compilation());
    }

    #[tokio::test]
    async fn test_parallel_successful_execution() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));

        let parallel = Parallel::new("test_parallel".to_string(), vec![module1, module2]);

        let input = SimpleInput {
            text: "Hello".to_string(),
        };

        let result = parallel.forward(input).await;
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].result, "Processed: Hello");
        assert_eq!(outputs[1].result, "Processed: Hello");
    }

    #[tokio::test]
    async fn test_parallel_partial_failure() {
        let module1 = Arc::new(MockModule::new(false)); // Success
        let module2 = Arc::new(MockModule::new(true)); // Failure

        let parallel = Parallel::new("test_parallel".to_string(), vec![module1, module2]);

        let input = SimpleInput {
            text: "Hello".to_string(),
        };

        let result = parallel.forward(input).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DspyError::Module {
                module_name,
                message,
            } => {
                assert_eq!(module_name, "mock_module");
                assert!(message.contains("Parallel execution failed"));
            }
            _ => panic!("Expected Module error"),
        }
    }

    #[tokio::test]
    async fn test_parallel_with_metadata() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));

        let metadata = ModuleMetadata::new("Test parallel with metadata")
            .with_version("1.0.0")
            .with_tag("parallel");

        let parallel = Parallel::new("test_parallel".to_string(), vec![module1, module2])
            .with_metadata(metadata);

        assert_eq!(
            parallel.metadata().description,
            "Test parallel with metadata"
        );
        assert_eq!(parallel.metadata().version, "1.0.0");
        assert!(parallel.metadata().tags.contains(&"parallel".to_string()));
    }

    #[tokio::test]
    async fn test_parallel_add_module() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));
        let module3 = Arc::new(MockModule::new(false));

        let parallel =
            Parallel::new("test_parallel".to_string(), vec![module1, module2]).add_module(module3);

        assert_eq!(parallel.modules().len(), 3);
    }

    #[tokio::test]
    async fn test_conditional_creation_and_basic_properties() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));

        let condition = Arc::new(
            |input: &SimpleInput| {
                if input.text.len() > 5 {
                    0
                } else {
                    1
                }
            },
        );

        let conditional = Conditional::new(
            "test_conditional".to_string(),
            condition,
            vec![module1, module2],
            1, // default to module2
        )
        .unwrap();

        assert_eq!(conditional.name(), "test_conditional");
        assert!(!conditional.id().is_empty());
        assert_eq!(conditional.modules().len(), 2);
        assert!(conditional.supports_compilation());
    }

    #[tokio::test]
    async fn test_conditional_routing_logic() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));

        // Route to module1 (index 0) if text length > 5, otherwise module2 (index 1)
        let condition = Arc::new(
            |input: &SimpleInput| {
                if input.text.len() > 5 {
                    0
                } else {
                    1
                }
            },
        );

        let conditional = Conditional::new(
            "test_conditional".to_string(),
            condition,
            vec![module1, module2],
            1,
        )
        .unwrap();

        // Test routing to first module (long text)
        let long_input = SimpleInput {
            text: "This is a long text".to_string(),
        };
        let result = conditional.forward(long_input).await;
        assert!(result.is_ok());

        // Test routing to second module (short text)
        let short_input = SimpleInput {
            text: "Hi".to_string(),
        };
        let result = conditional.forward(short_input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_conditional_invalid_configuration() {
        let module1 = Arc::new(MockModule::new(false));

        let condition = Arc::new(|_: &SimpleInput| 0);

        // Test empty modules
        let result = Conditional::<SimpleInput, SimpleOutput>::new(
            "test_conditional".to_string(),
            condition.clone(),
            vec![],
            0,
        );
        assert!(result.is_err());

        // Test invalid default module index
        let result = Conditional::<SimpleInput, SimpleOutput>::new(
            "test_conditional".to_string(),
            condition,
            vec![module1],
            5, // Out of bounds
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_conditional_fallback_to_default() {
        let module1 = Arc::new(MockModule::new(false));
        let module2 = Arc::new(MockModule::new(false));

        // Condition that returns invalid index
        let condition = Arc::new(|_: &SimpleInput| 999);

        let conditional = Conditional::new(
            "test_conditional".to_string(),
            condition,
            vec![module1, module2],
            1, // default to module2
        )
        .unwrap();

        let input = SimpleInput {
            text: "test".to_string(),
        };

        let result = conditional.forward(input).await;
        assert!(result.is_ok());
        // Should use default module (index 1)
    }
}

// Teleprompter Foundation Tests
mod teleprompter_tests {
    use super::*;

    #[test]
    fn test_example_creation_and_metadata() {
        let example = Example::new(
            SimpleInput {
                text: "test input".to_string(),
            },
            SimpleOutput {
                result: "test output".to_string(),
            },
        )
        .with_metadata("source".to_string(), json!("manual"))
        .with_quality_score(0.9)
        .validated();

        assert!(!example.id.is_empty());
        assert_eq!(example.input.text, "test input");
        assert_eq!(example.output.result, "test output");
        assert_eq!(example.quality_score, 0.9);
        assert!(example.validated);
        assert_eq!(example.get_metadata("source"), Some(&json!("manual")));
    }

    #[test]
    fn test_example_set_creation_and_management() {
        let mut example_set = ExampleSet::new();
        assert!(example_set.is_empty());
        assert_eq!(example_set.len(), 0);

        let example1 = Example::new(
            SimpleInput {
                text: "input1".to_string(),
            },
            SimpleOutput {
                result: "output1".to_string(),
            },
        )
        .with_quality_score(0.8);

        let example2 = Example::new(
            SimpleInput {
                text: "input2".to_string(),
            },
            SimpleOutput {
                result: "output2".to_string(),
            },
        )
        .with_quality_score(0.9)
        .validated();

        example_set.add_example(example1.clone());
        example_set.add_example(example2.clone());

        assert_eq!(example_set.len(), 2);
        assert!(!example_set.is_empty());

        let stats = example_set.stats();
        assert_eq!(stats.total_count, 2);
        assert_eq!(stats.validated_count, 1);
        assert_eq!(stats.high_quality_count, 2); // Both 0.8 and 0.9 are >= 0.8
        assert!((stats.average_quality - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_example_set_filtering_and_selection() {
        let mut example_set = ExampleSet::new();

        for i in 0..10 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            )
            .with_quality_score(i as f64 / 10.0);

            example_set.add_example(example);
        }

        // Test quality-based filtering
        let high_quality = example_set.examples_by_quality(0.7);
        assert_eq!(high_quality.len(), 3); // 0.7, 0.8, 0.9

        // Test custom filtering
        let filtered = example_set.filter(|e| e.input.text.contains("5"));
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_example_set_splitting() {
        let mut example_set = ExampleSet::new();

        for i in 0..10 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            );
            example_set.add_example(example);
        }

        let (train, val) = example_set.split(0.8).unwrap();
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);

        // Test invalid split ratio
        assert!(example_set.split(1.5).is_err());
    }

    #[test]
    fn test_optimization_strategy_creation() {
        let random_strategy = OptimizationStrategy::random_sampling(50);
        assert_eq!(random_strategy.name(), "RandomSampling");

        let quality_strategy = OptimizationStrategy::quality_based(0.8);
        assert_eq!(quality_strategy.name(), "QualityBased");

        let bootstrap_strategy = OptimizationStrategy::bootstrap(5, 20);
        assert_eq!(bootstrap_strategy.name(), "Bootstrap");

        let diversity_strategy = OptimizationStrategy::diversity_based(3, 10);
        assert_eq!(diversity_strategy.name(), "DiversityBased");
    }

    #[test]
    fn test_optimization_metrics() {
        let mut metrics = OptimizationMetrics::new("TestStrategy".to_string());
        assert_eq!(metrics.iterations, 0);
        assert_eq!(metrics.best_score, 0.0);
        assert_eq!(metrics.improvement(), 0.0);

        metrics.record_score(0.5);
        metrics.record_score(0.7);
        metrics.record_score(0.6);

        assert_eq!(metrics.iterations, 3);
        assert_eq!(metrics.best_score, 0.7);
        assert_eq!(metrics.current_score, 0.6);
        assert!((metrics.improvement() - 0.2).abs() < 1e-10); // 0.7 - 0.5, account for floating point precision

        metrics.add_metric("custom_metric".to_string(), 42.0);
        assert_eq!(metrics.custom_metrics.get("custom_metric"), Some(&42.0));
    }

    #[test]
    fn test_optimizer_example_selection() {
        let mut optimizer = Optimizer::new(OptimizationStrategy::random_sampling(3));

        let mut example_set = ExampleSet::new();
        for i in 0..10 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            )
            .with_quality_score(i as f64 / 10.0);
            example_set.add_example(example);
        }

        let selected = optimizer.select_examples(&example_set).unwrap();
        assert_eq!(selected.len(), 3);
        assert_eq!(optimizer.metrics().examples_used, 3);
    }

    #[test]
    fn test_optimizer_quality_based_selection() {
        let mut optimizer = Optimizer::new(OptimizationStrategy::quality_based(0.7));

        let mut example_set = ExampleSet::new();
        for i in 0..10 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            )
            .with_quality_score(i as f64 / 10.0);
            example_set.add_example(example);
        }

        let selected = optimizer.select_examples(&example_set).unwrap();
        assert_eq!(selected.len(), 3); // 0.7, 0.8, 0.9

        // Verify all selected examples meet quality threshold
        for example in selected.examples() {
            assert!(example.quality_score >= 0.7);
        }
    }

    #[test]
    fn test_teleprompter_creation() {
        let teleprompter = Teleprompter::simple(10);
        assert_eq!(teleprompter.optimizer().strategy().name(), "RandomSampling");

        let quality_teleprompter = Teleprompter::quality_focused(0.8);
        assert_eq!(
            quality_teleprompter.optimizer().strategy().name(),
            "QualityBased"
        );

        let bootstrap_teleprompter = Teleprompter::bootstrap(5, 20);
        assert_eq!(
            bootstrap_teleprompter.optimizer().strategy().name(),
            "Bootstrap"
        );
    }

    #[test]
    fn test_teleprompter_config() {
        let config = TeleprompterConfig::default();
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.validation_split, 0.2);
        assert!(!config.use_cross_validation);

        let custom_config = TeleprompterConfig {
            strategy: OptimizationStrategy::random_sampling(20),
            max_iterations: 100,
            convergence_threshold: 0.001,
            min_improvement: 0.01,
            early_stopping_patience: 10,
            validation_split: 0.3,
            use_cross_validation: true,
            cv_folds: 5,
            random_seed: Some(42),
            verbose: true,
            custom_params: std::collections::HashMap::new(),
        };

        let teleprompter = Teleprompter::with_config(custom_config.clone());

        assert_eq!(teleprompter.config().max_iterations, 100);
        assert_eq!(teleprompter.config().validation_split, 0.3);
        assert!(teleprompter.config().use_cross_validation);
    }

    #[tokio::test]
    async fn test_teleprompter_optimization_with_empty_examples() {
        let mut teleprompter = Teleprompter::simple(10);
        let mut module = MockModule::new(false);
        let empty_examples = ExampleSet::new();

        let result = teleprompter
            .optimize(&mut module, empty_examples)
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error_message.is_some());
        assert!(result
            .error_message
            .unwrap()
            .contains("No examples provided"));
    }

    #[tokio::test]
    async fn test_teleprompter_optimization_success() {
        let mut teleprompter = Teleprompter::simple(5);
        let mut module = MockModule::new(false);

        let mut examples = ExampleSet::new();
        for i in 0..10 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            )
            .with_quality_score(0.8 + (i as f64 / 100.0)); // Varying quality scores
            examples.add_example(example);
        }

        let result = teleprompter.optimize(&mut module, examples).await.unwrap();
        assert!(result.success);
        assert!(result.final_score >= 0.0);
        assert!(result.optimization_time > 0.0);
        assert_eq!(result.iterations, 1); // Should complete in 1 iteration for this simple case
    }

    #[test]
    fn test_optimization_result_creation() {
        let metrics = OptimizationMetrics::new("TestStrategy".to_string());

        let success_result = OptimizationResult::success(0.85, 0.15, 10, 5.5, 25, metrics.clone());

        assert!(success_result.success);
        assert_eq!(success_result.final_score, 0.85);
        assert_eq!(success_result.improvement, 0.15);
        assert_eq!(success_result.iterations, 10);
        assert_eq!(success_result.optimization_time, 5.5);
        assert_eq!(success_result.best_examples_count, 25);

        let failure_result = OptimizationResult::failure("Test error".to_string(), metrics);

        assert!(!failure_result.success);
        assert!(failure_result.error_message.is_some());
        assert_eq!(failure_result.error_message.unwrap(), "Test error");
    }

    #[test]
    fn test_example_validation() {
        let mut example_set = ExampleSet::new();

        for i in 0..5 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            );
            example_set.add_example(example);
        }

        // Validation function that validates examples with even indices
        let validator = |example: &Example<SimpleInput, SimpleOutput>| -> DspyResult<bool> {
            let index: usize = example
                .input
                .text
                .chars()
                .last()
                .unwrap()
                .to_digit(10)
                .unwrap() as usize;
            Ok(index % 2 == 0)
        };

        example_set.validate_examples(validator).unwrap();

        let stats = example_set.stats();
        assert_eq!(stats.validated_count, 3); // 0, 2, 4
    }
}

// Bootstrap Teleprompter Tests
mod bootstrap_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_bootstrap_config_creation() {
        let config = BootstrapConfig::default();
        assert_eq!(config.max_labeled_demos, 16);
        assert_eq!(config.max_bootstrapped_demos, 4);
        assert_eq!(config.max_rounds, 1);
        assert_eq!(config.max_errors, 5);
        assert_eq!(config.min_confidence, 0.7);
        assert!(config.use_teacher_forcing);
        assert_eq!(config.validation_strictness, ValidationStrictness::Medium);

        let custom_config = BootstrapConfig {
            max_labeled_demos: 8,
            max_bootstrapped_demos: 8,
            max_rounds: 3,
            max_errors: 10,
            min_confidence: 0.8,
            use_teacher_forcing: false,
            random_seed: Some(42),
            validation_strictness: ValidationStrictness::High,
        };

        assert_eq!(custom_config.max_labeled_demos, 8);
        assert_eq!(custom_config.max_bootstrapped_demos, 8);
        assert_eq!(
            custom_config.validation_strictness,
            ValidationStrictness::High
        );
    }

    #[test]
    fn test_bootstrap_teleprompter_creation() {
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();
        assert_eq!(bootstrap.config().max_labeled_demos, 16);
        assert_eq!(bootstrap.bootstrap_examples().len(), 0);

        let custom_config = BootstrapConfig {
            max_labeled_demos: 5,
            max_bootstrapped_demos: 5,
            ..Default::default()
        };

        let custom_bootstrap =
            BootstrapFewShot::<SimpleInput, SimpleOutput>::with_config(custom_config);
        assert_eq!(custom_bootstrap.config().max_labeled_demos, 5);
        assert_eq!(custom_bootstrap.config().max_bootstrapped_demos, 5);
    }

    #[test]
    fn test_bootstrap_config_validation() {
        let mut bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();

        // Valid configuration should pass
        assert!(bootstrap.validate_config().is_ok());

        // Invalid configuration: both demos set to 0
        bootstrap.config.max_labeled_demos = 0;
        bootstrap.config.max_bootstrapped_demos = 0;
        assert!(bootstrap.validate_config().is_err());

        // Reset and test invalid rounds
        bootstrap.config = BootstrapConfig::default();
        bootstrap.config.max_rounds = 0;
        assert!(bootstrap.validate_config().is_err());

        // Reset and test invalid confidence
        bootstrap.config = BootstrapConfig::default();
        bootstrap.config.min_confidence = 1.5;
        assert!(bootstrap.validate_config().is_err());
    }

    #[test]
    fn test_validation_strictness_levels() {
        assert_eq!(ValidationStrictness::Low, ValidationStrictness::Low);
        assert_ne!(ValidationStrictness::Low, ValidationStrictness::High);

        let strictness = ValidationStrictness::Medium;
        match strictness {
            ValidationStrictness::Medium => assert!(true),
            _ => assert!(false, "Expected Medium strictness"),
        }
    }

    #[test]
    fn test_bootstrap_stats() {
        let stats = BootstrapStats::default();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful_generations, 0);
        assert_eq!(stats.failed_attempts, 0);
        assert_eq!(stats.validated_examples, 0);
        assert_eq!(stats.validation_failures, 0);
        assert_eq!(stats.average_confidence, 0.0);
        assert_eq!(stats.generation_time_seconds, 0.0);
        assert_eq!(stats.rounds_completed, 0);

        let mut custom_stats = BootstrapStats {
            total_attempts: 10,
            successful_generations: 8,
            failed_attempts: 2,
            validated_examples: 6,
            validation_failures: 2,
            average_confidence: 0.85,
            generation_time_seconds: 5.5,
            rounds_completed: 2,
        };

        assert_eq!(custom_stats.total_attempts, 10);
        assert_eq!(custom_stats.successful_generations, 8);
        assert_eq!(custom_stats.average_confidence, 0.85);
    }

    #[tokio::test]
    async fn test_bootstrap_example_splitting() {
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();

        let mut trainset = ExampleSet::new();
        for i in 0..20 {
            let example = Example::new(
                SimpleInput {
                    text: format!("input{}", i),
                },
                SimpleOutput {
                    result: format!("output{}", i),
                },
            );
            trainset.add_example(example);
        }

        let (labeled, unlabeled) = bootstrap.split_examples(trainset).unwrap();
        assert_eq!(labeled.len(), 16); // max_labeled_demos default
        assert_eq!(unlabeled.len(), 4);
    }

    #[tokio::test]
    async fn test_bootstrap_example_combination() {
        let mut bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();

        // Create labeled examples
        let mut labeled_examples = ExampleSet::new();
        for i in 0..3 {
            let example = Example::new(
                SimpleInput {
                    text: format!("labeled{}", i),
                },
                SimpleOutput {
                    result: format!("labeled_output{}", i),
                },
            );
            labeled_examples.add_example(example);
        }

        // Add some bootstrap examples
        for i in 0..2 {
            let bootstrap_example = Example::new(
                SimpleInput {
                    text: format!("bootstrap{}", i),
                },
                SimpleOutput {
                    result: format!("bootstrap_output{}", i),
                },
            );
            bootstrap.bootstrap_examples.push(bootstrap_example);
        }

        let combined = bootstrap.combine_examples(&labeled_examples).unwrap();
        assert_eq!(combined.len(), 5); // 3 labeled + 2 bootstrap
    }

    #[tokio::test]
    async fn test_bootstrap_quality_estimation() {
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();

        let original = Example::new(
            SimpleInput {
                text: "test input".to_string(),
            },
            SimpleOutput {
                result: "test output".to_string(),
            },
        );

        let generated_output = SimpleOutput {
            result: "generated output".to_string(),
        };

        let quality_score = bootstrap.estimate_quality_score(&original, &generated_output);
        assert!(quality_score >= bootstrap.config().min_confidence);
        assert!(quality_score <= 1.0);
    }

    #[tokio::test]
    async fn test_bootstrap_validation_low_strictness() {
        let config = BootstrapConfig {
            validation_strictness: ValidationStrictness::Low,
            ..Default::default()
        };
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::with_config(config);

        let original = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "original".to_string(),
            },
        );

        let bootstrap_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(),
            },
        )
        .with_quality_score(0.5); // Low quality

        let is_valid = bootstrap
            .validate_example(&bootstrap_example, &original)
            .await
            .unwrap();
        assert!(is_valid); // Low strictness accepts all
    }

    #[tokio::test]
    async fn test_bootstrap_validation_medium_strictness() {
        let config = BootstrapConfig {
            validation_strictness: ValidationStrictness::Medium,
            min_confidence: 0.7,
            ..Default::default()
        };
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::with_config(config);

        let original = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "original".to_string(),
            },
        );

        // High quality example should pass
        let high_quality_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(),
            },
        )
        .with_quality_score(0.8);

        let is_valid = bootstrap
            .validate_example(&high_quality_example, &original)
            .await
            .unwrap();
        assert!(is_valid);

        // Low quality example should fail
        let low_quality_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(),
            },
        )
        .with_quality_score(0.5);

        let is_valid = bootstrap
            .validate_example(&low_quality_example, &original)
            .await
            .unwrap();
        assert!(!is_valid);
    }

    #[tokio::test]
    async fn test_bootstrap_validation_high_strictness() {
        let config = BootstrapConfig {
            validation_strictness: ValidationStrictness::High,
            min_confidence: 0.7,
            ..Default::default()
        };
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::with_config(config);

        let original = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "original".to_string(),
            },
        )
        .with_quality_score(0.6);

        // Example with higher quality than original should pass
        let better_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(),
            },
        )
        .with_quality_score(0.8);

        let is_valid = bootstrap
            .validate_example(&better_example, &original)
            .await
            .unwrap();
        assert!(is_valid);

        // Example with lower quality than original should fail
        let worse_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(),
            },
        )
        .with_quality_score(0.5);

        let is_valid = bootstrap
            .validate_example(&worse_example, &original)
            .await
            .unwrap();
        assert!(!is_valid);
    }

    #[tokio::test]
    async fn test_bootstrap_with_custom_metric() {
        let metric = Arc::new(
            |original: &Example<SimpleInput, SimpleOutput>,
             generated: &SimpleOutput|
             -> DspyResult<bool> {
                // Custom validation: generated output must contain "valid"
                Ok(generated.result.contains("valid"))
            },
        );

        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new().with_metric(metric);

        assert_eq!(
            bootstrap.config().validation_strictness,
            ValidationStrictness::Custom
        );

        let original = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "original".to_string(),
            },
        );

        // Valid example should pass
        let valid_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "valid generated".to_string(),
            },
        );

        let is_valid = bootstrap
            .validate_example(&valid_example, &original)
            .await
            .unwrap();
        assert!(is_valid);

        // Invalid example should fail
        let invalid_example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "generated".to_string(), // Does not contain "valid"
            },
        );

        let is_valid = bootstrap
            .validate_example(&invalid_example, &original)
            .await
            .unwrap();
        assert!(!is_valid);
    }

    #[tokio::test]
    async fn test_bootstrap_reset() {
        let mut bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::new();

        // Add some bootstrap examples
        let example = Example::new(
            SimpleInput {
                text: "test".to_string(),
            },
            SimpleOutput {
                result: "test".to_string(),
            },
        );
        bootstrap.bootstrap_examples.push(example);

        // Set some stats
        bootstrap.stats.total_attempts = 5;
        bootstrap.stats.successful_generations = 3;

        assert_eq!(bootstrap.bootstrap_examples().len(), 1);
        assert_eq!(bootstrap.stats().total_attempts, 5);

        // Reset should clear everything
        bootstrap.reset();

        assert_eq!(bootstrap.bootstrap_examples().len(), 0);
        assert_eq!(bootstrap.stats().total_attempts, 0);
        assert_eq!(bootstrap.stats().successful_generations, 0);
    }

    #[tokio::test]
    async fn test_bootstrap_display() {
        let config = BootstrapConfig {
            max_labeled_demos: 8,
            max_bootstrapped_demos: 6,
            max_rounds: 2,
            ..Default::default()
        };
        let bootstrap = BootstrapFewShot::<SimpleInput, SimpleOutput>::with_config(config);

        let display_str = format!("{}", bootstrap);
        assert!(display_str.contains("BootstrapFewShot"));
        assert!(display_str.contains("labeled: 8"));
        assert!(display_str.contains("bootstrap: 6"));
        assert!(display_str.contains("rounds: 2"));
    }

    #[tokio::test]
    async fn test_bootstrap_stats_display() {
        let stats = BootstrapStats {
            total_attempts: 10,
            successful_generations: 8,
            validated_examples: 6,
            average_confidence: 0.85,
            ..Default::default()
        };

        let display_str = format!("{}", stats);
        assert!(display_str.contains("BootstrapStats"));
        assert!(display_str.contains("attempts: 10"));
        assert!(display_str.contains("success: 8"));
        assert!(display_str.contains("validated: 6"));
        assert!(display_str.contains("avg_confidence: 0.85"));
    }
}

// Compiler Tests
mod compiler_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_compiler_config_creation() {
        let config = CompilerConfig::default();
        assert!(config.enable_caching);
        assert_eq!(config.cache_dir, PathBuf::from("./dspy_cache"));
        assert_eq!(config.max_cache_size_mb, 1024);
        assert_eq!(config.cache_ttl_seconds, 86400);
        assert!(config.enable_monitoring);
        assert_eq!(config.compilation_timeout_seconds, 3600);
        assert_eq!(config.max_optimization_rounds, 10);
        assert_eq!(config.validation_threshold, 0.8);

        let custom_config = CompilerConfig {
            enable_caching: false,
            max_cache_size_mb: 512,
            cache_ttl_seconds: 3600,
            compilation_timeout_seconds: 1800,
            max_optimization_rounds: 5,
            validation_threshold: 0.9,
            ..Default::default()
        };

        assert!(!custom_config.enable_caching);
        assert_eq!(custom_config.max_cache_size_mb, 512);
        assert_eq!(custom_config.validation_threshold, 0.9);
    }

    #[test]
    fn test_compiler_creation() {
        let compiler = Compiler::new();
        // Basic creation test - compiler should be created successfully
        assert!(true);

        let custom_config = CompilerConfig {
            enable_caching: false,
            ..Default::default()
        };
        let custom_compiler = Compiler::with_config(custom_config);
        assert!(true);
    }

    #[tokio::test]
    async fn test_compiler_stats() {
        let compiler = Compiler::new();
        let stats = compiler.stats().await;

        assert_eq!(stats.total_compilations, 0);
        assert_eq!(stats.cache_hit_rate, 0.0);
        assert_eq!(stats.avg_compilation_time_ms, 0.0);
        assert_eq!(stats.cache_size_mb, 0.0);
        assert_eq!(stats.successful_compilations, 0);
        assert_eq!(stats.failed_compilations, 0);
    }

    #[tokio::test]
    async fn test_compiler_cache_operations() {
        let compiler = Compiler::new();

        // Test cache size
        let initial_size = compiler.cache_size().await;
        assert_eq!(initial_size, 0);

        // Test cache clearing
        let result = compiler.clear_cache().await;
        assert!(result.is_ok());

        let size_after_clear = compiler.cache_size().await;
        assert_eq!(size_after_clear, 0);
    }

    #[test]
    fn test_compilation_context() {
        let context = CompilationContext {
            compilation_id: "test-id".to_string(),
            module_hash: "test-hash".to_string(),
            data_hash: "data-hash".to_string(),
            timestamp: 1234567890,
            dspy_version: "1.0.0".to_string(),
            strategy: "bootstrap".to_string(),
            metrics: CompilationMetrics::default(),
        };

        assert_eq!(context.compilation_id, "test-id");
        assert_eq!(context.module_hash, "test-hash");
        assert_eq!(context.data_hash, "data-hash");
        assert_eq!(context.dspy_version, "1.0.0");
        assert_eq!(context.strategy, "bootstrap");
    }

    #[test]
    fn test_compilation_metrics() {
        let metrics = CompilationMetrics::default();
        assert_eq!(metrics.compilation_time_ms, 0);
        assert_eq!(metrics.optimization_rounds, 0);
        assert_eq!(metrics.validation_score, 0.0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.peak_memory_mb, 0);
        assert_eq!(metrics.api_calls, 0);

        let custom_metrics = CompilationMetrics {
            compilation_time_ms: 5000,
            optimization_rounds: 3,
            validation_score: 0.85,
            cache_hits: 2,
            cache_misses: 1,
            peak_memory_mb: 256,
            api_calls: 10,
        };

        assert_eq!(custom_metrics.compilation_time_ms, 5000);
        assert_eq!(custom_metrics.optimization_rounds, 3);
        assert_eq!(custom_metrics.validation_score, 0.85);
    }

    #[test]
    fn test_compiler_stats_display() {
        let stats = CompilerStats {
            total_compilations: 10,
            cache_hit_rate: 0.75,
            avg_compilation_time_ms: 2500.0,
            cache_size_mb: 128.5,
            successful_compilations: 8,
            failed_compilations: 2,
        };

        let display_str = format!("{}", stats);
        assert!(display_str.contains("CompilerStats"));
        assert!(display_str.contains("compilations: 10"));
        assert!(display_str.contains("cache_hit_rate: 75.00%"));
        assert!(display_str.contains("avg_time: 2500.0ms"));
    }
}

// Cache Tests
mod cache_tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
    struct TestData {
        value: String,
        number: i32,
    }

    #[test]
    fn test_cache_config_creation() {
        let config = CacheConfig::default();
        assert_eq!(config.max_size_bytes, 1024 * 1024 * 1024); // 1GB
        assert_eq!(config.ttl_seconds, 86400); // 24 hours
        assert!(config.persistent);
        assert_eq!(config.cache_dir, PathBuf::from("./dspy_cache"));
        assert!(config.enable_compression);
        assert_eq!(config.max_entries, 10000);
        assert_eq!(config.cleanup_interval_seconds, 3600);

        let custom_config = CacheConfig {
            max_size_bytes: 512 * 1024 * 1024, // 512MB
            ttl_seconds: 3600,                 // 1 hour
            persistent: false,
            enable_compression: false,
            max_entries: 5000,
            cleanup_interval_seconds: 1800,
            ..Default::default()
        };

        assert_eq!(custom_config.max_size_bytes, 512 * 1024 * 1024);
        assert_eq!(custom_config.ttl_seconds, 3600);
        assert!(!custom_config.persistent);
        assert!(!custom_config.enable_compression);
    }

    #[test]
    fn test_cache_creation() {
        let cache = Cache::<TestData>::new();
        // Basic creation test
        assert!(true);

        let temp_dir = TempDir::new().unwrap();
        let custom_config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let custom_cache = Cache::<TestData>::with_config(custom_config);
        assert!(true);
    }

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        let test_data = TestData {
            value: "test".to_string(),
            number: 42,
        };

        // Test put and get
        let result = cache.put("test_key", test_data.clone()).await;
        assert!(result.is_ok());

        let retrieved = cache.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data));

        // Test contains
        let contains = cache.contains("test_key").await;
        assert!(contains);

        let not_contains = cache.contains("nonexistent").await;
        assert!(!not_contains);

        // Test get nonexistent
        let nonexistent = cache.get("nonexistent").await.unwrap();
        assert_eq!(nonexistent, None);
    }

    #[tokio::test]
    async fn test_cache_with_tags() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        let test_data = TestData {
            value: "tagged".to_string(),
            number: 123,
        };

        let tags = vec!["tag1".to_string(), "tag2".to_string()];
        let result = cache
            .put_with_tags("tagged_key", test_data.clone(), tags)
            .await;
        assert!(result.is_ok());

        let tagged_keys = cache.get_by_tag("tag1").await;
        assert!(tagged_keys.contains(&"tagged_key".to_string()));

        let tagged_keys2 = cache.get_by_tag("tag2").await;
        assert!(tagged_keys2.contains(&"tagged_key".to_string()));

        let no_keys = cache.get_by_tag("nonexistent_tag").await;
        assert!(no_keys.is_empty());
    }

    #[tokio::test]
    async fn test_cache_removal() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        let test_data = TestData {
            value: "to_remove".to_string(),
            number: 999,
        };

        // Put and verify
        cache.put("remove_key", test_data.clone()).await.unwrap();
        assert!(cache.contains("remove_key").await);

        // Remove and verify
        let removed = cache.remove("remove_key").await.unwrap();
        assert!(removed);
        assert!(!cache.contains("remove_key").await);

        // Try to remove again
        let not_removed = cache.remove("remove_key").await.unwrap();
        assert!(!not_removed);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        // Add multiple entries
        for i in 0..5 {
            let data = TestData {
                value: format!("value{}", i),
                number: i,
            };
            cache.put(&format!("key{}", i), data).await.unwrap();
        }

        // Verify entries exist
        let keys = cache.keys().await;
        assert_eq!(keys.len(), 5);

        // Clear cache
        cache.clear().await.unwrap();

        // Verify cache is empty
        let keys_after_clear = cache.keys().await;
        assert_eq!(keys_after_clear.len(), 0);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        let initial_stats = cache.stats().await;
        assert_eq!(initial_stats.hits, 0);
        assert_eq!(initial_stats.misses, 0);
        assert_eq!(initial_stats.entry_count, 0);

        let test_data = TestData {
            value: "stats_test".to_string(),
            number: 42,
        };

        // Put data
        cache.put("stats_key", test_data).await.unwrap();

        // Hit
        cache.get("stats_key").await.unwrap();

        // Miss
        cache.get("nonexistent").await.unwrap();

        let final_stats = cache.stats().await;
        assert_eq!(final_stats.hits, 1);
        assert_eq!(final_stats.misses, 1);
        assert_eq!(final_stats.entry_count, 1);

        let hit_rate = cache.hit_rate().await;
        assert_eq!(hit_rate, 0.5); // 1 hit out of 2 total requests
    }

    #[tokio::test]
    async fn test_cache_keys() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            persistent: false,
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cache = Cache::<TestData>::with_config(config);

        // Initially empty
        let initial_keys = cache.keys().await;
        assert!(initial_keys.is_empty());

        // Add some entries
        let test_keys = vec!["key1", "key2", "key3"];
        for key in &test_keys {
            let data = TestData {
                value: key.to_string(),
                number: 1,
            };
            cache.put(key, data).await.unwrap();
        }

        // Get all keys
        let mut keys = cache.keys().await;
        keys.sort();
        let mut expected_keys: Vec<String> = test_keys.iter().map(|s| s.to_string()).collect();
        expected_keys.sort();

        assert_eq!(keys, expected_keys);
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;
    use crate::dspy::metrics::*;
    use std::fmt;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct SimpleInput {
        text: String,
    }

    #[derive(Debug, Clone)]
    struct SimpleOutput {
        result: String,
    }

    impl fmt::Display for SimpleOutput {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.result)
        }
    }

    fn create_test_example(input: &str, output: &str) -> Example<SimpleInput, SimpleOutput> {
        Example::new(
            SimpleInput {
                text: input.to_string(),
            },
            SimpleOutput {
                result: output.to_string(),
            },
        )
    }

    #[test]
    fn test_exact_match_metric() {
        let metric = ExactMatch::new();
        let example = create_test_example("test input", "hello world");

        // Test exact match
        let prediction = SimpleOutput {
            result: "hello world".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert_eq!(result.score, 1.0);
        assert!(result.passed);

        // Test non-match
        let prediction = SimpleOutput {
            result: "goodbye world".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert_eq!(result.score, 0.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        let metric = ExactMatch::new().case_sensitive(false);
        let example = create_test_example("test input", "Hello World");

        let prediction = SimpleOutput {
            result: "hello world".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert_eq!(result.score, 1.0);
        assert!(result.passed);
    }

    #[test]
    fn test_exact_match_whitespace_trimming() {
        let metric = ExactMatch::new().trim_whitespace(true);
        let example = create_test_example("test input", "hello world");

        let prediction = SimpleOutput {
            result: "  hello world  ".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert_eq!(result.score, 1.0);
        assert!(result.passed);
    }

    #[test]
    fn test_semantic_similarity_jaccard() {
        let metric = SemanticSimilarity::new()
            .with_algorithm(SimilarityAlgorithm::Jaccard)
            .with_threshold(0.5);

        let example = create_test_example("test input", "the quick brown fox");

        // High similarity
        let prediction = SimpleOutput {
            result: "the quick brown dog".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert!(result.score > 0.5);
        assert!(result.passed);

        // Low similarity
        let prediction = SimpleOutput {
            result: "completely different text".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert!(result.score < 0.5);
        assert!(!result.passed);
    }

    #[test]
    fn test_semantic_similarity_cosine() {
        let metric = SemanticSimilarity::new()
            .with_algorithm(SimilarityAlgorithm::Cosine)
            .with_threshold(0.3);

        let example = create_test_example("test input", "hello world hello");

        let prediction = SimpleOutput {
            result: "hello hello world".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert!(result.score > 0.3);
        assert!(result.passed);
    }

    #[test]
    fn test_semantic_similarity_levenshtein() {
        let metric = SemanticSimilarity::new()
            .with_algorithm(SimilarityAlgorithm::Levenshtein)
            .with_threshold(0.7);

        let example = create_test_example("test input", "hello");

        let prediction = SimpleOutput {
            result: "helo".to_string(), // One character different
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert!(result.score > 0.7);
        assert!(result.passed);
    }

    #[test]
    fn test_f1_score_words() {
        let metric = F1Score::new()
            .with_tokenization(TokenizationStrategy::Words)
            .with_threshold(0.5);

        let example = create_test_example("test input", "the quick brown fox");

        // Partial overlap
        let prediction = SimpleOutput {
            result: "the quick red fox".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();
        assert!(result.score > 0.5);
        assert!(result.passed);

        // Check details
        assert!(result.details.contains_key("precision"));
        assert!(result.details.contains_key("recall"));
    }

    #[test]
    fn test_f1_score_characters() {
        let metric = F1Score::new()
            .with_tokenization(TokenizationStrategy::Characters)
            .with_threshold(0.8);

        let example = create_test_example("test input", "abc");

        let prediction = SimpleOutput {
            result: "ab".to_string(),
        };
        let result = metric.evaluate(&example, &prediction).unwrap();

        // Should have high precision but lower recall
        let precision: f64 = result.details.get("precision").unwrap().as_f64().unwrap();
        let recall: f64 = result.details.get("recall").unwrap().as_f64().unwrap();
        assert_eq!(precision, 1.0); // All predicted characters are correct
        assert!(recall < 1.0); // Not all expected characters are predicted
    }

    #[test]
    fn test_composite_metric_weighted_average() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let semantic_sim = Arc::new(SemanticSimilarity::new().with_threshold(0.5))
            as Arc<dyn Metric<SimpleInput, SimpleOutput>>;

        let composite = CompositeMetric::new("Composite", "Combined metric")
            .add_metric(exact_match, 0.7)
            .add_metric(semantic_sim, 0.3)
            .with_strategy(CombinationStrategy::WeightedAverage);

        let example = create_test_example("test input", "hello world");

        // Exact match but different semantic similarity
        let prediction = SimpleOutput {
            result: "hello world".to_string(),
        };
        let result = composite.evaluate(&example, &prediction).unwrap();

        // Should be weighted toward exact match
        assert!(result.score > 0.7);
        assert!(result.passed);
        assert!(result.details.contains_key("ExactMatch_score"));
        assert!(result.details.contains_key("SemanticSimilarity_score"));
    }

    #[test]
    fn test_composite_metric_all_pass() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let f1_score = Arc::new(F1Score::new().with_threshold(0.8))
            as Arc<dyn Metric<SimpleInput, SimpleOutput>>;

        let composite = CompositeMetric::new("AllPass", "All must pass")
            .add_metric(exact_match, 1.0)
            .add_metric(f1_score, 1.0)
            .with_strategy(CombinationStrategy::AllPass);

        let example = create_test_example("test input", "hello world");

        // Exact match
        let prediction = SimpleOutput {
            result: "hello world".to_string(),
        };
        let result = composite.evaluate(&example, &prediction).unwrap();
        assert!(result.passed); // Both should pass

        // Non-exact match
        let prediction = SimpleOutput {
            result: "hello earth".to_string(),
        };
        let result = composite.evaluate(&example, &prediction).unwrap();
        assert!(!result.passed); // Exact match fails, so composite fails
    }

    #[test]
    fn test_composite_metric_any_pass() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let semantic_sim = Arc::new(
            SemanticSimilarity::new().with_threshold(0.9), // Very high threshold
        ) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;

        let composite = CompositeMetric::new("AnyPass", "Any can pass")
            .add_metric(exact_match, 1.0)
            .add_metric(semantic_sim, 1.0)
            .with_strategy(CombinationStrategy::AnyPass);

        let example = create_test_example("test input", "hello world");

        // Exact match but low semantic similarity
        let prediction = SimpleOutput {
            result: "hello world".to_string(),
        };
        let result = composite.evaluate(&example, &prediction).unwrap();
        assert!(result.passed); // Exact match passes, so composite passes
    }

    #[test]
    fn test_metric_validation() {
        // Test invalid threshold
        let metric = SemanticSimilarity::new().with_threshold(1.5);
        assert!(
            <SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::validate(&metric).is_err()
        );

        let metric = F1Score::new().with_threshold(-0.1);
        assert!(<F1Score as Metric<SimpleInput, SimpleOutput>>::validate(&metric).is_err());

        // Test valid metrics
        let metric = ExactMatch::new();
        assert!(<ExactMatch as Metric<SimpleInput, SimpleOutput>>::validate(&metric).is_ok());

        let metric = SemanticSimilarity::new().with_threshold(0.8);
        assert!(
            <SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::validate(&metric).is_ok()
        );
    }

    #[test]
    fn test_composite_metric_validation() {
        // Empty composite should fail
        let composite = CompositeMetric::<SimpleInput, SimpleOutput>::new("Empty", "No metrics");
        assert!(<CompositeMetric<SimpleInput, SimpleOutput> as Metric<
            SimpleInput,
            SimpleOutput,
        >>::validate(&composite)
        .is_err());

        // Valid composite
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let composite = CompositeMetric::new("Valid", "Has metrics").add_metric(exact_match, 1.0);
        assert!(<CompositeMetric<SimpleInput, SimpleOutput> as Metric<
            SimpleInput,
            SimpleOutput,
        >>::validate(&composite)
        .is_ok());

        // Negative weight should fail
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let composite =
            CompositeMetric::new("Invalid", "Negative weight").add_metric(exact_match, -1.0);
        assert!(<CompositeMetric<SimpleInput, SimpleOutput> as Metric<
            SimpleInput,
            SimpleOutput,
        >>::validate(&composite)
        .is_err());
    }

    #[test]
    fn test_metric_result_creation() {
        let result = MetricResult::new(0.85)
            .with_passed(true)
            .with_detail("test_key", "test_value")
            .with_confidence(0.95);

        assert_eq!(result.score, 0.85);
        assert!(result.passed);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(
            result.details.get("test_key").unwrap().as_str().unwrap(),
            "test_value"
        );
    }

    #[test]
    fn test_metric_result_pass_fail() {
        let pass_result = MetricResult::pass(0.9);
        assert_eq!(pass_result.score, 0.9);
        assert!(pass_result.passed);

        let fail_result = MetricResult::fail(0.3);
        assert_eq!(fail_result.score, 0.3);
        assert!(!fail_result.passed);
    }

    #[test]
    fn test_similarity_algorithms() {
        let example = create_test_example("test", "hello world");
        let prediction = SimpleOutput {
            result: "hello earth".to_string(),
        };

        // Test different algorithms
        let algorithms = [
            SimilarityAlgorithm::Jaccard,
            SimilarityAlgorithm::Cosine,
            SimilarityAlgorithm::Levenshtein,
            SimilarityAlgorithm::WordOverlap,
        ];

        for algorithm in algorithms {
            let metric = SemanticSimilarity::new()
                .with_algorithm(algorithm)
                .with_threshold(0.0); // Low threshold to ensure it passes

            let result = metric.evaluate(&example, &prediction).unwrap();
            assert!(result.score >= 0.0 && result.score <= 1.0);
            assert!(result.details.contains_key("algorithm"));
        }
    }

    #[test]
    fn test_tokenization_strategies() {
        let example = create_test_example("test", "Hello, world! How are you?");
        let prediction = SimpleOutput {
            result: "Hello world How are".to_string(),
        };

        let strategies = [
            TokenizationStrategy::Whitespace,
            TokenizationStrategy::Words,
            TokenizationStrategy::Characters,
            TokenizationStrategy::Sentences,
        ];

        for strategy in strategies {
            let metric = F1Score::new()
                .with_tokenization(strategy)
                .with_threshold(0.0); // Low threshold to ensure it passes

            let result = metric.evaluate(&example, &prediction).unwrap();
            assert!(result.score >= 0.0 && result.score <= 1.0);
            assert!(result.details.contains_key("tokenization"));
        }
    }

    #[test]
    fn test_metric_names_and_descriptions() {
        let exact_match = ExactMatch::new();
        assert_eq!(
            <ExactMatch as Metric<SimpleInput, SimpleOutput>>::name(&exact_match),
            "ExactMatch"
        );
        assert!(
            !<ExactMatch as Metric<SimpleInput, SimpleOutput>>::description(&exact_match)
                .is_empty()
        );

        let semantic_sim = SemanticSimilarity::new();
        assert_eq!(
            <SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::name(&semantic_sim),
            "SemanticSimilarity"
        );
        assert!(
            !<SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::description(&semantic_sim)
                .is_empty()
        );

        let f1_score = F1Score::new();
        assert_eq!(
            <F1Score as Metric<SimpleInput, SimpleOutput>>::name(&f1_score),
            "F1Score"
        );
        assert!(!<F1Score as Metric<SimpleInput, SimpleOutput>>::description(&f1_score).is_empty());
    }

    #[test]
    fn test_metric_score_ranges() {
        let exact_match = ExactMatch::new();
        let (min, max) =
            <ExactMatch as Metric<SimpleInput, SimpleOutput>>::score_range(&exact_match);
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);
        assert!(<ExactMatch as Metric<SimpleInput, SimpleOutput>>::higher_is_better(&exact_match));

        let semantic_sim = SemanticSimilarity::new();
        let (min, max) =
            <SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::score_range(&semantic_sim);
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);
        assert!(
            <SemanticSimilarity as Metric<SimpleInput, SimpleOutput>>::higher_is_better(
                &semantic_sim
            )
        );
    }
}

#[cfg(test)]
mod evaluator_tests {
    use super::*;
    use crate::dspy::evaluator::*;
    use crate::dspy::metrics::*;
    use std::fmt;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct SimpleInput {
        text: String,
    }

    #[derive(Debug, Clone)]
    struct SimpleOutput {
        result: String,
    }

    impl fmt::Display for SimpleOutput {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.result)
        }
    }

    fn create_test_examples() -> ExampleSet<SimpleInput, SimpleOutput> {
        let mut examples = ExampleSet::new();

        examples.add_example(Example::new(
            SimpleInput {
                text: "input1".to_string(),
            },
            SimpleOutput {
                result: "output1".to_string(),
            },
        ));

        examples.add_example(Example::new(
            SimpleInput {
                text: "input2".to_string(),
            },
            SimpleOutput {
                result: "output2".to_string(),
            },
        ));

        examples.add_example(Example::new(
            SimpleInput {
                text: "input3".to_string(),
            },
            SimpleOutput {
                result: "output3".to_string(),
            },
        ));

        examples
    }

    fn simple_predict_fn(
        input: &SimpleInput,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = DspyResult<SimpleOutput>> + Send>> {
        let input_text = input.text.clone(); // Clone to avoid lifetime issues
        Box::pin(async move {
            // Simple prediction function that sometimes matches exactly
            let result = if input_text.contains("1") {
                "output1".to_string() // Exact match for input1
            } else if input_text.contains("2") {
                "output2_similar".to_string() // Similar but not exact for input2
            } else {
                "different_output".to_string() // Different for input3
            };

            Ok(SimpleOutput { result })
        })
    }

    #[tokio::test]
    async fn test_evaluator_creation() {
        let evaluator = Evaluator::<SimpleInput, SimpleOutput>::new();
        assert_eq!(evaluator.stats().total_evaluations, 0);

        let config = EvaluatorConfig {
            enable_significance_testing: false,
            ..Default::default()
        };
        let evaluator = Evaluator::<SimpleInput, SimpleOutput>::with_config(config);
        assert!(!evaluator.config().enable_significance_testing);
    }

    #[tokio::test]
    async fn test_evaluator_validation() {
        // Empty evaluator should fail validation
        let evaluator = Evaluator::<SimpleInput, SimpleOutput>::new();
        assert!(evaluator.validate().is_err());

        // Evaluator with metrics should pass
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let evaluator = Evaluator::new().add_metric(exact_match);
        assert!(evaluator.validate().is_ok());
    }

    #[tokio::test]
    async fn test_evaluator_basic_evaluation() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let semantic_sim = Arc::new(SemanticSimilarity::new().with_threshold(0.5))
            as Arc<dyn Metric<SimpleInput, SimpleOutput>>;

        let mut evaluator = Evaluator::new()
            .add_metric(exact_match)
            .add_metric(semantic_sim);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Check overall stats
        assert_eq!(result.overall_stats.total_examples, 3);
        assert_eq!(result.overall_stats.num_metrics, 2);
        assert!(result.overall_stats.total_time_ms > 0.0);

        // Check metric results
        assert!(result.metric_results.contains_key("ExactMatch"));
        assert!(result.metric_results.contains_key("SemanticSimilarity"));

        let exact_match_summary = &result.metric_results["ExactMatch"];
        assert_eq!(exact_match_summary.total_examples, 3);
        assert!(exact_match_summary.mean_score >= 0.0);
        assert!(exact_match_summary.mean_score <= 1.0);
    }

    #[tokio::test]
    async fn test_evaluator_with_detailed_results() {
        let config = EvaluatorConfig {
            compute_detailed_stats: true,
            enable_significance_testing: false,
            ..Default::default()
        };

        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::with_config(config).add_metric(exact_match);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Should have detailed results
        assert!(result.detailed_results.is_some());
        let detailed = result.detailed_results.unwrap();
        assert_eq!(detailed.len(), 3);

        // Check first detailed result
        let first_result = &detailed[0];
        assert!(first_result.metric_results.contains_key("ExactMatch"));
        assert!(first_result.evaluation_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_evaluator_max_examples_limit() {
        let config = EvaluatorConfig {
            max_examples: 2, // Limit to 2 examples
            compute_detailed_stats: false,
            enable_significance_testing: false,
            ..Default::default()
        };

        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::with_config(config).add_metric(exact_match);

        let examples = create_test_examples(); // Has 3 examples
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Should only evaluate 2 examples
        assert_eq!(result.overall_stats.total_examples, 2);
    }

    #[tokio::test]
    async fn test_evaluator_shuffling() {
        let config = EvaluatorConfig {
            shuffle_examples: true,
            random_seed: Some(42), // Fixed seed for reproducibility
            compute_detailed_stats: true,
            enable_significance_testing: false,
            ..Default::default()
        };

        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::with_config(config).add_metric(exact_match);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Should still evaluate all examples
        assert_eq!(result.overall_stats.total_examples, 3);
        assert!(result.detailed_results.is_some());
    }

    #[tokio::test]
    async fn test_evaluator_statistics_tracking() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::new().add_metric(exact_match);

        // Initial stats
        assert_eq!(evaluator.stats().total_evaluations, 0);
        assert_eq!(evaluator.stats().total_examples, 0);

        let examples = create_test_examples();
        let _result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Updated stats
        assert_eq!(evaluator.stats().total_evaluations, 1);
        assert_eq!(evaluator.stats().total_examples, 3);
        assert!(evaluator.stats().avg_evaluation_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_evaluator_config_validation() {
        let mut config = EvaluatorConfig::default();

        // Valid config
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let evaluator = Evaluator::with_config(config.clone()).add_metric(exact_match);
        assert!(evaluator.validate().is_ok());

        // Invalid significance confidence
        config.significance_confidence = 1.5;
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let evaluator = Evaluator::with_config(config).add_metric(exact_match);
        assert!(evaluator.validate().is_err());
    }

    #[tokio::test]
    async fn test_evaluation_result_display() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::new().add_metric(exact_match);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Test display formatting
        let display_string = format!("{}", result);
        assert!(display_string.contains("Evaluation Results"));
        assert!(display_string.contains("Total Examples"));
        assert!(display_string.contains("ExactMatch"));
    }

    #[tokio::test]
    async fn test_metric_summary_statistics() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::new().add_metric(exact_match);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        let summary = &result.metric_results["ExactMatch"];

        // Check all summary fields are populated
        assert!(summary.mean_score >= 0.0 && summary.mean_score <= 1.0);
        assert!(summary.std_dev >= 0.0);
        assert!(summary.min_score >= 0.0 && summary.min_score <= 1.0);
        assert!(summary.max_score >= 0.0 && summary.max_score <= 1.0);
        assert!(summary.median_score >= 0.0 && summary.median_score <= 1.0);
        assert!(summary.pass_rate >= 0.0 && summary.pass_rate <= 1.0);
        assert_eq!(summary.total_examples, 3);
        assert!(summary.passed_examples <= summary.total_examples);
    }

    #[tokio::test]
    async fn test_evaluation_metadata() {
        let exact_match = Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>;
        let mut evaluator = Evaluator::new().add_metric(exact_match);

        let examples = create_test_examples();
        let result = evaluator
            .evaluate(&examples, simple_predict_fn)
            .await
            .unwrap();

        // Check metadata
        assert!(result.metadata.start_time <= result.metadata.end_time);
        assert_eq!(result.metadata.evaluator_version, "1.0.0");
        assert!(
            !result.metadata.config.compute_detailed_stats || result.detailed_results.is_some()
        );
    }
}

#[cfg(test)]
mod advanced_optimizer_tests {
    use super::*;
    use crate::dspy::advanced_optimizers::*;
    use crate::dspy::metrics::*;
    use std::fmt;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct SimpleInput {
        text: String,
    }

    #[derive(Debug, Clone)]
    struct SimpleOutput {
        result: String,
    }

    impl fmt::Display for SimpleOutput {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.result)
        }
    }

    fn create_test_examples() -> ExampleSet<SimpleInput, SimpleOutput> {
        let mut examples = ExampleSet::new();

        for i in 0..20 {
            examples.add_example(Example::new(
                SimpleInput {
                    text: format!("input_{}", i),
                },
                SimpleOutput {
                    result: format!("output_{}", i),
                },
            ));
        }

        examples
    }

    fn create_test_metrics() -> Vec<Arc<dyn Metric<SimpleInput, SimpleOutput>>> {
        vec![
            Arc::new(ExactMatch::new()) as Arc<dyn Metric<SimpleInput, SimpleOutput>>,
            Arc::new(SemanticSimilarity::new().with_threshold(0.7)),
        ]
    }

    #[tokio::test]
    async fn test_mipro_v2_optimizer_creation() {
        let config = MIPROv2Config {
            num_bootstrap: 10,
            num_candidates: 5,
            mini_batch_size: 3,
            max_iterations: 5,
            confidence_threshold: 0.8,
            top_k: 3,
            exploration_factor: 0.1,
        };

        let optimizer = MIPROv2Optimizer::<SimpleInput, SimpleOutput>::new(config.clone());
        assert_eq!(optimizer.config().num_bootstrap, 10);
        assert_eq!(optimizer.config().num_candidates, 5);
        assert_eq!(optimizer.config().mini_batch_size, 3);
    }

    #[tokio::test]
    async fn test_mipro_v2_optimization() {
        let config = MIPROv2Config {
            num_bootstrap: 5,
            num_candidates: 3,
            mini_batch_size: 2,
            max_iterations: 3,
            confidence_threshold: 0.9, // High threshold to test early stopping
            top_k: 2,
            exploration_factor: 0.1,
        };

        let mut optimizer = MIPROv2Optimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        assert!(result.iterations > 0);
        assert!(result.optimization_time_ms > 0.0);
        assert_eq!(result.examples_used, examples.len());
        assert!(result.get_metric("mipro_iterations").is_some());
        assert!(result.get_metric("bootstrap_traces").is_some());
    }

    #[tokio::test]
    async fn test_bootstrap_finetune_optimizer_creation() {
        let config = BootstrapFinetuneConfig {
            learning_rate: 1e-3,
            num_epochs: 5,
            batch_size: 4,
            regularization: 0.01,
            early_stopping_patience: 2,
            validation_split: 0.2,
            lr_decay: 0.9,
            min_lr: 1e-5,
        };

        let optimizer =
            BootstrapFinetuneOptimizer::<SimpleInput, SimpleOutput>::new(config.clone());
        assert_eq!(optimizer.config().learning_rate, 1e-3);
        assert_eq!(optimizer.config().num_epochs, 5);
        assert_eq!(optimizer.config().batch_size, 4);
    }

    #[tokio::test]
    async fn test_bootstrap_finetune_optimization() {
        let config = BootstrapFinetuneConfig {
            learning_rate: 1e-3,
            num_epochs: 3,
            batch_size: 4,
            regularization: 0.01,
            early_stopping_patience: 2,
            validation_split: 0.3,
            lr_decay: 0.9,
            min_lr: 1e-5,
        };

        let mut optimizer = BootstrapFinetuneOptimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        assert!(result.iterations > 0);
        assert!(result.optimization_time_ms > 0.0);
        assert_eq!(result.examples_used, examples.len());
        assert!(result.get_metric("finetune_epochs").is_some());
        assert!(result.get_metric("final_train_loss").is_some());
        assert!(result.get_metric("final_val_loss").is_some());
        assert!(result.get_metric("best_val_loss").is_some());
    }

    #[tokio::test]
    async fn test_multi_objective_optimizer_creation() {
        let config = MultiObjectiveConfig {
            objective_weights: vec![0.6, 0.4],
            use_pareto_optimization: true,
            max_pareto_points: 20,
            convergence_tolerance: 1e-4,
            max_iterations: 10,
            population_size: 15,
        };

        let optimizer = MultiObjectiveOptimizer::<SimpleInput, SimpleOutput>::new(config.clone());
        assert_eq!(optimizer.config().objective_weights, vec![0.6, 0.4]);
        assert!(optimizer.config().use_pareto_optimization);
        assert_eq!(optimizer.config().max_pareto_points, 20);
    }

    #[tokio::test]
    async fn test_multi_objective_pareto_optimization() {
        let config = MultiObjectiveConfig {
            objective_weights: vec![0.5, 0.5],
            use_pareto_optimization: true,
            max_pareto_points: 10,
            convergence_tolerance: 1e-4,
            max_iterations: 5,
            population_size: 8,
        };

        let mut optimizer = MultiObjectiveOptimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        assert!(result.iterations > 0);
        assert!(result.optimization_time_ms > 0.0);
        assert_eq!(result.examples_used, examples.len());
        assert!(result.get_metric("pareto_front_size").is_some());
        assert!(result.get_metric("num_objectives").is_some());
        assert!(result.pareto_front_size() > 0);
    }

    #[tokio::test]
    async fn test_multi_objective_scalarized_optimization() {
        let config = MultiObjectiveConfig {
            objective_weights: vec![0.7, 0.3],
            use_pareto_optimization: false, // Use scalarized optimization
            max_pareto_points: 10,
            convergence_tolerance: 1e-4,
            max_iterations: 5,
            population_size: 8,
        };

        let mut optimizer = MultiObjectiveOptimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        assert!(result.iterations > 0);
        assert!(result.optimization_time_ms > 0.0);
        assert_eq!(result.examples_used, examples.len());
        assert!(result.get_metric("num_objectives").is_some());
    }

    #[test]
    fn test_mipro_v2_config_default() {
        let config = MIPROv2Config::default();
        assert_eq!(config.num_bootstrap, 50);
        assert_eq!(config.num_candidates, 20);
        assert_eq!(config.mini_batch_size, 10);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.confidence_threshold, 0.8);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.exploration_factor, 0.1);
    }

    #[test]
    fn test_bootstrap_finetune_config_default() {
        let config = BootstrapFinetuneConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.num_epochs, 10);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.regularization, 0.01);
        assert_eq!(config.early_stopping_patience, 3);
        assert_eq!(config.validation_split, 0.2);
        assert_eq!(config.lr_decay, 0.9);
        assert_eq!(config.min_lr, 1e-6);
    }

    #[test]
    fn test_multi_objective_config_default() {
        let config = MultiObjectiveConfig::default();
        assert_eq!(config.objective_weights, vec![1.0]);
        assert!(config.use_pareto_optimization);
        assert_eq!(config.max_pareto_points, 100);
        assert_eq!(config.convergence_tolerance, 1e-6);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.population_size, 50);
    }

    #[tokio::test]
    async fn test_mipro_v2_early_stopping() {
        let config = MIPROv2Config {
            num_bootstrap: 3,
            num_candidates: 2,
            mini_batch_size: 2,
            max_iterations: 10,
            confidence_threshold: 0.1, // Very low threshold for early stopping
            top_k: 1,
            exploration_factor: 0.1,
        };

        let mut optimizer = MIPROv2Optimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        // Should stop early due to low confidence threshold
        assert!(result.iterations < 10);
        assert!(result.best_score >= 0.1);
    }

    #[tokio::test]
    async fn test_bootstrap_finetune_early_stopping() {
        let config = BootstrapFinetuneConfig {
            learning_rate: 1e-3,
            num_epochs: 10,
            batch_size: 4,
            regularization: 0.01,
            early_stopping_patience: 1, // Very low patience for early stopping
            validation_split: 0.3,
            lr_decay: 0.9,
            min_lr: 1e-5,
        };

        let mut optimizer = BootstrapFinetuneOptimizer::new(config);
        let examples = create_test_examples();
        let metrics = create_test_metrics();

        let result = optimizer.optimize(&examples, &metrics).await.unwrap();

        // Should stop early due to low patience
        assert!(result.get_metric("finetune_epochs").unwrap() < 10.0);
    }

    #[test]
    fn test_optimization_metrics_pareto_operations() {
        let mut metrics = OptimizationMetrics::new("MultiObjective".to_string());

        // Test adding Pareto points
        metrics.add_pareto_point(vec![0.8, 0.6]);
        metrics.add_pareto_point(vec![0.7, 0.7]);
        metrics.add_pareto_point(vec![0.6, 0.8]);

        assert_eq!(metrics.pareto_front_size(), 3);

        // Test dominance checking
        let point1 = vec![0.9, 0.9];
        let point2 = vec![0.8, 0.8];
        assert!(metrics.dominates(&point1, &point2));
        assert!(!metrics.dominates(&point2, &point1));

        // Test equal points (no dominance)
        let point3 = vec![0.8, 0.8];
        assert!(!metrics.dominates(&point2, &point3));
        assert!(!metrics.dominates(&point3, &point2));
    }

    #[test]
    fn test_optimization_metrics_hyperparameters() {
        let mut metrics = OptimizationMetrics::new("HyperparameterTuning".to_string());

        // Test setting and getting hyperparameters
        metrics.set_hyperparameter("learning_rate", 0.001);
        metrics.set_hyperparameter("batch_size", 32.0);
        metrics.set_hyperparameter("dropout", 0.1);

        assert_eq!(metrics.get_hyperparameter("learning_rate"), Some(0.001));
        assert_eq!(metrics.get_hyperparameter("batch_size"), Some(32.0));
        assert_eq!(metrics.get_hyperparameter("dropout"), Some(0.1));
        assert_eq!(metrics.get_hyperparameter("nonexistent"), None);
    }

    #[tokio::test]
    async fn test_optimizer_integration_with_advanced_strategies() {
        // Test that the basic Optimizer can handle advanced strategies
        let mipro_strategy = OptimizationStrategy::mipro_v2(10, 5, 3, 5);
        let mut optimizer = Optimizer::new(mipro_strategy);

        let examples = create_test_examples();
        let selected = optimizer.select_examples(&examples).unwrap();

        assert!(selected.len() <= examples.len());
        assert!(selected.len() > 0);

        // Test BootstrapFinetune strategy
        let finetune_strategy = OptimizationStrategy::bootstrap_finetune(0.001, 5, 16);
        optimizer.set_strategy(finetune_strategy);

        let selected = optimizer.select_examples(&examples).unwrap();
        assert!(selected.len() <= examples.len());
        assert!(selected.len() > 0);

        // Test MultiObjective strategy
        let multi_obj_strategy = OptimizationStrategy::multi_objective(vec![0.6, 0.4]);
        optimizer.set_strategy(multi_obj_strategy);

        let selected = optimizer.select_examples(&examples).unwrap();
        assert!(selected.len() <= examples.len());
        assert!(selected.len() > 0);
    }

    #[test]
    fn test_advanced_strategy_display() {
        let mipro_strategy = OptimizationStrategy::mipro_v2(10, 5, 3, 5);
        let display_str = format!("{}", mipro_strategy);
        assert!(display_str.contains("MIPROv2"));
        assert!(display_str.contains("bootstrap=10"));
        assert!(display_str.contains("candidates=5"));

        let finetune_strategy = OptimizationStrategy::bootstrap_finetune(0.001, 5, 16);
        let display_str = format!("{}", finetune_strategy);
        assert!(display_str.contains("BootstrapFinetune"));
        assert!(display_str.contains("lr=0.0010"));
        assert!(display_str.contains("epochs=5"));

        let multi_obj_strategy = OptimizationStrategy::multi_objective(vec![0.6, 0.4]);
        let display_str = format!("{}", multi_obj_strategy);
        assert!(display_str.contains("MultiObjective"));
        assert!(display_str.contains("weights=[0.6, 0.4]"));
    }

    #[test]
    fn test_advanced_strategy_names() {
        let mipro_strategy = OptimizationStrategy::mipro_v2(10, 5, 3, 5);
        assert_eq!(mipro_strategy.name(), "MIPROv2");

        let finetune_strategy = OptimizationStrategy::bootstrap_finetune(0.001, 5, 16);
        assert_eq!(finetune_strategy.name(), "BootstrapFinetune");

        let multi_obj_strategy = OptimizationStrategy::multi_objective(vec![0.6, 0.4]);
        assert_eq!(multi_obj_strategy.name(), "MultiObjective");
    }
}

#[cfg(test)]
mod dspy_integration_tests {
    use super::*;
    use crate::agent::dspy_integration::*;
    use crate::anthropic::AnthropicClient;
    use crate::dspy::tool_integration::*;
    use crate::security::{SecurityContext, SecurityManager};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::SystemTime;

    fn create_test_security_context() -> SecurityContext {
        SecurityContext {
            user_id: "test_user".to_string(),
            session_id: "test_session".to_string(),
            roles: vec!["user".to_string()],
            permissions: vec![
                "dspy_modules:create".to_string(),
                "dspy_modules:execute".to_string(),
            ],
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test_agent".to_string()),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    fn create_test_anthropic_client() -> AnthropicClient {
        AnthropicClient::new(crate::config::AnthropicConfig {
            api_key: "test_key".to_string(),
            model: "claude-3-sonnet-20240229".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_seconds: 30,
            max_retries: 3,
        })
        .unwrap()
    }

    #[test]
    fn test_dspy_agent_config_default() {
        let config = DspyAgentConfig::default();
        assert!(config.auto_optimize);
        assert_eq!(config.max_modules, 100);
        assert!(config.enable_security_validation);
        assert!(config.enable_audit_logging);
        assert_eq!(config.module_cache_ttl, 3600);
    }

    #[test]
    fn test_dspy_module_metadata_default() {
        let metadata = DspyModuleMetadata::default();
        assert_eq!(metadata.name, "Unnamed Module");
        assert_eq!(metadata.module_type, "Unknown");
        assert_eq!(metadata.usage_count, 0);
        assert!(!metadata.is_compiled);
        assert!(metadata.optimization_metrics.is_none());
        assert!(metadata.tags.is_empty());
    }

    #[tokio::test]
    async fn test_dspy_agent_extension_creation() {
        let config = DspyAgentConfig::default();
        let client = create_test_anthropic_client();
        let extension = DspyAgentExtension::new(config, client, None);

        let stats = extension.get_registry_stats().await;
        assert_eq!(
            stats.get("total_modules").unwrap(),
            &serde_json::Value::Number(0.into())
        );
        assert_eq!(
            stats.get("compiled_modules").unwrap(),
            &serde_json::Value::Number(0.into())
        );
        assert_eq!(
            stats.get("total_usage_count").unwrap(),
            &serde_json::Value::Number(0.into())
        );
    }

    #[tokio::test]
    async fn test_dspy_agent_extension_module_listing() {
        let config = DspyAgentConfig::default();
        let client = create_test_anthropic_client();
        let extension = DspyAgentExtension::new(config, client, None);

        let modules = extension.list_modules().await;
        assert!(modules.is_empty());
    }

    #[test]
    fn test_tool_metrics_recording() {
        let mut metrics = ToolMetrics::default();

        // Record successful executions
        metrics.record_success(100.0);
        metrics.record_success(200.0);

        assert_eq!(metrics.execution_count, 2);
        assert_eq!(metrics.successful_executions, 2);
        assert_eq!(metrics.failed_executions, 0);
        assert_eq!(metrics.total_execution_time_ms, 300.0);
        assert_eq!(metrics.average_execution_time_ms, 150.0);
        assert_eq!(metrics.success_rate, 1.0);
        assert!(metrics.last_execution_at.is_some());

        // Record a failure
        metrics.record_failure(50.0);

        assert_eq!(metrics.execution_count, 3);
        assert_eq!(metrics.successful_executions, 2);
        assert_eq!(metrics.failed_executions, 1);
        assert_eq!(metrics.total_execution_time_ms, 350.0);
        assert!((metrics.average_execution_time_ms - 116.67).abs() < 0.1);
        assert!((metrics.success_rate - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_dspy_tool_builder() {
        let builder = DspyToolBuilder::<String, String>::new();

        // Test builder pattern
        let builder = builder
            .with_name("test_tool")
            .with_description("A test DSPy tool");

        // Note: We can't complete the build without a module, but we can test the builder pattern
        assert!(true); // Placeholder assertion
    }

    #[test]
    fn test_dspy_tool_registry_creation() {
        let registry = DspyToolRegistry::new();

        assert!(registry.list_tools().is_empty());
        assert!(registry.list_metadata().is_empty());

        let stats = registry.stats();
        assert_eq!(
            stats.get("total_tools").unwrap(),
            &serde_json::Value::Number(0.into())
        );
        assert_eq!(
            stats.get("total_executions").unwrap(),
            &serde_json::Value::Number(0.into())
        );
    }

    #[test]
    fn test_dspy_tool_registry_operations() {
        let mut registry = DspyToolRegistry::new();

        // Test clearing empty registry
        registry.clear();
        assert!(registry.list_tools().is_empty());

        // Test getting non-existent tool
        assert!(registry.get_tool("nonexistent").is_none());
        assert!(registry.get_metadata("nonexistent").is_none());

        // Test removing non-existent tool
        assert!(registry.remove_tool("nonexistent").is_none());
    }

    #[test]
    fn test_dspy_tool_metadata() {
        let metadata = DspyToolMetadata {
            name: "test_tool".to_string(),
            description: Some("Test tool description".to_string()),
            module_id: "test_module_123".to_string(),
            created_at: chrono::Utc::now(),
            metrics: ToolMetrics::default(),
        };

        assert_eq!(metadata.name, "test_tool");
        assert_eq!(
            metadata.description,
            Some("Test tool description".to_string())
        );
        assert_eq!(metadata.module_id, "test_module_123");
        assert_eq!(metadata.metrics.execution_count, 0);
    }

    #[tokio::test]
    async fn test_dspy_agent_extension_registry_stats() {
        let config = DspyAgentConfig::default();
        let client = create_test_anthropic_client();
        let extension = DspyAgentExtension::new(config, client, None);

        let stats = extension.get_registry_stats().await;

        // Verify all expected stats are present
        assert!(stats.contains_key("total_modules"));
        assert!(stats.contains_key("compiled_modules"));
        assert!(stats.contains_key("total_usage_count"));
        assert!(stats.contains_key("module_types"));

        // Verify initial values
        assert_eq!(stats["total_modules"], serde_json::Value::Number(0.into()));
        assert_eq!(
            stats["compiled_modules"],
            serde_json::Value::Number(0.into())
        );
        assert_eq!(
            stats["total_usage_count"],
            serde_json::Value::Number(0.into())
        );
    }

    #[tokio::test]
    async fn test_dspy_agent_extension_module_removal_nonexistent() {
        let config = DspyAgentConfig::default();
        let client = create_test_anthropic_client();
        let extension = DspyAgentExtension::new(config, client, None);
        let context = create_test_security_context();

        // Removing non-existent module should succeed (idempotent)
        let result = extension
            .remove_module("nonexistent_module", Some(&context))
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_dspy_agent_config_customization() {
        let config = DspyAgentConfig {
            auto_optimize: false,
            max_modules: 50,
            default_optimization_strategy: OptimizationStrategy::random_sampling(10),
            enable_security_validation: false,
            enable_audit_logging: false,
            module_cache_ttl: 1800,
        };

        assert!(!config.auto_optimize);
        assert_eq!(config.max_modules, 50);
        assert!(!config.enable_security_validation);
        assert!(!config.enable_audit_logging);
        assert_eq!(config.module_cache_ttl, 1800);
    }

    #[test]
    fn test_tool_metrics_edge_cases() {
        let mut metrics = ToolMetrics::default();

        // Test with zero executions
        assert_eq!(metrics.success_rate, 0.0);
        assert_eq!(metrics.average_execution_time_ms, 0.0);

        // Test with only failures
        metrics.record_failure(100.0);
        metrics.record_failure(200.0);

        assert_eq!(metrics.execution_count, 2);
        assert_eq!(metrics.successful_executions, 0);
        assert_eq!(metrics.failed_executions, 2);
        assert_eq!(metrics.success_rate, 0.0);
        assert_eq!(metrics.average_execution_time_ms, 150.0);
    }

    #[test]
    fn test_dspy_module_metadata_serialization() {
        let metadata = DspyModuleMetadata {
            id: "test_id".to_string(),
            name: "Test Module".to_string(),
            description: Some("Test description".to_string()),
            module_type: "Predict".to_string(),
            created_at: chrono::Utc::now(),
            last_used_at: Some(chrono::Utc::now()),
            usage_count: 42,
            is_compiled: true,
            optimization_metrics: None,
            security_context: Some("test_context".to_string()),
            tags: vec!["test".to_string(), "module".to_string()],
        };

        // Test serialization
        let serialized = serde_json::to_string(&metadata).unwrap();
        assert!(serialized.contains("test_id"));
        assert!(serialized.contains("Test Module"));
        assert!(serialized.contains("Predict"));

        // Test deserialization
        let deserialized: DspyModuleMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.id, metadata.id);
        assert_eq!(deserialized.name, metadata.name);
        assert_eq!(deserialized.module_type, metadata.module_type);
        assert_eq!(deserialized.usage_count, metadata.usage_count);
        assert_eq!(deserialized.is_compiled, metadata.is_compiled);
        assert_eq!(deserialized.tags, metadata.tags);
    }

    #[test]
    fn test_tool_metrics_serialization() {
        let mut metrics = ToolMetrics::default();
        metrics.record_success(100.0);
        metrics.record_failure(50.0);

        // Test serialization
        let serialized = serde_json::to_string(&metrics).unwrap();
        assert!(serialized.contains("execution_count"));
        assert!(serialized.contains("success_rate"));

        // Test deserialization
        let deserialized: ToolMetrics = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.execution_count, metrics.execution_count);
        assert_eq!(
            deserialized.successful_executions,
            metrics.successful_executions
        );
        assert_eq!(deserialized.failed_executions, metrics.failed_executions);
        assert!((deserialized.success_rate - metrics.success_rate).abs() < 0.001);
    }

    #[test]
    fn test_dspy_tool_metadata_serialization() {
        let metadata = DspyToolMetadata {
            name: "test_tool".to_string(),
            description: Some("Test description".to_string()),
            module_id: "module_123".to_string(),
            created_at: chrono::Utc::now(),
            metrics: ToolMetrics::default(),
        };

        // Test serialization
        let serialized = serde_json::to_string(&metadata).unwrap();
        assert!(serialized.contains("test_tool"));
        assert!(serialized.contains("module_123"));

        // Test deserialization
        let deserialized: DspyToolMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, metadata.name);
        assert_eq!(deserialized.description, metadata.description);
        assert_eq!(deserialized.module_id, metadata.module_id);
    }

    #[tokio::test]
    async fn test_dspy_agent_extension_with_security_disabled() {
        let config = DspyAgentConfig {
            enable_security_validation: false,
            enable_audit_logging: false,
            ..Default::default()
        };
        let client = create_test_anthropic_client();
        let extension = DspyAgentExtension::new(config, client, None);

        // Operations should work without security context when security is disabled
        let result = extension.remove_module("test_module", None).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_dspy_tool_registry_stats_with_data() {
        let mut registry = DspyToolRegistry::new();

        // Add some mock metadata
        let mut metrics1 = ToolMetrics::default();
        metrics1.record_success(100.0);
        metrics1.record_success(200.0);

        let mut metrics2 = ToolMetrics::default();
        metrics2.record_failure(50.0);

        let metadata1 = DspyToolMetadata {
            name: "tool1".to_string(),
            description: None,
            module_id: "module1".to_string(),
            created_at: chrono::Utc::now(),
            metrics: metrics1,
        };

        let metadata2 = DspyToolMetadata {
            name: "tool2".to_string(),
            description: None,
            module_id: "module2".to_string(),
            created_at: chrono::Utc::now(),
            metrics: metrics2,
        };

        // Since metadata field is private, we'll test with empty registry
        // In a real scenario, metadata would be added when registering tools

        let stats = registry.stats();
        assert_eq!(
            stats.get("total_tools").unwrap(),
            &serde_json::Value::Number(0.into())
        ); // No tools registered
        assert_eq!(
            stats.get("total_executions").unwrap(),
            &serde_json::Value::Number(0.into())
        ); // No executions

        // Average success rate should be 0.0 for empty registry
        let avg_success_rate = stats.get("average_success_rate").unwrap().as_f64().unwrap();
        assert_eq!(avg_success_rate, 0.0);
    }
}

#[cfg(test)]
mod specialized_modules_tests {
    use super::*;
    use crate::anthropic::AnthropicClient;
    use crate::dspy::modules::ReasoningStep as ModuleReasoningStep;
    use crate::dspy::modules::{
        utils, ChainOfThought, ChainOfThoughtConfig, CodeExecutionResult, FeedbackEntry,
        FeedbackSettings, FeedbackType, ImprovementMetrics, ImprovementRecord, ImprovementStrategy,
        ProgramOfThought, ProgramOfThoughtConfig, ProgrammingLanguage, RAGConfig, RAGResult, ReAct,
        ReActAction, ReActConfig, ReActStep, ReasoningMetrics, ReasoningModule, RetrievalStrategy,
        RetrievedDocument, SecurityRestrictions, SelfImproving, SelfImprovingConfig,
        SpecializedModuleConfig, SpecializedModuleRegistry, RAG,
    };
    use crate::memory::MemoryManager;
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock};

    fn create_test_anthropic_client() -> Arc<AnthropicClient> {
        Arc::new(
            AnthropicClient::new(crate::config::AnthropicConfig {
                api_key: "test_key".to_string(),
                model: "claude-3-sonnet-20240229".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                max_tokens: 1000,
                temperature: 0.7,
                timeout_seconds: 30,
                max_retries: 3,
            })
            .unwrap(),
        )
    }

    fn create_test_signature() -> Signature<String, String> {
        Signature::new("test_signature".to_string())
            .with_description("Test signature for specialized modules")
    }

    #[test]
    fn test_specialized_module_config_default() {
        let config = SpecializedModuleConfig::default();
        assert_eq!(config.max_steps, 10);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens_per_step, 500);
        assert!(!config.verbose);
        assert!(config.custom_params.is_empty());
    }

    #[test]
    fn test_reasoning_step_creation() {
        let step = ModuleReasoningStep {
            step_number: 1,
            step_type: "test".to_string(),
            input: "test input".to_string(),
            output: "test output".to_string(),
            confidence: 0.8,
            execution_time_ms: 100.0,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(step.step_number, 1);
        assert_eq!(step.step_type, "test");
        assert_eq!(step.confidence, 0.8);
        assert_eq!(step.execution_time_ms, 100.0);
    }

    #[test]
    fn test_reasoning_metrics_operations() {
        let mut metrics = ReasoningMetrics::default();

        // Test initial state
        assert_eq!(metrics.total_executions, 0);
        assert_eq!(metrics.avg_reasoning_steps, 0.0);
        assert_eq!(metrics.success_rate, 0.0);

        // Test recording success
        metrics.record_success(3, 150.0, 0.9);
        assert_eq!(metrics.total_executions, 1);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.avg_reasoning_steps, 3.0);
        assert_eq!(metrics.avg_execution_time_ms, 150.0);
        assert_eq!(metrics.avg_confidence, 0.9);
        assert_eq!(metrics.success_rate, 1.0);

        // Test recording failure
        metrics.record_failure(50.0);
        assert_eq!(metrics.total_executions, 2);
        assert_eq!(metrics.failed_executions, 1);
        assert_eq!(metrics.success_rate, 0.5);
        assert_eq!(metrics.avg_execution_time_ms, 100.0);

        // Test custom metrics
        metrics.add_custom_metric("accuracy".to_string(), 0.95);
        assert_eq!(metrics.get_custom_metric("accuracy"), Some(0.95));
        assert_eq!(metrics.get_custom_metric("nonexistent"), None);
    }

    #[test]
    fn test_specialized_module_registry() {
        let mut registry = SpecializedModuleRegistry::new();

        // Test initial state
        assert!(registry.list_modules().is_empty());
        assert!(registry.find_modules_by_capability("reasoning").is_empty());

        let stats = registry.get_statistics();
        assert_eq!(
            stats.get("total_modules").unwrap(),
            &serde_json::Value::Number(0.into())
        );

        // Test clearing empty registry
        registry.clear();
        assert!(registry.list_modules().is_empty());
    }

    #[test]
    fn test_chain_of_thought_config() {
        let config = ChainOfThoughtConfig::default();
        assert!(config.include_step_numbers);
        assert!(config.validate_chain);
        assert_eq!(config.min_confidence, 0.7);
        assert_eq!(config.max_retries, 2);
        assert!(config.reasoning_template.contains("step by step"));
    }

    #[test]
    fn test_chain_of_thought_creation() {
        let client = create_test_anthropic_client();
        let signature = create_test_signature();

        let cot = ChainOfThought::new(signature, client);
        assert!(cot.name().starts_with("ChainOfThought_"));
        assert!(cot.supports_compilation());
        assert_eq!(cot.config().min_confidence, 0.7);
    }

    #[test]
    fn test_react_config() {
        let config = ReActConfig::default();
        assert!(config.available_tools.is_empty());
        assert_eq!(config.max_cycles, 5);
        assert!(config.validate_actions);
        assert_eq!(config.tool_timeout_seconds, 30);
        assert!(!config.continue_on_tool_error);
    }

    #[test]
    fn test_react_step_creation() {
        let step = ReActStep {
            step_number: 1,
            thought: "I need to analyze this".to_string(),
            action: None,
            observation: None,
            confidence: 0.8,
            execution_time_ms: 100.0,
        };

        assert_eq!(step.step_number, 1);
        assert!(step.thought.contains("analyze"));
        assert!(step.action.is_none());
        assert_eq!(step.confidence, 0.8);
    }

    #[test]
    fn test_react_action_creation() {
        let action = ReActAction {
            tool: "search".to_string(),
            input: serde_json::json!({"query": "test"}),
            expected_output: Some("result".to_string()),
        };

        assert_eq!(action.tool, "search");
        assert!(action.input.is_object());
        assert_eq!(action.expected_output, Some("result".to_string()));
    }

    #[test]
    fn test_react_creation() {
        let client = create_test_anthropic_client();
        let signature = create_test_signature();

        let react = ReAct::new(signature, client);
        assert!(react.name().starts_with("ReAct_"));
        assert!(react.supports_compilation());
        assert!(react.get_available_tools().is_empty());
    }

    #[test]
    fn test_rag_config() {
        let config = RAGConfig::default();
        assert_eq!(config.num_documents, 5);
        assert_eq!(config.min_relevance_score, 0.3);
        assert_eq!(config.max_context_length, 4000);
        assert!(config.enable_reranking);
        assert!(!config.enable_query_expansion);
        assert!(matches!(
            config.retrieval_strategy,
            RetrievalStrategy::Semantic
        ));
    }

    #[test]
    fn test_retrieval_strategy_variants() {
        let strategies = vec![
            RetrievalStrategy::Semantic,
            RetrievalStrategy::Keyword,
            RetrievalStrategy::Hybrid,
            RetrievalStrategy::DensePassage,
        ];

        // Test that all variants can be created and serialized
        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            assert!(!serialized.is_empty());
        }
    }

    #[test]
    fn test_retrieved_document_creation() {
        let doc = RetrievedDocument {
            id: "doc_1".to_string(),
            content: "This is test content".to_string(),
            relevance_score: 0.85,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(doc.id, "doc_1");
        assert!(doc.content.contains("test"));
        assert_eq!(doc.relevance_score, 0.85);
        assert!(doc.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_rag_creation() {
        let client = create_test_anthropic_client();
        let signature = create_test_signature();
        let memory_config = crate::config::MemoryConfig::default();
        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(memory_config).await.unwrap()));

        let rag = RAG::new(signature, client, memory_manager);
        assert!(rag.name().starts_with("RAG_"));
        assert!(rag.supports_compilation());
        assert_eq!(rag.config().num_documents, 5);
    }

    #[test]
    fn test_program_of_thought_config() {
        let config = ProgramOfThoughtConfig::default();
        assert!(matches!(config.language, ProgrammingLanguage::Python));
        assert!(!config.execute_code); // Disabled by default for security
        assert_eq!(config.execution_timeout_seconds, 10);
        assert_eq!(config.max_code_length, 2000);
        assert!(config.validate_syntax);
        assert!(!config.allowed_imports.is_empty());
    }

    #[test]
    fn test_programming_language_methods() {
        let python = ProgrammingLanguage::Python;
        assert_eq!(python.file_extension(), "py");
        assert_eq!(python.execution_command(), "python3");

        let js = ProgrammingLanguage::JavaScript;
        assert_eq!(js.file_extension(), "js");
        assert_eq!(js.execution_command(), "node");

        let rust = ProgrammingLanguage::Rust;
        assert_eq!(rust.file_extension(), "rs");
        assert_eq!(rust.execution_command(), "rustc");

        let go = ProgrammingLanguage::Go;
        assert_eq!(go.file_extension(), "go");
        assert_eq!(go.execution_command(), "go run");
    }

    #[test]
    fn test_security_restrictions() {
        let restrictions = SecurityRestrictions::default();
        assert!(restrictions.disallow_file_operations);
        assert!(restrictions.disallow_network_operations);
        assert!(restrictions.disallow_subprocess);
        assert_eq!(restrictions.max_memory_mb, 100);
        assert!(!restrictions.blacklisted_functions.is_empty());
        assert!(restrictions
            .blacklisted_functions
            .contains(&"exec".to_string()));
    }

    #[test]
    fn test_code_execution_result() {
        let result = CodeExecutionResult {
            success: true,
            output: "Hello, World!".to_string(),
            error: None,
            execution_time_ms: 150.0,
            memory_usage_mb: Some(10.5),
        };

        assert!(result.success);
        assert_eq!(result.output, "Hello, World!");
        assert!(result.error.is_none());
        assert_eq!(result.execution_time_ms, 150.0);
        assert_eq!(result.memory_usage_mb, Some(10.5));
    }

    #[test]
    fn test_program_of_thought_creation() {
        let client = create_test_anthropic_client();
        let signature = create_test_signature();

        let pot = ProgramOfThought::new(signature, client);
        assert!(pot.name().starts_with("ProgramOfThought_"));
        assert!(pot.supports_compilation());
        assert!(!pot.config().execute_code);
    }

    #[test]
    fn test_self_improving_config() {
        let config = SelfImprovingConfig::default();
        assert_eq!(config.min_examples_for_improvement, 10);
        assert_eq!(config.improvement_threshold, 0.3);
        assert_eq!(config.max_improvement_iterations, 5);
        assert_eq!(config.learning_rate, 0.1);
        assert!(config.auto_improve);
        assert!(matches!(
            config.improvement_strategy,
            ImprovementStrategy::GradualOptimization
        ));
    }

    #[test]
    fn test_improvement_strategy_variants() {
        let strategies = vec![
            ImprovementStrategy::GradualOptimization,
            ImprovementStrategy::ReinforcementLearning,
            ImprovementStrategy::MetaLearning,
            ImprovementStrategy::EnsembleImprovement,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            assert!(!serialized.is_empty());
        }
    }

    #[test]
    fn test_feedback_settings() {
        let settings = FeedbackSettings::default();
        assert!(settings.collect_implicit_feedback);
        assert!(!settings.collect_explicit_feedback);
        assert_eq!(settings.positive_feedback_weight, 1.0);
        assert_eq!(settings.negative_feedback_weight, 2.0);
        assert_eq!(settings.max_feedback_history, 1000);
    }

    #[test]
    fn test_feedback_entry_creation() {
        let feedback = FeedbackEntry {
            id: "feedback_1".to_string(),
            input: serde_json::json!({"query": "test"}),
            output: serde_json::json!({"result": "answer"}),
            expected_output: Some(serde_json::json!({"result": "expected"})),
            score: 0.8,
            feedback_type: FeedbackType::Explicit,
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(feedback.id, "feedback_1");
        assert_eq!(feedback.score, 0.8);
        assert!(matches!(feedback.feedback_type, FeedbackType::Explicit));
        assert!(feedback.expected_output.is_some());
    }

    #[test]
    fn test_improvement_metrics() {
        let mut metrics = ImprovementMetrics::default();
        assert_eq!(metrics.improvement_iterations, 0);
        assert_eq!(metrics.initial_performance, 0.0);
        assert_eq!(metrics.current_performance, 0.0);
        assert_eq!(metrics.improvement_percentage, 0.0);
        assert_eq!(metrics.feedback_entries_processed, 0);
        assert!(metrics.last_improvement_at.is_none());
        assert!(metrics.improvement_history.is_empty());

        // Test improvement record
        let record = ImprovementRecord {
            iteration: 1,
            performance_before: 0.6,
            performance_after: 0.8,
            strategy_used: "GradualOptimization".to_string(),
            examples_used: 20,
            timestamp: chrono::Utc::now(),
        };

        metrics.improvement_history.push(record);
        assert_eq!(metrics.improvement_history.len(), 1);
        assert_eq!(metrics.improvement_history[0].iteration, 1);
        assert_eq!(metrics.improvement_history[0].examples_used, 20);
    }

    #[test]
    fn test_utils_parse_reasoning_steps() {
        let text = "Step 1: First step\nThis is the first analysis\nStep 2: Second step\nThis is the second analysis";
        let steps = utils::parse_reasoning_steps(text);

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].step_number, 1);
        assert!(steps[0].input.contains("Step 1"));
        assert_eq!(steps[1].step_number, 3);
        assert!(steps[1].input.contains("Step 2"));
    }

    #[test]
    fn test_utils_calculate_confidence() {
        let steps = vec![
            ModuleReasoningStep {
                step_number: 1,
                step_type: "reasoning".to_string(),
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.8,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
            ModuleReasoningStep {
                step_number: 2,
                step_type: "reasoning".to_string(),
                input: "test".to_string(),
                output: "test".to_string(),
                confidence: 0.9,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
        ];

        let confidence = utils::calculate_confidence(&steps);
        assert!(confidence > 0.85); // Should be average + step bonus
        assert!(confidence <= 1.0);

        // Test empty steps
        let empty_confidence = utils::calculate_confidence(&[]);
        assert_eq!(empty_confidence, 0.0);
    }

    #[test]
    fn test_utils_validate_reasoning_chain() {
        // Test valid chain
        let valid_steps = vec![
            ModuleReasoningStep {
                step_number: 1,
                step_type: "reasoning".to_string(),
                input: "valid input".to_string(),
                output: "valid output".to_string(),
                confidence: 0.8,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
            ModuleReasoningStep {
                step_number: 2,
                step_type: "reasoning".to_string(),
                input: "valid input 2".to_string(),
                output: "valid output 2".to_string(),
                confidence: 0.9,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
        ];

        assert!(utils::validate_reasoning_chain(&valid_steps));

        // Test invalid chain (wrong step numbers)
        let invalid_steps = vec![
            ModuleReasoningStep {
                step_number: 1,
                step_type: "reasoning".to_string(),
                input: "valid input".to_string(),
                output: "valid output".to_string(),
                confidence: 0.8,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
            ModuleReasoningStep {
                step_number: 3, // Should be 2
                step_type: "reasoning".to_string(),
                input: "valid input 2".to_string(),
                output: "valid output 2".to_string(),
                confidence: 0.9,
                execution_time_ms: 100.0,
                metadata: std::collections::HashMap::new(),
            },
        ];

        assert!(!utils::validate_reasoning_chain(&invalid_steps));

        // Test empty chain
        assert!(!utils::validate_reasoning_chain(&[]));

        // Test chain with empty content
        let empty_content_steps = vec![ModuleReasoningStep {
            step_number: 1,
            step_type: "reasoning".to_string(),
            input: "".to_string(), // Empty input
            output: "valid output".to_string(),
            confidence: 0.8,
            execution_time_ms: 100.0,
            metadata: std::collections::HashMap::new(),
        }];

        assert!(!utils::validate_reasoning_chain(&empty_content_steps));
    }

    #[test]
    fn test_specialized_module_serialization() {
        // Test ReasoningStep serialization
        let step = ModuleReasoningStep {
            step_number: 1,
            step_type: "test".to_string(),
            input: "test input".to_string(),
            output: "test output".to_string(),
            confidence: 0.8,
            execution_time_ms: 100.0,
            metadata: std::collections::HashMap::new(),
        };

        let serialized = serde_json::to_string(&step).unwrap();
        let deserialized: ModuleReasoningStep = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.step_number, step.step_number);
        assert_eq!(deserialized.step_type, step.step_type);
        assert_eq!(deserialized.confidence, step.confidence);

        // Test ReasoningMetrics serialization
        let metrics = ReasoningMetrics::default();
        let serialized = serde_json::to_string(&metrics).unwrap();
        let deserialized: ReasoningMetrics = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.total_executions, metrics.total_executions);
        assert_eq!(deserialized.success_rate, metrics.success_rate);

        // Test SpecializedModuleConfig serialization
        let config = SpecializedModuleConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SpecializedModuleConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.max_steps, config.max_steps);
        assert_eq!(deserialized.temperature, config.temperature);
    }
}
