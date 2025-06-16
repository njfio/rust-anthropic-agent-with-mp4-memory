//! Comprehensive tests for DSPy core functionality
//!
//! This module provides extensive test coverage for the DSPy integration,
//! ensuring all components work correctly and integrate properly with
//! the existing agent system.

use super::*;
use crate::dspy::{
    error::{DspyError, ErrorSeverity},
    module::{BaseModule, ExecutionContext, Module, ModuleMetadata, ModuleStats},
    predictor::{Predict, PredictConfig},
    signature::{Field, FieldConstraint, FieldType, Signature, SignatureBuilder},
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
