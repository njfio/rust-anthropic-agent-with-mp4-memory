//! # DSPy Predict Module
//!
//! This module implements the core `Predict` module for DSPy, which handles LLM prediction
//! tasks with type-safe input/output handling, prompt template generation, and integration
//! with the existing Anthropic client infrastructure.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::anthropic::client::AnthropicClient;
use crate::anthropic::models::{ApiMessage, ChatRequest, ContentBlock, MessageRole};
use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::module::{ExecutionContext, Module, ModuleMetadata, ModuleStats};
use crate::dspy::signature::Signature;
use crate::security::SecurityManager;

/// Configuration for the Predict module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictConfig {
    /// Model to use for predictions
    pub model: String,
    /// Maximum tokens for the response
    pub max_tokens: u32,
    /// Temperature for response generation
    pub temperature: f32,
    /// Whether to enable streaming responses
    pub stream: bool,
    /// Maximum number of retries for failed requests
    pub max_retries: u32,
    /// Timeout for requests in seconds
    pub timeout_seconds: u64,
    /// Whether to enable security validation
    pub enable_security_validation: bool,
    /// Whether to enable rate limiting
    pub enable_rate_limiting: bool,
}

impl Default for PredictConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
            stream: false,
            max_retries: 3,
            timeout_seconds: 120,
            enable_security_validation: true,
            enable_rate_limiting: true,
        }
    }
}

/// Core Predict module for DSPy
///
/// This module handles LLM prediction tasks with type-safe input/output handling,
/// automatic prompt generation from signatures, and comprehensive error handling.
#[derive(Debug)]
pub struct Predict<I, O> {
    /// Module ID
    id: String,
    /// The signature defining input/output types and constraints
    signature: Signature<I, O>,
    /// Anthropic client for API calls
    anthropic_client: Arc<AnthropicClient>,
    /// Security manager for validation
    security_manager: Option<Arc<SecurityManager>>,
    /// Module configuration
    config: PredictConfig,
    /// Module metadata
    metadata: ModuleMetadata,
    /// Execution statistics
    stats: Arc<tokio::sync::RwLock<ModuleStats>>,
    /// Phantom data for type safety
    _phantom: PhantomData<(I, O)>,
}

impl<I, O> Predict<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    /// Create a new Predict module
    pub fn new(signature: Signature<I, O>, anthropic_client: Arc<AnthropicClient>) -> Self {
        let id = format!("predict_{}", signature.name);
        let metadata = ModuleMetadata::new("DSPy Predict Module")
            .with_version("1.0.0")
            .with_author("DSPy Integration")
            .with_tag("prediction")
            .with_tag("llm")
            .with_custom("signature_name", signature.name.clone());

        Self {
            id,
            signature,
            anthropic_client,
            security_manager: None,
            config: PredictConfig::default(),
            metadata,
            stats: Arc::new(tokio::sync::RwLock::new(ModuleStats::default())),
            _phantom: PhantomData,
        }
    }

    /// Create a new Predict module with custom configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: PredictConfig,
    ) -> Self {
        let id = format!("predict_{}", signature.name);
        let metadata = ModuleMetadata::new("DSPy Predict Module")
            .with_version("1.0.0")
            .with_author("DSPy Integration")
            .with_tag("prediction")
            .with_tag("llm")
            .with_custom("signature_name", signature.name.clone());

        Self {
            id,
            signature,
            anthropic_client,
            security_manager: None,
            config,
            metadata,
            stats: Arc::new(tokio::sync::RwLock::new(ModuleStats::default())),
            _phantom: PhantomData,
        }
    }

    /// Set the security manager for validation
    pub fn with_security_manager(mut self, security_manager: Arc<SecurityManager>) -> Self {
        self.security_manager = Some(security_manager);
        self
    }

    /// Get the current configuration
    pub fn config(&self) -> &PredictConfig {
        &self.config
    }

    // Test helper methods
    #[cfg(test)]
    pub async fn test_generate_prompt(&self, input: &I) -> DspyResult<String> {
        self.generate_prompt(input).await
    }

    #[cfg(test)]
    pub fn test_substitute_variables(
        &self,
        template: &str,
        input: &serde_json::Value,
    ) -> DspyResult<String> {
        self.substitute_variables(template, input)
    }

    #[cfg(test)]
    pub async fn test_parse_output(&self, response_text: &str) -> DspyResult<O> {
        self.parse_output(response_text).await
    }

    #[cfg(test)]
    pub async fn test_create_chat_request(&self, prompt: &str) -> DspyResult<ChatRequest> {
        self.create_chat_request(prompt).await
    }

    /// Generate a prompt from the signature and input
    async fn generate_prompt(&self, input: &I) -> DspyResult<String> {
        debug!("Generating prompt from signature: {}", self.signature.name);

        // Serialize input to JSON for template processing
        let input_json = serde_json::to_value(input).map_err(|e| {
            DspyError::type_validation("input", &format!("Failed to serialize input: {}", e))
        })?;

        // Generate the base prompt template from signature
        let template = self.signature.generate_prompt_template();

        // Perform variable substitution
        let prompt = self.substitute_variables(&template, &input_json)?;

        debug!("Generated prompt: {}", prompt);
        Ok(prompt)
    }

    /// Substitute variables in the prompt template
    fn substitute_variables(
        &self,
        template: &str,
        input: &serde_json::Value,
    ) -> DspyResult<String> {
        let mut result = template.to_string();

        // Handle object inputs
        if let serde_json::Value::Object(obj) = input {
            for (key, value) in obj {
                let placeholder = format!("{{{}}}", key);
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                };
                result = result.replace(&placeholder, &value_str);
            }
        }

        // Check for unresolved placeholders
        if result.contains('{') && result.contains('}') {
            warn!("Prompt contains unresolved placeholders: {}", result);
        }

        Ok(result)
    }

    /// Validate input using signature constraints
    async fn validate_input(&self, input: &I) -> DspyResult<()> {
        debug!("Validating input against signature constraints");

        // Serialize input for validation
        let input_json = serde_json::to_value(input).map_err(|e| {
            DspyError::type_validation("input", &format!("Failed to serialize input: {}", e))
        })?;

        // Validate using signature - we need to deserialize back to I for validation
        let input_for_validation: I = serde_json::from_value(input_json.clone()).map_err(|e| {
            DspyError::type_validation(
                "input",
                &format!("Failed to deserialize input for validation: {}", e),
            )
        })?;
        self.signature.validate_input(&input_for_validation)?;

        // Security validation if enabled
        if self.config.enable_security_validation {
            if let Some(security_manager) = &self.security_manager {
                let input_str = serde_json::to_string(&input_json).map_err(|e| {
                    DspyError::type_validation(
                        "input",
                        &format!("Failed to serialize input for security validation: {}", e),
                    )
                })?;

                security_manager
                    .validate_input_comprehensive(&input_str, "dspy_input")
                    .await
                    .map_err(|e| {
                        DspyError::type_validation(
                            "security",
                            &format!("Security validation failed: {}", e),
                        )
                    })?;
            }
        }

        Ok(())
    }

    /// Parse and validate output from LLM response
    async fn parse_output(&self, response_text: &str) -> DspyResult<O> {
        debug!("Parsing LLM response to output type");

        // Try to parse as JSON first
        let output_json: serde_json::Value = serde_json::from_str(response_text).map_err(|e| {
            DspyError::type_validation(
                "output",
                &format!("Failed to parse response as JSON: {}", e),
            )
        })?;

        // Deserialize to output type first
        let output: O = serde_json::from_value(output_json).map_err(|e| {
            DspyError::type_validation("output", &format!("Failed to deserialize output: {}", e))
        })?;

        // Validate output using signature
        self.signature.validate_output(&output)?;

        Ok(output)
    }

    /// Create a chat request for the Anthropic API
    async fn create_chat_request(&self, prompt: &str) -> DspyResult<ChatRequest> {
        let message = ApiMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: prompt.to_string(),
            }],
        };

        let request = ChatRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            messages: vec![message],
            system: Some("You are a helpful AI assistant. Please respond with valid JSON that matches the expected output format.".to_string()),
            tools: None,
            tool_choice: None,
            temperature: Some(self.config.temperature),
            stream: if self.config.stream { Some(true) } else { None },
        };

        Ok(request)
    }

    /// Execute the prediction with comprehensive error handling
    async fn execute_prediction(&self, input: I) -> DspyResult<O> {
        let start_time = std::time::Instant::now();
        let mut context = ExecutionContext::new(&self.id);

        // Validate input
        self.validate_input(&input).await?;

        // Generate prompt
        let prompt = self.generate_prompt(&input).await?;
        context = context.with_metadata("prompt_length", prompt.len());

        // Create API request
        let request = self.create_chat_request(&prompt).await?;

        // Make API call with retries
        let response =
            self.anthropic_client.chat(request).await.map_err(|e| {
                DspyError::integration("anthropic", &format!("API call failed: {}", e))
            })?;

        // Extract text from response
        let response_text = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        if response_text.is_empty() {
            return Err(DspyError::integration(
                "anthropic",
                "Empty response from API",
            ));
        }

        // Parse output
        let output = self.parse_output(&response_text).await?;

        // Update statistics
        let duration = start_time.elapsed();
        let mut stats = self.stats.write().await;
        stats.record_success(duration);

        // Update context
        context = context
            .with_metadata("response_tokens", response.usage.output_tokens)
            .with_metadata("input_tokens", response.usage.input_tokens)
            .complete_success();

        info!("Prediction completed successfully in {:?}", duration);
        Ok(output)
    }
}

#[async_trait]
impl<I, O> Module for Predict<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    type Input = I;
    type Output = O;

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        self.execute_prediction(input).await
    }

    async fn compile(&mut self, _examples: Vec<(Self::Input, Self::Output)>) -> DspyResult<()> {
        // Basic compilation - in a full implementation, this would optimize prompts
        info!("Compiling Predict module: {}", self.signature.name);
        Ok(())
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.signature
    }

    fn name(&self) -> &str {
        "predict"
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }

    fn stats(&self) -> &ModuleStats {
        // Return a default stats reference for the trait
        // In practice, this would need to be handled differently for async access
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }
}
