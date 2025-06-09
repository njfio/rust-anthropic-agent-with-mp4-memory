use reqwest::{Client, RequestBuilder};
use serde_json::json;
use std::time::Duration;
use tracing::{debug, error, info, warn};


use crate::anthropic::models::{ChatRequest, ChatResponse};
use crate::config::AnthropicConfig;
use crate::utils::error::{AgentError, Result};

/// HTTP client for the Anthropic API
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    client: Client,
    config: AnthropicConfig,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .connect_timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(60))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| AgentError::config(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Send a chat request to the Anthropic API
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut retries = 0;
        let max_retries = self.config.max_retries;

        loop {
            match self.send_chat_request(&request).await {
                Ok(response) => return Ok(response),
                Err(e) if retries < max_retries && e.is_retryable() => {
                    retries += 1;

                    // Enhanced retry logic with exponential backoff and jitter
                    let base_delay = 1000 * (2_u64.pow(retries - 1));

                    // Add jitter to prevent thundering herd
                    let jitter = rand::random::<u64>() % (base_delay / 4 + 1);
                    let delay = Duration::from_millis(base_delay + jitter);

                    // For 529 overloaded errors, use longer delays
                    let final_delay = if e.to_string().contains("529") || e.to_string().contains("overloaded") {
                        Duration::from_millis((base_delay + jitter) * 2)
                    } else {
                        delay
                    };

                    warn!(
                        "Request failed (attempt {}/{}), retrying in {:?}: {}",
                        retries, max_retries, final_delay, e
                    );
                    tokio::time::sleep(final_delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Send a single chat request without retries
    async fn send_chat_request(&self, request: &ChatRequest) -> Result<ChatResponse> {
        debug!("Sending chat request to Anthropic API");

        // Log the complete request payload
        let request_json = serde_json::to_string_pretty(&request)
            .unwrap_or_else(|_| "Failed to serialize request".to_string());
        info!("🚀 ANTHROPIC REQUEST PAYLOAD:\n{}", request_json);

        // Check if streaming is enabled
        if request.stream == Some(true) {
            warn!("Streaming is enabled but not yet implemented, falling back to non-streaming");
        }

        let url = format!("{}/v1/messages", self.config.base_url);
        let mut req_builder = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        // Add beta headers if needed
        if self.needs_beta_headers(request) {
            let beta_headers = self.get_beta_headers(request);
            req_builder = req_builder.header("anthropic-beta", beta_headers);
        }

        // Create a modified request without streaming for now
        let mut modified_request = request.clone();
        modified_request.stream = None; // Disable streaming until implemented

        info!("📡 Sending HTTP request to Anthropic API...");
        let response = req_builder
            .json(&modified_request)
            .send()
            .await
            .map_err(|e| {
                error!("❌ HTTP request failed: {}", e);
                if e.is_timeout() {
                    AgentError::anthropic_api("Request timed out - consider increasing timeout_seconds in config".to_string())
                } else if e.is_connect() {
                    AgentError::anthropic_api("Connection failed - check internet connection and API endpoint".to_string())
                } else {
                    AgentError::Http(e)
                }
            })?;

        let status = response.status();

        // Try to get response as bytes first, then convert to string with better error handling
        let response_bytes = response
            .bytes()
            .await
            .map_err(|e| AgentError::Http(e))?;

        let response_text = match String::from_utf8(response_bytes.to_vec()) {
            Ok(text) => text,
            Err(e) => {
                error!("❌ Response contains invalid UTF-8: {}", e);
                // Try to recover by replacing invalid UTF-8 sequences
                match String::from_utf8_lossy(&response_bytes).into_owned() {
                    recovered => {
                        warn!("🔄 Recovered response text with lossy UTF-8 conversion");
                        recovered
                    }
                }
            }
        };

        debug!("Received response with status: {}", status);

        // Log the complete response payload
        info!("📥 ANTHROPIC RESPONSE PAYLOAD:\n{}", response_text);

        if !status.is_success() {
            return self.handle_error_response(status, &response_text);
        }

        let chat_response: ChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| {
                error!("Failed to parse JSON response: {}", e);
                error!("Response text: {}", response_text);
                AgentError::Json(e)
            })?;

        info!(
            "Chat request successful. Input tokens: {}, Output tokens: {}",
            chat_response.usage.input_tokens, chat_response.usage.output_tokens
        );

        Ok(chat_response)
    }

    /// Check if the request needs beta headers
    fn needs_beta_headers(&self, request: &ChatRequest) -> bool {
        if let Some(tools) = &request.tools {
            tools.iter().any(|tool| {
                matches!(
                    tool.tool_type.as_str(),
                    "code_execution_20250522" | "text_editor_20250429" | "text_editor_20250124"
                )
            })
        } else {
            false
        }
    }

    /// Get the appropriate beta headers for the request
    fn get_beta_headers(&self, request: &ChatRequest) -> String {
        let mut headers = Vec::new();

        if let Some(tools) = &request.tools {
            for tool in tools {
                match tool.tool_type.as_str() {
                    "code_execution_20250522" => {
                        if !headers.contains(&"code-execution-2025-05-22") {
                            headers.push("code-execution-2025-05-22");
                        }
                    }
                    "text_editor_20250429" | "text_editor_20250124" => {
                        // Text editor tools don't require beta headers in current API
                    }
                    _ => {}
                }
            }
        }

        headers.join(",")
    }

    /// Handle error responses from the API
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Result<ChatResponse> {
        error!("API request failed with status {}: {}", status, body);

        // Try to parse error response
        if let Ok(error_json) = serde_json::from_str::<serde_json::Value>(body) {
            let error_message = error_json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");

            match status.as_u16() {
                401 => Err(AgentError::authentication(format!(
                    "Invalid API key: {}",
                    error_message
                ))),
                429 => Err(AgentError::rate_limit(format!(
                    "Rate limit exceeded: {}",
                    error_message
                ))),
                529 => {
                    // Handle 529 overloaded error specifically - this is retryable
                    warn!("Anthropic API is overloaded (529), this request will be retried");
                    Err(AgentError::anthropic_api(format!(
                        "API overloaded ({}): {}",
                        status, error_message
                    )))
                },
                400..=499 => Err(AgentError::anthropic_api(format!(
                    "Client error ({}): {}",
                    status, error_message
                ))),
                500..=599 => Err(AgentError::anthropic_api(format!(
                    "Server error ({}): {}",
                    status, error_message
                ))),
                _ => Err(AgentError::anthropic_api(format!(
                    "HTTP error ({}): {}",
                    status, error_message
                ))),
            }
        } else {
            Err(AgentError::anthropic_api(format!(
                "HTTP error ({}): {}",
                status, body
            )))
        }
    }

    /// Create a streaming chat request (placeholder for future implementation)
    pub async fn chat_stream(&self, _request: ChatRequest) -> Result<ChatResponse> {
        // TODO: Implement streaming support
        Err(AgentError::config(
            "Streaming is not yet implemented".to_string(),
        ))
    }

    /// Test the API connection
    pub async fn test_connection(&self) -> Result<()> {
        let test_request = ChatRequest {
            model: self.config.model.clone(),
            max_tokens: 10,
            messages: vec![crate::anthropic::models::ApiMessage::user("Hello")],
            system: None,
            tools: None,
            tool_choice: None,
            temperature: Some(0.0),
            stream: None,
        };

        self.chat(test_request).await?;
        info!("API connection test successful");
        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &AnthropicConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: AnthropicConfig) -> Result<()> {
        // Recreate the client with new timeout if needed
        if config.timeout_seconds != self.config.timeout_seconds {
            self.client = Client::builder()
                .timeout(Duration::from_secs(config.timeout_seconds))
                .build()
                .map_err(|e| AgentError::config(format!("Failed to create HTTP client: {}", e)))?;
        }

        self.config = config;
        Ok(())
    }
}
