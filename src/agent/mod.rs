pub mod conversation;
pub mod memory_manager;
pub mod tool_orchestrator;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::anthropic::{AnthropicClient, ChatMessage, ChatRequest, MessageRole};
use crate::config::AgentConfig;
use crate::memory::MemoryManager;
use crate::security::{SecurityManager, SecurityContext, SecurityEvent};
use crate::security::policy::{SecurityPolicy, PolicyEffect, PolicyCondition, PolicyOperator, PolicyValue};
use crate::utils::error::{AgentError, Result};

pub use conversation::ConversationManager;
pub use tool_orchestrator::ToolOrchestrator;

/// Main agent that orchestrates conversations, tools, and memory
#[derive(Debug)]
pub struct Agent {
    /// Configuration
    config: AgentConfig,
    /// Anthropic API client
    anthropic_client: AnthropicClient,
    /// Memory manager
    memory_manager: Arc<Mutex<MemoryManager>>,
    /// Tool orchestrator
    tool_orchestrator: ToolOrchestrator,
    /// Conversation manager
    conversation_manager: ConversationManager,
    /// Security manager
    security_manager: Option<Arc<SecurityManager>>,
    /// Current security context
    security_context: Option<SecurityContext>,
    /// Current conversation ID
    current_conversation_id: Option<String>,
}

/// Builder for creating agents with custom configurations
pub struct AgentBuilder {
    config: AgentConfig,
    custom_tools: Vec<Box<dyn crate::tools::Tool>>,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("config", &self.config)
            .field("custom_tool_count", &self.custom_tools.len())
            .finish()
    }
}

impl Agent {
    /// Create a new agent with the given configuration
    pub async fn new(config: AgentConfig) -> Result<Self> {
        config.validate()?;

        info!("Initializing agent with model: {}", config.anthropic.model);

        // Create Anthropic client
        let anthropic_client = AnthropicClient::new(config.anthropic.clone())?;

        // Test API connection
        anthropic_client.test_connection().await?;

        // Create memory manager
        let memory_manager = Arc::new(Mutex::new(MemoryManager::new(config.memory.clone()).await?));

        // Create tool orchestrator
        let mut tool_orchestrator = ToolOrchestrator::new(memory_manager.clone());

        // Register built-in tools based on configuration
        tool_orchestrator.register_builtin_tools(&config).await?;

        // Create conversation manager
        let conversation_manager = ConversationManager::new(memory_manager.clone());

        // Initialize security manager if security config is provided
        let security_manager = if let Some(security_config) = &config.security {
            match SecurityManager::new(security_config.clone()).await {
                Ok(sm) => {
                    info!("Security manager initialized successfully");
                    Some(Arc::new(sm))
                }
                Err(e) => {
                    warn!("Failed to initialize security manager: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            anthropic_client,
            memory_manager,
            tool_orchestrator,
            conversation_manager,
            security_manager,
            security_context: None,
            current_conversation_id: None,
        })
    }

    /// Initialize security context for the agent
    pub async fn initialize_security_context(
        &mut self,
        user_id: String,
        ip_address: Option<String>,
    ) -> Result<()> {
        if let Some(security_manager) = &self.security_manager {
            let context = security_manager
                .create_context(user_id, ip_address)
                .await?;

            // Initialize default roles for agent operations
            let mut enhanced_context = context;
            enhanced_context.add_role("agent_user".to_string());
            enhanced_context.add_permission("chat".to_string());
            enhanced_context.add_permission("tool_execution".to_string());
            enhanced_context.add_permission("memory_access".to_string());

            self.security_context = Some(enhanced_context);

            // Initialize encryption keys for agent operations
            self.initialize_encryption_keys().await?;

            // Add agent-specific security policies
            self.add_agent_security_policies().await?;

            info!("Security context initialized for agent operations");
        } else {
            debug!("No security manager configured, skipping security context initialization");
        }
        Ok(())
    }

    /// Initialize encryption keys for secure data handling
    async fn initialize_encryption_keys(&self) -> Result<()> {
        if let Some(security_manager) = &self.security_manager {
            // Generate encryption keys for different data types
            let keys_to_generate = [
                ("agent_memory", crate::security::EncryptionAlgorithm::Aes256Gcm),
                ("agent_chat", crate::security::EncryptionAlgorithm::Aes256Gcm),
                ("agent_config", crate::security::EncryptionAlgorithm::ChaCha20Poly1305),
            ];

            for (key_id, algorithm) in &keys_to_generate {
                if !security_manager.encryption_key_exists(key_id).await? {
                    security_manager
                        .generate_encryption_key(key_id, algorithm.clone())
                        .await?;
                    debug!("Generated encryption key: {}", key_id);
                }
            }
        }
        Ok(())
    }

    /// Validate security context and check permissions
    async fn validate_security_context(&self, action: &str, resource: &str) -> Result<bool> {
        if let (Some(security_manager), Some(context)) = (&self.security_manager, &self.security_context) {
            // Validate the security context
            if !security_manager.validate_context(context).await? {
                return Err(AgentError::security("Invalid security context".to_string()));
            }

            // Evaluate security policies for this operation
            let policy_decision = security_manager
                .evaluate_all_policies(context, resource, action)
                .await?;

            if !policy_decision.granted {
                let deciding_policy = policy_decision.deciding_policy.clone().unwrap_or("unknown".to_string());

                // Log policy-based denial
                security_manager
                    .log_security_event(SecurityEvent::PolicyViolation {
                        user_id: Some(context.user_id.clone()),
                        policy: deciding_policy.clone(),
                        violation: policy_decision.reason.clone(),
                    })
                    .await?;

                return Err(AgentError::security(format!(
                    "Policy denied: {} - {}",
                    policy_decision.reason,
                    deciding_policy
                )));
            }

            // Check specific permission for the action
            let has_permission = security_manager
                .check_permission(context, resource, action)
                .await?;

            if !has_permission {
                // Log authorization failure
                security_manager
                    .log_security_event(SecurityEvent::AuthorizationCheck {
                        user_id: context.user_id.clone(),
                        resource: resource.to_string(),
                        action: action.to_string(),
                        granted: false,
                    })
                    .await?;

                return Err(AgentError::security(format!(
                    "Permission denied: {} on {}",
                    action, resource
                )));
            }

            // Log successful authorization
            security_manager
                .log_security_event(SecurityEvent::AuthorizationCheck {
                    user_id: context.user_id.clone(),
                    resource: resource.to_string(),
                    action: action.to_string(),
                    granted: true,
                })
                .await?;

            Ok(true)
        } else {
            // If no security manager is configured, allow the operation but log it
            debug!("No security context available, allowing operation: {} on {}", action, resource);
            Ok(true)
        }
    }

    /// Validate and sanitize input message for security
    async fn validate_input_message(&self, message: &str) -> Result<String> {
        if let Some(security_manager) = &self.security_manager {
            // Check for potential security threats in the input
            if message.len() > 50000 {
                return Err(AgentError::security(
                    "Input message exceeds maximum allowed length".to_string(),
                ));
            }

            // Check for potential injection patterns
            let suspicious_patterns = [
                "javascript:",
                "<script",
                "eval(",
                "exec(",
                "system(",
                "shell_exec(",
                "passthru(",
                "file_get_contents(",
                "file_put_contents(",
                "fopen(",
                "fwrite(",
                "include(",
                "require(",
            ];

            let lower_message = message.to_lowercase();
            for pattern in &suspicious_patterns {
                if lower_message.contains(pattern) {
                    // Log security incident
                    if let Some(_context) = &self.security_context {
                        security_manager
                            .log_security_event(SecurityEvent::SecurityIncident {
                                incident_type: "potential_injection".to_string(),
                                severity: "medium".to_string(),
                                description: format!("Suspicious pattern detected: {}", pattern),
                            })
                            .await?;
                    }

                    warn!("Suspicious pattern detected in input: {}", pattern);
                }
            }

            // Basic sanitization - remove null bytes and control characters
            let sanitized = message
                .chars()
                .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
                .collect::<String>();

            Ok(sanitized)
        } else {
            Ok(message.to_string())
        }
    }

    /// Log security event for tool execution
    async fn log_tool_execution_event(&self, tool_name: &str, _success: bool) -> Result<()> {
        if let (Some(security_manager), Some(context)) = (&self.security_manager, &self.security_context) {
            security_manager
                .log_security_event(SecurityEvent::DataAccess {
                    user_id: context.user_id.clone(),
                    resource: format!("tool:{}", tool_name),
                    action: "execute".to_string(),
                    sensitive: tool_name.contains("file") || tool_name.contains("system"),
                })
                .await?;
        }
        Ok(())
    }

    /// Add agent-specific security policies
    pub async fn add_agent_security_policies(&self) -> Result<()> {
        if let Some(security_manager) = &self.security_manager {
            let agent_policies = vec![
                SecurityPolicy {
                    name: "agent_memory_encryption".to_string(),
                    description: "Require encryption for sensitive memory content".to_string(),
                    version: "1.0".to_string(),
                    effect: PolicyEffect::Allow,
                    resources: vec!["memory:credentials".to_string(), "memory:personal".to_string(), "memory:confidential".to_string()],
                    actions: vec!["encrypt".to_string()],
                    conditions: vec![],
                    priority: 100,
                    enabled: true,
                    metadata: std::collections::HashMap::new(),
                },
                SecurityPolicy {
                    name: "agent_rate_limit".to_string(),
                    description: "Rate limit agent operations to prevent abuse".to_string(),
                    version: "1.0".to_string(),
                    effect: PolicyEffect::Deny,
                    resources: vec!["*".to_string()],
                    actions: vec!["*".to_string()],
                    conditions: vec![PolicyCondition {
                        field: "request.count_per_minute".to_string(),
                        operator: PolicyOperator::GreaterThan,
                        value: PolicyValue::Number(100.0),
                        negate: false,
                    }],
                    priority: 200,
                    enabled: true,
                    metadata: std::collections::HashMap::new(),
                },
                SecurityPolicy {
                    name: "agent_tool_security".to_string(),
                    description: "Enhanced security for sensitive tool operations".to_string(),
                    version: "1.0".to_string(),
                    effect: PolicyEffect::Deny,
                    resources: vec!["tool:file_system".to_string(), "tool:network".to_string(), "tool:system".to_string()],
                    actions: vec!["execute".to_string()],
                    conditions: vec![PolicyCondition {
                        field: "user.roles".to_string(),
                        operator: PolicyOperator::Contains,
                        value: PolicyValue::String("admin".to_string()),
                        negate: true,
                    }],
                    priority: 150,
                    enabled: true,
                    metadata: std::collections::HashMap::new(),
                },
            ];

            for policy in agent_policies {
                if !security_manager.security_policy_exists(&policy.name).await? {
                    security_manager.add_security_policy(policy).await?;
                    debug!("Added agent security policy");
                }
            }
        }
        Ok(())
    }

    /// Terminate security context
    pub async fn terminate_security_context(&mut self, reason: String) -> Result<()> {
        if let (Some(security_manager), Some(context)) = (&self.security_manager, &self.security_context) {
            security_manager
                .terminate_context(context, reason)
                .await?;
            self.security_context = None;
            info!("Security context terminated");
        }
        Ok(())
    }

    /// Start a new conversation
    pub async fn start_conversation(&mut self, title: Option<String>) -> Result<String> {
        let conversation_id = self.conversation_manager.start_conversation(title).await?;
        self.current_conversation_id = Some(conversation_id.clone());

        info!("Started new conversation: {}", conversation_id);
        Ok(conversation_id)
    }

    /// Send a message and get a response
    pub async fn chat<S: Into<String>>(&mut self, message: S) -> Result<String> {
        let message = message.into();
        debug!("Processing chat message: {} chars", message.len());

        // Security validation: Check permissions for chat operation
        self.validate_security_context("chat", "conversation").await?;

        // Security validation: Sanitize input message
        let sanitized_message = self.validate_input_message(&message).await?;

        // Ensure we have an active conversation
        if self.current_conversation_id.is_none() {
            self.start_conversation(None).await?;
        }

        // Encrypt sensitive chat content if needed
        let processed_message = self.encrypt_chat_message_if_needed(&sanitized_message).await?;

        // Add user message to conversation (using processed message)
        let user_message = ChatMessage::user(processed_message);
        self.conversation_manager
            .add_message(user_message.clone())
            .await?;

        // Get conversation history for context
        let history = self
            .conversation_manager
            .get_recent_history(self.config.agent.max_history_length)
            .await?;

        // Create chat request with validated conversation history
        let filtered_history = self.validate_and_clean_conversation_history(&history)?;

        let mut request = ChatRequest {
            model: self.config.anthropic.model.clone(),
            max_tokens: self.config.anthropic.max_tokens,
            messages: filtered_history.clone(),
            system: self.config.agent.system_prompt.clone(),
            tools: Some(self.tool_orchestrator.get_tool_definitions()),
            tool_choice: None,
            temperature: Some(self.config.anthropic.temperature),
            stream: Some(self.config.agent.enable_streaming),
        };

        // Log request size for debugging
        info!(
            "Initial API request: {} messages, {} tools",
            filtered_history.len(),
            request.tools.as_ref().map(|t| t.len()).unwrap_or(0)
        );

        // Send request to Anthropic with timeout (use HTTP client timeout + buffer)
        info!("Making initial API call to Anthropic...");
        let timeout_duration =
            std::time::Duration::from_secs(self.config.anthropic.timeout_seconds + 30);
        info!(
            "Using timeout of {} seconds for API call",
            timeout_duration.as_secs()
        );

        // Use streaming if enabled, otherwise use regular chat
        let mut response = if self.config.agent.enable_streaming {
            info!("Using streaming API call");
            let api_call = self.anthropic_client.chat_stream(request.clone());
            match tokio::time::timeout(timeout_duration, api_call).await {
                Ok(result) => {
                    info!("Initial streaming API call completed successfully");
                    result?
                }
                Err(_) => {
                    error!(
                        "Initial streaming API call timed out after {} seconds!",
                        timeout_duration.as_secs()
                    );
                    return Err(AgentError::anthropic_api(format!("Initial streaming API call timed out after {} seconds - this may indicate a network issue or API problem", timeout_duration.as_secs())));
                }
            }
        } else {
            info!("Using non-streaming API call");
            let api_call = self.anthropic_client.chat(request.clone());
            match tokio::time::timeout(timeout_duration, api_call).await {
                Ok(result) => {
                    info!("Initial API call completed successfully");
                    result?
                }
                Err(_) => {
                    error!(
                        "Initial API call timed out after {} seconds!",
                        timeout_duration.as_secs()
                    );
                    return Err(AgentError::anthropic_api(format!("Initial API call timed out after {} seconds - this may indicate a network issue or API problem", timeout_duration.as_secs())));
                }
            }
        };

        // Process tool calls if any
        let mut tool_iterations = 0;
        let max_tool_iterations = self.config.agent.max_tool_iterations;
        let mut recent_tool_calls: Vec<String> = Vec::new(); // Track recent tool calls for loop detection

        while response.content.iter().any(|block| {
            matches!(
                block,
                crate::anthropic::models::ContentBlock::ToolUse { .. }
                    | crate::anthropic::models::ContentBlock::ServerToolUse { .. }
            )
        }) && tool_iterations < max_tool_iterations
        {
            tool_iterations += 1;
            debug!("Processing tool calls (iteration {})", tool_iterations);

            // Check for infinite loops - detect if we're repeating the same tool calls
            let current_tool_calls: Vec<String> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    crate::anthropic::models::ContentBlock::ToolUse { name, input, .. }
                    | crate::anthropic::models::ContentBlock::ServerToolUse {
                        name, input, ..
                    } => Some(format!(
                        "{}:{}",
                        name,
                        serde_json::to_string(&input).unwrap_or_default()
                    )),
                    _ => None,
                })
                .collect();

            // Check if we've seen these exact tool calls recently (within last 2 iterations)
            let tool_call_signature = current_tool_calls.join("|");
            if recent_tool_calls
                .iter()
                .rev()
                .take(2)
                .any(|prev| prev == &tool_call_signature)
            {
                warn!("INFINITE LOOP DETECTED: Same tool calls repeated. Breaking loop to prevent infinite iterations.");
                warn!("Repeated tool call signature: {}", tool_call_signature);

                // Create error results for the repeated tool calls to satisfy API requirements
                let loop_error_results: Vec<_> = response.content.iter()
                    .filter_map(|block| {
                        if let crate::anthropic::models::ContentBlock::ToolUse { id, name, input }
                            | crate::anthropic::models::ContentBlock::ServerToolUse { id, name, input } = block
                        {
                            let missing_params = if name == "local_file_editor" {
                                let has_command = input.get("command").is_some();
                                let has_path = input.get("path").is_some();
                                let has_old_str = input.get("old_str").is_some();
                                let has_new_str = input.get("new_str").is_some();

                                if has_command && has_path && has_old_str && !has_new_str {
                                    " You are missing the 'new_str' parameter which is required for str_replace operations."
                                } else if has_command && !has_path {
                                    " You are missing the 'path' parameter which is required for all file operations."
                                } else {
                                    " Please check that you have provided all required parameters."
                                }
                            } else {
                                " Please check your tool call parameters."
                            };

                            Some(crate::tools::ToolResult::error(
                                format!("🔄 INFINITE LOOP DETECTED - CORRECTIVE ACTION REQUIRED:\n\
❌ PROBLEM: The same {} tool call was repeated {} times with identical parameters.\n\
🔍 ANALYSIS:{}\n\
\n\
💡 CORRECTIVE ACTIONS:\n\
1. Check that you have provided ALL required parameters\n\
2. If missing parameters, add them to your next tool call\n\
3. If all parameters are present, try a different approach\n\
4. Consider breaking the task into smaller steps\n\
\n\
⚠️  IMPORTANT: Do not repeat the same incomplete tool call again.\n\
🎯 NEXT STEP: Provide a complete tool call with all required parameters or try a different approach.",
                                name, recent_tool_calls.len(), missing_params)
                            ).to_content_block(id.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();

                if !loop_error_results.is_empty() {
                    let loop_error_message = ChatMessage {
                        role: MessageRole::User,
                        content: loop_error_results,
                        id: Some(Uuid::new_v4().to_string()),
                        timestamp: Some(chrono::Utc::now()),
                    };
                    self.conversation_manager
                        .add_message(loop_error_message)
                        .await?;
                }
                break;
            }

            // Add current tool calls to recent history
            recent_tool_calls.push(tool_call_signature);
            // Keep only last 5 tool call signatures to prevent memory bloat
            if recent_tool_calls.len() > 5 {
                recent_tool_calls.remove(0);
            }

            // Add assistant message with tool calls
            let assistant_message = ChatMessage {
                role: MessageRole::Assistant,
                content: response.content.clone(),
                id: Some(response.id.clone()),
                timestamp: Some(chrono::Utc::now()),
            };
            self.conversation_manager
                .add_message(assistant_message)
                .await?;

            // Security validation: Check permissions for tool execution
            self.validate_security_context("execute", "tools").await?;

            // Log tool execution attempt for security audit
            for block in &response.content {
                if let crate::anthropic::models::ContentBlock::ToolUse { name, .. }
                | crate::anthropic::models::ContentBlock::ServerToolUse { name, .. } = block
                {
                    self.log_tool_execution_event(name, true).await?;
                }
            }

            // Execute tools and collect results
            info!("Executing tools for iteration {}...", tool_iterations);
            let tool_execution_future = self.tool_orchestrator.execute_tools(&response.content);
            let mut tool_results = match tokio::time::timeout(
                std::time::Duration::from_secs(120),
                tool_execution_future,
            )
            .await
            {
                Ok(result) => {
                    info!("Tool execution completed successfully");
                    result?
                }
                Err(_) => {
                    error!("Tool execution timed out after 120 seconds!");
                    return Err(AgentError::tool("memory_stats", "Tool execution timed out - this may indicate a performance issue or infinite loop"));
                }
            };
            // Include any server tool results that may have been returned by the API
            let server_results = self.tool_orchestrator.take_server_tool_results();
            if !server_results.is_empty() {
                debug!("Received {} server tool results", server_results.len());
                tool_results.extend(server_results);
            }

            info!(
                "Tool execution completed: {} tool_use blocks, {} tool_result blocks",
                response
                    .content
                    .iter()
                    .filter(|b| matches!(
                        b,
                        crate::anthropic::models::ContentBlock::ToolUse { .. }
                            | crate::anthropic::models::ContentBlock::ServerToolUse { .. }
                    ))
                    .count(),
                tool_results.len()
            );

            // CRITICAL: Always create tool result message if there were any tool uses
            // This ensures every tool_use block has a corresponding tool_result block
            // The API requires this pairing regardless of whether tools succeed or fail
            let has_tool_uses = response.content.iter().any(|block| {
                matches!(
                    block,
                    crate::anthropic::models::ContentBlock::ToolUse { .. }
                        | crate::anthropic::models::ContentBlock::ServerToolUse { .. }
                )
            });

            if has_tool_uses {
                // If we have tool uses but no results (shouldn't happen), create error results
                let final_tool_results = if tool_results.is_empty() {
                    // Create error results for any unmatched tool uses
                    response
                        .content
                        .iter()
                        .filter_map(|block| {
                            if let crate::anthropic::models::ContentBlock::ToolUse { id, .. } =
                                block
                            {
                                Some(
                                    crate::tools::ToolResult::error(
                                        "Tool execution failed - no result generated".to_string(),
                                    )
                                    .to_content_block(id.clone()),
                                )
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    tool_results
                };

                // Only add the message if it has content
                if !final_tool_results.is_empty() {
                    let tool_result_message = ChatMessage {
                        role: MessageRole::User,
                        content: final_tool_results,
                        id: Some(Uuid::new_v4().to_string()),
                        timestamp: Some(chrono::Utc::now()),
                    };
                    self.conversation_manager
                        .add_message(tool_result_message)
                        .await?;

                    // Capture optional follow-up instructions from the user
                    if self.config.agent.enable_human_in_loop
                        && self.config.agent.human_input_after_iterations.is_none()
                    {
                        if let Ok(extra) = self
                            .request_human_input(&self.config.agent.human_input_prompt)
                            .await
                        {
                            if !extra.trim().is_empty() {
                                let msg = ChatMessage::user(extra);
                                self.conversation_manager.add_message(msg).await?;
                            }
                        }
                    }
                } else {
                    warn!("Skipping tool result message with empty content");
                }
            }

            // Check if we're about to hit the limit
            if tool_iterations >= max_tool_iterations {
                warn!(
                    "Maximum tool iterations ({}) reached, stopping tool execution",
                    max_tool_iterations
                );
                break;
            }

            // Check for human-in-the-loop intervention
            if self.config.agent.enable_human_in_loop {
                if let Some(human_input_threshold) = self.config.agent.human_input_after_iterations
                {
                    if tool_iterations >= human_input_threshold {
                        info!(
                            "Human-in-the-loop threshold ({}) reached after {} iterations",
                            human_input_threshold, tool_iterations
                        );
                        let prompt = &self.config.agent.human_input_prompt;
                        if let Ok(input) = self.request_human_input(prompt).await {
                            if !input.trim().is_empty() {
                                let user_msg = ChatMessage::user(input);
                                self.conversation_manager.add_message(user_msg).await?;
                            }
                        }
                    }
                }
            }

            // Get updated history and make another request
            // Use smaller history for follow-up calls to reduce request size and improve performance
            let follow_up_history_length = std::cmp::min(self.config.agent.max_history_length, 20);
            info!(
                "Getting updated conversation history for next API call (last {} messages)...",
                follow_up_history_length
            );
            let history_future = self
                .conversation_manager
                .get_recent_history(follow_up_history_length);
            let updated_history = match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                history_future,
            )
            .await
            {
                Ok(result) => {
                    info!("Successfully retrieved conversation history");
                    result?
                }
                Err(_) => {
                    error!("Conversation history retrieval timed out after 30 seconds!");
                    return Err(AgentError::memory("Memory operation timed out - this may indicate a database lock or performance issue".to_string()));
                }
            };

            // Filter out messages with empty content and validate tool_use/tool_result pairing
            let filtered_messages =
                self.validate_and_clean_conversation_history(&updated_history)?;

            info!(
                "Follow-up API request: {} messages, {} tools",
                filtered_messages.len(),
                request.tools.as_ref().map(|t| t.len()).unwrap_or(0)
            );
            request.messages = filtered_messages;

            // Add timeout to prevent infinite hanging (use HTTP client timeout + buffer)
            let timeout_duration =
                std::time::Duration::from_secs(self.config.anthropic.timeout_seconds + 30);

            // Use streaming if enabled, otherwise use regular chat
            response = if self.config.agent.enable_streaming {
                let api_call = self.anthropic_client.chat_stream(request.clone());
                match tokio::time::timeout(timeout_duration, api_call).await {
                    Ok(result) => {
                        info!("Follow-up streaming API call completed successfully");
                        result?
                    }
                    Err(_) => {
                        error!(
                            "Follow-up streaming API call timed out after {} seconds!",
                            timeout_duration.as_secs()
                        );
                        return Err(AgentError::anthropic_api(format!("Follow-up streaming API call timed out after {} seconds - this may indicate a network issue or API problem", timeout_duration.as_secs())));
                    }
                }
            } else {
                let api_call = self.anthropic_client.chat(request.clone());
                match tokio::time::timeout(timeout_duration, api_call).await {
                    Ok(result) => {
                        info!("Follow-up API call completed successfully");
                        result?
                    }
                    Err(_) => {
                        error!(
                            "Follow-up API call timed out after {} seconds!",
                            timeout_duration.as_secs()
                        );
                        return Err(AgentError::anthropic_api(format!("Follow-up API call timed out after {} seconds - this may indicate a network issue or API problem", timeout_duration.as_secs())));
                    }
                }
            };
        }

        // Tool iterations completed (either no more tool uses or max iterations reached)
        info!(
            "Tool processing loop completed after {} iterations",
            tool_iterations
        );

        // Check if the final response has tool uses that need results
        let final_has_tool_uses = response.content.iter().any(|block| {
            matches!(
                block,
                crate::anthropic::models::ContentBlock::ToolUse { .. }
                    | crate::anthropic::models::ContentBlock::ServerToolUse { .. }
            )
        });
        info!("Final response has tool uses: {}", final_has_tool_uses);

        // CRITICAL: Check for unprocessed tool uses BEFORE adding message to conversation
        // This prevents corrupting conversation history with incomplete tool calls
        if final_has_tool_uses {
            error!("Final response contains tool uses but max iterations reached. This indicates a fundamental issue.");

            // Count the unprocessed tool calls
            let tool_use_count = response
                .content
                .iter()
                .filter(|block| {
                    matches!(
                        block,
                        crate::anthropic::models::ContentBlock::ToolUse { .. }
                            | crate::anthropic::models::ContentBlock::ServerToolUse { .. }
                    )
                })
                .count();

            // Instead of creating placeholder results, return a detailed error
            let error_message = format!(
                "Maximum iterations ({}) reached with {} unprocessed tool calls.\n\
                \n\
                This suggests:\n\
                1. The model is generating incomplete tool calls (missing required parameters)\n\
                2. Tool validation is not providing effective feedback to the model\n\
                3. The conversation may need human intervention\n\
                \n\
                Check the logs for repeated tool signatures to identify the root cause.\n\
                \n\
                Recommendation: Review the conversation history and tool call patterns.",
                self.config.agent.max_tool_iterations, tool_use_count
            );

            // DO NOT add the problematic message to conversation history
            warn!("Skipping addition of final message with unprocessed tool uses to prevent conversation corruption");
            return Err(AgentError::tool("iteration_limit", &error_message));
        }

        // Only add final assistant response if it doesn't have unprocessed tool uses
        info!("Adding final assistant message to conversation...");
        let final_message = ChatMessage {
            role: MessageRole::Assistant,
            content: response.content.clone(),
            id: Some(response.id),
            timestamp: Some(chrono::Utc::now()),
        };
        self.conversation_manager
            .add_message(final_message.clone())
            .await?;
        info!("Final assistant message added successfully");

        // Extract text response
        let response_text = final_message.get_text();

        info!("Chat response generated: {} chars", response_text.len());
        Ok(response_text)
    }

    /// Get conversation history
    pub async fn get_conversation_history(&self) -> Result<Vec<ChatMessage>> {
        self.conversation_manager
            .get_recent_history(usize::MAX)
            .await
    }

    /// Search memory
    pub async fn search_memory<S: Into<String>>(
        &self,
        query: S,
        limit: usize,
    ) -> Result<Vec<crate::memory::SearchResult>> {
        // Security validation: Check permissions for memory search
        self.validate_security_context("read", "memory").await?;

        let memory_manager = self.memory_manager.lock().await;
        let results = memory_manager.search_raw(&query.into(), limit).await?;

        // Decrypt any encrypted content in the results
        let mut decrypted_results = Vec::new();
        for mut result in results {
            result.content = self.decrypt_content_if_needed(&result.content).await?;
            decrypted_results.push(result);
        }

        Ok(decrypted_results)
    }

    /// Decrypt content if it was encrypted
    async fn decrypt_content_if_needed(&self, content: &str) -> Result<String> {
        if let Some(security_manager) = &self.security_manager {
            if content.starts_with("ENCRYPTED:") {
                let encrypted_content = &content[10..]; // Remove "ENCRYPTED:" prefix

                // Decrypt the content
                let decrypted = security_manager
                    .decrypt_string(encrypted_content, "agent_memory")
                    .await?;

                // Log data decryption event
                if let Some(context) = &self.security_context {
                    security_manager
                        .log_security_event(SecurityEvent::DataAccess {
                            user_id: context.user_id.clone(),
                            resource: "encrypted_memory".to_string(),
                            action: "decrypt".to_string(),
                            sensitive: true,
                        })
                        .await?;
                }

                Ok(decrypted)
            } else {
                Ok(content.to_string())
            }
        } else {
            Ok(content.to_string())
        }
    }

    /// Save information to memory
    pub async fn save_to_memory<S: Into<String>>(
        &mut self,
        content: S,
        entry_type: S,
    ) -> Result<()> {
        // Security validation: Check permissions for memory write
        self.validate_security_context("write", "memory").await?;

        let content_str = content.into();
        let entry_type_str = entry_type.into();

        // Encrypt sensitive content before storing
        let processed_content = self.encrypt_sensitive_content(&content_str, &entry_type_str).await?;

        let mut memory_manager = self.memory_manager.lock().await;
        let metadata = std::collections::HashMap::new();
        memory_manager
            .save_memory(processed_content, entry_type_str, metadata)
            .await?;
        Ok(())
    }

    /// Encrypt sensitive content based on content type and security policies
    async fn encrypt_sensitive_content(&self, content: &str, entry_type: &str) -> Result<String> {
        if let Some(security_manager) = &self.security_manager {
            // Determine if content should be encrypted based on type and content analysis
            let should_encrypt = self.should_encrypt_content(content, entry_type).await?;

            if should_encrypt {
                // Encrypt the content using the memory encryption key
                let encrypted = security_manager
                    .encrypt_string(content, "agent_memory")
                    .await?;

                // Log data encryption event
                if let Some(context) = &self.security_context {
                    security_manager
                        .log_security_event(SecurityEvent::DataAccess {
                            user_id: context.user_id.clone(),
                            resource: "encrypted_memory".to_string(),
                            action: "encrypt".to_string(),
                            sensitive: true,
                        })
                        .await?;
                }

                // Prefix with encryption marker for later identification
                Ok(format!("ENCRYPTED:{}", encrypted))
            } else {
                Ok(content.to_string())
            }
        } else {
            Ok(content.to_string())
        }
    }

    /// Determine if content should be encrypted based on sensitivity analysis
    async fn should_encrypt_content(&self, content: &str, entry_type: &str) -> Result<bool> {
        // Always encrypt certain types
        let always_encrypt_types = [
            "credentials", "password", "token", "key", "secret",
            "personal", "private", "confidential", "sensitive"
        ];

        if always_encrypt_types.iter().any(|&t| entry_type.to_lowercase().contains(t)) {
            return Ok(true);
        }

        // Content-based sensitivity detection
        let sensitive_patterns = [
            r"(?i)password\s*[:=]\s*\S+",
            r"(?i)api[_\s]*key\s*[:=]\s*\S+",
            r"(?i)token\s*[:=]\s*\S+",
            r"(?i)secret\s*[:=]\s*\S+",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", // Email addresses
            r"\b\d{3}-\d{2}-\d{4}\b", // SSN pattern
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", // Credit card pattern
        ];

        for pattern in &sensitive_patterns {
            if regex::Regex::new(pattern)
                .map_err(|e| AgentError::validation(format!("Invalid regex pattern: {}", e)))?
                .is_match(content)
            {
                return Ok(true);
            }
        }

        // Policy-based encryption decision
        if let (Some(security_manager), Some(context)) = (&self.security_manager, &self.security_context) {
            // Check if there's a policy requiring encryption for this content type
            let policy_decision = security_manager
                .evaluate_all_policies(context, &format!("memory:{}", entry_type), "encrypt")
                .await?;

            return Ok(policy_decision.granted);
        }

        Ok(false)
    }

    /// Encrypt chat message if it contains sensitive information
    async fn encrypt_chat_message_if_needed(&self, message: &str) -> Result<String> {
        if let Some(security_manager) = &self.security_manager {
            // Check if message contains sensitive information
            let should_encrypt = self.should_encrypt_content(message, "chat").await?;

            if should_encrypt {
                // Encrypt the message using the chat encryption key
                let encrypted = security_manager
                    .encrypt_string(message, "agent_chat")
                    .await?;

                // Log chat encryption event
                if let Some(context) = &self.security_context {
                    security_manager
                        .log_security_event(SecurityEvent::DataAccess {
                            user_id: context.user_id.clone(),
                            resource: "encrypted_chat".to_string(),
                            action: "encrypt".to_string(),
                            sensitive: true,
                        })
                        .await?;
                }

                // Prefix with encryption marker for later identification
                Ok(format!("ENCRYPTED_CHAT:{}", encrypted))
            } else {
                Ok(message.to_string())
            }
        } else {
            Ok(message.to_string())
        }
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> Result<crate::memory::MemoryStats> {
        // Security validation: Check permissions for memory stats
        self.validate_security_context("read", "memory_stats").await?;

        let memory_manager = self.memory_manager.lock().await;
        memory_manager.get_stats().await
    }

    /// Get security statistics and policy evaluation metrics
    pub async fn get_security_stats(&self) -> Result<Option<crate::security::policy::PolicyStatistics>> {
        // Security validation: Check permissions for security stats
        self.validate_security_context("read", "security_stats").await?;

        if let Some(security_manager) = &self.security_manager {
            let stats = security_manager.get_policy_statistics().await?;
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    /// Register a custom tool
    pub fn register_tool<T: crate::tools::Tool + 'static>(&mut self, tool: T) {
        self.tool_orchestrator.register_tool(tool);
    }

    /// Get available tools
    pub fn get_available_tools(&self) -> Vec<String> {
        self.tool_orchestrator.get_tool_names()
    }

    /// Execute a specific tool directly
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<crate::tools::ToolResult> {
        // Security validation: Check permissions for direct tool execution
        self.validate_security_context("execute", "tools").await?;

        // Log tool execution for security audit
        self.log_tool_execution_event(tool_name, true).await?;

        self.tool_orchestrator
            .execute_tool_direct(tool_name, input)
            .await
    }

    /// Update configuration
    pub async fn update_config(&mut self, config: AgentConfig) -> Result<()> {
        config.validate()?;

        // Update Anthropic client if needed
        if config.anthropic != self.config.anthropic {
            self.anthropic_client
                .update_config(config.anthropic.clone())?;
        }

        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get current conversation ID
    pub fn current_conversation_id(&self) -> Option<&str> {
        self.current_conversation_id.as_deref()
    }

    /// Switch to a different conversation
    pub async fn switch_conversation(&mut self, conversation_id: String) -> Result<()> {
        // Verify the conversation exists
        let memory_manager = self.memory_manager.lock().await;
        if memory_manager
            .get_conversation(&conversation_id)
            .await?
            .is_some()
        {
            self.current_conversation_id = Some(conversation_id);
            Ok(())
        } else {
            Err(AgentError::invalid_input("Conversation not found"))
        }
    }

    /// Build and finalize memory (no-op for JSON storage)
    pub async fn finalize_memory(&mut self) -> Result<()> {
        // No-op for JSON storage - data is already persisted
        Ok(())
    }

    /// Request human input during agent execution
    /// This is a placeholder for human-in-the-loop functionality
    pub async fn request_human_input(&self, prompt: &str) -> Result<String> {
        use crossterm::terminal;
        use std::io::{self, Write};

        // If raw mode is enabled (e.g., interactive chat), temporarily disable it
        let was_raw = terminal::is_raw_mode_enabled().unwrap_or(false);
        if was_raw {
            let _ = terminal::disable_raw_mode();
        }

        println!("\n{}", prompt);
        print!("> ");
        io::stdout().flush().map_err(AgentError::Io)?;

        let mut input = String::new();
        io::stdin().read_line(&mut input).map_err(AgentError::Io)?;

        if was_raw {
            let _ = terminal::enable_raw_mode();
        }

        Ok(input.trim().to_string())
    }

    /// Set maximum tool iterations
    pub fn set_max_tool_iterations(&mut self, max_iterations: usize) {
        self.config.agent.max_tool_iterations = max_iterations;
    }

    /// Get maximum tool iterations
    pub fn get_max_tool_iterations(&self) -> usize {
        self.config.agent.max_tool_iterations
    }

    /// Enable or disable human-in-the-loop
    pub fn set_human_in_loop(&mut self, enabled: bool) {
        self.config.agent.enable_human_in_loop = enabled;
    }

    /// Check if human-in-the-loop is enabled
    pub fn is_human_in_loop_enabled(&self) -> bool {
        self.config.agent.enable_human_in_loop
    }

    /// Get the effective system prompt being used
    pub fn get_system_prompt(&self) -> Option<&str> {
        self.config.agent.system_prompt.as_deref()
    }

    /// Set a custom system prompt
    pub fn set_system_prompt<S: Into<String>>(&mut self, prompt: Option<S>) {
        self.config.agent.system_prompt = prompt.map(|p| p.into());
    }

    /// Enable or disable streaming responses
    pub fn set_streaming_enabled(&mut self, enabled: bool) {
        self.config.agent.enable_streaming = enabled;
    }

    /// Check if streaming is enabled
    pub fn is_streaming_enabled(&self) -> bool {
        self.config.agent.enable_streaming
    }

    /// Validate and clean conversation history to prevent tool_use/tool_result pairing issues
    fn validate_and_clean_conversation_history(
        &self,
        history: &[ChatMessage],
    ) -> Result<Vec<crate::anthropic::models::ApiMessage>> {
        use crate::anthropic::models::{ApiMessage, ContentBlock, MessageRole};
        use std::collections::{HashMap, HashSet};

        let mut cleaned_messages = Vec::new();
        let mut pending_tool_uses = HashMap::new(); // Track tool_use IDs that need results
        let mut orphaned_tool_results = HashSet::new();

        // Process messages in order to maintain proper tool_use/tool_result pairing
        for msg in history.iter().filter(|msg| !msg.has_empty_content()) {
            let mut cleaned_content = Vec::new();
            let mut has_valid_content = false;

            match msg.role {
                MessageRole::Assistant => {
                    // Assistant messages can contain tool_use blocks
                    for block in &msg.content {
                        match block {
                            ContentBlock::ToolUse { id, .. } => {
                                // Track this tool_use as pending a result
                                pending_tool_uses.insert(id.clone(), true);
                                cleaned_content.push(block.clone());
                                has_valid_content = true;
                            }
                            _ => {
                                cleaned_content.push(block.clone());
                                has_valid_content = true;
                            }
                        }
                    }
                }
                MessageRole::User => {
                    // User messages can contain tool_result blocks
                    for block in &msg.content {
                        match block {
                            ContentBlock::ToolResult { tool_use_id, .. } => {
                                // Check if this tool_result has a corresponding pending tool_use
                                if pending_tool_uses.remove(tool_use_id).is_some() {
                                    cleaned_content.push(block.clone());
                                    has_valid_content = true;
                                } else {
                                    warn!("Removing orphaned tool_result block with ID: {} (no corresponding tool_use found)", tool_use_id);
                                    orphaned_tool_results.insert(tool_use_id.clone());
                                }
                            }
                            _ => {
                                cleaned_content.push(block.clone());
                                has_valid_content = true;
                            }
                        }
                    }
                }
                _ => {
                    // System messages and others - include as-is
                    cleaned_content = msg.content.clone();
                    has_valid_content = !cleaned_content.is_empty();
                }
            }

            // Only add the message if it has valid content after cleaning
            if has_valid_content && !cleaned_content.is_empty() {
                cleaned_messages.push(ApiMessage {
                    role: msg.role.clone(),
                    content: cleaned_content,
                });
            }
        }

        // Log validation results
        if !orphaned_tool_results.is_empty() {
            warn!(
                "Removed {} orphaned tool_result blocks: {:?}",
                orphaned_tool_results.len(),
                orphaned_tool_results
            );
        }

        if !pending_tool_uses.is_empty() {
            warn!(
                "Found {} tool_use blocks without results: {:?}",
                pending_tool_uses.len(),
                pending_tool_uses.keys().collect::<Vec<_>>()
            );
        }

        info!(
            "Conversation history validation: {} original messages, {} cleaned messages",
            history.len(),
            cleaned_messages.len()
        );

        Ok(cleaned_messages)
    }
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            custom_tools: Vec::new(),
        }
    }

    /// Set the configuration
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the Anthropic API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.anthropic.api_key = api_key.into();
        self
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.anthropic.model = model.into();
        self
    }

    /// Set the memory path
    pub fn with_memory_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        let path = path.into();
        self.config.memory.memory_path = path.clone();
        self.config.memory.index_path = path.with_extension("json");
        self
    }

    /// Add a custom tool
    pub fn with_tool<T: crate::tools::Tool + 'static>(mut self, tool: T) -> Self {
        self.custom_tools.push(Box::new(tool));
        self
    }

    /// Build the agent
    pub async fn build(self) -> Result<Agent> {
        let mut agent = Agent::new(self.config).await?;

        // Register custom tools
        for tool in self.custom_tools {
            agent.tool_orchestrator.register_boxed_tool(tool);
        }

        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
