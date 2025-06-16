//! Program of Thought (PoT) module
//!
//! This module implements Program of Thought reasoning, which generates and executes
//! code to solve problems that require computational thinking.

use super::{
    ModuleInfo, ReasoningMetrics, ReasoningModule, ReasoningStep, SpecializedModuleConfig,
};
use crate::anthropic::AnthropicClient;
use crate::dspy::{
    error::{DspyError, DspyResult},
    module::{Module, ModuleMetadata, ModuleStats},
    signature::Signature,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for Program of Thought module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramOfThoughtConfig {
    /// Base configuration
    pub base: SpecializedModuleConfig,
    /// Programming language to use
    pub language: ProgrammingLanguage,
    /// Whether to execute generated code
    pub execute_code: bool,
    /// Timeout for code execution in seconds
    pub execution_timeout_seconds: u64,
    /// Maximum code length
    pub max_code_length: usize,
    /// Whether to validate code syntax
    pub validate_syntax: bool,
    /// Allowed imports/libraries
    pub allowed_imports: Vec<String>,
    /// Security restrictions
    pub security_restrictions: SecurityRestrictions,
}

impl Default for ProgramOfThoughtConfig {
    fn default() -> Self {
        Self {
            base: SpecializedModuleConfig::default(),
            language: ProgrammingLanguage::Python,
            execute_code: false, // Disabled by default for security
            execution_timeout_seconds: 10,
            max_code_length: 2000,
            validate_syntax: true,
            allowed_imports: vec![
                "math".to_string(),
                "statistics".to_string(),
                "json".to_string(),
                "re".to_string(),
            ],
            security_restrictions: SecurityRestrictions::default(),
        }
    }
}

/// Supported programming languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgrammingLanguage {
    Python,
    JavaScript,
    Rust,
    Go,
}

impl ProgrammingLanguage {
    /// Get file extension for the language
    pub fn file_extension(&self) -> &str {
        match self {
            ProgrammingLanguage::Python => "py",
            ProgrammingLanguage::JavaScript => "js",
            ProgrammingLanguage::Rust => "rs",
            ProgrammingLanguage::Go => "go",
        }
    }

    /// Get execution command for the language
    pub fn execution_command(&self) -> &str {
        match self {
            ProgrammingLanguage::Python => "python3",
            ProgrammingLanguage::JavaScript => "node",
            ProgrammingLanguage::Rust => "rustc",
            ProgrammingLanguage::Go => "go run",
        }
    }
}

/// Security restrictions for code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRestrictions {
    /// Disallow file system operations
    pub disallow_file_operations: bool,
    /// Disallow network operations
    pub disallow_network_operations: bool,
    /// Disallow subprocess creation
    pub disallow_subprocess: bool,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Blacklisted functions/modules
    pub blacklisted_functions: Vec<String>,
}

impl Default for SecurityRestrictions {
    fn default() -> Self {
        Self {
            disallow_file_operations: true,
            disallow_network_operations: true,
            disallow_subprocess: true,
            max_memory_mb: 100,
            blacklisted_functions: vec![
                "exec".to_string(),
                "eval".to_string(),
                "open".to_string(),
                "input".to_string(),
                "__import__".to_string(),
            ],
        }
    }
}

/// Result of code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Whether execution was successful
    pub success: bool,
    /// Output from the code
    pub output: String,
    /// Error message if execution failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: Option<f64>,
}

/// Program of Thought reasoning module
#[derive(Debug)]
pub struct ProgramOfThought<I, O> {
    /// Module ID
    id: String,
    /// Module name
    name: String,
    /// Module signature
    signature: Signature<I, O>,
    /// Configuration
    config: ProgramOfThoughtConfig,
    /// Anthropic client
    anthropic_client: Arc<AnthropicClient>,
    /// Module metadata
    metadata: ModuleMetadata,
    /// Module statistics
    stats: Arc<RwLock<ModuleStats>>,
    /// Performance metrics
    metrics: Arc<RwLock<ReasoningMetrics>>,
    /// Last reasoning steps
    last_reasoning_steps: Arc<RwLock<Vec<ReasoningStep>>>,
    /// Last confidence score
    last_confidence: Arc<RwLock<f64>>,
    /// Last generated code
    last_generated_code: Arc<RwLock<Option<String>>>,
    /// Last execution result
    last_execution_result: Arc<RwLock<Option<CodeExecutionResult>>>,
}

impl<I, O> ProgramOfThought<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    /// Create a new Program of Thought module
    pub fn new(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let name = format!("ProgramOfThought_{}", &id[..8]);

        Self {
            id,
            name,
            signature,
            config: ProgramOfThoughtConfig::default(),
            anthropic_client,
            metadata: ModuleMetadata::default(),
            stats: Arc::new(RwLock::new(ModuleStats::default())),
            metrics: Arc::new(RwLock::new(ReasoningMetrics::default())),
            last_reasoning_steps: Arc::new(RwLock::new(Vec::new())),
            last_confidence: Arc::new(RwLock::new(0.0)),
            last_generated_code: Arc::new(RwLock::new(None)),
            last_execution_result: Arc::new(RwLock::new(None)),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        signature: Signature<I, O>,
        anthropic_client: Arc<AnthropicClient>,
        config: ProgramOfThoughtConfig,
    ) -> Self {
        let mut module = Self::new(signature, anthropic_client);
        module.config = config;
        module
    }

    /// Generate code generation prompt
    fn generate_code_prompt(&self, input: &I) -> DspyResult<String> {
        let input_str = serde_json::to_string_pretty(input)
            .map_err(|e| DspyError::serialization("input", &e.to_string()))?;

        let language_name = match self.config.language {
            ProgrammingLanguage::Python => "Python",
            ProgrammingLanguage::JavaScript => "JavaScript",
            ProgrammingLanguage::Rust => "Rust",
            ProgrammingLanguage::Go => "Go",
        };

        let mut prompt = format!(
            "You need to solve the following problem by writing {} code.\n\n\
            Problem: {}\n\n\
            Please write a complete {} program that:\n\
            1. Solves the given problem step by step\n\
            2. Includes clear comments explaining the logic\n\
            3. Outputs the final result in JSON format\n\n",
            language_name, input_str, language_name
        );

        // Add language-specific instructions
        match self.config.language {
            ProgrammingLanguage::Python => {
                prompt.push_str("Requirements:\n");
                prompt.push_str("- Use only standard library functions\n");
                prompt.push_str("- Print the final result as JSON using json.dumps()\n");
                if !self.config.allowed_imports.is_empty() {
                    prompt.push_str(&format!("- You may import: {}\n", self.config.allowed_imports.join(", ")));
                }
            }
            ProgrammingLanguage::JavaScript => {
                prompt.push_str("Requirements:\n");
                prompt.push_str("- Use only built-in JavaScript functions\n");
                prompt.push_str("- Output the final result using console.log(JSON.stringify(result))\n");
            }
            _ => {
                prompt.push_str("Requirements:\n");
                prompt.push_str("- Use only standard library functions\n");
                prompt.push_str("- Output the final result in JSON format\n");
            }
        }

        prompt.push_str("\nCode:\n```");
        prompt.push_str(self.config.language.file_extension());
        prompt.push('\n');

        Ok(prompt)
    }

    /// Extract code from response
    fn extract_code(&self, response: &str) -> DspyResult<String> {
        // Look for code blocks
        let code_block_start = format!("```{}", self.config.language.file_extension());
        let code_block_end = "```";

        if let Some(start_pos) = response.find(&code_block_start) {
            let code_start = start_pos + code_block_start.len();
            if let Some(end_pos) = response[code_start..].find(code_block_end) {
                let code = response[code_start..code_start + end_pos].trim();
                return Ok(code.to_string());
            }
        }

        // Fallback: look for any code block
        if let Some(start_pos) = response.find("```") {
            let code_start = start_pos + 3;
            // Skip language identifier if present
            let code_start = if let Some(newline_pos) = response[code_start..].find('\n') {
                code_start + newline_pos + 1
            } else {
                code_start
            };
            
            if let Some(end_pos) = response[code_start..].find("```") {
                let code = response[code_start..code_start + end_pos].trim();
                return Ok(code.to_string());
            }
        }

        Err(DspyError::module(self.name(), "No code block found in response"))
    }

    /// Validate code for security and syntax
    fn validate_code(&self, code: &str) -> DspyResult<()> {
        // Check code length
        if code.len() > self.config.max_code_length {
            return Err(DspyError::module(
                &self.name,
                &format!("Code exceeds maximum length of {} characters", self.config.max_code_length),
            ));
        }

        // Check for blacklisted functions
        for blacklisted in &self.config.security_restrictions.blacklisted_functions {
            if code.contains(blacklisted) {
                return Err(DspyError::module(
                    &self.name,
                    &format!("Code contains blacklisted function: {}", blacklisted),
                ));
            }
        }

        // Language-specific security checks
        match self.config.language {
            ProgrammingLanguage::Python => {
                if self.config.security_restrictions.disallow_file_operations {
                    let file_ops = ["open(", "file(", "with open", "os."];
                    for op in &file_ops {
                        if code.contains(op) {
                            return Err(DspyError::module(self.name(), "File operations are not allowed"));
                        }
                    }
                }

                if self.config.security_restrictions.disallow_network_operations {
                    let network_ops = ["urllib", "requests", "socket", "http"];
                    for op in &network_ops {
                        if code.contains(op) {
                            return Err(DspyError::module(self.name(), "Network operations are not allowed"));
                        }
                    }
                }

                if self.config.security_restrictions.disallow_subprocess {
                    let subprocess_ops = ["subprocess", "os.system", "os.popen"];
                    for op in &subprocess_ops {
                        if code.contains(op) {
                            return Err(DspyError::module(self.name(), "Subprocess operations are not allowed"));
                        }
                    }
                }
            }
            _ => {
                // TODO: Add security checks for other languages
            }
        }

        Ok(())
    }

    /// Execute code safely
    async fn execute_code(&self, code: &str) -> DspyResult<CodeExecutionResult> {
        if !self.config.execute_code {
            return Ok(CodeExecutionResult {
                success: false,
                output: "Code execution is disabled".to_string(),
                error: Some("Execution disabled in configuration".to_string()),
                execution_time_ms: 0.0,
                memory_usage_mb: None,
            });
        }

        let start_time = std::time::Instant::now();

        // Create temporary file
        let temp_dir = std::env::temp_dir();
        let file_name = format!("pot_{}_{}.{}", 
            self.id, 
            chrono::Utc::now().timestamp_millis(),
            self.config.language.file_extension()
        );
        let file_path = temp_dir.join(file_name);

        // Write code to file
        std::fs::write(&file_path, code)
            .map_err(|e| DspyError::module(&self.name, &format!("Failed to write code to file: {}", e)))?;

        // Execute code
        let mut command = Command::new(self.config.language.execution_command());
        command.arg(&file_path);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.execution_timeout_seconds),
            tokio::task::spawn_blocking(move || command.output())
        ).await;

        // Clean up temporary file
        let _ = std::fs::remove_file(&file_path);

        let execution_time = start_time.elapsed().as_millis() as f64;

        match output {
            Ok(Ok(Ok(output))) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    Ok(CodeExecutionResult {
                        success: true,
                        output: stdout.to_string(),
                        error: if stderr.is_empty() { None } else { Some(stderr.to_string()) },
                        execution_time_ms: execution_time,
                        memory_usage_mb: None, // TODO: Implement memory monitoring
                    })
                } else {
                    Ok(CodeExecutionResult {
                        success: false,
                        output: stdout.to_string(),
                        error: Some(stderr.to_string()),
                        execution_time_ms: execution_time,
                        memory_usage_mb: None,
                    })
                }
            }
            Ok(Ok(Err(e))) => Ok(CodeExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Execution error: {}", e)),
                execution_time_ms: execution_time,
                memory_usage_mb: None,
            }),
            Ok(Err(_)) => Ok(CodeExecutionResult {
                success: false,
                output: String::new(),
                error: Some("Task execution failed".to_string()),
                execution_time_ms: execution_time,
                memory_usage_mb: None,
            }),
            Err(_) => Ok(CodeExecutionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Execution timeout after {} seconds", self.config.execution_timeout_seconds)),
                execution_time_ms: execution_time,
                memory_usage_mb: None,
            }),
        }
    }

    /// Extract answer from execution result
    fn extract_answer_from_execution(&self, execution_result: &CodeExecutionResult) -> DspyResult<O> {
        if !execution_result.success {
            return Err(DspyError::module(
                &self.name,
                &format!("Code execution failed: {}", 
                    execution_result.error.as_deref().unwrap_or("Unknown error")),
            ));
        }

        // Try to parse JSON output
        let output = execution_result.output.trim();
        
        // Look for JSON in the output
        for line in output.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with('{') || trimmed_line.starts_with('[') || trimmed_line.starts_with('"') {
                if let Ok(answer) = serde_json::from_str::<O>(trimmed_line) {
                    return Ok(answer);
                }
            }
        }

        // Fallback: try to parse the entire output
        if let Ok(answer) = serde_json::from_str::<O>(output) {
            return Ok(answer);
        }

        // Last resort: wrap output as string
        if let Ok(answer) = serde_json::from_str::<O>(&format!("\"{}\"", output)) {
            return Ok(answer);
        }

        Err(DspyError::module(self.name(), "Could not extract valid answer from code execution output"))
    }

    /// Get the last generated code
    pub async fn get_last_generated_code(&self) -> Option<String> {
        self.last_generated_code.read().await.clone()
    }

    /// Get the last execution result
    pub async fn get_last_execution_result(&self) -> Option<CodeExecutionResult> {
        self.last_execution_result.read().await.clone()
    }

    /// Get configuration
    pub fn config(&self) -> &ProgramOfThoughtConfig {
        &self.config
    }
}

#[async_trait]
impl<I, O> Module for ProgramOfThought<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    type Input = I;
    type Output = O;

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature<Self::Input, Self::Output> {
        &self.signature
    }

    async fn forward(&self, input: Self::Input) -> DspyResult<Self::Output> {
        let start_time = std::time::Instant::now();
        let mut reasoning_steps = Vec::new();

        // Step 1: Generate code prompt
        let prompt = self.generate_code_prompt(&input)?;
        reasoning_steps.push(ReasoningStep {
            step_number: 1,
            step_type: "prompt_generation".to_string(),
            input: "Generate code prompt".to_string(),
            output: "Code generation prompt created".to_string(),
            confidence: 0.9,
            execution_time_ms: 0.0,
            metadata: HashMap::new(),
        });

        // Step 2: Generate code
        // TODO: Call Anthropic API with the prompt
        let response = format!(
            "```{}\n# Solution code\nimport json\n\n# Process the input\nresult = {{\"answer\": \"computed_result\"}}\n\n# Output result\nprint(json.dumps(result))\n```",
            self.config.language.file_extension()
        );

        let code = self.extract_code(&response)?;
        reasoning_steps.push(ReasoningStep {
            step_number: 2,
            step_type: "code_generation".to_string(),
            input: "Problem description".to_string(),
            output: format!("Generated {} lines of code", code.lines().count()),
            confidence: 0.8,
            execution_time_ms: 0.0,
            metadata: HashMap::new(),
        });

        // Step 3: Validate code
        self.validate_code(&code)?;
        reasoning_steps.push(ReasoningStep {
            step_number: 3,
            step_type: "code_validation".to_string(),
            input: "Generated code".to_string(),
            output: "Code validation passed".to_string(),
            confidence: 0.9,
            execution_time_ms: 0.0,
            metadata: HashMap::new(),
        });

        // Step 4: Execute code
        let execution_result = self.execute_code(&code).await?;
        reasoning_steps.push(ReasoningStep {
            step_number: 4,
            step_type: "code_execution".to_string(),
            input: "Validated code".to_string(),
            output: if execution_result.success {
                "Code executed successfully".to_string()
            } else {
                "Code execution failed".to_string()
            },
            confidence: if execution_result.success { 0.9 } else { 0.3 },
            execution_time_ms: execution_result.execution_time_ms,
            metadata: HashMap::new(),
        });

        // Step 5: Extract answer
        let answer = self.extract_answer_from_execution(&execution_result)?;
        reasoning_steps.push(ReasoningStep {
            step_number: 5,
            step_type: "answer_extraction".to_string(),
            input: "Execution output".to_string(),
            output: "Answer extracted successfully".to_string(),
            confidence: 0.8,
            execution_time_ms: 0.0,
            metadata: HashMap::new(),
        });

        // Calculate overall confidence
        let overall_confidence = reasoning_steps.iter()
            .map(|s| s.confidence)
            .sum::<f64>() / reasoning_steps.len() as f64;

        // Update state
        *self.last_reasoning_steps.write().await = reasoning_steps.clone();
        *self.last_confidence.write().await = overall_confidence;
        *self.last_generated_code.write().await = Some(code);
        *self.last_execution_result.write().await = Some(execution_result);

        // Update metrics
        let execution_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.record_success(reasoning_steps.len(), execution_time, overall_confidence);

        info!(
            "Program of Thought completed successfully with confidence {:.3}",
            overall_confidence
        );

        Ok(answer)
    }

    fn metadata(&self) -> &ModuleMetadata {
        &self.metadata
    }

    fn stats(&self) -> &ModuleStats {
        static DEFAULT_STATS: std::sync::OnceLock<ModuleStats> = std::sync::OnceLock::new();
        DEFAULT_STATS.get_or_init(ModuleStats::default)
    }

    fn supports_compilation(&self) -> bool {
        true
    }
}

#[async_trait]
impl<I, O> ReasoningModule<I, O> for ProgramOfThought<I, O>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
    O: Serialize + for<'de> Deserialize<'de> + Send + Sync + Clone,
{
    fn get_reasoning_steps(&self) -> Vec<ReasoningStep> {
        Vec::new() // Simplified for now
    }

    fn get_confidence(&self) -> f64 {
        0.0 // Simplified for now
    }

    fn get_performance_metrics(&self) -> ReasoningMetrics {
        ReasoningMetrics::default() // Simplified for now
    }

    async fn reset_state(&mut self) -> DspyResult<()> {
        *self.last_reasoning_steps.write().await = Vec::new();
        *self.last_confidence.write().await = 0.0;
        *self.last_generated_code.write().await = None;
        *self.last_execution_result.write().await = None;
        Ok(())
    }

    fn configure_reasoning(&mut self, config: SpecializedModuleConfig) -> DspyResult<()> {
        self.config.base = config;
        Ok(())
    }
}

impl ModuleInfo for ProgramOfThought<(), ()> {
    fn name(&self) -> &str {
        "ProgramOfThought"
    }

    fn description(&self) -> Option<&str> {
        Some("Program of Thought module that generates and executes code to solve computational problems")
    }

    fn module_type(&self) -> &str {
        "computational_reasoning"
    }

    fn reasoning_patterns(&self) -> Vec<String> {
        vec![
            "computational_thinking".to_string(),
            "code_generation".to_string(),
            "algorithmic_reasoning".to_string(),
        ]
    }

    fn supports_capability(&self, capability: &str) -> bool {
        matches!(capability, "computation" | "code_generation" | "algorithmic_thinking" | "mathematical_reasoning")
    }
}
