use async_trait::async_trait;
use rust_memvid_agent::{AgentBuilder, Tool, ToolResult};
use serde_json::json;

/// Custom tool for calculating mathematical expressions
#[derive(Debug, Clone)]
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn definition(&self) -> rust_memvid_agent::anthropic::models::ToolDefinition {
        rust_memvid_agent::anthropic::models::ToolDefinition {
            tool_type: "function".to_string(),
            name: "calculator".to_string(),
            description: Some("Perform mathematical calculations".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                    }
                },
                "required": ["expression"]
            })),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    async fn execute(&self, input: serde_json::Value) -> rust_memvid_agent::Result<ToolResult> {
        let expression = input
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| rust_memvid_agent::AgentError::invalid_input("Missing expression parameter"))?;

        // Simple calculator implementation (in a real tool, you'd use a proper math parser)
        let result = match expression {
            expr if expr.contains('+') => {
                let parts: Vec<&str> = expr.split('+').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    let b: f64 = parts[1].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    a + b
                } else {
                    return Ok(ToolResult::error("Complex expressions not supported in this example"));
                }
            }
            expr if expr.contains('-') => {
                let parts: Vec<&str> = expr.split('-').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    let b: f64 = parts[1].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    a - b
                } else {
                    return Ok(ToolResult::error("Complex expressions not supported in this example"));
                }
            }
            expr if expr.contains('*') => {
                let parts: Vec<&str> = expr.split('*').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    let b: f64 = parts[1].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    a * b
                } else {
                    return Ok(ToolResult::error("Complex expressions not supported in this example"));
                }
            }
            expr if expr.contains('/') => {
                let parts: Vec<&str> = expr.split('/').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    let b: f64 = parts[1].trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid number"))?;
                    if b == 0.0 {
                        return Ok(ToolResult::error("Division by zero"));
                    }
                    a / b
                } else {
                    return Ok(ToolResult::error("Complex expressions not supported in this example"));
                }
            }
            expr => {
                // Try to parse as a single number
                expr.trim().parse().map_err(|_| rust_memvid_agent::AgentError::invalid_input("Invalid expression"))?
            }
        };

        Ok(ToolResult::success(format!("{} = {}", expression, result)))
    }

    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> Option<&str> {
        Some("Perform mathematical calculations")
    }
}

/// Custom tool for generating random numbers
#[derive(Debug, Clone)]
struct RandomNumberTool;

#[async_trait]
impl Tool for RandomNumberTool {
    fn definition(&self) -> rust_memvid_agent::anthropic::models::ToolDefinition {
        rust_memvid_agent::anthropic::models::ToolDefinition {
            tool_type: "function".to_string(),
            name: "random_number".to_string(),
            description: Some("Generate a random number within a specified range".to_string()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Minimum value (inclusive)",
                        "default": 1
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximum value (inclusive)",
                        "default": 100
                    }
                },
                "required": []
            })),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    async fn execute(&self, input: serde_json::Value) -> rust_memvid_agent::Result<ToolResult> {
        let min = input.get("min").and_then(|v| v.as_i64()).unwrap_or(1);
        let max = input.get("max").and_then(|v| v.as_i64()).unwrap_or(100);

        if min > max {
            return Ok(ToolResult::error("Minimum value cannot be greater than maximum value"));
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(min..=max);

        Ok(ToolResult::success(format!("Random number between {} and {}: {}", min, max, random_number)))
    }

    fn name(&self) -> &str {
        "random_number"
    }

    fn description(&self) -> Option<&str> {
        Some("Generate a random number within a specified range")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    rust_memvid_agent::init().await?;

    println!("=== Custom Tool Development Example ===\n");

    // Create agent with custom tools using the builder pattern
    let mut agent = AgentBuilder::new()
        .with_api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .with_memory_path("examples/tool_development.json")
        .with_model("claude-opus-4-20250514")
        .with_tool(CalculatorTool)
        .with_tool(RandomNumberTool)
        .build()
        .await?;

    // Start a conversation
    agent.start_conversation(Some("Custom Tools Demo".to_string())).await?;

    // Test the custom tools
    let messages = vec![
        "What tools do you have available?",
        "Can you calculate 15 + 27?",
        "What's 144 / 12?",
        "Generate a random number between 1 and 10",
        "Generate a random number between 50 and 100",
        "Can you save the results of these calculations to memory?",
    ];

    for (i, message) in messages.iter().enumerate() {
        println!("User: {}", message);
        let response = agent.chat(*message).await?;
        println!("Agent: {}\n", response);
        
        if i < messages.len() - 1 {
            println!("---\n");
        }
    }

    // Show available tools
    let tools = agent.get_available_tools();
    println!("=== Available Tools ===");
    for tool in tools {
        println!("• {}", tool);
    }

    // Finalize memory
    agent.finalize_memory().await?;
    println!("\nExample completed!");

    Ok(())
}
