use rust_memvid_agent::{Agent, AgentConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    rust_memvid_agent::init().await?;

    // Create agent configuration
    let config = AgentConfig::default()
        .with_anthropic_key(std::env::var("ANTHROPIC_API_KEY")?)
        .with_memory_path("examples/basic_memory.json")
        .with_model("claude-opus-4-20250514");

    // Create the agent
    let mut agent = Agent::new(config).await?;

    // Start a conversation
    agent.start_conversation(Some("Basic Example".to_string())).await?;

    // Send some messages
    println!("=== Basic Agent Example ===\n");

    let response1 = agent.chat("Hello! Can you help me understand what tools you have available?").await?;
    println!("Agent: {}\n", response1);

    let response2 = agent.chat("Can you save a note to memory that says 'This is my first memory entry'?").await?;
    println!("Agent: {}\n", response2);

    let response3 = agent.chat("Now search for that note in memory").await?;
    println!("Agent: {}\n", response3);

    // Show memory statistics
    let stats = agent.get_memory_stats().await?;
    println!("Memory Stats:");
    println!("  Total chunks: {}", stats.total_chunks);
    println!("  Total conversations: {}", stats.total_conversations);
    println!("  Total memories: {}", stats.total_memories);

    // Finalize memory
    agent.finalize_memory().await?;
    println!("\nMemory finalized!");

    Ok(())
}
