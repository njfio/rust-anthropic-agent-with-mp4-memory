use rust_memvid_agent::{Agent, AgentConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    rust_memvid_agent::init().await?;

    // Create agent configuration
    let config = AgentConfig::default()
        .with_anthropic_key(std::env::var("ANTHROPIC_API_KEY")?)
        .with_memory_path("examples/memory_chat.json")
        .with_model("claude-opus-4-20250514")
        .with_system_prompt(
            "You are a helpful AI assistant with persistent memory. \
             You can save important information to memory and search through it. \
             Use your memory tools to remember important facts about the user and previous conversations."
        );

    // Create the agent
    let mut agent = Agent::new(config).await?;

    // Pre-populate memory with some information
    println!("=== Memory-Enhanced Chat Example ===\n");
    println!("Pre-populating memory with some information...\n");

    agent.save_to_memory(
        "The user's name is Alice and she works as a software engineer at TechCorp.",
        "fact"
    ).await?;

    agent.save_to_memory(
        "Alice is interested in Rust programming and AI/ML technologies.",
        "fact"
    ).await?;

    agent.save_to_memory(
        "Alice's current project involves building a distributed system using microservices.",
        "note"
    ).await?;

    // Start a conversation
    agent.start_conversation(Some("Memory Chat with Alice".to_string())).await?;

    // Simulate a conversation where the agent uses memory
    let messages = vec![
        "Hi! I'm back. Do you remember me?",
        "What do you know about my current work project?",
        "I've been learning about Rust's async programming. Can you save that to memory?",
        "What programming languages am I interested in?",
        "Can you search for all the information you have about me?",
    ];

    for (i, message) in messages.iter().enumerate() {
        println!("User: {}", message);
        let response = agent.chat(*message).await?;
        println!("Agent: {}\n", response);
        
        if i < messages.len() - 1 {
            println!("---\n");
        }
    }

    // Show final memory statistics
    let stats = agent.get_memory_stats().await?;
    println!("=== Final Memory Statistics ===");
    println!("Total chunks: {}", stats.total_chunks);
    println!("Total conversations: {}", stats.total_conversations);
    println!("Total memories: {}", stats.total_memories);
    println!("Memory file size: {:.2} KB", stats.memory_file_size as f64 / 1024.0);

    // Finalize memory
    agent.finalize_memory().await?;
    println!("\nMemory finalized and saved!");

    Ok(())
}
