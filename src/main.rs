use clap::{Parser, Subcommand};
use rust_memvid_agent::{Agent, AgentConfig};
use std::io::{self, Write};
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "memvid-agent")]
#[command(about = "AI agent with MP4 memory and Anthropic integration")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Anthropic API key (overrides config and environment)
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    api_key: Option<String>,

    /// Model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Memory file path
    #[arg(long)]
    memory_path: Option<String>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Initial message to send
        message: Option<String>,
        
        /// Conversation title
        #[arg(short, long)]
        title: Option<String>,
    },
    
    /// Search through memory
    Search {
        /// Search query
        query: String,
        
        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },
    
    /// Save information to memory
    Save {
        /// Content to save
        content: String,
        
        /// Entry type
        #[arg(short, long, default_value = "note")]
        entry_type: String,
        
        /// Tags (comma-separated)
        #[arg(short, long)]
        tags: Option<String>,
    },
    
    /// Show memory statistics
    Stats,
    
    /// List available tools
    Tools,
    
    /// Initialize a new agent configuration
    Init {
        /// Output configuration file path
        #[arg(short, long, default_value = "agent_config.toml")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    rust_memvid_agent::init_with_logging(log_level).await?;

    // Load configuration
    let mut config = if let Some(config_path) = &cli.config {
        AgentConfig::from_file(config_path)?
    } else {
        AgentConfig::default()
    };

    // Override config with CLI arguments
    if let Some(api_key) = cli.api_key {
        config.anthropic.api_key = api_key;
    }
    if let Some(model) = cli.model {
        config.anthropic.model = model;
    }
    if let Some(memory_path) = cli.memory_path {
        config = config.with_memory_path(memory_path);
    }

    match cli.command {
        Commands::Chat { message, title } => {
            run_chat(config, message, title).await?;
        }
        Commands::Search { query, limit } => {
            run_search(config, query, limit).await?;
        }
        Commands::Save { content, entry_type, tags } => {
            run_save(config, content, entry_type, tags).await?;
        }
        Commands::Stats => {
            run_stats(config).await?;
        }
        Commands::Tools => {
            run_tools(config).await?;
        }
        Commands::Init { output } => {
            run_init(output).await?;
        }
    }

    Ok(())
}

async fn run_chat(config: AgentConfig, initial_message: Option<String>, title: Option<String>) -> anyhow::Result<()> {
    info!("Starting interactive chat session");
    
    let mut agent = Agent::new(config).await?;
    agent.start_conversation(title).await?;

    // Send initial message if provided
    if let Some(message) = initial_message {
        println!("You: {}", message);
        let response = agent.chat(message).await?;
        println!("Agent: {}", response);
        println!();
    }

    // Interactive chat loop
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() || input == "quit" || input == "exit" {
            break;
        }

        match agent.chat(input).await {
            Ok(response) => {
                println!("Agent: {}", response);
                println!();
            }
            Err(e) => {
                error!("Error: {}", e);
            }
        }
    }

    // Finalize memory
    agent.finalize_memory().await?;
    info!("Chat session ended");
    
    Ok(())
}

async fn run_search(config: AgentConfig, query: String, limit: usize) -> anyhow::Result<()> {
    let agent = Agent::new(config).await?;
    let results = agent.search_memory(query, limit).await?;

    if results.is_empty() {
        println!("No results found.");
    } else {
        println!("Found {} results:\n", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("{}. Score: {:.3}", i + 1, result.score);
            println!("   {}\n", result.content);
        }
    }

    Ok(())
}

async fn run_save(config: AgentConfig, content: String, entry_type: String, tags: Option<String>) -> anyhow::Result<()> {
    let mut agent = Agent::new(config).await?;
    
    let entry_type = match entry_type.as_str() {
        "note" => rust_memvid_agent::MemoryEntryType::Note,
        "code" => rust_memvid_agent::MemoryEntryType::Code,
        "document" => rust_memvid_agent::MemoryEntryType::Document,
        "fact" => rust_memvid_agent::MemoryEntryType::Fact,
        other => rust_memvid_agent::MemoryEntryType::Custom(other.to_string()),
    };

    agent.save_to_memory(content, entry_type).await?;
    
    if let Some(tags) = tags {
        println!("Saved to memory with tags: {}", tags);
    } else {
        println!("Saved to memory");
    }

    Ok(())
}

async fn run_stats(config: AgentConfig) -> anyhow::Result<()> {
    let agent = Agent::new(config).await?;
    let stats = agent.get_memory_stats().await?;

    println!("Memory Statistics:");
    println!("  Total chunks: {}", stats.total_chunks);
    println!("  Total conversations: {}", stats.total_conversations);
    println!("  Total memories: {}", stats.total_memories);
    println!("  Memory file size: {:.2} MB", stats.memory_file_size as f64 / 1024.0 / 1024.0);
    println!("  Index file size: {:.2} KB", stats.index_file_size as f64 / 1024.0);

    Ok(())
}

async fn run_tools(config: AgentConfig) -> anyhow::Result<()> {
    let agent = Agent::new(config).await?;
    let tools = agent.get_available_tools();

    println!("Available Tools ({}):", tools.len());
    for tool in tools {
        println!("  • {}", tool);
    }

    Ok(())
}

async fn run_init(output: String) -> anyhow::Result<()> {
    let config = AgentConfig::default();
    config.save_to_file(&output)?;
    
    println!("Created configuration file: {}", output);
    println!("Please edit the file to set your Anthropic API key and other preferences.");
    
    Ok(())
}
