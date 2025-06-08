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

    /// Maximum tool iterations before stopping
    #[arg(long)]
    max_tool_iterations: Option<usize>,

    /// Enable human-in-the-loop for complex tasks
    #[arg(long)]
    enable_human_in_loop: bool,

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

    /// Analyze code files and directories
    Analyze {
        /// Path to file or directory to analyze
        path: String,

        /// Type of analysis to perform
        #[arg(short, long, default_value = "analyze")]
        action: String,

        /// Programming language (auto-detect if not specified)
        #[arg(short, long)]
        language: Option<String>,

        /// Output format
        #[arg(short = 'f', long, default_value = "json")]
        format: String,

        /// Maximum directory depth
        #[arg(short = 'd', long, default_value = "10")]
        max_depth: u32,

        /// Include hidden files
        #[arg(long)]
        include_hidden: bool,

        /// Symbol name pattern for symbol search (supports wildcards)
        #[arg(long)]
        symbol_name: Option<String>,

        /// Type of symbol to search for
        #[arg(long)]
        symbol_type: Option<String>,

        /// Tree-sitter query pattern for pattern matching
        #[arg(long)]
        pattern: Option<String>,

        /// Enable detailed analysis for explain/security/refactor actions
        #[arg(long)]
        detailed: bool,

        /// Enable learning mode for explanations
        #[arg(long)]
        learning: bool,

        /// Enable compliance assessment for security scanning
        #[arg(long)]
        compliance: bool,

        /// Focus on quick wins for refactoring suggestions
        #[arg(long)]
        quick_wins: bool,
    },

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
    if let Some(max_iterations) = cli.max_tool_iterations {
        config.agent.max_tool_iterations = max_iterations;
    }
    if cli.enable_human_in_loop {
        config.agent.enable_human_in_loop = true;
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
        Commands::Analyze { path, action, language, format, max_depth, include_hidden, symbol_name, symbol_type, pattern, detailed, learning, compliance, quick_wins } => {
            run_analyze(config, path, action, language, format, max_depth, include_hidden, symbol_name, symbol_type, pattern, detailed, learning, compliance, quick_wins).await?;
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
    
    // entry_type is now just a string for the simple memory system

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

async fn run_analyze(
    config: AgentConfig,
    path: String,
    action: String,
    language: Option<String>,
    format: String,
    max_depth: u32,
    include_hidden: bool,
    symbol_name: Option<String>,
    symbol_type: Option<String>,
    pattern: Option<String>,
    detailed: bool,
    learning: bool,
    compliance: bool,
    quick_wins: bool
) -> anyhow::Result<()> {
    let agent = Agent::new(config).await?;

    // Prepare the tool input
    let mut tool_input = serde_json::json!({
        "action": action,
        "path": path,
        "max_depth": max_depth,
        "include_hidden": include_hidden,
        "output_format": format
    });

    if let Some(lang) = language {
        tool_input["language"] = serde_json::Value::String(lang);
    }

    if let Some(name) = symbol_name {
        tool_input["symbol_name"] = serde_json::Value::String(name);
    }

    if let Some(sym_type) = symbol_type {
        tool_input["symbol_type"] = serde_json::Value::String(sym_type);
    }

    if let Some(pat) = pattern {
        tool_input["pattern"] = serde_json::Value::String(pat);
    }

    // Add AI feature flags
    if detailed {
        tool_input["detailed"] = serde_json::Value::Bool(detailed);
    }

    if learning {
        tool_input["learning"] = serde_json::Value::Bool(learning);
    }

    if compliance {
        tool_input["compliance"] = serde_json::Value::Bool(compliance);
    }

    if quick_wins {
        tool_input["quick_wins"] = serde_json::Value::Bool(quick_wins);
    }

    // Execute the code analysis tool
    match agent.execute_tool("code_analysis", tool_input).await {
        Ok(result) => {
            if result.is_error {
                eprintln!("Error: {}", result.content);
            } else {
                println!("{}", result.content);
            }
        }
        Err(e) => {
            eprintln!("Failed to execute code analysis: {}", e);
        }
    }

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
