//! # Synaptic Agent Binary - Full Distributed Power
//!
//! This binary provides a command-line interface for the rust_memvid_agent
//! with full synaptic memory system integration and distributed power.

use std::env;
use std::io::{self, Write};
use clap::{Arg, Command};
use tokio;
use tracing::{info, error, warn};
use tracing_subscriber;

use memvid_agent::agent::synaptic_agent::{SynapticAgent, SynapticAgentConfig};
use memvid_agent::memory::synaptic::config::{
    FullPowerSynapticConfig,
    CoreMemoryConfig,
    DistributedConfig,
    EmbeddingsConfig,
    AnalyticsConfig,
    RealtimeConfig,
    EmbeddingModel,
    ConsensusAlgorithm
};
use memvid_agent::memory::synaptic::bridge::{BridgeConfig, MemorySystem};
use memvid_agent::config::AgentConfig;
use memvid_agent::utils::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let matches = Command::new("Synaptic Agent - Full Distributed Power")
        .version("1.0.0")
        .author("Your Name <your.email@example.com>")
        .about("AI Agent with state-of-the-art synaptic memory system")
        .arg(
            Arg::new("command")
                .help("Command to execute")
                .value_parser(["chat", "stats", "migrate", "checkpoint", "search", "semantic-search"])
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("message")
                .help("Message to send (for chat command)")
                .short('m')
                .long("message")
                .value_name("MESSAGE"),
        )
        .arg(
            Arg::new("query")
                .help("Search query (for search commands)")
                .short('q')
                .long("query")
                .value_name("QUERY"),
        )
        .arg(
            Arg::new("limit")
                .help("Limit number of results")
                .short('l')
                .long("limit")
                .value_name("LIMIT")
                .value_parser(clap::value_parser!(usize))
                .default_value("10"),
        )
        .arg(
            Arg::new("enable-migration")
                .help("Enable migration from simple memory")
                .long("enable-migration")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("distributed")
                .help("Enable distributed consensus features")
                .long("distributed")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("embeddings")
                .help("Enable embeddings and semantic search")
                .long("embeddings")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("analytics")
                .help("Enable advanced analytics")
                .long("analytics")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("realtime")
                .help("Enable real-time features")
                .long("realtime")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Get API key from environment
    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| memvid_agent::utils::error::AgentError::config("ANTHROPIC_API_KEY environment variable not set".to_string()))?;

    // Create synaptic agent configuration
    let mut synaptic_config = create_full_power_config(&matches);
    let bridge_config = create_bridge_config(&matches);
    let base_config = create_base_config(api_key);

    let agent_config = SynapticAgentConfig {
        base_config,
        synaptic_config,
        bridge_config,
        enable_migration: matches.get_flag("enable-migration"),
        enable_full_power: true,
    };

    // Create the synaptic agent
    info!("🚀 Initializing Synaptic Agent with FULL DISTRIBUTED POWER...");
    let mut agent = SynapticAgent::new(agent_config).await?;

    // Verify full power is available
    if agent.has_full_power().await {
        info!("✅ Full distributed power features are ACTIVE!");
    } else {
        warn!("⚠️  Some distributed features may not be available");
    }

    // Execute the requested command
    let command = matches.get_one::<String>("command").unwrap();
    match command.as_str() {
        "chat" => {
            if let Some(message) = matches.get_one::<String>("message") {
                // Single message mode
                execute_chat_single(&mut agent, message).await?;
            } else {
                // Interactive mode
                execute_chat_interactive(&mut agent).await?;
            }
        }
        "stats" => {
            execute_stats(&agent).await?;
        }
        "migrate" => {
            execute_migration(&agent).await?;
        }
        "checkpoint" => {
            execute_checkpoint(&agent).await?;
        }
        "search" => {
            if let Some(query) = matches.get_one::<String>("query") {
                let limit = *matches.get_one::<usize>("limit").unwrap();
                execute_search(&agent, query, limit).await?;
            } else {
                error!("Search query is required for search command");
            }
        }
        "semantic-search" => {
            #[cfg(feature = "embeddings")]
            {
                if let Some(query) = matches.get_one::<String>("query") {
                    let limit = Some(*matches.get_one::<usize>("limit").unwrap());
                    execute_semantic_search(&agent, query, limit).await?;
                } else {
                    error!("Search query is required for semantic-search command");
                }
            }
            #[cfg(not(feature = "embeddings"))]
            {
                error!("Semantic search requires embeddings feature to be enabled");
            }
        }
        _ => {
            error!("Unknown command: {}", command);
        }
    }

    Ok(())
}

/// Create full power synaptic configuration
fn create_full_power_config(matches: &clap::ArgMatches) -> FullPowerSynapticConfig {
    let mut config = FullPowerSynapticConfig::default();

    // Enable features based on command line flags
    if matches.get_flag("distributed") {
        config.distributed.enable_consensus = true;
        config.distributed.consensus_algorithm = ConsensusAlgorithm::Raft;
        config.distributed.enable_failover = true;
        config.distributed.enable_load_balancing = true;
    }

    if matches.get_flag("embeddings") {
        config.embeddings.enable_embeddings = true;
        config.embeddings.enable_semantic_search = true;
        config.embeddings.enable_ai_consolidation = true;
        config.embeddings.embedding_model = EmbeddingModel::OpenAI3Small;
    }

    if matches.get_flag("analytics") {
        config.analytics.enable_performance_analytics = true;
        config.analytics.enable_memory_analytics = true;
        config.analytics.enable_distributed_metrics = true;
    }

    if matches.get_flag("realtime") {
        config.realtime.enable_realtime_sync = true;
        config.realtime.enable_memory_streaming = true;
        config.realtime.enable_notifications = true;
    }

    config
}

/// Create bridge configuration
fn create_bridge_config(matches: &clap::ArgMatches) -> BridgeConfig {
    BridgeConfig {
        primary_system: MemorySystem::Synaptic,
        enable_dual_write: false,
        enable_fallback_reads: true,
        enable_migration: matches.get_flag("enable-migration"),
        migration_batch_size: 1000,
        enable_distributed_consensus: matches.get_flag("distributed"),
        enable_realtime_sync: matches.get_flag("realtime"),
        enable_analytics: matches.get_flag("analytics"),
    }
}

/// Create base agent configuration
fn create_base_config(api_key: String) -> AgentConfig {
    let mut config = AgentConfig::default();
    config.anthropic.api_key = api_key;
    config.anthropic.model = "claude-3-5-sonnet-20241022".to_string();
    config.anthropic.max_tokens = 4096;
    config.anthropic.temperature = 0.7;
    config
}

/// Execute single chat message
async fn execute_chat_single(agent: &mut SynapticAgent, message: &str) -> Result<()> {
    info!("💬 Processing message: {}", message);
    let response = agent.chat_with_synaptic_power(message).await?;
    println!("\n🤖 Agent Response:\n{}\n", response);
    Ok(())
}

/// Execute interactive chat mode
async fn execute_chat_interactive(agent: &mut SynapticAgent) -> Result<()> {
    println!("🚀 Synaptic Agent - Interactive Mode (Full Distributed Power)");
    println!("Type 'quit' or 'exit' to end the conversation\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("👋 Goodbye!");
            break;
        }

        match agent.chat_with_synaptic_power(input).await {
            Ok(response) => {
                println!("\n🤖 Agent: {}\n", response);
            }
            Err(e) => {
                error!("Error: {}", e);
            }
        }
    }

    Ok(())
}

/// Execute stats command
async fn execute_stats(agent: &SynapticAgent) -> Result<()> {
    info!("📊 Retrieving memory statistics...");
    let stats = agent.get_memory_stats().await?;
    
    println!("📊 Synaptic Memory Statistics:");
    println!("  Primary System: {:?}", stats.primary_system);
    println!("  Distributed Features: {}", stats.distributed_enabled);
    println!("  Real-time Features: {}", stats.realtime_enabled);
    println!("  Analytics Features: {}", stats.analytics_enabled);
    
    if let Some(synaptic_stats) = &stats.synaptic_stats {
        println!("  Short-term Memories: {}", synaptic_stats.short_term_count);
        println!("  Long-term Memories: {}", synaptic_stats.long_term_count);
        println!("  Total Size: {} bytes", synaptic_stats.total_size);
        println!("  Session ID: {}", synaptic_stats.session_id);
    }
    
    Ok(())
}

/// Execute migration command
async fn execute_migration(agent: &SynapticAgent) -> Result<()> {
    info!("🔄 Starting migration to synaptic memory...");
    let result = agent.migrate_to_synaptic().await?;
    
    println!("🔄 Migration Results:");
    println!("  Migrated: {}", result.migrated_count);
    println!("  Failed: {}", result.failed_count);
    println!("  Total: {}", result.total_count);
    println!("  Duration: {:?}", result.duration);
    
    Ok(())
}

/// Execute checkpoint command
async fn execute_checkpoint(agent: &SynapticAgent) -> Result<()> {
    info!("💾 Creating memory checkpoint...");
    let checkpoint_id = agent.create_checkpoint().await?;
    
    println!("💾 Checkpoint created with ID: {}", checkpoint_id);
    
    Ok(())
}

/// Execute search command
async fn execute_search(agent: &SynapticAgent, query: &str, limit: usize) -> Result<()> {
    info!("🔍 Searching memories for: {}", query);
    let results = agent.search_relevant_memories(query, limit).await?;
    
    println!("🔍 Search Results ({} found):", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [Score: {:.2}] {}", 
            i + 1, 
            result.relevance_score, 
            result.content.chars().take(100).collect::<String>()
        );
    }
    
    Ok(())
}

/// Execute semantic search command
#[cfg(feature = "embeddings")]
async fn execute_semantic_search(agent: &SynapticAgent, query: &str, limit: Option<usize>) -> Result<()> {
    info!("🧠 Performing semantic search for: {}", query);
    let results = agent.semantic_search(query, limit).await?;
    
    println!("🧠 Semantic Search Results ({} found):", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("  {}. [Similarity: {:.2}] {}", 
            i + 1, 
            result.similarity_score, 
            result.content.chars().take(100).collect::<String>()
        );
    }
    
    Ok(())
}
