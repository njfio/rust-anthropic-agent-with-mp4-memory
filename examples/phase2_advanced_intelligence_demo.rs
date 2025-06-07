use rust_memvid_agent::{Agent, AgentConfig};
use std::env;
use tracing::info;

/// Comprehensive demonstration of Phase 2 Advanced Intelligence features
/// 
/// This example showcases:
/// - Advanced Memory Analytics (knowledge graphs, temporal analysis, content synthesis)
/// - Enhanced Code Analysis (semantic analysis, pattern recognition, advanced security)
/// - AI-Powered Insights (multi-dimensional analysis, confidence scoring, automated roadmaps)
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Phase 2 Advanced Intelligence Demo");

    // Initialize the agent system
    rust_memvid_agent::init().await?;

    // Get API key from environment
    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable not set"))?;

    // Create configuration with all advanced features enabled
    let config = AgentConfig::default()
        .with_anthropic_key(api_key)
        .with_memory_path("phase2_demo_memory.mp4")
        .with_model("claude-3-5-sonnet-20241022")
        .with_system_prompt(
            "You are an advanced AI assistant with Phase 2 Intelligence capabilities. \
             You have access to advanced memory analytics, enhanced code analysis, and \
             AI-powered insights. Use these tools to provide comprehensive analysis and \
             intelligent recommendations."
        );

    // Create the agent
    let mut agent = Agent::new(config).await?;
    
    info!("✅ Agent initialized with Phase 2 Advanced Intelligence features");

    // Start a conversation for the demo
    agent.start_conversation(Some("Phase 2 Advanced Intelligence Demo".to_string())).await?;

    // === Phase 1: Populate Memory with Diverse Content ===
    info!("📝 Phase 1: Populating memory with diverse content...");
    
    let sample_entries = vec![
        ("Rust is a systems programming language focused on safety and performance", "fact"),
        ("Learning functional programming patterns in Rust", "learning"),
        ("Implemented async/await patterns for better concurrency", "achievement"),
        ("Need to improve error handling in the authentication module", "task"),
        ("Memory management in Rust uses ownership and borrowing", "fact"),
        ("Exploring advanced trait patterns and generics", "learning"),
        ("Successfully optimized database queries for 50% performance improvement", "achievement"),
        ("Code review revealed potential security vulnerabilities", "task"),
        ("Rust's type system prevents many common programming errors", "fact"),
        ("Understanding lifetime parameters and their applications", "learning"),
    ];

    for (content, entry_type) in sample_entries {
        agent.save_to_memory(content, entry_type).await?;
    }

    info!("✅ Memory populated with {} entries", 10);

    // === Phase 2: Advanced Memory Analytics Demo ===
    info!("🧠 Phase 2: Demonstrating Advanced Memory Analytics...");

    // Knowledge Graph Generation
    info!("🔗 Generating knowledge graph from memory...");
    let response = agent.chat(
        "Use the advanced_memory_analytics tool to build a knowledge graph from my memory. \
         Set semantic=true, patterns=true, and confidence_threshold=0.7"
    ).await?;
    println!("\n🔗 Knowledge Graph Analysis:\n{}\n", response);

    // Temporal Analysis
    info!("⏰ Performing temporal analysis...");
    let response = agent.chat(
        "Use the advanced_memory_analytics tool to perform temporal analysis of my memory \
         over the last 30 days. Show learning patterns and knowledge evolution."
    ).await?;
    println!("⏰ Temporal Analysis:\n{}\n", response);

    // Content Synthesis
    info!("🔄 Synthesizing content insights...");
    let response = agent.chat(
        "Use the advanced_memory_analytics tool to synthesize insights from my memory. \
         Focus on learning trends and knowledge connections."
    ).await?;
    println!("🔄 Content Synthesis:\n{}\n", response);

    // Analytics Dashboard
    info!("📊 Generating analytics dashboard...");
    let response = agent.chat(
        "Use the advanced_memory_analytics tool to generate an analytics dashboard \
         with detailed metadata about my memory usage and learning patterns."
    ).await?;
    println!("📊 Analytics Dashboard:\n{}\n", response);

    // === Phase 3: Enhanced Code Analysis Demo ===
    info!("🔍 Phase 3: Demonstrating Enhanced Code Analysis...");

    // Advanced AI Analysis
    info!("🤖 Performing advanced AI analysis on source code...");
    let response = agent.chat(
        "Use the code_analysis tool to perform advanced_ai analysis on the 'src' directory. \
         Enable semantic analysis, pattern recognition, and set confidence_threshold to 0.8"
    ).await?;
    println!("🤖 Advanced AI Analysis:\n{}\n", response);

    // Advanced Security Scanning
    info!("🔒 Performing advanced security scanning...");
    let response = agent.chat(
        "Use the code_analysis tool to perform advanced_security scanning on the 'src' directory. \
         Enable OWASP Top 10 detection, secrets detection, and compliance checking."
    ).await?;
    println!("🔒 Advanced Security Scan:\n{}\n", response);

    // Smart Refactoring Analysis
    info!("🔧 Performing smart refactoring analysis...");
    let response = agent.chat(
        "Use the code_analysis tool to perform smart_refactor analysis on the 'src' directory. \
         Generate a comprehensive roadmap with quick wins and benchmarking suggestions."
    ).await?;
    println!("🔧 Smart Refactoring Analysis:\n{}\n", response);

    // Test Coverage Analysis
    info!("🧪 Analyzing test coverage...");
    let response = agent.chat(
        "Use the code_analysis tool to perform test_coverage analysis on the 'src' directory. \
         Identify missing tests and provide quality recommendations."
    ).await?;
    println!("🧪 Test Coverage Analysis:\n{}\n", response);

    // === Phase 4: AI-Powered Insights Demo ===
    info!("💡 Phase 4: Demonstrating AI-Powered Insights...");

    // Multi-dimensional Analysis
    info!("🌐 Performing multi-dimensional analysis...");
    let response = agent.chat(
        "Combine insights from my memory analytics and code analysis to provide \
         a comprehensive multi-dimensional analysis of my learning journey and \
         development patterns. Include confidence scores and actionable recommendations."
    ).await?;
    println!("🌐 Multi-dimensional Analysis:\n{}\n", response);

    // Learning Path Generation
    info!("🎯 Generating personalized learning paths...");
    let response = agent.chat(
        "Use the code_analysis tool to generate learning_paths based on my current \
         codebase complexity and the learning patterns from my memory. Provide \
         personalized recommendations with estimated time and resources."
    ).await?;
    println!("🎯 Learning Path Generation:\n{}\n", response);

    // === Phase 5: Integration and Summary ===
    info!("📋 Phase 5: Integration and Summary...");

    let response = agent.chat(
        "Based on all the Phase 2 Advanced Intelligence analysis we've performed, \
         provide a comprehensive summary that includes: \
         1. Key insights from memory analytics \
         2. Critical findings from code analysis \
         3. Prioritized recommendations for improvement \
         4. A roadmap for continued learning and development \
         5. Confidence scores for all recommendations \
         \
         Format this as an executive summary with actionable next steps."
    ).await?;
    println!("📋 Executive Summary:\n{}\n", response);

    // Display memory statistics
    let stats = agent.get_memory_stats().await?;
    info!("📊 Final Memory Statistics:");
    info!("  Total Chunks: {}", stats.total_chunks);
    info!("  Total Conversations: {}", stats.total_conversations);
    info!("  Memory File Size: {} bytes", stats.memory_file_size);
    info!("  Index File Size: {} bytes", stats.index_file_size);

    // Finalize memory
    agent.finalize_memory().await?;
    
    info!("🎉 Phase 2 Advanced Intelligence Demo completed successfully!");
    info!("💾 Memory has been saved to: phase2_demo_memory.mp4");
    
    println!("\n🚀 Phase 2 Advanced Intelligence Demo Complete!");
    println!("✨ The agent demonstrated:");
    println!("   • Knowledge graph generation and analysis");
    println!("   • Temporal analysis of learning patterns");
    println!("   • AI-powered content synthesis");
    println!("   • Advanced security scanning with OWASP compliance");
    println!("   • Smart refactoring with automated roadmaps");
    println!("   • Comprehensive test coverage analysis");
    println!("   • Multi-dimensional insights with confidence scoring");
    println!("   • Personalized learning path generation");
    println!("\n💡 All analysis results have been saved to memory for future reference!");

    Ok(())
}
