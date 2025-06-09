# Rust MemVid Agent 🦀🤖

A comprehensive AI agent system in Rust that integrates with Anthropic's Claude API and uses MP4-based memory storage for persistent conversations and context.

## Recent Updates

### 🚀 **Production-Ready Agent System** (Latest - December 2024)
- **Complete HTTP Connection Optimization**: Enhanced timeout handling (300s), connection pooling, TCP keepalive, and robust retry logic with exponential backoff
- **Advanced Parameter Validation**: Comprehensive tool parameter checking with detailed error messages and clear guidance for create vs str_replace operations
- **Enhanced System Prompt**: Implementation-focused behavior with explicit parameter requirements and format examples
- **Increased Iteration Capacity**: Expanded from 25 to 50 iterations for complex tasks like game development and large codebases
- **Complete Request/Response Logging**: Full visibility into Anthropic API communication for debugging and optimization
- **Robust Error Handling**: Specific error messages for timeout, connection, and parameter issues with actionable guidance
- **Context Size Optimization**: Simplified tool descriptions to reduce token pressure while maintaining functionality
- **File Operation Excellence**: Flawless create and str_replace operations with proper parameter handling

### ⚡ **Fast JSON Memory System**
- **Removed slow memvid dependency**: Replaced complex MP4 video memory system with fast JSON storage
- **Simple JSON Storage**: Lightning-fast memory operations using simple JSON file persistence
- **Instant Memory Access**: No video encoding/decoding overhead - immediate read/write operations
- **Simplified Architecture**: Clean, maintainable codebase without complex video processing dependencies
- **Fast Search**: Simple text-based search through memory chunks without embedding overhead
- **Immediate Persistence**: Changes saved instantly to JSON file for reliability
- **Reduced Dependencies**: Eliminated heavy video processing and embedding model dependencies
- **Performance Focus**: Optimized for speed and simplicity over advanced features

## 🌟 Features

### 🔧 **Complete Anthropic Integration**
- **All Latest Tools**: Support for code execution, web search, and text editor tools
- **Latest Models**: Compatible with Claude Opus 4, Sonnet 4, and Sonnet 3.7
- **Tool Versioning**: Automatic tool version selection based on model
- **Production-Ready HTTP**: Enhanced connection handling with 300s timeouts, connection pooling, and retry logic
- **Robust Error Handling**: Comprehensive error recovery with specific timeout and connection error messages
- **Parameter Validation**: Advanced tool parameter checking with detailed error messages and guidance
- **Streaming Support**: Architecture ready for streaming responses (HTTP-based implementation complete)

### 💾 **Fast JSON Memory System**
- **Lightning-Fast Storage**: Simple JSON-based memory with instant read/write operations
- **Text-Based Search**: Fast search through memory chunks without embedding overhead
- **Memory Tools**: Built-in tools for saving and searching memory
- **Conversation History**: Automatic conversation persistence and retrieval
- **Immediate Persistence**: Changes saved instantly to JSON file for reliability
- **Simplified Architecture**: Clean, maintainable codebase without complex dependencies

### 🛠️ **Extensible Tool System**
- **Built-in Tools**: Text editor, file system, memory, and utility tools
- **AI Code Analysis**: Intelligent code explanations, security scanning, and refactoring suggestions
- **Custom Tools**: Easy framework for developing custom tools
- **Server Tools**: Integration with Anthropic's server-side tools
- **Tool Orchestration**: Intelligent tool execution and result handling
- **Advanced Parameter Validation**: Comprehensive tool parameter checking with detailed error messages
- **Robust File Operations**: Flawless create and str_replace operations with proper parameter handling
- **Enhanced Error Recovery**: Clear guidance for tool usage and parameter requirements

### 🧠 **AI-Powered Code Analysis**
- **Smart Explanations**: AI-generated code explanations with learning mode
- **Security Scanning**: Automated vulnerability detection and compliance checking
- **Refactoring Suggestions**: Intelligent code improvement recommendations
- **Pattern Recognition**: Advanced code pattern analysis and insights

### 🚀 **Phase 2 Advanced Intelligence Features**

#### 🧠 Advanced Memory Analytics
- **Knowledge Graph Generation**: Build conceptual knowledge graphs from memory content with confidence scoring
- **Temporal Analysis**: Track memory evolution, learning patterns, and knowledge growth over time
- **Content Synthesis**: AI-powered content summarization, insights generation, and trend analysis
- **Analytics Dashboard**: Comprehensive memory usage metrics, engagement patterns, and activity analysis
- **Concept Extraction**: Intelligent extraction of key concepts with semantic categorization
- **Relationship Analysis**: Discover hidden connections and relationships between memory entries

#### 🔍 Enhanced Code Analysis
- **Semantic Analysis**: Deep understanding of code concepts, abstractions, and domain patterns
- **Pattern Recognition**: Automatic detection of architecture patterns, design patterns, and anti-patterns
- **Learning Path Generation**: Personalized learning recommendations based on code complexity and skill level
- **Advanced Security Scanning**: OWASP Top 10 vulnerability detection with CWE mapping and compliance assessment
- **Secrets Detection**: Entropy-based detection of API keys, tokens, and credentials with risk assessment
- **Smart Refactoring**: Automated code improvement suggestions with impact analysis and roadmap generation
- **Performance Optimization**: Hotspot identification, algorithmic improvements, and benchmarking suggestions
- **Test Coverage Analysis**: Comprehensive testing gap analysis, quality assessment, and test recommendations
- **Dependency Management**: Security scanning, license compliance, outdated dependency detection, and update recommendations

#### 🤖 AI-Powered Insights
- **Multi-dimensional Analysis**: Combine semantic, temporal, and structural analysis for comprehensive insights
- **Confidence Scoring**: AI confidence levels for all recommendations, insights, and analysis results
- **Automated Roadmaps**: Generate comprehensive improvement roadmaps with phases and effort estimation
- **Cross-reference Intelligence**: Discover hidden connections across different domains and contexts
- **Adaptive Learning**: System learns from usage patterns to improve recommendations and insights over time

### 🏗️ **Robust Architecture**
- **Type Safety**: Comprehensive type system for all API interactions
- **Error Handling**: Detailed error types and recovery mechanisms with specific guidance
- **Async/Await**: High-performance async operations throughout
- **Configuration**: Flexible configuration system with TOML support
- **Production-Ready HTTP**: Enhanced connection handling with timeouts, pooling, and retry logic
- **Advanced Parameter Validation**: Comprehensive tool parameter checking and error recovery
- **Optimized Context Management**: Reduced token pressure with simplified tool descriptions
- **Enhanced Iteration Capacity**: Support for complex tasks with 50-iteration limit

## 🎯 **Production-Ready Capabilities**

### ✅ **Proven Performance**
- **Complex Application Development**: Successfully creates complete applications (e.g., 3D games with 600+ lines of code)
- **Robust File Operations**: Flawless create and str_replace operations with comprehensive parameter validation
- **Enhanced HTTP Reliability**: 300-second timeouts, connection pooling, and exponential backoff retry logic
- **Advanced Error Recovery**: Specific error messages with actionable guidance for quick resolution
- **Optimized Context Management**: Reduced token pressure while maintaining full functionality

### 🔧 **Enterprise-Grade Features**
- **50-Iteration Capacity**: Handle complex, multi-step tasks like game development and large codebase modifications
- **Complete Request/Response Logging**: Full visibility into API communication for debugging and optimization
- **Parameter Validation Excellence**: Comprehensive tool parameter checking with detailed error messages
- **Implementation-Focused Behavior**: Enhanced system prompt ensures actual code creation, not just descriptions
- **Robust Connection Handling**: TCP keepalive, connection pooling, and enhanced timeout management

### 🎮 **Real-World Testing**
- **Successfully implemented**: Complete 3D Frogger game with Three.js graphics, collision detection, and game mechanics
- **File operation excellence**: Created 600+ line JavaScript files with proper parameter handling
- **Complex task completion**: Multi-file projects with HTML, CSS, and JavaScript coordination
- **Error recovery validation**: Comprehensive testing of timeout, connection, and parameter error scenarios

## 🚀 Quick Start

### Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **Anthropic API Key**: Get one from [Anthropic Console](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/njfio/rust-anthropic-agent-with-mp4-memory.git
cd rust-anthropic-agent-with-mp4-memory
cargo build --release
```

### Basic Usage

```rust
use rust_memvid_agent::{Agent, AgentConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the agent system
    rust_memvid_agent::init().await?;

    // Create configuration
    let config = AgentConfig::default()
        .with_anthropic_key("your-api-key")
        .with_memory_path("agent_memory.json")
        .with_model("claude-opus-4-20250514");

    // Create and use the agent
    let mut agent = Agent::new(config).await?;
    
    // Start a conversation
    agent.start_conversation(Some("My First Chat".to_string())).await?;
    
    // Chat with the agent
    let response = agent.chat("Hello! Can you help me write some Rust code?").await?;
    println!("Agent: {}", response);
    
    // The agent can use tools, search memory, and maintain context
    let response = agent.chat("Save a note that I'm learning Rust").await?;
    println!("Agent: {}", response);
    
    Ok(())
}
```

### CLI Usage

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key"

# Start an interactive chat
cargo run -- chat

# Search memory
cargo run -- search "rust programming"

# Save information to memory
cargo run -- save "Important fact about Rust" --entry-type fact

# Show memory statistics
cargo run -- stats

# List available tools
cargo run -- tools

# AI-powered code analysis
cargo run -- analyze src --action explain --detailed --learning
cargo run -- analyze src --action security --compliance
cargo run -- analyze src --action refactor --quick-wins

# Phase 2 Advanced Intelligence Features
cargo run -- analyze src --action advanced_ai --semantic --patterns
cargo run -- analyze src --action owasp_scan --owasp_top10 --secrets
cargo run -- analyze src --action smart_refactor --roadmap --benchmarks
cargo run -- analyze src --action test_coverage --missing_tests
cargo run -- analyze src --action dependency_scan --security_deps
```

## 🧠 AI Code Analysis

The agent includes powerful AI-driven code analysis capabilities powered by the integrated rust-treesitter-agent-code-utility.

### Available Analysis Actions

#### 🔍 **Code Explanations** (`explain`)
Generate intelligent explanations of your codebase with AI insights:

```bash
# Basic explanation
cargo run -- analyze src/main.rs --action explain

# Detailed explanation with learning mode
cargo run -- analyze src --action explain --detailed --learning
```

#### 🔒 **Security Scanning** (`security`)
Automated security vulnerability detection and compliance checking:

```bash
# Basic security scan
cargo run -- analyze src --action security

# Security scan with compliance assessment
cargo run -- analyze src --action security --compliance
```

#### 🔧 **Refactoring Suggestions** (`refactor`)
Smart code improvement recommendations:

```bash
# General refactoring suggestions
cargo run -- analyze src --action refactor

# Quick wins for immediate improvements
cargo run -- analyze src --action refactor --quick-wins
```

#### 📊 **Other Analysis Types**
- `analyze` - Comprehensive code analysis
- `insights` - Generate architectural insights
- `find_symbols` - Find specific symbols
- `query_patterns` - Query code patterns
- `stats` - Generate code statistics
- `map_structure` - Map code structure

#### 🚀 **Phase 2 Advanced Analysis Actions**
- `advanced_ai` / `semantic_analysis` - Deep semantic understanding with concept recognition
- `pattern_recognition` - Architecture and design pattern detection
- `learning_paths` - Personalized learning recommendations
- `advanced_security` / `owasp_scan` - OWASP Top 10 vulnerability detection
- `secrets_detection` - Advanced secrets detection with entropy analysis
- `vulnerability_scan` - Comprehensive vulnerability scanning
- `smart_refactor` / `code_smells` - Intelligent refactoring with roadmaps
- `design_patterns` - Design pattern analysis and recommendations
- `performance_optimize` - Performance optimization suggestions
- `test_coverage` - Test coverage analysis and gap identification
- `missing_tests` - Missing test identification and prioritization
- `test_quality` - Test quality assessment and improvement suggestions
- `coverage_gaps` - Coverage gap analysis and remediation
- `dependency_scan` - Dependency analysis and optimization
- `security_deps` - Security dependency scanning
- `outdated_deps` - Outdated dependency detection
- `license_check` - License compliance checking

### Interactive AI Analysis

The agent can also perform code analysis through natural conversation:

```bash
cargo run -- chat "Can you perform a security scan on my src directory?"
cargo run -- chat "Explain the code structure in src/main.rs with learning insights"
cargo run -- chat "Give me refactoring suggestions for quick wins"

# Advanced Memory Analytics through conversation
cargo run -- chat "Build a knowledge graph from my memory"
cargo run -- chat "Show me temporal analysis of my learning patterns"
cargo run -- chat "Synthesize insights from my recent conversations"
cargo run -- chat "Generate an analytics dashboard for my memory usage"
```

## 📖 Examples

### Memory-Enhanced Chat

```rust
use rust_memvid_agent::{Agent, AgentConfig, MemoryEntryType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rust_memvid_agent::init().await?;

    let config = AgentConfig::default()
        .with_anthropic_key(std::env::var("ANTHROPIC_API_KEY")?)
        .with_memory_path("memory_chat.json")
        .with_system_prompt(
            "You are a helpful AI assistant with persistent memory. \
             Use your memory tools to remember important information."
        );

    let mut agent = Agent::new(config).await?;
    
    // Pre-populate memory
    agent.save_to_memory(
        "User prefers functional programming patterns",
        MemoryEntryType::Fact
    ).await?;

    // Start conversation
    agent.start_conversation(Some("Learning Session".to_string())).await?;
    
    let response = agent.chat("What do you know about my programming preferences?").await?;
    println!("Agent: {}", response);
    
    Ok(())
}
```

### Custom Tool Development

```rust
use async_trait::async_trait;
use rust_memvid_agent::{AgentBuilder, Tool, ToolResult};
use serde_json::json;

#[derive(Debug, Clone)]
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn definition(&self) -> rust_memvid_agent::anthropic::models::ToolDefinition {
        rust_memvid_agent::tools::create_tool_definition(
            "get_weather",
            "Get current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    }
                },
                "required": ["location"]
            })
        )
    }

    async fn execute(&self, input: serde_json::Value) -> rust_memvid_agent::Result<ToolResult> {
        let location = input["location"].as_str().unwrap_or("Unknown");
        // Implement actual weather API call here
        Ok(ToolResult::success(format!("Weather in {}: Sunny, 22°C", location)))
    }

    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> Option<&str> { Some("Get current weather") }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = AgentBuilder::new()
        .with_api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .with_tool(WeatherTool)
        .build()
        .await?;

    let response = agent.chat("What's the weather like in Tokyo?").await?;
    println!("Agent: {}", response);
    
    Ok(())
}
```

## 🔧 Configuration

Create a `agent_config.toml` file:

```toml
[anthropic]
api_key = "your-api-key"
model = "claude-opus-4-20250514"
max_tokens = 4096
temperature = 0.7

[memory]
memory_path = "agent_memory.json"
auto_save = true
enable_search = true
max_chunks = 1000

[tools]
enable_text_editor = true
enable_memory_tools = true
enable_file_tools = true
enable_code_execution = true
enable_web_search = true

[agent]
name = "MyAgent"
persist_conversations = true
max_history_length = 50
```

## 🛠️ Available Tools

### Built-in Tools

- **Memory Tools**: Search, save, and manage persistent memory
- **Advanced Memory Analytics**: Knowledge graphs, temporal analysis, content synthesis, and analytics dashboards
- **Code Analysis**: AI-powered code explanations, security scanning, and refactoring suggestions with Phase 2 advanced intelligence
- **Text Editor**: View and edit files with full Anthropic text editor support
- **File System**: Read, write, and list files and directories
- **HTTP Requests**: Make web requests with domain filtering
- **Shell Commands**: Execute system commands (with safety restrictions)
- **UUID Generator**: Generate unique identifiers

### Anthropic Server Tools

- **Code Execution**: Run Python code in secure sandboxes
- **Web Search**: Search the web for current information
- **Text Editor**: Advanced file editing capabilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Agent       │    │  Tool            │    │   Memory        │
│                 │    │  Orchestrator    │    │   Manager       │
│ • Conversation  │◄──►│                  │◄──►│                 │
│ • Tool Calls    │    │ • Client Tools   │    │ • JSON Storage  │
│ • Memory Mgmt   │    │ • Server Tools   │    │ • Fast Search   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Anthropic     │    │   Tool Registry  │    │   Simple        │
│   Client        │    │                  │    │   Memory        │
│                 │    │ • Registration   │    │                 │
│ • Enhanced HTTP │    │ • Validation     │    │ • JSON Files    │
│ • Retry Logic   │    │ • Error Recovery │    │ • Text Search   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with features
cargo test --features all-tools

# Run examples
cargo run --example basic_agent
cargo run --example memory_chat
cargo run --example tool_development
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com/) for the Claude API
- [rust-mp4-memory](https://github.com/njfio/rust-mp4-memory) for the memory system
- [rust-treesitter-agent-code-utility](https://github.com/njfio/rust-treesitter-agent-code-utility) for AI code analysis
- The Rust community for excellent crates and tools

## 🔗 Related Projects

- [rust-mp4-memory](https://github.com/njfio/rust-mp4-memory) - MP4-based memory storage
- [rust-treesitter-agent-code-utility](https://github.com/njfio/rust-treesitter-agent-code-utility) - AI-powered code analysis
- [Anthropic API](https://docs.anthropic.com/) - Claude AI API documentation

---

**Ready to build intelligent agents with persistent memory? Get started with rust_memvid_agent!** 🚀
