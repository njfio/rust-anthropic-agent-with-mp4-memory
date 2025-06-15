# Rust MemVid Agent

A Rust-based AI agent system with Anthropic Claude integration and persistent memory using rust-synaptic.

## Recent Updates

### 🧠 **Advanced Memory System Integration** (Latest - December 2024)
- **rust-synaptic Integration**: Integrated rust-synaptic for persistent memory with JSON storage backend
- **Performance Optimization**: Added in-memory LRU caching layer with 1000-entry capacity and 5-minute TTL
- **Batch Operations**: Implemented bulk memory store/search operations for improved throughput
- **Multimodal Support**: Basic file type detection and binary content storage capabilities
- **Security Features**: Basic encryption, access control policies, and audit logging through rust-synaptic
- **Analytics**: Memory usage statistics, cache hit rates, and basic performance metrics
- **Bug Fixes**: Fixed timeout issues in memory stats and server tool result handling

### 🔒 **Security & Reliability** (December 2024)
- **Audit Logging**: Structured JSON logging with configurable severity levels
- **Security Headers**: HTTP security headers (CSP, HSTS, X-Frame-Options) with configurable policies
- **Input Validation**: Path, command, and URL validation with basic injection prevention
- **Rate Limiting**: Basic rate limiting with configurable thresholds
- **Resource Monitoring**: Memory/CPU monitoring with configurable limits
- **Timeout Protection**: Added timeout handling to prevent hanging operations
- **Error Recovery**: Improved error handling and recovery mechanisms

### 🚀 **Core Agent System**
- **HTTP Connection Handling**: Timeout handling (120s), connection pooling, and retry logic with exponential backoff
- **Parameter Validation**: Tool parameter checking with error messages and guidance
- **System Prompt**: Implementation-focused behavior with parameter requirements
- **Iteration Capacity**: Support for up to 50 iterations for complex tasks
- **Request/Response Logging**: API communication logging for debugging
- **Error Handling**: Specific error messages for timeout, connection, and parameter issues
- **File Operations**: Create and str_replace operations with proper parameter handling

### ⚡ **Memory System**
- **JSON Storage**: Fast memory operations using JSON file persistence with rust-synaptic backend
- **Memory Access**: Direct read/write operations without video processing overhead
- **Simple Architecture**: Clean, maintainable codebase focused on core functionality
- **Text Search**: Basic text-based search through memory chunks
- **Persistence**: Changes saved to JSON files for reliability
- **Caching**: In-memory LRU cache for improved performance

## 🌟 Features

### 🔧 **Anthropic Integration**
- **Server Tools**: Support for Anthropic's web search, code execution, and text editor tools
- **Models**: Compatible with Claude Sonnet 4 and other Claude models
- **HTTP Client**: Connection handling with timeouts, connection pooling, and retry logic
- **Error Handling**: Error recovery with timeout and connection error messages
- **Parameter Validation**: Tool parameter checking with error messages and guidance

### 🧠 **Memory System (rust-synaptic)**
- **Persistent Storage**: JSON-based storage with rust-synaptic backend for conversations and memory
- **Knowledge Graphs**: Basic relationship mapping between memory entries (placeholder implementation)
- **Caching Layer**: In-memory LRU cache with 1000-entry capacity and 5-minute TTL
- **Semantic Search**: Basic TF-IDF search (embeddings feature available but not fully implemented)
- **Batch Operations**: Bulk store/search operations for improved throughput
- **Analytics**: Cache hit rates, memory usage statistics, and performance metrics
- **Fast Stats**: Optimized memory statistics with timeout protection

### 🛠️ **Tool System**
- **Built-in Tools**: Text editor, file system, memory, and utility tools
- **Code Analysis**: Code explanations, basic security scanning, and refactoring suggestions via rust-treesitter-agent-code-utility
- **Custom Tools**: Framework for developing custom tools
- **Server Tools**: Integration with Anthropic's server-side tools (web search, code execution)
- **Tool Orchestration**: Tool execution and result handling
- **Parameter Validation**: Tool parameter checking with error messages
- **File Operations**: Create and str_replace operations with parameter handling

### 🧠 **Code Analysis**
- **Code Explanations**: AI-generated code explanations
- **Security Scanning**: Basic vulnerability detection
- **Refactoring Suggestions**: Code improvement recommendations
- **Pattern Recognition**: Basic code pattern analysis

### 🚀 **Advanced Features**

#### 🧠 Memory Analytics
- **Basic Analytics**: Memory usage statistics, cache hit rates, and performance metrics
- **Knowledge Graph APIs**: Basic relationship creation and discovery (placeholder implementation)
- **Memory Evolution**: Version tracking and intelligent updates (basic implementation)
- **Content Analysis**: Simple content analysis and concept extraction

#### 🔍 Code Analysis Features
- **Code Explanations**: Generate explanations of code structure and functionality
- **Security Scanning**: Basic vulnerability detection and security analysis
- **Refactoring Suggestions**: Code improvement recommendations
- **Pattern Detection**: Basic detection of code patterns and structures
- **Dependency Analysis**: Basic dependency scanning and analysis
- **Test Analysis**: Basic test coverage and quality analysis

### 🔒 **Security Features**
- **Audit Logging**: JSON structured logging with configurable severity filtering
- **Security Headers**: HTTP security headers (CSP, HSTS, X-Frame-Options) with basic configurations
- **Input Validation**: Basic validation for paths, commands, URLs with injection prevention
- **Rate Limiting**: Basic rate limiting with configurable thresholds
- **Resource Monitoring**: Memory/CPU monitoring with configurable limits
- **Timeout Protection**: Timeout handling to prevent hanging operations

### 🏗️ **Architecture**
- **Type Safety**: Type system for API interactions
- **Error Handling**: Error types and recovery mechanisms
- **Async/Await**: Async operations throughout
- **Configuration**: TOML-based configuration system
- **HTTP Client**: Connection handling with timeouts, pooling, and retry logic
- **Parameter Validation**: Tool parameter checking and error recovery
- **Iteration Support**: Support for up to 50 iterations for complex tasks

## 🎯 **Capabilities**

### ✅ **Core Functionality**
- **Application Development**: Can create applications and multi-file projects
- **File Operations**: Create and str_replace operations with parameter validation
- **HTTP Reliability**: 120-second timeouts, connection pooling, and retry logic
- **Error Recovery**: Error messages with guidance for resolution
- **Context Management**: Handles conversation context and memory

### 🔧 **System Features**
- **Multi-Step Tasks**: Handle complex tasks with up to 50 iterations
- **Request/Response Logging**: API communication logging for debugging
- **Parameter Validation**: Tool parameter checking with error messages
- **Memory Persistence**: Conversation and memory persistence across sessions
- **Tool Integration**: Integration with Anthropic's server-side tools

### 🧪 **Testing & Validation**
- **Integration Tests**: Comprehensive test suite for memory and tool functionality
- **Performance Benchmarks**: Benchmarks for memory operations and caching
- **Error Handling**: Timeout and connection error handling validation
- **Security Testing**: Basic security validation and input sanitization

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
        .with_model("claude-sonnet-4-20250514");

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

# 🎯 Enhanced Input Modes
# Interactive multi-line mode with full editing capabilities
cargo run -- interactive

# Load complex prompts from files
cargo run -- file prompt.txt

# Pipe input for command-line workflows
echo "Your prompt" | cargo run -- pipe
cat complex_prompt.txt | cargo run -- pipe

# Traditional chat modes
cargo run -- chat                    # Interactive chat
cargo run -- chat "Single message"   # Direct message

# Memory operations
cargo run -- search "rust programming"
cargo run -- save "Important fact about Rust" --entry-type fact
cargo run -- stats

# List available tools
cargo run -- tools

# Code analysis
cargo run -- analyze src --action explain --detailed
cargo run -- analyze src --action security
cargo run -- analyze src --action refactor
```

## 🧠 Code Analysis

The agent includes code analysis capabilities powered by rust-treesitter-agent-code-utility.

### Available Analysis Actions

#### 🔍 **Code Explanations** (`explain`)
Generate explanations of your codebase:

```bash
# Basic explanation
cargo run -- analyze src/main.rs --action explain

# Detailed explanation
cargo run -- analyze src --action explain --detailed
```

#### 🔒 **Security Scanning** (`security`)
Basic security vulnerability detection:

```bash
# Security scan
cargo run -- analyze src --action security
```

#### 🔧 **Refactoring Suggestions** (`refactor`)
Code improvement recommendations:

```bash
# Refactoring suggestions
cargo run -- analyze src --action refactor
```

#### 📊 **Other Analysis Types**
- `analyze` - General code analysis
- `insights` - Architectural insights
- `find_symbols` - Find specific symbols
- `query_patterns` - Query code patterns
- `stats` - Generate code statistics
- `map_structure` - Map code structure

### Interactive Analysis

The agent can also perform code analysis through conversation:

```bash
cargo run -- chat "Can you perform a security scan on my src directory?"
cargo run -- chat "Explain the code structure in src/main.rs"
cargo run -- chat "Give me refactoring suggestions"
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
model = "claude-sonnet-4-20250514"
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
# Security: Dangerous tools disabled by default
enable_code_execution = false
enable_web_search = false
enable_shell_commands = false

# Rate limiting configuration
[tools.rate_limiting]
max_requests_per_minute = 100
per_tool_limiting = true
window_duration_seconds = 60

# Security configuration
[tools.security]
max_file_size = 10485760  # 10MB
max_path_length = 4096
max_command_length = 8192
allowed_domains = []  # Empty = all allowed

# Audit logging
[audit]
log_file_path = "audit.log"
max_file_size = 104857600  # 100MB
max_files = 10
minimum_severity = "low"

# Resource monitoring
[monitoring]
max_memory_bytes = 2147483648  # 2GB
max_memory_percentage = 25.0
max_cpu_percentage = 80.0
max_threads = 100
monitoring_interval_seconds = 30

[agent]
name = "MyAgent"
persist_conversations = true
max_history_length = 50
```

## 🛠️ Available Tools

### Built-in Tools

- **Memory Tools**: Search, save, and manage persistent memory with basic analytics
- **Code Analysis**: Code explanations, basic security scanning, and refactoring suggestions
- **Text Editor**: View and edit files with basic security validation
- **File System**: Read, write, and list files with path validation and size limits
- **HTTP Requests**: Make web requests with basic domain filtering and security headers
- **Shell Commands**: Execute system commands with basic security filtering
- **UUID Generator**: Generate unique identifiers
- **Security Tools**: Basic audit logging, resource monitoring, and input validation

### Anthropic Server Tools

- **Code Execution**: Run Python code in Anthropic's secure sandboxes
- **Web Search**: Search the web for current information
- **Text Editor**: File editing capabilities provided by Anthropic

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

# Run security tests specifically
cargo test security_tests --lib

# Run with features
cargo test --features all-tools

# Run examples
cargo run --example basic_agent
cargo run --example memory_chat
cargo run --example tool_development

# Security testing
cargo audit                    # Dependency vulnerability scan
cargo test --test security    # Security-specific tests
```

## 🔒 Security

This project implements basic security measures including:

- **Audit Logging**: Sensitive operations are logged with JSON format
- **Input Validation**: Basic validation to prevent path traversal and command injection
- **Rate Limiting**: Configurable rate limits to prevent resource exhaustion
- **Security Headers**: HTTP requests include basic security headers
- **Resource Monitoring**: Basic monitoring to prevent resource exhaustion
- **Timeout Protection**: Timeout handling to prevent hanging operations

For security issues, please see [SECURITY.md](SECURITY.md) for our vulnerability reporting process.

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
- [rust-synaptic](https://github.com/njfio/rust-synaptic) for the advanced AI memory system
- [rust-treesitter-agent-code-utility](https://github.com/njfio/rust-treesitter-agent-code-utility) for AI code analysis
- The Rust community for excellent crates and tools

## 🔗 Related Projects

- [rust-synaptic](https://github.com/njfio/rust-synaptic) - Advanced AI memory system with knowledge graphs
- [rust-treesitter-agent-code-utility](https://github.com/njfio/rust-treesitter-agent-code-utility) - AI-powered code analysis
- [Anthropic API](https://docs.anthropic.com/) - Claude AI API documentation

---

**Ready to build intelligent agents with persistent memory? Get started with rust_memvid_agent!** 🚀
