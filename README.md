# Rust MemVid Agent 🦀🤖

A comprehensive AI agent system in Rust that integrates with Anthropic's Claude API and uses MP4-based memory storage for persistent conversations and context.

## 🌟 Features

### 🔧 **Complete Anthropic Integration**
- **All Latest Tools**: Support for code execution, web search, and text editor tools
- **Latest Models**: Compatible with Claude Opus 4, Sonnet 4, and Sonnet 3.7
- **Tool Versioning**: Automatic tool version selection based on model
- **Streaming Support**: Ready for streaming responses (implementation in progress)

### 💾 **MP4 Memory System**
- **Persistent Memory**: Store conversations and data in MP4 video files
- **Semantic Search**: Find relevant information using natural language queries
- **Memory Tools**: Built-in tools for saving and searching memory
- **Conversation History**: Automatic conversation persistence and retrieval

### 🛠️ **Extensible Tool System**
- **Built-in Tools**: Text editor, file system, memory, and utility tools
- **Custom Tools**: Easy framework for developing custom tools
- **Server Tools**: Integration with Anthropic's server-side tools
- **Tool Orchestration**: Intelligent tool execution and result handling

### 🏗️ **Robust Architecture**
- **Type Safety**: Comprehensive type system for all API interactions
- **Error Handling**: Detailed error types and recovery mechanisms
- **Async/Await**: High-performance async operations throughout
- **Configuration**: Flexible configuration system with TOML support

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
        .with_memory_path("agent_memory.mp4")
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
        .with_memory_path("memory_chat.mp4")
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
memory_path = "agent_memory.mp4"
index_path = "agent_memory.json"
auto_save = true
enable_search = true

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
│ • Tool Calls    │    │ • Client Tools   │    │ • MP4 Storage   │
│ • Memory Mgmt   │    │ • Server Tools   │    │ • Search        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Anthropic     │    │   Tool Registry  │    │   MemVid        │
│   Client        │    │                  │    │   Wrapper       │
│                 │    │ • Registration   │    │                 │
│ • API Calls     │    │ • Execution      │    │ • Video Encode  │
│ • Streaming     │    │ • Validation     │    │ • Search Index  │
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
- The Rust community for excellent crates and tools

## 🔗 Related Projects

- [rust-mp4-memory](https://github.com/njfio/rust-mp4-memory) - MP4-based memory storage
- [Anthropic API](https://docs.anthropic.com/) - Claude AI API documentation

---

**Ready to build intelligent agents with persistent memory? Get started with rust_memvid_agent!** 🚀
