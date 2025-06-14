# Rust Anthropic Agent

A Rust agent system with Anthropic Claude integration and audio processing capabilities.

## Features

### Core Agent System
- **Anthropic Claude Integration**: Claude Sonnet 4 with tool calling
- **Async/Await Architecture**: Async operations using tokio
- **Tool System**: Tool framework with built-in tools
- **Memory Management**: JSON-based persistent memory with search
- **Configuration**: TOML-based configuration
- **Error Handling**: Error types and recovery mechanisms

### Audio Processing System
- **Multi-Format Support**: WAV, MP3, FLAC, OGG, AAC, M4A encoding/decoding
- **Audio I/O**: Cross-platform audio I/O with CPAL
- **Audio Effects**: Noise reduction, normalization, filtering, fade effects
- **Metadata Extraction**: Audio metadata reading and validation
- **Speech-to-Text**: OpenAI Whisper API integration
- **Text-to-Speech**: OpenAI TTS API integration
- **Audio Analysis**: RMS, peak detection, dynamic range analysis

### Security Features
- **Input Validation**: Path traversal protection, command injection prevention
- **Rate Limiting**: Per-tool and global rate limits
- **Audit Logging**: JSON logging with rotation
- **Resource Monitoring**: Memory and CPU usage monitoring
- **Security Headers**: HTTP security headers

## Built-in Tools

- **Memory Tools**: Save, search, and manage persistent memory entries
- **File System Tools**: Read, write, and list files with security validation
- **Text Editor**: View and edit files
- **HTTP Client**: Web requests with security headers and domain filtering
- **Audio Processing**: Audio encoding, decoding, and effects processing
- **UUID Generator**: Generate unique identifiers
- **Shell Commands**: Execute system commands with security filtering

## Test Coverage

Test coverage includes:

- **Audio Module**: 34 tests covering audio processing functionality
- **Security Tests**: Input validation, rate limiting, and audit logging
- **Tool Tests**: Built-in tools with error handling scenarios
- **Integration Tests**: Agent functionality
- **Memory Tests**: JSON storage and search functionality

## Quick Start

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

    // Agent can use tools, search memory, and maintain context
    let response = agent.chat("Save a note that I'm learning Rust").await?;
    println!("Agent: {}", response);

    Ok(())
}
```

### CLI Usage

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key"

# Interactive chat
cargo run -- chat

# Direct message
cargo run -- chat "Hello, can you help me with Rust?"

# Memory operations
cargo run -- search "rust programming"
cargo run -- save "Important fact about Rust" --entry-type fact

# List available tools
cargo run -- tools
```

## Configuration

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
enable_audio_tools = true

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

## Testing

```bash
# Run all tests
cargo test

# Run audio tests specifically
cargo test audio::

# Run security tests
cargo test security_tests --lib

# Run with all features
cargo test --features all-tools
```

## Architecture

System architecture:

- **Agent Core**: Manages conversations and coordinates tool usage
- **Tool System**: Framework for adding new capabilities
- **Memory Manager**: JSON-based persistent storage with search
- **Anthropic Client**: HTTP client with retry logic and error handling
- **Audio Processing**: Audio codec and effects system
- **Security Layer**: Input validation, rate limiting, and audit logging

## Dependencies

Key dependencies include:

- **tokio**: Async runtime
- **reqwest**: HTTP client for Anthropic API
- **serde**: JSON serialization
- **symphonia**: Audio codec support
- **cpal**: Cross-platform audio I/O
- **hound**: WAV codec
- **rustfft**: FFT for audio effects

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
