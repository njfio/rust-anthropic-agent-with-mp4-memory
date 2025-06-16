# Rust MemVid Agent

An AI agent system in Rust with Anthropic Claude integration, DSPy framework, rust-synaptic memory, and audio processing capabilities.

## Features

### Core Agent System

- **Anthropic Claude Integration**: Claude API integration with tool calling support
- **DSPy Framework**: Production-ready DSPy implementation with advanced reasoning modules
- **Rust-Synaptic Memory**: JSON-based persistent memory using rust-synaptic library
- **Async/Await Architecture**: Tokio-based async runtime
- **Tool Framework**: Extensible tool system with built-in tools
- **Configuration Management**: TOML-based configuration with environment variable support
- **Error Handling**: Structured error types and recovery mechanisms

### DSPy Framework

- **Core Modules**: Predict, Chain, Parallel composition with type-safe I/O
- **Optimization**: MIPROv2, BootstrapFinetune, multi-objective optimization
- **Specialized Modules**: Chain of Thought, ReAct, RAG, Program of Thought, Self-Improving
- **Multi-Modal Support**: Vision-language integration with image processing
- **Performance Optimization**: Async batching, caching, connection pooling
- **CLI Integration**: Complete command-line interface for DSPy operations
- **Agent Integration**: Seamless integration with agent tool system

### Audio Processing System

- **Multi-Format Support**: WAV, MP3, FLAC, OGG, AAC, M4A codec support via symphonia
- **Audio I/O**: Cross-platform audio input/output with CPAL
- **Audio Effects**: Noise reduction, normalization, filtering, and fade effects
- **Metadata Extraction**: Audio metadata reading and validation
- **Speech-to-Text**: OpenAI Whisper API integration for transcription
- **Text-to-Speech**: OpenAI TTS API integration for synthesis
- **Audio Analysis**: RMS, peak detection, and dynamic range analysis

### Monitoring System

- **Performance Monitoring**: Metrics collection and export
- **Resource Tracking**: Memory, CPU, and system resource monitoring
- **Alert Management**: Configurable thresholds and alert notifications
- **Metrics Export**: Prometheus, console, and CSV export formats
- **Health Checks**: Component health monitoring and status reporting

### Caching System

- **Multi-Backend Support**: Memory, Redis, and file-based caching
- **Cache Strategies**: Write-through, cache-aside, read-through patterns
- **Invalidation Policies**: TTL, LRU, and custom invalidation strategies
- **Performance Metrics**: Cache hit rates and performance tracking

### Security Features

- **Input Validation**: Path traversal and command injection prevention
- **Rate Limiting**: Per-tool and global rate limiting with configurable windows
- **Audit Logging**: Structured JSON logging with rotation and severity levels
- **Resource Monitoring**: Memory and CPU usage limits with enforcement
- **Security Headers**: HTTP security headers for web requests
- **Encryption**: Ring-based encryption for sensitive data

### Built-in Tools

- **File Operations**: Local file reading, writing, and management with security validation
- **HTTP Client**: HTTP requests with authentication and error handling
- **WebSocket Client**: Real-time WebSocket communication
- **Memory Tools**: Rust-synaptic memory operations and search capabilities
- **Code Analysis**: Code parsing and analysis capabilities
- **UUID Generator**: UUID generation for unique identifiers
- **Audio Processing**: Audio encoding, decoding, and effects processing
- **DSPy Tools**: DSPy modules as tools with performance monitoring and composition

## Test Coverage

The codebase includes comprehensive test coverage with 781+ tests:

- **DSPy Framework**: 200+ tests covering all DSPy modules, optimization, and integration
- **Monitoring System**: 32 integration tests covering monitoring pipeline
- **Audio Processing**: 34 tests covering audio functionality and codecs
- **Security Features**: Input validation, rate limiting, and audit logging tests
- **Tool System**: Built-in tools with error handling and validation scenarios
- **Memory Management**: JSON storage, search, and rust-synaptic memory functionality
- **Caching System**: Multi-backend caching with performance metrics
- **Agent Core**: Conversation management and tool coordination tests
- **CLI System**: Comprehensive CLI testing with integration and golden file tests

## Quick Start

### Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **Anthropic API Key**: Get one from [Anthropic Console](https://console.anthropic.com/)
3. **OpenAI API Key**: Required for audio transcription and synthesis features

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
    // Create configuration
    let config = AgentConfig::default()
        .with_anthropic_key("your-api-key")
        .with_memory_path("agent_memory.json")
        .with_model("claude-3-5-sonnet-20241022");

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
# Set your API keys
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# Interactive chat mode
cargo run -- chat

# Direct message
cargo run -- chat "Hello, can you help me with Rust?"

# Memory operations
cargo run -- memory search "rust programming"
cargo run -- memory save "Important fact about Rust"

# Audio processing
cargo run -- audio transcribe input.wav
cargo run -- audio synthesize "Hello world" output.mp3

# DSPy operations
cargo run -- dspy modules list
cargo run -- dspy modules create predict_module
cargo run -- dspy benchmark run my_module
cargo run -- dspy optimize run my_module
cargo run -- dspy pipeline create my_pipeline

# System monitoring
cargo run -- monitor status
cargo run -- monitor metrics

# Configuration management
cargo run -- config validate
cargo run -- config show
```

## Configuration

Create a `config.toml` file:

```toml
[anthropic]
api_key = "your-anthropic-api-key"
model = "claude-3-5-sonnet-20241022"
max_tokens = 4096
timeout = 30

[audio]
openai_api_key = "your-openai-api-key"
max_file_size = 104857600  # 100MB
enable_caching = true
default_quality = "high"

[memory]
file_path = "memory.json"
max_entries = 10000
enable_search = true
backup_enabled = true

[monitoring]
collection_interval = 30
enable_resource_monitoring = true
enable_prometheus = true
prometheus_endpoint = "0.0.0.0:9090"

[caching]
backend = "memory"  # or "redis" or "file"
ttl_seconds = 3600
max_size = 1000

[security]
enable_rate_limiting = true
max_requests_per_minute = 60
enable_audit_logging = true
log_level = "info"

[agent]
name = "MemVidAgent"
persist_conversations = true
max_history_length = 50

[dspy]
auto_optimize = true
max_modules = 100
enable_security_validation = true
enable_audit_logging = true
module_cache_ttl = 3600
```

## Testing

```bash
# Run all tests
cargo test --lib

# Run DSPy framework tests
cargo test dspy --lib

# Run monitoring integration tests
cargo test monitoring::integration_tests --lib

# Run audio processing tests
cargo test audio --lib

# Run security and validation tests
cargo test security --lib

# Run with verbose output
cargo test --lib -- --nocapture

# Run specific test modules
cargo test caching --lib
cargo test tools --lib
cargo test memory --lib
```

## Architecture

System architecture:

- **Agent Core**: Manages conversations and coordinates tool usage
- **DSPy Framework**: Advanced language model programming with optimization
- **Tool System**: Extensible framework for adding new capabilities
- **Rust-Synaptic Memory**: JSON-based persistent storage with search using rust-synaptic
- **Anthropic Client**: HTTP client with retry logic and error handling
- **Audio Processing**: Multi-format audio codec and effects system
- **Monitoring System**: Metrics collection and export
- **Caching Layer**: Multi-backend caching with performance optimization
- **Security Layer**: Input validation, rate limiting, and audit logging
- **CLI System**: Comprehensive command-line interface for all operations

## Dependencies

Key dependencies include:

- **tokio**: Async runtime for concurrent operations
- **reqwest**: HTTP client for Anthropic API communication
- **serde**: JSON serialization and deserialization
- **symphonia**: Multi-format audio codec support
- **cpal**: Cross-platform audio input/output
- **hound**: WAV audio codec implementation
- **rustfft**: Fast Fourier Transform for audio effects
- **rust-synaptic**: Synaptic memory management
- **tracing**: Structured logging and instrumentation
- **chrono**: Date and time handling
- **uuid**: UUID generation for unique identifiers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
