# Development Guide

This document provides information for developers working on the rust_memvid_agent project.

## Project Structure

```
rust_memvid_agent/
├── src/
│   ├── lib.rs                     # Main library exports
│   ├── main.rs                    # CLI application
│   ├── agent/                     # Core agent implementation
│   │   ├── mod.rs                 # Agent orchestration
│   │   ├── conversation.rs        # Conversation management
│   │   └── tool_orchestrator.rs   # Tool execution coordination
│   ├── anthropic/                 # Anthropic API integration
│   │   ├── mod.rs                 # Module exports
│   │   ├── client.rs              # HTTP client
│   │   ├── models.rs              # API data structures
│   │   └── tools.rs               # Tool definitions
│   ├── config/                    # Configuration management
│   │   ├── mod.rs                 # Module exports
│   │   └── settings.rs            # Configuration structures
│   ├── memory/                    # Memory system
│   │   ├── mod.rs                 # Memory manager
│   │   ├── memvid_wrapper.rs      # MP4 memory wrapper
│   │   └── search.rs              # Search functionality
│   ├── tools/                     # Tool system
│   │   ├── mod.rs                 # Tool registry and traits
│   │   ├── text_editor.rs         # Text editor tool
│   │   ├── memory_tools.rs        # Memory-specific tools
│   │   ├── file_system.rs         # File system tools
│   │   └── custom_tools.rs        # Example custom tools
│   └── utils/                     # Utilities
│       ├── mod.rs                 # Module exports
│       ├── error.rs               # Error handling
│       └── logging.rs             # Logging setup
├── examples/                      # Example applications
├── tests/                         # Integration tests
└── docs/                          # Documentation
```

## Key Components

### Agent System
- **Agent**: Main orchestrator that manages conversations, tools, and memory
- **AgentBuilder**: Builder pattern for creating agents with custom configurations
- **ConversationManager**: Handles conversation persistence and retrieval
- **ToolOrchestrator**: Manages tool registration and execution

### Anthropic Integration
- **AnthropicClient**: HTTP client for the Anthropic API with retry logic
- **Models**: Complete type definitions for API requests and responses
- **Tools**: Support for all latest Anthropic tools (code execution, web search, text editor)

### Memory System
- **MemoryManager**: High-level interface for memory operations
- **MemvidWrapper**: Wrapper around rust-mp4-memory for video-based storage
- **SearchResult**: Search functionality with scoring and metadata

### Tool System
- **Tool trait**: Interface for implementing custom tools
- **ToolRegistry**: Registry for managing and executing tools
- **Built-in tools**: Text editor, file system, memory, and utility tools

## Development Workflow

### Building
```bash
# Debug build
cargo build

# Release build
cargo build --release

# Check without building
cargo check
```

### Testing
```bash
# Run all tests
cargo test

# Run with specific features
cargo test --features all-tools

# Run examples
cargo run --example basic_agent
cargo run --example memory_chat
cargo run --example tool_development
```

### Linting and Formatting
```bash
# Format code
cargo fmt

# Run clippy
cargo clippy

# Fix warnings
cargo fix --lib -p rust_memvid_agent
```

## Adding New Tools

### 1. Implement the Tool Trait

```rust
use async_trait::async_trait;
use crate::tools::{Tool, ToolResult};
use crate::anthropic::models::ToolDefinition;

#[derive(Debug, Clone)]
pub struct MyCustomTool;

#[async_trait]
impl Tool for MyCustomTool {
    fn definition(&self) -> ToolDefinition {
        // Define the tool schema
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult> {
        // Implement tool logic
    }

    fn name(&self) -> &str {
        "my_custom_tool"
    }

    fn description(&self) -> Option<&str> {
        Some("Description of what this tool does")
    }
}
```

### 2. Register the Tool

```rust
// In the agent builder
let agent = AgentBuilder::new()
    .with_tool(MyCustomTool)
    .build()
    .await?;

// Or after agent creation
agent.register_tool(MyCustomTool);
```

## Memory Integration

The memory system uses MP4 files to store conversation history and arbitrary data. The integration with rust-mp4-memory provides:

- **Persistent Storage**: Conversations and data stored in MP4 format
- **Semantic Search**: Find relevant information using natural language
- **Efficient Retrieval**: Fast access to historical context

### Memory Operations

```rust
// Save to memory
agent.save_to_memory("Important information", MemoryEntryType::Fact).await?;

// Search memory
let results = agent.search_memory("rust programming", 5).await?;

// Get memory statistics
let stats = agent.get_memory_stats().await?;
```

## Configuration

The system uses TOML configuration files with environment variable overrides:

```toml
[anthropic]
api_key = "your-key"
model = "claude-opus-4-20250514"

[memory]
memory_path = "agent_memory.mp4"
enable_search = true

[tools]
enable_text_editor = true
enable_memory_tools = true
```

## Error Handling

The project uses a comprehensive error system with specific error types:

- **AgentError**: Main error type with variants for different failure modes
- **Result<T>**: Type alias for `std::result::Result<T, AgentError>`
- **Retryable errors**: Automatic retry for transient failures

## Logging

Structured logging using the `tracing` crate:

```rust
// Initialize logging
rust_memvid_agent::init().await?;

// Or with custom level
rust_memvid_agent::init_with_logging(tracing::Level::DEBUG).await?;
```

## Performance Considerations

- **Async/Await**: All I/O operations are async for high performance
- **Connection Pooling**: HTTP client reuses connections
- **Memory Efficiency**: Streaming and chunked processing where possible
- **Caching**: Tool definitions and configurations are cached

## Security

- **API Key Protection**: Keys are loaded from environment variables
- **Path Traversal Prevention**: File operations are sandboxed
- **Command Filtering**: Shell commands are filtered for safety
- **Input Validation**: All tool inputs are validated

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Use meaningful variable and function names
- Add documentation for public APIs
- Include examples in documentation
- Write comprehensive tests

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

## Debugging

### Common Issues

1. **API Key Not Set**: Ensure `ANTHROPIC_API_KEY` is set
2. **Memory File Permissions**: Check file system permissions
3. **Tool Execution Failures**: Check tool input validation
4. **Network Issues**: Verify internet connectivity and API endpoints

### Debug Logging

Enable debug logging to see detailed operation logs:

```bash
RUST_LOG=debug cargo run -- chat
```

### Memory Debugging

Check memory statistics and file integrity:

```bash
cargo run -- stats
```

## Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build release binary
5. Create git tag
6. Publish to crates.io (if applicable)

## Dependencies

Key dependencies and their purposes:

- **tokio**: Async runtime
- **reqwest**: HTTP client for Anthropic API
- **serde**: Serialization/deserialization
- **tracing**: Structured logging
- **clap**: CLI interface
- **rust_mem_vid**: MP4 memory storage
- **anyhow/thiserror**: Error handling

## Future Enhancements

- Streaming response support
- Plugin system for external tools
- Web interface
- Multi-agent coordination
- Enhanced memory search algorithms
- Performance optimizations
