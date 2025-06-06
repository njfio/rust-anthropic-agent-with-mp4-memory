# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-06-06

### Added
- Initial release of rust-anthropic-agent-with-mp4-memory
- Complete Anthropic Claude API integration with all latest tools
- MP4-based persistent memory system using rust-mp4-memory
- Hybrid memory approach (RAM cache + MP4 persistence)
- Extensible tool system with built-in and custom tool support
- CLI interface for agent interaction
- Memory management tools (save, search, stats)
- File system tools (read, write, list)
- HTTP request tool with domain filtering
- Shell command tool with safety restrictions
- UUID generator tool
- Conversation management and persistence
- Configuration system with TOML support
- Comprehensive error handling and logging
- Type-safe API interactions
- Async/await throughout for high performance
- Retry logic for API calls
- Tool orchestration and execution
- Server-side tool integration (text editor, code execution, web search)
- Example applications and documentation
- Development guide and contribution guidelines

### Features
- **Anthropic Integration**: Support for Claude Opus 4, Sonnet 4, and Sonnet 3.7
- **Memory System**: Persistent storage in MP4 format with semantic search
- **Tool Framework**: Easy-to-use trait for custom tool development
- **CLI Interface**: Interactive chat, memory search, and statistics
- **Configuration**: Flexible TOML-based configuration with environment overrides
- **Error Handling**: Comprehensive error types with recovery mechanisms
- **Performance**: Async operations with connection pooling and caching

### Technical Details
- Built with Rust 1.70+
- Uses tokio for async runtime
- Integrates with rust-mp4-memory for video-based storage
- Supports all Anthropic tool types and versions
- Implements proper tool versioning based on model capabilities
- Provides type-safe API interactions
- Includes comprehensive test suite and examples

### Documentation
- Complete README with usage examples
- Development guide for contributors
- API documentation with examples
- Configuration reference
- Tool development guide
- Architecture overview

[Unreleased]: https://github.com/njfio/rust-anthropic-agent-with-mp4-memory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/njfio/rust-anthropic-agent-with-mp4-memory/releases/tag/v0.1.0
