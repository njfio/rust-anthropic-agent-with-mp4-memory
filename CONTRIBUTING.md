# Contributing to Rust Anthropic Agent with JSON/Synaptic Memory

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **Git**: For version control
3. **Anthropic API Key**: For testing (get from [Anthropic Console](https://console.anthropic.com/))

### Development Setup

```bash
# Clone the repository
git clone https://github.com/njfio/rust-anthropic-agent-with-mp4-memory.git
cd rust-anthropic-agent-with-mp4-memory

# Build the project
cargo build

# Run tests
cargo test

# Set up environment for testing
export ANTHROPIC_API_KEY="your-api-key"
```

## 📋 How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/njfio/rust-anthropic-agent-with-mp4-memory.git
   ```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new memory search algorithm"
git commit -m "fix: resolve tool execution timeout issue"
git commit -m "docs: update API documentation"
git commit -m "test: add integration tests for memory system"
```

### 5. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request on GitHub
3. Provide a clear description of your changes
4. Link any related issues

## 🎯 Areas for Contribution

### High Priority
- **Performance optimizations** for memory operations
- **Additional tool implementations** (database, email, etc.)
- **Streaming response support** for real-time interactions
- **Enhanced error recovery** mechanisms
- **Memory compression** algorithms

### Medium Priority
- **Web interface** for agent interaction
- **Plugin system** for external tools
- **Multi-agent coordination** features
- **Advanced search algorithms** for memory
- **Configuration validation** improvements

### Documentation
- **Tutorial content** for beginners
- **Advanced usage examples**
- **API reference improvements**
- **Architecture documentation**
- **Performance benchmarks**

## 🔧 Development Guidelines

### Code Style

- Use `cargo fmt` for formatting
- Run `cargo clippy` and fix warnings
- Follow Rust naming conventions
- Add documentation for public APIs
- Include examples in documentation

### Testing

- Write unit tests for new functions
- Add integration tests for features
- Test error conditions
- Ensure tests are deterministic
- Mock external dependencies when appropriate

### Documentation

- Document all public APIs
- Include usage examples
- Update README for new features
- Add inline comments for complex logic
- Keep documentation up to date

### Performance

- Profile performance-critical code
- Use async/await appropriately
- Minimize memory allocations
- Consider caching strategies
- Benchmark significant changes

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
cargo test

# Run with specific features
cargo test --features all-tools

# Run integration tests
cargo test --test integration

# Run with logging
RUST_LOG=debug cargo test
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_save_and_search() {
        // Test implementation
    }
    
    #[test]
    fn test_tool_definition() {
        // Test implementation
    }
}
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical operations

## 📝 Documentation Standards

### Code Documentation

```rust
/// Brief description of the function
/// 
/// # Arguments
/// 
/// * `param1` - Description of parameter
/// * `param2` - Description of parameter
/// 
/// # Returns
/// 
/// Description of return value
/// 
/// # Errors
/// 
/// Description of possible errors
/// 
/// # Examples
/// 
/// ```rust
/// let result = function_name(param1, param2)?;
/// ```
pub fn function_name(param1: Type1, param2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

### README Updates

When adding new features:
1. Update the features list
2. Add usage examples
3. Update configuration options
4. Add to the tools list if applicable

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Ensure you're using the latest version
3. Try to reproduce with minimal example
4. Check the documentation

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. macOS, Linux, Windows]
- Rust version: [e.g. 1.70.0]
- Project version: [e.g. 0.1.0]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've considered.

**Additional context**
Any other context about the feature request.
```

## 🔄 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create release PR
6. Tag release after merge
7. Publish to crates.io (if applicable)

## 🤝 Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone.

### Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the README and docs/ directory

## 🙏 Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- README.md contributors section
- Release notes for significant contributions

Thank you for contributing to rust-anthropic-agent-with-mp4-memory! 🚀
