#!/bin/bash
set -e

echo "=== Setting up Rust MemVid Agent Development Environment ==="

# Update system packages
sudo apt-get update

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $HOME/.profile
    source $HOME/.cargo/env
else
    echo "Rust is already installed"
fi

# Ensure we have the latest stable Rust
rustup update stable
rustup default stable

# Install required system dependencies for the project
echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl

# Install additional dependencies that might be needed for the synaptic memory system
sudo apt-get install -y \
    cmake \
    libclang-dev \
    llvm-dev

# Navigate to project directory
cd /mnt/persist/workspace

# Clean any previous builds
cargo clean

# Fix the compilation error by adding the missing has_empty_content method (if not already added)
echo "Fixing compilation errors..."

# Check if the method is already added
if ! grep -q "has_empty_content" src/anthropic/models.rs; then
    # Add the method to the ChatMessage impl block
    sed -i '/pub fn get_tool_uses(&self) -> Vec<&ContentBlock> {/i\
\
    /// Check if the message has empty content\
    pub fn has_empty_content(&self) -> bool {\
        self.content.is_empty() || \
        self.content.iter().all(|block| match block {\
            ContentBlock::Text { text } => text.trim().is_empty(),\
            _ => false,\
        })\
    }\
' src/anthropic/models.rs
fi

# Remove the problematic example that references missing file
sed -i '/\[\[example\]\]/,/path = "test_synaptic_integration.rs"/d' Cargo.toml

# Fix the memory_chat example by removing MemoryEntryType references
if [ -f "examples/memory_chat.rs" ]; then
    sed -i 's/MemoryEntryType::Fact/"fact"/g' examples/memory_chat.rs
    sed -i 's/MemoryEntryType::Note/"note"/g' examples/memory_chat.rs
fi

# Remove unused import from main.rs
sed -i 's/, BufRead//g' src/main.rs

# Update Cargo index
cargo update

# Build the project to ensure all dependencies are downloaded and compiled
echo "Building project..."
cargo build

# Build with all features to ensure everything compiles
echo "Building with all features..."
cargo build --features all-tools

echo "=== Environment setup complete ==="