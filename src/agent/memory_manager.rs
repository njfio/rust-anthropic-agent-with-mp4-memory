// This module is currently empty as the memory management functionality
// is implemented directly in the memory module. This file exists to maintain
// the module structure and can be used for agent-specific memory management
// logic in the future.

// Re-export memory types for convenience
pub use crate::memory::{MemoryManager, MemoryEntry, Conversation, SearchResult};
