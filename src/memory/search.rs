use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result from a memory search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The content that was found
    pub content: String,
    /// Relevance score (0.0 to 1.0, higher is more relevant)
    pub score: f32,
    /// Unique identifier for the chunk
    pub chunk_id: usize,
    /// Optional metadata about the result
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(content: String, score: f32, chunk_id: usize) -> Self {
        Self {
            content,
            score,
            chunk_id,
            metadata: None,
        }
    }

    /// Add metadata to the search result
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get a metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.as_ref()?.get(key)
    }

    /// Check if the result has a minimum score
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.score >= threshold
    }

    /// Get a preview of the content (first N characters)
    pub fn preview(&self, max_chars: usize) -> String {
        if self.content.len() <= max_chars {
            self.content.clone()
        } else {
            format!("{}...", &self.content[..max_chars])
        }
    }
}

/// Search query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The search text
    pub query: String,
    /// Maximum number of results to return
    pub limit: usize,
    /// Minimum relevance score threshold
    pub threshold: Option<f32>,
    /// Optional filters to apply
    pub filters: Option<HashMap<String, serde_json::Value>>,
    /// Whether to include metadata in results
    pub include_metadata: bool,
}

impl SearchQuery {
    /// Create a new search query
    pub fn new<S: Into<String>>(query: S) -> Self {
        Self {
            query: query.into(),
            limit: 10,
            threshold: None,
            filters: None,
            include_metadata: false,
        }
    }

    /// Set the result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the score threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Add a filter
    pub fn with_filter<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        if self.filters.is_none() {
            self.filters = Some(HashMap::new());
        }
        self.filters.as_mut().unwrap().insert(key.into(), value);
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }
}

/// Search results container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// The search query that was executed
    pub query: String,
    /// The results found
    pub results: Vec<SearchResult>,
    /// Total number of results before limiting
    pub total_found: usize,
    /// Time taken to execute the search (in milliseconds)
    pub search_time_ms: u64,
}

impl SearchResults {
    /// Create new search results
    pub fn new(query: String, results: Vec<SearchResult>) -> Self {
        let total_found = results.len();
        Self {
            query,
            results,
            total_found,
            search_time_ms: 0,
        }
    }

    /// Set the search time
    pub fn with_search_time(mut self, time_ms: u64) -> Self {
        self.search_time_ms = time_ms;
        self
    }

    /// Set the total found count
    pub fn with_total_found(mut self, total: usize) -> Self {
        self.total_found = total;
        self
    }

    /// Check if any results were found
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Get the best result (highest score)
    pub fn best_result(&self) -> Option<&SearchResult> {
        self.results.first()
    }

    /// Filter results by minimum score
    pub fn filter_by_score(mut self, min_score: f32) -> Self {
        self.results.retain(|r| r.score >= min_score);
        self
    }

    /// Get results as a formatted string
    pub fn format_results(&self, max_preview_chars: usize) -> String {
        if self.results.is_empty() {
            return format!("No results found for query: '{}'", self.query);
        }

        let mut formatted = format!(
            "Found {} results for query: '{}'\n\n",
            self.results.len(),
            self.query
        );

        for (i, result) in self.results.iter().enumerate() {
            formatted.push_str(&format!(
                "{}. Score: {:.3}\n{}\n\n",
                i + 1,
                result.score,
                result.preview(max_preview_chars)
            ));
        }

        formatted
    }
}

/// Search configuration and settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_limit: usize,
    /// Default minimum score threshold
    pub default_threshold: f32,
    /// Maximum number of results that can be requested
    pub max_limit: usize,
    /// Whether to enable fuzzy matching
    pub enable_fuzzy_matching: bool,
    /// Whether to enable semantic search (vs. text search)
    pub enable_semantic_search: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            default_threshold: 0.1,
            max_limit: 100,
            enable_fuzzy_matching: true,
            enable_semantic_search: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_creation() {
        let result = SearchResult::new("Test content".to_string(), 0.8, 1);
        assert_eq!(result.content, "Test content");
        assert_eq!(result.score, 0.8);
        assert_eq!(result.chunk_id, 1);
        assert!(result.metadata.is_none());
    }

    #[test]
    fn test_search_result_preview() {
        let result = SearchResult::new("This is a long piece of content".to_string(), 0.8, 1);
        let preview = result.preview(10);
        assert_eq!(preview, "This is a ...");
    }

    #[test]
    fn test_search_query_builder() {
        let query = SearchQuery::new("test query")
            .with_limit(5)
            .with_threshold(0.5)
            .with_metadata();

        assert_eq!(query.query, "test query");
        assert_eq!(query.limit, 5);
        assert_eq!(query.threshold, Some(0.5));
        assert!(query.include_metadata);
    }

    #[test]
    fn test_search_results() {
        let results = vec![
            SearchResult::new("Result 1".to_string(), 0.9, 1),
            SearchResult::new("Result 2".to_string(), 0.7, 2),
        ];

        let search_results = SearchResults::new("test".to_string(), results);
        assert_eq!(search_results.len(), 2);
        assert!(!search_results.is_empty());
        assert_eq!(search_results.best_result().unwrap().score, 0.9);
    }

    #[test]
    fn test_filter_by_score() {
        let results = vec![
            SearchResult::new("High score".to_string(), 0.9, 1),
            SearchResult::new("Low score".to_string(), 0.3, 2),
        ];

        let search_results = SearchResults::new("test".to_string(), results)
            .filter_by_score(0.5);

        assert_eq!(search_results.len(), 1);
        assert_eq!(search_results.results[0].content, "High score");
    }
}
