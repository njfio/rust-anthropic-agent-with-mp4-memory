use super::*;
use std::collections::HashMap;

/// Create a test document
fn create_test_document(id: &str, content: &str) -> Document {
    Document {
        id: id.to_string(),
        content: content.to_string(),
        metadata: HashMap::new(),
        timestamp: chrono::Utc::now(),
        vector: None,
    }
}

/// Create a test document with vector
fn create_test_document_with_vector(id: &str, content: &str, vector: Vec<f64>) -> Document {
    Document {
        id: id.to_string(),
        content: content.to_string(),
        metadata: HashMap::new(),
        timestamp: chrono::Utc::now(),
        vector: Some(vector),
    }
}

#[tokio::test]
async fn test_inverted_index_creation() {
    let index = InvertedIndex::new();
    let stats = index.get_stats();

    assert_eq!(stats.get("total_documents"), Some(&0));
    assert_eq!(stats.get("unique_terms"), Some(&0));
}

#[tokio::test]
async fn test_inverted_index_add_document() {
    let mut index = InvertedIndex::new();
    let doc = create_test_document("1", "hello world test");

    let result = index.add_document(doc);
    assert!(result.is_ok());

    let stats = index.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
    assert!(stats.get("unique_terms").unwrap() > &0);
}

#[tokio::test]
async fn test_inverted_index_search() {
    let mut index = InvertedIndex::new();

    // Add test documents
    index
        .add_document(create_test_document("1", "hello world"))
        .unwrap();
    index
        .add_document(create_test_document("2", "world peace"))
        .unwrap();
    index
        .add_document(create_test_document("3", "hello there"))
        .unwrap();

    // Search for "hello"
    let results = index.search("hello", 10).unwrap();
    assert_eq!(results.len(), 2);

    // Search for "world"
    let results = index.search("world", 10).unwrap();
    assert_eq!(results.len(), 2);

    // Search for non-existent term
    let results = index.search("nonexistent", 10).unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_inverted_index_tf_idf_scoring() {
    let mut index = InvertedIndex::new();

    // Add documents with different term frequencies
    index
        .add_document(create_test_document("1", "cat cat dog"))
        .unwrap();
    index
        .add_document(create_test_document("2", "cat bird"))
        .unwrap();
    index
        .add_document(create_test_document("3", "dog bird"))
        .unwrap();

    let results = index.search("cat", 10).unwrap();
    assert_eq!(results.len(), 2);

    // Document 1 should have higher score due to higher term frequency
    assert!(results[0].score >= results[1].score);
}

#[tokio::test]
async fn test_trie_index_creation() {
    let index = TrieIndex::new();
    let stats = index.get_stats();

    assert_eq!(stats.get("total_documents"), Some(&0));
}

#[tokio::test]
async fn test_trie_index_add_document() {
    let mut index = TrieIndex::new();
    let doc = create_test_document("1", "hello world");

    let result = index.add_document(doc);
    assert!(result.is_ok());

    let stats = index.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_trie_index_prefix_search() {
    let mut index = TrieIndex::new();

    // Add test documents
    index
        .add_document(create_test_document("1", "hello world"))
        .unwrap();
    index
        .add_document(create_test_document("2", "help me"))
        .unwrap();
    index
        .add_document(create_test_document("3", "helicopter"))
        .unwrap();

    // Search for prefix "hel"
    let results = index.search_prefix("hel", 10).unwrap();
    assert_eq!(results.len(), 3); // All documents contain words starting with "hel"

    // Search for prefix "hello"
    let results = index.search_prefix("hello", 10).unwrap();
    assert_eq!(results.len(), 1); // Only one document contains "hello"

    // Search for non-existent prefix
    let results = index.search_prefix("xyz", 10).unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_ngram_index_creation() {
    let index = NGramIndex::new(3);
    let stats = index.get_stats();

    assert_eq!(stats.get("total_documents"), Some(&0));
    assert_eq!(stats.get("total_ngrams"), Some(&0));
}

#[tokio::test]
async fn test_ngram_index_add_document() {
    let mut index = NGramIndex::new(3);
    let doc = create_test_document("1", "hello");

    let result = index.add_document(doc);
    assert!(result.is_ok());

    let stats = index.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
    assert!(stats.get("total_ngrams").unwrap() > &0);
}

#[tokio::test]
async fn test_ngram_index_search() {
    let mut index = NGramIndex::new(3);

    // Add test documents
    index
        .add_document(create_test_document("1", "hello"))
        .unwrap();
    index
        .add_document(create_test_document("2", "help"))
        .unwrap();
    index
        .add_document(create_test_document("3", "world"))
        .unwrap();

    // Search for "hel" - should match "hello" and "help"
    let results = index.search("hel", 10).unwrap();
    assert!(results.len() >= 2);

    // Search for exact match
    let results = index.search("hello", 10).unwrap();
    assert!(results.len() >= 1);
}

#[tokio::test]
async fn test_ngram_generate_ngrams() {
    let index = NGramIndex::new(3);
    let ngrams = index.generate_ngrams("hello");

    // "hello" should generate: "hel", "ell", "llo"
    assert_eq!(ngrams.len(), 3);
    assert!(ngrams.contains(&"hel".to_string()));
    assert!(ngrams.contains(&"ell".to_string()));
    assert!(ngrams.contains(&"llo".to_string()));
}

#[tokio::test]
async fn test_vector_index_creation() {
    let index = VectorIndex::new(100);
    assert_eq!(index.dimension, 100);
}

#[tokio::test]
async fn test_vector_index_add_document() {
    let mut index = VectorIndex::new(3);
    let doc = create_test_document_with_vector("1", "test", vec![1.0, 2.0, 3.0]);

    let result = index.add_document(doc);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_vector_index_dimension_mismatch() {
    let mut index = VectorIndex::new(3);
    let doc = create_test_document_with_vector("1", "test", vec![1.0, 2.0]); // Wrong dimension

    let result = index.add_document(doc);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_vector_index_search() {
    let mut index = VectorIndex::new(3);

    // Add test documents with vectors
    index
        .add_document(create_test_document_with_vector(
            "1",
            "doc1",
            vec![1.0, 0.0, 0.0],
        ))
        .unwrap();
    index
        .add_document(create_test_document_with_vector(
            "2",
            "doc2",
            vec![0.0, 1.0, 0.0],
        ))
        .unwrap();
    index
        .add_document(create_test_document_with_vector(
            "3",
            "doc3",
            vec![1.0, 1.0, 0.0],
        ))
        .unwrap();

    // Search with query vector similar to doc1
    let query_vector = vec![0.9, 0.1, 0.0];
    let results = index.search(&query_vector, 10).unwrap();

    assert_eq!(results.len(), 3);
    // First result should be most similar (doc1 or doc3)
    assert!(results[0].score > 0.5);
}

#[tokio::test]
async fn test_vector_index_cosine_similarity() {
    let index = VectorIndex::new(3);

    // Test identical vectors
    let similarity = index.cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
    assert!((similarity - 1.0).abs() < 1e-10);

    // Test orthogonal vectors
    let similarity = index.cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
    assert!((similarity - 0.0).abs() < 1e-10);

    // Test opposite vectors
    let similarity = index.cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
    assert!((similarity - (-1.0)).abs() < 1e-10);
}

#[tokio::test]
async fn test_fuzzy_matcher_creation() {
    let matcher = FuzzyMatcher::new(3);
    assert_eq!(matcher.max_distance, 3);
}

#[tokio::test]
async fn test_fuzzy_matcher_add_document() {
    let mut matcher = FuzzyMatcher::new(3);
    let doc = create_test_document("1", "hello world");

    let result = matcher.add_document(doc);
    assert!(result.is_ok());

    let stats = matcher.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_fuzzy_matcher_search() {
    let mut matcher = FuzzyMatcher::new(2);

    // Add test documents
    matcher
        .add_document(create_test_document("1", "hello"))
        .unwrap();
    matcher
        .add_document(create_test_document("2", "helo"))
        .unwrap(); // 1 edit distance
    matcher
        .add_document(create_test_document("3", "help"))
        .unwrap(); // 2 edit distance
    matcher
        .add_document(create_test_document("4", "world"))
        .unwrap(); // > 2 edit distance

    // Search for "hello"
    let results = matcher.search("hello", 10).unwrap();
    assert_eq!(results.len(), 3); // Should match first 3 documents

    // Results should be sorted by edit distance (lower distance = higher score)
    assert!(results[0].score >= results[1].score);
    assert!(results[1].score >= results[2].score);
}

#[tokio::test]
async fn test_fuzzy_matcher_levenshtein_distance() {
    let matcher = FuzzyMatcher::new(5);

    // Test identical strings
    assert_eq!(matcher.levenshtein_distance("hello", "hello"), 0);

    // Test single character substitution
    assert_eq!(matcher.levenshtein_distance("hello", "hallo"), 1);

    // Test single character insertion
    assert_eq!(matcher.levenshtein_distance("hello", "helloo"), 1);

    // Test single character deletion
    assert_eq!(matcher.levenshtein_distance("hello", "helo"), 1);

    // Test multiple operations
    assert_eq!(matcher.levenshtein_distance("kitten", "sitting"), 3);
}

#[tokio::test]
async fn test_search_algorithm_factory() {
    // Test creating different algorithm types
    let inverted_index = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::InvertedIndex);
    let trie_index = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::Trie);
    let ngram_index = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::NGram);
    let _vector_index = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::VectorSimilarity);
    let fuzzy_matcher = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::FuzzyMatch);

    // Verify they implement the trait
    assert!(inverted_index.get_stats().contains_key("total_documents"));
    assert!(trie_index.get_stats().contains_key("total_documents"));
    assert!(ngram_index.get_stats().contains_key("total_documents"));
    assert!(fuzzy_matcher.get_stats().contains_key("total_documents"));
}

#[tokio::test]
async fn test_search_algorithm_trait_inverted_index() {
    let mut algorithm = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::InvertedIndex);

    // Test adding document
    let doc = create_test_document("1", "hello world");
    let result = algorithm.add_document(doc);
    assert!(result.is_ok());

    // Test search
    let results = algorithm.search("hello", 10);
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 1);

    // Test stats
    let stats = algorithm.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_search_algorithm_trait_trie_index() {
    let mut algorithm = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::Trie);

    // Test adding document
    let doc = create_test_document("1", "hello world");
    let result = algorithm.add_document(doc);
    assert!(result.is_ok());

    // Test search (prefix search)
    let results = algorithm.search("hel", 10);
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 1);

    // Test stats
    let stats = algorithm.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_search_algorithm_trait_ngram_index() {
    let mut algorithm = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::NGram);

    // Test adding document
    let doc = create_test_document("1", "hello world");
    let result = algorithm.add_document(doc);
    assert!(result.is_ok());

    // Test search
    let results = algorithm.search("hello", 10);
    assert!(results.is_ok());
    assert!(results.unwrap().len() >= 1);

    // Test stats
    let stats = algorithm.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_search_algorithm_trait_fuzzy_matcher() {
    let mut algorithm = SearchAlgorithmFactory::create_algorithm(SearchAlgorithm::FuzzyMatch);

    // Test adding document
    let doc = create_test_document("1", "hello");
    let result = algorithm.add_document(doc);
    assert!(result.is_ok());

    // Test search with exact match
    let results = algorithm.search("hello", 10);
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 1);

    // Test search with fuzzy match
    let results = algorithm.search("helo", 10); // 1 edit distance
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 1);

    // Test stats
    let stats = algorithm.get_stats();
    assert_eq!(stats.get("total_documents"), Some(&1));
}

#[tokio::test]
async fn test_document_creation() {
    let doc = create_test_document("test_id", "test content");

    assert_eq!(doc.id, "test_id");
    assert_eq!(doc.content, "test content");
    assert!(doc.metadata.is_empty());
    assert!(doc.vector.is_none());
}

#[tokio::test]
async fn test_document_with_vector_creation() {
    let vector = vec![1.0, 2.0, 3.0];
    let doc = create_test_document_with_vector("test_id", "test content", vector.clone());

    assert_eq!(doc.id, "test_id");
    assert_eq!(doc.content, "test content");
    assert_eq!(doc.vector, Some(vector));
}

#[tokio::test]
async fn test_search_algorithm_enum_equality() {
    assert_eq!(SearchAlgorithm::Linear, SearchAlgorithm::Linear);
    assert_eq!(
        SearchAlgorithm::InvertedIndex,
        SearchAlgorithm::InvertedIndex
    );
    assert_ne!(SearchAlgorithm::Linear, SearchAlgorithm::Binary);
}

#[tokio::test]
async fn test_search_metrics_creation() {
    let metrics = SearchMetrics {
        algorithm: SearchAlgorithm::InvertedIndex,
        search_time_us: 1000,
        documents_scanned: 100,
        comparisons: 500,
        memory_usage_bytes: 1024,
        cache_hits: 5,
    };

    assert_eq!(metrics.algorithm, SearchAlgorithm::InvertedIndex);
    assert_eq!(metrics.search_time_us, 1000);
    assert_eq!(metrics.documents_scanned, 100);
    assert_eq!(metrics.comparisons, 500);
    assert_eq!(metrics.memory_usage_bytes, 1024);
    assert_eq!(metrics.cache_hits, 5);
}
