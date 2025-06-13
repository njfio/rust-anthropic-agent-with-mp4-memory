use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

use crate::utils::error::{AgentError, Result};
use super::search::SearchResult;

/// Advanced search algorithms for memory optimization
#[cfg(test)]
mod tests;

/// Search algorithm implementations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchAlgorithm {
    /// Linear search (O(n))
    Linear,
    /// Binary search (O(log n)) - requires sorted data
    Binary,
    /// Hash-based search (O(1) average)
    Hash,
    /// Trie-based prefix search
    Trie,
    /// Inverted index search
    InvertedIndex,
    /// Vector similarity search
    VectorSimilarity,
    /// Fuzzy string matching
    FuzzyMatch,
    /// N-gram based search
    NGram,
}

/// Document for indexing and searching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Document timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Document vector representation (for vector similarity)
    pub vector: Option<Vec<f64>>,
}

/// Inverted index for fast text search
#[derive(Debug, Clone)]
pub struct InvertedIndex {
    /// Term to document mapping
    index: HashMap<String, HashSet<String>>,
    /// Document storage
    documents: HashMap<String, Document>,
    /// Term frequencies
    term_frequencies: HashMap<String, HashMap<String, usize>>,
    /// Document frequencies
    document_frequencies: HashMap<String, usize>,
    /// Total number of documents
    total_documents: usize,
}

/// Trie data structure for prefix matching
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Child nodes
    children: HashMap<char, TrieNode>,
    /// Whether this node represents end of a word
    is_end_of_word: bool,
    /// Document IDs containing this prefix
    document_ids: HashSet<String>,
}

/// Trie-based search index
#[derive(Debug, Clone)]
pub struct TrieIndex {
    /// Root node of the trie
    root: TrieNode,
    /// Document storage
    documents: HashMap<String, Document>,
}

/// N-gram index for fuzzy matching
#[derive(Debug, Clone)]
pub struct NGramIndex {
    /// N-gram size
    n: usize,
    /// N-gram to document mapping
    ngrams: HashMap<String, HashSet<String>>,
    /// Document storage
    documents: HashMap<String, Document>,
}

/// Vector similarity search using cosine similarity
#[derive(Debug, Clone)]
pub struct VectorIndex {
    /// Document vectors
    vectors: HashMap<String, Vec<f64>>,
    /// Document storage
    documents: HashMap<String, Document>,
    /// Vector dimension
    dimension: usize,
}

/// Fuzzy string matcher using Levenshtein distance
#[derive(Debug, Clone)]
pub struct FuzzyMatcher {
    /// Maximum edit distance
    max_distance: usize,
    /// Document storage
    documents: HashMap<String, Document>,
}

/// Search performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Algorithm used
    pub algorithm: SearchAlgorithm,
    /// Search time in microseconds
    pub search_time_us: u64,
    /// Number of documents scanned
    pub documents_scanned: usize,
    /// Number of comparisons made
    pub comparisons: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Cache hits
    pub cache_hits: usize,
}

impl InvertedIndex {
    /// Create a new inverted index
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            documents: HashMap::new(),
            term_frequencies: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
        }
    }

    /// Add a document to the index
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        let terms = self.tokenize(&document.content);
        
        // Update term frequencies for this document
        let mut doc_term_freq = HashMap::new();
        for term in &terms {
            *doc_term_freq.entry(term.clone()).or_insert(0) += 1;
        }
        
        // Update inverted index
        for term in terms.iter().collect::<HashSet<_>>() {
            self.index.entry(term.clone()).or_insert_with(HashSet::new).insert(doc_id.clone());
            *self.document_frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        
        self.term_frequencies.insert(doc_id.clone(), doc_term_freq);
        self.documents.insert(doc_id, document);
        self.total_documents += 1;
        
        Ok(())
    }

    /// Search the inverted index
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_terms = self.tokenize(query);
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        
        // Calculate TF-IDF scores for each document
        for term in &query_terms {
            if let Some(doc_ids) = self.index.get(term) {
                let idf = self.calculate_idf(term);
                
                for doc_id in doc_ids {
                    let tf = self.calculate_tf(doc_id, term);
                    let score = tf * idf;
                    *doc_scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }
        
        // Convert to search results and sort by score
        let mut results: Vec<SearchResult> = doc_scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.documents.get(&doc_id).map(|doc| SearchResult {
                    content: doc.content.clone(),
                    score: score as f32,
                    chunk_id: doc_id.parse().unwrap_or(0),
                    metadata: Some(doc.metadata.clone()),
                })
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(limit);
        
        Ok(results)
    }

    /// Tokenize text into terms
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Calculate term frequency
    fn calculate_tf(&self, doc_id: &str, term: &str) -> f64 {
        if let Some(doc_terms) = self.term_frequencies.get(doc_id) {
            if let Some(&count) = doc_terms.get(term) {
                let total_terms: usize = doc_terms.values().sum();
                return count as f64 / total_terms as f64;
            }
        }
        0.0
    }

    /// Calculate inverse document frequency
    fn calculate_idf(&self, term: &str) -> f64 {
        if let Some(&doc_count) = self.document_frequencies.get(term) {
            (self.total_documents as f64 / doc_count as f64).ln()
        } else {
            0.0
        }
    }

    /// Get index statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), self.total_documents);
        stats.insert("unique_terms".to_string(), self.index.len());
        stats.insert("total_postings".to_string(), 
                     self.index.values().map(|set| set.len()).sum());
        stats
    }
}

impl TrieIndex {
    /// Create a new trie index
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            documents: HashMap::new(),
        }
    }

    /// Add a document to the trie
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        let words = self.tokenize(&document.content);
        
        for word in words {
            self.root.insert(&word, &doc_id);
        }
        
        self.documents.insert(doc_id, document);
        Ok(())
    }

    /// Search for documents with prefix matching
    pub fn search_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let doc_ids = self.root.search_prefix(prefix);
        let mut results: Vec<SearchResult> = doc_ids
            .into_iter()
            .filter_map(|doc_id| {
                self.documents.get(&doc_id).map(|doc| SearchResult {
                    content: doc.content.clone(),
                    score: 1.0, // Simple scoring for prefix match
                    chunk_id: doc_id.parse().unwrap_or(0),
                    metadata: Some(doc.metadata.clone()),
                })
            })
            .collect();
        
        results.truncate(limit);
        Ok(results)
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}

impl TrieNode {
    /// Create a new trie node
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end_of_word: false,
            document_ids: HashSet::new(),
        }
    }

    /// Insert a word into the trie
    fn insert(&mut self, word: &str, doc_id: &str) {
        let mut current = self;
        
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
            current.document_ids.insert(doc_id.to_string());
        }
        
        current.is_end_of_word = true;
    }

    /// Search for documents with a given prefix
    fn search_prefix(&self, prefix: &str) -> HashSet<String> {
        let mut current = self;
        
        // Navigate to the prefix node
        for ch in prefix.chars() {
            if let Some(node) = current.children.get(&ch) {
                current = node;
            } else {
                return HashSet::new(); // Prefix not found
            }
        }
        
        // Return all document IDs from this subtree
        current.document_ids.clone()
    }
}

impl NGramIndex {
    /// Create a new N-gram index
    pub fn new(n: usize) -> Self {
        Self {
            n,
            ngrams: HashMap::new(),
            documents: HashMap::new(),
        }
    }

    /// Add a document to the N-gram index
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        let ngrams = self.generate_ngrams(&document.content);
        
        for ngram in ngrams {
            self.ngrams.entry(ngram).or_insert_with(HashSet::new).insert(doc_id.clone());
        }
        
        self.documents.insert(doc_id, document);
        Ok(())
    }

    /// Search using N-gram similarity
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_ngrams = self.generate_ngrams(query);
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        
        for ngram in &query_ngrams {
            if let Some(doc_ids) = self.ngrams.get(ngram) {
                for doc_id in doc_ids {
                    *doc_scores.entry(doc_id.clone()).or_insert(0.0) += 1.0;
                }
            }
        }
        
        // Normalize scores by query length
        let query_len = query_ngrams.len() as f64;
        for score in doc_scores.values_mut() {
            *score /= query_len;
        }
        
        let mut results: Vec<SearchResult> = doc_scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                self.documents.get(&doc_id).map(|doc| SearchResult {
                    content: doc.content.clone(),
                    score: score as f32,
                    chunk_id: doc_id.parse().unwrap_or(0),
                    metadata: Some(doc.metadata.clone()),
                })
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(limit);
        
        Ok(results)
    }

    /// Generate N-grams from text
    fn generate_ngrams(&self, text: &str) -> Vec<String> {
        let text = text.to_lowercase();
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = Vec::new();
        
        if chars.len() >= self.n {
            for i in 0..=chars.len() - self.n {
                let ngram: String = chars[i..i + self.n].iter().collect();
                ngrams.push(ngram);
            }
        }
        
        ngrams
    }
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            documents: HashMap::new(),
            dimension,
        }
    }

    /// Add a document with its vector representation
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        
        if let Some(vector) = &document.vector {
            if vector.len() != self.dimension {
                return Err(AgentError::validation(format!(
                    "Vector dimension mismatch: expected {}, got {}", 
                    self.dimension, vector.len()
                )));
            }
            self.vectors.insert(doc_id.clone(), vector.clone());
        }
        
        self.documents.insert(doc_id, document);
        Ok(())
    }

    /// Search using cosine similarity
    pub fn search(&self, query_vector: &[f64], limit: usize) -> Result<Vec<SearchResult>> {
        if query_vector.len() != self.dimension {
            return Err(AgentError::validation(format!(
                "Query vector dimension mismatch: expected {}, got {}", 
                self.dimension, query_vector.len()
            )));
        }
        
        let mut similarities: Vec<(String, f64)> = self.vectors
            .iter()
            .map(|(doc_id, vector)| {
                let similarity = self.cosine_similarity(query_vector, vector);
                (doc_id.clone(), similarity)
            })
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        similarities.truncate(limit);
        
        let results: Vec<SearchResult> = similarities
            .into_iter()
            .filter_map(|(doc_id, similarity)| {
                self.documents.get(&doc_id).map(|doc| SearchResult {
                    content: doc.content.clone(),
                    score: similarity as f32,
                    chunk_id: doc_id.parse().unwrap_or(0),
                    metadata: Some(doc.metadata.clone()),
                })
            })
            .collect();
        
        Ok(results)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

impl FuzzyMatcher {
    /// Create a new fuzzy matcher
    pub fn new(max_distance: usize) -> Self {
        Self {
            max_distance,
            documents: HashMap::new(),
        }
    }

    /// Add a document to the fuzzy matcher
    pub fn add_document(&mut self, document: Document) -> Result<()> {
        let doc_id = document.id.clone();
        self.documents.insert(doc_id, document);
        Ok(())
    }

    /// Search using fuzzy string matching
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut matches: Vec<(String, usize)> = self.documents
            .iter()
            .filter_map(|(doc_id, doc)| {
                let distance = self.levenshtein_distance(query, &doc.content);
                if distance <= self.max_distance {
                    Some((doc_id.clone(), distance))
                } else {
                    None
                }
            })
            .collect();
        
        matches.sort_by_key(|(_, distance)| *distance);
        matches.truncate(limit);
        
        let results: Vec<SearchResult> = matches
            .into_iter()
            .filter_map(|(doc_id, distance)| {
                self.documents.get(&doc_id).map(|doc| {
                    let score = 1.0 - (distance as f32 / self.max_distance as f32);
                    SearchResult {
                        content: doc.content.clone(),
                        score,
                        chunk_id: doc_id.parse().unwrap_or(0),
                        metadata: Some(doc.metadata.clone()),
                    }
                })
            })
            .collect();
        
        Ok(results)
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1,      // deletion
                        matrix[i][j - 1] + 1       // insertion
                    ),
                    matrix[i - 1][j - 1] + cost    // substitution
                );
            }
        }
        
        matrix[len1][len2]
    }
}

/// Search algorithm factory
pub struct SearchAlgorithmFactory;

impl SearchAlgorithmFactory {
    /// Create a search algorithm instance
    pub fn create_algorithm(algorithm: SearchAlgorithm) -> Box<dyn SearchAlgorithmTrait> {
        match algorithm {
            SearchAlgorithm::InvertedIndex => Box::new(InvertedIndex::new()),
            SearchAlgorithm::Trie => Box::new(TrieIndex::new()),
            SearchAlgorithm::NGram => Box::new(NGramIndex::new(3)), // Default 3-gram
            SearchAlgorithm::VectorSimilarity => Box::new(VectorIndex::new(100)), // Default 100-dim
            SearchAlgorithm::FuzzyMatch => Box::new(FuzzyMatcher::new(3)), // Max distance 3
            _ => Box::new(InvertedIndex::new()), // Default fallback
        }
    }
}

/// Common trait for all search algorithms
pub trait SearchAlgorithmTrait: Send + Sync {
    /// Add a document to the algorithm's index
    fn add_document(&mut self, document: Document) -> Result<()>;
    
    /// Search for documents
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;
    
    /// Get algorithm statistics
    fn get_stats(&self) -> HashMap<String, usize>;
}

// Implement the trait for each algorithm
impl SearchAlgorithmTrait for InvertedIndex {
    fn add_document(&mut self, document: Document) -> Result<()> {
        self.add_document(document)
    }
    
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search(query, limit)
    }
    
    fn get_stats(&self) -> HashMap<String, usize> {
        self.get_stats()
    }
}

impl SearchAlgorithmTrait for TrieIndex {
    fn add_document(&mut self, document: Document) -> Result<()> {
        self.add_document(document)
    }
    
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_prefix(query, limit)
    }
    
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), self.documents.len());
        stats
    }
}

impl SearchAlgorithmTrait for NGramIndex {
    fn add_document(&mut self, document: Document) -> Result<()> {
        self.add_document(document)
    }
    
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search(query, limit)
    }
    
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), self.documents.len());
        stats.insert("total_ngrams".to_string(), self.ngrams.len());
        stats
    }
}

impl SearchAlgorithmTrait for VectorIndex {
    fn add_document(&mut self, document: Document) -> Result<()> {
        self.add_document(document)
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // For text query, create a simple vector representation
        // In a real implementation, this would use proper text-to-vector conversion
        let query_vector: Vec<f64> = query.chars()
            .take(self.dimension)
            .map(|c| c as u8 as f64)
            .chain(std::iter::repeat(0.0))
            .take(self.dimension)
            .collect();

        self.search(&query_vector, limit)
    }

    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), self.documents.len());
        stats.insert("vector_dimension".to_string(), self.dimension);
        stats
    }
}

impl SearchAlgorithmTrait for FuzzyMatcher {
    fn add_document(&mut self, document: Document) -> Result<()> {
        self.add_document(document)
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search(query, limit)
    }

    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), self.documents.len());
        stats
    }
}
