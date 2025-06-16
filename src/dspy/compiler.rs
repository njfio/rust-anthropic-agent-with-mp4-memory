//! DSPy compilation system for module optimization and caching
//!
//! This module provides the core compilation infrastructure for DSPy modules,
//! including optimization strategies, caching mechanisms, and performance monitoring.

use crate::dspy::error::{DspyError, DspyResult};
use crate::dspy::examples::{Example, ExampleSet};
use crate::dspy::module::Module;
use crate::dspy::teleprompter::{OptimizationResult, Teleprompter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for the compilation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    /// Enable compilation caching
    pub enable_caching: bool,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Compilation timeout in seconds
    pub compilation_timeout_seconds: u64,
    /// Maximum number of optimization rounds
    pub max_optimization_rounds: usize,
    /// Validation threshold for accepting optimizations
    pub validation_threshold: f64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_dir: PathBuf::from("./dspy_cache"),
            max_cache_size_mb: 1024, // 1GB
            cache_ttl_seconds: 86400, // 24 hours
            enable_monitoring: true,
            compilation_timeout_seconds: 3600, // 1 hour
            max_optimization_rounds: 10,
            validation_threshold: 0.8,
        }
    }
}

/// Compilation context containing metadata and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationContext {
    /// Unique compilation ID
    pub compilation_id: String,
    /// Module signature hash
    pub module_hash: String,
    /// Training data hash
    pub data_hash: String,
    /// Compilation timestamp
    pub timestamp: u64,
    /// DSPy version
    pub dspy_version: String,
    /// Optimization strategy used
    pub strategy: String,
    /// Performance metrics
    pub metrics: CompilationMetrics,
}

/// Performance metrics for compilation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompilationMetrics {
    /// Total compilation time
    pub compilation_time_ms: u64,
    /// Number of optimization rounds
    pub optimization_rounds: usize,
    /// Final validation score
    pub validation_score: f64,
    /// Cache hit/miss statistics
    pub cache_hits: usize,
    pub cache_misses: usize,
    /// Memory usage during compilation
    pub peak_memory_mb: usize,
    /// API calls made during optimization
    pub api_calls: usize,
}

/// Cached compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCompilation {
    /// Compilation context
    pub context: CompilationContext,
    /// Optimized module parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Optimization result
    pub result: OptimizationResult,
    /// Cache creation time
    pub created_at: u64,
    /// Last access time
    pub last_accessed: u64,
    /// Access count
    pub access_count: usize,
}

/// DSPy module compiler with caching and optimization
pub struct Compiler {
    /// Compiler configuration
    config: CompilerConfig,
    /// Compilation cache
    cache: Arc<RwLock<HashMap<String, CachedCompilation>>>,
    /// Performance statistics
    stats: Arc<RwLock<CompilerStats>>,
}

/// Compiler statistics
#[derive(Debug, Default, Clone)]
pub struct CompilerStats {
    /// Total compilations performed
    pub total_compilations: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average compilation time
    pub avg_compilation_time_ms: f64,
    /// Total cache size
    pub cache_size_mb: f64,
    /// Successful compilations
    pub successful_compilations: usize,
    /// Failed compilations
    pub failed_compilations: usize,
}

impl Compiler {
    /// Create a new compiler with default configuration
    pub fn new() -> Self {
        Self::with_config(CompilerConfig::default())
    }

    /// Create a new compiler with custom configuration
    pub fn with_config(config: CompilerConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CompilerStats::default())),
        }
    }

    /// Compile a module with the given teleprompter and training data
    pub async fn compile<I, O, M>(
        &self,
        module: &mut M,
        teleprompter: &mut Teleprompter,
        trainset: ExampleSet<I, O>,
    ) -> DspyResult<OptimizationResult>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        let start_time = Instant::now();
        info!("Starting module compilation");

        // Generate compilation context
        let context = self.generate_context(module, &trainset, teleprompter).await?;

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.get_cached_compilation(&context.module_hash).await? {
                info!("Using cached compilation result");
                self.update_cache_access(&context.module_hash).await?;
                self.update_stats(true, start_time.elapsed()).await;
                return Ok(cached_result.result);
            }
        }

        // Perform compilation
        let result = self.perform_compilation(module, teleprompter, trainset, &context).await?;

        // Cache the result
        if self.config.enable_caching {
            self.cache_compilation(&context, &result).await?;
        }

        // Update statistics
        self.update_stats(false, start_time.elapsed()).await;

        info!(
            "Module compilation completed in {}ms",
            start_time.elapsed().as_millis()
        );

        Ok(result)
    }

    /// Generate compilation context
    async fn generate_context<I, O, M>(
        &self,
        module: &M,
        trainset: &ExampleSet<I, O>,
        _teleprompter: &mut Teleprompter,
    ) -> DspyResult<CompilationContext>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        let compilation_id = uuid::Uuid::new_v4().to_string();
        let module_hash = self.calculate_module_hash(module)?;
        let data_hash = self.calculate_data_hash(trainset)?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(CompilationContext {
            compilation_id,
            module_hash,
            data_hash,
            timestamp,
            dspy_version: env!("CARGO_PKG_VERSION").to_string(),
            strategy: "teleprompter".to_string(),
            metrics: CompilationMetrics::default(),
        })
    }

    /// Calculate hash for module signature
    fn calculate_module_hash<I, O, M>(&self, module: &M) -> DspyResult<String>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash module metadata
        module.metadata().description.hash(&mut hasher);
        module.metadata().version.hash(&mut hasher);
        module.metadata().tags.hash(&mut hasher);
        
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Calculate hash for training data
    fn calculate_data_hash<I, O>(&self, trainset: &ExampleSet<I, O>) -> DspyResult<String>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
    {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash training set characteristics
        trainset.len().hash(&mut hasher);
        
        // Hash a sample of examples for efficiency
        let sample_size = std::cmp::min(10, trainset.len());
        for example in trainset.examples().iter().take(sample_size) {
            example.id.hash(&mut hasher);
            example.quality_score.to_bits().hash(&mut hasher);
        }
        
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Perform the actual compilation
    async fn perform_compilation<I, O, M>(
        &self,
        module: &mut M,
        teleprompter: &mut Teleprompter,
        trainset: ExampleSet<I, O>,
        _context: &CompilationContext,
    ) -> DspyResult<OptimizationResult>
    where
        I: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        O: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + Hash,
        M: Module<Input = I, Output = O> + Send + Sync,
    {
        debug!("Performing compilation for module: {}", module.name());

        // Set compilation timeout
        let timeout = Duration::from_secs(self.config.compilation_timeout_seconds);
        
        // Perform optimization with timeout
        let result = tokio::time::timeout(
            timeout,
            teleprompter.optimize(module, trainset)
        ).await
        .map_err(|_| DspyError::compilation("compilation_timeout", "Compilation timed out"))?
        .map_err(|e| DspyError::compilation("optimization_failed", &format!("Optimization failed: {}", e)))?;

        // Validate result
        if result.metrics.best_score < self.config.validation_threshold {
            warn!(
                "Compilation result below threshold: {} < {}",
                result.metrics.best_score,
                self.config.validation_threshold
            );
        }

        Ok(result)
    }

    /// Get cached compilation result
    async fn get_cached_compilation(&self, module_hash: &str) -> DspyResult<Option<CachedCompilation>> {
        let cache = self.cache.read().await;
        
        if let Some(cached) = cache.get(module_hash) {
            // Check if cache entry is still valid
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            if now - cached.created_at < self.config.cache_ttl_seconds {
                return Ok(Some(cached.clone()));
            } else {
                debug!("Cache entry expired for module hash: {}", module_hash);
            }
        }
        
        Ok(None)
    }

    /// Cache compilation result
    async fn cache_compilation(
        &self,
        context: &CompilationContext,
        result: &OptimizationResult,
    ) -> DspyResult<()> {
        let mut cache = self.cache.write().await;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let cached_compilation = CachedCompilation {
            context: context.clone(),
            parameters: HashMap::new(), // TODO: Extract module parameters
            result: result.clone(),
            created_at: now,
            last_accessed: now,
            access_count: 1,
        };
        
        cache.insert(context.module_hash.clone(), cached_compilation);
        
        // Clean up old entries if cache is too large
        self.cleanup_cache(&mut cache).await;
        
        Ok(())
    }

    /// Update cache access statistics
    async fn update_cache_access(&self, module_hash: &str) -> DspyResult<()> {
        let mut cache = self.cache.write().await;
        
        if let Some(cached) = cache.get_mut(module_hash) {
            cached.last_accessed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            cached.access_count += 1;
        }
        
        Ok(())
    }

    /// Clean up old cache entries
    async fn cleanup_cache(&self, cache: &mut HashMap<String, CachedCompilation>) {
        // Remove expired entries
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        cache.retain(|_, cached| {
            now - cached.created_at < self.config.cache_ttl_seconds
        });
        
        // TODO: Implement size-based cleanup if needed
    }

    /// Update compiler statistics
    async fn update_stats(&self, cache_hit: bool, compilation_time: Duration) {
        let mut stats = self.stats.write().await;
        
        stats.total_compilations += 1;
        
        if cache_hit {
            stats.cache_hit_rate = (stats.cache_hit_rate * (stats.total_compilations - 1) as f64 + 1.0) 
                / stats.total_compilations as f64;
        } else {
            stats.cache_hit_rate = (stats.cache_hit_rate * (stats.total_compilations - 1) as f64) 
                / stats.total_compilations as f64;
            stats.successful_compilations += 1;
        }
        
        let time_ms = compilation_time.as_millis() as f64;
        stats.avg_compilation_time_ms = (stats.avg_compilation_time_ms * (stats.total_compilations - 1) as f64 + time_ms) 
            / stats.total_compilations as f64;
    }

    /// Get compiler statistics
    pub async fn stats(&self) -> CompilerStats {
        self.stats.read().await.clone()
    }

    /// Clear compilation cache
    pub async fn clear_cache(&self) -> DspyResult<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Compilation cache cleared");
        Ok(())
    }

    /// Get cache size
    pub async fn cache_size(&self) -> usize {
        self.cache.read().await.len()
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CompilerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompilerStats[compilations: {}, cache_hit_rate: {:.2}%, avg_time: {:.1}ms]",
            self.total_compilations,
            self.cache_hit_rate * 100.0,
            self.avg_compilation_time_ms
        )
    }
}
