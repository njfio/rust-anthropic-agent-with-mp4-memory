//! # Synaptic Memory Configuration - Full Distributed Power
//!
//! This module provides configuration for the rust-synaptic memory system
//! with all advanced features enabled for maximum performance and capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Full power configuration for Synaptic memory system
/// Enables ALL advanced features for maximum distributed power
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullPowerSynapticConfig {
    /// Core memory configuration
    pub core: CoreMemoryConfig,
    /// Distributed system configuration
    pub distributed: DistributedConfig,
    /// External integrations configuration
    pub integrations: IntegrationsConfig,
    /// Embeddings and AI configuration
    pub embeddings: EmbeddingsConfig,
    /// Analytics and monitoring configuration
    pub analytics: AnalyticsConfig,
    /// Real-time features configuration
    pub realtime: RealtimeConfig,
}

/// Core memory system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMemoryConfig {
    /// Enable knowledge graph functionality
    pub enable_knowledge_graph: bool,
    /// Enable temporal tracking
    pub enable_temporal_tracking: bool,
    /// Enable advanced memory management
    pub enable_advanced_management: bool,
    /// Maximum number of short-term memories
    pub max_short_term_memories: usize,
    /// Maximum number of long-term memories
    pub max_long_term_memories: usize,
    /// Similarity threshold for memory matching
    pub similarity_threshold: f64,
    /// Checkpoint interval for memory persistence
    pub checkpoint_interval: usize,
    /// Memory compression settings
    pub compression_enabled: bool,
    /// Memory encryption settings
    pub encryption_enabled: bool,
}

/// Distributed system configuration for maximum power
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Enable distributed consensus (Raft/PBFT)
    pub enable_consensus: bool,
    /// Consensus algorithm to use
    pub consensus_algorithm: ConsensusAlgorithm,
    /// Kafka configuration for distributed messaging
    pub kafka_config: Option<KafkaConfig>,
    /// Node configuration for distributed cluster
    pub node_config: NodeConfig,
    /// Replication factor for distributed storage
    pub replication_factor: usize,
    /// Enable automatic failover
    pub enable_failover: bool,
    /// Enable load balancing
    pub enable_load_balancing: bool,
}

/// External integrations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationsConfig {
    /// PostgreSQL configuration
    pub postgresql: Option<PostgreSQLConfig>,
    /// Redis configuration
    pub redis: Option<RedisConfig>,
    /// Elasticsearch configuration
    pub elasticsearch: Option<ElasticsearchConfig>,
    /// Vector database configuration (Pinecone, Weaviate, etc.)
    pub vector_db: Option<VectorDBConfig>,
    /// Enable webhook notifications
    pub enable_webhooks: bool,
    /// Webhook endpoints
    pub webhook_endpoints: Vec<String>,
}

/// Embeddings and AI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    /// Enable vector embeddings
    pub enable_embeddings: bool,
    /// Embedding model to use
    pub embedding_model: EmbeddingModel,
    /// Embedding dimensions
    pub embedding_dimensions: usize,
    /// Enable semantic search
    pub enable_semantic_search: bool,
    /// Enable AI-powered memory consolidation
    pub enable_ai_consolidation: bool,
    /// Enable automatic memory tagging
    pub enable_auto_tagging: bool,
    /// OpenAI API configuration
    pub openai_config: Option<OpenAIConfig>,
    /// Anthropic API configuration
    pub anthropic_config: Option<AnthropicConfig>,
}

/// Analytics and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable performance analytics
    pub enable_performance_analytics: bool,
    /// Enable memory usage analytics
    pub enable_memory_analytics: bool,
    /// Enable query analytics
    pub enable_query_analytics: bool,
    /// Enable distributed system metrics
    pub enable_distributed_metrics: bool,
    /// Metrics collection interval (seconds)
    pub metrics_interval: u64,
    /// Enable Prometheus metrics export
    pub enable_prometheus: bool,
    /// Prometheus endpoint
    pub prometheus_endpoint: Option<String>,
}

/// Real-time features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Enable real-time synchronization
    pub enable_realtime_sync: bool,
    /// Enable WebSocket connections
    pub enable_websockets: bool,
    /// WebSocket port
    pub websocket_port: u16,
    /// Enable real-time notifications
    pub enable_notifications: bool,
    /// Enable live memory streaming
    pub enable_memory_streaming: bool,
    /// Sync interval (milliseconds)
    pub sync_interval: u64,
}

/// Consensus algorithms available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,
    /// Practical Byzantine Fault Tolerance
    PBFT,
    /// Proof of Stake consensus
    PoS,
}

/// Kafka configuration for distributed messaging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    /// Kafka bootstrap servers
    pub bootstrap_servers: Vec<String>,
    /// Topic for memory operations
    pub memory_topic: String,
    /// Topic for consensus messages
    pub consensus_topic: String,
    /// Consumer group ID
    pub consumer_group: String,
    /// Enable SSL
    pub enable_ssl: bool,
    /// Authentication configuration
    pub auth_config: Option<KafkaAuthConfig>,
}

/// Node configuration for distributed cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node ID
    pub node_id: String,
    /// Node address
    pub node_address: String,
    /// Node port
    pub node_port: u16,
    /// Cluster peers
    pub cluster_peers: Vec<String>,
    /// Enable leader election
    pub enable_leader_election: bool,
}

/// PostgreSQL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgreSQLConfig {
    /// Connection string
    pub connection_string: String,
    /// Database name
    pub database_name: String,
    /// Table prefix
    pub table_prefix: String,
    /// Connection pool size
    pub pool_size: u32,
    /// Enable SSL
    pub enable_ssl: bool,
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis URL
    pub url: String,
    /// Database number
    pub database: u8,
    /// Key prefix
    pub key_prefix: String,
    /// Connection pool size
    pub pool_size: u32,
    /// Enable clustering
    pub enable_clustering: bool,
}

/// Elasticsearch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticsearchConfig {
    /// Elasticsearch URL
    pub url: String,
    /// Index name
    pub index_name: String,
    /// Number of shards
    pub shards: u32,
    /// Number of replicas
    pub replicas: u32,
    /// Authentication
    pub auth: Option<ElasticsearchAuth>,
}

/// Vector database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDBConfig {
    /// Vector database type
    pub db_type: VectorDBType,
    /// Connection configuration
    pub connection: VectorDBConnection,
    /// Index configuration
    pub index_config: VectorIndexConfig,
}

/// Available embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// OpenAI text-embedding-ada-002
    OpenAIAda002,
    /// OpenAI text-embedding-3-small
    OpenAI3Small,
    /// OpenAI text-embedding-3-large
    OpenAI3Large,
    /// Sentence Transformers
    SentenceTransformers(String),
    /// Custom model
    Custom(String),
}

/// OpenAI API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key
    pub api_key: String,
    /// Organization ID
    pub organization_id: Option<String>,
    /// Base URL (for custom endpoints)
    pub base_url: Option<String>,
}

/// Anthropic API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key
    pub api_key: String,
    /// Base URL (for custom endpoints)
    pub base_url: Option<String>,
}

/// Kafka authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaAuthConfig {
    /// SASL mechanism
    pub sasl_mechanism: String,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
}

/// Elasticsearch authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticsearchAuth {
    /// Username
    pub username: String,
    /// Password
    pub password: String,
}

/// Vector database types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorDBType {
    /// Pinecone
    Pinecone,
    /// Weaviate
    Weaviate,
    /// Qdrant
    Qdrant,
    /// Milvus
    Milvus,
}

/// Vector database connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDBConnection {
    /// Connection URL
    pub url: String,
    /// API key
    pub api_key: Option<String>,
    /// Additional headers
    pub headers: HashMap<String, String>,
}

/// Vector index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    /// Index name
    pub index_name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Number of replicas
    pub replicas: u32,
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
}

impl Default for FullPowerSynapticConfig {
    fn default() -> Self {
        Self {
            core: CoreMemoryConfig::default(),
            distributed: DistributedConfig::default(),
            integrations: IntegrationsConfig::default(),
            embeddings: EmbeddingsConfig::default(),
            analytics: AnalyticsConfig::default(),
            realtime: RealtimeConfig::default(),
        }
    }
}

impl Default for CoreMemoryConfig {
    fn default() -> Self {
        Self {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            max_short_term_memories: 10000,
            max_long_term_memories: 100000,
            similarity_threshold: 0.8,
            checkpoint_interval: 1000,
            compression_enabled: true,
            encryption_enabled: true,
        }
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enable_consensus: true,
            consensus_algorithm: ConsensusAlgorithm::Raft,
            kafka_config: None, // Will be configured based on environment
            node_config: NodeConfig::default(),
            replication_factor: 3,
            enable_failover: true,
            enable_load_balancing: true,
        }
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            node_address: "127.0.0.1".to_string(),
            node_port: 8080,
            cluster_peers: vec![],
            enable_leader_election: true,
        }
    }
}

impl Default for IntegrationsConfig {
    fn default() -> Self {
        Self {
            postgresql: None,
            redis: None,
            elasticsearch: None,
            vector_db: None,
            enable_webhooks: false,
            webhook_endpoints: vec![],
        }
    }
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            enable_embeddings: true,
            embedding_model: EmbeddingModel::OpenAI3Small,
            embedding_dimensions: 1536,
            enable_semantic_search: true,
            enable_ai_consolidation: true,
            enable_auto_tagging: true,
            openai_config: None,
            anthropic_config: None,
        }
    }
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_performance_analytics: true,
            enable_memory_analytics: true,
            enable_query_analytics: true,
            enable_distributed_metrics: true,
            metrics_interval: 60,
            enable_prometheus: true,
            prometheus_endpoint: Some("http://localhost:9090".to_string()),
        }
    }
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            enable_realtime_sync: true,
            enable_websockets: true,
            websocket_port: 8081,
            enable_notifications: true,
            enable_memory_streaming: true,
            sync_interval: 1000,
        }
    }
}
