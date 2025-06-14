// Cache Backend Implementations
// Provides various backend storage implementations for different cache tiers

use super::strategies::DataSource;
use crate::utils::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// In-memory data source for testing and development
pub struct InMemoryDataSource {
    /// Data storage
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Simulate latency for testing
    simulate_latency: bool,
    /// Simulated latency in milliseconds
    latency_ms: u64,
}

/// File system data source
pub struct FileSystemDataSource {
    /// Base directory for file storage
    base_dir: String,
    /// File extension
    file_extension: String,
    /// Enable compression
    enable_compression: bool,
}

/// HTTP data source for REST APIs
pub struct HttpDataSource {
    /// Base URL for the API
    base_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Authentication headers
    auth_headers: HashMap<String, String>,
    /// Request timeout
    timeout: std::time::Duration,
}

/// Database data source (simplified interface)
pub struct DatabaseDataSource {
    /// Connection string
    connection_string: String,
    /// Table name
    table_name: String,
    /// Key column name
    key_column: String,
    /// Value column name
    value_column: String,
}

/// Mock data source for testing
pub struct MockDataSource {
    /// Mock data
    mock_data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Simulate failures
    failure_rate: f64,
    /// Health status
    healthy: Arc<RwLock<bool>>,
}

impl InMemoryDataSource {
    /// Create a new in-memory data source
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            simulate_latency: false,
            latency_ms: 0,
        }
    }

    /// Create with simulated latency for testing
    pub fn with_latency(latency_ms: u64) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            simulate_latency: true,
            latency_ms,
        }
    }

    /// Add test data
    pub async fn add_test_data<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let mut storage = self.data.write().await;
        storage.insert(key.to_string(), data);
        
        Ok(())
    }

    /// Get data count
    pub async fn count(&self) -> usize {
        let storage = self.data.read().await;
        storage.len()
    }

    /// Clear all data
    pub async fn clear(&self) {
        let mut storage = self.data.write().await;
        storage.clear();
    }
}

#[async_trait::async_trait]
impl DataSource for InMemoryDataSource {
    async fn load<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        // Simulate latency if enabled
        if self.simulate_latency {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }

        let storage = self.data.read().await;
        
        if let Some(data) = storage.get(key) {
            let value: T = serde_json::from_slice(data)
                .map_err(|e| AgentError::validation(format!("Deserialization failed: {}", e)))?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn save<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        // Simulate latency if enabled
        if self.simulate_latency {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }

        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let mut storage = self.data.write().await;
        storage.insert(key.to_string(), data);
        
        debug!("Saved data to in-memory source: {}", key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut storage = self.data.write().await;
        let removed = storage.remove(key).is_some();
        
        if removed {
            debug!("Deleted data from in-memory source: {}", key);
        }
        
        Ok(removed)
    }

    async fn health_check(&self) -> Result<bool> {
        // In-memory source is always healthy
        Ok(true)
    }
}

impl FileSystemDataSource {
    /// Create a new file system data source
    pub fn new(base_dir: String) -> Self {
        Self {
            base_dir,
            file_extension: "json".to_string(),
            enable_compression: false,
        }
    }

    /// Create with compression enabled
    pub fn with_compression(base_dir: String) -> Self {
        Self {
            base_dir,
            file_extension: "json.gz".to_string(),
            enable_compression: true,
        }
    }

    /// Get file path for key
    fn get_file_path(&self, key: &str) -> String {
        // Sanitize key for file system
        let safe_key = key.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        format!("{}/{}.{}", self.base_dir, safe_key, self.file_extension)
    }

    /// Ensure base directory exists
    async fn ensure_directory(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_dir).await
            .map_err(|e| AgentError::tool("filesystem_cache", &format!("Failed to create directory: {}", e)))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl DataSource for FileSystemDataSource {
    async fn load<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        let file_path = self.get_file_path(key);
        
        match tokio::fs::read(&file_path).await {
            Ok(data) => {
                let decompressed_data = if self.enable_compression {
                    // Decompress data
                    use flate2::read::GzDecoder;
                    use std::io::Read;
                    
                    let mut decoder = GzDecoder::new(&data[..]);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed)
                        .map_err(|e| AgentError::tool("filesystem_cache", &format!("Decompression failed: {}", e)))?;
                    decompressed
                } else {
                    data
                };
                
                let value: T = serde_json::from_slice(&decompressed_data)
                    .map_err(|e| AgentError::validation(format!("Deserialization failed: {}", e)))?;
                
                Ok(Some(value))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AgentError::tool("filesystem_cache", &format!("Failed to read file: {}", e))),
        }
    }

    async fn save<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        self.ensure_directory().await?;
        
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let final_data = if self.enable_compression {
            // Compress data
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::Write;
            
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data)
                .map_err(|e| AgentError::tool("filesystem_cache", &format!("Compression failed: {}", e)))?;
            encoder.finish()
                .map_err(|e| AgentError::tool("filesystem_cache", &format!("Compression finalization failed: {}", e)))?
        } else {
            data
        };
        
        let file_path = self.get_file_path(key);
        tokio::fs::write(&file_path, final_data).await
            .map_err(|e| AgentError::tool("filesystem_cache", &format!("Failed to write file: {}", e)))?;
        
        debug!("Saved data to file system: {}", key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let file_path = self.get_file_path(key);
        
        match tokio::fs::remove_file(&file_path).await {
            Ok(()) => {
                debug!("Deleted file: {}", key);
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(AgentError::tool("filesystem_cache", &format!("Failed to delete file: {}", e))),
        }
    }

    async fn health_check(&self) -> Result<bool> {
        // Check if base directory is accessible
        match tokio::fs::metadata(&self.base_dir).await {
            Ok(metadata) => Ok(metadata.is_dir()),
            Err(_) => {
                // Try to create directory
                Ok(self.ensure_directory().await.is_ok())
            }
        }
    }
}

impl HttpDataSource {
    /// Create a new HTTP data source
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            base_url,
            client,
            auth_headers: HashMap::new(),
            timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Add authentication header
    pub fn with_auth_header(mut self, key: String, value: String) -> Self {
        self.auth_headers.insert(key, value);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get URL for key
    fn get_url(&self, key: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), key)
    }
}

#[async_trait::async_trait]
impl DataSource for HttpDataSource {
    async fn load<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        let url = self.get_url(key);
        let mut request = self.client.get(&url);
        
        // Add authentication headers
        for (header_key, header_value) in &self.auth_headers {
            request = request.header(header_key, header_value);
        }
        
        match request.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let data = response.bytes().await
                        .map_err(|e| AgentError::tool("http_cache", &format!("Failed to read response: {}", e)))?;
                    
                    let value: T = serde_json::from_slice(&data)
                        .map_err(|e| AgentError::validation(format!("Deserialization failed: {}", e)))?;
                    
                    Ok(Some(value))
                } else if response.status() == 404 {
                    Ok(None)
                } else {
                    Err(AgentError::tool("http_cache", &format!("HTTP error: {}", response.status())))
                }
            }
            Err(e) => Err(AgentError::tool("http_cache", &format!("Request failed: {}", e))),
        }
    }

    async fn save<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        let url = self.get_url(key);
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let mut request = self.client.put(&url)
            .header("Content-Type", "application/json")
            .body(data);
        
        // Add authentication headers
        for (header_key, header_value) in &self.auth_headers {
            request = request.header(header_key, header_value);
        }
        
        let response = request.send().await
            .map_err(|e| AgentError::tool("http_cache", &format!("Request failed: {}", e)))?;
        
        if response.status().is_success() {
            debug!("Saved data via HTTP: {}", key);
            Ok(())
        } else {
            Err(AgentError::tool("http_cache", &format!("HTTP error: {}", response.status())))
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let url = self.get_url(key);
        let mut request = self.client.delete(&url);
        
        // Add authentication headers
        for (header_key, header_value) in &self.auth_headers {
            request = request.header(header_key, header_value);
        }
        
        match request.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    debug!("Deleted data via HTTP: {}", key);
                    Ok(true)
                } else if response.status() == 404 {
                    Ok(false)
                } else {
                    Err(AgentError::tool("http_cache", &format!("HTTP error: {}", response.status())))
                }
            }
            Err(e) => Err(AgentError::tool("http_cache", &format!("Request failed: {}", e))),
        }
    }

    async fn health_check(&self) -> Result<bool> {
        // Perform a simple HEAD request to check connectivity
        let mut request = self.client.head(&self.base_url);
        
        // Add authentication headers
        for (header_key, header_value) in &self.auth_headers {
            request = request.header(header_key, header_value);
        }
        
        match request.send().await {
            Ok(response) => Ok(response.status().is_success() || response.status() == 404),
            Err(_) => Ok(false),
        }
    }
}

impl MockDataSource {
    /// Create a new mock data source
    pub fn new() -> Self {
        Self {
            mock_data: Arc::new(RwLock::new(HashMap::new())),
            failure_rate: 0.0,
            healthy: Arc::new(RwLock::new(true)),
        }
    }

    /// Create with failure simulation
    pub fn with_failure_rate(failure_rate: f64) -> Self {
        Self {
            mock_data: Arc::new(RwLock::new(HashMap::new())),
            failure_rate: failure_rate.clamp(0.0, 1.0),
            healthy: Arc::new(RwLock::new(true)),
        }
    }

    /// Set health status
    pub async fn set_healthy(&self, healthy: bool) {
        let mut health = self.healthy.write().await;
        *health = healthy;
    }

    /// Add mock data
    pub async fn add_mock_data<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let mut storage = self.mock_data.write().await;
        storage.insert(key.to_string(), data);
        
        Ok(())
    }

    /// Check if operation should fail
    fn should_fail(&self) -> bool {
        if self.failure_rate > 0.0 {
            use rand::Rng;
            rand::thread_rng().gen::<f64>() < self.failure_rate
        } else {
            false
        }
    }
}

#[async_trait::async_trait]
impl DataSource for MockDataSource {
    async fn load<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + Send,
    {
        if self.should_fail() {
            return Err(AgentError::tool("mock_cache", "Simulated failure"));
        }

        let storage = self.mock_data.read().await;
        
        if let Some(data) = storage.get(key) {
            let value: T = serde_json::from_slice(data)
                .map_err(|e| AgentError::validation(format!("Deserialization failed: {}", e)))?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn save<T>(&self, key: &str, value: &T) -> Result<()>
    where
        T: Serialize + Send + Sync,
    {
        if self.should_fail() {
            return Err(AgentError::tool("mock_cache", "Simulated failure"));
        }

        let data = serde_json::to_vec(value)
            .map_err(|e| AgentError::validation(format!("Serialization failed: {}", e)))?;
        
        let mut storage = self.mock_data.write().await;
        storage.insert(key.to_string(), data);
        
        debug!("Saved data to mock source: {}", key);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        if self.should_fail() {
            return Err(AgentError::tool("mock_cache", "Simulated failure"));
        }

        let mut storage = self.mock_data.write().await;
        let removed = storage.remove(key).is_some();
        
        if removed {
            debug!("Deleted data from mock source: {}", key);
        }
        
        Ok(removed)
    }

    async fn health_check(&self) -> Result<bool> {
        let healthy = self.healthy.read().await;
        Ok(*healthy)
    }
}

impl Default for InMemoryDataSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MockDataSource {
    fn default() -> Self {
        Self::new()
    }
}
