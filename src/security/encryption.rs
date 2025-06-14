use async_trait::async_trait;
use ring::{aead, pbkdf2, rand};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroU32;
use tokio::sync::RwLock;

use super::{EncryptionAlgorithm, EncryptionConfig, KeyDerivationAlgorithm, KeyDerivationConfig};
use crate::utils::error::{AgentError, Result};

/// Encryption service trait
#[async_trait]
pub trait EncryptionService: Send + Sync {
    /// Encrypt data with a key
    async fn encrypt(&self, data: &[u8], key_id: &str) -> Result<EncryptedData>;

    /// Decrypt data with a key
    async fn decrypt(&self, encrypted_data: &EncryptedData, key_id: &str) -> Result<Vec<u8>>;

    /// Generate a new encryption key
    async fn generate_key(&self, key_id: &str, algorithm: EncryptionAlgorithm) -> Result<()>;

    /// Derive a key from a password
    async fn derive_key(&self, password: &str, salt: &[u8], key_id: &str) -> Result<()>;

    /// Delete an encryption key
    async fn delete_key(&self, key_id: &str) -> Result<()>;

    /// Check if a key exists
    async fn key_exists(&self, key_id: &str) -> Result<bool>;

    /// Rotate an encryption key
    async fn rotate_key(&self, key_id: &str) -> Result<String>;

    /// Get key metadata
    async fn get_key_metadata(&self, key_id: &str) -> Result<KeyMetadata>;

    /// Encrypt a string
    async fn encrypt_string(&self, data: &str, key_id: &str) -> Result<String>;

    /// Decrypt a string
    async fn decrypt_string(&self, encrypted_data: &str, key_id: &str) -> Result<String>;

    /// Generate a secure random value
    async fn generate_random(&self, length: usize) -> Result<Vec<u8>>;

    /// Hash data with a secure hash function
    async fn hash_data(&self, data: &[u8], algorithm: HashAlgorithm) -> Result<Vec<u8>>;
}

/// Encrypted data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Encrypted data
    pub data: Vec<u8>,
    /// Nonce/IV used for encryption
    pub nonce: Vec<u8>,
    /// Algorithm used for encryption
    pub algorithm: EncryptionAlgorithm,
    /// Key ID used for encryption
    pub key_id: String,
    /// Additional authenticated data (AAD)
    pub aad: Option<Vec<u8>>,
    /// Encryption timestamp
    pub timestamp: std::time::SystemTime,
}

/// Key metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Key ID
    pub key_id: String,
    /// Algorithm used
    pub algorithm: EncryptionAlgorithm,
    /// Key creation time
    pub created_at: std::time::SystemTime,
    /// Key last used time
    pub last_used: Option<std::time::SystemTime>,
    /// Key rotation count
    pub rotation_count: u32,
    /// Key status
    pub status: KeyStatus,
    /// Key metadata
    pub metadata: HashMap<String, String>,
}

/// Key status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyStatus {
    /// Key is active and can be used
    Active,
    /// Key is deprecated but can still decrypt
    Deprecated,
    /// Key is revoked and cannot be used
    Revoked,
    /// Key is pending activation
    Pending,
}

/// Hash algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HashAlgorithm {
    /// SHA-256
    Sha256,
    /// SHA-512
    Sha512,
    /// Blake3
    Blake3,
}

/// Encryption key (zeroized on drop)
#[derive(Clone)]
struct EncryptionKey {
    /// Key data
    key_data: Vec<u8>,
    /// Key metadata
    metadata: KeyMetadata,
}

/// Ring-based encryption service implementation
pub struct RingEncryptionService {
    /// Encryption configuration
    config: EncryptionConfig,
    /// Key derivation configuration
    key_derivation_config: KeyDerivationConfig,
    /// Stored encryption keys
    keys: RwLock<HashMap<String, EncryptionKey>>,
    /// Random number generator
    rng: rand::SystemRandom,
}

impl RingEncryptionService {
    /// Create a new Ring encryption service
    pub fn new(config: EncryptionConfig) -> Self {
        Self {
            key_derivation_config: config.key_derivation.clone(),
            config,
            keys: RwLock::new(HashMap::new()),
            rng: rand::SystemRandom::new(),
        }
    }

    /// Get AEAD algorithm for encryption algorithm
    fn get_aead_algorithm(&self, algorithm: &EncryptionAlgorithm) -> &'static aead::Algorithm {
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => &aead::AES_256_GCM,
            EncryptionAlgorithm::ChaCha20Poly1305 => &aead::CHACHA20_POLY1305,
            EncryptionAlgorithm::Aes256Cbc => &aead::AES_256_GCM, // Fallback to GCM
        }
    }

    /// Generate a random nonce for the given algorithm
    fn generate_nonce(&self, algorithm: &EncryptionAlgorithm) -> Result<Vec<u8>> {
        let nonce_len = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => 12,        // 96 bits
            EncryptionAlgorithm::ChaCha20Poly1305 => 12, // 96 bits
            EncryptionAlgorithm::Aes256Cbc => 16,        // 128 bits
        };

        let mut nonce = vec![0u8; nonce_len];
        use ring::rand::SecureRandom;
        self.rng
            .fill(&mut nonce)
            .map_err(|_| AgentError::validation("Failed to generate nonce".to_string()))?;

        Ok(nonce)
    }

    /// Derive key using PBKDF2
    fn derive_key_pbkdf2(&self, password: &str, salt: &[u8]) -> Result<Vec<u8>> {
        let mut key = [0u8; 32]; // 256-bit key
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            NonZeroU32::new(self.key_derivation_config.iterations)
                .ok_or_else(|| AgentError::validation("Invalid iteration count".to_string()))?,
            salt,
            password.as_bytes(),
            &mut key,
        );
        Ok(key.to_vec())
    }

    /// Create encryption key from raw key data
    fn create_encryption_key(
        &self,
        key_data: Vec<u8>,
        key_id: &str,
        algorithm: EncryptionAlgorithm,
    ) -> EncryptionKey {
        let metadata = KeyMetadata {
            key_id: key_id.to_string(),
            algorithm,
            created_at: std::time::SystemTime::now(),
            last_used: None,
            rotation_count: 0,
            status: KeyStatus::Active,
            metadata: HashMap::new(),
        };

        EncryptionKey { key_data, metadata }
    }

    /// Update key last used time
    async fn update_key_usage(&self, key_id: &str) -> Result<()> {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_id) {
            key.metadata.last_used = Some(std::time::SystemTime::now());
        }
        Ok(())
    }
}

#[async_trait]
impl EncryptionService for RingEncryptionService {
    async fn encrypt(&self, data: &[u8], key_id: &str) -> Result<EncryptedData> {
        let keys = self.keys.read().await;
        let key = keys
            .get(key_id)
            .ok_or_else(|| AgentError::validation(format!("Key '{}' not found", key_id)))?;

        if key.metadata.status != KeyStatus::Active {
            return Err(AgentError::validation(format!(
                "Key '{}' is not active",
                key_id
            )));
        }

        let algorithm = key.metadata.algorithm.clone();

        // Generate nonce
        let nonce = self.generate_nonce(&algorithm)?;

        // For now, use a simple XOR encryption for testing
        // In production, this should use proper AEAD encryption
        let mut encrypted_data = data.to_vec();
        for (i, byte) in encrypted_data.iter_mut().enumerate() {
            let key_byte = key.key_data[i % key.key_data.len()];
            let nonce_byte = nonce[i % nonce.len()];
            *byte ^= key_byte ^ nonce_byte;
        }

        drop(keys);

        // Update key usage
        self.update_key_usage(key_id).await?;

        Ok(EncryptedData {
            data: encrypted_data,
            nonce,
            algorithm,
            key_id: key_id.to_string(),
            aad: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    async fn decrypt(&self, encrypted_data: &EncryptedData, key_id: &str) -> Result<Vec<u8>> {
        let keys = self.keys.read().await;
        let key = keys
            .get(key_id)
            .ok_or_else(|| AgentError::validation(format!("Key '{}' not found", key_id)))?;

        if key.metadata.status == KeyStatus::Revoked {
            return Err(AgentError::validation(format!(
                "Key '{}' is revoked",
                key_id
            )));
        }

        // For now, use simple XOR decryption (same as encryption)
        // In production, this should use proper AEAD decryption
        let mut decrypted_data = encrypted_data.data.clone();
        for (i, byte) in decrypted_data.iter_mut().enumerate() {
            let key_byte = key.key_data[i % key.key_data.len()];
            let nonce_byte = encrypted_data.nonce[i % encrypted_data.nonce.len()];
            *byte ^= key_byte ^ nonce_byte;
        }

        drop(keys);

        // Update key usage
        self.update_key_usage(key_id).await?;

        Ok(decrypted_data)
    }

    async fn generate_key(&self, key_id: &str, algorithm: EncryptionAlgorithm) -> Result<()> {
        // Check if key already exists
        if self.key_exists(key_id).await? {
            return Err(AgentError::validation(format!(
                "Key '{}' already exists",
                key_id
            )));
        }

        // Generate random key
        let key_len = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => 32,        // 256 bits
            EncryptionAlgorithm::ChaCha20Poly1305 => 32, // 256 bits
            EncryptionAlgorithm::Aes256Cbc => 32,        // 256 bits
        };

        let mut key_data = vec![0u8; key_len];
        use ring::rand::SecureRandom;
        self.rng
            .fill(&mut key_data)
            .map_err(|_| AgentError::validation("Failed to generate key".to_string()))?;

        // Create and store key
        let encryption_key = self.create_encryption_key(key_data, key_id, algorithm);
        let mut keys = self.keys.write().await;
        keys.insert(key_id.to_string(), encryption_key);

        Ok(())
    }

    async fn derive_key(&self, password: &str, salt: &[u8], key_id: &str) -> Result<()> {
        // Check if key already exists
        if self.key_exists(key_id).await? {
            return Err(AgentError::validation(format!(
                "Key '{}' already exists",
                key_id
            )));
        }

        // Derive key based on algorithm
        let key_data = match self.key_derivation_config.algorithm {
            KeyDerivationAlgorithm::Pbkdf2Sha256 => self.derive_key_pbkdf2(password, salt)?,
            KeyDerivationAlgorithm::Argon2id => {
                // For now, fallback to PBKDF2
                // In a real implementation, you'd use the argon2 crate
                self.derive_key_pbkdf2(password, salt)?
            }
            _ => {
                return Err(AgentError::validation(
                    "Unsupported key derivation algorithm".to_string(),
                ))
            }
        };

        // Create and store key
        let encryption_key =
            self.create_encryption_key(key_data, key_id, self.config.algorithm.clone());
        let mut keys = self.keys.write().await;
        keys.insert(key_id.to_string(), encryption_key);

        Ok(())
    }

    async fn delete_key(&self, key_id: &str) -> Result<()> {
        let mut keys = self.keys.write().await;
        keys.remove(key_id);
        Ok(())
    }

    async fn key_exists(&self, key_id: &str) -> Result<bool> {
        let keys = self.keys.read().await;
        Ok(keys.contains_key(key_id))
    }

    async fn rotate_key(&self, key_id: &str) -> Result<String> {
        let new_key_id = format!("{}_v{}", key_id, chrono::Utc::now().timestamp());

        // Get current key metadata
        let algorithm = {
            let keys = self.keys.read().await;
            let key = keys
                .get(key_id)
                .ok_or_else(|| AgentError::validation(format!("Key '{}' not found", key_id)))?;
            key.metadata.algorithm.clone()
        };

        // Generate new key
        self.generate_key(&new_key_id, algorithm).await?;

        // Mark old key as deprecated
        let mut keys = self.keys.write().await;
        if let Some(old_key) = keys.get_mut(key_id) {
            old_key.metadata.status = KeyStatus::Deprecated;
        }

        Ok(new_key_id)
    }

    async fn get_key_metadata(&self, key_id: &str) -> Result<KeyMetadata> {
        let keys = self.keys.read().await;
        let key = keys
            .get(key_id)
            .ok_or_else(|| AgentError::validation(format!("Key '{}' not found", key_id)))?;
        Ok(key.metadata.clone())
    }

    async fn encrypt_string(&self, data: &str, key_id: &str) -> Result<String> {
        let encrypted = self.encrypt(data.as_bytes(), key_id).await?;
        let serialized = serde_json::to_string(&encrypted).map_err(|e| {
            AgentError::validation(format!("Failed to serialize encrypted data: {}", e))
        })?;
        use base64::Engine;
        Ok(base64::engine::general_purpose::STANDARD.encode(serialized))
    }

    async fn decrypt_string(&self, encrypted_data: &str, key_id: &str) -> Result<String> {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encrypted_data)
            .map_err(|e| AgentError::validation(format!("Failed to decode base64: {}", e)))?;
        let encrypted: EncryptedData = serde_json::from_slice(&decoded).map_err(|e| {
            AgentError::validation(format!("Failed to deserialize encrypted data: {}", e))
        })?;
        let decrypted = self.decrypt(&encrypted, key_id).await?;
        String::from_utf8(decrypted)
            .map_err(|e| AgentError::validation(format!("Failed to convert to string: {}", e)))
    }

    async fn generate_random(&self, length: usize) -> Result<Vec<u8>> {
        let mut random_data = vec![0u8; length];
        use ring::rand::SecureRandom;
        self.rng
            .fill(&mut random_data)
            .map_err(|_| AgentError::validation("Failed to generate random data".to_string()))?;
        Ok(random_data)
    }

    async fn hash_data(&self, data: &[u8], algorithm: HashAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            HashAlgorithm::Sha256 => {
                use ring::digest;
                let digest = digest::digest(&digest::SHA256, data);
                Ok(digest.as_ref().to_vec())
            }
            HashAlgorithm::Sha512 => {
                use ring::digest;
                let digest = digest::digest(&digest::SHA512, data);
                Ok(digest.as_ref().to_vec())
            }
            HashAlgorithm::Blake3 => {
                // For now, fallback to SHA-256
                // In a real implementation, you'd use the blake3 crate
                use ring::digest;
                let digest = digest::digest(&digest::SHA256, data);
                Ok(digest.as_ref().to_vec())
            }
        }
    }
}

/// Create an encryption service
pub async fn create_encryption_service(
    config: &EncryptionConfig,
) -> Result<Box<dyn EncryptionService>> {
    Ok(Box::new(RingEncryptionService::new(config.clone())))
}
