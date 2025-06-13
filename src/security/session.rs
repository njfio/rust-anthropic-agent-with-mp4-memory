use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::utils::error::{AgentError, Result};
use super::{SessionConfig, SessionStorageBackend};

/// Session manager trait
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Create a new session
    async fn create_session(&self, user_id: &str, ip_address: Option<String>) -> Result<String>;
    
    /// Get session information
    async fn get_session(&self, session_id: &str) -> Result<Option<Session>>;
    
    /// Update session
    async fn update_session(&self, session: Session) -> Result<()>;
    
    /// Terminate a session
    async fn terminate_session(&self, session_id: &str) -> Result<()>;
    
    /// Check if session is valid
    async fn is_session_valid(&self, session_id: &str) -> Result<bool>;
    
    /// Refresh session (extend expiration)
    async fn refresh_session(&self, session_id: &str) -> Result<()>;
    
    /// Get all sessions for a user
    async fn get_user_sessions(&self, user_id: &str) -> Result<Vec<Session>>;
    
    /// Terminate all sessions for a user
    async fn terminate_user_sessions(&self, user_id: &str) -> Result<u32>;
    
    /// Clean up expired sessions
    async fn cleanup_expired_sessions(&self) -> Result<u32>;
    
    /// Get session statistics
    async fn get_session_statistics(&self) -> Result<SessionStatistics>;
    
    /// Set session data
    async fn set_session_data(&self, session_id: &str, key: &str, value: String) -> Result<()>;
    
    /// Get session data
    async fn get_session_data(&self, session_id: &str, key: &str) -> Result<Option<String>>;
    
    /// Remove session data
    async fn remove_session_data(&self, session_id: &str, key: &str) -> Result<()>;
}

/// Session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Session creation time
    pub created_at: SystemTime,
    /// Session last accessed time
    pub last_accessed: SystemTime,
    /// Session expiration time
    pub expires_at: SystemTime,
    /// IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Session status
    pub status: SessionStatus,
    /// Session data
    pub data: HashMap<String, String>,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Session is active
    Active,
    /// Session is expired
    Expired,
    /// Session is terminated
    Terminated,
    /// Session is suspended
    Suspended,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total active sessions
    pub active_sessions: u32,
    /// Total expired sessions
    pub expired_sessions: u32,
    /// Total terminated sessions
    pub terminated_sessions: u32,
    /// Sessions by user
    pub sessions_by_user: HashMap<String, u32>,
    /// Sessions by IP address
    pub sessions_by_ip: HashMap<String, u32>,
    /// Average session duration in seconds
    pub avg_session_duration_seconds: f64,
    /// Total sessions created today
    pub sessions_created_today: u32,
    /// Peak concurrent sessions
    pub peak_concurrent_sessions: u32,
}

/// In-memory session manager implementation
pub struct MemorySessionManager {
    /// Session configuration
    config: SessionConfig,
    /// Active sessions
    sessions: RwLock<HashMap<String, Session>>,
    /// User session mapping
    user_sessions: RwLock<HashMap<String, Vec<String>>>,
    /// Session statistics
    statistics: RwLock<SessionStatistics>,
}

impl MemorySessionManager {
    /// Create a new memory session manager
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            sessions: RwLock::new(HashMap::new()),
            user_sessions: RwLock::new(HashMap::new()),
            statistics: RwLock::new(SessionStatistics {
                active_sessions: 0,
                expired_sessions: 0,
                terminated_sessions: 0,
                sessions_by_user: HashMap::new(),
                sessions_by_ip: HashMap::new(),
                avg_session_duration_seconds: 0.0,
                sessions_created_today: 0,
                peak_concurrent_sessions: 0,
            }),
        }
    }

    /// Check if session has expired
    fn is_session_expired(&self, session: &Session) -> bool {
        SystemTime::now() > session.expires_at
    }

    /// Calculate session expiration time
    fn calculate_expiration(&self) -> SystemTime {
        SystemTime::now() + Duration::from_secs(self.config.timeout_seconds)
    }

    /// Update session statistics
    async fn update_statistics(&self, operation: StatisticsOperation) {
        let mut stats = self.statistics.write().await;
        
        match operation {
            StatisticsOperation::SessionCreated { user_id, ip_address } => {
                stats.active_sessions += 1;
                stats.sessions_created_today += 1;
                
                if stats.active_sessions > stats.peak_concurrent_sessions {
                    stats.peak_concurrent_sessions = stats.active_sessions;
                }
                
                *stats.sessions_by_user.entry(user_id).or_insert(0) += 1;
                
                if let Some(ip) = ip_address {
                    *stats.sessions_by_ip.entry(ip).or_insert(0) += 1;
                }
            }
            StatisticsOperation::SessionTerminated => {
                if stats.active_sessions > 0 {
                    stats.active_sessions -= 1;
                }
                stats.terminated_sessions += 1;
            }
            StatisticsOperation::SessionExpired => {
                if stats.active_sessions > 0 {
                    stats.active_sessions -= 1;
                }
                stats.expired_sessions += 1;
            }
        }
    }

    /// Enforce concurrent session limits
    async fn enforce_session_limits(&self, user_id: &str) -> Result<()> {
        let oldest_session_id = {
            let user_sessions = self.user_sessions.read().await;
            if let Some(session_ids) = user_sessions.get(user_id) {
                if session_ids.len() >= self.config.max_concurrent_sessions as usize {
                    // Get oldest session ID
                    session_ids.first().cloned()
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(session_id) = oldest_session_id {
            self.terminate_session(&session_id).await?;
        }

        Ok(())
    }
}

/// Statistics operation types
enum StatisticsOperation {
    SessionCreated { user_id: String, ip_address: Option<String> },
    SessionTerminated,
    SessionExpired,
}

#[async_trait]
impl SessionManager for MemorySessionManager {
    async fn create_session(&self, user_id: &str, ip_address: Option<String>) -> Result<String> {
        // Enforce session limits
        self.enforce_session_limits(user_id).await?;

        let session_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();
        
        let session = Session {
            id: session_id.clone(),
            user_id: user_id.to_string(),
            created_at: now,
            last_accessed: now,
            expires_at: self.calculate_expiration(),
            ip_address: ip_address.clone(),
            user_agent: None,
            status: SessionStatus::Active,
            data: HashMap::new(),
            metadata: HashMap::new(),
        };

        // Store session
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);
        drop(sessions);

        // Update user sessions mapping
        let mut user_sessions = self.user_sessions.write().await;
        user_sessions.entry(user_id.to_string()).or_insert_with(Vec::new).push(session_id.clone());
        drop(user_sessions);

        // Update statistics
        self.update_statistics(StatisticsOperation::SessionCreated {
            user_id: user_id.to_string(),
            ip_address,
        }).await;

        Ok(session_id)
    }

    async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }

    async fn update_session(&self, session: Session) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session);
        Ok(())
    }

    async fn terminate_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(mut session) = sessions.remove(session_id) {
            session.status = SessionStatus::Terminated;
            
            // Remove from user sessions mapping
            let mut user_sessions = self.user_sessions.write().await;
            if let Some(session_ids) = user_sessions.get_mut(&session.user_id) {
                session_ids.retain(|id| id != session_id);
                if session_ids.is_empty() {
                    user_sessions.remove(&session.user_id);
                }
            }
            drop(user_sessions);
            
            // Update statistics
            self.update_statistics(StatisticsOperation::SessionTerminated).await;
        }

        Ok(())
    }

    async fn is_session_valid(&self, session_id: &str) -> Result<bool> {
        let sessions = self.sessions.read().await;
        
        if let Some(session) = sessions.get(session_id) {
            if session.status != SessionStatus::Active {
                return Ok(false);
            }
            
            if self.is_session_expired(session) {
                return Ok(false);
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn refresh_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            if session.status == SessionStatus::Active && !self.is_session_expired(session) {
                session.last_accessed = SystemTime::now();
                session.expires_at = self.calculate_expiration();
            }
        }

        Ok(())
    }

    async fn get_user_sessions(&self, user_id: &str) -> Result<Vec<Session>> {
        let user_sessions = self.user_sessions.read().await;
        let sessions = self.sessions.read().await;
        
        if let Some(session_ids) = user_sessions.get(user_id) {
            let user_session_list: Vec<Session> = session_ids
                .iter()
                .filter_map(|id| sessions.get(id).cloned())
                .collect();
            Ok(user_session_list)
        } else {
            Ok(Vec::new())
        }
    }

    async fn terminate_user_sessions(&self, user_id: &str) -> Result<u32> {
        let user_sessions = self.user_sessions.read().await;
        let session_ids = user_sessions.get(user_id).cloned().unwrap_or_default();
        drop(user_sessions);

        let mut terminated_count = 0;
        for session_id in session_ids {
            if self.terminate_session(&session_id).await.is_ok() {
                terminated_count += 1;
            }
        }

        Ok(terminated_count)
    }

    async fn cleanup_expired_sessions(&self) -> Result<u32> {
        let mut sessions = self.sessions.write().await;
        let mut user_sessions = self.user_sessions.write().await;
        
        let mut expired_sessions = Vec::new();
        let now = SystemTime::now();

        // Find expired sessions
        for (session_id, session) in sessions.iter() {
            if session.status == SessionStatus::Active && now > session.expires_at {
                expired_sessions.push(session_id.clone());
            }
        }

        // Remove expired sessions
        let mut cleanup_count = 0;
        for session_id in expired_sessions {
            if let Some(session) = sessions.remove(&session_id) {
                // Remove from user sessions mapping
                if let Some(session_ids) = user_sessions.get_mut(&session.user_id) {
                    session_ids.retain(|id| id != &session_id);
                    if session_ids.is_empty() {
                        user_sessions.remove(&session.user_id);
                    }
                }
                cleanup_count += 1;
            }
        }

        drop(sessions);
        drop(user_sessions);

        // Update statistics
        for _ in 0..cleanup_count {
            self.update_statistics(StatisticsOperation::SessionExpired).await;
        }

        Ok(cleanup_count)
    }

    async fn get_session_statistics(&self) -> Result<SessionStatistics> {
        let stats = self.statistics.read().await;
        Ok(stats.clone())
    }

    async fn set_session_data(&self, session_id: &str, key: &str, value: String) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            if session.status == SessionStatus::Active && !self.is_session_expired(session) {
                session.data.insert(key.to_string(), value);
                session.last_accessed = SystemTime::now();
            } else {
                return Err(AgentError::validation("Session is not active or has expired".to_string()));
            }
        } else {
            return Err(AgentError::validation("Session not found".to_string()));
        }

        Ok(())
    }

    async fn get_session_data(&self, session_id: &str, key: &str) -> Result<Option<String>> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            if session.status == SessionStatus::Active && !self.is_session_expired(session) {
                session.last_accessed = SystemTime::now();
                Ok(session.data.get(key).cloned())
            } else {
                Err(AgentError::validation("Session is not active or has expired".to_string()))
            }
        } else {
            Err(AgentError::validation("Session not found".to_string()))
        }
    }

    async fn remove_session_data(&self, session_id: &str, key: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            if session.status == SessionStatus::Active && !self.is_session_expired(session) {
                session.data.remove(key);
                session.last_accessed = SystemTime::now();
            } else {
                return Err(AgentError::validation("Session is not active or has expired".to_string()));
            }
        } else {
            return Err(AgentError::validation("Session not found".to_string()));
        }

        Ok(())
    }
}

/// Create a session manager
pub async fn create_session_manager(config: &SessionConfig) -> Result<Box<dyn SessionManager>> {
    match config.storage_backend {
        SessionStorageBackend::Memory => {
            Ok(Box::new(MemorySessionManager::new(config.clone())))
        }
        _ => Err(AgentError::validation("Session storage backend not supported".to_string())),
    }
}
