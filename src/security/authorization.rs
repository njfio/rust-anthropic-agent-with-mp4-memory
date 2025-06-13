use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

use crate::utils::error::{AgentError, Result};
use super::SecurityContext;

/// Authorization service trait
#[async_trait]
pub trait AuthorizationService: Send + Sync {
    /// Check if a user has permission to perform an action on a resource
    async fn check_permission(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<bool>;
    
    /// Get all permissions for a user
    async fn get_user_permissions(&self, user_id: &str) -> Result<Vec<Permission>>;
    
    /// Get all roles for a user
    async fn get_user_roles(&self, user_id: &str) -> Result<Vec<Role>>;
    
    /// Add a role to a user
    async fn add_user_role(&self, user_id: &str, role_name: &str) -> Result<()>;
    
    /// Remove a role from a user
    async fn remove_user_role(&self, user_id: &str, role_name: &str) -> Result<()>;
    
    /// Create a new role
    async fn create_role(&self, role: Role) -> Result<()>;
    
    /// Update a role
    async fn update_role(&self, role: Role) -> Result<()>;
    
    /// Delete a role
    async fn delete_role(&self, role_name: &str) -> Result<()>;
    
    /// Create a new permission
    async fn create_permission(&self, permission: Permission) -> Result<()>;
    
    /// Update a permission
    async fn update_permission(&self, permission: Permission) -> Result<()>;
    
    /// Delete a permission
    async fn delete_permission(&self, permission_name: &str) -> Result<()>;
    
    /// Check if a role exists
    async fn role_exists(&self, role_name: &str) -> Result<bool>;
    
    /// Check if a permission exists
    async fn permission_exists(&self, permission_name: &str) -> Result<bool>;
    
    /// Get role hierarchy
    async fn get_role_hierarchy(&self) -> Result<HashMap<String, Vec<String>>>;
    
    /// Evaluate a policy
    async fn evaluate_policy(&self, context: &SecurityContext, policy: &Policy) -> Result<bool>;
}

/// Permission definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Permission {
    /// Permission name
    pub name: String,
    /// Permission description
    pub description: String,
    /// Resource type this permission applies to
    pub resource_type: String,
    /// Actions allowed by this permission
    pub actions: Vec<String>,
    /// Conditions for this permission
    pub conditions: Vec<Condition>,
    /// Permission metadata
    pub metadata: HashMap<String, String>,
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: Vec<String>,
    /// Parent roles (for role hierarchy)
    pub parent_roles: Vec<String>,
    /// Role metadata
    pub metadata: HashMap<String, String>,
}

/// Authorization condition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Condition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Field to check
    pub field: String,
    /// Operator for comparison
    pub operator: ConditionOperator,
    /// Value to compare against
    pub value: String,
}

/// Condition types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionType {
    /// Time-based condition
    Time,
    /// IP address condition
    IpAddress,
    /// User attribute condition
    UserAttribute,
    /// Resource attribute condition
    ResourceAttribute,
    /// Context attribute condition
    ContextAttribute,
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionOperator {
    /// Equal to
    Equals,
    /// Not equal to
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Contains
    Contains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
    /// In list
    In,
    /// Not in list
    NotIn,
    /// Matches regex
    Regex,
}

/// Authorization policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy effect (allow or deny)
    pub effect: PolicyEffect,
    /// Resources this policy applies to
    pub resources: Vec<String>,
    /// Actions this policy applies to
    pub actions: Vec<String>,
    /// Conditions for this policy
    pub conditions: Vec<Condition>,
    /// Policy priority (higher number = higher priority)
    pub priority: i32,
}

/// Policy effect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PolicyEffect {
    /// Allow access
    Allow,
    /// Deny access
    Deny,
}

/// Authorization decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationDecision {
    /// Whether access is granted
    pub granted: bool,
    /// Reason for the decision
    pub reason: String,
    /// Policies that were evaluated
    pub evaluated_policies: Vec<String>,
    /// Permissions that were checked
    pub checked_permissions: Vec<String>,
    /// Decision metadata
    pub metadata: HashMap<String, String>,
}

/// Role-based access control (RBAC) authorization service
pub struct RbacAuthorizationService {
    /// User roles mapping
    user_roles: RwLock<HashMap<String, HashSet<String>>>,
    /// Role definitions
    roles: RwLock<HashMap<String, Role>>,
    /// Permission definitions
    permissions: RwLock<HashMap<String, Permission>>,
    /// Policies
    policies: RwLock<HashMap<String, Policy>>,
}

impl RbacAuthorizationService {
    /// Create a new RBAC authorization service
    pub fn new() -> Self {
        Self {
            user_roles: RwLock::new(HashMap::new()),
            roles: RwLock::new(HashMap::new()),
            permissions: RwLock::new(HashMap::new()),
            policies: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize with default roles and permissions
    pub async fn initialize_defaults(&self) -> Result<()> {
        // Create default permissions
        let default_permissions = vec![
            Permission {
                name: "read".to_string(),
                description: "Read access".to_string(),
                resource_type: "*".to_string(),
                actions: vec!["read".to_string(), "view".to_string()],
                conditions: Vec::new(),
                metadata: HashMap::new(),
            },
            Permission {
                name: "write".to_string(),
                description: "Write access".to_string(),
                resource_type: "*".to_string(),
                actions: vec!["write".to_string(), "create".to_string(), "update".to_string()],
                conditions: Vec::new(),
                metadata: HashMap::new(),
            },
            Permission {
                name: "delete".to_string(),
                description: "Delete access".to_string(),
                resource_type: "*".to_string(),
                actions: vec!["delete".to_string()],
                conditions: Vec::new(),
                metadata: HashMap::new(),
            },
            Permission {
                name: "admin".to_string(),
                description: "Administrative access".to_string(),
                resource_type: "*".to_string(),
                actions: vec!["*".to_string()],
                conditions: Vec::new(),
                metadata: HashMap::new(),
            },
        ];

        for permission in default_permissions {
            self.create_permission(permission).await?;
        }

        // Create default roles
        let default_roles = vec![
            Role {
                name: "user".to_string(),
                description: "Basic user role".to_string(),
                permissions: vec!["read".to_string()],
                parent_roles: Vec::new(),
                metadata: HashMap::new(),
            },
            Role {
                name: "editor".to_string(),
                description: "Editor role with read/write access".to_string(),
                permissions: vec!["read".to_string(), "write".to_string()],
                parent_roles: vec!["user".to_string()],
                metadata: HashMap::new(),
            },
            Role {
                name: "admin".to_string(),
                description: "Administrator role with full access".to_string(),
                permissions: vec!["read".to_string(), "write".to_string(), "delete".to_string(), "admin".to_string()],
                parent_roles: vec!["editor".to_string()],
                metadata: HashMap::new(),
            },
        ];

        for role in default_roles {
            self.create_role(role).await?;
        }

        Ok(())
    }

    /// Get all permissions for a role, including inherited permissions
    async fn get_role_permissions(&self, role_name: &str) -> Result<Vec<String>> {
        let roles = self.roles.read().await;
        let mut permissions = HashSet::new();
        let mut visited = HashSet::new();
        
        self.collect_role_permissions(role_name, &roles, &mut permissions, &mut visited)?;
        
        Ok(permissions.into_iter().collect())
    }

    /// Recursively collect permissions from a role and its parents
    fn collect_role_permissions(
        &self,
        role_name: &str,
        roles: &HashMap<String, Role>,
        permissions: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        if visited.contains(role_name) {
            return Ok(());
        }
        visited.insert(role_name.to_string());

        if let Some(role) = roles.get(role_name) {
            // Add direct permissions
            for permission in &role.permissions {
                permissions.insert(permission.clone());
            }

            // Add permissions from parent roles
            for parent_role in &role.parent_roles {
                self.collect_role_permissions(parent_role, roles, permissions, visited)?;
            }
        }

        Ok(())
    }

    /// Check if a permission allows a specific action on a resource
    async fn permission_allows_action(&self, permission_name: &str, resource: &str, action: &str) -> Result<bool> {
        let permissions = self.permissions.read().await;
        
        if let Some(permission) = permissions.get(permission_name) {
            // Check resource type
            if permission.resource_type != "*" && permission.resource_type != resource {
                return Ok(false);
            }

            // Check actions
            if permission.actions.contains(&"*".to_string()) || permission.actions.contains(&action.to_string()) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Evaluate conditions for a permission
    async fn evaluate_conditions(&self, conditions: &[Condition], context: &SecurityContext) -> Result<bool> {
        for condition in conditions {
            if !self.evaluate_condition(condition, context).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Evaluate a single condition
    async fn evaluate_condition(&self, condition: &Condition, context: &SecurityContext) -> Result<bool> {
        let field_value = match condition.condition_type {
            ConditionType::IpAddress => context.ip_address.clone().unwrap_or_default(),
            ConditionType::UserAttribute => context.metadata.get(&condition.field).cloned().unwrap_or_default(),
            ConditionType::ContextAttribute => context.metadata.get(&condition.field).cloned().unwrap_or_default(),
            _ => String::new(),
        };

        match condition.operator {
            ConditionOperator::Equals => Ok(field_value == condition.value),
            ConditionOperator::NotEquals => Ok(field_value != condition.value),
            ConditionOperator::Contains => Ok(field_value.contains(&condition.value)),
            ConditionOperator::StartsWith => Ok(field_value.starts_with(&condition.value)),
            ConditionOperator::EndsWith => Ok(field_value.ends_with(&condition.value)),
            ConditionOperator::In => {
                let values: Vec<&str> = condition.value.split(',').collect();
                Ok(values.contains(&field_value.as_str()))
            }
            ConditionOperator::NotIn => {
                let values: Vec<&str> = condition.value.split(',').collect();
                Ok(!values.contains(&field_value.as_str()))
            }
            _ => Ok(true), // Default to allow for unsupported operators
        }
    }
}

#[async_trait]
impl AuthorizationService for RbacAuthorizationService {
    async fn check_permission(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<bool> {
        // Get user roles
        let user_roles = self.user_roles.read().await;
        let roles = user_roles.get(&context.user_id).cloned().unwrap_or_default();
        drop(user_roles);

        // Get all permissions for user roles
        let mut all_permissions = HashSet::new();
        for role_name in &roles {
            let role_permissions = self.get_role_permissions(role_name).await?;
            all_permissions.extend(role_permissions);
        }

        // Check if any permission allows the action
        for permission_name in &all_permissions {
            if self.permission_allows_action(permission_name, resource, action).await? {
                // Check permission conditions
                let permissions = self.permissions.read().await;
                if let Some(permission) = permissions.get(permission_name) {
                    if self.evaluate_conditions(&permission.conditions, context).await? {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    async fn get_user_permissions(&self, user_id: &str) -> Result<Vec<Permission>> {
        let user_roles = self.user_roles.read().await;
        let roles = user_roles.get(user_id).cloned().unwrap_or_default();
        drop(user_roles);

        let mut all_permissions = HashSet::new();
        for role_name in &roles {
            let role_permissions = self.get_role_permissions(&role_name).await?;
            all_permissions.extend(role_permissions);
        }

        let permissions = self.permissions.read().await;
        let result: Vec<Permission> = all_permissions
            .into_iter()
            .filter_map(|name| permissions.get(&name).cloned())
            .collect();

        Ok(result)
    }

    async fn get_user_roles(&self, user_id: &str) -> Result<Vec<Role>> {
        let user_roles = self.user_roles.read().await;
        let role_names = user_roles.get(user_id).cloned().unwrap_or_default();
        drop(user_roles);

        let roles = self.roles.read().await;
        let result: Vec<Role> = role_names
            .into_iter()
            .filter_map(|name| roles.get(&name).cloned())
            .collect();

        Ok(result)
    }

    async fn add_user_role(&self, user_id: &str, role_name: &str) -> Result<()> {
        // Check if role exists
        if !self.role_exists(role_name).await? {
            return Err(AgentError::validation(format!("Role '{}' does not exist", role_name)));
        }

        let mut user_roles = self.user_roles.write().await;
        user_roles.entry(user_id.to_string()).or_insert_with(HashSet::new).insert(role_name.to_string());
        Ok(())
    }

    async fn remove_user_role(&self, user_id: &str, role_name: &str) -> Result<()> {
        let mut user_roles = self.user_roles.write().await;
        if let Some(roles) = user_roles.get_mut(user_id) {
            roles.remove(role_name);
        }
        Ok(())
    }

    async fn create_role(&self, role: Role) -> Result<()> {
        let mut roles = self.roles.write().await;
        if roles.contains_key(&role.name) {
            return Err(AgentError::validation(format!("Role '{}' already exists", role.name)));
        }
        roles.insert(role.name.clone(), role);
        Ok(())
    }

    async fn update_role(&self, role: Role) -> Result<()> {
        let mut roles = self.roles.write().await;
        roles.insert(role.name.clone(), role);
        Ok(())
    }

    async fn delete_role(&self, role_name: &str) -> Result<()> {
        let mut roles = self.roles.write().await;
        roles.remove(role_name);
        
        // Remove role from all users
        let mut user_roles = self.user_roles.write().await;
        for (_, user_role_set) in user_roles.iter_mut() {
            user_role_set.remove(role_name);
        }
        
        Ok(())
    }

    async fn create_permission(&self, permission: Permission) -> Result<()> {
        let mut permissions = self.permissions.write().await;
        if permissions.contains_key(&permission.name) {
            return Err(AgentError::validation(format!("Permission '{}' already exists", permission.name)));
        }
        permissions.insert(permission.name.clone(), permission);
        Ok(())
    }

    async fn update_permission(&self, permission: Permission) -> Result<()> {
        let mut permissions = self.permissions.write().await;
        permissions.insert(permission.name.clone(), permission);
        Ok(())
    }

    async fn delete_permission(&self, permission_name: &str) -> Result<()> {
        let mut permissions = self.permissions.write().await;
        permissions.remove(permission_name);
        Ok(())
    }

    async fn role_exists(&self, role_name: &str) -> Result<bool> {
        let roles = self.roles.read().await;
        Ok(roles.contains_key(role_name))
    }

    async fn permission_exists(&self, permission_name: &str) -> Result<bool> {
        let permissions = self.permissions.read().await;
        Ok(permissions.contains_key(permission_name))
    }

    async fn get_role_hierarchy(&self) -> Result<HashMap<String, Vec<String>>> {
        let roles = self.roles.read().await;
        let mut hierarchy = HashMap::new();
        
        for (role_name, role) in roles.iter() {
            hierarchy.insert(role_name.clone(), role.parent_roles.clone());
        }
        
        Ok(hierarchy)
    }

    async fn evaluate_policy(&self, context: &SecurityContext, policy: &Policy) -> Result<bool> {
        // Evaluate conditions
        self.evaluate_conditions(&policy.conditions, context).await
    }
}

/// Create an authorization service
pub async fn create_authorization_service() -> Result<Box<dyn AuthorizationService>> {
    let service = RbacAuthorizationService::new();
    service.initialize_defaults().await?;
    Ok(Box::new(service))
}
