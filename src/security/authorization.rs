use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

use super::SecurityContext;
use crate::utils::error::{AgentError, Result};

/// Authorization service trait
#[async_trait]
pub trait AuthorizationService: Send + Sync {
    /// Check if a user has permission to perform an action on a resource
    async fn check_permission(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &str,
    ) -> Result<bool>;

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

    /// Get detailed authorization decision with policy evaluation results
    async fn get_authorization_decision(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<AuthorizationDecision> {
        // Default implementation for backward compatibility
        let granted = self.check_permission(context, resource, action).await?;
        Ok(AuthorizationDecision {
            granted,
            reason: if granted { "Access granted".to_string() } else { "Access denied".to_string() },
            evaluated_policies: Vec::new(),
            checked_permissions: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    /// Downcast to concrete type for accessing implementation-specific methods
    fn as_any(&self) -> &dyn std::any::Any;
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

impl Default for RbacAuthorizationService {
    fn default() -> Self {
        Self::new()
    }
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
                actions: vec![
                    "write".to_string(),
                    "create".to_string(),
                    "update".to_string(),
                ],
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
                permissions: vec![
                    "read".to_string(),
                    "write".to_string(),
                    "delete".to_string(),
                    "admin".to_string(),
                ],
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
    async fn permission_allows_action(
        &self,
        permission_name: &str,
        resource: &str,
        action: &str,
    ) -> Result<bool> {
        let permissions = self.permissions.read().await;

        if let Some(permission) = permissions.get(permission_name) {
            // Check resource type
            if permission.resource_type != "*" && permission.resource_type != resource {
                return Ok(false);
            }

            // Check actions
            if permission.actions.contains(&"*".to_string())
                || permission.actions.contains(&action.to_string())
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Evaluate conditions for a permission
    async fn evaluate_conditions(
        &self,
        conditions: &[Condition],
        context: &SecurityContext,
    ) -> Result<bool> {
        for condition in conditions {
            if !self.evaluate_condition(condition, context).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Evaluate a single condition
    async fn evaluate_condition(
        &self,
        condition: &Condition,
        context: &SecurityContext,
    ) -> Result<bool> {
        let field_value = match condition.condition_type {
            ConditionType::IpAddress => context.ip_address.clone().unwrap_or_default(),
            ConditionType::UserAttribute => context
                .metadata
                .get(&condition.field)
                .cloned()
                .unwrap_or_default(),
            ConditionType::ContextAttribute => context
                .metadata
                .get(&condition.field)
                .cloned()
                .unwrap_or_default(),
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
    async fn check_permission(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &str,
    ) -> Result<bool> {
        // Use policy-based evaluation first, with RBAC fallback
        let decision = self.evaluate_policies(context, resource, action).await?;
        Ok(decision.granted)
    }

    async fn get_user_permissions(&self, user_id: &str) -> Result<Vec<Permission>> {
        let user_roles = self.user_roles.read().await;
        let roles = user_roles.get(user_id).cloned().unwrap_or_default();
        drop(user_roles);

        let mut all_permissions = HashSet::new();
        for role_name in &roles {
            let role_permissions = self.get_role_permissions(role_name).await?;
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
            return Err(AgentError::validation(format!(
                "Role '{}' does not exist",
                role_name
            )));
        }

        let mut user_roles = self.user_roles.write().await;
        user_roles
            .entry(user_id.to_string())
            .or_insert_with(HashSet::new)
            .insert(role_name.to_string());
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
            return Err(AgentError::validation(format!(
                "Role '{}' already exists",
                role.name
            )));
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
            return Err(AgentError::validation(format!(
                "Permission '{}' already exists",
                permission.name
            )));
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

    async fn get_authorization_decision(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<AuthorizationDecision> {
        // Use the comprehensive policy evaluation
        self.evaluate_policies(context, resource, action).await
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RbacAuthorizationService {
    /// Create a new policy
    pub async fn create_policy(&self, policy: Policy) -> Result<()> {
        let mut policies = self.policies.write().await;
        if policies.contains_key(&policy.name) {
            return Err(AgentError::validation(format!(
                "Policy '{}' already exists",
                policy.name
            )));
        }
        policies.insert(policy.name.clone(), policy);
        Ok(())
    }

    /// Update an existing policy
    pub async fn update_policy(&self, policy: Policy) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.insert(policy.name.clone(), policy);
        Ok(())
    }

    /// Delete a policy
    pub async fn delete_policy(&self, policy_name: &str) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.remove(policy_name);
        Ok(())
    }

    /// Check if a policy exists
    pub async fn policy_exists(&self, policy_name: &str) -> Result<bool> {
        let policies = self.policies.read().await;
        Ok(policies.contains_key(policy_name))
    }

    /// Get all policies
    pub async fn get_all_policies(&self) -> Result<Vec<Policy>> {
        let policies = self.policies.read().await;
        Ok(policies.values().cloned().collect())
    }

    /// Get policies that apply to a specific resource and action
    pub async fn get_applicable_policies(&self, resource: &str, action: &str) -> Result<Vec<Policy>> {
        let policies = self.policies.read().await;
        let mut applicable = Vec::new();

        for policy in policies.values() {
            // Check if policy applies to this resource
            let resource_matches = policy.resources.iter().any(|r| {
                r == "*" || r == resource || resource.starts_with(&format!("{}:", r))
            });

            // Check if policy applies to this action
            let action_matches = policy.actions.iter().any(|a| {
                a == "*" || a == action
            });

            if resource_matches && action_matches {
                applicable.push(policy.clone());
            }
        }

        // Sort by priority (higher priority first)
        applicable.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(applicable)
    }

    /// Evaluate all applicable policies for a request
    pub async fn evaluate_policies(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<AuthorizationDecision> {
        let applicable_policies = self.get_applicable_policies(resource, action).await?;
        let mut evaluated_policies = Vec::new();
        let mut final_decision = false;
        let mut decision_reason = "No applicable policies found".to_string();
        let mut metadata = HashMap::new();

        // Track policy evaluation statistics
        metadata.insert("total_policies".to_string(), applicable_policies.len().to_string());
        metadata.insert("resource".to_string(), resource.to_string());
        metadata.insert("action".to_string(), action.to_string());
        metadata.insert("user_id".to_string(), context.user_id.clone());

        for policy in &applicable_policies {
            evaluated_policies.push(policy.name.clone());

            // Evaluate policy conditions
            let policy_result = self.evaluate_policy(context, policy).await?;

            if policy_result {
                match policy.effect {
                    PolicyEffect::Allow => {
                        final_decision = true;
                        decision_reason = format!("Allowed by policy: {}", policy.name);
                        metadata.insert("deciding_policy".to_string(), policy.name.clone());
                        metadata.insert("policy_effect".to_string(), "Allow".to_string());
                        break; // First allow policy wins
                    }
                    PolicyEffect::Deny => {
                        final_decision = false;
                        decision_reason = format!("Denied by policy: {}", policy.name);
                        metadata.insert("deciding_policy".to_string(), policy.name.clone());
                        metadata.insert("policy_effect".to_string(), "Deny".to_string());
                        break; // First deny policy wins (higher priority)
                    }
                }
            }
        }

        // If no policies matched, fall back to RBAC
        if evaluated_policies.is_empty() || (!final_decision && decision_reason.contains("No applicable policies")) {
            let rbac_result = self.check_permission_rbac_only(context, resource, action).await?;
            if rbac_result {
                final_decision = true;
                decision_reason = "Allowed by RBAC permissions".to_string();
                metadata.insert("fallback_to_rbac".to_string(), "true".to_string());
            } else {
                decision_reason = "Denied by RBAC permissions".to_string();
                metadata.insert("fallback_to_rbac".to_string(), "true".to_string());
            }
        }

        Ok(AuthorizationDecision {
            granted: final_decision,
            reason: decision_reason,
            evaluated_policies,
            checked_permissions: Vec::new(), // Will be populated by RBAC fallback if needed
            metadata,
        })
    }

    /// Check permission using only RBAC (without policy evaluation)
    async fn check_permission_rbac_only(&self, context: &SecurityContext, resource: &str, action: &str) -> Result<bool> {
        // Get user roles
        let user_roles = self.user_roles.read().await;
        let roles = user_roles
            .get(&context.user_id)
            .cloned()
            .unwrap_or_default();
        drop(user_roles);

        // Get all permissions for user roles
        let mut all_permissions = HashSet::new();
        for role_name in &roles {
            let role_permissions = self.get_role_permissions(role_name).await?;
            all_permissions.extend(role_permissions);
        }

        // Check if any permission allows the action
        for permission_name in &all_permissions {
            if self
                .permission_allows_action(permission_name, resource, action)
                .await?
            {
                // Check permission conditions
                let permissions = self.permissions.read().await;
                if let Some(permission) = permissions.get(permission_name) {
                    if self
                        .evaluate_conditions(&permission.conditions, context)
                        .await?
                    {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Initialize default security policies
    pub async fn initialize_default_policies(&self) -> Result<()> {
        let default_policies = vec![
            Policy {
                name: "admin_full_access".to_string(),
                description: "Administrators have full access to all resources".to_string(),
                effect: PolicyEffect::Allow,
                resources: vec!["*".to_string()],
                actions: vec!["*".to_string()],
                conditions: vec![Condition {
                    field: "user.roles".to_string(),
                    condition_type: ConditionType::UserAttribute,
                    operator: ConditionOperator::Contains,
                    value: "admin".to_string(),
                }],
                priority: 1000,
            },
            Policy {
                name: "business_hours_access".to_string(),
                description: "Allow access only during business hours (9 AM - 5 PM)".to_string(),
                effect: PolicyEffect::Deny,
                resources: vec!["sensitive".to_string(), "financial".to_string(), "admin".to_string()],
                actions: vec!["*".to_string()],
                conditions: vec![Condition {
                    field: "time.hour".to_string(),
                    condition_type: ConditionType::ContextAttribute,
                    operator: ConditionOperator::NotIn,
                    value: "9,10,11,12,13,14,15,16,17".to_string(),
                }],
                priority: 800,
            },
            Policy {
                name: "ip_whitelist".to_string(),
                description: "Deny access from non-whitelisted IP addresses".to_string(),
                effect: PolicyEffect::Deny,
                resources: vec!["*".to_string()],
                actions: vec!["*".to_string()],
                conditions: vec![Condition {
                    field: "ip_address".to_string(),
                    condition_type: ConditionType::IpAddress,
                    operator: ConditionOperator::NotIn,
                    value: "192.168.1.0/24,10.0.0.0/8,172.16.0.0/12".to_string(),
                }],
                priority: 900,
            },
            Policy {
                name: "read_only_guest".to_string(),
                description: "Guest users can only read public resources".to_string(),
                effect: PolicyEffect::Allow,
                resources: vec!["public".to_string()],
                actions: vec!["read".to_string(), "view".to_string()],
                conditions: vec![Condition {
                    field: "user.roles".to_string(),
                    condition_type: ConditionType::UserAttribute,
                    operator: ConditionOperator::Contains,
                    value: "guest".to_string(),
                }],
                priority: 100,
            },
            Policy {
                name: "sensitive_data_protection".to_string(),
                description: "Require special permission for sensitive data access".to_string(),
                effect: PolicyEffect::Deny,
                resources: vec!["sensitive".to_string(), "pii".to_string(), "financial".to_string()],
                actions: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
                conditions: vec![Condition {
                    field: "user.clearance_level".to_string(),
                    condition_type: ConditionType::UserAttribute,
                    operator: ConditionOperator::NotEquals,
                    value: "high".to_string(),
                }],
                priority: 950,
            },
        ];

        for policy in default_policies {
            if !self.policy_exists(&policy.name).await? {
                self.create_policy(policy).await?;
            }
        }

        Ok(())
    }
}

/// Create an authorization service
pub async fn create_authorization_service() -> Result<Box<dyn AuthorizationService>> {
    let service = RbacAuthorizationService::new();
    service.initialize_defaults().await?;
    service.initialize_default_policies().await?;
    Ok(Box::new(service))
}
