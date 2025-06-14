use async_trait::async_trait;
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

use super::SecurityContext;
use crate::utils::error::{AgentError, Result};

/// Policy engine trait
#[async_trait]
pub trait PolicyEngine: Send + Sync {
    /// Evaluate a policy against a context
    async fn evaluate_policy(
        &self,
        policy_name: &str,
        context: &SecurityContext,
    ) -> Result<PolicyDecision>;

    /// Evaluate all applicable policies
    async fn evaluate_all_policies(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &str,
    ) -> Result<PolicyDecision>;

    /// Add a policy
    async fn add_policy(&self, policy: SecurityPolicy) -> Result<()>;

    /// Update a policy
    async fn update_policy(&self, policy: SecurityPolicy) -> Result<()>;

    /// Remove a policy
    async fn remove_policy(&self, policy_name: &str) -> Result<()>;

    /// Get a policy
    async fn get_policy(&self, policy_name: &str) -> Result<Option<SecurityPolicy>>;

    /// List all policies
    async fn list_policies(&self) -> Result<Vec<SecurityPolicy>>;

    /// Check if a policy exists
    async fn policy_exists(&self, policy_name: &str) -> Result<bool>;

    /// Validate policy syntax
    async fn validate_policy(&self, policy: &SecurityPolicy) -> Result<PolicyValidationResult>;

    /// Get policy evaluation statistics
    async fn get_evaluation_statistics(&self) -> Result<PolicyStatistics>;
}

/// Security policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy version
    pub version: String,
    /// Policy effect (allow or deny)
    pub effect: PolicyEffect,
    /// Resources this policy applies to
    pub resources: Vec<String>,
    /// Actions this policy applies to
    pub actions: Vec<String>,
    /// Conditions for this policy
    pub conditions: Vec<PolicyCondition>,
    /// Policy priority (higher number = higher priority)
    pub priority: i32,
    /// Whether policy is enabled
    pub enabled: bool,
    /// Policy metadata
    pub metadata: HashMap<String, String>,
}

/// Policy effect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PolicyEffect {
    /// Allow access
    Allow,
    /// Deny access
    Deny,
}

/// Policy condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    /// Field to evaluate
    pub field: String,
    /// Operator for comparison
    pub operator: PolicyOperator,
    /// Value to compare against
    pub value: PolicyValue,
    /// Whether to negate the condition
    pub negate: bool,
}

/// Policy operators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PolicyOperator {
    /// Equal to
    Equals,
    /// Not equal to
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Greater than or equal to
    GreaterThanOrEqual,
    /// Less than or equal to
    LessThanOrEqual,
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
    /// IP address in CIDR range
    IpInRange,
    /// Time within range
    TimeInRange,
}

/// Policy value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyValue {
    /// String value
    String(String),
    /// Number value
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// List of strings
    StringList(Vec<String>),
    /// List of numbers
    NumberList(Vec<f64>),
    /// Time range
    TimeRange { start: String, end: String },
    /// IP CIDR range
    IpRange(String),
}

/// Policy decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    /// Whether access is granted
    pub granted: bool,
    /// Decision reason
    pub reason: String,
    /// Policies that were evaluated
    pub evaluated_policies: Vec<String>,
    /// Policy that made the final decision
    pub deciding_policy: Option<String>,
    /// Decision metadata
    pub metadata: HashMap<String, String>,
}

/// Policy validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyValidationResult {
    /// Whether policy is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

/// Policy evaluation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyStatistics {
    /// Total policy evaluations
    pub total_evaluations: u64,
    /// Evaluations by policy
    pub evaluations_by_policy: HashMap<String, u64>,
    /// Allow decisions
    pub allow_decisions: u64,
    /// Deny decisions
    pub deny_decisions: u64,
    /// Average evaluation time in microseconds
    pub avg_evaluation_time_us: f64,
    /// Policy cache hit rate
    pub cache_hit_rate: f64,
}

/// Simple policy engine implementation
pub struct SimplePolicyEngine {
    /// Stored policies
    policies: RwLock<HashMap<String, SecurityPolicy>>,
    /// Evaluation statistics
    statistics: RwLock<PolicyStatistics>,
}

impl Default for SimplePolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplePolicyEngine {
    /// Create a new simple policy engine
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
            statistics: RwLock::new(PolicyStatistics {
                total_evaluations: 0,
                evaluations_by_policy: HashMap::new(),
                allow_decisions: 0,
                deny_decisions: 0,
                avg_evaluation_time_us: 0.0,
                cache_hit_rate: 0.0,
            }),
        }
    }

    /// Initialize with default policies
    pub async fn initialize_defaults(&self) -> Result<()> {
        let default_policies = vec![
            SecurityPolicy {
                name: "admin_full_access".to_string(),
                description: "Administrators have full access to all resources".to_string(),
                version: "1.0".to_string(),
                effect: PolicyEffect::Allow,
                resources: vec!["*".to_string()],
                actions: vec!["*".to_string()],
                conditions: vec![PolicyCondition {
                    field: "user.roles".to_string(),
                    operator: PolicyOperator::Contains,
                    value: PolicyValue::String("admin".to_string()),
                    negate: false,
                }],
                priority: 100,
                enabled: true,
                metadata: HashMap::new(),
            },
            SecurityPolicy {
                name: "deny_suspended_users".to_string(),
                description: "Deny access to suspended users".to_string(),
                version: "1.0".to_string(),
                effect: PolicyEffect::Deny,
                resources: vec!["*".to_string()],
                actions: vec!["*".to_string()],
                conditions: vec![PolicyCondition {
                    field: "user.status".to_string(),
                    operator: PolicyOperator::Equals,
                    value: PolicyValue::String("suspended".to_string()),
                    negate: false,
                }],
                priority: 200,
                enabled: true,
                metadata: HashMap::new(),
            },
            SecurityPolicy {
                name: "business_hours_only".to_string(),
                description: "Allow access only during business hours".to_string(),
                version: "1.0".to_string(),
                effect: PolicyEffect::Deny,
                resources: vec!["sensitive/*".to_string()],
                actions: vec!["read".to_string(), "write".to_string()],
                conditions: vec![PolicyCondition {
                    field: "time.hour".to_string(),
                    operator: PolicyOperator::TimeInRange,
                    value: PolicyValue::TimeRange {
                        start: "09:00".to_string(),
                        end: "17:00".to_string(),
                    },
                    negate: true,
                }],
                priority: 50,
                enabled: false, // Disabled by default
                metadata: HashMap::new(),
            },
        ];

        for policy in default_policies {
            self.add_policy(policy).await?;
        }

        Ok(())
    }

    /// Evaluate a single condition
    async fn evaluate_condition(
        &self,
        condition: &PolicyCondition,
        context: &SecurityContext,
    ) -> Result<bool> {
        let field_value = self.extract_field_value(&condition.field, context)?;

        let result = match condition.operator {
            PolicyOperator::Equals => self.compare_equals(&field_value, &condition.value),
            PolicyOperator::NotEquals => !self.compare_equals(&field_value, &condition.value),
            PolicyOperator::Contains => self.compare_contains(&field_value, &condition.value),
            PolicyOperator::StartsWith => self.compare_starts_with(&field_value, &condition.value),
            PolicyOperator::EndsWith => self.compare_ends_with(&field_value, &condition.value),
            PolicyOperator::In => self.compare_in(&field_value, &condition.value),
            PolicyOperator::NotIn => !self.compare_in(&field_value, &condition.value),
            PolicyOperator::GreaterThan => {
                self.compare_greater_than(&field_value, &condition.value)
            }
            PolicyOperator::LessThan => self.compare_less_than(&field_value, &condition.value),
            _ => false, // Unsupported operators default to false
        };

        Ok(if condition.negate { !result } else { result })
    }

    /// Extract field value from context
    fn extract_field_value(&self, field: &str, context: &SecurityContext) -> Result<String> {
        match field {
            "user.id" => Ok(context.user_id.clone()),
            "user.roles" => Ok(context.roles.join(",")),
            "user.permissions" => Ok(context.permissions.join(",")),
            "session.id" => Ok(context.session_id.clone()),
            "request.ip" => Ok(context.ip_address.clone().unwrap_or_default()),
            "request.user_agent" => Ok(context.user_agent.clone().unwrap_or_default()),
            "time.hour" => {
                let now = chrono::Utc::now();
                Ok(now.hour().to_string())
            }
            "time.day_of_week" => {
                let now = chrono::Utc::now();
                Ok(now.weekday().to_string())
            }
            _ => {
                // Check metadata
                if let Some(value) = context.metadata.get(field) {
                    Ok(value.clone())
                } else {
                    Ok(String::new())
                }
            }
        }
    }

    /// Compare values for equality
    fn compare_equals(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::String(s) => field_value == s,
            PolicyValue::Number(n) => field_value.parse::<f64>() == Ok(*n),
            PolicyValue::Boolean(b) => field_value.parse::<bool>() == Ok(*b),
            _ => false,
        }
    }

    /// Compare if field contains value
    fn compare_contains(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::String(s) => field_value.contains(s),
            _ => false,
        }
    }

    /// Compare if field starts with value
    fn compare_starts_with(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::String(s) => field_value.starts_with(s),
            _ => false,
        }
    }

    /// Compare if field ends with value
    fn compare_ends_with(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::String(s) => field_value.ends_with(s),
            _ => false,
        }
    }

    /// Compare if field is in list
    fn compare_in(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::StringList(list) => list.contains(&field_value.to_string()),
            PolicyValue::NumberList(list) => {
                if let Ok(num) = field_value.parse::<f64>() {
                    list.contains(&num)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Compare if field is greater than value
    fn compare_greater_than(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::Number(n) => field_value.parse::<f64>().is_ok_and(|v| v > *n),
            _ => false,
        }
    }

    /// Compare if field is less than value
    fn compare_less_than(&self, field_value: &str, policy_value: &PolicyValue) -> bool {
        match policy_value {
            PolicyValue::Number(n) => field_value.parse::<f64>().is_ok_and(|v| v < *n),
            _ => false,
        }
    }

    /// Check if resource matches pattern
    fn resource_matches(&self, resource: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if let Some(prefix) = pattern.strip_suffix("*") {
            return resource.starts_with(prefix);
        }

        resource == pattern
    }

    /// Check if action matches pattern
    fn action_matches(&self, action: &str, pattern: &str) -> bool {
        pattern == "*" || action == pattern
    }

    /// Update evaluation statistics
    async fn update_statistics(&self, policy_name: &str, granted: bool, evaluation_time_us: u64) {
        let mut stats = self.statistics.write().await;
        stats.total_evaluations += 1;
        *stats
            .evaluations_by_policy
            .entry(policy_name.to_string())
            .or_insert(0) += 1;

        if granted {
            stats.allow_decisions += 1;
        } else {
            stats.deny_decisions += 1;
        }

        // Update average evaluation time
        let total_time = stats.avg_evaluation_time_us * (stats.total_evaluations - 1) as f64;
        stats.avg_evaluation_time_us =
            (total_time + evaluation_time_us as f64) / stats.total_evaluations as f64;
    }
}

#[async_trait]
impl PolicyEngine for SimplePolicyEngine {
    async fn evaluate_policy(
        &self,
        policy_name: &str,
        context: &SecurityContext,
    ) -> Result<PolicyDecision> {
        let start_time = std::time::Instant::now();

        let policies = self.policies.read().await;
        let policy = policies
            .get(policy_name)
            .ok_or_else(|| AgentError::validation(format!("Policy '{}' not found", policy_name)))?;

        if !policy.enabled {
            return Ok(PolicyDecision {
                granted: false,
                reason: "Policy is disabled".to_string(),
                evaluated_policies: vec![policy_name.to_string()],
                deciding_policy: Some(policy_name.to_string()),
                metadata: HashMap::new(),
            });
        }

        // Evaluate all conditions
        let mut all_conditions_met = true;
        for condition in &policy.conditions {
            if !self.evaluate_condition(condition, context).await? {
                all_conditions_met = false;
                break;
            }
        }

        let granted = if all_conditions_met {
            policy.effect == PolicyEffect::Allow
        } else {
            false
        };

        let evaluation_time = start_time.elapsed().as_micros() as u64;
        drop(policies);

        self.update_statistics(policy_name, granted, evaluation_time)
            .await;

        Ok(PolicyDecision {
            granted,
            reason: if granted {
                "Policy allows access".to_string()
            } else {
                "Policy denies access".to_string()
            },
            evaluated_policies: vec![policy_name.to_string()],
            deciding_policy: Some(policy_name.to_string()),
            metadata: HashMap::new(),
        })
    }

    async fn evaluate_all_policies(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &str,
    ) -> Result<PolicyDecision> {
        let policies = self.policies.read().await;
        let mut applicable_policies: Vec<&SecurityPolicy> = policies
            .values()
            .filter(|p| p.enabled)
            .filter(|p| {
                p.resources
                    .iter()
                    .any(|r| self.resource_matches(resource, r))
            })
            .filter(|p| p.actions.iter().any(|a| self.action_matches(action, a)))
            .collect();

        // Sort by priority (higher priority first)
        applicable_policies.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut evaluated_policies = Vec::new();
        let mut final_decision = PolicyDecision {
            granted: false, // Default deny
            reason: "No applicable policies found".to_string(),
            evaluated_policies: Vec::new(),
            deciding_policy: None,
            metadata: HashMap::new(),
        };

        for policy in applicable_policies {
            evaluated_policies.push(policy.name.clone());

            // Evaluate all conditions for this policy
            let mut all_conditions_met = true;
            for condition in &policy.conditions {
                if !self.evaluate_condition(condition, context).await? {
                    all_conditions_met = false;
                    break;
                }
            }

            if all_conditions_met {
                final_decision = PolicyDecision {
                    granted: policy.effect == PolicyEffect::Allow,
                    reason: format!(
                        "Policy '{}' {} access",
                        policy.name,
                        if policy.effect == PolicyEffect::Allow {
                            "allows"
                        } else {
                            "denies"
                        }
                    ),
                    evaluated_policies: evaluated_policies.clone(),
                    deciding_policy: Some(policy.name.clone()),
                    metadata: HashMap::new(),
                };

                // If this is a deny policy, stop evaluation (deny takes precedence)
                if policy.effect == PolicyEffect::Deny {
                    break;
                }
            }
        }

        final_decision.evaluated_policies = evaluated_policies;
        Ok(final_decision)
    }

    async fn add_policy(&self, policy: SecurityPolicy) -> Result<()> {
        let validation = self.validate_policy(&policy).await?;
        if !validation.valid {
            return Err(AgentError::validation(format!(
                "Policy validation failed: {}",
                validation.errors.join(", ")
            )));
        }

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

    async fn update_policy(&self, policy: SecurityPolicy) -> Result<()> {
        let validation = self.validate_policy(&policy).await?;
        if !validation.valid {
            return Err(AgentError::validation(format!(
                "Policy validation failed: {}",
                validation.errors.join(", ")
            )));
        }

        let mut policies = self.policies.write().await;
        policies.insert(policy.name.clone(), policy);
        Ok(())
    }

    async fn remove_policy(&self, policy_name: &str) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.remove(policy_name);
        Ok(())
    }

    async fn get_policy(&self, policy_name: &str) -> Result<Option<SecurityPolicy>> {
        let policies = self.policies.read().await;
        Ok(policies.get(policy_name).cloned())
    }

    async fn list_policies(&self) -> Result<Vec<SecurityPolicy>> {
        let policies = self.policies.read().await;
        Ok(policies.values().cloned().collect())
    }

    async fn policy_exists(&self, policy_name: &str) -> Result<bool> {
        let policies = self.policies.read().await;
        Ok(policies.contains_key(policy_name))
    }

    async fn validate_policy(&self, policy: &SecurityPolicy) -> Result<PolicyValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate policy name
        if policy.name.is_empty() {
            errors.push("Policy name cannot be empty".to_string());
        }

        // Validate resources
        if policy.resources.is_empty() {
            errors.push("Policy must specify at least one resource".to_string());
        }

        // Validate actions
        if policy.actions.is_empty() {
            errors.push("Policy must specify at least one action".to_string());
        }

        // Validate conditions
        for (i, condition) in policy.conditions.iter().enumerate() {
            if condition.field.is_empty() {
                errors.push(format!("Condition {} has empty field", i));
            }
        }

        // Validate priority
        if policy.priority < 0 {
            warnings.push("Negative priority values may cause unexpected behavior".to_string());
        }

        Ok(PolicyValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    async fn get_evaluation_statistics(&self) -> Result<PolicyStatistics> {
        let stats = self.statistics.read().await;
        Ok(stats.clone())
    }
}

/// Create a policy engine
pub async fn create_policy_engine() -> Result<Box<dyn PolicyEngine>> {
    let engine = SimplePolicyEngine::new();
    engine.initialize_defaults().await?;
    Ok(Box::new(engine))
}
