//! Signature system for type-safe DSPy module definitions
//!
//! This module provides the core signature system that defines input and output
//! types for DSPy modules, enabling type-safe composition and validation.

use crate::dspy::error::{DspyError, DspyResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use tracing::debug;

/// Field definition for signature inputs and outputs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field description for documentation and prompt generation
    pub description: String,
    /// Field type information
    pub field_type: FieldType,
    /// Whether the field is required
    pub required: bool,
    /// Default value if field is optional
    pub default_value: Option<serde_json::Value>,
    /// Validation constraints
    pub constraints: Vec<FieldConstraint>,
}

/// Supported field types for DSPy signatures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldType {
    /// String type
    String,
    /// Integer number
    Integer,
    /// Floating point number
    Float,
    /// Boolean value
    Boolean,
    /// Array of another type
    Array(Box<FieldType>),
    /// Object with named fields
    Object(Vec<Field>),
    /// JSON value (any valid JSON)
    Json,
    /// Custom type with type name
    Custom(String),
}

/// Field validation constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldConstraint {
    /// Minimum length for strings or arrays
    MinLength(usize),
    /// Maximum length for strings or arrays
    MaxLength(usize),
    /// Minimum value for numbers
    MinValue(f64),
    /// Maximum value for numbers
    MaxValue(f64),
    /// Regular expression pattern for strings
    Pattern(String),
    /// Enumeration of allowed values
    Enum(Vec<serde_json::Value>),
    /// Custom validation function name
    Custom(String),
}

impl Field {
    /// Create a new field
    pub fn new<S: Into<String>>(name: S, description: S, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            field_type,
            required: true,
            default_value: None,
            constraints: Vec::new(),
        }
    }

    /// Make the field optional with a default value
    pub fn optional(mut self, default_value: Option<serde_json::Value>) -> Self {
        self.required = false;
        self.default_value = default_value;
        self
    }

    /// Add a validation constraint
    pub fn with_constraint(mut self, constraint: FieldConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Validate a value against this field's constraints
    pub fn validate_value(&self, value: &serde_json::Value) -> DspyResult<()> {
        // Check type compatibility
        if !self.is_type_compatible(value) {
            return Err(DspyError::type_validation(
                &self.name,
                &format!("Value type does not match field type {:?}", self.field_type),
            ));
        }

        // Apply constraints
        for constraint in &self.constraints {
            self.validate_constraint(value, constraint)?;
        }

        Ok(())
    }

    /// Check if a value is compatible with the field type
    fn is_type_compatible(&self, value: &serde_json::Value) -> bool {
        match (&self.field_type, value) {
            (FieldType::String, serde_json::Value::String(_)) => true,
            (FieldType::Integer, serde_json::Value::Number(n)) => n.is_i64(),
            (FieldType::Float, serde_json::Value::Number(_)) => true,
            (FieldType::Boolean, serde_json::Value::Bool(_)) => true,
            (FieldType::Array(_), serde_json::Value::Array(_)) => true,
            (FieldType::Object(_), serde_json::Value::Object(_)) => true,
            (FieldType::Json, _) => true,
            (FieldType::Custom(_), _) => true, // Custom types require external validation
            _ => false,
        }
    }

    /// Validate a specific constraint
    fn validate_constraint(
        &self,
        value: &serde_json::Value,
        constraint: &FieldConstraint,
    ) -> DspyResult<()> {
        match constraint {
            FieldConstraint::MinLength(min_len) => {
                let len = match value {
                    serde_json::Value::String(s) => s.len(),
                    serde_json::Value::Array(a) => a.len(),
                    _ => return Ok(()), // Skip for non-applicable types
                };
                if len < *min_len {
                    return Err(DspyError::type_validation(
                        &self.name,
                        &format!("Length {} is less than minimum {}", len, min_len),
                    ));
                }
            }
            FieldConstraint::MaxLength(max_len) => {
                let len = match value {
                    serde_json::Value::String(s) => s.len(),
                    serde_json::Value::Array(a) => a.len(),
                    _ => return Ok(()), // Skip for non-applicable types
                };
                if len > *max_len {
                    return Err(DspyError::type_validation(
                        &self.name,
                        &format!("Length {} exceeds maximum {}", len, max_len),
                    ));
                }
            }
            FieldConstraint::MinValue(min_val) => {
                if let serde_json::Value::Number(n) = value {
                    if let Some(val) = n.as_f64() {
                        if val < *min_val {
                            return Err(DspyError::type_validation(
                                &self.name,
                                &format!("Value {} is less than minimum {}", val, min_val),
                            ));
                        }
                    }
                }
            }
            FieldConstraint::MaxValue(max_val) => {
                if let serde_json::Value::Number(n) = value {
                    if let Some(val) = n.as_f64() {
                        if val > *max_val {
                            return Err(DspyError::type_validation(
                                &self.name,
                                &format!("Value {} exceeds maximum {}", val, max_val),
                            ));
                        }
                    }
                }
            }
            FieldConstraint::Pattern(pattern) => {
                if let serde_json::Value::String(s) = value {
                    let regex = regex::Regex::new(pattern).map_err(|e| {
                        DspyError::type_validation(
                            &self.name,
                            &format!("Invalid regex pattern: {}", e),
                        )
                    })?;
                    if !regex.is_match(s) {
                        return Err(DspyError::type_validation(
                            &self.name,
                            &format!("String '{}' does not match pattern '{}'", s, pattern),
                        ));
                    }
                }
            }
            FieldConstraint::Enum(allowed_values) => {
                if !allowed_values.contains(value) {
                    return Err(DspyError::type_validation(
                        &self.name,
                        &format!("Value {:?} is not in allowed enumeration", value),
                    ));
                }
            }
            FieldConstraint::Custom(_) => {
                // Custom constraints require external validation
                debug!(
                    "Skipping custom constraint validation for field '{}'",
                    self.name
                );
            }
        }
        Ok(())
    }
}

/// Type-safe signature definition for DSPy modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature<I, O> {
    /// Signature name for identification
    pub name: String,
    /// Signature description
    pub description: String,
    /// Input field definitions
    pub input_fields: Vec<Field>,
    /// Output field definitions
    pub output_fields: Vec<Field>,
    /// Signature metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Phantom data for type safety
    #[serde(skip)]
    _input_type: PhantomData<I>,
    #[serde(skip)]
    _output_type: PhantomData<O>,
}

impl<I, O> Signature<I, O>
where
    I: Serialize + for<'de> Deserialize<'de>,
    O: Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new signature
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            input_fields: Vec::new(),
            output_fields: Vec::new(),
            metadata: HashMap::new(),
            _input_type: PhantomData,
            _output_type: PhantomData,
        }
    }

    /// Set the signature description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }

    /// Add an input field
    pub fn with_input_field<S: Into<String>>(
        mut self,
        name: S,
        description: S,
        field_type: FieldType,
    ) -> Self {
        let field = Field::new(name, description, field_type);
        self.input_fields.push(field);
        self
    }

    /// Add an output field
    pub fn with_output_field<S: Into<String>>(
        mut self,
        name: S,
        description: S,
        field_type: FieldType,
    ) -> Self {
        let field = Field::new(name, description, field_type);
        self.output_fields.push(field);
        self
    }

    /// Add metadata
    pub fn with_metadata<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Validate input data against the signature
    pub fn validate_input(&self, input: &I) -> DspyResult<()> {
        let input_json = serde_json::to_value(input).map_err(|e| {
            DspyError::serialization(
                "input_validation",
                &format!("Failed to serialize input: {}", e),
            )
        })?;

        self.validate_fields(&input_json, &self.input_fields, "input")
    }

    /// Validate output data against the signature
    pub fn validate_output(&self, output: &O) -> DspyResult<()> {
        let output_json = serde_json::to_value(output).map_err(|e| {
            DspyError::serialization(
                "output_validation",
                &format!("Failed to serialize output: {}", e),
            )
        })?;

        self.validate_fields(&output_json, &self.output_fields, "output")
    }

    /// Validate fields against JSON data
    fn validate_fields(
        &self,
        data: &serde_json::Value,
        fields: &[Field],
        context: &str,
    ) -> DspyResult<()> {
        let obj = data.as_object().ok_or_else(|| {
            DspyError::type_validation(context, "Expected object for field validation")
        })?;

        for field in fields {
            if field.required && !obj.contains_key(&field.name) {
                return Err(DspyError::type_validation(
                    &field.name,
                    &format!("Required field '{}' is missing", field.name),
                ));
            }

            if let Some(value) = obj.get(&field.name) {
                field.validate_value(value)?;
            } else if field.required {
                return Err(DspyError::type_validation(
                    &field.name,
                    &format!("Required field '{}' is missing", field.name),
                ));
            }
        }

        Ok(())
    }

    /// Get field by name from input fields
    pub fn get_input_field(&self, name: &str) -> Option<&Field> {
        self.input_fields.iter().find(|f| f.name == name)
    }

    /// Get field by name from output fields
    pub fn get_output_field(&self, name: &str) -> Option<&Field> {
        self.output_fields.iter().find(|f| f.name == name)
    }

    /// Get all field names for inputs
    pub fn input_field_names(&self) -> Vec<&str> {
        self.input_fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get all field names for outputs
    pub fn output_field_names(&self) -> Vec<&str> {
        self.output_fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Check if signature is compatible with another signature
    pub fn is_compatible_with<I2, O2>(&self, other: &Signature<I2, O2>) -> bool {
        // Check if output fields of this signature match input fields of other
        if self.output_fields.len() != other.input_fields.len() {
            return false;
        }

        for (output_field, input_field) in self.output_fields.iter().zip(other.input_fields.iter())
        {
            if output_field.field_type != input_field.field_type {
                return false;
            }
        }

        true
    }

    /// Generate a prompt template from the signature
    pub fn generate_prompt_template(&self) -> String {
        let mut template = String::new();

        if !self.description.is_empty() {
            template.push_str(&format!("Task: {}\n\n", self.description));
        }

        if !self.input_fields.is_empty() {
            template.push_str("Input:\n");
            for field in &self.input_fields {
                template.push_str(&format!("- {}: {}\n", field.name, field.description));
            }
            template.push('\n');
        }

        if !self.output_fields.is_empty() {
            template.push_str("Output:\n");
            for field in &self.output_fields {
                template.push_str(&format!("- {}: {}\n", field.name, field.description));
            }
        }

        template
    }
}

impl<I, O> fmt::Display for Signature<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signature[{}]", self.name)
    }
}

/// Builder for creating signatures with a fluent interface
pub struct SignatureBuilder<I, O> {
    signature: Signature<I, O>,
}

impl<I, O> SignatureBuilder<I, O>
where
    I: Serialize + for<'de> Deserialize<'de>,
    O: Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new signature builder
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            signature: Signature::new(name),
        }
    }

    /// Set description
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.signature = self.signature.with_description(description);
        self
    }

    /// Add input field
    pub fn input_field<S: Into<String>>(
        mut self,
        name: S,
        description: S,
        field_type: FieldType,
    ) -> Self {
        self.signature = self
            .signature
            .with_input_field(name, description, field_type);
        self
    }

    /// Add output field
    pub fn output_field<S: Into<String>>(
        mut self,
        name: S,
        description: S,
        field_type: FieldType,
    ) -> Self {
        self.signature = self
            .signature
            .with_output_field(name, description, field_type);
        self
    }

    /// Add metadata
    pub fn metadata<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.signature = self.signature.with_metadata(key, value);
        self
    }

    /// Build the signature
    pub fn build(self) -> Signature<I, O> {
        self.signature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Serialize, Deserialize)]
    struct TestInput {
        question: String,
        context: Option<String>,
    }

    #[derive(Serialize, Deserialize)]
    struct TestOutput {
        answer: String,
        confidence: f64,
    }

    #[test]
    fn test_field_creation() {
        let field = Field::new("test_field", "A test field", FieldType::String)
            .optional(Some(json!("default")))
            .with_constraint(FieldConstraint::MinLength(1));

        assert_eq!(field.name, "test_field");
        assert_eq!(field.description, "A test field");
        assert_eq!(field.field_type, FieldType::String);
        assert!(!field.required);
        assert_eq!(field.default_value, Some(json!("default")));
        assert_eq!(field.constraints.len(), 1);
    }

    #[test]
    fn test_field_validation() {
        let field = Field::new("test", "Test field", FieldType::String)
            .with_constraint(FieldConstraint::MinLength(3))
            .with_constraint(FieldConstraint::MaxLength(10));

        assert!(field.validate_value(&json!("hello")).is_ok());
        assert!(field.validate_value(&json!("hi")).is_err()); // Too short
        assert!(field.validate_value(&json!("this is too long")).is_err()); // Too long
        assert!(field.validate_value(&json!(123)).is_err()); // Wrong type
    }

    #[test]
    fn test_signature_creation() {
        let signature = SignatureBuilder::<TestInput, TestOutput>::new("test_qa")
            .description("A test Q&A signature")
            .input_field("question", "The question to answer", FieldType::String)
            .input_field("context", "Optional context", FieldType::String)
            .output_field("answer", "The answer", FieldType::String)
            .output_field("confidence", "Confidence score", FieldType::Float)
            .metadata("version", "1.0")
            .build();

        assert_eq!(signature.name, "test_qa");
        assert_eq!(signature.description, "A test Q&A signature");
        assert_eq!(signature.input_fields.len(), 2);
        assert_eq!(signature.output_fields.len(), 2);
        assert_eq!(signature.metadata.get("version"), Some(&json!("1.0")));
    }

    #[test]
    fn test_signature_validation() {
        let signature = SignatureBuilder::<TestInput, TestOutput>::new("test")
            .input_field("question", "Question", FieldType::String)
            .output_field("answer", "Answer", FieldType::String)
            .output_field("confidence", "Confidence", FieldType::Float)
            .build();

        let valid_input = TestInput {
            question: "What is 2+2?".to_string(),
            context: None,
        };

        let valid_output = TestOutput {
            answer: "4".to_string(),
            confidence: 0.95,
        };

        assert!(signature.validate_input(&valid_input).is_ok());
        assert!(signature.validate_output(&valid_output).is_ok());
    }

    #[test]
    fn test_prompt_template_generation() {
        let signature = SignatureBuilder::<TestInput, TestOutput>::new("qa_task")
            .description("Answer questions based on context")
            .input_field("question", "The question to answer", FieldType::String)
            .input_field("context", "Relevant context", FieldType::String)
            .output_field("answer", "The final answer", FieldType::String)
            .build();

        let template = signature.generate_prompt_template();

        assert!(template.contains("Task: Answer questions based on context"));
        assert!(template.contains("Input:"));
        assert!(template.contains("- question: The question to answer"));
        assert!(template.contains("- context: Relevant context"));
        assert!(template.contains("Output:"));
        assert!(template.contains("- answer: The final answer"));
    }
}
