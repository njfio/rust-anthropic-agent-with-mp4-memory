//! Comprehensive tests for DSPy advanced reasoning patterns
//!
//! This test suite validates the advanced reasoning capabilities including
//! tree-of-thought, graph-based reasoning, analogical reasoning, causal inference,
//! and meta-cognitive reasoning.

use rust_memvid_agent::anthropic::AnthropicClient;
use rust_memvid_agent::dspy::*;
use rust_memvid_agent::dspy::reasoning::*;
use rust_memvid_agent::dspy::causal_reasoning::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

fn create_test_client() -> Arc<AnthropicClient> {
    Arc::new(AnthropicClient::new(rust_memvid_agent::config::AnthropicConfig {
        api_key: "test_key".to_string(),
        model: "claude-3-sonnet-20240229".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        timeout_seconds: 30,
        max_retries: 3,
    }).unwrap())
}

fn create_advanced_reasoning_input() -> AdvancedReasoningInput {
    AdvancedReasoningInput {
        problem: "How can we solve the climate crisis while maintaining economic growth?".to_string(),
        context: Some("Global environmental and economic challenge".to_string()),
        constraints: vec![
            "Must be economically viable".to_string(),
            "Must reduce emissions by 50% by 2030".to_string(),
        ],
        goals: vec![
            "Achieve net-zero emissions".to_string(),
            "Maintain GDP growth".to_string(),
            "Create green jobs".to_string(),
        ],
        available_knowledge: vec![
            "Renewable energy costs are declining rapidly".to_string(),
            "Carbon pricing can drive innovation".to_string(),
            "Green technology creates new industries".to_string(),
        ],
        reasoning_preferences: ReasoningPreferences {
            preferred_strategies: vec![ReasoningStrategy::Analogical, ReasoningStrategy::CausalReasoning],
            max_reasoning_time: Some(300),
            confidence_threshold: 0.8,
            exploration_depth: 4,
            enable_meta_cognition: true,
            enable_analogical_reasoning: true,
            enable_counterfactual_reasoning: true,
        },
        metadata: HashMap::new(),
    }
}

fn create_causal_reasoning_input() -> CausalReasoningInput {
    CausalReasoningInput {
        query: "What is the causal effect of carbon pricing on renewable energy adoption?".to_string(),
        context: Some("Climate policy analysis".to_string()),
        variables: vec![
            Variable {
                name: "carbon_price".to_string(),
                variable_type: VariableType::Continuous,
                possible_values: Vec::new(),
                description: Some("Price per ton of CO2".to_string()),
            },
            Variable {
                name: "renewable_adoption".to_string(),
                variable_type: VariableType::Continuous,
                possible_values: Vec::new(),
                description: Some("Percentage of renewable energy".to_string()),
            },
            Variable {
                name: "economic_growth".to_string(),
                variable_type: VariableType::Continuous,
                possible_values: Vec::new(),
                description: Some("GDP growth rate".to_string()),
            },
        ],
        observations: {
            let mut obs = HashMap::new();
            obs.insert("carbon_price".to_string(), serde_json::Value::Number(50.into()));
            obs.insert("renewable_adoption".to_string(), serde_json::Value::Number(30.into()));
            obs
        },
        causal_assumptions: vec![
            "No unmeasured confounders".to_string(),
            "Stable unit treatment value assumption".to_string(),
        ],
        reasoning_type: CausalReasoningType::EffectEstimation,
        metadata: HashMap::new(),
    }
}

#[tokio::test]
async fn test_advanced_reasoning_creation() {
    let client = create_test_client();
    let signature = Signature::<AdvancedReasoningInput, AdvancedReasoningOutput>::new("test_reasoning".to_string())
        .with_description("Test advanced reasoning capabilities");
    
    let module = AdvancedReasoning::new(signature, client);
    
    assert!(module.name().starts_with("AdvancedReasoning_"));
    assert!(module.supports_compilation());
    assert_eq!(module.config().max_reasoning_steps, 20);
    assert_eq!(module.config().max_thought_tree_depth, 6);
    assert!(module.config().enable_parallel_reasoning);
    assert!(module.config().enable_uncertainty_quantification);
}

#[tokio::test]
async fn test_advanced_reasoning_with_config() {
    let client = create_test_client();
    let signature = Signature::<AdvancedReasoningInput, AdvancedReasoningOutput>::new("test_reasoning".to_string());
    
    let config = AdvancedReasoningConfig {
        max_reasoning_steps: 10,
        max_thought_tree_depth: 4,
        max_graph_nodes: 50,
        enable_parallel_reasoning: false,
        enable_uncertainty_quantification: false,
        enable_explanation_generation: true,
        pruning_threshold: 0.5,
        confidence_calibration: false,
        timeout_seconds: 120,
    };
    
    let module = AdvancedReasoning::with_config(signature, client, config);
    
    assert_eq!(module.config().max_reasoning_steps, 10);
    assert_eq!(module.config().max_thought_tree_depth, 4);
    assert!(!module.config().enable_parallel_reasoning);
    assert!(module.config().enable_explanation_generation);
}

#[tokio::test]
async fn test_advanced_reasoning_input_validation() {
    let client = create_test_client();
    let signature = Signature::<AdvancedReasoningInput, AdvancedReasoningOutput>::new("test_reasoning".to_string());
    let module = AdvancedReasoning::new(signature, client);
    
    // Valid input
    let valid_input = create_advanced_reasoning_input();
    assert!(module.validate_input(&valid_input).await.is_ok());
    
    // Empty problem
    let mut invalid_input = create_advanced_reasoning_input();
    invalid_input.problem = "".to_string();
    assert!(module.validate_input(&invalid_input).await.is_err());
    
    // Invalid confidence threshold
    let mut invalid_input = create_advanced_reasoning_input();
    invalid_input.reasoning_preferences.confidence_threshold = 1.5;
    assert!(module.validate_input(&invalid_input).await.is_err());
    
    // Exploration depth too high
    let mut invalid_input = create_advanced_reasoning_input();
    invalid_input.reasoning_preferences.exploration_depth = 100;
    assert!(module.validate_input(&invalid_input).await.is_err());
}

#[tokio::test]
async fn test_thought_node_structure() {
    let node = ThoughtNode {
        id: "node_1".to_string(),
        content: "Initial thought about climate solutions".to_string(),
        confidence: 0.8,
        depth: 1,
        parent_id: Some("root".to_string()),
        children: vec!["node_2".to_string(), "node_3".to_string()],
        evaluation_score: Some(0.75),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("reasoning_type".to_string(), serde_json::Value::String("deductive".to_string()));
            meta
        },
    };
    
    assert_eq!(node.id, "node_1");
    assert_eq!(node.confidence, 0.8);
    assert_eq!(node.depth, 1);
    assert_eq!(node.children.len(), 2);
    assert_eq!(node.evaluation_score, Some(0.75));
    assert!(node.metadata.contains_key("reasoning_type"));
}

#[tokio::test]
async fn test_tree_of_thought_structure() {
    let root_node = ThoughtNode {
        id: "root".to_string(),
        content: "Root problem".to_string(),
        confidence: 1.0,
        depth: 0,
        parent_id: None,
        children: vec!["child_1".to_string()],
        evaluation_score: None,
        metadata: HashMap::new(),
    };
    
    let tree = TreeOfThought {
        root: root_node,
        max_depth: 5,
        max_branches: 3,
        pruning_threshold: 0.3,
        exploration_strategy: ExplorationStrategy::BestFirst,
    };
    
    assert_eq!(tree.root.id, "root");
    assert_eq!(tree.max_depth, 5);
    assert_eq!(tree.max_branches, 3);
    assert_eq!(tree.exploration_strategy, ExplorationStrategy::BestFirst);
}

#[tokio::test]
async fn test_reasoning_graph_structure() {
    let mut nodes = HashMap::new();
    
    let fact_node = ReasoningNode {
        id: "fact_1".to_string(),
        node_type: ReasoningNodeType::Fact,
        content: "Renewable energy costs are declining".to_string(),
        confidence: 0.9,
        activation_level: 0.8,
        metadata: HashMap::new(),
    };
    nodes.insert("fact_1".to_string(), fact_node);
    
    let hypothesis_node = ReasoningNode {
        id: "hyp_1".to_string(),
        node_type: ReasoningNodeType::Hypothesis,
        content: "Renewable energy will become dominant".to_string(),
        confidence: 0.7,
        activation_level: 0.6,
        metadata: HashMap::new(),
    };
    nodes.insert("hyp_1".to_string(), hypothesis_node);
    
    let edge = ReasoningEdge {
        id: "edge_1".to_string(),
        from_node: "fact_1".to_string(),
        to_node: "hyp_1".to_string(),
        edge_type: ReasoningEdgeType::Supports,
        weight: 0.8,
        confidence: 0.9,
    };
    
    let graph = ReasoningGraph {
        nodes,
        edges: vec![edge],
        entry_points: vec!["fact_1".to_string()],
        goal_nodes: vec!["hyp_1".to_string()],
    };
    
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.entry_points.len(), 1);
    assert_eq!(graph.goal_nodes.len(), 1);
    
    let fact_node = graph.nodes.get("fact_1").unwrap();
    assert_eq!(fact_node.node_type, ReasoningNodeType::Fact);
    assert_eq!(fact_node.confidence, 0.9);
    
    let edge = &graph.edges[0];
    assert_eq!(edge.edge_type, ReasoningEdgeType::Supports);
    assert_eq!(edge.weight, 0.8);
}

#[tokio::test]
async fn test_analogy_mapping() {
    let concept_mapping = ConceptMapping {
        source_concept: "water flow".to_string(),
        target_concept: "electric current".to_string(),
        mapping_type: MappingType::Functional,
        confidence: 0.85,
    };
    
    let analogy = AnalogyMapping {
        source_domain: "hydraulics".to_string(),
        target_domain: "electronics".to_string(),
        mappings: vec![concept_mapping],
        structural_similarity: 0.8,
        semantic_similarity: 0.7,
        pragmatic_relevance: 0.9,
    };
    
    assert_eq!(analogy.source_domain, "hydraulics");
    assert_eq!(analogy.target_domain, "electronics");
    assert_eq!(analogy.mappings.len(), 1);
    assert_eq!(analogy.structural_similarity, 0.8);
    
    let mapping = &analogy.mappings[0];
    assert_eq!(mapping.mapping_type, MappingType::Functional);
    assert_eq!(mapping.confidence, 0.85);
}

#[tokio::test]
async fn test_meta_cognition() {
    let trigger = AdaptationTrigger {
        trigger_type: TriggerType::LowConfidence,
        threshold: 0.5,
        action: AdaptationAction::SwitchStrategy,
    };
    
    let mut monitoring_metrics = HashMap::new();
    monitoring_metrics.insert("progress_rate".to_string(), 0.7);
    monitoring_metrics.insert("confidence_trend".to_string(), 0.8);
    
    let meta_cog = MetaCognition {
        current_strategy: ReasoningStrategy::Deductive,
        confidence_in_strategy: 0.8,
        alternative_strategies: vec![
            ReasoningStrategy::Inductive,
            ReasoningStrategy::Analogical,
        ],
        monitoring_metrics,
        adaptation_triggers: vec![trigger],
    };
    
    assert_eq!(meta_cog.current_strategy, ReasoningStrategy::Deductive);
    assert_eq!(meta_cog.confidence_in_strategy, 0.8);
    assert_eq!(meta_cog.alternative_strategies.len(), 2);
    assert_eq!(meta_cog.monitoring_metrics.len(), 2);
    assert_eq!(meta_cog.adaptation_triggers.len(), 1);
    
    let trigger = &meta_cog.adaptation_triggers[0];
    assert_eq!(trigger.trigger_type, TriggerType::LowConfidence);
    assert_eq!(trigger.action, AdaptationAction::SwitchStrategy);
}

#[tokio::test]
async fn test_reasoning_step() {
    let step = ReasoningStep {
        step_number: 1,
        strategy_used: ReasoningStrategy::Analogical,
        input: "Climate change is like a fever in the Earth's system".to_string(),
        output: "We need to treat the underlying causes, not just symptoms".to_string(),
        confidence: 0.8,
        reasoning_type: "analogical_inference".to_string(),
        evidence_used: vec!["Medical analogy".to_string()],
        assumptions_made: vec!["Earth systems behave like biological systems".to_string()],
        execution_time_ms: 150.0,
    };
    
    assert_eq!(step.step_number, 1);
    assert_eq!(step.strategy_used, ReasoningStrategy::Analogical);
    assert_eq!(step.confidence, 0.8);
    assert_eq!(step.evidence_used.len(), 1);
    assert_eq!(step.assumptions_made.len(), 1);
}

#[tokio::test]
async fn test_causal_reasoning_creation() {
    let client = create_test_client();
    let signature = Signature::<CausalReasoningInput, CausalReasoningOutput>::new("test_causal".to_string())
        .with_description("Test causal reasoning capabilities");
    
    let module = CausalReasoning::new(signature, client);
    
    assert!(module.name().starts_with("CausalReasoning_"));
    assert!(module.supports_compilation());
    assert_eq!(module.config().max_variables, 20);
    assert_eq!(module.config().max_counterfactuals, 5);
    assert!(module.config().enable_causal_discovery);
    assert!(module.config().enable_counterfactual_reasoning);
}

#[tokio::test]
async fn test_causal_reasoning_input_validation() {
    let client = create_test_client();
    let signature = Signature::<CausalReasoningInput, CausalReasoningOutput>::new("test_causal".to_string());
    let module = CausalReasoning::new(signature, client);
    
    // Valid input
    let valid_input = create_causal_reasoning_input();
    assert!(module.validate_input(&valid_input).await.is_ok());
    
    // Empty query
    let mut invalid_input = create_causal_reasoning_input();
    invalid_input.query = "".to_string();
    assert!(module.validate_input(&invalid_input).await.is_err());
    
    // Too many variables
    let mut invalid_input = create_causal_reasoning_input();
    invalid_input.variables = (0..25).map(|i| Variable {
        name: format!("var_{}", i),
        variable_type: VariableType::Binary,
        possible_values: Vec::new(),
        description: None,
    }).collect();
    assert!(module.validate_input(&invalid_input).await.is_err());
}

#[tokio::test]
async fn test_causal_graph_structure() {
    let mut nodes = HashMap::new();
    
    let treatment_node = CausalNode {
        id: "treatment".to_string(),
        name: "carbon_pricing".to_string(),
        node_type: CausalNodeType::Treatment,
        observed: true,
        value: Some(serde_json::Value::Number(50.into())),
        distribution: Some("normal".to_string()),
        metadata: HashMap::new(),
    };
    nodes.insert("treatment".to_string(), treatment_node);
    
    let outcome_node = CausalNode {
        id: "outcome".to_string(),
        name: "renewable_adoption".to_string(),
        node_type: CausalNodeType::Outcome,
        observed: true,
        value: Some(serde_json::Value::Number(30.into())),
        distribution: Some("normal".to_string()),
        metadata: HashMap::new(),
    };
    nodes.insert("outcome".to_string(), outcome_node);
    
    let edge = CausalEdge {
        id: "causal_edge_1".to_string(),
        from_node: "treatment".to_string(),
        to_node: "outcome".to_string(),
        edge_type: CausalEdgeType::DirectCause,
        strength: 0.7,
        confidence: 0.8,
        mechanism: Some("Economic incentive mechanism".to_string()),
    };
    
    let graph = CausalGraph {
        nodes,
        edges: vec![edge],
        confounders: vec!["economic_conditions".to_string()],
        mediators: vec!["technology_investment".to_string()],
        colliders: vec!["policy_support".to_string()],
    };
    
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.confounders.len(), 1);
    assert_eq!(graph.mediators.len(), 1);
    
    let treatment = graph.nodes.get("treatment").unwrap();
    assert_eq!(treatment.node_type, CausalNodeType::Treatment);
    assert!(treatment.observed);
    
    let edge = &graph.edges[0];
    assert_eq!(edge.edge_type, CausalEdgeType::DirectCause);
    assert_eq!(edge.strength, 0.7);
}

#[tokio::test]
async fn test_counterfactual_scenario() {
    let intervention = Intervention {
        target_variable: "carbon_price".to_string(),
        intervention_type: InterventionType::DoIntervention,
        value: serde_json::Value::Number(100.into()),
        mechanism: Some("Policy intervention".to_string()),
    };
    
    let mut predicted_outcomes = HashMap::new();
    predicted_outcomes.insert("renewable_adoption".to_string(), serde_json::Value::Number(50.into()));
    
    let scenario = CounterfactualScenario {
        scenario_id: "scenario_1".to_string(),
        description: "What if carbon price was doubled?".to_string(),
        interventions: vec![intervention],
        predicted_outcomes,
        confidence: 0.75,
        assumptions: vec![
            "No general equilibrium effects".to_string(),
            "Technology remains constant".to_string(),
        ],
    };
    
    assert_eq!(scenario.interventions.len(), 1);
    assert_eq!(scenario.predicted_outcomes.len(), 1);
    assert_eq!(scenario.confidence, 0.75);
    assert_eq!(scenario.assumptions.len(), 2);
    
    let intervention = &scenario.interventions[0];
    assert_eq!(intervention.intervention_type, InterventionType::DoIntervention);
    assert_eq!(intervention.target_variable, "carbon_price");
}

#[tokio::test]
async fn test_causal_effect() {
    let effect = CausalEffect {
        treatment: "carbon_pricing".to_string(),
        outcome: "renewable_adoption".to_string(),
        effect_size: 0.4,
        confidence_interval: (0.2, 0.6),
        p_value: Some(0.01),
        method_used: "Backdoor adjustment".to_string(),
        assumptions: vec![
            "Unconfoundedness".to_string(),
            "Positivity".to_string(),
        ],
    };
    
    assert_eq!(effect.treatment, "carbon_pricing");
    assert_eq!(effect.outcome, "renewable_adoption");
    assert_eq!(effect.effect_size, 0.4);
    assert_eq!(effect.confidence_interval, (0.2, 0.6));
    assert_eq!(effect.p_value, Some(0.01));
    assert_eq!(effect.assumptions.len(), 2);
}

#[tokio::test]
async fn test_reasoning_strategies() {
    let strategies = vec![
        ReasoningStrategy::Deductive,
        ReasoningStrategy::Inductive,
        ReasoningStrategy::Abductive,
        ReasoningStrategy::Analogical,
        ReasoningStrategy::CausalReasoning,
        ReasoningStrategy::CounterfactualReasoning,
        ReasoningStrategy::ProbabilisticReasoning,
        ReasoningStrategy::ConstraintSatisfaction,
    ];
    
    // Test that all strategies are distinct
    for (i, strategy1) in strategies.iter().enumerate() {
        for (j, strategy2) in strategies.iter().enumerate() {
            if i != j {
                assert_ne!(strategy1, strategy2);
            }
        }
    }
}

#[tokio::test]
async fn test_exploration_strategies() {
    let strategies = vec![
        ExplorationStrategy::BreadthFirst,
        ExplorationStrategy::DepthFirst,
        ExplorationStrategy::BestFirst,
        ExplorationStrategy::MonteCarloTreeSearch,
        ExplorationStrategy::BeamSearch { beam_width: 3 },
    ];
    
    // Test beam search configuration
    if let ExplorationStrategy::BeamSearch { beam_width } = &strategies[4] {
        assert_eq!(*beam_width, 3);
    } else {
        panic!("Expected BeamSearch strategy");
    }
}

#[tokio::test]
async fn test_reasoning_preferences() {
    let prefs = ReasoningPreferences {
        preferred_strategies: vec![
            ReasoningStrategy::Analogical,
            ReasoningStrategy::CausalReasoning,
        ],
        max_reasoning_time: Some(600),
        confidence_threshold: 0.85,
        exploration_depth: 6,
        enable_meta_cognition: true,
        enable_analogical_reasoning: true,
        enable_counterfactual_reasoning: false,
    };
    
    assert_eq!(prefs.preferred_strategies.len(), 2);
    assert_eq!(prefs.max_reasoning_time, Some(600));
    assert_eq!(prefs.confidence_threshold, 0.85);
    assert_eq!(prefs.exploration_depth, 6);
    assert!(prefs.enable_meta_cognition);
    assert!(prefs.enable_analogical_reasoning);
    assert!(!prefs.enable_counterfactual_reasoning);
}

#[tokio::test]
async fn test_variable_types() {
    let variables = vec![
        Variable {
            name: "binary_var".to_string(),
            variable_type: VariableType::Binary,
            possible_values: vec![
                serde_json::Value::Bool(true),
                serde_json::Value::Bool(false),
            ],
            description: Some("Binary variable".to_string()),
        },
        Variable {
            name: "categorical_var".to_string(),
            variable_type: VariableType::Categorical,
            possible_values: vec![
                serde_json::Value::String("low".to_string()),
                serde_json::Value::String("medium".to_string()),
                serde_json::Value::String("high".to_string()),
            ],
            description: Some("Categorical variable".to_string()),
        },
        Variable {
            name: "continuous_var".to_string(),
            variable_type: VariableType::Continuous,
            possible_values: Vec::new(),
            description: Some("Continuous variable".to_string()),
        },
    ];
    
    assert_eq!(variables[0].variable_type, VariableType::Binary);
    assert_eq!(variables[0].possible_values.len(), 2);
    
    assert_eq!(variables[1].variable_type, VariableType::Categorical);
    assert_eq!(variables[1].possible_values.len(), 3);
    
    assert_eq!(variables[2].variable_type, VariableType::Continuous);
    assert_eq!(variables[2].possible_values.len(), 0);
}

#[tokio::test]
async fn test_reasoning_integration() {
    // Test that advanced reasoning and causal reasoning can work together
    let client = create_test_client();
    
    // Create advanced reasoning module
    let adv_signature = Signature::<AdvancedReasoningInput, AdvancedReasoningOutput>::new("advanced_test".to_string());
    let adv_module = AdvancedReasoning::new(adv_signature, client.clone());
    
    // Create causal reasoning module
    let causal_signature = Signature::<CausalReasoningInput, CausalReasoningOutput>::new("causal_test".to_string());
    let causal_module = CausalReasoning::new(causal_signature, client);
    
    // Test that both modules can handle related problems
    let adv_input = create_advanced_reasoning_input();
    let causal_input = create_causal_reasoning_input();
    
    assert!(adv_module.validate_input(&adv_input).await.is_ok());
    assert!(causal_module.validate_input(&causal_input).await.is_ok());
    
    // Both modules should support compilation
    assert!(adv_module.supports_compilation());
    assert!(causal_module.supports_compilation());
}
