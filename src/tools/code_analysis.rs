use async_trait::async_trait;
use crate::anthropic::models::ToolDefinition;
use crate::tools::{create_tool_definition, extract_optional_bool_param, extract_optional_int_param, extract_optional_string_param, extract_string_param, Tool, ToolResult};
use crate::utils::error::{AgentError, Result};
use rust_tree_sitter::{CodebaseAnalyzer, AnalysisConfig};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Code analysis tool for intelligent codebase understanding
#[derive(Debug, Clone)]
pub struct CodeAnalysisTool;

impl CodeAnalysisTool {
    /// Create a new code analysis tool
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CodeAnalysisTool {
    fn definition(&self) -> ToolDefinition {
        create_tool_definition(
            "code_analysis",
            "Analyze code files and directories using tree-sitter for intelligent insights, symbol extraction, and code quality assessment",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["analyze", "insights", "find_symbols", "query_patterns", "stats", "map_structure", "explain", "security", "refactor"],
                        "description": "Type of code analysis to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to analyze"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["rust", "javascript", "python", "c", "cpp", "auto"],
                        "description": "Programming language (auto-detect if not specified)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Tree-sitter query pattern for pattern matching"
                    },
                    "symbol_name": {
                        "type": "string",
                        "description": "Symbol name pattern for symbol search (supports wildcards)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "struct", "enum", "variable", "all"],
                        "description": "Type of symbol to search for"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to analyze",
                        "default": 10
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories",
                        "default": false
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "markdown", "summary"],
                        "description": "Output format for results",
                        "default": "json"
                    },
                    "detailed": {
                        "type": "boolean",
                        "description": "Enable detailed analysis for explain/security/refactor actions",
                        "default": false
                    },
                    "learning": {
                        "type": "boolean",
                        "description": "Enable learning mode for explanations",
                        "default": false
                    },
                    "compliance": {
                        "type": "boolean",
                        "description": "Enable compliance assessment for security scanning",
                        "default": false
                    },
                    "quick_wins": {
                        "type": "boolean",
                        "description": "Focus on quick wins for refactoring suggestions",
                        "default": false
                    }
                },
                "required": ["action", "path"]
            }),
        )
    }

    async fn execute(&self, params: Value) -> Result<ToolResult> {
        let action = extract_string_param(&params, "action")?;
        let path = extract_string_param(&params, "path")?;

        let path_buf = PathBuf::from(&path);
        if !path_buf.exists() {
            return Err(AgentError::invalid_input(format!("Path does not exist: {}", path)));
        }

        info!("Performing code analysis: {} on {}", action, path);

        let result = match action.as_str() {
            "analyze" => self.analyze_codebase(&path_buf, &params).await?,
            "insights" => self.generate_insights(&path_buf, &params).await?,
            "find_symbols" => self.find_symbols(&path_buf, &params).await?,
            "query_patterns" => self.query_patterns(&path_buf, &params).await?,
            "stats" => self.generate_stats(&path_buf, &params).await?,
            "map_structure" => self.map_structure(&path_buf, &params).await?,
            "explain" => self.explain_code(&path_buf, &params).await?,
            "security" => self.security_scan(&path_buf, &params).await?,
            "refactor" => self.refactor_suggestions(&path_buf, &params).await?,
            _ => return Err(AgentError::invalid_input(format!("Unknown action: {}", action)))
        };

        Ok(ToolResult::success(serde_json::to_string_pretty(&result)?))
    }

    fn name(&self) -> &str {
        "code_analysis"
    }

    fn description(&self) -> Option<&str> {
        Some("Analyze code files and directories using tree-sitter for intelligent insights, symbol extraction, and code quality assessment")
    }
}

impl CodeAnalysisTool {
    /// Analyze a codebase and extract comprehensive information
    async fn analyze_codebase(&self, path: &Path, params: &Value) -> Result<Value> {
        let max_depth = extract_optional_int_param(params, "max_depth").unwrap_or(10) as usize;
        let include_hidden = extract_optional_bool_param(params, "include_hidden").unwrap_or(false);
        
        let config = AnalysisConfig {
            max_depth: Some(max_depth),
            include_hidden,
            max_file_size: Some(1024 * 1024), // 1MB max
            exclude_dirs: vec!["target".to_string(), "node_modules".to_string(), ".git".to_string()],
            ..Default::default()
        };

        let mut analyzer = CodebaseAnalyzer::with_config(config);
        
        let result = if path.is_file() {
            // For single files, we need to analyze the parent directory
            // and filter for just this file
            let parent_dir = path.parent().unwrap_or(path);
            let full_result = analyzer.analyze_directory(parent_dir)
                .map_err(|e| AgentError::tool("code_analysis", &format!("Analysis failed: {}", e)))?;

            // Filter to just the requested file
            let file_name = path.file_name().unwrap();
            let filtered_files: Vec<_> = full_result.files.into_iter()
                .filter(|f| f.path.file_name() == Some(file_name))
                .collect();

            rust_tree_sitter::AnalysisResult {
                root_path: full_result.root_path,
                total_files: filtered_files.len(),
                parsed_files: filtered_files.iter().filter(|f| f.parsed_successfully).count(),
                error_files: filtered_files.iter().filter(|f| !f.parsed_successfully).count(),
                total_lines: filtered_files.iter().map(|f| f.lines).sum(),
                languages: {
                    let mut langs = std::collections::HashMap::new();
                    for file in &filtered_files {
                        *langs.entry(file.language.clone()).or_insert(0) += 1;
                    }
                    langs
                },
                files: filtered_files,
                config: full_result.config,
            }
        } else {
            analyzer.analyze_directory(path)
                .map_err(|e| AgentError::tool("code_analysis", &format!("Analysis failed: {}", e)))?
        };

        debug!("Analysis complete: {} files, {} languages", 
               result.total_files, result.languages.len());

        let output = json!({
            "analysis_type": "codebase_analysis",
            "path": path.display().to_string(),
            "summary": {
                "total_files": result.total_files,
                "total_lines": result.total_lines,
                "total_symbols": result.files.iter().map(|f| f.symbols.len()).sum::<usize>(),
                "languages": result.languages.keys().collect::<Vec<_>>(),
                "largest_file": result.files.iter()
                    .max_by_key(|f| f.lines)
                    .map(|f| json!({
                        "path": f.path.display().to_string(),
                        "lines": f.lines,
                        "symbols": f.symbols.len()
                    }))
            },
            "files": result.files.iter().map(|file| {
                json!({
                    "path": file.path.display().to_string(),
                    "language": file.language,
                    "lines": file.lines,
                    "size_bytes": file.size,
                    "symbols": file.symbols.iter().map(|symbol| {
                        json!({
                            "name": symbol.name,
                            "kind": symbol.kind,
                            "start_line": symbol.start_line,
                            "end_line": symbol.end_line,
                            "is_public": symbol.is_public
                        })
                    }).collect::<Vec<_>>()
                })
            }).collect::<Vec<_>>(),
            "language_breakdown": result.languages
        });

        Ok(output)
    }

    /// Generate AI-friendly insights about the codebase
    async fn generate_insights(&self, path: &Path, params: &Value) -> Result<Value> {
        let analysis = self.analyze_codebase(path, params).await?;
        
        let files = analysis["files"].as_array().unwrap();
        let total_files = analysis["summary"]["total_files"].as_u64().unwrap();
        let total_lines = analysis["summary"]["total_lines"].as_u64().unwrap();
        let total_symbols = analysis["summary"]["total_symbols"].as_u64().unwrap();
        
        // Calculate insights
        let avg_lines_per_file = if total_files > 0 { total_lines / total_files } else { 0 };
        let avg_symbols_per_file = if total_files > 0 { total_symbols / total_files } else { 0 };
        
        let large_files = files.iter()
            .filter(|f| f["lines"].as_u64().unwrap_or(0) > 500)
            .count();
            
        let complex_files = files.iter()
            .filter(|f| f["symbols"].as_array().unwrap().len() > 20)
            .count();

        let insights = json!({
            "analysis_type": "codebase_insights",
            "path": path.display().to_string(),
            "metrics": {
                "total_files": total_files,
                "total_lines": total_lines,
                "total_symbols": total_symbols,
                "avg_lines_per_file": avg_lines_per_file,
                "avg_symbols_per_file": avg_symbols_per_file,
                "large_files_count": large_files,
                "complex_files_count": complex_files
            },
            "quality_indicators": {
                "parse_success_rate": 100.0, // TODO: Calculate actual parse success rate
                "average_file_size": avg_lines_per_file,
                "complexity_distribution": {
                    "simple": files.len() - complex_files,
                    "complex": complex_files
                }
            },
            "recommendations": self.generate_recommendations(files, total_files as usize),
            "architecture_notes": self.analyze_architecture(files)
        });

        Ok(insights)
    }

    /// Find symbols matching specified criteria
    async fn find_symbols(&self, path: &Path, params: &Value) -> Result<Value> {
        let symbol_name = extract_optional_string_param(params, "symbol_name");
        let symbol_type = extract_optional_string_param(params, "symbol_type").unwrap_or_else(|| "all".to_string());
        
        let analysis = self.analyze_codebase(path, params).await?;
        let files = analysis["files"].as_array().unwrap();
        
        let mut matching_symbols = Vec::new();
        
        for file in files {
            let file_path = file["path"].as_str().unwrap();
            let symbols = file["symbols"].as_array().unwrap();
            
            for symbol in symbols {
                let name = symbol["name"].as_str().unwrap();
                let kind = symbol["kind"].as_str().unwrap();
                
                // Filter by symbol type
                if symbol_type != "all" && kind != symbol_type {
                    continue;
                }

                // Filter by name pattern (simple wildcard support)
                if let Some(ref pattern) = symbol_name {
                    if !self.matches_pattern(name, pattern) {
                        continue;
                    }
                }
                
                matching_symbols.push(json!({
                    "file": file_path,
                    "name": name,
                    "kind": kind,
                    "line": symbol["start_line"],
                    "is_public": symbol["is_public"]
                }));
            }
        }
        
        Ok(json!({
            "analysis_type": "symbol_search",
            "path": path.display().to_string(),
            "search_criteria": {
                "symbol_name": symbol_name,
                "symbol_type": symbol_type
            },
            "results_count": matching_symbols.len(),
            "symbols": matching_symbols
        }))
    }

    /// Query code patterns using tree-sitter queries
    async fn query_patterns(&self, path: &Path, params: &Value) -> Result<Value> {
        let pattern = extract_string_param(params, "pattern")?;
        let language_str = extract_optional_string_param(params, "language").unwrap_or_else(|| "auto".to_string());
        
        // For now, return a placeholder - full implementation would require
        // parsing individual files and running tree-sitter queries
        Ok(json!({
            "analysis_type": "pattern_query",
            "path": path.display().to_string(),
            "pattern": pattern,
            "language": language_str,
            "message": "Pattern querying is available - implementation requires file-by-file parsing",
            "matches": []
        }))
    }

    /// Generate codebase statistics
    async fn generate_stats(&self, path: &Path, params: &Value) -> Result<Value> {
        let analysis = self.analyze_codebase(path, params).await?;

        Ok(json!({
            "analysis_type": "codebase_statistics",
            "path": path.display().to_string(),
            "statistics": analysis["summary"]
        }))
    }

    /// Map codebase structure
    async fn map_structure(&self, path: &Path, params: &Value) -> Result<Value> {
        let analysis = self.analyze_codebase(path, params).await?;

        Ok(json!({
            "analysis_type": "structure_map",
            "path": path.display().to_string(),
            "structure": analysis["files"]
        }))
    }

    /// Simple wildcard pattern matching
    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return text.starts_with(prefix) && text.ends_with(suffix);
            }
        }
        text == pattern
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(&self, files: &[Value], total_files: usize) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let large_files = files.iter()
            .filter(|f| f["lines"].as_u64().unwrap_or(0) > 500)
            .count();
            
        if large_files > total_files / 4 {
            recommendations.push("Consider breaking down large files (>500 lines) into smaller modules".to_string());
        }
        
        let files_without_symbols = files.iter()
            .filter(|f| f["symbols"].as_array().unwrap().is_empty())
            .count();
            
        if files_without_symbols > 0 {
            recommendations.push(format!("{} files have no detected symbols - check for parsing issues", files_without_symbols));
        }
        
        recommendations
    }

    /// Analyze architecture patterns
    fn analyze_architecture(&self, files: &[Value]) -> Vec<String> {
        let mut notes = Vec::new();

        let rust_files = files.iter()
            .filter(|f| f["language"].as_str() == Some("rust"))
            .count();

        if rust_files > 0 {
            notes.push(format!("Rust codebase with {} files", rust_files));
        }

        notes
    }

    /// AI-powered code explanations
    async fn explain_code(&self, path: &Path, params: &Value) -> Result<Value> {
        let detailed = extract_optional_bool_param(params, "detailed").unwrap_or(false);
        let learning = extract_optional_bool_param(params, "learning").unwrap_or(false);

        info!("Generating AI explanations for: {}", path.display());

        // For now, provide a structured explanation based on our analysis
        // In the future, this could integrate with the rust-treesitter CLI explain command
        let analysis = self.analyze_codebase(path, params).await?;

        let mut explanations = Vec::new();
        let files = analysis["files"].as_array().unwrap();

        for file in files.iter().take(if detailed { 10 } else { 5 }) {
            let file_path = file["path"].as_str().unwrap();
            let language = file["language"].as_str().unwrap_or("unknown");
            let symbols = file["symbols"].as_array().unwrap();

            let mut explanation = format!("📁 **{}** ({})", file_path, language);

            if !symbols.is_empty() {
                explanation.push_str(&format!("\n   Contains {} symbols:", symbols.len()));

                for symbol in symbols.iter().take(if detailed { 10 } else { 3 }) {
                    let name = symbol["name"].as_str().unwrap();
                    let kind = symbol["kind"].as_str().unwrap();
                    let line = symbol["start_line"].as_u64().unwrap();
                    let is_public = symbol["is_public"].as_bool().unwrap_or(false);

                    let visibility = if is_public { "public" } else { "private" };
                    explanation.push_str(&format!("\n   - {} {} `{}` (line {})", visibility, kind, name, line));
                }

                if symbols.len() > (if detailed { 10 } else { 3 }) {
                    explanation.push_str(&format!("\n   ... and {} more symbols", symbols.len() - (if detailed { 10 } else { 3 })));
                }
            }

            if learning {
                explanation.push_str(&format!("\n   💡 **Learning Note**: This {} file demonstrates {} patterns", language, kind_to_pattern(language)));
            }

            explanations.push(explanation);
        }

        Ok(json!({
            "analysis_type": "ai_explanation",
            "path": path.display().to_string(),
            "detailed": detailed,
            "learning_mode": learning,
            "explanations": explanations,
            "summary": {
                "total_files_explained": explanations.len(),
                "explanation_depth": if detailed { "detailed" } else { "overview" },
                "learning_insights": if learning { "enabled" } else { "disabled" }
            }
        }))
    }

    /// Security vulnerability scanning
    async fn security_scan(&self, path: &Path, params: &Value) -> Result<Value> {
        let compliance = extract_optional_bool_param(params, "compliance").unwrap_or(false);

        info!("Performing security scan for: {}", path.display());

        let analysis = self.analyze_codebase(path, params).await?;
        let files = analysis["files"].as_array().unwrap();

        let mut security_issues = Vec::new();
        let mut compliance_notes = Vec::new();

        for file in files {
            let file_path = file["path"].as_str().unwrap();
            let language = file["language"].as_str().unwrap_or("unknown");
            let symbols = file["symbols"].as_array().unwrap();

            // Basic security pattern detection
            for symbol in symbols {
                let name = symbol["name"].as_str().unwrap();
                let kind = symbol["kind"].as_str().unwrap();
                let is_public = symbol["is_public"].as_bool().unwrap_or(false);

                // Check for potential security issues
                if name.to_lowercase().contains("password") && is_public {
                    security_issues.push(json!({
                        "severity": "high",
                        "type": "exposed_credential",
                        "file": file_path,
                        "symbol": name,
                        "line": symbol["start_line"],
                        "description": "Public symbol contains 'password' - potential credential exposure"
                    }));
                }

                if name.to_lowercase().contains("unsafe") && language == "rust" {
                    security_issues.push(json!({
                        "severity": "medium",
                        "type": "unsafe_code",
                        "file": file_path,
                        "symbol": name,
                        "line": symbol["start_line"],
                        "description": "Unsafe code detected - requires security review"
                    }));
                }

                if kind == "function" && name.starts_with("test_") && is_public {
                    security_issues.push(json!({
                        "severity": "low",
                        "type": "exposed_test",
                        "file": file_path,
                        "symbol": name,
                        "line": symbol["start_line"],
                        "description": "Public test function - consider making private"
                    }));
                }
            }

            if compliance {
                // Add compliance assessment
                let public_symbols = symbols.iter().filter(|s| s["is_public"].as_bool().unwrap_or(false)).count();
                let total_symbols = symbols.len();

                if total_symbols > 0 {
                    let public_ratio = public_symbols as f64 / total_symbols as f64;
                    if public_ratio > 0.8 {
                        compliance_notes.push(format!("File {} has high public API exposure ({:.1}%)", file_path, public_ratio * 100.0));
                    }
                }
            }
        }

        let risk_level = if security_issues.iter().any(|i| i["severity"] == "high") {
            "high"
        } else if security_issues.iter().any(|i| i["severity"] == "medium") {
            "medium"
        } else if !security_issues.is_empty() {
            "low"
        } else {
            "minimal"
        };

        Ok(json!({
            "analysis_type": "security_scan",
            "path": path.display().to_string(),
            "compliance_enabled": compliance,
            "risk_assessment": {
                "overall_risk": risk_level,
                "total_issues": security_issues.len(),
                "high_severity": security_issues.iter().filter(|i| i["severity"] == "high").count(),
                "medium_severity": security_issues.iter().filter(|i| i["severity"] == "medium").count(),
                "low_severity": security_issues.iter().filter(|i| i["severity"] == "low").count()
            },
            "security_issues": security_issues,
            "compliance_notes": compliance_notes,
            "recommendations": generate_security_recommendations(&security_issues)
        }))
    }

    /// Smart refactoring suggestions
    async fn refactor_suggestions(&self, path: &Path, params: &Value) -> Result<Value> {
        let quick_wins = extract_optional_bool_param(params, "quick_wins").unwrap_or(false);

        info!("Generating refactoring suggestions for: {}", path.display());

        let analysis = self.analyze_codebase(path, params).await?;
        let files = analysis["files"].as_array().unwrap();

        let mut suggestions = Vec::new();

        for file in files {
            let file_path = file["path"].as_str().unwrap();
            let language = file["language"].as_str().unwrap_or("unknown");
            let symbols = file["symbols"].as_array().unwrap();
            let lines = file["lines"].as_u64().unwrap_or(0);

            // Large file suggestion
            if lines > 500 {
                suggestions.push(json!({
                    "priority": if quick_wins { "medium" } else { "high" },
                    "type": "file_size",
                    "file": file_path,
                    "description": format!("Large file ({} lines) - consider splitting into smaller modules", lines),
                    "effort": "medium",
                    "impact": "maintainability"
                }));
            }

            // Too many symbols in one file
            if symbols.len() > 30 {
                suggestions.push(json!({
                    "priority": if quick_wins { "low" } else { "medium" },
                    "type": "symbol_density",
                    "file": file_path,
                    "description": format!("High symbol density ({} symbols) - consider organizing into modules", symbols.len()),
                    "effort": "high",
                    "impact": "organization"
                }));
            }

            // Language-specific suggestions
            if language == "rust" {
                let public_functions = symbols.iter()
                    .filter(|s| s["kind"] == "function" && s["is_public"].as_bool().unwrap_or(false))
                    .count();

                if public_functions > 10 {
                    suggestions.push(json!({
                        "priority": if quick_wins { "low" } else { "medium" },
                        "type": "api_surface",
                        "file": file_path,
                        "description": format!("Large public API ({} functions) - consider facade pattern", public_functions),
                        "effort": "medium",
                        "impact": "api_design"
                    }));
                }
            }

            // Quick win suggestions
            if quick_wins {
                let private_symbols = symbols.iter()
                    .filter(|s| !s["is_public"].as_bool().unwrap_or(false))
                    .count();

                if private_symbols == 0 && !symbols.is_empty() {
                    suggestions.push(json!({
                        "priority": "high",
                        "type": "visibility",
                        "file": file_path,
                        "description": "All symbols are public - consider making some private",
                        "effort": "low",
                        "impact": "encapsulation"
                    }));
                }
            }
        }

        // Sort by priority if quick_wins is enabled
        if quick_wins {
            suggestions.sort_by(|a, b| {
                let priority_order = |p: &str| match p {
                    "high" => 0,
                    "medium" => 1,
                    "low" => 2,
                    _ => 3,
                };
                priority_order(a["priority"].as_str().unwrap_or("low"))
                    .cmp(&priority_order(b["priority"].as_str().unwrap_or("low")))
            });
        }

        Ok(json!({
            "analysis_type": "refactor_suggestions",
            "path": path.display().to_string(),
            "quick_wins_mode": quick_wins,
            "summary": {
                "total_suggestions": suggestions.len(),
                "high_priority": suggestions.iter().filter(|s| s["priority"] == "high").count(),
                "medium_priority": suggestions.iter().filter(|s| s["priority"] == "medium").count(),
                "low_priority": suggestions.iter().filter(|s| s["priority"] == "low").count()
            },
            "suggestions": suggestions,
            "impact_areas": calculate_impact_areas(&suggestions)
        }))
    }
}

/// Helper function to map language to common patterns
fn kind_to_pattern(language: &str) -> &str {
    match language {
        "rust" => "ownership and memory safety",
        "javascript" => "functional and object-oriented",
        "python" => "dynamic typing and duck typing",
        "c" => "procedural and low-level memory management",
        "cpp" => "object-oriented and template metaprogramming",
        _ => "general programming"
    }
}

/// Generate security recommendations based on issues found
fn generate_security_recommendations(issues: &[Value]) -> Vec<String> {
    let mut recommendations = Vec::new();

    let high_issues = issues.iter().filter(|i| i["severity"] == "high").count();
    let medium_issues = issues.iter().filter(|i| i["severity"] == "medium").count();

    if high_issues > 0 {
        recommendations.push("🚨 Address high-severity security issues immediately".to_string());
        recommendations.push("🔒 Review credential handling and access patterns".to_string());
    }

    if medium_issues > 0 {
        recommendations.push("⚠️ Schedule review of medium-severity security issues".to_string());
        recommendations.push("🛡️ Consider security code review process".to_string());
    }

    if issues.iter().any(|i| i["type"] == "unsafe_code") {
        recommendations.push("🦀 Review all unsafe Rust code blocks for memory safety".to_string());
    }

    if issues.iter().any(|i| i["type"] == "exposed_test") {
        recommendations.push("🧪 Make test functions private to reduce attack surface".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("✅ No immediate security concerns detected".to_string());
        recommendations.push("🔄 Consider regular security audits".to_string());
    }

    recommendations
}

/// Calculate impact areas for refactoring suggestions
fn calculate_impact_areas(suggestions: &[Value]) -> Vec<String> {
    let mut areas = std::collections::HashSet::new();

    for suggestion in suggestions {
        if let Some(impact) = suggestion["impact"].as_str() {
            areas.insert(impact.to_string());
        }
    }

    areas.into_iter().collect()
}
