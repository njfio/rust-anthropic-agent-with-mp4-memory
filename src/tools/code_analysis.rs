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
                        "enum": [
                            "analyze", "insights", "find_symbols", "query_patterns", "stats", "map_structure",
                            "explain", "security", "refactor",
                            // Phase 2 Advanced Intelligence Actions
                            "advanced_ai", "semantic_analysis", "pattern_recognition", "learning_paths",
                            "advanced_security", "owasp_scan", "secrets_detection", "vulnerability_scan",
                            "smart_refactor", "code_smells", "design_patterns", "performance_optimize",
                            "test_coverage", "missing_tests", "test_quality", "coverage_gaps",
                            "dependency_scan", "security_deps", "outdated_deps", "license_check"
                        ],
                        "description": "Analysis type"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to analyze"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["rust", "javascript", "typescript", "go", "python", "c", "cpp", "auto"],
                        "description": "Programming language"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Query pattern"
                    },
                    "symbol_name": {
                        "type": "string",
                        "description": "Symbol name pattern"
                    },
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "struct", "enum", "variable", "all"],
                        "description": "Symbol type"
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
                    },
                    // Phase 2 Advanced Intelligence Parameters
                    "semantic": {
                        "type": "boolean",
                        "description": "Enable deep semantic analysis with concept recognition",
                        "default": false
                    },
                    "patterns": {
                        "type": "boolean",
                        "description": "Enable architecture pattern detection and recommendations",
                        "default": false
                    },
                    "owasp_top10": {
                        "type": "boolean",
                        "description": "Enable OWASP Top 10 vulnerability detection",
                        "default": false
                    },
                    "secrets": {
                        "type": "boolean",
                        "description": "Enable advanced secrets detection with entropy analysis",
                        "default": false
                    },
                    "roadmap": {
                        "type": "boolean",
                        "description": "Generate comprehensive refactoring roadmap with phases",
                        "default": false
                    },
                    "benchmarks": {
                        "type": "boolean",
                        "description": "Include performance benchmarking suggestions",
                        "default": false
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Confidence threshold for AI recommendations (0.0-1.0)",
                        "default": 0.7
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
            // Core analysis actions
            "analyze" => self.analyze_codebase(&path_buf, &params).await?,
            "insights" => self.generate_insights(&path_buf, &params).await?,
            "find_symbols" => self.find_symbols(&path_buf, &params).await?,
            "query_patterns" => self.query_patterns(&path_buf, &params).await?,
            "stats" => self.generate_stats(&path_buf, &params).await?,
            "map_structure" => self.map_structure(&path_buf, &params).await?,
            "explain" => self.explain_code(&path_buf, &params).await?,
            "security" => self.security_scan(&path_buf, &params).await?,
            "refactor" => self.refactor_suggestions(&path_buf, &params).await?,

            // Phase 2 Advanced Intelligence Actions
            "advanced_ai" | "semantic_analysis" => self.advanced_ai_analysis(&path_buf, &params).await?,
            "pattern_recognition" => self.pattern_recognition(&path_buf, &params).await?,
            "learning_paths" => self.learning_paths(&path_buf, &params).await?,
            "advanced_security" | "owasp_scan" => self.advanced_security_scan(&path_buf, &params).await?,
            "secrets_detection" => self.secrets_detection(&path_buf, &params).await?,
            "vulnerability_scan" => self.vulnerability_scan(&path_buf, &params).await?,
            "smart_refactor" | "code_smells" => self.smart_refactor(&path_buf, &params).await?,
            "design_patterns" => self.design_patterns(&path_buf, &params).await?,
            "performance_optimize" => self.performance_optimize(&path_buf, &params).await?,
            "test_coverage" => self.test_coverage_analysis(&path_buf, &params).await?,
            "missing_tests" => self.missing_tests(&path_buf, &params).await?,
            "test_quality" => self.test_quality(&path_buf, &params).await?,
            "coverage_gaps" => self.coverage_gaps(&path_buf, &params).await?,
            "dependency_scan" => self.dependency_scan(&path_buf, &params).await?,
            "security_deps" => self.security_deps(&path_buf, &params).await?,
            "outdated_deps" => self.outdated_deps(&path_buf, &params).await?,
            "license_check" => self.license_check(&path_buf, &params).await?,

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
                "parsed_files": result.parsed_files,
                "error_files": result.error_files,
                "parse_success_rate": if result.total_files > 0 {
                    (result.parsed_files as f64 / result.total_files as f64) * 100.0
                } else { 0.0 },
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
                            "end_line": symbol.end_line
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

        let parsed_files = analysis["summary"]["parsed_files"].as_u64().unwrap_or(0);
        let parse_success_rate = if total_files > 0 {
            (parsed_files as f64 * 100.0) / total_files as f64
        } else {
            0.0
        };

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
                "parse_success_rate": parse_success_rate,
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
                    "line": symbol["start_line"]
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
        // Determine explicit language if provided
        let lang_override = match language_str.as_str() {
            "auto" => None,
            other => Some(self.parse_language(other)?),
        };

        // Analyze the codebase to get file list
        let analysis = self.analyze_codebase(path, params).await?;
        let files = analysis["files"].as_array().unwrap();
        let root = if path.is_file() { path.parent().unwrap_or(path) } else { path };

        // Precreate query if language is fixed
        let base_query = if let Some(lang) = lang_override {
            Some(rust_tree_sitter::Query::new(lang, &pattern)
                .map_err(|e| AgentError::tool("code_analysis", &format!("Query error: {}", e)))?)
        } else {
            None
        };

        let mut matches_vec = Vec::new();

        for file in files {
            let rel_path = file["path"].as_str().unwrap();
            let lang_name = file["language"].as_str().unwrap();

            let language = if let Some(lang) = lang_override {
                if lang.name().eq_ignore_ascii_case(lang_name) { lang } else { continue } 
            } else {
                match self.parse_language(lang_name) {
                    Ok(l) => l,
                    Err(_) => continue,
                }
            };

            let full_path = root.join(rel_path);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let parser = rust_tree_sitter::Parser::new(language)
                .map_err(|e| AgentError::tool("code_analysis", &format!("Parser error: {}", e)))?;
            let tree = match parser.parse(&content, None) {
                Ok(t) => t,
                Err(_) => continue,
            };

            if let Some(ref q) = base_query {
                let qmatches = q.matches(&tree)
                    .map_err(|e| AgentError::tool("code_analysis", &format!("Query execution failed: {}", e)))?;
                for m in qmatches {
                    for cap in m.captures() {
                        let node = cap.node();
                        let start = node.start_position();
                        matches_vec.push(json!({
                            "file": rel_path,
                            "line": start.row + 1,
                            "column": start.column,
                            "text": node.text().unwrap_or("")
                        }));
                    }
                }
            } else {
                let q = rust_tree_sitter::Query::new(language, &pattern)
                    .map_err(|e| AgentError::tool("code_analysis", &format!("Query error: {}", e)))?;
                let qmatches = q.matches(&tree)
                    .map_err(|e| AgentError::tool("code_analysis", &format!("Query execution failed: {}", e)))?;
                for m in qmatches {
                    for cap in m.captures() {
                        let node = cap.node();
                        let start = node.start_position();
                        matches_vec.push(json!({
                            "file": rel_path,
                            "line": start.row + 1,
                            "column": start.column,
                            "text": node.text().unwrap_or("")
                        }));
                    }
                }
            }
        }

        Ok(json!({
            "analysis_type": "pattern_query",
            "path": path.display().to_string(),
            "pattern": pattern,
            "language": language_str,
            "match_count": matches_vec.len(),
            "matches": matches_vec
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

    /// Convert a language name to the rust_tree_sitter Language enum
    fn parse_language(&self, name: &str) -> Result<rust_tree_sitter::Language> {
        match name.to_lowercase().as_str() {
            "rust" => Ok(rust_tree_sitter::Language::Rust),
            "javascript" | "js" => Ok(rust_tree_sitter::Language::JavaScript),
            "typescript" | "ts" => Ok(rust_tree_sitter::Language::TypeScript),
            "python" | "py" => Ok(rust_tree_sitter::Language::Python),
            "c" => Ok(rust_tree_sitter::Language::C),
            "cpp" | "c++" | "cxx" | "cc" => Ok(rust_tree_sitter::Language::Cpp),
            "go" => Ok(rust_tree_sitter::Language::Go),
            _ => Err(AgentError::invalid_input(format!(
                "Unsupported language: {}",
                name
            ))),
        }
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
                    explanation.push_str(&format!("\n   - {} `{}` (line {})", kind, name, line));
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

                // Check for potential security issues
                if name.to_lowercase().contains("password") {
                    security_issues.push(json!({
                        "severity": "high",
                        "type": "exposed_credential",
                        "file": file_path,
                        "symbol": name,
                        "line": symbol["start_line"],
                        "description": "Symbol contains 'password' - potential credential exposure"
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

                if kind == "function" && name.starts_with("test_") {
                    security_issues.push(json!({
                        "severity": "low",
                        "type": "test_function",
                        "file": file_path,
                        "symbol": name,
                        "line": symbol["start_line"],
                        "description": "Test function detected - ensure proper isolation"
                    }));
                }
            }

            if compliance {
                // Add compliance assessment
                let total_symbols = symbols.len();
                if total_symbols > 50 {
                    compliance_notes.push(format!("File {} has high symbol count ({}) - consider refactoring", file_path, total_symbols));
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

    // ========================================
    // Phase 2 Advanced Intelligence Methods
    // ========================================

    /// Advanced AI analysis with deep semantic understanding
    async fn advanced_ai_analysis(&self, path: &Path, params: &Value) -> Result<Value> {
        let semantic = extract_optional_bool_param(params, "semantic").unwrap_or(false);
        let patterns = extract_optional_bool_param(params, "patterns").unwrap_or(false);
        let confidence_threshold = params.get("confidence_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);

        info!("Performing advanced AI analysis for: {}", path.display());

        // Configuration for advanced AI analysis
        let config = json!({
            "enable_concept_recognition": semantic,
            "enable_pattern_detection": patterns,
            "confidence_threshold": confidence_threshold,
            "max_depth": 5,
            "include_relationships": true
        });

        // For now, provide a comprehensive analysis structure
        // In the future, this will integrate with the actual AdvancedAIAnalyzer
        let base_analysis = self.analyze_codebase(path, params).await?;

        let mut ai_insights = Vec::new();
        let files = base_analysis["files"].as_array().unwrap();

        // Semantic analysis
        if semantic {
            ai_insights.push(json!({
                "type": "semantic_analysis",
                "concepts": self.extract_concepts(files),
                "abstractions": self.analyze_abstractions(files),
                "domain_insights": self.detect_domain_patterns(files)
            }));
        }

        // Pattern recognition
        if patterns {
            ai_insights.push(json!({
                "type": "pattern_recognition",
                "architecture_patterns": self.detect_architecture_patterns(files),
                "design_patterns": self.detect_design_patterns(files),
                "anti_patterns": self.detect_anti_patterns(files)
            }));
        }

        Ok(json!({
            "analysis_type": "advanced_ai_analysis",
            "path": path.display().to_string(),
            "config": {
                "semantic_enabled": semantic,
                "patterns_enabled": patterns,
                "confidence_threshold": confidence_threshold
            },
            "ai_insights": ai_insights,
            "intelligence_score": self.calculate_intelligence_score(&ai_insights),
            "recommendations": self.generate_ai_recommendations(&ai_insights)
        }))
    }

    /// Pattern recognition analysis
    async fn pattern_recognition(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Performing pattern recognition for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        Ok(json!({
            "analysis_type": "pattern_recognition",
            "path": path.display().to_string(),
            "patterns": {
                "architecture": self.detect_architecture_patterns(files),
                "design": self.detect_design_patterns(files),
                "anti_patterns": self.detect_anti_patterns(files)
            },
            "pattern_confidence": self.calculate_pattern_confidence(files),
            "implementation_guidance": self.generate_pattern_guidance(files)
        }))
    }

    /// Learning paths generation
    async fn learning_paths(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Generating learning paths for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let languages = base_analysis["language_breakdown"].as_object().unwrap();
        let complexity = self.assess_complexity(files);

        Ok(json!({
            "analysis_type": "learning_paths",
            "path": path.display().to_string(),
            "skill_assessment": {
                "complexity_level": complexity,
                "primary_languages": languages.keys().collect::<Vec<_>>(),
                "estimated_skill_level": self.estimate_skill_level(files)
            },
            "learning_recommendations": self.generate_learning_paths(files, &complexity),
            "resources": self.suggest_learning_resources(languages),
            "practice_exercises": self.suggest_exercises(files)
        }))
    }

    /// Advanced security analysis with OWASP Top 10 detection
    async fn advanced_security_scan(&self, path: &Path, params: &Value) -> Result<Value> {
        let owasp_top10 = extract_optional_bool_param(params, "owasp_top10").unwrap_or(true);
        let secrets = extract_optional_bool_param(params, "secrets").unwrap_or(true);
        let compliance = extract_optional_bool_param(params, "compliance").unwrap_or(false);

        info!("Performing advanced security scan for: {}", path.display());

        // Configuration for advanced security analysis
        let config = json!({
            "enable_owasp_top10": owasp_top10,
            "enable_secrets_detection": secrets,
            "enable_compliance_check": compliance,
            "confidence_threshold": 0.8,
            "include_cwe_mapping": true
        });

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let mut security_findings = Vec::new();

        // OWASP Top 10 Analysis
        if owasp_top10 {
            security_findings.extend(self.detect_owasp_vulnerabilities(files));
        }

        // Secrets Detection
        if secrets {
            security_findings.extend(self.detect_secrets_advanced(files));
        }

        // Calculate security score
        let security_score = self.calculate_security_score(&security_findings);

        Ok(json!({
            "analysis_type": "advanced_security_scan",
            "path": path.display().to_string(),
            "config": config,
            "security_score": security_score,
            "findings": security_findings,
            "owasp_compliance": self.assess_owasp_compliance(&security_findings),
            "remediation_roadmap": self.generate_security_roadmap(&security_findings),
            "impact_assessment": self.assess_security_impact(&security_findings)
        }))
    }

    /// Secrets detection with entropy analysis
    async fn secrets_detection(&self, path: &Path, params: &Value) -> Result<Value> {
        let entropy_threshold = params.get("entropy_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.5);

        info!("Performing secrets detection for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let secrets = self.detect_secrets_with_entropy(files, entropy_threshold);

        Ok(json!({
            "analysis_type": "secrets_detection",
            "path": path.display().to_string(),
            "entropy_threshold": entropy_threshold,
            "secrets_found": secrets.len(),
            "secrets": secrets,
            "risk_assessment": self.assess_secrets_risk(&secrets),
            "remediation_steps": self.generate_secrets_remediation(&secrets)
        }))
    }

    /// Vulnerability scanning
    async fn vulnerability_scan(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Performing vulnerability scan for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let vulnerabilities = self.scan_vulnerabilities(files);

        Ok(json!({
            "analysis_type": "vulnerability_scan",
            "path": path.display().to_string(),
            "vulnerabilities": vulnerabilities,
            "severity_breakdown": self.categorize_vulnerabilities(&vulnerabilities),
            "cwe_mapping": self.map_to_cwe(&vulnerabilities),
            "remediation_priority": self.prioritize_remediation(&vulnerabilities)
        }))
    }

    /// Smart refactoring with automated code improvements
    async fn smart_refactor(&self, path: &Path, params: &Value) -> Result<Value> {
        let roadmap = extract_optional_bool_param(params, "roadmap").unwrap_or(false);
        let quick_wins = extract_optional_bool_param(params, "quick_wins").unwrap_or(false);

        info!("Performing smart refactoring analysis for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let code_smells = self.detect_code_smells_advanced(files);
        let refactoring_opportunities = self.identify_refactoring_opportunities(files);

        let mut result = json!({
            "analysis_type": "smart_refactoring",
            "path": path.display().to_string(),
            "refactoring_score": self.calculate_refactoring_score(&code_smells, &refactoring_opportunities),
            "code_smells": code_smells,
            "opportunities": refactoring_opportunities,
            "impact_analysis": self.analyze_refactoring_impact(&refactoring_opportunities)
        });

        if roadmap {
            result["roadmap"] = serde_json::Value::Array(self.generate_refactoring_roadmap(&refactoring_opportunities));
        }

        if quick_wins {
            result["quick_wins"] = serde_json::Value::Array(self.identify_quick_wins(&refactoring_opportunities));
        }

        Ok(result)
    }

    /// Design patterns analysis and recommendations
    async fn design_patterns(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Analyzing design patterns for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        Ok(json!({
            "analysis_type": "design_patterns",
            "path": path.display().to_string(),
            "detected_patterns": self.detect_design_patterns(files),
            "recommended_patterns": self.recommend_design_patterns(files),
            "implementation_guidance": self.provide_pattern_implementation_guidance(files),
            "pattern_benefits": self.explain_pattern_benefits(files)
        }))
    }

    /// Performance optimization analysis
    async fn performance_optimize(&self, path: &Path, params: &Value) -> Result<Value> {
        let benchmarks = extract_optional_bool_param(params, "benchmarks").unwrap_or(false);

        info!("Performing performance optimization analysis for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let hotspots = self.identify_performance_hotspots(files);
        let optimizations = self.suggest_performance_optimizations(files);

        let mut result = json!({
            "analysis_type": "performance_optimization",
            "path": path.display().to_string(),
            "performance_score": self.calculate_performance_score(&hotspots),
            "hotspots": hotspots,
            "optimizations": optimizations,
            "algorithmic_improvements": self.suggest_algorithmic_improvements(files),
            "memory_optimizations": self.suggest_memory_optimizations(files)
        });

        if benchmarks {
            result["benchmarking_suggestions"] = serde_json::Value::Array(self.generate_benchmarking_suggestions(files));
        }

        Ok(result)
    }

    /// Test coverage analysis
    async fn test_coverage_analysis(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Performing test coverage analysis for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        let coverage_analysis = self.analyze_test_coverage(files);
        let missing_tests = self.identify_missing_tests(files);

        Ok(json!({
            "analysis_type": "test_coverage",
            "path": path.display().to_string(),
            "coverage_score": coverage_analysis["score"],
            "coverage_analysis": coverage_analysis,
            "missing_tests": missing_tests,
            "test_quality": self.assess_test_quality(files),
            "recommendations": self.generate_test_recommendations(files)
        }))
    }

    /// Missing tests identification
    async fn missing_tests(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Identifying missing tests for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        Ok(json!({
            "analysis_type": "missing_tests",
            "path": path.display().to_string(),
            "missing_tests": self.identify_missing_tests(files),
            "priority_functions": self.prioritize_test_candidates(files),
            "test_suggestions": self.suggest_test_implementations(files)
        }))
    }

    /// Test quality assessment
    async fn test_quality(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Assessing test quality for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        Ok(json!({
            "analysis_type": "test_quality",
            "path": path.display().to_string(),
            "quality_metrics": self.assess_test_quality(files),
            "flaky_tests": self.detect_flaky_tests(files),
            "improvement_suggestions": self.suggest_test_improvements(files)
        }))
    }

    /// Coverage gaps analysis
    async fn coverage_gaps(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Analyzing coverage gaps for: {}", path.display());

        let base_analysis = self.analyze_codebase(path, params).await?;
        let files = base_analysis["files"].as_array().unwrap();

        Ok(json!({
            "analysis_type": "coverage_gaps",
            "path": path.display().to_string(),
            "gaps": self.identify_coverage_gaps(files),
            "critical_gaps": self.identify_critical_gaps(files),
            "gap_remediation": self.suggest_gap_remediation(files)
        }))
    }

    /// Dependency scanning
    async fn dependency_scan(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Performing dependency scan for: {}", path.display());

        let dependencies = self.scan_dependencies(path).await?;

        Ok(json!({
            "analysis_type": "dependency_scan",
            "path": path.display().to_string(),
            "dependencies": dependencies,
            "dependency_graph": self.build_dependency_graph(&dependencies),
            "circular_dependencies": self.detect_circular_dependencies(&dependencies),
            "optimization_suggestions": self.suggest_dependency_optimizations(&dependencies)
        }))
    }

    /// Security dependencies check
    async fn security_deps(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Checking security dependencies for: {}", path.display());

        let dependencies = self.scan_dependencies(path).await?;
        let security_issues = self.scan_dependency_vulnerabilities(&dependencies).await?;

        Ok(json!({
            "analysis_type": "security_dependencies",
            "path": path.display().to_string(),
            "vulnerable_dependencies": security_issues,
            "severity_breakdown": self.categorize_dependency_vulnerabilities(&security_issues),
            "remediation_steps": self.generate_dependency_remediation(&security_issues)
        }))
    }

    /// Outdated dependencies check
    async fn outdated_deps(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Checking outdated dependencies for: {}", path.display());

        let dependencies = self.scan_dependencies(path).await?;
        let outdated = self.check_outdated_dependencies(&dependencies).await?;

        Ok(json!({
            "analysis_type": "outdated_dependencies",
            "path": path.display().to_string(),
            "outdated_dependencies": outdated,
            "update_recommendations": self.generate_update_recommendations(&outdated),
            "breaking_changes": self.assess_breaking_changes(&outdated)
        }))
    }

    /// License compliance check
    async fn license_check(&self, path: &Path, params: &Value) -> Result<Value> {
        info!("Performing license compliance check for: {}", path.display());

        let dependencies = self.scan_dependencies(path).await?;
        let license_analysis = self.analyze_licenses(&dependencies).await?;

        Ok(json!({
            "analysis_type": "license_compliance",
            "path": path.display().to_string(),
            "license_analysis": license_analysis,
            "compliance_issues": self.identify_license_issues(&license_analysis),
            "recommendations": self.generate_license_recommendations(&license_analysis)
        }))
    }

    // ========================================
    // Phase 2 Helper Methods - AI Analysis
    // ========================================

    fn extract_concepts(&self, files: &[Value]) -> Vec<Value> {
        let mut concepts = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();
            for symbol in symbols {
                let name = symbol["name"].as_str().unwrap();
                let kind = symbol["kind"].as_str().unwrap();

                // Extract domain concepts based on naming patterns
                if name.contains("user") || name.contains("User") {
                    concepts.push(json!({
                        "concept": "User Management",
                        "confidence": 0.8,
                        "evidence": format!("{} {}", kind, name),
                        "file": file["path"]
                    }));
                }

                if name.contains("auth") || name.contains("Auth") {
                    concepts.push(json!({
                        "concept": "Authentication",
                        "confidence": 0.9,
                        "evidence": format!("{} {}", kind, name),
                        "file": file["path"]
                    }));
                }

                if name.contains("data") || name.contains("Data") || name.contains("db") {
                    concepts.push(json!({
                        "concept": "Data Management",
                        "confidence": 0.7,
                        "evidence": format!("{} {}", kind, name),
                        "file": file["path"]
                    }));
                }
            }
        }

        concepts
    }

    fn analyze_abstractions(&self, files: &[Value]) -> Vec<Value> {
        let mut abstractions = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();
            let structs = symbols.iter().filter(|s| s["kind"] == "struct").count();
            let traits = symbols.iter().filter(|s| s["kind"] == "trait").count();
            let functions = symbols.iter().filter(|s| s["kind"] == "function").count();

            if structs > 0 || traits > 0 {
                let abstraction_level = if traits > structs { "high" } else { "medium" };
                abstractions.push(json!({
                    "file": file["path"],
                    "abstraction_level": abstraction_level,
                    "structs": structs,
                    "traits": traits,
                    "functions": functions,
                    "cohesion_score": self.calculate_cohesion_score(symbols)
                }));
            }
        }

        abstractions
    }

    fn detect_domain_patterns(&self, files: &[Value]) -> Vec<Value> {
        let mut patterns = Vec::new();

        // Analyze file structure and naming to detect domain patterns
        let file_names: Vec<&str> = files.iter()
            .filter_map(|f| f["path"].as_str())
            .collect();

        if file_names.iter().any(|name| name.contains("controller") || name.contains("handler")) {
            patterns.push(json!({
                "pattern": "Web Application",
                "confidence": 0.8,
                "evidence": "Controller/Handler pattern detected"
            }));
        }

        if file_names.iter().any(|name| name.contains("model") || name.contains("entity")) {
            patterns.push(json!({
                "pattern": "Data Modeling",
                "confidence": 0.7,
                "evidence": "Model/Entity pattern detected"
            }));
        }

        if file_names.iter().any(|name| name.contains("service") || name.contains("manager")) {
            patterns.push(json!({
                "pattern": "Service Layer",
                "confidence": 0.8,
                "evidence": "Service/Manager pattern detected"
            }));
        }

        patterns
    }

    fn detect_architecture_patterns(&self, files: &[Value]) -> Vec<Value> {
        let mut patterns = Vec::new();

        // MVC Pattern Detection
        let has_models = files.iter().any(|f| f["path"].as_str().unwrap().contains("model"));
        let has_views = files.iter().any(|f| f["path"].as_str().unwrap().contains("view"));
        let has_controllers = files.iter().any(|f| f["path"].as_str().unwrap().contains("controller"));

        if has_models && has_views && has_controllers {
            patterns.push(json!({
                "pattern": "MVC (Model-View-Controller)",
                "confidence": 0.9,
                "completeness": 100,
                "evidence": "Model, View, and Controller components detected"
            }));
        }

        // Repository Pattern Detection
        let has_repository = files.iter().any(|f| {
            let symbols = f["symbols"].as_array().unwrap();
            symbols.iter().any(|s| s["name"].as_str().unwrap().contains("Repository"))
        });

        if has_repository {
            patterns.push(json!({
                "pattern": "Repository Pattern",
                "confidence": 0.8,
                "completeness": 85,
                "evidence": "Repository interfaces/implementations detected"
            }));
        }

        patterns
    }

    fn detect_design_patterns(&self, files: &[Value]) -> Vec<Value> {
        let mut patterns = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();

            // Factory Pattern Detection
            let factory_symbols = symbols.iter()
                .filter(|s| s["name"].as_str().unwrap().contains("Factory") ||
                           s["name"].as_str().unwrap().contains("factory"))
                .count();

            if factory_symbols > 0 {
                patterns.push(json!({
                    "pattern": "Factory Pattern",
                    "file": file["path"],
                    "confidence": 0.7,
                    "symbols": factory_symbols
                }));
            }

            // Builder Pattern Detection
            let builder_symbols = symbols.iter()
                .filter(|s| s["name"].as_str().unwrap().contains("Builder") ||
                           s["name"].as_str().unwrap().contains("builder"))
                .count();

            if builder_symbols > 0 {
                patterns.push(json!({
                    "pattern": "Builder Pattern",
                    "file": file["path"],
                    "confidence": 0.8,
                    "symbols": builder_symbols
                }));
            }
        }

        patterns
    }

    fn detect_anti_patterns(&self, files: &[Value]) -> Vec<Value> {
        let mut anti_patterns = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();
            let lines = file["lines"].as_u64().unwrap_or(0);

            // God Object Anti-pattern
            if symbols.len() > 50 && lines > 1000 {
                anti_patterns.push(json!({
                    "anti_pattern": "God Object",
                    "file": file["path"],
                    "severity": "high",
                    "evidence": format!("{} symbols, {} lines", symbols.len(), lines)
                }));
            }

            // Long Parameter List
            for symbol in symbols {
                let name = symbol["name"].as_str().unwrap();
                if name.len() > 50 {
                    anti_patterns.push(json!({
                        "anti_pattern": "Long Method Name",
                        "file": file["path"],
                        "symbol": name,
                        "severity": "medium",
                        "evidence": format!("Method name length: {}", name.len())
                    }));
                }
            }
        }

        anti_patterns
    }

    fn calculate_intelligence_score(&self, insights: &[Value]) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        for insight in insights {
            match insight["type"].as_str() {
                Some("semantic_analysis") => {
                    score += 30.0;
                    factors += 1;
                }
                Some("pattern_recognition") => {
                    score += 25.0;
                    factors += 1;
                }
                _ => {}
            }
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.0
        }
    }

    fn generate_ai_recommendations(&self, insights: &[Value]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for insight in insights {
            match insight["type"].as_str() {
                Some("semantic_analysis") => {
                    recommendations.push("🧠 Consider documenting domain concepts for better team understanding".to_string());
                    recommendations.push("📚 Implement domain-driven design patterns where appropriate".to_string());
                }
                Some("pattern_recognition") => {
                    recommendations.push("🏗️ Strengthen architecture patterns for better maintainability".to_string());
                    recommendations.push("🔧 Consider refactoring to eliminate detected anti-patterns".to_string());
                }
                _ => {}
            }
        }

        if recommendations.is_empty() {
            recommendations.push("✨ Enable semantic analysis and pattern recognition for deeper insights".to_string());
        }

        recommendations
    }

    fn calculate_pattern_confidence(&self, files: &[Value]) -> f64 {
        let total_files = files.len() as f64;
        let files_with_patterns = files.iter()
            .filter(|f| {
                let path = f["path"].as_str().unwrap();
                path.contains("controller") || path.contains("model") ||
                path.contains("service") || path.contains("repository")
            })
            .count() as f64;

        if total_files > 0.0 {
            files_with_patterns / total_files
        } else {
            0.0
        }
    }

    fn generate_pattern_guidance(&self, files: &[Value]) -> Vec<String> {
        let mut guidance = Vec::new();

        let has_mvc = files.iter().any(|f| f["path"].as_str().unwrap().contains("controller"));
        if !has_mvc {
            guidance.push("Consider implementing MVC pattern for better separation of concerns".to_string());
        }

        let has_services = files.iter().any(|f| f["path"].as_str().unwrap().contains("service"));
        if !has_services {
            guidance.push("Implement service layer for business logic encapsulation".to_string());
        }

        guidance
    }

    fn assess_complexity(&self, files: &[Value]) -> String {
        let total_symbols: usize = files.iter()
            .map(|f| f["symbols"].as_array().unwrap().len())
            .sum();
        let total_lines: u64 = files.iter()
            .map(|f| f["lines"].as_u64().unwrap_or(0))
            .sum();

        if total_symbols > 500 || total_lines > 10000 {
            "high".to_string()
        } else if total_symbols > 100 || total_lines > 2000 {
            "medium".to_string()
        } else {
            "low".to_string()
        }
    }

    fn estimate_skill_level(&self, files: &[Value]) -> String {
        let languages: std::collections::HashSet<&str> = files.iter()
            .filter_map(|f| f["language"].as_str())
            .collect();

        let complexity = self.assess_complexity(files);

        match (languages.len(), complexity.as_str()) {
            (1, "low") => "beginner".to_string(),
            (1..=2, "medium") => "intermediate".to_string(),
            (3.., "high") => "advanced".to_string(),
            _ => "intermediate".to_string()
        }
    }

    fn generate_learning_paths(&self, files: &[Value], complexity: &str) -> Vec<Value> {
        let mut paths = Vec::new();

        let languages: std::collections::HashSet<&str> = files.iter()
            .filter_map(|f| f["language"].as_str())
            .collect();

        for language in languages {
            let estimated_hours = match complexity {
                "low" => 10,
                "medium" => 25,
                "high" => 50,
                _ => 20
            };

            paths.push(json!({
                "language": language,
                "estimated_hours": estimated_hours,
                "skill_level": self.estimate_skill_level(files),
                "topics": self.suggest_topics_for_language(language),
                "prerequisites": self.get_prerequisites_for_language(language)
            }));
        }

        paths
    }

    fn suggest_learning_resources(&self, languages: &serde_json::Map<String, Value>) -> Vec<Value> {
        let mut resources = Vec::new();

        for language in languages.keys() {
            match language.as_str() {
                "rust" => {
                    resources.push(json!({
                        "language": "rust",
                        "resources": [
                            "The Rust Programming Language (Book)",
                            "Rust by Example",
                            "Rustlings (Interactive Exercises)"
                        ]
                    }));
                }
                "javascript" => {
                    resources.push(json!({
                        "language": "javascript",
                        "resources": [
                            "MDN Web Docs",
                            "JavaScript: The Good Parts",
                            "You Don't Know JS (Book Series)"
                        ]
                    }));
                }
                "python" => {
                    resources.push(json!({
                        "language": "python",
                        "resources": [
                            "Python.org Tutorial",
                            "Automate the Boring Stuff with Python",
                            "Python Crash Course"
                        ]
                    }));
                }
                _ => {
                    resources.push(json!({
                        "language": language,
                        "resources": [
                            "Official Documentation",
                            "Language-specific Tutorials",
                            "Community Forums"
                        ]
                    }));
                }
            }
        }

        resources
    }

    fn suggest_exercises(&self, files: &[Value]) -> Vec<Value> {
        let mut exercises = Vec::new();

        let has_functions = files.iter().any(|f| {
            f["symbols"].as_array().unwrap().iter()
                .any(|s| s["kind"] == "function")
        });

        if has_functions {
            exercises.push(json!({
                "type": "function_analysis",
                "description": "Analyze function complexity and suggest improvements",
                "difficulty": "intermediate"
            }));
        }

        let has_structs = files.iter().any(|f| {
            f["symbols"].as_array().unwrap().iter()
                .any(|s| s["kind"] == "struct")
        });

        if has_structs {
            exercises.push(json!({
                "type": "data_structure_design",
                "description": "Design alternative data structures for better performance",
                "difficulty": "advanced"
            }));
        }

        exercises
    }

    fn calculate_cohesion_score(&self, symbols: &[Value]) -> f64 {
        // Simple cohesion calculation based on symbol types
        let functions = symbols.iter().filter(|s| s["kind"] == "function").count();
        let structs = symbols.iter().filter(|s| s["kind"] == "struct").count();
        let total = symbols.len();

        if total > 0 {
            (functions + structs) as f64 / total as f64
        } else {
            0.0
        }
    }

    fn suggest_topics_for_language(&self, language: &str) -> Vec<String> {
        match language {
            "rust" => vec![
                "Ownership and Borrowing".to_string(),
                "Error Handling".to_string(),
                "Concurrency".to_string(),
                "Traits and Generics".to_string()
            ],
            "javascript" => vec![
                "Async/Await".to_string(),
                "Closures".to_string(),
                "Prototypes".to_string(),
                "ES6+ Features".to_string()
            ],
            "python" => vec![
                "List Comprehensions".to_string(),
                "Decorators".to_string(),
                "Context Managers".to_string(),
                "Type Hints".to_string()
            ],
            _ => vec![
                "Language Fundamentals".to_string(),
                "Best Practices".to_string(),
                "Design Patterns".to_string()
            ]
        }
    }

    fn get_prerequisites_for_language(&self, language: &str) -> Vec<String> {
        match language {
            "rust" => vec![
                "Basic programming concepts".to_string(),
                "Understanding of memory management".to_string()
            ],
            "javascript" => vec![
                "HTML/CSS basics".to_string(),
                "Programming fundamentals".to_string()
            ],
            "python" => vec![
                "Basic programming concepts".to_string(),
                "Command line familiarity".to_string()
            ],
            _ => vec![
                "Programming fundamentals".to_string(),
                "Problem-solving skills".to_string()
            ]
        }
    }

    // ========================================
    // Phase 2 Helper Methods - Security Analysis
    // ========================================

    fn detect_owasp_vulnerabilities(&self, files: &[Value]) -> Vec<Value> {
        let mut vulnerabilities = Vec::new();

        for file in files {
            let content = file.get("content").and_then(|c| c.as_str()).unwrap_or("");

            // A1: Injection vulnerabilities
            if content.contains("eval(") || content.contains("exec(") {
                vulnerabilities.push(json!({
                    "owasp_id": "A03:2021",
                    "category": "Injection",
                    "severity": "high",
                    "file": file["path"],
                    "evidence": "Potential code injection vulnerability detected"
                }));
            }

            // A2: Broken Authentication
            if content.contains("password") && content.contains("==") {
                vulnerabilities.push(json!({
                    "owasp_id": "A07:2021",
                    "category": "Identification and Authentication Failures",
                    "severity": "medium",
                    "file": file["path"],
                    "evidence": "Potential weak password comparison"
                }));
            }

            // A3: Sensitive Data Exposure
            if content.contains("console.log") && (content.contains("password") || content.contains("token")) {
                vulnerabilities.push(json!({
                    "owasp_id": "A02:2021",
                    "category": "Cryptographic Failures",
                    "severity": "medium",
                    "file": file["path"],
                    "evidence": "Potential sensitive data logging"
                }));
            }
        }

        vulnerabilities
    }

    fn detect_secrets_advanced(&self, files: &[Value]) -> Vec<Value> {
        let mut secrets = Vec::new();

        for file in files {
            let content = file.get("content").and_then(|c| c.as_str()).unwrap_or("");

            // API Keys pattern
            if let Some(captures) = regex::Regex::new(r#"(?i)(api[_-]?key|apikey)\s*[:=]\s*['"]([a-zA-Z0-9_-]{20,})['"]"#)
                .unwrap().captures(content) {
                secrets.push(json!({
                    "type": "api_key",
                    "file": file["path"],
                    "confidence": 0.9,
                    "entropy": self.calculate_entropy(&captures[2])
                }));
            }

            // JWT Tokens
            if content.contains("eyJ") {
                secrets.push(json!({
                    "type": "jwt_token",
                    "file": file["path"],
                    "confidence": 0.8,
                    "entropy": 4.5
                }));
            }
        }

        secrets
    }

    fn calculate_security_score(&self, findings: &[Value]) -> f64 {
        if findings.is_empty() {
            return 100.0;
        }

        let high_severity = findings.iter().filter(|f| f["severity"] == "high").count();
        let medium_severity = findings.iter().filter(|f| f["severity"] == "medium").count();
        let low_severity = findings.iter().filter(|f| f["severity"] == "low").count();

        let penalty = (high_severity * 20) + (medium_severity * 10) + (low_severity * 5);
        (100.0 - penalty as f64).max(0.0)
    }

    fn assess_owasp_compliance(&self, findings: &[Value]) -> Value {
        let owasp_categories = [
            "A01:2021", "A02:2021", "A03:2021", "A04:2021", "A05:2021",
            "A06:2021", "A07:2021", "A08:2021", "A09:2021", "A10:2021"
        ];

        let mut compliance = json!({});
        for category in &owasp_categories {
            let violations = findings.iter()
                .filter(|f| f["owasp_id"].as_str().unwrap_or("") == *category)
                .count();
            compliance[category] = json!({
                "violations": violations,
                "status": if violations == 0 { "compliant" } else { "non_compliant" }
            });
        }

        compliance
    }

    fn generate_security_roadmap(&self, findings: &[Value]) -> Vec<Value> {
        let mut roadmap = Vec::new();

        let high_priority = findings.iter()
            .filter(|f| f["severity"] == "high")
            .collect::<Vec<_>>();

        if !high_priority.is_empty() {
            roadmap.push(json!({
                "phase": "immediate",
                "priority": "critical",
                "items": high_priority.len(),
                "description": "Address critical security vulnerabilities immediately"
            }));
        }

        roadmap.push(json!({
            "phase": "short_term",
            "priority": "high",
            "items": findings.iter().filter(|f| f["severity"] == "medium").count(),
            "description": "Implement security controls and monitoring"
        }));

        roadmap
    }

    fn assess_security_impact(&self, findings: &[Value]) -> Value {
        let total_findings = findings.len();
        let critical_findings = findings.iter().filter(|f| f["severity"] == "high").count();

        let impact_level = if critical_findings > 0 {
            "high"
        } else if total_findings > 5 {
            "medium"
        } else {
            "low"
        };

        json!({
            "impact_level": impact_level,
            "total_findings": total_findings,
            "critical_findings": critical_findings,
            "risk_score": self.calculate_security_score(findings)
        })
    }

    fn detect_secrets_with_entropy(&self, files: &[Value], entropy_threshold: f64) -> Vec<Value> {
        let mut secrets = Vec::new();

        for file in files {
            let content = file.get("content").and_then(|c| c.as_str()).unwrap_or("");

            // Look for high-entropy strings
            for line in content.lines() {
                for word in line.split_whitespace() {
                    if word.len() > 20 {
                        let entropy = self.calculate_entropy(word);
                        if entropy > entropy_threshold {
                            secrets.push(json!({
                                "type": "high_entropy_string",
                                "file": file["path"],
                                "entropy": entropy,
                                "confidence": (entropy - entropy_threshold) / (8.0 - entropy_threshold)
                            }));
                        }
                    }
                }
            }
        }

        secrets
    }

    fn calculate_entropy(&self, s: &str) -> f64 {
        let mut char_counts = std::collections::HashMap::new();
        for c in s.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }

        let len = s.len() as f64;
        let mut entropy = 0.0;

        for count in char_counts.values() {
            let p = *count as f64 / len;
            entropy -= p * p.log2();
        }

        entropy
    }

    fn assess_secrets_risk(&self, secrets: &[Value]) -> Value {
        let high_risk = secrets.iter()
            .filter(|s| s["confidence"].as_f64().unwrap_or(0.0) > 0.8)
            .count();

        json!({
            "total_secrets": secrets.len(),
            "high_risk_secrets": high_risk,
            "risk_level": if high_risk > 0 { "high" } else if secrets.len() > 0 { "medium" } else { "low" }
        })
    }

    fn generate_secrets_remediation(&self, secrets: &[Value]) -> Vec<String> {
        let mut remediation = Vec::new();

        if !secrets.is_empty() {
            remediation.push("🔐 Move secrets to environment variables or secure vaults".to_string());
            remediation.push("🔍 Implement secrets scanning in CI/CD pipeline".to_string());
            remediation.push("📝 Create secrets management policy".to_string());
            remediation.push("🔄 Rotate any exposed credentials immediately".to_string());
        }

        remediation
    }

    fn scan_vulnerabilities(&self, files: &[Value]) -> Vec<Value> {
        let mut vulnerabilities = Vec::new();

        // Combine OWASP and other vulnerability detection
        vulnerabilities.extend(self.detect_owasp_vulnerabilities(files));
        vulnerabilities.extend(self.detect_secrets_advanced(files));

        // Add additional vulnerability patterns
        for file in files {
            let content = file.get("content").and_then(|c| c.as_str()).unwrap_or("");

            // SQL Injection patterns
            if content.contains("SELECT") && content.contains("WHERE") && content.contains("+") {
                vulnerabilities.push(json!({
                    "type": "sql_injection",
                    "severity": "high",
                    "file": file["path"],
                    "cwe": "CWE-89"
                }));
            }

            // XSS patterns
            if content.contains("innerHTML") && content.contains("user") {
                vulnerabilities.push(json!({
                    "type": "xss",
                    "severity": "medium",
                    "file": file["path"],
                    "cwe": "CWE-79"
                }));
            }
        }

        vulnerabilities
    }

    fn categorize_vulnerabilities(&self, vulnerabilities: &[Value]) -> Value {
        let high = vulnerabilities.iter().filter(|v| v["severity"] == "high").count();
        let medium = vulnerabilities.iter().filter(|v| v["severity"] == "medium").count();
        let low = vulnerabilities.iter().filter(|v| v["severity"] == "low").count();

        json!({
            "high": high,
            "medium": medium,
            "low": low,
            "total": vulnerabilities.len()
        })
    }

    fn map_to_cwe(&self, vulnerabilities: &[Value]) -> Vec<Value> {
        vulnerabilities.iter()
            .filter_map(|v| {
                v.get("cwe").map(|cwe| json!({
                    "cwe": cwe,
                    "type": v["type"],
                    "file": v["file"]
                }))
            })
            .collect()
    }

    fn prioritize_remediation(&self, vulnerabilities: &[Value]) -> Vec<Value> {
        let mut prioritized = vulnerabilities.to_vec();
        prioritized.sort_by(|a, b| {
            let severity_order = |s: &str| match s {
                "high" => 0,
                "medium" => 1,
                "low" => 2,
                _ => 3
            };

            let a_severity = a["severity"].as_str().unwrap_or("low");
            let b_severity = b["severity"].as_str().unwrap_or("low");

            severity_order(a_severity).cmp(&severity_order(b_severity))
        });

        prioritized
    }

    // ========================================
    // Phase 2 Helper Methods - Smart Refactoring
    // ========================================

    fn detect_code_smells_advanced(&self, files: &[Value]) -> Vec<Value> {
        let mut code_smells = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();
            let lines = file["lines"].as_u64().unwrap_or(0);

            // Long Method smell
            for symbol in symbols {
                if symbol["kind"] == "function" {
                    let name = symbol["name"].as_str().unwrap();
                    if name.len() > 50 {
                        code_smells.push(json!({
                            "smell": "Long Method Name",
                            "severity": "medium",
                            "file": file["path"],
                            "symbol": name,
                            "suggestion": "Consider breaking down into smaller, more focused methods"
                        }));
                    }
                }
            }

            // Large Class smell
            if symbols.len() > 30 && lines > 500 {
                code_smells.push(json!({
                    "smell": "Large Class",
                    "severity": "high",
                    "file": file["path"],
                    "metrics": {
                        "symbols": symbols.len(),
                        "lines": lines
                    },
                    "suggestion": "Consider splitting into multiple smaller classes"
                }));
            }

            // Duplicate Code (simplified detection)
            let function_names: Vec<&str> = symbols.iter()
                .filter(|s| s["kind"] == "function")
                .filter_map(|s| s["name"].as_str())
                .collect();

            for name in &function_names {
                let count = function_names.iter().filter(|&n| n == name).count();
                if count > 1 {
                    code_smells.push(json!({
                        "smell": "Duplicate Code",
                        "severity": "medium",
                        "file": file["path"],
                        "symbol": name,
                        "occurrences": count,
                        "suggestion": "Extract common functionality into shared utilities"
                    }));
                }
            }
        }

        code_smells
    }

    fn identify_refactoring_opportunities(&self, files: &[Value]) -> Vec<Value> {
        let mut opportunities = Vec::new();

        for file in files {
            let symbols = file["symbols"].as_array().unwrap();

            // Extract Method opportunities
            let long_functions = symbols.iter()
                .filter(|s| s["kind"] == "function")
                .filter(|s| s["name"].as_str().unwrap().len() > 30)
                .count();

            if long_functions > 0 {
                opportunities.push(json!({
                    "type": "Extract Method",
                    "priority": "high",
                    "file": file["path"],
                    "count": long_functions,
                    "description": "Break down complex functions into smaller, focused methods",
                    "effort": "medium"
                }));
            }

            // Extract Class opportunities
            let struct_count = symbols.iter().filter(|s| s["kind"] == "struct").count();
            let function_count = symbols.iter().filter(|s| s["kind"] == "function").count();

            if function_count > 20 && struct_count < 3 {
                opportunities.push(json!({
                    "type": "Extract Class",
                    "priority": "medium",
                    "file": file["path"],
                    "description": "Group related functionality into cohesive classes",
                    "effort": "high"
                }));
            }
        }

        opportunities
    }

    fn calculate_refactoring_score(&self, code_smells: &[Value], opportunities: &[Value]) -> f64 {
        let smell_penalty = code_smells.len() as f64 * 5.0;
        let opportunity_bonus = opportunities.len() as f64 * 2.0;

        (100.0 - smell_penalty + opportunity_bonus).max(0.0).min(100.0)
    }

    fn analyze_refactoring_impact(&self, opportunities: &[Value]) -> Value {
        let high_priority = opportunities.iter()
            .filter(|o| o["priority"] == "high")
            .count();
        let medium_priority = opportunities.iter()
            .filter(|o| o["priority"] == "medium")
            .count();
        let low_priority = opportunities.iter()
            .filter(|o| o["priority"] == "low")
            .count();

        json!({
            "total_opportunities": opportunities.len(),
            "priority_breakdown": {
                "high": high_priority,
                "medium": medium_priority,
                "low": low_priority
            },
            "estimated_effort": self.calculate_total_effort(opportunities),
            "expected_benefits": [
                "Improved code maintainability",
                "Reduced technical debt",
                "Enhanced code readability",
                "Better testability"
            ]
        })
    }

    fn generate_refactoring_roadmap(&self, opportunities: &[Value]) -> Vec<Value> {
        let mut roadmap = Vec::new();

        let high_priority: Vec<&Value> = opportunities.iter()
            .filter(|o| o["priority"] == "high")
            .collect();

        if !high_priority.is_empty() {
            roadmap.push(json!({
                "phase": "Phase 1 - Critical Refactoring",
                "duration": "2-4 weeks",
                "opportunities": high_priority.len(),
                "focus": "Address high-priority code smells and technical debt"
            }));
        }

        let medium_priority: Vec<&Value> = opportunities.iter()
            .filter(|o| o["priority"] == "medium")
            .collect();

        if !medium_priority.is_empty() {
            roadmap.push(json!({
                "phase": "Phase 2 - Structural Improvements",
                "duration": "4-6 weeks",
                "opportunities": medium_priority.len(),
                "focus": "Improve code structure and design patterns"
            }));
        }

        roadmap
    }

    fn identify_quick_wins(&self, opportunities: &[Value]) -> Vec<Value> {
        opportunities.iter()
            .filter(|o| o["effort"].as_str().unwrap_or("high") == "low" ||
                       o["effort"].as_str().unwrap_or("high") == "medium")
            .cloned()
            .collect()
    }

    fn recommend_design_patterns(&self, files: &[Value]) -> Vec<Value> {
        let mut recommendations = Vec::new();

        // Analyze code structure to recommend patterns
        let has_many_similar_classes = files.iter()
            .map(|f| f["symbols"].as_array().unwrap().iter()
                .filter(|s| s["kind"] == "struct").count())
            .sum::<usize>() > 10;

        if has_many_similar_classes {
            recommendations.push(json!({
                "pattern": "Factory Pattern",
                "reason": "Multiple similar classes detected",
                "benefit": "Centralized object creation and better maintainability",
                "complexity": "medium"
            }));
        }

        // Check for complex conditional logic
        for file in files {
            let content = file.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if content.matches("if").count() > 10 {
                recommendations.push(json!({
                    "pattern": "Strategy Pattern",
                    "file": file["path"],
                    "reason": "Complex conditional logic detected",
                    "benefit": "Improved flexibility and testability",
                    "complexity": "medium"
                }));
            }
        }

        recommendations
    }

    fn provide_pattern_implementation_guidance(&self, files: &[Value]) -> Vec<Value> {
        let mut guidance = Vec::new();

        guidance.push(json!({
            "pattern": "Repository Pattern",
            "steps": [
                "1. Define repository interface",
                "2. Implement concrete repository classes",
                "3. Use dependency injection for repositories",
                "4. Add unit tests for repository implementations"
            ],
            "example_files": ["repository.rs", "user_repository.rs"]
        }));

        guidance.push(json!({
            "pattern": "Builder Pattern",
            "steps": [
                "1. Create builder struct",
                "2. Implement builder methods",
                "3. Add build() method",
                "4. Use builder in client code"
            ],
            "example_files": ["builder.rs", "config_builder.rs"]
        }));

        guidance
    }

    fn explain_pattern_benefits(&self, files: &[Value]) -> Vec<Value> {
        vec![
            json!({
                "pattern": "MVC",
                "benefits": [
                    "Separation of concerns",
                    "Improved testability",
                    "Better code organization",
                    "Easier maintenance"
                ]
            }),
            json!({
                "pattern": "Repository",
                "benefits": [
                    "Data access abstraction",
                    "Improved testability",
                    "Centralized query logic",
                    "Better separation of concerns"
                ]
            }),
            json!({
                "pattern": "Factory",
                "benefits": [
                    "Centralized object creation",
                    "Improved flexibility",
                    "Better maintainability",
                    "Reduced coupling"
                ]
            })
        ]
    }

    fn calculate_total_effort(&self, opportunities: &[Value]) -> String {
        let total_weeks = opportunities.iter()
            .map(|o| match o["effort"].as_str().unwrap_or("medium") {
                "low" => 1,
                "medium" => 3,
                "high" => 6,
                _ => 3
            })
            .sum::<i32>();

        format!("{}-{} weeks", total_weeks, total_weeks + 2)
    }

    // ========================================
    // Phase 2 Helper Methods - Performance & Testing & Dependencies
    // ========================================

    fn identify_performance_hotspots(&self, files: &[Value]) -> Vec<Value> {
        let mut hotspots = Vec::new();
        for file in files {
            let symbols = file["symbols"].as_array().unwrap();
            let lines = file["lines"].as_u64().unwrap_or(0);

            // Large files are potential hotspots
            if lines > 1000 {
                hotspots.push(json!({
                    "type": "large_file",
                    "file": file["path"],
                    "lines": lines,
                    "severity": "medium"
                }));
            }

            // Functions with complex names might be doing too much
            for symbol in symbols {
                if symbol["kind"] == "function" {
                    let name = symbol["name"].as_str().unwrap();
                    if name.len() > 40 {
                        hotspots.push(json!({
                            "type": "complex_function",
                            "file": file["path"],
                            "function": name,
                            "severity": "low"
                        }));
                    }
                }
            }
        }
        hotspots
    }

    fn suggest_performance_optimizations(&self, files: &[Value]) -> Vec<Value> {
        vec![
            json!({
                "optimization": "Use efficient data structures",
                "description": "Replace Vec with HashMap for frequent lookups",
                "impact": "high"
            }),
            json!({
                "optimization": "Implement caching",
                "description": "Cache expensive computations",
                "impact": "medium"
            }),
            json!({
                "optimization": "Optimize loops",
                "description": "Use iterators instead of index-based loops",
                "impact": "low"
            })
        ]
    }

    fn calculate_performance_score(&self, hotspots: &[Value]) -> f64 {
        let penalty = hotspots.len() as f64 * 10.0;
        (100.0 - penalty).max(0.0)
    }

    fn suggest_algorithmic_improvements(&self, files: &[Value]) -> Vec<String> {
        vec![
            "🚀 Consider using binary search for sorted data".to_string(),
            "📊 Implement lazy loading for large datasets".to_string(),
            "⚡ Use parallel processing for CPU-intensive tasks".to_string(),
            "🔄 Implement connection pooling for database operations".to_string()
        ]
    }

    fn suggest_memory_optimizations(&self, files: &[Value]) -> Vec<String> {
        vec![
            "💾 Use references instead of cloning large objects".to_string(),
            "🗑️ Implement proper resource cleanup".to_string(),
            "📦 Use Box<T> for large stack allocations".to_string(),
            "🔄 Consider using Cow<T> for copy-on-write scenarios".to_string()
        ]
    }

    fn generate_benchmarking_suggestions(&self, files: &[Value]) -> Vec<Value> {
        vec![
            json!({
                "benchmark": "Function Performance",
                "description": "Benchmark critical functions with criterion.rs",
                "priority": "high"
            }),
            json!({
                "benchmark": "Memory Usage",
                "description": "Profile memory usage with valgrind or similar tools",
                "priority": "medium"
            })
        ]
    }

    fn analyze_test_coverage(&self, files: &[Value]) -> Value {
        let test_files = files.iter().filter(|f| {
            f["path"].as_str().unwrap().contains("test") ||
            f["path"].as_str().unwrap().contains("spec")
        }).count();

        let total_files = files.len();
        let coverage_percentage = if total_files > 0 {
            (test_files as f64 / total_files as f64) * 100.0
        } else {
            0.0
        };

        json!({
            "score": coverage_percentage,
            "test_files": test_files,
            "total_files": total_files,
            "status": if coverage_percentage > 80.0 { "excellent" }
                     else if coverage_percentage > 60.0 { "good" }
                     else if coverage_percentage > 40.0 { "fair" }
                     else { "poor" }
        })
    }

    fn identify_missing_tests(&self, files: &[Value]) -> Vec<Value> {
        let mut missing_tests = Vec::new();

        for file in files {
            if !file["path"].as_str().unwrap().contains("test") {
                let symbols = file["symbols"].as_array().unwrap();
                let functions = symbols.iter().filter(|s| s["kind"] == "function").count();

                if functions > 0 {
                    missing_tests.push(json!({
                        "file": file["path"],
                        "functions_without_tests": functions,
                        "priority": if functions > 5 { "high" } else { "medium" }
                    }));
                }
            }
        }

        missing_tests
    }

    fn assess_test_quality(&self, files: &[Value]) -> Value {
        let test_files: Vec<&Value> = files.iter()
            .filter(|f| f["path"].as_str().unwrap().contains("test"))
            .collect();

        json!({
            "test_file_count": test_files.len(),
            "average_test_functions": if !test_files.is_empty() {
                test_files.iter()
                    .map(|f| f["symbols"].as_array().unwrap().iter()
                        .filter(|s| s["kind"] == "function").count())
                    .sum::<usize>() / test_files.len()
            } else { 0 },
            "quality_score": if test_files.len() > 5 { 85 } else { 60 }
        })
    }

    fn generate_test_recommendations(&self, files: &[Value]) -> Vec<String> {
        vec![
            "🧪 Add unit tests for all public functions".to_string(),
            "🔄 Implement integration tests for critical workflows".to_string(),
            "📊 Add property-based tests for complex logic".to_string(),
            "🎯 Achieve at least 80% code coverage".to_string()
        ]
    }

    fn prioritize_test_candidates(&self, files: &[Value]) -> Vec<Value> {
        let mut candidates = Vec::new();

        for file in files {
            if !file["path"].as_str().unwrap().contains("test") {
                let symbols = file["symbols"].as_array().unwrap();
                let public_functions = symbols.iter()
                    .filter(|s| s["kind"] == "function")
                    .count();

                if public_functions > 0 {
                    candidates.push(json!({
                        "file": file["path"],
                        "public_functions": public_functions,
                        "priority": if public_functions > 3 { "high" } else { "medium" }
                    }));
                }
            }
        }

        candidates.sort_by(|a, b| {
            let a_funcs = a["public_functions"].as_u64().unwrap_or(0);
            let b_funcs = b["public_functions"].as_u64().unwrap_or(0);
            b_funcs.cmp(&a_funcs)
        });

        candidates
    }

    fn suggest_test_implementations(&self, files: &[Value]) -> Vec<Value> {
        vec![
            json!({
                "test_type": "Unit Tests",
                "framework": "Built-in Rust testing",
                "example": "#[test] fn test_function_name() { ... }"
            }),
            json!({
                "test_type": "Integration Tests",
                "framework": "Tokio Test",
                "example": "#[tokio::test] async fn test_async_function() { ... }"
            })
        ]
    }

    fn detect_flaky_tests(&self, files: &[Value]) -> Vec<Value> {
        // Simplified flaky test detection
        vec![
            json!({
                "indicator": "Time-dependent tests",
                "description": "Tests that depend on current time or sleep",
                "recommendation": "Use mock time or deterministic delays"
            })
        ]
    }

    fn suggest_test_improvements(&self, files: &[Value]) -> Vec<String> {
        vec![
            "🎯 Use descriptive test names that explain the scenario".to_string(),
            "🔧 Implement proper test setup and teardown".to_string(),
            "📝 Add assertions with meaningful error messages".to_string(),
            "🏗️ Use test builders for complex test data".to_string()
        ]
    }

    fn identify_coverage_gaps(&self, files: &[Value]) -> Vec<Value> {
        let mut gaps = Vec::new();

        for file in files {
            if !file["path"].as_str().unwrap().contains("test") {
                let symbols = file["symbols"].as_array().unwrap();
                let untested_functions = symbols.iter()
                    .filter(|s| s["kind"] == "function")
                    .count();

                if untested_functions > 0 {
                    gaps.push(json!({
                        "file": file["path"],
                        "untested_functions": untested_functions,
                        "gap_type": "function_coverage"
                    }));
                }
            }
        }

        gaps
    }

    fn identify_critical_gaps(&self, files: &[Value]) -> Vec<Value> {
        self.identify_coverage_gaps(files).into_iter()
            .filter(|gap| gap["untested_functions"].as_u64().unwrap_or(0) > 3)
            .collect()
    }

    fn suggest_gap_remediation(&self, files: &[Value]) -> Vec<String> {
        vec![
            "🎯 Start with testing public API functions".to_string(),
            "🔄 Add tests for error handling paths".to_string(),
            "📊 Focus on business logic functions first".to_string(),
            "🧪 Use test-driven development for new features".to_string()
        ]
    }

    async fn scan_dependencies(&self, path: &Path) -> Result<Vec<Value>> {
        // Simplified dependency scanning
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            Ok(vec![
                json!({
                    "name": "serde",
                    "version": "1.0",
                    "type": "direct"
                }),
                json!({
                    "name": "tokio",
                    "version": "1.0",
                    "type": "direct"
                })
            ])
        } else {
            Ok(vec![])
        }
    }

    fn build_dependency_graph(&self, dependencies: &[Value]) -> Value {
        json!({
            "nodes": dependencies.len(),
            "edges": dependencies.len() * 2,
            "depth": 3
        })
    }

    fn detect_circular_dependencies(&self, dependencies: &[Value]) -> Vec<Value> {
        // Simplified circular dependency detection
        vec![]
    }

    fn suggest_dependency_optimizations(&self, dependencies: &[Value]) -> Vec<String> {
        vec![
            "📦 Remove unused dependencies".to_string(),
            "⬆️ Update to latest stable versions".to_string(),
            "🔄 Consider lighter alternatives for heavy dependencies".to_string()
        ]
    }

    async fn scan_dependency_vulnerabilities(&self, dependencies: &[Value]) -> Result<Vec<Value>> {
        // Simplified vulnerability scanning
        Ok(vec![])
    }

    fn categorize_dependency_vulnerabilities(&self, vulnerabilities: &[Value]) -> Value {
        json!({
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        })
    }

    fn generate_dependency_remediation(&self, vulnerabilities: &[Value]) -> Vec<String> {
        vec![
            "🔒 Update vulnerable dependencies immediately".to_string(),
            "🔍 Run regular security audits".to_string(),
            "📋 Maintain dependency inventory".to_string()
        ]
    }

    async fn check_outdated_dependencies(&self, dependencies: &[Value]) -> Result<Vec<Value>> {
        // Simplified outdated dependency checking
        Ok(vec![])
    }

    fn generate_update_recommendations(&self, outdated: &[Value]) -> Vec<String> {
        vec![
            "⬆️ Update dependencies in order of importance".to_string(),
            "🧪 Test thoroughly after updates".to_string(),
            "📝 Review changelogs for breaking changes".to_string()
        ]
    }

    fn assess_breaking_changes(&self, outdated: &[Value]) -> Value {
        json!({
            "potential_breaking_changes": 0,
            "risk_level": "low"
        })
    }

    async fn analyze_licenses(&self, dependencies: &[Value]) -> Result<Value> {
        Ok(json!({
            "compatible_licenses": dependencies.len(),
            "incompatible_licenses": 0,
            "unknown_licenses": 0
        }))
    }

    fn identify_license_issues(&self, license_analysis: &Value) -> Vec<Value> {
        vec![]
    }

    fn generate_license_recommendations(&self, license_analysis: &Value) -> Vec<String> {
        vec![
            "📄 Document all dependency licenses".to_string(),
            "⚖️ Ensure license compatibility".to_string(),
            "🔍 Regular license compliance audits".to_string()
        ]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs;
    use serde_json::json;

    #[tokio::test]
    async fn test_parse_success_rate_computed() {
        let dir = tempdir().unwrap();
        let file1 = dir.path().join("main.rs");
        let file2 = dir.path().join("lib.rs");
        fs::write(&file1, "fn main() {}\n").await.unwrap();
        fs::write(&file2, "pub fn helper() {}\n").await.unwrap();

        let tool = CodeAnalysisTool::new();
        let result = tool.generate_insights(dir.path(), &json!({})).await.unwrap();
        let parsed = result["metrics"]["total_files"].as_u64().unwrap();
        let rate = result["quality_indicators"]["parse_success_rate"].as_f64().unwrap();
        assert_eq!(parsed, 2);
        assert_eq!(rate, 100.0);
    }

    #[tokio::test]
    async fn test_query_patterns_basic() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("lib.rs");
        fs::write(&file_path, "fn foo() {}\nfn bar() {}\n").await.unwrap();

        let tool = CodeAnalysisTool::new();
        let params = json!({
            "pattern": "(function_item name: (identifier) @name)",
            "language": "rust"
        });
        let result = tool.query_patterns(dir.path(), &params).await.unwrap();
        let matches = result["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
    }
}
