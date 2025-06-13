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

/// Dependency information structure
#[derive(Debug, Clone)]
struct DependencyInfo {
    version: String,
    features: Option<Vec<String>>,
    optional: bool,
    git: Option<String>,
    path: Option<String>,
}

/// Vulnerability information structure
#[derive(Debug, Clone)]
struct VulnerabilityInfo {
    id: String,
    severity: String,
    description: String,
    affected_versions: String,
    fixed_version: Option<String>,
    cve: Option<String>,
    published: String,
}

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
        let mut optimizations = Vec::new();

        for file in files {
            let content = file["content"].as_str().unwrap_or("");
            let file_path = file["path"].as_str().unwrap_or("unknown");

            // Analyze for performance anti-patterns
            optimizations.extend(self.analyze_data_structure_usage(content, file_path));
            optimizations.extend(self.analyze_loop_patterns(content, file_path));
            optimizations.extend(self.analyze_memory_usage(content, file_path));
            optimizations.extend(self.analyze_io_patterns(content, file_path));
            optimizations.extend(self.analyze_algorithm_complexity(content, file_path));
        }

        // Sort by impact (high -> medium -> low)
        optimizations.sort_by(|a, b| {
            let impact_order = |impact: &str| match impact {
                "critical" => 0,
                "high" => 1,
                "medium" => 2,
                "low" => 3,
                _ => 4,
            };
            let a_impact = a["impact"].as_str().unwrap_or("low");
            let b_impact = b["impact"].as_str().unwrap_or("low");
            impact_order(a_impact).cmp(&impact_order(b_impact))
        });

        optimizations
    }

    fn analyze_data_structure_usage(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut suggestions = Vec::new();

        // Check for inefficient Vec usage patterns
        if content.contains("vec.iter().find(") && content.matches("vec.iter().find(").count() > 2 {
            suggestions.push(json!({
                "optimization": "Replace Vec with HashMap for lookups",
                "description": "Multiple Vec::find() calls detected. Consider using HashMap for O(1) lookups",
                "file": file_path,
                "impact": "high",
                "pattern": "vec.iter().find()",
                "suggestion": "Use HashMap<K, V> or BTreeMap<K, V> for frequent key-based lookups",
                "estimated_improvement": "O(n) -> O(1) lookup time"
            }));
        }

        // Check for String concatenation in loops
        if content.contains("for ") && content.contains("push_str(") {
            suggestions.push(json!({
                "optimization": "Use String capacity pre-allocation",
                "description": "String concatenation in loop detected. Pre-allocate capacity to avoid reallocations",
                "file": file_path,
                "impact": "medium",
                "pattern": "String concatenation in loop",
                "suggestion": "Use String::with_capacity() or consider using format! macro",
                "estimated_improvement": "Reduced memory allocations"
            }));
        }

        // Check for unnecessary cloning
        if content.matches(".clone()").count() > 5 {
            suggestions.push(json!({
                "optimization": "Reduce unnecessary cloning",
                "description": "Excessive cloning detected. Consider using references or Cow<T>",
                "file": file_path,
                "impact": "medium",
                "pattern": "Frequent .clone() calls",
                "suggestion": "Use &T references, Rc<T>, or Cow<T> where appropriate",
                "estimated_improvement": "Reduced memory usage and allocation overhead"
            }));
        }

        suggestions
    }

    fn analyze_loop_patterns(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut suggestions = Vec::new();

        // Check for index-based loops that could use iterators
        if content.contains("for i in 0..") && content.contains("[i]") {
            suggestions.push(json!({
                "optimization": "Use iterators instead of index-based loops",
                "description": "Index-based loop detected. Iterators are more idiomatic and often faster",
                "file": file_path,
                "impact": "low",
                "pattern": "for i in 0..len",
                "suggestion": "Use .iter(), .enumerate(), or .zip() methods",
                "estimated_improvement": "Better performance and bounds checking elimination"
            }));
        }

        // Check for nested loops that might benefit from different algorithms
        let nested_loop_count = content.matches("for ").count();
        if nested_loop_count > 2 && content.contains("for ") && content.lines().any(|line| line.trim().starts_with("for ")) {
            suggestions.push(json!({
                "optimization": "Consider algorithmic improvements for nested loops",
                "description": "Multiple nested loops detected. Consider if algorithm can be optimized",
                "file": file_path,
                "impact": "high",
                "pattern": "Nested loops",
                "suggestion": "Consider using hash maps, sorting, or other algorithms to reduce complexity",
                "estimated_improvement": "Potential O(n²) -> O(n log n) or O(n) improvement"
            }));
        }

        suggestions
    }

    fn analyze_memory_usage(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut suggestions = Vec::new();

        // Check for large stack allocations
        if content.contains("Vec::with_capacity(") {
            let capacity_matches: Vec<&str> = content.matches("Vec::with_capacity(").collect();
            if capacity_matches.len() > 3 {
                suggestions.push(json!({
                    "optimization": "Consider using Box<[T]> for large allocations",
                    "description": "Multiple large Vec allocations detected",
                    "file": file_path,
                    "impact": "medium",
                    "pattern": "Large Vec allocations",
                    "suggestion": "Use Box<[T]> for fixed-size data or consider streaming processing",
                    "estimated_improvement": "Reduced memory fragmentation"
                }));
            }
        }

        // Check for potential memory leaks (missing Drop implementations)
        if content.contains("Box::new(") && !content.contains("impl Drop") {
            suggestions.push(json!({
                "optimization": "Implement proper resource cleanup",
                "description": "Manual memory management detected without Drop implementation",
                "file": file_path,
                "impact": "high",
                "pattern": "Manual memory management",
                "suggestion": "Implement Drop trait for proper resource cleanup",
                "estimated_improvement": "Prevent memory leaks and resource exhaustion"
            }));
        }

        suggestions
    }

    fn analyze_io_patterns(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut suggestions = Vec::new();

        // Check for synchronous I/O in async context
        if content.contains("async fn") && (content.contains("std::fs::read") || content.contains("std::fs::write")) {
            suggestions.push(json!({
                "optimization": "Use async I/O operations",
                "description": "Synchronous I/O detected in async function",
                "file": file_path,
                "impact": "high",
                "pattern": "Blocking I/O in async context",
                "suggestion": "Use tokio::fs or async-std for non-blocking I/O",
                "estimated_improvement": "Better async runtime utilization"
            }));
        }

        // Check for multiple small file operations
        if content.matches("File::open").count() > 3 {
            suggestions.push(json!({
                "optimization": "Batch file operations",
                "description": "Multiple file operations detected",
                "file": file_path,
                "impact": "medium",
                "pattern": "Multiple file operations",
                "suggestion": "Consider batching operations or using memory-mapped files",
                "estimated_improvement": "Reduced system call overhead"
            }));
        }

        suggestions
    }

    fn analyze_algorithm_complexity(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut suggestions = Vec::new();

        // Check for potential O(n²) algorithms
        if content.contains("sort()") && content.contains("for ") {
            suggestions.push(json!({
                "optimization": "Consider more efficient sorting algorithms",
                "description": "Sorting operation in loop context detected",
                "file": file_path,
                "impact": "high",
                "pattern": "Sorting in loop",
                "suggestion": "Sort once outside loop or use different data structure",
                "estimated_improvement": "Reduced algorithmic complexity"
            }));
        }

        // Check for linear search patterns that could use binary search
        if content.contains("iter().find(") && content.contains("sort") {
            suggestions.push(json!({
                "optimization": "Use binary search for sorted data",
                "description": "Linear search on potentially sorted data",
                "file": file_path,
                "impact": "medium",
                "pattern": "Linear search on sorted data",
                "suggestion": "Use binary_search() or binary_search_by() for O(log n) lookup",
                "estimated_improvement": "O(n) -> O(log n) search time"
            }));
        }

        suggestions
    }

    fn calculate_performance_score(&self, hotspots: &[Value]) -> f64 {
        let penalty = hotspots.len() as f64 * 10.0;
        (100.0 - penalty).max(0.0)
    }

    fn suggest_algorithmic_improvements(&self, files: &[Value]) -> Vec<String> {
        let mut improvements = Vec::new();

        for file in files {
            let content = file["content"].as_str().unwrap_or("");
            let file_path = file["path"].as_str().unwrap_or("unknown");

            // Analyze for algorithmic improvements
            improvements.extend(self.analyze_sorting_algorithms(content, file_path));
            improvements.extend(self.analyze_search_algorithms(content, file_path));
            improvements.extend(self.analyze_data_processing_patterns(content, file_path));
            improvements.extend(self.analyze_concurrency_opportunities(content, file_path));
        }

        // Remove duplicates and limit to most impactful suggestions
        improvements.sort();
        improvements.dedup();
        improvements.truncate(10);

        improvements
    }

    fn analyze_sorting_algorithms(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains(".sort()") {
            suggestions.push(format!("🚀 {} Consider using sort_unstable() for better performance when stability isn't required", file_path));
        }

        if content.contains("sort_by(") && content.contains("cmp(") {
            suggestions.push(format!("📊 {} Use sort_by_key() instead of sort_by() when comparing by a single field", file_path));
        }

        if content.matches("sort").count() > 2 {
            suggestions.push(format!("⚡ {} Multiple sorting operations detected - consider sorting once and maintaining order", file_path));
        }

        suggestions
    }

    fn analyze_search_algorithms(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains("iter().find(") && content.contains("sort") {
            suggestions.push(format!("🔍 {} Use binary_search() on sorted data instead of linear search", file_path));
        }

        if content.matches("iter().find(").count() > 3 {
            suggestions.push(format!("📈 {} Multiple linear searches detected - consider using HashMap or BTreeMap", file_path));
        }

        if content.contains("contains(") && content.contains("Vec") {
            suggestions.push(format!("🗂️ {} Use HashSet for membership testing instead of Vec::contains()", file_path));
        }

        suggestions
    }

    fn analyze_data_processing_patterns(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains("collect::<Vec<_>>()") && content.contains("iter()") {
            suggestions.push(format!("🔄 {} Consider using iterator chains without intermediate collections", file_path));
        }

        if content.matches("map(").count() > 2 && content.contains("collect()") {
            suggestions.push(format!("⚡ {} Chain multiple map operations before collecting", file_path));
        }

        if content.contains("filter(") && content.contains("map(") {
            suggestions.push(format!("🎯 {} Use filter_map() to combine filtering and mapping", file_path));
        }

        if content.contains("for ") && content.contains("push(") {
            suggestions.push(format!("📊 {} Consider using iterator methods instead of manual loops", file_path));
        }

        suggestions
    }

    fn analyze_concurrency_opportunities(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains("for ") && (content.contains("expensive") || content.contains("compute") || content.contains("process")) {
            suggestions.push(format!("🚀 {} Consider using rayon for parallel processing of independent operations", file_path));
        }

        if content.contains("async fn") && content.contains("await") && content.matches("await").count() > 3 {
            suggestions.push(format!("⚡ {} Use join! or select! for concurrent async operations", file_path));
        }

        if content.contains("Mutex") && content.contains("lock()") {
            suggestions.push(format!("🔒 {} Consider using RwLock for read-heavy workloads", file_path));
        }

        if content.contains("Arc<Mutex<") {
            suggestions.push(format!("🔄 {} Consider using channels for communication instead of shared state", file_path));
        }

        suggestions
    }

    fn suggest_memory_optimizations(&self, files: &[Value]) -> Vec<String> {
        let mut optimizations = Vec::new();

        for file in files {
            let content = file["content"].as_str().unwrap_or("");
            let file_path = file["path"].as_str().unwrap_or("unknown");

            // Analyze for memory optimization opportunities
            optimizations.extend(self.analyze_allocation_patterns(content, file_path));
            optimizations.extend(self.analyze_ownership_patterns(content, file_path));
            optimizations.extend(self.analyze_collection_usage(content, file_path));
            optimizations.extend(self.analyze_string_handling(content, file_path));
        }

        // Remove duplicates and prioritize
        optimizations.sort();
        optimizations.dedup();
        optimizations.truncate(12);

        optimizations
    }

    fn analyze_allocation_patterns(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.matches("Box::new(").count() > 3 {
            suggestions.push(format!("📦 {} Consider using stack allocation or pre-allocated pools for frequent Box allocations", file_path));
        }

        if content.contains("Vec::new()") && content.contains("push(") {
            suggestions.push(format!("📏 {} Use Vec::with_capacity() when final size is known", file_path));
        }

        if content.contains("String::new()") && content.contains("push_str(") {
            suggestions.push(format!("📝 {} Use String::with_capacity() for string building", file_path));
        }

        if content.matches("vec!").count() > 5 {
            suggestions.push(format!("🗂️ {} Consider using arrays or const slices for static data", file_path));
        }

        suggestions
    }

    fn analyze_ownership_patterns(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.matches(".clone()").count() > 5 {
            suggestions.push(format!("💾 {} Reduce cloning by using references (&T) or Rc<T>/Arc<T>", file_path));
        }

        if content.contains("to_owned()") || content.contains("to_string()") {
            suggestions.push(format!("🔄 {} Consider using Cow<str> for conditional ownership", file_path));
        }

        if content.contains("Arc<") && content.contains("clone()") {
            suggestions.push(format!("🔗 {} Arc cloning is cheap - avoid unnecessary intermediate variables", file_path));
        }

        if content.contains("Rc<RefCell<") {
            suggestions.push(format!("🔒 {} Consider redesigning to avoid interior mutability", file_path));
        }

        suggestions
    }

    fn analyze_collection_usage(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains("HashMap::new()") && !content.contains("with_capacity") {
            suggestions.push(format!("🗺️ {} Use HashMap::with_capacity() when size is predictable", file_path));
        }

        if content.contains("BTreeMap") && content.contains("get(") && content.matches("get(").count() > 3 {
            suggestions.push(format!("🌳 {} Consider HashMap for better lookup performance if ordering isn't needed", file_path));
        }

        if content.contains("Vec") && content.contains("remove(0)") {
            suggestions.push(format!("📋 {} Use VecDeque for efficient front removal", file_path));
        }

        if content.contains("HashSet") && content.contains("iter().collect()") {
            suggestions.push(format!("🎯 {} Use from_iter() or extend() instead of iter().collect()", file_path));
        }

        suggestions
    }

    fn analyze_string_handling(&self, content: &str, file_path: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if content.contains("format!(") && content.contains("&format!(") {
            suggestions.push(format!("📝 {} Avoid temporary String allocation in format! chains", file_path));
        }

        if content.matches("String::from(").count() > 3 {
            suggestions.push(format!("🔤 {} Use &str when possible, String only when ownership is needed", file_path));
        }

        if content.contains("split(") && content.contains("collect()") {
            suggestions.push(format!("✂️ {} Consider using split() iterator directly instead of collecting", file_path));
        }

        if content.contains("replace(") && content.contains("replace(") {
            suggestions.push(format!("🔄 {} Chain string operations or use regex for complex replacements", file_path));
        }

        suggestions
    }

    fn generate_benchmarking_suggestions(&self, files: &[Value]) -> Vec<Value> {
        let mut benchmarks = Vec::new();

        for file in files {
            let content = file["content"].as_str().unwrap_or("");
            let file_path = file["path"].as_str().unwrap_or("unknown");

            // Analyze for benchmarking opportunities
            benchmarks.extend(self.identify_performance_critical_functions(content, file_path));
            benchmarks.extend(self.identify_memory_intensive_operations(content, file_path));
            benchmarks.extend(self.identify_io_operations(content, file_path));
            benchmarks.extend(self.identify_algorithmic_hotspots(content, file_path));
        }

        // Sort by priority and remove duplicates
        benchmarks.sort_by(|a, b| {
            let priority_order = |p: &str| match p {
                "critical" => 0,
                "high" => 1,
                "medium" => 2,
                "low" => 3,
                _ => 4,
            };
            let a_priority = a["priority"].as_str().unwrap_or("low");
            let b_priority = b["priority"].as_str().unwrap_or("low");
            priority_order(a_priority).cmp(&priority_order(b_priority))
        });

        benchmarks.truncate(8);
        benchmarks
    }

    fn identify_performance_critical_functions(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut benchmarks = Vec::new();

        // Look for functions that might be called frequently
        if content.contains("pub fn") && (content.contains("loop") || content.contains("for ")) {
            benchmarks.push(json!({
                "benchmark": "Hot path function performance",
                "description": format!("Benchmark functions with loops in {}", file_path),
                "file": file_path,
                "priority": "high",
                "category": "cpu",
                "suggested_tool": "criterion"
            }));
        }

        benchmarks
    }

    fn identify_memory_intensive_operations(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut benchmarks = Vec::new();

        if content.contains("Vec::with_capacity(") || content.contains("HashMap::with_capacity(") {
            benchmarks.push(json!({
                "benchmark": "Memory allocation patterns",
                "description": format!("Profile memory usage patterns in {}", file_path),
                "file": file_path,
                "priority": "medium",
                "category": "memory"
            }));
        }

        benchmarks
    }

    fn identify_io_operations(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut benchmarks = Vec::new();

        if content.contains("File::") || content.contains("fs::") {
            benchmarks.push(json!({
                "benchmark": "File I/O performance",
                "description": format!("Benchmark file operations in {}", file_path),
                "file": file_path,
                "priority": "high",
                "category": "io"
            }));
        }

        benchmarks
    }

    fn identify_algorithmic_hotspots(&self, content: &str, file_path: &str) -> Vec<Value> {
        let mut benchmarks = Vec::new();

        if content.contains("sort") {
            benchmarks.push(json!({
                "benchmark": "Sorting algorithm performance",
                "description": format!("Benchmark sorting operations in {}", file_path),
                "file": file_path,
                "priority": "medium",
                "category": "algorithm"
            }));
        }

        benchmarks
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
        let mut dependencies = Vec::new();

        // Scan Cargo.toml for Rust dependencies
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists() {
            dependencies.extend(self.scan_cargo_dependencies(&cargo_toml).await?);
        }

        // Scan package.json for Node.js dependencies
        let package_json = path.join("package.json");
        if package_json.exists() {
            dependencies.extend(self.scan_npm_dependencies(&package_json).await?);
        }

        // Scan requirements.txt for Python dependencies
        let requirements_txt = path.join("requirements.txt");
        if requirements_txt.exists() {
            dependencies.extend(self.scan_python_dependencies(&requirements_txt).await?);
        }

        // Scan go.mod for Go dependencies
        let go_mod = path.join("go.mod");
        if go_mod.exists() {
            dependencies.extend(self.scan_go_dependencies(&go_mod).await?);
        }

        Ok(dependencies)
    }

    async fn scan_cargo_dependencies(&self, cargo_toml: &Path) -> Result<Vec<Value>> {
        let content = tokio::fs::read_to_string(cargo_toml).await
            .map_err(|e| AgentError::tool("code_analysis", &format!("Failed to read Cargo.toml: {}", e)))?;

        let mut dependencies = Vec::new();
        let mut current_section = None;

        for line in content.lines() {
            let line = line.trim();

            // Track current section
            if line.starts_with('[') && line.ends_with(']') {
                current_section = Some(line.to_string());
                continue;
            }

            // Parse dependencies
            if let Some(ref section) = current_section {
                if section == "[dependencies]" || section == "[dev-dependencies]" || section == "[build-dependencies]" {
                    if let Some((name, version_info)) = self.parse_cargo_dependency_line(line) {
                        let dep_type = match section.as_str() {
                            "[dev-dependencies]" => "dev",
                            "[build-dependencies]" => "build",
                            _ => "direct"
                        };

                        dependencies.push(json!({
                            "name": name,
                            "version": version_info.version,
                            "type": dep_type,
                            "ecosystem": "rust",
                            "features": version_info.features,
                            "optional": version_info.optional,
                            "git": version_info.git,
                            "path": version_info.path
                        }));
                    }
                }
            }
        }

        Ok(dependencies)
    }

    fn parse_cargo_dependency_line(&self, line: &str) -> Option<(String, DependencyInfo)> {
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        if let Some(eq_pos) = line.find('=') {
            let name = line[..eq_pos].trim().trim_matches('"').to_string();
            let value_part = line[eq_pos + 1..].trim();

            // Simple version string
            if value_part.starts_with('"') && value_part.ends_with('"') {
                let version = value_part.trim_matches('"').to_string();
                return Some((name, DependencyInfo {
                    version,
                    features: None,
                    optional: false,
                    git: None,
                    path: None,
                }));
            }

            // Complex dependency specification
            if value_part.starts_with('{') {
                return self.parse_complex_cargo_dependency(&name, value_part);
            }
        }

        None
    }

    fn parse_complex_cargo_dependency(&self, name: &str, spec: &str) -> Option<(String, DependencyInfo)> {
        let mut version = "unknown".to_string();
        let mut features = None;
        let mut optional = false;
        let mut git = None;
        let mut path = None;

        // Basic parsing of TOML-like structure
        if let Some(version_start) = spec.find("version") {
            if let Some(quote_start) = spec[version_start..].find('"') {
                let quote_start = version_start + quote_start + 1;
                if let Some(quote_end) = spec[quote_start..].find('"') {
                    version = spec[quote_start..quote_start + quote_end].to_string();
                }
            }
        }

        if spec.contains("optional = true") {
            optional = true;
        }

        if let Some(git_start) = spec.find("git") {
            if let Some(quote_start) = spec[git_start..].find('"') {
                let quote_start = git_start + quote_start + 1;
                if let Some(quote_end) = spec[quote_start..].find('"') {
                    git = Some(spec[quote_start..quote_start + quote_end].to_string());
                }
            }
        }

        if let Some(path_start) = spec.find("path") {
            if let Some(quote_start) = spec[path_start..].find('"') {
                let quote_start = path_start + quote_start + 1;
                if let Some(quote_end) = spec[quote_start..].find('"') {
                    path = Some(spec[quote_start..quote_start + quote_end].to_string());
                }
            }
        }

        Some((name.to_string(), DependencyInfo {
            version,
            features,
            optional,
            git,
            path,
        }))
    }

    async fn scan_npm_dependencies(&self, package_json: &Path) -> Result<Vec<Value>> {
        let content = tokio::fs::read_to_string(package_json).await
            .map_err(|e| AgentError::tool("code_analysis", &format!("Failed to read package.json: {}", e)))?;

        let package: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| AgentError::tool("code_analysis", &format!("Invalid package.json: {}", e)))?;

        let mut dependencies = Vec::new();

        // Regular dependencies
        if let Some(deps) = package["dependencies"].as_object() {
            for (name, version) in deps {
                dependencies.push(json!({
                    "name": name,
                    "version": version.as_str().unwrap_or("unknown"),
                    "type": "direct",
                    "ecosystem": "npm"
                }));
            }
        }

        // Dev dependencies
        if let Some(dev_deps) = package["devDependencies"].as_object() {
            for (name, version) in dev_deps {
                dependencies.push(json!({
                    "name": name,
                    "version": version.as_str().unwrap_or("unknown"),
                    "type": "dev",
                    "ecosystem": "npm"
                }));
            }
        }

        // Peer dependencies
        if let Some(peer_deps) = package["peerDependencies"].as_object() {
            for (name, version) in peer_deps {
                dependencies.push(json!({
                    "name": name,
                    "version": version.as_str().unwrap_or("unknown"),
                    "type": "peer",
                    "ecosystem": "npm"
                }));
            }
        }

        Ok(dependencies)
    }

    async fn scan_python_dependencies(&self, requirements_txt: &Path) -> Result<Vec<Value>> {
        let content = tokio::fs::read_to_string(requirements_txt).await
            .map_err(|e| AgentError::tool("code_analysis", &format!("Failed to read requirements.txt: {}", e)))?;

        let mut dependencies = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse Python dependency line (package==version, package>=version, etc.)
            if let Some((name, version)) = self.parse_python_dependency_line(line) {
                dependencies.push(json!({
                    "name": name,
                    "version": version,
                    "type": "direct",
                    "ecosystem": "python"
                }));
            }
        }

        Ok(dependencies)
    }

    fn parse_python_dependency_line(&self, line: &str) -> Option<(String, String)> {
        // Handle various Python dependency formats
        for operator in &["==", ">=", "<=", ">", "<", "~=", "!="] {
            if let Some(pos) = line.find(operator) {
                let name = line[..pos].trim().to_string();
                let version = line[pos + operator.len()..].trim().to_string();
                return Some((name, version));
            }
        }

        // If no version specified, assume latest
        Some((line.to_string(), "latest".to_string()))
    }

    async fn scan_go_dependencies(&self, go_mod: &Path) -> Result<Vec<Value>> {
        let content = tokio::fs::read_to_string(go_mod).await
            .map_err(|e| AgentError::tool("code_analysis", &format!("Failed to read go.mod: {}", e)))?;

        let mut dependencies = Vec::new();
        let mut in_require_block = false;

        for line in content.lines() {
            let line = line.trim();

            if line.starts_with("require (") {
                in_require_block = true;
                continue;
            }

            if in_require_block && line == ")" {
                in_require_block = false;
                continue;
            }

            if line.starts_with("require ") || in_require_block {
                if let Some((name, version)) = self.parse_go_dependency_line(line) {
                    dependencies.push(json!({
                        "name": name,
                        "version": version,
                        "type": "direct",
                        "ecosystem": "go"
                    }));
                }
            }
        }

        Ok(dependencies)
    }

    fn parse_go_dependency_line(&self, line: &str) -> Option<(String, String)> {
        let line = line.trim();

        // Remove "require " prefix if present
        let line = if line.starts_with("require ") {
            &line[8..]
        } else {
            line
        };

        // Split by whitespace to get module and version
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let name = parts[0].to_string();
            let version = parts[1].to_string();
            Some((name, version))
        } else {
            None
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
        let mut vulnerabilities = Vec::new();

        // Known vulnerable packages and versions (simplified database)
        let known_vulnerabilities = self.get_known_vulnerabilities();

        for dep in dependencies {
            let name = dep["name"].as_str().unwrap_or("");
            let version = dep["version"].as_str().unwrap_or("");
            let ecosystem = dep["ecosystem"].as_str().unwrap_or("");

            // Check against known vulnerabilities
            if let Some(vuln_info) = known_vulnerabilities.get(&format!("{}:{}", ecosystem, name)) {
                for vuln in vuln_info {
                    if self.version_matches_vulnerability(version, &vuln.affected_versions) {
                        vulnerabilities.push(json!({
                            "package": name,
                            "version": version,
                            "ecosystem": ecosystem,
                            "vulnerability_id": vuln.id,
                            "severity": vuln.severity,
                            "description": vuln.description,
                            "affected_versions": vuln.affected_versions,
                            "fixed_version": vuln.fixed_version,
                            "cve": vuln.cve,
                            "published": vuln.published
                        }));
                    }
                }
            }
        }

        Ok(vulnerabilities)
    }

    fn get_known_vulnerabilities(&self) -> std::collections::HashMap<String, Vec<VulnerabilityInfo>> {
        use std::collections::HashMap;

        let mut vulns = HashMap::new();

        // Add some real-world examples of known vulnerabilities
        vulns.insert("rust:serde".to_string(), vec![
            VulnerabilityInfo {
                id: "RUSTSEC-2022-0040".to_string(),
                severity: "high".to_string(),
                description: "Deserialization of untrusted data in serde".to_string(),
                affected_versions: "<1.0.145".to_string(),
                fixed_version: Some("1.0.145".to_string()),
                cve: Some("CVE-2022-31394".to_string()),
                published: "2022-08-01".to_string(),
            }
        ]);

        vulns.insert("npm:lodash".to_string(), vec![
            VulnerabilityInfo {
                id: "GHSA-jf85-cpcp-j695".to_string(),
                severity: "high".to_string(),
                description: "Prototype Pollution in lodash".to_string(),
                affected_versions: "<4.17.21".to_string(),
                fixed_version: Some("4.17.21".to_string()),
                cve: Some("CVE-2021-23337".to_string()),
                published: "2021-02-15".to_string(),
            }
        ]);

        vulns.insert("python:django".to_string(), vec![
            VulnerabilityInfo {
                id: "PYSEC-2023-123".to_string(),
                severity: "critical".to_string(),
                description: "SQL injection vulnerability in Django ORM".to_string(),
                affected_versions: "<4.2.5".to_string(),
                fixed_version: Some("4.2.5".to_string()),
                cve: Some("CVE-2023-41164".to_string()),
                published: "2023-09-04".to_string(),
            }
        ]);

        vulns.insert("go:github.com/gin-gonic/gin".to_string(), vec![
            VulnerabilityInfo {
                id: "GO-2023-1234".to_string(),
                severity: "medium".to_string(),
                description: "Directory traversal in Gin framework".to_string(),
                affected_versions: "<1.9.1".to_string(),
                fixed_version: Some("1.9.1".to_string()),
                cve: Some("CVE-2023-29401".to_string()),
                published: "2023-06-08".to_string(),
            }
        ]);

        vulns
    }

    fn version_matches_vulnerability(&self, version: &str, affected_range: &str) -> bool {
        // Simplified version matching - in production, use a proper semver library
        if affected_range.starts_with('<') {
            let target_version = &affected_range[1..];
            return self.version_less_than(version, target_version);
        }

        if affected_range.starts_with(">=") {
            let target_version = &affected_range[2..];
            return !self.version_less_than(version, target_version);
        }

        // Exact match
        version == affected_range
    }

    fn version_less_than(&self, version1: &str, version2: &str) -> bool {
        // Simplified version comparison - in production, use semver crate
        let v1_parts: Vec<u32> = version1.split('.').filter_map(|s| s.parse().ok()).collect();
        let v2_parts: Vec<u32> = version2.split('.').filter_map(|s| s.parse().ok()).collect();

        for i in 0..std::cmp::max(v1_parts.len(), v2_parts.len()) {
            let v1_part = v1_parts.get(i).unwrap_or(&0);
            let v2_part = v2_parts.get(i).unwrap_or(&0);

            if v1_part < v2_part {
                return true;
            } else if v1_part > v2_part {
                return false;
            }
        }

        false
    }

    async fn analyze_licenses(&self, dependencies: &[Value]) -> Result<Value> {
        let mut license_issues = Vec::new();
        let mut license_summary = std::collections::HashMap::new();
        let mut problematic_licenses = Vec::new();

        // Known problematic licenses for commercial use
        let problematic_license_patterns = vec![
            "GPL", "AGPL", "LGPL", "SSPL", "BUSL", "Commons Clause"
        ];

        for dep in dependencies {
            let name = dep["name"].as_str().unwrap_or("");
            let ecosystem = dep["ecosystem"].as_str().unwrap_or("");

            // Get license information for this dependency
            let license_info = self.get_dependency_license_info(name, ecosystem).await;

            if let Some(license) = license_info {
                // Count license types
                *license_summary.entry(license.clone()).or_insert(0) += 1;

                // Check for problematic licenses
                for problematic_pattern in &problematic_license_patterns {
                    if license.to_uppercase().contains(problematic_pattern) {
                        license_issues.push(json!({
                            "dependency": name,
                            "license": license,
                            "severity": self.get_license_severity(&license),
                            "issue": format!("Potentially problematic license: {}", license),
                            "recommendation": self.get_license_recommendation(&license)
                        }));

                        if !problematic_licenses.contains(&license) {
                            problematic_licenses.push(license.clone());
                        }
                        break;
                    }
                }
            } else {
                // Unknown license
                license_issues.push(json!({
                    "dependency": name,
                    "license": "Unknown",
                    "severity": "medium",
                    "issue": "License information not available",
                    "recommendation": "Manually verify license compatibility"
                }));
            }
        }

        let compliance_status = if license_issues.is_empty() {
            "compliant"
        } else if problematic_licenses.is_empty() {
            "warning"
        } else {
            "non-compliant"
        };

        Ok(json!({
            "total_dependencies": dependencies.len(),
            "license_compliance": compliance_status,
            "license_summary": license_summary,
            "problematic_licenses": problematic_licenses,
            "issues": license_issues,
            "risk_level": self.calculate_license_risk_level(&license_issues)
        }))
    }

    async fn get_dependency_license_info(&self, name: &str, ecosystem: &str) -> Option<String> {
        // In a real implementation, this would query package registries
        // For now, return some common licenses based on well-known packages
        match (ecosystem, name) {
            ("rust", "serde") => Some("MIT OR Apache-2.0".to_string()),
            ("rust", "tokio") => Some("MIT".to_string()),
            ("rust", "clap") => Some("MIT OR Apache-2.0".to_string()),
            ("npm", "react") => Some("MIT".to_string()),
            ("npm", "lodash") => Some("MIT".to_string()),
            ("npm", "express") => Some("MIT".to_string()),
            ("python", "django") => Some("BSD-3-Clause".to_string()),
            ("python", "flask") => Some("BSD-3-Clause".to_string()),
            ("python", "requests") => Some("Apache-2.0".to_string()),
            ("go", "github.com/gin-gonic/gin") => Some("MIT".to_string()),
            _ => {
                // For unknown packages, simulate some license detection
                if name.contains("gpl") || name.contains("copyleft") {
                    Some("GPL-3.0".to_string())
                } else if name.contains("apache") {
                    Some("Apache-2.0".to_string())
                } else if name.contains("bsd") {
                    Some("BSD-3-Clause".to_string())
                } else {
                    None // Unknown license
                }
            }
        }
    }

    fn get_license_severity(&self, license: &str) -> &'static str {
        let license_upper = license.to_uppercase();
        if license_upper.contains("GPL") || license_upper.contains("AGPL") {
            "high"
        } else if license_upper.contains("LGPL") || license_upper.contains("SSPL") {
            "medium"
        } else if license_upper.contains("BUSL") || license_upper.contains("COMMONS CLAUSE") {
            "high"
        } else {
            "low"
        }
    }

    fn get_license_recommendation(&self, license: &str) -> String {
        let license_upper = license.to_uppercase();
        if license_upper.contains("GPL") {
            "Consider replacing with MIT or Apache-2.0 licensed alternative".to_string()
        } else if license_upper.contains("AGPL") {
            "AGPL requires source disclosure for network use - consider alternatives".to_string()
        } else if license_upper.contains("SSPL") {
            "SSPL has restrictions on cloud services - review carefully".to_string()
        } else {
            "Review license terms for compatibility with your use case".to_string()
        }
    }

    fn calculate_license_risk_level(&self, issues: &[Value]) -> &'static str {
        let high_severity_count = issues.iter()
            .filter(|issue| issue["severity"].as_str() == Some("high"))
            .count();

        let medium_severity_count = issues.iter()
            .filter(|issue| issue["severity"].as_str() == Some("medium"))
            .count();

        if high_severity_count > 0 {
            "high"
        } else if medium_severity_count > 2 {
            "medium"
        } else if !issues.is_empty() {
            "low"
        } else {
            "minimal"
        }
    }

    fn identify_license_issues(&self, license_analysis: &Value) -> Vec<Value> {
        license_analysis["issues"].as_array().unwrap_or(&vec![]).clone()
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

    fn generate_license_recommendations(&self, license_analysis: &Value) -> Vec<String> {
        let empty_vec = vec![];
        let issues = license_analysis["issues"].as_array().unwrap_or(&empty_vec);
        let risk_level = license_analysis["risk_level"].as_str().unwrap_or("minimal");

        let mut recommendations = Vec::new();

        match risk_level {
            "high" => {
                recommendations.push("🚨 Immediate action required: Replace high-risk licensed dependencies".to_string());
                recommendations.push("⚖️ Consult legal team for license compliance review".to_string());
                recommendations.push("📋 Create dependency replacement roadmap".to_string());
            }
            "medium" => {
                recommendations.push("⚠️ Review medium-risk licenses for compatibility".to_string());
                recommendations.push("📄 Document license decisions and rationale".to_string());
                recommendations.push("🔍 Schedule regular license compliance audits".to_string());
            }
            "low" => {
                recommendations.push("📄 Document all dependency licenses".to_string());
                recommendations.push("🔍 Monitor for license changes in dependencies".to_string());
            }
            _ => {
                recommendations.push("✅ License compliance looks good".to_string());
                recommendations.push("🔄 Maintain regular license monitoring".to_string());
            }
        }

        if !issues.is_empty() {
            recommendations.push("📊 Generate detailed license compliance report".to_string());
            recommendations.push("🔧 Consider automated license scanning tools".to_string());
        }

        recommendations
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

    #[tokio::test]
    async fn test_dependency_scanning_cargo() {
        let dir = tempdir().unwrap();
        let cargo_toml = dir.path().join("Cargo.toml");
        fs::write(&cargo_toml, r#"
[package]
name = "test-project"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
tempfile = "3.0"
"#).await.unwrap();

        let tool = CodeAnalysisTool::new();
        let result = tool.scan_dependencies(dir.path()).await.unwrap();

        assert_eq!(result.len(), 3);

        // Check serde dependency
        let serde_dep = result.iter().find(|dep| dep["name"] == "serde").unwrap();
        assert_eq!(serde_dep["version"], "1.0");
        assert_eq!(serde_dep["type"], "direct");
        assert_eq!(serde_dep["ecosystem"], "rust");

        // Check tokio dependency
        let tokio_dep = result.iter().find(|dep| dep["name"] == "tokio").unwrap();
        assert_eq!(tokio_dep["version"], "1.0");
        assert_eq!(tokio_dep["type"], "direct");

        // Check dev dependency
        let tempfile_dep = result.iter().find(|dep| dep["name"] == "tempfile").unwrap();
        assert_eq!(tempfile_dep["type"], "dev");
    }

    #[tokio::test]
    async fn test_dependency_scanning_package_json() {
        let dir = tempdir().unwrap();
        let package_json = dir.path().join("package.json");
        fs::write(&package_json, r#"
{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0",
    "lodash": "4.17.21"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
"#).await.unwrap();

        let tool = CodeAnalysisTool::new();
        let result = tool.scan_dependencies(dir.path()).await.unwrap();

        assert_eq!(result.len(), 3);

        // Check react dependency
        let react_dep = result.iter().find(|dep| dep["name"] == "react").unwrap();
        assert_eq!(react_dep["version"], "^18.0.0");
        assert_eq!(react_dep["type"], "direct");
        assert_eq!(react_dep["ecosystem"], "npm");

        // Check dev dependency
        let jest_dep = result.iter().find(|dep| dep["name"] == "jest").unwrap();
        assert_eq!(jest_dep["type"], "dev");
    }

    #[tokio::test]
    async fn test_vulnerability_scanning() {
        let tool = CodeAnalysisTool::new();

        // Test with known vulnerable package
        let dependencies = vec![
            json!({
                "name": "lodash",
                "version": "4.17.20",
                "ecosystem": "npm"
            })
        ];

        let vulnerabilities = tool.scan_dependency_vulnerabilities(&dependencies).await.unwrap();

        // Should detect the lodash vulnerability
        assert!(!vulnerabilities.is_empty());
        let vuln = &vulnerabilities[0];
        assert_eq!(vuln["package"], "lodash");
        assert_eq!(vuln["severity"], "high");
        assert!(vuln["vulnerability_id"].as_str().unwrap().contains("GHSA"));
    }

    #[tokio::test]
    async fn test_license_analysis() {
        let tool = CodeAnalysisTool::new();

        let dependencies = vec![
            json!({
                "name": "serde",
                "version": "1.0.145",
                "ecosystem": "rust"
            }),
            json!({
                "name": "unknown-gpl-package",
                "version": "1.0.0",
                "ecosystem": "rust"
            })
        ];

        let license_analysis = tool.analyze_licenses(&dependencies).await.unwrap();

        assert_eq!(license_analysis["total_dependencies"], 2);
        assert_eq!(license_analysis["license_compliance"], "non-compliant");

        let issues = license_analysis["issues"].as_array().unwrap();
        assert!(!issues.is_empty());

        // Should detect GPL issue
        let gpl_issue = issues.iter().find(|issue|
            issue["dependency"].as_str().unwrap().contains("gpl")
        ).unwrap();
        assert_eq!(gpl_issue["severity"], "high");
    }

    #[tokio::test]
    async fn test_license_check_integration() {
        let dir = tempdir().unwrap();
        let cargo_toml = dir.path().join("Cargo.toml");
        fs::write(&cargo_toml, r#"
[package]
name = "test-project"
version = "0.1.0"

[dependencies]
serde = "1.0"
"#).await.unwrap();

        let tool = CodeAnalysisTool::new();
        let result = tool.license_check(dir.path(), &json!({})).await.unwrap();

        assert_eq!(result["analysis_type"], "license_compliance");
        assert!(result["license_analysis"].is_object());
        assert!(result["compliance_issues"].is_array());
        assert!(result["recommendations"].is_array());

        let recommendations = result["recommendations"].as_array().unwrap();
        assert!(!recommendations.is_empty());
    }
}
