use std::env;
use std::fs;
use std::io::{self, Write, BufRead, BufReader};
use std::path::Path;
use std::process::Command;
use tempfile::NamedTempFile;
use crate::utils::error::{AgentError, Result};

/// Enhanced prompt system with multiple input modes
pub struct PromptSystem {
    history: Vec<String>,
    history_index: usize,
    editor_command: String,
}

/// Input modes for the prompt system
#[derive(Debug, Clone)]
pub enum InputMode {
    /// Single line input (default)
    SingleLine,
    /// Multi-line editor using external editor
    Editor,
    /// Interactive multi-line console
    Interactive,
    /// Load from file
    File(String),
    /// Read from stdin pipe
    Pipe,
}

impl PromptSystem {
    pub fn new() -> Self {
        let editor_command = env::var("EDITOR")
            .or_else(|_| env::var("VISUAL"))
            .unwrap_or_else(|_| {
                // Try to find a suitable editor
                for editor in &["nano", "vim", "vi", "code", "subl", "atom"] {
                    if Command::new("which").arg(editor).output().map(|o| o.status.success()).unwrap_or(false) {
                        return editor.to_string();
                    }
                }
                "nano".to_string() // Default fallback
            });

        Self {
            history: Vec::new(),
            history_index: 0,
            editor_command,
        }
    }

    /// Get user input based on the specified mode
    pub fn get_input(&mut self, mode: InputMode, prompt: &str) -> Result<String> {
        match mode {
            InputMode::SingleLine => self.get_single_line_input(prompt),
            InputMode::Editor => self.get_editor_input(prompt),
            InputMode::Interactive => self.get_interactive_input(prompt),
            InputMode::File(path) => self.get_file_input(&path),
            InputMode::Pipe => self.get_pipe_input(),
        }
    }

    /// Single line input with basic editing
    fn get_single_line_input(&mut self, prompt: &str) -> Result<String> {
        print!("{}", prompt);
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        let input = input.trim().to_string();
        if !input.is_empty() {
            self.add_to_history(input.clone());
        }
        
        Ok(input)
    }

    /// Multi-line input using external editor
    fn get_editor_input(&mut self, prompt: &str) -> Result<String> {
        println!("{}", prompt);
        println!("Opening editor ({})... Save and close to continue.", self.editor_command);
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new()?;
        
        // Write initial prompt as comment
        writeln!(temp_file, "# Enter your prompt below (lines starting with # are ignored)")?;
        writeln!(temp_file, "# Save and close the editor to continue")?;
        writeln!(temp_file, "")?;
        
        temp_file.flush()?;
        
        // Open editor
        let status = Command::new(&self.editor_command)
            .arg(temp_file.path())
            .status()?;
        
        if !status.success() {
            return Err(AgentError::invalid_input("Editor exited with error"));
        }
        
        // Read content
        let content = fs::read_to_string(temp_file.path())?;
        
        // Filter out comment lines and empty lines
        let input: String = content
            .lines()
            .filter(|line| !line.trim().starts_with('#') && !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        
        if !input.is_empty() {
            self.add_to_history(input.clone());
        }
        
        Ok(input)
    }

    /// Interactive multi-line console with editing capabilities
    fn get_interactive_input(&mut self, prompt: &str) -> Result<String> {
        println!("{}", prompt);
        println!("Interactive mode - Enter your prompt (Ctrl+D to finish, Ctrl+C to cancel):");
        println!("Commands: :help, :history, :clear, :editor, :save <file>, :load <file>");
        
        let mut lines = Vec::new();
        let mut line_number = 1;
        
        loop {
            print!("{:3}> ", line_number);
            io::stdout().flush()?;
            
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => break, // EOF (Ctrl+D)
                Ok(_) => {
                    let line = line.trim_end_matches('\n');
                    
                    // Handle commands
                    if line.starts_with(':') {
                        match self.handle_interactive_command(line, &mut lines)? {
                            CommandResult::Continue => {
                                continue;
                            }
                            CommandResult::Finish(content) => {
                                return Ok(content);
                            }
                            CommandResult::Cancel => {
                                return Ok(String::new());
                            }
                        }
                    } else {
                        lines.push(line.to_string());
                        line_number += 1;
                    }
                }
                Err(e) => return Err(AgentError::Io(e)),
            }
        }
        
        let input = lines.join("\n").trim().to_string();
        if !input.is_empty() {
            self.add_to_history(input.clone());
        }
        
        Ok(input)
    }

    /// Load input from file
    fn get_file_input(&self, path: &str) -> Result<String> {
        if !Path::new(path).exists() {
            return Err(AgentError::invalid_input(format!("File not found: {}", path)));
        }
        
        let content = fs::read_to_string(path)?;
        
        Ok(content.trim().to_string())
    }

    /// Read input from stdin pipe
    fn get_pipe_input(&self) -> Result<String> {
        let stdin = io::stdin();
        let reader = BufReader::new(stdin.lock());
        
        let mut content = String::new();
        for line in reader.lines() {
            let line = line?;
            content.push_str(&line);
            content.push('\n');
        }
        
        Ok(content.trim().to_string())
    }

    /// Handle interactive commands
    fn handle_interactive_command(&mut self, command: &str, lines: &mut Vec<String>) -> Result<CommandResult> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        
        match parts.get(0).copied() {
            Some(":help") => {
                println!("Available commands:");
                println!("  :help          - Show this help");
                println!("  :history       - Show input history");
                println!("  :clear         - Clear current input");
                println!("  :editor        - Open external editor");
                println!("  :save <file>   - Save current input to file");
                println!("  :load <file>   - Load input from file");
                println!("  :finish        - Finish input (same as Ctrl+D)");
                println!("  :cancel        - Cancel input");
                Ok(CommandResult::Continue)
            }
            Some(":history") => {
                println!("Input history:");
                for (i, item) in self.history.iter().enumerate() {
                    println!("  {}: {}", i + 1, 
                        if item.len() > 50 { 
                            format!("{}...", &item[..50]) 
                        } else { 
                            item.clone() 
                        }
                    );
                }
                Ok(CommandResult::Continue)
            }
            Some(":clear") => {
                lines.clear();
                println!("Input cleared.");
                Ok(CommandResult::Continue)
            }
            Some(":editor") => {
                let current_content = lines.join("\n");
                let mut temp_file = NamedTempFile::new()?;
                
                write!(temp_file, "{}", current_content)?;
                temp_file.flush()?;
                
                let status = Command::new(&self.editor_command)
                    .arg(temp_file.path())
                    .status()?;
                
                if status.success() {
                    let content = fs::read_to_string(temp_file.path())?;
                    *lines = content.lines().map(|s| s.to_string()).collect();
                    println!("Content updated from editor.");
                }
                Ok(CommandResult::Continue)
            }
            Some(":save") => {
                if let Some(filename) = parts.get(1) {
                    let content = lines.join("\n");
                    fs::write(filename, content)?;
                    println!("Saved to {}", filename);
                } else {
                    println!("Usage: :save <filename>");
                }
                Ok(CommandResult::Continue)
            }
            Some(":load") => {
                if let Some(filename) = parts.get(1) {
                    match fs::read_to_string(filename) {
                        Ok(content) => {
                            *lines = content.lines().map(|s| s.to_string()).collect();
                            println!("Loaded from {}", filename);
                        }
                        Err(e) => println!("Error loading file: {}", e),
                    }
                } else {
                    println!("Usage: :load <filename>");
                }
                Ok(CommandResult::Continue)
            }
            Some(":finish") => {
                let content = lines.join("\n").trim().to_string();
                Ok(CommandResult::Finish(content))
            }
            Some(":cancel") => {
                Ok(CommandResult::Cancel)
            }
            _ => {
                println!("Unknown command: {}. Type :help for available commands.", command);
                Ok(CommandResult::Continue)
            }
        }
    }

    /// Add input to history
    fn add_to_history(&mut self, input: String) {
        self.history.push(input);
        self.history_index = self.history.len();
        
        // Keep history size reasonable
        if self.history.len() > 100 {
            self.history.remove(0);
            self.history_index = self.history.len();
        }
    }

    /// Get history
    pub fn get_history(&self) -> &[String] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history_index = 0;
    }
}

/// Result of interactive command processing
enum CommandResult {
    Continue,
    Finish(String),
    Cancel,
}

impl Default for PromptSystem {
    fn default() -> Self {
        Self::new()
    }
}
