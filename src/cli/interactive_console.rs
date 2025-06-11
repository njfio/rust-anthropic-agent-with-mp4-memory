use std::io::{self, Write, BufRead};
use std::fs;
use std::path::Path;
use colored::*;
use crate::utils::error::{AgentError, Result};

/// Enhanced interactive console with multi-line editing and commands
pub struct InteractiveConsole {
    history: Vec<String>,
    current_input: Vec<String>,
    prompt_prefix: String,
}

impl InteractiveConsole {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current_input: Vec::new(),
            prompt_prefix: "memvid".to_string(),
        }
    }

    /// Start the interactive console session
    pub fn start(&mut self) -> Result<()> {
        self.print_welcome();
        
        loop {
            match self.get_command()? {
                ConsoleCommand::Input(text) => {
                    if !text.trim().is_empty() {
                        return Ok(());
                    }
                }
                ConsoleCommand::Exit => {
                    println!("{}", "Goodbye!".green());
                    std::process::exit(0);
                }
                ConsoleCommand::Continue => continue,
            }
        }
    }

    /// Get user input with enhanced editing capabilities
    pub fn get_multiline_input(&mut self, initial_prompt: &str) -> Result<String> {
        println!("{}", initial_prompt.cyan());
        println!("{}", "Enter your prompt (type 'END' on a new line to finish, ':help' for commands):".yellow());
        
        self.current_input.clear();
        let mut line_number = 1;
        
        loop {
            print!("{} {:3}> ", self.prompt_prefix.blue(), line_number.to_string().bright_black());
            io::stdout().flush()?;
            
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            let line = line.trim_end_matches('\n');
            
            // Check for end marker
            if line.trim() == "END" {
                break;
            }
            
            // Handle commands
            if line.starts_with(':') {
                match self.handle_command(line)? {
                    CommandResult::Continue => continue,
                    CommandResult::Finish => break,
                    CommandResult::Exit => {
                        println!("{}", "Cancelled.".yellow());
                        return Ok(String::new());
                    }
                }
            } else {
                self.current_input.push(line.to_string());
                line_number += 1;
            }
        }
        
        let input = self.current_input.join("\n").trim().to_string();
        if !input.is_empty() {
            self.add_to_history(input.clone());
        }
        
        Ok(input)
    }

    /// Handle console commands
    fn handle_command(&mut self, command: &str) -> Result<CommandResult> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        
        match parts.get(0).copied() {
            Some(":help") | Some(":h") => {
                self.print_help();
                Ok(CommandResult::Continue)
            }
            Some(":history") | Some(":hist") => {
                self.show_history();
                Ok(CommandResult::Continue)
            }
            Some(":clear") | Some(":c") => {
                self.current_input.clear();
                println!("{}", "Current input cleared.".green());
                Ok(CommandResult::Continue)
            }
            Some(":show") | Some(":s") => {
                self.show_current_input();
                Ok(CommandResult::Continue)
            }
            Some(":save") => {
                if let Some(filename) = parts.get(1) {
                    self.save_to_file(filename)?;
                } else {
                    println!("{}", "Usage: :save <filename>".red());
                }
                Ok(CommandResult::Continue)
            }
            Some(":load") | Some(":l") => {
                if let Some(filename) = parts.get(1) {
                    self.load_from_file(filename)?;
                } else {
                    println!("{}", "Usage: :load <filename>".red());
                }
                Ok(CommandResult::Continue)
            }
            Some(":undo") | Some(":u") => {
                if !self.current_input.is_empty() {
                    let removed = self.current_input.pop().unwrap();
                    println!("{} {}", "Removed line:".yellow(), removed.bright_black());
                } else {
                    println!("{}", "Nothing to undo.".yellow());
                }
                Ok(CommandResult::Continue)
            }
            Some(":finish") | Some(":f") => {
                Ok(CommandResult::Finish)
            }
            Some(":exit") | Some(":quit") | Some(":q") => {
                Ok(CommandResult::Exit)
            }
            Some(":paste") | Some(":p") => {
                println!("{}", "Paste mode - enter multiple lines, then type 'END' to finish:".cyan());
                self.paste_mode()?;
                Ok(CommandResult::Continue)
            }
            _ => {
                println!("{} {} {}", "Unknown command:".red(), command, "(type :help for available commands)".yellow());
                Ok(CommandResult::Continue)
            }
        }
    }

    /// Paste mode for bulk input
    fn paste_mode(&mut self) -> Result<()> {
        loop {
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            let line = line.trim_end_matches('\n');
            
            if line.trim() == "END" {
                break;
            }
            
            self.current_input.push(line.to_string());
        }
        
        println!("{} {} {}", "Added".green(), self.current_input.len(), "lines from paste.".green());
        Ok(())
    }

    /// Get a simple command
    fn get_command(&mut self) -> Result<ConsoleCommand> {
        print!("{} > ", self.prompt_prefix.blue());
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        match input {
            "exit" | "quit" | ":q" => Ok(ConsoleCommand::Exit),
            "" => Ok(ConsoleCommand::Continue),
            _ => Ok(ConsoleCommand::Input(input.to_string())),
        }
    }

    /// Print welcome message
    pub fn print_welcome(&self) {
        println!("{}", "=".repeat(60).bright_blue());
        println!("{}", "🤖 Rust MemVid Agent - Interactive Console".bright_cyan().bold());
        println!("{}", "=".repeat(60).bright_blue());
        println!();
        println!("{}", "Available modes:".yellow());
        println!("  {} - Single line input", "chat <message>".green());
        println!("  {} - Multi-line interactive mode", "interactive".green());
        println!("  {} - Load prompt from file", "file <path>".green());
        println!("  {} - Read from stdin pipe", "pipe".green());
        println!();
        println!("{} {} for commands or {} to quit.", "Type".yellow(), ":help".green().bold(), "exit".red().bold());
        println!();
    }

    /// Print help information
    fn print_help(&self) {
        println!();
        println!("{}", "📚 Available Commands:".cyan().bold());
        println!("{}", "-".repeat(40).bright_black());
        
        let commands = vec![
            (":help, :h", "Show this help message"),
            (":history, :hist", "Show input history"),
            (":clear, :c", "Clear current input"),
            (":show, :s", "Show current input"),
            (":save <file>", "Save current input to file"),
            (":load <file>, :l", "Load input from file"),
            (":undo, :u", "Remove last line"),
            (":paste, :p", "Enter paste mode for bulk input"),
            (":finish, :f", "Finish current input"),
            (":exit, :quit, :q", "Exit the console"),
            ("END", "Finish multi-line input (on new line)"),
        ];
        
        for (cmd, desc) in commands {
            println!("  {:<15} - {}", cmd.green(), desc);
        }
        println!();
    }

    /// Show input history
    fn show_history(&self) {
        if self.history.is_empty() {
            println!("{}", "No history available.".yellow());
            return;
        }
        
        println!();
        println!("{}", "📜 Input History:".cyan().bold());
        println!("{}", "-".repeat(40).bright_black());
        
        for (i, item) in self.history.iter().enumerate().rev().take(10) {
            let preview = if item.len() > 60 {
                format!("{}...", &item[..60])
            } else {
                item.clone()
            };
            
            println!("  {}: {}", 
                format!("{:2}", i + 1).bright_black(), 
                preview.replace('\n', " ").bright_white()
            );
        }
        
        if self.history.len() > 10 {
            println!("  {} (showing last 10 of {})", "...".bright_black(), self.history.len());
        }
        println!();
    }

    /// Show current input
    fn show_current_input(&self) {
        if self.current_input.is_empty() {
            println!("{}", "No current input.".yellow());
            return;
        }
        
        println!();
        println!("{}", "📝 Current Input:".cyan().bold());
        println!("{}", "-".repeat(40).bright_black());
        
        for (i, line) in self.current_input.iter().enumerate() {
            println!("  {:3}: {}", 
                format!("{}", i + 1).bright_black(), 
                line.bright_white()
            );
        }
        println!();
    }

    /// Save current input to file
    fn save_to_file(&self, filename: &str) -> Result<()> {
        let content = self.current_input.join("\n");
        fs::write(filename, content)?;
        println!("{} {}", "Saved to".green(), filename.bright_white());
        Ok(())
    }

    /// Load input from file
    fn load_from_file(&mut self, filename: &str) -> Result<()> {
        if !Path::new(filename).exists() {
            println!("{} {}", "File not found:".red(), filename);
            return Ok(());
        }
        
        let content = fs::read_to_string(filename)?;
        self.current_input = content.lines().map(|s| s.to_string()).collect();
        
        println!("{} {} {} {}", 
            "Loaded".green(), 
            self.current_input.len(), 
            "lines from".green(), 
            filename.bright_white()
        );
        Ok(())
    }

    /// Add to history
    fn add_to_history(&mut self, input: String) {
        self.history.push(input);
        
        // Keep history reasonable size
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Read from stdin pipe
    pub fn read_from_pipe() -> Result<String> {
        let stdin = io::stdin();
        let mut content = String::new();
        
        for line in stdin.lock().lines() {
            let line = line?;
            content.push_str(&line);
            content.push('\n');
        }
        
        Ok(content.trim().to_string())
    }

    /// Load from file (static method)
    pub fn load_file(path: &str) -> Result<String> {
        if !Path::new(path).exists() {
            return Err(AgentError::invalid_input(format!("File not found: {}", path)));
        }
        
        let content = fs::read_to_string(path)?;
        Ok(content.trim().to_string())
    }
}

/// Console command results
enum ConsoleCommand {
    Input(String),
    Exit,
    Continue,
}

/// Command processing results
enum CommandResult {
    Continue,
    Finish,
    Exit,
}

impl Default for InteractiveConsole {
    fn default() -> Self {
        Self::new()
    }
}
