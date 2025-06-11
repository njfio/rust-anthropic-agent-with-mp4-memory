use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor, Stylize},
    terminal::{self, ClearType},
};

use crate::agent::Agent;
use crate::utils::error::Result;

/// Enhanced interactive chat mode with real-time human input
pub struct InteractiveChat {
    agent: Arc<Mutex<Agent>>,
    input_sender: mpsc::UnboundedSender<String>,
    input_receiver: mpsc::UnboundedReceiver<String>,
    debug_mode: bool,
}

/// Chat event types
#[derive(Debug, Clone)]
pub enum ChatEvent {
    UserInput(String),
    AgentThinking,
    ToolExecution { tool_name: String, input: String },
    ToolResult { tool_name: String, result: String, is_error: bool },
    AgentResponse(String),
    Error(String),
    SystemMessage(String),
}

impl InteractiveChat {
    /// Create a new interactive chat session
    pub fn new(agent: Agent, debug_mode: bool) -> Self {
        let (input_sender, input_receiver) = mpsc::unbounded_channel();
        
        Self {
            agent: Arc::new(Mutex::new(agent)),
            input_sender,
            input_receiver,
            debug_mode,
        }
    }

    /// Start the interactive chat session
    pub async fn start(&mut self) -> Result<()> {
        // Enable raw mode for better terminal control
        terminal::enable_raw_mode().map_err(|e| {
            crate::utils::error::AgentError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to enable raw mode: {}", e)
            ))
        })?;

        // Clear screen and show welcome message
        self.show_welcome().await?;

        // Start input handler in background
        let input_sender = self.input_sender.clone();
        tokio::spawn(async move {
            Self::handle_input(input_sender).await;
        });

        // Main chat loop
        let mut current_input = String::new();
        let mut agent_busy = false;

        loop {
            // Check for user input
            if let Ok(input) = self.input_receiver.try_recv() {
                if input.trim() == "/quit" || input.trim() == "/exit" {
                    break;
                }
                
                if input.trim() == "/help" {
                    self.show_help().await?;
                    continue;
                }

                if input.trim() == "/clear" {
                    self.clear_screen().await?;
                    continue;
                }

                if input.trim() == "/debug" {
                    self.debug_mode = !self.debug_mode;
                    self.print_system_message(&format!("Debug mode: {}", 
                        if self.debug_mode { "ON" } else { "OFF" })).await?;
                    continue;
                }

                // Process chat input
                current_input = input;
                agent_busy = true;
                
                self.print_event(ChatEvent::UserInput(current_input.clone())).await?;
                
                // Process with agent in background
                let agent = self.agent.clone();
                let input_clone = current_input.clone();
                let debug_mode = self.debug_mode;
                
                tokio::spawn(async move {
                    Self::process_agent_response(agent, input_clone, debug_mode).await;
                });
            }

            // Small delay to prevent busy waiting
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Cleanup
        terminal::disable_raw_mode().map_err(|e| {
            crate::utils::error::AgentError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to disable raw mode: {}", e)
            ))
        })?;

        self.print_system_message("Goodbye! 👋").await?;
        Ok(())
    }

    /// Handle keyboard input in a separate task
    async fn handle_input(sender: mpsc::UnboundedSender<String>) {
        let mut input_buffer = String::new();
        
        loop {
            if let Ok(true) = event::poll(Duration::from_millis(100)) {
                if let Ok(event) = event::read() {
                    match event {
                        Event::Key(KeyEvent { code, modifiers, .. }) => {
                            match code {
                                KeyCode::Enter => {
                                    if !input_buffer.trim().is_empty() {
                                        let _ = sender.send(input_buffer.clone());
                                        input_buffer.clear();
                                    }
                                }
                                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                                    let _ = sender.send("/quit".to_string());
                                    break;
                                }
                                KeyCode::Char(c) => {
                                    input_buffer.push(c);
                                    print!("{}", c);
                                    let _ = io::stdout().flush();
                                }
                                KeyCode::Backspace => {
                                    if !input_buffer.is_empty() {
                                        input_buffer.pop();
                                        print!("\x08 \x08"); // Backspace, space, backspace
                                        let _ = io::stdout().flush();
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Process agent response with enhanced UI
    async fn process_agent_response(agent: Arc<Mutex<Agent>>, input: String, debug_mode: bool) {
        // Show thinking indicator
        if !debug_mode {
            Self::print_thinking().await;
        }

        let mut agent_guard = agent.lock().await;
        
        match agent_guard.chat(input).await {
            Ok(response) => {
                if !debug_mode {
                    Self::clear_thinking().await;
                }
                Self::print_agent_response(&response).await;
            }
            Err(e) => {
                if !debug_mode {
                    Self::clear_thinking().await;
                }
                Self::print_error(&format!("Error: {}", e)).await;
            }
        }
    }

    /// Print a chat event with appropriate formatting
    async fn print_event(&self, event: ChatEvent) -> Result<()> {
        match event {
            ChatEvent::UserInput(input) => {
                execute!(
                    io::stdout(),
                    SetForegroundColor(Color::Cyan),
                    Print("👤 You: "),
                    ResetColor,
                    Print(&input),
                    Print("\n")
                ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
            }
            ChatEvent::AgentThinking => {
                if !self.debug_mode {
                    execute!(
                        io::stdout(),
                        SetForegroundColor(Color::Yellow),
                        Print("🤖 Agent is thinking"),
                        ResetColor
                    ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
                }
            }
            ChatEvent::ToolExecution { tool_name, input } => {
                if self.debug_mode {
                    execute!(
                        io::stdout(),
                        SetForegroundColor(Color::Magenta),
                        Print(&format!("🔧 Executing tool: {} with input: {}\n", tool_name, input)),
                        ResetColor
                    ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
                } else {
                    execute!(
                        io::stdout(),
                        SetForegroundColor(Color::Magenta),
                        Print(&format!("🔧 Using {}\n", tool_name)),
                        ResetColor
                    ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
                }
            }
            ChatEvent::ToolResult { tool_name, result, is_error } => {
                let color = if is_error { Color::Red } else { Color::Green };
                let icon = if is_error { "❌" } else { "✅" };
                
                if self.debug_mode {
                    execute!(
                        io::stdout(),
                        SetForegroundColor(color),
                        Print(&format!("{} {} result: {}\n", icon, tool_name, result)),
                        ResetColor
                    ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
                } else {
                    execute!(
                        io::stdout(),
                        SetForegroundColor(color),
                        Print(&format!("{} {} completed\n", icon, tool_name)),
                        ResetColor
                    ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
                }
            }
            ChatEvent::AgentResponse(response) => {
                execute!(
                    io::stdout(),
                    SetForegroundColor(Color::Green),
                    Print("🤖 Agent: "),
                    ResetColor,
                    Print(&response),
                    Print("\n\n")
                ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
            }
            ChatEvent::Error(error) => {
                execute!(
                    io::stdout(),
                    SetForegroundColor(Color::Red),
                    Print("❌ Error: "),
                    Print(&error),
                    Print("\n"),
                    ResetColor
                ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
            }
            ChatEvent::SystemMessage(message) => {
                execute!(
                    io::stdout(),
                    SetForegroundColor(Color::Blue),
                    Print("ℹ️  "),
                    Print(&message),
                    Print("\n"),
                    ResetColor
                ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
            }
        }
        
        io::stdout().flush().map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
        Ok(())
    }

    /// Show welcome message
    async fn show_welcome(&self) -> Result<()> {
        execute!(
            io::stdout(),
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0),
            SetForegroundColor(Color::Cyan),
            Print("🤖 MemVidAgent Interactive Chat\n"),
            Print("================================\n\n"),
            ResetColor,
            Print("Commands:\n"),
            Print("  /help  - Show this help\n"),
            Print("  /clear - Clear screen\n"),
            Print("  /debug - Toggle debug mode\n"),
            Print("  /quit  - Exit chat\n"),
            Print("  Ctrl+C - Exit chat\n\n"),
            SetForegroundColor(Color::Yellow),
            Print("Type your message and press Enter to chat!\n\n"),
            ResetColor
        ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;

        Ok(())
    }

    /// Show help message
    async fn show_help(&self) -> Result<()> {
        self.print_system_message("Available commands:").await?;
        self.print_system_message("  /help  - Show this help").await?;
        self.print_system_message("  /clear - Clear screen").await?;
        self.print_system_message("  /debug - Toggle debug mode").await?;
        self.print_system_message("  /quit  - Exit chat").await?;
        Ok(())
    }

    /// Clear screen
    async fn clear_screen(&self) -> Result<()> {
        execute!(
            io::stdout(),
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        ).map_err(|e| anyhow::anyhow!("Terminal error: {}", e))?;
        Ok(())
    }

    /// Print system message
    async fn print_system_message(&self, message: &str) -> Result<()> {
        self.print_event(ChatEvent::SystemMessage(message.to_string())).await
    }

    /// Show thinking animation
    async fn print_thinking() {
        let _ = execute!(
            io::stdout(),
            SetForegroundColor(Color::Yellow),
            Print("🤖 Agent is thinking..."),
            ResetColor
        );
        let _ = io::stdout().flush();
    }

    /// Clear thinking indicator
    async fn clear_thinking() {
        let _ = execute!(
            io::stdout(),
            Print("\r"),
            terminal::Clear(ClearType::CurrentLine)
        );
        let _ = io::stdout().flush();
    }

    /// Print agent response
    async fn print_agent_response(response: &str) {
        let _ = execute!(
            io::stdout(),
            SetForegroundColor(Color::Green),
            Print("🤖 Agent: "),
            ResetColor,
            Print(response),
            Print("\n\n")
        );
        let _ = io::stdout().flush();
    }

    /// Print error message
    async fn print_error(error: &str) {
        let _ = execute!(
            io::stdout(),
            SetForegroundColor(Color::Red),
            Print("❌ Error: "),
            Print(error),
            Print("\n"),
            ResetColor
        );
        let _ = io::stdout().flush();
    }
}
