//! Mini coding agent CLI — a baby Claude Code in ~250 lines.
//!
//! Features:
//!   - Interactive REPL with multi-turn conversation
//!   - All built-in tools (bash, read/write/edit files, search, list)
//!   - Streaming text output with colored tool feedback
//!   - Token usage after each turn
//!
//! Run:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example cli
//!   ANTHROPIC_API_KEY=sk-... cargo run --example cli -- --model claude-sonnet-5
//!   ANTHROPIC_API_KEY=sk-... cargo run --example cli -- --skills ./skills
//!
//! Run with a named provider (zai, qwen, openai, xai, groq, deepseek, mistral, minimax, ollama, google):
//!   API_KEY=... cargo run --example cli -- --provider zai --model glm-4.7
//!   DASHSCOPE_API_KEY=... cargo run --example cli -- --provider qwen --model qwen3.6-plus
//!   cargo run --example cli -- --provider ollama --model llama3.1:8b
//!
//! Run with LM Studio / local OpenAI-compatible server:
//!   cargo run --example cli -- --api-url http://localhost:1234/v1 --model local-model
//!
//! Commands:
//!   /quit, /exit    Exit the agent
//!   /clear          Clear conversation history
//!   /model <name>   Switch model mid-session

use std::io::{self, BufRead, Write};
use yoagent::agent::Agent;
use yoagent::provider::ModelConfig;
use yoagent::skills::SkillSet;
use yoagent::tools::default_tools;
use yoagent::*;

// ANSI color helpers
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const RED: &str = "\x1b[31m";

const SYSTEM_PROMPT: &str = r#"You are a coding assistant working in the user's terminal.
You have access to the filesystem and shell. Be direct and concise.
When the user asks you to do something, do it — don't just explain how.
Use tools proactively: read files to understand context, run commands to verify your work.
After making changes, run tests or verify the result when appropriate."#;

fn print_banner() {
    println!("\n{BOLD}{CYAN}  yoagent cli{RESET} {DIM}— mini coding agent{RESET}");
    println!("{DIM}  Type /quit to exit, /clear to reset{RESET}\n");
}

/// Print the session rollup that `AgentEnd` carries.
///
/// Hit rate counts `cache_write` against you — those are prompt tokens the
/// provider processed and billed. Read it against the session length rather
/// than against 100%: every turn's new content is necessarily a miss.
fn print_stats(stats: &SessionStats) {
    let u = &stats.usage;
    if u.input == 0 && u.output == 0 && u.cache_read == 0 && u.cache_write == 0 {
        return;
    }
    println!(
        "\n{DIM}  tokens: {} in / {} out over {} turn(s){RESET}",
        u.input, u.output, stats.turns
    );
    if u.cache_read > 0 || u.cache_write > 0 {
        println!(
            "{DIM}  cache: {:.1}% hit ({} read, {} write){RESET}",
            stats.cache_hit_rate() * 100.0,
            u.cache_read,
            u.cache_write
        );
    }
    if stats.compactions > 0 {
        println!("{DIM}  compactions: {}{RESET}", stats.compactions);
    }
    if let Some(cost) = stats.cost_usd {
        println!("{DIM}  cost: ${cost:.4}{RESET}");
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let api_url = args
        .iter()
        .position(|a| a == "--api-url")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let provider_name = args
        .iter()
        .position(|a| a == "--provider")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let api_key_optional = api_url.is_some() || provider_name.as_deref() == Some("ollama");
    let api_key = if provider_name.as_deref() == Some("qwen") {
        std::env::var("DASHSCOPE_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .expect("Set DASHSCOPE_API_KEY or API_KEY")
    } else if provider_name.as_deref() == Some("meta") {
        std::env::var("META_API_KEY")
            .or_else(|_| std::env::var("MODEL_API_KEY"))
            .expect("Set META_API_KEY or MODEL_API_KEY")
    } else if api_key_optional {
        std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .unwrap_or_default() // empty string OK for local/Ollama
    } else {
        std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .expect("Set ANTHROPIC_API_KEY or API_KEY")
    };

    let default_model = match provider_name.as_deref() {
        Some("zai") => "glm-4.7",
        Some("qwen") => "qwen3.6-plus",
        Some("openai") => "gpt-5.5",
        Some("xai") => "grok-4-1-fast",
        Some("groq") => "llama-3.3-70b-versatile",
        Some("deepseek") => "deepseek-v4-flash",
        Some("mistral") => "mistral-large-latest",
        Some("minimax") => "MiniMax-Text-01",
        Some("meta") => "muse-spark-1.1",
        Some("ollama") => "llama3.1:8b",
        Some("google") => "gemini-2.5-pro",
        _ => "claude-sonnet-5",
    };

    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default_model.into());

    // Collect --skills directories (can be specified multiple times)
    let skill_dirs: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--skills")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();

    let skills = if skill_dirs.is_empty() {
        SkillSet::empty()
    } else {
        SkillSet::load(&skill_dirs).expect("Failed to load skills")
    };

    let mut agent = build_agent(&api_url, &provider_name, &model)
        .with_system_prompt(SYSTEM_PROMPT)
        // from_config already resolves the provider-conventional env key; this
        // override preserves the CLI's API_KEY fallback (empty = leave to env).
        .with_api_key(&api_key)
        .with_skills(skills.clone())
        .with_tools(default_tools());

    // Graceful Ctrl+C exit
    tokio::spawn(async {
        tokio::signal::ctrl_c().await.ok();
        println!("\n{DIM}  bye 👋{RESET}\n");
        std::process::exit(0);
    });

    print_banner();
    println!("{DIM}  model: {model}{RESET}");
    if !skills.is_empty() {
        println!("{DIM}  skills: {} loaded{RESET}", skills.len());
    }
    println!(
        "{DIM}  cwd:   {}{RESET}\n",
        std::env::current_dir().unwrap().display()
    );

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        // Prompt
        print!("{BOLD}{GREEN}> {RESET}");
        io::stdout().flush().ok();

        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Commands
        match input {
            "/quit" | "/exit" => break,
            "/clear" => {
                agent.clear_messages();
                println!("{DIM}  (conversation cleared){RESET}\n");
                continue;
            }
            s if s.starts_with("/model ") => {
                let new_model = s.trim_start_matches("/model ").trim();
                agent = build_agent(&api_url, &provider_name, new_model)
                    .with_system_prompt(SYSTEM_PROMPT)
                    .with_api_key(&api_key)
                    .with_skills(skills.clone())
                    .with_tools(default_tools());
                println!("{DIM}  (switched to {new_model}, conversation cleared){RESET}\n");
                continue;
            }
            _ => {}
        }

        // Send to agent
        let mut rx = agent.prompt(input).await;
        let mut session_stats = SessionStats::default();
        let mut in_text = false;

        while let Some(event) = rx.recv().await {
            match event {
                AgentEvent::ToolExecutionStart {
                    tool_name, args, ..
                } => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                    let summary = match tool_name.as_str() {
                        "bash" => {
                            let cmd = args
                                .get("command")
                                .and_then(|v| v.as_str())
                                .unwrap_or("...");
                            format!("$ {}", truncate(cmd, 80))
                        }
                        "read_file" => {
                            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                            format!("read {}", path)
                        }
                        "write_file" => {
                            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                            format!("write {}", path)
                        }
                        "edit_file" => {
                            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                            format!("edit {}", path)
                        }
                        "list_files" => {
                            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                            format!("ls {}", path)
                        }
                        "search" => {
                            let pat = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
                            format!("search '{}'", truncate(pat, 60))
                        }
                        _ => tool_name.clone(),
                    };
                    print!("{YELLOW}  ▶ {summary}{RESET}");
                    io::stdout().flush().ok();
                }
                AgentEvent::ToolExecutionEnd { is_error, .. } => {
                    if is_error {
                        println!(" {RED}✗{RESET}");
                    } else {
                        println!(" {GREEN}✓{RESET}");
                    }
                }
                AgentEvent::MessageUpdate {
                    delta: StreamDelta::Text { delta },
                    ..
                } => {
                    if !in_text {
                        println!();
                        in_text = true;
                    }
                    print!("{}", delta);
                    io::stdout().flush().ok();
                }
                AgentEvent::MessageEnd {
                    message:
                        AgentMessage::Llm(Message::Assistant {
                            stop_reason: StopReason::Error,
                            error_message,
                            ..
                        }),
                } => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                    let msg = error_message.as_deref().unwrap_or("unknown error");
                    println!("{RED}  error: {msg}{RESET}");
                }
                AgentEvent::MessageEnd {
                    message:
                        AgentMessage::Llm(Message::Assistant {
                            stop_reason: StopReason::Refusal,
                            error_message,
                            ..
                        }),
                } => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                    let msg = error_message
                        .as_deref()
                        .unwrap_or("request declined by the model's safety system");
                    println!("{RED}  refused: {msg}{RESET}");
                }
                AgentEvent::AgentEnd { stats, .. } => {
                    session_stats = stats.clone();
                }
                _ => {}
            }
        }

        if in_text {
            println!();
        }
        print_stats(&session_stats);
        println!();
    }

    println!("\n{DIM}  bye 👋{RESET}\n");
}

/// Select the config for the requested provider/URL and build an agent from
/// it. A local/OpenAI-compatible URL wins; then a named provider; else
/// Anthropic. Every branch flows through `from_config`, so the provider,
/// model id, and context window all come from a single `ModelConfig`.
fn build_agent(api_url: &Option<String>, provider_name: &Option<String>, model: &str) -> Agent {
    if let Some(url) = api_url {
        let config = if provider_name.as_deref() == Some("ollama") {
            ModelConfig::ollama(url, model)
        } else {
            ModelConfig::local(url, model)
        };
        Agent::from_config(config)
    } else if let Some(prov) = provider_name {
        make_provider_agent(prov, model)
    } else {
        Agent::from_config(ModelConfig::anthropic(model, model))
    }
}

fn make_provider_agent(provider: &str, model: &str) -> Agent {
    // One `from_config` per provider: the config carries the protocol (so the
    // right provider is selected), the model id, context window, and pricing.
    // No provider↔config pairing to get wrong, and no model id passed twice.
    let config = match provider {
        "zai" => ModelConfig::zai(model, model),
        "qwen" => ModelConfig::qwen(model, model),
        "openai" => ModelConfig::openai(model, model),
        "xai" => ModelConfig::xai(model, model),
        "groq" => ModelConfig::groq(model, model),
        "deepseek" => ModelConfig::deepseek(model, model),
        "mistral" => ModelConfig::mistral(model, model),
        "minimax" => ModelConfig::minimax(model, model),
        "meta" => ModelConfig::meta(model, model),
        "ollama" => ModelConfig::ollama("http://localhost:11434/v1", model),
        "google" => ModelConfig::google(model, model),
        other => {
            eprintln!("Unknown provider: {other}. Supported: zai, qwen, openai, xai, groq, deepseek, mistral, minimax, meta, ollama, google.");
            std::process::exit(1);
        }
    };
    Agent::from_config(config)
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        match s.char_indices().nth(max) {
            Some((idx, _)) => &s[..idx],
            None => s,
        }
    }
}
