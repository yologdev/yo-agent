//! Recursive Language Model (RLM) example.
//!
//! Demonstrates true RLM: an LLM that autonomously explores a codebase,
//! recursively delegating file-level analysis to sub-agents. All agents
//! communicate through shared state.
//!
//!   Parent (Rust) → lead_analyst (LLM, discovers + delegates)
//!                      → file_analyst (LLM, reads + analyzes)
//!
//! The lead_analyst uses file system tools to explore, then spawns
//! file_analyst sub-agents for deep analysis. No hardcoded file lists.
//!
//! Run (analyzes current directory):
//!   XAI_API_KEY=xai-... cargo run --example rlm
//!
//! Run on a specific directory:
//!   XAI_API_KEY=xai-... cargo run --example rlm -- path/to/dir

use std::sync::{Arc, Mutex};
use yoagent::provider::model::ModelConfig;
use yoagent::provider::{OpenAiCompatProvider, StreamProvider};
#[allow(unused_imports)]
use yoagent::shared_state::{FileBackend, SharedState};
use yoagent::sub_agent::SubAgentTool;
use yoagent::tools;
use yoagent::*;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("XAI_API_KEY").expect("Set XAI_API_KEY");
    let mut model_config = ModelConfig::xai("grok-4-1-fast-reasoning", "Grok 4.1 Fast Reasoning");
    model_config.reasoning = true;
    let provider: Arc<dyn StreamProvider> = Arc::new(OpenAiCompatProvider);

    let target_dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());

    println!("RLM Codebase Analyzer (Grok)");
    println!("Target: {}\n", target_dir);

    let state = SharedState::new();

    // Or use filesystem backend — each key becomes a file for persistence
    // and easy inspection of agent outputs:
    // let state = SharedState::with_backend(FileBackend::new(".rlm-state"));

    // Store the target directory so agents know where to look
    state
        .set("target_dir", target_dir.clone())
        .await
        .expect("store target dir");

    println!("--- RLM: 2-level recursive agent delegation ---");
    println!("Parent → lead_analyst (explores) → file_analyst (analyzes)\n");

    // --- Level 2: file_analyst (leaf agent) ---
    // Has read_file + shared_state. Reads a file, writes summary to shared state.
    let file_analyst =
        SubAgentTool::from_provider("file_analyst", Arc::clone(&provider), model_config.clone())
            .with_description(
                "Analyzes a single source file in depth. \
             Call with a task specifying the file path to analyze.",
            )
            .with_system_prompt(
                "You are a file-level code analyst. When given a file to analyze:\n\
             1. Use read_file to read the file content\n\
             2. Analyze it: purpose, key types/functions, design patterns, quality\n\
             3. Write a concise summary (under 200 words) to shared state with \
                key 'summary:<filepath>'\n\n\
             Be specific and technical. Focus on what makes this code interesting.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_tools(vec![Arc::new(tools::ReadFileTool::new())])
            .with_max_turns(5);

    // --- Level 1: lead_analyst (orchestrator agent) ---
    // Has list_files + read_file to explore, file_analyst to delegate, shared_state for results.
    let lead_analyst =
        SubAgentTool::from_provider("lead_analyst", Arc::clone(&provider), model_config)
            .with_description("Orchestrates codebase analysis")
            .with_system_prompt(
                "You are a lead code analyst orchestrating a codebase review.\n\n\
             IMPORTANT: Only analyze files within the target directory. Do NOT explore \
             parent directories or other parts of the project.\n\n\
             Steps:\n\
             1. Read 'target_dir' from shared state\n\
             2. Use list_files to discover source files ONLY within that directory\n\
             3. Pick the 2 most important files\n\
             4. For EACH chosen file, delegate to the 'file_analyst' tool: \
                'Analyze <filepath>'\n\
             5. After all files are analyzed, read each summary from shared state \
                (keys are 'summary:<filepath>')\n\
             6. Write a final synthesis report to shared state under key 'final_report'\n\n\
             The final report should identify cross-cutting themes, architectural patterns, \
             and how the files relate to each other. Keep it under 300 words.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_tools(vec![
                Arc::new(tools::ListFilesTool::new()),
                Arc::new(tools::ReadFileTool::new()),
                Arc::new(file_analyst),
            ])
            .with_max_turns(25);

    // --- Parent: single call triggers the full recursive chain ---
    let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let ctx = ToolContext::new("tc-rlm", "lead_analyst").with_on_update(Arc::new({
        let buf = buf.clone();
        move |result: ToolResult| {
            for content in &result.content {
                if let Content::Text { text } = content {
                    let mut b = buf.lock().unwrap();
                    if text.starts_with("[sub-agent calling tool:") {
                        if !b.is_empty() {
                            eprintln!("[lead] {}", b.drain(..).collect::<String>());
                        }
                        eprintln!("[lead] {}", text);
                        return;
                    }
                    b.push_str(text);
                    while let Some(pos) = b.find('\n') {
                        let line: String = b.drain(..=pos).collect();
                        eprint!("[lead] {}", line);
                    }
                }
            }
        }
    }));

    let result = lead_analyst
        .execute(
            serde_json::json!({"task": "Explore and analyze this Rust crate."}),
            ctx,
        )
        .await;

    // Flush remaining buffer
    {
        let b = buf.lock().unwrap();
        if !b.is_empty() {
            eprintln!("[lead] {}", *b);
        }
    }

    match result {
        Ok(result) => {
            eprintln!("[lead] details: {}", result.details);
            for content in &result.content {
                if let Content::Text { text } = content {
                    if !text.is_empty() {
                        eprintln!("[lead] (final) {}", text);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("\nError: {}", e);
            std::process::exit(1);
        }
    }

    // --- Read results from shared state ---
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  RLM Results");
    println!("═══════════════════════════════════════════════════════════\n");

    // Print all per-file summaries
    let keys = state.keys().await;
    for key in &keys {
        if let Some(file) = key.strip_prefix("summary:") {
            println!("── {} ──\n", file);
            if let Some(summary) = state.get(key).await {
                println!("{}\n", summary);
            }
        }
    }

    // Final report (written by lead_analyst, level 1)
    println!("── Final Synthesis ──\n");
    match state.get("final_report").await {
        Some(report) => println!("{}", report),
        None => println!("(lead_analyst did not produce a final report)"),
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Shared state: {}", state.summary().await);
    println!("═══════════════════════════════════════════════════════════");
}
