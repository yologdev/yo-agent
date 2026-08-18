//! Code review agent: parallel sub-agents reviewing the same source file.
//!
//! Demonstrates:
//!   - Reading a real file and storing it in SharedState
//!   - 3 parallel sub-agents each analyzing a different aspect
//!   - Sub-agents writing structured findings back to shared state
//!   - Parent aggregating all reviews into a unified report
//!
//! Run on any source file:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example code_review -- path/to/file.rs
//!
//! Try it on this repo:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example code_review -- src/shared_state.rs

use std::sync::{Arc, Mutex};
use yoagent::provider::{AnthropicProvider, ModelConfig, StreamProvider};
use yoagent::shared_state::SharedState;
use yoagent::sub_agent::SubAgentTool;
use yoagent::*;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("Set ANTHROPIC_API_KEY");
    let config = ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
    let provider: Arc<dyn StreamProvider> = Arc::new(AnthropicProvider);

    // --- Read the target file from CLI args ---
    let file_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example code_review -- <file_path>");
        std::process::exit(1);
    });

    let source_code = std::fs::read_to_string(&file_path).unwrap_or_else(|e| {
        eprintln!("Failed to read '{}': {}", file_path, e);
        std::process::exit(1);
    });

    println!("Reviewing: {} ({} bytes)\n", file_path, source_code.len());

    // --- Store the source code once in shared state ---
    let state = SharedState::new();
    state
        .set("source_code", source_code)
        .await
        .expect("store source code");
    state
        .set("file_path", file_path.clone())
        .await
        .expect("store file path");

    // --- Three reviewers, each focused on a different aspect ---

    let bug_reviewer =
        SubAgentTool::from_provider("bug_reviewer", Arc::clone(&provider), config.clone())
            .with_description("Reviews code for bugs and logic errors")
            .with_system_prompt(
                "You are a bug-finding specialist. Read the source code from shared state \
             (key: 'source_code'), look for bugs, logic errors, off-by-one errors, \
             race conditions, and potential panics. Write your findings to shared state \
             under key 'bugs_review'. Format: bullet points, each with line reference \
             and severity (critical/warning/info). If no bugs found, say so.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    let quality_reviewer =
        SubAgentTool::from_provider("quality_reviewer", Arc::clone(&provider), config.clone())
            .with_description("Reviews code quality and style")
            .with_system_prompt(
                "You are a code quality reviewer. Read the source code from shared state \
             (key: 'source_code'), evaluate naming, structure, idiomatic usage, \
             error handling patterns, and API design. Write your findings to shared state \
             under key 'quality_review'. Format: bullet points with specific suggestions. \
             Mention what's done well too.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    let docs_reviewer =
        SubAgentTool::from_provider("docs_reviewer", Arc::clone(&provider), config.clone())
            .with_description("Reviews documentation completeness")
            .with_system_prompt(
                "You are a documentation reviewer. Read the source code from shared state \
             (key: 'source_code') and the file path (key: 'file_path'). Evaluate: \
             are public items documented? Are doc comments accurate? Are edge cases \
             explained? Are examples provided where helpful? Write findings to shared \
             state under key 'docs_review'. Format: bullet points.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    // --- Run all three in parallel with streaming ---
    println!("Dispatching 3 reviewers in parallel...\n");

    let make_ctx = |label: &str| -> (ToolContext, Arc<Mutex<String>>) {
        let label = label.to_string();
        let buf: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let ctx =
            ToolContext::new(format!("tc-{}", label), label.clone()).with_on_update(Arc::new({
                let label = label.clone();
                let buf = buf.clone();
                move |result: ToolResult| {
                    for content in &result.content {
                        if let Content::Text { text } = content {
                            let mut b = buf.lock().unwrap();
                            // Tool call events: flush buffer first, print on own line
                            if text.starts_with("[sub-agent calling tool:") {
                                if !b.is_empty() {
                                    eprintln!("[{}] {}", label, b.drain(..).collect::<String>());
                                }
                                eprintln!("[{}] {}", label, text);
                                continue;
                            }
                            b.push_str(text);
                            while let Some(pos) = b.find('\n') {
                                let line: String = b.drain(..=pos).collect();
                                eprint!("[{}] {}", label, line);
                            }
                        }
                    }
                }
            }));
        (ctx, buf)
    };

    let (ctx1, buf1) = make_ctx("bugs");
    let (ctx2, buf2) = make_ctx("quality");
    let (ctx3, buf3) = make_ctx("docs");

    let (r1, r2, r3) = tokio::join!(
        bug_reviewer.execute(
            serde_json::json!({"task": "Review the source code for bugs and logic errors."}),
            ctx1,
        ),
        quality_reviewer.execute(
            serde_json::json!({"task": "Review the source code for quality and style."}),
            ctx2,
        ),
        docs_reviewer.execute(
            serde_json::json!({"task": "Review the source code for documentation completeness."}),
            ctx3,
        ),
    );

    // Flush any remaining buffered text
    for (label, buf) in [("bugs", buf1), ("quality", buf2), ("docs", buf3)] {
        let b = buf.lock().unwrap();
        if !b.is_empty() {
            eprintln!("[{}] {}", label, *b);
        }
    }
    r1.expect("bug reviewer failed");
    r2.expect("quality reviewer failed");
    r3.expect("docs reviewer failed");

    // --- Print unified review ---
    println!("═══════════════════════════════════════════════════════════");
    println!("  Code Review: {}", file_path);
    println!("═══════════════════════════════════════════════════════════\n");

    let sections = [
        ("bugs_review", "Bug Analysis"),
        ("quality_review", "Code Quality"),
        ("docs_review", "Documentation"),
    ];

    for (key, title) in sections {
        println!("── {} ──\n", title);
        match state.get(key).await {
            Some(value) => println!("{}\n", value),
            None => println!("(reviewer did not produce findings)\n"),
        }
    }

    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Review complete. Shared state keys: {}",
        state.summary().await
    );
    println!("═══════════════════════════════════════════════════════════");
}
