//! Pre-release smoke test against a **live** provider.
//!
//! Run with: `ANTHROPIC_API_KEY=sk-... cargo run --example release_smoke`
//!
//! # Why this exists
//!
//! Every one of the 615 unit tests uses `MockProvider`, which accepts any
//! message sequence you hand it. The v0.18.0 release gate found a bug that no
//! mock can catch: loop detection injected a message between an assistant's
//! `tool_use` blocks and their `tool_result`s, a shape every real provider
//! rejects with a 400. The suite was green. A real request is the only thing
//! that checks it.
//!
//! So this covers what a mock structurally cannot:
//!
//! 1. **Transcript validity after loop detection** — including a *second*
//!    prompt on the same agent, because the abort used to leave an orphaned
//!    `tool_use` in history that poisoned every later call.
//! 2. **Stash retrieval through the model** — the model must actually follow
//!    the marker and receive content that is not in the head or tail. The
//!    existing test asserted a Rust-side `get`, one hop short of the real path.
//! 3. **Cost and usage accounting** against real provider numbers.
//! 4. **Prefix caching** — that a stable system prompt still produces cache
//!    reads, which the compaction marker's byte-stability depends on.
//!
//! Exit code is non-zero if any check fails, so this can gate a release.

use yoagent::agent::Agent;
use yoagent::context::ContextConfig;
use yoagent::provider::ModelConfig;
use yoagent::shared_state::SharedState;
use yoagent::types::{
    AgentMessage, AgentTool, Content, Message, ToolContext, ToolError, ToolResult,
};
use yoagent::*;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Always returns the same thing, so a model asked to retry will loop.
#[derive(Clone)]
struct StuckTool;

#[async_trait::async_trait]
impl AgentTool for StuckTool {
    fn name(&self) -> &str {
        "check_status"
    }
    fn label(&self) -> &str {
        "Check status"
    }
    fn description(&self) -> &str {
        "Check the deployment status. Returns the current state."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "service": { "type": "string" } },
            "required": ["service"]
        })
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "status: PENDING".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

/// Returns a long output with a unique needle buried in the middle, so a
/// head+tail truncation provably cannot contain it.
#[derive(Clone)]
struct BigOutputTool;

const NEEDLE: &str = "ZEPHYR-4417-QUINCE";

#[async_trait::async_trait]
impl AgentTool for BigOutputTool {
    fn name(&self) -> &str {
        "read_log"
    }
    fn label(&self) -> &str {
        "Read log"
    }
    fn description(&self) -> &str {
        "Read the build log. Returns the full log text."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let mut lines = Vec::new();
        for i in 0..2000 {
            if i == 1000 {
                lines.push(format!("line {i}: BUILD TOKEN {NEEDLE}"));
            } else {
                lines.push(format!("line {i}: routine build output, nothing notable"));
            }
        }
        Ok(ToolResult {
            content: vec![Content::Text {
                text: lines.join("\n"),
            }],
            details: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Report {
    checks: Vec<(String, bool, String)>,
}

impl Report {
    fn record(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
        let detail = detail.into();
        println!(
            "  {} {name}\n      {detail}",
            if ok { "PASS" } else { "FAIL" }
        );
        self.checks.push((name.to_string(), ok, detail));
    }
}

/// Drain an event stream, returning the assistant text and whether the run
/// reported a provider error.
async fn drain(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
) -> (String, Option<String>) {
    let mut text = String::new();
    let mut err = None;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::MessageUpdate {
                delta: StreamDelta::Text { delta },
                ..
            } => text.push_str(&delta),
            AgentEvent::MessageEnd {
                message:
                    AgentMessage::Llm(Message::Assistant {
                        stop_reason: StopReason::Error,
                        error_message,
                        ..
                    }),
            } => {
                err = Some(error_message.unwrap_or_else(|| "unknown provider error".into()));
            }
            _ => {}
        }
    }
    (text, err)
}

/// Every `tool_use` must be answered by its `tool_result`s before anything
/// else. This is the invariant the release-gate bug violated.
fn transcript_is_well_formed(messages: &[AgentMessage]) -> Result<(), String> {
    let mut pending: Vec<String> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let AgentMessage::Llm(msg) = m else { continue };
        match msg {
            Message::Assistant { content, .. } => {
                if !pending.is_empty() {
                    return Err(format!("assistant at [{i}] while {pending:?} unanswered"));
                }
                pending = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::ToolCall { id, .. } => Some(id.clone()),
                        _ => None,
                    })
                    .collect();
            }
            Message::ToolResult { tool_call_id, .. } => {
                if let Some(at) = pending.iter().position(|p| p == tool_call_id) {
                    pending.remove(at);
                }
            }
            Message::User { .. } => {
                if !pending.is_empty() {
                    return Err(format!(
                        "user message at [{i}] lands between an assistant's tool_use and its \
                         tool_result — {pending:?} unanswered. This is the shape every provider \
                         rejects."
                    ));
                }
            }
        }
    }
    if !pending.is_empty() {
        return Err(format!("run ended with {pending:?} unanswered"));
    }
    Ok(())
}

fn model() -> ModelConfig {
    match std::env::var("SMOKE_MODEL").ok().as_deref() {
        Some("gpt") => ModelConfig::openai("gpt-5.5", "GPT-5.5"),
        Some("gemini") => ModelConfig::google("gemini-3-pro", "Gemini 3 Pro"),
        _ => ModelConfig::claude_sonnet_5(),
    }
}

#[tokio::main]
async fn main() {
    let cfg = model();
    println!("\nyoagent release smoke — live provider: {}\n", cfg.name);

    let mut report = Report { checks: Vec::new() };

    // -----------------------------------------------------------------
    // 1. Loop detection produces a transcript the provider accepts, and
    //    leaves the agent usable afterwards.
    // -----------------------------------------------------------------
    println!("[1/4] loop detection — transcript validity and agent reuse");
    {
        let mut agent = Agent::from_config(cfg.clone())
            .with_system_prompt(
                "You are checking a deployment. If the status is PENDING, check again. \
                 Keep checking until it changes. Do not give up.",
            )
            .with_tools(vec![Box::new(StuckTool) as Box<dyn AgentTool>]);

        let mut rx = agent.prompt("Check the status of service 'api'.").await;
        let (_text, err) = drain(&mut rx).await;

        report.record(
            "loop run completes without a provider error",
            err.is_none(),
            err.clone()
                .unwrap_or_else(|| "no provider error".to_string()),
        );

        match transcript_is_well_formed(agent.messages()) {
            Ok(()) => report.record(
                "transcript is well-formed after loop detection",
                true,
                format!(
                    "{} messages, every tool_use answered",
                    agent.messages().len()
                ),
            ),
            Err(e) => report.record("transcript is well-formed after loop detection", false, e),
        }

        // The poisoning check. The abort used to leave an orphaned tool_use in
        // history, so this second prompt is what a real user hits next.
        let mut rx2 = agent
            .prompt("Never mind. Reply with the single word: ok")
            .await;
        let (text2, err2) = drain(&mut rx2).await;
        report.record(
            "agent is still usable after a loop abort",
            err2.is_none(),
            err2.unwrap_or_else(|| format!("follow-up succeeded: {:?}", text2.trim())),
        );
    }

    // -----------------------------------------------------------------
    // 2. Truncation stash — the model follows the marker and gets the middle.
    // -----------------------------------------------------------------
    println!("\n[2/4] truncation stash — retrieval through the model");
    {
        let state = SharedState::new();
        let mut agent = Agent::from_config(cfg.clone())
            .with_system_prompt(
                "You inspect build logs. Tool output may be truncated; when it is, the marker \
                 tells you a shared_state key holding the full text. Use it.",
            )
            .with_tools(vec![Box::new(BigOutputTool) as Box<dyn AgentTool>])
            .with_shared_state(state)
            .with_context_config(ContextConfig {
                truncate_tool_output_on_append: true,
                tool_output_max_lines: 40,
                ..Default::default()
            });

        let mut rx = agent
            .prompt(
                "Read the build log and tell me the BUILD TOKEN. It is in the middle of the \
                 log, not near the start or end. Reply with just the token.",
            )
            .await;
        let (text, err) = drain(&mut rx).await;

        let found = text.contains(NEEDLE);
        report.record(
            "model retrieved stashed content it could not otherwise see",
            found && err.is_none(),
            if found {
                format!("recovered {NEEDLE} from the stash")
            } else {
                format!(
                    "did NOT recover the token — head+tail truncation cannot contain it, so the \
                     stash path failed. err={err:?} reply={:?}",
                    text.trim()
                )
            },
        );
    }

    // -----------------------------------------------------------------
    // 3. Cost and usage accounting against real numbers.
    // -----------------------------------------------------------------
    println!("\n[3/4] cost and usage accounting");
    {
        let mut agent = Agent::from_config(cfg.clone()).with_system_prompt("Be concise.");
        let mut rx = agent.prompt("Name three primary colours.").await;
        let (_t, err) = drain(&mut rx).await;

        let cost = agent.session_cost_usd();
        let ok = err.is_none() && cost.map(|c| c > 0.0).unwrap_or(false);
        report.record(
            "session cost is computed from real usage",
            ok,
            match cost {
                Some(c) => format!("session_cost_usd = ${c:.6}"),
                None => "session_cost_usd = None — the preset reports unpriced".to_string(),
            },
        );
    }

    // -----------------------------------------------------------------
    // 4. Prefix caching still produces cache reads.
    // -----------------------------------------------------------------
    println!("\n[4/4] prefix cache");
    {
        // A system prompt long enough to be worth caching.
        let big_prompt = format!(
            "You are a precise assistant. Reference material follows.\n{}",
            "Rust's ownership model moves values by default; borrows are checked. ".repeat(400)
        );

        let mut cache_read_total = 0u64;
        for round in 0..2 {
            let mut agent = Agent::from_config(cfg.clone()).with_system_prompt(&big_prompt);
            let mut rx = agent.prompt("Reply with the single word: ping").await;
            while let Some(event) = rx.recv().await {
                let AgentEvent::MessageEnd { message } = event else {
                    continue;
                };
                let AgentMessage::Llm(Message::Assistant { usage, .. }) = &message else {
                    continue;
                };
                if round == 1 {
                    cache_read_total += usage.cache_read;
                }
            }
        }
        report.record(
            "second run reads from the prefix cache",
            cache_read_total > 0,
            format!(
                "cache_read on run 2 = {cache_read_total} tokens{}",
                if cache_read_total == 0 {
                    " — 0 means caching is off for this provider/model, or the prefix moved"
                } else {
                    ""
                }
            ),
        );
    }

    // -----------------------------------------------------------------
    println!("\n{}", "-".repeat(72));
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|(_, ok, _)| !ok)
        .map(|(n, _, _)| n.as_str())
        .collect();
    println!(
        "{} of {} checks passed",
        report.checks.len() - failed.len(),
        report.checks.len()
    );
    if failed.is_empty() {
        println!("\nRelease smoke: PASS");
    } else {
        println!("\nRelease smoke: FAIL");
        for f in &failed {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
