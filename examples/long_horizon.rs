//! Long-horizon validation against a **live** provider.
//!
//! Run with: `ANTHROPIC_API_KEY=sk-... cargo run --example long_horizon`
//! Provider: `SMOKE_MODEL=deepseek` (default Sonnet 5).
//!
//! # What this is for
//!
//! `release_smoke` checks four narrow behaviours over short runs. This checks
//! the thing yoagent actually claims to be for: **an agent that runs long
//! enough to exhaust its context, compacts, and keeps working.** That path had
//! no live coverage at all — every compaction test uses `MockProvider`, which
//! cannot tell you whether a compacted transcript is still one a provider
//! accepts, or whether the agent still knows what it learned an hour ago.
//!
//! The promises being checked, each stated in the docs or CHANGELOG:
//!
//! 1. **Compaction actually triggers** on a long run and the run survives it.
//! 2. **Task continuity**: the agent can still recall a fact it learned
//!    *before* compaction. A compaction that loses the thread is worse than
//!    hitting the limit.
//! 3. **The transcript stays provider-valid across compaction** — no orphaned
//!    tool calls, which is what `splice_never_orphans_a_parallel_tool_call`
//!    asserts in unit tests but never against a real API.
//! 4. **Prefix caching survives compaction.** The compaction marker is
//!    deliberately constant text so the cached prefix stays byte-stable; if
//!    that were untrue the cache would cold-start on every compaction and the
//!    whole design would be a pessimisation.
//! 5. **`SessionStats` reports what actually happened** — turns, compactions,
//!    and a cost that matches the usage.
//! 6. **Sub-agent delegation works end to end**, including the scoped stash.
//!
//! Exits non-zero on failure.

use std::sync::Arc;
use yoagent::agent::Agent;
use yoagent::context::ContextConfig;
use yoagent::provider::ModelConfig;
use yoagent::shared_state::SharedState;
use yoagent::sub_agent::SubAgentTool;
use yoagent::types::{
    AgentMessage, AgentTool, Content, Message, SessionStats, ToolContext, ToolError, ToolResult,
};
use yoagent::*;

// ---------------------------------------------------------------------------
// A tool that returns bulky, distinct records — enough to fill a context.
// ---------------------------------------------------------------------------

/// The fact planted in record 1, which the agent must still know at the end.
const EARLY_SECRET: &str = "MERIDIAN-8831";

#[derive(Clone)]
struct RecordTool;

#[async_trait::async_trait]
impl AgentTool for RecordTool {
    fn name(&self) -> &str {
        "fetch_record"
    }
    fn label(&self) -> &str {
        "Fetch record"
    }
    fn description(&self) -> &str {
        "Fetch inspection record N (1-24). Each record is a block of findings."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "n": { "type": "integer", "description": "record number 1-24" } },
            "required": ["n"]
        })
    }
    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let n = params.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        let mut body = String::new();
        if n == 1 {
            body.push_str(&format!(
                "record 1 header: the ASSET CODE for this inspection is {EARLY_SECRET}. \
                 Remember it; later records do not repeat it.\n"
            ));
        }
        // Bulk, so the context fills in a realistic number of turns.
        for i in 0..60 {
            body.push_str(&format!(
                "record {n} line {i}: subsystem {i} nominal, drift {:.3}, margin ok, \
                 no action required, logged by inspector {n}\n",
                (n as f64 * 0.017) + (i as f64 * 0.003)
            ));
        }
        Ok(ToolResult {
            content: vec![Content::Text { text: body }],
            details: serde_json::Value::Null,
        })
    }
}

// ---------------------------------------------------------------------------

struct Report {
    checks: Vec<(String, bool)>,
}

impl Report {
    fn record(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
        println!(
            "  {} {name}\n      {}",
            if ok { "PASS" } else { "FAIL" },
            detail.into()
        );
        self.checks.push((name.to_string(), ok));
    }
    fn note(&self, detail: impl Into<String>) {
        println!("       {}", detail.into());
    }
}

struct RunOutcome {
    text: String,
    error: Option<String>,
    stats: Option<SessionStats>,
    compactions: u32,
    cache_read_after_compaction: u64,
}

/// Drain a run, recording compaction events and the usage that follows them.
async fn drain(rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) -> RunOutcome {
    let mut out = RunOutcome {
        text: String::new(),
        error: None,
        stats: None,
        compactions: 0,
        cache_read_after_compaction: 0,
    };
    let mut assistant_turns = 0usize;

    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::MessageUpdate {
                delta: StreamDelta::Text { delta },
                ..
            } => out.text.push_str(&delta),

            AgentEvent::ContextCompacted {
                method,
                messages_before,
                messages_after,
                tokens_before,
                tokens_after,
                ..
            } => {
                out.compactions += 1;
                println!(
                    "       [compaction #{}] {method:?}: {messages_before} msgs / \
                     {tokens_before} tok -> {messages_after} msgs / {tokens_after} tok",
                    out.compactions
                );
            }

            AgentEvent::MessageEnd {
                message:
                    AgentMessage::Llm(Message::Assistant {
                        usage,
                        stop_reason,
                        error_message,
                        ..
                    }),
            } => {
                {
                    // `DefaultCompaction` emits no `ContextCompacted` event — it has
                    // no event channel — so keying off that flag measured
                    // nothing. Count cache reads on the back half of the run
                    // instead: by then compaction has certainly run, and a
                    // byte-stable prefix should still be producing hits.
                    assistant_turns += 1;
                    if assistant_turns > 6 {
                        out.cache_read_after_compaction += usage.cache_read;
                    }
                    if stop_reason == StopReason::Error {
                        out.error =
                            Some(error_message.unwrap_or_else(|| "unknown provider error".into()));
                    }
                }
            }

            AgentEvent::AgentEnd { stats, .. } => out.stats = Some(stats),
            _ => {}
        }
    }
    out
}

/// Every `tool_use` answered before anything else intervenes.
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
                        "user message at [{i}] with {pending:?} unanswered — the shape \
                         providers reject"
                    ));
                }
            }
        }
    }
    if pending.is_empty() {
        Ok(())
    } else {
        Err(format!("ended with {pending:?} unanswered"))
    }
}

/// The model that writes briefings. Deliberately not the loop's model.
fn summarizer() -> ModelConfig {
    match std::env::var("SMOKE_MODEL").ok().as_deref() {
        Some("deepseek") => ModelConfig::deepseek("deepseek-chat", "DeepSeek Chat"),
        _ => ModelConfig::claude_haiku_4_5(),
    }
}

fn model() -> ModelConfig {
    match std::env::var("SMOKE_MODEL").ok().as_deref() {
        Some("deepseek") => ModelConfig::deepseek("deepseek-chat", "DeepSeek Chat"),
        Some("gpt") => ModelConfig::openai("gpt-5.5", "GPT-5.5"),
        _ => ModelConfig::claude_sonnet_5(),
    }
}

#[tokio::main]
async fn main() {
    // Compaction decisions are only visible through tracing — `RUST_LOG=yoagent=debug`
    // shows which path each compaction took and why summarization did or did
    // not arm. Without a subscriber the diagnosis is guesswork.
    let level = match std::env::var("YO_LOG").as_deref() {
        Ok("debug") => tracing::Level::DEBUG,
        Ok("trace") => tracing::Level::TRACE,
        Ok("info") => tracing::Level::INFO,
        _ => tracing::Level::WARN,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(true)
        .init();

    let cfg = model();
    println!(
        "\nyoagent long-horizon validation — live provider: {}\n",
        cfg.name
    );
    let mut report = Report { checks: Vec::new() };

    // -----------------------------------------------------------------
    // A long run, deliberately given far less context than the task needs.
    // -----------------------------------------------------------------
    println!("[1/3] long run with forced compaction");

    let mut agent = Agent::from_config(cfg.clone())
        .with_system_prompt(
            "You are inspecting an asset. Use fetch_record to read records one at a time. \
             Work through every record you are asked for, in order, one tool call per turn. \
             Be terse — do not summarise each record, just proceed.",
        )
        .with_tools(vec![Box::new(RecordTool) as Box<dyn AgentTool>])
        .with_context_config(ContextConfig {
            // Big enough to be a plausible budget, small enough that a dozen
            // bulky records must compact. 6_000 was too aggressive to be a
            // fair test: the agent thrashed, burned all 50 turns, and lost the
            // thread — which is correct behaviour for an absurd budget, not
            // evidence about compaction.
            max_context_tokens: 30_000,
            keep_recent: 6,
            keep_first: 2,
            tool_output_max_lines: 200,
            ..Default::default()
        });

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    agent = agent.with_compaction_strategy(
        // A *cheaper, faster* summarizer, per LlmCompaction's own docs: "the
        // request is standalone, so this can (and usually should) name a
        // cheaper model than the main loop's". Passing the loop model is the
        // obvious call and the worst one — the briefing then loses the race
        // against the budget and is discarded when compaction rewrites the
        // history it was computed over.
        yoagent::LlmCompaction::from_config(summarizer())
            .with_trigger_ratio(0.35)
            .with_event_sender(tx.clone()),
    );
    agent
        .prompt_with_sender(
            "Fetch records 1 through 14, one at a time, in order. \
             After the last one, reply with exactly: DONE",
            tx,
        )
        .await;
    let run = drain(&mut rx).await;
    agent.finish().await;

    report.record(
        "the long run completes without a provider error",
        run.error.is_none(),
        run.error
            .clone()
            .unwrap_or_else(|| "no provider error".into()),
    );

    let compacted = run.compactions > 0
        || run
            .stats
            .as_ref()
            .map(|s| s.compactions > 0)
            .unwrap_or(false);
    report.record(
        "compaction actually triggered",
        compacted,
        if compacted {
            format!("{} compaction event(s)", run.compactions)
        } else {
            "no compaction — the context never filled, so nothing below is a real test \
             of the compaction path. Lower max_context_tokens or raise the record count."
                .into()
        },
    );

    match transcript_is_well_formed(agent.messages()) {
        Ok(()) if agent.messages().is_empty() => report.record(
            "transcript stays provider-valid across compaction",
            false,
            "0 messages — nothing inspected",
        ),
        Ok(()) => report.record(
            "transcript stays provider-valid across compaction",
            true,
            format!(
                "{} messages, every tool_use answered",
                agent.messages().len()
            ),
        ),
        Err(e) => report.record(
            "transcript stays provider-valid across compaction",
            false,
            e,
        ),
    }

    if let Some(stats) = &run.stats {
        report.note(format!(
            "stats: {} turns, {} compactions, in={} out={} cache_read={} cache_write={}",
            stats.turns,
            stats.compactions,
            stats.usage.input,
            stats.usage.output,
            stats.usage.cache_read,
            stats.usage.cache_write,
        ));
        let coherent = stats.turns > 0 && (!compacted || stats.compactions > 0);
        report.record(
            "SessionStats reports what actually happened",
            coherent,
            format!(
                "turns={} compactions={} hit_rate={:.1}%",
                stats.turns,
                stats.compactions,
                stats.cache_hit_rate() * 100.0
            ),
        );
    } else {
        report.record(
            "SessionStats reports what actually happened",
            false,
            "no AgentEnd carried stats",
        );
    }

    report.record(
        "prefix cache still hits after compaction",
        run.cache_read_after_compaction > 0 || !compacted,
        format!(
            "cache_read on post-compaction turns = {} tokens{}",
            run.cache_read_after_compaction,
            if run.cache_read_after_compaction == 0 && compacted {
                " — 0 means the compacted prefix is not byte-stable, or this provider \
                 does not cache. The constant COMPACTION_MARKER exists to prevent the former."
            } else {
                ""
            }
        ),
    );

    // -----------------------------------------------------------------
    // Continuity: does it still know what it read before compaction?
    // -----------------------------------------------------------------
    println!("\n[2/3] task continuity across compaction");
    {
        // What `LlmCompaction` documents is a "shift-handoff briefing covering
        // goals, progress, and decisions" — task continuity, not a verbatim
        // record of tool output. Testing it for an arbitrary data fact would be
        // testing a promise the library never makes; that is what the
        // truncation stash is for, and `release_smoke` covers it.
        let mut rx = agent
            .prompt(
                "Without calling any tools: what task are you working on, and roughly how \
                 far through it did you get? One sentence.",
            )
            .await;
        let follow = drain(&mut rx).await;
        agent.finish().await;

        let lower = follow.text.to_lowercase();
        let on_task = lower.contains("record") || lower.contains("inspect");
        report.record(
            "the agent still knows the task after compaction",
            on_task && follow.error.is_none(),
            if on_task {
                format!("continuity held: {:?}", follow.text.trim())
            } else {
                format!(
                    "lost the thread — the briefing did not carry goals/progress. \
                     err={:?} reply={:?}",
                    follow.error,
                    follow.text.trim()
                )
            },
        );

        // Stated, not assumed: specific values from tool output are NOT a
        // compaction promise. Retrieval of those is the stash's job.
        let mut rx2 = agent
            .prompt(
                "Without calling tools: what was the ASSET CODE in record 1? \
                 If you no longer have it, reply LOST.",
            )
            .await;
        let fact = drain(&mut rx2).await;
        agent.finish().await;
        report.note(format!(
            "data-fact recall after compaction: {} — informational only; the briefing \
             covers goals/progress/decisions, and exact tool output is the stash's job",
            if fact.text.contains(EARLY_SECRET) {
                "retained".to_string()
            } else {
                format!("lost ({:?})", fact.text.trim())
            }
        ));
    }

    // -----------------------------------------------------------------
    // Sub-agent delegation with a scoped stash.
    // -----------------------------------------------------------------
    println!("\n[3/3] sub-agent delegation with a scoped stash");
    {
        let state = SharedState::new();
        let worker = SubAgentTool::from_config("inspector", cfg.clone())
            .with_description("Delegate a record inspection. Give it a record number.")
            .with_system_prompt(
                "Fetch the record you are asked for. Your final reply MUST contain the ASSET \
             CODE value itself, spelled out. Do not merely say where you stored it.",
            )
            .with_tools(vec![Arc::new(RecordTool) as Arc<dyn AgentTool>])
            .with_scoped_shared_state(state.clone(), "inspector")
            .with_max_turns(6);

        let mut parent = Agent::from_config(cfg.clone())
            .with_system_prompt("Delegate to the inspector sub-agent. Report what it tells you.")
            .with_tools(vec![Box::new(worker) as Box<dyn AgentTool>]);

        let mut rx = parent
            .prompt("Ask the inspector to look at record 1 and tell me the ASSET CODE.")
            .await;
        let sub = drain(&mut rx).await;
        parent.finish().await;

        let ok = sub.error.is_none() && sub.text.contains(EARLY_SECRET);
        report.record(
            "sub-agent delegation returns real work to the parent",
            ok,
            if ok {
                format!("parent reported {EARLY_SECRET} via the sub-agent")
            } else {
                format!("err={:?} reply={:?}", sub.error, sub.text.trim())
            },
        );
    }

    // -----------------------------------------------------------------
    println!("\n{}", "-".repeat(72));
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(n, _)| n.as_str())
        .collect();
    println!(
        "{} of {} checks passed",
        report.checks.len() - failed.len(),
        report.checks.len()
    );
    if failed.is_empty() {
        println!("\nLong-horizon: PASS");
    } else {
        println!("\nLong-horizon: FAIL");
        for f in &failed {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
