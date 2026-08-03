//! Prefix-cache stability of compaction.
//!
//! Providers cache request prefixes — automatically on DeepSeek, explicitly via
//! `cache_control` on Anthropic. A cache hit requires the new request to share a
//! byte-identical prefix with the previous one, so any in-place rewrite of
//! history costs every token from the rewrite point onward.
//!
//! These tests replay a synthetic tool-heavy session through `compact_messages`
//! turn by turn, render each turn's message list the way a provider body would
//! see it (timestamps excluded — no provider serializes them), and measure the
//! shared prefix between consecutive turns. The resulting ratio is the direct
//! analogue of DeepSeek's `cache_hit_tokens / input_tokens`.

use yoagent::context::{compact_messages, ContextConfig};
use yoagent::types::{AgentMessage, Content, Message, StopReason, Usage};

// ---------------------------------------------------------------------------
// Wire rendering — what the provider actually receives
// ---------------------------------------------------------------------------

fn render(messages: &[AgentMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let AgentMessage::Llm(m) = msg else { continue };
        match m {
            Message::User { content, .. } => {
                out.push_str("<user>");
                render_content(content, &mut out);
            }
            Message::Assistant { content, .. } => {
                out.push_str("<assistant>");
                render_content(content, &mut out);
            }
            Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                out.push_str("<tool_result id=");
                out.push_str(tool_call_id);
                out.push_str(if *is_error { " error>" } else { ">" });
                render_content(content, &mut out);
            }
        }
    }
    out
}

fn render_content(content: &[Content], out: &mut String) {
    for c in content {
        match c {
            Content::Text { text } => {
                out.push_str("<text>");
                out.push_str(text);
            }
            Content::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                out.push_str("<tool_use id=");
                out.push_str(id);
                out.push(' ');
                out.push_str(name);
                out.push('>');
                out.push_str(&arguments.to_string());
            }
            Content::Thinking { thinking, .. } => {
                out.push_str("<thinking>");
                out.push_str(thinking);
            }
            Content::Image { data, .. } => {
                out.push_str("<image>");
                out.push_str(data);
            }
            _ => out.push_str("<other>"),
        }
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count()
}

// ---------------------------------------------------------------------------
// Synthetic session
// ---------------------------------------------------------------------------

fn tool_output(turn: usize) -> String {
    // Deterministic, varied length: some outputs cross tool_output_max_lines,
    // some don't — the mix a real coding agent produces.
    let lines = 20 + (turn * 37) % 160;
    (0..lines)
        .map(|i| format!("turn {turn} output line {i}: {}", "data ".repeat(6)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_with_tool_call(turn: usize) -> AgentMessage {
    AgentMessage::Llm(
        Message::assistant(
            vec![
                Content::Text {
                    text: format!("Turn {turn}: inspecting the workspace before the next edit."),
                },
                Content::tool_call(
                    format!("tc-{turn}"),
                    "bash",
                    serde_json::json!({ "command": format!("rg --files -g '*.rs' | sed -n '{turn}p'") }),
                ),
            ],
            StopReason::ToolUse,
            "test-model",
            "test",
            Usage::default(),
        )
        .with_timestamp(1_700_000_000_000 + turn as u64),
    )
}

fn tool_result(turn: usize) -> AgentMessage {
    AgentMessage::Llm(Message::ToolResult {
        tool_call_id: format!("tc-{turn}"),
        tool_name: "bash".into(),
        content: vec![Content::Text {
            text: tool_output(turn),
        }],
        is_error: false,
        timestamp: 1_700_000_000_000 + turn as u64,
    })
}

struct Replay {
    /// Bytes the provider would have to re-process (no cache hit).
    uncached_bytes: usize,
    /// Total request bytes across all turns.
    total_bytes: usize,
    /// Turns whose shared prefix with the previous request was shorter than
    /// the previous request itself — i.e. history was rewritten.
    invalidations: Vec<Invalidation>,
    turns: usize,
}

struct Invalidation {
    turn: usize,
    /// Fraction of the previous request that survived as a shared prefix.
    retained: f64,
    /// Index of the first message that differs from the previous request.
    diverged_at: usize,
    /// Messages in the previous request.
    prev_messages: usize,
    /// Messages in this request.
    messages: usize,
}

/// Index of the first message whose rendering differs.
fn first_divergence(prev: &[String], cur: &[String]) -> usize {
    prev.iter().zip(cur).take_while(|(a, b)| a == b).count()
}

impl Replay {
    fn hit_rate(&self) -> f64 {
        1.0 - (self.uncached_bytes as f64 / self.total_bytes as f64)
    }

    fn report(&self, label: &str) {
        println!(
            "{label}: {} turns, prefix-cache hit rate {:.2}%, {} invalidations",
            self.turns,
            self.hit_rate() * 100.0,
            self.invalidations.len()
        );
        for inv in &self.invalidations {
            println!(
                "  turn {:>3}: retained {:>5.1}% | diverged at message {}/{} (now {} messages)",
                inv.turn,
                inv.retained * 100.0,
                inv.diverged_at,
                inv.prev_messages,
                inv.messages
            );
        }
    }
}

/// Replay `turns` turns of a tool-heavy session, compacting before each turn
/// exactly as the agent loop does, and measure prefix reuse between turns.
fn replay(turns: usize, config: &ContextConfig) -> Replay {
    let mut history: Vec<AgentMessage> = vec![AgentMessage::Llm(
        Message::user("Refactor the provider layer.").with_timestamp(1_700_000_000_000),
    )];
    let mut previous = String::new();
    let mut previous_parts: Vec<String> = Vec::new();
    let mut uncached_bytes = 0usize;
    let mut total_bytes = 0usize;
    let mut invalidations = Vec::new();

    for turn in 1..=turns {
        if turn % 5 == 0 {
            history.push(AgentMessage::Llm(
                Message::user(format!("Also check item {turn} while you are there."))
                    .with_timestamp(1_700_000_000_000 + turn as u64),
            ));
        }
        history.push(assistant_with_tool_call(turn));
        let mut result = tool_result(turn);
        if config.truncate_tool_output_on_append {
            result = yoagent::context::truncate_tool_output(result, config.tool_output_max_lines);
        }
        history.push(result);

        // The loop compacts immediately before streaming and writes the result
        // back into the live history.
        history = compact_messages(std::mem::take(&mut history), config);

        let parts: Vec<String> = history
            .iter()
            .map(|m| render(std::slice::from_ref(m)))
            .collect();
        let current = parts.concat();
        let shared = common_prefix_len(&previous, &current);
        uncached_bytes += current.len() - shared;
        total_bytes += current.len();

        if !previous.is_empty() && shared < previous.len() {
            invalidations.push(Invalidation {
                turn,
                retained: shared as f64 / previous.len() as f64,
                diverged_at: first_divergence(&previous_parts, &parts),
                prev_messages: previous_parts.len(),
                messages: parts.len(),
            });
        }
        previous = current;
        previous_parts = parts;
    }

    Replay {
        uncached_bytes,
        total_bytes,
        invalidations,
        turns,
    }
}

/// A tool-heavy session on a mid-sized context window: compaction engages
/// part-way through and then cycles, which is the regime prefix-cache
/// behaviour actually matters in.
fn session_config() -> ContextConfig {
    ContextConfig {
        max_context_tokens: 102_400, // a 128K window at the default 80% reserve
        system_prompt_tokens: 4_000,
        ..Default::default()
    }
}

/// The same session with oversized tool output capped as it is appended.
fn on_append_config() -> ContextConfig {
    ContextConfig {
        truncate_tool_output_on_append: true,
        ..session_config()
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// Thresholds are regression guards, not targets. The measured figures for this
// replay at 300 turns are:
//
//   yoagent 0.14.2                          95.80%   16 rewrites
//   + idempotent truncation, hysteresis      96.85%   15 rewrites
//   + truncate_tool_output_on_append         98.51%    2 rewrites
//
// The ceiling is set by request size over per-turn growth: a session that adds
// 1/50th of its context each turn cannot exceed ~98% however stable compaction
// is, because that new tail was never cached to begin with.
// ---------------------------------------------------------------------------

#[test]
fn compaction_preserves_the_prefix_cache_across_a_long_session() {
    let result = replay(300, &session_config());
    result.report("default");

    assert!(
        result.hit_rate() > 0.96,
        "prefix-cache hit rate regressed to {:.2}%",
        result.hit_rate() * 100.0
    );
}

#[test]
fn truncating_on_append_keeps_the_prefix_cache_nearly_intact() {
    let result = replay(300, &on_append_config());
    result.report("truncate-on-append");

    // Capping output on the way in removes the retroactive rewrites entirely,
    // leaving only the genuine compaction events.
    assert!(
        result.hit_rate() > 0.98,
        "prefix-cache hit rate regressed to {:.2}%",
        result.hit_rate() * 100.0
    );
    assert!(
        result.invalidations.len() <= 4,
        "history was rewritten on {} of 300 turns",
        result.invalidations.len()
    );
}

#[test]
fn compaction_does_not_fire_every_turn_once_over_budget() {
    // Without headroom, the first turn past the budget puts the session into a
    // state where every subsequent turn re-compacts and rewrites history.
    let result = replay(300, &session_config());

    assert!(
        result.invalidations.len() <= 25,
        "history was rewritten on {} of 300 turns; compaction is firing far too often",
        result.invalidations.len()
    );
}

#[test]
fn compaction_is_deterministic() {
    // Non-deterministic output (wall-clock timestamps, drifting markers) would
    // make prefix reuse impossible to reason about — and is itself a rewrite.
    let config = session_config();
    let a = replay(60, &config);
    let b = replay(60, &config);
    assert_eq!(a.uncached_bytes, b.uncached_bytes);
    assert_eq!(a.total_bytes, b.total_bytes);
}

#[test]
fn repeated_compaction_of_settled_history_is_a_no_op() {
    // Once compaction has run, running it again on the same history with the
    // same config must not change a single byte — otherwise every turn past the
    // budget pays a fresh invalidation.
    //
    // Two budgets, so both the truncate-only path and the drop path are
    // covered: 50K is reachable by Level 1 alone, 8K forces Level 2/3.
    for max_context_tokens in [50_000usize, 8_000] {
        let config = ContextConfig {
            max_context_tokens,
            system_prompt_tokens: 0,
            ..Default::default()
        };
        let mut history: Vec<AgentMessage> = vec![AgentMessage::Llm(
            Message::user("start").with_timestamp(1_700_000_000_000),
        )];
        for turn in 1..=60 {
            history.push(assistant_with_tool_call(turn));
            history.push(tool_result(turn));
        }

        let once = compact_messages(history, &config);
        let twice = compact_messages(once.clone(), &config);
        let thrice = compact_messages(twice.clone(), &config);

        assert_eq!(
            render(&once),
            render(&twice),
            "second compaction pass rewrote settled history at {max_context_tokens} tokens"
        );
        assert_eq!(
            render(&twice),
            render(&thrice),
            "third compaction pass rewrote settled history at {max_context_tokens} tokens"
        );
    }
}
