//! Prefix-cache stability of compaction.
//!
//! Providers cache request prefixes — automatically on DeepSeek, explicitly via
//! `cache_control` on Anthropic. A cache hit requires the new request to share a
//! byte-identical prefix with the previous one, so any in-place rewrite of
//! history costs every token from the rewrite point onward.
//!
//! # Reproducing the published baseline
//!
//! The old-vs-new figures in `CHANGELOG.md` and `docs/concepts/prompt-caching.md`
//! compare against a real 0.14.2 checkout, not against 0.15 with the new
//! settings turned off — those are different things and give different answers.
//! To re-measure:
//!
//! ```text
//! git checkout <0.14.2 commit> -- src/      # NOT `git stash`: a clean tree
//! grep -c compact_headroom_turns src/context.rs   # must print 0
//! # strip the post-0.14.2 APIs from a copy of this file, run it, then:
//! git checkout HEAD -- src/
//! ```
//!
//! `git stash push src/` is *not* a substitute — with a clean working tree it
//! stashes nothing and silently measures the current code.
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

/// The tool mix a real coding agent produces, measured from 808 archived runs
/// of an agent built on yoagent: bash 41%, edit/write 36%, read 19%, search 4%.
/// Output shape differs sharply per tool, which is the whole reason the line
/// budget is per-tool.
fn tool_for_turn(turn: usize) -> &'static str {
    match turn % 100 {
        0..=40 => "bash",       // build logs, grep, git — head+tail suits these
        41..=76 => "edit_file", // a confirmation line
        77..=95 => "read_file", // paged by the tool itself
        _ => "search",
    }
}

thread_local! {
    static READ_CAP: std::cell::Cell<usize> =
        const { std::cell::Cell::new(yoagent::tools::DEFAULT_READ_MAX_LINES) };
}

/// Replay with the read tool's page size overridden.
fn replay_with_read_cap(turns: usize, config: &ContextConfig, read_cap: usize) -> Replay {
    READ_CAP.with(|c| c.set(read_cap));
    let r = replay(turns, config);
    READ_CAP.with(|c| c.set(yoagent::tools::DEFAULT_READ_MAX_LINES));
    r
}

/// Deterministic output whose length distribution reflects the tool.
fn tool_output(turn: usize, tool: &str) -> String {
    let lines = match tool {
        // cargo/grep/git: routinely hundreds of lines, occasionally thousands.
        "bash" => 40 + (turn * 37) % 700,
        // The read tool pages itself at DEFAULT_READ_MAX_LINES, so what reaches
        // the context is already bounded — source files run 100..1500 lines.
        "read_file" => (100 + (turn * 53) % 1400).min(READ_CAP.with(|c| c.get())),
        "search" => 10 + (turn * 17) % 120,
        // edit/write return a short confirmation.
        _ => 1 + (turn * 3) % 4,
    };
    (0..lines)
        .map(|i| format!("turn {turn} {tool} line {i}: {}", "data ".repeat(6)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_with_tool_call(turn: usize) -> AgentMessage {
    let tool = tool_for_turn(turn);
    AgentMessage::Llm(
        Message::assistant(
            vec![
                Content::Text {
                    text: format!("Turn {turn}: inspecting the workspace before the next edit."),
                },
                Content::tool_call(
                    format!("tc-{turn}"),
                    tool,
                    serde_json::json!({ "arg": format!("target-{turn}") }),
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
    let tool = tool_for_turn(turn);
    AgentMessage::Llm(Message::ToolResult {
        tool_call_id: format!("tc-{turn}"),
        tool_name: tool.into(),
        content: vec![Content::Text {
            text: tool_output(turn, tool),
        }],
        is_error: false,
        timestamp: 1_700_000_000_000 + turn as u64,
    })
}

/// Per-million-token input prices used to turn byte counts into dollars.
/// Hit rate alone is a misleading objective once a change also moves context
/// size — a bigger context can raise the ratio while raising the bill.
const PRICES: [(&str, f64, f64); 2] = [("deepseek", 0.27, 0.07), ("anthropic", 3.00, 0.30)];

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

    /// Input-token spend across the whole replay, at the named price point.
    fn cost_usd(&self, p_input: f64, p_cache: f64) -> f64 {
        let cached = (self.total_bytes - self.uncached_bytes) as f64 / 4.0;
        let uncached = self.uncached_bytes as f64 / 4.0;
        (cached * p_cache + uncached * p_input) / 1e6
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
    // Growth measurement, mirroring what the agent loop feeds the headroom
    // policy. Without this the replay would not model the shipped behaviour.
    let (mut growth_total, mut growth_samples) = (0usize, 0usize);
    let mut last_total: Option<usize> = None;

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
            result = yoagent::context::truncate_tool_output(result, config);
        }
        history.push(result);

        let live = yoagent::context::total_tokens(&history);
        if let Some(previous_total) = last_total {
            growth_samples += 1;
            growth_total += live.saturating_sub(previous_total);
        }
        let growth = if growth_samples > 0 {
            growth_total as f64 / growth_samples as f64
        } else {
            0.0
        };

        // The loop compacts immediately before streaming and writes the result
        // back into the live history, resolving the headroom policy first.
        let effective = ContextConfig {
            compact_target_ratio: config.effective_target_ratio(growth),
            ..config.clone()
        };
        history = compact_messages(std::mem::take(&mut history), &effective);
        last_total = Some(yoagent::context::total_tokens(&history));

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

/// The 0.14.2 settings, for comparison: cap every tool at 50 lines, apply it
/// only retroactively during compaction, and compact to whatever just fits.
fn legacy_config() -> ContextConfig {
    ContextConfig {
        tool_output_max_lines: 50,
        tool_output_max_lines_overrides: Default::default(),
        truncate_tool_output_on_append: false,
        compact_target_ratio: 1.0,
        ..session_config()
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// Thresholds are regression guards, not targets. The ceiling is set by request
// size over per-turn growth: a session that adds 1/50th of its context each
// turn cannot exceed ~98% however stable compaction is, because that new tail
// was never cached to begin with.
// ---------------------------------------------------------------------------

#[test]
fn compaction_preserves_the_prefix_cache_across_a_long_session() {
    let result = replay(300, &session_config());
    result.report("defaults");

    assert!(
        result.hit_rate() > 0.955,
        "prefix-cache hit rate regressed to {:.2}%",
        result.hit_rate() * 100.0
    );
    assert!(
        result.invalidations.len() <= 12,
        "history was rewritten on {} of 300 turns",
        result.invalidations.len()
    );
}

#[test]
fn defaults_beat_the_legacy_settings() {
    let legacy = replay(300, &legacy_config());
    let current = replay(300, &session_config());
    legacy.report("legacy settings");

    assert!(
        current.hit_rate() > legacy.hit_rate(),
        "defaults ({:.2}%) should beat legacy settings ({:.2}%)",
        current.hit_rate() * 100.0,
        legacy.hit_rate() * 100.0
    );
    assert!(
        current.invalidations.len() < legacy.invalidations.len(),
        "defaults should rewrite history less often ({} vs {})",
        current.invalidations.len(),
        legacy.invalidations.len()
    );
}

#[test]
fn read_output_is_exempt_from_head_tail_truncation() {
    // Cutting the middle out of a source file removes exactly what was asked
    // for; the read tool bounds itself by paging instead.
    let config = session_config();
    let big = (0..900)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let msg = AgentMessage::Llm(Message::ToolResult {
        tool_call_id: "tc-1".into(),
        tool_name: "read_file".into(),
        content: vec![Content::Text { text: big.clone() }],
        is_error: false,
        timestamp: 1,
    });
    let out = yoagent::context::truncate_tool_output(msg, &config);
    let AgentMessage::Llm(Message::ToolResult { content, .. }) = &out else {
        panic!("expected a tool result")
    };
    let Content::Text { text } = &content[0] else {
        panic!("expected text")
    };
    assert_eq!(
        text, &big,
        "read_file output must not be head+tail truncated"
    );

    // The same output from a command tool is capped.
    let msg = AgentMessage::Llm(Message::ToolResult {
        tool_call_id: "tc-2".into(),
        tool_name: "bash".into(),
        content: vec![Content::Text { text: big }],
        is_error: false,
        timestamp: 1,
    });
    let out = yoagent::context::truncate_tool_output(msg, &config);
    let AgentMessage::Llm(Message::ToolResult { content, .. }) = &out else {
        panic!("expected a tool result")
    };
    let Content::Text { text } = &content[0] else {
        panic!("expected text")
    };
    assert_eq!(text.lines().count(), config.tool_output_max_lines);
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

#[test]
fn unbounded_read_pages_cost_cache() {
    // Why `read_file` pages at all. Reads are ~19% of tool calls but the
    // largest token sink, and they are exempt from head+tail truncation
    // (cutting the middle out of a source file removes what was asked for), so
    // the tool's page size is what bounds them.
    //
    // Measured with the adaptive target OFF, to isolate the read page: with a
    // fixed ratio, every increase in page size costs cache monotonically.
    // `compact_headroom_turns` largely absorbs this by compacting harder as
    // growth rises, which is why the exact page size matters far less than
    // that one exists at all.
    let config = ContextConfig {
        compact_headroom_turns: None,
        ..session_config()
    };
    let p300 = replay_with_read_cap(300, &config, 300).hit_rate();
    let p500 = replay_with_read_cap(300, &config, 500).hit_rate();
    let p1000 = replay_with_read_cap(300, &config, 1000).hit_rate();
    let p2000 = replay_with_read_cap(300, &config, 2000).hit_rate();

    assert!(
        p500 > p1000 && p1000 > p2000,
        "larger read pages must cost cache under a fixed ratio: \
         500={:.2}% 1000={:.2}% 2000={:.2}%",
        p500 * 100.0,
        p1000 * 100.0,
        p2000 * 100.0
    );
    assert!(
        (p300 - p500).abs() < 0.015,
        "300 and 500 should sit on the same plateau ({:.2}% vs {:.2}%)",
        p300 * 100.0,
        p500 * 100.0
    );
    assert_eq!(yoagent::tools::DEFAULT_READ_MAX_LINES, 500);
}

/// and reports the mean turns between compactions.
fn mean_compaction_interval(turns: usize, base: &ContextConfig) -> (f64, usize) {
    let budget = base
        .max_context_tokens
        .saturating_sub(base.system_prompt_tokens);
    let mut history: Vec<AgentMessage> = vec![AgentMessage::Llm(
        Message::user("start").with_timestamp(1_700_000_000_000),
    )];
    let (mut growth_total, mut growth_samples) = (0usize, 0usize);
    let mut last_total: Option<usize> = None;
    let mut intervals: Vec<usize> = Vec::new();
    let mut last_compaction = 0usize;

    for turn in 1..=turns {
        history.push(assistant_with_tool_call(turn));
        let mut result = tool_result(turn);
        if base.truncate_tool_output_on_append {
            result = yoagent::context::truncate_tool_output(result, base);
        }
        history.push(result);

        let live = yoagent::context::total_tokens(&history);
        if let Some(previous) = last_total {
            growth_samples += 1;
            growth_total += live.saturating_sub(previous);
        }
        let growth = if growth_samples > 0 {
            growth_total as f64 / growth_samples as f64
        } else {
            0.0
        };

        if live > budget {
            let cfg = ContextConfig {
                compact_target_ratio: base.effective_target_ratio(growth),
                ..base.clone()
            };
            let candidate = compact_messages(history.clone(), &cfg);
            if render(&candidate) != render(&history) {
                intervals.push(turn - last_compaction);
                last_compaction = turn;
            }
            history = candidate;
        }
        last_total = Some(yoagent::context::total_tokens(&history));
    }

    let n = intervals.len();
    let mean = if n == 0 {
        0.0
    } else {
        intervals.iter().sum::<usize>() as f64 / n as f64
    };
    (mean, n)
}

#[test]
fn headroom_policy_holds_the_compaction_interval_across_session_lengths() {
    // A fixed ratio cannot know how fast the session is growing, so the room it
    // leaves collapses as history accumulates. The headroom policy targets the
    // interval directly and should hold it roughly flat.
    let fixed = ContextConfig {
        compact_headroom_turns: None,
        ..session_config()
    };
    let dynamic = session_config(); // headroom on by default

    let mut fixed_intervals = Vec::new();
    let mut dynamic_intervals = Vec::new();
    for turns in [300usize, 1200, 2400] {
        let (f, fc) = mean_compaction_interval(turns, &fixed);
        let (d, dc) = mean_compaction_interval(turns, &dynamic);
        println!(
            "{turns:>4} turns: fixed ratio interval {f:>5.1} ({fc} compactions)  |  \
             headroom interval {d:>5.1} ({dc} compactions)"
        );
        fixed_intervals.push(f);
        dynamic_intervals.push(d);
    }

    // The fixed ratio degrades markedly from the shortest to the longest run.
    let fixed_drop = fixed_intervals[0] - fixed_intervals[2];
    let dynamic_drop = dynamic_intervals[0] - dynamic_intervals[2];
    assert!(
        fixed_drop > 8.0,
        "expected the fixed ratio to degrade, got {fixed_intervals:?}"
    );
    assert!(
        dynamic_drop < fixed_drop / 2.0,
        "headroom policy should hold the interval far better: fixed {fixed_intervals:?} \
         vs headroom {dynamic_intervals:?}"
    );
    // And it must compact less often at the long horizon.
    assert!(
        dynamic_intervals[2] > fixed_intervals[2] * 1.2,
        "headroom interval {} should beat fixed {} at 2400 turns",
        dynamic_intervals[2],
        fixed_intervals[2]
    );
}

#[test]
fn report_cost_for_docs() {
    for turns in [300usize, 1200, 2400] {
        let r = replay(turns, &session_config());
        let costs: Vec<String> = PRICES
            .iter()
            .map(|(name, pi, pc)| format!("{name} ${:.4}", r.cost_usd(*pi, *pc)))
            .collect();
        println!(
            "NEW turns={turns}: {}  hit {:.2}%  rewrites {}",
            costs.join("  "),
            r.hit_rate() * 100.0,
            r.invalidations.len()
        );
    }
}
