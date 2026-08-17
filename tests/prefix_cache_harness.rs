//! Prefix-cache measurement harness.
//!
//! Every rewrite of an already-sent message discards the provider's prefix cache
//! from that point on, so "how often does a session rewrite its own history?" is
//! the number that decides whether prompt caching pays. This harness measures it
//! by driving a [`CompactionStrategy`] the way [`agent_loop`] does — append a
//! turn, compact, **feed the result back** — and counting the rounds where the
//! returned history is not an extension of what went in.
//!
//! # Why this is a shared harness, not a per-feature script
//!
//! Prefix-cache effectiveness has two independent halves, and they are worked on
//! separately:
//!
//! 1. **History stability** — does compaction rewrite the prefix? Measured here,
//!    strategy-agnostic, via [`measure`].
//! 2. **Breakpoint placement** — do providers actually emit `cache_control` (or
//!    the OpenAI / Gemini equivalents) at the stable boundary? Not measured yet;
//!    `CacheStrategy` is currently honoured only by `anthropic.rs`.
//!
//! Work on (2) should extend this file with provider-side assertions rather than
//! reimplement session simulation. [`SessionShape`] and [`Report`] are the seam:
//! a parity test wants the same driven session, with the emitted request bodies
//! inspected instead of (or alongside) the returned history.
//!
//! # Running
//!
//! The measurements are `#[ignore]`d — they take minutes and are a benchmark,
//! not a pass/fail gate. There is one cheap non-ignored test that keeps the
//! harness itself honest.
//!
//! ```text
//! cargo test --test prefix_cache_harness -- --ignored --nocapture
//! ```
//!
//! [`agent_loop`]: yoagent::agent_loop

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use yoagent::context::{total_tokens, CompactionStrategy, ContextConfig, DefaultCompaction};
use yoagent::llm_compaction::{LlmCompaction, SUMMARY_MARKER};
use yoagent::provider::{ModelConfig, ProviderError, StreamConfig, StreamEvent, StreamProvider};
use yoagent::types::*;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// The session to simulate.
#[derive(Debug, Clone)]
pub struct SessionShape {
    /// `max_context_tokens`; `system_prompt_tokens` is held at 0 so the budget
    /// is exactly this number.
    pub budget: usize,
    /// Approximate tokens added per turn.
    pub tokens_per_turn: usize,
    pub rounds: usize,
    pub keep_first: usize,
    pub keep_recent: usize,
}

impl SessionShape {
    pub fn new(budget: usize, tokens_per_turn: usize, rounds: usize) -> Self {
        Self {
            budget,
            tokens_per_turn,
            rounds,
            keep_first: 2,
            keep_recent: 10,
        }
    }

    fn context_config(&self) -> ContextConfig {
        ContextConfig {
            max_context_tokens: self.budget,
            system_prompt_tokens: 0,
            keep_first: self.keep_first,
            keep_recent: self.keep_recent,
            ..Default::default()
        }
    }
}

/// What a driven session did.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Rounds where the returned history was **not** an extension of the input
    /// — i.e. the provider's prefix cache would have been discarded.
    pub cache_breaks: usize,
    /// Which rounds those were, for spotting whether breaks cluster or spread.
    pub break_rounds: Vec<usize>,
    pub peak_tokens: usize,
    pub final_tokens: usize,
    /// Rounds whose result contained a spliced LLM briefing.
    pub rounds_with_summary: usize,
}

impl Report {
    /// Mean rounds between rewrites — the interval prompt caching gets to work
    /// with. `None` when nothing was ever rewritten.
    pub fn mean_interval(&self) -> Option<f64> {
        (self.cache_breaks > 1).then(|| {
            let first = *self.break_rounds.first().unwrap() as f64;
            let last = *self.break_rounds.last().unwrap() as f64;
            (last - first) / (self.cache_breaks - 1) as f64
        })
    }
}

/// Number of leading messages the two histories share verbatim.
fn common_prefix(before: &[AgentMessage], after: &[AgentMessage]) -> usize {
    before
        .iter()
        .zip(after.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn text_turn(i: usize, tokens: usize) -> Vec<AgentMessage> {
    // ~4 bytes per token, split across the user and assistant halves.
    let bulk = (tokens * 4) / 2;
    vec![
        AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: format!("u{i}: {}", "x".repeat(bulk)),
            }],
            timestamp: i as u64,
        }),
        AgentMessage::Llm(
            Message::assistant(
                vec![Content::Text {
                    text: format!("a{i}: {}", "y".repeat(bulk)),
                }],
                StopReason::Stop,
                "harness",
                "harness",
                Usage::default(),
            )
            .with_timestamp(i as u64),
        ),
    ]
}

fn has_summary(messages: &[AgentMessage]) -> bool {
    messages.iter().any(|m| {
        matches!(m, AgentMessage::Llm(Message::User { content, .. })
            if content.iter().any(|c| matches!(c, Content::Text { text }
                if text.starts_with(SUMMARY_MARKER))))
    })
}

/// Drive `strategy` through a simulated session and report prefix stability.
///
/// The feed-the-result-back step is the whole point: it is what
/// `agent_loop.rs` does, and measuring against pristine input instead would
/// describe a call pattern the loop never uses.
pub async fn measure(strategy: &dyn CompactionStrategy, shape: &SessionShape) -> Report {
    let config = shape.context_config();
    let mut messages: Vec<AgentMessage> = Vec::new();
    let mut report = Report::default();

    for round in 0..shape.rounds {
        messages.extend(text_turn(round, shape.tokens_per_turn));
        let before = messages.clone();
        messages = strategy.compact(std::mem::take(&mut messages), &config);

        if common_prefix(&before, &messages) < before.len() {
            report.cache_breaks += 1;
            report.break_rounds.push(round);
        }
        if has_summary(&messages) {
            report.rounds_with_summary += 1;
        }
        report.peak_tokens = report.peak_tokens.max(total_tokens(&messages));

        // Let any background summarization land, as wall-clock between real
        // turns would. Cheap and harmless for synchronous strategies.
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }
    report.final_tokens = total_tokens(&messages);
    report
}

/// Counts summarization requests so a run can show cost alongside benefit.
struct CountingProvider {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl StreamProvider for CountingProvider {
    async fn stream(
        &self,
        _config: StreamConfig,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::assistant(
            vec![Content::Text {
                text: "## Goal\nHarness briefing.\n## Open items\nNone.".into(),
            }],
            StopReason::Stop,
            "harness",
            "harness",
            Usage::default(),
        ))
    }
}

fn llm_strategy() -> (LlmCompaction, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let strategy = LlmCompaction::from_provider(
        Arc::new(CountingProvider {
            calls: Arc::clone(&calls),
        }),
        ModelConfig::mock(),
    );
    (strategy, calls)
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// The figures cited in `llm_compaction`'s module docs.
///
/// Re-run and update those docs after any change to compaction sizing —
/// `compact_target_ratio`, `compact_headroom_turns`, `DEFAULT_TRIGGER_RATIO`,
/// or `retain_tail_tokens`' derivation all move these numbers.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "benchmark: minutes to run; see the module docs for the command"]
async fn compaction_strategies_prefix_stability() {
    println!(
        "\n{:<10} {:<22} {:>7} {:>9} {:>8} {:>9} {:>9}",
        "budget", "strategy", "breaks", "interval", "peak", "final", "requests"
    );
    for (budget, tokens_per_turn, rounds) in
        [(20_000usize, 460usize, 120usize), (100_000, 460, 600)]
    {
        let shape = SessionShape::new(budget, tokens_per_turn, rounds);

        let d = measure(&DefaultCompaction, &shape).await;
        println!(
            "{budget:<10} {:<22} {:>7} {:>9} {:>8} {:>9} {:>9}",
            "DefaultCompaction",
            d.cache_breaks,
            d.mean_interval()
                .map(|i| format!("{i:.1}"))
                .unwrap_or_else(|| "-".into()),
            d.peak_tokens,
            d.final_tokens,
            0,
        );

        let (llm, calls) = llm_strategy();
        let l = measure(&llm, &shape).await;
        println!(
            "{budget:<10} {:<22} {:>7} {:>9} {:>8} {:>9} {:>9}",
            "LlmCompaction",
            l.cache_breaks,
            l.mean_interval()
                .map(|i| format!("{i:.1}"))
                .unwrap_or_else(|| "-".into()),
            l.peak_tokens,
            l.final_tokens,
            calls.load(Ordering::SeqCst),
        );
    }
    println!();
}

/// Sweep turn size against a fixed budget. This is the shape that exposed the
/// fallback livelock: a strategy can look fine at one turn size and pay for
/// summaries it never splices at another, so `wasted` is the column that
/// matters — requests issued with nothing spliced to show for them.
///
/// The budget must be large enough that a split can exist at all: with
/// `keep_recent: 10` and a 2k budget the history never reaches eleven messages,
/// so the strategy is correctly inert at every turn size and the sweep measures
/// nothing.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "benchmark: minutes to run; see the module docs for the command"]
async fn llm_compaction_across_turn_sizes() {
    println!(
        "\n{:<16} {:>7} {:>9} {:>9} {:>8}",
        "tokens/turn", "breaks", "splices", "requests", "wasted"
    );
    for tokens_per_turn in [100usize, 400, 1_000, 3_000, 6_000] {
        let (llm, calls) = llm_strategy();
        let shape = SessionShape::new(20_000, tokens_per_turn, 60);
        let r = measure(&llm, &shape).await;
        let requests = calls.load(Ordering::SeqCst);
        println!(
            "{tokens_per_turn:<16} {:>7} {:>9} {:>9} {:>8}",
            r.cache_breaks,
            r.rounds_with_summary,
            requests,
            requests > 0 && r.rounds_with_summary == 0,
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

/// Keeps the harness itself honest: a strategy that never touches history must
/// report zero breaks, and one that rewrites must report them. Without this,
/// a broken `common_prefix` would silently make every measurement read "0".
#[tokio::test(flavor = "multi_thread")]
async fn harness_detects_rewrites_and_only_rewrites() {
    struct Untouched;
    impl CompactionStrategy for Untouched {
        fn compact(&self, m: Vec<AgentMessage>, _c: &ContextConfig) -> Vec<AgentMessage> {
            m
        }
    }
    struct RewritesEveryTurn;
    impl CompactionStrategy for RewritesEveryTurn {
        fn compact(&self, mut m: Vec<AgentMessage>, _c: &ContextConfig) -> Vec<AgentMessage> {
            if !m.is_empty() {
                m[0] = AgentMessage::Llm(Message::user("rewritten"));
            }
            m
        }
    }

    let shape = SessionShape::new(1_000_000, 100, 6);
    assert_eq!(
        measure(&Untouched, &shape).await.cache_breaks,
        0,
        "append-only history must register no cache breaks"
    );
    let rewritten = measure(&RewritesEveryTurn, &shape).await;
    assert!(
        rewritten.cache_breaks >= 5,
        "a strategy rewriting index 0 every turn must register breaks, got {}",
        rewritten.cache_breaks
    );
}
