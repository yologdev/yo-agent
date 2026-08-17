//! LLM-based background compaction — a [`CompactionStrategy`] that summarizes
//! old history with a standalone LLM request instead of discarding it.
//!
//! The default [`DefaultCompaction`](crate::context::DefaultCompaction) is
//! deterministic: truncate → one-line summaries → drop. That is fast and
//! byte-stable, but lossy in the worst way — decisions and constraints from
//! early turns vanish. The alternative used by most coding agents (Pi, Claude
//! Code, Codex) is an LLM summarization request: a "shift-handoff briefing"
//! that keeps goals, progress, and key decisions in prose.
//!
//! The catch: [`CompactionStrategy::compact`] is synchronous and sits on the
//! hot path, directly before each LLM turn. Blocking the loop for a
//! summarization round-trip would add seconds of latency exactly when the
//! session is busiest. So this strategy runs the request **in the
//! background** and splices the result in later:
//!
//! ```text
//! turn N   (usage crosses trigger_ratio · budget)
//!   └── spawn: summarize messages[0..cut) with a standalone request
//! turn N+1..M: loop continues untouched — full prefix-cache reuse
//! turn M   (usage crosses the budget)
//!   └── splice: [head keep_first][summary message][tail cut..] — one
//!       deliberate cache break, at the latest possible moment
//! ```
//!
//! Because yoagent history is append-only and tool outputs are settled on
//! append (see [`ContextConfig::truncate_tool_output_on_append`]), the
//! snapshotted head cannot change while the task runs. A fingerprint verifies
//! this before splicing; on any mismatch the summary is discarded.
//!
//! **The loop can never wedge on this strategy.** If the budget is exceeded
//! before a summary is ready — provider down, model slow, rate-limited — it
//! falls back to the deterministic [`compact_messages`] for that turn and
//! keeps going.
//!
//! # Model routing
//!
//! The summarization request is standalone (its own system prompt, no session
//! history), so it can use a different, cheaper model than the main loop at
//! no cost to quality of the session itself. Pass any provider + model:
//!
//! ```no_run
//! use std::sync::Arc;
//! use yoagent::compaction_llm::LlmCompaction;
//! use yoagent::provider::AnthropicProvider;
//! use yoagent::Agent;
//!
//! let compaction = LlmCompaction::new(
//!     Arc::new(AnthropicProvider),
//!     "claude-haiku-4-5-20251001",           // cheap model for summaries
//!     std::env::var("ANTHROPIC_API_KEY").unwrap(),
//! );
//!
//! let agent = Agent::new(AnthropicProvider)
//!     .with_model("claude-sonnet-4-20250514") // main loop model
//!     .with_compaction_strategy(compaction);
//! ```

use crate::context::{
    self, compact_messages, message_timestamp, safe_head_end, safe_turn_start, total_tokens,
    CompactionStrategy, ContextConfig,
};
use crate::provider::{StreamConfig, StreamProvider};
use crate::types::CacheConfig;
use crate::types::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Default fraction of the budget at which background summarization starts.
///
/// Low enough that the summary is almost always ready before the budget is
/// hit (the gap is `(1 − ratio) · budget` tokens of session growth), high
/// enough that short sessions never pay for a summarization request at all.
pub const DEFAULT_TRIGGER_RATIO: f32 = 0.6;

/// Default token budget for the retained tail of recent messages.
///
/// Matches the ballpark Pi ships (20k ≈ 5–20 turns): enough recent detail
/// that the model does not lose its immediate working state.
pub const DEFAULT_RETAIN_TAIL_TOKENS: usize = 20_000;

/// Constant first line of the spliced summary message.
///
/// Constant on purpose, like [`context`]'s own compaction marker: the summary
/// body varies, but a stable prefix makes splices recognizable in transcripts
/// and replays.
pub const SUMMARY_MARKER: &str = "[Context compacted — summary of earlier conversation]";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a context summarization assistant. You produce \
     structured handoff briefings of agent conversations so that work can \
     continue seamlessly with the summary in place of the original messages.";

const DEFAULT_INSTRUCTION: &str = "Summarize the conversation above as a handoff briefing for \
     an agent that will continue this work without access to the original \
     messages. Use exactly these sections:\n\
     ## Goal\nWhat the user is trying to accomplish, verbatim where possible.\n\
     ## State & progress\nWhat has been done, what is currently in flight.\n\
     ## Key decisions & constraints\nDecisions made and why; constraints, \
     preferences, and facts that must not be re-litigated.\n\
     ## Open items\nUnresolved questions and concrete next steps.\n\
     Be dense and factual. Include exact identifiers (paths, names, versions, \
     numbers) — those are the details the next agent cannot reconstruct.";

/// Cap on the characters of any single message serialized into the
/// summarization transcript. Long tool outputs were already truncated on
/// append; this bounds the pathological rest.
const TRANSCRIPT_PER_MESSAGE_CHARS: usize = 2_000;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Identity of a snapshotted head, checked before splicing.
///
/// History is append-only, so `messages[0..cut)` should be unchanged when the
/// background task finishes — but "should" is not a correctness argument. The
/// fingerprint folds each head message's index, timestamp, and token estimate;
/// any rewrite (a custom `transform_context`, a manual `replace_messages`)
/// changes it and the stale summary is dropped instead of spliced.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    cut: usize,
    hash: u64,
}

fn fingerprint(messages: &[AgentMessage], cut: usize) -> Fingerprint {
    let mut hasher = DefaultHasher::new();
    for (i, msg) in messages[..cut].iter().enumerate() {
        i.hash(&mut hasher);
        message_timestamp(msg).hash(&mut hasher);
        context::message_tokens(msg).hash(&mut hasher);
    }
    Fingerprint {
        cut,
        hash: hasher.finish(),
    }
}

#[derive(Default)]
struct State {
    /// A summarization task is running; don't spawn another.
    inflight: bool,
    /// Completed summary waiting to be spliced.
    ready: Option<(Fingerprint, String)>,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Background LLM summarization behind the synchronous
/// [`CompactionStrategy`] trait. See the [module docs](self) for the design.
pub struct LlmCompaction {
    provider: Arc<dyn StreamProvider>,
    model: String,
    api_key: String,
    trigger_ratio: f32,
    retain_tail_tokens: usize,
    system_prompt: String,
    instruction: String,
    max_summary_tokens: u32,
    state: Arc<Mutex<State>>,
}

impl LlmCompaction {
    /// A compaction strategy that summarizes with `model` on `provider`.
    ///
    /// The request is standalone, so this can (and usually should) be a
    /// cheaper model than the main loop's.
    pub fn new(
        provider: Arc<dyn StreamProvider>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            model: model.into(),
            api_key: api_key.into(),
            trigger_ratio: DEFAULT_TRIGGER_RATIO,
            retain_tail_tokens: DEFAULT_RETAIN_TAIL_TOKENS,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            instruction: DEFAULT_INSTRUCTION.into(),
            max_summary_tokens: 2_000,
            state: Arc::new(Mutex::new(State::default())),
        }
    }

    /// Fraction of the budget at which background summarization starts.
    /// Clamped to `[0.1, 0.95]`. Default: [`DEFAULT_TRIGGER_RATIO`].
    pub fn with_trigger_ratio(mut self, ratio: f32) -> Self {
        self.trigger_ratio = if ratio.is_finite() {
            ratio.clamp(0.1, 0.95)
        } else {
            DEFAULT_TRIGGER_RATIO
        };
        self
    }

    /// Token budget for the retained tail of recent messages.
    /// Default: [`DEFAULT_RETAIN_TAIL_TOKENS`].
    pub fn with_retain_tail_tokens(mut self, tokens: usize) -> Self {
        self.retain_tail_tokens = tokens;
        self
    }

    /// Replace the summarization system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Replace the summarization instruction (the "handoff briefing" prompt).
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }

    /// Cap on summary length in output tokens. Default: 2000.
    pub fn with_max_summary_tokens(mut self, tokens: u32) -> Self {
        self.max_summary_tokens = tokens;
        self
    }

    /// Choose the head/tail split: the largest `cut` (on a safe turn
    /// boundary, past `keep_first`) that leaves at least
    /// `retain_tail_tokens` of tail — i.e. summarize as much as possible
    /// while keeping the recent working set intact.
    fn choose_cut(&self, messages: &[AgentMessage], config: &ContextConfig) -> Option<usize> {
        let len = messages.len();
        let head_min = safe_head_end(messages, config.keep_first.min(len));

        // Walk back from the end accumulating the tail budget.
        let mut tail_tokens = 0usize;
        let mut cut = len;
        while cut > head_min && tail_tokens < self.retain_tail_tokens {
            cut -= 1;
            tail_tokens += context::message_tokens(&messages[cut]);
        }
        // Also honour keep_recent as a floor on the tail, then land the
        // boundary on a turn start so no tool result is orphaned.
        cut = cut.min(len.saturating_sub(config.keep_recent));
        let cut = safe_turn_start(messages, cut);

        // A summary of two messages is not worth a request or a cache break.
        (cut > head_min && cut >= 4).then_some(cut)
    }

    /// Spawn the standalone summarization request for `messages[0..cut)`.
    fn spawn_summarize(&self, messages: &[AgentMessage], cut: usize) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // Called outside a runtime (direct compact_messages-style use):
            // background summarization is impossible; deterministic fallback
            // will handle the budget. Not an error.
            tracing::debug!("llm compaction: no tokio runtime, skipping background summarize");
            return;
        };

        let fp = fingerprint(messages, cut);
        let transcript = serialize_transcript(&messages[..cut]);
        let provider = Arc::clone(&self.provider);
        let state = Arc::clone(&self.state);
        let stream_config = StreamConfig {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            messages: vec![Message::user(format!(
                "<conversation>\n{transcript}\n</conversation>\n\n{}",
                self.instruction
            ))],
            tools: vec![],
            thinking_level: ThinkingLevel::default(),
            api_key: self.api_key.clone(),
            max_tokens: Some(self.max_summary_tokens),
            temperature: Some(0.0),
            model_config: None,
            // One-shot request; nothing to reuse.
            cache_config: CacheConfig {
                enabled: false,
                ..Default::default()
            },
            output_schema: None,
        };

        state.lock().unwrap().inflight = true;
        tracing::debug!(
            "llm compaction: summarizing {} messages in background (cut={})",
            cut,
            cut
        );

        handle.spawn(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            // Drain events we don't consume so the provider never blocks.
            let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
            let cancel = tokio_util::sync::CancellationToken::new();
            let result = provider.stream(stream_config, tx, cancel).await;
            drain.abort();

            let mut state = state.lock().unwrap();
            state.inflight = false;
            match result {
                Ok(message) => {
                    let text = assistant_text(&message);
                    if text.trim().is_empty() {
                        tracing::warn!("llm compaction: empty summary, discarding");
                    } else {
                        tracing::debug!(
                            "llm compaction: summary ready ({} chars for {} messages)",
                            text.len(),
                            fp.cut
                        );
                        state.ready = Some((fp, text));
                    }
                }
                Err(e) => {
                    // Next trigger crossing will retry; until then the
                    // deterministic fallback covers the budget.
                    tracing::warn!("llm compaction: summarization failed: {e}");
                }
            }
        });
    }

    /// Splice a ready summary: `[head keep_first][summary][tail cut..]`.
    fn splice(
        &self,
        messages: Vec<AgentMessage>,
        config: &ContextConfig,
        fp: &Fingerprint,
        summary: &str,
    ) -> Vec<AgentMessage> {
        let head_end = safe_head_end(&messages, config.keep_first.min(messages.len()));
        let ts = message_timestamp(&messages[fp.cut.saturating_sub(1)]);
        let summary_msg = AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: format!("{SUMMARY_MARKER}\n\n{summary}"),
            }],
            timestamp: ts,
        });

        let mut result = Vec::with_capacity(messages.len() - fp.cut + head_end + 1);
        result.extend_from_slice(&messages[..head_end]);
        result.push(summary_msg);
        result.extend_from_slice(&messages[fp.cut..]);
        tracing::info!(
            "llm compaction: spliced summary, {} -> {} messages ({} tokens)",
            messages.len(),
            result.len(),
            total_tokens(&result)
        );
        result
    }
}

impl CompactionStrategy for LlmCompaction {
    fn compact(&self, messages: Vec<AgentMessage>, config: &ContextConfig) -> Vec<AgentMessage> {
        let budget = config
            .max_context_tokens
            .saturating_sub(config.system_prompt_tokens);
        let used = total_tokens(&messages);
        let trigger = (budget as f32 * self.trigger_ratio) as usize;

        // 1. Over budget and a summary is ready → splice (verify identity).
        if used > budget {
            let ready = self.state.lock().unwrap().ready.take();
            if let Some((fp, summary)) = ready {
                if fp.cut <= messages.len() && fingerprint(&messages, fp.cut) == fp {
                    let result = self.splice(messages, config, &fp, &summary);
                    // Safety net: if even the spliced result is over budget
                    // (huge tail), let the deterministic tiers finish the job.
                    if total_tokens(&result) > budget {
                        return compact_messages(result, config);
                    }
                    return result;
                }
                tracing::warn!("llm compaction: history changed under summary, discarding");
            }
        }

        // 2. Crossed the trigger and idle → start a background summarization.
        {
            let state = self.state.lock().unwrap();
            let idle = !state.inflight && state.ready.is_none();
            drop(state);
            if used > trigger && idle {
                if let Some(cut) = self.choose_cut(&messages, config) {
                    self.spawn_summarize(&messages, cut);
                }
            }
        }

        // 3. Over budget with no summary ready → deterministic fallback.
        //    The loop always makes progress; a slow or dead summarizer can
        //    never wedge it.
        if used > budget {
            tracing::debug!("llm compaction: summary not ready, deterministic fallback");
            return compact_messages(messages, config);
        }

        messages
    }
}

// ---------------------------------------------------------------------------
// Transcript serialization
// ---------------------------------------------------------------------------

fn clip(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    // Back off to a char boundary.
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Serialize head messages into a plain-text transcript for the
/// summarization request. Bounded per message; tool arguments and results
/// are included in clipped form — identifiers matter, repetition does not.
fn serialize_transcript(messages: &[AgentMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        let AgentMessage::Llm(message) = msg else {
            continue; // Extension messages never reach the LLM anyway.
        };
        match message {
            Message::User { content, .. } => {
                for c in content {
                    if let Content::Text { text } = c {
                        out.push_str("User: ");
                        out.push_str(clip(text, TRANSCRIPT_PER_MESSAGE_CHARS));
                        out.push('\n');
                    }
                }
            }
            Message::Assistant { content, .. } => {
                for c in content {
                    match c {
                        Content::Text { text } => {
                            out.push_str("Assistant: ");
                            out.push_str(clip(text, TRANSCRIPT_PER_MESSAGE_CHARS));
                            out.push('\n');
                        }
                        Content::ToolCall {
                            name, arguments, ..
                        } => {
                            let args = arguments.to_string();
                            out.push_str(&format!("[tool call] {name}({})\n", clip(&args, 300)));
                        }
                        _ => {}
                    }
                }
            }
            Message::ToolResult {
                tool_name, content, ..
            } => {
                for c in content {
                    if let Content::Text { text } = c {
                        out.push_str(&format!(
                            "[tool result: {tool_name}] {}\n",
                            clip(text, TRANSCRIPT_PER_MESSAGE_CHARS)
                        ));
                    }
                }
            }
        }
    }
    out
}

fn assistant_text(message: &Message) -> String {
    let Message::Assistant { content, .. } = message else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;

    fn turn(i: usize, bulk: usize) -> Vec<AgentMessage> {
        vec![
            AgentMessage::Llm(Message::User {
                content: vec![Content::Text {
                    text: format!("user message {i}: {}", "x".repeat(bulk)),
                }],
                timestamp: i as u64,
            }),
            AgentMessage::Llm(
                Message::assistant(
                    vec![Content::Text {
                        text: format!("assistant reply {i}: {}", "y".repeat(bulk)),
                    }],
                    StopReason::Stop,
                    "mock",
                    "mock",
                    Usage::default(),
                )
                .with_timestamp(i as u64),
            ),
        ]
    }

    fn history(turns: usize, bulk: usize) -> Vec<AgentMessage> {
        (0..turns).flat_map(|i| turn(i, bulk)).collect()
    }

    fn config(budget: usize) -> ContextConfig {
        ContextConfig {
            max_context_tokens: budget,
            system_prompt_tokens: 0,
            keep_first: 1,
            keep_recent: 2,
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn splices_summary_when_over_budget() {
        let strategy = LlmCompaction::new(
            Arc::new(MockProvider::text("## Goal\nShip the parser.")),
            "mock-model",
            "test-key",
        )
        .with_trigger_ratio(0.1)
        .with_retain_tail_tokens(200);

        let messages = history(30, 400); // well over a 2k-token budget
        let cfg = config(2_000);

        // First pass: crosses trigger, spawns background summarization,
        // falls back deterministically for this turn (over budget already).
        let out = strategy.compact(messages.clone(), &cfg);
        assert!(total_tokens(&out) <= 2_000, "fallback must fit budget");

        // Let the mock summarization finish.
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if strategy.state.lock().unwrap().ready.is_some() {
                break;
            }
        }
        assert!(
            strategy.state.lock().unwrap().ready.is_some(),
            "summary should be ready"
        );

        // Second pass over the SAME append-only history: splice.
        let out = strategy.compact(messages.clone(), &cfg);
        let spliced = out.iter().any(|m| {
            matches!(m, AgentMessage::Llm(Message::User { content, .. })
                if content.iter().any(|c| matches!(c, Content::Text { text }
                    if text.starts_with(SUMMARY_MARKER) && text.contains("Ship the parser"))))
        });
        assert!(spliced, "summary message must be spliced in");
        assert!(total_tokens(&out) < total_tokens(&messages));
        // Tail preserved verbatim: the last original message survives.
        assert_eq!(out.last(), messages.last());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn under_trigger_is_a_no_op() {
        let strategy = LlmCompaction::new(
            Arc::new(MockProvider::text("unused")),
            "mock-model",
            "test-key",
        );
        let messages = history(3, 50);
        let out = strategy.compact(messages.clone(), &config(1_000_000));
        assert_eq!(out, messages, "below trigger nothing may change");
        assert!(!strategy.state.lock().unwrap().inflight);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stale_summary_is_discarded_not_spliced() {
        let strategy = LlmCompaction::new(
            Arc::new(MockProvider::text("## Goal\nStale.")),
            "mock-model",
            "test-key",
        )
        .with_trigger_ratio(0.1)
        .with_retain_tail_tokens(200);

        let messages = history(30, 400);
        let cfg = config(2_000);
        strategy.compact(messages.clone(), &cfg);
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if strategy.state.lock().unwrap().ready.is_some() {
                break;
            }
        }

        // History rewritten (not append-only): fingerprint must not match.
        let mut mutated = messages.clone();
        mutated[0] = AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: "rewritten".into(),
            }],
            timestamp: 999,
        });
        let out = strategy.compact(mutated, &cfg);
        let spliced = out.iter().any(|m| {
            matches!(m, AgentMessage::Llm(Message::User { content, .. })
                if content.iter().any(|c| matches!(c, Content::Text { text }
                    if text.starts_with(SUMMARY_MARKER))))
        });
        assert!(!spliced, "stale summary must be discarded");
        assert!(total_tokens(&out) <= 2_000, "fallback still fits budget");
    }

    /// A provider that always fails — MockProvider never errors, so the
    /// failure path needs its own double.
    struct FailingProvider;

    #[async_trait::async_trait]
    impl StreamProvider for FailingProvider {
        async fn stream(
            &self,
            _config: StreamConfig,
            _tx: tokio::sync::mpsc::UnboundedSender<crate::provider::StreamEvent>,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Message, crate::provider::ProviderError> {
            Err(crate::provider::ProviderError::Network("down".into()))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn provider_failure_falls_back_deterministically() {
        let strategy = LlmCompaction::new(
            Arc::new(FailingProvider),
            "mock-model",
            "test-key",
        )
        .with_trigger_ratio(0.1)
        .with_retain_tail_tokens(200);

        let messages = history(30, 400);
        let cfg = config(2_000);
        let out = strategy.compact(messages.clone(), &cfg);
        assert!(total_tokens(&out) <= 2_000);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let state = strategy.state.lock().unwrap();
        assert!(state.ready.is_none(), "failed request leaves no summary");
        assert!(!state.inflight, "inflight flag must clear on failure");
    }
}
