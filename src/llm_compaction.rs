//! LLM-based background compaction — a [`CompactionStrategy`] that summarizes
//! old history with a standalone LLM request instead of discarding it.
//!
//! The default [`DefaultCompaction`](crate::context::DefaultCompaction) is
//! deterministic: truncate → one-line summaries → drop. That is fast,
//! byte-stable and free, but lossy in the worst way — decisions and constraints
//! from early turns simply vanish. This strategy spends tokens to get them back
//! as prose: a "shift-handoff briefing" covering goals, progress, and decisions,
//! spliced in where the dropped span used to be.
//!
//! # What it buys, and what it costs
//!
//! **Buys: retention quality, at no added loop latency.**
//! [`CompactionStrategy::compact`] is synchronous and sits on the hot path
//! directly before each LLM turn, so a summarization round-trip cannot happen
//! there — it would add seconds of latency exactly when the session is busiest.
//! Instead the request runs in the background and the result is spliced in on a
//! later turn:
//!
//! ```text
//! turn N   (usage crosses trigger_ratio · budget)
//!   └── spawn: summarize messages[head_end..cut) with a standalone request
//! turn N+1..M: loop continues untouched, no stall, no request
//! turn M   (usage crosses the budget)
//!   └── splice: [head][summary][tail cut..]
//! ```
//!
//! **Costs: tokens the default never spends.** Each summarization request pays
//! input tokens for the whole summarized span plus output tokens for the
//! briefing, on top of the session's own traffic, and a long session summarizes
//! repeatedly. Route it at a cheap model (see below) and watch the numbers on
//! [`AgentEvent::ContextCompacted`], which reports the request's own usage next
//! to the tokens it saved.
//!
//! **Does not buy: fewer prefix-cache breaks.** Both strategies rewrite history
//! when the budget is crossed, and neither rewrites in between, so the number of
//! cache breaks over a session is a wash — measured at 6 vs 6 over 120 turns at
//! a 20k budget, and 6 vs 5 over 600 turns at 100k. Splicing at the last
//! possible moment describes when this strategy breaks the cache relative to a
//! *synchronous* summarizer, which must break it the moment it decides to
//! summarize. It is not an advantage over
//! [`DefaultCompaction`](crate::context::DefaultCompaction).
//!
//! # Guarantees
//!
//! **The loop can never wedge on this strategy.** Summary not ready, provider
//! down, rate-limited, no tokio runtime — compaction falls back to the
//! deterministic [`compact_messages`] for that turn and the loop keeps going.
//!
//! **A stale summary is never spliced.** The snapshotted prefix is fingerprinted
//! over its serialized bytes; on any mismatch the summary is discarded and a
//! fresh one is started.
//!
//! **A summary is never paid for and then thrown away.** The background request
//! is always fingerprinted against the history `compact` is about to *return*,
//! never the one it received. That distinction is load-bearing: the agent loop
//! writes the return value straight back into the session, so anchoring on the
//! input meant that whenever the deterministic fallback rewrote history the
//! pending summary was invalidated by the same call that ordered it — discarded
//! on arrival, replaced by another request against a prefix the next fallback
//! would rewrite in turn. That state was absorbing, and it cost one billed
//! request per compaction for a result identical to
//! [`DefaultCompaction`](crate::context::DefaultCompaction)'s.
//!
//! **The briefing survives the safety net.** When a splice still exceeds the
//! budget the tail is compacted in place rather than the whole history: the
//! summary sits exactly where `level3_drop_middle` starts cutting, so the naive
//! move destroys what was just paid for. If head plus briefing exceed the budget
//! on their own the briefing genuinely cannot be kept — the event then reports
//! [`CompactionMethod::Deterministic`], because claiming a splice the result
//! does not contain is worse than reporting the fallback.
//!
//! # One accepted limit
//!
//! **The API key is resolved once, at construction.**
//! [`from_config`](LlmCompaction::from_config) reads the provider-conventional
//! environment variable when the strategy is built and holds the result for its
//! lifetime. Note this is *not* [`Agent`](crate::Agent)'s contract: `Agent`
//! resolves lazily per request, so it does pick up a rotated environment key.
//! Build a fresh strategy if credentials rotate mid-session.
//!
//! # Known limitation: the headroom policy does not apply
//!
//! [`ContextConfig::compact_target_ratio`] and
//! [`ContextConfig::compact_headroom_turns`] size the deterministic tiers, and
//! the agent loop adapts the ratio to the session's observed growth. The
//! *spliced* result ignores both — its size is set by
//! [`with_retain_tail_tokens`](LlmCompaction::with_retain_tail_tokens) instead.
//! The deterministic fallback and the safety net do still honour them, since
//! both run through [`compact_messages`]. Consuming the adapted target on the
//! splice path is follow-up work; until then, tune the tail directly if the
//! interval between compactions matters to you.
//!
//! # Model routing
//!
//! The summarization request is standalone — its own system prompt, no session
//! history, no tools — so it can use a different, cheaper model than the main
//! loop at no cost to the session itself:
//!
//! ```no_run
//! use yoagent::provider::ModelConfig;
//! use yoagent::{Agent, LlmCompaction};
//!
//! // Provider and key resolved from the config, same as `Agent::from_config`.
//! let compaction = LlmCompaction::from_config(ModelConfig::anthropic(
//!     "claude-haiku-4-5",
//!     "Haiku 4.5",
//! ));
//!
//! let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
//!     .with_compaction_strategy(compaction);
//! ```

use crate::context::{
    self, compact_messages, message_timestamp, safe_head_end, safe_turn_start, total_tokens,
    CompactionStrategy, ContextConfig,
};
use crate::provider::{ModelConfig, StreamConfig, StreamProvider};
use crate::types::CacheConfig;
use crate::types::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

/// Default fraction of the budget at which background summarization starts.
///
/// Low enough that the summary is almost always ready before the budget is
/// hit (the gap is `(1 − ratio) · budget` tokens of session growth), high
/// enough that short sessions never pay for a summarization request at all.
pub const DEFAULT_TRIGGER_RATIO: f32 = 0.6;

/// Ceiling on the derived default for the retained tail of recent messages.
///
/// The default is `min(DEFAULT_RETAIN_TAIL_TOKENS, budget / 4)`, not this
/// constant flat — see
/// [`with_retain_tail_tokens`](LlmCompaction::with_retain_tail_tokens) for why
/// a fixed number silently disables the strategy on smaller budgets. 20k is the
/// ballpark Pi ships (≈ 5–20 turns): enough recent detail that the model does
/// not lose its immediate working state.
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

/// A span shorter than this is not worth a request or a cache break.
const MIN_SUMMARIZED_SPAN: usize = 4;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Identity of a snapshotted prefix, checked before splicing.
///
/// History is append-only during a run, so `messages[0..cut)` should be
/// unchanged when the background task finishes — but "should" is not a
/// correctness argument, and two things really can invalidate it:
/// [`Agent::replace_messages`](crate::Agent::replace_messages), and this
/// strategy's own deterministic fallback rewriting history on a turn where the
/// budget was crossed before the summary landed. (`transform_context` cannot:
/// the loop runs it on a clone and never writes the result back.)
///
/// The hash folds each prefix message's index and its serialized bytes, so any
/// edit at all changes it and the stale summary is dropped instead of spliced.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    cut: usize,
    hash: u64,
}

/// Where the verbatim head may end without orphaning a tool call.
///
/// [`safe_head_end`] walks back past an assistant that *opens* tool calls, but
/// not past the results themselves. With `ToolExecutionStrategy::Parallel` — the
/// default — one assistant message yields several `ToolResult` messages, so a
/// boundary landing strictly inside that run keeps the assistant and only
/// *some* of its results; the rest fall into the summarized span and the
/// provider rejects the request outright.
///
/// [`DefaultCompaction`](crate::context::DefaultCompaction) never hit this
/// because its Level 2 rewrites the whole pre-boundary region before Level 3
/// cuts. Splicing is the first path that re-emits `messages[..head_end]`
/// verbatim, so it is the first to need both pullbacks — applied to a fixed
/// point, since one can expose the other.
fn safe_head_boundary(messages: &[AgentMessage], end: usize) -> usize {
    let mut end = end;
    for _ in 0..=messages.len() {
        let pulled = safe_head_end(messages, safe_turn_start(messages, end));
        if pulled == end {
            break;
        }
        end = pulled;
    }
    end
}

fn fingerprint(messages: &[AgentMessage], cut: usize) -> Fingerprint {
    let mut hasher = DefaultHasher::new();
    for (i, msg) in messages[..cut].iter().enumerate() {
        i.hash(&mut hasher);
        match serde_json::to_vec(msg) {
            Ok(bytes) => bytes.hash(&mut hasher),
            // Not reachable for the `AgentMessage` shapes serde_json can emit,
            // but hashing a sentinel keeps the fold total rather than silently
            // skipping a message and weakening the check.
            Err(_) => u8::MAX.hash(&mut hasher),
        }
    }
    Fingerprint {
        cut,
        hash: hasher.finish(),
    }
}

/// A finished summary waiting to be spliced.
struct Summary {
    fingerprint: Fingerprint,
    /// Where the verbatim head ends. The summary covers `[head_end, cut)`;
    /// `messages[..head_end]` survives the splice untouched. Captured at spawn
    /// time so the splice cannot disagree with what was actually summarized.
    head_end: usize,
    text: String,
    /// What the summarization request itself cost.
    usage: Usage,
}

#[derive(Default)]
struct State {
    /// A summarization task is running; don't spawn another.
    inflight: bool,
    /// Completed summary waiting to be spliced.
    ready: Option<Summary>,
    /// The "no split is possible" warning has been emitted once already.
    warned_inert: bool,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Background LLM summarization behind the synchronous
/// [`CompactionStrategy`] trait. See the [module docs](self) for the design,
/// the cost trade-off, and the known limitation.
///
/// The state machine is per-session. Do not share one instance across
/// concurrently running loops — `Agent` never does, but a hand-built
/// [`AgentLoopConfig`](crate::agent_loop::AgentLoopConfig) could.
pub struct LlmCompaction {
    provider: Arc<dyn StreamProvider>,
    model: String,
    api_key: String,
    /// `None` when built via [`from_provider`](LlmCompaction::from_provider);
    /// the request then carries no per-provider base URL, headers, or pricing.
    model_config: Option<ModelConfig>,
    trigger_ratio: f32,
    /// `None` derives it from the budget at call time — see
    /// [`with_retain_tail_tokens`](LlmCompaction::with_retain_tail_tokens).
    retain_tail_tokens: Option<usize>,
    system_prompt: String,
    instruction: String,
    max_summary_tokens: u32,
    events: Option<UnboundedSender<AgentEvent>>,
    state: Arc<Mutex<State>>,
}

impl LlmCompaction {
    /// A compaction strategy that summarizes with the model in `config`,
    /// selecting the built-in provider for the config's protocol and resolving
    /// the API key from the provider-conventional environment variable.
    ///
    /// This mirrors [`Agent::from_config`](crate::Agent::from_config), and for
    /// the same reason: a bare `(provider, model, key)` triple lets the three
    /// drift apart, and drops the `base_url`, headers, and compat flags that
    /// every non-Anthropic provider needs.
    ///
    /// The request is standalone, so this can (and usually should) name a
    /// cheaper model than the main loop's.
    pub fn from_config(config: ModelConfig) -> Self {
        let provider = crate::provider::ProviderRegistry::default()
            .resolve(&config.api)
            .expect("default registry covers all built-in protocols");
        let api_key = crate::provider::resolve_api_key(&config.provider).unwrap_or_default();
        Self::build(provider, config.id.clone(), api_key, Some(config))
    }

    /// A compaction strategy that summarizes with an explicit provider.
    ///
    /// The escape hatch for custom [`StreamProvider`] implementations and test
    /// doubles. Without a [`ModelConfig`] the request carries no `base_url`,
    /// headers, or compat flags, and [`AgentEvent::ContextCompacted`] reports
    /// no cost — prefer [`from_config`](Self::from_config) for anything talking
    /// to a real API.
    pub fn from_provider(
        provider: Arc<dyn StreamProvider>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::build(provider, model.into(), api_key.into(), None)
    }

    fn build(
        provider: Arc<dyn StreamProvider>,
        model: String,
        api_key: String,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self {
            provider,
            model,
            api_key,
            model_config,
            trigger_ratio: DEFAULT_TRIGGER_RATIO,
            retain_tail_tokens: None,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            instruction: DEFAULT_INSTRUCTION.into(),
            max_summary_tokens: 2_000,
            events: None,
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
    ///
    /// Setting this too high relative to the context budget disables the
    /// strategy outright: summarization is first attempted at
    /// `trigger_ratio · budget`, and if the tail alone would consume all the
    /// history that exists at that point, there is nothing left to summarize.
    /// The default therefore derives from the budget —
    /// `min(`[`DEFAULT_RETAIN_TAIL_TOKENS`]`, budget / 4)`, recomputed per call
    /// so it tracks the loop's calibrated budget — rather than being a fixed
    /// number that silently no-ops on smaller context windows.
    ///
    /// An explicit value here is used as given. If it turns out to be too
    /// large, a one-time `warn!` says so instead of failing quietly.
    pub fn with_retain_tail_tokens(mut self, tokens: usize) -> Self {
        self.retain_tail_tokens = Some(tokens);
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

    /// Emit [`AgentEvent::ContextCompacted`] on this channel when compaction
    /// runs, by either path.
    ///
    /// [`CompactionStrategy::compact`] has no access to the loop's event
    /// channel, so the sender has to come in from the side. Pair it with
    /// [`Agent::prompt_with_sender`](crate::Agent::prompt_with_sender), where
    /// the caller owns the channel:
    ///
    /// ```no_run
    /// # use yoagent::provider::ModelConfig;
    /// # use yoagent::{Agent, LlmCompaction};
    /// let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    /// let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
    ///     .with_compaction_strategy(
    ///         LlmCompaction::from_config(ModelConfig::anthropic(
    ///             "claude-haiku-4-5",
    ///             "Haiku 4.5",
    ///         ))
    ///         .with_event_sender(tx.clone()),
    ///     );
    /// # let _ = (agent, rx);
    /// ```
    pub fn with_event_sender(mut self, events: UnboundedSender<AgentEvent>) -> Self {
        self.events = Some(events);
        self
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.events {
            // A dropped receiver is not this strategy's problem to report.
            let _ = tx.send(event);
        }
    }

    /// Cost of the summarization request, when the model's rates are known.
    fn summary_cost(&self, usage: &Usage) -> Option<f64> {
        let cost = &self.model_config.as_ref()?.cost;
        cost.is_configured().then(|| cost.cost_usd(usage))
    }

    /// The tail budget for this call: explicit if set, else derived from the
    /// context budget so it cannot swallow the whole history.
    fn retain_tail_tokens(&self, budget: usize) -> usize {
        self.retain_tail_tokens
            .unwrap_or_else(|| DEFAULT_RETAIN_TAIL_TOKENS.min(budget / 4))
    }

    /// Choose the head/tail split: the largest `cut` (on a safe turn boundary,
    /// past the verbatim head) that leaves at least `retain_tail_tokens` of
    /// tail — i.e. summarize as much as possible while keeping the recent
    /// working set intact. Returns `(head_end, cut)`.
    fn choose_cut(
        &self,
        messages: &[AgentMessage],
        config: &ContextConfig,
        budget: usize,
    ) -> Option<(usize, usize)> {
        let len = messages.len();
        let head_end = safe_head_boundary(messages, config.keep_first.min(len));
        let retain = self.retain_tail_tokens(budget);

        // Walk back from the end accumulating the tail budget.
        let mut tail_tokens = 0usize;
        let mut cut = len;
        while cut > head_end && tail_tokens < retain {
            cut -= 1;
            tail_tokens += context::message_tokens(&messages[cut]);
        }
        // Also honour keep_recent as a floor on the tail, then land the
        // boundary on a turn start so no tool result is orphaned.
        cut = cut.min(len.saturating_sub(config.keep_recent));
        let cut = safe_turn_start(messages, cut);

        (cut > head_end && cut - head_end >= MIN_SUMMARIZED_SPAN).then_some((head_end, cut))
    }

    /// Say once — and only once — that no summary can ever be produced under
    /// the current settings. The failure this replaces was completely silent.
    fn warn_inert_once(&self, used: usize, budget: usize) {
        {
            let mut state = self.state.lock().unwrap();
            if state.warned_inert {
                return;
            }
            state.warned_inert = true;
        }
        tracing::warn!(
            "llm compaction is inert: past the trigger ({} of {} budget tokens) there is still \
             no split leaving {} tail tokens and {} messages to summarize, so no summary can \
             be produced and every compaction will fall back to the deterministic tiers. \
             Lower retain_tail_tokens (effective: {}) or raise trigger_ratio (currently {}).",
            used,
            budget,
            self.retain_tail_tokens(budget),
            MIN_SUMMARIZED_SPAN,
            self.retain_tail_tokens(budget),
            self.trigger_ratio,
        );
    }

    /// Spawn the standalone summarization request for `messages[head_end..cut)`.
    fn spawn_summarize(&self, messages: &[AgentMessage], head_end: usize, cut: usize) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // Called outside a runtime (direct compact_messages-style use):
            // background summarization is impossible; deterministic fallback
            // will handle the budget. Not an error.
            tracing::debug!("llm compaction: no tokio runtime, skipping background summarize");
            return;
        };

        // Fingerprint the whole prefix, not just the summarized span: the
        // splice re-emits messages[..head_end] verbatim too, so all of
        // [0, cut) has to be unchanged for the result to be coherent.
        let fp = fingerprint(messages, cut);
        // Summarize only [head_end, cut). The head survives the splice
        // verbatim, so including it would put those messages in the context
        // twice — once as themselves, once inside the briefing.
        let transcript = serialize_transcript(&messages[head_end..cut]);
        let provider = Arc::clone(&self.provider);
        let state = Arc::clone(&self.state);

        // `StreamConfig` is #[non_exhaustive]: construct via `new` and mutate,
        // per its own documented convention.
        let mut stream_config = StreamConfig::new(self.model.clone(), self.api_key.clone());
        stream_config.system_prompt = self.system_prompt.clone();
        stream_config.messages = vec![Message::user(format!(
            "<conversation>\n{transcript}\n</conversation>\n\n{}",
            self.instruction
        ))];
        stream_config.max_tokens = Some(self.max_summary_tokens);
        stream_config.model_config = self.model_config.clone();
        // `temperature` is left at the `StreamConfig::new` default of `None`:
        // there is no temperature quirk flag in the compat matrix, so an
        // explicit value goes through verbatim, and the newest reasoning models
        // reject sampling parameters outright.
        debug_assert!(stream_config.temperature.is_none());
        // One-shot request; nothing to reuse.
        stream_config.cache_config = CacheConfig {
            enabled: false,
            ..Default::default()
        };

        state.lock().unwrap().inflight = true;
        tracing::debug!(
            "llm compaction: summarizing messages[{}..{}) in background",
            head_end,
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
                            fp.cut - head_end
                        );
                        state.ready = Some(Summary {
                            fingerprint: fp,
                            head_end,
                            usage: assistant_usage(&message),
                            text,
                        });
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

    /// Splice a ready summary: `[head][summary][tail cut..]`.
    fn splice(&self, messages: &[AgentMessage], summary: &Summary) -> Vec<AgentMessage> {
        let cut = summary.fingerprint.cut;
        let head_end = summary.head_end;
        let ts = message_timestamp(&messages[cut.saturating_sub(1)]);
        let summary_msg = AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: format!("{SUMMARY_MARKER}\n\n{}", summary.text),
            }],
            timestamp: ts,
        });

        let mut result = Vec::with_capacity(messages.len() - cut + head_end + 1);
        result.extend_from_slice(&messages[..head_end]);
        result.push(summary_msg);
        result.extend_from_slice(&messages[cut..]);
        result
    }

    /// Bring an over-budget splice back under budget **without** discarding the
    /// briefing that was just paid for.
    ///
    /// Handing the whole spliced history to [`compact_messages`] destroys it:
    /// the summary sits at `head_end`, which is exactly where
    /// `level3_drop_middle` begins cutting, so it is the *first* message
    /// dropped. Compacting only the span after it leaves head and summary
    /// intact and shrinks the tail, which is what actually overflowed.
    fn shrink_tail(
        &self,
        mut result: Vec<AgentMessage>,
        config: &ContextConfig,
        budget: usize,
        head_end: usize,
    ) -> Vec<AgentMessage> {
        let keep = head_end + 1; // the verbatim head plus the summary message
        if keep >= result.len() {
            return result;
        }
        let tail = result.split_off(keep);
        let fixed = total_tokens(&result);
        let tail_config = ContextConfig {
            max_context_tokens: budget.saturating_sub(fixed),
            system_prompt_tokens: 0,
            ..config.clone()
        };
        result.extend(compact_messages(tail, &tail_config));
        result
    }

    /// Start a background summarization if `messages` has crossed the trigger
    /// and nothing is already in flight.
    ///
    /// **Callers must pass the history `compact` is about to return, not the
    /// one it received.** The agent loop writes the return value straight back
    /// into the session, so a summary fingerprinted over the *input* is
    /// invalidated by the very same call whenever the deterministic fallback
    /// rewrites history — the summary is discarded on arrival, a replacement is
    /// spawned against a prefix the next fallback will rewrite in turn, and the
    /// strategy pays for a request per compaction while never splicing again.
    fn arm(&self, messages: &[AgentMessage], config: &ContextConfig, budget: usize) {
        let used = total_tokens(messages);
        if used <= (budget as f32 * self.trigger_ratio) as usize {
            return;
        }
        let idle = {
            let state = self.state.lock().unwrap();
            !state.inflight && state.ready.is_none()
        };
        if !idle {
            return;
        }
        match self.choose_cut(messages, config, budget) {
            Some((head_end, cut)) => self.spawn_summarize(messages, head_end, cut),
            // No split is possible. Silence here was the original bug: the
            // strategy looked configured and did nothing, forever.
            None => self.warn_inert_once(used, budget),
        }
    }
}

impl CompactionStrategy for LlmCompaction {
    fn compact(&self, messages: Vec<AgentMessage>, config: &ContextConfig) -> Vec<AgentMessage> {
        let budget = config
            .max_context_tokens
            .saturating_sub(config.system_prompt_tokens);
        let used = total_tokens(&messages);
        let messages_before = messages.len();

        // 1. Over budget and a summary is ready → splice (verify identity).
        if used > budget {
            let ready = self.state.lock().unwrap().ready.take();
            if let Some(summary) = ready {
                let fp = &summary.fingerprint;
                if fp.cut <= messages.len() && fingerprint(&messages, fp.cut) == *fp {
                    let mut summarized = fp.cut - summary.head_end;
                    let mut method = CompactionMethod::Summarized;
                    let mut result = self.splice(&messages, &summary);
                    if total_tokens(&result) > budget {
                        // The tail alone overflows. Shrink it in place so the
                        // briefing survives — see `shrink_tail`.
                        result = self.shrink_tail(result, config, budget, summary.head_end);
                        if total_tokens(&result) > budget {
                            // Head plus briefing alone exceed the budget, so the
                            // briefing cannot be kept at all. Report what really
                            // happened rather than claiming a splice the result
                            // does not contain.
                            tracing::warn!(
                                "llm compaction: head + summary exceed the budget on their own; \
                                 discarding the summary and compacting deterministically"
                            );
                            result = compact_messages(result, config);
                            method = CompactionMethod::Deterministic;
                            summarized = 0;
                        }
                    }
                    let after = total_tokens(&result);
                    tracing::info!(
                        "llm compaction: spliced a summary of {} messages, {} -> {} messages \
                         ({} -> {} tokens)",
                        summarized,
                        messages_before,
                        result.len(),
                        used,
                        after
                    );
                    self.emit(AgentEvent::ContextCompacted {
                        method,
                        messages_before,
                        messages_after: result.len(),
                        tokens_before: used,
                        tokens_after: after,
                        messages_summarized: summarized,
                        // The request was paid for either way, so its cost is
                        // reported even when the briefing could not be kept.
                        summary_cost_usd: self.summary_cost(&summary.usage),
                        summary_usage: Some(summary.usage),
                    });
                    self.arm(&result, config, budget);
                    return result;
                }
                tracing::warn!("llm compaction: history changed under summary, discarding");
            }
        }

        // 2. Over budget with no usable summary → deterministic fallback.
        //    The loop always makes progress; a slow or dead summarizer can
        //    never wedge it.
        if used > budget {
            tracing::debug!("llm compaction: summary not ready, deterministic fallback");
            let result = compact_messages(messages, config);
            let after = total_tokens(&result);
            self.emit(AgentEvent::ContextCompacted {
                method: CompactionMethod::Deterministic,
                messages_before,
                messages_after: result.len(),
                tokens_before: used,
                tokens_after: after,
                messages_summarized: 0,
                summary_usage: None,
                summary_cost_usd: None,
            });
            self.arm(&result, config, budget);
            return result;
        }

        // 3. Under budget → arm against the history we are handing back.
        self.arm(&messages, config, budget);
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

/// Serialize the summarized span into a plain-text transcript for the
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

fn assistant_usage(message: &Message) -> Usage {
    match message {
        Message::Assistant { usage, .. } => usage.clone(),
        _ => Usage::default(),
    }
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

    fn mock(text: &str) -> LlmCompaction {
        LlmCompaction::from_provider(Arc::new(MockProvider::text(text)), "mock-model", "test-key")
    }

    /// Poll until the background summarization lands, or give up.
    async fn await_summary(strategy: &LlmCompaction) -> bool {
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if strategy.state.lock().unwrap().ready.is_some() {
                return true;
            }
        }
        false
    }

    fn has_summary(messages: &[AgentMessage]) -> bool {
        messages.iter().any(|m| {
            matches!(m, AgentMessage::Llm(Message::User { content, .. })
                if content.iter().any(|c| matches!(c, Content::Text { text }
                    if text.starts_with(SUMMARY_MARKER))))
        })
    }

    async fn settle() {
        for _ in 0..25 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }

    /// Outcome of driving the strategy the way the agent loop does.
    struct Run {
        messages: Vec<AgentMessage>,
        rounds_with_summary: usize,
        peak_tokens: usize,
    }

    /// Append a turn, compact, **feed the result back** — the call pattern in
    /// `agent_loop.rs`. Tests must use this rather than re-passing pristine
    /// input: a summary is fingerprinted against the history `compact` returns,
    /// so replaying the original vector exercises a contract the loop never
    /// offers, which is precisely how the fallback livelock stayed hidden.
    async fn drive<F>(
        strategy: &LlmCompaction,
        cfg: &ContextConfig,
        rounds: usize,
        mut next_turn: F,
    ) -> Run
    where
        F: FnMut(usize) -> Vec<AgentMessage>,
    {
        let mut messages: Vec<AgentMessage> = Vec::new();
        let mut run = Run {
            messages: Vec::new(),
            rounds_with_summary: 0,
            peak_tokens: 0,
        };
        for i in 0..rounds {
            messages.extend(next_turn(i));
            messages = strategy.compact(std::mem::take(&mut messages), cfg);
            if has_summary(&messages) {
                run.rounds_with_summary += 1;
            }
            run.peak_tokens = run.peak_tokens.max(total_tokens(&messages));
            settle().await;
        }
        run.messages = messages;
        run
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn splices_summary_when_over_budget() {
        let (provider, _) = ScriptedProvider::new("## Goal\nShip the parser.");
        let strategy = LlmCompaction::from_provider(provider, "mock", "k")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);
        let cfg = config(2_000);

        let run = drive(&strategy, &cfg, 30, |i| turn(i, 400)).await;

        assert!(
            run.rounds_with_summary > 0,
            "a summary must be spliced at some point"
        );
        assert!(
            has_summary(&run.messages),
            "summary must be present at the end"
        );
        assert!(run.messages.iter().any(|m| {
            matches!(m, AgentMessage::Llm(Message::User { content, .. })
                if content.iter().any(|c| matches!(c, Content::Text { text }
                    if text.contains("Ship the parser"))))
        }));
        assert!(run.peak_tokens <= 2_000, "the budget must hold throughout");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn under_trigger_is_a_no_op() {
        let strategy = mock("unused");
        let messages = history(3, 50);
        let out = strategy.compact(messages.clone(), &config(1_000_000));
        assert_eq!(out, messages, "below trigger nothing may change");
        assert!(!strategy.state.lock().unwrap().inflight);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stale_summary_is_discarded_not_spliced() {
        let strategy = mock("## Goal\nStale.")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);

        let messages = history(30, 400);
        let cfg = config(2_000);
        strategy.compact(messages.clone(), &cfg);
        assert!(await_summary(&strategy).await);

        // History rewritten (not append-only): fingerprint must not match.
        let mut mutated = messages.clone();
        mutated[0] = AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: "rewritten".into(),
            }],
            timestamp: 999,
        });
        let out = strategy.compact(mutated, &cfg);
        assert!(!has_summary(&out), "stale summary must be discarded");
        assert!(total_tokens(&out) <= 2_000, "fallback still fits budget");
    }

    /// An edit the old (timestamp, token-estimate) fingerprint could not see:
    /// same timestamp, same length, different bytes.
    #[tokio::test(flavor = "multi_thread")]
    async fn same_shape_rewrite_is_still_detected() {
        let strategy = mock("## Goal\nStale.")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);

        let messages = history(30, 400);
        let cfg = config(2_000);
        strategy.compact(messages.clone(), &cfg);
        assert!(await_summary(&strategy).await);

        let mut mutated = messages.clone();
        mutated[0] = AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                // Same byte length and timestamp as the original, different
                // content: indistinguishable under the old fingerprint.
                text: format!("user message 0: {}", "z".repeat(400)),
            }],
            timestamp: 0,
        });
        let out = strategy.compact(mutated, &cfg);
        assert!(
            !has_summary(&out),
            "a same-length, same-timestamp rewrite must still invalidate the summary"
        );
    }

    /// Regression: a fixed 20k tail made the strategy a silent no-op on any
    /// budget under ~33k — `choose_cut` never found a split, nothing was
    /// logged, and behaviour was identical to `DefaultCompaction`.
    #[tokio::test(flavor = "multi_thread")]
    async fn default_retain_tail_scales_to_a_small_budget() {
        let strategy = mock("## Goal\nSmall budget."); // all defaults
        let cfg = config(2_000); // a fixed 20k tail would swallow all of it

        assert_eq!(
            strategy.retain_tail_tokens(2_000),
            500,
            "the tail must derive from the budget, not the 20k ceiling"
        );

        let run = drive(&strategy, &cfg, 40, |i| turn(i, 400)).await;
        assert!(
            run.rounds_with_summary > 0,
            "defaults must still splice on a small budget"
        );
        assert!(run.peak_tokens <= 2_000);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn impossible_settings_warn_once_instead_of_silently_no_oping() {
        let strategy = mock("unused")
            .with_trigger_ratio(0.1)
            // Far larger than the whole history: no split can ever be found.
            .with_retain_tail_tokens(10_000_000);
        let messages = history(30, 400);
        let cfg = config(2_000);

        for _ in 0..3 {
            let out = strategy.compact(messages.clone(), &cfg);
            assert!(!has_summary(&out));
        }
        let state = strategy.state.lock().unwrap();
        assert!(
            state.warned_inert,
            "the inert case must be reported, not silent"
        );
        assert!(!state.inflight, "nothing should have been spawned");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emits_an_event_on_both_compaction_paths() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let strategy = mock("## Goal\nObserve me.")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200)
            .with_event_sender(tx);
        let cfg = config(2_000);

        // Start already over budget so the first compaction must fall back,
        // then keep driving until a splice lands. Both paths therefore emit.
        let mut messages = history(30, 400);
        for i in 100..140 {
            messages.extend(turn(i, 400));
            messages = strategy.compact(std::mem::take(&mut messages), &cfg);
            settle().await;
        }

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(!events.is_empty(), "compaction must emit events");

        let mut saw_deterministic = false;
        let mut saw_summarized = false;
        for event in &events {
            match event {
                AgentEvent::ContextCompacted {
                    method,
                    tokens_before,
                    tokens_after,
                    messages_summarized,
                    summary_usage,
                    summary_cost_usd,
                    ..
                } => {
                    assert!(tokens_after <= tokens_before);
                    match method {
                        CompactionMethod::Deterministic => {
                            saw_deterministic = true;
                            assert_eq!(*messages_summarized, 0);
                        }
                        CompactionMethod::Summarized => {
                            saw_summarized = true;
                            assert!(*messages_summarized >= MIN_SUMMARIZED_SPAN);
                            assert!(
                                summary_usage.is_some(),
                                "the request's cost must be visible"
                            );
                            // from_provider carries no ModelConfig, no rates.
                            assert!(summary_cost_usd.is_none());
                        }
                    }
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert!(saw_deterministic, "the fallback path must emit");
        assert!(saw_summarized, "the splice path must emit");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn head_is_kept_verbatim_or_summarized_but_not_both() {
        let strategy = mock("summary")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);
        let messages = history(30, 400);
        let cfg = config(2_000);
        let (head_end, cut) = strategy
            .choose_cut(&messages, &cfg, 2_000)
            .expect("a split exists");
        assert_eq!(head_end, 1, "keep_first = 1");
        let transcript = serialize_transcript(&messages[head_end..cut]);
        assert!(
            !transcript.contains("user message 0"),
            "the verbatim head must not also appear inside the summarized span"
        );
        assert!(transcript.contains("user message 1"));
    }

    /// A turn whose tool output is far longer than the default line cap, so
    /// Level 1 has something real to truncate.
    fn tool_turn(i: usize, lines: usize) -> Vec<AgentMessage> {
        let call_id = format!("call-{i}");
        vec![
            AgentMessage::Llm(Message::User {
                content: vec![Content::Text {
                    text: format!("run command {i}"),
                }],
                timestamp: i as u64,
            }),
            AgentMessage::Llm(
                Message::assistant(
                    vec![Content::ToolCall {
                        id: call_id.clone(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": format!("cmd {i}")}),
                        provider_metadata: None,
                    }],
                    StopReason::ToolUse,
                    "mock",
                    "mock",
                    Usage::default(),
                )
                .with_timestamp(i as u64),
            ),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: call_id,
                tool_name: "bash".into(),
                content: vec![Content::Text {
                    text: (0..lines)
                        .map(|l| format!("output line {l} of turn {i}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                }],
                is_error: false,
                timestamp: i as u64,
            }),
        ]
    }

    /// The handoff asked specifically whether Level 1 stays idempotent on a
    /// spliced history — it must, or a later compaction pass rewrites those
    /// bytes again and costs another prefix-cache break. Driven with a verbose
    /// summarizer so the safety net actually runs over the spliced shape.
    #[tokio::test(flavor = "multi_thread")]
    async fn spliced_history_is_level_1_stable() {
        let (provider, _) = ScriptedProvider::new(format!("## Goal\n{}", "verbose ".repeat(500)));
        let strategy = LlmCompaction::from_provider(provider, "mock", "k")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);
        let cfg = config(2_000);

        let run = drive(&strategy, &cfg, 30, |i| tool_turn(i, 400)).await;

        assert!(run.rounds_with_summary > 0, "expected a splice");
        assert!(run.peak_tokens <= 2_000, "the budget must hold throughout");

        // Re-truncating changes nothing, so a later pass will not rewrite these
        // bytes and break the provider's prefix cache a second time.
        for msg in &run.messages {
            assert_eq!(
                &context::truncate_tool_output(msg.clone(), &cfg),
                msg,
                "spliced history must already be Level-1 stable"
            );
        }
        assert!(
            orphaned_tool_calls(&run.messages).is_empty(),
            "spliced tool history must stay structurally valid"
        );
    }

    /// Always returns the same summary and counts requests, so a test can
    /// prove the strategy never pays for a summary it cannot use.
    /// `MockProvider::text` yields its text only once, which is not what a
    /// multi-compaction run needs.
    struct ScriptedProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        text: String,
    }

    impl ScriptedProvider {
        fn new(text: impl Into<String>) -> (Arc<Self>, Arc<std::sync::atomic::AtomicUsize>) {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            (
                Arc::new(Self {
                    calls: Arc::clone(&calls),
                    text: text.into(),
                }),
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl StreamProvider for ScriptedProvider {
        async fn stream(
            &self,
            _config: StreamConfig,
            _tx: tokio::sync::mpsc::UnboundedSender<crate::provider::StreamEvent>,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Message, crate::provider::ProviderError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Message::assistant(
                vec![Content::Text {
                    text: self.text.clone(),
                }],
                StopReason::Stop,
                "mock",
                "mock",
                Usage::default(),
            ))
        }
    }

    /// Every `tool_use` id in `messages` that has no matching `tool_result`.
    /// Providers reject such a sequence outright, so this must always be empty.
    fn orphaned_tool_calls(messages: &[AgentMessage]) -> Vec<String> {
        let (mut opened, mut answered) = (Vec::new(), Vec::new());
        for msg in messages {
            match msg {
                AgentMessage::Llm(Message::Assistant { content, .. }) => {
                    for c in content {
                        if let Content::ToolCall { id, .. } = c {
                            opened.push(id.clone());
                        }
                    }
                }
                AgentMessage::Llm(Message::ToolResult { tool_call_id, .. }) => {
                    answered.push(tool_call_id.clone())
                }
                _ => {}
            }
        }
        opened
            .into_iter()
            .filter(|i| !answered.contains(i))
            .collect()
    }

    /// An assistant message opening two tool calls followed by both results —
    /// the shape `ToolExecutionStrategy::Parallel` produces by default.
    fn parallel_tool_turn(i: usize) -> Vec<AgentMessage> {
        let (a, b) = (format!("call-{i}a"), format!("call-{i}b"));
        vec![
            AgentMessage::Llm(Message::User {
                content: vec![Content::Text {
                    text: format!("do {i}"),
                }],
                timestamp: i as u64,
            }),
            AgentMessage::Llm(
                Message::assistant(
                    vec![
                        Content::tool_call(a.clone(), "bash", serde_json::json!({"c": i})),
                        Content::tool_call(b.clone(), "bash", serde_json::json!({"c": i})),
                    ],
                    StopReason::ToolUse,
                    "mock",
                    "mock",
                    Usage::default(),
                )
                .with_timestamp(i as u64),
            ),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: a,
                tool_name: "bash".into(),
                content: vec![Content::Text {
                    text: format!("out a {i}: {}", "z".repeat(300)),
                }],
                is_error: false,
                timestamp: i as u64,
            }),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: b,
                tool_name: "bash".into(),
                content: vec![Content::Text {
                    text: format!("out b {i}: {}", "z".repeat(300)),
                }],
                is_error: false,
                timestamp: i as u64,
            }),
        ]
    }

    /// Regression: the agent loop writes `compact()`'s result straight back
    /// (`agent_loop.rs`), so a summary fingerprinted over the *input* was
    /// invalidated by the same call whenever the fallback rewrote history. That
    /// state was absorbing — measured at 22 paid requests and 0 splices over 25
    /// rounds. Every other test in this file re-passes the pristine input, which
    /// is exactly why none of them caught it.
    #[tokio::test(flavor = "multi_thread")]
    async fn feeding_the_result_back_still_splices() {
        for turn_chars in [200usize, 800, 2000] {
            let (provider, calls) = ScriptedProvider::new("## Goal\nThe briefing.");
            let strategy =
                LlmCompaction::from_provider(provider, "mock", "k").with_retain_tail_tokens(200);
            let cfg = config(2_000);

            let mut messages: Vec<AgentMessage> = Vec::new();
            let mut spliced_rounds = 0usize;
            for i in 0..25 {
                messages.extend(turn(i, turn_chars));
                // The line that matters: feed the result back, as the loop does.
                messages = strategy.compact(std::mem::take(&mut messages), &cfg);
                if has_summary(&messages) {
                    spliced_rounds += 1;
                }
                for _ in 0..25 {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
            let requests = calls.load(std::sync::atomic::Ordering::SeqCst);
            // The invariant the fix establishes: never pay for a summary that
            // cannot be used. Either the run splices, or it issues no requests
            // at all (turns so large relative to the budget that the compacted
            // history is never long enough to be worth summarizing). Before the
            // fix this configuration issued 22 requests and spliced zero times.
            assert!(
                spliced_rounds > 0 || requests == 0,
                "turn_chars={turn_chars}: {requests} summarization requests paid for and never \
                 spliced — the fallback is invalidating its own pending summary"
            );
        }
    }

    /// Regression: a session restored over budget (e.g. `Agent::with_messages`)
    /// enters the loop on the fallback path. It must still reach a splice.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_that_starts_over_budget_recovers() {
        let (provider, calls) = ScriptedProvider::new("## Goal\nRecovered.");
        let strategy =
            LlmCompaction::from_provider(provider, "mock", "k").with_retain_tail_tokens(200);
        let cfg = config(2_000);

        let mut messages = history(30, 400);
        assert!(total_tokens(&messages) > 2_000, "must start over budget");

        let mut spliced_ever = false;
        for i in 100..130 {
            messages.extend(turn(i, 400));
            messages = strategy.compact(std::mem::take(&mut messages), &cfg);
            if has_summary(&messages) {
                spliced_ever = true;
            }
            for _ in 0..25 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        }
        assert!(
            spliced_ever,
            "{} requests issued, never spliced",
            calls.load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    /// Regression: the summary sits exactly where `level3_drop_middle` starts
    /// cutting, so routing an over-budget splice through `compact_messages`
    /// deleted the briefing while the event still reported `Summarized`.
    /// A verbose summarizer is what pushes the spliced result over the budget.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_safety_net_preserves_the_briefing_it_paid_for() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // ~1000 tokens of briefing: big enough that head + summary + tail
        // overflows a 2k budget, small enough that head + summary still fits.
        let (provider, _) = ScriptedProvider::new(format!("## Goal\n{}", "verbose ".repeat(500)));
        let strategy = LlmCompaction::from_provider(provider, "mock", "k")
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200)
            .with_event_sender(tx);
        let cfg = config(2_000);

        let run = drive(&strategy, &cfg, 30, |i| turn(i, 400)).await;

        assert!(run.rounds_with_summary > 0, "expected at least one splice");
        assert!(run.peak_tokens <= 2_000, "the budget must hold throughout");

        let mut saw_summarized = false;
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::ContextCompacted {
                method,
                messages_summarized,
                tokens_after,
                ..
            } = event
            {
                if method == CompactionMethod::Summarized {
                    saw_summarized = true;
                    assert!(
                        messages_summarized > 0,
                        "a Summarized event must report a real span"
                    );
                    assert!(tokens_after <= 2_000);
                }
            }
        }
        assert!(
            saw_summarized,
            "the briefing was paid for but never reported as spliced"
        );
    }

    /// Regression: `safe_head_end` walks back past an assistant that opens tool
    /// calls but not past the results, so a `keep_first` landing inside a
    /// parallel tool-result run kept the assistant and only some of its
    /// results. `keep_first: 3` orphaned `call-0b`; providers reject that.
    #[tokio::test(flavor = "multi_thread")]
    async fn splice_never_orphans_a_parallel_tool_call() {
        for keep_first in 1..=6usize {
            let (provider, _) = ScriptedProvider::new("## Goal\nX.");
            let strategy = LlmCompaction::from_provider(provider, "mock", "k")
                .with_trigger_ratio(0.1)
                .with_retain_tail_tokens(300);
            let cfg = ContextConfig {
                max_context_tokens: 2_000,
                system_prompt_tokens: 0,
                keep_first,
                keep_recent: 2,
                ..Default::default()
            };

            let run = drive(&strategy, &cfg, 30, parallel_tool_turn).await;

            assert!(
                run.rounds_with_summary > 0,
                "keep_first={keep_first}: expected a splice"
            );
            let orphans = orphaned_tool_calls(&run.messages);
            assert!(
                orphans.is_empty(),
                "keep_first={keep_first}: orphaned tool_use ids {orphans:?} — a provider \
                 would reject this outright"
            );
        }
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
        let strategy =
            LlmCompaction::from_provider(Arc::new(FailingProvider), "mock-model", "test-key")
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
