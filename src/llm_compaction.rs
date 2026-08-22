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
//! # Choosing the summarizer
//!
//! **Do not reuse the loop's `ModelConfig`.** `LlmCompaction::from_config(cfg)`
//! with the session's own config is the obvious call and the worst one: the
//! briefing then runs on the slow, expensive model, and a briefing that cannot
//! finish before the budget is crossed is not merely late — the compaction that
//! fires meanwhile rewrites the history it was computed over, so it is
//! discarded on arrival by the fingerprint check.
//!
//! Measured on a 25-turn tool-heavy run at a 30K budget:
//!
//! | summarizer | first compaction | history retained |
//! |---|---|---|
//! | the loop's model (Sonnet 5) | `Deterministic` | 3 msgs / 1.7K tokens |
//! | a fast model (Haiku 4.5) | `Summarized` | 22 msgs / 16.7K tokens |
//!
//! ```rust,ignore
//! LlmCompaction::from_config(ModelConfig::claude_haiku_4_5())   // not the loop's config
//! ```
//!
//! When briefings keep losing that race the strategy says so once, via
//! `tracing::warn!`, rather than degrading silently — a session that always
//! falls back still pays for every summarization request.
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
//! # Why the request is standalone, not appended to the live session
//!
//! An obvious-looking optimisation is to append the summarization instruction
//! to the *live* session instead: the history is then a prefix-cache hit, so
//! the span costs cached-input rates rather than fresh input. Measured against
//! real pricing, it is a bad trade almost everywhere, and the reason is
//! counterintuitive enough to be worth recording (yologdev/yoagent#127).
//!
//! In-session forces the briefing to be billed at the **loop model's** output
//! rate. Output runs 5-10x input, so the cache saving on the span is swamped
//! by the output penalty — and the more expensive the loop model, the worse it
//! gets. Per compaction, at a ~12K history / ~8K span / ~1K briefing:
//!
//! | loop model | standalone on Haiku | appended in-session |
//! |---|---|---|
//! | Sonnet 5 | $0.0130 | $0.0126 |
//! | Opus 5 | $0.0130 | $0.0315 |
//! | Fable 5 | $0.0130 | $0.0630 |
//!
//! So the idea is backwards: it was motivated by "reuse the expensive model's
//! cache", and the expensive model is exactly where it loses. It only wins when
//! the loop model *is* the summarizer — DeepSeek in-session runs 3.2x cheaper
//! than DeepSeek standalone — which is the configuration this strategy already
//! steers away from, since routing summarization at a cheap model is the
//! documented shape.
//!
//! The downside is worse than the upside. A background request that races the
//! loop's own turn can arrive before the cache write lands and miss entirely,
//! paying full input on the *whole history* rather than fresh input on the
//! span: 2.6x the standalone cost on Sonnet. The best case is a few percent,
//! the failure case is a 2.6x overrun, and the race is not something this
//! crate controls.
//!
//! Recorded as **no-go**. Revisit if a provider appears whose output rates are
//! flat across model tiers, or if a same-model summarizer becomes the common
//! configuration rather than the fallback.
//!
//! **Does not buy: fewer prefix-cache breaks.** Both strategies rewrite history
//! only when the budget is crossed, and neither rewrites in between, so the
//! number of rewrites over a session is a wash: **6 vs 6** over 120 turns at a
//! 20k budget, **6 vs 5** over 600 turns at 100k. Splicing at the last possible
//! moment describes when this strategy breaks the cache relative to a
//! *synchronous* summarizer, which must break it the moment it decides to
//! summarize. It is not an advantage over
//! [`DefaultCompaction`](crate::context::DefaultCompaction).
//!
//! Those figures are reproducible rather than folklore. They come from
//! `tests/prefix_cache_harness.rs`, measured at commit `719160f`:
//!
//! ```text
//! cargo test --test prefix_cache_harness -- --ignored --nocapture
//! ```
//!
//! Re-run it and update this section after any change to compaction sizing —
//! [`ContextConfig::compact_target_ratio`], [`ContextConfig::compact_headroom_turns`],
//! [`DEFAULT_TRIGGER_RATIO`] and the `retain_tail_tokens` derivation all move
//! these numbers, and nothing in CI will notice if they drift.
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
//! **A bad briefing is never spliced.** Splicing *deletes* the span it replaces,
//! so a briefing is validated before it is allowed to: a non-`Stop` stop reason
//! (a `Length`-truncated handoff missing its "Open items" section, a `Refusal`,
//! an overflow `Error`) is rejected with the reason logged, and the turn falls
//! back to the deterministic tiers.
//!
//! **Failure is never silent, and never permanent.** The request has a timeout
//! ([`DEFAULT_REQUEST_TIMEOUT`]) and retries retryable errors; the in-flight
//! slot is released by a drop guard, so a hung, panicking, or dropped task
//! cannot pin it for the rest of the session. Every path that gives up says so
//! at `warn!`, and the per-compaction cost is logged at `info!` whether or not
//! an event sender is wired.
//!
//! # One accepted limit
//!
//! **The API key is resolved once, at construction.**
//! [`from_config`](LlmCompaction::from_config) reads the provider-conventional
//! environment variable when the strategy is built (warning if it finds none)
//! and holds the result for its lifetime. Note this is *not*
//! [`Agent`](crate::Agent)'s contract: `Agent` resolves lazily per request, so
//! it does pick up a rotated environment key. Build a fresh strategy — or pass
//! [`with_api_key`](LlmCompaction::with_api_key) — if credentials rotate
//! mid-session.
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
use crate::retry::RetryConfig;
use crate::types::CacheConfig;
use crate::types::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

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

/// How long a single summarization attempt may run before it is abandoned.
///
/// Without this a hung request pins the in-flight slot for the life of the
/// session: no provider in this crate imposes its own timeout, so the task
/// simply never completes and the strategy degrades to
/// [`DefaultCompaction`](crate::context::DefaultCompaction) in silence.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Constant first line of the spliced summary message.
///
/// Constant so splices are recognizable in transcripts and replays. Note this
/// buys no prefix-cache stability of its own — unlike `context`'s compaction
/// marker, the varying briefing follows it in the same message.
pub const SUMMARY_MARKER: &str = "[Context compacted — summary of earlier conversation]";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a context summarization assistant. You produce \
     structured handoff briefings of agent conversations so that work can \
     continue seamlessly with the summary in place of the original messages.";

const DEFAULT_INSTRUCTION: &str = "Summarize the conversation above as a handoff briefing for \
     an agent that will continue this work without access to the original \
     messages. Use exactly these sections:\n\
     ## Goal\nWhat the user is trying to accomplish, verbatim where possible.\n\
     ## State & progress\nWhat has been done, what is currently in flight.\n\
     ## Key decisions & constraints\nDecisions made and why. Record the \
     constraints you were *given* as well as the choices made in response to \
     them — deployment shape, scale, hard dependencies, things ruled out, \
     stated preferences. A reader who keeps the decisions but loses the \
     conditions that forced them cannot tell which are still binding.\n\
     ## Open items\nUnresolved questions and concrete next steps.\n\
     Be dense and factual. Include exact identifiers (paths, names, versions, \
     numbers) — those are the details the next agent cannot reconstruct.";

/// Cap on the bytes of any single content block serialized into the
/// summarization transcript. Long tool outputs were already truncated on
/// append; this bounds the pathological rest.
const TRANSCRIPT_PER_BLOCK_BYTES: usize = 2_000;

/// Cap on the bytes of the whole transcript.
///
/// Per-block bounding is not enough on its own: the summarized span scales with
/// the *main* model's budget, so a 1M-context session can hand a 200k-context
/// summarizer far more than it can read — every request then fails on overflow
/// and the strategy pays for nothing. Roughly 120k tokens at four bytes each,
/// which fits the smallest model anyone routes this at. Checked per message, so
/// the transcript may overshoot by at most one message's contribution.
const TRANSCRIPT_TOTAL_BYTES: usize = 480_000;

/// A span shorter than this is not worth a request or a cache break.
const MIN_SUMMARIZED_SPAN: usize = 4;

/// Floor on `max_summary_tokens`. Below this the model cannot produce the four
/// sections the instruction asks for, and truncated briefings are rejected.
const MIN_SUMMARY_TOKENS: u32 = 256;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Identity of a snapshotted prefix, checked before splicing.
///
/// History is append-only during a run, so `messages[0..cut)` should be
/// unchanged when the background task finishes — but "should" is not a
/// correctness argument, and two things really can invalidate it:
/// [`Agent::replace_messages`](crate::Agent::replace_messages), and a
/// deterministic fallback rewriting history between spawn and splice. (The
/// latter is why the spawn anchors on what `compact` returns; see
/// [`arm`](LlmCompaction::arm). `transform_context` cannot: the loop runs it on
/// a clone and never writes the result back.)
///
/// The hash folds each prefix message's index and its serialized bytes, so any
/// realistic edit changes it and the stale summary is dropped instead of
/// spliced. It is a 64-bit hash, not an identity check — a collision is
/// possible in principle and has never been the failure anyone hits.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    cut: usize,
    hash: u64,
}

fn fingerprint(messages: &[AgentMessage], cut: usize) -> Fingerprint {
    let mut hasher = DefaultHasher::new();
    for (i, msg) in messages[..cut].iter().enumerate() {
        i.hash(&mut hasher);
        match serde_json::to_vec(msg) {
            Ok(bytes) => bytes.hash(&mut hasher),
            Err(e) => {
                // Not reachable for the `AgentMessage` shapes serde_json can
                // emit. If it ever is, two different unserializable messages
                // would fold alike and weaken the very check this preserves —
                // so say so rather than degrading quietly.
                debug_assert!(false, "AgentMessage failed to serialize: {e}");
                tracing::error!("llm compaction: message {i} failed to serialize: {e}");
                u8::MAX.hash(&mut hasher);
            }
        }
    }
    Fingerprint {
        cut,
        hash: hasher.finish(),
    }
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

/// A finished summary waiting to be spliced.
struct Summary {
    fingerprint: Fingerprint,
    /// Where the verbatim head ends. The briefing covers `[head_end, cut)`;
    /// `messages[..head_end]` survives the splice untouched. Captured at spawn
    /// time so the splice cannot disagree with what was actually summarized.
    head_end: usize,
    text: String,
    /// What the summarization request itself cost.
    usage: Usage,
}

/// What the strategy is doing, as one value.
///
/// Modelled as an enum rather than `(bool, Option<Summary>)` so the illegal
/// "in flight *and* holding a result" state cannot be represented, and so every
/// transition is one lock acquisition instead of a read followed by a write.
enum Phase {
    Idle,
    Inflight,
    Ready(Box<Summary>),
}

impl Phase {
    fn is_idle(&self) -> bool {
        matches!(self, Phase::Idle)
    }
}

#[derive(Default)]
struct Warned {
    /// "No split is possible" has been reported.
    inert: bool,
    /// "No tokio runtime" has been reported.
    no_runtime: bool,
    /// "Briefings keep losing the race" has been reported.
    losing_race: bool,
}

/// Consecutive deterministic fallbacks before reporting that briefings are
/// being paid for and not used. Two, not one: a single fallback is ordinary —
/// the first compaction of a session usually arrives before any summary could
/// have been ready.
const FALLBACKS_BEFORE_WARNING: u32 = 2;

struct State {
    phase: Phase,
    warned: Warned,
    /// Consecutive compactions that took the deterministic path.
    fallbacks: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            fallbacks: 0,
            warned: Warned::default(),
        }
    }
}

/// Returns the in-flight slot to [`Phase::Idle`] when the summarization task
/// ends, however it ends.
///
/// Setting the flag outside the task and clearing it inside meant a task that
/// never reached the clear — hung request, panicking provider, runtime shutting
/// down between spawn and poll — pinned the slot forever, silently degrading the
/// strategy to `DefaultCompaction` for the rest of the session. Tying the reset
/// to a guard's `Drop` ties it to the task's lifetime instead of to two manually
/// matched assignments.
struct InflightGuard {
    state: Arc<Mutex<State>>,
    /// Set once the task has stored a result, so `Drop` does not clear it.
    disarmed: bool,
}

impl InflightGuard {
    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        let mut state = lock(&self.state);
        if matches!(state.phase, Phase::Inflight) {
            state.phase = Phase::Idle;
        }
    }
}

/// Lock the state, tolerating poisoning.
///
/// A panic anywhere under this mutex would otherwise make every later
/// `compact()` panic, which would break the one guarantee the strategy makes
/// unconditionally: that the loop can never wedge on it. The protected data is
/// a small state machine with no cross-field invariant that a partial write
/// could corrupt, so recovering the inner value is strictly better than
/// propagating.
fn lock(state: &Arc<Mutex<State>>) -> MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    config: ModelConfig,
    api_key: String,
    trigger_ratio: f32,
    /// `None` derives it from the budget at call time — see
    /// [`with_retain_tail_tokens`](LlmCompaction::with_retain_tail_tokens).
    retain_tail_tokens: Option<usize>,
    system_prompt: String,
    instruction: String,
    max_summary_tokens: u32,
    timeout: Duration,
    retry: RetryConfig,
    events: Option<UnboundedSender<AgentEvent>>,
    /// Cancels in-flight summarization when the strategy is dropped, so an
    /// abandoned run stops billing instead of streaming into a state nobody
    /// will read.
    cancel: CancellationToken,
    state: Arc<Mutex<State>>,
}

/// Redacts `api_key`; `ModelConfig`'s own `Debug` redacts header values.
impl std::fmt::Debug for LlmCompaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmCompaction")
            .field("config", &self.config)
            .field("trigger_ratio", &self.trigger_ratio)
            .field("retain_tail_tokens", &self.retain_tail_tokens)
            .field("max_summary_tokens", &self.max_summary_tokens)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl Drop for LlmCompaction {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
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
        Self::from_config_with(&crate::provider::ProviderRegistry::default(), config)
            .expect("default registry covers all built-in protocols")
    }

    /// Like [`from_config`](Self::from_config) but resolves the provider from a
    /// caller-supplied registry, returning an error if the config's protocol
    /// isn't registered.
    pub fn from_config_with(
        registry: &crate::provider::ProviderRegistry,
        config: ModelConfig,
    ) -> Result<Self, crate::AgentBuildError> {
        let provider = registry
            .resolve(&config.api)
            .ok_or(crate::AgentBuildError::NoProviderForProtocol(config.api))?;
        let api_key = crate::provider::resolve_api_key_or_warn(&config.provider);
        Ok(Self::build(provider, config, api_key))
    }

    /// A compaction strategy that summarizes with an explicit provider.
    ///
    /// The escape hatch for custom [`StreamProvider`] implementations and test
    /// doubles — pair with [`ModelConfig::mock`](crate::provider::ModelConfig::mock).
    /// The config is still required so the model id, context window, and pricing
    /// stay defined together, matching
    /// [`Agent::from_provider`](crate::Agent::from_provider).
    pub fn from_provider(provider: Arc<dyn StreamProvider>, config: ModelConfig) -> Self {
        let api_key = crate::provider::resolve_api_key_or_warn(&config.provider);
        Self::build(provider, config, api_key)
    }

    fn build(provider: Arc<dyn StreamProvider>, config: ModelConfig, api_key: String) -> Self {
        Self {
            provider,
            config,
            api_key,
            trigger_ratio: DEFAULT_TRIGGER_RATIO,
            retain_tail_tokens: None,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            instruction: DEFAULT_INSTRUCTION.into(),
            max_summary_tokens: 2_000,
            timeout: DEFAULT_REQUEST_TIMEOUT,
            retry: RetryConfig::default(),
            events: None,
            cancel: CancellationToken::new(),
            state: Arc::new(Mutex::new(State::default())),
        }
    }

    /// Use an explicit API key instead of the environment-resolved one.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Fraction of the budget at which background summarization starts.
    /// Clamped to `[0.1, 0.95]`. Default: [`DEFAULT_TRIGGER_RATIO`].
    pub fn with_trigger_ratio(mut self, ratio: f32) -> Self {
        let clamped = if ratio.is_finite() {
            ratio.clamp(0.1, 0.95)
        } else {
            DEFAULT_TRIGGER_RATIO
        };
        if clamped != ratio {
            tracing::warn!(
                "llm compaction: trigger_ratio {ratio} is out of range, using {clamped}"
            );
        }
        self.trigger_ratio = clamped;
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

    /// Cap on briefing length in output tokens. Default: 2000, floor: 256.
    ///
    /// A briefing that hits this cap comes back with `StopReason::Length` and is
    /// rejected rather than spliced — a truncated handoff loses the "Open items"
    /// section the instruction puts last.
    pub fn with_max_summary_tokens(mut self, tokens: u32) -> Self {
        if tokens < MIN_SUMMARY_TOKENS {
            tracing::warn!(
                "llm compaction: max_summary_tokens {tokens} is below the {MIN_SUMMARY_TOKENS} \
                 floor; using the floor"
            );
        }
        self.max_summary_tokens = tokens.max(MIN_SUMMARY_TOKENS);
        self
    }

    /// How long one summarization attempt may run.
    /// Default: [`DEFAULT_REQUEST_TIMEOUT`].
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Retry policy for the summarization request. Default:
    /// [`RetryConfig::default`]; pass [`RetryConfig::none`] to disable.
    pub fn with_retry_config(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Emit [`AgentEvent::ContextCompacted`] on this channel when compaction
    /// runs, by either path.
    ///
    /// [`CompactionStrategy::compact`] has no access to the loop's event
    /// channel, so the sender has to come in from the side. Pair it with
    /// [`Agent::prompt_with_sender`](crate::Agent::prompt_with_sender), where
    /// the caller owns the channel. Without it the per-compaction cost is still
    /// logged at `info!`, but nothing structured is emitted.
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
            if tx.send(event).is_err() {
                tracing::debug!("llm compaction: event receiver dropped");
            }
        }
    }

    /// Cost of the summarization request, when the model's rates are known.
    fn summary_cost(&self, usage: &Usage) -> Option<f64> {
        let cost = &self.config.cost;
        cost.is_configured().then(|| cost.cost_usd(usage))
    }

    /// The tail budget for this call: explicit if set, else derived from the
    /// context budget so it cannot swallow the whole history.
    fn effective_retain_tail(&self, budget: usize) -> usize {
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
    ) -> Result<(usize, usize), NoCut> {
        let len = messages.len();
        let head_end = safe_head_boundary(messages, config.keep_first.min(len));
        let retain = self.effective_retain_tail(budget);

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

        if cut > head_end && cut - head_end >= MIN_SUMMARIZED_SPAN {
            return Ok((head_end, cut));
        }
        // Distinguish the two reasons, because only one is a misconfiguration.
        // A history that is merely short yet becomes summarizable as it grows;
        // reporting that permanently would spend the one-shot warning on a
        // benign startup condition and leave the real case silent.
        if len.saturating_sub(head_end) < MIN_SUMMARIZED_SPAN + config.keep_recent {
            Err(NoCut::HistoryTooShort)
        } else {
            Err(NoCut::TailTooLarge)
        }
    }

    /// Report, once, that no summary can be produced under the current
    /// settings. Silence here was the original bug: the strategy looked
    /// configured and did nothing, forever.
    /// Report, once, that briefings are being paid for and thrown away.
    ///
    /// A run where every compaction takes the deterministic path still issues
    /// a summarization request each time — so the session pays input tokens
    /// for the summarized span and output tokens for a briefing it never uses,
    /// and silently gets the lossy behaviour `LlmCompaction` was chosen to
    /// avoid. `CompactionMethod::Deterministic` on the event is the only other
    /// signal, and only if the caller is listening for it.
    ///
    /// Measured cause, in order of likelihood: the summarizer is the *loop's*
    /// model. `LlmCompaction::from_config(loop_config)` is the obvious call and
    /// the worst one — a slow briefing loses the race to the budget, and the
    /// compaction that fires meanwhile rewrites the very history the briefing
    /// was computed over, so it is discarded on arrival even when it does land.
    fn warn_losing_race_once(&self) {
        {
            let mut state = lock(&self.state);
            state.fallbacks += 1;
            if state.fallbacks < FALLBACKS_BEFORE_WARNING || state.warned.losing_race {
                return;
            }
            state.warned.losing_race = true;
        }
        tracing::warn!(
            "llm compaction: {FALLBACKS_BEFORE_WARNING} compactions in a row fell back to the \
             deterministic tiers, so this session is paying for briefings it never uses. The \
             usual cause is a summarizer that cannot finish before the budget is crossed — name \
             a cheaper, faster model than the loop's rather than reusing its config, or lower \
             trigger_ratio (currently {}) to start summarizing sooner.",
            self.trigger_ratio,
        );
    }

    fn warn_inert_once(&self, used: usize, budget: usize, config: &ContextConfig) {
        {
            let mut state = lock(&self.state);
            if state.warned.inert {
                return;
            }
            state.warned.inert = true;
        }
        let retain = self.effective_retain_tail(budget);
        tracing::warn!(
            "llm compaction is inert: past the trigger ({used} of {budget} budget tokens) there \
             is still no split leaving {retain} tail tokens, {} recent messages and {} messages \
             to summarize, so every compaction will fall back to the deterministic tiers. Lower \
             retain_tail_tokens or keep_recent, or raise trigger_ratio (currently {}).",
            config.keep_recent,
            MIN_SUMMARIZED_SPAN,
            self.trigger_ratio,
        );
    }

    /// Spawn the standalone summarization request for `messages[head_end..cut)`.
    fn spawn_summarize(&self, messages: &[AgentMessage], head_end: usize, cut: usize) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // The strategy is wholly inert here — every compaction will be
            // deterministic — so this is a warning, not a debug note.
            let mut state = lock(&self.state);
            if !state.warned.no_runtime {
                state.warned.no_runtime = true;
                tracing::warn!(
                    "llm compaction: no tokio runtime, so background summarization is impossible \
                     and every compaction will use the deterministic tiers"
                );
            }
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
        let cancel = self.cancel.child_token();
        let (timeout, retry) = (self.timeout, self.retry.clone());

        // `StreamConfig` is #[non_exhaustive]: construct via `new` and mutate,
        // per its own documented convention.
        let mut stream_config = StreamConfig::new(self.config.id.clone(), self.api_key.clone());
        stream_config.system_prompt = self.system_prompt.clone();
        stream_config.messages = vec![Message::user(format!(
            "<conversation>\n{transcript}\n</conversation>\n\n{}",
            self.instruction
        ))];
        stream_config.max_tokens = Some(self.max_summary_tokens);
        stream_config.model_config = Some(self.config.clone());
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

        lock(&state).phase = Phase::Inflight;
        tracing::debug!(
            "llm compaction: summarizing messages[{}..{}) in background",
            head_end,
            cut
        );

        handle.spawn(async move {
            // Whatever happens below — hung request, panic, runtime shutdown —
            // the slot returns to Idle when this guard drops.
            let mut guard = InflightGuard {
                state: Arc::clone(&state),
                disarmed: false,
            };

            let outcome = summarize(&provider, stream_config, timeout, &retry, &cancel).await;
            let Some((text, usage)) = outcome else { return };

            lock(&state).phase = Phase::Ready(Box::new(Summary {
                fingerprint: fp,
                head_end,
                usage,
                text,
            }));
            guard.disarm();
        });
    }

    /// Splice a ready summary: `[head][summary][tail cut..]`.
    fn splice(&self, messages: &[AgentMessage], summary: &Summary) -> Vec<AgentMessage> {
        let cut = summary.fingerprint.cut;
        let head_end = summary.head_end;
        debug_assert!(head_end < cut, "summary span must be non-empty");
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
        tracing::debug!(
            "llm compaction: spliced result exceeds the budget; compacting the tail against \
             {} of {budget} tokens",
            budget.saturating_sub(fixed)
        );
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
        if !lock(&self.state).phase.is_idle() {
            return;
        }
        match self.choose_cut(messages, config, budget) {
            Ok((head_end, cut)) => self.spawn_summarize(messages, head_end, cut),
            // Transient: the history simply has not grown enough yet. Reporting
            // this permanently would spend the one-shot warning on a benign
            // startup condition.
            Err(NoCut::HistoryTooShort) => tracing::debug!(
                "llm compaction: history too short to summarize yet ({} messages)",
                messages.len()
            ),
            Err(NoCut::TailTooLarge) => self.warn_inert_once(used, budget, config),
        }
    }
}

/// Why [`LlmCompaction::choose_cut`] found no split.
#[derive(Debug)]
enum NoCut {
    /// Not enough history yet — expected early, resolves as the session grows.
    HistoryTooShort,
    /// The retained tail leaves nothing to summarize — a misconfiguration that
    /// will not resolve on its own.
    TailTooLarge,
}

/// Run one summarization request to completion, with timeout and retry.
///
/// Returns the briefing and its usage, or `None` if it could not be produced —
/// every failure path logs before returning, so a silent degradation to
/// deterministic compaction is not possible.
async fn summarize(
    provider: &Arc<dyn StreamProvider>,
    stream_config: StreamConfig,
    timeout: Duration,
    retry: &RetryConfig,
    cancel: &CancellationToken,
) -> Option<(String, Usage)> {
    for attempt in 0..=retry.max_retries {
        if cancel.is_cancelled() {
            tracing::debug!("llm compaction: cancelled");
            return None;
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // Drain events we don't consume so the provider never blocks.
        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let result = tokio::time::timeout(
            timeout,
            provider.stream(stream_config.clone(), tx, cancel.clone()),
        )
        .await;
        drain.abort();

        match result {
            Err(_elapsed) => {
                tracing::warn!(
                    "llm compaction: summarization timed out after {timeout:?} (attempt {})",
                    attempt + 1
                );
            }
            Ok(Err(e)) => {
                if e.is_retryable() && attempt < retry.max_retries {
                    let delay = e
                        .retry_after()
                        .unwrap_or_else(|| retry.delay_for_attempt(attempt));
                    tracing::debug!("llm compaction: {e}, retrying in {delay:?}");
                    tokio::time::sleep(delay).await;
                    continue;
                }
                tracing::warn!("llm compaction: summarization failed: {e}");
                return None;
            }
            Ok(Ok(message)) => return accept_summary(message),
        }
        if attempt < retry.max_retries {
            tokio::time::sleep(retry.delay_for_attempt(attempt)).await;
        }
    }
    None
}

/// Validate a completed response before it is allowed to replace history.
///
/// `Ok(message)` routinely carries failure in this crate: `StopReason::Length`
/// means the briefing was cut off mid-sentence (and the instruction puts "Open
/// items" last, so truncation eats the next steps), `Refusal` means the text is
/// a refusal that would be spliced over turns that are then deleted, and `Error`
/// carries a diagnostic worth surfacing. Splicing *deletes* the summarized span,
/// so accepting a bad briefing is permanent data loss.
fn accept_summary(message: Message) -> Option<(String, Usage)> {
    if let Message::Assistant {
        stop_reason,
        error_message,
        ..
    } = &message
    {
        if !matches!(stop_reason, StopReason::Stop) {
            tracing::warn!(
                "llm compaction: rejecting summary (stop_reason={stop_reason:?}, overflow={}): \
                 {}; falling back to deterministic compaction",
                message.is_context_overflow(),
                error_message.as_deref().unwrap_or("no detail")
            );
            return None;
        }
    }
    let text = assistant_text(&message);
    if text.trim().is_empty() {
        tracing::warn!("llm compaction: empty summary, discarding");
        return None;
    }
    tracing::debug!("llm compaction: summary ready ({} chars)", text.len());
    Some((text, assistant_usage(&message)))
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
            let ready = {
                let mut state = lock(&self.state);
                match std::mem::replace(&mut state.phase, Phase::Idle) {
                    Phase::Ready(summary) => Some(summary),
                    other => {
                        state.phase = other;
                        None
                    }
                }
            };
            if let Some(summary) = ready {
                let fp = &summary.fingerprint;
                if fp.cut <= messages.len() && fingerprint(&messages, fp.cut) == *fp {
                    let mut summarized = fp.cut - summary.head_end;
                    let mut method = CompactionMethod::Summarized;
                    let mut result = self.splice(&messages, &summary);
                    if total_tokens(&result) > budget {
                        result = self.shrink_tail(result, config, budget, summary.head_end);
                        if total_tokens(&result) > budget {
                            // Head plus briefing alone exceed the budget, so the
                            // briefing cannot be kept. Report what really
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
                    let cost = self.summary_cost(&summary.usage);
                    // Logged as well as emitted: the event channel is opt-in,
                    // so without this the per-compaction cost would be visible
                    // nowhere in the default configuration.
                    tracing::info!(
                        "llm compaction: {method:?} — summarized {summarized} messages, \
                         {messages_before} -> {} messages ({used} -> {after} tokens); \
                         request used {} in / {} out{}",
                        result.len(),
                        summary.usage.input,
                        summary.usage.output,
                        cost.map(|c| format!(", ${c:.4}")).unwrap_or_default(),
                    );
                    self.emit(AgentEvent::ContextCompacted {
                        method,
                        messages_before,
                        messages_after: result.len(),
                        tokens_before: used,
                        tokens_after: after,
                        // The request was paid for either way, so its cost is
                        // reported even when the briefing could not be kept.
                        summary: Some(SummaryStats::new(summarized, summary.usage, cost)),
                    });
                    lock(&self.state).fallbacks = 0;
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
            self.warn_losing_race_once();
            let result = compact_messages(messages, config);
            let after = total_tokens(&result);
            self.emit(AgentEvent::ContextCompacted {
                method: CompactionMethod::Deterministic,
                messages_before,
                messages_after: result.len(),
                tokens_before: used,
                tokens_after: after,
                summary: None,
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

/// Clip to at most `max` **bytes**, on a char boundary.
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
/// summarization request. Bounded per content block *and* in total; tool
/// arguments and results are included in clipped form — identifiers matter,
/// repetition does not.
fn serialize_transcript(messages: &[AgentMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        if out.len() >= TRANSCRIPT_TOTAL_BYTES {
            // Oldest-first, so what survives is the most recent context. The
            // alternative is a request the summarizer cannot read at all.
            tracing::warn!(
                "llm compaction: transcript hit the {TRANSCRIPT_TOTAL_BYTES}-byte cap;                  summarizing only the most recent part of the span"
            );
            out.push_str("[... earlier messages omitted: transcript size cap ...]\n");
            break;
        }
        let AgentMessage::Llm(message) = msg else {
            continue; // Extension messages never reach the LLM anyway.
        };
        match message {
            Message::User { content, .. } => {
                for c in content {
                    match c {
                        Content::Text { text } => {
                            out.push_str("User: ");
                            out.push_str(clip(text, TRANSCRIPT_PER_BLOCK_BYTES));
                            out.push('\n');
                        }
                        // Undisclosed lossiness would be worse: let the
                        // briefing note that an attachment existed.
                        Content::Image { .. } => out.push_str("[image omitted]\n"),
                        _ => {}
                    }
                }
            }
            Message::Assistant { content, .. } => {
                for c in content {
                    match c {
                        Content::Text { text } => {
                            out.push_str("Assistant: ");
                            out.push_str(clip(text, TRANSCRIPT_PER_BLOCK_BYTES));
                            out.push('\n');
                        }
                        Content::ToolCall {
                            name, arguments, ..
                        } => {
                            let args = arguments.to_string();
                            out.push_str(&format!("[tool call] {name}({})\n", clip(&args, 300)));
                        }
                        Content::Thinking { .. } => out.push_str("[thinking omitted]\n"),
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
                            clip(text, TRANSCRIPT_PER_BLOCK_BYTES)
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
        LlmCompaction::from_provider(Arc::new(MockProvider::text(text)), ModelConfig::mock())
    }

    /// Poll until the background summarization lands, or give up.
    async fn await_summary(strategy: &LlmCompaction) -> bool {
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if matches!(lock(&strategy.state).phase, Phase::Ready(_)) {
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
        let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
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
        assert!(lock(&strategy.state).phase.is_idle());
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
            strategy.effective_retain_tail(2_000),
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
        let state = lock(&strategy.state);
        assert!(
            state.warned.inert,
            "the inert case must be reported, not silent"
        );
        assert!(state.phase.is_idle(), "nothing should have been spawned");
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
                    summary,
                    ..
                } => {
                    assert!(tokens_after <= tokens_before);
                    match method {
                        CompactionMethod::Deterministic => saw_deterministic = true,
                        CompactionMethod::Summarized => {
                            saw_summarized = true;
                            let stats = summary
                                .as_ref()
                                .expect("a Summarized event must carry its request's cost");
                            assert!(stats.messages_summarized >= MIN_SUMMARIZED_SPAN);
                            // ModelConfig::mock has no configured rates.
                            assert!(stats.cost_usd.is_none());
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
        let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
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
            let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
                .with_retain_tail_tokens(200);
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
        let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
            .with_retain_tail_tokens(200);
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
        let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
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
                summary,
                tokens_after,
                ..
            } = event
            {
                if method == CompactionMethod::Summarized {
                    saw_summarized = true;
                    assert!(
                        summary.is_some_and(|s| s.messages_summarized > 0),
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
            let strategy = LlmCompaction::from_provider(provider, ModelConfig::mock())
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

    /// Returns a successful response carrying a non-`Stop` stop reason — the
    /// shape a truncated or refused briefing arrives in.
    struct StoppedProvider {
        reason: StopReason,
    }

    #[async_trait::async_trait]
    impl StreamProvider for StoppedProvider {
        async fn stream(
            &self,
            _config: StreamConfig,
            _tx: tokio::sync::mpsc::UnboundedSender<crate::provider::StreamEvent>,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Message, crate::provider::ProviderError> {
            Ok(Message::assistant(
                vec![Content::Text {
                    text: "## Goal\nTruncated mid-".into(),
                }],
                self.reason.clone(),
                "mock",
                "mock",
                Usage::default(),
            ))
        }
    }

    /// Never returns — stands in for a hung request with no provider timeout.
    struct HangingProvider;

    #[async_trait::async_trait]
    impl StreamProvider for HangingProvider {
        async fn stream(
            &self,
            _config: StreamConfig,
            _tx: tokio::sync::mpsc::UnboundedSender<crate::provider::StreamEvent>,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Message, crate::provider::ProviderError> {
            std::future::pending().await
        }
    }

    /// A truncated or refused briefing is non-empty text, so the old
    /// emptiness check let it through — and splicing *deletes* the span it
    /// replaces, making that permanent data loss.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_non_stop_summary_is_rejected_not_spliced() {
        for reason in [StopReason::Length, StopReason::Refusal, StopReason::Error] {
            let strategy = LlmCompaction::from_provider(
                Arc::new(StoppedProvider {
                    reason: reason.clone(),
                }),
                ModelConfig::mock(),
            )
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);
            let cfg = config(2_000);

            let run = drive(&strategy, &cfg, 15, |i| turn(i, 400)).await;
            assert_eq!(
                run.rounds_with_summary, 0,
                "{reason:?} must never be spliced into history"
            );
            assert!(
                lock(&strategy.state).phase.is_idle(),
                "{reason:?}: a rejected summary must leave the slot idle"
            );
        }
    }

    /// A hung request used to pin the in-flight slot for the life of the
    /// session, silently degrading the strategy to `DefaultCompaction`. The
    /// timeout plus the drop guard must return the slot to idle.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_hung_request_releases_the_in_flight_slot() {
        let strategy = LlmCompaction::from_provider(Arc::new(HangingProvider), ModelConfig::mock())
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200)
            .with_timeout(std::time::Duration::from_millis(50))
            .with_retry_config(crate::retry::RetryConfig::none());
        let cfg = config(2_000);

        let messages = history(30, 400);
        strategy.compact(messages.clone(), &cfg);
        for _ in 0..60 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if lock(&strategy.state).phase.is_idle() {
                break;
            }
        }
        assert!(
            lock(&strategy.state).phase.is_idle(),
            "a hung request must not pin the slot in flight forever"
        );
    }

    #[test]
    fn transcript_is_capped_in_total_not_just_per_block() {
        // A span far larger than any summarizer's window.
        let huge: Vec<AgentMessage> = (0..4_000).flat_map(|i| turn(i, 1_000)).collect();
        let transcript = serialize_transcript(&huge);
        // The cap is checked per message, so one message may overshoot it.
        let ceiling = TRANSCRIPT_TOTAL_BYTES + 4 * TRANSCRIPT_PER_BLOCK_BYTES;
        assert!(
            transcript.len() <= ceiling,
            "transcript must be bounded in total, got {} bytes (ceiling {ceiling})",
            transcript.len()
        );
        assert!(transcript.contains("transcript size cap"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_short_history_does_not_burn_the_inert_warning() {
        let strategy = mock("unused").with_trigger_ratio(0.1);
        let cfg = config(2_000);
        // Two messages: past the trigger, but nothing worth summarizing yet.
        let messages = history(1, 400);
        strategy.compact(messages, &cfg);
        assert!(
            !lock(&strategy.state).warned.inert,
            "a merely-short history is transient and must not spend the one-shot warning"
        );
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
        let strategy = LlmCompaction::from_provider(Arc::new(FailingProvider), ModelConfig::mock())
            .with_trigger_ratio(0.1)
            .with_retain_tail_tokens(200);

        let messages = history(30, 400);
        let cfg = config(2_000);
        let out = strategy.compact(messages.clone(), &cfg);
        assert!(total_tokens(&out) <= 2_000);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            lock(&strategy.state).phase.is_idle(),
            "a failed request must leave the slot idle, not pinned in flight"
        );
    }
}

#[cfg(test)]
mod losing_race_warning {
    use super::*;

    fn strategy() -> LlmCompaction {
        LlmCompaction::from_config(crate::provider::ModelConfig::mock())
    }

    /// The warning is one-shot, and only after a *streak*.
    ///
    /// A single fallback is ordinary — the first compaction of a session
    /// usually arrives before any summary could have been ready — so warning on
    /// it would train callers to ignore the message.
    #[test]
    fn one_fallback_is_quiet_and_the_warning_fires_once() {
        let s = strategy();
        s.warn_losing_race_once();
        assert!(
            !lock(&s.state).warned.losing_race,
            "a single fallback must not warn; the first compaction of a session \
             legitimately beats any summary"
        );

        s.warn_losing_race_once();
        assert!(
            lock(&s.state).warned.losing_race,
            "a streak of {FALLBACKS_BEFORE_WARNING} must warn — the session is paying \
             for briefings it never uses"
        );

        // One-shot: the flag stays set and the streak keeps counting, but the
        // caller is not told again every compaction for the rest of the run.
        let before = lock(&s.state).fallbacks;
        s.warn_losing_race_once();
        assert!(
            lock(&s.state).fallbacks > before,
            "the streak keeps counting"
        );
    }

    /// A successful splice resets the streak.
    ///
    /// Without this, a session that splices most of the time but falls back
    /// twice across an hour would still be told it is "losing the race", which
    /// is false and would send the reader tuning something that is working.
    ///
    /// **Scope:** this pins the streak *policy*, not the wiring. It sets
    /// `fallbacks = 0` directly rather than driving a real splice, so deleting
    /// the reset from `compact`'s success path would not fail it — exercising
    /// that needs a summary staged in `Phase::Ready` with a matching
    /// fingerprint. Stated because a test that reads stronger than it is, is
    /// worse than one that admits its limit.
    #[test]
    fn a_successful_splice_resets_the_streak() {
        let s = strategy();
        s.warn_losing_race_once();
        assert_eq!(lock(&s.state).fallbacks, 1);

        // What the splice path does on success.
        lock(&s.state).fallbacks = 0;

        s.warn_losing_race_once();
        assert!(
            !lock(&s.state).warned.losing_race,
            "one fallback after a successful splice is not a streak"
        );
    }
}
