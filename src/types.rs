use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

/// Content block of a message.
///
/// Exhaustiveness policy (two separate levers):
/// - The **enum** is `#[non_exhaustive]`: new content kinds may be added in
///   minor releases, so downstream `match` arms need a wildcard.
/// - The `ToolCall` and `Thinking` **variants** are separately
///   `#[non_exhaustive]`: their fields grow with provider features (PR #32
///   added `provider_metadata`), so downstream constructs them via the
///   `Content::tool_call*` / `Content::thinking*` constructors and uses `..`
///   in patterns. `Text` and `Image` stay literally constructible — they are
///   user-facing shapes that do not grow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "thinking")]
    #[non_exhaustive]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "toolCall")]
    #[non_exhaustive]
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
        /// Provider-specific metadata (e.g. Gemini thought signatures).
        /// Not passed to tool execution; used by providers when building
        /// the next request.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerMetadata",
            alias = "provider_metadata"
        )]
        provider_metadata: Option<serde_json::Value>,
    },
}

impl Content {
    /// Construct a tool-call content block.
    ///
    /// The `ToolCall` variant is `#[non_exhaustive]` so provider-specific
    /// fields can be added without breaking downstream crates — use this
    /// constructor instead of a struct literal.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
            provider_metadata: None,
        }
    }

    /// Construct a thinking content block without a signature.
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: text.into(),
            signature: None,
        }
    }

    /// Construct a thinking content block with a provider signature.
    pub fn thinking_signed(text: impl Into<String>, signature: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: text.into(),
            signature: Some(signature.into()),
        }
    }

    /// Construct a tool-call content block carrying provider metadata
    /// (e.g. a Gemini thought signature).
    pub fn tool_call_with_metadata(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
        provider_metadata: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
            provider_metadata: Some(provider_metadata),
        }
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        content: Vec<Content>,
        timestamp: u64,
    },
    #[serde(rename = "assistant")]
    #[non_exhaustive]
    Assistant {
        content: Vec<Content>,
        #[serde(rename = "stopReason")]
        stop_reason: StopReason,
        model: String,
        provider: String,
        usage: Usage,
        timestamp: u64,
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "errorMessage",
            alias = "error_message"
        )]
        error_message: Option<String>,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        content: Vec<Content>,
        #[serde(rename = "isError")]
        is_error: bool,
        timestamp: u64,
    },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![Content::Text { text: text.into() }],
            timestamp: now_ms(),
        }
    }

    /// Construct an assistant message.
    ///
    /// The `Assistant` variant is `#[non_exhaustive]` — its fields grow with
    /// provider features (`error_message` was itself a later addition), so
    /// custom `StreamProvider` implementations construct it here instead of
    /// with a struct literal. `timestamp` is set to now and `error_message`
    /// to `None`; use [`Message::with_error_message`] /
    /// [`Message::with_timestamp`] to override.
    pub fn assistant(
        content: Vec<Content>,
        stop_reason: StopReason,
        model: impl Into<String>,
        provider: impl Into<String>,
        usage: Usage,
    ) -> Self {
        Self::Assistant {
            content,
            stop_reason,
            model: model.into(),
            provider: provider.into(),
            usage,
            timestamp: now_ms(),
            error_message: None,
        }
    }

    /// Set the error message (no-op on non-assistant messages).
    pub fn with_error_message(mut self, msg: impl Into<String>) -> Self {
        if let Self::Assistant { error_message, .. } = &mut self {
            *error_message = Some(msg.into());
        }
        self
    }

    /// Override the timestamp (applies to all message kinds).
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        match &mut self {
            Self::User { timestamp, .. }
            | Self::Assistant { timestamp, .. }
            | Self::ToolResult { timestamp, .. } => *timestamp = ts,
        }
        self
    }

    pub fn role(&self) -> &str {
        match self {
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "toolResult",
        }
    }

    /// Check if this assistant message represents a context overflow error.
    ///
    /// Some providers (SSE-based: Anthropic, OpenAI) return overflow as a
    /// `StopReason::Error` message rather than an HTTP error. This method
    /// checks the `error_message` field against known overflow patterns.
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::Assistant {
                stop_reason: StopReason::Error,
                error_message: Some(msg),
                ..
            } => crate::provider::is_context_overflow_message(msg),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// AgentMessage — LLM messages + extensible custom types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionMessage {
    pub role: String,
    pub kind: String,
    pub data: serde_json::Value,
}

impl ExtensionMessage {
    pub fn new(kind: impl Into<String>, data: impl Serialize) -> Self {
        Self {
            role: "extension".into(),
            kind: kind.into(),
            data: serde_json::to_value(data).unwrap_or(serde_json::Value::Null),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
    /// Standard LLM message
    Llm(Message),
    /// App-specific message (UI-only, notifications, etc.)
    Extension(ExtensionMessage),
}

impl AgentMessage {
    pub fn role(&self) -> &str {
        match self {
            Self::Llm(m) => m.role(),
            Self::Extension(ext) => &ext.role,
        }
    }

    pub fn as_llm(&self) -> Option<&Message> {
        match self {
            Self::Llm(m) => Some(m),
            Self::Extension(_) => None,
        }
    }
}

impl From<Message> for AgentMessage {
    fn from(m: Message) -> Self {
        Self::Llm(m)
    }
}

// ---------------------------------------------------------------------------
// Stop reasons & usage
// ---------------------------------------------------------------------------

/// Why the model stopped generating.
///
/// `#[non_exhaustive]` since 0.17.0: stop reasons grow with provider features,
/// so every addition was otherwise a breaking release. Downstream `match` arms
/// need a `_ =>` wildcard; inside this crate the enum is still exhaustive, so
/// adding a variant remains a compile error where it matters most.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    /// The provider's safety system declined the request. The stream completes
    /// normally (HTTP 200) but with `stop_reason: refusal` and empty or partial
    /// content; `error_message` carries an explanation. Currently emitted by
    /// Anthropic models that support the `refusal` stop reason (e.g. Claude
    /// Fable 5). The agent loop does not special-case it (the turn ends like a
    /// normal `Stop`); callers can match on it to retry on a fallback model.
    Refusal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    // camelCase on the wire (AgentEvent contract); `alias` keeps session
    // files written by yoagent < 0.13 loadable.
    #[serde(default, rename = "cacheRead", alias = "cache_read")]
    pub cache_read: u64,
    #[serde(default, rename = "cacheWrite", alias = "cache_write")]
    pub cache_write: u64,
    #[serde(default, rename = "totalTokens", alias = "total_tokens")]
    pub total_tokens: u64,
}

impl Usage {
    /// Fraction of input tokens served from cache (0.0–1.0).
    /// Returns 0.0 if no input tokens were processed.
    pub fn cache_hit_rate(&self) -> f64 {
        let total_input = self.input + self.cache_read + self.cache_write;
        if total_input == 0 {
            return 0.0;
        }
        self.cache_read as f64 / total_input as f64
    }
}

// ---------------------------------------------------------------------------
// Cache configuration
// ---------------------------------------------------------------------------

/// Controls yoagent-managed prompt caching hints.
///
/// By default, caching is enabled with automatic breakpoint placement. What
/// that produces on the wire depends on the protocol — see [`CacheStrategy`]
/// for the per-provider table. `enabled: false` suppresses every hint yoagent
/// would otherwise send; it cannot switch off a provider's *automatic*
/// server-side caching, which is not under client control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CacheConfig {
    /// Master switch — set to false to disable all caching hints.
    /// Default: true.
    pub enabled: bool,
    /// How cache breakpoints are placed.
    pub strategy: CacheStrategy,
    /// Stable identifier for this conversation, for providers that route cache
    /// lookups by key rather than by explicit breakpoints (OpenAI's
    /// `prompt_cache_key`).
    ///
    /// Leave `None` and one is derived from the request's stable head — the
    /// system prompt plus the first user message. That derivation is correct
    /// for the common case and costs nothing, but two sessions opening with
    /// identical text share a key. Set this explicitly when sessions must be
    /// routed apart, or when the head is not distinctive.
    ///
    /// Ignored everywhere except the key-routed path — by Anthropic, which
    /// takes explicit breakpoints, and by Google, Vertex, Bedrock, Azure and
    /// Responses, which yoagent sends no cache hints to at all.
    ///
    /// Note for [`crate::SubAgentTool`]: it holds one `CacheConfig` and clones
    /// it into every invocation, so a key set there is shared by every run the
    /// tool performs rather than identifying one conversation.
    #[serde(default)]
    pub session_key: Option<String>,
}

impl CacheConfig {
    /// Caching enabled with automatic breakpoint placement.
    ///
    /// Identical to [`Default::default`]; provided because this struct is
    /// `#[non_exhaustive]` and `new()` is where callers look first.
    pub fn new() -> Self {
        Self::default()
    }

    /// All caching hints suppressed.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Set the session key used by key-routed providers.
    ///
    /// A blank key is treated as unset — sending `prompt_cache_key: ""` would
    /// route every caller who did that onto one cache, which is worse than
    /// sending nothing. `with_session_key(format!("tenant-{id}"))` with an
    /// empty `id` is the ordinary way to arrive here.
    pub fn with_session_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        self.session_key = if key.trim().is_empty() {
            None
        } else {
            Some(key)
        };
        self
    }

    /// Set the breakpoint-placement strategy.
    ///
    /// Needed because this struct is `#[non_exhaustive]`: downstream crates
    /// cannot use a struct literal, so `Manual { .. }` would otherwise be
    /// unreachable from outside.
    pub fn with_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Whether yoagent should emit any caching hint at all.
    ///
    /// Three configurations mean "no": `enabled: false`, `Disabled`, and
    /// `Manual` with every flag off. The third is easy to miss — Anthropic
    /// honours it correctly by placing no breakpoints, so a key-routed
    /// provider that still sent a key would be reading the same value as
    /// "yes". One predicate, so every protocol agrees on what off means.
    pub fn hints_enabled(&self) -> bool {
        if !self.enabled {
            return false;
        }
        !matches!(
            self.strategy,
            CacheStrategy::Disabled
                | CacheStrategy::Manual {
                    cache_system: false,
                    cache_tools: false,
                    cache_messages: false,
                }
        )
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: CacheStrategy::Auto,
            session_key: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution strategy
// ---------------------------------------------------------------------------

/// Controls how multiple tool calls from a single LLM response are executed.
///
/// When the LLM returns multiple tool calls (e.g., "read file A, read file B,
/// run bash C"), this determines whether they run sequentially or in parallel.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolExecutionStrategy {
    /// Run tools one at a time, check steering between each.
    /// Use for debugging or tools with shared mutable state.
    Sequential,
    /// Run all tool calls concurrently, check steering after all complete.
    /// Default — most tool calls are independent and this gives the best latency.
    #[default]
    Parallel,
    /// Run in batches of N, check steering between batches.
    /// Balances speed with human-in-the-loop control.
    Batched { size: usize },
}

/// Strategy for prompt caching.
///
/// Providers expose caching in two different shapes, and this enum means
/// something different in each:
///
/// | provider | shape | what this enum controls |
/// |---|---|---|
/// | Anthropic | explicit breakpoints | where `cache_control` markers are placed |
/// | OpenAI (native) | key-routed | whether `prompt_cache_key` is sent |
/// | DeepSeek, Gemini, and other automatic backends | automatic, server-side | nothing |
/// | Azure, OpenAI Responses, Bedrock | supported but **not yet wired** | nothing *yet* |
///
/// **Explicit breakpoints** (Anthropic) are the model this enum was designed
/// around: the client chooses cache boundaries and pays a write premium for
/// them. [`Auto`](Self::Auto) and [`Manual`](Self::Manual) select which
/// boundaries.
///
/// **Key-routed** (OpenAI) caches automatically on prefixes of ~1024 tokens or
/// more; there are no breakpoints to place. `prompt_cache_key` only improves
/// *routing* — it steers requests from one conversation toward the same cache
/// — so the `Auto`/`Manual` distinction has nothing to act on and both send the
/// key. Only [`Disabled`](Self::Disabled) is meaningful. The key comes from
/// [`CacheConfig::session_key`], or is derived from the request head. Gated on
/// [`OpenAiCompat::supports_prompt_cache_key`](crate::provider::OpenAiCompat),
/// because the field is OpenAI's and a strict compat server may reject unknown
/// keys outright rather than ignore them.
///
/// **Automatic, server-side** (DeepSeek, Gemini) caches on its own with nothing
/// to configure. DeepSeek quantises hits to 64-token blocks and charges nothing
/// to populate; Gemini reports `cachedContentTokenCount` but its explicit
/// cached-content API is a separate create-then-reference resource with its own
/// TTL and billing, deliberately not driven from here. For these, this setting
/// is inert — including `Disabled`, which cannot switch off caching the client
/// never asked for.
///
/// **Not yet wired** is a separate row on purpose. Azure and the OpenAI
/// Responses API both accept `prompt_cache_key`, and Bedrock accepts explicit
/// `cachePoint` blocks; yoagent sends none of them today. That is a gap in this
/// crate, not a property of those vendors, and conflating the two would make
/// the omission read as a deliberate design decision.
///
/// Two practical consequences. A hit rate is not comparable across protocols
/// without knowing which shape produced it. And only Anthropic, the
/// OpenAI-compat path and Gemini populate [`Usage::cache_read`] at all — on
/// Azure, Responses and Bedrock there is no hit rate to read. See
/// `docs/concepts/prompt-caching.md`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum CacheStrategy {
    /// Automatic placement (recommended).
    ///
    /// Anthropic: caches system prompt, tool definitions, and recent history.
    /// OpenAI: sends `prompt_cache_key`. Elsewhere: no effect.
    #[default]
    Auto,
    /// Send no caching hints at all.
    ///
    /// Does **not** disable a provider's automatic server-side caching — that
    /// is not client-controllable. On Anthropic this means paying full input
    /// price for every prefix, so reach for it only when a rewrite-heavy
    /// workload makes cache writes pure loss.
    Disabled,
    /// Fine-grained control over what gets cached.
    ///
    /// Anthropic-only in effect: no other protocol exposes placement. Treated
    /// as [`Auto`](Self::Auto) by key-routed providers, which have one knob
    /// rather than three.
    Manual {
        /// Cache the system prompt.
        cache_system: bool,
        /// Cache tool definitions.
        cache_tools: bool,
        /// Cache conversation history (second-to-last message).
        cache_messages: bool,
    },
}

// ---------------------------------------------------------------------------
// Thinking level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

/// Callback for streaming partial results during tool execution.
///
/// Tools call this to emit progress updates (e.g., partial output, status messages)
/// that are forwarded as `AgentEvent::ToolExecutionUpdate` events for UI consumption.
/// Partial results are **not** sent to the LLM — only the final `ToolResult` is.
pub type ToolUpdateFn = Arc<dyn Fn(ToolResult) + Send + Sync>;

/// Callback for emitting user-facing progress messages during tool execution.
///
/// Each invocation emits an `AgentEvent::ProgressMessage` event. Unlike `ToolUpdateFn`,
/// these are simple text messages intended for user-facing display (e.g., status lines,
/// notifications), not structured tool results.
pub type ProgressFn = Arc<dyn Fn(String) + Send + Sync>;

/// Context passed to tool execution. Bundles all per-invocation state.
///
/// Using a struct instead of individual parameters future-proofs the trait —
/// adding fields to `ToolContext` is non-breaking, which `#[non_exhaustive]`
/// is what actually makes true. Tools receive this rather than build it, so
/// the attribute costs implementors nothing; it only stops a struct literal in
/// downstream test code from breaking on every new field.
#[non_exhaustive]
pub struct ToolContext {
    /// The ID of this tool call (for correlation).
    pub tool_call_id: String,
    /// The name of the tool being invoked.
    pub tool_name: String,
    /// Cancellation token — check `is_cancelled()` in long-running tools.
    pub cancel: tokio_util::sync::CancellationToken,
    /// Optional callback for streaming partial `ToolResult`s (UI/logging only).
    pub on_update: Option<ToolUpdateFn>,
    /// Optional callback for emitting user-facing progress messages.
    pub on_progress: Option<ProgressFn>,
}

impl ToolContext {
    /// A context for one tool invocation, with no cancellation token and no
    /// callbacks.
    ///
    /// This struct is `#[non_exhaustive]`, so downstream crates build it here
    /// rather than with a struct literal. The loop constructs the real one;
    /// this is for tests and for callers driving a tool directly.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            cancel: tokio_util::sync::CancellationToken::new(),
            on_update: None,
            on_progress: None,
        }
    }

    /// Use the given cancellation token instead of a fresh one.
    pub fn with_cancel(mut self, cancel: tokio_util::sync::CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Stream partial [`ToolResult`]s to this callback (UI/logging only).
    pub fn with_on_update(mut self, on_update: ToolUpdateFn) -> Self {
        self.on_update = Some(on_update);
        self
    }

    /// Emit user-facing progress messages through this callback.
    pub fn with_on_progress(mut self, on_progress: ProgressFn) -> Self {
        self.on_progress = Some(on_progress);
        self
    }
}

impl Clone for ToolContext {
    fn clone(&self) -> Self {
        Self {
            tool_call_id: self.tool_call_id.clone(),
            tool_name: self.tool_name.clone(),
            cancel: self.cancel.clone(),
            on_update: self.on_update.clone(),
            on_progress: self.on_progress.clone(),
        }
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("tool_call_id", &self.tool_call_id)
            .field("tool_name", &self.tool_name)
            .field("cancel", &self.cancel)
            .field("on_update", &self.on_update.as_ref().map(|_| "<callback>"))
            .field(
                "on_progress",
                &self.on_progress.as_ref().map(|_| "<callback>"),
            )
            .finish()
    }
}

/// A tool the agent can call. Implement this trait for your tools.
#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    /// Unique tool name (used in LLM tool_use)
    fn name(&self) -> &str;
    /// Human-readable label for UI
    fn label(&self) -> &str;
    /// Description for the LLM
    fn description(&self) -> &str;
    /// JSON Schema for parameters
    fn parameters_schema(&self) -> serde_json::Value;
    /// Execute the tool.
    ///
    /// The `ctx` parameter provides per-invocation context:
    /// - `ctx.tool_call_id` / `ctx.tool_name` — for correlation and logging
    /// - `ctx.cancel` — cancellation token; check `is_cancelled()` in long-running tools
    /// - `ctx.on_update` — optional callback for streaming partial `ToolResult`s (UI/logging only)
    /// - `ctx.on_progress` — optional callback for user-facing progress text (`ProgressMessage`)
    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<Content>,
    #[serde(default)]
    pub details: serde_json::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("{0}")]
    Failed(String),
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("Cancelled")]
    Cancelled,
}

// ---------------------------------------------------------------------------
// Agent events (for streaming UI updates)
// ---------------------------------------------------------------------------

/// Events emitted by the agent loop for streaming UI updates.
///
/// # Wire format (stability contract)
///
/// `AgentEvent` and [`StreamDelta`] serialize as internally-tagged JSON —
/// `{"type": "<camelCase variant>", ...camelCase fields}` — so external
/// frontends (websocket fanout servers, TypeScript clients, JSONL pipes) can
/// consume the event stream directly:
///
/// ```json
/// {"type":"messageUpdate","message":{...},"delta":{"type":"text","delta":"hi"}}
/// {"type":"toolExecutionEnd","toolCallId":"tc_1","toolName":"bash","result":{...},"isError":false}
/// ```
///
/// This shape is a **public contract**: variant tags, field names, and the
/// internal tagging are frozen by snapshot tests. Renaming a variant or field
/// is a breaking change for wire clients, not just for Rust callers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[non_exhaustive]
pub enum AgentEvent {
    AgentStart,
    /// The run finished. Carries the messages it produced and a
    /// [`SessionStats`] rollup of what they cost.
    ///
    /// The variant is `#[non_exhaustive]` — the payload is expected to grow.
    /// Match with `..`.
    #[non_exhaustive]
    AgentEnd {
        messages: Vec<AgentMessage>,
        /// `#[serde(default)]`: `AgentEvent` is a frozen wire format, and
        /// archived streams predate this field.
        #[serde(default)]
        stats: SessionStats,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<Message>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        delta: StreamDelta,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        partial_result: ToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: ToolResult,
        is_error: bool,
    },
    ProgressMessage {
        tool_call_id: String,
        tool_name: String,
        text: String,
    },
    InputRejected {
        reason: String,
    },
    /// History was compacted before a turn.
    ///
    /// Emitted by [`LlmCompaction`](crate::LlmCompaction) on both of its paths
    /// — the spliced summary and the deterministic fallback — so a consumer can
    /// tell which one ran and what it cost. The built-in
    /// [`DefaultCompaction`](crate::context::DefaultCompaction) does not emit
    /// this; it has no event channel and never issues a request.
    /// A tool was called repeatedly with identical arguments.
    ///
    /// Emitted on both escalations: the first trip steers the model and
    /// continues, a later trip on the same signature stops the run. `aborted`
    /// distinguishes them, so a caller can tell a nudge from a stop and an
    /// audit can record why a run ended.
    #[non_exhaustive]
    LoopDetected {
        tool_name: String,
        repetitions: usize,
        aborted: bool,
    },
    ContextCompacted {
        /// Which compaction path produced this result.
        method: CompactionMethod,
        messages_before: usize,
        messages_after: usize,
        tokens_before: usize,
        tokens_after: usize,
        /// What the summarization request produced and cost, when one was
        /// made. `None` on a purely deterministic compaction.
        ///
        /// Present as one optional payload rather than three sibling fields so
        /// the cost, the span it bought, and the fact that a request happened
        /// cannot disagree with each other.
        summary: Option<SummaryStats>,
    },
}

impl AgentEvent {
    /// Construct an [`AgentEvent::AgentEnd`].
    ///
    /// The variant is `#[non_exhaustive]` — its payload grows — so downstream
    /// crates (and tests) build it here rather than with a struct literal.
    pub fn agent_end(messages: Vec<AgentMessage>, stats: SessionStats) -> Self {
        Self::AgentEnd { messages, stats }
    }

    /// Construct an [`AgentEvent::LoopDetected`].
    ///
    /// `aborted` separates the two escalations: `false` is a steer the model
    /// can recover from, `true` means the run stopped.
    pub fn loop_detected(tool_name: impl Into<String>, repetitions: usize, aborted: bool) -> Self {
        Self::LoopDetected {
            tool_name: tool_name.into(),
            repetitions,
            aborted,
        }
    }
}

/// Session-level rollup carried by [`AgentEvent::AgentEnd`].
///
/// The per-turn numbers already existed — `Usage` on every assistant message,
/// `tokens_cached` on the `llm_stream` span, `cache_read` in the GASP record —
/// but nothing summed them, so answering "what was this run's cache hit rate"
/// meant replaying the whole event stream. Any change that moves caching
/// (breakpoint placement, compaction strategy, model choice) had to be judged
/// by hand-built harnesses instead of a number the library reports.
///
/// ```
/// # use yoagent::{SessionStats, Usage};
/// # let stats = SessionStats::default();
/// // Reading your cache hit rate:
/// println!("{:.1}% cached over {} turns", stats.cache_hit_rate() * 100.0, stats.turns);
/// ```
///
/// **Hit rate is `cache_read / (input + cache_read + cache_write)`** — cache
/// writes count against you, because they are prompt tokens the provider
/// processed and billed. Counting only `input` shrinks the denominator and so
/// **overstates** the rate for a write-charging provider: Anthropic books a
/// re-processed prefix to `cache_write`, and an `input`-only metric makes it
/// look roughly ten times cheaper than it is. See
/// `docs/evals/llm-compaction-live.md`, where that error was made and caught.
///
/// Read a rate against its session length, not against 100%: every turn's new
/// content is necessarily a miss, so the ceiling is about `(n-1)/(n+1)` — ~88%
/// at 15 turns, ~96% only past 49.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SessionStats {
    /// Provider usage summed over every LLM turn in this run.
    ///
    /// `total_tokens` is **not** summed and stays 0 — see [`record_turn`] for
    /// why. Derive a total from the four components instead.
    ///
    /// [`record_turn`]: SessionStats::record_turn
    #[serde(default)]
    pub usage: Usage,
    /// LLM turns taken. Tool executions are not turns; an errored turn counts,
    /// because the provider billed it, and provider retries within one turn do
    /// not appear separately.
    #[serde(default)]
    pub turns: u32,
    /// Dollar cost of [`usage`](Self::usage), when the model's rates are
    /// configured (see [`CostConfig`](crate::provider::CostConfig)).
    ///
    /// `None` means "cannot price this", never "free" — all-zero rates mean
    /// pricing is unknown, which is the case for custom and local models.
    ///
    /// Scope: this run's own turns. A [`SubAgentTool`](crate::SubAgentTool)
    /// runs its own loop on a private channel, so a delegating agent's real
    /// spend is higher than this reports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Times the loop observed compaction rewrite history.
    ///
    /// Counts any turn where the strategy returned a different message count or
    /// a different *estimated* token total ([`context::total_tokens`]), so
    /// in-place tool-output truncation is included alongside reshaping. A
    /// compaction that runs and reclaims nothing is indistinguishable from no
    /// compaction at all.
    ///
    /// [`context::total_tokens`]: crate::context::total_tokens
    ///
    /// Deliberately **not** split into spliced-summary vs deterministic
    /// fallback, and carrying no summarization spend. That detail exists — on
    /// [`AgentEvent::ContextCompacted`], with its [`SummaryStats`] — but
    /// [`CompactionStrategy::compact`](crate::context::CompactionStrategy::compact)
    /// is synchronous and has no event channel, so the loop cannot see it. Wire
    /// `LlmCompaction::with_event_sender` to the same channel and aggregate the
    /// two together — the events *describe* the same compactions this counts,
    /// with the breakdown attached, so do not sum the two. Folding a guess in
    /// here would be worse than the gap.
    #[serde(default)]
    pub compactions: u32,
}

impl SessionStats {
    /// A rollup with the given figures.
    ///
    /// This struct is `#[non_exhaustive]`, so downstream crates cannot use a
    /// struct literal; without this the only construction path would be
    /// `Default::default()` plus field assignment.
    pub fn new(usage: Usage, turns: u32, cost_usd: Option<f64>, compactions: u32) -> Self {
        Self {
            usage,
            turns,
            cost_usd,
            compactions,
        }
    }

    /// Fraction of prompt tokens served from cache across the whole session
    /// (0.0–1.0). Delegates to [`Usage::cache_hit_rate`] so there is one
    /// definition of the metric rather than two that can drift.
    pub fn cache_hit_rate(&self) -> f64 {
        self.usage.cache_hit_rate()
    }

    /// Fold one turn's usage into the rollup, costing it when rates are known.
    ///
    /// `total_tokens` is deliberately not summed. It is a per-response provider
    /// report, and the providers disagree on it: `anthropic.rs` never sets it
    /// at all, `bedrock.rs` computes `input + output` and so excludes cache,
    /// and the rest pass through a payload value that includes cached tokens.
    /// Summing it would launder that inconsistency into a session-level number
    /// that reads as authoritative and is 0 for every Anthropic run. The four
    /// components sum cleanly; derive a total from those.
    ///
    /// Cost accrues per turn rather than once at the end. Today that is
    /// arithmetically identical — `CostConfig::cost_usd` is linear in every
    /// `Usage` field and `config.model_config` is fixed for the life of a run
    /// (`run_loop` holds `&AgentLoopConfig`, and `set_model` needs `&mut self`).
    /// It is written this way so a per-turn model override stays correct if one
    /// is ever introduced, and so `cost_usd` reflects whether any turn was
    /// priceable rather than requiring a separate check.
    pub(crate) fn record_turn(
        &mut self,
        usage: &Usage,
        cost: Option<&crate::provider::CostConfig>,
    ) {
        self.usage.input += usage.input;
        self.usage.output += usage.output;
        self.usage.cache_read += usage.cache_read;
        self.usage.cache_write += usage.cache_write;
        self.turns += 1;

        if let Some(cost) = cost.filter(|c| c.is_configured()) {
            *self.cost_usd.get_or_insert(0.0) += cost.cost_usd(usage);
        }
    }
}

/// What a summarization request produced, carried by
/// [`AgentEvent::ContextCompacted`].
///
/// Weigh [`usage`](Self::usage) against the event's `tokens_before -
/// tokens_after` to decide whether an LLM compaction strategy earns its keep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SummaryStats {
    /// Messages the briefing replaced.
    ///
    /// Zero when a briefing was produced but could not be kept — the request
    /// was still paid for, so the event still reports it, but `method` will be
    /// [`CompactionMethod::Deterministic`].
    pub messages_summarized: usize,
    /// Tokens the summarization request itself consumed.
    pub usage: Usage,
    /// Dollar cost of `usage`, when the summarization model's rates are
    /// configured (see [`CostConfig`](crate::provider::CostConfig)).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

impl SummaryStats {
    /// A record of one summarization request.
    pub fn new(messages_summarized: usize, usage: Usage, cost_usd: Option<f64>) -> Self {
        Self {
            messages_summarized,
            usage,
            cost_usd,
        }
    }
}

/// Which compaction path produced an [`AgentEvent::ContextCompacted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum CompactionMethod {
    /// History was replaced by an LLM-written summary.
    Summarized,
    /// Deterministic tiered compaction ran: truncate → summarize → drop.
    ///
    /// On [`LlmCompaction`](crate::LlmCompaction) this means no briefing made
    /// it into the result — none was ready, one was discarded as stale, or one
    /// was produced but could not be kept within the budget. The loop stayed
    /// unblocked; the compaction was lossy. Check `summary` to tell a free
    /// fallback from one that still paid for a request.
    Deterministic,
}

/// Incremental content delta carried by [`AgentEvent::MessageUpdate`].
///
/// Serializes internally tagged (`{"type":"text","delta":"..."}`); see the
/// wire-format contract on [`AgentEvent`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[non_exhaustive]
pub enum StreamDelta {
    Text { delta: String },
    Thinking { delta: String },
    ToolCallDelta { delta: String },
}

// ---------------------------------------------------------------------------
// Agent context (passed to the loop)
// ---------------------------------------------------------------------------

pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Box<dyn AgentTool>>,
}

// ---------------------------------------------------------------------------
// Input filtering
// ---------------------------------------------------------------------------

/// Result of applying an input filter to a user message.
#[derive(Debug, Clone)]
pub enum FilterResult {
    /// Message passes unchanged.
    Pass,
    /// Message passes, but append a warning to context for the LLM to see.
    Warn(String),
    /// Message is rejected. Agent loop returns immediately.
    Reject(String),
}

/// Synchronous filter applied to user input before the LLM call.
///
/// Implement this for injection detection, content moderation, PII redaction, etc.
/// Filters run in the hot path and must be fast — use `before_turn` callbacks
/// for async moderation (external API calls).
pub trait InputFilter: Send + Sync {
    fn filter(&self, text: &str) -> FilterResult;
}

// ---------------------------------------------------------------------------
// Tool middleware (permissions)
// ---------------------------------------------------------------------------

/// Decision returned by a [`ToolMiddleware`] before a tool executes.
///
/// Deliberately NOT `#[non_exhaustive]` (same policy as [`StopReason`]):
/// this is a control-flow enum — a new variant should be a compile error for
/// matchers, not a silent wildcard fallthrough. Interactive flows like
/// "ask the user" need no variant: the hook is `async`, so prompt inside the
/// middleware and return `Allow`/`Deny`.
#[derive(Debug, Clone)]
pub enum ToolDecision {
    /// Execute the tool with the current arguments.
    Allow,
    /// Execute the tool with replacement arguments (e.g. a sandboxed path).
    Modify(serde_json::Value),
    /// Block the call. The reason is returned to the LLM as an error tool
    /// result so it can adapt (pick another tool, ask the user, ...); the
    /// loop itself continues.
    Deny(String),
}

/// Async hook that gates every tool call — the mechanism behind permission
/// prompts, policy engines, and argument rewriting.
///
/// yoagent ships the mechanism, not a policy: install middleware via
/// [`Agent::with_tool_middleware`](crate::Agent::with_tool_middleware) (or
/// [`AgentLoopConfig::tool_middleware`](crate::agent_loop::AgentLoopConfig))
/// and decide per call. Middleware run in a chain: each may rewrite the
/// arguments seen by later ones; the first `Deny` wins. With no middleware
/// installed, every call is allowed — behavior is unchanged.
///
/// The hook is `async` so an interactive app can prompt a human. Under the
/// default [`ToolExecutionStrategy::Parallel`], middleware for parallel tool
/// calls runs concurrently — serialize approval prompts inside your
/// implementation (or use `Sequential`) if you need one-at-a-time UX.
/// Borrowed view of a pending tool call, passed to
/// [`ToolMiddleware::before_tool`].
///
/// Marked `#[non_exhaustive]`: fields may be added in minor releases (turn
/// number, history access, ...) without breaking middleware implementations.
/// Constructed by the loop; middleware only reads it.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolCallRequest<'a> {
    /// Provider-assigned id of this tool call.
    pub tool_call_id: &'a str,
    /// Name of the tool the model wants to run.
    pub tool_name: &'a str,
    /// Arguments as the model provided them (possibly rewritten by earlier
    /// middleware in the chain).
    pub args: &'a serde_json::Value,
}

#[async_trait::async_trait]
pub trait ToolMiddleware: Send + Sync {
    async fn before_tool(&self, call: &ToolCallRequest<'_>) -> ToolDecision;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ToolUse => write!(f, "toolUse"),
            Self::Error => write!(f, "error"),
            Self::Aborted => write!(f, "aborted"),
            Self::Refusal => write!(f, "refusal"),
        }
    }
}

/// Freezes the serde wire contract for `AgentEvent` and `StreamDelta`.
///
/// Both enums are documented as a stable wire format for websocket fanout
/// servers, TypeScript clients and JSONL pipes, so a changed tag, a changed
/// payload shape, or a variant that quietly serializes however serde happens
/// to derive it are all breaking changes for consumers who never rebuild
/// against this crate.
///
/// This lives in the defining crate on purpose. Both enums are
/// `#[non_exhaustive]`, so an integration test *cannot* match them
/// exhaustively — its match needs a `_` arm, and a wildcard turns "adding a
/// variant fails to compile" into "adding a variant is silently untested".
/// Inside this crate exhaustiveness still applies.
#[cfg(test)]
mod wire_tag_freeze {
    use super::*;
    use std::collections::BTreeSet;

    /// Declares the frozen tag **and** a sample value for every variant of a
    /// `#[serde(tag = "type")]` enum, from a single list.
    ///
    /// This is the fix for the class of bug that made the old guard useless: a
    /// hand-written sample list and a hand-written variant count could never
    /// notice a *new* variant, because a new variant appears in neither. Here
    /// the generated match has no wildcard, so adding a variant fails to
    /// compile — and the only way to fix that is to add a line below, which
    /// supplies the sample in the same breath. The two lists cannot drift
    /// because they are one list.
    ///
    /// Three further mistakes are caught by the compiler rather than by luck:
    /// a duplicated pattern makes the later arm `unreachable_pattern` (an
    /// error under CI's `-Dwarnings`), a sample of the wrong type does not
    /// compile, and a missing arm is a non-exhaustive match.
    ///
    /// What the *tests* below add, which no macro can: that the declared tag
    /// is the one serde actually emits, that each sample survives a
    /// round-trip, and that a sample on the wrong line is caught (via the
    /// distinctness check).
    macro_rules! wire_freeze {
        ($ty:ty, $tag_of:ident, $samples:ident, $($pat:pat => $tag:literal = $sample:expr),+ $(,)?) => {
            /// The frozen `"type"` tag per variant. Changing one breaks every
            /// deployed wire client — do not edit casually.
            fn $tag_of(v: &$ty) -> &'static str {
                match v { $($pat => $tag,)+ }
            }

            /// One sample per variant, positionally paired with the tags above.
            fn $samples() -> Vec<$ty> { vec![$($sample,)+] }
        };
    }

    fn msg() -> AgentMessage {
        AgentMessage::Llm(Message::Assistant {
            content: vec![Content::Text { text: "hi".into() }],
            stop_reason: StopReason::Stop,
            model: "mock".into(),
            provider: "mock".into(),
            usage: Usage::default(),
            timestamp: 7,
            error_message: None,
        })
    }

    fn tool_result() -> ToolResult {
        ToolResult {
            content: vec![Content::Text { text: "ok".into() }],
            details: serde_json::Value::Null,
        }
    }

    wire_freeze! {
        AgentEvent, expected_event_tag, event_samples,
        AgentEvent::AgentStart => "agentStart" = AgentEvent::AgentStart,
        AgentEvent::AgentEnd { .. } => "agentEnd"
            = AgentEvent::agent_end(vec![msg()], SessionStats::default()),
        AgentEvent::TurnStart => "turnStart" = AgentEvent::TurnStart,
        AgentEvent::TurnEnd { .. } => "turnEnd" = AgentEvent::TurnEnd {
            message: msg(),
            tool_results: vec![],
        },
        AgentEvent::MessageStart { .. } => "messageStart"
            = AgentEvent::MessageStart { message: msg() },
        AgentEvent::MessageUpdate { .. } => "messageUpdate" = AgentEvent::MessageUpdate {
            message: msg(),
            delta: StreamDelta::Text { delta: "hi".into() },
        },
        AgentEvent::MessageEnd { .. } => "messageEnd"
            = AgentEvent::MessageEnd { message: msg() },
        AgentEvent::ToolExecutionStart { .. } => "toolExecutionStart"
            = AgentEvent::ToolExecutionStart {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                args: serde_json::json!({"command": "ls"}),
            },
        AgentEvent::ToolExecutionUpdate { .. } => "toolExecutionUpdate"
            = AgentEvent::ToolExecutionUpdate {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                partial_result: tool_result(),
            },
        AgentEvent::ToolExecutionEnd { .. } => "toolExecutionEnd"
            = AgentEvent::ToolExecutionEnd {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                result: tool_result(),
                is_error: false,
            },
        AgentEvent::ProgressMessage { .. } => "progressMessage"
            = AgentEvent::ProgressMessage {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                text: "50% done".into(),
            },
        AgentEvent::InputRejected { .. } => "inputRejected"
            = AgentEvent::InputRejected { reason: "injection detected".into() },
        AgentEvent::LoopDetected { .. } => "loopDetected"
            = AgentEvent::loop_detected("bash", 3, false),
        AgentEvent::ContextCompacted { .. } => "contextCompacted"
            = AgentEvent::ContextCompacted {
                method: CompactionMethod::Summarized,
                messages_before: 40,
                messages_after: 13,
                tokens_before: 96_500,
                tokens_after: 41_200,
                summary: None,
            },
    }

    wire_freeze! {
        StreamDelta, expected_delta_tag, delta_samples,
        StreamDelta::Text { .. } => "text" = StreamDelta::Text { delta: "hi".into() },
        StreamDelta::Thinking { .. } => "thinking"
            = StreamDelta::Thinking { delta: "hmm".into() },
        StreamDelta::ToolCallDelta { .. } => "toolCallDelta"
            = StreamDelta::ToolCallDelta { delta: "{}".into() },
    }

    /// Asserts the frozen contract for one sample: declared tag == emitted tag,
    /// payload keys are camelCase, and the value survives a round-trip.
    ///
    /// `seen` collects tags so a sample paired with the wrong pattern is caught
    /// — that is the one error the macro cannot catch, since both lines
    /// compile fine and the mismatch only shows as a repeated tag.
    fn assert_frozen<T>(
        sample: &T,
        declared: &str,
        seen: &mut BTreeSet<&'static str>,
        declared_static: &'static str,
    ) where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let v = serde_json::to_value(sample).expect("serialize");

        assert_eq!(
            v["type"], declared,
            "wire tag drifted: {sample:?} serializes as {} but wire_freeze! declares {declared}. \
             Changing a tag breaks every deployed client — if this is intentional it is a \
             breaking change, not a test fix",
            v["type"]
        );

        for key in v
            .as_object()
            .expect("tagged enums serialize as objects")
            .keys()
        {
            assert!(
                !key.contains('_'),
                "payload key {key:?} on {declared} is not camelCase — the enum carries \
                 rename_all_fields = \"camelCase\" and TS clients hardcode these names"
            );
        }

        let back: T = serde_json::from_value(v).expect("round-trip deserialize");
        assert_eq!(
            &back, sample,
            "{declared} did not survive a JSON round-trip"
        );

        assert!(
            seen.insert(declared_static),
            "two samples serialize as {declared} — a sample in wire_freeze! does not match \
             the pattern on its own line"
        );
    }

    #[test]
    fn every_event_variant_is_frozen_tagged_and_round_trips() {
        let samples = event_samples();
        let mut seen = BTreeSet::new();
        for sample in &samples {
            let declared = expected_event_tag(sample);
            assert_frozen(sample, declared, &mut seen, declared);
        }
        assert_eq!(
            seen.len(),
            samples.len(),
            "every variant must contribute a distinct tag"
        );
    }

    #[test]
    fn every_delta_variant_is_frozen_tagged_and_round_trips() {
        let samples = delta_samples();
        let mut seen = BTreeSet::new();
        for sample in &samples {
            let declared = expected_delta_tag(sample);
            assert_frozen(sample, declared, &mut seen, declared);
        }
        assert_eq!(
            seen.len(),
            samples.len(),
            "every variant must contribute a distinct tag"
        );
    }
}
