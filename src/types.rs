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
/// Deliberately NOT `#[non_exhaustive]`: this is a control-flow enum, and a
/// new variant (like `Refusal`, added for the 0.9.0 breaking release)
/// should be a compile error for
/// downstream matches rather than silently falling into a wildcard arm.
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

/// Controls yoagent-managed prompt caching hints for providers that support
/// explicit cache controls.
///
/// By default, caching is enabled with automatic breakpoint placement.
/// Providers with automatic server-side caching, such as DeepSeek, may ignore
/// this setting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Master switch — set to false to disable all caching hints.
    /// Default: true.
    pub enabled: bool,
    /// How cache breakpoints are placed.
    pub strategy: CacheStrategy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: CacheStrategy::Auto,
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

/// Strategy for placing cache breakpoints (Anthropic-specific; other providers
/// handle caching automatically regardless of this setting).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum CacheStrategy {
    /// Automatic breakpoint placement (recommended).
    /// Caches: system prompt, tool definitions, and recent conversation history.
    #[default]
    Auto,
    /// Disable explicit cache hints.
    Disabled,
    /// Fine-grained control over what gets cached.
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
/// adding fields to `ToolContext` is non-breaking.
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
    AgentEnd {
        messages: Vec<AgentMessage>,
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
    ContextCompacted {
        /// Which compaction path produced this result.
        method: CompactionMethod,
        messages_before: usize,
        messages_after: usize,
        tokens_before: usize,
        tokens_after: usize,
        /// Messages replaced by the summary. Zero on the deterministic path.
        messages_summarized: usize,
        /// What producing the summary cost. `None` on the deterministic path,
        /// which issues no request.
        ///
        /// This is the number to weigh against `tokens_before - tokens_after`
        /// when deciding whether an LLM compaction strategy earns its keep.
        summary_usage: Option<Usage>,
        /// Dollar cost of `summary_usage`, when the summarization model's rates
        /// are configured (see [`CostConfig`](crate::provider::CostConfig)).
        summary_cost_usd: Option<f64>,
    },
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
    /// On [`LlmCompaction`](crate::LlmCompaction) this means no summary was
    /// ready in time — the loop stayed unblocked, but the compaction was lossy.
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

/// Wire-tag freeze for [`AgentEvent`] and [`StreamDelta`].
///
/// These matches deliberately have **no wildcard arm**: adding a variant must
/// fail to compile here until its serde tag is pinned. That guarantee is the
/// reason they live in this crate rather than in `tests/` — both enums are
/// `#[non_exhaustive]`, so an integration test *cannot* match them
/// exhaustively and the compile-time freeze would silently degrade to a
/// wildcard. Inside the defining crate, exhaustiveness still applies.
///
/// A tag change is a breaking change for wire clients — do not edit casually.
/// `tests/serialization_test.rs` covers the round-trips and payload shapes;
/// this covers only "no variant escapes without a pinned tag".
#[cfg(test)]
mod wire_tag_freeze {
    use super::*;

    fn expected_event_tag(event: &AgentEvent) -> &'static str {
        match event {
            AgentEvent::AgentStart => "agentStart",
            AgentEvent::AgentEnd { .. } => "agentEnd",
            AgentEvent::TurnStart => "turnStart",
            AgentEvent::TurnEnd { .. } => "turnEnd",
            AgentEvent::MessageStart { .. } => "messageStart",
            AgentEvent::MessageUpdate { .. } => "messageUpdate",
            AgentEvent::MessageEnd { .. } => "messageEnd",
            AgentEvent::ToolExecutionStart { .. } => "toolExecutionStart",
            AgentEvent::ToolExecutionUpdate { .. } => "toolExecutionUpdate",
            AgentEvent::ToolExecutionEnd { .. } => "toolExecutionEnd",
            AgentEvent::ProgressMessage { .. } => "progressMessage",
            AgentEvent::InputRejected { .. } => "inputRejected",
            AgentEvent::ContextCompacted { .. } => "contextCompacted",
        }
    }

    fn expected_delta_tag(delta: &StreamDelta) -> &'static str {
        match delta {
            StreamDelta::Text { .. } => "text",
            StreamDelta::Thinking { .. } => "thinking",
            StreamDelta::ToolCallDelta { .. } => "toolCallDelta",
        }
    }

    #[test]
    fn every_event_variant_has_a_pinned_tag() {
        // Sampling one variant is enough: the compiler enforces coverage, this
        // just proves the mapping matches what serde actually emits.
        let event = AgentEvent::AgentStart;
        let v = serde_json::to_value(&event).expect("serialize");
        assert_eq!(v["type"], expected_event_tag(&event));
    }

    #[test]
    fn every_delta_variant_has_a_pinned_tag() {
        let delta = StreamDelta::Text { delta: "x".into() };
        let v = serde_json::to_value(&delta).expect("serialize");
        assert_eq!(v["type"], expected_delta_tag(&delta));
    }
}
