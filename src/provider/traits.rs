use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::model::ModelConfig;

/// Events emitted during LLM streaming
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamEvent {
    /// Stream started, partial assistant message
    Start,
    /// Text content delta
    TextDelta { content_index: usize, delta: String },
    /// Thinking content delta
    ThinkingDelta { content_index: usize, delta: String },
    /// Tool call started
    ToolCallStart {
        content_index: usize,
        id: String,
        name: String,
    },
    /// Tool call argument delta
    ToolCallDelta { content_index: usize, delta: String },
    /// Tool call ended
    ToolCallEnd { content_index: usize },
    /// Stream completed successfully
    Done { message: Message },
    /// Stream errored
    Error { message: Message },
}

/// Configuration for a streaming LLM call.
///
/// Marked `#[non_exhaustive]`: fields are added in minor releases (this
/// release alone added `output_schema`). Construct with
/// [`StreamConfig::new`] and mutate the public fields:
///
/// ```
/// # use yoagent::provider::StreamConfig;
/// let mut config = StreamConfig::new("claude-sonnet-5", "sk-key");
/// config.system_prompt = "be brief".into();
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct StreamConfig {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub thinking_level: ThinkingLevel,
    pub api_key: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Optional model configuration for multi-provider support.
    /// When set, providers use this for base_url, compat flags, headers, etc.
    pub model_config: Option<ModelConfig>,
    /// Prompt caching configuration. Default: enabled with auto strategy.
    pub cache_config: CacheConfig,
    /// Structured-output constraint. When set, providers enforce the schema
    /// natively (Anthropic: forced tool call; OpenAI-compat: `json_schema`
    /// response format; Gemini: `responseSchema`). Providers without support
    /// log a warning and ignore it.
    pub output_schema: Option<OutputSchema>,
}

/// Redacts `api_key`. A `{:?}` of this struct reaches logs and panic
/// messages; a derived `Debug` would print the credential verbatim.
impl std::fmt::Debug for StreamConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamConfig")
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field("tools", &self.tools)
            .field("thinking_level", &self.thinking_level)
            .field("api_key", &Redacted(&self.api_key))
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("model_config", &self.model_config)
            .field("cache_config", &self.cache_config)
            .field("output_schema", &self.output_schema)
            .finish_non_exhaustive()
    }
}

/// Prints a placeholder instead of a secret, preserving only whether one is set.
pub(crate) struct Redacted<'a>(pub(crate) &'a str);

impl std::fmt::Debug for Redacted<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            f.write_str("\"\"")
        } else {
            f.write_str("\"[redacted]\"")
        }
    }
}

impl StreamConfig {
    /// A config with the given model and API key; everything else defaults
    /// (empty prompt/messages/tools, thinking off, caching enabled).
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            thinking_level: ThinkingLevel::Off,
            api_key: api_key.into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        }
    }

    /// The cache-routing key for this request, or `None` when no key should be
    /// sent.
    ///
    /// Returns [`CacheConfig::session_key`] when the caller set a non-blank
    /// one. Otherwise derives a key from the **system prompt alone**.
    ///
    /// # Why only the system prompt
    ///
    /// `prompt_cache_key` routes a request to a machine; the cache itself is
    /// content-addressed. So the key should group requests that share a
    /// *cacheable prefix*, and the system prompt is exactly that prefix —
    /// it is what the provider caches, and `StreamConfig` carries it in its
    /// own field where compaction cannot reach it.
    ///
    /// An earlier version also mixed in the first user message, for session
    /// discrimination. That was wrong, and measurably so: compaction rewrites
    /// the head. When [`crate::context::compact_messages`] drops far enough to
    /// re-enter `keep_within_budget`, it inserts a *constant* marker message at
    /// index 0 — constant on purpose, so the cached prefix stays byte-stable.
    /// The derived key then drifted mid-session **and** collapsed onto one
    /// value for every session sharing a system prompt. Session identity is not
    /// recoverable from a per-request snapshot; inferring it from mutable
    /// content produced a key that failed exactly on long sessions, which are
    /// the ones caching exists for.
    ///
    /// # Consequence, and the escape hatch
    ///
    /// Sessions sharing a system prompt now share a key **by design**. That is
    /// the correct grouping — they share the cached prefix — but it also
    /// concentrates traffic on one cache. High-volume deployments should set
    /// [`CacheConfig::session_key`] explicitly to spread load, which is also
    /// what OpenAI recommends for a hot key.
    ///
    /// Only consumed by key-routed protocols; providers taking explicit
    /// breakpoints ignore it.
    pub fn cache_session_key(&self) -> Option<String> {
        let cache = &self.cache_config;
        if !cache.hints_enabled() {
            return None;
        }
        // A blank explicit key is worse than none: it would send
        // `prompt_cache_key: ""` and route every such caller together. Treat it
        // as unset rather than honouring it literally.
        if let Some(key) = cache.session_key.as_deref().map(str::trim) {
            if !key.is_empty() {
                return Some(key.to_string());
            }
        }

        match self.system_prompt.trim() {
            "" => None,
            sys => Some(format!("yo-{:016x}", fnv1a(sys))),
        }
    }
}

/// FNV-1a. Chosen over `DefaultHasher` for the same reason the GASP recorder
/// does: `DefaultHasher` is neither guaranteed stable across Rust releases nor
/// documented as such, and a cache key that moves between compiler versions
/// silently stops matching.
fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// JSON-Schema constraint for structured outputs.
///
/// Marked `#[non_exhaustive]`: fields may be added in minor releases (e.g.
/// strictness flags). Construct with [`OutputSchema::new`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputSchema {
    /// Schema name (doubles as the forced tool name on Anthropic).
    pub name: String,
    /// The JSON Schema the model's reply must satisfy.
    pub schema: serde_json::Value,
}

impl OutputSchema {
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            schema,
        }
    }
}

/// Tool definition sent to the LLM (schema only, no execute fn)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

use serde::{Deserialize, Serialize};

/// The core provider trait. Implement this for each LLM backend.
#[async_trait]
pub trait StreamProvider: Send + Sync {
    /// Stream a completion, sending [`StreamEvent`]s through the channel.
    ///
    /// On success returns the final complete assistant [`Message`].
    /// On failure returns a [`ProviderError`] (used by retry logic to decide
    /// whether the call is retryable).
    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError>;

    /// The API protocol this provider speaks, if it maps to a single one.
    ///
    /// Built-in providers return `Some(_)`; the default is `None` (for test
    /// doubles and multi-protocol adapters). Used to verify registry wiring
    /// (a resolved provider should report the protocol it was registered for)
    /// and to enable protocol-mismatch diagnostics.
    fn protocol(&self) -> Option<crate::provider::ApiProtocol> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("API error: {0}")]
    Api(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Auth error: {0}")]
    Auth(String),
    #[error("Rate limited, retry after {retry_after_ms:?}ms")]
    RateLimited { retry_after_ms: Option<u64> },
    #[error("Context overflow: {message}")]
    ContextOverflow { message: String },
    #[error("Cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

impl ProviderError {
    /// Classify an HTTP error response into the appropriate error variant.
    ///
    /// Detects context overflow, rate limits, auth errors, and general API errors
    /// from the HTTP status code and response body.
    pub fn classify(status: u16, message: &str) -> Self {
        Self::classify_with_retry_after(status, message, None)
    }

    /// Like [`classify`](Self::classify), carrying a parsed `Retry-After`
    /// value (milliseconds) into the `RateLimited` variant when present.
    pub fn classify_with_retry_after(
        status: u16,
        message: &str,
        retry_after_ms: Option<u64>,
    ) -> Self {
        if is_context_overflow(status, message) {
            Self::ContextOverflow {
                message: message.to_string(),
            }
        } else if status == 429 {
            Self::RateLimited { retry_after_ms }
        } else if status == 401 || status == 403 {
            Self::Auth(message.to_string())
        } else {
            Self::Api(message.to_string())
        }
    }

    /// Returns true if this error indicates a context overflow.
    pub fn is_context_overflow(&self) -> bool {
        matches!(self, Self::ContextOverflow { .. })
    }
}

/// Extract a classified error from a `reqwest_eventsource::Error`.
///
/// - `InvalidStatusCode` — reads the response body and classifies via
///   [`ProviderError::classify()`] (context overflow, rate limit, auth, etc.).
/// - `Transport` — maps to [`ProviderError::Network`] (retryable).
/// - `StreamEnded` — the HTTP body ended *legally* (terminal chunk, exact
///   `Content-Length`, or a close-framed response) but the SSE payload carried
///   no terminator. Note this is **not** the connection-reset case: a mid-body
///   TCP reset is a decode error and surfaces as `Transport`. Mapped to
///   [`ProviderError::Network`] (retryable) because a gateway that returns a
///   well-framed body with a truncated payload is usually transient.
///
///   Reaching this arm must mean no complete response was assembled. A provider
///   that can finish a response before a terminator-less close is required to
///   catch `StreamEnded` itself and break instead — otherwise a finished
///   response gets retried and re-billed. `openai_compat` (`saw_finish_reason`)
///   and `anthropic` (`saw_stop_reason`) do this. `openai_responses` and
///   `azure_openai` instead break on every terminal event they can receive
///   (`response.completed` / `.incomplete` / `.failed`), which upholds the same
///   invariant without a flag.
/// - All other variants (protocol/parse errors like `InvalidContentType`,
///   `Utf8`, `Parser`, `InvalidLastEventId`) — maps to [`ProviderError::Other`]
///   (non-retryable, fail fast).
pub async fn classify_eventsource_error(error: reqwest_eventsource::Error) -> ProviderError {
    match error {
        reqwest_eventsource::Error::InvalidStatusCode(status, response) => {
            let status_code = status.as_u16();
            let retry_after_ms = parse_retry_after(response.headers());
            let body = response.text().await.unwrap_or_default();
            ProviderError::classify_with_retry_after(
                status_code,
                &format!(
                    "HTTP {} {}: {}",
                    status_code,
                    status.canonical_reason().unwrap_or(""),
                    body
                ),
                retry_after_ms,
            )
        }
        reqwest_eventsource::Error::Transport(e) => ProviderError::Network(format!("{:?}", e)),
        reqwest_eventsource::Error::StreamEnded => ProviderError::Network(
            "SSE body ended without a terminator and no complete response was assembled \
             (if this repeats, the endpoint may be returning 200 with an empty or \
             truncated body rather than an error status)"
                .into(),
        ),
        other => ProviderError::Other(other.to_string()),
    }
}

/// Parse a `Retry-After` header (seconds form) into milliseconds.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|s| *s >= 0.0)
        .map(|secs| (secs * 1000.0) as u64)
}

/// Classify an SSE-embedded error event message into a [`ProviderError`].
///
/// Checks the error text for known patterns (context overflow, etc.).
/// Used by providers that receive `"error"` events in the SSE stream.
pub fn classify_sse_error_event(message: &str) -> ProviderError {
    if is_context_overflow_message(message) {
        ProviderError::ContextOverflow {
            message: message.to_string(),
        }
    } else {
        ProviderError::Api(message.to_string())
    }
}

/// Known phrases that indicate context overflow across LLM providers.
///
/// Covers: Anthropic, OpenAI, Google Gemini, AWS Bedrock, xAI, Groq,
/// OpenRouter, llama.cpp, LM Studio, MiniMax, Kimi, GitHub Copilot,
/// and generic patterns.
const OVERFLOW_PHRASES: &[&str] = &[
    "prompt is too long",                 // Anthropic
    "input is too long",                  // AWS Bedrock
    "exceeds the context window",         // OpenAI (Completions & Responses)
    "exceeds the maximum",                // Google Gemini ("input token count exceeds the maximum")
    "maximum prompt length",              // xAI
    "reduce the length of the messages",  // Groq
    "maximum context length",             // OpenRouter
    "exceeds the limit of",               // GitHub Copilot
    "exceeds the available context size", // llama.cpp
    "greater than the context length",    // LM Studio
    "context window exceeds limit",       // MiniMax
    "exceeded model token limit",         // Kimi
    "context length exceeded",            // Generic
    "context_length_exceeded",            // Generic (underscore variant)
    "model_context_window_exceeded",      // Anthropic in-stream stop_reason
    "too many tokens",                    // Generic
    "token limit exceeded",               // Generic
];

/// Check if an error message indicates context overflow (for use by types.rs).
pub(crate) fn is_context_overflow_message(message: &str) -> bool {
    let lower = message.to_lowercase();
    OVERFLOW_PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// Check if an HTTP error response indicates context overflow.
fn is_context_overflow(status: u16, message: &str) -> bool {
    // Some providers (Cerebras, Mistral) return 400/413 with empty body on overflow
    if (status == 400 || status == 413) && message.trim().is_empty() {
        return true;
    }
    is_context_overflow_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_retry_after(value: &str) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
        h
    }

    #[test]
    fn parse_retry_after_whole_seconds() {
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("5")),
            Some(5000)
        );
    }

    #[test]
    fn parse_retry_after_fractional_seconds() {
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("1.5")),
            Some(1500)
        );
    }

    #[test]
    fn parse_retry_after_rejects_negative() {
        assert_eq!(parse_retry_after(&headers_with_retry_after("-1")), None);
    }

    #[test]
    fn parse_retry_after_rejects_http_date() {
        // The HTTP-date form of Retry-After is not supported; must not
        // misparse as a huge delay.
        assert_eq!(
            parse_retry_after(&headers_with_retry_after("Wed, 21 Oct 2015 07:28:00 GMT")),
            None
        );
    }

    #[test]
    fn parse_retry_after_missing_header() {
        assert_eq!(parse_retry_after(&reqwest::header::HeaderMap::new()), None);
    }

    #[test]
    fn classify_anthropic_overflow() {
        let err =
            ProviderError::classify(400, "prompt is too long: 213462 tokens > 200000 maximum");
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_openai_overflow() {
        let err =
            ProviderError::classify(400, "Your input exceeds the context window of this model");
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_google_overflow() {
        let err = ProviderError::classify(
            400,
            "The input token count (1196265) exceeds the maximum number of tokens allowed",
        );
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_bedrock_overflow() {
        let err = ProviderError::classify(400, "input is too long for requested model");
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_xai_overflow() {
        let err = ProviderError::classify(
            400,
            "This model's maximum prompt length is 131072 but request contains 537812 tokens",
        );
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_groq_overflow() {
        let err = ProviderError::classify(
            400,
            "Please reduce the length of the messages or completion",
        );
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_empty_body_overflow() {
        // Cerebras/Mistral return 400/413 with empty body
        let err = ProviderError::classify(413, "");
        assert!(err.is_context_overflow());
        let err = ProviderError::classify(400, "  ");
        assert!(err.is_context_overflow());
    }

    #[test]
    fn classify_rate_limit() {
        let err = ProviderError::classify(429, "rate limit exceeded");
        assert!(matches!(err, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn classify_auth_error() {
        let err = ProviderError::classify(401, "invalid api key");
        assert!(matches!(err, ProviderError::Auth(_)));
        let err = ProviderError::classify(403, "forbidden");
        assert!(matches!(err, ProviderError::Auth(_)));
    }

    #[test]
    fn classify_regular_api_error() {
        let err = ProviderError::classify(400, "invalid request format");
        assert!(matches!(err, ProviderError::Api(_)));
        assert!(!err.is_context_overflow());
    }

    #[test]
    fn overflow_message_case_insensitive() {
        assert!(is_context_overflow_message("PROMPT IS TOO LONG"));
        assert!(is_context_overflow_message("Too Many Tokens in request"));
    }

    #[test]
    fn non_overflow_messages() {
        assert!(!is_context_overflow_message("invalid api key"));
        assert!(!is_context_overflow_message("internal server error"));
        assert!(!is_context_overflow_message(""));
    }

    fn config_with(system: &str, messages: Vec<Message>) -> StreamConfig {
        let mut c = StreamConfig::new("m", "k");
        c.system_prompt = system.into();
        c.messages = messages;
        c
    }

    #[test]
    fn session_key_is_none_when_caching_is_off() {
        let mut c = config_with("sys", vec![Message::user("hi")]);
        c.cache_config = CacheConfig::disabled();
        assert!(c.cache_session_key().is_none());

        c.cache_config = CacheConfig {
            strategy: CacheStrategy::Disabled,
            ..CacheConfig::default()
        };
        assert!(c.cache_session_key().is_none());
    }

    #[test]
    fn explicit_session_key_is_returned_verbatim() {
        let mut c = config_with("sys", vec![Message::user("hi")]);
        c.cache_config = CacheConfig::default().with_session_key("abc");
        assert_eq!(c.cache_session_key().as_deref(), Some("abc"));
    }

    /// The load-bearing property: appending turns must not move the key, or
    /// every request routes to a fresh cache.
    #[test]
    fn derived_key_ignores_everything_after_the_first_user_message() {
        let a = config_with("sys", vec![Message::user("first")]);
        let b = config_with(
            "sys",
            vec![
                Message::user("first"),
                Message::user("second"),
                Message::user("third"),
            ],
        );
        assert_eq!(a.cache_session_key(), b.cache_session_key());
        assert!(a.cache_session_key().is_some());
    }

    /// Regression: the key must survive **compaction**, not merely appending.
    ///
    /// The first version of this derivation mixed in the first user message.
    /// `compact_messages` can drop the head and insert a deliberately constant
    /// marker at index 0, so the key drifted mid-session *and* every session
    /// sharing a system prompt collapsed onto one value — the exact
    /// route-unrelated-sessions-together harm the empty-head guard exists to
    /// prevent, arriving through a different door. Appending-only tests could
    /// not see it, because they assert stability under the one condition where
    /// the old derivation held.
    #[test]
    fn derived_key_survives_compaction_that_rewrites_the_head() {
        use crate::context::{compact_messages, ContextConfig};

        let cfg = ContextConfig {
            // Small enough that the retained tail alone busts the target, which
            // is what re-enters `keep_within_budget` and drops the head.
            max_context_tokens: 400,
            system_prompt_tokens: 0,
            ..ContextConfig::default()
        };

        let conversation = |tag: &str| -> Vec<crate::types::AgentMessage> {
            (0..80)
                .map(|i| {
                    crate::types::AgentMessage::Llm(Message::User {
                        content: vec![Content::Text {
                            text: format!("{tag} message {i} {}", "filler ".repeat(40)),
                        }],
                        timestamp: i as u64,
                    })
                })
                .collect()
        };

        let key_of = |msgs: &[crate::types::AgentMessage]| {
            let mut c = StreamConfig::new("m", "k");
            c.system_prompt = "shared system prompt".into();
            c.messages = msgs
                .iter()
                .filter_map(|m| match m {
                    crate::types::AgentMessage::Llm(l) => Some(l.clone()),
                    _ => None,
                })
                .collect();
            c.cache_session_key()
        };

        let (a, b) = (conversation("alpha"), conversation("bravo"));
        let (a_before, b_before) = (key_of(&a), key_of(&b));
        let (a_len, b_len) = (a.len(), b.len());
        let (a_compacted, b_compacted) = (compact_messages(a, &cfg), compact_messages(b, &cfg));

        // Without this the test passes vacuously if a future ContextConfig
        // default stops triggering compaction at this size.
        assert!(
            a_compacted.len() < a_len && b_compacted.len() < b_len,
            "precondition: compaction must actually have fired"
        );
        assert!(
            matches!(
                &a_compacted[0],
                crate::types::AgentMessage::Llm(Message::User { content, .. })
                    if content.iter().any(|c| matches!(
                        c, Content::Text { text } if text == crate::context::COMPACTION_MARKER
                    ))
            ),
            "precondition: the head was dropped and replaced by the marker"
        );

        let (a_after, b_after) = (key_of(&a_compacted), key_of(&b_compacted));
        assert!(a_before.is_some(), "precondition: a key is derived at all");
        assert_eq!(a_before, a_after, "compaction must not move the key");
        assert_eq!(b_before, b_after, "compaction must not move the key");
    }

    #[test]
    fn distinct_system_prompts_derive_distinct_keys() {
        let a = config_with("sys A", vec![Message::user("same")]);
        let b = config_with("sys B", vec![Message::user("same")]);
        assert_ne!(a.cache_session_key(), b.cache_session_key());
    }

    /// Documents the deliberate consequence of keying on the system prompt
    /// alone: sessions of the same agent share a key. That is the correct
    /// grouping — they share the cached prefix — and the reason
    /// `session_key` exists for deployments that need them apart.
    #[test]
    fn sessions_sharing_a_system_prompt_share_a_key_by_design() {
        let a = config_with("same", vec![Message::user("deploy the API")]);
        let b = config_with("same", vec![Message::user("write a poem")]);
        assert_eq!(a.cache_session_key(), b.cache_session_key());

        let mut split = config_with("same", vec![Message::user("write a poem")]);
        split.cache_config = CacheConfig::default().with_session_key("tenant-7/session-3");
        assert_ne!(a.cache_session_key(), split.cache_session_key());
    }

    /// Message content of every kind is outside the derivation, so a
    /// multimodal or tool-only head cannot produce a degenerate key.
    #[test]
    fn non_text_content_does_not_affect_the_key() {
        let text_only = config_with("sys", vec![Message::user("hello")]);
        let image_only = config_with(
            "sys",
            vec![Message::User {
                content: vec![Content::Image {
                    data: "AAAA".into(),
                    mime_type: "image/png".into(),
                }],
                timestamp: 0,
            }],
        );
        assert_eq!(
            text_only.cache_session_key(),
            image_only.cache_session_key()
        );
        assert!(text_only.cache_session_key().is_some());
    }

    /// Nothing distinctive to key on — hashing the empty string would route
    /// unrelated sessions together, which is worse than sending no key.
    #[test]
    fn empty_system_prompt_yields_no_key() {
        assert!(config_with("", vec![]).cache_session_key().is_none());
        assert!(config_with("   ", vec![Message::user("hi")])
            .cache_session_key()
            .is_none());
    }

    /// A blank explicit key would send `prompt_cache_key: ""`, routing every
    /// caller who did that onto one cache — strictly worse than no key.
    #[test]
    fn blank_explicit_session_key_is_treated_as_unset() {
        let mut c = config_with("sys", vec![Message::user("hi")]);
        c.cache_config = CacheConfig::default().with_session_key("   ");
        assert_eq!(
            c.cache_session_key(),
            config_with("sys", vec![]).cache_session_key(),
            "blank key must fall through to derivation, not send an empty string"
        );
    }

    /// `Manual` with every flag off means "cache nothing" — Anthropic honours
    /// it by placing no breakpoints, so a key-routed provider must not read the
    /// same value as "yes".
    #[test]
    fn manual_with_all_flags_off_sends_no_key() {
        let mut c = config_with("sys", vec![Message::user("hi")]);
        c.cache_config = CacheConfig::default().with_strategy(CacheStrategy::Manual {
            cache_system: false,
            cache_tools: false,
            cache_messages: false,
        });
        assert!(c.cache_session_key().is_none());

        // But a Manual that asks for *something* still routes.
        c.cache_config = CacheConfig::default().with_strategy(CacheStrategy::Manual {
            cache_system: true,
            cache_tools: false,
            cache_messages: false,
        });
        assert!(c.cache_session_key().is_some());
    }

    #[test]
    fn a_system_prompt_alone_is_enough_to_derive_a_key() {
        assert!(config_with("sys", vec![]).cache_session_key().is_some());
    }
}
