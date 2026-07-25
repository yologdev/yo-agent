use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;

use super::model::ModelConfig;

/// Events emitted during LLM streaming
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
}
