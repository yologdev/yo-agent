//! Model configuration and provider compatibility flags.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which API protocol a model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ApiProtocol {
    AnthropicMessages,
    OpenAiCompletions,
    OpenAiResponses,
    AzureOpenAiResponses,
    GoogleGenerativeAi,
    GoogleVertex,
    BedrockConverseStream,
}

impl std::fmt::Display for ApiProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AnthropicMessages => write!(f, "anthropic_messages"),
            Self::OpenAiCompletions => write!(f, "openai_completions"),
            Self::OpenAiResponses => write!(f, "openai_responses"),
            Self::AzureOpenAiResponses => write!(f, "azure_openai_responses"),
            Self::GoogleGenerativeAi => write!(f, "google_generative_ai"),
            Self::GoogleVertex => write!(f, "google_vertex"),
            Self::BedrockConverseStream => write!(f, "bedrock_converse_stream"),
        }
    }
}

/// Cost per million tokens (input/output).
///
/// # These are a snapshot, not an authority
///
/// The built-in presets carry rates verified against the vendor's published
/// pricing on the date noted at each constructor. Vendors reprice, and a
/// compiled-in number cannot notice — `claude_sonnet_5` shipped Sonnet 4.6's
/// rates across 18 releases, v0.9.0 through v0.16.5, overstating every
/// `cost_usd` for that model by 50%, and nothing detected it.
///
/// `tests/price_audit.rs` now diffs every preset against models.dev; run it
/// before a release:
///
/// ```text
/// cargo test --test price_audit -- --ignored --nocapture
/// ```
///
/// `ModelConfig::cost` is a public field and `CostConfig` is `Deserialize`, so
/// a caller never has to wait for a release — override it for a negotiated
/// rate, or load rates from configuration:
///
/// ```
/// # use yoagent::provider::ModelConfig;
/// let mut config = ModelConfig::claude_sonnet_5();
/// config.cost.input_per_million = 1.80; // your negotiated rate
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    pub input_per_million: f64,
    pub output_per_million: f64,
    #[serde(default)]
    pub cache_read_per_million: f64,
    #[serde(default)]
    pub cache_write_per_million: f64,
}

impl CostConfig {
    /// Whether any rate is set. All-zero rates mean pricing is unknown
    /// (custom/local models), not that the model is free.
    pub fn is_configured(&self) -> bool {
        self.input_per_million != 0.0
            || self.output_per_million != 0.0
            || self.cache_read_per_million != 0.0
            || self.cache_write_per_million != 0.0
    }

    /// Dollar cost of a usage record at these per-million-token rates.
    ///
    /// Consumed by [`crate::Agent::session_cost_usd`]; also usable directly
    /// in `after_turn` callbacks for per-turn cost tracking.
    pub fn cost_usd(&self, usage: &crate::types::Usage) -> f64 {
        (usage.input as f64 * self.input_per_million
            + usage.output as f64 * self.output_per_million
            + usage.cache_read as f64 * self.cache_read_per_million
            + usage.cache_write as f64 * self.cache_write_per_million)
            / 1_000_000.0
    }
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            input_per_million: 0.0,
            output_per_million: 0.0,
            cache_read_per_million: 0.0,
            cache_write_per_million: 0.0,
        }
    }
}

/// How a provider handles the `max_tokens` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    #[default]
    MaxTokens,
    MaxCompletionTokens,
}

/// How a provider formats thinking/reasoning output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingFormat {
    #[default]
    OpenAi,
    Xai,
    Qwen,
}

/// Compatibility flags for OpenAI-compatible providers.
/// Different providers have different quirks even though they share the same base API.
///
/// Marked `#[non_exhaustive]`: this is the crate's most literal instance of a
/// growing quirk list — every new provider difference adds a flag, and without
/// the attribute each one is a downstream break. Construct from a preset
/// ([`OpenAiCompat::openai`], [`OpenAiCompat::deepseek`], …) or
/// [`Default::default`] and adjust fields. New flags carry `#[serde(default)]`
/// so persisted configs keep deserializing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiCompat {
    /// Supports the `store` parameter for conversation persistence.
    pub supports_store: bool,
    /// Supports `developer` role (system-level instructions).
    pub supports_developer_role: bool,
    /// Supports `reasoning_effort` parameter.
    pub supports_reasoning_effort: bool,
    /// Supports DeepSeek-style `thinking` mode control.
    #[serde(default)]
    pub supports_thinking_control: bool,
    /// Includes usage data in streaming responses.
    pub supports_usage_in_streaming: bool,
    /// Which field name to use for max tokens.
    pub max_tokens_field: MaxTokensField,
    /// Tool results must include a `name` field.
    pub requires_tool_result_name: bool,
    /// Must insert an assistant message after tool results.
    #[serde(default)]
    pub requires_assistant_after_tool_result: bool,
    /// How thinking/reasoning content is formatted in streaming.
    pub thinking_format: ThinkingFormat,
    /// Accepts OpenAI's `prompt_cache_key` for routing cache lookups.
    ///
    /// Off by default: the field is OpenAI's, and a strict compat server that
    /// validates unknown keys would reject the whole request rather than
    /// ignore it. Providers that cache automatically (DeepSeek, Groq) lose
    /// nothing by leaving this off — they were never reading it.
    #[serde(default)]
    pub supports_prompt_cache_key: bool,
}

impl Default for OpenAiCompat {
    fn default() -> Self {
        Self {
            supports_store: false,
            supports_developer_role: false,
            supports_reasoning_effort: false,
            supports_thinking_control: false,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxTokens,
            requires_tool_result_name: false,
            requires_assistant_after_tool_result: false,
            thinking_format: ThinkingFormat::OpenAi,
            supports_prompt_cache_key: false,
        }
    }
}

impl OpenAiCompat {
    /// Compat flags for native OpenAI.
    pub fn openai() -> Self {
        Self {
            supports_store: true,
            supports_developer_role: true,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_prompt_cache_key: true,
            ..Default::default()
        }
    }

    /// Compat flags for the Meta Model API (Muse Spark).
    ///
    /// OpenAI-compatible chat completions. Meta documents `reasoning_effort`
    /// (default `medium` server-side) and streamed usage via
    /// `stream_options.include_usage`; `max_tokens` is deprecated in favor of
    /// `max_completion_tokens`.
    pub fn meta() -> Self {
        Self {
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            ..Default::default()
        }
    }

    /// Compat flags for xAI (Grok).
    pub fn xai() -> Self {
        Self {
            supports_usage_in_streaming: true,
            thinking_format: ThinkingFormat::Xai,
            ..Default::default()
        }
    }

    /// Compat flags for Groq.
    pub fn groq() -> Self {
        Self {
            supports_usage_in_streaming: true,
            ..Default::default()
        }
    }

    /// Compat flags for Cerebras.
    pub fn cerebras() -> Self {
        Self::default()
    }

    /// Compat flags for OpenRouter.
    pub fn openrouter() -> Self {
        Self {
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            ..Default::default()
        }
    }

    /// Compat flags for Mistral.
    pub fn mistral() -> Self {
        Self {
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxTokens,
            ..Default::default()
        }
    }

    /// Compat flags for DeepSeek.
    pub fn deepseek() -> Self {
        Self {
            supports_reasoning_effort: true,
            supports_thinking_control: true,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxTokens,
            ..Default::default()
        }
    }

    /// Compat flags for Z.ai (Zhipu AI).
    pub fn zai() -> Self {
        Self {
            supports_usage_in_streaming: true,
            ..Default::default()
        }
    }

    /// Compat flags for MiniMax.
    pub fn minimax() -> Self {
        Self {
            supports_usage_in_streaming: true,
            ..Default::default()
        }
    }

    /// Compat flags for Qwen / DashScope.
    pub fn qwen() -> Self {
        Self {
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxTokens,
            thinking_format: ThinkingFormat::Qwen,
            ..Default::default()
        }
    }

    /// Compat flags for Ollama's OpenAI-compatible API.
    pub fn ollama() -> Self {
        Self {
            requires_assistant_after_tool_result: true,
            ..Default::default()
        }
    }
}

/// Quirk flags for the Anthropic Messages protocol (only for AnthropicMessages).
///
/// When `ModelConfig.anthropic` is `None`, providers use `AnthropicCompat::default()`,
/// which targets the current model generation (Claude 4.6+ / Fable 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicCompat {
    /// Use adaptive thinking (`thinking: {"type": "adaptive"}` plus
    /// `output_config.effort`). Required by Claude Fable 5, Opus 5, Opus 4.7/4.8,
    /// and Sonnet 5; recommended on Opus 4.6 / Sonnet 4.6. Set to `false` for
    /// pre-4.6 models, which only accept `{"type": "enabled", "budget_tokens": N}`.
    pub adaptive_thinking: bool,
    /// Send the API key as `Authorization: Bearer {key}` instead of the
    /// Anthropic-native `x-api-key` header. Needed for OpenAI-style gateways
    /// that speak the Anthropic Messages protocol (e.g. OpenCode Zen/Go).
    pub bearer_auth: bool,
}

impl Default for AnthropicCompat {
    fn default() -> Self {
        Self {
            adaptive_thinking: true,
            bearer_auth: false,
        }
    }
}

impl AnthropicCompat {
    /// Compat flags for pre-4.6 Claude models (budget-based extended thinking).
    pub fn legacy() -> Self {
        Self {
            adaptive_thinking: false,
            bearer_auth: false,
        }
    }
}

/// The two OpenCode gateways (<https://opencode.ai>).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenCodeGateway {
    /// Pay-per-use gateway (`opencode.ai/zen/v1`).
    Zen,
    /// Subscription gateway for open models (`opencode.ai/zen/go/v1`).
    Go,
}

impl OpenCodeGateway {
    fn provider_name(self) -> &'static str {
        match self {
            Self::Zen => "opencode-zen",
            Self::Go => "opencode-go",
        }
    }

    fn base_url(self) -> &'static str {
        match self {
            Self::Zen => "https://opencode.ai/zen/v1",
            Self::Go => "https://opencode.ai/zen/go/v1",
        }
    }
}

/// Full model configuration. Knows everything needed to make API calls.
///
/// Marked `#[non_exhaustive]`: fields may be added in minor releases (e.g.
/// the `anthropic` compat flags, slated for 0.9.0). Construct via the
/// `ModelConfig::*` preset constructors — or [`ModelConfig::custom`] for
/// protocols without a preset — and mutate fields to customize. Note that
/// downstream struct literals and functional-record-update
/// (`ModelConfig { .. }`) no longer compile; field mutation is the supported
/// pattern. New fields must carry `#[serde(default)]` so previously
/// persisted configs keep deserializing.
#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelConfig {
    /// Model identifier sent to the API (e.g. "gpt-4o", "claude-sonnet-4-20250514").
    pub id: String,
    /// Human-friendly name.
    pub name: String,
    /// Which API protocol to use.
    pub api: ApiProtocol,
    /// Provider name (e.g. "openai", "anthropic", "xai").
    pub provider: String,
    /// Base URL for API requests (without trailing slash).
    pub base_url: String,
    /// Whether this model supports reasoning/thinking. When `false` and a
    /// `thinking_level` is requested, the [`Agent`](crate::Agent) wrapper
    /// logs a warning; sub-agents and direct `agent_loop` calls do not. The
    /// request is still sent either way — gate behavior stays with the
    /// caller.
    pub reasoning: bool,
    /// Context window size in tokens.
    pub context_window: u32,
    /// Default max output tokens.
    pub max_tokens: u32,
    /// Cost configuration.
    #[serde(default)]
    pub cost: CostConfig,
    /// Additional headers to send with requests.
    ///
    /// May carry credentials (`Authorization`, `x-api-key`). `Debug` prints
    /// header *names* with redacted values, but `Serialize` is intentionally
    /// lossless so configs round-trip — do not serialize a `ModelConfig` into
    /// logs or telemetry.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// OpenAI-compat quirk flags (only for OpenAiCompletions protocol).
    #[serde(default)]
    pub compat: Option<OpenAiCompat>,
    /// Anthropic Messages quirk flags (only for AnthropicMessages protocol).
    /// `None` behaves like `AnthropicCompat::default()` (current generation).
    #[serde(default)]
    pub anthropic: Option<AnthropicCompat>,
}

/// Redacts header values. Headers routinely carry credentials, and a
/// derived `Debug` would print them into any log line or panic message.
impl std::fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let headers: Vec<&str> = self.headers.keys().map(String::as_str).collect();
        f.debug_struct("ModelConfig")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("reasoning", &self.reasoning)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .field("header_names", &headers)
            .finish_non_exhaustive()
    }
}

impl ModelConfig {
    /// A minimal config for tests. `provider` is `"mock"`, cost rates are all
    /// zero, and `base_url` points at a non-routable host.
    ///
    /// Use it **only** with
    /// [`Agent::from_provider`](crate::Agent::from_provider) /
    /// [`SubAgentTool::from_provider`](crate::SubAgentTool::from_provider) and a
    /// [`MockProvider`](crate::provider::MockProvider): those take the provider
    /// explicitly, so the config's protocol is never consulted.
    ///
    /// **Do not** pass it to
    /// [`Agent::from_config`](crate::Agent::from_config) — that dispatches on
    /// the protocol (here `AnthropicMessages`) and would build the **real**
    /// Anthropic provider pointed at the non-routable `base_url`, so the first
    /// prompt fails with a network error instead of returning a mock response.
    pub fn mock() -> Self {
        Self::custom(
            ApiProtocol::AnthropicMessages,
            "mock",
            "http://mock.invalid",
            "mock",
            "Mock",
        )
    }

    /// Create a config for any protocol without a dedicated preset
    /// (Bedrock, Vertex, Azure, or future protocols).
    ///
    /// Since `ModelConfig` is `#[non_exhaustive]`, this is the construction
    /// path when no `ModelConfig::*` preset fits. Defaults: 128K context,
    /// 16K max output, no compat flags — mutate fields to adjust.
    pub fn custom(
        api: ApiProtocol,
        provider: impl Into<String>,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            id: model_id.into(),
            name: name.into(),
            api,
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 16_000,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            compat: None,
            anthropic: None,
        }
    }

    /// Create a new Anthropic model config.
    pub fn anthropic(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::AnthropicMessages,
            provider: "anthropic".into(),
            base_url: "https://api.anthropic.com/v1".into(),
            reasoning: true,
            context_window: 200_000,
            max_tokens: 16_000,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: None,
        }
    }

    /// Claude Fable 5 — Anthropic's most capable model.
    /// 1M context; defaults to 64K of the model's 128K max output.
    ///
    /// Rates verified against <https://platform.claude.com/docs/en/about-claude/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    pub fn claude_fable_5() -> Self {
        Self {
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: CostConfig {
                input_per_million: 10.0,
                output_per_million: 50.0,
                cache_read_per_million: 1.0,
                cache_write_per_million: 12.5,
            },
            ..Self::anthropic("claude-fable-5", "Claude Fable 5")
        }
    }

    /// Claude Opus 5. 1M context; defaults to 64K of the model's 128K max output.
    ///
    /// Opus 5 thinks whenever a request omits `thinking`, so `ThinkingLevel::Off`
    /// does not disable thinking here — the provider omits the field rather than
    /// sending `{"type": "disabled"}`, and those tokens still count against
    /// `max_tokens`. Any other level takes the adaptive path that
    /// `AnthropicCompat::default()` selects, which Opus 5 accepts unchanged.
    ///
    /// Rates verified against <https://platform.claude.com/docs/en/about-claude/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    pub fn claude_opus_5() -> Self {
        Self {
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: CostConfig {
                input_per_million: 5.0,
                output_per_million: 25.0,
                cache_read_per_million: 0.5,
                cache_write_per_million: 6.25,
            },
            ..Self::anthropic("claude-opus-5", "Claude Opus 5")
        }
    }

    /// Claude Opus 4.8. 1M context; defaults to 64K of the model's 128K max output.
    ///
    /// Rates verified against <https://platform.claude.com/docs/en/about-claude/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    pub fn claude_opus_4_8() -> Self {
        Self {
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: CostConfig {
                input_per_million: 5.0,
                output_per_million: 25.0,
                cache_read_per_million: 0.5,
                cache_write_per_million: 6.25,
            },
            ..Self::anthropic("claude-opus-4-8", "Claude Opus 4.8")
        }
    }

    /// Claude Sonnet 5. 1M context; defaults to 64K of the model's 128K max output.
    ///
    /// Rates verified against <https://platform.claude.com/docs/en/about-claude/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    pub fn claude_sonnet_5() -> Self {
        Self {
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: CostConfig {
                input_per_million: 2.0,
                output_per_million: 10.0,
                cache_read_per_million: 0.2,
                cache_write_per_million: 2.5,
            },
            ..Self::anthropic("claude-sonnet-5", "Claude Sonnet 5")
        }
    }

    /// Claude Haiku 4.5. 200K context; defaults to 32K of the model's 64K max output.
    ///
    /// Rates verified against <https://platform.claude.com/docs/en/about-claude/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    pub fn claude_haiku_4_5() -> Self {
        Self {
            context_window: 200_000,
            max_tokens: 32_000,
            cost: CostConfig {
                input_per_million: 1.0,
                output_per_million: 5.0,
                cache_read_per_million: 0.1,
                cache_write_per_million: 1.25,
            },
            ..Self::anthropic("claude-haiku-4-5", "Claude Haiku 4.5")
        }
    }

    /// GPT-5.5. ~1M context; defaults to 64K of the model's 128K max output.
    /// Uses the Chat Completions API.
    ///
    /// Rates verified against <https://developers.openai.com/api/docs/pricing>
    /// on 2026-08-19. See [`CostConfig`] — they are a snapshot, not an authority.
    ///
    /// **Flat rate, tiered upstream.** OpenAI charges roughly double above a
    /// ~272K context tier ($10/$45/$1), and this preset declares a 1M window —
    /// so `cost_usd` understates long-context calls by up to 2x. `CostConfig`
    /// has no tier concept; override `cost` for long-context workloads.
    /// `tests/price_audit.rs` records this as a known gap.
    pub fn gpt_5_5() -> Self {
        Self {
            reasoning: true,
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: CostConfig {
                input_per_million: 5.0,
                output_per_million: 30.0,
                cache_read_per_million: 0.5,
                cache_write_per_million: 0.0,
            },
            ..Self::openai("gpt-5.5", "GPT-5.5")
        }
    }

    /// Create a new OpenAI model config.
    pub fn openai(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::openai()),
        }
    }

    /// Create a config for a local OpenAI-compatible server (LM Studio, Ollama, etc.).
    /// No API key required — sends an empty Bearer token.
    pub fn local(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            id: model_id.into(),
            name: "Local Model".into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "local".into(),
            base_url: base_url.into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::default()),
        }
    }

    /// Create a config for a model served by OpenCode Zen
    /// (<https://opencode.ai/docs/zen>), OpenCode's pay-per-use gateway.
    ///
    /// Zen serves each model family over a different protocol; the protocol is
    /// selected from the model id:
    /// - `gpt-*` → OpenAI Responses API (pair with `OpenAiResponsesProvider`)
    /// - `claude-*`, `qwen*` → Anthropic Messages API (pair with `AnthropicProvider`)
    /// - everything else (DeepSeek, MiniMax, GLM, Kimi, ...) → Chat Completions
    ///   (pair with `OpenAiCompatProvider`)
    ///
    /// Gemini models are not supported — Zen serves them over a Google-native
    /// endpoint shape yoagent does not target. A `gemini-*` id falls through to
    /// Chat Completions (with a warning) and will likely fail at request time.
    ///
    /// The routing mirrors the Zen endpoint tables as of mid-2026; if a model
    /// errors, verify its protocol against `https://opencode.ai/zen/v1/models`.
    ///
    /// Context window and max output default conservatively (128K / 16K);
    /// override the fields for models with larger limits.
    pub fn opencode_zen(model_id: impl Into<String>) -> Self {
        Self::opencode(model_id.into(), OpenCodeGateway::Zen)
    }

    /// Create a config for a model served by OpenCode Go
    /// (<https://opencode.ai/docs/go>), OpenCode's subscription gateway for
    /// open models.
    ///
    /// Protocol is selected from the model id:
    /// - `qwen*`, `minimax-*` → Anthropic Messages API (pair with `AnthropicProvider`)
    /// - everything else (GLM, Kimi, DeepSeek, MiMo, ...) → Chat Completions
    ///   (pair with `OpenAiCompatProvider`)
    pub fn opencode_go(model_id: impl Into<String>) -> Self {
        Self::opencode(model_id.into(), OpenCodeGateway::Go)
    }

    fn opencode(id: String, gateway: OpenCodeGateway) -> Self {
        let lower = id.to_ascii_lowercase();
        if lower.starts_with("gemini-") {
            tracing::warn!(
                "OpenCode serves Gemini models over a Google-native endpoint yoagent \
                 does not target; '{}' is routed to /chat/completions and will likely \
                 fail at request time",
                id
            );
        }
        let anthropic_protocol = match gateway {
            OpenCodeGateway::Zen => lower.starts_with("claude-") || lower.starts_with("qwen"),
            OpenCodeGateway::Go => lower.starts_with("qwen") || lower.starts_with("minimax-"),
        };
        let (api, reasoning, compat, anthropic) = if anthropic_protocol {
            (
                ApiProtocol::AnthropicMessages,
                true,
                None,
                // Gateways use OpenAI-style Bearer auth, not x-api-key.
                Some(AnthropicCompat {
                    adaptive_thinking: true,
                    bearer_auth: true,
                }),
            )
        } else if gateway == OpenCodeGateway::Zen && lower.starts_with("gpt-") {
            (ApiProtocol::OpenAiResponses, true, None, None)
        } else {
            (
                ApiProtocol::OpenAiCompletions,
                false,
                Some(OpenAiCompat::default()),
                None,
            )
        };
        Self {
            id: id.clone(),
            name: id,
            api,
            provider: gateway.provider_name().into(),
            base_url: gateway.base_url().into(),
            reasoning,
            context_window: 128_000,
            max_tokens: 16_000,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            compat,
            anthropic,
        }
    }

    /// Create a config for a custom OpenAI-compatible endpoint with explicit compat flags.
    pub fn openai_compat(
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        provider: impl Into<String>,
        compat: OpenAiCompat,
    ) -> Self {
        let id = model_id.into();
        Self {
            id: id.clone(),
            name: id,
            api: ApiProtocol::OpenAiCompletions,
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(compat),
        }
    }

    /// Create a config for Ollama's OpenAI-compatible API.
    ///
    /// Default local base URL: `http://localhost:11434/v1`.
    pub fn ollama(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        let id = model_id.into();
        Self {
            id: id.clone(),
            name: id,
            api: ApiProtocol::OpenAiCompletions,
            provider: "ollama".into(),
            base_url: base_url.into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::ollama()),
        }
    }

    /// Create a new Z.ai (Zhipu AI) model config.
    ///
    /// Models: `glm-4.7`, `glm-4.5-air`, `glm-5`, etc.
    pub fn zai(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "zai".into(),
            base_url: "https://api.z.ai/api/paas/v4".into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::zai()),
        }
    }

    /// Create a new Meta Model API config (Muse Spark).
    ///
    /// Models: `muse-spark-1.1` — 1,048,576-token context; 128K max output
    /// per Meta's integration examples (no official model card yet).
    /// US-only public preview as of July 2026. OpenAI-compatible endpoint at
    /// `https://api.meta.ai/v1`. Key resolves from `META_API_KEY`, then
    /// Meta's documented `MODEL_API_KEY`.
    ///
    /// Reasoning: Meta's endpoint defaults to `reasoning_effort: medium`
    /// server-side. Set a [`ThinkingLevel`](crate::types::ThinkingLevel) to
    /// tune it; `Off` omits the field, which means Meta's default (medium)
    /// applies — not "no reasoning".
    ///
    /// Rates are Muse Spark 1.1/1.2, verified 2026-08-19. This constructor is
    /// generic over the model id, so a different tier needs `config.cost`
    /// overridden: the contributor tier runs 12x lower on input, 21x on output
    /// and 75x on cache reads, so `ModelConfig::meta("muse-spark-1.2-contributor", ..)`
    /// overstates cost badly. See [`CostConfig`].
    pub fn meta(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "meta".into(),
            base_url: "https://api.meta.ai/v1".into(),
            reasoning: true,
            context_window: 1_048_576,
            max_tokens: 131_072,
            cost: CostConfig {
                input_per_million: 1.25,
                output_per_million: 4.25,
                cache_read_per_million: 0.15,
                cache_write_per_million: 0.0,
            },
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::meta()),
        }
    }

    /// Create a new MiniMax model config.
    ///
    /// Models: `MiniMax-Text-01`, `MiniMax-M1`, etc.
    pub fn minimax(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "minimax".into(),
            base_url: "https://api.minimaxi.chat/v1".into(),
            reasoning: false,
            context_window: 1_000_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::minimax()),
        }
    }

    /// Create a new Qwen / DashScope model config.
    ///
    /// Models: `qwen3.6-plus`, `qwen3.5-plus`, `qwen-plus`, `qwen-flash`, etc.
    pub fn qwen(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "qwen".into(),
            base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".into(),
            reasoning: true,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::qwen()),
        }
    }

    /// Create a new xAI (Grok) model config.
    ///
    /// Models: `grok-4-1-fast`, `grok-4-1`, etc.
    pub fn xai(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "xai".into(),
            base_url: "https://api.x.ai/v1".into(),
            reasoning: false,
            context_window: 131_072,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::xai()),
        }
    }

    /// Create a new Groq model config.
    ///
    /// Models: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, etc.
    pub fn groq(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "groq".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::groq()),
        }
    }

    /// Create a new DeepSeek model config.
    ///
    /// Models: `deepseek-v4-flash`, `deepseek-v4-pro`, etc.
    ///
    /// Legacy aliases `deepseek-chat` and `deepseek-reasoner` are accepted by
    /// DeepSeek for now, but are scheduled for deprecation on 2026-07-24.
    pub fn deepseek(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            reasoning: true,
            context_window: 1_000_000,
            max_tokens: 384_000,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::deepseek()),
        }
    }

    /// Create a new Mistral model config.
    ///
    /// Models: `mistral-large-latest`, `mistral-small-latest`, etc.
    pub fn mistral(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::OpenAiCompletions,
            provider: "mistral".into(),
            base_url: "https://api.mistral.ai/v1".into(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: Some(OpenAiCompat::mistral()),
        }
    }

    /// Create a new Google Generative AI (Gemini) model config.
    pub fn google(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: ApiProtocol::GoogleGenerativeAi,
            provider: "google".into(),
            base_url: "https://generativelanguage.googleapis.com".into(),
            reasoning: false,
            context_window: 1_000_000,
            max_tokens: 8192,
            cost: CostConfig::default(),
            headers: HashMap::new(),
            anthropic: None,
            compat: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_preset_matches_launch_specs() {
        let mc = ModelConfig::meta("muse-spark-1.1", "Muse Spark 1.1");
        assert_eq!(mc.provider, "meta");
        assert_eq!(mc.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(mc.base_url, "https://api.meta.ai/v1");
        assert_eq!(mc.context_window, 1_048_576);
        assert_eq!(mc.max_tokens, 131_072);
        assert!(mc.cost.is_configured());
        assert_eq!(mc.cost.input_per_million, 1.25);
        assert_eq!(mc.cost.output_per_million, 4.25);
        // Meta documents a cached-input rate; cache writes are not charged.
        assert_eq!(mc.cost.cache_read_per_million, 0.15);
        assert_eq!(mc.cost.cache_write_per_million, 0.0);
        let compat = mc.compat.expect("compat flags set");
        assert!(matches!(
            compat.max_tokens_field,
            MaxTokensField::MaxCompletionTokens
        ));
        // Documented in Meta's chat-completions schemas.
        assert!(compat.supports_reasoning_effort);
        assert!(compat.supports_usage_in_streaming);
    }

    #[test]
    fn test_model_config_anthropic() {
        let config = ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
        assert_eq!(config.api, ApiProtocol::AnthropicMessages);
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.base_url, "https://api.anthropic.com/v1");
        assert!(config.compat.is_none());
        assert!(config.anthropic.is_none());
    }

    #[test]
    fn test_cost_usd() {
        let cost = CostConfig {
            input_per_million: 3.0,
            output_per_million: 15.0,
            cache_read_per_million: 0.3,
            cache_write_per_million: 3.75,
        };
        let usage = crate::types::Usage {
            input: 1_000_000,
            output: 100_000,
            cache_read: 2_000_000,
            cache_write: 400_000,
            total_tokens: 0,
        };
        // 3.0 + 1.5 + 0.6 + 1.5 = 6.6
        assert!((cost.cost_usd(&usage) - 6.6).abs() < 1e-9);
        // zero rates (default) => zero cost
        assert_eq!(CostConfig::default().cost_usd(&usage), 0.0);
    }

    #[test]
    fn test_new_generation_presets() {
        let fable = ModelConfig::claude_fable_5();
        assert_eq!(fable.id, "claude-fable-5");
        assert_eq!(fable.api, ApiProtocol::AnthropicMessages);
        assert_eq!(fable.context_window, 1_000_000);
        assert_eq!(fable.cost.input_per_million, 10.0);
        assert_eq!(fable.cost.output_per_million, 50.0);

        let opus_5 = ModelConfig::claude_opus_5();
        assert_eq!(opus_5.id, "claude-opus-5");
        assert_eq!(opus_5.api, ApiProtocol::AnthropicMessages);
        assert_eq!(opus_5.context_window, 1_000_000);
        assert_eq!(opus_5.max_tokens, 64_000);
        assert_eq!(opus_5.cost.input_per_million, 5.0);
        assert_eq!(opus_5.cost.output_per_million, 25.0);
        // Derived rates: cache reads bill at 0.1x input, writes at 1.25x.
        assert_eq!(opus_5.cost.cache_read_per_million, 0.5);
        assert_eq!(opus_5.cost.cache_write_per_million, 6.25);

        let opus = ModelConfig::claude_opus_4_8();
        assert_eq!(opus.id, "claude-opus-4-8");
        assert_eq!(opus.context_window, 1_000_000);
        assert_eq!(opus.cost.input_per_million, 5.0);

        let sonnet = ModelConfig::claude_sonnet_5();
        assert_eq!(sonnet.id, "claude-sonnet-5");
        assert_eq!(sonnet.cost.output_per_million, 10.0);

        let haiku = ModelConfig::claude_haiku_4_5();
        assert_eq!(haiku.id, "claude-haiku-4-5");
        assert_eq!(haiku.context_window, 200_000);

        let gpt = ModelConfig::gpt_5_5();
        assert_eq!(gpt.id, "gpt-5.5");
        assert_eq!(gpt.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(gpt.context_window, 1_000_000);
        assert_eq!(gpt.cost.output_per_million, 30.0);
        assert!(gpt.compat.is_some());
    }

    #[test]
    fn test_opencode_zen_protocol_selection() {
        // GPT models → Responses API
        let gpt = ModelConfig::opencode_zen("gpt-5.5");
        assert_eq!(gpt.api, ApiProtocol::OpenAiResponses);
        assert_eq!(gpt.provider, "opencode-zen");
        assert_eq!(gpt.base_url, "https://opencode.ai/zen/v1");

        // Claude and Qwen models → Anthropic Messages with Bearer auth
        for id in ["claude-sonnet-5", "qwen3.7-max"] {
            let config = ModelConfig::opencode_zen(id);
            assert_eq!(config.api, ApiProtocol::AnthropicMessages, "{id}");
            let compat = config.anthropic.expect("anthropic compat set");
            assert!(compat.bearer_auth);
        }

        // Everything else → Chat Completions
        for id in ["deepseek-v4-pro", "minimax-m3", "glm-5.2", "kimi-k2.7-code"] {
            let config = ModelConfig::opencode_zen(id);
            assert_eq!(config.api, ApiProtocol::OpenAiCompletions, "{id}");
            assert!(config.compat.is_some());
        }
    }

    #[test]
    fn test_opencode_go_protocol_selection() {
        // Qwen and MiniMax models → Anthropic Messages with Bearer auth
        for id in ["qwen3.7-max", "minimax-m3"] {
            let config = ModelConfig::opencode_go(id);
            assert_eq!(config.api, ApiProtocol::AnthropicMessages, "{id}");
            assert_eq!(config.base_url, "https://opencode.ai/zen/go/v1");
            assert!(config.anthropic.expect("anthropic compat set").bearer_auth);
        }

        // Everything else → Chat Completions (Go has no GPT models)
        for id in [
            "glm-5.2",
            "kimi-k2.7-code",
            "deepseek-v4-flash",
            "mimo-v2.5",
        ] {
            let config = ModelConfig::opencode_go(id);
            assert_eq!(config.api, ApiProtocol::OpenAiCompletions, "{id}");
            assert_eq!(config.provider, "opencode-go", "{id}");
        }
    }

    #[test]
    fn test_model_config_openai() {
        let config = ModelConfig::openai("gpt-4o", "GPT-4o");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        let compat = config.compat.unwrap();
        assert!(compat.supports_store);
        assert!(compat.supports_developer_role);
        assert_eq!(compat.max_tokens_field, MaxTokensField::MaxCompletionTokens);
    }

    #[test]
    fn test_openai_compat_variants() {
        let xai = OpenAiCompat::xai();
        assert_eq!(xai.thinking_format, ThinkingFormat::Xai);
        assert!(!xai.supports_store);

        let groq = OpenAiCompat::groq();
        assert!(groq.supports_usage_in_streaming);
        assert!(!groq.supports_store);

        let deepseek = OpenAiCompat::deepseek();
        assert_eq!(deepseek.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(deepseek.supports_reasoning_effort);
        assert!(deepseek.supports_thinking_control);

        let zai = OpenAiCompat::zai();
        assert!(zai.supports_usage_in_streaming);
        assert!(!zai.supports_store);

        let minimax = OpenAiCompat::minimax();
        assert!(minimax.supports_usage_in_streaming);
        assert!(!minimax.supports_store);

        let ollama = OpenAiCompat::ollama();
        assert!(ollama.requires_assistant_after_tool_result);
        assert!(!ollama.requires_tool_result_name);

        let qwen = OpenAiCompat::qwen();
        assert_eq!(qwen.thinking_format, ThinkingFormat::Qwen);
        assert_eq!(qwen.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(qwen.supports_usage_in_streaming);
        assert!(!qwen.supports_reasoning_effort);
        assert!(!qwen.supports_thinking_control);
    }

    #[test]
    fn test_model_config_deserializes_without_anthropic_field() {
        // Configs persisted before 0.9.0 have no `anthropic` field.
        let mut value = serde_json::to_value(ModelConfig::anthropic("m", "M")).unwrap();
        value.as_object_mut().unwrap().remove("anthropic");
        let config: ModelConfig = serde_json::from_value(value).unwrap();
        assert!(config.anthropic.is_none());
    }

    #[test]
    fn test_anthropic_compat_deserializes_from_partial_json() {
        // Container-level serde(default): missing fields use Default (adaptive on).
        let compat: AnthropicCompat = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(compat.adaptive_thinking);
        assert!(!compat.bearer_auth);

        let compat: AnthropicCompat =
            serde_json::from_value(serde_json::json!({"bearer_auth": true})).unwrap();
        assert!(compat.adaptive_thinking);
        assert!(compat.bearer_auth);
    }

    #[test]
    fn test_openai_compat_deserializes_without_assistant_after_tool_result_flag() {
        let compat: OpenAiCompat = serde_json::from_value(serde_json::json!({
            "supports_store": false,
            "supports_developer_role": false,
            "supports_reasoning_effort": false,
            "supports_thinking_control": false,
            "supports_usage_in_streaming": true,
            "max_tokens_field": "max_tokens",
            "requires_tool_result_name": false,
            "thinking_format": "open_ai"
        }))
        .unwrap();

        assert!(!compat.requires_assistant_after_tool_result);
    }

    #[test]
    fn test_model_config_local_remains_neutral() {
        let config = ModelConfig::local("http://localhost:1234/v1", "local-model");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "local");
        assert_eq!(config.base_url, "http://localhost:1234/v1");
        let compat = config.compat.unwrap();
        assert!(!compat.requires_assistant_after_tool_result);
    }

    #[test]
    fn test_model_config_ollama() {
        let config = ModelConfig::ollama("http://localhost:11434/v1", "llama3.1:8b");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.id, "llama3.1:8b");
        assert_eq!(config.name, "llama3.1:8b");
        assert_eq!(config.base_url, "http://localhost:11434/v1");
        let compat = config.compat.unwrap();
        assert!(compat.requires_assistant_after_tool_result);
    }

    #[test]
    fn test_model_config_openai_compat() {
        let config = ModelConfig::openai_compat(
            "http://localhost:1234/v1",
            "qwen3-local",
            "qwen",
            OpenAiCompat::qwen(),
        );
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "qwen");
        assert_eq!(config.id, "qwen3-local");
        assert_eq!(config.name, "qwen3-local");
        assert_eq!(config.base_url, "http://localhost:1234/v1");
        let compat = config.compat.unwrap();
        assert_eq!(compat.thinking_format, ThinkingFormat::Qwen);
    }

    #[test]
    fn test_model_config_qwen() {
        let config = ModelConfig::qwen("qwen3.6-plus", "Qwen 3.6 Plus");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "qwen");
        assert_eq!(
            config.base_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        );
        assert!(config.reasoning);
        let compat = config.compat.unwrap();
        assert_eq!(compat.thinking_format, ThinkingFormat::Qwen);
        assert_eq!(compat.max_tokens_field, MaxTokensField::MaxTokens);
    }

    #[test]
    fn test_model_config_zai() {
        let config = ModelConfig::zai("glm-4.7", "GLM 4.7");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "zai");
        assert_eq!(config.base_url, "https://api.z.ai/api/paas/v4");
        assert!(config.compat.is_some());
    }

    #[test]
    fn test_model_config_minimax() {
        let config = ModelConfig::minimax("MiniMax-Text-01", "MiniMax Text 01");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "minimax");
        assert_eq!(config.base_url, "https://api.minimaxi.chat/v1");
        assert_eq!(config.context_window, 1_000_000);
        assert!(config.compat.is_some());
    }

    #[test]
    fn test_model_config_deepseek() {
        let config = ModelConfig::deepseek("deepseek-v4-flash", "DeepSeek V4 Flash");
        assert_eq!(config.api, ApiProtocol::OpenAiCompletions);
        assert_eq!(config.provider, "deepseek");
        assert_eq!(config.base_url, "https://api.deepseek.com");
        assert_eq!(config.context_window, 1_000_000);
        assert_eq!(config.max_tokens, 384_000);
        assert!(config.reasoning);
        assert!(config.compat.is_some());
    }

    #[test]
    fn test_api_protocol_display() {
        assert_eq!(
            ApiProtocol::AnthropicMessages.to_string(),
            "anthropic_messages"
        );
        assert_eq!(
            ApiProtocol::OpenAiCompletions.to_string(),
            "openai_completions"
        );
        assert_eq!(
            ApiProtocol::GoogleGenerativeAi.to_string(),
            "google_generative_ai"
        );
    }

    #[test]
    fn test_cost_config_default() {
        let cost = CostConfig::default();
        assert_eq!(cost.input_per_million, 0.0);
        assert_eq!(cost.output_per_million, 0.0);
    }
}
