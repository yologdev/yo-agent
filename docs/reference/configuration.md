# Configuration

## AgentLoopConfig

The main configuration for the agent loop:

```rust
pub struct AgentLoopConfig {
    pub provider: Arc<dyn StreamProvider>,
    pub model: String,
    pub api_key: String,
    pub thinking_level: ThinkingLevel,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub model_config: Option<ModelConfig>,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub get_steering_messages: Option<GetMessagesFn>,
    pub get_follow_up_messages: Option<GetMessagesFn>,
    pub context_config: Option<ContextConfig>,
    pub compaction_strategy: Option<Arc<dyn CompactionStrategy>>,
    pub execution_limits: Option<ExecutionLimits>,
    pub cache_config: CacheConfig,
    pub tool_execution: ToolExecutionStrategy,
    pub retry_config: RetryConfig,
    pub before_turn: Option<BeforeTurnFn>,
    pub after_turn: Option<AfterTurnFn>,
    pub on_error: Option<OnErrorFn>,
    pub input_filters: Vec<Arc<dyn InputFilter>>,
    pub turn_delay: Option<Duration>,
}
```

## StreamConfig

Passed to `StreamProvider::stream()`:

```rust
pub struct StreamConfig {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub thinking_level: ThinkingLevel,
    pub api_key: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub model_config: Option<ModelConfig>,
    pub cache_config: CacheConfig,
}
```

## ContextConfig

Controls context window compaction:

```rust
pub struct ContextConfig {
    pub max_context_tokens: usize,                            // Default: 100,000
    pub system_prompt_tokens: usize,                          // Default: 4,000
    pub keep_recent: usize,                                   // Default: 10
    pub keep_first: usize,                                    // Default: 2
    pub tool_output_max_lines: usize,                         // Default: 200
    pub tool_output_max_lines_overrides: HashMap<String, usize>, // Default: {"read_file": MAX}
    pub compact_target_ratio: f32,                            // Default: 0.7
    pub truncate_tool_output_on_append: bool,                 // Default: true
}
```

`compact_target_ratio` is the headroom compaction leaves behind: it triggers at the full budget but reduces to `budget × ratio`, so history is not rewritten every turn once the budget is crossed. `truncate_tool_output_on_append` caps tool output as it enters the context rather than retroactively, and `tool_output_max_lines_overrides` gives per-tool budgets so a tool that head+tail would damage (a paging reader) can opt out. All three exist to keep the provider's prefix cache intact — see [Context Management](../concepts/context-management.md#prefix-cache-stability).

When `context_config` is not explicitly set, it is automatically derived from `ModelConfig.context_window` (80% for context, 20% reserved for output). If neither is set, `ContextConfig::default()` (100K) is used.

```rust
// Derive from a model's context window:
let config = ContextConfig::from_context_window(200_000);
// config.max_context_tokens == 160_000
```

## ExecutionLimits

Prevents runaway agents:

```rust
pub struct ExecutionLimits {
    pub max_turns: usize,              // Default: 50
    pub max_total_tokens: usize,       // Default: 1,000,000
    pub max_duration: Duration,        // Default: 600s
}
```

## ThinkingLevel

```rust
pub enum ThinkingLevel {
    Off,        // No thinking (default)
    Minimal,    // Anthropic: effort "low" (adaptive) / 1,024-token budget (legacy)
    Low,        // Anthropic: effort "low" / 1,024
    Medium,     // Anthropic: effort "medium" / 2,048
    High,       // Anthropic: effort "high" / 8,192
}
```

OpenAI-family providers map these levels to `reasoning_effort` where the
compat flags enable it; the Google and Bedrock providers currently ignore
`thinking_level`.

## CostConfig

Token pricing per million:

```rust
pub struct CostConfig {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}
```

## ModelConfig Presets

yoagent provides first-class `ModelConfig::*` constructors for Anthropic, OpenAI, Google Gemini, xAI, Groq, DeepSeek, Mistral, MiniMax, Z.ai, Qwen, Ollama, and local OpenAI-compatible servers.

See [Model Presets](../providers/model-presets.md) for the full table of constructors, default base URLs, context windows, and DeepSeek legacy alias notes.
