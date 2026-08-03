# Context Management

Long-running agents accumulate messages that exceed the model's context window. yoagent provides token tracking, overflow detection, tiered compaction, and execution limits.

## Token Estimation

Fast estimation without external tokenizer dependencies:

```rust
use yoagent::context::{estimate_tokens, message_tokens, total_tokens};

estimate_tokens("Hello world");          // ~3 tokens (chars / 4)
message_tokens(&agent_message);          // estimate for a single message
total_tokens(&messages);                 // estimate for all messages
```

## Context Tracking

`ContextTracker` combines real token counts from provider responses with estimation for new messages — more accurate than pure estimation:

```rust
use yoagent::context::ContextTracker;

let mut tracker = ContextTracker::new();

// After each assistant response, record the real usage:
tracker.record_usage(&assistant_usage, message_index);

// Get current context size (real usage + estimated trailing):
let tokens = tracker.estimate_context_tokens(agent.messages());

// After compaction, reset the tracker:
tracker.reset();
```

When no usage data is available, it falls back to chars/4 estimation.

## Context Overflow Detection

When the context exceeds a model's window, providers return overflow errors. yoagent detects these automatically across all major providers.

### HTTP-level detection

Providers that check before streaming (Google, Bedrock, Vertex) return `ProviderError::ContextOverflow`:

```rust
use yoagent::provider::ProviderError;

match agent.prompt("...").await {
    // The loop already handles this — but you can also match it:
    Err(ProviderError::ContextOverflow { message }) => {
        // Compact and retry
    }
    _ => {}
}
```

`ProviderError::classify()` auto-detects overflow from error messages covering Anthropic, OpenAI, Google, AWS Bedrock, xAI, Groq, OpenRouter, llama.cpp, LM Studio, MiniMax, Kimi, GitHub Copilot, and generic patterns.

### Message-level detection

SSE-based providers (Anthropic, OpenAI) return overflow as a `StopReason::Error` message. Check with:

```rust
if message.is_context_overflow() {
    // Compact and retry
}
```

### Handling overflow in your application

yoagent provides the detection and building blocks. Your application wires the compaction strategy:

```rust
// Proactive: check before each prompt
let tokens = tracker.estimate_context_tokens(agent.messages());
if tokens > context_window - reserve {
    let compacted = compact_messages(agent.messages().to_vec(), &config);
    agent.replace_messages(compacted);
}

// Reactive: catch overflow errors
// ... on ContextOverflow or message.is_context_overflow():
//   compact, then retry with agent.continue_loop()
```

For LLM-based summarization (asking the model to summarize old messages), implement that in your application layer — yoagent provides `replace_messages()` and `compact_messages()` as building blocks.

## ContextConfig

```rust
pub struct ContextConfig {
    pub max_context_tokens: usize,               // Default: 100,000
    pub system_prompt_tokens: usize,             // Default: 4,000
    pub keep_recent: usize,                      // Default: 10
    pub keep_first: usize,                       // Default: 2
    pub tool_output_max_lines: usize,            // Default: 50
    pub compact_target_ratio: f32,               // Default: 0.7
    pub truncate_tool_output_on_append: bool,    // Default: false
}
```

### Auto-Derivation from ModelConfig

When you set a `ModelConfig` but don't explicitly set a `ContextConfig`, the compaction budget is automatically derived from the model's `context_window` — reserving 80% for context and 20% for output:

```rust
// MiniMax with 1M context → compacts at 800K (no manual config needed)
let agent = Agent::from_config(ModelConfig::minimax("MiniMax-Text-01", "MiniMax Text 01"));

// Anthropic with 200K context → compacts at 160K
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"));
```

The priority chain:
1. Explicit `with_context_config(...)` → always wins
2. Has `model_config` → auto-derives from `context_window` (80%)
3. Neither → `ContextConfig::default()` (100K)

You can also derive manually:

```rust
let config = ContextConfig::from_context_window(1_000_000);
// config.max_context_tokens == 800_000
```

## Tiered Compaction

`compact_messages()` tries each level in order, stopping as soon as messages fit the budget:

### Level 1: Truncate Tool Outputs

Replaces long tool outputs with head + tail, fitting the result into exactly `tool_output_max_lines` lines (the `[... N lines truncated ...]` marker is charged against that budget). This is the cheapest level — it preserves conversation structure and typically saves 50-70% in coding sessions.

Truncation is idempotent: re-running it on an already-truncated output returns it byte for byte, so a session that sits above the budget does not re-truncate the same outputs turn after turn.

### Level 2: Summarize Old Turns

Keeps the last `keep_recent` messages in full detail. Older assistant messages are replaced with one-line summaries like `"[Summary] [Assistant used 3 tool(s)]"`, and their tool results are dropped. The boundary is pulled back to a turn start so an assistant message and its tool results are never split across it.

### Level 3: Drop Middle Messages

Drops the smallest span of middle messages that reaches the target, keeping at least `keep_first` from the start and `keep_recent` from the end. A constant marker message stands in for what was removed; the count goes to the debug log rather than into the marker, so the text does not change from pass to pass.

## Prefix Cache Stability

Providers cache request prefixes — automatically on DeepSeek, explicitly via `cache_control` on Anthropic. A cache hit needs the new request to share a byte-identical prefix with the last one, so **every rewrite of already-sent history costs full price for every token from the rewrite point onward**.

Compaction is built around that:

- Levels are idempotent where they can be, so re-running on settled history changes nothing.
- `compact_target_ratio` gives compaction headroom. It still *triggers* at the full budget, but the lossy levels reduce to `budget × ratio` (default 70%) rather than to whatever just barely fits — otherwise the next turn crosses the budget again and history is rewritten every single turn.
- Markers and generated summaries carry no wall-clock timestamps or drifting counts, so the same history always compacts to the same bytes.

### truncate_tool_output_on_append

The largest remaining source of cache loss is *retroactive* truncation: a tool output is sent in full, cached, and then rewritten by Level 1 once the session goes over budget. Setting `truncate_tool_output_on_append` applies `tool_output_max_lines` when the tool result is appended instead, so the bytes the provider cached are the bytes that stay:

```rust
let agent = Agent::from_config(ModelConfig::deepseek("deepseek-chat", "DeepSeek Chat"))
    .with_context_config(ContextConfig {
        truncate_tool_output_on_append: true,
        ..ContextConfig::from_context_window(128_000)
    });
```

The trade-off is that a long tool output is trimmed even when the budget had room for it, which is why it is off by default. The untruncated output is always available in the `AgentEvent` stream either way.

Measured over a 300-turn synthetic tool-heavy session on a 128K window (`tests/context_cache_test.rs`):

| | prefix-cache hit rate | history rewrites |
|---|---|---|
| 0.14.2 | 95.80% | 16 |
| 0.15.0 defaults | 96.85% | 15 |
| 0.15.0 + `truncate_tool_output_on_append` | 98.51% | 2 |

## ExecutionLimits

Prevents runaway agents:

```rust
pub struct ExecutionLimits {
    pub max_turns: usize,              // Default: 50
    pub max_total_tokens: usize,       // Default: 1,000,000
    pub max_duration: Duration,        // Default: 600s (10 min)
}
```

When a limit is reached, the agent stops with a message like `"[Agent stopped: Max turns reached (50/50)]"`.

## Disabling Context Management

```rust
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .without_context_management();
```

This sets both `context_config` and `execution_limits` to `None`.
