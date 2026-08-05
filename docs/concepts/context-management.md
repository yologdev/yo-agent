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

### Bounding tool output: two layers

The largest source of cache loss is *retroactive* truncation — output is sent in full, cached, then rewritten by Level 1 in one sweep once the session goes over budget. Bounding it as it arrives fixes that, but a single global cap is the wrong instrument, because tools differ in how well they survive being cut:

**Command output** (`bash`, `search`) takes a head+tail cut well: the first error is at the top, the summary at the bottom, and the middle is repetition. `tool_output_max_lines` (200) is applied on append, controlled by `truncate_tool_output_on_append` (on by default).

**File reads** do not. The middle of a source file is usually the part that was asked for, so head+tail removes exactly the wrong lines. `read_file` bounds itself instead, by paging: it returns `DEFAULT_READ_MAX_LINES` (500) at a time with a header stating the true total, and the agent asks for the next range with `offset`/`limit`. That bound is lossless and directed. `read_file` is therefore exempt from head+tail truncation by default, via `tool_output_max_lines_overrides`.

Custom tools pick their own budget the same way:

```rust
let mut config = ContextConfig::from_context_window(128_000);
config.tool_output_max_lines_overrides.insert("my_paging_tool".into(), usize::MAX);
config.tool_output_max_lines_overrides.insert("noisy_tool".into(), 40);
```

To restore pre-0.15 behaviour, set `truncate_tool_output_on_append: false` and construct `ReadFileTool { max_lines: usize::MAX, ..Default::default() }`.

### Measured effect

`tests/context_cache_test.rs` replays 300 turns on a 128K window. The tool mix (bash 41%, edit/write 36%, read 19%, search 4%) and file-size distribution come from 808 archived runs of a production agent built on yoagent.

| | prefix-cache hit rate | history rewrites |
|---|---|---|
| 0.14.2 | 93.58% | 23 |
| 0.15.0, 0.14.2-equivalent settings | 94.38% | 22 |
| 0.15.0 defaults | **96.04%** | **8** |

Read page size is the dominant remaining lever — hit rate is flat between a 300- and 500-line page (96.2% / 96.0%) and falls to 94.2% at 1000 and 92.9% at 2000. 500 is the largest page that costs nothing extra. Lowering it to 200 buys about a further 1.2 points, at the cost of more read round-trips.

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
