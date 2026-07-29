# Anthropic Provider

`AnthropicProvider` implements the Anthropic Messages API with SSE streaming.

## Usage

```rust
use yoagent::provider::ModelConfig;

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"));
```

## Features

### Streaming SSE

Uses `reqwest-eventsource` to parse Anthropic's SSE stream. Events handled:

- `message_start` — Input token usage, cache stats
- `content_block_start` — Text, thinking, or tool_use block
- `content_block_delta` — Text, thinking, input JSON, or signature deltas
- `content_block_stop` — Block complete
- `message_delta` — Stop reason, output usage
- `message_stop` — Stream complete

### Thinking

Set `thinking_level` to enable thinking. By default the provider sends
**adaptive thinking** (`thinking: {"type": "adaptive"}`), which the current
model generation requires (Claude Fable 5, Opus 5, Opus 4.7/4.8, Sonnet 5
reject budget-based thinking with a 400). The level maps to an `output_config.effort`
hint:

| Level | Effort |
|-------|--------|
| `Minimal`, `Low` | `low` |
| `Medium` | `medium` |
| `High` | `high` |

For pre-4.6 models, opt into legacy budget-based thinking via
`AnthropicCompat::legacy()`:

```rust
let mut config = ModelConfig::anthropic("claude-sonnet-4-5", "Claude Sonnet 4.5");
config.anthropic = Some(AnthropicCompat::legacy());
```

Legacy budgets: `Minimal`/`Low` 1,024 (the API minimum), `Medium` 2,048,
`High` 8,192. `max_tokens` is automatically raised above the budget when
needed.

Thinking content is streamed as `Content::Thinking` with a cryptographic `signature` for verification.

### Stop Reasons

Every documented Anthropic `stop_reason` maps explicitly:

| Wire value | `StopReason` | Notes |
|---|---|---|
| `end_turn`, `stop_sequence` | `Stop` | |
| `tool_use` | `ToolUse` | |
| `max_tokens` | `Length` | |
| `refusal` | `Refusal` | Sets `error_message`; see below |
| `model_context_window_exceeded` | `Error` | In-stream overflow; keeps the phrase `Message::is_context_overflow()` matches, so compaction-retry hooks still fire |
| `pause_turn` | `Error` | The model stopped mid-turn expecting the conversation to be re-sent. This transport cannot resume, so reporting it as a normal stop would return a truncated answer as though it were complete |

Anything unrecognized maps to `Stop` and is logged at `warn`, so a stop reason
added by Anthropic later is visible rather than silently treated as a finish.

**Refusals.** Models with safety classifiers (e.g. Claude Fable 5) can decline a
request with `stop_reason: "refusal"`. The agent loop stops the turn like a
normal `Stop`, and callers can match on the variant to retry on a fallback model.

### Tool Calls With Unusable Arguments

A tool call's arguments arrive as `input_json_delta` fragments and are assembled
at `content_block_stop`. When that assembly cannot happen — the accumulated text
is not valid JSON, or the `content_block_stop` event itself is unusable — the
turn fails with `StopReason::Error` and an `error_message` naming each affected
tool and quoting its input.

This matters because the alternative is silent: a tool executed with empty
arguments falls back to its defaults, so `list_files` asked for `/etc` would
list the working directory instead, with nothing in the response indicating the
model's actual input was dropped.

`agent_loop` returns on `StopReason::Error` before extracting tool calls, so
nothing executes. Every tool call in the message is replaced with a text block —
not just the unusable one — because the turn runs none of them, and a `tool_use`
block with no matching `tool_result` is rejected by the API on the *next*
request.

### Cache Control

Automatic prompt caching via `cache_control` markers:

- **System prompt**: Always cached with `{"type": "ephemeral"}`
- **Second-to-last message**: Gets `cache_control` on its last content block, creating a cache breakpoint

This means on repeated calls, only the latest message is processed at full price.

## Configuration

| Setting | Value |
|---------|-------|
| API URL | `{base_url}/messages` (default `https://api.anthropic.com/v1/messages`) |
| API Version | `2023-06-01` |
| Auth Header | `x-api-key` (or `Authorization: Bearer` with `AnthropicCompat { bearer_auth: true }` / a custom `authorization` header in `ModelConfig.headers`) |
| Default Max Tokens | request `max_tokens`, else `ModelConfig.max_tokens`, else 8,192 |

Setting `ModelConfig.base_url` retargets the provider at any gateway that
speaks the Anthropic Messages protocol (e.g. OpenCode Zen/Go — see
[OpenCode Zen & Go](opencode.md)).

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key |
