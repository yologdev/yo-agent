# Sub-Agents

Sub-agents let a parent agent delegate tasks to child agent loops, each with their own system prompt, tools, and provider. The parent LLM invokes them like any other tool.

## Overview

```
Parent Agent
├── prompt("Research X and implement Y")
│   ├── calls SubAgentTool("researcher", task="Research X")
│   │   └── child agent_loop() with read/search tools → returns findings
│   ├── calls SubAgentTool("coder", task="Implement Y based on findings")
│   │   └── child agent_loop() with edit/write tools → returns result
│   └── summarizes both results
```

Each sub-agent invocation starts a fresh conversation — no state leaks between calls.

## Creating Sub-Agents

```rust
use std::sync::Arc;
use yoagent::sub_agent::SubAgentTool;
use yoagent::provider::ModelConfig;
use yoagent::tools;

let researcher = SubAgentTool::from_config(
    "researcher",
    ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"),
)
    .with_description("Searches and reads files to gather information.")
    .with_system_prompt("You are a research assistant. Be thorough and concise.")
    .with_tools(vec![
        Arc::new(tools::ReadFileTool::new()),
        Arc::new(tools::SearchTool::new()),
    ])
    .with_max_turns(10);
```

## Registering on a Parent Agent

```rust
use yoagent::agent::Agent;

let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_system_prompt("You coordinate between sub-agents.")
    .with_sub_agent(researcher)
    .with_sub_agent(coder);
```

The parent sees sub-agents as regular tools. It decides when to delegate based on its system prompt.

## Parallel Execution

When the parent LLM calls multiple sub-agents in a single response, they run concurrently (default `Parallel` strategy). Two sub-agents each taking 50ms complete in ~50ms total, not 100ms.

## Configuration

| Method | Purpose |
|--------|---------|
| `with_description()` | What the parent LLM sees (helps it decide when to delegate) |
| `with_system_prompt()` | The sub-agent's own instructions |
| `with_skills()` | Attach a `SkillSet` — its index is appended to the sub-agent's system prompt (mirrors `Agent::with_skills`) |
| `from_config(name, config)` / `from_provider(name, provider, config)` | Set the sub-agent's model, provider, and metadata from a `ModelConfig` — resolves the env key automatically and can use a different model than the parent |
| `with_api_key()` | Override the env-resolved API key explicitly |
| `with_tools()` | Tools available to the sub-agent (accepts `Vec<Arc<dyn AgentTool>>`) |
| `with_max_turns(N)` | Turn limit (default: 10). Primary guard against runaway execution. |
| `with_thinking()` | Enable extended thinking for the sub-agent |
| `with_cache_config()` | Prompt caching settings |
| `with_turn_delay()` | Inter-turn delay to throttle API calls (useful for rate-limit-sensitive providers) |
| `with_retry_config()` | Custom retry configuration for transient errors |
| `with_tool_execution()` | Tool execution strategy (`Parallel`, `Sequential`, `Batched`) |

## Event Forwarding

When the parent provides an `on_update` callback (standard for all tools), sub-agent events are forwarded as `ToolExecutionUpdate` events. The parent's UI sees real-time progress from the child:

- Text deltas from the sub-agent's LLM responses
- Tool call notifications from the sub-agent's tool usage

## Shared State

By default, each sub-agent invocation is isolated — to pass data between sub-agents, the parent must re-paste it into every prompt. For large artifacts (CI logs, codebases, analysis results), this wastes context tokens.

`SharedState` solves this: store an artifact once, and any number of sub-agents read/write it by reference.

```rust
use yoagent::shared_state::SharedState;

let state = SharedState::new();
state.set("ci_log", large_log_text).await.unwrap();

let analyzer = SubAgentTool::from_provider(
    "analyzer",
    provider.clone(),
    ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"),
)
    .with_system_prompt("Analyze the CI log for failures.")
    .with_shared_state(state.clone());  // opt-in
```

When `.with_shared_state()` is used, the sub-agent automatically gets:

1. A `shared_state` tool with `get`, `set`, `list`, and `remove` actions
2. A system prompt appendix listing available keys and their sizes

The sub-agent reads the artifact via tool call instead of having it pasted into the prompt:

```
Sub-agent calls: shared_state(action="get", key="ci_log")
Sub-agent calls: shared_state(action="set", key="summary", value="...")
```

The parent reads results back programmatically:

```rust
let summary = state.get("summary").await.expect("sub-agent wrote this");
```

### Parallel Sub-Agents with Shared State

Multiple sub-agents can share the same `SharedState` concurrently. Each gets its own clone of the `Arc` handle — reads are concurrent, writes are serialized by `tokio::sync::RwLock`.

```rust
let error_analyst = SubAgentTool::from_provider(
    "error_analyst",
    provider.clone(),
    ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"),
)
    .with_shared_state(state.clone());
let perf_analyst = SubAgentTool::from_provider(
    "perf_analyst",
    provider.clone(),
    ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"),
)
    .with_shared_state(state.clone());

// Both run in parallel, reading the same artifact and writing different keys
```

### Backends

`SharedState` is backed by a pluggable `SharedStateBackend` trait. Two built-in backends are provided:

**MemoryBackend** (default) — in-memory `HashMap` with a byte capacity limit:

```rust
let state = SharedState::new();                          // 10MB default
let state = SharedState::with_max_bytes(50 * 1024 * 1024); // 50MB
```

A `set` call that would exceed capacity returns `Err(CapacityError)`.

**FileBackend** — one file per key, persistent across process restarts:

```rust
use yoagent::shared_state::FileBackend;

let state = SharedState::with_backend(FileBackend::new(".agent-state"));
```

Keys are percent-encoded to filenames (reversible, no collisions). Useful for debugging (inspect state with `ls` / `cat`) and for long-running workflows where memory limits matter.

**Custom backends** implement the `SharedStateBackend` trait:

```rust
use yoagent::shared_state::{SharedStateBackend, SharedStateError};

#[async_trait::async_trait]
impl SharedStateBackend for MyRedisBackend {
    async fn get(&self, key: &str) -> Result<Option<String>, SharedStateError> { ... }
    async fn set(&self, key: &str, value: String) -> Result<(), SharedStateError> { ... }
    async fn remove(&self, key: &str) -> Result<bool, SharedStateError> { ... }
    async fn keys(&self) -> Result<Vec<String>, SharedStateError> { ... }
    async fn summary(&self) -> Result<String, SharedStateError> { ... }
}

let state = SharedState::with_backend(MyRedisBackend::new());
```

See [`examples/shared_state.rs`](../../examples/shared_state.rs) for a complete parallel analysis demo.

## Multi-Provider Support

Sub-agents can use any provider supported by yoagent — not just Anthropic. Pass a `ModelConfig` to configure the base URL, compat flags, and other provider-specific settings:

```rust
use yoagent::provider::ModelConfig;

let model_config = ModelConfig::xai("grok-4-1-fast-reasoning", "Grok 3 Mini Fast");

// `from_config` resolves OpenAiCompatProvider from the config's protocol and
// the key from XAI_API_KEY.
let analyst = SubAgentTool::from_config("analyst", model_config)
    .with_tools(vec![...]);
```

This works with all providers: OpenAI, Groq, DeepSeek, Gemini, Mistral, xAI, and more. See [Model Presets](../providers/model-presets.md) for the full list of first-class factory methods.

## Design Decisions

- **Context isolation**: Each invocation starts fresh. Sub-agents don't accumulate history across calls.
- **Nesting supported**: Sub-agents can be given other `SubAgentTool`s for recursive delegation (see [`examples/rlm.rs`](../../examples/rlm.rs)). Use `with_max_turns()` to prevent infinite chains.
- **Cancellation propagation**: The parent's cancellation token is forwarded. Aborting the parent aborts all sub-agents.
- **Turn limiting**: The default 10-turn limit prevents runaway execution. The parent's execution limits also apply to total wall-clock time.

## Isolating sub-agents from each other

`SharedState` is a single flat namespace by design — a parent stores an artifact once and several sub-agents read it by reference, which is the whole point of the type. Every holder sees every key.

When a sub-agent should *not* see its siblings' data, hand it a scoped view:

```rust
let state = SharedState::new();

let researcher = SubAgentTool::from_config("researcher", config.clone())
    .with_scoped_shared_state(state.clone(), "researcher");
let writer = SubAgentTool::from_config("writer", config)
    .with_scoped_shared_state(state.clone(), "writer");

// The parent's unscoped handle still sees everything both wrote.
for key in state.keys().await { /* … */ }
```

Keys are transparently prefixed, so a scoped sub-agent cannot read, overwrite, or **enumerate** anything outside its scope — including via a crafted key, since the prefix is applied on the way in. `summary()` is scoped too, which matters most: it is injected into the sub-agent's system prompt, so an unscoped summary would disclose every sibling's key names.

Scoping nests (`state.scoped("a").scoped("b")`), so a sub-agent can narrow its own view but never widen it. Keep plain `with_shared_state` when sharing is the intent.

## Examples

- [`examples/sub_agent.rs`](../../examples/sub_agent.rs) — Coordinator with researcher and coder sub-agents
- [`examples/code_review.rs`](../../examples/code_review.rs) — 3 parallel sub-agents reviewing a file via shared state
- [`examples/rlm.rs`](../../examples/rlm.rs) — Recursive Language Model: nested sub-agents with autonomous file discovery
