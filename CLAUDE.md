# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**yoagent** is a Rust library (crate) for building AI coding agents. It provides a core agent loop, multi-provider LLM streaming, built-in tools, MCP integration, and context management. Published to crates.io as `yoagent`.

## Build & Development Commands

```bash
cargo test --all-features            # Run all tests — what CI runs
cargo clippy --all-targets --all-features   # Lint — what CI runs (-Dwarnings)
cargo fmt                            # Auto-format code
cargo fmt -- --check                 # Check formatting (CI uses this)

cargo test <test_name>               # Run a single test by name
cargo test --test agent_test         # Run a specific test file
cargo run --example cli              # Run the interactive CLI example
cargo run --example basic            # Run the minimal example
```

**Pass `--all-features` when checking your work.** `openapi` and `gasp` are off
by default, and some targets exist only behind them: `gasp_emit` and
`llm_compaction_live` are both `required-features = ["gasp"]`, so plain
`cargo clippy --all-targets` skips them entirely and reports clean on code CI
will reject. This is not hypothetical — a breaking change to `CostConfig`
passed local clippy while `llm_compaction_live` no longer compiled.

CI (`RUSTFLAGS="-Dwarnings"`) treats all clippy warnings as errors. Integration tests in `tests/integration_anthropic.rs` require a live API key and are skipped by default.

## Architecture

### Core Loop Pattern

The central abstraction is a **stateless agent loop** (`agent_loop.rs`) driven by two traits:

- **`StreamProvider`** (`provider/traits.rs`) — streams LLM responses via SSE into an mpsc channel, returning a complete `Message`
- **`AgentTool`** (`types.rs`) — defines tool name/schema/execution; the primary extension point for custom tools

The loop: stream assistant response → extract tool calls → execute tools (parallel by default) → append results → repeat until `StopReason::Stop` with no follow-ups.

`agent_loop` and `agent_loop_continue` are **free functions**, not methods. The `Agent` struct (`agent.rs`) is an optional stateful wrapper that manages message history, tool registry, steering/follow-up queues, and provider selection. The `_with_sender` methods (`prompt_with_sender`, `prompt_messages_with_sender`, `continue_loop_with_sender`) accept a caller-provided `mpsc::UnboundedSender<AgentEvent>` for real-time event consumption on a separate task.

### Provider System

7 provider implementations behind `StreamProvider`. `Agent` holds a concrete provider directly; `ProviderRegistry` maps `ApiProtocol` → provider for registry-based dispatch (e.g. custom multi-protocol routers):

| Protocol | File | Covers |
|----------|------|--------|
| `Anthropic` | `anthropic.rs` | Claude models |
| `OpenAiCompat` | `openai_compat.rs` | OpenAI, Groq, Together, DeepSeek, Fireworks, Mistral, xAI, etc. (15+) |
| `OpenAiResponses` | `openai_responses.rs` | OpenAI Responses API |
| `AzureOpenAi` | `azure_openai.rs` | Azure OpenAI |
| `Google` | `google.rs` | Gemini |
| `GoogleVertex` | `google_vertex.rs` | Vertex AI |
| `Bedrock` | `bedrock.rs` | Amazon Bedrock (ConverseStream) |

`ModelConfig` + `OpenAiCompat` flags handle per-provider quirks (auth style, reasoning format, max_tokens field name, etc.).

### Key Types

- **`Content`** — enum: `Text`, `Image`, `Thinking`, `ToolCall`
- **`Message`** — enum: `User`, `Assistant`, `ToolResult` — each variant carries its own fields
- **`AgentMessage`** — `Llm(Message)` | `Extension(ExtensionMessage)` — extension messages (`role`, `kind`, `data`) don't enter LLM context
- **`AgentEvent`** — full event stream emitted to callers: `AgentStart`, `TurnStart`, `MessageStart/Update/End`, `ToolExecutionStart/Update/End`, `ProgressMessage`, `InputRejected`, `TurnEnd`, `AgentEnd`
- **`StopReason`** — `Stop`, `Length`, `ToolUse`, `Error`, `Aborted`, `Refusal`

### Context Management (`context.rs`)

- **`ContextTracker`** — hybrid real-usage + estimation; the loop uses it to calibrate the compaction budget against real provider usage
- **`compact_messages()`** — tiered compaction: Level 1 (truncate tool outputs) → Level 2 (summarize old turns) → Level 3 (drop middle turns)
- **`ExecutionLimits`/`ExecutionTracker`** — max turns (50), max tokens (1M), max duration (10 min)

### Tool Execution (`agent_loop.rs`)

`ToolExecutionStrategy` controls concurrency:
- `Parallel` (default) — `futures::join_all` for all tool calls
- `Sequential` — one at a time, checks steering queue between each
- `Batched { size }` — concurrent within batch, steering check between batches

**Structured outputs** (`Agent::prompt_structured::<T>(text, schema)`): the schema is threaded per-call through `prompt_messages_internal` into `StreamConfig.output_schema` (`OutputSchema` in `provider/traits.rs`) — never stored on the Agent. Errors: `Provider` (API failure), `Parse { source, raw }`, `NoOutput`; only this call's messages are scanned. Anthropic enforces it by tool-forcing (a synthetic tool + `tool_choice`; the loop's `unwrap_structured_tool_call` converts the forced call back to text **before** tool-call extraction); OpenAI-compat uses `response_format: json_schema`; Gemini uses `responseSchema`. Responses/Azure/Vertex/Bedrock warn and ignore.

**Tool middleware** (`ToolMiddleware` in `types.rs`): async approve/deny/modify hooks gating every tool call, run in a chain at the single choke point (`execute_single_tool`) shared by all three strategies. `before_tool` takes a `#[non_exhaustive]` `ToolCallRequest` context struct (extensible without breaking implementors). `Deny(reason)` becomes an error tool result the LLM sees (loop continues); `Modify(args)` rewrites the call; a panicking middleware is contained as a denial. Installed via `Agent::with_tool_middleware` / `SubAgentTool::with_tool_middleware` / `AgentLoopConfig::tool_middleware`. Empty chain = allow all.

### OpenAPI Integration (`openapi/`, feature-gated)

Behind the `openapi` Cargo feature. `OpenApiToolAdapter` parses an OpenAPI 3.0 spec and creates one `AgentTool` per operation. Factory methods: `from_str`, `from_file`, `from_url`, `from_spec`. `OperationFilter` controls which operations become tools. Added to `Agent` via `with_openapi_file()` / `with_openapi_url()` / `with_openapi_spec()`.

### MCP Integration (`mcp/`)

`McpClient` communicates via `McpTransport` trait (stdio or HTTP). `McpToolAdapter` wraps MCP tools to implement `AgentTool`, making them transparent to the agent loop. Added via `Agent::with_mcp_server_stdio()` / `with_mcp_server_http()`.

### GASP Bridge (`gasp.rs`, feature-gated)

Behind the `gasp` Cargo feature (dep: `yoagent-state`, the GASP reference runtime). `GaspRecorder` consumes the `AgentEvent` stream (via `recording_sender`, which also tees to a forward sender) and maps it onto `yoagent_state`'s `YoAgentStateSink`: AgentStart→run.started, assistant MessageEnd→model.called/finished pair, ToolExecutionStart/End→tool.called/finished, AgentEnd→run.finished + one git commit per run at stream close (scaffolding committed at init so clones restore). Stale open runs are closed as "interrupted" on open; a dropped sender finishes the run with the derived outcome; InputRejected → "rejected". Recording failures stop recording but the forward tee keeps flowing (error surfaces only via the returned handle — await it); events are forwarded BEFORE recording. `Ok(None)` when no AgentStart arrived. Zero loop changes. CI job `gasp-conformance` emits a repo via `examples/gasp_emit.rs` and runs the gasp conformance checker (7 checks) against it.

### Session Trees (`session.rs`)

`Session` stores history as an id/parent_id tree: `append` advances the head, `seek`/`seek_checkpoint` move it, appending after a seek forks a new branch (never overwrites). `path_messages()` feeds a branch into `Agent::with_messages`; `append_new(agent.messages())` is the post-run sync. JSONL persistence (`to_jsonl`/`from_jsonl`, head = last line). Freestanding — no loop changes; maps to GASP's `transcripts/` tier.

### Shared State (`shared_state.rs`)

`SharedState` is a pluggable key-value store (`Arc<dyn SharedStateBackend>`) for sub-agent communication. It lets a parent store large artifacts once and have multiple sub-agents read/write by reference — no re-pasting into prompts.

- Two built-in backends: `MemoryBackend` (default, `HashMap` with 10MB cap) and `FileBackend` (one file per key, persistent)
- Custom backends implement the `SharedStateBackend` trait
- Opt-in via `SubAgentTool::with_shared_state(state)` — injects a `shared_state` tool and appends a state summary to the sub-agent's system prompt automatically
- Actions: `get`, `set`, `list`, `remove`
- Opt in on a parent via `Agent::with_shared_state(state)`, which also registers
  the `shared_state` tool for that run
- The loop reads it at one point only: `AgentLoopConfig::tool_output_sink`, used
  on the **append path** to stash the full text of a truncated tool result so the
  marker can name a retrievable key. Never on the compaction path — by then the
  middle is already gone, and re-deriving there would move marker bytes that the
  prefix cache depends on

### Construction API

The primary constructor is `Agent::from_config(ModelConfig)`: it selects the built-in provider from `config.api`, sets the model id from `config.id`, and resolves the API key from the provider-conventional env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, …; see `provider::resolve_api_key`). This removes the provider↔config pairing footgun and the doubly-specified model id.

```rust
// provider auto-selected, key from XAI_API_KEY
let agent = Agent::from_config(ModelConfig::xai("grok-4-1-fast", "Grok 4.1 Fast"));
```

Other constructors:
- `Agent::from_provider(provider, config)` — explicit provider (custom impls, test doubles). Pair with `ModelConfig::mock()` in tests.
- `Agent::from_config_with(&registry, config) -> Result<_, AgentBuildError>` — resolve against a custom `ProviderRegistry`.
- `Agent::set_model(config)` — switch model mid-session (re-resolves the env key; re-selects the provider only when it was registry-resolved, never clobbering an explicit one; explicit keys preserved).
- `Agent::new(provider)` + `with_model`/`with_model_config` — the original builder, still supported.

`SubAgentTool` mirrors these: `from_config`, `from_config_with`, `from_provider`.

`SubAgentTool` mirrors these: `SubAgentTool::from_config(name, config)` and `SubAgentTool::from_provider(name, provider, config)`.

`AgentLoopConfig` also supports `turn_delay: Option<Duration>` — an inter-turn delay to throttle API calls for rate-limit-sensitive providers. Exposed on `SubAgentTool` via `with_turn_delay()`.

### Testing

All unit tests use `MockProvider` (`provider/mock.rs`) to simulate LLM responses without network. Test files are in `tests/` — `agent_test.rs`, `agent_loop_test.rs`, `tools_test.rs`. Follow the existing pattern of constructing a `MockProvider` with predetermined responses.

### Telemetry

The loop emits `tracing` spans: `agent_loop` → `llm_stream` per turn (records tokens_in/out/cached + cost_usd from `CostConfig` when configured) → `tool` per execution (records is_error). Futures are instrumented with `.instrument(span)` (never hold an entered guard across `.await`). OTel is app-side via `tracing-opentelemetry` — the library has no OTel dependency by design.

## Key Design Conventions

- Context overflow detection is centralized in `OVERFLOW_PHRASES` (`provider/traits.rs`) covering 15+ provider-specific error strings; both HTTP errors and SSE-embedded errors are classified
- Tools return stdout/stderr even on failure so the LLM can self-correct
- Retry logic (`retry.rs`) uses exponential backoff with ±20% jitter; only retries `RateLimited` and `Network` errors
- The `skills.rs` module loads `<name>/SKILL.md` files with YAML frontmatter per the AgentSkills standard
