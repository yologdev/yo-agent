<div align="center">

<picture>
  <img alt="yoagent" src="docs/images/banner.png" width="100%" height="auto">
</picture>

<a href="https://crates.io/crates/yoagent">crates.io</a> · <a href="https://yologdev.github.io/yoagent/">Docs</a> · <a href="https://github.com/yologdev/yoagent">GitHub</a> · <a href="https://deepwiki.com/yologdev/yoagent">DeepWiki</a> · <a href="https://github.com/yologdev/yoagent/issues">Issues</a> · <a href="https://github.com/yologdev/yoagent/releases">Releases</a>

[![][crates-shield]][crates-link]
[![][ci-shield]][ci-link]
[![][license-shield]][license-link]
[![][docs-shield]][docs-link]
[![][last-commit-shield]][last-commit-link]

</div>

---

## Overview

yoagent is a simple, effective agent loop with tool execution and event streaming in Rust. Inspired by [pi-agent-core](https://github.com/badlogic/pi-mono/tree/main/packages/agent).

The loop is the product. No over-engineered planning/reflection/RAG layers — just:

```
Prompt → LLM Stream → Tool Execution → Loop if tool calls → Done
```

Everything is observable via events. Supports 7 API protocols covering 20+ LLM providers out of the box.

## Features

**Agent Loop**
- Stateful agent with steering (interrupt mid-run) and follow-up (queue work after completion)
- Full event stream: `AgentStart` → `TurnStart` → `MessageUpdate` (deltas) → `ToolExecution` → `TurnEnd` → `AgentEnd`
- Parallel tool execution by default — sequential and batched strategies also available
- Sub-agents via `SubAgentTool` — delegate tasks to child agent loops with their own tools and system prompts
- Tool middleware — async approve/deny/modify hooks gating every tool call (`with_tool_middleware`), the mechanism for permission prompts and policy engines
- Structured outputs — `prompt_structured::<T>()` returns typed, schema-validated replies, enforced natively where supported (Anthropic tool-forcing, OpenAI `json_schema`, Gemini `responseSchema`; other protocols log a warning)
- Real-time event streaming — `prompt()` spawns the loop concurrently and returns events immediately; `prompt_with_sender()` accepts a caller-provided channel for custom consumption
- Streaming tool output — tools emit real-time progress via `on_update` callback
- Multimodal support — `Content::Image` flows through tool results across all providers
- Automatic retry with exponential backoff and jitter for rate limits and network errors
- Custom message types via `AgentMessage::Extension` — app-specific messages that don't pollute LLM context
- State persistence — `save_messages()` / `restore_messages()` for pause/resume workflows
- Session trees — branching history with fork, checkpoints, and JSONL persistence (`Session`); edit an earlier turn and re-run without losing the original branch
- Lifecycle callbacks — `before_turn`, `after_turn`, `on_error` for observability and control
- Telemetry — `tracing` spans for the loop, each LLM stream (tokens + cost fields), and each tool execution; bridge to OpenTelemetry via `tracing-opentelemetry`, negligible overhead when no subscriber is installed
- Full serde support — all core types implement `Serialize`/`Deserialize`/`PartialEq`
- [AgentSkills](https://agentskills.io)-compatible skills — load skill directories, inject into system prompt, agent activates on demand

**Multi-Provider**
- 7 API protocols, 20+ providers behind one `StreamProvider` trait
- One OpenAI-compatible implementation covers OpenAI, xAI, Groq, Cerebras, OpenRouter, Mistral, and more
- Per-provider quirk flags (`OpenAiCompat`, `AnthropicCompat`) handle auth, reasoning format, and tool handling differences
- Capability notes: thinking/reasoning controls are wired for **all 7 protocols** — Anthropic (adaptive/budget), OpenAI-compatible where the `ModelConfig` opts in (the `openai()`/`deepseek()`/`meta()` presets do; other compat presets silently drop `thinking_level`), OpenAI Responses & Azure (reasoning effort), Gemini & Vertex (`thinkingConfig` with thought summaries streamed back), Bedrock (Anthropic-style budgets, reasoning deltas streamed back). Client-side prompt-cache breakpoints are Anthropic-specific; most other providers cache server-side automatically, but Bedrock has no automatic caching

**Built-in Tools**
- `bash` — Shell execution with timeout, output truncation, command deny patterns
- `read_file` / `write_file` — File I/O with line numbers, path restrictions, auto-mkdir
- `edit_file` — Surgical search/replace with fuzzy match error hints
- `list_files` — Directory exploration via `find`
- `search` — Pattern search via ripgrep/grep with context lines

**Integrations**
- OpenAPI tool adapter — auto-generate tools from any OpenAPI 3.0 spec (`features = ["openapi"]`)
- MCP (Model Context Protocol) — connect to MCP tool servers via stdio or HTTP
- GASP (`features = ["gasp"]`) — record runs into a [GASP](https://github.com/yologdev/gasp) agent repo (append-only semantic event log; restore = clone + replay); yoagent is a **tested** GASP-conformant runtime — the protocol's 7-check conformance suite runs in CI

**Context Management**
- Context overflow detection across all major providers (Anthropic, OpenAI, Google, Bedrock, xAI, Groq, OpenRouter, llama.cpp, and more)
- `ContextTracker` — hybrid real-usage + estimation for accurate token tracking
- Tiered compaction: truncate tool outputs → summarize old turns → drop middle
- Execution limits (max turns, max tokens, timeout)
- Building blocks for LLM-based summarization (`replace_messages()`, `compact_messages()`)

---

## Quick Start

### Install

```bash
cargo add yoagent tokio --features tokio/full
```

Or add to `Cargo.toml`:

```toml
[dependencies]
yoagent = "0.14"
tokio = { version = "1", features = ["full"] }
```

### Basic Usage

```rust
use yoagent::agent::Agent;
use yoagent::provider::ModelConfig;
use yoagent::types::*;

#[tokio::main]
async fn main() {
    // Provider is selected from the config's protocol; the key is read from
    // ANTHROPIC_API_KEY. Call `.with_api_key(key)` to pass one explicitly.
    let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
        .with_system_prompt("You are a helpful assistant.");

    let mut rx = agent.prompt("What is Rust's ownership model?").await;

    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::MessageUpdate {
                delta: StreamDelta::Text { delta }, ..
            } => print!("{}", delta),
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }
}
```

### Skills ([AgentSkills](https://agentskills.io) compatible)

Skills extend the agent with domain expertise. A skill is a directory with a `SKILL.md`:

```
skills/
└── git/
    ├── SKILL.md       # YAML frontmatter + instructions
    └── scripts/       # Optional resources
```

```rust
use yoagent::SkillSet;

let skills = SkillSet::load(&["./skills"])?;

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
    .with_system_prompt("You are a coding assistant.")
    .with_skills(skills)   // Injects skill index into system prompt
    .with_tools(tools);
```

The agent sees a compact index of available skills. When a task matches, it reads the full SKILL.md using the `read_file` tool — no special infrastructure needed. Skills are cross-compatible with Claude Code, Codex CLI, Gemini CLI, Cursor, and other AgentSkills-compatible agents.

See [docs/concepts/skills.md](docs/concepts/skills.md) for the full guide.

### Interactive CLI (mini coding agent)

```bash
ANTHROPIC_API_KEY=sk-... cargo run --example cli
# With skills:
ANTHROPIC_API_KEY=sk-... cargo run --example cli -- --skills ./skills
# With Ollama:
cargo run --example cli -- --provider ollama --model llama3.1:8b
# With another local server (LM Studio, llama.cpp, vLLM):
cargo run --example cli -- --api-url http://localhost:1234/v1 --model my-model
```

A ~250-line interactive coding agent with all built-in tools, skills support, streaming output, and colored tool feedback. Like a baby Claude Code.

```
  yoagent cli — mini coding agent
  Type /quit to exit, /clear to reset

  model: claude-sonnet-5
  skills: 3 loaded
  cwd:   /home/user/my-project

> find all TODO comments in src/

  ▶ search 'TODO' ✓

Found 3 TODOs:
  src/main.rs:42: // TODO: handle edge case
  src/lib.rs:15:  // TODO: add tests
  src/utils.rs:8: // TODO: optimize this

  tokens: 1250 in / 89 out
```

<details>
<summary>OpenAI-compatible provider example</summary>

```rust
use yoagent::{Agent, provider::ModelConfig};

// Pick a first-class preset — the provider is inferred from it, and the API
// key is read from that provider's conventional env var (GROQ_API_KEY here).
let mut agent = Agent::from_config(ModelConfig::groq("llama-3.3-70b-versatile", "Llama 3.3 70B"));

// Or Qwen / DashScope (DASHSCOPE_API_KEY):
let mut agent = Agent::from_config(ModelConfig::qwen("qwen3.6-plus", "Qwen 3.6 Plus"));

// Or Google Gemini (GEMINI_API_KEY):
let mut agent = Agent::from_config(ModelConfig::google("gemini-2.5-pro", "Gemini 2.5 Pro"));
```

</details>

---

## Providers

| Protocol | Providers |
|----------|-----------|
| Anthropic Messages | Anthropic (Claude) |
| OpenAI Completions | OpenAI, xAI, Groq, Mistral, DeepSeek, MiniMax, Z.ai, Qwen, Meta (Muse Spark), Ollama, local servers, and custom compatible APIs |
| OpenAI Responses | OpenAI (newer API) |
| Azure OpenAI | Azure OpenAI |
| Google Generative AI | Google Gemini |
| Google Vertex | Google Vertex AI |
| Bedrock ConverseStream | Amazon Bedrock |

OpenAI-compatible providers share one implementation with per-provider quirk flags for differences in auth, reasoning format, tool handling, and more. Adding a new compatible provider is just a `ModelConfig` with the right `base_url`.

---

## Architecture

```
yoagent/
├── src/
│   ├── types.rs            # Message, AgentMessage, AgentEvent, AgentTool trait
│   ├── agent_loop.rs       # Core loop (agent_loop + agent_loop_continue)
│   ├── agent.rs            # Stateful Agent with steering/follow-up queues
│   ├── context.rs          # Token estimation, smart truncation, execution limits
│   ├── sub_agent.rs        # SubAgentTool — delegate tasks to child agent loops
│   ├── tools/
│   │   ├── bash.rs         # Shell execution (timeout, deny patterns, confirm_fn)
│   │   ├── file.rs         # Read/write files (line numbers, path restrictions)
│   │   ├── edit.rs         # Search/replace editing with fuzzy match hints
│   │   ├── list.rs         # Directory listing via find
│   │   └── search.rs       # Pattern search via ripgrep/grep
│   └── provider/
│       ├── traits.rs           # StreamProvider trait, StreamEvent, ProviderError
│       ├── model.rs            # ModelConfig, ApiProtocol, OpenAiCompat
│       ├── registry.rs         # ProviderRegistry — dispatch by protocol
│       ├── anthropic.rs        # Anthropic Messages API
│       ├── openai_compat.rs    # OpenAI Chat Completions (15+ providers)
│       ├── openai_responses.rs # OpenAI Responses API
│       ├── azure_openai.rs     # Azure OpenAI
│       ├── google.rs           # Google Generative AI (Gemini)
│       ├── google_vertex.rs    # Google Vertex AI
│       ├── bedrock.rs          # Amazon Bedrock (ConverseStream)
│       ├── sse.rs              # Shared SSE parsing utility
│       └── mock.rs             # Mock provider for testing
├── docs/                   # mdBook documentation
├── examples/               # Usage examples
└── tests/                  # Integration tests
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- [Documentation](https://yologdev.github.io/yoagent/) — Full reference
- [pi-agent-core](https://github.com/badlogic/pi-mono/tree/main/packages/agent) — Original inspiration (TypeScript)

<!-- Badge link definitions -->
[crates-shield]: https://img.shields.io/crates/v/yoagent?labelColor=black&style=flat-square&logo=rust&color=orange
[crates-link]: https://crates.io/crates/yoagent
[ci-shield]: https://img.shields.io/github/actions/workflow/status/yologdev/yoagent/ci.yml?labelColor=black&style=flat-square&logo=github&label=CI
[ci-link]: https://github.com/yologdev/yoagent/actions/workflows/ci.yml
[license-shield]: https://img.shields.io/badge/license-MIT-white?labelColor=black&style=flat-square
[license-link]: https://github.com/yologdev/yoagent/blob/main/LICENSE
[docs-shield]: https://img.shields.io/badge/docs-mdBook-blue?labelColor=black&style=flat-square
[docs-link]: https://yologdev.github.io/yoagent/
[last-commit-shield]: https://img.shields.io/github/last-commit/yologdev/yoagent?color=c4f042&labelColor=black&style=flat-square
[last-commit-link]: https://github.com/yologdev/yoagent/commits/main
