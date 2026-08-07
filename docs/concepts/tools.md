# Tools

## The AgentTool Trait

Every tool implements `AgentTool`:

```rust
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError>;
}
```

| Method | Purpose |
|--------|---------|
| `name()` | Unique ID sent to LLM (e.g., `"bash"`) |
| `label()` | Human-readable name for UI (e.g., `"Run Command"`) |
| `description()` | Tells the LLM what the tool does |
| `parameters_schema()` | JSON Schema for the tool's parameters |
| `execute()` | Runs the tool, returns `ToolResult` or `ToolError`. Receives a `ToolContext` with cancellation, update, and progress callbacks. |

## ToolContext

All execution context is bundled into a single struct, making the trait easier to extend in the future:

```rust
pub struct ToolContext {
    pub tool_call_id: String,
    pub tool_name: String,
    pub cancel: CancellationToken,
    pub on_update: Option<ToolUpdateFn>,
    pub on_progress: Option<ProgressFn>,
}
```

| Field | Purpose |
|-------|---------|
| `tool_call_id` | Unique ID for this tool call (for correlating events) |
| `tool_name` | Name of the tool being executed |
| `cancel` | Cancellation token — check `ctx.cancel.is_cancelled()` in long-running tools |
| `on_update` | Callback for streaming partial `ToolResult` updates to the UI (emits `ToolExecutionUpdate`) |
| `on_progress` | Callback for emitting user-facing progress messages (emits `ProgressMessage`) |

`ToolContext` implements `Clone` and `Debug`.

## ToolResult

```rust
pub struct ToolResult {
    pub content: Vec<Content>,
    pub details: serde_json::Value,
}
```

The `content` is sent back to the LLM. The `details` field holds metadata (not sent to the LLM) for UI/logging.

## ToolError

```rust
pub enum ToolError {
    Failed(String),
    NotFound(String),
    InvalidArgs(String),
    Cancelled,
}
```

Errors are converted to `ToolResult` with `is_error: true` and sent back to the LLM so it can recover.

## Implementing a Custom Tool

```rust
use yoagent::types::*;
use async_trait::async_trait;

pub struct WeatherTool;

#[async_trait]
impl AgentTool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn label(&self) -> &str { "Weather" }
    fn description(&self) -> &str {
        "Get current weather for a city."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let city = params["city"].as_str()
            .ok_or(ToolError::InvalidArgs("missing city".into()))?;

        // Call weather API...
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("Weather in {}: 72°F, sunny", city),
            }],
            details: serde_json::Value::Null,
        })
    }
}
```

Register custom tools alongside defaults:

```rust
use yoagent::tools::default_tools;

let mut tools = default_tools();
tools.push(Box::new(WeatherTool));
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5")).with_tools(tools);
```

## Sandboxing the built-in file tools

`ReadFileTool`, `WriteFileTool`, `EditFileTool`, `ListFilesTool` and `SearchTool` accept an allowlist of directory roots. Empty (the default) means unrestricted:

```rust
let roots = vec!["/srv/workspace".to_string()];

let tools: Vec<Box<dyn AgentTool>> = vec![
    Box::new(ReadFileTool::new().with_allowed_paths(roots.clone())),
    Box::new(WriteFileTool::new().with_allowed_paths(roots.clone())),
    Box::new(EditFileTool::new().with_allowed_paths(roots)),
];
```

Enforcement is against the **resolved** path, not the string, so neither `..` nor a symlink pointing outside can escape — and a file that does not exist yet still resolves through its real parent, so writes are checked too. The rejection message deliberately does not echo the allowed roots, since tool results reach the model's transcript.

## `BashTool` is not a sandbox

`BashTool` runs whatever the model asks through `bash -c`, with the agent process's own environment and filesystem access.

`deny_patterns` is a substring check that catches typos and obvious mistakes — `rm  -rf /` with two spaces, a base64-decoded pipe, or an equivalent `find -delete` all sail past it. Treat it as a guardrail, never as a security boundary.

Real isolation belongs outside the tool: run the agent in a container or VM, or gate calls through [`ToolMiddleware`](#tool-middleware), which sees the arguments before execution and can deny them.

For credentials specifically, commands inherit every environment variable the agent process holds — including any `*_API_KEY`. When the model composes the command, restrict what it can read:

```rust
let bash = BashTool::default()
    .with_env_allowlist(vec!["RUST_LOG".to_string()]);  // plus PATH, HOME, PWD
```

## Error Handling

**Return `Err(ToolError)` on failure, not `Ok` with error text.** When a tool returns `Err`, the agent loop converts it to a `Message::ToolResult` with `is_error: true` and sends it to the LLM. The LLM sees the error and can self-correct — retry with different arguments, try a different approach, or explain the failure to the user.

```rust
async fn execute(&self, params: serde_json::Value, _ctx: ToolContext) -> Result<ToolResult, ToolError> {
    let path = params["path"].as_str()
        .ok_or(ToolError::InvalidArgs("missing 'path'".into()))?;

    let content = std::fs::read_to_string(path)
        .map_err(|e| ToolError::Failed(format!("Cannot read {}: {}", path, e)))?;

    Ok(ToolResult {
        content: vec![Content::Text { text: content }],
        details: serde_json::Value::Null,
    })
}
```

**Exception: BashTool.** The built-in `BashTool` returns `Ok` even on non-zero exit codes, with both stdout and stderr in the result. This is intentional — the LLM needs to see the actual error output (compilation errors, test failures, etc.) to diagnose and fix issues. Only truly exceptional failures (e.g., command not found, cancellation) return `Err`.

## Tool Execution Flow

1. LLM returns `Content::ToolCall` blocks in its response
2. Agent loop emits `ToolExecutionStart` for each
3. Tool's `execute()` is called with parsed arguments
4. Result (or error) is wrapped in `Message::ToolResult`
5. `ToolExecutionEnd` is emitted
6. All tool results are added to context
7. Loop continues with another LLM call

## Streaming Tool Output

Long-running tools can stream progress updates to the UI via the `on_update` callback. Each call emits a `ToolExecutionUpdate` event. Partial results are **for UI/logging only** — they are not sent to the LLM. Only the final `ToolResult` returned from `execute()` becomes part of the conversation.

### The `ToolUpdateFn` type

```rust
pub type ToolUpdateFn = Arc<dyn Fn(ToolResult) + Send + Sync>;
```

### Basic usage

Call `on_update` whenever you have progress to report:

```rust
use yoagent::types::*;

struct DataProcessorTool;

#[async_trait]
impl AgentTool for DataProcessorTool {
    // ... name, label, description, parameters_schema ...

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let rows = fetch_rows(&params)?;
        let total = rows.len();

        for (i, row) in rows.iter().enumerate() {
            // Check for cancellation
            if ctx.cancel.is_cancelled() {
                return Err(ToolError::Cancelled);
            }

            process_row(row);

            // Stream progress every 100 rows
            if i % 100 == 0 {
                if let Some(ref cb) = &ctx.on_update {
                    cb(ToolResult {
                        content: vec![Content::Text {
                            text: format!("Processed {}/{} rows", i, total),
                        }],
                        details: serde_json::json!({"progress": i as f64 / total as f64}),
                    });
                }
            }
        }

        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("Processed all {} rows", total),
            }],
            details: serde_json::Value::Null,
        })
    }
}
```

### Consuming updates in your UI

Updates arrive as `AgentEvent::ToolExecutionUpdate` events on the same event stream as all other agent events:

```rust
while let Some(event) = rx.recv().await {
    match event {
        AgentEvent::ToolExecutionStart { tool_name, .. } => {
            println!("⏳ {} started", tool_name);
        }
        AgentEvent::ToolExecutionUpdate { tool_name, partial_result, .. } => {
            // Show progress in your UI
            if let Some(Content::Text { text }) = partial_result.content.first() {
                println!("  📊 {}: {}", tool_name, text);
            }
        }
        AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
            println!("{} {}", if is_error { "❌" } else { "✅" }, tool_name);
        }
        AgentEvent::ProgressMessage { tool_name, text, .. } => {
            println!("  💬 {}: {}", tool_name, text);
        }
        _ => {}
    }
}
```

### Progress Messages

In addition to `on_update` (which streams partial `ToolResult` values), tools can emit lightweight text-only progress messages via `ctx.on_progress`. These appear as `AgentEvent::ProgressMessage` events:

```rust
async fn execute(&self, params: serde_json::Value, ctx: ToolContext) -> Result<ToolResult, ToolError> {
    if let Some(ref progress) = &ctx.on_progress {
        progress("Starting analysis...".into());
    }

    // ... do work ...

    if let Some(ref progress) = &ctx.on_progress {
        progress("Almost done...".into());
    }

    Ok(ToolResult { /* ... */ })
}
```

Use `on_progress` for simple status text. Use `on_update` when you need structured data (progress percentages, partial results).

### Guidelines

- **Call `on_update` as often as useful** — there's no rate limit. The callback is synchronous and cheap.
- **Always check `ctx.on_update.is_some()`** before building the `ToolResult`. If `None`, the loop isn't interested in updates (e.g., testing).
- **Use `details` for structured data** — `content` is for human-readable text, `details` can carry progress percentages, byte counts, etc.
- **Don't rely on updates reaching the LLM** — they won't. Only the final return value is added to context.
- **Simple tools don't need it** — if your tool completes in <1 second, just ignore `ctx` (prefix with `_ctx` to suppress the warning).

### End-to-end example

Here's a complete example: a CLI agent with a deploy tool that streams progress. The human sees real-time output while the LLM only gets the final result.

```rust
use yoagent::agent::Agent;
use yoagent::provider::ModelConfig;
use yoagent::types::*;

/// A tool that deploys an app and streams each step.
struct DeployTool;

#[async_trait]
impl AgentTool for DeployTool {
    fn name(&self) -> &str { "deploy" }
    fn label(&self) -> &str { "Deploy App" }
    fn description(&self) -> &str { "Deploy the application to production." }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "env": { "type": "string", "description": "Target environment" }
            },
            "required": ["env"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let env = params["env"].as_str().unwrap_or("staging");

        let steps = ["Building image", "Running tests", "Pushing to registry", "Rolling out"];
        for (i, step) in steps.iter().enumerate() {
            if ctx.cancel.is_cancelled() {
                return Err(ToolError::Cancelled);
            }

            // Stream each step to the UI
            if let Some(ref cb) = &ctx.on_update {
                cb(ToolResult {
                    content: vec![Content::Text {
                        text: format!("[{}/{}] {}...", i + 1, steps.len(), step),
                    }],
                    details: serde_json::json!({
                        "step": i + 1,
                        "total": steps.len(),
                        "phase": step,
                    }),
                });
            }

            // Simulate work
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }

        // Only this final result is sent to the LLM
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("Successfully deployed to {}", env),
            }],
            details: serde_json::json!({"env": env, "status": "success"}),
        })
    }
}

#[tokio::main]
async fn main() {
    let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
        .with_system_prompt("You are a deployment assistant.")
        .with_tools(vec![Box::new(DeployTool)]);

    let mut rx = agent.prompt("Deploy to production").await;

    while let Some(event) = rx.recv().await {
        match event {
            // LLM text streaming
            AgentEvent::MessageUpdate {
                delta: StreamDelta::Text { delta }, ..
            } => print!("{}", delta),

            // Tool progress streaming
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                println!("\n🚀 Starting {}...", tool_name);
            }
            AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
                if let Some(Content::Text { text }) = partial_result.content.first() {
                    println!("  {}", text);
                }
            }
            AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
                if is_error {
                    println!("  ❌ {} failed", tool_name);
                } else {
                    println!("  ✅ {} complete", tool_name);
                }
            }
            AgentEvent::ProgressMessage { text, .. } => {
                println!("  💬 {}", text);
            }

            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }
}
```

Running this produces:

```
🚀 Starting deploy...
  [1/4] Building image...
  [2/4] Running tests...
  [3/4] Pushing to registry...
  [4/4] Rolling out...
  ✅ deploy complete
Successfully deployed to production. The deployment completed all 4 stages.
```

The human sees each step as it happens. The LLM only sees "Successfully deployed to production" and can continue the conversation from there.

### How agents benefit

When an AI agent (like a coding assistant) uses yoagent, streaming tool output helps in two ways:

1. **Human oversight** — The human watching the agent work sees real-time progress instead of waiting for a tool to finish. A bash command running `cargo build` can stream compiler output as it happens, so the human can interrupt early if something is wrong.

2. **Agent UIs** — Tools like web dashboards, IDE extensions, or chat interfaces can render live progress bars, log tails, or status indicators. The `details` field in `ToolResult` carries structured data (progress percentage, byte counts, etc.) that UIs can render however they want.

The LLM itself doesn't see updates — it works with final results only. This is intentional: partial output would waste context tokens and confuse the model. The streaming is purely a **human-facing** feature.

## Execution Strategies

When the LLM returns multiple tool calls in a single response (e.g., "read file A, read file B, run bash C"), `ToolExecutionStrategy` controls how they run:

| Strategy | Behavior |
|----------|----------|
| `Sequential` | One at a time. Steering checked between each tool. Use for debugging or tools with shared mutable state. |
| **`Parallel`** (default) | All tool calls run concurrently via `futures::join_all`. Steering checked after all complete. Best latency for independent tools. |
| `Batched { size }` | Run in groups of N. Steering checked between batches. Balances speed with human-in-the-loop control. |

### Configuration

```rust
use yoagent::agent::Agent;
use yoagent::types::ToolExecutionStrategy;

// Default — parallel (fastest)
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"));

// Sequential (debug / shared state)
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_tool_execution(ToolExecutionStrategy::Sequential);

// Batched — 3 at a time
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_tool_execution(ToolExecutionStrategy::Batched { size: 3 });
```

### When to use each

- **Parallel** (default): Most tool calls are independent — file reads, searches, API calls. Running them concurrently can cut latency dramatically (3 tools × 50ms = ~50ms instead of ~150ms).
- **Sequential**: When tools have side effects that depend on order, or when you need fine-grained steering control between each tool.
- **Batched**: When you want parallelism but also want steering checkpoints. For example, `Batched { size: 3 }` runs 3 tools concurrently, checks for user interrupts, then runs the next 3.

Steering messages are always checked between execution units (between each tool in Sequential, after all tools in Parallel, between batches in Batched). If a user interrupts, remaining tools are skipped.

## Permissions: Tool Middleware

Every tool call can be gated by an async **middleware chain** — the mechanism
behind permission prompts, policy engines, and argument rewriting. yoagent
ships the hook, not a policy: with no middleware installed, every call runs.

```rust
use yoagent::{ToolCallRequest, ToolDecision, ToolMiddleware};

struct ReadOnlyPolicy;

#[async_trait::async_trait]
impl ToolMiddleware for ReadOnlyPolicy {
    async fn before_tool(&self, call: &ToolCallRequest<'_>) -> ToolDecision {
        match call.tool_name {
            "write_file" | "edit_file" | "bash" => {
                ToolDecision::Deny("read-only session".into())
            }
            _ => ToolDecision::Allow,
        }
    }
}

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
    .with_tools(default_tools())
    .with_tool_middleware(ReadOnlyPolicy);
```

Semantics:

- **`Allow`** — the call proceeds (with the current arguments).
- **`Modify(args)`** — the call proceeds with replacement arguments (e.g.
  rewrite a path into a sandbox). Later middleware in the chain see the
  rewritten arguments; the `ToolExecutionStart` event carries what actually
  runs.
- **`Deny(reason)`** — the call never executes. The reason is returned to the
  LLM as an **error tool result** (`"Tool call denied: ..."`), so the model can
  adapt — pick another tool, ask the user — and the loop continues. A denial
  never aborts the run.

A middleware that panics is contained: the call is denied (reason `"tool
middleware panicked"`) and the loop continues — a buggy policy can't kill the
run.

The hook is `async`, so an interactive app can prompt a human before deciding.
Note that under the default `Parallel` execution strategy, middleware for
parallel tool calls run concurrently — if you need one-at-a-time approval UX,
serialize inside your middleware (e.g. a `tokio::sync::Mutex`) or switch to
`ToolExecutionStrategy::Sequential`.

Sub-agents gate their own tool calls the same way via
`SubAgentTool::with_tool_middleware`.
