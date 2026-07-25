# MCP Integration

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a JSON-RPC 2.0 protocol that lets AI agents discover and call tools from external servers. It defines a standard way for agents to connect to tool providers over two transports:

- **Stdio** — spawn a child process, communicate via stdin/stdout (newline-delimited JSON)
- **HTTP** — POST JSON-RPC requests to an HTTP endpoint, including the
  request/response subset of Streamable HTTP

## Connecting to MCP Servers

### Stdio Transport

Use `with_mcp_server_stdio()` to spawn an MCP server process and register its tools:

```rust
use yoagent::Agent;
use yoagent::provider::ModelConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
        .with_system_prompt("You are a helpful assistant with file access.")
        .with_mcp_server_stdio(
            "npx",
            &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            None,
        )
        .await?;

    let rx = agent.prompt("List files in /tmp").await;
    // handle events...
    Ok(())
}
```

You can pass environment variables to the server process:

```rust
use std::collections::HashMap;

let mut env = HashMap::new();
env.insert("API_TOKEN".into(), "secret".into());

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_mcp_server_stdio("my-mcp-server", &["--port", "0"], Some(env))
    .await?;
```

### HTTP Transport

For remote MCP servers exposed over HTTP:

```rust
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_mcp_server_http("http://localhost:8080/mcp")
    .await?;
```

`HttpTransport` handles both the plain JSON-RPC-over-POST shape and the
**request/response subset of Streamable HTTP**:

- Responses framed as `text/event-stream` are parsed out of their SSE frames,
  so servers following the Streamable HTTP spec work without a custom transport.
- Requests advertise `Accept: application/json, text/event-stream`, letting the
  server pick its framing.
- An `Mcp-Session-Id` returned on `initialize` is captured and replayed on every
  later request, and released with a `DELETE` when the client closes. Servers
  that reject `DELETE` are tolerated — teardown is best-effort.
- `202 Accepted` with an empty body (how notifications are acknowledged) is a
  success, not a parse failure.

**Not supported:** the `GET` server→client stream and `Last-Event-ID`
resumability. `McpTransport` is strictly request/response, so a server-initiated
message has nowhere to be delivered — those would need a different transport
trait. In practice this only matters for servers that push notifications
unprompted; ordinary tool discovery and invocation are unaffected.

## How MCP Tools Work

When you call `with_mcp_server_stdio()` or `with_mcp_server_http()`, yoagent:

1. Connects to the MCP server and performs the `initialize` handshake
2. Calls `tools/list` to discover available tools
3. Wraps each MCP tool as an `AgentTool` via `McpToolAdapter`
4. Adds them to the agent's tool list

MCP tools appear alongside built-in tools. The LLM sees them with their original names, descriptions, and JSON Schema parameters — it can call them just like any other tool.

## Mixing Built-in and MCP Tools

```rust
use yoagent::tools::default_tools;

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_tools(default_tools())  // bash, read, write, edit, list, search
    .with_mcp_server_stdio("my-db-server", &[], None)
    .await?;
// Agent now has both built-in coding tools AND MCP database tools
```

## Using the MCP Client Directly

For lower-level control, use `McpClient` directly:

```rust
use yoagent::mcp::{McpClient, McpToolAdapter};
use std::sync::Arc;
use tokio::sync::Mutex;

let client = McpClient::connect_stdio("my-server", &[], None).await?;
let tools = client.list_tools().await?;

for tool in &tools {
    println!("{}: {}", tool.name, tool.description.as_deref().unwrap_or(""));
}

// Call a tool directly
let result = client.call_tool("read_file", serde_json::json!({"path": "/tmp/test.txt"})).await?;

// Or wrap as AgentTool adapters
let client = Arc::new(Mutex::new(client));
let adapters = McpToolAdapter::from_client(client).await?;
```

## Error Handling

MCP operations return `McpError`:

- `McpError::Transport` — connection or I/O failure
- `McpError::Protocol` — unexpected response format
- `McpError::JsonRpc` — server returned a JSON-RPC error
- `McpError::ConnectionClosed` — server process exited

When an MCP tool returns `isError: true`, the adapter converts it to a `ToolError::Failed`, which the agent loop sends back to the LLM with `is_error: true` so it can self-correct.
