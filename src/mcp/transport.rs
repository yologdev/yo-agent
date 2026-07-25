//! MCP transport implementations: stdio and HTTP+SSE.

use super::types::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Transport trait for MCP communication.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and receive the response.
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError>;
    /// Close the transport.
    async fn close(&self) -> Result<(), McpError>;
}

// ---------------------------------------------------------------------------
// Stdio Transport
// ---------------------------------------------------------------------------

/// Communicates with an MCP server via stdin/stdout of a child process.
/// One JSON-RPC message per line (newline-delimited JSON).
pub struct StdioTransport {
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    stdout: Arc<Mutex<BufReader<tokio::process::ChildStdout>>>,
    child: Arc<Mutex<Child>>,
}

impl StdioTransport {
    /// Spawn a child process and create a stdio transport.
    pub async fn new(
        command: &str,
        args: &[&str],
        env: Option<HashMap<String, String>>,
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(env_vars) = env {
            for (k, v) in env_vars {
                cmd.env(k, v);
            }
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("Failed to spawn '{}': {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture stdout".into()))?;

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
            child: Arc::new(Mutex::new(child)),
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        // Write request
        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| McpError::Transport(format!("Write error: {}", e)))?;
            stdin
                .flush()
                .await
                .map_err(|e| McpError::Transport(format!("Flush error: {}", e)))?;
        }

        // Read response
        let mut response_line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            let bytes_read = stdout
                .read_line(&mut response_line)
                .await
                .map_err(|e| McpError::Transport(format!("Read error: {}", e)))?;
            if bytes_read == 0 {
                return Err(McpError::ConnectionClosed);
            }
        }

        let response: JsonRpcResponse = serde_json::from_str(response_line.trim())?;
        Ok(response)
    }

    async fn close(&self) -> Result<(), McpError> {
        // Drop stdin to signal EOF, then kill the child
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HTTP Transport
// ---------------------------------------------------------------------------

/// Communicates with an MCP server via HTTP POST (JSON-RPC over HTTP).
///
/// Covers the **request/response subset of Streamable HTTP**: responses framed
/// as `text/event-stream`, `Mcp-Session-Id` capture and replay, `202 Accepted`
/// with no body, and session teardown on [`close`](McpTransport::close). Plain
/// JSON-RPC bodies keep working unchanged.
///
/// Not covered: the `GET` server→client stream and `Last-Event-ID`
/// resumability. [`McpTransport`] is strictly request/response, so a
/// server-initiated message has nowhere to be delivered; supporting those would
/// need a different transport trait.
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    /// Session assigned by the server on `initialize`, replayed on subsequent
    /// requests. `Mutex` because [`McpTransport::send`] takes `&self`.
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    /// Create a new HTTP transport.
    pub fn new(url: &str) -> Result<Self, McpError> {
        let client = reqwest::Client::new();
        Ok(Self {
            client,
            base_url: url.trim_end_matches('/').to_string(),
            session_id: Mutex::new(None),
        })
    }

    /// Extract a JSON-RPC response from a body that may be raw JSON or SSE.
    ///
    /// Tries JSON first so plain JSON-RPC servers take the same path they
    /// always did. Otherwise walks SSE frames for the first `data:` line that
    /// parses — servers may interleave comments, `event:`, and `id:` lines.
    fn parse_body(body: &str) -> Result<JsonRpcResponse, McpError> {
        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(body) {
            return Ok(response);
        }

        for frame in body.split("\n\n") {
            for line in frame.lines() {
                let Some(data) = line
                    .strip_prefix("data:")
                    .map(|d| d.trim_start_matches(' ').trim())
                else {
                    continue;
                };
                if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(data) {
                    return Ok(response);
                }
            }
        }

        Err(McpError::Transport(format!(
            "could not parse a JSON-RPC response from the body (tried JSON and SSE frames): {}",
            body.chars().take(200).collect::<String>()
        )))
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        let request_id = request.id;

        let mut builder = self
            .client
            .post(&self.base_url)
            // Streamable HTTP servers pick their framing from this; JSON-only
            // servers still match `application/json`.
            .header("Accept", "application/json, text/event-stream")
            .json(&request);

        if let Some(session) = self.session_id.lock().await.as_ref() {
            builder = builder.header("Mcp-Session-Id", session);
        }

        let resp = builder
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(McpError::Transport(format!(
                "HTTP {} from server",
                resp.status()
            )));
        }

        // The server assigns the session on `initialize`; hold it for the rest
        // of the connection.
        if let Some(session) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
        {
            *self.session_id.lock().await = Some(session);
        }

        let body = resp
            .text()
            .await
            .map_err(|e| McpError::Transport(format!("Response read error: {}", e)))?;

        // 202 Accepted with no body is how Streamable HTTP acknowledges a
        // notification. There is no JSON-RPC response to return, so synthesize
        // an empty success rather than fail a call that did succeed.
        if body.trim().is_empty() {
            return Ok(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: Some(request_id),
                result: None,
                error: None,
            });
        }

        Self::parse_body(&body)
    }

    async fn close(&self) -> Result<(), McpError> {
        // Without a session there is nothing server-side to release.
        let session = self.session_id.lock().await.take();
        if let Some(session) = session {
            // Best-effort: session teardown is optional in the spec and plenty
            // of servers reject DELETE. Failing close() over it would turn a
            // successful run into an error.
            let _ = self
                .client
                .delete(&self.base_url)
                .header("Mcp-Session-Id", session)
                .send()
                .await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stdio_transport_with_cat() {
        // Use `cat` as a simple echo server — it reflects stdin to stdout.
        let transport = StdioTransport::new("cat", &[], None).await.unwrap();

        let request = JsonRpcRequest::new("test/echo", Some(serde_json::json!({"hello": "world"})));
        let request_id = request.id;

        // Write the request; cat will echo it back as-is.
        // Since cat echoes JSON-RPC requests, the "response" will actually be the request.
        // This tests the transport layer I/O, not protocol correctness.
        let mut line = serde_json::to_string(&request).unwrap();
        line.push('\n');

        {
            let mut stdin = transport.stdin.lock().await;
            stdin.write_all(line.as_bytes()).await.unwrap();
            stdin.flush().await.unwrap();
        }

        let mut response_line = String::new();
        {
            let mut stdout = transport.stdout.lock().await;
            stdout.read_line(&mut response_line).await.unwrap();
        }

        // Cat echoes the request, so we can parse it as a request
        let echoed: JsonRpcRequest = serde_json::from_str(response_line.trim()).unwrap();
        assert_eq!(echoed.id, request_id);
        assert_eq!(echoed.method, "test/echo");

        transport.close().await.unwrap();
    }

    #[test]
    fn test_http_transport_creation() {
        let transport = HttpTransport::new("http://localhost:8080/mcp").unwrap();
        assert_eq!(transport.base_url, "http://localhost:8080/mcp");

        // Trailing slash stripped
        let transport = HttpTransport::new("http://localhost:8080/mcp/").unwrap();
        assert_eq!(transport.base_url, "http://localhost:8080/mcp");
    }
}
