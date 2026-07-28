//! MCP transport implementations: stdio and HTTP+SSE.

use super::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, warn};

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
/// Covers the **request/response subset of Streamable HTTP** — servers that
/// answer a POST with an SSE-framed response and then close the stream:
/// `text/event-stream` bodies, `Mcp-Session-Id` capture and replay, `202`/`204`
/// acknowledgements, and session teardown on [`close`](McpTransport::close).
/// Plain JSON-RPC bodies keep working.
///
/// The body is parsed incrementally and [`send`](McpTransport::send) returns at
/// the frame carrying this request's response, so a server that keeps the POST
/// stream open after answering does not block the call.
///
/// Not covered: the `GET` server→client stream and `Last-Event-ID`
/// resumability. [`McpTransport`] is `send`/`close` only, so a server-initiated
/// message has nowhere to be delivered — supporting them would mean growing the
/// trait an inbound channel. Notifications that arrive on the POST stream
/// *before* the response are read and skipped; any that trail it are not, since
/// the call has already returned by then.
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

    /// Decide whether `payload` is *this request's* JSON-RPC response.
    ///
    /// Selection is structural rather than "does it deserialize". Every field
    /// of [`JsonRpcResponse`] except `jsonrpc` is optional and unknown keys are
    /// ignored, so a server→client notification like
    /// `{"jsonrpc":"2.0","method":"notifications/progress",...}` deserializes
    /// cleanly into an all-`None` shell. Streamable HTTP servers routinely emit
    /// progress and logging notifications on the POST stream *ahead of* the
    /// result, so accepting the first thing that parses would return the
    /// notification and silently discard the answer.
    fn response_for(payload: &str, request_id: u64) -> Option<JsonRpcResponse> {
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        // Responses never carry `method`; notifications and server→client
        // requests always do. Redundant with the result-or-error check below
        // for every frame shape seen in practice — kept because it states the
        // JSON-RPC invariant directly, so the next reader does not have to
        // derive it.
        if value.get("method").is_some() {
            return None;
        }
        // A response carries a result or an error. This is the check that also
        // rejects a bare `{"jsonrpc":"2.0","id":N}` ack, which carries no
        // method and would otherwise pass.
        if value.get("result").is_none() && value.get("error").is_none() {
            return None;
        }
        let response: JsonRpcResponse = serde_json::from_value(value).ok()?;
        // Correlate. An absent id is tolerated: JSON-RPC allows a null id on
        // errors the server could not attribute to a request.
        if response.id.is_some_and(|id| id != request_id) {
            return None;
        }
        Some(response)
    }

    /// Join an SSE event's `data:` lines into its payload, per the SSE spec.
    ///
    /// Returns `None` for events that carry no data — comments, bare `event:`
    /// or `id:` lines, keep-alives.
    fn event_payload(event: &str) -> Option<String> {
        let data: Vec<&str> = event
            .lines()
            .filter_map(|line| {
                line.strip_prefix("data:")
                    .map(|d| d.trim_start_matches(' '))
            })
            .collect();
        if data.is_empty() {
            return None;
        }
        let payload = data.join("\n");
        let payload = payload.trim().to_string();
        (!payload.is_empty()).then_some(payload)
    }

    /// Note a frame that was valid JSON-RPC but not our answer, so a call that
    /// never finds its response can say what it saw instead.
    fn note_skipped(payload: &str, method: &str, skipped: &mut Vec<String>) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            let what = match value.get("method").and_then(|m| m.as_str()) {
                Some(m) => m.to_string(),
                None => match value.get("id").and_then(|i| i.as_u64()) {
                    Some(id) => format!("response for id {id}"),
                    None => "unrecognized JSON-RPC frame".to_string(),
                },
            };
            debug!("skipping SSE frame that is not the response to '{method}': {what}");
            skipped.push(what);
        }
    }

    /// Read the response body, returning as soon as this request's answer
    /// arrives rather than draining to EOF.
    ///
    /// The early return is the point: Streamable HTTP permits a server to keep
    /// the POST stream open after answering (to deliver further notifications),
    /// so buffering the whole body would block until the server gave up. It
    /// also means a long `tools/call` that streams progress frames returns the
    /// moment the result lands, instead of waiting out the trailing traffic.
    ///
    /// A plain JSON-RPC body has no event boundaries, so it falls through to
    /// the whole-body parse at EOF — the same result as before, and the shape
    /// that made `resp.json()` sufficient until now.
    async fn read_response(
        resp: reqwest::Response,
        request_id: u64,
        method: &str,
        status: reqwest::StatusCode,
    ) -> Result<JsonRpcResponse, McpError> {
        // Scanning happens on bytes, not text: a chunk boundary can split a
        // multi-byte character, and decoding each chunk independently would
        // corrupt it. Whole events decode cleanly.
        let mut buf: Vec<u8> = Vec::new();
        let mut scanned = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                McpError::Transport(format!("Response read error on '{method}': {e}"))
            })?;
            // Drop CR so event boundaries are plain `\n\n` whichever framing the
            // server uses. A raw CR is illegal inside a JSON string, so this
            // cannot corrupt a payload.
            buf.extend(chunk.iter().copied().filter(|b| *b != b'\r'));

            while let Some(pos) = buf[scanned..].windows(2).position(|w| w == b"\n\n") {
                let event = String::from_utf8_lossy(&buf[scanned..scanned + pos]).into_owned();
                scanned += pos + 2;

                let Some(payload) = Self::event_payload(&event) else {
                    continue;
                };
                if let Some(response) = Self::response_for(&payload, request_id) {
                    return Ok(response);
                }
                Self::note_skipped(&payload, method, &mut skipped);
            }
        }

        let body = String::from_utf8_lossy(&buf).into_owned();

        if body.trim().is_empty() {
            // 202/204 with no body is how Streamable HTTP acknowledges a
            // notification — there is no JSON-RPC response to return, so
            // synthesize an empty success. Any other empty 2xx is a real
            // failure (a proxy answering instead of the MCP server, a drained
            // upstream) and must not be dressed up as one.
            if status == reqwest::StatusCode::ACCEPTED || status == reqwest::StatusCode::NO_CONTENT
            {
                return Ok(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: Some(request_id),
                    result: None,
                    error: None,
                });
            }
            return Err(McpError::Transport(format!(
                "HTTP {status} with an empty body on '{method}' (expected a JSON-RPC response; \
                 an empty 2xx usually means a proxy or gateway answered instead of the MCP server)"
            )));
        }

        // Plain JSON-RPC: no event boundaries, so nothing matched above.
        if let Some(response) = Self::response_for(&body, request_id) {
            return Ok(response);
        }

        // A final event the server never terminated with a blank line.
        if scanned < buf.len() {
            let tail = String::from_utf8_lossy(&buf[scanned..]).into_owned();
            if let Some(payload) = Self::event_payload(&tail) {
                if let Some(response) = Self::response_for(&payload, request_id) {
                    return Ok(response);
                }
                Self::note_skipped(&payload, method, &mut skipped);
            }
        }

        let seen = if skipped.is_empty() {
            String::new()
        } else {
            format!(
                " (skipped {} frame(s): {})",
                skipped.len(),
                skipped.join(", ")
            )
        };
        Err(McpError::Transport(format!(
            "HTTP {status}: no JSON-RPC response for '{method}' (id {request_id}) in the body{seen}: {}",
            body.chars().take(200).collect::<String>()
        )))
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        let request_id = request.id;
        let method = request.method.clone();

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

        let status = resp.status();
        if !status.is_success() {
            // Per the spec, 404 on a request carrying a session means the
            // server dropped it. Keeping the dead id would make every later
            // call fail identically, with an error that reads like a bad URL.
            if status == reqwest::StatusCode::NOT_FOUND {
                if let Some(dead) = self.session_id.lock().await.take() {
                    warn!(
                        "MCP session {dead} was rejected (HTTP 404); reconnect to start a new one"
                    );
                    return Err(McpError::Transport(format!(
                        "MCP session expired (HTTP 404 on '{method}'); reconnect to start a new session"
                    )));
                }
            }
            // Carry the body: servers explain themselves in it, and dropping it
            // leaves the caller with a bare status to guess from.
            let body = resp.text().await.unwrap_or_default();
            let detail = body.trim();
            return Err(McpError::Transport(if detail.is_empty() {
                format!("HTTP {status} from server on '{method}'")
            } else {
                format!(
                    "HTTP {status} from server on '{method}': {}",
                    detail.chars().take(200).collect::<String>()
                )
            }));
        }

        // Servers assign the session on `initialize`; any response carrying one
        // updates it.
        match resp.headers().get("mcp-session-id").map(|v| v.to_str()) {
            Some(Ok(session)) => *self.session_id.lock().await = Some(session.to_owned()),
            // A header we cannot read means every later request goes out
            // sessionless — the server then rejects them or silently starts a
            // fresh session, discarding the handshake. Only the operator can
            // fix that, so say so.
            Some(Err(e)) => warn!("ignoring unreadable Mcp-Session-Id header: {e}"),
            None => {}
        }

        Self::read_response(resp, request_id, &method, status).await
    }

    async fn close(&self) -> Result<(), McpError> {
        let session = self.session_id.lock().await.take();
        if let Some(session) = session {
            // Best-effort: session teardown is optional in the spec and plenty
            // of servers reject DELETE. Failing close() over it would turn a
            // successful run into an error. Best-effort is not the same as
            // unobservable, though — a rejection is fine, but a DELETE that
            // never reached the server leaks the session there, and that only
            // surfaces later as an unrelated connect failure.
            match self
                .client
                .delete(&self.base_url)
                .header("Mcp-Session-Id", &session)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {}
                Ok(resp) => debug!(
                    "MCP session teardown rejected with HTTP {}; it is optional, continuing",
                    resp.status()
                ),
                Err(e) => warn!(
                    "MCP session {session} teardown did not reach {}: {e}; \
                     the session may leak server-side",
                    self.base_url
                ),
            }
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
