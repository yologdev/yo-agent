//! MCP transport implementations: stdio and HTTP+SSE.

use super::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::{BTreeMap, HashMap};
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
/// answer a POST with an SSE-framed response, whether or not they then close
/// the stream: `text/event-stream` bodies, `Mcp-Session-Id` capture and replay,
/// `202`/`204` acknowledgements, and session teardown on
/// [`close`](McpTransport::close). Plain JSON-RPC bodies keep working.
///
/// The body is parsed incrementally and [`send`](McpTransport::send) returns at
/// the blank-line-terminated frame carrying this request's response, so a
/// server that keeps the POST stream open after answering does not block the
/// call. Two consequences worth knowing: returning mid-body forgoes connection
/// reuse (a server that has not already closed the body costs a fresh
/// connection, and TLS handshake, next call), and a plain JSON-RPC body has no
/// frames to return early at, so it is read to the end as before.
///
/// A stalled server — one that accepts the POST and then sends nothing — is
/// bounded by an idle read timeout (120s) rather than hanging. The timer resets
/// on every read, so a slow-but-progressing call is never cut off.
///
/// Not covered: the `GET` server→client stream and `Last-Event-ID`
/// resumability. [`McpTransport`] is `send`/`close` only, so a server-initiated
/// message has nowhere to be delivered — supporting them would mean growing the
/// trait an inbound channel. Notifications that arrive on the POST stream
/// *before* the response are read and skipped; any that trail it are not, since
/// the call has already returned by then. A server that blocks awaiting a reply
/// to a `sampling/createMessage` it sent on this stream will therefore time out
/// rather than be answered.
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    /// Session assigned by the server on `initialize`, replayed on subsequent
    /// requests. `Mutex` because [`McpTransport::send`] takes `&self`.
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    /// Idle bound between reads, not a bound on the whole call.
    ///
    /// A `tools/call` may legitimately run for minutes; what must not be
    /// tolerated is a server that accepts the POST and then sends *nothing*.
    /// `read_timeout` resets on every successful read, so a long call that
    /// streams progress frames keeps its connection alive while a stalled one
    /// is cut. (A whole-request `timeout` cannot tell those apart, which is why
    /// it is deliberately not used here.)
    const READ_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

    /// Create a new HTTP transport.
    pub fn new(url: &str) -> Result<Self, McpError> {
        let client = reqwest::Client::builder()
            .read_timeout(Self::READ_IDLE_TIMEOUT)
            .build()
            .map_err(|e| McpError::Transport(format!("Failed to build HTTP client: {e}")))?;
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
        // Responses never carry `method`. The result-or-error check below
        // happens to reject every *well-formed* frame this one does; what this
        // still catches is a malformed frame carrying both — a server that
        // conflates the two shapes. Cheap insurance, and pinned by
        // `frame_with_both_method_and_result_is_not_the_response`.
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

    /// Note a frame that was valid JSON-RPC but not our answer.
    ///
    /// Counted by kind rather than collected verbatim: this ends up in an error
    /// string that `McpToolAdapter` hands to the model as a tool result, and a
    /// progress-heavy stream can emit thousands of identical frame names.
    /// Reaches a caller only on the EOF path — a mid-stream read error reports
    /// its own cause instead.
    fn note_skipped(payload: &str, method: &str, skipped: &mut BTreeMap<String, usize>) {
        let what = match serde_json::from_str::<serde_json::Value>(payload) {
            Ok(value) => match value.get("method").and_then(|m| m.as_str()) {
                Some(m) => m.to_string(),
                None => match value.get("id").and_then(|i| i.as_u64()) {
                    Some(id) => format!("response for id {id}"),
                    None => "unrecognized JSON-RPC frame".to_string(),
                },
            },
            // No `else`-less `if let` here: a frame that is not valid JSON may
            // be the server's own answer, truncated mid-write. Dropping it
            // unrecorded is how that becomes undiagnosable.
            Err(e) => {
                warn!(
                    "SSE frame on '{method}' is not valid JSON ({} bytes): {e}; \
                     a truncated frame may be the server's real answer",
                    payload.len()
                );
                format!("malformed non-JSON frame ({} bytes)", payload.len())
            }
        };
        debug!("skipping SSE frame that is not the response to '{method}': {what}");
        *skipped.entry(what).or_default() += 1;
    }

    /// Render the skipped-frame tally for an error message.
    fn describe_skipped(skipped: &BTreeMap<String, usize>) -> String {
        if skipped.is_empty() {
            return String::new();
        }
        let total: usize = skipped.values().sum();
        let listed: Vec<String> = skipped
            .iter()
            .take(10)
            .map(|(what, n)| {
                if *n > 1 {
                    format!("{what} x{n}")
                } else {
                    what.clone()
                }
            })
            .collect();
        let more = skipped.len().saturating_sub(listed.len());
        let tail = if more > 0 {
            format!(", and {more} other kind(s)")
        } else {
            String::new()
        };
        format!(" (skipped {total} frame(s): {}{tail})", listed.join(", "))
    }

    /// Decode an event slice, refusing rather than substituting on bad UTF-8.
    ///
    /// Lossy decoding would replace invalid bytes with U+FFFD, and since that
    /// is legal JSON string content the frame would go on to parse and be
    /// returned as a successful — but silently mutated — tool result.
    fn decode(bytes: &[u8], method: &str) -> Result<String, McpError> {
        std::str::from_utf8(bytes).map(str::to_owned).map_err(|e| {
            McpError::Transport(format!(
                "invalid UTF-8 in the response body on '{method}' at byte {}: {e}",
                e.valid_up_to()
            ))
        })
    }

    /// Read the response body, returning as soon as this request's answer
    /// arrives rather than draining to EOF.
    ///
    /// The early return is the point: Streamable HTTP permits a server to keep
    /// the POST stream open after answering, so buffering the whole body would
    /// block until the server gave up. It also means a long `tools/call` that
    /// streams progress frames returns the moment the result lands, instead of
    /// waiting out the trailing traffic.
    ///
    /// Two costs come with it, both deliberate. Returning mid-body prevents the
    /// connection from being pooled, so a server that has not already closed
    /// the body costs a fresh connection (and TLS handshake) on the next call.
    /// And the early return only fires on a **blank-line-terminated** frame: an
    /// unterminated final event is recovered at EOF instead, so a server that
    /// neither terminates the frame nor closes the stream is bounded by the
    /// client's read timeout rather than returning promptly.
    ///
    /// A plain JSON-RPC body yields no `data:`-prefixed lines — a raw newline is
    /// illegal inside a JSON string, so no line can begin mid-string — and so
    /// never produces an event payload. It falls through to the whole-body parse
    /// at EOF. Note this reverses the previous ordering: JSON bodies now
    /// traverse the SSE scan first.
    async fn read_response(
        resp: reqwest::Response,
        request_id: u64,
        method: &str,
        status: reqwest::StatusCode,
    ) -> Result<JsonRpcResponse, McpError> {
        // Both `application/json` and `text/event-stream` mandate UTF-8, and
        // reading the body as a byte stream gives up the charset transcoding
        // `Response::text()` would have done. Refuse a declared non-UTF-8
        // charset rather than hand back mangled text.
        if let Some(charset) = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .and_then(|ct| {
                ct.split(';').skip(1).find_map(|p| {
                    p.trim()
                        .strip_prefix("charset=")
                        .map(|c| c.trim_matches('"').to_ascii_lowercase())
                })
            })
        {
            if !matches!(charset.as_str(), "utf-8" | "utf8" | "us-ascii" | "ascii") {
                return Err(McpError::Transport(format!(
                    "unsupported charset '{charset}' on '{method}': MCP bodies are UTF-8 \
                     (both application/json and text/event-stream mandate it)"
                )));
            }
        }

        // Scanning happens on bytes, not text: a chunk boundary can split a
        // multi-byte character, and decoding each chunk independently would
        // corrupt it. Whole events decode cleanly, and 0x0D can never appear
        // inside a multi-byte sequence (continuation bytes are >= 0x80), which
        // is what makes the CR normalization below safe to do pre-decode.
        //
        // `buf` is never compacted — `scanned` is a read cursor — so a stream
        // of discarded progress frames is retained for the life of the call.
        let mut buf: Vec<u8> = Vec::new();
        let mut scanned = 0usize;
        // Boundary-free below this point; a new 2-byte window can only be
        // completed by a newly-appended byte. Without it, a body with no
        // boundary (a plain JSON body, or one large SSE frame) rescans
        // everything on every chunk — quadratic, and seconds of CPU on a
        // multi-megabyte result.
        let mut searched = 0usize;
        let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
        let mut pending_cr = false;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                McpError::Transport(format!(
                    "Response read error on '{method}' after {} byte(s){}: {e}",
                    buf.len(),
                    Self::describe_skipped(&skipped)
                ))
            })?;

            // SSE accepts CRLF, LF, or a bare CR as a line terminator.
            // Normalize all three to LF so boundaries are plain `\n\n`.
            // `pending_cr` carries the state across a chunk that ends mid-CRLF.
            // Within a JSON payload a raw CR is legal only as inter-token
            // whitespace (it is illegal unescaped inside a string, and a `\r`
            // escape is two ASCII bytes), so rewriting it cannot alter a value.
            for &b in chunk.iter() {
                if b == b'\r' {
                    buf.push(b'\n');
                    pending_cr = true;
                } else {
                    if pending_cr && b == b'\n' {
                        pending_cr = false;
                        continue;
                    }
                    pending_cr = false;
                    buf.push(b);
                }
            }

            loop {
                let from = scanned.max(searched);
                let Some(pos) = buf[from..].windows(2).position(|w| w == b"\n\n") else {
                    // The trailing byte may yet start a boundary.
                    searched = buf.len().saturating_sub(1);
                    break;
                };
                let end = from + pos;
                let event = Self::decode(&buf[scanned..end], method)?;
                scanned = end + 2;
                searched = scanned;

                let Some(payload) = Self::event_payload(&event) else {
                    continue;
                };
                if let Some(response) = Self::response_for(&payload, request_id) {
                    return Ok(response);
                }
                Self::note_skipped(&payload, method, &mut skipped);
            }
        }

        let body = Self::decode(&buf, method)?;
        // `Response::text()` used to strip a BOM; `from_utf8` does not, and
        // U+FEFF is not whitespace, so `trim()` leaves it in place to break the
        // JSON parse below.
        let body = body.strip_prefix('\u{feff}').unwrap_or(&body).to_string();

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

        // Plain JSON-RPC: never yields a `data:` line, so nothing matched above.
        if let Some(response) = Self::response_for(&body, request_id) {
            return Ok(response);
        }

        // A final event the server never terminated with a blank line.
        if scanned < buf.len() {
            let tail = Self::decode(&buf[scanned..], method)?;
            if let Some(payload) = Self::event_payload(&tail) {
                if let Some(response) = Self::response_for(&payload, request_id) {
                    return Ok(response);
                }
                Self::note_skipped(&payload, method, &mut skipped);
            }
        }

        Err(McpError::Transport(format!(
            "HTTP {status}: no JSON-RPC response for '{method}' (id {request_id}) in the body{}: {}",
            Self::describe_skipped(&skipped),
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
