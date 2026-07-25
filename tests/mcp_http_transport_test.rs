//! Tests for `HttpTransport` — the request/response subset of MCP Streamable
//! HTTP (issue #82), plus the plain JSON-RPC behavior it must not regress.

use wiremock::matchers::{body_string_contains, header, method};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
use yoagent::mcp::transport::McpTransport;
use yoagent::mcp::types::JsonRpcRequest;
use yoagent::mcp::HttpTransport;

/// Matcher: the request must NOT carry the given header.
struct HeaderAbsent(&'static str);

impl wiremock::Match for HeaderAbsent {
    fn matches(&self, request: &Request) -> bool {
        !request.headers.contains_key(self.0)
    }
}

/// Matcher: the named header's value contains the given substring.
///
/// `wiremock::matchers::header` splits comma-separated values and compares each
/// one, so it can never match a multi-value header like
/// `Accept: application/json, text/event-stream` as a whole string.
struct HeaderContains(&'static str, &'static str);

impl wiremock::Match for HeaderContains {
    fn matches(&self, request: &Request) -> bool {
        request
            .headers
            .get(self.0)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains(self.1))
    }
}

fn sse(payload: &str) -> String {
    format!("event: message\ndata: {payload}\n\n")
}

/// Backward compatibility: a plain JSON-RPC body keeps working. This was the
/// only shape supported before #82.
#[tokio::test]
async fn plain_json_response_still_parses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#,
            "application/json",
        ))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    let response = transport
        .send(JsonRpcRequest::new("ping", None))
        .await
        .expect("plain JSON must still parse");

    assert_eq!(response.result.unwrap()["ok"], true);
}

/// The reported failure: servers returning `text/event-stream` used to hit
/// `Response parse error` because the body was parsed as a single JSON object.
#[tokio::test]
async fn sse_framed_response_parses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    let response = transport
        .send(JsonRpcRequest::new("tools/list", None))
        .await
        .expect("SSE-framed response must parse");

    assert!(response.result.unwrap()["tools"].is_array());
}

/// Real SSE streams carry comments, `event:` and `id:` lines, and may deliver
/// the payload in a later frame — the parser must walk to it.
#[tokio::test]
async fn sse_with_noise_frames_finds_the_payload() {
    let server = MockServer::start().await;
    let body = concat!(
        ": keep-alive comment\n\n",
        "event: ping\ndata: {\"unrelated\":true}\n\n",
        "id: 42\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"found\":true}}\n\n"
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    let response = transport
        .send(JsonRpcRequest::new("tools/list", None))
        .await
        .expect("payload in a later frame must be found");

    assert_eq!(response.result.unwrap()["found"], true);
}

/// Streamable HTTP assigns a session on `initialize` and expects it echoed on
/// every subsequent request.
#[tokio::test]
async fn session_id_is_captured_then_replayed() {
    let server = MockServer::start().await;

    // First call: no session header yet; server assigns one.
    Mock::given(method("POST"))
        .and(HeaderAbsent("mcp-session-id"))
        .and(body_string_contains("initialize"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Mcp-Session-Id", "sess-abc123")
                .set_body_raw(
                    sse(r#"{"jsonrpc":"2.0","id":1,"result":{"initialized":true}}"#),
                    "text/event-stream",
                ),
        )
        .expect(1)
        .mount(&server)
        .await;

    // Second call: must carry the assigned session.
    Mock::given(method("POST"))
        .and(header("mcp-session-id", "sess-abc123"))
        .and(body_string_contains("tools/list"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#),
            "text/event-stream",
        ))
        .expect(1)
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    transport
        .send(JsonRpcRequest::new("initialize", None))
        .await
        .expect("initialize");
    transport
        .send(JsonRpcRequest::new("tools/list", None))
        .await
        .expect("follow-up must reuse the session");
    // Mock `expect(1)` assertions verify the headers when the server drops.
}

/// Notifications get `202 Accepted` with no body. `send()` must still return a
/// response rather than failing a call that succeeded.
#[tokio::test]
async fn accepted_with_empty_body_is_not_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(202))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    let request = JsonRpcRequest::new("notifications/initialized", None);
    let request_id = request.id;

    let response = transport
        .send(request)
        .await
        .expect("202 with no body must not be an error");

    assert_eq!(response.id, Some(request_id));
    assert!(response.result.is_none() && response.error.is_none());
}

/// Every request advertises both framings so a Streamable HTTP server can pick.
#[tokio::test]
async fn accept_header_advertises_both_framings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(HeaderContains("accept", "application/json"))
        .and(HeaderContains("accept", "text/event-stream"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
            "application/json",
        ))
        .expect(1)
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    transport
        .send(JsonRpcRequest::new("ping", None))
        .await
        .expect("send");
}

/// `close()` releases the session server-side. Without a session it must stay a
/// no-op — the pre-#82 behavior.
#[tokio::test]
async fn close_deletes_the_session_when_one_exists() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Mcp-Session-Id", "sess-xyz")
                .set_body_raw(
                    r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
                    "application/json",
                ),
        )
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(header("mcp-session-id", "sess-xyz"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    transport
        .send(JsonRpcRequest::new("initialize", None))
        .await
        .expect("initialize");
    transport.close().await.expect("close");
}

/// A server that rejects DELETE must not turn a successful run into an error.
#[tokio::test]
async fn close_ignores_a_server_that_rejects_delete() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Mcp-Session-Id", "sess-xyz")
                .set_body_raw(
                    r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
                    "application/json",
                ),
        )
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(405))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    transport
        .send(JsonRpcRequest::new("initialize", None))
        .await
        .expect("initialize");
    transport
        .close()
        .await
        .expect("a 405 on DELETE must not fail close()");
}

/// An unparseable body must still be a loud error, with a snippet to debug from.
#[tokio::test]
async fn unparseable_body_is_an_error_with_context() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("<html>gateway</html>", "text/html"))
        .mount(&server)
        .await;

    let transport = HttpTransport::new(&server.uri()).unwrap();
    let err = transport
        .send(JsonRpcRequest::new("ping", None))
        .await
        .expect_err("an unparseable body must be an error");

    assert!(
        err.to_string().contains("gateway"),
        "error should quote the body to debug from, got: {err}"
    );
}
