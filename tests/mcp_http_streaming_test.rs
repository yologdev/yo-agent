//! Proves `HttpTransport` returns as soon as this request's response frame
//! arrives, rather than draining the body to EOF.
//!
//! Streamable HTTP permits a server to keep the POST stream open after
//! answering. Reading to completion would block until the server gave up — so
//! this uses a raw TCP server that writes the response and then deliberately
//! never terminates the chunked body. `wiremock` cannot express that: it always
//! writes a complete body and closes.

use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use yoagent::mcp::transport::McpTransport;
use yoagent::mcp::types::JsonRpcRequest;
use yoagent::mcp::HttpTransport;

/// Serves one request: headers, then a chunk carrying `frames`, then holds the
/// connection open without the terminating zero-length chunk.
///
/// Returns the bound address. The task is detached; it parks until the test
/// process ends.
async fn serve_then_hold_open(frames: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = format!("http://{}", listener.local_addr().unwrap());

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();

        // Drain the request head so the client's write completes.
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;

        let head = "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n";
        socket.write_all(head.as_bytes()).await.unwrap();
        socket
            .write_all(format!("{:x}\r\n{}\r\n", frames.len(), frames).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();

        // No terminating "0\r\n\r\n": the body stays open, exactly like a server
        // holding the stream for further notifications.
        tokio::time::sleep(Duration::from_secs(300)).await;
    });

    addr
}

/// The response arrives in the first chunk; the server never closes the body.
/// Draining to EOF would hang here.
#[tokio::test]
async fn returns_without_waiting_for_the_server_to_close_the_stream() {
    let request = JsonRpcRequest::new("tools/list", None);
    let frames = format!(
        "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"tools\":[]}}}}\n\n",
        request.id
    );
    let addr = serve_then_hold_open(frames).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("send must return once the response frame arrives, not at EOF")
        .expect("the response must parse");

    assert!(response.result.unwrap()["tools"].is_array());
}

/// The same, with progress notifications ahead of the result: the transport
/// must skip them, take the result, and still not wait for EOF.
#[tokio::test]
async fn returns_at_the_result_frame_despite_leading_notifications() {
    let request = JsonRpcRequest::new("tools/call", None);
    let frames = format!(
        concat!(
            "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{\"progress\":1}}}}\n\n",
            "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{{\"level\":\"info\"}}}}\n\n",
            "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"content\":[{{\"type\":\"text\"}}]}}}}\n\n"
        ),
        request.id
    );
    let addr = serve_then_hold_open(frames).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("send must return at the result frame")
        .expect("the response must parse");

    assert!(
        response.result.is_some(),
        "the result must be selected over the notifications"
    );
}
