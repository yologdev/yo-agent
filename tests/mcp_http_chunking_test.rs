//! Incremental assembly tests for `HttpTransport`.
//!
//! The wiremock suite delivers each body as a single stream item, so it cannot
//! see anything that depends on where chunk boundaries fall. These drive a raw
//! TCP server that writes each piece as its own HTTP chunk, which is what makes
//! the seams observable.
//!
//! Separation comes from HTTP chunk framing, not from timing — no sleeps, and a
//! slower runner makes the pieces *more* separated, not less. Every timeout here
//! is a failure deadline, never a synchronisation point.

use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use yoagent::mcp::transport::McpTransport;
use yoagent::mcp::types::JsonRpcRequest;
use yoagent::mcp::HttpTransport;

/// Writes each piece as its own HTTP chunk. `terminate` controls whether the
/// body ever ends; when it does not, only an early return can complete the call.
///
/// Write errors are ignored on purpose: the transport hangs up as soon as the
/// answer lands, so later writes legitimately hit BrokenPipe. Unwrapping there
/// would panic inside a detached task and print noise that fails nothing.
async fn serve_pieces(pieces: Vec<Vec<u8>>, terminate: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = format!("http://{}", listener.local_addr().unwrap());

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;

        let head = "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n";
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.flush().await;

        for piece in pieces {
            let _ = socket
                .write_all(format!("{:x}\r\n", piece.len()).as_bytes())
                .await;
            let _ = socket.write_all(&piece).await;
            let _ = socket.write_all(b"\r\n").await;
            let _ = socket.flush().await;
        }
        if terminate {
            let _ = socket.write_all(b"0\r\n\r\n").await;
            let _ = socket.flush().await;
        }
        tokio::time::sleep(Duration::from_secs(300)).await;
    });

    addr
}

fn split_at(bytes: &[u8], at: usize) -> Vec<Vec<u8>> {
    vec![bytes[..at].to_vec(), bytes[at..].to_vec()]
}

/// The classic incremental-parser bug: scanning only the newly arrived bytes
/// misses a boundary whose two newlines straddle a chunk seam. Since the server
/// never closes the body here, that bug surfaces as the exact hang this
/// transport exists to avoid.
#[tokio::test]
async fn event_boundary_split_across_chunks_is_still_found() {
    let request = JsonRpcRequest::new("tools/list", None);
    let frames = format!(
        "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"tools\":[]}}}}\n\n",
        request.id
    );
    let at = frames.len() - 1;
    assert_eq!(
        &frames.as_bytes()[at - 1..at + 1],
        b"\n\n",
        "split the boundary"
    );
    let addr = serve_pieces(split_at(frames.as_bytes(), at), false).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("a boundary spanning a chunk seam must still be found")
        .expect("the response must parse");

    assert!(response.result.unwrap()["tools"].is_array());
}

/// The stated reason scanning happens on bytes rather than text. Decoding each
/// chunk independently substitutes U+FFFD for the split character — which is
/// still valid JSON string content, so the frame parses and the corruption is
/// returned as a success. Only an exact string comparison catches it.
#[tokio::test]
async fn multibyte_character_split_across_chunks_is_not_corrupted() {
    let request = JsonRpcRequest::new("tools/call", None);
    let text = "café 日本語 🎉";
    let frames = format!(
        "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"text\":\"{}\"}}}}\n\n",
        request.id, text
    );
    let at = frames.find('日').unwrap() + 1;
    assert!(
        !frames.is_char_boundary(at),
        "split inside a multi-byte char"
    );
    let addr = serve_pieces(split_at(frames.as_bytes(), at), false).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("must not hang")
        .expect("the response must parse");

    assert_eq!(
        response.result.unwrap()["text"].as_str().unwrap(),
        text,
        "a character split across chunks must survive intact"
    );
}

/// CRLF framing split mid-sequence: the `\r` ends one chunk and the `\n` begins
/// the next. Normalizing CR to LF has to carry that state across the seam, or
/// the boundary is mangled into `\n\n\n` and the event mis-split.
#[tokio::test]
async fn crlf_boundary_split_mid_sequence_is_normalized() {
    let request = JsonRpcRequest::new("tools/list", None);
    let frames = format!(
        "event: message\r\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"ok\":true}}}}\r\n\r\n",
        request.id
    );
    // Split between the final \r and its \n.
    let at = frames.len() - 1;
    assert_eq!(&frames.as_bytes()[at - 1..at + 1], b"\r\n");
    let addr = serve_pieces(split_at(frames.as_bytes(), at), false).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("CRLF split across a seam must still frame")
        .expect("the response must parse");

    assert_eq!(response.result.unwrap()["ok"], true);
}

/// SSE permits a bare CR as a line terminator. Deleting CR outright — rather
/// than normalizing it — collapses such a stream into one unsplittable line and
/// the response is never found.
#[tokio::test]
async fn bare_cr_framing_is_supported() {
    let request = JsonRpcRequest::new("tools/list", None);
    let frames = format!(
        "event: message\rdata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"ok\":true}}}}\r\r",
        request.id
    );
    let addr = serve_pieces(vec![frames.into_bytes()], true).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("must not hang")
        .expect("bare-CR framing must parse");

    assert_eq!(response.result.unwrap()["ok"], true);
}

/// Real servers close without a trailing blank line. The recovery path for that
/// runs only at EOF, so nothing else exercises it. The leading notification
/// matters: it advances `scanned`, where an off-by-one in the tail slice shows.
#[tokio::test]
async fn unterminated_final_event_after_a_notification_is_parsed() {
    let request = JsonRpcRequest::new("tools/call", None);
    let frames = format!(
        concat!(
            "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}}\n\n",
            "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"ok\":true}}}}\n"
        ),
        request.id
    );
    let addr = serve_pieces(vec![frames.into_bytes()], true).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("must not hang")
        .expect("an unterminated final event must be parsed at EOF");

    assert_eq!(response.result.unwrap()["ok"], true);
}

/// A connection that dies *after* delivering the answer is a success. This is a
/// load-bearing property of the early return: any refactor that drains before
/// returning turns an intermittent server-side disconnect into a tool failure.
#[tokio::test]
async fn truncation_after_the_answer_still_succeeds() {
    let request = JsonRpcRequest::new("tools/list", None);
    let frames = format!(
        "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"ok\":true}}}}\n\n",
        request.id
    );
    // A bogus chunk header after the answer: the body dies mid-stream.
    let pieces = vec![frames.into_bytes(), b"garbage-not-a-chunk".to_vec()];
    let addr = serve_pieces(pieces, false).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(10), transport.send(request))
        .await
        .expect("must not hang")
        .expect("a body that dies after the answer must not fail the call");

    assert_eq!(response.result.unwrap()["ok"], true);
}

/// One byte per chunk splits every boundary and every multi-byte character at
/// once — a backstop for seam handling generally. Less diagnostic than the
/// targeted tests above when it fails, so it complements them rather than
/// replacing them.
#[tokio::test]
async fn byte_at_a_time_delivery_assembles_correctly() {
    let request = JsonRpcRequest::new("tools/call", None);
    let text = "日本語";
    let frames = format!(
        "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"text\":\"{}\"}}}}\n\n",
        request.id, text
    );
    let pieces: Vec<Vec<u8>> = frames.as_bytes().iter().map(|b| vec![*b]).collect();
    let addr = serve_pieces(pieces, false).await;

    let transport = HttpTransport::new(&addr).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(20), transport.send(request))
        .await
        .expect("must not hang")
        .expect("the response must parse");

    assert_eq!(response.result.unwrap()["text"].as_str().unwrap(), text);
}
