//! Streaming tests for `OpenAiCompatProvider` against a local mock server.
//!
//! Covers the DONE-less close behavior (issue #76): some providers (MiniMax
//! confirmed in the field) close the SSE connection without the
//! OpenAI-standard `data: [DONE]` terminator, which surfaces as
//! `reqwest_eventsource::Error::StreamEnded`. After a `finish_reason` that
//! close is a completed response and must finish cleanly; before any
//! `finish_reason` it is genuine truncation and must stay an error.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use yoagent::provider::{ModelConfig, OpenAiCompatProvider, StreamConfig, StreamProvider};
use yoagent::types::*;

fn chunk(json: &str) -> String {
    format!("data: {json}\n\n")
}

fn stream_config(base_url: &str) -> StreamConfig {
    let mut mc = ModelConfig::minimax("MiniMax-M2.7", "MiniMax M2.7");
    mc.base_url = base_url.to_string();
    let mut config = StreamConfig::new("MiniMax-M2.7", "test-key");
    config.system_prompt = "test".into();
    config.messages = vec![Message::user("hi")];
    config.max_tokens = Some(256);
    config.model_config = Some(mc);
    config
}

async fn run_stream(config: StreamConfig) -> Result<Message, yoagent::provider::ProviderError> {
    let (tx, _rx) = mpsc::unbounded_channel();
    OpenAiCompatProvider
        .stream(config, tx, CancellationToken::new())
        .await
}

/// DONE-less close AFTER finish_reason (MiniMax's normal ending): the
/// response is complete — clean finish with accumulated content, no error.
#[tokio::test]
async fn test_stream_ended_after_finish_reason_is_clean_eof() {
    let server = MockServer::start().await;
    let body = [
        chunk(r#"{"choices":[{"delta":{"content":"Hello"},"index":0}]}"#),
        chunk(r#"{"choices":[{"delta":{"content":" world"},"index":0}]}"#),
        chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#),
        // No `data: [DONE]` — the body just ends.
    ]
    .concat();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let msg = run_stream(stream_config(&server.uri()))
        .await
        .expect("DONE-less close after finish_reason must not be an error");

    let Message::Assistant {
        content,
        stop_reason,
        ..
    } = &msg
    else {
        panic!("expected assistant message");
    };
    assert_eq!(*stop_reason, StopReason::Stop);
    assert!(
        matches!(&content[0], Content::Text { text } if text == "Hello world"),
        "accumulated content must survive the DONE-less close: {content:?}"
    );
}

/// DONE-less close BEFORE any finish_reason: genuine mid-stream truncation —
/// must remain an error (retry semantics stay honest for real drops).
#[tokio::test]
async fn test_stream_ended_without_finish_reason_is_error() {
    let server = MockServer::start().await;
    let body = chunk(r#"{"choices":[{"delta":{"content":"Hel"},"index":0}]}"#);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let result = run_stream(stream_config(&server.uri())).await;
    assert!(
        result.is_err(),
        "truncation before finish_reason must stay an error, got {result:?}"
    );
}

/// DONE-less close where a usage chunk arrives AFTER finish_reason (the
/// OpenAI `stream_options.include_usage` shape): usage must be captured.
#[tokio::test]
async fn test_usage_chunk_after_finish_reason_survives_doneless_close() {
    let server = MockServer::start().await;
    let body = [
        chunk(r#"{"choices":[{"delta":{"content":"Hi"},"index":0}]}"#),
        chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#),
        chunk(r#"{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#),
    ]
    .concat();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let msg = run_stream(stream_config(&server.uri()))
        .await
        .expect("clean");
    let Message::Assistant { usage, .. } = &msg else {
        panic!("expected assistant");
    };
    assert_eq!(
        usage.input, 7,
        "usage chunk after finish_reason must be captured"
    );
    assert_eq!(usage.output, 3);
}

/// A tool call split across chunks reassembles into the arguments the model
/// sent.
///
/// This module backs 15+ providers — OpenAI, Groq, Together, DeepSeek,
/// Fireworks, Mistral, xAI — and had three tests, none of which touched tool
/// calls at all. Argument streaming is the fiddliest part of the format and the
/// part with the widest blast radius.
#[tokio::test]
async fn tool_call_arguments_reassemble_across_chunks() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}{}",
        chunk(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":"}}]}}]}"#
        ),
        chunk(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]}}]}"#
        ),
        chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri()))
        .await
        .expect("stream should complete");
    let Message::Assistant { content, .. } = &message else {
        panic!("expected assistant message");
    };
    let args = content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall {
                name, arguments, ..
            } if name == "search" => Some(arguments),
            _ => None,
        })
        .expect("the tool call must reach the caller");
    assert_eq!(
        *args,
        serde_json::json!({"q": "rust"}),
        "arguments split across chunks must reassemble, not truncate at the first"
    );
}

/// A tool with no parameters is callable here too.
///
/// The Anthropic sibling of this shipped a bug: an empty argument stream was
/// treated as malformed and failed the whole turn. This module reaches the same
/// outcome by a different route — `from_str("")` fails and it falls back to
/// `{}` — so the behaviour is correct but incidental. Pinned so it stays.
#[tokio::test]
async fn a_zero_argument_tool_call_is_usable() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}",
        chunk(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"list_files","arguments":""}}]}}]}"#
        ),
        chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri()))
        .await
        .expect("stream should complete");
    let Message::Assistant {
        content,
        stop_reason,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(
        *stop_reason,
        StopReason::ToolUse,
        "a no-argument tool call is a usable turn"
    );
    let args = content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall {
                name, arguments, ..
            } if name == "list_files" => Some(arguments),
            _ => None,
        })
        .expect("the tool call must reach the caller");
    assert_eq!(*args, serde_json::json!({}));
}

/// **Documents a provider divergence, deliberately.**
///
/// Anthropic fails the turn on unparseable tool arguments, and says why: "a
/// tool handed `{"__partial_json": ...}` runs on its defaults instead of what
/// the model asked for". This module does the opposite — it falls back to `{}`
/// and warns, so the tool *does* run on its defaults.
///
/// Pinned rather than fixed, because changing it is a behavioural decision
/// affecting 15+ providers, not a test fix. If the divergence is ever closed,
/// this test should fail and be updated deliberately.
#[tokio::test]
async fn truncated_tool_arguments_fall_back_to_empty_unlike_anthropic() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}",
        chunk(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":"}}]}}]}"#
        ),
        chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri()))
        .await
        .expect("stream should complete");
    let Message::Assistant { content, .. } = &message else {
        panic!("expected assistant message");
    };
    let args = content
        .iter()
        .find_map(|c| match c {
            Content::ToolCall {
                name, arguments, ..
            } if name == "search" => Some(arguments),
            _ => None,
        })
        .expect("the call still reaches the caller here");
    assert_eq!(
        *args,
        serde_json::json!({}),
        "truncated arguments currently degrade to an empty object — the tool runs on its \
         defaults. Anthropic fails the turn instead. If this assertion changes, the \
         divergence was closed on purpose"
    );
}
