//! Streaming tests for `AnthropicProvider` against a local mock server.
//!
//! These cover the response-parsing and auth behavior that unit tests on
//! `build_request_body` can't reach: stop-reason mapping from SSE events and
//! the request headers actually sent on the wire.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
use yoagent::provider::{
    AnthropicCompat, AnthropicProvider, ModelConfig, StreamConfig, StreamProvider,
};
use yoagent::types::*;

/// Matcher: the request must NOT carry the given header.
struct HeaderAbsent(&'static str);

impl wiremock::Match for HeaderAbsent {
    fn matches(&self, request: &Request) -> bool {
        !request.headers.contains_key(self.0)
    }
}

/// Canned SSE body for a stream that ends with the given stop_reason and no
/// content blocks (the shape of a pre-output refusal).
fn sse_empty_with_stop(stop_reason: &str) -> String {
    format!(
        "event: message_start\n\
         data: {{\"type\":\"message_start\",\"message\":{{\"usage\":{{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}}}}\n\n\
         event: message_delta\n\
         data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{stop_reason}\"}},\"usage\":{{\"output_tokens\":0}}}}\n\n\
         event: message_stop\n\
         data: {{\"type\":\"message_stop\"}}\n\n"
    )
}

fn stream_config(base_url: &str, anthropic: Option<AnthropicCompat>) -> StreamConfig {
    let mut mc = ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
    mc.base_url = base_url.to_string();
    mc.anthropic = anthropic;
    let mut config = StreamConfig::new("claude-sonnet-5", "test-key");
    config.system_prompt = "test".into();
    config.messages = vec![Message::user("hi")];
    config.max_tokens = Some(256);
    config.model_config = Some(mc);
    config
}

async fn run_stream(config: StreamConfig) -> Result<Message, yoagent::provider::ProviderError> {
    let (tx, _rx) = mpsc::unbounded_channel();
    AnthropicProvider
        .stream(config, tx, CancellationToken::new())
        .await
}

#[tokio::test]
async fn refusal_stop_reason_maps_to_refusal_with_error_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("refusal"), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");

    let Message::Assistant {
        stop_reason,
        error_message,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(*stop_reason, StopReason::Refusal);
    assert!(
        error_message.as_deref().unwrap_or("").contains("refusal"),
        "error_message should explain the refusal, got {error_message:?}"
    );
}

#[tokio::test]
async fn context_window_exceeded_maps_to_overflow_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse_empty_with_stop("model_context_window_exceeded"),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");

    assert!(
        message.is_context_overflow(),
        "in-stream overflow must trigger the documented recovery hook"
    );
}

#[tokio::test]
async fn bearer_auth_sends_authorization_and_no_x_api_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("authorization", "Bearer test-key"))
        .and(HeaderAbsent("x-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("end_turn"), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let config = stream_config(
        &server.uri(),
        Some(AnthropicCompat {
            adaptive_thinking: true,
            bearer_auth: true,
        }),
    );
    run_stream(config).await.expect("stream should succeed");
    // Mock expectation (`expect(1)`) verifies the headers on drop.
}

#[tokio::test]
async fn default_auth_sends_x_api_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("end_turn"), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");
}

#[tokio::test]
async fn user_authorization_header_suppresses_x_api_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("authorization", "Bearer custom-token"))
        .and(HeaderAbsent("x-api-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("end_turn"), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let mut config = stream_config(&server.uri(), None);
    if let Some(mc) = &mut config.model_config {
        mc.headers
            .insert("Authorization".into(), "Bearer custom-token".into());
    }
    run_stream(config).await.expect("stream should succeed");
}

#[tokio::test]
async fn rate_limit_carries_retry_after_from_header() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_string(
                    r#"{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}"#,
                ),
        )
        .mount(&server)
        .await;

    let err = run_stream(stream_config(&server.uri(), None))
        .await
        .expect_err("429 must surface as an error");

    match err {
        yoagent::provider::ProviderError::RateLimited { retry_after_ms } => {
            assert_eq!(retry_after_ms, Some(7000));
        }
        other => panic!("expected RateLimited, got: {:?}", other),
    }
}

/// Issue #81: the returned message must carry the `ModelConfig.provider`, not a
/// hardcoded "anthropic". Gateways that speak the Anthropic Messages protocol
/// (OpenCode Zen, Copilot) set their own provider name for cost and session
/// attribution — yoagent's own `ModelConfig::opencode_zen()` preset routes
/// Claude model ids over this provider, so the hardcoded value mis-attributed
/// a first-class preset.
#[tokio::test]
async fn provider_comes_from_model_config_not_hardcoded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("end_turn"), "text/event-stream"),
        )
        .mount(&server)
        .await;

    // Build from the real preset, not a hand-set field, so this also guards
    // ModelConfig::opencode_zen() continuing to route Claude ids here.
    let mut mc = ModelConfig::opencode_zen("claude-sonnet-5");
    mc.base_url = server.uri();
    assert_eq!(mc.provider, "opencode-zen", "preset sets the provider name");
    let mut config = stream_config(&server.uri(), None);
    config.model_config = Some(mc);

    let message = run_stream(config).await.expect("stream should succeed");

    let Message::Assistant { provider, .. } = &message else {
        panic!("expected assistant message");
    };
    assert_eq!(
        provider, "opencode-zen",
        "provider must be propagated from ModelConfig, not hardcoded"
    );
}

/// Issue #83: a terminator-less close BEFORE any `message_delta` is genuine
/// truncation. It must surface as a retryable `Network` error, not the
/// non-retryable `Other` it used to be — a proxy or load balancer closing
/// mid-response sends a FIN, which the eventsource reports as `StreamEnded`.
#[tokio::test]
async fn stream_ended_without_stop_reason_is_retryable_network_error() {
    let server = MockServer::start().await;
    // message_start only, then the body ends: no stop_reason, no message_stop.
    let truncated = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(truncated, "text/event-stream"))
        .mount(&server)
        .await;

    let err = run_stream(stream_config(&server.uri(), None))
        .await
        .expect_err("truncation before stop_reason must be an error");

    assert!(
        matches!(err, yoagent::provider::ProviderError::Network(_)),
        "expected retryable Network, got: {err:?}"
    );
    assert!(err.is_retryable(), "truncation must be retryable");
}

/// The other half of #83: a terminator-less close AFTER `message_delta` means
/// the response is already complete (stop_reason and usage arrived), so it is a
/// clean EOF — NOT a retry. Without this guard, making StreamEnded retryable
/// would re-bill a finished response, the bug #76 fixed for openai_compat.
#[tokio::test]
async fn stream_ended_after_stop_reason_is_clean_eof() {
    let server = MockServer::start().await;
    // Complete response, but the body ends without `message_stop`.
    let no_terminator = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(no_terminator, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("close after message_delta must not be an error");

    let Message::Assistant {
        stop_reason,
        content,
        usage,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(*stop_reason, StopReason::Stop);
    assert_eq!(usage.output, 5, "usage from message_delta must survive");
    let text: String = content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        text, "hello",
        "content must survive the terminator-less close"
    );
}

/// Review follow-up: the clean-EOF guard must be armed only by a `message_delta`
/// that actually carried a terminal `stop_reason`. An intermediate delta (usage
/// update, empty `delta`) parses fine, and arming on it turned a genuine
/// truncation into a silently-partial success.
#[tokio::test]
async fn intermediate_message_delta_does_not_arm_the_clean_eof_guard() {
    let server = MockServer::start().await;
    // A delta with NO stop_reason, then the body ends: still truncation.
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":3}}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let err = run_stream(stream_config(&server.uri(), None))
        .await
        .expect_err("a delta without stop_reason must not mark the response complete");
    assert!(
        err.is_retryable(),
        "expected retryable truncation, got: {err:?}"
    );
}

/// Review follow-up: `content_block_stop` is what parses a tool call's
/// accumulated `__partial_json` buffer into real arguments. If the stream ends
/// after `message_delta` but before that, returning the message would hand the
/// tool its own sentinel key as input — and the loop would execute it.
#[tokio::test]
async fn clean_eof_guard_rejects_unterminated_tool_call() {
    let server = MockServer::start().await;
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\": \\\"rm -rf /tm\"}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let err = run_stream(stream_config(&server.uri(), None))
        .await
        .expect_err("an unterminated tool_use block must not be returned as success");
    assert!(
        err.is_retryable(),
        "expected retryable truncation, got: {err:?}"
    );
}

/// Review follow-up: gateways relaying this protocol sometimes omit `usage` on
/// `message_delta`. That used to fail the whole parse, silently dropping the
/// stop_reason it carried and downgrading `tool_use` to `Stop`.
#[tokio::test]
async fn message_delta_without_usage_still_yields_its_stop_reason() {
    let server = MockServer::start().await;
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("a usage-less message_delta must still parse");
    let Message::Assistant { stop_reason, .. } = &message else {
        panic!("expected assistant message");
    };
    assert_eq!(
        *stop_reason,
        StopReason::Length,
        "stop_reason must survive a message_delta with no usage field"
    );
}

/// Issue #89: malformed tool-call JSON used to be replaced with an empty
/// object, so the tool ran with default arguments and neither the caller nor
/// the model learned the model's actual input had been dropped — silently wrong
/// action. It must fail the turn instead.
///
/// This is on the happy path for this crate's own configuration: we send the
/// `fine-grained-tool-streaming` beta, which Anthropic documents as able to
/// emit incomplete tool JSON when a response hits `max_tokens`.
#[tokio::test]
async fn malformed_tool_arguments_fail_the_turn_instead_of_defaulting() {
    let server = MockServer::start().await;
    // `input_json_delta` never completes into valid JSON, but the block and the
    // message are both properly terminated — so this is not truncation, it is a
    // well-formed stream carrying an unparseable tool input.
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{not valid json\"}}\n\n\
         event: content_block_stop\n\
         data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("the stream itself is well-formed");

    let Message::Assistant {
        stop_reason,
        error_message,
        content,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };

    // The agent loop returns on Error *before* extracting tool calls, so this
    // is what stops the tool from running with fabricated arguments.
    assert_eq!(
        *stop_reason,
        StopReason::Error,
        "a tool call we cannot parse must not be presented as a usable turn"
    );
    assert!(
        error_message.as_deref().unwrap_or("").contains("bash"),
        "the error must name the tool, got: {error_message:?}"
    );

    // No tool_use may survive: it would go back to the API with no matching
    // tool_result and be rejected on the next request.
    assert!(
        !content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. })),
        "the unusable tool call must not remain in the message"
    );
    // And the internal accumulator must never leak into the message.
    assert!(
        !format!("{content:?}").contains("__partial_json"),
        "the accumulator sentinel must not escape: {content:?}"
    );

    // Replaced, not removed. Removing would shift `content.len()` out of step
    // with the provider's block indices, so a later `content_block_start` pads
    // with duplicate placeholders; it would also erase the turn from the
    // transcript, since an assistant message with no blocks is dropped whole
    // from the next request.
    let Some(Content::Text { text }) = content.first() else {
        panic!("the dropped tool call must leave a text block behind: {content:?}");
    };
    assert!(
        text.contains("bash"),
        "the replacement must record which tool was dropped, got: {text}"
    );
    assert_eq!(content.len(), 1, "no other blocks expected: {content:?}");

    // The quoted input is what makes the error actionable in a log.
    assert!(
        error_message
            .as_deref()
            .unwrap_or("")
            .contains("{not valid json"),
        "the error must quote the unparseable input, got: {error_message:?}"
    );
}

/// Issue #89: `content_block_stop` used to default a missing index to 0, closing
/// a block the event was never about.
///
/// The damage is concrete: block 0 is still accumulating here, so closing it
/// early parses a half-written `{"cmd":` and — correctly, per the fix above —
/// fails the whole turn. Ignoring the index-less event lets block 0 finish and
/// the turn succeed.
#[tokio::test]
async fn content_block_stop_without_an_index_does_not_close_block_zero() {
    let server = MockServer::start().await;
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\"}}\n\n\
         event: content_block_stop\n\
         data: {\"type\":\"content_block_stop\"}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\n\
         event: content_block_stop\n\
         data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");

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
        "an index-less stop must not close block 0 mid-accumulation and fail the turn"
    );
    let Some(Content::ToolCall { arguments, .. }) = content
        .iter()
        .find(|c| matches!(c, Content::ToolCall { .. }))
    else {
        panic!("the tool call must survive: {content:?}");
    };
    assert_eq!(
        arguments["cmd"], "ls",
        "arguments must assemble fully: {arguments:?}"
    );
}

/// Issue #89 follow-up: `end_turn` is what an ordinary completion carries. It
/// must be matched explicitly — folding it into the unknown-reason arm makes the
/// "we hit a stop reason we don't handle" warning fire on every healthy turn,
/// which buries the signal it exists to give.
#[tokio::test]
async fn end_turn_and_stop_sequence_are_recognized_stop_reasons() {
    for reason in ["end_turn", "stop_sequence"] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(sse_empty_with_stop(reason), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let message = run_stream(stream_config(&server.uri(), None))
            .await
            .unwrap_or_else(|e| panic!("[{reason}] stream should succeed: {e}"));
        let Message::Assistant { stop_reason, .. } = &message else {
            panic!("[{reason}] expected assistant message");
        };
        assert_eq!(*stop_reason, StopReason::Stop, "[{reason}]");
    }
}

/// `pause_turn` means the model stopped mid-turn and expects the conversation
/// to be re-sent to continue — it is a shipped Anthropic stop reason, not a
/// hypothetical. Reporting it as a normal stop hands back a truncated answer as
/// though it were complete; this transport cannot resume, so it must say so.
#[tokio::test]
async fn pause_turn_is_reported_as_incomplete_rather_than_finished() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_empty_with_stop("pause_turn"), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");
    let Message::Assistant {
        stop_reason,
        error_message,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(
        *stop_reason,
        StopReason::Error,
        "a paused turn is not a finished one"
    );
    assert!(
        error_message
            .as_deref()
            .unwrap_or("")
            .contains("pause_turn"),
        "the error must name the cause, got: {error_message:?}"
    );
}

/// A stop reason we genuinely do not recognize still maps to `Stop` — the safe
/// default — and is logged. Uses a string Anthropic does not define, so this
/// keeps testing the fallback rather than a value that later gains meaning.
#[tokio::test]
async fn unrecognized_stop_reason_falls_back_to_a_normal_stop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            sse_empty_with_stop("reason_that_does_not_exist"),
            "text/event-stream",
        ))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("an unrecognized stop reason must not fail the stream");
    let Message::Assistant { stop_reason, .. } = &message else {
        panic!("expected assistant message");
    };
    assert_eq!(*stop_reason, StopReason::Stop);
}

/// Issue #89 follow-up: a `content_block_stop` that is unusable — not JSON, or
/// carrying no index — leaves the block holding the streaming accumulator. If
/// that reaches the caller, the loop executes the tool with
/// `{"__partial_json": ...}` as its input, and the tool falls back to its
/// defaults: `list_files` asked for `/etc` lists the process's cwd instead.
/// That is the wrong-arguments execution this whole fix exists to stop.
#[tokio::test]
async fn an_unfinalized_tool_call_never_reaches_the_caller() {
    for (label, stop_line) in [
        ("index-less", r#"data: {"type":"content_block_stop"}"#),
        ("non-JSON body", "data: not-json-at-all"),
    ] {
        let server = MockServer::start().await;
        let body = format!(
            "event: message_start\n\
             data: {{\"type\":\"message_start\",\"message\":{{\"usage\":{{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}}}}\n\n\
             event: content_block_start\n\
             data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{{}}}}}}\n\n\
             event: content_block_delta\n\
             data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"{{\\\"cmd\\\":\\\"ls\\\"}}\"}}}}\n\n\
             event: content_block_stop\n\
             {stop_line}\n\n\
             event: message_delta\n\
             data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"tool_use\"}},\"usage\":{{\"output_tokens\":9}}}}\n\n\
             event: message_stop\n\
             data: {{\"type\":\"message_stop\"}}\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let message = run_stream(stream_config(&server.uri(), None))
            .await
            .unwrap_or_else(|e| panic!("[{label}] stream should not error: {e}"));
        let Message::Assistant {
            content,
            stop_reason,
            ..
        } = &message
        else {
            panic!("[{label}] expected assistant message");
        };
        assert!(
            !format!("{content:?}").contains("__partial_json"),
            "[{label}] the accumulator escaped as tool arguments: {content:?}"
        );
        assert_ne!(
            *stop_reason,
            StopReason::ToolUse,
            "[{label}] an unfinalized tool call must not be presented as runnable"
        );
    }
}

/// Issue #89 follow-up: the malformed block is replaced precisely because a
/// `tool_use` with no matching `tool_result` is rejected on the next request.
/// The turn executes nothing, so a *healthy* sibling would never get a
/// `tool_result` either — leaving it would move the same 400 one turn later.
#[tokio::test]
async fn a_healthy_sibling_tool_call_does_not_dangle() {
    let server = MockServer::start().await;
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_0\",\"name\":\"read\",\"input\":{}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"p\\\":\\\"a\\\"}\"}}\n\n\
         event: content_block_stop\n\
         data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
         event: content_block_start\n\
         data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"bash\",\"input\":{}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{not valid\"}}\n\n\
         event: content_block_stop\n\
         data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");
    let Message::Assistant {
        content,
        stop_reason,
        error_message,
        ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(*stop_reason, StopReason::Error);
    assert!(error_message.as_deref().unwrap_or("").contains("bash"));
    assert!(
        !content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. })),
        "no tool_use may survive an errored turn unanswered: {content:?}"
    );
}

/// A trailing `message_delta` carrying neither a stop reason nor usage must not
/// downgrade what an earlier one established. Gateways relaying this protocol
/// emit such deltas.
#[tokio::test]
async fn a_trailing_empty_delta_preserves_stop_reason_and_usage() {
    let server = MockServer::start().await;
    let body = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"},\"usage\":{\"output_tokens\":42}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let message = run_stream(stream_config(&server.uri(), None))
        .await
        .expect("stream should succeed");
    let Message::Assistant {
        stop_reason, usage, ..
    } = &message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(
        *stop_reason,
        StopReason::Refusal,
        "a trailing delta must not downgrade a terminal stop reason"
    );
    assert_eq!(
        usage.output, 42,
        "a usage-less trailing delta must not zero the count"
    );
}
