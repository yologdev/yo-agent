//! Serde round-trip tests for core types.

use yoagent::*;

fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    value: &T,
) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(*value, back);
}

// ---------------------------------------------------------------------------
// Message variants
// ---------------------------------------------------------------------------

#[test]
fn test_message_user_roundtrip() {
    let msg = Message::User {
        content: vec![Content::Text {
            text: "Hello".into(),
        }],
        timestamp: 123456,
    };
    roundtrip(&msg);
}

#[test]
fn test_message_assistant_roundtrip() {
    let msg = Message::assistant(
        vec![
            Content::Text {
                text: "Hi there".into(),
            },
            Content::tool_call("tc-1", "read_file", serde_json::json!({"path": "foo.rs"})),
        ],
        StopReason::ToolUse,
        "claude-sonnet",
        "anthropic",
        Usage {
            input: 100,
            output: 50,
            cache_read: 10,
            cache_write: 5,
            total_tokens: 165,
        },
    )
    .with_timestamp(789);
    roundtrip(&msg);
}

#[test]
fn test_message_tool_result_roundtrip() {
    let msg = Message::ToolResult {
        tool_call_id: "tc-1".into(),
        tool_name: "bash".into(),
        content: vec![Content::Text {
            text: "exit code 0".into(),
        }],
        is_error: false,
        timestamp: 999,
    };
    roundtrip(&msg);
}

// ---------------------------------------------------------------------------
// AgentMessage
// ---------------------------------------------------------------------------

#[test]
fn test_agent_message_roundtrip() {
    let am = AgentMessage::Llm(Message::user("test prompt"));
    roundtrip(&am);
}

#[test]
fn test_extension_message_roundtrip() {
    let ext = ExtensionMessage::new("status_update", serde_json::json!({"status": "running"}));
    roundtrip(&ext);

    let am = AgentMessage::Extension(ext);
    roundtrip(&am);
}

// ---------------------------------------------------------------------------
// Content variants
// ---------------------------------------------------------------------------

#[test]
fn test_content_variants_roundtrip() {
    roundtrip(&Content::Text {
        text: "hello".into(),
    });
    roundtrip(&Content::Image {
        data: "base64data".into(),
        mime_type: "image/png".into(),
    });
    roundtrip(&Content::thinking_signed("let me think...", "sig123"));
    roundtrip(&Content::thinking("unsigned thought"));
    roundtrip(&Content::tool_call(
        "tc-1",
        "bash",
        serde_json::json!({"command": "ls"}),
    ));
}

// ---------------------------------------------------------------------------
// Full conversation
// ---------------------------------------------------------------------------

#[test]
fn test_full_conversation_roundtrip() {
    let conversation: Vec<AgentMessage> = vec![
        AgentMessage::Llm(Message::user("Read the file")),
        AgentMessage::Llm(
            Message::assistant(
                vec![Content::tool_call(
                    "tc-1",
                    "read_file",
                    serde_json::json!({"path": "main.rs"}),
                )],
                StopReason::ToolUse,
                "mock",
                "mock",
                Usage::default(),
            )
            .with_timestamp(100),
        ),
        AgentMessage::Llm(Message::ToolResult {
            tool_call_id: "tc-1".into(),
            tool_name: "read_file".into(),
            content: vec![Content::Text {
                text: "fn main() {}".into(),
            }],
            is_error: false,
            timestamp: 200,
        }),
        AgentMessage::Llm(
            Message::assistant(
                vec![Content::Text {
                    text: "The file contains a main function.".into(),
                }],
                StopReason::Stop,
                "mock",
                "mock",
                Usage::default(),
            )
            .with_timestamp(300),
        ),
        AgentMessage::Extension(ExtensionMessage::new(
            "ui_event",
            serde_json::json!({"action": "scroll"}),
        )),
    ];

    let json = serde_json::to_string(&conversation).expect("serialize");
    let back: Vec<AgentMessage> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(conversation, back);
}

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[test]
fn test_execution_limits_roundtrip() {
    use yoagent::context::ExecutionLimits;
    let limits = ExecutionLimits {
        max_turns: 25,
        max_total_tokens: 500_000,
        max_duration: std::time::Duration::from_secs(300),
    };
    let json = serde_json::to_string(&limits).expect("serialize");
    let back: ExecutionLimits = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(limits.max_turns, back.max_turns);
    assert_eq!(limits.max_total_tokens, back.max_total_tokens);
    assert_eq!(limits.max_duration, back.max_duration);
}

#[test]
fn test_tool_execution_strategy_roundtrip() {
    roundtrip(&ToolExecutionStrategy::Sequential);
    roundtrip(&ToolExecutionStrategy::Parallel);
    roundtrip(&ToolExecutionStrategy::Batched { size: 4 });
}

#[test]
fn test_refusal_stop_reason_round_trip() {
    use yoagent::types::*;

    let message = Message::assistant(
        vec![],
        StopReason::Refusal,
        "claude-fable-5",
        "anthropic",
        Usage::default(),
    )
    .with_timestamp(1)
    .with_error_message("declined");

    let json = serde_json::to_string(&message).unwrap();
    let back: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(message, back);
    assert_eq!(StopReason::Refusal.to_string(), "refusal");
}

#[test]
fn test_constructors_pin_fields() {
    // The non_exhaustive variants make these the mandated construction path
    // for downstream crates — pin every field explicitly.
    let tc = Content::tool_call("id-1", "bash", serde_json::json!({"cmd": "ls"}));
    let Content::ToolCall {
        id,
        name,
        arguments,
        provider_metadata,
        ..
    } = &tc
    else {
        panic!("expected tool call");
    };
    assert_eq!(id, "id-1");
    assert_eq!(name, "bash");
    assert_eq!(arguments["cmd"], "ls");
    assert!(provider_metadata.is_none());

    let think = Content::thinking_signed("hmm", "sig");
    let Content::Thinking {
        thinking,
        signature,
        ..
    } = &think
    else {
        panic!("expected thinking");
    };
    assert_eq!(thinking, "hmm");
    assert_eq!(signature.as_deref(), Some("sig"));

    let msg = Message::assistant(
        vec![Content::Text { text: "hi".into() }],
        StopReason::Stop,
        "model-x",
        "provider-y",
        Usage::default(),
    )
    .with_timestamp(42)
    .with_error_message("oops");
    let Message::Assistant {
        model,
        provider,
        timestamp,
        error_message,
        stop_reason,
        ..
    } = &msg
    else {
        panic!("expected assistant");
    };
    assert_eq!(model, "model-x");
    assert_eq!(provider, "provider-y");
    assert_eq!(*timestamp, 42);
    assert_eq!(error_message.as_deref(), Some("oops"));
    assert_eq!(*stop_reason, StopReason::Stop);
}

#[test]
fn test_tool_call_with_metadata_roundtrip() {
    // Thought signatures must survive session persistence.
    roundtrip(&Content::tool_call_with_metadata(
        "tc-9",
        "get_weather",
        serde_json::json!({"city": "Paris"}),
        serde_json::json!({"thought_signature": "sig-xyz"}),
    ));
}

// ---------------------------------------------------------------------------
// AgentEvent wire format (public contract — see the doc comment on AgentEvent)
// ---------------------------------------------------------------------------

fn sample_assistant() -> AgentMessage {
    AgentMessage::Llm(Message::assistant(
        vec![Content::Text { text: "hi".into() }],
        StopReason::Stop,
        "claude-sonnet",
        "anthropic",
        Usage::default(),
    ))
}

fn sample_tool_result() -> ToolResult {
    ToolResult {
        content: vec![Content::Text {
            text: "exit code 0".into(),
        }],
        details: serde_json::json!({"exit_code": 0}),
    }
}

/// One value of every `AgentEvent` variant.
fn all_agent_events() -> Vec<AgentEvent> {
    vec![
        AgentEvent::AgentStart,
        AgentEvent::agent_end(vec![sample_assistant()], SessionStats::default()),
        AgentEvent::TurnStart,
        AgentEvent::TurnEnd {
            message: sample_assistant(),
            tool_results: vec![Message::ToolResult {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                content: vec![Content::Text { text: "ok".into() }],
                is_error: false,
                timestamp: 7,
            }],
        },
        AgentEvent::MessageStart {
            message: sample_assistant(),
        },
        AgentEvent::MessageUpdate {
            message: sample_assistant(),
            delta: StreamDelta::Text { delta: "hi".into() },
        },
        AgentEvent::MessageEnd {
            message: sample_assistant(),
        },
        AgentEvent::ToolExecutionStart {
            tool_call_id: "tc-1".into(),
            tool_name: "bash".into(),
            args: serde_json::json!({"command": "ls"}),
        },
        AgentEvent::ToolExecutionUpdate {
            tool_call_id: "tc-1".into(),
            tool_name: "bash".into(),
            partial_result: sample_tool_result(),
        },
        AgentEvent::ToolExecutionEnd {
            tool_call_id: "tc-1".into(),
            tool_name: "bash".into(),
            result: sample_tool_result(),
            is_error: false,
        },
        AgentEvent::ProgressMessage {
            tool_call_id: "tc-1".into(),
            tool_name: "bash".into(),
            text: "50% done".into(),
        },
        AgentEvent::InputRejected {
            reason: "injection detected".into(),
        },
        AgentEvent::ContextCompacted {
            method: CompactionMethod::Summarized,
            messages_before: 40,
            messages_after: 13,
            tokens_before: 96_500,
            tokens_after: 41_200,
            summary: Some(SummaryStats::new(
                28,
                Usage {
                    input: 54_000,
                    output: 1_100,
                    cache_read: 0,
                    cache_write: 0,
                    total_tokens: 55_100,
                },
                Some(0.0451),
            )),
        },
    ]
}

/// The deterministic half of the same variant. Sampled separately because it
/// is the *more common* path and pins two things the `Summarized` sample
/// cannot: the wire value of `CompactionMethod::Deterministic`, and the shape
/// of an absent `summary`.
fn deterministic_compaction_event() -> AgentEvent {
    AgentEvent::ContextCompacted {
        method: CompactionMethod::Deterministic,
        messages_before: 40,
        messages_after: 12,
        tokens_before: 96_500,
        tokens_after: 40_000,
        summary: None,
    }
}

#[test]
fn test_context_compacted_wire_shape_is_frozen() {
    let v = serde_json::to_value(&all_agent_events()[12]).expect("serialize");
    assert_eq!(v["type"], "contextCompacted");
    assert_eq!(v["method"], "summarized");
    // camelCase, including the acronym casing a TS client hardcodes.
    assert_eq!(v["messagesBefore"], 40);
    assert_eq!(v["tokensAfter"], 41_200);
    assert_eq!(v["summary"]["messagesSummarized"], 28);
    assert_eq!(v["summary"]["usage"]["totalTokens"], 55_100);
    assert_eq!(v["summary"]["costUsd"], 0.0451);

    let d = serde_json::to_value(deterministic_compaction_event()).expect("serialize");
    assert_eq!(d["method"], "deterministic");
    assert!(
        d["summary"].is_null(),
        "an absent summary serializes as null"
    );
    roundtrip(&deterministic_compaction_event());
}

#[test]
fn test_agent_event_every_variant_roundtrips() {
    for event in all_agent_events() {
        roundtrip(&event);
    }
}

fn all_stream_deltas() -> Vec<StreamDelta> {
    vec![
        StreamDelta::Text { delta: "a".into() },
        StreamDelta::Thinking { delta: "b".into() },
        StreamDelta::ToolCallDelta { delta: "c".into() },
    ]
}

#[test]
fn test_stream_delta_every_variant_roundtrips() {
    for delta in all_stream_deltas() {
        roundtrip(&delta);
    }
}

/// The frozen `"type"` tag for every `AgentEvent` variant. Exhaustive match
/// The compile-time half of this freeze moved into `src/types.rs`
/// (`wire_tag_freeze`) when `AgentEvent` and `StreamDelta` became
/// `#[non_exhaustive]` — an integration test can no longer match them
/// exhaustively, so the "adding a variant fails to compile" guarantee only
/// holds inside the defining crate. What remains here is the round-trip and
/// payload-shape coverage, plus [`EVENT_VARIANT_COUNT`], which still fails
/// until a new variant gets a sample in `all_agent_events`.
/// A tag change is a breaking change for wire clients — do not edit casually.
fn expected_event_tag(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agentStart",
        AgentEvent::AgentEnd { .. } => "agentEnd",
        AgentEvent::TurnStart => "turnStart",
        AgentEvent::TurnEnd { .. } => "turnEnd",
        AgentEvent::MessageStart { .. } => "messageStart",
        AgentEvent::MessageUpdate { .. } => "messageUpdate",
        AgentEvent::MessageEnd { .. } => "messageEnd",
        AgentEvent::ToolExecutionStart { .. } => "toolExecutionStart",
        AgentEvent::ToolExecutionUpdate { .. } => "toolExecutionUpdate",
        AgentEvent::ToolExecutionEnd { .. } => "toolExecutionEnd",
        AgentEvent::ProgressMessage { .. } => "progressMessage",
        AgentEvent::InputRejected { .. } => "inputRejected",
        AgentEvent::ContextCompacted { .. } => "contextCompacted",
        _ => "unknown",
    }
}

/// Same exhaustive-match freeze for `StreamDelta` tags.
fn expected_delta_tag(delta: &StreamDelta) -> &'static str {
    match delta {
        StreamDelta::Text { .. } => "text",
        StreamDelta::Thinking { .. } => "thinking",
        StreamDelta::ToolCallDelta { .. } => "toolCallDelta",
        _ => "unknown",
    }
}

/// Number of arms in `expected_event_tag` — bump together with the match.
const EVENT_VARIANT_COUNT: usize = 13;

#[test]
fn test_agent_event_type_tags_are_frozen() {
    let events = all_agent_events();
    assert_eq!(
        events.len(),
        EVENT_VARIANT_COUNT,
        "a variant was added to expected_event_tag without a sample in all_agent_events — \
         the new variant's tag and round-trip are untested until one is added"
    );
    for event in events {
        let v: serde_json::Value = serde_json::to_value(&event).expect("serialize");
        assert_eq!(
            v["type"],
            expected_event_tag(&event),
            "tag drifted for {event:?}"
        );
    }
}

#[test]
fn test_stream_delta_type_tags_are_frozen() {
    for delta in all_stream_deltas() {
        let v: serde_json::Value = serde_json::to_value(&delta).expect("serialize");
        assert_eq!(
            v["type"],
            expected_delta_tag(&delta),
            "tag drifted for {delta:?}"
        );
    }
}

/// Freezes the FULL nested payload shape — most of the wire bytes in a real
/// stream are the `message` payload, so the contract extends to `Message`,
/// `Content`, and `Usage` serialization. This is the one test comparing
/// against a complete JSON literal: any serde-attribute change on those types
/// shows up here as a client-visible wire break.
#[test]
fn test_message_end_full_payload_shape_is_frozen() {
    let event = AgentEvent::MessageEnd {
        message: AgentMessage::Llm(
            Message::assistant(
                vec![
                    Content::Text { text: "hi".into() },
                    Content::Image {
                        data: "aGk=".into(),
                        mime_type: "image/png".into(),
                    },
                    Content::thinking_signed("hmm", "sig-1"),
                    Content::tool_call_with_metadata(
                        "tc-1",
                        "bash",
                        serde_json::json!({"command": "ls"}),
                        serde_json::json!({"thought_signature": "sig-2"}),
                    ),
                ],
                StopReason::ToolUse,
                "claude-sonnet",
                "anthropic",
                Usage {
                    input: 100,
                    output: 50,
                    cache_read: 10,
                    cache_write: 5,
                    total_tokens: 165,
                },
            )
            .with_timestamp(1234)
            .with_error_message("boom"),
        ),
    };
    let expected = serde_json::json!({
        "type": "messageEnd",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
                {"type": "thinking", "thinking": "hmm", "signature": "sig-1"},
                {"type": "toolCall", "id": "tc-1", "name": "bash",
                 "arguments": {"command": "ls"},
                 "providerMetadata": {"thought_signature": "sig-2"}},
            ],
            "stopReason": "toolUse",
            "model": "claude-sonnet",
            "provider": "anthropic",
            "usage": {
                "input": 100,
                "output": 50,
                "cacheRead": 10,
                "cacheWrite": 5,
                "totalTokens": 165,
            },
            "timestamp": 1234,
            "errorMessage": "boom",
        },
    });
    let actual = serde_json::to_value(&event).expect("serialize");
    assert_eq!(actual, expected, "nested payload wire shape drifted");
}

/// Session files and `save_messages` blobs written by yoagent < 0.13 used
/// snake_case for `cache_read`/`cache_write`/`total_tokens`/`error_message`/
/// `provider_metadata`. The `alias` attributes must keep them loadable.
#[test]
fn test_legacy_snake_case_payload_still_deserializes() {
    let legacy = r#"{
        "role": "assistant",
        "content": [
            {"type": "toolCall", "id": "tc-1", "name": "bash",
             "arguments": {}, "provider_metadata": {"sig": "x"}}
        ],
        "stopReason": "stop",
        "model": "m",
        "provider": "p",
        "usage": {"input": 1, "output": 2, "cache_read": 3,
                  "cache_write": 4, "total_tokens": 10},
        "timestamp": 1,
        "error_message": "old"
    }"#;
    let msg: Message = serde_json::from_str(legacy).expect("legacy payload must load");
    let Message::Assistant {
        usage,
        error_message,
        content,
        ..
    } = &msg
    else {
        panic!("expected assistant");
    };
    assert_eq!(usage.cache_read, 3);
    assert_eq!(usage.cache_write, 4);
    assert_eq!(usage.total_tokens, 10);
    assert_eq!(error_message.as_deref(), Some("old"));
    let Content::ToolCall {
        provider_metadata, ..
    } = &content[0]
    else {
        panic!("expected toolCall");
    };
    assert_eq!(provider_metadata.as_ref().unwrap()["sig"], "x");
}

/// Forward compatibility for wire clients: unknown fields inside a known
/// event are ignored (additive evolution), while an unknown or wrong-cased
/// `"type"` tag is a clean `Err`, never a panic.
#[test]
fn test_agent_event_forward_compat_deserialization() {
    // Unknown extra field → ignored.
    let line = r#"{"type":"inputRejected","reason":"x","newField":1}"#;
    let event: AgentEvent = serde_json::from_str(line).expect("unknown fields are ignored");
    assert!(matches!(event, AgentEvent::InputRejected { .. }));

    // Unknown tag → Err.
    assert!(serde_json::from_str::<AgentEvent>(r#"{"type":"compactionStart"}"#).is_err());
    // Wrong casing → Err (tags are case-sensitive).
    assert!(
        serde_json::from_str::<AgentEvent>(r#"{"type":"InputRejected","reason":"x"}"#).is_err()
    );
}

/// Shape snapshot: camelCase field names on the wire (`rename_all_fields`).
#[test]
fn test_agent_event_fields_are_camel_case() {
    let end = AgentEvent::ToolExecutionEnd {
        tool_call_id: "tc-1".into(),
        tool_name: "bash".into(),
        result: sample_tool_result(),
        is_error: true,
    };
    let v = serde_json::to_value(&end).expect("serialize");
    assert_eq!(v["toolCallId"], "tc-1");
    assert_eq!(v["toolName"], "bash");
    assert_eq!(v["isError"], true);
    assert!(
        v.get("tool_call_id").is_none(),
        "snake_case leaked onto the wire"
    );

    let update = AgentEvent::MessageUpdate {
        message: sample_assistant(),
        delta: StreamDelta::Text { delta: "hi".into() },
    };
    let v = serde_json::to_value(&update).expect("serialize");
    assert_eq!(v["type"], "messageUpdate");
    assert_eq!(v["delta"]["type"], "text");
    assert_eq!(v["delta"]["delta"], "hi");
    assert!(v.get("message").is_some());

    let rejected = AgentEvent::InputRejected {
        reason: "nope".into(),
    };
    let v = serde_json::to_value(&rejected).expect("serialize");
    assert_eq!(v["reason"], "nope");
}

/// Unit variants carry only the tag: `{"type":"agentStart"}`.
#[test]
fn test_agent_event_unit_variant_shape() {
    let json = serde_json::to_string(&AgentEvent::AgentStart).expect("serialize");
    assert_eq!(json, r#"{"type":"agentStart"}"#);
    let json = serde_json::to_string(&AgentEvent::TurnStart).expect("serialize");
    assert_eq!(json, r#"{"type":"turnStart"}"#);
}

/// A wire client's inbound path: parse an event from a raw JSON line.
#[test]
fn test_agent_event_deserializes_from_raw_json() {
    let line = r#"{"type":"toolExecutionStart","toolCallId":"tc-9","toolName":"read","args":{"path":"a.rs"}}"#;
    let event: AgentEvent = serde_json::from_str(line).expect("deserialize");
    let AgentEvent::ToolExecutionStart {
        tool_call_id,
        tool_name,
        args,
    } = event
    else {
        panic!("expected ToolExecutionStart");
    };
    assert_eq!(tool_call_id, "tc-9");
    assert_eq!(tool_name, "read");
    assert_eq!(args["path"], "a.rs");
}

// ---------------------------------------------------------------------------
// Secrets must not reach Debug output — it lands in logs and panic messages.
// ---------------------------------------------------------------------------

#[test]
fn stream_config_debug_redacts_the_api_key() {
    let mut config = yoagent::provider::StreamConfig::new("m", "sk-super-secret-key");
    config.system_prompt = "be brief".into();
    let rendered = format!("{:?}", config);

    assert!(
        !rendered.contains("sk-super-secret-key"),
        "api_key leaked into Debug: {rendered}"
    );
    assert!(
        rendered.contains("[redacted]"),
        "no redaction marker: {rendered}"
    );
    // Still useful for debugging.
    assert!(rendered.contains("be brief"));
}

#[test]
fn stream_config_debug_distinguishes_an_empty_key() {
    let config = yoagent::provider::StreamConfig::new("m", "");
    let rendered = format!("{:?}", config);
    assert!(
        !rendered.contains("[redacted]"),
        "an unset key should not look like a set one: {rendered}"
    );
}

#[test]
fn model_config_debug_redacts_header_values_but_keeps_names() {
    let mut config = yoagent::provider::ModelConfig::anthropic("claude-sonnet-5", "Sonnet");
    config
        .headers
        .insert("Authorization".into(), "Bearer sk-leaked-token".into());
    let rendered = format!("{:?}", config);

    assert!(
        !rendered.contains("sk-leaked-token"),
        "header credential leaked into Debug: {rendered}"
    );
    assert!(
        rendered.contains("Authorization"),
        "header names should survive for debugging: {rendered}"
    );
}

#[test]
fn model_config_serialization_stays_lossless() {
    // Debug redacts; Serialize must not, or saved configs lose their headers.
    let mut config = yoagent::provider::ModelConfig::anthropic("claude-sonnet-5", "Sonnet");
    config.headers.insert("x-api-key".into(), "keep-me".into());

    let json = serde_json::to_string(&config).unwrap();
    let back: yoagent::provider::ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        back.headers.get("x-api-key").map(String::as_str),
        Some("keep-me")
    );
}
