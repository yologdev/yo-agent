//! Tests for SharedState and its integration with SubAgentTool.

use std::sync::Arc;
use yoagent::provider::mock::*;
use yoagent::provider::MockProvider;
use yoagent::provider::ModelConfig;
use yoagent::shared_state::SharedState;
use yoagent::sub_agent::SubAgentTool;
use yoagent::*;

// ---------------------------------------------------------------------------
// Integration: parent stores a value, sub-agent reads it via shared_state tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_reads_shared_state() {
    let state = SharedState::new();
    state
        .set("artifact", "LINE1: build failed\nLINE2: exit code 1".into())
        .await
        .unwrap();

    // Sub-agent mock: first call issues shared_state get, second returns text
    let sub_provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "get", "key": "artifact"}),
        }]),
        MockResponse::Text("The build failed with exit code 1".into()),
    ]));

    let sub_agent = SubAgentTool::from_provider("analyzer", sub_provider, ModelConfig::mock())
        .with_description("Analyzes artifacts")
        .with_system_prompt("Analyze the artifact.")
        .with_shared_state(state.clone());

    let result = sub_agent
        .execute(
            serde_json::json!({"task": "What happened in the build?"}),
            ToolContext::new("tc-1", "analyzer"),
        )
        .await
        .expect("sub-agent should succeed");

    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    assert!(text.contains("build failed"));
}

// ---------------------------------------------------------------------------
// Integration: sub-agent writes a value, parent reads it back
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_writes_shared_state() {
    let state = SharedState::new();

    // Sub-agent mock: sets a value then responds with text
    let sub_provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({
                "action": "set",
                "key": "summary",
                "value": "Root cause: OOM in test runner"
            }),
        }]),
        MockResponse::Text("Done, wrote summary.".into()),
    ]));

    let sub_agent = SubAgentTool::from_provider("writer", sub_provider, ModelConfig::mock())
        .with_description("Writes summaries")
        .with_system_prompt("Summarize findings.")
        .with_shared_state(state.clone());

    sub_agent
        .execute(
            serde_json::json!({"task": "Summarize"}),
            ToolContext::new("tc-1", "writer"),
        )
        .await
        .expect("sub-agent should succeed");

    // Parent reads back the value the sub-agent stored
    let summary = state.get("summary").await.expect("summary should exist");
    assert_eq!(summary, "Root cause: OOM in test runner");
}

// ---------------------------------------------------------------------------
// Integration: two parallel sub-agents share state
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_parallel_sub_agents_share_state() {
    let state = SharedState::new();
    state.set("input", "shared data".into()).await.unwrap();

    // Agent A reads then writes result_a
    let provider_a = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "get", "key": "input"}),
        }]),
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "set", "key": "result_a", "value": "from A"}),
        }]),
        MockResponse::Text("A done".into()),
    ]));

    // Agent B reads then writes result_b
    let provider_b = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "get", "key": "input"}),
        }]),
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "set", "key": "result_b", "value": "from B"}),
        }]),
        MockResponse::Text("B done".into()),
    ]));

    let agent_a = SubAgentTool::from_provider("agent_a", provider_a, ModelConfig::mock())
        .with_system_prompt("You are agent A.")
        .with_shared_state(state.clone());

    let agent_b = SubAgentTool::from_provider("agent_b", provider_b, ModelConfig::mock())
        .with_system_prompt("You are agent B.")
        .with_shared_state(state.clone());

    let ctx = || ToolContext::new("tc", "test");

    // Run in parallel
    let (ra, rb) = tokio::join!(
        agent_a.execute(serde_json::json!({"task": "process"}), ctx()),
        agent_b.execute(serde_json::json!({"task": "process"}), ctx()),
    );
    ra.unwrap();
    rb.unwrap();

    assert_eq!(state.get("result_a").await, Some("from A".into()));
    assert_eq!(state.get("result_b").await, Some("from B".into()));
    assert_eq!(state.get("input").await, Some("shared data".into()));
}

// ---------------------------------------------------------------------------
// SubAgentTool without shared_state works as before
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_without_shared_state_unchanged() {
    let sub_provider = Arc::new(MockProvider::text("hello"));

    let sub_agent = SubAgentTool::from_provider("plain", sub_provider, ModelConfig::mock())
        .with_system_prompt("You are plain.");
    // No .with_shared_state() — existing behavior

    let result = sub_agent
        .execute(
            serde_json::json!({"task": "say hi"}),
            ToolContext::new("tc-1", "plain"),
        )
        .await
        .expect("should work without shared state");

    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text"),
    };
    assert_eq!(text, "hello");
}

// ---------------------------------------------------------------------------
// System prompt includes shared state summary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_shared_state_summary_in_system_prompt() {
    let state = SharedState::new();
    state.set("log", "x".repeat(2048)).await.unwrap();

    // We can't inspect the system prompt directly from outside, but we can
    // verify the sub-agent gets the shared_state tool by having it call list
    let sub_provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "shared_state".into(),
            provider_metadata: None,
            arguments: serde_json::json!({"action": "list"}),
        }]),
        MockResponse::Text("Listed state".into()),
    ]));

    let sub_agent = SubAgentTool::from_provider("lister", sub_provider, ModelConfig::mock())
        .with_system_prompt("List state.")
        .with_shared_state(state);

    let result = sub_agent
        .execute(
            serde_json::json!({"task": "list"}),
            ToolContext::new("tc-1", "lister"),
        )
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text"),
    };
    assert_eq!(text, "Listed state");
}

// ---------------------------------------------------------------------------
// Scoped views — opt-in isolation between sub-agents.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scoped_views_cannot_see_or_touch_each_other() {
    let state = SharedState::new();
    let a = state.scoped("researcher");
    let b = state.scoped("writer");

    a.set("notes", "secret research".into()).await.unwrap();
    b.set("notes", "draft prose".into()).await.unwrap();

    // Same key name, independent values.
    assert_eq!(a.get("notes").await.as_deref(), Some("secret research"));
    assert_eq!(b.get("notes").await.as_deref(), Some("draft prose"));

    // Neither can enumerate the other.
    assert_eq!(a.keys().await, vec!["notes".to_string()]);
    assert_eq!(b.keys().await, vec!["notes".to_string()]);

    // A sibling's remove does not reach across.
    assert!(b.remove("notes").await);
    assert_eq!(a.get("notes").await.as_deref(), Some("secret research"));
}

#[tokio::test]
async fn scoped_summary_does_not_disclose_sibling_keys() {
    // The summary goes into the sub-agent's system prompt — the leak that
    // matters most.
    let state = SharedState::new();
    state
        .scoped("writer")
        .set("private_draft", "x".into())
        .await
        .unwrap();
    let researcher = state.scoped("researcher");
    researcher.set("sources", "y".into()).await.unwrap();

    let summary = researcher.summary().await;
    assert!(summary.contains("sources"), "own key missing: {summary}");
    assert!(
        !summary.contains("private_draft"),
        "sibling key leaked into the prompt: {summary}"
    );
}

#[tokio::test]
async fn a_scoped_view_cannot_escape_its_scope() {
    let state = SharedState::new();
    state.set("root_secret", "topsecret".into()).await.unwrap();
    let sub = state.scoped("sub");

    // Neither a plain key nor one crafted with a separator reaches the root.
    assert!(sub.get("root_secret").await.is_none());
    assert!(sub.get("\u{1f}root_secret").await.is_none());
    assert!(sub.get("../root_secret").await.is_none());

    // Nesting narrows, never widens.
    let deeper = sub.scoped("deeper");
    deeper.set("k", "v".into()).await.unwrap();
    assert!(sub.get("k").await.is_none());
    assert_eq!(deeper.get("k").await.as_deref(), Some("v"));
}

#[tokio::test]
async fn the_parent_still_sees_everything_scopes_write() {
    // Collecting sub-agent results is the reason scoping is a view, not a
    // separate store.
    let state = SharedState::new();
    state
        .scoped("researcher")
        .set("out", "findings".into())
        .await
        .unwrap();

    let all = state.keys().await;
    assert_eq!(all.len(), 1);
    assert!(all[0].contains("researcher"), "got {all:?}");
    assert_eq!(
        state.scoped("researcher").get("out").await.as_deref(),
        Some("findings")
    );
}

#[tokio::test]
async fn unscoped_behaviour_is_unchanged() {
    let state = SharedState::new();
    state.set("k", "v".into()).await.unwrap();
    assert_eq!(state.get("k").await.as_deref(), Some("v"));
    assert_eq!(state.keys().await, vec!["k".to_string()]);
    assert!(state.scope().is_none());
    assert!(state.summary().await.contains("k"));
}

// ---------------------------------------------------------------------------
// Truncation → retrieval (issue #125)
// ---------------------------------------------------------------------------

use yoagent::context::{tool_output_key, truncate_tool_output, ContextConfig};

fn big_tool_result(id: &str, lines: usize) -> AgentMessage {
    AgentMessage::Llm(Message::ToolResult {
        tool_call_id: id.into(),
        tool_name: "bash".into(),
        content: vec![Content::Text {
            text: (0..lines)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        }],
        is_error: false,
        timestamp: 0,
    })
}

fn text_of(msg: &AgentMessage) -> String {
    match msg {
        AgentMessage::Llm(Message::ToolResult { content, .. }) => content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The key must come from the tool call id, not a counter: a counter needs
/// mutable state and yields different keys on replay, and the key ends up
/// inside marker text that later compaction passes re-read.
#[test]
fn stash_key_is_derived_from_the_tool_call_id() {
    assert_eq!(tool_output_key("abc123"), "tool-out-abc123");
    assert_eq!(tool_output_key("abc123"), tool_output_key("abc123"));
    assert_ne!(tool_output_key("abc123"), tool_output_key("def456"));
}

/// The load-bearing property. Level 1 runs on settled history every turn, so a
/// marker whose bytes move on a later pass breaks the provider's prefix cache —
/// the exact thing Level-1 idempotence exists to protect.
#[test]
fn an_annotated_marker_survives_re_truncation_byte_for_byte() {
    let config = ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    };

    let truncated = truncate_tool_output(big_tool_result("tc-1", 500), &config);
    // Simulate what the loop does after stashing.
    let annotated = yoagent::context::annotate_marker_with_key(truncated, "tool-out-tc-1");
    let before = text_of(&annotated);

    // Three further compaction passes must not move a byte.
    let mut msg = annotated;
    for pass in 1..=3 {
        msg = truncate_tool_output(msg, &config);
        assert_eq!(
            text_of(&msg),
            before,
            "pass {pass} rewrote the marker; the prefix cache would break"
        );
    }
    assert!(before.contains("tool-out-tc-1"), "marker must name the key");
}

#[tokio::test]
async fn file_backend_evicts_oldest_to_stay_under_its_cap() {
    let dir = tempfile::tempdir().unwrap();
    // Cap at 300 bytes; each value is 100.
    let state = SharedState::with_backend(yoagent::shared_state::FileBackend::with_max_bytes(
        dir.path(),
        300,
    ));

    for i in 0..6 {
        state.set(&format!("k{i}"), "x".repeat(100)).await.unwrap();
    }

    let keys = state.keys().await;
    assert!(
        keys.len() <= 3,
        "cap must bound the directory, got {} keys: {keys:?}",
        keys.len()
    );
    assert!(
        keys.contains(&"k5".to_string()),
        "the most recent write must never be the one evicted, got {keys:?}"
    );
    assert!(
        !keys.contains(&"k0".to_string()),
        "the oldest must go first, got {keys:?}"
    );
}
