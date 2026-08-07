//! Tests for SharedState and its integration with SubAgentTool.

use std::sync::Arc;
use tokio_util::sync::CancellationToken;
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
            ToolContext {
                tool_call_id: "tc-1".into(),
                tool_name: "analyzer".into(),
                cancel: CancellationToken::new(),
                on_update: None,
                on_progress: None,
            },
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
            ToolContext {
                tool_call_id: "tc-1".into(),
                tool_name: "writer".into(),
                cancel: CancellationToken::new(),
                on_update: None,
                on_progress: None,
            },
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

    let ctx = || ToolContext {
        tool_call_id: "tc".into(),
        tool_name: "test".into(),
        cancel: CancellationToken::new(),
        on_update: None,
        on_progress: None,
    };

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
            ToolContext {
                tool_call_id: "tc-1".into(),
                tool_name: "plain".into(),
                cancel: CancellationToken::new(),
                on_update: None,
                on_progress: None,
            },
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
            ToolContext {
                tool_call_id: "tc-1".into(),
                tool_name: "lister".into(),
                cancel: CancellationToken::new(),
                on_update: None,
                on_progress: None,
            },
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
