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

/// The key must survive replay and must not collide. The tool call id alone
/// does not: `google.rs`/`google_vertex.rs` synthesize ids as a per-response
/// index that restarts each turn, so turn 1 and turn 5 both produce
/// `google-fc-0` — and turn 1's frozen marker would then resolve to turn 5's
/// content. Hashing the output disambiguates.
#[test]
fn stash_key_is_stable_and_collision_resistant() {
    assert_eq!(
        tool_output_key("tc-1", "output"),
        tool_output_key("tc-1", "output"),
        "same call, same content — replay must give the same key"
    );
    assert_ne!(
        tool_output_key("google-fc-0", "turn one output"),
        tool_output_key("google-fc-0", "turn five output"),
        "a reused provider id must not alias two different outputs"
    );
    assert_ne!(
        tool_output_key("tc-1", "same"),
        tool_output_key("tc-2", "same"),
        "different calls stay distinct"
    );
}

/// The load-bearing property. Level 1 runs on settled history every turn, so a
/// marker whose bytes move on a later pass breaks the provider's prefix cache —
/// the exact thing Level-1 idempotence exists to protect.
#[test]
fn a_keyed_marker_survives_re_truncation_byte_for_byte() {
    let config = ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    };

    let keyed = yoagent::context::truncate_tool_output_keyed(
        big_tool_result("tc-1", 500),
        &config,
        Some("tool-out-tc-1-deadbeef"),
    )
    .0;
    let before = text_of(&keyed);
    assert!(
        before.contains("tool-out-tc-1-deadbeef"),
        "marker must name the key"
    );

    let mut msg = keyed;
    for pass in 1..=3 {
        msg = truncate_tool_output(msg, &config);
        assert_eq!(
            text_of(&msg),
            before,
            "Level-1 pass {pass} rewrote the marker; the prefix cache would break"
        );
    }
}

/// A budget too small for a marker truncates but emits nothing to name a key.
/// Stashing then would leave an entry no marker points at — unreachable
/// forever, and consuming cap quota that evicts reachable entries.
#[test]
fn a_budget_too_small_for_a_marker_reports_no_marker() {
    for max_lines in 1..=4 {
        let config = ContextConfig {
            tool_output_max_lines: max_lines,
            ..Default::default()
        };
        let (msg, emitted) = yoagent::context::truncate_tool_output_keyed(
            big_tool_result("tc-1", 100),
            &config,
            Some("tool-out-tc-1-deadbeef"),
        );
        assert!(
            !emitted,
            "max_lines={max_lines} has no room for a marker, so none was emitted"
        );
        assert!(
            !text_of(&msg).contains("tool-out-tc-1"),
            "max_lines={max_lines} must not name a key it did not emit"
        );
    }
}

/// Tool output that itself contains a marker — a coding agent reading a log or
/// a session transcript — must not have its content rewritten into a false
/// retrieval instruction.
#[test]
fn a_marker_inside_the_tools_own_output_is_left_alone() {
    let config = ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    };
    let mut lines: Vec<String> = (0..100).map(|i| format!("line {i}")).collect();
    lines[1] = "[... 42 lines truncated ...]".into(); // in the retained head
    let msg = AgentMessage::Llm(Message::ToolResult {
        tool_call_id: "tc-1".into(),
        tool_name: "bash".into(),
        content: vec![Content::Text {
            text: lines.join("\n"),
        }],
        is_error: false,
        timestamp: 0,
    });

    let out =
        yoagent::context::truncate_tool_output_keyed(msg, &config, Some("tool-out-tc-1-deadbeef"))
            .0;
    let text = text_of(&out);
    assert_eq!(
        text.matches("tool-out-tc-1-deadbeef").count(),
        1,
        "exactly one retrieval instruction — the one truncation emitted: {text}"
    );
    assert!(
        text.contains("[... 42 lines truncated ...]"),
        "the tool's own marker must survive verbatim: {text}"
    );
}

#[tokio::test]
async fn file_backend_evicts_oldest_to_stay_under_its_cap() {
    let dir = tempfile::tempdir().unwrap();
    let state = SharedState::with_backend(yoagent::shared_state::FileBackend::with_max_bytes(
        dir.path(),
        300,
    ));

    // Deliberately non-monotonic names: written z,y,x,w,v,a but sorting
    // lexically a,v,w,x,y,z. With names that sort in write order the filename
    // tiebreak agrees with mtime, so the test would pass whether or not the
    // age ordering worked at all.
    for name in ["z", "y", "x", "w", "v", "a"] {
        state.set(name, "q".repeat(100)).await.unwrap();
    }

    let keys = state.keys().await;
    assert_eq!(
        keys.len(),
        3,
        "cap must bound the directory without emptying it, got {keys:?}"
    );
    assert!(
        keys.contains(&"a".to_string()),
        "the newest write is 'a', which sorts first — it must survive anyway: {keys:?}"
    );
    assert!(
        !keys.contains(&"z".to_string()),
        "the oldest write is 'z', which sorts last — it must go first anyway: {keys:?}"
    );
}

/// A value larger than the cap is rejected, not written-then-evicted. Returning
/// `Ok(())` for a value already deleted would let the loop annotate a marker
/// naming a key that never existed — and would take unrelated keys with it.
#[tokio::test]
async fn an_over_cap_value_is_rejected_and_leaves_the_store_intact() {
    let dir = tempfile::tempdir().unwrap();
    let state = SharedState::with_backend(yoagent::shared_state::FileBackend::with_max_bytes(
        dir.path(),
        300,
    ));
    state.set("keepme", "small".into()).await.unwrap();

    let err = state.set("huge", "x".repeat(5000)).await;
    assert!(
        err.is_err(),
        "an over-cap value must be refused, not silently dropped"
    );
    assert_eq!(
        state.get("keepme").await.as_deref(),
        Some("small"),
        "a refused write must not take existing keys with it"
    );
}

/// What real compaction does to a retrieval pointer.
///
/// Level 1 preserves it — that is the byte-stability property above. But the
/// lossy levels drop whole turns, taking the marker with them while the stash
/// entry lives on, consuming cap quota that can evict entries whose markers
/// *are* still live. Asserting it here so the behaviour is a recorded decision
/// rather than a surprise.
#[test]
fn lossy_compaction_drops_the_marker_while_the_stash_survives() {
    use yoagent::context::compact_messages;

    let config = ContextConfig {
        tool_output_max_lines: 20,
        max_context_tokens: 400,
        system_prompt_tokens: 0,
        ..Default::default()
    };

    let keyed = yoagent::context::truncate_tool_output_keyed(
        big_tool_result("tc-1", 500),
        &config,
        Some("tool-out-tc-1-deadbeef"),
    )
    .0;
    assert!(text_of(&keyed).contains("tool-out-tc-1-deadbeef"));

    // Bury it under enough history that the lossy levels engage.
    let mut history = vec![keyed];
    for i in 0..40 {
        history.push(AgentMessage::Llm(Message::user(format!(
            "turn {i} {}",
            "filler ".repeat(30)
        ))));
    }

    let compacted = compact_messages(history, &config);
    let survives = compacted
        .iter()
        .any(|m| text_of(m).contains("tool-out-tc-1-deadbeef"));

    assert!(
        !survives,
        "documented behaviour: lossy compaction discards the pointer. If this \
         now passes, the stash's lifetime story changed and the cap policy \
         should be revisited"
    );
}
