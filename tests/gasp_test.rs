//! Tests for the GASP bridge (feature `gasp`): event mapping, commit
//! behavior, goal reuse, and interrupted-run recovery.
#![cfg(feature = "gasp")]

use std::process::Command;
use yoagent::gasp::{GaspRecorder, GoalRef};
use yoagent::provider::mock::*;
use yoagent::provider::{MockProvider, ModelConfig};
use yoagent::*;

struct NoopTool;

#[async_trait::async_trait]
impl AgentTool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn label(&self) -> &str {
        "Noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            content: vec![Content::Text { text: "ok".into() }],
            details: serde_json::Value::Null,
        })
    }
}

/// CI runners have no git identity configured; commit_run shells out to
/// plain `git commit`, so provide one via env (idempotent across tests).
fn ensure_git_identity() {
    for (k, v) in [
        ("GIT_AUTHOR_NAME", "yoagent-test"),
        ("GIT_AUTHOR_EMAIL", "test@yolog.dev"),
        ("GIT_COMMITTER_NAME", "yoagent-test"),
        ("GIT_COMMITTER_EMAIL", "test@yolog.dev"),
    ] {
        std::env::set_var(k, v);
    }
}

fn tool_then_text_provider() -> MockProvider {
    MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "noop".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ])
}

/// Like `event_kinds`, but tolerant of a file that is still being written:
/// missing file, or a torn final line, yield what is readable so far. Used by
/// barriers that poll while the recorder appends.
fn event_kinds_lenient(repo: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(repo.join("state/events.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .ok()?
                .get("kind")?
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

fn event_kinds(repo: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(repo.join("state/events.jsonl"))
        .expect("events.jsonl exists")
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["kind"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn records_a_full_run_with_expected_kinds_and_commit() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "test goal".into(),
        },
    )
    .await
    .unwrap();

    let mut agent = Agent::from_provider(tool_then_text_provider(), ModelConfig::mock())
        .with_tools(vec![Box::new(NoopTool)]);
    let (tx, handle) = recorder.recording_sender("do the thing", None);
    agent.prompt_with_sender("do the thing", tx).await;
    let run_id = handle.await.unwrap().unwrap().expect("run recorded");

    let kinds = event_kinds(dir.path());
    // Semantic skeleton, in order (ops_applied lines interleave freely).
    // Allowlist the semantic kinds (bookkeeping kinds like state.ops_applied
    // may grow in yoagent-state minors without breaking this test).
    let semantic: Vec<&str> = kinds
        .iter()
        .map(|s| s.as_str())
        .filter(|k| {
            ["goal.", "run.", "model.", "tool."]
                .iter()
                .any(|p| k.starts_with(p))
        })
        .collect();
    assert_eq!(
        semantic,
        vec![
            "goal.created",
            "run.started",
            "model.called",
            "model.finished",
            "tool.called",
            "tool.finished",
            "model.called",
            "model.finished",
            "run.finished",
        ],
        "full log: {kinds:?}"
    );

    // The run is committed (append-only history is what conformance walks).
    let log = Command::new("git")
        .args(["log", "--oneline"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&log.stdout);
    assert!(
        log.contains(&format!("run {run_id}")),
        "commit missing: {log}"
    );
    assert!(log.contains("completed"));
}

#[tokio::test]
async fn events_are_teed_to_the_forward_sender() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "tee".into(),
        },
    )
    .await
    .unwrap();

    let (ui_tx, mut ui_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut agent = Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("t", Some(ui_tx));
    agent.prompt_with_sender("t", tx).await;
    handle.await.unwrap().unwrap();

    let mut forwarded = 0;
    while ui_rx.try_recv().is_ok() {
        forwarded += 1;
    }
    assert!(forwarded > 0, "UI sender must receive the teed events");
}

#[tokio::test]
async fn goal_is_reused_across_runs_and_recorder_reopens() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "persistent goal".into(),
        },
    )
    .await
    .unwrap();
    let goal = recorder.goal().clone();

    // Run 1.
    let mut agent = Agent::from_provider(MockProvider::text("one"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("run one", None);
    agent.prompt_with_sender("one", tx).await;
    handle.await.unwrap().unwrap();
    drop(recorder);

    // Reopen with the SAME goal — no new goal.created may appear.
    let recorder = GaspRecorder::open(
        dir.path().to_path_buf(),
        "test-agent",
        "w1",
        GoalRef::Existing(goal.clone()),
    )
    .await
    .unwrap();
    assert_eq!(recorder.goal(), &goal);

    let mut agent = Agent::from_provider(MockProvider::text("two"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("run two", None);
    agent.prompt_with_sender("two", tx).await;
    handle.await.unwrap().unwrap();

    let kinds = event_kinds(dir.path());
    assert_eq!(
        kinds.iter().filter(|k| *k == "goal.created").count(),
        1,
        "existing goal must be reused"
    );
    assert_eq!(kinds.iter().filter(|k| *k == "run.finished").count(), 2);
}

#[tokio::test]
async fn dropped_sender_without_agent_end_closes_run_as_interrupted() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "crash".into(),
        },
    )
    .await
    .unwrap();

    // Simulate a crashed loop: send AgentStart, then drop the sender.
    let (tx, handle) = recorder.recording_sender("doomed", None);
    tx.send(AgentEvent::AgentStart).unwrap();
    drop(tx);
    handle.await.unwrap().unwrap();

    let kinds = event_kinds(dir.path());
    assert!(kinds.iter().any(|k| k == "run.finished"));
    let last_finish = std::fs::read_to_string(dir.path().join("state/events.jsonl"))
        .unwrap()
        .lines()
        .rfind(|l| l.contains("run.finished"))
        .unwrap()
        .to_string();
    assert!(last_finish.contains("interrupted"), "got: {last_finish}");
}

// ---------------------------------------------------------------------------
// Review batch: restore-from-clone, failure paths, outcomes, validation
// ---------------------------------------------------------------------------

/// The GASP restore operation IS `git clone` — a clone must contain the
/// manifest, identity, and the committed event log.
#[tokio::test]
async fn fresh_clone_restores_manifest_identity_and_log() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "clone me".into(),
        },
    )
    .await
    .unwrap();

    let mut agent = Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("t", None);
    agent.prompt_with_sender("t", tx).await;
    handle.await.unwrap().unwrap().expect("run recorded");

    let clone_dir = tempfile::tempdir().unwrap();
    let clone_path = clone_dir.path().join("restored");
    let out = Command::new("git")
        .args(["clone", "-q", dir.path().to_str().unwrap()])
        .arg(&clone_path)
        .output()
        .unwrap();
    assert!(out.status.success());

    assert!(
        clone_path.join("AGENT.md").is_file(),
        "manifest must restore"
    );
    assert!(
        clone_path.join("identity").is_dir(),
        "identity must restore"
    );
    let kinds = event_kinds(&clone_path);
    assert!(kinds.iter().any(|k| k == "goal.created"));
    assert!(kinds.iter().any(|k| k == "run.finished"));
}

/// A mid-run recording failure must NOT blind the UI tee: forwarding
/// continues, the error surfaces via the handle, and nothing new commits.
#[tokio::test]
#[cfg(unix)]
async fn sink_failure_keeps_tee_alive_and_surfaces_error() {
    use std::os::unix::fs::PermissionsExt;
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "doomed".into(),
        },
    )
    .await
    .unwrap();

    // Make the log unwritable so the first append fails.
    let events = dir.path().join("state/events.jsonl");
    std::fs::set_permissions(&events, std::fs::Permissions::from_mode(0o444)).unwrap();

    let (ui_tx, mut ui_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut agent = Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("t", Some(ui_tx));
    agent.prompt_with_sender("t", tx).await;
    let result = handle.await.unwrap();

    // Restore permissions so the tempdir cleans up everywhere.
    std::fs::set_permissions(&events, std::fs::Permissions::from_mode(0o644)).unwrap();

    assert!(
        result.is_err(),
        "recording failure must surface via the handle"
    );
    // The tee survived: events kept flowing, ending with AgentEnd.
    let mut last = None;
    while let Ok(e) = ui_rx.try_recv() {
        last = Some(e);
    }
    assert!(
        matches!(last, Some(AgentEvent::AgentEnd { .. })),
        "tee must deliver events through AgentEnd despite recording failure"
    );
}

/// with_store's crash recovery: an aborted consumer leaves an open run; the
/// next open closes it as interrupted and can record again.
#[tokio::test]
async fn reopen_after_crash_closes_stale_run_and_records_again() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "crashy".into(),
        },
    )
    .await
    .unwrap();
    let goal = recorder.goal().clone();

    // Open a run, then kill the consumer before its drop-fallback can close
    // it — a faithful crash simulation.
    let (tx, handle) = recorder.recording_sender("doomed", None);
    tx.send(AgentEvent::AgentStart).unwrap();

    // Wait for the run to be *fully* open, not just logged. `record_run_started`
    // writes the `run.started` event, opens the run marker, and only then
    // applies the ops that create the run node — a window its own source
    // comments call out. Aborting on `run.started` alone can land inside it,
    // leaving a run whose node was never written; the reopen below then fails
    // with "node not found" instead of exercising the stale-run recovery this
    // test is about. The node-creating ops arrive as the `state.ops_applied`
    // event that follows, so that is the barrier.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let kinds = event_kinds_lenient(dir.path());
        if let Some(started) = kinds.iter().position(|k| k == "run.started") {
            if kinds[started..].iter().any(|k| k == "state.ops_applied") {
                break;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run never opened fully; saw events: {:?}",
            event_kinds_lenient(dir.path())
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    handle.abort();
    let _ = handle.await;
    drop(tx);
    drop(recorder);

    // Reopen: the stale run must be closed and a new run must succeed.
    let recorder = GaspRecorder::open(
        dir.path().to_path_buf(),
        "test-agent",
        "w1",
        GoalRef::Existing(goal),
    )
    .await
    .expect("reopen after crash");

    let mut agent = Agent::from_provider(MockProvider::text("recovered"), ModelConfig::mock());
    let (tx, handle) = recorder.recording_sender("recovery run", None);
    agent.prompt_with_sender("go", tx).await;
    handle.await.unwrap().unwrap().expect("new run recorded");

    let kinds = event_kinds(dir.path());
    assert_eq!(kinds.iter().filter(|k| *k == "run.started").count(), 2);
    assert_eq!(kinds.iter().filter(|k| *k == "run.finished").count(), 2);
}

/// No AgentStart → Ok(None): callers never get an id that isn't in the log.
#[tokio::test]
async fn no_agent_start_returns_none_and_writes_nothing() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "nothing".into(),
        },
    )
    .await
    .unwrap();

    let (tx, handle) = recorder.recording_sender("never runs", None);
    drop(tx);
    let outcome = handle.await.unwrap().unwrap();
    assert!(outcome.is_none(), "no run happened — no RunId");

    let kinds = event_kinds(dir.path());
    assert!(!kinds.iter().any(|k| k.starts_with("run.")));
}

/// outcome_for mapping, pinned through the durable log via synthetic events.
#[tokio::test]
async fn stop_reasons_map_to_distinct_outcomes() {
    ensure_git_identity();
    for (stop, expected) in [
        (StopReason::Length, "truncated"),
        (StopReason::Error, "error"),
        (StopReason::Aborted, "aborted"),
        (StopReason::Refusal, "refused"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let recorder = GaspRecorder::init(
            dir.path(),
            "test-agent",
            "w1",
            GoalRef::New {
                title: "outcomes".into(),
            },
        )
        .await
        .unwrap();
        let (tx, handle) = recorder.recording_sender("t", None);
        tx.send(AgentEvent::AgentStart).unwrap();
        tx.send(AgentEvent::MessageEnd {
            message: AgentMessage::Llm(Message::assistant(
                vec![Content::Text { text: "x".into() }],
                stop.clone(),
                "m",
                "mock",
                Usage::default(),
            )),
        })
        .unwrap();
        tx.send(AgentEvent::AgentEnd { messages: vec![] }).unwrap();
        drop(tx);
        handle.await.unwrap().unwrap().expect("recorded");

        let log = std::fs::read_to_string(dir.path().join("state/events.jsonl")).unwrap();
        let finish = log
            .lines()
            .rfind(|l| l.contains("run.finished"))
            .unwrap()
            .to_string();
        assert!(
            finish.contains(expected),
            "stop {stop:?} must record outcome {expected}; got {finish}"
        );
    }
}

/// An input-filter rejection is a policy outcome, not a crash.
#[tokio::test]
async fn input_rejected_runs_record_outcome_rejected() {
    ensure_git_identity();
    struct RejectAll;
    impl InputFilter for RejectAll {
        fn filter(&self, _text: &str) -> FilterResult {
            FilterResult::Reject("policy".into())
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "reject".into(),
        },
    )
    .await
    .unwrap();

    let mut agent = Agent::from_provider(MockProvider::text("unused"), ModelConfig::mock())
        .with_input_filter(RejectAll);
    let (tx, handle) = recorder.recording_sender("t", None);
    agent.prompt_with_sender("anything", tx).await;
    handle.await.unwrap().unwrap().expect("recorded");

    let log = std::fs::read_to_string(dir.path().join("state/events.jsonl")).unwrap();
    let finish = log.lines().rfind(|l| l.contains("run.finished")).unwrap();
    assert!(finish.contains("rejected"), "got: {finish}");
}

/// A dangling GoalRef::Existing must fail loudly at open.
#[tokio::test]
async fn dangling_existing_goal_errors_at_open() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    // Create a valid repo first.
    let r = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "real".into(),
        },
    )
    .await
    .unwrap();
    drop(r);

    let bogus = yoagent::gasp::GoalId::generate();
    let err = GaspRecorder::open(
        dir.path().to_path_buf(),
        "test-agent",
        "w1",
        GoalRef::Existing(bogus),
    )
    .await;
    assert!(err.is_err(), "dangling goal id must be rejected");
}

/// The tee must deliver the complete stream: identical event kinds to a
/// direct (untee'd) run of the same deterministic agent.
#[tokio::test]
async fn tee_delivers_the_complete_event_stream() {
    ensure_git_identity();
    fn kind_of(e: &AgentEvent) -> &'static str {
        match e {
            AgentEvent::AgentStart => "AgentStart",
            AgentEvent::AgentEnd { .. } => "AgentEnd",
            AgentEvent::TurnStart => "TurnStart",
            AgentEvent::TurnEnd { .. } => "TurnEnd",
            AgentEvent::MessageStart { .. } => "MessageStart",
            AgentEvent::MessageUpdate { .. } => "MessageUpdate",
            AgentEvent::MessageEnd { .. } => "MessageEnd",
            AgentEvent::ToolExecutionStart { .. } => "ToolExecutionStart",
            AgentEvent::ToolExecutionUpdate { .. } => "ToolExecutionUpdate",
            AgentEvent::ToolExecutionEnd { .. } => "ToolExecutionEnd",
            AgentEvent::ProgressMessage { .. } => "ProgressMessage",
            AgentEvent::InputRejected { .. } => "InputRejected",
            _ => "Unknown",
        }
    }

    // Direct run.
    let mut agent = Agent::from_provider(MockProvider::text("same"), ModelConfig::mock());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    agent.prompt_with_sender("t", tx).await;
    let mut direct = Vec::new();
    while let Ok(e) = rx.try_recv() {
        direct.push(kind_of(&e));
    }

    // Teed run of the identical agent config.
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "tee-eq".into(),
        },
    )
    .await
    .unwrap();
    let mut agent = Agent::from_provider(MockProvider::text("same"), ModelConfig::mock());
    let (ui_tx, mut ui_rx) = tokio::sync::mpsc::unbounded_channel();
    let (tx, handle) = recorder.recording_sender("t", Some(ui_tx));
    agent.prompt_with_sender("t", tx).await;
    handle.await.unwrap().unwrap().expect("recorded");
    let mut teed = Vec::new();
    while let Ok(e) = ui_rx.try_recv() {
        teed.push(kind_of(&e));
    }

    assert_eq!(direct, teed, "tee must deliver every event, in order");
}

#[tokio::test]
async fn tool_fingerprint_and_usage_reach_the_log() {
    ensure_git_identity();
    let dir = tempfile::tempdir().unwrap();
    let recorder = GaspRecorder::init(
        dir.path(),
        "test-agent",
        "w1",
        GoalRef::New {
            title: "measurement goal".into(),
        },
    )
    .await
    .unwrap();

    let mut agent = Agent::from_provider(tool_then_text_provider(), ModelConfig::mock())
        .with_tools(vec![Box::new(NoopTool)]);
    let (tx, handle) = recorder.recording_sender("measured run", None);
    agent.prompt_with_sender("measured run", tx).await;
    handle.await.unwrap().unwrap().expect("run recorded");

    let raw = std::fs::read_to_string(dir.path().join("state/events.jsonl")).unwrap();
    let events: Vec<serde_json::Value> = raw
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // tool.called must carry a stable fingerprint so calls can be matched
    // across the log — input_summary is truncated and cannot be.
    let tool_called = events
        .iter()
        .find(|e| e["kind"] == "tool.called")
        .expect("tool.called recorded");
    let fp = tool_called["payload"]["metadata"]["args_fingerprint"]
        .as_str()
        .expect("args_fingerprint present");
    assert!(fp.starts_with("noop:"), "got {fp}");

    // model.finished must carry usage — this is what makes the log
    // sufficient for cost analysis and compaction inference.
    let model_finished = events
        .iter()
        .find(|e| e["kind"] == "model.finished")
        .expect("model.finished recorded");
    let usage = &model_finished["payload"]["metadata"]["usage"];
    assert!(usage["input"].is_number(), "usage.input missing: {usage}");
    assert!(usage["output"].is_number());
    assert!(usage["cache_read"].is_number());
    assert!(usage["cache_write"].is_number());
}
