//! GASP bridge — record agent runs into a [GASP](https://github.com/yologdev/gasp)
//! agent repo (feature `gasp`).
//!
//! GASP ("the repo is the agent") keeps an agent's durable self in a git
//! repository: an append-only semantic event log (`state/events.jsonl`) that
//! folds into a typed goal/run/model/tool graph, with restore = `clone +
//! replay`. This module is the bridge between yoagent's [`AgentEvent`] stream
//! and the [`yoagent_state`] reference runtime — **zero agent-loop changes**;
//! the recorder is just another consumer of the event stream.
//!
//! ```no_run
//! use yoagent::{Agent, gasp::{GaspRecorder, GoalRef}, provider::ModelConfig};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let recorder = GaspRecorder::init(
//!     "./my-agent-repo",
//!     "my-agent",
//!     "worker-1",
//!     GoalRef::New { title: "ship the feature".into() },
//! )
//! .await?;
//!
//! let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"));
//! let (tx, record_handle) = recorder.recording_sender("implement the parser", None);
//! agent.prompt_with_sender("implement the parser", tx).await;
//! let run_id = record_handle.await??.expect("run recorded");
//! # let _ = run_id; Ok(())
//! # }
//! ```
//!
//! # What is persisted
//!
//! The semantic log stores bounded one-line summaries — the **task string
//! (verbatim)**, model ids, and the **first 200 characters of tool inputs,
//! tool outputs, and assistant text** — never full transcripts. If secrets
//! can flow through tool arguments or outputs (API keys, connection
//! strings), install a redacting summarizer via
//! [`GaspRecorder::with_summarizer`] **before** recording: the log lives in a
//! git repo designed to be cloned and shared, and committed history is hard
//! to scrub. Full transcripts belong in GASP's cold `transcripts/` tier —
//! [`Session::to_jsonl`](crate::Session::to_jsonl) is a natural format for it.
//!
//! Recording requires a git identity (`user.name`/`user.email`), like any
//! git workflow.

use crate::types::*;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use yoagent_state::{
    Goal, YoAgentModelCalled, YoAgentModelFinished, YoAgentRunFinished, YoAgentRunStarted,
    YoAgentStateAdapter, YoAgentStateSink, YoAgentToolCalled, YoAgentToolFinished,
};
// The extension path (see [`GaspRecorder::state`]): applications recording the
// goal/task/verdict tier need no direct `yoagent-state` dependency, so every
// type on that path must be nameable here. Three rounds of bug reports each
// found one tier still missing, so the rule is now mechanical rather than
// judged — see `tests/gasp_test.rs`:
//
//   1. the receiver (`YoAgentState`) and what it needs        — #111
//   2. the recorded structs and their id types                — #115
//   3. those structs' *field* types, and the argument types
//      of every `YoAgentState` method                         — #117
//
// (3) is the one that subsumes the others: a struct can be constructible
// without being usable, because a smart constructor defaults a field whose
// type the caller can never name.
pub use yoagent_state::{
    ActorRef, ArtifactRef, Decision, DecisionId, DecisionStatus, EvalId, EvalResult, EvalStatus,
    Event, EventId, EventStore, ExpectedEffect, Frame, FrameId, GitEventStore, Goal as GaspGoal,
    GoalId, GoalStatus, Hypothesis, HypothesisId, ModelCall, Node, NodeId, Observation,
    ObservationId, PatchId, PatchStatus, Precondition, ProjectRef, ProjectSnapshot, RunId,
    StateError, StateOp, StatePatch, Task, TaskId, TaskStatus, ToolCall as GaspToolCall,
    YoAgentState,
};

/// Which GASP goal recorded runs belong to (stamped into each run-boundary
/// commit's `Goal:` trailer).
#[derive(Debug, Clone)]
pub enum GoalRef {
    /// Use an existing goal. Validated at open: the goal must exist in the
    /// repo's graph, so a typo'd persisted id fails loudly instead of
    /// chaining runs to a goal that exists nowhere.
    Existing(GoalId),
    /// Create a new goal with this title when the recorder opens.
    New { title: String },
}

type Summarizer = std::sync::Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Records an agent's [`AgentEvent`] stream into a GASP agent repo.
///
/// One recorder = one writer (`worker_id` names it in the repo lease) and one
/// goal; each [`recording_sender`](Self::recording_sender) call records one
/// run. **Runs are sequential**: a second in-flight `recording_sender` run
/// fails (`run X is already open`) — one run at a time per repo. Events are
/// appended to `state/events.jsonl` as they arrive and committed when the run
/// closes, so the git history stays append-only (GASP conformance check 4).
/// The repo must have a **single writer**: two live workers sharing a repo
/// contend on the lease and can interrupt each other's runs.
pub struct GaspRecorder {
    state: YoAgentState<GitEventStore>,
    store: GitEventStore,
    actor: ActorRef,
    goal: GoalId,
    summarize: Summarizer,
}

impl GaspRecorder {
    /// Initialize a fresh agent repo at `root` (git init + minimal manifest,
    /// committed so a clone restores it) and open a recorder on it.
    pub async fn init(
        root: impl AsRef<std::path::Path>,
        agent_id: &str,
        worker_id: &str,
        goal: GoalRef,
    ) -> Result<Self, StateError> {
        let store = yoagent_state::init_agent_repo(root, agent_id, worker_id)?;
        Self::with_store(store, agent_id, goal).await
    }

    /// Open a recorder on an existing GASP agent repo.
    pub async fn open(
        root: impl Into<std::path::PathBuf>,
        agent_id: &str,
        worker_id: &str,
        goal: GoalRef,
    ) -> Result<Self, StateError> {
        let store = GitEventStore::open(root, worker_id)?;
        Self::with_store(store, agent_id, goal).await
    }

    /// Open a recorder on a store the caller owns.
    ///
    /// This is the extension path for applications that record more than the
    /// run/model/tool tier — goals, tasks, verdicts, evals — into the *same*
    /// ledger: open the [`GitEventStore`] yourself, build a
    /// [`YoAgentState`](yoagent_state::YoAgentState) on a clone of it for your
    /// own `record_*` calls, and hand the recorder this handle. One store, one
    /// writer process, no cross-process coordination.
    ///
    /// # Writer model
    ///
    /// - **One store per agent**; concurrent writer *processes* on one clone
    ///   are not supported (the open-run check below assumes it).
    /// - `worker_id` (in [`GitEventStore::open`]) distinguishes writers in the
    ///   log; use a distinct id per host/CI lane.
    /// - The recorder commits at **run close** (stream end). Events appended
    ///   between commits are durable in the working tree but unpushed —
    ///   callers on ephemeral runners should push after each run, or crashed
    ///   sessions vanish from the corpus.
    /// - A run left open by a crashed process is closed as `"interrupted"` on
    ///   the next open.
    pub async fn with_store(
        store: GitEventStore,
        agent_id: &str,
        goal: GoalRef,
    ) -> Result<Self, StateError> {
        let actor = ActorRef::agent(agent_id);
        let state = YoAgentState::load(store.clone()).await?;

        // The open-run marker is in-memory only; `resume_open_run` restores
        // it from the log. A run left open by a crashed process is closed
        // here for log hygiene — no unpaired `run.started` may leak across
        // process boundaries.
        if let Some(stale) = state.resume_open_run().await? {
            tracing::warn!(run = %stale, "closing stale open run as interrupted");
            state
                .record_run_finished(actor.clone(), stale, "interrupted")
                .await
                .map_err(|e| {
                    StateError::Validation(format!(
                        "found an open run and could not close it ({e}); if another \
                         worker is live on this repo, do not share it — GASP repos \
                         are single-writer"
                    ))
                })?;
        }

        let goal = match goal {
            GoalRef::Existing(id) => {
                if state.get_node(NodeId::new(id.as_str())).await.is_none() {
                    return Err(StateError::Validation(format!(
                        "goal {id} does not exist in this repo's graph"
                    )));
                }
                id
            }
            GoalRef::New { title } => {
                let id = GoalId::generate();
                state
                    .record_goal(Goal::new(id.clone(), title.clone(), title, actor.clone()))
                    .await?;
                id
            }
        };

        // Commit the scaffolding (AGENT.md, identity/, .gitignore) plus any
        // events recorded above. Without this, a fresh `git clone` — the
        // restore operation GASP is built around — has no manifest and fails
        // conformance check 6.
        commit_scaffolding(&store)?;

        Ok(Self {
            state,
            store,
            actor,
            goal,
            summarize: std::sync::Arc::new(|text: &str| summarize(text)),
        })
    }

    /// Replace the default summarizer (single line, 200 chars) — e.g. to
    /// redact secrets from tool arguments/outputs before they are persisted
    /// to the shareable git repo.
    pub fn with_summarizer(
        mut self,
        summarize: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        self.summarize = std::sync::Arc::new(summarize);
        self
    }

    /// The goal this recorder's runs belong to (persist it to reuse across
    /// processes via [`GoalRef::Existing`]).
    /// The recorder's own [`YoAgentState`], for recording tiers this crate
    /// does not model — goals, tasks, evals, decisions, patches.
    ///
    /// Prefer this over opening a second [`GitEventStore`] on the same root: a
    /// GASP repo is single-writer behind a 600-second lease, so a second store
    /// collides with this one rather than cooperating.
    ///
    /// ```no_run
    /// # use yoagent::gasp::{GaspRecorder, GoalRef, Task, TaskId, TaskStatus};
    /// # async fn demo(recorder: &GaspRecorder) -> Result<(), Box<dyn std::error::Error>> {
    /// recorder.state().record_task(Task {
    ///     id: TaskId::new("task_1"),
    ///     title: "ship the thing".into(),
    ///     summary: "planned this session".into(),
    ///     status: TaskStatus::Open,
    ///     goal: Some(recorder.goal().clone()),
    ///     created_by: recorder.actor().clone(),
    ///     metadata: serde_json::json!({}),
    /// }).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn state(&self) -> &YoAgentState<GitEventStore> {
        &self.state
    }

    /// The underlying event store, for callers that need it directly (e.g. to
    /// scan events). Shares this recorder's lease — see [`state`](Self::state).
    pub fn store(&self) -> &GitEventStore {
        &self.store
    }

    /// The actor recorded events are attributed to — the first argument of the
    /// `YoAgentState::record_*` methods.
    pub fn actor(&self) -> &ActorRef {
        &self.actor
    }

    pub fn goal(&self) -> &GoalId {
        &self.goal
    }

    /// Returns a sender to pass to
    /// [`Agent::prompt_with_sender`](crate::Agent::prompt_with_sender) (or
    /// the raw loop) and a handle resolving to the recorded [`RunId`] —
    /// `Ok(None)` when the stream carried no run at all (`AgentStart` never
    /// arrived), so callers never receive an id that isn't in the log.
    ///
    /// **The handle is the only error channel** — always await it. If
    /// recording fails mid-run (disk, lease, git), recording stops and the
    /// error is returned by the handle, but **event forwarding continues**:
    /// every event is teed to `forward` (your UI) before recording, so a
    /// recorder failure never blinds the UI.
    pub fn recording_sender(
        &self,
        task: impl Into<String>,
        forward: Option<mpsc::UnboundedSender<AgentEvent>>,
    ) -> (
        mpsc::UnboundedSender<AgentEvent>,
        JoinHandle<Result<Option<RunId>, StateError>>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = YoAgentStateAdapter::new(self.state.clone(), self.actor.clone());
        let store = self.store.clone();
        let goal = self.goal.clone();
        let task = task.into();
        let summarize = self.summarize.clone();
        let handle = tokio::spawn(consume(sink, store, goal, task, summarize, rx, forward));
        (tx, handle)
    }
}

/// Per-run bookkeeping threaded through event recording.
struct RunTracking {
    run_id: RunId,
    task: String,
    started: bool,
    finished: bool,
    turn: usize,
    outcome: String,
}

/// Map the event stream onto the GASP sink. Runs in its own task; ends when
/// the sender side is dropped (the loop finished, or the caller dropped the
/// `*_with_sender` future mid-run / the loop task panicked — `AgentEnd` is
/// otherwise sent unconditionally).
async fn consume(
    sink: YoAgentStateAdapter<GitEventStore>,
    store: GitEventStore,
    goal: GoalId,
    task: String,
    summarize: Summarizer,
    mut rx: mpsc::UnboundedReceiver<AgentEvent>,
    forward: Option<mpsc::UnboundedSender<AgentEvent>>,
) -> Result<Option<RunId>, StateError> {
    let mut tracking = RunTracking {
        run_id: RunId::generate(),
        task,
        started: false,
        finished: false,
        turn: 0,
        outcome: "interrupted".to_string(),
    };
    let mut recording_error: Option<StateError> = None;

    while let Some(event) = rx.recv().await {
        // Forward FIRST: the tee observes the loop, not the recorder's disk.
        // It must neither lag behind per-event fsyncs nor die when recording
        // fails.
        if let Some(fwd) = &forward {
            let _ = fwd.send(event.clone());
        }
        if recording_error.is_some() {
            continue; // recording is dead; keep draining + forwarding
        }
        if let Err(e) = record_event(&sink, &summarize, &mut tracking, &event).await {
            tracing::error!(
                run = %tracking.run_id,
                error = %e,
                "GASP recording failed; recording stops but event forwarding continues"
            );
            recording_error = Some(e);
        }
    }

    if let Some(e) = recording_error {
        let _ = store.release_lease();
        return Err(e);
    }
    if !tracking.started {
        // No AgentStart ever arrived: nothing was recorded — say so instead
        // of fabricating a RunId that exists nowhere in the log.
        return Ok(None);
    }
    if !tracking.finished {
        // Sender dropped without AgentEnd: close the run with the outcome
        // derived so far (matches the commit trailer below).
        sink.on_run_finished(YoAgentRunFinished {
            run_id: tracking.run_id.clone(),
            outcome: tracking.outcome.clone(),
            metadata: serde_json::json!({}),
        })
        .await?;
    }
    store.commit_run(&tracking.run_id, &goal, &tracking.outcome, &[])?;
    // Free the lease so another worker (or the next process) can record
    // immediately instead of waiting out the TTL.
    let _ = store.release_lease();
    Ok(Some(tracking.run_id))
}

/// Record a single event. Mutates tracking state; any sink error aborts
/// recording (handled by the caller) without touching the forwarding path.
async fn record_event(
    sink: &YoAgentStateAdapter<GitEventStore>,
    summarize: &Summarizer,
    tracking: &mut RunTracking,
    event: &AgentEvent,
) -> Result<(), StateError> {
    match event {
        AgentEvent::AgentStart => {
            sink.on_run_started(YoAgentRunStarted {
                run_id: tracking.run_id.clone(),
                task: tracking.task.clone(),
                metadata: serde_json::json!({}),
            })
            .await?;
            tracking.started = true;
        }
        AgentEvent::MessageEnd {
            message:
                AgentMessage::Llm(Message::Assistant {
                    content,
                    model,
                    stop_reason,
                    usage,
                    ..
                }),
        } if tracking.started => {
            tracking.turn += 1;
            sink.on_model_called(YoAgentModelCalled {
                run_id: tracking.run_id.clone(),
                model: model.clone(),
                prompt_summary: if tracking.turn == 1 {
                    summarize(&tracking.task)
                } else {
                    format!("turn {}", tracking.turn)
                },
            })
            .await?;
            let text = content
                .iter()
                .find_map(|c| match c {
                    Content::Text { text } if !text.is_empty() => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("(no text)");
            // Token usage makes the log sufficient for offline cost analysis,
            // and makes compactions inferable: a sharp drop in input tokens
            // between consecutive model calls in one run is the compaction
            // signature (no separate event kind needed).
            sink.on_model_finished(YoAgentModelFinished {
                run_id: tracking.run_id.clone(),
                model: model.clone(),
                output_summary: summarize(text),
                metadata: serde_json::json!({
                    "usage": {
                        "input": usage.input,
                        "output": usage.output,
                        "cache_read": usage.cache_read,
                        "cache_write": usage.cache_write,
                    }
                }),
            })
            .await?;
            tracking.outcome = outcome_for(stop_reason).to_string();
        }
        AgentEvent::ToolExecutionStart {
            tool_name, args, ..
        } if tracking.started => {
            sink.on_tool_called(YoAgentToolCalled {
                run_id: tracking.run_id.clone(),
                tool: tool_name.clone(),
                input_summary: summarize(&args.to_string()),
                metadata: serde_json::json!({
                    "args_fingerprint": args_fingerprint(tool_name, args),
                }),
            })
            .await?;
        }
        AgentEvent::ToolExecutionEnd {
            tool_name,
            result,
            is_error,
            ..
        } if tracking.started => {
            let text = result
                .content
                .iter()
                .find_map(|c| match c {
                    Content::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("(no output)");
            sink.on_tool_finished(YoAgentToolFinished {
                run_id: tracking.run_id.clone(),
                tool: tool_name.clone(),
                output_summary: summarize(text),
                success: !is_error,
            })
            .await?;
        }
        AgentEvent::LoopDetected {
            aborted: true,
            tool_name,
            ..
        } if tracking.started => {
            // Without this the run's outcome stays whatever the last stop
            // reason mapped to — `ToolUse` -> "completed" — so the durable log
            // recorded a budget-exhausting loop abort as a clean success, the
            // exact opposite of what this event exists to make auditable.
            tracking.outcome = format!("loop_aborted:{tool_name}");
        }
        AgentEvent::InputRejected { .. } if tracking.started => {
            // A policy rejection is not a crash — label it distinctly in the
            // durable log.
            tracking.outcome = "rejected".to_string();
        }
        AgentEvent::AgentEnd { stats, .. } if tracking.started && !tracking.finished => {
            // The session rollup goes into the durable record so an audit can
            // compare runs without replaying every model.finished entry.
            sink.on_run_finished(YoAgentRunFinished {
                run_id: tracking.run_id.clone(),
                outcome: tracking.outcome.clone(),
                metadata: {
                    // Serialize the struct rather than hand-building the
                    // object: a second hand-maintained shape would drift from
                    // the event wire format, and an auditor comparing the two
                    // would silently be comparing different keys.
                    let mut m = serde_json::to_value(stats).unwrap_or_else(
                        |_| serde_json::json!({ "error": "session stats not serializable" }),
                    );
                    // Derived, so it is not a struct field — but it is the
                    // number an audit actually reads.
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert(
                            "cacheHitRate".into(),
                            serde_json::json!(stats.cache_hit_rate()),
                        );
                    }
                    m
                },
            })
            .await?;
            tracking.finished = true;
        }
        _ => {}
    }
    Ok(())
}

/// Commit the repo scaffolding (manifest, identity, gitignore) and any
/// pre-run events (goal creation, stale-run closure) so `git clone` restores
/// a complete agent. No-op when nothing changed.
fn commit_scaffolding(store: &GitEventStore) -> Result<(), StateError> {
    let events = store.events_path();
    let root = events
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| StateError::Store("events path has no repo root".into()))?
        .to_path_buf();
    let run = |args: &[&str]| -> Result<std::process::Output, StateError> {
        std::process::Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .map_err(|e| StateError::Store(format!("git {}: {e}", args.join(" "))))
    };
    run(&[
        "add",
        "--",
        "AGENT.md",
        "identity",
        ".gitignore",
        "state/events.jsonl",
    ])?;
    let staged = run(&["diff", "--cached", "--quiet"])?;
    if !staged.status.success() {
        let out = run(&["commit", "-q", "-m", "gasp: agent scaffolding"])?;
        if !out.status.success() {
            return Err(StateError::Store(format!(
                "scaffolding commit failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
    }
    Ok(())
}

/// One-line, bounded summary for semantic events — full content belongs in
/// the transcripts tier, not the event log.
/// Stable identity for a tool call, so calls can be *matched* across a log —
/// `input_summary` is a truncated human summary and cannot be.
///
/// Normalized per tool rather than hashing the raw argument bytes: the file
/// tools page and re-slice (`offset`/`limit` on `read_file` since 0.15), so a
/// re-read of the same file arrives with different argument bytes. Hashing the
/// full JSON would then undercount re-fetches for exactly the tools where they
/// are most diagnostic. Identity per tool:
///
/// - `read_file` / `edit_file` / `write_file` / `list_files`: the `path` /
///   `directory` argument
/// - `bash`: the `command`
/// - anything else: the full arguments, serialized
fn args_fingerprint(tool_name: &str, args: &serde_json::Value) -> String {
    let identity = match tool_name {
        "read_file" | "edit_file" | "write_file" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        "list_files" => args
            .get("directory")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        "bash" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    }
    .unwrap_or_else(|| args.to_string());

    // FNV-1a: stable across runs and platforms (DefaultHasher is neither
    // guaranteed stable across Rust releases nor documented as such).
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in identity.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{tool_name}:{hash:016x}")
}

fn summarize(text: &str) -> String {
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= 200 {
        one_line
    } else {
        let truncated: String = one_line.chars().take(200).collect();
        format!("{truncated}…")
    }
}

fn outcome_for(stop_reason: &StopReason) -> &'static str {
    match stop_reason {
        StopReason::Stop | StopReason::ToolUse => "completed",
        StopReason::Length => "truncated",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        StopReason::Refusal => "refused",
    }
}

#[cfg(test)]
mod tests {
    use super::{args_fingerprint, summarize};

    #[test]
    fn fingerprint_is_stable_and_tool_scoped() {
        let args = serde_json::json!({"path": "src/main.rs"});
        let a = args_fingerprint("read_file", &args);
        let b = args_fingerprint("read_file", &args);
        assert_eq!(a, b, "same call must fingerprint identically");
        assert!(a.starts_with("read_file:"));
        // Same args through a different tool is a different call.
        assert_ne!(a, args_fingerprint("edit_file", &args));
    }

    #[test]
    fn fingerprint_ignores_read_paging() {
        // read_file pages since 0.15: a re-read of a lost file arrives with
        // different offset/limit bytes. Identity is the path, or re-fetches
        // of exactly the most diagnostic tool go uncounted.
        let full = serde_json::json!({"path": "src/big.rs"});
        let page = serde_json::json!({"path": "src/big.rs", "offset": 501, "limit": 500});
        assert_eq!(
            args_fingerprint("read_file", &full),
            args_fingerprint("read_file", &page)
        );
        // Different files stay distinct.
        let other = serde_json::json!({"path": "src/other.rs"});
        assert_ne!(
            args_fingerprint("read_file", &full),
            args_fingerprint("read_file", &other)
        );
    }

    #[test]
    fn fingerprint_bash_keys_on_command_and_default_on_full_args() {
        let a = serde_json::json!({"command": "cargo test"});
        let b = serde_json::json!({"command": "cargo test", "timeout": 60});
        // bash identity is the command; ancillary knobs don't split it.
        assert_eq!(args_fingerprint("bash", &a), args_fingerprint("bash", &b));
        // Unknown tools fall back to full-args identity.
        let x = serde_json::json!({"q": "foo"});
        let y = serde_json::json!({"q": "bar"});
        assert_ne!(
            args_fingerprint("web_search", &x),
            args_fingerprint("web_search", &y)
        );
        // Malformed / non-object args must not panic.
        let weird = serde_json::json!("just a string");
        assert!(args_fingerprint("read_file", &weird).starts_with("read_file:"));
    }

    #[test]
    fn summarize_collapses_and_truncates_on_char_boundaries() {
        assert_eq!(summarize("a\nb\t c"), "a b c");
        // 300 multibyte chars: must truncate at 200 CHARS (not bytes) + '…'.
        let long: String = "ö".repeat(300);
        let s = summarize(&long);
        assert_eq!(s.chars().count(), 201);
        assert!(s.ends_with('…'));
        // Exactly 200 chars: untouched.
        let exact: String = "x".repeat(200);
        assert_eq!(summarize(&exact), exact);
    }
}
