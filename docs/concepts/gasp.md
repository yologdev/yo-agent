# GASP: Your Agent Is a Git Repo

[GASP](https://github.com/yologdev/gasp) — the Git Agent State Protocol —
keeps an agent's durable self in a git repository: an **append-only semantic
event log** (`state/events.jsonl`) that folds into a typed
goal/run/model/tool graph, alongside identity, skills, and memory tiers.
Restore = `git clone` + replay. Clone your agent onto a new machine and it
remembers everything, with lineage.

yoagent bridges to GASP through the `gasp` feature (backed by
[`yoagent-state`](https://crates.io/crates/yoagent-state), the reference
runtime). The bridge is a consumer of the [`AgentEvent`] stream — **zero
agent-loop changes**:

```toml
yoagent = { version = "0.15", features = ["gasp"] }
```


```rust
use yoagent::gasp::{GaspRecorder, GoalRef};

let recorder = GaspRecorder::init(
    "./my-agent-repo", "my-agent", "worker-1",
    GoalRef::New { title: "ship the feature".into() },
).await?;

let (tx, record_handle) = recorder.recording_sender("implement the parser", None);
agent.prompt_with_sender("implement the parser", tx).await;
let run_id = record_handle.await??;   // events appended + committed
```

## What gets recorded

| Agent activity | GASP events |
|---|---|
| loop starts | `run.started` (the goal is stamped in the run commit's `Goal:` trailer) |
| each assistant turn | `model.called` / `model.finished` paired nodes |
| each tool execution | `tool.called` / `tool.finished` (with success flag) |
| loop ends | `run.finished` with the outcome (`completed` / `error` / `aborted` / ...) |

The semantic log stores bounded one-line summaries — the **task string
(verbatim)**, model ids, and the **first 200 characters of tool inputs, tool
outputs, and assistant text** — never full transcripts. If secrets can flow
through tool arguments or outputs, install a redacting summarizer
(`recorder.with_summarizer(...)`) before recording: the repo is designed to
be cloned and shared, and committed history is hard to scrub. Full
transcripts belong in GASP's cold `transcripts/` tier —
[`Session::to_jsonl`](session-trees.md) is a natural format for it.

Robustness: a crashed process leaves no open run — the recorder closes stale
runs as `interrupted` on open, and a dropped event stream finishes the run
with the outcome derived so far. One git commit per run keeps the history
append-only, and the init scaffolding (manifest, identity) is committed so a
fresh clone is a complete, conformant agent. If recording fails mid-run
(disk, lease, git), recording stops and the error surfaces via the returned
handle — **always await it** — while event forwarding to your UI continues
uninterrupted. GASP repos are single-writer: don't share one repo between
live workers, and record one run at a time.

## Tested conformance

yoagent's CI emits an agent repo with a mock provider and runs the GASP
conformance checker against it — all seven mechanical checks (envelope
round-trip, replay, vocabulary, append-only git history, causation integrity,
restore, domain↔ops consistency) must pass on every commit. Try it yourself:

```bash
cargo run --example gasp_emit --features gasp -- /tmp/my-agent
git clone https://github.com/yologdev/gasp && cd gasp/conformance-check
cargo run -q -- /tmp/my-agent
# conformant: all checks passed
```

[`AgentEvent`]: https://docs.rs/yoagent/latest/yoagent/enum.AgentEvent.html
