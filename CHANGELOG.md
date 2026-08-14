# Changelog

All notable changes to `yoagent` are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- **The documented `gasp` extension path was unreachable**
  ([#111](https://github.com/yologdev/yoagent/issues/111)). 0.16.0 re-exported
  the types the `YoAgentState::record_*` methods *take* but not the receiver
  they are called *on*, and `GaspRecorder` kept its state, store and actor
  private — so an application could construct a `Task` and have nothing to
  record it with. The workaround was a direct `yoagent-state` dependency,
  precisely the version-skew hazard the doc comment warned against.

  Both halves are fixed: `ActorRef`, `GitEventStore`, `Node`, `NodeId` and
  `YoAgentState` are now re-exported, and `GaspRecorder` gained
  [`state()`](https://docs.rs/yoagent/latest/yoagent/gasp/struct.GaspRecorder.html#method.state),
  `store()` and `actor()`.

  Prefer the accessors over opening your own store: a GASP repo is
  single-writer behind a 600-second lease, so a second `GitEventStore` on the
  same root collides with the recorder's rather than cooperating.

### Changed

- **Breaking: `AgentEvent`, `StreamEvent`, `StreamDelta`, `StopReason` and
  `ApiProtocol` are now `#[non_exhaustive]`.** These five grow with provider
  and protocol features, so every addition was previously a breaking change —
  the reason yoagent#104 routed around adding a compaction event rather than
  adding one. Downstream `match` arms need a `_ =>` wildcard; new variants are
  additive from here.

  Other enums were deliberately left exhaustive: `Message`, `AgentMessage`,
  `FilterResult`, `ToolDecision` and the config enums are closed shapes, and
  marking them would break a great deal of downstream code for no expected
  growth.

  The `AgentEvent` / `StreamDelta` **wire-tag freeze moved into the crate**
  (`types::wire_tag_freeze`). It relies on exhaustive matching to fail
  compilation until a new variant's serde tag is pinned — a guarantee that no
  longer holds from an integration test once the enum is `#[non_exhaustive]`.
  Verified by adding a probe variant: it still fails to compile.

### Added

- **`SharedState::scoped` — opt-in isolation between sub-agents.** The store
  is a shared flat namespace by design, so this is a view rather than a
  default: keys are transparently prefixed, and `keys()` / `summary()` report
  only that scope. A scoped sub-agent cannot read, overwrite or enumerate
  anything outside it (prefixing is applied on the way in, so a crafted key
  cannot escape), while the parent's unscoped handle still sees everything —
  which is what lets it collect results. Scoping nests, so a view can narrow
  but never widen.

  `summary()` matters most here: it is injected into the sub-agent's system
  prompt, so an unscoped one disclosed every sibling's key names.
- `SubAgentTool::with_scoped_shared_state` — the isolating counterpart to
  `with_shared_state`.

## 0.16.2

### Fixed

- **CI's MSRV job broke on unrelated PRs.** `Cargo.lock` is not committed, so
  every run re-resolves to the newest compatible releases — and when `icu_*`
  2.3.0 (via `reqwest` → `url` → `idna`) raised its own rust-version to 1.88,
  the MSRV job started failing on `main` and every open PR without a commit
  causing it. Setting `resolver = "3"` turns on cargo's MSRV-aware resolution,
  so dependency versions are chosen against our `rust-version` instead of
  ignoring it. No CI or dependency changes; downstream consumers resolve with
  their own resolver and are unaffected.

- **`allowed_paths` was declared but never enforced.** `ReadFileTool` accepted
  an allowlist of directory roots, defaulted it, and then ignored it — callers
  who set it believed reads were sandboxed and they were not. It is now
  enforced, and the same allowlist was added to `WriteFileTool`,
  `EditFileTool`, `ListFilesTool` and `SearchTool`, which previously had no
  path restriction at all.

  Checks run against the **resolved** path via the new
  [`tools::PathSandbox`], so `..` and symlinks cannot escape, and a
  not-yet-created file resolves through its real parent so writes are covered.
  Rejections do not echo the allowed roots (tool results reach the model's
  transcript). Default stays unrestricted — no behaviour change unless the
  allowlist is set.

- **Credentials could reach `Debug` output.** `StreamConfig` derived `Debug`
  with a plaintext `api_key`, and `ModelConfig` derived it with a `headers`
  map that conventionally carries `Authorization` / `x-api-key`. Any `{:?}` in
  a log line or panic message printed them. Both now have manual `Debug` impls
  that redact — `StreamConfig` still distinguishes a set key from an empty
  one, and `ModelConfig` keeps header *names*. `ModelConfig`'s `Serialize` is
  intentionally unchanged so saved configs still round-trip.

### Added

- `BashTool::with_env_allowlist` — pass only named environment variables (plus
  `PATH`, `HOME`, `PWD`) to commands. Commands otherwise inherit the agent's
  full environment, so a model-authored command can read any `*_API_KEY` the
  process holds. Default behaviour unchanged.

### Changed

- Documented plainly that **`BashTool` is not a sandbox**: `deny_patterns` is
  a substring check for typos, bypassed by whitespace, quoting or encoding.
  Isolation belongs in a container/VM or `ToolMiddleware`.

## 0.16.1

### Added

- `SkillSet::load_resilient` / `SkillSet::load_dir_resilient` — keep valid
  skills available while reporting malformed or unreadable `SKILL.md` files
  individually, instead of one bad skill discarding the whole set (#74).
  `load` / `load_dir` stay strict; their first reported error is now
  deterministic (sorted path order) rather than filesystem order.

## 0.16.0

GASP recorder v2 ([#104](https://github.com/yologdev/yoagent/issues/104)): the
recorded log becomes sufficient for offline cost analysis, compaction
inference, and call matching — the substrate for measuring whether compaction
actually hurts an agent, rather than only what it costs.

### Added

- **`tool.called` carries a stable argument fingerprint**
  (`metadata.args_fingerprint`). `input_summary` is a 200-char human summary,
  so calls could not be matched across a log. The fingerprint is normalized
  per tool — path for the file tools, command for `bash`, full args otherwise
  — because `read_file` pages since 0.15 and a re-read of a lost file arrives
  with different offset bytes; hashing raw args would undercount re-fetches
  for exactly the most diagnostic tool. FNV-1a, stable across platforms.
- **`model.finished` carries token usage** (`metadata.usage`: input, output,
  cache_read, cache_write). This makes real cost computable from the log, and
  makes compactions *inferable* — a sharp input-token drop between
  consecutive model calls in one run is the compaction signature — which is
  why no compaction event kind (and no `AgentEvent` break) is needed.
- **`GaspRecorder::with_store` is public** — the extension path for
  applications recording the goal/task/verdict tier into the same ledger:
  open the `GitEventStore` yourself, record goals/tasks on your own
  `YoAgentState`, hand the recorder the same handle. One store, one writer.
  The writer model (one store per agent, `worker_id` per writer, commit at
  run close, push-per-run advice for ephemeral runners) is now documented on
  the method.
- **`gasp` module re-exports the extension-path types** (`Task`,
  `TaskStatus`, `StatePatch`, `EvalResult`, `Goal as GaspGoal`, …) so
  applications need no direct `yoagent-state` dependency kept in version
  lockstep.
- Compaction levels 1 and 2 now emit `tracing::debug!` (tokens/messages
  before → after); previously only level 3 and budget calibration reported.

### Changed

- **Breaking:** `yoagent-state` 0.4 → 0.5. Its types are re-exported from
  `yoagent::gasp`, so this changes public type identity for code that also
  depends on `yoagent-state` directly. Sidecar processes with their own
  `yoagent-state` (separate binaries) are unaffected; 0.5 events are
  wire-compatible both ways with 0.4 logs.

## 0.15.0

### Fixed

- **Compaction rewrote history in place, discarding the provider's prefix
  cache** ([#99](https://github.com/yologdev/yoagent/issues/99)). A cache hit
  requires a byte-identical prefix, so every rewrite of already-sent history
  costs full price for every token from the rewrite point on — automatically on
  DeepSeek, and via `cache_control` on Anthropic. Four separate sources of
  churn are gone:
  - `truncate_text_head_tail` was not idempotent. The `[... N lines truncated
    ...]` marker pushed the result past `tool_output_max_lines`, so the next
    compaction pass re-truncated it and restated the count (`950 lines
    truncated` became `3 lines truncated`) — a second full-prefix
    invalidation, and a marker that lied about how much was dropped. The
    marker is now charged against the line budget, so the result fits exactly
    and re-truncating is a no-op.
  - Compaction reduced to whatever just barely fit, so the next turn crossed
    the budget again and rewrote history *every turn*. The new
    `ContextConfig::compact_target_ratio` (default `0.7`) makes the lossy
    levels reduce to a fraction of the budget while still triggering at the
    full budget. Set it to `1.0` for the old behaviour.
  - Level 3's marker embedded a message count and sat at a fixed position near
    the front, so the cached prefix changed on every pass. The marker text is
    now constant and the count goes to the debug log.
  - Level 2's generated summaries and the Level 3 marker were stamped with
    `now_ms()`, making compaction non-deterministic. They now inherit the
    timestamp of the content they replace.
- **Compaction could orphan tool calls.** Level 2's boundary
  (`len - keep_recent`) could land mid-turn, summarizing away an assistant
  message while its tool results stayed in the kept section; Level 3 and
  `keep_within_budget` could cut the mirror image. Every provider rejects both.
  Boundaries now snap to turn starts.

### Added

- `ContextConfig::compact_headroom_turns` (**default `Some(30)`**) — compact
  hard enough to buy that many more turns at the session's observed growth
  rate, instead of to a fixed fraction of the budget:

  ```text
  target = budget − turns × growth_per_turn
  ```

  A fixed ratio cannot know how fast a session is growing, so the room it
  leaves collapses as history accumulates: at the old default the gap between
  compactions fell from 36 turns to 22 over a 2400-turn session, and the
  rewrites piled up. The adaptive target holds it flat (36 → 35). The effective
  ratio is `min(derived, compact_target_ratio)` floored at
  `MIN_HEADROOM_RATIO`, so the policy can only compact harder than the ratio
  alone, never softer; `None` restores pure ratio behaviour. Growth is measured
  by the agent loop, so a direct `compact_messages` call is unaffected.
- `ContextConfig::truncate_tool_output_on_append` (**default `true`**) applies
  the line cap when a tool result is appended rather than retroactively during
  compaction. Retroactive truncation was the single largest remaining source of
  cache loss: output is sent in full, cached, then rewritten in one sweep once
  the session goes over budget. Capping on the way in also slows context
  growth, so compaction runs less often. The untruncated output is still
  carried by the `AgentEvent` stream.
- `ContextConfig::tool_output_max_lines_overrides` — per-tool line budgets. One
  global number cannot serve every tool: head+tail is a good cut for command
  output (first error at the top, summary at the bottom, repetition in the
  middle) and the *wrong* cut for a file read, where the middle is the part
  that was asked for. The default exempts `read_file`, which bounds itself by
  paging instead. `usize::MAX` disables truncation for a tool.
- `ReadFileTool::max_lines` / `tools::DEFAULT_READ_MAX_LINES` (500) — an
  unqualified read used to return the whole file. Measured against a real Rust
  codebase, the median source file is 1364 lines and 96% exceed 200, so one
  read could spend ~15K tokens; reads are ~19% of tool calls. Paging is the
  right bound for a file — unlike head+tail it is lossless and directed, and
  the header now states the true total and tells the agent to use
  `offset`/`limit`. 500 was chosen by measurement: hit rate is flat between a
  300- and 500-line page and falls off sharply above it.
- `context::truncate_tool_output()` — the single-message helper behind Level 1,
  now public for custom loops and compaction strategies.
- `tests/context_cache_test.rs` replays a session through `compact_messages`
  turn by turn and measures the shared prefix between consecutive requests —
  the direct analogue of DeepSeek's `cache_hit_tokens / input_tokens`. The tool
  mix (bash 41%, edit/write 36%, read 19%, search 4%) and the file-size
  distribution are taken from 808 archived runs of a production agent built on
  yoagent. Over 300 turns on a 128K window:

  | session | 0.14.2 hit | 0.15.0 hit | 0.14.2 rewrites | 0.15.0 rewrites |
  |---|---|---|---|---|
  | 300 turns | 93.83% | **95.69%** | 34 | **8** |
  | 1200 turns | 94.24% | **95.39%** | 169 | **35** |
  | 2400 turns | 94.77% | **95.27%** | 415 | **70** |

  Priced as input-token spend over the whole session — the metric that matters,
  since hit rate alone rewards carrying a larger context:

  | session | DeepSeek 0.14.2 → 0.15.0 | Anthropic 0.14.2 → 0.15.0 |
  |---|---|---|
  | 300 turns | $1.6511 → $1.4985 (**−9.2%**) | $9.3567 → $7.9347 (**−15.2%**) |
  | 1200 turns | $6.9307 → $5.7563 (**−16.9%**) | $38.7284 → $30.8415 (**−20.4%**) |
  | 2400 turns | $14.3065 → $11.2566 (**−21.3%**) | $78.4390 → $60.5804 (**−22.8%**) |

### Changed

- Level 3 now drops the smallest span of middle messages that reaches the
  target instead of always collapsing to `keep_first` + `keep_recent`, so
  compaction destroys only as much history as the budget requires.
- `ContextConfig::tool_output_max_lines` default raised from 50 to 200. It now
  applies on append rather than only under budget pressure, and 50 lines
  (≈23 head + 24 tail) cuts the error list out of most build output.
- **Breaking:** `ContextConfig` gained four public fields
  (`compact_target_ratio`, `compact_headroom_turns`,
  `truncate_tool_output_on_append`, `tool_output_max_lines_overrides`) and
  `ReadFileTool` gained `max_lines`.
  Code that constructs either with an exhaustive struct literal needs
  `..Default::default()`; code using `default()`, `new()`,
  `from_context_window()`, or functional update syntax is unaffected.
- **Breaking (behaviour):** tool output is now capped as it enters the context,
  and an unqualified `read_file` returns 500 lines instead of the whole file.
  To restore the old behaviour set `truncate_tool_output_on_append: false` and
  `ReadFileTool { max_lines: usize::MAX, ..Default::default() }`.

## 0.14.2

### Fixed

- **docs.rs was rendering an incomplete crate.** Without
  `[package.metadata.docs.rs]`, docs.rs built with default features, so the
  `openapi` and `gasp` modules — both advertised in the README — were missing
  from the published API reference. The build now enables all features and
  passes `--cfg docsrs`, and both feature-gated modules carry `doc(cfg(..))`
  availability badges.

### Changed

- **README rewritten for first-time readers.** It now leads with the one
  command that runs a real coding agent against a local model with no API key,
  a Quick Start that actually calls a tool (every snippet compile-checked), a
  diagram of the loop, and a section stating plainly what yoagent does *not* do
  and which crates to use instead. Capabilities moved from a ~40-bullet wall
  into six collapsible groups.
- Surfaced what the README previously omitted: `SharedState`, `InputFilter`,
  `MockProvider`, cost tracking, `set_model`, the OpenCode gateways, all ten
  examples (nine were unreferenced), and the testing story — 456 of 463 tests
  run with no network and no API keys.
- Corrected stale claims: the CLI example is 370 lines, not ~250; the
  OpenAI-compatible implementation covers 12 compat profiles, not "15+"; and
  the module map now covers all of `src/`, including `mcp/`, `openapi/`,
  `session`, `skills`, `shared_state`, `retry`, and `gasp` — roughly 4,000
  lines the old architecture tree left out.
- `docs/introduction.md`, the landing page the `homepage` field points at, was
  a 27-line subset that never mentioned sub-agents, tool middleware, structured
  outputs, skills, session trees, GASP, MCP, or OpenAPI. Rewritten to match the
  README and link the pages that cover each area.
- Crate metadata: filled the three unused crates.io category slots and swapped
  two low-traffic keywords for `anthropic` and `openai`.

### Added

- `CONTRIBUTING.md`, `SECURITY.md`, issue templates (the bug report asks for
  the protocol and model id up front, since most reports are provider-specific)
  and a pull request template — the repository previously had none.
- Loop and sub-agent/shared-state diagrams in `docs/images/`, with light and
  dark variants.
- A **Built with yoagent** section listing the projects that depend on the
  crate, headed by [yoyo-evolve](https://github.com/yologdev/yoyo-evolve), plus
  an invitation to add your own.

## 0.14.1

### Fixed

- **Tool calls with unusable arguments are surfaced instead of executed**
  (#89, Anthropic provider). `content_block_stop` is what turns a tool call's
  streamed `input_json_delta` accumulator into real arguments. When it did not
  — the accumulated text was not valid JSON — the provider substituted an empty
  object and carried on, so the tool executed with default arguments while
  neither the caller nor the model learned the model's actual input had been
  dropped. `list_files` asked for `/etc` would list the process's working
  directory instead.

  A single check after the stream now fails the turn if *any* tool call still
  carries the accumulator, whichever route left it there: unparseable JSON, a
  `content_block_stop` whose body is not JSON, or one with no `index` (which no
  longer silently closes block 0 — a block the event was never about). The error
  names each tool and quotes its input, truncated, since the accumulator can be
  a whole `max_tokens` worth of JSON and it reaches logs, session files, and —
  through `SubAgentTool` — the parent model's context.

  `agent_loop` returns on `StopReason::Error` before extracting tool calls, so
  nothing runs. *Every* tool call in the message is replaced with text, not just
  the unusable one: the turn executes none of them, so any left behind would
  return to the API as a `tool_use` with no `tool_result` and be rejected on the
  next request — breaking the conversation rather than just the turn. Replacing
  in place keeps `content.len()` aligned with the provider's block indices.

  A refusal reported in the same response keeps its `Refusal` stop reason and
  its explanation; the tool-argument note is appended rather than substituted,
  so an in-stream context overflow still matches `Message::is_context_overflow()`
  and its compaction-retry hook.

- **`end_turn` and `stop_sequence` are recognized stop reasons** (Anthropic).
  They previously fell through to the catch-all. Harmless until the catch-all
  gained a warning — at which point every healthy turn logged one.
- **`pause_turn` is reported as incomplete rather than finished** (Anthropic).
  It means the model stopped mid-turn and expects the conversation to be re-sent
  to continue; mapping it to a normal stop handed back a truncated answer as
  though it were complete. This transport cannot resume, so it now says so.
- **A trailing `message_delta` no longer zeroes the output token count**
  (Anthropic). `usage` is optional on the wire, and a defaulted struct made an
  absent block indistinguishable from a reported zero, wiping cost accounting
  and `ContextTracker` calibration for the turn. Such a delta already no longer
  overwrites a stop reason an earlier one established.
- **Structured-output responses keep their error message** (`agent_loop`). The
  rebuild that unwraps a forced tool call preserved the stop reason but dropped
  the explanation, so a failed structured call surfaced as
  `"provider error (no detail)"`.

## 0.14.0

### Added

- **MCP `HttpTransport` speaks the request/response subset of Streamable HTTP**
  (#82). Servers that answer a POST with a `text/event-stream` body — the
  transport introduced in MCP revision 2025-03-26 — previously failed with
  `Response parse error`, because the body was parsed as a single JSON object.

  `HttpTransport` now parses the JSON-RPC payload out of SSE events (joining
  each event's `data:` lines per the SSE spec), advertises
  `Accept: application/json, text/event-stream`, captures the `Mcp-Session-Id`
  the server assigns and replays it on later requests, releases it with a
  best-effort `DELETE` on `McpClient::close()`, and treats `202`/`204` with an
  empty body as an acknowledgement rather than a parse failure.

  Response selection is structural, not "first frame that parses": a frame is
  this request's response only if it carries no `method`, carries a `result` or
  an `error`, and its id matches. That matters because every field of
  `JsonRpcResponse` except `jsonrpc` is optional, so the `notifications/progress`
  and `notifications/message` frames a server emits *ahead of* the result
  deserialize into an empty response — accepting the first frame that parsed
  would have returned the notification and discarded the answer.

  Diagnostics improve alongside: a non-2xx now carries the server's explanation
  instead of a bare status, an empty 2xx that is not an acknowledgement is
  reported rather than dressed up as success, and a `404` on a session-bearing
  request clears the dead session and says it expired instead of failing every
  later call with what looks like a bad URL.

  The body is parsed incrementally: a call returns at the blank-line-terminated
  frame carrying its response rather than at end-of-stream, so a server that
  holds the POST stream open after answering does not block it, and a long
  `tools/call` streaming progress frames returns the moment its result lands. A
  stalled server is bounded by an idle read timeout (120s, reset on every read,
  so a slow-but-progressing call is never cut off) rather than hanging.

  No API-surface changes — `McpTransport` is unchanged, `HttpTransport`'s fields
  were already private, and `McpClient::connect_http` is untouched. Behavior on
  the wire does change: every POST now carries an `Accept` header (plus
  `Mcp-Session-Id` once assigned), `close()` went from a pure no-op to a network
  round-trip, and returning mid-body means an SSE response forgoes connection
  reuse unless the server had already closed the body — a fresh connection, and
  TLS handshake, on the next call to such a server.

  Not covered: the `GET` server→client stream and `Last-Event-ID` resumability.
  `McpTransport` is `send`/`close` only, so a server-initiated message has
  nowhere to be delivered.

  Reported by @markokocic.

## 0.13.3

### Fixed

- **Anthropic messages carry the configured provider** (#81). `AnthropicProvider`
  hardcoded `provider: "anthropic"`, the sole outlier among providers — every
  other one propagates `ModelConfig.provider`. This mis-attributed gateways
  speaking the Anthropic Messages protocol, including yoagent's own
  `ModelConfig::opencode_zen()` preset, which routes Claude model ids over this
  provider under the name `"opencode-zen"`. Falls back to `"anthropic"` when no
  `ModelConfig` is supplied.
- **Truncated SSE streams retry instead of failing hard** (#83). A stream whose
  body ended without an SSE terminator mapped to the non-retryable
  `ProviderError::Other`, so the agent loop gave up immediately even though a
  gateway returning a well-framed body with a truncated payload is usually
  transient. It now maps to `Network`, which the retry policy already covers.
  (A mid-body connection reset is a decode error and surfaces as `Transport`,
  not `StreamEnded` — the two are distinct.)

  Three providers needed work so that reclassification could not retry an
  already-billed response: `anthropic` gains the clean-EOF guard `openai_compat`
  got in #76, armed only by a `message_delta` carrying a terminal `stop_reason`
  and refusing to return a tool call whose arguments never finished streaming;
  `openai_responses` and `azure_openai` now break on `response.incomplete` and
  `response.failed`, which previously fell through and turned one billed
  generation into four.
- **`message_delta` without a `usage` field no longer loses its stop reason**
  (Anthropic). The field was required, so relaying gateways that omit it failed
  the whole parse and silently downgraded `tool_use` / `max_tokens` / `refusal`
  to a plain stop.
- **Dropping a streaming `Agent` no longer orphans its task** (#84). `JoinHandle`
  does not cancel on drop, so the spawned loop kept running — burning tokens,
  holding the tools, and keeping the event channel open so the caller's receiver
  never closed. A `Drop` impl now cancels the token and aborts the handle.

  **Behavior change:** a dropped `Agent` now *kills* its run. Keep the `Agent`
  alive until you have drained the receiver — dropping it early closes the
  channel without an `AgentEnd` event, which a consumer looping on `recv()`
  cannot distinguish from a clean finish. It logs a warning when this happens.
  Tools are dropped rather than recovered, and a tool blocked in synchronous
  code still runs until it yields; both are documented on the impl.

Thanks to @markokocic for reporting #81, #83, and #84.

## 0.13.2

### Added

- **`ModelConfig::claude_opus_5()` preset** (#85). Consumers mapping model
  strings to presets had no constructor for `claude-opus-5` and fell through
  to the generic `ModelConfig::anthropic()`, losing cost tracking and context
  metadata. The preset sets a 1M context window, a 64K default `max_tokens`
  (of the model's 128K ceiling), and $5 / $25 per M input/output with
  $0.50 / $6.25 cache read/write — the same base rates as Opus 4.8. No new
  compat flag was needed: Opus 5 accepts the same adaptive-thinking encoding
  as Opus 4.7/4.8, which `AnthropicCompat::default()` already emits whenever
  a thinking level is set. Note that Opus 5 thinks server-side when a request
  omits `thinking`, so `ThinkingLevel::Off` does not disable thinking on this
  model — see the preset's doc comment.

## 0.13.1

### Fixed

- **openai_compat: DONE-less SSE close after `finish_reason` is now a clean
  EOF** (#76). Providers that close the connection without the
  OpenAI-standard `data: [DONE]` terminator (MiniMax confirmed in the field)
  no longer return `ProviderError::Other("Stream ended")` after the complete
  response already streamed. A `StreamEnded` with no `finish_reason` remains
  an error — genuine truncation still surfaces (network-level drops retry;
  deliberate server closes fail fast).

## 0.13.0

### Added

- **Serializable event stream** — `AgentEvent` and `StreamDelta` now derive
  `Serialize`, `Deserialize`, and `PartialEq`, so external frontends
  (websocket fanout servers, TypeScript clients, JSONL pipes) can consume
  the agent's event stream as JSON. The wire format is internally tagged
  camelCase — `{"type":"toolExecutionEnd","toolCallId":...,"isError":false}`
  — and is a **frozen public contract** guarded by snapshot tests: variant
  tags, field names, and the tagging scheme will not change in minor
  releases. Additive only: no variant, field, or signature changes.

### Changed

- **Message payload serialization normalized to camelCase** — the five
  remaining snake_case fields in serialized messages now match the rest of
  the wire format: `usage.cacheRead`/`cacheWrite`/`totalTokens`,
  `errorMessage`, and `providerMetadata`. Session files and `save_messages`
  blobs written by older versions still load (`serde` aliases accept the old
  names). Files **written** by 0.13 load in older versions *without error*,
  but the renamed fields are silently dropped there: cache/total token
  counts read as 0, and `errorMessage`/`providerMetadata` — including
  Gemini thought signatures — are lost. Don't round-trip session files
  through yoagent < 0.13. The full nested payload shape (message, content
  blocks, usage) is frozen by an exact-JSON snapshot test.
- `serde` minimum version is now `1.0.177` (the release that added
  `rename_all_fields`, July 2023); no practical impact.

## 0.12.0

### Added

- **Meta Model API (Muse Spark)** — `ModelConfig::meta("muse-spark-1.1",
  ...)` preset for Meta's OpenAI-compatible endpoint (US-only public preview
  at launch): 1M context, 128K output, launch pricing pre-configured
  ($1.25/$4.25 per M, $0.15/M cached input). `reasoning_effort` is wired
  (`ThinkingLevel` tunes it; note Meta's server default is `medium`). Key
  resolves from `META_API_KEY`, then Meta's documented `MODEL_API_KEY`. Also
  available in the CLI example via `--provider meta`.

- **GASP bridge** (feature `gasp`) — `gasp::GaspRecorder` records agent runs
  into a [GASP](https://github.com/yologdev/gasp) agent repo via
  `yoagent-state`: append-only `state/events.jsonl` (goal/run/model/tool
  events), one git commit per run (scaffolding committed at init so `git
  clone` restores a complete agent), stale/interrupted runs closed safely,
  events teed to your UI **before** recording (a recording failure never
  blinds the UI; the error surfaces via the returned handle). Redaction hook
  via `with_summarizer` — summaries of tool inputs/outputs are persisted to
  a shareable repo. yoagent is now a **tested** GASP-conformant runtime: CI
  emits a repo and runs the protocol's 7-check suite against a **fresh
  clone** (the actual restore operation). New `gasp_emit` example and docs
  page.

## 0.11.0

### Fixed (pre-release review of the items below)

- `prompt_structured` now surfaces provider failures as
  `StructuredPromptError::Provider` (previously laundered into
  `Parse { raw: "" }`), scans only messages produced by the current call
  (never stale history), and threads the schema per-call so a dropped/timed-
  out future can't leave the agent stuck in schema-forcing mode.
- Bedrock replays `Content::Thinking` blocks (with signatures) on subsequent
  requests — previously captured and then dropped, breaking multi-turn
  thinking + tool use with a ValidationException.
- Anthropic structured outputs disable extended thinking for that request
  (forced tool choice + thinking is an API-level conflict) with a warning.
- Vertex now round-trips Gemini thought signatures on function calls (parity
  with the Gemini API provider).
- `Session::append_new` verifies the history still extends the session path
  and returns `HistoryDiverged` instead of silently corrupting the tree (the
  usual cause: context compaction); `from_jsonl` validates ids and parents
  (duplicates/dangling/cycles rejected); `seek_checkpoint` is latest-wins.
- A panicking `ToolMiddleware` is contained as a denial instead of killing
  the loop task (which stripped the agent of its tools).
- Middleware denials are logged (`tracing::warn!`) so operators see them.
- `ToolMiddleware::before_tool` takes a `ToolCallRequest` context struct
  (extensible without breaking implementors); `StreamConfig` and
  `OutputSchema` are `#[non_exhaustive]` with constructors.

### Added

- **Telemetry** — `tracing` spans: `agent_loop` (model), `llm_stream` per
  turn (tokens in/out/cached + `cost_usd` when pricing is configured), and
  `tool` per execution (name, `is_error`). Bridge to OpenTelemetry
  app-side with `tracing-opentelemetry`; zero overhead with no subscriber.
  New `telemetry` example and docs page.

- **Cross-provider thinking (7/7)** — `thinking_level` is now honored by
  every protocol: Gemini and Vertex send `thinkingConfig` (with thought
  summaries streamed back as `Content::Thinking`), Bedrock sends
  Anthropic-style `additionalModelRequestFields.thinking` (reasoning deltas
  and signatures streamed back), Azure sends Responses-style reasoning
  effort. The "not yet wired" warnings are gone.

- **Session trees** — `Session`: branching conversation history with
  `append`/`seek`/`checkpoint`, fork-preserving edits, `path_messages()` for
  branch resume, and JSONL persistence. The pi-style id/parent_id tree; maps
  to GASP's `transcripts/` tier.

- **Structured outputs** — `Agent::prompt_structured::<T>(text, schema)`
  returns a typed, schema-validated reply. Enforcement is native per
  provider: Anthropic (forced tool call, unwrapped by the loop),
  OpenAI-compatible (`response_format: json_schema, strict`), Gemini
  (`responseSchema`). Providers without support log a warning. New
  `OutputSchema` type on `StreamConfig`/`AgentLoopConfig`; new
  `StructuredPromptError` with the raw text preserved on parse failure.

- **Tool middleware (permissions)** — `ToolMiddleware`, an async
  approve/deny/modify hook gating every tool call, installed via
  `Agent::with_tool_middleware` / `SubAgentTool::with_tool_middleware` /
  `AgentLoopConfig::tool_middleware`. `Deny(reason)` becomes an error tool
  result the LLM can adapt to (the loop continues); `Modify(args)` rewrites
  the call. Empty chain = allow all (no behavior change).

## 0.10.0

The headline change is a **config-first construction API**. You now build an
agent from a single `ModelConfig` — the provider, model id, context window, and
pricing all come from one place, and the API key is resolved from the
provider-conventional environment variable.

### Added

- `Agent::from_config(ModelConfig)` — the new primary constructor. Selects the
  built-in provider for `config.api` and resolves the API key from the
  provider-conventional env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `XAI_API_KEY`, …; see `provider::resolve_api_key`).
- `Agent::from_provider(provider, ModelConfig)` — explicit provider (custom
  `StreamProvider` impls and test doubles). Pair with `ModelConfig::mock()`.
- `Agent::from_config_with(&registry, ModelConfig) -> Result<Agent, AgentBuildError>`
  — resolve the provider from a caller-supplied `ProviderRegistry`.
- `Agent::set_model(ModelConfig)` — switch model mid-session. Re-resolves the
  env key; re-selects the provider only when it was registry-resolved (an
  explicitly-supplied provider is never silently replaced).
- `SubAgentTool::from_config`, `from_config_with`, and `from_provider` mirror
  the above.
- `ModelConfig::mock()` — a throwaway config for tests (use only with
  `from_provider`).
- `AgentBuildError` (exported) — the error type for the fallible
  `from_config_with` path.
- `ProviderRegistry::resolve(&ApiProtocol) -> Option<Arc<dyn StreamProvider>>`
  and `StreamProvider::protocol() -> Option<ApiProtocol>`.
- Automatic env-var API-key resolution and a `with_temperature()` builder
  (from the 0.9.x adoption-funnel work, now the default construction path).

### Deprecated

The following are deprecated since 0.10.0 and will be **removed in 1.0**. They
still work; you'll get a compiler warning pointing at the replacement:

- `Agent::new`, `Agent::with_model`, `Agent::with_model_config`
- `SubAgentTool::new`, `SubAgentTool::with_model`, `SubAgentTool::with_model_config`

### Migration

The old builder made you pair a provider with a matching config by hand and
pass the model id twice. The new one takes a single config:

```rust
// before (0.9): provider and config paired manually; model id passed twice
let agent = Agent::new(OpenAiCompatProvider)
    .with_model_config(ModelConfig::zai("glm-4.7", "GLM 4.7"))
    .with_model("glm-4.7")
    .with_api_key(key);

// after (0.10): provider inferred from config.api; key from ZAI_API_KEY
let agent = Agent::from_config(ModelConfig::zai("glm-4.7", "GLM 4.7"));
```

Per constructor:

| Before | After |
|---|---|
| `Agent::new(AnthropicProvider).with_model("m").with_api_key(k)` | `Agent::from_config(ModelConfig::anthropic("m", "Name")).with_api_key(k)` (drop `with_api_key` to use `ANTHROPIC_API_KEY`) |
| `Agent::new(P).with_model_config(cfg).with_model(cfg.id)` | `Agent::from_config(cfg)` |
| `Agent::new(customProvider).with_model("m")` | `Agent::from_provider(customProvider, cfg)` |
| `Agent::new(MockProvider::text("hi")).with_model("mock")` | `Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock())` |
| `SubAgentTool::new(name, provider).with_model_config(cfg)` | `SubAgentTool::from_config(name, cfg)` or `from_provider(name, provider, cfg)` |

`with_api_key` is **not** deprecated — keep it wherever you want to pass a key
explicitly instead of via the environment.

### Fixed

- Google/Vertex usage no longer double-counts cached tokens.
- `Retry-After` is clamped to `max_delay_ms`.
- Compaction budget calibration subtracts measured overhead instead of scaling
  by a ratio (the old formula could collapse the budget toward zero).
- `session_cost_usd()` returns `None` for unpriced models instead of `0.0`.
- Missing API keys now log a warning naming the env var to set.
