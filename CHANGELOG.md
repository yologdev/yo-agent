# Changelog

All notable changes to `yoagent` are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- **Fixed: summarization retries panicked in debug builds, silently.**
  `RetryConfig::delay_for_attempt` documents a 1-indexed attempt and computes
  `attempt - 1`; `llm_compaction.rs` passed the raw `0..=max_retries` loop
  variable, so the first retry underflowed `usize`. `agent_loop.rs` increments
  before calling and was correct — this was confined to compaction.

  It landed on a **detached** task, so nothing surfaced: the summarization
  simply vanished, no briefing arrived, and compaction fell back
  deterministically. That is one of the behaviours
  [#150](https://github.com/yologdev/yoagent/issues/150) was filed about.

  `delay_for_attempt` now uses `saturating_sub`, so a caller that misses the
  1-indexed contract loses the backoff rather than the task.

  `provider_failure_falls_back_deterministically` had been passing *because* of
  this panic: its `FailingProvider` returns a retryable error, and the instant
  death released the in-flight slot inside the test's 100ms window. With
  retries working the first backoff is ~1s, so the test now configures no
  retries — it is about the drop guard, not about backoff timing.

- **`LlmCompaction` says so when briefings keep losing the race**
  ([#150](https://github.com/yologdev/yoagent/issues/150)). A session whose
  compactions all take the deterministic path gets `DefaultCompaction`'s
  retention while still issuing summarization requests it never splices —
  paying input tokens for the span and output tokens for briefings it discards.
  After **five** consecutive fallbacks it warns once, naming the likely cause,
  and a splice that lands both resets the streak and clears the latch. Five, not
  two: a session measured at seven splices in nine compactions — a 78% success
  rate — still contained a run of two, and a two-in-a-row threshold reported it
  as broken for the rest of the run. A signal that cannot retract goes stale the
  moment the configuration improves.

  It is also gated on a request having actually been issued, and on the briefing
  surviving into the result. Without the first it fired in the *inert*
  configuration — where `choose_cut` never finds a split and nothing is ever
  spawned — claiming a cost of zero tokens, in the same session as
  `warn_inert_once` giving the opposite `trigger_ratio` advice. Without the
  second, the "briefing produced, then discarded as too large" path reset the
  streak rather than counting it, so alternating it with a plain miss sawtoothed
  the counter and it could never reach the threshold.

- **A briefing rejected on fingerprint mismatch now reports what it cost.** The
  event carried `summary: None`, so a caller doing cost accounting off
  `SummaryStats` under-counted exactly the failure mode that wastes the most.
  `types.rs` states the contract plainly — "the request was still paid for, so
  the event still reports it" — and the sibling discard branch already honoured
  it.

  Not one wasted request per fallback: `arm` starts no second request while one
  is in flight, so a very slow summarizer costs fewer requests than fallbacks —
  measured at 19 fallbacks against 7 billed requests. The waste is worst in the
  middle regime, where the briefing lands but always just too late.

  The cause is a documentation problem, not a logic one: reusing the loop's
  `ModelConfig` is the obvious call and, for a slow loop model, the worst one —
  `compact` then finds no briefing ready. (It is *not* that the fallback
  invalidates a pending summary; `arm` fingerprints the history `compact` is
  about to return, never the one it received, precisely so that cannot happen.)

  Measured on the `long_horizon` harness at a 30K configured budget, **one run
  each — the model is not deterministic, so read these as the shape of the
  effect, not calibrated figures**. Both rows are the harness's `[compaction #1]`
  line, and the runs differ only in the summarizer:

  | summarizer | first compaction | history retained |
  |---|---|---|
  | the loop's model (Sonnet 5) | `Deterministic` | 3 msgs / 1.7K tokens |
  | a fast model (Haiku 4.5) | `Summarized` | 22 msgs / 16.7K tokens |

  The warning is gated on a request having actually been issued. Without that
  it also fired in the *inert* configuration — where `choose_cut` never finds a
  split and nothing is ever spawned — telling the caller they were paying for
  briefings having spent nothing, in the same session as `warn_inert_once`
  giving the opposite `trigger_ratio` advice.

  **Still open on #150:** `compact_headroom_turns` defaults to `Some(30)`, so
  any session growing faster than ~2.8% of its budget per turn pins
  `effective_target_ratio` to its `MIN_HEADROOM_RATIO` floor of 0.15 — about
  740 tokens/turn at this harness's 26K effective budget, about 2.7K at the 96K
  default. That is the aggression that makes compaction destructive. Changing
  the constant alters compaction for every user and wants measurement across
  growth rates first, not a guess.

## 0.18.0

### Added

- **Fixed: tools with no parameters were uncallable on Anthropic.** A tool
  whose schema takes no arguments has no JSON to stream, and Anthropic still
  emits an `input_json_delta` carrying `""`. `serde_json::from_str("")` fails
  with "EOF while parsing a value", so the `__partial_json` sentinel survived
  `content_block_stop` and the post-stream sweep failed the whole turn with
  *"tool call(s) with unusable arguments, not executed"*. Every no-argument
  tool — `get_status`, `list_files`, `read_log` — was affected.

  An empty accumulator is an empty argument object, not malformed input. The
  decision is now a small pure function, `resolve_tool_arguments`, so it has a
  regression test; genuinely truncated JSON still fails, which is what the
  sentinel exists for.

  Found by `examples/release_smoke.rs` against a live provider, not by the
  suite: `MockProvider` never streams SSE, so nothing exercised the tool-call
  accumulator at all.

- **`examples/release_smoke.rs` — a pre-release check against a live provider.**
  All 618 unit tests use `MockProvider`, which accepts any message sequence
  handed to it. That is structural, not an oversight, and it is where this
  release's worst bug lived: loop detection injected a message between an
  assistant's `tool_use` blocks and their `tool_result`s, which every real
  provider rejects, and the suite stayed green.

  Four checks, chosen for what a mock cannot verify: that loop detection
  produces a transcript the provider accepts *and leaves the agent usable for a
  second prompt*; that the model follows a truncation marker and retrieves
  content head+tail truncation cannot contain; that cost comes from real usage;
  and that a stable system prompt still produces cache reads.

  ```
  ANTHROPIC_API_KEY=... cargo run --example release_smoke
  SMOKE_MODEL=deepseek DEEPSEEK_API_KEY=... cargo run --example release_smoke
  ```

  Verified 6/6 on Sonnet 5 and 5/5 on DeepSeek (its config is unpriced, so the
  cost check reports n/a rather than failing). Exits non-zero on failure.

- **Stash eviction protects caller-owned keys; the default backend no longer
  wedges** ([#144](https://github.com/yologdev/yoagent/issues/144)).

  `MemoryBackend` — what `SharedState::new()` returns, and the documented path
  for `Agent::with_shared_state` — rejected rather than evicted, and nothing
  ever removed `tool-out-*`. At ~300KB per stashed build or grep output a 10MB
  cap holds ~33 results, after which **every** write failed for the rest of the
  run, including the model's own `shared_state set`, with the bytes never
  reclaimed. It now evicts stashed output, oldest first.

  `FileBackend` had the opposite bug: it evicted whatever was oldest, including
  keys a caller had set through the backend's own API. A parent that stored a
  `plan` got `None` back later with no error path anywhere.

  Both now draw the same line. **Stash entries are evictable; caller keys are
  not.** Losing a stash entry degrades a marker to an ordinary "key not found"
  the agent can act on, and the head+tail is still in the transcript — nothing
  regenerates a caller's artifact. When only caller keys remain, the write
  reports capacity instead of destroying data. Every eviction now logs; a
  successful one was previously silent, which mattered because the
  verify-after-store loop only checks the keys written for the current result
  and cannot see that this write evicted an earlier one.

  The line is drawn scope-aware: a sub-agent's stash lands as
  `scope\u{1f}tool-out-…`, so a whole-key prefix test would read it as
  caller-owned and never evict it — re-creating the wedge for every delegating
  run and starving the parent's keys instead.

- **A looping sub-agent no longer reports success to its parent**
  ([#146](https://github.com/yologdev/yoagent/issues/146)). `extract_error`
  matched only `StopReason::Error`, but a loop abort leaves `ToolUse` on the
  last assistant message, and `extract_final_text` scans assistant messages
  only — so it never saw the trailing `[Agent stopped: …]` and fell through to
  `"(sub-agent produced no text output)"`. The parent received
  `is_error: false`. A sub-agent that burned its entire budget looping was
  indistinguishable from one that had nothing to say.

  The marker prefix is now a public constant, `agent_loop::AGENT_STOPPED_PREFIX`,
  so recognising a self-stopped run does not mean matching a magic string.

  Also: `FileBackend::set` returned `Err` for a value already written to disk —
  `evict_to_fit` propagates `read_dir`/`next_entry` errors, both of which run
  *after* the write, and the loop's error arm never records the key so rollback
  could not reach it. It now unlinks before propagating. And a failed rollback
  is logged rather than discarded.

- **Loop detection: accurate name, bounded state, and the coverage it lacked**
  ([#145](https://github.com/yologdev/yoagent/issues/145)).
  `max_identical_tool_calls` is now `max_consecutive_identical_tool_calls`. The
  implementation caps consecutive-run length only, so an alternating
  `[a, b, a, b, …]` loop is never detected; the doc was honest about that, the
  name was not, and a public field is far cheaper to rename before release.

  Deliberately **not** widened to a windowed count — that would regress
  `an_interleaved_different_call_resets_the_counter`, which pins a considered
  trade-off: an agent working through a list calls one tool repeatedly and
  legitimately, and a detector firing on interleaved repeats would be worse than
  none. The limitation is documented on the field and pinned by a test, so it is
  a decision rather than a surprise.

  `steered` held a full clone of every steered signature's arguments — for a
  file-write tool, the entire file body — unbounded, inside the one type whose
  job is bounding runaway resource use. Now a capped list of FNV hashes, sharing
  the one `fnv1a` that `tool_output_key` had inlined.

- **Release-gate fixes found by a whole-release review** (pre-0.18.0). Eight
  commits were each reviewed alone; reviewing them together found defects in
  the interactions no single-PR review could see.

  **Loop detection corrupted the transcript.** The steering nudge and the abort
  message were pushed the moment the verdict came back — which is *before* the
  tools run — so the next request carried `assistant(tool_use)` →
  `user(text)` → `user(tool_result)`. Every provider rejects that shape, and
  the resulting 400 names tool_result blocks rather than loop detection. The
  abort was worse: it returned with the calls unanswered, and since
  `Agent::prompt` keeps those messages, a loop-aborted agent was poisoned —
  every *later* prompt failed too. The nudge is now deferred until the results
  are appended, and the abort synthesizes an `is_error` result per outstanding
  call before stopping. It still does not run the tools.

  This invariant is treated as sacred elsewhere in the crate — `context.rs`
  calls an unanswered call "an orphan every provider rejects" and
  `llm_compaction.rs` has a dedicated test for it. Loop detection was the one
  path that broke it, and no test caught it because `MockProvider` ignores
  message shape and every loop test asserted only on the event stream. There is
  now a transcript-invariant assertion, and reverting the fix fails it.

  **Stash retrieval was re-truncated by the cap it exists to escape.** The
  marker says `shared_state get "tool-out-…"`; the model calls it; the full
  text came back through the same append-path cap that stashed it. 2000 lines
  in, 200 out, plus a fresh stash entry against the cap on every attempt. The
  headline feature of this release did not work through the path the model
  actually takes — only from Rust, which is what the tests asserted.
  `shared_state` now joins `read_file` in `default_tool_output_overrides`.

  **`is_configured()` ignored `context_tier`,** so a config priced only above
  its threshold reported "pricing unknown" and billed every request at $0.

  **GASP recorded a loop abort as `"completed"`.** `LoopDetected` fell into a
  `_ => {}` wildcard, so the outcome stayed whatever the last stop reason
  mapped to. The CHANGELOG claimed the opposite. Now `loop_aborted:{tool}`.

  **`Agent::with_shared_state` shipped undocumented** — its doc block ran into
  `take_tools`'s with no separator, landing the whole thing on a private
  function. The release's headline API would have rendered blank on docs.rs.

  **An `Abort` later in a batch was downgraded to an earlier `Steer`,** and its
  signature never reached `steered`, giving it a free pass the next turn too.
  `LoopVerdict` now derives `Ord` — declaration order is severity order — and
  the loop keeps the worst verdict rather than the first. It is also
  `#[must_use]`: an ignored verdict silently disabled detection while still
  advancing the tracker.

  **The loop-detection text reaching the model carried ~40-space runs**, from
  literals wrapped across source lines without `\` continuations. `rustfmt`
  does not reformat literal contents and clippy has no lint for it.

  **The price audit could pass having compared zero fields.** Its guard derived
  the expected count from the same list it was checking, so an empty
  `presets()` reported "0 compared, 0 drifted" and passed. It now pins a floor
  against the number of priced constructors.

  **Breaking: `CostConfig::new` and `ContextTier::new` take two rates, not
  four.** Cache rates move to `with_cache_read`/`with_cache_write`. Four
  same-typed `f64`s in a row is a transposition hazard, and no vendor publishes
  them in one order — Anthropic lists input/cache-write/cache-read/output,
  OpenAI lists input/cached-input/output. A transposed config still returns
  `true` from `is_configured`, so every downstream guard passes; that is how
  `claude_sonnet_5` billed 50% high for 18 releases. This repo had already made
  the mistake: `examples/llm_compaction_live.rs` silently dropped cache-write
  pricing in a cost-measurement harness. `ContextTier::cache_write_per_million`
  was also unreachable — `new` hardcoded it to 0.0 with no setter on a
  `#[non_exhaustive]` struct.

  **Breaking: `CostConfig::context_tier: Option<ContextTier>` is now
  `context_tiers: Vec<ContextTier>`,** kept sorted by threshold. Vendors
  publish multi-step schedules and models.dev already models this as an array —
  which this release's own audit reads at `/tiers/0/`. Leaving it as one tier
  would have cost a second breaking release, and would have broken the serde
  key as well as the field type, invalidating persisted configs.

  **Breaking: `ExecutionLimits` is `#[non_exhaustive]`** with `with_*` builders,
  closing a break this crate has now taken twice. Also note `ExecutionTracker`
  lost struct-literal construction in this release (it gained three private
  fields) — an undeclared fourth breaking change, recorded here.

- **The `AgentEvent` wire freeze now forces a sample for every variant**
  ([#137](https://github.com/yologdev/yoagent/issues/137)). `EVENT_VARIANT_COUNT`'s
  message claimed it failed when a variant was added without a serialization
  sample. It could not fire for that: the integration test's match carries a
  `_ => "unknown"` arm, forced by `#[non_exhaustive]`, so adding a variant
  requires no edit to that file at all — the count and the sample list are both
  hand-written and a new variant appears in neither. #136 added a 14th variant
  and the suite stayed green, but only because the author added the sample by
  hand and said so in the commit message; nothing would have complained
  otherwise.

  `wire_tag_freeze` in `src/types.rs` now declares the frozen tag **and** a
  sample for every variant from one list, via a `wire_freeze!` macro that emits
  both the wildcard-free match and the sample vector:

  ```rust,ignore
  AgentEvent::LoopDetected { .. } => "loopDetected"
      = AgentEvent::loop_detected("bash", 3, false),
  ```

  Adding a variant is a non-exhaustive-match compile error, and the only way to
  fix it is to add a line here — which supplies the sample in the same breath.
  The specifier is `pat_param`, not `pat`, and that is load-bearing: `pat` would
  accept an or-pattern, letting someone answer the compile error by widening an
  unrelated arm (`TurnStart | NewVariant => "turnStart"`) and leave the new
  variant with no coverage at all.

  Each sample is checked for its frozen tag, a JSON round-trip, and camelCase
  keys **recursively** — a snake_case key at `agentEnd.messages[0].usage`
  reaches clients as surely as a top-level one. Samples are deliberately
  populated rather than defaulted: a round-trip cannot notice a field that
  `#[serde(skip)]` drops if the value it reconstructs is the default anyway.

  Scope, stated plainly because this entry is about a guard that overclaimed:
  this freezes tags, per-variant sample coverage, and round-tripping. It does
  **not** freeze payload shape — a `#[serde(rename)]` on a field still passes.
  Field names are pinned only where `tests/serialization_test.rs` asserts them
  by literal.

  Verified by mutation, since a guard that cannot fire is the bug being fixed:
  adding a 15th variant is a compile error (the integration test stays green on
  the same mutation), an or-pattern bypass is a macro parse error, a tag typo
  fails, dropping `Usage`'s `totalTokens` rename fails on the nested key, and
  `#[serde(skip)]` on `AgentEnd.stats` now fails the round-trip.

  Two related fixes. `ToolResult` reaches the wire as `toolExecutionEnd.result`
  but carried no `rename_all = "camelCase"` — a no-op for
  today's single-word fields, and a silent snake_case leak for the first
  multi-word one. And `ContextCompacted`'s doc block had been concatenated onto
  `LoopDetected` since #136, leaving `ContextCompacted` undocumented.

  In `tests/serialization_test.rs`: `test_context_compacted_wire_shape_is_frozen`
  indexed the sample list positionally (`all_agent_events()[12]`), so inserting
  a variant earlier broke it with a confusing `contextCompacted` tag mismatch
  rather than an edit to this test; it now looks the variant up. The count guard
  is kept under an accurate name and message — it never caught an *added*
  variant, but it does catch a sample being *deleted*, which silently drops that
  variant's payload coverage.

- **Truncated tool output is retrievable instead of lost**
  ([#125](https://github.com/yologdev/yoagent/issues/125)). `truncate_tool_output`
  head-tail-truncates on append and the middle was gone irrecoverably — the full
  text survived only in the event stream, which the *agent* cannot read.

  `Agent::with_shared_state(state)` now stashes the full output and the marker
  names where it went:

  ```
  [... 1847 lines truncated — full output: shared_state get "tool-out-tc_01abc-9f2a..." ...]
  ```

  The `shared_state` tool is registered for the run so the model can act on it.
  `SharedState` was always a parent↔sub-agent medium in Rust, but only
  `SubAgentTool` registered the tool — so the parent's *model* could not reach a
  store its Rust caller could read and write freely.

  **Opt-in.** Without a store attached, truncation behaves exactly as before and
  the marker advertises no retrieval it cannot honour.

  **Known limitation:** the pointer survives Level 1 byte-for-byte, but the
  lossy levels drop whole turns and take the marker with them, while the stash
  entry lives on and keeps consuming cap quota. Pinned by
  `lossy_compaction_drops_the_marker_while_the_stash_survives` so a change to
  that behaviour is a decision rather than a surprise.

  Three constraints shaped it. The stash happens **only on the append path** —
  by the time compaction runs, append-path truncation has already discarded the
  middle, so there is nothing left to store. The key is threaded into marker
  **generation** rather than substituted afterwards: rewriting the rendered text
  would hit every occurrence of the marker's shape, including one that came from
  the tool's own output (ordinary for an agent reading a log or transcript), and
  would silently do nothing when the budget is too small for a marker at all.
  And the key combines the tool call id with a **hash of the output**, because
  `google.rs`/`google_vertex.rs` synthesize ids as a per-response index that
  restarts every turn — id alone would let turn 1's frozen marker resolve to
  turn 5's content.

- **`CostConfig` models context tiers; no preset uses one**
  ([#138](https://github.com/yologdev/yoagent/issues/138)). `cost_usd` applied
  one rate per token category regardless of request size, while some vendors
  price long-context requests higher. It now takes an optional
  `CostConfig::context_tier`, and selects rates by the request's **prompt**
  tokens (`input + cache_read + cache_write`) so a long reply to a short prompt
  stays on the base rate.

  ```rust,ignore
  let cost = CostConfig::new(5.0, 30.0, 0.5, 0.0)
      .with_context_tier(ContextTier::new(272_000, 10.0, 45.0, 1.0));
  ```

  **Every shipped preset stays flat**, `gpt_5_5` included — and getting there
  took three passes, each reversing the last. models.dev records a 272K tier for
  gpt-5.5 at $10/$45/$1. Reading OpenAI's pricing page said no such row existed.
  A closer read said it did, as columns rather than rows. Parsing the page's
  actual markup resolved it: the `>272K input tokens` column group is real, but
  **gpt-5.5 has no row in it** — the $10/$1/$45 triple belongs to
  `gpt-5.6-sol`, a model whose short-context rates are identical to gpt-5.5's.
  The one gpt-5.5 row that does appear in a long-context table,
  `gpt-5.5-cyber`, has `-` in all four long-context cells.

  What settled it was checking models.dev's tier data where a vendor states the
  answer unambiguously: 34 Claude entries carry tiers, including
  `claude-4.6-sonnet` and `claude-opus-4.6` — models Anthropic's own page says
  bill the full 1M window at standard rates. Its tier data is demonstrably wrong
  where it is checkable, so a models.dev tier alone is not grounds to double
  what callers are charged. #132's doctrine, applied: drift alarm, never
  authority.

  `tests/price_audit.rs` now asserts the disputed rates rather than merely
  naming the keys, so a revision upstream fails the audit and the decision gets
  re-made against new evidence instead of silently inheriting this one.

  One caveat for whoever tiers a preset later: prompt size is derived as
  `input + cache_read + cache_write`, which holds only where the provider
  subtracts cached tokens out of `input`. `bedrock.rs` populates neither cache
  field, so a heavily-cached prompt reads small there and would select the cheap
  tier.

  **Breaking: `CostConfig` is now `#[non_exhaustive]`** and gained a fifth
  field, so out-of-crate struct-literal construction no longer compiles. Use
  `CostConfig::new(input, output, cache_read, cache_write)`, optionally with
  `.with_context_tier(..)`; `..Default::default()` no longer rescues a literal.
  Field *reads* and mutation through a binding are unaffected. This is the last
  cheap moment to close it — every future rate category (per-image, per-second
  audio, a second tier) would otherwise be another breaking release.

- **Truncation stash: one key per content block, and `SubAgentTool` can now
  reach it at all** ([#134](https://github.com/yologdev/yoagent/issues/134)).

  A tool result carrying several text blocks got one marker per block, all
  naming the **same** key, while the stash held the blocks joined by `\n` with
  any image between them silently dropped. So a fetch returned every block
  concatenated — never the block whose marker the model had just followed — and
  the value called "full output" was missing content the transcript still
  showed. Keys are now block-qualified and each marker names exactly the block it sits
  in — the suffix is the block's position in the content vector, so
  text/image/text yields `…-b0` and `…-b2`, not `-b0`/`-b1`.

  Making that path reachable exposed a second problem it had been hiding: the
  sub-agent's system prompt embeds a `SharedState` summary, so a second
  invocation would have seen the first run's `tool-out-*` keys — a *different*
  system prompt each time, breaking prefix caching on every sub-agent turn and
  filling the most cache-sensitive text in the request with pointers to
  kilobyte-sized blobs nobody asked about. The cost is a cold prefix on the
  first turn of every subsequent delegation, not on every turn — the prompt is
  built once per invocation. `SharedState::prompt_summary` excludes them; the
  `shared_state` tool's `list` still shows them, so they stay discoverable at
  runtime where changing text costs nothing.

  Separately, the sink wired into `SubAgentTool` in #133 was **dead code**.
  Sub-agents set `context_config: None`, and the loop gates truncation and
  stashing together on that being `Some` — so sub-agents never truncated tool
  output and therefore never stashed. `SubAgentTool::with_context_config` makes
  both reachable. `max_turns` remains the guard on a sub-agent's *length*; this
  is about the size of any single tool result it takes in.

- **Recorded why `LlmCompaction` summarizes standalone rather than in-session**
  ([#127](https://github.com/yologdev/yoagent/issues/127)). Appending the
  summarization instruction to the live session makes the history a prefix-cache
  hit — an obvious-looking saving that measures badly, because the briefing is
  then billed at the *loop* model's output rate. Per compaction: Sonnet 5 gains
  3%, Opus 5 loses 2.4x, Fable 5 loses 4.8x. The idea was motivated by reusing
  the expensive model's cache and the expensive model is where it loses.

  It wins only when the loop model *is* the summarizer (DeepSeek in-session runs
  3.2x cheaper than DeepSeek standalone), which is the configuration cheap-model
  routing exists to avoid. And a background request racing the loop's own turn
  can miss the cache entirely, paying full input on the whole history — 2.6x the
  standalone cost on Sonnet. Best case a few percent, failure case a 2.6x
  overrun. Documented as a no-go in the `llm_compaction` module docs.

- **Loop detection in `ExecutionLimits`**
  ([#126](https://github.com/yologdev/yoagent/issues/126)). The cheapest
  catastrophic failure mode — a model calling one tool with the same arguments
  forever — tripped none of the three existing limits early. They all fire
  eventually, but only after burning the full turn, token and wall-clock budget,
  so the run costs its maximum to discover it achieved nothing.

  `max_consecutive_identical_tool_calls: Option<usize>`, default `Some(3)`, `None` to
  disable. Two escalations, mirroring the house pattern of steering before
  aborting:

  1. **First trip** injects a steering message and continues. A model repeating
     a call is often retrying something transient, and aborting immediately
     would regress that legitimate case.
  2. **A later trip on the same signature** stops the run.

  Both emit the new `AgentEvent::LoopDetected { tool_name, repetitions, aborted }`,
  so a UI can show the intervention and GASP can record why a run ended — an
  audit wants loop aborts distinguishable from turn-limit stops.

  Signatures compare `serde_json::Value`, not serialized text: two calls
  differing only in key order are the same call, and string comparison would
  miss the loop. Counting covers duplicates *within* one batch as well as across
  turns, because `ToolExecutionStrategy` defaults to `Parallel` and a model can
  emit the same call three times in a single message. Any different call resets
  the streak.

  The check runs **before** tool execution, so a stuck model does not also pay
  for the tool run it was never going to learn from.

- **Breaking: `ExecutionLimits` gained `max_consecutive_identical_tool_calls`.** The struct
  is not `#[non_exhaustive]`, so downstream struct-literal construction needs
  the new field or `..Default::default()`. Behaviour changes only for
  pathological sessions, but it *is* on by default.

- **`tests/price_audit.rs` — a drift alarm for the crate's hardcoded prices**
  ([#132](https://github.com/yologdev/yoagent/issues/132)). `ModelConfig`'s
  priced presets are `f64` literals compiled into the crate; when a vendor
  reprices they keep billing at the old numbers until someone edits the source.
  `claude_sonnet_5` shipped Sonnet 4.6's rates across 18 releases, v0.9.0
  through v0.16.5 — 50% high on every `cost_usd` — and nothing detected it. It
  remains uncorrected on the `release/0.16.x` maintenance line.

  ```text
  cargo test --test price_audit -- --ignored --nocapture
  ```

  Diffs all four cost fields of every priced preset against
  [models.dev](https://github.com/anomalyco/models.dev), which unlike most price
  aggregators carries `cache_read`/`cache_write` — the fields this crate needs.
  26 fields compared clean at time of writing, with 2 (`cache_write` on
  `gpt-5.5` and `muse-spark-1.2`) not listed upstream and reported as `—`
  rather than silently compared against 0. Includes `ModelConfig::meta`, whose
  rates had never been checked against any source (they match Muse Spark
  1.1/1.2).

  The audit refuses to go quiet: it fails if a preset vanishes from the database
  (a rename would otherwise drop it out of coverage silently), if fewer fields
  are accounted for than expected (a schema change would otherwise yield a green
  run that compared nothing), or if models.dev carries cost structure the flat
  `CostConfig` cannot express. That last one fires on `gpt_5_5` today —
  models.dev lists a context tier above 272K at double the input rate — recorded
  as a known gap rather than silently certified.

  A failure is an **alarm, not an instruction**: models.dev is
  community-maintained and not authoritative, so the test names the vendor's own
  page and the constructor to edit, and never updates a constant itself. A model
  absent from the database is reported rather than failed.

  Presets now carry the source URL and verification date, and `CostConfig`'s
  docs state plainly that they are a snapshot — with the override path, since
  `cost` is a public field and `CostConfig` is `Deserialize`, so nobody has to
  wait for a release for a negotiated rate.

### Changed

- **`FileBackend` is capped at 10MB** — the same limit as `MemoryBackend`,
  though this backend evicts oldest-first where `MemoryBackend` rejects the
  write. Without a bound, an agent stashing truncated output would grow the
  directory until the disk filled. `FileBackend::with_max_bytes` overrides it.

  A value larger than the cap is **rejected** rather than written and then
  evicted: returning `Ok(())` for a value already deleted would let the loop
  annotate a marker naming a key that never existed. Eviction skips the write
  that triggered it, logs every failed unlink, and warns if it finishes still
  over the cap. **The directory is owned exclusively by the backend** — eviction
  unlinks regular files it finds there.

  A stale key fails to read, which a model handles as an ordinary tool error.

- **Breaking: `AgentLoopConfig` gained `tool_output_sink`.** The struct is not
  `#[non_exhaustive]`, so downstream struct-literal construction needs the new
  field.

## 0.17.0

> The fixes below also shipped on the **0.16.x maintenance line** — the gasp
> extension-path and MSRV fixes in [0.16.3], the gasp id types in [0.16.4], and
> the method-surface closure in [0.16.5] — cut from `v0.16.2` so consumers could
> take them without the breaking changes released here.

### Added

- **`LlmCompaction` — an opt-in compaction strategy that summarizes instead of
  dropping.** `DefaultCompaction`'s lossy tiers discard early decisions and
  constraints outright. `LlmCompaction` sends a standalone summarization request
  and splices a prose handoff briefing in where the dropped span used to be.

  `CompactionStrategy::compact` is synchronous and sits on the hot path before
  every turn, so the request runs in the background: it starts when usage crosses
  `trigger_ratio · budget` and the result is spliced when the budget is actually
  crossed. The loop never stalls on it, and it can never wedge — an unfinished,
  failed, or stale summary falls back to `compact_messages` for that turn.

  Two invariants are load-bearing and were each a bug first, caught by review
  before release. The background request is fingerprinted against the history
  `compact` is about to **return**, not the one it received: the loop writes the
  return value straight back, so anchoring on the input meant the deterministic
  fallback invalidated the summary its own call had just ordered — an absorbing
  state costing one billed request per compaction for a result identical to
  `DefaultCompaction`'s (measured: 22 requests, 0 splices over 25 turns). And an
  over-budget splice compacts only its tail, because the summary sits exactly
  where `level3_drop_middle` starts cutting and the naive whole-history pass
  deleted the briefing while the event still reported `Summarized`.

  This buys **retention quality, and costs tokens**: each summarization pays
  input tokens for the summarized span plus output tokens for the briefing, which
  `DefaultCompaction` never pays. It does **not** reduce prefix-cache breaks —
  6 vs 6 over 120 turns at a 20k budget, 6 vs 5 over 600 turns at 100k, per
  `tests/prefix_cache_harness.rs`. Route it at a cheap model; both paths report
  their own cost on the new event below.

  ```rust,ignore
  let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
      .with_compaction_strategy(LlmCompaction::from_config(
          ModelConfig::anthropic("claude-haiku-4-5", "Haiku 4.5"),
      ));
  ```

  Known limitation: `compact_target_ratio` / `compact_headroom_turns` do not
  apply to this strategy — its result is sized by `with_retain_tail_tokens`
  instead. Consuming the loop's adapted target is follow-up work.

- **`AgentEvent::ContextCompacted`**, with the new `CompactionMethod` enum and
  `SummaryStats` payload. Emitted by `LlmCompaction` on both paths — the spliced
  summary and the deterministic fallback — carrying messages and tokens
  before/after plus, when a request was made, a `summary: Option<SummaryStats>`
  holding the span it bought, its `Usage`, and its dollar cost. That last part is
  the point: it makes an LLM compaction strategy priceable against
  `DefaultCompaction` instead of a guess. One optional payload rather than three
  sibling `Option`s so the cost, the span, and the fact that a request happened
  cannot disagree. `DefaultCompaction` does not emit it (no event channel, and no
  request to price).

- **`tests/prefix_cache_harness.rs`** — an `#[ignore]`d measurement harness that
  drives any `CompactionStrategy` through a simulated session (append a turn,
  compact, feed the result back, as the loop does) and counts the rounds that
  rewrite history. Every prefix-cache figure quoted in the docs comes from it and
  can be regenerated with
  `cargo test --test prefix_cache_harness -- --ignored --nocapture`.

  Deliberately not scoped to `LlmCompaction`: prefix-cache effectiveness has two
  halves, and this covers history stability. Wiring `CacheStrategy` through the
  non-Anthropic providers is the other half, and should extend this harness with
  provider-side breakpoint assertions rather than reimplement the session
  simulation.

- **`examples/llm_compaction_live.rs`** — a live evaluation harness for the one
  thing mocks cannot check: whether the briefings are any good. Runs a real
  multi-turn session at a small context budget, forces splices, records into a
  GASP repo, and prints every briefing verbatim next to what it cost. The turns
  establish constraints early and ask the model to recall them after the splice,
  so a summary that drops them shows up as self-contradiction.

  `YO_DRY_RUN=1` exercises the whole pipeline against a stub — no key, no bill —
  which is how the harness's own wiring was verified before it was handed over.

- `StopReason` is documented as `#[non_exhaustive]`, which it has been since
  0.17.0 — the doc comment still claimed the opposite.

- **`AgentEvent::AgentEnd` carries a `SessionStats` rollup**
  ([#124](https://github.com/yologdev/yoagent/issues/124)). Every per-turn
  number already existed — `Usage` on each assistant message, `tokens_cached`
  on the `llm_stream` span, `cache_read` in the GASP record — but nothing
  summed them, so "what was this run's cache hit rate" meant replaying the
  event stream. Cache-affecting changes were judged by hand-built harnesses
  instead of a number the library reports; #123 and the #119 evaluation both
  had to do exactly that.

  ```rust,ignore
  AgentEvent::AgentEnd { stats, .. } => {
      println!("{:.1}% cached over {} turns", stats.cache_hit_rate() * 100.0, stats.turns);
  }
  ```

  Carries summed `usage`, `turns`, `cost_usd` when rates are configured, and
  `compactions`. Hit rate delegates to `Usage::cache_hit_rate` so there is one
  definition rather than two that drift — and it counts `cache_write` against
  you, because those are prompt tokens the provider processed and billed.

  `usage.total_tokens` is deliberately **not** summed and stays 0. It is a
  per-response provider report and the providers disagree: `anthropic.rs` never
  sets it, `bedrock.rs` computes `input + output` and excludes cache, and the
  rest pass through a payload value that includes cached tokens. Summing it
  would launder that into a session-level figure that reads as authoritative
  and is 0 for every Anthropic run. Derive a total from the four components.

  Scope: a run's own turns. `SubAgentTool` runs its own loop on a private
  channel, so a delegating agent's real spend is higher than this reports.

  `compactions` counts turns where the strategy changed the message count *or*
  the token total, so in-place tool-output truncation is included rather than
  missed by a length check. It is deliberately not split into spliced-summary
  vs deterministic fallback and carries no summarization spend: that detail
  lives on `AgentEvent::ContextCompacted`, and `CompactionStrategy::compact` is
  synchronous with no event channel, so the loop cannot see it. Wire
  `LlmCompaction::with_event_sender` to the same channel to aggregate both.

  The GASP run-close record now carries the same rollup, so an audit can
  compare runs without replaying every `model.finished` entry.

- **`CacheStrategy` is no longer Anthropic-only**
  ([#123](https://github.com/yologdev/yoagent/issues/123)). The 0.16.x line
  engineered byte-stable compaction so cached prefixes survive — and that
  engineering paid off on one protocol out of seven, because `cache_config`
  was read only by `anthropic.rs`. This takes it to two of seven: Azure, the
  OpenAI Responses path and Bedrock all accept cache directives and remain
  unwired.

  Native OpenAI now receives `prompt_cache_key`. OpenAI caches prefixes ≥1024
  tokens automatically, so there are no breakpoints to place; the key routes
  requests to a machine, and the cache itself stays content-addressed. It is
  derived from the **system prompt**, which is the cacheable prefix and which
  `StreamConfig` carries in its own field where compaction cannot reach it.

  Sessions sharing a system prompt therefore share a key. That is the correct
  grouping — they share the cached prefix — but it concentrates traffic, so
  high-volume deployments should set the new `CacheConfig::session_key`
  explicitly to spread load.

  A first implementation also mixed in the first user message, for session
  discrimination, and review caught that it does not survive the crate's own
  compaction: `compact_messages` can drop the head and insert a *constant*
  marker at index 0 (constant on purpose, so the cached prefix stays
  byte-stable). The derived key then drifted mid-session **and** collapsed onto
  one value for every session sharing a system prompt — failing precisely on
  long sessions, which are the ones caching exists for. Session identity is not
  recoverable from a per-request snapshot. `derived_key_survives_compaction_that_rewrites_the_head`
  pins it; the original tests only appended messages, which is the one condition
  under which the broken derivation held.

  `CacheConfig` with every `Manual` flag off now means "no hints" on every
  protocol, matching what Anthropic already did by placing no breakpoints.
  Setting `session_key` on a provider that cannot carry it now warns rather
  than discarding it silently, per the convention stated on
  `StreamConfig::output_schema`.

  Gated on the new `OpenAiCompat::supports_prompt_cache_key`, on only for
  native OpenAI. The field is OpenAI's, and a strict compat server that
  validates unknown keys would reject the request rather than ignore it.

  **Gemini stays implicit-only, on purpose.** Its explicit caching is a
  stateful `CachedContent` resource with its own handle, TTL and billing line
  — not something `CacheStrategy` can express without misrepresenting it as a
  per-request flag. Recorded in the `google` module docs so it isn't
  re-litigated.

  `CacheStrategy`'s rustdoc previously read *"Anthropic-specific; other
  providers handle caching automatically regardless of this setting"* and is
  rewritten around the two shapes — explicit-breakpoint vs key-routed vs
  automatic — with a per-protocol table. Anthropic behaviour is unchanged.

### Changed

- **Breaking: `ToolContext` is `#[non_exhaustive]`** and gained
  `ToolContext::new` plus `with_cancel` / `with_on_update` / `with_on_progress`.

  Its doc has always claimed that "adding fields to `ToolContext` is
  non-breaking". That was false — the struct carried no attribute, so every
  field addition broke downstream struct literals. Making it true cost 20
  literal rewrites across this repo's own tests and examples, which is a fair
  measure of what a future field would have cost everyone else. Tools *receive*
  this type rather than construct it, so implementors are unaffected; only
  code that builds one directly needs the constructor.

- **Breaking: `AgentEvent::AgentEnd` gained a `stats` field** and the variant is
  now `#[non_exhaustive]`. Match with `..`, and construct via
  `AgentEvent::agent_end(messages, stats)` rather than a struct literal. The
  field and every `SessionStats` field carry `#[serde(default)]`, so archived
  event streams written before this release still deserialize — `AgentEvent` is
  a frozen wire format, and a missing-field error would have broken every
  replay consumer.

- **Breaking: `MockResponse` is `#[non_exhaustive]`** and gained
  `TextWithUsage(String, Usage)`, so a test can set a turn's usage. `Text`
  reports `Usage::default()`, which cannot distinguish a rollup that sums from
  one that copies the last turn.

- **Breaking: `CacheConfig` is `#[non_exhaustive]`** and gained `session_key`.
  Construct with `CacheConfig::new()` / `::disabled()` / `::default()` and the
  `with_session_key` / `with_strategy` builders instead of a struct literal.
  `with_strategy` exists because `CacheStrategy::Manual` would otherwise be
  unreachable from outside the crate.

- **Breaking: `OpenAiCompat` is `#[non_exhaustive]`** and gained
  `supports_prompt_cache_key`. It is the crate's most literal instance of a
  growing quirk list — ten flags, one per provider difference — and the
  documented extension point for custom compat servers, so it was the struct
  most likely to be literal-constructed downstream and the one where every
  future flag would break users again. Construct from a preset
  (`OpenAiCompat::openai()`, `::deepseek()`, …) or `Default::default()` and
  adjust fields. The new flag carries `#[serde(default)]`, so a `ModelConfig`
  persisted by an older version deserializes with cache routing **off** until
  re-created from a preset.

- **Breaking: `CacheStrategy` is `#[non_exhaustive]`.** It is a data/policy
  enum whose growth this release was already debating, which is the crate's
  stated test for the attribute (`StopReason` took the same trade; control-flow
  enums like `ToolDecision` deliberately do not). Downstream `match` arms need
  a `_ =>`.

### Fixed

- **The `gasp` method surface is now closed mechanically, not by judgement**
  ([#117](https://github.com/yologdev/yoagent/issues/117)). Three rounds of the
  same bug — [#111] (the receiver), [#115] (the recorded structs' id types),
  [#117] (those structs' *field* types and every `YoAgentState` method's
  argument types) — each found one more tier of `yoagent-state` type that the
  documented extension path needs but could not name. (3) subsumes the others:
  a struct can be constructible without being usable, because a smart
  constructor defaults a field whose type the caller can never name.

  Now also re-exported: `Event`, `EventId`, `Frame`, `FrameId`, `ModelCall`,
  `PatchStatus`, `ProjectSnapshot`, and `ToolCall as GaspToolCall`. The rule is
  enforced by `tests/gasp_test.rs` rather than by inspection, which is what
  makes a fourth round unlikely.

  Forward-ported from the 0.16.x line, where it shipped as [0.16.5]. `main` did
  not have it: the maintenance branch would have been ahead of the development
  branch on this fix, so 0.17.0 would have regressed it.

- **`ModelConfig::claude_sonnet_5` carried Sonnet 4.6's prices**, overstating
  every `cost_usd` for that model by 50%. It shipped as $3/$15 per MTok
  ($3.75 write, $0.30 read) against the published $2/$10 ($2.50 write, $0.20
  read) — the introductory rate that has since become standard. Callers using
  `CostConfig::cost_usd`, `Agent::session_cost_usd`, or the `cost_usd` field on
  the `llm_stream` tracing span will see Sonnet 5 figures drop by a third.
  Fable 5, Opus 5, Opus 4.8 and Haiku 4.5 were checked against the same table
  and were correct.

- **The `gasp` extension path could not construct its arguments**
  ([#115](https://github.com/yologdev/yoagent/issues/115)). The recorded
  structs were re-exported without the id types their fields require, so
  `StatePatch`, `EvalResult` and `Decision` were nameable and unbuildable.
  `Task` worked only because `TaskId` was already in the list — which is why
  the gap survived inspection, and why the 0.16.3 example, which constructs a
  `Task`, compiled while its siblings did not.

  Now re-exported: `PatchId`, `EvalId`, `DecisionId`, `HypothesisId`,
  `ObservationId`, `ArtifactRef`, `ExpectedEffect`, `Precondition`,
  `ProjectRef`, `StateOp` — found by auditing every re-exported struct's
  fields rather than only the three in the report. A test now constructs every
  re-exported struct from `yoagent::gasp` alone, so a struct that is nameable
  but unbuildable fails to compile.

### Changed

- CI runs on `release/**` as well as `main`. The 0.16.x maintenance line had
  no CI, so 0.16.3 shipped verified only locally.

[0.16.3]: https://github.com/yologdev/yoagent/releases/tag/v0.16.3
[0.16.4]: https://github.com/yologdev/yoagent/releases/tag/v0.16.4
[0.16.5]: https://github.com/yologdev/yoagent/releases/tag/v0.16.5
[#111]: https://github.com/yologdev/yoagent/issues/111
[#115]: https://github.com/yologdev/yoagent/issues/115
[#117]: https://github.com/yologdev/yoagent/issues/117


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
