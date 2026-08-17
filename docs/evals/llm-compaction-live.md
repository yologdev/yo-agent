# LlmCompaction — live evaluation report

Branch `feat/llm-compaction`, PR #119. Numbers below are from real API runs
against Claude Sonnet 5 (session) + Claude Haiku 4.5 (summarizer), reproducible
via `cargo run --example llm_compaction_live --features gasp`.

Every live run in this report was **GASP-recorded**: the harness routes the
agent's event stream through `GaspRecorder`, so each turn is a run in a git-backed
event log (`state/events.jsonl` plus one commit per run) rather than console
output that scrolled past. The transcripts the briefings were judged from are
reconstructable from that record, not from memory.

All three reproduction commands below were run from a **clean clone of this
branch** before publication. The prefix-cache figures reproduced exactly (they
are deterministic — no model in the loop). The live figures are given as ranges
across three runs, because the model is not deterministic and a single decimal
would be false precision: session cache hit rate landed at 75.4%, 76.4% and
79.2%; summarization cost at $0.0149, $0.0171 and $0.0184.

---

## Tweet — option A (the methodology point)

> Shipped an LLM-summarization compaction strategy for my Rust agent crate.
> 502 tests green, 5 review agents, 3 correctness bugs found and fixed.
>
> Then I ran it against a real model for the first time — and it disproved my
> own diagnosis of a bug I'd "found" the day before.
>
> Mocks verify the plumbing. They cannot verify the thing the feature is for.

## Tweet — option B (the concrete bug)

> Wrote a background-summarization compaction strategy. All tests passed.
>
> Then I simulated the actual agent-loop call pattern — feed the result back,
> not the input — and measured: 22 billed summarization requests, 0 splices.
> An absorbing state. It paid full price for zero benefit, forever.
>
> After the fix: 0 requests in that regime (none were owed), and 11 requests /
> 21 splices in the regime where they were.
>
> Every test re-passed pristine input. The bug had the same shape as the tests.

## Tweet — option C (thread opener)

> Six commits of review on one 660-line feature. What the process actually
> caught, with numbers 🧵

---

## Thread (if you want the long version)

**1/**
Built `LlmCompaction` for yoagent: when an agent's context fills up, summarize
the old turns with a cheap model instead of dropping them. Background request,
spliced in later, never blocks the loop.

**2/**
Ran a 5-agent review over it — correctness, tests, comments, silent failures,
type design. Four of five independently flagged the same critical bug. Two
contradicted each other on another. I reproduced every finding before touching
code, which is how the contradiction got settled.

**3/**
The big one: the strategy fingerprinted its pending summary against `compact()`'s
*input*, then returned a rewritten history. The agent loop writes that back, so
the summary was stale on arrival. Discard → respawn → rewrite → forever.

Measured: **22 billed requests, 0 splices** over 25 turns. After the fix, 0
requests in that regime — nothing is owed there, because after a fallback the
compacted history is never long enough to be worth summarizing — and 11
requests / 21 splices at the adjacent turn size where a summary *is* owed.

**4/**
Why no test caught it: every test re-passed the *original* message vector to the
second call. The agent loop never does that. The tests described a contract that
didn't exist — the bug and the tests had the same shape.

**5/**
Two more: the over-budget safety net deleted the briefing it had just paid for
(the summary sits exactly where the fallback starts cutting), while the event
still reported success. And a `keep_first` landing inside a parallel tool-result
block emitted an orphaned `tool_use` — which providers reject outright.

**6/**
Then the part I'd been putting off: running it against a real model.

2 splices, ~$0.015 of summarization on a 12k-token budget, 76% prompt-cache hit
rate across the session. The briefings were genuinely good — exact key formats,
TTLs, memory figures, decisions *with their reasoning*.

**7/**
But I'd claimed a "retention gap": a constraint from turn 1 missing from the
briefing. The live run showed why — that turn was in the retained verbatim head,
which is *excluded from the summarized span by design*. The summarizer never saw
it. Nothing was lost. My diagnosis was wrong.

**8/**
So I added an A/B control and re-ran with the head disabled. Old prompt: 3/4
probes retained their target constraint. New prompt: 4/4. One term moved the
other way.

n=1 per arm against a nondeterministic model. That's not a result. I kept the
prompt change and labelled it a judgement call, not a measured win.

**9/**
What I'd take from it: mock tests verified the state machine, the fingerprint,
the boundary arithmetic, the cost accounting — everything except whether the
feature does its job. That needed one real session and cost 2 cents.

---

## The numbers, for anyone who wants them

| | |
|---|---|
| Tests | 502 passing (193 pre-existing + 18 new unit, plus integration) |
| Review agents | 5, run in parallel; every finding reproduced before fixing |
| Critical bugs | 3 |
| Livelock, before → after | at the turn size that livelocked: 22 requests / 0 splices → **0 / 0** (nothing was owed — after a fallback the compacted history is never long enough to be worth summarizing). At the adjacent size where a summary *is* owed: 9 requests / 21 splices → **11 / 21**. The fix removes wasted spend, not the feature. |
| Live runs | 3 runs, 2 splices each, spans of 5–11 messages |
| Summarization cost | **$0.015–$0.018** per run (both briefings) |
| Prompt-cache hit rate | **74–75% before first splice, 79–82% after, 75–79% session** |
| Cache breaks vs. default | 6 vs 6 at 20k budget / 120 turns; 6 vs 5 at 100k / 600 |

**On that last row:** the strategy does *not* reduce prefix-cache breaks. Both
it and the deterministic default rewrite history only when the budget is
crossed. It buys retention quality and costs tokens. The docs say so explicitly,
and the figures come from a committed harness
(`tests/prefix_cache_harness.rs`), not from a scratch file — an earlier draft of
this work claimed a cache win that measurement did not support.

## Reproducing

```bash
git clone https://github.com/yologdev/yoagent && cd yoagent
git checkout feat/llm-compaction

# prefix-cache measurements (no API key needed)
cargo test --test prefix_cache_harness -- --ignored --nocapture --test-threads=1

# live briefing evaluation (needs ANTHROPIC_API_KEY; ~$0.15/run)
cargo run --example llm_compaction_live --features gasp

# harness plumbing only, no key, no bill
YO_DRY_RUN=1 cargo run --example llm_compaction_live --features gasp
```

## Caveats worth stating if anyone asks

- Three live runs, and the briefing-quality verdict is a judgement from reading
  them, not a benchmark. Someone else reading the same briefings could
  reasonably grade them differently.
- The A/B on the instruction change is underpowered and I'm not claiming it.
- The cache-hit figures come from one session shape and vary a few points
  between runs; they will move much more with turn size, budget, and how often
  compaction fires.
- `keep_first: 0` in the live harness is *not* the crate default — it is set to
  force the briefing to be the only carrier, which is what makes the retention
  probe meaningful.
