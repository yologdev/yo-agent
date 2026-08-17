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

## Provider comparison: DeepSeek vs Anthropic

**"Not cached" is `input + cache_write`.** An earlier version of this page
counted only `input`, which understated Anthropic roughly tenfold: Anthropic
books a re-processed prefix to `cache_write`, DeepSeek has no write category, so
the two are not comparable on `input` alone. The figures below supersede that.

| | DeepSeek v4 Flash | Claude Sonnet 5 |
|---|---|---|
| session hit rate | 83.7% | 79.2% |
| steady-state turn | ~98% | ~81% |
| `cache_write` per session | 0 | 92,733 |
| not-cached from compaction | 91.9% | 49.6% |
| turns | 15 | 20 |

- **Session rates are close** despite yoagent placing explicit `cache_control`
  breakpoints for Anthropic and sending nothing to DeepSeek.
- **DeepSeek's cost is almost entirely compaction** (91.9%): populating its cache
  is free, so between rewrites it pays only for genuinely new content.
- **Anthropic pays continuously** — ~3,600–4,600 cache-write tokens per turn at
  1.25× — which is why its steady-state sits near 81% and compaction is only
  half its non-cached total. Those writes buy the cheap reads that follow.
- **Trigger ratio 0.6 → 0.35 was a null result** across eight earlier runs.

Session lengths differ (15 vs 20 turns), so the percentages are comparable and
the absolute totals are not. n=1 per provider at the corrected metric.

### Why these numbers are lower than the replay figures

`docs/concepts/prompt-caching.md` reports 93–96% from 300–2400 turn replays.
These live runs are 15–20 turns, where the arithmetic ceiling is ~88–90%: every
turn's new content is necessarily a miss, so the best achievable rate is about
`(n-1)/(n+1)`. **80% over 15 turns and 95% over 300 turns describe the same
behaviour.** Hit rates are not comparable across session lengths; rewrite counts
and dollars are.

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
