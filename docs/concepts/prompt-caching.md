# Prompt Caching

yoagent automatically optimizes API costs through prompt caching. For providers that support it, stable content (system prompts, tool definitions, conversation history) is cached between turns, giving you up to **90% savings** on input tokens.

## How It Works

In a multi-turn agent loop, each request sends the full context: system prompt + tools + conversation history. Without caching, you pay full price for all of it every turn. With caching, the provider reuses previously processed prefixes.

### Provider Support

| Provider | Caching Type | Savings | Framework Action |
|----------|-------------|---------|-----------------|
| **Anthropic** | Explicit (cache breakpoints) | 90% on hits | ✅ Auto-placed |
| **OpenAI** | Automatic (>1024 tokens) | 50% on hits | None needed |
| **DeepSeek** | Automatic prefix cache | Varies by model | None needed |
| **Google Gemini** | Implicit (automatic) | Varies | None needed |
| **Azure OpenAI** | Automatic (same as OpenAI) | 50% on hits | None needed |
| **Amazon Bedrock** | None (no automatic caching) | — | Not supported |

### What Gets Cached (Anthropic)

yoagent places up to 3 cache breakpoints automatically:

1. **System prompt** — stable across all turns
2. **Tool definitions** — rarely change between turns
3. **Conversation history** — second-to-last message, so the growing prefix is cached

This means on a typical multi-turn conversation, only the latest user message and the new assistant response cost full price.

### DeepSeek

DeepSeek's API manages context caching automatically. yoagent does not send
Anthropic-style `cache_control` markers for DeepSeek; instead, keep stable
prefixes stable (system prompt, tool definitions, and earlier messages) and
monitor DeepSeek's `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
usage fields through `Usage.cache_read` and `Usage.input`.

## Configuration

Caching is **enabled by default** with automatic breakpoint placement. No configuration needed for optimal behavior.

### Disable Explicit Cache Hints

```rust
use yoagent::{CacheConfig, CacheStrategy};

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_cache_config(CacheConfig {
        enabled: false,
        ..Default::default()
    });
```

This disables yoagent-managed cache hints for providers such as Anthropic. It
does not turn off automatic server-side caching for providers such as DeepSeek
or OpenAI.

### Fine-Grained Control

```rust
let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_cache_config(CacheConfig {
        enabled: true,
        strategy: CacheStrategy::Manual {
            cache_system: true,
            cache_tools: true,
            cache_messages: false, // Don't cache conversation history
        },
    });
```

## Monitoring Cache Usage

Every `Usage` struct includes cache statistics:

```rust
// After a response:
let usage = message.usage(); // from assistant message
println!("Cache read: {} tokens", usage.cache_read);
println!("Cache write: {} tokens", usage.cache_write);
println!("Cache hit rate: {:.1}%", usage.cache_hit_rate() * 100.0);
```

- **`cache_read`** — tokens served from cache (cheap)
- **`cache_write`** — tokens written to cache when the provider reports that metric
- **`cache_hit_rate()`** — fraction of input tokens from cache (0.0–1.0)

## Cost Impact

For a typical 10-turn agent conversation with Anthropic Claude:

| Without Caching | With Caching (auto) |
|-----------------|-------------------|
| ~500K input tokens billed at full price | ~50K at full price + ~450K at 10% price |
| **$1.50** (Claude Sonnet 5) | **$0.29** (Claude Sonnet 5) |

That's an **~80% cost reduction** with zero configuration.

## Cache-Stable Compaction

Everything above is about placing cache breakpoints correctly. That only pays off if the bytes *behind* the breakpoint stop changing — and in a long session the thing that changes them is compaction.

A cache hit requires the new request to share a **byte-identical prefix** with the previous one. So the moment compaction rewrites a message that has already been sent, every token from that point onward is uncached and billed at full price. Before 0.15, compaction decided purely on a token budget and never counted that cost.

yoagent 0.15 treats this as a first-class design constraint, in three parts.

### 1. History stops churning

Compaction is a pure function of its input, and idempotent where it can be:

- **Tool-output truncation fits its own budget.** The `[... N lines truncated ...]` marker is charged against `tool_output_max_lines`, so truncating an already-truncated result returns it byte for byte. Previously the marker pushed the result *over* the limit, so the next pass re-truncated it and restated the count — a second full-prefix invalidation on top of the first.
- **Markers carry no drifting state.** The compaction marker's text is constant and generated summaries inherit the timestamp of what they replace, so the same history always compacts to the same bytes.
- **Boundaries snap to turn starts**, which also prevents orphaned `tool_use`/`tool_result` pairs that providers reject outright.

### 2. Output is bounded where it can be bounded well

One global line cap cannot serve every tool, because tools differ in how well they survive being cut:

| output shape | bound | why |
|---|---|---|
| Command output (`bash`, `search`) | `tool_output_max_lines` (200), head+tail, applied on append | first error at the top, summary at the bottom, repetition in the middle |
| File reads (`read_file`) | pages at `DEFAULT_READ_MAX_LINES` (500), exempt from truncation | the middle of a source file is the part that was asked for; paging is lossless and directed |

Applying the cap **on append** rather than retroactively is what keeps the prefix intact: the bytes the provider cached are the bytes that stay. Per-tool budgets live in `tool_output_max_lines_overrides`; `usize::MAX` exempts a tool.

### 3. The compaction target adapts

This is the part a fixed setting cannot do.

Compaction triggers at the budget but reduces to a *target* below it, so the next turn does not immediately cross the budget again and rewrite history. Expressing that target as a fixed fraction has a flaw: **a ratio has no idea how fast the session is growing**, so the headroom it leaves is arbitrary, and the interval between compactions collapses as history accumulates.

`compact_headroom_turns` targets the interval directly:

```text
target        = budget − turns × growth_per_turn
effective     = clamp( min(target / budget, compact_target_ratio), MIN_HEADROOM_RATIO, 1.0 )
```

- `budget` — `max_context_tokens − system_prompt_tokens`
- `turns` — `compact_headroom_turns`, how many more turns you want before the next compaction (default 30)
- `growth_per_turn` — mean tokens added per turn, measured by the agent loop
- `compact_target_ratio` — a **ceiling on retention**: the policy may compact harder than the ratio, never softer
- `MIN_HEADROOM_RATIO` (0.15) — a floor, so runaway growth cannot ask compaction to discard everything

One interpretable knob — *how often am I willing to compact* — that behaves the same at turn 50 and turn 2400, and self-adjusts to workload: an agent producing huge tool output gets compacted harder automatically. It is also self-limiting; on short sessions where compaction is already rare, the derived ratio ties the fixed one and nothing changes.

Mean turns between compactions:

| session | fixed ratio | adaptive target |
|---|---|---|
| 300 turns | 36.1 | 36.1 |
| 1200 turns | 27.8 | **34.3** |
| 2400 turns | 22.6 | **34.7** |

Set `compact_headroom_turns: None` to restore pure ratio behaviour.

#### Choosing a value

Replaying a 2400-turn session at a 1:10 cache-to-input price ratio:

| setting | mean context | cost | compactions |
|---|---|---|---|
| policy off (fixed ratio) | 67,402 | $3.33 | 110 |
| 20 | 61,740 | $3.02 | 86 |
| **30 (default)** | **59,033** | **$2.83** | **70** |
| 40 | 54,069 | $2.58 | 63 |
| 60 and above | 55,283 | $2.61 | 62 |

**The trade is proportional, so pick on memory rather than cost.** Cost per unit of retained context barely moves across the whole range — 4.94 down to 4.73, about 4%. There is no efficient point in this table for the data to single out: lowering the setting sells retained history for money at a nearly fixed exchange rate. Going from 30 to 40 costs 8.7% less and keeps 8.4% less. Choose it on how much history your agent needs to do its job; the cost follows mechanically.

The 4% that is *not* proportional is the genuinely free part, and it comes from compacting less often rather than from retaining less.

**Above roughly 60 the setting stops doing anything.** The derived ratio hits [`MIN_HEADROOM_RATIO`] (0.15) and clamps, so 60, 80, 100 and 140 produce byte-identical runs. To compact harder than that, lower `compact_target_ratio` — it is the ceiling the headroom policy is capped by, not an independent knob.

**Cheaper cache argues for a smaller context, not a larger one.** The same step from a fixed ratio to headroom 60 saves 19.4% at a 1:4 cache-to-input ratio and 26.0% at 1:50. That reads backwards until you notice that once cached tokens are nearly free per token, they dominate the *count* — and you are re-sending them on every request. Vendor ratios in practice run from about 1:10 to 1:50, so the payoff for a smaller working context is larger than a 1:4 assumption would suggest.

[`MIN_HEADROOM_RATIO`]: https://docs.rs/yoagent/latest/yoagent/context/constant.MIN_HEADROOM_RATIO.html

### Measured effect

`tests/context_cache_test.rs` replays a session through compaction turn by turn, renders each turn as a provider body would see it, and measures the shared prefix between consecutive requests — the direct analogue of DeepSeek's `prompt_cache_hit_tokens / prompt_tokens`. The tool mix (bash 41%, edit/write 36%, read 19%, search 4%) and file-size distribution come from 808 archived runs of a production agent built on yoagent. Old and new are measured through the identical code path, on a 128K window.

| session | 0.14.2 hit rate | 0.15.0 hit rate | 0.14.2 rewrites | 0.15.0 rewrites |
|---|---|---|---|---|
| 300 turns | 93.83% | **95.69%** | 34 | **8** |
| 1200 turns | 94.24% | **95.39%** | 169 | **35** |
| 2400 turns | 94.77% | **95.27%** | 415 | **70** |

The rewrite count is the sharper signal: 0.14.2 kept its hit rate up by compacting constantly to a small context — 415 rewrites over 2400 turns — which is precisely the churn that costs money.

Priced as input-token spend across the whole session:

| session | DeepSeek | Anthropic |
|---|---|---|
| 300 turns | $1.6511 → **$1.4985** (−9.2%) | $9.3567 → **$7.9347** (−15.2%) |
| 1200 turns | $6.9307 → **$5.7563** (−16.9%) | $38.7284 → **$30.8415** (−20.4%) |
| 2400 turns | $14.3065 → **$11.2566** (−21.3%) | $78.4390 → **$60.5804** (−22.8%) |

The gap widens with session length, which is the point: the old behaviour degraded as history accumulated, and the new behaviour does not.

### Read a hit rate against its session length, not against 100%

A hit rate is only interpretable next to the number of turns that produced it.
Every turn's *new* content has never been sent before, so it can never hit. With
`n` turns of roughly equal size, the arithmetic ceiling is about

```text
(n − 1) / (n + 1)
```

— roughly 87% at 14 turns, 90% at 19, 95% at 39, and 96% only past 49. The
93–96% figures in the replay table above come from 300–2400 turn sessions, where
the ceiling is above 99% and the measured shortfall is genuinely about rewrites.
The same code measured over a 14-turn session reads ~80%, and that is not a
regression — it is the ceiling.

The practical consequence: **do not compare hit rates across sessions of
different lengths**, and do not treat a published figure as a target for a short
session. Compare rewrite counts, or compare dollars.

### Measured live: DeepSeek vs Anthropic

`examples/llm_compaction_live.rs` runs a real multi-turn session and reports
per-turn cache accounting. **"Not cached" means `input + cache_write`** — both
halves are prompt tokens the provider had to process and bill for. Counting only
`input` is the trap: Anthropic books a re-processed prefix to `cache_write` and
DeepSeek has no write category at all, so an `input`-only metric makes Anthropic
look roughly ten times cheaper than it is.

| | DeepSeek v4 Flash | Claude Sonnet 5 |
|---|---|---|
| session hit rate | **79.0–83.7%** (n=5) | **75.5–79.2%** (n=5) |
| steady-state turn | ~98% | ~81% |
| `cache_write` per session | **0** | 92,733 |
| not-cached tokens from compaction | 91.9% | 49.6% |
| turns | 14–15 | 19–20 |

The session hit rate is measured across five runs per provider; the per-turn
figures come from one run each, because earlier runs did not record per-turn
`cache_write` and could not be re-derived.

**Session rates are close; the mechanisms are not.** Both providers land near
80%, even though yoagent places explicit `cache_control` breakpoints for
Anthropic and sends nothing at all to DeepSeek. What differs is where the cost
falls.

**DeepSeek pays almost nothing between compactions.** Its steady-state turns run
~98%: populating the cache is free, so the only non-cached tokens are the turn's
genuinely new content. Consequently **92% of its non-cached tokens come from the
two turns where compaction rewrote history** — for this provider, compaction is
essentially the whole cache story.

**Anthropic pays a write premium continuously.** Every turn writes ~3,600–4,600
tokens at 1.25× to extend the cached prefix, which holds steady-state turns near
81% rather than 98%. Compaction accounts for only ~50% of its non-cached tokens;
the other half is that ongoing write traffic. The trade is real rather than
wasteful — those writes buy the 99%-cheap reads that follow — but it means the
two providers reach a similar hit rate with materially different bills, which is
the point of the section below.

One null result worth recording: lowering `DEFAULT_TRIGGER_RATIO` from 0.6 to
0.35, on the theory that a slow summarizer needs more wall-clock headroom before
the budget is crossed, changed nothing measurable on either provider. The
condition it was meant to address — repeated deterministic fallbacks because no
summary arrived in time — did not occur in those runs, so the hypothesis is
untested rather than refuted.

### Judge changes in dollars, not hit rate

Hit rate is a fair proxy while you are removing needless rewrites. It stops being one the moment a change also moves context size — **a larger context can raise the hit rate while raising the bill**, because more of what you carry is cached but you are carrying more of it. Every tuning decision above was made on input-token spend for that reason.

### Why there is no cost-benefit gate

A natural next step is to price each compaction and skip the ones that do not pay. Compaction is worth it when

```text
R  >  (I / H) · (P_input − P_cache) / P_cache
```

where `H` is the tokens it stops resending, `I` the prefix it actually invalidates, and `R` the remaining calls. Measured across every compaction event in the replay, `I/H` runs 0.02–0.72 at typical session lengths, putting break-even at **under three remaining calls** on DeepSeek pricing. The gate can only fire in the last turns of a session, and enabling it changed total cost by less than 0.2%.

That is a consequence of the work above rather than an argument against the model: once compaction is byte-stable up to the cut point, the invalidation it has to pay for is a fraction of what it saves. The economics are documented by [bash-agent](https://lloydzhou.github.io/bash-agent/), whose measurements prompted this work.

## Best Practices

1. **Keep system prompts stable** — changing the system prompt between turns invalidates the cache
2. **Don't shuffle tools** — tool order matters for cache prefix matching
3. **Let it work automatically** — the default `CacheStrategy::Auto` is optimal for most use cases
4. **Monitor `cache_hit_rate()`** — if it's consistently low, check if your system prompt or tools are changing unexpectedly
5. **Don't rewrite history yourself** — a custom `CompactionStrategy` or `transform_context` that edits already-sent messages costs the whole prefix from that point on. Append, or cut at a boundary and leave everything before it byte-identical.
6. **Tune `compact_headroom_turns`, not `compact_target_ratio`** — the ratio has to be re-guessed for every workload and session length; the headroom policy adapts on its own. Lower it if compaction is still too frequent, raise it to preserve more history.
7. **Judge in dollars** — see the note above; a higher hit rate does not always mean a smaller bill.
