//! Live evaluation harness for [`LlmCompaction`] — the one thing mocks cannot
//! check: whether the briefings are any good.
//!
//! Everything about the strategy is verified against `MockProvider`: the state
//! machine, the fingerprint, the boundary arithmetic, the cost accounting. None
//! of that tells you whether `DEFAULT_INSTRUCTION` produces a handoff a real
//! agent could actually continue from. This runs a real multi-turn session at a
//! deliberately small context budget, forces the strategy to splice, records the
//! whole thing into a GASP repo, and prints every briefing verbatim next to what
//! it cost.
//!
//! Read the briefings before the default prompt ossifies into released docs.
//!
//! ```text
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --example llm_compaction_live --features gasp
//! ```
//!
//! Environment overrides:
//!
//! | var | default | why you'd change it |
//! |---|---|---|
//! | `YO_TRIGGER` | `0.6` | fraction of budget at which summarization starts; lower buys wall-clock headroom |
//! | `YO_BUDGET` | `12000` (`4000` in dry run) | smaller splices sooner and costs less; larger is more realistic |
//! | `YO_MODEL` | `claude-sonnet-5` | the session's model |
//! | `YO_SUMMARIZER` | `claude-haiku-4-5` | the model that writes briefings — the thing under evaluation |
//! | `YO_MAX_TURNS` | `40` | hard cap, in case the budget is never crossed |
//! | `YO_KEEP_RECENT` | `4` | messages held verbatim; the default 10 needs a production budget |
//! | `YO_KEEP_FIRST` | `0` | opening turns held verbatim. **0 on purpose** — at the crate default of 2 the turn that states the constraints never leaves the context, so a retention probe passes without the briefing carrying anything |
//! | `YO_REPO` | `/tmp/yoagent-compaction-live` | GASP repo path |

use std::collections::HashSet;
use std::sync::Arc;
use yoagent::context::ContextConfig;
use yoagent::gasp::{GaspRecorder, GoalRef};
use yoagent::llm_compaction::SUMMARY_MARKER;
use yoagent::provider::{
    CostConfig, ModelConfig, ProviderError, StreamConfig, StreamEvent, StreamProvider,
};
use yoagent::*;

/// Dry-run double (`YO_DRY_RUN=1`): bulky deterministic text, so the harness's
/// own plumbing — GASP recording, event collection, briefing extraction, the
/// cost table — can be exercised without a key or a bill. Verify the harness
/// works, *then* spend money on the thing it measures.
struct BulkProvider {
    text: String,
}

impl BulkProvider {
    fn answer() -> Self {
        Self {
            text: format!(
                "Design note. {}",
                "We weighed the tradeoff and held to the constraint. ".repeat(20)
            ),
        }
    }
    fn briefing() -> Self {
        Self {
            text: "## Goal\n(dry run)\n## State & progress\n(dry run)\n\
                   ## Key decisions & constraints\n(dry run)\n## Open items\n(dry run)"
                .into(),
        }
    }
}

#[async_trait::async_trait]
impl StreamProvider for BulkProvider {
    async fn stream(
        &self,
        _config: StreamConfig,
        _tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        Ok(Message::assistant(
            vec![Content::Text {
                text: self.text.clone(),
            }],
            StopReason::Stop,
            "dry-run",
            "dry-run",
            Usage::default(),
        ))
    }
}

/// Resolve a model id to a **priced** preset where one exists.
///
/// `ModelConfig::anthropic` leaves `CostConfig` at zero, which makes
/// `is_configured()` false and blanks the cost column — the one number this
/// harness exists to surface.
fn priced(id: &str) -> ModelConfig {
    match id {
        "claude-sonnet-5" => ModelConfig::claude_sonnet_5(),
        "claude-haiku-4-5" => ModelConfig::claude_haiku_4_5(),
        "claude-opus-5" => ModelConfig::claude_opus_5(),
        "claude-opus-4-8" => ModelConfig::claude_opus_4_8(),
        "claude-fable-5" => ModelConfig::claude_fable_5(),
        // DeepSeek caches automatically, server-side: this crate sends no
        // `cache_control` on the OpenAI-compat path, and `openai_compat.rs`
        // maps `prompt_cache_hit_tokens` onto `Usage::cache_read`. So a hit
        // rate here measures *their* caching, not yoagent's placement.
        //
        // Rates live here rather than in `ModelConfig::deepseek` because
        // DeepSeek prices by time of day — off-peak is exactly half of peak
        // (peak = 01:00-04:00 and 06:00-10:00 UTC). A crate-level preset would
        // be wrong half the day. These are the **peak** rates, so the cost
        // column over-reports rather than under-reports off-peak runs.
        // DeepSeek has no cache-write category at all: populating its cache
        // is free, which is the asymmetry the provider comparison turns on.
        "deepseek-v4-flash" => deepseek_priced("deepseek-v4-flash", 0.44, 1.32, 0.014),
        "deepseek-v4-pro" => deepseek_priced("deepseek-v4-pro", 1.32, 3.96, 0.044),
        other if other.starts_with("deepseek") => ModelConfig::deepseek(other, other),
        other => {
            eprintln!("note: no priced preset for '{other}' — the cost column will be blank");
            ModelConfig::anthropic(other, other)
        }
    }
}

/// A DeepSeek config carrying peak-window rates. `cache_write` is 0 because
/// DeepSeek has no write category — populating its cache is free.
fn deepseek_priced(id: &str, input: f64, output: f64, cache_read: f64) -> ModelConfig {
    let mut config = ModelConfig::deepseek(id, id);
    config.cost = CostConfig::new(input, output).with_cache_read(cache_read);
    config
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Phase 1 — build a real session and establish constraints worth losing.
///
/// Turn 1 states the *given* conditions (12 stateless nodes, no sticky
/// routing); later turns make decisions in response to them. A briefing that
/// keeps the decisions but drops the conditions is the failure this harness
/// exists to catch.
const ESTABLISH: &[&str] = &[
    "We're designing a distributed rate limiter for an API gateway. Hard \
     constraints: exactly 12 stateless nodes, and the load balancer does NOT \
     do sticky routing, so any node may see any client. Lay out the options \
     and pick one.",
    "Go with the approach you picked. What's the data structure per key, and \
     what exactly is stored in Redis?",
    "We've decided Redis is a hard dependency and we accept its failure mode. \
     What happens on a Redis partition? Be specific about the tradeoff.",
    "Add burst allowances on top of that, without changing the storage format \
     we already settled on.",
    "A customer needs per-endpoint limits, not just per-key. How does that \
     change the key schema?",
    "What's the memory footprint at 2 million active keys with that schema?",
    "Walk me through the exact Lua script, and why it has to be a script \
     rather than pipelined commands.",
    "How do we test the partition behaviour in CI without a real cluster?",
    "What metrics should this emit, and which one would page someone at 3am?",
    "Write the migration plan from the current naive per-node limiter.",
    "Draft the design-doc section covering failure modes.",
    "What's the riskiest part of this design as it now stands?",
];

/// Phase 2 — asked **only after a splice**, so the model is answering from the
/// briefing rather than from the turns it replaced. Each probe names the
/// constraint it is checking for.
const PROBES: &[(&str, &str)] = &[
    (
        "deployment shape",
        "How many nodes are we deploying across, and what did I tell you about \
         sticky routing? Answer from what you know — do not hedge.",
    ),
    (
        "routing assumption",
        "A colleague proposes keeping counters in each node's local memory and \
         relying on the load balancer sending a client back to the same node. \
         Is that compatible with our setup? Why or why not?",
    ),
    (
        "hard dependency",
        "Is Redis optional in this design, and what did we agree about its \
         failure mode?",
    ),
    (
        "storage format",
        "What exactly is stored in a Redis value, and what did we rule out \
         storing there?",
    ),
];

/// Per-turn prompt-cache accounting, so a session-level hit rate can be
/// decomposed rather than just reported.
///
/// A growing conversation has an inherent miss floor: every turn's *new*
/// content has never been seen, so it cannot hit. With `n` turns each adding
/// roughly the same number of tokens, the best achievable rate is about
/// `(n-1)/(n+1)` — 88% at 16 turns, 90% at 19, and 96% only past ~49. Comparing
/// a measured rate against that ceiling separates "the cache is working" from
/// "the session was too short for a high number to be possible".
struct TurnCache {
    turn: usize,
    hit: u64,
    /// Prompt tokens *not* served from cache — `input + cache_write`.
    ///
    /// Both halves count: a provider that re-processes a rewritten prefix bills
    /// it either way. Anthropic books it to `cache_write` (and gets a reusable
    /// entry for the 1.25x premium); DeepSeek has no write category and books it
    /// to `input`. Counting only `input` made compaction look ~10x cheaper on
    /// Anthropic than on DeepSeek when both re-process the same tokens.
    miss: u64,
    /// A compaction rewrote history on this turn, so the next request's prefix
    /// diverges from what the provider cached.
    compacted: bool,
}

/// One probe's answer and which constraints it turned out to carry.
struct ProbeResult {
    label: &'static str,
    answer: String,
    checks: Vec<(&'static str, bool)>,
}

/// Constraint terms a correct post-splice answer should contain.
const RETENTION_CHECKS: &[(&str, &[&str])] = &[
    (
        "12 nodes",
        &["12 node", "12 stateless", "twelve node", " 12 "],
    ),
    ("no sticky routing", &["sticky", "affinity", "any node"]),
    (
        "redis hard dependency",
        &["hard dependency", "not optional", "required"],
    ),
    (
        "counters only",
        &[
            "integer counter",
            "plain integer",
            "no timestamp",
            "counter",
        ],
    ),
];

fn mentions(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_lowercase();
    needles.iter().any(|n| lower.contains(&n.to_lowercase()))
}

/// The most recent assistant reply.
fn last_answer(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Llm(Message::Assistant { content, .. }) => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

/// Session token usage, accumulated per turn from the assistant messages.
#[derive(Default)]
struct Tokens {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl Tokens {
    fn add(&mut self, u: &Usage) {
        self.input += u.input;
        self.output += u.output;
        self.cache_read += u.cache_read;
        self.cache_write += u.cache_write;
    }
    /// Fraction of input served from cache.
    fn cache_hit_rate(&self) -> f64 {
        let total = self.input + self.cache_read + self.cache_write;
        if total == 0 {
            0.0
        } else {
            self.cache_read as f64 / total as f64
        }
    }
}

fn last_usage(messages: &[AgentMessage]) -> Option<Usage> {
    messages.iter().rev().find_map(|m| match m {
        AgentMessage::Llm(Message::Assistant { usage, .. }) => Some(usage.clone()),
        _ => None,
    })
}

/// Briefings currently present in the history. Each splice supersedes the
/// previous one (the new summarized span contains the old summary), so this is
/// polled after every turn rather than read once at the end.
fn briefings(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Llm(Message::User { content, .. }) => {
                content.iter().find_map(|c| match c {
                    Content::Text { text } if text.starts_with(SUMMARY_MARKER) => {
                        Some(text.clone())
                    }
                    _ => None,
                })
            }
            _ => None,
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // INFO surfaces the strategy's own per-compaction line, which reports the
    // cost even when no event sender is wired. `env-filter` is not among the
    // crate's tracing-subscriber features, so this is level-based rather than
    // per-target.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        // stderr, so the log does not interleave with the per-turn progress
        // line on stdout. `2>/dev/null` gives you just the report.
        .with_writer(std::io::stderr)
        .init();

    let dry_run = std::env::var("YO_DRY_RUN").is_ok();
    if dry_run && std::env::var("YOAGENT_API_KEY").is_err() {
        // The stub never authenticates; this only silences the resolver warning.
        std::env::set_var("YOAGENT_API_KEY", "dry-run");
    }
    let model: String = env_or("YO_MODEL", "claude-sonnet-5".to_string());
    let summarizer: String = env_or("YO_SUMMARIZER", "claude-haiku-4-5".to_string());
    if !dry_run {
        for cfg in [priced(&model), priced(&summarizer)] {
            if yoagent::provider::resolve_api_key(&cfg.provider).is_none() {
                eprintln!(
                    "no API key found for provider '{}' (needed by model '{}').\n\
                     Set YO_DRY_RUN=1 to exercise the harness against a stub instead.",
                    cfg.provider, cfg.id
                );
                std::process::exit(1);
            }
        }
    }

    // The stub's answers are much shorter than a real model's, so the dry run
    // needs a smaller budget to reach a splice in a sensible number of turns.
    let budget: usize = env_or("YO_BUDGET", if dry_run { 4_000 } else { 12_000 });
    let max_turns: usize = env_or("YO_MAX_TURNS", 40);
    let trigger_ratio: f32 = env_or("YO_TRIGGER", 0.6);
    let repo: String = env_or("YO_REPO", "/tmp/yoagent-compaction-live".to_string());

    if dry_run {
        println!("*** DRY RUN — stub provider, no API calls, briefings are placeholders ***\n");
    }
    println!(
        "session model : {}",
        if dry_run { "dry-run stub" } else { &model }
    );
    println!(
        "summarizer    : {}   <- the thing under evaluation",
        if dry_run { "dry-run stub" } else { &summarizer }
    );
    println!("budget        : {budget} tokens");
    println!("trigger ratio : {trigger_ratio}");
    println!("gasp repo     : {repo}\n");

    let recorder = GaspRecorder::init(
        &repo,
        "llm-compaction-eval",
        "harness",
        GoalRef::New {
            title: "evaluate LlmCompaction briefings against a real model".into(),
        },
    )
    .await?;

    // `recording_sender` models one GASP *run*, and every `prompt` emits its own
    // AgentStart — so a sender reused across turns tries to open a second run
    // inside the first. One sender (and one run) per turn instead.
    //
    // Compaction events therefore need their own channel: the strategy outlives
    // any single turn, which is the whole point of it.
    let (compact_tx, mut compact_rx) = tokio::sync::mpsc::unbounded_channel();

    // A/B control: `YO_OLD_INSTRUCTION=1` restores the pre-fix instruction so a
    // run can isolate whether the added "constraints you were *given*" clause
    // is what carries them, rather than the summarized span merely containing
    // the text (which `keep_first` decides).
    const OLD_INSTRUCTION_CLAUSE: &str =
        "Summarize the conversation above as a handoff briefing for \
         an agent that will continue this work without access to the original \
         messages. Use exactly these sections:\n\
         ## Goal\nWhat the user is trying to accomplish, verbatim where possible.\n\
         ## State & progress\nWhat has been done, what is currently in flight.\n\
         ## Key decisions & constraints\nDecisions made and why; constraints, \
         preferences, and facts that must not be re-litigated.\n\
         ## Open items\nUnresolved questions and concrete next steps.\n\
         Be dense and factual. Include exact identifiers (paths, names, versions, \
         numbers) — those are the details the next agent cannot reconstruct.";

    let compaction = if dry_run {
        LlmCompaction::from_provider(Arc::new(BulkProvider::briefing()), ModelConfig::mock())
    } else {
        LlmCompaction::from_config(priced(&summarizer))
    }
    // The event carries the cost; the strategy's own `info!` carries the rest.
    .with_event_sender(compact_tx.clone())
    // The trigger is expressed in tokens but what it has to outrun is
    // wall-clock: the summarization must finish inside the
    // `(1 - ratio) x budget` of growth left before the budget is crossed. A
    // slow summarizer against fast-growing turns needs a lower ratio.
    .with_trigger_ratio(trigger_ratio);
    let compaction = if std::env::var("YO_OLD_INSTRUCTION").is_ok() {
        println!("*** CONTROL: pre-fix instruction ***");
        compaction.with_instruction(OLD_INSTRUCTION_CLAUSE)
    } else {
        compaction
    };

    let mut agent = if dry_run {
        Agent::from_provider(BulkProvider::answer(), ModelConfig::mock())
    } else {
        Agent::from_config(priced(&model))
    }
    .with_system_prompt(
        "You are a staff engineer. Answer substantively — a few hundred words — \
             and hold to constraints established earlier in the conversation.",
    )
    .with_context_config(ContextConfig {
        max_context_tokens: budget,
        system_prompt_tokens: 500,
        ..Default::default()
    })
    .with_compaction_strategy(compaction);

    let mut seen_briefings: HashSet<String> = HashSet::new();
    let mut ordered_briefings: Vec<String> = Vec::new();
    let mut compactions: Vec<(CompactionMethod, usize, usize, Option<SummaryStats>)> = Vec::new();
    let mut run_ids: Vec<yoagent::gasp::RunId> = Vec::new();
    let mut tokens = Tokens::default();
    let mut pre_splice = Tokens::default();
    let mut first_splice_turn: Option<usize> = None;
    let mut probe_results: Vec<ProbeResult> = Vec::new();
    let mut per_turn: Vec<TurnCache> = Vec::new();

    let splices = |c: &[(CompactionMethod, usize, usize, Option<SummaryStats>)]| {
        c.iter()
            .filter(|(m, ..)| *m == CompactionMethod::Summarized)
            .count()
    };

    let mut turn = 0usize;
    let mut probe_idx = 0usize;
    loop {
        turn += 1;
        if turn > max_turns {
            println!("\nhit YO_MAX_TURNS={max_turns} before finishing the probes.");
            break;
        }
        // Phase 1 until a splice lands, then phase 2.
        let in_probe_phase = first_splice_turn.is_some();
        let (label, prompt) = if in_probe_phase {
            if probe_idx >= PROBES.len() {
                break;
            }
            let (l, p) = PROBES[probe_idx];
            probe_idx += 1;
            (l, p)
        } else {
            ("establish", ESTABLISH[(turn - 1) % ESTABLISH.len()])
        };

        print!("turn {turn:>2} [{label:<18}] ... ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        let (gasp_tx, handle) = recorder.recording_sender(prompt, None);
        agent.prompt_with_sender(prompt, gasp_tx).await;
        if let Ok(Ok(Some(id))) = handle.await {
            run_ids.push(id);
        }

        let turn_usage = last_usage(agent.messages());
        if let Some(u) = &turn_usage {
            tokens.add(u);
            if first_splice_turn.is_none() {
                pre_splice.add(u);
            }
        }
        let compactions_before = compactions.len();
        while let Ok(event) = compact_rx.try_recv() {
            if let AgentEvent::ContextCompacted {
                method,
                tokens_before,
                tokens_after,
                summary,
                ..
            } = event
            {
                compactions.push((method, tokens_before, tokens_after, summary));
            }
        }
        if let Some(u) = &turn_usage {
            per_turn.push(TurnCache {
                turn,
                hit: u.cache_read,
                miss: u.input + u.cache_write,
                compacted: compactions.len() > compactions_before,
            });
        }
        for briefing in briefings(agent.messages()) {
            if seen_briefings.insert(briefing.clone()) {
                ordered_briefings.push(briefing);
            }
        }
        if first_splice_turn.is_none() && splices(&compactions) > 0 {
            first_splice_turn = Some(turn);
        }

        // A probe's answer is written from the briefing, not the turns it replaced.
        if in_probe_phase {
            let answer = last_answer(agent.messages());
            let checks = RETENTION_CHECKS
                .iter()
                .map(|(name, needles)| (*name, mentions(&answer, needles)))
                .collect();
            probe_results.push(ProbeResult {
                label,
                answer,
                checks,
            });
        }

        println!(
            "{} msgs, ~{} tokens, {} splice(s)",
            agent.messages().len(),
            yoagent::context::total_tokens(agent.messages()),
            splices(&compactions)
        );
    }

    drop(agent);
    drop(compact_tx);

    // ---------------------------------------------------------------------
    // Report
    // ---------------------------------------------------------------------
    println!("\n{}", "=".repeat(72));
    println!("BRIEFINGS ({} produced)", ordered_briefings.len());
    println!("{}", "=".repeat(72));
    if ordered_briefings.is_empty() {
        println!("\nNone — the budget was never crossed, or every compaction fell back.");
    }
    for (i, briefing) in ordered_briefings.iter().enumerate() {
        println!("\n--- briefing {} ---\n{briefing}", i + 1);
    }

    println!("\n{}", "=".repeat(72));
    println!("POST-SPLICE RETENTION");
    println!("{}", "=".repeat(72));
    match first_splice_turn {
        Some(t) => println!("\nfirst splice at turn {t}; probes below ran after it.\n"),
        None => println!("\nno splice occurred — probes did not run.\n"),
    }
    for ProbeResult {
        label,
        answer,
        checks,
    } in &probe_results
    {
        let hits: Vec<&str> = checks
            .iter()
            .filter(|(_, ok)| *ok)
            .map(|(n, _)| *n)
            .collect();
        let miss: Vec<&str> = checks
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(n, _)| *n)
            .collect();
        println!("probe [{label}]");
        println!(
            "  retained: {}",
            if hits.is_empty() {
                "none".into()
            } else {
                hits.join(", ")
            }
        );
        println!(
            "  missing : {}",
            if miss.is_empty() {
                "none".into()
            } else {
                miss.join(", ")
            }
        );
        let excerpt: String = answer.chars().take(220).collect();
        println!("  answer  : {}...\n", excerpt.replace('\n', " "));
    }

    println!("{}", "=".repeat(72));
    println!("COMPACTIONS");
    println!("{}", "=".repeat(72));
    println!(
        "\n{:<14} {:>10} {:>10} {:>8} {:>10} {:>9}",
        "method", "before", "after", "span", "req in/out", "cost"
    );
    let mut total_cost = 0.0;
    for (method, before, after, summary) in &compactions {
        let (span, io, cost) = match summary {
            Some(s) => (
                s.messages_summarized.to_string(),
                format!("{}/{}", s.usage.input, s.usage.output),
                s.cost_usd,
            ),
            None => ("-".into(), "-".into(), None),
        };
        total_cost += cost.unwrap_or(0.0);
        println!(
            "{:<14} {before:>10} {after:>10} {span:>8} {io:>10} {:>9}",
            format!("{method:?}"),
            cost.map(|c| format!("${c:.4}"))
                .unwrap_or_else(|| "-".into()),
        );
    }
    println!("\nsummarization cost: ${total_cost:.4}");

    println!("\n{}", "=".repeat(72));
    println!("SESSION TOKENS & PROMPT CACHE");
    println!("{}", "=".repeat(72));
    println!(
        "\n{:<22} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "phase", "input", "cache_read", "cache_write", "output", "hit rate"
    );
    let post = Tokens {
        input: tokens.input - pre_splice.input,
        output: tokens.output - pre_splice.output,
        cache_read: tokens.cache_read - pre_splice.cache_read,
        cache_write: tokens.cache_write - pre_splice.cache_write,
    };
    for (name, t) in [
        ("before first splice", &pre_splice),
        ("after first splice", &post),
        ("whole session", &tokens),
    ] {
        println!(
            "{name:<22} {:>12} {:>12} {:>12} {:>12} {:>9.1}%",
            t.input,
            t.cache_read,
            t.cache_write,
            t.output,
            t.cache_hit_rate() * 100.0
        );
    }
    println!(
        "\n{:<6} {:>10} {:>12} {:>8}  note",
        "turn", "hit", "not-cached", "rate"
    );
    for t in &per_turn {
        let total = t.hit + t.miss;
        let rate = if total == 0 {
            0.0
        } else {
            t.hit as f64 / total as f64 * 100.0
        };
        println!(
            "{:<6} {:>10} {:>12} {:>7.1}%  {}",
            t.turn,
            t.hit,
            t.miss,
            rate,
            if t.compacted {
                "<- compaction rewrote history"
            } else {
                ""
            }
        );
    }

    // Decompose the misses. Turn 1 can never hit; a turn whose predecessor
    // rewrote history starts from a prefix the provider has not seen.
    let n = per_turn.len().max(1);
    let first_turn_miss: u64 = per_turn.first().map(|t| t.miss).unwrap_or(0);
    // The miss lands on the turn whose request carries the rewritten history —
    // that is the compacting turn itself, not the one after it.
    let post_compaction_miss: u64 = per_turn
        .iter()
        .skip(1)
        .filter(|t| t.compacted)
        .map(|t| t.miss)
        .sum();
    let total_miss: u64 = per_turn.iter().map(|t| t.miss).sum();
    let steady_state_miss = total_miss
        .saturating_sub(first_turn_miss)
        .saturating_sub(post_compaction_miss);
    let ceiling = (n as f64 - 1.0) / (n as f64 + 1.0) * 100.0;

    println!("\nnot-cached decomposition ({total_miss} tokens = input + cache_write):");
    println!("  {first_turn_miss:>10}  turn 1 — nothing to hit yet");
    println!("  {post_compaction_miss:>10}  turns where compaction rewrote history");
    println!("  {steady_state_miss:>10}  steady state — each turn's genuinely new content");
    println!(
        "\nceiling for a {n}-turn session of roughly uniform turns: ~{ceiling:.0}%\n\
         (every turn's new content is necessarily a miss, so the best achievable\n\
          rate rises with turn count: ~88% at 16 turns, ~90% at 19, ~96% past 49.\n\
          Compare the measured rate against this, not against 100%.)"
    );

    println!("\n{}", "=".repeat(72));
    println!("GASP RECORD");
    println!("{}", "=".repeat(72));
    println!("\n{} run(s) recorded into {repo}", run_ids.len());
    println!("  git -C {repo} log --oneline");

    println!(
        "\nWhat to look for in the briefings: are the constraints established in \n\
         turns 1-3 (12 stateless nodes, no sticky routing, Redis as a hard \n\
         dependency, the partition tradeoff) still stated correctly after the \n\
         splice? Turns 12, 13 and 16 deliberately ask the model to recall them."
    );
    Ok(())
}
