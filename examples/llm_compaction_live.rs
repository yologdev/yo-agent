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
//! | `YO_BUDGET` | `12000` (`4000` in dry run) | smaller splices sooner and costs less; larger is more realistic |
//! | `YO_MODEL` | `claude-sonnet-5` | the session's model |
//! | `YO_SUMMARIZER` | `claude-haiku-4-5` | the model that writes briefings — the thing under evaluation |
//! | `YO_SPLICES` | `2` | stop after this many splices |
//! | `YO_MAX_TURNS` | `40` | hard cap, in case the budget is never crossed |
//! | `YO_KEEP_RECENT` | `4` | messages held verbatim; the default 10 needs a production budget |
//! | `YO_REPO` | `/tmp/yoagent-compaction-live` | GASP repo path |

use std::collections::HashSet;
use std::sync::Arc;
use yoagent::context::ContextConfig;
use yoagent::gasp::{GaspRecorder, GoalRef};
use yoagent::llm_compaction::SUMMARY_MARKER;
use yoagent::provider::{ModelConfig, ProviderError, StreamConfig, StreamEvent, StreamProvider};
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
        other => {
            eprintln!("note: no priced preset for '{other}' — the cost column will be blank");
            ModelConfig::anthropic(other, other)
        }
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A coherent engineering task, not filler. The briefing is only worth reading
/// if there were real decisions and constraints to lose — these turns
/// deliberately establish some early and depend on them later, so a summary
/// that drops them shows up as the model contradicting itself.
const TURNS: &[&str] = &[
    "We're designing a distributed rate limiter for an API gateway. Constraint: \
     it must work across 12 stateless nodes with no sticky routing. Start by \
     laying out the options and pick one.",
    "Go with the approach you picked. What's the data structure per key, and \
     what exactly is stored in Redis?",
    "We've decided Redis is a hard dependency and we accept its failure mode. \
     Now: what happens on a Redis partition? Be specific about the tradeoff.",
    "Add support for burst allowances on top of that, without changing the \
     storage format we already settled on.",
    "A customer needs per-endpoint limits, not just per-key. How does that \
     change the key schema?",
    "What's the memory footprint at 2 million active keys with the schema you \
     just described?",
    "Now walk me through the exact Lua script you'd run in Redis, and why it \
     has to be a script rather than pipelined commands.",
    "How do we test the partition behaviour we agreed on earlier, in CI, \
     without a real Redis cluster?",
    "What metrics should this emit, and which one would page someone at 3am?",
    "Someone proposes replacing Redis with an in-memory count plus gossip. \
     Argue against it using the constraints we established.",
    "Write the migration plan from the current naive per-node limiter.",
    "What did we decide about burst allowances, and why that way rather than \
     the alternative?",
    "Summarize every constraint we've agreed on so far, in order.",
    "What's the rollback plan if the Lua script has a bug in production?",
    "How would you shard this if 12 nodes became 200?",
    "Revisit the partition tradeoff: has anything we've added since changed it?",
    "What's the single riskiest part of this design as it now stands?",
    "Draft the section of the design doc covering failure modes.",
    "Which of our decisions would you most expect a reviewer to push back on?",
    "If we had to ship a reduced version in one week, what would you cut?",
];

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
    if !dry_run && std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!(
            "ANTHROPIC_API_KEY is not set — this harness makes real API calls.\n\
             Set YO_DRY_RUN=1 to exercise the harness itself against a stub first."
        );
        std::process::exit(1);
    }

    // The stub's answers are much shorter than a real model's, so the dry run
    // needs a smaller budget to reach a splice in a sensible number of turns.
    let budget: usize = env_or("YO_BUDGET", if dry_run { 4_000 } else { 12_000 });
    let want_splices: usize = env_or("YO_SPLICES", 2);
    let max_turns: usize = env_or("YO_MAX_TURNS", 40);
    let model: String = env_or("YO_MODEL", "claude-sonnet-5".to_string());
    let summarizer: String = env_or("YO_SUMMARIZER", "claude-haiku-4-5".to_string());
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

    let compaction = if dry_run {
        LlmCompaction::from_provider(Arc::new(BulkProvider::briefing()), ModelConfig::mock())
    } else {
        LlmCompaction::from_config(priced(&summarizer))
    }
    // The event carries the cost; the strategy's own `info!` carries the rest.
    .with_event_sender(compact_tx.clone());

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

    for (i, prompt) in TURNS.iter().cycle().take(max_turns).enumerate() {
        print!("turn {:>2} ... ", i + 1);
        use std::io::Write;
        std::io::stdout().flush().ok();

        let (gasp_tx, handle) = recorder.recording_sender(*prompt, None);
        agent.prompt_with_sender(*prompt, gasp_tx).await;
        if let Ok(Ok(Some(id))) = handle.await {
            run_ids.push(id);
        }

        // Drain whatever the turn produced, keeping only what we report on.
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

        for briefing in briefings(agent.messages()) {
            if seen_briefings.insert(briefing.clone()) {
                ordered_briefings.push(briefing);
            }
        }

        println!(
            "{} msgs, ~{} tokens, {} splice(s)",
            agent.messages().len(),
            yoagent::context::total_tokens(agent.messages()),
            compactions
                .iter()
                .filter(|(m, ..)| *m == CompactionMethod::Summarized)
                .count()
        );

        let splices = compactions
            .iter()
            .filter(|(m, ..)| *m == CompactionMethod::Summarized)
            .count();
        if splices >= want_splices {
            println!("\nreached {splices} splice(s); stopping.");
            break;
        }
    }

    drop(agent);
    drop(compact_tx);

    // ---------------------------------------------------------------------
    // The point of the exercise.
    // ---------------------------------------------------------------------
    println!("\n{}", "=".repeat(72));
    println!("BRIEFINGS ({} produced)", ordered_briefings.len());
    println!("{}", "=".repeat(72));
    if ordered_briefings.is_empty() {
        println!(
            "\nNone. The budget was never crossed, or every compaction fell back.\n\
             Lower YO_BUDGET or raise YO_MAX_TURNS, and check the warn-level logs \n\
             for `llm compaction is inert`."
        );
    }
    for (i, briefing) in ordered_briefings.iter().enumerate() {
        println!("\n--- briefing {} ---\n{briefing}", i + 1);
    }

    println!("\n{}", "=".repeat(72));
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
    if total_cost > 0.0 {
        println!("\ntotal summarization cost: ${total_cost:.4}");
    }

    println!("\n{}", "=".repeat(72));
    println!("GASP RECORD");
    println!("{}", "=".repeat(72));
    println!("\n{} run(s) recorded into {repo}", run_ids.len());
    println!("  cat {repo}/state/events.jsonl | jq -c 'select(.kind | startswith(\"model\"))'");
    println!("  git -C {repo} log --oneline");

    println!(
        "\nWhat to look for in the briefings: are the constraints established in \n\
         turns 1-3 (12 stateless nodes, no sticky routing, Redis as a hard \n\
         dependency, the partition tradeoff) still stated correctly after the \n\
         splice? Turns 12, 13 and 16 deliberately ask the model to recall them."
    );
    Ok(())
}
