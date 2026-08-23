//! Offline sweep of the compaction headroom policy — the measurement
//! [#150](https://github.com/yologdev/yoagent/issues/150) asks for.
//!
//! Run with: `cargo run --example headroom_sweep`
//!
//! # Why offline
//!
//! Two of the four things that matter are **pure functions of the compaction
//! code** — how often a session compacts, and how much history survives each
//! time — so they can be swept densely for free and deterministically, without
//! a provider in the loop. Only prefix-cache hit rate and briefing splice rate
//! need live runs, and those are worth spending on a shortlist rather than a
//! grid.
//!
//! This drives the real `effective_target_ratio` and `compact_messages`, not a
//! reimplementation of them.
//!
//! # The question
//!
//! `compact_headroom_turns` defaults to `Some(30)`: "leave room for 30 more
//! turns". `effective_target_ratio` computes `budget - turns * growth`, floored
//! at `MIN_HEADROOM_RATIO` (0.15). For any session growing faster than
//! `budget/30` per turn that demand exceeds the budget outright and pins the
//! ratio to the floor.
//!
//! Whether that is *wrong* is the open question. Compacting to 15% means
//! compacting rarely but destructively; compacting to 70% means often but
//! gently. Frequent compaction is the expensive one for prefix caching, since
//! each one invalidates the cached prefix past the head — so "aggressive but
//! rare" may well be the right trade, and the floor defensible.
//!
//! What this measures: for a realistic range of growth rates, how many
//! compactions a session takes and how much history it holds, under each
//! candidate policy.

use yoagent::context::{compact_messages, total_tokens, ContextConfig, MIN_HEADROOM_RATIO};
use yoagent::types::{AgentMessage, Content, Message};

/// One turn of a tool-using agent: an assistant reply plus a bulky tool result.
fn turn(i: usize, bulk_chars: usize) -> Vec<AgentMessage> {
    vec![
        AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: format!("step {i}"),
            }],
            timestamp: i as u64,
        }),
        AgentMessage::Llm(Message::User {
            content: vec![Content::Text {
                text: format!("result {i}: {}", "x".repeat(bulk_chars)),
            }],
            timestamp: i as u64,
        }),
    ]
}

struct Outcome {
    compactions: usize,
    /// Mean tokens held across the session — the retention a model actually
    /// works with, not the peak.
    mean_held: usize,
    /// Tokens held right after each compaction, worst case.
    min_after_compaction: usize,
    ratio_used: f32,
}

/// Drive `turns` turns at a fixed growth rate under one policy, compacting
/// exactly where the agent loop would.
fn simulate(
    budget: usize,
    headroom_turns: Option<usize>,
    bulk_chars: usize,
    turns: usize,
) -> Outcome {
    let config = ContextConfig {
        max_context_tokens: budget,
        system_prompt_tokens: 0,
        keep_recent: 6,
        keep_first: 2,
        compact_headroom_turns: headroom_turns,
        ..Default::default()
    };

    let mut messages: Vec<AgentMessage> = Vec::new();
    let mut compactions = 0usize;
    let mut held_samples: Vec<usize> = Vec::new();
    let mut min_after = usize::MAX;
    let mut ratio_used = config.compact_target_ratio;

    // Growth measured the way the loop measures it: the mean delta per turn.
    let mut last = 0usize;
    let mut growth_total = 0f64;
    let mut growth_samples = 0f64;

    for i in 0..turns {
        messages.extend(turn(i, bulk_chars));
        let now = total_tokens(&messages);
        if i > 0 {
            growth_total += now.saturating_sub(last) as f64;
            growth_samples += 1.0;
        }
        last = now;
        let growth = if growth_samples > 0.0 {
            growth_total / growth_samples
        } else {
            0.0
        };

        // The loop resolves the headroom policy against the budget, then hands
        // the adapted config to the strategy.
        let ratio = config.effective_target_ratio(growth);
        ratio_used = ratio;
        let effective = ContextConfig {
            compact_target_ratio: ratio,
            ..config.clone()
        };

        if total_tokens(&messages) > budget {
            messages = compact_messages(messages, &effective);
            compactions += 1;
            let after = total_tokens(&messages);
            min_after = min_after.min(after);
            last = after;
        }
        held_samples.push(total_tokens(&messages));
    }

    Outcome {
        compactions,
        mean_held: held_samples.iter().sum::<usize>() / held_samples.len().max(1),
        min_after_compaction: if min_after == usize::MAX {
            0
        } else {
            min_after
        },
        ratio_used,
    }
}

fn main() {
    const BUDGET: usize = 96_000; // the crate default: 100K minus the 4K reserve
    const TURNS: usize = 120;

    println!("\nHeadroom policy sweep — #150");
    println!(
        "budget {BUDGET} tokens, {TURNS} turns, keep_first=2 keep_recent=6, \
         MIN_HEADROOM_RATIO={MIN_HEADROOM_RATIO}\n"
    );
    println!(
        "Growth is per-turn token growth. `ratio` is what effective_target_ratio resolved to."
    );
    println!("`mean held` is the context a model actually works with; higher is better retention.");
    println!("`compactions` is how often the prefix cache was invalidated; lower is better.\n");

    // Growth rates spanning light chat through bulky tool output. The crate
    // default pins the floor above ~3.2K/turn at this budget.
    let growths = [
        ("light      ", 800usize),
        ("moderate   ", 4_000),
        ("tool-heavy ", 12_000),
        ("very heavy ", 32_000),
    ];
    let policies = [
        ("Some(30)  [current]", Some(30usize)),
        ("Some(10)", Some(10)),
        ("Some(5)", Some(5)),
        ("Some(3)", Some(3)),
        ("None  [flat 0.7]", None),
    ];

    for (label, bulk) in growths {
        println!("--- growth: {label} (~{} tok/turn) ---", bulk / 4);
        println!(
            "  {:<20} {:>6}  {:>12}  {:>12}  {:>7}",
            "policy", "compac", "mean held", "min after", "ratio"
        );
        for (name, turns_cfg) in policies {
            let o = simulate(BUDGET, turns_cfg, bulk, TURNS);
            println!(
                "  {:<20} {:>6}  {:>12}  {:>12}  {:>7.2}",
                name, o.compactions, o.mean_held, o.min_after_compaction, o.ratio_used
            );
        }
        println!();
    }

    println!("Reading this: a policy is better when it holds more context (mean held) for");
    println!("fewer cache invalidations (compactions). Those pull against each other, so the");
    println!("question is whether Some(30) sits at a defensible point or an extreme one.");
}
