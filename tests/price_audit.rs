//! Price drift audit — do this crate's hardcoded token prices still match
//! reality?
//!
//! `ModelConfig`'s priced presets are `f64` literals compiled into the crate.
//! When a vendor reprices, they keep billing at the old numbers until someone
//! edits the source and cuts a release, and nothing detects the gap. That is
//! not hypothetical: `claude_sonnet_5` shipped Sonnet **4.6's** rates —
//! $3/$15 against the published $2/$10 — through the whole 0.16.x line,
//! overstating every `cost_usd` for that model by 50%. It was found by someone
//! asking, not by any mechanism.
//!
//! ```text
//! cargo test --test price_audit -- --ignored --nocapture
//! ```
//!
//! Run it before a release, next to the version bump.
//!
//! # Why models.dev, and what it is not
//!
//! [models.dev](https://github.com/anomalyco/models.dev) is an MIT-licensed,
//! schema-validated database served as JSON. Unlike most price aggregators it
//! carries `cache_read` and `cache_write`, which is what this crate needs —
//! cache pricing is the whole subject of the strategy and telemetry work.
//!
//! It is **community-maintained and therefore not authoritative.** A failure
//! here is a *drift alarm* that sends a human to the vendor's own pricing page.
//! If the two disagree, the answer is "go read Anthropic", never "copy
//! models.dev". This test deliberately does not, and should never, update the
//! constants for you.
//!
//! A model missing from the database is reported, not failed — the database
//! lagging a new release is not this crate being wrong.

use yoagent::provider::{CostConfig, ModelConfig};

const DB_URL: &str = "https://models.dev/api.json";

/// A priced preset and where to look when it drifts.
struct Preset {
    /// The constructor, so a failure names the function to edit.
    constructor: &'static str,
    /// models.dev provider key.
    provider: &'static str,
    /// models.dev model key.
    model: &'static str,
    /// The vendor's own page — the authority when the two disagree.
    vendor_page: &'static str,
    cost: CostConfig,
}

fn presets() -> Vec<Preset> {
    let anthropic = "https://platform.claude.com/docs/en/about-claude/pricing";
    let openai = "https://developers.openai.com/api/docs/pricing";
    vec![
        Preset {
            constructor: "ModelConfig::claude_fable_5",
            provider: "anthropic",
            model: "claude-fable-5",
            vendor_page: anthropic,
            cost: ModelConfig::claude_fable_5().cost,
        },
        Preset {
            constructor: "ModelConfig::claude_opus_5",
            provider: "anthropic",
            model: "claude-opus-5",
            vendor_page: anthropic,
            cost: ModelConfig::claude_opus_5().cost,
        },
        Preset {
            constructor: "ModelConfig::claude_opus_4_8",
            provider: "anthropic",
            model: "claude-opus-4-8",
            vendor_page: anthropic,
            cost: ModelConfig::claude_opus_4_8().cost,
        },
        Preset {
            constructor: "ModelConfig::claude_sonnet_5",
            provider: "anthropic",
            model: "claude-sonnet-5",
            vendor_page: anthropic,
            cost: ModelConfig::claude_sonnet_5().cost,
        },
        Preset {
            constructor: "ModelConfig::claude_haiku_4_5",
            provider: "anthropic",
            model: "claude-haiku-4-5",
            vendor_page: anthropic,
            cost: ModelConfig::claude_haiku_4_5().cost,
        },
        Preset {
            constructor: "ModelConfig::gpt_5_5",
            provider: "openai",
            model: "gpt-5.5",
            vendor_page: openai,
            cost: ModelConfig::gpt_5_5().cost,
        },
        Preset {
            // Generic over the model id; these rates are Muse Spark 1.1/1.2.
            // The contributor tier is priced differently and callers on it must
            // override `cost` themselves.
            constructor: "ModelConfig::meta",
            provider: "meta",
            model: "muse-spark-1.2",
            vendor_page: "https://models.dev/",
            cost: ModelConfig::meta("muse-spark-1.2", "Muse Spark 1.2").cost,
        },
    ]
}

/// models.dev omits `cache_write` where a provider does not charge one; absent
/// and zero mean the same thing here.
fn field(cost: &serde_json::Value, name: &str) -> f64 {
    cost.get(name).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

#[tokio::test]
#[ignore = "network: fetches models.dev; run before a release"]
async fn hardcoded_prices_have_not_drifted() {
    let db: serde_json::Value = reqwest::get(DB_URL)
        .await
        .expect("fetch models.dev")
        .json()
        .await
        .expect("parse models.dev");

    let mut drift: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0usize;

    println!(
        "\n{:<30} {:<14} {:>10} {:>12}  ",
        "model", "field", "yoagent", "models.dev"
    );
    println!("{}", "-".repeat(72));

    for p in presets() {
        let Some(cost) = db
            .get(p.provider)
            .and_then(|v| v.get("models"))
            .and_then(|v| v.get(p.model))
            .and_then(|v| v.get("cost"))
        else {
            missing.push(format!(
                "{} ({}/{}) — not in the database; verify by hand at {}",
                p.constructor, p.provider, p.model, p.vendor_page
            ));
            continue;
        };

        for (name, ours) in [
            ("input", p.cost.input_per_million),
            ("output", p.cost.output_per_million),
            ("cache_read", p.cost.cache_read_per_million),
            ("cache_write", p.cost.cache_write_per_million),
        ] {
            let theirs = field(cost, name);
            let same = (ours - theirs).abs() < 1e-9;
            checked += 1;
            println!(
                "{:<30} {:<14} {:>10} {:>12}  {}",
                p.model,
                name,
                ours,
                theirs,
                if same { "ok" } else { "DRIFT" }
            );
            if !same {
                drift.push(format!(
                    "{}: {name} is {ours} in the crate, {theirs} in models.dev — \
                     check {} and edit {} if the vendor agrees",
                    p.model, p.vendor_page, p.constructor
                ));
            }
        }
    }

    println!("{}", "-".repeat(72));
    println!(
        "{checked} fields checked, {} drifted, {} not listed",
        drift.len(),
        missing.len()
    );
    for m in &missing {
        println!("  not listed: {m}");
    }

    assert!(
        drift.is_empty(),
        "\n\nPrice drift detected. models.dev is community-maintained and NOT \
         authoritative — confirm against the vendor page before changing any \
         constant, and never copy models.dev blindly.\n\n{}\n",
        drift.join("\n")
    );
}

/// A preset with all-zero rates reports "pricing unknown", not "free" — the
/// distinction `CostConfig::is_configured` exists to preserve. This one needs
/// no network, so it guards the invariant on every ordinary test run.
#[test]
fn unpriced_presets_report_unknown_rather_than_free() {
    let unpriced = ModelConfig::deepseek("deepseek-v4-flash", "DeepSeek");
    assert!(
        !unpriced.cost.is_configured(),
        "a preset with no rates must report pricing as unknown"
    );

    let priced = ModelConfig::claude_sonnet_5();
    assert!(priced.cost.is_configured());
}
