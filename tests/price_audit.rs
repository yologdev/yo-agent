//! Price drift audit — do this crate's hardcoded token prices still match
//! reality?
//!
//! `ModelConfig`'s priced presets are `f64` literals compiled into the crate.
//! When a vendor reprices, they keep billing at the old numbers until someone
//! edits the source and cuts a release, and nothing detects the gap. That is
//! not hypothetical: `claude_sonnet_5` carried Sonnet **4.6's** rates —
//! $3/$15 against the published $2/$10 — from **v0.9.0 through v0.16.5**,
//! 18 tagged releases, overstating every `cost_usd` for that model by 50%.
//! It was found by someone asking, not by any mechanism.
//!
//! It is **still uncorrected on the `release/0.16.x` maintenance line**, which
//! does not carry this test — a 0.16.6 would re-ship it.
//!
//! ```text
//! cargo test --test price_audit -- --ignored --nocapture
//! ```
//!
//! Run it before a release, at step 3 of the release checklist.
//!
//! # Why models.dev, and what it is not
//!
//! [models.dev](https://github.com/anomalyco/models.dev) is an MIT-licensed,
//! schema-validated database served as JSON. It carries `cache_read` and
//! `cache_write`, which this crate needs and which many aggregators omit —
//! cache pricing is the subject of the compaction and telemetry work.
//!
//! It is **community-maintained and therefore not authoritative.** A failure
//! here is a *drift alarm* that sends a human to the vendor's own pricing page.
//! If the two disagree, the answer is "go read Anthropic", never "copy
//! models.dev". This test deliberately does not, and should never, update the
//! constants for you.
//!
//! # How this instrument avoids going quiet
//!
//! An audit that reports "clean" because it compared nothing is worse than no
//! audit — it converts an open question into false assurance. Review of the
//! first version found six distinct ways to reach a green pass having checked
//! zero fields: a renamed provider key, a restructured envelope, a renamed
//! `models`/`cost` level, an HTTP error with a JSON body, `null`, or `{}`.
//! Each is plausible for a community database with ~190 providers.
//!
//! So the pass condition is not "no drift found". It is:
//!
//! - **every field is accounted for** — compared against a number, or recorded
//!   as absent upstream. A count short of the expected total fails.
//! - **every preset is found** unless it carries an explicit, dated
//!   [`Preset::absent_upstream`] note. A rename silently dropping a model out
//!   of coverage is the failure this catches.
//! - **absent is not zero.** A missing `cache_write` prints `—`, never `0`, so
//!   the table never claims a comparison it did not make.
//! - **unknown cost keys fail.** models.dev carries tiered pricing that flat
//!   [`CostConfig`] cannot express; ignoring the structure would certify a
//!   preset that is knowably wrong above the tier boundary.
//! - **HTTP status is checked**, and a non-JSON body reports the status,
//!   content type and first bytes rather than a bare parse error.

use yoagent::provider::{CostConfig, ModelConfig};

const DB_URL: &str = "https://models.dev/api.json";

/// Cost keys this audit understands. Anything else means models.dev is
/// describing pricing structure the flat [`CostConfig`] cannot represent.
const KNOWN_COST_KEYS: [&str; 4] = ["input", "output", "cache_read", "cache_write"];

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
    /// Set only when models.dev genuinely lacks this model, with the date it
    /// was checked by hand. `None` means "must be present": absence is a
    /// failure, because a rename dropping a model out of coverage looks
    /// exactly like a database that has not caught up yet.
    absent_upstream: Option<&'static str>,
    /// Set when models.dev lists cost structure a flat `CostConfig` cannot
    /// express. Records *exactly* what was acknowledged, so the note cannot
    /// become a blanket amnesty for whatever appears later.
    flat_rate_gap: Option<FlatRateGap>,
}

/// A recorded, verified gap between what models.dev carries and what a flat
/// `CostConfig` can express.
///
/// Prose alone made this a waiver: the audit checked only that a note existed,
/// so a *new* unknown key — a second tier, an audio rate, a reasoning rate —
/// folded silently into the old note and the preset stayed green forever. The
/// reverse went unnoticed too: if the upstream claim vanished, or its key was
/// renamed (indistinguishable from removal), nothing said the recorded decision
/// had gone stale.
///
/// So the acknowledgement names the keys and the rates it was made against, and
/// both directions are assertions.
struct FlatRateGap {
    /// Cost keys this gap covers. Any *other* unknown key is drift.
    keys: &'static [&'static str],
    /// Upstream rates the decision was reasoned against, as
    /// `(json_pointer, value)`. A revision makes the recorded reasoning stale.
    rates: &'static [(&'static str, f64)],
    why: &'static str,
}

fn presets() -> Vec<Preset> {
    let anthropic = "https://platform.claude.com/docs/en/about-claude/pricing";
    let openai = "https://developers.openai.com/api/docs/pricing";
    let claude = |constructor, model, cost| Preset {
        constructor,
        provider: "anthropic",
        model,
        vendor_page: anthropic,
        cost,
        absent_upstream: None,
        flat_rate_gap: None,
    };
    vec![
        claude(
            "ModelConfig::claude_fable_5",
            "claude-fable-5",
            ModelConfig::claude_fable_5().cost,
        ),
        claude(
            "ModelConfig::claude_opus_5",
            "claude-opus-5",
            ModelConfig::claude_opus_5().cost,
        ),
        claude(
            "ModelConfig::claude_opus_4_8",
            "claude-opus-4-8",
            ModelConfig::claude_opus_4_8().cost,
        ),
        claude(
            "ModelConfig::claude_sonnet_5",
            "claude-sonnet-5",
            ModelConfig::claude_sonnet_5().cost,
        ),
        claude(
            "ModelConfig::claude_haiku_4_5",
            "claude-haiku-4-5",
            ModelConfig::claude_haiku_4_5().cost,
        ),
        Preset {
            constructor: "ModelConfig::gpt_5_5",
            provider: "openai",
            model: "gpt-5.5",
            vendor_page: openai,
            cost: ModelConfig::gpt_5_5().cost,
            absent_upstream: None,
            flat_rate_gap: Some(FlatRateGap {
                keys: &["tiers", "context_over_200k"],
                rates: &[
                    ("/context_over_200k/input", 10.0),
                    ("/context_over_200k/output", 45.0),
                    ("/context_over_200k/cache_read", 1.0),
                    ("/tiers/0/input", 10.0),
                    ("/tiers/0/output", 45.0),
                    ("/tiers/0/cache_read", 1.0),
                    ("/tiers/0/tier/size", 272000.0),
                ],
                why: "models.dev says gpt-5.5 is tiered at 272K; the preset is \
                      deliberately flat and `ModelConfig::gpt_5_5` documents why. \
                      Short version: OpenAI publishes a >272K schedule but gpt-5.5 \
                      has no row in it, the one gpt-5.5 row that appears in a \
                      long-context table has `-` in every long-context cell, and \
                      $10/$1/$45 is verbatim `gpt-5.6-sol`'s long-context rates \
                      on a model with identical short-context rates. models.dev \
                      also contradicts itself here — `tiers[0].tier.size` is \
                      272000 beside a key named `context_over_200k`. \
                      \
                      The `rates` above are asserted, so this stays honest in both \
                      directions: if upstream revises the numbers or drops the \
                      claim, this fails and the decision gets re-made against new \
                      evidence rather than silently inheriting an old one. \
                      `CostConfig::with_context_tier` is ready if OpenAI publishes \
                      a gpt-5.5 long-context row.",
            }),
        },
        Preset {
            // Generic over the model id; these rates are Muse Spark 1.1/1.2.
            constructor: "ModelConfig::meta",
            provider: "meta",
            model: "muse-spark-1.2",
            vendor_page: "https://dev.meta.ai/docs/pricing-rate-limits",
            cost: ModelConfig::meta("muse-spark-1.2", "Muse Spark 1.2").cost,
            absent_upstream: None,
            flat_rate_gap: None,
        },
    ]
}

/// What models.dev says about one cost field.
///
/// Absent is deliberately not folded into `0.0`: a preset whose rate is
/// legitimately zero would then be unauditable on that field forever, and the
/// operator-facing table would print a comparison that never happened.
#[derive(Debug)]
enum Upstream {
    Value(f64),
    Absent,
    /// Present but not a number — the schema retyped, which is always drift.
    Malformed(String),
}

fn field(cost: &serde_json::Value, name: &str) -> Upstream {
    match cost.get(name) {
        None | Some(serde_json::Value::Null) => Upstream::Absent,
        Some(v) => match v.as_f64() {
            Some(n) => Upstream::Value(n),
            None => Upstream::Malformed(v.to_string()),
        },
    }
}

#[tokio::test]
#[ignore = "network: fetches models.dev; run before a release"]
async fn hardcoded_prices_have_not_drifted() {
    let resp = reqwest::get(DB_URL)
        .await
        .unwrap_or_else(|e| panic!("GET {DB_URL} failed: {e}"));
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("<none>")
        .to_string();
    let body = resp.text().await.expect("read models.dev body");
    let head = &body[..body.len().min(300)];

    // `reqwest::get` returns Ok for 4xx/5xx, so without this a JSON error
    // envelope parses cleanly and every lookup misses — a green run that
    // compared nothing.
    assert!(
        status.is_success(),
        "GET {DB_URL} -> {status} ({content_type})\nfirst 300 bytes:\n{head}"
    );
    let db: serde_json::Value = serde_json::from_str(&body).unwrap_or_else(|e| {
        panic!(
            "GET {DB_URL} -> {status} ({content_type}) is not JSON: {e}\nfirst 300 bytes:\n{head}"
        )
    });

    let mut drift: Vec<String> = Vec::new();
    let mut unexpectedly_missing: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let (mut compared, mut absent) = (0usize, 0usize);

    println!(
        "\n{:<30} {:<14} {:>10} {:>12}",
        "model", "field", "yoagent", "models.dev"
    );
    println!("{}", "-".repeat(72));

    let all = presets();
    for p in &all {
        let Some(cost) = db
            .get(p.provider)
            .and_then(|v| v.get("models"))
            .and_then(|v| v.get(p.model))
            .and_then(|v| v.get("cost"))
        else {
            match p.absent_upstream {
                Some(why) => {
                    absent += 4;
                    notes.push(format!("{} absent upstream — {why}", p.model));
                }
                None => unexpectedly_missing.push(format!(
                    "{} ({}/{}) — was covered, now absent. A rename or removal drops it \
                     silently out of coverage; confirm at {} and either fix the key or \
                     set `absent_upstream` with today's date and a reason.",
                    p.constructor, p.provider, p.model, p.vendor_page
                )),
            }
            continue;
        };

        // Structure this audit does not understand is a reason to fail, not to
        // read past. Tiered rates mean the flat preset is wrong somewhere.
        if let Some(obj) = cost.as_object() {
            let unknown: Vec<String> = obj
                .keys()
                .filter(|k| !KNOWN_COST_KEYS.contains(&k.as_str()))
                .cloned()
                .collect();

            match &p.flat_rate_gap {
                Some(gap) => {
                    // Anything beyond what was acknowledged is new structure,
                    // not covered by an old decision. Without this the note is
                    // a blanket amnesty: a second tier or an audio rate would
                    // fold into it and the preset would stay green forever.
                    let unacknowledged: Vec<&String> = unknown
                        .iter()
                        .filter(|k| !gap.keys.contains(&k.as_str()))
                        .collect();
                    if !unacknowledged.is_empty() {
                        drift.push(format!(
                            "{}: models.dev carries cost keys beyond the recorded gap: \
                             {unacknowledged:?}. The note covers {:?} only. Check {} and \
                             either fix {} or widen the acknowledgement.",
                            p.model, gap.keys, p.vendor_page, p.constructor
                        ));
                    }

                    // The reverse is just as important. If the claim vanishes —
                    // or a key is renamed, which is indistinguishable from
                    // removal — the recorded decision has gone stale and
                    // nothing would otherwise say so.
                    let missing: Vec<&&str> =
                        gap.keys.iter().filter(|k| !obj.contains_key(**k)).collect();
                    if !missing.is_empty() {
                        drift.push(format!(
                            "{}: the recorded gap names {missing:?}, which models.dev no \
                             longer carries. Either upstream dropped the claim — re-check \
                             {} and delete the note — or a key was renamed and this preset \
                             is now silently uncovered.",
                            p.model, p.vendor_page
                        ));
                    }

                    // And the rates the decision was reasoned against.
                    for (pointer, expected) in gap.rates {
                        match cost.pointer(pointer).and_then(|v| v.as_f64()) {
                            Some(actual) if (actual - expected).abs() < 1e-9 => {}
                            other => drift.push(format!(
                                "{}: recorded gap expects {pointer} == {expected}, models.dev \
                                 now says {other:?}. The decision in {} was reasoned against \
                                 the old figure; re-check {}.",
                                p.model, p.constructor, p.vendor_page
                            )),
                        }
                    }
                    notes.push(format!("{}: {}", p.model, gap.why));
                }
                None if !unknown.is_empty() => drift.push(format!(
                    "{}: models.dev carries cost keys this audit ignores: {unknown:?}. \
                     `CostConfig` is one flat rate — if those are tiers, {} is wrong \
                     above the boundary. Check {}, then either fix the preset or record \
                     it in `flat_rate_gap`.",
                    p.model, p.constructor, p.vendor_page
                )),
                None => {}
            }
        }

        for (name, ours) in [
            ("input", p.cost.input_per_million),
            ("output", p.cost.output_per_million),
            ("cache_read", p.cost.cache_read_per_million),
            ("cache_write", p.cost.cache_write_per_million),
        ] {
            match field(cost, name) {
                Upstream::Value(theirs) => {
                    compared += 1;
                    let same = (ours - theirs).abs() < 1e-9;
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
                Upstream::Absent => {
                    absent += 1;
                    // `—`, never `0`: the table must not claim a comparison it
                    // did not make.
                    println!(
                        "{:<30} {:<14} {:>10} {:>12}  not listed upstream",
                        p.model, name, ours, "—"
                    );
                    if ours != 0.0 {
                        drift.push(format!(
                            "{}: {name} is {ours} in the crate and models.dev does not list \
                             it — one of the two is wrong. Check {}.",
                            p.model, p.vendor_page
                        ));
                    }
                }
                Upstream::Malformed(raw) => {
                    compared += 1;
                    drift.push(format!(
                        "{}: {name} is {raw} in models.dev, not a number — the schema \
                         changed and this audit's key path needs re-deriving.",
                        p.model
                    ));
                }
            }
        }
    }

    println!("{}", "-".repeat(72));
    println!(
        "{compared} compared, {absent} not listed upstream, {} drifted, {} unexpectedly missing",
        drift.len(),
        unexpectedly_missing.len()
    );
    for n in &notes {
        println!("  note: {n}");
    }

    // A rename must not quietly reduce coverage.
    assert!(
        unexpectedly_missing.is_empty(),
        "\n\nPresets vanished from models.dev:\n\n{}\n",
        unexpectedly_missing.join("\n")
    );

    // The load-bearing assertion. Without it, every schema change at or above
    // the `cost` level yields drift.is_empty() == true and a green pass having
    // verified nothing.
    // `expected` below is derived from `all`, so it cannot notice `all` itself
    // shrinking — an empty `presets()` produced "0 compared, 0 drifted" and a
    // green pass. Pin the floor against something independent: the number of
    // priced constructors in `src/provider/model.rs`. Raise it when you add
    // one, which is the moment you should also be adding it here.
    const PRICED_PRESETS: usize = 7;
    assert!(
        all.len() >= PRICED_PRESETS,
        "\n\nThe audit is checking {} presets but the crate ships at least {PRICED_PRESETS}. \
         A priced preset is unaudited — its rates can drift for as many releases as it takes \
         someone to notice, which for `claude_sonnet_5` was 18.\n",
        all.len()
    );

    let expected = all.len() * 4;
    assert_eq!(
        compared + absent,
        expected,
        "\n\nThe audit accounted for {} of {expected} fields. It is not reporting \
         clean prices — it is reporting nothing. models.dev's schema or hosting \
         changed underneath this test; re-derive the key path against {DB_URL}.\n",
        compared + absent
    );

    assert!(
        drift.is_empty(),
        "\n\nPrice drift detected. models.dev is community-maintained and NOT \
         authoritative — confirm against the vendor page before changing any \
         constant, and never copy models.dev blindly.\n\n{}\n",
        drift.join("\n")
    );
}

/// `is_configured` means *any* rate is set — all-zero is "pricing unknown",
/// never "free".
///
/// Asserted against `CostConfig` values rather than particular presets: pinning
/// a preset as unpriced would forbid a future improvement (pricing DeepSeek is
/// exactly what this file encourages) and would fail with a message describing
/// a regression that did not happen.
///
/// The partial case is the load-bearing one. A preset with no cache-write rate
/// is priced, and only that assertion distinguishes `any` from `all` — an
/// earlier version used two presets whose fields were all-nonzero and all-zero,
/// so it survived flipping the `||` chain to `&&`.
#[test]
fn is_configured_means_any_rate_set() {
    assert!(
        !CostConfig::default().is_configured(),
        "all-zero rates mean pricing is unknown, not that the model is free"
    );

    let no_cache_write = CostConfig::new(5.0, 30.0).with_cache_read(0.5);
    assert!(
        no_cache_write.is_configured(),
        "a provider that charges nothing for cache writes is priced, not unknown"
    );

    let only_one_field = CostConfig::new(0.0, 0.0).with_cache_read(0.1);
    assert!(
        only_one_field.is_configured(),
        "any single rate is enough to count as priced"
    );

    for p in presets() {
        assert!(
            p.cost.is_configured(),
            "{} is in the price audit but reads as unpriced",
            p.constructor
        );
    }
}

/// A request above the tier boundary must cost the tier rate.
///
/// Built from a literal, not a preset: no shipped preset is tiered today, and
/// this must keep testing the mechanism if that stays true. The boundary is
/// compared against *prompt* tokens — `input + cache_read + cache_write` — so a
/// long reply to a short prompt stays on the base rate.
#[test]
fn cost_usd_applies_the_context_tier_by_prompt_size() {
    use yoagent::provider::{ContextTier, CostConfig};
    use yoagent::types::Usage;

    let cfg = CostConfig::new(5.0, 30.0)
        .with_cache_read(0.5)
        .with_context_tier(ContextTier::new(272_000, 10.0, 45.0).with_cache_read(1.0));

    let usage = |input: u64, output: u64| Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        total_tokens: input + output,
    };

    // Below: base rates. 100k in, 1k out => 100k*$5/M + 1k*$30/M.
    let below = cfg.cost_usd(&usage(100_000, 1_000));
    assert!(
        (below - (0.5 + 0.03)).abs() < 1e-9,
        "below the boundary must use base rates, got {below}"
    );

    // Above: tier rates. 300k in, 1k out => 300k*$10/M + 1k*$45/M.
    let above = cfg.cost_usd(&usage(300_000, 1_000));
    assert!(
        (above - (3.0 + 0.045)).abs() < 1e-9,
        "above the boundary must use tier rates, got {above}"
    );

    // Exactly at the boundary is *below* — the field is `above_prompt_tokens`.
    let at = cfg.cost_usd(&usage(272_000, 0));
    assert!(
        (at - 1.36).abs() < 1e-9,
        "the boundary itself must stay on the base rate, got {at}"
    );

    // Cached tokens count toward prompt size: 272k cache reads and 1 fresh
    // token is a 272_001-token prompt, so the tier applies.
    let cached = cfg.cost_usd(&Usage {
        input: 1,
        output: 0,
        cache_read: 272_000,
        cache_write: 0,
        total_tokens: 272_001,
    });
    assert!(
        (cached - (272_000.0 + 10.0) / 1_000_000.0).abs() < 1e-9,
        "cache reads must count toward the boundary and bill at the tier's \
         cache rate, got {cached}"
    );

    // Output does not push a short prompt over the line.
    let long_reply = cfg.cost_usd(&usage(1_000, 400_000));
    assert!(
        (long_reply - (0.005 + 12.0)).abs() < 1e-9,
        "the tier keys on prompt size, not total, got {long_reply}"
    );
}
