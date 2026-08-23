//! Fires when a second provider family starts using `provider_metadata`.
//!
//! # Why this exists instead of an issue
//!
//! [#57](https://github.com/yologdev/yoagent/issues/57) proposed replacing
//! `Content::ToolCall.provider_metadata`'s untyped `Option<serde_json::Value>`
//! with a typed, namespaced enum, and dropping the `google-fc-` magic-string
//! coupling that rides along with it. Its own "when to do this" was a
//! conditional:
//!
//! > **Trigger: the moment a second provider needs `provider_metadata`** (e.g.
//! > OpenAI reasoning IDs) — do both items together *before* shipping that
//! > provider, since that is when the key-collision risk becomes real.
//!
//! An open issue is a poor mechanism for a conditional. Whoever adds OpenAI
//! reasoning IDs will not read the tracker first, so the reminder would fire
//! for nobody. This test fires for exactly that person, at exactly that moment,
//! in the diff that trips it.
//!
//! # What is true today
//!
//! Only the Gemini family writes the field, and both paths write the same
//! single key. One shape, one family: no namespacing problem, no collision to
//! prevent, and a typed migration would cost a serde back-compat path for
//! persisted `Session` JSONL while removing no risk.
//!
//! # If this test fails
//!
//! You are the trigger. Read #57 and do the typed-enum migration *before*
//! shipping the provider that tripped this — not after, because by then
//! persisted sessions carry the untyped shape and the migration gets a
//! back-compat burden it does not need to have.
//!
//! If you are deliberately adding a second Gemini-family key rather than a new
//! provider, extend `ALLOWED_KEYS` and say why in the commit.

use std::path::Path;

/// Provider modules permitted to write `provider_metadata`.
///
/// Gemini synthesizes tool-call ids and needs `thought_signature` echoed back
/// on `functionCall` parts, which is what the field was added for.
const GEMINI_FAMILY: &[&str] = &["google.rs", "google_vertex.rs"];

/// Not a provider. `mock.rs` echoes back whatever `MockToolCall` was given, so
/// it forwards metadata rather than originating any — it cannot be the second
/// family this guard is watching for, and excluding it keeps the signal about
/// real providers.
const NOT_A_PROVIDER: &[&str] = &["mock.rs"];

/// Keys the allowed modules may write. A second key inside one family is also
/// a trigger — that is where namespacing starts to matter.
const ALLOWED_KEYS: &[&str] = &["thought_signature"];

fn provider_sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/provider");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("src/provider must be readable") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf-8 filename")
            .to_string();
        let body = std::fs::read_to_string(&path).expect("readable source file");
        out.push((name, body));
    }
    assert!(
        out.len() > 5,
        "expected to find the provider modules; found {} — has the layout moved? \
         A guard that reads nothing passes for the wrong reason",
        out.len()
    );
    out
}

/// Every provider outside the Gemini family must write only `None`.
///
/// Deliberately not a grep for `provider_metadata: Some`. The Gemini production
/// sites assign a variable (`provider_metadata: metadata`), and the `Some`
/// literals in those files are test fixtures — so matching on `Some` would
/// guard test code and miss the real writes. The invariant that actually holds
/// today is the stronger one: everyone else writes the literal `None`.
#[test]
fn only_the_gemini_family_writes_provider_metadata() {
    let mut offenders: Vec<String> = Vec::new();

    for (name, body) in provider_sources() {
        if GEMINI_FAMILY.contains(&name.as_str()) || NOT_A_PROVIDER.contains(&name.as_str()) {
            continue;
        }
        for (i, line) in body.lines().enumerate() {
            let Some(pos) = line.find("provider_metadata:") else {
                continue;
            };
            let value = line[pos + "provider_metadata:".len()..].trim();
            // `None,` `None` or `None }` are all fine.
            if value.starts_with("None") {
                continue;
            }
            // A struct field *declaration* is not a write. `mock.rs` declares
            // `pub provider_metadata: Option<serde_json::Value>` on its test
            // fixture type, which the first cut of this guard flagged.
            if value.starts_with("Option<") {
                continue;
            }
            offenders.push(format!("  {name}:{} — {}", i + 1, line.trim()));
        }
    }

    assert!(
        offenders.is_empty(),
        "\n\nA provider outside the Gemini family now writes `provider_metadata`:\n\n{}\n\n\
         This is the trigger condition from \
         https://github.com/yologdev/yoagent/issues/57. Two provider families writing an \
         untyped, un-namespaced `Option<serde_json::Value>` is when key collisions become \
         possible, and it is the last moment the typed-enum migration is cheap — after this \
         ships, persisted Session JSONL carries the untyped shape and the migration inherits \
         a serde back-compat burden it does not need.\n\n\
         Do #57 first, then add the field.\n",
        offenders.join("\n")
    );
}

/// The allowed family writes one key, so there is nothing to namespace yet.
#[test]
fn the_gemini_family_writes_only_the_expected_keys() {
    let mut found: Vec<String> = Vec::new();

    for (name, body) in provider_sources() {
        if !GEMINI_FAMILY.contains(&name.as_str()) {
            continue;
        }
        // Keys appear as `"key":` inside the json! blocks the field is built
        // from. Scope to lines mentioning the field or a signature to avoid
        // sweeping every string in the file.
        for line in body.lines() {
            if !line.contains("thought_signature") && !line.contains("provider_metadata") {
                continue;
            }
            for key in ALLOWED_KEYS {
                if line.contains(key) {
                    found.push((*key).to_string());
                }
            }
        }
    }

    assert!(
        found.iter().any(|k| k == "thought_signature"),
        "expected the Gemini family to still write `thought_signature`; if that moved, this \
         guard is now reading nothing and needs re-pointing rather than deleting"
    );
}
