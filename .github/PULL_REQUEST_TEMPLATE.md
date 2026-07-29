<!--
Title convention: feat: / fix: / docs: / chore: / release:
-->

## What this changes

<!-- One or two sentences. Link the issue if there is one: "Closes #123". -->

## Why

<!-- What was broken or missing. For provider fixes, name the provider and protocol. -->

## How it was verified

<!--
Say what you actually ran, not what should pass. If a test is new, say what it would catch —
ideally confirm it fails without the fix.
-->

- [ ] `cargo fmt -- --check`
- [ ] `cargo clippy --all-targets --all-features` (CI runs with `-Dwarnings`)
- [ ] `cargo test --all-features`

## Checklist

- [ ] Tests cover the change
- [ ] `CHANGELOG.md` updated under `## Unreleased`
- [ ] Docs updated ([`docs/`](../docs/) and/or rustdoc) if user-facing
- [ ] No new public API without a doc comment

<!--
Breaking change? Say so explicitly here and describe the migration.
-->
