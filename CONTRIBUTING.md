# Contributing to yoagent

Thanks for taking the time. Bug reports, provider quirks, and small focused PRs are all welcome.

## Getting set up

```bash
git clone https://github.com/yologdev/yoagent && cd yoagent
cargo build --all-features
cargo test --all-features
```

You need **Rust 1.86 or newer** — that's the MSRV, and CI checks it on every PR.

No API keys are required to develop on yoagent. Of the 463 tests, 456 run with no network and no
credentials; the 7 that need a live key are `#[ignore]`d by default and live in
`tests/integration_*.rs`.

## Before you open a PR

CI runs with `RUSTFLAGS="-Dwarnings"`, so **any** clippy warning fails the build. Run the same
checks locally first:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
```

CI additionally runs a Windows compile check, an MSRV (1.86) check, and a GASP conformance job
that emits an agent repo and validates it against the protocol's checker.

## Conventions

**Branches** — `feat/<feature-name>` off `main`.

**Commits and PR titles** — prefix with `feat:`, `fix:`, `docs:`, `chore:`, or `release:`.

**Tests** — new behaviour needs a test. Unit tests live beside the code in `src/`; integration
tests go in `tests/`. Use `MockProvider` (`src/provider/mock.rs`) to script LLM responses rather
than hitting a real API:

```rust
let provider = MockProvider::new(vec![
    MockResponse::ToolCalls(vec![/* ... */]),
    MockResponse::Text("done".into()),
]);
let agent = Agent::from_provider(provider, ModelConfig::mock());
```

A test that only passes because of a timing assumption isn't a test. If you're unsure a new test
actually covers the change, break the change deliberately and confirm the test fails.

**Docs** — user-facing changes belong in the [mdBook](docs/) as well as in rustdoc. Build the site
locally with `mdbook build` and open `book/index.html`.

**Changelog** — add an entry under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md).

## Live smoke

Unit tests run against `MockProvider`. It validates message shape, but it cannot
stream SSE, cannot price a request, and cannot tell you whether a model actually
follows a truncation marker. Bugs that reached released versions all lived in
that gap — tools with zero parameters were uncallable on Anthropic, and its
first ever run is what found that.

Before a release, run the **Live Smoke** workflow against the release branch, or
locally:

```bash
ANTHROPIC_API_KEY=... cargo run --example release_smoke
SMOKE_MODEL=deepseek DEEPSEEK_API_KEY=... cargo run --example release_smoke
```

It exits non-zero on any failed check. Both providers matter: they are separate
SSE parsers and tool-call accumulators.

`examples/long_horizon.rs` is the diagnostic sibling — compaction, cache
behaviour across compaction, sub-agent delegation. Slower, and its output wants
reading rather than a pass/fail, so it is deliberately not a gate.

## Adding a provider

Most new providers are OpenAI-compatible and need no new code — just a `ModelConfig` with the right
`base_url` and an `OpenAiCompat` flag set for any quirks (auth style, reasoning format,
`max_tokens` field name). See `src/provider/model.rs` for the existing presets.

A genuinely new wire protocol means implementing `StreamProvider` (`src/provider/traits.rs`), adding
an `ApiProtocol` variant, wiring it into `ProviderRegistry`, and adding a `wiremock`-based SSE test
alongside the existing ones in `tests/*_stream_test.rs`.

If a provider reports context overflow with a new error string, add it to `OVERFLOW_PHRASES` in
`src/provider/traits.rs` — that list is the single place overflow is detected.

## Adding your project to the README

If you've built something on yoagent, open a PR adding a row to the **Built with yoagent** table in
[README.md](README.md) — project link and one line on what it is. It doesn't need to be big or
finished. Seeing what people build with the loop is genuinely useful for deciding what to work on
next.

## Reporting bugs

Provider-specific issues are the most common kind, so please include the protocol and provider
along with the model id. The [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) asks for
these. A minimal reproduction using `MockProvider` is the fastest path to a fix, but a description
of the failing request is fine too.

## Security

Please don't open a public issue for a vulnerability — see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the MIT License.
