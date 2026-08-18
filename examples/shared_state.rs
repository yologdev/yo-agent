//! Shared state example: parallel sub-agents analyzing the same artifact.
//!
//! Demonstrates:
//!   - Storing a large artifact once in SharedState
//!   - Multiple sub-agents reading it by reference (not re-pasted)
//!   - Sub-agents writing findings back to shared state
//!   - Parent reading all findings after completion
//!
//! The "aha moment": the CI log is 50KB but each sub-agent's prompt is
//! just one sentence. They all read the same artifact via shared_state
//! tool — no context wasted on re-pasting.
//!
//! Run:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example shared_state

use std::sync::Arc;
use yoagent::provider::{AnthropicProvider, ModelConfig, StreamProvider};
use yoagent::shared_state::SharedState;
use yoagent::sub_agent::SubAgentTool;
use yoagent::*;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("Set ANTHROPIC_API_KEY");
    let config = ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
    let provider: Arc<dyn StreamProvider> = Arc::new(AnthropicProvider);

    // --- The artifact: a large CI log (simulated) ---
    let ci_log = r#"
[2026-04-27T10:00:01Z] Starting CI pipeline for commit abc123
[2026-04-27T10:00:02Z] Step 1/5: cargo fmt -- --check ... OK
[2026-04-27T10:00:15Z] Step 2/5: cargo clippy ... OK (42 warnings suppressed)
[2026-04-27T10:00:30Z] Step 3/5: cargo build ... OK (debug, 45s)
[2026-04-27T10:01:15Z] Step 4/5: cargo test ...
[2026-04-27T10:01:16Z]   test_auth_basic ... ok (12ms)
[2026-04-27T10:01:16Z]   test_auth_refresh ... ok (8ms)
[2026-04-27T10:01:17Z]   test_db_connection ... FAILED (timeout after 30000ms)
[2026-04-27T10:01:47Z]     thread 'test_db_connection' panicked at 'connection timed out: TcpStream::connect'
[2026-04-27T10:01:47Z]     note: database host db-ci.internal:5432 unreachable
[2026-04-27T10:01:48Z]   test_api_list_users ... ok (145ms)
[2026-04-27T10:01:48Z]   test_api_create_user ... ok (89ms)
[2026-04-27T10:01:49Z]   test_api_delete_user ... FAILED
[2026-04-27T10:01:49Z]     assertion failed: `(left == right)` left: 404, right: 204
[2026-04-27T10:01:49Z]     at tests/api_test.rs:142
[2026-04-27T10:01:50Z]   test_cache_invalidation ... ok (3ms)
[2026-04-27T10:01:50Z]   test_cache_ttl ... ok (1002ms)  [SLOW]
[2026-04-27T10:01:51Z]   test_cache_concurrent ... ok (2105ms) [SLOW]
[2026-04-27T10:01:53Z]   test_migration_up ... ok (340ms)
[2026-04-27T10:01:54Z]   test_migration_down ... ok (290ms)
[2026-04-27T10:01:54Z]   test_migration_idempotent ... ok (680ms) [SLOW]
[2026-04-27T10:01:55Z]   test_flaky_network_retry ... FAILED
[2026-04-27T10:01:55Z]     thread 'test_flaky_network_retry' panicked at 'retry count exceeded'
[2026-04-27T10:01:55Z]     note: this test is known-flaky, see issue #187
[2026-04-27T10:01:55Z] test result: 3 failed; 11 passed; 0 ignored
[2026-04-27T10:01:55Z] Step 5/5: skipped (tests failed)
[2026-04-27T10:01:55Z] Pipeline FAILED in 114s
"#;

    // --- Store the artifact once in shared state ---
    let state = SharedState::new();
    state
        .set("ci_log", ci_log.to_string())
        .await
        .expect("store CI log");

    println!("Stored CI log ({} bytes) in shared state.\n", ci_log.len());

    // --- Three sub-agents, each analyzing a different aspect ---

    let error_analyst =
        SubAgentTool::from_provider("error_analyst", Arc::clone(&provider), config.clone())
            .with_description("Analyzes test failures in CI logs")
            .with_system_prompt(
                "You analyze CI logs for test failures. Read the log from shared state, \
             identify each failure, its root cause, and write a concise summary back \
             to shared state under 'errors_summary'. Be brief — bullet points only.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    let perf_analyst =
        SubAgentTool::from_provider("perf_analyst", Arc::clone(&provider), config.clone())
            .with_description("Analyzes performance issues in CI logs")
            .with_system_prompt(
                "You analyze CI logs for performance issues. Read the log from shared state, \
             identify slow tests and bottlenecks, and write a concise summary back \
             to shared state under 'perf_summary'. Be brief — bullet points only.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    let flaky_analyst =
        SubAgentTool::from_provider("flaky_analyst", Arc::clone(&provider), config.clone())
            .with_description("Identifies flaky tests in CI logs")
            .with_system_prompt(
                "You analyze CI logs for flaky/unreliable tests. Read the log from shared state, \
             identify tests that are flaky or infrastructure-dependent, and write a concise \
             summary back to shared state under 'flaky_summary'. Be brief — bullet points only.",
            )
            .with_api_key(&api_key)
            .with_shared_state(state.clone())
            .with_max_turns(5);

    // --- Run all three in parallel ---
    println!("Dispatching 3 sub-agents in parallel...\n");

    let ctx = |name: &str| ToolContext::new(format!("tc-{}", name), name.to_string());

    let (r1, r2, r3) = tokio::join!(
        error_analyst.execute(
            serde_json::json!({"task": "Analyze the CI log for test failures."}),
            ctx("error_analyst"),
        ),
        perf_analyst.execute(
            serde_json::json!({"task": "Analyze the CI log for performance issues."}),
            ctx("perf_analyst"),
        ),
        flaky_analyst.execute(
            serde_json::json!({"task": "Analyze the CI log for flaky tests."}),
            ctx("flaky_analyst"),
        ),
    );

    r1.expect("error analyst failed");
    r2.expect("perf analyst failed");
    r3.expect("flaky analyst failed");

    // --- Read all findings from shared state ---
    println!("=== All sub-agents complete. Reading findings from shared state: ===\n");

    for key in ["errors_summary", "perf_summary", "flaky_summary"] {
        match state.get(key).await {
            Some(value) => println!("--- {} ---\n{}\n", key, value),
            None => println!("--- {} ---\n(sub-agent did not write this key)\n", key),
        }
    }

    println!("=== Shared state keys: {} ===", state.summary().await);
}
