//! Regression guard for the retry/re-bill interaction introduced by #83.
//!
//! Making `StreamEnded` retryable means any terminal SSE event a provider fails
//! to handle stops the loop from breaking, so the body close is misread as
//! truncation and the whole generation is re-requested — re-billing a response
//! the server already produced. `response.incomplete` and `response.failed`
//! were both unhandled in the Responses-style providers.
//!
//! These assert at the *loop* layer, counting requests on the mock server;
//! a provider-level test cannot see a retry because the retry lives in
//! `agent_loop`.

use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};
use yoagent::agent::Agent;
use yoagent::provider::{ModelConfig, OpenAiResponsesProvider};
use yoagent::retry::RetryConfig;

/// Fast retries so a regression fails in milliseconds rather than seconds.
fn quick_retry() -> RetryConfig {
    RetryConfig {
        max_retries: 3,
        initial_delay_ms: 1,
        backoff_multiplier: 1.0,
        max_delay_ms: 5,
    }
}

async fn run_and_count(body: &'static str) -> usize {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let mut mc = ModelConfig::openai("gpt-5.5", "GPT-5.5");
    mc.base_url = server.uri();
    let mut agent = Agent::from_provider(OpenAiResponsesProvider, mc)
        .with_api_key("test-key")
        .with_retry_config(quick_retry());

    let mut rx = agent.prompt("hi").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;

    server.received_requests().await.unwrap().len()
}

/// `response.incomplete` is terminal — the model stopped early (e.g. it hit the
/// output cap). Retrying regenerates the same capped response at full cost.
#[tokio::test]
async fn response_incomplete_is_terminal_not_retried() {
    let body = "event: response.output_text.delta\n\
        data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n\
        event: response.incomplete\n\
        data: {\"type\":\"response.incomplete\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"total_tokens\":6}}}\n\n";

    assert_eq!(
        run_and_count(body).await,
        1,
        "response.incomplete must end the turn; retrying re-bills a generation \
         the server already completed"
    );
}

/// `response.failed` is terminal too, and it is an error — not a truncation to
/// retry blindly.
#[tokio::test]
async fn response_failed_is_terminal_not_retried() {
    let body = "event: response.failed\n\
        data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"content policy\"}}}\n\n";

    assert_eq!(
        run_and_count(body).await,
        1,
        "response.failed must surface as an error, not be retried"
    );
}
