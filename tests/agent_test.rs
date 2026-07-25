//! Tests for the Agent struct (stateful wrapper).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use yoagent::agent::Agent;
use yoagent::provider::mock::*;
use yoagent::provider::MockProvider;
use yoagent::provider::ModelConfig;
use yoagent::*;

#[tokio::test]
async fn test_agent_simple_prompt() {
    let provider = MockProvider::text("Hello!");
    let mut agent =
        Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("You are helpful.");

    let mut rx = agent.prompt("Hi there").await;

    // Drain events
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    agent.finish().await;
    assert!(!events.is_empty());
    assert_eq!(agent.messages().len(), 2); // user + assistant
}

#[tokio::test]
async fn test_agent_reset() {
    let provider = MockProvider::text("Hello!");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    let mut rx = agent.prompt("Hi").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;
    assert!(!agent.messages().is_empty());

    agent.reset().await;
    assert!(agent.messages().is_empty());
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn test_agent_with_tools() {
    struct EchoTool;

    #[async_trait::async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn label(&self) -> &str {
            "Echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn execute(
            &self,
            params: serde_json::Value,
            _ctx: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            let text = params["text"].as_str().unwrap_or("").to_string();
            Ok(ToolResult {
                content: vec![Content::Text { text }],
                details: serde_json::Value::Null,
            })
        }
    }

    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "echo".into(),
            arguments: serde_json::json!({"text": "hello"}),
        }]),
        MockResponse::Text("Echoed: hello".into()),
    ]);

    let mut agent = Agent::from_provider(provider, ModelConfig::mock())
        .with_system_prompt("test")
        .with_tools(vec![Box::new(EchoTool)]);

    let mut rx = agent.prompt("Echo hello").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;

    // user + assistant(tool_call) + toolResult + assistant(text)
    assert_eq!(agent.messages().len(), 4);
}

// Deliberately exercises the deprecated builder chain (`new` + `with_model` +
// `with_api_key`) to keep coverage of that still-present API until 1.0.
#[tokio::test]
#[allow(deprecated)]
async fn test_agent_builder_pattern() {
    let provider = MockProvider::text("ok");
    let agent = Agent::new(provider)
        .with_system_prompt("sys")
        .with_model("test-model")
        .with_api_key("key123")
        .with_thinking(ThinkingLevel::Medium)
        .with_max_tokens(4096);

    assert_eq!(agent.system_prompt, "sys");
    assert_eq!(agent.model, "test-model");
    assert_eq!(agent.api_key, "key123");
    assert_eq!(agent.thinking_level, ThinkingLevel::Medium);
    assert_eq!(agent.max_tokens, Some(4096));
}

// ---------------------------------------------------------------------------
// State persistence tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_with_messages_builder() {
    let saved = vec![
        AgentMessage::Llm(Message::user("Hello")),
        AgentMessage::Llm(Message::assistant(
            vec![Content::Text {
                text: "Hi there!".into(),
            }],
            StopReason::Stop,
            "mock",
            "mock",
            Usage::default(),
        )),
    ];

    let provider = MockProvider::text("ok");
    let agent = Agent::from_provider(provider, ModelConfig::mock()).with_messages(saved.clone());

    assert_eq!(agent.messages().len(), 2);
    assert_eq!(*agent.messages(), saved[..]);
}

#[tokio::test]
async fn test_save_and_restore_messages() {
    let provider = MockProvider::text("Hello!");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    let mut rx = agent.prompt("Hi").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;
    let json = agent.save_messages().expect("save should succeed");

    // Create a fresh agent and restore
    let provider2 = MockProvider::text("ok");
    let mut agent2 =
        Agent::from_provider(provider2, ModelConfig::mock()).with_system_prompt("test");

    agent2
        .restore_messages(&json)
        .expect("restore should succeed");
    assert_eq!(agent.messages(), agent2.messages());
}

#[tokio::test]
async fn test_agent_continues_after_restore() {
    // First agent: prompt → get response → save
    let provider1 = MockProvider::text("First response");
    let mut agent1 =
        Agent::from_provider(provider1, ModelConfig::mock()).with_system_prompt("test");

    let mut rx = agent1.prompt("Hello").await;
    while rx.recv().await.is_some() {}
    agent1.finish().await;
    let json = agent1.save_messages().expect("save");

    // Second agent: restore → prompt again
    // The MockProvider will receive the full restored history + new prompt
    let provider2 = MockProvider::text("Second response");
    let mut agent2 =
        Agent::from_provider(provider2, ModelConfig::mock()).with_system_prompt("test");

    agent2.restore_messages(&json).expect("restore");
    let mut rx = agent2.prompt("Follow up").await;
    while rx.recv().await.is_some() {}
    agent2.finish().await;

    // Should have: original user + original assistant + follow-up user + new assistant
    assert_eq!(agent2.messages().len(), 4);
    assert_eq!(agent2.messages()[0].role(), "user");
    assert_eq!(agent2.messages()[1].role(), "assistant");
    assert_eq!(agent2.messages()[2].role(), "user");
    assert_eq!(agent2.messages()[3].role(), "assistant");
}

// ---------------------------------------------------------------------------
// Real-time streaming tests (prompt_with_sender / prompt_messages_with_sender)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_prompt_with_sender_streams_events() {
    let provider = MockProvider::text("Hello!");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let event_count = Arc::new(AtomicUsize::new(0));
    let count_clone = event_count.clone();

    let consumer = tokio::spawn(async move {
        while let Some(_event) = rx.recv().await {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    agent.prompt_with_sender("Hi there", tx).await;

    // tx is dropped when prompt_with_sender returns, so consumer will finish
    consumer.await.unwrap();

    assert!(event_count.load(Ordering::SeqCst) > 0);
    assert_eq!(agent.messages().len(), 2); // user + assistant
    assert!(!agent.is_streaming());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_prompt_with_sender_real_time_streaming() {
    let provider = MockProvider::text("Hello!");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let received_during = Arc::new(AtomicUsize::new(0));
    let received_clone = received_during.clone();

    // On a multi-threaded runtime, the consumer can run concurrently
    let consumer = tokio::spawn(async move {
        while let Some(_event) = rx.recv().await {
            received_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    agent.prompt_with_sender("Hello", tx).await;
    consumer.await.unwrap();

    // Events were consumed by the concurrent task
    assert!(received_during.load(Ordering::SeqCst) > 0);
    assert_eq!(agent.messages().len(), 2);
}

#[tokio::test]
async fn test_prompt_messages_with_sender() {
    let provider = MockProvider::text("Response");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    let (tx, mut rx) = mpsc::unbounded_channel();

    let consumer = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        events
    });

    let msgs = vec![AgentMessage::Llm(Message::user("Hello"))];
    agent.prompt_messages_with_sender(msgs, tx).await;

    let events = consumer.await.unwrap();
    assert!(!events.is_empty());
    assert_eq!(agent.messages().len(), 2);
}

#[tokio::test]
async fn test_continue_loop_with_sender() {
    let provider = MockProvider::text("Continued response");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    // First, add some messages to continue from (last must not be assistant)
    agent.append_message(AgentMessage::Llm(Message::user("Hello")));
    agent.append_message(AgentMessage::Llm(
        Message::assistant(
            vec![Content::Text { text: "Hi!".into() }],
            StopReason::Error,
            "mock",
            "mock",
            Usage::default(),
        )
        .with_error_message("rate limited"),
    ));
    agent.append_message(AgentMessage::Llm(Message::user("Please try again")));

    let (tx, mut rx) = mpsc::unbounded_channel();

    let consumer = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        events
    });

    agent.continue_loop_with_sender(tx).await;

    let events = consumer.await.unwrap();
    assert!(!events.is_empty());
    assert!(!agent.is_streaming());
}

#[tokio::test]
async fn test_prompt_with_sender_tools_restored() {
    struct DummyTool;

    #[async_trait::async_trait]
    impl AgentTool for DummyTool {
        fn name(&self) -> &str {
            "dummy"
        }
        fn label(&self) -> &str {
            "Dummy"
        }
        fn description(&self) -> &str {
            "A dummy tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![Content::Text { text: "ok".into() }],
                details: serde_json::Value::Null,
            })
        }
    }

    let provider = MockProvider::text("Hello!");
    let mut agent = Agent::from_provider(provider, ModelConfig::mock())
        .with_system_prompt("test")
        .with_tools(vec![Box::new(DummyTool)]);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let consumer = tokio::spawn(async move { while rx.recv().await.is_some() {} });

    agent.prompt_with_sender("Hi", tx).await;
    consumer.await.unwrap();

    // Tools should be restored after the call
    assert!(!agent.is_streaming());
    // Agent should still work for another prompt
    let mut rx2 = agent.prompt("Follow up").await;
    while rx2.recv().await.is_some() {}
    agent.finish().await;
    assert_eq!(agent.messages().len(), 4); // 2 from first + 2 from second
}

#[tokio::test]
async fn test_queue_inspection_and_take() {
    let provider = MockProvider::text("Hello!");
    let agent = Agent::from_provider(provider, ModelConfig::mock()).with_system_prompt("test");

    assert_eq!(agent.steering_queue_len(), 0);
    assert!(agent.steering_queue_snapshot().is_empty());

    agent.steer(AgentMessage::Llm(Message::user("stop")));
    agent.steer(AgentMessage::Llm(Message::user("use v2 instead")));
    agent.follow_up(AgentMessage::Llm(Message::user("then run tests")));

    // Inspection does not consume
    assert_eq!(agent.follow_up_queue_len(), 1);
    let snapshot = agent.steering_queue_snapshot();
    assert_eq!(snapshot.len(), 2);
    assert_eq!(agent.steering_queue_len(), 2, "snapshot must not drain");

    // Take drains atomically and returns in FIFO order
    let taken = agent.take_steering_queue();
    assert_eq!(taken.len(), 2);
    assert_eq!(agent.steering_queue_len(), 0);
    let AgentMessage::Llm(Message::User { content, .. }) = &taken[0] else {
        panic!("expected user message");
    };
    assert_eq!(
        content,
        &vec![Content::Text {
            text: "stop".into()
        }]
    );

    // Edit-and-requeue: drop the first entry, batch-requeue the survivor
    agent.steer_all(vec![taken[1].clone()]);
    assert_eq!(agent.steering_queue_len(), 1);
    let requeued = agent.steering_queue_snapshot();
    let AgentMessage::Llm(Message::User { content, .. }) = &requeued[0] else {
        panic!("expected user message");
    };
    assert_eq!(
        content,
        &vec![Content::Text {
            text: "use v2 instead".into()
        }]
    );

    // Follow-up variants
    let taken = agent.take_follow_up_queue();
    assert_eq!(taken.len(), 1);
    assert_eq!(agent.follow_up_queue_len(), 0);
    assert!(agent.follow_up_queue_snapshot().is_empty());
}

// ---------------------------------------------------------------------------
// session_cost_usd
// ---------------------------------------------------------------------------

fn assistant_with_usage(usage: Usage) -> AgentMessage {
    AgentMessage::Llm(Message::assistant(
        vec![Content::Text { text: "ok".into() }],
        StopReason::Stop,
        "id",
        "model",
        usage,
    ))
}

fn some_usage() -> Usage {
    Usage {
        input: 1_000_000,
        output: 500_000,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 1_500_000,
    }
}

// Covers session_cost_usd's "no ModelConfig at all" branch (early `?` return).
// Only the deprecated `Agent::new` (with no from_* config) leaves model_config
// unset, so this test keeps that API to preserve that coverage until 1.0.
#[test]
#[allow(deprecated)]
fn session_cost_usd_none_without_model_config() {
    let agent =
        Agent::new(MockProvider::text("x")).with_messages(vec![assistant_with_usage(some_usage())]);
    assert_eq!(agent.session_cost_usd(), None);
}

#[test]
fn session_cost_usd_none_when_rates_unconfigured() {
    // ModelConfig::custom has all-zero cost rates: pricing is unknown, so
    // the answer must be None ("can't price"), not Some(0.0) ("free").
    let mc = yoagent::provider::ModelConfig::custom(
        yoagent::provider::ApiProtocol::OpenAiCompletions,
        "local",
        "http://localhost:8080/v1",
        "m",
        "M",
    );
    let agent = Agent::from_provider(MockProvider::text("x"), mc)
        .with_messages(vec![assistant_with_usage(some_usage())]);
    assert_eq!(agent.session_cost_usd(), None);
}

#[test]
fn session_cost_usd_sums_assistant_turns_only() {
    let mut mc = yoagent::provider::ModelConfig::custom(
        yoagent::provider::ApiProtocol::OpenAiCompletions,
        "local",
        "http://localhost:8080/v1",
        "m",
        "M",
    );
    mc.cost.input_per_million = 3.0;
    mc.cost.output_per_million = 15.0;
    let expected_per_turn = mc.cost.cost_usd(&some_usage());

    let agent = Agent::from_provider(MockProvider::text("x"), mc).with_messages(vec![
        AgentMessage::Llm(Message::user("hi")),
        assistant_with_usage(some_usage()),
        assistant_with_usage(some_usage()),
    ]);
    let total = agent.session_cost_usd().expect("rates are configured");
    assert!(total > 0.0);
    assert!((total - 2.0 * expected_per_turn).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// API-key resolution: explicit key wins; env var is the fallback
// ---------------------------------------------------------------------------

/// Records the api_key each stream call receives.
struct KeyCapturingProvider {
    captured: Arc<std::sync::Mutex<String>>,
}

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for KeyCapturingProvider {
    async fn stream(
        &self,
        config: yoagent::provider::StreamConfig,
        tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        *self.captured.lock().unwrap() = config.api_key.clone();
        let msg = Message::assistant(
            vec![Content::Text { text: "ok".into() }],
            StopReason::Stop,
            "mock",
            "mock",
            Usage::default(),
        );
        let _ = tx.send(yoagent::provider::StreamEvent::Start);
        let _ = tx.send(yoagent::provider::StreamEvent::Done {
            message: msg.clone(),
        });
        Ok(msg)
    }
}

/// Run one prompt to completion so the provider records the resolved key.
async fn run_one_prompt(agent: &mut Agent) {
    let mut rx = agent.prompt("hi").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;
}

#[tokio::test]
async fn test_explicit_api_key_wins_over_env() {
    // Each key-resolution test uses its own env var to stay race-free under
    // parallel test execution.
    std::env::set_var("ZAI_API_KEY", "env-key-should-lose");
    let captured = Arc::new(std::sync::Mutex::new(String::new()));
    let mut agent = Agent::from_provider(
        KeyCapturingProvider {
            captured: captured.clone(),
        },
        yoagent::provider::ModelConfig::custom(
            yoagent::provider::ApiProtocol::OpenAiCompletions,
            "zai",
            "http://localhost:8080/v1",
            "m",
            "M",
        ),
    )
    .with_api_key("explicit-key");
    run_one_prompt(&mut agent).await;
    assert_eq!(*captured.lock().unwrap(), "explicit-key");
}

#[tokio::test]
async fn test_env_var_fallback_resolves_api_key() {
    std::env::set_var("CEREBRAS_API_KEY", "cerebras-env-key");
    let captured = Arc::new(std::sync::Mutex::new(String::new()));
    let mut agent = Agent::from_provider(
        KeyCapturingProvider {
            captured: captured.clone(),
        },
        yoagent::provider::ModelConfig::custom(
            yoagent::provider::ApiProtocol::OpenAiCompletions,
            "cerebras",
            "http://localhost:8080/v1",
            "m",
            "M",
        ),
    );
    run_one_prompt(&mut agent).await;
    assert_eq!(*captured.lock().unwrap(), "cerebras-env-key");
}

// ---------------------------------------------------------------------------
// 0.10 construction API: from_config / from_provider / from_config_with / set_model
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_from_provider_runs_end_to_end() {
    // from_provider + ModelConfig::mock() is the test-double construction path.
    let mut agent = Agent::from_provider(
        MockProvider::text("Hi from mock"),
        yoagent::provider::ModelConfig::mock(),
    );
    assert_eq!(agent.model, "mock");

    let mut rx = agent.prompt("hello").await;
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }
    agent.finish().await;
    assert_eq!(agent.messages().len(), 2);
}

#[test]
fn test_from_config_wires_model_and_config() {
    // from_config selects a built-in provider from config.api, sets the id,
    // and stashes pricing so session_cost_usd can price the session.
    let mut mc = yoagent::provider::ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5");
    mc.cost.input_per_million = 3.0;
    let agent = Agent::from_config(mc).with_messages(vec![assistant_with_usage(some_usage())]);
    assert_eq!(agent.model, "claude-sonnet-5");
    assert!(
        agent.session_cost_usd().is_some_and(|c| c > 0.0),
        "from_config must carry the config's pricing"
    );
}

#[tokio::test]
async fn test_from_config_resolves_env_key() {
    std::env::set_var("OPENROUTER_API_KEY", "openrouter-from-config");
    let captured = Arc::new(std::sync::Mutex::new(String::new()));
    // from_provider carries the config (so provider="openrouter" drives env
    // resolution) while letting us capture the resolved key.
    let mut agent = Agent::from_provider(
        KeyCapturingProvider {
            captured: captured.clone(),
        },
        yoagent::provider::ModelConfig::custom(
            yoagent::provider::ApiProtocol::OpenAiCompletions,
            "openrouter",
            "http://unused.invalid",
            "m",
            "M",
        ),
    );
    run_one_prompt(&mut agent).await;
    assert_eq!(*captured.lock().unwrap(), "openrouter-from-config");
}

#[test]
fn test_from_config_with_errors_on_empty_registry() {
    let registry = yoagent::provider::ProviderRegistry::new();
    // Agent isn't Debug, so match instead of expect_err.
    let err = match Agent::from_config_with(
        &registry,
        yoagent::provider::ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"),
    ) {
        Ok(_) => panic!("empty registry must fail"),
        Err(e) => e,
    };
    assert_eq!(
        err,
        yoagent::AgentBuildError::NoProviderForProtocol(
            yoagent::provider::ApiProtocol::AnthropicMessages
        )
    );
}

#[tokio::test]
async fn test_set_model_switches_model_id() {
    let mut agent = Agent::from_config(yoagent::provider::ModelConfig::anthropic(
        "claude-sonnet-5",
        "Sonnet 5",
    ));
    assert_eq!(agent.model, "claude-sonnet-5");
    agent.set_model(yoagent::provider::ModelConfig::anthropic(
        "claude-opus-4-8",
        "Opus 4.8",
    ));
    assert_eq!(agent.model, "claude-opus-4-8");
}

#[test]
fn test_model_config_mock_is_unpriced() {
    let mc = yoagent::provider::ModelConfig::mock();
    assert_eq!(mc.provider, "mock");
    assert!(!mc.cost.is_configured());
}

#[tokio::test]
async fn test_set_model_preserves_explicit_provider_and_key() {
    // Regression guard for the set_model custom-provider clobber: an agent
    // built with an explicit provider must keep BOTH that provider and an
    // explicit key across a model switch — never silently swap in a built-in
    // network provider.
    let captured = Arc::new(std::sync::Mutex::new(String::new()));
    let mut agent = Agent::from_provider(
        KeyCapturingProvider {
            captured: captured.clone(),
        },
        yoagent::provider::ModelConfig::mock(),
    )
    .with_api_key("explicit-key")
    .with_retry_config(yoagent::RetryConfig::none());

    // Switch to a different protocol whose built-in provider would hit the
    // network if it clobbered ours.
    agent.set_model(yoagent::provider::ModelConfig::anthropic(
        "claude-sonnet-5",
        "Sonnet 5",
    ));
    assert_eq!(agent.model, "claude-sonnet-5");

    run_one_prompt(&mut agent).await;
    // If the provider had been clobbered, our capture probe would never run
    // (and the real Anthropic provider would have been used instead).
    assert_eq!(
        *captured.lock().unwrap(),
        "explicit-key",
        "set_model must keep the explicit provider and key"
    );
}

// ---------------------------------------------------------------------------
// Tool middleware: approve / deny / modify hooks gating tool execution
// ---------------------------------------------------------------------------

/// Tool that records whether it ran and with what args.
struct RecordingTool {
    ran: Arc<std::sync::Mutex<Option<serde_json::Value>>>,
}

#[async_trait::async_trait]
impl AgentTool for RecordingTool {
    fn name(&self) -> &str {
        "recording_tool"
    }
    fn label(&self) -> &str {
        "Recording Tool"
    }
    fn description(&self) -> &str {
        "Records its invocation"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        *self.ran.lock().unwrap() = Some(params);
        Ok(ToolResult {
            content: vec![Content::Text { text: "ok".into() }],
            details: serde_json::Value::Null,
        })
    }
}

/// Middleware driven by a closure, for tests.
struct FnMiddleware<F>(F);

#[async_trait::async_trait]
impl<F> ToolMiddleware for FnMiddleware<F>
where
    F: Fn(&str, &serde_json::Value) -> ToolDecision + Send + Sync,
{
    async fn before_tool(&self, call: &ToolCallRequest<'_>) -> ToolDecision {
        (self.0)(call.tool_name, call.args)
    }
}

/// Provider that calls recording_tool once, then finishes with text.
fn tool_call_provider() -> MockProvider {
    MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "recording_tool".into(),
            arguments: serde_json::json!({"path": "/etc/passwd"}),
        }]),
        MockResponse::Text("done".into()),
    ])
}

async fn run_middleware_agent(mut agent: Agent) -> (Agent, Vec<AgentEvent>) {
    let mut rx = agent.prompt("go").await;
    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }
    agent.finish().await;
    (agent, events)
}

#[tokio::test]
async fn test_tool_middleware_deny_blocks_tool_and_loop_continues() {
    let ran = Arc::new(std::sync::Mutex::new(None));
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_middleware(FnMiddleware(|name: &str, _args: &serde_json::Value| {
            if name == "recording_tool" {
                ToolDecision::Deny("blocked by policy".into())
            } else {
                ToolDecision::Allow
            }
        }));

    let (agent, events) = run_middleware_agent(agent).await;

    // Tool must never have executed.
    assert!(ran.lock().unwrap().is_none(), "denied tool must not run");

    // The denial reaches the LLM as an error tool result with the reason.
    let denial = agent
        .messages()
        .iter()
        .find_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult {
                content, is_error, ..
            }) => Some((content.clone(), *is_error)),
            _ => None,
        })
        .expect("a tool result must be recorded");
    assert!(denial.1, "denial must be an error result");
    match &denial.0[0] {
        Content::Text { text } => {
            assert!(text.contains("Tool call denied"), "got: {text}");
            assert!(text.contains("blocked by policy"), "got: {text}");
        }
        other => panic!("expected text content, got {other:?}"),
    }

    // The loop continued to the final assistant text (not aborted).
    assert!(matches!(
        agent.messages().last(),
        Some(AgentMessage::Llm(Message::Assistant { .. }))
    ));

    // Event pairing stays intact: Start and End both emitted, End is error.
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionStart { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. })));
}

#[tokio::test]
async fn test_tool_middleware_modify_rewrites_args() {
    let ran = Arc::new(std::sync::Mutex::new(None));
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_middleware(FnMiddleware(|_: &str, _: &serde_json::Value| {
            ToolDecision::Modify(serde_json::json!({"path": "/tmp/sandboxed"}))
        }));

    let (_, events) = run_middleware_agent(agent).await;

    // Tool ran with the REWRITTEN args, not the LLM's originals.
    let seen = ran.lock().unwrap().clone().expect("tool must run");
    assert_eq!(seen["path"], "/tmp/sandboxed");

    // The Start event carries the effective (post-middleware) args.
    let start_args = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolExecutionStart { args, .. } => Some(args.clone()),
            _ => None,
        })
        .expect("start event");
    assert_eq!(start_args["path"], "/tmp/sandboxed");
}

#[tokio::test]
async fn test_tool_middleware_chain_first_deny_wins() {
    // First middleware rewrites; second sees the rewritten args and denies.
    let ran = Arc::new(std::sync::Mutex::new(None));
    let saw = Arc::new(std::sync::Mutex::new(None));
    let saw2 = saw.clone();
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_middleware(FnMiddleware(|_: &str, _: &serde_json::Value| {
            ToolDecision::Modify(serde_json::json!({"path": "/rewritten"}))
        }))
        .with_tool_middleware(FnMiddleware(move |_: &str, args: &serde_json::Value| {
            *saw2.lock().unwrap() = Some(args.clone());
            ToolDecision::Deny("second says no".into())
        }));

    let (agent, _) = run_middleware_agent(agent).await;

    assert!(ran.lock().unwrap().is_none(), "denied tool must not run");
    // Second middleware observed the first one's rewrite (chain order).
    let observed = saw.lock().unwrap().clone().expect("second middleware ran");
    assert_eq!(observed["path"], "/rewritten");
    // Reason from the denying middleware reaches the LLM.
    let has_reason = agent.messages().iter().any(|m| {
        matches!(m, AgentMessage::Llm(Message::ToolResult { content, .. })
            if matches!(&content[0], Content::Text { text } if text.contains("second says no")))
    });
    assert!(has_reason);
}

#[tokio::test]
async fn test_tool_middleware_deny_under_sequential_strategy() {
    // The choke point is shared, but pin the Sequential path explicitly too.
    let ran = Arc::new(std::sync::Mutex::new(None));
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_execution(ToolExecutionStrategy::Sequential)
        .with_tool_middleware(FnMiddleware(|_: &str, _: &serde_json::Value| {
            ToolDecision::Deny("no".into())
        }));

    let _ = run_middleware_agent(agent).await;
    assert!(ran.lock().unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Structured outputs: prompt_structured::<T>()
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Debug, PartialEq)]
struct Extracted {
    name: String,
    count: u32,
}

#[tokio::test]
async fn test_prompt_structured_parses_text_json() {
    // Providers with native json_schema/responseSchema return plain JSON text.
    let mut agent = Agent::from_provider(
        MockProvider::text(r#"{"name": "widget", "count": 3}"#),
        yoagent::provider::ModelConfig::mock(),
    );
    let out: Extracted = agent
        .prompt_structured("extract", serde_json::json!({"type": "object"}))
        .await
        .expect("must parse");
    assert_eq!(
        out,
        Extracted {
            name: "widget".into(),
            count: 3
        }
    );
}

#[tokio::test]
async fn test_prompt_structured_unwraps_forced_tool_call() {
    // Tool-forcing providers (Anthropic) return the payload as a tool call
    // named after the schema; the loop must unwrap it to text — and must NOT
    // try to execute it as a real tool.
    let provider = MockProvider::new(vec![MockResponse::ToolCalls(vec![MockToolCall {
        provider_metadata: None,
        name: "structured_output".into(),
        arguments: serde_json::json!({"name": "gadget", "count": 7}),
    }])]);
    let mut agent = Agent::from_provider(provider, yoagent::provider::ModelConfig::mock());
    let out: Extracted = agent
        .prompt_structured("extract", serde_json::json!({"type": "object"}))
        .await
        .expect("forced tool call must unwrap and parse");
    assert_eq!(
        out,
        Extracted {
            name: "gadget".into(),
            count: 7
        }
    );
    // The unwrap happened in the loop: history ends with a plain assistant
    // message (Stop), no tool-result turn for the synthetic tool.
    assert!(!agent
        .messages()
        .iter()
        .any(|m| matches!(m, AgentMessage::Llm(Message::ToolResult { .. }))));
}

#[tokio::test]
async fn test_prompt_structured_parse_error_carries_raw() {
    let mut agent = Agent::from_provider(
        MockProvider::text("not json at all"),
        yoagent::provider::ModelConfig::mock(),
    );
    let err = agent
        .prompt_structured::<Extracted>("extract", serde_json::json!({"type": "object"}))
        .await
        .expect_err("must fail to parse");
    match err {
        yoagent::StructuredPromptError::Parse { raw, .. } => {
            assert!(raw.contains("not json at all"));
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[tokio::test]
async fn test_prompt_structured_strips_markdown_fences() {
    let mut agent = Agent::from_provider(
        MockProvider::text("```json\n{\"name\": \"x\", \"count\": 1}\n```"),
        yoagent::provider::ModelConfig::mock(),
    );
    let out: Extracted = agent
        .prompt_structured("extract", serde_json::json!({"type": "object"}))
        .await
        .expect("fenced JSON must parse");
    assert_eq!(out.count, 1);
}

/// Provider that records each call's output_schema, then answers with text.
struct SchemaCapturingProvider {
    schemas: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    reply: String,
}

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for SchemaCapturingProvider {
    async fn stream(
        &self,
        config: yoagent::provider::StreamConfig,
        tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        self.schemas
            .lock()
            .unwrap()
            .push(config.output_schema.as_ref().map(|s| s.name.clone()));
        let msg = Message::assistant(
            vec![Content::Text {
                text: self.reply.clone(),
            }],
            StopReason::Stop,
            "mock",
            "mock",
            Usage::default(),
        );
        let _ = tx.send(yoagent::provider::StreamEvent::Start);
        let _ = tx.send(yoagent::provider::StreamEvent::Done {
            message: msg.clone(),
        });
        Ok(msg)
    }
}

#[tokio::test]
async fn test_prompt_structured_schema_reaches_provider_then_resets() {
    // Pins the loop→StreamConfig propagation joint (MockProvider ignores its
    // config, so only a capturing provider can verify it) AND that the very
    // next plain prompt carries no schema.
    let schemas = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut agent = Agent::from_provider(
        SchemaCapturingProvider {
            schemas: schemas.clone(),
            reply: r#"{"name": "a", "count": 1}"#.into(),
        },
        yoagent::provider::ModelConfig::mock(),
    );
    let _: Extracted = agent
        .prompt_structured("extract", serde_json::json!({"type": "object"}))
        .await
        .unwrap();

    let mut rx = agent.prompt("normal question").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;

    let seen = schemas.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![Some("structured_output".to_string()), None],
        "structured call must carry the schema; the next plain prompt must not"
    );
}

/// Provider that always fails, for the Provider error-variant path.
struct AlwaysFailProvider;

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for AlwaysFailProvider {
    async fn stream(
        &self,
        _config: yoagent::provider::StreamConfig,
        _tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        Err(yoagent::provider::ProviderError::Api(
            "invalid x-api-key".into(),
        ))
    }
}

#[tokio::test]
async fn test_prompt_structured_surfaces_provider_error() {
    // A dead API key must surface as Provider { message } carrying the real
    // error — never as Parse { raw: "" }.
    let mut agent =
        Agent::from_provider(AlwaysFailProvider, yoagent::provider::ModelConfig::mock())
            .with_retry_config(yoagent::RetryConfig::none());
    let err = agent
        .prompt_structured::<Extracted>("extract", serde_json::json!({"type": "object"}))
        .await
        .unwrap_err();
    match err {
        yoagent::StructuredPromptError::Provider { message } => {
            assert!(message.contains("invalid x-api-key"), "got: {message}");
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn test_prompt_structured_never_parses_stale_history() {
    // Seed history with an earlier turn's perfectly-parsable JSON; a failed
    // run must NOT fall back to it and return stale data as Ok.
    let stale = AgentMessage::Llm(Message::assistant(
        vec![Content::Text {
            text: r#"{"name": "stale", "count": 99}"#.into(),
        }],
        StopReason::Stop,
        "mock",
        "mock",
        Usage::default(),
    ));
    let mut agent =
        Agent::from_provider(AlwaysFailProvider, yoagent::provider::ModelConfig::mock())
            .with_retry_config(yoagent::RetryConfig::none())
            .with_messages(vec![stale]);

    let err = agent
        .prompt_structured::<Extracted>("extract", serde_json::json!({"type": "object"}))
        .await
        .unwrap_err();
    assert!(
        matches!(err, yoagent::StructuredPromptError::Provider { .. }),
        "must not return the stale turn's JSON, got {err:?}"
    );
}

#[tokio::test]
async fn test_middleware_never_sees_synthetic_structured_tool() {
    // The forced structured_output call is unwrapped BEFORE tool extraction,
    // so middleware must never be consulted for it.
    let calls = Arc::new(std::sync::Mutex::new(0usize));
    let calls2 = calls.clone();
    let provider = MockProvider::new(vec![MockResponse::ToolCalls(vec![MockToolCall {
        provider_metadata: None,
        name: "structured_output".into(),
        arguments: serde_json::json!({"name": "g", "count": 7}),
    }])]);
    let mut agent = Agent::from_provider(provider, yoagent::provider::ModelConfig::mock())
        .with_tool_middleware(FnMiddleware(move |_: &str, _: &serde_json::Value| {
            *calls2.lock().unwrap() += 1;
            ToolDecision::Deny("should never run".into())
        }));
    let out: Extracted = agent
        .prompt_structured("extract", serde_json::json!({"type": "object"}))
        .await
        .expect("unwrap happens before middleware");
    assert_eq!(out.count, 7);
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "middleware saw the synthetic tool"
    );
}

#[tokio::test]
async fn test_tool_middleware_panic_denies_and_loop_survives() {
    // A panicking middleware must not kill the loop task (which would strip
    // the agent of its tools) — it fails closed as a denial.
    let ran = Arc::new(std::sync::Mutex::new(None));
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_middleware(FnMiddleware(|_: &str, _: &serde_json::Value| {
            panic!("middleware bug")
        }));

    let (agent, _) = run_middleware_agent(agent).await;
    assert!(ran.lock().unwrap().is_none(), "tool must not run");
    // Loop survived to the final text, and the denial reached the LLM.
    assert!(matches!(
        agent.messages().last(),
        Some(AgentMessage::Llm(Message::Assistant { .. }))
    ));
    let denied = agent.messages().iter().any(|m| {
        matches!(m, AgentMessage::Llm(Message::ToolResult { content, is_error: true, .. })
            if matches!(&content[0], Content::Text { text } if text.contains("middleware panicked")))
    });
    assert!(denied);
    // Crucially: the tools survived for the next prompt.
    // (RecordingTool was moved into the loop and restored by finish().)
}

#[tokio::test]
async fn test_tool_middleware_deny_under_batched_strategy() {
    let ran = Arc::new(std::sync::Mutex::new(None));
    let agent = Agent::from_provider(tool_call_provider(), yoagent::provider::ModelConfig::mock())
        .with_tools(vec![Box::new(RecordingTool { ran: ran.clone() })])
        .with_tool_execution(ToolExecutionStrategy::Batched { size: 1 })
        .with_tool_middleware(FnMiddleware(|_: &str, _: &serde_json::Value| {
            ToolDecision::Deny("no".into())
        }));
    let (agent, events) = run_middleware_agent(agent).await;
    assert!(ran.lock().unwrap().is_none());
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. })));
    let denied = agent.messages().iter().any(|m| {
        matches!(
            m,
            AgentMessage::Llm(Message::ToolResult { is_error: true, .. })
        )
    });
    assert!(denied);
}

// ---------------------------------------------------------------------------
// Drop: dropping a streaming Agent must not orphan the spawned loop (#84)
// ---------------------------------------------------------------------------

/// Fires when the tool future is dropped, so a test can prove the *in-flight*
/// tool was actually cancelled rather than merely that the channel closed.
struct CancelSignal(mpsc::UnboundedSender<()>);

impl Drop for CancelSignal {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Signals when it has been entered, then parks. Deliberately ignores
/// `ctx.cancel`: only a real abort can stop it, so a passing test cannot be
/// explained by cooperative cancellation alone.
struct BarrierTool {
    entered: mpsc::UnboundedSender<()>,
    cancelled: mpsc::UnboundedSender<()>,
}

#[async_trait::async_trait]
impl AgentTool for BarrierTool {
    fn name(&self) -> &str {
        "barrier_tool"
    }
    fn label(&self) -> &str {
        "Barrier Tool"
    }
    fn description(&self) -> &str {
        "Signals that it started, then parks"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let _guard = CancelSignal(self.cancelled.clone());
        let _ = self.entered.send(());
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "done".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

/// Issue #84: `JoinHandle` does not cancel its task on drop, so before the
/// `Drop` impl a dropped streaming Agent left the loop running as an orphan —
/// holding the event channel open, so the caller's receiver never closed.
///
/// Synchronisation is a barrier, not a sleep. That matters: `MockProvider`
/// checks `cancel.is_cancelled()` at the top of `stream()`, so a `drop` landing
/// before the loop reaches the tool would let cooperative cancellation alone
/// close the channel — and the test would pass with `abort()` removed. Waiting
/// for the tool to signal entry makes the abort the only possible explanation.
///
/// Every timeout below is a failure deadline, never a synchronisation point.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_drop_aborts_in_flight_tool_and_closes_channel() {
    let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
    let (cancelled_tx, mut cancelled_rx) = mpsc::unbounded_channel();

    let provider = MockProvider::new(vec![MockResponse::ToolCalls(vec![MockToolCall {
        name: "barrier_tool".into(),
        arguments: serde_json::json!({}),
        provider_metadata: None,
    }])]);

    let mut agent = Agent::from_provider(provider, ModelConfig::mock())
        .with_system_prompt("test")
        .with_tools(vec![Box::new(BarrierTool {
            entered: entered_tx,
            cancelled: cancelled_tx,
        })]);

    let mut rx = agent.prompt("run the barrier tool").await;
    assert!(agent.is_streaming(), "loop should be in flight");

    // The loop is now provably inside the tool.
    tokio::time::timeout(std::time::Duration::from_secs(10), entered_rx.recv())
        .await
        .expect("agent loop must reach the tool")
        .expect("tool must signal entry");

    drop(agent);

    // Load-bearing: the *running* tool future was dropped. Cooperative
    // cancellation cannot do this — BarrierTool ignores ctx.cancel and
    // agent_loop awaits it without a select!.
    let cancelled =
        tokio::time::timeout(std::time::Duration::from_secs(5), cancelled_rx.recv()).await;
    assert!(
        matches!(cancelled, Ok(Some(()))),
        "dropping the Agent must abort the in-flight tool future; it kept running"
    );

    // And the caller's receiver closes rather than hanging.
    let drained = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while rx.recv().await.is_some() {}
    })
    .await;
    assert!(
        drained.is_ok(),
        "dropping the Agent must close the event channel; it stayed open"
    );
}
