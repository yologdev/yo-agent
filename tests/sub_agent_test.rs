//! Tests for SubAgentTool using MockProvider.

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use yoagent::agent_loop::{agent_loop, AgentLoopConfig};
use yoagent::provider::mock::*;
use yoagent::provider::MockProvider;
use yoagent::provider::ModelConfig;
use yoagent::sub_agent::SubAgentTool;
use yoagent::*;

fn make_config(provider: MockProvider) -> AgentLoopConfig {
    AgentLoopConfig {
        provider: std::sync::Arc::new(provider),
        model: "mock".into(),
        api_key: "test".into(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: None,
        temperature: None,
        model_config: None,
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        context_config: None,
        compaction_strategy: None,
        execution_limits: None,
        cache_config: CacheConfig::default(),
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::default(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    }
}

fn collect_events(mut rx: mpsc::UnboundedReceiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}

// ---------------------------------------------------------------------------
// Basic sub-agent execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_basic() {
    // The sub-agent's mock provider returns a simple text response
    let sub_provider = Arc::new(MockProvider::text("Research result: Rust is great"));

    let sub_agent = SubAgentTool::from_provider("researcher", sub_provider, ModelConfig::mock())
        .with_description("Researches topics")
        .with_system_prompt("You are a research assistant.");

    // Execute the sub-agent tool directly
    let params = serde_json::json!({"task": "Tell me about Rust"});

    let result = sub_agent
        .execute(params, ToolContext::new("tc-1", "researcher"))
        .await
        .expect("sub-agent should succeed");

    // Should contain the sub-agent's response text
    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    assert_eq!(text, "Research result: Rust is great");

    // Details should include sub-agent metadata
    assert_eq!(result.details["sub_agent"], "researcher");
}

// ---------------------------------------------------------------------------
// Sub-agent with its own tools
// ---------------------------------------------------------------------------

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
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            }
        })
    }
    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let text = params["text"].as_str().unwrap_or("(empty)");
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("echoed: {}", text),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_sub_agent_with_tools() {
    // Sub-agent first calls the echo tool, then responds with text
    let sub_provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "echo".into(),
            arguments: serde_json::json!({"text": "hello"}),
        }]),
        MockResponse::Text("The echo returned: echoed: hello".into()),
    ]));

    let echo_tool: Arc<dyn AgentTool> = Arc::new(EchoTool);

    let sub_agent = SubAgentTool::from_provider("echo_agent", sub_provider, ModelConfig::mock())
        .with_description("Agent that echoes")
        .with_system_prompt("Use the echo tool.")
        .with_tools(vec![echo_tool]);

    let params = serde_json::json!({"task": "Echo hello"});

    let result = sub_agent
        .execute(params, ToolContext::new("tc-1", "echo_agent"))
        .await
        .expect("sub-agent should succeed");

    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    assert_eq!(text, "The echo returned: echoed: hello");
}

// ---------------------------------------------------------------------------
// Cancellation propagation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_cancellation() {
    // Sub-agent provider returns text, but we cancel before execution
    let sub_provider = Arc::new(MockProvider::text("Should not appear"));

    let sub_agent =
        SubAgentTool::from_provider("cancelled_agent", sub_provider, ModelConfig::mock());

    let cancel = CancellationToken::new();
    cancel.cancel(); // Cancel immediately

    let params = serde_json::json!({"task": "Do something"});

    let result = sub_agent
        .execute(
            params,
            ToolContext::new("tc-1", "cancelled_agent").with_cancel(cancel),
        )
        .await
        .expect("should return a result even when cancelled");

    // When cancelled before the loop runs, we get the fallback message
    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    // The loop exits early on cancellation, so the mock response should not appear
    assert_ne!(
        text, "Should not appear",
        "Sub-agent ran despite cancellation"
    );
}

// ---------------------------------------------------------------------------
// Max turns limit
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_max_turns() {
    // Sub-agent keeps calling tools indefinitely — max_turns should stop it.
    // With max_turns=1, the sub-agent gets 1 LLM call.
    // Response 1: tool call → executes tool → hits turn limit → returns limit message
    // The sub-agent won't get a second LLM call to produce text.
    let sub_provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "echo".into(),
            arguments: serde_json::json!({"text": "loop"}),
        }]),
        // This response won't be reached due to turn limit
        MockResponse::Text("Should not reach".into()),
    ]));

    let echo_tool: Arc<dyn AgentTool> = Arc::new(EchoTool);

    let sub_agent = SubAgentTool::from_provider("limited_agent", sub_provider, ModelConfig::mock())
        .with_tools(vec![echo_tool])
        .with_max_turns(1); // Only 1 turn allowed

    let params = serde_json::json!({"task": "Keep going"});

    let result = sub_agent
        .execute(params, ToolContext::new("tc-1", "limited_agent"))
        .await
        .expect("sub-agent should succeed");

    // The sub-agent was stopped by turn limit — it won't have the second text response
    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    // Should NOT contain the text from the second response
    assert_ne!(text, "Should not reach");
}

// ---------------------------------------------------------------------------
// Parallel sub-agent execution (via parent agent loop)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_parallel() {
    // Two sub-agents that each take ~50ms, run in parallel via the parent loop.
    // The parent's mock emits both sub-agent tool calls, then a final text.

    struct SlowProvider {
        delay_ms: u64,
        text: String,
    }

    #[async_trait::async_trait]
    impl yoagent::provider::StreamProvider for SlowProvider {
        async fn stream(
            &self,
            _config: yoagent::provider::StreamConfig,
            tx: tokio::sync::mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
            cancel: tokio_util::sync::CancellationToken,
        ) -> Result<Message, yoagent::provider::ProviderError> {
            if cancel.is_cancelled() {
                return Err(yoagent::provider::ProviderError::Cancelled);
            }
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;

            let _ = tx.send(yoagent::provider::StreamEvent::Start);
            let _ = tx.send(yoagent::provider::StreamEvent::TextDelta {
                content_index: 0,
                delta: self.text.clone(),
            });
            let msg = Message::assistant(
                vec![Content::Text {
                    text: self.text.clone(),
                }],
                StopReason::Stop,
                "slow",
                "slow",
                Usage::default(),
            );
            let _ = tx.send(yoagent::provider::StreamEvent::Done {
                message: msg.clone(),
            });
            Ok(msg)
        }
    }

    let sub_a = SubAgentTool::from_provider(
        "agent_a",
        Arc::new(SlowProvider {
            delay_ms: 50,
            text: "Result A".into(),
        }),
        ModelConfig::mock(),
    );

    let sub_b = SubAgentTool::from_provider(
        "agent_b",
        Arc::new(SlowProvider {
            delay_ms: 50,
            text: "Result B".into(),
        }),
        ModelConfig::mock(),
    );

    // Parent provider: first call triggers both sub-agents, second returns final text
    let parent_provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "agent_a".into(),
                arguments: serde_json::json!({"task": "Do A"}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "agent_b".into(),
                arguments: serde_json::json!({"task": "Do B"}),
            },
        ]),
        MockResponse::Text("Both sub-agents completed.".into()),
    ]);

    let config = make_config(parent_provider);

    let mut context = AgentContext {
        system_prompt: "You are a coordinator.".into(),
        messages: Vec::new(),
        tools: vec![Box::new(sub_a), Box::new(sub_b)],
    };

    let prompt = AgentMessage::Llm(Message::user("Run both agents"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let elapsed = start.elapsed();

    let _events = collect_events(rx);

    // Both tool results should be present
    let tool_results: Vec<_> = new_messages
        .iter()
        .filter(|m| m.role() == "toolResult")
        .collect();
    assert_eq!(tool_results.len(), 2);

    // Should complete in roughly 50-100ms (parallel), not 100ms+ (sequential)
    assert!(
        elapsed.as_millis() < 130,
        "Parallel sub-agents took {}ms, expected <130ms",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// Event forwarding via on_update
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_event_forwarding() {
    let sub_provider = Arc::new(MockProvider::text("Sub-agent done"));

    let sub_agent =
        SubAgentTool::from_provider("streaming_agent", sub_provider, ModelConfig::mock());

    let params = serde_json::json!({"task": "Do work"});

    // Collect on_update calls
    let updates: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let updates_clone = updates.clone();
    let on_update: ToolUpdateFn = Arc::new(move |result: ToolResult| {
        if let Some(Content::Text { text }) = result.content.first() {
            updates_clone.lock().unwrap().push(text.clone());
        }
    });

    let result = sub_agent
        .execute(
            params,
            ToolContext::new("tc-1", "streaming_agent").with_on_update(on_update),
        )
        .await
        .expect("sub-agent should succeed");

    // Final result should contain the sub-agent's text
    let text = match &result.content[0] {
        Content::Text { text } => text.as_str(),
        _ => panic!("Expected text content"),
    };
    assert_eq!(text, "Sub-agent done");

    // on_update should have received streaming deltas
    let collected = updates.lock().unwrap();
    assert!(
        !collected.is_empty(),
        "Expected on_update to receive streaming events"
    );
    // Should contain the text delta from the mock provider
    assert!(
        collected.iter().any(|t| t.contains("Sub-agent done")),
        "Expected text delta in updates, got: {:?}",
        *collected
    );
}

// ---------------------------------------------------------------------------
// Invalid parameters
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_missing_task_parameter() {
    let sub_provider = Arc::new(MockProvider::text("Should not run"));

    let sub_agent = SubAgentTool::from_provider("test_agent", sub_provider, ModelConfig::mock());

    let params = serde_json::json!({}); // Missing "task"

    let result = sub_agent
        .execute(params, ToolContext::new("tc-1", "test_agent"))
        .await;
    assert!(result.is_err());

    match result.unwrap_err() {
        ToolError::InvalidArgs(msg) => assert!(msg.contains("task")),
        other => panic!("Expected InvalidArgs, got: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Skills: with_skills injects the skills index into the sub-agent system prompt
// ---------------------------------------------------------------------------

/// Provider that records the system prompt it is dispatched with, so tests can
/// assert on the exact prompt the sub-agent assembles.
struct CapturingProvider {
    captured: Arc<std::sync::Mutex<String>>,
}

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for CapturingProvider {
    async fn stream(
        &self,
        config: yoagent::provider::StreamConfig,
        tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        *self.captured.lock().unwrap() = config.system_prompt.clone();
        let _ = tx.send(yoagent::provider::StreamEvent::Start);
        let msg = Message::assistant(
            vec![Content::Text {
                text: "done".into(),
            }],
            StopReason::Stop,
            "mock",
            "mock",
            Usage::default(),
        );
        let _ = tx.send(yoagent::provider::StreamEvent::Done {
            message: msg.clone(),
        });
        Ok(msg)
    }
}

/// RAII guard for a per-test temp skills directory. Holds a unique path
/// (avoids collisions under parallel `cargo test`) and removes it on drop,
/// so cleanup runs even if the test panics.
struct SkillsDir(std::path::PathBuf);

impl SkillsDir {
    /// Create a temp dir containing a single `<name>/SKILL.md`. `unique` must
    /// differ per test to avoid concurrent collisions on the shared temp dir.
    fn with_one_skill(unique: &str, name: &str, description: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("yoagent-test-skills-{unique}"));
        let _ = std::fs::remove_dir_all(&dir);
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n\nBody.\n"),
        )
        .unwrap();
        Self(dir)
    }

    fn load(&self) -> yoagent::skills::SkillSet {
        yoagent::skills::SkillSet::load(&[self.0.to_string_lossy().to_string()]).unwrap()
    }
}

impl Drop for SkillsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Dispatch a sub-agent through a `CapturingProvider` and return the system
/// prompt the provider was called with.
async fn capture_system_prompt(
    build: impl FnOnce(Arc<CapturingProvider>) -> SubAgentTool,
) -> String {
    let captured = Arc::new(std::sync::Mutex::new(String::new()));
    let provider = Arc::new(CapturingProvider {
        captured: captured.clone(),
    });
    let sub_agent = build(provider);

    sub_agent
        .execute(
            serde_json::json!({"task": "do work"}),
            ToolContext::new("tc-1", "sub"),
        )
        .await
        .expect("sub-agent should succeed");

    let prompt = captured.lock().unwrap().clone();
    prompt
}

#[tokio::test]
async fn test_sub_agent_with_skills() {
    let skills_dir = SkillsDir::with_one_skill(
        "with-skills",
        "research",
        "How to call the search and read APIs",
    );
    let skills = skills_dir.load();
    assert_eq!(skills.len(), 1, "expected the research skill to load");

    let prompt = capture_system_prompt(|provider| {
        SubAgentTool::from_provider("researcher", provider, ModelConfig::mock())
            .with_system_prompt("You are a research assistant.")
            .with_skills(skills)
    })
    .await;

    // Base system prompt is preserved...
    assert!(
        prompt.contains("You are a research assistant."),
        "base system prompt missing, got: {prompt}"
    );
    // ...and the skills index is appended.
    assert!(
        prompt.contains("<available_skills>") && prompt.contains("<name>research</name>"),
        "skills index not injected into sub-agent system prompt, got: {prompt}"
    );
}

#[tokio::test]
async fn test_sub_agent_with_skills_empty_base_prompt() {
    // Exercises the `system_prompt.is_empty()` branch: skills become the entire
    // prompt with no leading blank line. assert_eq pins the exact output.
    let skills_dir = SkillsDir::with_one_skill("empty-base", "research", "desc");
    let skills = skills_dir.load();
    let expected = skills.format_for_prompt();
    assert!(!expected.is_empty());

    let prompt = capture_system_prompt(|provider| {
        // No with_system_prompt() call — base prompt is empty.
        SubAgentTool::from_provider("researcher", provider, ModelConfig::mock()).with_skills(skills)
    })
    .await;

    assert_eq!(
        prompt, expected,
        "with empty base prompt, the skills index should be the whole prompt verbatim"
    );
}

#[tokio::test]
async fn test_sub_agent_with_empty_skillset_is_noop() {
    // An empty SkillSet must not alter the system prompt (no trailing "\n\n").
    let prompt = capture_system_prompt(|provider| {
        SubAgentTool::from_provider("researcher", provider, ModelConfig::mock())
            .with_system_prompt("Base prompt.")
            .with_skills(yoagent::skills::SkillSet::empty())
    })
    .await;

    assert_eq!(prompt, "Base prompt.", "empty SkillSet should be a no-op");
}

#[tokio::test]
async fn test_sub_agent_skills_before_shared_state() {
    // Skills and shared-state both append to the prompt; lock in the order
    // base -> skills -> shared-state.
    let skills_dir = SkillsDir::with_one_skill("ordering", "research", "desc");
    let skills = skills_dir.load();
    let state = SharedState::new();

    let prompt = capture_system_prompt(|provider| {
        SubAgentTool::from_provider("researcher", provider, ModelConfig::mock())
            .with_system_prompt("Base prompt.")
            .with_skills(skills)
            .with_shared_state(state)
    })
    .await;

    let skills_at = prompt
        .find("<available_skills>")
        .expect("skills index present");
    let shared_at = prompt
        .find("## Shared State")
        .expect("shared-state block present");
    assert!(
        skills_at < shared_at,
        "skills index should precede the shared-state block, got: {prompt}"
    );
}

// ---------------------------------------------------------------------------
// Integration: sub-agent tool in a parent agent loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sub_agent_in_parent_loop() {
    // Parent calls sub-agent, sub-agent returns text, parent summarizes
    let sub_provider = Arc::new(MockProvider::text("42 is the answer"));

    let sub_agent = SubAgentTool::from_provider("calculator", sub_provider, ModelConfig::mock())
        .with_description("Calculates things");

    let parent_provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "calculator".into(),
            arguments: serde_json::json!({"task": "What is 6*7?"}),
        }]),
        MockResponse::Text("The calculator says: 42 is the answer".into()),
    ]);

    let config = make_config(parent_provider);

    let mut context = AgentContext {
        system_prompt: "You are a coordinator.".into(),
        messages: Vec::new(),
        tools: vec![Box::new(sub_agent)],
    };

    let prompt = AgentMessage::Llm(Message::user("What is 6*7?"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    // Should have: user, assistant(tool_call), toolResult, assistant(text)
    assert_eq!(new_messages.len(), 4);
    assert_eq!(new_messages[0].role(), "user");
    assert_eq!(new_messages[1].role(), "assistant");
    assert_eq!(new_messages[2].role(), "toolResult");
    assert_eq!(new_messages[3].role(), "assistant");

    // Tool result should contain sub-agent's output
    if let AgentMessage::Llm(Message::ToolResult { content, .. }) = &new_messages[2] {
        let text = match &content[0] {
            Content::Text { text } => text.as_str(),
            _ => panic!("Expected text content"),
        };
        assert_eq!(text, "42 is the answer");
    } else {
        panic!("Expected tool result message");
    }

    // Should have tool execution events
    let has_tool_start = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "calculator"));
    let has_tool_end = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "calculator"));
    assert!(has_tool_start);
    assert!(has_tool_end);
}

// ---------------------------------------------------------------------------
// Config plumbing: temperature and env-var key fallback reach the provider
// ---------------------------------------------------------------------------

/// Records the api_key and temperature each stream call receives.
struct StreamConfigCapture {
    captured: Arc<std::sync::Mutex<(String, Option<f32>)>>,
}

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for StreamConfigCapture {
    async fn stream(
        &self,
        config: yoagent::provider::StreamConfig,
        tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        *self.captured.lock().unwrap() = (config.api_key.clone(), config.temperature);
        let msg = Message::assistant(
            vec![Content::Text {
                text: "done".into(),
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

async fn run_sub_agent(tool: &SubAgentTool) {
    tool.execute(
        serde_json::json!({"task": "go"}),
        ToolContext::new("tc-cfg", "cfg"),
    )
    .await
    .expect("sub-agent should succeed");
}

#[tokio::test]
async fn test_sub_agent_temperature_reaches_provider() {
    let captured = Arc::new(std::sync::Mutex::new((String::new(), None)));
    let tool = SubAgentTool::from_provider(
        "cfg",
        Arc::new(StreamConfigCapture {
            captured: captured.clone(),
        }),
        ModelConfig::mock(),
    )
    .with_temperature(0.3);

    run_sub_agent(&tool).await;
    assert_eq!(captured.lock().unwrap().1, Some(0.3));
}

#[tokio::test]
async fn test_sub_agent_env_key_fallback() {
    // Own env var (not shared with other tests) to stay race-free under
    // parallel execution.
    std::env::set_var("MINIMAX_API_KEY", "minimax-env-key");
    let captured = Arc::new(std::sync::Mutex::new((String::new(), None)));
    let tool = SubAgentTool::from_provider(
        "cfg",
        Arc::new(StreamConfigCapture {
            captured: captured.clone(),
        }),
        yoagent::provider::ModelConfig::custom(
            yoagent::provider::ApiProtocol::OpenAiCompletions,
            "minimax",
            "http://localhost:8080/v1",
            "m",
            "M",
        ),
    );

    run_sub_agent(&tool).await;
    assert_eq!(captured.lock().unwrap().0, "minimax-env-key");
}

#[tokio::test]
async fn test_sub_agent_from_provider_construction() {
    // from_provider + ModelConfig::mock() mirrors Agent's construction path.
    let tool = SubAgentTool::from_provider(
        "researcher",
        Arc::new(MockProvider::text("Research result")),
        yoagent::provider::ModelConfig::mock(),
    )
    .with_description("Researches topics");

    let result = tool
        .execute(
            serde_json::json!({"task": "go"}),
            ToolContext::new("tc-fp", "researcher"),
        )
        .await
        .expect("sub-agent should succeed");
    let text = match &result.content[0] {
        Content::Text { text } => text,
        other => panic!("expected text, got {other:?}"),
    };
    assert!(text.contains("Research result"));
}

#[test]
fn test_sub_agent_from_config_wires_model() {
    // from_config selects a built-in provider from config.api and sets the id.
    let tool = SubAgentTool::from_config(
        "analyst",
        yoagent::provider::ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"),
    );
    assert_eq!(tool.name(), "analyst");
}

#[test]
fn test_sub_agent_from_config_with_errors_on_empty_registry() {
    let registry = yoagent::provider::ProviderRegistry::new();
    let err = match SubAgentTool::from_config_with(
        &registry,
        "analyst",
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

// ---------------------------------------------------------------------------
// Tool middleware wiring: sub-agent's own tool calls are gated
// ---------------------------------------------------------------------------

struct DenyAll;

#[async_trait::async_trait]
impl yoagent::ToolMiddleware for DenyAll {
    async fn before_tool(&self, _call: &yoagent::ToolCallRequest<'_>) -> yoagent::ToolDecision {
        yoagent::ToolDecision::Deny("sub-agent policy".into())
    }
}

/// A tool that must never run under DenyAll.
struct MustNotRun {
    ran: Arc<std::sync::Mutex<bool>>,
}

#[async_trait::async_trait]
impl AgentTool for MustNotRun {
    fn name(&self) -> &str {
        "must_not_run"
    }
    fn label(&self) -> &str {
        "Must Not Run"
    }
    fn description(&self) -> &str {
        "test"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        *self.ran.lock().unwrap() = true;
        Ok(ToolResult {
            content: vec![Content::Text { text: "ran".into() }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_sub_agent_tool_middleware_denies() {
    let ran = Arc::new(std::sync::Mutex::new(false));
    let provider = Arc::new(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "must_not_run".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("finished".into()),
    ]));

    let tool =
        SubAgentTool::from_provider("gated", provider, yoagent::provider::ModelConfig::mock())
            .with_tools(vec![Arc::new(MustNotRun { ran: ran.clone() })])
            .with_tool_middleware(DenyAll);

    let result = tool
        .execute(
            serde_json::json!({"task": "go"}),
            ToolContext::new("tc-mw", "gated"),
        )
        .await
        .expect("sub-agent completes despite denial");

    assert!(
        !*ran.lock().unwrap(),
        "denied tool must not run in sub-agent"
    );
    // Sub-agent still produced its final text.
    let text = match &result.content[0] {
        Content::Text { text } => text,
        other => panic!("expected text, got {other:?}"),
    };
    assert!(text.contains("finished"));
}
