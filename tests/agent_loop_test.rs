//! Tests for the core agent loop using MockProvider.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use yoagent::agent_loop::{agent_loop, agent_loop_continue, AgentLoopConfig};
use yoagent::context::ExecutionLimits;
use yoagent::provider::mock::*;
use yoagent::provider::MockProvider;
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
        tool_output_sink: None,
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

#[tokio::test]
async fn test_simple_text_response() {
    let provider = MockProvider::text("Hello, world!");
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    // Should have: AgentStart, TurnStart, MessageStart(user), MessageEnd(user),
    //              MessageStart(assistant), MessageEnd(assistant), TurnEnd, AgentEnd
    let event_types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::AgentStart => "AgentStart",
            AgentEvent::AgentEnd { .. } => "AgentEnd",
            AgentEvent::TurnStart => "TurnStart",
            AgentEvent::TurnEnd { .. } => "TurnEnd",
            AgentEvent::MessageStart { .. } => "MessageStart",
            AgentEvent::MessageEnd { .. } => "MessageEnd",
            AgentEvent::MessageUpdate { .. } => "MessageUpdate",
            AgentEvent::ToolExecutionStart { .. } => "ToolExecStart",
            AgentEvent::ToolExecutionUpdate { .. } => "ToolExecUpdate",
            AgentEvent::ToolExecutionEnd { .. } => "ToolExecEnd",
            AgentEvent::ProgressMessage { .. } => "ProgressMessage",
            AgentEvent::InputRejected { .. } => "InputRejected",
            _ => "Unknown",
        })
        .collect();

    assert!(event_types.contains(&"AgentStart"));
    assert!(event_types.contains(&"AgentEnd"));
    assert!(event_types.contains(&"TurnStart"));
    assert!(event_types.contains(&"TurnEnd"));

    // new_messages should contain user prompt + assistant response
    assert_eq!(new_messages.len(), 2);
    assert_eq!(new_messages[0].role(), "user");
    assert_eq!(new_messages[1].role(), "assistant");

    // Context should have both messages
    assert_eq!(context.messages.len(), 2);
}

#[tokio::test]
async fn test_tool_call_and_response() {
    // Mock: first call returns tool use, second returns text
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "test.txt"}),
        }]),
        MockResponse::Text("The file contains: hello".into()),
    ]);

    // Define a simple tool
    struct ReadFileTool;

    #[async_trait::async_trait]
    impl AgentTool for ReadFileTool {
        fn name(&self) -> &str {
            "read_file"
        }
        fn label(&self) -> &str {
            "Read File"
        }
        fn description(&self) -> &str {
            "Read a file"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                }
            })
        }
        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![Content::Text {
                    text: "hello".into(),
                }],
                details: serde_json::Value::Null,
            })
        }
    }

    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "You are helpful.".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ReadFileTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("Read test.txt"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    let event_types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::AgentStart => "AgentStart",
            AgentEvent::AgentEnd { .. } => "AgentEnd",
            AgentEvent::TurnStart => "TurnStart",
            AgentEvent::TurnEnd { .. } => "TurnEnd",
            AgentEvent::MessageStart { .. } => "MessageStart",
            AgentEvent::MessageEnd { .. } => "MessageEnd",
            AgentEvent::MessageUpdate { .. } => "MessageUpdate",
            AgentEvent::ToolExecutionStart { .. } => "ToolExecStart",
            AgentEvent::ToolExecutionUpdate { .. } => "ToolExecUpdate",
            AgentEvent::ToolExecutionEnd { .. } => "ToolExecEnd",
            AgentEvent::ProgressMessage { .. } => "ProgressMessage",
            AgentEvent::InputRejected { .. } => "InputRejected",
            _ => "Unknown",
        })
        .collect();

    // Should have tool execution events
    assert!(event_types.contains(&"ToolExecStart"));
    assert!(event_types.contains(&"ToolExecEnd"));

    // Messages: user, assistant(tool_call), toolResult, assistant(text)
    assert_eq!(new_messages.len(), 4);
    assert_eq!(new_messages[0].role(), "user");
    assert_eq!(new_messages[1].role(), "assistant");
    assert_eq!(new_messages[2].role(), "toolResult");
    assert_eq!(new_messages[3].role(), "assistant");
}

#[tokio::test]
async fn test_abort_cancels_loop() {
    // Provider that returns text — but we cancel before it runs
    let provider = MockProvider::text("Should not appear");
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    // Cancel immediately
    cancel.cancel();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Should have user message but loop should exit early
    // The prompt is added before the loop checks cancellation
    assert!(new_messages.len() <= 2); // user + possibly error
}

#[tokio::test]
async fn test_continue_from_tool_result() {
    let provider = MockProvider::text("Done processing.");
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: vec![
            AgentMessage::Llm(Message::user("do something")),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: "tc-1".into(),
                tool_name: "test_tool".into(),
                content: vec![Content::Text {
                    text: "result".into(),
                }],
                is_error: false,
                timestamp: 0,
            }),
        ],
        tools: Vec::new(),
    };

    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop_continue(&mut context, &config, tx, cancel).await;

    assert!(!new_messages.is_empty());
    assert_eq!(new_messages[0].role(), "assistant");
}

#[tokio::test]
async fn test_tool_error_is_reported() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "failing_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("Tool failed, sorry.".into()),
    ]);

    struct FailingTool;

    #[async_trait::async_trait]
    impl AgentTool for FailingTool {
        fn name(&self) -> &str {
            "failing_tool"
        }
        fn label(&self) -> &str {
            "Failing Tool"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Err(ToolError::Failed("Something went wrong".into()))
        }
    }

    let config = make_config(provider);
    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(FailingTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("Use the tool"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    // Tool error should be reported
    let tool_end_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. }))
        .collect();
    assert_eq!(tool_end_events.len(), 1);

    // Should still get a final assistant response
    assert_eq!(new_messages.last().unwrap().role(), "assistant");
}

#[tokio::test]
async fn test_unknown_tool_reports_error() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("I couldn't find that tool.".into()),
    ]);

    let config = make_config(provider);
    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(), // No tools registered
    };

    let prompt = AgentMessage::Llm(Message::user("Use nonexistent tool"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let _new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);
    let tool_errors: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. }))
        .collect();
    assert_eq!(tool_errors.len(), 1);
}

// ---------------------------------------------------------------------------
// Parallel tool execution tests
// ---------------------------------------------------------------------------

/// A tool that records execution timestamps to verify parallelism.
struct TimedTool {
    name: String,
    delay_ms: u64,
}

#[async_trait::async_trait]
impl AgentTool for TimedTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn label(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "Timed tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("done:{}", self.name),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_parallel_tool_execution_faster_than_sequential() {
    // 3 tools each taking 50ms. Sequential = 150ms+, Parallel = ~50ms.
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "tool_a".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_b".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_c".into(),
                arguments: serde_json::json!({}),
            },
        ]),
        MockResponse::Text("All done.".into()),
    ]);

    let mut config = make_config(provider);
    config.tool_execution = ToolExecutionStrategy::Parallel;

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![
            Box::new(TimedTool {
                name: "tool_a".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_b".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_c".into(),
                delay_ms: 50,
            }),
        ],
    };

    let prompt = AgentMessage::Llm(Message::user("Run all tools"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let elapsed = start.elapsed();

    let events = collect_events(rx);

    // All 3 tool results should be present
    let tool_results: Vec<_> = new_messages
        .iter()
        .filter(|m| m.role() == "toolResult")
        .collect();
    assert_eq!(tool_results.len(), 3);

    // Should complete in roughly 50-100ms, not 150ms+
    assert!(
        elapsed.as_millis() < 130,
        "Parallel execution took {}ms, expected <130ms",
        elapsed.as_millis()
    );

    // Should have 3 ToolExecutionStart and 3 ToolExecutionEnd events
    let starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
        .count();
    let ends = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .count();
    assert_eq!(starts, 3);
    assert_eq!(ends, 3);
}

#[tokio::test]
async fn test_sequential_tool_execution_is_slower() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "tool_a".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_b".into(),
                arguments: serde_json::json!({}),
            },
        ]),
        MockResponse::Text("Done.".into()),
    ]);

    let mut config = make_config(provider);
    config.tool_execution = ToolExecutionStrategy::Sequential;

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![
            Box::new(TimedTool {
                name: "tool_a".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_b".into(),
                delay_ms: 50,
            }),
        ],
    };

    let prompt = AgentMessage::Llm(Message::user("Run tools"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let _new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let elapsed = start.elapsed();

    // Sequential should take 100ms+ (2 × 50ms)
    assert!(
        elapsed.as_millis() >= 95,
        "Sequential execution took {}ms, expected >=95ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn test_batched_tool_execution() {
    // 4 tools, batch size 2: two batches of 2
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "tool_a".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_b".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_c".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "tool_d".into(),
                arguments: serde_json::json!({}),
            },
        ]),
        MockResponse::Text("All done.".into()),
    ]);

    let mut config = make_config(provider);
    config.tool_execution = ToolExecutionStrategy::Batched { size: 2 };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![
            Box::new(TimedTool {
                name: "tool_a".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_b".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_c".into(),
                delay_ms: 50,
            }),
            Box::new(TimedTool {
                name: "tool_d".into(),
                delay_ms: 50,
            }),
        ],
    };

    let prompt = AgentMessage::Llm(Message::user("Run all tools"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let start = std::time::Instant::now();
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let elapsed = start.elapsed();

    let _events = collect_events(rx);

    // All 4 results present
    let tool_results: Vec<_> = new_messages
        .iter()
        .filter(|m| m.role() == "toolResult")
        .collect();
    assert_eq!(tool_results.len(), 4);

    // 2 batches × 50ms = ~100ms (not 200ms sequential, not 50ms full parallel)
    assert!(
        elapsed.as_millis() >= 90 && elapsed.as_millis() < 160,
        "Batched execution took {}ms, expected 90-160ms",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// Streaming tool output (on_update callback) tests
// ---------------------------------------------------------------------------

/// A tool that emits progress updates via on_update callback.
struct ProgressTool;

#[async_trait::async_trait]
impl AgentTool for ProgressTool {
    fn name(&self) -> &str {
        "progress_tool"
    }
    fn label(&self) -> &str {
        "Progress"
    }
    fn description(&self) -> &str {
        "A tool that streams progress"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    async fn execute(
        &self,
        _params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        for i in 1..=3 {
            if let Some(ref cb) = ctx.on_update {
                cb(ToolResult {
                    content: vec![Content::Text {
                        text: format!("step {}/3", i),
                    }],
                    details: serde_json::Value::Null,
                });
            }
        }
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "done".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_tool_execution_update_events_emitted() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("All done.".into()),
    ]);

    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ProgressTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    let updates: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
                if let Some(Content::Text { text }) = partial_result.content.first() {
                    Some(text.clone())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    assert_eq!(updates, vec!["step 1/3", "step 2/3", "step 3/3"]);
}

// ---------------------------------------------------------------------------
// Retry with backoff tests
// ---------------------------------------------------------------------------

/// A provider that fails N times with a given error, then delegates to a MockProvider.
struct FailThenSucceedProvider {
    fail_count: std::sync::atomic::AtomicUsize,
    max_failures: usize,
    error: ProviderError,
    inner: MockProvider,
}

use yoagent::provider::{ProviderError, StreamConfig, StreamEvent, StreamProvider};

struct UsageProvider {
    usage: Usage,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl StreamProvider for UsageProvider {
    async fn stream(
        &self,
        _config: StreamConfig,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<yoagent::Message, ProviderError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let message = Message::assistant(
            vec![Content::Text {
                text: "usage response".into(),
            }],
            StopReason::Stop,
            "usage-test",
            "usage-test",
            self.usage.clone(),
        );

        let _ = tx.send(StreamEvent::Start);
        let _ = tx.send(StreamEvent::Done {
            message: message.clone(),
        });
        Ok(message)
    }
}

#[tokio::test]
async fn test_execution_limit_counts_cached_tokens() {
    let provider = std::sync::Arc::new(UsageProvider {
        usage: Usage {
            input: 1,
            output: 1,
            cache_read: 99,
            cache_write: 0,
            total_tokens: 101,
        },
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let provider_for_config: std::sync::Arc<dyn StreamProvider> = provider.clone();

    let config = AgentLoopConfig {
        provider: provider_for_config,
        model: "usage-test".into(),
        api_key: "test".into(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: None,
        temperature: None,
        model_config: None,
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: Some(Box::new(|| {
            vec![AgentMessage::Llm(Message::user("follow up"))]
        })),
        context_config: None,
        compaction_strategy: None,
        execution_limits: Some(ExecutionLimits {
            max_turns: 50,
            max_total_tokens: 100,
            max_duration: std::time::Duration::from_secs(60),
            ..Default::default()
        }),
        cache_config: CacheConfig::default(),
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::default(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(
        vec![AgentMessage::Llm(Message::user("start"))],
        &mut context,
        &config,
        tx,
        cancel,
    )
    .await;

    assert_eq!(
        provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "cached prompt tokens should trip the limit before a second provider call"
    );
    assert!(new_messages.iter().any(|msg| {
        matches!(
            msg,
            AgentMessage::Llm(Message::User { content, .. })
                if content.iter().any(|c| matches!(
                    c,
                    Content::Text { text } if text.contains("[Agent stopped: Max tokens reached")
                ))
        )
    }));
}

#[async_trait::async_trait]
impl StreamProvider for FailThenSucceedProvider {
    async fn stream(
        &self,
        config: StreamConfig,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<yoagent::Message, ProviderError> {
        let attempt = self
            .fail_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if attempt < self.max_failures {
            return Err(match &self.error {
                ProviderError::RateLimited { retry_after_ms } => ProviderError::RateLimited {
                    retry_after_ms: *retry_after_ms,
                },
                ProviderError::Network(msg) => ProviderError::Network(msg.clone()),
                ProviderError::Auth(msg) => ProviderError::Auth(msg.clone()),
                other => ProviderError::Other(other.to_string()),
            });
        }
        self.inner.stream(config, tx, cancel).await
    }
}

#[tokio::test]
async fn test_retry_on_rate_limit_succeeds() {
    let provider: std::sync::Arc<FailThenSucceedProvider> =
        std::sync::Arc::new(FailThenSucceedProvider {
            fail_count: std::sync::atomic::AtomicUsize::new(0),
            max_failures: 2,
            error: ProviderError::RateLimited {
                retry_after_ms: Some(10), // 10ms for fast tests
            },
            inner: MockProvider::text("Success after retries"),
        });

    let config = AgentLoopConfig {
        provider: provider.clone(),
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
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            backoff_multiplier: 2.0,
            max_delay_ms: 100,
        },
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Should have succeeded after 2 failures + 1 success
    assert_eq!(new_messages.len(), 2); // user + assistant
    let events = collect_events(rx);
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));

    // Verify the provider was called 3 times (2 failures + 1 success)
    assert_eq!(
        provider
            .fail_count
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

#[tokio::test]
async fn test_retry_exhausted_returns_error() {
    let provider: std::sync::Arc<FailThenSucceedProvider> =
        std::sync::Arc::new(FailThenSucceedProvider {
            fail_count: std::sync::atomic::AtomicUsize::new(0),
            max_failures: 10, // more failures than retries
            error: ProviderError::Network("connection reset".into()),
            inner: MockProvider::text("never reached"),
        });

    let config = AgentLoopConfig {
        provider: provider.clone(),
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
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig {
            max_retries: 2,
            initial_delay_ms: 10,
            backoff_multiplier: 2.0,
            max_delay_ms: 100,
        },
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Should have an error message (StopReason::Error)
    let last = new_messages.last().unwrap();
    if let AgentMessage::Llm(Message::Assistant {
        stop_reason,
        error_message,
        ..
    }) = last
    {
        assert_eq!(*stop_reason, StopReason::Error);
        assert!(error_message.as_ref().unwrap().contains("connection reset"));
    } else {
        panic!("Expected error assistant message");
    }

    // 1 initial + 2 retries = 3 attempts
    assert_eq!(
        provider
            .fail_count
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

#[tokio::test]
async fn test_no_retry_on_auth_error() {
    let provider: std::sync::Arc<FailThenSucceedProvider> =
        std::sync::Arc::new(FailThenSucceedProvider {
            fail_count: std::sync::atomic::AtomicUsize::new(0),
            max_failures: 1,
            error: ProviderError::Auth("invalid key".into()),
            inner: MockProvider::text("never reached"),
        });

    let config = AgentLoopConfig {
        provider: provider.clone(),
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
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::default(), // 3 retries, but auth is not retryable
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Should have been called exactly once — no retries for auth errors
    assert_eq!(
        provider
            .fail_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn test_retry_none_disables_retries() {
    let provider: std::sync::Arc<FailThenSucceedProvider> =
        std::sync::Arc::new(FailThenSucceedProvider {
            fail_count: std::sync::atomic::AtomicUsize::new(0),
            max_failures: 1,
            error: ProviderError::RateLimited {
                retry_after_ms: None,
            },
            inner: MockProvider::text("never reached"),
        });

    let config = AgentLoopConfig {
        provider: provider.clone(),
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
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::none(), // disabled
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Only 1 attempt — no retries
    assert_eq!(
        provider
            .fail_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

// ---------------------------------------------------------------------------
// Event streaming bug fix test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_message_update_events_emitted_during_streaming() {
    // This test verifies the fix for: text deltas not emitted because
    // partial_message was None when deltas arrived (MessageStart was only
    // emitted on Done, after all deltas had already been processed).
    let provider = MockProvider::text("Hello, world!");
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let events = collect_events(rx);

    // Collect MessageUpdate text deltas
    let deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageUpdate {
                delta: StreamDelta::Text { delta },
                ..
            } => Some(delta.clone()),
            _ => None,
        })
        .collect();

    // Should have at least one text delta with "Hello, world!"
    assert!(
        !deltas.is_empty(),
        "Expected MessageUpdate events with text deltas, got none"
    );
    let full_text: String = deltas.into_iter().collect();
    assert_eq!(full_text, "Hello, world!");

    // Verify event ordering: MessageStart before MessageUpdate before MessageEnd
    let event_types: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageStart { .. } => Some("Start"),
            AgentEvent::MessageUpdate { .. } => Some("Update"),
            AgentEvent::MessageEnd { .. } => Some("End"),
            _ => None,
        })
        .collect();

    // Should be: Start (user), End (user), Start (assistant), Update(s), End (assistant)
    // Find the assistant sequence
    let assistant_start = event_types.iter().rposition(|&e| e == "Start").unwrap();
    let assistant_end = event_types.iter().rposition(|&e| e == "End").unwrap();

    // All Updates should be between the last Start and last End
    for (i, &et) in event_types.iter().enumerate() {
        if et == "Update" {
            assert!(
                i > assistant_start && i < assistant_end,
                "MessageUpdate at index {} should be between MessageStart ({}) and MessageEnd ({})",
                i,
                assistant_start,
                assistant_end
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle callback tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_before_turn_can_abort() {
    // Provider with 5 text responses, but before_turn aborts after 2 turns.
    // We need tool calls to keep the loop going for multiple turns.
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        // These should never be reached
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("Final".into()),
    ]);

    let turn_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let turn_count_clone = turn_count.clone();

    let mut config = make_config(provider);
    config.before_turn = Some(std::sync::Arc::new(move |_msgs, _turn| {
        let count = turn_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        count < 2 // Allow turns 0 and 1, abort on turn 2
    }));

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ProgressTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // before_turn was called 3 times (allowed 0, allowed 1, rejected 2)
    assert_eq!(turn_count.load(std::sync::atomic::Ordering::SeqCst), 3);

    // Only 2 assistant messages should be produced
    let assistant_count = new_messages
        .iter()
        .filter(|m| m.role() == "assistant")
        .count();
    assert_eq!(assistant_count, 2);
}

#[tokio::test]
async fn test_after_turn_receives_messages() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("Done.".into()),
    ]);

    let message_counts: std::sync::Arc<std::sync::Mutex<Vec<usize>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let counts_clone = message_counts.clone();

    let mut config = make_config(provider);
    config.after_turn = Some(std::sync::Arc::new(move |msgs, _usage| {
        counts_clone.lock().unwrap().push(msgs.len());
    }));

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ProgressTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let counts = message_counts.lock().unwrap();
    // after_turn called twice (one per LLM response)
    assert_eq!(counts.len(), 2);
    // Message count should increase between calls
    assert!(counts[1] > counts[0], "counts: {:?}", *counts);
}

#[tokio::test]
async fn test_on_error_fires_on_provider_error() {
    let provider = FailThenSucceedProvider {
        fail_count: std::sync::atomic::AtomicUsize::new(0),
        max_failures: 10, // more failures than retries
        error: ProviderError::Network("connection reset".into()),
        inner: MockProvider::text("never reached"),
    };

    let error_msgs: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let error_msgs_clone = error_msgs.clone();

    let config = AgentLoopConfig {
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
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::none(),
        before_turn: None,
        after_turn: None,
        on_error: Some(std::sync::Arc::new(move |err| {
            error_msgs_clone.lock().unwrap().push(err.to_string());
        })),
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let errors = error_msgs.lock().unwrap();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("connection reset"), "got: {}", errors[0]);
}

#[tokio::test]
async fn test_callbacks_are_optional() {
    // Verify the loop works fine with all callbacks set to None (same as before)
    let provider = MockProvider::text("Hello!");
    let config = make_config(provider);
    // make_config already sets all callbacks to None

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    assert_eq!(new_messages.len(), 2);
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));
}

// ---------------------------------------------------------------------------
// ProgressMessage tests (Addition 1)
// ---------------------------------------------------------------------------

/// A tool that calls on_progress to emit user-facing progress messages.
struct ProgressMessageTool;

#[async_trait::async_trait]
impl AgentTool for ProgressMessageTool {
    fn name(&self) -> &str {
        "progress_msg_tool"
    }
    fn label(&self) -> &str {
        "ProgressMsg"
    }
    fn description(&self) -> &str {
        "Emits progress messages"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        if let Some(ref progress) = ctx.on_progress {
            progress("Working...".into());
        }
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "done".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_progress_message_event_emitted() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_msg_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("ok".into()),
    ]);
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ProgressMessageTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    let progress_msgs: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ProgressMessage {
                tool_call_id,
                tool_name,
                text,
            } => Some((tool_call_id.clone(), tool_name.clone(), text.clone())),
            _ => None,
        })
        .collect();

    assert_eq!(progress_msgs.len(), 1);
    assert_eq!(progress_msgs[0].1, "progress_msg_tool");
    assert_eq!(progress_msgs[0].2, "Working...");
}

/// A tool that does NOT call on_progress — should cause no panics, no events.
struct SilentTool;

#[async_trait::async_trait]
impl AgentTool for SilentTool {
    fn name(&self) -> &str {
        "silent_tool"
    }
    fn label(&self) -> &str {
        "Silent"
    }
    fn description(&self) -> &str {
        "Does not call progress"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Intentionally ignores on_progress
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "quiet".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_tool_ignoring_progress_no_panic() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "silent_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("ok".into()),
    ]);
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(SilentTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    // No ProgressMessage events
    let progress_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ProgressMessage { .. }))
        .count();
    assert_eq!(progress_count, 0);
}

/// Two parallel tools both emit progress — events are distinguishable by tool_call_id.
struct NamedProgressTool {
    tool_name: String,
}

#[async_trait::async_trait]
impl AgentTool for NamedProgressTool {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn label(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        "Named progress tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        if let Some(ref progress) = ctx.on_progress {
            progress(format!("progress from {}", self.tool_name));
        }
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("done:{}", self.tool_name),
            }],
            details: serde_json::Value::Null,
        })
    }
}

#[tokio::test]
async fn test_parallel_tools_progress_distinguishable() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "pa".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "pb".into(),
                arguments: serde_json::json!({}),
            },
        ]),
        MockResponse::Text("done".into()),
    ]);
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![
            Box::new(NamedProgressTool {
                tool_name: "pa".into(),
            }),
            Box::new(NamedProgressTool {
                tool_name: "pb".into(),
            }),
        ],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    let progress_msgs: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ProgressMessage {
                tool_name, text, ..
            } => Some((tool_name.clone(), text.clone())),
            _ => None,
        })
        .collect();

    assert_eq!(progress_msgs.len(), 2);
    let names: Vec<&str> = progress_msgs.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"pa"));
    assert!(names.contains(&"pb"));
}

#[tokio::test]
async fn test_on_update_still_works_after_refactor() {
    // Existing ProgressTool uses on_update (not on_progress) — ensure it still works.
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "progress_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("ok".into()),
    ]);
    let config = make_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(ProgressTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("go"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    let updates: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
                if let Some(Content::Text { text }) = partial_result.content.first() {
                    Some(text.clone())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    assert_eq!(updates, vec!["step 1/3", "step 2/3", "step 3/3"]);
}

// ---------------------------------------------------------------------------
// InputFilter tests (Addition 2)
// ---------------------------------------------------------------------------

use std::sync::Arc;

struct PassFilter;
impl InputFilter for PassFilter {
    fn filter(&self, _text: &str) -> FilterResult {
        FilterResult::Pass
    }
}

struct WarnFilter {
    warning: String,
}
impl InputFilter for WarnFilter {
    fn filter(&self, _text: &str) -> FilterResult {
        FilterResult::Warn(self.warning.clone())
    }
}

struct RejectFilter {
    reason: String,
}
impl InputFilter for RejectFilter {
    fn filter(&self, _text: &str) -> FilterResult {
        FilterResult::Reject(self.reason.clone())
    }
}

#[tokio::test]
async fn test_filter_pass_message_goes_through() {
    let provider = MockProvider::text("Hello!");
    let mut config = make_config(provider);
    config.input_filters = vec![Arc::new(PassFilter)];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    // Message went through normally
    assert_eq!(new_messages.len(), 2); // user + assistant
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));
}

#[tokio::test]
async fn test_filter_warn_injects_warning_message() {
    let provider = MockProvider::text("Got it.");
    let mut config = make_config(provider);
    config.input_filters = vec![Arc::new(WarnFilter {
        warning: "danger".into(),
    })];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // user (with appended warning) + assistant = 2
    assert_eq!(new_messages.len(), 2);
    // The warning should be appended to the user message's content
    if let AgentMessage::Llm(Message::User { content, .. }) = &new_messages[0] {
        assert_eq!(content.len(), 2, "expected original text + warning");
        let warning = match &content[1] {
            Content::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(warning.contains("[Warning: danger]"), "got: {}", warning);
    } else {
        panic!("Expected user message at index 0");
    }
}

#[tokio::test]
async fn test_filter_reject_returns_empty() {
    let provider = MockProvider::text("Should not reach");
    let mut config = make_config(provider);
    config.input_filters = vec![Arc::new(RejectFilter {
        reason: "blocked".into(),
    })];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Bad input"));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;
    let events = collect_events(rx);

    // Rejected — empty messages returned
    assert!(new_messages.is_empty());
    // Context should NOT contain the rejected prompt
    assert!(
        context.messages.is_empty(),
        "Rejected prompts should not leak into context, got {} messages",
        context.messages.len()
    );
    // InputRejected event should carry the reason
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::InputRejected { reason } if reason == "blocked")));
    // AgentStart + InputRejected + AgentEnd
    assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentStart)));
    assert!(events
        .iter()
        .any(|e| matches!(e, AgentEvent::AgentEnd { messages, .. } if messages.is_empty())));
}

#[tokio::test]
async fn test_filter_chain_first_reject_wins() {
    let provider = MockProvider::text("Should not reach");
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    struct CountingRejectFilter {
        counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl InputFilter for CountingRejectFilter {
        fn filter(&self, _text: &str) -> FilterResult {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            FilterResult::Reject("first rejects".into())
        }
    }

    struct NeverCalledFilter {
        counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl InputFilter for NeverCalledFilter {
        fn filter(&self, _text: &str) -> FilterResult {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            FilterResult::Pass
        }
    }

    let count2 = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut config = make_config(provider);
    config.input_filters = vec![
        Arc::new(CountingRejectFilter {
            counter: call_count.clone(),
        }),
        Arc::new(NeverCalledFilter {
            counter: count2.clone(),
        }),
    ];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Bad"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    assert!(new_messages.is_empty());
    // First filter was called
    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    // Second filter was NOT called (first reject short-circuits)
    assert_eq!(count2.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_filter_multiple_warns_accumulate() {
    let provider = MockProvider::text("Got warnings.");
    let mut config = make_config(provider);
    config.input_filters = vec![
        Arc::new(WarnFilter {
            warning: "warn1".into(),
        }),
        Arc::new(WarnFilter {
            warning: "warn2".into(),
        }),
    ];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("Hi"));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // user (with appended warnings) + assistant = 2
    assert_eq!(new_messages.len(), 2);
    if let AgentMessage::Llm(Message::User { content, .. }) = &new_messages[0] {
        // Original text + appended warning block
        assert!(content.len() >= 2, "expected original text + warning");
        let warning = match content.last().unwrap() {
            Content::Text { text } => text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(warning.contains("[Warning: warn1]"), "got: {}", warning);
        assert!(warning.contains("[Warning: warn2]"), "got: {}", warning);
    } else {
        panic!("Expected user message");
    }
}

#[tokio::test]
async fn test_filter_non_text_content_only_text_extracted() {
    // User message with Image content — filter should receive only text portions
    let provider = MockProvider::text("Ok");

    let call_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let call_text_clone = call_text.clone();

    struct CapturingFilter {
        captured: std::sync::Arc<std::sync::Mutex<String>>,
    }
    impl InputFilter for CapturingFilter {
        fn filter(&self, text: &str) -> FilterResult {
            *self.captured.lock().unwrap() = text.to_string();
            FilterResult::Pass
        }
    }

    let mut config = make_config(provider);
    config.input_filters = vec![Arc::new(CapturingFilter {
        captured: call_text_clone,
    })];

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::User {
        content: vec![
            Content::Text {
                text: "Check this image".into(),
            },
            Content::Image {
                data: "base64data".into(),
                mime_type: "image/png".into(),
            },
        ],
        timestamp: 0,
    });
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let captured = call_text.lock().unwrap();
    // Filter should have received only the text portion
    assert_eq!(*captured, "Check this image");
}

// ---------------------------------------------------------------------------
// CompactionStrategy tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_default_compaction_matches_compact_messages() {
    use yoagent::context::{compact_messages, ContextConfig, DefaultCompaction};
    use yoagent::CompactionStrategy;

    let mut messages = Vec::new();
    for i in 0..100 {
        messages.push(AgentMessage::Llm(Message::user(format!(
            "Message {} {}",
            i,
            "x".repeat(200)
        ))));
    }

    let config = ContextConfig {
        max_context_tokens: 500,
        system_prompt_tokens: 100,
        keep_recent: 5,
        keep_first: 2,
        tool_output_max_lines: 20,
        ..Default::default()
    };

    let result_direct = compact_messages(messages.clone(), &config);
    let result_trait = DefaultCompaction.compact(messages, &config);

    // Compare lengths and structure, not deep equality — Level 3 compaction
    // inserts marker messages with now_ms() timestamps that differ between calls.
    assert_eq!(result_direct.len(), result_trait.len());
    assert!(
        result_direct.len() < 100,
        "compaction should have reduced messages"
    );
    assert!(
        result_direct.len() >= 2,
        "should keep at least keep_first messages"
    );
}

#[tokio::test]
async fn test_custom_compaction_strategy_is_called() {
    use yoagent::context::ContextConfig;
    use yoagent::CompactionStrategy;

    /// A custom strategy that prepends a marker message, then delegates
    /// to the default compaction.
    struct MarkerCompaction;

    impl CompactionStrategy for MarkerCompaction {
        fn compact(
            &self,
            messages: Vec<AgentMessage>,
            _config: &ContextConfig,
        ) -> Vec<AgentMessage> {
            let mut result = vec![AgentMessage::Llm(Message::user("[compacted]"))];
            // Keep only the last message to prove we ran
            if let Some(last) = messages.last() {
                result.push(last.clone());
            }
            result
        }
    }

    // Provider returns a simple text response
    let provider = MockProvider::text("Got it.");

    let config = AgentLoopConfig {
        provider: std::sync::Arc::new(provider),
        model: "test".into(),
        api_key: "test".into(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: None,
        temperature: None,
        model_config: None,
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        context_config: Some(ContextConfig {
            max_context_tokens: 10, // Tiny budget to force compaction
            system_prompt_tokens: 0,
            keep_recent: 1,
            keep_first: 1,
            tool_output_max_lines: 10,
            ..Default::default()
        }),
        compaction_strategy: Some(std::sync::Arc::new(MarkerCompaction)),
        execution_limits: None,
        cache_config: CacheConfig::default(),
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::none(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let prompt = AgentMessage::Llm(Message::user("Hello"));
    let mut context = AgentContext {
        system_prompt: String::new(),
        messages: vec![],
        tools: vec![],
    };

    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // The custom strategy should have inserted "[compacted]" as the first message
    assert!(
        context.messages.iter().any(|m| {
            if let AgentMessage::Llm(Message::User { content, .. }) = m {
                content
                    .iter()
                    .any(|c| matches!(c, Content::Text { text } if text == "[compacted]"))
            } else {
                false
            }
        }),
        "Custom compaction marker not found in context: {:?}",
        context
            .messages
            .iter()
            .filter_map(|m| {
                if let AgentMessage::Llm(Message::User { content, .. }) = m {
                    Some(content)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_none_compaction_strategy_uses_default() {
    use yoagent::context::ContextConfig;

    // Provider returns a simple text response
    let provider = MockProvider::text("Got it.");

    let config = AgentLoopConfig {
        provider: std::sync::Arc::new(provider),
        model: "test".into(),
        api_key: "test".into(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: None,
        temperature: None,
        model_config: None,
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        context_config: Some(ContextConfig {
            max_context_tokens: 10, // Tiny budget to force compaction
            system_prompt_tokens: 0,
            keep_recent: 1,
            keep_first: 1,
            tool_output_max_lines: 10,
            ..Default::default()
        }),
        compaction_strategy: None, // Should fall back to DefaultCompaction
        execution_limits: None,
        cache_config: CacheConfig::default(),
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::none(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    };

    let prompt = AgentMessage::Llm(Message::user("Hello"));
    let mut context = AgentContext {
        system_prompt: String::new(),
        messages: vec![],
        tools: vec![],
    };

    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    // Should not panic — DefaultCompaction handles everything
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Agent should have produced at least the user message + assistant response
    assert!(
        !new_messages.is_empty(),
        "Agent should have produced messages"
    );
}

/// Tool calls with provider_metadata (e.g. Gemini thought signatures)
/// must still be executed by the agent loop.
#[tokio::test]
async fn test_tool_call_with_provider_metadata_executes() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "echo_tool".into(),
            arguments: serde_json::json!({"text": "hello"}),
            provider_metadata: Some(serde_json::json!({"thought_signature": "SIG_DATA"})),
        }]),
        MockResponse::Text("done".into()),
    ]);

    struct EchoTool;

    #[async_trait::async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
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

    let config = make_config(provider);
    let mut context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: vec![Box::new(EchoTool)],
    };

    let prompt = AgentMessage::Llm(Message::user("echo hello"));
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    // Tool should have been executed despite provider_metadata being Some
    let mut got_tool_start = false;
    let mut got_tool_end = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                assert_eq!(tool_name, "echo_tool");
                got_tool_start = true;
            }
            AgentEvent::ToolExecutionEnd { .. } => {
                got_tool_end = true;
            }
            _ => {}
        }
    }
    assert!(got_tool_start, "Tool should have been executed");
    assert!(got_tool_end, "Tool execution should have completed");

    // Should have: user, assistant (tool call), tool result, assistant (final)
    assert!(
        new_messages.len() >= 4,
        "Expected at least 4 messages, got {}",
        new_messages.len()
    );

    // Verify tool result contains echoed text
    let has_tool_result = new_messages.iter().any(|m| {
        if let AgentMessage::Llm(Message::ToolResult { content, .. }) = m {
            content
                .iter()
                .any(|c| matches!(c, Content::Text { text } if text == "hello"))
        } else {
            false
        }
    });
    assert!(has_tool_result, "Tool result should contain 'hello'");
}

// ---------------------------------------------------------------------------
// Budget calibration: measured overhead (system prompt, tool schemas, estimate
// shortfall) is subtracted from the compaction budget once real usage arrives
// ---------------------------------------------------------------------------

/// Records the ContextConfig each compact call receives; never modifies
/// messages, so the loop's behavior is otherwise unaffected.
struct RecordingCompaction {
    calls: std::sync::Mutex<Vec<(usize, usize)>>, // (max_context_tokens, system_prompt_tokens)
}

impl yoagent::CompactionStrategy for RecordingCompaction {
    fn compact(
        &self,
        messages: Vec<AgentMessage>,
        config: &yoagent::context::ContextConfig,
    ) -> Vec<AgentMessage> {
        self.calls
            .lock()
            .unwrap()
            .push((config.max_context_tokens, config.system_prompt_tokens));
        messages
    }
}

fn calibration_config(
    provider: std::sync::Arc<dyn StreamProvider>,
    strategy: std::sync::Arc<RecordingCompaction>,
    max_context_tokens: usize,
) -> AgentLoopConfig {
    AgentLoopConfig {
        provider,
        model: "usage-test".into(),
        api_key: "test".into(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: None,
        temperature: None,
        model_config: None,
        convert_to_llm: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: Some(Box::new(|| {
            vec![AgentMessage::Llm(Message::user("follow up"))]
        })),
        context_config: Some(yoagent::context::ContextConfig {
            max_context_tokens,
            system_prompt_tokens: 500,
            keep_recent: 1,
            keep_first: 1,
            tool_output_max_lines: 10,
            ..Default::default()
        }),
        compaction_strategy: Some(strategy),
        execution_limits: Some(ExecutionLimits {
            max_turns: 2,
            max_total_tokens: 1_000_000,
            max_duration: std::time::Duration::from_secs(60),
            ..Default::default()
        }),
        cache_config: CacheConfig::default(),
        tool_output_sink: None,
        output_schema: None,
        tool_execution: ToolExecutionStrategy::default(),
        retry_config: yoagent::RetryConfig::none(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        tool_middleware: vec![],
        turn_delay: None,
    }
}

async fn run_calibration_loop(max_context_tokens: usize) -> Vec<(usize, usize)> {
    // Real usage (5010 tokens) dwarfs the char-based estimate of the tiny
    // messages, so the measured overhead is ~5000 tokens.
    let provider = std::sync::Arc::new(UsageProvider {
        usage: Usage {
            input: 5000,
            output: 10,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 5010,
        },
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let strategy = std::sync::Arc::new(RecordingCompaction {
        calls: std::sync::Mutex::new(Vec::new()),
    });
    let config = calibration_config(provider, strategy.clone(), max_context_tokens);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    let (tx, _rx) = mpsc::unbounded_channel();
    agent_loop(
        vec![AgentMessage::Llm(Message::user("start"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    let calls = strategy.calls.lock().unwrap().clone();
    calls
}

#[tokio::test]
async fn test_calibration_subtracts_measured_overhead() {
    let calls = run_calibration_loop(8000).await;
    assert!(calls.len() >= 2, "expected 2 turns, got {:?}", calls);

    // Turn 1: no real usage yet — config passes through unchanged.
    assert_eq!(calls[0], (8000, 500));

    // Turn 2: usage anchored at ~5010 vs a tiny estimate, so ~5000 tokens of
    // overhead are subtracted and the static reserve is zeroed (the measured
    // overhead already includes the real system prompt).
    let (max, reserve) = calls[1];
    assert_eq!(reserve, 0);
    assert!(
        (2500..=3500).contains(&max),
        "expected ~3000 calibrated budget, got {}",
        max
    );
}

#[tokio::test]
async fn test_calibration_floor_prevents_budget_collapse() {
    // Overhead (~5000) exceeds the whole budget (4000). Without the floor the
    // budget would hit 0 and compaction would wipe the conversation; the
    // calibrated budget must never drop below 10% of the configured one.
    let calls = run_calibration_loop(4000).await;
    assert!(calls.len() >= 2, "expected 2 turns, got {:?}", calls);
    assert_eq!(calls[0], (4000, 500));
    assert_eq!(calls[1], (400, 0));
}

// ---------------------------------------------------------------------------
// Tool output capped on append (prefix-cache stability)
// ---------------------------------------------------------------------------

struct BigOutputTool;

#[async_trait::async_trait]
impl AgentTool for BigOutputTool {
    fn name(&self) -> &str {
        "big_output"
    }
    fn label(&self) -> &str {
        "Big output"
    }
    fn description(&self) -> &str {
        "Returns a long output"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let text = (0..500)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolResult {
            content: vec![Content::Text { text }],
            details: serde_json::Value::Null,
        })
    }
}

fn tool_then_text() -> MockProvider {
    MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "big_output".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ])
}

async fn run_with_context_config(
    config: Option<yoagent::context::ContextConfig>,
) -> (Vec<AgentMessage>, Vec<AgentEvent>) {
    let mut loop_config = make_config(tool_then_text());
    loop_config.context_config = config;

    let mut context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: vec![Box::new(BigOutputTool)],
    };
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &loop_config,
        tx,
        cancel,
    )
    .await;

    (context.messages, collect_events(rx))
}

fn tool_result_lines(messages: &[AgentMessage]) -> usize {
    messages
        .iter()
        .find_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult { content, .. }) => match content.first() {
                Some(Content::Text { text }) => Some(text.lines().count()),
                _ => None,
            },
            _ => None,
        })
        .expect("expected a tool result in context")
}

#[tokio::test]
async fn tool_output_is_capped_on_append_when_enabled() {
    let (messages, events) = run_with_context_config(Some(yoagent::context::ContextConfig {
        // Budget far above the output: without on-append capping nothing is
        // truncated, which is exactly the history compaction later rewrites.
        max_context_tokens: 1_000_000,
        system_prompt_tokens: 0,
        tool_output_max_lines: 25,
        truncate_tool_output_on_append: true,
        ..Default::default()
    }))
    .await;

    assert_eq!(tool_result_lines(&messages), 25);

    // The event stream still carries the untruncated output — the cap is a
    // context concern, not a tool-result one.
    let full = events.iter().any(|e| match e {
        AgentEvent::ToolExecutionEnd { result, .. } => match result.content.first() {
            Some(Content::Text { text }) => text.lines().count() == 500,
            _ => false,
        },
        _ => false,
    });
    assert!(full, "ToolExecutionEnd should carry the untruncated output");
}

#[tokio::test]
async fn tool_output_is_capped_on_append_by_default() {
    let (messages, _) = run_with_context_config(Some(yoagent::context::ContextConfig {
        max_context_tokens: 1_000_000,
        system_prompt_tokens: 0,
        tool_output_max_lines: 25,
        ..Default::default()
    }))
    .await;

    assert_eq!(
        tool_result_lines(&messages),
        25,
        "on-append capping is the default"
    );
}

#[tokio::test]
async fn on_append_capping_can_be_turned_off() {
    let (messages, _) = run_with_context_config(Some(yoagent::context::ContextConfig {
        max_context_tokens: 1_000_000,
        system_prompt_tokens: 0,
        tool_output_max_lines: 25,
        truncate_tool_output_on_append: false,
        ..Default::default()
    }))
    .await;

    assert_eq!(tool_result_lines(&messages), 500);
}

#[tokio::test]
async fn per_tool_override_exempts_a_tool_from_capping() {
    // A tool that head+tail would damage opts out by name rather than by
    // disabling capping for everything.
    let mut overrides = std::collections::HashMap::new();
    overrides.insert("big_output".to_string(), usize::MAX);

    let (messages, _) = run_with_context_config(Some(yoagent::context::ContextConfig {
        max_context_tokens: 1_000_000,
        system_prompt_tokens: 0,
        tool_output_max_lines: 25,
        tool_output_max_lines_overrides: overrides,
        ..Default::default()
    }))
    .await;

    assert_eq!(tool_result_lines(&messages), 500);
}

#[tokio::test]
async fn tool_output_is_untouched_without_a_context_config() {
    let (messages, _) = run_with_context_config(None).await;
    assert_eq!(tool_result_lines(&messages), 500);
}

// ---------------------------------------------------------------------------
// Session rollup on AgentEnd (issue #124)
// ---------------------------------------------------------------------------

fn usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
    }
}

fn agent_end_stats(events: &[AgentEvent]) -> SessionStats {
    events
        .iter()
        .find_map(|e| match e {
            AgentEvent::AgentEnd { stats, .. } => Some(stats.clone()),
            _ => None,
        })
        .expect("AgentEnd must be emitted")
}

/// Distinct usage per turn, so a rollup that copies the last turn instead of
/// summing cannot pass.
#[tokio::test]
async fn agent_end_carries_a_summed_session_rollup() {
    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "silent_tool".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::TextWithUsage("done".into(), usage(7, 70, 700, 7000)),
    ]);

    // The first turn's usage rides on the tool-call response, which
    // MockProvider reports as Usage::default() — so this asserts the loop sums
    // whatever each turn reported, zeros included.
    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(SilentTool)],
        system_prompt: String::new(),
    };
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &make_config(provider),
        tx,
        CancellationToken::new(),
    )
    .await;

    let stats = agent_end_stats(&collect_events(rx));
    assert_eq!(stats.turns, 2, "two LLM turns: tool call, then text");
    assert_eq!(stats.usage.input, 7);
    assert_eq!(stats.usage.output, 70);
    assert_eq!(stats.usage.cache_read, 700);
    assert_eq!(stats.usage.cache_write, 7000);

    // 700 / (7 + 700 + 7000) — cache_write counts against the rate, because
    // those are prompt tokens the provider processed and billed.
    let expected = 700.0 / 7707.0;
    assert!(
        (stats.cache_hit_rate() - expected).abs() < 1e-9,
        "hit rate {} != {expected}",
        stats.cache_hit_rate()
    );
}

/// Three turns with different usage each: proves accumulation, not overwrite.
#[tokio::test]
async fn rollup_accumulates_across_every_turn() {
    let provider = MockProvider::new(vec![
        MockResponse::TextWithUsage("a".into(), usage(1, 2, 3, 4)),
        MockResponse::TextWithUsage("b".into(), usage(10, 20, 30, 40)),
        MockResponse::TextWithUsage("c".into(), usage(100, 200, 300, 400)),
    ]);

    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    let mut config = make_config(provider);
    // Two follow-ups, so the loop takes three turns in one run.
    let pending = std::sync::Arc::new(std::sync::Mutex::new(vec![
        AgentMessage::Llm(Message::user("second")),
        AgentMessage::Llm(Message::user("third")),
    ]));
    let handout = pending.clone();
    config.get_follow_up_messages = Some(Box::new(move || {
        let mut q = handout.lock().unwrap();
        if q.is_empty() {
            vec![]
        } else {
            vec![q.remove(0)]
        }
    }));

    agent_loop(
        vec![AgentMessage::Llm(Message::user("first"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    let stats = agent_end_stats(&collect_events(rx));
    assert_eq!(stats.turns, 3);
    assert_eq!(stats.usage.input, 111);
    assert_eq!(stats.usage.output, 222);
    assert_eq!(stats.usage.cache_read, 333);
    assert_eq!(stats.usage.cache_write, 444);
}

/// A run that produced no LLM turn reports zeros rather than a bogus rate.
#[tokio::test]
async fn rollup_of_an_empty_run_is_zero_not_nan() {
    let stats = SessionStats::default();
    assert_eq!(stats.turns, 0);
    assert_eq!(stats.cache_hit_rate(), 0.0);
    assert!(stats.cache_hit_rate().is_finite());
}

/// The wire format is frozen and archived streams predate `stats`. Without
/// `#[serde(default)]` on the field and on every `SessionStats` field, replaying
/// yesterday's JSONL fails outright.
#[test]
fn archived_agent_end_without_stats_still_deserializes() {
    let legacy = r#"{"type":"agentEnd","messages":[]}"#;
    let event: AgentEvent = serde_json::from_str(legacy).expect("legacy agentEnd must load");
    match event {
        AgentEvent::AgentEnd { stats, .. } => assert_eq!(stats, SessionStats::default()),
        other => panic!("wrong variant: {other:?}"),
    }

    // ...and a partially-written rollup, which is what a future field addition
    // looks like to an older reader.
    let partial = r#"{"type":"agentEnd","messages":[],"stats":{"turns":3}}"#;
    let event: AgentEvent = serde_json::from_str(partial).expect("partial stats must load");
    match event {
        AgentEvent::AgentEnd { stats, .. } => {
            assert_eq!(stats.turns, 3);
            assert_eq!(stats.compactions, 0);
            assert_eq!(stats.cost_usd, None);
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

/// `cost_usd` accrues only when the model has configured rates. `None` means
/// "cannot price this", never "free".
#[tokio::test]
async fn cost_accrues_when_rates_are_configured_and_stays_none_otherwise() {
    let responses = || {
        vec![
            MockResponse::TextWithUsage("a".into(), usage(1_000_000, 0, 0, 0)),
            MockResponse::TextWithUsage("b".into(), usage(0, 1_000_000, 0, 0)),
        ]
    };

    // Unpriced: no model_config at all.
    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(responses()));
    config.get_follow_up_messages = follow_up_once();
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(
        agent_end_stats(&collect_events(rx)).cost_usd,
        None,
        "an unpriced model must report None, not Some(0.0)"
    );

    // Priced: $3/M in, $15/M out. One million of each => 3.0 + 15.0.
    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    let mut priced = yoagent::provider::ModelConfig::anthropic("mock", "Mock");
    priced.cost = yoagent::provider::CostConfig {
        input_per_million: 3.0,
        output_per_million: 15.0,
        cache_read_per_million: 0.0,
        cache_write_per_million: 0.0,
    };
    let mut config = make_config(MockProvider::new(responses()));
    config.model_config = Some(priced);
    config.get_follow_up_messages = follow_up_once();
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;
    let cost = agent_end_stats(&collect_events(rx))
        .cost_usd
        .expect("priced model must report a cost");
    assert!((cost - 18.0).abs() < 1e-9, "cost {cost} != 18.0");
}

/// Hand out exactly one follow-up, so the loop takes two turns.
fn follow_up_once() -> Option<Box<dyn Fn() -> Vec<AgentMessage> + Send + Sync>> {
    let pending = std::sync::Arc::new(std::sync::Mutex::new(vec![AgentMessage::Llm(
        Message::user("again"),
    )]));
    Some(Box::new(move || {
        let mut q = pending.lock().unwrap();
        if q.is_empty() {
            vec![]
        } else {
            vec![q.remove(0)]
        }
    }))
}

/// `total_tokens` is not summed: providers disagree on it (Anthropic never sets
/// it, Bedrock excludes cache), so a session-level sum would read as
/// authoritative while being 0 for every Anthropic run.
#[tokio::test]
async fn total_tokens_is_not_summed_into_the_rollup() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &make_config(MockProvider::new(vec![MockResponse::TextWithUsage(
            "a".into(),
            usage(5, 6, 7, 8),
        )])),
        tx,
        CancellationToken::new(),
    )
    .await;

    let stats = agent_end_stats(&collect_events(rx));
    assert_eq!(stats.usage.input, 5);
    assert_eq!(
        stats.usage.total_tokens, 0,
        "total_tokens must stay 0; derive a total from the components"
    );
}

/// A strategy that never changes anything must not be counted. This is the
/// most likely implementation slip — counting *invocations* of `compact`
/// rather than *effective* compactions — and it would make the number useless
/// for anyone tuning a strategy.
#[tokio::test]
async fn a_no_op_compaction_is_not_counted() {
    struct NoOp;
    impl yoagent::context::CompactionStrategy for NoOp {
        fn compact(
            &self,
            messages: Vec<AgentMessage>,
            _config: &yoagent::context::ContextConfig,
        ) -> Vec<AgentMessage> {
            messages
        }
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::TextWithUsage("a".into(), usage(10, 10, 0, 0)),
        MockResponse::TextWithUsage("b".into(), usage(10, 10, 0, 0)),
    ]));
    // A tiny budget guarantees `compact` is called on every turn.
    config.context_config = Some(yoagent::context::ContextConfig {
        max_context_tokens: 10,
        system_prompt_tokens: 0,
        ..Default::default()
    });
    config.compaction_strategy = Some(std::sync::Arc::new(NoOp));
    config.get_follow_up_messages = follow_up_once();

    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(
        agent_end_stats(&collect_events(rx)).compactions,
        0,
        "compact() ran every turn but changed nothing — nothing to count"
    );
}

/// The `||` clause the counting comment exists for: a strategy that rewrites
/// content in place, leaving the message count identical. A length-only check
/// silently misses it.
#[tokio::test]
async fn an_in_place_rewrite_is_counted_despite_an_unchanged_message_count() {
    struct ShrinkInPlace;
    impl yoagent::context::CompactionStrategy for ShrinkInPlace {
        fn compact(
            &self,
            messages: Vec<AgentMessage>,
            _config: &yoagent::context::ContextConfig,
        ) -> Vec<AgentMessage> {
            let before = messages.len();
            let out: Vec<AgentMessage> = messages
                .into_iter()
                .map(|m| match m {
                    AgentMessage::Llm(Message::User { timestamp, .. }) => {
                        AgentMessage::Llm(Message::User {
                            content: vec![Content::Text { text: "x".into() }],
                            timestamp,
                        })
                    }
                    other => other,
                })
                .collect();
            assert_eq!(out.len(), before, "this strategy must not change length");
            out
        }
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![MockResponse::TextWithUsage(
        "a".into(),
        usage(10, 10, 0, 0),
    )]));
    config.context_config = Some(yoagent::context::ContextConfig {
        max_context_tokens: 10,
        system_prompt_tokens: 0,
        ..Default::default()
    });
    config.compaction_strategy = Some(std::sync::Arc::new(ShrinkInPlace));

    agent_loop(
        vec![AgentMessage::Llm(Message::user(
            "a considerably longer opening message than the replacement",
        ))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(
        agent_end_stats(&collect_events(rx)).compactions,
        1,
        "token total moved while the count did not — the || clause must catch it"
    );
}

// ---------------------------------------------------------------------------
// Truncation → retrieval (issue #125)
// ---------------------------------------------------------------------------

struct StashBigTool;

#[async_trait::async_trait]
impl AgentTool for StashBigTool {
    fn name(&self) -> &str {
        "big_output"
    }
    fn label(&self) -> &str {
        "Big"
    }
    fn description(&self) -> &str {
        "emits many lines"
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
            content: vec![Content::Text {
                text: (0..500)
                    .map(|i| format!("line {i}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            }],
            details: serde_json::Value::Null,
        })
    }
}

async fn run_with_sink(sink: Option<yoagent::shared_state::SharedState>) -> Vec<AgentMessage> {
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(StashBigTool)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "big_output".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = sink;

    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await
}

fn tool_result_text(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .find_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult { content, .. }) => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .expect("a tool result must be in history")
}

/// Opt-in: with no sink the behaviour is exactly what it was before, and the
/// marker must not advertise a retrieval that does not exist.
#[tokio::test]
async fn without_a_sink_truncation_is_unchanged() {
    let text = tool_result_text(&run_with_sink(None).await);
    assert!(text.contains("lines truncated"), "still truncated");
    assert!(
        !text.contains("shared_state"),
        "no sink means the marker must not name one: {text}"
    );
}

/// The whole point: what head-tail truncation elided is retrievable.
#[tokio::test]
async fn a_truncated_output_is_stashed_and_the_marker_names_the_key() {
    let state = yoagent::shared_state::SharedState::new();
    let messages = run_with_sink(Some(state.clone())).await;
    let text = tool_result_text(&messages);

    assert!(
        text.contains("lines truncated"),
        "context is still truncated"
    );
    assert!(
        text.contains("shared_state get"),
        "marker must tell the model how to retrieve: {text}"
    );

    // The key in the marker must actually resolve, and to the *full* output.
    let keys = state.keys().await;
    assert_eq!(keys.len(), 1, "exactly one stash, got {keys:?}");
    assert!(
        text.contains(&keys[0]),
        "marker names {:?} but the store holds {keys:?}",
        text
    );
    let full = state.get(&keys[0]).await.expect("key must resolve");
    assert!(full.contains("line 250"), "the elided middle must be there");
    assert!(
        full.lines().count() >= 500,
        "the whole output, not the truncation"
    );
}

/// Small outputs are not stashed — nothing was lost, so there is nothing to
/// retrieve, and a key per tool call would fill the store with noise.
#[tokio::test]
async fn output_that_fits_is_not_stashed() {
    struct SmallTool;
    #[async_trait::async_trait]
    impl AgentTool for SmallTool {
        fn name(&self) -> &str {
            "small"
        }
        fn label(&self) -> &str {
            "Small"
        }
        fn description(&self) -> &str {
            "short"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            _p: serde_json::Value,
            _c: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![Content::Text { text: "ok".into() }],
                details: serde_json::Value::Null,
            })
        }
    }

    let state = yoagent::shared_state::SharedState::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(SmallTool)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "small".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = Some(state.clone());

    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    assert!(
        state.keys().await.is_empty(),
        "nothing elided, nothing stashed"
    );
}

/// A sink that always fails, to exercise the `Err` arm — previously uncovered.
struct FailingBackend;

#[async_trait::async_trait]
impl yoagent::shared_state::SharedStateBackend for FailingBackend {
    async fn get(
        &self,
        _k: &str,
    ) -> Result<Option<String>, yoagent::shared_state::SharedStateError> {
        Ok(None)
    }
    async fn set(
        &self,
        _k: &str,
        _v: String,
    ) -> Result<(), yoagent::shared_state::SharedStateError> {
        Err(yoagent::shared_state::SharedStateError::Io(
            std::io::Error::other("disk on fire"),
        ))
    }
    async fn remove(&self, _k: &str) -> Result<bool, yoagent::shared_state::SharedStateError> {
        Ok(false)
    }
    async fn keys(&self) -> Result<Vec<String>, yoagent::shared_state::SharedStateError> {
        Ok(vec![])
    }
    async fn summary(&self) -> Result<String, yoagent::shared_state::SharedStateError> {
        Ok(String::new())
    }
}

/// When the stash fails, the marker must fall back to the plain form rather
/// than promising a retrieval that cannot happen.
#[tokio::test]
async fn a_failed_stash_leaves_an_unkeyed_marker() {
    let sink = yoagent::shared_state::SharedState::with_backend(FailingBackend);
    let text = tool_result_text(&run_with_sink(Some(sink)).await);

    assert!(text.contains("lines truncated"), "still truncated");
    assert!(
        !text.contains("shared_state get"),
        "a failed stash must not advertise retrieval: {text}"
    );
}

/// Parallel tool execution is the default, so one turn can produce several
/// truncated results. Each needs its own key, and each key must resolve.
#[tokio::test]
async fn parallel_truncated_results_each_get_a_resolvable_key() {
    let state = yoagent::shared_state::SharedState::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(StashBigTool)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![
            MockToolCall {
                provider_metadata: None,
                name: "big_output".into(),
                arguments: serde_json::json!({}),
            },
            MockToolCall {
                provider_metadata: None,
                name: "big_output".into(),
                arguments: serde_json::json!({}),
            },
        ]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = Some(state.clone());

    let messages = agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    // Both tool results carry a marker, and every key named resolves.
    let named: Vec<String> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult { content, .. }) => {
                content.iter().find_map(|c| match c {
                    Content::Text { text } => text
                        .split_once("shared_state get \"")
                        .and_then(|(_, rest)| rest.split_once('"').map(|(k, _)| k.to_string())),
                    _ => None,
                })
            }
            _ => None,
        })
        .collect();

    assert_eq!(
        named.len(),
        2,
        "both results must name a key, got {named:?}"
    );
    for key in &named {
        assert!(
            state.get(key).await.is_some(),
            "key {key} named in a marker must resolve"
        );
    }
}

/// The advertised public entry point. It does two things — registers the tool
/// and wires the sink — and neither half was covered: the other tests set
/// `config.tool_output_sink` directly, bypassing it entirely.
#[tokio::test]
async fn agent_with_shared_state_registers_the_tool_and_wires_the_sink() {
    use yoagent::provider::ModelConfig;

    let state = yoagent::shared_state::SharedState::new();
    let mut agent = Agent::from_provider(
        MockProvider::new(vec![
            MockResponse::ToolCalls(vec![MockToolCall {
                provider_metadata: None,
                name: "big_output".into(),
                arguments: serde_json::json!({}),
            }]),
            MockResponse::Text("done".into()),
        ]),
        ModelConfig::mock(),
    )
    .with_tools(vec![Box::new(StashBigTool)])
    .with_shared_state(state.clone())
    .with_context_config(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });

    let mut rx = agent.prompt("go").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;

    // Half one: the sink is wired, so the output was stashed.
    let keys = state.keys().await;
    assert_eq!(
        keys.len(),
        1,
        "with_shared_state must wire the sink, got {keys:?}"
    );

    // Half two: the tool is registered, so the model can act on the marker.
    let text = agent
        .messages()
        .iter()
        .find_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult { content, .. }) => {
                content.iter().find_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                })
            }
            _ => None,
        })
        .expect("a tool result");
    assert!(
        text.contains(&keys[0]),
        "the marker must name the stashed key: {text}"
    );
}

/// Calling it twice must not register two tools — providers reject duplicate
/// tool names with a 400, which would brick the agent at the first request.
#[tokio::test]
async fn with_shared_state_is_idempotent() {
    use yoagent::provider::ModelConfig;

    let state = yoagent::shared_state::SharedState::new();
    let mut agent = Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock())
        .with_shared_state(state.clone())
        .with_shared_state(state);

    let mut rx = agent.prompt("go").await;
    while rx.recv().await.is_some() {}
    agent.finish().await;
    // The run completes; a duplicate tool name would have been a build-time
    // duplicate in the tool list handed to the provider.
    assert!(!agent.messages().is_empty());
}

// ---------------------------------------------------------------------------
// Loop detection (issue #126)
// ---------------------------------------------------------------------------

fn loop_events(events: &[AgentEvent]) -> Vec<(String, usize, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::LoopDetected {
                tool_name,
                repetitions,
                aborted,
                ..
            } => Some((tool_name.clone(), *repetitions, *aborted)),
            _ => None,
        })
        .collect()
}

fn repeat_call(n: usize, args: serde_json::Value) -> Vec<MockResponse> {
    (0..n)
        .map(|_| {
            MockResponse::ToolCalls(vec![MockToolCall {
                provider_metadata: None,
                name: "silent_tool".into(),
                arguments: args.clone(),
            }])
        })
        .collect()
}

async fn run_with_limits(
    responses: Vec<MockResponse>,
    limits: yoagent::context::ExecutionLimits,
) -> Vec<AgentEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(SilentTool)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(responses));
    config.execution_limits = Some(limits);
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;
    collect_events(rx)
}

/// Three identical calls steer; continued repetition aborts. The steer comes
/// first on purpose — a model repeating a call is often retrying something
/// transient, and aborting immediately would regress that.
#[tokio::test]
async fn repeated_identical_calls_steer_then_abort() {
    let events = run_with_limits(
        repeat_call(12, serde_json::json!({"q": "same"})),
        yoagent::context::ExecutionLimits {
            max_identical_tool_calls: Some(3),
            ..Default::default()
        },
    )
    .await;

    let detected = loop_events(&events);
    assert!(
        detected.iter().any(|(_, _, aborted)| !aborted),
        "the first trip must steer, got {detected:?}"
    );
    assert!(
        detected.iter().any(|(_, _, aborted)| *aborted),
        "continued repetition must abort, got {detected:?}"
    );

    // Steer strictly precedes abort.
    let first_abort = detected.iter().position(|(_, _, a)| *a).unwrap();
    let first_steer = detected.iter().position(|(_, _, a)| !*a).unwrap();
    assert!(
        first_steer < first_abort,
        "steer must come first: {detected:?}"
    );
}

/// Distinct arguments are not a loop. This is the false positive that would
/// make the feature worse than useless — an agent working through a list of
/// files calls one tool repeatedly and legitimately.
#[tokio::test]
async fn distinct_arguments_never_trip() {
    let responses: Vec<MockResponse> = (0..12)
        .map(|i| {
            MockResponse::ToolCalls(vec![MockToolCall {
                provider_metadata: None,
                name: "silent_tool".into(),
                arguments: serde_json::json!({ "file": format!("f{i}.rs") }),
            }])
        })
        .collect();

    let events = run_with_limits(
        responses,
        yoagent::context::ExecutionLimits {
            max_identical_tool_calls: Some(3),
            ..Default::default()
        },
    )
    .await;
    assert!(
        loop_events(&events).is_empty(),
        "distinct arguments are progress, not repetition"
    );
}

/// A different call between repeats breaks the streak.
#[tokio::test]
async fn an_interleaved_different_call_resets_the_counter() {
    let mut responses = Vec::new();
    for i in 0..12 {
        responses.push(MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "silent_tool".into(),
            // same, same, different, repeating — never 3 in a row
            arguments: if i % 3 == 2 {
                serde_json::json!({"q": "other"})
            } else {
                serde_json::json!({"q": "same"})
            },
        }]));
    }

    let events = run_with_limits(
        responses,
        yoagent::context::ExecutionLimits {
            max_identical_tool_calls: Some(3),
            ..Default::default()
        },
    )
    .await;
    assert!(
        loop_events(&events).is_empty(),
        "a streak broken before the threshold is not a loop"
    );
}

/// `None` disables the check entirely.
#[tokio::test]
async fn detection_can_be_switched_off() {
    let events = run_with_limits(
        repeat_call(12, serde_json::json!({"q": "same"})),
        yoagent::context::ExecutionLimits {
            max_identical_tool_calls: None,
            ..Default::default()
        },
    )
    .await;
    assert!(
        loop_events(&events).is_empty(),
        "None must disable detection"
    );
}

// ---------------------------------------------------------------------------
// Multimodal + scoped stash (issue #134)
// ---------------------------------------------------------------------------

/// A tool result carrying text, an image, and more text.
struct MultimodalTool;

#[async_trait::async_trait]
impl AgentTool for MultimodalTool {
    fn name(&self) -> &str {
        "multimodal"
    }
    fn label(&self) -> &str {
        "Multimodal"
    }
    fn description(&self) -> &str {
        "emits text, an image, and more text"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(
        &self,
        _p: serde_json::Value,
        _c: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let long = |tag: &str| {
            (0..300)
                .map(|i| format!("{tag} {i}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        Ok(ToolResult {
            content: vec![
                Content::Text {
                    text: long("alpha"),
                },
                Content::Image {
                    data: "AAAA".into(),
                    mime_type: "image/png".into(),
                },
                Content::Text {
                    text: long("omega"),
                },
            ],
            details: serde_json::Value::Null,
        })
    }
}

fn markers_in(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|m| match m {
            AgentMessage::Llm(Message::ToolResult { content, .. }) => content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => text
                        .split_once("shared_state get \"")
                        .and_then(|(_, r)| r.split_once('"').map(|(k, _)| k.to_string())),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        })
        .collect()
}

/// Each marker must resolve to *its own* block. Sharing one key across blocks
/// made every fetch return all blocks concatenated, with the image between
/// them silently dropped — so what the model got back was never the block whose
/// marker it followed.
#[tokio::test]
async fn each_text_block_gets_its_own_key_and_resolves_to_itself() {
    let state = yoagent::shared_state::SharedState::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(MultimodalTool)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "multimodal".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = Some(state.clone());

    let messages = agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    let keys = markers_in(&messages);
    assert_eq!(
        keys.len(),
        2,
        "one marker per truncated text block: {keys:?}"
    );
    assert_ne!(keys[0], keys[1], "blocks must not share a key: {keys:?}");

    // Each key resolves to exactly the block whose marker named it.
    let first = state.get(&keys[0]).await.expect("first key resolves");
    let second = state.get(&keys[1]).await.expect("second key resolves");
    assert!(
        first.contains("alpha 150") && !first.contains("omega"),
        "the first marker's key must return only the first block"
    );
    assert!(
        second.contains("omega 150") && !second.contains("alpha"),
        "the second marker's key must return only the second block"
    );

    // The image survives in the transcript untouched — truncation never
    // rewrote it, and the stash never claimed to hold it.
    let has_image = messages.iter().any(|m| match m {
        AgentMessage::Llm(Message::ToolResult { content, .. }) => {
            content.iter().any(|c| matches!(c, Content::Image { .. }))
        }
        _ => false,
    });
    assert!(has_image, "non-text content must pass through untouched");
}

/// Only the second text block is long enough to truncate, so `marked` is
/// `[2]` while `blocks` holds entries at 0 and 2. Every other multi-block test
/// truncates *all* blocks, which makes `marked`'s values and `blocks`'
/// positions coincide — hiding the natural refactor
/// (`marked.iter().enumerate()` indexing `blocks[n]`) that would store one
/// block's text under another block's key.
#[tokio::test]
async fn a_partially_truncated_result_keys_the_right_block() {
    struct ShortThenLong;
    #[async_trait::async_trait]
    impl AgentTool for ShortThenLong {
        fn name(&self) -> &str {
            "short_then_long"
        }
        fn label(&self) -> &str {
            "Mixed"
        }
        fn description(&self) -> &str {
            "a short block, an image, then a long one"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            _p: serde_json::Value,
            _c: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![
                    Content::Text {
                        text: "short and safe".into(),
                    },
                    Content::Image {
                        data: "AAAA".into(),
                        mime_type: "image/png".into(),
                    },
                    Content::Text {
                        text: (0..300)
                            .map(|i| format!("omega {i}"))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    },
                ],
                details: serde_json::Value::Null,
            })
        }
    }

    let state = yoagent::shared_state::SharedState::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(ShortThenLong)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "short_then_long".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = Some(state.clone());

    let messages = agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    let keys = markers_in(&messages);
    assert_eq!(keys.len(), 1, "only the long block truncates: {keys:?}");
    assert!(
        keys[0].ends_with("-b2"),
        "the key must name the long block's own position (2), not its ordinal \
         among text blocks (1): {}",
        keys[0]
    );
    let full = state.get(&keys[0]).await.expect("the key resolves");
    assert!(
        full.contains("omega 150") && !full.contains("short and safe"),
        "the key must return the block whose marker named it"
    );
}

/// Image-only results are correctly benign: nothing truncates, so nothing is
/// stashed and no marker appears. Pinned so a future change to the stash gate
/// cannot regress it into storing an empty string under a live key.
#[tokio::test]
async fn an_image_only_result_stashes_nothing() {
    struct ImageOnly;
    #[async_trait::async_trait]
    impl AgentTool for ImageOnly {
        fn name(&self) -> &str {
            "image_only"
        }
        fn label(&self) -> &str {
            "Image"
        }
        fn description(&self) -> &str {
            "an image and nothing else"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            _p: serde_json::Value,
            _c: ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                content: vec![Content::Image {
                    data: "AAAA".into(),
                    mime_type: "image/png".into(),
                }],
                details: serde_json::Value::Null,
            })
        }
    }

    let state = yoagent::shared_state::SharedState::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut context = AgentContext {
        messages: vec![],
        tools: vec![Box::new(ImageOnly)],
        system_prompt: String::new(),
    };
    let mut config = make_config(MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "image_only".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]));
    config.context_config = Some(yoagent::context::ContextConfig {
        tool_output_max_lines: 20,
        ..Default::default()
    });
    config.tool_output_sink = Some(state.clone());

    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .await;

    assert!(
        state.keys().await.is_empty(),
        "nothing was elided, so nothing may be stored"
    );
}
