//! Tests that the loop emits the documented tracing spans with their fields.
//!
//! Attaches a capturing subscriber to the loop future via `with_subscriber`
//! and drives `agent_loop` directly in the current task — spans created in
//! separately-spawned tasks would not reach this scoped subscriber.

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::instrument::WithSubscriber;
use tracing_subscriber::layer::SubscriberExt;
use yoagent::provider::mock::*;
use yoagent::provider::MockProvider;
use yoagent::*;

/// Layer that records every new span's name.
struct SpanCollector(Arc<Mutex<Vec<String>>>);

impl<S> tracing_subscriber::Layer<S> for SpanCollector
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.0
            .lock()
            .unwrap()
            .push(attrs.metadata().name().to_string());
    }
}

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
        "echoes"
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

fn loop_config(provider: MockProvider) -> yoagent::agent_loop::AgentLoopConfig {
    yoagent::agent_loop::AgentLoopConfig {
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
        tool_execution: ToolExecutionStrategy::default(),
        tool_middleware: vec![],
        output_schema: None,
        retry_config: yoagent::RetryConfig::none(),
        before_turn: None,
        after_turn: None,
        on_error: None,
        input_filters: vec![],
        turn_delay: None,
    }
}

#[tokio::test]
async fn loop_emits_agent_llm_and_tool_spans() {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(SpanCollector(spans.clone()));

    let provider = MockProvider::new(vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            provider_metadata: None,
            name: "echo".into(),
            arguments: serde_json::json!({}),
        }]),
        MockResponse::Text("done".into()),
    ]);
    let config = loop_config(provider);

    let mut context = AgentContext {
        system_prompt: "test".into(),
        messages: Vec::new(),
        tools: vec![Box::new(EchoTool)],
    };
    let (tx, _rx) = mpsc::unbounded_channel();

    // with_subscriber attaches the subscriber across every poll of the
    // future (a plain `with_default(sub, || fut).await` would only cover
    // creating the future, not running it).
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .with_subscriber(subscriber)
    .await;

    let names = spans.lock().unwrap().clone();
    assert!(
        names.contains(&"agent_loop".to_string()),
        "expected agent_loop span, got: {names:?}"
    );
    // Two turns → two llm_stream spans; one tool execution → one tool span.
    assert_eq!(
        names.iter().filter(|n| *n == "llm_stream").count(),
        2,
        "got: {names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| *n == "tool").count(),
        1,
        "got: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Field values: tokens + cost recorded on llm_stream (not just span names)
// ---------------------------------------------------------------------------

/// Records (span_name, field_name, value_debug) for every record() call.
struct FieldCollector {
    names: Arc<Mutex<std::collections::HashMap<u64, String>>>,
    records: Arc<Mutex<Vec<(String, String, String)>>>,
}

struct FieldVisitor<'a> {
    span: String,
    out: &'a Mutex<Vec<(String, String, String)>>,
}

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.out.lock().unwrap().push((
            self.span.clone(),
            field.name().to_string(),
            format!("{value:?}"),
        ));
    }
}

impl<S> tracing_subscriber::Layer<S> for FieldCollector
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        self.names
            .lock()
            .unwrap()
            .insert(id.into_u64(), attrs.metadata().name().to_string());
    }
    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let span = self
            .names
            .lock()
            .unwrap()
            .get(&id.into_u64())
            .cloned()
            .unwrap_or_default();
        let mut visitor = FieldVisitor {
            span,
            out: &self.records,
        };
        values.record(&mut visitor);
    }
}

/// Provider returning fixed non-zero usage so token/cost fields are real.
struct UsageProvider;

#[async_trait::async_trait]
impl yoagent::provider::StreamProvider for UsageProvider {
    async fn stream(
        &self,
        _config: yoagent::provider::StreamConfig,
        tx: mpsc::UnboundedSender<yoagent::provider::StreamEvent>,
        _cancel: CancellationToken,
    ) -> Result<Message, yoagent::provider::ProviderError> {
        let msg = Message::assistant(
            vec![Content::Text { text: "ok".into() }],
            StopReason::Stop,
            "m",
            "mock",
            Usage {
                input: 1_000_000,
                output: 500_000,
                cache_read: 7,
                cache_write: 0,
                total_tokens: 1_500_007,
            },
        );
        let _ = tx.send(yoagent::provider::StreamEvent::Done {
            message: msg.clone(),
        });
        Ok(msg)
    }
}

#[tokio::test]
async fn llm_stream_records_tokens_and_cost() {
    let names = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let records = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(FieldCollector {
        names: names.clone(),
        records: records.clone(),
    });

    let mut config = loop_config(MockProvider::text("unused"));
    config.provider = std::sync::Arc::new(UsageProvider);
    let mut mc = yoagent::provider::ModelConfig::mock();
    mc.cost.input_per_million = 3.0;
    mc.cost.output_per_million = 15.0;
    config.model_config = Some(mc);

    let mut context = AgentContext {
        system_prompt: "t".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    let (tx, _rx) = mpsc::unbounded_channel();
    agent_loop(
        vec![AgentMessage::Llm(Message::user("go"))],
        &mut context,
        &config,
        tx,
        CancellationToken::new(),
    )
    .with_subscriber(subscriber)
    .await;

    let recs = records.lock().unwrap().clone();
    let get = |field: &str| -> String {
        recs.iter()
            .find(|(span, f, _)| span == "llm_stream" && f == field)
            .map(|(_, _, v)| v.clone())
            .unwrap_or_else(|| panic!("field {field} not recorded; got {recs:?}"))
    };
    assert_eq!(get("tokens_in"), "1000000");
    assert_eq!(get("tokens_out"), "500000");
    assert_eq!(get("tokens_cached"), "7");
    // 1M in @ $3/M + 0.5M out @ $15/M = 10.5
    assert_eq!(get("cost_usd"), "10.5");
    assert_eq!(get("error"), "false");
}
