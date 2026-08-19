//! Integration tests against the OpenCode Zen gateway (https://opencode.ai/docs/zen).
//! Run with: OPENCODE_API_KEY=... cargo test --test integration_opencode -- --ignored
//!
//! These tests are #[ignore] by default so they don't run in CI without a key.

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use yoagent::agent_loop::{agent_loop, AgentLoopConfig};
use yoagent::provider::{AnthropicProvider, ModelConfig, OpenAiCompatProvider, StreamProvider};
use yoagent::types::*;

fn api_key() -> String {
    std::env::var("OPENCODE_API_KEY").expect("OPENCODE_API_KEY must be set")
}

fn make_config(
    provider: std::sync::Arc<dyn StreamProvider>,
    model_config: ModelConfig,
) -> AgentLoopConfig {
    AgentLoopConfig {
        provider,
        model: model_config.id.clone(),
        api_key: api_key(),
        thinking_level: ThinkingLevel::Off,
        max_tokens: Some(1024),
        temperature: None,
        model_config: Some(model_config),
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

fn extract_assistant_text(messages: &[AgentMessage]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            if let AgentMessage::Llm(Message::Assistant { content, .. }) = m {
                Some(
                    content
                        .iter()
                        .filter_map(|c| {
                            if let Content::Text { text } = c {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                )
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

async fn run_simple_prompt(config: AgentLoopConfig) -> String {
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let mut context = AgentContext {
        system_prompt: "Reply with exactly one word.".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("What color is the sky?"));
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    assert!(
        !new_messages.is_empty(),
        "Expected at least one new message"
    );
    let text = extract_assistant_text(&new_messages);
    assert!(!text.is_empty(), "Expected non-empty text response");
    text
}

/// Chat-completions model through the Zen gateway.
#[tokio::test]
#[ignore]
async fn test_opencode_zen_chat_completions() {
    let model = ModelConfig::opencode_zen("glm-5.2");
    let config = make_config(std::sync::Arc::new(OpenAiCompatProvider), model);
    let text = run_simple_prompt(config).await;
    println!("Zen chat-completions response: {}", text);
}

/// Anthropic-protocol model through the Zen gateway (Bearer auth + /messages).
#[tokio::test]
#[ignore]
async fn test_opencode_zen_anthropic_messages() {
    let model = ModelConfig::opencode_zen("claude-haiku-4-5");
    let config = make_config(std::sync::Arc::new(AnthropicProvider), model);
    let text = run_simple_prompt(config).await;
    println!("Zen /messages response: {}", text);
}
