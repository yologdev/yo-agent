//! Integration tests against the real Google Gemini API.
//! Run with: GEMINI_API_KEY=... cargo test --test integration_gemini -- --ignored

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use yoagent::agent_loop::{agent_loop, AgentLoopConfig};
use yoagent::provider::{GoogleProvider, ModelConfig};
use yoagent::tools;
use yoagent::types::*;

fn api_key() -> String {
    std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set")
}

fn make_config(model: &str) -> AgentLoopConfig {
    let model_config = ModelConfig::google(model, model);
    AgentLoopConfig {
        provider: std::sync::Arc::new(GoogleProvider),
        model: model.into(),
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
        cache_config: CacheConfig::disabled(),
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

/// Simple text response.
#[tokio::test]
#[ignore]
async fn test_gemini_simple_text() {
    let config = make_config("gemini-3-flash-preview");
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let mut context = AgentContext {
        system_prompt: "Reply with exactly one word.".into(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let prompt = AgentMessage::Llm(Message::user("What color is the sky?"));
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let text = extract_assistant_text(&new_messages);
    assert!(!text.is_empty(), "Expected non-empty response");
    println!("Response: {}", text);
}

/// Function tool use with Gemini.
#[tokio::test]
#[ignore]
async fn test_gemini_tool_use() {
    let config = make_config("gemini-3-flash-preview");
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let mut context = AgentContext {
        system_prompt: "Use the bash tool to answer. Be concise.".into(),
        messages: Vec::new(),
        tools: tools::default_tools(),
    };

    let prompt = AgentMessage::Llm(Message::user(
        "What is the output of `echo hello_gemini`? Use bash to run it.",
    ));
    let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

    let text = extract_assistant_text(&new_messages);
    assert!(
        text.contains("hello_gemini"),
        "Expected response with tool output, got: {}",
        text
    );

    let mut got_tool_execution = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::ToolExecutionStart { .. }) {
            got_tool_execution = true;
        }
    }
    assert!(got_tool_execution, "Expected tool execution");
    println!("Response: {}", text);
}
