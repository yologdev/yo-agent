//! OpenAI Chat Completions compatible provider.
//!
//! One implementation covers OpenAI, xAI, Groq, Cerebras, OpenRouter,
//! Mistral, DeepSeek, MiniMax, HuggingFace, Kimi, and any other provider
//! that implements the OpenAI Chat Completions API.
//!
//! Behavioral differences are handled via `OpenAiCompat` flags in ModelConfig.

use super::model::{MaxTokensField, ModelConfig, OpenAiCompat, ThinkingFormat};
use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::EventSource;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, warn};

pub struct OpenAiCompatProvider;

#[async_trait]
impl StreamProvider for OpenAiCompatProvider {
    fn protocol(&self) -> Option<crate::provider::ApiProtocol> {
        Some(crate::provider::ApiProtocol::OpenAiCompletions)
    }

    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        let model_config = config.model_config.as_ref().ok_or_else(|| {
            ProviderError::Other("ModelConfig required for OpenAI provider".into())
        })?;
        let compat = model_config.compat.as_ref().cloned().unwrap_or_default();

        let base_url = &model_config.base_url;
        let url = format!("{}/chat/completions", base_url);

        let body = build_request_body(&config, model_config, &compat);
        debug!("OpenAI compat request: model={} url={}", config.model, url);

        let client = reqwest::Client::new();
        let mut request = client
            .post(&url)
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", config.api_key));

        // Add any extra headers from model config
        for (k, v) in &model_config.headers {
            request = request.header(k, v);
        }

        let request = request.json(&body);

        let mut es =
            EventSource::new(request).map_err(|e| ProviderError::Network(e.to_string()))?;

        let mut content: Vec<Content> = Vec::new();
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::Stop;
        let mut saw_finish_reason = false;
        let mut tool_call_buffers: Vec<ToolCallBuffer> = Vec::new();

        let _ = tx.send(StreamEvent::Start);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    es.close();
                    return Err(ProviderError::Cancelled);
                }
                event = es.next() => {
                    match event {
                        None => break,
                        Some(Ok(reqwest_eventsource::Event::Open)) => {}
                        Some(Ok(reqwest_eventsource::Event::Message(msg))) => {
                            if msg.data == "[DONE]" {
                                break;
                            }

                            let chunk: OpenAiChunk = match serde_json::from_str(&msg.data) {
                                Ok(c) => c,
                                Err(e) => {
                                    debug!("Failed to parse OpenAI chunk: {} data={}", e, &msg.data);
                                    continue;
                                }
                            };

                            // Process usage
                            if let Some(u) = &chunk.usage {
                                let cache_read = u
                                    .prompt_cache_hit_tokens
                                    .or_else(|| {
                                        u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens)
                                    })
                                    .unwrap_or(0);
                                usage.input = u.prompt_cache_miss_tokens.unwrap_or_else(|| {
                                    u.prompt_tokens.saturating_sub(cache_read)
                                });
                                usage.output = u.completion_tokens;
                                usage.total_tokens = u.total_tokens;
                                usage.cache_read = cache_read;
                            }

                            for choice in &chunk.choices {
                                let delta = &choice.delta;

                                // Handle reasoning/thinking content
                                let reasoning = match compat.thinking_format {
                                    ThinkingFormat::Xai => delta.reasoning.as_deref(),
                                    _ => delta.reasoning_content.as_deref(),
                                };
                                if let Some(reasoning_text) = reasoning {
                                    // Find or create thinking block
                                    let thinking_idx = content.iter().position(|c| matches!(c, Content::Thinking { .. }));
                                    let idx = match thinking_idx {
                                        Some(i) => i,
                                        None => {
                                            content.push(Content::Thinking { thinking: String::new(), signature: None });
                                            content.len() - 1
                                        }
                                    };
                                    if let Some(Content::Thinking { thinking, .. }) = content.get_mut(idx) {
                                        thinking.push_str(reasoning_text);
                                    }
                                    let _ = tx.send(StreamEvent::ThinkingDelta {
                                        content_index: idx,
                                        delta: reasoning_text.to_string(),
                                    });
                                }

                                // Handle text content
                                if let Some(text) = &delta.content {
                                    let text_idx = content.iter().position(|c| matches!(c, Content::Text { .. }));
                                    let idx = match text_idx {
                                        Some(i) => i,
                                        None => {
                                            content.push(Content::Text { text: String::new() });
                                            content.len() - 1
                                        }
                                    };
                                    if let Some(Content::Text { text: t }) = content.get_mut(idx) {
                                        t.push_str(text);
                                    }
                                    let _ = tx.send(StreamEvent::TextDelta {
                                        content_index: idx,
                                        delta: text.clone(),
                                    });
                                }

                                // Handle tool calls
                                if let Some(tool_calls) = &delta.tool_calls {
                                    for tc in tool_calls {
                                        let tc_index = tc.index as usize;
                                        while tool_call_buffers.len() <= tc_index {
                                            tool_call_buffers.push(ToolCallBuffer::default());
                                        }
                                        let buf = &mut tool_call_buffers[tc_index];
                                        if let Some(id) = &tc.id {
                                            buf.id = id.clone();
                                        }
                                        if let Some(f) = &tc.function {
                                            if let Some(name) = &f.name {
                                                buf.name.clone_from(name);
                                                let _ = tx.send(StreamEvent::ToolCallStart {
                                                    content_index: content.len() + tc_index,
                                                    id: buf.id.clone(),
                                                    name: name.clone(),
                                                });
                                            }
                                            if let Some(args) = &f.arguments {
                                                buf.arguments.push_str(args);
                                                let _ = tx.send(StreamEvent::ToolCallDelta {
                                                    content_index: content.len() + tc_index,
                                                    delta: args.clone(),
                                                });
                                            }
                                        }
                                    }
                                }

                                // Handle finish reason
                                if let Some(reason) = &choice.finish_reason {
                                    saw_finish_reason = true;
                                    stop_reason = match reason.as_str() {
                                        "stop" => StopReason::Stop,
                                        "length" => StopReason::Length,
                                        "tool_calls" => StopReason::ToolUse,
                                        _ => StopReason::Stop,
                                    };
                                }
                            }
                        }
                        // Some providers (e.g. MiniMax) close the connection
                        // without the OpenAI-standard `data: [DONE]` terminator.
                        // If a finish_reason was already received, the response
                        // is complete — treat as clean EOF. (This eventsource
                        // surfaces a body close as StreamEnded; an I/O failure
                        // mid-body surfaces as Transport instead.) A
                        // StreamEnded with NO finish_reason is truncation and
                        // stays an error — a retryable one since #83, since a
                        // well-framed body carrying a truncated payload is
                        // usually a transient gateway fault.
                        Some(Err(reqwest_eventsource::Error::StreamEnded)) if saw_finish_reason => {
                            debug!("provider closed stream without [DONE] after finish_reason");
                            break;
                        }
                        Some(Err(e)) => {
                            let provider_err = classify_eventsource_error(e).await;
                            warn!("OpenAI SSE error: {}", provider_err);
                            return Err(provider_err);
                        }
                    }
                }
            }
        }

        // Finalize tool calls
        for buf in &tool_call_buffers {
            let args = serde_json::from_str(&buf.arguments).unwrap_or_else(|e| {
                if !buf.arguments.is_empty() {
                    warn!(
                        tool = %buf.name,
                        len = buf.arguments.len(),
                        "tool-call arguments failed to parse ({e}); using empty object"
                    );
                }
                serde_json::Value::Object(Default::default())
            });
            content.push(Content::ToolCall {
                provider_metadata: None,
                id: buf.id.clone(),
                name: buf.name.clone(),
                arguments: args,
            });
            let _ = tx.send(StreamEvent::ToolCallEnd {
                content_index: content.len() - 1,
            });
        }

        if !tool_call_buffers.is_empty() {
            stop_reason = StopReason::ToolUse;
        }

        let message = Message::Assistant {
            content,
            stop_reason,
            model: config.model.clone(),
            provider: model_config.provider.clone(),
            usage,
            timestamp: now_ms(),
            error_message: None,
        };

        let _ = tx.send(StreamEvent::Done {
            message: message.clone(),
        });
        Ok(message)
    }
}

#[derive(Default)]
struct ToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
}

fn build_request_body(
    config: &StreamConfig,
    model_config: &ModelConfig,
    compat: &OpenAiCompat,
) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    // System prompt
    if !config.system_prompt.is_empty() {
        let role = if compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        messages.push(serde_json::json!({
            "role": role,
            "content": config.system_prompt,
        }));
    }

    for msg in &config.messages {
        if !matches!(msg, Message::ToolResult { .. } | Message::Assistant { .. }) {
            maybe_insert_assistant_after_tool_results(&mut messages, compat);
        }

        match msg {
            Message::User { content, .. } => {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content_to_openai(content),
                }));
            }
            Message::Assistant { content, .. } => {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();

                for c in content {
                    match c {
                        Content::Text { text } if text.is_empty() => {}
                        Content::Text { text } => {
                            parts.push(serde_json::json!({"type": "text", "text": text}));
                        }
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => {
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments.to_string()},
                            }));
                        }
                        _ => {}
                    }
                }

                let mut msg_obj = serde_json::json!({"role": "assistant"});
                if !parts.is_empty() {
                    msg_obj["content"] = serde_json::json!(parts);
                }
                if !tool_calls.is_empty() {
                    msg_obj["tool_calls"] = serde_json::json!(tool_calls);
                }
                messages.push(msg_obj);
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                let content_val = if content.iter().any(|c| matches!(c, Content::Image { .. })) {
                    // Images present: use array format for multimodal tool results
                    content_to_openai(content)
                } else {
                    // Text-only: use plain string for maximum compat
                    let text = content
                        .iter()
                        .find_map(|c| match c {
                            Content::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    serde_json::json!(text)
                };

                let mut msg_obj = serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content_val,
                });
                if compat.requires_tool_result_name {
                    msg_obj["name"] = serde_json::json!(tool_name);
                }
                messages.push(msg_obj);
            }
        }
    }
    maybe_insert_assistant_after_tool_results(&mut messages, compat);

    let max_tokens_val = config.max_tokens.unwrap_or(model_config.max_tokens);
    let mut body = serde_json::json!({
        "model": config.model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": messages,
    });

    match compat.max_tokens_field {
        MaxTokensField::MaxCompletionTokens => {
            body["max_completion_tokens"] = serde_json::json!(max_tokens_val);
        }
        MaxTokensField::MaxTokens => {
            body["max_tokens"] = serde_json::json!(max_tokens_val);
        }
    }

    if compat.supports_thinking_control {
        let thinking_type = if config.thinking_level == ThinkingLevel::Off {
            "disabled"
        } else {
            "enabled"
        };
        body["thinking"] = serde_json::json!({ "type": thinking_type });
    }

    if !config.tools.is_empty() {
        let tools: Vec<serde_json::Value> = config
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::json!(tools);
    }

    // Prompt caching. OpenAI caches prefixes automatically once they exceed
    // ~1024 tokens, so there are no breakpoints to place — `prompt_cache_key`
    // only routes requests from one conversation toward the same cache. Gated
    // on the compat flag because the field is OpenAI's: a strict compat server
    // that validates unknown keys would reject the request outright, and the
    // providers that cache automatically were never reading it anyway.
    if compat.supports_prompt_cache_key {
        if let Some(key) = config.cache_session_key() {
            body["prompt_cache_key"] = serde_json::json!(key);
        }
    } else if config.cache_config.session_key.is_some() && config.cache_config.hints_enabled() {
        // A *derived* key going unsent is a missed optimization and stays
        // quiet. An *explicitly configured* one going unsent is a user
        // instruction being discarded — someone isolating tenants gets exactly
        // the sharing they were preventing. Matches the convention stated on
        // `StreamConfig::output_schema` and honoured by five providers.
        warn!(
            "CacheConfig::session_key is set, but provider '{}' does not accept \
             prompt_cache_key; the key is ignored and requests will not be routed \
             by session",
            model_config.provider
        );
    }

    // Structured outputs: native json_schema response format.
    if let Some(schema) = &config.output_schema {
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": schema.name,
                "schema": schema.schema,
                "strict": true,
            },
        });
    }

    if config.thinking_level != ThinkingLevel::Off && compat.supports_reasoning_effort {
        let effort = match config.thinking_level {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::Off => unreachable!(),
        };
        body["reasoning_effort"] = serde_json::json!(effort);
    }

    if let Some(temp) = config.temperature {
        body["temperature"] = serde_json::json!(temp);
    }

    body
}

fn maybe_insert_assistant_after_tool_results(
    messages: &mut Vec<serde_json::Value>,
    compat: &OpenAiCompat,
) {
    if !compat.requires_assistant_after_tool_result {
        return;
    }

    let last_is_tool = messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(|role| role.as_str())
        == Some("tool");
    if last_is_tool {
        messages.push(serde_json::json!({
            "role": "assistant",
            "content": "",
        }));
    }
}

fn content_to_openai(content: &[Content]) -> serde_json::Value {
    if content.len() == 1 {
        if let Content::Text { text } = &content[0] {
            if !text.is_empty() {
                return serde_json::json!(text);
            }
        }
    }
    let parts: Vec<serde_json::Value> = content
        .iter()
        .filter(|c| !matches!(c, Content::Text { text } if text.is_empty()))
        .filter_map(|c| match c {
            Content::Text { text } => Some(serde_json::json!({"type": "text", "text": text})),
            Content::Image { data, mime_type } => Some(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{};base64,{}", mime_type, data)},
            })),
            _ => None,
        })
        .collect();
    serde_json::json!(parts)
}

// OpenAI streaming response types
#[derive(Deserialize)]
struct OpenAiChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Deserialize)]
struct OpenAiToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Deserialize)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    prompt_cache_miss_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::model::ModelConfig;

    #[test]
    fn structured_output_sets_json_schema_response_format() {
        let mc = ModelConfig::openai("gpt-5.5", "GPT-5.5");
        let config = StreamConfig {
            model: "gpt-5.5".into(),
            system_prompt: "".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(mc.clone()),
            cache_config: CacheConfig::default(),
            output_schema: Some(crate::provider::OutputSchema::new(
                "structured_output",
                serde_json::json!({"type": "object"}),
            )),
        };
        let body = build_request_body(&config, &mc, &OpenAiCompat::openai());
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "structured_output"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn test_build_request_body_basic() {
        let model_config = ModelConfig::openai("gpt-4o", "GPT-4o");
        let config = StreamConfig {
            model: "gpt-4o".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &OpenAiCompat::openai());
        assert_eq!(body["model"], "gpt-4o");
        assert!(body["stream"].as_bool().unwrap());
        // Developer role for OpenAI
        assert_eq!(body["messages"][0]["role"], "developer");
        assert_eq!(body["messages"][1]["role"], "user");
        // max_completion_tokens for OpenAI
        assert!(body["max_completion_tokens"].is_number());
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let model_config = ModelConfig::openai("gpt-4o", "GPT-4o");
        let compat = OpenAiCompat::openai();
        let config = StreamConfig {
            model: "gpt-4o".into(),
            system_prompt: String::new(),
            messages: vec![Message::user("List files")],
            tools: vec![ToolDefinition {
                name: "bash".into(),
                description: "Run a command".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: Some(1024),
            temperature: Some(0.5),
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        assert_eq!(body["temperature"], 0.5);
    }

    #[test]
    fn test_build_request_body_deepseek_off_uses_current_api_shape() {
        let model_config = ModelConfig::deepseek("deepseek-v4-flash", "DeepSeek V4 Flash");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "deepseek-v4-flash".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
        // Anthropic's breakpoint markers never belong on this path.
        assert!(!body.to_string().contains("cache_control"));
        // Nor does OpenAI's routing key: DeepSeek caches automatically and
        // does not read it, so `supports_prompt_cache_key` stays off. Caching
        // is enabled here — this asserts the compat gate, not the master
        // switch.
        assert!(config.cache_config.enabled);
        assert!(!compat.supports_prompt_cache_key);
        assert!(body.get("prompt_cache_key").is_none());
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(
            vec![Content::Text { text: text.into() }],
            StopReason::Stop,
            "gpt-5.5",
            "openai",
            Usage::default(),
        )
    }

    /// Build a body against native OpenAI, which is the only compat provider
    /// with `supports_prompt_cache_key` on.
    fn openai_body(cache_config: CacheConfig, messages: Vec<Message>) -> serde_json::Value {
        let model_config = ModelConfig::openai("gpt-5.5", "GPT-5.5");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let mut config = StreamConfig::new("gpt-5.5", "test");
        config.system_prompt = "You are helpful.".into();
        config.messages = messages;
        config.model_config = Some(model_config.clone());
        config.cache_config = cache_config;
        build_request_body(&config, &model_config, &compat)
    }

    #[test]
    fn prompt_cache_key_is_sent_when_caching_is_enabled() {
        let body = openai_body(CacheConfig::default(), vec![Message::user("Hello")]);
        let key = body["prompt_cache_key"]
            .as_str()
            .expect("enabled caching must send a routing key");
        assert!(key.starts_with("yo-"), "unexpected key shape: {key}");
    }

    #[test]
    fn prompt_cache_key_is_absent_when_caching_is_off() {
        for cfg in [
            CacheConfig::disabled(),
            CacheConfig {
                strategy: CacheStrategy::Disabled,
                ..CacheConfig::default()
            },
        ] {
            let body = openai_body(cfg, vec![Message::user("Hello")]);
            assert!(
                body.get("prompt_cache_key").is_none(),
                "disabled caching must send no routing key"
            );
        }
    }

    #[test]
    fn explicit_session_key_wins_over_derivation() {
        let body = openai_body(
            CacheConfig::default().with_session_key("tenant-42/session-7"),
            vec![Message::user("Hello")],
        );
        assert_eq!(body["prompt_cache_key"], "tenant-42/session-7");
    }

    /// The point of keying on the *head* rather than the whole message list:
    /// a key that changed every turn would route each request to a fresh cache
    /// and defeat the feature it exists to serve.
    #[test]
    fn derived_key_is_stable_as_the_conversation_grows() {
        let turn_1 = openai_body(CacheConfig::default(), vec![Message::user("Hello")]);
        let turn_5 = openai_body(
            CacheConfig::default(),
            vec![
                Message::user("Hello"),
                assistant("Hi"),
                Message::user("Second"),
                assistant("Sure"),
                Message::user("Third"),
            ],
        );
        assert_eq!(turn_1["prompt_cache_key"], turn_5["prompt_cache_key"]);
    }

    #[test]
    fn key_survives_a_compaction_marker_replacing_the_head() {
        let normal = openai_body(
            CacheConfig::default(),
            vec![Message::user("Deploy the API")],
        );
        // What `compact_messages` leaves at index 0 once it drops the head.
        // Referencing the constant rather than copying its text, so changing
        // the marker cannot silently stop covering this scenario.
        let compacted = openai_body(
            CacheConfig::default(),
            vec![
                Message::user(crate::context::COMPACTION_MARKER),
                Message::user("Later turn"),
            ],
        );
        assert_eq!(normal["prompt_cache_key"], compacted["prompt_cache_key"]);
    }

    /// `prompt_cache_key` must reach **only** native OpenAI. The failure this
    /// guards is the one the gate exists for: a strict compat server rejects
    /// the entire request over an unknown field. Nothing else pins this — a
    /// future preset written as `..OpenAiCompat::openai()` (the pattern
    /// `gpt_5_5` already uses) would switch it on silently.
    #[test]
    fn prompt_cache_key_reaches_only_native_openai() {
        let must_not_send = [
            ModelConfig::deepseek("deepseek-v4-flash", "DeepSeek"),
            ModelConfig::groq("llama-3.3-70b", "Llama"),
            ModelConfig::xai("grok-4-1-fast", "Grok"),
            ModelConfig::mistral("mistral-large", "Mistral"),
            ModelConfig::zai("glm-4", "Z.ai"),
            ModelConfig::qwen("qwen-max", "Qwen"),
            ModelConfig::minimax("abab-6", "MiniMax"),
            ModelConfig::meta("llama-4", "Meta"),
        ];

        for model_config in must_not_send {
            let compat = model_config.compat.as_ref().cloned().unwrap_or_default();
            assert!(
                !compat.supports_prompt_cache_key,
                "{} must not advertise prompt_cache_key support",
                model_config.provider
            );

            let mut config = StreamConfig::new(model_config.id.clone(), "test");
            config.system_prompt = "You are helpful.".into();
            config.messages = vec![Message::user("Hello")];
            config.model_config = Some(model_config.clone());

            let body = build_request_body(&config, &model_config, &compat);
            assert!(
                body.get("prompt_cache_key").is_none(),
                "{} must not receive prompt_cache_key",
                model_config.provider
            );
        }

        // ...and the one that must.
        let openai = ModelConfig::openai("gpt-5.5", "GPT-5.5");
        assert!(openai.compat.as_ref().unwrap().supports_prompt_cache_key);
    }

    /// An explicit key set against a provider that cannot carry it is a user
    /// instruction being discarded; the request goes out without it and the
    /// call site warns rather than dropping it silently.
    #[test]
    fn ungated_provider_sends_no_key_even_when_set_explicitly() {
        let model_config = ModelConfig::deepseek("deepseek-v4-flash", "DeepSeek V4 Flash");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let mut config = StreamConfig::new("deepseek-v4-flash", "test");
        config.system_prompt = "You are helpful.".into();
        config.messages = vec![Message::user("Hello")];
        config.model_config = Some(model_config.clone());
        config.cache_config = CacheConfig::default().with_session_key("tenant-42");

        let body = build_request_body(&config, &model_config, &compat);
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn test_build_request_body_deepseek_thinking_enabled() {
        let model_config = ModelConfig::deepseek("deepseek-v4-pro", "DeepSeek V4 Pro");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "deepseek-v4-pro".into(),
            system_prompt: String::new(),
            messages: vec![Message::user("Solve this")],
            tools: vec![],
            thinking_level: ThinkingLevel::High,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["max_tokens"], 384_000);
    }

    #[test]
    fn test_build_request_body_qwen_uses_max_tokens_and_streaming_usage() {
        let model_config = ModelConfig::qwen("qwen3.6-plus", "Qwen 3.6 Plus");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "qwen3.6-plus".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::High,
            api_key: "test".into(),
            max_tokens: Some(2048),
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["max_tokens"], 2048);
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_build_request_body_qwen_tools_use_openai_shape() {
        let model_config = ModelConfig::qwen("qwen3-coder-plus", "Qwen 3 Coder Plus");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "qwen3-coder-plus".into(),
            system_prompt: String::new(),
            messages: vec![Message::user("List files")],
            tools: vec![ToolDefinition {
                name: "list_files".into(),
                description: "List files".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }),
            }],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "list_files");
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn test_deepseek_usage_cache_fields_parse() {
        let chunk: OpenAiChunk = serde_json::from_value(serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 70,
                "prompt_cache_miss_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 110
            }
        }))
        .unwrap();

        let u = chunk.usage.unwrap();
        let cache_read = u.prompt_cache_hit_tokens.unwrap_or(0);
        let input = u
            .prompt_cache_miss_tokens
            .unwrap_or_else(|| u.prompt_tokens.saturating_sub(cache_read));
        assert_eq!(input, 30);
        assert_eq!(cache_read, 70);
        assert_eq!(u.completion_tokens, 10);
    }

    #[test]
    fn test_content_to_openai_simple_text() {
        let content = vec![Content::Text {
            text: "hello".into(),
        }];
        let result = content_to_openai(&content);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_content_to_openai_filters_empty_text() {
        let content = vec![
            Content::Text { text: "".into() },
            Content::Text {
                text: "hello".into(),
            },
            Content::Text { text: "".into() },
        ];
        let result = content_to_openai(&content);
        let parts = result.as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn test_content_to_openai_single_empty_text_filtered() {
        let content = vec![Content::Text { text: "".into() }];
        let result = content_to_openai(&content);
        let parts = result.as_array().unwrap();
        assert!(parts.is_empty());
    }

    #[test]
    fn test_content_to_openai_multipart() {
        let content = vec![
            Content::Text {
                text: "look at this".into(),
            },
            Content::Image {
                data: "abc".into(),
                mime_type: "image/png".into(),
            },
        ];
        let result = content_to_openai(&content);
        assert!(result.is_array());
        assert_eq!(result[0]["type"], "text");
        assert_eq!(result[1]["type"], "image_url");
    }

    #[test]
    fn test_tool_result_with_image() {
        let model_config = ModelConfig::openai("gpt-4o", "GPT-4o");
        let compat = OpenAiCompat::openai();
        let config = StreamConfig {
            model: "gpt-4o".into(),
            system_prompt: String::new(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        provider_metadata: None,
                        id: "call-1".into(),
                        name: "read_file".into(),
                        arguments: serde_json::json!({"path": "img.png"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    content: vec![Content::Image {
                        data: "aW1hZ2VkYXRh".into(),
                        mime_type: "image/png".into(),
                    }],
                    is_error: false,
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        let msgs = body["messages"].as_array().unwrap();
        // tool result is the last message (after system + assistant)
        let tool_msg = msgs.last().unwrap();
        assert_eq!(tool_msg["role"], "tool");
        // content should be an array with image_url
        let content = tool_msg["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image_url");
        assert!(content[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_tool_result_text_only_uses_string() {
        let model_config = ModelConfig::openai("gpt-4o", "GPT-4o");
        let compat = OpenAiCompat::openai();
        let config = StreamConfig {
            model: "gpt-4o".into(),
            system_prompt: String::new(),
            messages: vec![Message::ToolResult {
                tool_call_id: "call-1".into(),
                tool_name: "bash".into(),
                content: vec![Content::Text {
                    text: "hello".into(),
                }],
                is_error: false,
                timestamp: 0,
            }],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        let msgs = body["messages"].as_array().unwrap();
        let tool_msg = msgs.last().unwrap();
        // Text-only: content should be a plain string
        assert_eq!(tool_msg["content"], "hello");
    }

    #[test]
    fn test_ollama_inserts_assistant_after_tool_result_run() {
        let model_config = ModelConfig::ollama("http://localhost:11434/v1", "llama3.1:8b");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "llama3.1:8b".into(),
            system_prompt: String::new(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        provider_metadata: None,
                        id: "call-1".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"cmd": "ls"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "call-1".into(),
                    tool_name: "bash".into(),
                    content: vec![Content::Text {
                        text: "a.txt\nb.txt".into(),
                    }],
                    is_error: false,
                    timestamp: 0,
                },
                Message::User {
                    content: vec![Content::Text {
                        text: "which is largest?".into(),
                    }],
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "");
        assert_eq!(msgs[3]["role"], "user");
    }

    #[test]
    fn test_ollama_inserts_one_assistant_after_multiple_tool_results() {
        let model_config = ModelConfig::ollama("http://localhost:11434/v1", "qwen2.5-coder:7b");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "qwen2.5-coder:7b".into(),
            system_prompt: String::new(),
            messages: vec![
                Message::ToolResult {
                    tool_call_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    content: vec![Content::Text { text: "a".into() }],
                    is_error: false,
                    timestamp: 0,
                },
                Message::ToolResult {
                    tool_call_id: "call-2".into(),
                    tool_name: "read_file".into(),
                    content: vec![Content::Text { text: "b".into() }],
                    is_error: false,
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "");
    }

    #[test]
    fn test_ollama_does_not_insert_assistant_before_existing_assistant() {
        let model_config = ModelConfig::ollama("http://localhost:11434/v1", "llama3.1:8b");
        let compat = model_config.compat.as_ref().unwrap().clone();
        let config = StreamConfig {
            model: "llama3.1:8b".into(),
            system_prompt: String::new(),
            messages: vec![
                Message::ToolResult {
                    tool_call_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    content: vec![Content::Text { text: "a".into() }],
                    is_error: false,
                    timestamp: 0,
                },
                Message::Assistant {
                    content: vec![Content::Text {
                        text: "The file contains a.".into(),
                    }],
                    stop_reason: StopReason::Stop,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::User {
                    content: vec![Content::Text {
                        text: "thanks".into(),
                    }],
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: Some(model_config.clone()),
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, &model_config, &compat);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["text"], "The file contains a.");
        assert_eq!(msgs[2]["role"], "user");
    }
}
