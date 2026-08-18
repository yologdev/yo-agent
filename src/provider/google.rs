//! Google Generative AI (Gemini) provider.
//!
//! Uses the `streamGenerateContent` endpoint with SSE streaming.
//! API key is passed as a query parameter.
//!
//! # Prompt caching: implicit only, deliberately
//!
//! This provider sends no cache directives and ignores [`CacheStrategy`].
//! Gemini caches implicitly on its own, and the `cachedContentTokenCount` this
//! module reads back into [`Usage::cache_read`] reports the result of that
//! automatic behaviour — it is telemetry, not an acknowledgement of anything
//! the client asked for.
//!
//! Gemini's *explicit* caching is a separate resource with a different
//! lifecycle: you create a `CachedContent` object, receive a handle, reference
//! it by name on later requests, and manage its TTL and its own billing line.
//! That does not fit behind [`CacheStrategy`], which describes where to place
//! markers inside a single request. Wiring it here would mean either
//! misrepresenting a stateful resource as a per-request flag, or silently
//! creating server-side objects with a lifetime the caller cannot see.
//!
//! It is worth doing as its own API — with explicit create/reference/expire
//! surface — and it is not worth pretending the current enum can express it.
//! See yologdev/yoagent#123.

use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, warn};

pub struct GoogleProvider;

#[async_trait]
impl StreamProvider for GoogleProvider {
    fn protocol(&self) -> Option<crate::provider::ApiProtocol> {
        Some(crate::provider::ApiProtocol::GoogleGenerativeAi)
    }

    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        let model_config = config
            .model_config
            .as_ref()
            .ok_or_else(|| ProviderError::Other("ModelConfig required".into()))?;

        let base_url = &model_config.base_url;
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            base_url, config.model, config.api_key
        );

        let body = build_request_body(&config);
        debug!("Google GenAI request: model={}", config.model);

        let client = reqwest::Client::new();
        let mut request = client.post(&url).header("content-type", "application/json");

        for (k, v) in &model_config.headers {
            request = request.header(k, v);
        }

        // Google streams JSON chunks separated by newlines, not SSE.
        // With alt=sse, it does use SSE format.
        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::classify(
                status.as_u16(),
                &format!("Google API error {}: {}", status, body),
            ));
        }

        let mut content: Vec<Content> = Vec::new();
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::Stop;
        let mut error_message: Option<String> = None;

        let _ = tx.send(StreamEvent::Start);

        // Parse SSE stream
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    return Err(ProviderError::Cancelled);
                }
                chunk = stream.next() => {
                    match chunk {
                        None => break,
                        Some(Err(e)) => {
                            // Match the other providers: a transport failure is an
                            // error (and retryable), not a silently truncated turn.
                            let provider_err = ProviderError::Network(e.to_string());
                            warn!("Google stream error: {}", provider_err);
                            return Err(provider_err);
                        }
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));

                            // Process complete SSE events (handle both \n\n and \r\n\r\n)
                            while let Some(data) = next_sse_data(&mut buffer) {
                                if data.is_empty() {
                                    continue;
                                }

                                // Google reports mid-stream failures as
                                // {"error": {...}} payloads, which would otherwise
                                // deserialize into an empty chunk and vanish.
                                if is_error_payload(&data) {
                                    let provider_err = classify_sse_error_event(&data);
                                    warn!("Google in-stream error: {}", provider_err);
                                    return Err(provider_err);
                                }

                                let chunk: GoogleChunk = match serde_json::from_str(&data) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        warn!("Failed to parse Google chunk: {}", e);
                                        continue;
                                    }
                                };

                                // Process candidates
                                for candidate in &chunk.candidates.unwrap_or_default() {
                                    if let Some(c) = &candidate.content {
                                        for part in &c.parts {
                                            if let Some(text) = part_text(part) {
                                                if part.thought.unwrap_or(false) {
                                                    // Thought summary part → Thinking content.
                                                    let think_idx = content.iter().position(|c| matches!(c, Content::Thinking { .. }));
                                                    let idx = match think_idx {
                                                        Some(i) => i,
                                                        None => {
                                                            content.push(Content::thinking(String::new()));
                                                            content.len() - 1
                                                        }
                                                    };
                                                    if let Some(Content::Thinking { thinking, .. }) = content.get_mut(idx) {
                                                        thinking.push_str(text);
                                                    }
                                                    let _ = tx.send(StreamEvent::ThinkingDelta {
                                                        content_index: idx,
                                                        delta: text.to_string(),
                                                    });
                                                    continue;
                                                }
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
                                                    delta: text.to_string(),
                                                });
                                            }
                                            if let Some(fc) = &part.function_call {
                                                let id = fc.id.clone().unwrap_or_else(|| format!("google-fc-{}", content.len()));
                                                let args = fc.args.clone().unwrap_or(serde_json::Value::Object(Default::default()));
                                                let metadata = part.thought_signature.as_ref().map(|sig| {
                                                    serde_json::json!({"thought_signature": sig})
                                                });
                                                let idx = content.len();
                                                content.push(Content::ToolCall {
                                                    id: id.clone(),
                                                    name: fc.name.clone(),
                                                    arguments: args,
                                                    provider_metadata: metadata,
                                                });
                                                let _ = tx.send(StreamEvent::ToolCallStart {
                                                    content_index: idx,
                                                    id,
                                                    name: fc.name.clone(),
                                                });
                                                let _ = tx.send(StreamEvent::ToolCallEnd { content_index: idx });
                                                stop_reason = StopReason::ToolUse;
                                            }
                                        }
                                    }
                                    if let Some(reason) = &candidate.finish_reason {
                                        // Don't override ToolUse -- Gemini returns "STOP"
                                        // even when it emits function calls
                                        if stop_reason != StopReason::ToolUse {
                                            stop_reason = match reason.as_str() {
                                                "STOP" => StopReason::Stop,
                                                "MAX_TOKENS" | "RECITATION" => StopReason::Length,
                                                "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST"
                                                | "SPII" => {
                                                    warn!(
                                                        "Gemini blocked the response (finishReason={})",
                                                        reason
                                                    );
                                                    error_message = Some(format!(
                                                        "Response blocked by Gemini safety filters (finishReason: {})",
                                                        reason
                                                    ));
                                                    StopReason::Refusal
                                                }
                                                _ => StopReason::Stop,
                                            };
                                        }
                                    }
                                }

                                // Process usage
                                if let Some(u) = &chunk.usage_metadata {
                                    // promptTokenCount includes cached tokens;
                                    // keep `input` as the uncached remainder so
                                    // downstream sums don't double-count.
                                    usage.input = u
                                        .prompt_token_count
                                        .unwrap_or(0)
                                        .saturating_sub(u.cached_content_token_count.unwrap_or(0));
                                    usage.output = u.candidates_token_count.unwrap_or(0);
                                    usage.total_tokens = u.total_token_count.unwrap_or(0);
                                    usage.cache_read = u.cached_content_token_count.unwrap_or(0);
                                }
                            }
                        }
                    }
                }
            }
        }

        let message = Message::Assistant {
            content,
            stop_reason,
            model: config.model.clone(),
            provider: model_config.provider.clone(),
            usage,
            timestamp: now_ms(),
            error_message,
        };

        let _ = tx.send(StreamEvent::Done {
            message: message.clone(),
        });
        Ok(message)
    }
}

/// Pop the next complete SSE event from `buffer` and return its `data:`
/// payload (empty string when the event carries no data line). Handles both
/// `\n\n` and `\r\n\r\n` event separators, splitting at whichever occurs
/// first. Returns `None` until a complete event is buffered. Only the first
/// `data:` line of an event is returned.
fn next_sse_data(buffer: &mut String) -> Option<String> {
    let lf = buffer.find("\n\n");
    let crlf = buffer.find("\r\n\r\n");
    let (pos, sep_len) = match (lf, crlf) {
        (Some(l), Some(c)) if c < l => (c, 4),
        (Some(l), _) => (l, 2),
        (None, Some(c)) => (c, 4),
        (None, None) => return None,
    };
    let event_str = buffer[..pos].to_string();
    *buffer = buffer[pos + sep_len..].to_string();
    let data = event_str
        .lines()
        .map(|l| l.trim_end_matches('\r'))
        .find(|l| l.starts_with("data: "))
        .map(|l| l[6..].to_string())
        .unwrap_or_default();
    Some(data)
}

/// Whether an SSE data payload is a Google error envelope
/// (`{"error": {...}}`) rather than a content chunk.
fn is_error_payload(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data)
        .map(|v| v.get("error").is_some())
        .unwrap_or(false)
}

/// Non-empty text of a part. Gemini streams empty text parts while thinking;
/// those must be skipped.
fn part_text(part: &GooglePart) -> Option<&str> {
    part.text.as_deref().filter(|t| !t.is_empty())
}

/// Token budget for Gemini's thinkingConfig per level.
fn gemini_thinking_budget(level: ThinkingLevel) -> u32 {
    match level {
        ThinkingLevel::Off => 0,
        ThinkingLevel::Minimal | ThinkingLevel::Low => 1024,
        ThinkingLevel::Medium => 8192,
        ThinkingLevel::High => 24576,
    }
}

fn build_request_body(config: &StreamConfig) -> serde_json::Value {
    let mut contents: Vec<serde_json::Value> = Vec::new();

    for msg in &config.messages {
        match msg {
            Message::User { content, .. } => {
                let parts = content_to_google_parts(content);
                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": parts,
                }));
            }
            Message::Assistant { content, .. } => {
                let parts = content_to_google_parts(content);
                contents.push(serde_json::json!({
                    "role": "model",
                    "parts": parts,
                }));
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                let text = content
                    .iter()
                    .find_map(|c| match c {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();

                let mut fr = serde_json::json!({
                    "name": tool_name,
                    "response": {"result": text},
                });
                if !tool_call_id.is_empty() && !tool_call_id.starts_with("google-fc-") {
                    fr["id"] = serde_json::json!(tool_call_id);
                }
                let mut parts = vec![serde_json::json!({"functionResponse": fr})];

                // Append image parts if present
                for c in content {
                    if let Content::Image { data, mime_type } = c {
                        parts.push(serde_json::json!({
                            "inlineData": {"mimeType": mime_type, "data": data},
                        }));
                    }
                }

                contents.push(serde_json::json!({
                    "role": "user",
                    "parts": parts,
                }));
            }
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
    });

    if !config.system_prompt.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": [{"text": config.system_prompt}],
        });
    }

    let mut generation_config = serde_json::json!({});
    if let Some(max) = config.max_tokens {
        generation_config["maxOutputTokens"] = serde_json::json!(max);
    }
    if let Some(temp) = config.temperature {
        generation_config["temperature"] = serde_json::json!(temp);
    }
    // Structured outputs: native responseSchema (Gemini's OpenAPI-style
    // schema dialect — pass the caller's schema through as given).
    if let Some(schema) = &config.output_schema {
        generation_config["responseMimeType"] = serde_json::json!("application/json");
        generation_config["responseSchema"] = schema.schema.clone();
    }

    // Thinking: Gemini 2.5's thinkingConfig. Budget scales with the level;
    // includeThoughts streams thought summaries back as thought parts.
    if config.thinking_level != ThinkingLevel::Off {
        generation_config["thinkingConfig"] = serde_json::json!({
            "thinkingBudget": gemini_thinking_budget(config.thinking_level),
            "includeThoughts": true,
        });
    }

    if generation_config != serde_json::json!({}) {
        body["generationConfig"] = generation_config;
    }

    if !config.tools.is_empty() {
        let declarations: Vec<serde_json::Value> = config
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body["tools"] = serde_json::json!([{
            "functionDeclarations": declarations,
        }]);
    }

    body
}

fn content_to_google_parts(content: &[Content]) -> Vec<serde_json::Value> {
    content
        .iter()
        .filter(|c| !matches!(c, Content::Text { text } if text.is_empty()))
        .filter_map(|c| match c {
            Content::Text { text } => Some(serde_json::json!({"text": text})),
            Content::Image { data, mime_type } => Some(serde_json::json!({
                "inlineData": {"mimeType": mime_type, "data": data},
            })),
            Content::ToolCall {
                id,
                name,
                arguments,
                provider_metadata,
            } => {
                let mut fc = serde_json::json!({"name": name, "args": arguments});
                if !id.is_empty() && !id.starts_with("google-fc-") {
                    fc["id"] = serde_json::json!(id);
                }
                let mut part = serde_json::json!({"functionCall": fc});
                if let Some(sig) = provider_metadata
                    .as_ref()
                    .and_then(|m| m.get("thought_signature"))
                    .and_then(|v| v.as_str())
                {
                    part["thoughtSignature"] = serde_json::json!(sig);
                }
                Some(part)
            }
            Content::Thinking { .. } => None,
        })
        .collect()
}

// Google API response types
#[derive(Deserialize)]
struct GoogleChunk {
    #[serde(default)]
    candidates: Option<Vec<GoogleCandidate>>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GoogleUsageMetadata>,
}

#[derive(Deserialize)]
struct GoogleCandidate {
    #[serde(default)]
    content: Option<GoogleContent>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GoogleContent {
    #[serde(default)]
    parts: Vec<GooglePart>,
}

#[derive(Deserialize)]
struct GooglePart {
    #[serde(default)]
    text: Option<String>,
    /// True when this part is a thought summary (thinkingConfig.includeThoughts).
    #[serde(default)]
    thought: Option<bool>,
    #[serde(default, rename = "functionCall")]
    function_call: Option<GoogleFunctionCall>,
    #[serde(default, rename = "thoughtSignature")]
    thought_signature: Option<String>,
}

#[derive(Deserialize)]
struct GoogleFunctionCall {
    name: String,
    #[serde(default)]
    args: Option<serde_json::Value>,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Deserialize)]
struct GoogleUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
    #[serde(default, rename = "totalTokenCount")]
    total_token_count: Option<u64>,
    #[serde(default, rename = "cachedContentTokenCount")]
    cached_content_token_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_blocks_are_dropped_on_replay() {
        // Gemini does not accept echoed thought summaries; signatures ride on
        // functionCall parts instead. Pin the drop so it stays deliberate.
        let mut config = StreamConfig::new("gemini-2.5-pro", "k");
        config.messages = vec![
            Message::user("go"),
            Message::assistant(
                vec![
                    Content::thinking("thought"),
                    Content::Text {
                        text: "answer".into(),
                    },
                ],
                StopReason::Stop,
                "m",
                "google",
                Usage::default(),
            ),
        ];
        let body = build_request_body(&config);
        let parts = body["contents"][1]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1, "only the text part is replayed");
        assert_eq!(parts[0]["text"], "answer");
    }

    #[test]
    fn thinking_level_sets_thinking_config() {
        let config = StreamConfig {
            model: "gemini-2.5-pro".into(),
            system_prompt: "".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            thinking_level: ThinkingLevel::Medium,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };
        let body = build_request_body(&config);
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            8192
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn thinking_off_omits_thinking_config() {
        let config = StreamConfig {
            model: "gemini-2.5-pro".into(),
            system_prompt: "".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };
        let body = build_request_body(&config);
        assert!(body["generationConfig"]["thinkingConfig"].is_null());
    }

    #[test]
    fn structured_output_sets_response_schema() {
        let config = StreamConfig {
            model: "gemini-2.5-pro".into(),
            system_prompt: "".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: Some(crate::provider::OutputSchema::new(
                "structured_output",
                serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
            )),
        };
        let body = build_request_body(&config);
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            body["generationConfig"]["responseSchema"]["properties"]["x"]["type"],
            "number"
        );
    }

    #[test]
    fn test_build_google_request() {
        let config = StreamConfig {
            model: "gemini-2.0-flash".into(),
            system_prompt: "Be helpful".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: Some(1024),
            temperature: Some(0.7),
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config);
        assert!(body["contents"].is_array());
        assert_eq!(body["contents"][0]["role"], "user");
        assert!(body["systemInstruction"].is_object());
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 1024);
        let temp = body["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_content_to_google_parts_text() {
        let content = vec![Content::Text {
            text: "hello".into(),
        }];
        let parts = content_to_google_parts(&content);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn test_content_to_google_parts_filters_empty_text() {
        let content = vec![
            Content::Text { text: "".into() },
            Content::Text {
                text: "hello".into(),
            },
            Content::Text { text: "".into() },
        ];
        let parts = content_to_google_parts(&content);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "hello");
    }

    #[test]
    fn test_content_to_google_parts_tool_call() {
        let content = vec![Content::ToolCall {
            id: "tc-1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            provider_metadata: None,
        }];
        let parts = content_to_google_parts(&content);
        assert_eq!(parts[0]["functionCall"]["name"], "bash");
    }

    #[test]
    fn test_parse_chunk_with_function_call_and_thought_signature() {
        let data = r#"{"candidates": [{"content": {"parts": [{"functionCall": {"name": "bash", "args": {"command": "echo hi"}, "id": "abc123"}, "thoughtSignature": "SIG_DATA"}], "role": "model"}, "finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}"#;

        let chunk: GoogleChunk = serde_json::from_str(data).unwrap();
        let candidates = chunk.candidates.unwrap();
        assert_eq!(candidates.len(), 1);

        let parts = &candidates[0].content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 1);

        let fc = parts[0].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "bash");
        assert_eq!(fc.id.as_deref(), Some("abc123"));
        assert_eq!(fc.args.as_ref().unwrap()["command"], "echo hi");

        assert_eq!(parts[0].thought_signature.as_deref(), Some("SIG_DATA"));
    }

    #[test]
    fn test_parse_chunk_with_empty_text() {
        // Gemini sends empty text parts during thinking -- part_text (used by
        // the streaming loop) must skip them and keep non-empty ones.
        let data = r#"{"candidates": [{"content": {"parts": [{"text": ""}, {"text": "Hello"}], "role": "model"}, "index": 0}]}"#;

        let chunk: GoogleChunk = serde_json::from_str(data).unwrap();
        let candidates = chunk.candidates.unwrap();
        let parts = &candidates[0].content.as_ref().unwrap().parts;
        assert_eq!(part_text(&parts[0]), None, "empty text parts are skipped");
        assert_eq!(part_text(&parts[1]), Some("Hello"));
    }

    #[test]
    fn test_parse_chunk_with_crlf_sse() {
        // Full pipeline: next_sse_data (the production splitter) on a CRLF
        // stream, then chunk parsing.
        let mut buf = "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Blue\"}], \"role\": \"model\"}, \"finishReason\": \"STOP\", \"index\": 0}]}\r\n\r\n".to_string();

        let data = next_sse_data(&mut buf).expect("complete CRLF event");
        assert!(buf.is_empty(), "event consumed from buffer");
        assert_eq!(next_sse_data(&mut buf), None);

        let chunk: GoogleChunk = serde_json::from_str(&data).unwrap();
        let candidates = chunk.candidates.unwrap();
        let text = &candidates[0].content.as_ref().unwrap().parts[0].text;
        assert_eq!(text.as_deref(), Some("Blue"));
    }

    #[test]
    fn test_next_sse_data_partial_events_stay_buffered() {
        let mut buf = "data: {\"a\":1}\n\ndata: partial".to_string();
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some("{\"a\":1}"));
        assert_eq!(next_sse_data(&mut buf), None, "incomplete event waits");
        assert_eq!(buf, "data: partial");
        buf.push_str("\n\n");
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some("partial"));
    }

    #[test]
    fn test_next_sse_data_consumes_events_without_data_lines() {
        // SSE comments/keepalives and leading separators must be CONSUMED
        // (returning Some("")), never None — returning None would wedge the
        // buffer and drop every subsequent event.
        let mut buf = ": keepalive\n\n\n\ndata: x\n\n".to_string();
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some(""));
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some(""));
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some("x"));
        assert_eq!(next_sse_data(&mut buf), None);
    }

    #[test]
    fn test_is_error_payload() {
        assert!(is_error_payload(
            r#"{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}"#
        ));
        assert!(!is_error_payload(r#"{"candidates": []}"#));
        assert!(!is_error_payload("not json"));
    }

    #[test]
    fn test_next_sse_data_splits_earliest_separator_first() {
        // A CRLF-separated event earlier in the buffer must split before a
        // later LF separator (the old inline logic preferred the LF match and
        // merged the two events).
        let mut buf = "data: one\r\n\r\ndata: two\n\n".to_string();
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some("one"));
        assert_eq!(next_sse_data(&mut buf).as_deref(), Some("two"));
        assert_eq!(next_sse_data(&mut buf), None);
    }

    #[test]
    fn test_thought_signature_round_trip() {
        let content = vec![Content::ToolCall {
            id: "abc123".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "echo hi"}),
            provider_metadata: Some(serde_json::json!({"thought_signature": "SIG_DATA"})),
        }];

        let parts = content_to_google_parts(&content);
        assert_eq!(parts.len(), 1);

        assert_eq!(parts[0]["functionCall"]["name"], "bash");
        assert_eq!(parts[0]["functionCall"]["id"], "abc123");
        assert_eq!(parts[0]["functionCall"]["args"]["command"], "echo hi");
        assert_eq!(parts[0]["thoughtSignature"], "SIG_DATA");
    }

    #[test]
    fn test_tool_call_without_thought_signature() {
        // Synthetic IDs (google-fc-*) should not be sent to Gemini
        let content = vec![Content::ToolCall {
            id: "google-fc-0".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
            provider_metadata: None,
        }];

        let parts = content_to_google_parts(&content);
        assert!(parts[0]["functionCall"].get("id").is_none());
        assert!(parts[0].get("thoughtSignature").is_none());
    }

    #[test]
    fn test_function_response_includes_id() {
        let config = StreamConfig {
            model: "gemini-2.5-flash".into(),
            system_prompt: "".into(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        id: "abc123".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "echo hi"}),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "abc123".into(),
                    tool_name: "bash".into(),
                    content: vec![Content::Text { text: "hi".into() }],
                    is_error: false,
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test".into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config);
        let msgs = body["contents"].as_array().unwrap();
        let tool_result = &msgs[1]["parts"][0]["functionResponse"];
        assert_eq!(tool_result["name"], "bash");
        assert_eq!(tool_result["id"], "abc123");
        assert_eq!(tool_result["response"]["result"], "hi");
    }

    #[test]
    fn test_function_response_synthetic_id_omitted() {
        let config = StreamConfig {
            model: "gemini-2.5-flash".into(),
            system_prompt: "".into(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        id: "google-fc-0".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "ls"}),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "google-fc-0".into(),
                    tool_name: "bash".into(),
                    content: vec![Content::Text {
                        text: "output".into(),
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
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config);
        let msgs = body["contents"].as_array().unwrap();
        let tool_result = &msgs[1]["parts"][0]["functionResponse"];
        assert!(
            tool_result.get("id").is_none(),
            "Synthetic ID should not be included"
        );
    }
}
