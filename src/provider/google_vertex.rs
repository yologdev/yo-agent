//! Google Vertex AI provider.
//!
//! Similar to Google Generative AI but uses OAuth2 authentication
//! and a different base URL pattern with project/location.
//!
//! The API key in StreamConfig is expected to be an OAuth2 access token.
//!
//! Prompt caching behaves exactly as in [`super::google`], including the
//! reasoning for why explicit `CachedContent` is deliberately not wired — see
//! that module's docs rather than re-deriving it here.
//! Callers are responsible for obtaining the token (e.g., via service account JWT).

use super::model::ModelConfig;
use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;

pub struct GoogleVertexProvider;

impl GoogleVertexProvider {
    /// Build the Vertex AI URL from model config.
    /// Expects base_url in format: `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models`
    fn vertex_url(model_config: &ModelConfig, model: &str) -> String {
        format!(
            "{}/{}:streamGenerateContent?alt=sse",
            model_config.base_url, model
        )
    }
}

#[async_trait]
impl StreamProvider for GoogleVertexProvider {
    fn protocol(&self) -> Option<crate::provider::ApiProtocol> {
        Some(crate::provider::ApiProtocol::GoogleVertex)
    }

    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        if config.output_schema.is_some() {
            tracing::warn!(
                "structured outputs are not yet wired for the Google Vertex provider; output_schema will be ignored"
            );
        }
        let model_config = config
            .model_config
            .as_ref()
            .ok_or_else(|| ProviderError::Other("ModelConfig required".into()))?;

        // Override the base_url to use Vertex format.
        // The GoogleProvider's stream will use model_config.base_url, but we need
        // a different URL pattern. We delegate to GoogleProvider with a modified config.
        let vertex_url = Self::vertex_url(model_config, &config.model);

        // Create a modified model config with the Vertex URL pattern
        let mut vertex_model = model_config.clone();
        // For Vertex, auth is via Bearer token (OAuth2), not API key in query param.
        // We need to add the Authorization header.
        vertex_model.headers.insert(
            "authorization".to_string(),
            format!("Bearer {}", config.api_key),
        );

        // Build request body same as Google (same content format)
        let body = build_vertex_request_body(&config);

        let client = reqwest::Client::new();
        let mut request = client
            .post(&vertex_url)
            .header("content-type", "application/json");

        for (k, v) in &vertex_model.headers {
            request = request.header(k, v);
        }

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
                &format!("Vertex AI error {}: {}", status, body),
            ));
        }

        // Delegate SSE parsing to the Google provider's streaming logic.
        // Since the response format is identical, we reuse GoogleProvider.
        // However, we already have the response, so we'll parse it inline.
        // For simplicity, delegate fully to GoogleProvider with modified config.
        let mut modified_config = config.clone();
        modified_config.model_config = Some(vertex_model);

        // Actually, let's just delegate to GoogleProvider. The key difference
        // is auth (Bearer vs API key in URL). We handle that by using a modified
        // model config. But GoogleProvider builds its own URL... so let's just
        // use GoogleProvider with a trick: empty api_key and auth in headers.
        // We can't easily reuse GoogleProvider because it constructs its own URL.
        // Instead, parse the SSE response directly (same format as Google GenAI).
        parse_google_sse_response(response, &config, &model_config.provider, tx, cancel).await
    }
}

/// Parse a Google-format SSE response stream. Shared between Google and Vertex.
async fn parse_google_sse_response(
    response: reqwest::Response,
    config: &StreamConfig,
    provider_name: &str,
    tx: mpsc::UnboundedSender<StreamEvent>,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<Message, ProviderError> {
    use futures::StreamExt;
    use serde::Deserialize;
    use tracing::{debug, warn};

    let mut content: Vec<Content> = Vec::new();
    let mut usage = Usage::default();
    let mut stop_reason = StopReason::Stop;

    let _ = tx.send(StreamEvent::Start);

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
                        warn!("Vertex stream error: {}", e);
                        break;
                    }
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(pos) = buffer.find("\n\n") {
                            let event_str = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            let data = event_str
                                .lines()
                                .find(|l| l.starts_with("data: "))
                                .map(|l| &l[6..])
                                .unwrap_or("");

                            if data.is_empty() {
                                continue;
                            }

                            #[derive(Deserialize)]
                            struct Chunk {
                                #[serde(default)]
                                candidates: Option<Vec<Candidate>>,
                                #[serde(default, rename = "usageMetadata")]
                                usage_metadata: Option<UsageMeta>,
                            }
                            #[derive(Deserialize)]
                            struct Candidate {
                                #[serde(default)]
                                content: Option<CContent>,
                                #[serde(default, rename = "finishReason")]
                                finish_reason: Option<String>,
                            }
                            #[derive(Deserialize)]
                            struct CContent {
                                #[serde(default)]
                                parts: Vec<Part>,
                            }
                            #[derive(Deserialize)]
                            struct Part {
                                #[serde(default)]
                                text: Option<String>,
                                #[serde(default)]
                                thought: Option<bool>,
                                #[serde(default, rename = "functionCall")]
                                function_call: Option<FCall>,
                                #[serde(default, rename = "thoughtSignature")]
                                thought_signature: Option<String>,
                            }
                            #[derive(Deserialize)]
                            struct FCall {
                                name: String,
                                #[serde(default)]
                                args: Option<serde_json::Value>,
                            }
                            #[derive(Deserialize)]
                            struct UsageMeta {
                                #[serde(default, rename = "promptTokenCount")]
                                prompt_token_count: Option<u64>,
                                #[serde(default, rename = "candidatesTokenCount")]
                                candidates_token_count: Option<u64>,
                                #[serde(default, rename = "totalTokenCount")]
                                total_token_count: Option<u64>,
                                #[serde(default, rename = "cachedContentTokenCount")]
                                cached_content_token_count: Option<u64>,
                            }

                            let parsed: Chunk = match serde_json::from_str(data) {
                                Ok(c) => c,
                                Err(e) => {
                                    debug!("Failed to parse Vertex chunk: {}", e);
                                    continue;
                                }
                            };

                            for candidate in parsed.candidates.unwrap_or_default() {
                                if let Some(c) = candidate.content {
                                    for part in c.parts {
                                        if let Some(text) = part.text {
                                            if part.thought.unwrap_or(false) {
                                                let think_idx = content.iter().position(|c| matches!(c, Content::Thinking { .. }));
                                                let idx = match think_idx {
                                                    Some(i) => i,
                                                    None => {
                                                        content.push(Content::thinking(String::new()));
                                                        content.len() - 1
                                                    }
                                                };
                                                if let Some(Content::Thinking { thinking, .. }) = content.get_mut(idx) {
                                                    thinking.push_str(&text);
                                                }
                                                let _ = tx.send(StreamEvent::ThinkingDelta {
                                                    content_index: idx,
                                                    delta: text,
                                                });
                                                continue;
                                            }
                                            let idx = content.iter().position(|c| matches!(c, Content::Text { .. }));
                                            let idx = match idx {
                                                Some(i) => i,
                                                None => {
                                                    content.push(Content::Text { text: String::new() });
                                                    content.len() - 1
                                                }
                                            };
                                            if let Some(Content::Text { text: t }) = content.get_mut(idx) {
                                                t.push_str(&text);
                                            }
                                            let _ = tx.send(StreamEvent::TextDelta {
                                                content_index: idx,
                                                delta: text,
                                            });
                                        }
                                        if let Some(fc) = part.function_call {
                                            let id = format!("vertex-fc-{}", content.len());
                                            let args = fc.args.unwrap_or(serde_json::Value::Object(Default::default()));
                                            // Gemini thinking conversations require the
                                            // thought signature replayed on function calls.
                                            let metadata = part.thought_signature.as_ref().map(|sig| {
                                                serde_json::json!({"thought_signature": sig})
                                            });
                                            let idx = content.len();
                                            content.push(Content::ToolCall { provider_metadata: metadata,
                                                id: id.clone(),
                                                name: fc.name.clone(),
                                                arguments: args,
                                            });
                                            let _ = tx.send(StreamEvent::ToolCallStart {
                                                content_index: idx,
                                                id,
                                                name: fc.name,
                                            });
                                            let _ = tx.send(StreamEvent::ToolCallEnd { content_index: idx });
                                            stop_reason = StopReason::ToolUse;
                                        }
                                    }
                                }
                                if let Some(reason) = candidate.finish_reason {
                                    stop_reason = match reason.as_str() {
                                        "STOP" => StopReason::Stop,
                                        "MAX_TOKENS" => StopReason::Length,
                                        _ => StopReason::Stop,
                                    };
                                }
                            }

                            if let Some(u) = parsed.usage_metadata {
                                // promptTokenCount includes cached tokens;
                                // keep `input` as the uncached remainder so
                                // downstream sums don't double-count.
                                usage.input = u
                                    .prompt_token_count
                                    .unwrap_or(0)
                                    .saturating_sub(u.cached_content_token_count.unwrap_or(0));
                                usage.output = u.candidates_token_count.unwrap_or(0);
                                usage.cache_read = u.cached_content_token_count.unwrap_or(0);
                                usage.total_tokens = u.total_token_count.unwrap_or(0);
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
        provider: provider_name.to_string(),
        usage,
        timestamp: now_ms(),
        error_message: None,
    };

    let _ = tx.send(StreamEvent::Done {
        message: message.clone(),
    });
    Ok(message)
}

/// Build the request body for Vertex AI (same format as Google GenAI).
/// Token budget for Vertex's thinkingConfig per level (same scale as Gemini).
fn vertex_thinking_budget(level: ThinkingLevel) -> u32 {
    match level {
        ThinkingLevel::Off => 0,
        ThinkingLevel::Minimal | ThinkingLevel::Low => 1024,
        ThinkingLevel::Medium => 8192,
        ThinkingLevel::High => 24576,
    }
}

fn build_vertex_request_body(config: &StreamConfig) -> serde_json::Value {
    // Same format as Google GenAI
    let mut contents: Vec<serde_json::Value> = Vec::new();

    for msg in &config.messages {
        match msg {
            Message::User { content, .. } => {
                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter(|c| !matches!(c, Content::Text { text } if text.is_empty()))
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(serde_json::json!({"text": text})),
                        Content::Image { data, mime_type } => Some(serde_json::json!({
                            "inlineData": {"mimeType": mime_type, "data": data},
                        })),
                        _ => None,
                    })
                    .collect();
                contents.push(serde_json::json!({"role": "user", "parts": parts}));
            }
            Message::Assistant { content, .. } => {
                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter(|c| !matches!(c, Content::Text { text } if text.is_empty()))
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(serde_json::json!({"text": text})),
                        Content::ToolCall {
                            name,
                            arguments,
                            provider_metadata,
                            ..
                        } => {
                            let mut part = serde_json::json!({
                                "functionCall": {"name": name, "args": arguments},
                            });
                            if let Some(sig) = provider_metadata
                                .as_ref()
                                .and_then(|m| m.get("thought_signature"))
                                .and_then(|v| v.as_str())
                            {
                                part["thoughtSignature"] = serde_json::json!(sig);
                            }
                            Some(part)
                        }
                        _ => None,
                    })
                    .collect();
                contents.push(serde_json::json!({"role": "model", "parts": parts}));
            }
            Message::ToolResult {
                tool_name, content, ..
            } => {
                let text = content
                    .iter()
                    .find_map(|c| match c {
                        Content::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();

                let mut parts = vec![serde_json::json!({
                    "functionResponse": {"name": tool_name, "response": {"result": text}}
                })];

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

    let mut body = serde_json::json!({"contents": contents});

    if !config.system_prompt.is_empty() {
        body["systemInstruction"] = serde_json::json!({"parts": [{"text": config.system_prompt}]});
    }

    let mut gen_config = serde_json::json!({});
    if let Some(max) = config.max_tokens {
        gen_config["maxOutputTokens"] = serde_json::json!(max);
    }
    if let Some(temp) = config.temperature {
        gen_config["temperature"] = serde_json::json!(temp);
    }
    // Thinking: same thinkingConfig as the Gemini API.
    if config.thinking_level != ThinkingLevel::Off {
        gen_config["thinkingConfig"] = serde_json::json!({
            "thinkingBudget": vertex_thinking_budget(config.thinking_level),
            "includeThoughts": true,
        });
    }
    if gen_config != serde_json::json!({}) {
        body["generationConfig"] = gen_config;
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
        body["tools"] = serde_json::json!([{"functionDeclarations": declarations}]);
    }

    body
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(level: ThinkingLevel) -> StreamConfig {
        StreamConfig {
            model: "gemini-2.5-pro".into(),
            system_prompt: "".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            thinking_level: level,
            api_key: "token".into(),
            max_tokens: None,
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        }
    }

    #[test]
    fn tool_call_thought_signature_is_replayed() {
        // Parity with the Gemini API provider: signatures captured into
        // provider_metadata must ride back on functionCall parts.
        let mut c = config(ThinkingLevel::Off);
        c.messages = vec![
            Message::user("go"),
            Message::assistant(
                vec![Content::ToolCall {
                    id: "vertex-fc-0".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Paris"}),
                    provider_metadata: Some(serde_json::json!({"thought_signature": "sig-9"})),
                }],
                StopReason::ToolUse,
                "m",
                "vertex",
                Usage::default(),
            ),
        ];
        let body = build_vertex_request_body(&c);
        let part = &body["contents"][1]["parts"][0];
        assert_eq!(part["functionCall"]["name"], "get_weather");
        assert_eq!(part["thoughtSignature"], "sig-9");
    }

    #[test]
    fn thinking_level_sets_thinking_config() {
        let body = build_vertex_request_body(&config(ThinkingLevel::High));
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            24576
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn thinking_off_omits_thinking_config() {
        let body = build_vertex_request_body(&config(ThinkingLevel::Off));
        assert!(body["generationConfig"]["thinkingConfig"].is_null());
    }
}
