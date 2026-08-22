//! Anthropic Claude provider (Messages API with streaming)

use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, warn};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// Trim a model-supplied string for inclusion in an error or log line.
///
/// The accumulator can be a whole `max_tokens` worth of JSON — a truncated file
/// write or a base64 blob. That string reaches `on_error`, the assistant
/// message, session JSONL, and (via `SubAgentTool`) the parent model's context,
/// so quoting it whole floods every one of them.
fn truncate_for_error(s: &str) -> String {
    const MAX: usize = 200;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let head: String = s.chars().take(MAX).collect();
    format!("{head}\u{2026} ({} bytes total)", s.len())
}

/// Map an Anthropic `stop_reason` onto [`StopReason`], recording a diagnosis for
/// the terminal ones that carry one.
///
/// Every documented value is listed explicitly. A catch-all that silently means
/// "finished normally" is how `end_turn` — the ordinary completion — would land
/// in the unknown arm and warn on every healthy turn, and how a resumable state
/// would be handed back as a finished answer.
fn map_stop_reason(reason: &str, error_message: &mut Option<String>) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::Length,
        "model_context_window_exceeded" => {
            // In-stream overflow (HTTP 200). Map to the same Error +
            // overflow-phrase shape as an HTTP 400 overflow so
            // `Message::is_context_overflow()` and compaction-retry hooks keep
            // working.
            warn!("Anthropic context window exceeded mid-stream");
            *error_message = Some("model_context_window_exceeded".into());
            StopReason::Error
        }
        "refusal" => {
            warn!("Anthropic declined the request (stop_reason=refusal)");
            *error_message =
                Some("Request declined by the model's safety system (stop_reason: refusal)".into());
            StopReason::Refusal
        }
        "pause_turn" => {
            // The model paused mid-turn and expects the conversation to be
            // re-sent to continue — emitted when a server-side tool loop hits
            // its iteration limit. There is no resume path here, so reporting
            // it as a normal stop would hand back a truncated answer as though
            // it were complete.
            warn!("Anthropic paused the turn (stop_reason=pause_turn); response is incomplete");
            *error_message = Some(
                "Anthropic paused the turn (stop_reason: pause_turn); the response is \
                 incomplete and must be resumed by re-sending the conversation"
                    .into(),
            );
            StopReason::Error
        }
        other => {
            warn!("unrecognized Anthropic stop_reason '{other}'; treating it as a normal stop");
            StopReason::Stop
        }
    }
}

/// Resolve the request URL: `{base_url}/messages` when a `ModelConfig` is set
/// (e.g. a gateway like OpenCode Zen), the official endpoint otherwise.
fn request_url(config: &StreamConfig) -> String {
    match &config.model_config {
        Some(mc) => {
            let base = mc.base_url.trim_end_matches('/');
            // Configs created before 0.9.0 carry the un-versioned official
            // host (base_url used to be ignored); keep them working.
            if base == "https://api.anthropic.com" {
                API_URL.to_string()
            } else {
                format!("{}/messages", base)
            }
        }
        None => API_URL.to_string(),
    }
}

/// Effective Anthropic quirk flags. `None` means current-generation defaults.
fn anthropic_compat(config: &StreamConfig) -> crate::provider::AnthropicCompat {
    config
        .model_config
        .as_ref()
        .and_then(|mc| mc.anthropic.clone())
        .unwrap_or_default()
}

pub struct AnthropicProvider;

#[async_trait]
impl StreamProvider for AnthropicProvider {
    fn protocol(&self) -> Option<crate::provider::ApiProtocol> {
        Some(crate::provider::ApiProtocol::AnthropicMessages)
    }

    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        let is_oauth = config.api_key.contains("sk-ant-oat");
        let body = build_request_body(&config, is_oauth);
        let url = request_url(&config);
        debug!(
            "Anthropic request: model={}, url={}, oauth={}",
            config.model, url, is_oauth
        );
        let client = reqwest::Client::new();
        let mut builder = client
            .post(&url)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");

        // Custom headers from ModelConfig. A user-supplied `authorization`
        // header takes over auth entirely (bring-your-own-token gateways).
        let mut user_auth = false;
        if let Some(mc) = &config.model_config {
            for (key, value) in &mc.headers {
                if key.eq_ignore_ascii_case("authorization") {
                    user_auth = true;
                }
                builder = builder.header(key, value);
            }
        }

        let compat = anthropic_compat(&config);
        if user_auth {
            // Auth fully managed via ModelConfig.headers.
        } else if is_oauth {
            // OAuth token — Bearer auth with Claude Code identity headers
            builder = builder
                .header("authorization", format!("Bearer {}", config.api_key))
                .header(
                    "anthropic-beta",
                    "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
                )
                .header("anthropic-dangerous-direct-browser-access", "true")
                .header("user-agent", "claude-cli/2.1.2 (external, cli)")
                .header("x-app", "cli");
        } else if compat.bearer_auth {
            // OpenAI-style gateway speaking the Anthropic protocol
            builder = builder.header("authorization", format!("Bearer {}", config.api_key));
        } else {
            builder = builder.header("x-api-key", &config.api_key);
        }

        let request = builder.json(&body);

        let mut es =
            EventSource::new(request).map_err(|e| ProviderError::Network(e.to_string()))?;

        let mut content: Vec<Content> = Vec::new();
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::Stop;
        // Whether a `message_delta` carrying a terminal `stop_reason` arrived —
        // i.e. the model finished, even if `message_stop` never shows up. Gates
        // the terminator-less-close guard below. Deliberately not set by a
        // delta without a stop_reason: that proves nothing about completeness.
        let mut saw_stop_reason = false;
        let mut error_message: Option<String> = None;

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
                        Some(Ok(Event::Open)) => {}
                        Some(Ok(Event::Message(msg))) => {
                            match msg.event.as_str() {
                                "message_start" => {
                                    if let Ok(data) = serde_json::from_str::<AnthropicMessageStart>(&msg.data) {
                                        usage.input = data.message.usage.input_tokens;
                                        usage.cache_read = data.message.usage.cache_read_input_tokens;
                                        usage.cache_write = data.message.usage.cache_creation_input_tokens;
                                    }
                                }
                                "content_block_start" => {
                                    if let Ok(data) = serde_json::from_str::<AnthropicContentBlockStart>(&msg.data) {
                                        let idx = data.index as usize;
                                        match data.content_block {
                                            AnthropicContentBlock::Text { .. } => {
                                                while content.len() <= idx {
                                                    content.push(Content::Text { text: String::new() });
                                                }
                                            }
                                            AnthropicContentBlock::Thinking { .. } => {
                                                while content.len() <= idx {
                                                    content.push(Content::Thinking { thinking: String::new(), signature: None });
                                                }
                                            }
                                            AnthropicContentBlock::ToolUse { id, name, .. } => {
                                                while content.len() <= idx {
                                                    content.push(Content::ToolCall { provider_metadata: None,
                                                        id: id.clone(),
                                                        name: name.clone(),
                                                        arguments: serde_json::Value::Object(Default::default()),
                                                    });
                                                }
                                                let _ = tx.send(StreamEvent::ToolCallStart {
                                                    content_index: idx,
                                                    id,
                                                    name,
                                                });
                                            }
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    if let Ok(data) = serde_json::from_str::<AnthropicContentBlockDelta>(&msg.data) {
                                        let idx = data.index as usize;
                                        match data.delta {
                                            AnthropicDelta::TextDelta { text } => {
                                                if let Some(Content::Text { text: ref mut t }) = content.get_mut(idx) {
                                                    t.push_str(&text);
                                                }
                                                let _ = tx.send(StreamEvent::TextDelta {
                                                    content_index: idx,
                                                    delta: text,
                                                });
                                            }
                                            AnthropicDelta::ThinkingDelta { thinking } => {
                                                if let Some(Content::Thinking { thinking: ref mut t, .. }) = content.get_mut(idx) {
                                                    t.push_str(&thinking);
                                                }
                                                let _ = tx.send(StreamEvent::ThinkingDelta {
                                                    content_index: idx,
                                                    delta: thinking,
                                                });
                                            }
                                            AnthropicDelta::InputJsonDelta { partial_json } => {
                                                // Accumulate JSON into a buffer for this tool call
                                                if let Some(Content::ToolCall { ref mut arguments, .. }) = content.get_mut(idx) {
                                                    // Append to string buffer stored in arguments
                                                    // We accumulate the raw JSON string and parse it at content_block_stop
                                                    let buf = arguments
                                                        .as_object_mut()
                                                        .and_then(|o| o.get_mut("__partial_json"))
                                                        .and_then(|v| v.as_str().map(|s| s.to_string()));
                                                    let new_buf = format!("{}{}", buf.unwrap_or_default(), partial_json);
                                                    if let Some(obj) = arguments.as_object_mut() {
                                                        obj.insert("__partial_json".into(), serde_json::Value::String(new_buf));
                                                    }
                                                }
                                                let _ = tx.send(StreamEvent::ToolCallDelta {
                                                    content_index: idx,
                                                    delta: partial_json,
                                                });
                                            }
                                            AnthropicDelta::SignatureDelta { signature } => {
                                                if let Some(Content::Thinking { signature: ref mut s, .. }) = content.get_mut(idx) {
                                                    *s = Some(signature);
                                                }
                                            }
                                        }
                                    }
                                }
                                "content_block_stop" => match serde_json::from_str::<serde_json::Value>(&msg.data) {
                                    // Skipping this silently would leave a tool
                                    // call's accumulator unparsed *and* never emit
                                    // ToolCallEnd, so a UI tracking the call's
                                    // lifecycle hangs it open forever.
                                    Err(e) => warn!(
                                        "content_block_stop body is not JSON ({e}); \
                                         any tool call it closes is left unfinalized: {}",
                                        msg.data
                                    ),
                                    Ok(data) => match data["index"].as_u64() {
                                        // Defaulting to 0 would close the wrong
                                        // block: block 0 gets block N's arguments
                                        // and block N keeps its raw accumulator.
                                        None => warn!(
                                            "content_block_stop without a numeric index; \
                                             refusing to guess which block it closes: {}",
                                            msg.data
                                        ),
                                        Some(idx) => {
                                            let idx = idx as usize;
                                            // A block whose accumulator does not
                                            // parse is left carrying it. The sweep
                                            // after the loop is what fails the
                                            // turn — doing it here would miss the
                                            // blocks this arm never reaches (a
                                            // non-JSON stop event, or one with no
                                            // index), which leak the same way.
                                            if let Some(Content::ToolCall { name, arguments, .. }) = content.get_mut(idx) {
                                                if let Some(partial) = arguments.as_object()
                                                    .and_then(|o| o.get("__partial_json"))
                                                    .and_then(|v| v.as_str())
                                                    .map(|s| s.to_string())
                                                {
                                                    match resolve_tool_arguments(&partial) {
                                                        Some(parsed) => *arguments = parsed,
                                                        None => warn!(
                                                            "tool call `{}` has malformed JSON arguments: {}",
                                                            name,
                                                            truncate_for_error(&partial)
                                                        ),
                                                    }
                                                }
                                            }
                                            let _ = tx.send(StreamEvent::ToolCallEnd { content_index: idx });
                                        }
                                    },
                                },
                                "message_delta" => {
                                    if let Ok(data) = serde_json::from_str::<AnthropicMessageDelta>(&msg.data) {
                                        // Only a delta that actually carried a
                                        // terminal stop_reason proves the response
                                        // is finished. An intermediate delta (usage
                                        // update, empty `delta`) parses fine and
                                        // must NOT arm the clean-EOF guard.
                                        // A delta with no stop_reason leaves the
                                        // current one alone — overwriting it would
                                        // clobber a terminal reason an earlier
                                        // delta already set. Only a delta that
                                        // actually carried one proves the response
                                        // is finished, so only that arms the
                                        // clean-EOF guard.
                                        if let Some(reason) = data.delta.stop_reason.as_deref() {
                                            saw_stop_reason = true;
                                            stop_reason =
                                                map_stop_reason(reason, &mut error_message);
                                        }
                                        // A usage-less delta must not zero a real
                                        // count: `AnthropicUsage` defaults every
                                        // field, so an absent block would otherwise
                                        // overwrite the output tokens with 0.
                                        if let Some(delta_usage) = data.usage {
                                            usage.output = delta_usage.output_tokens;
                                        }
                                    }
                                }
                                "message_stop" => break,
                                "ping" => {}
                                "error" => {
                                    let provider_err = classify_sse_error_event(&msg.data);
                                    warn!("Anthropic stream error: {}", provider_err);
                                    return Err(provider_err);
                                }
                                other => {
                                    debug!("Unknown Anthropic event: {}", other);
                                }
                            }
                        }
                        // A gateway can deliver a complete response and then
                        // close without the `message_stop` terminator —
                        // `message_delta` already carried stop_reason and
                        // usage, so the message is whole. Treat as clean EOF;
                        // classifying it would make it retryable (Network) and
                        // re-bill a finished response. Same shape as the
                        // openai_compat DONE-less guard (#76).
                        Some(Err(reqwest_eventsource::Error::StreamEnded)) if saw_stop_reason => {
                            // A stop_reason arrived, but a tool_use block may
                            // still be unterminated: `content_block_stop` is
                            // what parses the accumulated `__partial_json`
                            // buffer into real arguments. Returning that buffer
                            // as a tool call would run the tool with the
                            // sentinel key as its input — treat as truncation.
                            if content.iter().any(|c| {
                                matches!(c, Content::ToolCall { arguments, .. }
                                    if arguments.get("__partial_json").is_some())
                            }) {
                                warn!(
                                    "stream ended after message_delta with an unterminated \
                                     tool_use block; treating as truncation"
                                );
                                return Err(ProviderError::Network(
                                    "stream ended with an incomplete tool_use block".into(),
                                ));
                            }
                            debug!("provider closed stream without message_stop after message_delta");
                            break;
                        }
                        Some(Err(e)) => {
                            let provider_err = classify_eventsource_error(e).await;
                            warn!("SSE error: {}", provider_err);
                            return Err(provider_err);
                        }
                    }
                }
            }
        }

        let has_tool_calls = content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }));
        // Invariant: no tool call leaves this provider still carrying the
        // streaming accumulator. `content_block_stop` is what turns
        // `__partial_json` into real arguments; if it never arrived, or arrived
        // unusable (not JSON, no index, or holding JSON that does not parse),
        // the sentinel is still there — and a tool handed
        // `{"__partial_json": ...}` runs on its defaults instead of what the
        // model asked for. One sweep covers every route to that state.
        let unusable: Vec<(String, String)> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall {
                    name, arguments, ..
                } => arguments
                    .get("__partial_json")
                    .and_then(|v| v.as_str())
                    .map(|partial| (name.clone(), partial.to_string())),
                _ => None,
            })
            .collect();

        if !unusable.is_empty() {
            let detail = unusable
                .iter()
                .map(|(name, partial)| format!("`{name}` ({})", truncate_for_error(partial)))
                .collect::<Vec<_>>()
                .join(", ");
            let note = format!("tool call(s) with unusable arguments, not executed: {detail}");
            // Append rather than replace: a refusal or an overflow reported by
            // `message_delta` carries its own diagnosis, and the overflow phrase
            // is what `Message::is_context_overflow()` matches on.
            error_message = Some(match error_message.take() {
                Some(existing) => format!("{existing}; {note}"),
                None => note,
            });
            // A refusal is the more specific verdict and nothing can execute
            // once the calls are gone, so leave it standing.
            if stop_reason != StopReason::Refusal {
                stop_reason = StopReason::Error;
            }
            // Every tool call goes, not just the unusable one. The turn executes
            // none of them, so any that survived would return to the API as a
            // `tool_use` with no `tool_result` and be rejected on the *next*
            // request — breaking the conversation rather than just this turn.
            // Replacing in place keeps `content.len()` aligned with the
            // provider's block indices.
            for block in content.iter_mut() {
                if let Content::ToolCall { name, .. } = block {
                    let name = name.clone();
                    *block = Content::Text {
                        text: format!(
                            "[tool call `{name}` not executed: the turn failed on unusable tool arguments]"
                        ),
                    };
                }
            }
        }

        // Never let the tool-call fallback mask a refusal or an error signal.
        if has_tool_calls && !matches!(stop_reason, StopReason::Refusal | StopReason::Error) {
            stop_reason = StopReason::ToolUse;
        }

        let message = Message::Assistant {
            content,
            stop_reason,
            model: config.model.clone(),
            // Gateways that speak the Anthropic Messages protocol (OpenCode
            // Zen, Copilot) carry their own provider name for cost and session
            // attribution — don't overwrite it. Falls back to "anthropic" when
            // no ModelConfig was supplied.
            provider: config
                .model_config
                .as_ref()
                .map(|mc| mc.provider.clone())
                .unwrap_or_else(|| "anthropic".into()),
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

// ---------------------------------------------------------------------------
// Anthropic API request/response types
// ---------------------------------------------------------------------------

fn build_request_body(config: &StreamConfig, is_oauth: bool) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for msg in &config.messages {
        match msg {
            Message::User { content, .. } => {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content_to_anthropic(content),
                }));
            }
            Message::Assistant { content, .. } => {
                // A refused or errored turn can leave an assistant message with
                // no serializable content; the API rejects empty assistant
                // content blocks mid-conversation, so skip such messages.
                let blocks = content_to_anthropic(content);
                if !blocks.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": blocks,
                    }));
                }
            }
            Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let result_content = if content.iter().any(|c| matches!(c, Content::Image { .. })) {
                    // Multi-content with images: use array format
                    serde_json::json!(content_to_anthropic(content))
                } else {
                    // Text-only: use string shorthand
                    let text = content
                        .iter()
                        .find_map(|c| match c {
                            Content::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    serde_json::json!(text)
                };

                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result_content,
                        "is_error": is_error,
                    }],
                }));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Prompt caching — place cache_control breakpoints based on CacheConfig.
    //
    // Anthropic caches the full prefix (tools → system → messages) up to each
    // breakpoint. We use up to 3 breakpoints:
    //   1. System prompt (stable across turns)
    //   2. Last tool definition (tools rarely change)
    //   3. Second-to-last message (conversation history grows, cache the prefix)
    //
    // When caching is disabled or strategy is Disabled, no markers are added.
    // -----------------------------------------------------------------------
    let cache = &config.cache_config;
    let caching_enabled = cache.enabled && cache.strategy != CacheStrategy::Disabled;
    let (cache_system, cache_tools, cache_messages) = match &cache.strategy {
        CacheStrategy::Auto => (true, true, true),
        CacheStrategy::Disabled => (false, false, false),
        CacheStrategy::Manual {
            cache_system,
            cache_tools,
            cache_messages,
        } => (*cache_system, *cache_tools, *cache_messages),
    };

    // Breakpoint 3: scan backwards from second-to-last message to find one with
    // non-empty content to place the cache breakpoint on
    if caching_enabled && cache_messages && messages.len() >= 2 {
        for idx in (0..messages.len() - 1).rev() {
            if let Some(content) = messages[idx]["content"].as_array_mut() {
                if let Some(last_block) = content.last_mut() {
                    let is_empty_text = last_block.get("type").and_then(|t| t.as_str())
                        == Some("text")
                        && last_block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .is_empty();
                    if !is_empty_text {
                        last_block["cache_control"] = serde_json::json!({"type": "ephemeral"});
                        break;
                    }
                }
            }
        }
    }

    // Default max_tokens: explicit request value, then the model's configured
    // default, then a conservative fallback.
    let mut max_tokens = config
        .max_tokens
        .or(config.model_config.as_ref().map(|mc| mc.max_tokens))
        .unwrap_or(8192);
    let compat = anthropic_compat(config);
    // Legacy (budget-based) thinking requires max_tokens > budget_tokens.
    if config.thinking_level != ThinkingLevel::Off && !compat.adaptive_thinking {
        let budget = legacy_thinking_budget(config.thinking_level);
        if max_tokens <= budget {
            debug!(
                "Raising max_tokens from {} to {} to exceed the thinking budget",
                max_tokens,
                budget + 1024
            );
            max_tokens = budget + 1024;
        }
    }

    let mut body = serde_json::json!({
        "model": config.model,
        "max_tokens": max_tokens,
        "stream": true,
        "messages": messages,
    });

    // Breakpoint 1: system prompt
    if is_oauth {
        let mut system_blocks = vec![serde_json::json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        })];
        if !config.system_prompt.is_empty() {
            system_blocks.push(serde_json::json!({
                "type": "text",
                "text": config.system_prompt,
            }));
        }
        // Cache the last system block
        if caching_enabled && cache_system {
            if let Some(last) = system_blocks.last_mut() {
                last["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
        }
        body["system"] = serde_json::json!(system_blocks);
    } else if !config.system_prompt.is_empty() {
        let mut block = serde_json::json!({
            "type": "text",
            "text": config.system_prompt,
        });
        if caching_enabled && cache_system {
            block["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }
        body["system"] = serde_json::json!([block]);
    }

    // Breakpoint 2: last tool definition (tools are stable between turns)
    if !config.tools.is_empty() {
        let mut tools: Vec<serde_json::Value> = config
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        if caching_enabled && cache_tools {
            if let Some(last_tool) = tools.last_mut() {
                last_tool["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
        }
        body["tools"] = serde_json::json!(tools);
    }

    // Structured outputs via tool-forcing: append a synthetic tool built from
    // the schema and force the model to call it. The loop unwraps the forced
    // call back into plain text (`unwrap_structured_tool_call`).
    if let Some(schema) = &config.output_schema {
        let synthetic = serde_json::json!({
            "name": schema.name,
            "description": "Produce the final answer in the required schema.",
            "input_schema": schema.schema,
        });
        match body.get_mut("tools").and_then(|v| v.as_array_mut()) {
            Some(arr) => arr.push(synthetic),
            None => body["tools"] = serde_json::json!([synthetic]),
        }
        body["tool_choice"] = serde_json::json!({ "type": "tool", "name": schema.name });
    }

    // Forced tool_choice and extended thinking are mutually exclusive at the
    // API level — a structured-output request wins and thinking is skipped
    // for this call (warned, not silent).
    let thinking_requested = config.thinking_level != ThinkingLevel::Off;
    if thinking_requested && config.output_schema.is_some() {
        tracing::warn!(
            "structured outputs force tool_choice, which Anthropic rejects with \
             extended thinking; thinking is disabled for this request"
        );
    }
    if thinking_requested && config.output_schema.is_none() {
        if compat.adaptive_thinking {
            // Current generation (Claude 4.6+ / Fable 5): adaptive thinking with
            // an effort hint. Budget-based thinking is rejected with a 400.
            let effort = match config.thinking_level {
                ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
                ThinkingLevel::Medium => "medium",
                ThinkingLevel::High => "high",
                ThinkingLevel::Off => unreachable!(),
            };
            body["thinking"] = serde_json::json!({ "type": "adaptive" });
            body["output_config"] = serde_json::json!({ "effort": effort });
        } else {
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": legacy_thinking_budget(config.thinking_level),
            });
        }
    }

    if let Some(temp) = config.temperature {
        body["temperature"] = serde_json::json!(temp);
    }

    body
}

/// Budget tokens for legacy (pre-4.6) extended thinking. The API requires a
/// minimum of 1024. (`Off` returns 0 but never reaches a thinking-enabled
/// request — both call sites guard on `!= ThinkingLevel::Off`.)
fn legacy_thinking_budget(level: ThinkingLevel) -> u32 {
    match level {
        ThinkingLevel::Off => 0,
        ThinkingLevel::Minimal | ThinkingLevel::Low => 1024,
        ThinkingLevel::Medium => 2048,
        ThinkingLevel::High => 8192,
    }
}

fn content_to_anthropic(content: &[Content]) -> Vec<serde_json::Value> {
    content
        .iter()
        .filter(|c| !matches!(c, Content::Text { text } if text.is_empty()))
        .map(|c| match c {
            Content::Text { text } => serde_json::json!({"type": "text", "text": text}),
            Content::Image { data, mime_type } => serde_json::json!({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": data},
            }),
            Content::Thinking {
                thinking,
                signature,
            } => serde_json::json!({
                "type": "thinking",
                "thinking": thinking,
                "signature": signature.as_deref().unwrap_or(""),
            }),
            Content::ToolCall {
                id,
                name,
                arguments,
                ..
            } => serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": arguments,
            }),
        })
        .collect()
}

// Anthropic SSE event types
#[derive(Deserialize)]
struct AnthropicMessageStart {
    message: AnthropicMessageInfo,
}

#[derive(Deserialize)]
struct AnthropicMessageInfo {
    usage: AnthropicUsage,
}

#[derive(Deserialize, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

#[derive(Deserialize)]
struct AnthropicContentBlockStart {
    index: u64,
    content_block: AnthropicContentBlock,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize)]
struct AnthropicContentBlockDelta {
    index: u64,
    delta: AnthropicDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Deserialize)]
struct AnthropicMessageDelta {
    delta: AnthropicMessageDeltaInner,
    // Gateways relaying this protocol sometimes omit `usage` on message_delta.
    // Without the default the whole event fails to parse, silently dropping the
    // stop_reason it carried (downgrading tool_use/max_tokens/refusal to Stop).
    // `Option` rather than a defaulted struct so an absent block is
    // distinguishable from a reported zero — otherwise a trailing usage-less
    // delta overwrites a real output count with 0.
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicMessageDeltaInner {
    stop_reason: Option<String>,
}

/// Resolve a tool call's accumulated `input_json_delta` text into arguments.
///
/// `None` means malformed, and the caller leaves the `__partial_json` sentinel
/// in place so the post-stream sweep fails the turn rather than running the
/// tool on its defaults.
///
/// An **empty** accumulator is not malformed. A tool with no parameters has no
/// JSON to stream, and Anthropic still emits an `input_json_delta` carrying
/// `""` — so `from_str("")` failed with "EOF while parsing a value", the
/// sentinel survived, and the sweep rejected the whole turn. Every
/// no-argument tool (`get_status`, `list_files`, …) was uncallable.
fn resolve_tool_arguments(partial: &str) -> Option<serde_json::Value> {
    if partial.trim().is_empty() {
        return Some(serde_json::Value::Object(Default::default()));
    }
    serde_json::from_str(partial).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::traits::ToolDefinition;

    fn make_config(cache: CacheConfig) -> StreamConfig {
        StreamConfig {
            model: "claude-sonnet-4-20250514".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![
                Message::user("Hello"),
                Message::User {
                    content: vec![Content::Text {
                        text: "What is 2+2?".into(),
                    }],
                    timestamp: 0,
                },
            ],
            tools: vec![ToolDefinition {
                name: "bash".into(),
                description: "Run commands".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            thinking_level: ThinkingLevel::Off,
            api_key: "test-key".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: None,
            cache_config: cache,
            output_schema: None,
        }
    }

    #[test]
    fn test_cache_auto_places_all_breakpoints() {
        let body = build_request_body(&make_config(CacheConfig::default()), false);

        // System prompt should have cache_control
        let system = &body["system"][0];
        assert_eq!(system["cache_control"]["type"], "ephemeral");

        // Last tool should have cache_control
        let tools = body["tools"].as_array().unwrap();
        let last_tool = tools.last().unwrap();
        assert_eq!(last_tool["cache_control"]["type"], "ephemeral");

        // Second-to-last message should have cache_control
        let msgs = body["messages"].as_array().unwrap();
        let second_to_last = &msgs[msgs.len() - 2];
        let content = second_to_last["content"].as_array().unwrap();
        let last_block = content.last().unwrap();
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_cache_disabled_no_breakpoints() {
        let config = CacheConfig {
            enabled: false,
            strategy: CacheStrategy::Auto,
        };
        let body = build_request_body(&make_config(config), false);

        // System prompt should NOT have cache_control
        let system = &body["system"][0];
        assert!(system.get("cache_control").is_none());

        // Tools should NOT have cache_control
        let tools = body["tools"].as_array().unwrap();
        assert!(tools.last().unwrap().get("cache_control").is_none());

        // Messages should NOT have cache_control on any block
        let msgs = body["messages"].as_array().unwrap();
        for msg in msgs {
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    assert!(block.get("cache_control").is_none());
                }
            }
        }
    }

    #[test]
    fn test_cache_manual_system_only() {
        let config = CacheConfig {
            enabled: true,
            strategy: CacheStrategy::Manual {
                cache_system: true,
                cache_tools: false,
                cache_messages: false,
            },
        };
        let body = build_request_body(&make_config(config), false);

        // System: cached
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        // Tools: not cached
        assert!(body["tools"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .get("cache_control")
            .is_none());
        // Messages: not cached
        let msgs = body["messages"].as_array().unwrap();
        let second = &msgs[msgs.len() - 2];
        let content = second["content"].as_array().unwrap();
        assert!(content.last().unwrap().get("cache_control").is_none());
    }

    #[test]
    fn test_usage_cache_hit_rate() {
        let usage = Usage {
            input: 100,
            output: 50,
            cache_read: 900,
            cache_write: 0,
            total_tokens: 1050,
        };
        let rate = usage.cache_hit_rate();
        assert!((rate - 0.9).abs() < 0.001); // 900 / (100 + 900 + 0) = 0.9

        let empty = Usage::default();
        assert_eq!(empty.cache_hit_rate(), 0.0);
    }

    #[test]
    fn structured_output_disables_thinking() {
        // Forced tool_choice + extended thinking is an API-level conflict:
        // the schema wins and no thinking field may be emitted.
        let mut config = make_config(CacheConfig::default());
        config.thinking_level = ThinkingLevel::High;
        config.output_schema = Some(crate::provider::OutputSchema::new(
            "structured_output",
            serde_json::json!({"type": "object"}),
        ));
        let body = build_request_body(&config, false);
        assert!(body["thinking"].is_null(), "thinking must be skipped");
        assert!(body["output_config"].is_null());
        assert_eq!(body["tool_choice"]["name"], "structured_output");
    }

    #[test]
    fn test_structured_output_forces_synthetic_tool() {
        let mut config = make_config(CacheConfig::default());
        config.output_schema = Some(crate::provider::OutputSchema::new(
            "structured_output",
            serde_json::json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
        ));
        let body = build_request_body(&config, false);

        let tools = body["tools"].as_array().unwrap();
        let synthetic = tools.last().unwrap();
        assert_eq!(synthetic["name"], "structured_output");
        assert_eq!(
            synthetic["input_schema"]["properties"]["answer"]["type"],
            "string"
        );
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "structured_output");
    }

    #[test]
    fn test_tool_result_with_image() {
        let config = StreamConfig {
            model: "claude-sonnet-4-20250514".into(),
            system_prompt: "".into(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        provider_metadata: None,
                        id: "tc-1".into(),
                        name: "read_file".into(),
                        arguments: serde_json::json!({"path": "test.png"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "tc-1".into(),
                    tool_name: "read_file".into(),
                    content: vec![
                        Content::Text {
                            text: "screenshot".into(),
                        },
                        Content::Image {
                            data: "aW1hZ2VkYXRh".into(),
                            mime_type: "image/png".into(),
                        },
                    ],
                    is_error: false,
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test-key".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: None,
            cache_config: CacheConfig {
                enabled: false,
                strategy: CacheStrategy::Disabled,
            },
            output_schema: None,
        };

        let body = build_request_body(&config, false);
        let msgs = body["messages"].as_array().unwrap();
        // The ToolResult message (second message)
        let tool_msg = &msgs[1];
        let tool_result = &tool_msg["content"][0];
        assert_eq!(tool_result["type"], "tool_result");
        // content should be an array (not a string) since it has images
        let content = tool_result["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
    }

    #[test]
    fn test_tool_result_text_only_uses_string() {
        let config = StreamConfig {
            model: "claude-sonnet-4-20250514".into(),
            system_prompt: "".into(),
            messages: vec![
                Message::Assistant {
                    content: vec![Content::ToolCall {
                        provider_metadata: None,
                        id: "tc-1".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "echo hi"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    usage: Usage::default(),
                    timestamp: 0,
                    error_message: None,
                },
                Message::ToolResult {
                    tool_call_id: "tc-1".into(),
                    tool_name: "bash".into(),
                    content: vec![Content::Text {
                        text: "hello".into(),
                    }],
                    is_error: false,
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test-key".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: None,
            cache_config: CacheConfig {
                enabled: false,
                strategy: CacheStrategy::Disabled,
            },
            output_schema: None,
        };

        let body = build_request_body(&config, false);
        let msgs = body["messages"].as_array().unwrap();
        let tool_result = &msgs[1]["content"][0];
        // Text-only: content should be a plain string
        assert_eq!(tool_result["content"], "hello");
    }

    #[test]
    fn test_content_to_anthropic_filters_empty_text() {
        let content = vec![
            Content::Text { text: "".into() },
            Content::Text {
                text: "hello".into(),
            },
            Content::Text { text: "".into() },
        ];
        let result = content_to_anthropic(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["text"], "hello");
    }

    #[test]
    fn test_cache_control_not_set_on_empty_text_block() {
        let config = StreamConfig {
            model: "claude-sonnet-4-20250514".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![
                Message::User {
                    content: vec![Content::Text {
                        text: "first message".into(),
                    }],
                    timestamp: 0,
                },
                Message::User {
                    content: vec![Content::Text { text: "".into() }],
                    timestamp: 0,
                },
                Message::User {
                    content: vec![Content::Text {
                        text: "last".into(),
                    }],
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test-key".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };
        let body = build_request_body(&config, false);
        let msgs = body["messages"].as_array().unwrap();
        // The second-to-last message has only an empty text block which gets filtered,
        // so its content array should be empty
        let second_to_last = &msgs[msgs.len() - 2];
        let content = second_to_last["content"].as_array().unwrap();
        assert!(
            content.is_empty(),
            "empty text blocks should be filtered out"
        );

        // Cache breakpoint should fall back to the first message instead
        let first = &msgs[0];
        let first_content = first["content"].as_array().unwrap();
        let last_block = first_content.last().unwrap();
        assert_eq!(
            last_block["cache_control"]["type"], "ephemeral",
            "cache_control should fall back to an earlier message with content"
        );
    }

    #[test]
    fn test_cache_breakpoint_falls_back_when_second_to_last_is_empty() {
        // When the second-to-last message has only empty text blocks, the cache
        // breakpoint should scan backwards and land on an earlier non-empty message.
        let config = StreamConfig {
            model: "claude-sonnet-4-20250514".into(),
            system_prompt: "You are helpful.".into(),
            messages: vec![
                Message::User {
                    content: vec![Content::Text {
                        text: "first message".into(),
                    }],
                    timestamp: 0,
                },
                Message::User {
                    content: vec![Content::Text { text: "".into() }],
                    timestamp: 0,
                },
                Message::User {
                    content: vec![Content::Text {
                        text: "last message".into(),
                    }],
                    timestamp: 0,
                },
            ],
            tools: vec![],
            thinking_level: ThinkingLevel::Off,
            api_key: "test-key".into(),
            max_tokens: Some(1024),
            temperature: None,
            model_config: None,
            cache_config: CacheConfig::default(),
            output_schema: None,
        };

        let body = build_request_body(&config, false);
        let msgs = body["messages"].as_array().unwrap();

        // cache_control should be placed on the first message (fallback)
        let first_content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(
            first_content.last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn test_adaptive_thinking_by_default() {
        let mut config = make_config(CacheConfig::default());
        config.thinking_level = ThinkingLevel::High;

        let body = build_request_body(&config, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn test_adaptive_thinking_effort_mapping() {
        for (level, effort) in [
            (ThinkingLevel::Minimal, "low"),
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "medium"),
            (ThinkingLevel::High, "high"),
        ] {
            let mut config = make_config(CacheConfig::default());
            config.thinking_level = level;
            let body = build_request_body(&config, false);
            assert_eq!(body["output_config"]["effort"], effort);
        }
    }

    #[test]
    fn test_legacy_thinking_budget_clamped_to_api_minimum() {
        let mut config = make_config(CacheConfig::default());
        config.thinking_level = ThinkingLevel::Minimal;
        let mut mc = crate::provider::ModelConfig::anthropic("claude-sonnet-4-5", "Sonnet 4.5");
        mc.anthropic = Some(crate::provider::AnthropicCompat::legacy());
        config.model_config = Some(mc);

        let body = build_request_body(&config, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        // API minimum is 1024
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
        // max_tokens must exceed budget_tokens even though the request asked for 1024
        assert!(body["max_tokens"].as_u64().unwrap() > 1024);
    }

    #[test]
    fn test_thinking_off_sends_no_thinking_field() {
        let config = make_config(CacheConfig::default());
        let body = build_request_body(&config, false);
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn test_max_tokens_falls_back_to_model_config() {
        let mut config = make_config(CacheConfig::default());
        config.max_tokens = None;
        config.model_config = Some(crate::provider::ModelConfig::anthropic(
            "claude-sonnet-5",
            "Claude Sonnet 5",
        ));

        let body = build_request_body(&config, false);
        assert_eq!(body["max_tokens"], 16_000);
    }

    #[test]
    fn test_explicit_max_tokens_beats_model_config() {
        let mut config = make_config(CacheConfig::default());
        config.max_tokens = Some(1024);
        config.model_config = Some(crate::provider::ModelConfig::claude_sonnet_5());

        let body = build_request_body(&config, false);
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn test_adaptive_thinking_does_not_clamp_max_tokens() {
        let mut config = make_config(CacheConfig::default());
        config.thinking_level = ThinkingLevel::High;
        config.max_tokens = Some(1024);

        let body = build_request_body(&config, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["max_tokens"], 1024);
    }

    #[test]
    fn test_empty_assistant_message_is_skipped() {
        let mut config = make_config(CacheConfig::default());
        // A refused turn leaves an assistant message with no serializable content.
        config.messages = vec![
            Message::user("Hello"),
            Message::Assistant {
                content: vec![],
                stop_reason: StopReason::Refusal,
                model: "claude-sonnet-5".into(),
                provider: "anthropic".into(),
                usage: Usage::default(),
                timestamp: 0,
                error_message: Some("refused".into()),
            },
            Message::user("Try again"),
        ];

        let body = build_request_body(&config, false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "empty assistant message must be dropped");
        assert!(msgs.iter().all(|m| m["role"] == "user"));
    }

    #[test]
    fn test_legacy_official_base_url_without_version_still_works() {
        // Configs persisted before 0.9.0 carry the un-versioned host.
        let mut config = make_config(CacheConfig::default());
        let mut mc = crate::provider::ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
        mc.base_url = "https://api.anthropic.com".into();
        config.model_config = Some(mc);
        assert_eq!(
            request_url(&config),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_request_url_from_model_config_base_url() {
        let mut config = make_config(CacheConfig::default());
        assert_eq!(
            request_url(&config),
            "https://api.anthropic.com/v1/messages"
        );

        let mut mc = crate::provider::ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5");
        mc.base_url = "https://opencode.ai/zen/v1".into();
        config.model_config = Some(mc.clone());
        assert_eq!(request_url(&config), "https://opencode.ai/zen/v1/messages");

        // trailing slash is tolerated
        mc.base_url = "https://opencode.ai/zen/v1/".into();
        config.model_config = Some(mc);
        assert_eq!(request_url(&config), "https://opencode.ai/zen/v1/messages");
    }

    #[test]
    fn test_default_base_url_yields_official_endpoint() {
        let mut config = make_config(CacheConfig::default());
        config.model_config = Some(crate::provider::ModelConfig::anthropic(
            "claude-opus-4-8",
            "Claude Opus 4.8",
        ));
        assert_eq!(
            request_url(&config),
            "https://api.anthropic.com/v1/messages"
        );
    }
}

#[cfg(test)]
mod no_arg_tool_calls {
    use super::resolve_tool_arguments;

    /// A tool with no parameters must be callable.
    ///
    /// Found on the 0.18 line by a live smoke harness, not by the suite:
    /// `MockProvider` never streams SSE, so nothing exercised the
    /// `input_json_delta` accumulator. Anthropic answered a no-argument tool
    /// call with an empty argument stream and the whole turn failed with
    /// "tool call(s) with unusable arguments".
    #[test]
    fn an_empty_argument_stream_is_an_empty_object() {
        for partial in ["", "   ", "\n"] {
            assert_eq!(
                resolve_tool_arguments(partial),
                Some(serde_json::json!({})),
                "a no-parameter tool streams no JSON; {partial:?} must resolve to {{}}"
            );
        }
    }

    #[test]
    fn well_formed_arguments_parse() {
        assert_eq!(
            resolve_tool_arguments(r#"{"service":"api"}"#),
            Some(serde_json::json!({"service": "api"}))
        );
    }

    /// Truncated input must still fail, or a cut-off stream would silently run
    /// the tool on its defaults — the case the sentinel exists for.
    #[test]
    fn malformed_arguments_still_fail() {
        for partial in [r#"{"service":"#, "not json", "{"] {
            assert_eq!(
                resolve_tool_arguments(partial),
                None,
                "{partial:?} must not resolve"
            );
        }
    }
}
