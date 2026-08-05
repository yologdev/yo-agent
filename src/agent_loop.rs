//! The core agent loop: prompt → LLM stream → tool execution → repeat.
//!
//! This is the heart of yoagent. Inspired by pi-agent-core's agent-loop.ts:
//!
//! - `agent_loop()` starts with new prompt messages
//! - `agent_loop_continue()` resumes from existing context
//!
//! Both return a stream of `AgentEvent`s.

use crate::context::{
    self, CompactionStrategy, ContextConfig, ContextTracker, DefaultCompaction, ExecutionLimits,
    ExecutionTracker,
};
use crate::provider::{ModelConfig, StreamConfig, StreamEvent, StreamProvider, ToolDefinition};
use crate::types::*;
use std::sync::Arc;

/// Type alias for convert_to_llm callback.
pub type ConvertToLlmFn = Box<dyn Fn(&[AgentMessage]) -> Vec<Message> + Send + Sync>;
/// Type alias for transform_context callback.
pub type TransformContextFn = Box<dyn Fn(Vec<AgentMessage>) -> Vec<AgentMessage> + Send + Sync>;
/// Type alias for steering/follow-up message callbacks.
pub type GetMessagesFn = Box<dyn Fn() -> Vec<AgentMessage> + Send + Sync>;
/// Called before each LLM turn. Return `false` to abort the loop.
pub type BeforeTurnFn = Arc<dyn Fn(&[AgentMessage], usize) -> bool + Send + Sync>;
/// Called after each LLM turn with the current messages and the turn's usage.
pub type AfterTurnFn = Arc<dyn Fn(&[AgentMessage], &Usage) + Send + Sync>;
/// Called when the LLM returns a `StopReason::Error`.
pub type OnErrorFn = Arc<dyn Fn(&str) + Send + Sync>;
use tokio::sync::mpsc;
use tracing::warn;

/// Configuration for the agent loop
pub struct AgentLoopConfig {
    pub provider: Arc<dyn StreamProvider>,
    pub model: String,
    pub api_key: String,
    pub thinking_level: ThinkingLevel,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,

    /// Optional model configuration for multi-provider support.
    /// When set, passed through to StreamConfig so providers can use
    /// base_url, headers, compat flags, etc.
    pub model_config: Option<ModelConfig>,

    /// Convert AgentMessage[] → Message[] before each LLM call.
    /// Default: keep only LLM-compatible messages.
    pub convert_to_llm: Option<ConvertToLlmFn>,

    /// Transform context before convert_to_llm (for pruning, compaction).
    pub transform_context: Option<TransformContextFn>,

    /// Get steering messages (user interruptions mid-run).
    pub get_steering_messages: Option<GetMessagesFn>,

    /// Get follow-up messages (queued work after agent finishes).
    pub get_follow_up_messages: Option<GetMessagesFn>,

    /// Context window configuration (auto-compaction).
    pub context_config: Option<ContextConfig>,

    /// Custom compaction strategy. When set, replaces the default
    /// `compact_messages()` call. Invoked when `context_config` is `Some`.
    pub compaction_strategy: Option<Arc<dyn CompactionStrategy>>,

    /// Execution limits (max turns, tokens, duration).
    pub execution_limits: Option<ExecutionLimits>,

    /// Prompt caching configuration.
    pub cache_config: CacheConfig,

    /// Tool execution strategy (sequential, parallel, or batched).
    pub tool_execution: ToolExecutionStrategy,

    /// Tool middleware chain — approve/deny/modify every tool call before it
    /// executes (see [`ToolMiddleware`]). Empty = allow all.
    pub tool_middleware: Vec<Arc<dyn ToolMiddleware>>,

    /// Structured-output constraint, passed through to the provider (see
    /// [`OutputSchema`](crate::provider::OutputSchema)). Usually set via
    /// [`Agent::prompt_structured`](crate::Agent::prompt_structured).
    pub output_schema: Option<crate::provider::OutputSchema>,

    /// Retry configuration for transient provider errors.
    pub retry_config: crate::retry::RetryConfig,

    /// Called before each LLM turn. Return `false` to abort the loop.
    pub before_turn: Option<BeforeTurnFn>,
    /// Called after each LLM turn with the current messages and the turn's usage.
    pub after_turn: Option<AfterTurnFn>,
    /// Called when the LLM returns a `StopReason::Error`.
    pub on_error: Option<OnErrorFn>,

    /// Input filters applied to user messages before the LLM call.
    /// Filters run in order; first `Reject` wins and discards any accumulated
    /// warnings. `Warn` messages accumulate and are appended to the user message.
    pub input_filters: Vec<Arc<dyn InputFilter>>,

    /// Optional delay between turns. Useful for rate-limit-sensitive scenarios
    /// (e.g., OAuth tokens with low request-per-minute caps). Skipped on the
    /// first turn so the agent starts immediately.
    pub turn_delay: Option<std::time::Duration>,
}

/// Default convert_to_llm: keep only user/assistant/toolResult messages.
fn default_convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|m| m.as_llm().cloned())
        .collect()
}

/// Start an agent loop with new prompt messages.
pub async fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    tx: mpsc::UnboundedSender<AgentEvent>,
    cancel: tokio_util::sync::CancellationToken,
) -> Vec<AgentMessage> {
    tx.send(AgentEvent::AgentStart).ok();

    // Apply input filters before adding prompts to context
    let prompts = if !config.input_filters.is_empty() {
        let user_text: String = prompts
            .iter()
            .filter_map(|m| {
                if let AgentMessage::Llm(Message::User { content, .. }) = m {
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
                            .join("\n"),
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut warnings: Vec<String> = Vec::new();
        for filter in &config.input_filters {
            match filter.filter(&user_text) {
                FilterResult::Pass => {}
                FilterResult::Warn(w) => warnings.push(w),
                FilterResult::Reject(reason) => {
                    tx.send(AgentEvent::InputRejected {
                        reason: reason.clone(),
                    })
                    .ok();
                    tx.send(AgentEvent::AgentEnd { messages: vec![] }).ok();
                    return vec![];
                }
            }
        }

        // Append warnings to the last user message's content (avoids consecutive user messages)
        if !warnings.is_empty() {
            let warning_text = warnings
                .iter()
                .map(|w| format!("[Warning: {}]", w))
                .collect::<Vec<_>>()
                .join("\n");

            let mut modified = prompts;
            // Find and extend the last user message
            for msg in modified.iter_mut().rev() {
                if let AgentMessage::Llm(Message::User { content, .. }) = msg {
                    content.push(Content::Text { text: warning_text });
                    break;
                }
            }
            modified
        } else {
            prompts
        }
    } else {
        prompts
    };

    let mut new_messages: Vec<AgentMessage> = prompts.clone();

    // Add prompts to context
    for prompt in &prompts {
        context.messages.push(prompt.clone());
    }

    tx.send(AgentEvent::TurnStart).ok();

    // Emit events for each prompt message
    for prompt in &prompts {
        tx.send(AgentEvent::MessageStart {
            message: prompt.clone(),
        })
        .ok();
        tx.send(AgentEvent::MessageEnd {
            message: prompt.clone(),
        })
        .ok();
    }

    {
        use tracing::Instrument;
        run_loop(context, &mut new_messages, config, &tx, &cancel)
            .instrument(tracing::info_span!("agent_loop", model = %config.model))
            .await;
    }

    tx.send(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .ok();
    new_messages
}

/// Continue an agent loop from existing context (for retries).
pub async fn agent_loop_continue(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    tx: mpsc::UnboundedSender<AgentEvent>,
    cancel: tokio_util::sync::CancellationToken,
) -> Vec<AgentMessage> {
    assert!(
        !context.messages.is_empty(),
        "Cannot continue: no messages in context"
    );

    if let Some(last) = context.messages.last() {
        assert!(
            last.role() != "assistant",
            "Cannot continue from assistant message"
        );
    }

    let mut new_messages: Vec<AgentMessage> = Vec::new();

    tx.send(AgentEvent::AgentStart).ok();
    tx.send(AgentEvent::TurnStart).ok();

    {
        use tracing::Instrument;
        run_loop(context, &mut new_messages, config, &tx, &cancel)
            .instrument(tracing::info_span!("agent_loop", model = %config.model))
            .await;
    }

    tx.send(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .ok();
    new_messages
}

/// Main loop logic shared by agent_loop and agent_loop_continue.
///
/// Outer loop: continues when follow-up messages arrive after agent would stop.
/// Inner loop: process tool calls and steering messages.
async fn run_loop(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
) {
    let mut first_turn = true;
    let mut turn_number: usize = 0;
    // Blends real provider usage with estimation for compaction sizing.
    let mut context_tracker = ContextTracker::new();
    let mut tracker = config
        .execution_limits
        .as_ref()
        .map(|limits| ExecutionTracker::new(limits.clone()));

    // Check for steering messages at start
    let mut pending: Vec<AgentMessage> = config
        .get_steering_messages
        .as_ref()
        .map(|f| f())
        .unwrap_or_default();

    // Outer loop: follow-ups after agent would stop
    loop {
        if cancel.is_cancelled() {
            return;
        }

        let mut steering_after_tools: Option<Vec<AgentMessage>> = None;

        // Inner loop: runs at least once, then continues if tool calls or pending messages
        loop {
            if cancel.is_cancelled() {
                return;
            }

            if !first_turn {
                tx.send(AgentEvent::TurnStart).ok();
            } else {
                first_turn = false;
            }

            // Inject pending messages
            if !pending.is_empty() {
                for msg in pending.drain(..) {
                    tx.send(AgentEvent::MessageStart {
                        message: msg.clone(),
                    })
                    .ok();
                    tx.send(AgentEvent::MessageEnd {
                        message: msg.clone(),
                    })
                    .ok();
                    context.messages.push(msg.clone());
                    new_messages.push(msg);
                }
            }

            // Check execution limits
            if let Some(ref tracker) = tracker {
                if let Some(reason) = tracker.check_limits() {
                    warn!("Execution limit reached: {}", reason);
                    let limit_msg = AgentMessage::Llm(Message::User {
                        content: vec![Content::Text {
                            text: format!("[Agent stopped: {}]", reason),
                        }],
                        timestamp: now_ms(),
                    });
                    tx.send(AgentEvent::MessageStart {
                        message: limit_msg.clone(),
                    })
                    .ok();
                    tx.send(AgentEvent::MessageEnd {
                        message: limit_msg.clone(),
                    })
                    .ok();
                    context.messages.push(limit_msg.clone());
                    new_messages.push(limit_msg);
                    return;
                }
            }

            // before_turn callback — abort if it returns false
            if let Some(ref before_turn) = config.before_turn {
                if !before_turn(&context.messages, turn_number) {
                    return;
                }
            }

            // Inter-turn delay — throttle API calls to stay under rate limits.
            // Skipped on the first turn so the agent starts immediately.
            if turn_number > 0 {
                if let Some(delay) = config.turn_delay {
                    tokio::time::sleep(delay).await;
                }
            }

            turn_number += 1;

            // Compact context if configured (tiered: tool outputs → summarize → drop).
            //
            // Calibration: the tracker's hybrid figure is anchored on provider
            // usage, which counts the FULL request (system prompt + tool
            // schemas + any estimation shortfall on messages), while
            // `total_tokens` counts messages only. The difference is a
            // measured overhead; subtracting it from the budget makes the
            // strategy's message-token checks equivalent to "true window
            // occupancy <= max_context_tokens". The static
            // `system_prompt_tokens` reserve is zeroed in the calibrated
            // config because the measured overhead already includes the real
            // system prompt. A floor of 10% of the configured budget
            // guarantees a mis-measured overhead can never wipe the history.
            if let Some(ref ctx_config) = config.context_config {
                let estimated = context::total_tokens(&context.messages);
                let hybrid = context_tracker.estimate_context_tokens(&context.messages);
                let overhead = hybrid.saturating_sub(estimated);
                let calibrated;
                let effective_config = if overhead > 0 {
                    let floor = ctx_config.max_context_tokens / 10;
                    calibrated = ContextConfig {
                        max_context_tokens: ctx_config
                            .max_context_tokens
                            .saturating_sub(overhead)
                            .max(floor),
                        system_prompt_tokens: 0,
                        ..ctx_config.clone()
                    };
                    tracing::debug!(
                        "compaction budget calibrated: {} -> {} (measured overhead: {} tokens)",
                        ctx_config.max_context_tokens,
                        calibrated.max_context_tokens,
                        overhead
                    );
                    &calibrated
                } else {
                    ctx_config
                };
                let strategy: &dyn CompactionStrategy = config
                    .compaction_strategy
                    .as_deref()
                    .unwrap_or(&DefaultCompaction);
                let before_len = context.messages.len();
                context.messages =
                    strategy.compact(std::mem::take(&mut context.messages), effective_config);
                if context.messages.len() != before_len {
                    // Messages shifted; re-baseline from the next real usage.
                    context_tracker.reset();
                }
            }

            // Stream assistant response, under an llm_stream span that
            // records tokens and (when rates are configured) dollar cost.
            let llm_span = tracing::info_span!(
                "llm_stream",
                turn = turn_number,
                model = %config.model,
                tokens_in = tracing::field::Empty,
                tokens_out = tracing::field::Empty,
                tokens_cached = tracing::field::Empty,
                cost_usd = tracing::field::Empty,
                error = tracing::field::Empty,
            );
            let message = {
                use tracing::Instrument;
                stream_assistant_response(context, config, tx, cancel)
                    .instrument(llm_span.clone())
                    .await
            };
            if let Message::Assistant {
                usage, stop_reason, ..
            } = &message
            {
                llm_span.record("error", *stop_reason == StopReason::Error);
                llm_span.record("tokens_in", usage.input);
                llm_span.record("tokens_out", usage.output);
                llm_span.record("tokens_cached", usage.cache_read);
                if let Some(mc) = &config.model_config {
                    if mc.cost.is_configured() {
                        llm_span.record("cost_usd", mc.cost.cost_usd(usage));
                    }
                }
            }
            // Tool-forcing providers (Anthropic) deliver structured output as
            // a forced tool call — unwrap it into plain text BEFORE tool-call
            // extraction, so the loop never tries to execute the synthetic tool.
            let message = unwrap_structured_tool_call(message, config.output_schema.as_ref());

            let agent_msg: AgentMessage = message.clone().into();
            context.messages.push(agent_msg.clone());
            new_messages.push(agent_msg.clone());
            if let Message::Assistant { usage, .. } = &message {
                context_tracker.record_usage(usage, context.messages.len() - 1);
            }

            // Check for error/abort
            if let Message::Assistant {
                ref stop_reason,
                ref error_message,
                ref usage,
                ..
            } = message
            {
                if *stop_reason == StopReason::Error || *stop_reason == StopReason::Aborted {
                    if *stop_reason == StopReason::Error {
                        if let Some(ref on_error) = config.on_error {
                            let err_str = error_message.as_deref().unwrap_or("Unknown error");
                            on_error(err_str);
                        }
                    }
                    // Call after_turn even on error/abort so callers tracking usage don't miss this turn
                    if let Some(ref after_turn) = config.after_turn {
                        after_turn(&context.messages, usage);
                    }
                    tx.send(AgentEvent::TurnEnd {
                        message: agent_msg,
                        tool_results: vec![],
                    })
                    .ok();
                    return;
                }
            }

            // Extract tool calls
            let tool_calls: Vec<_> = match &message {
                Message::Assistant { content, .. } => content
                    .iter()
                    .filter_map(|c| match c {
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                            ..
                        } => Some((id.clone(), name.clone(), arguments.clone())),
                        _ => None,
                    })
                    .collect(),
                _ => vec![],
            };

            let has_tool_calls = !tool_calls.is_empty();
            let mut tool_results: Vec<Message> = Vec::new();

            if has_tool_calls {
                let execution = execute_tool_calls(
                    &context.tools,
                    &tool_calls,
                    tx,
                    cancel,
                    config.get_steering_messages.as_ref(),
                    &config.tool_execution,
                    &config.tool_middleware,
                )
                .await;

                tool_results = execution.tool_results;
                steering_after_tools = execution.steering_messages;

                // Cap oversized output on the way in when configured, so
                // compaction never has to rewrite a tool result the provider
                // has already cached. Per-tool budgets apply, so a tool that
                // tolerates head+tail badly can opt out. The events emitted
                // above still carry the untruncated output — this is a context
                // concern only.
                let append_cap = config
                    .context_config
                    .as_ref()
                    .filter(|c| c.truncate_tool_output_on_append);

                for result in &tool_results {
                    let mut am: AgentMessage = result.clone().into();
                    if let Some(ctx_config) = append_cap {
                        am = context::truncate_tool_output(am, ctx_config);
                    }
                    context.messages.push(am.clone());
                    new_messages.push(am);
                }
            }

            // Track turn for execution limits
            if let Some(ref mut tracker) = tracker {
                let turn_tokens = match &message {
                    Message::Assistant { usage, .. } => {
                        (usage.input + usage.output + usage.cache_read + usage.cache_write) as usize
                    }
                    _ => context::message_tokens(&agent_msg),
                };
                tracker.record_turn(turn_tokens);
            }

            // after_turn callback
            if let Some(ref after_turn) = config.after_turn {
                let usage = match &message {
                    Message::Assistant { usage, .. } => usage.clone(),
                    _ => Usage::default(),
                };
                after_turn(&context.messages, &usage);
            }

            tx.send(AgentEvent::TurnEnd {
                message: agent_msg,
                tool_results,
            })
            .ok();

            // Check steering after turn
            if let Some(steering) = steering_after_tools.take() {
                if !steering.is_empty() {
                    pending = steering;
                    continue;
                }
            }

            pending = config
                .get_steering_messages
                .as_ref()
                .map(|f| f())
                .unwrap_or_default();

            // Exit inner loop if no more tool calls and no pending messages
            if !has_tool_calls && pending.is_empty() {
                break;
            }
        }

        // Agent would stop. Check for follow-ups.
        let follow_ups = config
            .get_follow_up_messages
            .as_ref()
            .map(|f| f())
            .unwrap_or_default();

        if !follow_ups.is_empty() {
            pending = follow_ups;
            continue;
        }

        break;
    }
}

/// Stream an assistant response from the LLM.
async fn stream_assistant_response(
    context: &AgentContext,
    config: &AgentLoopConfig,
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
) -> Message {
    // Apply context transform
    let messages = if let Some(transform) = &config.transform_context {
        transform(context.messages.clone())
    } else {
        context.messages.clone()
    };

    // Convert to LLM messages
    let convert = config.convert_to_llm.as_ref();
    let llm_messages = match convert {
        Some(f) => f(&messages),
        None => default_convert_to_llm(&messages),
    };

    // Build tool definitions
    let tool_defs: Vec<ToolDefinition> = context
        .tools
        .iter()
        .map(|t| ToolDefinition {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters_schema(),
        })
        .collect();

    // Retry loop for transient provider errors
    let retry = &config.retry_config;
    let mut attempt = 0;
    let result = loop {
        let stream_config = StreamConfig {
            model: config.model.clone(),
            system_prompt: context.system_prompt.clone(),
            messages: llm_messages.clone(),
            tools: tool_defs.clone(),
            thinking_level: config.thinking_level,
            api_key: config.api_key.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            model_config: config.model_config.clone(),
            cache_config: config.cache_config.clone(),
            output_schema: config.output_schema.clone(),
        };

        let (stream_tx, mut stream_rx) = mpsc::unbounded_channel();
        let provider_cancel = cancel.clone();

        // Spawn a task to forward events in real-time as the provider streams
        let event_tx = tx.clone();
        let model_for_events = config.model.clone();
        let forward_handle = tokio::spawn(async move {
            let mut partial_message: Option<AgentMessage> = None;
            while let Some(event) = stream_rx.recv().await {
                match &event {
                    StreamEvent::Start => {
                        let placeholder = AgentMessage::Llm(Message::Assistant {
                            content: Vec::new(),
                            stop_reason: StopReason::Stop,
                            model: model_for_events.clone(),
                            provider: String::new(),
                            usage: Usage::default(),
                            timestamp: now_ms(),
                            error_message: None,
                        });
                        partial_message = Some(placeholder.clone());
                        event_tx
                            .send(AgentEvent::MessageStart {
                                message: placeholder,
                            })
                            .ok();
                    }
                    StreamEvent::TextDelta { delta, .. } => {
                        if let Some(ref msg) = partial_message {
                            event_tx
                                .send(AgentEvent::MessageUpdate {
                                    message: msg.clone(),
                                    delta: StreamDelta::Text {
                                        delta: delta.clone(),
                                    },
                                })
                                .ok();
                        }
                    }
                    StreamEvent::ThinkingDelta { delta, .. } => {
                        if let Some(ref msg) = partial_message {
                            event_tx
                                .send(AgentEvent::MessageUpdate {
                                    message: msg.clone(),
                                    delta: StreamDelta::Thinking {
                                        delta: delta.clone(),
                                    },
                                })
                                .ok();
                        }
                    }
                    StreamEvent::ToolCallDelta { delta, .. } => {
                        if let Some(ref msg) = partial_message {
                            event_tx
                                .send(AgentEvent::MessageUpdate {
                                    message: msg.clone(),
                                    delta: StreamDelta::ToolCallDelta {
                                        delta: delta.clone(),
                                    },
                                })
                                .ok();
                        }
                    }
                    StreamEvent::Done { message } => {
                        let am: AgentMessage = message.clone().into();
                        partial_message = Some(am.clone());
                        event_tx.send(AgentEvent::MessageEnd { message: am }).ok();
                    }
                    StreamEvent::Error { message } => {
                        let am: AgentMessage = message.clone().into();
                        if partial_message.is_none() {
                            event_tx
                                .send(AgentEvent::MessageStart {
                                    message: am.clone(),
                                })
                                .ok();
                        }
                        partial_message = Some(am.clone());
                        event_tx.send(AgentEvent::MessageEnd { message: am }).ok();
                    }
                    _ => {}
                }
            }
        });

        // Provider streams concurrently — events are forwarded in real-time
        // When provider returns, stream_tx is dropped, ending the forwarder
        let result = config
            .provider
            .stream(stream_config, stream_tx, provider_cancel)
            .await;

        match &result {
            Err(e) if e.is_retryable() && attempt < retry.max_retries && !cancel.is_cancelled() => {
                // Abort forwarder to prevent forwarding events from failed attempt
                forward_handle.abort();
                attempt += 1;
                // Server-provided Retry-After wins over backoff, but is
                // clamped to max_delay_ms so a bad header can't stall the loop.
                let delay = e
                    .retry_after()
                    .map(|d| d.min(std::time::Duration::from_millis(retry.max_delay_ms)))
                    .unwrap_or_else(|| retry.delay_for_attempt(attempt));
                crate::retry::log_retry(attempt, retry.max_retries, &delay, e);
                tokio::time::sleep(delay).await;
                continue;
            }
            _ => {
                // Final attempt — wait for forwarder to finish processing remaining events
                let _ = forward_handle.await;
                break result;
            }
        }
    };

    match result {
        Ok(msg) => msg,
        Err(e) => {
            warn!("Provider error: {}", e);
            Message::Assistant {
                content: vec![Content::Text {
                    text: String::new(),
                }],
                stop_reason: StopReason::Error,
                model: config.model.clone(),
                provider: "unknown".into(),
                usage: Usage::default(),
                timestamp: now_ms(),
                error_message: Some(e.to_string()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

struct ToolExecutionResult {
    tool_results: Vec<Message>,
    steering_messages: Option<Vec<AgentMessage>>,
}

/// Convert a forced structured-output tool call back into a plain-text
/// assistant message with `StopReason::Stop`. No-op unless a schema is set
/// and the message carries a tool call named after it.
fn unwrap_structured_tool_call(
    message: Message,
    schema: Option<&crate::provider::OutputSchema>,
) -> Message {
    let Some(schema) = schema else {
        return message;
    };
    let Message::Assistant {
        content,
        model,
        provider,
        usage,
        stop_reason,
        error_message,
        ..
    } = &message
    else {
        return message;
    };
    let Some(payload) = content.iter().find_map(|c| match c {
        Content::ToolCall {
            name, arguments, ..
        } if *name == schema.name => Some(arguments.clone()),
        _ => None,
    }) else {
        return message;
    };

    // Remove ONLY the synthetic call; any real tool calls (shouldn't occur
    // under forced tool_choice, but defensively) stay and execute normally.
    // The payload is appended AFTER any preamble text — consumers take the
    // last text block.
    let mut new_content: Vec<Content> = content
        .iter()
        .filter(|c| !matches!(c, Content::ToolCall { name, .. } if *name == schema.name))
        .cloned()
        .collect();
    new_content.push(Content::Text {
        text: payload.to_string(),
    });
    // ToolUse becomes Stop (the forced call was the "answer"); every other
    // stop reason (Length = truncated payload, Error, ...) is preserved so
    // truncation isn't laundered into success.
    let new_stop = if *stop_reason == StopReason::ToolUse {
        StopReason::Stop
    } else {
        stop_reason.clone()
    };
    let rebuilt = Message::assistant(
        new_content,
        new_stop,
        model.clone(),
        provider.clone(),
        usage.clone(),
    );
    // `Message::assistant` nulls `error_message`, so carry it across explicitly.
    // The stop reason is preserved a few lines up for the same reason; dropping
    // the explanation with it leaves the caller a bare `Error` and
    // `StructuredPromptError::Provider { message: "provider error (no detail)" }`.
    match error_message {
        Some(msg) => rebuilt.with_error_message(msg.clone()),
        None => rebuilt,
    }
}

async fn execute_tool_calls(
    tools: &[Box<dyn AgentTool>],
    tool_calls: &[(String, String, serde_json::Value)],
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
    get_steering: Option<&GetMessagesFn>,
    strategy: &ToolExecutionStrategy,
    middleware: &[Arc<dyn ToolMiddleware>],
) -> ToolExecutionResult {
    match strategy {
        ToolExecutionStrategy::Sequential => {
            execute_sequential(tools, tool_calls, tx, cancel, get_steering, middleware).await
        }
        ToolExecutionStrategy::Parallel => {
            execute_batch(tools, tool_calls, tx, cancel, get_steering, middleware).await
        }
        ToolExecutionStrategy::Batched { size } => {
            let mut results: Vec<Message> = Vec::new();
            let mut steering_messages: Option<Vec<AgentMessage>> = None;

            for (batch_idx, batch) in tool_calls.chunks(*size).enumerate() {
                let batch_result = execute_batch(tools, batch, tx, cancel, None, middleware).await;
                results.extend(batch_result.tool_results);

                // Check steering between batches
                if let Some(get_steering_fn) = get_steering {
                    let steering = get_steering_fn();
                    if !steering.is_empty() {
                        steering_messages = Some(steering);
                        // Skip remaining batches
                        let executed = (batch_idx + 1) * *size;
                        if executed < tool_calls.len() {
                            for (skip_id, skip_name, _) in &tool_calls[executed..] {
                                results.push(skip_tool_call(skip_id, skip_name, tx));
                            }
                        }
                        break;
                    }
                }
            }

            ToolExecutionResult {
                tool_results: results,
                steering_messages,
            }
        }
    }
}

/// Execute tool calls one at a time, checking steering between each.
async fn execute_sequential(
    tools: &[Box<dyn AgentTool>],
    tool_calls: &[(String, String, serde_json::Value)],
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
    get_steering: Option<&GetMessagesFn>,
    middleware: &[Arc<dyn ToolMiddleware>],
) -> ToolExecutionResult {
    let mut results: Vec<Message> = Vec::new();
    let mut steering_messages: Option<Vec<AgentMessage>> = None;

    for (index, (id, name, args)) in tool_calls.iter().enumerate() {
        let (result_msg, _is_error) =
            execute_single_tool(tools, id, name, args, tx, cancel, middleware).await;
        results.push(result_msg);

        // Check for steering — skip remaining tools if user interrupted
        if let Some(get_steering_fn) = get_steering {
            let steering = get_steering_fn();
            if !steering.is_empty() {
                steering_messages = Some(steering);
                for (skip_id, skip_name, _) in &tool_calls[index + 1..] {
                    results.push(skip_tool_call(skip_id, skip_name, tx));
                }
                break;
            }
        }
    }

    ToolExecutionResult {
        tool_results: results,
        steering_messages,
    }
}

/// Execute a batch of tool calls concurrently using futures::join_all.
async fn execute_batch(
    tools: &[Box<dyn AgentTool>],
    tool_calls: &[(String, String, serde_json::Value)],
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
    get_steering: Option<&GetMessagesFn>,
    middleware: &[Arc<dyn ToolMiddleware>],
) -> ToolExecutionResult {
    use futures::future::join_all;

    let futures: Vec<_> = tool_calls
        .iter()
        .map(|(id, name, args)| execute_single_tool(tools, id, name, args, tx, cancel, middleware))
        .collect();

    let batch_results = join_all(futures).await;

    let results: Vec<Message> = batch_results.into_iter().map(|(msg, _)| msg).collect();

    // Check steering after batch completes
    let steering_messages = if let Some(get_steering_fn) = get_steering {
        let steering = get_steering_fn();
        if steering.is_empty() {
            None
        } else {
            Some(steering)
        }
    } else {
        None
    };

    ToolExecutionResult {
        tool_results: results,
        steering_messages,
    }
}

/// Execute a single tool call and emit events.
async fn execute_single_tool(
    tools: &[Box<dyn AgentTool>],
    id: &str,
    name: &str,
    args: &serde_json::Value,
    tx: &mpsc::UnboundedSender<AgentEvent>,
    cancel: &tokio_util::sync::CancellationToken,
    middleware: &[Arc<dyn ToolMiddleware>],
) -> (Message, bool) {
    // Middleware chain runs first: each hook may rewrite the args seen by
    // later hooks; the first Deny short-circuits into an error tool result
    // (the LLM sees the reason and can adapt — the loop continues).
    let mut effective_args = args.clone();
    for mw in middleware {
        let call = ToolCallRequest {
            tool_call_id: id,
            tool_name: name,
            args: &effective_args,
        };
        // A panicking middleware must not kill the loop task (which would
        // strip the agent of its tools) — contain it and fail closed.
        let decision = {
            use futures::FutureExt;
            std::panic::AssertUnwindSafe(mw.before_tool(&call))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| {
                    tracing::warn!(tool = name, "tool middleware panicked; denying the call");
                    ToolDecision::Deny("tool middleware panicked".into())
                })
        };
        match decision {
            ToolDecision::Allow => {}
            ToolDecision::Modify(new_args) => effective_args = new_args,
            ToolDecision::Deny(reason) => {
                return denied_tool_call(id, name, &effective_args, &reason, tx);
            }
        }
    }
    let args = &effective_args;

    let tool = tools.iter().find(|t| t.name() == name);

    // The Start event carries the effective (post-middleware) args — what
    // actually runs.
    tx.send(AgentEvent::ToolExecutionStart {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        args: args.clone(),
    })
    .ok();

    let on_update: Option<ToolUpdateFn> = {
        let tx = tx.clone();
        let id = id.to_string();
        let name = name.to_string();
        Some(Arc::new(move |partial: ToolResult| {
            tx.send(AgentEvent::ToolExecutionUpdate {
                tool_call_id: id.clone(),
                tool_name: name.clone(),
                partial_result: partial,
            })
            .ok();
        }))
    };

    let on_progress: Option<ProgressFn> = {
        let tx = tx.clone();
        let id = id.to_string();
        let name = name.to_string();
        Some(Arc::new(move |text: String| {
            tx.send(AgentEvent::ProgressMessage {
                tool_call_id: id.clone(),
                tool_name: name.clone(),
                text,
            })
            .ok();
        }))
    };

    let ctx = ToolContext {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        cancel: cancel.child_token(),
        on_update,
        on_progress,
    };

    let tool_span = tracing::info_span!(
        "tool",
        tool = %name,
        tool_call_id = %id,
        is_error = tracing::field::Empty,
    );
    use tracing::Instrument;
    let (result, is_error) = match tool {
        Some(tool) => {
            let execution = tool
                .execute(args.clone(), ctx)
                .instrument(tool_span.clone())
                .await;
            match execution {
                Ok(r) => (r, false),
                Err(e) => (
                    ToolResult {
                        content: vec![Content::Text {
                            text: e.to_string(),
                        }],
                        details: serde_json::Value::Null,
                    },
                    true,
                ),
            }
        }
        None => (
            ToolResult {
                content: vec![Content::Text {
                    text: format!("Tool {} not found", name),
                }],
                details: serde_json::Value::Null,
            },
            true,
        ),
    };

    tool_span.record("is_error", is_error);

    tx.send(AgentEvent::ToolExecutionEnd {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        result: result.clone(),
        is_error,
    })
    .ok();

    let tool_result_msg = Message::ToolResult {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        content: result.content,
        is_error,
        timestamp: now_ms(),
    };

    tx.send(AgentEvent::MessageStart {
        message: tool_result_msg.clone().into(),
    })
    .ok();
    tx.send(AgentEvent::MessageEnd {
        message: tool_result_msg.clone().into(),
    })
    .ok();

    (tool_result_msg, is_error)
}

/// Emit events and build the error tool result for a middleware-denied call.
/// Start/End are both emitted so UI event pairing stays intact.
fn denied_tool_call(
    id: &str,
    name: &str,
    args: &serde_json::Value,
    reason: &str,
    tx: &mpsc::UnboundedSender<AgentEvent>,
) -> (Message, bool) {
    // Operator-visible signal: without this, a denial exists only in the
    // event stream / message history, invisible to telemetry.
    tracing::warn!(
        tool = name,
        tool_call_id = id,
        reason,
        "tool call denied by middleware"
    );
    tx.send(AgentEvent::ToolExecutionStart {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        args: args.clone(),
    })
    .ok();

    let result = ToolResult {
        content: vec![Content::Text {
            text: format!("Tool call denied: {}", reason),
        }],
        details: serde_json::Value::Null,
    };

    tx.send(AgentEvent::ToolExecutionEnd {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        result: result.clone(),
        is_error: true,
    })
    .ok();

    let msg = Message::ToolResult {
        tool_call_id: id.to_string(),
        tool_name: name.to_string(),
        content: result.content,
        is_error: true,
        timestamp: now_ms(),
    };

    tx.send(AgentEvent::MessageStart {
        message: msg.clone().into(),
    })
    .ok();
    tx.send(AgentEvent::MessageEnd {
        message: msg.clone().into(),
    })
    .ok();

    (msg, true)
}

fn skip_tool_call(
    tool_call_id: &str,
    tool_name: &str,
    tx: &mpsc::UnboundedSender<AgentEvent>,
) -> Message {
    let result = ToolResult {
        content: vec![Content::Text {
            text: "Skipped due to queued user message.".into(),
        }],
        details: serde_json::Value::Null,
    };

    tx.send(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        args: serde_json::Value::Null,
    })
    .ok();

    tx.send(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        result: result.clone(),
        is_error: true,
    })
    .ok();

    let msg = Message::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        content: result.content,
        is_error: true,
        timestamp: now_ms(),
    };

    tx.send(AgentEvent::MessageStart {
        message: msg.clone().into(),
    })
    .ok();
    tx.send(AgentEvent::MessageEnd {
        message: msg.clone().into(),
    })
    .ok();

    msg
}
