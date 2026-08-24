//! Mock provider for testing. No real API calls.

use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use tokio::sync::mpsc;

/// A mock response: either plain text or tool calls.
///
/// `#[non_exhaustive]`: this is a test double whose shapes grow with the
/// features under test — `TextWithUsage` was itself a later addition.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MockResponse {
    Text(String),
    ToolCalls(Vec<MockToolCall>),
    /// Text plus a chosen `Usage`, for asserting on token accounting.
    ///
    /// `Text` reports `Usage::default()`, which cannot distinguish a rollup
    /// that sums from one that copies the last turn.
    TextWithUsage(String, Usage),
}

#[derive(Debug, Clone)]
pub struct MockToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    #[allow(dead_code)]
    pub provider_metadata: Option<serde_json::Value>,
}

/// Mock LLM provider for tests. Supply a sequence of responses.
pub struct MockProvider {
    responses: std::sync::Mutex<Vec<MockResponse>>,
    /// Whether to reject a request whose transcript a real provider would.
    ///
    /// On by default, and that default is the point. This type used to discard
    /// `StreamConfig` entirely, so 600+ tests drove the agent loop without one
    /// of them noticing when it built a message sequence every provider
    /// rejects. Loop detection shipped exactly that bug to a release branch:
    /// a steering message injected between an assistant's `tool_use` blocks and
    /// their `tool_result`s, which poisons the agent for every later prompt.
    ///
    /// Turning this off is legitimate for a test that deliberately constructs a
    /// malformed history — say the compaction orphan-handling paths — but it
    /// should be deliberate and commented.
    validate_transcript: bool,
}

/// Reject a message sequence a real provider would reject.
///
/// The invariant every provider enforces: an assistant's `tool_use` blocks must
/// be answered by their `tool_result`s before anything else intervenes.
/// Anthropic returns "tool_use ids were found without tool_result blocks
/// immediately after"; OpenAI returns the equivalent about tool messages. The
/// crate treats this as sacred elsewhere — `context.rs` calls an unanswered
/// call "an orphan every provider rejects", and `llm_compaction.rs` has a
/// dedicated test for it — but nothing enforced it on the loop's own output.
///
/// Returns the reason a provider would give, or `None` if the transcript is
/// well-formed.
fn transcript_violation(messages: &[Message]) -> Option<String> {
    let mut pending: Vec<String> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        match msg {
            Message::Assistant { content, .. } => {
                if !pending.is_empty() {
                    return Some(format!(
                        "assistant message at [{i}] while tool calls {pending:?} are still \
                         unanswered"
                    ));
                }
                pending = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::ToolCall { id, .. } => Some(id.clone()),
                        _ => None,
                    })
                    .collect();
            }
            Message::ToolResult { tool_call_id, .. } => {
                match pending.iter().position(|p| p == tool_call_id) {
                    Some(at) => {
                        pending.remove(at);
                    }
                    None => {
                        return Some(format!(
                            "tool_result at [{i}] for {tool_call_id:?} answers no open tool call"
                        ))
                    }
                }
            }
            Message::User { .. } => {
                if !pending.is_empty() {
                    return Some(format!(
                        "a user message at [{i}] lands between an assistant's tool_use and its \
                         tool_result — {pending:?} still unanswered"
                    ));
                }
            }
        }
    }
    None
}

impl MockProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            validate_transcript: true,
        }
    }

    /// Accept message sequences a real provider would reject.
    ///
    /// For tests that construct a malformed history *on purpose* — orphan
    /// handling, compaction edge cases, recovery paths. Say why at the call
    /// site; the default exists because the alternative let a real bug through.
    pub fn without_transcript_validation(mut self) -> Self {
        self.validate_transcript = false;
        self
    }

    /// Convenience: provider that always returns the same text
    pub fn text(text: impl Into<String>) -> Self {
        Self::new(vec![MockResponse::Text(text.into())])
    }

    /// Convenience: sequence of text responses
    pub fn texts(texts: Vec<impl Into<String>>) -> Self {
        Self::new(
            texts
                .into_iter()
                .map(|t| MockResponse::Text(t.into()))
                .collect(),
        )
    }
}

#[async_trait]
impl StreamProvider for MockProvider {
    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        // A mock that accepts anything tests the loop against a provider that
        // does not exist. Panic rather than return an error: a malformed
        // transcript is a defect in the code under test, and the loop is
        // designed to survive provider *errors*, so returning one would be
        // swallowed by the very retry path that should never see this.
        if self.validate_transcript {
            if let Some(why) = transcript_violation(&config.messages) {
                panic!(
                    "MockProvider received a transcript a real provider would reject: {why}.\n\n\
                     This is the shape that poisons an agent — the history is kept, so every \
                     later prompt fails too. If the malformed sequence is deliberate, use \
                     `MockProvider::without_transcript_validation()` and say why.\n\n\
                     Messages: {:#?}",
                    config.messages
                );
            }
        }
        let response = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                MockResponse::Text("(no more mock responses)".into())
            } else {
                responses.remove(0)
            }
        };

        if cancel.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }

        let _ = tx.send(StreamEvent::Start);

        let message = match response {
            MockResponse::TextWithUsage(text, usage) => {
                let _ = tx.send(StreamEvent::TextDelta {
                    content_index: 0,
                    delta: text.clone(),
                });
                Message::Assistant {
                    content: vec![Content::Text { text }],
                    stop_reason: StopReason::Stop,
                    model: "mock".into(),
                    provider: "mock".into(),
                    usage,
                    timestamp: now_ms(),
                    error_message: None,
                }
            }
            MockResponse::Text(text) => {
                let _ = tx.send(StreamEvent::TextDelta {
                    content_index: 0,
                    delta: text.clone(),
                });
                Message::Assistant {
                    content: vec![Content::Text { text }],
                    stop_reason: StopReason::Stop,
                    model: "mock".into(),
                    provider: "mock".into(),
                    usage: Usage::default(),
                    timestamp: now_ms(),
                    error_message: None,
                }
            }
            MockResponse::ToolCalls(calls) => {
                let content: Vec<Content> = calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| {
                        let id = format!("mock-tool-{}", i);
                        let _ = tx.send(StreamEvent::ToolCallStart {
                            content_index: i,
                            id: id.clone(),
                            name: call.name.clone(),
                        });
                        let _ = tx.send(StreamEvent::ToolCallEnd { content_index: i });
                        Content::ToolCall {
                            id,
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            provider_metadata: call.provider_metadata.clone(),
                        }
                    })
                    .collect();

                Message::Assistant {
                    content,
                    stop_reason: StopReason::ToolUse,
                    model: "mock".into(),
                    provider: "mock".into(),
                    usage: Usage::default(),
                    timestamp: now_ms(),
                    error_message: None,
                }
            }
        };

        let _ = tx.send(StreamEvent::Done {
            message: message.clone(),
        });
        Ok(message)
    }
}
