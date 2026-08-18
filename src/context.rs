//! Context window management — smart truncation and token counting.
//!
//! The #1 engineering challenge for agents. This module provides:
//! - Token estimation (fast, no external deps)
//! - Tiered compaction (tool output truncation → turn summarization → full summary)
//! - Execution limits (max turns, tokens, duration)
//!
//! Designed based on Claude Code's approach: clear old tool outputs first,
//! then summarize conversation if needed.

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token estimate: ~4 chars per token for English text.
/// Good enough for context budgeting. Use tiktoken-rs for precision.
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Estimate tokens for a single message
pub fn message_tokens(msg: &AgentMessage) -> usize {
    match msg {
        AgentMessage::Llm(m) => match m {
            Message::User { content, .. } => content_tokens(content) + 4,
            Message::Assistant { content, .. } => content_tokens(content) + 4,
            Message::ToolResult {
                content, tool_name, ..
            } => content_tokens(content) + estimate_tokens(tool_name) + 8,
        },
        AgentMessage::Extension(ext) => estimate_tokens(&ext.data.to_string()) + 4,
    }
}

fn content_tokens(content: &[Content]) -> usize {
    content
        .iter()
        .map(|c| match c {
            Content::Text { text } => estimate_tokens(text),
            Content::Image { data, .. } => {
                // Estimate tokens from base64 data length:
                // base64 len * 3/4 = raw bytes; ~750 bytes per token for images.
                // Floor at 85 (Anthropic minimum), cap at 16000.
                let raw_bytes = data.len() * 3 / 4;
                (raw_bytes / 750).clamp(85, 16_000)
            }
            Content::Thinking { thinking, .. } => estimate_tokens(thinking),
            Content::ToolCall {
                name, arguments, ..
            } => estimate_tokens(name) + estimate_tokens(&arguments.to_string()) + 8,
        })
        .sum()
}

/// Estimate total tokens for a message list
pub fn total_tokens(messages: &[AgentMessage]) -> usize {
    messages.iter().map(message_tokens).sum()
}

// ---------------------------------------------------------------------------
// Context tracking (real usage + estimates)
// ---------------------------------------------------------------------------

/// Tracks context size using real token counts from provider responses
/// combined with estimates for messages added after the last response.
///
/// This gives more accurate context size tracking than pure estimation,
/// since providers report actual token counts in their usage data.
///
/// # Example
///
/// ```rust
/// use yoagent::context::ContextTracker;
/// use yoagent::types::Usage;
///
/// let mut tracker = ContextTracker::new();
/// // After receiving an assistant response with usage data:
/// tracker.record_usage(&Usage { input: 1500, output: 200, ..Default::default() }, 3);
/// ```
pub struct ContextTracker {
    /// Last known total token count from provider usage
    last_usage_tokens: Option<usize>,
    /// Index of the message that had the last usage
    last_usage_index: Option<usize>,
}

impl ContextTracker {
    pub fn new() -> Self {
        Self {
            last_usage_tokens: None,
            last_usage_index: None,
        }
    }

    /// Record usage from an assistant response.
    ///
    /// Call this after each assistant message to update the tracker
    /// with real token counts from the provider.
    pub fn record_usage(&mut self, usage: &Usage, message_index: usize) {
        let total = usage.input + usage.output + usage.cache_read + usage.cache_write;
        if total > 0 {
            self.last_usage_tokens = Some(total as usize);
            self.last_usage_index = Some(message_index);
        }
    }

    /// Estimate current context size.
    ///
    /// Uses real usage from the last assistant response as a baseline,
    /// then adds estimates (chars/4) for any messages added since.
    /// Falls back to pure estimation if no usage data is available.
    pub fn estimate_context_tokens(&self, messages: &[AgentMessage]) -> usize {
        match (self.last_usage_tokens, self.last_usage_index) {
            (Some(usage_tokens), Some(idx)) if idx < messages.len() => {
                let trailing: usize = messages[idx + 1..].iter().map(message_tokens).sum();
                usage_tokens + trailing
            }
            _ => total_tokens(messages),
        }
    }

    /// Reset tracking (e.g. after compaction replaces messages).
    pub fn reset(&mut self) {
        self.last_usage_tokens = None;
        self.last_usage_index = None;
    }
}

impl Default for ContextTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Context configuration
// ---------------------------------------------------------------------------

/// Configuration for context management.
///
/// Sizes the built-in tiered compaction. [`LlmCompaction`](crate::LlmCompaction),
/// the summarizing alternative, reads `keep_first`, `keep_recent` and the token
/// budget the same way, but sizes its *spliced* result by its own retained-tail
/// setting rather than by `compact_target_ratio` / `compact_headroom_turns` —
/// which still apply on its deterministic fallback path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum context tokens (leave room for response)
    pub max_context_tokens: usize,
    /// Tokens reserved for the system prompt
    pub system_prompt_tokens: usize,
    /// Minimum recent messages to always keep (full detail)
    pub keep_recent: usize,
    /// Minimum first messages to always keep
    pub keep_first: usize,
    /// Max lines to keep per tool output, for tools with no override below.
    ///
    /// Tuned for command output — build logs, test runs, `grep` — where
    /// head+tail is a good cut: the first error is at the top, the summary at
    /// the bottom, and the middle is repetition.
    pub tool_output_max_lines: usize,
    /// Per-tool overrides for `tool_output_max_lines`, keyed by tool name.
    ///
    /// One global number cannot serve every tool. Head+tail truncation suits
    /// command output but is the *wrong* cut for a file read, where the middle
    /// is usually the part that matters — a paging read tool bounds itself far
    /// better than a blind truncation can. The default therefore exempts the
    /// built-in `read_file` (see [`DEFAULT_READ_MAX_LINES`]).
    ///
    /// `usize::MAX` disables truncation for a tool.
    ///
    /// [`DEFAULT_READ_MAX_LINES`]: crate::tools::DEFAULT_READ_MAX_LINES
    #[serde(default)]
    pub tool_output_max_lines_overrides: HashMap<String, usize>,
    /// Fraction of the budget that lossy compaction (Level 2/3) aims for.
    ///
    /// Compaction still *triggers* at the full budget, but once it has to
    /// summarize or drop history it reduces to `budget * ratio` instead of to
    /// whatever just barely fits. Without this headroom the very next turn
    /// crosses the budget again and history is rewritten every single turn,
    /// which discards the provider's prefix cache each time.
    ///
    /// Clamped to `[0.05, 1.0]` (a non-finite value falls back to `1.0`).
    /// `1.0` restores the old compact-to-just-fit behaviour. Default: `0.7`.
    ///
    /// Acts as a ceiling on retention when `compact_headroom_turns` is also
    /// set — the headroom policy may compact harder, never softer.
    pub compact_target_ratio: f32,
    /// Compact hard enough to buy this many more turns before the next
    /// compaction, at the session's observed growth rate.
    ///
    /// A fixed ratio has no idea how fast a session is growing, so the
    /// headroom it leaves is arbitrary and the interval between compactions
    /// collapses as history accumulates. This targets the interval directly:
    ///
    /// ```text
    /// target = budget - turns × observed_growth_per_turn
    /// ```
    ///
    /// The effective ratio is `min(target / budget, compact_target_ratio)`,
    /// floored at [`MIN_HEADROOM_RATIO`], so this can only make compaction
    /// more aggressive than the ratio alone — never less.
    ///
    /// Growth is measured by the agent loop, so this is only honoured there;
    /// a direct [`compact_messages`] call uses `compact_target_ratio` as-is.
    /// `None` disables the policy. Default: `Some(30)`.
    #[serde(default = "default_headroom_turns")]
    pub compact_headroom_turns: Option<usize>,
    /// Apply `tool_output_max_lines` when a tool result is appended, rather
    /// than only once the session is over budget.
    ///
    /// Retroactive truncation is the single largest source of prefix-cache
    /// loss: a tool result that has already been sent in full is rewritten
    /// later, invalidating the cache from that message onward. Capping on the
    /// way in means the bytes the provider cached are the bytes that stay.
    /// It also slows context growth, so compaction runs less often.
    ///
    /// The cost is that a long tool output is trimmed even when the budget
    /// had room for it — which is why tools that tolerate it badly are exempted
    /// via `tool_output_max_lines_overrides` rather than by turning this off.
    /// The full output is always visible in the
    /// [`AgentEvent`](crate::types::AgentEvent) stream either way.
    ///
    /// On by default. Only honoured by the agent loop when a `ContextConfig`
    /// is set.
    pub truncate_tool_output_on_append: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 100_000,
            system_prompt_tokens: 4_000,
            keep_recent: 10,
            keep_first: 2,
            tool_output_max_lines: 200,
            tool_output_max_lines_overrides: default_tool_output_overrides(),
            compact_target_ratio: 0.7,
            compact_headroom_turns: default_headroom_turns(),
            truncate_tool_output_on_append: true,
        }
    }
}

/// Lower bound on the ratio the headroom policy may derive.
///
/// A session whose growth per turn approaches the whole budget would otherwise
/// ask compaction to discard essentially everything.
pub const MIN_HEADROOM_RATIO: f32 = 0.15;

fn default_headroom_turns() -> Option<usize> {
    Some(30)
}

/// Tools whose output the default head+tail cap would damage.
///
/// `read_file` bounds itself by paging (`DEFAULT_READ_MAX_LINES`), which is
/// both lossless and directed; cutting the middle out of a source file on top
/// of that would remove exactly the part the agent asked for.
fn default_tool_output_overrides() -> HashMap<String, usize> {
    HashMap::from([("read_file".to_string(), usize::MAX)])
}

impl ContextConfig {
    /// Derive a context config from a model's context window size.
    ///
    /// Reserves 20% of the context window for output tokens, uses the rest
    /// as the compaction budget. All other settings use defaults.
    pub fn from_context_window(context_window: u32) -> Self {
        let max_context_tokens = (context_window as usize) * 80 / 100;
        Self {
            max_context_tokens,
            ..Default::default()
        }
    }

    /// The compaction ratio to use given the session's observed growth rate.
    ///
    /// Returns `compact_target_ratio` unchanged when no headroom policy is
    /// set, or when the growth rate is not yet known. Otherwise derives a
    /// ratio that leaves room for `compact_headroom_turns` more turns, and
    /// takes whichever of the two is more aggressive.
    ///
    /// See [`compact_headroom_turns`](Self::compact_headroom_turns).
    pub fn effective_target_ratio(&self, growth_tokens_per_turn: f64) -> f32 {
        let Some(turns) = self.compact_headroom_turns else {
            return self.compact_target_ratio;
        };
        // NaN and non-positive growth both mean "no usable estimate yet".
        if turns == 0 || !growth_tokens_per_turn.is_finite() || growth_tokens_per_turn <= 0.0 {
            return self.compact_target_ratio;
        }
        let budget = self
            .max_context_tokens
            .saturating_sub(self.system_prompt_tokens) as f64;
        if budget <= 0.0 {
            return self.compact_target_ratio;
        }
        let target = budget - (turns as f64) * growth_tokens_per_turn;
        let derived = (target / budget) as f32;
        derived
            .min(self.compact_target_ratio)
            .max(MIN_HEADROOM_RATIO)
    }

    /// The line cap that applies to a given tool's output.
    pub fn max_lines_for(&self, tool_name: &str) -> usize {
        self.tool_output_max_lines_overrides
            .get(tool_name)
            .copied()
            .unwrap_or(self.tool_output_max_lines)
    }

    /// The token count lossy compaction reduces to, given the trigger budget.
    fn compaction_target(&self, budget: usize) -> usize {
        let ratio = if self.compact_target_ratio.is_finite() {
            self.compact_target_ratio.clamp(0.05, 1.0)
        } else {
            1.0
        };
        ((budget as f64) * (ratio as f64)) as usize
    }
}

// ---------------------------------------------------------------------------
// Compaction strategy
// ---------------------------------------------------------------------------

/// Strategy for compacting messages when context exceeds budget.
///
/// Implement this to customize what happens during compaction:
/// - Index discarded content into a memory store before removal
/// - Apply custom preservation rules (e.g., always keep decisions)
/// - Emit metadata about what was compressed
///
/// See the [Custom Compaction](https://yologdev.github.io/yoagent/concepts/agent-loop.html#custom-compaction)
/// docs for examples.
pub trait CompactionStrategy: Send + Sync {
    /// Compact messages to fit within the token budget defined by `config`.
    ///
    /// Called before each LLM turn when `context_config` is set.
    fn compact(&self, messages: Vec<AgentMessage>, config: &ContextConfig) -> Vec<AgentMessage>;
}

/// Default 3-level compaction: truncate tool outputs → summarize turns → drop middle.
///
/// This is used automatically when no custom `CompactionStrategy` is set.
/// You can also compose it inside a custom strategy — run your logic first,
/// then delegate to `compact_messages()` for the actual reduction.
///
/// This is deterministic and free, but the lossy tiers discard early decisions
/// outright. [`LlmCompaction`](crate::LlmCompaction) is the alternative: it
/// spends tokens on a background summarization request and splices the result
/// in as prose. It costs money per compaction and does not reduce prefix-cache
/// breaks — see its module docs for the trade-off.
pub struct DefaultCompaction;

impl CompactionStrategy for DefaultCompaction {
    fn compact(&self, messages: Vec<AgentMessage>, config: &ContextConfig) -> Vec<AgentMessage> {
        compact_messages(messages, config)
    }
}

// ---------------------------------------------------------------------------
// Tiered compaction
// ---------------------------------------------------------------------------

/// Compact messages to fit within the token budget using tiered strategy.
///
/// - Level 1: Truncate tool outputs (keep head + tail)
/// - Level 2: Summarize old turns (replace details with one-liner)
/// - Level 3: Drop old messages (keep first + recent only)
///
/// Each level is tried in order. Returns as soon as messages fit.
///
/// # Prefix-cache behaviour
///
/// Every rewrite of an already-sent message discards the provider's prefix
/// cache from that point on, so compaction is built to keep history
/// byte-stable:
///
/// - Level 1 is idempotent — re-running it on settled history is a no-op, so
///   staying above the budget does not re-truncate the same outputs.
/// - Level 1 alone only has to reach the budget; the lossy levels aim for
///   `compact_target_ratio` of it, so once history is rewritten it takes many
///   turns before it must be rewritten again.
/// - Level 3 drops the smallest span that reaches the target and its marker
///   text is constant, so a boundary that lands in the same place twice
///   reproduces the same bytes.
pub fn compact_messages(messages: Vec<AgentMessage>, config: &ContextConfig) -> Vec<AgentMessage> {
    let budget = config
        .max_context_tokens
        .saturating_sub(config.system_prompt_tokens);

    // Already fits?
    if total_tokens(&messages) <= budget {
        return messages;
    }

    let target = config.compaction_target(budget);
    let before = total_tokens(&messages);

    // Level 1: Truncate tool outputs. Checked against the full budget, not the
    // target: this level is idempotent and cheap to repeat, so there is nothing
    // to gain by escalating to the lossy levels before we have to.
    let compacted = level1_truncate_tool_outputs(&messages, config);
    let after_l1 = total_tokens(&compacted);
    if after_l1 < before {
        tracing::debug!(
            "compaction level 1: tool outputs truncated, {} -> {} tokens",
            before,
            after_l1
        );
    }
    if after_l1 <= budget {
        return compacted;
    }

    // Level 2: Summarize old turns (keep recent N full, summarize the rest)
    let before_l2 = compacted.len();
    let compacted = level2_summarize_old_turns(&compacted, config.keep_recent);
    if compacted.len() != before_l2 {
        tracing::debug!(
            "compaction level 2: old turns summarized, {} -> {} messages ({} tokens)",
            before_l2,
            compacted.len(),
            total_tokens(&compacted)
        );
    }
    if total_tokens(&compacted) <= target {
        return compacted;
    }

    // Level 3: Drop the smallest middle span that reaches the target
    level3_drop_middle(&compacted, config, target)
}

/// Level 1: Truncate long tool outputs to head + tail.
///
/// This is the cheapest compaction — preserves conversation structure,
/// just removes verbose tool output middles. In practice this saves
/// 50-70% of context in coding sessions.
fn level1_truncate_tool_outputs(
    messages: &[AgentMessage],
    config: &ContextConfig,
) -> Vec<AgentMessage> {
    messages
        .iter()
        .map(|msg| truncate_tool_output(msg.clone(), config))
        .collect()
}

/// Cap a single tool result's text at the tool's line budget, keeping head and
/// tail.
///
/// The budget comes from [`ContextConfig::max_lines_for`], so a tool that
/// tolerates head+tail badly can be exempted. Non-tool-result messages pass
/// through untouched, and the operation is idempotent, so applying it when the
/// message is first appended means compaction never has to rewrite it later —
/// which is what keeps the provider's prefix cache intact. See
/// [`ContextConfig::truncate_tool_output_on_append`].
pub fn truncate_tool_output(msg: AgentMessage, config: &ContextConfig) -> AgentMessage {
    match msg {
        AgentMessage::Llm(Message::ToolResult {
            tool_call_id,
            tool_name,
            content,
            is_error,
            timestamp,
        }) => {
            let max_lines = config.max_lines_for(&tool_name);
            let truncated_content: Vec<Content> = content
                .into_iter()
                .map(|c| match c {
                    Content::Text { text } => Content::Text {
                        text: truncate_text_head_tail(&text, max_lines),
                    },
                    other => other,
                })
                .collect();

            AgentMessage::Llm(Message::ToolResult {
                tool_call_id,
                tool_name,
                content: truncated_content,
                is_error,
                timestamp,
            })
        }
        other => other,
    }
}

/// Lines the truncation marker occupies: a blank line, the marker, a blank line.
const TRUNCATION_MARKER_LINES: usize = 3;

/// Truncate text keeping the head and tail, dropping the middle.
///
/// The marker is charged against `max_lines`, so the result is exactly
/// `max_lines` lines and truncating it again returns it unchanged. That
/// idempotence matters: compaction runs before every turn once a session is
/// over budget, and a marker that shrank on each pass (`950 lines truncated`
/// → `3 lines truncated`) both misreported the loss and rewrote settled
/// history, discarding the provider's prefix cache a second time.
fn truncate_text_head_tail(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }

    // Not enough room for head + marker + tail — keep the head and skip the
    // marker rather than emitting something that re-truncates forever.
    if max_lines <= TRUNCATION_MARKER_LINES + 1 {
        return lines[..max_lines].join("\n");
    }

    let keep = max_lines - TRUNCATION_MARKER_LINES;
    let head = keep / 2;
    let tail = keep - head;
    let omitted = lines.len() - head - tail;

    let mut result = lines[..head].join("\n");
    result.push_str(&format!("\n\n[... {} lines truncated ...]\n\n", omitted));
    result.push_str(&lines[lines.len() - tail..].join("\n"));
    result
}

/// Level 2: Summarize old assistant turns.
///
/// Keeps the last `keep_recent` messages in full detail.
/// For older messages: assistant messages with tool calls get replaced
/// with a short summary, and their tool results get dropped.
fn level2_summarize_old_turns(messages: &[AgentMessage], keep_recent: usize) -> Vec<AgentMessage> {
    let len = messages.len();
    if len <= keep_recent {
        return messages.to_vec();
    }

    // `len - keep_recent` can land in the middle of a turn, which would
    // summarize away an assistant message while its tool results stay in the
    // kept section — an orphan every provider rejects. Pull the boundary back
    // onto the turn start so the whole turn is kept intact.
    let boundary = safe_turn_start(messages, len - keep_recent);
    if boundary == 0 {
        return messages.to_vec();
    }
    let mut result = Vec::new();

    let mut i = 0;
    while i < boundary {
        let msg = &messages[i];
        match msg {
            AgentMessage::Llm(Message::Assistant {
                content, timestamp, ..
            }) => {
                // Summarize: extract text content, skip tool call details
                let text_parts: Vec<&str> = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => {
                            if text.len() > 200 {
                                None // Too long, will be replaced
                            } else {
                                Some(text.as_str())
                            }
                        }
                        _ => None,
                    })
                    .collect();

                let tool_count = content
                    .iter()
                    .filter(|c| matches!(c, Content::ToolCall { .. }))
                    .count();

                let summary = if !text_parts.is_empty() {
                    text_parts.join(" ")
                } else if tool_count > 0 {
                    format!("[Assistant used {} tool(s)]", tool_count)
                } else {
                    "[Assistant response]".into()
                };

                // Inherit the summarized turn's timestamp rather than stamping
                // wall-clock time: compaction must be a pure function of its
                // input, or the same history compacts to different bytes on
                // every pass.
                result.push(AgentMessage::Llm(Message::User {
                    content: vec![Content::Text {
                        text: format!("[Summary] {}", summary),
                    }],
                    timestamp: *timestamp,
                }));

                // Skip following tool results that belong to this turn
                i += 1;
                while i < boundary {
                    if let AgentMessage::Llm(Message::ToolResult { .. }) = &messages[i] {
                        i += 1;
                    } else {
                        break;
                    }
                }
                continue;
            }
            AgentMessage::Llm(Message::ToolResult { .. }) => {
                // Skip orphaned tool results in old section
                i += 1;
                continue;
            }
            other => {
                // Keep user messages as-is (they provide intent)
                result.push(other.clone());
            }
        }
        i += 1;
    }

    // Append recent messages in full
    result.extend_from_slice(&messages[boundary..]);
    result
}

/// Marker left in place of dropped history.
///
/// The text is constant on purpose. It sits near the front of the message list,
/// so embedding a message count here would change the bytes of the cached
/// prefix on every compaction pass — the count goes to the debug log instead.
pub(crate) const COMPACTION_MARKER: &str =
    "[Context compacted: earlier messages removed to fit the context window]";

fn compaction_marker(timestamp: u64) -> AgentMessage {
    AgentMessage::Llm(Message::User {
        content: vec![Content::Text {
            text: COMPACTION_MARKER.into(),
        }],
        timestamp,
    })
}

pub(crate) fn message_timestamp(msg: &AgentMessage) -> u64 {
    match msg {
        AgentMessage::Llm(
            Message::User { timestamp, .. }
            | Message::Assistant { timestamp, .. }
            | Message::ToolResult { timestamp, .. },
        ) => *timestamp,
        AgentMessage::Extension(_) => 0,
    }
}

/// Does this message open tool calls that a later message must answer?
fn opens_tool_calls(msg: &AgentMessage) -> bool {
    matches!(
        msg,
        AgentMessage::Llm(Message::Assistant { content, .. })
            if content.iter().any(|c| matches!(c, Content::ToolCall { .. }))
    )
}

fn is_tool_result(msg: &AgentMessage) -> bool {
    matches!(msg, AgentMessage::Llm(Message::ToolResult { .. }))
}

/// Pull a kept-head boundary back so the head never ends on an assistant
/// message whose tool results are about to be dropped — providers reject a
/// `tool_use` with no matching `tool_result`.
pub(crate) fn safe_head_end(messages: &[AgentMessage], mut end: usize) -> usize {
    while end > 0 && opens_tool_calls(&messages[end - 1]) {
        end -= 1;
    }
    end
}

/// Push a kept-tail boundary forward so the tail never opens on a tool result
/// whose originating tool call was dropped — the mirror-image rejection.
fn safe_tail_start(messages: &[AgentMessage], mut start: usize) -> usize {
    while start < messages.len() && is_tool_result(&messages[start]) {
        start += 1;
    }
    start
}

/// Pull a boundary back onto the start of the turn it lands in, so the turn's
/// assistant message and its tool results stay on the same side of the split.
pub(crate) fn safe_turn_start(messages: &[AgentMessage], mut start: usize) -> usize {
    while start > 0 && start < messages.len() && is_tool_result(&messages[start]) {
        start -= 1;
    }
    start
}

/// Level 3: Drop the smallest middle span that brings the history to `target`.
///
/// Keeps `keep_first` messages at the front and at least `keep_recent` at the
/// back, and removes only as much of the middle as the target requires — the
/// old fixed `first + recent` shape discarded everything in between even when
/// a fraction of it would have done.
fn level3_drop_middle(
    messages: &[AgentMessage],
    config: &ContextConfig,
    target: usize,
) -> Vec<AgentMessage> {
    let len = messages.len();
    let head_end = safe_head_end(messages, config.keep_first.min(len));
    // The furthest the cut may go: everything past it is the guaranteed tail.
    let max_cut = len.saturating_sub(config.keep_recent);

    if head_end >= max_cut {
        // Can't split — just keep as many recent as fit
        return keep_within_budget(messages, target);
    }

    // suffix[i] = tokens of messages[i..]
    let mut suffix = vec![0usize; len + 1];
    for i in (0..len).rev() {
        suffix[i] = suffix[i + 1] + message_tokens(&messages[i]);
    }
    let head_tokens: usize = messages[..head_end].iter().map(message_tokens).sum();
    let marker_tokens = message_tokens(&compaction_marker(0));
    let fixed = head_tokens + marker_tokens;

    // Smallest cut that reaches the target — the least history destroyed.
    let cut = suffix[head_end..=max_cut]
        .iter()
        .position(|&tokens| fixed + tokens <= target)
        .map_or(max_cut, |offset| head_end + offset);
    // Snapping forward only drops more, so the target still holds.
    let cut = safe_tail_start(messages, cut).max(head_end);

    let removed = cut - head_end;
    if removed == 0 {
        return messages.to_vec();
    }
    tracing::debug!(
        "compaction level 3: dropping {} of {} messages (target {} tokens)",
        removed,
        len,
        target
    );

    let mut result = messages[..head_end].to_vec();
    result.push(compaction_marker(message_timestamp(&messages[head_end])));
    result.extend_from_slice(&messages[cut..]);

    // If still too big, progressively drop from recent
    if total_tokens(&result) > target {
        return keep_within_budget(&result, target);
    }

    result
}

/// Keep as many recent messages as fit within budget.
fn keep_within_budget(messages: &[AgentMessage], budget: usize) -> Vec<AgentMessage> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut kept = 0usize;
    let mut remaining = budget;

    for msg in messages.iter().rev() {
        let tokens = message_tokens(msg);
        if tokens > remaining {
            break;
        }
        remaining -= tokens;
        kept += 1;
    }

    // Never open on a tool result whose tool call was just dropped.
    let start = safe_tail_start(messages, messages.len() - kept);
    let mut result = messages[start..].to_vec();

    if start > 0 {
        tracing::debug!(
            "compaction: keeping {} of {} messages within {} tokens",
            result.len(),
            messages.len(),
            budget
        );
        result.insert(0, compaction_marker(message_timestamp(&messages[0])));
    }

    result
}

// ---------------------------------------------------------------------------
// Execution limits
// ---------------------------------------------------------------------------

/// Execution limits for the agent loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLimits {
    /// Maximum number of turns (LLM calls)
    pub max_turns: usize,
    /// Maximum total tokens consumed
    pub max_total_tokens: usize,
    /// Maximum wall-clock time
    pub max_duration: std::time::Duration,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_turns: 50,
            max_total_tokens: 1_000_000,
            max_duration: std::time::Duration::from_secs(600),
        }
    }
}

/// Tracks execution state against limits
pub struct ExecutionTracker {
    pub limits: ExecutionLimits,
    pub turns: usize,
    pub tokens_used: usize,
    pub started_at: std::time::Instant,
}

impl ExecutionTracker {
    pub fn new(limits: ExecutionLimits) -> Self {
        Self {
            limits,
            turns: 0,
            tokens_used: 0,
            started_at: std::time::Instant::now(),
        }
    }

    pub fn record_turn(&mut self, tokens: usize) {
        self.turns += 1;
        self.tokens_used += tokens;
    }

    /// Check if any limit has been exceeded. Returns the reason if so.
    pub fn check_limits(&self) -> Option<String> {
        if self.turns >= self.limits.max_turns {
            return Some(format!(
                "Max turns reached ({}/{})",
                self.turns, self.limits.max_turns
            ));
        }
        if self.tokens_used >= self.limits.max_total_tokens {
            return Some(format!(
                "Max tokens reached ({}/{})",
                self.tokens_used, self.limits.max_total_tokens
            ));
        }
        let elapsed = self.started_at.elapsed();
        if elapsed >= self.limits.max_duration {
            return Some(format!(
                "Max duration reached ({:.0}s/{:.0}s)",
                elapsed.as_secs_f64(),
                self.limits.max_duration.as_secs_f64()
            ));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_tokens("hello world") > 0);
        assert!(estimate_tokens("hello world") < 10);
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_context_config_from_context_window() {
        let config = ContextConfig::from_context_window(200_000);
        assert_eq!(config.max_context_tokens, 160_000); // 80% of 200K
        assert_eq!(config.system_prompt_tokens, 4_000); // default
        assert_eq!(config.keep_recent, 10); // default

        let config = ContextConfig::from_context_window(1_000_000);
        assert_eq!(config.max_context_tokens, 800_000); // 80% of 1M

        let config = ContextConfig::from_context_window(128_000);
        assert_eq!(config.max_context_tokens, 102_400); // 80% of 128K
    }

    #[test]
    fn test_truncate_head_tail() {
        let text = (1..=100)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_text_head_tail(&text, 10);
        assert!(result.contains("line 1")); // head
        assert!(result.contains("line 100")); // tail
        assert!(result.contains("truncated"));
        assert!(!result.contains("line 50")); // middle removed

        // The marker is charged against the line budget, so the result fits
        // exactly and a second pass leaves it alone.
        assert_eq!(result.lines().count(), 10);
        assert_eq!(truncate_text_head_tail(&result, 10), result);
    }

    #[test]
    fn test_truncate_is_idempotent_and_keeps_an_honest_count() {
        let text = (1..=1000)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");

        let once = truncate_text_head_tail(&text, 50);
        assert!(once.contains("[... 953 lines truncated ...]"));
        assert_eq!(once.lines().count(), 50);

        // Re-truncating must not shrink the content further or restate the
        // omitted count — that churn rewrote settled history on every pass.
        let twice = truncate_text_head_tail(&once, 50);
        assert_eq!(twice, once);
        assert_eq!(truncate_text_head_tail(&twice, 50), once);
    }

    #[test]
    fn test_truncate_degenerate_line_budget() {
        let text = (1..=100)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        // Too small for head + marker + tail: keep the head, stay idempotent.
        for max_lines in [1usize, 2, 3, 4] {
            let result = truncate_text_head_tail(&text, max_lines);
            assert_eq!(result.lines().count(), max_lines);
            assert_eq!(truncate_text_head_tail(&result, max_lines), result);
        }
    }

    // -- helpers for the compaction-shape tests ---------------------------

    fn tool_turn(i: usize, output_lines: usize) -> Vec<AgentMessage> {
        let output = (0..output_lines)
            .map(|n| format!("turn {} line {} {}", i, n, "x".repeat(40)))
            .collect::<Vec<_>>()
            .join("\n");
        vec![
            AgentMessage::Llm(Message::Assistant {
                content: vec![Content::tool_call(
                    format!("tc-{}", i),
                    "bash",
                    serde_json::json!({ "command": format!("ls {}", i) }),
                )],
                stop_reason: StopReason::ToolUse,
                model: "test".into(),
                provider: "test".into(),
                usage: Usage::default(),
                timestamp: i as u64,
                error_message: None,
            }),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: format!("tc-{}", i),
                tool_name: "bash".into(),
                content: vec![Content::Text { text: output }],
                is_error: false,
                timestamp: i as u64,
            }),
        ]
    }

    fn tool_session(turns: usize, output_lines: usize) -> Vec<AgentMessage> {
        let mut messages = vec![AgentMessage::Llm(Message::user("start"))];
        for i in 0..turns {
            messages.extend(tool_turn(i, output_lines));
        }
        messages
    }

    #[test]
    fn test_compact_target_ratio_leaves_headroom() {
        let messages = tool_session(40, 120);
        let config = ContextConfig {
            max_context_tokens: 8_000,
            system_prompt_tokens: 0,
            compact_target_ratio: 0.5,
            ..Default::default()
        };
        assert!(total_tokens(&messages) > config.max_context_tokens);

        let result = compact_messages(messages, &config);
        // Compacting to just-fits means the next turn compacts again; the
        // point of the ratio is that it lands well clear of the budget.
        assert!(
            total_tokens(&result) <= 4_000,
            "compacted to {} tokens, expected <= 4000 (50% of budget)",
            total_tokens(&result)
        );
    }

    #[test]
    fn test_headroom_policy_derives_the_ratio_from_growth() {
        let config = ContextConfig {
            max_context_tokens: 100_000,
            system_prompt_tokens: 0,
            compact_target_ratio: 0.7,
            compact_headroom_turns: Some(30),
            ..Default::default()
        };

        // 1000 tokens/turn * 30 turns = 30K headroom -> target 70K -> 0.70,
        // which ties the ratio; nothing changes.
        assert!((config.effective_target_ratio(1000.0) - 0.7).abs() < 1e-6);

        // Faster growth needs a deeper cut than the ratio would give.
        assert!((config.effective_target_ratio(2000.0) - 0.4).abs() < 1e-6);

        // Slow growth would allow keeping more, but the ratio is a ceiling.
        assert!((config.effective_target_ratio(100.0) - 0.7).abs() < 1e-6);

        // Runaway growth is floored so compaction cannot wipe history.
        assert_eq!(
            config.effective_target_ratio(1_000_000.0),
            MIN_HEADROOM_RATIO
        );
    }

    #[test]
    fn test_headroom_policy_is_inert_without_a_growth_estimate() {
        let config = ContextConfig {
            max_context_tokens: 100_000,
            system_prompt_tokens: 0,
            compact_target_ratio: 0.6,
            compact_headroom_turns: Some(30),
            ..Default::default()
        };
        // No growth measured yet, or a degenerate policy.
        assert_eq!(config.effective_target_ratio(0.0), 0.6);
        assert_eq!(config.effective_target_ratio(-5.0), 0.6);
        assert_eq!(config.effective_target_ratio(f64::NAN), 0.6);
        assert_eq!(
            ContextConfig {
                compact_headroom_turns: Some(0),
                ..config.clone()
            }
            .effective_target_ratio(1000.0),
            0.6
        );
        assert_eq!(
            ContextConfig {
                compact_headroom_turns: None,
                ..config
            }
            .effective_target_ratio(9999.0),
            0.6
        );
    }

    #[test]
    fn test_ratio_of_one_restores_compact_to_just_fit() {
        let config = ContextConfig {
            max_context_tokens: 10_000,
            system_prompt_tokens: 0,
            compact_target_ratio: 1.0,
            ..Default::default()
        };
        assert_eq!(config.compaction_target(10_000), 10_000);
    }

    #[test]
    fn test_compaction_target_rejects_nonsense_ratios() {
        let mut config = ContextConfig::default();
        for (ratio, expected) in [
            (0.0f32, 500usize),      // clamped up to the 0.05 floor
            (-1.0, 500),             // ditto
            (2.0, 10_000),           // clamped down to 1.0
            (f32::NAN, 10_000),      // non-finite falls back to 1.0
            (f32::INFINITY, 10_000), // ditto
        ] {
            config.compact_target_ratio = ratio;
            assert_eq!(
                config.compaction_target(10_000),
                expected,
                "ratio {}",
                ratio
            );
        }
    }

    #[test]
    fn test_level3_drops_only_what_the_target_requires() {
        // Budget comfortably holds most of the session, so a correct Level 3
        // sheds a few turns rather than collapsing to first + recent.
        let messages = tool_session(40, 20);
        let config = ContextConfig {
            max_context_tokens: total_tokens(&messages) * 3 / 4,
            system_prompt_tokens: 0,
            compact_target_ratio: 0.9,
            ..Default::default()
        };

        let result = compact_messages(messages.clone(), &config);
        assert!(total_tokens(&result) <= config.max_context_tokens);
        // The old fixed shape kept keep_first + marker + keep_recent = 13.
        assert!(
            result.len() > 13,
            "kept only {} messages; Level 3 is still collapsing history it did not need to",
            result.len()
        );
    }

    #[test]
    fn test_compaction_never_orphans_tool_calls() {
        // Every provider rejects a tool_result with no matching tool_use, and
        // a tool_use with no result. Compaction must not create either.
        for (max_context_tokens, keep_recent, keep_first) in [
            (400usize, 10usize, 2usize),
            (1_000, 10, 2),
            (4_000, 10, 2),
            (20_000, 10, 2),
            // Sweep the Level 2 / Level 3 boundaries across turn parities: a
            // boundary landing mid-turn is exactly how an orphan is created.
            (4_000, 9, 1),
            (4_000, 11, 3),
            (8_000, 7, 4),
            (8_000, 12, 5),
            (12_000, 13, 0),
        ] {
            let messages = tool_session(30, 60);
            let config = ContextConfig {
                max_context_tokens,
                system_prompt_tokens: 0,
                keep_recent,
                keep_first,
                ..Default::default()
            };
            let result = compact_messages(messages, &config);

            let mut open: Vec<String> = Vec::new();
            for msg in &result {
                match msg {
                    AgentMessage::Llm(Message::Assistant { content, .. }) => {
                        open = content
                            .iter()
                            .filter_map(|c| match c {
                                Content::ToolCall { id, .. } => Some(id.clone()),
                                _ => None,
                            })
                            .collect();
                    }
                    AgentMessage::Llm(Message::ToolResult { tool_call_id, .. }) => {
                        let answered = open.iter().position(|id| id == tool_call_id);
                        assert!(
                            answered.is_some(),
                            "orphaned tool result {} at budget {}",
                            tool_call_id,
                            max_context_tokens
                        );
                        open.remove(answered.unwrap());
                    }
                    _ => {
                        assert!(
                            open.is_empty(),
                            "unanswered tool call {:?} at budget {}",
                            open,
                            max_context_tokens
                        );
                    }
                }
            }
            assert!(
                open.is_empty(),
                "history ends on an unanswered tool call at budget {}",
                max_context_tokens
            );
        }
    }

    #[test]
    fn test_compaction_marker_carries_no_drifting_count() {
        // The marker sits near the front of the request. A message count in it
        // would change the cached prefix on every compaction pass.
        let messages = tool_session(40, 120);
        let config = ContextConfig {
            max_context_tokens: 2_000,
            system_prompt_tokens: 0,
            ..Default::default()
        };
        let result = compact_messages(messages, &config);

        let marker = result
            .iter()
            .find_map(|m| match m {
                AgentMessage::Llm(Message::User { content, .. }) => match content.first() {
                    Some(Content::Text { text }) if text.starts_with("[Context compacted") => {
                        Some(text.clone())
                    }
                    _ => None,
                },
                _ => None,
            })
            .expect("expected a compaction marker");
        assert_eq!(marker, COMPACTION_MARKER);
        assert!(!marker.chars().any(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_truncate_tool_output_helper() {
        let big = (0..500)
            .map(|i| format!("l{}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let msg = AgentMessage::Llm(Message::ToolResult {
            tool_call_id: "tc-1".into(),
            tool_name: "bash".into(),
            content: vec![Content::Text { text: big }],
            is_error: false,
            timestamp: 7,
        });

        let config = ContextConfig {
            tool_output_max_lines: 50,
            ..Default::default()
        };
        let once = truncate_tool_output(msg, &config);
        let twice = truncate_tool_output(once.clone(), &config);
        assert_eq!(once, twice, "on-append truncation must be idempotent");

        match &once {
            AgentMessage::Llm(Message::ToolResult {
                content, timestamp, ..
            }) => {
                assert_eq!(*timestamp, 7, "timestamp must survive truncation");
                let Content::Text { text } = &content[0] else {
                    panic!("expected text")
                };
                assert_eq!(text.lines().count(), 50);
            }
            _ => panic!("expected a tool result"),
        }

        // Non-tool-result messages pass through untouched.
        let user = AgentMessage::Llm(Message::user("hello"));
        assert_eq!(truncate_tool_output(user.clone(), &config), user);
    }

    #[test]
    fn test_level1_truncation() {
        let big_output = (1..=200)
            .map(|i| format!("output line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let messages = vec![
            AgentMessage::Llm(Message::user("do something")),
            AgentMessage::Llm(Message::ToolResult {
                tool_call_id: "tc-1".into(),
                tool_name: "bash".into(),
                content: vec![Content::Text { text: big_output }],
                is_error: false,
                timestamp: 0,
            }),
        ];

        let compacted = level1_truncate_tool_outputs(
            &messages,
            &ContextConfig {
                tool_output_max_lines: 20,
                ..Default::default()
            },
        );
        let tool_msg = &compacted[1];
        if let AgentMessage::Llm(Message::ToolResult { content, .. }) = tool_msg {
            if let Content::Text { text } = &content[0] {
                assert!(text.contains("truncated"));
                assert!(text.contains("output line 1")); // head
                assert!(text.contains("output line 200")); // tail
                assert!(text.lines().count() < 50);
            } else {
                panic!("expected text content");
            }
        } else {
            panic!("expected tool result");
        }
    }

    #[test]
    fn test_compact_within_budget() {
        let messages = vec![
            AgentMessage::Llm(Message::user("Hello")),
            AgentMessage::Llm(Message::user("World")),
        ];
        let config = ContextConfig::default();
        let result = compact_messages(messages.clone(), &config);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_compact_drops_middle_when_needed() {
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

        let result = compact_messages(messages, &config);
        assert!(result.len() < 100);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_context_tracker_no_usage() {
        let tracker = ContextTracker::new();
        let messages = vec![
            AgentMessage::Llm(Message::user("Hello")),
            AgentMessage::Llm(Message::user("World")),
        ];
        // Without usage data, falls back to estimation
        let tokens = tracker.estimate_context_tokens(&messages);
        assert!(tokens > 0);
        assert_eq!(tokens, total_tokens(&messages));
    }

    #[test]
    fn test_context_tracker_with_usage() {
        let mut tracker = ContextTracker::new();
        let messages = vec![
            AgentMessage::Llm(Message::user("Hello")),
            AgentMessage::Llm(Message::Assistant {
                content: vec![Content::Text {
                    text: "Hi there!".into(),
                }],
                stop_reason: StopReason::Stop,
                model: "test".into(),
                provider: "test".into(),
                usage: Usage {
                    input: 100,
                    output: 50,
                    ..Default::default()
                },
                timestamp: 0,
                error_message: None,
            }),
            AgentMessage::Llm(Message::user("Follow up question here")),
        ];
        // Record usage at index 1 (assistant message)
        tracker.record_usage(
            &Usage {
                input: 100,
                output: 50,
                ..Default::default()
            },
            1,
        );
        let tokens = tracker.estimate_context_tokens(&messages);
        // Should be 150 (real usage) + estimate for the trailing user message
        let trailing_estimate = message_tokens(&messages[2]);
        assert_eq!(tokens, 150 + trailing_estimate);
    }

    #[test]
    fn test_context_tracker_reset() {
        let mut tracker = ContextTracker::new();
        tracker.record_usage(
            &Usage {
                input: 1000,
                output: 500,
                ..Default::default()
            },
            5,
        );
        tracker.reset();
        let messages = vec![AgentMessage::Llm(Message::user("test"))];
        // After reset, should fall back to estimation
        assert_eq!(
            tracker.estimate_context_tokens(&messages),
            total_tokens(&messages)
        );
    }

    #[test]
    fn test_execution_limits() {
        let limits = ExecutionLimits {
            max_turns: 3,
            max_total_tokens: 1000,
            max_duration: std::time::Duration::from_secs(60),
        };

        let mut tracker = ExecutionTracker::new(limits);
        assert!(tracker.check_limits().is_none());

        tracker.record_turn(100);
        tracker.record_turn(100);
        assert!(tracker.check_limits().is_none());

        tracker.record_turn(100);
        assert!(tracker.check_limits().is_some());
    }
}
