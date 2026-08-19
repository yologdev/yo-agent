//! Sub-agent tool — delegates tasks to a child agent loop.
//!
//! The `SubAgentTool` implements `AgentTool` and internally runs `agent_loop()`
//! with its own system prompt, tools, and provider. The parent LLM invokes it
//! like any other tool, passing a natural-language `task` string.
//!
//! # Design
//!
//! - **Context isolation**: each invocation starts a fresh conversation
//! - **Nesting supported**: sub-agents can contain other SubAgentTools for recursive delegation (use `with_max_turns()` to bound depth)
//! - **Cancellation propagation**: the parent's cancel token is forwarded
//! - **Event forwarding**: sub-agent events stream to the parent via `on_update`
//!
//! # Example
//!
//! ```rust,no_run
//! use yoagent::sub_agent::SubAgentTool;
//! use yoagent::provider::ModelConfig;
//!
//! // Provider selected from the config's protocol; key from ANTHROPIC_API_KEY.
//! let researcher = SubAgentTool::from_config(
//!     "researcher",
//!     ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"),
//! )
//! .with_description("Searches codebases and documents")
//! .with_system_prompt("You are a research assistant.");
//! ```

use crate::agent_loop::{agent_loop, AgentLoopConfig};
use crate::context::ExecutionLimits;
use crate::provider::model::ModelConfig;
use crate::provider::StreamProvider;
use crate::shared_state::SharedState;
use crate::tools::shared_state_tool::SharedStateTool;
use crate::types::*;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Default max turns for sub-agents (prevents runaway execution).
const DEFAULT_MAX_TURNS: usize = 10;

/// A tool that delegates work to a child agent loop.
///
/// When the parent LLM calls this tool, it spawns a fresh `agent_loop()` with
/// its own system prompt, tools, and provider. The sub-agent runs to completion
/// and its final text output is returned as the tool result.
pub struct SubAgentTool {
    tool_name: String,
    tool_description: String,
    system_prompt: String,
    skills_prompt: String,
    model: String,
    api_key: String,
    provider: Arc<dyn StreamProvider>,
    tools: Vec<Arc<dyn AgentTool>>,
    thinking_level: ThinkingLevel,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    cache_config: CacheConfig,
    tool_execution: ToolExecutionStrategy,
    retry_config: crate::retry::RetryConfig,
    max_turns: usize,
    shared_state: Option<SharedState>,
    turn_delay: Option<std::time::Duration>,
    model_config: Option<ModelConfig>,
    tool_middleware: Vec<Arc<dyn ToolMiddleware>>,
}

impl SubAgentTool {
    /// Create a new sub-agent tool with a name and provider.
    #[deprecated(
        since = "0.10.0",
        note = "use SubAgentTool::from_config(name, config) — provider + env key \
                resolved automatically — or SubAgentTool::from_provider(name, provider, config) \
                for a custom provider; will be removed in 1.0"
    )]
    pub fn new(name: impl Into<String>, provider: Arc<dyn StreamProvider>) -> Self {
        Self::build(name, provider)
    }

    /// Internal constructor shared by `new` and the `from_*` builders (not
    /// deprecated, so the builders don't trip the deprecation lint).
    fn build(name: impl Into<String>, provider: Arc<dyn StreamProvider>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Delegate a task to the '{}' sub-agent", name),
            tool_name: name,
            system_prompt: String::new(),
            skills_prompt: String::new(),
            model: String::new(),
            api_key: String::new(),
            provider,
            tools: Vec::new(),
            thinking_level: ThinkingLevel::Off,
            max_tokens: None,
            temperature: None,
            cache_config: CacheConfig::default(),
            tool_execution: ToolExecutionStrategy::default(),
            retry_config: crate::retry::RetryConfig::default(),
            max_turns: DEFAULT_MAX_TURNS,
            shared_state: None,
            turn_delay: None,
            model_config: None,
            tool_middleware: Vec::new(),
        }
    }

    /// Create a sub-agent from a name and [`ModelConfig`], selecting the
    /// built-in provider for the config's protocol.
    ///
    /// Mirrors [`Agent::from_config`](crate::Agent::from_config): the model
    /// id, provider, and pricing come from one config, and the API key is
    /// resolved from the provider-conventional env var unless set explicitly
    /// with [`with_api_key`](Self::with_api_key).
    ///
    /// # Panics
    ///
    /// Never panics — the default registry covers every [`ApiProtocol`]
    /// variant. Use [`from_config_with`](Self::from_config_with) with a custom
    /// registry when a protocol may be unregistered and you want a `Result`.
    ///
    /// [`ApiProtocol`]: crate::provider::ApiProtocol
    pub fn from_config(name: impl Into<String>, config: ModelConfig) -> Self {
        Self::from_config_with(&crate::provider::ProviderRegistry::default(), name, config)
            .expect("default registry covers all built-in protocols")
    }

    /// Like [`from_config`](Self::from_config) but resolves the provider from a
    /// caller-supplied registry, returning an error if the config's protocol
    /// isn't registered. Mirrors
    /// [`Agent::from_config_with`](crate::Agent::from_config_with).
    pub fn from_config_with(
        registry: &crate::provider::ProviderRegistry,
        name: impl Into<String>,
        config: ModelConfig,
    ) -> Result<Self, crate::AgentBuildError> {
        let provider = registry
            .resolve(&config.api)
            .ok_or(crate::AgentBuildError::NoProviderForProtocol(config.api))?;
        Ok(Self::build(name, provider).configured_for(config))
    }

    /// Create a sub-agent from a name, explicit provider, and [`ModelConfig`].
    ///
    /// The escape hatch for custom providers and test doubles (pair with
    /// [`ModelConfig::mock`](crate::provider::ModelConfig::mock)). Mirrors
    /// [`Agent::from_provider`](crate::Agent::from_provider).
    pub fn from_provider(
        name: impl Into<String>,
        provider: Arc<dyn StreamProvider>,
        config: ModelConfig,
    ) -> Self {
        Self::build(name, provider).configured_for(config)
    }

    /// Set the model id and stash the config on a freshly-constructed
    /// sub-agent (provider already wired).
    fn configured_for(mut self, config: ModelConfig) -> Self {
        self.model = config.id.clone();
        self.model_config = Some(config);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.tool_description = desc.into();
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Attach a skill set so the sub-agent sees the skills index.
    ///
    /// Mirrors [`Agent::with_skills`](crate::agent::Agent::with_skills): the skills
    /// index is formatted as XML per the [AgentSkills standard](https://agentskills.io)
    /// and appended to the sub-agent's system prompt at dispatch time. The sub-agent
    /// can then read individual SKILL.md files via the `read_file` tool when it
    /// decides a skill is relevant (make sure the sub-agent has such a tool).
    pub fn with_skills(mut self, skills: crate::skills::SkillSet) -> Self {
        self.skills_prompt = skills.format_for_prompt();
        self
    }

    #[deprecated(
        since = "0.10.0",
        note = "the model id now comes from the ModelConfig passed to \
                SubAgentTool::from_config / from_provider; will be removed in 1.0"
    )]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn AgentTool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a tool middleware for the sub-agent's own tool calls. Mirrors
    /// [`Agent::with_tool_middleware`](crate::Agent::with_tool_middleware).
    pub fn with_tool_middleware(mut self, middleware: impl ToolMiddleware + 'static) -> Self {
        self.tool_middleware.push(Arc::new(middleware));
        self
    }

    pub fn with_thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }

    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set the sampling temperature for the sub-agent. Note: the newest
    /// reasoning models (e.g. Claude Fable 5 / Opus 4.7+) reject sampling
    /// parameters — leave unset for those.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_cache_config(mut self, config: CacheConfig) -> Self {
        self.cache_config = config;
        self
    }

    pub fn with_tool_execution(mut self, strategy: ToolExecutionStrategy) -> Self {
        self.tool_execution = strategy;
        self
    }

    pub fn with_retry_config(mut self, config: crate::retry::RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    pub fn with_max_turns(mut self, max: usize) -> Self {
        self.max_turns = max;
        self
    }

    /// Attach a shared key-value store. Sub-agents get a `shared_state` tool
    /// to read/write variables. The parent can also read/write programmatically
    /// via the `SharedState` handle.
    pub fn with_shared_state(mut self, state: SharedState) -> Self {
        self.shared_state = Some(state);
        self
    }

    /// Attach a shared store restricted to this sub-agent's own namespace.
    ///
    /// Equivalent to [`with_shared_state(state.scoped(name))`](Self::with_shared_state),
    /// spelled out because the isolating variant is easy to forget. The
    /// sub-agent cannot read, overwrite, or even enumerate keys outside its
    /// scope, while the parent's unscoped handle still sees everything it
    /// writes.
    ///
    /// Use when sub-agents should not see each other's data; keep
    /// `with_shared_state` when sharing is the point.
    pub fn with_scoped_shared_state(mut self, state: SharedState, scope: impl AsRef<str>) -> Self {
        self.shared_state = Some(state.scoped(scope));
        self
    }

    /// Add an inter-turn delay to throttle API requests.
    /// Useful when using OAuth tokens or providers with low rate limits.
    /// The delay is applied before each turn except the first.
    pub fn with_turn_delay(mut self, delay: std::time::Duration) -> Self {
        self.turn_delay = Some(delay);
        self
    }

    /// Set the model configuration for multi-provider support.
    /// Required for non-Anthropic providers (OpenAI-compat, Google, etc.)
    /// to specify base URL, compat flags, and other provider-specific settings.
    #[deprecated(
        since = "0.10.0",
        note = "pass the ModelConfig to SubAgentTool::from_config(name, config) or \
                from_provider(name, provider, config) instead; will be removed in 1.0"
    )]
    pub fn with_model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }
}

/// Thin adapter: wraps `Arc<dyn AgentTool>` so it can be placed in a
/// `Vec<Box<dyn AgentTool>>` (required by `AgentContext`).
struct ArcToolWrapper(Arc<dyn AgentTool>);

#[async_trait::async_trait]
impl AgentTool for ArcToolWrapper {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn label(&self) -> &str {
        self.0.label()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn parameters_schema(&self) -> serde_json::Value {
        self.0.parameters_schema()
    }
    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        self.0.execute(params, ctx).await
    }
}

#[async_trait::async_trait]
impl AgentTool for SubAgentTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn label(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to this sub-agent"
                }
            },
            "required": ["task"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let cancel = ctx.cancel;
        let on_update = ctx.on_update;
        let on_progress = ctx.on_progress;
        // Extract the task parameter
        let task = params
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("Missing required 'task' parameter".into()))?
            .to_string();

        // Build tool list from Arc wrappers
        let mut tools: Vec<Box<dyn AgentTool>> = self
            .tools
            .iter()
            .map(|t| Box::new(ArcToolWrapper(Arc::clone(t))) as Box<dyn AgentTool>)
            .collect();

        // Append the skills index (if any) so the sub-agent can discover skills.
        let mut system_prompt = self.system_prompt.clone();
        if !self.skills_prompt.is_empty() {
            if system_prompt.is_empty() {
                system_prompt = self.skills_prompt.clone();
            } else {
                system_prompt = format!("{}\n\n{}", system_prompt, self.skills_prompt);
            }
        }

        // Inject SharedStateTool when shared state is configured
        if let Some(ref state) = self.shared_state {
            tools.push(Box::new(SharedStateTool::new(state.clone())));
            let summary = state.summary().await;
            system_prompt.push_str(&format!(
                "\n\n## Shared State\nYou have access to a shared variable store via the `shared_state` tool.\nAvailable: {}",
                summary
            ));
        }

        // Fresh context for the sub-agent
        let mut context = AgentContext {
            system_prompt,
            messages: Vec::new(),
            tools,
        };

        // Config with Arc'd provider
        let config = AgentLoopConfig {
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: if self.api_key.is_empty() {
                crate::provider::resolve_api_key_or_warn(
                    self.model_config
                        .as_ref()
                        .map(|m| m.provider.as_str())
                        .unwrap_or("anthropic"),
                )
            } else {
                self.api_key.clone()
            },
            thinking_level: self.thinking_level,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            model_config: self.model_config.clone(),
            convert_to_llm: None,
            transform_context: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            context_config: None,
            compaction_strategy: None,
            execution_limits: Some(ExecutionLimits {
                max_turns: self.max_turns,
                // Generous token/duration limits — turn limit is the primary guard
                max_total_tokens: 1_000_000,
                max_duration: std::time::Duration::from_secs(300),
            }),
            cache_config: self.cache_config.clone(),
            tool_output_sink: self.shared_state.clone(),
            tool_execution: self.tool_execution.clone(),
            retry_config: self.retry_config.clone(),
            before_turn: None,
            after_turn: None,
            on_error: None,
            input_filters: vec![],
            tool_middleware: self.tool_middleware.clone(),
            output_schema: None,
            turn_delay: self.turn_delay,
        };

        // Channel for sub-agent events
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Forward sub-agent events to parent via on_update and on_progress callbacks
        let forward_handle = if on_update.is_some() || on_progress.is_some() {
            let tool_name = self.tool_name.clone();
            Some(tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    // Forward progress messages via on_progress
                    if let AgentEvent::ProgressMessage { text, .. } = &event {
                        if let Some(ref cb) = on_progress {
                            cb(text.clone());
                        }
                    }

                    // Convert interesting events to ToolResult updates for the parent
                    if let Some(ref on_update) = on_update {
                        let update_text = match &event {
                            AgentEvent::MessageUpdate {
                                delta: StreamDelta::Text { delta },
                                ..
                            } => Some(delta.clone()),
                            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                                Some(format!("[sub-agent calling tool: {}]", tool_name))
                            }
                            _ => None,
                        };

                        if let Some(text) = update_text {
                            on_update(ToolResult {
                                content: vec![Content::Text { text }],
                                details: serde_json::json!({ "sub_agent": tool_name }),
                            });
                        }
                    }
                }
            }))
        } else {
            None
        };

        // Run the sub-agent loop
        let prompt = AgentMessage::Llm(Message::user(task));
        let new_messages = agent_loop(vec![prompt], &mut context, &config, tx, cancel).await;

        // Wait for event forwarding to complete
        if let Some(handle) = forward_handle {
            let _ = handle.await;
        }

        // Check if the last message was an error
        if let Some(error_msg) = extract_error(&new_messages) {
            return Err(ToolError::Failed(format!(
                "Sub-agent '{}' failed: {}",
                self.tool_name, error_msg
            )));
        }

        // Extract final assistant text from the returned messages
        let result_text = extract_final_text(&new_messages);

        // Include full sub-agent conversation in details for debugging
        let details = serde_json::json!({
            "sub_agent": self.tool_name,
            "turns": new_messages.len(),
        });

        Ok(ToolResult {
            content: vec![Content::Text { text: result_text }],
            details,
        })
    }
}

/// Check if the last assistant message was an error, return the error message.
fn extract_error(messages: &[AgentMessage]) -> Option<String> {
    for msg in messages.iter().rev() {
        if let AgentMessage::Llm(Message::Assistant {
            stop_reason,
            error_message,
            ..
        }) = msg
        {
            if *stop_reason == StopReason::Error {
                return Some(
                    error_message
                        .clone()
                        .unwrap_or_else(|| "Unknown error".into()),
                );
            }
        }
    }
    None
}

/// Extract the final assistant text from agent messages.
/// Collects text from the last assistant message, or returns a fallback.
fn extract_final_text(messages: &[AgentMessage]) -> String {
    for msg in messages.iter().rev() {
        if let AgentMessage::Llm(Message::Assistant { content, .. }) = msg {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } if !text.is_empty() => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if !texts.is_empty() {
                return texts.join("\n");
            }
        }
    }
    "(sub-agent produced no text output)".to_string()
}
