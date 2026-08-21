//! Stateful Agent struct — wraps the agent loop with state management,
//! steering/follow-up queues, and abort support.

use crate::agent_loop::{
    agent_loop, agent_loop_continue, AfterTurnFn, AgentLoopConfig, BeforeTurnFn, OnErrorFn,
};
use crate::context::{CompactionStrategy, ContextConfig, ExecutionLimits};
use crate::mcp::{McpClient, McpError, McpToolAdapter};
use crate::provider::{ModelConfig, StreamProvider};
use crate::types::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Queue mode for steering and follow-up messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    /// Deliver one message per turn
    OneAtATime,
    /// Deliver all queued messages at once
    All,
}

/// The main Agent. Owns state, tools, and provider.
pub struct Agent {
    // State
    pub system_prompt: String,
    pub model: String,
    pub api_key: String,
    pub thinking_level: ThinkingLevel,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    model_config: Option<ModelConfig>,
    messages: Vec<AgentMessage>,
    tools: Vec<Box<dyn AgentTool>>,
    provider: Arc<dyn StreamProvider>,
    /// Whether `provider` was supplied explicitly by the caller (`new` /
    /// `from_provider`) rather than resolved from a registry (`from_config`).
    /// `set_model` uses this to avoid silently discarding a caller's provider.
    provider_is_explicit: bool,

    // Queues (shared with the loop via Arc<Mutex>)
    steering_queue: Arc<Mutex<Vec<AgentMessage>>>,
    follow_up_queue: Arc<Mutex<Vec<AgentMessage>>>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,

    // Context, limits & caching
    pub context_config: Option<ContextConfig>,
    context_management_disabled: bool,
    pub execution_limits: Option<ExecutionLimits>,
    pub cache_config: CacheConfig,
    /// Shared key-value store. Truncated tool output is stashed here, and the
    /// `shared_state` tool is registered for the run so the model can fetch it.
    ///
    /// Prefer [`with_shared_state`](Self::with_shared_state); registration
    /// happens when the run's tool list is assembled, so setting this field
    /// directly works too.
    pub shared_state: Option<crate::shared_state::SharedState>,
    pub tool_execution: ToolExecutionStrategy,
    pub retry_config: crate::retry::RetryConfig,

    // Lifecycle callbacks
    before_turn: Option<BeforeTurnFn>,
    after_turn: Option<AfterTurnFn>,
    on_error: Option<OnErrorFn>,

    // Input filters
    input_filters: Vec<Arc<dyn InputFilter>>,

    // Tool middleware (permissions/policy hooks)
    tool_middleware: Vec<Arc<dyn ToolMiddleware>>,

    // Custom compaction strategy
    compaction_strategy: Option<Arc<dyn CompactionStrategy>>,

    // Control
    cancel: Option<CancellationToken>,
    is_streaming: bool,

    // Pending completion from a spawned agent loop
    #[allow(clippy::type_complexity)]
    pending_completion: Option<JoinHandle<(Vec<Box<dyn AgentTool>>, Vec<AgentMessage>)>>,
}

/// Error building an [`Agent`] from a [`ModelConfig`] and a registry.
///
/// Only returned by [`Agent::from_config_with`] /
/// [`SubAgentTool::from_config_with`](crate::SubAgentTool::from_config_with);
/// the built-in `from_config` uses a complete registry and cannot fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AgentBuildError {
    /// The registry has no provider registered for the config's protocol.
    #[error(
        "no provider registered for protocol {0}; register one on the \
         ProviderRegistry or use Agent::from_provider with an explicit provider"
    )]
    NoProviderForProtocol(crate::provider::ApiProtocol),
}

/// Error from [`Agent::prompt_structured`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StructuredPromptError {
    /// The run produced no assistant text to parse. Only messages produced by
    /// this call are considered — earlier history is never scanned.
    #[error("model returned no output to parse")]
    NoOutput,
    /// The provider call itself failed (auth, network, rate limits, a
    /// schema-induced 400, ...). Retrying the parse is pointless; the message
    /// carries the underlying provider error.
    #[error("provider error during structured prompt: {message}")]
    Provider { message: String },
    /// The model's output did not deserialize into the requested type.
    /// `raw` carries the model's text so callers can retry or salvage.
    #[error("failed to parse structured output: {source}; raw output: {raw}")]
    Parse {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
}

impl Agent {
    /// Construct from an explicit provider, then configure with
    /// [`with_model`](Self::with_model) / [`with_api_key`](Self::with_api_key).
    ///
    /// Prefer [`from_config`](Self::from_config) (provider + env key resolved
    /// from one `ModelConfig`) or [`from_provider`](Self::from_provider) for a
    /// custom provider; both avoid the provider↔config mismatch this
    /// constructor allows.
    #[deprecated(
        since = "0.10.0",
        note = "use Agent::from_config(config) — provider + env key resolved \
                automatically — or Agent::from_provider(provider, config) for a \
                custom provider; will be removed in 1.0"
    )]
    pub fn new(provider: impl StreamProvider + 'static) -> Self {
        Self::with_provider_arc(Arc::new(provider))
    }

    /// Build an [`Agent`] from a [`ModelConfig`], selecting the built-in
    /// provider for the config's protocol and resolving the API key from the
    /// provider-conventional environment variable.
    ///
    /// This is the primary constructor: the model id, provider, context
    /// window, and pricing all come from the one `ModelConfig`, so there's no
    /// provider↔config mismatch to get wrong and no model id to pass twice.
    /// An explicit key set later via [`with_api_key`](Self::with_api_key)
    /// always overrides the environment.
    ///
    /// ```no_run
    /// use yoagent::{Agent, provider::ModelConfig};
    /// // provider auto-selected from config.api; key from ANTHROPIC_API_KEY
    /// let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"));
    /// ```
    ///
    /// # Panics
    ///
    /// Never panics — the default registry covers every [`ApiProtocol`]
    /// variant (enforced by a unit test). Use
    /// [`from_config_with`](Self::from_config_with) with a custom registry
    /// when a protocol may be unregistered and you want a `Result` instead.
    ///
    /// [`ApiProtocol`]: crate::provider::ApiProtocol
    pub fn from_config(config: ModelConfig) -> Self {
        Self::from_config_with(&crate::provider::ProviderRegistry::default(), config)
            .expect("default registry covers all built-in protocols")
    }

    /// Like [`from_config`](Self::from_config) but resolves the provider from a
    /// caller-supplied registry, returning an error if the config's protocol
    /// isn't registered.
    pub fn from_config_with(
        registry: &crate::provider::ProviderRegistry,
        config: ModelConfig,
    ) -> Result<Self, AgentBuildError> {
        let provider = registry
            .resolve(&config.api)
            .ok_or(AgentBuildError::NoProviderForProtocol(config.api))?;
        let mut agent = Self::with_provider_arc(provider).configured_for(config);
        // The provider came from the registry, not the caller — set_model may
        // safely re-resolve it on a model switch.
        agent.provider_is_explicit = false;
        Ok(agent)
    }

    /// Build an [`Agent`] from an explicit provider plus its [`ModelConfig`].
    ///
    /// The escape hatch for custom [`StreamProvider`] implementations that
    /// aren't in any registry (including test doubles — pair with
    /// [`ModelConfig::mock`](crate::provider::ModelConfig::mock)). The config
    /// is still required so the model id, context window, and pricing stay
    /// defined together; the API key is resolved from the environment unless
    /// set explicitly.
    pub fn from_provider(provider: impl StreamProvider + 'static, config: ModelConfig) -> Self {
        Self::with_provider_arc(Arc::new(provider)).configured_for(config)
    }

    /// Switch the model mid-session, re-resolving the environment API key from
    /// the new config's provider (an explicit key set via
    /// [`with_api_key`](Self::with_api_key) is preserved, since the key is
    /// resolved lazily and an explicit one always wins).
    ///
    /// Provider handling depends on how the agent was built:
    /// - Built with [`from_config`](Self::from_config): the built-in provider
    ///   for the new protocol is selected from the default registry.
    /// - Built with an **explicit** provider ([`from_provider`](Self::from_provider)
    ///   or [`new`](Self::new)): that provider is **kept** — it is never
    ///   silently replaced — and a warning is logged if it may not serve the
    ///   new protocol. Reconstruct with `from_provider` to change providers.
    pub fn set_model(&mut self, config: ModelConfig) {
        if self.provider_is_explicit {
            tracing::warn!(
                "set_model: keeping the explicitly-supplied provider; it may not \
                 serve the new protocol {}. Reconstruct with from_provider to \
                 change providers.",
                config.api
            );
        } else if let Some(provider) =
            crate::provider::ProviderRegistry::default().resolve(&config.api)
        {
            self.provider = provider;
        } else {
            tracing::warn!(
                "set_model: no built-in provider for protocol {}; keeping the \
                 previous provider, which will not match the new model.",
                config.api
            );
        }
        self.model = config.id.clone();
        self.model_config = Some(config);
    }

    /// Apply a `ModelConfig` to a freshly-constructed agent: set the model id
    /// and stash the config (provider is already wired). Key stays lazily
    /// resolved from the config's provider env var unless set explicitly.
    fn configured_for(mut self, config: ModelConfig) -> Self {
        self.model = config.id.clone();
        self.model_config = Some(config);
        self
    }

    fn with_provider_arc(provider: Arc<dyn StreamProvider>) -> Self {
        Self {
            system_prompt: String::new(),
            model: String::new(),
            api_key: String::new(),
            thinking_level: ThinkingLevel::Off,
            max_tokens: None,
            temperature: None,
            model_config: None,
            messages: Vec::new(),
            tools: Vec::new(),
            provider,
            // Explicit by default (new / from_provider); from_config_with
            // flips this to false after resolving from the registry.
            provider_is_explicit: true,
            steering_queue: Arc::new(Mutex::new(Vec::new())),
            follow_up_queue: Arc::new(Mutex::new(Vec::new())),
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            context_config: None,
            context_management_disabled: false,
            execution_limits: Some(ExecutionLimits::default()),
            cache_config: CacheConfig::default(),
            shared_state: None,
            tool_execution: ToolExecutionStrategy::default(),
            retry_config: crate::retry::RetryConfig::default(),
            before_turn: None,
            after_turn: None,
            on_error: None,
            input_filters: Vec::new(),
            tool_middleware: Vec::new(),
            compaction_strategy: None,
            cancel: None,
            is_streaming: false,
            pending_completion: None,
        }
    }

    // -- Builder-style setters --

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    #[deprecated(
        since = "0.10.0",
        note = "the model id now comes from the ModelConfig passed to \
                Agent::from_config / from_provider; will be removed in 1.0"
    )]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    pub fn with_thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn AgentTool>>) -> Self {
        self.tools = tools;
        self
    }

    #[deprecated(
        since = "0.10.0",
        note = "pass the ModelConfig to Agent::from_config(config) or \
                from_provider(provider, config) instead; will be removed in 1.0"
    )]
    pub fn with_model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set the sampling temperature. Note: the newest reasoning models
    /// (e.g. Claude Fable 5 / Opus 4.7+) reject sampling parameters — leave
    /// unset for those.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Take the tool list for a run, injecting `shared_state` when a store is
    /// configured.
    ///
    /// Injection happens here rather than in `with_shared_state` so that
    /// neither a repeat call (duplicate tool names are a provider 400) nor a
    /// later `with_tools` (which replaces the vector) can desynchronise the
    /// tool from the sink. A marker naming a tool the model cannot see is the
    /// failure this ordering prevents.
    fn take_tools(&mut self) -> Vec<Box<dyn AgentTool>> {
        let mut tools = std::mem::take(&mut self.tools);
        if let Some(state) = &self.shared_state {
            if !tools.iter().any(|t| t.name() == "shared_state") {
                tools.push(Box::new(crate::tools::SharedStateTool::new(state.clone())));
            }
        }
        tools
    }

    /// Attach a shared key-value store.
    ///
    /// Two things follow. The `shared_state` tool is registered, so the model
    /// can read and write the store — `SharedState` was always described as a
    /// parent↔sub-agent medium, but until now only `SubAgentTool` wired it, so
    /// a parent could populate it and never read it back.
    ///
    /// And truncated tool output is stashed here: head-tail truncation
    /// currently discards the middle irrecoverably, and with a store attached
    /// the marker names a key the model can fetch instead.
    pub fn with_shared_state(mut self, state: crate::shared_state::SharedState) -> Self {
        // Only the field. The tool is injected where the tool list is
        // assembled, mirroring `SubAgentTool` — pushing here would double a
        // second call (providers reject duplicate tool names with a 400), and
        // a later `with_tools` would replace the vector, silently dropping the
        // tool while leaving the sink set. That combination is the worst one:
        // markers would name a tool the model cannot see.
        self.shared_state = Some(state);
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

    /// Load skills and append their index to the system prompt.
    ///
    /// The skills index is appended as XML per the [AgentSkills standard](https://agentskills.io).
    /// The agent can then read individual SKILL.md files using the `read_file` tool
    /// when it decides a skill is relevant.
    pub fn with_skills(mut self, skills: crate::skills::SkillSet) -> Self {
        let prompt_fragment = skills.format_for_prompt();
        if !prompt_fragment.is_empty() {
            if self.system_prompt.is_empty() {
                self.system_prompt = prompt_fragment;
            } else {
                self.system_prompt = format!("{}\n\n{}", self.system_prompt, prompt_fragment);
            }
        }
        self
    }

    pub fn with_execution_limits(mut self, limits: ExecutionLimits) -> Self {
        self.execution_limits = Some(limits);
        self
    }

    pub fn with_messages(mut self, msgs: Vec<AgentMessage>) -> Self {
        self.messages = msgs;
        self
    }

    pub fn on_before_turn(
        mut self,
        f: impl Fn(&[AgentMessage], usize) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.before_turn = Some(Arc::new(f));
        self
    }

    pub fn on_after_turn(
        mut self,
        f: impl Fn(&[AgentMessage], &Usage) + Send + Sync + 'static,
    ) -> Self {
        self.after_turn = Some(Arc::new(f));
        self
    }

    pub fn on_error(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Add an input filter. Filters run in order on user messages before the LLM call.
    pub fn with_input_filter(mut self, filter: impl InputFilter + 'static) -> Self {
        self.input_filters.push(Arc::new(filter));
        self
    }

    /// Add a tool middleware — an async approve/deny/modify hook that gates
    /// every tool call (see [`ToolMiddleware`]). Middleware run in
    /// installation order; each may rewrite the arguments seen by later ones,
    /// and the first `Deny` turns the call into an error tool result carrying
    /// the reason (the LLM sees it and adapts — the loop continues).
    pub fn with_tool_middleware(mut self, middleware: impl ToolMiddleware + 'static) -> Self {
        self.tool_middleware.push(Arc::new(middleware));
        self
    }

    /// Set a custom compaction strategy. When set, replaces the default
    /// `compact_messages()` call during context compaction.
    pub fn with_compaction_strategy(mut self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.compaction_strategy = Some(Arc::new(strategy));
        self
    }

    /// Add a sub-agent tool. The sub-agent runs its own `agent_loop()` when invoked.
    pub fn with_sub_agent(mut self, sub: crate::sub_agent::SubAgentTool) -> Self {
        self.tools.push(Box::new(sub));
        self
    }

    /// Disable automatic context compaction and execution limits.
    /// This takes precedence over auto-derivation from `ModelConfig.context_window`.
    pub fn without_context_management(mut self) -> Self {
        self.context_config = None;
        self.context_management_disabled = true;
        self.execution_limits = None;
        self
    }

    // -- OpenAPI integration --

    /// Load tools from an OpenAPI spec file and add them to the agent.
    #[cfg(feature = "openapi")]
    pub async fn with_openapi_file(
        mut self,
        path: impl AsRef<std::path::Path>,
        config: crate::openapi::OpenApiConfig,
        filter: &crate::openapi::OperationFilter,
    ) -> Result<Self, crate::openapi::OpenApiError> {
        let adapters = crate::openapi::OpenApiToolAdapter::from_file(path, config, filter).await?;
        for adapter in adapters {
            self.tools.push(Box::new(adapter));
        }
        Ok(self)
    }

    /// Fetch an OpenAPI spec from a URL and add its tools to the agent.
    #[cfg(feature = "openapi")]
    pub async fn with_openapi_url(
        mut self,
        url: &str,
        config: crate::openapi::OpenApiConfig,
        filter: &crate::openapi::OperationFilter,
    ) -> Result<Self, crate::openapi::OpenApiError> {
        let adapters = crate::openapi::OpenApiToolAdapter::from_url(url, config, filter).await?;
        for adapter in adapters {
            self.tools.push(Box::new(adapter));
        }
        Ok(self)
    }

    /// Parse an OpenAPI spec string and add its tools to the agent.
    #[cfg(feature = "openapi")]
    pub fn with_openapi_spec(
        mut self,
        spec_str: &str,
        config: crate::openapi::OpenApiConfig,
        filter: &crate::openapi::OperationFilter,
    ) -> Result<Self, crate::openapi::OpenApiError> {
        let adapters = crate::openapi::OpenApiToolAdapter::from_str(spec_str, config, filter)?;
        for adapter in adapters {
            self.tools.push(Box::new(adapter));
        }
        Ok(self)
    }

    // -- MCP integration --

    /// Connect to an MCP server via stdio and add its tools to the agent.
    pub async fn with_mcp_server_stdio(
        mut self,
        command: &str,
        args: &[&str],
        env: Option<HashMap<String, String>>,
    ) -> Result<Self, McpError> {
        let client = McpClient::connect_stdio(command, args, env).await?;
        let client = Arc::new(tokio::sync::Mutex::new(client));
        let adapters = McpToolAdapter::from_client(client).await?;
        for adapter in adapters {
            self.tools.push(Box::new(adapter));
        }
        Ok(self)
    }

    /// Connect to an MCP server via HTTP and add its tools to the agent.
    pub async fn with_mcp_server_http(mut self, url: &str) -> Result<Self, McpError> {
        let client = McpClient::connect_http(url).await?;
        let client = Arc::new(tokio::sync::Mutex::new(client));
        let adapters = McpToolAdapter::from_client(client).await?;
        for adapter in adapters {
            self.tools.push(Box::new(adapter));
        }
        Ok(self)
    }

    // -- State access --

    pub fn messages(&self) -> &[AgentMessage] {
        &self.messages
    }

    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    pub fn set_tools(&mut self, tools: Vec<Box<dyn AgentTool>>) {
        self.tools = tools;
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    pub fn append_message(&mut self, msg: AgentMessage) {
        self.messages.push(msg);
    }

    pub fn replace_messages(&mut self, msgs: Vec<AgentMessage>) {
        self.messages = msgs;
    }

    pub fn save_messages(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.messages)
    }

    pub fn restore_messages(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let msgs: Vec<AgentMessage> = serde_json::from_str(json)?;
        self.messages = msgs;
        Ok(())
    }

    // -- Queue management --

    /// Queue a steering message (interrupts agent mid-tool-execution)
    pub fn steer(&self, msg: AgentMessage) {
        self.steering_queue.lock().unwrap().push(msg);
    }

    /// Queue a follow-up message (processed after agent finishes)
    pub fn follow_up(&self, msg: AgentMessage) {
        self.follow_up_queue.lock().unwrap().push(msg);
    }

    /// Queue multiple steering messages under a single lock acquisition.
    ///
    /// Use this to requeue messages from [`Agent::take_steering_queue`] — a
    /// per-message [`Agent::steer`] loop can be interleaved by the running
    /// loop's steering checks, splitting the batch across turns.
    pub fn steer_all(&self, msgs: Vec<AgentMessage>) {
        self.steering_queue.lock().unwrap().extend(msgs);
    }

    /// Queue multiple follow-up messages under a single lock acquisition.
    pub fn follow_up_all(&self, msgs: Vec<AgentMessage>) {
        self.follow_up_queue.lock().unwrap().extend(msgs);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue.lock().unwrap().clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().unwrap().clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Snapshot of the messages currently waiting in the steering queue.
    ///
    /// The snapshot is a point-in-time copy for display purposes — the agent
    /// loop may drain the queue at any moment. For atomic remove-and-edit,
    /// use [`Agent::take_steering_queue`].
    pub fn steering_queue_snapshot(&self) -> Vec<AgentMessage> {
        self.steering_queue.lock().unwrap().clone()
    }

    /// Snapshot of the messages currently waiting in the follow-up queue.
    pub fn follow_up_queue_snapshot(&self) -> Vec<AgentMessage> {
        self.follow_up_queue.lock().unwrap().clone()
    }

    /// Number of messages currently waiting in the steering queue.
    pub fn steering_queue_len(&self) -> usize {
        self.steering_queue.lock().unwrap().len()
    }

    /// Number of messages currently waiting in the follow-up queue.
    pub fn follow_up_queue_len(&self) -> usize {
        self.follow_up_queue.lock().unwrap().len()
    }

    /// Atomically drain the steering queue and return its messages.
    ///
    /// Enables edit-and-requeue UIs: take the queue, let the user edit or
    /// drop entries, then push the survivors back with [`Agent::steer_all`].
    ///
    /// Only the drain itself is atomic. For edit-and-requeue flows:
    /// - Messages the running loop has already picked up (but not yet
    ///   injected into history) are no longer in the queue and cannot be
    ///   retracted here.
    /// - If the run finishes while the queue is taken, requeued messages
    ///   are delivered at the start of the *next* run.
    /// - After [`Agent::reset`], discard taken messages instead of
    ///   requeueing them — they belong to the discarded conversation.
    pub fn take_steering_queue(&self) -> Vec<AgentMessage> {
        std::mem::take(&mut *self.steering_queue.lock().unwrap())
    }

    /// Atomically drain the follow-up queue and return its messages.
    pub fn take_follow_up_queue(&self) -> Vec<AgentMessage> {
        std::mem::take(&mut *self.follow_up_queue.lock().unwrap())
    }

    pub fn set_steering_mode(&mut self, mode: QueueMode) {
        self.steering_mode = mode;
    }

    pub fn set_follow_up_mode(&mut self, mode: QueueMode) {
        self.follow_up_mode = mode;
    }

    // -- Control --

    pub fn abort(&self) {
        if let Some(ref cancel) = self.cancel {
            cancel.cancel();
        }
    }

    pub async fn reset(&mut self) {
        // Cancel cooperatively first, then await to recover tools
        if let Some(ref cancel) = self.cancel {
            cancel.cancel();
        }
        if let Some(handle) = self.pending_completion.take() {
            // Await the cancelled task to recover tools; ignore panic
            if let Ok((tools, _messages)) = handle.await {
                self.tools = tools;
            }
        }
        self.messages.clear();
        self.clear_all_queues();
        self.is_streaming = false;
        self.cancel = None;
    }

    // -- Prompting --

    /// Send a text prompt. Returns a receiver of AgentEvents immediately,
    /// with the agent loop running concurrently so events stream in real-time.
    ///
    /// Call [`finish()`](Self::finish) after draining the receiver to restore
    /// agent state (messages, tools). `finish()` is also called automatically
    /// at the start of the next `prompt` / `continue_loop` call.
    ///
    /// # Panics
    ///
    /// Panics if the agent still counts as streaming — in practice only when
    /// a previous `*_with_sender` future was dropped mid-run, which leaves
    /// the agent stuck in the streaming state ([`Agent::finish`] cannot
    /// recover it; recreate the agent). Runs started by the
    /// receiver-returning methods are joined automatically. A
    /// misuse-`Result` variant is planned for 0.10.
    pub async fn prompt(&mut self, text: impl Into<String>) -> mpsc::UnboundedReceiver<AgentEvent> {
        let msg = AgentMessage::Llm(Message::user(text));
        self.prompt_messages(vec![msg]).await
    }

    /// Send a prompt and parse the reply into `T`, with the JSON Schema
    /// enforced natively by the provider (Anthropic: forced tool call;
    /// OpenAI-compatible: `json_schema` response format; Gemini:
    /// `responseSchema`; other providers log a warning and return free text,
    /// which still must parse into `T`).
    ///
    /// Runs the loop to completion internally (no event receiver). Derive the
    /// schema however you like — by hand or e.g. with the `schemars` crate.
    ///
    /// Note: on Anthropic the forced tool call preempts regular tools for
    /// that request — treat structured prompts as extraction/finalization
    /// calls, not agentic tool-using turns.
    pub async fn prompt_structured<T: serde::de::DeserializeOwned>(
        &mut self,
        text: impl Into<String>,
        schema: serde_json::Value,
    ) -> Result<T, StructuredPromptError> {
        // The schema is threaded through this call's loop config only — it is
        // never stored on the agent, so a dropped/timed-out future cannot
        // leave the agent stuck in schema-forcing mode.
        let schema = crate::provider::OutputSchema::new("structured_output", schema);
        let history_len = self.messages.len();

        let msg = AgentMessage::Llm(Message::user(text));
        let mut rx = self.prompt_messages_internal(vec![msg], Some(schema)).await;
        while rx.recv().await.is_some() {}
        self.finish().await;

        // Only messages produced by THIS run count — never parse stale text
        // from earlier turns. (Compaction can shrink history below
        // history_len; saturating slice keeps the scan sound.)
        let run_messages = self.messages.get(history_len.min(self.messages.len())..);
        let last_assistant = run_messages
            .into_iter()
            .flatten()
            .rev()
            .find_map(|m| match m {
                AgentMessage::Llm(Message::Assistant {
                    content,
                    stop_reason,
                    error_message,
                    ..
                }) => Some((content, stop_reason, error_message)),
                _ => None,
            });

        let Some((content, stop_reason, error_message)) = last_assistant else {
            return Err(StructuredPromptError::NoOutput);
        };

        // A failed provider call is not a parse problem — surface it as such.
        if *stop_reason == StopReason::Error {
            return Err(StructuredPromptError::Provider {
                message: error_message
                    .clone()
                    .unwrap_or_else(|| "provider error (no detail)".into()),
            });
        }

        // The structured payload is the LAST text block: tool-forcing unwrap
        // appends it after any preamble text the model produced.
        let raw = content
            .iter()
            .rev()
            .find_map(|c| match c {
                Content::Text { text } if !text.is_empty() => Some(text.clone()),
                _ => None,
            })
            .ok_or(StructuredPromptError::NoOutput)?;

        // Defensive: some models wrap JSON in markdown fences.
        let cleaned = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        serde_json::from_str(cleaned).map_err(|source| StructuredPromptError::Parse {
            source,
            raw: raw.clone(),
        })
    }

    /// Send messages as a prompt. Returns a receiver immediately with the
    /// agent loop running concurrently for true streaming.
    ///
    /// Call [`finish()`](Self::finish) after draining events to restore state.
    ///
    /// # Panics
    ///
    /// Panics if the agent still counts as streaming — in practice only when
    /// a previous `*_with_sender` future was dropped mid-run, which leaves
    /// the agent stuck in the streaming state ([`Agent::finish`] cannot
    /// recover it; recreate the agent). Runs started by the
    /// receiver-returning methods are joined automatically. A
    /// misuse-`Result` variant is planned for 0.10.
    pub async fn prompt_messages(
        &mut self,
        messages: Vec<AgentMessage>,
    ) -> mpsc::UnboundedReceiver<AgentEvent> {
        self.prompt_messages_internal(messages, None).await
    }

    /// Shared plumbing for `prompt_messages` and `prompt_structured`. The
    /// structured-output schema is per-call state: it lives on this run's
    /// `AgentLoopConfig` only, never on the agent.
    async fn prompt_messages_internal(
        &mut self,
        messages: Vec<AgentMessage>,
        output_schema: Option<crate::provider::OutputSchema>,
    ) -> mpsc::UnboundedReceiver<AgentEvent> {
        self.finish().await; // restore from previous if needed

        assert!(
            !self.is_streaming,
            "Agent is already streaming. Use steer() or follow_up()."
        );

        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());
        self.is_streaming = true;

        let (tx, rx) = mpsc::unbounded_channel();

        let mut context = AgentContext {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.take_tools(),
        };

        let mut config = self.build_config();
        config.output_schema = output_schema;

        let handle = tokio::spawn(async move {
            let _new_messages = agent_loop(messages, &mut context, &config, tx, cancel).await;
            (context.tools, context.messages)
        });

        self.pending_completion = Some(handle);
        rx
    }

    /// Send a text prompt, streaming events to a caller-provided sender.
    ///
    /// The caller provides an external sender and sets up a consumer task
    /// before calling this method. This method blocks until the loop finishes
    /// and state is restored — unlike [`prompt()`](Self::prompt) which spawns
    /// the loop concurrently and returns immediately.
    ///
    /// ```rust,no_run
    /// # use yoagent::Agent;
    /// # use yoagent::provider::{MockProvider, ModelConfig};
    /// # async fn example() {
    /// let mut agent = Agent::from_provider(MockProvider::text("hi"), ModelConfig::mock());
    /// let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    /// tokio::spawn(async move {
    ///     while let Some(event) = rx.recv().await { /* real-time */ }
    /// });
    /// agent.prompt_with_sender("hello", tx).await;
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the agent still counts as streaming — in practice only when
    /// a previous `*_with_sender` future was dropped mid-run, which leaves
    /// the agent stuck in the streaming state ([`Agent::finish`] cannot
    /// recover it; recreate the agent). Runs started by the
    /// receiver-returning methods are joined automatically. A
    /// misuse-`Result` variant is planned for 0.10.
    pub async fn prompt_with_sender(
        &mut self,
        text: impl Into<String>,
        tx: mpsc::UnboundedSender<AgentEvent>,
    ) {
        let msg = AgentMessage::Llm(Message::user(text));
        self.prompt_messages_with_sender(vec![msg], tx).await;
    }

    /// Send messages as a prompt, streaming events to a caller-provided sender.
    /// Blocks until the loop finishes and state is restored.
    ///
    /// # Panics
    ///
    /// Panics if the agent still counts as streaming — in practice only when
    /// a previous `*_with_sender` future was dropped mid-run, which leaves
    /// the agent stuck in the streaming state ([`Agent::finish`] cannot
    /// recover it; recreate the agent). Runs started by the
    /// receiver-returning methods are joined automatically. A
    /// misuse-`Result` variant is planned for 0.10.
    pub async fn prompt_messages_with_sender(
        &mut self,
        messages: Vec<AgentMessage>,
        tx: mpsc::UnboundedSender<AgentEvent>,
    ) {
        self.finish().await; // restore from previous if needed

        assert!(
            !self.is_streaming,
            "Agent is already streaming. Use steer() or follow_up()."
        );

        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());
        self.is_streaming = true;

        // Move tools temporarily into context for the loop; restored after
        let mut context = AgentContext {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.take_tools(),
        };

        let config = self.build_config();

        let _new_messages = agent_loop(messages, &mut context, &config, tx, cancel).await;

        self.tools = context.tools;
        self.messages = context.messages;
        self.is_streaming = false;
        self.cancel = None;
    }

    /// Continue from current context (for retries after errors). Returns a
    /// receiver immediately with the loop running concurrently.
    ///
    /// Call [`finish()`](Self::finish) after draining events to restore state.
    ///
    /// # Panics
    ///
    /// Panics if there are no messages to continue from, or if a previous
    /// `*_with_sender` future was dropped mid-run (the agent is stuck in the
    /// streaming state; [`Agent::finish`] cannot recover it — recreate the
    /// agent). A misuse-`Result` variant is planned for 0.10.
    pub async fn continue_loop(&mut self) -> mpsc::UnboundedReceiver<AgentEvent> {
        self.finish().await; // restore from previous if needed

        assert!(!self.is_streaming, "Agent is already streaming.");
        assert!(!self.messages.is_empty(), "No messages to continue from.");

        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());
        self.is_streaming = true;

        let (tx, rx) = mpsc::unbounded_channel();

        let mut context = AgentContext {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.take_tools(),
        };

        let config = self.build_config();

        let handle = tokio::spawn(async move {
            let _new_messages = agent_loop_continue(&mut context, &config, tx, cancel).await;
            (context.tools, context.messages)
        });

        self.pending_completion = Some(handle);
        rx
    }

    /// Continue from current context, streaming events to a caller-provided sender.
    /// Blocks until the loop finishes and state is restored.
    ///
    /// # Panics
    ///
    /// Panics if there are no messages to continue from, or if a previous
    /// `*_with_sender` future was dropped mid-run (the agent is stuck in the
    /// streaming state; [`Agent::finish`] cannot recover it — recreate the
    /// agent). A misuse-`Result` variant is planned for 0.10.
    pub async fn continue_loop_with_sender(&mut self, tx: mpsc::UnboundedSender<AgentEvent>) {
        self.finish().await; // restore from previous if needed

        assert!(!self.is_streaming, "Agent is already streaming.");
        assert!(!self.messages.is_empty(), "No messages to continue from.");

        let cancel = CancellationToken::new();
        self.cancel = Some(cancel.clone());
        self.is_streaming = true;

        // Move tools temporarily into context for the loop; restored after
        let mut context = AgentContext {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.take_tools(),
        };

        let config = self.build_config();

        let _new_messages = agent_loop_continue(&mut context, &config, tx, cancel).await;

        self.tools = context.tools;
        self.messages = context.messages;
        self.is_streaming = false;
        self.cancel = None;
    }

    /// Wait for the running agent loop to finish and restore state
    /// (messages, tools, streaming flag).
    ///
    /// Called automatically at the start of all prompting methods
    /// ([`prompt()`](Self::prompt), [`prompt_messages()`](Self::prompt_messages),
    /// [`prompt_messages_with_sender()`](Self::prompt_messages_with_sender),
    /// [`continue_loop()`](Self::continue_loop),
    /// [`continue_loop_with_sender()`](Self::continue_loop_with_sender)).
    /// Call explicitly when you need to access [`messages()`](Self::messages)
    /// right after draining events.
    pub async fn finish(&mut self) {
        if let Some(handle) = self.pending_completion.take() {
            match handle.await {
                Ok((tools, messages)) => {
                    self.tools = tools;
                    self.messages = messages;
                }
                Err(e) => {
                    // Task panicked or was cancelled — log and leave state as-is
                    tracing::error!("Agent loop task failed: {}", e);
                }
            }
            self.is_streaming = false;
            self.cancel = None;
        }
    }

    // -- Internal --

    /// The explicit key if set, else the provider-conventional env var
    /// (see [`crate::provider::resolve_api_key`]).
    fn resolved_api_key(&self) -> String {
        if !self.api_key.is_empty() {
            return self.api_key.clone();
        }
        let provider = self
            .model_config
            .as_ref()
            .map(|m| m.provider.as_str())
            .unwrap_or("anthropic");
        crate::provider::resolve_api_key_or_warn(provider)
    }

    /// Total dollar cost of the assistant turns currently in history, using
    /// the model's [`CostConfig`](crate::provider::CostConfig) rates.
    ///
    /// Returns `None` when no `ModelConfig` is set or when the config's
    /// rates are all zero (pricing unknown — e.g. custom or local models),
    /// so `None` means "can't price this", never "free". Rates come from the
    /// *current* model config; sessions that switched models mid-way are
    /// priced entirely at the current rates.
    pub fn session_cost_usd(&self) -> Option<f64> {
        let cost = &self.model_config.as_ref()?.cost;
        if !cost.is_configured() {
            return None;
        }
        Some(
            self.messages
                .iter()
                .filter_map(|m| match m {
                    AgentMessage::Llm(Message::Assistant { usage, .. }) => {
                        Some(cost.cost_usd(usage))
                    }
                    _ => None,
                })
                .sum(),
        )
    }

    fn build_config(&self) -> AgentLoopConfig {
        if self.thinking_level != ThinkingLevel::Off {
            if let Some(mc) = &self.model_config {
                if !mc.reasoning {
                    tracing::warn!(
                        "thinking_level is set but model '{}' is not marked \
                         reasoning-capable (ModelConfig.reasoning = false)",
                        mc.id
                    );
                }
            }
        }
        let steering_queue = self.steering_queue.clone();
        let steering_mode = self.steering_mode;

        let follow_up_queue = self.follow_up_queue.clone();
        let follow_up_mode = self.follow_up_mode;

        AgentLoopConfig {
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: self.resolved_api_key(),
            thinking_level: self.thinking_level,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            model_config: self.model_config.clone(),
            convert_to_llm: None,
            transform_context: None,
            get_steering_messages: Some(Box::new(move || {
                let mut queue = steering_queue.lock().unwrap();
                match steering_mode {
                    QueueMode::OneAtATime => {
                        if queue.is_empty() {
                            vec![]
                        } else {
                            vec![queue.remove(0)]
                        }
                    }
                    QueueMode::All => queue.drain(..).collect(),
                }
            })),
            context_config: if self.context_management_disabled {
                None
            } else {
                Some(self.context_config.clone().unwrap_or_else(|| {
                    self.model_config
                        .as_ref()
                        .map(|m| ContextConfig::from_context_window(m.context_window))
                        .unwrap_or_default()
                }))
            },
            compaction_strategy: self.compaction_strategy.clone(),
            execution_limits: self.execution_limits.clone(),
            cache_config: self.cache_config.clone(),
            tool_output_sink: self.shared_state.clone(),
            tool_execution: self.tool_execution.clone(),
            retry_config: self.retry_config.clone(),
            get_follow_up_messages: Some(Box::new(move || {
                let mut queue = follow_up_queue.lock().unwrap();
                match follow_up_mode {
                    QueueMode::OneAtATime => {
                        if queue.is_empty() {
                            vec![]
                        } else {
                            vec![queue.remove(0)]
                        }
                    }
                    QueueMode::All => queue.drain(..).collect(),
                }
            })),
            before_turn: self.before_turn.clone(),
            after_turn: self.after_turn.clone(),
            on_error: self.on_error.clone(),
            input_filters: self.input_filters.clone(),
            tool_middleware: self.tool_middleware.clone(),
            output_schema: None,
            turn_delay: None,
        }
    }
}

/// Cancel and abort any in-flight agent loop when the `Agent` goes away.
///
/// [`JoinHandle`] does not cancel its task on drop, so without this a dropped
/// streaming `Agent` leaves the spawned loop running as an orphan — burning
/// tokens on work nobody will read, holding the tools, and keeping the event
/// channel alive so the caller's receiver never closes.
///
/// Applies only to the receiver-returning methods ([`prompt`](Agent::prompt),
/// [`continue_loop`](Agent::continue_loop)), which spawn. The `*_with_sender`
/// variants run inline and never set `pending_completion`, so this is a no-op
/// for them.
///
/// **A dropped `Agent` kills its run.** Keep the `Agent` alive until you have
/// drained the receiver; dropping it early closes the channel *without* an
/// [`AgentEvent::AgentEnd`], which a consumer looping on `recv()` cannot
/// distinguish from a clean finish.
///
/// Three limitations, inherent to `Drop` rather than oversights:
///
/// - **Tools are dropped, not recovered.** `Drop` cannot await the task to take
///   them back. [`reset`](Agent::reset) cancels then awaits (but also clears
///   messages and every queue); [`finish`](Agent::finish) awaits natural
///   completion without cancelling, so it blocks until the run ends.
/// - **In-flight tool side effects may be half-applied** — a partly written
///   file, an open transaction. Nothing is rolled back.
/// - **A tool blocked in synchronous code keeps running.** Abort only takes
///   effect at an await point. `tokio::task::spawn_blocking` makes the *loop*
///   promptly abortable but does not stop the blocking closure itself; for
///   genuinely interruptible work, have the tool poll the cancellation token it
///   receives on [`ToolContext`].
impl Drop for Agent {
    fn drop(&mut self) {
        // Cancel first so the child tokens handed to tools via ToolContext, and
        // any sub-agent tasks spawned beneath them, observe the shutdown — those
        // outlive the abort below. The loop task itself is stopped by abort(),
        // which drops it at its next poll rather than polling it again.
        if let Some(ref cancel) = self.cancel {
            cancel.cancel();
        }
        if let Some(handle) = self.pending_completion.take() {
            if self.is_streaming && !handle.is_finished() {
                tracing::warn!(
                    "Agent dropped mid-run; aborting the loop. The event receiver will \
                     close without an AgentEnd event and in-flight tool side effects may \
                     be partially applied. Call finish() or reset() to end a run cleanly."
                );
            }
            handle.abort();
        }
    }
}
