# Security Policy

## Supported versions

yoagent is pre-1.0. Security fixes land on the latest published minor version; there are no
long-term support branches. Please upgrade to the newest release before reporting.

## Reporting a vulnerability

**Please do not open a public issue.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/yologdev/yoagent/security/advisories/new).
If that is unavailable, email the maintainer at `yuanhao@yolog.dev` with "yoagent security" in the
subject.

Please include the affected version, a description of the impact, and a reproduction if you have
one. You can expect an initial response within a week.

## Scope

yoagent handles API credentials and executes tools, so the areas most worth scrutiny are:

- **Credential handling** — API keys are resolved from environment variables and held in memory.
  yoagent does not log them. A path that leaks a key into logs, `tracing` output, error messages,
  or a serialized session is a vulnerability.
- **Tool execution** — the `bash` tool runs shell commands and the file tools read and write disk.
  These are *designed* to execute what the model asks for; that is the crate's purpose, not a flaw.
  A bypass of a configured restriction — `ToolMiddleware` denials, `bash` deny patterns, or file
  path restrictions — is a vulnerability.
- **Deserialization** — `Session` JSONL, MCP responses, and provider SSE streams all parse
  untrusted input. Panics or memory-safety issues there are in scope.
- **MCP and OpenAPI adapters** — these connect to third-party servers and turn their descriptions
  into tools the model can call.

## Out of scope

- An agent taking a destructive action the model chose and no configured policy blocked. yoagent
  ships **no** default policy — `ToolMiddleware` is the mechanism, and installing a policy is the
  application's responsibility.
- Prompt injection causing an LLM to misbehave, absent a bypass of a yoagent-enforced restriction.
- Vulnerabilities in the upstream LLM providers themselves.
