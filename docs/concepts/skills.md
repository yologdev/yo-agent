# Skills

Skills extend an agent with domain expertise using the [AgentSkills](https://agentskills.io) open standard. A skill is a directory containing a `SKILL.md` file with instructions the agent can load on demand.

## How it works

Skills use **progressive disclosure** to manage context efficiently:

1. **Metadata** (~100 tokens/skill) — name + description, always in the system prompt
2. **Instructions** (<5k tokens) — SKILL.md body, loaded when the agent decides the skill is relevant
3. **Resources** (unlimited) — scripts, references, assets — loaded only when needed

The agent decides when to activate a skill based on the description alone. No trigger engine needed.

## Skill format

```
my-skill/
├── SKILL.md          # Required: YAML frontmatter + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation loaded on demand
└── assets/           # Optional: templates, static resources
```

SKILL.md uses YAML frontmatter:

```markdown
---
name: git
description: Git operations — commit, branch, merge, rebase. Use when the user mentions version control.
---

# Git Skill

## Workflow
1. Run `git status` first
2. Stage changes, write conventional commit messages
3. For merges, check for conflicts first

## Scripts
For complex diffs: `bash {baseDir}/scripts/diff_summary.sh`
```

## Loading skills

```rust
use yoagent::SkillSet;

// Load from multiple directories (later dirs override earlier on name conflict)
let skills = SkillSet::load(&["./skills", "~/.yoagent/skills"])?;

// Or load from a single directory with a label
let workspace_skills = SkillSet::load_dir("./skills", "workspace")?;
```

If skills may be updated independently at runtime, use resilient loading so one
malformed `SKILL.md` does not discard the rest of the set:

```rust
let (skills, errors) = SkillSet::load_resilient(&["./skills", "~/.yoagent/skills"]);

for error in errors {
    tracing::warn!(%error, "skipping malformed skill");
}
```

`load_dir_resilient` provides the same behavior for a single directory with a
custom source label. The original `load` and `load_dir` methods remain strict
and return the first error they encounter.

## Using with Agent

```rust
use yoagent::{Agent, SkillSet};

let skills = SkillSet::load(&["./skills"])?;

let agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Claude Sonnet 5"))
    .with_system_prompt("You are a coding assistant.")
    .with_skills(skills)  // Appends skill index to system prompt
    .with_tools(tools);
```

The agent's system prompt will include:

```xml
<available_skills>
  <skill>
    <name>git</name>
    <description>Git operations — commit, branch, merge, rebase.</description>
    <location>/path/to/skills/git/SKILL.md</location>
  </skill>
</available_skills>
```

When the agent encounters a task matching a skill, it reads the SKILL.md using the `read_file` tool and follows the instructions. No special infrastructure needed.

## Precedence

When loading from multiple directories, later directories take precedence. A skill in `./skills/` overrides the same-named skill in `~/.yoagent/skills/`.

You can also merge skill sets explicitly:

```rust
let mut base = SkillSet::load_dir("/usr/share/yoagent/skills", "bundled")?;
let user = SkillSet::load_dir("~/.yoagent/skills", "user")?;
let workspace = SkillSet::load_dir("./skills", "workspace")?;

base.merge(user);
base.merge(workspace); // workspace wins on conflict
```

## Compatibility

By following the AgentSkills standard, skills written for yoagent work with Claude Code, Codex CLI, Gemini CLI, Cursor, OpenCode, Goose, and any other compatible agent. Write once, use everywhere.

## Design philosophy

Skills are deliberately simple:

- **No trigger engine** — the LLM decides from descriptions
- **No compile-time registration** — skills use existing tools (read_file, bash)
- **No plugin API** — skills are just files
- **No runtime loading** — loaded at startup, that's it

If a skill needs a custom tool, it can provide an [MCP](./mcp.md) server.
