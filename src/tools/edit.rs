//! Edit tool — surgical search/replace edits on files.
//!
//! This is the most important tool for coding agents. Instead of rewriting
//! entire files, the agent specifies exact text to find and replace.
//! Modeled after Claude Code's Edit tool and Aider's search/replace blocks.

use crate::types::*;
use async_trait::async_trait;

/// Surgical file editing via exact text search/replace.
pub struct EditFileTool {
    /// Allowed directory roots (empty = no restriction).
    ///
    /// Enforced against the *resolved* path, so `..` and symlinks cannot
    /// escape. See [`PathSandbox`](crate::tools::PathSandbox).
    pub allowed_paths: Vec<String>,
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl EditFileTool {
    pub fn new() -> Self {
        Self {
            allowed_paths: Vec::new(),
        }
    }

    /// Restrict edits to these directory roots.
    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths;
        self
    }
}

#[async_trait]
impl AgentTool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn label(&self) -> &str {
        "Edit File"
    }

    fn description(&self) -> &str {
        "Make a surgical edit to a file by specifying exact text to find and replace. The old_text must match exactly (including whitespace and indentation). For creating new files, use write_file instead."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must match exactly, including whitespace)"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let cancel = ctx.cancel;
        let path = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'path' parameter".into()))?;
        let old_text = params["old_text"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'old_text' parameter".into()))?;
        let new_text = params["new_text"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'new_text' parameter".into()))?;

        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        crate::tools::PathSandbox::new(self.allowed_paths.clone()).check(path)?;

        // Read existing file
        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            ToolError::Failed(format!(
                "Cannot read {}: {}. Use write_file to create new files.",
                path, e
            ))
        })?;

        // Find the old text
        let match_count = content.matches(old_text).count();

        if match_count == 0 {
            // Provide helpful error with context
            let suggestion = find_similar_text(&content, old_text);
            let hint = if let Some(similar) = suggestion {
                format!(
                    "\n\nDid you mean:\n```\n{}\n```\nMake sure old_text matches exactly, including whitespace and indentation.",
                    similar
                )
            } else {
                "\n\nTip: Use read_file to see the current file contents, then copy the exact text you want to replace.".into()
            };

            return Err(ToolError::Failed(format!(
                "old_text not found in {}.{}",
                path, hint
            )));
        }

        if match_count > 1 {
            return Err(ToolError::Failed(format!(
                "old_text matches {} locations in {}. Include more surrounding context to make the match unique.",
                match_count, path
            )));
        }

        // Perform the replacement
        let new_content = content.replacen(old_text, new_text, 1);

        tokio::fs::write(path, &new_content)
            .await
            .map_err(|e| ToolError::Failed(format!("Cannot write {}: {}", path, e)))?;

        // Show what changed
        let old_lines = old_text.lines().count();
        let new_lines = new_text.lines().count();
        let diff_summary = if old_text == new_text {
            "No changes (old_text == new_text)".into()
        } else {
            format!(
                "Replaced {} line{} with {} line{} in {}",
                old_lines,
                if old_lines == 1 { "" } else { "s" },
                new_lines,
                if new_lines == 1 { "" } else { "s" },
                path
            )
        };

        Ok(ToolResult {
            content: vec![Content::Text { text: diff_summary }],
            details: serde_json::json!({
                "path": path,
                "old_lines": old_lines,
                "new_lines": new_lines,
            }),
        })
    }
}

/// Try to find similar text in the file (fuzzy match for better error messages).
fn find_similar_text(content: &str, target: &str) -> Option<String> {
    let target_trimmed = target.trim();
    if target_trimmed.is_empty() {
        return None;
    }

    // Try to find the first line of target in the content
    let first_line = target_trimmed.lines().next()?;
    let first_line_trimmed = first_line.trim();

    if first_line_trimmed.is_empty() {
        return None;
    }

    // Search for lines containing the first line (case-sensitive)
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains(first_line_trimmed) {
            // Return a few lines of context
            let start = i;
            let target_line_count = target_trimmed.lines().count();
            let end = (i + target_line_count + 1).min(lines.len());
            return Some(lines[start..end].join("\n"));
        }
    }

    None
}
