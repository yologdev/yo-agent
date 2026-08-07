//! List files tool — directory exploration.

use crate::types::*;
use async_trait::async_trait;
use std::time::Duration;
use tokio::process::Command;

/// List files and directories. Uses `find` or `fd` for efficient traversal.
pub struct ListFilesTool {
    pub max_results: usize,
    pub timeout: Duration,
    /// Allowed directory roots (empty = no restriction).
    ///
    /// Enforced against the *resolved* path, so `..` and symlinks cannot
    /// escape. See [`PathSandbox`](crate::tools::PathSandbox).
    pub allowed_paths: Vec<String>,
}

impl ListFilesTool {
    /// Restrict listing to these directory roots.
    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths;
        self
    }
}

impl Default for ListFilesTool {
    fn default() -> Self {
        Self {
            max_results: 200,
            timeout: Duration::from_secs(10),
            allowed_paths: Vec::new(),
        }
    }
}

impl ListFilesTool {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl AgentTool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn label(&self) -> &str {
        "List Files"
    }

    fn description(&self) -> &str {
        "List files and directories. Optionally filter by glob pattern. Use to explore project structure before reading specific files."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files, e.g. '*.rs' (optional)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth (default: 3)"
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let cancel = ctx.cancel;
        let path = params["path"].as_str().unwrap_or(".");
        let pattern = params["pattern"].as_str();
        let max_depth = params["max_depth"].as_u64().unwrap_or(3);

        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        crate::tools::PathSandbox::new(self.allowed_paths.clone()).check(path)?;

        // Check path exists
        if !std::path::Path::new(path).exists() {
            return Err(ToolError::Failed(format!(
                "Directory not found: {}. Check the path and try again.",
                path
            )));
        }

        let mut cmd = Command::new("find");
        cmd.arg(path);
        cmd.args(["-maxdepth", &max_depth.to_string()]);

        if let Some(pat) = pattern {
            cmd.args(["-name", pat]);
        }

        // Exclude common noise
        cmd.args(["-not", "-path", "*/target/*"]);
        cmd.args(["-not", "-path", "*/.git/*"]);
        cmd.args(["-not", "-path", "*/node_modules/*"]);

        cmd.arg("-type").arg("f");
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let timeout = self.timeout;

        let result = tokio::select! {
            _ = cancel.cancelled() => return Err(ToolError::Cancelled),
            _ = tokio::time::sleep(timeout) => return Err(ToolError::Failed("Listing timed out".into())),
            result = cmd.output() => {
                result.map_err(|e| ToolError::Failed(format!("Failed to list: {}", e)))?
            }
        };

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let mut lines: Vec<&str> = stdout.lines().collect();
        lines.sort();

        let total = lines.len();
        let truncated = total > self.max_results;
        if truncated {
            lines.truncate(self.max_results);
        }

        let text = if lines.is_empty() {
            format!("No files found in {}", path)
        } else if truncated {
            format!(
                "{}\n\n... ({} files, showing first {})",
                lines.join("\n"),
                total,
                self.max_results
            )
        } else {
            format!("{}\n\n({} files)", lines.join("\n"), total)
        };

        Ok(ToolResult {
            content: vec![Content::Text { text }],
            details: serde_json::json!({ "total": total, "truncated": truncated }),
        })
    }
}
