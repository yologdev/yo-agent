//! File tools — read and write files with safety limits.

use crate::types::*;
use async_trait::async_trait;
use base64::Engine;
use std::path::Path;

/// 20 MB limit for image files
const MAX_IMAGE_SIZE_BYTES: u64 = 20 * 1024 * 1024;

fn is_image_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp")
    )
}

fn get_image_mime_type(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("png") => Some("image/png"),
        Some("webp") => Some("image/webp"),
        Some("gif") => Some("image/gif"),
        Some("bmp") => Some("image/bmp"),
        _ => None,
    }
}

/// Default number of lines returned when the caller gives no explicit `limit`.
///
/// An unbounded read used to return whole files. In a real Rust codebase the
/// median source file runs well over a thousand lines, so a single unqualified
/// read could spend ~15K tokens — and reads are roughly a fifth of all tool
/// calls in a coding agent. That is what drives a session to its context budget
/// and triggers the compaction that rewrites history.
///
/// Paging is the right bound for a file: unlike head+tail truncation it is
/// lossless and directed — the header states the true total, and the agent asks
/// for the range it actually wants.
///
/// 500 comes from measurement against a fixed compaction target, where hit rate
/// is flat between a 300- and 500-line page and degrades monotonically above
/// it — making 500 the largest page that cost nothing extra.
///
/// With [`compact_headroom_turns`] on (the default) the choice matters much
/// less: the adaptive target compacts harder as growth rises, absorbing most of
/// the difference. What matters is that a bound exists at all.
///
/// [`compact_headroom_turns`]: crate::context::ContextConfig::compact_headroom_turns
pub const DEFAULT_READ_MAX_LINES: usize = 500;

/// Read a file's contents. Supports line range for large files.
pub struct ReadFileTool {
    /// Max file size to read (prevents OOM)
    pub max_bytes: usize,
    /// Allowed directory roots (empty = no restriction)
    pub allowed_paths: Vec<String>,
    /// Lines returned when the call specifies no `limit`.
    /// Set to `usize::MAX` to restore unbounded reads.
    pub max_lines: usize,
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024, // 1MB
            allowed_paths: Vec::new(),
            max_lines: DEFAULT_READ_MAX_LINES,
        }
    }
}

impl ReadFileTool {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl AgentTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn label(&self) -> &str {
        "Read File"
    }

    fn description(&self) -> &str {
        "Read a file's contents. Supports text files with optional offset/limit, and image files (jpg, png, webp, gif, bmp) which are returned as base64-encoded images. Long files are returned one page at a time — the header states the total line count, so pass offset/limit to read further."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (optional)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'path' parameter".into()))?;

        if ctx.cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        // Check file exists and size
        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|e| ToolError::Failed(format!("Cannot access {}: {}", path, e)))?;

        // Handle image files: read as binary, return base64-encoded Content::Image
        let file_path = Path::new(path);
        if is_image_file(file_path) {
            if metadata.len() > MAX_IMAGE_SIZE_BYTES {
                return Err(ToolError::Failed(format!(
                    "Image too large ({}MB, max 20MB)",
                    metadata.len() / (1024 * 1024)
                )));
            }
            let mime_type = get_image_mime_type(file_path)
                .ok_or_else(|| ToolError::Failed("Unknown image format".into()))?;
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| ToolError::Failed(format!("Cannot read {}: {}", path, e)))?;
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            return Ok(ToolResult {
                content: vec![Content::Image {
                    data,
                    mime_type: mime_type.to_string(),
                }],
                details: serde_json::json!({ "path": path, "bytes": bytes.len() }),
            });
        }

        // Text files: check size limit and apply line offset/limit
        if metadata.len() as usize > self.max_bytes {
            return Err(ToolError::Failed(format!(
                "File too large ({} bytes, max {}). Use offset/limit for partial reads.",
                metadata.len(),
                self.max_bytes
            )));
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::Failed(format!("Cannot read {}: {}", path, e)))?;

        let offset = params["offset"].as_u64().map(|v| v.max(1) as usize);
        let limit = params["limit"].as_u64().map(|v| v as usize);

        // Always show line numbers — helps agent reference exact lines for edit_file
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        // No explicit limit falls back to `max_lines` rather than the whole
        // file. The header below reports the true total, so a partial read is
        // visible and the agent can page with offset/limit.
        let effective_limit = limit.unwrap_or(self.max_lines);
        let (start, end) = match offset {
            Some(off) => {
                let s = (off - 1).min(total);
                (s, s.saturating_add(effective_limit).min(total))
            }
            None => (0, effective_limit.min(total)),
        };

        let numbered: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4} | {}", start + i + 1, line))
            .collect();

        let header = if start > 0 || end < total {
            format!(
                "[Lines {}-{} of {} — use offset/limit to read the rest]",
                start + 1,
                end,
                total
            )
        } else {
            format!("[{} lines]", total)
        };

        let output = format!("{}\n{}", header, numbered.join("\n"));

        Ok(ToolResult {
            content: vec![Content::Text { text: output }],
            details: serde_json::json!({ "path": path }),
        })
    }
}

// ---------------------------------------------------------------------------

/// Write content to a file. Creates parent directories if needed.
pub struct WriteFileTool;

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteFileTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentTool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn label(&self) -> &str {
        "Write File"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories automatically."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'path' parameter".into()))?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'content' parameter".into()))?;

        if ctx.cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }

        // Create parent directories
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ToolError::Failed(format!("Cannot create directory: {}", e)))?;
            }
        }

        tokio::fs::write(path, content)
            .await
            .map_err(|e| ToolError::Failed(format!("Cannot write {}: {}", path, e)))?;

        let bytes = content.len();
        Ok(ToolResult {
            content: vec![Content::Text {
                text: format!("Wrote {} bytes to {}", bytes, path),
            }],
            details: serde_json::json!({ "path": path, "bytes": bytes }),
        })
    }
}
