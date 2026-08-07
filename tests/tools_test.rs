//! Tests for built-in tools.

use base64::Engine;
use tokio_util::sync::CancellationToken;
use yoagent::tools::edit::EditFileTool;
use yoagent::tools::list::ListFilesTool;
use yoagent::tools::*;
use yoagent::types::*;

/// Helper to build a ToolContext for tests.
fn ctx(name: &str) -> ToolContext {
    ToolContext {
        tool_call_id: "t1".into(),
        tool_name: name.into(),
        cancel: CancellationToken::new(),
        on_update: None,
        on_progress: None,
    }
}

fn ctx_with_cancel(name: &str, cancel: CancellationToken) -> ToolContext {
    ToolContext {
        tool_call_id: "t1".into(),
        tool_name: name.into(),
        cancel,
        on_update: None,
        on_progress: None,
    }
}

#[tokio::test]
async fn test_bash_echo() {
    let tool = BashTool::new();
    let result = tool
        .execute(serde_json::json!({"command": "echo hello"}), ctx("bash"))
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("hello"));
    assert!(text.contains("Exit code: 0"));
}

#[tokio::test]
async fn test_bash_failure() {
    // Non-zero exit codes return Ok with exit code in output (for LLM self-correction)
    let tool = BashTool::new();
    let result = tool
        .execute(serde_json::json!({"command": "false"}), ctx("bash"))
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("Exit code: 1"));
}

#[tokio::test]
async fn test_bash_deny_pattern() {
    let tool = BashTool::new();
    let result = tool
        .execute(serde_json::json!({"command": "rm -rf /"}), ctx("bash"))
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("blocked"));
}

#[tokio::test]
async fn test_bash_timeout() {
    let tool = BashTool::new().with_timeout(std::time::Duration::from_millis(100));
    let result = tool
        .execute(serde_json::json!({"command": "sleep 10"}), ctx("bash"))
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("timed out"));
}

#[tokio::test]
async fn test_bash_cancel() {
    let tool = BashTool::new();
    let cancel = CancellationToken::new();
    cancel.cancel();

    let result = tool
        .execute(
            serde_json::json!({"command": "echo should not run"}),
            ctx_with_cancel("bash", cancel),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_write_file() {
    let tmp = std::env::temp_dir().join("yoagent-test-rw.txt");
    let path = tmp.to_str().unwrap();

    // Write
    let write_tool = WriteFileTool::new();
    let result = write_tool
        .execute(
            serde_json::json!({"path": path, "content": "hello from yoagent"}),
            ctx("write_file"),
        )
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("Wrote"));

    // Read
    let read_tool = ReadFileTool::new();
    let result = read_tool
        .execute(serde_json::json!({"path": path}), ctx("read_file"))
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("hello from yoagent"));

    // Cleanup
    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_file_with_offset_limit() {
    let tmp = std::env::temp_dir().join("yoagent-test-lines.txt");
    let path = tmp.to_str().unwrap();

    let content = (1..=20)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&tmp, &content).unwrap();

    let tool = ReadFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": path, "offset": 5, "limit": 3}),
            ctx("read_file"),
        )
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("line 5"));
    assert!(text.contains("line 7"));
    assert!(!text.contains("line 8"));

    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_file_not_found() {
    let tool = ReadFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": "/nonexistent/file.txt"}),
            ctx("read_file"),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_write_creates_directories() {
    let tmp = std::env::temp_dir().join("yoagent-test-nested/deep/dir/file.txt");
    let path = tmp.to_str().unwrap();

    let tool = WriteFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": path, "content": "nested!"}),
            ctx("write_file"),
        )
        .await;

    assert!(result.is_ok());
    assert!(tmp.exists());

    // Cleanup
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("yoagent-test-nested"));
}

#[tokio::test]
async fn test_search_pattern() {
    let tmp_dir = std::env::temp_dir().join("yoagent-test-search");
    let _ = std::fs::create_dir_all(&tmp_dir);
    std::fs::write(tmp_dir.join("a.txt"), "hello world\nfoo bar\nhello again").unwrap();
    std::fs::write(tmp_dir.join("b.txt"), "no match here\nhello there").unwrap();

    let tool = SearchTool::new().with_root(tmp_dir.to_str().unwrap());
    let result = tool
        .execute(serde_json::json!({"pattern": "hello"}), ctx("search"))
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("hello"));
    assert!(text.contains("3 matches") || text.contains("matches")); // 3 lines match

    let _ = std::fs::remove_dir_all(tmp_dir);
}

#[tokio::test]
async fn test_search_no_matches() {
    let tmp_dir = std::env::temp_dir().join("yoagent-test-search-empty");
    let _ = std::fs::create_dir_all(&tmp_dir);
    std::fs::write(tmp_dir.join("a.txt"), "nothing interesting").unwrap();

    let tool = SearchTool::new().with_root(tmp_dir.to_str().unwrap());
    let result = tool
        .execute(
            serde_json::json!({"pattern": "zzzznotfound"}),
            ctx("search"),
        )
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("No matches"));

    let _ = std::fs::remove_dir_all(tmp_dir);
}

// --- Edit tool tests ---

#[tokio::test]
async fn test_edit_file() {
    let tmp = std::env::temp_dir().join("yoagent-test-edit.txt");
    let path = tmp.to_str().unwrap();
    std::fs::write(&tmp, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let tool = EditFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({
                "path": path,
                "old_text": "println!(\"hello\")",
                "new_text": "println!(\"goodbye\")"
            }),
            ctx("edit_file"),
        )
        .await
        .unwrap();

    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("Replaced"));
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("goodbye"));
    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_edit_file_no_match() {
    let tmp = std::env::temp_dir().join("yoagent-test-edit-nomatch.txt");
    let path = tmp.to_str().unwrap();
    std::fs::write(&tmp, "hello world\n").unwrap();
    let tool = EditFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": path, "old_text": "nonexistent", "new_text": "bar"}),
            ctx("edit_file"),
        )
        .await;
    assert!(result.is_err());
    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_list_files_tool() {
    let tmp_dir = std::env::temp_dir().join("yoagent-test-list2");
    let _ = std::fs::create_dir_all(tmp_dir.join("sub"));
    std::fs::write(tmp_dir.join("a.rs"), "").unwrap();
    std::fs::write(tmp_dir.join("sub/c.rs"), "").unwrap();
    let tool = ListFilesTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": tmp_dir.to_str().unwrap()}),
            ctx("list_files"),
        )
        .await
        .unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("a.rs"));
    let _ = std::fs::remove_dir_all(tmp_dir);
}

#[tokio::test]
async fn test_read_file_line_numbers() {
    let tmp = std::env::temp_dir().join("yoagent-test-lineno2.txt");
    let path = tmp.to_str().unwrap();
    std::fs::write(&tmp, "first\nsecond\nthird\n").unwrap();
    let tool = ReadFileTool::new();
    let result = tool
        .execute(serde_json::json!({"path": path}), ctx("read_file"))
        .await
        .unwrap();
    let text = match &result.content[0] {
        Content::Text { text } => text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("   1 | first"));
    assert!(text.contains("   2 | second"));
    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_bash_blocked_command() {
    let tool = BashTool::new();
    let result = tool
        .execute(serde_json::json!({"command": "rm -rf /"}), ctx("bash"))
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("blocked"));
}

#[tokio::test]
async fn test_default_tools_complete() {
    let tools = yoagent::tools::default_tools();
    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert_eq!(names.len(), 6);
    assert!(names.contains(&"bash"));
    assert!(names.contains(&"edit_file"));
    assert!(names.contains(&"list_files"));
}

// --- Image support tests ---

#[tokio::test]
async fn test_read_image_file() {
    // Minimal valid PNG (1x1 pixel, transparent)
    let png_bytes: Vec<u8> = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // 8-bit RGB
        0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT chunk
        0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, // IEND chunk
        0xAE, 0x42, 0x60, 0x82,
    ];

    let tmp = std::env::temp_dir().join("yoagent-test-image.png");
    std::fs::write(&tmp, &png_bytes).unwrap();

    let tool = ReadFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": tmp.to_str().unwrap()}),
            ctx("read_file"),
        )
        .await
        .unwrap();

    match &result.content[0] {
        Content::Image { data, mime_type } => {
            assert_eq!(mime_type, "image/png");
            assert!(!data.is_empty());
            // Verify round-trip: decode should match original bytes
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(data)
                .unwrap();
            assert_eq!(decoded, png_bytes);
        }
        _ => panic!("expected Content::Image"),
    }

    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_jpeg_file() {
    let tmp = std::env::temp_dir().join("yoagent-test-image.jpg");
    std::fs::write(&tmp, b"fake-jpeg-data").unwrap();

    let tool = ReadFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": tmp.to_str().unwrap()}),
            ctx("read_file"),
        )
        .await
        .unwrap();

    match &result.content[0] {
        Content::Image { mime_type, .. } => {
            assert_eq!(mime_type, "image/jpeg");
        }
        _ => panic!("expected Content::Image for .jpg"),
    }

    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_text_file_unchanged() {
    // Non-image files should still return Content::Text
    let tmp = std::env::temp_dir().join("yoagent-test-notimage.txt");
    std::fs::write(&tmp, "just text").unwrap();

    let tool = ReadFileTool::new();
    let result = tool
        .execute(
            serde_json::json!({"path": tmp.to_str().unwrap()}),
            ctx("read_file"),
        )
        .await
        .unwrap();

    match &result.content[0] {
        Content::Text { text } => {
            assert!(text.contains("just text"));
        }
        _ => panic!("expected Content::Text for .txt file"),
    }

    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_file_pages_long_files_by_default() {
    let tmp = std::env::temp_dir().join("yoagent-test-paging.txt");
    let path = tmp.to_str().unwrap();

    let total = yoagent::tools::DEFAULT_READ_MAX_LINES * 2;
    let content = (1..=total)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&tmp, &content).unwrap();

    let tool = ReadFileTool::new();
    let result = tool
        .execute(serde_json::json!({"path": path}), ctx("read_file"))
        .await
        .unwrap();
    let Content::Text { text } = &result.content[0] else {
        panic!("expected text")
    };

    // One page, and the header states the true total so the agent can page on.
    assert!(text.contains(&format!("of {}", total)));
    assert!(text.contains("offset/limit"));
    assert!(text.contains("line 1\n") || text.contains("| line 1"));
    assert!(!text.contains(&format!("| line {}", total)));
    assert_eq!(
        text.lines().count(),
        yoagent::tools::DEFAULT_READ_MAX_LINES + 1, // + header
    );

    // Paging forward reaches the end.
    let result = tool
        .execute(
            serde_json::json!({"path": path, "offset": yoagent::tools::DEFAULT_READ_MAX_LINES + 1}),
            ctx("read_file"),
        )
        .await
        .unwrap();
    let Content::Text { text } = &result.content[0] else {
        panic!("expected text")
    };
    assert!(text.contains(&format!("| line {}", total)));

    // An explicit limit still wins, and short files are unaffected.
    let result = tool
        .execute(
            serde_json::json!({"path": path, "limit": 3}),
            ctx("read_file"),
        )
        .await
        .unwrap();
    let Content::Text { text } = &result.content[0] else {
        panic!("expected text")
    };
    assert_eq!(text.lines().count(), 4); // header + 3

    let _ = std::fs::remove_file(tmp);
}

#[tokio::test]
async fn test_read_file_unbounded_when_max_lines_disabled() {
    let tmp = std::env::temp_dir().join("yoagent-test-unbounded.txt");
    let path = tmp.to_str().unwrap();
    let total = yoagent::tools::DEFAULT_READ_MAX_LINES + 50;
    let content = (1..=total)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&tmp, &content).unwrap();

    let tool = ReadFileTool {
        max_lines: usize::MAX,
        ..Default::default()
    };
    let result = tool
        .execute(serde_json::json!({"path": path}), ctx("read_file"))
        .await
        .unwrap();
    let Content::Text { text } = &result.content[0] else {
        panic!("expected text")
    };
    assert_eq!(text.lines().count(), total + 1);
    assert!(text.contains(&format!("[{} lines]", total)));

    let _ = std::fs::remove_file(tmp);
}

// ---------------------------------------------------------------------------
// Path sandboxing — allowed_paths must actually be enforced, on every tool
// that takes a path, against the resolved path (not the string).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_tool_rejects_paths_outside_allowed_roots() {
    let tmp = std::env::temp_dir().join("yoagent-sandbox-read");
    let ws = tmp.join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("ok.txt"), "inside").unwrap();
    std::fs::write(tmp.join("secret.txt"), "outside").unwrap();

    let tool = ReadFileTool::new().with_allowed_paths(vec![ws.to_string_lossy().to_string()]);

    // Inside the root: allowed.
    assert!(tool
        .execute(
            serde_json::json!({"path": ws.join("ok.txt").to_str().unwrap()}),
            ctx("read_file")
        )
        .await
        .is_ok());

    // Absolute path outside: rejected.
    assert!(tool
        .execute(
            serde_json::json!({"path": tmp.join("secret.txt").to_str().unwrap()}),
            ctx("read_file")
        )
        .await
        .is_err());

    // Traversal that is lexically "inside" the root: rejected.
    let escape = ws.join("../secret.txt");
    assert!(
        tool.execute(
            serde_json::json!({"path": escape.to_str().unwrap()}),
            ctx("read_file")
        )
        .await
        .is_err(),
        "`..` must not escape the sandbox"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn write_and_edit_tools_reject_paths_outside_allowed_roots() {
    let tmp = std::env::temp_dir().join("yoagent-sandbox-write");
    let ws = tmp.join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("edit.txt"), "hello").unwrap();
    let outside = tmp.join("victim.txt");
    std::fs::write(&outside, "original").unwrap();

    let roots = vec![ws.to_string_lossy().to_string()];
    let write = WriteFileTool::new().with_allowed_paths(roots.clone());
    let edit = EditFileTool::new().with_allowed_paths(roots);

    // A write outside the sandbox must fail *and* leave the file untouched.
    assert!(write
        .execute(
            serde_json::json!({"path": outside.to_str().unwrap(), "content": "pwned"}),
            ctx("write_file")
        )
        .await
        .is_err());
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "original");

    // Writing a not-yet-existing file inside the sandbox still works.
    assert!(write
        .execute(
            serde_json::json!({"path": ws.join("new/deep.txt").to_str().unwrap(), "content": "ok"}),
            ctx("write_file")
        )
        .await
        .is_ok());

    assert!(edit
        .execute(
            serde_json::json!({
                "path": outside.to_str().unwrap(),
                "old_text": "original",
                "new_text": "pwned"
            }),
            ctx("edit_file")
        )
        .await
        .is_err());
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "original");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn list_and_search_tools_reject_paths_outside_allowed_roots() {
    let tmp = std::env::temp_dir().join("yoagent-sandbox-scan");
    let ws = tmp.join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(tmp.join("secret.txt"), "needle").unwrap();

    let roots = vec![ws.to_string_lossy().to_string()];
    let list = ListFilesTool::default().with_allowed_paths(roots.clone());
    let search = SearchTool::default().with_allowed_paths(roots);

    assert!(list
        .execute(
            serde_json::json!({"path": tmp.to_str().unwrap()}),
            ctx("list_files")
        )
        .await
        .is_err());
    assert!(search
        .execute(
            serde_json::json!({"pattern": "needle", "path": tmp.to_str().unwrap()}),
            ctx("search")
        )
        .await
        .is_err());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn unrestricted_tools_are_unchanged_by_default() {
    // The default is no sandbox; adding enforcement must not break it.
    let tmp = std::env::temp_dir().join("yoagent-sandbox-default");
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("f.txt"), "content").unwrap();

    let tool = ReadFileTool::new();
    assert!(tool
        .execute(
            serde_json::json!({"path": tmp.join("f.txt").to_str().unwrap()}),
            ctx("read_file")
        )
        .await
        .is_ok());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn bash_env_allowlist_hides_other_variables() {
    // Model-authored commands inherit the agent's environment by default,
    // including credentials. The allowlist is the mitigation.
    unsafe {
        std::env::set_var("YOAGENT_TEST_SECRET", "leaked-value");
        std::env::set_var("YOAGENT_TEST_KEEP", "kept-value");
    }

    let guarded = BashTool::default().with_env_allowlist(vec!["YOAGENT_TEST_KEEP".to_string()]);
    let result = guarded
        .execute(
            serde_json::json!({"command": "echo \"$YOAGENT_TEST_SECRET|$YOAGENT_TEST_KEEP\""}),
            ctx("bash"),
        )
        .await
        .unwrap();
    let out = match &result.content[0] {
        Content::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(
        !out.contains("leaked-value"),
        "secret reached the command: {out}"
    );
    assert!(out.contains("kept-value"), "allowlisted var missing: {out}");

    // Default behaviour is unchanged: the full environment is inherited.
    let plain = BashTool::default();
    let result = plain
        .execute(
            serde_json::json!({"command": "echo \"$YOAGENT_TEST_SECRET\""}),
            ctx("bash"),
        )
        .await
        .unwrap();
    let out = match &result.content[0] {
        Content::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(
        out.contains("leaked-value"),
        "default should inherit env: {out}"
    );

    unsafe {
        std::env::remove_var("YOAGENT_TEST_SECRET");
        std::env::remove_var("YOAGENT_TEST_KEEP");
    }
}
