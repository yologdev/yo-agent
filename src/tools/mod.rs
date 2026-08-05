pub mod bash;
pub mod edit;
pub mod file;
pub mod list;
pub mod search;
pub mod shared_state_tool;

pub use bash::BashTool;
pub use edit::EditFileTool;
pub use file::{ReadFileTool, WriteFileTool, DEFAULT_READ_MAX_LINES};
pub use list::ListFilesTool;
pub use search::SearchTool;
pub use shared_state_tool::SharedStateTool;

use crate::types::AgentTool;

/// Get the standard set of coding agent tools.
pub fn default_tools() -> Vec<Box<dyn AgentTool>> {
    vec![
        Box::new(BashTool::default()),
        Box::new(ReadFileTool::default()),
        Box::new(WriteFileTool::new()),
        Box::new(EditFileTool::new()),
        Box::new(ListFilesTool::default()),
        Box::new(SearchTool::default()),
    ]
}
