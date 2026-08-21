//! Shared key-value state for sub-agent communication.
//!
//! `SharedState` is a pluggable key-value store that multiple sub-agents (and
//! the parent) can read/write. The default backend is in-memory; a filesystem
//! backend is also available for persistence and large artifacts.
//!
//! # Example
//!
//! ```rust,no_run
//! use yoagent::shared_state::SharedState;
//!
//! # async fn example() {
//! let state = SharedState::new();
//! state.set("log", "big CI output...".into()).await.unwrap();
//!
//! assert_eq!(state.get("log").await, Some("big CI output...".into()));
//! assert_eq!(state.keys().await, vec!["log"]);
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

/// Default capacity for the memory backend: 10 MB.
const DEFAULT_MAX_BYTES: usize = 10 * 1024 * 1024;

/// Error returned when a `set` would exceed the capacity limit.
#[derive(Debug, Clone)]
pub struct CapacityError {
    pub key: String,
    pub value_bytes: usize,
    pub current_bytes: usize,
    pub max_bytes: usize,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SharedState capacity exceeded: storing '{}' ({} bytes) would bring total to {} / {} bytes",
            self.key, self.value_bytes, self.current_bytes + self.value_bytes, self.max_bytes
        )
    }
}

impl std::error::Error for CapacityError {}

/// Error type for shared state operations.
#[derive(Debug)]
pub enum SharedStateError {
    Capacity(CapacityError),
    Io(std::io::Error),
}

impl fmt::Display for SharedStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity(e) => write!(f, "{}", e),
            Self::Io(e) => write!(f, "SharedState I/O error: {}", e),
        }
    }
}

impl std::error::Error for SharedStateError {}

impl From<CapacityError> for SharedStateError {
    fn from(e: CapacityError) -> Self {
        Self::Capacity(e)
    }
}

impl From<std::io::Error> for SharedStateError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Pluggable storage backend for `SharedState`.
///
/// Implement this trait to back shared state with a custom store
/// (database, Redis, HTTP service, etc.).
#[async_trait::async_trait]
pub trait SharedStateBackend: Send + Sync {
    /// Get a value by key. Returns `None` if the key doesn't exist.
    async fn get(&self, key: &str) -> Result<Option<String>, SharedStateError>;

    /// Store a value. Implementations should enforce their own capacity limits.
    async fn set(&self, key: &str, value: String) -> Result<(), SharedStateError>;

    /// Remove a key. Returns `true` if the key existed.
    async fn remove(&self, key: &str) -> Result<bool, SharedStateError>;

    /// List all keys (sorted).
    async fn keys(&self) -> Result<Vec<String>, SharedStateError>;

    /// Human-readable summary of stored variables (key names + sizes).
    async fn summary(&self) -> Result<String, SharedStateError>;
}

// ---------------------------------------------------------------------------
// Memory backend (default)
// ---------------------------------------------------------------------------

/// In-memory backend backed by `HashMap` with a byte capacity limit.
pub struct MemoryBackend {
    /// Value plus the sequence number it was written at, so eviction has a
    /// definition of "oldest" — a `HashMap` has no insertion order.
    inner: RwLock<HashMap<String, (u64, String)>>,
    next_seq: std::sync::atomic::AtomicU64,
    max_bytes: usize,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            next_seq: std::sync::atomic::AtomicU64::new(0),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            next_seq: std::sync::atomic::AtomicU64::new(0),
            max_bytes,
        }
    }
}

#[async_trait::async_trait]
impl SharedStateBackend for MemoryBackend {
    async fn get(&self, key: &str) -> Result<Option<String>, SharedStateError> {
        Ok(self.inner.read().await.get(key).map(|(_, v)| v.clone()))
    }

    async fn set(&self, key: &str, value: String) -> Result<(), SharedStateError> {
        let mut map = self.inner.write().await;

        // Calculate current total excluding the old value for this key.
        let mut current: usize = map
            .iter()
            .filter(|(k, _)| k.as_str() != key)
            .map(|(k, (_, v))| k.len() + v.len())
            .sum();
        let new_entry = key.len() + value.len();

        // Evict stashed tool output, oldest first, to make room. Without this
        // the default backend wedged permanently: at ~300KB per stashed build
        // or grep output a 10MB cap holds ~33 results, after which *every*
        // write failed for the rest of the run — including the model's own
        // `shared_state set` — with no way to recover and the bytes never
        // reclaimed.
        if current + new_entry > self.max_bytes {
            let mut evictable: Vec<(u64, String, usize)> = map
                .iter()
                .filter(|(k, _)| k.as_str() != key && is_stash_key(k))
                .map(|(k, (seq, v))| (*seq, k.clone(), k.len() + v.len()))
                .collect();
            evictable.sort_by_key(|(seq, _, _)| *seq);

            for (_, victim, size) in evictable {
                if current + new_entry <= self.max_bytes {
                    break;
                }
                map.remove(&victim);
                current = current.saturating_sub(size);
                // Never silent: a marker in the transcript may still name this
                // key, and the agent will get "not found" when it follows it.
                warn!("shared state: evicted {victim} to make room for {key}");
            }
        }

        if current + new_entry > self.max_bytes {
            // Only caller-owned keys remain. Refuse rather than destroy data
            // nothing can regenerate.
            return Err(CapacityError {
                key: key.to_string(),
                value_bytes: value.len(),
                current_bytes: current,
                max_bytes: self.max_bytes,
            }
            .into());
        }

        let seq = self
            .next_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        map.insert(key.to_string(), (seq, value));
        Ok(())
    }

    async fn remove(&self, key: &str) -> Result<bool, SharedStateError> {
        Ok(self.inner.write().await.remove(key).is_some())
    }

    async fn keys(&self) -> Result<Vec<String>, SharedStateError> {
        let map = self.inner.read().await;
        let mut keys: Vec<String> = map.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    async fn summary(&self) -> Result<String, SharedStateError> {
        let map = self.inner.read().await;
        Ok(format_summary(
            map.iter().map(|(k, (_, v))| (k.as_str(), v.len())),
        ))
    }
}

// ---------------------------------------------------------------------------
// Filesystem backend
// ---------------------------------------------------------------------------

/// Filesystem backend — each key is stored as a file in a directory.
///
/// Keys are sanitized to safe filenames. Values are stored as plain text
/// (no extension) for easy inspection and debugging.
///
/// ```rust,no_run
/// use yoagent::shared_state::{SharedState, FileBackend};
///
/// # async fn example() {
/// let state = SharedState::with_backend(FileBackend::new("/tmp/agent-state"));
/// state.set("summary", "analysis results...".into()).await.unwrap();
/// // Creates /tmp/agent-state/summary with the content
/// # }
/// ```
pub struct FileBackend {
    dir: PathBuf,
    max_bytes: usize,
}

impl FileBackend {
    /// Create a new filesystem backend. The directory is created lazily on first write.
    ///
    /// **The directory is owned exclusively by this backend.** Eviction unlinks
    /// regular files it finds there, so pointing this at a directory holding
    /// anything else will lose those files.
    ///
    /// Capped at [`DEFAULT_MAX_BYTES`], matching [`MemoryBackend`]. Without a
    /// cap an agent that stashes truncated tool output would grow the directory
    /// without bound; with one, the oldest entries are evicted and a stale key
    /// simply fails to read, which a model handles as an ordinary tool error.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Create a backend with a custom byte cap.
    pub fn with_max_bytes(dir: impl AsRef<Path>, max_bytes: usize) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            max_bytes,
        }
    }

    /// Evict oldest-first until the directory fits within `max_bytes`.
    ///
    /// Ordering is by modification time. Ties (same-second writes, which are
    /// ordinary on a coarse-grained filesystem) fall back to filename so
    /// eviction is deterministic rather than filesystem-order-dependent.
    async fn evict_to_fit(&self, keep: Option<&str>) -> Result<(), SharedStateError> {
        let keep_name = keep.map(|k| {
            self.key_to_path(k)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        let mut entries: Vec<(std::time::SystemTime, String, PathBuf, u64)> = Vec::new();
        let mut total: u64 = 0;

        let mut dir = match tokio::fs::read_dir(&self.dir).await {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        while let Some(entry) = dir.next_entry().await? {
            let meta = match entry.metadata().await {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            let name = entry.file_name().to_string_lossy().into_owned();
            total += meta.len();
            entries.push((modified, name, entry.path(), meta.len()));
        }

        if total as usize <= self.max_bytes {
            return Ok(());
        }

        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        // Only stashed tool output is evictable. This directory previously
        // evicted whatever was oldest, including keys a caller had set through
        // this very backend — a parent that stored a `plan` got `None` back
        // later with no error path anywhere. The total still counts every file,
        // so caller keys constrain the budget without being destroyed by it;
        // when only they remain the write reports capacity instead.
        for (_, name, path, len) in entries.into_iter().filter(|(_, n, _, _)| is_stash_key(n)) {
            if total as usize <= self.max_bytes {
                break;
            }
            // Never evict the write that triggered this pass.
            if keep_name.as_deref() == Some(name.as_str()) {
                continue;
            }
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {
                    total = total.saturating_sub(len);
                    // Never silent: a marker frozen in the transcript may still
                    // name this key, and the verify-after-store loop only
                    // checks keys written for the current result — it cannot
                    // see that this write just evicted an earlier one.
                    warn!(
                        "shared state: evicted {} to stay under the cap",
                        path.display()
                    );
                }
                // A failed unlink must not fail the write that triggered
                // eviction — the cap is a bound, not a guarantee — but it must
                // not be silent either, or the directory grows without bound
                // while every `set` reports success.
                Err(e) => warn!("shared state: could not evict {}: {e}", path.display()),
            }
        }
        if total as usize > self.max_bytes {
            warn!(
                "shared state: {} bytes still over the {} byte cap after eviction",
                total, self.max_bytes
            );
        }
        Ok(())
    }

    /// Encode a key into a safe, reversible filename.
    /// Percent-encodes any character that isn't alphanumeric, `-`, `_`, or `.`.
    /// This avoids collisions: distinct keys always produce distinct filenames.
    fn key_to_path(&self, key: &str) -> PathBuf {
        let encoded: String = key
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c.to_string()
                } else {
                    format!("%{:02X}", c as u32)
                }
            })
            .collect();
        self.dir.join(encoded)
    }

    /// Decode a filename back into the original key.
    fn path_to_key(filename: &str) -> String {
        let mut result = String::new();
        let mut chars = filename.chars();
        while let Some(c) = chars.next() {
            if c == '%' {
                let hex: String = chars.by_ref().take(2).collect();
                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                    if let Some(decoded) = char::from_u32(code) {
                        result.push(decoded);
                        continue;
                    }
                }
                // Fallback: keep the raw percent sequence
                result.push('%');
                result.push_str(&hex);
            } else {
                result.push(c);
            }
        }
        result
    }
}

#[async_trait::async_trait]
impl SharedStateBackend for FileBackend {
    async fn get(&self, key: &str) -> Result<Option<String>, SharedStateError> {
        let path = self.key_to_path(key);
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => Ok(Some(content)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn set(&self, key: &str, value: String) -> Result<(), SharedStateError> {
        // Reject an oversized value rather than writing it and then evicting
        // it on the next line. `MemoryBackend` rejects here too, so the two
        // backends agree that over-cap is an error the caller must see — a
        // silent `Ok(())` for a value that is already gone would let the loop
        // annotate a truncation marker with a key that never existed.
        if value.len() > self.max_bytes {
            return Err(SharedStateError::Capacity(CapacityError {
                key: key.to_string(),
                value_bytes: value.len(),
                current_bytes: 0,
                max_bytes: self.max_bytes,
            }));
        }
        tokio::fs::create_dir_all(&self.dir).await?;
        let path = self.key_to_path(key);
        tokio::fs::write(&path, &value).await?;
        // Eviction runs after the write and skips this key, so the value that
        // triggered it is never the one removed.
        //
        // Unlink on failure. `evict_to_fit` propagates errors from `read_dir`
        // and `next_entry`, both of which run *after* the write — so a failure
        // returned `Err` for a value already on disk. The loop's error arm
        // never records the key, so rollback could not reach it: orphan bytes,
        // referenced by no marker, consuming cap quota forever and going on to
        // evict other keys. The exact inverse of the case handled above.
        if let Err(e) = self.evict_to_fit(Some(key)).await {
            let _ = tokio::fs::remove_file(&path).await;
            warn!("shared state: eviction failed after writing {key}; write rolled back: {e}");
            return Err(e);
        }
        Ok(())
    }

    async fn remove(&self, key: &str) -> Result<bool, SharedStateError> {
        let path = self.key_to_path(key);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    async fn keys(&self) -> Result<Vec<String>, SharedStateError> {
        let mut keys = Vec::new();
        let mut entries = match tokio::fs::read_dir(&self.dir).await {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(keys),
            Err(e) => return Err(e.into()),
        };
        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                // Skip hidden files
                if !name.starts_with('.') {
                    keys.push(Self::path_to_key(name));
                }
            }
        }
        keys.sort();
        Ok(keys)
    }

    async fn summary(&self) -> Result<String, SharedStateError> {
        let mut entries = Vec::new();
        let mut dir = match tokio::fs::read_dir(&self.dir).await {
            Ok(dir) => dir,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok("(empty)".to_string()),
            Err(e) => return Err(e.into()),
        };
        while let Some(entry) = dir.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if !name.starts_with('.') {
                    let meta = entry.metadata().await?;
                    entries.push((Self::path_to_key(name), meta.len() as usize));
                }
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(format_summary(
            entries.iter().map(|(k, s)| (k.as_str(), *s)),
        ))
    }
}

// ---------------------------------------------------------------------------
// SharedState (public API)
// ---------------------------------------------------------------------------

/// Separator between a scope name and a key. Non-printable, so it cannot
/// collide with a key a caller would plausibly choose.
const SCOPE_SEP: char = '\u{1f}';

/// Whether a key holds machine-generated stashed tool output.
///
/// The eviction line. Stash entries are *recyclable*: losing one degrades the
/// marker to an ordinary "key not found" tool result the agent can act on, and
/// the content is still in the transcript's head+tail. A caller-owned key is
/// not recyclable — nothing regenerates a parent's `plan` — so it is never
/// evicted, and a store that is full of caller keys reports capacity rather
/// than silently destroying them.
///
/// Scope-aware: a scoped write lands as `scope\u{1f}tool-out-…`, so a
/// whole-key prefix test would miss it and treat it as caller-owned.
fn is_stash_key(key: &str) -> bool {
    key.rsplit(SCOPE_SEP)
        .next()
        .unwrap_or(key)
        .starts_with(crate::context::TOOL_OUTPUT_KEY_PREFIX)
}

/// A shared string key-value store for sub-agent communication.
///
/// Cheaply cloneable (wraps `Arc`). Delegates all operations to a
/// pluggable [`SharedStateBackend`].
///
/// # Scoping
///
/// Sharing is the point of this type — a parent stores an artifact once and
/// several sub-agents read it by reference. So the default is a single flat
/// namespace where every holder sees every key.
///
/// When a sub-agent should *not* see its siblings' data, hand it a scoped
/// view via [`scoped`](Self::scoped). Keys are transparently prefixed, and
/// [`keys`](Self::keys) / [`summary`](Self::summary) report only that scope
/// with the prefix stripped — so the sub-agent cannot enumerate, read, or
/// overwrite anything outside it. Prefixing is applied on the way in, so a
/// crafted key cannot escape the scope. The unscoped handle still sees
/// everything, which is what lets the parent collect results.
///
/// ```rust
/// # use yoagent::shared_state::SharedState;
/// # async fn demo() {
/// let state = SharedState::new();
/// let researcher = state.scoped("researcher");
/// researcher.set("notes", "…".into()).await.unwrap();
///
/// // A sibling sees nothing of it.
/// assert!(state.scoped("writer").get("notes").await.is_none());
/// // The parent does.
/// assert!(researcher.get("notes").await.is_some());
/// # }
/// ```
#[derive(Clone)]
pub struct SharedState {
    backend: Arc<dyn SharedStateBackend>,
    /// `None` = the root view: full, unprefixed access.
    scope: Option<Arc<str>>,
}

impl SharedState {
    /// Create a new in-memory store with the default 10 MB capacity.
    pub fn new() -> Self {
        Self {
            backend: Arc::new(MemoryBackend::new()),
            scope: None,
        }
    }

    /// Create a new in-memory store with a custom byte capacity.
    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            backend: Arc::new(MemoryBackend::with_max_bytes(max_bytes)),
            scope: None,
        }
    }

    /// Create a store backed by a custom backend.
    pub fn with_backend(backend: impl SharedStateBackend + 'static) -> Self {
        Self {
            backend: Arc::new(backend),
            scope: None,
        }
    }

    /// A view of this store restricted to `scope`.
    ///
    /// The view shares the same backend, so the parent still sees everything
    /// the scope writes. Scoping a scoped view nests (`a` then `b` behaves as
    /// `a/b`), so a sub-agent cannot widen its own access.
    pub fn scoped(&self, scope: impl AsRef<str>) -> Self {
        let scope = match &self.scope {
            Some(existing) => format!("{existing}{SCOPE_SEP}{}", scope.as_ref()),
            None => scope.as_ref().to_string(),
        };
        Self {
            backend: Arc::clone(&self.backend),
            scope: Some(scope.into()),
        }
    }

    /// The scope this view is restricted to, if any.
    pub fn scope(&self) -> Option<&str> {
        self.scope.as_deref()
    }

    /// Map a caller-facing key to its backend key.
    fn full_key(&self, key: &str) -> String {
        match &self.scope {
            Some(scope) => format!("{scope}{SCOPE_SEP}{key}"),
            None => key.to_string(),
        }
    }

    /// The backend-key prefix for this view, if scoped.
    fn prefix(&self) -> Option<String> {
        self.scope.as_ref().map(|s| format!("{s}{SCOPE_SEP}"))
    }

    /// Get a value by key. Returns `None` if the key doesn't exist.
    pub async fn get(&self, key: &str) -> Option<String> {
        match self.backend.get(&self.full_key(key)).await {
            Ok(val) => val,
            Err(e) => {
                warn!("shared state: get({:?}) error: {}", key, e);
                None
            }
        }
    }

    /// Store a value. Returns `Err` if the backend rejects it (capacity, I/O, etc.).
    pub async fn set(&self, key: &str, value: String) -> Result<(), SharedStateError> {
        self.backend.set(&self.full_key(key), value).await
    }

    /// Remove a key. Returns `true` if the key existed.
    pub async fn remove(&self, key: &str) -> bool {
        match self.backend.remove(&self.full_key(key)).await {
            Ok(existed) => existed,
            Err(e) => {
                warn!("shared state: remove({:?}) error: {}", key, e);
                false
            }
        }
    }

    /// List keys (sorted). A scoped view lists only its own, prefix stripped.
    pub async fn keys(&self) -> Vec<String> {
        match self.backend.keys().await {
            Ok(keys) => match self.prefix() {
                Some(prefix) => keys
                    .into_iter()
                    .filter_map(|k| k.strip_prefix(&prefix).map(str::to_string))
                    .collect(),
                None => keys,
            },
            Err(e) => {
                warn!("shared state: keys() error: {}", e);
                Vec::new()
            }
        }
    }

    /// Summary for a system prompt: like [`summary`](Self::summary) but
    /// excluding machine-generated truncation stashes.
    ///
    /// The system prompt is the most prefix-cache-sensitive text in a request.
    /// A stash entry appearing here would change the prompt on every
    /// truncation, breaking the cache on every subsequent turn and filling it
    /// with kilobyte-sized keys nobody asked about. The model can still
    /// discover them at runtime via the `shared_state` tool's `list`, which
    /// uses the complete [`summary`](Self::summary).
    pub async fn prompt_summary(&self) -> String {
        let mut entries: Vec<(String, usize)> = Vec::new();
        for key in self.keys().await {
            // A scoped write lands as `scope␟tool-out-…`, so a prefix test on
            // the whole key misses every stash a scoped sub-agent made — and it
            // is the *unscoped* parent whose prompt would then carry them.
            // Compare the part after the last scope separator.
            let bare = key.rsplit(SCOPE_SEP).next().unwrap_or(&key);
            if bare.starts_with(crate::context::TOOL_OUTPUT_KEY_PREFIX) {
                continue;
            }
            let len = self.get(&key).await.map(|v| v.len()).unwrap_or(0);
            entries.push((key, len));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        format_summary(entries.iter().map(|(k, n)| (k.as_str(), *n)))
    }

    /// Human-readable summary of stored variables (key names + byte sizes).
    /// Suitable for injecting into a system prompt.
    ///
    /// A scoped view summarizes only its own keys. This matters more than the
    /// other accessors: the summary is injected into the sub-agent's system
    /// prompt, so an unscoped one would disclose every sibling's key names.
    pub async fn summary(&self) -> String {
        if self.scope.is_none() {
            return match self.backend.summary().await {
                Ok(s) => s,
                Err(e) => {
                    warn!("shared state: summary() error: {}", e);
                    "(error reading state)".to_string()
                }
            };
        }
        // Scoped: rebuild from this view's keys so nothing outside leaks.
        let mut entries: Vec<(String, usize)> = Vec::new();
        for key in self.keys().await {
            let len = self.get(&key).await.map(|v| v.len()).unwrap_or(0);
            entries.push((key, len));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        format_summary(entries.iter().map(|(k, n)| (k.as_str(), *n)))
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_summary<'a>(entries: impl Iterator<Item = (&'a str, usize)>) -> String {
    let entries: Vec<_> = entries.collect();
    if entries.is_empty() {
        return "(empty)".to_string();
    }
    entries
        .iter()
        .map(|(k, size)| format_entry(k, *size))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_entry(key: &str, bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{} ({:.1} MB)", key, bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} ({:.1} KB)", key, bytes as f64 / 1024.0)
    } else {
        format!("{} ({} bytes)", key, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_set_remove() {
        let state = SharedState::new();
        assert_eq!(state.get("x").await, None);

        state.set("x", "hello".into()).await.unwrap();
        assert_eq!(state.get("x").await, Some("hello".into()));

        assert!(state.remove("x").await);
        assert_eq!(state.get("x").await, None);
        assert!(!state.remove("x").await);
    }

    #[tokio::test]
    async fn test_keys_sorted() {
        let state = SharedState::new();
        state.set("c", "3".into()).await.unwrap();
        state.set("a", "1".into()).await.unwrap();
        state.set("b", "2".into()).await.unwrap();
        assert_eq!(state.keys().await, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn test_overwrite_same_key() {
        let state = SharedState::with_max_bytes(100);
        state.set("k", "short".into()).await.unwrap();
        state.set("k", "also short".into()).await.unwrap();
        assert_eq!(state.get("k").await, Some("also short".into()));
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let state = SharedState::with_max_bytes(20);
        state.set("a", "12345".into()).await.unwrap(); // 1 + 5 = 6 bytes
        let err = state.set("b", "12345678901234567890".into()).await;
        assert!(err.is_err());
        let e = err.unwrap_err();
        assert!(e.to_string().contains("capacity exceeded"));
    }

    #[tokio::test]
    async fn test_overwrite_within_capacity() {
        let state = SharedState::with_max_bytes(30);
        state.set("k", "aaaaaaaaaa".into()).await.unwrap(); // 1+10=11
                                                            // Overwrite with larger value — old value excluded from budget
        state.set("k", "bbbbbbbbbbbbbbbbbb".into()).await.unwrap(); // 1+18=19
        assert_eq!(state.get("k").await, Some("bbbbbbbbbbbbbbbbbb".into()));
    }

    #[tokio::test]
    async fn test_summary_formatting() {
        let state = SharedState::new();
        assert_eq!(state.summary().await, "(empty)");

        state.set("small", "hi".into()).await.unwrap();
        let s = state.summary().await;
        assert!(s.contains("small"));
        assert!(s.contains("bytes)"));
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let state = SharedState::new();
        let mut handles = vec![];
        for i in 0..10 {
            let s = state.clone();
            handles.push(tokio::spawn(async move {
                s.set(&format!("k{}", i), format!("v{}", i)).await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(state.keys().await.len(), 10);
    }

    #[tokio::test]
    async fn test_file_backend() {
        let dir = tempfile::tempdir().unwrap();
        let state = SharedState::with_backend(FileBackend::new(dir.path()));

        // Empty state
        assert_eq!(state.get("x").await, None);
        assert_eq!(state.keys().await, Vec::<String>::new());
        assert_eq!(state.summary().await, "(empty)");

        // Set and get
        state.set("report", "analysis done".into()).await.unwrap();
        assert_eq!(state.get("report").await, Some("analysis done".into()));

        // File actually exists on disk
        let content = std::fs::read_to_string(dir.path().join("report")).unwrap();
        assert_eq!(content, "analysis done");

        // Keys
        state.set("log", "build output".into()).await.unwrap();
        assert_eq!(state.keys().await, vec!["log", "report"]);

        // Summary
        let summary = state.summary().await;
        assert!(summary.contains("report"));
        assert!(summary.contains("log"));

        // Remove
        assert!(state.remove("report").await);
        assert_eq!(state.get("report").await, None);
        assert!(!state.remove("report").await);
    }

    #[tokio::test]
    async fn test_file_backend_key_encoding() {
        let dir = tempfile::tempdir().unwrap();
        let state = SharedState::with_backend(FileBackend::new(dir.path()));

        // Keys with special chars are percent-encoded (reversible)
        state
            .set("summary:src/main.rs", "file analysis".into())
            .await
            .unwrap();
        assert_eq!(
            state.get("summary:src/main.rs").await,
            Some("file analysis".into())
        );

        // The file on disk uses percent-encoded name
        let encoded = dir.path().join("summary%3Asrc%2Fmain.rs");
        assert!(encoded.exists());

        // keys() returns the original key, not the filename
        let keys = state.keys().await;
        assert!(keys.contains(&"summary:src/main.rs".to_string()));

        // No collision: distinct keys produce distinct files
        state
            .set("summary_src_main.rs", "different".into())
            .await
            .unwrap();
        assert_eq!(
            state.get("summary:src/main.rs").await,
            Some("file analysis".into())
        );
        assert_eq!(
            state.get("summary_src_main.rs").await,
            Some("different".into())
        );
        assert_eq!(state.keys().await.len(), 2);
    }

    #[tokio::test]
    async fn test_with_backend() {
        // Verify with_backend works with MemoryBackend directly
        let state = SharedState::with_backend(MemoryBackend::new());
        state.set("k", "v".into()).await.unwrap();
        assert_eq!(state.get("k").await, Some("v".into()));
    }
}
