//! Retry with exponential backoff and jitter for provider calls.

use crate::provider::ProviderError;
use std::time::Duration;
use tracing::warn;

/// Configuration for automatic retry of transient provider errors.
///
/// Defaults: 3 retries, 1s initial delay, 2x backoff, 30s max delay.
/// Use `RetryConfig::none()` to disable retries entirely.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: usize,
    /// Initial delay before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each attempt.
    pub backoff_multiplier: f64,
    /// Maximum delay between retries (milliseconds).
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30_000,
        }
    }
}

impl RetryConfig {
    /// No retries — fail immediately on any error.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Calculate the delay for a given attempt (1-indexed).
    /// Uses exponential backoff with ±20% jitter.
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        // `saturating_sub`, not `attempt - 1`: this is a public method whose
        // 1-indexed contract is easy to miss, and `usize` underflow panics in
        // debug. `llm_compaction.rs` passed it 0-indexed and died on the first
        // retry — on a *detached* task, so the summarization simply vanished
        // and compaction fell back deterministically with nothing logged.
        // A misuse should cost the backoff, not the task.
        let base_ms = self.initial_delay_ms as f64
            * self
                .backoff_multiplier
                .powi(attempt.saturating_sub(1) as i32);
        let capped_ms = base_ms.min(self.max_delay_ms as f64);

        // Jitter: ±20% (multiply by 0.8–1.2)
        let jitter = 0.8 + rand::random::<f64>() * 0.4;
        Duration::from_millis((capped_ms * jitter) as u64)
    }
}

impl ProviderError {
    /// Whether this error is safe to retry.
    ///
    /// Retryable: rate limits (429) and network/transient errors.
    /// Not retryable: auth errors, API errors (bad request), cancellation.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Network(_))
    }

    /// If this is a rate limit with a server-specified retry delay, return it.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited {
                retry_after_ms: Some(ms),
            } => Some(Duration::from_millis(*ms)),
            _ => None,
        }
    }
}

/// Log a retry attempt.
pub(crate) fn log_retry(attempt: usize, max: usize, delay: &Duration, error: &ProviderError) {
    warn!(
        "Provider error (attempt {}/{}), retrying in {:.1}s: {}",
        attempt,
        max,
        delay.as_secs_f64(),
        error
    );
}

#[cfg(test)]
mod attempt_indexing {
    use super::RetryConfig;

    /// A zero attempt must not panic.
    ///
    /// `delay_for_attempt` documents 1-indexed and computed `attempt - 1`, so a
    /// 0-indexed caller hit `usize` underflow — a debug panic. `llm_compaction`
    /// did exactly that, and because the retry runs on a detached task the
    /// panic was invisible: the summarization vanished and compaction fell back
    /// deterministically, which is one of the behaviours #150 was filed about.
    #[test]
    fn a_zero_attempt_does_not_panic() {
        let cfg = RetryConfig {
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 60_000,
            ..RetryConfig::default()
        };
        // Reaching this line at all is the point — the old code panicked here.
        let zero = cfg.delay_for_attempt(0).as_millis();
        // Jitter is +/-20% of the base 1000ms, so 0 degrades into the same band
        // as attempt 1 rather than to something wild.
        assert!(
            (800..=1200).contains(&zero),
            "attempt 0 must degrade to the base delay, got {zero}ms"
        );
    }

    /// Backoff still grows for the documented 1-indexed usage.
    #[test]
    fn backoff_grows_with_the_attempt_number() {
        let cfg = RetryConfig {
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 60_000,
            ..RetryConfig::default()
        };
        // Jitter is +/-20%, so compare with margin rather than exactly.
        let first = cfg.delay_for_attempt(1).as_millis();
        let third = cfg.delay_for_attempt(3).as_millis();
        assert!(
            third > first * 2,
            "attempt 3 must back off well beyond attempt 1, got {first}ms then {third}ms"
        );
    }
}
