use anyhow::Result;
use tracing::{info, warn};

use crate::llm::{
    build_window_prompt, validate_patch, AnthropicClient, ValidationConfig, SYSTEM_PROMPT,
};
use crate::models::{TokenizedTranscript, Window, WindowPatch, WindowSet};

/// Configuration for Stage 1
#[derive(Debug, Clone)]
pub struct Stage1Config {
    /// Edit budget as percentage of tokens
    pub edit_budget_percent: f64,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Maximum retries per window on validation failure
    pub max_retries: u32,
}

impl Default for Stage1Config {
    fn default() -> Self {
        Self {
            edit_budget_percent: 3.0,
            validation: ValidationConfig::default(),
            max_retries: 2,
        }
    }
}

/// Result of Stage 1 processing
#[derive(Debug)]
pub struct Stage1Result {
    /// Collected patches from all windows
    pub patches: Vec<WindowPatch>,
    /// Number of windows processed
    pub windows_processed: usize,
    /// Number of windows skipped (no problem zones)
    pub windows_skipped: usize,
    /// Number of validation failures
    pub validation_failures: usize,
}

/// Execute Stage 1: LLM relabeling
///
/// For each window that intersects a problem zone:
/// 1. Build the prompt with tokens and constraints
/// 2. Call Claude API with tool use
/// 3. Validate the returned patch
/// 4. Collect valid patches for reconciliation
pub async fn execute_stage1(
    client: &AnthropicClient,
    transcript: &TokenizedTranscript,
    windows: &WindowSet,
    config: &Stage1Config,
) -> Result<Stage1Result> {
    let mut patches = Vec::new();
    let mut validation_failures = 0;

    let problem_windows: Vec<&Window> = windows.problem_windows().collect();
    let problem_window_count = problem_windows.len();
    let windows_skipped = windows.total_windows() - problem_window_count;

    info!(
        "Stage 1: Processing {} problem windows ({} skipped)",
        problem_window_count,
        windows_skipped
    );

    for window in problem_windows {
        match process_window(client, transcript, window, config).await {
            Ok(patch) => {
                if !patch.is_empty() {
                    info!(
                        "Window {}: {} relabels, {} turn edits",
                        window.window_id,
                        patch.relabel_count(),
                        patch.turn_edits.len()
                    );
                    patches.push(patch);
                } else {
                    info!("Window {}: no changes", window.window_id);
                }
            }
            Err(e) => {
                warn!("Window {} failed: {}", window.window_id, e);
                validation_failures += 1;
            }
        }
    }

    Ok(Stage1Result {
        windows_processed: problem_window_count,
        windows_skipped,
        patches,
        validation_failures,
    })
}

/// Process a single window
async fn process_window(
    client: &AnthropicClient,
    transcript: &TokenizedTranscript,
    window: &Window,
    config: &Stage1Config,
) -> Result<WindowPatch> {
    let prompt = build_window_prompt(transcript, window, config.edit_budget_percent);

    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        if attempt > 0 {
            info!(
                "Window {}: retry {} of {}",
                window.window_id, attempt, config.max_retries
            );
        }

        match client.send_with_tool(SYSTEM_PROMPT, &prompt).await {
            Ok(patch) => {
                // Validate the patch
                let validation = validate_patch(&patch, transcript, window, &config.validation);

                if validation.is_valid {
                    return Ok(patch);
                } else {
                    last_error = Some(anyhow::anyhow!(
                        "Validation failed: {:?}",
                        validation.errors
                    ));
                    warn!(
                        "Window {} validation failed: {:?}",
                        window.window_id, validation.errors
                    );
                }
            }
            Err(e) => {
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage1_config_default() {
        let config = Stage1Config::default();
        assert_eq!(config.edit_budget_percent, 3.0);
        assert_eq!(config.max_retries, 2);
    }
}
