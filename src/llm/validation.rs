use std::collections::HashSet;

use crate::models::{PatchValidation, TokenizedTranscript, Window, WindowPatch};

/// Configuration for patch validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum percentage of tokens that can be relabeled
    pub max_edit_budget_percent: f64,
    /// Allowed speaker IDs
    pub allowed_speakers: Vec<u32>,
    /// Maximum cost increase allowed
    pub max_cost_increase: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_edit_budget_percent: 3.0,
            allowed_speakers: vec![0, 1, 2, 3],
            max_cost_increase: 10.0,
        }
    }
}

/// Validate a patch against the rules
pub fn validate_patch(
    patch: &WindowPatch,
    transcript: &TokenizedTranscript,
    window: &Window,
    config: &ValidationConfig,
) -> PatchValidation {
    let mut errors = Vec::new();

    // 1. Check for self-reported violations
    if patch.has_violations() {
        errors.push(format!(
            "Patch has self-reported violations: {:?}",
            patch.violations
        ));
    }

    // 2. Check all token_ids are in the window
    let window_token_ids: HashSet<&str> = window
        .token_indices
        .iter()
        .filter_map(|&i| transcript.tokens.get(i))
        .map(|t| t.token_id.as_str())
        .collect();

    for relabel in &patch.token_relabels {
        if !window_token_ids.contains(relabel.token_id.as_str()) {
            errors.push(format!(
                "Token {} is not in the editable window",
                relabel.token_id
            ));
        }
    }

    // 3. Check all new speakers are allowed
    let allowed: HashSet<u32> = config.allowed_speakers.iter().cloned().collect();
    for relabel in &patch.token_relabels {
        if !allowed.contains(&relabel.new_speaker) {
            errors.push(format!(
                "Speaker {} is not allowed (allowed: {:?})",
                relabel.new_speaker, config.allowed_speakers
            ));
        }
    }

    // 4. Check edit budget
    let edit_budget = (window.token_count() as f64 * config.max_edit_budget_percent / 100.0).ceil() as usize;
    let edit_count = patch.relabel_count();
    let edit_budget_used = if window.token_count() > 0 {
        edit_count as f64 / window.token_count() as f64 * 100.0
    } else {
        0.0
    };

    if edit_count > edit_budget {
        errors.push(format!(
            "Edit budget exceeded: {} edits > {} allowed ({}%)",
            edit_count, edit_budget, config.max_edit_budget_percent
        ));
    }

    // 5. Verify no word or timestamp changes (should be impossible with our schema)
    // This is enforced by the schema, but we double-check
    for relabel in &patch.token_relabels {
        if let Some(token) = transcript.get_token(&relabel.token_id) {
            // The token exists and we're only changing speaker
            // Word and timestamp are not in the relabel struct, so they can't be changed
            let _ = token; // Just verify it exists
        }
    }

    // 6. Check cost function (simplified)
    let cost_before = compute_cost(transcript, window);
    let cost_after = compute_cost_after_patch(transcript, window, patch);
    let cost_increase = cost_after - cost_before;

    if cost_increase > config.max_cost_increase {
        errors.push(format!(
            "Cost increase too high: {:.2} > {:.2} max",
            cost_increase, config.max_cost_increase
        ));
    }

    if errors.is_empty() {
        PatchValidation::valid(edit_budget_used)
    } else {
        PatchValidation::invalid(errors)
    }
}

/// Compute cost function for current state
/// cost = 5*(#speaker_switches) + 2*(#turns_under_700ms)
fn compute_cost(transcript: &TokenizedTranscript, window: &Window) -> f64 {
    let mut switches = 0;
    let mut short_turns = 0;

    // Count speaker switches within window
    let window_tokens: Vec<_> = window
        .token_indices
        .iter()
        .filter_map(|&i| transcript.tokens.get(i))
        .collect();

    for pair in window_tokens.windows(2) {
        if pair[0].speaker != pair[1].speaker {
            switches += 1;
        }
    }

    // Count short turns overlapping window
    for turn in &transcript.turns {
        if turn.start_ms < window.end_ms && turn.end_ms > window.start_ms {
            if turn.duration_ms() < 700 {
                short_turns += 1;
            }
        }
    }

    (5 * switches + 2 * short_turns) as f64
}

/// Compute cost function after applying patch
fn compute_cost_after_patch(
    transcript: &TokenizedTranscript,
    window: &Window,
    patch: &WindowPatch,
) -> f64 {
    // Build a map of token_id -> new_speaker
    let relabels: std::collections::HashMap<&str, u32> = patch
        .token_relabels
        .iter()
        .map(|r| (r.token_id.as_str(), r.new_speaker))
        .collect();

    // Get effective speakers for window tokens
    let speakers: Vec<u32> = window
        .token_indices
        .iter()
        .filter_map(|&i| transcript.tokens.get(i))
        .map(|t| {
            relabels
                .get(t.token_id.as_str())
                .cloned()
                .unwrap_or(t.speaker)
        })
        .collect();

    // Count switches
    let mut switches = 0;
    for pair in speakers.windows(2) {
        if pair[0] != pair[1] {
            switches += 1;
        }
    }

    // For simplicity, assume turn count doesn't change dramatically
    // A more accurate implementation would rebuild turns and count short ones
    let short_turns = transcript
        .turns
        .iter()
        .filter(|t| t.start_ms < window.end_ms && t.end_ms > window.start_ms)
        .filter(|t| t.duration_ms() < 700)
        .count();

    (5 * switches + 2 * short_turns) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{PatchNotes, ReasonCode, TokenRelabel};

    #[test]
    fn test_validate_empty_patch() {
        let patch = WindowPatch {
            window_id: "w_0".to_string(),
            token_relabels: vec![],
            turn_edits: vec![],
            violations: vec![],
            notes: PatchNotes::default(),
        };

        // Create minimal transcript and window
        let transcript = TokenizedTranscript {
            tokens: vec![],
            turns: vec![],
            speakers: vec![0, 1],
        };

        let window = Window {
            window_id: "w_0".to_string(),
            start_ms: 0,
            end_ms: 1000,
            token_indices: vec![],
            anchor_prefix_indices: vec![],
            anchor_suffix_indices: vec![],
            is_problem_zone: false,
            problem_types: vec![],
        };

        let config = ValidationConfig::default();
        let result = validate_patch(&patch, &transcript, &window, &config);

        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_patch_with_violations() {
        let patch = WindowPatch {
            window_id: "w_0".to_string(),
            token_relabels: vec![],
            turn_edits: vec![],
            violations: vec!["I changed a word".to_string()],
            notes: PatchNotes::default(),
        };

        let transcript = TokenizedTranscript {
            tokens: vec![],
            turns: vec![],
            speakers: vec![0, 1],
        };

        let window = Window {
            window_id: "w_0".to_string(),
            start_ms: 0,
            end_ms: 1000,
            token_indices: vec![],
            anchor_prefix_indices: vec![],
            anchor_suffix_indices: vec![],
            is_problem_zone: false,
            problem_types: vec![],
        };

        let config = ValidationConfig::default();
        let result = validate_patch(&patch, &transcript, &window, &config);

        assert!(!result.is_valid);
        assert!(result.errors[0].contains("self-reported violations"));
    }
}
