use std::collections::HashMap;

use tracing::info;

use crate::heuristics::micro_turns::rebuild_turns;
use crate::models::{TokenizedTranscript, Window, WindowPatch, WindowSet};

/// Configuration for Stage 2 reconciliation
#[derive(Debug, Clone)]
pub struct Stage2Config {
    /// Minimum turn duration in milliseconds
    pub min_turn_duration_ms: u64,
    /// Maximum speaker switches per second (soft limit)
    pub max_switches_per_second: f64,
    /// Minimum confidence to protect stable spans
    pub stable_span_confidence: f64,
    /// Minimum windows agreeing to override stable span
    pub min_windows_for_override: usize,
}

impl Default for Stage2Config {
    fn default() -> Self {
        Self {
            min_turn_duration_ms: 700,
            max_switches_per_second: 2.0,
            stable_span_confidence: 0.8,
            min_windows_for_override: 2,
        }
    }
}

/// Result of Stage 2 reconciliation
#[derive(Debug)]
pub struct Stage2Result {
    /// Number of tokens that were relabeled
    pub tokens_relabeled: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
}

/// Candidate label for a token from a window
#[derive(Debug, Clone)]
struct LabelCandidate {
    speaker: u32,
    window_id: String,
    weight: f64,
}

/// Execute Stage 2: Global reconciliation
///
/// Because windows overlap, we may get conflicting edits. This stage:
/// 1. Collects all candidate labels for each token
/// 2. Applies weighted voting to choose final labels
/// 3. Enforces constraints (min turn duration, max switches)
pub fn execute_stage2(
    transcript: &mut TokenizedTranscript,
    windows: &WindowSet,
    patches: &[WindowPatch],
    config: &Stage2Config,
) -> Stage2Result {
    // Build a map of token_id -> list of candidate labels
    let mut candidates: HashMap<String, Vec<LabelCandidate>> = HashMap::new();

    // Collect candidates from all patches
    for patch in patches {
        let window = windows
            .windows
            .iter()
            .find(|w| w.window_id == patch.window_id);

        let window = match window {
            Some(w) => w,
            None => continue,
        };

        for relabel in &patch.token_relabels {
            // Find the token to get its timestamp for proximity calculation
            let token_timestamp = transcript
                .get_token(&relabel.token_id)
                .map(|t| t.start_ms)
                .unwrap_or(window.center_ms());

            let proximity = window.proximity_to_center(token_timestamp);
            let weight = proximity; // Could also include LLM confidence if available

            candidates
                .entry(relabel.token_id.clone())
                .or_default()
                .push(LabelCandidate {
                    speaker: relabel.new_speaker,
                    window_id: patch.window_id.clone(),
                    weight,
                });
        }
    }

    info!(
        "Stage 2: Reconciling {} token candidates",
        candidates.len()
    );

    let mut tokens_relabeled = 0;
    let mut conflicts_resolved = 0;

    // Apply weighted voting for each token
    for (token_id, token_candidates) in &candidates {
        let token = match transcript
            .tokens
            .iter_mut()
            .find(|t| t.token_id == *token_id)
        {
            Some(t) => t,
            None => continue,
        };

        // Check if this is a stable span that should be protected
        if token.speaker_conf >= config.stable_span_confidence {
            // Only override if multiple windows agree
            let agreeing_windows: Vec<_> = token_candidates
                .iter()
                .filter(|c| c.speaker != token.speaker)
                .collect();

            if agreeing_windows.len() < config.min_windows_for_override {
                continue;
            }
        }

        // If there are multiple different candidates, we have a conflict
        let unique_speakers: std::collections::HashSet<_> =
            token_candidates.iter().map(|c| c.speaker).collect();
        if unique_speakers.len() > 1 {
            conflicts_resolved += 1;
        }

        // Weighted vote
        let final_speaker = weighted_vote(token_candidates);

        if final_speaker != token.speaker {
            token.speaker = final_speaker;
            tokens_relabeled += 1;
        }
    }

    // Rebuild turns after all changes
    if tokens_relabeled > 0 {
        rebuild_turns(transcript);

        // Apply post-reconciliation constraints
        apply_constraints(transcript, config);
    }

    info!(
        "Stage 2: {} tokens relabeled, {} conflicts resolved",
        tokens_relabeled, conflicts_resolved
    );

    Stage2Result {
        tokens_relabeled,
        conflicts_resolved,
    }
}

/// Compute weighted vote for speaker assignment
fn weighted_vote(candidates: &[LabelCandidate]) -> u32 {
    let mut speaker_weights: HashMap<u32, f64> = HashMap::new();

    for candidate in candidates {
        *speaker_weights.entry(candidate.speaker).or_default() += candidate.weight;
    }

    speaker_weights
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(speaker, _)| speaker)
        .unwrap_or(0)
}

/// Apply post-reconciliation constraints
fn apply_constraints(transcript: &mut TokenizedTranscript, config: &Stage2Config) {
    // Constraint 1: Minimum turn duration
    // Find turns that are too short and try to merge them
    let short_turns: Vec<usize> = transcript
        .turns
        .iter()
        .enumerate()
        .filter(|(_, t)| t.duration_ms() < config.min_turn_duration_ms)
        .map(|(i, _)| i)
        .collect();

    for &turn_idx in short_turns.iter().rev() {
        // Try to merge with surrounding turns
        let turn = &transcript.turns[turn_idx];

        // Check if previous and next turns have the same speaker
        let prev_speaker = if turn_idx > 0 {
            Some(transcript.turns[turn_idx - 1].speaker)
        } else {
            None
        };

        let next_speaker = if turn_idx + 1 < transcript.turns.len() {
            Some(transcript.turns[turn_idx + 1].speaker)
        } else {
            None
        };

        // If surrounded by same speaker, relabel to that speaker
        if let (Some(prev), Some(next)) = (prev_speaker, next_speaker) {
            if prev == next && turn.speaker != prev {
                for &token_idx in &turn.token_indices {
                    if let Some(token) = transcript.tokens.get_mut(token_idx) {
                        token.speaker = prev;
                    }
                }
            }
        }
    }

    // Rebuild turns after constraint application
    rebuild_turns(transcript);

    // Constraint 2: Maximum switches per second (soft - just log warnings)
    let total_duration_s = transcript.duration_ms() as f64 / 1000.0;
    if total_duration_s > 0.0 {
        let switches_per_second = (transcript.turns.len() - 1) as f64 / total_duration_s;
        if switches_per_second > config.max_switches_per_second {
            tracing::warn!(
                "High switch rate: {:.2} switches/sec (limit: {:.2})",
                switches_per_second,
                config.max_switches_per_second
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_vote() {
        let candidates = vec![
            LabelCandidate {
                speaker: 0,
                window_id: "w_0".to_string(),
                weight: 0.8,
            },
            LabelCandidate {
                speaker: 1,
                window_id: "w_1".to_string(),
                weight: 0.3,
            },
            LabelCandidate {
                speaker: 0,
                window_id: "w_2".to_string(),
                weight: 0.5,
            },
        ];

        // Speaker 0 has total weight 1.3, speaker 1 has 0.3
        assert_eq!(weighted_vote(&candidates), 0);
    }

    #[test]
    fn test_weighted_vote_single() {
        let candidates = vec![LabelCandidate {
            speaker: 2,
            window_id: "w_0".to_string(),
            weight: 1.0,
        }];

        assert_eq!(weighted_vote(&candidates), 2);
    }
}
