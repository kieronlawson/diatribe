use crate::models::TokenizedTranscript;

use super::{HeuristicsConfig, HeuristicsResult};
use super::micro_turns::rebuild_turns;

/// Floor scores for each speaker
#[derive(Debug, Clone)]
pub struct FloorState {
    /// Current floor score per speaker (speaker_id -> score)
    pub scores: std::collections::HashMap<u32, f64>,
    /// Current time in milliseconds
    pub current_time_ms: u64,
}

impl FloorState {
    pub fn new() -> Self {
        Self {
            scores: std::collections::HashMap::new(),
            current_time_ms: 0,
        }
    }

    /// Update floor scores based on time elapsed and speaker activity
    pub fn update(&mut self, speaker: u32, duration_ms: u64, timestamp_ms: u64, config: &HeuristicsConfig) {
        // Decay all scores based on time elapsed
        let elapsed_seconds = (timestamp_ms.saturating_sub(self.current_time_ms)) as f64 / 1000.0;
        let decay = (-config.floor_decay_per_second * elapsed_seconds).exp();

        for score in self.scores.values_mut() {
            *score *= decay;
        }

        // Boost the speaking speaker's score
        let boost = (duration_ms as f64 / 1000.0) * 0.5; // 0.5 points per second of speech
        *self.scores.entry(speaker).or_insert(0.0) += boost;

        // Normalize scores to [0, 1]
        let max_score = self.scores.values().cloned().fold(0.0f64, f64::max);
        if max_score > 1.0 {
            for score in self.scores.values_mut() {
                *score /= max_score;
            }
        }

        self.current_time_ms = timestamp_ms;
    }

    /// Get the current floor holder (speaker with highest score above threshold)
    pub fn floor_holder(&self, min_score: f64) -> Option<u32> {
        self.scores
            .iter()
            .filter(|(_, score)| **score >= min_score)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(speaker, _)| *speaker)
    }

    /// Get score for a specific speaker
    pub fn get_score(&self, speaker: u32) -> f64 {
        *self.scores.get(&speaker).unwrap_or(&0.0)
    }
}

impl Default for FloorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply floor-holding model to resolve ambiguous speaker assignments
///
/// Maintains a short-term floor score per speaker. Penalizes flipping
/// the floor for 1-2 tokens when a speaker has strong floor presence.
pub fn apply_floor_holding(
    transcript: &mut TokenizedTranscript,
    config: &HeuristicsConfig,
) -> HeuristicsResult {
    let mut changed_indices = Vec::new();
    let mut needs_llm = false;
    let mut floor_state = FloorState::new();

    // First pass: build floor state without making changes
    for token in &transcript.tokens {
        floor_state.update(token.speaker, token.duration_ms(), token.start_ms, config);
    }

    // Second pass: identify tokens that may be misattributed
    floor_state = FloorState::new();

    for i in 0..transcript.tokens.len() {
        let token = &transcript.tokens[i];
        let duration = token.duration_ms();
        let timestamp = token.start_ms;

        // Update floor state
        floor_state.update(token.speaker, duration, timestamp, config);

        // Skip if confidence is high
        if token.speaker_conf >= 0.8 {
            continue;
        }

        // Check for rapid floor flip
        if is_rapid_floor_flip(transcript, i, &floor_state, config) {
            let floor_holder = floor_state.floor_holder(config.min_floor_score);

            if let Some(holder) = floor_holder {
                if transcript.tokens[i].speaker != holder {
                    // This might be a misattributed token
                    // Check if surrounding tokens suggest it should be the floor holder
                    if should_relabel_to_floor_holder(transcript, i, holder) {
                        transcript.tokens[i].speaker = holder;
                        changed_indices.push(i);
                    } else {
                        needs_llm = true;
                    }
                }
            }
        }
    }

    if !changed_indices.is_empty() {
        rebuild_turns(transcript);
    }

    HeuristicsResult {
        tokens_relabeled: changed_indices.len(),
        changed_indices,
        needs_llm,
    }
}

/// Check if a token represents a rapid floor flip (1-2 token interruption)
fn is_rapid_floor_flip(
    transcript: &TokenizedTranscript,
    token_idx: usize,
    floor_state: &FloorState,
    config: &HeuristicsConfig,
) -> bool {
    let token = &transcript.tokens[token_idx];

    // Get the floor holder
    let floor_holder = match floor_state.floor_holder(config.min_floor_score) {
        Some(h) => h,
        None => return false,
    };

    // If the token is from the floor holder, not a flip
    if token.speaker == floor_holder {
        return false;
    }

    // Check if this is an isolated attribution (1-2 tokens)
    let mut consecutive_count = 1;

    // Count consecutive tokens with same speaker before
    for j in (0..token_idx).rev() {
        if transcript.tokens[j].speaker == token.speaker {
            consecutive_count += 1;
        } else {
            break;
        }
    }

    // Count consecutive tokens with same speaker after
    for j in (token_idx + 1)..transcript.tokens.len() {
        if transcript.tokens[j].speaker == token.speaker {
            consecutive_count += 1;
        } else {
            break;
        }
    }

    // Rapid flip if only 1-2 consecutive tokens
    consecutive_count <= 2
}

/// Check if a token should be relabeled to the floor holder
fn should_relabel_to_floor_holder(
    transcript: &TokenizedTranscript,
    token_idx: usize,
    floor_holder: u32,
) -> bool {
    // Check surrounding tokens
    let prev_speaker = if token_idx > 0 {
        Some(transcript.tokens[token_idx - 1].speaker)
    } else {
        None
    };

    let next_speaker = if token_idx + 1 < transcript.tokens.len() {
        Some(transcript.tokens[token_idx + 1].speaker)
    } else {
        None
    };

    // If both neighbors are the floor holder, relabel
    matches!((prev_speaker, next_speaker), (Some(p), Some(n)) if p == floor_holder && n == floor_holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_state() {
        let config = HeuristicsConfig::default();
        let mut state = FloorState::new();

        // Speaker 0 talks for 2 seconds
        state.update(0, 2000, 0, &config);
        assert!(state.get_score(0) > 0.0);

        // Speaker 1 says one word
        state.update(1, 200, 2000, &config);

        // Speaker 0 should still be the floor holder
        assert_eq!(state.floor_holder(0.3), Some(0));
    }
}
