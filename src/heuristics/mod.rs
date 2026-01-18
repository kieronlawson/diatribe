pub mod backchannels;
pub mod floor_holding;
pub mod micro_turns;

pub use backchannels::*;
pub use floor_holding::*;
pub use micro_turns::*;

use crate::models::TokenizedTranscript;

/// Configuration for all heuristics
#[derive(Debug, Clone)]
pub struct HeuristicsConfig {
    /// Maximum turn duration in ms to consider for micro-turn collapse
    pub micro_turn_max_ms: u64,
    /// Backchannel words to recognize
    pub backchannel_words: Vec<String>,
    /// Decay factor for floor holding score (per second)
    pub floor_decay_per_second: f64,
    /// Minimum floor score to consider a speaker as holding the floor
    pub min_floor_score: f64,
}

impl Default for HeuristicsConfig {
    fn default() -> Self {
        Self {
            micro_turn_max_ms: 300,
            backchannel_words: vec![
                "yeah".to_string(),
                "yes".to_string(),
                "yep".to_string(),
                "uh-huh".to_string(),
                "mhm".to_string(),
                "mm-hmm".to_string(),
                "okay".to_string(),
                "ok".to_string(),
                "right".to_string(),
                "sure".to_string(),
                "hmm".to_string(),
                "hm".to_string(),
                "ah".to_string(),
                "oh".to_string(),
                "uh".to_string(),
                "um".to_string(),
            ],
            floor_decay_per_second: 0.2,
            min_floor_score: 0.3,
        }
    }
}

/// Result of applying heuristics
#[derive(Debug, Clone)]
pub struct HeuristicsResult {
    /// Number of tokens relabeled
    pub tokens_relabeled: usize,
    /// Token indices that were changed
    pub changed_indices: Vec<usize>,
    /// Whether more processing is needed (heuristics disagreed or low confidence)
    pub needs_llm: bool,
}

/// Apply all deterministic heuristics to the transcript
///
/// This runs cheap fixes before calling the LLM:
/// 1. Collapse micro-turns (<300ms surrounded by same speaker)
/// 2. Apply backchannel rules (single-word acknowledgements)
/// 3. Use floor-holding model to resolve ambiguous cases
pub fn apply_heuristics(
    transcript: &mut TokenizedTranscript,
    config: &HeuristicsConfig,
) -> HeuristicsResult {
    let mut total_changed = Vec::new();

    // 1. Collapse micro-turns
    let micro_result = collapse_micro_turns(transcript, config.micro_turn_max_ms);
    total_changed.extend(micro_result.changed_indices.clone());

    // 2. Apply backchannel rules
    let backchannel_result = apply_backchannel_rules(transcript, &config.backchannel_words);
    total_changed.extend(backchannel_result.changed_indices.clone());

    // 3. Apply floor-holding model
    let floor_result = apply_floor_holding(transcript, config);
    total_changed.extend(floor_result.changed_indices.clone());

    // Deduplicate
    total_changed.sort();
    total_changed.dedup();

    // Check if LLM processing is still needed
    let needs_llm = micro_result.needs_llm || backchannel_result.needs_llm || floor_result.needs_llm;

    HeuristicsResult {
        tokens_relabeled: total_changed.len(),
        changed_indices: total_changed,
        needs_llm,
    }
}
