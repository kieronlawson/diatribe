use crate::models::{Token, TokenizedTranscript, Window};

/// System prompt for the LLM (non-negotiable constraints)
pub const SYSTEM_PROMPT: &str = r#"You are editing a diarized transcript. You MUST follow these rules:

1. You MUST NOT add, remove, or change any words.
2. You MUST NOT change timestamps.
3. You may only reassign speaker labels for existing tokens and adjust turn boundaries.
4. Output MUST be valid JSON matching the provided schema.
5. If uncertain, do not change anything.

CONSTRAINTS:
- You have an edit budget: you may relabel at most 3% of tokens in this window. Prefer fewer changes.
- Use only the provided reason codes for changes.
- Tokens marked as "anchor" are READ-ONLY and must not be changed.
- Minimize speaker switches while maintaining conversational coherence.

REASON CODES (use only these):
- jitter_short_turn: Short turn caused by speaker jitter
- overlap_boundary: Token near overlap boundary
- lexical_continuity: Lexical continuity with surrounding tokens
- dialogue_pairing: Question/answer dialogue pairing
- backchannel_attribution: Backchannel attribution (e.g., "yeah", "uh-huh")
- do_not_change: Explicitly keeping unchanged

If you violate any rule, list it in the "violations" array."#;

/// Build the user prompt for a window
pub fn build_window_prompt(
    transcript: &TokenizedTranscript,
    window: &Window,
    edit_budget_percent: f64,
) -> String {
    let mut prompt = String::new();

    // Header with window info
    prompt.push_str(&format!(
        "# Window: {}\n",
        window.window_id
    ));
    prompt.push_str(&format!(
        "Time range: {}ms - {}ms\n",
        window.start_ms, window.end_ms
    ));
    prompt.push_str(&format!(
        "Edit budget: {} tokens ({}% of {})\n\n",
        (window.token_count() as f64 * edit_budget_percent / 100.0).ceil() as usize,
        edit_budget_percent,
        window.token_count()
    ));

    // Speaker hints (if available)
    let speaker_stats = compute_speaker_stats(transcript, window);
    if !speaker_stats.is_empty() {
        prompt.push_str("## Speaker Hints\n");
        for (speaker, stats) in &speaker_stats {
            prompt.push_str(&format!(
                "- Speaker {}: {} words, avg turn {}ms, common words: {}\n",
                speaker,
                stats.word_count,
                stats.avg_turn_duration_ms,
                stats.common_words.join(", ")
            ));
        }
        prompt.push_str("\n");
    }

    // Anchor prefix (read-only)
    if !window.anchor_prefix_indices.is_empty() {
        prompt.push_str("## Anchor Prefix (READ-ONLY)\n");
        prompt.push_str("```json\n");
        prompt.push_str(&format_tokens(transcript, &window.anchor_prefix_indices, true));
        prompt.push_str("\n```\n\n");
    }

    // Main window tokens (editable)
    prompt.push_str("## Tokens (EDITABLE)\n");
    prompt.push_str("```json\n");
    prompt.push_str(&format_tokens(transcript, &window.token_indices, false));
    prompt.push_str("\n```\n\n");

    // Anchor suffix (read-only)
    if !window.anchor_suffix_indices.is_empty() {
        prompt.push_str("## Anchor Suffix (READ-ONLY)\n");
        prompt.push_str("```json\n");
        prompt.push_str(&format_tokens(transcript, &window.anchor_suffix_indices, true));
        prompt.push_str("\n```\n\n");
    }

    // Instructions
    prompt.push_str("## Instructions\n");
    prompt.push_str("Analyze the tokens and submit a patch using the submit_patch tool.\n");
    prompt.push_str("Only relabel tokens where you are confident there is an error.\n");
    prompt.push_str("Focus on:\n");
    prompt.push_str("- Short turns that may be speaker jitter\n");
    prompt.push_str("- Backchannels attributed to the wrong speaker\n");
    prompt.push_str("- Overlap boundaries where speaker attribution may be incorrect\n");

    prompt
}

/// Format tokens as JSON for the prompt
fn format_tokens(transcript: &TokenizedTranscript, indices: &[usize], is_anchor: bool) -> String {
    let tokens: Vec<TokenDisplay> = indices
        .iter()
        .filter_map(|&i| transcript.tokens.get(i))
        .map(|t| TokenDisplay {
            token_id: t.token_id.clone(),
            word: t.word.clone(),
            start_ms: t.start_ms,
            end_ms: t.end_ms,
            speaker: t.speaker,
            speaker_conf: t.speaker_conf,
            overlap_flag: t.is_overlap_region,
            turn_id: t.turn_id.clone(),
            anchor: is_anchor,
        })
        .collect();

    serde_json::to_string_pretty(&tokens).unwrap_or_else(|_| "[]".to_string())
}

#[derive(serde::Serialize)]
struct TokenDisplay {
    token_id: String,
    word: String,
    start_ms: u64,
    end_ms: u64,
    speaker: u32,
    speaker_conf: f64,
    overlap_flag: bool,
    turn_id: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    anchor: bool,
}

/// Compute speaker statistics for hints
fn compute_speaker_stats(
    transcript: &TokenizedTranscript,
    window: &Window,
) -> Vec<(u32, SpeakerStats)> {
    use std::collections::HashMap;

    let mut stats: HashMap<u32, SpeakerStatsBuilder> = HashMap::new();

    // Collect words per speaker
    for &idx in &window.token_indices {
        if let Some(token) = transcript.tokens.get(idx) {
            let entry = stats.entry(token.speaker).or_insert_with(SpeakerStatsBuilder::new);
            entry.words.push(token.word.to_lowercase());
        }
    }

    // Compute turn durations
    for turn in &transcript.turns {
        // Check if turn overlaps with window
        if turn.start_ms < window.end_ms && turn.end_ms > window.start_ms {
            if let Some(entry) = stats.get_mut(&turn.speaker) {
                entry.turn_durations.push(turn.duration_ms());
            }
        }
    }

    // Build final stats
    stats
        .into_iter()
        .map(|(speaker, builder)| {
            let word_count = builder.words.len();
            let avg_turn_duration_ms = if builder.turn_durations.is_empty() {
                0
            } else {
                builder.turn_durations.iter().sum::<u64>() / builder.turn_durations.len() as u64
            };

            // Find most common words
            let mut word_counts: HashMap<&str, usize> = HashMap::new();
            for word in &builder.words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
            let mut common: Vec<_> = word_counts.into_iter().collect();
            common.sort_by(|a, b| b.1.cmp(&a.1));
            let common_words: Vec<String> = common
                .into_iter()
                .take(5)
                .map(|(w, _)| w.to_string())
                .collect();

            (
                speaker,
                SpeakerStats {
                    word_count,
                    avg_turn_duration_ms,
                    common_words,
                },
            )
        })
        .collect()
}

struct SpeakerStatsBuilder {
    words: Vec<String>,
    turn_durations: Vec<u64>,
}

impl SpeakerStatsBuilder {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            turn_durations: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct SpeakerStats {
    word_count: usize,
    avg_turn_duration_ms: u64,
    common_words: Vec<String>,
}
