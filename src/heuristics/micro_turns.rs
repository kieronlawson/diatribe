use crate::models::TokenizedTranscript;

use super::HeuristicsResult;

/// Collapse micro-turns: turns <300ms surrounded by the same speaker
///
/// If a very short turn is sandwiched between tokens from the same speaker,
/// it's likely a diarization error and should be relabeled.
pub fn collapse_micro_turns(transcript: &mut TokenizedTranscript, max_duration_ms: u64) -> HeuristicsResult {
    let mut changed_indices = Vec::new();
    let mut needs_llm = false;

    // Find turns that are candidates for collapse
    let micro_turns: Vec<(usize, u32)> = transcript
        .turns
        .iter()
        .enumerate()
        .filter(|(_, turn)| turn.duration_ms() < max_duration_ms)
        .map(|(i, turn)| (i, turn.speaker))
        .collect();

    for (turn_idx, _turn_speaker) in micro_turns {
        let turn = &transcript.turns[turn_idx];

        // Check speakers before and after this turn
        let before_speaker = if turn_idx > 0 {
            Some(transcript.turns[turn_idx - 1].speaker)
        } else {
            None
        };

        let after_speaker = if turn_idx + 1 < transcript.turns.len() {
            Some(transcript.turns[turn_idx + 1].speaker)
        } else {
            None
        };

        // If surrounded by same speaker, relabel
        if let (Some(before), Some(after)) = (before_speaker, after_speaker) {
            if before == after {
                // Relabel all tokens in this turn to the surrounding speaker
                for &token_idx in &turn.token_indices {
                    if transcript.tokens[token_idx].speaker != before {
                        transcript.tokens[token_idx].speaker = before;
                        changed_indices.push(token_idx);
                    }
                }
            } else {
                // Surrounding speakers differ - need LLM to decide
                needs_llm = true;
            }
        }
    }

    // Rebuild turns after changes
    if !changed_indices.is_empty() {
        rebuild_turns(transcript);
    }

    HeuristicsResult {
        tokens_relabeled: changed_indices.len(),
        changed_indices,
        needs_llm,
    }
}

/// Rebuild turn boundaries after token speaker changes
pub fn rebuild_turns(transcript: &mut TokenizedTranscript) {
    if transcript.tokens.is_empty() {
        transcript.turns.clear();
        return;
    }

    let mut new_turns = Vec::new();
    let mut current_turn_id = 0u64;
    let mut current_speaker = transcript.tokens[0].speaker;
    let mut current_turn_start_index = 0usize;
    let mut current_turn_start_ms = transcript.tokens[0].start_ms;

    // First pass: build turns (read-only access to tokens)
    for i in 0..transcript.tokens.len() {
        let token = &transcript.tokens[i];
        if token.speaker != current_speaker {
            // Close current turn - get end_ms from previous token
            let prev_end_ms = if i > 0 {
                transcript.tokens[i - 1].end_ms
            } else {
                current_turn_start_ms
            };
            new_turns.push(crate::models::Turn {
                turn_id: format!("turn_{}", current_turn_id),
                speaker: current_speaker,
                start_ms: current_turn_start_ms,
                end_ms: prev_end_ms,
                token_indices: (current_turn_start_index..i).collect(),
            });
            current_turn_id += 1;
            current_speaker = token.speaker;
            current_turn_start_index = i;
            current_turn_start_ms = token.start_ms;
        }
    }

    // Close final turn
    let last_end_ms = transcript.tokens.last().map(|t| t.end_ms).unwrap_or(0);
    new_turns.push(crate::models::Turn {
        turn_id: format!("turn_{}", current_turn_id),
        speaker: current_speaker,
        start_ms: current_turn_start_ms,
        end_ms: last_end_ms,
        token_indices: (current_turn_start_index..transcript.tokens.len()).collect(),
    });

    // Second pass: update turn_ids on tokens
    for turn in &new_turns {
        for &idx in &turn.token_indices {
            transcript.tokens[idx].turn_id = turn.turn_id.clone();
        }
    }

    transcript.turns = new_turns;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::parse_deepgram_json;

    #[test]
    fn test_collapse_micro_turns() {
        // Speaker 0, then brief speaker 1 (micro-turn), then speaker 0 again
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95, "speaker": 0},
                            {"word": "there", "start": 0.6, "end": 1.0, "confidence": 0.95, "speaker": 0},
                            {"word": "yes", "start": 1.1, "end": 1.2, "confidence": 0.95, "speaker": 1},
                            {"word": "how", "start": 1.3, "end": 1.6, "confidence": 0.95, "speaker": 0},
                            {"word": "are", "start": 1.7, "end": 2.0, "confidence": 0.95, "speaker": 0}
                        ]
                    }]
                }]
            }
        }"#;

        let mut transcript = parse_deepgram_json(json).unwrap();
        let result = collapse_micro_turns(&mut transcript, 300);

        // The "yes" token should have been relabeled to speaker 0
        assert_eq!(result.tokens_relabeled, 1);
        assert_eq!(transcript.tokens[2].speaker, 0);
    }
}
