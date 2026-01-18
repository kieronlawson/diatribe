use crate::models::TokenizedTranscript;

use super::HeuristicsResult;
use super::micro_turns::rebuild_turns;

/// Apply backchannel rules
///
/// Single-word acknowledgements in overlap-adjacent zones should default
/// to the speaker who is NOT holding the floor (the listener), unless
/// speaker confidence is high.
pub fn apply_backchannel_rules(
    transcript: &mut TokenizedTranscript,
    backchannel_words: &[String],
) -> HeuristicsResult {
    let mut changed_indices = Vec::new();
    let mut needs_llm = false;

    // Identify likely backchannels
    for (i, token) in transcript.tokens.iter_mut().enumerate() {
        // Check if this is a backchannel word
        let word_lower = token.word.to_lowercase();
        let is_backchannel = backchannel_words.iter().any(|b| word_lower == *b);

        if !is_backchannel {
            continue;
        }

        // Only consider if in overlap region or low confidence
        if !token.is_overlap_region && token.speaker_conf >= 0.7 {
            continue;
        }

        // Find the floor holder (speaker with the most words in surrounding context)
        let floor_holder = find_floor_holder(transcript, i, 5000); // 5 second context

        // If the backchannel is attributed to the floor holder, it's probably wrong
        if let Some(holder) = floor_holder {
            if token.speaker == holder {
                // Find an alternative speaker (the listener)
                let listener = find_listener(transcript, i, holder);
                if let Some(new_speaker) = listener {
                    token.speaker = new_speaker;
                    changed_indices.push(i);
                } else {
                    // Can't determine listener - need LLM
                    needs_llm = true;
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

/// Find the speaker holding the floor in the surrounding context
fn find_floor_holder(transcript: &TokenizedTranscript, token_idx: usize, context_ms: u64) -> Option<u32> {
    let token = &transcript.tokens[token_idx];
    let start_time = token.start_ms.saturating_sub(context_ms);
    let end_time = token.end_ms + context_ms;

    // Count words per speaker in the context window (excluding the target token)
    let mut speaker_words: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for (i, t) in transcript.tokens.iter().enumerate() {
        if i == token_idx {
            continue;
        }
        if t.start_ms >= start_time && t.end_ms <= end_time {
            *speaker_words.entry(t.speaker).or_insert(0) += 1;
        }
    }

    // Return speaker with most words
    speaker_words
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(speaker, _)| speaker)
}

/// Find a listener (non-floor-holder) in the transcript
fn find_listener(transcript: &TokenizedTranscript, token_idx: usize, floor_holder: u32) -> Option<u32> {
    // Look for other speakers in the surrounding context
    let token = &transcript.tokens[token_idx];
    let context_ms = 10_000u64; // 10 second context
    let start_time = token.start_ms.saturating_sub(context_ms);
    let end_time = token.end_ms + context_ms;

    for t in &transcript.tokens {
        if t.start_ms >= start_time && t.end_ms <= end_time && t.speaker != floor_holder {
            return Some(t.speaker);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::parse_deepgram_json;

    #[test]
    fn test_find_floor_holder() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "so", "start": 0.0, "end": 0.2, "confidence": 0.95, "speaker": 0},
                            {"word": "i", "start": 0.3, "end": 0.4, "confidence": 0.95, "speaker": 0},
                            {"word": "think", "start": 0.5, "end": 0.7, "confidence": 0.95, "speaker": 0},
                            {"word": "yeah", "start": 0.8, "end": 0.9, "confidence": 0.95, "speaker": 0},
                            {"word": "that", "start": 1.0, "end": 1.2, "confidence": 0.95, "speaker": 0}
                        ]
                    }]
                }]
            }
        }"#;

        let transcript = parse_deepgram_json(json).unwrap();
        let holder = find_floor_holder(&transcript, 3, 5000);

        assert_eq!(holder, Some(0)); // Speaker 0 has the most words
    }
}
