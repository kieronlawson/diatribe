use crate::models::TokenizedTranscript;

use super::micro_turns::rebuild_turns;
use super::HeuristicsResult;

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

    // First pass: identify candidate backchannels (read-only)
    let candidates: Vec<(usize, u32, bool, f64)> = transcript
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(i, token)| {
            let word_lower = token.word.to_lowercase();
            let is_backchannel = backchannel_words.iter().any(|b| word_lower == *b);

            if !is_backchannel {
                return None;
            }

            // Only consider if in overlap region or low confidence
            if !token.is_overlap_region && token.speaker_conf >= 0.7 {
                return None;
            }

            Some((i, token.speaker, token.is_overlap_region, token.speaker_conf))
        })
        .collect();

    // Second pass: process each candidate
    for (i, current_speaker, _is_overlap, _conf) in candidates {
        // Find the floor holder (speaker with the most words in surrounding context)
        let floor_holder = find_floor_holder(transcript, i, 5000); // 5 second context

        // If the backchannel is attributed to the floor holder, it's probably wrong
        if let Some(holder) = floor_holder {
            if current_speaker == holder {
                // Find an alternative speaker (the listener)
                let listener = find_listener(transcript, i, holder);
                if let Some(new_speaker) = listener {
                    transcript.tokens[i].speaker = new_speaker;
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
fn find_floor_holder(
    transcript: &TokenizedTranscript,
    token_idx: usize,
    context_ms: u64,
) -> Option<u32> {
    let token = &transcript.tokens[token_idx];
    let start_time = token.start_ms.saturating_sub(context_ms);
    let end_time = token.end_ms + context_ms;

    // Count words per speaker in the context window (excluding the target token)
    let mut speaker_words: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();

    for (i, t) in transcript.tokens.iter().enumerate() {
        if i == token_idx {
            continue;
        }
        // Check for any overlap with the context window (not just fully contained)
        if t.start_ms < end_time && t.end_ms > start_time {
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
fn find_listener(
    transcript: &TokenizedTranscript,
    token_idx: usize,
    floor_holder: u32,
) -> Option<u32> {
    // Look for other speakers in the surrounding context
    let token = &transcript.tokens[token_idx];
    let context_ms = 10_000u64; // 10 second context
    let start_time = token.start_ms.saturating_sub(context_ms);
    let end_time = token.end_ms + context_ms;

    for t in &transcript.tokens {
        // Check for any overlap with the context window (not just fully contained)
        if t.start_ms < end_time && t.end_ms > start_time && t.speaker != floor_holder {
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

    #[test]
    fn test_find_floor_holder_with_partial_overlap() {
        // Test that tokens partially overlapping the context window are counted
        // Target token at 5.0-5.1s with 500ms context window (4.5s - 5.6s)
        // Token at 4.4-4.6s partially overlaps (ends at 4.6s > 4.5s start)
        // Token at 5.5-5.7s partially overlaps (starts at 5.5s < 5.6s end)
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "before", "start": 4.4, "end": 4.6, "confidence": 0.95, "speaker": 1},
                            {"word": "target", "start": 5.0, "end": 5.1, "confidence": 0.95, "speaker": 0},
                            {"word": "after", "start": 5.5, "end": 5.7, "confidence": 0.95, "speaker": 1}
                        ]
                    }]
                }]
            }
        }"#;

        let transcript = parse_deepgram_json(json).unwrap();
        // Context window: 5000ms - 500ms = 4500ms to 5100ms + 500ms = 5600ms
        // "before" (4400-4600ms) overlaps: 4400 < 5600 && 4600 > 4500 = true
        // "after" (5500-5700ms) overlaps: 5500 < 5600 && 5700 > 4500 = true
        let holder = find_floor_holder(&transcript, 1, 500);

        // Both "before" and "after" should be counted, both are speaker 1
        assert_eq!(holder, Some(1));
    }

    #[test]
    fn test_partial_overlap_not_counted_with_old_logic() {
        // This test demonstrates the bug that was fixed:
        // With the old logic (fully contained), edge tokens would be missed
        // Token at 4.4-4.6s is NOT fully contained in 4.5-5.6s (starts before)
        // Token at 5.5-5.7s is NOT fully contained in 4.5-5.6s (ends after)

        // Verify our overlap logic is correct
        let start_time = 4500u64;
        let end_time = 5600u64;

        // Token "before": 4400-4600ms
        let before_start = 4400u64;
        let before_end = 4600u64;

        // Old logic (fully contained): would return false
        let old_logic = before_start >= start_time && before_end <= end_time;
        assert!(!old_logic, "Old logic incorrectly excludes partial overlap");

        // New logic (any overlap): returns true
        let new_logic = before_start < end_time && before_end > start_time;
        assert!(new_logic, "New logic correctly includes partial overlap");

        // Token "after": 5500-5700ms
        let after_start = 5500u64;
        let after_end = 5700u64;

        // Old logic (fully contained): would return false
        let old_logic = after_start >= start_time && after_end <= end_time;
        assert!(!old_logic, "Old logic incorrectly excludes partial overlap");

        // New logic (any overlap): returns true
        let new_logic = after_start < end_time && after_end > start_time;
        assert!(new_logic, "New logic correctly includes partial overlap");
    }
}
