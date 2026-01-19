use anyhow::Result;

use crate::llm::speaker_id_prompt::{build_speaker_id_system_prompt, build_speaker_id_user_prompt};
use crate::llm::AnthropicClient;
use crate::models::{Participant, SpeakerIdConfig, SpeakerIdResult, TokenizedTranscript};

/// Execute the speaker identification stage
///
/// Analyzes the transcript to identify which participant corresponds to each
/// numeric speaker ID based on transcript content.
pub async fn execute_speaker_id(
    client: &AnthropicClient,
    transcript: &TokenizedTranscript,
    participants: &[Participant],
    config: &SpeakerIdConfig,
) -> Result<SpeakerIdResult> {
    // Build speaker context (excerpts for each speaker)
    let speaker_excerpts = build_speaker_context(transcript, config);

    // Build prompts
    let system_prompt = build_speaker_id_system_prompt();
    let user_prompt =
        build_speaker_id_user_prompt(participants, &speaker_excerpts, &transcript.speakers);

    // Send to LLM
    let (identifications, usage) = client
        .send_speaker_id_request(&system_prompt, &user_prompt)
        .await?;

    // Build result with confidence-filtered display names
    Ok(SpeakerIdResult::from_identifications(
        identifications,
        config.confidence_threshold,
        usage,
    ))
}

/// Build representative excerpts for each speaker
///
/// Extracts turns from the transcript to provide context for identification.
/// Prioritizes early turns (where introductions are likely) and turns with
/// more content.
fn build_speaker_context(
    transcript: &TokenizedTranscript,
    config: &SpeakerIdConfig,
) -> Vec<(u32, Vec<String>)> {
    let mut result = Vec::new();
    let mut total_chars = 0;

    for &speaker_id in &transcript.speakers {
        let mut excerpts = Vec::new();

        // Get turns for this speaker, sorted by start time
        let speaker_turns: Vec<_> = transcript
            .turns
            .iter()
            .filter(|t| t.speaker == speaker_id)
            .collect();

        // Select representative turns
        // Strategy: take first few turns (likely introductions) plus longest turns
        let mut selected_indices: Vec<usize> = Vec::new();

        // First 2 turns (introductions)
        for i in 0..speaker_turns.len().min(2) {
            selected_indices.push(i);
        }

        // Add longest turns if we have room
        if speaker_turns.len() > 2 {
            let mut by_length: Vec<(usize, usize)> = speaker_turns
                .iter()
                .enumerate()
                .skip(2) // Skip already-selected first 2
                .map(|(i, t)| (i, t.token_indices.len()))
                .collect();
            by_length.sort_by(|a, b| b.1.cmp(&a.1)); // Descending by length

            for (idx, _) in by_length
                .into_iter()
                .take(config.max_excerpts_per_speaker.saturating_sub(2))
            {
                if !selected_indices.contains(&idx) {
                    selected_indices.push(idx);
                }
            }
        }

        selected_indices.sort(); // Restore chronological order

        // Build excerpt strings
        for &idx in &selected_indices {
            if excerpts.len() >= config.max_excerpts_per_speaker {
                break;
            }

            let turn = &speaker_turns[idx];

            // Build excerpt text from tokens
            let words: Vec<&str> = turn
                .token_indices
                .iter()
                .filter_map(|&i| transcript.tokens.get(i))
                .map(|t| t.punctuated_word.as_deref().unwrap_or(&t.word))
                .collect();

            let excerpt = words.join(" ");

            // Check total context limit
            if total_chars + excerpt.len() > config.max_context_chars {
                break;
            }

            total_chars += excerpt.len();
            excerpts.push(excerpt);
        }

        if !excerpts.is_empty() {
            result.push((speaker_id, excerpts));
        }
    }

    result
}

/// Parse participants from a comma-separated string
pub fn parse_participants_string(input: &str) -> Vec<Participant> {
    input
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|name| Participant::new(name))
        .collect()
}

/// Parse participants from a JSON file
///
/// Expected format:
/// ```json
/// [
///   {"name": "Alice Chen", "hints": "Project manager"},
///   {"name": "Bob Smith", "hints": "Technical lead"},
///   {"name": "Carol Davis"}
/// ]
/// ```
pub fn parse_participants_file(path: &std::path::Path) -> Result<Vec<Participant>> {
    let content = std::fs::read_to_string(path)?;
    let participants: Vec<Participant> = serde_json::from_str(&content)?;
    Ok(participants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_participants_string() {
        let result = parse_participants_string("Alice Chen, Bob Smith, Carol Davis");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "Alice Chen");
        assert_eq!(result[1].name, "Bob Smith");
        assert_eq!(result[2].name, "Carol Davis");
    }

    #[test]
    fn test_parse_participants_string_with_extra_spaces() {
        let result = parse_participants_string("  Alice  ,  Bob  ");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "Alice");
        assert_eq!(result[1].name, "Bob");
    }

    #[test]
    fn test_parse_participants_string_empty() {
        let result = parse_participants_string("");
        assert_eq!(result.len(), 0);
    }
}
