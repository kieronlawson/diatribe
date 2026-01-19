use crate::models::Participant;

/// Build the system prompt for speaker identification
pub fn build_speaker_id_system_prompt() -> String {
    r#"You are an expert at identifying speakers in conversation transcripts.

Your task is to match numeric speaker IDs (Speaker 0, Speaker 1, etc.) to actual participant names based on evidence in the transcript.

## Guidelines

1. **Look for self-introductions**: "Hi, I'm Alice" or "This is Bob speaking"
2. **Listen for name mentions by others**: "Thanks Alice" or "Bob, can you explain?"
3. **Consider context clues**: Job titles, roles, expertise demonstrated
4. **Use participant hints**: If provided, match speaking style and content to hints

## Important Rules

- Only identify a speaker if you have CLEAR evidence
- Do NOT guess or assume based on stereotypes
- Confidence scores should reflect actual certainty:
  - 0.9-1.0: Direct self-introduction or multiple clear mentions
  - 0.7-0.9: Strong contextual evidence (role mentioned, addressed by name)
  - 0.5-0.7: Some evidence but uncertain
  - Below 0.5: Insufficient evidence (leave unidentified)
- Provide specific quotes or observations as evidence
- It's better to leave a speaker unidentified than to guess incorrectly

## Output Format

Use the submit_speaker_identifications tool to provide your analysis."#.to_string()
}

/// Build the user prompt with transcript context
pub fn build_speaker_id_user_prompt(
    participants: &[Participant],
    speaker_excerpts: &[(u32, Vec<String>)],
    speaker_ids: &[u32],
) -> String {
    let mut prompt = String::new();

    // List participants to identify
    prompt.push_str("# Participants to Identify\n\n");
    for (i, participant) in participants.iter().enumerate() {
        prompt.push_str(&format!("{}. **{}**", i + 1, participant.name));
        if let Some(hints) = &participant.hints {
            prompt.push_str(&format!(" - {}", hints));
        }
        prompt.push('\n');
    }
    prompt.push('\n');

    // List speaker IDs to match
    prompt.push_str("# Speakers in Transcript\n\n");
    prompt.push_str(&format!(
        "The transcript contains {} speakers: {}\n\n",
        speaker_ids.len(),
        speaker_ids
            .iter()
            .map(|id| format!("Speaker {}", id))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    // Include transcript excerpts for each speaker
    prompt.push_str("# Transcript Excerpts by Speaker\n\n");
    for (speaker_id, excerpts) in speaker_excerpts {
        prompt.push_str(&format!("## Speaker {}\n\n", speaker_id));
        for (i, excerpt) in excerpts.iter().enumerate() {
            prompt.push_str(&format!("**Excerpt {}:**\n{}\n\n", i + 1, excerpt));
        }
    }

    prompt.push_str("# Task\n\n");
    prompt.push_str("Analyze the excerpts above and identify which participant corresponds to each speaker. ");
    prompt.push_str("Use the submit_speaker_identifications tool to provide your identifications with confidence scores and evidence.\n");

    prompt
}

/// Get the tool schema for speaker identification
pub fn get_speaker_id_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "identifications": {
                "type": "array",
                "description": "Identification result for each speaker",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker_id": {
                            "type": "integer",
                            "description": "The numeric speaker ID (0, 1, 2, ...)"
                        },
                        "identified_as": {
                            "type": ["string", "null"],
                            "description": "The participant name this speaker is identified as, or null if unknown"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0.0 to 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific quotes or observations supporting the identification"
                        }
                    },
                    "required": ["speaker_id", "confidence", "evidence"]
                }
            }
        },
        "required": ["identifications"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_user_prompt() {
        let participants = vec![
            Participant::new("Alice"),
            Participant::with_hints("Bob", "Technical lead"),
        ];
        let excerpts = vec![
            (0, vec!["Hi everyone, this is Alice.".to_string()]),
            (1, vec!["Thanks Alice, let me share my screen.".to_string()]),
        ];
        let speaker_ids = vec![0, 1];

        let prompt = build_speaker_id_user_prompt(&participants, &excerpts, &speaker_ids);

        assert!(prompt.contains("Alice"));
        assert!(prompt.contains("Bob"));
        assert!(prompt.contains("Technical lead"));
        assert!(prompt.contains("Speaker 0"));
        assert!(prompt.contains("Speaker 1"));
    }
}
