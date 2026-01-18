use std::path::Path;

use anyhow::{Context, Result};

use crate::models::{DeepgramResponse, Token, TokenizedTranscript, Turn};

/// Parse a Deepgram JSON file into a TokenizedTranscript
pub fn parse_deepgram_file(path: &Path) -> Result<TokenizedTranscript> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("Failed to read file: {:?}", path))?;
    parse_deepgram_json(&content)
}

/// Parse Deepgram JSON string into a TokenizedTranscript
pub fn parse_deepgram_json(json: &str) -> Result<TokenizedTranscript> {
    let response: DeepgramResponse =
        serde_json::from_str(json).context("Failed to parse Deepgram JSON")?;
    tokenize_deepgram_response(&response)
}

/// Convert a Deepgram response into a TokenizedTranscript
fn tokenize_deepgram_response(response: &DeepgramResponse) -> Result<TokenizedTranscript> {
    let words = response.words();

    if words.is_empty() {
        return Ok(TokenizedTranscript {
            tokens: vec![],
            turns: vec![],
            speakers: vec![],
        });
    }

    let mut tokens: Vec<Token> = Vec::with_capacity(words.len());
    let mut turns = Vec::new();
    let mut speakers = std::collections::HashSet::new();

    let segment_id = "seg_0".to_string();
    let mut current_turn_id = 0u64;
    let mut current_speaker: Option<u32> = None;
    let mut current_turn_start_index: usize = 0;
    let mut current_turn_start_ms: u64 = 0;

    for (index, word) in words.iter().enumerate() {
        let speaker_changed = current_speaker.is_some_and(|s| s != word.speaker);

        if speaker_changed {
            // Close the current turn
            if let Some(speaker) = current_speaker {
                if let Some(last_token) = tokens.last() {
                    let turn = Turn {
                        turn_id: format!("turn_{}", current_turn_id),
                        speaker,
                        start_ms: current_turn_start_ms,
                        end_ms: last_token.end_ms,
                        token_indices: (current_turn_start_index..tokens.len()).collect(),
                    };
                    turns.push(turn);
                    current_turn_id += 1;
                }
            }
            current_turn_start_index = tokens.len();
            current_turn_start_ms = (word.start * 1000.0) as u64;
        }

        if current_speaker.is_none() {
            current_turn_start_ms = (word.start * 1000.0) as u64;
        }

        current_speaker = Some(word.speaker);
        speakers.insert(word.speaker);

        let turn_id = format!("turn_{}", current_turn_id);
        let token = Token::from_deepgram(word, index, &segment_id, &turn_id);
        tokens.push(token);
    }

    // Close the final turn
    if let (Some(speaker), Some(last_token)) = (current_speaker, tokens.last()) {
        let turn = Turn {
            turn_id: format!("turn_{}", current_turn_id),
            speaker,
            start_ms: current_turn_start_ms,
            end_ms: last_token.end_ms,
            token_indices: (current_turn_start_index..tokens.len()).collect(),
        };
        turns.push(turn);
    }

    let mut speakers: Vec<u32> = speakers.into_iter().collect();
    speakers.sort();

    Ok(TokenizedTranscript {
        tokens,
        turns,
        speakers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_deepgram_json() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "hello", "start": 0.5, "end": 0.8, "confidence": 0.95, "speaker": 0, "speaker_confidence": 0.85},
                            {"word": "world", "start": 0.9, "end": 1.2, "confidence": 0.92, "speaker": 0},
                            {"word": "how", "start": 1.5, "end": 1.7, "confidence": 0.90, "speaker": 1},
                            {"word": "are", "start": 1.8, "end": 2.0, "confidence": 0.91, "speaker": 1},
                            {"word": "you", "start": 2.1, "end": 2.3, "confidence": 0.93, "speaker": 1}
                        ]
                    }]
                }]
            }
        }"#;

        let transcript = parse_deepgram_json(json).unwrap();

        assert_eq!(transcript.tokens.len(), 5);
        assert_eq!(transcript.turns.len(), 2);
        assert_eq!(transcript.speakers, vec![0, 1]);

        // First turn: speaker 0, "hello world"
        assert_eq!(transcript.turns[0].speaker, 0);
        assert_eq!(transcript.turns[0].token_indices, vec![0, 1]);

        // Second turn: speaker 1, "how are you"
        assert_eq!(transcript.turns[1].speaker, 1);
        assert_eq!(transcript.turns[1].token_indices, vec![2, 3, 4]);
    }

    #[test]
    fn test_empty_response() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": []
                    }]
                }]
            }
        }"#;

        let transcript = parse_deepgram_json(json).unwrap();

        assert!(transcript.tokens.is_empty());
        assert!(transcript.turns.is_empty());
        assert!(transcript.speakers.is_empty());
    }
}
