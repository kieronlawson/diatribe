use serde::{Deserialize, Serialize};

use super::DeepgramWord;

/// Internal token representation with millisecond timestamps and generated IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Unique identifier for this token (UUID)
    pub token_id: String,
    /// The word text - immutable, never changed by the pipeline
    pub word: String,
    /// Start timestamp in milliseconds
    pub start_ms: u64,
    /// End timestamp in milliseconds
    pub end_ms: u64,
    /// Numeric speaker ID (from Deepgram)
    pub speaker: u32,
    /// Speaker assignment confidence (0-1)
    pub speaker_conf: f64,
    /// Transcription accuracy confidence (0-1)
    pub transcription_conf: f64,
    /// Whether this token is in an overlap region
    pub is_overlap_region: bool,
    /// Segment identifier
    pub segment_id: String,
    /// Turn identifier (changes when speaker changes)
    pub turn_id: String,
    /// Original index in the source transcript
    pub original_index: usize,
}

impl Token {
    /// Create a new token from a Deepgram word
    pub fn from_deepgram(word: &DeepgramWord, index: usize, segment_id: &str, turn_id: &str) -> Self {
        Self {
            token_id: uuid::Uuid::new_v4().to_string(),
            word: word.word.clone(),
            start_ms: (word.start * 1000.0) as u64,
            end_ms: (word.end * 1000.0) as u64,
            speaker: word.speaker,
            speaker_conf: word.speaker_confidence.unwrap_or(0.5),
            transcription_conf: word.confidence,
            is_overlap_region: false,
            segment_id: segment_id.to_string(),
            turn_id: turn_id.to_string(),
            original_index: index,
        }
    }

    /// Duration of this token in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A turn is a contiguous sequence of tokens from the same speaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Unique identifier for this turn
    pub turn_id: String,
    /// Speaker ID for this turn
    pub speaker: u32,
    /// Start time in milliseconds (from first token)
    pub start_ms: u64,
    /// End time in milliseconds (from last token)
    pub end_ms: u64,
    /// Indices into the token array
    pub token_indices: Vec<usize>,
}

impl Turn {
    /// Duration of this turn in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of tokens in this turn
    pub fn token_count(&self) -> usize {
        self.token_indices.len()
    }
}

/// Processed transcript with tokens and turns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedTranscript {
    /// All tokens in order
    pub tokens: Vec<Token>,
    /// All turns in order
    pub turns: Vec<Turn>,
    /// Set of unique speaker IDs
    pub speakers: Vec<u32>,
}

impl TokenizedTranscript {
    /// Get a token by its ID
    pub fn get_token(&self, token_id: &str) -> Option<&Token> {
        self.tokens.iter().find(|t| t.token_id == token_id)
    }

    /// Get a token by its index
    pub fn get_token_by_index(&self, index: usize) -> Option<&Token> {
        self.tokens.get(index)
    }

    /// Get a turn by its ID
    pub fn get_turn(&self, turn_id: &str) -> Option<&Turn> {
        self.turns.iter().find(|t| t.turn_id == turn_id)
    }

    /// Total duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.tokens
            .last()
            .map(|t| t.end_ms)
            .unwrap_or(0)
            .saturating_sub(self.tokens.first().map(|t| t.start_ms).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_from_deepgram() {
        let dg_word = DeepgramWord {
            word: "hello".to_string(),
            start: 0.5,
            end: 0.8,
            confidence: 0.95,
            speaker: 0,
            speaker_confidence: Some(0.85),
            punctuated_word: None,
        };

        let token = Token::from_deepgram(&dg_word, 0, "seg_0", "turn_0");

        assert_eq!(token.word, "hello");
        assert_eq!(token.start_ms, 500);
        assert_eq!(token.end_ms, 800);
        assert_eq!(token.duration_ms(), 300);
        assert_eq!(token.speaker, 0);
        assert_eq!(token.speaker_conf, 0.85);
    }
}
