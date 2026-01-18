use serde::{Deserialize, Serialize};

/// Root response from Deepgram API
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepgramResponse {
    pub results: DeepgramResults,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepgramResults {
    pub channels: Vec<DeepgramChannel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepgramChannel {
    pub alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepgramAlternative {
    pub words: Vec<DeepgramWord>,
    #[serde(default)]
    pub transcript: Option<String>,
}

/// A single word from Deepgram with diarization info
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeepgramWord {
    /// The recognized text
    pub word: String,
    /// Start timestamp in seconds
    pub start: f64,
    /// End timestamp in seconds
    pub end: f64,
    /// Transcription accuracy score (0-1)
    pub confidence: f64,
    /// Numeric speaker identifier
    pub speaker: u32,
    /// Reliability of speaker assignment (0-1), only for pre-recorded
    #[serde(default)]
    pub speaker_confidence: Option<f64>,
    /// Whether this word is punctuated (if available)
    #[serde(default)]
    pub punctuated_word: Option<String>,
}

impl DeepgramResponse {
    /// Extract all words from the first channel's first alternative
    pub fn words(&self) -> &[DeepgramWord] {
        self.results
            .channels
            .first()
            .and_then(|c| c.alternatives.first())
            .map(|a| a.words.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_deepgram_response() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "hello", "start": 0.5, "end": 0.8, "confidence": 0.95, "speaker": 0, "speaker_confidence": 0.85},
                            {"word": "world", "start": 0.9, "end": 1.2, "confidence": 0.92, "speaker": 1}
                        ]
                    }]
                }]
            }
        }"#;

        let response: DeepgramResponse = serde_json::from_str(json).unwrap();
        let words = response.words();

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].speaker, 0);
        assert_eq!(words[0].speaker_confidence, Some(0.85));
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].speaker, 1);
        assert_eq!(words[1].speaker_confidence, None);
    }
}
