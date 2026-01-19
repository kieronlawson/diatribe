use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::llm::Usage;

/// A participant that may be identified in the transcript
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// The participant's name
    pub name: String,
    /// Optional hints about the participant (e.g., role, speaking style)
    pub hints: Option<String>,
}

impl Participant {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hints: None,
        }
    }

    pub fn with_hints(name: impl Into<String>, hints: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hints: Some(hints.into()),
        }
    }
}

/// Identification result for a single speaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerIdentification {
    /// The numeric speaker ID from the transcript (0, 1, 2, ...)
    pub speaker_id: u32,
    /// The identified participant name, if any
    pub identified_as: Option<String>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Evidence supporting the identification
    pub evidence: Vec<String>,
}

/// Result of the speaker identification process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerIdResult {
    /// Identifications for each speaker
    pub identifications: Vec<SpeakerIdentification>,
    /// High-confidence mappings (speaker_id -> name) for use in output
    pub display_names: HashMap<u32, String>,
    /// API token usage
    #[serde(skip)]
    pub usage: Usage,
}

impl SpeakerIdResult {
    /// Create a new result from identifications with the given confidence threshold
    pub fn from_identifications(
        identifications: Vec<SpeakerIdentification>,
        confidence_threshold: f64,
        usage: Usage,
    ) -> Self {
        let display_names = identifications
            .iter()
            .filter(|id| id.confidence >= confidence_threshold && id.identified_as.is_some())
            .map(|id| (id.speaker_id, id.identified_as.clone().unwrap()))
            .collect();

        Self {
            identifications,
            display_names,
            usage,
        }
    }
}

/// Configuration for the speaker identification stage
#[derive(Debug, Clone)]
pub struct SpeakerIdConfig {
    /// Minimum confidence threshold for including a name in display_names
    pub confidence_threshold: f64,
    /// Maximum number of transcript excerpts to include per speaker
    pub max_excerpts_per_speaker: usize,
    /// Maximum total characters of transcript context to send to the LLM
    pub max_context_chars: usize,
}

impl Default for SpeakerIdConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            max_excerpts_per_speaker: 5,
            max_context_chars: 8000,
        }
    }
}
