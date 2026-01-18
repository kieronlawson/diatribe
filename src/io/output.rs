use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::models::TokenizedTranscript;

/// Machine-readable output format
#[derive(Debug, Clone, Serialize)]
pub struct MachineTranscript {
    /// Tokens with final speaker assignments
    pub tokens: Vec<MachineToken>,
    /// Turn boundaries
    pub turns: Vec<MachineTurn>,
    /// Speaker IDs present in the transcript
    pub speakers: Vec<u32>,
    /// Metadata about the processing
    pub metadata: TranscriptMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct MachineToken {
    pub token_id: String,
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: u32,
    pub original_speaker: u32,
    pub was_relabeled: bool,
    pub speaker_confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MachineTurn {
    pub turn_id: String,
    pub speaker: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub word_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptMetadata {
    pub total_tokens: usize,
    pub total_turns: usize,
    pub tokens_relabeled: usize,
    pub duration_ms: u64,
    pub windows_processed: usize,
}

impl MachineTranscript {
    /// Create from a TokenizedTranscript with relabeling info
    pub fn from_transcript(
        transcript: &TokenizedTranscript,
        original_speakers: &[u32],
        metadata: TranscriptMetadata,
    ) -> Self {
        let tokens: Vec<MachineToken> = transcript
            .tokens
            .iter()
            .zip(original_speakers.iter())
            .map(|(t, &orig)| MachineToken {
                token_id: t.token_id.clone(),
                word: t.word.clone(),
                start_ms: t.start_ms,
                end_ms: t.end_ms,
                speaker: t.speaker,
                original_speaker: orig,
                was_relabeled: t.speaker != orig,
                speaker_confidence: t.speaker_conf,
            })
            .collect();

        let turns: Vec<MachineTurn> = transcript
            .turns
            .iter()
            .map(|t| MachineTurn {
                turn_id: t.turn_id.clone(),
                speaker: t.speaker,
                start_ms: t.start_ms,
                end_ms: t.end_ms,
                word_count: t.token_indices.len(),
            })
            .collect();

        Self {
            tokens,
            turns,
            speakers: transcript.speakers.clone(),
            metadata,
        }
    }

    /// Write to a JSON file
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create file: {:?}", path))?;
        serde_json::to_writer_pretty(file, self).context("Failed to write JSON")?;
        Ok(())
    }
}

/// Human-readable transcript format
pub struct HumanTranscript<'a> {
    transcript: &'a TokenizedTranscript,
}

impl<'a> HumanTranscript<'a> {
    pub fn new(transcript: &'a TokenizedTranscript) -> Self {
        Self { transcript }
    }

    /// Format the transcript as human-readable text
    pub fn format(&self) -> String {
        let mut output = String::new();

        for turn in &self.transcript.turns {
            // Format speaker header with timestamp
            let start_time = format_timestamp(turn.start_ms);
            output.push_str(&format!("[{}] Speaker {}:\n", start_time, turn.speaker));

            // Collect words for this turn
            let words: Vec<&str> = turn
                .token_indices
                .iter()
                .filter_map(|&i| self.transcript.tokens.get(i))
                .map(|t| t.word.as_str())
                .collect();

            // Join words with spaces, wrapping at ~80 characters
            let text = words.join(" ");
            let wrapped = wrap_text(&text, 80);
            output.push_str(&wrapped);
            output.push_str("\n\n");
        }

        output
    }

    /// Write to a text file
    pub fn write_file(&self, path: &Path) -> Result<()> {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create file: {:?}", path))?;
        write!(file, "{}", self.format())?;
        Ok(())
    }
}

/// Format milliseconds as MM:SS.mmm
fn format_timestamp(ms: u64) -> String {
    let seconds = ms / 1000;
    let millis = ms % 1000;
    let minutes = seconds / 60;
    let secs = seconds % 60;
    format!("{:02}:{:02}.{:03}", minutes, secs, millis)
}

/// Wrap text at approximately the given width
fn wrap_text(text: &str, width: usize) -> String {
    let mut result = String::new();
    let mut line_len = 0;

    for word in text.split_whitespace() {
        if line_len + word.len() + 1 > width && line_len > 0 {
            result.push('\n');
            line_len = 0;
        }
        if line_len > 0 {
            result.push(' ');
            line_len += 1;
        }
        result.push_str(word);
        line_len += word.len();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(0), "00:00.000");
        assert_eq!(format_timestamp(1500), "00:01.500");
        assert_eq!(format_timestamp(65_000), "01:05.000");
        assert_eq!(format_timestamp(3_661_500), "61:01.500");
    }

    #[test]
    fn test_wrap_text() {
        let text = "This is a test of the text wrapping function that should wrap at 20 chars";
        let wrapped = wrap_text(text, 20);
        for line in wrapped.lines() {
            assert!(line.len() <= 25); // Allow some slack for long words
        }
    }
}
