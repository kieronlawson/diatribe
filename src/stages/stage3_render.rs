use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use tracing::info;

use crate::io::{HumanTranscript, MachineTranscript, TranscriptMetadata};
use crate::models::{SpeakerIdentification, TokenizedTranscript};

/// Configuration for Stage 3 rendering
#[derive(Debug, Clone)]
pub struct Stage3Config {
    /// Whether to generate machine-readable output
    pub generate_machine: bool,
    /// Whether to generate human-readable output
    pub generate_human: bool,
}

impl Default for Stage3Config {
    fn default() -> Self {
        Self {
            generate_machine: true,
            generate_human: true,
        }
    }
}

/// Result of Stage 3 rendering
#[derive(Debug)]
pub struct Stage3Result {
    /// Path to machine transcript (if generated)
    pub machine_path: Option<std::path::PathBuf>,
    /// Path to human transcript (if generated)
    pub human_path: Option<std::path::PathBuf>,
}

/// Execute Stage 3: Rendering
///
/// Produces two output views:
/// 1. Machine transcript: JSON with tokens, final speaker IDs, and timestamps
/// 2. Human transcript: Formatted text with speaker labels and turns
pub fn execute_stage3(
    transcript: &TokenizedTranscript,
    original_speakers: &[u32],
    metadata: TranscriptMetadata,
    machine_output: Option<&Path>,
    human_output: Option<&Path>,
    config: &Stage3Config,
    speaker_names: Option<&HashMap<u32, String>>,
    speaker_identifications: Option<Vec<SpeakerIdentification>>,
) -> Result<Stage3Result> {
    let mut result = Stage3Result {
        machine_path: None,
        human_path: None,
    };

    // Generate machine transcript
    if config.generate_machine {
        if let Some(path) = machine_output {
            info!("Writing machine transcript to {:?}", path);
            let machine = MachineTranscript::from_transcript(
                transcript,
                original_speakers,
                metadata,
                speaker_names,
                speaker_identifications,
            );
            machine.write_json(path)?;
            result.machine_path = Some(path.to_path_buf());
        }
    }

    // Generate human transcript
    if config.generate_human {
        if let Some(path) = human_output {
            info!("Writing human transcript to {:?}", path);
            let human = if let Some(names) = speaker_names {
                HumanTranscript::with_speaker_names(transcript, names)
            } else {
                HumanTranscript::new(transcript)
            };
            human.write_file(path)?;
            result.human_path = Some(path.to_path_buf());
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage3_config_default() {
        let config = Stage3Config::default();
        assert!(config.generate_machine);
        assert!(config.generate_human);
    }
}
