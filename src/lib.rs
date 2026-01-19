pub mod heuristics;
pub mod io;
pub mod llm;
pub mod models;
pub mod stages;

pub use heuristics::{apply_heuristics, HeuristicsConfig};
pub use io::{parse_deepgram_file, parse_deepgram_json, HumanTranscript, MachineTranscript, TranscriptMetadata};
pub use llm::{AnthropicClient, AnthropicConfig};
pub use models::{
    DeepgramResponse, Participant, ProblemZoneConfig, SpeakerIdConfig, SpeakerIdResult,
    SpeakerIdentification, Token, TokenizedTranscript, WindowConfig, WindowPatch,
};
pub use stages::{
    execute_speaker_id, execute_stage1, execute_stage2, execute_stage3, normalize,
    parse_participants_file, parse_participants_string, Stage1Config, Stage2Config, Stage3Config,
};
