use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use diatribe::{
    apply_heuristics, execute_stage1, execute_stage2, execute_stage3, normalize,
    parse_deepgram_file, AnthropicClient, AnthropicConfig, HeuristicsConfig, ProblemZoneConfig,
    Stage1Config, Stage2Config, Stage3Config, TranscriptMetadata, WindowConfig,
};

#[derive(Parser)]
#[command(name = "diatribe")]
#[command(author, version, about = "Transcript diarization improvement pipeline", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a diarized transcript to improve speaker attribution
    Process {
        /// Input transcript file (Deepgram JSON format)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for machine-readable transcript (JSON)
        #[arg(short, long)]
        output: PathBuf,

        /// Output file for human-readable transcript (text)
        #[arg(long)]
        human_readable: Option<PathBuf>,

        /// Maximum number of speakers
        #[arg(long, default_value = "4")]
        max_speakers: u32,

        /// Edit budget as percentage of tokens (0-100)
        #[arg(long, default_value = "3.0")]
        edit_budget: f64,

        /// Window size in milliseconds
        #[arg(long, default_value = "45000")]
        window_size_ms: u64,

        /// Window stride in milliseconds
        #[arg(long, default_value = "15000")]
        window_stride_ms: u64,

        /// Minimum turn duration in milliseconds
        #[arg(long, default_value = "700")]
        min_turn_ms: u64,

        /// Skip LLM processing (only run heuristics)
        #[arg(long)]
        heuristics_only: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Analyze a transcript without making changes
    Analyze {
        /// Input transcript file (Deepgram JSON format)
        #[arg(short, long)]
        input: PathBuf,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Process {
            input,
            output,
            human_readable,
            max_speakers,
            edit_budget,
            window_size_ms,
            window_stride_ms,
            min_turn_ms,
            heuristics_only,
            verbose,
        } => {
            setup_logging(verbose);
            process_transcript(
                input,
                output,
                human_readable,
                max_speakers,
                edit_budget,
                window_size_ms,
                window_stride_ms,
                min_turn_ms,
                heuristics_only,
            )
            .await
        }
        Commands::Analyze { input, verbose } => {
            setup_logging(verbose);
            analyze_transcript(input)
        }
    }
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
    tracing::subscriber::set_global_default(subscriber).ok();
}

async fn process_transcript(
    input: PathBuf,
    output: PathBuf,
    human_readable: Option<PathBuf>,
    max_speakers: u32,
    edit_budget: f64,
    window_size_ms: u64,
    window_stride_ms: u64,
    min_turn_ms: u64,
    heuristics_only: bool,
) -> Result<()> {
    info!("Loading transcript from {:?}", input);
    let mut transcript =
        parse_deepgram_file(&input).context("Failed to parse input transcript")?;

    info!(
        "Loaded {} tokens, {} turns, {} speakers",
        transcript.tokens.len(),
        transcript.turns.len(),
        transcript.speakers.len()
    );

    // Save original speakers for comparison
    let original_speakers: Vec<u32> = transcript.tokens.iter().map(|t| t.speaker).collect();

    // Stage 0: Normalize and detect problem zones
    info!("Stage 0: Normalizing transcript...");
    let window_config = WindowConfig {
        window_size_ms,
        stride_ms: window_stride_ms,
        anchor_size_ms: 5000,
        filter_problem_zones: true,
    };
    let problem_config = ProblemZoneConfig {
        min_turn_duration_ms: min_turn_ms,
        ..Default::default()
    };
    let norm_result = normalize(&mut transcript, &window_config, &problem_config);

    info!(
        "Found {} problem zones, {} windows ({} need processing)",
        norm_result.problem_zones.len(),
        norm_result.windows.total_windows(),
        norm_result.windows.problem_window_count()
    );

    // Apply heuristics
    info!("Applying heuristics...");
    let heuristics_config = HeuristicsConfig::default();
    let heuristics_result = apply_heuristics(&mut transcript, &heuristics_config);
    info!(
        "Heuristics: {} tokens relabeled, needs_llm={}",
        heuristics_result.tokens_relabeled, heuristics_result.needs_llm
    );

    let mut windows_processed = 0;

    // Stage 1 & 2: LLM processing (if not heuristics-only)
    if !heuristics_only && heuristics_result.needs_llm {
        info!("Stage 1: LLM relabeling...");

        let api_config = AnthropicConfig::from_env()?;
        let client = AnthropicClient::new(api_config);

        let stage1_config = Stage1Config {
            edit_budget_percent: edit_budget,
            validation: diatribe::llm::ValidationConfig {
                max_edit_budget_percent: edit_budget,
                allowed_speakers: (0..max_speakers).collect(),
                ..Default::default()
            },
            ..Default::default()
        };

        let stage1_result =
            execute_stage1(&client, &transcript, &norm_result.windows, &stage1_config).await?;

        info!(
            "Stage 1: {} windows processed, {} patches, {} failures",
            stage1_result.windows_processed,
            stage1_result.patches.len(),
            stage1_result.validation_failures
        );

        windows_processed = stage1_result.windows_processed;

        // Stage 2: Reconciliation
        if !stage1_result.patches.is_empty() {
            info!("Stage 2: Reconciling patches...");
            let stage2_config = Stage2Config {
                min_turn_duration_ms: min_turn_ms,
                ..Default::default()
            };
            let stage2_result = execute_stage2(
                &mut transcript,
                &norm_result.windows,
                &stage1_result.patches,
                &stage2_config,
            );
            info!(
                "Stage 2: {} tokens relabeled, {} conflicts resolved",
                stage2_result.tokens_relabeled, stage2_result.conflicts_resolved
            );
        }
    } else if heuristics_only {
        info!("Skipping LLM processing (--heuristics-only)");
    } else {
        info!("Skipping LLM processing (heuristics sufficient)");
    }

    // Stage 3: Rendering
    info!("Stage 3: Rendering output...");
    let metadata = TranscriptMetadata {
        total_tokens: transcript.tokens.len(),
        total_turns: transcript.turns.len(),
        tokens_relabeled: transcript
            .tokens
            .iter()
            .zip(original_speakers.iter())
            .filter(|(t, &orig)| t.speaker != orig)
            .count(),
        duration_ms: transcript.duration_ms(),
        windows_processed,
    };

    let stage3_config = Stage3Config::default();
    let stage3_result = execute_stage3(
        &transcript,
        &original_speakers,
        metadata,
        Some(&output),
        human_readable.as_deref(),
        &stage3_config,
    )?;

    info!("Output written to {:?}", stage3_result.machine_path);
    if let Some(human_path) = stage3_result.human_path {
        info!("Human-readable output written to {:?}", human_path);
    }

    // Summary
    let relabeled = transcript
        .tokens
        .iter()
        .zip(original_speakers.iter())
        .filter(|(t, &orig)| t.speaker != orig)
        .count();
    let relabel_pct = if !transcript.tokens.is_empty() {
        relabeled as f64 / transcript.tokens.len() as f64 * 100.0
    } else {
        0.0
    };

    info!(
        "Complete: {} tokens relabeled ({:.1}%)",
        relabeled, relabel_pct
    );

    Ok(())
}

fn analyze_transcript(input: PathBuf) -> Result<()> {
    info!("Analyzing transcript from {:?}", input);
    let mut transcript =
        parse_deepgram_file(&input).context("Failed to parse input transcript")?;

    println!("Transcript Analysis");
    println!("==================");
    println!("Total tokens: {}", transcript.tokens.len());
    println!("Total turns: {}", transcript.turns.len());
    println!("Speakers: {:?}", transcript.speakers);
    println!(
        "Duration: {:.1}s",
        transcript.duration_ms() as f64 / 1000.0
    );
    println!();

    // Detect problem zones
    let window_config = WindowConfig::default();
    let problem_config = ProblemZoneConfig::default();
    let norm_result = normalize(&mut transcript, &window_config, &problem_config);

    println!("Problem Zones");
    println!("-------------");
    let mut jitter = 0;
    let mut short_turns = 0;
    let mut overlap = 0;
    let mut low_conf = 0;

    for zone in &norm_result.problem_zones {
        match zone.problem_type {
            diatribe::models::ProblemType::SpeakerJitter => jitter += 1,
            diatribe::models::ProblemType::ShortTurn => short_turns += 1,
            diatribe::models::ProblemType::OverlapAdjacent => overlap += 1,
            diatribe::models::ProblemType::LowConfidence => low_conf += 1,
        }
    }

    println!("Speaker jitter zones: {}", jitter);
    println!("Short turn zones: {}", short_turns);
    println!("Overlap-adjacent zones: {}", overlap);
    println!("Low confidence zones: {}", low_conf);
    println!();

    println!("Windows");
    println!("-------");
    println!("Total windows: {}", norm_result.windows.total_windows());
    println!(
        "Problem windows: {}",
        norm_result.windows.problem_window_count()
    );

    // Speaker stats
    println!();
    println!("Speaker Statistics");
    println!("------------------");
    for speaker in &transcript.speakers {
        let word_count = transcript
            .tokens
            .iter()
            .filter(|t| t.speaker == *speaker)
            .count();
        let turn_count = transcript
            .turns
            .iter()
            .filter(|t| t.speaker == *speaker)
            .count();
        let avg_turn_duration = if turn_count > 0 {
            transcript
                .turns
                .iter()
                .filter(|t| t.speaker == *speaker)
                .map(|t| t.duration_ms())
                .sum::<u64>()
                / turn_count as u64
        } else {
            0
        };
        let avg_confidence = transcript
            .tokens
            .iter()
            .filter(|t| t.speaker == *speaker)
            .map(|t| t.speaker_conf)
            .sum::<f64>()
            / word_count.max(1) as f64;

        println!(
            "Speaker {}: {} words, {} turns, avg turn {:.0}ms, avg conf {:.2}",
            speaker, word_count, turn_count, avg_turn_duration, avg_confidence
        );
    }

    Ok(())
}
