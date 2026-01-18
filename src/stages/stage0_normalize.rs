use crate::models::{
    ProblemType, ProblemZoneConfig, TokenizedTranscript, Window, WindowConfig,
    WindowSet,
};

/// Result of Stage 0 normalization
#[derive(Debug)]
pub struct NormalizationResult {
    /// The tokenized transcript with problem zones marked
    pub transcript: TokenizedTranscript,
    /// Generated windows for processing
    pub windows: WindowSet,
    /// Detected problem zones
    pub problem_zones: Vec<ProblemZone>,
}

/// A detected problem zone in the transcript
#[derive(Debug, Clone)]
pub struct ProblemZone {
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Type of problem
    pub problem_type: ProblemType,
    /// Affected token indices
    pub token_indices: Vec<usize>,
}

/// Perform Stage 0: Pre-LLM normalization
///
/// This stage:
/// 1. Detects problem zones (jitter, short turns, overlap, low confidence)
/// 2. Marks overlap regions on tokens
/// 3. Builds processing windows
/// 4. Filters windows to only those intersecting problem zones
pub fn normalize(
    transcript: &mut TokenizedTranscript,
    window_config: &WindowConfig,
    problem_config: &ProblemZoneConfig,
) -> NormalizationResult {
    // Detect overlap regions
    detect_overlap_regions(transcript);

    // Detect all problem zones
    let problem_zones = detect_problem_zones(transcript, problem_config);

    // Build windows
    let windows = build_windows(transcript, window_config, &problem_zones);

    NormalizationResult {
        transcript: transcript.clone(),
        windows,
        problem_zones,
    }
}

/// Detect overlap regions where multiple speakers might be active
fn detect_overlap_regions(transcript: &mut TokenizedTranscript) {
    // Simple heuristic: if two consecutive tokens have different speakers
    // and their time ranges overlap or are very close, mark as overlap region
    let overlap_threshold_ms = 100; // 100ms gap or less

    for i in 0..transcript.tokens.len() {
        if i + 1 < transcript.tokens.len() {
            let current = &transcript.tokens[i];
            let next = &transcript.tokens[i + 1];

            // Check if speakers differ and times are close/overlapping
            if current.speaker != next.speaker {
                let gap = next.start_ms.saturating_sub(current.end_ms);
                if gap <= overlap_threshold_ms || next.start_ms < current.end_ms {
                    // Mark both tokens as overlap region
                    transcript.tokens[i].is_overlap_region = true;
                    transcript.tokens[i + 1].is_overlap_region = true;
                }
            }
        }
    }
}

/// Detect all problem zones in the transcript
fn detect_problem_zones(
    transcript: &TokenizedTranscript,
    config: &ProblemZoneConfig,
) -> Vec<ProblemZone> {
    let mut zones = Vec::new();

    // 1. Detect speaker jitter
    zones.extend(detect_speaker_jitter(transcript, config));

    // 2. Detect short turns
    zones.extend(detect_short_turns(transcript, config));

    // 3. Detect overlap-adjacent regions
    zones.extend(detect_overlap_adjacent(transcript, config));

    // 4. Detect low confidence regions
    zones.extend(detect_low_confidence(transcript, config));

    zones
}

/// Detect regions with high speaker switch rate
fn detect_speaker_jitter(
    transcript: &TokenizedTranscript,
    config: &ProblemZoneConfig,
) -> Vec<ProblemZone> {
    let mut zones = Vec::new();
    let window_ms = 10_000u64; // 10 second windows

    if transcript.tokens.is_empty() {
        return zones;
    }

    let total_duration = transcript.duration_ms();
    let mut window_start = 0u64;

    while window_start < total_duration {
        let window_end = window_start + window_ms;

        // Find tokens in this window
        let tokens_in_window: Vec<usize> = transcript
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.start_ms >= window_start && t.start_ms < window_end)
            .map(|(i, _)| i)
            .collect();

        if tokens_in_window.len() < 2 {
            window_start += window_ms / 2; // 50% overlap
            continue;
        }

        // Count speaker switches
        let mut switches = 0u32;
        for pair in tokens_in_window.windows(2) {
            let t1 = &transcript.tokens[pair[0]];
            let t2 = &transcript.tokens[pair[1]];
            if t1.speaker != t2.speaker {
                switches += 1;
            }
        }

        if switches > config.max_switches_per_10s {
            let first_idx = tokens_in_window[0];
            let last_idx = *tokens_in_window.last().unwrap();
            zones.push(ProblemZone {
                start_ms: transcript.tokens[first_idx].start_ms,
                end_ms: transcript.tokens[last_idx].end_ms,
                problem_type: ProblemType::SpeakerJitter,
                token_indices: tokens_in_window,
            });
        }

        window_start += window_ms / 2; // 50% overlap for detection
    }

    zones
}

/// Detect turns that are too short
fn detect_short_turns(
    transcript: &TokenizedTranscript,
    config: &ProblemZoneConfig,
) -> Vec<ProblemZone> {
    transcript
        .turns
        .iter()
        .filter(|turn| turn.duration_ms() < config.min_turn_duration_ms)
        .map(|turn| ProblemZone {
            start_ms: turn.start_ms,
            end_ms: turn.end_ms,
            problem_type: ProblemType::ShortTurn,
            token_indices: turn.token_indices.clone(),
        })
        .collect()
}

/// Detect regions adjacent to overlap
fn detect_overlap_adjacent(
    transcript: &TokenizedTranscript,
    config: &ProblemZoneConfig,
) -> Vec<ProblemZone> {
    let mut zones = Vec::new();

    // Find all overlap region tokens
    let overlap_times: Vec<(u64, u64)> = transcript
        .tokens
        .iter()
        .filter(|t| t.is_overlap_region)
        .map(|t| (t.start_ms, t.end_ms))
        .collect();

    if overlap_times.is_empty() {
        return zones;
    }

    // Find tokens within proximity of overlap regions
    for (overlap_start, overlap_end) in &overlap_times {
        let zone_start = overlap_start.saturating_sub(config.overlap_proximity_ms);
        let zone_end = overlap_end + config.overlap_proximity_ms;

        let affected: Vec<usize> = transcript
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                !t.is_overlap_region && t.end_ms >= zone_start && t.start_ms <= zone_end
            })
            .map(|(i, _)| i)
            .collect();

        if !affected.is_empty() {
            zones.push(ProblemZone {
                start_ms: zone_start,
                end_ms: zone_end,
                problem_type: ProblemType::OverlapAdjacent,
                token_indices: affected,
            });
        }
    }

    zones
}

/// Detect regions with low speaker confidence
fn detect_low_confidence(
    transcript: &TokenizedTranscript,
    config: &ProblemZoneConfig,
) -> Vec<ProblemZone> {
    let mut zones = Vec::new();
    let mut current_zone_tokens: Vec<usize> = Vec::new();

    for (i, token) in transcript.tokens.iter().enumerate() {
        if token.speaker_conf < config.min_speaker_confidence {
            current_zone_tokens.push(i);
        } else if !current_zone_tokens.is_empty() {
            // Close current zone
            let first = &transcript.tokens[current_zone_tokens[0]];
            let last = &transcript.tokens[*current_zone_tokens.last().unwrap()];
            zones.push(ProblemZone {
                start_ms: first.start_ms,
                end_ms: last.end_ms,
                problem_type: ProblemType::LowConfidence,
                token_indices: current_zone_tokens.clone(),
            });
            current_zone_tokens.clear();
        }
    }

    // Don't forget the last zone
    if !current_zone_tokens.is_empty() {
        let first = &transcript.tokens[current_zone_tokens[0]];
        let last = &transcript.tokens[*current_zone_tokens.last().unwrap()];
        zones.push(ProblemZone {
            start_ms: first.start_ms,
            end_ms: last.end_ms,
            problem_type: ProblemType::LowConfidence,
            token_indices: current_zone_tokens,
        });
    }

    zones
}

/// Build processing windows from the transcript
fn build_windows(
    transcript: &TokenizedTranscript,
    config: &WindowConfig,
    problem_zones: &[ProblemZone],
) -> WindowSet {
    let mut windows = Vec::new();

    if transcript.tokens.is_empty() {
        return WindowSet {
            windows,
            problem_window_indices: vec![],
        };
    }

    let total_duration = transcript.duration_ms();
    let start_offset = transcript.tokens[0].start_ms;
    let mut window_start = start_offset;
    let mut window_id = 0u64;

    while window_start < start_offset + total_duration {
        let window_end = window_start + config.window_size_ms;

        // Find tokens in this window
        let token_indices: Vec<usize> = transcript
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.start_ms >= window_start && t.start_ms < window_end)
            .map(|(i, _)| i)
            .collect();

        // Find anchor prefix tokens
        let anchor_start = window_start.saturating_sub(config.anchor_size_ms);
        let anchor_prefix_indices: Vec<usize> = transcript
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.start_ms >= anchor_start && t.start_ms < window_start)
            .map(|(i, _)| i)
            .collect();

        // Find anchor suffix tokens
        let anchor_end = window_end + config.anchor_size_ms;
        let anchor_suffix_indices: Vec<usize> = transcript
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.start_ms >= window_end && t.start_ms < anchor_end)
            .map(|(i, _)| i)
            .collect();

        // Check if window intersects any problem zone
        let (is_problem_zone, problem_types) =
            check_problem_intersection(window_start, window_end, problem_zones);

        if !token_indices.is_empty() {
            windows.push(Window {
                window_id: format!("w_{}", window_id),
                start_ms: window_start,
                end_ms: window_end,
                token_indices,
                anchor_prefix_indices,
                anchor_suffix_indices,
                is_problem_zone,
                problem_types,
            });
            window_id += 1;
        }

        window_start += config.stride_ms;
    }

    // Identify problem windows
    let problem_window_indices: Vec<usize> = if config.filter_problem_zones {
        windows
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_problem_zone)
            .map(|(i, _)| i)
            .collect()
    } else {
        (0..windows.len()).collect()
    };

    WindowSet {
        windows,
        problem_window_indices,
    }
}

/// Check if a window intersects any problem zone
fn check_problem_intersection(
    window_start: u64,
    window_end: u64,
    problem_zones: &[ProblemZone],
) -> (bool, Vec<ProblemType>) {
    let mut types = Vec::new();

    for zone in problem_zones {
        // Check for overlap
        if zone.start_ms < window_end && zone.end_ms > window_start {
            if !types.contains(&zone.problem_type) {
                types.push(zone.problem_type);
            }
        }
    }

    (!types.is_empty(), types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::parse_deepgram_json;

    #[test]
    fn test_detect_short_turns() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.2, "confidence": 0.95, "speaker": 0},
                            {"word": "yes", "start": 0.3, "end": 0.4, "confidence": 0.95, "speaker": 1},
                            {"word": "how", "start": 0.5, "end": 0.7, "confidence": 0.95, "speaker": 0}
                        ]
                    }]
                }]
            }
        }"#;

        let mut transcript = parse_deepgram_json(json).unwrap();
        let config = ProblemZoneConfig::default();
        let result = normalize(&mut transcript, &WindowConfig::default(), &config);

        // The middle turn (just "yes") should be flagged as short
        let short_turn_zones: Vec<_> = result
            .problem_zones
            .iter()
            .filter(|z| z.problem_type == ProblemType::ShortTurn)
            .collect();

        assert!(!short_turn_zones.is_empty());
    }
}
