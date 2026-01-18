use serde::{Deserialize, Serialize};


/// Configuration for window generation
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Window size in milliseconds
    pub window_size_ms: u64,
    /// Stride between windows in milliseconds
    pub stride_ms: u64,
    /// Anchor context size in milliseconds (before and after window)
    pub anchor_size_ms: u64,
    /// Only process windows intersecting problem zones
    pub filter_problem_zones: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 45_000,  // 45 seconds
            stride_ms: 15_000,        // 15 seconds
            anchor_size_ms: 5_000,    // 5 seconds
            filter_problem_zones: true,
        }
    }
}

/// A processing window containing tokens and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    /// Unique identifier for this window
    pub window_id: String,
    /// Start timestamp in milliseconds
    pub start_ms: u64,
    /// End timestamp in milliseconds
    pub end_ms: u64,
    /// Indices of tokens in this window (into the main token array)
    pub token_indices: Vec<usize>,
    /// Indices of anchor prefix tokens (read-only context before window)
    pub anchor_prefix_indices: Vec<usize>,
    /// Indices of anchor suffix tokens (read-only context after window)
    pub anchor_suffix_indices: Vec<usize>,
    /// Whether this window intersects a problem zone
    pub is_problem_zone: bool,
    /// Problem zone types detected in this window
    pub problem_types: Vec<ProblemType>,
}

impl Window {
    /// Duration of this window in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Total number of tokens (excluding anchors)
    pub fn token_count(&self) -> usize {
        self.token_indices.len()
    }

    /// Check if a token index is in the editable region (not an anchor)
    pub fn is_editable(&self, token_index: usize) -> bool {
        self.token_indices.contains(&token_index)
    }

    /// Get the center timestamp of this window
    pub fn center_ms(&self) -> u64 {
        (self.start_ms + self.end_ms) / 2
    }

    /// Calculate proximity to window center (0.0 at edges, 1.0 at center)
    pub fn proximity_to_center(&self, timestamp_ms: u64) -> f64 {
        let center = self.center_ms() as f64;
        let half_duration = self.duration_ms() as f64 / 2.0;
        if half_duration == 0.0 {
            return 1.0;
        }
        let distance = (timestamp_ms as f64 - center).abs();
        (1.0 - distance / half_duration).max(0.0)
    }
}

/// Types of problem zones that trigger LLM processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemType {
    /// High speaker switch rate (>N switches per 10s)
    SpeakerJitter,
    /// Very short turns (<400-800ms)
    ShortTurn,
    /// Within ±2s of overlap region
    OverlapAdjacent,
    /// Low speaker confidence
    LowConfidence,
}

/// Configuration for problem zone detection
#[derive(Debug, Clone)]
pub struct ProblemZoneConfig {
    /// Maximum speaker switches per 10 seconds before flagging jitter
    pub max_switches_per_10s: u32,
    /// Minimum turn duration in ms before flagging as short
    pub min_turn_duration_ms: u64,
    /// Proximity to overlap in ms to flag as overlap-adjacent
    pub overlap_proximity_ms: u64,
    /// Minimum speaker confidence threshold
    pub min_speaker_confidence: f64,
}

impl Default for ProblemZoneConfig {
    fn default() -> Self {
        Self {
            max_switches_per_10s: 3,
            min_turn_duration_ms: 800,
            overlap_proximity_ms: 2_000,
            min_speaker_confidence: 0.6,
        }
    }
}

/// Result of window generation
#[derive(Debug, Clone)]
pub struct WindowSet {
    /// All generated windows
    pub windows: Vec<Window>,
    /// Windows that should be processed by the LLM (intersect problem zones)
    pub problem_window_indices: Vec<usize>,
}

impl WindowSet {
    /// Get windows that need LLM processing
    pub fn problem_windows(&self) -> impl Iterator<Item = &Window> {
        self.problem_window_indices
            .iter()
            .filter_map(|&i| self.windows.get(i))
    }

    /// Total number of windows
    pub fn total_windows(&self) -> usize {
        self.windows.len()
    }

    /// Number of windows needing LLM processing
    pub fn problem_window_count(&self) -> usize {
        self.problem_window_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_proximity_to_center() {
        let window = Window {
            window_id: "w_0".to_string(),
            start_ms: 0,
            end_ms: 10_000,
            token_indices: vec![],
            anchor_prefix_indices: vec![],
            anchor_suffix_indices: vec![],
            is_problem_zone: true,
            problem_types: vec![],
        };

        assert!((window.proximity_to_center(5_000) - 1.0).abs() < 0.001);
        assert!((window.proximity_to_center(0) - 0.0).abs() < 0.001);
        assert!((window.proximity_to_center(10_000) - 0.0).abs() < 0.001);
        assert!((window.proximity_to_center(2_500) - 0.5).abs() < 0.001);
    }
}
