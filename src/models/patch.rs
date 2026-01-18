use serde::{Deserialize, Serialize};

/// Reason codes for token relabeling - restricted enum to reduce hallucination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// Short turn caused by speaker jitter
    JitterShortTurn,
    /// Token near overlap boundary
    OverlapBoundary,
    /// Lexical continuity with surrounding tokens
    LexicalContinuity,
    /// Question/answer dialogue pairing
    DialoguePairing,
    /// Backchannel attribution (e.g., "yeah", "uh-huh")
    BackchannelAttribution,
    /// Explicitly keeping the token unchanged
    DoNotChange,
}

/// A single token relabeling operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRelabel {
    /// ID of the token to relabel
    pub token_id: String,
    /// New speaker ID to assign
    pub new_speaker: u32,
    /// Reason for the change
    pub reason: ReasonCode,
}

/// Type of turn edit operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEditType {
    /// Merge two consecutive turns into one
    MergeTurns,
    /// Split a turn at a specific token
    SplitTurn,
}

/// A turn edit operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEdit {
    /// Type of edit
    #[serde(rename = "type")]
    pub edit_type: TurnEditType,
    /// For merge: the first turn ID; for split: the turn to split
    pub turn_id: String,
    /// For merge: the second turn ID to merge into the first
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_turn_id: Option<String>,
    /// For split: the token ID where the split should occur
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_at_token_id: Option<String>,
    /// Reason for the edit
    pub reason: ReasonCode,
}

/// Notes from the LLM about the patch
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchNotes {
    /// Tokens the LLM was uncertain about
    #[serde(default)]
    pub uncertain_tokens: Vec<String>,
    /// Summary of changes made
    #[serde(default)]
    pub summary: String,
}

/// Complete patch output from the LLM for a single window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowPatch {
    /// ID of the window this patch applies to
    pub window_id: String,
    /// Token relabeling operations
    #[serde(default)]
    pub token_relabels: Vec<TokenRelabel>,
    /// Turn edit operations
    #[serde(default)]
    pub turn_edits: Vec<TurnEdit>,
    /// Self-reported violations (if non-empty, reject the patch)
    #[serde(default)]
    pub violations: Vec<String>,
    /// Notes and metadata
    #[serde(default)]
    pub notes: PatchNotes,
}

impl WindowPatch {
    /// Check if the patch has any self-reported violations
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Count the number of relabel operations
    pub fn relabel_count(&self) -> usize {
        self.token_relabels.len()
    }

    /// Check if the patch is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.token_relabels.is_empty() && self.turn_edits.is_empty()
    }
}

/// Validation result for a patch
#[derive(Debug, Clone)]
pub struct PatchValidation {
    /// Whether the patch is valid
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// Edit budget usage (0.0 - 1.0)
    pub edit_budget_used: f64,
}

impl PatchValidation {
    pub fn valid(edit_budget_used: f64) -> Self {
        Self {
            is_valid: true,
            errors: vec![],
            edit_budget_used,
        }
    }

    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            edit_budget_used: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_window_patch() {
        let json = r#"{
            "window_id": "w_0123",
            "token_relabels": [
                {"token_id": "t_1829", "new_speaker": 1, "reason": "lexical_continuity"}
            ],
            "turn_edits": [
                {"type": "merge_turns", "turn_id": "turn_88", "to_turn_id": "turn_89", "reason": "jitter_short_turn"}
            ],
            "violations": [],
            "notes": {
                "uncertain_tokens": ["t_1831"],
                "summary": "Moved short backchannel"
            }
        }"#;

        let patch: WindowPatch = serde_json::from_str(json).unwrap();

        assert_eq!(patch.window_id, "w_0123");
        assert_eq!(patch.token_relabels.len(), 1);
        assert_eq!(patch.token_relabels[0].new_speaker, 1);
        assert_eq!(patch.token_relabels[0].reason, ReasonCode::LexicalContinuity);
        assert!(!patch.has_violations());
    }
}
