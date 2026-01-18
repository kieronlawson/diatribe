# Diatribe Architecture

## Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stage 0       │    │   Stage 1       │    │   Stage 2       │    │   Stage 3       │
│   Normalize     │───▶│   LLM Edit      │───▶│   Reconcile     │───▶│   Render        │
│                 │    │                 │    │                 │    │                 │
│ - Parse input   │    │ - Window tokens │    │ - Merge patches │    │ - Machine JSON  │
│ - Detect zones  │    │ - Call Claude   │    │ - Weighted vote │    │ - Human text    │
│ - Build windows │    │ - Validate      │    │ - Constraints   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Flow

### Input (Deepgram Format)
```json
{
  "results": {
    "channels": [{
      "alternatives": [{
        "words": [
          {"word": "hello", "start": 0.5, "end": 0.8, "confidence": 0.95, "speaker": 0, "speaker_confidence": 0.85}
        ]
      }]
    }]
  }
}
```

### Internal Token Representation
```rust
Token {
    token_id: "uuid-v4",
    word: "hello",
    start_ms: 500,
    end_ms: 800,
    speaker: 0,
    speaker_conf: 0.85,
    transcription_conf: 0.95,
    is_overlap_region: false,
    segment_id: "seg_0",
    turn_id: "turn_0"
}
```

### Window Structure
```rust
Window {
    window_id: "w_0",
    start_ms: 0,
    end_ms: 45000,
    tokens: Vec<Token>,
    anchor_prefix: Vec<Token>,  // Read-only context before
    anchor_suffix: Vec<Token>,  // Read-only context after
    is_problem_zone: true
}
```

### LLM Patch Output
```rust
WindowPatch {
    window_id: "w_0",
    token_relabels: Vec<TokenRelabel>,
    turn_edits: Vec<TurnEdit>,
    violations: Vec<String>,
    notes: PatchNotes
}
```

## Module Structure

### `models/`
- `deepgram.rs` - Deepgram input format types
- `token.rs` - Internal token representation
- `window.rs` - Processing window with anchors
- `patch.rs` - LLM output patch types

### `stages/`
- `stage0_normalize.rs` - Parse, detect problem zones, build windows
- `stage1_llm_edit.rs` - LLM relabeling orchestration
- `stage2_reconcile.rs` - Merge overlapping window patches
- `stage3_render.rs` - Generate output formats

### `heuristics/`
- `micro_turns.rs` - Collapse <300ms turns
- `backchannels.rs` - Handle single-word acknowledgements
- `floor_holding.rs` - Track speaker floor dominance

### `llm/`
- `client.rs` - Anthropic API client
- `prompts.rs` - Prompt construction
- `validation.rs` - Patch validation

### `io/`
- `input.rs` - Parse Deepgram JSON
- `output.rs` - Write machine/human transcripts

## Problem Zone Detection

Windows are only processed by the LLM if they intersect "problem zones":

1. **Speaker Jitter**: >3 speaker switches in 10s
2. **Short Turns**: Any turn <800ms
3. **Overlap Adjacent**: Within 2s of detected overlap
4. **Low Confidence**: Average speaker_confidence <0.6

## Reconciliation Strategy

When multiple windows produce different labels for the same token:

```
final_label = argmax(sum(
    vote_weight[window] * indicator(window_label == candidate)
))

where vote_weight = llm_confidence * window_quality * proximity_to_center
```

Additional constraints:
- Minimum turn duration: 700ms
- Maximum switches/second: 2.0
- Stable spans protected unless multiple windows agree
