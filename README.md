# diatribe

A Rust-based transcript diarization improvement pipeline using Claude as the LLM for constrained speaker attribution correction.

## Overview

Diatribe takes diarized transcripts (where every word has timestamps and speaker IDs) and improves speaker attribution accuracy without modifying the words themselves. It uses a multi-stage pipeline:

1. **Stage 0 (Normalize)**: Parse input, detect problem zones, build processing windows
2. **Heuristics**: Apply deterministic fixes (micro-turn collapse, backchannel rules, floor-holding)
3. **Stage 1 (LLM Edit)**: Use Claude to relabel tokens in problem zones
4. **Stage 2 (Reconcile)**: Merge overlapping window patches with weighted voting
5. **Stage 3 (Render)**: Output machine JSON and human-readable transcripts

## Installation

Requires Rust toolchain. Install via:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build the project:

```bash
cargo build --release
```

## Usage

### Process a transcript

```bash
# Basic usage
diatribe process --input transcript.json --output corrected.json

# With human-readable output
diatribe process \
  --input transcript.json \
  --output corrected.json \
  --human-readable output.txt

# Heuristics only (no LLM)
diatribe process \
  --input transcript.json \
  --output corrected.json \
  --heuristics-only

# Full options
diatribe process \
  --input transcript.json \
  --output corrected.json \
  --human-readable output.txt \
  --max-speakers 4 \
  --edit-budget 3.0 \
  --window-size-ms 45000 \
  --window-stride-ms 15000 \
  --min-turn-ms 700 \
  --verbose
```

### Analyze a transcript

```bash
diatribe analyze --input transcript.json
```

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

## Input Format

Accepts Deepgram JSON format with diarization:

```json
{
  "results": {
    "channels": [{
      "alternatives": [{
        "words": [
          {
            "word": "hello",
            "start": 0.5,
            "end": 0.8,
            "confidence": 0.95,
            "speaker": 0,
            "speaker_confidence": 0.85
          }
        ]
      }]
    }]
  }
}
```

## Output Formats

### Machine Transcript (JSON)

Contains tokens with final speaker assignments, turn boundaries, and metadata about changes made.

### Human Transcript (Text)

Formatted text with speaker labels and timestamps:

```
[00:00.500] Speaker 0:
Hello world, how are you today?

[00:02.100] Speaker 1:
I'm doing great, thanks for asking.
```

## Key Constraints

The LLM operates under strict constraints to preserve evidentiary integrity:

- **No word changes**: Words are immutable
- **No timestamp changes**: Timestamps are immutable
- **Edit budget**: Maximum 3% of tokens can be relabeled per window
- **Reason codes**: Changes must use predefined reason codes
- **Self-validation**: LLM reports any rule violations

## License

MIT
