# diatribe

A Rust-based transcript diarization improvement pipeline using Claude as the LLM for constrained speaker attribution correction.

## Overview

Diatribe takes diarized transcripts (where every word has timestamps and speaker IDs) and improves speaker attribution accuracy without modifying the words themselves. It uses a multi-stage pipeline:

1. **Stage 0 (Normalize)**: Parse input, detect problem zones, build processing windows
2. **Heuristics**: Apply deterministic fixes (micro-turn collapse, backchannel rules, floor-holding)
3. **Stage 1 (LLM Edit)**: Use Claude to relabel tokens in problem zones
4. **Stage 2 (Reconcile)**: Merge overlapping window patches with weighted voting
5. **(Optional) Speaker ID**: Identify speakers by name from participant list
6. **Stage 3 (Render)**: Output machine JSON and human-readable transcripts

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

# With speaker identification (comma-separated names)
diatribe process \
  --input transcript.json \
  --output corrected.json \
  --human-readable output.txt \
  --participants "Alice Chen,Bob Smith,Carol Davis"

# With participants file (supports hints)
diatribe process \
  --input transcript.json \
  --output corrected.json \
  --participants-file participants.json \
  --speaker-id-confidence 0.8

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
  --participants "Alice Chen,Bob Smith" \
  --speaker-id-confidence 0.7 \
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

## Speaker Identification

When participants are provided (via `--participants` or `--participants-file`), diatribe attempts to match anonymous speaker IDs (Speaker 0, Speaker 1, etc.) to actual participant names using LLM analysis.

### How It Works

1. Extracts representative excerpts from each speaker's turns
2. Sends excerpts to Claude along with the participant list
3. Returns confidence scores and evidence for each identification

### Confidence Threshold

Use `--speaker-id-confidence` (default: 0.7) to set the minimum confidence required. Identifications below this threshold will not be applied to the output.

### Participants File Format

```json
[
  {"name": "Alice Chen", "hints": "Project manager, often asks about timelines"},
  {"name": "Bob Smith", "hints": "Engineer, uses technical terminology"},
  {"name": "Carol Davis"}
]
```

The `hints` field is optional but can help improve identification accuracy by providing context about each participant.

## Output Formats

### Machine Transcript (JSON)

Contains tokens with final speaker assignments, turn boundaries, and metadata about changes made. When speaker identification is enabled, includes `speaker_identifications` array and `speaker_name` fields:

```json
{
  "tokens": [...],
  "turns": [
    {
      "turn_id": "turn_0",
      "speaker": 0,
      "speaker_name": "Alice Chen",
      "start_ms": 500,
      "end_ms": 2100,
      "word_count": 6
    }
  ],
  "speaker_identifications": [
    {
      "speaker_id": 0,
      "identified_as": "Alice Chen",
      "confidence": 0.92,
      "evidence": ["Introduced self as Alice", "Discussed project management tasks"]
    },
    {
      "speaker_id": 1,
      "identified_as": "Bob Smith",
      "confidence": 0.85,
      "evidence": ["Referenced code architecture", "Used technical terminology"]
    }
  ]
}
```

### Human Transcript (Text)

Formatted text with speaker labels and timestamps. When speaker identification is enabled, uses participant names instead of generic speaker IDs:

```
[00:00.500] Alice Chen:
Hello world, how are you today?

[00:02.100] Bob Smith:
I'm doing great, thanks for asking.
```

Without speaker identification:

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
