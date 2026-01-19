# Diatribe Requirements Specification

## LLM Component Goal

Input: a diarized transcript where every word token already has (start_ms, end_ms, speaker_id, confidence). Overlap may exist (multiple speakers active), but your ASR produced a single word stream; you want "assign each emitted word to the right speaker" and a readable transcript. The LLM should not invent, delete, or paraphrase words. It can only:
- (a) relabel tokens
- (b) adjust turn boundaries
- (c) add formatting (punctuation/casing) as a separate derived view

## Architecture

Run the LLM as a constrained local editor on sliding windows, produce a structured diff, then reconcile globally.

### Stage 0: Pre-LLM normalization (deterministic)

- Convert transcript into tokens: `t[i] = {token_id, word, start_ms, end_ms, speaker, speaker_conf, is_overlap_region?, segment_id}`
- Detect "problem zones" so you only spend LLM where needed:
  1. **Speaker jitter**: high speaker-switch rate (e.g., >N switches per 10s)
  2. **Very short turns**: runs <400–800 ms
  3. **Overlap-adjacent regions**: within ±2s of any overlap region
  4. **Low confidence spans**: avg speaker_conf below threshold
- Build windows (e.g., 45s with 15s stride). Only schedule windows intersecting problem zones.

### Stage 1: LLM relabel + turn cleanup (edit-only)

For each window, provide:
- Token list with timestamps and current speaker labels
- Constraints: max speakers = 4; only these speaker IDs allowed; no word changes; timestamps immutable
- Local context outside window: include a short "anchor" prefix and suffix (e.g., 3–5s) marked read-only to reduce boundary artefacts

LLM outputs a JSON patch:
- `token_relabels`: `[{token_id, new_speaker_id, rationale_code}]`
- `turn_splits/merges`: `[{type, start_token_id, end_token_id}]`
- optional: per-token "keep_as_is" to be explicit

No changes to word text. No insertions/deletions.

### Stage 2: Global reconciliation (deterministic)

Because windows overlap, you'll get conflicting edits:
- For each token_id, collect candidate labels from all windows that touched it
- Choose final label by weighted vote:
  ```
  weight = LLM_confidence * window_quality * proximity_to_window_center
  ```
  plus a penalty if the change increases rapid flips
- Enforce constraints:
  - Minimum turn duration (e.g., 700 ms) unless it's a backchannel list
  - Maximum speaker switches per second (soft constraint)
  - Do not relabel tokens in "stable spans" (high conf, long runs) unless multiple windows agree

### Optional: Speaker Identification

After reconciliation, an optional speaker identification stage can map anonymous speaker IDs to participant names:

- Requires a participant list (names with optional hints)
- Uses a separate LLM call with tool use for structured output
- Does not affect core speaker attribution logic
- Returns confidence scores and evidence for each identification
- Only applies identifications above a configurable confidence threshold

### Stage 3: Rendering (derived view)

Produce two views:
1. **Canonical machine transcript**: tokens with final speaker_id and timestamps, optionally with speaker names and identification metadata
2. **Human transcript**: turns with punctuation/casing and paragraphing, using participant names when available (this can be a second LLM pass, but it must not alter words; only add punctuation/case and line breaks)

## Prompt Design

### System (non-negotiable)

```
You are editing a diarized transcript. You MUST NOT add, remove, or change any words. You MUST NOT change timestamps. You may only reassign speaker labels for existing tokens and adjust turn boundaries. Speaker IDs allowed: S1, S2, S3, S4. Output MUST be valid JSON matching the provided schema. If uncertain, do not change anything.
```

### User payload structure (per window)

1. **Rules (short)**
   - Allowed operations, forbidden operations
   - Objective function (minimize speaker flips, maximize local coherence per speaker, keep changes minimal)
   - "Do not touch" anchors (explicit token_id ranges)

2. **Data**
   - speakers: list of speaker IDs with short "voiceprint hints" derived deterministically (no LLM): typical words/phrases, avg turn length, avg speaking rate. (Optional but useful.)
   - tokens: array of objects:
     ```json
     {token_id, word, start_ms, end_ms, speaker, speaker_conf, overlap_flag, turn_id}
     ```

3. **Output schema**

Example schema:
```json
{
  "window_id": "w_0123",
  "token_relabels": [
    {"token_id": "t_1829", "new_speaker": "S2", "reason": "continuity"},
    ...
  ],
  "turn_edits": [
    {"type": "merge_turns", "from_turn_id": "turn_88", "to_turn_id": "turn_89", "reason": "jitter"},
    {"type": "split_turn", "turn_id": "turn_91", "split_at_token_id": "t_1904", "reason": "topic_shift"}
  ],
  "notes": {
    "uncertain_tokens": ["t_1831","t_1832"],
    "summary": "Moved short backchannel to S2; merged two jitter turns."
  }
}
```

## Constraint Tricks (Hallucination Risk Reduction)

### 1. "Edit distance budget"
Tell the model it has a max edit budget, e.g., "You may relabel at most 3% of tokens in this window. Prefer fewer." This forces conservatism.

### 2. "Reason codes only"
Restrict rationales to a small enum:
- `jitter_short_turn`
- `overlap_boundary`
- `lexical_continuity`
- `dialogue_pairing` (question/answer alignment)
- `backchannel_attribution`
- `do_not_change`

### 3. "Locality constraint"
Disallow moving words across speakers unless:
- token is within an overlap_flag span OR within X ms of a speaker change OR speaker_conf below Y

### 4. "Speaker consistency features"
Provide the LLM with computed per-speaker bag-of-words from earlier stable spans (e.g., names used, typical acknowledgements) and explicit prior: "S1 often says 'yep'; S3 never does" when it's true. Keep it strictly derived from your data to avoid bias.

## Deterministic Heuristics (Pre-LLM Cheap Wins)

Run before calling the LLM:
- **Collapse micro-turns**: if a turn <300 ms and surrounded by same speaker, relabel it automatically
- **Backchannel rule**: single-word acknowledgements in overlap-adjacent zones should default to the speaker who is not holding the floor, unless speaker_conf is high
- **"Floor holding" model**: maintain a short-term floor score per speaker; penalize flipping the floor for 1–2 tokens

Then run the LLM only when heuristics disagree or confidence is low.

## LLM Model Interaction Pattern

- Use function/tool calling (strict JSON schema) so you never parse free text
- Temperature low (0–0.2). You want determinism.
- Include a self-check field in JSON: `"violations": []` and require the model to list any rule it might have violated; if non-empty, reject output

## Post-LLM Validation (Must-Have)

Reject the patch if:
- Any token_id not in window
- Any new_speaker not in {S1..S4}
- Patch exceeds edit budget
- Patch changes any word string or timestamp (should be impossible if schema excludes it)
- Patch increases a simple cost function too much:
  ```
  cost = 5*(#speaker_switches) + 2*(#turns_under_700ms) - 1*(avg_per_speaker_lexical_coherence_gain)
  ```
  You can compute lexical coherence as cosine similarity of TF-IDF (or embeddings) per speaker before/after within the window.

## Deliverables

1. A "speaker-attributed token stream" with improved label stability and better overlap-boundary attribution
2. A "rendered transcript" that looks sensible, without compromising evidentiary integrity (words unchanged)

## Input Format

Deepgram diarization format. Word objects have:
- `word`: The recognized text
- `start`/`end`: Timestamp boundaries (in seconds)
- `confidence`: Transcription accuracy score (0-1)
- `speaker`: Numeric speaker identifier
- `speaker_confidence`: Reliability of speaker assignment (pre-recorded only)

## Configuration

- API key: `ANTHROPIC_API_KEY` environment variable
- Model: Claude (configurable)
- Temperature: 0-0.2 for determinism
