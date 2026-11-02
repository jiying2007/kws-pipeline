# Offline three-ASR batch screening of authorized voice-clone campaigns

**Authority: diagnostic-only.** This program actually invokes three local
recognizer batch *providers* when explicitly requested, but cannot prove that
a provider executable implemented its advertised model or transcribed speech
correctly. It NEVER creates reviewed labels, authenticated consensus receipts,
real-human gold, or production qualification.

Entry point: `tools/voice_clone_three_asr_batch.py`.
It consumes a successful private
`tools/authorized_voice_clone_batch.py` campaign, retaining its
source/rights/reference identity and actual generated WAV digests.

## What it does automatically

1. Verify the clone campaign/manifest and each real WAV/decoded PCM via
   `voice_clone_effect_audit.py`, including cross-speaker duplication.
2. Build a private, canonical **audio-only batch**: each record carries only
   `item_id`, absolute `audio` path, `file_sha256`, and `pcm_sha256`.
   Crucially, requested target text and kind are never passed in model input.
3. Verify three separately declared and *hash-pinned* local recognizer models,
   executables, adapter assets, and a fixed Chinese transcription setting with
   no prompt/hotwords. Distinct model IDs/families/weight SHA are required.
   Different strings alone do not prove model independence.
4. Run **one batch command per model** (avoids weight reload per utterance).
   Each adapter reads the WAV-only JSONL and creates its output JSONL.
   An explicit 1–256 WAV budget and per-recognizer 1–1200 second limit apply.
5. Require each result to match every source `item_id`/file SHA exactly,
   with `status=complete` and its actual `raw_text` (possibly blank).
   Retain raw outputs, provider identity, exact bytes and one-shot attempt
   receipts. Unknown/partly failed invocation **cannot be retried silently**.
6. Recheck every source WAV, build a private `screen.jsonl`, invoke the
   effect auditor, and emit a summary with all promotion flags false.

The screen report can be used to identify failed positive pronunciations,
negative text confusion, and error families for the next *development* batch.
A blank ASR transcript is unknown; three blanks do not certify silence.

## Private plan (schema example, not a runnable model configuration)

```json
{
  "schema_version": 1,
  "policy": "offline-authorized-clone-three-asr-batch-screen-v1",
  "max_recordings": 24,
  "recognizers": [
    {
      "model_family": "independent_family_a",
      "model_id": "asr_a",
      "model_revision": "pinned_rev_1",
      "decode_config": {"language":"zh", "task":"transcribe", "prompt":"", "hotwords":[]},
      "executable": {"path": "/private/python", "sha256": "64-lowercase-hex-for-actual-file"},
      "assets": [
        {"role":"model", "path": "/private/models/model_a", "sha256":"64-lowercase-hex-for-actual-weights"},
        {"role":"adapter", "path": "/private/adapters/a.py", "sha256":"64-lowercase-hex-for-actual-file"}
      ],
      "argv_template": [
        "{executable}", "{asset:adapter}", "{input}", "{output}", "{asset:model}"
      ],
      "timeout_seconds": 600
    }
  ]
}
```

The runnable plan requires **exactly three** recognizer records of the same
shape, with genuinely distinct pinned model assets and honest identities.
All example hashes are placeholders and deliberately fail closed. The adapter
must accept positional `input`, `output`, and model path as shown, or use
another direct-argument template pinned in the plan. It must produce one JSONL
per input WAV, **exactly**:

```json
{"item_id":"id-from-input","file_sha256":"same-actual-input-sha","raw_text":"what-ASR-really-heard","status":"complete"}
```

A real adapter may use sherpa-onnx, whisper, SenseVoice, or other supported
offline recognizers, but it must not replace independent evidence by renaming
one model three times. Pin model weights, source, native decode options,
executable, and actual captures before considering authenticated admission.

## Commands

```bash
# Read-only preflight, no ASR process and no output root created.
python3 tools/voice_clone_three_asr_batch.py \
  --plan /restricted/asr/three-model-plan.json \
  --campaign-root /restricted/clone-output \
  --output-root /restricted/asr-screen-output

# Explicit bounded provider execution, never in normal CI.
python3 tools/voice_clone_three_asr_batch.py \
  --plan /restricted/asr/three-model-plan.json \
  --campaign-root /restricted/clone-output \
  --output-root /restricted/asr-screen-output \
  --execute
```

The plan and output must be outside the public source tree. The caller must
run trusted offline model providers in a **network-isolated environment** with
no private tokens. The tool strips common credential environment variables,
but cannot sandbox a malicious provider executable on its own.

## Exact evidence and limits

- `audio-only-batch.jsonl`: immutable input (no target text).
- `attempts/<model_family>.intent.json`: durable atomic reservation *before*
  invocation; `attempts/<model_family>.receipt.json`: success readback.
- `raw/<model_family>.jsonl`: original per-model output, retained private.
- `screen.jsonl`: aligned three-model diagnostics compatible with
  `voice_clone_effect_audit.py --asr-screen`.
- `summary.json`: exact batch/model/raw/screen identities and measured
  diagnostic counts, all release/training authorizations false.

Do not commit sensitive references, WAV, prompts, rights evidence, model output
or private transcripts to GitHub. The CI tests use three harmless fake offline
ASR adapter executions with invented audio. They only establish protocol
integrity, not real recognizer quality.

**Next required admission step:** actual source-verifying recognizer execution
and `asr_source_resolver` / `speech_asr_consensus` / training admission
contracts; a generated `screen.jsonl` never passes those gates by itself.
Then use independent real-voice/AFE development followed by sealed
Phase A/B/C acceptance. The production-model objective remains unchanged.
