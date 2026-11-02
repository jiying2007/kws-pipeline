# Authorized offline voice-clone batch (research-only)

This is a **private, bounded generation entry point**, not a shipping model, proof
of speaker independence, or an automatic route into model training. It addresses
large speaker-roster production of Mandarin wake/near-miss/ordinary-negative
*generation intents* without calling a model during ordinary CI.

Entry point: \`tools/authorized_voice_clone_batch.py\`.
Focused contract tests: \`tests/test_authorized_voice_cloning.py\`.

## Requirements and scope

- A stable, pseudonymous **original** \`speaker_id\` and \`source_family_id\`
  must accompany each verified reference WAV. A clone, another random seed,
  another text or another acoustic augmentation is **not another real speaker**.
  Preserve transitive source-family connections in any later split audit.
- Reference WAV: 16-kHz mono PCM16, 3–30 seconds, exact SHA-256. Do not copy
  source or reference speech into the public repository.
- Rights evidence is a **source-specific, retained local JSON document** with
  explicit \`voice_cloning_permitted=true\` and
  \`synthetic_kws_training_permitted=true\`, a source-family identity, a
  \`covered_speaker_ids\` list that names the speaker, and a concrete
  \`authorization_basis\`. The JSON and reference WAV are independently hashed.
  Public availability, CC0, generic dataset download terms, a model's license
  and an AI-generated statement do not by themselves prove a living person's
  voice may be cloned. A data steward must verify the source-specific rights
  and permitted purposes **once per source contract**, not once per generated
  clip; the program checks record integrity, not legal ownership or consent.
- A local **trusted, pinned, offline** provider executable and all asset files
  are required. No paid API is needed. The provider is responsible for actual
  inference and for using its per-speaker \`{prompt_cache}\` directory to cache
  speaker embeddings/prompts. This wrapper does not supply a model, download
  arbitrary datasets, synthesize speech by itself, or prevent a malicious
  executable from connecting to the network. Run the provider in a
  network-isolated environment without private tokens.
- The roster is capped at 512 references; one campaign at 32,768 attempts;
  128 attempts per reference; 1–4 speaker-level workers; 1–8 declared seeds;
  3–64 declared utterance texts. Negative intent text cannot literally contain
  either full positive trigger. This lexical check is not acoustic verification.
- Only the private **research training-source generation** lane is supported.
  All emitted rows are unreviewed synthetic candidates with
  \`training_admitted=false\` and \`qualification_allowed=false\`. ASR
  agreement, pronunciation checks and the existing human-reviewed admission
  pathway are separate requirements. Real human/final-AFE heldout and physical
  SSC305 evaluation are never manufactured from cloned samples.

## Private input example

Prepare the following files **outside the Git checkout**. The example hash
strings are placeholders and deliberately cannot pass preflight; use exact
SHA-256 of the real local files. No voice/reference/body is committed here.

\`rights-family-A.json\`:

\`\`\`json
{
  "schema_version": 1,
  "source_family_id": "family-A",
  "voice_cloning_permitted": true,
  "synthetic_kws_training_permitted": true,
  "covered_speaker_ids": ["anonymous-1"],
  "authorization_basis": "verified written source-specific cloning and KWS training permission"
}
\`\`\`

\`plan.json\`:

\`\`\`json
{
  "schema_version": 1,
  "policy": "authorized-offline-voice-clone-v1",
  "campaign_id": "authorized-zh-batch-001",
  "max_attempts": 6,
  "provider": {
    "kind": "offline-voice-clone",
    "executable": {"path": "/private/bin/offline-tts", "sha256": "<64-lowercase-hex>"},
    "assets": [
      {"role": "model", "path": "/private/models/model-file", "sha256": "<64-lowercase-hex>"}
    ],
    "argv_template": [
      "{executable}", "--model", "{asset:model}", "--ref", "{reference}",
      "--ref-text", "{reference_text}", "--text", "{text}", "--seed", "{seed}",
      "--speaker-cache", "{prompt_cache}", "--output", "{output}"
    ],
    "timeout_seconds": 600
  },
  "speakers": [
    {
      "speaker_id": "anonymous-1",
      "source_family_id": "family-A",
      "reference": {"path": "ref-1.wav", "sha256": "<64-lowercase-hex>"},
      "rights_evidence": {"path": "rights-family-A.json", "sha256": "<64-lowercase-hex>"},
      "reference_text": "reference transcript if available"
    }
  ],
  "phrases": [
    {"id": "wake-1", "intent": "wake", "text": "你好小窝"},
    {"id": "near-1", "intent": "near_miss", "text": "你好小屋"},
    {"id": "negative-1", "intent": "ordinary_negative", "text": "今天的天气不错"}
  ],
  "seeds": [0, 1]
}
\`\`\`

Command template placeholders are expanded as **argv only, never through a
shell**. The first token must be \`{executable}\`. Required placeholders:
\`{reference}\`, \`{text}\`, \`{output}\`, \`{prompt_cache}\`, and \`{seed}\`.
Optional \`{reference_text}\` is available. Asset placeholders are
\`{asset:role}\`. All template entries are source-controlled/authorized
configuration; do not accept arbitrary command templates from downloaded
dataset metadata.

## Dry-run and actual run

\`\`\`sh
python3 tools/authorized_voice_clone_batch.py \
  --plan /private/kws-clone-campaign/plan.json \
  --output-root /private/kws-clone-output

python3 tools/authorized_voice_clone_batch.py \
  --plan /private/kws-clone-campaign/plan.json \
  --output-root /private/kws-clone-output \
  --execute --workers 2

python3 -S -B tests/test_authorized_voice_cloning.py
\`\`\`

Dry-run validates rights, SHA-256, provider assets, PCM and the full bounded
plan. It launches **zero** model calls and does not write the output directory.
Execution persists a campaign identity, a per-attempt intent **before** the
provider call, a verified WAV hash, and a success receipt. Source and model
assets should be immutable during the run. Known completed receipts are
rechecked without calling the model. A claimed attempt with no success receipt
is **unknown/consumed**, not eligible for automatic retry. To recover it,
inspect the private provider logs/outputs and make a new separately authorized
campaign only if justified; never silently restart an exhausted slot.

On success, \`manifest.jsonl\` and \`summary.json\` are immutable for that
campaign/output root. Private WAV, reference data, rights proof, prompts,
generated speech and any sensitive transcript remain off GitHub. The manifest
carries original speaker/source family, reference hash, derivation family, model
identity hash, generation-intent label and PCM/file hashes. **It is not an
\`admit-reviewed\` manifest**, a review receipt, a pronunciation assertion,
or independent qualification evidence.

For real production-model progression, screen voice-cloned output with
source-bound ASR/label checks, admit only reviewed utterances with actual
transcripts, isolate all descendants of one original person/source family
across train/development/calibration/test, then run strictly fresh acoustic
comparison and FAR/FRR evaluation. Final-AFE real-human and SSC305 target-board
acceptance remain separate mandatory product gates.

All repository tests for this entry point use a harmless local fake WAV provider;
a passing test **does not** mean any Qwen/CosyVoice model was run or any
licensed speaker source was imported.


## Typed challenge families (optional, backwards-compatible)

The campaign's individual phrase objects may now include an explicit
\`"challenge_family"\` *generation-intent taxonomy*:

| \`intent\` | Allowed \`challenge_family\` values |
| --- | --- |
| \`wake\` | \`exact_requested\` |
| \`near_miss\` | \`homophone_requested\`, \`partial_prefix_requested\`, \`partial_suffix_requested\`, \`cross_sentence_requested\`, \`reordered_requested\`, \`other_near_miss_requested\` |
| \`ordinary_negative\` | \`ordinary_conversation_requested\`, \`media_script_requested\`, \`other_nonwake_requested\` |

These labels describe what the TTS command **asked** for, not what the
generated waveform contains or whether it caused a false accept.
The exact chosen subtype is included in the immutable campaign fingerprint,
attempt intent and completed WAV receipt. Replay of a mismatched subtype fails.
Legacy campaigns without a subtype retain **exactly their old fingerprints and
receipts** and appear under \`unclassified_legacy\` in the effect-screen report.

The optional subtype lets the offline
[\`voice_clone_effect_audit.py\`](../tools/voice_clone_effect_audit.py) report
\`screen_by_challenge_family\` alongside text and speaker-family yield. It never
creates a true negative label, authenticated ASR evidence, independent
speaker count, or qualification. The current v1 generator still requires wake
text to normalize to one complete product keyword and will not silently
treat partial keyword phrases as positive. Contextual positives, acoustic
pause assertions and actual error labels need their own reviewed source
contracts before extending this taxonomy.

See [private three-ASR batch screening](VOICE_CLONE_THREE_ASR_SCREEN.md) for an
explicitly opted-in offline pipeline that executes three pinned recognizer batch
providers and preserves source hashes without passing target/requested text.

Empty ASR transcripts are allowed as *screen observations* and produce
`ASR_ALL_EMPTY_UNVERIFIED` or `ASR_PARTIAL_EMPTY_OR_DISAGREEMENT` diagnostic
outcomes; three empty transcripts cannot prove that a WAV is truly silent.
They never authenticate labels or qualify a negative source for training.
