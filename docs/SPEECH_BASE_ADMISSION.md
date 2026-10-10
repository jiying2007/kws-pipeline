# Speech base admission

Future real-speech/real-TTS CTC bases require supplied actual-audio review. A
requested utterance, deterministic generation, or file checksum alone does not
establish what was spoken. Existing corpus and release identities are unchanged.

## Three explicit lanes

- `reviewed-real-v1` (default): new CTC admission after the complete checks below.
  "Real" describes the audio-bearing training lane; TTS is still synthetic and
  is not human qualification evidence.
- `synthetic-fixture-v1`: explicitly generated contract-test fixtures. Every row,
  split summary and bundle states `ctc_training_allowed=false`. A fixture row
  cannot be promoted by changing the configuration mode.
- `historical-frozen-diagnostic-v1`: read/diagnose the exact immutable bundle SHA
  in `configs/training/product-speech-like-base-v1.json`. Caller-supplied expected
  hashes cannot extend this exception. This lane cannot materialize a new base
  and cannot authorize new CTC training. It does not manufacture human reviews
  or change existing index, summary, corpus, release or model identities.

The loader selects the lane with `generator.external_base_admission_mode`.
Generated fixtures must also use `--admission-mode synthetic-fixture-v1` at the
label and base-index materializers. The historical product configuration
materializer now emits a diagnostic configuration, not training admission.

## Human review v2

Each JSONL record has `schema_version=2` and
`evidence_class=speech-like-audio-review-v2`, with these required fields:

- `source_id`, `file_sha256`, `pcm_sha256`: exact recording identity;
- `reviewer_id`, `review_origin=human`, `review_revision` (positive integer);
- `supersedes_review_sha256`: null for revision 1, otherwise the preceding
  revision's canonical JSON SHA-256; histories start at 1 with no gaps or forks;
- `intended_text`: original request; `actual_text` and `actual_tokens`: what was
  actually heard, including all spoken material;
- `transcript_policy=xiaowo-four-syllable-transcript-v1`;
- `acoustic_complete`: explicit verdict on complete audio/transcription;
- `speech_present`, `kind`, `keyword_id`, `verdict`;
- `allowed_purpose=ctc-training-only` and a stable `source_family_id` shared by
  related source recordings/derivatives, never a convenient new split-local ID.

Optional known `speaker_id`, `reference_audio_sha256`, `session_id`, and
`derivation_family_id` are retained and must match supplied source metadata.
Missing identity is unknown. It is not proof of speaker/model independence.

All known revisions must be supplied. The latest supplied rejected, uncertain or acoustically
incomplete review stops admission. A previously rejected OOV observation can be
retained and superseded by a valid corrected review. Historical receipts are
never expanded or relabeled automatically. Relabeling updates actual kind/tokens;
it does not copy requested text into the actual transcript.

The current mapper intentionally supports only `你→ni3`, `好→hao3`, `小→xiao3`,
`窝→wo1`, plus a fixed punctuation/whitespace allowlist. Accepted positive labels
must exactly match canonical keyword 1 or 2. Negatives/confusables cannot contain
a complete wake token path, even with intervening tokens. Unknown text or OOV
requires exclusion or an explicitly reviewed vocabulary/mapper extension; it
must not become an empty CTC target. Empty background targets require an empty
actual transcript and explicit reviewed absence of speech.

## Binding and consumption

Materialized rows retain original `intended_text`/`intended_tokens`, then use the
reviewed actual text/tokens as CTC labels. `admission.review_history` embeds full
revision records, `review_record_sha256` binds the latest canonical record, and
`review_sha256` records the original supplied review-file hash. Downstream code
can validate the embedded chain without access to the review-file path; that
file hash is provenance, not proof that an external receipt archive was fetched.

The label gate, base-index materializer and external loader independently reject
missing or conflicting admission. The loader recomputes WAV/PCM/frame identity,
canonical corpus identity and vocabulary/keyword bindings. Before augmentation,
all four splits are checked for shared decoded PCM, source IDs, voice groups,
source families, and any supplied speaker/reference/session/derivation identity.
Rewrapping identical PCM into different WAV headers does not evade isolation.
Provider/engine names alone are not independence or leakage evidence. Existing
voice-group split isolation remains a conservative requirement.

Rendered lineage and the real trainer must revalidate these receipts against the
original sources; a boolean admission flag alone never grants training. Admission
does not authorize qualification, deployment, new holdout claims, data sharing,
or a new training run.

## Two-phase generation

`run_speech_like_corpus_generation.py` and its bootstrap accept `--generate-only`
to create a batch for listening. This stops with `pending-review.json` and does
not produce labels or an admitted training bundle. After genuine review, use
`speech_like_corpus_plan.py admit-reviewed` as a read-only check, then its
`materialize --audio-review ...` command on those exact retained files. Materialize
all split indexes with `materialize_speech_like_base_index.py` and validate the
four-split external configuration. The generation runner can forward an existing
`--audio-review` for an exactly matching deterministic batch; stale hashes fail.
The runner refuses unreviewed default materialization before generating audio.

Without an authoritative external review catalog, a self-contained chain cannot
prove that no newer revision was omitted. Operators must supply current complete
histories; reports explicitly leave global review freshness unverified.

Receipt validation proves internal consistency and byte binding. It cannot
independently authenticate a reviewer, prove that listening happened, certify
source rights, or establish distinct generator/speaker families. No operational human review,
new real speech/model generation, or training was performed to implement this gate.
Verification generates synthetic toy WAVs only.
