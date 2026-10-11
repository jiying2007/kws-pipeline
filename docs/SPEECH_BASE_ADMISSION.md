# Speech base admission

Future real-speech/real-TTS CTC bases require supplied actual-audio label evidence.
Human review v2 remains the operational admission route. Three independent ASRs
may establish a prospective training label under the bounded policy below, but
self-authored metadata cannot authenticate those ASRs or admit training. A
requested utterance, deterministic generation, or checksum alone does not
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

## Three-ASR machine-consensus training-label policy v1

The user's three-ASR agreement rule is represented by
`speech-three-asr-consensus-v1` records and
`speech-three-asr-training-admission-v1` receipt envelopes. These are distinct
from human review v2: `label_origin=machine-consensus` and
`label_source=machine-three-asr-consensus-v1`. The outer `reviewed-real-v1` name
continues to describe the audio-bearing lane, not a human source or human gold.
TTS stays synthetic; recorded and synthetic source provenance must not be
rewritten by a labeling decision.

This version implements structural label eligibility only. Every constructed
machine receipt has `label_eligible=true`, `ctc_training_allowed=false`,
`authentication_independently_verified=false`, and
`status=PENDING_EXTERNAL_VERIFICATION`. No trusted out-of-band authentication
route is connected. The base materializer, external loader and trainer reject
these receipts, including one pending row in an otherwise human-reviewed
manifest. Changing `verified=true`, admission flags, or an outer evidence tag
cannot enable training. Explicit diagnostic runs remain non-promotable under
the existing warm-start taint rules. No machine-labeled stage can currently
enter promotable dataset ancestry, so its schema is unchanged.

Requirements checked without executing ASR:

- Exactly three actual, successful, complete-source ASR observations, each from
  one member of a frozen `speech-three-asr-recognizer-policy-v1` roster. Missing
  outputs, repeated passes, or a human observation as the third vote fail.
- Distinct recognizer IDs, provider/model identities, model-artifact digests and
  model families. Provider names alone do not establish independence. Versions,
  model digests and decode configuration are bound to each observation.
- Fixed Mandarin transcription settings, beam size and temperature; empty
  prompts and hotword lists; no target conditioning. The complete v1 decode
  fields are hashed; changing settings changes policy identity. The effective
  decoding recipe is distinct from the provider's native parameter names:
  `native_decode_config_sha256` binds the actual retained native configuration.
  A width-one/greedy decoder may use effective `beam_size=1` only when that
  mapping is established by its pinned decoder contract and retained settings;
  never invent a provider API parameter or fill an unobserved default. If a
  setting cannot be represented or verified, it is unsupported by this v1
  structural gate and must await genuine configuration evidence or a separately
  reviewed policy revision. A matching digest alone does not authenticate it.
- Every result binds the same source ID, WAV SHA-256, decoded PCM SHA-256 and
  positive frame count, with coverage from frame zero to the measured end.
  `coverage_complete`/`transcript_complete` are machine-result claims. They do
  not establish human acoustic completeness or prove that a source utterance
  was not clipped. No human `acoustic_complete` field is fabricated.
- Unanimity after only the existing fixed punctuation/whitespace removal.
  Raw transcripts remain in the observations. No synonym, homophone, Unicode,
  or simplified/traditional substitutions are allowed; “屋” never becomes “窝”.
  The existing five-token vocabulary and positive/negative wake-path checks
  still apply. Voiced OOV is quarantined, never blank or `wo1`.
- Three empty outputs cannot establish silence. This policy only labels nonempty
  speech; background/absence-of-speech still needs independent evidence.
- The hash-bound `source_provenance` contains the actual `synthetic` flag,
  `source_type`, train-only `source_role`, `license_id` and
  `rights_evidence_sha256`. Inspection requires exact equality with source
  metadata. Fixture, screen-only and held-out roles cannot become training
  sources by labeling them. Rights-reference consistency is not a rights audit.
- All supplied human revisions are retained verbatim in `human_review_history`.
  A current disagreement/rejection/uncertainty quarantines the machine label;
  it never overwrites what the listener heard. `human_conflict` and
  `historical_human_conflict` are recomputed, retaining earlier conflict even
  after a valid later human correction. Supplied history cannot prove global
  freshness or absence of omitted observations.

An optional `speech-asr-authentication-verification-v1` report may name its
issuer, trust policy, UTC verification time and evidence digest, and bind the
frozen roster plus complete observations. Null explicitly means no report was
supplied. A supplied report is checked for internal consistency only; its JSON
claims do not establish an independent authority, signatures or authentication.
No verification facility, recognizer output or signature is fabricated.

For already retained sources, the read-only command is:

```bash
python3 tools/speech_like_corpus_plan.py inspect-three-asr \
  --intents /controlled/train-intents.jsonl \
  --generated-root /controlled/generated \
  --consensus-receipts /controlled/three-asr-receipts.jsonl \
  --expected-count 3
```

It compares supplied receipts against measured original source WAV/PCM/frames,
source roles and provenance, then reports eligibility and zero admitted rows.
It does not run ASR, create labels/bundles, revise human history or start training.
The existing two-ASR Qwen observations remain insufficient; this implementation
creates no third output and admits no existing rows.

`evaluation_gold_allowed=false` and `qualification_allowed=false` apply to these
machine labels. Generic development diagnostics may inspect them, but release
metric validation, real-human corpus intake, final-qualification AFE input and real-human qualification scoring
reject machine-only gold, including nested provenance. This restriction does not
permanently disqualify a model trained on independently admitted machine labels:
a future model can still undergo independent human qualification. Source rights,
split isolation, lineage/ancestry, coverage, resources and shipping gates remain
separate requirements; no label consensus authorizes a model run or deployment.

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

## Warm-start dataset ancestry

New promotable checkpoints require `training_admission.dataset_ancestry`
schema v1. Its ordered stages retain the reviewed manifest and lineage hashes,
canonical measured WAV/PCM/frame corpus, reviewed row count, source checkpoint
hash, and preceding ancestry digest. The corpus uses
`hashed-path-and-metadata-v1`: private path and speaker/session/source/room/device
text is replaced by SHA-256 identifiers, while the original corpus digest is
retained. Audit comparison independently hashes the supplied audited paths.
No audio, transcripts, credentials, or private path text is copied into ancestry. The complete stage list has its own
canonical SHA-256. A cold start has one stage and no parent; each warm start
copies all validated stages and appends the current stage. Recording indices
remain scoped to their stage, including when different manifests share a name.
This binds supplied evidence, not listener or checkpoint-author authenticity.

`training_manifests`, `training_corpus_identity`, and the admission's top-level
`manifests` continue to describe only the current optimization inputs. They are
checked against the final ancestry stage before optimization and at export.
The trainer rereads current manifest, lineage, and WAV/PCM identity before saving
to reject input drift. Provenance normalization preserves the source-checkpoint
link so the independent qualification gate can check it again.

Qualification must audit the union of every stage's exact manifest, lineage,
and measured audio identities against held-out audio. Reusing the exact same
dataset across generations reuses one audit binding while preserving each
checkpoint stage; different content under the same basename is never collapsed.
The existing audit policy still requires all distinct named splits to be
disjoint. Partially overlapping but different training-stage manifests therefore
fail closed; this change does not introduce an exception for those overlaps.

Missing, malformed, or legacy-only ancestor evidence cannot be reconstructed
from the child dataset. Such a warm start retains the existing permanent
`diagnostic-or-unverified` promotion taint. Historical provenance remains
readable, and immutable registry byte verification is unchanged; it does not
grant new candidate eligibility or retrofit reviewed ancestry to retained models.

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
