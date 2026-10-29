# Speech base admission

Future real-speech/real-TTS CTC bases require supplied actual-audio label evidence.
Human review v2 and code-rooted three-ASR source verification are distinct label
routes. The three-ASR verification and consumption path is implemented, with
**zero production train roots and zero newly admitted real recordings** today.
Self-authored metadata cannot authenticate ASRs or admit training. A
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

Optional `source_family`, `family_id` and `derivation_family` fields are also
retained and bound to the supplied review/provenance. All recording/source
family fields, including `source_id`, share one leakage namespace: the same
value in different family fields cannot cross splits. Speaker, session,
reference-file and PCM identities retain separate namespaces. See
[the corpus identity contract](CORPUS_IDENTITY.md#recording-and-family-identity-namespace).


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

A JSON-only eligibility check constructs `label_eligible=true`,
`ctc_training_allowed=false`, `authentication_independently_verified=false`, and
`status=PENDING_EXTERNAL_VERIFICATION`. Caller `verified=true` or edited receipt
flags are never authority. A separate controlled resolver can construct
`VERIFIED_SOURCE_TRAIN_LABEL` only after the source evidence checks below; every
consumer re-resolves the original evidence. This is exercised with invented
container fixtures, not real ASR executions. The two existing retained source
roots are permanently registered as `source_screen_only`, not train roots.

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

## Native v2 and the user's three-ASR preference

New native captures use `speech-three-asr-consensus-v2` and the explicit
`label_selection_policy=trusted-three-asr-preferred-training-v2`. Three trusted,
distinct frozen model families with conservative unanimous nonempty canonical text
may supply a train-only label even when a listener's ordinary observation is
uncertain, rejected, or different. Both current and historical human conflicts
remain audited. The source/rights/explicit sample-use ban still wins; a listening
verdict alone is not a usage ban. V1's prior disagreement rule is unchanged.

The native config and capture validators distinguish Qwen3-ASR generation,
SenseVoice full-input greedy CTC, and short-input Whisper decoding. They preserve
native null/auto/absent parameter semantics; they do not invent a generic beam,
temperature, language, or hotword API. A complete frozen configuration, resolved
runtime configuration, actual loaded asset/source locks, and captured completion
are required. Effective configuration dependency digests must resolve to retained
bytes: Qwen's actual resolved GenerationConfig snapshot, SenseVoice's tokenizer,
parser and frontend, and Whisper's tokenizer. The latter dependencies must also
match the loaded asset/executed source inventory. No model is imported or run by
these validators.

Full original PCM input coverage, decoder termination without truncation, and
acoustic word-for-word completeness are distinct. V2 requires captured native
termination evidence and leaves `acoustic_verbatim_completeness=unknown`.
Whisper's final returned text/tokens cannot prove natural EOT: finalize can insert
EOT and strip it from the result. Its supported contract therefore requires the
pre-finalize decoder-update trace and actual stop reason; missing/unknown capture
is rejected. The public Whisper candidate source lock is **inactive**; expected
weight hashes are published source metadata, not downloaded/verified assets.
Small/base are the same family and can contribute only one vote. Distinct model
families can still share errors; three votes do not establish statistical error
independence. No CPU, memory,
latency or accuracy result is claimed.

V2 persisted consensus records require empty inline `human_review_history` and
retain `human_observation_history` as exact private artifact references and
hashes only. A source-bound wrapper can reference an unchanged unstructured note
with `certainty=unknown` and `observed_text=null`; no reviewer ID, training purpose,
accepted verdict, acoustic-complete flag, or mappable tokens are fabricated.
Recognized original human-review-v2 JSON must use `original_format=human-review-v2`:
the resolver validates its full original revision chain, usage prohibitions and
conflicts, while the admission carries no original body. Known typed history
cannot be hidden as an opaque raw note. Original OOV “屋” remains “屋”. All wrappers,
original byte hashes and conflict flags are bound to the controlled source policy.
Later trusted history changes require a reviewed replacement/revocation of the
old root, not an additional alias that leaves stale receipts enabled. This
invalidates stale receipts rather than silently removing observations. The current roots do
not import any private listening notes. Publishing any original note, source
identity, path or transcript requires separate authorization; the implementation
and public tests contain invented metadata only.

## Controlled retained-source resolver

`tools/asr_source_resolver.py` contains a small code-owned root index: root ID,
control-plane `trust_root_commit`, existing provenance path/SHA, source role, and
supported capture contract. The existing provenance remains the only archive and
exact-member inventory. Its `executed_head_sha` and run ID describe the actual
execution separately from the control-plane commit. Adding a genuine train root
requires reviewed code and independently checked source/rights/capture provenance;
there is no receipt, environment variable, CLI, or config option to supply a root.
The schema parser does not invent signatures or establish external authority on
its own. Authority is the reviewed immutable root, not a caller's verified report.

The operator may supply storage through `--asr-evidence-dir` at producer/base
entry points or `KWS_ASR_EVIDENCE_DIR` at independent loaders/trainers. This is a
portable **location only**. Archive names are `<archive-sha256>.zip` and must match
the pinned provenance. The resolver checks exact member paths and hashes, source
WAV/PCM/frame identity, source role/origin/rights, complete source-bound history,
three distinct model/source locks, actual native configurations, starts,
completions and complete raw outputs. Unsupported/partial historical captures
cannot be repaired by defaults or a third invented output.

Each independent entry opens a new bounded verification session. It uses no-follow
directory/file descriptors, regular-file checks and bounded actual reads; ZIP
members are also size-limited before and during reads. A batch shares immutable
byte snapshots instead of rereading a large archive per row. Before output effects
and at successful session exit, watched paths are reopened and strongly rehashed.
The current conservative limits are 128 MiB per compressed archive, 32 MiB per
member and 256 MiB total retained snapshot bytes; larger evidence is unsupported
until a separately reviewed bounded-streaming design exists. This is not a
writable-filesystem lock: later consumers independently recheck, and any detected
change aborts. Closed, cross-record and caller-JSON contexts are rejected.

`admit-reviewed` and `materialize` accept optional `--asr-consensus` beside human
`--audio-review`; the two files must cover disjoint records and jointly cover the
entire intended batch. Only machine train rows are eligible. This producer labels
previously unlabeled generated sources (`text` is the original request). An
existing admission or fixed source `actual_text` is an immutable labeled artifact;
relabeling it requires a separately designed source-bound supersession that retains
the old value and record hash. Ordinary differing/uncertain human observations
belong in trusted private history references, not in that fixed source field, and
do not veto native-v2 selection. All source identities, roles, origin and rights
must remain exact. Generic generation manifests do not yet establish all of
`source_role`, `source_type`, `rights_evidence_sha256` and source lineage. Future
onboarding must bind those facts to the reviewed source policy and retained rights
evidence; the producer never fills a missing `train` role, license or lineage with
a default. The base materializer, four-split
external loader and trainer independently verify source evidence before admission;
the trainer binds the original source frames, not an augmented scene's duration.
`inspect-three-asr` remains read-only structural eligibility with zero admissions.
No existing screen-only source is automatically upgraded.

## Binding and consumption

Materialized rows retain original `intended_text`/`intended_tokens`, then use the
admitted actual text/tokens as CTC labels. Machine rows retain their typed
admission, source origin/rights/role, label provenance and verified source proof;
they cannot be replaced with a human-looking or fixture receipt. For human rows, `admission.review_history` embeds full
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
New per-manifest `label_evidence` summaries identify human-review and verified
machine-consensus row counts plus each machine row's record/proof/root/provenance
hashes. Every ancestry stage validates these summaries. Legacy absent summaries
remain `legacy-unknown`, not presumed human or machine-free; their bytes, schema-v1
parent hashes and digest rules are unchanged. The compatibility field
`reviewed_rows` counts admitted rows, not necessarily human-reviewed rows.
`training.admission` is the authoritative preserved carrier through export and
warm starts; corpus normalization intentionally remains identity-only. Machine
training does not make `promotion_allowed=false`; an independently human-qualified
model is still eligible, subject to all existing source isolation and release gates.

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
