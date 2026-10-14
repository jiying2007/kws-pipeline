# Existing audio inventory and group split (2026-10-14)

This is an executed, read-only inventory and a conservative split plan, not a
training run, a label admission receipt, a new speech corpus, or qualification.
It does not change r1–r5 evidence, the runtime, a model or any threshold.

## Measured result

- 385 existing mono 16-kHz PCM16 files, 0.338807743 hours of file duration,
  385 distinct whole-file decoded PCM hashes. This is not independent exposure.
- 16 screen32 derived files; 240 restored original FAR domain files plus their
  120 source files; 3 existing continuous synthetic fixtures; 6 original k1
  punctuation files. The 22 native float source WAVs are retained separately,
  not silently converted or counted again as training samples.
- All 240 restored domain metadata records match the frozen original index
  except relocated paths. Every domain WAV and source WAV hash was checked;
  each source decoded PCM hash also matches. No reconstruction was run here.
- All six k1 derived WAV hashes, decoded PCM hashes and frame counts match
  the original r5 input freeze. Native WAV hashes also match. Existing r5
  transcripts yield one three-family machine consensus (k1p6-002: 你好小屋)
  and five disagreements. Exact target consensus 0/6 does not mean six errors.
- 385/385 records lack speaker and session identities. Providers, engine names,
  synthetic seeds and whole-file byte differences do not fill those identities.
- The partition is development=385, train=0, sealed_acceptance=0, quarantine=0.
  Existing observed families are deliberately locked to development, including
  sources and augmentations. No independent acceptance hours are available.
  Empty manifests represent that result; they are not successful training inputs.
- Original 2 FA / 8 h remains failed. Seeds 1103 and 3301 and their associated
  source families are development-only. Their 9569 seconds of prior restored
  prefixes overlap existing exposure and are never added to independent hours.

`unreviewed_or_incomplete` means formal review/admission evidence is incomplete
in these public inputs. It does **not** assert that no informal human observation
exists. Private listening observations/chat are not copied into this inventory.
Machine consensus remains distinct from human review and from human gold.

## Run against existing audio

Run from the repository root, supplying paths to the original already-restored
public synthetic assets (these paths are not embedded in the report):

    python tools/inventory_effect_data.py --far-root "$FAR_ROOT" \
      --k1-root "$K1_MEMBERS" --r5-root "$R5_MEMBERS" --output "$NEW_OUTPUT"

Repository-only mode is supported by omitting all three external-root options;
its report explicitly distinguishes historical metadata from verified audio.
K1 and r5 roots must be provided together. The command does not fetch anything.
Outputs require a new directory so existing split snapshots cannot be overwritten.

To re-measure and re-plan a generated JSONL against its logical roots:

    python training/effect_dataset_split.py --manifest "$NEW_OUTPUT/development.jsonl" \
      --root repository=. --root known_far_restored="$FAR_ROOT" \
      --root k1_original="$K1_MEMBERS" --output "$SECOND_NEW_OUTPUT"

For new input JSONL, give unique `record_id`, an audio path, source/speaker/session
metadata and either human review records or distinct-family `asr_evidence`.
The CLI reads actual WAV bytes and rejects declared hash mismatches. Group
components join speaker, session, original/derived PCM, source/derivation family,
reference identity and source paths transitively. Missing metadata or incomplete/
disputed labels go to quarantine unless a known development-only family requires
retention in development. Machine-consensus candidates can only go to train.
Fully described human-review candidates use a deterministic group-level 70/15/15
hash allocation; tiny group sets need not populate every partition. The seed and
input membership are part of the snapshot: adding records requires a new plan
and review, not silently changing an existing sealed split.

`human_review_recorded` is a classification of supplied evidence, not listener
authentication or label-admission validation. All outputs forcibly retain
`human_gold=false`, `training_admitted=false`, and
`product_qualification_allowed=false`. Actual consumption must still pass
`training/training_admission.py` and the existing live-byte lineage audit.
A planned sealed candidate is not proof of independent recording identity.

## Verification

- `python tests/test_effect_dataset_split.py` (21 focused tests)
- `python tests/test_dataset_audit.py` (existing audit test passes)
- Full 385-row CLI remeasurement with the logical-root mappings above passes.
- `tests/test_training_audit_closure.py` is blocked in this executor because
  PyTorch is not installed. No full training test suite or training was run.

`report.json` includes the measured inventory, metadata coverage, transitive
partition, quality tiers, original-pool source overlap and separation status.
No private absolute paths, waveform bodies, user listening text or credentials
are published here. All accompanying JSONL files are metadata-only split plans.

## Local inventory boundary and recoverable sources

The 385 files are only the materialized inputs scanned in this run, not the size
of the project or every available corpus. `remote_declared_not_materialized`
records the hash-bound public release `speech-like-base-5204b798033f`: 384 synthetic
recordings / 24 voice slots, originally 8 kHz and normalized to 16 kHz (normalizing
does not restore lost bandwidth). Its declared partitions remain train 128,
calibration 64, test 64 and qualification 128. Those held-out partitions cannot
be recycled into training. The existing governed materializer can restore the
release in an authorized environment; this inventory did not run that downloader
or verify current remote availability. “Not materialized here” is not “lost.”

The retained model provenance also declares 1856 historical training records.
The report separately compares their declared PCM hashes to measured local
files, without claiming that their waveform bodies were all read or that they
supply new independent hours. The generic planner accepts additional JSONL and
logical audio roots, so these sources can use the same measurement and grouping
logic once materialized. An existing `test`, `qualification`, `sealed_acceptance`
or `calibration` split is preserved as a held-out reservation, never moved to
train by the hash allocation, even if label metadata is incomplete. Retaining
a reservation does not validate its independence or qualify its labels.

An existing train/held-out lineage overlap is rejected rather than reassigned.

## Historical-partition preservation fix

Only wholly new, unpartitioned source components may enter the random allocation.
Existing train/training remains train; development/dev/selection/calibration/
validation/val remains development; test/qualification/sealed_acceptance remains
reserved; quarantine remains quarantine. Both previous_split and planned_split
are checked alongside split. Training/tuning history flags are binding too.
An inherited role applies to all previously unpartitioned derivatives in the
same component. Conflicting roles (including qualification plus development-only)
are rejected, and unknown historical role names fail closed. No old development
or training sample is silently promoted into a fresh sealed partition.

Identity placeholders now follow the paired-runner rule: blank/unknown/none/null/
pending/n/a/na/unset/unspecified/missing/tbd/-/? and the multi-character forms
followed by _, -, colon or space are unknown. Natural-language heard_text is
checked only for nonempty text, not filtered as an identity. Source aliases
(source_id/source_family/source_family_id/family_id/derivation_family/
derivation_family_id) share one source namespace; PCM ancestor aliases share
one PCM namespace; WAV/reference-file hashes share a separate file namespace.
All present known audio hashes must be 64 lowercase hexadecimal characters.
