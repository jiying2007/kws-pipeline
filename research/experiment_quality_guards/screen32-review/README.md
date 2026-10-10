# SOURCE listening handoff, 16 retained clips

This is a label-free listening packet and an unfilled receipt template. No filled
human receipts are checked in. It does not grant dataset, CTC, training, shipping or
independent-accuracy qualification. No inference or new audio is generated.

## Listen and record

1. Use the previously delivered `01.wav` through `16.wav`. These are the exact
   derived mono 16 kHz PCM16 files from generation run 38018029787, with no further
   trimming, gain change, resampling or conversion. Native float audio is a
   different representation and is not covered by this listening declaration.
2. Match the filename and WAV SHA-256 against `listening-packet.json`. The adapter
   additionally verifies the raw PCM bytes and retained decoder-input bindings.
   Do not assume that listening number and recognizer clip number are the same.
3. Work from audio alone where possible. Write the words actually heard, including
   repetitions and unexpected words. Do not repair them using a generation plan
   or a recognizer answer. If only part is clear, keep `actual_text` null and put
   those words in `partial_text`, with an incomplete/ambiguous status. Inaudible,
   unknown and empty labels are never inferred to mean silence or a negative.
4. Copy `pending-receipts.json` to a new private file. Leave all immutable row
   fields and the packet hash unchanged. Keep any rows not yet reviewed pending,
   or remove them from this submission; either way the report retains all 16.
   A duplicate row within one submission is an error.
5. For a reviewed row set `label_origin` to `human_listening` and explicitly
   declare whether you listened, listened to the entire clip, transcribed it
   completely, and supplied independent human evidence. Set the status to clean,
   ambiguous, incomplete or inaudible. Record acoustic completeness separately as
   COMPLETE, INCOMPLETE or UNKNOWN. These are declarations, not measured proof.
6. Declare `hypothesis_exposure` as EXPOSED if you previously saw a plan or
   recognizer result, NONE_DECLARED if you did not, or UNKNOWN if unsure. The
   existing owner has seen recognizer tables: a new label-free packet does not
   make that listening blind. Exposed direct listening can still supply human
   actual words; it never establishes blind accuracy or held-out independence.
7. `reviewer_id` and `source_reference` start null. Both must be nonempty for a
   row to qualify as complete human actual-word evidence. A minimal pseudonym
   and private receipt/chat-message reference suffice; no real name or email is
   required. Pending or unqualified partial reviews may leave them null. Missing
   either keeps even an otherwise complete submission unqualified. These fields
   establish declared accountability, not authenticated identity. Do not enter
   credentials or unnecessary identity details. Filled receipts and derived reports can
   contain private information. Do not commit or publish them automatically.

`complete` is the reviewer's declaration that the actual-word transcription is
complete. `listened_entire_clip` is a separate full-playback declaration. Both,
clean status, independent human evidence, explicit acoustic COMPLETE and nonempty
actual words without unresolved partial text are required
by the adapter's existing human-truth gate. Reviewer-declared acoustic completeness
is retained separately; the text-only guard's `acoustic_completeness` stays UNKNOWN.
Do not mark a known incomplete utterance clean and complete.

## Offline commands

Run from the repository root with Python's standard library only:

```sh
python3 -B research/experiment_quality_guards/screen32_human_review.py packet
python3 -B research/experiment_quality_guards/screen32_human_review.py template
python3 -B research/experiment_quality_guards/screen32_human_review.py report
python3 -B research/experiment_quality_guards/screen32_human_review.py report \
  --receipts /private/review-1.json --output /private/source-report.json
```

`--output` creates a new file exclusively and refuses to overwrite an existing
file. Exit 0 means a report was produced, including a wholly pending report; it
never means the audio or dataset passed. Invalid input exits 2 without creating
an output. No model, download, subprocess, Docker, ASR or training is invoked.

For corrections, retain the original receipt document. Increment the document's
`revision`, set `supersedes_receipt_sha256` to the preceding document's canonical
SHA-256 (UTF-8 JSON, sorted keys, compact separators, unescaped Unicode), and
include only the rows being updated. Pass every receipt file, oldest first, with
repeated `--receipts`. The report also lists the canonical hashes. Revisions must
be contiguous. The latest submitted row wins even if incomplete or pending; an
older complete label is never silently reused. Keep the complete history when
moving these private files. Hashes cannot authenticate a reviewer or prove that
anyone listened, and an omitted later revision cannot be detected from old files.

## Dual-ASR-first weak screening

Use the separate opt-in view to prioritize agreement of the two **saved** ASR
observations while retaining all human declarations and receipt history:

```sh
python3 -B research/experiment_quality_guards/screen32_human_review.py screening \
  --receipts /private/review-1.json --output /private/screening-report.json
```

The packet, template and original `report` behavior are unchanged. Screening
first validates the complete chronological receipt history and all original
source/audio/decoder bindings. It then joins both saved ASR observations by WAV
identity, in the explicit order `qwen06`, `sensevoice`. The output binds the
canonical original report digest and preserves every original report-row field.
A separate `screening` object retains each recognizer's raw text, status and
quality flags, plus comparison-only normalized strings. `actual_text` and
`partial_text` always remain the human's own declaration or null.

Every one of the 16 rows belongs to exactly one partition:

- `human_and_both_asr_agree`: a complete human receipt matches both ASR strings.
- `dual_asr_agree_human_uncertain`: ASR agrees but the human receipt is absent,
  pending, ambiguous, partial, unattributed or otherwise not complete.
- `human_vs_dual_asr_disagreement`: a complete human declaration conflicts with
  the agreeing ASR pair; neither side is adjudicated correct.
- `dual_asr_disagree`: two complete usable ASR observations disagree.
- `dual_asr_missing_incomplete_or_flagged`: either observation is missing,
  noncomplete, empty after normalization, or carries quality flags.

Only two complete, nonempty, unflagged observations with equal normalized
strings yield `screening_candidate_text`. Normalization removes whitespace and
Unicode punctuation for comparison; it does not fold characters or repair words.
All three agreement partitions are **weak machine consensus**, including
human-agreeing rows. A complete receipt means the declaration was sufficiently
recorded, not that it is gold. Counts use the full 16-row denominator and are not
accuracy or error rates. Disagreement and unusable rows get no candidate.

Screening without `--receipts` is allowed but explicitly reveals saved machine
hypotheses. Pending human words, human completeness, acoustic completeness and
original label diagnostics remain unknown/unqualified. Do not give this view to
someone who is meant to review from audio alone. Filled receipts and screening
reports remain private unless publication is separately authorized.

The production-vocabulary diagnostic remains conditional on the preserved
human declaration. Machine consensus, even for an OOV word, does not expand the
production vocabulary, convert review formats, create a CTC/blank target, admit
training, qualify shipping or independent accuracy, authorize a new run, or
publish anything. Both humans and ASR may mishear; final adjudicated text stays
null. Corrections still require new receipt revisions, never overwriting history.

## Boundaries and reproducibility

The packet binds both immutable run identities, the unchanged 74-file source
freeze, generation receipt, exact original blind-job bytes, its distinct
canonical JSON digest, retained decoder inputs, and both saved recognizer
observations. The adapter verifies the source freeze, retained input hashes,
WAV/PCM bytes and primary-raw/terminal freeze linkage. It imports no hypotheses
into the reviewer packet. Only after validating submitted receipts does the
internal SOURCE report join the original plan and saved recognizer diagnostics
by verified WAV identity. Pending rows receive no joined label diagnostics.

`quality_gates.human_truth` and `label_preparation` remain the actual-word and
original-plan diagnostics. Neither is dataset admission. The report preserves
unmodified actual words, including unsupported characters, and identifies the
specific production four-syllable transcript policy when reporting OOV characters.
A research vocabulary can differ; this report does not generalize OOV across
policies. It never folds characters, creates a blank target, invokes a production
label builder, invents a coverage policy, assigns a split, verifies voice lineage,
or changes `ctc_target=null` / `training_admitted=false`.

All checked-in rows remain unreviewed. Test labels are invented fixtures, not
human review of these recordings. The existing execution freeze and retained
machine/audio evidence are not modified by this handoff.

### Test scope added on 2026-10-10

The parent package README is retained historical PR479 source and documents the
original 77-test, directory-only scope. This handoff adds 26 focused tests, so the
current workflow covers 103 tests. The new adapter/tests additionally read the
retained generation WAVs, both run manifests/results and frozen source files;
they are not standalone when copied out of this repository. All 103 tests pass
locally in normal, `-O` and `-OO` modes using the standard library. The workflow
also checks test-file inventory and runs the new adapter tests in all three modes.

### Weak-screening extension

The opt-in screening view adds 15 synthetic-label regression tests to the same
test file: 41 focused tests and 118 package tests in total. The existing workflow
already runs this file in normal, `-O` and `-OO` modes, so no CI or historical
retention metadata changes are needed. The added cases cover all partitions,
missing/failed/blank/flagged observations, exact comparison, invalid types,
private output non-overwrite, receipt supersession, preserved human/ASR words,
OOV diagnostics, and no promotion to gold or admission. No private filled review
records are used as repository fixtures.
