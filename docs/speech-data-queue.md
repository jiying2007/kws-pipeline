# Maintained speech data queue (research lane)

This small offline tool consumes saved results and emits a ready-to-read weak
research training pool, an isolation pool, statistics and proposed future work.
It never downloads, loads, calls, retries or trains a model. No existing runner,
scientific pin, product admission or retained-source file is changed.

## Source / configuration / task boundary

- `tools/speech_data_queue.py`: shared PCM/label utilities plus offline screening,
  grouping, accounting and full local result capture.
- `tools/import_saved_speech_result.py`: reads existing per-slot native ZIPs;
  verifies roster, lengths and hashes; retains all original bytes before parsing.
- `training/weak_speech_dataset.py`: runnable research loader; uses the existing
  `frontend_spec.features_pcm16` and returns `(feature_matrix, target_ids)`, the
  pair interface used by `train_ctc.Manifest`. This does not invoke the protected
  product training CLI or mint its admission documents.
- Plan JSON freezes recipes, original lineage roots, rights references, vocabulary,
  recognizer identities/configuration and total call budget. Private plan/result
  bodies belong in private storage, never in the public repository.
- A task is a proposed generation or anonymous ASR slot, not execution authority.
  Existing verified launchers must own actual resource limits and durable claims.

The current native saved-result adapter supports Qwen3-ASR and SenseVoice and the
retained single-result Sohee TTS receipt. It reuses the maintained Qwen transcript
parser and corrected Sense parser/binding constants. It does not copy a runner or
use the old default tokenizer path, and does not load SentencePiece. Other TTS,
clone, or ASR formats need a narrow saved-result adapter, not a cloned pipeline.
Real speech and environment recordings may enter directly with source metadata.
Augmentation requires original source/reference roots; lineage assertions are
caller-supplied evidence and not speaker identity authentication.

## Run in the repository

`python tools/import_saved_speech_result.py --archive SAVED.zip --staging SLOT --item-id ITEM --operation qwen_asr --slot OPAQUE --plan-sha256 PLAN_HASH`

The importer emits `event.json`; artifact paths are relative to SLOT. When merging
slot events under one input root, prefix their paths with SLOT, retain original
archive metadata and use the hash of the exact final plan. Do not rewrite recipes,
seeds or prompts to restart an exhausted slot.

`python tools/speech_data_queue.py --plan plan.json --events events.json --input-root INPUT --output NEW_PACKAGE`

`python training/weak_speech_dataset.py NEW_PACKAGE`

A minimal complete synthetic plan/event example is constructed by
`tests/test_speech_data_queue.py`. Run without ML dependencies:

`PYTHONPATH=tools:training:research/qwen6_asr/core python -m unittest discover -s tests -p test_speech_data_queue.py -v`

## Data semantics

Two independent successful, blind, file/PCM-bound native recognizers may produce
weak machine labels. Input event transcripts are ignored: text comes from saved
native output, checked against native decoder evidence. Families cannot be
renamed to produce independent votes. Only punctuation/whitespace is removed;
窝 and 屋 remain different. Actual text determines targets and class. A requested
positive that actually says something else never inherits its requested label.
Negative/confusable/background intent containing a wake path is isolated. Empty
ASR text does not establish speech absence. OOV text is retained and isolated,
never converted to blank. CTC sequence feasibility is checked before export.

Rights records require explicit research-use allowance and retained evidence
hashes. This is evidence consistency, not a legal opinion or authenticated rights
certificate. Missing/unknown rights or resources prevent admission and new task
proposals; existing ASR observations can still be classified without blocking
other items. Model licensing alone does not prove rights to every voice/output.

TTS/clone inputs require a source-bound successful generation receipt, clean
signal flags and EOS evidence. The currently supported native receipt is the
saved Sohee format; unsupported generators are isolated until a parser is added.
Neither research admission nor two/three votes establish acoustic verbatim
completeness, human gold, independent product testing or model qualification.

Original speaker/reference roots form connected components, including transitive
shared lineage. Components crossing train/validation/test are isolated. Renaming
clone/augmentation slots is not a new independent speaker or split.

The optional third recognizer is configuration only. Disagreement emits
`third-route.jsonl` for review/execution under an independently authorized task;
the tool never schedules or invokes it. The current parser supports only the two
native families above; adding a third family also requires its saved-result
parser. It must not be represented as an existing executable third vote.

## Attempts and recovery

Events carry exact plan hash and per-item/per-operation consecutive attempt IDs.
Success/content failure/known invocation/unknown invocation cannot retry. Only a
proven pre-call infrastructure failure can propose the next unchanged-recipe
attempt. Both known and unknown invocations consume budget; proposals reserve
remaining budget deterministically. This offline queue is not a concurrent
execution ledger: repeated planning can reproduce the same proposal; executors
must atomically claim a slot and persist its actual event before execution.

A failed native slot with matching raw outcome, `attempted=1` summary and bound
claim counts as an actual adapter attempt even if no usable transcript survives.
It is an infrastructure/binding failure, not content success or an unused slot.
Without invocation evidence it remains unknown and consumes budget conservatively.
Statistics retain every supplied attempt, failure and item; effective yield is
weak accepted items / all planned items, including quarantined items.

## Output and delivery

A fresh package directory is required. Raw plan/events are written and fsynced
before interpretation. Every available raw artifact, failed result, supplied audio,
source-rights reference and vocabulary is retained with hashes. Model/resource
assets are inventoried and checked but not duplicated into the delivery package. Missing or changed
artifacts are reported, not silently deleted. `weak-train.jsonl`, weak heldout,
quarantine, third-route, report and a package manifest follow. The loader verifies
the retained bytes, weak scope, split, targets and PCM before computing features.
Checksums provide integrity, not execution or Library authentication. Storage
survival across host loss is not guaranteed by fsync.

This full local package is separate from the existing small `result_delivery`
status capsule. Do not truncate audio to fit that capsule or claim Library upload
or readback in code. The parent delivery tool must upload the full private package
and verify actual Library readback. Only source/tests/docs and synthetic test
metadata may be published from this candidate.

## Verification limits

Fixture tests exercise successful weak loading using synthesized PCM and native
result-shaped fixtures, not real recognizers. A separate private read-only Sohee
replay consumed six retained slot archives: 2 TTS + 4 ASR attempts. The first item
lacks two successful recognizers; the second has foreign-language disagreement.
Both remain isolated, yield 0/2, no new slots, no new ASR/TTS/training calls. This
replay does not establish ASR accuracy or the quality of a newly trained KWS model.
