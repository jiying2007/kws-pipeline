# Bounded native batch weak import (offline research candidate)

`tools/batch_weak_import.py` consumes the explicit
`bounded-batch-weak-import-v1` private configuration. It does not schedule,
arm, import a model/tokenizer, generate, decode, train, or download anything.
The archived execution schema is `bounded-native-execution-v1`; individual ASR
raw evidence keeps `source-language-conditioned-asr-v1`. These are distinct
from anonymous automatic-language/raw protocols.

## Evidence boundary

The private config contains frozen path/SHA references to the three complete
TTS, Qwen, Sense result ZIPs, original execution plan, original/readback receipt
sidecars and readback ZIP bytes, source-rights review and all its evidence,
official FSMN dictionary, legacy dictionary, and the item/opaque-ID mapping.
All references are contained within the private input root. Output must be new;
existing evidence is never overwritten. No private result/audio allowlist lives
in public code. `batch-source-contract.json` contains only public source/model
identities, generation recipe, and explicit tokenizer metadata token IDs.

The importer retains whole archives, every archive member and sidecar before
screening individual items. ZIP roster, member hashes, bounds, duplicates,
traversal and symlinks are checked. Native session summary must confirm each
exact terminal hash; terminal, retained-file roster, parsed/raw, durable
consume, native claim, controller summary, and input bindings must agree.
Known historical execution counts remain separate from offline import rejection.
The original supervisor/child identity and same-PID-namespace cleanup evidence
must show no live group members. Unknown or cross-namespace absence is rejected.

Provided Library tool receipts, readback bytes, identities and versions are
cross-checked. This code does **not** authenticate a JSON receipt or create a
Library acknowledgement. The parent retains responsibility for actual tool
receipt provenance and independent review. Saved checksums establish consistency,
not cryptographic execution authenticity.

TTS checks bind the actual request, seed, built-in speaker, locked model assets,
revision, recipe, EOS/sequence/codebook capture, native float waveform and
statistics, retained derivative WAV/PCM and geometry. The archived derivative
is checked; the offline importer does not regenerate or numerically resample it.
Both ASRs must consume those exact anonymous WAV/PCM bytes. Qwen native input
prefix, mutation checks, generated tail, token/EOS bounds and saved dual decodes
are verified. Sense native capture, raw decode attempts, token/metadata boundary,
tokenizer identity, lexical unknown and parser output are verified. Saved token
IDs are not independently re-decoded by a newly loaded tokenizer.

### Explicit Sense metadata extension

The old validator defaults remain NEUTRAL=25004 and EMO_UNKNOWN=25009. The new
batch entry passes an explicit reviewed map also containing ANGRY=25003 and
SURPRISED=25007. These IDs were read from the official, SHA-pinned SentencePiece
ModelProto repeated-piece order using standard-library wire parsing; no tokenizer
package was imported. The metadata strings, token IDs, pieces, raw capture and
parsed output must all agree, including the metadata re-extracted from the saved
decoded string. This permits metadata boundaries, not lexical
repair or an emotion ground-truth claim. Old protocol callers keep old defaults.

## Rights, splits and labels

A new `bounded-batch-source-rights-review-v1` record is required. It must bind
the official model card and full asset lock to this batch's actual load,
execution plan, original generation population, every request/seed/terminal and
WAV/PCM. The model-lock file itself must match the public source-contract SHA;
actual loaded source members must match its source lock, not a self-supplied
replacement inventory. An arbitrary license string or old Sohee review cannot admit a batch.
Only official built-in Dylan/train and Serena/development_a source rows are
supported here. The original source row and voice lineage remain intact,
including old preparation statuses; new observations are separate fields.
The supported rights scope is internal project noncommercial research only.
Public output-audio redistribution and commercial qualification are unestablished.

A candidate needs full-text dual-ASR agreement after punctuation/whitespace only.
The original intended text and each native raw transcript remain separate.
Fillers, 王, 屋 and 窝 are not removed, substituted or merged. Empty ASR is not
proof of background silence; intended negatives containing a wake-keyword
subsequence are isolated. Fixed Chinese/zh is recorded as a source-language
condition, with detected_language=null and independent-language-observation=false.

Outputs:
- `weak-fulltext.jsonl`, `fsmn-candidates.json`: full transcript machine-weak
  candidates with original splits and current_experiment_admission=false.
- `quarantine.jsonl`: disagreement or evidence rejection with exact reason.
- `report.json`: per-item provenance, consumer-specific compatibility and original
  completed-call counts. Offline import never adds to those execution budgets.
- `four-token-consumer/`: separate legacy-compatible weak manifests and audio;
  OOV is consumer-specific, not a claim that the full transcript is invalid.

`load_fsmn_candidates()` is a new read-only **metadata/PCM adapter**, compatible
with the existing official FSMN data-row contract. It verifies official
2599-class raw-ID-minus-one targets, blank0 exclusion, audio/PCM, 300ms zero-pad
identity, output frame geometry, report/manifest provenance and split. It re-screens
the original source row, both raw votes and measured PCM and requires identical
semantics; pool assignment must follow the original split and actual content. It does
not compute FSMN features or invoke the old top-level prepare/train scripts.
It grants no current-experiment admission. No empty CTC replacement is used.

The unchanged `training/weak_speech_dataset.py` remains the actual legacy
four-token feature loader. It only consumes its train manifest; development rows
stay separate. Machine weak data, including data readable by either consumer,
is not human gold, independent test evidence, product acceptance, or permission
to launch training. No accuracy improvement or causal effect is inferred.

## Offline usage and tests

From the candidate root:

```
python3 -S -B tools/batch_weak_import.py import PRIVATE/config.json NEW_OUTPUT
python3 -S -B tools/batch_weak_import.py inspect NEW_OUTPUT
python3 -S -B training/weak_speech_dataset.py NEW_OUTPUT/four-token-consumer
python3 -S -B -m unittest discover -s tests -p 'test_*.py'
python3 -S -B -O -m unittest discover -s tests -p 'test_*.py'
python3 -S -B -OO -m unittest discover -s tests -p 'test_*.py'
```

Fixtures are manually constructed bytes and are not executions or real Library
receipts. Semantic negatives rebind raw, parsed, terminal, native-summary and
archive/manifest hashes before checking rejection. Tests cover prefix/input
mutation, metadata/token mismatch, forced-language mislabelling, PCM/lineage,
partial terminal, TTS EOS/codebook/request, source-rights scope/lock/split,
namespace unknown, receipt inconsistency and consumer provenance tampering.
Private actual audio, transcripts, results and rights packets are excluded from
the public patch. Public candidate contains code, artificial fixtures and public
source metadata only.
