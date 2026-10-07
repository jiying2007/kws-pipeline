# Restricted development corpus structural validation

`tools/validate_real_human_development_corpus.py` is a standalone, standard-library
CLI for checking the **declared structure and exact local bytes** of a restricted
`development-feedback` corpus. It does not run a model, train, invoke final AFE,
consume held-out qualification data, upload audio, or change any qualification
contract. Run it only on an already authorized development corpus in its
restricted environment. The repository test uses generated temporary WAVs only.

## Use

Requires Python 3.10+ and POSIX no-follow/directory-descriptor operations (Linux).
Use a stable, access-controlled local snapshot, without concurrent writers.

```sh
python3 tools/validate_real_human_development_corpus.py \
  --manifest /restricted/development/manifest.json \
  --audio-root /restricted/development/audio \
  --summary /restricted/reports/development-structure.json
```

The summary's parent directory must already exist. The destination must be new
and outside the audio root. Existing files, directories, symlinks and hardlinks
are never overwritten. Directory ancestors and input files must not be symlinks;
input files must be regular files with one hard link. The CLI traverses audio
paths relative to a pinned root directory descriptor. It reads inputs only and
publishes the completed summary atomically with mode `0600`. On validation or
publication failure it exits `2`, emits a fixed-label error to stderr, and emits
no successful summary to stdout. On success it exits `0`, writes the JSON, and
prints the same JSON. A preexisting output is left unchanged on failure, so
consumers must check the current exit status rather than trust an old report.
If publication succeeds but temporary-link cleanup fails, exit `2` can leave a
complete `0600` summary and its private temporary link. The neutral error does
not claim that no output exists; inspect the destination before retrying.

`--summary` is intentionally not the dormant workflow's `--public-summary` flag.
There is no compatibility integration or implicit public-upload permission.

## Manifest contract

The machine-readable structural reference is
[`commercial/real-human-development-corpus.schema.json`](../commercial/real-human-development-corpus.schema.json).
The CLI is authoritative for strict JSON token types and byte/filesystem/event
checks that JSON Schema alone cannot express. It rejects duplicate object keys,
nonfinite numbers (including overflow such as `1e999`), booleans or strings used
as numbers, and float tokens used as integers. JSON Schema itself treats `1.0`
as an integer; the CLI deliberately does not.

Only these top-level fields are accepted:

- `schema_version`: integer `1`
- `dataset_id`: 8–64 characters matching `[a-z0-9][a-z0-9._-]*`
- `corpus_role`: exactly `development-feedback`
- `recordings`: 2–10,000 records, collectively covering keyword IDs `1` and `2`
  and at least one wholly negative recording (`expected: []`)

Every record requires these fields; only `snr_db` is optional:

- `recording`, `speaker_id`, `session_id`, `source_id`, `room_id`, `device_id`:
  opaque identifiers, 1–64 characters using the same character set as dataset ID
- `input_path`: canonical relative `.wav` path under the audio root; at most 8
  components, 64 characters/component and 512 characters total; components start
  with an ASCII letter or digit, then use only letters, digits, `.`, `_` or `-`
- `input_sha256`: 64 lowercase hexadecimal digits of the **entire WAV file**
- `input_bytes`: integer 46–536,870,912 (512 MiB), matching exact file length
- `duration_s`: finite number greater than zero and at most 3,600 seconds,
  matching actual PCM frame duration within `1e-9` second
- `distance_m`: finite number from 0 through 10
- `azimuth_deg`: finite number from 0 inclusive through 360 exclusive
- `snr_db`, if present: `null` or finite number from -100 through 100
- `tags`: unique values from `quiet`, `household`, `speech`, `music`, `traffic`,
  `noise`, `near`, `far`, `rear`; `rear` is required exactly when azimuth is
  between 135 and 225 degrees inclusive
- `expected`: at most 10,000 objects with exactly `keyword_id` (integer `1` or
  `2`), `start_s`, `end_s`; events must be ordered, nonoverlapping and positive
  length within the actual recording duration; adjoining events are allowed
- `capture`: exactly `sample_rate_hz` (integer 8,000–192,000), `channels`
  (integer 1–16), `sample_format` (`pcm_s16le`), matching WAV headers
- `consent_scope`: exactly `product-kws-development`
- `retention_class`: exactly `restricted-raw-audio`

All objects are closed. There is no free-text `notes`, transcript, general
metadata map, or extra nested property. Recording IDs, audio paths and audio
content hashes must be unique. Identifiers are syntax-constrained declarations,
not evidence of anonymization. Manifest size is limited to 16 MiB.

Supported audio is canonical little-endian RIFF/WAVE PCM16: the 16-byte PCM
`fmt ` chunk followed immediately by a `data` chunk, with no other chunks or
trailing data. Compressed, extensible, RF64, extra-metadata-chunk, empty,
truncated and partial-frame WAVs are rejected. The tool verifies RIFF/file/data
sizes, format, alignment, byte rate, capture, duration, SHA256, and reads the
entire PCM payload, checking file metadata before and after reading. It does
not silently convert unsupported audio. Perform any authorized conversion
separately and update the manifest to match those exact bytes.

## What the result means

The aggregate report contains record and pseudonymous-positive-speaker counts,
expected-event counts by keyword, and wholly-negative audio exposure. It never
copies dataset/recording/person/session/source/room/device identifiers, paths,
per-file hashes, tags, or arbitrary strings from the manifest. Errors use fixed
field labels and row/event positions rather than private values or OS exception
text. The report does not include a corpus identity digest and is not an
attestation that can authorize a downstream job.

`structural_checks_passed` and `audio_bytes_hashes_and_payload_verified` describe
these checks only. The report explicitly sets `consent_verified`,
`labels_verified`, `anonymization_verified`, `training_authority`,
`qualification_authority`, `shipping_authority`, and `publication_authority`
to `false`.

Human review and independent records must establish consent, permitted uses,
actual speaker identity and independence, correct wake/negative labels, lawful
retention, and anonymization. A declaration does not prove any of those facts.
Even aggregate statistics can be sensitive for a small group. Review before
sharing the report; keep audio and manifests restricted. This tool grants no
training, qualification, shipping, or publication permission. It is not a
sandbox against a privileged or concurrently malicious local filesystem owner.

## Tests and design provenance

```sh
python3 tests/test_restricted_development_dataset.py
```

CI runs this synthetic adversarial suite with the existing tracked-worktree
guard. Tests cover successful aggregation and authority flags; strict JSON and
closed nested metadata; bounded identities/types; root traversal, symlinks,
hardlinks and nonregular files; hashes, capture and full WAV payloads;
nonoverlap/coverage; private error redaction; and atomic no-clobber publication.
No real corpus, model, inference, training, upload, or network access is needed.

Design source: branch `feat/restricted-development-dataset-iteration-v1`, commit
`064493c5bd71c4de40aa9d6569254724ee1ada46` in `jiying2007/kws-pipeline`.
Reviewed source-file SHA256 values:

- `commercial/real-human-development-corpus.schema.json`:
  `b90a2b724b044c95633f675f76360a55c8c2978bbb9af28362fe74b363c9a1ad`
- `tools/validate_real_human_development_corpus.py`:
  `03bd5f4d0f52d34ce8e95b068616948a4f0813f3ee7d3872dd05b0e97ae85698`
- `tests/test_restricted_development_dataset.py`:
  `a725a9be1e78e51d6dc3fc355f3ef3ea3f87ec9b876599f82111247e454be042`

This is a newly hardened standalone port of the useful development-role/schema
idea, not a merge of that branch. Its historical workflow, runner, AFE, sealer,
qualification-validator coupling, permissive coercions, PII field blacklist,
and inferred publication/feedback authority are not restored. No existing
qualification caller is redirected to this CLI.
