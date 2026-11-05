# D1 native Zipformer observations

`tools/run_d1_zipformer_system.py` is a separate, bounded native system
diagnostic. It does not change the three CTC candidates, their nine training
arms, their development jobs, the old D0 runner, or any historical receipt.

## Inputs and completed preparation

The job consumes the same completed preparation as the D1 training and
development jobs. It requires all 64 projected rows, including uncertain
labels and retained native generation failures, with the declared global
source IDs from the first 64 cells of the frozen plan. It checks the complete
preparation inventory, replays objective projection and source-group separation,
and verifies every supplied WAV and PCM hash. The split remains 48 training
and 16 development rows, with two predeclared development families.

The recipe uses `completed_preparation` with four fields:

| Field | Binding |
| --- | --- |
| `queue_root` | Absolute path of the original, real training queue |
| `job_id` | Its actual preparation job ID |
| `receipt_sha256` | SHA-256 of that completed job's original terminal receipt |
| `manifest_sha256` | SHA-256 of the preparation `MANIFEST.json` |

These hashes are filled only after preparation completes. The consumer reads
the original queue, claim, terminal receipt, logs and declared output records;
it does not create a virtual dependency or change the old queue request. It
then checks the preparation's request hash against the original job recipe.
The projected label roster hash is derived from these verified preparation
bytes, never predicted before preparation.

Other recipe fields are `schema: kws-d1-zipformer-system-job-v1`, the frozen
`plan_sha256`, sorted unique `source_ids` (64), sorted
`held_out_source_family_ids` (two), absolute `models`, `profile_sha256`,
`source_files`, and `timeout_seconds` (positive and at most 120). Obtain the
source mapping from the entrypoint's `source_records()` after freezing source.

## Native execution contract

The existing `zipformer-system-20261105.lock.json` remains fixed: native FP32
chunk 8, the original model/token/lexicon hashes and runtime files, one CPU
thread, a feed of 160 real PCM samples, the same keyword definitions and
default thresholds. No padding, gain adjustment, threshold search, ASR text,
intended label, or weak label enters the decoder. `decode_samples` and engine
construction are reused from the old runner without editing it.

Before the first NumPy or native inference import, the new worker sets the
offline and ORT telemetry environment and installs the existing Linux native
network guard. All model assets are local and hash-verified. The worker
inherits the actual queue's heavy-model lease and stays in that command's
process group. Both the job and this separate one-job queue have a maximum
allocation of 120 seconds; actual elapsed time remains the queue's measurement.
The original 19-job queue and its budget remain unchanged.

The queue command is the core Python 3.12.16 interpreter followed by `-I -B`,
the frozen entrypoint, `--recipe {recipe}`, `--output {output_dir}`, and
`--heavy-lock` with the shared absolute lock path. `depends_on` is empty because
the separately completed preparation is explicitly bound by the recipe.
Required outputs are `summary.json`, `observations.jsonl`, and `MANIFEST.json`.
An isolated `-I -S -B ... --help` call is safe for a stdlib-only CLI check;
actual inference needs the fixed runtime's installed packages.

Every cell receives a create-only claim immediately before native decoding.
Native events, token timestamps, supplied sample counts and EOF accounting are
saved before deriving label observations. A claimed cell with no result is
consumed and cannot be retried. A new invocation requires an empty output
directory; a repeated run never overwrites a partial or completed attempt.
`audit_finished(recipe, output)` provides a separate read-only replay of the
sealed result, original native claims, exact observations and split summaries.

## Meaning of the reported counts

All 64 rows contribute to raw native observations. Event counts and clips with
events are reported separately for training and development.

For each keyword, a clip contributes to machine-weak positive or negative
counts only when its projected keyword mask is known. A technical generation
failure, uncertain label, quarantined label or unknown keyword mask contributes
no label denominator. Its raw events remain visible. These exclusions apply
even if a recognizer appeared to agree with a failed waveform.

CTC vocabulary or frame feasibility does not define the native system's
keyword denominator. There is no shared token edit score, loss comparison,
formal qualification denominator, independent human test, population false
alarm rate, device latency claim, or quality winner. Native timestamps remain
native timestamps and supplied PCM coordinates; they are not invented acoustic
completion times.

## Verification

`tests/test_d1_zipformer_system.py` uses only the Python standard library and
synthetic protocol doubles. It covers exact roster/source/profile pins,
per-keyword known-mask denominators, failed/unknown/quarantined exclusions,
original PCM sample accounting, no padding, consumed claims, modified native
bytes, actual external queue terminal verification, and a real successor
queue's inherited lease/deadline protocol. Run it in normal, `-O`, and `-OO`
modes. These checks perform no model construction or inference.

The native guard and streaming decoder retain their existing separate tests.
The new endpoint's actual model readiness requires its fixed runtime/assets
metadata check; successful stdlib tests alone do not claim a real Zipformer
result.
