# Explicit private candidate host smoke

This local-only research entry point checks **transport equivalence**, not speech
accuracy, calibration, independent test performance, or production readiness.
It does not change the default model, train, deserialize a checkpoint, contact a
model service, or run on target hardware. No private files are bundled here.

## Inputs and execution

Build the normal host tools with strict warnings. Supply a locally approved
candidate archive, its expected SHA-256, the chosen keyword pack and its expected
SHA-256, the exact training tokens, and a **new** output directory under a private
parent. Keep outputs private: they include the admitted model snapshot, pack,
posterior traces, generated audio, events, and statistics.

```sh
python3 -S -B tools/private_candidate_host_smoke.py --execute \
  --archive /private/candidate.zip --archive-sha256 EXPECTED_ARCHIVE_SHA256 \
  --pack /private/keywords.kwk --pack-sha256 EXPECTED_PACK_SHA256 \
  --tokens /private/tokens.txt \
  --wav-runner ./build/kws_wav --replay-runner ./build/kws_decoder_replay \
  --source-revision BUILD_SOURCE_REVISION --output-root /private/new-smoke-output
```

Without `--execute`, the tool reports `NOT_RUN`. Missing required arguments or
failed admission are not passes. Existing output directories are refused. A
failure after output creation leaves a receipt and any completed logs; do not
overwrite them or automatically retry.

The archive must contain `PACKAGE-MANIFEST.json`, `candidate.kwm`,
`candidate.research.pt`, `candidate.kwm.provenance.json`, and
`candidate.research-sidecar.json`. All archive members except the manifest must
be listed with exact byte lengths and SHA-256 values. Admission checks research
purpose flags, model/checkpoint/provenance binding, training-token SHA, and
model/pack vocabulary fingerprint. Checkpoint bytes are hashed, never executed.
Archive compressed and total uncompressed size are limited to 16 MiB with at
most 64 members. All admitted model/pack bytes are snapshotted before execution.

Expected digests must come from the approved input selection, not be refreshed
merely to make a rejected candidate pass. Digests and an unsigned receipt do not
prove publisher identity or execution authenticity. The supplied source revision
is explicitly a label; tool and harness hashes are recorded separately. This
does not prove that the executables were built from that label.

## Fixed budget and acceptance

The matrix uses 399, 400, 1201, and 32000 samples at 16 kHz, each with 160/320
sample blocks, plus a fresh-process repeat of 1201 samples in both block sizes.
There are exactly ten live and ten replay processes, each with a 30-second
timeout, under a 30-minute wall budget. No tuning or retries are built in.

Each live process has a fresh engine. Expected frame counts are 0, 1, 3, and 99;
no end-of-file padding is assumed. Live/replay events, consumed frame counts,
detection and suppression counters must agree. Block-size and fresh-start
repetitions must have identical live events, statistics, and traces. Live sample
count is checked against the full WAV length, not against the last frame end.

Success is always `TRANSPORT_PASS_EFFECT_NOT_QUALIFIED`. The input is a 440 Hz
synthetic tone, not speech. Any detected keywords are retained as synthetic
negative failures, not erased because transport agrees. Never extrapolate these
short clips into real-world FAR/h or treat repeats as independent evidence.
Reusing a vocabulary-compatible pack is not candidate calibration.

## Public CI versus private execution

Public CI runs only `--self-test` in normal, `-O`, and `-OO` Python modes, plus the
default no-execution path. These artificial admission cases never launch a model
or native runner. Real private execution remains a separate opt-in local action.
The CLI can be rolled back independently of the native tools and production
defaults. Preserve prior failed receipts when updating the harness.
