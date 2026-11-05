# Native Zipformer system diagnostic

This active research entry preserves the upstream model's native 80-bin
frontend and transducer decoder. It accepts the same verified mono 16 kHz
PCM16 view used by the small-model experiments; it does not force the model
through their 32-feature, five-output ABI. It is a system observation, not a
single-variable causal comparison or SSC305 qualification.

## Fixed source and recipe

[`zipformer-system-20261105.lock.json`](../configs/research/zipformer-system-20261105.lock.json)
records the official release asset, per-model hashes, exact Python/package
versions, native library hashes, keyword encoding and inference settings.
The upstream package is
[sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/).
Its release archive SHA256 is
`68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6`.
The chosen encoder is FP32 chunk 8. The runtime is sherpa-onnx/core 2.0.1 on
CPython 3.12.16. Model and package binaries remain external pinned inputs.

The preregistered decoder uses one CPU thread, max-active-paths 4, trailing
blanks 1, keyword score 1 and threshold 0.25. Its native frontend has dither 0,
normalization enabled and snip-edges false. Keyword tokens were checked with
the pinned upstream `text2token` implementation and source lexicon before
inference. No weak transcript or intended label chooses the threshold.

Feed exactly 160 samples at a time. A final short feed stays short; add no
silence. Finish input once, drain remaining model output, and reset after each
detection. These feed coordinates do not establish acoustic-end or device
callback latency. The pinned Python result exposes native token timestamps
but no `start_time`; the entry records that absence and infers no offset.

## Run and recover

Materialize the exact locked model files and install the exact wheel versions
and hashes using the project's checked CPython interpreter. Keep outputs in a
private directory outside the repository, model directory and input view:

```sh
python3 tools/run_zipformer_system.py \
  --models /private/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \
  --view /private/d0-weak-view \
  --view-manifest-sha256 EXPECTED_VIEW_MANIFEST_SHA256 \
  --output /private/zipformer-diagnostic \
  --heavy-lock /private/research/heavy-model.lock \
  --max-seconds 600
```

The entry checks every inventoried view file, each model and bound native
library before executing. The identity binds the profile, runner, lock helper,
view and every input WAV/PCM. Each clip gets an immutable call claim and native
result. A repeated completed command rechecks those bytes and makes no new
model calls. A claim without a completed result is unknown; it requires an
explicit recovery decision and is never silently retried. The local heavy
model lock also supports a validated inherited descriptor from the research
controller. All participating local workers must use the same lock path.

The deadline is checked between native decoding calls. A supervising process
must enforce a hard wall-clock timeout for a hung native call; the actual D0
attempt used an external 600-second subprocess timeout.

## Completed D0 observation

The fixed 2026-11-05 attempt processed all 64 clips: 2,082,560 supplied samples,
64 new native calls, zero added padding and no process failure. Twelve clips
triggered: seven K1 and five K2. Total inference was 3.862973 seconds, model
loading 0.432657 seconds, outer subprocess wall time 5.227807 seconds and peak
RSS 90,272 KiB. These measurements belong to this host and fixed recipe.

The identical command then verified all 131 output files unchanged and made
zero new model calls. The complete view had 790 verified inventoried files.
The experiment identity SHA256 is
`4855e58cc44c189c48a8dc51a20a6add1aee1c0858584bf21435065465811782`.
Original audio, native events and complete receipts are preserved privately.

This corpus contains four synthetic voice-preset source families and is
train-only. It has no independent human test labels, authenticated wake-end
times or long population-negative stream. The observations establish neither
FRR nor population FAR, hardware latency, generalization or a winning model.

## Offline verification

`tests/test_zipformer_system.py` exercises exact short-tail/EOF accounting,
deadline handling, format rejection, complete evidence binding, immutable
receipt publication and tampered claim/result cache rejection. Its fake
engine generates no model-quality evidence. `tests/test_research_job_lock.py`
uses real bounded child processes to check inherited lock exclusion.
