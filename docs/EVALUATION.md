# Continuous-audio evaluation

KWS release quality is measured on continuous recordings, not isolated clip accuracy. In `v0.3.x`, evaluation evidence is also bound to the **actual WAV bytes**, not only to a reference JSONL file.

## Reference and audio contract

Reference JSONL owns the expected events and declared duration of each recording:

```json
{"recording":"living-room-001","path":"living-room-001.wav","duration_s":3600.0,"expected":[{"keyword_id":1,"start_s":91.2,"end_s":92.1}],"speaker_id":"spk101","session_id":"sess08","source_id":"field-2026-08","domain":{"distance_band":"far"}}
```

Negative-only recordings use `"expected": []`.

For every row, `run_corpus.py` reopens the referenced WAV and records:

- file SHA256;
- decoded mono-16-kHz PCM16 SHA256;
- frame count;
- real duration from frames/sample-rate;
- stable recording/path identity;
- speaker/session/source/room/device metadata when supplied.

The declared `duration_s` must match the real WAV duration. This prevents FAR exposure from being inflated by declaring a short file as many hours of audio.

Keep speakers, sessions, source recordings and device sessions disjoint across train/calibration/test/qualification. Final human qualification should require speaker/session/source metadata through `training/audit_dataset.py`.

### Development split roles

The historical split names remain stable for artifact and consumer compatibility, but their semantic roles are explicit:

- `train` = **development-training**;
- `calibration` = **development-calibration**;
- `test` = **development-feedback**.

The development `test` split is **not** an independent held-out product test: its metrics may feed curriculum, replay, candidate selection and subsequent development decisions. Once evidence from that split influences training or policy, it must continue to be treated as development feedback.

Reserved held-out/protected authority belongs to qualification stages that do not feed the same evidence back into training, including synthetic qualification where applicable and, for shipping authority, the governed real-human and target-DUT/cohort lanes. The file/key name `test` is retained only for compatibility; new diagnostics should expose the canonical role `development-feedback`.

Runtime detections are JSONL:

```json
{"recording":"living-room-001","keyword_id":1,"time_s":92.25,"confidence":0.73}
```

## Run the real C runtime

```bash
python3 eval/run_corpus.py \
  --runner ./build/kws_wav \
  --model release/base.kwm \
  --keywords release/xiaowo.kwk \
  --references qualification/references.jsonl \
  --audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --provenance qualification/detections.provenance.json

python3 eval/score_events.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --summary qualification/eval-summary.json \
  --false-positives qualification/false-positives.jsonl \
  --false-rejects qualification/false-rejects.jsonl
```

Evaluation provenance schema v2 binds:

- exact runner;
- model;
- keyword pack;
- references;
- detections;
- canonical evaluation-corpus identity containing every real WAV hash/PCM hash/frame count.

The scorer reports FAR/hour, FRR, per-keyword counts and p50/p95 post-end latency. Matching is monotonic per keyword: maximize valid matches first, then minimize total distance to annotated keyword end. NaN/Inf and out-of-range values are rejected.

### Posterior replay cache identity

When `--posterior-dump`, `--decoder-replay` and `--posterior-cache` are supplied
together, cached traces are keyed by model SHA256, posterior-dump executable
SHA256 and original WAV SHA256. The cache sidecar uses schema v2 and
`kws-posterior-trace-cache-v2`; its producer hash must match the requested
executable. Legacy v1 entries and entries with missing or mismatched producer
identity are never reused or relabelled. Changing the generator forces a fresh
trace; decoder, keyword-pack and replay-override changes can reuse it.

Evaluation and boundary-tool provenance retain the producer SHA256 recorded for
each trace, including cache hits. The executable is checked against the run's
initial hash before cache admission and after trace generation; a change during
generation fails without publishing that trace. This binds the executable bytes,
not external shared libraries or the surrounding execution environment.

## Domain metrics

When references contain domain metadata:

```bash
python3 eval/domain_metrics.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --output qualification/domain-metrics.json
```

The report includes available buckets such as distance, azimuth, RT60, noise, local playback, composite domain ids and keyword confusion/miss matrices. Aggregate FRR must not hide a weak far/side/reverb/motor/playback domain.

Synthetic calibration/test/qualification rendering rotates positive domains `far -> mid -> near` to prevent accidental absence of far positives. That is a deterministic software/data-orchestration contract only; a real product corpus must independently contain enough genuine samples in every claimed bucket.

## Long continuous FAR

`kws_raw_stream` consumes raw PCM16 blocks through the same C engine without per-clip resets. `eval/long_far_stream.py` exercises continuous negative streams and preserves false-accept evidence.

`.github/workflows/far-nightly.yml` is a recurring hosted/synthetic regression. It may detect state accumulation regressions, but its generated exposure is **not** a shipping FAR/hour measurement. Shipping FAR needs long real continuous background audio through the final microphone/enclosure/AFE path and those original WAV bytes must be retained/bound by evaluation provenance.

## Current diagnostic priority and admission boundary

The highest-value next validation is a matched-context, event-matched check of
continuous state and VAD behavior under one frozen source/model/pack/config/AFE
tuple. Retain frame phase, VAD decisions, acoustic/decoder/refractory state
provenance and annotated event boundaries. Positive and negative streams must
use the same declared context policy; compare per-keyword misses, false accepts
and matched-event latency, not clip-presence accuracy or surrogate scores alone.
Use the [existing context contract](../eval/README.md#match-positive-and-negative-context-declarations).
Missing historical initial state leaves the original false-accept cause unknown,
even if PCM bytes match or a cold-start/reference-only diagnostic reproduces an
error under its own conditions.

Preserve continuous state within a recording. Reset only at an intended session
boundary or an actual declared capture discontinuity; blanket per-clip resets
are not a demonstrated fix. Reference-only analysis and native C/backend
numerical admission are separate gates. D20 remains **FAIL** and D90 **NOT_RUN**;
the [current admission checklist](../research/d20-diagnostic-admission-v1/PLAN-SCHEMA.md#current-admission-checklist)
keeps exact source/build, saved-input consumption, resource scope and separately
authorized numerical execution explicit. Saved D20 CMVN is a separate call of
the same C function on the same input, not an observed internal stage-0 hook.

This is the next validation priority, not a completed experiment or permission
to execute one. It introduces no blanket reset, threshold sweep, new model,
training run or product qualification claim.

## Audio discontinuities

A production capture pipeline can lose timeline continuity because of XRUN, device/route changes, clock resets or suspend/resume. The product integration must call `kws_engine_notify_discontinuity()` at those boundaries so pre-gap frontend/RNN/decoder state cannot be joined to post-gap audio.

A qualification recording itself must represent the intended continuous timeline. If the capture system reports an XRUN, retain that event in the acquisition metadata and either reject the recording from strict qualification or replay the exact discontinuity behavior through the product integration path.

## Required product domains

At minimum include:

- normal home conversation, near-homophones and partial wake phrases;
- repeated syllables and shared-prefix phrases;
- television/phone/smart-speaker playback;
- own-device TTS/music through shipping AEC;
- AEC residual/double-talk;
- robot motor, gearbox, fan, pump and chassis vibration;
- silence/impulses/microphone handling;
- expected angles and distances, including the claimed 3–5 m far-field region;
- representative RT60/room states;
- moving/static device states;
- day/night gain and final AGC/AEC/NS modes.

Positive coverage should be bucketed by speaker, session, distance, angle, SPL/SNR, speaking rate and device state.

## Failure replay

False accepts can feed training:

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root data/calibration \
  --output-dir build/hard-negatives \
  --manifest build/hard-negatives.tsv
```

False rejects may be replayed with `eval/mine_false_rejects.py`. Retraining must use the exact token vocabulary and a compatible warm start when `--head-only` is selected.

Never mine from the final qualification-heldout corpus and then report those same source recordings as unbiased final evidence. A held-out corpus stops being held-out once its failures influence model, threshold or keyword-policy tuning.

## Release evidence

For every shipping model/pack tuple retain:

- KWSP v2 + model provenance schema v3;
- KWKP v3;
- exact checkpoint/training tokens/training manifests and real training-corpus identity;
- clean dataset audit;
- release tokens/config and frontend identity;
- exact evaluation runner/references/**original evaluation WAVs**/detections/provenance/summary;
- domain metrics and failure-replay outputs;
- corpus identity/version;
- target-board benchmark and machine/raw target evidence;
- approved SKU policy;
- qualification manifest schema v3 and gate result schema v4.

Hosted CI and cross compilation are software correctness signals, not target-board acoustic qualification.
