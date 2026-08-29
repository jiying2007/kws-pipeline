# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a separate artifact-bound qualification bundle built from the exact model, model-export provenance, source checkpoint, training vocabulary/manifests/environment, keyword pack, release vocabulary, runtime config, evaluation runner/corpus, target-board benchmark binary/audio and measured board evidence.

## Artifact contract

Shipping artifacts use separate ABIs:

- model: **`KWSP` ABI v2**;
- keyword pack: **`KWKP` ABI v3**;
- model lineage/frontend contract: **frontend-spec v2**, with `frontend_kind=0` logmel or `frontend_kind=1` pcen-lite;
- qualification policy: **schema v2**, including statistical FAR/FRR bounds.

The release manifest carries runtime frontend identity, and the gate requires it to match model lineage exactly.

```text
locked training env + checkpoint + training tokens + training manifests
                 |
                 v
        export_model.py -> model.kwm + model.kwm.provenance.json
                                  |
model/provenance + concrete lineage + KWKP v3 + release tokens/config
                                  |
                                  +-> exact kws_wav + held-out references
                                  |      -> detections + provenance + metrics
                                  |
                                  +-> exact target kws_board_bench + board WAV
                                  |      -> board summary
                                  |
                                  +-> target resource/soak/thermal/power evidence
                                                       |
                                                       v
                                            qualification_manifest.py
                                                       |
                                             SKU policy schema v2
                                                       |
                                                       v
                                            qualification_gate.py
```

## 1. Freeze the training environment and model lineage

`training/requirements.lock` pins the real CTC dependency set and `training/Dockerfile` defines the repository training container. Build/deploy the training image by immutable digest and expose that digest as `KWS_TRAINING_IMAGE_DIGEST=sha256:<digest>`.

For a shipping candidate, require the digest at training time:

```bash
KWS_TRAINING_IMAGE_DIGEST=sha256:<image-digest> \
python3 training/train_ctc.py \
  --manifest data/train.jsonl \
  --tokens release/training-tokens.txt \
  --frontend logmel \
  --require-container-digest \
  --output release/base.pt

python3 training/export_model.py \
  --checkpoint release/base.pt \
  --tokens release/tokens.txt \
  --output release/base.kwm
```

Use `--frontend pcen-lite` when that is the selected model frontend. Warm start/export must not change frontend identity silently.

The checkpoint records Python/PyTorch/platform/CUDA/cuDNN identity when present, repository SHA, training-code hashes, requirements-lock/Dockerfile hashes and container image digest. The exporter carries that environment into `base.kwm.provenance.json` together with checkpoint hash, training/export vocabulary identities, training-manifest hashes, frontend identity/spec, hyperparameters and per-matrix int8 quantization diagnostics.

Retain the **actual checkpoint**, **actual training token file** and **every training manifest** referenced by the checkpoint. Qualification re-hashes these bytes; provenance hashes are not treated as self-authenticating declarations.

Training and release token files may differ byte-for-byte only when their canonical token→ID mapping/fingerprint is identical.

## 2. Compile and freeze KWKP v3

Compile the exact release keywords:

```bash
python3 tools/compile_keywords.py \
  --tokens release/tokens.txt \
  --keywords release/keywords.tsv \
  --out-pack release/xiaowo.kwk \
  --out-json release/keywords.json
```

The pack stores per-keyword threshold, token path, trailing-blank requirement, priority and prefix policy. A pack change creates a new release tuple even when `.kwm` is unchanged. For prefix-overlapping phrases, use the explicit policy columns shown in `keywords/zh_cn_overlap_example.tsv`.

## 3. Audit split isolation

Run `training/audit_dataset.py` before training/tuning/final qualification. It always gates exact decoded mono-16-kHz PCM16 duplication, including rewrapped/renamed WAVs.

Real-human JSONL should use the richer identity schema:

```json
{"audio":"audio/u001.wav","speaker_id":"spk001","session_id":"s01","source_id":"src001","room_id":"living-room-a","device_id":"robot-a","target_ids":[1,2]}
```

For independent human qualification, require identity metadata explicitly:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=data/qualification.jsonl \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json
```

Speaker/session/source overlap across splits is always a hard failure. Room and device overlap are reported and can be made hard gates with `--fail-room-overlap` / `--fail-device-overlap` according to the SKU evidence design.

The final qualification-heldout corpus must never be mined into replay and then still be called independent heldout.

## 4. Use measured dual-mic RIR before human far-field collection is complete

The domain renderer can consume a real robot/room dual-mic RIR JSONL manifest. Each entry identifies room/position/distance/azimuth/RT60/device pose and both RIR WAVs; file hashes and a canonical entry hash are validated. A qualification-capable measured-RIR manifest must cover near/mid/far bands.

Example fields:

```json
{"room_id":"room-a","position_id":"p4m-front","distance_m":4.0,"azimuth_deg":0.0,"rt60_s":0.46,"device_pose":"normal","mic1_rir":"rir/p4m-front-mic1.wav","mic2_rir":"rir/p4m-front-mic2.wav"}
```

Set `domains.rir_manifest` in the training config. The renderer convolves clean speech/TTS with the exact two RIRs, records RIR manifest/entry/file SHA256, onset/ITD and then passes the dual-mic result through the selected AFE backend.

Measured RIR makes the room/enclosure/mic transfer function real, but **does not make synthetic speech a human held-out corpus**.

## 5. Bind the shipping audio-pipeline adapter

For the final AFE chain use `domains.afe.backend=command`. The command must write `{output}` and `{result}`. The sidecar must report integer `latency_samples >= 0`; it may report `pipeline_sha256`, `source_sha` and `toolchain`.

The AFE provenance identity binds:

- command template SHA256 (not temporary expanded paths);
- actual invoked executable SHA256;
- declared configuration bundle SHA256;
- left/right input hashes;
- output and result-sidecar hashes;
- reported latency and optional pipeline/source/toolchain identity.

This prevents a random temporary directory from changing identity and prevents an un-hashed `audio-pipeline` binary from silently participating in qualification. AFE latency is added to expected event timing before wake-latency scoring.

## 6. Run held-out continuous audio

Use the exact retained `kws_wav` binary:

```bash
python3 eval/run_corpus.py \
  --runner qualification/kws_wav.eval \
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

python3 eval/domain_metrics.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --output qualification/domain-metrics.json
```

Domain metrics use a global monotonic one-to-one event/detection assignment for keyword confusion. A negative-only domain remains eligible for hard-domain FAR based on negative exposure even when it contains zero expected wake events.

A product far-field claim requires real far positives in the held-out corpus; synthetic deterministic or measured-RIR/TTS far coverage is not a substitute.

## 7. Run frozen-model long-FAR regression

`far-nightly.yml` freezes one model/pack per frontend and runs multiple independent negative shards against that exact tuple. `eval/aggregate_far.py` refuses to combine hours from different runner/model/pack hashes and computes a one-sided Poisson FAR upper bound.

This is useful regression evidence, but its evidence class remains synthetic streaming FAR. Training-seed robustness must be evaluated separately and never counted as additional negative exposure hours for one model.

## 8. Run the physical target-board benchmark

Cross-build `kws_board_bench` with the target toolchain and run the exact `.kwm/.kwk` with representative post-AEC/post-NS audio:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Retain the exact target executable as `qualification/kws_board_bench.target`. The report binds runner/model/pack/audio hashes and records model/pack/arena bytes, mean/p50/p95/p99/max processing time, RTF and p99 scheduling headroom.

Hosted x86 or cross-build success is not a target-board measurement. The v0.2 Cortex-A32 build enables the guarded NEON int8-weight × float-activation GEMV path but still requires physical-board timing/thermal/power evidence.

## 9. Record target evidence

Copy `configs/qualification.evidence.example.json` and replace all placeholders with candidate measurements. Required evidence covers target/board revision/SoC/toolchain/compiler flags/governor/audio-front-end identity, soak duration, CPU, RSS, stack high-water mark, maximum temperature and average power.

Use `kws_engine_get_stats()` for zero-I/O runtime counters when diagnosing long soaks; publish/snapshot the struct from the control path rather than adding logging or filesystem I/O to the real-time callback.

## 10. Build the byte-complete manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.jsonl \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

Repeat `--training-manifest` for every manifest used by the checkpoint.

The verifier independently checks:

- canonical KWSP model ABI v2 and KWKP keyword-pack ABI v3 layouts;
- fixed 16-kHz/400/320 geometry, dimensions, finite values, token bounds, padding and duplicate paths;
- model/pack/release-token vocabulary identity;
- training-token mapping against release tokens;
- actual checkpoint/training tokens/training-manifest multiset against model provenance;
- model frontend kind/name and frontend-spec v2 lineage;
- runtime frontend kind/name against model lineage;
- actual model/provenance/checkpoint/tokens/manifests/pack/config/eval/board SHA256 relationships;
- reference/detection counts, event bounds, audio hours and FAR/FRR/latency formulas;
- board WAV geometry, block count and benchmark formulas;
- target/toolchain/governor/audio-front-end/resource evidence.

The manifest contains no generated timestamp; identical inputs produce identical JSON.

## 11. Apply an explicit SKU policy with statistical bounds

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Exit codes:

- `0`: valid evidence and all policy gates passed;
- `1`: valid evidence but one or more thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy, including runtime/model frontend mismatch.

Policy schema v2 gates both point estimates and one-sided confidence bounds:

- `confidence_level`;
- `max_frr` and `max_frr_upper_bound`;
- `max_far_per_hour` and `max_far_upper_bound_per_hour`;
- minimum audio hours and expected wakes;
- p95 wake latency, p99 runtime, RTF/headroom, soak, CPU, RSS, stack, temperature and power.

FRR upper bounds use Wilson binomial bounds. FAR upper bounds use exact one-sided Poisson rate bounds. Therefore `0 FA / 24 h` is **not** treated as true FAR=0; at 95% confidence its upper rate is about 0.125 FA/hour. Evidence duration/event count must be large enough for the approved SKU bound.

`configs/qualification.policy.example.json` is an example schema, not a shipping commitment.

## 12. Retain the complete release tuple

Retain together:

- source SHA;
- `.kwm` + `.kwm.provenance.json`;
- source checkpoint, training tokens and every training manifest;
- training container digest plus lock/Dockerfile/code hashes;
- release token vocabulary;
- KWKP v3 `.kwk` + keyword source TSV;
- shipping runtime config and exact frontend identity;
- measured RIR manifest/RIR hashes if used;
- shipping AFE executable/config/input/output/latency provenance;
- exact evaluation runner/references/detections/provenance/summary/domain metrics;
- exact target benchmark runner/board WAV/board summary;
- target evidence JSON and approved SKU policy;
- dataset audit report;
- qualification manifest + gate result;
- final bundle SHA256.

## Software baseline vs shipping qualification

A green repository can be a mature **software baseline** without being an acoustically qualified Mandarin SKU. The domain-aware loop, measured-RIR adapter and frozen-model nightly long-FAR provide strong deterministic/regression evidence, but shipping claims remain blocked until a real Mandarin model, independent human/device held-out corpus, shipping audio-pipeline evidence and physical target-board evidence exist. Repository issue #2 tracks that external evidence gate.
