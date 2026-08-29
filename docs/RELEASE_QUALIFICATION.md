# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a separate artifact-bound qualification bundle built from the exact model lineage, the exact original audio bytes, the final runtime/AFE artifacts and machine-verifiable physical-target evidence.

`v0.3.x` deliberately hard-cuts the qualification evidence schemas. Older qualification manifests/provenance are not accepted as v0.3 shipping evidence.

## v0.3 artifact and evidence contracts

- deployable model: **KWSP ABI v2**;
- deployable keyword pack: **KWKP ABI v3**;
- frontend lineage: **frontend-spec v2** (`logmel` or `pcen-lite`);
- model provenance: **schema v3**;
- evaluation provenance: **schema v2**;
- dataset audit: **schema v3**;
- target evidence: **schema v2**;
- qualification manifest: **schema v2**;
- qualification policy: **schema v2**;
- qualification gate result: **schema v3**.

The software release version may change while KWSP/KWKP remain unchanged. v0.3 changes the public runtime API and evidence contracts, not the on-device model/keyword binary layouts.

## Evidence graph

```text
immutable training OCI image
        +
training JSONL/TSV + every training WAV byte/decoded-PCM identity
        +
clean dataset audit (speaker/session/source isolation)
        |
        v
train_ctc.py -> checkpoint with canonical training corpus identity
        |
        v
export_model.py -> KWSP v2 + model provenance schema v3

KWSP + KWKP + runtime config + final AFE
        |
        +-> exact kws_wav + held-out references + every held-out WAV
        |       -> detections + evaluation provenance schema v2 + metrics
        |
        +-> exact target kws_board_bench + representative board WAV
        |       -> board timing summary
        |
        +-> collect_target_evidence.py + raw measurements
                -> target evidence schema v2
                         |
                         v
                qualification_manifest.py schema v2
                         |
                  SKU policy schema v2
                         |
                         v
                qualification_gate.py -> schema v3 PASS/FAIL
```

## 1. Freeze the training environment

`training/Dockerfile` no longer resolves or upgrades Python packages from the network. It requires a prebuilt base image referenced by immutable OCI digest:

```text
registry.example/kws-training-base@sha256:<64 hex>
```

Build the repository wrapper with:

```bash
python3 training/build_container.py \
  --base-image "$KWS_TRAINING_BASE_IMAGE" \
  --tag kws-training:v0.3
```

The base image must already contain the Python/PyTorch versions declared by `training/requirements.lock`. Pass the final image digest into training:

```bash
KWS_TRAINING_IMAGE_DIGEST=sha256:<final-image-digest> \
python3 training/train_ctc.py \
  --manifest data/train.jsonl \
  --tokens release/training-tokens.txt \
  --frontend logmel \
  --require-container-digest \
  --output release/base.pt
```

The checkpoint records repository/training-code hashes, lock/Dockerfile hashes, Python/PyTorch/platform identity and the final training image digest. `.github/workflows/training-integration.yml` can periodically exercise the real `torch_ctc` path when `KWS_TRAINING_BASE_IMAGE` is configured.

## 2. Bind the real training corpus

`train_ctc.py` accepts either TSV (`WAV<TAB>token_ids`) or JSONL. Human qualification projects should use JSONL so identity metadata is retained:

```json
{"audio":"audio/u001.wav","tokens":[1,2,3],"speaker_id":"spk001","session_id":"s01","source_id":"src001","room_id":"living-room-a","device_id":"robot-a"}
```

At checkpoint save time the trainer reopens every WAV and records:

- file SHA256;
- decoded mono-16-kHz PCM16 SHA256;
- frame count and duration;
- stable manifest-relative path;
- identity metadata when present;
- canonical whole-corpus SHA256.

Changing a WAV without changing the manifest therefore changes model lineage and is detectable.

## 3. Audit split isolation

Before final training/qualification run:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=data/qualification.jsonl \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json \
  --fail-within-split
```

Speaker/session/source overlap across splits is always a hard violation for final human qualification. Room/device overlap is reported and may be promoted to a hard failure for the SKU. The final qualification-heldout corpus must never be mined into replay and then still be called independent heldout.

The v0.3 qualification manifest requires the audit to be `clean=true`, schema v3, to require speaker/session/source metadata, and to cover the selected training/qualification manifests.

## 4. Export and freeze the model

```bash
python3 training/export_model.py \
  --checkpoint release/base.pt \
  --tokens release/tokens.txt \
  --output release/base.kwm
```

Model provenance schema v3 binds:

- exact KWSP bytes;
- exact checkpoint;
- training/export token mappings;
- frontend identity/spec;
- training hyperparameters/environment;
- quantization diagnostics;
- canonical real training-corpus identity.

Training and release token files may differ byte-for-byte only when their canonical token→ID mapping/fingerprint is identical.

## 5. Compile and freeze KWKP v3

```bash
python3 tools/compile_keywords.py \
  --tokens release/tokens.txt \
  --keywords release/keywords.tsv \
  --out-pack release/xiaowo.kwk \
  --out-json release/keywords.json
```

A pack change creates a new release tuple even when `.kwm` is unchanged. Shared-prefix phrases must use explicit `min_trailing_blanks`, priority and `immediate`/`longest`/`grace` policy as appropriate.

## 6. Bind the final audio-pipeline/AFE

The final KWS threshold and acoustic qualification must use the exact shipping audio chain. When the domain renderer invokes `audio-pipeline` through the command backend, retain the executable/config SHA256 and reported latency. AFE latency is part of wake-latency accounting.

Any change to shipping microphones, enclosure, BF/AEC/RES/NS/AGC configuration, gain policy or AFE executable creates a new qualification tuple.

## 7. Run held-out continuous audio

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

Evaluation provenance schema v2 reopens every referenced WAV and binds file SHA256, decoded PCM SHA256, frames and canonical corpus SHA256. Every `references.duration_s` must equal the real WAV duration derived from frames/sample-rate. Declaring a one-second WAV as many hours of negative exposure is rejected.

Shipping FAR/hour therefore comes from the real continuous WAV bytes, not from a self-reported duration field.

## 8. Real far-field coverage

A product 3–5 m claim requires genuine human positives through the final microphones/enclosure/AFE. Synthetic distance rendering, TTS or measured-RIR/TTS data remain useful development evidence but cannot substitute for real held-out far-field positives.

At minimum cover:

- near/mid/far distance buckets including the claimed 3–5 m region;
- expected azimuths;
- representative RT60 and SNR;
- normal home conversation and near-homophones;
- TV/phone/smart-speaker playback;
- local-device TTS/music through shipping AEC;
- double-talk/AEC residual;
- motor/fan/gear/pump/chassis noise;
- moving/static device states;
- final AGC/AEC/NS modes.

## 9. Physical target-board benchmark

Cross-build `kws_board_bench` with the real target toolchain and run the exact `.kwm/.kwk`:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Retain the target executable. The summary binds runner/model/pack/audio and records mean/p50/p95/p99/max process time, RTF and p99 scheduling headroom. Hosted x86 or ARM cross-build success is not target-board timing evidence.

## 10. Capture target evidence with the repository collector

Use the exact retained collector from the released source tree:

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --soak-hours 24 \
  --cpu-percent <measured> \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --power-raw qualification/power.csv \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

The collector records machine/kernel/CPU/governor/RSS/thermal identity where available and SHA256-binds raw evidence files. External instrument measurements must retain instrument/calibration identity and raw measurement bytes. Do not type a final power/CPU/thermal claim into an otherwise unbound JSON and call it qualified evidence.

## 11. Build the byte-complete manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.jsonl \
  --dataset-audit qualification/dataset-audit.json \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --eval-audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --evidence-collector qualification/collect_target_evidence.py \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

Repeat `--training-manifest` and `--raw-evidence` for every participating artifact.

The verifier independently checks the deployable ABIs, vocabulary/frontend lineage, actual training WAV identity, dataset-audit coverage, actual held-out WAV identity and duration, evaluation formulas/provenance, board benchmark formulas/artifacts, evidence collector identity and raw target evidence hashes.

## 12. Apply the SKU policy

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Policy schema v2 gates point estimates and one-sided statistical confidence bounds for FAR/FRR, plus latency, p99 processing time, RTF/headroom, minimum evidence exposure, soak, CPU, RSS, stack, temperature and power.

Exit codes:

- `0`: structurally valid evidence and all policy gates passed;
- `1`: structurally valid evidence but one or more SKU thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy.

The gate result is schema v3 and binds the exact manifest/policy identity.

## 13. Retain the complete release tuple

Retain together:

- source SHA and released version/tag;
- `.kwm` + model provenance schema v3;
- source checkpoint and training image digest;
- training vocabulary/manifests and original training corpus bytes or immutable storage identities;
- dataset audit report;
- release vocabulary and KWKP v3;
- shipping runtime/AFE config and exact AFE executable/config identity;
- measured RIR manifest/RIR hashes if used;
- exact evaluation runner/references/original held-out WAVs/detections/provenance/summary/domain metrics;
- exact target benchmark runner/board WAV/board summary;
- exact target evidence collector, target evidence JSON and every raw evidence file;
- approved SKU policy;
- qualification manifest + gate result;
- final distribution bundle hash/signature/attestation.

## Software baseline vs shipping qualification

A green v0.3 repository can be a mature software baseline without being an acoustically qualified Mandarin SKU. CI proves parser/runtime contracts, synthetic regressions, corpus/evidence integrity mechanisms, cross-build compatibility, coverage/static-analysis gates and reproducible SDK output. It cannot prove final Mandarin FAR/FRR, real 3–5 m wake performance or physical Cortex-A32 CPU/thermal/power behavior.

Issue #2 remains open until independent human/device acoustic evidence and physical target-board evidence exist and pass the approved SKU policy.
