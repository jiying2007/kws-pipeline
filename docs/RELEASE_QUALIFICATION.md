# Release qualification

Repository CI proves software contracts. A shipping wake-word claim requires a separate artifact-bound qualification bundle built from the exact model lineage, original audio bytes, final runtime/AFE artifacts and trusted physical-target evidence.

`v0.3.x` deliberately hard-cuts the evidence schemas. Older qualification manifests/provenance are not accepted as v0.3 shipping evidence.

## v0.3 contracts

- deployable model: **KWSP ABI v2**;
- deployable keyword pack: **KWKP ABI v3**;
- frontend lineage: **frontend-spec v2**;
- model provenance: **schema v3**;
- evaluation provenance: **schema v2**;
- dataset audit: **schema v3**;
- runtime-soak evidence: **schema v2**;
- target evidence: **schema v2**, `evidence_class=product-board`;
- attestation verification: **schema v1**;
- qualification manifest: **schema v2**;
- qualification policy: **schema v2**;
- qualification gate result: **schema v3**.

The software version may change while KWSP/KWKP remain unchanged. v0.3 changes public discontinuity/evidence contracts, not the on-device model/keyword binary layouts.

## Evidence graph

```text
digest-pinned training image
 + training manifests + every selected training WAV byte identity
 + clean dataset audit covering the exact final references.jsonl
        |
        v
train_ctc.py -> checkpoint + canonical training corpus identity
        |
export_model.py -> KWSP v2 + model provenance schema v3

KWSP + KWKP + exact runtime/AFE
        |
        +-> exact eval runner + references + every held-out WAV
        |       -> eval provenance schema v2 + metrics
        |
        +-> exact target board runner + representative board WAV
        |       -> board timing summary
        |
        +-> actual product process under collect_runtime_soak.py
        |       -> runtime-soak schema v2
        |
        +-> stack/power/other raw measurement files
                 |
          canonical evidence-raw.jsonl
                 |
          trusted attestation verification
                 |
                 v
       collect_target_evidence.py
       -> product-board evidence schema v2
                 |
                 v
       qualification_manifest.py schema v2
                 |
        shipping-approved SKU policy v2
                 |
                 v
       qualification_gate.py -> result schema v3
```

## 1. Freeze the training environment

`training/Dockerfile` performs no network package installation. The base must be an immutable OCI reference such as:

```text
registry.example/kws-training-base@sha256:<64 lowercase hex>
```

Build the repository wrapper with `training/build_container.py`; shipping training may require `KWS_TRAINING_IMAGE_DIGEST` so checkpoint provenance carries the final environment identity.

The real `torch_ctc` integration workflow uses repository variable **`KWS_TRAINING_IMAGE`** or the manual `training_image` input. It only accepts `name@sha256:<digest>`.

## 2. Bind the real training corpus

`train_ctc.py` accepts TSV (`WAV<TAB>token_ids`) or JSONL. Human qualification projects should use JSONL identity metadata:

```json
{"audio":"audio/u001.wav","tokens":[1,2,3],"speaker_id":"spk001","session_id":"s01","source_id":"src001","room_id":"living-room-a","device_id":"robot-a"}
```

At checkpoint save time the trainer reopens every WAV and records file SHA256, decoded mono-16-kHz PCM16 SHA256, frame count/duration, stable metadata and canonical whole-corpus SHA256. Replacing a WAV underneath an unchanged manifest changes the model lineage.

## 3. Audit the exact final held-out manifest

The final dataset audit must include the **same `qualification/references.jsonl`** later passed to `qualification_manifest.py`:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=qualification/references.jsonl \
  --audio-root qualification=qualification/audio \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json \
  --fail-within-split
```

Final held-out recordings must never be mined into replay/tuning and then reused as unbiased evidence.

## 4. Export and freeze the model and KWKP

```bash
python3 training/export_model.py \
  --checkpoint release/base.pt \
  --tokens release/tokens.txt \
  --output release/base.kwm

python3 tools/compile_keywords.py \
  --tokens release/tokens.txt \
  --keywords release/keywords.tsv \
  --out-pack release/xiaowo.kwk \
  --out-json release/keywords.json
```

Model provenance schema v3 binds the exact model bytes, checkpoint, token mappings, frontend identity/spec, training environment and canonical real training-corpus identity. Any model or keyword-pack change creates a new qualification tuple.

## 5. Bind the final AFE

Final thresholds and qualification must use the exact shipping microphones, enclosure and audio-pipeline configuration. Retain executable/config SHA256 and reported latency. Any BF/AEC/RES/NS/AGC/gain-policy or physical sound-path change creates a new qualification tuple.

## 6. Run held-out continuous audio

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
```

Evaluation provenance schema v2 reopens every referenced WAV and binds file SHA256, decoded PCM SHA256, frame count and canonical corpus SHA256. Every declared `duration_s` must equal real WAV duration, so FAR exposure cannot be inflated by metadata alone.

## 7. Real far-field coverage

A 3–5 m product claim requires genuine human positives through the final microphones/enclosure/AFE. Synthetic distance rendering, TTS or measured-RIR/TTS remain development evidence only.

At minimum real qualification should include near/mid/far distance, expected azimuth, RT60/SNR, home conversation, near-homophones, TV/phone/smart-speaker playback, local-device playback through shipping AEC, double-talk/AEC residual, motor/fan/gear/pump/chassis noise, moving/static states and final AGC/AEC/NS modes.

## 8. Physical target-board timing

Run the exact target binary/model/pack and retain all bytes:

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json
```

Hosted x86 or cross-build success is not physical target timing evidence.

## 9. Measure the actual runtime process

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Runtime-soak schema v2 retains child-process CPU/RSS/thermal samples and fails on early exit. Summary CPU/RSS/temperature values are independently recomputed from those samples during later verification.

## 10. Freeze canonical raw evidence

Stack high-water remains product-harness specific and must have retained raw evidence. Power requires a raw instrument trace plus instrument/calibration identity.

After acquisition, freeze the exact selected files in `qualification/evidence-raw.jsonl`. Each JSONL row is:

```json
{"name":"runtime-soak.json","sha256":"<64 lowercase hex>","bytes":12345}
```

The row set must exactly match runtime-soak + every `--raw-evidence` + `--power-raw`. Extra, missing, duplicate or mismatched rows are rejected.

## 11. Verify the product-board attestation

The product qualification trust layer must produce `qualification/attestation-verification.json`. This is not generated by the target-evidence collector. Schema v1 must report `verified=true`, identify a trusted issuer/policy, use a UTC verification timestamp and bind:

- canonical `evidence-raw.jsonl` SHA256 as `subject_sha256`;
- exact `collect_target_evidence.py` SHA256;
- exact board runner SHA256;
- exact model SHA256;
- exact keyword-pack SHA256.

The repository verifies this result and its artifact bindings; organizational trust in the issuer/policy belongs to controlled qualification/signing infrastructure.

## 12. Assemble product-board evidence

Builder and DUT identities must be distinct. Use the complete v0.3 CLI:

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --soc cortex-a32 \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --board-runner qualification/kws_board_bench.target \
  --model release/base.kwm \
  --keyword-pack release/xiaowo.kwk \
  --board-audio qualification/board-audio.wav \
  --sku product-sku-a \
  --source-sha "$(git rev-parse HEAD)" \
  --builder-id qualification-builder-01 \
  --dut-id product-dut-01 \
  --collector-id qualification-station-01 \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

Target evidence schema v2 is accepted as shipping resource evidence only when `evidence_class=product-board` and all SKU/source/artifact/raw/attestation bindings agree.

## 13. Build the byte-complete qualification manifest

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest data/train.jsonl \
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
  --evidence-collector tools/collect_target_evidence.py \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --sku product-sku-a \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json
```

Repeat `--training-manifest` and `--raw-evidence` as needed. The verifier independently checks deployable ABIs, vocabulary/frontend lineage, real training WAV identity, exact dataset-audit coverage, held-out WAV identity/duration, evaluation formulas, board benchmark formulas/artifacts, product-board SKU/source identity, canonical raw-evidence identity and attestation bindings.

## 14. Apply the shipping-approved SKU policy

```bash
python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Policy schema v2 must identify the same SKU and set `shipping_approved=true`. It gates FAR/FRR point estimates and one-sided confidence bounds plus latency, p99 process time, RTF/headroom, soak, CPU, RSS, stack, temperature and power.

Exit codes:

- `0`: structurally valid evidence and all policy gates passed;
- `1`: structurally valid evidence but thresholds failed;
- `2`: malformed, inconsistent or tampered evidence/policy.

Gate result schema v3 binds the exact manifest/policy identity.

## 15. Retain the complete release tuple

Retain:

- source SHA/version/tag and exact runtime build identity;
- training image digest, checkpoint, model provenance, original training manifests/audio identity;
- clean dataset audit;
- KWSP, KWKP, vocabulary and runtime config;
- final AFE identity;
- eval runner, exact references/original held-out WAVs, detections/provenance/metrics;
- board runner/audio/summary;
- runtime-soak raw trace, stack raw evidence, power raw trace, instrument/calibration identity;
- canonical `evidence-raw.jsonl`;
- attestation-verification result and trusted issuer/policy identity;
- SKU/source/builder/DUT/collector identities;
- exact target-evidence collector and evidence JSON;
- shipping-approved SKU policy, qualification manifest and gate result;
- final distribution checksums/SBOM/signing/attestation artifacts.

## Software baseline vs shipping qualification

A green `v0.3.0` repository and signed release can be a mature software/evidence-engineering baseline without being an acoustically qualified Mandarin SKU. CI proves parser/runtime contracts, deterministic/synthetic regressions, corpus/evidence integrity mechanisms, cross-build compatibility, coverage/static-analysis gates and reproducible SDK output.

It cannot prove final Mandarin FAR/FRR, genuine 3–5 m wake performance or physical Cortex-A32 product behavior. Issue #2 remains open only for those independent real product measurements.
