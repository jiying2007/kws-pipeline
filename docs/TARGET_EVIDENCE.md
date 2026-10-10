# Target evidence collection

Target-board qualification must retain raw, machine-collected evidence alongside summarized SKU metrics. v0.3 separates **runtime soak acquisition**, **raw evidence identity**, **external attestation verification** and **final target evidence assembly** so a JSON summary cannot silently substitute for the measured artifacts.

This document describes the software evidence contract. Hosted fixtures exercise the contract, but they are not physical product evidence.

## Runtime timing

Cross-build and retain `kws_board_bench`, then run the exact shipping `.kwm/.kwk` and representative post-AFE WAV. Preserve the emitted JSON plus the exact benchmark executable, model, keyword pack and board-audio bytes.

`kws_board_bench` measures per-hop processing time/RTF/headroom. It does not replace sustained product-process CPU/RSS/thermal/soak evidence.

## Runtime soak

Run the actual product/KWS qualification process under `tools/collect_runtime_soak.py`:

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

Runtime-soak schema v3 records requested/actual duration, early-exit state, child PID/command, max child-process RSS, aggregate process CPU time, observed thread counts, thermal samples/max temperature and the retained raw time series. If the process exits before the requested duration, the collector fails.

### CPU units and unavailable audio exposure

All CPU gates use the fixed `measurement_contract_id=process-cpu-one-core-v1` and `cpu_percent_semantics="process_cpu_seconds / wall_seconds * 100"`:

- `process_cpu_seconds` is the final minus initial `/proc/<pid>/stat` user + system CPU time, summed over that process's threads. Descendant processes are outside this sampler's scope; run the actual product process directly, not a launcher that forks the workload away.
- `wall_seconds` equals `elapsed_seconds` and the final sample's `elapsed_s`. Termination and cleanup waits are excluded from the denominator.
- `average_cpu_percent` in the raw soak becomes `cpu_percent` in evidence and scores. 100% means one occupied core, 150% means 1.5 cores. Values are not clipped to 100 and are never divided by online cores, affinity, configured threads or measured threads.
- `cpu_capacity_count` is descriptive topology only. Two-core hardware using 20 CPU-seconds in 100 wall-seconds measures 20%, so a 10% budget FAILS.
- Every sample retains an actual `/proc/<pid>/status` `thread_count`; `max_thread_count` is the maximum observed at sample times, not a guarantee about threads between samples.
- This sampler has no retained audio sample counter. `audio_seconds` and `cpu_seconds_per_audio_second` must both be null; wall time is not audio exposure. A future CPU/audio-second measurement must bind real processed-sample counters and their sample rate. `kws_board_bench` separately reports processing-time RTF against its known input WAV duration/repeats; that audio duration cannot be reused for the soak.

Collection and independent qualification both recompute the same schema-v3 contract. Missing/reset CPU counters, missing/invalid thread counts, inconsistent endpoint durations or changed units fail closed. Runtime-soak v2 and target-evidence v2 used online-capacity units and are rejected, even on a single-core host. Do not relabel historical evidence or simply change its version/semantics string; recollect and attest the current contract. Historical archived artifacts retain their original bytes and authority boundaries.

Also retain audio-pipeline XRUN/backpressure counters and `kws_engine_get_stats()` discontinuity snapshots when the SKU requires them.

## Stack and power evidence

Generic Linux does not provide a portable, trustworthy stack high-water metric for an arbitrary product thread. `stack_high_water_bytes` therefore remains an approved product-harness measurement and must have retained raw evidence.

Power normally comes from an external instrument. Preserve its original CSV/trace, instrument ID and calibration ID. A summarized average-power number is not accepted without the raw instrument file.

## Canonical raw-evidence manifest

Before assembling target evidence, freeze the exact raw files selected for qualification. `--evidence-raw` is a JSONL manifest containing exactly one row per retained raw artifact:

```json
{"name":"runtime-soak.json","sha256":"<64 lowercase hex>","bytes":12345}
{"name":"stack-watermark.txt","sha256":"<64 lowercase hex>","bytes":456}
{"name":"power.csv","sha256":"<64 lowercase hex>","bytes":789}
```

The set must exactly equal:

- the runtime-soak JSON;
- every repeated `--raw-evidence` file;
- the `--power-raw` file.

Release and Phase-B qualification also require the exact schema-v2 `board-summary.json` as a raw artifact, binding timing/workload results through this attested manifest. Phase-B bundles retain byte-identical copies at `board-summary.json` and `raw/board-summary.json`. Duplicate names, missing rows, extra rows, size mismatches or hash mismatches are rejected. This manifest should be emitted by the controlled qualification harness after measurements are frozen.

## External attestation verification

`--attestation-verification` is the output of the product qualification trust layer, not a self-attestation generated by `collect_target_evidence.py`. Schema v1 must report `verified=true` and bind the exact selected artifacts **and the full final-AFE identity inherited from Phase A**:

```json
{
  "schema_version": 1,
  "verified": true,
  "subject_kind": "kws-target-evidence",
  "issuer": "<trusted verifier>",
  "trust_policy": "<approved policy id>",
  "verified_at_utc": "2026-08-30T00:00:00Z",
  "subject_sha256": "<evidence-raw.jsonl sha256>",
  "collector_sha256": "<collect_target_evidence.py sha256>",
  "board_runner_sha256": "<kws_board_bench sha256>",
  "model_sha256": "<base.kwm sha256>",
  "keyword_pack_sha256": "<xiaowo.kwk sha256>",
  "audio_frontend_identity_sha256": "<Phase-A final_afe_identity_sha256>"
}
```

The repository independently parses this verifier result during Phase-B scoring and checks every binding again. The full AFE identity is content-addressed from executable/config bundle/command template/SKU/microphone/enclosure/audio route/toolchain, so reusing the same executable with a different product configuration cannot pass as the Phase-A tuple.

Production trust in `issuer`/`trust_policy` belongs to the controlled qualification/signing infrastructure.

## Assemble product-board evidence

Run the exact retained collector with explicit SKU, source, builder and DUT identity. Builder and DUT IDs must be distinct.

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --soc cortex-a32 \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <final-afe-executable-sha256> \
  --audio-frontend-identity-sha256 <phase-a-final-afe-identity-sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/board-summary.json \
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

The collector emits target evidence schema v3 with `evidence_class=product-board`. It derives soak hours, CPU, RSS and max temperature from the retained runtime-soak bytes, embeds those exact soak bytes for independent recomputation, and binds:

- SKU and exact source SHA;
- builder/DUT/collector identity;
- collector hash;
- canonical raw-evidence manifest hash;
- external attestation-verification hash;
- board runner/model/keyword-pack/board-audio hashes;
- final-AFE executable SHA and full Phase-A final-AFE identity SHA;
- runtime-soak, power and additional raw artifacts;
- target/board/SoC/toolchain/kernel/governor/AFE identity;
- stack, power, instrument and calibration identity.

## Qualification binding

`qualification_manifest.py` must receive the same evidence trust tuple and every raw artifact:

```bash
--evidence qualification/evidence.json \
--evidence-collector tools/collect_target_evidence.py \
--evidence-raw qualification/evidence-raw.jsonl \
--attestation-verification qualification/attestation-verification.json \
--raw-evidence qualification/runtime-soak.json \
--raw-evidence qualification/stack-watermark.txt \
--raw-evidence qualification/board-summary.json \
--raw-evidence qualification/power.csv \
--sku product-sku-a
```

The manifest independently re-hashes those files and verifies that the external attestation, canonical raw manifest, collector output and selected release artifacts all describe the same SKU/source tuple.

The Phase-B `target-dut-qualification` scorer adds a stricter product binding on top: it requires the externally attested `audio_frontend_identity_sha256` to equal the immutable Phase-A `final_afe_identity_sha256` before any physical DUT can qualify.

## Trust boundary

SHA256 binding prevents accidental substitution and summary-only tampering; it does not by itself establish who performed a physical measurement. Production authenticity still depends on controlled DUT inventory, qualification stations, verifier/signing identity and release/OTA trust roots.

No hosted fixture, synthetic audio run or cross-compile result should be labeled `product-board` evidence for a shipping claim.
