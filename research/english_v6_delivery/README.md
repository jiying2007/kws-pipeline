# Hey Mero software candidate delivery

This is a separate research software installation path. `selection.json` is
currently inert: both `selected` and `manifest` are null. There is no candidate
pack in this directory. Original B0 sources, packer, model pins and historical
evidence are unchanged.

## Authority and final binding

The trusted Git selection pins an immutable delivery manifest by its exact
bytes and SHA-256. A downloaded manifest, a `PASS` string, an installation
receipt or a loader result cannot replace that trust root. Before changing the
selection, the independent original-evidence reader must authenticate both
quality panels and all original engineering evidence, including all 140 saved
outputs, 84 native event CSVs, the original producer/Git identities and actual
pack/SDK bytes. This delivery utility does not repeat or replace that evaluator.

After those checks have actually passed:

1. Retain the original four engineering ZIPs as `result.zip`, `producer.zip`,
   `consumed.zip`, and `input_proof.zip`, together with the exact complete
   readback report `completion.json` and its actual `readback.py` source.
   These original files are local packaging inputs, not public Git payloads.
2. Create the manifest required by `manifest_contract()` in `delivery.py`.
   Its top-level fields are `schema`, `candidate_id`, `pipeline_commit`,
   `candidate_pack`, `files`, `engineering`, `quality_locks`, `readback`, and
   `scope`. Use schema `heym43-software-delivery-v1`. Copy `scope` exactly from
   the inert selection. `quality_locks` contains the complete original primary
   and replication locks, not result summaries. `engineering` holds repository
   `jiying2007/kws-data`, all four artifact pins (`id`, `run_id`, `run_head`,
   `bytes`, `sha256`), and original `RESULT.json`/`MANIFEST.json` SHA-256 values.
   `readback.report` and `readback.verifier` each pin exact `bytes` and `sha256`.
3. Populate `files` with every original `models/` and `sdk-{gcc,clang,cortex-a32-qemu}/`
   member plus `LICENSE`, `README.md`, and the separately authenticated
   `models/B0.pack`. Each entry contains `bytes`, `sha256`, and integer `mode`.
   Exactly the SDK `bin/heym43-live` and `bin/heym43_host_cli` files use 493
   (`0755`); all other files use 420 (`0644`). The fallback pack must have
   SHA-256 `ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b`
   and exactly 216,960 bytes. `candidate_pack` names the unchanged original
   engineering pack member. The pipeline commit remains
   `88a043c6c00bb3a1a62f45098a023c7601042ded`.
4. Store that manifest as `manifests/<its-sha256>.json`; update `selection.json`
   with its path/bytes/SHA and matching `candidate_id`. Preserve prior immutable
   manifests and their real artifacts. Do not enable the selection from a
   successful training or source-only result.

`package` verifies the pinned readback source/report, whole original ZIP
identities and CRCs, the complete engineering file manifest, and every copied
model/SDK byte. It copies existing packs; it does not construct new weights or
run any model. The tar contains only the selected software payload and the
minimal delivery manifest, never full API records or raw readback inputs.

```sh
python3 -I -S -B research/english_v6_delivery/delivery.py check
python3 -I -S -B research/english_v6_delivery/delivery.py package \
  --originals /absolute/original-engineering \
  --baseline-pack /absolute/authenticated-B0.pack --output /absolute/delivery.tar.gz
python3 -I -S -B research/english_v6_delivery/delivery.py install \
  --archive /absolute/delivery.tar.gz --prefix /absolute/heym43
python3 -I -S -B research/english_v6_delivery/delivery.py verify-installed \
  --prefix /absolute/heym43
```

Installation creates `PREFIX/versions/<delivery-manifest-sha256>/`. It validates
the entire archive before writes, refuses links/unlisted files, preserves an
existing identical version, and checks every installed byte and mode. The
0755/0644 modes are an explicit reviewed prescription: GitHub upload ZIP modes
are not claimed to preserve executable permissions. No running process is
changed by packaging or installation.

## Installed pack loader check

After installation, build this separate clean consumer against the verified
installed SDK. This does not change or rebuild the frozen SDK sources. Supply
the exact version path printed by `verify-installed`:

```sh
cmake -S research/english_v6_delivery -B /absolute/loader-build \
  -DHeyMero43_DIR=/absolute/heym43/versions/MANIFEST_SHA/sdk-gcc/lib/cmake/HeyMero43
cmake --build /absolute/loader-build --parallel 2
python3 -I -S -B research/english_v6_delivery/delivery.py check-load \
  --prefix /absolute/heym43 --model candidate --backend gcc \
  --checker /absolute/loader-build/heym43_delivery_load_check
```

The C program reads exactly one 216,960-byte pack, compares installed-library
and header storage sizes and calls `en_v5_m43_model_load()` once. It never
initializes a stream, reads PCM or evaluates a frame. Its output describes a
format/loading check, not independent acoustic, physical or product evidence.
Use the binary built from this reviewed consumer source; an arbitrary program
that prints similar JSON is not a loader verification.

## Explicit startup and B0 rollback

`launch` verifies the installed version again and passes the selected exact pack
path as the live adapter's first argument. Both decoder settings are mandatory
caller inputs; this layer supplies no calibrated defaults.

```sh
python3 -I -S -B research/english_v6_delivery/delivery.py launch \
  --prefix /absolute/heym43 --model candidate --backend gcc \
  --max-gap-samples CALLER_VALUE --cooldown-samples CALLER_VALUE
```

To roll back, stop the existing consumer and run the same command with
`--model B0`. This starts clean model/state memory. There is no hot swap,
automatic fallback, silent process termination or implicit reuse of an active
stream. ARM/QEMU SDK files are delivered as authenticated software artifacts;
this launcher does not choose an emulator or establish board compatibility.

Physical SSC305 execution, field FRR, continuous FAR/h and physical latency
remain NOT_RUN. Product promotion and CAL access stay false. No Release
publisher or automatic deployment workflow is added.

## Source verification

`test_delivery.py` uses temporary invented bytes and mocked process entrypoints
only. It checks closed gates, origin/manifest tampering, exact inventories,
safe extraction, immutable installs, explicit model paths, permissions and the
loader's no-forward source boundary. It does not compile or execute a consumer,
load real weights, read audio, or contact a service. Actual installed candidate
loading remains a separate post-qualification check described above.
