# CosyVoice3 exact CPU-pilot dependency input gate

This directory defines the CPython 3.12.14, Ubuntu 24.04 x86_64 input closure and its admitted-runner qualification path. It is not a claim that CosyVoice3 has already run successfully.

## Runner interface

After the parent workflow's exact-head/source/budget/publication admission:

```sh
python runtime_lock.py --output "$RUNNER_TEMP/runtime-input-lock.json"
python install_runtime.py --lock "$RUNNER_TEMP/runtime-input-lock.json" --work "$RUNNER_TEMP/cosyvoice-runtime" --receipts "$RUNNER_TEMP/cosyvoice-receipts"
"$RUNNER_TEMP/cosyvoice-runtime/venv/bin/python" qualify_runtime.py --lock "$RUNNER_TEMP/runtime-input-lock.json" --output "$RUNNER_TEMP/cosyvoice-receipts/runtime-qualification.json"
```

The installer requires GITHUB_ACTIONS=true, GITHUB_REPOSITORY=jiying2007/kws-pipeline, Python3.12.14/Linux/x86_64, an inherited g++ compiler, and a fresh directory strictly inside RUNNER_TEMP. It creates an isolated final venv and a separate locked builder venv. It never changes an existing qualified environment or installs system packages.

`runtime-lock.json.xz.b64` is a lossless transport encoding of the exact reviewed 734,574-byte `runtime-lock.json` (SHA256 `c9a2a0db05af4d90861cbab7f06837007a74c6a3b82d255b9fad231e0bc00406`). The stdlib decoder enforces that original identity and bounded decompression, and the controller writes the unchanged lock only into its new job directory. This packaging avoids oversized publication API payloads; no pin, artifact, dependency edge or runtime behavior changes.

The decoded file is an exact upstream input lock: each artifact has name/version, official PyPI URL, SHA256, compressed bytes, requirements, marker/extras closure and target compatibility. Four tiny sdists are pinned inputs; their derived wheel hashes cannot be known before the actual pinned-input build. Those realized hashes, METADATA, expansion and input linkage are recorded in `realized-wheel-lock.json`. There is no source fallback for any other package.

Artifacts are acquired and installed sequentially, with no persistent cache. Wheel METADATA is checked against reviewed sidecar hashes when available, and paths/expanded allocation are checked before installation. Package installs use --no-deps only because the entire exact closure is already locked; they do not waive final pip check or independent installed-metadata validation. Source builds use the exact builder closure, no index, no build isolation and Python-network audit blocking. Only the installer's own successful ephemeral staging/build paths are deleted.

## Receipts and interpretation

- `install-receipt.json`: installed_pending_offline_qualification or failed; input lock SHA256; compiler identity; exact download verification; command logs; observed install disk peak/free reserve
- `realized-wheel-lock.json`: every installed official/source-built wheel identity and SHA256, exact source input link, expanded bytes and metadata
- `runtime-qualification.json`: status qualified only after every runtime check passes; schema cosyvoice3.runtime-qualification.v1, input lock SHA256, interpreter identity, exact versions, pip_check.returncode=0, all installed active dependency edges, CPU/API tests and empty network_attempts

Qualifier tests: isolated interpreter + exact closed package set; installed METADATA checks; all direct distribution imports; CPU-only Torch; deterministic TorchAudio resampling/fbank; Whisper128-mel finite shape; pyworld native dio/stonemask; SoundFile PCM16 roundtrip; tiny randomly initialized Qwen2 FP32 forward; ORT CPU provider availability; HyperPyYAML constructor. No model weights, source checkout, original audio or external service is opened. No inference quality claim follows from these tests.

The generation stage separately qualifies pinned CosyVoice/Matcha source and the SoundFile adapter binding, then measures source/reference-inclusive disk and memory admission before model acquisition. A runtime-qualification failure blocks model acquisition. No alternate package pin or repeat generation is an automatic recovery action.

## Resource gates

During install, reserve 1,073,741,824 free bytes and a 14,000,000,000-byte effective job ceiling. Before each wheel install, reserve its actual archive expansion rounded to the measured filesystem allocation block plus 64 MiB allowance; before source build, reserve expanded source plus 512 MiB scratch allowance. Monitor whole-filesystem free space on each download chunk and every 0.2 seconds of subprocess execution. On a local failure, terminate pip and fail the phase; the outer controller must terminate its entire inherited process group, including compilers. No installer subprocess creates a new session. Per-command log files are hard-capped at 1 MiB; the aggregate receipt/log budget is 16 MiB, including a 1 MiB reserved allowance for final JSON receipts. Record observed peaks honestly: the sampler is not proof of every between-sample transient.

The runtime closure's full expanded peak is not inferred from compressed sizes. Even successful dependency installation does not admit the 5,427,029,103-byte model. After source and reference staging, the next gate must have at least 6,635,020,288 bytes free (adjust per-file model allocation to the measured block size), including 128 MiB output allowance and untouched 1 GiB reserve. The installer neither downloads models nor cleans unrelated runner files.

## Static verification and provenance

`test_dependency_gate.py` exercises wheel path/hash/metadata guards and the non-runner qualification guard without importing/installing candidate packages. `finalize_lock.py` independently recomputes active marker/extras constraints, checks target wheel tags, confirms reused Torch2.12.1/TorchAudio2.11.0/ORT1.30.0 identities and exact reviewed sdist hashes, and writes `LOCK-VALIDATION.json`.

Raw metadata JSON/sidecars, sdist archives and locally generated caches are research inputs, not required public payload. Publish only reviewed scripts, the final runtime lock, static validation, this README, source review/receipts and selected compact advisory inventory. The runtime installer acquires official inputs on the admitted runner. No CPU-index-host fallback is used.
