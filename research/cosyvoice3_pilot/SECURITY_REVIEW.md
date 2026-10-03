# Bounded dependency security review

## Exact provenance

All 123 selected artifact identities were independently rechecked against current official PyPI exact-version JSON: filename, URL, SHA256, compressed size, and non-yanked status all agree. `advisory-inventory.json` contains the full version-matched advisory snapshot; `advisory-summary.json` is the compact shareable view. PyPI entries may duplicate one issue under GHSA/PYSEC aliases, and an empty list is not a security certification.

Transitive aiohttp was changed to 3.14.3 rather than inheriting vulnerable 3.13.5. Build/installer tools are pip26.2 and wheel0.46.2, which have no entries in this snapshot. These changes do not change the declared model graph. Current pip always uses PEP517, including its documented setuptools fallback for projects without pyproject.toml: https://pip.pypa.io/en/stable/reference/build-system/index.html . Its release history documents the removed legacy-build interface: https://pip.pypa.io/en/stable/news/ .

## Remaining older upstream-compatible roots

The fixed source/runtime proposal retains upstream-compatible roots that have advisories. They must remain confined to this exact official-source, local-model, offline synthetic-inference experiment. They are unsuitable for exposing a service or ingesting arbitrary configurations, checkpoints, datasets or code. A broad production-safety claim is not made.

- diffusers0.29.0: remote pipeline-loader code execution/trust_remote_code bypass. The pilot imports its model components; it must never invoke DiffusionPipeline.from_pretrained or custom pipeline loaders
- gdown5.1.0: archive extraction traversal. Imported by Matcha helpers; it must not acquire/extract any pilot data. The reviewed streaming acquirer is the only download path
- hydra-core1.3.2: arbitrary object construction from untrusted config. All model/YAML/config sources are exact pinned official bytes, reviewed before loading; no arbitrary user/_target_ overrides
- lightning2.2.4 and pytorch-lightning2.2.4: LightningApp endpoints, plugin extraction and unsafe checkpoint/object loading. No application/server, trainer or Lightning checkpoint loader is used; stock model state loading remains separately restricted and qualified
- modelscope1.20.0: configured arbitrary module execution. Its CLI import is needed, but local preverified model_dir must exist, and no ModelScope pipeline/remote loader is permitted
- pyarrow18.1.0: malformed Arrow IPC file prebuffering use-after-free. Imported by the unpruned training-module source path; the pilot does not read Arrow files
- setuptools81.0.0: manifest Unicode exclusion bypass when building/publishing sdists on normalization-preserving filesystems. This Linux runner builds wheels from four exact pre-reviewed sdists and publishes no package source distribution; Torch requires setuptools<82, so83 is not silently substituted
- torch2.12.1: GHSA-rrmf-rvhw-rf47/CVE-2025-3000, torch.jit.script memory corruption, fixed2.13. The pilot has no requested JIT compilation or JIT checkpoint route, and must not accept user-controlled scripted code. This does not invalidate the separate plain FP32 CPU operator qualification, but it is a retained known issue
- transformers4.51.3: various conversion/training/image/tokenizer and custom-loader issues, including GHSA-29pf-2h5f-8g72 for malicious config `_attn_implementation_internal`. `trust_remote_code=False` alone is insufficient protection. Before loading the pinned Qwen2 local model, reject dangerous attention/custom-code selectors and verify its complete config/tokenizer identities. No checkpoint converter, trainer, custom generation loader, image URL ingestion or tokenizer save_pretrained path is used

The locally reviewed pinned BlankEN config contains only standard Qwen2 numerical/architecture fields; no `_attn_implementation_internal`, auto_map, custom pipeline or custom generation key is present. The generation stage must recheck actual acquired bytes rather than trust this description. Its model/source/adapter gates are independently required before model execution.

## Installation controls

Official exact artifacts only; no alternative index/CPU host. SHA256 and byte length before use. No symlink/traversal archive members. METADATA identity/hash checks. Patched pip, disabled indexes, disabled self-update checks/caches, separate exact build environment, controlled compiler identity. Python network auditing blocks source-build networking; this is not kernel-level network isolation. No source backend is imported locally during preparation.

Assertions enforce many qualification gates, so both runtime entrypoints explicitly refuse an optimized Python interpreter before side effects. The installer clears inherited PYTHONOPTIMIZE/PYTHONPATH/PYTHONHOME, pip destination/index overrides and configuration. The qualifier checks distribution locations inside the isolated venv and rejects duplicate package identities. Resource watchdogs and remote qualification still determine whether this candidate can proceed.
