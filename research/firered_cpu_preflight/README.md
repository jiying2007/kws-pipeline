# FireRed CPU dependency-only preflight

This separate research task qualifies only an isolated Python3.11 CPU dependency environment. It does not download/load ASR model weights, audio, labels or existing transcripts. It does not change the Qwen/SenseVoice environment, score model quality, merge or deploy anything.

Fourteen exact package versions are proposed. Twelve PyPI wheel hashes are already pinned. The two Torch2.10.0+cpu / TorchAudio2.10.0+cpu hashes are deliberately not invented: the preflight obtains them from their official CPU-index entries and verifies both downloaded wheel files before installation. It verifies wheel METADATA identities and active dependency closure, installs only local wheels into a new venv with no dependency resolver, and performs pip-check plus tiny CPU/NumPy/fbank checks. No CUDA/NVIDIA/Transformers packages or source builds are included. The resulting receipt must be reviewed before a complete runtime lock or model qualification is claimed.

The optional job is one owner-labelled run on the fixed research branch and exact PR-head marker. Ordinary PR events run only offline fixture tests. It uses the free public standard ubuntu-24.04 runner, read-only repository access, credential-free checkout, and a new temporary venv. Bounds:20-minute job;15-minute supervised process group;3GiB sampled RSS;2GiB initial available memory;5GiB initial free disk. The supervisor is a sampled guard, not a kernel memory sandbox. Wheel bodies and the venv are not uploaded; only bounded JSON/METADATA receipts are retained for7days.

Run local fixtures with: python3 -I research/firered_cpu_preflight/test_preflight.py

No model inference can be launched by this folder. A subsequent FireRed model experiment needs separately reviewed source/assets, actual runner resources, strict checkpoint load/readback and the latest human-label overlay. A corrected019 label is not part of this dependency-only task.
