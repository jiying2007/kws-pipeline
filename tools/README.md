# Engineering utilities

This directory contains hosted/offline tools used to build and qualify the device runtime. Terminal-hardening utilities include:

- `corpus_identity.py`: canonical WAV/file/decoded-PCM identity primitives;
- `verify_corpus_identity.py`: verify retained corpus identity manifests against current bytes;
- `collect_target_evidence.py`: capture target system/resource identity and bind external power raw data;
- `collect_runtime_soak.py`: supervise long-running target qualification commands and retain liveness/RSS samples;
- `check_reproducible_sdk.py`: compare two independently installed SDK trees byte-for-byte;
- `check_repo_terminal_state.py`: verify a clean expected branch checkout for release automation.

These tools do not substitute for real product measurements. They make the measurements reproducible and auditable once collected.
