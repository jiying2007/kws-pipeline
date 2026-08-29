# Terminal hardening contract

This document records the final software-side gaps that must be closed before the repository may call a KWS release tuple terminal. It intentionally does not claim real Mandarin/device qualification before those measurements exist.

## Byte-complete corpus identity

A release tuple must bind the actual audio bytes used for training and qualification, not only the TSV/JSONL files that point at them. Every retained corpus identity records both the source WAV SHA256 and decoded mono-16-kHz PCM SHA256 so renamed or rewrapped duplicates remain detectable.

The binding chain is:

```text
training WAV bytes -> training corpus identity -> checkpoint -> model provenance
qualification WAV bytes -> evaluation provenance -> qualification manifest
```

Changing any audio byte changes the release evidence identity.

## Target evidence authenticity

Board metrics must be produced by the retained target collector whenever the platform exposes the metric. Free-form evidence JSON is not considered a terminal proof source by itself. External power measurements must retain instrument/calibration identity and a hash of the raw measurement file.

## Audio discontinuity

Capture XRUNs, route changes and clock resets are product events. The runtime must be explicitly notified so partial frontend frames, PCEN state, recurrent state and decoder state cannot bridge missing audio.

## Production training path

The lightweight prototype loop remains a fast deterministic CI regression. A lower-frequency integration workflow must also exercise the real `torch_ctc` train/export/quantize/runtime path end to end so the production backend cannot silently rot.

## Reproducibility

Training and release builders should use digest-pinned container images and hash-pinned dependencies. Deterministic archive metadata is not the same as a reproducible compiled binary; a release may claim bit-reproducible SDK binaries only after independent builds compare equal.

## Repository governance

Terminal product state requires protected release history. Single-maintainer operation may omit an approving reviewer, but should still require pull requests, required CI, no force-push/delete on `main`, immutable release tags and auditable administrative bypass.

## External product qualification

The repository cannot manufacture evidence for the shipping product. Real Mandarin held-out speech through the final microphones, enclosure and audio frontend, 0.3-5 m acoustic coverage, physical Cortex-A32 CPU/RSS/stack/thermal/power measurements and long-duration soak remain external product gates.
