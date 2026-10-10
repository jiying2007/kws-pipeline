# Qwen16 actual result · 2026-10-10

**Overall workflow: FAILURE. TTS completed; heterogeneous ASR stopped after a partial result.**
[Run 38018029787](https://github.com/jiying2007/kws-pipeline/actions/runs/38018029787)
executed head [`621ba90acd2d78976f4dcf07a385cd4e615ac358`](https://github.com/jiying2007/kws-pipeline/tree/621ba90acd2d78976f4dcf07a385cd4e615ac358).
All generated audio and raw terminal evidence are retained here, including failures.

The [source README](../../README.md), plan and readiness files are the **historical
pre-execution preparation snapshot**, not the latest execution status. Their 66
frozen files are preserved by the
[historical execution freeze](../../history/executed-v3-621ba90/execution-freeze.json)
and [reconstruction map](../../history/executed-v3-621ba90/snapshot.json); its SHA-256 is
`abfc1121ed82367eb8d18fb79ad28911a8453b867be49298ac3cfd93e5ebf2f2`.
The [later inert software repair](../../ASR_REPAIR_2026-10-10.md) has a separate
current source freeze; it does not rewrite this run. This dated result page and
derived summary sit outside the historical freeze. No active
workflow or approved release is added to main by evidence retention.

## Outcome and limits

- Qwen VoiceDesign: **16/16 generated, exactly one attempt each**, two fixed generic
  adult designs × eight fixed phrases. All 16 reported framework EOS, with no
  native/derived quality flags. Acoustic completeness remains **UNKNOWN**.
- FireRed TTS: **16/16 NOT_RUN, zero attempts**. Its free-CPU route remains blocked.
- Qwen-ASR-0.6B: **16 successful one-attempt decodes**.
- SenseVoice: the first opaque clip consumed **one failed attempt** (`ValueError`,
  model forward completion unconfirmed); **15 NOT_RUN**. Its container exited 1,
  reported no OOM, and completed cleanup. The precise exception message was not
  retained; the artifact alone does not establish its root cause.
- **0/16 successful two-recognizer pairs**. All 16 clips remain unresolved;
  Whisper was disabled and made zero calls. No retry or regeneration occurred.
- Human actual-word review: **PENDING for all 16**. Human gold, training admission
  and shipping approval remain false. D20 remains FAIL; D90 remains NOT_RUN.

These are synthetic source-screen observations. They do not establish human word
labels, acoustic truth, target/negative admissibility, FAR/FRR, unseen-voice
coverage, or physical-board qualification.

## Single-ASR lexical observations

The following table was joined **after** the raw ASR results were frozen: TTS cell
→ derived WAV SHA-256 → opaque blind ID → terminal ASR result. Intended text and
voice design were excluded from both ASR inputs. The [original blind job](blind/job.json)
and canonical copies in each ASR output retain that binding.

Qwen-ASR alone matches **9/16** intended phrases: design-a **4/8**, design-b **5/8**.
Comparison removes whitespace and Unicode punctuation only, using
[`quality_gates.normalized_actual`](../../../experiment_quality_guards/quality_gates.py)
from the frozen source SHA-256
`e9a0a838e134746bbd0fdd60aeff48ca9ad2963b721e3ea45bd0d8b45690b15d`.
There is no homophone folding (窝, 屋 and 吴 remain distinct), script conversion,
or edit-distance tolerance. This is **single-ASR intended-text agreement**, not
voice quality, two-ASR agreement, or human gold. Every cell is shown below.

| Cell | Design | Intended text | Raw Qwen-ASR text | Normalized match | SenseVoice | Human actual words |
|---|---|---|---|---|---|---|
| 001 | design-a | 你好小窝 | 你好，小窝。 | yes | NOT_RUN | PENDING |
| 002 | design-a | 小窝小窝 | 小窝，小窝。 | yes | NOT_RUN | PENDING |
| 003 | design-a | 你好小屋 | 你好，小屋。 | yes | NOT_RUN | PENDING |
| 004 | design-a | 小屋小屋 | 小吴，小吴。 | no | NOT_RUN | PENDING |
| 005 | design-a | 小窝小屋 | 小屋，小屋。 | no | NOT_RUN | PENDING |
| 006 | design-a | 小屋小窝 | 小窝，小窝。 | no | NOT_RUN | PENDING |
| 007 | design-a | 你好你好 | 你好，你好。 | yes | NOT_RUN | PENDING |
| 008 | design-a | 小窝 | 小屋。 | no | NOT_RUN | PENDING |
| 009 | design-b | 你好小窝 | 你好，小窝。 | yes | NOT_RUN | PENDING |
| 010 | design-b | 小窝小窝 | 小窝，小窝。 | yes | failed once | PENDING |
| 011 | design-b | 你好小屋 | 你好，小吴。 | no | NOT_RUN | PENDING |
| 012 | design-b | 小屋小屋 | 小屋，小屋。 | yes | NOT_RUN | PENDING |
| 013 | design-b | 小窝小屋 | 小窝，小屋。 | yes | NOT_RUN | PENDING |
| 014 | design-b | 小屋小窝 | 小屋，小屋。 | no | NOT_RUN | PENDING |
| 015 | design-b | 你好你好 | 你好，你好。 | yes | NOT_RUN | PENDING |
| 016 | design-b | 小窝 | 小吴。 | no | NOT_RUN | PENDING |

The [derived result summary](result-summary.json) binds all input hashes, preserves
both native and derived WAV paths and PCM hashes, and leaves human actual text null.
The 32-row [TTS ledger](generation/tts/tts-receipt.json) also retains the unrun
FireRed denominator. [Qwen terminal outcomes](asr/qwen06/terminal-outcomes.json),
[SenseVoice terminal outcomes](asr/sensevoice/terminal-outcomes.json),
[raw primary freeze](asr/primary-raw-freeze.json) and
[frozen disputes](asr/frozen-disputes.json) are original artifact bytes.

## Execution/resource evidence

Actual inference kernel readbacks reported 12 GiB memory.max, zero swap, a four-CPU
quota, PID limit 256, UID 1001, no capabilities, no-new-privileges, seccomp, private
cgroup root, read-only root and only loopback networking. These are **enforced
limits and observed configuration, not measured peak memory or CPU usage**.
Both TTS and ASR retained their scope/loader/decoder evidence; no model weights,
installed runtime, credentials or unrestricted logs are included here.

- TTS setup verified 4,956,729,799 downloaded bytes in 147.13 s; TTS container
  237.18 s; complete TTS host phase 400.55 s.
- ASR setup verified 3,340,966,447 downloaded bytes in 120.85 s; Qwen-ASR container
  158.93 s; stopped ASR host phase 312.86 s.
- Each job pulled the exact official Python image. Default-interface observations
  were 45,688,228 B (TTS) and 45,880,156 B (ASR), below each 128 MiB observational
  reservation. They include protocol/background traffic and are not exact daemon
  transfer counts or a strict hard transfer cutoff. Image/config completion was
  verified. Setup download counts exclude the image observation.

## Durable evidence and verification

[Provenance](provenance.json) records the run/jobs, immutable executed head, archive
IDs, exact archive sizes/digests and every original member digest. The generation
archive's **58 files** and ASR archive's **108 files** are copied byte-for-byte into
`generation/` and `asr/`. Each original `artifact-freeze.json` binds all other
members. The original blind `job.json` is retained; its 16 WAVs are byte-identical
to the derived WAVs already under `generation/tts/`, so they are not duplicated.
Archive container bodies are not retained or reconstructed.

Read the native 24 kHz float WAVs and derived 16 kHz PCM16 WAVs directly from the
per-cell links in result-summary.json. These are approved public generic synthetic
voices, not person-voice clones. Archive hashes verify retained bytes; they do
not certify human listening outcomes.

Earlier failed [v1](../run-38012657985/provenance.json),
[v2](../run-38014209435/provenance.json), and
[probe1](../probe-1-38016468182/provenance.json) evidence remains unchanged.
The successful [probe2 evidence at the executed head](https://github.com/jiying2007/kws-pipeline/tree/621ba90acd2d78976f4dcf07a385cd4e615ac358/research/source-screen32-v1/evidence/probe-2-38017741858)
proved the setup container scope; this run separately records actual model execution.
Both diagnostic slots and the approved v3 run are consumed. This retention record
provides no new execution permission.
