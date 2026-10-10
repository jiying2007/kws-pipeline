# Inert retained-audio SenseVoice continuation · 2026-10-10

**Prepared software only. Execution approval is pending; release remains false,
with no active workflow or execution reference.**

This narrow continuation addresses the incomplete ASR screen in
[v3 run 38018029787](evidence/run-38018029787/README.md). It does not regenerate
speech or turn the v3 failed run into a pass. The first SenseVoice clip already
consumed one attempt; the remaining 15 consumed none. A separately approved
continuation can make at most one new SenseVoice call per retained clip, stop at
the first failure, and preserve all untouched rows.

## Fixed inputs and allowed calls

`continuation-lock.json` binds the exact original blind job, all 16 derived WAVs,
the generation/ASR artifact freezes, Qwen raw freeze and prior terminal outcomes.
Inputs are checked before any image acquisition, then copied into a new directory
containing only the original `job.json` and opaque hash-named WAV files. The
ASR container sees only that audio-only input and the reviewed blind source
profile, never intended text, designs or prior ASR answers.

The fixed `sense-continuation` host phase has exactly two possible container
stages: existing ASR setup, followed by SenseVoice. It has no model selector,
arbitrary input argument, resume flag or restart path. No TTS, Qwen-ASR, Whisper
or additional probe can execute through that phase. Completed Qwen output files
are copied byte-for-byte from v3 into the host-side comparison evidence; they
remain identified as the original run's decodes and are never mounted into the
SenseVoice container.

The separate cross-run ledger retains every clip, its prior consumed attempt,
current consumed attempt and terminal status. Maximum cumulative attempts are
**two for clip-000001, one for each remaining clip**. Setup failure gives all 16
current attempts zero without losing the first clip's prior failed attempt.
A partial decoder failure records the consumed current attempt and leaves the
rest NOT_RUN. Old receipts are never overwritten. Human actual labels remain
pending, and machine comparison never supplies human gold.

## Identity and proof compatibility

The sole proposed identity is `qwen16-sensevoice-continuation-v1`, on branch
`research/qwen16-sensevoice-continuation-v1`, using its exact inert workflow
template. Activation requires an explicitly approved release, immutable reviewed
source freeze, the first creation of that public branch, workflow run number 1
and run attempt 1. Workflow numbering is not assumed available: any unexpected
number fails closed. The old v3 and both diagnostic identities remain consumed.

The worker contract's existing `run_id` is derived from the blind-job hash and
therefore stays the same for these unchanged inputs. It is an input-contract
identifier, **not a globally unique execution ID**. The new host branch, workflow
run/head, exclusively created output root, artifact freeze and cross-run ledger
jointly distinguish this continuation's consumed attempts from the old run.

The [retained probe2 proof](evidence/probe-2-38017741858) is verified against its
**original 66-file source freeze**, image, kernel receipt and completed cleanup.
Compatibility additionally requires exact unchanged source hashes for the scope,
probe, setup and image configuration, plus fixed AST hashes of the create,
inspect, supervision and proof-verification functions. This establishes only
reuse of the unchanged container/setup security contract. It does **not** claim
the new whole source freeze passed probe2 or that the repaired recognizer works.
The original stdlib probe replaced the TTS setup command tail; the continuation
uses the separately pinned ASR setup tail with the same container limits.
Actual kernel readback and setup/worker checks still occur before imports in
any future execution.

## Resources and public evidence

Existing ASR setup is reused unchanged: 3,340,966,447 bytes of exact pinned
runtime/model inputs. It includes both retained recognizer packages/assets for
compatibility, although only SenseVoice is loaded/called. The unchanged ASR
setup cap is 4 GiB of controlled input downloads, with a separate 128 MiB image
observation reservation; together they remain below the declared 6 GiB overall
allocation. The image observation is not a hard Docker-daemon transfer cutoff.
New code and retained input bytes arrive through the pinned checkout and are
not hidden model-download inputs.

The same free hosted runner, 12 GiB hard memory/zero swap, four-CPU quota,
PID256, unprivileged/read-only scope, no capabilities, seccomp and network-disabled
inference apply. Setup and inference deadlines, single-container starts, bounded
sanitized failure records and public-artifact limits are reused unchanged.
No weights, installed runtime or credentials enter the public evidence artifact.

The [old 66-file execution freeze](history/executed-v3-621ba90/execution-freeze.json)
and [69-file inert repair freeze](history/asr-repair-fd3049f/execution-freeze.json)
are preserved with only changed originals and complete reconstruction mappings.
The current freeze covers this inert continuation preparation. These are separate
identities; no historical successful observation is relabelled as a new run.

Local stdlib tests cover literal inputs, proof compatibility, unique identity,
audio-only staging, exact prior-Qwen retention, cumulative attempt denominators,
and the fixed two-stage host path including setup/decoder failure. They make no
Docker, model, installation or network calls. Actual continuation is NOT_RUN.
Human gold, training admission and shipping approval remain false; D20 FAIL and
D90 NOT_RUN remain unchanged.
