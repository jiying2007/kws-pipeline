# V4 English streaming C: measured saved-weight parity

## Completed outcome

Both previously trained English63 FP32 candidates now have a standalone C
streaming implementation and one successful public-runner numerical comparison.
The run reused the exact original 63 recordings, 189 saved ASR observations and
both trained checkpoints. It performed **zero optimizer updates and zero new
TTS/ASR/IPA calls**. This is L1 implementation evidence. It does not promote a
model or establish acoustic or physical-board qualification.

- Pipeline [PR #808](https://github.com/jiying2007/kws-pipeline/pull/808)
  merged at `376034f7037b797cc29f6e654b0b891d54825ccd` after 13 successful checks.
- Data [PR #121](https://github.com/jiying2007/kws-data/pull/121)
  merged at `0d5db0eaa0101ee8009882aa86dc85dc513cd174` after 34 successful checks
  and 14 designed skips.
- Actual [run 43605922752](https://github.com/jiying2007/kws-data/actions/runs/43605922752)
  succeeded on its first attempt. It executed reviewed pipeline source
  `6eab7c5e72d44483a14c5095a53805ea94126070` and the merged data source above.
- Result artifact `14117638750`: 22,016,580 bytes, SHA256
  `e85d14c87727f93b6e33edc196af00a6c9ce023d9e7d41d844109617fa5fb397`.
- The compact machine-readable result is
  [measured-20261114.json](../research/english63_online_c/measured-20261114.json).
  The full original artifact preserves the complete traces, generated weight
  header, native executable, build provenance, resource samples and terminal.

## Fixed numerical contract and observed result

The input contained 2,101,760 PCM samples at 16 kHz: 131.36 seconds. Each model
was checked on all 63 isolated clips (13,010 frames) and on one concatenated
continuous stream (13,134 frames). The continuous model and sample clock were
never reset at clip boundaries. EOF produced no padding samples or extra events.

The maximum absolute error limit of **0.0002** for normalized features, logits
and probabilities was frozen before this run. Every frame was checked.

| Candidate | Maximum feature error | Maximum logit error | Maximum probability error | Python/C event class and sample clock |
| --- | ---: | ---: | ---: | --- |
| GRU | 4.7683716e-7 | 1.1205673e-5 | 2.6226044e-6 | Exact |
| DS-TCN | 4.7683716e-7 | 1.4305115e-6 | 3.5762787e-7 | Exact |

For both models, native traces were **byte-identical** between 160-sample feeds
and the repeating mixed plan `1,159,161,3199,320,7`, in both isolated and
continuous modes. An independent saved-byte reader verified the closed
17-file comparison inventory, all eight C traces, both reference NPZs, event
identities, chunk equality, and all ten resource records. This audit imported
no project model and executed no model forward.

A separate source audit verified the outer terminal's 22 member hashes and all
nine build-source hashes against the executed commit. It parsed the generated
C hexadecimal constants and matched the two original NPZ parameter payloads
byte for byte. All 128 saved reference-probability traces (64 per model) were
also byte-identical to the earlier training artifact; its 100 GRU and 8 DS-TCN
event identities were reproduced without altering the original checkpoints.

Observed proposal counts remain GRU 34 isolated / 66 continuous and DS-TCN
7 isolated / 1 continuous. These are uncalibrated research events, not correct
detection counts, recall, false alarms or a basis for selecting either model.

## Native host resource measurements

The dedicated continuous/160 profile disabled trace output. It used a fresh
native child process under GCC 13.3.0 on an x86_64 Linux public runner. Each
candidate received 13,136 feed calls. Neither recorded a feed longer than
10 ms in this particular observation.

| Observation | GRU | DS-TCN |
| --- | ---: | ---: |
| Trainable-model parameters already saved | 6,531 | 4,803 |
| FP32 parameter bytes | 26,124 | 19,212 |
| Caller-owned C state bytes | 38,168 | 38,168 |
| Shared frontend table bytes | 3,894 | 3,894 |
| Feed P50, microseconds | 10.787 | 9.495 |
| Feed P95, microseconds | 11.588 | 10.315 |
| Feed P99, microseconds | 16.314 | 14.913 |
| Maximum observed feed, microseconds | 50.877 | 78.749 |
| Process CPU seconds for 131.36 seconds of PCM | 0.147479 | 0.130467 |
| RSS high-water captured through inference, KiB | 2,440 | 2,396 |

The 80,040-byte native executable contains both models. Per-model parameter
bytes, shared tables, caller state and process RSS are different quantities.
Feed timers surround the core call; process CPU/wall time also includes PCM
input reads. RSS was sampled after inference but **before** timing-array
sorting and JSON output, so it is not a measured peak over the complete process
lifetime. Stack usage was not measured. None of these host figures is SSC305
performance, a worst-case execution-time guarantee or word-end latency.

## Implementation and integration boundary

The [C module](../research/english63_online_c/README.md) preserves the original
400-sample window / 160-sample hop, 32-bin mel frontend, train-only normalizer,
model arithmetic, 200-state trailing readout and fixed proposal rule. The GRU
hidden state retains an unbounded acoustic past even though readout pooling is
finite. The existing SDK's 320-sample hop has not been substituted.

The core uses caller-owned state and readonly weights, with no heap allocation,
locks, threads or file access. The separate CLI provides file I/O and profiling.
The generated header exports exact FP32 hexadecimal constants from the original
NPZ weights; checkpoint bytes were verified before and after execution.

Use the existing saved-result verification command to recheck an extracted
artifact without executing a model:

```sh
python3 -I -S -B research/english63_online_c/c_replay.py \
  --verify-result-dir /absolute/result/comparison
```

## Qualification status

Formal training admission, human-gold labels, model promotion and model or
threshold selection remain false. The 30 weak train-role, 12 exposed-development
and 21 quarantined nearphone roles are unchanged. Independent English FRR,
continuous natural-negative FAR/hour, word-end latency and physical SSC305
profiling remain **NOT_RUN**. Further qualification requires the corresponding
independent acoustic evidence and physical-board measurements.
