# FSMN2599 two-session long-stream development observation

Record date: 2026-10-20. Proposed repository location:
`research/fsmn2599-longstream-development-20261020/`.

## Result and interpretation

The unchanged default B model completed one serial pass over each of two full
sessions, labeled A and B for this record, with threshold 0.7. The run consumed
66,552,539 original mono samples: 4,159.5336875 seconds, or
1.1554260243055556 full-session hours. Final events were zero for both targets,
**你好小窝** and **小窝小窝**. The observed metric is **0 total
triggers/full-session hour**.

All target truth remains unknown and `expected` is null. Accepted negative
seconds and admitted independent hours are both zero. Source exposure,
speaker, and pretraining independence are unknown. This observation establishes
neither FAR=0 nor recall/FRR, certified independent negatives, or production
suitability. The existing **B76 strict numeric gate remains FAIL unchanged**;
the rejected FP64 candidate was not used. There is no model or default promotion.

## Public source and attribution

Audio came from AliMeeting, provided by Alibaba Group. The official
[OpenSLR SLR119 resource](https://www.openslr.org/119/) lists CC BY-SA 4.0
and links the [official Test archive](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz).
Dataset attribution: Fan Yu et al., *M2MeT: The ICASSP 2022 Multi-Channel
Multi-Party Meeting Transcription Challenge*, ICASSP 2022.

This record contains aggregate observations and public code identity metadata.
Sessions appear only as A/B; audio, annotations, model payloads, raw outputs,
and direct source-session or speaker identifiers are not included. Omitting
these identifiers does not guarantee anonymity: exact durations and sample
counts may allow re-identification against the public dataset.

## Frozen runtime and input protocol

- Repository: `jiying2007/kws-pipeline`.
- Frozen main commit: `b8e2f4adbfb9e964197276ad82c4fb855d747114`.
- Model: unchanged default B. Final threshold: 0.7.
- Compiler: GCC (Debian 14.2.0-19) 14.2.0, target `x86_64-linux-gnu`.
- Strict numeric flags: `-std=c11 -O2 -Wall -Wextra -Werror -pedantic`
  `-fno-fast-math -ffp-contract=off -fexcess-precision=standard`
  `-UKWS_AFFINE_BLOCK2 -UKWS_B_PROFILE`.
- Original format: PCM16LE, 16 kHz, eight channels. Select raw channel 0
  (zero-based) as one mono stream, with no mixing, gain, resampling, or beamforming.
- Whole sessions were processed once, serially. No warmup, retry, threshold
  sweep, replay, or model change was performed.
- One model initialization, create call, and begin call per session; generation 1.
  Two model initializations in total. No mid-session reset, finish call, EOF
  flush, or appended padding samples.
- Annotation gaps and end tails remain in the denominator and retain unknown
  target truth. End-near detections that need future context are right-censored.

## Aggregate observations

| Observation | Session A | Session B | Total |
|---|---:|---:|---:|
| Original mono samples | 33,064,320 | 33,488,219 | 66,552,539 |
| Full-session seconds | 2,066.52 | 2,093.0136875 | 4,159.5336875 |
| Feed calls | 103,326 | 104,651 | 207,977 |
| Completed fbank frames | 206,650 | 209,299 | 415,949 |
| Completed network rows | 68,883 | 69,766 | 138,649 |
| Completed softmax rows | 68,883 | 69,766 | 138,649 |
| Completed decoder rows | 68,883 | 69,766 | 138,649 |
| Completed callbacks | 68,883 | 69,766 | 138,649 |
| Final events: 你好小窝 | 0 | 0 | 0 |
| Final events: 小窝小窝 | 0 | 0 | 0 |
| Total final events | 0 | 0 | 0 |
| Total triggers/full-session hour | 0 | 0 | 0 |

The numerator counts final accepted, deduplicated, refractory-filtered event
callbacks. The denominator is every consumed original sample divided by
16,000 and then by 3,600. It is not a certified negative-duration denominator.
The JSON companion preserves per-session durations and the same counts.

Both sessions had `support_ok=true`, zero empty-beam-window rows, zero
zero-mass-window rows, and zero candidate-level threshold rejections. These
are diagnostics, not accuracy or strict numeric qualification. A zero rejection
counter does not establish zero posterior values, zero keyword probabilities,
or absence of earlier paths/candidates; it supplies no evidence that lowering
the final threshold alone would help.

## Timing, resource bounds, and review

The sum of reaped-child CPU was **103.488708 seconds** and the sum of
per-session supervision wall time was **105.342389 seconds**, reported to six
decimal places. These are not exact whole-invocation CPU or wall totals:
supervisor CPU and whole-invocation totals were not persisted. Peak sampled
child RSS was **4,485,120 bytes**, approximately **4.28 MiB**. RSS is sampled
and is distinct from the enforced address-space limit.

Applied limits were 900 seconds total wall and CPU; 600 seconds wall and CPU
per session; 512 MiB process address space; 8 MiB captured stdout plus stderr
per session; 20 MiB controller storage; 32 MiB new task data; and a 2 GiB free-disk
floor. Both native session exits and the outer invocation exit were zero.
Successful outer exit and the reviewed final resource checks support budget
compliance; no resource or clock failure was observed.

Independent postflight review passed engineering run integrity and bounded
observation. It checked original full-session inputs, frozen execution binding,
raw terminal/count consistency, and aggregate arithmetic without running the
models again. Native full-row completion invariants passed. Because no event
was emitted, event-node and event-clock checks are vacuous, with no observed
event timing to validate. **No detection latency was measured or claimed.**

This two-file record adds no scripts, controller changes, framework, CI gates,
or model execution. Its preparation uses saved evidence only.

## Public numeric-source identity

The following public source paths and SHA-256 values are bound to the frozen
main commit. They identify the PCM, fbank, network, softmax, and decoder source
used by the reviewed runtime, and do not include audio or model hashes.

- `research/fsmn2599-stream-c/canonical_pcm.c`
  - SHA-256: `9631201666df788d1a40649ca43a0786d91408acea520c1235a25199847df57b`
- `research/fsmn2599-stream-c/canonical_fbank.c`
  - SHA-256: `7e84597f4bf55880301a26134a030b090aad4b4e42ca2f43e5ed7970fc954bc9`
- `research/fsmn2599-stream-c/fsmn2599.c`
  - SHA-256: `12b237eb78fdccb16658e66f096c32cac60814131e2afcd457af648d3c6c1dd7`
- `research/fsmn2599-stream-c/softmax2599.c`
  - SHA-256: `d2cb8ee4891809445bbd3b04912196d847d8f943066b3dd64479b861a863b6e4`
- `research/fsmn2599-three-window-c/kws-fsmn2599-three-window-c-preparation-20261018/kws3.c`
  - SHA-256: `e6d57e167fe85f150517eeb7b54f38eb9ce46a54922bff8897eaa8f7bedca41f`
