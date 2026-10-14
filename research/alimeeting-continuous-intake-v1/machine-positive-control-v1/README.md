# Frozen machine-labelled development response control

## Observed result, 2026-10-14

One authorized pass processed all 13 reviewed generated utterances unchanged: **444,160 real samples, 27.76 seconds, 1,375 feature frames**. There were **zero trigger events**.

| Machine target | Selected clips | Matching-trigger clips | Wrong-keyword events |
| --- | ---: | ---: | ---: |
| 你好小窝 | 2 | 0 | 0 |
| 小窝小窝 | 11 | 0 | 0 |

This frozen configuration did not pass this machine-labelled development response control. Zero events on the separate 4.205-hour AliMeeting stream cannot establish low false accepts or useful detection sensitivity. These utterances are synthetic speech with three-ASR consensus, **not human-gold references**. No human FRR, FAR, generalization, training admission or shipping approval is inferred.

## Fixed identities and procedure

- Selected every and only `report.json` slot with `three_model_training_candidate == true`, before any KWS response was observed. No sample replacement or selection based on KWS output.
- Independent ASR review passed: artifact `f8bf70a0539d6ca7bd11dfed9229a087c0c7d27b943550f73136cd765f54ef0c`; report `5c4c4c909d9e3b8a02e5d39b0dfbafae1630d13028227d611aef8379a50538f4`. Parent supplied the permission to use these candidates after that review.
- Source commit `11b93abe92427c3ece2a01f9a497cdb74bab0062`, tree `b8ae29e82097c38e92dbb1bb1cd4db10ade6cdc6`.
- Same executable `53a5dcb50c599d88bb72b2cfbaf6ff5d6e7a379ecef3011f28f5820261d155a9`, model `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`, keyword pack `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`, build receipt `78b2988e60a4bf6b875934222fd5e732633ca34de64e470c433e74289f57ac8a` as the AliMeeting diagnostic.
- Both existing keyword thresholds decode as float32 `0.550000011920929`. No command-line threshold overrides, training, modified weights, padding, flush frames, trimming or resampling.
- Block size 160. One process at each real generated-clip boundary; continuous state within the complete unchanged clip. Exactly 13 model invocations. No retry.
- Pre/post file identities verified, including original AliMeeting summary `e34600b0cc8816e3e23dda888782c8eabb831fb9bea658518bb41a12f51f1c06`, which remains unchanged.
- New result `run-p4-11b93abe/summary.json`: `4510f8f8d2b5008da89c274d8cb074fae646788855753a161e098c6cca32647a`.

Aggregate execution stats: 879 speech-gated frames, 1,317 blank-top1 frames, 58 nonblank-top1 frames, zero decoder hits, zero detections, zero discontinuities/lost samples/external-VAD frames. These counters verify progress; they do not retain the actual per-frame posterior sequence. A zero-event file is accompanied by a successful per-clip invocation, full sample/frame counts, and immutable output hashes.

## Package boundary and tests

The additive public package contains scripts, small hash-bound receipts, empty event files and aggregate/clip metrics. The 13 WAV bodies under `run-p4-11b93abe/audio/` are **not included**. The public ASR artifact, frozen upstream source, model/pack bodies, executable, and the original local absolute paths are external replay prerequisites. This is a metadata/evidence package, not a standalone acoustic reproduction bundle. The existing 90-file AliMeeting intake package and its allowlist remain unchanged.

`run_control.py` depends only on Python standard library plus the retained sibling `../diagnostic_runner.py`; its `prepare` and `run` enforce the fixed original identities. It is not an invitation to repeat the actual diagnostic. The offline tests never open real audio or call a model:

```sh
python3 -I -S test_run_control.py
python3 -I -S -O test_run_control.py
python3 -I -S -OO test_run_control.py
```

`preexecution.json` establishes the selected 13 inputs and fresh destination before the first invocation. `execution.*`, per-clip invocations/completions/stats, `inputs.json`, `events.jsonl`, and `summary.json` preserve the single actual pass. Any later observation or improvement experiment must use a new versioned directory and retain this failed control unchanged.
