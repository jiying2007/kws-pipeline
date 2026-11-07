# V3 actual model results: R1-DEV-FSMN-20261107

This records completed numerical work, not another preparation plan. The new FSMN candidate is a **provisional, single-seed development research model**, with exact same-environment retraining reproduction and a tested offline inference package. It is not an independently qualified continuous wake detector or an SSC305-qualified model.

## Integrated software

- Pipeline #766 merged at `7bfcb33a47a1df3e317c835b81e419d14e07dbf8`: partial-known keyword comparison and corrected V3 priorities. Post-merge main CI run `42364395379` succeeded.
- Data #81 merged at `ec2ff456d08b4142233b61f1071402c41cadf60b`: native-verified keyword evidence. All 14 exact-head PR workflows passed. Hosted Python3.12.16 executed (71 unit + 7 real upstream-verifier composition + 20 existing tests) in normal/-O/-OO, 294 test executions. Full retention passed with 1948 retained /1957 tracked files, preserving the old1945 rows.
- New keyword-positive proposals still do not grant training admission, event timing, human-gold or natural-FAR authority. The present model experiment uses the previously established96-row weak research cohort, not labels newly proposed by that adapter.

## Actual bounded experiments

All candidates consume the same retained96 eligible training rows; the34 development rows and4 unknown zero-loss rows never enter the optimizer. Original TTS/ASR and source roles remain unchanged. Every new experiment freezes its own recipe before invocation and uses one shared heavy-model slot in an isolated local executor. Remote Dot/D1/D2 state and unknown calls are not modified or replayed.

| System, primary0ms | K1 complete greedy sequence | K2 complete greedy sequence | Development token edits | Exact full texts |
| --- | ---: | ---: | ---: | ---: |
| Retained balanced64 C1, no rerun | 4/10 | 9/10 | 140/287 | 4/34 |
| New C1 RF63 | 1/10 | 8/10 | 133/287 | 5/34 |
| New pretrained FSMN tail | 10/10 | 10/10 | 20/287 | 18/34 |

Each keyword has24 known weak-negative development rows, with zero complete-sequence false-positive rows for these three systems. These are previously exposed synthetic weak-label clip results, NOT natural FAR, online event FRR, word-end latency, fresh human accuracy or a proof of perfect recognition. Different pretrained initialization, native frontend, augmentation and loss terms make the FSMN/C1 comparison a **system comparison**, not a single-variable architecture ablation. The development rule required no per-keyword/per-old-new-cohort loss, at least one gain, and no additional weak-negative false positives. The FSMN candidate passes that rule; RF63 is rejected and retained.

### RF63 C1

One real model,2000 AdamW updates,8000 training forwards and402 final observations (134 rows × offsets0/200/500ms). Offsets are repeated conditions, not additional independent samples. Expanded the existing causal depthwise-temporal encoder from3 blocks/RF15 to5 blocks/RF63, retaining original shared zero-step tensors and using a fixed new-block initialization. This changes depth and initialization as well as receptive field. No warm-start from the trained control, threshold/checkpoint sweep, or repeated baseline forward.

- Plan SHA256: `6ebe275a6435f73b05b8675dd8c1d4d9a842cf738591704749d7e99cc914ab82`
- Final checkpoint: `494ff2b7dbd7ba4899bec9b13d4dfcee1d13136abf64267a7644b10d3f11ffb7`
- Final state: `0c00f5e49b9cfadcadb63b72b66f2166d14ff1b991cf8663e46ea2452786575e`
- Retain as a negative result; do not promote or tune against these same results until a separate hypothesis is declared.

### FSMN tail

Use the existing pinned ModelScope `thuduj12/kws_wenwen_fsmn_ctc` revision `cd0c18d0dda6742c371fbb72ec66185906a9885d`, donor `avg_30.pt` SHA256 `d3d89b5dde08c2926d46a97eab6f7179a87b1946ccebd16c02e77042ea8389fe`. The original repository vocabulary extension and native80mel/25ms/10ms/left2-right2/context400/skip3 frontend are reused. Train only the last FSMN block and output layers:467667 trainable parameters of756415 total; all other donor tensors remain bitwise unchanged.

Fixed AdamW lr0.0001, weight_decay0.01, clip5, seed20261105, batch4,2000 updates/8000 forwards, and134 final forwards. No added-zero blank auxiliary term or offset augmentation in this system. One final checkpoint only; no threshold, learning-rate or checkpoint search. Actual canonical environment: CPython3.12.16 / Torch2.13.0+cpu, one intraop and one interop thread, unchanged numerical runtime policies.

- Plan SHA256: `769f9e9c5aef304be643f3da5a15cb19b2270cb74da012e326511155d5485274`
- Final checkpoint SHA256: `99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571`
- Final model tensor-state SHA256: `71cf089fe9c163a0c76d394ed8a4e1937856dd831473ad3d2fb59a902be76b58`
- Initial state SHA256: `fb540f505d012d5409209bb74ad7cc2a5567e50ba4a18ea21b763c41aa7cabcd`
- Worker250.708929383s, supervisor252.080344654s; peak process RSS399048KiB. These are host training measurements, not target-board inference performance.

## Actual reproducibility and inference validation

A separate read-only auditor verified all retained file hashes,2000 completed optimizer pairs,8000 completed training-forward pairs, source/role isolation, initial/final tensor identities, and every saved feature/logit/greedy projection. This auditor performs zero model construction/forward/update. The original nested `summary.actual.status` is a pre-terminal write snapshot; final `status.json` and successful supervised exit are the terminal authority. Original evidence bytes are not rewritten.

A fresh directory then restored2041 exact original capsule files and repeated the FSMN training with a new optimizer, the same frozen seed and schedule. **All model tensor bytes match, and all134 feature/logit/greedy identities match.** Different relocated metadata means whole checkpoint serialization bytes need not match. This is exact same-host/software numerical reproduction, not cross-platform or multi-seed generalization evidence.

- Reproduction plan SHA256: `3c3c42e185df9654337dcbfab69d4d8e65c2e72f6de7646d580c658a7cff7e33`
- Reproduced state: `71cf089fe9c163a0c76d394ed8a4e1937856dd831473ad3d2fb59a902be76b58`
- A second saved-output audit verified the reproduced run independently of its trainer.
- Packaged inference CLI was actually invoked on four fixed cases (K1, K2, ordinary weak negative, unknown/unscored). All four source WAV/PCM, feature, logit and greedy identities matched the saved final results.

Aggregate new numerical work:6000 optimizer updates,24000 training forwards,670 final training readback forwards, plus4 packaged inference forwards. New TTS0, ASR0, retained C1 forwards0, unknown-call replays0. This includes the explicit retraining experiment, not hidden duplicate execution.

## Durable owner-held delivery

The private source/input capsule is `checkpoint-99399e8c75808d4b4f74b15d.zip`, SHA256 `84048654b73de0d6da955472eacaff608b7a54534d0f48c035b56c9b46b9a6bc`. It is restored with exact retained source/label/PCM checks; raw voices are not copied into this public report.

Delivered in the owner's Library under `/kws-pipeline/V3-R1-DEV-20261107/`:

- `KWS_R1_DEV_FSMN_20261107.zip`,9143788bytes, SHA256 `4bc513f406f5c38af5bc0fced57b931c2d7aa88e6a255ebfa2fbccc5863b0eec`: actual weights, fixed source/terms/vocabulary, tested offline inference CLI, exact recipe, bounded capsule-restore/retraining CLI and audit/reproduction receipts.
- `KWS_V3_Research_20261107.zip`,18465455bytes, SHA256 `c1876f4046c18045ff01f53bd27e88ce4195b2eb82e6f4977f9b2e9db3754454`: experimental sources, weights, per-step ledgers, reports and retained failures. Audio and full numeric matrices are not duplicated in this compact record package.
- Nine separate numeric archives retain all402 RF63 and134 FSMN float conditions; the134 reproduction files are byte-identical and reuse the FSMN objects rather than duplicate them. Recovery index SHA256 `ec88e02ccb80c1092b41a3a2a9a7153f97f2a3348ac21e3a7690ed5dcbe4efc7` binds every archive/member. They are saved in the same Library's `numeric` subdirectory.

Public archival here is aggregate evidence only. It does not grant new rights to raw recordings, donor-derived weights or training data. Exact research use and source terms remain applicable.

## Failures retained and boundaries remaining

Source staging's403 Git-tree publication failure occurred after294 tests passed; connector-based source integration subsequently succeeded without weakening the guard. The first RF63 adapter attempt failed at an old saved-feature field lookup, and the first FSMN adapter attempt failed on an absolute path rejected by the existing reader. Both were verified to have zero model construction/training before explicit corrected attempts. A launcher executable-path error also created no worker. No failed/unknown attempt is silently overwritten.

The selected FSMN remains a **full-recording, right-context research model**. Its CLI does not invent online wake events or word times. Multi-seed confirmation, fresh human/continuous-negative evaluation, calibrated streaming decisions, complete C/int8 export and SSC305 physical resource/soak evidence remain NOT_RUN. Direct-KWS remains conditional on its valid objective/window labels; no claim that all three V3 lanes trained or that a globally best model has been established. The natural73 windows retain development-only roles and were not turned into negative exposure or fresh test data.
