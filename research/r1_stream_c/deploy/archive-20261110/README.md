# KWS V3 source-only research deliveries — 2026-11-10

This directory retains **exact, byte-verified source-only ZIP archives and aggregate reports** from three successive V3 research checkpoints. The active, maintained R1 streaming SDK source entry is [../](../), which uses a pinned source materializer. The archived standalone SDK ZIP is a separate frozen snapshot; **do not assume the two implementations are byte-identical or interchangeable without independent verification**.

These archives are inert historical research attachments: no workflows, training, model inference, or data admission are triggered by their presence.

| File | SHA-256 |
| --- | --- |
| `KWS_V3_20261110_source_bundle.zip` | `c823a1b844e8f57c55b465613dac8fe432a463d3b07c09f0cf80b8cd7b5d8e14` |
| `KWS_V3_GRAPH2_20261110_source.zip` | `3f7df5ea6b95109e3d17e64f512eac55e96f5098f3c53014dc05cd7b9e83bc5b` |
| `KWS_FSMN_R1_SOURCE_20261110.zip` | `2785c8377c2cd23754673cc8cbbe09d0ceba637ce450c68a5262d7809f9ae102` |
| `KWS_V3_20261110_REPORT.zh-CN.txt` | `e191b7af0a3cbc179d5e6d722107bc5fb14a082fb48fa9bca52d8ce9d1c09e20` |
| `KWS_V3_GRAPH2_20261110_REPORT.zh-CN.txt` | `3fa92109686f79b9a749c1c07c0c9d84e693fc709fb8236c8c84097d00d10bec` |
| `KWS_FSMN_R1_20261110_REPORT.zh-CN.txt` | `ab68c57e0bd77b4133c6b215142b0db244d2decf72de5bb4370840f74e619ce7` |

## Deliberate public/private split

**Not uploaded to this public repository**:

- `KWS_FSMN_R1_SDK_20261110.zip`: includes owner-held derivative FSMN weights and compiled host binaries; source-only equivalent is retained above and the maintained generator is in `../`. The checkpoint/donor rights notice does not grant broad public redistribution.
- `KWS_FSMN_R1_EVIDENCE_20261110.zip`: contains 134 original PCM recordings, individually bound source/label evidence, and replay outputs. It is owner-held, not a public data artifact.
- `KWS_V3_20261110_execution_bundle.zip` and `KWS_V3_GRAPH2_20261110_execution.zip`: contain research computation artifacts, checkpoints or per-record numeric/evidence files with separate provenance and rights.
- Raw ASR transcripts, reference voices, private identifiers, and authorization records.

The excluded original archives remain with the owner under their original hashes; their absence from GitHub must not be represented as deletion or loss. This repository is public and will not use a private repository as an overflow workaround.

## Scope and verification

- The historical 60%/70% small-model complete-recall claim was corrected. See [PR #772](https://github.com/jiying2007/kws-pipeline/pull/772).
- The strong R1 FSMN results are **exposed synthetic weak-label development**, not independent human accuracy or continuous real-world FAR.
- The active reference64 SDK and the historical standalone SDK both retain **uncalibrated complete-keyword proposals**; original FP32-vs-Torch numerical failure remains FAIL.
- Physical SSC305/AFE, ARM wrapper execution, INT8, independent human FRR/FAR, and word-end callback latency remain NOT_RUN until actual evidence exists.

No source-only archive or CI green status authorizes public release of the excluded models/audio, or qualifies a product.
