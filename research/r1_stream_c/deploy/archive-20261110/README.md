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
- These three frozen archives predate the later INT8 and ARM/QEMU replay evidence in the maintained SDK. Physical SSC305/AFE, independent human FRR/FAR, and word-end callback latency remain NOT_RUN; see the subsequent-delivery section below and the maintained SDK reports for later completed work.

No source-only archive or CI green status authorizes public release of the excluded models/audio, or qualifies a product.


## Subsequent confirmation and resource-probe delivery

The following increment was completed on 2026-11-10 after the three frozen
archives above. Full findings and remaining qualification boundaries are in
[the V3 confirmation report](../../../../docs/KWS_V3_CONFIRMATION_20261110.md).
The primary R1 checkpoint is retained; two additional runs confirm only the
initialization of the two appended output rows. There is no best-seed selection.

| Public file | SHA-256 |
| --- | --- |
| `KWS_R1_INIT_CONFIRMATION_SOURCE_20261110.zip` | `7a461c2366326a1bff48d0cdd9a3e6f36f25f2422f331062ccec26a6433d0243` |
| `KWS_R1_INIT_CONFIRMATION_20261110.json` | `feaae4122050b3a0f627ac2a13831bce097c51ffcc9f47a2ea0ea3f46f38f756` |
| `KWS_R1_INIT_CONFIRMATION_REVIEW_20261110.json` | `8b41c2a35962b5584b132e78a39030c46169897c317134c5180acf0453c88361` |
| `ARM_RESOURCE_BUILD_20261110.md` | `87d488f9431239911e0417f15de5d0e7a575169eeaaf10f9dca272bbdd6fc03b` |
| `ARM_RESOURCE_BUILD_20261110.json` | `abf29aa0d9f52a84be29f5a50c194ade58a30f72e5448decd108cbe6c65e459c` |

The source ZIP has 13 payload files plus a complete SHA-256 manifest and no
weights, PCM, weak-label transcripts or per-record numeric payloads.

Separately retained owner deliveries (not uploaded here):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `KWS_R1_INIT_CONFIRMATION_OWNER_EVIDENCE_20261110.zip` | 269266984 | `c879b5ed00c7b6a84d010eee8ca4a167149946c539c7beb664eeaa671d8b90d9` |
| `KWS_R1_ARM_RESOURCE_PROBES_20261110.zip` | 142286 | `0ca4b8671be786cad0af07a9daaac1b44fda4af81cfc00b75015feb840c7ddcf` |
| `KWS_R1_ARM_TOOLCHAIN_INPUTS_20261110.zip` | 65689464 | `1a7f2c4eff08550052d9ccda1fe78aca0a87fa47a33c28e71e85757138f70f49` |

The owner confirmation archive retains both new checkpoints, all 268 numeric
outputs, immutable plans, ledgers, terminal receipts, independent review and
portable read-only audit inputs. Its 371 payloads were checked against the
closed member set and every SHA-256. The toolchain archive preserves three
existing public Actions artifacts; no new toolchain workflow was run.

The ARM resource probes compiled and rebuilt byte-identically, with no model,
QEMU or physical-board execution in this increment. Their maximum required
symbol version is GLIBC_2.35; real SSC305 rootfs/toolchain compatibility remains
unqualified. See [ARM_RESOURCE_BUILD_20261110.md](ARM_RESOURCE_BUILD_20261110.md).

