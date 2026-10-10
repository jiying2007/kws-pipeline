# Research map / 研究入口

This page separates reusable software, retained experimental findings, and product
qualification. A passing software check never promotes a research model. The
shipping model registry, runtime ABI, thresholds and `shipping_approved` policy
are unchanged by this consolidation.

本入口把可复用工具、实验结论与产品资格分开。研究代码通过测试，不代表模型通过
声学或板端验证。失败结论保留；旧实验的一次性执行许可不因整理而重新生效。

## Current research and retention / 当前研究与保留状态

**2026-10-10 parallel candidate review:** [Evidence-based optimization review](../docs/KWS_PARALLEL_OPTIMIZATION_REVIEW_2026-10-10.md) and [disabled candidate plan](../configs/research/parallel-candidate-screen-2026-10-10.json) separate a fresh same-data RNN64 control, causal DS-TCN and conditional causal FSMN from the historical shipping anchor. The proposed 6+2 fit ceiling is not a run permit, a verified fair budget or a measured result. Original-FA causal attribution, native-A20 admission, actual-label/PCM data admission and final product gates retain their own boundaries.

**2026-10-10 retained public source-screen result:** [Completed SenseVoice continuation and full joint table](source-screen32-v1/evidence/run-38021467257/README.md).
The existing 16 WAVs now have both recognizers' outputs: 13/16 transcripts agree,
but only 8/16 pairs match the intended phrase; 5 agree on a different phrase and
3 disagree. These are machine observations, not actual-word truth or training
admission. Current listening/screening tools are linked below; the public packet
contains no filled human receipts and does not report private review progress.
The [original partial failure](source-screen32-v1/evidence/run-38018029787/README.md)
and all raw evidence remain intact. Frozen preparation/readiness documents are
historical snapshots; no further run or training is authorized by this result.

Current delivery: [complete public archive + index + verified restoration / 完整归档＋索引＋可验证恢复](consolidation/ARCHIVE_DELIVERY_2026-10-09.md).
All 1,886 public logical members / 1,403 unique objects were restored and verified
at fixed data commit `d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a`.
The old 500-file expanded-copy target is cancelled; its historical status remains
NOT_PUSHED. This does not claim all 500 wrapper files are byte-identical archive
members. Earlier dated snapshots remain available from the delivery page.

- D20/D90 v2: two raw-logit coordinates exceeded the gate in the same D20 failed fixture; D90 not run
- Frozen Fixed50 outcome: EVAL_INCONCLUSIVE_LABEL_SUPPORT; K2 4 clips / 4 voices < 5,
  targets 11 < 12, unknown 8; no admissible KWS comparison
- Old frozen-model nightly: 2 FA / 8 h, upper95 0.786974 FA/h, failing 0 / 0.40;
  this is not D20/D90 evidence
- Human and physical-board qualification remains deferred

## Integrated reusable software

| Entry | What is usable | Boundary |
|---|---|---|
| [Native A20](native_a20/README.md) | Isolated C runtime, numerical contracts, exporter, invented-input tests and compile-only ARM checks | Research only. Original strict official-FP32 compatibility gate failed; the later PCM-derived mathematical contract is separate. Generic ARM compilation is not SSC305 SDK/ABI, physical-board or acoustic qualification |
| [Saved-prediction admission](native_a20_quality/README.md) | Source/exposure/label/context checks and metadata-only scoring | Existing 50-asset ledger is inventory, not independent FAR/FRR evidence |
| [Experiment quality guards](experiment_quality_guards/README.md) | Reject misleading split, label, comparison and ASR-support claims | Original-plan support, human-word support and training eligibility are distinct decisions |
| [Matched-context event/clip scoring](../eval/README.md#match-positive-and-negative-context-declarations) | Exact input bindings and equal declared positive/negative context policy | `MATCHED_DECLARED_CONTEXT` verifies declarations, not acoustic truth. Clip-presence mode reports no FAR/FRR or word-end latency |
| [Fixed-slot CTC scorer](k1-fixed-slot-ctc-v1/README.md) | Allocation-free C11 path-mass calculation with independent synthetic oracle | Caller supplies the fixed prefix and sealed interval. No automatic alignment, occurrence detector, acceptance threshold or K2 implementation |

Quick checks from the repository root:

```sh
python3 -B tools/verify_research_consolidation.py
python3 -B research/consolidation/tests/test_research_consolidation.py
python3 -B research/native_a20/build.py --cc gcc --output /tmp/a20-research-build
python3 -B -m unittest discover -s research/native_a20_quality/tests -v
python3 -B -m unittest discover -s research/experiment_quality_guards/tests -v
python3 tools/prepare_eval_context_fixtures.py
python3 tests/test_eval.py
```

The fixture-preparation command explicitly fetches three immutable public numeric
logs totaling 59,574 bytes, then checks their size and SHA-256. The test itself
never downloads or silently skips missing evidence. See each module for its
complete tests and exact scope; the CTC workflow runs both independent oracles.

## Current opt-in engineering tools (2026-10-10)

- [Saved-only evidence safety](offline-evidence-safety-v1/README.md): explicit preflight,
  real saved-reference admission, and failure capture. Tests fetch a pinned public
  archive to load one helper only; no old attempt, model or audio is executed.
- [Decoder scratch lifetime v1](decoder_scratch_v1/README.md): a verified derived
  build with 33,600 bytes less host workspace. The historical baseline stays intact;
  callers must rebuild for its new workspace ABI. This is opt-in research, not a
  shipping/default or board-qualified replacement.
- [Listening handoff and private receipt revisions](experiment_quality_guards/screen32-review/README.md#listen-and-record)
  ([PR #517](https://github.com/jiying2007/kws-pipeline/pull/517)) and
  [dual-ASR-first weak screening](experiment_quality_guards/screen32-review/README.md#dual-asr-first-weak-screening)
  ([PR #518](https://github.com/jiying2007/kws-pipeline/pull/518)): join retained
  audio identities without replacing human declarations. Machine consensus,
  including human-agreeing rows, remains weak screening; filled receipts and
  derived reports stay private. Neither view grants CTC or training admission.
- [D20 saved-input byte materialization and independent readback](d20-diagnostic-admission-v1/PLAN-SCHEMA.md#saved-input-materializer-byte-only)
  ([PR #519](https://github.com/jiying2007/kws-pipeline/pull/519)): the fixed four
  sources and all 798 planned jobs are byte-bound. This completed byte-only step
  does not prove backend consumption, historical dispatch or numerical admission.
  CMVN is a saved separate call, not an internal stage-0 observation; D20 FAIL,
  D90 NOT_RUN and execution NOT_READY remain unchanged.
- [Current admission checklist / 当前唯一准入清单](d20-diagnostic-admission-v1/PLAN-SCHEMA.md#current-admission-checklist):
  start here before proposing further diagnostics. Establish and independently
  verify a hard-memory scope before any exact-backend import inside that scope.
  Exact backend, historical source binding and independently accepted one-stage
  wrappers remain blockers; metadata-only tests never admit numerical execution.
  Prefer saved traces and a reviewed decoder-state observation plan; do not
  duplicate audio acquisition or expand training. FA causality is unresolved;
  matching PCM across runs does not establish identical state.
- [Historical PR #502 saved-evidence supplement](diagnostic-readiness-2026-10-09/PUBLIC-PROTOCOL.zh-CN.md)
  and its [dated observations](diagnostic-readiness-2026-10-09/PUBLIC-READINESS-SUMMARY.json)
  preserve evidence identities and findings. Its former preparation order is
  superseded by the current checklist; it is not an execution permit.
- [Earlier saved failure diagnosis](saved-diagnostics-2026-10-09/REPORT.zh-CN.md):
  retained saved-array/source attribution, invented-logit boundary tests and the
  original unexecuted plan. Historical background only; synthetic tests cannot
  relabel numerical failures.

For the next scientific validation priority, see the
[continuous-state, VAD and event-matched evaluation boundary](../docs/EVALUATION.md#current-diagnostic-priority-and-admission-boundary).
This proposes no new run, blanket reset, threshold sweep, model or training.

当前只以以上准入清单为操作入口；PR #502 补充及更早协议保留为历史证据。
必须先建立并验证硬限制，再在该隔离范围内检查精确后端 import；不得无约束安装、
import 后再判断是否符合预算。先复核已有 trace，不重复采集、不扩大训练。
D20 raw FAIL、D90 NOT_RUN、FA 根因未定、fixed50 支持不足、真人及实机资格 deferred
均不变；整理与 CI 改进不构成实验执行许可。

## Results worth retaining

The companion public evidence is now integrated into `kws-data` main by
[#21](https://github.com/jiying2007/kws-data/pull/21). Use its
[current research index](https://github.com/jiying2007/kws-data/blob/main/docs/RESEARCH_INDEX.md)
for navigation; the immutable links below preserve each original evidence identity.
They are historical observations, not instructions to rerun an experiment.

| Question | Preserved outcome | Evidence |
|---|---|---|
| Is the original A20 strict official-FP32 gate satisfied? | **FAIL**. Preserve the distinct later mathematical contract without relabeling the failed gate | [A20 source contract](native_a20/NUMERICAL_CONTRACT.md), [complete public archive](https://github.com/jiying2007/kws-data/tree/7af8f8597b0b7fbe8761c9a2400f2e4028395551) |
| May the fixed300 / Cosy49 trained candidates replace the shipping model? | **REJECTED**. The terminated candidate and negative results remain research evidence | [Cosy49 terminal archive](https://github.com/jiying2007/kws-data/tree/c81a217e4dff8eee8858cdb0772b37932a55bcd4), [fixed300 source record](https://github.com/jiying2007/kws-pipeline/tree/a44a8d785bacfc2daf3f94951547e74063e30d26) |
| Does fixed30 ASR calibration establish accuracy or automatic training labels? | **NO**. Exposed calibration: original-plan lexical support is 17/30, distinct from actual-human-word support 18/30. No acoustic-completeness certification follows | [fixed30 retained results](https://github.com/jiying2007/kws-data/tree/13e458ce4fc88718ffddcd2e7bc9b8a0792eda94/research/2026-10-05-fixed30-exposed-asr-regression) |
| Does the six-cell Qwen screen provide an admissible K1/K2 pair? | **REJECTED**. One weakly supported clip; five quarantined; no preset supports both keywords and no training admission | [Qwen6 source screen](https://github.com/jiying2007/kws-data/tree/35ab76b5f483195dcea8bb6a58bfcca456ce33c8/research/2026-10-05-qwen6-source-screen) |
| What did matched-context Melo diagnostics establish? | **DIAGNOSTIC ONLY**. Leading 1.5 s zeros plus 0.3 s tail recovered M2/K2; M3's false K1 persisted. The original machine text gate remains FAIL, 2/6 both-ASR plan matches | [Melo6 and context evidence](https://github.com/jiying2007/kws-data/tree/dc90c3aa325700fdf17da449452437c90373818e) |
| Did whole N1 establish terminal 窝/屋 separation? | **NO**. One K1 and no duplicate/K2 in one exposed positive clip; context qualification remains unverified, with no word boundaries or negative denominator | [N1 whole-clip appendix](https://github.com/jiying2007/kws-data/tree/aeb7bb86b5d2886ddca42a0692c5447b625031b6/research/2026-10-07-n1-wholeclip-leading) |
| Did first-prefix marginal CTC fix M3 while retaining N1? | **FAIL / STOPPED**. M3 falsely accepts K1 while N1 remains accepted. Later 屋 cannot retract already absorbed first-prefix 窝 mass | [failed prefix candidate](https://github.com/jiying2007/kws-data/tree/aeb7bb86b5d2886ddca42a0692c5447b625031b6/research/2026-10-07-k1-prefix-marginal) |
| Is the product acoustically and physically qualified? | **PENDING**. Synthetic, exposed, metadata, numerical and generic cross-build results do not substitute for real independent acoustic and physical-board evidence | [Qualification contract](../docs/RELEASE_QUALIFICATION.md) |

## Provenance and integration rules

- [Core retention map](consolidation/core-2026-10-07.json) binds all 102 imported
  source files to exact public commits, Git blob SHA-1, SHA-256, mode and size
- Integration order is #463 → #464, #465, #476 → #479, #485, #486. The #464
  manifest/build fixes and #479 label-decision fixes supersede their earlier
  file versions; the original versions remain addressable at the pinned commits
- The core batch changed no `src/`, `include/`, `models/`, `configs/`, training
  or shipping source. Its five pre-existing modified files belonged only to the
  offline scorer, its tests, documentation and fixture-preparation CI step
- The retention verifier checks current bytes, not external approval or scientific
  truth. An intentional later edit needs an explicit new provenance record;
  historical source pins and failed-result records must remain immutable
- Do not revive retired execution paths. See
  [repository governance](../docs/REPOSITORY_GOVERNANCE.md#retired-execution-lane-policy)
- No private natural recordings, derived features, learned private heads or
  identifying raw logs are included by this public-source consolidation
- [Public source integration](consolidation/SOURCE_INTEGRATION.md) retains the
  remaining isolated source modules, disabled workflow archives, sibling recovery
  variants and historical conclusions. Its original-path static checks run only
  in temporary offline projections. Old one-shot release files and branch names
  grant no new execution permission
- [Historical outcome index](consolidation/history/BRANCH_OUTCOMES.md) records
  dormant-branch decisions and corrected interpretations. Older narrative notes
  are retained verbatim as historical evidence, not current guidance. The
  [historical-note reading guide](consolidation/history/READING_NOTES.md) explains
  relocated links and unavailable old build artifacts
- The [2026-10-07 inventory](consolidation/branch-retention-2026-10-07.json)
  is historical, not a current claim that all 61 branches remain. The approved
  cleanup subsequently deleted 57 pipeline and 22 data branches, preserving
  two recovery tags. See the [completed cleanup record](consolidation/git-atomic-prune-2026-10-08.json)
  and the dated current status above. Historical snapshots are unchanged;
  no new deletion or experiment is authorized by this navigation update.
