# B-specific 76-frame numerical result, 2026-10-19

**STRICT_NUMERICAL_MISMATCH; stopped; no promotion.** This is a saved-result
record for the exact original seed0 B identity, on one reused 76-by-400 context
and the recorded CPU environment. It does not generalize to other inputs,
other model identities, other numerical backends, audio-frontend parity,
independent acoustic quality, continuous FAR, or SSC305/device performance.

The [aggregate and public source identities](B76-NUMERICAL-RESULT-2026-10-19.json)
retain the unchanged elementwise rule: abs(C - Torch) <= 1e-5 + 1e-5 * abs(Torch).
The reference is the saved current Torch 2.11.0+cpu B execution, not a
reproduction of the historical Torch endpoint. Checkpoint and payload hashes
match the existing public B identity record; weights are not included.

## What actually completed

- Default four-lane C versus current Torch: 76 frames each, one completed
  process per side, zero warmups and zero resets between frames. Of 197,524
  logits, 209 exceed tolerance. Of 428,032 cache values, 3,983 exceed tolerance.
- First-layer-only FP64 arm: one new C process, 76 frames, zero new Torch
  runs and zero new baseline runs. It reused those saved references.
  Logit failures fall from 209 to 0, while cache failures rise from 3,983 to
  6,342: 4,505 newly failing positions and 2,146 no longer failing positions.
- Cache failure counts worsen in all 76 frames. Cache maximum absolute error
  rises from 0.000125885009765625 to 0.000152587890625; its maximum tolerance
  ratio rises from 6.861050742106082 to 7.688657692172506.
- Logit aggregate maximum error and tolerance ratio improve, but per-frame
  maximum absolute error worsens in 47 frames and maximum tolerance ratio in
  52 frames. Passing every candidate logit is not whole-network parity.

The completed baseline followed an earlier incomplete controller-I/O-race
attempt. That earlier consumed work remains recorded; it was not refunded or
reclassified as a completed run. The FP64 arm also consumed one separate
preflight compilation and one released-execution compilation. Neither arm
used automatic numerical retries or kernel/threshold sweeps.

## Mathematical source difference, not a runtime integration

The unchanged baseline kernel is pinned to public commit
`b8e2f4adbfb9e964197276ad82c4fb855d747114`. The locally evaluated candidate
changes only its first 400-to-140 affine: each binary32 input/weight pair is
multiplied into one of four binary64 accumulation lanes, the lanes are reduced
pairwise, bias is added in binary64, and the result is rounded once to binary32.
All remaining affine calls use the original implementation. The local helper
is explicitly opt-in; removing that one guarded substitution and helper include
restores the byte-exact public kernel.

This publication records that difference and its source digests. It does not
install the candidate helper, modify the public runtime or production defaults,
change thresholds, or authorize a new model run. The candidate remains
unpromoted; no whole-network PASS follows from the reduced logit failure count.

## Evidence boundary and readback

The public files contain aggregates and public-source metadata only. Private
audio, contexts, raw tensors, per-recording identities, weights, runtime
binaries and private storage references are excluded. Independent readback
verified the sealed member identities and recomputed both saved logits/cache
aggregates, exact failure-position transitions and per-frame worsening counts.
It used no model forward, Torch import, compilation or decoder execution.
This validates the saved-result arithmetic; it is not an independent acoustic
test or a fresh qualification/CI run.

The historical old-head and frontend failures retain their original identities
and scopes. This result supersedes only the earlier statement that no
B-specific strict test had completed: a strict B-specific test has now
completed and failed on this fixed input/environment.

## Why no B76 decoder-threshold sweep follows

A separate read-only check of all three saved logit arrays found blank argmax
in every frame. Under the pinned 2599-class decoder rules and a fresh empty
decoder state, no target token is admissible: the floating-point softmax
upper-bound calculation peaks at approximately 0.001935, and even a coarse
0.002 envelope is 25 times below the fixed exclusive 0.05 token floor.
That bound uses exp(target_logit - maximum_logit), since the full softmax
denominator is at least one. It is not a formally certified rounding bound.
No full target candidate can be reached under those conditions, so a final
score-threshold sweep has no useful sensitivity coverage on this fixture.
This is a source-and-saved-logit deduction, not an executed decoder result,
an acoustic negative label or an independent test; strict failure is unchanged.

## 中文摘要

原 seed0 B 已完成限定输入、限定 CPU 环境的 76 帧网络专项严格数值比较，结论为
**STRICT_NUMERICAL_MISMATCH**，不再是“尚未完成 B 专项验证”。默认 C 的 logits
超限 209/197,524，cache 超限 3,983/428,032。首层 FP64 单臂只新增一次 76 帧 C
执行，复用已存 Torch/默认 C 结果；logits 超限降到 0，但 cache 升至 6,342，
新增 4,505 个失败位置、消失 2,146 个，全部 76 帧的 cache 失败计数均恶化。

候选已停止且不晋升。这里只记录首层数学差异，未把候选代码合入默认运行时，
未改生产默认、阈值或容差。旧头、旧前端、其他输入/后端、独立声学效果、FAR
和真实设备资格不能由此推断。原始私有材料未公开；本次仅对已存结果重新验算。

另对三组保存 logits 做只读检查：在冻结规则与空解码初始状态下，目标 token
达不到固定 0.05 准入下限，本夹具不能产生完整目标候选，因此不继续扫描末端阈值。
这不是已执行的 decoder 结果，也不能把未知源音频标成真负例。
