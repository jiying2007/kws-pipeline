# 生产模型闭环推进记录：2026-10-29

本页记录审查后的实现和交付边界。源码基线为 `d09a6c55241d7f903e3c9e6e766d878454bbd321`；PR 和 Actions 链接提供后续状态。机器放行权威仍是 `configs/shipping.xiaowo.json` 与 `commercial/*.policy.json`，本页不替代这些门禁。

## 当前结论

已合入的修复提高了 WAV 边界、输出保护、事件序列化和硬负例标签的可靠性。新增训练证据类型检查与关键词输出保护已形成可审查提交和实际回归结果。新的生产模型尚未形成完整验收链：现有训练基座被限定为历史诊断用途；保留的 FSMN B/OR 没有获得生产晋升；最终 AFE 真人和实体目标板资格仍未完成。

产品范围保持两个完整唤醒词：**你好小窝**、**小窝小窝**。独立“小窝”不是当前 shipping wake word。

## 修复落地与精确证据

| 项目 | 2026-10-29 检查点 | 证据 |
| --- | --- | --- |
| binary32 阈值与 C 头文件一致性 | #660 已合入 | [PR #660](https://github.com/jiying2007/kws-pipeline/pull/660) |
| Unicode recording ID 的 JSONL 边界 | #661 已合入 | [PR #661](https://github.com/jiying2007/kws-pipeline/pull/661) |
| C WAV RIFF 声明边界与填充检查 | #662 已合入 | [PR #662](https://github.com/jiying2007/kws-pipeline/pull/662) |
| WAV CLI 统计输出覆盖输入保护 | #663 已合入 | [PR #663](https://github.com/jiying2007/kws-pipeline/pull/663) |
| 硬负例裁剪保留人工正例约束 | #655 已合入；主干 CI 成功 | [PR #655](https://github.com/jiying2007/kws-pipeline/pull/655)、[main CI 40826759240](https://github.com/jiying2007/kws-pipeline/actions/runs/40826759240) |
| PCM 身份、导出、源谱系准入 | #657 已与 #644 后主干整合至 `de243888aea27e81909f2f1d4201baea4cdd7dbc`；新集成 CI 与本地验收分别记录 | [PR #657](https://github.com/jiying2007/kws-pipeline/pull/657) |
| 训练 JSON 严格类型、关键词输出别名 | #664 初始实现 `93de8021b269e09ceb88be0ace17ccab08dd5817` | [PR #664](https://github.com/jiying2007/kws-pipeline/pull/664) |
| Actions 依赖身份维护 | #644 的依赖与当前 workflow 身份维护已合入 `6a38552d35bf6c48e407f489c93f6382477b300d`；PR CI 成功 | [PR #644](https://github.com/jiying2007/kws-pipeline/pull/644) |

#664 初始实现的实际执行证据：

- [GCC / 工程 CI 40835793148](https://github.com/jiying2007/kws-pipeline/actions/runs/40835793148)：CTest 7/7；关键词阈值 wire 14 个场景、0 个失败检查；关键词输出保护 57 个场景通过；frontend parity 与 reproducible SDK 通过。
- [训练/数据契约 40835793124](https://github.com/jiying2007/kws-pipeline/actions/runs/40835793124)：严格训练证据 JSON 类型 466 项检查通过；有效收据、非法类型、缺失字段、命名空间数组和错误时的 continuation 输出保护均通过实际验证器执行。
- 这些结论绑定上述初始实现。后续提交需要其自身的 required checks；此前成功不能替代最新提交验收。

#657 的整合保留全部 718 条原始保留源记录、provenance、科学验收固定身份以及先前 137 条 workflow 维护记录（含 #644），并保留原 #657 记录和新增整合记录。当前两个 workflow 的内容等于此前已审查的 #657 文件。原始实验失败和验收标准继续保留。

## 本地与主干验收

#657 和 #664 的本轮代码要求原授权 Dot 工作区对最终待合并提交执行 [CONTRIBUTING 的 24 条本地命令](../CONTRIBUTING.md#required-local-checks)，记录 HEAD、tree、工作区状态、退出码与原始日志。当前会话未获得该执行入口，本地状态为 **NOT_RUN**。公开 Actions 是单独的工程证据。

集成顺序：先完成各自最终代码的本地验收；检查当前 main 与 PR head；处理任何后续合并差异；等待六项 strict required checks；通过受保护的 squash 合并；再核验新 main CI。依据 [仓库治理](REPOSITORY_GOVERNANCE.md)，完整软件里程碑还要求清理已完成的实现 PR/分支。

## 新模型闭环：下一关与完成证据

| 阶段 | 当前状态 | 必须形成的交付 |
| --- | --- | --- |
| 1. 修复和工程集成 | IN_PROGRESS | #657 / #664 的精确提交、本地收据、受保护合并与新 main CI |
| 2. 新训练数据准入 | BLOCKED | 经审核的实际语音、完整标签和 speaker/session/source 身份；跨 train/development/密封资格集隔离；音频文件、解码 PCM、manifest 与审核收据一致 |
| 3. 候选选择和开发验证 | NOT_QUALIFIED | 冻结数据、初态、预算和比较变量；两个词分别记录真实运行时事件；独立 seed 验证召回与连续误触发；未观察的验收数据保持隔离 |
| 4. 可部署模型与来源绑定 | NOT_READY | 同一 source、checkpoint、model、tokens、keyword pack、config、训练环境、数据 manifest 与运行日志；通过适用的 frontend/network/decoder/ARM 检查，再执行正式训练与资格、不可变模型及 deployment 晋升 |
| 5. 真人最终 AFE（Phase A） | NOT_RUN | 冻结麦克风、结构和 AFE；私有原始/AFE 音频与独立标注；受控评分、覆盖与置信界收据 |
| 6. 物理目标板（Phase B） | NOT_RUN | 同一 deployment/AFE/SKU/revision 的多台 DUT，原始性能、功耗、连续性、soak 与外部 attestation |
| 7. 产品放行（Phase C） | NOT_RUN | Phase A 与 Phase B 及所有制品身份一致；由现有 promotion/approval 路径产生可回读的最终资格与 shipping 状态 |

### 最早的实际阻塞：训练数据

`configs/training/product-speech-like-base-v1.json` 仍指向历史 `speech-like-base-5204b798033f`，384 条记录的原始带宽为 8 kHz。当前 [训练入口验证器](../training/verify_training_entry_contract.py) 明确拒绝把 `historical-frozen-diagnostic-v1` 用于新产品训练；[入口契约测试](../tests/test_product_training_entry_contract.py) 和 [数据契约 workflow](../.github/workflows/product-training-data-contract.yml) 保留该拒绝结果。

下一次训练需要先提供通过现有 [语音基座审核](SPEECH_BASE_ADMISSION.md) 的新数据与审核收据，再绑定到有效训练配置。历史 calibration/test、正式 qualification、已观察过的开发材料应保持其实际暴露状态。先完成数据和开发验证，随后才更新独立的 model-training request；该请求 PR 按现有 workflow 要求只包含请求文件。

本轮正式 `model-training` 被 PR 条件跳过。466 项类型检查、训练可复现性 fixture 或其他 CI 成功都不是新的产品训练 run。

### 模型与数值证据

- 历史正式模型仍为 `model-749187ec1d66`，已镜像到 [模型 registry](../models/registry/model-749187ec1d66/registry.json)。它的历史资格不等于当前源码资格。
- 已回读的 [frozen nightly 40783278948](https://github.com/jiying2007/kws-pipeline/actions/runs/40783278948) 在 8 小时回归中有 2 次误接受，点估计 0.25/h，95% 单侧上界约 0.786974/h，聚合门槛失败。固定 seed 重跑不增加独立暴露时长。
- [FSMN B 模型卡](../research/fsmn2599-seed0-b-candidate-v1/MODEL_CARD.md) 保留候选身份；[固定 B76 数值结果](../research/fsmn2599-stream-c/B76-NUMERICAL-RESULT-2026-10-19.md) 仍为 STRICT_NUMERICAL_MISMATCH。保留的 B/OR、posterior-only decoder 和完整 C 网络属于不同验证范围。
- 量化、导出或 runtime 改动后，使用同一候选的精确输入和运行状态复验。修复数值差异需要可重算的候选权重、输入、cache 与参考运行记录；目前的负结果不能通过改宽容差转为生产验收。

正式资格 seed `271838` 已消费并冻结；下一个保留 seed 为 `271843`。新 seed 的消费必须发生在既有 guarded qualification 条件满足之后，并保留使用状态。

### 真人与目标板的具体验收合同

[Phase A policy](../commercial/real-human-qualification.policy.json) 要求至少 40 名说话人、每人至少 2 个 session、总计 1200 次预期唤醒且每词至少 600 次，以及至少 24 小时负音频。现行总 FRR 门槛为 5%、其 95% 上界为 6.5%；FAR 门槛为 0.1/h、其 95% 上界为 0.13/h；另有逐词、距离、方位、播放、双讲和运动切片门槛，以及 500 ms 的 p95 词尾后延迟要求。阈值的选择数据与最终资格数据需要独立。

[Phase B policy](../commercial/target-qualification.policy.json) 要求至少 3 台独立 DUT，每台 soak 至少 24 小时，其中至少 1 台达到 72 小时。CPU、RSS、stack、温度和平均功耗必须绑定冻结的 `resource-budget.json`；音频 xrun、丢样本、backpressure 和非预期 discontinuity 必须满足现行零事件门槛。原始 `runtime-soak.json`、`power.csv`、`audio-continuity.json` 以及外部 attestation 按现有采集合同保留。

这两份产品 policy 目前仍绑定旧 `deployment-c20f3eb88e43`。新的候选需要通过现有 [目标板与 shipping 晋升流程](TARGET_AND_SHIPPING_PROMOTION.md) 绑定新的 deployment，然后取得同一身份的资格证据。私人真人音频和完整私有执行日志继续使用受控入口；公开记录可包含去标识化统计、内容 hash、provenance 和 attestation。

## 完成判据

只有修复、数据、训练、部署、真人 AFE、实体板和最终 approval 的证据指向同一候选，且每一关实际通过，才声明“新的生产模型闭环完成”。当前 `shipping_approved=false`、`recalibration_required=true` 与上述未完成项一致。
