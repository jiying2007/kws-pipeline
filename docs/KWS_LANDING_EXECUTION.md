# 唤醒词训练与产品落地执行检查点

## 冻结基线与目标

- 审查基线：`15b97d89b88835d722371e03a4bba3a7a2a6a65a`。修改工作树后应重新记录源码身份。
- 目标：形成同一源码、模型、关键词包、最终 AFE 和目标板身份绑定的两词唤醒产品候选，并按 Phase A、Phase B、Phase C 获得放行证据。
- 当前公开产品基线仍为 `model-749187ec1d66` / `deployment-c20f3eb88e43`，`shipping_approved=false`。合成资格不能替代最终 AFE 真人和实体板资格。
- 2026-09-26 的 `model-training` run `36237188880` 绑定源码 `15b97d89b88835d722371e03a4bba3a7a2a6a65a`。其诊断制品 `10907587838` 已回读：基础训练四轮均无 calibration/test 严格双通过；精修摘要明确是开发集门槛失败，calibration 命中 `3/64` 且有 12 次误触发，test 命中 `6/64` 且有 24 次误触发。正式资格 cohort 未生成，seed `271843` 未消费。这确认失败类别，不证明具体声学或优化机制。
- 缩减预算的 PR 预检 run `36228310178` 在第二轮达到 calibration `38/64`、test `49/64`；精修后为 `22/64`、`36/64`，同时误触发分别为 53、54 次。该预检仅要求每词保留非零命中，成功状态不是模型质量或完整预算可迁移性的证据。两次运行的预算和源码身份不同，不能直接把差异归因于某个损失或 epoch 数。
- 2026-09-24 的独立干净语音消融 run `35981037816` 在 36 epoch/seed 1337 下，全部关闭辅助损失的 CTC-only 模型 train/calibration/test 命中为 `0/32`、`0/16`、`0/16`；现有组合损失为 `14/32`、`7/16`、`6/16`，但 test 仍有 5 次误触发。该负结果只能否定“同时删除全部辅助项即可恢复”这一方向，不能分辨单项损失的贡献，也不能替代域渲染训练结论。
- 完整训练精修的 calibration 阈值曲线在 `0.50–0.60` 网格中执行 28 次试验，严格通过次数为 0；即使按各词分别取曲线最低 FRR，仍约为 `0.953/0.891`，最小 FAR 仍约 `110/h`。这只关闭当前模型与当前网格内的阈值微调方向，不代表网格外存在可放行点。
- 缩减预检的声学/C runtime 抽样每词每 split 为 8 条：calibration 关键词 1 有 6 条 greedy 子序列但仅 3 条被 C runtime 正确匹配；关键词 2 的 greedy 完整序列为 0 条，C runtime 仍匹配 6 条。该抽样显示代理分数、greedy 结构和实际事件之间并非一一对应，不能单凭任一代理指标判定训练成败。

## 执行阶段

| 阶段与责任 | 问题和交付 | 完成标准与验证 | 停止条件 |
| --- | --- | --- | --- |
| A：训练/算法 owner | 回读本轮 `adversarial-refinement/summary.json` 和 compact diagnostics，按每词 calibration/test 的 FRR、FA、阈值曲线判定失败类别。 | 已完成：确认四轮基础训练无严格候选、精修 calibration/test 双失败、formal seed 未消费；制品 `10907587838` 可复核。 | 机制仍未知，不把开发门槛失败自动归因为模型容量、epoch、阈值或代码异常。 |
| B：算法 owner | 若 A 证明是行为失败，以同一训练基座和实际 C runtime 做一个变量的开发集对照。优先区分声学序列缺失、训练 surrogate 与可执行解码路径不符、未说完整词误触发、跨片段前缀及阈值饱和。 | 冻结 train/calibration/test、模型初态、源码和训练预算；保留 per-keyword 曲线、连续流和上下文控制，预先声明通过条件；独立 seed 复现。 | 任何校验/测试集参与训练、多个变量同时改变、只靠零观测 FAR 或一个 seed 选胜者。 |
| C：候选 owner | 只在 B 有可重复改善时更新正式训练配置，重新执行完整训练、真实 C 评估、shadow、全新 formal qualification、robustness 和连续 FAR，再提升不可变模型及 deployment。 | 模型、包、配置、运行时、正式 seed 和 Release 证据逐项同一身份；所有原有门槛保持有效。 | 正式 seed 已消费、任何门槛失败、旧产品策略仍钉住其他 deployment。 |
| D：声学/硬件 owner | 用冻结候选准备最终双麦、结构、AFE、真人语料和受控 runner；先完成私有 intake 与 AFE 身份预检，再按 no-retry 规则运行 Phase A。 | 满足 `commercial/real-human-qualification.policy.json` 的人数、两词、场景、24 小时负样本及置信界要求。 | AFE、硬件或语料身份未冻结；保留集被调参或重复用作 fresh qualification。 |
| E：目标板/发布 owner | 同一 tuple 完成至少三台 DUT 的性能、音频连续性、功耗与 soak，再执行 Phase C。 | 每台至少 24 小时，一台至少 72 小时；原始证据、外部 attestation 和最终 approval Release 可核验。 | DUT、AFE、模型、关键词包或板卡 revision 漂移。 |

## 诊断读取顺序

1. `adversarial-refinement/summary.json` 若存在，先看 `qualified`、`record.calibration_gate`、`record.test_gate`、`record.calibration.per_keyword`、`record.test.per_keyword`。不完整的 summary 不能冒充算法失败。
2. `adversarial-refinement/progress.json` 的 `current_phase` 只说明最后记录的阶段；若没有 summary，原因仍为未知，需要认证任务日志。
3. `compact-training-diagnostics.json` 的 `adversarial_refinement_outcome` 是上述证据的摘要；`cause_verified=false` 时不得据此启动训练参数修改。
4. 在确认为模型行为失败后，先用 `acoustic_alignment` 和 C runtime 的实际事件区分声学、解码、VAD 与事件匹配，再选单变量试验。

## 方案与算法审查发现

| 类别 | 已验证的行为或证据边界 | 当前处理 |
| --- | --- | --- |
| 产品事件匹配 | 通用评分默认允许词首前 150 ms 检测匹配为真唤醒，并把其词尾后延迟截为 0。最小复现可同时得到 `matched=1`、`false_accepts=0`、`false_rejects=0`。 | Phase A 显式采用词首前 0 ms、词尾后 500 ms，并把两个容差写入评分摘要和产品策略；资格评分从标注与原始检测独立重算，再核对摘要、误报和漏报清单，以及语料 intake 与最终 AFE 已冻结的输入 hash。通用开发评分默认保持原样。 |
| 真人标注有效性 | 语料预检原先允许 `start_s == end_s` 的零时长正样本；这类标注可增加预期唤醒计数却不代表完整发声。 | Phase A intake 拒绝零时长正样本；保留通用评分器对其他开发测试的既有语义。 |
| 训练预检输入 | 原预检把小数/布尔计数转成整数，且未核对每词及整体 FRR 与计数；矛盾指标可被接受。 | 对资格和精修可用性检查采用原生非负整数，逐词及整体计数/FRR 必须一致；负例测试覆盖。 |
| 声学目标与 C 解码 | CTC/辅助损失是可微代理；实际 C 端另有主导 token、词根、VAD、前缀生存和边界重置约束。预检抽样中可观察到声学序列或 surrogate 达标但 C runtime 未匹配的记录，完整训练又呈现高漏唤醒和高误触发。 | 下一步在同一冻结数据上分开测量声学序列、可执行解码路径、事件匹配和训练预算；不凭单次 CI 红灯修改正式目标。 |
| 数据与产品分布 | 现有基座为离线合成语音，源音频带宽 8 kHz，场景仍用代理 AFE 与名义 60 mm 双麦。 | 仅作开发与合成回归；真实双麦、结构、最终 AFE 与真人资格另行获取。 |

## 阶段 B 的首个单变量对照

双 seed 的 12/36 epoch、ordered-token 范围和负样本 margin 单变量对照已完成。输入身份、真实 C 事件、每词结果、负结果与环境边界见 [`research/CLEAN_SPEECH_OBJECTIVE_CONTROLS_2026-09-27.md`](research/CLEAN_SPEECH_OBJECTIVE_CONTROLS_2026-09-27.md)。三个控制项均未形成跨 seed 的产品级改善，正式训练配置保持原样。下一步应先审词前/词内/词后的多余 token 与活动区监督，再设计单变量开发实验；没有稳定正结果不得消费新的 formal seed。

随后对已有启动上下文 `grounded-ctc` 做了活动区间限制 × 区间外 blank 监督的四组拆分及双 seed 关键复核，结果见 [`research/STARTUP_CONTEXT_FACTORIAL_2026-09-27.md`](research/STARTUP_CONTEXT_FACTORIAL_2026-09-27.md)。活动区间限制是当前最强的可学习性信号，blank 对连续流的补益随 seed 改变；两者仍未解决合成开发集上的近邻误触发，不能据此修改正式训练默认值或提升模型。下一个阶段 B 单变量问题是：在保留活动区间约束和 C 运行时边界的前提下，能否抑制不完整、重复与倒序近邻，同时保持两词命中。

阶段 B 后续用同一冻结池对 C 可执行/稀疏负例路径损失及 train-only 语速扰动做了受控实验，见 [`research/STARTUP_CONTEXT_INTERVENTIONS_2026-09-27.md`](research/STARTUP_CONTEXT_INTERVENTIONS_2026-09-27.md)。负例路径损失在已训练模型上几乎没有梯度；语速扰动在一个 seed 降误触发、另一个 seed 使训练可学习性失效。没有跨 seed 稳定候选。当前缺少受控真人正例、近邻负例、最终 AFE 和目标板入口，阶段 C/D/E 的产品放行仍需这些输入；不能以反复观察的合成开发集代替。

## 连续性与收口

- 本地检查（2026-09-26）：独立 `/tmp` CMake Release/strict 构建成功，5/5 CTest 通过；Python 3.12 下诊断、阶段进度、训练预检、事件评分、真人资格 fixture 和终态文档测试通过；`tools/kws_landing_status.py --verify` 与 `git diff --check` 通过。这些是源码/工具层证据，不包含本轮 PyTorch 精修复现或板端结果。
- `goal_statement`：获得可复核的两词产品放行，而非仅使一次合成训练运行变绿。
- `completion_claim`：当前完成源码级诊断增强、产品评分边界和训练预检输入加固，并确认本轮精修的开发集门槛失败；具体算法机制、模型候选和产品放行均未完成。
- `required_evidence`：阶段 A 的认证诊断、阶段 B 的配对开发集结果、阶段 C 的不可变模型与资格链、阶段 D/E 的受控现场收据。
- `claimant`：本地实现者；`verifier`：新鲜代码复审及对应算法、声学、硬件 owner。当前只有本地自审，不能代替 owner 验收。
- `open_items`：对训练预算与声学/解码失配开展单变量实验，并完成后续候选及产品资格。
- `plan_completeness`：A→E 阶段、责任、完成标准和停止条件齐全；当前 `logical_task_open=true`，`milestone_close=source-safety-hardening-only`。
- `attestation_readback`：本地源码检查、测试及运行 `36237188880` 的诊断制品可回读；新模型发布和现场资格收据仍待取得。
- `heartbeat`：每次推进记录最新 run/HEAD/验证日期；`staleness_threshold` 为外部 Actions 状态 24 小时。
- `stop_condition`：当前为 `replan`，阶段 A 失败类别已确认；阶段 B 先冻结数据、预算和比较变量，未有配对结果不得修改正式训练目标。
- 本逻辑任务的重试预算为同一失败类别最多两次定向验证；两次仍无增量时更新假设并重规划，不继续盲目全量训练。
- 外部 Actions 状态超过 24 小时未核对视为过期；每次引用运行结论需绑定 run ID、HEAD、状态和查询时间。
- 当前里程碑是训练失败分类与软件资格门禁加固。产品完成声明必须同时具备 C 端候选证据、Phase A、Phase B、Phase C 和最终同一 tuple 的证据回读。
- 私人真人原始/AFE 后音频、完整任务日志、凭证与二进制不得写入本仓或公开制品；只保留脱敏摘要、hash 和受控证据引用。
