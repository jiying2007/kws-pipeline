# 唤醒词训练与产品落地执行检查点

## 冻结基线与目标

- 审查基线：`15b97d89b88835d722371e03a4bba3a7a2a6a65a`。修改工作树后应重新记录源码身份。
- 目标：形成同一源码、模型、关键词包、最终 AFE 和目标板身份绑定的两词唤醒产品候选，并按 Phase A、Phase B、Phase C 获得放行证据。
- 当前公开产品基线仍为 `model-749187ec1d66` / `deployment-c20f3eb88e43`，`shipping_approved=false`。合成资格不能替代最终 AFE 真人和实体板资格。
- 截至 2026-09-26，`model-training` run `36237188880` 的基础训练成功，精修任务失败。公开任务状态只能定位到精修步骤；具体退出原因需要该运行的认证日志或 `xiaowo-training-diagnostics` 制品，不得由步骤名称推断。

## 执行阶段

| 阶段与责任 | 问题和交付 | 完成标准与验证 | 停止条件 |
| --- | --- | --- | --- |
| A：训练/算法 owner | 读取本轮精修日志、`adversarial-refinement/summary.json`、`progress.json` 和 compact diagnostics；按每词 calibration/test 的 FRR、FA、阈值曲线及阶段进度判定失败类别。 | 记录实际异常或明确的门槛失败、源 SHA、有效配置 SHA、候选模型 SHA、失败阶段；分类能由原始制品复核。 | 诊断制品不可读时保持 `cause=unknown`，不改损失、epoch、阈值或 seed。 |
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
| 产品事件匹配 | 通用评分默认允许词首前 150 ms 检测匹配为真唤醒，并把其词尾后延迟截为 0。最小复现可同时得到 `matched=1`、`false_accepts=0`、`false_rejects=0`。 | Phase A 显式采用词首前 0 ms、词尾后 500 ms，并把两个容差写入评分摘要和产品策略；资格评分从标注与原始检测独立重算，再核对摘要、误报和漏报清单。通用开发评分默认保持原样。 |
| 真人标注有效性 | 语料预检原先允许 `start_s == end_s` 的零时长正样本；这类标注可增加预期唤醒计数却不代表完整发声。 | Phase A intake 拒绝零时长正样本；保留通用评分器对其他开发测试的既有语义。 |
| 训练预检输入 | 原预检把小数/布尔计数转成整数，且未核对每词及整体 FRR 与计数；矛盾指标可被接受。 | 对资格和精修可用性检查采用原生非负整数，逐词及整体计数/FRR 必须一致；负例测试覆盖。 |
| 声学目标与 C 解码 | CTC/辅助损失是可微代理；实际 C 端另有主导 token、词根、VAD、前缀生存和边界重置约束。已留存的开发实验观察到不完整词与跨片段前缀误触发。 | 先读取本轮精修诊断，再在冻结开发数据上做单变量对照；不凭代理损失或一次 CI 红灯修改正式目标。 |
| 数据与产品分布 | 现有基座为离线合成语音，源音频带宽 8 kHz，场景仍用代理 AFE 与名义 60 mm 双麦。 | 仅作开发与合成回归；真实双麦、结构、最终 AFE 与真人资格另行获取。 |

## 连续性与收口

- 本地检查（2026-09-26）：独立 `/tmp` CMake Release/strict 构建成功，5/5 CTest 通过；Python 3.12 下诊断、阶段进度、训练预检、事件评分、真人资格 fixture 和终态文档测试通过；`tools/kws_landing_status.py --verify` 与 `git diff --check` 通过。这些是源码/工具层证据，不包含本轮 PyTorch 精修复现或板端结果。
- `goal_statement`：获得可复核的两词产品放行，而非仅使一次合成训练运行变绿。
- `completion_claim`：当前完成源码级诊断增强、产品评分边界和训练预检输入加固；训练故障根因、模型候选和产品放行均未完成。
- `required_evidence`：阶段 A 的认证诊断、阶段 B 的配对开发集结果、阶段 C 的不可变模型与资格链、阶段 D/E 的受控现场收据。
- `claimant`：本地实现者；`verifier`：新鲜代码复审及对应算法、声学、硬件 owner。当前只有本地自审，不能代替 owner 验收。
- `open_items`：读取本轮诊断、判定精修失败类别、开展单变量实验、完成后续产品资格。
- `plan_completeness`：A→E 阶段、责任、完成标准和停止条件齐全；当前 `logical_task_open=true`，`milestone_close=source-safety-hardening-only`。
- `attestation_readback`：本地源码检查与测试可回读；训练、发布和现场资格收据仍待取得。
- `heartbeat`：每次推进记录最新 run/HEAD/验证日期；`staleness_threshold` 为外部 Actions 状态 24 小时。
- `stop_condition`：当前为 `split`，先完成可独立验证的诊断里程碑；缺少阶段 A 证据时禁止进入算法改动。
- 本逻辑任务的重试预算为同一失败类别最多两次定向验证；两次仍无增量时更新假设并重规划，不继续盲目全量训练。
- 外部 Actions 状态超过 24 小时未核对视为过期；每次引用运行结论需绑定 run ID、HEAD、状态和查询时间。
- 当前里程碑仅是训练故障分类与非行为诊断增强。产品完成声明必须同时具备 C 端候选证据、Phase A、Phase B、Phase C 和最终同一 tuple 的证据回读。
- 私人真人原始/AFE 后音频、完整任务日志、凭证与二进制不得写入本仓或公开制品；只保留脱敏摘要、hash 和受控证据引用。
