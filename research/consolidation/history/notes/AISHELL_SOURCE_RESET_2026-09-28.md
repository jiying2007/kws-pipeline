# 移除 AISHELL3 低置信标签来源的训练池重置（reviewing）

## 触发与冻结假设

277 条训练池的新增 18 条 Qwen3 音频均通过固定 ASR 精确转写和连续正例低能量间隔门槛，但同配方 600 epoch 模型在已审 Serena/Eric 双词由 4/4 降至 2/4，并出现“你好你好”误触发；独立 Spark 通过 ASR 的关键词 2 从旧模型 2/4 降至 1/4。**停止同家族扩量与参数扫描。**原 AISHELL3 128 条按两套 ASR 精确匹配仅 14/128、22/128，第一词连续/停顿样本均 0/16；这是可检验的来源和监督矛盾线索，不证明每条真实读错。

本轮仅做一次**训练来源重置**：从上一轮 277 条中移除原 AISHELL3 128 条，保留用户逐条确认的 Qwen3 训练声线 12 条、新批次经固定 ASR/连续性门槛的 Qwen3 18 条、20 名训练说话人的 HI-MIA-CW ASR 合格“你好你好”119 条，总计 149 条。真实近邻的 15 名留出说话人仍完全不入训；Serena/Eric 两批、Spark 两批和模拟场景也不入训。仍用相同 RNN/H64、seed2346、600 epoch、batch16、lr0.001、CTC VAD 与活动帧辅助目标、C runner/关键词包。source reset 会让 exact wake 从原 38 条变为 14 条，因此按**预设训练质量比** `wake_weight=(38/102)*(135/14)=3.5924369748` 同时设置两词；该值只由输入计数计算，不从 C 回读扫描。

验收沿用[软件侧闭环检查点](SOFTWARE_ONLY_CLOSURE_2026-09-28.md)：已审未训练声线双词 4/4、近邻 0；真人近邻 7,006 条事件不超过 51；独立 Spark ASR 合格词1不少于旧模型 2/2、词2不少于旧模型 2/4、合格近邻 0；新 Serena/Eric ASR 合格子集逐词不低于旧模型且近邻 0；模拟 near_clean 正例不少于旧模型、mid_fan 近邻 0；900 秒固定连续流 0 次。全部为已观察/新封存的软件研究来源，**即使通过也不宣称真人目标词或产品资格**。

`goal_statement`：检验 AISHELL3 低置信监督是否主导跨来源退化；`required_evidence`：3 份固定训练 manifest SHA、ASR 收据/连续性、训练 corpus SHA、KWM/runner/词包 SHA、逐组 C 事件；`retry_budget=1`；`staleness_threshold=source-or-policy-change`；`claimant=本地实现者`；`verifier=新鲜复验`；`completion_claim=pending`；`logical_task_open=true`；`milestone_close=pending`；`stop_condition=replan`。若仍不过共同门槛，停止四 token CTC/RNN 的同式训练，改走显式事件/拒绝建模或更大可靠训练来源的设计评审。

## 配对回读与判定

149 条输入 1 epoch 烟测、600 epoch 冷启动和 C 导出完成；训练 corpus canonical SHA-256 `e1ea1eb23c6975c8b07b0b4749eead4e0933e1e6cbbde158c6735b574ab7a610`，C KWM SHA-256 `09ac1e0236b9a62fa920cd113220fa0ee6b9faf195c1bbbe32119ad64a84a70c`。与 277 条对照使用同一训练脚本、模型结构、seed、epoch、C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39` 和关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`；配方中移除了 AISHELL3，同时由固定计数公式调整正例权重，故只能归因于**来源重置配方**整体，不能声称单独移除 AISHELL3 的纯因果效果。

[机器回读报告](../../build/software-closure-20260928/source-reset-software-report.json) SHA-256 `533f05fe4eb90a751e44a4a29827d8957a7bec01f8e226564cc6ab590c3a2d40`，逐一核对 model/runner/pack/reference/detection SHA，结论 `software_candidate=false`：

| 软件回读 | 来源重置结果 | 门槛/解释 |
| --- | ---: | --- |
| 已审 Serena/Eric 未训练声线双词 | 正确 1/4，另 1 条词2误判词1；近邻0 | 需要 4/4 与近邻0 |
| 新 Serena/Eric ASR 合格正例 | 词1 1/1、词2 0/1；近邻0 | 两词各 1/1 |
| 独立 Spark ASR 合格正例 | 词1 2/2、词2 2/4；合格缺首0 | 达到旧固定模型，但非跨来源净改善 |
| HI-MIA-CW 15 名留出说话人 7,006 条 | 0 次事件 | 低于旧模型 51 次，但单侧过度拒绝 |
| Qwen3 代理近场正例/近邻 | 正确 3/10、近邻1次 | 低于旧模型近场 8/10，近邻应为0 |
| 900 秒固定代理连续流 | 完整注入15个近邻，1次事件 | 旧/上一模型同流均为0；0.25小时不能估产品 FAR |

训练最后几个 epoch 的 loss 有波动，最终第 600 epoch 为 `0.076570`，所以不能只以训练拟合或低负例事件推断稳定性。**此来源重置没有形成软件候选**，且与 277 条扩量模型的失败类别不同：它在独立 Spark 文字合格子集上回到旧模型水平，真人短音频误触发降为零，却显著损失已审 Qwen3 正例，并在固定连续流产生事件。当前五类输出表的同式配方重试预算已用尽；下一步按[显式近邻音节竞争设计](FOUR_TOKEN_VOCAB_LIMIT_2026-09-28.md)验证结构性问题，不以同一小集扫阈值或权重。`completion_claim=source-reset-negative-over-rejection`，`logical_task_open=false`，`milestone_close=five-class-pool-reset-negative`，`stop_condition=replan`。
