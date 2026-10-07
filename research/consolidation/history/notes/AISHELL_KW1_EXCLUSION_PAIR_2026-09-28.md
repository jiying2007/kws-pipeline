# AISHELL3 第一词疑似失配样本剔除配对实验（reviewing）

## 冻结问题

[固定 ASR 交叉回读](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)显示原 AISHELL3 train128 中“你好小窝”及停顿版共 16 条，两套 ASR 均为 **0/16** 与预期文本精确一致；另一词“小窝小窝”连续版在 Qwen3-ASR 下为 8/8。此结果不是逐条听辨真值，却足以把“第一词训练音频可能与四 token 监督失配”升级为可检验假设。[上一轮 128+12 Qwen3 混合训练](REVIEWED_QWEN_MIX_TRAIN_NEGATIVE_2026-09-28.md)在真人近邻上由旧模型 157 次变为 540 次，不能继续只调阈值。

## 单变量处理与边界

- 对照：上一轮冷启动 RNN/H64、seed 2346、600 epoch、batch16、lr0.001、VAD 对齐 CTC，原 AISHELL3 128 条 + 用户已审 Qwen3 训练声线 12 条，共 140 条；C 模型 SHA-256 `d8b45c4586908e832091a9fc70fe43d34022e526197d5036afdedb6f5b2c41f4`。
- 处理：**仅剔除** AISHELL3 中 `kind=positive && keyword_id=1` 的 16 条，保留其余 AISHELL3 112 条和同一 Qwen3 12 条，合计 124 条。原始 128 条索引、WAV、Qwen3 split 不变；不改损失、阈值、C runner、词包或 seed，不做暖启动。
- 输入实读：原冻结 `train.tsv` SHA-256 `23db9b52e8cbddd49baa6ec62eabb2ed7aa558cb8bb41100b0c84f72c722d039` 的 128 行 WAV 哈希与 token 序列逐行匹配 base `dataset-index.jsonl`。处理保留清单 SHA-256 `be8b25d8d074bc255d75ef0af207aaf859059c665dcd83fc74434150d187cb63`，剔除清单 SHA-256 `9a2f4a2621885bd730efe91b5d2c5eb34cc0e28c9f4ab1f5626c1780930553bc`；仅移除上述 16 个 source ID。1 epoch 烟测读取 124 条，corpus SHA-256 `8d7a6f3240edd7911b10881510f19f968fe1402fb588af7afa73767af49716f5`，checkpoint SHA-256 `f8ea3602703d9a73b37a724eab45570c5db4c3cd3950a803390317caba2c6599`。本次 `training/train_ctc.py` SHA-256 `0d3823307edac97132cb00a568fb605518cacd678b46f63e66fcb7d71410ad4b`，与上一轮混合训练 provenance 相同；Python 3.12、torch 2.13.0+cpu 环境沿用已存在的本地训练 venv。
- 这是开发诊断，不宣称新的说话人或独立封存集；Qwen3 Serena/Eric、AISHELL3 calibration/test 与 HI-MIA-CW 都已查看。由于处理后第一词训练正例大幅减少，若第一词召回下降不能直接证明原 16 条发音正确；只有与真人近邻误触发和另一词命中一起比较才能判断处理方向。
- 评估标签边界：[Qwen3-ASR 对 AISHELL3 calibration/test 正例回读](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)发现第一词各 0/8 精确匹配，第二词为 5/8 和 7/8。因同来源偏差，AISHELL3 calibration/test 的第一词 C 事件仅作历史域内诊断；主要看用户已听准的 Qwen3 两词和真人 HI-MIA-CW 近邻。
- 软件验收：先定向 1 epoch 输入烟测与训练输入 SHA 校验，随后完成 600 epoch；用**同一** C runner/词包回读 Qwen3 训练与未见声线、AISHELL3 calibration/test 和 HI-MIA-CW。若两词完整命中与近邻误触发不能共同改善，处理模型不得提升，也不继续同类数据剔除尝试。

## 长任务检查点

- `goal_statement`：判断 AISHELL3 第一词疑似失配是否是混合训练退化的可操作因素。
- `completion_claim`：仅在配对训练、C 端结果和定向复核后陈述；产品模型仍未放行。
- `required_evidence`：剔除清单、源 WAV 哈希、训练命令与 checkpoint/KWM 身份、同 runner 逐来源结果。
- `claimant`：本地算法实验；`verifier`：训练输入审计、同一 C runner 重算及新鲜结果复核。
- `logical_task_open`：true；`milestone_close`：待本轮数据消融回读。
- `retry_budget`：同一失败类别最多两次定向修正；`staleness_threshold`：本地输入/制品一经变动立即失效；`heartbeat`：每阶段记录输入及制品 SHA。
- `stop_condition`：当前为 `replan`；任何标签真值不确定、训练无法复现、资源超限或 C 指标退化时停止候选提升。
- `attestation_readback`：以生成后的 manifest、模型和评测摘要实读为准；仅计划不算完成证据。

## 配对运行与结果

600 epoch 处理 checkpoint SHA-256 `e2af939dc34a14b0ead15f2acb339d5a30c059b425f264cea61fc09f3893ceab`，C KWM SHA-256 `8a2dc8abe74c32cc6eae39c6a60e5344c8965ddf5be0a4b7e588333a154855`。上一轮混合模型和本轮处理模型 provenance 中 batch、epoch、lr、seed、VAD 对齐、辅助损失、ordered token scope、负例策略、path purity 配置，以及特征 32/H64 全部一致；语料 canonical SHA-256 由 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6` 变为 `8d7a6f3240edd7911b10881510f19f968fe1402fb588af7afa73767af49716f5`。两模型均为开发用，本地 checkpoint 未绑定可发布容器 digest。

同一 C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39` 和词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`。HI-MIA-CW full references SHA-256 `0c5ce6bd2c0bafb5961f81cfe449e897cb9cde52bfcde055d66f635184b5637b`，AISHELL3 calibration/test references SHA-256 `b3f5d5466dc20043de978822aa4fdad5e0b05c83a1e3cfdb965878fd8c53293c` / `8becfdd31a36da1a0cd9cf8dbb522d8c8d3206f1e055140f7bc616b9066be74a`。

| 已观察开发来源 | 旧固定模型 | 128+12 混合模型 | 剔除 16 条处理模型 |
| --- | ---: | ---: | ---: |
| Qwen3 训练声线目标词 | 4/6 | 6/6 | 6/6 |
| Qwen3 未见声线目标词与近邻事件 | 3/4、0 FA | 2/4、2 FA | **2/4、2 FA** |
| AISHELL3 calibration 目标词、负例事件 | 14/16、0 | 9/16、0 | 8/16、0 |
| AISHELL3 test 目标词、负例事件 | 12/16、2 | 12/16、1 | 9/16、3 |
| HI-MIA-CW 16,343 条、6.6448 h 真人近邻事件 | 157 | 540 | **137** |

处理模型在 HI-MIA-CW 上关键词1/2 事件为 103/34；旧模型为 132/25，混合模型为 430/110。与旧模型误触发录音只重合 9 条，说明事件分布明显变化，不能仅凭总数 137 断言模型更安全。处理模型的常见触发短语为“好米好米”25 次、“你好亚”20 次、“你米亚”18 次。Qwen3 未见声线两个词各 1/2，近邻事件仍为 2；处理并未恢复已审真实目标词的跨声线泛化。

处理模型的 Qwen3/校准/测试/真人近邻检测文件 SHA-256 依次为 `60e1d94715848746f5bd67734de7e2602712bce3d5bc5c2a675f02b7d0b191ec`、`0a743fc8122121c277fc47e92f66d9c02d044c09c014bd06d0c6e6ec29f24a74`、`71ac32c56ab54df2512a8b81692c9d57ae8f608ae5feceb3641d8c52226de6ce`、`27dd28daa62edcb4873db7299b1d73eeb0ca9904891303b13827fabb5eb04556`。

## 判定

**单独剔除疑似失配的 AISHELL3 第一词 16 条不是完整解法，处理模型不提升。** 数据噪声对真人负例事件有显著影响，但 Qwen3 未见声线的命中/近邻误触发没有同步改善。下一步停止重复同类剔除或盲目补 seed；需要分辨现有四 token CTC 辅助损失、词前缀/末尾完整性与 C 解码器触发目标之间的失配，并在可信双词正例及独立负例上配对验证。第一词的真人语料与最终 AFE/目标板仍缺。

`goal_statement`：已通过输入哈希、训练/导出与 C 端同 tuple 回读完成这次机制分诊；`completion_claim` 仅为开发负结果。`logical_task_open=false`，`milestone_close=negative-data-ablation`，`stop_condition=replan`。没有产品模型完成声明。
