# 真人近邻混训的 exact wake 权重守恒配对（reviewing）

## 假设与冻结变量

[119 条真人“你好你好”近邻混训](HI_MIA_SPEAKER_DISJOINT_HARDNEG_2026-09-28.md)把 15 名未训练说话人的事件从活动帧基线 143 降到 6，但已审 Qwen3 未见声线从 4/4 命中降到 1/4。训练 exact wake 样本 38 条未变，总非空样本由 140 增到 259，默认权重下 exact wake 的相对质量由 38/140 降为 38/259；这是过度拒绝的一个可检验因素。

处理只更改 `--wake-keyword-weights` 为 `{"1":2.1666666667,"2":2.1666666667}`，取两词相同因子 `(259-38)/(140-38)=221/102`。在总样本加权归一化之前，它使 exact wake 与其他非空样本的权重质量比回到原 140 条配方。其余 259 条 WAV/标签、20/15 人划分、ASR 收据、`--aux-vad-align`、RNN/H64、seed2346、600 epoch、batch16、lr0.001、损失和 C runner/词包不变；冷启动。本值由冻结数据计数公式决定，不在回读结果上扫描。

先 1 epoch 输入与 provenance 回读，再完成一次 600 epoch。第一验收：已审 Qwen3 未见 Serena/Eric 两词至少 3/4，近邻 0 FA；第二验收：HI-MIA-CW 未训练 15 人 7,006 条事件不高于旧固定模型在同组的 51 次。训练声线 6/6、负例单侧变好或 AISHELL3 域内第一词高分均不能替代两门共同满足。前述资料已观察，不是独立产品资格。

`goal_statement`：确定真人近邻加入后正例权重稀释是否是过度拒绝的主要可修因素；`required_evidence`：259 条语料同一 SHA、两词权重 provenance、KWM SHA、同 C runner 的正负分组结果；`retry_budget=1`，`staleness_threshold=source-or-policy-change`，`logical_task_open=true`，`milestone_close=pending`，`stop_condition=replan`。无真实目标词/最终 AFE/板端资格时不宣称模型落地。

1 epoch 输入与导出烟测完成：259 条 canonical SHA-256 与未加权真人近邻模型一致，仍为 `f804d313bf557cd99d60a7da5363463a474a91e7762d0789c9564dd8516ec046`；KWM provenance 回读两词权重均 `2.1666666667`。exact wake 在样本权重总质量中的占比由原 140 条的 `0.27142857`、未加权 259 条的 `0.14671815`，恢复为 `0.27142857`；这是由预声明公式得到的输入性质，不是模型质量结论。

## 600 epoch 配对结果

冷启动 checkpoint SHA-256 `dc7b07bc8282df0915e8dfcd7ce6d9514ad84e6da822211252fc421d8a7e79a7`；C KWM SHA-256 `de64633bd7ff67b456cc5548e5c2586f3d00892fa78e701e940767cc13b50421`。导出 provenance 仍为 `development_only=true`，两词训练权重均为 `2.1666666667`，语料 canonical SHA-256 与未加权 259 条模型相同。C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`、关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`。

| 固定开发回读 | 旧固定模型 | 活动帧 140 条模型 | 真人近邻 259 条未加权 | 真人近邻 259 条重加权 |
| --- | ---: | ---: | ---: | ---: |
| 已审 Qwen3 训练声线目标词 | 4/6 | 6/6 | 6/6 | 6/6 |
| 已审 Qwen3 未见声线目标词 | 3/4 | 4/4 | 1/4 | **4/4** |
| 同组 Qwen3 近邻事件 | 0 | 1 | 0 | **0** |
| HI-MIA-CW 未训练 15 人、7,006 条近邻事件 | 51 | 143 | 6 | **3** |

两道**预声明开发门槛**同时通过。15 人负例合计 2.807084 小时，重加权的 3 次均为关键词 1、均来自“你好亚”，发生在 2 人身上；原始事件率约 1.07 次/小时，只描述该固定语料，不外推家庭连续声学 FAR。Qwen3/HI-MIA 检测文件 SHA-256 分别为 `29a492ffcbf14a9a76788390e0d58f38310b9c468d56a2cab905daaedef62ae1`、`fae3ef5070f4bf4d2b88f2efaec4c3ead4649b3f9c6de651bbb81b1ca2716306`。这次单因素配对支持“真人负例加入后 exact wake 权重被稀释，是过度拒绝的一个可修因素”；它不能证明这是唯一原因。

## 跨生成器与场景诊断

再用未加入训练的 Spark-TTS 18 条、原始 Qwen3 审听声线的 80 条模拟声学场景运行**同一个 C runner/关键词包/模型**。Spark 参考清单 SHA-256 `182e4266403230bbe99a68a0b2ee1d4ad02754ccbc458801eb2e9d6ce88ad540`；检测/运行 provenance SHA-256 分别为 `79dca4bab81acee3ad414c8b0c18455bfca64d223fb7373695294d1f5c268543`、`e5c99db0e9a0af5a9b0ccfe8e93947c533dd00e1e08e617d13ab319123e5fa50`。按生成时预期文本的 12 条正例，检出 8 条；6 条缺首近邻中 2 条触发关键词 2，旧固定模型在同一批上为 0。仅看先前独立 ASR **精确匹配文本**的 8 条正例，检出 5 条；未检出的包括男性基础“小窝小窝”、女性低速“小窝小窝”、男性低速“你好小窝”。Spark 的来源级试听认为可用，但尚无逐条人工收据，故这些是跨来源警报，不报告为正式 FRR。两条误触发分别是女性基础“窝小窝”（ASR 转“小窝”，首字真值有疑点）和女性高速“窝小窝”（ASR 转“我小窝”，仍非完整唤醒词）。

Qwen3 80 条场景参考 SHA-256 `85cb9ab08650dc6f89a4a886b9ed1b10a835073fcf326583bd85b2ad9a725a7e`；重加权模型检测/运行 provenance SHA-256 分别为 `2c9b41014bd1c51cfdf2310e97cc2c7e579e825ed58b4983fde97729400c7c01`、`138c69edcbf2f0c1223b116f17658092a4e3e07706b4e5b54cc40c1a10f1a9f8`。40 条按原始审听音频文本衍生的正例中，17 条正确关键词触发，另有 2 条触发错误关键词；40 条近邻中 2 条误触发，均为中距离风扇场景的“窝小窝”。按场景分层的正确触发为 near_clean 10/10、mid_fan 6/10、far_motor 1/10、rear_playback 0/10。与旧固定模型在同一场景的 12 条正确触发和 5 条近邻事件相比，整体有改善但离模拟远场/播放稳健性仍有明显距离；场景派生标签未复听，代理 AFE 不代表最终 AFE。

**判定：**这是一枚通过预声明窄域开发门槛的研究候选，不能提升为产品候选。Spark 近邻误触发、跨生成器正例漏检以及场景远场漏检显示当前算法与数据覆盖仍不足。停止在这批已观察的 Qwen3/Spark/HI-MIA 小集合上继续调阈值、权重或 hard 数量；下一轮应先获得新的、按说话人和来源隔离且逐条发音可靠的双词正例及连续负例，并在最终 AFE/目标板上另行验收。`completion_claim=predeclared-development-gates-pass-cross-source-gap-open`，`logical_task_open=false`，`milestone_close=research-candidate-only`，`stop_condition=replan`。
