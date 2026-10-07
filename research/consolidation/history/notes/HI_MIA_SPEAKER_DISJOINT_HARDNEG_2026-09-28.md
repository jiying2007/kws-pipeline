# HI-MIA-CW 真人“你好你好”近邻分组训练实验（reviewing）

## 问题与受控边界

[活动帧辅助目标模型](AUXILIARY_VAD_ALIGNMENT_PAIR_2026-09-28.md)在用户已审 Qwen3 未见声线目标词达到 4/4，但 Serena“你好你好”误触发词1、confidence `0.902061`，HI-MIA-CW 全量 16,343 条中误触发 363 次，其中该真人短语 44 次。对应 C 后验轨迹证实模型在这条负例上给出 `ni3,hao3,xiao3,wo1` 假完整路径。当前 AISHELL3 + Qwen3 训练池“你好你好”负例只有少量合成声线，缺跨说话人真人近邻监督。当前用户明确是非商业研究；[OpenSLR SLR120](https://openslr.org/120/) 的 CC BY-SA 4.0 语音只在本地受控实验使用，原 WAV 不入代码仓、正式模型或产品资格。

## 冻结数据处理

1. 从已回读的 16,343 条参考清单中只选官方 transcript 精确为“你好你好”的 1,365 条、35 人；按 SHA-256(`kws-himia-hardneg-20260928:` + speaker_id) 排序，前 20 人训练，余 15 人回读，切分不依据模型检出率。该数据来源此前已被查看，因此 15 人只能称**说话人不重叠的开发回读**，不是新鲜盲测。
2. 每个训练说话人最多取 6 条：从当前活动帧模型已触发的该短语中按录音名哈希取最多 2 条，再从其余按录音名哈希补足 6 条。明确记录哪些是模型驱动的 hard 选择；不让任何训练录音进入回读组。
3. 用固定 Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` 逐条转写入选候选；仅保留 NFKC、去标点/空白后与“你好你好”一致且 WAV SHA 匹配的音频。未通过的保持隔离。训练 token 监督为 `[ni3,hao3,ni3,hao3]`，不用 blank 标签掩盖真实“你好”前缀；不改变任何原 AISHELL3/Qwen3 样本与 split。
4. 使用前一活动帧模型相同的 140 条、相同 `--aux-vad-align`、RNN/H64、seed2346、600 epoch、batch16、lr0.001、相同损失/词包/C runner，仅增加这批真人近邻。冷启动，且模型/音频与 ASR/来源许可身份入 manifest/provenance。

## 评估与停止

- 第一门：已审 Qwen3 未见声线两词 4/4 保持，Serena“你好你好”和其他 Qwen3 近邻 **0 次**事件。若正例漏检增加或仍有近邻事件，停止提升。
- 第二门：HI-MIA-CW **15 名未训练说话人**全部短语上的事件不得高于旧固定模型在相同 15 人上的事件，并按关键词/短语分层；训练说话人的低误触发不算独立改善。另列 AISHELL3 calibration/test 仅作有发音疑点的域内参考。
- 本研究不承诺单次实验通过；若失败，不在同一组已观察资料上继续扫 hard 数量、阈值或 loss 权重。目标词真人正例、最终 AFE/板和连续负例仍需另采。
- `goal_statement`：验证真人“你好你好”近邻监督能否消除活动帧模型的假完整词且保留双词召回；`completion_claim` 待 C 回读；`required_evidence` 是分组、ASR 哈希清单、训练 KWM 身份、同 runner 比较。`retry_budget=1`、`staleness_threshold=source-or-policy-change`、`logical_task_open=true`、`milestone_close=pending`、`stop_condition=replan`。

## 输入检查点

实际固定说话人切分 SHA-256 `d9577a42b98748fb0691585117abfdf3722acaf07dd2db489abf092796d758b1`；120 条候选清单 SHA-256 `13f9546a830e44cf0531b05315d77747104feb3bef3ecdd9aa90408e0f715f86`，其中 16 条为活动帧模型触发的 hard 选择。15 名未训练说话人全短语 7,006 条、约 2.8071 小时，references SHA-256 `64b37be8be6cdd51fac4d53f997f1a6005a946b99fcf1b3aa918dcc2670780bd`；该组旧模型 51 次事件（关键词1/2 为 42/9），活动帧模型 143 次（125/18），均按同一批已观察源数据计算。

固定 Qwen3-ASR 对 120 条候选逐条回读，119 条与“你好你好”精确匹配，1 条隔离；机器收据 SHA-256 `3631faec845549ee25c159a0a731140f156d6e47db8a8d18f4d915a50e455fb8`。119 条训练 manifest SHA-256 `c07e20ed42cec67f9f8e38b4c037294679be0766b15e619ac1593284b78af5d1`，覆盖 20 人、全部 16 条 hard 选择，119 个 WAV SHA-256 唯一，且与原 AISHELL3 128/Qwen3 训练 12 条无 WAV 哈希重合。1 epoch 输入烟测读取 259 条、vocab5、feature32/H64，语料 canonical SHA-256 `f804d313bf557cd99d60a7da5363463a474a91e7762d0789c9564dd8516ec046`；这只验证数据与梯度路径，不代表最终模型改善。

为避免 `/tmp` 清理后丢失官方压缩来源，本地忽略目录 `build/himia-hardneg-20260928/source-bundles/` 已复制 exact OpenSLR 语音/资源包，SHA-256 分别为 `5de169ac1931a0eab46546c477965edad23069f0a2c0c4eb14e798814f83c91a` 与 `8628c75e6ec534a9458229d5ac5267e696b953b5f905c26f889187f6de1ee4e4`。它是本地可回读副本，**不是**已发布独立数据仓；原始真人语音不入 Git。

## 配对训练与停止结果

600 epoch 冷启动 checkpoint SHA-256 `d2fe946caa780c237604383f923685b98c0c39b082943e24d9460106871d59d9`，C KWM SHA-256 `ccdd0b6ac276ffc7def7597f6b1fec5bc08e6d344a07ae159d68d2690f79f512`。与前一活动帧模型的 feature32/H64、seed2346、batch16、lr0.001、600 epoch、VAD/辅助损失参数完全一致；语料 canonical SHA 从原 140 条的 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6` 变为 259 条的 `f804d313bf557cd99d60a7da5363463a474a91e7762d0789c9564dd8516ec046`。无可发布容器 digest，`development_only=true`。

| 已观察开发来源 | 旧固定模型 | 活动帧模型（140 条） | 新增真人近邻（259 条） |
| --- | ---: | ---: | ---: |
| Qwen3 训练声线目标词 | 4/6 | 6/6 | 6/6 |
| Qwen3 未见声线目标词、近邻事件 | 3/4、0 FA | 4/4、1 FA | **1/4、0 FA** |
| HI-MIA-CW 未训练 15 人、7,006 条近邻事件 | 51 | 143 | **6** |

15 人回读组事件由 143 降至 6（关键词1/2 为 5/1），其中“你好你好”3 次；真人负例改善显著。然而 Qwen3 未见 Serena 的双词均未命中、Eric 只命中词1，说明模型转为**过度拒绝**。Qwen3/15 人回读检测文件 SHA-256 分别为 `f2e7cb8d71e67ffb4f2c13ac937449cfe82da1b31512ef1477d18531a47b98f3`、`1e0f349ec0c12bb719147606cf3d696b9337972da3f818d920e02365f341d24a`。本模型不提升，不以 6 次负例事件替代目标词召回。

训练 exact wake 正例仍为 38 条，但总体非空训练样本从 140 增至 259；默认每条非空权重相同，exact wake 的原始权重占比从 `38/140≈27.14%` 降至 `38/259≈14.67%`。这提供一个可计算、可一次性配对检验的采样质量假设：保持同 259 条不变，仅将两词 exact wake 权重乘 `221/102≈2.1667`，使其相对非唤醒非空样本的权重比例回到原 140 条。该校正若仍不能同时保留 Qwen3 召回与15 人近邻抗性，停止同类权重优化，不按小集扫参数。

本轮逻辑任务完成，`completion_claim=negative-over-rejection-with-real-hardneg`，`logical_task_open=false`，`milestone_close=hardneg-unweighted-negative`，`stop_condition=replan`。
