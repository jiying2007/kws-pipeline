# 辅助损失与 C 活动帧对齐配对实验（reviewing）

## 上一阶段停止点与新假设

[非活动 blank 约束](INACTIVE_FRAME_GHOST_TOKEN_EXPERIMENT_2026-09-28.md)消除了 Qwen3 片段开头非活动帧的伪 token，并把两个未见声线完整第二词的末字峰值拉回活动帧；默认 C 解码仍只有 2/4 命中、1 次缺首近邻误触发，真人近邻 175 次，高于旧模型 157。固定轨迹上只放慢 decoder blank 路径能恢复两个完整词，却保留缺首近邻误触发，说明训练端还没有直接把**辅助目标**绑定到 C 可触发的活动帧路径。

当前 `--ctc-vad-align` 只在 CTC 分支遮蔽非活动帧；ordered token、sequence margin、prefix completion、path purity 仍读取原始后验，可能对 C 无法触发的帧赋予序列/前缀奖励。本实验只改这些辅助目标读取的**时间轴**：非空 target 的每条录音按固定 -55 dBFS VAD 保留原始活动帧，按原顺序压紧；空 target 仍保留原始全段，以免负例失去背景监督。辅助损失使用新的活动帧长度，recurrent release、CTC、权重、数据、模型、阈值与 C runner 均不改。开关默认关闭，只有与 `--ctc-vad-align` 同时使用才允许。

预检中止记录：最初直接将辅助目标的非活动帧设为 blank-only，按原音频长度四等分的 ordered-token 目标未同步调整；140 条中有 **9 条 Qwen3** 训练样本的至少一个目标区间完全不含活动帧，loss 在该区间不可学习。该设计在约 100 epoch 的诊断运行中已主动中止，未生成 600 epoch 候选或 C 指标，不能记为模型质量负结果。现改为按活动帧压紧，并先用定向梯度测试证明原先空区间可学习，再重新执行本配对。两种处理的训练政策身份必须不同，不复用初版标识。

## 冻结配对

- 对照：原 AISHELL3 128 + 用户已审 Qwen3 12，共 140 条；旧 600 epoch seed2346 冷启动混合模型 KWM SHA-256 `d8b45c4586908e832091a9fc70fe43d34022e526197d5036afdedb6f5b2c41f4`。训练参数 feature32/H64、batch16、lr0.001，所有辅助损失默认权重、C runner 与关键词包固定。
- 处理：同一 140 条及训练配置，只启用 `--aux-vad-align`；**不同时启用**上轮失败的 `--inactive-blank-loss-weight`。输入 canonical SHA 应保持 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6`。先验证单帧遮蔽和 1 epoch，再完成一次 600 epoch。
- 评估：相同 C runner/词包回读 Qwen3 训练/未见声线、AISHELL3 calibration/test 和 HI-MIA-CW full；记录第二词末尾、缺首近邻与非活动帧后验。所有来源已观察，只能作开发结果。
- 接受门槛：Qwen3 未见声线至少 3/4 命中且“窝小窝”0 FA；HI-MIA-CW 事件不高于旧 157；不能用训练内 6/6 或 AISHELL3 第一词域内命中替代。
- `retry_budget`：只跑预设开关一次，不扫权重/阈值；`staleness_threshold`：任一输入/源码/C 制品身份变更须重新校验；`heartbeat`：烟测、训练、导出、评测每段记录 SHA。
- `goal_statement`：验证辅助目标与 C 可触发活动帧失配是否是漏检/近邻误触发的主要机制；`completion_claim`：只凭同 C 回读作开发判定；`logical_task_open=true`，`milestone_close=pending`，`stop_condition=replan`。产品放行证据另行收集。

## 有效处理的结果与停止点

修正后的活动帧压紧处理通过 Python 3.12/torch 2.13 的定向 VAD 梯度测试：原先首个四等分区间全静音的合成样本，现在在压紧时间轴的第一目标区间有可学习的活动帧梯度。默认关闭开关后的 1 epoch float state SHA-256 与历史同 140 条/seed2346 完全一致，为 `8a42a2e0cfc090511fb8431eb562ec883d6467325b74a8e8c3587f983b7c6b27`。新处理 1 epoch 和 600 epoch 的输入 canonical SHA-256 仍为 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6`；checkpoint SHA-256 `ce1c6f04938d3073e0d2472d899af8d8284080c3f83797b5b1c8d4c317b9f88f`，C KWM SHA-256 `ee6aa85d3dd2a9910d62312a550adbbfa3d0a82fe8437d8165e72cb36679e63c`。Provenance 回读到 `development-auxiliary-active-frame-compress-v1`，除开关外训练参数/损失权重与原 140 条混合模型相同；仅供开发。

| 已观察开发来源 | 旧固定模型 | 原 140 条混合模型 | 活动帧辅助目标模型 |
| --- | ---: | ---: | ---: |
| Qwen3 训练声线目标词 | 4/6 | 6/6 | 6/6 |
| Qwen3 未见声线目标词、近邻事件 | 3/4、0 FA | 2/4、2 FA | **4/4、1 FA** |
| AISHELL3 calibration 目标词、负例事件 | 14/16、0 | 9/16、0 | 13/16、1 |
| AISHELL3 test 目标词、负例事件 | 12/16、2 | 12/16、1 | 11/16、1 |
| HI-MIA-CW 16,343 条、6.6448 h 真人近邻事件 | 157 | 540 | **363** |

Qwen3 唯一近邻事件为 Serena 的“你好你好”误触发**词1**，C confidence `0.902061`；同声线真实“你好小窝”词1 confidence `0.697705`，因此一个统一上调的词1阈值无法同时保留这条正例并拒绝这条负例。两条音频的旧/混合/活动帧处理后验轨迹清单 SHA-256 `8009af708d26edec7ec34e94dcad1937e52e4cea7d68432190881ac45eae73e4`：新模型在“你好你好”上于活动帧内输出 `ni3,hao3,xiao3,wo1` top-1 完整路径，而旧模型没有该完整路径。这是声学模型对未见声线近邻的假完整词，不是单纯 terminal VAD 越界。

真人近邻事件 363 次中关键词1/2 为 310/53；“你好亚”123 次、“你好你好”44 次。正例召回改善没有迁移到真人近邻抗性，故**本模型不提升**，也不按同一小集合继续扫 VAD/阈值参数。Qwen3、AISHELL3 calibration/test、真人近邻检测 SHA-256 依次为 `ce142ee7677405af691f07a0fd6fde5c24b07e87057243fe1fb5a101c4d29123`、`75cdcde8cc99d81741ee2d035abd5b0415c6f3a3a565c6bb9d8d0e8641224e32`、`432f1b0f1371b645b2f4e301ee1cf7be00eb6732a532d41ca29674691b274020`、`ed98cb8b1f6b69cd672d21418a12fe3f5bc7c69800f369bcd416d790a766bf32`。

本逻辑配对已完成：`completion_claim=development-negative-generalization`，`logical_task_open=false`，`milestone_close=aux-active-compress-negative`，`stop_condition=replan`。下一研究变量应是以独立真人说话人的“你好你好”等近邻加强可分性，并冻结训练/回读说话人分组；此前 HI-MIA-CW 已用于开发观察，后续该来源不具新鲜封存身份。真人目标词、最终 AFE 和板端证据仍缺。
