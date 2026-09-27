# 关键词 2 非语音伪词根与近邻误触发归因（reviewing）

本检查延续 [`ACTIVE_CTC_VAD_ALIGNMENT_2026-09-27.md`](ACTIVE_CTC_VAD_ALIGNMENT_2026-09-27.md) 的冻结合成语音开发池。`pool.json` SHA-256 为 `20346a75b33086d8085ebee7e72eacf4baad55c453a241e4f6babd710e222ce5`；原 C runner SHA-256 为 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`。未读取 qualification，也未引入真人最终 AFE 或目标板证据。calibration/test 已多次用于诊断，结果只可检验失败类别，不能选产品参数或模型。

## 复现路径与排除项

seed 1337、600 epoch 的 VAD 对齐模型在 train 的 8 次、test 的 3 次误触发，全部来自非唤醒转写 `(4,3,4)`，对应关键词 2 `(3,4,3,4)` 的缺首音节近邻。8 条 train 误触发来自 8 个不同声音的同一近邻家族；其 posterior trace 均在约 20 ms 的 **VAD 非活跃帧**先出现 token `3`，随后真实的 `4,3,4` 接成唤醒路径。代表录音 `train-00007` 的该帧，Python 浮点模型输出 token `3` 概率约 0.9986，C 量化模型约 0.9985；全部 47 帧的 Python/C 特征最大绝对差约 `1.31×10⁻⁶`。因此这条误触发不是单独由 C 前端差异或 int8 导出造成，伪词根已存在于浮点模型输出。

现有 `src/decoder.c` 只要求终止候选时 `speech_active`，允许根节点在非语音帧启动。训练侧 VAD 对齐 CTC 把该帧当作固定 blank，却不向模型的原始 blank 后验反传梯度；这解释了为何伪词根仍可保持高置信度。把 C 根启动临时限制到语音活跃帧，seed 1337 的 600 epoch 模型 train 从 28/32 命中、8 FA 变为 28/32、0 FA，但 calibration 从 11/16、2 FA 变为 10/16、0 FA，test 从 14/16、3 FA 变为 13/16、1 FA；36 epoch 模型的 train/test 召回也下降。seed 2346 的 600 epoch 模型 train/calibration 不变，test 仍 12/16、2 FA。该 C 试验通过现有 5/5 CTest，但改变了产品解码语义；实验改动已撤回，不能在缺少最终 AFE 和新鲜资格时直接改默认运行时。

## 非语音 blank 监督单变量

在既有显式 `--ctc-vad-align` 开发模式上，只增加非空转写样本的 VAD 非活跃帧 `-log P(blank)`，并按输入帧数归一化；其余训练输入、辅助损失、C runtime、关键词包和评分不变。实验 checkpoint/KWM 只留在本地 `/tmp`，源码实验改动已撤回。seed 1337、36 epoch 的 test 由原 VAD 对齐的 6/16 命中、8 FA 变为 4/16、11 FA，说明短预算下没有改善。

| seed / 600 epoch | train 命中 / FA | calibration 命中 / FA | test 命中 / FA |
| --- | ---: | ---: | ---: |
| 1337 VAD 对齐基线 | 28/32 / 8 | 11/16 / 2 | 14/16 / 3 |
| 1337 加非语音 blank | 31/32 / 0 | 11/16 / 0 | 13/16 / 1 |
| 2346 VAD 对齐基线 | 25/32 / 0 | 14/16 / 0 | 12/16 / 2 |
| 2346 加非语音 blank | 27/32 / 0 | 10/16 / 0 | 12/16 / 2 |

seed 1337 的代表负例在补充监督后，原先非语音帧的高置信度 token `3` 变为 blank，训练集同类误触发消失；未见声音 test 仍有一条 `(4,3,4)` 误触发。seed 2346 的 test 没有收益、calibration 召回下降，两个 seed 不支持把新目标升为正式默认值。更低的训练损失或单一 seed 的零误触发也不能替代产品 FAR 上界与双词 FRR。

## 下一阶段判定

1. 冻结新的、此前未被查看的**开发**语音，包含 `(4,3,4)`、其他缺首/重复近邻、低能量首音节及连续流；对每个候选预先声明双词命中、误触发和 VAD 根启动诊断。现有 calibration/test 不再承担选择职责。
2. 分别验证训练侧非语音 blank 监督、C 根启动规则和最终 AFE 外部 VAD 的影响；任何 C 行为改变都需保留旧前缀过期测试，并重新做阈值、长时负样本与板端资格。
3. 只有跨 seed、跨未见声音和连续流同时改善且产品门槛通过，才可解除 development-only 标记，重新运行 fresh qualification。当前模型、正式训练默认值、C 默认解码器和发布状态均未变。
