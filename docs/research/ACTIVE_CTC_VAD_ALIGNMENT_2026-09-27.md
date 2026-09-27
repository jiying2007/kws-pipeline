# 正式训练器与 C 端 VAD 门槛失配归因（reviewing）

本记录只使用既有冻结合成语音开发池，pool SHA-256 为 `20346a75b33086d8085ebee7e72eacf4baad55c453a241e4f6babd710e222ce5`。train 128 条、32 次预期唤醒；calibration/test 各 64 条、16 次预期唤醒，未读取 qualification。评估使用同一 C runtime 和关键词包，严格事件匹配为词首前 0 ms、词尾后 500 ms。所有模型、posterior trace 和计数原始文件仅在本地 `/tmp`，不是发布候选；这些 calibration/test 样本已有多次研究观察，不能再当新鲜选择集。

## 机制定位

正式训练器的 CTC、ordered-token 和序列 margin 只看声学后验，C decoder 则要求终止路径所在帧 `speech_active`，并在 blank 主导帧按静音保留率衰减、在长时间 VAD 非活跃后清空旧前缀。36 epoch 原目标模型在 train 只命中 2/32；把预算增至 600 epoch 后，seed 1337 为 14/32、calibration 7/16、test 4/16，seed 2346 为 20/32、8/16、5/16。更长训练可以解除部分静默，但距离产品门槛仍远。

seed 1337 的 600 epoch 原目标模型中，32/32 train 正例的稀疏顺序路径代理分数超过 0.55，贪心 token 路径均包含目标词，C runtime 只触发 14/32。对完全相同的 C posterior trace 把所有 VAD 标记强制为活跃，train 命中由 14/32 增至 24/32，但误触发由 3 增至 7；test 由 4/16 增至 8/16，误触发由 2 增至 4。因此 VAD/路径交互是漏唤醒原因之一，直接绕开门槛又损害精度。只把普通状态保留率从 0.94 改至 0.99，对 train 事件无影响；提高 blank 帧保留率会提高召回也增加误触发。试验性地让 VAD 活跃 blank 按语音保留率衰减，现有 C 旧前缀过期测试失败，该 C 改动已撤回。

## 单变量训练对照

训练侧实验保持 C runtime、模型结构、初态、语料、样本顺序、batch、学习率、辅助损失和关键词阈值不变，仅在**非空转写样本的 CTC** 中把 C 判为非语音的帧强制当 blank；空转写负例保留原始 CTC。掩码用同一 400-sample 帧和 320-sample hop 的 PCM dBFS 公式及 C 默认 -55 dBFS 门槛计算。对 train 128 条录音的 6095 个 C posterior 帧逐项比对，训练侧掩码与 C `speech_active` 零差异。

| seed / 600 epoch | train 命中 / FA | calibration 命中 / FA | test 命中 / FA |
| --- | ---: | ---: | ---: |
| 1337 原目标 | 14/32 / 3 | 7/16 / 2 | 4/16 / 2 |
| 1337 VAD 对齐 CTC | 28/32 / 8 | 11/16 / 2 | 14/16 / 3 |
| 2346 原目标 | 20/32 / 5 | 8/16 / 2 | 5/16 / 2 |
| 2346 VAD 对齐 CTC | 25/32 / 0 | 14/16 / 0 | 12/16 / 2 |

36 epoch 的 seed 1337 对照也由原目标 train 2/32、calibration 0/16、test 0/16 提升到 VAD 对齐后的 16/32、4/16、6/16，但近邻误触发分别增至 19、7、8。600 epoch 后两个 seed 均提高双词召回，仍未通过 FRR/FAR；seed 1337 的 train 8 次和 test 3 次误触发全部来自关键词 2 的缺首音节近邻转写 `(4,3,4)`。现有严格前缀完成损失不覆盖这种后缀近邻，且负例 margin 的可微代理不能代替 C 事件。不能从训练损失下降推导产品合格。

## 落地边界

代码仅提供显式 `--ctc-vad-align` 开发实验入口；默认正式训练数学保持不变。启用时 checkpoint/KWM provenance 绑定 `development-pcm-dbfs-gated-ctc-v1`、-55 dBFS、参数契约 SHA-256 与开发标记；热启动拒绝把该 checkpoint 当正式来源，产品模型提升门禁拒绝其 provenance。重构后的 36 epoch 浮点权重 SHA-256 与原实验实现完全一致。此入口使用**C 的默认 PCM dBFS**，不能替代最终 AFE 后真实外部 VAD；改变 VAD 门槛或源会使结果失效。

下一阶段需在未被观察过的受控开发语料上预先声明双词 FRR/FAR 与连续流指标，并针对 `(4,3,4)` 等近邻负例设计与实际 C 路径一致的约束；随后才可考虑正式算法开关和 fresh qualification。真人最终 AFE、目标板与长时负样本证据目前仍缺失。
