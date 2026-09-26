# 冻结语音上的训练预算与损失作用域对照（开发证据）

## 结论与范围

本次对照没有得到可提升的两词唤醒模型。在固定的干净合成语音池上，12 epoch 的两组 seed 均全拒；36 epoch 恢复了部分“小窝小窝”命中，同时出现近似词误触发，“你好小窝”仍未形成稳定命中。只收窄 ordered-token 监督范围或只切换序列 margin 的负样本路径策略，都没有跨两个 seed 同时改善召回和误触发。正式训练目标、阈值、模型发布和任何 protected qualification 均未改变。

这份记录只使用 train/calibration/development-test。没有读取冻结 qualification split；没有真人、最终 AFE、连续流或实体板证据。所有正确率均为真实 C runtime 事件评分，不把训练 loss、greedy 代理或短语音上的 FAR/hour 升格为产品指标。

## 输入身份与可比性

- 不可变语音 release：`speech-like-base-5204b798033f`；归档 SHA256 `18c0401b6b8d2d4a01d6b28f3d17ffbacb33a6c359e50e890d7882f478e4bfd6`。
- 开发池 `pool.json` SHA256：`20346a75b33086d8085ebee7e72eacf4baad55c453a241e4f6babd710e222ce5`；train/calibration/test 为 `128/64/64` 条。其中每个开发 split 有两词各 8 个正样本。
- C runner SHA256：`0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`；posterior dump SHA256：`0b2dfc61effd6ed4e2ad7fec4005023cd461019bc067866b97fbfb9e41feb154`。
- 本地实验为 Python `3.12.13`、PyTorch `2.13.0+cpu`、NumPy `2.3.2`、单线程确定性设置。治理训练锁定 Python `3.12.11`；因此本地浮点模型 hash 不与托管运行做跨环境 bit-exact 声明。
- 预算实验的源码树为 `fbd7ad58bc291318d105cf4ac659ed0e0fe2a6fe`。同 seed 的 12/36 epoch 两臂除 epoch 上限外，语料、损失、runner、源码树和前 12 个 epoch 的完整指标轨迹相同。
- ordered-token 范围实验的源码树为 `5e19ea69ad036d803989a6df0d7adbaec41b7dca`；负样本路径策略实验的源码树为 `d7025d1afceb19fbf329ca602f545e51ef6fa10a`。各组的 trial input、checkpoint 与模型 provenance 均回读并确认仅声明的控制项改变。两次扩展后的默认臂与扩展前相同 seed 的浮点状态 hash、test 命中和误触发完全一致。

`training/frozen_speech_ablation.py` 的 `current` treatment 显式采用 `runtime-executable-v1` 负样本 margin；治理训练默认是 `sparse-chronological-v1`，其余域渲染、replay、curriculum 也不同。因此干净语音结果只能回答声明的局部问题，不能直接替换治理训练结论。

## 1. 训练预算：12 与 36 epoch

固定 `variant=current`、seed 和所有训练/评估输入，只改变 epoch 上限。表内为“正确命中/预期唤醒，误触发次数”。

| Seed | Epoch | Train | Calibration | Development test |
| --- | ---: | --- | --- | --- |
| 1337 | 12 | 0/32，FA 0 | 0/16，FA 0 | 0/16，FA 0 |
| 1337 | 36 | 14/32，FA 3 | 7/16，FA 1 | 6/16，FA 4 |
| 2346 | 12 | 0/32，FA 0 | 0/16，FA 0 | 0/16，FA 0 |
| 2346 | 36 | 5/32，FA 13 | 2/16，FA 8 | 1/16，FA 4 |

训练损失与 ordered-token 准确率从 epoch 12 到 36 均改善，但 C 端仍同时存在高漏唤醒和误触发。因此“把正式训练缩成 12 epoch”不是已证实的修复；低损失也不是实际 KWS 可用性的充分条件。

## 2. Ordered-token 监督范围

固定 36 epoch、`runtime-executable-v1` 负样本 margin 和两个 seed，只把 ordered-token 损失从 `all-nonempty-targets-v1` 改为 `exact-configured-wake-targets-v1`。

| Seed | 范围 | Calibration | Development test | Test 关键词 1/2 命中 |
| --- | --- | --- | --- | --- |
| 1337 | 所有非空转写 | 7/16，FA 1 | 6/16，FA 4 | 0/8、6/8 |
| 1337 | 仅完整唤醒词 | 7/16，FA 2 | 6/16，FA 2 | 0/8、6/8 |
| 2346 | 所有非空转写 | 2/16，FA 8 | 1/16，FA 4 | 0/8、1/8 |
| 2346 | 仅完整唤醒词 | 2/16，FA 8 | 0/16，FA 1 | 0/8、0/8 |

收窄范围降低了部分误触发，但一个 seed 的召回退化为零；未恢复关键词 1。不得据此更改正式默认值。

## 3. 序列 margin 负样本路径策略

固定 36 epoch、`all-nonempty-targets-v1` ordered-token 范围和两个 seed，只比较 `runtime-executable-v1` 与 `sparse-chronological-v1`。

| Seed | 负样本路径 | Calibration | Development test | Test 关键词 1/2 命中 |
| --- | --- | --- | --- | --- |
| 1337 | runtime-executable | 7/16，FA 1 | 6/16，FA 4 | 0/8、6/8 |
| 1337 | sparse-chronological | 7/16，FA 3 | 6/16，FA 2 | 0/8、6/8 |
| 2346 | runtime-executable | 2/16，FA 8 | 1/16，FA 4 | 0/8、1/8 |
| 2346 | sparse-chronological | 3/16，FA 11 | 4/16，FA 6 | 0/8、4/8 |

一组 test 误触发减少，另一组召回与误触发同时增加；方向不稳定，且关键词 1 始终没有命中。不得把任一策略视为已证明的产品修复。

## 后验与词项归因

两组 seed 的关键词 1 在干净开发 test 上均没有完整 greedy CTC 序列。代表性路径出现 `ni3-xiao3-hao3-xiao3-wo1` 或在 `ni3` 前多出 `xiao3`，关键词 2 也常在目标前后多出 `ni3`/`xiao3`。按标注事件区间裁剪后验未稳定恢复完整序列；这种裁剪使用了 oracle 标注且受帧时间对齐影响，不是可部署推理方案。

在上述 36 epoch 模型中，量化推理与 C runtime logits 的最大差异量级为 `10^-5`；当前证据优先指向声学 token 路径和近似词可分离性，而非 C 数值实现偏差。训练数据中的“好小窝”“窝小窝”“你好窝”和换序词曾产生误触发，但短时 FA 次数不等于产品 FAR/hour。

额外只读检查显示：同一不可变基座的 train/calibration/test 共 64 条正样本中，停顿词事件中段低于 `-55 dBFS` 的最长连续 20 ms 帧至多 10 帧，未达到当前 12 帧解码器边界重置条件。此结果仅覆盖未渲染原始语音，不能替代域渲染/最终 AFE 的 VAD 与停顿验证。

## 下一步决策

停止把 epoch、阈值、ordered-token 范围或负样本 margin 单独当作修复。先在开发池上预声明一个**活动区/词序监督**假设，检查多余 token 出现在词前、词内还是词后，并比较真实 C 事件与近似词误触发；正结果必须在独立 seed、域渲染和连续流复现。任何新模型仍要走新的 formal seed、真人最终 AFE 与实体板资格，不能复用本次失败运行或旧 Release 的放行声明。
