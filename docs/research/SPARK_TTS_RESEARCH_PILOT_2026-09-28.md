# Spark-TTS 非商业研究小样本（reviewing）

## 范围与来源

用户已明确当前是非商业研究，允许导入更多 TTS。本轮实际导入 [Spark-TTS-0.5B 官方模型](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) 和[官方推理代码](https://github.com/SparkAudio/Spark-TTS)，模型 revision `642071559bfc6346c2359d19dcb6be3f9dd8a05d`，代码 commit `2f1ea9082400547242641f5271b6f941c9f439d1`。权重许可为 **CC BY-NC-SA 4.0**，与代码仓 Apache-2.0 区分。所有原始/派生音频仅作为 `research-only-cc-by-nc-sa-4.0`，保存在 `.gitignore` 的 `build/` 下，不并入未来商业产品语料。

权重 SHA-256：LLM `54825baf0a2f6076eb3c78fa1d22a95aee225f59070a8b295f8169db860eb109`，BiCodec `e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec`，wav2vec2 `314340227371a608f71adcd5f0de5933824fe77e55822aa4b24dba9c1c364dcb`，config `49a07abffcdfb5358e91eda83b1f4f8068f54b0a4e1ef1ef6ac22d9a312a7ae3`。固定这些文件是生成来源追溯，不证明声音权利或词读音。

## 基础六条预检

在 Python 3.12、CPU、隔离且 `--network none` 的容器中运行官方 `SparkTTS` voice creation，无参考音频。女性控制：pitch `moderate`、speed `moderate`；男性控制：pitch `low`、speed `moderate`。每种控制分别生成“你好小窝”“小窝小窝”“窝小窝”；seed 3101–3106。输出均为 16 kHz 单声道 PCM16，6 个文件 SHA-256 不同，manifest SHA-256 `ce73aaf5c34f449c77db9185c82fc4943337f665251c25f1b9e910c31a943df4`。

| 控制声线 | 文本 | 独立 ASR 转写 | 旧 C 模型事件 | Qwen3 混合训练模型事件 |
| --- | --- | --- | --- | --- |
| 女性 | 你好小窝 | 你好小窝 | 词1 | 词1 |
| 女性 | 小窝小窝 | 小窝小屋 | 词2 | 词2 |
| 女性 | 窝小窝 | 小窝 | 无 | 无 |
| 男性 | 你好小窝 | 你好小吴 | 词1 | 无 |
| 男性 | 小窝小窝 | 小窝小窝 | 词2 | 无 |
| 男性 | 窝小窝 | 窝小窝 | 无 | 无 |

独立 ASR 摘要 SHA-256 `2d6e5838716c739edd1ee4f9b06053b6f2441b59f895d53c0688ed3c16ecb35d`；C 输入 corpus SHA-256 `fe0a5bf67e7311f33dc94ef5c87a1f4a67fd069e91eba7141eef87de0000ed58`。ASR 的“小屋”“小吴”及女性近邻缺首是**发音或转写疑点**，需要人工复核；旧 C 的目标词事件也不等于音频读对。样本量太小，不能估计 FRR/FAR，也不能把男性/女性控制风格当作真人性别或年龄覆盖。

本地 [六条逐条审听页](../../build/review-packs/kws-spark-review-20260928/review.html)及[便携包](../../build/review-packs/kws-spark-review-20260928.zip)已生成，ZIP SHA-256 `378923b8e6b2cff44414a719eaa45e72401bfed6b9b14e82793747b06fe9fe86`；可导出与 WAV SHA-256、文本、类别、匿名审听 ID 绑定的 JSONL 收据。**六条仍待人工审听，不进入合格语料或训练。**

用户已给出来源级试听反馈：Spark-TTS 的结果与 ASR 疑点相近，但总体可用；听感低于 Qwen3，高于此前试过的其他来源。这是**来源级排序**，不代表六条逐条发音全合格；用户表示会补交逐条收据。

## 男/女 × 低/高语速十二条探针

同一官方模型、相同两组性别/音高控制，增加 `speed=low/high`，每组合各双目标词与缺首近邻，总计 12 条；seed 3201–3212、16 kHz 单声道 PCM16、各 WAV SHA-256 唯一。manifest SHA-256 `2abf67282e1111b50cd08eba3ccc9799955dae826868f870cba221e97d558afe`；ASR 摘要 SHA-256 `825bdf484c400b305e4402be6859411e7cde6a1221b6bf2fa2badf9fa18e4518`，C 输入 corpus SHA-256 `655f5b940eee836bf8cc668c8a93513214155458e8b226ff9d00ebbfb37e7be1`。

| 控制组合 | 双目标词独立 ASR 精确转写 | 旧 C 模型目标词事件 | Qwen3 混合模型目标词事件 | 缺首近邻 C 事件 |
| --- | --- | --- | --- | --- |
| 女 / 低速 | 2/2 | 1/2 | 1/2 | 0 |
| 女 / 高速 | 1/2；“你好小窝”转为“你好小吴” | 2/2 | 1/2 | 0 |
| 男 / 低速 | 2/2 | 2/2 | 1/2 | 0 |
| 男 / 高速 | 1/2；“你好小窝”转为“你好小吴” | 1/2 | 1/2 | 0 |

低速组两性别的 4 条目标词都被 ASR 精确转写，高速组为 2/4，但每个速度单元使用不同 seed，**不能把差异单独归因于语速**。四条缺首近邻的 ASR 转写分别是“郭小窝”“我小窝”“我小窝”“我想我”，其中可能有生成或转写错误；C 端均未触发。旧 C 模型在高音速疑似“小吴”的女性 kw1 上仍触发词1，进一步表明 C 事件不是音素真值。模型在此矩阵中未见负例事件，样本量仍不足以估计 FAR。

[十二条语速探针审听页](../../build/review-packs/kws-spark-speed-review-20260928/review.html)和[便携包](../../build/review-packs/kws-spark-speed-review-20260928.zip)已生成，ZIP SHA-256 `459d597facf60b8d4212695b46142758c55b061343d4e9251b3760ad9cd317a4`。全部 12 条仍为待审研究样本，不能按文本直接归档为带真值训练数据。

## 下一阶段判定

等待基础六条及语速十二条的逐条人工收据；疑似“小屋/小吴”、近邻首字错读、短音节吞并必须按录音 ID 处理。若某个控制组合反复将“窝”读作“屋/吴”，停止扩该组合；如果 C 检出与读音审查不一致，保留为定向算法难例，不能重写真值来迎合模型。通过声源审查后才考虑独立场景渲染；同一个基础声音的场景变体保持同一 split。若要分辨语速控制与随机 seed 的影响，应在下一轮对相同文本/性别使用同 seed 做成对 A/B，而不能使用本轮不同 seed 的时长差作因果结论。完整产品资格需要另行获得真人和最终 AFE/板端证据。
