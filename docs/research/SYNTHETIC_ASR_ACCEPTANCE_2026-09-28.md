# 合成语音 ASR 准入标准（非商业研究，reviewing）

## 决策与边界

2026-09-28 用户明确允许后续以 ASR 作为合成数据的批量发音准入标准。本轮选用固定的 `sherpa-onnx` 中文 Zipformer CTC 离线识别器；对 16 kHz 单声道 PCM16 逐条转写，只有在 Unicode NFKC 规范化、删除空白和标点后与预期文本**完全一致**的音频标为 `accepted`。不做同音字自动等价，也不把 C 端唤醒事件当作发音真值。正例、近邻与负例均适用同一文本一致性判定；ASR 不一致的录音 `rejected`，不按原始文本入训。

该标准是保守的**自动入选**条件，不是对不一致音频“必然读错”的断言。用户已确认 Qwen3 20 条全部实听正确，但严格 ASR 只保留 14 条，其中正例保留 6/10、近邻保留 8/10；余下六条是 ASR 相对于本次人审的错剔。已有哈希绑定的人工审听收据仍可通过单独 `--audio-review` 路径使用，机器收据不能伪装成人审。新的模型/说话风格首先进行少量抽听，若 ASR 系统性失效须重审标准与识别器身份。

## 固定识别器身份

- `sherpa-onnx` v1.13.8 离线程序 SHA-256 `3cca8d3f4a7edc19717fe49aecceab632727ed6a11f230fb64037cdb284c184a`。
- 中文 Zipformer CTC int8 模型 SHA-256 `e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07`。
- tokens SHA-256 `6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd`。
- 本地可回读副本存于 `.gitignore` 的 `build/asr-standard-20260928/`，三项哈希与原下载件一致。原始二进制/权重不入代码仓；正式长期归档仍需独立受控数据存储。

`tools/speech_like_corpus_plan.py asr-review` 读取一个或多个合成 manifest、可选 formal intents，逐条验证 WAV 文件哈希/格式、来源 ID/标签、识别器身份和转写，再生成 `speech-like-asr-review-v1` JSONL。`materialize --asr-review` 会拒绝任何缺行、过期 WAV 哈希、标签差异、识别器身份漂移或非 `accepted` 的计划录音；人工 `--audio-review` 保持另一证据类型。两类收据可以在同一批物化中**按不同 WAV 不重叠地共同覆盖全部计划录音**，例如人审 Qwen3 加 ASR 入选 Spark；同一 WAV 重复申报、缺行或多出录音都会失败。对预期文本无标点的连续正例，物化阶段还读取原 WAV，以 10 ms RMS 帧、峰值 RMS 的 1% 为活动阈值，拒绝首尾活动区间内连续低能量达到 **120 ms** 的音频。该阈值由本轮已审 Qwen3 与已知停顿错误的 CosyVoice A/B 小样本作保守区分，属于研究阶段边界保护，不是普适音素对齐。预期文本显式含停顿标点的计划样本保留自己的标签，不应用连续正例门槛。正式 provider manifest 不带类别时应传 `--intents` 取计划真值；多个 provider group 的 manifest 要全部传入，以覆盖同一批 intents。

这份 ONNX 权重的输入 batch 维固定为 1。曾用官方离线 CLI 同时传入两条 WAV 试图复用模型加载，ONNX Runtime 明确报 `Got: 2 Expected: 1` 并退出；因此当前入口坚持每次单条识别，128 条运行耗时较长。后续若需要大批量高吞吐，应另行验证支持逐条复用 recognizer 的受控 API，不能直接把多 WAV 参数当作这个权重可用的批处理方式。

示例（路径按实际语料替换）：

```bash
rtk python3 tools/speech_like_corpus_plan.py asr-review \
  --manifest build/generated/train/manifest.jsonl \
  --manifest build/generated/generalization-search/manifest.jsonl \
  --manifest build/generated/generalization-freeze/manifest.jsonl \
  --intents build/plan/intents.jsonl \
  --asr-binary build/asr-standard-20260928/sherpa-onnx-v1.13.8-linux-x64-shared/bin/sherpa-onnx-offline \
  --asr-model build/asr-standard-20260928/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx \
  --asr-tokens build/asr-standard-20260928/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt \
  --output build/plan/asr-review.jsonl
```

工具不会自动丢弃拒绝样本后把原计划强行物化；先根据筛选结果建立新的、可追溯的计划或隔离批次，再运行 `materialize --asr-review`。合成校准/测试即使 ASR 全通过，也只是开发集证据；算法候选仍需独立 C 端误触发与真人/最终 AFE 验证。

## 本轮已回读来源

| 来源 | ASR 精确接受 | 正例接受 | 解释 |
| --- | --- | --- | --- |
| Qwen3：用户已审听正确 20 条 | 14/20 | 6/10 | 证明严格 ASR 会错剔好音频；人工证据优先用于已审样本 |
| Spark 基础六条 | 3/6 | 2/4 | 两条正例有“小屋/小吴”疑点 |
| Spark 语速十二条 | 6/12 | 6/8 | 低速正例 4/4，高速正例 2/4；不同 seed，不做语速因果归因 |
| VoxCPM2 六条 | 1/6 | 1/4 | 用户来源级试听认为与 ASR 疑点相近，停止扩量 |
| CosyVoice SFT 六条 | 3/6 | 2/4 | 用户已听出女性“你好小窝”像“温”，含逗号修正候选也不适合作连续正例 |

原 AISHELL3 128 条的固定 Zipformer 回读已完成：只精确接受 **14/128**，其中正例 **2/32**、缺词近邻 **8/48**、普通负例 **4/48**；“你好小窝”及其停顿版本 **0/16**。机器收据 SHA-256 `20830a1af80cfb991dce544f6206cf6a1464ca4e1198e52e3275cac912626666`，输入 manifest SHA-256 `a435ad6fb4f91d2b96e102e6b70b5d0d1217fbea4ecdccd53932f1ba12857c5f`。这些数是 ASR 与预期文本的精确匹配率，不是 AISHELL 音频真实合格率；原 128 条不能按 14 条自动接受子集构成平衡双词训练池。

为核验识别器的来源偏差，另用[Qwen3-ASR-0.6B 官方模型](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`、[官方代码](https://github.com/QwenLM/Qwen3-ASR) commit `7c6daf77a2421100f5fb066495372c00129d39ff` 在断网 CPU 上回读已有人审的 Qwen3 20 条及 AISHELL3 的 32 条正例。模型权重 SHA-256 `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`，配置 SHA-256 `76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c`；结果文件 SHA-256 `cabf8bf1929f47be3545af731a03d3a46276661c5bd7ae0966c3e1c10d4724c1`。它对 Qwen3 正例同样只精确转写 **6/10**；对 AISHELL3 正例提升至 **11/32**，但两套 ASR 对 AISHELL3 的“你好小窝”及停顿版都为 **0/16**。这比单一 ASR 的 2/32 更能支持“原训练池第一词发音/域适配存在系统性问题”的假设，但还不是逐条声学真值。

随后 Qwen3-ASR 又回读 AISHELL3 余下 96 条：缺词近邻精确匹配 **8/48**、普通负例 **3/48**，文件 SHA-256 `b908da7084c36450315c7cf872d3852f542fb4bb967cc999146dc63833c3162a`。整个 AISHELL3 128 条按该第二识别器只有 **22/128** 与预期文本精确匹配，且没有负例被它转写成两个完整唤醒词。这仍不能证明其余 106 条真实读错或所有负例安全；只说明按文本精确标准，原训练池绝大部分无法自动入选。

对同来源开发集的 32 条正例再做 Qwen3-ASR 回读，calibration/test 的第一词各为 **0/8、0/8**，第二词为 **5/8、7/8**，结果文件 SHA-256 `3547e70efdd086ad8d0080360fbb76c8598bba8f8c348951dd6230525c69ac89`。三套切分都呈第一词失配现象；旧模型在 AISHELL3 calibration/test 上较高的第一词 C 事件，不能独立证明它识别了人听认可的目标发音。该推论仍需与人工逐条真值对照，不能把 ASR 自身的领域偏差排除掉。

控制样本还暴露 ASR 文本标准的另一边界：用户已判定不适合作为连续目标词的 CosyVoice 逗号版“你好，小窝。”，两套 ASR 均会转写成词汇上正确的“你好小窝”；第二套识别器的 19 条控制回读文件 SHA-256 `2697d937032e1c591bb8d1ef4c12c528278f8fcb9e220ff3e4289a383e1e43e1`。ASR 能筛词汇内容，**不能单独验证连续性或词间停顿**。本地 10 ms 帧能量探针显示该逗号版按 1% 阈值有约 170 ms 内部低能量间隔，原版为 0 ms；已审 Qwen3 连续正例在该阈值下最大约 110 ms。因此物化阶段加入上述 120 ms 保守拒绝门槛；它可能错剔自然停顿的正确音频，不能取代大量新来源的边界抽听。

当前所有 ASR 结果是**软件研究准入**，不会自动替代已存在的人工逐条接受结论，也不会被称为产品资格。
