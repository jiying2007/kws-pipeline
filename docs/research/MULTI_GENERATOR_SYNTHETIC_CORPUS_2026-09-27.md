# 多生成器合成语料实验契约（reviewing）

## 决策

可以用不同 TTS **提高训练覆盖和软件验证强度**。但同一模型换 speaker ID、语速或随机种子仍共享模型的发音与声码器偏差；训练、校准、测试若都由一个 TTS 生成，测试结果会高估迁移能力。上一轮同源 TTS 扩充后，在独立真人近邻集上的误触发从 157 增至 661 次，详见[真人声学与方案重规划](REAL_SPEECH_SCHEME_RESET_2026-09-27.md)。因此新实验先做生成器家族隔离，再逐步混合训练，而不是先定一个巨大的样本数。

| 候选生成器 | 可核实的来源 | 在本实验中的位置 |
| --- | --- | --- |
| 现有 AISHELL3 VITS | 当前仓库已固定资产、模型与许可身份；原生 8 kHz，经受控升采样到 16 kHz | 旧训练基线，继续保留以便配对归因 |
| [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | 模型卡标为 Apache 2.0、中文声线；StyleTTS2/ISTFTNet 架构 | 实际试听后“窝”发音风险较高，退出首选正例来源；仅留待逐条审听的鲁棒性探针 |
| [MeloTTS-Chinese](https://huggingface.co/myshell-ai/MeloTTS-Chinese) | 模型卡标为 MIT，官方示例可在 CPU 运行并设置 speed；中文声线数量有限 | 可作为另一份训练声学来源，不能靠速度变体冒充多说话人，也不能与同家族 VITS 视为完全独立架构 |
| [Qwen3-TTS 0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 模型卡标为 Apache 2.0；有五个原生中文预置声线，无需克隆参考音频 | 20 条已获逐条人工确认；混入 12 条的既有训练配方仍退化，先解决目标函数与场景鲁棒性 |
| [CosyVoice-300M-SFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) | 模型卡标为 Apache 2.0；本轮使用自带中文男/女预置声线，无需克隆音频 | 已离线 CPU 生成 6 条；“你好小窝”有 ASR 发音疑点，待逐条审听；先作为封存检验来源 |
| [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | 官方权重为 CC BY-NC-SA 4.0，当前只用于用户明确的非商业研究；无参考音频的性别/音高/速度控制 | 本机已生成 18 条基础与速度探针；用户反馈整体可用、质量仅次于 Qwen3，逐条收据仍待取得；研究结果与商业语料池隔离 |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 官方模型与代码标 Apache-2.0；不同于 Qwen3 的 tokenizer-free 扩散自回归家族，无参考音频文字声音设计 | 本机已生成 6 条；ASR 仅一条目标词全词精确，混合 C 模型对一条缺首近邻有事件；待实听后决定是否扩量 |
| [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | 模型卡标为 Apache 2.0；零样本声音生成需要有权使用的 prompt 音频及可用算力 | 候选的合成封存检验来源；没有授权 prompt 时不启动声音克隆，也不将网络示例音频当作已获权素材 |

表中混合了已运行来源与规划来源，以各行状态及实验记录为准。本机只有 GT 710，CosyVoice2 的可行运行方式与耗时需先做小批量 CPU/外部受控计算预检；不得把模型卡许可直接扩展为 prompt 音频或第三方语料的授权。[Matcha-icefall-zh-baker](https://huggingface.co/csukuangfj/matcha-icefall-zh-baker) 的模型卡明确基础数据仅限非商用，因此不进入产品候选路径。

2026-09-28 更新：Kokoro 的官方 int8 包已完成 30 条本地小批量生成与 C/ASR 预检，详情见[Kokoro 跨生成器小批量回读](KOKORO_CROSS_GENERATOR_PILOT_2026-09-28.md)；MeloTTS 官方 ONNX 包完成 15 条、三档语速预检，见[MeloTTS 小批量回读](MELO_CROSS_GENERATOR_PILOT_2026-09-28.md)。[Qwen3-TTS 的 CPU 小批量回读](QWEN3_TTS_PILOT_2026-09-28.md)也已完成。用户试听排序为 Qwen3 最好、Melo 尚可、Kokoro 仍有“窝/沃”问题；随后 Qwen3 固定 20 条得到逐条确认。[CosyVoice SFT 与场景预检](COSYVOICE_SFT_AND_SCENE_PROBE_2026-09-28.md)已离线生成 6 条和 80 条 Qwen3 场景派生样本，CosyVoice 仍待审听。CosyVoice2 尚未运行。

后续 Qwen3 20 条已得到用户“全部正确”的审听结论，并形成独立归档候选；但是[在原 128 条训练池中增加 12 条 Qwen3 的配对实验](REVIEWED_QWEN_MIX_TRAIN_NEGATIVE_2026-09-28.md)使 AISHELL3 calibration 召回和真人近邻抗性退化。生成音质改善不等于现有四 token CTC 训练目标适配；不能沿用相同配方直接扩量训练。

用户进一步反馈 CosyVoice SFT 第一条“你好小窝”的“窝”实听很像“温”，已隔离该条；文本前端与标点 A/B 结果见[CosyVoice SFT 与场景预检](COSYVOICE_SFT_AND_SCENE_PROBE_2026-09-28.md)。另按官方权重许可和无参考音频能力筛选了[VoxCPM2、FireRedTTS3、Qwen3 VoiceDesign 等下一批候选](HIGH_QUALITY_CHINESE_TTS_CANDIDATES_2026-09-28.md)。

用户进一步明确当前为非商业研究，已按权重许可导入[Spark-TTS 研究预检](SPARK_TTS_RESEARCH_PILOT_2026-09-28.md)共 18 条音频。它的来源级听感仅次于 Qwen3，但各录音仍需人工逐条裁决，`research-only-cc-by-nc-sa-4.0` 不可混入未来商业语料。其后研究候选保留 VoxCPM2、FireRedTTS3-Instruct 及 Qwen3 VoiceDesign；后一项仍属 Qwen 家族。

[VoxCPM2 六条独立家族预检](VOXCPM2_RESEARCH_PILOT_2026-09-28.md)已完成。该来源在本轮 ASR/旧与混合 C 端均暴露疑点，尚未获逐条人工收据，不因模型卡的通用质量指标而扩大目标词正例。

用户后续明确允许用[固定 ASR 批量准入标准](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)。自动收据与人审收据分开；ASR 精确一致可作为研究批次的保守入选条件，但在 Qwen3 已有人审的 20 条上错剔 6 条。VoxCPM2 用户已给来源级实听反馈，认为声音与 ASR 疑点相近，继续暂停该来源扩量；未据此伪造逐条人工收据。

45 条本地审听页 `/tmp/kws-kokoro-20260928/review.html` 与便携包 `/tmp/kws-multitts-review-20260928.zip` 已生成；便携包 SHA-256 `0adeaff01b6b2b599f1b83ff5f8bdb0db41a00f90b9a353d0cea43a03df72cf5`。两者是临时审听材料，不进入仓库或公开制品。人工结果需要另作逐条 WAV 哈希绑定的收据。

## 数据设计

1. **文本标签。** 两个完整词“你好小窝”“小窝小窝”；每词覆盖正常、句首/句中、连读、停顿、快慢语速；负例包含缺首、缺尾、同音/近音、重复、颠倒、普通话句子与长时背景。任何发音失真、字词吞并或把负例读成目标词的录音必须剔除，不能依赖文本自动标注为真值。
2. **正交变化。** 生成器模型与权重、声线、文本韵律、采样率/重采样器、RIR/房间、噪声源、SNR、距离、方向、播放干扰分别记录。参数取覆盖表，不能仅把随机数种子当作独立声学来源；同一干净语音的增强派生样本只留在同一 split。
3. **分层切分。** 先固定实际已审来源与 split；CosyVoice SFT 六条通过逐条审听后可作候选封存检查来源。若将原留出生成器加入训练，原开发集立即降级为已见来源，重新指定未参与训练的生成器和声音做独立评估。合成“qualification”只表示生成器封存检查，**不得冒用** `commercial/real-human-qualification.policy.json` 的真人资格。
4. **规模先按覆盖补洞。** 先每生成器、每目标词、每声线和每近邻类别做小批量质量审听与 C 事件回读；通过后再扩展。每一批记录可理解度、错误发音/静音/截断比例、PCM 重复率、每词 FRR 与不同来源负例误触发。没有质量审查的百万条合成音频不算高质量数据集。
5. **回归与停止条件。** 先用同一模型和 C runner 做跨生成器切片，再只改变训练来源做配对试验。候选必须同时改善两个词的完整词命中与独立负例；若只提升合成分数而真人 HI-MIA-CW 误触发不降，停止扩大该生成器。真人近邻集已经被查看，之后只能用于开发回归，不可宣布新鲜盲测。

## 当前工具变化与缺口

`tools/generate_speech_like_command_provider.py --emit-provider-identity` 会在每条新生成记录中保留 hash 绑定的 `provider_identity_sha256`。`tools/speech_like_corpus_plan.py materialize --require-cross-provider-holdout` 允许训练集合并多个各自固定的 TTS，但会拒绝缺少生成器身份、训练与评估复用同一生成器、评估组内部生成器漂移，以及校准/测试生成器不一致；合成封存集必须再独立。不加新选项时旧归档仍按旧契约复现。这是**数据隔离门禁**，不是已经生成了可训练的多 TTS 语料，也不能证明不同模型训练数据完全独立。

不同 provider 的 `train/manifest.jsonl` 可先用 `tools/speech_like_corpus_plan.py merge --input-manifest A --input-manifest B --output-manifest MERGED` 组合；工具校验每条来源 ID、WAV SHA-256 和 provider 身份，改写为可追溯的绝对音频路径，并拒绝覆盖已有输出。`MERGED` 再作为物化阶段的 `generated_root/train/manifest.jsonl`。provider 身份哈希可能因运行适配器变化而变化，合并门禁本身不能证明模型权重不同；准入审查必须另外比对模型文件 SHA-256 和生成架构。

2026-09-28 的发音预检又暴露输入文本与真实音频不一致。新的 `materialize --audio-review REVIEW.jsonl` 要求每条计划录音都有审听收据，按 `source_id`、WAV `file_sha256`、预期文本、正/负类别和关键词 ID 逐项绑定；只有 `verdict=accepted` 且有非空匿名 `reviewer_id` 时才允许物化。缺项、旧 WAV 哈希、错标签、不确定或拒绝均失败。该选项为新语料启用，旧归档默认输出不变；它验证收据完整性，实际是否听过音频仍需 owner 审查。收据只放在受控工作区，不提交姓名或原始语音。

真正生成前仍需：固定各模型及许可证据、适配生成器为 16 kHz PCM16、确认可运行算力、处理 CosyVoice2 prompt 权利、人工审听发音，并为多生成器训练建立能合并多个 provider manifest 且保留原始身份的受控入口。目标词真人与最终 AFE/板仍是产品资格的必需输入，不能由任何合成集替代。
