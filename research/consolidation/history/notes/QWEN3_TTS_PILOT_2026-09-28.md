# Qwen3-TTS 中文预置声线小批量回读（reviewing）

## 结论

用户试听上一批后反馈 Kokoro 的“窝”听起来像“沃”，MeloTTS 好一些。为寻找发音更可靠、无需克隆参考音频的来源，已在本机 CPU 实际运行 [Qwen3-TTS 0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) 的三个中文预置声线 Vivian、Serena、Uncle_Fu，生成双目标词 6 条和缺首近邻 1 条。7 条均为 16 kHz、单声道 PCM16，文件 SHA-256 各异，总长 13.6 秒。CPU 首条加载约 2.36 秒、生成约 15.10 秒；后续六条生成各约 12–15 秒，说明小批量本地预检可行。

独立中文 ASR 把三条“小窝小窝”、Serena 的“你好小窝”和“窝小窝”转写为预期文本；Vivian/Uncle_Fu 的“你好小窝”末字转成罕见字。旧 VAD 对齐模型在文本标记的 6 条正例上触发 5 条，扩充同源 TTS 训练模型 0 条；两者在唯一近邻负例上均无触发。**ASR 与 C 事件均不能取代听辨**。本地 12 条 A/B 审听包已取得来源级用户反馈，逐条发音真值仍未确认；当前不训练或提升模型。旧模型/扩充模型结果与 [Kokoro](KOKORO_CROSS_GENERATOR_PILOT_2026-09-28.md) 的跨来源对照方向一致，但样本量很小，且全部合成，不是产品 FRR/FAR。

2026-09-28 用户试听 12 条 A/B 包后反馈“Qwen3 效果最好，Melo 还行，Kokoro 还是原来的问题”。据此 Qwen3 升为**首选合成正例来源**，Melo 为补充来源，Kokoro 只留难例/鲁棒性探针。这是来源级听感选择，用户尚未给出逐条录音的接受/拒绝 ID；不能把 7 条 Qwen3 全部标记为已审听合格，也不能把合成集称为真人资格。

## 模型、运行环境与数据身份

- 官方模型仓库精确 revision `85e237c12c027371202489a0ec509ded67b5e4b5`；模型卡标注 Apache 2.0、预置中文声线，无需外部参考音频。`model.safetensors` SHA-256 `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`，`speech_tokenizer/model.safetensors` SHA-256 `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`。全部 13 个仓库文件下载到已忽略的 `build/qwen-tts-20260928/model/`，不进入 Git。
- `qwen-tts` 0.1.1 wheel SHA-256 `11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d`，隔离 CPU 环境 `torch/torchaudio 2.11.0+cpu`、`transformers 4.57.3`、`accelerate 1.12.0`、`scipy 1.18.1`、`soundfile 0.13.1`；推理使用 `torch.float32`、CPU、eager attention、4 线程、逐条固定 seed。模型原生输出 24 kHz，以 `scipy.signal.resample_poly` 转为 16 kHz 并保存 PCM16。缺少 flash-attn 与 SoX 可执行文件时首批推理仍成功；此观察仅适用于 12Hz CustomVoice 路径。
- 容器内生成 manifest SHA-256 `e677256071ec4b71a27eb360e28836fe5f1498e3ca64ecf63fdb2033de283ef0`。宿主机首次直接读取失败，是 manifest 里的 `/work/...` 容器路径不可见；逐条核验原 WAV SHA-256 后生成只改路径的参考清单，SHA-256 `ad4345738d72382b0654dd54aa10548e4f33509f405d644f830e229f9fab7157`。因此首次失败不计作音频或算法故障。
- 独立 ASR 初筛摘要 SHA-256 `5688c1aa4319bfb3415482d146965c6dc63fd2b21b495a69ed40194402c24f9b`。C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`、关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`，输入 corpus SHA-256 `16e61c7f2e0a1dc4ae4d9ea2610407ffa00b8e2f243b84e052a0e23fb07c5ae0`。旧/扩充检测摘要 SHA-256 分别 `d531ba4b6a4c86bb71d0ff0e7fddedb7b288cc375963cd84ac7ae7ae1d5549bd`、`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- 12 条 Qwen/Melo/Kokoro 同词 A/B 试听页在 `/tmp/kws-hq-tts-review-20260928/review.html`，便携包 `/tmp/kws-hq-tts-review-20260928.zip` SHA-256 `6ed72c4466b605fdfb8047cbc8f99ecb5766c8826a63a512174a6bf7b761ce43`。音频与逐条检测保存在受控本地目录，不提交仓库。

## 第二批预声明开发探针

在读取新增声音的 ASR/C 结果前，冻结第二批为 5 个预置中文声线 Vivian、Serena、Uncle_Fu、Dylan、Eric，每声线各“你好小窝”“小窝小窝”“窝小窝”“你好你好”一条，共 20 条、10 正/10 负。首批已有的 7 条纳入但明确为已观察；新增 13 条使用 seed 1344–1356。模型 revision、CPU/float32、`do_sample=True`、`max_new_tokens=120`、16 kHz PCM16 规范化、C runner/词包均与首批相同。新增 Dylan/Eric 为不同中文口音探针，不冒充真人或独立生成器。只做一次配对 C 事件和 ASR 初筛，再由人工审听；不训练、不消费 formal qualification。

### 第二批结果与审听入口

13 条新增语音均生成成功；合计 20 条、10 正/10 负、34.96 秒，20 个 WAV 与解码 PCM 均无重复，文件格式均为 16 kHz 单声道 PCM16。manifest SHA-256 `828111a19189bce43abc0d2239faf7e70c2f1e910142ce5739dcb7769fd2fade`，C 输入 corpus SHA-256 `00b20c2553802b4668a8a16d84b4e9a85bd74d23981f56b8e14c7b34b36072fe`。

| 预声明固定模型 | 10 条输入文本正例的 C 触发 | 10 条输入文本负例的 C 触发 | 仅看 6 条 ASR 字面匹配正例 |
| --- | ---: | ---: | ---: |
| 旧 VAD 对齐模型 | 7 | 0 | 6/6 |
| 扩充同源 TTS 训练模型 | 3 | 1（Eric 的“你好你好”） | 1/6 |

旧/扩充逐条检测 SHA-256 分别 `68256caa7da672325d55aeae47dafad7b3760ae7d928470220602ae32af3f711`、`b1686fb4dfd2e59c3fb04a57c305d47ee6d6dfa2393593883b57eccbc8997859`。ASR 摘要 SHA-256 `28b7cc0b18f55b04ae0b6acf12a2f115200061aa7890a0ae2b926a3b13b7bd12`；它对 6 条正例、8 条负例给出字面匹配，Eric 的“小窝小窝”转成“小五小五”、“你好你好”转成“李浩你好”，另有若干“你好小窝”末字为同音/近音罕字。**ASR 字面匹配不是发音真值**，表中比值仅是保守的诊断切片。旧模型在该切片好于扩充模型，与全量真人近邻负例上的退化方向一起提示跨来源泛化失败，但尚未构成产品指标。

可播放的 20 条逐条审听页为 `/tmp/kws-qwen3-stage2-review-20260928/review.html`，便携包 `/tmp/kws-qwen3-stage2-review-20260928.zip` SHA-256 `d7e6ecf2f0b0d093047b9d67c3253f4c551c22c5d0378eaa041e4794a285274c`。用户将同一哈希的包复制到 `/vsdata/leiwenjun/work/tmp/` 后审听，明确确认 20 条全部正确。按每条 `source_id`、WAV SHA-256、文本、正负类别和关键词 ID 生成的匿名收据 SHA-256 `affa420dba5b9b84b000772d8d5d7fbd0788caf3b90fdcd679e97f519ff20613` 已通过 `validate_audio_review` 门禁。这是用户对该固定 20 条的确认，不推及该生成器未来批次。

已将它分为训练 12 条（Vivian、Uncle_Fu、Dylan）、开发回读 A 4 条（Serena）、B 4 条（Eric），spec SHA-256 `94237f3187a07aa9648f9305deb888b8db12feca298384d5490a6686cc2c06c6`。speaker/source/PCM 跨 split 无泄漏，与原 128 条训练池也无 PCM 重复；对应审计 SHA-256 分别 `36e18994debe8db8b446fabc96b100bc408e4c6e9287237e5b24ae69a80863`、`fd1531a14d71312de3277bf3952f64d889bcb26a29259b1749d5b91a27876f50`。128+12 条、seed 2346 的 1 epoch 输入烟测成功，checkpoint SHA-256 `46d07af11cac4d6228f3c4b64d0a22697c87d31637200bf2937aac15c383e896`；它只验证训练输入与监督路径，不能证明 600 epoch 模型质量。原 128 条 AISHELL3 的逐条人工审听收据仍未取得。

## 下一阶段门槛

1. 已完成首批 20 条的用户审听与哈希绑定；下一轮新增声线/seed/声学增强须另做逐条审听，不能复用本批收据。
2. 使用既定的 128+12 配对开发训练，已看过的 Serena/Eric 和 HI-MIA-CW 只能作探索性回读；后续要预先封存新的声线、来源和连续负例。不能用同一声线不同 seed 冒充真人或独立说话人。
3. 训练候选必须在同一 C runner 下同时改善两词命中与真人近邻、连续负例误触发；本次扩充模型对 Qwen3 0/6 的原因需要进一步分辨声学分布与目标函数失配，不能直接改阈值发布。
4. 最终真人目标词、AFE、目标板及 `commercial/real-human-qualification.policy.json` 的 Phase A/B/C 仍是产品放行门禁，合成结果不能替代。

当前状态：Qwen3 的首批 20 条已审听、可作为开发语料候选；128+12 的完整 600 epoch 配对训练结果为负，详见[已审 Qwen3 混合训练负结果](REVIEWED_QWEN_MIX_TRAIN_NEGATIVE_2026-09-28.md)。产品候选和发布均未形成。本仓 Knowledge Provider route 未解析，本文是 reviewing 草案。
