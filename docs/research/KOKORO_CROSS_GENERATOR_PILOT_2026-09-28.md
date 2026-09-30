# Kokoro 跨生成器小批量可行性回读（reviewing）

## 结论

已在本地实际运行第二种中文 TTS 生成链，取得 30 条 16 kHz PCM16 小样本、6 个声线、0.9/1.0/1.1 三档语速，全部 PCM 不重复。**这不是已合格训练集**：运行时对 30/30 条报 `Unknown token: ❓`；独立中文 ASR 对近邻“窝小窝”的 6 条均识别为“我想问”，部分目标词也丢尾音或识别成其他音节。当前将样本标为 `needs-human-audio-review`，不加入正式训练、校准、测试或 qualification。音频和逐条明细只在本地 `/tmp/kws-kokoro-20260928/`，其中 `review.html` 可逐条播放并对照预期文本、ASR 初筛和 C 事件。

2026-09-28 用户试听反馈：“窝”听起来很像“沃”，MeloTTS 相对好一些。该反馈未给出逐条录音 ID，不能把 30 条全部判为同一结果；但对目标唤醒词正例的韵母/发音风险已经明确，因此 Kokoro **退出首选正例生成来源**。它只保留为待审听的跨生成器鲁棒性探针，任何录音仍需单条审听收据才能用作有真值样本。

软件层发现值得继续验证：同一 C runner/关键词包下，增加同源 AISHELL3 TTS 训练的模型对新 Kokoro 声音并未稳定改善；在预先选定的四个新增声线、每人两词共 8 条上，旧 VAD 对齐模型触发 4 条，扩充模型触发 0 条。ASR 对其中若干条给出同音或准确文本，但全部仍需人工听辨，不能把 4/8、0/8 宣称为真人或产品 FRR。该结果与 [真人近邻来源回读](REAL_SPEECH_SCHEME_RESET_2026-09-27.md) 的跨来源退化方向一致，不能仅靠增加同源 TTS 样本解决。

## 来源和身份

- [Kokoro-82M-v1.1-zh 模型卡](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)标明中文声线及 Apache 2.0；本次使用 [sherpa-onnx 发布的 int8 中文/英文包](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html)，不是原始 fp32 模型。模型包 SHA-256 `a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6`；包内 `LICENSE` 为 Apache 2.0，归档 417 个成员均为安全相对路径/普通文件/目录。`model.int8.onnx` SHA-256 `bda15858163726a492d02a9a727bc263551b86ac77f90812c4b30ff41d380e26`，`voices.bin` SHA-256 `e64a5a581d8c2a350d848f51c3121657cd83aa07ed6109172177345874a7244c`。
- sherpa-onnx v1.13.8 Linux x64 shared 归档 SHA-256 `c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4`，与仓库既有固定值一致；TTS 可执行文件 SHA-256 `dde4fbd93181de356d9dc32cd734407be0da64c10ad5883a5a1a1328e9fbc879`。原生输出 24 kHz，经 `/usr/bin/ffmpeg` SHA-256 `eedcb5eb3f8eb8486ba2d1e6d2796e0037f6002414ba5ffddf36a5228d52ca3d` 转为 16 kHz、单声道、PCM16。
- ASR 仅用于发音初筛：[官方中文 Zipformer CTC int8](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html)归档 SHA-256 `f3ad1814fea34c407eab0cc3df6f6b625419ac9a60d8aebd8efe772a8e85ef67`，模型 SHA-256 `e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07`。该模型固定 batch=1；批量传 10 文件时因维度 10≠1 中止，改为逐条调用后全部完成。已知既有 AISHELL3 VITS 的“你好小窝”也被它转写成“你好小哦”，所以同音字差异不能自动当作 TTS 错读。
- 三批 manifest SHA-256：初始 10 条 `c79e8ab03a3c36e06d8cad8a7cd71d30d2357709e8b69d401de8872bc42e04c7`；语速 12 条 `389bd9f81cf55eb1cc10271352c4316a560c62b2a0cf666abbeac160395fefeb`；新增声线 8 条 `e295c3e07bb91d78c70f4c3cad019413e67ce7799754a8f3be91bab62cb6c9c1`。合计目标词 1/2 各 10 条、近邻/其他负例 10 条、总长 33.13 秒；声线 ID 为 3、10、25、58、60、75。

## 对齐回读

| 批次 | 目标词 / 负例 | 旧模型按文本标注的目标词触发 / 负例触发 | 扩充模型对应触发 | 发音初筛 |
| --- | ---: | ---: | ---: | --- |
| 初始声线 3、58；语速 1.0 | 4 / 6 | 4 / 1 | 3 / 2 | 两条“小窝小窝”ASR 准确；“窝小窝”两条为“我想问” |
| 声线 3、58；语速 0.9/1.1 | 8 / 4 | 6 / 0 | 5 / 0 | 女声“小窝小窝”两档准确；男声 0.9 漏尾、1.1 为“叫我想问” |
| 未见声线 10、25、60、75；语速 1.0 | 8 / 0 | 4 / 0 | 0 / 0 | 三条“小窝小窝”准确、一条同音字；“你好小窝”含同音、异音和截断转写 |

上述“触发”只是执行事件统计；预期文本尚未经听辨确认为语音真值。C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`，关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`，旧/扩充模型 SHA-256 分别 `09ca0150ffc450a8725f3d1fc37cfc036aaa8c3c30d9a46b8b343e2951f8db21`、`09398cd44aa18a752280e8cc905013157805fe1c98aacef15a860e02a7406485`。三批 C 输入 corpus SHA-256 依次 `9f90f6b9a8f5d300b1c949b04d6e42e9eef188c64ce99c05100ae19040df0e9c`、`8e33606c965ae148d8a302b3b373a0548de9351252ba2c5e3dc3456a33d7a494`、`37431a555a795ee357d0c4339e2edc8c1357ee070630abf5f010ba0b0e8d7c2f`。

## 决策

1. 保留 30 条及审听页作为本地开发证据；先人工标记每条“完整正确／错读／不确定”，并核对“窝/沃”及词尾近邻音节。未完成审听，**不扩大 Kokoro 批量、不把 ASR 文本自动升格为标签**。
2. 若审听确认已有可用声线，先只把合格样本放入开发训练候选；用 `materialize --audio-review` 将逐条审听收据绑定 WAV 哈希后，再按生成器家族、声线和干净源录音隔离训练/开发/新鲜评估。语速变体留在同一 split；`Unknown token` 警告原因需解释或作为显式质量风险记录。
3. 继续用已见真人近邻集作开发回归，并寻求新的真实目标词及最终 AFE 声音。合成语料即使补齐三类 TTS，也不能替代产品 policy 的真人 Phase A/板端门禁。

状态：研究预检完成，`needs-human-audio-review`；模型训练与产品资格未推进。知识 Provider 的当前仓库 route 未解析，本文只作为仓库 reviewing 草案，不声明已归档到跨仓知识库。
