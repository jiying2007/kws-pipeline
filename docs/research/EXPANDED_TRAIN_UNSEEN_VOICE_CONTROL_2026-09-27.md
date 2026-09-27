# 扩展训练声音后的未见声音配对对照（reviewing）

本检查仅为软件开发研究。上一轮 [`FRESH_SPEAKER_DEVELOPMENT_READBACK_2026-09-27.md`](FRESH_SPEAKER_DEVELOPMENT_READBACK_2026-09-27.md) 的 A/B 两组已被查看，因此这次将它们并入训练后，不再把 A/B 当作独立评估。新生成的 C1/C2 在模型训练完成前未运行推理；qualification、真人最终 AFE 和目标板均未读取或使用。

## 冻结设计与身份

- 预声明 spec SHA-256：`da332127fc40fd80a415e4a9e5172fdd5fe1587da2a5570f371d545e389245ec`。原 train 128 条，加已观察 A/B 的 48 条，冷启动训练共 176 条。新增训练 manifest SHA-256 为 `a2ac6ec59bcc170ba4bcb1a3ae3b5a278351a18a2b0cd2ccef399c0ec21fb5a2`；原 train manifest SHA-256 为 `23db9b52e8cbddd49baa6ec62eabb2ed7aa558cb8bb41100b0c84f72c722d039`。
- 模型保持 TinyStreamingRNN、32 特征/H64、seed 2346、batch 16、学习率 0.001、600 epoch 和显式 VAD 对齐 CTC；无 warm-start。新 checkpoint 记录 176 条、两份 manifest、corpus SHA-256 `759d5fdaf0f0e65a9684082acf1d92dc2f3de00c467ddccd89096039a7a8d208`，标记 development-only。导出 KWM SHA-256 为 `09398cd44aa18a752280e8cc905013157805fe1c98aacef15a860e02a7406485`；固定旧模型 KWM SHA-256 为 `09ca0150ffc450a8725f3d1fc37cfc036aaa8c3c30d9a46b8b343e2951f8db21`。
- C1 speaker ID 32–35，references SHA-256 `9d94b194e15e6b7c9057c1ae15e5ba1a4794269d4f94caf7de7374437673eccf`；C2 speaker ID 36–39，references SHA-256 `02165537d6ae90a005dfc28b77557d5eb8f89fee36f858de8f8ac37ed455bb40`。每组 24 条、8 次双词预期唤醒、16 条近邻负例。48 条新 PCM 与旧 256 条开发录音及前次 A/B 的 48 条均无重复；speaker ID 32–39 也与先前的 0–31 分离。cohort summary SHA-256 为 `8863a9cec788f0a3b1d283f7a1543fdc8d10b3e1c818861dbe5b9e42fea4608d`。
- 评分使用相同 C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`、关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`、阈值 0.55 和词首前 0 ms/词尾后 500 ms。C1/C2 之间未改变模型、训练、阈值或词包。

## 一次性 C 事件结果

| 固定模型 | C1 命中 / FA | C2 命中 / FA | 近邻与漏唤醒 |
| --- | ---: | ---: | --- |
| 旧 VAD 对齐 seed 2346 | 8/8 / 0 | 8/8 / 1 | C2 的一次 FA 为关键词 2 缺首音节后缀近邻 |
| 扩展 176 条训练的 VAD 对齐 seed 2346 | 7/8 / 0 | 7/8 / 0 | C1 漏关键词 1 一条，C2 漏关键词 2 一条 |

扩大同源离线 TTS 训练覆盖后，观察到误触发减少一条，同时两个未见说话人组各新增一次漏唤醒，**没有形成召回与精度的共同改善**。训练收敛及更低的可微损失不能替代 C 端事件。C1/C2 合计负例约 28.6 秒；C1 零误触发的负例 FAR 95% 上界仍约 786/h，远高于产品门槛。两组各 7/8 的 FRR 均为 12.5%，也高于整体 5% 目标。

本地 Docker 训练 checkpoint 的 `repository_sha` 为 `null`，且模型具有 development-only 标记；它只用于本次配对诊断，不能作为可提升制品。所有原始 WAV、提供器资产、训练 checkpoint/KWM 和检测明细仅在本地 `/tmp`，本仓只保留哈希、统计和负结果。

## 决策

不修改正式训练默认值、不消费 formal qualification、不提升扩展训练模型。现有证据同时支持：VAD 对齐可显著缓解原目标的 C 端静默，但关键词 2 的缺首音节近邻与跨声音召回仍需一起解决；仅增加同源 TTS 样本或只放宽 C 路径都不是已验证的产品解法。后续应先冻结更广的声学来源与连续负例开发集，并获得真人最终 AFE 与目标板证据，再进行模型候选选择和 fresh qualification。
