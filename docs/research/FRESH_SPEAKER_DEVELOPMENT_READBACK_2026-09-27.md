# 计划外说话人开发集一次性回读（reviewing）

本次只评估已冻结的本地开发模型，既不训练新模型，也不消费 formal qualification。目标是在此前多次观察的 train/calibration/test 之外，检验关键词 2 缺首音节近邻和双词召回是否具有跨说话人稳定性。所有 WAV、模型、生成脚本、原始检测和提供器二进制仅在本地 `/tmp`；本记录只保留脱敏统计、哈希及执行边界，不是产品资格。

## 输入冻结与隔离

离线提供器为仓库钉住的 `icefall-tts-aishell3-vits-low-2024-04-06`，8 kHz 源音频经固定 Lanczos 2× 归一到 16 kHz PCM16；运行时资产 archive SHA-256 为 `ab468db3a3308cdd861495e0db2f25d79418a0c00639f74944c7cdf5dd8c6ec1`，backend archive SHA-256 为 `c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4`。提供器准备阶段逐项核对 executable、模型、词表、lexicon、FST、adapter 和许可材料的既定哈希。输入 spec 在生成前冻结，SHA-256 为 `40f77449d2a193aeeafab4f9b8b3f7e7b80d42e69afac9205207220224133ae8`。

- development A：speaker ID 24–27；references SHA-256 `fac9bf7cb9d52b57db3232de63c1099cf177fa14d26e2911798674fac53e69c6`。
- development B：speaker ID 28–31；references SHA-256 `02edef3e839db5362e8d78933ab459ec77194e7157c8d33a51ce4ddca7c6ead3`。
- 两组各 24 条：双词正例各 4 条，关键词 1/2 的严格前缀与后缀近邻各 4 条。48 条 PCM 在新集内部以及既有 256 条冻结开发录音之间均无重复；ID 24–31 与 Stage A train/calibration/test/qualification 的 0–23 完全分离。
- 输出 cohort summary SHA-256 为 `5b6881201cb10ab41662c16151d5631ffe8d76a321b7beb98be31f1d69246515`。音频无最终 AFE、房间/麦克风变化与长期背景，不能代表真实用户或板端。

评分固定为同一 C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`、关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`、原 0.55 关键词阈值、词首前 0 ms/词尾后 500 ms。没有在两组之间修改训练、模型、阈值或文本。

## A 组选择与 B 组独立回读

预先列出的六个模型只在 A 组做一次回读。A 组读完、B 组仍未评分时，固定 VAD 对齐 seed 2346 和 VAD 对齐+非语音 blank seed 1337 两项，均为 7/8 命中、0 次误触发；原目标 seed 2346 作为固定对照。该选择记录 SHA-256 为 `dd2e463deee29c73a86b782df449b978d213940ebb05d10313d811f4a1840a17`。

| 600 epoch 冻结模型 | A 命中 / FA | B 命中 / FA | B 双词命中 |
| --- | ---: | ---: | ---: |
| 原目标 seed 2346（固定对照） | 4/8 / 0 | 8/8 / 3 | 关键词 1：4/4；关键词 2：4/4 |
| VAD 对齐 seed 2346 | 7/8 / 0 | 7/8 / 0 | 关键词 1：4/4；关键词 2：3/4 |
| VAD 对齐+非语音 blank seed 1337 | 7/8 / 0 | 5/8 / 0 | 关键词 1：3/4；关键词 2：2/4 |

A 组另外三个预列模型分别为：原目标 seed 1337 4/8、1 FA；VAD 对齐 seed 1337 6/8、2 FA；VAD 对齐+非语音 blank seed 2346 6/8、0 FA。B 组只对 A 组已固定的两项及对照做一次性回读。B 组对照的 3 次误触发均为关键词 2 缺首音节后缀近邻；VAD 对齐 seed 2346 在两组均为 7/8、0 FA，说明先前失效类别在计划外声音上仍可复现且训练对齐有软件层收益。附加非语音 blank 监督从 A 的 7/8 降到 B 的 5/8，未形成稳定召回改善。

## 判定

VAD 对齐 seed 2346 的本地 KWM SHA-256 为 `09ca0150ffc450a8725f3d1fc37cfc036aaa8c3c30d9a46b8b343e2951f8db21`，仍标记 `development_only`，产品提升门禁拒绝它。两组各 1/8 漏唤醒，FRR 均为 12.5%，已高于当前产品目标。A/B 合计仅约 27 秒负例音频；即使两组均零误触发，各组负例 FAR 的 95% 上界仍约 793/h，不可能证明产品级长时误触发率。所有候选都未达到产品放行条件。

这次新增证据支持继续研究 VAD 对齐目标，并反对把附加 blank 监督视为可靠默认值。下一轮若要继续软件模型选择，必须另行冻结未见声音、不同声学场景和更长负例的开发集；不得在本次 B 组上重新选阈值或损失权重。最终还必须完成真人最终 AFE、同一模型/关键词包身份的实体板与长时负样本资格。
