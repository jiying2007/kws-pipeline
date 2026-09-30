# 已审 Qwen3 语音混合训练的跨来源负结果（reviewing）

## 结论

按预声明方案把**用户明确审听通过的 12 条 Qwen3** 加入原 128 条 AISHELL3 训练池后，TinyStreamingRNN/H64、seed 2346、600 epoch、冷启动和 VAD 对齐 CTC 均保持不变。新模型在 Qwen3 训练声线的 6 条正例上命中 6/6，但对未参与训练的 Serena/Eric 声线正例只命中 2/4，新增 2 次“窝小窝”近邻误触发；固定旧模型在相同未见声线组为 3/4、0 次误触发。原 AISHELL3 calibration 从旧模型 14/16 降到新模型 9/16；全量真人 HI-MIA-CW 近邻负例由 157 增至 540 次。**这次数据扩充没有共同改善召回与误唤醒，模型明确不提升。**

原 128 条 AISHELL3 有用户“基本没问题、质量不及 Qwen3”的来源级反馈，但没有逐条审听收据；所以本实验是开发诊断，不能升级为已归档合格数据或产品候选。Qwen3 Serena/Eric、AISHELL3 calibration/test 和 HI-MIA-CW 在本轮之前均已被查看，也不具备新鲜盲测身份。HI-MIA-CW 是密集短语负例，其事件/小时只表示该来源的诊断密度，不能外推家庭连续 FAR。

## 冻结输入与身份

- Qwen3 用户审听包 SHA-256 `d7e6ecf2f0b0d093047b9d67c3253f4c551c22c5d0378eaa041e4794a285274c`，用户在 `/vsdata/leiwenjun/work/tmp/` 拷贝件与原包哈希一致后明确确认 20/20 音频正确；逐条 WAV 哈希绑定的匿名审听收据 SHA-256 `affa420dba5b9b84b000772d8d5d7fbd0788caf3b90fdcd679e97f519ff20613` 已通过 `validate_audio_review`。Qwen3 12/4/4 说话人切分 spec SHA-256 `94237f3187a07aa9648f9305deb888b8db12feca298384d5490a6686cc2c06c6`；训练/开发 A/开发 B manifest SHA-256 为 `86e068e365e693cbbe4d358fe9b2b668a7cb9be38960a2b0fa9c51b800f7526f`、`0fbc13ca92087e51f67617ff5e5cdd8ad6d07759ba8d04140dbdbb9dc868df71`、`fe0dbb7a94cde7aa9358a8ddc100310fa717481c4d7e02e2379685433d7fbd40`。
- 原基座 train TSV SHA-256 `23db9b52e8cbddd49baa6ec62eabb2ed7aa558cb8bb41100b0c84f72c722d039`，128 条。Qwen3 加入 12 条，训练集总 140 条；跨三组 speaker/source/PCM 检查 `clean=true`，与原基座也无 PCM 重复。审计 SHA-256 为 `36e18994debe8db8b446fabc96b100bc408e4c6e9287237e5b24ae69a80863`、`fd1531a14d71312de3277bf3952f64d889bcb26a29259b1749d5b91a27876f50`。
- 1 epoch 输入烟测通过；600 epoch checkpoint SHA-256 `b7f648ab67ad4454fa0a510c5a54acc2d2c07c462ba0785da9508b99ac093fd3`，C KWM SHA-256 `d8b45c4586908e832091a9fc70fe43d34022e526197d5036afdedb6f5b2c41f4`，训练语料 canonical SHA-256 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6`。KWM provenance 标记 `development_only=true`，checkpoint `repository_sha=null` 且无容器 digest；本地权重不可提升。训练参数：feature 32/H64、batch 16、lr 0.001、600 epoch、seed 2346、冷启动、VAD 对齐 CTC。
- C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`，关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`；旧固定模型 SHA-256 `09ca0150ffc450a8725f3d1fc37cfc036aaa8c3c30d9a46b8b343e2951f8db21`。以下全部保持这两个 C 组件和词包不变。

## C 端配对回读

| 已观察开发来源 | 旧固定模型 | Qwen3 混合训练模型 | 判断 |
| --- | ---: | ---: | --- |
| Qwen3 训练声线 Vivian/Uncle_Fu/Dylan，目标词 6 条 | 4/6 | 6/6 | 训练内拟合增加 |
| Qwen3 未训练声线 Serena/Eric，目标词 4 条、近邻 4 条 | 3/4、0 FA | 2/4、2 FA | 跨声线泛化退化；两次 FA 均为“窝小窝” |
| AISHELL3 calibration：16 次目标唤醒，64 条总录音 | 14/16、0 FA | 9/16、0 FA | 召回明显退化 |
| AISHELL3 test：16 次目标唤醒，64 条总录音 | 12/16、2 FA | 12/16、1 FA | 仅少一次 FA；FRR 仍 25% |
| HI-MIA-CW：16,343 条真人近邻、6.6448 小时 | 157 次（23.63/h） | 540 次（81.27/h） | 跨来源近邻抗性明显退化 |

HI-MIA-CW 新模型 540 次中关键词 1/2 分别 430/110 次；旧模型为 132/25 次。这不是阈值或单一关键词问题。原 128+48 同源 AISHELL3 扩充模型在这批真人近邻上为 661 次，说明加入少量 Qwen3 有所回落，但仍远差于旧固定模型，也没有恢复未见声音召回。AISHELL3 test 时长极短，1–2 次 FA 对应的事件/小时没有稳定产品含义。

## 机制判断与下一步

此次只把 12 条已审 Qwen3 加入原训练集，标签、模型架构、训练参数、C runner 和阈值不变。因此可判定**“在现有 CTC/四 token/解码器目标下直接添加这一小批 Qwen3”不是解法**；不能据此否定 Qwen3 声学来源本身，也不能断言唯一根因是模型容量。训练内 6/6 与跨声线 2/4、真人近邻 540 次的分离，优先提示声线过拟合、正负采样失衡及代理损失与 C 事件不一致，需要用一个主变量的配对实验分辨。

下一步应先把已审 Qwen3 扩成更多**独立声线/来源**的平衡正负开发池，并引入可合规使用的连续真人负例与真实双麦场景；再比较当前 CTC 与直接完整词事件/显式拒绝头的可部署方案。声线/语速/RIR/噪声的增强派生样本不得跨 split，声音的“年龄风格”不得算真人年龄覆盖。每个新候选仍须在同一 C runner 下同时改善两词召回与多来源误触发，之后才可谈全新资格和目标板。

当前状态：开发实验为负，保留旧固定模型，Qwen3 合格小批量只作为独立数据归档候选；模型落地和产品放行仍未完成。
