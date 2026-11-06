# 固定观察样本的声学与起始偏移视图

`training/research_acoustic_augment.py` 消费已封存、完整全文投影有效的观察行，固定父检查点 manifest SHA、每行来源和实际 WAV。只使用 ctc_mask=true 的原行，不提升 failed-native、uncertain 或 OOV 行。

实际 2026-11-06 轮次从 context v2 的 16 行中使用 14 行（10 train、4 development），原两条 failed-native 仍排除。产生 124 条衍生音频：160/480 ms 起始零填充 28 条；0.1/0.3 数字增益 28 条；两个同 split 的真实 RIR 卷积 28 条；train 内记录噪声的 20/10/5/0 dB 混合 40 条。合计 train 100、development 24。

84 条纯起始偏移、增益或 RIR 视图保留父本完整机器弱标签 CTC；40 条叠加背景因为背景没有可靠全文转录，CTC mask=false、weight=0、target_ids=null，不制造 blank 负例。后者只保留父本已知为正的关键词标签，原负关键词变 unknown；未启动任何关键词头训练。

这些是派生训练研究视图，不是 124 个独立来源或新增 TTS/ASR 调用。数字衰减不代表真人轻声；RIR 卷积不代表项目实测距离；接受的自然连续负例小时数仍为 0。不得将 development 素材或母本移动到 train。

每个派生行保存父全局身份、说话人/来源组/split、原始行与 WAV 哈希、配方、噪声或 RIR 身份、固定词表 SHA、派生 PCM/WAV 哈希。原音和 RIR 全尾保留。使用第 0 声道的决定显式记录；不伪造双麦或设备结果。

SNR 按 20 ms 帧 RMS 达到全句最大帧 RMS 的 5% 所得能量掩码计算，这是数值配方，不是人工 VAD。统一峰值增益同时作用于信号与噪声，不改变分量 SNR；报告 SNR 在 PCM 量化之前的数值。音量低不触发删除。

这些偏移样本回应 C1 固定八条 2000-step 试验中出现的句首先验迹象，不能据此声称问题已修复。此工具本身 0 模型调用、0 训练步；效果需要后续预声明对照验证。
