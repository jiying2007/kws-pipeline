# 整词单 token CTC + 显式 reject 的流式研究原型（reviewing）

## 假设与固定配方

四音节路径的 259 条基线在已审未训练声线可达 4/4，但在代理场景和部分跨生成器近邻出现误触发。增加干净 Qwen3、移除 AISHELL3、扩充负例音节及加入经用户/ASR 分证据筛选的场景音频均未同时改善双词召回与近邻抗性。因此用**现有 TinyStreamingRNN/C 引擎**做一次最小架构对照：输出类别改为 `<blk>, kw1, kw2, reject`，C 关键词包将“你好小窝”“小窝小窝”分别编译为单 token，reject 不进入关键词包。这个试验检查直接整词类别是否更适合当前少量受控数据；它没有显式词结束帧监督，不能冒称已完成[直接事件重设计](DIRECT_EVENT_REDESIGN_2026-09-28.md)。

数据固定为：已审 Qwen3 训练声线12条、新批次严格 ASR/连续性通过的同声线18条、用户/ASR 分证据通过的场景派生22条、HI-MIA-CW 训练说话人“你好你好”119条，预计171条。每条完整词的目标仅为 `[kw1]` 或 `[kw2]`；可信近邻仅为 `[reject]`。原 AISHELL3 128条发音/标签疑点较大，退出本原型。按源音频/PCM/说话人 split 审计，Serena/Eric 两组与 HI-MIA-CW 15名留出说话人不入训。确切计数和 SHA 由物化脚本回读；若少一条或标签不一致，停止。

模型保持 16 kHz/32维 logmel/RNN H64、seed2346、batch16、lr0.001、600 epoch、冷启动；启用现有 CTC VAD 对齐，关闭针对多音节路径的 ordered/margin/prefix-completion/recurrent-release 辅助项。两词 wake 权重按受控数据计数恢复原 140 条正负质量比，公式 `nonwake_rows / wake_rows × 38/102`，不从 C 结果搜索。新 token/关键词 pack 同模型指纹绑定，单词 token 要求2帧 trailing blank；阈值仍0.55，不能把旧模型和新 pack交叉使用。训练只导出最后第600 epoch，不按已看开发集挑中途 checkpoint。

共同验收保持[软件侧检查点](SOFTWARE_ONLY_CLOSURE_2026-09-28.md)：已审未训练声线双词4/4且近邻0；独立 Spark 严格 ASR 子集词1至少2/2、词2至少2/4、近邻0；HI-MIA-CW 留出15人7,006条事件不超过旧固定模型51；原 near_clean代理场景至少8/10且近邻0、mid_fan近邻0；固定900秒近邻注入流0次。另报告发事件相对短录音末尾的时间，不用早于词结束的 C 事件充当正确唤醒。新 Serena/Eric ASR 合格子集与60条新代理扰动也分层列出。单 seed全通过后再以seed1337复验；否则停止该单 token 原型，不改阈值求绿。

事件时序的冻结软件诊断：仅对已审干净 Qwen3 的未训练声线四条正例，以 10 ms RMS 帧、每条峰值 RMS 的1%定位最后活动帧；C 事件须落在该帧结束前 **100 ms** 至结束后 **500 ms** 之间。该能量边界是粗代理，不能证明最后一个音素已经完成；若事件明显更早则足以判为过早，本原型不晋级。场景噪声与播放会改变能量尾界，另列其事件时序，不套用这个干净录音门槛。

`goal_statement=whole-event-ctc-development-pilot`；`required_evidence=171条语料身份、vocab4/KWK指纹、固定训练provenance、C runner逐词/负例/时序/连续流`；`retry_budget=1`；`staleness_threshold=source-or-label-change`；`claimant=本地实现者`；`verifier=新鲜源码/声学复核`；`completion_claim=pending`；`logical_task_open=true`；`milestone_close=pending`；`stop_condition=replan`。真人目标词、最终AFE、目标板仍单列为后续产品资格，不进入本研究门槛。

## 输入检查点

按四份固定来源逐条校验 WAV SHA、16 kHz PCM16、来源唯一性及 Qwen3/HI-MIA 说话人组，物化 171 条（词1 10、词2 14、reject 147），manifest SHA-256 `8a69cd517189eeeadf5e211145260b22ba748d5cfc36ac51630f1c770e2453c1`。四份原清单 SHA-256 分别为已审 Qwen3 训练 `86e068e365e693cbbe4d358fe9b2b668a7cb9be38960a2b0fa9c51b800f7526f`、新 ASR 合格 Qwen3 `3f11178a7f3e1482589945c47e4bbe8e3674a72980668d3944d350d0ac28ae60`、人耳/ASR 场景混合 `c57702be7a5c5461b136e1ce94ba795c5e5ffcdfc665add323882e7538bc543f`、HI-MIA 训练近邻 `c07e20ed42cec67f9f8e38b4c037294679be0766b15e619ac1593284b78af5d1`。exact wake 24、其他 147，按预设公式两词权重 `2.2818627451`。研究 pack vocab4 fingerprint `0xb4514d918317e578`；单 token 不沿用四音节 pack。

## 冷启动回读与算法判定

1 epoch 输入烟测与 600 epoch 冷启动完成，训练语料 canonical SHA-256 `3a67365758c66750e49b7d3f8a95f53cc791201579727d7fca3e15c614038b3e`；checkpoint SHA-256 `0a7aa9f22f4c82e45f92c290460889eba3778bc75fc1290031bc35496c65c224`，vocab4 C KWM SHA-256 `bda2c128683541ac1e6b2170cf5d7b2ce0179b07d723f027707e88db5addea3d`，对应 KWK SHA-256 `494bae61a4b6aa78bcf59f161a44ef7e4cdc9adb49eb0cd4b21a37539e6d62e8`。最后训练 CTC loss `0.011412`；这不是唤醒质量指标。[机器回读报告](../../build/software-closure-20260928/whole-event-v1/software-report.json) SHA-256 `c22ec2d7df92b06a5699339c3e25756e77b6f145edfd293a1f8e7f0e5df9ac27`，逐组核验模型、pack、runner、参考和事件文件 SHA，`software_candidate=false`。

| 开发切片 | 本原型 C 结果 | 关键解释 |
| --- | ---: | --- |
| 已审 Serena/Eric 双词 | **0/4**，近邻0 | 直接丢失未训练声线召回 |
| 新 Serena/Eric ASR 合格词1/词2 | 1/1、0/1；另有一条 ASR 拒绝的缺首音频误触发词1 | 无跨声线双词泛化 |
| 独立 Spark ASR 合格词1/词2 | 0/2、2/4，合格近邻0 | 仅词2部分迁移，词1失败 |
| HI-MIA-CW 留出15人7,006条 | 0次事件 | 强拒绝伴随目标词过度拒绝 |
| 原代理 near_clean / 新代理 near_clean_v2 | 2/10 / 1/10 正确，近邻均0 | 场景正例大面积漏检 |
| 固定900秒流 | 0次、15条近邻全覆盖 | 单侧负例结果不能晋级 |

已审四条未训练声线按最后活动帧前100 ms至后500 ms的冻结时序窗口检查，[时序报告](../../build/software-closure-20260928/whole-event-v1/qwen-dev-timing.json) SHA-256 `6e50facf8e9324b00ea05942b757670577764f8ac3e9bbffff3f6f2bac8fd145`：四条都**没有 C 事件**，故时序门失败。C 后验进一步定位：训练声线“小窝小窝”103帧中空白始终为 top-1，未训练 Serena 的91帧亦全为空白；训练声线“你好小窝”才有3帧词1为 top-1。独立 Spark 一条词2虽触发却在约0.105 s发事件，另一条 ASR 拒绝的高速缺首近邻在约0.145 s误触发词1。单 token CTC 既未保住双词泛化，也没有可靠的完整词结束机制。

按预设停止条件，不运行第二 seed、不改阈值或追逐最低 loss，不提升此模型。需要真正的词结束时间监督和显式拒绝时序目标，再结合更广、可靠的双词正例；本次结果不能单独归因于模型大小、事件类别数或 C 阈值。`completion_claim=whole-event-ctc-negative-over-rejection-and-early-events`，`logical_task_open=false`，`milestone_close=single-token-ctc-negative`，`stop_condition=replan`。
