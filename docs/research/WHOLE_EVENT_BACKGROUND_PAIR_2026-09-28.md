# 整词时间目标的空转写背景监督配对（reviewing）

## 新失败类别与固定处理

[整词尾部监督配对](WHOLE_EVENT_TAIL_PAIR_2026-09-28.md)在相同171条数据和单 token CTC上仍使未训练双词0/4；固定900秒流产生292次事件，其中280次在注入近邻音频及其后0.5秒之外，词1占257次。训练171条全部是非空转写（24完整词、147近邻 reject），没有**空目标背景**；此前短音频负例0次不能说明流式纯噪声安全。此轮只补一个失败类别：100条固定 seed6601 的2秒单声道16k PCM16空转写音频，白/风扇/电机/媒体四种代理噪声各25条，振幅覆盖与现有流式 harness相同的600–8500范围，文件/PCM/生成参数各自绑定SHA。它们不作为“100个独立房间”或家庭背景时数。

训练以171条[整词原型语料](WHOLE_EVENT_CTC_PILOT_2026-09-28.md)为控制，只追加这100条 `tokens=[]`、`keyword_id=null`、`kind=background`；其余 vocab4/KWK、RNN/H64、seed2346、600 epoch、batch16、lr0.001、CTC VAD和 event-tail权重0.20/尾窗6帧完全相同，四个多音节辅助项仍0，冷启动。新数据源和回读流 seed5801/7903不重合。因空样本的基础权重为1、非空为2，两词 wake 权重由冻结样本质量比公式从原 `2.2818627451` 调为 `2.2818627451×(147×2+100)/(147×2)`，保持 exact wake 对其他样本的预设质量比，不看C结果调权重。1 epoch预检后只训练/导出第600 epoch。

第一门仍是已审 Serena/Eric双词4/4、近邻0及四条事件处于干净末活动帧前100ms至后500ms；第二门独立Spark ASR合格词1≥2/2、词2≥2/4、近邻0；第三门HI-MIA-CW留出15人7,006条事件≤旧模型51；第四门代理 near_clean≥8/10且近邻0、mid_fan近邻0；第五门固定900秒流**seed5801和新的seed7903均0次事件**，并覆盖各15条近邻注入。只压掉流式噪声但继续漏掉双词也不能晋级。若失败，停止在当前小合成池上继续叠加空背景数量，进入[带可靠词结束边界与显式多任务事件头](DIRECT_EVENT_REDESIGN_2026-09-28.md)的设计。

`goal_statement=whole-event-background-empty-target-pair`；`required_evidence=100条背景生成与空标签SHA、输入/默认关兼容、模型provenance、同C runner两seed流式及双词门`；`retry_budget=1`；`staleness_threshold=source-or-generator-change`；`completion_claim=pending`；`logical_task_open=true`；`milestone_close=pending`；`stop_condition=replan`。软件研究通过也不替代真人目标词、实际AFE与目标板证据。

## 输入检查点

已按固定 seed6601–6700 和四种噪声各25条生成100个2秒WAV；每条 16 kHz单声道PCM16、文件/PCM SHA唯一，空目标与生成参数进入[训练清单](../../build/software-closure-20260928/whole-event-v1/background/train.jsonl)，manifest SHA-256 `06ca4294cc0ab7df489d828719f1f8e408dc77971ecfd3e1a088fdb80273d798`，生成脚本 SHA-256 `2962ac0aa6b6313324ff4c9c7e21989e5289a4b369672a241efd1da53b4ed4fe`。与原171条合为271条；按冻结加权质量比两词 weight=`3.0580065359`。这组背景仍来自同一软件噪声族，只能检验该族/新随机种子上的空白学习，不能估计家庭连续声学误唤醒。

## 两条完整流与双词结果

271条输入烟测、600 epoch冷启动与C导出完成，训练 corpus canonical SHA-256 `50ef8ad1927941ad6fe52a8b23eef93e0f07fadc0cf032f24f58e7fcd04920c9`；checkpoint SHA-256 `dfb05d33931433303b9e4f19e089fd0e22368ad243a563d81f0edf927c62e272`，KWM SHA-256 `a1bcb3727fa0f9192d60bf7a1758b2980633a367b906506f025880a285e472d2`。模型仍为 `development_only`。[哈希验证的软件报告](../../build/software-closure-20260928/whole-event-v1/background-software-report.json) SHA-256 `39aa5f82cffe17d909921121d6dd32bb4e3b09944035d175f61835abf93bed9b`，`software_candidate=false`。

| 固定软件切片 | 171条尾部模型 | 加100条空背景后 | 判定 |
| --- | ---: | ---: | --- |
| 已审 Serena/Eric 双词 | 0/4 | 正确仍 **0/4**；两条词1被判词2 | 双词门失败 |
| 新 Serena/Eric ASR合格双词 | 0/2 | 0/2 | 无恢复 |
| 独立 Spark ASR合格词1、词2 | 0/2、0/4 | 0/2、0/4 | 无恢复 |
| HI-MIA-CW留出15人7,006条 | 0次事件 | 12次事件 | 低于旧固定模型51，但近邻抗性退化 |
| 原代理 near_clean 正例/近邻 | 正确6/10、近邻3次 | 正确6/10、近邻**4次** | 仍未达≥8/10且近邻0 |
| 900秒 seed5801 / seed7903 | 292 / 324次事件 | **67 / 69次**事件 | 纯噪声误触发显著减少，仍远高于0门槛 |

两条固定流都完成了15条近邻清单的全覆盖，模型/pack/runner身份一致；减少 292→67 与324→69是**同一代理噪声族**的配对改善，不是量产FAR。短片段上看到的12次真人近邻事件与连续流中的剩余事件分属不同统计口径，不能相互替代。最关键的是正例两词仍严重漏检/串词，且新旧代理近场都出现近邻误触发；追加空背景不能补齐可靠词结束监督和跨说话人正例。按预设停止条件，不再堆同族空噪声数量或扫阈值，单 token + VAD尾部原型不晋级。`completion_claim=background-pair-negative-noise-improves-but-joint-gate-fails`，`logical_task_open=false`，`milestone_close=empty-background-pair-negative`，`stop_condition=replan`。
