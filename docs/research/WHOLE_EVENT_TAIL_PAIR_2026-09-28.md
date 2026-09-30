# 单 token 整词事件的 VAD 尾部监督配对（reviewing）

## 假设和固定变量

[单 token CTC 原型](WHOLE_EVENT_CTC_PILOT_2026-09-28.md)使用171条、vocab4、seed2346、600 epoch，已审未训练双词0/4；训练声线词2的103帧和Serena词2的91帧后验均被 blank 主导，独立Spark一条词2约0.105秒过早触发。CTC只要求单 token 在整段音频某处出现，没有完整词结束位置的约束。本配对在**同一171条 WAV/标签、同一网络/pack/seed/epoch/优化器**上只启用一次尾部目标：

- 对目标长度恰为1且是配置中的词1/词2的样本，用已有PCM VAD得到最后活动帧；在该帧对正确事件 token 做交叉熵，在其之前至少6个活动帧以外抑制同 token 的提前概率。总训练损失加 `0.20 × event_tail_loss`。负例/reject不施加事件尾部正标签；CTC仍覆盖其全词目标。没有活动帧、时间掩码错位或非单 token关键词时拒绝启用。
- `tail_window_frames=6`、weight `0.20`、按最后活动帧及活动帧早区段的定义在读C结果前固定，不扫描。开发专用 policy/provenance `development-single-token-vad-tail-ce-v1`；原训练入口默认0，旧模型浮点结果不受默认路径影响。
- 其余保持171条 canonical语料 SHA-256 `3a67365758c66750e49b7d3f8a95f53cc791201579727d7fca3e15c614038b3e`、RNN/H64、seed2346、600 epoch、batch16、lr0.001、`--ctc-vad-align`、四个旧多音节辅助损失均0、wake权重 `2.2818627451`、KWK fingerprint `0xb4514d918317e578`。冷启动且只用最后epoch。

验收仍按[共同软件门槛](SOFTWARE_ONLY_CLOSURE_2026-09-28.md)和[干净词尾时序门槛](WHOLE_EVENT_CTC_PILOT_2026-09-28.md)：已审 Serena/Eric双词4/4、近邻0、四条事件相对最后活动帧处于[-0.10,+0.50]秒；独立Spark ASR合格词1≥2/2、词2≥2/4、近邻0；HI-MIA留出事件≤51；原代理near_clean≥8/10且近邻0、mid_fan近邻0；固定900秒流0次。若两词/时序不能共同改善，停止单 token 事件原型，转向可信音素/词级强制对齐与显式多任务事件头的设计，不能调此小集参数。VAD尾帧不是可靠音素终点，代理场景的噪声尾部尤其需要额外审听；即使通过也仍是development-only软件候选。

`goal_statement=whole-event-tail-single-factor-pair`；`required_evidence=默认关兼容测试、梯度/掩码测试、训练provenance、同C runner/pack分组事件及时序`；`retry_budget=1`；`staleness_threshold=source-or-label-change`；`completion_claim=pending`；`logical_task_open=true`；`milestone_close=pending`；`stop_condition=replan`。

## 配对结果与停止

新训练目标的定向梯度/掩码测试、1 epoch输入与导出烟测通过；同171条/seed/旧参数下，新源码默认关闭目标时的 `float_state_identity.sha256` 与修改前完全相同，均为 `1e5fb8070f4f3a37790efcc9044a6298105cbdd28e7a14c941ed682abc2d8e55`。开启权重0.20的600 epoch模型 checkpoint SHA-256 `33f67a3d81a06541357716698db4eada3097bab980172e075925895cb4fe2d7d`，C KWM SHA-256 `d9adc8461f122302abb3f9a02bc8d127602d4e56def262c231d489a95d6654e3`，导出 provenance 明确标 `development-single-token-vad-tail-ce-v1`。后段 event-tail损失发生波动，没有从已观察数据挑早期 epoch。

[哈希验证的 C 回读报告](../../build/software-closure-20260928/whole-event-v1/tail-software-report.json) SHA-256 `6368b0ed1e215be3eddc30a9f038b8a0fd24fe2bfa3f4e60d014827483d6bfb5`，`software_candidate=false`：已审 Serena/Eric 双词仍 0/4，独立 Spark 合格词1 0/2、词2 0/4，原代理 near_clean近邻与 mid_fan近邻均有事件；HI-MIA-CW留出15人7,006条0次只能说明过度拒绝的一个侧面。四条干净正例没有事件，[尾界时序报告](../../build/software-closure-20260928/whole-event-v1/tail-qwen-dev-timing.json) SHA-256 `7bbbed58163cc992b8c50165bc570de1001168760860a70317263e133d4c8851` 因而失败。

固定900秒流比短音频暴露更严重失败：seed5801 **292次**事件（词1 267、词2 25），其中280次在注入近邻语音及其后0.5秒窗口之外；新seed7903 **324次**。两组均完整覆盖15条近邻注入，均只是在同一代理噪声族上的软件压力结果，不能外推家庭FAR。当前171条训练池均为非空短语转写，缺纯背景空白监督；这给出一项可隔离的数据缺口，下一轮只增加固定空目标背景，详见[背景监督配对](WHOLE_EVENT_BACKGROUND_PAIR_2026-09-28.md)。本时间目标自身未通过软件门槛，不能提升。`completion_claim=event-tail-negative-continuous-noise-false-accepts`，`logical_task_open=false`，`milestone_close=tail-pair-negative`，`stop_condition=replan`。
