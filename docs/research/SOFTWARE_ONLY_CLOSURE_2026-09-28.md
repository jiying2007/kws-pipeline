# 双唤醒词软件侧闭环执行检查点（reviewing）

## 目标、范围与验收

`goal_statement`：在不使用真人目标词、最终 AFE 或目标板的条件下，形成可复跑、可审查、失败时能明确给出原因的软件研究闭环。产出必须绑定音频/标签/许可/ASR 收据、训练 checkpoint 与 C 模型、关键词包、运行器、分组参考清单和事件结果。不得把合成或代理 AFE 的结果写成产品资格。

软件候选的预声明门槛：已审 Qwen3 未训练 Serena/Eric 两词 4/4、同组近邻 0 次；HI-MIA-CW 未训练 15 人 7,006 条事件不超过旧模型的 51 次；**独立生成器** ASR 精确接受正例两词均有覆盖且各词不低于旧固定模型的正确事件数，明确报告分母；其缺首近邻 0 次；Qwen3 模拟场景 near_clean 正例不低于旧模型，mid_fan 近邻 0 次。新的声线/seed 不自动当作独立说话人。所有来源须记录 WAV SHA 和生成器权重/许可身份，生成器内部切分仅作开发回读。

277 条配方回读后补充安全约束：near_clean 场景的近邻也须 0 次事件；该模型已在 near_clean 出现 1 次，不会因补充条件改变其失败结论。此约束在 149 条来源重置模型导出和回读**之前**冻结；其余场景仍分层报告，不把远场派生标签当作已审听真值。

执行前补充冻结：新生成的 Serena/Eric 八条只作为同生成器、未入训声线封存回读；仅在固定 ASR 精确接受且连续正例内部低能量间隔小于 120 ms 的子集上，两个目标词各须检出唯一一条 ASR 合格正例、近邻须为 0。这相当于保持上一枚重加权开发模型在该固定子集上的结果；旧固定模型在这两条上均未检出。900 秒代理背景流固定 seed5801、15 名 HI-MIA-CW 未训练说话人各一条“你好亚”注入清单 SHA-256 `44d767674b83c6697e604fc5ec0eeb67d2aaef99c8e3eb2341122b99af0537a0`；新模型须与旧固定模型一样 0 次事件且完成每条至少一次注入。该流是合成连续压力检查，不是家庭 FAR；独立 Spark 只按 ASR 精确接受子集计召回/近邻事件，全部原始录音另列诊断。

现有重加权 259 条模型前两门通过，但 Spark 缺首近邻 2/6 次触发、Qwen3 mid_fan 缺首近邻 2/10 次触发，因而**未通过软件候选门槛**。这是本轮冻结基线，不针对这些已观察记录扫描阈值或权重。下一轮增加新生成音频并重建训练池；Spark 与 Qwen3 原有探针仍标记为已观察回归，另封新的来源/seed 作软件检查。

## 阶段与停止条件

| 阶段 | 交付与完成标准 | 失败处置 |
| --- | --- | --- |
| A 数据输入 | 固定生成器 revision/权重/许可、每条 WAV SHA/PCM/文本/类别、ASR 准入和连续性门槛；同源派生不跨 split | 错读或停顿音频隔离；不按生成文本入训 |
| B 训练消融 | 以当前重加权 259 条为控制，只增加冻结的合格新来源；冷启动同 seed/架构/epoch，记录训练 manifest 和 provenance | 一次成对实验无正负净收益则更新假设，不扫小集 |
| C C 端回读 | 同 runner/词包回读已审 Qwen3、说话人隔离真人近邻、独立生成器、场景及连续负例；逐词、逐场景报告 | 任一预声明门槛失败，明确 `software_candidate=false` |
| D 收口 | 源码和测试复验、制品 hash、可复跑命令、独立 review 与已知限制 | 缺证据维持 reviewing，不提升模型 |

`required_evidence`：A-D 各阶段有路径与 SHA；`retry_budget=1`（每种假设仅一次冻结配对）；`staleness_threshold=source-or-policy-change`；`claimant=本地实现者`；`verifier=新鲜独立复验`；`open_items=待推进`；`completion_claim=pending`；`logical_task_open=true`；`milestone_close=pending`；`stop_condition=replan`。Knowledge Provider 当前 route unresolved，本文是本仓 reviewing 草案，不声称已进入跨仓知识库。

## 第一轮：同家族数据扩量的负结果

Qwen3 训练声线新生成 24 条，生成 manifest SHA-256 `235dcba221eb7810a1179123afaf139924e949350d4718c22dd262c7fc1af421`。固定 Sherpa ASR 对其中 19 条文本精确接受，机器收据 SHA-256 `962b8db23d667da807fb0ab904c6b146ff5bbe725ce725d86eb17694235a68dd`；另有一条虽词汇转写正确但内部低能量间隔 370 ms，被连续词门槛隔离。最终训练增量 18 条（词1 三条、词2 五条、近邻十条），manifest SHA-256 `3f11178a7f3e1482589945c47e4bbe8e3674a72980668d3944d350d0ac28ae60`。与原 259 条合并为 277 条，1 epoch 烟测及 600 epoch 冷启动完成，语料 canonical SHA-256 `1a045f1d7b019b5c122b86310b94a4f1cb489c5f2679c09134a01ba2075e0c9f`；C KWM SHA-256 `9c4bf95c57ffe5cadfc8c7e0165d8fb9c39e5f969d4b3109fa5cd3e0a279223f`。训练模型仍为 `development_only`。

独立 Spark 新 seed 的 18 条 manifest SHA-256 `998e1ac90a84154233f7e8b9154892911500f7b54c70c93512ccbc71ad76f311`，固定 ASR 精确接受 8 条（词1 二条、词2 四条、缺首二条），机器收据 SHA-256 `b61513a31a98351561c6d9c2e44a5632be2358762f953bd80f633f0119b90310`。另一批未入训声线 Serena/Eric 新 seed 八条 manifest SHA-256 `133d891a736614066b50443550709c504cec0178bca2f831b0a32a835bf0c61c`，固定 ASR 接受四条（Serena 两词各一条、两人“你好你好”各一条），收据 SHA-256 `b87e37c60e955d6a924a45a8b4a0d59fc8110876681b32c911ac85612b4fd951`。两批的 ASR 拒绝样本不计正式分母，但保留全量 C 事件作诊断。

哈希验证的[机器回读报告](../../build/software-closure-20260928/qwen-extra-software-report.json) SHA-256 `d224815159d4697abc6b69945059ac90ce49b95639caa54ec34af9c9ef522d9d`，判定 `software_candidate=false`。已审 Serena/Eric 两词只有 1/4 正确且“你好你好”一条误触发；独立 Spark ASR 合格词1 为 2/2、词2 仅 1/4（旧模型 2/4）；HI-MIA-CW 15 人全组有 4 次事件；模拟 near_clean 正例仅 6/10，且近邻 1 次事件，mid_fan 近邻 0 次；新 Serena ASR 合格两词 2/2、近邻 0。900 秒代理连续流完整注入 15 名说话人近邻，各组 0 次事件，但该固定流不能抵消独立音频和近场的失败。**停止在现有 AISHELL3+Qwen3 训练池上继续增加同家族样本或扫权重。**来源重置实验见 [AISHELL3 来源重置](AISHELL_SOURCE_RESET_2026-09-28.md)。

## 后续分支与当前软件状态

| 固定分支 | 数据/算法改变 | 软件共同门槛 | 主要失败证据 |
| --- | --- | --- | --- |
| 149 条来源重置 | 移除 AISHELL3 128 条并按计数恢复正例质量比 | FAIL | 已审未训练双词 1/4；代理近场正确 3/10；900 秒流 1 次事件，详见[AISHELL3 来源重置](AISHELL_SOURCE_RESET_2026-09-28.md) |
| 291 条 vocab7 | 新增 32 条受限音节负例、`mi/ya` 输出类别及新 pack | FAIL | 已审未训练双词 1/4；HI-MIA-CW 留出 57 次；900 秒流 1 次，详见[负例音节实验](FOUR_TOKEN_VOCAB_LIMIT_2026-09-28.md) |
| 重复词首音节阻断 | 原 259 条模型的 C 解码器诊断策略 | FAIL | 旧场景删 2 次近邻事件；新 60 条代理场景误触发不减，反而漏掉 1 条真实词2；实验 C 代码已撤回，详见[负结果](LEADING_SUFFIX_VETO_NEGATIVE_2026-09-28.md) |
| Qwen3 训练声线代理场景增强 | 用户直接确认词1四条、隔离 Uncle_Fu 两条，另18条固定 ASR 准入；281条配方 | FAIL | 600 epoch C 回读已审未训练声线词2为0/2、原近场6/10且1次近邻，详见[场景增强配对](SCENE_TRAIN_AUGMENT_PAIR_2026-09-28.md) |
| 整词单 token CTC + reject | 171条、vocab4/KWK独立指纹、关闭多音节辅助损失 | FAIL | 已审未训练双词0/4、原近场2/10；Spark 合格词1 0/2，个别事件明显过早，详见[整词原型](WHOLE_EVENT_CTC_PILOT_2026-09-28.md) |
| 整词尾部时间目标 | 同171条，只启用VAD最后活动帧事件约束 | FAIL | 已审未训练双词仍0/4；固定900秒流seed5801/7903分别292/324次事件，详见[尾部配对](WHOLE_EVENT_TAIL_PAIR_2026-09-28.md) |
| 空转写背景监督 | 在尾部模型配方上新增100条四类代理背景，合计271条 | FAIL | 两流事件降为67/69次但未到0；已审未训练双词仍0/4且出现词1→词2串词，详见[背景配对](WHOLE_EVENT_BACKGROUND_PAIR_2026-09-28.md) |

当前 259 条重加权模型仍是最均衡的**development-only 研究基线**，但 Spark 已观察样本有缺首近邻事件、代理远场/播放召回薄弱；它没有通过共同软件门槛。尾部事件原型及100条空背景虽验证了一个流式纯噪声缺口，却继续在未训练声线双词0/4，因而也不晋级。`completion_claim=software-verification-fail-closed-no-passing-candidate`；`open_items=跨生成器独立的可靠双词正例覆盖、带可靠词结束边界的整词事件/拒绝训练目标与新鲜软件封存检验`；`logical_task_open=true`；`milestone_close=negative-evidence-and-data-gate-hardening`；`stop_condition=replan`。没有真人目标词、最终 AFE、目标板证据时，产品资格仍另列且未触发。

## 本地证据交接

已把新生成音频、ASR 收据、数据 split/哈希、三次冻结训练 checkpoint/KWM 与 C 检测/provenance、模拟场景、研究脚本和报告整理为[本地受控证据包](../../build/software-closure-evidence-20260928-v2.tar.gz)，约 10.8 MB，SHA-256 `b6057192fde4c31f13f02ff9096b25c342289057afa187010cc139cb01869cf7`，`gzip -t` 通过；大量可重建的后验缓存未入包。[源码与制品锁](../../build/software-closure-20260928/source-lock.json) SHA-256 `25b77f1b3fe16e9fc49d2076099e4909219825ef9edafdf8f62f9f38303a3a24`，绑定 75 个相关源码/配置文件和 24 个研究制品的哈希、外部 C/ASR 二进制身份与局限。训练容器未带 Git 元数据，checkpoint 中 `repository_sha=null`；工作树仍有未提交研究改动，故它是**本地可审查候选**，不是从干净提交重建的发布包。清单里的部分音频路径绑定原 `/tmp` 或 `/work` 挂载布局；恢复时须按包内 SHA 重映射，不能只复制路径字符串当成身份。原始第三方真人语音和 TTS 权重仍由各自受控来源引用，没有混入代码仓。Knowledge Provider route 未解析，未宣称跨仓归档成功。

用户对六条场景词1补充人耳判定后，已把四条接受、两条错读隔离的直接反馈收据、22 条混合准入、split 审计及 281 条场景增强负结果加入[本地证据包 v3](../../build/software-closure-evidence-20260928-v3.tar.gz)，约 11.7 MB，SHA-256 `ca2b9f03d24b0d31cd953d4a9f1639cc08fb666ce49002b7c73629e37fafceea`，`gzip -t` 通过。对应[源码与制品锁 v3](../../build/software-closure-20260928/source-lock-v3.json) SHA-256 `7d114fb322774edf3bc2d93dd13a5796980462d028f1d6da92db045bfdb84358`，75 个相关源码/配置文件与 39 个制品哈希逐项回读通过；记录 HEAD `c3f39c93526bc97951259b9c257bad126676a5a3` 和 dirty 工作树。**v3 与 v2 均保留为此前阶段快照**，均未发布远端。新增[场景增强配对](SCENE_TRAIN_AUGMENT_PAIR_2026-09-28.md)在已审未训练声线词2 0/2、原近场6/10且1次近邻，未通过软件门槛；不启动第二 seed。

随后新增[整词单 token CTC 原型](WHOLE_EVENT_CTC_PILOT_2026-09-28.md)的冻结171条训练、vocab4模型/pack、C检测与事件时序负结果，已整理为[本地证据包 v4](../../build/software-closure-evidence-20260928-v4.tar.gz)，约 12.6 MB，SHA-256 `ea44b4961ebe9d36a6fb8382a1efa184b77325d8cb5753520001d72e5ae9ed66`，`gzip -t` 通过；[源码与制品锁 v4](../../build/software-closure-20260928/source-lock-v4.json) SHA-256 `b1d1d80d4258f705953be8425ff7c0ab15099ee9f1dbf7713776ebbaf8d6f3fc`，77个源码/配置和57个研究制品哈希回读通过。**v4 是该阶段历史快照**；它早于尾帧/空背景实验与本次[框架、算法与测试审计](FRAMEWORK_ALGORITHM_TEST_AUDIT_2026-09-28.md)，不代表当前完整源码与制品交接。没有远端数据仓回读地址，也没有干净源码提交或产品资格。
