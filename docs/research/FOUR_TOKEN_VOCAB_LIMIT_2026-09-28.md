# 四目标音节输出表与近邻拒绝能力（design review，待实验）

当前模型 `keywords/tokens.example.txt` 只有 `<blk>, ni3, hao3, xiao3, wo1` 五类输出。C 解码关键词虽按四个完整音节顺序匹配，但“你好亚”“你好吴/温”等负例的尾音没有显式竞争类别；模型只能把这些声学帧分配给四个目标音节或空白。已观察到 HI-MIA-CW 说话人留出组中的剩余误触发集中在“你好亚”，跨生成器疑似“吴/温”目标发音也使正负解释模糊。**这只是结构性失配假设**，不能由输出表本身推断所有误触发的直接原因。

若[移除 AISHELL3 来源的固定配对](AISHELL_SOURCE_RESET_2026-09-28.md)仍不能同时通过软件门槛，则停止同一五类输出表的扩量/权重迭代，进入下列一次性设计验证：

1. 固定新的真人近邻训练说话人，只从其官方 transcript 中选不同近邻短语，不碰 15 名留出说话人；每条先通过固定 ASR 精确匹配和 WAV 哈希验证。若 ASR 对某短语系统性错剔，只报告不可用，不以生成文本或 C 事件伪造真值。
2. 将经过准入的非目标音节作为显式 CTC 类别加入 token 表；关键词仍只由 `ni3 hao3 xiao3 wo1` 与 `xiao3 wo1 xiao3 wo1` 编译。新增 token、标签映射、模型输出维度、关键词包 vocab fingerprint 和 C runner 绑定为一个实验 tuple。与旧模型的 pack 不能交叉替换。
3. 对照先固定训练来源、说话人切分和总样本数，再比较五类与扩展输出表；正例权重由实际正负计数公式一次确定。验证要包括逐短语误触发、两个目标词召回、独立 Spark ASR 合格子集、已审 Qwen3 和模拟场景；不因 HI-MIA 单侧改善而晋级。
4. 需要足够且发音可信的双词正例来平衡更多近邻类别。若准入后正例覆盖不足或扩展输出表仍失效，进一步评审显式整词事件头/拒绝头与流式边界目标，避免继续把问题归入阈值。

状态：设计 review；未修改正式 token 表、C 关键词包、模型 ABI 或产品配置。只有完成上述来源重置回读后才启动，且软件研究通过也不替代真人目标词/最终 AFE/目标板资格。

## 2026-09-28 受控结构验证启动

来源重置已按固定报告失败：已审未训练声线正确 1/4、独立 Spark 合格正例 2/2 与 2/4、真人近邻 0/7,006 次、近场正确 3/10 且近邻 1 次、固定连续流 1 次。这是同时存在过度拒绝与近邻误触发的具体依据。现启动一次输出表扩展验证，**不修改正式 `keywords/tokens.example.txt`**。

- 官方 HI-MIA-CW 训练说话人 20 名中，按短语/说话人/录音固定 SHA 排序，11 种非“你好你好”近邻短语各选 10 名说话人、一人一条，共 110 条候选；15 名说话人留出组没有参与选样。候选 manifest SHA-256 `a76ebd67d74ea41887cef5ee2c40c1c2484b31c8fb6204432ad4fdf0e3416caf`。逐条固定 ASR 检查进行中，预设每种短语至少 3 条精确匹配，否则不训练该配方。
- 研究 token 表[phoneme-competitor-v1.tokens.txt](../../configs/research/phoneme-competitor-v1.tokens.txt)在原 `<blk>,ni3,hao3,xiao3,wo1` 后增加 `mi3,ya4,mi1`。关键词仍是原双词的四音节顺序；新 pack 的 vocab8 fingerprint 为 `0xbe4af8e0c6b811fa`。旧 vocab5 模型与新 pack 互换时 C runner 已按预期拒绝；两种 pack 不可混用。
- 若数据准入成立，处理模型使用已审 Qwen3 12 + 新 ASR 合格 Qwen3 18 + 原训练说话人 HI-MIA “你好你好”119 + 新 ASR 合格竞争短语 M 条，输出 vocab8；冷启动 RNN/H64、seed2346、600 epoch、VAD/辅助目标不变。两词 exact wake 权重按输入计数一次计算为 `(38/102)*(135+M)/14`，以保留旧 140 条池的相对正例质量比。每条新近邻按完整官方短语的固定拼音映射做监督，不把无法 ASR 接受的音频按文本入训。
- 诊断控制使用**相同 WAV**和训练参数，但把非目标音节投影到现有四个目标 token 子表，明确标为 `reduced-vocab-projection`。它不是音素真值，作用只在区分新增声学音节监督的效果；是否执行须先确认处理组有足够准入语料。任何结果仍须在同一 C runner 的已审 Qwen3、独立 Spark、说话人留出真人近邻、代理场景和连续流上共同回读。

第一份真人近邻候选的入选结果为**负**：固定 Sherpa ASR 收据 SHA-256 `8c2f72b7946fd0bfac4fa233d6cb52ee1f70897ed6fad76622d77cf6dad9fb46`，仅 27/110 精确接受，只覆盖 4/11 种短语；含“亚”的短语均为 0。为排除单识别器来源偏差，对同一 110 条使用既有固定 Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` 再回读，收据 SHA-256 `e8bf5fbfe68dc3611f22eccfcf314110ee38ae2649a47809913078ee34a63a81`，仅 29/110 接受，覆盖 5/11 种短语；含“亚”的短语仍为 0。两套识别器一致的错剔可能来自来源读音、方言或识别偏差，尚无逐条人工真值，**不能把官方短语直接变成目标音节监督**。预设每类至少 3 条的门槛失败，此真人数据结构实验停在 A 数据准入阶段，未训练 vocab8 模型。

重新检验可生成高质量的合成近邻，但必须逐条固定 ASR 文本匹配并核对连续性；先用 Qwen3 三个训练声线各生成“你好亚／你好小吴”一条作六条 pilot。若关键短语仍不能准入，保持结构假设未证，不继续训练；若准入，单独冻结新的合成近邻配方，不能把新来源与本轮失败的真人收据混成同一数据集。

## 负例音节而非汉字的受限准入分支

六条 Qwen3 pilot manifest SHA-256 `5ce27da0eba86e65316ebf8122e78b0467944a58777341239a2ae361a8a0cae3`；Sherpa 收据 SHA-256 `ad7865c28ab2e457e804e86f7424f6598aa5514784d4fa1fd45ff0df9139fae7`。三条输入“你好亚”全部转写“你好呀”，两条“你好小吴”精确接受，另一条转“小屋”。它与真人近邻 Qwen3-ASR 的“你好亚”10/10 转“你好呀”方向一致。不能把“亚/呀”判为严格文本相等，也不能由 ASR 证明它们音调相同；但可以在**非目标负例研究**中把观察到的 `ya` 语音作为单独竞争输出类，不用它来接受目标词正例。

固定的[负例音节映射策略](../../configs/research/phonetic-negative-equivalence-v1.json) SHA-256 `7cf6f0a7c53e2ecfbe0e48b67010dad9427ddfa99f5a07c706b3b9e2ea16af00`，只允许四种短语的精确 ASR 观察：官方“你好”→“你好”；“你好亚”→“你好呀”；“你好米”→“你好米”；“你好咪”→“你好咪”。非目标米/咪统一标为 `mi`，亚/呀统一标为 `ya`，目标 `ni3/hao3/xiao3/wo1` 保持原 ID。其余 78 条候选继续隔离，任何正例不得走这条映射。按 Qwen3-ASR 固定回读、WAV SHA、训练说话人切分和每类至少 6 条门槛物化 32 条（9/10/7/6），manifest SHA-256 `2c8d7874ee3c28b2cdba86eb989c0159e6bf64f2813bda61e8e6ae45ba365def`；接受语义明确为 `phonetic-negative-equivalence-v1`，不冒充严格 ASR 精确接受或人工审听。

新研究 vocab7 表[phonetic-competitor-v2.tokens.txt](../../configs/research/phonetic-competitor-v2.tokens.txt)保留目标四音节并增加 `mi`,`ya`，关键词 pack fingerprint `0x22f6e73debdb9c86`。一次冻结训练使用之前表现最均衡的 259 条配方 + 32 条新负例，合计 291 条；exact wake 38、其他 253，两词权重按旧 140 条正负质量比设为 `253/102=2.4803921569`。其他 RNN/H64、seed2346、600 epoch、CTC/活动帧目标与 C runner 不变；仅新的 vocab7 pack 与模型成对使用。按同一[软件门槛](SOFTWARE_ONLY_CLOSURE_2026-09-28.md)回读两词、独立 Spark、真人留出、代理场景和连续流。若正例召回下降或近邻未改善，停止该结构方向；此分支的人工音调判定与真实声学资格仍另列，不提升正式模型。

## 七类配对结果与停止

入选逻辑已移入[仓库内只限负例的物化工具](../../tools/codex_assets/phonetic_negative_materialize.py)，非仓库 cwd 的 help/dry-run 成功，按录音 ID、WAV SHA 和 token 序列与本轮训练清单 32/32 一致；定向测试覆盖正例绕道、过期 WAV、目标词别名与目标 token 全路径。训练 corpus canonical SHA-256 `bea76640aef96b8b49a8aaa28bac3042102d55caaf7a6885d79ff8cdefe3f4e9`，600 epoch checkpoint 导出 C KWM SHA-256 `e88ce1950842fe4a8a75a7fc936e4b1b14f3cbe6eddccf82819092e5e9b0880e`，新 pack SHA-256 `fe6b61e59a227bbe67163bba7d70f719a02464c245439222039eae388e5ef480`。C runner 对旧 vocab5 模型 + 新 pack 按预期拒绝，新模型 + 新 pack 正常加载。训练最后第 600 epoch loss 为 `0.093151`，存在后期波动；没有从已观察资料挑选其他 epoch。

[哈希验证的软件回读报告](../../build/software-closure-20260928/phonetic-software-report.json) SHA-256 `ec842dcc0a26606525316498a672ff47a6610e20fbb83f64c7329b416d6f4d26`，`software_candidate=false`：已审 Serena/Eric 双词正确 1/4、近邻0；新 Serena ASR 合格两词 0/2；独立 Spark 合格词1 2/2、词2 1/4、近邻0；HI-MIA-CW 15 人 7,006 条 **57 次**事件，超过旧固定模型 51 次及上一枚重加权模型 3 次；模拟 near_clean 正确 4/10、mid_fan 近邻 1 次、rear_playback 近邻 4 次；固定 900 秒连续流 1 次。六条新 Qwen3 近邻 pilot 的事件从旧重加权模型 1 次增至本模型 3 次，其中一条“你好亚”ASR 转“你好呀”也触发词1。

本配对同时改变近邻训练音频、其受限监督与输出类别，**不能单独证明扩展 vocab 的因果影响**；结果足以否定当前具体七类配方可作为软件候选。停止在这 32 条/已观察集上继续扫描类数、损失权重和阈值。进一步改善需要重新设计流式**整词事件/显式拒绝**目标，并增加可靠跨生成器的双词正例；已有零散短音频不足以凭训练精度保证泛化。`completion_claim=phonetic-v2-negative`，`logical_task_open=false`，`milestone_close=research-structure-negative`，`stop_condition=replan`。
