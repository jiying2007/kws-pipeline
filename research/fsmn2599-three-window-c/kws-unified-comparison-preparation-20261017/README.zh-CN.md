# 有限阈值状态机研究结果

本目录保留阈值核心、6 项纯合成标准库测试和匿名聚合结果。核心与原测试保持原始字节；核心头注释中的 synthetic-only 是最初准备阶段说明，随后已在固定短音频后验上完成有限重放。这里的 CI 仍仅运行人工后验，不下载、加载或读取任何模型、真实后验和语料。

## 固定实现

本目录必须与既有 kws-fsmn-three-window-adapter-20261017 目录同级。复用其原始 8 个公开 reference 文件，不复制第二套 beam。adapter-source-pins.json 固定其中 5 个运行依赖；测试另导入已保留的 test_posterior_events.py 和 test_three_window_events.py。

阈值在精确候选去重后、两词共享 500 ms refractory 前执行。低分拒绝仍保留去重记录，局部 beam 已按原规则 reset，但不启动全局 refractory。因此不能用 θ0 终态事件过滤代替实际阈值重放。score 等于阈值时通过。score 为四节点原始概率乘积的平方根，范围 [0,1]，不是校准概率。

其余规则保持原三窗实现：126 输出帧窗口、42 帧步进、最多三束；完整 2599 类先 global top3，保留原概率，不裁表重 softmax。仅两个 canonical 词，不新增近音别名。阈值状态跨 chunk，单条音频/session 开始时 reset。

## 合成 CI

从本目录运行 python test_threshold_adapter.py、python -O test_threshold_adapter.py、python -OO test_threshold_adapter.py。每模式 6 项测试：θ0 逐帧/事件/状态等价、低分先至高分后至的后过滤反例、chunk 无关性、等阈值与去重、同帧固定排序、EOF/reset/非法参数。只需 Python 标准库。原三窗 reference 的测试和来源许可继续保留。

## 已执行的有限研究

共同完整后验交集为 68 条短音频：训练 26、历史开发 17、观察性诊断 25。每模型固定网格 0,0.1,...,0.9；每条 θ0 与原保存事件/greedy/summary 精确核验成功后才运行其余 9 套状态。三模型共 204 个 θ0 核验通过；14763 次 full-2599 float64 softmax 行、147630 adapter 行、220080 beam 更新、4530 窗，0 新模型推理/训练。float64 math.exp/math.fsum 口径与 canonical float32 数值路径不能混称逐位等价。

只有 17 条历史开发集参与选点：8 条参考正例、9 条非 canonical 参考。固定规则要求后者触发 0/9，再最小化参考正例漏失，同分取较低阈值。训练、观察性诊断和 unknown 不参与选点。原基线没有历史 posterior SHA，以固定 NPZ 身份及完整 θ0 输出核验，本轮首次记录其 posterior SHA；另两模型同时核历史 posterior SHA。

三模型均选 θ=0.6，参考正例漏 3/8、非 canonical 参考触发 0/9。θ=0.5 时 weak20 漏 2/8，另两模型 3/8，但三者仍触发 1/9 非 canonical 参考。此有限收益应保留，不能误写成全无收益。

历史人工确认的两条 TTS 正例在各选中点保留数依次为 2/2、2/2、1/2。weak20 的退化正例仅在本网格 θ≤0.2 时保留，而这些开发点仍有 1/9 非 canonical 参考触发。因此本次固定网格没有消除该取舍，不推广为所有校准方法不可能改善。

保留原 seed0_B 研究对照，不晋升候选。匿名 PUBLIC-AGGREGATES.json 中 status 是生成时的 executed_aggregate_for_review；执行证据与最终实际回读已独立审查通过，但这只验证执行与汇总，不是模型验收。

## 必须保留的边界

这些是依现参考标签的短音频条件指标，不是真实 FRR/FAR、FA/hour 或连续流独立验收。历史开发已经被观察；标签仍为历史接受/有限人工确认 TTS/机器弱标签等原证据等级。unknown 只计触发，不进入正负真值分母。诊断集合不用于选阈值。无新增独立小时、无真实说话人泛化结论。

其余 796 条缺共同完整后验，不能零 forward 重阈值，因此不在三模型交集中。完整真实逐条数据、标签审计、音频、权重、NPZ、私有标识和存储引用不公开。匿名汇总内 per_keyword 的 negative_reference_clips 是 one-vs-rest 分母，包含另一个词的正例；系统级 9 条非 canonical 参考分母是不同口径，不能混用。
