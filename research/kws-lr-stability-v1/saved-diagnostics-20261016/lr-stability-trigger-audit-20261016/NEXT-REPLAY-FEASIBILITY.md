# 下一步产品触发逻辑回放：可行性，尚未执行

## 判断

可以另行定义一个“保存LP驱动、控制信号明确标注”的原C decoder状态机诊断；现有输入不足以忠实重放这次研究模型在产品中的真实事件。标准 kws_decoder_replay 命令当前不能直接使用：它要求模型文件、关键词包和trace中的模型SHA256/词表指纹/前端信息完全匹配，而新A600没有导出对应.kwm。不得借用历史模型身份、伪造trace模型哈希或关闭这些校验。

本文件仅做静态评估和最小方案设计，没有编译或执行decoder回放，没有新增模型、训练或音频实验。

## 已有、可派生及缺失的输入

| 项目 | 当前证据 | 能做什么与限制 |
|---|---|---|
| token映射 | A600每帧5类；固定目标KW1=[1,2,3,4]、KW2=[3,4,3,4]；当前公开tokens文件为0=<blk>、1=ni3、2=hao3、3=xiao3、4=wo1 | 索引语义一致。需要新诊断显式冻结此映射；不能据此伪造缺失的模型/词表指纹绑定 |
| LP | 12条、1276帧完整有限log-prob已绑定accepted manifest | 可直接作为decoder的归一化logit输入。精确实数下加减逐帧常数不改softmax/差值，但保存的float32 log-softmax与原始未保存logit有舍入差异；不承诺原logit回放的逐bit等价 |
| 时间轴 | 16kHz、400窗、320步、无补零；训练RNN每帧输出 | 可用end_sample=400+320*k。没有真实设备时间、计算延迟、丢帧、断流或跨片段连续状态 |
| 模型blank | 每帧LP与argmax完整 | blank-dominant来自已观察输出，可以原样使用。它不等于声学静音或实际VAD |
| 逐帧RMS | 冻结队列有同窗、同步长的完整RMS和记录哈希 | 可构造RMS派生的默认能量门代理。原C前端公式为10*log10f(float(mean_square)+1e-12f)，再与-55dBFS比较；rms平方与原float舍入边界不保证完全相同，且没有外部VAD/AFE元数据。它是派生代理，不能冒充记录下来的speech_active |
| prefix_frames | 首个RMS>=1e-4前的长度 | 是训练anchor定义。阈值约-80dBFS，不是产品默认-55dBFS；不能直接把prefix前设静音、其后全设speech并称为真实VAD |
| VAD控制 | 没有实际speech_active、外部VAD概率或运行时逐帧元数据 | 全1/全0、固定门序列、人工拼接边界都必须标成合成控制信号。不得把模型blank当VAD，或按结果挑选门序列 |
| 关键词配置 | 产品公开默认值及两条目标可读 | 可显式建立诊断配置：两个完整词，阈值.55、immediate、尾blank=0、priority=0、grace=0；这些不是新研究模型已经校准/实际运行过的关键词包 |
| 状态与抑制 | 原decoder源码包含命中重置、非speech边界重置等；原engine抑制逻辑可读 | 从每片段空状态开始是新诊断的控制假设。保留源码命中重置；引擎抑制独立记账，不可用无重置beam代替。抑制掉的decoder命中仍已重置decoder，且不会延长原抑制截止时间 |
| 事件真值 | 没有人工词起止、真实设备事件或连续负音频 | 只能报告条件下的decoder命中/输出/被抑制次数，不能计算真实词末延迟、误触发、漏检、重复唤醒率或FA/h |

## 最小可执行方案（需要先固定定义，当前未执行）

1. 新建与本次证据分离的目录，固定主线提交e80527ad4b1d4cf299ab986441f3c3e60be32747、decoder.c/.h、公共头、参数生成器/契约的源哈希，以及A600、队列、token映射和新运行定义的哈希。保持模型前向、训练、TTS、ASR、音频/权重读取预算均为0。
2. 建立独立的decoder-only C harness，直接链接上述原版decoder.c，逐帧调用kws_decoder_step；不调用kws_engine_process或RNN，不包装成官方匹配模型的KWTRACE，不修改现有回放工具的身份校验。公开源码已有结构性重复token、blank门、边界、前缀仲裁的合成测试可作为源契约校验；测试不依赖这12条样本选参。
3. 最小第一次只固定一种控制条件：speech_active每帧恒为1，明确标成“全speech合成门条件”，不把它当真实VAD或性能上界。保留原始1276帧及blank分布，不补blank、不复制、不拼接样本。RMS派生门可作为以后另行固定的扩展，但不在这次最小回放中试多种门然后择优。
4. 参数完全采用公开固定值：词KW1/KW2、threshold=.55、immediate、min_trailing_blanks=0、priority=0、grace=0、state_retention=.94；blank retention/root margin/fuzzy cost/弃路径预算/12帧非speech边界常量均来自该提交参数契约。这里的.55仅用于声明一个未校准的控制条件，不能从本次结果认定它适用于研究模型。
5. 每条独立片段开始时清空decoder及抑制截止时间。保留原C decoder命中后的自身重置；在独立适配层按原engine规则处理1200ms（19200采样点）的抑制：仅被接受的事件更新截止，区间内命中计为suppressed且不续期。必须单独验证这个适配层的首个事件、窗内命中、恰好到期事件及各片段清空规则；该层须通过独立测试后，才可称为引擎抑制逻辑的镜像，不声称执行了完整产品引擎。
6. 一次遍历全部12条；保留每帧原token、输入LP身份、控制标记、decoder命中、keyword_id/confidence、源片段end_sample、是否被抑制。仅总结“该固定合成门/未校准配置下的命中、输出和抑制次数”，并列出完整轨迹。即使某片段有多个输出，也称为条件下的多事件，不能直接判为真实重复误唤醒。

## 停止条件

- 源/输入哈希、token数或映射、记录顺序、LP有限性/归一化、帧长度不匹配时，立即停止，不补值、不重算模型。
- 不能得到所需固定源码/头文件或不能通过独立合成状态机测试时，停止在构建/契约问题，不解释样本效果。
- 若必须使用官方model-bound回放、原logit逐bit一致、真实VAD或真实词末延迟，立即报告缺少对应模型/原logit/控制记录/标注；不能使用历史权重或伪造身份满足接口。
- 不进入任何训练入口、不读取音频和权重、不生成新声学输出、不执行参数搜索、不新增真实负例；出现这些依赖时停止并重新确定范围。
- 固定一次12条1276帧回放完成并验证输出后即结束；命中不理想不触发改阈值、改门、补静音、重排片段或新训练。后续动作需要新的明确分析定义和相应授权。

## 静态来源

- [token映射](https://github.com/jiying2007/kws-pipeline/blob/e80527ad4b1d4cf299ab986441f3c3e60be32747/keywords/tokens.example.txt)
- [标准回放工具的模型/trace校验](https://github.com/jiying2007/kws-pipeline/blob/e80527ad4b1d4cf299ab986441f3c3e60be32747/tools/kws_decoder_replay.c#L101-L120)
- [trace记录所需字段](https://github.com/jiying2007/kws-pipeline/blob/e80527ad4b1d4cf299ab986441f3c3e60be32747/tools/kws_trace_io.h)
- [前端能量门数值来源](https://github.com/jiying2007/kws-pipeline/blob/e80527ad4b1d4cf299ab986441f3c3e60be32747/src/frontend.c#L147-L153)
- [engine命中与抑制](https://github.com/jiying2007/kws-pipeline/blob/e80527ad4b1d4cf299ab986441f3c3e60be32747/src/kws.c#L321-L344)

五份新增只读源码快照的精确Git blob与SHA256见feasibility-source-manifest.json；未修改先前固定分析的source-manifest.json。
