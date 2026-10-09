# 保存证据诊断与 scratch 优化（2026-10-09）

## 结论

1. D20 v2 pcm9600 的原始 raw-logit FAIL 保留。没有调整任何原门限、重跑旧 once、模型 forward 或音频处理。D90 仍为 NOT_RUN。
2. 新保存数据检查把 cache 数值差追到已保存的 projection：C 每行 before/after 与移位寄存器逐位相等；Torch 每 callback 末 cache 与其保存 projection 逐位相等。由此不能把差异归因为这些检查点的移位/索引错误。Torch 未保存真实逐行 cache，检查限于重构加末状态，不能夸大为全程独立观察。
3. 最终层失败主要承接不同 stage19 输入；并非末层自己的算术残差主导。仍不能唯一归因上游某个算子，不能宣布修复。
4. 两条 shipping K2 FA 的 source-map 已从原 artifact 重取并校验。它们属于 ece44b47…d402 shipping 模型，不能记到 D20/D90。
5. 真正可落地的内存优化已完成：保留原 baseline，提供小型派生 patch、严格物化/编译工具与纯 decoder 测试。实测 host arena 333704→300104 B，节省33600 B（10.07%）；不是 SSC305 实测。

## D20 保存 NPZ 新分解

工具：analyze_saved.py，仅读取 NPZ 与既有 f32 权重做局部代数，没有加载框架或执行模型。

- pcm9600 共19行，callback划分9+10。
- part0 首次按原 raw 诊断容差出现超差的 stage 是7，即第二FSMN projection；4元素。part0最终logits仍通过。
- part1 cache 超差13元素：第二FSMN层11、第三层2；第一/第四层0。每个值都可由本后端自己的已保存projection和移位规则解释。
- 全局row12 / callback1 row3的class0：Torch25.205947875976562，C25.206327438354492，差0.0003795623779296875。
- 该class0的已保存stage19输入线性中心差0.00038061520617205247；C自身末层残差-2.231205691316518e-7，Torch自身残差8.297076732333153e-7。恒等式完全闭合（浮点中心仍不是精确实数）。
- class1观测差-0.00012123584747314453，输入中心差-0.00012088740307358847。
- part1 probability最大差5.960464477539063e-8并不撤销raw失败。raw和probability承担不同验收义务。

完整所有行、21层、4层cache与6类最终层分解见[saved-diagnostics-bitexact-v2.json](https://github.com/jiying2007/kws-data/blob/4b10cf76ca42386e0e932de719e6424fa6f2ed57/research/2026-10-09-saved-numerical-diagnostics/saved-diagnostics-bitexact-v2.json)；输入文件和权重manifest/payload SHA均在其中。

下一步若要查唯一根因，应先设计新的独立数值实验合同，冻结输入、逐算子观测及验收标准，再获得执行授权；不能以此保存数据分析假称重新资格通过。

## Shipping nightly 两个 K2 FA

原run37860130019 / artifact11585689303，ZIP355683 bytes，SHA256：
77e3d3fda81bef000c72927bf196dbed1b680e431c4455ee122ecdbdef1a38d0。
模型ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402。

- seed1103：7191.345s / confidence0.603046；active为xiao3 xiao3 xiao3 wo1。检测在active开始1.345s后，前一注入结束31.71125s后。
- seed3301：2372.645s / confidence0.962665；active为xiao3 wo1 xiao3 ni3。检测在active开始0.645s后，前一注入结束1.0255625s后；前一注入为xiao3 hao3 xiao3 wo1，注入间真实空隙0.3805625s。
- Source-map只证明时间归属。包中没有对应逐帧logits、VAD或decoder状态，因此无法区分声学混淆、非top路径、残留前缀或其他原因。

最小纯合成机制测试使用当前公开shipping decoder，无模型：0.55阈值、1.5boost、0.94retention及现有参数合同均不变。将源token转换成强one-hot式logits（±8）只是人为测试，绝不代表原模型输出。

- 单独3334、3431、3234均0次触发。
- 3234 + 19个speech-inactive blank + 3431：0次K2。
- 同样gap但19个blank标为speech-active：1次K2。

这是“是否真的出现连续inactive边界”会改变旧前缀保留的可运行机制repro，不是原FA复现或已确定根因。现有合同12 inactive帧即清理（20ms hop为240ms）；0.38s日历空隙不能证明当时VAD连续inactive。下一次另立获得授权的诊断，应保存检测前后logits/VAD/prefix状态，不应先降低阈值或重跑旧nightly来猜。

## Scratch版本与验证

repo新增research/decoder_scratch_v1；原research/native_a20保持逐字节不变。

命令：python3 -B research/decoder_scratch_v1/build.py --output <新的外部输出目录>

- 验证完整原source manifest，以及source/patch/derived哈希。
- 原source SHA：C 6a89f56f9d4a15f774c5f4fe13e9e6b86f574797879c0eda91ed3f49444d4937；H 8fff95d8b4bbac1e37d88d4e093e4f81a07467e397ead258d54f4a9edae6d2e4。
- 先复制全部保留hyp，结束next生命周期，再启用union的compact；不读inactive成员。排序、beam、节点编号/别名、算术序、门限不变。
- 70,000 B事务副本保留；不是删掉错误回滚安全性。remap也保留。
- 完整runtime仅编译不执行。实际sizeof测量：state70000不变；workspace170400→136800；arena333704→300104；weights1565280不变；合计1898984→1865384。
- 正常、ASan（无leak检查）、UBSan各72,000纯合成行通过。
- 原/新差分1,612次调用、17,177随机行、207次触发通过；包含重复token、blank、ties、空输入、NaN/Inf、不归一输入、capacity、clock overflow、早触发尾部校验与错误原子性。
- Python -O/-OO明确拒绝，避免assert移除导致假PASS。
- workspace ABI改变，所有调用方必须和新header一起重编译，不能混用对象/尺寸。
- 未测SSC305、板侧栈峰值、延迟或能耗，也未测新声学正确率。独立review已复跑新build与负测通过。

## 下一阶段协议草案（未执行；需要另行批准新执行）

### A. D20数值差：先隔离算子，后考虑新资格版本

问题：同一已保存输入与同一cache，C/reference各算子的差异是否主要来自累加/舍入；当前frontend微小输入差与上游累加误差各贡献多少？本轮能排除已存C逐行cache移位错位及Torch已存chunk末错位；不能排除未观测的Torch行内状态、上游输入传播或多算子累积，也不能证明任一是唯一根因。

预备冻结：两个pcm9600 NPZ、D20 export manifest/payload、本轮saved-diagnostics-bitexact-v2.json、21层顺序、所有原raw/probability门限、编译器/后端版本、运算模式与舍入环境。D90隔离不执行；shipping模型隔离不混用。禁止训练/调参、改变原门限或覆盖v1/v2结果。

最小新实验（只有获准后）：独立新版本的逐算子“同输入/同cache”实验，最多19行×21算子×2后端＝798个单行算子求值，不把新输出接到下一层，不做整网/音频forward；每次均消费冻结的保存pre-input。各有限记忆算子须分别标明“C真实saved cache”和“reference重构cache”，不可把后者冒充实测。若实际后端不支持单算子执行、dtype/运算语义无法冻结、hash不吻合，立即停止，不fallback到整网forward。

输出：按算子、行、类的同输入残差和传播项；预注册输入差线性分解，不重新拟合容差。首个无法满足冻结数值合同的算子即保存完整局部证据并停止继续资格推进。若全部局部通过但端到端原raw仍失败，结论仍是数值传播/预算问题未闭合，不能宣称资格修复。只有另立独立v3资格运行且原门限实际通过，才可讨论新结果；旧FAIL永不改写。

资源预算：单进程、单线程、无GPU、网络禁用，最大798算子行；总墙钟120秒、RSS512MiB作为预置上限。超限为中止/未完成，不能缩小输入/放宽门限后继续。若依赖框架本身超过此RSS，应在运行前重新审阅预算，不能运行中悄改。

### B. Shipping FA：观测缺口优先，机制与因果分开

问题1（seed1103）：3334声学输出是否真的出现3434候选，或由非top/历史路径触发？问题2（seed3301）：前条3234的尾部是否被保留跨入3431，是否连续出现至少12帧speech-inactive？本轮仅证明时间归属，纯合成结果表明边界inactive标志可能改变前缀保留，不证明旧FA由它造成。

先读旧产物查找是否另有逐帧logits/VAD/state（只读，不重复旧once）；若没有，旧FA因果标记保持UNDETERMINED。不得靠重新初始化的短wav片段声称复现原连续RNN状态。

可申请的新诊断协议：新实验ID和独立输出目录，固定shipping模型/关键词包/参数合同，两个目标场景各最多60秒、总120秒音频/模型求值，仅一次采集，记录每帧logits、VAD/speech_active、trie前缀/blank分数、边界reset、事件及时间源映射；显式标注新流从reset启动，与旧8小时nightly不同。不得调门限、不进行训练/TTS/ASR、不运行旧nightly。若没有完整输入身份、合法音频来源或连续初态定义，执行前停止并请求补齐。

采集后只在保存logits上做预注册paired decoder replay：完全原策略；在真实连续inactive边界做显式reset的诊断对照；两者均保留0.55阈值和其余参数。只允许这两个对照，禁止从结果寻找更优门限。只有状态轨迹明确指出哪条前缀从何时存活并完成，才可给新实验的机制归因；仍不追认旧两FA根因。如果事件不重现，记录不重现并停止，禁止换seed/音量/场景重试至成功。

预算和停止：采集最多120秒输入、单次、单线程CPU；运行前预估墙钟，硬上限10分钟/RSS2GiB。依赖或模型身份不符、输出漏记任一关键字段、超时/超内存立即失败关闭。没有通过这一步，不允许shipping算法/阈值变更或发布资格结论。

### C. Scratch工程验证的下一步

本轮已经交付可编译可回归版本，不需声学运行。真正板端部署前，独立使用SSC305官方工具链编译相同派生hash，记录ELF ABI、实际对齐、静态/栈峰值与运行延迟；caller全部重编译，保留同一合成decoder回归。未获板端环境与执行授权前只保留计划，host300104 B不得改标为板端结果。

复现工具必须显式提供--output新的文件路径，并以排他创建方式拒绝覆盖既有留证。cache逐位比较同时核对dtype、shape与C-order原始字节（区分+0/-0）。shipping复核最新留证为shipping-diagnosis-pinned-v2.json。
