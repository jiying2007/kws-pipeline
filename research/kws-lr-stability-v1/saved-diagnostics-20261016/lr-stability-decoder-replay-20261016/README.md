# 保存LP驱动的原C decoder受控回放

## 结果

在固定的全speech合成门、未校准阈值.55下，12条已暴露训练片段各产生1次目标词原decoder命中：KW1为4条，KW2为8条。单独的1200ms抑制镜像接受12次、抑制0次。本次没有靠抑制隐藏额外decoder命中。

这是原C解码器对已保存研究LP的条件诊断。它不是完整产品引擎或原始模型回放，不能换算为真实召回、重复误唤醒率、词末延迟或FA/h。原训练600步终点的greedy KW1 4/4、KW2 0/8保持不变。

- 保存数据只经原C decoder遍历1次，共12条、1276帧。执行后只做结果读取与校验，未重跑保存数据。
- 所有片段都只有一个目标词命中；未出现另一完整词ID，也未出现同片段多个输出。短词“小窝”没有被加入关键词集，不能从本结果评价独立短词负例。
- KW1命中第10–29帧；KW2命中第12–96帧。名义源片段坐标分别为225–605ms、265–1945ms，绝不是相对人工词末的延迟。
- 原decoder置信度范围0.550574958–0.724726558。最低值距固定门槛仅约0.000575；未做阈值搜索或敏感性实验，也没有为该研究模型校准.55。

## 逐条事件

帧索引从0开始。每行均为原decoder命中，且被独立抑制镜像接受。置信度是原C解码器的路径声学token几何平均分数，不是标准CTC序列概率或prefix-beam概率。

| 样本ID末段 | 类别/ID | 命中帧 | end_sample | 名义坐标ms | 原C置信度 |
|---|---|---:|---:|---:|---:|
| kw1s-05 | KW1/1 | 10 | 3600 | 225 | 0.724726558 |
| kw1s-06 | KW1/1 | 11 | 3920 | 245 | 0.700420499 |
| kw1s-07 | KW1/1 | 10 | 3600 | 225 | 0.668741822 |
| tqp-qwen06-05 | KW1/1 | 29 | 9680 | 605 | 0.719014406 |
| tqp-qwen06-06 | KW2/2 | 23 | 7760 | 485 | 0.550574958 |
| tqp-qwen06-10 | KW2/2 | 64 | 20880 | 1305 | 0.566899061 |
| tqp-qwen06-11 | KW2/2 | 96 | 31120 | 1945 | 0.582096457 |
| tqp-qwen06-14 | KW2/2 | 33 | 10960 | 685 | 0.577875495 |
| tqp-qwen06-15 | KW2/2 | 37 | 12240 | 765 | 0.595073760 |
| tqp-qwen17-06 | KW2/2 | 18 | 6160 | 385 | 0.588204861 |
| tqp-qwen17-14 | KW2/2 | 12 | 4240 | 265 | 0.570233583 |
| tqp-qwen17-15 | KW2/2 | 13 | 4560 | 285 | 0.575049222 |

## 固定条件与实现边界

analysis-plan.json与DEFINITIONS.md在保存数据执行前固定，preexecution-freeze.json进一步绑定实现、二进制、生成头、编译器命令、合成测试记录和输入传输文件。固定源码提交为[e80527ad4b1d4cf299ab986441f3c3e60be32747](https://github.com/jiying2007/kws-pipeline/tree/e80527ad4b1d4cf299ab986441f3c3e60be32747)。

replay_harness.c直接链接未修改的src/decoder.c。两词按KW1、KW2顺序插入，ID1/2，token[1,2,3,4]/[3,4,3,4]；threshold=.55、immediate、尾blank=0、priority=0、grace=0，state_retention=.94，其余算法常量来自原参数契约。.55取自公开产品/示例配置，参数契约本身没有通用threshold默认值，本次不能称为新模型的已校准设置。

speech_active恒1是人工指定的合成控制信号。未把RMS、训练prefix或模型blank伪装成实际VAD；原decoder的blank衰减、根词限制、非top1子路径代价和弃路径预算仍然生效，所以恒1也不是性能上界。没有补blank、拼接、复制、裁切或重排样本。

每个片段从空decoder状态开始；命中后由原decoder自行重置。refractory_mirror.h是单独实现并经合成边界测试的引擎抑制逻辑镜像：首次或到期后的命中被接受，并设置end_sample+19200的截止；窗内命中不续期。decoder的原始命中即使被镜像抑制也已发生自身重置，不能让镜像改变decoder输入或运行过程。

没有链接产品engine/frontend/model实现，没有运行官方model-bound回放工具，没有.kwm、KWTRACE或替代模型身份。输入是原float32 log-prob，以十六进制文本无损传入C float；1276帧每帧5个float位模式全部核对一致。原始未归一化logit没有保存，不能声称与它们的逐bit回放等价。

## 校验与证据

- 9份公开源码/配置/token快照的SHA256和Git blob SHA1均匹配固定提交。
- 原版test_decoder.c合成测试通过；harness合成测试覆盖无命中、首次命中、窗内不续期、到期前1采样点、恰好到期、新抑制窗、每片段清空，以及被抑制命中仍已reset。
- 以不执行native保存数据的独立验证核对1276帧的顺序、LP位模式、argmax、控制标记、每条事件与抑制递推，样本/类别/总计全部一致。
- 20个预执行冻结文件、9份原输入哈希未变；claim、完成记录、原始输出、汇总报告的哈希链闭合，native stderr为空。
- 模型前向、optimizer更新、TTS、ASR、音频读取、权重读取和远端发布均为0。原训练预算没有重启。

主要文件：report.json是结构化结论；native-output.jsonl包含1276条逐帧记录及末尾完成记录；execution-claim.json/ execution-complete.json固定本次单遍执行；source-manifest.json保存公开来源身份。native输出包含输入LP的float位模式，用于核验，不包含音频或模型权重。

关键SHA256：

- report.json：485300ae598867e8353c4834e8925abee9b773f23fd0b397e5b1e2db9011c4bd
- native-output.jsonl：6f3a3a1aa96307a94d6f4457bdabfe58c806895838f93470c5605d2d93ff7e01
- preexecution-freeze.json：c2502e7d4f2c39b126d6918822bec184a69e8dbf3b65f822fadafbb75f7e1781

## 只读核查与复现说明

解压后，无需编译、原输入或模型即可检查交付证据的哈希、逐帧记录、LP归一化及抑制递推：

```sh
python -B -I verify_delivered_evidence.py
```

若已有本次原始证据根目录，可额外核对全部9份原输入及LP逐位一致：

```sh
python -B -I verify_delivered_evidence.py --base /path/to/kws-next-stage-20261015
```

本地原工作目录还可运行controlled_replay.py verify --base BASE，该命令只读核查，不调用native数据回放。包不含宿主二进制和重复的LP传输文本；交付验证器会明确列出未附带的构建文件，不声称验证其本地存在。源码、编译命令与合成测试记录足以重建诊断程序；跨环境二进制逐bit一致不在本次承诺范围内。

controlled_replay.py的preflight负责哈希检查、参数头生成、原C编译、合成测试和冻结；execute在唯一调用前独占建立claim，存在claim就拒绝再次运行。已完成的一次性记录不得删除或覆盖以重跑。本次没有安排第二次保存数据运行；任何重新执行必须另定范围并放入独立新目录。

## 停止状态

已经按预定条件完成并停止。当前结果说明：在这组固定控制条件下，原C decoder可以从保存LP中形成KW2完整词命中，而greedy整句仍输出短词。它没有解决真实VAD、人工词末、独立负例、设备连续状态或泛化验证的缺口，不授权改阈值、上线或追加训练。
