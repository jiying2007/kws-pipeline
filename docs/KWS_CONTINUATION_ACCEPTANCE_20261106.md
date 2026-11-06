# 2026-11-06：后续研究的确认与集成状态

**#759 与 #760 均已实际合并；#760 合并后的 main CI 已完成，13个job全部成功。** 本页补充当前源码采用状态，原实验报告、执行索引和已保存胶囊保留各自封存时的真实状态。

## 已确认的源码状态

| 项目 | 已观察到的状态与固定身份 |
| --- | --- |
| [PR #759](https://github.com/jiying2007/kws-pipeline/pull/759) | 用户已明确接受固定头 `d833bb8920125802c7cea7013526e5ea84843a3f` 的两项已披露来源缺口；实际 squash 合并提交为 [`fed60af6cee05312019468818db7f956ca984d69`](https://github.com/jiying2007/kws-pipeline/commit/fed60af6cee05312019468818db7f956ca984d69)。 |
| [PR #760](https://github.com/jiying2007/kws-pipeline/pull/760) | 在补齐 squash 后的合并关系后，集成头为 `f3201fcffa641b8dd735d0b98199856b32343b3b`；实际 squash 合并提交为 [`1b1cc18b212068f2eab4ee5ab432dbead3f42372`](https://github.com/jiying2007/kws-pipeline/commit/1b1cc18b212068f2eab4ee5ab432dbead3f42372)。 |
| #760 合并前 CI | [run 42189754573](https://github.com/jiying2007/kws-pipeline/actions/runs/42189754573) 在精确集成头 `f3201fcffa641b8dd735d0b98199856b32343b3b` 上成功，6项必需检查均成功。 |
| #760 合并后 CI | [run 42192043677](https://github.com/jiying2007/kws-pipeline/actions/runs/42192043677) 在 main 合并提交 `1b1cc18b212068f2eab4ee5ab432dbead3f42372` 上实际完成；2026-11-06 09:23 UTC的读回为 `completed`、`success`，13个job全部成功。 |

主线合并提交、集成头与原结果发布头 `a717e8644d8dc61132f7482277c9768a1eb80626` 的树均为 `10e9d6b80627d9618c71f8309707ae285b105de3`。合并关系修复保持了全部已公开源码与结果字节；原发布头的 [CI 42185602653](https://github.com/jiying2007/kws-pipeline/actions/runs/42185602653) 继续按该头的历史记录读取。

用户接受的两项缺口分别是：部分此前采用的后续变更缺少原控制者身份；一项补充保存账本审计缺少原CLI命令、stdout和returncode，其保存数据的算术已独立检查。此次明确接受满足该固定头的追加确认，**不会补造缺失的原来源或执行收据**。

确认记录于2026-11-06 08:39:26 UTC保存；#759成功合并响应于08:42:16 UTC保存。对应原始记录的SHA256分别为 `f1d940612923809c2be72d38e01e56d17676e2cf8b887a7c75d6a3f50af67e90`、`8d25c3b0e6753d0d09ee979a7c44441be71b93cc293ec035c86980c198b14370`。#760成功合并响应于09:02:42 UTC收到，原始记录SHA256为 `541cc6c5caa61d16a869283a760114e0e3d9e9d97c674c19b903a28339505914`。合并后的完整CI运行与13个job终态读回已另行保存；早先进行中的快照作为历史证据保留。

## 已完成的研究与操作入口

- [完整 context52/tail7 结果报告](KWS_CONTEXT52_TAIL7_20261106.md)与[机器可读执行索引](../research/context52-tail7-20261106/execution-index.json)：7个自然窗口的21次ASR观察、108次固定checkpoint配对开发前向，以及完整52条训练数据的2,000次更新、8,000次训练前向和210次末次读回。
- [配对 C1 context-development 指南](research-c1-context-development.md)：两份固定checkpoint、原18条开发集、0/200/500ms条件与完整float32保存约束。
- [完整 context52 known-silence 阶段指南](research-context52-known-silence.md)：原完整全文、weak权重、固定训练预算、逐次调用账本及一次最终train/development读回合同。

上述研究的计算、审计与实际保存已经完成。原报告中的“#759待确认”和原handoff中的“#760 draft”描述的是确认与合并之前的封存状态；它们与本页的后续源码采用记录按时间顺序读取。冻结执行源、原标签、失败和未知结果保持原身份。

## 本轮新增的保存输出诊断与来源准备

- [CTC264完整结果](KWS_SAVED_CTC264_20261106.md)与[公开聚合执行索引](../research/saved-ctc264-20261106/execution-index.json)：固定264个已保存条件、876个单token删除对照、1404个逻辑／1382个去重评分已完成；独立audit验证全量结果。没有新增模型构造、前向、优化器更新、ASR或TTS。两份终态胶囊已取得真实provider保存与canonical ack。
- K1的posthoc分析保留了两个关键词全部四个位置和重复mention。新模型development未命中组5条row／7个mention的“好”删除Δ均为负，条件支持大多由blank压过；这些是保存模型输出的描述，不能当作音频真值、生产decoder故障证明或新的泛化通过。
- 新8个参考来源的离线准备已实际成功，新增角色为6个`train-candidate`、2个`development-only`；旧32个来源及角色保持不变，新旧身份与原始／归一化WAV、PCM哈希均无重叠。其8份文本状态仍为`dataset-provided`，不等于人工核准或训练准入。ASR输入准备尚在进行，本页不记录该批识别或训练完成。

新来源的原128MiB下载尝试仍为失败；这次离线恢复仅使用已确认的81MiB前缀，完成select、normalize、registry和ingest，没有新增HTTP请求或模型调用。第82次未确认HTTP请求保留原状态。实际worker为3.18305139秒、外层3.274063835秒；独立终态复审PASS。终态SHA256为`fd351aa05a0aad6878eb967c80ccc04dde964d09d9cb7cc4f9748d9e705c6563`，实际外层rc0回执SHA256为`bd967f90e606a90c7a73fe8191ae5e81480165ef82af32cbedd34e88cf6b5b53`。

## 继续有效的结果边界

完整52条训练后的开发全文错误数从187/219降至96/219，但全文完全匹配仍为0/18；按每行至少一次完整连续序列计数，K1为1/6、K2为6/6。开发集已曝光，标签仍为机器弱标签；这些结果不构成独立测试、自然连续音频FAR或物理设备测量，也不是单变量数据规模消融。

自然窗口仍没有可纳入训练的已验证负例时长；历史D2的1,744个未知调用继续保留unknown且不重投。源码合并与导航接入没有新增模型调用，也没有建立常驻自动监督进程。后续保存logit分析或实验应继续保留独立的输入、设置与实际执行证据。

## 后续参考音转写的固定入口

新增 [eight-reference ASR 入口](research-reference8-asr.md) 复用原生 batch controller，在同一队列任务中执行最多 6＋18 次调用，维持首次 540 秒绝对截止时间。6 项无模型控制用例已在普通、`-O` 和 `-OO` 模式通过；实际调用结果另以独立执行回执报告。数据准备源码已通过 [kws-data PR #78](https://github.com/jiying2007/kws-data/pull/78) 合并至 `6486ef6017d4ac748fccb303df806f4de8450643`，新旧来源身份核验和原失败记录继续保留。
