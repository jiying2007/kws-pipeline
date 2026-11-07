# 2026-11-07：64 个新自然窗与 73 窗覆盖回读

**64 个新自然语音窗口已完成三路原生识别并保存。** 加上旧 tail9，最终
73 窗的封存证据与准备输入均完成只读重放，父 PCM 区间并集为
**2,022.897 秒**。在执行前固定的全文 quorum 规则下，三路一致候选和
两路一致候选均为 **0**；73 窗全部保留为 uncertain，K1、K2 各有 73 个
masked unknown。本阶段增加了自然语音观察覆盖，没有形成新的 CTC 或逐关键词弱标签。

逐项机器摘要、来源版本与证据 SHA-256 见
[execution-summary.json](../research/natural-coverage-20261107/execution-summary.json)。
工具接口见[自然语音固定选窗与跨包覆盖](research-natural-coverage.md)。

所有来源维持 development 角色。accepted natural negative seconds 为 0；
人工金标、独立测试、自然流 FAR、词尾事件时间和物理 SSC305 测量均未由本阶段建立。
此前固定 3:1 候选的失败结论保留。本阶段没有追加 TTS、KWS 模型前向或优化器更新。

## 1. 实际覆盖

| 完成观察的集合 | 窗口数 | 输入秒数 | 父 PCM 并集秒数 | 重复覆盖秒数 |
| --- | ---: | ---: | ---: | ---: |
| 旧 tail9 | 9 | 267.897 | 262.897 | 5 |
| 本轮固定新窗 | 64 | 1,920 | 1,760 | 160 |
| 合并集合 | 73 | 2,187.897 | 2,022.897 | 165 |
| 满足全文 quorum 的机器弱候选 | 0 | 0 | 0 | 0 |

最终 `prepared` 与 `observed` 完全相同，`prepared_without_complete_observation`
为空。合并输入为 35,006,352 个 16 kHz 样本，父内并集为 32,366,352 个样本。
同一窗口的三个模型或两个关键词不会增加音频时长。区间并集也不建立独立
说话人数，不能作为已确认负例或自然 FAR 的分母。

### 固定选窗

8 段 AliMeeting Eval 父录音按会话字典序排列。每个长度为 `N` 的父 PCM，
对 `i = 1, 2, 3, 4` 取：

```text
anchor = floor((N - 880000) * i / 5)
A = [anchor,          anchor + 480000)
B = [anchor + 400000, anchor + 880000)
```

每窗 30 秒，组内重叠 5 秒、并集 55 秒。每父前两组和后两组各构成一个
四窗分片，共 16 个作业。选择仅依赖父长度和旧声明排除区间，未读取识别输出。
新窗与旧 tail9、不同新组之间没有交集；冲突时拒绝，不移窗或重新抽样。

固定 plan SHA-256 为
`3c98cd0b6e8ba55665e36b35d00eb12cb3e016dc91d7ed4cffb599068f8675d6`。
后续源码修复前后，`Parents`、`fixed_plan`、`overlaps`、`write_plan` 的 AST
与原执行版本一致，冻结声明没有重新生成。

## 2. 原生执行与跨批约束

每批 recipe 固定 4 窗、`qwen06` / `sensevoice` / `whisper` 三个模型，
形成 12 个原生 claim/native/result 单元。最终独立审计逐项连接了冻结 queue、
prepared index、recipe、source ID、原生身份、既有 seal/inventory 和实际保存回执。

| 执行项目 | 实际核对结果 |
| --- | ---: |
| 完整 generic 作业 / native 作业 | 16 / 16 |
| 新窗口 / 新原生识别调用 | 64 / 192 |
| 重用旧原生观察 / 重跑旧识别 | 27 / 0 |
| worker 执行记录 | 48 |
| 每个 worker 的 effective threads / interop threads | 4 / 1 |
| 重模型并发 | 1，共用固定 heavy lock |
| failed / timed out / unknown generic terminal / 自动重试 | 0 / 0 / 0 / 0 |
| 跨批 UTC 与同 boot 单调时钟检查 | 15 处相邻交接全部无重叠 |
| 相邻外层进程的实际间隔 | 24.053267–164.691268 秒 |
| 独立执行审计前后未变文件 | 1,552 |

17 份准备胶囊、16 份 intent、16 份 terminal 和最终汇总图共 50 个实际保存
ID 均唯一。准备对象的本地路径和归档 SHA 也逐项去重。最终 graph 的保存发生
在最后一份 terminal 保存完成之后；每一批的原始保存响应、版本、文件大小与
本地字节 pin 均已连接。服务给同名 intent 自动编号的实际返回名称被保留。

线程设置来自实际 worker 回执，锁证据包含共同路径、继承关系、监督状态和
进程组，并绑定原执行源码。历史 fd/inode 遥测及整机调度跟踪没有单独留存；
因此该结论限于记录中的协作锁协议和这些作业的实际时间区间。

## 3. 可用文本与弱标签支持

“原生结果完整”表示请求、结果、原生返回及封存文件可重放。
文本参与全文 quorum 还要求三路非空、decoder 结束且质量标志为空。

| 项目 | 旧 tail9 | 新 64 窗 | 合并 73 窗 |
| --- | ---: | ---: | ---: |
| 已验证原生观察 | 27 | 192 | 219 |
| 可用于冻结一致性规则的观察 | 26 | 190 | 216 |
| 不可用观察 | 1 | 2 | 3 |
| 三路全文一致，weight 1 | 0 | 0 | 0 |
| 两路全文一致且关键词向量一致，weight 0.5 | 0 | 0 | 0 |
| 三路 clean，但无全文 quorum 的窗口 | 8 | 62 | 70 |
| 缺少三路 clean 非空观察的窗口 | 1 | 2 | 3 |
| uncertain 窗口 | 9 | 64 | 73 |

3 个不可用观察均为 Whisper 的 `decoder_terminated=false`、
`quality_flags=["incomplete"]`，分别是 `tail2/clip-000`、
`natural64-005/clip-003`、`natural64-013/clip-001`。原调用已结束且原生证据
完整保留，这些文本没有进入弱标签准入，也没有自动重跑。

冻结规则如下：

| 三路观察条件 | CTC mask / weight | 逐关键词投影 |
| --- | --- | --- |
| 三路 clean，归一化全文相同 | true / 1.0 | 从该全文投影 K1、K2 |
| 三路 clean，恰两路全文相同，且三路对 K1、K2 的 literal presence 向量一致 | true / 0.5 | 从全文 quorum 投影，权重相同 |
| 其他情况 | false / 0 | targets 为 null，mask 为 false |

ordered nonliteral 路径仅用于诊断，不增加额外 veto。全文 quorum 是机器弱候选，
不证明声学转写完整性。没有全文 quorum 时，即使多个转写都没有关键词字符串，
也不产生负标签或空 CTC 目标。

| 关键词 | 机器弱 present | 机器弱 absent | masked unknown |
| --- | ---: | ---: | ---: |
| K1：你好小窝 | 0 | 0 | 73 |
| K2：小窝小窝 | 0 | 0 | 73 |

完整 evidence view 保留逐路 literal/ordered 诊断、decoder 与质量标志。
这两行统计是可准入的机器弱投影数量，不是实际声学出现次数。

## 4. 只读跨包回放与版本修复

最终回读覆盖 19 个准备包：3 个旧来源包由显式固定的历史准备 CLI 验证，
16 个新包由当前准备源码验证。完整父 WAV/PCM 会重新读取并重建窗口，
view 行再按 source、父 PCM、会话、family、split、clip ID 与样本区间连接。
只有实际完整的三路封存原生结果才获得 observed 覆盖。

2,576 个 view、原始输入、原生结果、父音频和源码文件的 SHA-256、修改时间与
链接数在最终回读前后相同；view、bundle、native 目录的文件名单也未变化。
before/after 清单的共同 SHA-256 为
`293b2f60a284599c44738410b88801dd3b2ef8da39e449bd05228d953c543078`。
该回读没有模型调用、训练、queue 写入或新 ASR seal。

真实旧 tail9 第一次回读曾因 `active_asr_batch.py` 的准备源码版本不同而拒绝。
旧 bindings 未改。兼容路径精确固定原 10 项 preparation sources、完整 14 文件
导入快照、Git commit 与每个真实 blob；额外模块、缓存、符号链接、任意 argv、
来源漂移和伪造 summary 均拒绝。仅在同一解释器的隔离进程中运行固定
`prepare_natural_asr_windows.py verify`，最长 60 秒。

后续核对还发现，普通 Git `cat-file` 在 partial/promisor clone 缺对象时可
自动下载。最终固定读取 argv 显式使用 `--no-lazy-fetch`，缺少本地对象直接拒绝。
新增本地 `file:// --filter=blob:none --no-checkout` 双克隆反例确认：旧命令实际
取回缺失 blob；新 verifier 拒绝，且对象文件名单和哈希不变、blob 仍缺失。
该反例只允许本地 file 传输。

## 5. 源码与实际验证

| 作用 | 已核验来源 commit |
| --- | --- |
| 执行前固定选窗 | `ce1f42e5e3a0fe5aef325a0599aba84df5e62b89` |
| 历史准备依赖 | `6f33a837896c38830e4de3a8a3be2e0c65669557` |
| 旧 tail9 首次成功 consumer 的原字节 | `f1b9c331bace29bf3524279e15adc3910335d8d7` |
| 原 23 项契约的跨版本 consumer | `94a7186275e591b4afa447ca68e0b5f1cb8afca6` |
| 最终 consumer，禁用 lazy fetch | `a99fb5e9e572f73aff444d0955795b373e09dfd5` |
| 自然 evidence view，kws-data | `f4b3931545a7e0a17f75cad56b9c74a171276f90` |

最终 consumer 文件 SHA-256 为
`247a873701f7e423d01f4614b102b1a5a2b9fbf74439600f875b1129140247d0`。
24 项契约在 CPython 3.12.16 的正常、`-O`、`-OO` 模式均实际通过，三次外层
测试共 7.614898941 秒。原 23 项版本、测试回执和旧集成失败记录继续保留。
中间 checks-003 缺少完整中间源码快照，因此仅作为已发生的测试时间和日志，
不作为最终源码验收依据。

Git 来源链的归档安排为保留 `archive/natural-coverage-source-20261107` 引用，
使原执行和 producer commits 在 squash 合并后仍保持可达。归档 ref 的实际
创建结果与 head SHA 由最终 Git 回执绑定。下方证据 pin 指向实际运行版本；
后续文档提交的 CI 结果单独按其对应 head 记录。

## 6. 保存、恢复与计时

4 份历史/最终 view 已从公开 Git blob 与 36 个既有胶囊恢复到新目录，
共 **1,865 个文件、405,306,517 字节**，每个恢复文件 `nlink=1`。
各 view 均使用其原 producer 独立回读通过；原 view 与既有胶囊保持不变。
恢复禁用 Git lazy fetch，没有新模型或网络获取。

新增恢复包为 689,769 字节，SHA-256：
`7db87506a42d1af3c93f3eb2628fd8f41d666fe2dc862e91f3f75f44c780d64c`。
实际保存 ID 为 `libfile_03ee0e970054819197c1e8a2e8845089`、版本 0。
它复用既有原生/音频对象和 Git 源码，不将缺少映射的 evidence 装入兜底载荷。
这份新恢复包的保存核对使用实际回执与本地恢复证据，未另行重下载该新包。

本轮 16 个 ASR 作业外层进程时间合计 **3,721.912091 秒**。
最终独立执行审计为 **2.321481072 秒**；最终 73 窗回读外层为
**18.320055018 秒**，其中 consumer 子进程为 15.921633705 秒。
子进程、native call 和 worker 时长均嵌套在相应外层进程中，不重复相加。

整个阶段上限为 480 runner 分钟，自然队列上限为 180 分钟。
全阶段预算由 root 汇总所有实际外层命令、单列的保存服务时间和实际 CI job；
本结果记录不填写尚未形成的最终全阶段总额。

### 主要证据 pin

| 证据 | SHA-256 |
| --- | --- |
| final73 view MANIFEST | `5dcf033a1c1897e8dfb387b5a3c520aeb8a4df3b30d9a7d24e55c6e16bdbaf03` |
| final73 data report | `d3a108c7cc8c0335cb5700ff2d617252cba6083a8df9c92fa4e04bf476a0d05d` |
| coverage request | `0aa50ae3320c66b370ce8cecbcb90cd34ddb45eff8f57fbcf788524bb117a12c` |
| coverage report | `8609d10420ee45404b5295a7e28da42dee3a10d8825e82356cece263f9e1a542` |
| 独立 16 批执行审计 | `f807abff62645d7c94e0c534dd405f617edc26337ce9872dcdd4e8d1adc02c63` |
| executor 最终保存图 | `7eb76a5401e47a0700052c91287c4402e2ecbdbd7fc7d709da034d73a9135160` |
| 源码恢复图 002 | `ffb12a068cada6f03e405dc48b0e9ef726845c8dddf1bab3013a67891151e605` |
| 四份 view 的实际恢复与回读证明 | `bf0a52b684e7b8a1f1bad2498a8ccd0124e26a2a57022d2b6d71f0cefef05ae0` |

完整文件大小、回执 pin、前后清单、逐父区间、逐关键词统计与不可用观察位置
均在 [execution-summary.json](../research/natural-coverage-20261107/execution-summary.json)
或其引用的原始证据中。

## 7. 研究判断

本轮完成了固定自然覆盖与原生证据链，冻结的一致性规则仍未产生可用全文
弱标签。70 个三路 clean 窗口的全文不一致，说明执行完成与全文支持是两个
独立问题；另外 3 个窗口保留了原始 decoder 不完整证据。

下一项优先工作是补足有来源绑定的词与关键词证据，并保留本轮已暴露的
development 角色。当前 73 窗继续使用 CTC weight 0、两项关键词 mask=false 和
`targets=[null, null]`；
不以观察到关键词字符串缺失来填充负标签，也不将这些覆盖秒数用于 FAR。
现有逐路差异可支持后续有限诊断，但新增目标正例、人工金标与独立自然负流
仍需各自真实来源和测量。本阶段没有因结果为零而放宽规则或追加识别搜索。

本记录由 root assistant 与内部 pipeline、data、runtime 代理协作完成；
源码与证据互审属于代理复核，未声称人工转写或人工验收。
