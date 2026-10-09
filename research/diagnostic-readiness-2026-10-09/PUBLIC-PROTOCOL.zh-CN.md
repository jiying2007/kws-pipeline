# 新诊断准入与保存证据补充（2026-10-09）

本轮仅核对公开字节、源码与保存结果。新增模型、单算子、decoder replay、TTS、ASR、训练调用均为0。D20原raw FAIL、D90 NOT_RUN不变；不作新的资格或shipping通过结论。

## D20：输入已齐，尚不具备执行条件

依据[原协议A/B](https://github.com/jiying2007/kws-pipeline/blob/1b3774a24f8712e043e8d82eea5eeb51911dfcf2/research/saved-diagnostics-2026-10-09/REPORT.zh-CN.md)。两个pcm9600 NPZ、D20 export manifest/payload已从[公开archive](https://github.com/jiying2007/kws-data/tree/7f421105eee4674f59c5437f315204a978721cfb/research/2026-10-08-d20-d90-host-research)取回，逐项SHA与既有保存诊断相符。实际counts为9+10；C有逐行cache_before/cache_after，Torch只有chunk末cache。完整身份见PUBLIC-READINESS-SUMMARY.json。

真正阻塞项：

1. 精确后端缺失：原Torch2.11.0+cpu、NumPy1.26.4；当前Torch未安装、NumPy2.3.5。不得以其他版本或NumPy计算冒充参考后端。已核对的Torch wheel SHA为f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338。
2. RSS512MiB硬限制未建立：当前/sys/fs/cgroup不可用，未验证等效机制，不能把RLIMIT_AS当RSS；也未测精确后端的只import峰值。
3. 单算子驱动尚未实现并独立验收：C仅公开全网a20_step，静态affine/memory/relu可由独立包装暴露；参考FSMNBlock.forward源码支持显式单行input/cache。必须冻结编译器、IEEE/RNE、gradual-underflow、FP/FMA设置及Torch dispatch语义；禁止fallback到全网。单行后端不自动等价于历史chunk dispatch。

剩余路径：先安装/核对确切依赖并完成无模型资源准入，制作及独立review单算子包装；冻结一个公共C保存pre-input/真实cache输入锚点，给两个后端相同字节。19×21×2=798；同时使用C和Torch两个锚点会需要1596，超出合同。每次仅消费保存输入，不把新输出传下一层。Torch重构cache只作为有明确标签的历史证据，不能冒称真实观测。单进程单线程CPU、禁网、无GPU；798行/120s/RSS512MiB。超限、身份或语义不符即停止，不能涨额度后继续。

原门限保留：raw atol1e-4/rtol1e-5；probability atol1e-5/rtol0；frontend atol1e-3/rtol0；composed_cmvn atol2e-4/rtol0。局部诊断不是v3整网资格，旧FAIL不覆盖。

## Shipping：已有保存证据，当前不建议新模型运行

重新下载并核验[PR450保存artifact](https://github.com/jiying2007/kws-pipeline/actions/runs/36689641752/artifacts/11085476672)：23,590,873bytes，SHA256 dafaa468c0d9d6ffb323dca2845df74128ea128f0fc3870ce2d4224fcfe26672。它绑定历史nightly run36640828135/artifact11068181171，和本轮run37860130019/artifact11585689303是不同运行。

一致性证据：模型ece44b47…d402、关键词包370ee3ee…b723、两个source hashes、seed和目标事件时间/置信度一致。独立读取WAV容器后，两段12s mixed-context音频最后4s PCM与本轮两个捕获逐字节相同；PCM SHA分别cf16ec6fb1b91b21f2454266852374e2bc65e8cd3cf78ab8f8c9aead61208221与f66c69436d72f6605fb70711688481f69f3fdddca4137c8b69b979b1c45fbe82。相同PCM不证明不同运行的连续初态或trie历史相同。

保存值的复核结论：

- seed1103：旧full posterior的pre-10s切片和phase-aligned cold窗口各600帧，帧端点/VAD相同；463帧logits不同，最后差异7190.745s，尾部130帧逐字节相同。已有保存记录显示12.005s冷流在7191.345s/0.603046触发。目标邻域一直speech_active；原full trace在7191.345s有一帧wo1为top。此处读取并复核既有结果，没有重新运行。
- seed3301：已保存12s reset流599帧；目标邻域持续speech_active，3234之后31个blank-active帧，再进入343…。保存的selected-path聚合记录为4次advance、0 fuzzy、31次blank retention；该冷流事件confidence0.993992，不等于原事件0.962665。聚合/shadow统计不是真实逐帧trie记录，不能据此宣称原FA根因已确定。

[PR450原结论及限制](https://github.com/jiying2007/kws-pipeline/pull/450#issuecomment-5907722111)仍适用：短冷流足以产生该1103事件，不能单独归因frame phase，也不能批准阈值/reset修复。当前优先完善保存trace读数和纯decoder状态观察的独立方案，不新采集音频或运行模型。若未来确需新采集，仍须另立输入/初态/完整观测合同，两个固定场景各≤60s、总≤120s，仅一次，600s/RSS2GiB；保持原参数、禁止旧nightly/once重跑及挑seed重试。

## 保留分支

只读检查确认pipeline三个保留设计分支仍应按现有记录保留，不能自动合入或删除。data七个非main分支分别对应已合并PR21–27；大部分因squash仍显示diverged，不能据此认定未合并。没有删除或改动任何分支引用。

## 保存证据持久化补充（2026-10-09）

PR450 原始 23,590,873 字节工件现有[固定 data 来源](https://github.com/jiying2007/kws-data/tree/528cdc880256ad417fcb839cdeba89ba59c2d331/research/2026-10-09-pr450-saved-trace-retention)及[机器可读身份指针](DURABLE-TRACE.json)。三分片封装恢复后得到原始 ZIP，44 项成员逐项核验，原 Actions 工件不再是唯一恢复入口。外层封装与原始工件 ZIP 的 SHA 不同，不能混用。恢复只读取保存字节，不执行历史脚本、模型或 decoder replay；仍不证明两个运行的连续初态一致，不改变原 FA 结论。

The original PR450 artifact now has a commit-pinned restoration source in kws-data. This is persistence of existing evidence, not a new replay or acoustic qualification. See DURABLE-TRACE.json for original/wrapper identities and the source verifier.
