# D1 已知训练样本的有界拟合诊断

`training/research_d1_fit_diagnostic.py` 用已封口 D1 preparation 的 4 或 8 条已知训练样本检查可学习性。它补充 D1 实际训练已经观察到的空序列、固定单 token 输出诊断，不修改原九组实验、原评估人口、原弱标签或原训练收据。

## 问题与复用边界

原 D1 执行已证明真实消费：固定 2601 词表、完整文本、CTC 权重、特征、初态和优化步都有回执。主线仍需回答：新初始化模型能否在少量固定训练输入上学会完整文本，以及空输出由怎样的 logits 和梯度伴随。任务成功、loss 下降、编辑数减少均不能单独回答这个问题。

已有 `tools/diagnose_acoustic_alignment.py` 的统计服务于 KWSP 二进制和 C runtime；它不是 D1 2601 输出 Torch checkpoint 的执行入口。新入口复用 `research_d1_evaluate.score_logits`、`save_numeric`、原 `fulltext_ctc`、原 `build_fresh`／FSMN 构造器和原 `train_batches`，没有复制 CTC、前端、编码器或优化循环。

历史 `research_d1_train.CODE_PATHS` 保持不变，已有 preparation 不需重新生成。新诊断具有自己的 schema、源码哈希、预声明 recipe 和队列 claim。它不覆盖、恢复或重新标记旧实验。

执行仍要求原 preparation 的全部 20 个源码文件逐字节匹配。已完成 accounted256 preparation 对应的公开执行 commit 是 `da268dc93fdf1b4b8b5276f042fd83a80e2a04f1`；后继主线向 runtime JSON 的 `validation` 添加历史记录也会触发原有严格检查。消费该历史 preparation 时，在其冻结 commit 的新独立工作树叠加本诊断的新增文件执行，保留两部分来源。不要改旧 runtime 文件、重写 preparation 哈希或把原 source 检查改成忽略某些字段。

## 固定样本与预算

`declare` 只读核验封口 preparation，重新检查原目标投影及 train/development 来源隔离，再从 **原训练 cohort** 选取固定样本。每四条包含：K1 完整正例、K2 完整正例、普通负例、部分唤醒负例。选择按已有内容标签、来源家族轮转和稳定 source ID 排序；不读取模型预测。8 条时每类各两条。缺少任一类别时明确报错，不用其它类别或开发样本补齐。

生成失败、不确定、OOV、掩码关闭、零权重或被隔离记录不能进入这个小集，也不会被转成空文本／负标签。全部目标 ID、完整文本投影、原置信权重及关键词掩码保持不变。近音比例对照仍属于原实验；该诊断不挑选近音赢家。

原 `features.pt` 与逐样本特征 SHA 原样复用，仍使用 2601 输出、blank=0，所有 target 为原非 blank ID。模型从同一 preparation 的原初态开始，RNN64/C1 和 FSMN head 分别保留原优化器配置。FSMN 的非 head 参数仍冻结。改变的只有抽样日程：每步 4 条，按已声明 4／8 条顺序循环。

默认预算为 400 步、180 秒；recipe 要求 1–2000 步且不超过 600 秒，执行还受现有队列的实际总预算和重模型互斥锁限制。无质量重试、自动延长、额外随机种子或最好 checkpoint 筛选。固定预算完成但未学会，仍是明确的阴性诊断结果。

## 入口

运行时沿用 `configs/runtime/research-runtime.json` 的 core 角色：Python 3.12.16、Torch 2.13.0+cpu、NumPy 2.3.2、SciPy 1.17.1。`--help` 和 `declare` 不导入 Torch，不构造模型。

以下变量均由现有已核验路径填写；`D1_PREPARED_SHA` 应来自已完成 preparation 的队列输出收据。

```bash
"$KWS_CORE_PYTHON" -B training/research_d1_fit_diagnostic.py declare \
  --prepared "$D1_PREPARED" \
  --prepared-manifest-sha256 "$D1_PREPARED_SHA" \
  --candidate c1-ctc2601 --count 4 --steps 400 --timeout-seconds 180 \
  --output "$D1_FIT_RECIPE"
```

将该 recipe 的绝对路径与实际 SHA 写入现有 `research_queue` 的新 job。其 `argv` 使用真实 core Python 路径、脚本绝对路径及：

```text
run --recipe {recipe} --output {output_dir} --heavy-lock <共享重模型锁绝对路径>
```

job 必须声明 `summary.json`、`consumption.jsonl`、`before.json`、`after.json`、`final.pt`、`MANIFEST.json` 六个输出。声明足够且固定的队列 timeout/global budget；先 `research_queue.py init`，再 `research_queue.py run`。直接绕过队列执行 `run` 会拒绝。每个候选使用独立 recipe 与输出目录，所有候选串行共用原 heavy-model lock。

## 记录与判断

诊断在优化前和固定最后一步后，各对同一小集执行一次前向，保存原浮点特征和 logits、每条真实 target NLL、greedy token 与编辑数。没有开发前向、ASR/TTS 调用或生成扩量。

新增统计为：blank 的 argmax 帧比例及平均后验、目标 token 集的概率质量、其余非 blank 的概率质量、各目标 token 最优竞争名次／top-3 帧数／argmax 帧数／最高概率／相对 blank 最大 logit 间隙。竞争名次对同分采用相同排名，单字曾进入 top-3 不代表整句存在有效按序对齐。

聚合明确列出空输出数、单 token 输出数、不同预测序列数，以及“所有片段固定同一单 token”。梯度报告来自原实际消费账本：有限性、非零／零梯度步数、范数范围；另核验真实参数变化、冻结参数未变、保存 checkpoint 回读一致。小集、原权重和每次抽样均关联原 row identity。

`status=completed` 仅表示固定执行与证据保存完成。`tiny_fit_passed=true` 还要求最后一个 checkpoint 对所有选中训练样本整句精确匹配，并出现非零梯度及真实参数更新；成功也只证明这批训练样本的记忆能力。它不提供独立人声质量、泛化、FAR、口音非劣或 SSC305 性能结论。

## 不重跑模型的回读

将最终队列收据中的 `MANIFEST.json` SHA 用于：

```bash
"$KWS_CORE_PYTHON" -B training/research_d1_fit_diagnostic.py audit \
  --output "$D1_FIT_OUTPUT" --manifest-sha256 "$D1_FIT_MANIFEST_SHA"
```

回读重新校验文件、所选训练行、原初始化与最终参数、实际消费及梯度账本，并从保存的 logits 重算前后统计。它没有模型构造、模型前向或优化步骤，也不改写原结果。

默认测试是元数据、选择与账本反例；`--numeric` 添加人工 logits 和小 tensor 检查，不下载权重或执行候选模型。

```bash
python3 -B tests/test_research_d1_fit_diagnostic.py
"$KWS_CORE_PYTHON" -B tests/test_research_d1_fit_diagnostic.py --numeric
```

## 2026-11-05 的实际固定 8 条执行

当日已从 accounted256 的 154 条已知训练 cohort 中，按上述预声明规则固定 8 条、6 个来源家族、合计 32 个目标 token。两关键词正例、普通负例、部分唤醒各两条；两个候选使用完全相同的来源和 row binding。两个独立任务各声明 400 步、每步 4 条、180 秒，队列总预算 6 分钟，没有追加步骤或改变学习率。

两个任务均实际完成，合计 800 个优化步、3,200 次训练曝光、32 次前后诊断前向；没有开发、ASR 或 TTS 前向。队列实际任务时间总和为 **99.164039888 秒**：RNN64 为 64.926815852 秒，C1 为 34.237224036 秒。该时长含进程和证据开销，不是目标设备实时率。

| 观察 | RNN64-CTC2601 | C1-CTC2601 |
| --- | --- | --- |
| 优化前 → 后整句精确匹配 | 0/8 → 0/8 | 0/8 → 1/8 |
| 优化前 → 后平均完整目标 CTC NLL | 606.225470543 → 12.924413264 | 593.955979347 → 5.193658620 |
| 最后 checkpoint 的编辑数／目标数 | 26/32 | 16/32 |
| 最后 checkpoint 的空输出 | 0/8 | 0/8 |
| 最后 checkpoint 的单 token 输出 | 8/8，同一个 token | 4/8，全部输出共有 5 种序列 |
| 最后 checkpoint 的 blank argmax 帧比例 | 98.750000% | 94.843750% |
| 最后 checkpoint 的平均 blank 后验 | 0.951962217 | 0.931247427 |
| 有限且非零总梯度范数的步骤 | 400/400 | 400/400 |
| 实际改变的可训练参数 tensor | 5/5 | 16/16 |
| 参数变化 L2 范数 | 46.954453525 | 25.358631872 |
| 完整记忆判据 `tiny_fit_passed` | false | false |

这些结果证明当前真实循环确实消费了输入、有非零梯度并更新了参数；它们同时表明，固定 400 步下两个候选都没有完整记住这 8 条训练样本。RNN64 的 loss 和编辑数下降伴随固定单 token 的退化，C1 仍有大量字符删除。单看 loss 下降不能据此扩大质量结论，也不能因本次未通过就断言架构永远不可学习。未执行数据重标、开发调参、额外种子或质量重试。

执行工作树为原源码 `da268dc93fdf1b4b8b5276f042fd83a80e2a04f1` 加上独立诊断提交；实际派生 commit 为 `5393a0f`。原准备包保持只读，SHA-256 为 `7ce118a7742f1554452a0b44d43fed4dc387cd5e797004e4a1d660948d751e3f`。两个已保存结果均在新进程完成 logits、梯度账本和参数回读；两次回读各自新增模型前向和优化步均为零。

| 证据 | SHA-256 |
| --- | --- |
| 共同 8 条 row binding | `ba70265d8e30678f7eac4c436c7af614a0de8993195842a10b2e183d8429a5ca` |
| RNN64 输出清单 | `12b9251e0e6a03bac509618ac7cb2a818c6402642fd2e079478f9e373db22fa9` |
| C1 输出清单 | `999d8a70120eb2ec383178d9aef6cb64a4967b3401ee904cc89031798ceffabd` |
| RNN64 保存结果回读 | `697b1cb94dfc6e947ccb288eafbf9be7f14bc6453f7d6ba7d44fcdadf2eb944d` |
| C1 保存结果回读 | `177af7487a35a048fdacbb792ee2ad952cfea08e59a639499899a275910c04a1` |

完整浮点输出、消费账本、原初始化绑定和任务结果单独保存。上述指标只对应这次机器弱标签训练小集，不是独立人声验收或自然误触率。
