# 唤醒词语料归档路线与首个候选包（reviewing）

## 决策建议

**音频本体归档到独立私有数据仓或受控对象存储，代码仓只保留不可变数据集引用、内容 SHA-256、许可与验收摘要。** 当前 `origin` 是公开的 `github.com/jiying2007/kws-pipeline`，没有 `.gitattributes` / Git LFS 跟踪规则；把 WAV 直接加入本仓会让合成语音与未来真人/最终 AFE 音频混用同一公开边界，也增加每次 clone、CI 与源码审查的负担。GitHub 的[仓库限制说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)建议二进制用 Git LFS、程序生成数据放对象存储；[Git LFS 协作说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage)也明确未安装 LFS 的协作者只会拿到指针文件。因此数据仓若选 Git，必须把 LFS/访问权限/配额及 CI 拉取方式一并固定。

| 数据状态 | 建议位置 | 代码仓记录什么 |
| --- | --- | --- |
| 已逐条审听的合成开发语音 | 独立私有 `kws-datasets` 仓（小批量可用 Git LFS）或受控对象存储 | 数据集 ID、版本、bundle/manifest SHA-256、许可/生成器/审听摘要与加载入口 |
| 生成成功但未审听或有错读疑点 | 受控暂存/隔离区，不进入“合格语料”版本 | 仅诊断记录与不合格原因，不被训练配置引用 |
| 真人原始语音、最终 AFE 后语音 | 权限受控对象存储，分离训练与封存资格访问权 | 脱敏的内容摘要、场景覆盖、设备/AFE 身份及只读受控证据引用；不公开原音频 |

当前 20 条 Qwen3 已由用户在拷贝后的审听包中逐条确认“全部正确”；拷贝包 SHA-256 与原包完全一致。它可成为**首个已审听合成开发语料候选**，但不自动成为产品资格。原 AISHELL3 训练池 128 条此前未找到逐条审听收据；即使其中 32 条正例与 96 条负例都通过格式、索引和 hash 检查，仍必须保持 `pending-review`，不能与 Qwen3 的 `accepted` 状态混称。

用户随后对 AISHELL3 训练池给出“音频基本没问题，但质量没有 Qwen3 高”的来源级反馈。该反馈足以继续把它作为历史基线开展探索性配对训练，不足以把 128 条全部逐条标记为 `accepted` 或进入“合格语料”数据版本；尤其负例需要确认没有误读成完整唤醒词。

用户随后授权后续用[固定 ASR 准入标准](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)筛选合成语料。原 128 条可按该标准逐条分为自动接受/拒绝的**非商业研究子集**；这不等于将原始 128 条整体改成已合格，也不覆盖 Qwen3 已有人审的接受结论。ASR 会错剔部分实听正确音频，见 Qwen3 20 条对照。逐条机器收据与人工收据须保留不同证据类型。

实际回读结果为 AISHELL3 原 128 条仅 14 条 ASR 精确匹配，正例仅 2/32，第一目标词两种文本合计 0/16；另一套 Qwen3-ASR 对 AISHELL3 正例虽提升到 11/32，第一目标词仍 0/16。因此不能把 14 条自动接受子集直接归档为完整的双词训练集，原有来源级“基本没问题”也不能被解释为逐条通过。机器收据仅作为研究筛选证据，后续应先补第一词的可信声源并排查标签/发音差异。

## 已准备的本地交付候选

- 数据集 ID：`qwen3-xiaowo-reviewed-development-20260928`；目录 `build/dataset-archive-candidates/qwen3-reviewed-v1/`，便携包 `build/dataset-archive-candidates/qwen3-reviewed-v1.zip`。`build/` 被 `.gitignore` 忽略，这只是本地候选，不代表已推送、备份或持久化到远端。
- 包 SHA-256 `ed7ccc918422cd2a241728dee853af57bf3b3705818c7eaaba19bc1e231cf196`；`manifest.json` SHA-256 `10eb407871a9f7f1ecf43c0085867a2e21b66574bf97b33aafb83f21d6270e7b`；匿名审听收据 SHA-256 `affa420dba5b9b84b000772d8d5d7fbd0788caf3b90fdcd679e97f519ff20613`；20/20 WAV 逐条 SHA-256 与包内清单一致，压缩包完整性检查通过。
- 包内有 20 条 16 kHz 单声道 PCM16 WAV、每条源 ID/文件与 PCM hash/声线/seed/正负标签、Qwen3 权重与生成代码精确身份、审听收据和用途说明。模型权重不在包内，只用固定 revision/哈希引用；`qualification_allowed=false`，`scope=internal-development-only`。
- 原 AISHELL3 128 条审听包在 `build/review-packs/kws-aishell3-train-review-20260928.zip`，SHA-256 `9c77ecffd37b982bc39a5ca8f605a86acd1d77c6bc08d0be49ac222abf73d098`。8 个声线 × 16 种短语，32 正例、96 负例，总 123.86 秒。它是**待审听包**，不得作为已归档合格版本发布。

## 正式数据仓契约

1. 每个版本不可变：`dataset_id`、版本号、bundle SHA-256 和 manifest SHA-256 绑定；修正一个录音/标签须发布新版本，不能覆盖旧包。训练清单记录确切版本与每条文件/PCM SHA-256，而非 `latest`。
2. 正例、负例、生成器、说话人、seed、原始源语音及所有声学派生关系都进入 manifest；同一源的增强变体只能落同一 split。人工审听收据绑定 `source_id`、文件 hash、预期文本、标签类别、关键词 ID 和匿名审听者；被拒/不确定的录音不进入可训练版本。
3. 数据仓与代码仓分权：数据仓只授权需要音频的训练/声学 owner，代码仓及普通 CI 不自动拉取原始音频。正式真人资格语料和开发训练语料还要进一步隔离，保留 no-retry/盲测语义；不能通过“私有 Git 仓”这个名称推定满足个人信息或产品资格要求。
4. 正式写入前核对目标位置的权限、LFS/对象存储配额、备份与恢复、制品取回方式以及模型许可证据。当前 Qwen3 [模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)标注 Apache 2.0，但权重许可与合成音频对外再分发仍分别审查；本候选只声明内部开发用途。

下一步由数据仓 owner 选择具体私有仓或对象存储地址后，按上述已可复核的 bundle 做受控发布，并把**发布后可回读的不可变地址和 hash**回填代码仓。当前没有远端数据仓发布收据；Knowledge Provider 对本仓 route 仍未解析，本文件只作为 reviewing 草案。
