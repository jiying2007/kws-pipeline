# 研究与保留状态：2026-10-09

观察时间：2026-10-09T02:43:25Z。本页是有日期的状态快照，不是 PR 或分支的实时状态。
[English](CURRENT_STATUS_2026-10-09.md)。

## 公开归档与入口

- [data PR #27](https://github.com/jiying2007/kws-data/pull/27) 在观察时仍为
  draft、open、未合并。完整公开归档有 **1886 个逻辑成员、1403 个唯一对象**，共八个分片。
- [pipeline PR #501](https://github.com/jiying2007/kws-pipeline/pull/501) 同样为
  draft、open、未合并。展开的 500 文件 pipeline 源码副本仍为 **NOT_PUSHED**，
  受发布工具阻塞。归档公开不代表这些源码已进入 pipeline main。
- [公开来源索引](source-retention-publication-2026-10-09.json) 记录不可变来源身份、
  校验值及精确 head 的 CI 快照。CATALOG 的修订 commit 可与原归档及分片 commit 不同。
- CI 显式获取两个固定身份的公开元数据文件 ARCHIVE.json、CATALOG.json（合计最多 2 MiB），
  验证 SHA-256、Git blob SHA-1，再核对成员数、唯一对象数、连续分片、CAS 映射及字节总数。
  离线检查使用 `--metadata-dir` 指向原始元数据；不下载对象正文、音频或权重。
  完整对象内容由 data 的发布 CI 验证。
- CI workflow ID、路径、事件、attempt 与 head 的检查仅针对已记录的快照身份。
  离线校验不能证明实时 PR 状态、为新改写的 CI 声明背书，或证明声学质量。

## 保留的研究结论

- **D20/D90 official-reference v2**：D20 同一失败 fixture 的两个 raw-logit 坐标超界，D90 未运行。
  没有形成通过的官方数值门槛或模型比较。
- **fixed50**：`EVAL_INCONCLUSIVE_LABEL_SUPPORT`。K2 仅 4 条、4 个 voice，低于 5；
  targets 仅 11，低于 12；unknown 8。没有可采纳的 KWS 比较或准确率结论。
- **旧 frozen 模型 nightly**：[run 37860130019](https://github.com/jiying2007/kws-pipeline/actions/runs/37860130019)
  为 8 小时 2 次误唤醒，upper95 为 0.786974 FA/h，违反 0 次误唤醒、upper95 ≤ 0.40 的要求。
  模型 SHA-256：`ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`。
  这是旧模型结果，不能归到 D20/D90。README 中历史工程资格不代表这次 nightly 通过。
- **真人与物理板端验证暂缓**。本次整理不授权新的听测或实机实验，shipping approval 仍为 false。

## 历史分支清单与已完成清理

2026-10-07 的 61 个非 main pipeline 分支清单是历史快照，不是当前分支清单，
也不代表这些分支仍全部保留。另行批准的清理已删除 pipeline 57 个、data 22 个分支。
两个恢复标签 `archive/branches-2026-10-07` 与
`archive/branches-2026-10-07-prune-anchor` 保留恢复锚点。
详见[清理完成记录](git-atomic-prune-2026-10-08.json)。
原始失败记录与旧 retention 快照保持不变；本页只替代过时的当前状态说明。
