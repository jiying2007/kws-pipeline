# Archive delivery and recovery / 归档交付与恢复

Decision recorded 2026-10-09. The selected delivery is **complete public archive + index + verifiable restoration**. The old 500-file expanded pipeline copy is **cancelled as a delivery target**, not published or silently treated as complete. Its historical publication status stays **NOT_PUSHED**.

2026-10-09 确定交付方式为**完整公开归档＋索引＋可验证恢复**。旧 500 文件展开副本不再作为交付目标，不继续重复发布；历史状态仍为 **NOT_PUSHED**，不能改写成已发布。

## Current entry points / 当前入口

- [Fixed archive, index and verifier](https://github.com/jiying2007/kws-data/tree/d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a/research/2026-10-08-d20-d90-host-research)
- [Delivery decision and evidence identities](archive-delivery-decision-2026-10-09.json)
- [Full restoration acceptance](archive-restoration-acceptance-2026-10-09.json)
- [Old expanded-copy reconciliation](archive-expanded-copy-reconciliation-2026-10-09.json)
- [Earlier status snapshot](CURRENT_STATUS_2026-10-09.md) / [早期状态快照](CURRENT_STATUS_2026-10-09.zh-CN.md) and [historical publication index](source-retention-publication-2026-10-09.json), preserved as point-in-time records, not current PR status or ongoing work.

The archive retains **1,886 logical members / 1,403 unique objects**. The eight-part ZIP is **59,417,250 bytes**, SHA-256 `8719a8ed67f440afe4adc450d47418523a7595dd02fb674c77da686964c84368`. Independent acceptance at fixed data commit `d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a` restored all members and re-read every path: **223,112,990 publication bytes; zero missing, extra, size-mismatched or hash-mismatched files**. All 29 pinned verifier safety tests passed without skips.

完整性范围是归档 CATALOG 声明的公开发布字节。公开转换前的原始字节、旧展开副本的全部 500 个包装文件，以及历史实验复跑，均不是此恢复验收的承诺。

## Relation to the cancelled 500-file copy / 与旧 500 文件副本的关系

The reconciliation distinguishes **471 exact archive matches**, **4 documented navigation projections** whose full originals are archived, and **25 files without an identical archived object**. The last group includes licensing/provenance metadata, manifests, CI, validators and an audio-free UI derivative; they must not all be described as scaffolding or as byte-exact archive coverage. Of those 25, **18 have exact public sidecar copies at the fixed data commit**, while **7 are cancelled expanded-delivery packaging or derivatives**, not exact retained bytes. The audio-free UI derivative retains the surrounding UI/JS from the archived full HTML but has different wrapper/payload bytes. In total, 489 of the old 500 have exact retained bytes; all 500 are classified. See the per-path reconciliation for evidence and disposition. Cancelling redundant expansion does not change what is retained. The retired expanded-copy workflow also invoked a historical 19-test synthetic admission suite; its source remains archived, but restoration acceptance does not run it or claim that other current safety tests replace it.

逐项对照区分 471 个字节一致文件、4 个有明确记录的导航投影（完整原件在归档），以及 25 个没有同 SHA 归档对象的文件。后者包括许可/来源元数据、清单、CI、验证器和去音频 UI 衍生文件；不能笼统称为脚手架，也不能说旧 500 个文件都被原样归档。其中 18 项在固定 data 提交中有同字节公开外围副本；余下 7 项是随展开目标取消的包装或衍生文件，不算原样保留。旧 500 项中共 489 项有同字节保留位置，500 项均已分类。取消重复展开目标与原样保留范围是两个不同事实。旧 CI 的 19 项 admission 测试仅保留源码，本次恢复验收未执行，也不声称现行其他安全测试提供等价覆盖。

## Reproduce restoration / 可复现恢复

Use a fresh checkout and Python 3 on POSIX with directory-descriptor and O_NOFOLLOW support. This downloads public archive content; it does not run archived research, models, training, TTS or ASR.

```sh
git clone https://github.com/jiying2007/kws-data.git kws-data-archive-review
cd kws-data-archive-review
git checkout --detach d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a
cd research/2026-10-08-d20-d90-host-research
python3 -m unittest -v test_verify_archive
python3 verify_archive.py --root . --restore ../restored-public-members
```

The destination must not already exist and its parent must exist. Verification checks archive parts, ZIP/object identity and catalog mappings before restoring data. Restored files remain data; old approval fields and executable examples do not authorize running them.

The acceptance receipt's `inventory_file` and its SHA-256 identify an audit-generated path/size/hash report, not an additional required download. The authoritative path and hash inventory is the fixed public `CATALOG.json`; independently re-read every restored member and compare `path`, `published_bytes` and `published_sha256`, including exact path-set equality. No restored member bodies or redundant full catalog are copied into this repository.

恢复目标目录必须尚不存在，父目录必须存在。验收回执中的 inventory 文件与哈希用于标识当次审计输出，不是另一个必需下载项；公开固定 CATALOG 是规范清单。恢复后可逐文件读取并核对路径、发布字节数和 SHA-256，且检查没有漏项或额外文件。

## Qualification remains unchanged / 产品资格不变

D20 raw-logit numerical gate failed; D90 was not run. Fixed50 remains `EVAL_INCONCLUSIVE_LABEL_SUPPORT`. The old frozen-model nightly failed separately. Human/final-AFE and physical target-device qualification remain deferred; `shipping_approved=false`. Archive integrity is not acoustic qualification, experiment reproduction, or authorization for new acquisition/training/device work.

D20 raw 数值门限失败、D90 未跑、fixed50 支持不足及旧模型 nightly 失败分别保留；真人/最终 AFE 与端侧实机未验，出货批准仍为 false。归档恢复通过不改变任何产品资格。

## Historical CI identity and current checks / 历史 CI 身份与当前检查

The delivery decision's `ci_change.after` is an immutable record of its original
workflow bytes. Its [non-active local snapshot](history/ci/research-source-consolidation-archive-delivery-2026-10-09.yml.txt)
is checked by size, SHA-256 and Git blob identity without network or Git history.
It is not a workflow to activate or execute. Subsequent CI changes do not rewrite
that decision or turn its old acceptance receipt into a new run.

Current archive metadata checks run in `ci.yml`'s unconditional `python-contracts`
job, alongside local retention checks. Their integration is tested independently
of the historical workflow hash; the archive check step downloads nothing and
never runs restored content. CI run results for a new commit must still be checked
separately; these structural assertions do not claim a new hosted CI pass.

已把历史字节身份与活跃 CI 结构分开：历史 decision 和验收回执保持原样，离线检查
核对非激活快照；现行检查随普通 CI 执行，不再因后续 workflow 合理调整而误报历史
证据失效。本次仅调整导航与检查接线，不新增模型、单算子、训练或采集执行。
