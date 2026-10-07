# 已审训练声线的代理场景增强配对（reviewing）

## 假设与冻结输入

前述 259 条重加权模型在已审 Qwen3 未训练 Serena/Eric 双词 4/4，真人近邻 15 人 7,006 条 3 次事件，但 Qwen3 原声的代理中距/播放场景与 Spark 缺首近邻暴露漏检/误触发。其训练输入主要是干净的短录音；“增加同家族干净 Qwen3”277 条配方已失败。因此本轮只检查**训练声学场景覆盖**：以原 259 条配方为控制，取用户逐条确认的 Qwen3 训练声线 12 条（Vivian、Uncle_Fu、Dylan），每条投影到两个事先固定的代理场景：

1. `near_room_train`：0.8 m、20°、RT60 0.25 s、SNR25 dB、风扇噪声；
2. `mid_room_train`：1.8 m、−45°、RT60 0.40 s、SNR18 dB、媒体噪声，无额外播放干扰。

每个原声与场景 seed 从 `source_id:scene-train-v1:scene_id` 的 SHA-256 派生，渲染输出为 16 kHz 单声道 PCM16；同一原声的全部派生仅留训练组。渲染器仍是仓库稀疏模拟 RIR + 代理双麦延时求和，不声称实际 AFE 或目标板。新增 24 条**候选**逐条用固定 Sherpa ASR 严格文本一致性及连续正例内部低能量间隔 `<120 ms` 筛选；每词至少 2 条正例、负例至少 2 条，未达则不训练。已审原声不能自动把强扰动派生音频标为发音合格。

若通过数据门槛，只增加通过的 `p` 条 exact wake、`q` 条其他非空样本；总样本数 `259+p+q`。两词权重从**计数公式** `(221+q)/(38+p) × 38/102` 得出，以保留原 140 条配方 exact wake 与非唤醒非空目标的训练质量比。仍为 RNN/H64、seed2346、600 epoch、batch16、lr0.001、同 CTC VAD/活动帧辅助目标、同词包/C runner；冷启动，不同时加入新音节词表或改解码器。

验收按[软件侧共同门槛](SOFTWARE_ONLY_CLOSURE_2026-09-28.md)：已审未训练声线双词 4/4、近邻0；HI-MIA-CW 留出15人事件不高于旧模型51次；独立 Spark 固定 ASR 合格词1至少2/2、词2至少2/4、合格缺首0；新 Serena ASR 合格双词各1/1、近邻0；原 near_clean 场景正例至少8/10且近邻0，mid_fan 近邻0；固定900秒注入流0次。任何一门失败就不晋级。场景已被用于开发诊断，不是新鲜产品资格；新增渲染仅为训练数据，不跨到验收 split。`retry_budget=1`、`staleness_threshold=source-or-policy-change`、`logical_task_open=true`、`completion_claim=pending`、`stop_condition=replan`。

## 数据准入检查点

已按冻结配置从 12 条训练声线原声渲染 24 条，manifest SHA-256 `d8bef0d4954bb9ef5e39627f033bdec497d5b78524f141db997ec1cb7ec58f59`。固定 Sherpa ASR 收据 SHA-256 `91e84992dc8bf6d8e96158a10bd8623c5a7bf7a7f43a213af6320b7730ec3ae9`：18/24 精确匹配，其中词1 **0/6**、词2 6/6、近邻 12/12。预设每词至少2条的门槛失败，**本轮未训练**。为检查识别器来源偏差，又用固定 Qwen3-ASR-0.6B 对同一24条只读回读，收据 SHA-256 `0d040d2cce0b042fe00872d58723ccd192db53d32e88032bdf718bff4bf47663`：词1同为 **0/6**、词2 6/6、近邻 9/12。词1被转成“小屋／小吴／小沃”等，不能按原文本自动加入正例；两个ASR同时失配仍不是人工真值，可能涉及场景失真和同向识别偏差。

已生成[六条词1场景审听页](../../build/review-packs/kws-scene-kw1-review-20260928/review.html)及[便携包](../../build/review-packs/kws-scene-kw1-review-20260928.zip)，ZIP SHA-256 `7d2f707db9f0afa9212ecf61506fb3cd7811a1aa621b4ceecf3b5473e55decaa`，`unzip -t` 通过。页面默认每条为“不确定”，可下载与 WAV SHA-256/来源/文本/类别绑定的 JSONL 收据。**待逐条人耳判定；没有收据前停在数据准入阶段。**`completion_claim=asr-coverage-failed-human-review-pending`，`logical_task_open=true`，`milestone_close=none`，`stop_condition=replan`。

## 用户直接审听反馈后的独立准入路径

用户对该六条反馈“qwen3-kw1-uncle_fu 不行，沃，其他四个尚可”。按审听页中仅有的两条 Uncle_Fu 场景与 Vivian/Dylan 各两条场景，将两条 Uncle_Fu 标为 `rejected`，其他四条标为**研究阶段可用**的 `accepted`；这项判定来源是用户在对话中的直接反馈，**不是页面导出的原始 JSONL 文件**。逐条按当前 WAV/source/text/kind/keyword 绑定形成的本地匿名收据 SHA-256 `2a8e91c6efd5998cbce80699bda44156f8e957059508ed416cb226e9aadc73df`，保留 `evidence_origin=user-direct-feedback-2026-09-28` 和“沃／尚可”的原意；如用户后来指出四条中的具体例外，须发布新收据并重训，不覆盖此版本。两条 Uncle_Fu 和机器错剔的其他词1录音都不能作为正例。

在与固定 Sherpa ASR 的**不重叠证据路径**下，词1 使用上述四条人耳接受，词2 六条与近邻十二条使用严格 ASR 精确接受，所有正例再经 `<120 ms` 内部低能量间隔检查。物化后为 22 条：词1 4、词2 6、近邻 12；manifest SHA-256 `c57702be7a5c5461b136e1ce94ba795c5e5ffcdfc665add323882e7538bc543f`。与原 259 条训练池合并为 281 条，exact wake 由 38 增至 48，其他非空由 221 增至 233；权重按预设质量比公式得 `(233/48)×(38/102)=1.8084150327`，不是 C 事件调参。1 epoch 输入烟测读取 281 条、vocab5，语料 canonical SHA-256 `f414d42cb9bebcc619c9836f9c9f815ecb0a7b2fc0c3c9129b87eb847060ebd4`。冷启动 600 epoch 与 C 端共同回读进行中。此分支只补充了人耳证据后重新评审数据准入，和上一段“纯机器标准失败”分别记录。

又将四条人耳接受与十八条机器接受拆成互不重叠的子收据，调用仓库 `validate_mixed_reviews` 重新验收全部22条，覆盖率与类别/关键词/WAV身份通过；人耳子收据 SHA-256 `a3f3e724c9810c225df92aad6ad71a1d9c8c8bf49d1560e9f2209b4b4ec73e1b`，机器子收据 SHA-256 `f8e3b4d62ffe093e32d733c20ae526dc958fc82c3090476e4e10d8a386fbd4ca`，[验证摘要](../../build/software-closure-20260928/scene-train/mixed-review-validation.json) SHA-256 `e92e606636902916894cb860a8fab28994bf1ecb231100f9d13c255d05f4827c`。该复验只证明收据与计划音频一致，不代替人耳对另外18条的逐条判断。

Qwen3 原训练声线12条与本轮入选派生22条组成34条训练子集；Serena/Eric 原未训练声线8条单列开发回读。按解码 PCM、`speaker_id`、`session_id`、`source_id` 运行[split 审计](../../build/software-closure-20260928/scene-train/qwen-split-audit.json)，报告 SHA-256 `69db96b612e305aaa7db76680f583165ea75d5daa48c7368a38f275ed5501c26`：34/8 条，跨 split PCM/身份泄漏 0、各组 WAV/PCM 唯一。派生训练音频与原干净训练音频保持同一说话人组；这个审计不能把 Qwen3 不同预置声线升级为真人独立说话人。

导出前补充一组未用于本次训练的代理场景压力切片：原已审 Qwen3 20 条按新种子渲染的 60 条，manifest SHA-256 `00a1d4ff32b2ffcba50ef461a0cf14eff7053b52a5d1ea5aee43a4ba0f0aaa62`，clean 源可能与本次 Qwen3 训练声线重复，因此它是**新场景参数回归**，不是独立源资格。259 条基线在 `near_clean_v2` 正确9/10、近邻2次，`mid_fan_v2` 正确7/10、近邻0，`mid_media_v2` 正确3/10、近邻2次。本轮处理模型须在前两场景分别正确至少9/10、7/10且近邻均0，在媒体场景正确至少3/10且近邻不高于2；派生标签未经复听，异常只作为压力线索。这些阈值在处理模型 C 回读前固定，不根据结果调整。

即使 seed2346 的首次配对全部通过，也只形成单 seed 开发候选；下一步固定相同语料、训练目标和门槛，以 seed1337 独立冷启动复验，两个 seed 共同达标才讨论软件候选封存。任一 seed 未达标就停止这一场景增强配方，不通过挑 seed、epoch 或回改权重晋级。

## 600 epoch C 回读与停止

seed2346 的 281 条冷启动完整训练和 C 导出完成：checkpoint SHA-256 `7171773dd3e4a892231bf29c2cfb4ce8a28951446409597c22d2e55460f52945`，KWM SHA-256 `c27c9f78ab14331bd85fb44d589c6db4b258a2ff2d1ea29484276cf7c533ca2d`；语料 canonical SHA-256 与 1 epoch 烟测同为 `f414d42cb9bebcc619c9836f9c9f815ecb0a7b2fc0c3c9129b87eb847060ebd4`。训练输入 281 条、vocab5，模型保持 `development_only`。同 runner/词包/参考文件的[机器回读报告](../../build/software-closure-20260928/scene-mixed-software-report.json) SHA-256 `ef6898413d34b77d7eee6e0cd73709e2e9b2ecee581d690faf41dfb1a4973351`，`software_candidate=false`。

| 冻结开发回读 | 原 259 条重加权模型 | 场景增强 281 条模型 | 判定 |
| --- | ---: | ---: | --- |
| 已审 Serena/Eric 两词 | 4/4、近邻0 | **词1 2/2、词2 0/2**，近邻0 | 双词门失败 |
| 新 Serena/Eric ASR 合格两词 | 2/2、近邻0 | 词1 0/1、词2 1/1，近邻0 | 词1门失败 |
| 独立 Spark ASR 合格正例 | 词1 2/2、词2 2/4、近邻0 | 同为 2/2、2/4、近邻0 | 持平 |
| HI-MIA-CW 留出15人、7,006条 | 3 次事件 | 6 次事件 | 小幅退化，但低于旧固定模型51次 |
| 原代理场景 near_clean 正例/近邻 | 10/10、0 | **6/10、1次** | 近场门失败 |
| 新代理场景 near_clean_v2、mid_fan_v2 | 9/10+2近邻、7/10+0近邻 | **6/10+1近邻、6/10+0近邻** | 两场景均未达预声明门 |
| 固定900秒连续流 | 0次、15近邻全覆盖 | 0次、15近邻全覆盖 | 该固定流通过 |

新增场景训练样本虽然有完整的人耳/机器分证据链，未使模型在**未训练声线的词2**或新的代理扰动上净改善；在原近场还出现额外近邻事件。按预设停止条件，**不启动 seed1337、不扫 epoch/权重/阈值，也不提升该模型**。从这些数据不能单独判断“场景渲染无效”：人耳只确认六条词1中的四条研究可用，独立生成器正例、模拟 AFE 与训练目标仍是不同变量。`completion_claim=scene-mixed-negative-keyword2-regression`，`logical_task_open=false`，`milestone_close=scene-augmentation-negative`，`stop_condition=replan`。
