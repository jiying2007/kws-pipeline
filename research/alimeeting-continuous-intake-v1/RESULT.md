# AliMeeting Eval 实际获取与审计结果

2026-10-14。已完成数据获取、字节/格式/转录/来源元数据审计，以及后来明确授权的一次冻结 C 诊断回放；未训练、改阈值或上传 GitHub。

## 已完成

- 官方唯一 URL：https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
- 一次正常受限获取成功：3,673,718,355 bytes。
- 完整 archive SHA256：`dc47343b2474b5ebcf458927e878155f6ddeb59c85e685b3645c32a1f9578d92`。
- 这是本次完整下载的观察锁；没有发布者 SHA256，不能改写为发布者校验通过。
- OpenSLR 官方数据页 https://www.openslr.org/119/ 标注 CC BY-SA 4.0；源码仓库 Apache 许可不替代数据许可。许可链接 https://creativecommons.org/licenses/by-sa/4.0/ 。
- 大包位于本目录 `raw/Eval_Ali.tar.gz`，实际审计位于 `audit/`；均在持久 workspace，未放 `/tmp`。
- archive 完整安全遍历成功：73 成员，声明文件正文合计 5,385,868,607 bytes。未解压全部成员；只暂存单个源 WAV 并保留所选 far channel-0 与 TextGrid。
- 8 个 far 源均为 8ch/16k/PCM16 WAVE_FORMAT_EXTENSIBLE，PCM GUID/有效位数、RIFF 大小、帧对齐与 channel mask 均通过。channel-0 是原始字节步长抽取，没有混音/增益/重采样。
- 8 个完整单声道文件合计 **242,211,440 样本 = 15,138.215 秒 = 4.205059722 小时**。每个会话连续，未跨会话拼接，不乘 8 通道。最长一段 2,239.4705 秒。

## 官方转录实际扫描

8 个 TextGrid，25 个说话人 tier，6,457 段人工转录。全部 tier 与分段边界均扫描；同 tier 跨段连接只产生保守文本候选提示，不推断声学连续发音。

预声明短语“你好小窝”“小窝小窝”“小屋”“小我”“小沃”“小卧”“小窝”均为 **0 字面命中**。这不是完整拼音/近音覆盖，也不证明未转录区没有目标词。官方人工转录是现成候选证据，不因缺新人工复听而永久否定这批数据。

真实 TextGrid 为稀疏话语区间，省略大量说话人间隙；这些标注间隙不是丢失的 PCM。全部区间时间有限、有序、不重叠且处于其 TextGrid 范围内。各 TextGrid 全局 xmax 早于 WAV 末尾，合计 **47.384 秒**；保留差异、不删尾、不移动时间戳。它们是下一步针对性复核点，不能未经核实称为已确认无目标的静音。

| 会话 | WAV 秒数 | TextGrid xmax 秒数 | WAV 末端超出 TextGrid 秒数 |
|---|---:|---:|---:|
| R8001_M8004 | 1573.8500000 | 1550.953000 | 22.8970000 |
| R8003_M8001 | 2068.0000000 | 2067.709000 | 0.2910000 |
| R8007_M8010 | 1856.3219375 | 1853.911000 | 2.4109375 |
| R8007_M8011 | 1861.5460000 | 1858.993000 | 2.5530000 |
| R8008_M8013 | 2239.4705000 | 2235.500000 | 3.9705000 |
| R8009_M8018 | 1654.6570000 | 1651.486000 | 3.1710000 |
| R8009_M8019 | 1973.9260000 | 1964.670000 | 9.2560000 |
| R8009_M8020 | 1910.4435625 | 1907.609000 | 2.8345625 |

## 已完成的自动来源排重

本次原 8ch WAV、派生 mono WAV 和 decoded mono PCM，分别在正确的 hash 类型空间，与以下保留元数据对照，**精确匹配均为 0**：

- 历史模型训练 1856 条（1728 train.tsv + 128 hard-negatives.tsv）
- 旧 FAR 两池共同的 224 派生注入 WAV，对应 112 原源家族；完整 domain index 含 120 原源家族
- 实测库存 385 文件及 22 个原生非规范源文件

875 个候选元数据/文档（含压缩收据）的 AliMeeting/会话名扫描也没有命中。历史 speaker/session 标识不完整、原训练 TSV 正文缺失；无全文件匹配不排除片段、其它通道、增益、重采样或其它变换。因此尚不宣称 speaker/session 级独立，也不把它计为 sealed qualification。

## 证据状态和下一步

`accepted_negative_duration_s=0`，8 条 references 仍为 `expected=null`。数据准入审核指标签/来源/评估证据审核，不是重新索取用户授权。

已从固定已合并主线 **11b93abe92427c3ece2a01f9a497cdb74bab0062**、tree **b8ae29e82097c38e92dbb1bb1cd4db10ade6cdc6** 构建专用 `kws_wav`：官方 CMake 3.31.6、GCC 14.2.0、strict Release。runner SHA256 `53a5dcb50c599d88bb72b2cfbaf6ff5d6e7a379ecef3011f28f5820261d155a9`；完整构建收据 SHA256 `78b2988e60a4bf6b875934222fd5e732633ca34de64e470c433e74289f57ac8a`。2382 可用源码叶匹配冻结清单，1 个不可用历史研究归档叶不属于构建输入。`kws_wav` 未调用 `kws_build_info()`，链接器省略该未引用对象；源码身份由构建收据与 binary hash 绑定，不声称 runner 内含该 revision 字符串。

对 8 个完整会话只回放一次：每会话单独进程、内部连续状态、160 样本块，无补零/flush/调参。**8/8 完成，242211440 样本、756905 帧，0 触发，诊断触发事件率 0 次/真实输入小时**。所有 processed_samples/frames、源/runner/model/pack/WAV 前后身份通过；无丢样、discontinuity 或 external VAD。完整 receipts 位于 `diagnostic-run-p4-11b93abe/`；空 detections 由每个成功 stats 支撑，不是失败输出。

**这不是已确认 FAR=0，不是准确性提升，也不提供 recall/FRR/延迟结论。** 未标负例、未做 sealed qualification。下一步可结合既有官方人工转录和注释边界/歧义内容做准入复核；无需因缺少新增人工复听而永久搁置数据。

## 模型实际执行与正例响应边界

读取真实 `.kwm` 与 C dispatch，确认是 **32 输入/64 状态/5 输出的非恒定 int8 tanh 小型 RNN**，不是全零或固定输出矩阵。wx 非零2015/2048，wh3999/4096，wo311/320；5个输出行各异。每个完整特征帧实际走 infer_frame→decoder，VAD 控制 speech_active 而非绕过推理。统计 speech_frames=621690，blank_top1_frames=746569，nonblank_top1_frames=10336。没有另跑 posterior trace，逐帧推进证据来自源码路径及已保存计数。

同 SHA provenance 记录 AdamW、36 epochs、1856 合成训练例。公开限制是合成 TTS/tone 与模拟声学、没有真人语音；**当前配置 backend=tone 不能证明历史训练一定是纯 tone**。这些证据不证明模型无效，也不证明真人检测效果合格。快速主机执行本身不证明算法被关闭；本次不是产品板端性能认证。

仍缺**与本次 binary/model/pack 绑定的正例响应控制**，不能把“从不触发”误判为效果好。限定的非 heldout 开发证据复核未找到该控制：near_pronunciation 仅合成负例机制记录；M1/N1 正接受属于另一个 A20/offline posterior 分析；历史同模型误触发不能替代当前正例敏感性。新的32条 TTS 未完成 ASR 文字确认，不能报 recall。未为此打开 heldout 语料或追加任何模型回放；未使用 heldout 指标。

可后续按已定方案找回有原始 hash/明确正标签的 TRAINING/开发正例，建立当前冻结 tuple 的响应控制。当前报告保留 `shipping_approved=false`、`recalibration_required=true`。模型静态边界和正控复核分别在 `model-scope-audit.json` 与 `positive-control-review/review.json`。

## 可追溯修复与测试

- versions/v1：原离线准备版；v2：增加下载失败保留部分字节/哈希收据。
- versions/v3：支持实际 PCM extensible 格式；首遍审计因稀疏 TextGrid 严格 partition 假设失败，产物保留 `audit-incomplete-v3/`，没有成功收据。
- versions/v4：支持明确的稀疏注释区间与较短 TextGrid 边界；完整 audit 成功，原 producer 保留。
- 当前 v5：修复重叠短语漏计，36 项合成测试在 normal/-O/-OO/-I -S 全通过；所有 8 个真实 TextGrid 重扫与 v4 成功收据完全一致，见 `textgrid-rescan-v5.json`。
- 来源比对脚本另有 14 项合成测试通过；所有原始 candidate/历史文件未改。

关键收据：`source-plan.json`（当前观察锁）、`raw/acquisition.json`、`audit/audit.json`、`overlap/alimeeting-overlap-result.json`、`textgrid-rescan-v5.json`、`SHA256SUMS`。仅数值、路径和哈希元数据可用于公开收据；本轮没有上传音频或转录正文。
