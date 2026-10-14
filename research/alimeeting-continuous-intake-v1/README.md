# AliMeeting Eval：实际获取与连续候选流审计（2026-10-14）

## 交付与当前状态

独立准备目录，不属于 candidate 补丁。`alimeeting_eval.py` 只依赖 Python 标准库；没有 Torch、ASR、KWS 模型调用。默认打印离线计划；网络获取必须另行显式执行 `acquire --allow-download`。在后续明确授权下，本目录已完成一次官方 archive 获取及本地审计；后续又获授权执行了一个冻结 C 诊断 pass；未训练、修改 candidate 或写 GitHub。实际结果与严格边界见 `RESULT.md`。需要 Python ≥3.12（本环境 3.12.14 支持 PCM WAVE_FORMAT_EXTENSIBLE）。

**真实 archive/channel-0/TextGrid 审计以及一次冻结 C 诊断均成功，0触发仅是诊断事件率，不是确认 FAR。** 已保留 versions/v1、v2、v3 与首次不完整审计。真实兼容修复只增加明确 PCM extensible 子类型和官方稀疏 TextGrid 支持；不静默重采样、混音、裁剪或补齐标注。36 项合成测试在 normal、-O、-OO 和 -I -S 下通过。

## 官方来源、许可与精确首轮选集

- 数据页：[OpenSLR SLR119](https://www.openslr.org/119/)。官方标注 CC BY-SA 4.0，Eval 为 8 个会议、约 4 小时，含 8 通道远场与同步近场。
- **唯一计划下载项**：[Eval_Ali.tar.gz](https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz)。此前 HEAD 观察为 **3,673,718,355 bytes**（约 3.42 GiB / 3.67 GB）。完整下载字节数与此相同；现场 SHA256 为 **dc47343b2474b5ebcf458927e878155f6ddeb59c85e685b3645c32a1f9578d92**。这是完整字节观察锁，**不是发布者提供的校验和**，publisher_sha256 仍为 null。
- [CC BY-SA 4.0 官方说明](https://creativecommons.org/licenses/by-sa/4.0/)：保留署名、来源、许可链接和修改说明；对外分发改编音频需遵循相同许可。官方代码仓库的 Apache-2.0 **不是语料许可**。本轮仅计划公开数值/哈希收据，不重新上传原始或派生音频、转录正文。
- [作者论文 §2、§3.1、§4](https://arxiv.org/html/2110.07393v3)：Eval 有逐会议 TextGrid，包含各说话人 tier 的分段文本和起止时间；远/近场同步；baseline 取第一通道。论文当时明确未提供 Test 转录，当前 Test 包是否已补齐未核实，因此本轮完全不选 Test。
- [作者官方数据准备脚本](https://raw.githubusercontent.com/yufan-aslp/AliMeeting/main/asr/local/alimeeting_data_prep.sh) 给出 `Eval_Ali_far/audio_dir` 和 `Eval_Ali_far/textgrid_dir`；[官方 TextGrid 处理](https://raw.githubusercontent.com/yufan-aslp/AliMeeting/main/asr/local/alimeeting_process_overlap_force.py) 在 `--mars` 模式用 WAV stem 前两段下划线字段匹配 TextGrid stem。当前准备实现兼容完全相同 stem 或该规则，但存在歧义立即失败。

精确集合：包内 **全部** `Eval_Ali_far/audio_dir/*.wav`；每个真实会议只选零基 channel 0；对应 `Eval_Ali_far/textgrid_dir/*.TextGrid` 全说话人 tier。预期恰好 8 个唯一会议，实际 ID 必须在获取后的 inventory 中列出并锁定，实际 8 个 ID 及全部哈希已记录在 `audit/audit.json`。忽略 near 音频、不使用其它 7 个远场通道、不平均/波束形成、不选训练集。RIRS/MUSAN 暂缓。

## 有界获取与审计命令

当前可安全执行：

```sh
cd /tmp/alimeeting-eval-prep
python3 alimeeting_eval.py plan
python3 test_alimeeting_eval.py
python3 -O test_alimeeting_eval.py
python3 -OO test_alimeeting_eval.py
python3 -I -S test_alimeeting_eval.py
```

实际已执行一次获取，数据在本目录 `raw/`；以下是可复现接口，不会自动重取。使用已批准的免费环境，建议至少 **6 GiB 可用持久磁盘**；大包不可放 `/tmp` tmpfs。`source-plan.json` 保存当前实际观察锁；CLI `plan` 仍打印通用离线获取蓝图，初始计划存于 versions/v1。

```sh
# DATA_PARENT 必须是未来明确选定且已有的工作目录；输出目录必须不存在。
python3 /tmp/alimeeting-eval-prep/alimeeting_eval.py acquire \
  --allow-download --out "$DATA_PARENT/alimeeting-eval-download"
```

固定 URL、禁止重定向、只接受 HTTP 200 和该历史 Content-Length；单次下载上限 4 GiB、连接/读取超时 30 秒、总预算一小时，不自动重试、不续传、不展开 archive。失败保留 partial 字节及其哈希/错误收据，不冒称成功。下载流现场计算 SHA256，首次结果标为 observation-only；它不是发布者提供的校验值。完整 acquisition.json 已保存，观察 SHA 已固定到当前 source-plan.json。后续另行需要重取时应传 `--expected-sha256`。如果长度变化，应重新审查官方来源，而不是随意放宽检查。

```sh
# SHA256 是未来 acquisition.json 内复核后的精确值，本文件不提供占位假哈希。
python3 /tmp/alimeeting-eval-prep/alimeeting_eval.py audit \
  --archive "$DATA_PARENT/alimeeting-eval-download/Eval_Ali.tar.gz" \
  --archive-sha256 "$SHA256" --out "$DATA_PARENT/alimeeting-eval-audit"
```

审计先校验 archive SHA256，顺序遍历 tar.gz，无 extractall。拒绝路径穿越、绝对路径、符号/硬链接、设备、重复路径、稀疏文件，并在 tarfile 读取正文前拒绝 PAX/GNU 扩展元数据头（真实包若采用这些头，须先另行有界兼容审查）；限制成员数 4096、单成员 1 GiB、展开声明总量 12 GiB。只暂存一个原始多通道 WAV，生成整会议 channel 0 后释放临时文件。输入必须为 8 通道/16 kHz/16-bit PCM RIFF：支持 classic tag 1 或明确 PCM GUID、cbSize=22、validBits=16 的 WAVE_FORMAT_EXTENSIBLE；严格验证 RIFF 长度、block alignment、channel-mask 位数和整帧数据；不满足时停止，由后续独立审查决定转换，不能静默重采样。

输出：

- `audit.json`：archive/member/派生 WAV/decoded mono PCM 的 SHA256、真实 sample_count、sample_rate、duration、源路径、会议与 TextGrid 匹配、全部成员清单、待审标记。
- `mono/*.wav`：固定第一通道的连续自然会议，没有增益、重复、拼接、补零、VAD 裁剪或句级重置。
- `textgrid/*.TextGrid`：供本地人工复核的原始转录，保持原字节，不作为公开收据上传。
- `review-template.json`：绑定 audit/PCM/TextGrid 哈希的待审模板，不等于审核通过。
- `reference-candidates.jsonl`：**expected=null**，现有 scorer 会拒绝；不能直接运行并宣称负例评测。审计中的 accepted_negative_duration_s 始终为 0。

异常中止可能留下部分派生文件；没有完整 `audit.json` 的输出目录不得使用，也不会被重试覆盖。应保留失败信息，在新的输出目录审查重试。

## 负例准入、屋/窝近音与训练接触

先固定筛选政策，再查看任何 KWS 检测。建议第一轮冻结全部符合条件的 Eval 会议为**新获取的自然中文开发评测集**，不用于训练/阈值调整；看过结果后继续称 development，不能追认 sealed qualification。

1. 逐会议检查所有说话人 TextGrid，包括重叠、非语音标签以及分段边界。官方人工转录是现成的候选证据；新人工听核优先处理命中、近音/歧义及注释覆盖边界，是否全量复听由后续准入方案决定，缺少新复听不等于永久阻塞。脚本按 tier 拼接中文文本仅作保守提示，不能证明没有目标词、误转录或发音近似。TextGrid 短格式/非 IntervalTier/未知格式直接失败，不能跳过未知 tier。
2. 对完整“你好小窝”“小窝小窝”出现，首轮预先规定**整会议排除主纯负集**并记录损失时长，或者转入另行标注的正例研究；不能把它赋 expected=[]。不按模型误报结果挑选、裁掉事件后拼接流。
3. “小屋”是近音 wu1，不等于“窝”wo1；不自动改字或改标签。“你好小屋”“小屋小屋”、单独“小窝”等保留为待人工确认的 no-wake/hard-negative 子组。确认真实词语且确实无完整目标后可按预定政策纳入开发负集，另报子组。模糊/听不清时整会议隔离，不能靠五类模型推断中文词义。脚本提示词表不是完整拼音/近音覆盖。
4. 跟现有所有训练/校准/挖掘来源比对：历史模型 1856 training PCM identities、已有 224 派生注入 WAV（112 原源家族；完整 domain 索引含 120 原源家族）、当前 385 文件的 transitive groups、所有近期训练 manifests。比较 source family / speaker / session 以及原始和派生 PCM，不把文件名不同或精确哈希不等当作充分独立证明。官方 Train/Eval speaker disjoint 仅说明其官方切分，不能证明本项目没有接触 Eval。当前自动比较已确认上述保留元数据中无精确原始/派生 WAV 或 mono PCM 匹配、无 AliMeeting/会话名称匹配，但**没有完成所有变换/片段及说话人级无接触证明**。
5. 只有按既定方案完成目标标注准入与训练来源证据审查、审计收据及入选集合冻结后，才为选中 whole meetings 写入 `expected: []` 的正式开发 references。记录审核者、政策版本、排除理由、全部保留/损失 sample_count。这个准备包不自动批准该步骤。

计时：`T = sum(每个唯一入选会议的实际 sample_count) / 16000 / 3600`。以解码后的音频时钟为分母，不用广告 4h、不累计说话人重叠时长、不乘 8 通道、不重复 near/far、不重复播放扩曝光。natural pauses 保留。若 4 小时零 FA，95% 单侧 Poisson 上界仍约 **0.749 次/小时**，不能证明低于 0.01 次/小时；不同域/会议和配对比较另报，不夸称部署覆盖。

## 冻结 C runner 接线（已按独立诊断脚本执行一次）

当前候选位置：`/workspace/scratch/6c2ef8b5a46e/kws-effect-iteration-v2-20261014/candidate`。`tools/kws_wav.c` 直接使用 C 默认配置，**没有运行时 JSON config 参数**；`configs/shipping.xiaowo.json` 是治理绑定，不会被该 CLI 读取。

已知冻结输入：

- `models/registry/model-749187ec1d66/xiaowo-model.kwm`，SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`
- 同目录 `xiaowo-keywords.kwk`，SHA256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`
- `configs/parameter-contract.json`，SHA256 `3171898c0c76367ca2fecfd31279ba2d25cfcd2029ebf1d9495337a5d5fbef06`
- 已从 P4 主线11b93abe/treeb8ae29构建并执行专用 `kws_wav`；完整构建/源码/二进制收据在 diagnostic-build/build-receipt.json。训练 head 不是当前 runner source identity。

每个会议新建一次进程，会议内部保持状态；默认 160 样本送入、400 窗/320 hop、内部能量 VAD、不加 AFE sidecar，不声称产品 AFE 后数据。模型/pack/decoder 阈值保持冻结。示例命令仅供未来使用：

```sh
"$RUNNER" "$MODEL" "$PACK" "$WAV" "$RECORDING_ID" \
  --block-samples 160 --stats-json "$OUT/$RECORDING_ID.stats.json" \
  > "$OUT/$RECORDING_ID.detections.jsonl" \
  2> "$OUT/$RECORDING_ID.stderr.log"
```

每条会话结束要求 exit=0，processed_samples 等于实测样本数，processed_frames 等于 `max(0, 1 + (N-400)//320)`，detections 等于合法 JSONL 行数，discontinuities/lost_samples/external_vad_frames 均为 0。空 JSONL 只有全部处理成功才代表零检测；校验 keyword_id∈{1,2}、时间在 [0,duration]、confidence∈[0,1] 且有限。绑定并复核执行前后 runner/model/pack/WAV 哈希。失败/不完整会议不得冒用计划分母，配对 arms 必须使用同一完整入选集合。

人工准入后的 references 每行结构：

```json
{"recording":"alimeeting-eval-实际会议ID","path":"mono/实际文件.wav","duration_s":1234.56,"expected":[],"session_id":"实际会议ID","source_id":"AliMeeting-Eval-实际会议ID","audio_sha256":"实际派生WAV的64位SHA256"}
```

`duration_s` 示例数值必须替换为实测值。既有 `eval/run_corpus.py --runner ... --model ... --keywords ... --references ... --audio-root ... --detections ... --provenance ... --corpus-identity ...` 能产出哈希 provenance，但**目前不转发 --stats-json/--block-samples**。因此后续最薄接线应选单次直接 runner + per-meeting stats/统一身份收据，或先小补 run_corpus 的 stats 转发；不要为了两套收据重复推理或假称现有一个调用两者都有。

```sh
python3 eval/score_events.py --references "$REFS" \
  --detections "$OUT/detections.jsonl" --pre-tolerance-ms 150 \
  --post-tolerance-ms 500 --summary "$OUT/summary.json" \
  --false-positives "$OUT/false-positives.jsonl"
```

报 `negative_recording_audio_hours`、`longest_negative_recording_seconds`、`negative_recording_false_accepts`、`negative_recording_far_per_hour`、`negative_recording_far_upper_95_per_hour`。负例集没有 FRR/命中延迟结论；scorer 的零值不能当成成功。`--require-matched-context` 要求正负同时存在，不适用于该负例独立开发集。输入/门限失败返回非零；有 summary 不等于通过。现有 event scorer 不验证 annotation_status，不能依赖它阻止误标。

当前只增加新的真实中文连续候选输入，还未增加已准入负曝光；不是独立 sealed qualification、最终产品 AFE 或板端证据。shipping_approved=false / recalibration_required=true 保留。
