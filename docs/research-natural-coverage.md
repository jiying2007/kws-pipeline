# 自然语音固定选窗与跨包覆盖

`tools/research_natural_coverage.py` 处理两个独立的元数据任务：生成固定的
自然录音窗口声明，以及汇总已验证输入和已有封存识别证据的覆盖。
它不识别音频、不训练、不补写 ASR seal，也不准入自然 FAR 负例。

## 固定选择规则

本次规则只读取父音频样本数与既有窗口声明。对每个长度为 `N` 的父 PCM，
令 `anchor = floor((N - 880000) * i / 5)`，`i = 1, 2, 3, 4`。每组两个窗口为：

- A：`[anchor, anchor + 480000)`；
- B：`[anchor + 400000, anchor + 880000)`。

在 16 kHz 下，每窗 30 秒、组内重叠 5 秒、组并集 55 秒。父会话按字典序排列，
每父的前两组和后两组各形成一个四窗分片。8 个父录音得到 16 个分片、64 窗，
输入总时长 1,920 秒，父内并集 1,760 秒，重复覆盖 160 秒。

所有新组必须完整位于父录音内，不能与任何既有声明区间相交，组间也不能
重叠。冲突时拒绝整份计划，不自动移窗、缩短或换样本。既有声明只作为
排除区间；声明本身不能证明旧窗口已经识别。选择不读取 ASR 文本、KWS
触发、能量或新实验结果。原始来源、channel、family 与 split 不改变。

使用固定 CPython 3.12.16，以重复参数传入每份既有声明及外部 SHA-256：

```sh
"$KWS_NATURAL_PYTHON" -I -S -B tools/research_natural_coverage.py plan \
  --prior-declaration /private/prior-tail2.json \
  --prior-declaration-sha256 ACTUAL_TAIL2_SHA256 \
  --prior-declaration /private/prior-tail7-a.json \
  --prior-declaration-sha256 ACTUAL_TAIL7_A_SHA256 \
  --prior-declaration /private/prior-tail7-b.json \
  --prior-declaration-sha256 ACTUAL_TAIL7_B_SHA256 \
  --output /private/new-natural-plan
```

输出目录必须不存在且位于 Git 工作树之外。目录包含各分片声明、合并声明、
来源哈希和 `plan.json`。后续仍由原 `prepare_natural_asr_windows.py` 执行父音频
验证与切片，并由既有持久队列执行已冻结的 ASR 请求；此命令只创建声明。

## prepared、observed 与 weak quorum 分开计数

汇总请求为 `kws-natural-coverage-request-v1`，包含四个字段：`schema`、
`prepared`、`views`、`verifier`。

`prepared` 每项精确包含 `bundle`、`audio_root`、`declaration_sha256`。工具调用
已有自然准备验证器，从完整父 WAV/PCM 重新构造输入并核验文件名单与哈希，
不会只相信 `summary.json`。一份已准备输入包不必有识别结果。

`views` 每项精确包含 `path` 和 `manifest_sha256`，指向 kws-data 的
`automation/active_natural_asr_weak_view.py` 产生的自然 evidence view。
`verifier` 包含该模块的绝对 `path` 和完整 `source_files_sha256` 映射；映射必须
与其 `PRODUCER_FILES` 相同。导入前、导入后及全部复核后检查这些源文件，
然后调用其只读 `verify_view(package, expected_manifest_sha256=...)`。
如果没有 views，verifier 必须为 null，不加载 data 模块。

每个 observed batch 必须已有最终 seal、输入/请求/清单身份和完整的三路原生
结果。view 验证器重放原生 claim、native、result、路由与封存清单，任何未完成
或未请求结果都不能当成 observation。汇总另将每行的原始 source、父 PCM、
会话、family、split、opaque clip ID 和 sample 区间绑定回已验证输入包；不能
靠填一个 `status=complete` 或只有 claim 就取得 observed 覆盖。

输出三个独立集合的父内样本并集：

| 集合 | 含义 |
| --- | --- |
| `prepared` | 已从完整父音频核验的输入窗口 |
| `observed` | 已有完整封存三路原生结果的窗口，可包含不可用或不一致文本 |
| `weak_fulltext_quorum` | natural evidence view 明确给出完整全文 quorum 和 CTC mask 的机器弱候选 |

一个窗口无论三路识别还是多个关键词，都只在相应集合计一次。交叠窗口按
同一父 PCM 的半开 sample 区间求并集；不同父时间轴分开处理，不宣称声学
或说话人独立。重复输入包、重复全局 source、跨 view 重复结果、父身份别名、
同会话第二个 channel、跨 split family 冲突均拒绝。相同父身份在多个分片中
正常复用；不同区间可相交，但重复时长不会被计入并集。

原生结果完整不保证文本可用，机器全文 quorum 不证明 acoustic completeness。
所有覆盖均不是已确认负例时长、事件标签或自然 FAR 分母；`accepted_negative_seconds`
固定为 0。工具不产生训练导入 manifest，也不把 unknown 自然窗变为空 CTC 目标。

```sh
"$KWS_NATURAL_PYTHON" -I -S -B tools/research_natural_coverage.py summarize \
  --request /private/coverage-request.json \
  --request-sha256 ACTUAL_REQUEST_SHA256
```

该命令只读输入并把报告写到 stdout，不调用可能补写 seal 的
`active_asr_selective.seal_or_verify`。

## 离线反例

`tests/test_research_natural_coverage.py` 以极短人工 PCM 建真实自然输入包，覆盖
重叠去重、整父篡改、未识别输入、缺失/未请求原生结果、重复/别名/多通道/角色
冲突、固定规则冲突拒绝及只读性。数据 view 验证器在这些单元测试中使用明确
的 double；原生 ASR 回放与机器弱标签政策由 data 侧契约测试及真实证据复核负责。
测试不运行模型，不产生任何真实标签或效果结论。
