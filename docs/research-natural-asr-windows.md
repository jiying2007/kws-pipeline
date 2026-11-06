# 自然长流的有界 ASR 窗口输入

`tools/prepare_natural_asr_windows.py` 将已声明的自然录音 sample 区间准备为
现有 `active-asr-inputs-v1` 与 `active-asr-source-map-v2`。它只读完整父音频，
复核身份，再写入一个新的私有输入包；不下载、不加载模型、不运行队列，
不生成标签、训练权重或已确认自然负时长。冻结的 TTS 专用
`active_asr_batch.py prepare` 入口没有修改，也没有借用参考人身份。

## 当前两窗声明

[`natural-alimeeting-eval-tail2-v1.json`](../configs/research/natural-alimeeting-eval-tail2-v1.json)
明确为 `declared_not_materialized`，保留了建立声明时父音频尚未恢复的描述。
随后，完整归档已按历史 SHA256 复核并恢复全部 8 会话，合计 242,211,440
样本；原 8 份 TextGrid 也逐文件匹配原 intake 哈希。恢复收据的状态仍为
`historical-identity-restored-not-negative-admitted`，没有新增负例准入。
这些恢复结果不改写冻结声明；真实窗口是否生成，以私有输入包的独立
`summary.json` 为准。工具在全部父哈希检查成功前不会产生输入包。
本变更只保留来源哈希与区间，不发布音频或转录正文。

来源是 AliMeeting **Eval** 的 `R8001_M8004`，固定零基 channel 0，保留
development 角色。选择依据是原注释末端覆盖缺口，不是 KWS 是否触发。
原始身份来自保留的
[`audit/audit.json`](../research/alimeeting-continuous-intake-v1/audit/audit.json)，
其 SHA256 为 `95b2f7b6abd409fe0a17e597f49a01aa0bbd97f86f4ec807762c609fd5a0aca3`。
数据出处与 CC BY-SA 4.0 许可见
[OpenSLR119](https://www.openslr.org/119/)；历史获取与范围说明见
[`RESULT.md`](../research/alimeeting-continuous-intake-v1/RESULT.md)。

| 窗 | 半开 sample 区间，16 kHz | 秒区间 | 窗输入时长 |
| --- | --- | --- | ---: |
| A | `[24335248, 24815248)` | `[1520.953, 1550.953)` | 30 s |
| B | `[24735248, 25181600)` | `[1545.953, 1573.850)` | 27.897 s |

两个窗的处理量为 **926,352 样本 / 57.897 秒**；按父 PCM 求并集为
**846,352 样本 / 52.897 秒**；重复覆盖 **80,000 样本 / 5 秒**。
这些是声明覆盖量，不是模型执行量，也不是自然 FAR 的负例分母。
这 52.897 秒不能代表其余会话、全部 4.205 小时或说话人独立性。

该声明文件的精确 SHA256 为
`56331de614464626b6bf2a53df0138ed456e6e97b239d126b18ea9e02b2deb83`。
恢复父音频不会改写声明；准备状态由新输入包的独立 `summary.json` 记录。

## 身份、切片与时间规则

- 只接受完整 classic PCM16 mono 16 kHz RIFF。每个父文件先核验完整 WAV
  SHA256、解码 PCM SHA256 与样本数，之后才切片。选窗外发生任何字节变化
  也会失败。已有 8ch 原源应通过原 intake 的固定 channel-0 转换恢复，不能
  用重采样或新容器替代哈希已锁定的 mono 文件。
- sample 下标必须是整数，使用 `[start_sample, end_sample)`；拒绝浮点、布尔、
  空窗、反向、越界、重复区间、未知父身份。最多 8 个父录音、64 个窗，
  每窗最多 30 秒，每个父 WAV 最多 256 MiB。不会自动截短、补零或选取有声部分。
- 沿用 `research_acoustic_asr.decode_pcm16` 的 `int16 / 32768` 精确 float32
  转换。窗口 WAV 的 PCM 与父切片逐字节一致；输出中的 `original_pcm` 明确为
  `<i2`，识别输入 `derived_pcm` 明确为 `<f4`。零值和低音量不作为拒绝条件。
- 每个数据集会话只允许一个父 mono 录音，防止同会话多通道重复计时；拒绝
  相同父 PCM 的别名。`source_family_id` 在声明内必须保持唯一 split，
  窗继承父 session/family/split，没有单独修改角色的入口。只支持 train 和
  development，拒绝 test/heldout。来源家族的真实性仍由上游证据负责。
- 所有窗口输出连续的 `clip-000` 等 opaque ID。识别输入和 WAV 容器没有
  会话名、split、来源文本或目标词；这些只在独立绑定中。全局 `source_id`
  由父 PCM SHA256、采样率、半开 sample 区间计算，换窗名称或重排 clip
  不会产生新的源身份。
- 并集只在同一父 PCM 的 sample 时间轴上计算，再汇总各父量。它不证明不同
  父录音的声学独立性。父记录的原 archive、原 8ch WAV、TextGrid 和 intake
  收据哈希作为上游声明保留；本工具没有读取或重新验证这些上游实体。

## 准备与独立回读

使用项目已固定的 CPython **3.12.16**。以下 `KWS_NATURAL_PYTHON`、
`KWS_NATURAL_AUDIO_ROOT`、`KWS_NATURAL_OUTPUT` 应设置为实际解释器、已恢复
`audit` 目录及一个尚不存在的私有目录。输出必须在 Git 工作树之外。

```sh
"$KWS_NATURAL_PYTHON" -I -S -B tools/prepare_natural_asr_windows.py plan \
  --declaration configs/research/natural-alimeeting-eval-tail2-v1.json

"$KWS_NATURAL_PYTHON" -I -S -B tools/prepare_natural_asr_windows.py prepare \
  --declaration configs/research/natural-alimeeting-eval-tail2-v1.json \
  --declaration-sha256 56331de614464626b6bf2a53df0138ed456e6e97b239d126b18ea9e02b2deb83 \
  --audio-root "$KWS_NATURAL_AUDIO_ROOT" \
  --output "$KWS_NATURAL_OUTPUT"

"$KWS_NATURAL_PYTHON" -I -S -B tools/prepare_natural_asr_windows.py verify \
  --bundle "$KWS_NATURAL_OUTPUT" \
  --declaration-sha256 56331de614464626b6bf2a53df0138ed456e6e97b239d126b18ea9e02b2deb83 \
  --audio-root "$KWS_NATURAL_AUDIO_ROOT"
```

`plan` 只校验声明，不读取音频或写文件。`prepare` 在全部父身份检查成功后
才创建目录，拒绝覆盖已有目录，完成时进行一次完整回读。`verify` 重新从
全部原父音频构造预期窗口、并集和来源绑定，核对每个文件及完整文件名单；
不改写已有包。输入声明的外部 SHA256 是回读锚点，不能由被改写的包自行替换。
准备源码及调用到的既有转换/验证源码也保留身份；回读应使用同一源码版本。

输出包括 `inputs.json`、`source-map.json`、`original/*.wav`、`pcm16k/*.f32`、
`window-audio.zip`、原始 `declaration.json`、准备源码和
`natural-window-bindings.json`、`summary.json`。ASR manifest 中
`source_archive_sha256` 绑定实际保存的 **窗口音频 ZIP**；它不是官方 AliMeeting
归档的哈希，后者只在父来源 provenance 中保留。

## 接入既有识别队列

只有输入包回读成功后，才可使用原入口声明识别请求：

```sh
"$KWS_ASR_PYTHON" -I -S -B tools/active_asr_selective.py declare \
  --runtime "$KWS_ASR_RUNTIME" \
  --inputs "$KWS_NATURAL_OUTPUT/inputs.json" \
  --source-map "$KWS_NATURAL_OUTPUT/source-map.json" \
  --policy configs/research/selective-asr-v2.json \
  --output "$KWS_NATURAL_ASR_OUTPUT" \
  --request "$KWS_NATURAL_ASR_REQUEST" \
  --threads 4 --budget-seconds 570
```

该命令只创建请求。后续执行继续交给现有 `research_queue.py`，绑定这个请求
的精确 SHA256、单重模型锁和 10 分钟任务上限，不能与已有声学/C1 任务抢占。
本工具没有启动该请求。两个自然窗保留 `quality_uncertain=true`，按现有
`source_quality_uncertain` 路由，每窗会请求 Qwen、SenseVoice、Whisper，最多
**6 次原生识别**。重新准备输入包不会产生重试授权或清除已有 claim。

既有一致样本的 20% 抽查规则保持原样：对
`kws-d1-whisper-agreement-audit-v2|20261105|plan_sha256|source_id` 的 UTF-8
字节取 SHA256，只有 `int(hash, 16) * 5 < 2**256` 才入选。它在没有 mandatory
reason 时生效，不是每 5 条固定挑 1 条。冻结本声明后进行的纯哈希核对显示，
两个窗的 sample hash 分别以 `26ff3e05`、`1fe7389d` 开头，均命中该抽样规则；
但本批实际第三路原因仍为 `source_quality_uncertain`，不是假定主路已经一致。
没有为了得到某种抽样数量调整声明、种子或源身份。

未来转写、边界拼接及污染审查仍需保留原生结果。现有识别配置没有词级对齐，
不能从窗口全文推造精确目标词时间戳；机器一致也不自动产生自然负标签。

## 离线验证

`tests/test_prepare_natural_asr_windows.py` 使用人工合成的极短 PCM 测试切片、
全父身份、并集、跨 split 家族冲突、重复父/会话、越界和别名路径、源身份稳定性、
包篡改拒绝、创建一次及只读回放。另将两窗声明与仓库保留的原 intake 收据核对。
所有测试只用标准库，不下载、导入 ML 依赖或调用模型。

```sh
python3 -I -S -B tests/test_prepare_natural_asr_windows.py
python3 -I -S -B -O tests/test_prepare_natural_asr_windows.py
python3 -I -S -B -OO tests/test_prepare_natural_asr_windows.py
```
