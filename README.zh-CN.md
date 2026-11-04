# kws-pipeline

> **当前只有 KWS 技术研究任务。** 真实效果由两条完整中文唤醒词、
> 独立连续负例 FAR/FRR 置信界、词尾触发时延及 SSC305 资源实测决定。
> 研究入口：[技术研究执行规范](docs/RESEARCH_ONLY_EXECUTION.md)、
> [数据/模型单变量对照](.github/workflows/dataset-driven-iteration.yml) 和
> [冻结研究基线模型](configs/research-baseline.xiaowo.json)。
> 下文历史发布术语仅用于保留的旧实验溯源，不构成研究数据/训练门禁。


[English](README.md) | 简体中文

研究入口：[可复用模块、保留的实验结论与资格边界](research/README.md)

当前状态：[模型与工程摘要](docs/KWS_LANDING_STATUS.md#current-model-and-engineering-status)，核对日期 2026-10-29。[历史工程审查与旧闭环记录](docs/KWS_PRODUCTION_CLOSURE_2026-10-29.md) 列出已合入修复、新增回归、待完成的本地验收，以及新训练数据、候选部署、真人 AFE 和目标板资格的下一关。历史 SKU、FSMN B/OR 与 posterior-only C 的证据范围保留；尚无经过独立真人声学与 SSC305 真机测量的新研究模型。

当前归档交付：[完整公开归档、索引与可验证恢复](research/consolidation/ARCHIVE_DELIVERY_2026-10-09.md)。旧展开副本目标已取消；历史状态仍为 NOT_PUSHED。D20 数值失败、D90 尚未执行数值验证、fixed50 支持不足、旧模型 nightly 失败分别保留；真人与实机验证暂缓。

> **证据边界 —— 引用下方任何数字前请先读这一段。**
> 本仓库记录的 qualification 全部产生于**合成语料**，属于工程证据，不是商用声学结论。
> 仍有两类独立科研测量缺少真实输入：真实人声经过最终麦克风/结构/AFE，以及物理目标板 performance/soak —— 它们需要硬件和真人受试者。旧人工 GitHub Release 发布工作流 `real-human-qualification.yml`、`target-dut-qualification.yml` 已从活动 Actions 退役；真人声学、AFE、目标设备资源的独立评分工具仍保留，需要真实数据/设备及合法授权方可运行。工具存在不等于真人/SSC305 已实测。历史 `configs/shipping.xiaowo.json` 不再决定科研准入。详见「验证边界」。

`kws-pipeline` 是用于**低算力嵌入式 Linux / RTOS 研究**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近 CPU 预算。

冻结的历史合成语音模型对照**严格只覆盖两个四字中文唤醒词**：

- **你好小窝** — `ni3 hao3 xiao3 wo1`
- **小窝小窝** — `xiao3 wo1 xiao3 wo1`

独立两字 **“小窝”只作部分短语/负例**，不是完整唤醒正例。冻结模型对照使用专用 5-token 词表：`<blk> / ni3 / hao3 / xiao3 / wo1`。框架本身支持更大的拼音词表和可更新 KWKP，但**没有独立证据证明任意中文短语都可以 L0 不训练直接替换**。

典型研究信号链：

```text
最终麦克风 / 双麦
 -> BF / AEC / RES / NS / AGC
 -> PCM16 16 kHz
 -> 25 ms / 20 ms hop log-mel 或 PCEN-lite
 -> int8 权重 tiny streaming RNN
 -> 拼音 token logits
 -> 共享前缀关键词 Trie
 -> CTC 重复 token + 前缀冲突仲裁
 -> speech / threshold / refractory 门控
 -> wake event
```

<a id="当前-engineering-qualified-模型"></a>

## 历史 Engineering-Qualified 模型

历史 GitHub Release 归档 [`model-749187ec1d66`](https://github.com/jiying2007/kws-pipeline/releases/tag/model-749187ec1d66) 保留以下旧来源，**当前科研对照从 Git 模型注册表读取而非下载 Release**：

- model-training run `34134789576`；
- exact trained HEAD `749187ec1d6662658f06aa9c76d47fde835968db`；
- `model.kwm` SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`；
- 上述两条冻结研究关键词；
- untouched synthetic qualification seed `271838`，现已**消费并永久冻结**；
- qualification：`256/256`、`0 FR`、`0 FA`；
- strict robustness 全通过；
- continuous synthetic hard-negative FAR：`0 FA` 且 negative manifest 全覆盖。

以上是**历史 Release qualification 结果**。后续同一旧模型的
[nightly run 37860130019](https://github.com/jiying2007/kws-pipeline/actions/runs/37860130019)
已失败：**8 小时内 2 次误接受**，upper95 **0.786974 FA/h**（门槛为 0 次误接受且
upper95 ≤ 0.40）。这不是 D20/D90 的结果；原始失败记录不构成任何科研模型通过或发布决定。
见[本次审计](research/consolidation/CI_AUDIT_2026-10-09.md)。另见[2026-10-14 历史入口](docs/KWS_LANDING_STATUS.md#2026-10-14-current-source-entry)：后续 nightly `38482297930` 仍失败，绑定其历史执行源码，并非对当前 main 的新增声学评估。


仓库仍明确保留 `models/registry/**` Git 追踪例外，冻结模型资产按 SHA-256 独立校验。
冻结的两词研究对照已迁到 [`configs/research/two-keyword-synthetic-comparator.json`](configs/research/two-keyword-synthetic-comparator.json)，保留历史阈值、模型/关键词哈希和需重新校准的事实；不包含商用发布授权，也不能代替真实人声、最终 AFE 或 SSC305 实测。旧配置只在 Git 历史中供审计。

当前源码 contract 另行记录 `recalibration_required=true`；历史资格不构成变更后
运行时源码的资格。各证据范围见[研究与历史状态](docs/KWS_LANDING_STATUS.md)。

### 冻结研究模型与关键词资产验证

[Git 注册表](models/registry/model-749187ec1d66/) 保留模型、检查点、
关键词、来源及不可变 SHA256。该旧模型仅是合成数据的**历史对照**，
不是已通过独立真人测试的当前优选模型。验证时只读取 Git 资产，不查询或发布 Release：

```bash
python3 tools/verify_model_registry.py \
  --registry-dir models/registry/model-749187ec1d66
python3 tools/verify_research_model_assets.py \
  --assets-dir models/registry/model-749187ec1d66 \
  --registry-tag model-749187ec1d66
```

校验器同时检查四项源代码固定哈希及历史 `MODEL_SHA256SUMS`，并拒绝
资产替换、哈希清单重算后伪造通过或遗漏固定资产。离线负例仍保留在
`tests/test_model_release_pin.py`（历史文件名）。

## Runtime / 技术研究特性

- 实时库只依赖 C11 + libm；PyTorch、`pypinyin` 仅在线下工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。这一点是被门禁强制的，而不是靠约定：`tools/check_runtime_purity.py` 把 `libkws_pipeline.a` 的未定义符号集与显式白名单（内部 `kws_*` 调用、`mem*`、ISO C 数学函数、stack protector 与 ARMv7 EABI 除法辅助函数）比对，新增外部依赖会直接让 CI 失败，而不是静静通过评审。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **KWSP ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、vocabulary fingerprint、frontend identity。
- **KWKP ABI v3**：每关键词 threshold、trailing blank、priority、`immediate/longest/grace` prefix policy。
- 相邻重复 acoustic token 遵守 CTC blank-separated 结构语义。
- 共享前缀仲裁不依赖 TSV 行顺序：`immediate` 依次比较 priority、depth、confidence；`longest`/`grace` 依次比较 depth、priority、confidence。以上全部相同时，选择较小的无符号 keyword ID。每帧先完成 pending 仲裁；若仍由原候选胜出且仅 confidence 提升，保留其已等待的 grace 帧数。
- `kws_engine_notify_discontinuity()` 在 XRUN、route、clock、suspend/resume 后清理 partial acoustic state。
- 外部 AFE metadata 和 runtime telemetry 均版本化、有边界，可用于可复现研究与嵌入式诊断。

设计文档中提到的约 420-token / 约 26-KB 几何是**框架容量设计项**，不是当前已 qualification 的 5-token 正式模型身份。

## 构建与安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

SDK 支持 CMake package 和 `pkg-config`。

## 冻结研究关键词包与定制边界

历史冻结研究 TSV 精确为：

```text
id  text      threshold  explicit-pinyin
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝小窝  0.55       xiao3 wo1 xiao3 wo1
```

编译：

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

L0 keyword-pack 实验只适用于**全部 acoustic token 已存在于加载模型词表**的场景；变更后的模型/关键词组合仍需独立声学评测。该冻结五 token 对照只覆盖上述两条完整四字词。新拼音 token 需要更大词表或重新训练、再进行独立测量，不能凭兼容性声明“任意中文零训练换词”。

详见 `docs/CUSTOMIZATION.md`。

## 自训练、自验证与自动迭代

正式离线闭环：

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.torch-domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

正式链使用真实 C runtime 评估，并固定执行四轮训练、calibration/test gate、hard-negative replay、adaptive domain curriculum、untouched qualification、robustness slice、continuous FAR。synthetic domain 覆盖名义 0.3–5.0 m、azimuth、RT60、SNR、white/fan/motor/media 噪声以及 playback/AEC-residual proxy。

正式 qualification seed `271838` 已在当前 Model Release 中暴露并验收，之后**绝不允许再次重试**。未来模型候选的有效 seed 与退役列表以 `configs/training/xiaowo.torch-domain.json` 为准；维护或重新资格验证不得复用退役 seed。

### Nightly 与 formal qualification 完全分离

`.github/workflows/far-nightly.yml` 不重新训练或发布模型。Nightly：

1. 从 Git 中的 `models/registry/model-749187ec1d66/` 读取冻结历史对照；
2. 根据 `configs/research-baseline.xiaowo.json` 和注册表校验模型、关键词、训练来源 SHA-256；
3. 使用独立的 `nightly-frozen-model-v1` synthetic negative namespace；
4. 执行四个 long-FAR 负例分片与置信界计算；
5. 只输出科研回归证据，不下载 GitHub Release，也不把历史失败变成 PASS。

`configs/nightly.xiaowo-frozen-model.json` 中不存在 formal qualification seed 或 formal FAR holdout namespace，因此 nightly 永远不能把 271838 再次消费，也不能冒充 fresh qualification。

## 最终 AFE 接入契约

当前正式 synthetic training 仍使用 built-in proxy AFE，但 renderer 已实现 fail-closed `command` AFE backend，用于后续接最终 `audio-pipeline`。它精确绑定：

- 实际 invoked executable SHA256；
- command template SHA256；
- ordered AFE config-file bundle SHA256；
- 双路输入 WAV SHA 和输出 WAV SHA；
- result sidecar SHA；
- `latency_samples`；
- 可选 pipeline SHA / source SHA / toolchain identity。

真实人声与目标板实验必须独立绑定合法授权的数据、实际 AFE 程序与配置、设备身份和测量来源；目前缺乏的声学/SSC305 结果只能标记为 NOT_RUN/BLOCKED，不能根据历史合成结果视为通过。

## 不可变训练环境

科研训练应使用 `name@sha256:<digest>` 固定 OCI 镜像。`training/Dockerfile` 不在 build 时浮动安装依赖；`training/build_container.py` 校验 immutable base 并记录 build receipt。真实 `torch_ctc` integration workflow 使用仓库变量 **`KWS_TRAINING_IMAGE`**（或手动 `training_image` input），只接受 digest-pinned image。

## 连续评测与统计边界

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references qualification/references.jsonl \
  --audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --provenance qualification/detections.provenance.json
```

Evaluation provenance 会逐 WAV 绑定真实字节并从 decoded frame 校验 duration。synthetic/nightly long-FAR 只属于严格 regression signal，不是量产真实 FAR/hour 结论；观测到 0 次 FA 也必须按统计置信上界解释，而不是声称真实概率等于 0。

## 真人数据 contract — 下一阶段再执行

真实数据引入后，最终 references 必须直接进入 speaker/session/source 隔离审计：

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=qualification/references.jsonl \
  --audio-root qualification=qualification/audio \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report qualification/dataset-audit.json \
  --fail-within-split
```

任何已经用于 tuning、hard-negative mining 或 false-reject replay 的录音，不能再作为 unbiased final qualification evidence。

## Product-board 证据合同 — 框架已准备，物理执行后置

后续实机阶段先让 collector 监督**真正的被测进程**：

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

然后冻结 raw evidence 并组装 target identity：

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --soc cortex-a32 \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/board-summary.json \
  --power-raw qualification/power.csv \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --board-runner qualification/kws_board_bench.target \
  --model release/base.kwm \
  --keyword-pack release/xiaowo.kwk \
  --board-audio qualification/board-audio.wav \
  --sku product-sku-a \
  --source-sha "$(git rev-parse HEAD)" \
  --builder-id qualification-builder-01 \
  --dut-id product-dut-01 \
  --collector-id qualification-station-01 \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`builder-id` 与 `dut-id` 必须不同。CPU/RSS/thermal/soak 会从保留 raw samples 独立重算；power 需要保留仪器原始数据和 calibration identity。详见 `docs/TARGET_EVIDENCE.md`。

## 制品绑定的最终 qualification

后续最终真人/实机 evidence 仍走同一套 artifact-bound contract，不接受手工总结替代：

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest data/train.jsonl \
  --dataset-audit qualification/dataset-audit.json \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --eval-audio-root qualification/audio \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --evidence-collector tools/collect_target_evidence.py \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/board-summary.json \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --sku product-sku-a \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --board-audio qualification/board-audio.wav \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

研究门禁保留 model ABI v2、keyword-pack ABI v3、frontend-spec v2、语料身份、数据审核和目标板来源完整性；真实人声/AFE、物理设备时延与资源测量需独立开展，不能转化为产品资格或伪造 PASS。

## 验证边界

CI 当前证明软件合同和 deterministic/synthetic regression：GCC/Clang、CTest、static analysis、coverage、ASan/UBSan、libFuzzer、Cortex-A32 cross-build（其 **CTest 套件实际在 `qemu-arm-static` 下执行**）以及 hosted-vs-Cortex-A32 的 int8 kernel 数值对拍门禁、frontend/decoder、corpus identity、自训练闭环、robustness/FAR regression、runtime-soak/target-evidence schema、SDK reproducibility、model supply-chain、冻结研究模型资产 SHA256 pin。

它**不宣称真实商用声学和物理板级 qualification 已完成**。完成本轮 non-real-data 工程闭环后，产品证据阶段只剩两步：① 真实普通话经过最终麦克风/结构/AFE；② 物理目标板 performance/soak 并最终批准完整 deployment tuple。这两步在本仓库**均无法执行**：都需要硬件、真人受试者与 self-hosted runner，这两条旧的商用 Release 发布工作流均已退役，独立真机与真人实验须有真实输入和合法授权。

详见 `docs/README.md`、`docs/CUSTOMIZATION.md`、`docs/RELEASE_QUALIFICATION.md`、`docs/TARGET_EVIDENCE.md`、`docs/CORPUS_IDENTITY.md`、`docs/AUDIO_DISCONTINUITY.md` 和 Issue #2。

## License

Apache-2.0，见 `LICENSE`。
