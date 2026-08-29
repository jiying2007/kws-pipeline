# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近 CPU 预算。它支持“**你好小窝**”“**小窝**”“**小窝小窝**”等可配置中文唤醒词，并设计为直接消费 [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline) 等 BF/AEC/RES/NS/AGC 前端输出的 16 kHz 单声道 PCM16。

```text
PCM16 16 kHz
 -> 25 ms / 20 ms hop log-mel 或 PCEN-lite
 -> int8 权重 tiny streaming RNN
 -> 拼音 token logits
 -> 共享前缀关键词 Trie
 -> CTC 重复 token + 前缀冲突仲裁
 -> speech / threshold / refractory 门控
 -> wake event
```

普通短语变化属于 L0 关键词包更新，不需要重新训练模型；现场 FAR/FRR 不达标时，再进入阈值校准、hard-negative/false-reject 回灌、`--head-only` 浅定制和 domain-aware 迭代。

## v0.3 软件终态硬化

`v0.3.x` 不改变设备端 **KWSP Model ABI v2** 与 **KWKP Keyword Pack ABI v3**，但对产品资格证据做硬切升级：

- 训练 checkpoint/model provenance 逐个绑定真实训练 WAV 的 file SHA256、decoded PCM SHA256 与 frame 数；
- evaluation provenance 逐个绑定真实 qualification WAV，并强制 `references.duration_s` 与真实 WAV 时长一致；
- 最终真人 qualification 必须有 clean dataset audit，并隔离 speaker/session/source；
- qualification manifest schema v2 会重新读取原始训练/评测音频，不再只相信 JSON 自报 hash；
- target resource evidence 来自被测进程的 runtime-soak 原始轨迹、功耗/其他 raw measurement 以及精确 evidence collector；
- 新增 `kws_engine_notify_discontinuity()`，在 XRUN、route、clock、suspend/resume 后清理 partial acoustic state；
- CI 增加 Clang static analyzer、C coverage、test inventory 和两次独立 SDK build 的 byte-for-byte 一致性门禁；
- 可选 production `torch_ctc` integration workflow 在 digest-pinned 训练镜像内验证真实 train/export/runtime 链路。

这些能力让**软件证据链**做到 byte-bound、可审计；但仍不能替代真实普通话人声、最终麦克风/结构/audio-pipeline 和物理 Cortex-A32 数据，Issue #2 继续作为最终产品资格门。

## 产品特性

- 实时库设备端只依赖 C11 + libm；PyTorch、`pypinyin` 只存在于离线工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **KWSP ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、词表 fingerprint 和 frontend identity。
- **KWKP ABI v3**：每个关键词独立 threshold、trailing blank、priority 和 `immediate/longest/grace` prefix policy。
- 相邻重复 token 遵守 CTC blank-separated 结构语义。
- “小窝 / 小窝小窝”这类共享前缀通过 priority、trailing blank 和 grace 确定性仲裁。
- L0 换关键词、L1 阈值/回灌校准、L2 `--head-only` 浅定制。
- 默认 32 feature / 48 hidden / 约 420 token 的 dense 声学计算约 **1.2 MMAC/s**，模型权重与 bias 约 **26 KB**；这是设计预算，不是 Cortex-A32 实板测量。

## 构建与安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

SDK 支持 CMake package 和 `pkg-config`。

## 自定义唤醒词

正式产品建议固定显式拼音：

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

```text
1	你好小窝	0.55	ni3 hao3 xiao3 wo1
2	小窝	0.55	xiao3 wo1	1	10	grace	3
3	小窝小窝	0.55	xiao3 wo1 xiao3 wo1	1	20	longest
```

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

## 训练与数据隔离

`train_ctc.py` 支持 TSV（`WAV<TAB>token_ids`）和 schema-rich JSONL。真人发布数据建议使用带 speaker/session/source/room/device 元数据的 JSONL。

正式发布前，dataset audit 必须直接覆盖随后传给 `qualification_manifest.py` 的**同一个最终 `references.jsonl`**：

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

v0.3 qualification 会验证 audit 覆盖精确的训练 manifests 和最终 held-out references，而不是另一个“看起来等价”的 qualification manifest。

训练与导出：

```bash
python3 training/train_ctc.py \
  --manifest data/train.jsonl \
  --tokens keywords/tokens.zh.txt \
  --frontend logmel \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

checkpoint 会记录 canonical real training-corpus identity；model provenance schema v3 把该身份带入正式模型 lineage。最终 qualification held-out 不得回灌后继续冒充 unbiased evidence。

## 不可变训练环境

正式训练建议使用预构建、通过 OCI digest 固定的镜像。`training/Dockerfile` 要求 base reference 包含 `@sha256:<digest>`，自身不再联网安装/升级依赖。使用 `training/build_container.py` 构建 wrapper，最终镜像 digest 通过 `KWS_TRAINING_IMAGE_DIGEST` 写入 shipping checkpoint。

配置仓库变量 `KWS_TRAINING_BASE_IMAGE` 后，`.github/workflows/training-integration.yml` 可周期执行真实 `torch_ctc` 集成测试。

## Domain-aware 自验证闭环

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

示例覆盖名义 **0.3–5.0 m**、azimuth、RT60、SNR、white/fan/motor/media 噪声和 playback/AEC residual，完整 rendered utterance 走真实 C runtime。

这只能称为 **synthetic-domain evidence**，不能表示真实 3–5 米已经量产通过。

## 连续音频与断流

正常链路可直接以 160 sample / 10 ms block 调用 `kws_engine_accept_pcm16()`。出现 XRUN、route、clock 或 suspend/resume 后必须通知引擎：

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

这会阻止断流前后的声学状态被错误拼接。

连续评测：

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

v0.3 evaluation provenance schema v2 逐文件绑定真实 WAV，reference `duration_s` 必须与 WAV 实际时长一致。synthetic/nightly long-FAR 仍只提供 regression，不替代真实 FAR/hour。

## 物理目标板证据

先让 runtime-soak collector 监督**真正的被测进程**：

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

该原始轨迹从 child process 采集 CPU/RSS，并记录 thermal/soak 状态。之后组装 target evidence：

```bash
python3 tools/collect_target_evidence.py \
  --output qualification/evidence.json \
  --target product-sku-a \
  --board-revision A \
  --toolchain arm-linux-gnueabihf-gcc-... \
  --compiler-flags '-O3 -mcpu=cortex-a32 ...' \
  --audio-frontend audio-pipeline-vX \
  --audio-frontend-sha256 <sha256> \
  --runtime-soak qualification/runtime-soak.json \
  --stack-high-water-bytes <measured> \
  --average-power-mw <measured> \
  --raw-evidence qualification/stack-watermark.txt \
  --power-raw qualification/power.csv \
  --instrument-id <meter-id> \
  --calibration-id <calibration-id>
```

`soak_hours`、CPU、RSS、max temperature 都从保留的 runtime-soak 文件导入，不再接受命令行自报。stack high-water 仍属于产品 harness 测量，应该保留 raw evidence；功耗必须绑定原始仪表导出、instrument ID 和 calibration ID。

## 制品绑定的发布认证

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
  --evidence-collector qualification/collect_target_evidence.py \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Gate 强制 model ABI v2、keyword-pack ABI v3、frontend-spec v2、corpus byte identity、dataset-audit coverage、target raw-evidence identity，以及 SKU FAR/FRR/latency/CPU/RSS/stack/soak/温度/功耗门禁。

## 验证边界

CI 覆盖 GCC/Clang、CTest、Clang static analysis、C coverage、ASan/UBSan、libFuzzer、Cortex-A32 cross-build、双 frontend parity、decoder/prefix、数据泄漏、corpus byte identity、domain-aware loop、streaming long-FAR、v0.3 qualification、runtime-soak/target-evidence collector、两次独立 SDK build reproducibility 和 clean SDK consumer。

这些证明的是**软件合同和 deterministic/synthetic regression**。真正量产仍必须使用真实中文模型、独立真人/设备 held-out corpus、最终双麦/结构/audio-pipeline 和物理 Cortex-A32 证据。Issue #2 保持 open，直到这些真实证据完整绑定并通过 SKU policy。

详见 `docs/RELEASE_QUALIFICATION.md`、`docs/TARGET_EVIDENCE.md`、`docs/CORPUS_IDENTITY.md`、`docs/AUDIO_DISCONTINUITY.md`。

## License

Apache-2.0，见 `LICENSE`。
