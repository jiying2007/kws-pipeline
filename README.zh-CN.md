# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近 CPU 预算。它支持“**你好小窝**”“**小窝**”“**小窝小窝**”等可配置中文唤醒词，并设计为直接消费 [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline) 等 BF/AEC/RES/NS/AGC 前端输出的 16 kHz 单声道 PCM16。

核心采用**开放 token KWS**，不是“一个唤醒词一个二分类模型”：

```text
PCM16 16 kHz
 -> 25 ms / 20 ms hop frontend（log-mel 或 PCEN-lite）
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

- 训练 checkpoint 与 model provenance 会逐个绑定真实训练 WAV 的 file SHA256、decoded PCM SHA256、frames/duration 和 canonical corpus identity；
- evaluation provenance 逐个绑定真实 qualification WAV，并强制 `references.duration_s` 与真实 WAV 时长一致；
- release qualification 强制要求 clean 的 dataset-audit，真人资格集要求 speaker/session/source 三类身份隔离；
- qualification manifest schema v2 会重新读取训练/评测原始音频，不再只相信 JSON 自报 hash；
- target evidence schema v2 绑定 evidence collector 和原始测量文件，CPU/温度/资源/功耗结果不再允许仅靠手写 JSON 构成最终证据；
- 新增 `kws_engine_notify_discontinuity()`，用于 XRUN、route change、clock reset、suspend/resume 后清理 partial frontend、PCEN、RNN hidden、decoder/pending/refractory 状态，同时保留关键词和累计 telemetry；
- CI 增加 C coverage、Clang static analyzer、Python test inventory 和两次独立 SDK build 的 byte-for-byte 一致性门禁；
- 可选 production `torch_ctc` integration workflow 在 digest-pinned 训练镜像内验证真实 train/export/runtime 链路。

这些能力让软件证据真正做到 byte-bound、可审计、不可用“换 WAV 不换 manifest”的方式漂移；但它仍不能替代真实普通话人声、最终麦克风/结构/audio-pipeline 和物理 Cortex-A32 数据，Issue #2 继续作为最终产品资格门。

## 产品特性

- 实时库设备端只依赖 C11 + libm；PyTorch、`pypinyin` 只存在于离线工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **`KWSP` Model ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、词表 fingerprint，并绑定 frontend identity。
- **`KWKP` Keyword Pack ABI v3**：每个关键词独立 threshold、`min_trailing_blanks`、priority、`immediate/longest/grace` prefix policy 和 `grace_frames`。
- `.kwm`、`.kwk`、训练 checkpoint、生成的 C 关键词表统一绑定 64-bit token vocabulary identity。
- `logmel` 与 `pcen-lite` 都贯通 C runtime、dependency-free reference 和训练/export/provenance。
- 相邻重复 token 遵守 CTC blank-separated 结构语义。
- “小窝 / 小窝小窝”这类共享前缀通过 priority、trailing blank 和 grace 策略确定性仲裁。
- L0 换关键词、L1 阈值/回灌校准、L2 `--head-only` 浅定制。
- 默认 32 feature / 48 hidden / 约 420 token 的 dense 声学计算约 **1.2 MMAC/s**，模型权重与 bias 约 **26 KB**；这是设计预算，不是 Cortex-A32 实板测量。

## 构建与安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

SDK 支持 CMake package 和 `pkg-config`：

```cmake
find_package(KwsPipeline CONFIG REQUIRED)
target_link_libraries(app PRIVATE KwsPipeline::core)
```

## 自定义唤醒词

正式产品建议固定显式拼音。关键词 TSV 最多 8 列：

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

共享前缀示例：

```text
1	你好小窝	0.55	ni3 hao3 xiao3 wo1
2	小窝	0.55	xiao3 wo1	1	10	grace	3
3	小窝小窝	0.55	xiao3 wo1 xiao3 wo1	1	20	longest
```

- `immediate`：terminal 达阈值后立即候选输出。
- `longest`：满足 trailing-blank 条件后才释放 pending terminal，给更长共享前缀路径机会覆盖短词。
- `grace`：在 `grace_frames` 内保留 bounded pending terminal，同时满足 `min_trailing_blanks`。
- 同帧多个 immediate 候选按 priority、路径深度、confidence 确定性排序。

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

## Frontend 与模型身份

`training/frontend_spec.py` 是 dependency-free 特征合同，支持 `logmel` 和 `pcen-lite`。两者统一使用 16 kHz、FFT512、mel 80–7600 Hz、400-sample frame、320-sample hop。Model header、checkpoint、export provenance 和 qualification gate 都绑定相同 frontend identity；CI 对两种 frontend 做 C/reference parity。

## 基础训练、数据身份与浅定制

真人发布数据应先做 decoded PCM 和身份隔离审计：

```bash
python3 training/audit_dataset.py \
  --split train=data/train.jsonl \
  --split calibration=data/calibration.jsonl \
  --split qualification=data/qualification.jsonl \
  --require-metadata speaker_id \
  --require-metadata session_id \
  --require-metadata source_id \
  --report build/dataset-audit.json \
  --fail-within-split
```

`train_ctc.py` 同时支持 TSV（`WAV<TAB>token_ids`）和 schema-rich JSONL：

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

checkpoint 会记录 canonical training-corpus identity；exporter 生成 model provenance schema v3，并把真实训练音频身份写入模型 lineage。L2 浅定制使用 `--warm-start ... --head-only`。最终 qualification held-out 不得回灌后继续冒充 unbiased evidence。

## 训练环境

正式训练建议使用预构建的不可变 OCI 镜像。`training/Dockerfile` 要求 base reference 包含 `@sha256:<digest>`，自身不再联网安装/升级 Python 包。使用 `training/build_container.py` 构建 wrapper，并把最终镜像 digest 传入 `KWS_TRAINING_IMAGE_DIGEST`；`train_ctc.py --require-container-digest` 会把该身份写入 checkpoint。

配置仓库变量 `KWS_TRAINING_BASE_IMAGE` 后，`.github/workflows/training-integration.yml` 可周期执行真实 `torch_ctc` train/export/runtime 集成测试。

## Domain-aware 自验证闭环

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

示例覆盖名义 **0.3–5.0 m**、azimuth、RT60、SNR、white/fan/motor/media 噪声、本机播放/AEC residual。calibration/test/qualification 正例按 `far -> mid -> near` 确定性轮转，完整 rendered utterance 走真实 C runtime。

这只能称为 **synthetic-domain evidence**，不能表示真实 3–5 米已经量产通过。

## 连续音频、断流与 long-FAR

正常链路可以直接以 160 sample / 10 ms block 调用 `kws_engine_accept_pcm16()`。出现 XRUN、route、clock 或 suspend/resume 后必须通知引擎：

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

这会阻止断流前后的声学状态被错误拼接。详见 `docs/AUDIO_DISCONTINUITY.md`。

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

v0.3 evaluation provenance schema v2 逐文件绑定真实 WAV，reference `duration_s` 必须与 WAV 实时长一致。`eval/long_far_stream.py` 与 nightly workflow 仍只提供 synthetic regression，不替代真实 FAR/hour。

## 制品绑定的发布认证

v0.3 qualification 会重新读取和交叉验证：model/provenance/checkpoint、训练 corpus、dataset audit、KWKP、release vocabulary/config、eval runner/references/**真实 eval WAV**/detections、target board benchmark、evidence collector 与 raw target evidence。

```bash
python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --model-provenance release/base.kwm.provenance.json \
  --checkpoint release/base.pt \
  --training-tokens release/training-tokens.txt \
  --training-manifest release/train.jsonl \
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
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

`--training-manifest` 和 `--raw-evidence` 可重复。Gate 强制 model ABI v2、keyword-pack ABI v3、frontend-spec v2、runtime/model-lineage frontend 一致、corpus byte identity、dataset-audit coverage、target evidence identity，以及 SKU FAR/FRR/latency/CPU/RSS/stack/soak/温度/功耗门禁。

## 验证边界

CI 已覆盖 GCC/Clang strict build、CTest、Clang static analysis、C coverage、ASan/UBSan、libFuzzer、Cortex-A32 ARMv7 hard-float cross-build、双 frontend parity、decoder/prefix contract、数据泄漏、corpus byte identity、domain-aware loop、streaming long-FAR、schema-v2 release qualification、target evidence collector、两次独立 SDK build reproducibility 和 clean SDK consumer。

这些证明的是**软件合同和 deterministic/synthetic regression**。真正量产仍必须使用真实中文模型、独立真人/设备 held-out corpus、最终双麦/结构/audio-pipeline 和物理 Cortex-A32 证据，记录 FAR/hour、FRR、延迟、CPU、内存、温度、功耗和 soak。Issue #2 保持 open，直到这些真实证据完整绑定并通过 SKU policy。

详见 `docs/README.md`、`docs/CORPUS_IDENTITY.md`、`docs/AUDIO_DISCONTINUITY.md`、`docs/TARGET_EVIDENCE.md`、`docs/REPRODUCIBILITY.md`、`docs/TESTING_STRATEGY.md`、`docs/RELEASE_QUALIFICATION.md`。

## License

Apache-2.0，见 `LICENSE`。
