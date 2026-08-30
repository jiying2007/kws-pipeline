# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近 CPU 预算。它支持“**你好小窝**”“**小窝**”“**小窝小窝**”等可配置中文唤醒词，并设计为消费 [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline) 等 BF/AEC/RES/NS/AGC 前端输出的 16 kHz 单声道 PCM16。

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

## v0.3 软件基线

`v0.3.x` 不改变设备端 **KWSP Model ABI v2** 与 **KWKP Keyword Pack ABI v3**，但对 qualification/evidence 合同做硬切升级：

- 训练 checkpoint/model provenance 逐个绑定训练 WAV 的 file SHA256、decoded PCM SHA256 与 frame 数；
- evaluation provenance 逐个绑定 held-out WAV，并强制 `references.duration_s` 与真实 WAV 时长一致；
- 最终真人 qualification 要求 clean dataset audit，并隔离 speaker/session/source；
- qualification manifest schema v2 会重新读取原始训练/评测音频，不再只相信 JSON 自报 hash；
- product-board evidence 绑定 SKU、source SHA、builder/DUT/collector 身份、精确 collector、runtime-soak 原始字节、canonical raw evidence manifest、外部 attestation verification、board runner、model、keyword pack 与 board audio；
- runtime-soak CPU/RSS/thermal summary 会从保留的 samples 独立重算；
- `kws_engine_notify_discontinuity()` 在 XRUN、route、clock、suspend/resume 后清理 partial acoustic state；
- CI 覆盖 GCC/Clang、静态分析、C coverage、ASan/UBSan、libFuzzer、Cortex-A32 cross-build、SDK 可复现性和 test inventory；
- 可选真实 `torch_ctc` integration workflow 在 digest-pinned 训练镜像内运行。

这些能力可以闭合**软件与证据工程链路**，但不能替代真实普通话人声、最终麦克风/结构/AFE、真实 0.3–5 m 声学资格和物理 Cortex-A32 测量；这些继续由 Issue #2 承担。

## 产品特性

- 实时库只依赖 C11 + libm；PyTorch、`pypinyin` 只存在于离线工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **KWSP ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、词表 fingerprint 和 frontend identity。
- **KWKP ABI v3**：每个关键词独立 threshold、trailing blank、priority 与 `immediate/longest/grace` prefix policy。
- 相邻重复 token 遵守 CTC blank-separated 结构语义。
- “小窝 / 小窝小窝”等共享前缀通过确定性仲裁解决，而不是依赖 TSV 顺序。
- L0 换关键词、L1 阈值/回灌校准、L2 `--head-only` 浅定制。
- 默认 32 feature / 48 hidden / 约 420 token 的 dense 声学计算约 **1.2 MMAC/s**，模型权重与 bias 约 **26 KB**；这是设计预算，不是实板测量。

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
id  text      threshold  explicit-pinyin          min_trailing_blanks  priority  prefix_policy  grace_frames
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝      0.55       xiao3 wo1                1                    10        grace          3
3   小窝小窝  0.55       xiao3 wo1 xiao3 wo1      1                    20        longest
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

`train_ctc.py` 支持 TSV（`WAV<TAB>token_ids`）和 schema-rich JSONL。真人发布数据建议使用带 speaker/session/source/room/device 元数据的 JSONL。正式发布前，dataset audit 必须直接覆盖随后进入 qualification 的**同一个最终 `references.jsonl`**：

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

checkpoint 会记录 canonical training-corpus identity；model provenance schema v3 把它带入正式模型 lineage。最终 qualification held-out 不得回灌后继续冒充 unbiased evidence。

## 不可变训练环境

正式训练应使用 `name@sha256:<digest>` 固定的 OCI 镜像。`training/Dockerfile` 不再联网安装依赖；`training/build_container.py` 校验不可变 base 并记录 build receipt；shipping training 可要求 `KWS_TRAINING_IMAGE_DIGEST`。

真实 `torch_ctc` 集成 workflow 使用仓库变量 **`KWS_TRAINING_IMAGE`**（或手动 `training_image` 输入），且只接受 digest-pinned 镜像引用。

## Domain-aware 自验证闭环

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

示例矩阵覆盖名义 0.3–5.0 m、azimuth、RT60、SNR、white/fan/motor/media 噪声和可选 playback/AEC residual，完整 rendered utterance 走真实 C runtime。

这仍然只能称为 **synthetic-domain evidence**，不能表示真实 3–5 米已经量产通过。

## 连续音频与断流

正常链路可直接以 160 sample / 10 ms block 调用 `kws_engine_accept_pcm16()`。出现 XRUN、route、clock 或 suspend/resume 后，在继续送入新音频前通知引擎：

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

这会阻止断流前后的 partial acoustic state 被错误拼接。

## 连续评测

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

## Product-board 证据合同

先让 runtime-soak collector 监督**真正的被测进程**：

```bash
python3 tools/collect_runtime_soak.py \
  --hours 24 \
  --sample-seconds 60 \
  --output qualification/runtime-soak.json \
  --command ./product-kws-soak --config qualification/product-config.json
```

之后把最终选定的 raw files 冻结到 `qualification/evidence-raw.jsonl`，每行固定 `{name, sha256, bytes}`；同时从受控 qualification trust layer 获取 `qualification/attestation-verification.json`，验证 raw manifest、collector、board runner、model、keyword pack 为同一个证据 tuple。

再按完整 v0.3 CLI 组装 target evidence：

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

`builder-id` 与 `dut-id` 必须不同。collector 会从保留的 runtime-soak samples 独立重算 CPU/RSS/thermal/soak，并绑定 raw/artifact 的精确身份。详见 `docs/TARGET_EVIDENCE.md`。

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
  --evidence-collector tools/collect_target_evidence.py \
  --evidence-raw qualification/evidence-raw.jsonl \
  --attestation-verification qualification/attestation-verification.json \
  --raw-evidence qualification/runtime-soak.json \
  --raw-evidence qualification/stack-watermark.txt \
  --raw-evidence qualification/power.csv \
  --source-sha "$(git rev-parse HEAD)" \
  --sku product-sku-a \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

Gate 强制 model ABI v2、keyword-pack ABI v3、frontend-spec v2、corpus byte identity、dataset-audit coverage、product-board evidence identity、`shipping_approved` SKU policy，以及配置的 FAR/FRR/latency/CPU/RSS/stack/soak/温度/功耗门禁。

## 验证边界

CI 可以证明的是软件合同和 deterministic/synthetic regression：GCC/Clang、CTest、static analysis、coverage、sanitizers、fuzz、Cortex-A32 cross-build、frontend/decoder、corpus identity、qualification integrity、runtime-soak/target-evidence validation、SDK reproducibility 和 clean SDK consumer。

即使仓库全绿并正式发布 `v0.3.0`，也**不代表量产中文 SKU 已实测合格**。独立真人/设备声学数据和物理目标板数据仍需要 Issue #2 完成；按当前边界，它应成为唯一剩余的产品证据门。

详见 `docs/RELEASE_QUALIFICATION.md`、`docs/TARGET_EVIDENCE.md`、`docs/CORPUS_IDENTITY.md`、`docs/AUDIO_DISCONTINUITY.md`。

## License

Apache-2.0，见 `LICENSE`。
