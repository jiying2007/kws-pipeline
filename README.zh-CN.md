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

## 产品特性

- 实时库设备端只依赖 C11 + libm；PyTorch、`pypinyin` 只存在于离线工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **`KWSP` Model ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、词表 fingerprint，并绑定 frontend identity。
- **`KWKP` Keyword Pack ABI v3**：每个关键词独立 threshold、`min_trailing_blanks`、priority、`immediate/longest/grace` prefix policy 和 `grace_frames`。
- `.kwm`、`.kwk`、训练 checkpoint、生成的 C 关键词表统一绑定 64-bit token vocabulary identity；词表数量相同但 token→ID 不同也会拒绝。
- `logmel` 与 `pcen-lite` 都贯通 C runtime、dependency-free reference 和训练/export/provenance；发布时禁止 runtime frontend 与模型 lineage 不一致。
- 相邻重复 token 遵守 CTC 结构语义：只有 blank-separated prefix 才能继续相同 token。
- “小窝 / 小窝小窝”这类共享前缀可以通过 priority、trailing blank 和 grace 策略确定性仲裁，而不是依赖 TSV 顺序。
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
- `grace`：在 `grace_frames` 内保留 bounded pending terminal，同时继续满足 `min_trailing_blanks`。
- 同一帧多个 immediate 候选按 priority、路径深度、confidence 确定性排序。

编译 `.kwk`：

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

探索阶段可以省略显式拼音，由 `pypinyin` 生成；正式发布应保留显式列，避免依赖升级造成 tokenization 漂移。

## Frontend 与模型身份

`training/frontend_spec.py` 是 dependency-free 特征合同：

- `logmel`：`frontend_kind=0`；
- `pcen-lite`：`frontend_kind=1`。

两者固定使用 16 kHz、FFT512、mel 80–7600 Hz、400-sample frame、320-sample hop。PCEN-lite 在相同 mel 能量上增加 bounded streaming smoothing/gain normalization，然后进入相同 feature normalization。

Model header 固化 frontend kind；checkpoint/provenance 固化 frontend identity 和 frontend-spec 版本；release gate 强制 runtime frontend 与 model lineage 完全一致。CI 同时比较 logmel/PCEN-lite 的真实 C frontend 与 reference 数值输出。

## 基础训练与浅定制

训练前先按**解码后的 PCM 内容**审计 split：

```bash
python3 training/audit_dataset.py \
  --split train=data/train.tsv \
  --split calibration=data/calibration.tsv \
  --split test=data/test.tsv \
  --split qualification=data/qualification.tsv \
  --report build/dataset-audit.json \
  --fail-within-split
```

使用精确 token vocabulary 和 frontend 训练：

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens keywords/tokens.zh.txt \
  --frontend logmel \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

PCEN-lite 模型使用 `--frontend pcen-lite`。exporter 同时生成 `.kwm.provenance.json`，绑定 checkpoint、token identities、training manifests、frontend、训练参数和 int8 量化诊断。

浅定制：

```bash
python3 training/train_ctc.py \
  --manifest data/xiaowo.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start build/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo.pt
```

warm-start 必须匹配 vocabulary、模型几何和 frontend identity。

## Domain-aware 自验证闭环

仓库提供 near/mid/far + frontend A/B 的 deterministic 软件验证链：

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

示例配置覆盖名义 **0.3–5.0 m** 距离、azimuth、RT60、SNR、white/fan/motor/media 噪声，以及可选本机播放/AEC residual。训练集按权重随机采样并由 worst-domain adaptive curriculum 回灌；calibration/test/qualification 的正例按 `far -> mid -> near` 确定性轮转，保证 far-field FRR gate 不会因为“刚好没抽到 far 正例”而误过或误判。

完整 rendered utterance 必须走真实 C runtime，随后统计整体和分 domain FAR/FRR/latency、keyword confusion，并比较 `logmel`/`pcen-lite` 候选，冻结 best model/pack/provenance。

这只能称为 **synthetic-domain evidence**。它不是“真实 3–5 米已经通过”，也不是目标机器人/真实 AFE/Cortex-A32 的量产声学认证。

## 连续音频与 long-FAR

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl \
  --provenance build/detections.provenance.json

python3 eval/score_events.py \
  --references data/eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/summary.json \
  --false-positives build/false-positives.jsonl \
  --false-rejects build/false-rejects.jsonl
```

`eval/domain_metrics.py` 在 reference 带 domain metadata 时输出 near/mid/far、角度、RT60、noise、playback 和 keyword confusion；`eval/long_far_stream.py` 使用 raw streaming C runtime 做长连续背景音频 FAR 回归，`.github/workflows/far-nightly.yml` 提供 hosted nightly regression。

Hosted/synthetic long-FAR 只能作为回归信号，不能替代真实量产 FAR/hour。

## 制品绑定的发布认证

正式 qualification 重新读取并 SHA256 绑定实际 `.kwm`、model provenance、checkpoint、training tokens/manifests、KWKP v3、release tokens/config、eval runner/references/detections、target benchmark runner/board audio 和实板资源证据。

`qualification_gate.py` 要求：

- Model ABI v2；
- Keyword Pack ABI v3；
- frontend-spec v2；
- runtime frontend 与 model lineage frontend 完全一致；
- 所有 byte-complete lineage/corpus/board cross-link 一致；
- SKU policy 中 FAR/FRR/latency/CPU/RSS/stack/soak/温度/功耗等门禁通过。

详见 `docs/RELEASE_QUALIFICATION.md`。

## 验证边界

CI 覆盖 GCC/Clang strict build、CTest、ASan/UBSan、libFuzzer、Cortex-A32 ARMv7 hard-float cross-build、双 frontend parity、decoder/prefix contract、数据泄漏审计、domain-aware multi-frontend loop、streaming long-FAR smoke、byte-complete release qualification 和 clean SDK consumer。

这些结果证明的是**软件合同和 deterministic synthetic regression**。真正量产仍必须使用真实中文模型、独立真人/设备 held-out corpus 和物理 Cortex-A32 证据，记录 FAR/hour、FRR、唤醒延迟、CPU、内存、热/功耗和 soak。仓库 issue #2 继续保持 open，专门跟踪这道真实证据门。

## License

Apache-2.0，见 `LICENSE`。
