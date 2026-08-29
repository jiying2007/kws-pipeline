# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近算力 CPU。它支持“**你好小窝**”“**小窝小窝**”这类可配置中文唤醒词，并设计为直接消费 [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline) 等轻量语音前端输出的 16 kHz 单声道 PCM。

核心不是“一个唤醒词训练一个二分类模型”，而是**开放 token KWS**：

```text
PCM16 16 kHz
 -> 25 ms 窗 / 20 ms hop log-mel
 -> int8 权重 tiny streaming RNN
 -> 拼音 token logits
 -> 共享前缀关键词 Trie
 -> speech / threshold / refractory 门控
 -> wake event
```

普通唤醒短语变化属于 L0 配置更新，不需要重新训练模型。如果现场 FAR/FRR 仍不达标，再进入连续音频校准、hard-negative 挖掘和浅层 output-head 微调。

## 产品特性

- 实时库设备端只依赖 C11 + libm；PyTorch、`pypinyin` 只存在于离线工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- 支持现场更新 `.kwk` 关键词包；改变唤醒词不需要重新链接 firmware。
- ABI v2 使用同一份 **token vocabulary identity** 绑定 `.kwm`、`.kwk`、训练 checkpoint 和生成的 C 关键词表；即使词表数量相同，只要 token→ID 映射不同也会拒绝。
- 关键词更新先完整验证，再替换活动 Trie；错误配置不会破坏上一份有效配置。
- 相邻重复声学 token 遵守 CTC 结构语义：必须从已经观察到 blank separator 的 prefix 状态才能继续相同 token；decoder 独立保留 nonblank / blank-separated 两个 Viterbi 子状态。
- L0 换关键词、L1 阈值/hard-negative 校准、L2 `--head-only` 浅定制。
- 默认 32 feature / 48 hidden / 约 420 token 的声学 dense 计算约 **1.2 MMAC/s**，模型权重与 bias 约 **26 KB**；这是设计预算，不是 Cortex-A32 实板数据。

## 构建与安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

安装后的 SDK 支持：

```cmake
find_package(KwsPipeline CONFIG REQUIRED)
target_link_libraries(app PRIVATE KwsPipeline::core)
```

也支持 `pkg-config --cflags --libs kws-pipeline`。

## 自定义唤醒词

量产建议固定第四列显式拼音，避免工具升级导致 tokenization 漂移：

```text
1    你好小窝    0.55    ni3 hao3 xiao3 wo1
2    小窝小窝    0.55    xiao3 wo1 xiao3 wo1
```

从同一份 vocabulary 同时生成现场可更新 `.kwk`、可选的固件内置 C 表和可读 manifest：

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-pack build/xiaowo.kwk \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

探索阶段可以省略第四列，由 `pypinyin` 自动生成带声调拼音；正式发布建议始终保留显式拼音。

## 基础训练与浅定制

训练前先按**解码后的 PCM 内容**审计 train / calibration / qualification 数据，而不是只比较文件名或 WAV 容器字节。这样即使同一段录音被改名、复制或重新封装并增加 RIFF metadata，也会被检测为跨 split 泄漏：

```bash
python3 training/audit_dataset.py \
  --split train=data/train.tsv \
  --split calibration=data/calibration.tsv \
  --split qualification=data/eval/references.jsonl \
  --audio-root qualification=data/eval \
  --report build/dataset-audit.json
```

基础 CTC 模型训练时必须直接提供后续 `.kwm/.kwk` 使用的 exact token vocabulary：

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --tokens keywords/tokens.zh.txt \
  --output build/base.pt

python3 training/export_model.py \
  --checkpoint build/base.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/base.kwm
```

checkpoint 会固化 vocabulary fingerprint、token 文件 hash、训练 manifest hash、frontend spec 版本、seed 和 optimizer 参数；warm-start 必须匹配同一 fingerprint，exporter 也会拒绝把 checkpoint 绑定到“数量相同但 token→ID 映射不同”的词表。

浅定制可以把正常样本和挖掘出的 hard negatives 一起训练，并冻结 input/recurrent backbone：

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

`--head-only` 必须配合 warm start，只更新声学输出 head。完整模型微调也支持，但需要更宽的回归语料。

`training/frontend_spec.py` 是 dependency-free 特征规范；CI 会通过真实 C frontend 的 `kws_feature_dump` 逐维对比该规范，PyTorch frontend 也复用同一套 mel/FFT/scale 常量，避免训练端与设备端特征慢慢漂移。

## Runtime API

现场更新路径保持为四步：

```c
kws_model_t model;
kws_keyword_pack_t pack;
kws_engine_t *engine;
kws_config_t cfg = kws_default_config();

kws_model_open(model_blob, model_blob_bytes, &model);
kws_keyword_pack_open(pack_blob, pack_blob_bytes, &model, &pack);
size_t arena_bytes = kws_engine_required_bytes(&model);
kws_engine_init(arena, arena_bytes, &model, &cfg, &engine);
kws_engine_set_keyword_pack(engine, &pack);

int detected = 0;
kws_detection_t hit;
kws_engine_accept_pcm16(engine, pcm, samples, &hit, &detected);
```

`kws_engine_set_keyword_pack()` 返回后，解析后的 pack 对象可以释放；`.kwm` 模型 blob 必须在 engine 整个生命周期内保持有效。

## 连续音频量产评测

`kws_wav` 使用**真实 C runtime**处理 16 kHz 单声道 PCM16 WAV；`run_corpus.py` 生成 SHA256 provenance sidecar；`score_events.py` 计算 FAR/hour、FRR 和唤醒延迟：

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
  --false-positives build/false-positives.jsonl
```

误唤醒可以通过 `eval/mine_hard_negatives.py` 转为 empty-target CTC 训练片段。最终 held-out 量产认证集不能先拿去挖 hard negative，再继续当作无偏发布证据。

## 制品绑定的发布认证

源码 CI 全绿只表示**软件基线可用**，不能直接等价为量产声学指标。正式认证路径会重新读取实际文件并重算语料/板端统计，而不是只相信 sidecar：

```bash
./kws_board_bench \
  release/base.kwm \
  release/xiaowo.kwk \
  qualification/board-audio.wav \
  10 > qualification/board-summary.json

cp /path/to/exact-target-kws_board_bench qualification/kws_board_bench.target
cp /path/to/exact-eval-kws_wav qualification/kws_wav.eval

python3 tools/qualification_manifest.py \
  --model release/base.kwm \
  --keywords release/xiaowo.kwk \
  --tokens release/tokens.txt \
  --config release/runtime.json \
  --eval-runner qualification/kws_wav.eval \
  --references qualification/references.jsonl \
  --detections qualification/detections.jsonl \
  --eval-summary qualification/eval-summary.json \
  --eval-provenance qualification/detections.provenance.json \
  --board-summary qualification/board-summary.json \
  --board-runner qualification/kws_board_bench.target \
  --board-audio qualification/board-audio.wav \
  --evidence qualification/evidence.json \
  --source-sha "$(git rev-parse HEAD)" \
  --corpus-id home-kws-heldout-v1 \
  --output qualification/qualification-manifest.json

python3 tools/qualification_gate.py \
  --manifest qualification/qualification-manifest.json \
  --policy qualification/sku-policy.json \
  --output qualification/gate-result.json
```

`qualification_manifest.py` 会独立重验 canonical ABI、runtime config、vocabulary identity、实际 eval runner/references/detections、reference/detection 数量、board WAV 时长和所有 SHA256/统计公式；`qualification_gate.py` 对 FAR/FRR/latency/p99/RTF/headroom、CPU、RSS、stack、soak、温度和功耗应用明确 SKU policy。

## 与 audio-pipeline 对接

推荐正式链路：

```text
双麦/单麦 -> BF/AEC/RES/NS/AGC -> mono S16 @ 16 kHz
          -> kws-pipeline
          -> 唤醒事件
          -> ASR/助手会话
```

如果 `audio-pipeline` 已经输出 16 kHz，不要再做一次 resample。KWS 阈值必须在最终发货的 AEC/NS/AGC 组合下重新评测，尤其覆盖本机扬声器播放、AEC 残留以及机器人电机/齿轮/风扇噪声。

## 验证

CI 当前门禁包括 GCC/Clang strict build、CTest、ASan/UBSan、decoder CTC-repeat contract、C/reference frontend 数值一致性、decoded-PCM 数据泄漏审计、关键词/评测/provenance/qualification 测试、真实制品 board benchmark、SDK install + clean consumer，以及 Cortex-A32 ARMv7 hard-float 交叉编译。独立 Clang **libFuzzer + ASan/UBSan** job 会持续从 canonical seed 变异 `.kwm/.kwk` parser 输入。

Hosted CI 数据只作为回归信号。真正量产仍必须使用真实训练模型、最终 held-out corpus 和目标 SoC，记录 FAR/hour、FRR、唤醒延迟、p95/p99 处理时间、CPU、内存、热/功耗和长时间连续背景音频结果。仓库 issue #2 专门跟踪这道真实证据 gate。

详细说明见 `docs/ARCHITECTURE.md`、`docs/CUSTOMIZATION.md`、`docs/EVALUATION.md`、`docs/PERFORMANCE.md`、`docs/INTEGRATION.md`、`docs/RELEASE_QUALIFICATION.md`、`THIRD_PARTY.md`。

## License

Apache-2.0，见 `LICENSE`。
