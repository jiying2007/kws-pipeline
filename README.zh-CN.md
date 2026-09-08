# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近 CPU 预算。

当前已经完成 synthetic qualification 的产品 SKU **严格只包含两个四字中文唤醒词**：

- **你好小窝** — `ni3 hao3 xiao3 wo1`
- **小窝小窝** — `xiao3 wo1 xiao3 wo1`

独立两字 **“小窝”不是当前 shipping wake word**。当前正式模型使用专用 5-token 词表：`<blk> / ni3 / hao3 / xiao3 / wo1`。框架本身支持更大的拼音词表和可更新 KWKP，但**当前发布模型并未证明任意中文短语都可以 L0 不训练直接替换**。

典型产品链：

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

## 当前 Engineering-Qualified 模型

当前不可变模型 Release 为 `model-749187ec1d66`，精确绑定：

- model-training run `34134789576`；
- exact trained HEAD `749187ec1d6662658f06aa9c76d47fde835968db`；
- `model.kwm` SHA256 `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`；
- 上述两条精确 shipping keyword；
- untouched synthetic qualification seed `271838`，现已**消费并永久冻结**；
- qualification：`256/256`、`0 FR`、`0 FA`；
- strict robustness 全通过；
- continuous synthetic hard-negative FAR：`0 FA` 且 negative manifest 全覆盖。

`configs/shipping.xiaowo.json` 是机器可读的当前产品 contract。它故意保持 `shipping_approved=false`：synthetic qualification 是强工程证据，但不能替代后续真实人声 + 最终 AFE 声学资格和物理目标板证据。

## Runtime / 产品特性

- 实时库只依赖 C11 + libm；PyTorch、`pypinyin` 仅在线下工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文/拼音转换。
- 调用方提供对齐 engine arena；模型 tensor 零拷贝引用只读 `.kwm` blob。
- **KWSP ABI v2**：固定 16 kHz / 400 sample / 320 sample 几何、vocabulary fingerprint、frontend identity。
- **KWKP ABI v3**：每关键词 threshold、trailing blank、priority、`immediate/longest/grace` prefix policy。
- 相邻重复 acoustic token 遵守 CTC blank-separated 结构语义。
- 共享前缀由确定性 Trie/priority/depth/confidence 策略处理，不依赖 TSV 行顺序。
- `kws_engine_notify_discontinuity()` 在 XRUN、route、clock、suspend/resume 后清理 partial acoustic state。
- 外部 AFE metadata 和 runtime telemetry 均是版本化、有边界的产品接口。

设计文档中提到的约 420-token / 约 26-KB 几何是**框架容量设计项**，不是当前已 qualification 的 5-token 正式模型身份。

## 构建与安装

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

SDK 支持 CMake package 和 `pkg-config`。

## Shipping keyword pack 与定制边界

当前正式 TSV 精确为：

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

L0 keyword-pack 更新只适用于**新关键词的全部 acoustic token 已存在于当前模型词表**的场景，并且新产品 tuple 仍需重新验证。当前 5-token shipping model 只正式验证了上述两个四字词；任何需要其他中文拼音 token 的新唤醒词，都应视为更大词表/新模型或定制模型工作，并经过 fresh qualification，不能宣传成“任意中文零训练换词”。

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

正式 qualification seed `271838` 已在当前 Model Release 中暴露并验收，之后**绝不允许再次重试**。只有未来真正形成新的模型候选时才进入新的 formal seed；当前维护阶段不消费 `271839`。

### Nightly 与 formal qualification 完全分离

`.github/workflows/far-nightly.yml` 不再重新训练模型，也不再生成 formal qualification cohort。Nightly 会：

1. 下载 exact immutable `model-749187ec1d66`；
2. 校验 Release checksums、promotion manifest、qualification / robustness / continuous-FAR 证据；
3. 使用独立 `nightly-frozen-model-v1` synthetic negative namespace；
4. 对 frozen released model 跑四个 long-FAR shard；
5. 只生成 regression evidence。

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

后续真实人声/设备阶段会把最终 `audio-pipeline` binary + config 注入这个 contract；在最终 AFE identity 真正存在并通过最终证据前，`shipping_approved` 始终保持 false。

## 不可变训练环境

正式训练必须使用 `name@sha256:<digest>` 固定 OCI 镜像。`training/Dockerfile` 不在 build 时浮动安装依赖；`training/build_container.py` 校验 immutable base 并记录 build receipt。真实 `torch_ctc` integration workflow 使用仓库变量 **`KWS_TRAINING_IMAGE`**（或手动 `training_image` input），只接受 digest-pinned image。

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

最终 gate 强制 model ABI v2、keyword-pack ABI v3、frontend-spec v2、exact corpus identity、dataset-audit coverage、product-board evidence identity，以及 policy 的 `shipping_approved=true`。仓库现在主动保持该状态为 false，直到后置的真实人声/最终 AFE 与物理目标板证据真正形成。

## 验证边界

CI 当前证明软件合同和 deterministic/synthetic regression：GCC/Clang、CTest、static analysis、coverage、ASan/UBSan、libFuzzer、Cortex-A32 cross-build、frontend/decoder、corpus identity、自训练闭环、robustness/FAR regression、runtime-soak/target-evidence schema、SDK reproducibility、model supply-chain。

它**不宣称真实商用声学和物理板级 qualification 已完成**。完成本轮 non-real-data 工程闭环后，产品证据阶段只剩两步：① 真实普通话经过最终麦克风/结构/AFE；② 物理目标板 performance/soak 并最终批准完整 deployment tuple。

详见 `docs/README.md`、`docs/CUSTOMIZATION.md`、`docs/RELEASE_QUALIFICATION.md`、`docs/TARGET_EVIDENCE.md`、`docs/CORPUS_IDENTITY.md`、`docs/AUDIO_DISCONTINUITY.md` 和 Issue #2。

## License

Apache-2.0，见 `LICENSE`。
