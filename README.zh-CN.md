# kws-pipeline

[English](README.md) | 简体中文

`kws-pipeline` 是面向**低算力嵌入式 Linux / RTOS 产品**的常驻端侧唤醒引擎，目标包括 Cortex-A32、Cortex-A7 及相近算力 CPU。它支持“**你好小窝**”“**小窝小窝**”这类可配置中文唤醒词，并可直接接在 [`jiying2007/audio-pipeline`](https://github.com/jiying2007/audio-pipeline) 后作为一级常驻唤醒。

核心不是“一个唤醒词训练一个二分类模型”，而是**开放 token KWS**：

```text
PCM16 16 kHz
 -> 25 ms 窗 / 20 ms hop 的 log-mel
 -> int8 权重 tiny streaming RNN
 -> 拼音 token logits
 -> 共享前缀关键词 Trie
 -> speech / threshold / refractory 门控
 -> wake event
```

新增普通唤醒短语通常只需要改变 token 路径与 threshold；如果 FAR/FRR 仍不可接受，再进入 hard-negative 校准或浅微调。

## 产品特性

- 设备端只有 C11 + libm；PyTorch、`pypinyin` 仅属于 PC 工具链。
- 实时路径无 heap、隐藏线程、锁、文件系统和中文转换。
- 调用方提供对齐 arena，模型 blob 只读且地址稳定。
- L0 换关键词、L1 校准、L2 `--head-only` 浅定制。
- 默认 32 feature / 48 hidden / 约 420 token 的声学 dense 计算约 **1.2 MMAC/s**，模型权重与 bias 约 **26 KB**；这是设计预算，不是 Cortex-A32 实板数据。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 自定义唤醒词

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

示例：

```text
1    你好小窝    0.55    ni3 hao3 xiao3 wo1
2    小窝小窝    0.55    xiao3 wo1 xiao3 wo1
```

正式产品建议固定第四列显式拼音，避免工具依赖升级导致 tokenization 漂移；探索阶段可以省略第四列，由 `pypinyin` 自动生成 tone-aware 拼音。

## 训练与浅定制

基础模型：

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --vocab-size 420 \
  --output build/base.pt
python3 training/export_model.py \
  --checkpoint build/base.pt \
  --output build/base.kwm
```

浅定制：

```bash
python3 training/train_ctc.py \
  --manifest data/xiaowo.tsv \
  --vocab-size 420 \
  --warm-start build/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo.pt
```

`--head-only` 冻结 input/recurrent 表达，仅更新声学输出 head，是本仓库默认的“浅定制”。

## 与 audio-pipeline 对接

推荐正式链路：

```text
双麦/单麦 -> BF/AEC/RES/NS/AGC -> mono S16 @ 16 kHz
          -> kws-pipeline
          -> 唤醒事件
          -> ASR/助手会话
```

如果 `audio-pipeline` 已经输出 16 kHz，不要再做一次 resample。KWS 阈值必须在最终发货的 AEC/NS/AGC 组合下重新评测，尤其包括本机扬声器播放、AEC 残留以及机器人电机/齿轮/风扇噪声。

## 验证

CI 覆盖 GCC/Clang `-Werror`、CTest、ASan/UBSan、关键词 compiler、Python 工具语法和 Cortex-A32 ARMv7 hard-float cross-build。仓库测试只能证明软件合同；真正量产必须在目标 SoC 和真实训练模型上记录 FAR/hour、FRR、唤醒延迟、CPU、内存、热稳定和长时间连续背景音频结果。

详细说明见 `docs/ARCHITECTURE.md`、`docs/CUSTOMIZATION.md`、`docs/PERFORMANCE.md`、`docs/INTEGRATION.md`、`THIRD_PARTY.md`。

## License

Apache-2.0，见 `LICENSE`。
