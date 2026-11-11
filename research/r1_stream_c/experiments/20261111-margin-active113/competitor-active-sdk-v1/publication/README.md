# ACTIVE113：独立六分数研究 SDK

ACTIVE113 保留原 R1 前端、骨干和检测器，只计算冻结选择的 113 个 OTHER 输出行以及原五个精确行。它使用独立的模型格式、公开头文件、ABI 与符号命名空间；默认 R1 SDK 和模型保持原样。

公开接口返回 **六个数值分数，加一个逐帧 OTHER 原始类别 ID**。六个数值不能作为 2,601 类 logits 数组、概率分布或 CTC 输出使用。本目录描述格式和接入契约；PCM 一致性、资源测量和后续审计结果应引用独立验证回执。

## 输出与平局语义

| 位置 | 数值含义 | 对应类别 ID |
|---|---|---|
| `scores[0..4]` | 原五行精确 logits | `0, 1027, 1401, 1462, 2093` |
| `scores[5]` | 113 个选定 OTHER 行的最大 logit | 回调提供的 `other_original_id` |

最终 winner 先比较分数，分数相同时选择**较小的原始类别 ID**。OTHER 内部也采用这一规则。实际 OTHER winner ID 原样传入检测器；不能把它替换为 sentinel 1，也不能按六维位置编号进行 argmax。

`active113_sdk_pick_original_token(scores, 6, other_original_id, &winner)` 实现这一比较，并检查分数是否有限、OTHER ID 是否属于冻结的 113 行。输入错误时不修改 `winner`。在已有帧回调中使用该函数，不会增加模型求值次数。

SDK 交付包的 API 头文件为 `sdk-v2/include/active113_sdk.h`，ABI 为 `0xA1130001`。公开及底层符号分别使用 `active113_`、`r1active113_`、`r1a113_` 前缀，避免与默认 2,601 输出实现发生链接冲突。

## 接入接口

帧观察回调的签名为：

```c
typedef int (*active113_frame_fn)(
    void *user,
    uint64_t frame,
    uint64_t dependency_end,
    uint64_t delivery_end,
    const double *features,       /* 400 values */
    const double *scores,         /* exactly 6 values */
    uint32_t other_original_id);
```

数组只在同步回调期间有效。事件结构保留关键词 ID、segment、模型帧、首 token 帧、源依赖末端和交付末端；时钟仍以输入采样点表示，不是声学标注的关键词结束位置。

`active113_sdk_create()` 分配状态并复制模型。`active113_sdk_create_view()` 借用不可变、8 字节对齐的模型存储，支持由 `read`／`fread` 填充的分配内存或只读 mmap。借用存储必须持续有效且不被任何调用方修改，直到 handle 销毁；不发生所有权转移。前端表在两种模式中均被复制。模型视图不能来自具有声明字节数组类型的对象，或从这类对象 `memcpy` 得到的存储；这是类型视图的别名契约。

输入为调用方 AFE 处理后的 16 kHz、单声道、PCM16 little-endian。每次 `active113_sdk_feed()` 最多 8,192 字节，并检查返回的 `consumed`。允许奇数字节分块，但 `finish()` 时若有未配对字节会失败。`finish()` 不补音频、不人为冲刷帧。独立录音之间调用 `reset()`；时间缺口使用 `notify_gap()`，不跨缺口延续状态。接口同步、单线程，回调不可重入或销毁当前 handle。

以下片段假设调用方已读取并校验模型／前端文件，且 `pcm` 是一个合法分块：

```c
#include "active113_sdk.h"

active113_sdk *sdk = active113_sdk_create(
    model_bytes, ACTIVE113_MODEL_BYTES,
    frontend_bytes, ACTIVE113_TABLE_BYTES,
    0, on_event, on_frame, user);
if (!sdk) {
    /* Reject initialization; do not substitute a different model format. */
    return 1;
}
size_t consumed = 0;
int rc = active113_sdk_feed(sdk, pcm, pcm_size, &consumed);
if (rc != ACTIVE113_OK || consumed != pcm_size) {
    active113_sdk_destroy(sdk);
    return 1;
}
/* Feed remaining chunks, then finish once and destroy. */
rc = active113_sdk_finish(sdk);
active113_sdk_destroy(sdk);
return rc == ACTIVE113_OK ? 0 : 1;
```

正常运行可以将帧观察回调设为 `NULL`。需要观察时，回调提供的分数和元数据已经来自本次实际求值，不应另行重算完整输出头。

## 模型格式与身份

| 项目 | 固定值 |
|---|---|
| 模型文件 | `model-active113.r1a113` |
| 文件大小 | **1,582,624 bytes** |
| SHA-256 | `38818ee7ba00c9e9bb762f9d1d2efeca9c1bac349b8f1cf835913fd30ef75984` |
| Magic／版本 | `R1A113V1`／1 |
| Header | 256 bytes，little-endian |
| 选择清单 SHA-256 | `8289f2ca46e40717747c1ed8d08152d8db2fbadcac5f4a84fdbb20d06375dc42` |
| 原模型 SHA-256 | `63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4` |
| 前端文件 | `frontend.f32le`，83,840 bytes |
| 前端 SHA-256 | `49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49` |

新文件由固定 header、五个精确 ID 与 113 个 OTHER ID、原 390,474 个 float32 骨干参数、113 行 int8 权重／float64 scale／float32 bias、五行精确 float32 权重及 bias 组成。选取后的参数字节保持原值，没有重新量化或拟合。

加载器验证精确大小、固定 header 和保留字段、冻结 ID 序列、可信的整文件 SHA-256、有限参数、正 scale、CMVN scale、禁用的 int8 `−128` 值及零 padding。复制模式与借用模式在所有检查完成后才发布状态；失败不修改原模型状态或复制目标。标准大小和哈希不能由不可信输入自行决定。

OTHER 每行保持原 par4 双精度四路求和顺序，再乘原 scale、加原 bias。五个精确行保持原标量双精度求和顺序。实际仅计算 113 个 OTHER 行，不会先计算全部 2,596 个 OTHER 行再覆盖输出。`r1active113_step()` 的可选 trace 保留原 stage 0–19；stage 20 只有六个分数。trace 容量为 5,006 个 double，stage 19 的 140D 隐层从 `19 * 250` 开始。

## 编译与链接

以下 `sdk-v2`、`pack-v2`、`build-v2` 路径属于独立 SDK 的 owner 交付包。公开仓库保存实际使用的准备、执行与审计源码；生成的 SDK、模型／前端文件和预编译库保留在 owner 交付包，不是公开 clone 的组成部分。

SDK 交付树保留 `sdk-v2/include`、`sdk-v2/core`、`sdk-v2/src`，预编译库位于 `build-v2/host` 和 `build-v2/arm`。取得该交付包后，使用对应目标架构的 `libactive113_sdk_static.a`，链接数学库：

```sh
gcc -std=c11 -O2 -fno-fast-math -ffp-contract=off \
  -fexcess-precision=standard -I /path/to/sdk-v2/include \
  app.c /path/to/build-v2/host/libactive113_sdk_static.a -lm -o app
```

从源码构建时，以下可重定位命令沿用已经执行的八单元 GCC 构建方式，在独立输出目录生成静态库：

```sh
active113_sdk=/path/to/sdk-v2
active113_out=/path/to/local-build
mkdir -p "$active113_out"
for unit in r1a113_stream r1a113_pcm reference64_fbank r1active113 sha256 active113_layout
do
  gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fPIC \
    -fno-fast-math -ffp-contract=off -fexcess-precision=standard \
    -I "$active113_sdk/include" -I "$active113_sdk/core" \
    -c "$active113_sdk/core/$unit.c" -o "$active113_out/$unit.o"
done
for unit in active113_sdk active113_detector
do
  gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fPIC \
    -fno-fast-math -ffp-contract=off -fexcess-precision=standard \
    -I "$active113_sdk/include" -I "$active113_sdk/core" \
    -c "$active113_sdk/src/$unit.c" -o "$active113_out/$unit.o"
done
ar rcsD "$active113_out/libactive113_sdk_static.a" \
  "$active113_out/r1a113_stream.o" "$active113_out/r1a113_pcm.o" \
  "$active113_out/reference64_fbank.o" "$active113_out/r1active113.o" \
  "$active113_out/sha256.o" "$active113_out/active113_layout.o" \
  "$active113_out/active113_sdk.o" "$active113_out/active113_detector.o"
```

内部 include 目录为 `include` 和 `core`；`core/vendor/fft_twiddles_f64.h` 是必须保留的固定前端依赖。浮点环境要求 IEEE binary32／binary64、little-endian、RNE、逐渐下溢和屏蔽异常 trap。

ARM 构建应将编译器／归档器替换为适合目标的 GCC hard-float 工具，并配置对应 sysroot；编译参数再加入 `-mcpu=cortex-a32 -mfpu=neon-vfpv4 -mfloat-abi=hard`。不能把 host 或 ARM QEMU 构建成功解释为 SSC305 实板资格。正式验证使用冻结的 `build_sdk.py` 回执绑定库身份；独立重编译需保留自己的构建回执。

## 当前验证状态

下面收录了已保存隐层筛查、实际 PCM 验证、单次同架构诊断以及独立补充协议的实际结果。补充协议在新增调用前冻结，并复用原捕获结果；原严格跨架构逐位门禁的失败记录保持原样。证据文件的整文件 SHA-256 记录在 `MODEL_CARD.json`，公开摘要不包含单条录音或失败位置。

| 阶段 | 已完成证据 | 结论与范围 |
|---|---|---|
| 已保存隐层上的输出头筛查 | 96 条训练、34 条已暴露开发记录；130 条的原始 winner ID、F/E 和时钟保留 | 筛查通过；训练 9,079 帧的 OTHER 最大值逐位一致。开发 3,176 帧中有 17 帧的子集 OTHER 最大值降低，未改变 winner 或事件 |
| 构建与加载器检查 | Host、ARM 静态库；各 87 项人工数据检查；独立静态审查 307 项断言 | 两架构构建通过；各 46 个定义符号均为独立命名空间，加载器检查不运行模型或 PCM |
| 原实际 PCM 验证 | 原计划 290 次；已执行 135 次候选捕获，134 次完整比较通过；130 条 host 全部通过 | 一次 ARM 捕获的 hidden140 与 x86_64 教师缓存未逐位相同，原严格跨架构 bit gate **失败并保留**；该捕获的 F/E 仍逐字节一致。原运行尚未进入 CPU／内存阶段 |
| 独立同 ARM 基线诊断 | 新增 1 次原基线 ARM 数值捕获，候选重跑 0 次；独立保存证据审计 1,417 项断言 | 在该案例上，基线 ARM 与已保存候选 ARM 的 hidden140、五个精确分数、winner 和 F/E 一致；基线同样出现相对 host 缓存的差异。这解释该案例的差异来源，不把原严格 gate 改成通过，也不外推全部 ARM 输入 |
| 独立补充 PCM 验证 | 复用原 135 次候选捕获，新增剩余 125 次 ARM 捕获；host／ARM 各 130 条已知记录、12,255 帧、86 个事件 | 两架构的 F/E、源／交付时钟、全局 winner 和 OTHER 原始 ID 全部一致。六分数通过对保留 host ACTIVE113 捕获的既有数值界限；原严格 bit gate 仍失败 |
| 同轮 host 资源测量 | 9 对 CPU、6 对内存观测，双方均为 mmap；共 30 次独立 plain probe 调用 | CPU 每对均下降，配对节省中位数 35.1463%；PSS 中位数减少 388 KiB、Private_Dirty 减少 20 KiB。范围为单个固定输入的 host 离线流式求值 |

补充协议的六分数界限引用已有 `CROSS_ARCH.md`：相对 host ACTIVE113 对应六分数，逐值满足 `abs(ARM − HOST) <= 1e-8 + 1e-8 * abs(HOST)`，且最大绝对误差不超过 `1e-6`。实测六分数最大绝对误差为 **5.258016244624741e-13**。这只覆盖独立六分数接口，未比较完整 2,601 类 logits 或后验概率。stage 19 hidden140 的最大绝对差为 **1.9184653865522705e-13**，仅作为诊断报告；没有把旧前端的误差限套用到 hidden140。原 R1 FP32 数值资格也保持 `FAIL_UNCHANGED`。

此次实际调用总数为 **291**：260 次唯一的候选功能捕获、18 次 CPU、12 次内存，以及 1 次原 ARM 基线诊断。候选功能捕获没有重跑，未知标签记录未进入筛查或实际调用。补充结果的独立保存证据审计通过 63,102 项断言，没有新增模型或检测器调用。

### 同轮资源结果

| 指标 | 原完整输出头，mmap | ACTIVE113，mmap | 配对结果 |
|---|---:|---:|---|
| CPU 秒／音频秒，中位数 | 0.021045392 | 0.013833457 | 9／9 对下降；配对节省中位数 **35.1463%**，范围 28.7817%–40.7557% |
| 进程自身 PSS，中位数 | 2,495 KiB | 2,107 KiB | 6 对差值中位数 **388 KiB** |
| 进程自身 Private_Dirty，中位数 | 448 KiB | 428 KiB | 6 对差值中位数 **20 KiB** |
| 模型文件大小 | 1,960,140 bytes | 1,582,624 bytes | 减少 **377,516 bytes（19.2596%）** |

CPU 使用无捕获代码的 plain probe，比较同一轮、同一固定输入、交替次序的九对观测；没有将旧轮次、QEMU 或捕获耗时计入加速结论。配对节省中位数与两组中位数的比值是不同统计量。内存在本进程完成流式输入后、销毁 SDK 前读取；不用 peak RSS 代表节省，也不用内存测量的时间作加速证据。模型文件大小下降不等同于目标板 RAM 资格。

这里没有训练新的输出头或重新量化参数。前两次 max-OTHER MLP 研究候选仍为拒绝状态；ACTIVE113 是固定训练帧 OTHER 最大行 ID 并集的独立子集方案，不能把当前筛查结果归因于那些被拒绝的候选。

## 研究范围与发布边界

113 行来自冻结的 96 条已知训练记录中，每帧 OTHER 集合最大行的原始 ID 并集，没有 top-k、数量上限或开发集选择；五个精确行、前端、骨干和检测器均保留。开发集已经暴露，且为合成、机器弱标签；当前研究不能提供独立 FAR／FRR。未选中的 OTHER 类仍可能在新输入上获胜，因此这里没有完整类别竞争、全量概率或 CTC 等价保证。模型资格与默认模型切换均需独立决策。

准备脚本 `pack_active.py`、`materialize.py`、`build_sdk.py` 依赖原 owner 的冻结选择清单、通过的缓存筛查回执和模型源文件；它们保留执行路径和哈希约束。公开准备与审计源码可供审查，自动重做仍需 owner 输入及原环境。上面的可重定位 GCC 指令用于编译交付包中已经生成的独立 SDK。公开源码不包含用于自动重做选择的私有来源映射。

初次库构建因漏带已有 FFT 表头而在编译阶段停止；随后只补齐该固定依赖及目录生成逻辑。原准备源码、计划、失败日志／回执以及首版输出保留在 owner 历史中，没有模型或 PCM 执行。SDK 组件来源由 owner 的 `INVENTORY.json` 绑定，参数、构建和验证身份由本目录 `MODEL_CARD.json` 绑定；完整公开提交的文件清单以同次公开结果清单为准。后续验证结果应单独追加，不回写冻结构建树。
