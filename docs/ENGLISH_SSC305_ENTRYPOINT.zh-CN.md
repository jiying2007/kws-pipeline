# Hey Mero V6.0 — SSC305 单关键词 43K 接入入口

> **唯一活动英文词：Hey Mero /heɪ ˈmɛroʊ/**。当前唯一可复核的软件部署基线为 B0（43,876 参数，256 步 FSMN-CTC）；B1/B2/B3/B4 是**独立研究候选**，任何新的 PT、ONNX 或 NPZ 均不可直接替换 B0 板端模型。此文是现场接入指南，**不构成已在 SSC305 运行或已通过产品 FRR/FAR 的证明**。

## 0. 单一可信入口与边界

| 项目 | 当前有效入口/证据 |
|---|---|
| Hey Mero 独立 C11 SDK | `research/english_v5_mero/{mero43_stream.[ch],mero_ctc.[ch],CMakeLists.txt}` |
| 实际软件模型 | `kws-data` run [44495594627](https://github.com/jiying2007/kws-data/actions/runs/44495594627)，artifact `14514228406`；ZIP SHA256 `91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702` |
| 原始权重重打包 | `research/english_v5_mero/pack_mero43.py`；只接受上述冻结 ZIP，输出 216,960 字节 `MERO43V1` 包和 SHA 回执 |
| SDK 构建/安装/消费 | [SSC305_SDK_HANDOFF.md](../research/english_v5_mero/SSC305_SDK_HANDOFF.md) |
| C11/ARM 软件一致性 | B0：[原始 PCM 10/10 x86](https://github.com/jiying2007/kws-data/actions/runs/44541193521)及[10/10 ARM32 QEMU](https://github.com/jiying2007/kws-data/actions/runs/44542782505)；B3 另有[28 例 C11/ARM32 数值证据](https://github.com/jiying2007/kws-data/actions/runs/44653612830)，但仍非晋级模型 |
| V6 质量/模型状态 | [ENGLISH_V6_MERO_EXECUTION.md](ENGLISH_V6_MERO_EXECUTION.md)、[V6_QUALITY_EVIDENCE.json](../research/english_v5_mero/V6_QUALITY_EVIDENCE.json) |

**禁止串线：**仓库历史 `research/english_stream_integration`、`ekws_pcm_stream` 和 `models/causal-head.bin` 属于旧英文 full-chain 研究路径，不是当前 Hey Mero 单词 43K B0 模型的集成入口。不能因为旧 SDK、旧 ARM 工程或旧历史测试通过，就把它们当作 V6 43K 产品验收。当前单词 SDK 的目录中保留 `english_v5_mero` 名称是源码路径事实，不表示该模型已通过 V6 质量晋级。

## 1. 收集真实 BSP/板端身份（没有硬件时保持 NOT_RUN）

本仓库可复用只读探针：

- `research/arm_runtime/scripts/probe_sdk_readonly.sh`：以现场真实 BSP C 编译器及目标板复制出的 ELF 为输入，核对架构/ABI；不修改目标机。
- `research/arm_runtime/scripts/probe_board_readonly.sh`：在真实板上记录内核、CPU、libc/loader 等。现场指令仅在具有合法设备访问、正确可写工作目录时执行。

实际接入时记录：SSC305 具体板型与 BSP 版本、真实交叉编译器路径、sysroot 来源、ELF 浮点 ABI、板端执行/文件传输入口、ALSA/MI_AI/AFE 的实际采样路径，以及设备供电和 CPU 调频条件。当前仓库**未提供这些现场事实**；不能用 Ubuntu ARM 工具链、QEMU 或示例配置填充它们。

在 BSP 主机先取得只读身份：

```sh
: "${KWS_PIPELINE_SOURCE:?仓库绝对路径}"
: "${KWS_BSP_CC:?真实 BSP C 编译器绝对路径}"
: "${KWS_TARGET_ELF_COPY:?从实际目标板复制的 ELF}"
sh "$KWS_PIPELINE_SOURCE/research/arm_runtime/scripts/probe_sdk_readonly.sh" \
   "$KWS_BSP_CC" "$KWS_TARGET_ELF_COPY" > sdk-inventory.txt
```

## 2. 仅打包已认证 B0 模型（在构建主机执行）

原始公开 Actions 模型 ZIP 必须按仓库文档的 artifact ID 和 SHA 获取，先以 `sha256sum` 检查完整 ZIP，再调用源码内置的强制校验器。以下均由现场提供实际路径，不自动下载、更换训练权重或重用已过期文件：

```sh
: "${KWS_PIPELINE_SOURCE:?}"
: "${KWS_B0_ORIGINAL_ZIP:?已下载且确认 SHA 的原始 artifact ZIP}"
: "${KWS_B0_PACK_OUTPUT:?新的、尚不存在的模型输出路径}"
python3 -I -B "$KWS_PIPELINE_SOURCE/research/english_v5_mero/pack_mero43.py" \
  --source-zip "$KWS_B0_ORIGINAL_ZIP" \
  --output "$KWS_B0_PACK_OUTPUT"
sha256sum "$KWS_B0_PACK_OUTPUT"
```

保留生成的 `*.json` 回执，记录原始 ZIP 和最终 pack 的 SHA、字节数、版本及来源。**Python 仅在构建/审核主机上使用**；SSC305 运行时不需要 Python。C11 加载器负责格式、参数数值校验，不负责认证文件 SHA；设备软件的模型装载链须另行执行与回执匹配的 SHA 验证。

## 3. 构建/安装单关键词 C11 SDK

先在 Linux 构建主机做无需模型权重的 C11 合成源测试和 CMake 安装冒烟：

```sh
cmake -S "$KWS_PIPELINE_SOURCE/research/english_v5_mero" \
  -B /tmp/heym43-native -DCMAKE_BUILD_TYPE=Release \
  -DHEY_MERO43_BUILD_TESTS=ON -DHEY_MERO43_BUILD_HOST_TOOL=OFF
cmake --build /tmp/heym43-native --parallel 2
ctest --test-dir /tmp/heym43-native --output-on-failure
```

随后使用**已查证为实际板型对应的** BSP 工具链和 sysroot 交叉编译。以下 CFLAGS 来自现场 ABI 审核，不预设 hard-float、NEON 或动态装载器：

```sh
: "${KWS_BSP_CC:?真实 BSP C 编译器}"
: "${KWS_BSP_SYSROOT:?真实 BSP sysroot}"
: "${KWS_BSP_CFLAGS:?已核实的目标 CFLAGS}"
: "${KWS_BSP_BUILD:?新的构建目录}"
: "${KWS_BSP_INSTALL:?SDK 安装目录}"
cmake -S "$KWS_PIPELINE_SOURCE/research/english_v5_mero" -B "$KWS_BSP_BUILD" \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_C_COMPILER="$KWS_BSP_CC" -DCMAKE_SYSROOT="$KWS_BSP_SYSROOT" \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_C_FLAGS="$KWS_BSP_CFLAGS" \
  -DHEY_MERO43_BUILD_TESTS=OFF -DHEY_MERO43_BUILD_HOST_TOOL=OFF \
  -DCMAKE_INSTALL_PREFIX="$KWS_BSP_INSTALL"
cmake --build "$KWS_BSP_BUILD" --parallel 2
cmake --install "$KWS_BSP_BUILD"
```

安装得到 `heym43/mero43_stream.h`、`heym43/mero_ctc.h`、`HeyMero43::stream` 静态库。实际应用需在同一 BSP/ABI 下链接 `libm`（库已声明链接依赖），并可由 CMake `find_package(HeyMero43 0.1 CONFIG REQUIRED)` 找到。**交叉编译成功只证明构建，不证明目标板已执行。**

## 4. 产品应用接线与运行边界

应用负责认证 `MERO43V1` 模型 pack（216,960 字节）、分配静态模型/单流内存、初始化并持续提供 **16 kHz 单声道有符号 PCM16**：

1. 在非实时线程完成模型来源、包 SHA 验证和 `en_v5_m43_model_load()`；模型常驻只读。
2. 用 `en_v5_m43_model_bytes()` 和 `en_v5_m43_stream_bytes()` 确认 arena，初始化 `en_v5_m43_stream_init()`，使用**单独、可审计、独立 CAL 得到的** `max_gap_samples` 和 `cooldown_samples`。这两个值尚未校准时，禁止将 SDK 事件作为产品唤醒决策。
3. 单 owner 线程每次调用 `en_v5_m43_feed()` 输入 **1–16000** 个连续样本及对应 `first_sample`；回调取得 4 个 Logit、样本可用时钟和完整词事件。回调时钟不是物理词尾时间。
4. PCM 丢包、重新打开录音、切换采集路径等真实断流必须调用 `en_v5_m43_discontinuity()`；新独立会话调用 `en_v5_m43_stream_reset()`。不能靠填充假音频来掩盖中断，也不能在并发线程复用同一个状态。
5. 严禁把现有研究回放工具 `mero43_host_cli.c` 中的 `max_gap_samples=1000000`、`cooldown_samples=0` 搬进产品：其 **62.5 秒**长窗口只用于离线数值回放，允许跨音段拼接 HEY/MEH/ROH，已经被 V6 声学审计识别为明显风险；真正的时间窗应依据未暴露 CAL 与连续自然负例选择并在 LOCKED 上验证。

模型权重/模型选用、解码政策、设备硬件身份必须作为三个**独立的版本化证据维度**。B3/B4 的研究结果无论改善多少，若未经过独立声学质量证据、对应权重的 C11↔ARM32 数值回放，以及物理 SSC305 实测，都不能自动覆盖本页 B0 装载路径。

## 5. 真实设备证据闭环

在真正板端具有可运行的集成应用、音频采集通道和经认证的 B0 pack 后，分别存档：

| 证据 | 必须包含 |
|---|---|
| 身份 | 板号/BSP/toolchain/sysroot、ELF/ABI、模型 pack SHA、应用 commit、PCM SHA |
| 配对一致性 | **同一原始 PCM** 在 host 与板端的逐帧 Token、事件计数和可用样本时钟；不能以 generic QEMU 替代 |
| 资源与实时性 | 真实设备 CPU、P95 端到端回调、峰值 RSS/PSS、栈、持续运行、耗电/温度以及采集链缓冲时延 |
| 声学质量 | 独立真人 3–5m 远场 FRR、长时间自然录音 FAR/h、近似词/其他背景挑战、不可用于调参的 LOCKED 数据 |
| 隔离 | 实际音频来源许可、声源组及会话隔离；合成开发样本及增益派生样本不能冒充独立真人或增加统计分母 |

现场观测可复用现有 `tools/collect_target_evidence.py` 和 `tools/collect_runtime_soak.py`，但必须提供实际采集值/原始记录，不可凭主机测试编造实板 PASS。当前状态：**单词 B0 SDK 软件层 PASS；B3 软件数值层 PASS；SSC305 实板、真实远场与自然 FAR/h = NOT_RUN**。

`research/english_v5_mero/SSC305_SDK_HANDOFF.md` 仍是具体 SDK API/构建交付参考；任何历史双词/旧 causal-head 说明都不得覆盖本文模型来源和研究边界。
