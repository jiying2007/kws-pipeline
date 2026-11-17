# 英文 KWS 的 SSC305 接入入口

本页复用现有 SDK 和只读 BSP 探针。它不提供设备地址、凭据或替代工具链，不代表已经在 SSC305 上运行。当前英文模型仍是弱研究候选；目标板接入与声学准入分别记录。

## 已有入口与当前缺口

- `research/arm_runtime/scripts/probe_sdk_readonly.sh`：在 SDK 主机上查询显式指定的真实编译器，并静态检查复制回来的目标 ELF。
- `research/arm_runtime/scripts/probe_board_readonly.sh`：在真实板上读取 kernel、CPU features、libc、loader 和已有 ELF 信息。
- `research/english_stream_integration/CMakeLists.txt`：可使用显式 BSP 编译器与 sysroot 构建现有英文 SDK；已校验冻结的 frontend/head/generated 源字节。
- 安装后的 `bin/ekws_pcm_stream`：消费 mono 16 kHz PCM16，输出事件 JSONL，没有 EOF 补零。

当前核验源为 kws-pipeline `2e2151aaae54211f6ea1faf1287c002a5ebee0ad`、audio-pipeline `379d6f88bcfc467b88ad31c2a924221d8349a806`。仓库中可见的 audio-pipeline `lab/examples/board.ssc305.example.json` 含有待替换项；它不是可用设备配置。现有 generic ARMhf/QEMU 产物不能证明实际 SSC305 BSP 兼容。尚缺实际 BSP 编译器/sysroot、板端 ELF/loader 证据和可访问设备入口。

## 1. 从真实 BSP 获取身份

在已具备真实 BSP 的主机上提供以下变量。不得从 generic Ubuntu ARM 工具链推断这些值。

```sh
: "${KWS_PIPELINE_SOURCE:?Set the absolute checked-out kws-pipeline path}"
: "${KWS_BSP_CC:?Set the actual absolute BSP C compiler path}"
: "${KWS_TARGET_ELF_COPY:?Set a copied executable from the actual target}"
sh "$KWS_PIPELINE_SOURCE/research/arm_runtime/scripts/probe_sdk_readonly.sh" \
  "$KWS_BSP_CC" "$KWS_TARGET_ELF_COPY" > sdk-inventory.txt
```

在真实目标板上运行原有 `probe_board_readonly.sh`，另保存必要的内存与调频上下文：

```sh
sh probe_board_readonly.sh > board-inventory.txt
{
  if [ -r /proc/meminfo ]; then
    grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached):' /proc/meminfo || :
  fi
  for KWS_CPU_PATH in /sys/devices/system/cpu/cpu[0-9]*; do
    for KWS_CPU_FIELD in scaling_governor scaling_cur_freq cpuinfo_max_freq; do
      KWS_CPU_FILE="$KWS_CPU_PATH/cpufreq/$KWS_CPU_FIELD"
      if [ -r "$KWS_CPU_FILE" ]; then
        printf '%s=' "$KWS_CPU_FILE"
        cat "$KWS_CPU_FILE"
      fi
    done
  done
} > board-memory-frequency.txt
```

这里只读取系统上下文，不调整 governor、不锁频、不安装依赖、不加载模型。记录板型、BSP 版本、实际 sysroot 来源，并检查 ELF32/ARM、浮点调用约定、动态解释器和 libc。原始输出可能包含本机路径或设备名称，作为现场记录保留；发布前只提取必要技术字段。

## 2. 用真实 BSP 构建现有 SDK

先按已有 artifact manifest 验证原始 fullchain generated 源与模型。为新编译选择独立空构建目录，保留旧 native/ARM 产物。

```sh
: "${KWS_PIPELINE_SOURCE:?}"
: "${KWS_BSP_CC:?}"
: "${KWS_BSP_SYSROOT:?Set the actual BSP sysroot}"
: "${KWS_BSP_CFLAGS:?Set the flags verified against the target ABI}"
: "${KWS_FULLCHAIN_GENERATED:?Set authenticated fullchain generated directory}"
: "${KWS_BSP_BUILD:?Set a new build directory}"
: "${KWS_BSP_INSTALL:?Set the SDK install directory}"
cmake -S "$KWS_PIPELINE_SOURCE/research/english_stream_integration" \
  -B "$KWS_BSP_BUILD" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_C_COMPILER="$KWS_BSP_CC" \
  -DCMAKE_SYSROOT="$KWS_BSP_SYSROOT" \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_C_FLAGS="$KWS_BSP_CFLAGS" \
  -DKWS_ENABLE_ENGLISH_RESEARCH_STREAM=ON \
  -DEKWS_PIPELINE_SOURCE="$KWS_PIPELINE_SOURCE" \
  -DEKWS_GENERATED_SOURCE_DIR="$KWS_FULLCHAIN_GENERATED" \
  -DEKWS_STATIC_RUNNER=OFF \
  -DCMAKE_INSTALL_PREFIX="$KWS_BSP_INSTALL"
cmake --build "$KWS_BSP_BUILD" --parallel 2
cmake --install "$KWS_BSP_BUILD"
sh "$KWS_PIPELINE_SOURCE/research/arm_runtime/scripts/probe_sdk_readonly.sh" \
  "$KWS_BSP_CC" "$KWS_BSP_INSTALL/bin/ekws_pcm_stream" > candidate-elf-inventory.txt
```

不硬编码 hard-float、NEON 或 CPU 扩展；这些参数来自实际 BSP。若真实 BSP 只支持另一种 libc/浮点 ABI，需要重新构建，不将主机库复制进板端以掩盖不匹配。上述构建不自动执行模型。

## 3. 先做同一 PCM 的目标板配对

模型文件使用已经验证的 SDK artifact 中 `models/causal-head.bin`，保持其哈希。现场通过已有授权传输方式放置 BSP 构建产物、模型及已保存的 mono 16 kHz PCM16 输入；不要把 generic QEMU runner 当作 BSP 构建结果。

```sh
: "${KWS_TARGET_RUNNER:?Set the BSP-built runner on the actual board}"
: "${KWS_TARGET_HEAD:?Set the authenticated causal-head.bin path}"
: "${KWS_TARGET_PCM:?Set the known PCM16 input path}"
"$KWS_TARGET_RUNNER" "$KWS_TARGET_HEAD" ssc305_pcm_pair < "$KWS_TARGET_PCM" > ssc305-events.jsonl
```

主机与板端消费同一输入、相同阈值和状态规程，记录原始事件、样本时钟、输出差异、退出状态及二进制/模型/输入哈希。录音 ID 的 `ssc305` 字样不是设备证明，仍需上述真实板端身份记录。此步骤不构成实时麦克风、FAR/FRR 或词尾回调验收。

## 4. 与完整性能基准及实时音频接续

后续使用独立的完整链 benchmark，报告 init、steady wall/process CPU、输入秒数、RTF、RSS/PSS 和服务时间。不要用 shell 外围 wall time 代替 accept 内部计时，也不要将 QEMU 时长当成 SSC305 性能。

实时麦克风接入复用 `ekws_stream_accept_pcm16()`：传入真实单调样本序号，在 owner 线程维护连续状态，以有界事件队列转交应用。只有真实断流才调用 discontinuity/reset。reset 后首个 ready head 需要 35,200 个真实样本（2.2 秒）；开机即说、恢复和换音路单独记录。

直到板端实际运行与记录完成，状态保持 `ssc305=NOT_RUN`。工具链库存检查、交叉编译通过和软件事件配对各自报告，不自动升级为独立声学或实板稳定性通过。
