# KWS V3：R1 借用权重 SDK 与初始化峰值内存优化

日期：2026-11-11。继续保留 **R1-DEV-FSMN-20261107** 为主研究模型；本次落地的是同一模型的资源优化接入版本，没有重新训练或更换候选权重。

## 已完成的实际改动

原分类头 INT8 `par4` SDK 初始化时，调用方先读取完整模型，SDK 再分配并复制一份 1,960,140 字节的权重。新增显式 `fsmn_sdk_create_view` 直接借用已经验证的调用方缓冲区，消除第二份模型分配和拷贝。

原 `fsmn_sdk_create` 和 `fsmn_run` 保持复制语义；新 `fsmn_run_view` 明确使用借用接口。模型必须 8 字节对齐，并从构造开始到析构完成保持有效且不变。SDK 不释放调用方模型；frontend tables 继续采用原复制语义。具体有效类型、对齐和生命周期要求见 [SDK 说明](../research/r1_stream_c/borrowed_weights/README.md)。

模型包仍为 **1,960,140 字节**，SHA256 为 `63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4`。原 checkpoint SHA256 为 `99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571`。没有新增量化层，没有改变权重、前端、推理数学、解码规则或数据划分。

## 新候选的实际回放

复用原 134 条 PCM、380.0800625 秒音频和已保存的 `arm_par4` 对照结果。两种运行时分别运行新借用候选一次，原对照只读，不重新执行。

| 新候选运行时 | 完成音频 | 模型帧 | 完整词 proposal | 每词 proposal | 与保存对照不一致 |
| --- | ---: | ---: | ---: | --- | ---: |
| x86 GCC 14.2 | 134 | 12,534 | 88 | K1=44，K2=44 | 0 |
| ARMv7 GCC 13.3 / QEMU | 134 | 12,534 | 88 | K1=44，K2=44 | 0 |

比较覆盖每帧 argmax、完整词事件、源依赖时钟和投喂时钟；全部 F/E trace 逐字节相同。没有测量完整 logits 的逐元素相等。原复制构造和其后的全部计算、feed/reset/gap/destroy 函数另经源码逐字节比较保持不变。

独立复核只读取保存的输出并重新计算：292 个 claim 无重复，调用形状与计划一致，268 条新 trace 和 24 份初始化回执全部通过。见 [独立聚合审计](../research/r1_stream_c/borrowed_weights/INDEPENDENT_AUDIT_20261111.json)。

实际新增 **268 次候选音频调用**；这是 134 条输入的两种运行时，独立样本数仍为 134。原有 100 条 train-role、34 条已暴露开发记录保持原用途；其中原 unknown 样本不因本次回放获得标签。没有新训练、TTS、ASR、准入、自动重试或阈值扫描。

## 初始化内存实测

同一 x86 二进制分别选择 copy/view，每种 12 个独立进程，顺序交替。每个进程只初始化和析构一次，**零 PCM 投喂、零模型前向**。

| 指标（12 次中位数） | 原复制构造 | 新借用构造 |
| --- | ---: | ---: |
| `/proc/self/status` VmHWM | 4,864 KiB | 3,036 KiB |
| 初始化后的 VmRSS | 3,040 KiB | 3,036 KiB |
| 模型构造时间，不含文件读取 | 9.4723165 ms | 8.7787700 ms |
| SDK 新增的模型拷贝 | 1,960,140 字节 | 0 字节 |

**初始化峰值中位数下降 1,828 KiB，约 37.58%。** 初始化后的驻留内存几乎不变；本实验没有测量流式运行期的稳态 RSS，不能把这个降幅推广到持续推理。时间数据来自本次主机的少量配对测量，不构成目标板性能承诺。

原始 `getrusage().ru_maxrss` 约 14,572 KiB，受到父进程继承高水位影响，已原样保存并排除出收益结论；没有用它替代当前 executable 的 VmHWM。全部原始测量保留在所有者证据包，公开 [机器可读聚合](../research/r1_stream_c/borrowed_weights/OBSERVED_20261111.json)。

## 验证与交付

新增六项 focused tests 覆盖真实 loader 的虚构结构输入、1–7 字节错位、身份/数值/格式拒绝、失败前后状态不变、借用/复制所有权、全部分配失败、析构失败，以及 stdout/stderr 的 `/dev/full`。实际完整 SDK 与四个 runner 已编译并拒绝虚构模型。真实 loader 与构造测试通过 ASan/UBSan；本地 LeakSanitizer 因 ptrace 环境不可用，失败日志保留，未写成泄漏检查通过。分配与释放另有确定性逐例平衡检查。

现有 required CI 接入新增测试的 normal、`-O`、`-OO` 三模式，使用 GCC/Clang，normal 同时启用 ASan/UBSan。仅更新当前 CI 文件在原保留清单中的身份，历史 718 个文件和 834 条 provenance 不变。没有新增 workflow、Scheduled 巡检或模型任务。

离线工具链实际编译了 host 与 Cortex-A32 / NEON-VFPv4 / hard-float 的静态库、共享库及四个程序。ARM 为 ELF32 小端 EABI5，动态加载器 `/lib/ld-linux-armhf.so.3`，最高必需 GLIBC 为 **2.35**。QEMU 已运行新候选；真实 SSC305 必须使用其 BSP/rootfs 兼容工具链或 sysroot 接入。

复现入口为 [owner_build.py](../research/r1_stream_c/borrowed_weights/owner_build.py) 与 [owner_validate.py](../research/r1_stream_c/borrowed_weights/owner_validate.py)。已消耗的固定回放不应自动重启；应先复用保留的终态、源代码、输入身份和逐条输出。

## V3 剩余验收项

强 FSMN 研究模型、有限初始化确认、C/混合量化 SDK、ARM/QEMU 回放和本次初始化内存优化均有实际产物。独立真人 FRR、有效连续自然负流 FAR、连续阈值校准、最终 AFE 词尾延迟，以及真实 SSC305 的 ABI、CPU/RSS、功耗、温升和长稳仍须实际测量。

本次没有改变 Natural73 的全 unknown 状态或增加自然负例秒数。原 FP32 对 Torch/MKL 数值门槛保持 **FAIL_UNCHANGED**，当前路线保持独立 Reference64/既有混合量化的声明范围。研究 SDK 输出仍是未校准的完整词 proposal，不能把回放相等或 CI 成功解释为完整 V3 声学和板端验收完成。
