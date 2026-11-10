# R1 FSMN 资源探针：ARM 构建记录（2026-11-10）

已为固定 R1-DEV-FSMN-20261107 的 Reference64、分类头 INT8 scalar 和 INT8 par4 三套 SDK 构建离线资源测量程序，并验证 ELF、动态依赖和可复现构建。此次工作没有执行目标程序，没有产生模型效果或设备性能数据。

## 源码绑定

测量入口为 `research/r1_stream_c/quant_head8/target_resource_bench.c`。

- 实际构建源码：[PR #779 初始提交 c2f453a5](https://github.com/jiying2007/kws-pipeline/commit/c2f453a50b0b66608a5d57055d25dd8b44698701)。
- 后续核验：[PR #779 提交 a4acf0cf](https://github.com/jiying2007/kws-pipeline/commit/a4acf0cf795947238b46bc9a7951ff3b833eda8f)。
- 两个提交中的该 C 文件逐字节相同，SHA256 均为 `8e00ecd232b56eabddf5e4613d278596dec65d5d4f9ad811937e6d408062e7d6`。

该入口已包含资源回执写入错误修复：完成 JSON 输出后检查 `fflush(stdout)` 和 `ferror(stdout)`，写失败返回非零。构建没有改变模型权重、解码器或任何声学验收门槛。

## 三个实际构建产物

| 变体 | ELF 文件字节数 | SHA256 |
| --- | ---: | --- |
| Reference64 | 33,280 | `06240db83c50e0fe67671c838e3d58acafb56f5882ef247a374455b7fd6c5419` |
| INT8 scalar | 33,288 | `044a921e6c89d18b678a871ef396dbd39860cf4b0fd6ded57f513ad5a16e37c5` |
| INT8 par4 | 33,288 | `f9be3e1185577d728b6adf2cc97ac0d89a7c3df07badcaca487e92432ca366f5` |

三者都是 ELF32、小端、ARM EABI5 hard-float。工具链为 Ubuntu GCC 13.3.0，目标选项为 `-mcpu=cortex-a32 -mfpu=neon-vfpv4 -mfloat-abi=hard`，保留 `-fno-fast-math -ffp-contract=off`。

动态加载器为 `/lib/ld-linux-armhf.so.3`；依赖为 `libm.so.6`、`libc.so.6`、`ld-linux-armhf.so.3`。三个程序最高必需符号版本均为 **GLIBC_2.35**。这需要与 SSC305 实际固件的加载器和 libc 核对；当前没有板端 ABI 适配通过的结论。实际固件采用不同或更旧 libc 时，应使用其兼容工具链和 sysroot 重新构建。

第一次构建和交付脚本在新目录中的独立构建，使用相同主机及固定工具链，三个程序均逐字节一致。此结论仅覆盖这些编译输入和构建条件。

## 既有工具链与离线复现

复用以下已经存在的 GitHub Actions 制品，验证外层 ZIP 和内层包的 SHA256；没有新建工作流或重跑这些任务：

| 原运行 | Artifact | 内容 |
| --- | --- | --- |
| [42801676506](https://github.com/jiying2007/kws-pipeline/actions/runs/42801676506) | 13755687974 | C 交叉工具链和 ARM 运行库 |
| [42802555925](https://github.com/jiying2007/kws-pipeline/actions/runs/42802555925) | 13757362176 | 编译器前端、glibc/Linux 开发包 |
| [42811783343](https://github.com/jiying2007/kws-pipeline/actions/runs/42811783343) | 13762189802 | binutils 包及所需主机库 |

交付脚本 `build_arm_resource.py` 使用 Python 3.12+ 和 `dpkg-deb -x` 离线恢复这些工具链输入，再校验固定 C 源码及三套 SDK 的六个头文件/静态库。它只执行构建主机工具，不执行编译出的 ARM 程序。

交付包保持原字节，重新检查如下：

| 包 | 字节数 | SHA256 | 完整性回读 |
| --- | ---: | --- | --- |
| `KWS_R1_ARM_RESOURCE_PROBES_20261110.zip` | 142,286 | `0ca4b8671be786cad0af07a9daaac1b44fda4af81cfc00b75015feb840c7ddcf` | 74 个载荷的大小/哈希及完整成员集合通过；加清单共 75 成员，ZIP CRC 通过 |
| `KWS_R1_ARM_TOOLCHAIN_INPUTS_20261110.zip` | 65,689,464 | `1a7f2c4eff08550052d9ccda1fe78aca0a87fa47a33c28e71e85757138f70f49` | 3 个原始 Artifact ZIP 的大小/哈希及完整成员集合通过；加来源清单共 4 成员，ZIP CRC 通过 |

资源探针包包含源码、脚本、程序、SDK 头文件/静态库、来源清单、构建回执、readelf 日志和许可说明。工具链输入包保存三个原始 Artifact ZIP，供远端制品过期后继续复现。两个包均不包含模型权重或录音。

## 结论边界

本次模型前向、QEMU 执行和物理板执行均为 **0**；未打开所有者归档中的权重成员，未读取 PCM。复用的 SDK 静态库和头文件已逐个按其原清单核验。

尚未测量 SSC305 的实际 ABI/rootfs/BSP 兼容性、CPU/RSS、功耗、栈高水位、温升与长稳，也没有最终 AFE、连续负流 FAR、独立真人 FRR 或人工词尾延迟结果。通用 ARM 构建通过不改变这些状态。

公开机器可读摘要为 [ARM_RESOURCE_BUILD_20261110.json](ARM_RESOURCE_BUILD_20261110.json)。实际板端实验范围和资源指标定义继续遵循 [TARGET_RESOURCE_PROFILE.md](https://github.com/jiying2007/kws-pipeline/blob/a4acf0cf795947238b46bc9a7951ff3b833eda8f/research/r1_stream_c/quant_head8/TARGET_RESOURCE_PROFILE.md)。

