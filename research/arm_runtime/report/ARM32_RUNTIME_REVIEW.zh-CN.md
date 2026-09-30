# ARM32 sherpa PCM 候选：静态核验与 SSC305 接入前置条件

日期：2026-09-30 UTC。状态：**仅本地研究和静态审计，等待真实 SSC305 SDK/BSP；未完成 ARM 适配器构建、链接、加载或板端运行。**

## 结论

官方 sherpa-onnx 1.13.8 有可下载、可固定 hash 的 Linux ARM32 hard-float shared 包。其 C API 与 ONNX Runtime 两库可形成不依赖 ALSA/PortAudio 的直接动态库闭包，但有两个不能忽略的接入门：

1. 两库实际要求 GLIBC_2.34、GLIBCXX_3.4.29；C API 另要求 CXXABI_1.3.13。SSC305 的真实 libc、工具链、sysroot 仍未知。ARMv7/NEON 相同不代表 ABI 可用，通用 Ubuntu ARM CI 也不能替代 BSP 证据。
2. 官方 C API 库还静态包含 eSpeak 实现，固定来源带 GPLv3 COPYING；不是只含 Apache-2.0 KWS 的小库。仅删除 TTS/音频 CLI 不会移除已链接代码。发布前需按实际组件审查许可，或用实际 BSP 重新构建明确关闭 TTS 的候选再复核。

因此现在不下载通用 106.5 MB Arm 工具链，不为未知 BSP 提供“已兼容”二进制。当前交付的是可复算静态证据、锁清单、模型来源配方及只读取证脚本。

## 已实际验证

- sherpa-onnx release：v1.13.8，源码提交 `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`
- [官方 ARM32 shared asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-linux-arm-gnueabihf-shared.tar.bz2)：32,556,415 B；SHA256 `832c200eb361e361c587812f116436760e7728d2ee93ea4759ad5d59ab5c2e6c`，与 GitHub asset digest 相同
- 两库均声明 ELF32 / little-endian / ARM / EABI5 / hard-float；ARM attributes 为 v7 / VFPv3 / NEONv1 / VFP 参数寄存器。此为 ELF 声明，不是 CPU 执行测试或完整反汇编验证
- 编译器 `.comment` 记录为 Arm GNU 11.2-2022.02 的 GCC 11.2.1 20220111
- DT_RPATH 均为 `$ORIGIN`；SONAME 与文件名一致
- 对两个真实库重新执行 `audit_arm_runtime.py` 通过；8 个解析/错误 hash/符号链接/锁 schema/版本上限单测通过；两个 shell 脚本 `sh -n` 通过
- 没有下载或运行声学模型，没有 ARM 编译/链接/加载、QEMU、声学实验或 SSC305 性能结果

符号需求来自 `readelf -V` 的版本需要表，并用动态符号表交叉核对。例如 C API 要求 `pthread_once@GLIBC_2.34`、`basic_string::reserve@GLIBCXX_3.4.29` 和 exception_ptr 的 CXXABI_1.3.13 符号。只看版本字符串“最大数字”不是完整加载判定；实际 SDK 须提供全部需要的符号及匹配 ABI。

## 实际体积与依赖边界

| 组件 | 实际文件字节 | SHA256 |
|---|---:|---|
| libsherpa-onnx-c-api.so | 5,101,612 | db57b8ab6136016c75f2becebb81159a6dcbdad875e16d8aef6d5aa5a5209969 |
| libonnxruntime.so | 20,617,061 | 4636872e9b50f985b7ecaa55a6829dee81bdb96f77010b5bc0a4059e51d872dc |
| 两库合计 | 25,718,673 | 由上两项求和 |
| 既有固定 INT8 encoder + FP32 decoder/joiner + tokens | 5,737,545 | 四文件 hash 见 model-fetch-recipe.json |
| 既有双关键词文件 | 75 | 需沿用冻结内容，当前未重新打包 |
| 以上选择项总和 | 31,456,293 | 不含尚未构建的 adapter |

该总和是已存在原始文件的算术清单，不是完成后的 SDK 包大小、RSS 或闪存占用保证。还不含 adapter/probe/调用程序、系统依赖、许可证/notice、调试符号裁剪差异和文件系统开销。两库仍包含非 KWS 代码；进一步裁剪后的大小没有测量。

完整 release 的 46 个普通文件合计 68,902,890 B。其中可不随 PCM-only 接入一起携带：全部官方 CLI、ALSA 两文件、C++ wrapper、cargs 库与头。实际需要的直接闭包为：

- 自研 `libkws_sherpa_pcm.so`（尚未 ARM 构建） → `libsherpa-onnx-c-api.so` → `libonnxruntime.so`
- 系统供应：`libm.so.6`、`libstdc++.so.6`、`libgcc_s.so.1`、`libc.so.6`、`ld-linux-armhf.so.3`

这里的“可不携带”只指已检查 DT_NEEDED 所需动态闭包，不证明删除库内静态第三方组件、没有 dlopen 路径或完成法律审查。不要使用宿主 `ldd` 来执行/加载未经验证的 ARM 文件。

## 供应链与许可

1. sherpa 顶层和已固定 C API 头为 Apache-2.0；官方 shared tar 不含 LICENSE/NOTICE/头文件，不能原样当完整合规 SDK。固定头已在 kws-pipeline 的 `research/sherpa_pcm/vendor/` 中保存，其 hash 受现有锁约束。
2. 上游 sherpa 固定 [ARM ORT 配方](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/cmake/onnxruntime-linux-arm.cmake) 指向维护者发布的 [ORT 1.28.2 ARM zip](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.28.2/onnxruntime-linux-arm-1.28.2.zip)。这不是 Microsoft 官方二进制发布的保证。
3. 该 zip 实际 8,616,811 B，SHA256 `a8b4a3f1c338798c52d15c6b10b5eed0174cc3f485059948ac3ef456e824d342`，与 sherpa 固定配方及 release digest 一致。内含库与 sherpa 包的 ORT **逐字节一致**。内含 VERSION_NUMBER=1.28.2、GIT_COMMIT_ID=`33ca9628233dc8f002435e868d4c2e9f82766ca1`。这是发行包自报源码 ID，未独立重构建验证。
4. ORT zip 包含 Microsoft MIT LICENSE（1,073 B）、ThirdPartyNotices.txt（325,054 B）及其他元数据，文件 hash 已记录。验证其存在不等于完整的最终产品法律审查，sherpa 的其他静态依赖不由这份 notice 自动覆盖。
5. 固定 sherpa CMake 默认 `SHERPA_ONNX_ENABLE_TTS=ON`；其 ARM build recipe 未关闭，且 [eSpeak CMake](https://github.com/k2-fsa/sherpa-onnx/blob/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/cmake/espeak-ng-for-piper.cmake) 显式临时设 `BUILD_SHARED_LIBS OFF`。实际 C API 库普通符号表存在本地已定义 `espeak_Initialize`、`espeak_Synth`、`espeak_ng_Initialize` 等函数。固定 [eSpeak 源码 COPYING](https://github.com/csukuangfj/espeak-ng/blob/ed530aa113046142eb5115cf2fc9157854d0ffe1/COPYING) 是 GPLv3。不能用“没有独立 libespeak.so”或“只调用 KWS”否认静态组件存在。
6. 自研源码、模型、runtime/toolchain 是不同许可/供应链对象。模型这里只记录原始 ModelScope revision/hash 的 fetch 配方；未重新下载、上传或公开重分发模型。精确再分发义务需由发布方按计划分发方式审核，本报告不作最终法律结论。

GPL 不禁止商业使用；是否保留该组件以及适用的许可义务，须按具体分发/集成方式审核。此处不把 GPL 表述为“商业不可用”。

建议的后续路径是得到真实 SDK 后，用固定 sherpa/ORT 源码与经审依赖重建，关闭 TTS、PortAudio、WebSocket、Python、JNI、无关示例及服务模块；不能仅靠顶层一个开关就承诺整个链接闭包已最小。重建后重新检查符号、库依赖、许可证及数值/声学行为，不沿用本次 release 的二进制 hash 作为新包认证。

`kws-only-build-recipe.json` 已明确记录固定源码、实际 SDK/sysroot 占位、BSP ORT include/lib 环境变量、逐项 OFF 的 CMake 配置和后续构建命令模板，**尚未执行**。名字中的 KWS-only 是预期接入范围，非已证明无全部非 KWS 代码。上游会处理其他依赖，必须先审查/预取，不能把这份模板当成离线依赖已齐备。

`check_native_gates.py` 可对后续未 strip 的真实构建进行两个必要门：解析已定义符号以查 eSpeak/phonemize；按显式传入的 GLIBC/GLIBCXX/CXXABI 上限作数值比较。没有 `.symtab` 会拒绝判通过，防止把 strip 后“看不到符号”当作组件消失。仍需源码/链接 map/notice 复核，符号缺席不是法律证明。

当前官方包在 no-eSpeak 门**按预期失败**（exit 1），即使版本上限设到其实际要求也失败；另用 2.31/3.4.28/1.3.11 演示阈值拒绝，报告明确这是示例上限，**不是 SSC305 的已知版本**。两份实际检查结果分别在 `pinned-runtime-no-espeak-gate.json` 和 `older-sdk-threshold-example.json`。

## SDK 取证最小输入

已搜索 kws-pipeline `766532c7184f1cdc85e76ac12ce457efac9c69c3` 的 docs/configs/commercial/CI 及既有本地 ARM feasibility、native/resource receipts。没有真实 SSC305 SDK/BSP/toolchain/sysroot 记录，只有硬件描述及 Ubuntu 通用 ARM CI。

需要用户给：真实 SSC305 SDK/BSP 发布版本或来源/压缩包、交叉 C/C++ 编译器与 sysroot；若现成已有 board 输出，补充 kernel、libc 家族/版本、ELF loader、hard/soft-float 和 NEON 特性即可。此时不要求跑模型，也不要求板卡先跑声学验收。

宿主只读取证：

```
sh scripts/probe_sdk_readonly.sh /absolute/trusted/SDK/bin/target-gcc /path/to/copied-target-busybox
```

板端可选只读取证：

```
sh scripts/probe_board_readonly.sh
```

脚本仅向 stdout 打印有限硬件/编译器/libc 元信息，不修改板端配置，不联网，不枚举凭据，不加载候选 runtime，不采集音频。SDK 脚本会执行用户明确提供的可信 SDK 编译器的只读查询选项；不会执行 target-ELF。返回结果含本地路径/设备元信息，公开前需复核。

## 模型/实时接入与后续验收契约

保持已合并 PCM adapter：mono 16 kHz PCM16；320 sample（20 ms）staging；单 CPU 推理线程；80维特征；maxpaths=4、score=1、threshold=0.25、trailing blank=1。create 加载一次，reset 仅重建 stream，finish 处理剩余样本并 EOF，不额外补静音。调用单线程、不重入，回调字符串只在回调期间有效。

现有 adapter 无显式 feed 文件 I/O，但上游可分配内存和缺页；不具备默认 C 产品的 allocation-free 实时保证。最低端侧包后续需含 C 头、ARM adapter、经审 runtime、独立模型配方/校验、启动/关闭 smoke、可测量的 PCM replay/采集接口与全部许可证；现阶段不假装这些已构建。

收到 SDK 后按顺序：

1. 对真实 sysroot 检查 ARM ABI、loader、全部符号版本/符号和 C++ runtime；遇到 glibc 不匹配按 BSP 重建，不往板上混装另一套 libc
2. ARM model-free lifecycle：独立 stub 构建，若已有兼容 QEMU 可运行；区分仅编译/链接通过和真实测试执行；根 CI 不会自动覆盖本 optional adapter
3. 不带模型的真实库加载/版本 smoke；确认最终链接 closure、源码/工具链/选项/hash/许可记录
4. 使用已授权冻结 PCM 数据做同语义 parity；这是后续工作，本任务没有新跑音频实验
5. SSC305 板端验收：分开记录启动 load wall/CPU 时间、读字节/系统调用/major/minor faults；预热后 steady reads、faults、RSS/峰值、分配与长期漂移；按单核归一 CPU-seconds/audio-second，decode p99/max、20 ms 音频 ingress deadline/队列 overrun；覆盖并行视频/ISP/音频 DMA 和 memory pressure。阈值须由真实产品预算冻结，当前不虚构 pass 数字

现有 x86 的 RTF/RSS/42 条回放结果只作为历史参考；此报告没有把它们换算成 A32 性能或推断 BSP 可用。

## 复算与停止条件

```
python3 scripts/audit_arm_runtime.py --library-dir /path/to/reviewed/unpacked/runtime/lib
python3 -m unittest discover -s tests -v
sh -n scripts/probe_board_readonly.sh
sh -n scripts/probe_sdk_readonly.sh
```

`runtime-arm32.lock.json` 是固定二进制的可复算清单，不是面向任意包的安全证明。审计器硬编码两份已审文件 hash/字节数，拒绝空锁、少库、重复库、非固定 basename、路径穿越和非匹配 hash；版本需要只取 `Version needs` 段，不把库自身定义的版本算作依赖。初次本地审计时未改远端仓库；本目录归档静态证据和源码，不包含运行库/模型/工具链；两项二进制归档下载合计 41,173,226 B（另有小型来源/元数据文本），未下载通用 toolchain。等待真实 SDK 是构建的明确依赖；静态审计完成不等于端侧交付完成。
