# 英文共享前端执行与模型完成边界 — 2026-11-16

> **真实执行和独立读回已完成。** 本次共享前端 worker 为 `EXECUTED_BACKBONE_WITHIN_PREDECLARED_LIMITS`，独立标准库读回为 `VERIFIED_RECOMPUTED_BACKBONE_PASS`，六组比较全部通过。完整英文模型仍未交付：当前目标的关键词头未训练，L1 未完成，L2/L3 未运行。配套 [JSON](research-v4/en-backbone-execution-20261116.json) 保存实际身份、逐阶段统计、调用计数和独立校验身份。

## 完成等级沿用 V4，工程阶段不替代完整模型

| 等级 | V4 含义 | 当前目标状态 |
| --- | --- | --- |
| L1 | 可复现研究源码、实际关键词权重和实际推理 | **未完成**：本轮训练门槛失败，0 optimizer updates，无新关键词头权重。共享前端的静态导出和单独数值验证只是局部工程证据。 |
| L2 | 冻结完整配置后的独立声学确认 | **NOT_RUN**：无独立每词 FRR、联合连续 FAR 或 word-end callback 结果。 |
| L3 | 真实 SSC305 BSP、音频链路、资源、延迟和稳定性确认 | **NOT_RUN**：host native / QEMU ARM 不等于物理目标板。 |

定义来自 [V4 执行基线](KWS_V4_EXECUTION_BASELINE_20261114.md)。当前目标仍是 **Hey Mero `/heɪ ˈmiːroʊ/`** 和 **Hey Vosen `/heɪ ˈvoʊzən/`**；机器转写、合成 preset 或代码测试不构成人工发音确认。

## 已核实的数据状态：固定最低量未通过

数据侧 [PR153](https://github.com/jiying2007/kws-data/pull/153) 已保留实际终态和禁用的后继源码，main 为 `393b03e41eff68ce91a1943b18b3da9641378515`。其固定版本 [读回报告](https://github.com/jiying2007/kws-data/blob/393b03e41eff68ce91a1943b18b3da9641378515/docs/ENGLISH_TERMINAL144_READBACK_20261116.md) 与 [机器可读记录](https://github.com/jiying2007/kws-data/blob/393b03e41eff68ce91a1943b18b3da9641378515/docs/research-v4/en-final144-terminal-readback-20261116.json) 是本节依据。

- 原 source run `43816869386` / head `8286ee743a2690d168c3e56f5b31ccc64ae490ed`：144 次声明/首次尝试，143 份有效原 WAV，1 条 `FAILED_NO_RETRY`；原 campaign 仍为 `INCOMPLETE_NO_RETRY`，原 observe/train jobs 仍为 skipped。
- 独立 observer run `43822597212` / head `21d807461c0afa30037362ddde7ea527d87cf3e6`：四路各完成 143 次，共 **572 次新观察**（429 ASR、143 XLSR），没有新 TTS 或优化更新。
- 不改 policy `27a18716b5c0e7f29a937eeedde085917c04d4d45f30db0d9115dc3c881a9fd5`，得到 **74 eligible requested-intent 研究行、70 UNKNOWN**；UNKNOWN 保留原生成失败，不改成负例。

| 原角色 | 普通句研究行 | Vosen 研究行 | Mero 研究行 |
| --- | ---: | ---: | ---: |
| train | 32 | **5，低于 8** | 10 |
| development | 16 | 3 | 8 |

Train Vosen 的 2 个 source families 已满足 families 门槛，缺的是行数。其余最低量组通过，但第一道总门 `MINIMUM_AFTER_DEDUPLICATION` 已失败。纯 saved-evidence audit 为 0 模型调用、0 optimizer updates；actual feature extraction 和第二道门 `MINIMUM_AFTER_FEATURE_GEOMETRY` 均为 **NOT_RUN**。第二道门仍是冻结代码中的必经步骤，实际特征窗口支持不足还可能排除输入；**不能把当前 5 与 8 的差额解释为“补三条就保证能训练”**。关键词训练、实际 saved-head/full-chain parity 均未运行，新的 keyword weights 未产生。训练配置仍 disabled；源码成功路径不等于实际完成。

源账本 canonical SHA-256 为 `b6e157eea8eeea38f9261d8756ab43d6570c35a6ef53b892436b7c97424efb38`，固定 selection canonical SHA-256 为 `f51849b945cd49c4a726ce79a3d22b656408c9088532b6376d829a10c2baf6b2`。Vosen 的 `/z ɪ n/` 不替代 `/z ə n/`；原 split、角色和消费过的实验身份不变。

## 已完成的共享前端静态导出

[静态导出 run 43823297841](https://github.com/jiying2007/kws-pipeline/actions/runs/43823297841) 在 `cfc69dcd7535ea909403f10a91bcfe26ccc1b331` 完成 `EXPORTED_STATIC_GRAPHS_ONLY`，模型推理调用为 0。原 artifact `14205188962` 为 **2,722,450 bytes**，ZIP SHA-256：

```text
1ffc968da201e07186929f05d9d8c651e18390a92f5436b6cdfec3a3cb1e9ac0
```

实际输出是两份固定图的 C 常量、描述符及导出 manifest。保留其原始身份及此前失败历史；不以静态成功替代数值相符、关键词 head 或 SSC305 结论。

## PR826：实际共享前端数值执行已完成

[实际 run 43829765872](https://github.com/jiying2007/kws-pipeline/actions/runs/43829765872) 在 head `cd3e07ebc25505648d10166a806a05afade37703`、attempt 1 完成，`source-contract` 和 `verify-backbone` 两个 job 均为 success。Artifact `14208696183`（`english-backbone-parity-43829765872`）为 **27,306,350 bytes**，ZIP SHA-256：

```text
dfe28da9c5a35fef3768f7958ff7945e587645554d838dcbb4aa1d0508a9db1f
```

保存的 RESULT 状态为 `EXECUTED_BACKBONE_WITHIN_PREDECLARED_LIMITS`；`COMMAND_EXIT` 为 0，存在 RESULT，无 FAILURE。RESULT 原字节 SHA-256 为 `75d48edbd790f17c55c887b4fb4af9980bdeea3f3925530183822eac89f63737`，MANIFEST 为 `7b744619f7d40f543dc1e42348963883b12061050ecb44219ded29ad647a7fa7`，记录结果目录内其余 49 个文件。原始 API、源码绑定、逐例 comparison、九份 trace 和具体最坏位置均由 JSON 保留对应身份与统计。

冻结 profile 为 `FIXED_SHARED_BACKBONE_ONLY`，PLAN SHA-256 仍为 `5657e4387909bf7b47ba4ed7adec723005a8428357bd75d3515c3ddfe26e550b`；实际 worker SHA-256 为 `5f077922dbebf018a026a96340e417464bcc71e248c36d9990a7de6e0933a376`。消费的是原静态生成字节与两份固定 ONNX。没有重新导出、训练关键词 head、TTS/ASR 或真实语音输入。

### 实际计数与数值统计

三段 zero / sine400 / lcg14 输入各 48,000 samples。ORT 实际 **156 次 attempts、156 次 completions**，完成 **78 个 frontend windows**；native 与 ARM 各 **3 次 process attempts/completions、78 个已验证 windows**。head evaluations、optimizer updates、新 TTS/ASR/phone calls、real-audio cases 均为 **0**。这些是本次终态 accounting，不是原静态或下载阶段的零调用记录。

每条完整 trace 为 **586,984 bytes**，包含 `EFFTR01\0` 和逐 embedding 的 stage 1/2/3：3104/2432/96 个 little-endian float32。worker 实际核验每例 26 个时钟 `16000 + 1280×k`（k=0…25，末点 48000），没有 stage 4 或 EOF 追加记录。每例新建流，chunk 固定循环 `1,1279,257,4096`。

| 人工信号 | C backend | raw mel 最大绝对误差 | normalized mel 最大绝对误差 | embedding 最大绝对误差 | 超限元素 |
| --- | --- | ---: | ---: | ---: | ---: |
| zero | native | 7.629395e-6 | 9.536743e-7 | 3.862381e-5 | 0 |
| zero | ARM/QEMU | 7.629395e-6 | 9.536743e-7 | 3.862381e-5 | 0 |
| sine400 | native | 5.741119e-4 | 5.769730e-5 | 4.186630e-4 | 0 |
| sine400 | ARM/QEMU | 5.741119e-4 | 5.769730e-5 | 4.186630e-4 | 0 |
| lcg14 | native | 2.288818e-5 | 1.907349e-6 | 7.843971e-5 | 0 |
| lcg14 | ARM/QEMU | 2.288818e-5 | 1.907349e-6 | 7.843971e-5 | 0 |

六组分别对 ORT reference 比较，18 个阶段统计共 878,592 次元素比较，超限总数为 0。raw mel、normalized mel、embedding 的 `(atol, rtol)` 仍分别为 `(0.01, 2e-5)`、`(0.001, 2e-5)`、`(0.001, 2e-4)`，未按结果放宽。最大 tolerance ratio 约 **0.1588981**。非零 reference 的最大相对误差约 **1.09036%**，出现在 lcg14 的 embedding、sample 39040、component 0，其 reference 接近零；它仍满足预声明的绝对加相对误差规则。JSON 保留各阶段最大绝对/相对误差与位置、违反数、zero-reference mismatch 和 ratio，不能单看相对百分比判断失败，也不由这些通过项推及任意语音。

### 实际耗时与缓冲区，不是实时或 SSC305 预算通过

| 3 秒人工输入 | ORT process CPU 秒 | native 整进程 CPU / wall 秒 | ARM/QEMU host CPU / wall 秒 |
| --- | ---: | ---: | ---: |
| zero | 0.111943 | 5.931988 / 5.932696 | 118.015903 / 118.028081 |
| sine400 | 0.109969 | 5.971936 / 5.972632 | 116.772510 / 116.781238 |
| lcg14 | 0.110420 | 5.990518 / 5.991225 | 116.195338 / 116.202802 |

ORT timing 排除 session 初始化和文件 I/O；C 包含进程启动、输入及 trace I/O，ARM 还包含 host QEMU。**带 trace 的 native 整进程 CPU 为 5.93–5.99 秒 / 3 秒输入，QEMU 为约 116–118 秒；本次不支持实时性或 SSC305 性能通过的声明。** 这些不同口径记录不能直接当作部署后处理速度比较。

两个 C backend 的 `workspace_bytes` 都是 **528,384**；`frontend_state_bytes` 为 native **32,080**、ARM **32,056**。这两个量是缓冲区与结构体，不是峰值 RSS 或完整 RAM；常量、可执行映射、输入缓冲、栈、libc 和模拟器开销不能据此省略。运行环境保存 numpy 1.26.4、onnx 1.17.0、onnxruntime 1.20.1、GCC 13.3.0、QEMU 8.2.2；未测物理 SSC305。

### 独立 saved-artifact 读回

独立标准库读回结论为 **`VERIFIED_RECOMPUTED_BACKBONE_PASS`**，`issues=[]`。核验覆盖原 ZIP 的 61 个成员、49 个结果文件的闭合 manifest、原静态导出、来源/计划/pins、三段 PCM、全部九份 trace 的 1,317,888 个有限 float32 和精确时钟，以及六组比较的 878,592 对元素。逐阶段最大误差、对应位置、相对误差、容差比例和零参考值统计与原保存报告全字段一致；native 与 ARM 的三份对应 trace 原字节也完全一致。

独立报告 `readback/independent-readback.json` 为 52,844 bytes，SHA-256 `861b7487e897583b9f299d963d8ff3d2cd94f374c3cb92e50fd9aa8399c3f531`；复算脚本 `readback/independent_backbone_readback.py` 为 32,166 bytes，SHA-256 `f3f8e8881ccd4a32e7a14b59c24cc6bee3045f3e56d7e98aab10359fa114202e`。读回没有导入生产 worker 或 ML 库，没有运行外部进程、C 或模型。原 artifact 未包含两个 ONNX 原文件，资产复核范围准确限定为原 acquisition receipts、固定静态导出身份和保存的生成字节。

本次 `head_ready=0/events=0` 只反映没有启用 head，不是触发覆盖、FRR/FAR 或词尾延迟测量。成功范围止于三段固定人工输入的共享前端数值与样本时钟；完整当前目标 L1、独立声学 L2 和 SSC305 L3 均未完成。

## 已交付但尚未审听的 mixed95 输入

独立混排审听包已单独交付给用户。公开记录只保留文件名 **`english_wake95_blind_review_20261116.zip`**、大小 **7,085,989 bytes** 及 SHA-256：

```text
0fb0a784d01cd81480d80e32fa1c6d4332f63987017118a6beedb1d40815a769
```

包覆盖全部 **95 条有效 wake-intent（48 Vosen + 47 Mero），共 97.92 秒**。96 条声明中生成失败的 1 条只保留原处分及 started/failed，无假音频。95 份原 16 kHz WAV、95 份 native WAV、192 份逐条 capture 收据均保留原字节；旧 48 条包未改。

第一遍采用固定 PCM-hash 匿名顺序，只提供自由听写；页面源码和内嵌 JSON 也不含目标词、声线、角色或机器结果。第二遍每条均提供 Mero/Vosen 两组独立字段及 Hey、重音、元音长度和不确定性。全部答案、审听者和时间初始为 `null`；离线页面没有外部资源、上传或自动准入。这个输入包不能消除既往提示，也不是已经完成人工审音的证据。

## 外部输入与后续边界

当前目标 L1 仍需要满足明确来源、角色、variant 和证据协议的可用研究输入，通过去重后与实际特征几何后两道冻结最低量门，以及真实保存的关键词权重和同权重推理/读回。人工审听需要真正的审听者身份、时间及独立判断；填写 95 条审听表不自动改变冻结 selection、原角色或 144 分母，其结果不能自动重写已消费实验的机器规则、最低量，也不授权补额、重试或追加优化。

L2 需要独立真人发音/唤醒与连续背景录音，在完整配置冻结后按原 V4 口径确认每词 FRR、联合 FAR、word-end callback 延迟及分层结果。L3 需要真实 SSC305 硬件和 BSP/音频链路的实测。既有 guards、pins、selection 及各自固定 PLAN 均不因本文改变。后继实现须使用独立的源码与执行记录，本文所记录的原结果保持原样。
