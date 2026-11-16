# mel512 专用 C 路径执行记录 — 2026-11-16

**单一 mel512 专用实现已完成 native 与 ARM/QEMU 的六次真实 C 回放，六份新 trace 均与各自 backend 的旧 trace 完整逐字节一致；对保存 ORT 结果的所有数值比较也通过原界限。独立 stdlib 读回确认了这些结论。** 本次新增 ORT 调用为 0，但实际运行了六个执行固定模型图的新 C 进程。证据只覆盖三段固定人工信号的共享前端，当前英文关键词模型的完整 L1、独立声学 L2 和设备 L3 均未完成。

## 实际身份与证据

源码来自 [PR827](https://github.com/jiying2007/kws-pipeline/pull/827)，已合并为 `6ed95e5ffcb680d2b9c68778c14289025eeefebd`。实际 [run 43836587076](https://github.com/jiying2007/kws-pipeline/actions/runs/43836587076) 为公开仓库 main push、attempt 1；原生 run/job API 与执行 source binding 一致绑定该合并 head。

| 实际项 | 记录 |
| --- | --- |
| 两个 job | `source-contract` 与 `verify-new-mel-implementation` 均 completed/success |
| Worker RESULT | `EXECUTED_NEW_MEL_IMPLEMENTATION_WITHIN_FIXED_LIMITS`，`passed=true` |
| Worker 退出与失败证据 | return code 0；RESULT 存在，FAILURE 不存在，retry 0 |
| Artifact | `14211845823`，`english-mel-specialization-43836587076` |
| ZIP | 55,470,166 bytes；62 个成员 |
| Result 闭合集 | MANIFEST 记录 50 个文件，加 MANIFEST 自身共 51 个 |
| 独立读回 | `VERIFIED_NEW_MEL_BITEXACT_AND_WITHIN_LIMITS`，`issues=[]` |

实际 ZIP SHA-256：

```text
6d27ec26868a545fa5e2c930eb03fbd20173961c605656cfe033f653988a553f
```

配套 [JSON](research-v4/en-mel512-execution-20261116.json) 保留原始 API、RESULT、MANIFEST、STARTED、source binding、build 与 command-exit 的字节数/SHA，六份 comparison 的完整指标和文件身份、实际计数及独立报告身份。原 RESULT SHA-256 为 `a02142d7f8b37b5207bd1fab152f10f196f3822025b86e9c5dae5b60e9bdb9d9`；MANIFEST SHA-256 为 `fe8cbead0b052bbcf542467e2a7677fc740306d3c79115e0a06c0e409922629d`。

## 固定 baseline 与单一候选

当前目标 IPA、数据来源和基线完整统计继续引用 [已保留的共享前端执行报告](ENGLISH_BACKBONE_EXECUTION_20261116.md) 及其 [JSON](research-v4/en-backbone-execution-20261116.json)。本记录不复制该 69,828-byte 报告。

固定 baseline 为 [run 43829765872](https://github.com/jiying2007/kws-pipeline/actions/runs/43829765872)、attempt 1、head `cd3e07ebc25505648d10166a806a05afade37703`，artifact **14208696183**，27,306,350 bytes，ZIP SHA-256 `dfe28da9c5a35fef3768f7958ff7945e587645554d838dcbb4aa1d0508a9db1f`。新任务直接读取其中三段 PCM、三份 ORT trace、六份旧 C trace 和原 generated C/header/manifest。实际没有运行旧二进制、重新运行 ORT/exporter 或下载新的 ONNX 资产。

| 冻结项 | SHA-256 / 值 |
| --- | --- |
| Profile | `EXACT_MEL512_C_SPECIALIZATION_SAVED_REFERENCE` |
| PLAN | `cc5cd38e2aa96b37672c10d61b8db2ac2364f3d794a0d9eb5f0afe961a2622aa` |
| Candidate C | `5a476954887b59032ea5d419e97d6fb913695e3ad99f5d09164513a4cbd87a89` |
| Worker | `fbce05b6364402f3eaa16be54cea4a97a445188062b1518da855266c4890b9ba` |
| 原 generic C | `c45821d94887d82a6b9f22375875210aacd9a80f6653a520f6761062bf2aee40` |

候选只匹配 Conv rank 3、`[1,1,16000] × [257,1,512] → [1,257,97]`、group 1、kernel 512、stride 160、dilation 1、零 padding、无 bias。保留 channel→frame→tap、`float sum=+0.0f` 和逐 tap 原乘加顺序；其他情况保持原通用路径。编译使用 `-O2 -fno-fast-math -ffp-contract=off`；没有额外累加器、reassociation、unroll、restrict 或 FFT。具体边界见 [专用路径 README](../research/english_mel_specialization/README.md)。

## 六次真实回放与三道验收门

实际使用保存的 zero / sine400 / lcg14，各 48,000 个 PCM16 samples、16 kHz、3 秒。chunk 为 `1,1279,257,4096`，每例重置，不补 EOF。每份 trace 恰含 26 个 embedding 时刻：`16000 + 1280 × k`，`k=0…25`，最后为 48000；每时刻依次记录 raw mel、normalized mel、96 维 embedding。

trace 为 `EFFTR01\0`，小端 `<IIQ>` 头（stage、count、available sample），随后为 little-endian float32；stage 1/2/3 的 count 为 3104/2432/96。每份 trace **586,984 bytes**，无 stage 4、无额外 EOF 输出。

| 实际计数 | native | ARM/QEMU |
| --- | ---: | ---: |
| Build attempts / completions | 1 / 1 | 1 / 1 |
| 新 C process attempts / completions | 3 / 3 | 3 / 3 |
| 完成的 frontend windows | 78 | 78 |

实际 new ORT attempts/completions **0/0**；新 ONNX 资产下载、旧实现进程、head evaluations、optimizer updates、新 TTS/ASR/phones calls、真实语音输入均为 **0**。这里的六次 C 进程确实执行了固定图，不能把整个实验称为“零模型推理”。head 未编译或链接，summary 中 `events=0` 不构成关键词触发覆盖。

三道门全部通过：

1. **同 backend 完整 trace 原字节一致。** 新 native 对旧 native，新 ARM 对旧 ARM；六对都逐字节相同，包含 headers、时钟、signed zero 和所有 float32，`first_different_byte=null`。这道门没有用数值容差替代。
2. **保存的 ORT 比较满足原界限。** 逐有限元素采用 `abs(candidate-saved_ORT) <= atol + rtol*abs(saved_ORT)`；六组比较覆盖 **878,592 对元素**，总 violations 为 0，没有新增 ORT 执行。
3. **process summary 与同 backend 基线相同。** 各例均为 48000 samples、26 windows、无 head；workspace 和 state 均保持原值。

| 信号 | Backend | 字节门 / ORT 门 / summary 门 | 新 host child CPU（秒） | 新整进程 wall（秒） |
| --- | --- | --- | ---: | ---: |
| zero | native | PASS / PASS / PASS | 4.686223 | 4.687644 |
| zero | ARM/QEMU | PASS / PASS / PASS | 99.993777 | 100.004215 |
| sine400 | native | PASS / PASS / PASS | 4.626492 | 4.627475 |
| sine400 | ARM/QEMU | PASS / PASS / PASS | 98.522435 | 98.531336 |
| lcg14 | native | PASS / PASS / PASS | 4.668119 | 4.669157 |
| lcg14 | ARM/QEMU | PASS / PASS / PASS | 98.623111 | 98.633368 |

## 数值统计与独立读回

两个 backend 的 stage 最大值一致。表中 tolerance ratio 为绝对误差除以该元素的 `atol + rtol × abs(reference)`；通过条件为每个元素均不大于 1。

| Stage | atol / rtol | 最大绝对误差 | 最大非零参考相对误差 | 最大 tolerance ratio | Violations |
| --- | --- | ---: | ---: | ---: | ---: |
| raw mel | 0.01 / 2e-5 | 0.0005741119384765625 | 0.00002353523647126582 | 0.05474054302924213 | 0 |
| normalized mel | 0.001 / 2e-5 | 0.000057697296142578125 | 0.00001299672461057624 | 0.052992249870317885 | 0 |
| embedding | 0.001 / 2e-4 | 0.00041866302490234375 | 0.010903604167852873 | 0.1588980704236248 | 0 |

最大相对误差出现在 lcg14 的 embedding sample 39040、component 0，参考值约 −0.00418735；它仍满足固定的绝对加相对误差界限。JSON 保留每个 case/backend 的全部原统计：最坏绝对/相对/容差比例及其位置、reference/candidate 值、元素数、零参考不一致数、violations，以及 old/new trace SHA 和每例完整计时；不能把单一汇总最大值替代这些记录。

独立审计报告为 **131,765 bytes**，SHA-256 `4d4be2b22e642310f25c2534bae2b4c3c214e58b8739def7ff3d33e03bc1ec26`；独立脚本为 **52,211 bytes**，SHA-256 `c0e5550d20465152486d5de541673c34d7091a2bdb2ddc48f45469503010ad8c`。审计以 stdlib 读取原件，核对 62 个外层 ZIP 成员、50 个 manifest 文件、固定 baseline/generated/PCM/reference 与实际 source/pins/run/jobs/artifact；逐字节重验六对 trace，并独立重算全部 878,592 对元素的保存指标。实际 build、C attempts/completions、窗口数及 workspace/state 均吻合，`issues=[]`。读回自身未导入生产 worker、未运行外部进程、C 或模型。

本次有完整成功终态和所有收据，未出现 partial trace 或失败文件。协议对缺终态仍保持“进度未知”的解释：STARTED 仅是尝试预留，局部 trace 只证明已保存部分，不能推断成零 C 推理，也不自动重试。

## 历史耗时与内存边界

**没有速度门，也没有速度参数搜索。** 下表只比较两次不同 job 已记录的整进程 wall 总和；包括启动、输入及 trace I/O，ARM 还包括 QEMU。它是历史 profiling，**不是 matched benchmark**。

| 三段输入合计 | 旧 wall（秒） | 新 wall（秒） | 历史旧/新比值 | 历史 wall 减少比例 |
| --- | ---: | ---: | ---: | ---: |
| native | 17.896553 | 13.984276 | 1.279762592 | 21.8605% |
| ARM/QEMU | 351.012121 | 297.168920 | 1.181187189 | 15.3394% |

新 native 每段 3 秒输入仍耗约 **4.63–4.69 秒整进程 wall**，host child CPU 也约 4.63–4.69 秒；ARM/QEMU 每段约 98.53–100.00 秒 wall。上述差异不确认部署加速、硬件实时性或 SSC305 预算。实际 worker step 为 09:06:30–09:11:47 UTC（317 秒），actual job 为 09:05:36–09:11:52（376 秒）；这两项含准备、编译或收尾，均不是模型单步延迟。

六个 C summary 实际报告相同的 **528,384-byte workspace**；frontend state 为 **native 32,080 bytes / ARM 32,056 bytes**，与基线相同。这些是缓冲区和结构体大小，**不是峰值 RSS 或完整 RAM**，也未测量物理 SSC305。native/ARM executable 分别为 2,442,632 / 2,875,396 bytes；文件大小同样不是运行内存。编译器为 GCC 13.3.0，QEMU 为 8.2.2；具体版本和二进制 SHA 保留在 JSON。

## 英文模型完成边界不变

本次只增加三段固定人工信号上的共享前端实现证据。关键词训练第一道门 `MINIMUM_AFTER_DEDUPLICATION` 仍因 train Vosen **5 < 8** 失败；keyword feature extraction 与第二道门 `MINIMUM_AFTER_FEATURE_GEOMETRY` 均为 **NOT_RUN**。第二道门仍须用实际特征可用的原样本重新检查样本数和家族最低量，不能仅据静态音频时长或数量缺口保证通过；**补三条并不保证可以开始训练**。

当前目标没有新的关键词头权重，keyword optimizer updates 为 0，完整 **L1 未完成**；独立声学 **L2** 和物理 SSC305 **L3** 均未完成。本次未改变冻结 selection、角色、144 分母或数据门，审听包填写也不会自动改变这些状态；没有发音、人类 gold、正式训练、独立 FRR/FAR、word-end latency 或设备合格结论。
