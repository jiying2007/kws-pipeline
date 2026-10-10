# 双唤醒词并行优化评审与决策路线

日期：2026-10-10。目标：`你好小窝`、`小窝小窝` 在最终产品 AFE、真实 3–5 m 和 SSC305 上同时满足误拒、误触发、延迟与资源要求。

**建议先修正数据、事件语义和流式评估的可信度，再用新训 RNN64 对照、因果 DS-TCN 首选挑战者、条件性因果 FSMN 做小而可解释的并行比较。** 不把更多模型、更多 TTS 或更长训练本身视为进展。当前没有可据此宣布的获胜架构。

**交付状态：本轮仅研究方案与元数据校验；C1/C2 encoder、候选 C 导出/runtime 和配对模型实验均未完成。** 没有新训练、模型推理、音频回放、benchmark、TTS 或 ASR，也没有改变 shipping runtime。源码审查基准为 tree `b6f8400bf6e8c15f961089bf5072e152507a2555`。机器计划见 [parallel-candidate-screen](../configs/research/parallel-candidate-screen-2026-10-10.json)，静态检查见 [validator](../tools/validate_parallel_candidate_plan.py) 与 [tests](../tests/test_parallel_candidate_plan.py)。计划的 `execution_enabled=false`；它不取代或扩展 [旧单因素效果链计划](../configs/research/effect-chain-iteration-2026-10-10.json)，也不授予数值执行、数据获取或产品放行许可。

## 1 先解决三个比增加模型更关键的问题

### 实际发音与现有五 token 监督并不等价

现有词表为 blank、`ni3`、`hao3`、`xiao3`、`wo1`。[标签准入](../tools/speech_label_admission.py) 只映射“你、好、小、窝”，拒绝“屋、挖”等词表外转录，也拒绝把有声语音写成空 blank 目标。因此：

- 经人工核对的“你好小屋”等可作为 **no-wake 事件挑战**，检查模型是否错误触发；这一标签不自动成为可训练的 CTC 目标。
- 五输出模型没有 `wu`/`wa` 后验，不能报告 `p(窝)/p(屋)`，更不能把 OOV 映射成 `wo1` 后声称学到了区分。[固定槽 CTC 工具](../research/k1-fixed-slot-ctc-v1/README.md) 的六列外部输入、已给定前缀/区间也不是现有五 token 模型的自动定位器。
- 若后续需要训练近音拒识，应把“扩展 token/音素库存”或“独立事件目标”登记为后续单因素实验。第一轮架构比较固定词表与目标；不能同时更换架构、标签和 decoder 后只归功于架构。

电视或手机播放完整唤醒词究竟应触发，仍待产品语义决定。决策前隔离这类依赖语义的标签；背景有播放、真人在播放上说唤醒词、播放内容本身含完整词，是三个不同情况。不把尚未决定的内容默认标负。

### 现有 AFE proxy 有重要的理想化条件

[训练配置](../configs/training/xiaowo.torch-domain.json) 选择 `afe.backend=proxy`。[acoustic_scene.py](../training/acoustic_scene.py) 的 proxy 用已知真实方位计算延时求和，再按整段音频 RMS 调增益；播放干扰是合成 media 经固定随机延迟的加性残留。它没有证明真实在线 AGC 的状态行为，也不是带播放参考、跟踪回声路径的自适应 AEC。

这些实现可生成受控扰动，但“far=3–5 m”标签不能代表真实远场通过。已存在的 command-AFE 接口会绑定可执行文件与配置身份；有接口不等于已经运行最终 AFE，更不等于完成整机验证。后续应固定同一双麦原音，成对保留 raw/post-AFE、时延和配置，检查播放参考对齐、双讲、回声路径变化、运动噪声及削波；不要只调模型去补偿一个理想化 proxy。

### 连续事件质量不是孤立短片准确率

[Google 流式 KWS 研究](https://arxiv.org/abs/2005.06720) 区分整片与增量推理，并显示按短片重置状态的评估不能代表长期连续流。这里需要保留前端、模型、VAD、decoder、refractory、帧相位与 EOF 行为，测真实事件。只有片段内词是否存在而没有词边界时，应使用 [clip-presence 模式](../eval/README.md)，不报告事件 FRR、词尾延迟或无充分负流支撑的 FAR。

现有原始 FA 的 saved logits/VAD 已存在，但完整原始初态缺失，故原事件因果归因仍为 `UNKNOWN`。这阻止“已找到原 FA 根因”的结论，不阻止独立候选的文献研究、实现准备和未来受控实验。D20 `FAIL`、D90 `NOT_RUN` 属于特定 native A20 数值兼容路线，不是所有新架构失败的证据。其既有执行要求仍按 [D20 准入清单](../research/d20-diagnostic-admission-v1/PLAN-SCHEMA.md#current-admission-checklist) 保持。

## 2 现有代码能复用什么 还缺什么

| 层 | 已核对的可复用基础 | 新候选必须补齐的部分 |
| --- | --- | --- |
| 模型与训练 | [TinyStreamingRNN](../training/model.py)；[train_ctc.py](../training/train_ctc.py) 直接构造该 RNN | 真正的模型工厂、确定性的流式状态契约与新 encoder；增加架构名不会自动生效 |
| 导出与 C | [export_model.py](../training/export_model.py) 的 v2 RNN 张量布局，32/40 特征、hidden ≤64；现有 C runner | 候选专用版本化布局、cache 大小/重置、C kernel 与输出契约；不得把卷积权重塞进旧 RNN ABI |
| 实验身份 | [两臂 spec](../training/product_development_pair_spec.py) 的唯一变量与来源绑定 | 多架构、多 seed、共享校准规则、最差切片与资源选择器；旧 pair runner 不是通用候选 runner |
| 数据与评估 | [PCM 身份](CORPUS_IDENTITY.md)、[标签准入](SPEECH_BASE_ADMISSION.md)、[连续事件评分](EVALUATION.md)、[saved-prediction 准入](../research/native_a20_quality/README.md) | 可训练的新 reviewed base、无泄漏 split、逐词/逐切片支持、可复核的长负流；元数据检查不证明实际发音正确 |
| AFE 与实机 | [最终 AFE 接口](INTEGRATION.md)、[Phase A](REAL_HUMAN_QUALIFICATION.md)、[Phase B](TARGET_AND_SHIPPING_PROMOTION.md) 合约 | 实际最终 AFE 输出、独立真人与 SSC305 多 DUT 测量；接口/交叉编译不能替代这些证据 |

冻结产品模型 `model-749187ec1d66` 是 32×64×5 tanh RNN，`.kwm` 6,812 B，int8 权重但浮点状态/激活/累加。它只能充当历史回归锚点。**公平的 C0 必须用本轮与挑战者相同的获准数据、split 和预算重新训练。** 把新数据训练的挑战者与冻结旧模型比较，无法单独估计架构收益。

## 3 候选优先级来自流式适配和可检验差异

| 候选 | 选择理由 | 关键风险与准入状态 |
| --- | --- | --- |
| C0 `rnn64_control` | 复用已知 RNN64，提供新数据同条件对照 | 仍需本轮数据/训练身份与导出验证；不是冻结产品模型的别名 |
| C1 `causal_ds_tcn_ctc` | 首选挑战者。左缓存、深度可分离时间卷积、pointwise 与残差有明确 C 映射；可维持帧级 CTC 和统一事件接口 | 尚未实现/训练/测量。缓存随层数、核长、dilation 增长；BN 必须冻结/折叠并核对；小权重不保证小 cache 或低 CPU |
| C2 `causal_fsmn_ctc` | 条件候选。投影维度与有限历史记忆提供不同的容量/缓存取舍 | 尚未实现或准入；必须严格 `rorder=0`，核对 tap 顺序/stride/cache。FSMN 也用逐通道时间滤波，不能夸称完全无关的机制 |
| 后续 trigger-only verifier | 在第一阶段产生候选事件后，以有限缓存复核双词/背景，可能减少特定误触发 | 当前不纳入首轮 6+2。不能找回第一阶段漏检；必须测条件 FRR、额外延迟、触发频率与突发 CPU，不能只报 verifier 自身准确率 |

C1 采用 [WeKWS TCN 固定源码](https://github.com/wenet-e2e/wekws/blob/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86/wekws/model/tcn.py) 的设计依据，代码为 Apache-2.0。其左缓存和 dilation 可直接审查，但 [公开 CTC 配方](https://github.com/wenet-e2e/wekws/blob/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86/examples/hi_xiaowen/s0/conf/ds_tcn_ctc.yaml) 是 40 特征、256 hidden；不能把该配方的行为或参数规模套给尚未冻结的小模型。

C2 可参考 [FunASR FSMN-KWS encoder](https://github.com/modelscope/FunASR/blob/e7e61293d308f4342ab9307895e710e4b51b949c/funasr/models/fsmn_kws/encoder.py)；固定版本 [LICENSE](https://github.com/modelscope/FunASR/blob/e7e61293d308f4342ab9307895e710e4b51b949c/LICENSE) 为 MIT。相较之下，所审 [WeKWS fsmn.py](https://github.com/wenet-e2e/wekws/blob/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86/wekws/model/fsmn.py) 有 `rorder=0` 时 `:-0` 空切片及构建 stride 的适配问题，不能当因果即插即用实现。[cFSMN 唤醒词论文](https://www.isca-archive.org/interspeech_2018/chen18c_interspeech.pdf) 提供中文远场研究动机，但其私有大量数据、HMM 体系与前视配置，不是本项目因果 CTC 收益的证据。FunASR 的 [KWS 指南](https://github.com/modelscope/FunASR/blob/e7e61293d308f4342ab9307895e710e4b51b949c/docs/keyword_spotting.md) 也不能当作已具备逐包唤醒事件的接口保证；此处只借鉴 encoder/cache，仍须实现本地逐帧 CTC、C 导出与连续 decoder 适配。

### 为什么暂不再加更多首轮模型

- [DS-CNN / Hello Edge](https://arxiv.org/abs/1711.07128)、[TC-ResNet](https://www.isca-archive.org/interspeech_2019/choi19_interspeech.pdf) 支持研究高效卷积，但主要短词整片设置、不同特征和计时范围不等于中文连续 FA/h。另开多种卷积臂会扩大相近结构搜索，降低首轮结论清晰度。
- [BC-ResNet 论文](https://arxiv.org/abs/2106.04140) 和 [官方代码](https://github.com/Qualcomm-AI-research/bcresnet/tree/656f947a2a757ab5f36f649ed303ea0191cf553e) 值得后续小 verifier 参考；原实现的对称时间 padding、整窗 pooling、5 子带归一化需要流式与频率维适配，不能直接接现有 32 特征。其 BSD-3-Clause-Clear 许可证明确没有专利许可，代码许可不能被简化为无限制产品许可。
- 更大 attention、RNN-T 或整套 ASR 系统同时引入上下文、前端、预测网络/搜索和资源变量。两条固定四音节词的第一轮，优先选择更小、可隔离的差异。此为工程优先级判断，不是这些模型质量较差的实验证明。
- [sherpa-host](../research/sherpa_host/README.md)、native A20/full-donor FSMN 可保留为独立系统参考。它们的输入维数、词表、前视、后端和 EOF/重置不同；C-facing API 也不代表内部纯 C。历史外部命中率不能混入本轮架构公平排名。

## 4 数据路线先建立可用对照 再有界增加覆盖

每个输入绑定原文件/PCM SHA-256、真实帧数与时长、母来源与派生参数/seed、speaker/voice/session/source/room/device、实际发音审核及 split 角色。近重复、同母录音裁剪、同克隆参考或同声音族的衍生不能跨训练与独立评估。TTS 不同 voice slot 只能支持声音槽位泛化；跨 generator、真人、房间/设备迁移分别留出。

初轮只使用通过准入的同一数据版本，固定增强分布、样本暴露量和相同的数据顺序/增强随机种子。源筛选可比较精确双词、缺首/缺尾/重复/近音等，但不因自然度或双 ASR 一致而自动晋级训练。现有 [source-screen32 结果](../research/source-screen32-v1/evidence/run-38021467257/README.md) 中有“两系统一致但都偏离提示词”的例子，足以说明机器共识不是实际发音真值；不在此发布私有人审状态。

[有限真实数据与合成语音研究](https://arxiv.org/abs/2002.01322) 支持离线 TTS 补覆盖；[2024 年 TTS 过拟合研究](https://www.isca-archive.org/syndata4genai_2024/park24b_syndata4genai.pdf) 同时发现 KWS 表征能辨别合成/真实域，提醒模型可能学习合成伪迹。其“无真实正例”实验仍使用真实负音频，不能推导为纯 TTS 足以完成本产品。[Odyssey 2022 近音负例研究](https://www.isca-archive.org/odyssey_2022/wang22c_odyssey.html) 支持定向混淆词实验；[播放干扰增强研究](https://arxiv.org/abs/1808.00563) 支持匹配残余回声域。这里只迁移可检验的方法，不移植论文准确率、私有数据或训练规模。

| 来源候选 | 有价值的用途与限制 | 许可与采用条件 |
| --- | --- | --- |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | 离线双词/近音定向合成；内置音色重复采样不是独立说话人 | 代码及所查 [1.7B Base 权重](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 标注 Apache-2.0；参考声音须自有/获授权，生成音频另审 |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 与 [Whisper](https://github.com/openai/whisper#license) | 不同家族弱标签复核；一致仍非真值，ForcedAligner 定位给定文本不证明读对 | Qwen 代码及所查 [1.7B 权重](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 为 Apache-2.0；Whisper 代码/权重为 MIT；只作离线候选 |
| [HI-MIA](https://www.openslr.org/85/) / [HI-MIA-CW](https://www.openslr.org/120/) | 前者借鉴家庭 1/3/5 m 与 speaker 分组，后者借鉴真人混淆词；均不能冒充“小窝”正例或专用验收 | 分别为 Apache-2.0 / CC BY-SA 4.0；任务词为“你好米雅”，许可义务与来源分别保留 |
| [AISHELL-1](https://www.openslr.org/33/) | 真实中文预训练/经筛查负音频；近场朗读不代表远场或自然连续负流 | 页面列 Apache-2.0，亦有 academic-use 表述；保留实际版本条款，产品采用前澄清范围 |
| [RIRS_NOISES](https://www.openslr.org/28/) / [MUSAN](https://www.openslr.org/17/) | 混响/噪声、真实背景语音/音乐；需补中文、本机电机与 AEC 域，按母来源分割 | 页面分别标 Apache-2.0 / CC BY 4.0；检查实际包与归属，合成 RIR 不证明真实 5 m |
| [WenetSpeech](https://wenet-e2e.github.io/WenetSpeech/) | 仅列非商业研究负池/方法候选，不直接进入 shipping 来源 | 官方同时有 non-commercial、CC BY 4.0 链接及原音频版权说明；不能仅凭仓库许可证解除限制 |
| 自有最终硬件连续录音 | 真人双词、自然中文负流、电机启动/变速、播放参考与双讲最贴近部署 | 同意/用途、speaker/session/source 分组及私有留存；开发采集与最终 sealed 资格隔离 |

以上新来源建议相对本轮训练准入均为 `NOT_READY`；未测资源/预算记为 `unknown/null`，不填零。既有历史样本身份与结果不被改写；采用前仍须审核实际版本、权利、输入与资源。本轮没有下载模型/语料或运行这些工具。

架构选出后，再逐项比较获权新来源、经人工确认的 hard negatives、真实或获权测量 RIR、最终 AFE 域。近音语音不作 blank；与新词表/事件目标有关的材料等对应单因素协议就绪后再训练。噪声、背景、音乐、房间 RIR 同样按母来源隔离。重复一段负流可测状态耐久，但不能虚增独立声学覆盖。

权利账本分别记录代码、权重、原始数据、生成音频和声音/同意范围。宽松代码 LICENSE 不自动覆盖下载权重、第三方数据、克隆参考或生成内容的重分发。固定 URL/版本/许可快照，权利不明则保留待审状态；私有原音、post-AFE 音频、私人特征/权重和凭据不进入公开 Git、CI 制品或本文。本文不启动下载、付费资源或新数据生成。

## 5 分阶段并行 不是全因子扫参

### 阶段 0 可以并行推进的准备

1. 数据/语义线：核对实际发音、PCM 血缘、split 与可训练支持；等待完整词播放语义决定，不阻塞无关的源码研究。
2. 架构线：C0/C1 的接口、cache、模型格式和资源静态审查；C2 单独解决因果配置与来源适配，不能因为写入候选表就算已准入。
3. 评估线：复用同一 event scorer，声明校准与选择数据、切片、曝光及状态边界；原 FA 保存状态清单另行补齐。
4. AFE/资源线：准备真实 command-AFE 身份与 SSC305 计量方案。尚无设备不阻止软件研究，但不能补造目标板数字。

各线完成的是下一阶段的输入。新计划的 schema/fixture 测试通过，只说明结构与拒收规则工作；不证明真实音频、候选数值、资源或运行权限已经就绪。

### 阶段 1 预注册首轮比较

上限建议为 **最多 3 个获准候选 ×2 个开发 seed =6 fits**；C2 未准入则缺席，不用未知新来源替补。随后只允许最多一个挑战者与 C0 使用第三个开发 seed 配对确认，**最多再 2 fits**。这是待冻结的研究规模上限，不是已验证的公平预算、已发生训练或现有执行许可；后续 verifier 不包含在这 8 fits 内。

预注册必须补齐：准确实现/数据身份、seed 未消费证明、固定结构与特征、目标/loss、优化器、样本暴露、每 fit 更新/时间/内存上限、checkpoint 选择、量化与导出方式、统一 decoder/VAD/endpoint、校准方法及比较/停止规则。

相同 seed 只配对可共享的数据顺序、采样与增强；形状不同的架构没有“相同初始权重”。相同 updates 也不等于相同计算或同等收敛。旧 1,000 updates 是历史拟议上限，不能在缺乏新预算审查时宣称适用于所有候选。记录实际样本暴露、优化器步、运行时间/资源，并预先固定末 checkpoint 或共同选择规则，不能事后给落后臂延长训练、扩大搜索或挑最佳 checkpoint。

### 阶段 2 用同一规则校准 再比较冻结的工作点

跨架构分数分布不同，不能机械共用产品旧阈值 0.55。各臂使用 **同一预注册校准方法、阈值搜索预算、独立 development-calibration 集与双词合计目标 FAR**；得到各自阈值后冻结，在分离的 development-selection 上比较。无可行工作点或数据支持不足时保持不可判定/失败，不能到选择集、原始错误集或最终留出上继续调阈值。

两个开发 seed 的结果先完整报告，最多选一个挑战者，再用第三个开发 seed 检验稳定性。三 seed 不是自动的统计胜利；对大量切片/多个候选的选择偏差必须披露。冻结产品模型可单列回归结果，不参与 fresh C0 的配对收益计算。

### 阶段 3 只组合被单独支持的改进

架构固定后，依次检验词表/事件目标、AFE/特征或量化等单因素。最终选择组合必须重新跑完整 PCM→AFE→特征→模型→decoder 链，测开发集并按规则校准，之后冻结 tuple。不同基线下的提升不可相加；交互可能抵消收益。只有组合后的新实测才能成为最终候选证据。

## 6 可测量的选择矩阵

不以一个加权总分掩盖硬门槛。以下是拟议开发决策规则，现行产品政策另见下一节；未冻结的统计非劣界、样本量和数值容差必须预注册，不能看结果后填写。

| 顺序 | 统一记录 | 选择或停止条件 |
| --- | --- | --- |
| 输入准入 | 来源/权利、实际转录或事件标签、PCM、母来源、角色/消费历史、逐词/切片数与负流小时 | 身份矛盾、泄漏、OOV 错监督、支持不足不得进入质量排名 |
| 流式正确性 | 单帧/多种分块、长流、reset、discontinuity、短尾/EOF；参考与 C 的逐层/logit/状态/事件差异 | 任一未定义状态、未来帧依赖、格式或数值门槛失败先阻断该候选；无声学收益抵消此项 |
| 工作点可行性 | 同校准规则下的各臂阈值、逐词及双词合计连续 FAR/h、单侧 95% 上界、实际曝光 | 满足预声明目标才比较召回；零次 FA 不等于零 FAR，不能无依据外推 |
| 召回与切片 | 双词各自 FRR、区间、最差已声明切片；同录音成对的成功/失败差异 | 两 seed 各自不得靠牺牲一个词/关键切片换总体平均；至少一项明确改善，否则无赢家 |
| 事件延迟 | 词尾起算的 p50/p95/p99、早触发/重复触发、AFE 与缓存延迟 | 不以 EOF 补齐、人为裁剪或重置收益掩盖线上代价；无词尾标签不报该指标 |
| 资源与稳定性 | 全链 CPU、每 hop p99、RTF/headroom、cache/engine/RSS/stack、I/O、线程、温度/功耗 | 先满足冻结资源预算，再在质量/资源 Pareto 候选中选择；参数量/MAC 只作解释 |
| 第三 seed 与最终组合 | 同候选/规则的再现、所有已观察失败、组合后全链重测 | 方向不稳定或支持不足则 `INCONCLUSIVE`，不继续换 seed 挑赢家 |

开发推断建议按 speaker/session/source 等独立组保留配对差异和区间，而非把大量增强副本当独立样本；需统计检验时预注册比较数与方法。正式 Phase A 仍使用仓库规定的 Wilson FRR 与精确 Poisson FAR 单侧界，不暗改评分政策。上表不是未经实现的新自动 promotion gate。

每个新 encoder 至少验证三层 parity：① 浮点参考的离线/分块/逐帧等价与状态；② 导出后数值容差、量化及 C 状态等价；③ 同 PCM/AFE/VAD/decoder/阈值下最终事件与时间等价。字节格式可变但接口要明确。全 int8 的激活/累加变化应作为后续独立变量，不因当前模型只有 int8 权重就声称已拥有 W8A8 实现。

## 7 研究获胜之后仍需跨过产品门槛

产品权威仍为 [shipping 合约](../configs/shipping.xiaowo.json)、[Phase A policy](../commercial/real-human-qualification.policy.json) 和 [Phase B policy](../commercial/target-qualification.policy.json)。当前 `shipping_approved=false`、`recalibration_required=true`；既有合成模型资格不覆盖变化后的源码，本文没有更新任何资格结果。

| 阶段 | 现行要求摘要 |
| --- | --- |
| Phase A 样本 | ≥40 speakers，每人 ≥2 sessions；≥1,200 次目标事件，每词 ≥600；≥24 h post-AFE 负流 |
| Phase A 质量 | 总体及每词 FRR ≤0.05；总体单侧 U95 ≤0.065，每词 U95 ≤0.08；FAR ≤0.10/h 且 U95 ≤0.13/h；p95 词尾后延迟 ≤500 ms |
| Phase A 匹配与覆盖 | 词首前容差 0 ms、词尾后 500 ms；3–5 m、后方、播放、双讲、运动及负流分布按 policy 的精确配额，不用 proxy 标签替代 |
| Phase B | ≥3 个同 SKU/板版/部署/Phase A/最终 AFE/资源预算 DUT；每台 ≥24 h，其中至少 1 台 ≥72 h；p99 在块 deadline 内、RTF <实时、p99 headroom >1；xrun、discontinuity、lost samples、backpressure 均为 0 |
| Phase B 资源与 Phase C | 按冻结 budget 验证 CPU/RSS/stack/温度/平均功率和原始受控测量；Phase A/B 通过后仍须显式 terminal shipping promotion |

切片精确样本/曝光配额与 FRR 上限以 policy 为准。3/4/5 m 逐距离与充电态可增加开发诊断，不私自替换正式切片。24 h 内零 FA 的单侧上界仍接近 0.13/h 门槛；一个 FA 可能使置信门失败，不能只看点估计。

SSC305 CPU 按 [单核归一化 process CPU 契约](TARGET_EVIDENCE.md#cpu-units-and-unavailable-audio-exposure) 统计，记录线程数，不因双核除以二。host 时延、QEMU 和通用 ARM 编译不换算成实机 CPU。Phase B 的板端音频须遵守 `non-human-public-safe` 和禁止 human-derived 输入的政策；不能把 Phase A 私有真人音频搬到公开板测通道。

最终 held-out 只在候选、阈值、AFE、词包、模型、源码及工具链 tuple 冻结后按现行 no-retry 流程消费。资格失败可指导下一轮开发，但那批数据随即不再 fresh；不能原地改阈值再称首次通过。没有最终 AFE 真人与实机证据时，只能完成软件研究阶段，不能关闭产品目标。

## 8 当前结论与下一次应交付的证据

目前最值得投入的是：清楚的近音/播放事件标签、可训练且无泄漏的声学底座、真实 AFE 域，以及可复现的 C0/C1 流式实现。C2 只有在因果实现、来源与资源门槛全部准备好时才加入；后级 verifier 等单阶段错误结构确实需要时再评估。

下一次里程碑应交付可审计的候选/数据/校准身份、未消费 seed 与预算、实现及 C parity 结果，并说明仍缺哪些支持；之后才是获准的配对训练与完整负流/逐词评估。结果可能是选出候选、明确无收益或支持不足，三者都应终止本轮择优并保留失败，不能自动扩成更大的扫参。

后续持续闭环按“错误类别 → 可证伪单因素假设 → 固定比较 → 明确停止/选择 → 全链组合重测 → fresh 产品资格”推进。研究与数据工作可以并行；依赖的运行准入、用户语义决定或受控输入未到位时，保留具体阻断，继续独立准备，不把等待改写成通过。

English summary: Compare fresh same-data RNN64, causal DS-TCN and conditionally admitted causal FSMN only after data, streaming and export gates pass. The 6+2 fit ceiling remains a proposal; no candidate has been trained or ranked. Label truth, real AFE, complete-chain gains and physical SSC305 evidence remain essential.
