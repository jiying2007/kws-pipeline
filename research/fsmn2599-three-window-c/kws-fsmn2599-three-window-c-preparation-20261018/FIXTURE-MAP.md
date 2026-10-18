# 独立 C 三窗口候选事件移植：46 项源基准映射

日期：2026-10-18 UTC。状态：**静态阅读与未来测试设计；未编译、未运行 C/Python 测试，未处理模型、真实后验或音频。** 本文件不构成执行许可，也不把历史 Python 通过结果记为 C 通过。当前重 CPU 训练在进行，任何下述执行须由父任务另行放行。

## 1. 阅读范围与判定口径

完整阅读同级历史目录 `../kws-fsmn-three-window-adapter-20261017/` 的三个测试文件、`posterior_events.py`、`three_window_events.py`、`beam_minimal_patch_candidate.py`、`vendor/stream_kws_ctc.py` 与 `vendor/decoder_tail_fix.py`。同时对照固定配置、README、FINAL-BUDGET 和独立审查记录。只做源代码、文档及 SHA-256 读取。

此处 `V01..V18`、`T01..T22`、`P01..P06` 分别稳定映射到三个源测试文件，不以新测试函数数量冒充源测试覆盖率：

- **reuse**：输入与关键预期语义原样保留；C 错误码、数组、测试夹具替代 Python 类/异常属于正常机械转换。允许用只在测试构建开放的内部 hook。
- **adapt**：保留该测试要检查的风险，但调整接口/断言，或把旧失败反例改为新 C 的正向要求；明确列出变化。
- **history-only**：用于解释源身份或已知旧缺陷，**不得要求新 C 复现旧缺陷**，不得作为 C 运行通过项计数。

汇总：**26 reuse + 15 adapt + 5 history-only = 46**。因此本方案有 **41 个 C 可执行语义用例槽位**和 **5 个历史/来源记录槽位**；这不等于已经实现或运行 41 项。

| 源组 | reuse | adapt | history-only | 合计 |
|---|---:|---:|---:|---:|
| v1 PosteriorFixtures | 7 | 8 | 3 | 18 |
| ThreeWindowFixtures | 17 | 4 | 1 | 22 |
| MinimalPatchFixtures | 2 | 3 | 1 | 6 |
| 总计 | 26 | 15 | 5 | 46 |

已静态读取准备中的 `kws3.h`、`kws3.c` 和 `kws3_test.h`。以下 `STEP`=`kws3_step`，`FINISH`=`kws3_finish`，`RESET`=`kws3_reset`，`SUMMARY`=`kws3_get_summary`；分块入口为`kws3_process`。`TOP3`=`kws3_test_top3`，`BEAM`=`kws3_test_beam`，`DETECT`=`kws3_test_detect_seed`，`MERGE`=`kws3_test_merge`，`INSPECT`=`kws3_test_snapshot`。`SEED`是构造`kws3_test_hyp`/候选输入的测试能力，不是已存在的通用写状态API。内部hooks仅在`KWS3_TESTING`构建中暴露；当前接口仍可能修订。

**覆盖目标与准备进度分开：** 本文41项是完整移植验证的目标，并不表示现有smoke/differential驱动已实现这些用例。实现者另准备C smoke及Python ctypes differential，当前均未执行；当前执行计划以同目录将冻结的`PLAN.json`为准。实现者报告当前差分驱动共28 methods：原v2 22方法（其中T13仅Python cache、T20仅源身份，均非C运行证据）、patch P01/P04/P05/P06四项及2个additional；C smoke为8 groups。静态计划C step为差分4164（含2次invalid）加smoke36，独立beam30+2、MERGE513+2、TOP3检查11+1。上述是实现者提供的准备数量，本文未运行或验证其通过。第6节仅为**尚未实现的完整41槽位扩展提案**，不可替代当前计划或将其算成已覆盖。

## 2. v1 18 项逐项映射

源文件：`test_posterior_events.py`。行号为本次静态读取版本。

| ID / 行号 / 源测试名 | 分类 | 新 C 必须检查的内容、调整与入口 |
|---|---|---|
| V01 / 20 / `test_01_canonical_score_and_ids` | reuse | 依次输入 `(1027,1401,1462,2093)`，每行目标 `.9`、非关键词 2 为 `.1`；一事件、keyword_id=1、start/end=(0,9)、score≈`.81`。`STEP`，double 级数值比较。 |
| V02 / 28 / `test_02_second_keyword` | reuse | `(1462,2093,1462,2093)` 发出 keyword_id=2；不能把两个词的字符集合等同于有序序列。`STEP`。 |
| V03 / 32 / `test_03_chunk_partition_invariance` | adapt | 25 行序列：词1、17行 blank、词2；参考连续输入与分块尺寸1/2/3/5/8/25所得事件及**新三窗 summary**相同。C没有惰性生成器也仍需证明调用批次边界不触发 reset/retirement。共7遍×25行。 |
| V04 / 45 / `test_04_two_events_one_chunk_no_dropped_rows` | reuse | 同25行序列必须得到id `[1,2]`，rows_decoded=25；绝对下一输入帧位置等价于75。事件返回不能提前中断剩余输入行；不必新增旧 `total_frames` 字段，允许用计数关系检查。 |
| V05 / 53 / `test_05_final_suffix_is_found` | adapt | `SEED+DETECT` 放入 `(1401,1027,1401,1462,2093)`，frames `(0,3,6,9,12)`；最终合法起点offset=1必须命中，start=3。C正常公开输入不应接受任意内状态；无需移植v1全局total_frames赋值。 |
| V06 / 59 / `test_06_rejected_score_does_not_accumulate` | adapt | `SEED+DETECT` 两次检测同一完整词，frames `(0,1,2,3)`、每节点`.8`：两次均因duration3拒绝，两次hit_score均`.64`。需提供只读最后检测score或DETECT诊断返回；不能把未激活得分强行记成事件。 |
| V07 / 69 / `test_07_refractory_global_and_inclusive` | adapt | 将旧局部last_active_pos改为**全局合并器**的end差测试：当前没有直接设置global last-end的hook，可在两个独立实例各先MERGE合法词1、nodes0/3/6/9，再分别送end58/59的词2候选，精确产生差49/50，期望拒绝/接受。此整体平移保留原边界，4次MERGE。正常3帧网格不能精确构造49/50差，不能仅用48/51代替。 |
| V08 / 77 / `test_08_duration_inclusive` | adapt | `SEED+DETECT` 同词frames `(0,1,2,duration)`，duration4/5/250/251，期望否/是/是/否。此为内部检测边界；公开STEP上的实际节点落3帧网格，不能声称公开流已覆盖5和250。 |
| V09 / 84 / `test_09_official_expiry_after_row` | history-only | 源v1在84行后total_frames252触发best-first-node-age>250的整束expiry。新C**不实现**该年龄expiry；三窗退役边界由T08覆盖。 |
| V10 / 95 / `test_10_expiry_boundary_can_miss_valid_word` | adapt | 同85行反例 `[1401]+80*blank+词1` 在新C应有一个事件，nodes243/246/249/252；可保留4行fresh-control。源v1预期空事件仅作历史说明。新C正向断言与T02重叠，但映射保留。 |
| V11 / 103 / `test_11_full_alphabet_top3_blocks_target` | reuse | 2599维中id1/2/3/1027概率`.34/.33/.32/.01`；全类top3均不在保留集合，当前beam保持空前缀。不能先裁关键词表再top3。`STEP+INSPECT`。 |
| V12 / 110 / `test_12_probability_mass_not_renormalized` | reuse | id1/2/1027为`.50/.44/.06`，节点1027的原概率仍`.06`，不能过滤后归一化。`STEP+INSPECT`。 |
| V13 / 117 / `test_13_fixed_ties_and_probability_floor` | reuse | 四token `(0,1027,1401,1462)`同为`.25`，`TOP3`必须 `[0,1027,1401]`；目标1027恰`.05`不得保留，beam空前缀。需明确输入精度，见第4节。 |
| V14 / 126 / `test_14_invalid_rows_latch_failure` | reuse | 5种初始非法输入：长度6、全零、全NaN、Inf+零、`[-.1,1.1]+零`。每次首次错误后summary.valid=false，随后合法blank仍报会话已失效；需要RESET恢复。C错误码替代异常；至少保留原因类别。 |
| V15 / 137 / `test_15_capacity_failure_not_silent_reset` | adapt | 不照搬不可达的128节点种子作为正常三窗状态。通过测试hook造成prefix/node错配、超过126窗口长度或静态槽容量的内部状态/扩展请求，要求显式失败并锁存，不能用静默reset掩盖。RESET后valid恢复；无“expiry_resets=0”旧字段要求。 |
| V16 / 147 / `test_16_streaming_iteration_and_no_torch` | adapt | C逐行入口创建/初始化不消费任何行；送满4行才输出row3的首词；阈值仍0、diagnostic候选标志明确。Python generator惰性与sys.modules断言替换为C无模型/音频/Torch依赖和无全流缓存的构建/源码检查。 |
| V17 / 155 / `test_17_known_empty_prefix_underflow_is_visible` | history-only | v1的1100行半概率blank导致zero mass后仍能靠节点score激活，是旧弱点而非C规范。禁止为兼容该断言制造zero-mass事件。T03/P03覆盖新C的相应正向行为。 |
| V18 / 165 / `test_18_nonempty_prefix_is_routinely_expired` | history-only | 336行交替token使v1整束expiry4次、max_prefix84；新C不继承这两个数值。三窗有界持续流由T15/T19覆盖。 |

## 3. 三窗22项及补丁6项映射

### 3.1 ThreeWindowFixtures

源文件：`test_three_window_events.py`。

| ID / 行号 / 源测试名 | 分类 | 新 C 必须检查的内容、调整与入口 |
|---|---|---|
| T01 / 20 / `test_01_canonical_score_and_absolute_time` | reuse | 90行blank+词1，共94行；一事件nodes `(270,273,276,279)`，score`.81`，duplicate_candidates=2。窗口局部时间不可冒充绝对时间。 |
| T02 / 28 / `test_02_old_expiry_counterexample_is_covered` | reuse | 同V10的85行新三窗正例：一事件nodes243/246/249/252，decision_row84。C只运行新逻辑；源测试中的old对照结果为历史已知事实，不能要求本C另外实现old decoder。 |
| T03 / 38 / `test_03_long_blank_does_not_zero_beam_mass` | reuse | 1100行blank`.5`后，empty/zero_mass_window_rows均0且各活动窗首hyp.pb>0；接4行词1产生一事件nodes3300/3303/3306/3309。不允许通过beam归一化偷偷改变规范。 |
| T04 / 48 / `test_04_geometry_all_42_phases` | reuse | 42个phase×span2..83=3444个纯整数几何组合；start=84+phase，window=floor(start/42)*42，断言window≤start且start+span≤window+125。不喂后验、不声称beam/声学召回保证。 |
| T05 / 56 / `test_05_actual_maximum_span_at_boundary_phases` | reuse | phase0/1/40/41；start=84+phase，节点行start/start+27/start+55/start+83；四条输入分别168/169/208/209行，一事件且249基准帧跨度。 |
| T06 / 68 / `test_06_chunk_partition_invariance` | adapt | 86行序列 `40blank+词1+38blank+词2`，一遍reference与块尺寸1/7/42/50/126共6遍；比较全部事件与FINISH后全部语义计数。使用C驱动层分组，不假定公共API必有batch函数。 |
| T07 / 82 / `test_07_start_and_partial_eof` | reuse | 初始化0窗口；送词1前三行无事件；FINISH不增加rows、关闭1窗口、live0、events0；重复FINISH幂等。随后送一合法row必须报EOF且按错误协议锁存。 |
| T08 / 96 / `test_08_window_retirement_and_beam_update_budget` | reuse | 125行后starts `[0,42,84]`；处理row125后starts `[42,84]`、retired1；row126开始新窗126，starts `[42,84,126]`；累计beam_updates255、topk127。先处理最后一行再退役，不提前删窗。 |
| T09 / 108 / `test_09_exact_duplicate_prefers_score_then_window` | reuse | `MERGE`同键3候选，score`.7/.8/.8`、window0/42/0，6排列结果一致，duplicate2、赢家`.8`/window0。绝对四节点tuple是key的一部分。 |
| T10 / 121 / `test_10_same_frame_multiple_distinct_events` | reuse | `MERGE`两候选ends9/69、scores`.9/.8`、decision_row100；两个排列都输出ends `[9,69]`，decision_frames均300。一次STEP的输出容量应足够0..3事件，不能强制每行仅1事件。 |
| T11 / 130 / `test_11_same_frame_sorting_precedes_refractory` | reuse | 相同两候选改scores`.8/.9`；高分end69先接受，end9随后因负的end差被拒绝，只输出69，refractory_rejections1。不能改成时间顺序或无符号差。 |
| T12 / 137 / `test_12_shared_refractory_and_local_candidate_consumption` | reuse | 115行：90blank+词1+词2+13blank+词2；仅id `[1,2]`，end `[279,342]`，有refractory拒绝。每窗duration有效候选必须先清该窗beam，即使最终重复/冷却拒绝；清beam不延长stop。 |
| T13 / 146 / `test_13_cache_is_computed_once_and_not_mutable_by_consumers` | adapt | 原Python用新list view避免consumer修改缓存。C可用const只读固定数组或逐consumer值拷贝，但须证明三窗复用同一全类top3结果，修改测试副本不影响后续读取且计算计数=1；不能向const内存强写制造UB。 |
| T14 / 156 / `test_14_full_class_top3_and_no_renormalization` | reuse | 两个独立1行实例复核V11/V12；一个保留原`.06`，另一个全类前三为filler且空前缀。 |
| T15 / 167 / `test_15_state_is_bounded_on_long_nonkeyword_prefix` | reuse | 1000行1027/1401交替，无事件；max_prefix126、max_live_windows3、node_refs≤7560、max_next≤80、beam_updates2874、topk1000；started=retired+live。 |
| T16 / 179 / `test_16_dedup_history_expires_and_has_fixed_bound` | reuse | `MERGE`在n4..503共500次每次3个key（不同keyword或首节点），最后seen=heap=378、max_dedup378。C若无heap，验证等价有界历史占用和到期规则，不能要求不必要的数据结构。 |
| T17 / 191 / `test_17_invalid_input_and_real_meeting_reset` | adapt | NaN row锁存invalid；RESET(new-meeting)后词1得到recording_id新值、event_index0、nodes0/3/6/9。若C核心将字符串/会话标签放在外层，驱动层必须证明此关联，不能宣称核心已自行存字符串。 |
| T18 / 202 / `test_18_empty_meeting_and_no_torch` | adapt | 空会话FINISH后windows_started0、无事件；threshold0。`torch not in sys.modules`替换为原生目标无Torch/model/audio依赖检查；不要求C读取Python解释器状态。 |
| T19 / 209 / `test_19_full_beam_staging_is_bounded` | reuse | 260行固定posterior：blank`.4`、1027`.3`、1401`.3`；无事件，至少一活动窗达到20path，next≤80、node_refs≤7560、prefix≤126。保留质量同分时的稳定顺序。 |
| T20 / 219 / `test_20_v1_source_remains_byte_identical` | history-only | v1源码SHA-256仍为`56d6ada2c531de25f72a713db77b07ef9d8326b8b1e238bd776d40155d8584ec`。这是来源保护，不是C运行语义。本次静态sha256读取相符；不可把它计为C功能测试通过。 |
| T21 / 224 / `test_21_later_higher_duplicate_does_not_rewrite_event` | reuse | `MERGE`row100先给score`.7`同键，row101再给`.9`；第二次不输出，duplicate1，已发事件仍`.7`。不等待后窗、不回写已发事件。 |
| T22 / 232 / `test_22_cross_keyword_score_tie_is_fixed` | reuse | 相同frames/score但两个keyword_id不同，两个输入排列都仅id1接受，refractory_rejections1。关键词不同不是duplicate；先keyword_id升序，再共享冷却。 |

### 3.2 MinimalPatchFixtures

源文件：`test_beam_minimal_patch_candidate.py`。

| ID / 行号 / 源测试名 | 分类 | 新 C 必须检查的内容、调整与入口 |
|---|---|---|
| P01 / 46 / `test_01_independent_fraction_ctc_oracle` | adapt | `BEAM`对3符号`(0,1027,1401)`、长度1..4、初始mass1和1e-9，共8组，与**独立路径枚举**的每prefix pb/pnb及总mass比较；rel_tol2e-14、abs_tol1e-30。C无需Python Fraction：把原4行概率写为整数分子/20，路径用uint64整数累积，最后除`20^L`及可选1e9。最大分母1.6e14；不要用待测beam计算oracle。 |
| P02 / 60 / `test_02_old_core_loses_positive_mass` | history-only | 旧math.isclose(abs_tol1e-6)在小mass下丢概率是已知缺陷；新C不实现/不运行old branch以“通过”此项。P01/P03/P05证明新正向要求。 |
| P03 / 67 / `test_03_repeated_target_after_blank_old_failure_new_pass` | adapt | C只检查patched一侧：20或62行blank`.5`后，每个词1 token连续2行，共28/70行；各得1事件、empty0、zero0。源old一侧0事件及empty>0保留为历史解释，不能搬回新实现。 |
| P04 / 81 / `test_04_old_alias_pollution_and_patch_isolation` | adapt | `SEED+BEAM`两hyp`(1027)` mass(.4,.2)和`(1027,1401)` mass(.1,.1)，逻辑共享首节点frame0/prob`.2`；第二节点frame3/prob`.3`。row6 token1027`.6`、filler2`.4`。输入保持不变；扩展prefix(1027,1401,1027)的frames为0/3/6；重复更新prefix(1027)末frame6。C若节点为值数组，不要求模拟Python对象身份，只断言输入/兄弟节点值隔离。 |
| P05 / 103 / `test_05_scale_preserves_mass_ratios_and_rank` | reuse | 同4行分别以mass1和1e-9启动BEAM，逐prefix pb/pnb比例1e-9，容差同P01；最终mass降序rank完全相同，平分仍保留首次插入顺序。 |
| P06 / 113 / `test_06_blank_last_writer_is_retained_without_input_mutation` | reuse | 一个prefix1027，mass(.4,.2)，末node frame0/prob`.2`；row6 token1027`.6`、blank`.4`。输入不变；目标prefix1027的**最终**末frame仍0。此为保留的blank后写metadata规则，不能“优化”为frame6。 |

## 4. 静态识别的移植陷阱与不变量

1. **输入/数值精度须先定合同。** 原fixture中的`.05/.9`都是Python binary64，现有`kws3_step`用double输入对齐。`kws3_step_f32`规定逐值精确提升：`.05f`升为double后略大于`.05`，因此通过`p>0.05`是该真实float输入的正确语义，不能假装它仍等于Python十进制`.05`。原fixture走double入口；未来float32对照须将相同binary32值提升后再交给Python oracle，不扩大/四舍五入floor以遮掩差异。pb/pnb不能降为float：固定窗口下`0.05^126≈1.1755e-164`仍在double正常数范围，却远低于float范围。不使用fast-math/FTZ改变契约，不做重新归一化或log域改写。
2. **全类top3先于token过滤。** 完整2599类校验并以概率降序、id升序取3；随后只保留`{0,1027,1401,1462,2093}`且原概率严格`>0.05`。无任何合格token时直接保留旧hyps，不新建空beam、不衰减概率、不reset。输入和必须接近1，容差`1e-5`，不更改行；C累加策略若不同于Python `math.fsum`，需单独声明边界误差，不顺便改变容差。
3. **顺序是语义。** retained beam是“筛后top3 token外循环，旧hyp内循环”。合并表记录第一次插入prefix的顺序，最终按pb+pnb**稳定降序**；同mass不是按prefix词典序，也不能交给不稳定qsort。下一行沿用此顺序。C哈希表遍历顺序不能成为平局规则。
4. **只修指定三处。** 重复token两分支分别判断`pnb>0`、`pb>0`，不能用1e-6 epsilon。更新重复token末node时需隔离输入/兄弟值。其余metadata赢家规则保留，特别是blank分支无条件复制该来源nodes，可能覆盖同prefix稍早产生的更新；P06不是错误golden。
5. **检测是第一命中，不是最佳合格命中搜索。** 遍历mass排序后的hyp，再按关键词1、2；suffix函数返回最早匹配起点，含最后合法起点。遇到第一个匹配就计算`score=sqrt(p0*p1*p2*p3)`，之后检查duration/threshold。该匹配duration不合格时，不继续寻找别的path、词或同path的后一个合法匹配。每次检测从hit_score=1开始。
6. **固定寿命，不是老age expiry。** 行n先在n%42=0启动[start=n, stop=n+126)的窗，所有活动窗处理当前n，再将rows增加1并移除stop≤rows的窗。局部candidate reset只重置beam/activation/score，绝不改start/stop/global clock，也保留本窗last_active_pos；源局部interval=0意味着其后候选end不能倒退，并非可删除所有local last-active语义。正常流新节点时间递增，不应触发此边界。EOF不padding、不flush新事件；重入FINISH幂等，EOF后STEP报错。
7. **三个层次的时刻分开。** node_frame、end_frame单位是10ms的基准帧，行步长3；decision_row和decision_frame=3*row说明何时决定。窗口起点也是输出行索引。不得把window局部frame或end_frame当真实声学available_at/告警延迟。
8. **先局部消费，再全局merge。** 任何局部duration有效候选都reset搜索，即使随后duplicate或shared refractory拒绝。两个词共用last_accepted_end；不把global拒绝反馈成保留局部旧词。
9. **merge顺序与seen写入次序。** 每行按`(-score,end,keyword_id,window_start,node_frames)`排序；同key`(keyword_id,四绝对node_frames)`保留当行最高score，同score最早window。跨行已seen的key不回写。对每个新key先加入seen，再看`end-last_accepted_end>=50`，所以冷却拒绝也占seen。不能只记录已发事件。
10. **精确淘汰边界和有符号差。** 在处理本行候选前淘汰`stored_end <= 3*row-378`。前126行减378不能发生unsigned下溢；refractory允许负差，T11中晚end先接受后早end必须被拒绝。`last=-1`哨兵不能无审查地改为无符号最大值。检查`3*row`、`start+126`和计数器溢出，显式失败优于绕回。
11. **指标取样位置不能随意移动。** prefix/max_next/empty/zero是该窗beam更新、path_beam剪枝后、candidate reset前；max_live_node_references是所有局部reset后、窗退役前。最大常驻node引用7560、一次暂存hyp≤80且每prefix≤126；去重≤378。若C用值复制，节点“引用数”应明确为等价逻辑数量，不当成实际字节RSS。
12. **错误锁存与数值诊断不同。** 输入/容量/不变量错误使整会话invalid，已发诊断候选不能当最终有效结果。empty/zero计数单独影响numerical_support_ok；源v2并不仅因这两个计数非0自动置valid=false。若C想更严格fail-closed，需要明确列为有意偏离，不能悄悄混淆两个状态。
13. **固定配置不是模型准入。** threshold0只输出研究候选；仅两个canonical词；无宽容alias、无声学forward、无FAR/FRR或SSC305资源结论。几何覆盖不保证beam裁剪、其他局部候选reset或声学错误后的召回。

## 5. 必须具备的内部测试能力

建议在单独测试翻译单元/编译宏下暴露这些能力，或让测试编译单元包含实现以调用内部静态函数。不要为方便测试给生产输入添加任意内存/状态注入入口。

| 能力 | 最小需要 | 覆盖 |
|---|---|---|
| `TOP3` | 对给定2599 double行返回固定3个id及原概率；有计算计数/缓存快照；不允许consumer改内部缓存。 | V13、T13、T14 |
| `SEED` | 仅测试：构造有序`kws3_test_hyp`数组，含prefix、pb/pnb、node frame/prob，token来自同下标tokens；构造两hyp逻辑共享首节点值；容量/对齐错误需可控失败途径。当前global last-end通过预先MERGE设置，非直接内状态注入。 | V05..V08、V15、P04、P06 |
| `BEAM` | 独立一步，不走窗口启动/退役/检测/全局merge；接受指定初始mass、完整行/已算top3；返回未剪枝hyp集合及稳定顺序，保留输入快照供比较。 | P01、P04、P05、P06 |
| `DETECT` | 在种子hyp上检查first-match、suffix、score及duration，提供未激活hit_score诊断；可用非3网格节点测精确边界。 | V05、V06、V08 |
| `MERGE` | 在指定decision_row直接注入最多3个合法候选；读取events、duplicate/refractory、seen数、last-end；不要求候选来自真实逐行声学流。 | V07、T09..T11、T16、T21、T22 |
| `INSPECT` | 只读活动窗start/stop/hyp和全部summary；节点值快照；不得依赖非稳定结构padding做memcmp。 | V11..V12、T03、T08、T14..T15、T19 |

无这些能力时，仅通过公开STEP不能完整复现46映射：duration5/250、end-gap49/50、同decision_row的历史候选排列、0/1e-9初始mass以及节点别名污染等都是刻意构造的内部状态。应如实记为“未覆盖/需要hook”，而不是用近似流替换后声称完整通过。

当前hooks与smoke准备补充（均未运行）：`kws3_test_detect_seed`已增加未激活时的diagnostic event.score；该score不是输出事件。smoke重复检测duration4的`.8`节点词，要求两次score相同且约`.64`，覆盖V06不累积风险。V15在smoke分别用不足输出cap和内部prefix扩展超过126来检查真实`kws3_step`的failure latch/未半提交。`kws3_test_beam`单独返回capacity仍不等于会话锁存。`kws3_test_top3`每次调用本身会重算，不可调用4次却期待缓存计数1；T13须通过STEP统计、消费者快照隔离和三窗读取证明一次计算与不可变性。

## 6. 未实现完整映射扩展预算提案（非当前执行计划，需另行放行）

### 6.1 固定范围与停止条件

本节仅假设未来把完整41个C语义槽位全部实现为纯C测试；**当前并未完成该扩展，当前smoke/differential以`PLAN.json`为准**。5个history-only项以来源记录核对，不运行旧decoder。假设的扩展方案为单线程、单进程顺序运行，不下载/安装、不调用Python/Torch、不加载模型权重/音频/真实posterior。构建与运行不得与当前重CPU训练抢占；由父任务选择获准时段。

建议分两阶段：A. 一个严格double语义构建及单次合成测试；A通过且预算未超后，B. 一个地址/UB sanitizer构建及单次同套测试。每阶段最多1次编译和1次测试进程；失败即记录并停止，不自动循环编译/重试。是否同时授权A与B由父任务决定。编译器须使用已有官方工具链，不为本方案安装软件。

建议构建约束：C11，warnings enabled并视为错误，禁用fast-math；sanitizer仅在已有工具链支持时使用，缺失不得安装补齐。测试构建启用hooks，生产构建不启用。没有O/-OO的Python运行需求，也不把NDEBUG差异等同Python优化模式。

### 6.2 每次合成测试的静态上限

下面是**从源fixture输入长度逐项推导的计划上限，不是测量值**。种子/检测/merge测试若不实际调用STEP，实际row调用将更少；不得借“剩余预算”增加未列测试。

| 范围 | 每次测试最大逐行输入尝试数 | 独立BEAM更新 |
|---|---:|---:|
| v1的15个非history映射（V10同时含4行fresh-control；seed场景各预留源step次数） | 324 | 1（V15独立prefix超容量请求；生产cap锁存仍计左侧） |
| 三窗的21个非history映射（T02只运行新三窗85行，不运行old85行） | 4066 | 0 |
| patch的5个非history映射（P03新侧28+70行） | 98 | 30 |
| **每次上限** | **4488** | **31** |

逐行数校核表达式：

- v1：`4+4+175+25+1+2+2+4+89+1+1+1+10+1+4 = 324`。
- 三窗：`94+85+1104+754+516+4+127+115+2+1000+5+260 = 4066`；纯几何、merge和独立cache测试不消耗STEP行。
- patch：P03 `28+70=98`；P01独立beam `2*(1+2+3+4)=20`、P04 `1`、P05 `8`、P06 `1`，合计30；另给V15一次独立beam超容量请求，合计31。

因此每次保守beam预算为`3*4488+31 = 13495`次；逐行输入的全类检查/top3最多4488行，相当于不超过`4488*2599 = 11664312`个输入概率位置（无效/seed场景实际上更少）。独立beam hooks也会各做一次全类校验，另加31行；TOP3专项最多预留6次完整行检查。包括这两项的总全类检查上限为4525次、`4525*2599 = 11760475`个概率位置。不预存全后验矩阵，只复用一行2599 double输入和固定状态。

额外固定逻辑工作：T04为3444个整数几何组合；P01路径枚举每scale有`3+9+27+81=120`条终路径，两scale共240条。按当前无global-last seed hook的方案，独立MERGE约定总上限517次（T09/T10/T11/T16/T21/T22共513，V07基线加边界共4），最多1534个注入候选；wrapper内部每行merge已包含在逐行预算。

若A、B均放行，总计最多2次编译、2次测试进程、**82个C槽位结果**、8976次逐行输入尝试、62次独立beam更新、26990次总beam更新上界、9050次全类校验/top3上界、6888个几何组合、480条oracle终路径。不能称“46×2 C功能测试”：10个history记录不属于运行功能用例。

资源门限提案：每次编译wall≤30s，每次测试wall≤30s，测试进程CPU≤30s；超过即停止并报阻塞，不为完成宣称私自扩大。RSS需报告实测且保留sanitizer映射开销与真实RSS区别；不从这里外推SSC305/4h资源。可先由实现的静态workspace大小决定合理RSS上限，不能在不知道结构布局时承诺板端内存。

### 6.3 首轮之外的 C 专属补强（不含在上述许可/预算）

后续确有需要再单独列预算，不自动运行：uint64/size_t时钟溢出、初期`current_frame-378`下溢、score/mass稳定平局的prefix顺序、duration无效首匹配挡住后合法匹配、无保留token的state保持、归一化和容差边界、输出数组容量不足/NULL指针约定、finish/reset与recording标签所有权。上述46源项对这些C专属边界不构成完整证明。

## 7. 本次读取的源身份

以下为本次静态`sha256sum`输出，不是编译/测试输出：

| 源文件（历史目录内） | SHA-256 |
|---|---|
| `test_posterior_events.py` | `d99a79e6984bb7cb60954d2c0ae3cc36230779c91f451dd32d504bcb79152b4c` |
| `test_three_window_events.py` | `c9a0584ef6e5d22488e8f3b831b95c0771b16e480f1858b8542f6ff1d90f1ed9` |
| `test_beam_minimal_patch_candidate.py` | `068d025c1e55101f0cec717471b7dbb33e8d38eb975fa8b1dec26e05a98a5889` |
| `three_window_events.py` | `f65403a78bbe55566cae914062b1671c055c198a87fb7cb5c000d0a80d60cd26` |
| `posterior_events.py` | `56d6ada2c531de25f72a713db77b07ef9d8326b8b1e238bd776d40155d8584ec` |
| `beam_minimal_patch_candidate.py` | `f822a40a136ea216cd9a3dd4dc9257000d69e15820cb6276360d9518a9800360` |
| `vendor/stream_kws_ctc.py` | `2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53` |
| `vendor/decoder_tail_fix.py` | `5fc55921a555c78cb08c12018850652d98d4283594f875366284dc1df0e71f4d` |

历史FINAL-BUDGET所记46×3 Python运行结果属于2026-10-17的既有合成证据；本次没有重跑、增补或借用其结果证明新C。新目录中应保留独立的源准备身份、未来构建/运行记录与本映射版本，避免混淆。
