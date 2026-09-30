# 非活动帧伪 token 与末字越界配对实验（reviewing）

## 预先冻结的机制证据

在原 128+12 条已审 Qwen3 混合训练模型、仅剔除 AISHELL3 第一词 16 条的处理模型，以及旧固定模型上，用相同 `kws_posterior_dump` 对 Qwen3 Vivian/Serena/Eric 的“小窝小窝”与 Serena/Eric 的“窝小窝”共五条音频生成 15 个 C 后验轨迹；清单 SHA-256 `fd604a5a7c37e284f3cfa99323b8a2c967b6c6d572d6256406a6c9b3ea728013`。旧模型在 Serena/Eric 完整第二词的末尾 `wo1` top-1 分别约 1.41/1.39 s 且 `speech_active=1`；混合模型约 1.60/1.41 s，两个末尾峰值的 `speech_active=0`。同一音频 VAD 帧相同，说明末字峰值相对固定 C 声学门后移。混合/剔除模型在开始非活动帧约 0.03–0.04 s 还共同产生伪 token `ni3`、`wo1`，并在缺首近邻的活动片段内产生多余的 `xiao3`，形成假完整路径；旧模型近邻主要是 `wo1,xiao3,wo1`。这不是单纯调关键词阈值即可修复的现象。

训练侧 `--ctc-vad-align` 以运行参数合同的 -55 dBFS 把非活动帧在**CTC loss** 中改为 blank-only，但这意味着模型原始输出在非活动帧不受该 CTC 分支梯度约束；辅助损失仍读取原始 `log_probs`。C 端在这些帧仍读取原始 token 后验，只有 terminal 触发受 VAD 活动约束。故提出一项单变量：在真实音频的 VAD 非活动帧、非空 target 样本上，用原始后验增加 blank 的交叉熵压力；不改声学前端、网络、词包、阈值、C decoder 或训练数据。recurrent release tail 另有原有目标，本次不重复约束追加的合成 tail 帧。

额外的只读诊断反事实：对上述旧/混合/剔除模型的 Serena/Eric 完整第二词与缺首近邻 12 条轨迹，保持后验和解码器不变，仅将原音频最后活动帧后的 100/200/300 ms 轨迹 VAD 标志人工延长，48 次 replay 清单 SHA-256 `7cb52df60bb70e3658aaba0cd958d29dde920dbab56e33ecdc3c23e10a10179b`。混合模型 Serena 完整第二词在 200 ms 延长后恢复词2事件，但 Eric 仍未恢复；两条“窝小窝”的误触发仍在。由此确认 VAD 末尾越界至少解释一部分漏检，却**不是**全部漏检与近邻误触发的唯一原因。人工改写轨迹仅作机制诊断，不是可部署或正式评估方案。

## 冻结实验与停止条件

- 对照：已完成的 128 AISHELL3 + 12 Qwen3，冷启动 RNN/H64、seed2346、600 epoch、batch16、lr0.001、VAD 对齐 CTC 及其余默认损失；KWM SHA-256 `d8b45c4586908e832091a9fc70fe43d34022e526197d5036afdedb6f5b2c41f4`。
- 处理：**同一 140 条及所有对照配置**，只启用 `inactive_blank_loss_weight=0.10`。该参数默认 0，只有同时启用 `--ctc-vad-align` 才能大于 0，checkpoint 与 KWM provenance 必须显式记录政策及权重。先做 1 epoch 输入/梯度烟测与来源 SHA 核对，再做 600 epoch。
- 评估：同一 C runner/词包，Qwen3 训练与未见声线两词及缺首近邻、AISHELL3 calibration/test、HI-MIA-CW 全量真人近邻。特别检查 Serena/Eric 第二词末字活动帧位置、非活动帧伪 token 和“窝小窝”误触发。已见开发集不能当作新鲜盲测。
- 接受条件：Qwen3 未见声线双词完整命中至少恢复到旧模型 3/4、无新增“窝小窝”FA，同时 HI-MIA-CW 事件不能高于旧模型 157。只改善非活动帧后验或真人负例总数而牺牲目标词，则负结果，不提升。
- `retry_budget`：本权重只跑一次预设 0.10；不在已观察小集上反复扫权重。`staleness_threshold`：源码、输入、C runner、词包或 ASR 证据身份变化即重做审计。`heartbeat`：训练/导出/评测每阶段实读 SHA。
- `goal_statement`：验证非活动帧监督缺口是否导致跨来源第二词末字越界和近邻假路径；`completion_claim`：仅在同 C 端回读后作开发结论；`logical_task_open=true`、`milestone_close=pending`、`stop_condition=replan`。真人双词与目标板证据仍是产品闭环独立缺口。

## 实施与同 C 端回读

`training/train_ctc.py` 新增默认关闭的 `--inactive-blank-loss-weight`；只有与 `--ctc-vad-align` 同时使用且权重大于 0 才启用，在真实音频非活动帧、非空 target 上计算原始后验 blank 交叉熵的每条均值，排除 padding 与原有 recurrent release tail。`training/export_model.py` 校验并保留 `development-real-inactive-blank-ce-v1` 与权重，不接受未绑定 VAD 的该目标。Python 3.12、torch 2.13.0+cpu 的定向梯度测试通过：只有真实非活动帧有该目标梯度；无 VAD 选项启用 0.10 被 CLI 拒绝。1 epoch 输入烟测和 600 epoch 训练均读取**完全相同的 140 条**，语料 canonical SHA-256 `47be2a7aacc24902805086f63c37eea1b146ba055f03f4988fdb2a984e5e4aa6`；checkpoint SHA-256 `90736ae0bda9066df91f55b1572baccca9861f93e5c51080d338e2a57ff1f818`，C KWM SHA-256 `d5f1e0f7daa487476c0c94ae81847bb9eec97ce37d5beedfa4174ec4889ba875`。KWM provenance 已回读权重 0.10、policy 和 `development_only=true`；无可发布容器 digest。

默认权重 0 的路径另按原 140 条/seed2346 做 1 epoch 回归，当前与此前 `reviewed-split/train-smoke.pt` 的 named-tensor raw-byte float state SHA-256 完全相同，均为 `8a42a2e0cfc090511fb8431eb562ec883d6467325b74a8e8c3587f983b7c6b27`。默认日志与 epoch history 不增加新字段，确保旧配方的数学与记录输出不因试验开关改变。

| 已观察开发来源 | 旧固定模型 | 原 140 条混合模型 | 非活动 blank 约束 0.10 |
| --- | ---: | ---: | ---: |
| Qwen3 训练声线目标词 | 4/6 | 6/6 | 5/6 |
| Qwen3 未见声线目标词、缺首近邻事件 | 3/4、0 FA | 2/4、2 FA | **2/4、1 FA** |
| AISHELL3 calibration 目标词、负例事件 | 14/16、0 | 9/16、0 | 10/16、0 |
| AISHELL3 test 目标词、负例事件 | 12/16、2 | 12/16、1 | 11/16、1 |
| HI-MIA-CW 16,343 条、6.6448 h 真人近邻事件 | 157 | 540 | **175** |

HI-MIA-CW 新目标的关键词1/2 事件 155/20，旧模型为 132/25，原混合模型为 430/110。新的目标大幅减少混合模型误触发，却仍高于旧模型，而且 Qwen3 跨声线完整词/近邻没有共同改善；**本模型不提升**。五条定向后验轨迹清单 SHA-256 `0d4b50505535c6e2b10147c9be5dcf3c2eaf86ba33ebfa2f7f7a88e57b2d18ca`：原混合模型在非活动开头的 `ni3,wo1` 伪 token 在新目标下消失；Serena/Eric 完整第二词的末尾 `wo1` top-1 回到 VAD 活动帧，C 默认仍未检出。这证明非活动帧监督缺口真实存在，也证明**末字越界不是完整词漏检的唯一根因**。

对新模型五条固定轨迹只改 decoder replay 参数的受限机制网格（默认、state retention 0.99、blank retention 0.9、fuzzy child cost -4、两 retention 合并）摘要 SHA-256 `d2f5e29ab9c050a8e0ff2407a0556455e366ec997f2f203923ce05493b317e11`。`blank retention=0.9` 可让 Serena/Eric 完整第二词两条均恢复 C 事件，但 Serena 的“窝小窝”近邻仍误触发词2。仅延长前缀生命或降低模糊 child 成本没有恢复完整词；调解码单项同样不能满足正负共同门槛。该网格只是已见小样本机制诊断，不用于选产品阈值。

本轮处理模型的 Qwen3、AISHELL3 calibration/test、HI-MIA-CW 检测 SHA-256 依次为 `e7ae28bb8a298e0c838d949fc9910ae9738eaaf412c7af4e0bad84f513a0e196`、`b98ec98a43aa11c5031b1d5be582d0ed2b53840454fe116f9dcf631ea6bca239`、`51788cff1e5ec3e12a3abfa6ec445afd7e4eeb888c9cc0eecd110868a9f785a5`、`d86ab2054b156a5959ed764b88b72a5a5056acb64e98a2cd41508e04cd04b929`。最终判定：`completion_claim=development-negative-mechanism-confirmed`，`logical_task_open=false`，`milestone_close=inactive-frame-control-negative`，`stop_condition=replan`。下一轮需要直接约束完整词与缺首近邻的**可分性**，并在新的说话人/来源与真人负例上封存回读；不继续同一小集上的 blank retention 或阈值扫描。
