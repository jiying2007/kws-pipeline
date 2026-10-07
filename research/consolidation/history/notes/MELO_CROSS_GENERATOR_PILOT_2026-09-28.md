# MeloTTS 中文单声线小批量回读（reviewing）

## 结论

第三条合成来源已在本地实际运行。[MeloTTS-Chinese 模型卡](https://huggingface.co/myshell-ai/MeloTTS-Chinese)标记 MIT，官方 [sherpa-onnx Melo 模型说明](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)说明它是单声线、44.1 kHz 的 VITS 转换包。经相同 ffmpeg 转为 16 kHz PCM16 后，五种短语 × 0.9/1.0/1.1 三档语速共 15 条均生成成功、无重复 PCM、无 `Unknown token` 日志。

2026-09-28 用户试听反馈 MeloTTS 相比 Kokoro“好一些”。这是来源级的主观初评，没有逐条接受清单；Melo 仍须经过哈希绑定的逐条审听，不能因此整体标记为合格。

然而，独立 ASR 对部分目标词给出缺词或不同音节：0.9 速“你好小窝”→“你好想窝”，1.0 速“小窝小窝”→“到我小窝”，1.1 速“小窝小窝”→“窝小窝”。C 端旧/扩充模型按输入文本看分别触发 4/6、5/6 条目标词，9 条负例均无触发；**这些事件不能证明实际发音正确**。不经人工听辨和逐条标签审查，Melo 样本仍为 `needs-human-audio-review`，不得并入训练或正式验证。该来源只有一个声线，语速变体不能充当独立说话人。

## 冻结身份与小批量结果

- 官方 sherpa-onnx 模型归档 SHA-256 `e58351ed7149f290a54534538badd4077cdbe6fddc964b24d0bee870415d1514`；23 个成员均为安全相对路径、普通文件或目录。包内 MIT `LICENSE` 与模型卡标注一致。`model.onnx` SHA-256 `bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d`，`lexicon.txt` SHA-256 `7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528`。
- 生成运行时与 [Kokoro 小批量回读](KOKORO_CROSS_GENERATOR_PILOT_2026-09-28.md)使用同一 sherpa-onnx v1.13.8 可执行文件 SHA-256 `dde4fbd93181de356d9dc32cd734407be0da64c10ad5883a5a1a1328e9fbc879`；ffmpeg SHA-256 `eedcb5eb3f8eb8486ba2d1e6d2796e0037f6002414ba5ffddf36a5228d52ca3d`。全部原始与规范化 WAV、ASR/C 明细只在 `/tmp/kws-kokoro-20260928/`，`review.html` 含两种生成链的本地逐条播放。
- manifest SHA-256 `012349428d173d04592e6964ec25e58e108e5b1416a0a867919b907e4796c060`；15 条、总长 10.99 秒，目标词 1/2 各 3 条，负例 9 条。独立 ASR 初筛摘要 SHA-256 `d120dd354ab8c7ac277f74a367450092d3d2cc7d08beb0aee5c3acfbdb37b17f`，只作错读探针，不作为人工真值。
- C runner SHA-256 `0599960febaccde62e09cede2a2adcba17a6d7665c792a2204f69ba1b21b3f39`，关键词包 SHA-256 `370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723`；输入 corpus SHA-256 `70afdeb8c5996f8ff70c7f8bacd5ef2968eb0494584fedea12dfcaff219dbff9`。旧模型在关键词 1/2 各触发 1/3，扩充模型各触发 2/3；两者均 0/9 负例触发。旧/扩充逐条检测 SHA-256 分别 `a13f4f65e361fe81f2633d3a3973013fd76d46860db8e7a9e3304d880f76d61f`、`007e11584f28cea612672927dae6452751b9a7431b4527a1d359b3cc00a62f09`。

## 方案判断

Melo 与原 AISHELL3 属于 VITS 大类，但其模型权重、训练来源、44.1 kHz 原生采样和语音前端不同；它能检验另一模型来源，不能单独构成跨网络结构验证。Kokoro 提供不同结构与更多声音，但 [实际样本](KOKORO_CROSS_GENERATOR_PILOT_2026-09-28.md)有未知 token 和更明显的短词转写疑点。两者的共同教训是：**生成文本不是音频标签，C 唤醒也不能验证词是否真的说对**。

下一步先在本地审听包标记每条的完整词、缺词和杂音，按 WAV 哈希形成 `--audio-review` 收据后再只导入通过审听的样本；导入后按生成器、原始语音、声线和增强派生关系隔离，保留已见 HI-MIA-CW 真人负例作开发回归。合成样本仍不能替代目标双词真人及最终 AFE/目标板。本文是 reviewing 草案，未训练或提升任何模型。
