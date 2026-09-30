# 中文高质量 TTS 候选筛选（reviewing）

## 与唤醒词任务相关的排序

以下按 2026-09-28 官方模型卡/代码筛选；本机已实际生成 Qwen3 0.6B CustomVoice、CosyVoice 300M SFT、Melo、Kokoro、Spark-TTS 和 VoxCPM2。用户已给出 Spark 来源级试听排序：Qwen3 更好，Spark 可用且优于此前其他来源；用户也反馈 VoxCPM2 实听与 ASR 疑点相近。这些都是来源级判断，不能推导所有单条录音的真值。2026-09-28 用户明确当前处于**非商业研究**，因此按各自协议将限制非商业用途的 TTS 也纳入实验候选；这不会将其输出自动归类为将来可用于商业模型的语料。所有新增模型均须固定 revision、权重/代码哈希，先离线生成少量“你好小窝”“小窝小窝”及近邻，再按[固定 ASR 标准](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)保守筛选，必要时少量抽听和 C runner 回读。

| 优先级 | 候选 | 对本任务的意义 | 当前约束 |
| --- | --- | --- | --- |
| 1 | [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | 无参考音频的 voice creation 可控制男/女、音高和五档语速；独立于现有 Qwen3-TTS 声码器/推理链 | 模型权重 CC BY-NC-SA 4.0，**只列研究用途**；官方推理代码固定 commit `2f1ea9082400547242641f5271b6f941c9f439d1`，模型 revision `642071559bfc6346c2359d19dcb6be3f9dd8a05d`。本机已生成 6 条基础 + 12 条语速探针，详情见[Spark 研究预检](SPARK_TTS_RESEARCH_PILOT_2026-09-28.md)；逐条人工收据待取得 |
| 2 | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 与 Qwen3 不同的 tokenizer-free 扩散自回归架构；官方支持中文、无需参考音频的文字声音设计，可指定性别、年龄风格、情绪及语速；权重/代码标 Apache-2.0 | 2B 参数；固定模型 revision `32279effe8c19989596f05d353d1447f51d9e915` 与代码 commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`。本机[六条 CPU 小样本](VOXCPM2_RESEARCH_PILOT_2026-09-28.md) ASR 仅 1/6 精确，用户来源级实听认为与 ASR 疑点相近，停止扩量；“年龄风格”不是真人年龄覆盖 |
| 3 | [FireRedTTS3-Instruct](https://huggingface.co/FireRedTeam/FireRedTTS3) | 独立模型家族；官方支持不提供参考音频的文字声音设计，描述性别、年龄风格、音色、速度、口音，模型卡标 Apache-2.0 | 官方完整模型仓约 20.8 GB，当前本机 CPU 小批量可行性未验证；先确认权重/依赖和资源预算 |
| 4 | [Qwen3-TTS 1.7B VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | 文字描述设计声音，无需参考音频；适合在现有 Qwen3 听感基础上补音色/风格，模型卡标 Apache-2.0 | 同 Qwen3 家族，**不是独立生成器留出集**；模型仓约 4.52 GB，CPU 速度需实测；声音风格不代表真人年龄 |
| 5 | [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | 官方宣称改善内容一致性与韵律，可作为已有 CosyVoice SFT 发音失败后的不同版本对照 | 官方常用流程依赖参考音频；没有权利明确的参考音频前，只研究可用的无参考模式，不启动克隆；应视为 CosyVoice 家族，不作完全独立来源 |

Spark-TTS 与 VoxCPM2 均已完成本机小样本；后者机器初筛和来源级实听均提示发音问题，已停止扩量，不能按官方通用榜单继续推进。其后仍可试 **FireRedTTS3-Instruct** 作为另一家族。Spark 的底座也使用 Qwen2.5，但并非现有 Qwen3-TTS 的同一模型或推理链；不同家族仍不能证明训练语料完全独立。Qwen3 VoiceDesign 用于扩展 Qwen 家族内部音色，不用于证明跨生成器迁移。官方质量指标来自各自测试集，不等于“小窝”的发音准确率，更不等于目标板唤醒指标。新的批量自动入选按[固定 ASR 标准](SYNTHETIC_ASR_ACCEPTANCE_2026-09-28.md)，已有人工结论另列。

## 非商业研究候选的用途边界

- [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) 的**权重**从 Apache-2.0 改为 CC BY-NC-SA 4.0，研究用生成样本单独标识 `research-only-cc-by-nc-sa-4.0`，不混入未来商业训练池。
- [F5-TTS 中文/英文官方预训练权重](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md)标 CC BY-NC 4.0，可作为另一种研究模型，但零样本语音需要参考音频；参考声源的许可和 split 依赖都要单独记录。
- [IndexTTS2](https://github.com/index-tts/index-tts/blob/main/LICENSE) 的协议禁止用其改进其他 AI 模型，**但明确例外包括非商业 AI 模型**。当前非商业研究可单独预检；若用途改变，需要重新核对其输出是否可进入后续训练池。该结论基于条款文字的任务适用性判断，不代替权利方解释。

## 下一步小样本门槛

每个新家族先固定 2–4 种文字声音设计，每种只做双目标词与“窝小窝”等近邻；先看是否完整读出“窝”、是否吞音、是否因提示词产生多余开头或停顿。通过后才加入不同语速、句中位置与声学场景。候选声线和干净音频按家族及原声身份切分，场景派生跟随原声；只用同一 C runner 对比现有模型与单变量训练候选。真人目标词、真实年龄/性别分布、最终 AFE 和目标板仍是产品资格的独立证据。
