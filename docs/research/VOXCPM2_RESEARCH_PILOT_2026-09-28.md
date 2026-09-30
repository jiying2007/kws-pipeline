# VoxCPM2 独立家族短词预检（reviewing）

## 来源与运行

实际导入 [OpenBMB VoxCPM2 官方模型](https://huggingface.co/openbmb/VoxCPM2) revision `32279effe8c19989596f05d353d1447f51d9e915` 及[官方源码](https://github.com/OpenBMB/VoxCPM) commit `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`。模型卡与源码标 Apache-2.0。本轮使用无参考音频的文字声音设计，不克隆真人声音。模型权重 SHA-256 `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`，AudioVAE SHA-256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`，配置 SHA-256 `405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c`。官方 `audiovae.pth` 由源码按 `torch.load(weights_only=True)` 读取；`model.safetensors` 用 safetensors 读取。

Python 3.12、CPU、隔离 `--network none` 容器，禁用可额外下载的降噪器、文本规范化和坏例重试，不启用 `torch.compile`；`cfg_value=2.0`、`inference_timesteps=10`、`max_len=256`。两种中文文字声音描述（普通话女性/男性、自然语速、清楚发音），每组固定双目标词“你好小窝”“小窝小窝”及缺首近邻“窝小窝”，seed 4101–4106。模型加载约 48.8 秒，六条生成各约 15.7–34.3 秒；共 7.68 秒音频。模型原生输出规范化为 16 kHz 单声道 PCM16，六个 WAV SHA-256 各异，manifest SHA-256 `ae0c1453fcc189a20959002bb511cd4e7b4c62815d625053bf69b50ef8ed9202`。文字描述的性别不是已验证的真实说话人性别，更不代表年龄覆盖。

## 发音与 C 端预检

| 文字声音描述 | 文本 | 独立 ASR 转写 | 旧 C 模型事件 | Qwen3 混合模型事件 |
| --- | --- | --- | --- | --- |
| 女性 | 你好小窝 | 你好小窝 | 词1 | 无 |
| 女性 | 小窝小窝 | 笑我笑我 | 无 | 无 |
| 女性 | 窝小窝 | 我小我 | 无 | 无 |
| 男性 | 你好小窝 | 你好小吴 | 词1 | 词1 |
| 男性 | 小窝小窝 | 小沃小 | 词2 | 词2 |
| 男性 | 窝小窝 | 我郭小波 | 无 | **词2** |

独立 ASR 摘要 SHA-256 `dfc2af5678d397f800177ee801e59cd469c2c3d7e1472264c491b1ab4ac0a296`，C 输入 corpus SHA-256 `8b4b1ceaf103e48f2ba3179b3d58c643a1495cb5ec4fcaf4564f63c2099d3486`。四条文本标记的正例只有一条 ASR 全词精确转写，其余有疑似错读或吞字；机器转写不能单独裁定发音。混合模型对男性缺首近邻触发词2，是该 C 模型在这一份音频上的误触发候选，仍要以实际听到的语义确认真值。**当前不扩量、不入训练或资格集**；六条先经逐条人工听辨。

用户审听六条后反馈“试听和 ASR 效果很像”。这是**来源级实听反馈**，支持将 ASR 所示的多处错读/吞音视为真实风险，因此继续停止扩充 VoxCPM2 本批文字声音设计。用户尚未提供每条 accepted/rejected 的 WAV 哈希绑定收据，不能推导哪一条已经逐条合格；六条均保持待裁决，不按原始文本加入训练。

本地[六条审听页](../../build/review-packs/kws-voxcpm2-review-20260928/review.html)和[便携包](../../build/review-packs/kws-voxcpm2-review-20260928.zip)已经生成，ZIP SHA-256 `bc4683ddde2e527163bc82ab55ea76e2a7d90a94f895c505050193ed4f18c913`。页面导出的 JSONL 收据绑定来源、WAV 哈希、预期文本、类别和匿名审听 ID。该来源与 Qwen3/Spark/CosyVoice 的模型家族不同，但不能单凭家族名证明训练数据独立，也不能用六条估计 FRR/FAR。

下一步：本轮先停用 VoxCPM2 作为目标词正例扩量来源。若后续仍研究这个模型，只试更明确的发音控制或极少量成对修正；仍错读则只保留为发音异常探针。若新的逐条审听合格，再扩声线与场景，并保持生成器、声音设计和原声身份隔离。
