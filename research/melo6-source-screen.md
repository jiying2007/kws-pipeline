# Melo six-phrase research screen

This is the current publication entry point. The READMEs and projection records
inside `melo6_tts` and `melo6_asr` retain their reviewed preparation history.
The current paired release is `melo6-source-screen-release.json`, and the exact
public data location is `melo6_asr/input-source.json`.

Six native 44.1-kHz mono float32 candidates and their fixed 16-kHz PCM16
derivatives were generated once each. The earlier session stopped at its
output-signature guard before any inference. Its loaded signature was not
saved and remains UNKNOWN. A reviewed guard repair accepted only unit batch
and channel specialization and saved the new signature before validation.
The two recorded attempts therefore contain two session constructions and
six synthesis calls, with no warmup or repeated phrase calls.

Source is published to `jiying2007/kws-pipeline`. Audio, generated evidence,
transcripts, and resource records belong to `jiying2007/kws-data`, under
`research/2026-10-05-melo6-source-screen/`. No audio bytes are committed to
this source repository. The blind transport pins one immutable data commit,
two fixed paths, exact lengths, and SHA256 values. The input commit is
[`fdf579cc43622cb5a752e7f5f3c191a149b7956c`](https://github.com/jiying2007/kws-data/tree/fdf579cc43622cb5a752e7f5f3c191a149b7956c/research/2026-10-05-melo6-source-screen). It performs one verified
TLS GET per input, rejects redirects, applies size and time bounds, and never
retries. A failed or partial transfer stops before package setup or ASR.

The transport-only adaptation preserves all existing ASR scientific Python,
the 124 package inputs, 13 model assets, decoder controls, and raw-result
freezes. The original source freeze and workflow are retained in
`melo6_history/`. The first creation of `research/melo6-source-screen-v1`
permits one standard public CPU job with a 50-minute limit. Each recognizer
decodes the same six opaque audio inputs once, at most 12 ASR calls overall.
The exact two input bodies add 171,800 bytes to the separately recorded
package/model response-body accounting. Infrastructure traffic is not a
globally measured total.

All six intended phrases remain fixed: 你好小窝; 小窝小窝; 你好小屋;
小屋小屋; 你好你好; 今天天气很好. The comparison is performed only after
generation, blind-input, and both raw-result freezes pass independent checks.
Recognizer agreement is weak machine text evidence. Human pronunciation,
acoustic completeness, voice rights, and speaker novelty remain unestablished.
CTC targets stay null; there is no training admission, source advancement,
speaker split, threshold search, or KWS improvement claim.

Private raw exception bodies and logs are excluded. Original failed records
are retained without relabeling; only safe diagnostic retention metadata is
public. Both repositories use independent draft PRs without merge or default
branch changes. Ordinary repository CI and the research execution have
separate evidence scopes.

## Pure checks

From `research/melo6_asr`, run `python -B -m unittest -v test_fetch_inputs`.
From `research/melo6_tts`, run the six test modules listed in the source
review: `test_conversion_recipe`, `test_adapter_mock`, `test_compare_melo6`,
`test_private_raw_cause`, `test_public_source`, and `test_output_signature`.
These checks use mocks and retained inputs and do not call speech models.

## 中文说明

本轮固定六句，每句仅合成一次。首次会话在推理前被输出签名检查拦截，
实际加载的旧签名未保存，仍为未知；独立审核后的修复仅接受单批次、单声道
维度具体化，并在检查前保存新签名。累计两次会话构造、六次合成调用。

源码存入 kws-pipeline，音频、转写及资源证据存入 kws-data。识别器仅接收
由不可变数据提交及精确哈希绑定的两个盲测文件，每个识别器最多识别六次。
机器转写一致不等于人工发音或声学真值；CTC 目标保持空值，不进入训练。
原始失败证据保留，私人异常内容和日志不公开，两个仓库均只建立草稿 PR。

The data archive verifier takes `research/melo6_history/tts-prior` as its
prior source and `research/melo6_tts` as its recovery source. Both are retained
here with their original plan/runner hash bindings.
