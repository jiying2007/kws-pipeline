# Attribution and distinct licenses

This harness prepares a prospective synthetic-source screen with weak machine evidence only.
It loads the unmodified FunAudioLLM/SenseVoiceSmall model from Alibaba's
FunASR/SenseVoice project and Qwen/Qwen3-ASR-0.6B from the Qwen team, at the
revisions and hashes in model-locks.json. No model weights are distributed in
this source candidate or output artifact.

SenseVoice model weights are covered by the **FunASR Model Open Source License
Agreement v1.1**, not by the code project's MIT license. The retained agreement
is FUNASR_MODEL_LICENSE (5306 bytes, SHA256
7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8).
Source: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
Pinned model-card reference:
https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/README.md
The name and source/author attribution remain in model-locks and output metadata.

Qwen model source:
https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0
Upstream package and model licenses remain separate from this harness. Installed
package .dist-info license files are retained in the environment; this candidate
does not relicense dependencies or substitute a code license for a model license.

The native-fbank frontend is the retained kaldi-native-fbank1.22.3 route:
FP32, CPU, no torchaudio fallback, no VAD, no ITN, dither1.0, snip_edges=True,
upsacle_samples=True. Dither's exact RNG reproducibility is not established.
Qwen keeps empty context, automatic language, no hotwords, eager attention,
batch1 and max_new_tokens256. Decoder completeness is not acoustic completeness.
