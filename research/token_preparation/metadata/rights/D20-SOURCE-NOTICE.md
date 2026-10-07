# Fixed20 D20 synthetic research audio provenance

This fixed corpus contains exactly 20 original 16kHz mono PCM16 WAV files, 31.12 seconds in total. WAV and decoded-PCM byte identities are recorded in the accompanying source mapping. The files were recovered from saved evidence and were not regenerated for publication.

- 13 recordings use Qwen's Vivian stock preset and 7 use its Uncle_Fu stock preset
- tts-candidate-011 uses Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice at revision 85e237c12c027371202489a0ec509ded67b5e4b5
- The remaining 19 recordings use Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice at revision 0c0e3051f131929182e2c023b9537f8b1c68adfe
- These are publisher-provided synthetic stock presets. No user-provided reference voice or real-person recording is identified in the source-generation records

The pinned official model cards declare Apache-2.0:

- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/README.md
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/README.md

Recorded model-card SHA-256 values are 1e3adeecc7a72d6756fdb77c2847f8e994195e105b51206a7a5c049b0dfa48a8 for 0.6B and 64f65e809b51cc0c35f393fbbfcc2d735c0cb3fdbbc6f3fdc4a5e6ce55e9d088 for 1.7B. The 1.7B card was also freshly read at its pinned public revision during this audit.

The model-card license is a model/source provenance declaration, not a blanket new license or warranty for every generated audio output. No additional commercial output license is established by this research publication. Preserve commercial_output_license: not-established. Do not relicense third-party contributions under a blanket project-wide notice.

All 20 transcripts are weak pseudo-labels derived from synthesis intent and normalized full-transcript agreement among three ASR systems. They are not human-gold labels, and no event/phoneme timing gold exists. This is an isolated research archive; ordinary trainer, product and qualification admission remain false. Eleven sources are keyword-positive by the historical weak labels and nine are roof-word foils. These inputs were used to fit the reported F/A checkpoints, so they are not unseen test data.

The separate 18 Serena diagnostic recordings are not included by this fixed20 audio scope. Neither this source recovery nor D20 fit repairs historic strict-FP32 native failures or establishes hardware/product/generalization qualification.
