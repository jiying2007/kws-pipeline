# Third-party references

The runtime in this repository is a clean-room implementation. The following projects informed architecture/evaluation choices and must be reviewed under their own licenses before copying code or assets:

- `k2-fsa/sherpa-onnx` — open-vocabulary/customized KWS and keyword-list decoding; useful as a richer fallback/benchmark implementation.
- `wenet-e2e/wekws` — production-oriented small-footprint streaming KWS and evaluation practices.
- `jesserockz/microWakeWord` — lightweight streaming feature cadence and temporal-consistency wake-word design for constrained devices.
- `jensen199105/aispeech-earbuds` and `jensen199105/aispeech-training` — embedded speech pipeline/training references supplied for this project.
- `jinchao123/audio_ai_pipeline` — embedded audio-AI pipeline reference supplied for this project.
- `jiying2007/audio-pipeline` — sibling low-compute DSP SDK whose memory ownership, build and target-board validation principles are mirrored here.

No pretrained weights from those projects are redistributed by `kws-pipeline`.
