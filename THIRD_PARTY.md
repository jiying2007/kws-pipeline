# Third-party references

The runtime in this repository is a clean-room implementation. The following projects informed architecture/evaluation choices and must be reviewed under their own licenses before copying code or assets:

- `k2-fsa/sherpa-onnx` — open-vocabulary/customized KWS and keyword-list decoding; useful as a richer fallback/benchmark implementation.
- `wenet-e2e/wekws` — production-oriented small-footprint streaming KWS and evaluation practices.
- `jesserockz/microWakeWord` — lightweight streaming feature cadence and temporal-consistency wake-word design for constrained devices.
- `jensen199105/aispeech-earbuds` and `jensen199105/aispeech-training` — embedded speech pipeline/training references supplied for this project.
- `jinchao123/audio_ai_pipeline` — embedded audio-AI pipeline reference supplied for this project.
- `jiying2007/audio-pipeline` — sibling low-compute DSP SDK whose memory ownership, build and target-board validation principles are mirrored here.

No pretrained weights from those projects are redistributed by `kws-pipeline`.

`research/sherpa_host` retains an extracted Sherpa KWS C API wrapper and one
bounded-Viterbi decoder patch from commit
`11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf` under Apache-2.0, with the original
copyright notice and the existing license in `research/sherpa_pcm/vendor/`.
Its offline recipe pins selected external sources and their complete retained
licenses/notices; see `research/sherpa_host/PROVENANCE.md`. It reuses the identical
public header and adds no model, runtime binary or default product dependency.

## Optional research adapter source

`research/sherpa_pcm/vendor/sherpa-onnx/c-api/c-api.h` is copied unchanged from
sherpa-onnx1.13.8 commit `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf` under Apache-2.0.
The upstream license is retained at `research/sherpa_pcm/vendor/LICENSE.sherpa-onnx`;
source URL and SHA256 are recorded in that research directory's dependency lock.
This optional research entry does not add sherpa/ONNX Runtime to the default
product library. No third-party model, runtime binary or wheel is redistributed.

`research/donor_fbank` is a separate C11 research implementation of a fixed
Kaldi/Hamming logfbank recipe. Its FFT skeleton is adapted from this repository's
Apache-2.0 frontend without modifying or linking the product frontend. Mathematical
Hamming/mel tables and synthetic reference vectors were generated through the
pinned torchaudio Kaldi source; its BSD2 license is retained locally and in the
fixture. The donor model-specific Apache2 declaration and CMVN provenance are in
`research/donor_fbank/fixtures/`; no complete acoustic-model weight file is included.
NumPy is a hash-pinned test dependency only. The native library uses libm, and this
research path does not enter the default product build or qualification gates.

`research/donor_fsmn` implements the fixed WeKws FSMN/splice contracts from
`wenet-e2e/wekws` commit `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`.
Its upstream Apache-2.0 license and attribution are retained in that directory.
The official2599-output donor weights are external local research inputs, not
included assets or self-authored parameters. This optional C11 research path
retains failed numerical gates and does not enter the default product build.
