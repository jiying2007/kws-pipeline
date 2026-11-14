# English63 uninterrupted native C-ABI KWS experiment

## Scope and reason

The previous source-verified FP32 and INT8/chunk16 replay proved
**five original-PCM events on 63 individually reset Qwen synthetic clips**
(plus four events observed only after a non-natural 660 ms added silence).
Real native x64 C ABI also reproduced those five events under fixed 320,
1600 and uneven input boundaries.

This new experiment asks what happens when an always-on stream is NOT
reset at the boundary of every recorded source sentence. It is only a
synthetic concatenation diagnostic, **not household audio or real field FAR**.

## Fixed replay configuration

- Reuse exact original 63 PCM16 WAVs, seven synthetic source families and
  request metadata from source ZIP SHA256
  f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc
  and independent three-evidence review JSON SHA256
  89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12
- Reuse the actual previous no-padding, individually reset Python readback
  source ZIP SHA256
  4f006138eb9dcc6f856ef0e61da8dcf10034904e4e683255db302775a19a62f6.
  The old reference is not re-executed
- Same original official Zh-En 3M INT8/chunk16 encoder and joiner, FP32
  decoder, original HEY_MERO phoneme alias and strict ID3. Model release
  SHA256 68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
- Same source-pinned official Sherpa v1.13.8 x64 C API/ORT runtime,
  compiled with the opt-in verified official-release host build.
  The legacy host runtime lock is not weakened or replaced
- Exactly two predefined orders for original source clips:
  sorted cell ID and source-family-then-requested-text order.
  Each order contains all 63 original PCM16 files once, without any
  inserted silence between files. The PCM samples are simply supplied
  sequentially to one KWS stream; no WAV or derived recording is published
- Three predefined caller partition sizes: fixed320, network1600 and the
  repeated irregular 1/159/321/17/640 sample pattern, using the existing
  C ABI staging buffer. Model threshold .25, score 1.0, feature80, paths4,
  CPU1 and original official words are unchanged
- One reset at stream start, no per-source-utterance reset, and one EOF
  at the end of the full 2,101,760-sample sequence. Feed only real PCM,
  never append artificial silent PCM
- Record all exact C event ID3 outputs and sample positions, mark ambiguous
  source concatenation boundaries, and compare identical event positions
  across caller partitions within the same ordering. Do not call a source
  interval an acoustic word-level label

## Rerun from exact prior public sources

Use research/sherpa_pcm/build_official_x64.py to construct the research
native library and its hash-bound build receipt. The model and audio are
passed through absolute create-only private runner paths.

    python3 research/sherpa_pcm/tests/english_contiguous_pcm_c.py \
      --native-library /private/host/libkws_sherpa_pcm.so \
      --build-receipt /private/host/build-receipt.json \
      --tts-zip /private/original-english63.zip \
      --review-json /private/original-review.json \
      --model-archive /private/original-zh-en3m.tar.bz2 \
      --baseline-archive /private/unpadded-events.zip \
      --output /private/new-contiguous-result.json

The guarded no-audio/no-NumPy/no-model source regression suite is included
in the existing research-sherpa-pcm workflow. One source-owning public PR
may execute the real experiment and publish machine-only metrics; afterward
close its one-off workflow PR instead of permanently adding an unused job.

## Interpretation and hard gates

Differences between 63 individually reset clips and a continuous
2-minute concatenation are useful evidence about decoder context and
stream state. Source clip boundaries are arbitrary splice points.
Neither order is natural home audio, guaranteed source-disjoint
independent truth, a speech-accurate FAR denominator, a human FRR test
or true word-ending latency.

No new TTS, ASR, training, model tuning, corpus additions, voice cloning,
shipped or SSC305 KWS claim is authorized. Genuine English microphone
positives/accents, continuous multi-hour home audio, objective false
alarm rate, SSC305 BSP/ABI, CPU/RSS and decode P99 remain NOT_RUN.


## Actual pinned native C model experiment (2026-11-14)

The real GitHub Actions run 43551611855 succeeded using authentic unchanged
original Qwen PCM, three-evidence review, original no-padding reference,
pretrained Zh-En INT8 chunk16 model, and the previously verified actual
Sherpa v1.13.8 x64 C API library. The original machine artifact
14095938700 has ZIP SHA256
e72a9f1af5b645504f8504f97474c8a0dc337bbd90f3956407c76315bdde9bea.
An independent byte readback verified the inner JSON file (133,134 bytes)
has SHA256
d9dae86d9547069063cc8adfb342ea964d63ee65940e4d92d35ce1ef49a367ab.
The source implementation and exact 6-session observations are bound in
ENGLISH_CONTIGUOUS_RESULT.json.

All six uninterrupted native C runs completed:
- Original cell-ID ordering: 11 events, identical event sample positions
  for all 320/1600/uneven caller input patterns; zero EOF-only events
- Original synthetic voice/phrase ordering: 7 events, again identical
  input-pattern event positions; zero EOF-only events
- Of the 11 recorded events in cell-ID ordering, 7 were mapped to source
  intervals requesting Hey Mero and 4 to other requested phrases; one
  event fell exactly on a source splice boundary
- Of the 7 events in voice/phrase ordering, 5 were mapped to Hey Mero
  request intervals and 2 to other request intervals; one event fell
  exactly on a splice boundary
- The model is **partition invariant under the tested caller boundaries**,
  but **depends on the synthetic order and preceding decoder context**.
  Counts differ from the five previously observed individually reset
  source events. Both results remain genuine machine observations

An event mapped to a non-Hey-Mero requested interval is NOT automatically
a trusted false alarm: the synthetic clips were artificially concatenated,
recorded request text is not independently human-reviewed acoustic truth,
and the decoder may span splice boundaries. No field FAR/hour or FRR can be
inferred. The variation makes real unbroken background and correct
end-of-word event timing higher priority than further quantization tuning.
