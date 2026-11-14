# Hey Mero: original PCM online input vs artificial-tail event study

Research-only frozen follow-up to the original 63-WAV FP32/INT8 chunk16
event study. Same official pretrained weights, 80-feature configuration,
one CPU thread, threshold=0.25 and 1,600-sample input partitions.

## Why the existing nine events need qualification

Original kws-data event run 43448541871 and quantization run 43537391922
appended 10,560 zero-valued PCM samples (660 ms) AFTER every source WAV.
Checking original event timestamps against authentic source clip sample counts
revealed four of nine events only after the artificial samples, four
strictly before clip end and one at clip end. This does not establish
end-of-word latency, because lexical boundaries were not verified.

This experiment runs both original published FP32/chunk16 and INT8/chunk16
weights with genuine 63 source WAV PCM and **no artificially added silence**.
It counts events in two disjoint phases:

- source_pcm: emitted while the original WAV samples are supplied
- eof_no_padding: emitted by input_finished() after the original PCM;
  not an event on an uninterrupted always-on audio stream

Original baseline events are read directly from exact existing GitHub
Actions artifact 14090411896, original ZIP SHA256
5f76390b557ab58c6d6ae131b8dad91ec232f3efb708466dcb4fc850b0eb13b5.
Both baseline JSON member SHA256 values are frozen in source code, and all
63 WAV/source roles must agree. The prior padded inference is not repeated.

## Reproduction

With sherpa-onnx==1.13.8 and numpy==1.26.4 and the same unchanged source ZIP,
three-evidence review JSON and the official Zh-En 3M model archive:

    python3 research/english63_weak_pilot/pretrained_live_timing.py \
      --tts-zip /restricted/original-source.zip \
      --review-json /restricted/original-review.json \
      --model-archive /restricted/official-model.tar.bz2 \
      --baseline-archive /restricted/original-quant-receipts.zip \
      --output /restricted/live-timing-result.json

Outputs are create-only and source-hash-bound. Hostile tests use no audio,
sherpa, NumPy, or Torch. A single data-owning public PR job may execute
actual inference with these original bytes, publish a machine-only receipt,
and be closed without merging an unnecessary workflow.

## Hard limits

Neither clip-end/EOF events nor baseline artificial-tail events prove
continuous household FAR/hour, genuine independent English FRR, true
end-of-word-to-event latency, SSC305 CPU/RAM or real 3–5m far-field quality.
Original prompt labels are synthetic weak intents, NOT acoustic truth.
No candidate model promotion, threshold tuning, training admission, new TTS,
ASR, source regeneration or scheduled patrol can result from this study.
