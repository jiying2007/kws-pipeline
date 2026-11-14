# English63 original-source Zh-En 3M quantized KWS event comparison

Research-only: neither a newly trained model nor an SSC305 deployment claim.

## Purpose

The saved English63 GRU/DS-TCN checkpoints reproduce final clip logits from
streaming PCM but lack a qualified continuous wake-event head. The official
Sherpa-ONNX Zh-En 3M model already produced 9 real Hey Mero events on 14
Mero-requested synthetic clips in original run 43448541871 (kws-data #111).
Compare **official** pretrained FP32 vs INT8 (without custom re-quantization)
on the exact same original audio to screen event stability, CPU and storage.

## Exact frozen source and runtime

- Original 63-Qwen source ZIP SHA256:
  f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc
- Three-evidence source-review JSON SHA256:
  89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12
- Official original Zh-En 3M model archive SHA256:
  68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
- Official frozen phonemes for HEY MERO:
  HH EY1 M EH1 R OW0 @HEY_MERO
- Models: fp32_c16, int8_c16, int8_c8. INT8 uses provided INT8 encoder and
  joiner plus FP32 decoder. No alternate fine-tuning or threshold search.
- All modes: CPU one thread, 80-band frontend, 4 paths, threshold=0.25,
  keyword_score=1.0, trailing_blanks=1, PCM 1600-sample input partitions,
  exact 10560-sample EOF padding.
- Separate process per model to report host decode loop elapsed seconds
  excluding model load and process peak RSS including setup. Neither is
  a target SSC305 measurement.

## Reproduce from privately verified original archives

With sherpa-onnx==1.13.8 and numpy==1.26.4:

    python3 research/english63_weak_pilot/pretrained_quant_events.py \
      --mode fp32_c16 --tts-zip /restricted/tts.zip \
      --review-json /restricted/review.json \
      --model-archive /restricted/zh-en-original.tar.bz2 \
      --output /restricted/fp32_c16.json

Repeat for int8_c16 and int8_c8 using unique output paths, then aggregate:

    python3 research/english63_weak_pilot/pretrained_quant_events.py \
      --compare /restricted/fp32_c16.json /restricted/int8_c16.json \
                /restricted/int8_c8.json \
      --output /restricted/comparison.json

All outputs are create-only. No new TTS, ASR, IPA, source reclassification,
training, threshold selection or model promotion. Source-only hostile tests
run under the existing normal/-O/-OO CI inventory.

## Scientific boundary

Results on requested synthetic speech are not independent true FRR, FAR/hour,
human accent verification or nearphone negatives. Hey Vosen remains outside
the official phoneme lexicon and is not tested; do not invent pronunciation.
The event readout cannot establish genuine end-of-word latency, SSC305
A32 CPU/RAM or an English research model winner. The fixed-policy INT8
candidate must not be promoted solely on host runtime or smaller files.


## Executed original-source outcome (2026-11-14)

Source-owning GitHub Actions run:
https://github.com/jiying2007/kws-data/actions/runs/43537391922 (success).
Artifact ID 14090411896, ZIP SHA256
5f76390b557ab58c6d6ae131b8dad91ec232f3efb708466dcb4fc850b0eb13b5.
Original JSON SHA256 values, source/program commits and measured aggregates
are retained in PRETRAINED_QUANT_RESULT.json. This exact artifact was
independently downloaded and read back, not inferred from CI setup logs.

| Original pretrained mode | Events on Mero-requested 14 | Other-requested 49 | Model components | Host loop wall | Process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 / chunk16 | 9 | 0 | 13,074,255 B | 3.271188 s | 105,448 KiB |
| INT8 / chunk16 | 9 | 0 | 5,446,114 B | 2.871730 s | 100,708 KiB |
| INT8 / chunk8 | 8 | 0 | 5,447,115 B | 3.930516 s | 97,788 KiB |

INT8/chunk16 has zero per-clip event-count changes versus FP32/chunk16 on
this source. This is a **measured screening outcome**, not independent accuracy.
INT8/chunk8 missed one event on cell en-85fa95925c392bbc6abf8f20
(the Qwen synthetic Eric requested Hey, Mero!). It is rejected for this
next research step, not globally proven inferior.

INT8/chunk16 is the **next research candidate** for host and Cortex-A32
runtime validation, NOT a deployable/promotable winner. A one-off host wall
time reduction is not a controlled repeated-performance benchmark.
The process RSS includes model loading/setup and Python, and the official
chunk sizes describe architecture latency constraints, not actual
end-of-word trigger latency. Genuine continuous FAR/hour, independent
human FRR, multi-accent/far-field 3–5m and SSC305 resource qualification
remain NOT_RUN. No unverified nearphones were treated as certified negatives.
