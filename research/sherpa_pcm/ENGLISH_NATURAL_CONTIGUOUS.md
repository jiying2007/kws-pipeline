# Real English read-speech through uninterrupted native C KWS

Research-only, fixed-policy follow-up to the actual native C model evidence
merged in #804 and the prior real read-speech segment experiments merged in
#801. The same 517 LibriSpeech dev-clean original recordings (40 authentic
audiobook speakers; 3,665.38525 s) previously yielded no Mero events
when EVERY utterance was decoded with a fresh stream and 660 ms EOF padding.
That experiment cannot establish the behavior of an always-on KWS stream.

## Original licensed source, pre-existing evidence and provenance

- OpenSLR LibriSpeech SLR12 dev-clean, CC BY 4.0, author credit:
  V. Panayotov, G. Chen, D. Povey and S. Khudanpur. Original archive
  337,926,286 bytes; SHA256
  76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3.
  This experiment does not package or redistribute original audio.
- Exactly the previously frozen SHA256(member name) first-8-byte
  big-endian modulo-5-equals-zero selection of 517 FLAC recordings. No
  filter based on acoustic detections, new pronunciations, tuning or
  model results.
- Verify the original [kws-data run 43538998200](https://github.com/jiying2007/kws-data/actions/runs/43538998200)
  segmented source-evidence ZIP, ID 14090383839, SHA256
  7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4.
  Its original JSON SHA256 is
  788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510.
  Verify each selected FLAC SHA256, transcript SHA256, speaker ID and
  sample count against this independent prior evidence before any model call.
- Source transcripts must lack the exact normalized phrases HEY MERO and
  HEY VOSEN; transcript absence is **not** manually reviewed acoustic
  truth and every record remains an **untrusted negative candidate**.
- Original unchanged official Zh-En 3M model release SHA256
  68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6.
  Original INT8/chunk16 encoder+joiner and FP32 decoder.
- Actual host-native C ABI built only through the pinned authentic Sherpa
  v1.13.8 x64 release and audited inner C API/ORT member libraries. Model
  keyword alias HEY_MERO, C event ID3; sample rate 16000 Hz, CPU1, 80 bands,
  max paths 4, score 1.0, threshold 0.25, one trailing blank. No new model
  checkpoint, training or phoneme generation.

## Two predeclared uninterrupted source orders

Each order uses every original decoded 16 kHz PCM16 source *once*, without
any interclip silence and with **one persistent KWS stream throughout**.

- source_member: lexical original archive FLAC member ordering
- speaker_round_robin: ascending original speaker ID, each original speaker's
  FLAC members lexically ordered and interleaved one-by-one

Each order is replayed twice with a fixed 320-sample caller PCM partition
and a fixed 1600-sample partition. Total: four original-input sessions,
each 58,646,164 PCM samples = 3,665.38525 original speech seconds.
Reset the native C stream only at the beginning of each complete session;
finish without artificially appended samples. The two source orderings
ARE artificial audiobook concatenations; these are not natural connected
home speech. The source splice points, exact native event input-sample
positions and events within 200 ms of a splice are retained.

An actual nonzero count here, if observed, is a meaningful **diagnostic
under source-dependent context**, not independent label-certified FAR/h.
Even a zero count does not prove zero household FAR.

## Reproduce

Obtain independently verified original source (not altered), the prior
segmented receipt, and the original official pretrained model. Build the
optional native host C ABI with:

    python3 research/sherpa_pcm/build_official_x64.py \
      --runtime-archive /restricted/official-sherpa-x64.tar.bz2 \
      --output /restricted/native

Then run (numpy 1.26.4, soundfile 0.13.1):

    python3 research/sherpa_pcm/tests/english_librispeech_contiguous_c.py \
      --native-library /restricted/native/libkws_sherpa_pcm.so \
      --build-receipt /restricted/native/build-receipt.json \
      --corpus-archive /restricted/original-dev-clean.tar.gz \
      --model-archive /restricted/original-zh-en-model.tar.bz2 \
      --reference-zip /restricted/original-segmented-receipt.zip \
      --output /restricted/real-contiguous-events.json

Source-only three-mode hostile tests run without the model/audio/NumPy;
the public source-owning Actions job may run the actual research model once.
No audio, full model, private credential or archived runtime goes into Git.

## Scientific and engineering stop conditions

Do not classify events mapped to a non-keyword audiobook transcript as
verified false positives. An artificial splice can create interactions,
and original audiobook transcriptions were not manually authenticated for
the wake phonetics. No FAR/hour, human FRR, natural household environment,
3–5 m far-field, true end-of-word latency, SSC305 ABI/CPU/RSS/temperature,
model promotion or shipping decision is established.

The final target is SSC305 Cortex-A32; its real BSP/sysroot/toolchain is
unavailable in the research source record. The inspected generic ARM32
Sherpa binary requires GLIBC_2.34, GLIBCXX_3.4.29 and CXXABI_1.3.13, so
do not assume compatibility by CPU architecture alone. Use the existing
read-only SSC305 SDK probes rather than patching libc or installing a
generic ARM runtime on the product board.


## Actual original-speech native C run (2026-11-14)

Public source-owning [Actions run 43561538859](https://github.com/jiying2007/kws-data/actions/runs/43561538859)
**completed successfully** on original LibriSpeech dev-clean 517 FLAC recordings
from 40 audiobook speakers. Exact artifact ID 14100332962, ZIP SHA256
4ebdfb5dd9fbf7f882497b17d178dcc527bdfadb761bf05b014dea8fe08728d2,
original JSON SHA256
f4a44e3b91e148832becd9f486be7a4cc97d1693bc42cb0cd8f8f4fd8ed6dca5
(224,834 bytes), independently downloaded and hash checked.

**All four actual uninterrupted native-C sessions emitted zero Hey Mero events**:
lexical source order and 40-speaker round-robin, each under fixed320 and
network1600 caller PCM chunk plans. Each session processed exactly 517
original recordings, 58,646,164 mono PCM16 samples (3,665.38525 seconds),
with NO inserted silence, NO between-utterance model reset and NO
artificial EOF tail. Both input patterns yielded identical results for
each source order. Host native decode loops took 48.48618–49.15344 sec
per 3,665.38525 sec of input on the hosted x64 runner, single thread.
One-shot host process peak RSS 434,196 KiB includes all 517 decoded
FLAC arrays buffered in Python and runtime/setup overhead: it **cannot**
be used as model-only memory or SSC305 RSS.

These four sessions repeat and reorder the same original 61.09 minutes,
not four independent hours. No events were emitted on this fixed
*concatenated real audiobook* screen, which strengthens the earlier
segmented-speech observation but does not prove a field FAR=0.
Results and source/runner/binary digest are retained in
ENGLISH_NATURAL_CONTIGUOUS_RESULT.json. The authentic original 517-row
machine receipt remains available from the source Actions artifact while
GitHub retains it; no audio, pretrained model or runtime binary was copied
to the research repository.

Strict next qualification gates remain genuine wake-positive human
microphone audio and accent/nearphone review, multi-hour household
background TV/noise/reverberation, actual end-of-word trigger latency
and SSC305 BSP/sysroot CPU/RAM/dependency verification. Neither this
experiment nor synthetic concatenation can discharge those gates.
