# Independent English read-speech negative-event research

A **separate** source-disjoint probe after the original English Qwen63
weak-label and pretrained quantization experiments. This does NOT train,
fine-tune, select or qualify a wake model.

## Licensed source and immutable selection

Source: LibriSpeech SLR12 dev-clean from OpenSLR, by V. Panayotov,
G. Chen, D. Povey, and S. Khudanpur, CC BY 4.0 (credit retained).
The archived 337,926,286-byte official dev-clean.tar.gz is fixed by:
- SHA256: 76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3
- OpenSLR official MD5: 42e2234ba48799c1f50f24a7926300a1
The full archive must match the source digest **before any decode**.

No synthetic audio, user voice recordings or other dataset is reused. Original
English63 Hey Mero/Qwen recordings remain separate and unchanged.

**Pre-registered subset rule**, before any inference: include original FLAC
member whose SHA256(member_name), first 8 bytes as unsigned big-endian integer,
modulo 5 equals zero. This is approximately 20% of dev-clean. All selected
records must have original transcript text not containing an exact normalized
"HEY MERO" or "HEY VOSEN" phrase. This is a transcript filter, not a claim of
human KWS ground truth. Explicit source membership, file hashes, decoded
mono16k geometry, per-speaker counts and exact audio seconds are recorded.

## Frozen model comparator

Reuse official single-thread Zh-En Zipformer 3M model archive SHA256
68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
and the original official Hey Mero phonemes
HH EY1 M EH1 R OW0 @HEY_MERO.

Only the already measured FP32 chunk16 and INT8 chunk16 modes execute.
No threshold sweep. Both use score 1, threshold .25, active paths 4, 80-bin
frontend, 1600-sample partitions, 10560-sample clip-tail padding. Each
utterance has a new stream, so the test is **segmented read speech**, not a
connected multiple-hour background stream. The tail padding is excluded from
speech-hour denominator. Host elapsed times are diagnostic only.

Report the absolute number of detected Hey Mero events in independent read
English speech, with exposed source audio seconds and event counts per mode.
Do NOT call events per read-speech hour "FAR/hour", calculate human FRR, or
assert SSC305 latency/RSS. It measures only one kind of unrelated natural
speech; real household music, TV, pets, far-field, reverberation, accents and
long continuous running remain separate admission gates.

## Reproducibility

From independently verified original archives, sherpa-onnx==1.13.8,
numpy==1.26.4, soundfile==0.13.1 and CPU1:

    python3 research/english63_weak_pilot/librispeech_negative_events.py \
      --source-archive /restricted/original-dev-clean.tar.gz \
      --model-archive /restricted/original-zh-en3m.tar.bz2 \
      --output /restricted/read-speech-results.json

All outputs are create-only. A bounded one-shot public Actions PR may do
this on the pre-registered exact source; only machine JSON may be retained.
Raw FLAC and source model weights must not be committed or redistributed.
No schedule, private repo, Work, Codex, paid inference or new TTS/ASR.

## Model qualification barrier

Even zero detected events on this subset means neither zero field FAR nor
independent accuracy. The model is still blocked on independent genuine
English positive voice/accent and nearphone review, long continuous background
audio evaluation and real SSC305 BSP/board CPU/RAM/latency measurements.


## Actual fixed-source execution

Public data-owning Actions [run 43538998200](https://github.com/jiying2007/kws-data/actions/runs/43538998200)
finished successfully. Artifact 14090383839 was independently downloaded:
original ZIP SHA256 7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4;
the complete 517-row machine JSON SHA256
788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510,
259,931 bytes. Full source/audio/model digests and result identities are
retained in LIBRISPEECH_NEGATIVE_RESULT.json.

The predeclared 20% filename hash sample selected **517** actual
LibriSpeech dev-clean FLAC utterances, **40** distinct natural audiobook
speakers, and **3,665.38525 seconds = 61.089754 minutes** of original audio.
Both actual pretrained FP32/chunk16 and INT8/chunk16 decoders produced
**zero Hey Mero event** in these recordings with unchanged score and
threshold. Inference-loop timing (excluding FLAC decode but not necessarily
all Python overhead): 74.90342 sec FP32 vs 70.20134 sec INT8; one hosted
run only, not a controlled resource benchmark or SSC305 timing.

Unlike source-verified 63 synthetic Qwen prompts, these are independent
original read-speech recordings; however, the pre-trained vendor's training
corpus disjointness is not established and the original audiobook text is not
human KWS acoustic gold. Zero detections **does not mean FAR/hour=0**.
The 660ms EOF padding per clip and stream reset differs from continuous
household audio, so the only supported conclusion is zero observed events
on 61.09 min of this particular segmented read-speech subset.
