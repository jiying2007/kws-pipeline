# Wake-word customization

The product uses three levels so teams do not retrain when a cheaper intervention is sufficient.

## ABI-v2 vocabulary identity

The acoustic model, training checkpoints, field-updatable keyword packs and generated C keyword tables are all bound to one token vocabulary. Training stores the exact vocabulary fingerprint; export refuses to attach a checkpoint to another same-sized token-to-ID mapping. The deployed `.kwm`, `.kwk` and generated C tables retain the same 64-bit fingerprint.

The fingerprint is computed from canonical token-ID order, so line reordering in the vocabulary file is harmless, while changing a token string or token ID changes the identity. A `.kwm` and `.kwk` with the same `vocab_size` but different mappings are rejected.

## Repeated-token semantics

The acoustic model is trained with CTC. Adjacent identical target labels therefore require a blank separator. The runtime trie enforces the same structural rule: each prefix retains separate nonblank and blank-separated Viterbi scores, and an identical child token can advance only from the blank-separated score.

For example, a future keyword tokenized as `bao3 bao3` cannot be completed from two consecutive `bao3`-dominant acoustic frames; a blank-separated prefix is required. Non-repeated paths such as `ni3 hao3 xiao3 wo1` retain the normal low-latency path.

This decoder is deliberately a small keyword-path Viterbi scorer, not a general CTC prefix-beam decoder.

## L0: keyword-only update

Compile text/pinyin to acoustic token IDs and tune the per-keyword threshold. No model weights change.

For firmware-linked products, generate a C header. For field-updatable products, generate a binary `.kwk` pack:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.zh.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-pack build/xiaowo.kwk \
  --out-json build/keywords.json
```

The runtime validates `.kwk` magic, ABI version, canonical size, vocabulary size/fingerprint, reserved fields, duplicate IDs, duplicate acoustic token paths, thresholds and token bounds before accepting it.

Production keyword manifests should carry explicit pinyin in the fourth TSV column so a dependency update cannot silently alter tokenization.

Typical runtime flow:

```c
kws_model_t model;
kws_keyword_pack_t pack;
kws_engine_t *engine;

kws_model_open(model_blob, model_bytes, &model);
kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack);
kws_engine_init(arena, arena_bytes, &model, NULL, &engine);
kws_engine_set_keyword_pack(engine, &pack);
```

`kws_engine_set_keyword_pack()` copies the token paths into the decoder trie, so the parsed pack object itself does not have to remain live after the setter returns. The underlying `.kwm` model blob must remain valid for the lifetime of the engine because model tensors are zero-copy views into that blob.

Keyword updates are validated before the active trie is rebuilt. A rejected update leaves the previous valid configuration intact.

## L1: calibration + hard negatives

Keep the model fixed and tune threshold, token boost, state retention and speech-energy gate using held-out positives plus long continuous negatives. Include near-homophones, partial phrases, TV/speaker playback, AEC residuals and the product's motor/fan/gear noise.

Release evaluation must report FRR and false accepts/hour; clip accuracy alone is not a KWS release metric. The official path is:

```text
.kwm + .kwk + reference corpus
 -> kws_wav / eval/run_corpus.py
 -> detections.jsonl
 -> eval/score_events.py
 -> FAR/hour + FRR + latency + false-positives.jsonl
```

`eval/mine_hard_negatives.py` converts false positives into empty-target CTC clips for retraining.

## L2: shallow acoustic customization

Before any base or shallow training, audit train/tuning/final qualification audio using decoded PCM hashes:

```bash
python3 training/audit_dataset.py \
  --split train=data/custom/train.tsv \
  --split qualification=data/eval/references.jsonl \
  --audio-root qualification=data/eval \
  --report build/dataset-audit.json
```

Warm-start the base model and freeze the input/recurrent backbone. Multiple `--manifest` options can mix normal training data with mined hard negatives, but all manifests and the warm start must use the exact same `--tokens` vocabulary:

```bash
python3 training/train_ctc.py \
  --manifest data/custom/train.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start models/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo-head.pt

python3 training/export_model.py \
  --checkpoint build/xiaowo-head.pt \
  --tokens keywords/tokens.zh.txt \
  --output build/xiaowo-head.kwm
```

Empty targets are intentional negative CTC examples. `--head-only` requires `--warm-start` and is this repository's default meaning of "shallow customization". Full-model fine-tuning is possible by omitting `--head-only`, but it requires a much wider regression corpus because unrelated keywords can regress.

Training checkpoints record the vocabulary fingerprint, token-file hash, manifest hashes, frontend spec version, seed and optimizer settings. Warm-start validation rejects incompatible metadata before loading weights.

Do not share the same decoded audio across train/calibration/evaluation splits. Rewrapping or renaming a WAV does not make it independent. Do not mine hard negatives from the final held-out qualification corpus and then reuse that corpus as unbiased release evidence.
