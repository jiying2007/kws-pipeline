# Wake-word customization

The product uses three levels so teams do not retrain when a cheaper intervention is sufficient.

## L0: keyword-only update

Compile text/pinyin to acoustic token IDs and tune the per-keyword threshold. No model weights change.

For firmware-linked products, generate a C header. For field-updatable products, generate a binary `.kwk` pack:

```bash
python3 tools/compile_keywords.py \
  --tokens tokens.zh.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-pack build/xiaowo.kwk \
  --out-json build/keywords.json
```

The runtime validates `.kwk` magic/version/size/vocabulary/token bounds before accepting it. The keyword pack must be built from the same token vocabulary as the `.kwm` model; a mismatched vocabulary is rejected.

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

## L1: calibration + hard negatives

Keep the model fixed and tune threshold, token boost, state retention and speech-energy gate using held-out positives plus long continuous negatives. Include near-homophones, partial phrases, TV/speaker playback, AEC residuals and the product's motor/fan/gear noise.

Release evaluation must report FRR and false accepts/hour; clip accuracy alone is not a KWS release metric. `eval/score_events.py` emits false positives that can be converted directly into empty-target CTC clips with `eval/mine_hard_negatives.py`.

## L2: shallow acoustic customization

Warm-start the base model and freeze the input/recurrent backbone. Multiple `--manifest` options can mix normal training data with mined hard negatives:

```bash
python3 training/train_ctc.py \
  --manifest data/custom/train.tsv \
  --manifest build/hard-negatives.tsv \
  --vocab-size 420 \
  --warm-start models/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo-head.pt
python3 training/export_model.py \
  --checkpoint build/xiaowo-head.pt \
  --output build/xiaowo-head.kwm
```

Empty targets are intentional negative CTC examples. `--head-only` requires `--warm-start` and is this repository's default meaning of "shallow customization". Full-model fine-tuning is possible by omitting `--head-only`, but it requires a much wider regression corpus because unrelated keywords can regress.

Do not share the same speaker/session/noise recording across train and evaluation splits. Do not mine hard negatives from the final held-out qualification corpus and then reuse that corpus as unbiased release evidence.
