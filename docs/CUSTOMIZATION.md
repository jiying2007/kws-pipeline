# Wake-word customization

The product uses three levels so teams do not retrain when a cheaper intervention is sufficient.

## L0: keyword-only update

Compile text/pinyin to acoustic token IDs and tune the per-keyword threshold. No model weights change.

```bash
python3 tools/compile_keywords.py \
  --tokens tokens.zh.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-json build/keywords.json
```

Production keyword manifests should carry explicit pinyin in the fourth TSV column so a dependency update cannot silently alter tokenization.

## L1: calibration + hard negatives

Keep the model fixed and tune threshold, token boost, state retention and speech-energy gate using held-out positives plus long continuous negatives. Include near-homophones, partial phrases, TV/speaker playback, AEC residuals and the product's motor/fan/gear noise.

Release evaluation must report FRR and false accepts/hour; clip accuracy alone is not a KWS release metric.

## L2: shallow acoustic customization

Warm-start the base model and freeze the input/recurrent backbone:

```bash
python3 training/train_ctc.py \
  --manifest data/custom/train.tsv \
  --vocab-size 420 \
  --warm-start models/base.pt \
  --head-only \
  --epochs 10 \
  --output build/xiaowo-head.pt
python3 training/export_model.py \
  --checkpoint build/xiaowo-head.pt \
  --output build/xiaowo-head.kwm
```

This is the repository's default meaning of "shallow customization". Full-model fine-tuning is possible by omitting `--head-only`, but it requires a much wider regression corpus because unrelated keywords can regress.

Do not share the same speaker/session/noise recording across train and evaluation splits.
