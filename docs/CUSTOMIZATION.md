# Wake-word customization

The runtime supports three customization levels, but product claims must distinguish **framework capability** from the vocabulary/model that has actually passed release qualification.

## Current shipping SKU

The current synthetic-qualified product model is `model-749187ec1d66`. It uses the dedicated five-token vocabulary:

```text
<blk> 0
ni3 1
hao3 2
xiao3 3
wo1 4
```

and exactly two shipping wake words:

```text
id  text      threshold  explicit-pinyin
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝小窝  0.55       xiao3 wo1 xiao3 wo1
```

The standalone two-character `小窝` is **not** a shipping wake word. `configs/shipping.xiaowo.json` is the machine-readable contract.

The deployed acoustic model is `KWSP` ABI v2 and the field-updatable keyword pack is `KWKP` ABI v3. Both carry the same vocabulary size/fingerprint. Model ABI v2 also binds the frontend kind (`logmel` or `pcen-lite`). Warm starts, export provenance and release qualification reject vocabulary/frontend mismatches.

## L0: keyword-pack-only update

L0 means the acoustic model weights remain fixed. It is valid only when **every acoustic token required by the requested phrase already exists in the loaded model vocabulary** and the resulting product tuple is revalidated.

The keyword TSV accepts:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

Production should provide explicit pinyin in column 4. Columns 5–8 are optional bounded runtime-policy metadata.

For the current five-token shipping model, only the exact two rows above have passed formal synthetic qualification. A phrase requiring any other Mandarin token is **not** an L0 update for this release: it requires an intentionally broader/new acoustic model or an L2-style customization followed by fresh qualification.

Compile a pack with:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.example.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-pack build/xiaowo.kwk \
  --out-json build/keywords.json
```

Policy semantics:

- `immediate`: emit a qualifying terminal immediately; simultaneous candidates use priority, depth and confidence.
- `longest`: delay a terminal until the configured trailing-blank condition, allowing a longer shared-prefix path to replace it.
- `grace`: hold a terminal for a bounded grace window plus trailing-blank condition.

`compile_keywords.py` defaults `longest` to one trailing blank when not supplied, and `grace` to three grace frames when not supplied. The runtime independently validates those requirements. Keyword updates are validated before the active trie is rebuilt; a rejected update leaves the previous valid configuration intact.

## Repeated-token semantics

Adjacent identical CTC target labels require a blank separator. The runtime keeps independent nonblank and blank-separated prefix scores, so a target such as `bao3 bao3` cannot complete from two consecutive `bao3`-dominant frames without a blank-separated state.

## L1: calibration and replay

Keep model weights fixed and tune keyword thresholds using development/calibration positives and continuous negatives. Include near-homophones, partial phrases, TV/speaker playback, AEC residuals, motor/fan/gear noise and the final AFE settings.

```text
.kwm + .kwk + continuous references
 -> kws_wav / run_corpus.py
 -> detections
 -> score_events.py + domain_metrics.py
 -> FAR/hour + FRR + latency + domain buckets
 -> false-positive / false-reject replay
```

`eval/mine_hard_negatives.py` produces empty-target clips from false accepts. `eval/mine_false_rejects.py` replays missed positives with the configured token target. Neither may consume the final untouched qualification set if that set will remain unbiased release evidence.

The accepted model's qualification seed `271838` is already exposed and frozen. It may not be used for L1 calibration/replay or any future re-qualification. `271839` remains reserved for a genuinely new future formal candidate, not routine maintenance.

## L2: acoustic customization / broader vocabulary

If the product requires wake phrases outside the current model vocabulary, first define the intended vocabulary/model identity, then audit split leakage and train an explicit new candidate. Example:

```bash
python3 training/audit_dataset.py \
  --split train=data/custom/train.tsv \
  --split calibration=data/custom/calibration.tsv \
  --split test=data/custom/test.tsv \
  --split qualification=data/custom/qualification.tsv \
  --report build/dataset-audit.json \
  --fail-within-split
```

Warm-start/head-only tuning is available when vocabulary, geometry and frontend remain compatible:

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

The warm start must match vocabulary, feature/hidden geometry, frontend identity and frontend-spec contract. A vocabulary expansion is a new acoustic-model lineage and must not be disguised as an L0 keyword-pack change.

## Domain-aware adaptation

For deterministic software/domain iteration:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.torch-domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The loop renders nominal near/mid/far scenes, uses the real C runtime for candidate evaluation, calibrates development candidates, scores worst domains, adapts curriculum weights, replays hard negatives and freezes the best strict-gate candidate before untouched qualification/robustness/FAR stages.

The synthetic 0.3–5 m distances are simulation parameters, not a claim that real human speech at those distances has passed.

## Nightly regression rule

Nightly is not another training/qualification path. It must download the exact immutable promoted model release and evaluate it against an independent regression corpus. It may not:

- call `train_ctc.py` or `iterate_domain.py`;
- create or rotate a formal qualification cohort;
- read `qualification_holdout_seed` from the formal training config;
- consume seed `271838` or reserve/expose `271839`;
- promote nightly regression results as fresh qualification evidence.

`configs/nightly.xiaowo-frozen-model.json` and `.github/workflows/far-nightly.yml` encode this separation.

## Final AFE customization

The shipping AFE is part of the release-qualification tuple. The renderer's `command` backend binds the actual executable, ordered config bundle, command template, I/O hashes, result sidecar and latency. Any change to the shipping BF/AEC/RES/NS/AGC binary/config creates a new product tuple and requires requalification even when `.kwm` and `.kwk` are unchanged.

The current synthetic-qualified model used proxy AFE evidence. The final real `audio-pipeline` identity will be bound in the deferred real-human/device qualification phase; until then the product contract remains `shipping_approved=false`.

## Release rule

Any change to `.kwm`, `.kwk`, token vocabulary, runtime behavior/config, or shipping AFE binary/config creates a new release-qualification tuple. Model provenance, runtime release identity, keyword-pack hash, AFE identity and final evidence must all refer to the same product deployment. See `docs/RELEASE_QUALIFICATION.md`.
