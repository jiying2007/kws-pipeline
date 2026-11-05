# Wake-word customization

> **Current authority: technical research only.** The frozen synthetic model and keyword source is `configs/research/two-keyword-synthetic-comparator.json`; it is not real-human FAR/FRR, a physical SSC305 measurement or commercial publication authority. Old product/Release terminology elsewhere on this page is historical context, not a workflow to execute.

The runtime supports three customization levels. Technical capability must be distinguished from what the frozen synthetic research comparator actually demonstrates; independent real-speech or physical-board results are not implied.

## Frozen synthetic research comparator

The frozen historical synthetic comparator is `model-749187ec1d66`. It uses the dedicated five-token vocabulary:

```text
<blk> 0
ni3 1
hao3 2
xiao3 3
wo1 4
```

and exactly two full research wake phrases:

```text
id  text      threshold  explicit-pinyin
1   你好小窝  0.55       ni3 hao3 xiao3 wo1
2   小窝小窝  0.55       xiao3 wo1 xiao3 wo1
```

The two-character `小窝` is a partial/negative phrase, not one of the two full research wake phrases. `configs/research/two-keyword-synthetic-comparator.json` is the frozen synthetic comparator.

The frozen comparator's acoustic model uses `KWSP` ABI v2 and its keyword pack uses `KWKP` ABI v3. Both carry the same vocabulary size/fingerprint. Model ABI v2 also binds the frontend kind (`logmel` or `pcen-lite`). Warm starts, research exports and source-identity checks reject vocabulary/frontend mismatches.

## L0: keyword-pack-only update

L0 means the acoustic model weights remain fixed. It is valid only when **every acoustic token required by the requested phrase already exists in the loaded model vocabulary** and the resulting research model/keyword source identity is revalidated.

The keyword TSV accepts:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

Research keyword definitions should provide explicit pinyin in column 4. Columns 5–8 are optional bounded runtime-policy metadata.

For the frozen five-token comparator, only the two rows above have historical synthetic evaluation evidence, not independent real-speech qualification. A phrase requiring any other Mandarin token is **not** a valid L0 extension of this reference: it requires an intentionally broader/new acoustic model and new independent research evaluation.

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

`eval/mine_hard_negatives.py` requires the original complete `--references` and produces empty-target review candidates only when the exported crop overlaps no annotated keyword; predictions never supply truth. Normal training admission still requires reviewed labels, source rights and lineage. `eval/mine_false_rejects.py` replays missed positives with the configured token target. Neither may consume the final untouched qualification set if that set will remain unbiased release evidence.

The accepted model's qualification seed `271838` is already exposed and frozen. It may not be used for L1 calibration/replay or any future re-qualification. Use the active seed and retired-seed list in `configs/training/xiaowo.torch-domain.json` for a genuinely new formal candidate; routine maintenance does not consume one.

## L2: acoustic customization / broader vocabulary

If a research candidate requires wake phrases outside the frozen model vocabulary, first define the intended vocabulary/model identity, then audit split leakage and train an explicit new candidate. Example:

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

### Export output safety

The exporter requires distinct checkpoint, token input, model output and
provenance output destinations wherever an output is involved. It rejects
relative-path, symbolic-link and existing hard-link aliases before loading the
checkpoint. The default provenance path is `OUTPUT.provenance.json` and is
checked by the same rule.

The full model and serialized provenance are prepared before writing output.
Both payloads are staged in their destination directories before replacement;
each file replacement is atomic. This is not a transaction across the pair.
Any staging/replacement error fails the command, and a failure during the
second replacement may leave the first file updated. Treat that result as a
failed export and verify the sidecar's model hash before using either artifact.
Do not concurrently mutate export paths or their inputs.


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

Nightly is not another training/qualification path. It loads the exact hash-pinned model and keyword pack from `models/registry/model-749187ec1d66/` and the `configs/research-baseline.xiaowo.json` comparator, with no GitHub Release downloads, and evaluates the fixed negative regression corpus. It may not:

- call `train_ctc.py` or `iterate_domain.py`;
- create or rotate a formal qualification cohort;
- read `qualification_holdout_seed` from the formal training config;
- consume seed `271838` or reserve/expose the active formal seed for routine maintenance;
- promote nightly regression results as fresh qualification evidence.

`configs/nightly.xiaowo-frozen-model.json` and `.github/workflows/far-nightly.yml` encode this separation.

## Final AFE customization

For final-AFE research comparisons, the renderer's `command` backend binds the actual executable, ordered config bundle, command template, I/O hashes, result sidecar and latency. Any change to BF/AEC/RES/NS/AGC binaries or configuration creates a different research evidence identity and requires a new independent measurement, even when `.kwm` and `.kwk` remain unchanged.

The frozen synthetic comparator used proxy AFE evidence. Real final-AFE and SSC305 measurements remain separately unmeasured research inputs. The retained historical `shipping_approved=false` flag does not activate a publication workflow.

## Research source and fresh-measurement boundary

Any change to `.kwm`, `.kwk`, token vocabulary, runtime behavior/config, or final-AFE binary/config creates a new research comparison identity. Model provenance, runtime source identity, keyword-pack hash, AFE identity and measured evidence must refer to the same exact candidate. `docs/RELEASE_QUALIFICATION.md` is an archival product-era source record, not an active approval procedure.
