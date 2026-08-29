# Wake-word customization

The product uses three levels so teams do not retrain when a cheaper intervention is sufficient.

## Artifact identity

The deployed acoustic model is `KWSP` ABI v2 and the field-updatable keyword pack is `KWKP` ABI v3. Both carry the same vocabulary size/fingerprint. Training checkpoints and generated C keyword tables bind to that vocabulary identity as well.

Model ABI v2 also binds the frontend kind (`logmel` or `pcen-lite`). Warm starts, export provenance and release qualification reject a frontend mismatch.

## L0: keyword-only update

The keyword TSV accepts:

```text
id  text  threshold  explicit-pinyin  min_trailing_blanks  priority  prefix_policy  grace_frames
```

Production should provide explicit pinyin in column 4. Columns 5–8 are optional bounded runtime policy metadata.

Example:

```text
1	小窝	0.55	xiao3 wo1	1	10	grace	3
2	小窝小窝	0.55	xiao3 wo1 xiao3 wo1	1	20	longest
```

Compile:

```bash
python3 tools/compile_keywords.py \
  --tokens keywords/tokens.zh.txt \
  --keywords keywords/zh_cn_example.tsv \
  --out-header build/keywords.generated.h \
  --out-pack build/xiaowo.kwk \
  --out-json build/keywords.json
```

Policy semantics:

- `immediate`: emit a qualifying terminal immediately; simultaneous immediate candidates use priority, depth and confidence.
- `longest`: delay a terminal until the configured trailing-blank condition, allowing a longer shared-prefix path to replace it.
- `grace`: hold a terminal for a bounded grace window plus trailing-blank condition.

`compile_keywords.py` defaults `longest` to one trailing blank when not supplied, and `grace` to three grace frames when not supplied. The runtime independently validates those requirements.

Keyword updates are validated before the active trie is rebuilt. A rejected update leaves the previous valid configuration intact.

## Repeated-token semantics

Adjacent identical CTC target labels require a blank separator. The runtime keeps independent nonblank and blank-separated prefix scores, so a target such as `bao3 bao3` cannot complete from two consecutive `bao3`-dominant frames without a blank-separated state.

## L1: calibration and replay

Keep model weights fixed and tune keyword thresholds using held-out positives and continuous negatives. Include near-homophones, partial phrases, TV/speaker playback, AEC residuals, motor/fan/gear noise and the final AFE settings.

The product path is:

```text
.kwm + .kwk + continuous references
 -> kws_wav / run_corpus.py
 -> detections
 -> score_events.py + domain_metrics.py
 -> FAR/hour + FRR + latency + domain buckets
 -> false-positive / false-reject replay
```

`eval/mine_hard_negatives.py` produces empty-target clips from false accepts. `eval/mine_false_rejects.py` replays missed positives with the configured token target. Neither may consume the final qualification-heldout set if that set will remain unbiased release evidence.

## L2: shallow acoustic customization

Audit split leakage first:

```bash
python3 training/audit_dataset.py \
  --split train=data/custom/train.tsv \
  --split calibration=data/custom/calibration.tsv \
  --split test=data/custom/test.tsv \
  --split qualification=data/custom/qualification.tsv \
  --report build/dataset-audit.json \
  --fail-within-split
```

Warm-start and freeze the input/recurrent backbone:

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

The warm start must match vocabulary, feature/hidden geometry, frontend identity and frontend-spec contract.

## Domain-aware adaptation

For deterministic software/domain iteration:

```bash
python3 training/iterate_domain.py \
  --config configs/training/xiaowo.domain.json \
  --runner build/kws_wav \
  --work-dir build/domain-loop
```

The loop can compare `logmel` and `pcen-lite`, render nominal near/mid/far acoustic scenes, calibrate each candidate with the real C runtime, score worst domains, reweight the next training round and freeze the best candidate before an untouched synthetic qualification pass.

The example's 0.3–5 m distances are simulation parameters, not a claim that real human speech at those distances has passed. Real recordings from the shipping microphones, enclosure, rooms, speaker playback and `audio-pipeline` configuration are still required.

## Release rule

Any change to `.kwm`, `.kwk`, token vocabulary, runtime config or shipping AFE configuration creates a new release qualification tuple. See `docs/RELEASE_QUALIFICATION.md`.
