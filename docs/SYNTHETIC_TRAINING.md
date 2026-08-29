# Synthetic self-training loop

This workflow exists to close the **software, data and model-control loop before real human speech is available**. Its strongest successful result is named **`synthetic-qualified`**.

`synthetic-qualified` means that the repository can deterministically generate independent data splits, fit/export a real ABI-v2 model, compile a real keyword pack, run the actual C runtime over continuous audio, calibrate thresholds, mine false accepts/rejects, iterate candidates, select a best candidate and pass a previously unseen synthetic-heldout gate.

It does **not** mean that Mandarin human speech, real speakers, real rooms or the shipping Cortex-A32 SKU have been acoustically qualified. Repository issue #2 remains the separate real-evidence gate.

## One-command loop

Build the hosted runtime first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKWS_STRICT=ON
cmake --build build --parallel
```

Then run:

```bash
python3 training/iterate.py \
  --config configs/training/xiaowo.synthetic.json \
  --runner build/kws_wav
```

The loop creates its workspace under `build/synthetic-loop` by default and replaces that workspace on each run.

## Four isolated data pools

The generator creates four distinct pools from different deterministic seeds:

- `train`: model fitting / CTC training only;
- `calibration`: keyword threshold search and FP/FN mining;
- `test`: candidate regression comparison after calibration;
- `qualification`: evaluated only after the best candidate has been selected.

Every split is passed through `training/audit_dataset.py`; decoded PCM SHA256 must not overlap across splits. Synthetic continuous recordings use a configurable clip gap. The shipped example pins `continuous_gap_ms=1600`, which is longer than the default 1200-ms runtime refractory period and prevents one synthetic clip from suppressing or completing another clip's decoder state.

## Positive and negative generation

Positive families are generated from the explicit pinyin path in the four-column keyword TSV. The dependency-free `tone` backend gives every active acoustic token a deterministic spectral carrier. Gain, pitch, SNR, simple echo and generated `white` / `fan` / `motor` / `media` noise vary by seed.

Negative families include:

- partial wake paths;
- token substitutions;
- adjacent-token swaps;
- random token strings;
- synthetic noise profiles.

A negative is rejected during generation if it contains any complete configured wake path as an ordered subsequence. This prevents a mislabeled negative from teaching or testing an impossible contract.

## TTS adapter

`training/synthetic_audio.py` also supports a generic command backend. The command is an argv list and may use `{text}` and `{output}` placeholders. The command must produce uncompressed mono PCM16 WAV at 16 kHz.

Example shape:

```json
{
  "generator": {
    "tts": {
      "backend": "command",
      "command": [
        "/path/to/offline-tts-wrapper",
        "--text", "{text}",
        "--output", "{output}"
      ]
    }
  }
}
```

Keep the exact TTS engine/model/voice identity in the experiment configuration or wrapper provenance when a real synthetic speech backend is connected. Do not silently replace one voice/model with another while comparing candidates.

## Dependency-free trained prototype

Hosted CI must not depend on a large training framework, so the `prototype` backend is a real deterministic learning path rather than a hand-written constant model:

1. generate isolated augmented token-fitting samples from training-only seeds;
2. extract features with the same dependency-free frontend specification used by the C parity gate;
3. split energetic token frames from lead/tail background frames;
4. add independent white/fan/motor/media background frames for the explicit blank class;
5. project the 32 frontend features through an ABI-v2 int8 identity input matrix and `tanh`;
6. train a class-balanced multiclass softmax acoustic head over blank + active tokens;
7. quantize the learned output head to int8 using the actual ABI-v2 global output scale;
8. rerun held-out token-fit validation **after quantization**;
9. reject the model if quantized validation accuracy falls below 99.5%;
10. export canonical `.kwm`, fitting sample hashes, optimizer diagnostics and a confusion matrix.

Calibration, test and qualification audio are never read by this optimizer. The emitted provenance uses `evidence_class: synthetic-trained-softmax-prototype` and binds the model hash to its fitting samples and diagnostics.

The prototype exists to prove a true train → quantize → ABI-v2 model → C runtime → calibration → replay → held-out qualification loop without PyTorch. It is deliberately synthetic and is not a substitute for a generic Mandarin acoustic model.

## Decoder admission contract

The runtime keyword Trie does not advance from every token with a small posterior. Each acoustic frame admits at most one nonblank label: the highest-logit nonblank token, and only when it also beats blank. Non-dominant token probabilities remain in normalization/confidence but cannot create a structural Trie transition.

This is important for synthetic confusable negatives such as reordered or missing-token sequences. Without the admission gate, low-probability transition-frame tails combined with `token_boost` can fabricate a keyword path that was never the greedy acoustic sequence. Prefix retention still permits gaps/unrelated dominant labels, and repeated identical keyword tokens still require a blank-separated state.

The decoder remains a bounded product-oriented Viterbi/greedy CTC hybrid, not a full CTC prefix beam search.

## Learning backend

For actual CTC weight learning, set:

```json
{
  "iteration": {
    "backend": "torch_ctc"
  }
}
```

That backend uses the repository's `training/train_ctc.py` and `training/export_model.py`. Round 0 trains from the synthetic train manifest. Later rounds warm-start the previous checkpoint with `--head-only` and additionally consume accumulated replay manifests.

## Automatic threshold calibration

Each candidate is evaluated on the calibration continuous recording through the real `kws_wav` executable. `iterate.py` performs per-keyword coordinate search across `calibration.thresholds`, recompiles `.kwk`, and scores every trial with `eval/score_events.py`.

Candidate selection uses calibration metrics. Test is a regression gate. Qualification remains untouched until after the best candidate is frozen.

The example synthetic policy keeps strict gates:

```text
FRR <= 5%
FAR/hour == 0
p95 post-end latency <= 800 ms
```

Do not relax these limits merely to make CI green. Fix data labeling, isolation, acoustic classification or decoder behavior instead.

## Bidirectional failure replay

Calibration failures feed the next learning round in both directions:

```text
false accept
  -> false-positives.jsonl
  -> mine_hard_negatives.py
  -> empty-target CTC replay

false reject
  -> false-rejects.jsonl
  -> mine_false_rejects.py
  -> positive replay with the configured token target
```

Replay clips are cumulative and de-duplicated by manifest line. Their final counts and hashes are written into `synthetic-loop-manifest.json`.

The dependency-free prototype always refits from its isolated token/background fitting domain; replay affects actual weight updates in the `torch_ctc` backend. Both backends still exercise FP/FN mining, candidate selection, provenance and stopping logic.

## Candidate selection and stopping

A candidate score heavily penalizes synthetic policy violations, then ranks FRR, FAR/hour and latency. A new candidate never overwrites the retained best candidate merely because it is newer.

The loop stops when either:

- the minimum round count is reached and the best candidate passes calibration + test gates while `stop_on_gate` is enabled; or
- no score improvement is observed for `patience` rounds; or
- `max_rounds` is reached.

Only then is the retained best `.kwm/.kwk` evaluated on synthetic qualification.

## Output contract

Important retained outputs include:

```text
build/synthetic-loop/
  dataset/
    train.tsv
    calibration.tsv
    test.tsv
    qualification.tsv
    calibration.continuous.wav
    test.continuous.wav
    qualification.continuous.wav
    *.references.jsonl
    dataset-index.jsonl
    dataset-summary.json
    token-carriers.json
  dataset-audit.json
  replay/
    hard-negatives.tsv
    missed-positives.tsv
  candidates/
    round-*/
      model.kwm
      model.kwm.synthetic-provenance.json
      prototype-fit/
        token-fit-samples.jsonl
        softmax-diagnostics.json
  best/
    model.kwm
    keywords.kwk
    keywords.tsv
    synthetic-qualification/
  synthetic-loop-manifest.json
```

The final manifest uses `evidence_class: "synthetic-only"`, records selected artifact/data hashes, candidate metrics, replay hashes and explicit limitations. CI additionally checks the model fitting provenance and post-quantization confusion diagnostics. A successful run may be described as **synthetic-qualified** only.

## Promotion to real evidence

When human speech becomes available, do not delete or weaken this loop. Add real train/calibration/test pools and keep a new real held-out qualification pool isolated from replay. The production release path remains `docs/RELEASE_QUALIFICATION.md`, including target-board measurements and issue #2 closure criteria.
