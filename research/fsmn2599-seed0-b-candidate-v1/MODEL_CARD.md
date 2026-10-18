# FSMN-2599 seed0 B: research candidate record

Status as of 2026-10-18: **research candidate; weights not included; not promoted**.
This directory publishes documentation and public-source metadata only. It does
not publish a checkpoint, create a model release, or qualify a deployable model.

## Exact candidate

- Intended filename: `model.final.pt`; 3,037,273 bytes.
- SHA-256: `a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77`.
- Full PyTorch state dictionary, with 30 recorded tensor entries. It is not a
  TorchScript/ONNX model, an optimizer checkpoint, or a shipping `.kwm` file.
- Base: ModelScope `thuduj12/kws_wenwen_fsmn_ctc`, revision
  `cd0c18d0dda6742c371fbb72ec66185906a9885d`, original `avg_30.pt` SHA-256
  `d3d89b5dde08c2926d46a97eab6f7179a87b1946ccebd16c02e77042ea8389fe`.
- Only the weight and bias of the existing final 140-to-2599 affine head were
  fine-tuned. Its dimensions did not change. The other 28 encoder/CMVN tensor
  hashes remained unchanged in the saved execution audit.
- The original schedule seed0 B, final step 220 is retained. Later order-seed
  experiments were sensitivity checks; no highest-scoring seed was selected.

See [identity and recipe](model-identity.json), [loading contract](LOADING.md),
[public source locks](source-manifest.json), and [evidence status](evidence-status.json).

## Training and label limitations

The fixed training set contained 146 unique recordings: 12 historical Qwen
stock-preset synthetic recordings, 14 historically exposed Qwen weak-label
recordings, and 120 MSWC published spoken-word recordings. Qwen generation used
the fixed 0.6B CustomVoice revision identified in the source manifest. There
were 880 sample presentations in 220 batches of four; repetitions do not create
additional independent examples. Only the final head was optimized from saved
offline-batch encoder features, with Adam, learning rate 0.0001, clipping norm 5,
and competition-loss coefficient 0.1. A was the matched coefficient 0 comparison.

Historical target acceptance and dual-ASR weak consensus are not new blind
human-gold labels. Shared TTS voices and prior development exposure remain part
of the lineage. MSWC word labels were inherited, not newly human-adjudicated.
Cosy recordings, AISHELL voice-clone material, the newer 15-candidate set, and
the eight historical background sessions were not training inputs for this head.
No corpus audio, per-recording identities, feature cache, logits, optimizer state,
raw transcripts, or private download references are distributed here. This is a
recipe summary, not a complete independently reproducible training-data package.

## What the saved evidence supports

1. Fixed 864-clip development replay compared the old head-only model, an earlier
   data-extension model, A, and B with the same decoder at diagnostic threshold 0.
   Their candidate-bearing clip counts were 39/28/40/41. B versus A added only
   the historical Eric stock-preset KW2 sample, score 0.1159931423, with no lost
   candidates. Both full-vocabulary greedy outputs remained `小屋小屋` rather
   than the intended `小窝小窝`. This is a limited low-score diagnostic gain.
2. Three fixed training orders reproduced that same within-order B-versus-A
   candidate-set gain. B scores were 0.1159931423/0.1130185379/0.1143355206;
   all six A/B greedy outputs remained `小屋小屋`. The seed1 pair also shared
   an additional historically unknown-label candidate. It remains unknown,
   neither an admitted positive nor a known false accept. Scores elsewhere
   changed; equal candidate sets do not mean equal behavior at other thresholds.
3. Eight historical background sessions provided 4.205059722 unique source
   hours, counted once across repeated heads. A and B each emitted zero raw
   candidates and zero events. Cached old-head posteriors exactly reproduced
   saved full-graph hashes; that cached-old stage did not run the decoder.
   The original old full-graph zero-candidate observation is separate inherited
   evidence. New independent validation hours remain 0 and admitted negative
   seconds remain 0. These sessions do not distinguish A from B or establish FAR.
4. Saved-result independent reviews covered identity, execution counts, paired
   comparisons and preservation. Review of saved JSON/hashes is not a new
   independent acoustic test or a claim that all numerical tensors were replayed.

These numbers are diagnostics on reused/weak development material. They are not
FRR, FAR, independently held-out recall, a solved near-homophone distinction,
or evidence that this is the final best model. Do not pool the three training
orders into a larger independent sample size.

## Deployment and publication gates still open

- The strict 76-frame C numerical-parity gate on the old-head reference failed
  for the original four-lane implementation and the fixed
  eight-lane/fused-multiply-add alternatives. This is an unresolved shared-runtime
  prerequisite, not a completed numerical test of the B checkpoint.
  No tolerance was relaxed to turn these results into passes. C/SSC305
  deployment acceptance, target CPU/memory/power/soak, and end-to-end runtime
  equivalence have not been established.
- No fresh, isolated human/final-microphone/final-AFE qualification, accepted
  continuous-negative exposure, calibrated operating point, or shipping approval
  is supplied by this record.
- The existing [model storage policy](../../models/README.md) admits only a
  governed promoted tuple to `models/registry/`. The candidate has no successful
  model-promotion receipt, qualifying export tuple, or immutable model release.
  Its checkpoint remains outside the public Git repository. This record creates
  no research-weight exception and leaves the existing registry and shipping
  contract unchanged.
- [Release qualification](../../docs/RELEASE_QUALIFICATION.md) describes the
  artifact-bound promotion/shipping evidence; this documentation does not
  substitute for any required gate or attest that a 2599-class model matches the
  current five-token shipping ABI/keyword contract.

## Rights and intended use

The current work is noncommercial research. The repository already uses
[Apache-2.0](../../LICENSE); this record does not create a new noncommercial
license or restrict rights already granted by upstream licenses. Public-source
checks support the base model's explicit Apache-2.0 declaration and MSWC's
CC-BY-4.0 dataset provenance, subject to their conditions. Code, model, source
data, synthetic-output scope and scientific qualification are separate matters.
See [NOTICE](NOTICE.md) for attribution and bounded rights conclusions.

No real-person consent, speaker endorsement, universal commercial-output rights,
privacy/memorization audit, or absence of third-party claims is asserted. A
research label is neither a rights waiver nor a substitute for product testing.
