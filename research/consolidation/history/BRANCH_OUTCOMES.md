# Dormant-branch retention audit

Audit baseline: `55a4e23379ad7072db507dbe419b38d8898e17fa` (2026-10-07).
All 62 remote heads were enumerated: 29 open-PR heads are handled separately, main is excluded, and every remaining 32 branch is covered here. Complete recursive trees were non-truncated. The companion JSON provides all 180 exact branch-delta path records, branch/main blob identities and path-level recommendations.

## Decision

- Do not merge dormant branches wholesale. Most are one-shot triggers, closed negative experiments, materially superseded implementations, or protected design references.
- Two provenance-tool branches are fully retained semantically by current main. One C file is exactly identical; the Python tool is extended by main.
- Preserve the detailed conclusions, immutable source pointers and 27 absent September 27–28 research notes. The roadmap already points to that old branch, but a historical archive improves discoverability without reviving the code.
- `research/a20-cosy30-cross-voice-v1` contains genuinely absent generation source. Its tip is armed. Disarm any integrated source, preserve the original armed SHA as provenance, and never equate workflow success with labels or product qualification.
- Restricted human-development validation and two keyword-set designs are genuinely absent designs, not verified ready-to-merge current functionality. Keep their exact manifests, and use a freshly scoped port with current security/data/qualification contracts if selected.
- Keep the existing repository-cleanup and canonical product-authority contract. No retired runtime path, model tuple or shipping state should change from this audit.

## Important scientific corrections

- #391/#392 are causal-invalid because 64 shared TTS files differed across hosted CPU families, not evidence that replay helped or failed.
- #385/#386, #428, #435 and #441 are infrastructure-failed, not completed algorithm verdicts.
- #407–411, #403, #415, #420 and #431 preserve negative results with explicit no-sweep/no-promotion boundaries. Better recall with more false events, or lower false events with collapsed recall, is not success.
- #450 supersedes #439’s older-history explanation: a phase-aligned 12.005-second cold crop reproduced seed1103. It does not isolate phase as the cause because 80 earlier samples also changed.
- #443’s exact-top best-eligible path is not proof of the exact emitted backpointer.
- The old adaptive-per-epoch label-prior implementation differs from canonical #426’s frozen initial-model prior. Restore neither old config names nor trainer code.

## Confidence and verification limits

High confidence in branch enumeration, file identities, explicit closure decisions and source supersets. Medium confidence in future utility of unmerged designs; current integration tests have not been run. Old audio/model/result artifacts were not downloaded or independently reproduced. Artifact URLs and comments are provenance, not substitute measurement. No external state or repository worktree was changed by this audit.

## Per-branch decisions

### codex/kws-expanded-train-readback-20260927

- Tip: `c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8`; 2 ahead / 72 behind baseline
- Classification: historical-notes-plus-stale-prototypes
- Finding: The expanded-train note and eval/run_corpus.py are byte-identical to main. Main roadmap section 1 already cites this exact tip and rejects whole-branch replacement. The remaining 27 notes preserve detailed negative experiments, source screening, evidence hashes and unfinished designs. Their local build/ evidence references were not recovered and historical user-approval text is not current permission.
- Action: Archive the 27 absent research notes as explicitly historical records; retain source pointers for the other stale additions. Do not merge the old trainer, event scoring, export or corpus-planning implementation wholesale.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8
  - https://github.com/jiying2007/kws-pipeline/pull/363

### diag/ctc-occurrence-target-blank-gradient

- Tip: `2468af38e1f2a797e4c1e15b7387fb524fe62e19`; 3 ahead / 28 behind baseline
- Classification: negative-pretraining-diagnostic
- Finding: #420 rejected training: combined shared-suffix update was positive on only 5/9 kw1 misses and 0/2 existing kw1 hits. Relative improvement was insufficient. This is a failed pre-training gate, not a failed training run.
- Action: Retain the result and exact diagnostic source pointer; keep the one-shot trigger/workflow out of main. Do not add the candidate to the trainer.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/2468af38e1f2a797e4c1e15b7387fb524fe62e19
  - https://github.com/jiying2007/kws-pipeline/pull/420
  - https://github.com/jiying2007/kws-pipeline/pull/420#issuecomment-5894233542
  - https://github.com/jiying2007/kws-pipeline/pull/420#issuecomment-5900742140

### diag/ctc-viterbi-separability-v1

- Tip: `0b45ab32c65c36a3613404dce8ec755d00fb0c50`; 3 ahead / 35 behind baseline
- Classification: negative-decoder-diagnostic
- Finding: #411 simple max-over-start token-synchronous CTC Viterbi: correct keyword won 9/32 retained positives; test kw1 0/8; ambiguity-stress pooled strict separation false. This rejects that formulation, not every possible CTC decoder.
- Action: Retain the bounded negative result and source pointer; do not implement this formulation in C or restore its trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/0b45ab32c65c36a3613404dce8ec755d00fb0c50
  - https://github.com/jiying2007/kws-pipeline/pull/411
  - https://github.com/jiying2007/kws-pipeline/pull/411#issuecomment-5886474808

### diag/frozen-far-history-decomposition-rerun-v1

- Tip: `22610f6f2012d5d7c3a86118c70b3ed187d4d951`; 2 ahead / 12 behind baseline
- Classification: trigger-only-superseded-interpretation
- Finding: #445 proves fresh decoder replay can reproduce the event from a short retained posterior slice, but PCM replay had an 80-sample frame-origin confound. #450 supersedes any long-acoustic-history requirement.
- Action: Archive the request identity and corrected result chain; do not restore its live trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/22610f6f2012d5d7c3a86118c70b3ed187d4d951
  - https://github.com/jiying2007/kws-pipeline/pull/445
  - https://github.com/jiying2007/kws-pipeline/pull/445#issuecomment-5907034980

### diag/frozen-far-source-replay-cases-v1-rerun1

- Tip: `d0828c3e2c35dffb8406cd1c504463951c3f8076`; 1 ahead / 17 behind baseline
- Classification: trigger-only-completed-diagnostic
- Finding: Standalone rendered sources produced 0 kw2 hits for both seed1103 and seed3301. This only says those isolated sources are insufficient; mixed-context diagnostics follow.
- Action: Retain #437 outcome and request SHA; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/d0828c3e2c35dffb8406cd1c504463951c3f8076
  - https://github.com/jiying2007/kws-pipeline/pull/437
  - https://github.com/jiying2007/kws-pipeline/pull/437#issuecomment-5905583093

### diag/frozen-far-source-replay-cases-v1

- Tip: `a9b29dccea1e4b03fd94a2f3745987bf88a363cb`; 1 ahead / 18 behind baseline
- Classification: trigger-only-infrastructure-failed
- Finding: #435 failed before replay because opaque domain family_id was parsed as an integer. No model/decoder/threshold verdict exists.
- Action: Retain failure classification and link the corrected #437 rerun; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/a9b29dccea1e4b03fd94a2f3745987bf88a363cb
  - https://github.com/jiying2007/kws-pipeline/pull/435
  - https://github.com/jiying2007/kws-pipeline/pull/435#issuecomment-5903769781

### diag/frozen-far-stream-context-replay-v1

- Tip: `73ff31a39090eff9d1e696c964c60cc31a218a2c`; 1 ahead / 16 behind baseline
- Classification: trigger-only-corrected-diagnostic
- Finding: #439 reproduced both historical stream FAs; 10-second pre-roll reproduced seed3301 but not seed1103. Its statement that seed1103 requires older history was later corrected by the phase-aligned 12.005-second cold replay in #450.
- Action: Retain #439 with explicit #445/#450 correction; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/73ff31a39090eff9d1e696c964c60cc31a218a2c
  - https://github.com/jiying2007/kws-pipeline/pull/439
  - https://github.com/jiying2007/kws-pipeline/pull/439#issuecomment-5905858220

### diag/fuzzy-competitor-token-provenance

- Tip: `924bc52c2bcc35fc69bd25e077f8dfed15e48811`; 6 ahead / 38 behind baseline
- Classification: functionality-retained-in-main
- Finding: Both changed files are retained semantically by main v4 temporal-neighborhood implementation: the branch v3 competitor evidence is extended with frame indices and temporal analysis. Full branch-to-main textual diff reviewed; no branch capability lost.
- Action: No code migration. Record canonical main paths and source provenance.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/924bc52c2bcc35fc69bd25e077f8dfed15e48811
  - https://github.com/jiying2007/kws-pipeline/pull/406

### diag/fuzzy-temporal-neighborhood-stack

- Tip: `a0e06819e679fdbcedce05499bb494dae0567f37`; 6 ahead / 37 behind baseline
- Classification: functionality-retained-in-main
- Finding: tools/kws_decoder_path_replay.c is byte-identical. Main Python implementation is the branch implementation plus whole-trace top1 recovery/distance/margin fields and tests. #406 outcome and later boundary-reset correction are retained in PR discussion.
- Action: No code migration. Record canonical main paths and source provenance.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/a0e06819e679fdbcedce05499bb494dae0567f37
  - https://github.com/jiying2007/kws-pipeline/pull/406

### diag/nonroot-exact-top-admission-probe

- Tip: `fb735bc609b7bc3be5418c36e9a91c5547d60cf2`; 8 ahead / 35 behind baseline
- Classification: negative-runtime-counterfactual
- Finding: #407 baseline equivalence held on all 520 traces; strict exact-top admission retained 11/32 positive hits and recovered 0/14 selected misses, while adding one new keyword pair. Closed negative, no shipping change.
- Action: Retain #407 outcome/source pointer. Do not restore runtime debug hooks or active trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/fb735bc609b7bc3be5418c36e9a91c5547d60cf2
  - https://github.com/jiying2007/kws-pipeline/pull/407
  - https://github.com/jiying2007/kws-pipeline/pull/407#issuecomment-5885599391

### diag/retained-frozen-far-path-summary-seed3301-rerun1

- Tip: `3a30f2878323b313d7361541faf451b78f643ecd`; 1 ahead / 14 behind baseline
- Classification: trigger-only-completed-diagnostic
- Finding: Seed3301 best-eligible kw2 path had one exact root start, three non-root exact advances and zero fuzzy advances. It does not prove that this snapshot is the exact emitted-candidate backpointer.
- Action: Retain #443 result and request identity; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/3a30f2878323b313d7361541faf451b78f643ecd
  - https://github.com/jiying2007/kws-pipeline/pull/443
  - https://github.com/jiying2007/kws-pipeline/pull/443#issuecomment-5906116534

### diag/retained-frozen-far-path-summary-seed3301

- Tip: `1e84b3c2b58fab504befa1c0a71b061f0347f7f5`; 1 ahead / 15 behind baseline
- Classification: trigger-only-infrastructure-failed
- Finding: Initial reader omitted root advances from accounting. Infrastructure failure, no decoder-path verdict.
- Action: Retain #441 failure and corrected #443 result; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/1e84b3c2b58fab504befa1c0a71b061f0347f7f5
  - https://github.com/jiying2007/kws-pipeline/pull/441
  - https://github.com/jiying2007/kws-pipeline/pull/441#issuecomment-5906031148

### diag/strict-fuzzy-child-cost-probe

- Tip: `f79f9540aa9b0811e34592f5b77cc9b1b70d7826`; 1 ahead / 35 behind baseline
- Classification: unvalidated-diagnostic-superseded-by-closure
- Finding: Only a diagnostic script is unique. It compares shipping -8.25 against a fixed -16 fuzzy-child cost through an already-present debug replay option. No PR or completed result is associated with this branch. Later #406/#410 closure explicitly stops fuzzy-cost/admission sweeps.
- Action: Keep exact script pointer as an archival design. Do not run it or promote the -16 cost; do not add a live workflow.
- Confidence: medium
- Source: https://github.com/jiying2007/kws-pipeline/tree/f79f9540aa9b0811e34592f5b77cc9b1b70d7826
  - https://github.com/jiying2007/kws-pipeline/pull/406

### exp/ctc-keyword-competition-positive-pair-v1

- Tip: `d08abbf3b5c43155f9bcb3986aef687e7bcb1ec3`; 1 ahead / 32 behind baseline
- Classification: trigger-only-causal-negative
- Finding: Causally valid round1 matched 23/64 to 12/64; FA 72 to 21, but kw1 12/32 to 2/32. The later mechanism correction says improved local prefix gradients did not predict training-time behavior; global blank pressure alone is not the direct prefix mechanism.
- Action: Retain #415 verdict and its corrected mechanism interpretation; no merge of trigger or policy promotion.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/d08abbf3b5c43155f9bcb3986aef687e7bcb1ec3
  - https://github.com/jiying2007/kws-pipeline/pull/415
  - https://github.com/jiying2007/kws-pipeline/pull/415#issuecomment-5891959617
  - https://github.com/jiying2007/kws-pipeline/pull/415#issuecomment-5892277815
  - https://github.com/jiying2007/kws-pipeline/pull/415#issuecomment-5892475455

### exp/decoder-exact-nonroot-probe-v1

- Tip: `03453a6c408cb038ad80f38b0afd0911ab7dd88a`; 8 ahead / 35 behind baseline
- Classification: negative-runtime-counterfactual
- Finding: Baseline equality passed on 520 traces. Exact-only child advance kept 11 positive hits, recovered 1/358 runtime-gap misses, and introduced one new membership. Negative, not product qualification.
- Action: Retain #408 outcome/source pointer; do not restore runtime debug hooks or trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/03453a6c408cb038ad80f38b0afd0911ab7dd88a
  - https://github.com/jiying2007/kws-pipeline/pull/408
  - https://github.com/jiying2007/kws-pipeline/pull/408#issuecomment-5886115027

### exp/decoder-nonroot-blank-evidence-probe-v1

- Tip: `9f7947a07d7a00b7ebf5b7c427e423837a5d2618`; 8 ahead / 35 behind baseline
- Classification: negative-runtime-counterfactual
- Finding: Positive hits rose 11 to 17, but total events 157 to 195 and 52 new keyword memberships failed precision proxy. #410 closes exact/fixed-hold/blank-weighted local decoder line without another sweep.
- Action: Retain #410 outcome/source pointer and local-decoder-line stop condition; no runtime restore.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/9f7947a07d7a00b7ebf5b7c427e423837a5d2618
  - https://github.com/jiying2007/kws-pipeline/pull/410
  - https://github.com/jiying2007/kws-pipeline/pull/410#issuecomment-5886325507

### exp/decoder-nonroot-defer-probe-v1

- Tip: `8589e99d85f01f4a4dd3499548329c2357a5aa8b`; 9 ahead / 35 behind baseline
- Classification: negative-runtime-counterfactual
- Finding: Positive hits rose 11 to 18, but events 157 to 257 and 120 new keyword memberships failed precision proxy. The fixed 0.70 hold is not a safe decoder improvement.
- Action: Retain #409 outcome/source pointer; do not restore fixed-hold implementation.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/8589e99d85f01f4a4dd3499548329c2357a5aa8b
  - https://github.com/jiying2007/kws-pipeline/pull/409
  - https://github.com/jiying2007/kws-pipeline/pull/409#issuecomment-5886256979

### exp/label-prior-primary-ctc-pair-v1-rerun1

- Tip: `d74ff5589568d6a3b81f38e04f6e86688b51aebf`; 1 ahead / 21 behind baseline
- Classification: trigger-only-causal-negative
- Finding: After infrastructure repair, causal-valid label-prior alpha=0.3 pair: matched 45/64 to 43/64, kw1 +6, kw2 -8, FA 69 to 74. Candidate rejected for product path; no sweep.
- Action: Retain #431 verdict; no trigger merge and no default-policy switch.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/d74ff5589568d6a3b81f38e04f6e86688b51aebf
  - https://github.com/jiying2007/kws-pipeline/pull/431
  - https://github.com/jiying2007/kws-pipeline/pull/431#issuecomment-5902262855
  - https://github.com/jiying2007/kws-pipeline/pull/431#issuecomment-5903452492

### exp/label-prior-primary-ctc-pair-v1

- Tip: `a06f36bdeab9a00c5b7581255ea3d8fd4c5a0f51`; 1 ahead / 22 behind baseline
- Classification: trigger-only-infrastructure-failed
- Finding: Exporter incorrectly required nonnegative label-prior CTC metadata although finite signed values are valid. No causal algorithm verdict from this initial run.
- Action: Retain #428 failure and #431 definitive rerun; no merge of trigger.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/a06f36bdeab9a00c5b7581255ea3d8fd4c5a0f51
  - https://github.com/jiying2007/kws-pipeline/pull/428
  - https://github.com/jiying2007/kws-pipeline/pull/428#issuecomment-5901738593
  - https://github.com/jiying2007/kws-pipeline/pull/428#issuecomment-5902027000
  - https://github.com/jiying2007/kws-pipeline/pull/428#issuecomment-5902208336

### exp/runtime-exec-replay-control-v1

- Tip: `54de5a632090bdb4c7645919a78a8ac47e8bebf5`; 1 ahead / 49 behind baseline
- Classification: trigger-only-causal-invalid
- Finding: Round0 model/float/corpus identity diverged despite no replay yet. 64 common hard-negative TTS WAVs differed across AMD/Intel hosts. Round1 differences do not test replay efficacy; #396 was the shared-cache rerun lane.
- Action: Retain #391 causal-invalid classification and source authority; no trigger merge.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/54de5a632090bdb4c7645919a78a8ac47e8bebf5
  - https://github.com/jiying2007/kws-pipeline/pull/391
  - https://github.com/jiying2007/kws-pipeline/pull/391#issuecomment-5872734207
  - https://github.com/jiying2007/kws-pipeline/pull/391#issuecomment-5873876682
  - https://github.com/jiying2007/kws-pipeline/pull/391#issuecomment-5874038181

### exp/runtime-exec-replay-treatment-v1

- Tip: `1edc7bc09a8739b4d49d1be9b03d5532898efb75`; 1 ahead / 49 behind baseline
- Classification: trigger-only-causal-invalid
- Finding: Same round0 counterfactual failure as #391. Preserve as reproducibility evidence, not an algorithmic negative or positive.
- Action: Retain #392 causal-invalid classification and source authority; no trigger merge.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/1edc7bc09a8739b4d49d1be9b03d5532898efb75
  - https://github.com/jiying2007/kws-pipeline/pull/392
  - https://github.com/jiying2007/kws-pipeline/pull/392#issuecomment-5872735108
  - https://github.com/jiying2007/kws-pipeline/pull/392#issuecomment-5873877364
  - https://github.com/jiying2007/kws-pipeline/pull/392#issuecomment-5874039399

### exp/runtime-search-aligned-positive-pair-v1

- Tip: `1455685aa050c3508ca755b480f80dca0ac9ff9e`; 1 ahead / 38 behind baseline
- Classification: trigger-only-causal-negative
- Finding: Valid round1 pair reduced FA 72 to 36 but collapsed matched 23 to 3; kw1 12 to 0 and kw2 11 to 3. Precision-only gain does not qualify the candidate.
- Action: Retain #403 verdict; no trigger merge or policy/weight sweep.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/1455685aa050c3508ca755b480f80dca0ac9ff9e
  - https://github.com/jiying2007/kws-pipeline/pull/403
  - https://github.com/jiying2007/kws-pipeline/pull/403#issuecomment-5884149405

### exp/vad-align-control-current-v1

- Tip: `fd44748f18728621df20b65b15ad14639350d5f6`; 1 ahead / 51 behind baseline
- Classification: trigger-only-infrastructure-failed
- Finding: Initial pair stopped before round1 because formal warm-start guard rejected development-only VAD-aligned checkpoint. Partial round0 is not a completed paired verdict.
- Action: Retain partial #385 result as infrastructure diagnosis; use #388/#389 rerun for treatment verdict.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/fd44748f18728621df20b65b15ad14639350d5f6
  - https://github.com/jiying2007/kws-pipeline/pull/385
  - https://github.com/jiying2007/kws-pipeline/pull/385#issuecomment-5870451775
  - https://github.com/jiying2007/kws-pipeline/pull/385#issuecomment-5871577020

### exp/vad-align-control-current-v2

- Tip: `8ffc1200a900728fde3d392479a863f42132be5c`; 1 ahead / 50 behind baseline
- Classification: trigger-only-completed-control
- Finding: Both base rounds completed after #387 fix. Round1 control matched 10/64 with 14 FA. Later diagnostic NameError does not erase completed base evidence, but no promotion follows.
- Action: Retain #388 base-round result with its post-base diagnostic failure boundary; no trigger merge.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/8ffc1200a900728fde3d392479a863f42132be5c
  - https://github.com/jiying2007/kws-pipeline/pull/388
  - https://github.com/jiying2007/kws-pipeline/pull/388#issuecomment-5871730999
  - https://github.com/jiying2007/kws-pipeline/pull/388#issuecomment-5872598173

### exp/vad-align-suffix-root-v1

- Tip: `2b38131cf20966ac0bc24b92b617ae308ba19cd1`; 1 ahead / 51 behind baseline
- Classification: trigger-only-infrastructure-failed
- Finding: Same warm-start failure as #385. Round0 increased FA substantially but is not the final paired treatment verdict.
- Action: Retain partial #386 risk signal only; use #389 complete base-round verdict.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/2b38131cf20966ac0bc24b92b617ae308ba19cd1
  - https://github.com/jiying2007/kws-pipeline/pull/386
  - https://github.com/jiying2007/kws-pipeline/pull/386#issuecomment-5870452737
  - https://github.com/jiying2007/kws-pipeline/pull/386#issuecomment-5871579550

### exp/vad-align-suffix-root-v2

- Tip: `e45902a23696a95018ac80b9445c36d05c7203bb`; 1 ahead / 50 behind baseline
- Classification: trigger-only-causal-negative
- Finding: Completed paired round1 treatment matched 4/64 vs control 10/64 and FA 20 vs 14. Recall and precision worsened; later unrelated diagnostic failure is separately disclosed.
- Action: Retain #389 rejection; no trigger merge or suffix-root weight sweep.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/e45902a23696a95018ac80b9445c36d05c7203bb
  - https://github.com/jiying2007/kws-pipeline/pull/389
  - https://github.com/jiying2007/kws-pipeline/pull/389#issuecomment-5871732184
  - https://github.com/jiying2007/kws-pipeline/pull/389#issuecomment-5872600013

### feat/label-prior-ctc-objective

- Tip: `8381df126a1f8c9decca2579dff612833bfcc5b1`; 5 ahead / 27 behind baseline
- Classification: superseded-implementation
- Finding: Branch updates train posterior priors each epoch; merged #426 instead estimates raw initial-model training priors once and freezes them, with native CTCLoss hot path and bound export/readback. These are materially different policies, not missing equivalent code. Later #431 rejects the frozen candidate product setting.
- Action: Retain design pointer only. Keep canonical training/ctc_primary.py and #426 contract; do not add the obsolete ctc_objective.py or overwrite trainer.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/8381df126a1f8c9decca2579dff612833bfcc5b1
  - https://github.com/jiying2007/kws-pipeline/pull/426

### feat/restricted-development-dataset-iteration-v1

- Tip: `064493c5bd71c4de40aa9d6569254724ee1ada46`; 15 ahead / 52 behind baseline
- Classification: unmerged-useful-design-needs-current-review
- Finding: Unique schema/validator/tests provide consent-scoped, PII-rejecting, WAV-hash/format/duration verified development feedback with both-keyword and negative-exposure checks and explicit no qualification/shipping authority. Main has generic dataset-driven iteration but lacks this dedicated restricted lane. No PR review or current integration test evidence found.
- Action: Preserve its manifest, intended capability and exact code pointers. If reopening, port only a newly scoped restricted development validator/runner onto current contracts and run new tests; do not wholesale restore stale workflow/core files.
- Confidence: medium
- Source: https://github.com/jiying2007/kws-pipeline/tree/064493c5bd71c4de40aa9d6569254724ee1ada46

### research/a20-cosy30-cross-voice-v1

- Tip: `d5c64d273f680f274d673439f9525cc634089d4a`; 2 ahead / 0 behind baseline
- Classification: completed-generation-source-armed-tip
- Finding: Tip arm.json is run-fixed30-once, reviewed parent 525ed66457e40db66cca2ab778f58b54604a4797. Run 37218182351 completed successfully; current artifacts endpoint returns an empty list, so saved output content was not independently revalidated here. Generation success is neither label acceptance nor KWS qualification.
- Action: Retain the 37-path source lineage and original run. Any integrated source must be disarmed/non-executing, with the original armed commit preserved only as provenance. Do not rerun generation.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/d5c64d273f680f274d673439f9525cc634089d4a
  - https://github.com/jiying2007/kws-pipeline/actions/runs/37218182351

### research/frozen-far-phase-aligned-control-20260930

- Tip: `0595b248b19b856df80902a6656bbea94ddd5a94`; 1 ahead / 9 behind baseline
- Classification: trigger-only-completed-corrective-diagnostic
- Finding: A 192080-sample/12.005-second phase-aligned cold crop reproduced seed1103 at 7191.345 seconds/confidence 0.603046. Long history is unnecessary for this event; phase alone is not proven causal because crop also adds 80 earlier samples. Main roadmap already retains this bounded correction.
- Action: Retain #450 as the authoritative correction to #439/#445; no trigger merge.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/0595b248b19b856df80902a6656bbea94ddd5a94
  - https://github.com/jiying2007/kws-pipeline/pull/450
  - https://github.com/jiying2007/kws-pipeline/pull/450#issuecomment-5907722111

### training/keyword-set-contract-v1

- Tip: `986d44c9e0efe15ce0656413a2f2112aeccaea96`; 17 ahead / 179 behind baseline
- Classification: explicitly-retained-design
- Finding: Main cleanup workflow and cleanup-contract test explicitly protect this design branch. Unique contract loader/generator/tests validate semantic keyword identities and safe corpus-plan generation. The active product still has a fixed shipping keyword authority; no current integration validation exists for this design.
- Action: Preserve branch identity and design in the index. Do not replace current product config/training/replay plumbing with this 179-commits-behind branch. Fresh scoped integration is needed if generic keyword contracts are selected.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/986d44c9e0efe15ce0656413a2f2112aeccaea96

### training/keyword-set-identity-v1

- Tip: `2d52b58f36163fb93b9ba1b1356fa184c187fa2f`; 6 ahead / 179 behind baseline
- Classification: explicitly-retained-design
- Finding: Explicitly protected by main cleanup workflow/test. This alternative hashes semantic token and keyword rows and binds config to a keyword-set identity. It is not byte-equivalent to the contract branch and must not be silently combined with it.
- Action: Preserve branch identity as an alternative design, not a second active implementation. Reconcile with keyword-set-contract before any new scoped integration.
- Confidence: high
- Source: https://github.com/jiying2007/kws-pipeline/tree/2d52b58f36163fb93b9ba1b1356fa184c187fa2f

