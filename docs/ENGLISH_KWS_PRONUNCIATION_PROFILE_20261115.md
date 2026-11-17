# English wake-word research pronunciation and software delivery — 2026-11-17

Status: active synthetic research policy, schema v3. The owner requested English delivery using the Chinese synthetic-data approach, without making new human recordings a prerequisite. V4.1 places the frozen speech embedding plus causal English head first. English FSMN and Chinese-weight transfer are conditional later research options.

## Two explicit evidence scopes

The active profile at `configs/research/english-pronunciation-profile-20261115.json` permits new synthetic US pronunciation hypotheses, machine-admitted research training, and reproducible C/ARM software delivery. Human pronunciation gold, new recordings, and physical SSC305 access are not prerequisites for these stages. No new owner IPA confirmation is required.

Hey Mero's final native-US pronunciation remains unresolved (`ipa: null`). The latest instruction about actual US pronunciation remains the authority for any eventual final native pronunciation claim. A versioned synthetic experiment can compare an explicitly named natural US-G2P candidate and a historical long-i candidate, retaining the exact input, generated G2P output and observed audio evidence. Neither becomes the final target by being generated or trained.

Hey Vosen's existing /heɪ ˈvoʊzən/ request is retained. New research plans may separately test natural /s/ and requested /z/ candidates; their names and machine observations remain explicit. An ordinary accent or a nearby candidate is not automatically a hard negative.

The original image request remains at `configs/research/history/english-pronunciation-profile-image-request-20261116.json`. The superseded v2 policy is preserved byte for byte at `configs/research/history/english-pronunciation-profile-native-evidence-policy-20261117.json`. Its human-evidence prerequisite no longer governs new synthetic research.

## Executable machine admission

`training/english_synthetic_silver.py` provides `build_admission(corpus, evidence, root, *, profile, plan)` and a byte-only CLI:

```sh
python3 training/english_synthetic_silver.py \
  --profile configs/research/english-pronunciation-profile-20261115.json \
  --plan /absolute/frozen-plan.json \
  --corpus /absolute/corpus/CORPUS_MANIFEST.json \
  --evidence /absolute/corpus/MACHINE_EVIDENCE.json \
  --root /absolute/corpus \
  --output /absolute/new-ADMISSION.json
```

The reviewed experiment pins this module, the profile, the plan and its actual capture worker. The resulting artifact SHA authenticates the retained bundle. The validator re-reads actual WAV/PCM and capture bytes; caller-authored JSON alone cannot establish where audio or observations came from.

The first additive design uses 24 previously unused Piper LibriTTS source identities and eight cells per identity: two Mero candidates, two Vosen candidates, and four ordinary sentences. Roles are frozen before generation: 12 train, four development, four calibration and four locked-test. Every intended row remains in the manifest, including failed generation and missing evidence.

Three distinct, pinned blind ASR families and a fourth independent phoneme observer are required. Their complete-source native observations bind the same PCM/WAV/sample identity. Requested text, generator G2P, target-conditioned alignment, and model events do not supply independent pronunciation evidence.

| Tier | Fixed machine rule | Allowed use |
| --- | --- | --- |
| silver | All three ASR transcripts map through predeclared compatible aliases; the independent phone sequence supports an entire declared phrase family | Synthetic research supervision in its assigned role |
| weak | Two ASR transcripts support the same phrase family and independent whole-phrase phones support it; explicit contradictory wake evidence vetoes | A separately declared research objective, with its tier and weight retained |
| unknown | Missing, incomplete, contradictory or insufficient evidence | Preserved for diagnosis; supervised loss and event denominator are masked |

Raw spelling equality is not required for brand words. Aliases and phonetic families are fixed in code before generation, without target-KWS results. Representation normalization may remove stress/length marks and map equivalent rhotic symbols; it does not certify duration, stress, final pronunciation or all nearphone equivalences. Missing the Hey nucleus or an incompatible name core remains unknown.

Ordinary negatives use four predeclared complete sentences. They need positive machine support for that sentence and no observed target path. ASR failure to recognize a wake word alone does not create a negative. The new batch contains no near-miss hard-negative class.

A source group must contain both positive classes and ordinary negatives in the declared usable tiers before it contributes supervised rows. The plan fixes minimum balanced source groups and train-only tier weights before execution. The trainer rechecks these floors after feature geometry. A deficit stops that experiment; it never causes resplitting or moving exposed development data into training.

## Reuse from Chinese research

The Chinese three-ASR path demonstrates synthetic machine-evidence admission separately from human truth. Its existing CTC transcript/token rules are Chinese-specific, and its source resolver has separate reviewed source roots. The English implementation reuses complete-source identities, raw-output retention, distinct recognizer families, explicit machine tiers and source-disjoint roles. It does not reinterpret English as Chinese CTC tokens or rewrite the immutable English63 weak bridge.

## Training and evaluation boundaries

Keep existing English63, final144/snapshot160, their original WAVs, old admission flags, requests, quarantines, source roles and saved experiments unchanged. New snapshots are additive. Do not regenerate known source clips or silently relabel historical audio.

Clone-parent, reference-speaker, augmentation-parent and dataset identity must remain within one role. Source speaker identities shared by LibriTTS and LibriSpeech must be excluded or grouped consistently. A generated voice is not an independent real speaker claim.

The current primary route uses the existing frozen speech embedding and causal 1536→32→3 head, cached features, new admitted data and explicit replay of the original TRAIN subset. Newly reserved development data can select the predeclared checkpoint; calibration can set only the separately declared calibration choices. Locked-test is opened for model evaluation only after selection. Previously exposed legacy development inputs remain unchanged regression evidence and cannot enter gradients, teacher caches or selection.

All existing frozen event/regression gates remain unchanged. A failed candidate remains rejected. Source-code tests and synthetic event results do not imply independent real-world quality.

## Software delivery and remaining measurement claims

Research software delivery requires exact data/code/weights, actual finite training records, source-held-out synthetic event results, unchanged legacy regression gates, complete streaming PCM/frontend/model/decoder parity, reset/discontinuity checks, native/ARM verification and a reproducible installable SDK. Report strict silver and weak counts separately and report actual accepted-source coverage.

The existing numeric targets remain: combined FAR ≤0.1/hour, per-word FRR ≤5%, and word-end-to-consumable-callback P95 ≤500 ms, with one-sided 95% upper bounds where applicable. Independent real-world FAR/FRR and physical SSC305 CPU, RAM, latency and long-run measurements retain their actual unknown/unmeasured status until corresponding evidence exists. Those future measurements do not block synthetic research training or verified software delivery.

Exact third-party revisions, weights, source rights, model licenses and dependencies remain part of each research asset review. No source or model is admitted simply by being named here.
