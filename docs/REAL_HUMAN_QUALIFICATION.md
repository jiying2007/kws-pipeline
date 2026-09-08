# Real-human final-AFE qualification

This document defines Phase A of product qualification for the frozen commercial candidate `deployment-c20f3eb88e43`. It does **not** authorize model retraining and it does **not** make the product shipping-approved. Phase B physical target-board evidence remains mandatory after Phase A passes.

## Frozen tuple

Phase A evaluates exactly:

- deployment Release: `deployment-c20f3eb88e43`;
- deployment source: `c20f3eb88e43b7e332ae79ba5ed147ccfeede27f`;
- Model Release: `model-749187ec1d66`;
- model SHA256: `ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`;
- wake words: `你好小窝` and `小窝小窝`, threshold `0.55`;
- final product microphones/enclosure and the frozen command-AFE adapter supplied to the qualification run.

Do not change threshold, decoder, replay, model weights or qualification seed to make a real-human run pass. Formal model seed `271838` is consumed/frozen. Reserved seed `271839` is used only if later evidence justifies an intentional new model candidate.

## Privacy boundary

Raw human audio is restricted product evidence. Do not commit it to this public repository and do not attach raw or post-AFE WAV files to a public GitHub Release or Actions artifact.

The repository may retain only content hashes, pseudonymous IDs, aggregate metrics, coverage, AFE identity/receipts, qualification summaries and attestations. The corpus contract explicitly rejects common PII fields such as name, email, phone, address and ID number.

Run the real workflow only on a controlled self-hosted runner carrying both labels:

```text
self-hosted
kws-real-audio
```

The runner must have local access to the restricted raw corpus and the final `audio-pipeline` executable/configuration. GitHub-hosted runners are not a valid substitute.

## Phase-A policy

The machine authority is `commercial/real-human-qualification.policy.json`.

Minimum evidence:

- at least 40 pseudonymous speakers;
- at least 2 independent sessions per positive speaker;
- at least 1200 expected wakes total;
- at least 600 expected wakes for each shipping keyword;
- at least 24 hours of post-AFE negative exposure;
- 95% one-sided statistical confidence bounds.

Aggregate gates:

- FRR <= 5%;
- one-sided FRR upper bound <= 6.5%;
- FAR <= 0.10/hour;
- one-sided FAR upper bound <= 0.13/hour;
- p95 post-end wake latency <= 500 ms.

The 24-hour/0-FA case is intentionally close to the 95% Poisson upper-bound requirement. One observed false accept may therefore fail the confidence gate even when observed FAR alone remains below 0.10/hour.

Critical positive coverage also includes real 3-5 m, rear, playback, double-talk and robot-motion slices. Critical negative exposure includes household, speech-confusion, playback and robot-motion time. Exact minima are in the policy JSON and are checked before the frozen KWS model is allowed to see the corpus.

## Private corpus preparation

Maintain a private draft JSON with `schema_version=1`, `corpus_role=fresh-held-out-qualification`, a stable `qualification_id`, deployment tag, and one recording entry per raw WAV. Use only pseudonymous `speaker_id`, `session_id`, `source_id`, `room_id` and `device_id` values.

Each recording supplies at least:

```json
{
  "recording": "q000001",
  "input_path": "speaker-001/session-01/q000001.wav",
  "speaker_id": "speaker-001",
  "session_id": "session-01",
  "source_id": "capture-000001",
  "room_id": "room-01",
  "device_id": "dut-01",
  "distance_m": 3.5,
  "azimuth_deg": 180.0,
  "snr_db": 8.0,
  "tags": ["rear", "robot_motion"],
  "expected": [
    {"keyword_id": 1, "start_s": 1.20, "end_s": 2.05}
  ],
  "consent_scope": "product-kws-qualification",
  "retention_class": "restricted-raw-audio"
}
```

Negative recordings use `"expected": []`. Do not label a negative recording with an expected wake just to increase positive counts.

Seal the private draft locally so hashes, byte sizes, duration and capture geometry come from the actual WAV rather than hand-entered metadata:

```bash
python3 tools/seal_real_human_corpus.py \
  --draft /restricted/kws/q1/draft.json \
  --audio-root /restricted/kws/q1/wav \
  --output /restricted/kws/q1/sealed.json
```

Then validate policy coverage and raw identities without exposing the corpus to the model:

```bash
python3 tools/validate_real_human_corpus.py \
  --manifest /restricted/kws/q1/sealed.json \
  --audio-root /restricted/kws/q1/wav \
  --policy commercial/real-human-qualification.policy.json \
  --public-summary /restricted/kws/q1/intake-summary.json
```

`commercial/real-human-corpus.schema.json` describes the sealed manifest.

## Final AFE adapter

The adapter is a private JSON file described by `commercial/final-afe-adapter.schema.json`. Example shape:

```json
{
  "schema_version": 1,
  "executable_path": "/opt/audio-pipeline/bin/audio_pipeline_eval",
  "command_argv": [
    "{executable}",
    "--config", "/opt/audio-pipeline/config/product.json",
    "--input", "{input}",
    "--output", "{output}",
    "--result", "{result}"
  ],
  "config_files": [
    "/opt/audio-pipeline/config/product.json"
  ],
  "pipeline_source_sha": "0123456789abcdef0123456789abcdef01234567",
  "sku": "PCR02",
  "microphone_revision": "REPLACE_ME",
  "enclosure_revision": "REPLACE_ME",
  "audio_route": "REPLACE_ME",
  "toolchain": "REPLACE_ME"
}
```

The command must create `{output}` as mono PCM16 16 kHz WAV and `{result}` as JSON containing a non-negative integer `latency_samples`.

Before the corpus is consumed, `tools/final_afe_identity.py` resolves the real executable/config files and freezes:

- executable SHA256;
- canonical config-bundle SHA256;
- command-template SHA256;
- pipeline source SHA where supplied;
- SKU/microphone/enclosure/audio-route/toolchain identity.

The command text itself is not copied into public evidence. `tools/run_final_afe_corpus.py` recomputes the identity before execution and refuses any drift.

## Fresh holdout / no-retry rule

Freshness is content-addressed by the sealed corpus identity. After corpus intake and final-AFE identity preflight succeed, but **before final AFE or KWS processing**, the workflow creates an immutable public marker:

```text
human-exposure-<corpus-sha16>
```

That marker contains hashes/identity only, never audio. Once the marker exists, the corpus is permanently `consumed-and-frozen` for fresh qualification. A failed run cannot tune the model, threshold or AFE and then reuse the same corpus as fresh qualification. It may later be used only as a regression set.

If the final AFE executable/config changes after the marker is created, the run fails and the corpus remains consumed.

## Running the workflow

The workflow must be dispatched from the exact current protected `main`; `governance/require_current_main.sh` rejects branch refs, stale main SHAs or weakened live rulesets before any held-out corpus is consumed.

The controlled runner keeps private paths out of public workflow inputs. It defines:

```text
KWS_REAL_HUMAN_ROOT
KWS_FINAL_AFE_ROOT
```

and the workflow accepts only pseudonymous `qualification_id` / `afe_profile` identifiers plus the frozen deployment tag.

The workflow performs, in order:

1. require exact current protected `main` and the live terminal ruleset;
2. verify the immutable deployment and all `DEPLOYMENT_SHA256SUMS` subjects;
3. build the exact deployment-source `kws_wav` runtime;
4. validate private human corpus identities/coverage/PII boundary;
5. freeze actual final-AFE identity;
6. create the immutable no-retry exposure marker;
7. execute the frozen final AFE;
8. run the frozen model/keyword pack through the real C runtime;
9. score FRR/FAR/latency with the existing evaluation engine;
10. use post-AFE negative duration for FAR exposure;
11. calculate one-sided Wilson/Poisson bounds;
12. enforce aggregate and critical-slice gates;
13. hash and attest non-audio evidence;
14. retain non-audio evidence only.

A failure after exposure leaves the corpus consumed. The job fails; it does not automatically retrain or change any gate.

## Successful Phase-A result

A passing run publishes an immutable marker:

```text
human-qualified-<corpus-sha16>
```

The public Release contains only aggregate/non-audio evidence: Phase-A receipt, qualification summary, corpus intake summary, frozen AFE identity, AFE corpus summary and evidence checksums. It explicitly retains `shipping_approved=false`.

The machine result shape is documented by `commercial/real-human-qualification-summary.schema.json`.

Phase A PASS changes the next gate to:

```text
physical-target-board-performance-and-soak
```

Continue with `TARGET_AND_SHIPPING_PROMOTION.md`. Phase B requires qualified physical evidence from multiple unique DUTs, and only the explicit Phase-C `shipping-approval` workflow may ultimately emit an immutable receipt with `shipping_approved=true`.
