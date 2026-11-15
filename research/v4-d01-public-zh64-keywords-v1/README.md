# V4 D01: public64 per-keyword research inputs

This saved-output adapter makes the completed Chinese public64 observations
consumable as **weak clip-level keyword presence**. It uses the original audio
identity, the three saved native ASR outputs and their completed independent
audit. It runs no recognizer, native audit, cache audit, resampler or optimizer.

## What becomes usable

The 64 recordings contain 128 keyword slots. Under the new versioned protocol,
6 slots have three complete, nonempty native outputs containing the full
literal target; 122 remain unknown and none is declared absent.

| Keyword | Research train | Research development | Unknown slots |
| --- | ---: | ---: | ---: |
| 你好小窝 | 1 | 0 | 63 |
| 小窝小窝 | 3 | 2 | 59 |

The newly usable full-sentence disagreement is `clip-000029` (Dylan, train).
Qwen returned `你好，小窝。`, SenseVoice returned `你好小窝`, and Whisper returned
`你好小窝儿`. The full four-character keyword occurs in every output. The
Whisper suffix is retained as evidence; it is not deleted to fabricate a full
transcript target. Its acoustic realization and the actual word endpoint are
unknown. The other five present slots are `clip-000002`, `clip-000006`,
`clip-000018`, `clip-000026` and `clip-000030`.

The three-way `笑我笑我` agreement does not match `小窝小窝`. Six two-vote
`你好小窝` cases, other partial or disputed words and all 48 nearphone recordings
remain unknown. Those nearphone recordings keep all 96 keyword slots in
quarantine, even if a future fixture supplies three positive votes.

## New research authority and preserved history

V4 permits a versioned per-keyword machine-label research protocol. The source
ZIP's unchanged policy says `use=controlled-noncommercial-research`, with
`intended_label_tier=machine_weak_label`; it does not prohibit all future
research loss. Its `future_admission_requires` list is retained verbatim in the
snapshot. This adapter does not claim those formal admission requirements have
all been satisfied, and does not call its output formally admitted silver.

The new `research_keyword_presence_eligible` and `allowed_use` fields describe
only this research protocol. The train consumer returns four recordings with
positive keyword masks and a fixed weak weight of **0.25**. This is a declared
research weight, not an estimated correctness probability. The development
consumer returns two recordings for masked weak-presence checks and sets
`research_optimizer_allowed=false`. Existing Vivian/Dylan train and
Serena/Uncle_Fu development roles are preserved. Development observations are
not independent human evaluation or formal FAR/FRR references.

Unknown targets are `null`, their masks are false and their weights are zero.
A known keyword does not make the other keyword absent. There is no negative
supervision in this package. It is a positive supplement for a separately
declared research recipe, not a standalone qualified training corpus.

All historical `training_admitted`, `formal_admission`,
`optimizer_loss_authorized`, human-gold and evaluation flags are copied without
change under `historical_authority`. New records retain
`training_admitted=false`, `formal_admission=false`, `human_gold=false` and
`independent_test=false`. The existing formal training and independent-reference
gates continue to reject these records. Only the new typed consumer admits
the stated weak research use. No existing admission validator or trust root is
changed.

Full-transcript CTC targets, event timestamps and acoustic word endpoints are
not produced. `ctc_mask=false`, `target_ids=null`, `full_text=null` and
`event_times=null` also hold in the consumed research rows. Text match indices
are Unicode character positions in saved transcripts; they are never sample
indices. A future strong temporal event experiment needs additional timing
evidence. A weak clip-presence loss cannot silently become frame labels.

## Evidence and normalization

The adapter pins the completed union audit and two original artifact ZIPs by
byte length and SHA-256. It reads their JSON metadata and saved indexes, joins
all 192 provider records to the original/derived WAV and PCM identities, and
retains per-provider run, policy, request, started/completed/raw-output hashes
and exact artifact member references. The completed native audit is reused;
it is not rerun. No audio is decoded or regenerated.

The per-word rule uses the existing
`speech_asr_consensus.normalized_transcript` fixed punctuation-removal policy.
There is no NFKC, homophone, spelling, simplified/traditional or general
whitespace folding. The old descriptive NFKC full-sentence comparison is
retained separately and is not relabeled as the runtime or new keyword rule.
The new span voter additionally rejects raw character spans crossing Chinese
or ASCII sentence-ending punctuation (`。！？.!?`), CR/LF, vertical/form feeds,
NEL, or Unicode line/paragraph separators. It retains each rejected span and
its boundary positions for inspection. Thus `你好。小窝` and `小窝！小窝` cannot
be joined into a keyword by normalization. The original normalizer is
unchanged; the comma in `你好，小窝` remains an allowed word-internal pause.
The new protocol was declared after all 192 ASR outputs were available; it is
not represented as a preregistered ASR or blind evaluation protocol.

Requested source text is not an input to the positive rule. Original intent
metadata is retained only as provenance and for the inherited nearphone
restriction. Three ASR votes are correlated machine evidence; Qwen TTS and
Qwen ASR are not independent acoustic truth. Real speaker independence remains
unestablished.

## Create, validate and consume

`--delivery` is the existing `KWS_V4_Execution_20261114` directory containing
the pinned `audits/public64_v2_cross_run.json` and `artifacts/` files. The
destination must be new and outside both the checkout and prior delivery.

```sh
python3 -B research/v4-d01-public-zh64-keywords-v1/project.py create \
  --delivery /path/to/KWS_V4_Execution_20261114 \
  --output /path/to/d01-public64-keywords

python3 -B research/v4-d01-public-zh64-keywords-v1/project.py verify \
  --delivery /path/to/KWS_V4_Execution_20261114 \
  --package /path/to/d01-public64-keywords \
  --manifest-sha256 THE_CREATED_MANIFEST_SHA256

python3 -B research/v4-d01-public-zh64-keywords-v1/project.py consume \
  --delivery /path/to/KWS_V4_Execution_20261114 \
  --package /path/to/d01-public64-keywords \
  --manifest-sha256 THE_CREATED_MANIFEST_SHA256 --phase train
```

Use `--phase development` for the two held-back development records. `consume`
revalidates the package against the pinned saved inputs before emitting the
typed rows on stdout. The Python entry point is
`consume(delivery, package, expected_manifest_sha256, phase)`. Pure helper
functions are for already verified data or software fixtures; they do not
authenticate arbitrary JSON or confer source authority.

The package contains `snapshot.json` (all 64 records and exact native evidence
references), `keyword-evidence.jsonl` (all 128 slots),
`present-evidence.jsonl` (the six present slots), and a `MANIFEST.json` binding
all payload and producer source bytes. Self-rehashing an edited label or
manifest does not pass verification: output must equal the projection of the
pinned original inputs. Neither package creation nor consumption starts a
training job.

## Checks

```sh
python3 -B research/v4-d01-public-zh64-keywords-v1/test_project.py
D01_SAVED_DELIVERY=/path/to/KWS_V4_Execution_20261114 \
  python3 -B research/v4-d01-public-zh64-keywords-v1/test_project.py
```

Fixtures cover suffix retention, partial words, homophones, missing native
outputs, unknown-versus-absent, nearphone quarantine, fixed normalization,
cross-sentence/line exclusion, per-keyword masks, train/development separation, target/type/weight/timing
tampering and continued rejection by existing formal admission checks. The
optional saved-output check projects the existing files, verifies the exact
4/2 consumer split, and rejects a rehashed forged label. It performs no model,
audio or cache execution.
