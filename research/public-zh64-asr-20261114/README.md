# First native three-ASR screen of the existing public Chinese64

This new lane consumes the **existing** public Qwen non-clone64 artifact from
run `41814320901`, artifact `13314781496`, TTS source commit
`f0a26c5f39910ae6a8cb674a31cf08e57fcfc2f2`. It generates no audio, runs no
optimizer, and does not reopen the old four-clip pilot or any consumed branch.
The 64 original WAV and float-PCM hashes differ from the four pilot clips.

The immutable input ZIP is 11,560,534 bytes, SHA256
`0704e28608f636fa14e8bfa30ebc5ed5b870b9576ec8c80fbe15da9e05033ac5`.
Its 97 indexed members plus index include 64 native mono 24 kHz float WAVs,
3,123,840 frames, 130.16 seconds. `input_adapter.py` verifies every member,
generation row, original WAV hash, original float-PCM hash and exact denominator
before any model setup. This is public preset synthesis with zero reference
voice inputs. The owner-controlled Chinese batches are outside this lane.

## Frozen research roles and qualification boundary

The roles are decided before looking at any ASR output. Both seeds and every
requested phrase from a preset remain together.

| Presets | Research role | Total rows | Positive intent | Nearphone intent |
| --- | --- | ---: | ---: | ---: |
| Vivian, Dylan | train | 32 | 8 | 24 |
| Serena, Uncle_Fu | dev | 32 | 8 | 24 |

These are prospective research roles, not permission to optimize a loss or a
claim of independent natural-speaker evaluation. The original source policy's
`intended_split=train_only` stays unchanged in the original ZIP and is retained
beside the new research role. **All 48 nearphone rows remain quarantined**,
including after ASR agreement. There are zero ordinary natural negatives.
The 16 positive-intent rows are candidates; requested text is not an observed
transcript. No CTC target, loss weight or training manifest is created here.

`optimizer_loss_authorized`, `training_admitted`, `human_gold`,
`natural_negative_qualified` and `evaluation_or_production_use` are false.
There is no 128-step training runner: repeatedly optimizing the eight train-side
positive intentions would not supply the missing ordinary-negative or natural
qualification evidence. This run produces candidate and challenge observations
for a later separately reviewed projection.

## Exact execution and source provenance

The new control code records its actual merged-main `GITHUB_SHA` and run ID.
That identity differs from both the original TTS commit above and the retained
native-helper commit `22505bf68b6c57747a8da35fd83e6ea4982d096c`.
The latter's tokenizer-v4 source lock is SHA256
`0a7a489e94f7bc940ae8e58cc4be43a8f59873e87cd599303297c6a8b962896a`:
all 127 original file identities and all 78 staged aliases are verified and
preserved. No historical source, prior source lock or prior claim is changed.

The actual models remain Qwen3-ASR-0.6B revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, SenseVoiceSmall revision
`3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`, and official multilingual
Whisper-small at the retained source/weight pins. Qwen uses empty context,
automatic language and genuine EOS capture; Sense uses full-input CTC with
the verified loaded SentencePiece binding and no ITN/VAD; Whisper uses its
Chinese short greedy decode with actual pre-finalize EOT observations.
The original Qwen/Sense and Whisper source/model/runtime locks, checkpoint
application checks, licenses and attribution remain in the retained closure.
SenseVoice's model license remains distinct from its code license. No model
weight is included in the public result.

The original float signal is converted once using the locked runtime's
NumPy 1.26.4 and SciPy 1.17.1: `resample_poly(2,3,window=('kaiser',5.0),
padtype='constant')`, then floor to PCM16. Nonfinite values, clipping, extra
frames, silence repair, gain and trimming are not accepted. All 64 original to
derived joins are retained. The derivation process can read producer metadata;
recognizers receive only content-addressed WAVs and opaque source/request IDs.
The original ZIP, role map and derivation code containing requested text are
not mounted in recognition containers.

One actual job performs bounded setup, observes all three loaded native
configurations with guarded zero-forward preparation, freezes their combined
policy, and immediately proceeds to real audio. This is not a separate no-audio
workflow. The existing balanced producer order gives four batches of 16, with
a distinct process/output directory per provider and batch. At most 64 calls
per provider, 192 total, are possible. An attempted cell is never retried.
An ordinary provider failure may leave later cells of that batch not run;
other still-unconsumed batches can proceed within the same global budget.
Unconfirmed container cleanup stops the entire campaign.

The bounded runtime is the retained official digest-pinned container:
4 CPU, 12 GiB memory, 256 PIDs, no privilege/capability additions, read-only code
and runtime, and no network during derivation, model loading or inference.
Setup has the original acquisition bounds and a 1,200-second stage ceiling;
each inference worker has a 1,500-second outer ceiling and its original
300-second cell ceiling. The job has a 9,000-second inner budget and a
165-minute Actions ceiling, including publication reserve. The 192 figure is
an upper bound, not a promise that timeouts are reported as completions.

## One-time trigger and persistent evidence

PR checks only verify source and exercise standard-library boundary tests.
Only a main push touching the fixed request can reach the claim job. That
separate job holds `contents:write` solely to create
`refs/tags/public-zh64-asr-20261114-v1-consumed`. The ref is never moved or
deleted; request reformatting cannot create a second claim. Both raw and
canonical request hashes are retained. Run attempt must be 1, but workflow
run number is not used, so a PR source check does not consume first execution.

The actual job has read-only repository permissions. Its token is scoped to
the artifact/claim download step, is absent from the model host environment,
and is never passed into a model container. It downloads the exact same-repo
original artifact rather than requiring cross-repository credentials.

The new artifact persists the original ZIP **as identical reused bytes**, all
64 derivatives and their original joins, the frozen role map, per-batch blind
jobs, policy, source bytes, actual worker failures, raw native records, and a
complete 64-by-3 terminal denominator. States distinguish complete native
observation, consumed failure, rejected evidence, unknown lifecycle state,
and not run. Text is rechecked against native tokens/decoder returns and input
hashes; decoder termination is not acoustic-verbatim completeness.
Some retained Qwen/Sense error paths do not persist a rejected decoder's full
text; their attempted slots stay failed or unknown, with the available original
failure evidence retained. Missing text is never reconstructed or called a
successful observation.
An inventory binds every retained member by length and SHA256. Runtime model
directories and host credentials are excluded. Keeping the original ZIP in
this new evidence artifact protects reproducibility after the original short
Actions TTL expires; it is not another generation run.

## Source-only validation

From a source checkout with the pinned retained checkout available:

```bash
python3 -B research/public-zh64-asr-20261114/host.py --retained /path/to/retained --check-source
KWS_PUBLIC64_RETAINED=/path/to/retained python3 -B research/public-zh64-asr-20261114/test_contract.py
```

These commands install nothing, download nothing and perform no model calls.
The same tests run under normal, optimized and stripped Python in the new
workflow. Source-only tests cannot establish a successful real ASR run; only
the first actual Actions execution and retained native evidence can do that.
# First execution of the sixteen unstarted Whisper cells

Run `43608230908` ended `incomplete_no_retry`. Its immutable artifact
`14119952748` is 42,715,116 bytes with SHA256
`e731141401bea5220c5fdb9dd066c044ece7bb860fb615a6cc50768acffe40e6`.
It contains 176 independently validated native observations: Qwen 64,
SenseVoice 64, Whisper 48. Batch 3 SenseVoice reported an output-stability
supervisor error after writing its sixteen complete native records; the
container cleanup completed. Those sixteen records remain consumed.

Batch 3 Whisper failed at the unchanged host-capacity check, before output
directory creation or any Docker operation. Its entire inference subtree is
absent from the 3,746-member closed original artifact and index, and global
cells 49–64 are `not_run`, `attempt_consumed=false`, with no native start,
claim, completion or text. `continuation_request.json` freezes only those
sixteen cells, the exact original WAV/PCM identities, local/global mapping,
blind job, identity map and preset roles. They are all nearphone quarantine
rows: eight research-train roles and eight research-dev roles; no optimizer
loss is authorized.

`continuation.py` verifies the original ZIP and every indexed member before
runtime setup. It copies the original derived PCM16 WAVs, blind `job.json`
and `identity-map-3.json` byte for byte. It neither generates audio nor
resamples, retries, reprocesses the previous 176 observations, loads Qwen or
SenseVoice, or runs training. It installs only the original pinned Whisper
runtime and performs fresh guarded zero-forward preparation immediately
before the sixteen first audio calls in the same actual job. The retained
14 GiB available-host-memory requirement, 12 GiB container limit, offline
recognition, decoder criteria, 300-second per-cell and 1,500-second stage
limits are unchanged. This controller has a 3,600-second inner limit and at
most sixteen new audio calls. Container cleanup uncertainty stops the route
and preserves unknown/reserved outcomes; it does not permit another launch.

The existing workflow gains a push path for the new request and separate
claim/actual jobs. Its fixed permanent claim is
`refs/tags/public-zh64-whisper16-first-20261114-v1-consumed`. The first campaign
request, `host.py`, original worker and old claim remain unchanged. The old
actual job skips when its already-existing claim is read. Formatting the new
request cannot create a second claim. PR runs execute source-only checks;
only the main-push claim job has write permission, and the model host and
containers receive no repository credential.

Fresh preparation records the **actual new controller head**. The original
Whisper source lock included the executed head, so the complete identity hash
must reflect that provenance change. The continuation allows exactly that
source-lock field and its dependent source/config identity hashes to change.
All source inventory entries and their saved bytes, installed-runtime
inventory, assets, model, effective decoding, input and execution settings
must remain identical. It does not claim an unchanged complete identity hash
or write the old head as a new execution's head. The new policy retains the
old Qwen/SenseVoice identity references and replaces only the freshly checked
Whisper identity; those two models are not loaded in this run. The previous
176 receipts retain their original policy and are never rebound to the new
one.

The new artifact separately retains the parent summary, observations,
policy, Whisper identity/source lock and index under `provenance/parent/`;
`provenance/parent-binding.json` records the exact parent ZIP, member hashes
and sixteen-cell exclusion proof. It also preserves both claims, the closed
parent-run API record, fresh prepare proof, actual native records and a
complete sixteen-slot terminal. `combined-observations.json` is an explicit
cross-run composition: it copies the 176 original observations and replaces
only the sixteen original unstarted Whisper entries. This is not a new
single-run three-model experiment. The original failed terminal remains
`incomplete_no_retry`, including both recorded supervisor failures.

All labels remain candidate/challenge evidence: `training_admitted=false`,
`human_gold=false`, `optimizer_loss_authorized=false`, and natural-negative,
evaluation and production qualification remain false. The two raw result
ZIPs are independently retained; the second artifact references the original
42 MB ZIP by exact identity rather than duplicating it.

Source-only checks:

```sh
python3 -B research/public-zh64-asr-20261114/continuation.py --retained /path/to/22505bf-checkout --check-source
python3 -B research/public-zh64-asr-20261114/test_continuation.py
python3 -B -O research/public-zh64-asr-20261114/test_continuation.py
python3 -B -OO research/public-zh64-asr-20261114/test_continuation.py
```
