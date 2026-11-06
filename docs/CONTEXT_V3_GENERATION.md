# Context v3: new24 full-text generation

This additive experiment declares 72 new requests for the 24 added D2 reference
identities. Every identity retains its existing family, speaker and split and
receives one K1 context, one K2 context and one ordinary speech request. The
overall intent mix is 24/24/24, with 54 train-candidate and 18 development-only
requests. The first eight requests contain three K1, three K2 and two ordinary
intents from eight identities.

`kws-data/research/auto-tts-voice-clone-20261102/context_supplement_v3.py`
owns the plan. It validates the complete new24 D2 parent and the original
context v2 parent, including the old8 D2, context v1 and D1 ancestors. All new
normalized prompt hashes and cell identities must be disjoint from those
historical requests. Merely changing an ID, seed or punctuation cannot turn an
old request into a new v3 request. The fixed full2601 vocabulary is unchanged.

The generation owner is `tools/research_context_v3_generate.py`. Its data and
execution source snapshots, references, runtime setup, model assets, recipes
and queue are hash bound. Recipe and stage generation settings must equal the
independently verified data plan's settings. No old claims are retried.

## Declared allocation and one-chunk execution

Preparation requires `--global-minutes 90 --job-minutes 10`. With the default
`--chunk-size 8`, the complete declaration has nine generation jobs. Every
worker has a 570-second internal deadline within its 10-minute queue limit.
The deadline exception cannot be swallowed as a per-cell signal-quality
failure. The separate selective ASR allocation is at most nine 10-minute jobs
and 216 native recognition calls. The new aggregate ceiling is therefore
180 runner-minutes. Previous D1, D2 or context allocations are neither reused
nor erased by this new declaration.

After preparing a frozen source tree and private stage:

```bash
python3 tools/research_context_v3_generate.py guarded-next \
  --stage /absolute/private/context-v3/stage.json \
  --checkpoint-wait-seconds 900
```

`guarded-next` keeps one live `research_d2_checkpoints.Gate` across the launch
intent acknowledgement, queue call and terminal acknowledgement. It emits a
capsule descriptor and waits for an external save receipt before taking the
heavy model lock or calling a model. The worker independently checks that the
saved intent binds its exact live owner, stage, queue, recipe and cell IDs.
Calling the queue directly without that authorization fails before model
construction. Native arrays and original FLOAT WAVs are retained before signal
quality screening; a failed screen never removes the native return.

One invocation advances at most one new generation job. A new invocation cannot
reuse an old pending intent, including an intent restored from its saved ZIP.
After a maintained generation/ASR campaign registers this stage, standalone
`guarded-next` is disabled under the same driver lock. The campaign then owns
the required ASR and durable chunk closure before the next generation job.
An authentic completed job whose terminal save was interrupted can be sealed
without a new model call; that terminal evidence is acknowledged before the
next generation job is eligible. Non-success terminals remain terminal and do
not authorize retries.

The external operator saves each exact ZIP, retains the actual provider
response, and uses `research_d2_checkpoints.py acknowledge`. A local ZIP or a
caller boolean is not persistent-save evidence. The checkpoint driver does
not upload anything itself. Each capsule contains the current chunk's native
evidence, the fixed source/runtime/model metadata, and acknowledgement metadata
linking earlier capsules. A completion capsule includes the preceding intent's
descriptor and actual provider response. Earlier chunks' audio and ZIPs are not
embedded again; restoring the complete campaign requires the saved capsule
chain.

## Recognition and labels

`export-asr` accepts only an authenticated completed queue job and keeps exact
native waveform bytes. Recognition receives audio only. The independently
maintained context v3 weak reader authenticates the plan, stage, recipe,
generation settings, native arrays and actual ASR evidence. Ordinary speech
has a nonempty full transcript when confirmed; it does not become an empty CTC
target. Keyword zeroes require observed confirmed absence. Unknown, failed
native, OOV or unsupported observations keep zero objective weights.

Generation completion, vocabulary representability and machine ASR agreement
are distinct from human gold and independent KWS quality. None is a product
qualification claim.
