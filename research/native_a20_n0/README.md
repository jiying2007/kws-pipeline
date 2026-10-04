# A20 N0 saved-evidence tools and archived source

Research-only tooling for the complete three-stream N0 deterministic-negative
record. No default build, ABI, model, threshold, training, shipping or deployment
path is changed. No model/probe/audio execution is part of these tools or CI.

## Portable saved-only validation

Use Python3.10+ and the standard library, with an offline kws-data checkout:

    python3 -B research/native_a20_n0/validate.py --data-root /path/to/kws-data/research/2026-10-04-n0-deterministic-controls
    A20_N0_DATA_ROOT=/path/to/kws-data/research/2026-10-04-n0-deterministic-controls python3 -B -m unittest discover -s research/native_a20_n0/tests -v

Validation verifies both closed manifests, the pinned data-manifest hash, exact
raw/geometry gzip decode identities, full trace coverage, original input hashes,
all179,982 logits and the complete result of the byte-identical frozen scorer.
Only temporary JSON files are materialized so the original scorer uses exact
original bytes, with no adapted raw header or identity labels. Integer geometry
is independently recomputed; no signal processing is executed. Known failure and
unknown-observation semantics are checked separately from descriptive counts.

Source publication is blocked while DATA-REFERENCE.json contains
PENDING_REVIEWED_DATA_PUBLICATION/publication_ready=false. Local staged validation
permits that explicitly gated state; --require-published and CI reject it.
After data is independently reviewed, published and fully read back, pin its
immutable40-character commit, change publication_ready to true, refresh the code
manifest, then independently review this source delta before publication.

## Historical code versus executable tools

- src/score_n0.py and src/geometry.py are byte-identical saved-only/pure-integer
  originals; neither executes model/audio/frontend/decoder code
- src/generate_inputs.py is byte-identical original deterministic generation/QA
  source and depends on geometry.py. Its historical main expects the old layout
  with GENERATOR-RECIPE.json, metadata and empty inputs/audio and inputs/pcm
  directories. This split public source checkout does not silently recreate it
- Tiny in-memory generator tests cover deterministic block boundaries, signed
  rounding and MA32 startup. They do not regenerate any300-second stream, qualify
  full input hashes, render/listen to audio or create additional experimental data
- src/collector.c and src/inputs.h describe the exact historical collector; CI
  does not compile or invoke them. Runtime/model payloads are referenced, not copied
- archive/*.txt are non-executable historical source text, not runnable entries
  or a future launch authorization. v1/v2 launch text replaces the private lock
  location with an explicit placeholder; v2 also replaces the actual private
  control-proposal identity. All admission comparisons and error gates remain
  inspectable. Original and projected hashes are both recorded in data provenance
- archive/guard_probe_v2.py.txt preserves the disabled second-probe entry
- There is no active public launcher, new supervisor implementation or new trial

The strict original probe remains FAILED_NO_RETRY. The completed v2 constructed
run used an explicitly reduced children-observability contract, preserved in data
protocol and resource projections: exact owned endpoint errno2 alone becomes
NOT_AVAILABLE/null; other errors/known children stop; mandatory live VmRSS/Threads
fail closed. No lifetime child absence or descendant-tree guarantee is inferred.

## Results and limits

Three300-second streams completed once, zero events each;3,000 callbacks,
89,994 fbank rows,29,997 model and actually searched rows,6,012 complete records.
900 seconds is input duration, not15 minutes wall-clock soak. Native CPU was
10.669888429 s and wall10.672298922 s. Supervisor10.848508173 s is run-phase only,
excluding admission; RSS is a sampled guard, not a kernel-hard RSS limit.
There were107 LIVE RSS/Threads observations, children NOT_AVAILABLE/null,
lifetime child absence NOT_PROVEN, and terminal I/O UNKNOWN. The saved-evidence
audit PASS is limited to descriptive evidence. It supplies no real FAR, FRR,
latency, board, numerical or product qualification. Probabilities were not saved.

All original-to-public source hashes, the complete scientific caveats and external
immutable references are in the separately pinned data record. The source tree
is additive to PR463 head804553286fd9caa294769cc4d44cc0c43dc66e88; inherited paths
and existing PRs are not changed.
