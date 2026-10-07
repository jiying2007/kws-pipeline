# Qwen20 saved-evidence tools and archival execution source

This isolated source-only research directory accompanies the numeric Qwen20 A20
record in `jiying2007/kws-data:research/2026-10-04-qwen20-descriptive`.
DATA-REFERENCE.json pins its immutable commit and manifest. No raw logits, model
weights or audio payloads are published in this source repository by this delta.
No shipping build, model registry, product ABI or numeric runtime is changed.

Historical outcome: **FAILED_NO_RETRY**, despite collector return code 0 and a
complete trace. Supervisor PermissionError at `/proc/7/io` remains FAIL; final
I/O and children observations remain UNKNOWN. Saved-trace completeness and
counts passed audit. Observed K1/K2 matching clips were 4/5 each, with one K1
activation among ten near-foil clips. These historically exposed synthetic clips
provide no independent FRR, FA/hour, true-tail, board or numerical qualification.

## Offline checks

With the data repository checked out at DATA-REFERENCE.json's immutable commit,
using Python 3.10+ on Linux:

```sh
export A20_QWEN20_DATA_ROOT=../kws-data/research/2026-10-04-qwen20-descriptive
python3 -B research/native_a20_qwen20/validate.py --data-root "$A20_QWEN20_DATA_ROOT"
python3 -B -m unittest discover -s research/native_a20_qwen20/tests -v
python3 -B -m unittest discover -s research/native_a20_qwen20/future/tests -v
python3 -B -m unittest discover -s research/native_a20_qwen20/future/review -v
```

Only standard-library metadata, invented and saved-scalar checks run. No dependency
installation, model/audio/native-library loading, collector process, softmax,
beam replay or live /proc probe occurs. The workflow checks out the pinned data
repository but performs no model/download installation. The actual data evidence
and complete diagnostic discussion live in the corresponding data record.

`validate.py` checks both complete file inventories, rejects missing/extra/unsafe
members and bounded gzip corruption, verifies the exact unchanged raw hash and
original binding header, and recomputes every descriptive result. The original
scorer remains byte-exact. Since public input metadata replaces reviewer receipts
by immutable annotation references, the adapter changes only the manifest-hash
label in a separate in-memory raw view. It compares the full derived scientific
report under the original identity references. This never alters raw evidence
or claims the new projection was acoustically run. The omitted original metadata
hash is referenced rather than independently recomputed by the default adapter.

`src/derive_saved.py` adapts the original saved-only diagnostic to pure functions.
It verifies complete all-20 callback/argmax/run arrays, focus scalar comparisons
and all nine event-score ranks. Greedy strings are not native beam paths or a
wake requirement. No probability vector or counterfactual threshold outcome is
reconstructed. `CODE-MANIFEST.json` inventories source files except itself; the
reviewed containing Git commit is its trust anchor.

## Exact historical execution source

- `src/collector.c`, `src/inputs.h`, `src/score_descriptive.py` and the original
  scorer tests are byte-identical to the frozen experiment
- `archive/launch_once.py.txt` and `archive/build_collector.sh.txt` preserve exact
  original source/commands as non-executable archival text
- Actual execution releases, private control records, machine identities and
  reviewer identifiers are excluded. Historical schema keys are source only,
  without usable release values

The collector originally used `runtime/` for the native source and library,
`inputs/a20.f32` for weights and `inputs/pcm/` for the manifest-pinned raw PCM.
The data record's immutable external references provide original source, WAV/PCM,
model and measured binary hashes. They are reused, not duplicated. Rebuilding
source is distinct from reproducing the original compiler/system image or binary;
no new collector build or acoustic execution is claimed by these saved-only tests.
Never delete/reset the historical attempt ledger or rerun that attempt.

## Future-only supervisor candidate

The exact `future/` candidate, patch and tests preserve a separately reviewed
engineering experiment. `run()` unconditionally raises before release reads,
locks, ledger/output creation or Popen. Its unreachable integration body is
historical review material, not a release procedure.

The candidate polls first; separates optional I/O UNKNOWN/UNAVAILABLE from
observed zero; fails closed on unavailable live RSS/thread/children; preserves
known resource crossings even at exit; and bounds final evidence output. Its
26 author and seven independent mocks are focused tests only, not live host,
kernel enforcement, acoustic performance or release readiness.

Known integration limitation: inherited outer exception cleanup and general
filesystem/write failures lack live validation. A kill/exit race there can
interrupt final evidence writing; the once-only reservation still prevents reuse.
Sampling may miss transient changes and peaks. Relative lock paths assume the
original nested layout and must be reviewed after relocation. A new protocol,
exact executing-source freeze and independent review are needed before future
integration. This code never retroactively passes the old FAILED_NO_RETRY run.

## License

Project-authored research source is covered by the repository's existing
Apache-2.0 license. The referenced runtime retains its existing Apache-2.0 and
BSD-2-Clause attribution at the [immutable native NOTICE](https://github.com/jiying2007/kws-pipeline/blob/804553286fd9caa294769cc4d44cc0c43dc66e88/research/native_a20/NOTICE.md).
No third-party runtime implementation, audio or model payload is copied here.
No additional commercial rights to synthetic audio or model outputs are granted.

### Optional exact original input-metadata reconstruction

Supply the already-public `datasets/qwen3-reviewed-v1/audio-review.jsonl` from
input commit `d9a1f3171bceae45908bc8ac84045d3bca23f43f` with the validator's
`--annotations` option. The tool verifies its exact file and per-line hashes,
restores the omitted receipt objects in memory, and verifies the reconstructed
original manifest SHA256 `75b322f33857152bc59d4700cba9b85dd818a36408d0e10a4fe837926b78ed58`.
It writes nothing and still performs no acoustic execution. The annotation file
is deliberately referenced rather than duplicated here.
