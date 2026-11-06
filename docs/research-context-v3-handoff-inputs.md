# Context v3: explicitly pinned mixed reader replay

`training/research_context_v3_handoff_inputs.py` is a pure replay and full2601
projection boundary for a context v3 campaign whose earlier packages used the
original reader and whose later packages use the additive container metadata
reader. It makes no TTS, ASR, or optimizer calls. The original reader, its
fourteen source files, and `training/research_context_v3_inputs.py` remain
unchanged.

Each package names its absolute directory, exact `MANIFEST.json` SHA256, reader
revision, absolute reader source root, and complete flat relative-path to SHA256
mapping. `context-v3-r1` requires the original fourteen files.
`context-v3-metadata-r2` requires those same fourteen files plus
`automation/active_context_v3_metadata_weak_view.py`. The dispatcher does not
infer a revision from the package schema or replace retained producer hashes.

```python
rows, report = aggregate(
    [{
        "path": "/absolute/weak-view",
        "manifest_sha256": "<exact manifest SHA256>",
        "reader": {
            "revision": "context-v3-r1",
            "root": "/absolute/frozen-data-source",
            "source_files_sha256": {"<each required relative path>": "<SHA256>"},
        },
    }],
    expected_plan_sha256="<one exact context v3 plan SHA256>",
    vocabulary_path="/absolute/full2601-vocabulary.json",
    vocabulary_sha256="<exact vocabulary file SHA256>",
    original_tokens_path="/absolute/fsmn-official-tokens.txt",
    prior_sources=(),
)
```

The command line accepts the same arguments as one JSON declaration:

```sh
python3 -I -S -B training/research_context_v3_handoff_inputs.py project \
  --declaration /absolute/mixed-packages.json \
  --output /absolute/new-projection.json
```

The output is create-only and contains `rows` and `report`. The parent imports
no data reader. Every package is verified and projected in its own fresh
isolated subprocess, with the selected reader roster and the three pipeline
source files checked before and after replay. This also prevents a caller's
already loaded generation modules from silently selecting a different reader.
The retained package producer hashes must equal the declared, executed reader
hashes.

Original rows retain `kws-context-v3-fulltext-projection-v1` and their exact
original values. Metadata rows use
`kws-context-v3-metadata-fulltext-projection-v1`, retain the original source
adapter and both native container hashes, and carry the explicit
`native_container_relation`. The additive reader authenticates the underlying
files; the projection additionally requires the exact byte-identical or
one-second PEAK timestamp relation and its unchanged-signal and unchanged-gate
flags. Failed native returns have no reconciliation relation and remain zero
weight.

All packages must use one fixed 72-cell plan and the same full2601 vocabulary.
Observed full transcripts are retained. Ordinary speech has complete nonblank
targets. Uncertain, failed, out-of-vocabulary, and CTC-infeasible rows retain
their provenance with zero CTC and keyword weight. Neither intended text nor
container metadata promotes an uncertain row to a confirmed label.

The dispatcher checks global source-cell and native-PCM uniqueness and carries
family, speaker, native/derived audio, reference audio, and parent-source aliases
between reader subprocesses. Cross-split aliases fail closed. The final report
records each package's actual reader, row consumer, source adapter, observed
coverage, and reconciliation IDs. It is deterministic and reports zero new
model calls and no original-package rewrites.

## Focused validation

The standard-library cases exercise typed fulltext targets, zero-weight and
geometry boundaries, exact reader rosters and source pins, strict JSON and
paths, create-only output, and cross-reader source identity guards. The opt-in
data integration case constructs two fictional package subsets from one exact
plan, replays them in fresh processes despite a poisoned parent module cache,
compares original rows to the unchanged original projection, retains the one
explicit metadata relation, and rejects a wrong reader revision and duplicate
native PCM or cell identity. It verifies repeatable reports and unchanged input
package bytes.

```sh
KWS_CONTEXT_METADATA_DATA_ROOT=/absolute/frozen-data-source \
  python3 -I -S -B tests/test_research_context_v3_handoff_inputs.py
KWS_CONTEXT_METADATA_DATA_ROOT=/absolute/frozen-data-source \
  python3 -I -S -B -O tests/test_research_context_v3_handoff_inputs.py
KWS_CONTEXT_METADATA_DATA_ROOT=/absolute/frozen-data-source \
  python3 -I -S -B -OO tests/test_research_context_v3_handoff_inputs.py
```

The data root must include both readers and their fixture sources. Without it,
only the cross-repository integration case is skipped. These are replay and
contract tests, not model-quality measurements or permission to resume an old
pending model intent. A separately declared continuation owner must enforce
durable phase acknowledgements and the no-retry boundary.
