# Native Zipformer observations after a consumed generation call

The explicit `kws-d1-zipformer-extension-job-v3` recipe uses the completed
observed-row preparation and an actual `d1-accounted-prior-evidence-v2` closure.
The v2 source snapshot and its original evidence remain unchanged. Publishing
this source does not imply that new generation, recognition, preparation or
native inference has completed.

## Accounted coverage

A 256-cell plan has three disjoint sets: observed legal audio, returned audio
that cannot form a legal input, and consumed calls with no observable return.
The union must equal the complete frozen plan. There may be no pending cell.
All original 64 native inputs must remain observable and byte/label identical.
The original plan role and family must agree for every member of every set.

The new native entrypoint first replays every actual weak view and projected
row through the exact data source retained by the completed preparation. It
checks all retained data-source file hashes before invoking that loader's
`verify_accounted_prior(path, expected_sha256, expected_count=256)`. That
stdlib verifier authenticates the original timed-out job, its claimed calls,
returned native evidence, the consumed call's absence of a return, completed
continuation owners and actual recognition evidence. The native coverage
adapter then independently checks the plan identity, prepared parent view
pins, exact observed roster, disjoint union, family/role assignment and old64
inclusion. A summary boolean or a copied count cannot replace this replay.

The helper's immutable evidence files are hashed again and copied without
changing their bytes. The coverage object and `unobservable.jsonl` keep each
original proof and reason. A consumed unknown receives no PCM, transcript,
blank label, negative label, weak-view row or native call. Its coverage entry
has no keyword target; known masks are false and weights zero. Generation
intent is not an observation. An invalid returned waveform remains a distinct
returned exclusion. A quiet or clipped but legal waveform stays observable,
including its source-failure masks when applicable.

| Output field | Meaning |
| --- | --- |
| `planned_cells` | Original 256 planned identities |
| `accounted_cells` | Exact union of observed, returned exclusions and consumed unknowns |
| `returned_generation_cells` | Observed legal audio plus explicit returned exclusions |
| `raw_observed_clips` | Actual retained legal PCM rows |
| `excluded_returned_cells` | Real returned outputs that cannot form legal inputs |
| `consumed_unknown_cells` | Occupied calls with no observable return |
| `unobservable_cells` | Returned exclusions plus consumed unknowns |
| `accounted_complete` | True only after all 256 cells belong to exactly one authenticated set |
| `complete_terminal_coverage` | False whenever a consumed unknown remains |

Thus 255 observed rows plus one consumed unknown can account for all 256 cells
while retaining only 255 returned audio observations. It permits exactly 191
new native calls and 64 reused calls. It cannot be described as 256 completed
generation terminals or 256 returned waveforms. Further returned exclusions
reduce new native calls; they do not change the consumed unknown count.

## Execution and comparison remain fixed

The prior recipe bindings, all 25 old source pins, fixed native profile/model
assets, strict old queue reader and original64 reuse checks remain required.
The new preparation uses the explicit partial-observation training input shim;
its remaining training, evaluation and source contracts are unchanged. A fresh
isolated old-source subprocess validates original native results and stays in
the parent's process group and deadline. No source identity function is
replaced to make old evidence pass a new controller.

Only the actual observed set difference receives new create-only claims.
The job has a 120-second maximum and uses the shared heavy-model lease.
The offline environment and native network guard are installed before model
imports. The existing decoder, PCM conversion, thresholds and lack of padding
remain fixed. No new observable rows means no engine or NumPy import.

The new result and observation schemas are explicitly v3. The original
observation objects, claims, native identities and queue timing remain old
measurements inside their wrappers. Summaries separate the original64 and
additional observations, including the original 16 development rows and the
additional observed development rows. Unknown and masked examples contribute
no negative-label or qualification denominator. These are machine-weak
engineering diagnostics, with no human event-FAR or device qualification.

`audit_finished` repeats the real source, coverage, observation, file and
summary checks without a model call. The stdlib tests use clearly fictional
proof fixtures to verify the adapter and exact union, 255-to-191 call count,
unknown/returned-exclusion distinction, original64 protection and source
verifier failure propagation. These fixtures do not authenticate production
partial-job evidence; production acceptance additionally requires the actual
source-pinned data replay and completed preparation.
