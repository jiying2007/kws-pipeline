# Research and retention status, 2026-10-09

Observed at 2026-10-09T02:43:25Z. This is a dated status snapshot, not a live
PR/branch monitor. [中文](CURRENT_STATUS_2026-10-09.zh-CN.md).

## Publication and navigation

- [Data PR #27](https://github.com/jiying2007/kws-data/pull/27) is **draft,
  open and unmerged** at this observation. The full public host-research archive
  contains **1,886 logical members / 1,403 unique objects**, in eight parts.
- [Pipeline PR #501](https://github.com/jiying2007/kws-pipeline/pull/501) is
  **draft, open and unmerged**. The expanded 500-file pipeline source copy remains
  **NOT_PUSHED**, blocked on publication tooling. Archive availability does not
  mean these expanded source files have landed in pipeline main.
- The [publication index](source-retention-publication-2026-10-09.json) records
  immutable source identities, hashes and exact-head CI snapshots. Its catalog
  identity may advance independently of the original archive/part commit.
- CI runs `tools/verify_research_publication.py --fetch-metadata`, explicitly
  retrieving only pinned `ARCHIVE.json` and `CATALOG.json` (combined maximum
  2 MiB), checking both SHA-256 and Git blob identity, then cross-checking counts,
  contiguous parts, package/CAS mappings and totals. Offline use requires these
  exact files via `--metadata-dir`. No objects, weights or audio are downloaded.
  Complete object payload verification belongs to data's publication CI.
- CI workflow IDs, paths, event, run attempt and head are checked as recorded
  snapshot identities. Offline validation does not prove current remote PR
  state, authenticate a newly edited CI claim, or establish acoustic quality.

## Research outcomes, unchanged by publication

| Entry | Preserved outcome | Boundary |
|---|---|---|
| D20/D90 official-reference v2 | Two raw-logit coordinates exceeded the gate in the same D20 failed fixture; D90 was not run | No successful official numerical gate or model comparison |
| Fixed50 label support | `EVAL_INCONCLUSIVE_LABEL_SUPPORT`; K2: 4 clips / 4 voices, below 5; targets: 11, below 12; 8 unknown | No admissible KWS comparison or accuracy claim |
| Old frozen-model nightly | [Run 37860130019](https://github.com/jiying2007/kws-pipeline/actions/runs/37860130019): 2 false accepts / 8 hours, upper95 0.786974 FA/h | Violates 0 false accepts / upper95 ≤ 0.40; not a D20/D90 result |
| Human and physical-board qualification | Deferred | No new human listening task or device experiment is authorized here |

The nightly model SHA-256 is
`ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402`.
It is the old frozen model, not a D20/D90 candidate. Historical engineering
qualification elsewhere in the README must not be read as a pass of this later
nightly. Shipping approval remains false.

## Completed branch cleanup versus historical retention

The 2026-10-07 inventory of 61 non-main pipeline branch names is a historical
snapshot. It is not the current live branch list or a claim that all remain.
The separately approved cleanup deleted 57 pipeline branches and 22 data
branches. The two recovery tags `archive/branches-2026-10-07` and
`archive/branches-2026-10-07-prune-anchor` preserve recovery anchors.
See the [completed cleanup record](git-atomic-prune-2026-10-08.json).
Original failure records and older retention snapshots are preserved unchanged;
this status page supersedes outdated current-tense navigation text only.
