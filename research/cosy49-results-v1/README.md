# Cosy49 fixed300 saved research result

This isolated research source addition accompanies the exact Cosy49 terminal checkpoint, native FP32 payload and saved scientific evidence in `kws-data/research/2026-10-05-cosy49-candidate`. The existing training source remains pinned to [PR473 head 28e90204e40b8da07f5d887fee43f3e19a3e865d](https://github.com/jiying2007/kws-pipeline/tree/28e90204e40b8da07f5d887fee43f3e19a3e865d/research/cosy49). No active training release or workflow is copied here.

The single 49-source, 300-update training attempt, zero-model export, matched build, eight exposed non-acoustic reference/native fixtures and one original-to-candidate 98-source endpoint pair are completed historical work. `historical/` preserves 84 exact whole historical source/configuration files used for numeric export/evaluation and endpoint acquisition/scoring, plus the eight exact supervision helper definitions selected by the historical endpoint caller. `SOURCE-PROVENANCE.json` binds original and public hashes, including endpoint runtime sources represented once under the numerical vendor tree. Historical preparation status fields remain historical; current outcome is in the data archive's `SUMMARY.json`.

This is a bounded execution-source archive, not the complete historical source workspace. The endpoint helper file is an exact-definition subset, not the original whole module: unused admission and launch code is omitted. Six numerical preparation/assembly or historical test files and the mixed98 input header are explicitly listed as absent in the source provenance. The source is preserved for inspection and controlled reconstruction, not turnkey replay. Private execution/admission records, machine-specific runtime inventory and the mixed98 natural-input table are deliberately absent. The full98 run cannot be replayed from this public package alone. Original weights and training features remain immutable dependencies. The exact previously exposed 120,216-byte numerical fixture archive is included; its earlier public-package presence is not assumed. Stored model scientific values and numeric source functions have not been edited. Generated binaries are represented by their saved identities, not redistributed.

## Restore saved bytes only

From a checkout of the data archive and this source:

```sh
python3 -B research/cosy49-results-v1/verify_archive.py --data-dir /path/to/kws-data/research/2026-10-05-cosy49-candidate
python3 -B research/cosy49-results-v1/verify_archive.py --data-dir /path/to/kws-data/research/2026-10-05-cosy49-candidate --output /path/to/new-cosy49-output
python3 -B -m unittest discover -s research/cosy49-results-v1/tests -v
```

The first command only checks bytes. Restoration requires a nonexistent output directory and an existing parent. The verifier uses only Python's standard library, rejects unsafe/duplicate paths, symlinks, extra/missing/altered chunks and member drift, and never imports model code, Torch or NumPy. It does not call a model, frontend, decoder, benchmark, compiler or trainer. Historic tests under `historical/` are source evidence; the command above runs only the new invented transport tests.

## Interpretation

The outcome is `FIT_ENDPOINT_GAIN` on already exposed fit-domain sources. C's Eric K1 miss is recovered; H/N and the old Qwen repeat-nihao incorrect K1 events disappear; A/B/O correct events remain. Two human-reviewed old Qwen development misses remain: Serena K1 and Eric K2. D20 has 11 weak wake sources all hit and nine weak nonwake sources with no events; those nine are not misses. Cosy12's two ASR-weak wake sources both remain hits.

Y's removed previously UNKNOWN K1 event is conditionally a false-event removal under primary hearing `你好小屋`. The alternative `你好小窝` implies a possible miss. Subjective 70:30 hearing uncertainty remains uncalibrated, non-gold and ineligible for automatic training. The original frozen `REVIEW_REQUIRED_UNADJUDICATED_CHANGES` decision is preserved.

Of 28 retained event matches, none is earlier, three are later and 25 have unchanged input availability. Qwen Vivian K2 is +300 ms, old human-reviewed Cosy K1 +180 ms and new A +120 ms. These are input-availability costs, not measured word-tail latency. Reaped process CPU is 19.114880 versus 19.158394 seconds, launch-to-reap wall 19.403674133000095 versus 19.48244427999998 seconds, n=1 per arm. Sampled RSS is 4,079,616 bytes for both. Separate lifetime maxrss uses a different process-history scope; no performance or memory optimization is claimed.

Numeric conformance passed all 1,334 fixed checks across 115 reference and 230 native model rows on eight exposed non-acoustic fixtures. Earlier strict official-FP32 failures remain failures. The mathematical reference is the existing high-precision PCM/Decimal80 contract; this is not broad official-FP32 equivalence, fresh acoustic holdout, real FAR/FRR, board or shipping qualification. No six-item blind-review inputs or results are included or inspected during this archive preparation.

## Publication boundaries and provenance

The data projection includes complete unchanged raw lines for 77 synthetic/deterministic sources, excludes all per-source raw/derived records from 21 natural sources and retains only a brief aggregate natural comparison. Natural audio, private paths/hostnames, Library locators, owner conversations and approval records are excluded. No WAV is newly published. Source-identifiable speech-derived data is not anonymous or promised non-invertible.

Research distribution only. Existing Apache-2.0 source/model and BSD-2-Clause frontend provenance remains in force for the components it covers. It does not establish generated voice/output rights or commercial clearance. `commercial_output_license=not-established`; no blanket relicense, endorsement or new license grant. See the exact training rights/provenance record and immutable dependencies in the data archive.

This branch is based on pipeline main `55a4e23379ad7072db507dbe419b38d8898e17fa`. It uses `research/cosy49-results-v1/**`; no default, ABI, catalog or model selection changes. No workflow is added, and no merge is requested.
