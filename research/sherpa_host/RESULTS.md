# B host engineering validation

Recorded 2026-10-01. These are compact summaries of separately executed,
reviewed host checks, not new model runs caused by this repository patch.
The following exact binaries were built with GCC/G++ 14.2.0, CMake 3.31.10,
GNU Make 4.4 and GNU ld 2.44:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| CLI | 51896 | fbb7683a4c9df9355d3f840967b39475abfda1133f1b5a9131b433ce56760c1d |
| B subset library | 1348240 | 167e160da440ea4ac2e834b43b2021fe2bfbaa9333b7fc0e95d96f60a31e379d |
| ORT 1.28.2 | 27026609 | 4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e |

- Static build/product review passed: exact subset exports, relative loader
  paths, corresponding sources and diagnostic path checks
- Actual ABI/loading and relocation checks passed for the fixed products
- All 145 same-Viterbi streams and their 101 event records matched the frozen
  reference; no reference model rerun or algorithm change was made
- Three real CLI cases passed: positive WAV (one event, including all float32
  time values), near-intent WAV (zero events), and a five-sample raw tail with
  dash-leading filename, caller-relative path and relocated kit with spaces

These gates cover engineering preservation and the stated CLI cases. They do
not reverse the permanent historical strict-official-parity failure, establish
general quality/false-accept rates, prove improved near-phonetic rejection,
measure acoustic latency, or qualify SSC305. Minimum B/ORT ISA is unknown.

Independent review identities (SHA256, retained audit records; raw private
paths/process records and audio are intentionally outside this source patch):

- Static product review: 543493e100168ea4be64a55c6c1bca519e4a03e266b7c4c275e966f3b3f51de7
- Real CLI post-run review: ec6736c773021f21f05c7ef32ac9be0a10d421b0397074a8cff00993801ed557
- Original build result: 1ad6a5d54af72fe7d3420e021ee76a6c090e4d5e185ed2c7fd18be3235f48aa9

The repository change additionally reconstructed all 491 selected production
files from local pinned inputs and checked equality with the tested source.
Default CI is intentionally model-free; its checks are described in README.
New Viterbi/Hypothesis/context-graph execution is not part of the offline job.
Local repository checks and remaining environment limits are reported with the
candidate review; remote CI has not run for this unpushed preparation.
