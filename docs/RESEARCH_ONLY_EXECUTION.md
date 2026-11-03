# KWS research-only execution boundary

The active KWS goal is independent acoustic and embedded engineering research,
not publication of a model. The performance target is both full Mandarin wake
phrases (你好小窝 and 小窝小窝), low FRR and low longstream FAR with one-sided
confidence bounds, word-end event latency, and reproducible CPU/RSS/I/O on the
SigmaStar SSC305. The standalone 小窝 remains a negative/partial phrase.

## One-variable experiments

The manual workflow dataset-driven-iteration.yml runs the same C replay and
dataset-event scorer as before. Its default historical reference model is now
selected from configs/research-baseline.xiaowo.json, never from a release
approval configuration. Every baseline SHA is compared to real Git-registry
files. An explicitly specified model+keyword pack can be compared without
changing the frozen baseline.

The old model-749187ec1d66 was exposed to synthetic testing and failed later
longstream negative evaluation. It is a *research comparator*, not measured
real-human acoustic evidence. Synthetic source receipts and source-only CI
tests cannot make a new model scientifically accepted.

## Data

The current dataset source pipeline is kws-data research_only_bulk_ingest.py,
then its canonical immutable WAV/PCM/ASR evidence and a source-family-disjoint
cohort. Use lawful data under the actual dataset terms (AISHELL-3 and other
relevant corpora); don't infer additional uses from public availability.
Do not inflate real-speaker counts with cloned descendants or move observed
development audio into a blind evaluation cohort.

## Quality gates

Training iterations require source-bound real sound, reviewed labels and
one controlled variable per paired test. Compare separately the exact two
wake phrases' FRR, independent continuous negative hours/FAR and upper 95%
confidence, and word-end-to-trigger p95. The embedded C runtime must be tested
for real-time block deadline, CPU/I/O/RSS, input continuity and target-board
behavior. These are technical measurements, not a rights or publication process.

## Historical evidence

The repository retains old model assets, provenance hashes, failed experiments
and prior workflow identities solely for reproducibility. Their original
release-oriented field names are historical evidence, not current approval
authority for new dataset research. Do not mutate retained history or declare
its frozen synthetic results a new independent real-world result.

## Frozen research training seed and keyword entry

The technical model-training entry uses
training/verify_training_entry_contract.py and the independent
configs/research-baseline.xiaowo.json. The historical held-out seed
271838 remains consumed, and the next reserved research seed is 271843.
The two four-character Mandarin phrases and their existing pinned
token/keyword geometry are still checked byte-for-byte. Old measured
validation audio cannot become a new blind holdout through a field rename.

The model-training Actions retain exact source-trained-data admission,
one-variable budget and technical readback tests. They do not consult
the legacy shipping configuration for model or seed permission.

## Retired automatic publication surfaces (2026-11-03)

The historical `release.yml` SDK publication workflow and
`deployment-release.yml` commercial-candidate publication workflow have
been removed from active GitHub Actions. They previously listened to release
tags/branches or the `deployment/commercial-candidate` push branch. Their exact
source and old run history remain preserved by Git/GitHub; none is a condition
for current reproducible data preparation, keyword experiments, FAR/FRR
measurement or resource profiling.

`model-promotion.yml` and related real-human/DUT qualification paths still
exist as legacy surfaces and require separate dependency migration under #704. Their presence does not confer research
approval or imply real-hardware evidence.

## Retired terminal shipping publication (2026-11-03)

The historical `shipping-approval.yml` and `target-cohort-promotion.yml`
no longer run as GitHub Actions. They formerly published approvals/releases;
their source and evidence remain in Git history. Cohort scoring remains in
`tools/score_target_cohort.py`, including independent multi-DUT identity,
CPU-percent semantics, resource budget, soak hours and failure classification.
Its next-step output is `independent-research-review`, never a shipping
approval. Authentic measurements, source rights and independent FAR/FRR
qualification retain their own evidence limits.
