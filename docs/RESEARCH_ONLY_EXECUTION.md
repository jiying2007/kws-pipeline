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
