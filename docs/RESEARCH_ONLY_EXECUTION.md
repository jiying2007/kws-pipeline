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

The former `real-human-qualification.yml` and `target-dut-qualification.yml`
were also retired from active Actions on 2026-11-04 because each could publish
GitHub Releases based on commercial deployment/qualification tuples. Their
original source remains in immutable Git/Actions history. Existing strict
private-corpus intake, final AFE, actual WAV scoring, independent FRR/FAR
confidence bounds, physical CPU/RSS/power/soak collectors and negative tests
remain in `tools/` and `tests/`. These standalone scientific tools are **not**
a new self-hosted research workflow; missing human/DUT evidence is NOT_RUN,
never PASS. Future physical experiments must independently bind rights, frozen
model identity, actual hardware and source hashes without a Release gate. The
former `model-promotion.yml` is also retired from active Actions.

## Retired terminal shipping publication (2026-11-03)

The historical `shipping-approval.yml` and `target-cohort-promotion.yml`
no longer run as GitHub Actions. They formerly published approvals/releases;
their source and evidence remain in Git history. Cohort scoring remains in
`tools/score_target_cohort.py`, including independent multi-DUT identity,
CPU-percent semantics, resource budget, soak hours and failure classification.
Its next-step output is `independent-research-review`, never a shipping
approval. Authentic measurements, source rights and independent FAR/FRR
qualification retain their own evidence limits.

## Retired model publication workflow (2026-11-03)

`model-promotion.yml` no longer publishes a model release or submits model
registry PRs. The historical trained model/keyword bytes, consumed seeds,
frozen comparators and Git-model registry evidence remain immutable. Current
`tests/test_research_model_provenance.py` checks the same training
source/asset/budget/numerical constraints without executing a publisher;
source and rights checks remain required for actual training.

## Frozen negative FAR research

The scheduled `far-nightly.yml` now uses the immutable Git research
comparator and `configs/research-baseline.xiaowo.json`, not a product
Release or the legacy shipping configuration. See
[research FAR regression](RESEARCH_FAR_REGRESSION.md). Historical false accepts
remain failed evidence, not a product suitability claim.

## Retired orphaned release-tag helper

`tools/verify_release_tag.py` was not invoked by current research Actions or
model/data experiments. The old tagged commercial bootstrap helper is retired
from the active tree; its Git source remains historical and may be inspected,
but does not determine current research admission or model provenance.

## Pinned local research training base (migration in progress)

The historical training-base source may be supplied explicitly as
`KWS_RESEARCH_BASE_ARCHIVE=/absolute/path/to/verified-archive.tar.gz`
when running `training/materialize_governed_product_base.sh`. Before any
generated output is deleted, the script verifies the exact frozen archive
SHA256 from `configs/training/product-speech-like-base-v1.json`, rejects
missing or mismatched files, and refuses a source under generated output.
The local path performs **no GitHub Release lookup or download**. It still
runs the existing provider, bundle, config and training-entry checks.

This option is not evidence that the archive is currently accessible in
CI or that its dataset rights have been re-reviewed. Current Actions do
not set `KWS_RESEARCH_BASE_ARCHIVE`; their old fallback remains a
**known live Release dependency** tracked by [#733](https://github.com/jiying2007/kws-pipeline/issues/733).
Do not claim full research-only terminal completion until a real
rights-reviewed source is available to every live training workflow,
the fallback is removed, and all exact-head and fresh-main gates pass.

## Retired GitHub Release-backed base recovery (2026-11-04)

The former `live-base-binding` job in `product-training-data-contract.yml`
used a published GitHub Release as a **live** input to historical model-base
materialization. That job is removed from current research CI. Scientific
source/role checks, deterministic training reproducibility replicas, candidate
encoder numeric fixtures, GCC/Clang/ARM portability and provenance checks
remain. Original SHA256 receipts and historical source-release metadata stay
in Git and Actions history, not current experiment admission. A new base
requires separately authorized and source-bound dataset research, never a
product Release.

## Active workflow publication guard (2026-11-04)

The required offline `tests/test_research_publication_boundary.py` now
reviews **every active** `.github/workflows/*.yml` and rejects GitHub Release
writers/downloaders, Release/tag-push entrypoints and release-publishing Actions.
The negative fixtures fail on a resurrected publisher, while ordinary
GCC/Clang, ARM, sanitizers, KWS acoustic scoring, TTS/ASR source contracts,
research training and deterministic data preparation remain unchanged.
Historical approvals, old Git/Actions records and archived source data are
provenance only; they cannot become research model admission or physical
SSC305/human evidence by renaming a file.
