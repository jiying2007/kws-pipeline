# Corpus identity contract

Model and qualification provenance must bind the actual audio bytes, not only the manifest files that name those bytes.

For each mono 16-kHz PCM16 WAV retain:

- source file SHA256;
- decoded PCM SHA256;
- frame count and duration;
- stable recording/path identity;
- speaker/session/source/room/device metadata when available.

The canonical corpus digest is computed over the ordered recording identity records. A renamed or rewrapped file with identical decoded PCM is therefore visible as the same acoustic payload while still retaining distinct source-file identity.

Training checkpoints must capture the corpus identity at training time. Qualification execution must capture the exact WAV identities that produced detections. The final qualification manifest must verify and cross-link both chains.

The final held-out qualification corpus must remain independent from hard-negative and false-reject replay sources.

## Restored human-reviewed synthetic batches

`tools/speech_like_corpus_plan.py admit-reviewed` is a read-only admission check
for the existing canonical request intents, generated provider JSONL manifests,
and supplied `speech-like-audio-review-v2` receipt histories. It does not synthesize audio, issue
review receipts, or convert an archive into a new corpus format.

```sh
python3 tools/speech_like_corpus_plan.py admit-reviewed \
  --intents /controlled/restored/intents.jsonl \
  --generated-root /controlled/restored/generated \
  --audio-review /controlled/restored/audio-review.jsonl \
  --expected-count 20
```

These paths illustrate the canonical inputs, not verified filenames inside the
unavailable Qwen archive. Resolve actual files only after the archive is restored.
Only provider groups named by the intents are required for this read-only command;
within each group it reads `manifest.jsonl`, using existing manifest-relative or
absolute audio path semantics. Rebinding obsolete container paths must preserve
WAV bytes, source IDs, texts, labels and original receipts; retain the original
manifest and record the revised manifest hash separately.

The command binds the latest accepted human-origin revision to the exact source,
WAV and decoded PCM, actual transcript/tokens, acoustic completeness, allowed
purpose, event kind and keyword ID. Full revision chains are retained and the
latest revision wins, including a later rejection. The bounded Mandarin mapper
checks actual text against token labels; unsupported/OOV speech is rejected,
never mapped to blank. A rejected OOV observation may remain in revision history
and be corrected by a later accepted review.

Actual WAV/PCM hashes and mono 16-kHz PCM16 format are recomputed. Empty/truncated
files, duplicate sources/PCM, invalid labels, and cross-split voices or supplied
source-family/speaker/reference/session/derivation identities fail. Stdout contains
input hashes and an internal-development report; failures return nonzero without
materializing a corpus. `materialize` now requires `--audio-review` by default.
Only an explicit `--admission-mode synthetic-fixture-v1` permits test fixtures
without listening receipts; those outputs are marked ineligible for CTC training.
See [speech base admission](SPEECH_BASE_ADMISSION.md) for the three separate lanes.

For the historical Qwen20 candidate, first compare the restored archive's SHA-256
with `ed7ccc918422cd2a241728dee853af57bf3b3705818c7eaaba19bc1e231cf196`
and the original review receipt with
`affa420dba5b9b84b000772d8d5d7fbd0788caf3b90fdcd679e97f519ff20613`.
The [historical archive record](https://github.com/jiying2007/kws-pipeline/blob/c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8/docs/research/KWS_DATASET_ARCHIVE_ROUTE_2026-09-28.md)
reports fixed-batch human acceptance. This tool does not revoke that acceptance
when files are unavailable, nor extend it to future generated samples. Archive
bytes and schema have not been recovered or tested here; do not invent missing
intents, receipt fields, or a successful real-batch admission. If the archive lacks
canonical planning inputs, recover their original controlled source before use.
A historical v1 receipt remains historical evidence, not a v2 actual-transcript
admission. Do not convert its intended label into an actual label or invent a
review revision. New training admission requires genuine supplied v2 review.
The generic validator reports split counts; it does not reconstruct or independently
prove the historical 12/4/4 split specification.

Receipt binding is not authentication of a reviewer or proof that listening took
place. Provider identity hashes alone do not establish distinct generator families,
weight provenance, source rights, or licensing. Already-observed clips remain
research/development regression data; even a legacy `qualification` split name
does not authorize product qualification. The former v1 intent-only receipt contract is superseded for new training. Tests
use only clearly marked synthetic schema fixtures, not operational human receipts.
