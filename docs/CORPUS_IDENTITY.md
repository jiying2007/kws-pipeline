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
and `speech-like-audio-review-v1` receipts. It does not synthesize audio, issue
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

The command binds every receipt to the exact WAV hash, source, intended text,
positive/negative kind and keyword ID. Missing, extra, duplicate, rejected or
uncertain receipts fail. Actual WAV/PCM hashes and mono 16-kHz PCM16 format are
checked using the existing corpus identity helper. Empty/truncated files,
duplicate source IDs or decoded PCM, invalid labels, and cross-split voice reuse
fail. Stdout contains the input hashes, split counts and a research-only report;
failures return nonzero without materializing a corpus. The existing `materialize`
command can use `--audio-review` to apply the same gate before writing and retain
its admission result in the summary. Without that option its existing contract
is unchanged; read-only admission does not invoke materialization.

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
The generic validator reports split counts; it does not reconstruct or independently
prove the historical 12/4/4 split specification.

Receipt binding is not authentication of a reviewer or proof that listening took
place. Provider identity hashes alone do not establish distinct generator families,
weight provenance, source rights, or licensing. Already-observed clips remain
research/development regression data; even a legacy `qualification` split name
does not authorize product qualification. The exact receipt validator is selectively
ported from `c4025ee2e686c0c6cfd1ba078e6d1dabc3f70ee8`; the surrounding old research
pipeline is not imported. Tests use only clearly marked synthetic fixtures.
