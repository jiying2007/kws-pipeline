> Current status, 2026-10-11: the first and second campaign attempts have both
> finished with setup failures and zero audio attempts. Their retained freezes,
> releases, evidence, and run identities are historical and must not be refreshed
> or reactivated. The shared setup now contains dormant dependency-first source
> changes, so the old freezes intentionally reject it as source drift. The
> preparation/activation instructions below describe historical preparation only.
> Do not run either old `prepare_freeze.py` to refresh those records.
> The separately authorized third attempt has not run: its official-pip transport
> remains blocked pending equivalent acquisition bounds. See
> [the current transport review](attempt-3/PIP_TRANSPORT_REVIEW.md).

# Qwen16 Whisper small third-route screen

This is a dormant, separately reviewed, one-shot experiment. Nothing here is an
active workflow and `execution-release.json` is unapproved. Only the parent
review/publish step may activate the exact frozen template. No model was installed,
loaded, or run during source preparation or static tests.

## Fixed scope before execution

- Exactly the existing 16 full-source, mono 16-kHz PCM16 WAVs from run
  `38018029787`, with its original content-hash order and blind job bytes.
  `sense_continuation.verify_retained` authenticates the complete public retained
  artifacts and producer mapping on the host. Only the anonymous job and WAVs
  enter inference. No new synthesis, training, Qwen, or SenseVoice calls.
- Exactly official multilingual `openai/whisper` small, native `small.pt`.
  Source commit `86098128c0b4f24f0e2aa2994de830614b474227` is byte-locked;
  checkpoint expected SHA256 is
  `9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`.
  This is a published expected checksum, not a claim that the checkpoint has been
  acquired during preparation. The admitted hosted setup must verify actual
  downloaded bytes before loading. Official source/checkpoint license is MIT;
  other distributors or conversions are not substituted.
- CPU float32, batch one, four Torch intra-op threads and one inter-op thread.
  Native short-form `DecodingTask.run`, language `zh`, transcribe, temperature
  zero, `beam_size/best_of/patience/length_penalty/sample_len` null, no prompt or
  prefix, native `-1` suppression and initial blank suppression, no timestamps,
  `max_initial_timestamp` null, fp16 false. All options are in native-v2.
- No fallback, alternate model, window retry, language retry, hotword, correction,
  text normalization, transcript selection, or result-conditioned model choice.
- Check `0 < N <= 480000` first, then zero-pad only to 480000. Never trim audio.
  Native log-mel frontend is used once per clip. Before/after float32 PCM hashes
  are independently reconstructed by the host from original PCM16 bytes.

## Isolation, resources, and durable evidence

The unchanged source-screen32-v1 hosted container constructor/supervisor is
reused, replacing only its fixed Python entrypoint. Its historical internal stage
name `sensevoice` is used solely as the existing ASR supervision profile; the new
command is explicitly `source-screen32-whisper-v1/worker.py`, and no SenseVoice
code/model is staged or called. The receipt records the new Whisper stage name.

The new workflow is limited to a first creation of one named branch, public
`jiying2007/kws-pipeline`, standard free `ubuntu-24.04`, run number/attempt one.
It has no manual dispatch, paid runner, GPU, repository writes, cache reuse, or
credentials persisted by checkout. Job timeout is 45 minutes; host-wide work is
capped at 40 minutes from host entry, load/setup at 20 minutes each, each clip at five minutes,
and the first failure stops further clips without retry. The old resource guard
checks current host headroom and actual cgroup-v2 12-GiB RAM, no swap, four CPU,
256 PID scope, non-root, no capabilities, no-new-privileges, read-only source and
runtime. Inference has network=none and sees only anonymous audio inputs.

All fixed dependency wheels are downloaded once with exact SHA256/size pins and
installed offline without dependency resolution or source builds. Official
Whisper source is imported directly; the CPU-only path intentionally does not
install Triton. This is grounded in the locked `timing.py` lazy CUDA imports, and
actual Torch CUDA absence plus model CPU devices are checked. Setup records all
installed runtime bytes, actual checkpoint bytes, tokenizer/frontend/source
bytes and their locks. Only public source/tokenizer/frontend assets, JSON audit
records, and native raw results enter the output artifact; no checkpoint,
credential, private file, runtime directory, or arbitrary runtime log is uploaded.

The requested policy/model/source are frozen before the run. Effective native
values are measured from the loaded model, native DecodingTask, actual tokenizer
and suppression filters and durably frozen before the first audio forward.
This timing is explicit: they are not claimed to have been measured before load.
The exact config is rechecked for every call.

Every attempt writes a started marker before recognition, durable greedy update
records, pre-finalize tokens, raw public native output, and a completion marker
only after validated natural EOT. The official finalize method can append EOT,
so public `DecodingResult` alone is never used as a natural-stop proof. The host
independently reconstructs stop reason from the complete update sequence and
checks PCM, start/config/raw/completion hashes. Sample/context limits retain real
raw output as incomplete. A timeout retains available markers/steps and the
remaining denominator stays not-run. Partial evidence does not become complete.
An external infrastructure failure that prevents artifact upload remains missing
evidence; it is not reported as a successful run.

## Interpretation

This third model is selected after the historical two-route screen. It is an
exploratory cross-run source-screen comparison, not a retrospectively preregistered
three-model roster. Prior results and the old failed SenseVoice attempt remain
unchanged. `source_role=source_screen_only`, `training_admitted=false`,
`shipping_approved=false`, `human_gold=false` throughout. Complete input processing,
natural decoder completion, and acoustic verbatim completeness are distinct;
the last stays unknown. Human listening approval is still separate.

## Static checks and activation handoff

Run `python3 -B research/source-screen32-whisper-v1/test_contract.py` in normal,
`-O`, and `-OO` modes. Tests are standard-library only and do not install or import
Torch/Whisper. The source freeze records the full host dependency closure and
literal historical input evidence. `prepare_freeze.py` only prepares metadata and
keeps approval false; it never downloads, installs, invokes a model, publishes,
or activates a workflow.

After independent review, the explicitly authorized parent can make a separate
activation commit that sets approval true for the exact reviewed freeze hash and
copies the byte-identical workflow template to
`.github/workflows/source-screen32-whisper-third-v1.yml` on the one-shot branch.
Creating that branch is the execution boundary. The prepare commit itself does
not authorize or trigger dispatch. There is no automatic retry after a failed run.

## First attempt retained; setup diagnostics revision

The first authorized run, [38105471896](https://github.com/jiying2007/kws-pipeline/actions/runs/38105471896),
ended during setup. Its exact seven artifact JSON members and a separate provenance
record are retained under [evidence/run-38105471896](evidence/run-38105471896/provenance.json).
The original ZIP was 11,518 bytes with SHA256
`04b6b23c85f25e0b29520972654c2d59f18f22506cf1b77c9fd1056bd357b0ff`;
executed head was `48a8135fc00489075ffeb456bb51aea6afef3cdd`, tree
`19ca10175c32b377ceefdff865219aef68811d8f`. The old receipt retains its original
`fdfb8a9040a10ec380ac7a658662d2d2935296301ee4269ac0ce79309cc48c2e`
source-freeze identity unchanged. All 16 rows are not-run with zero attempts.

The container exited 1, was not OOM-killed, and cleanup completed. Its setup
artifact contained only the two copied locks, without a download or setup
receipt. The host's `ValueError` came from its supervisor detecting the failed
container; it does not identify the setup process's inner exception. Exact
failure item, cause, and partial download byte count are unknown. The old host
started and inspected the container without attaching stdout or collecting its
log, then removed it; the old setup process printed only an exception class and
had no durable failure receipt. That explains the diagnostic gap, not the
underlying acquisition failure.

The revision adds only fail-closed setup observability:

- An exclusive first progress-file claim preserves the fresh-output boundary.
  Fixed stage names and completed-stage names are durable before each operation.
- Each source/model/wheel item is identified by its group, ordinal in the retained
  lock, expected per-asset SHA256 and expected size. Whole source/runtime lock SHA256
  identities are recorded separately after the locks are read. No arbitrary path
  or URL is copied.
  Verified items and the current incomplete item are distinguished.
- HTTP status and bytes actually returned to Python are recorded before hash
  checks. These body-byte counts exclude protocol traffic, redirect/error bodies not
  returned to the downloader, and bytes buffered internally but not returned; they are never called total network traffic.
- Exact observed file hashes are recorded only after reading completes. Length,
  hash, source-revision, host allowlist, byte, CPU/RAM and time limits remain
  unchanged. No retry or alternative source/model is introduced.
- Failures produce `setup-failure.json` with fixed reason categories and bounded
  HTTP status/errno/return-code integers. Exception messages, commands, URLs,
  response headers, raw logs, credentials and file bodies are never copied.
  A killed process may leave only the last durable progress snapshot, which stays
  partial rather than becoming a fabricated failure detail or success receipt.
  Container launch failures, Python import failures before handler entry, SIGKILL,
  and an unwritable output mount can leave no diagnostic. Handled constructor or
  signal-registration failures emit only a fixed safe summary if a receipt cannot
  be written; diagnostic persistence is never claimed without a successful write.

Current, separate source-preparation HEAD-only observations between 02:38 and
02:41 UTC on 2026-10-11 returned 200 for all 22 source URLs, 21 ordinary wheel
URLs, and the model URL; the locked Torch wheel host returned 403. No redirects
were followed and no wheel/model bodies were requested. This different
execution environment and later HEAD response cannot establish which item or
failure occurred in the first GitHub run. No URL or allowlist was changed and no
access denial was bypassed.

The retained failed-run evidence is outside the source freeze and blind code
mount. The revised freeze describes prospective code only and remains unapproved.
It neither changes the old run nor authorizes a second attempt. The existing
one-shot guards stay unchanged; another attempt requires separate user approval
and a separately reviewed activation.
