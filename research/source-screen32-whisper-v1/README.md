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
