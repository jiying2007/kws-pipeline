# Active local TTS smoke

`tools/setup_tts_runtime.py` builds the separate CPython 3.12.16 TTS role from
92 retained artifact pins, including CPU Torch/torchaudio 2.11.0 and qwen-tts
0.1.1. It verifies the public Qwen Base model revision and all 13 assets in
`configs/runtime/qwen-base-tts-model-lock.json`. Installation does not run a
model. The Base assets are distinct from the historical CustomVoice assets.

`tools/active_tts_smoke.py prepare` copies a reviewed source pool, its eight
WAV/rights records, and the selected producer implementation into a private
package. It freezes the complete 256-cell D1 plan and a separate recipe allowing
only the first eight cells, one per source. Preparation records raw file hashes
separately from the producer's canonical plan hash. The first eight preserve
their identities if the remaining plan is considered in a later task.

When a restored runtime retains its actual import receipt under another name,
`prepare --runtime-preflight PATH` accepts the existing
`kws-runtime-import-preflight-v1` TTS receipt. Before creating the prepared
directory it checks the successful import-only result, zero constructed models
and model calls, CPU/library versions against the retained setup, original
command/time fields and offline environment. Dependency warnings and both
captured streams remain byte-for-byte intact. The recipe binds the original
receipt path/hash as well as the copied bytes, and `run` rechecks that binding.
This option does not rerun an import or establish model readiness. Omitting it
keeps the legacy `runtime/preflight.json` interface and receipt format.

`run` requires the persistent research queue and its verified inherited heavy
model lock. It uses one process, CPU float32/SDPA, four Torch computation threads
and one interop thread. The recipe fixes a 600-second queue deadline, no retries,
and an explicit 256-codec-token limit. Sampling otherwise uses the existing
bulk defaults, with Chinese as an explicit synthesis language condition.
Model loading happens once before the per-cell loop; a load failure aborts the
job. A create-only invocation prevents a repeated command from advancing the
256-cell plan by another eight.

The reused producer chooses ICL only for a verified or
`dataset-provided-content-checked` reference. Other references pass no reference
text and use x-vector mode. This source text check is separate from future ASR
checking of generated outputs. The source roles remain attached to every cell.

The wrapper retains the exact returned array and a native FLOAT WAV before the
existing technical gate. A rejected output therefore retains its original
waveform and a consumed claim. Passing outputs additionally use the existing
24-to-16 kHz conversion (`resample_poly(2,3)`, round float64 times 32767 to PCM16,
without gain, trimming or clipping repair). The voice-clone API does not expose
the native termination reason; receipts do not assert EOS termination.

Queue output declarations should include `summary.json`, `inventory.json`,
`invocation.json`, and `generation/plan-lock.json`. The inventory binds every
saved call and media file; consumers must verify those bytes before use. All
source text, speech, generated audio and actual run receipts remain private.
Neither a technical audio pass nor queue completion admits training data or
establishes keyword correctness. The source tests run offline in the Python
contract CI step; they do not download weights or synthesize audio.

## Native offline enforcement

New workers set `ORT_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`DO_NOT_TRACK=1`, and the Hugging Face offline settings before any ML import.
ONNX Runtime's [privacy documentation](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md)
distinguishes this initialization-time opt-out from the API that disables only
nonessential events. The worker calls that API as an additional measure.

`tools/research_no_network.py` first requires Linux x86_64, a single thread, and
no inherited socket descriptors. It then attaches a seccomp filter to its own
process, rejects socket creation and communication syscalls, and rejects the
io_uring paths that could otherwise create sockets. Compat and x32 syscall ABIs
are rejected. Descendants inherit these additional restrictions. No host network
or existing sandbox policy is changed. The guard is installed before native
libraries can create background threads; unsupported hosts fail closed.

The recorded filter identity proves which restrictions were installed. It does
not count attempted requests, and receipts leave that count unknown. Regression
tests exercise Python and libc socket creation, inherited-socket rejection, and
successful ordinary file I/O without sending any request. An additional private
preflight imports the real installed TTS dependencies under the filter and
checks native FLOAT WAV I/O, without loading model weights.

## Separate mode comparison

`tools/active_tts_pair.py` freezes a separate four-call recipe only when at least
two references have three completed, distinct audio-only recognizer observations
matching their provided full text. The two reference IDs are selected in
lexicographic order before any smoke outcomes. Each reference supplies one ICL
and one x-vector call with the same target phrase, seed, model, generation
options and audio. The seed is reset immediately before generation, after
reference prompt construction. Mode is part of each independent case identity.

The pair uses a separate recipe and job, keeps all raw waveforms, and does not
modify the D1 plan or use generation quality to select references, parameters
or retries. It may follow the smoke job as an explicit dependency in the same
newly frozen queue.
Generated speech still requires separate audio-only ASR observations; four
synthetic calls cannot establish a general advantage for either mode.

## Preserved execution on 2026-11-05

The [aggregate execution receipt](../research/active-tts-20261105/execution-summary.json)
binds the separately preserved private raw evidence and independent readback.
The first attempt was interrupted after a dependency telemetry request was
rejected. It has no generation-call receipt or returned waveform; its actual
elapsed time remains unknown. Its original claim is retained and a conservative
600 seconds is reserved in the cross-attempt budget.

The second attempt enforced the offline environment and native network guard
before model imports. Eight smoke calls returned eight raw waveforms and passed
the existing technical gate. The separate four-call pair returned all four raw
waveforms; three passed the technical gate. One ICL waveform had RMS
0.00604118875 below the predeclared 0.008 threshold. Its 1.36-second original
FLOAT audio remains retained, without gain adjustment, replacement or retry.
Amplitude alone does not determine whether a wake phrase was spoken.

The two completed queue jobs consumed 216.447998461 measured seconds, with a
peak RSS of 5,873,564 KiB. This known duration plus the old 600-second reservation
is a conservative allocation account, not a recovered measurement of the old
attempt. The declared total includes 60 seconds for cleanup within 1,800 seconds.
Generated-content ASR remains separate. Only eight of the 256 D1 cells were
generated; the four paired outputs do not advance that roster.
