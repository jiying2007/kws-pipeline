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

The pair uses a new queue, keeps all raw waveforms, and does not modify the D1
plan or use generation quality to select references, parameters or retries.
Generated speech still requires separate audio-only ASR observations; four
synthetic calls cannot establish a general advantage for either mode.
