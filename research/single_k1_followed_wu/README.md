# One continuous Melo utterance

Exactly one synthesis call generated a candidate for **你好小窝，屋里有人**.
The native 44.1-kHz float32 signal has 65,536 samples (1.486077 seconds); its
fixed 16-kHz PCM16 derivative has 23,778 samples. A later human review confirmed the audible words
你好小窝，屋里有人 for this exact native WAV. Word boundaries, tail integrity
and confidence remain unknown. The review was exposed to the intended text
and is not independent or blinded. It establishes no CTC labels or training admission. No ASR, KWS, training, parameter
sweep, retake, appended silence, or concatenation was performed in this run.

## Executed source and projected plan

The executed adapter, conversion, supervisor, four original test files,
numeric input, model metadata, and mapping provenance are retained unchanged.
`adapter-freeze.json` is the historical preexecution source freeze; its false
inference flag describes that preparation stage. The runtime result and append-only human revision are in the
[separate data archive](https://github.com/jiying2007/kws-data/tree/research/single-k1-followed-wu-v1/research/2026-10-07-single-k1-followed-wu). New `setup_safety.py` and its tests are later reusable
setup fixes and were not part of the executed synthesis source.

`plan-projection.json` distinguishes the original executed plan SHA256
`9a74353b84ab6e1d9f49a448b8aee4f7a2a23392e8c327a0448fa2486d7e7e29`
from the public plan SHA256
`c30090ca35c6c40b26675312f02e3fb70a06f70e29b086e9afb0e4554650dab7`.
Only the unused top-level `authorization_status` annotation is omitted.
All scientific, resource, identity, binding and execution fields are equal.
The adapter never reads the omitted field. Published generation receipts
continue to name the executed hash; the public file does not reproduce that
original byte hash. Historical `approved=false` in the dependency lock and
preexecution status fields are not rewritten as successful execution.

## Frozen recipe and prerequisites

- One CPU session, ORT_SEQUENTIAL, intra-op 2/inter-op 1, batch 1, sid=[1]
- ORT seed 0 once before the session; noise_scale=0.6, length_scale=1.0,
  noise_scale_w=0.8; no bit-determinism claim for the two stochastic graph nodes
- Dictionary tones, retaining 你好3+3; 17 preblank phones and 35-element x/tones
- Official comma ID106/tone0, with fullwidth comma normalized to ASCII comma;
  no selected pause duration or inferred acoustic boundary
- Native mono float32 44.1 kHz retained unchanged; float64 SciPy resample_poly,
  ratio160/441, Kaiser5, constant-zero extension, then floor(x*32768), saturate
  and encode little-endian PCM16; no trim, gain, denoise or padding

Use CPython3.12.14, NumPy2.3.5, SciPy1.17.0 and packaging26.3. The exact
three additional wheels, URLs, sizes and SHA256 values are in
`dependency-lock.json` and `requirements-runtime.txt`: ONNX Runtime1.30.0,
flatbuffers25.12.19 and protobuf6.33.5, totaling23,783,002 bytes.

The pinned [model and mapping repository](https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/tree/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6)
provides model.onnx (170,429,550 bytes, SHA256
`bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d`),
tokens.txt (655 bytes), and lexicon.txt (6,837,671 bytes). Mapping body hashes
and Git blob identities are in `input-mapping-provenance.json`, together with
the relevant dictionary rows. Punctuation, blank insertion and fixed scales
come from the [pinned official frontend](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/scripts/melo-tts/test.py).
No frontend package or segmenter was installed or run.

Download only these exact assets after checking publisher pins. Use
`wheel_identity.py` to reject unsafe/duplicate members, invalid metadata,
corruption and excessive expansion. Check native loader dependencies first.
Install the three hash-locked wheels into a fresh task-local venv with
`--system-site-packages` solely for the pinned existing NumPy/SciPy/packaging:

```sh
python -m venv --system-site-packages runtime/venv
runtime/venv/bin/python -m pip install --no-index --find-links runtime/wheels --only-binary=:all: --no-deps --require-hashes --no-cache-dir -r requirements-runtime.txt
```

Do not upgrade the host, resolve extra packages or use an alternate model.
Maintain cumulative256-MiB response-body, 8-MiB mapping, 256-MiB expanded-wheel,
1-GiB setup-workspace and shared4-MiB log ceilings. A fresh complete setup has
a300-second wall bound. This run's earlier failed300-second setup window
remains failed; a separate bounded remaining-assets phase used180 seconds and
finished in39.150 seconds. See the data archive for that resource history.

`setup_safety.run_bounded` separates the installer file-size cap from the
shared stdout/stderr log cap and kills the owned process group on every exit.
It neither authorizes nor retries an install. `parse_smoke_stdout` requires one
complete exact JSON record while native stderr stays separate; it rejects
extra values/keys, duplicate keys and wrong scalar types. These address the
recorded file-cap and merged-stream parser failures without new dependencies.
Verify installed RECORD hashes and exact versions with an import-only smoke;
do not create an inference session or make an inference preflight call.

## Offline verification and explicit execution

With NumPy/SciPy already installed, these tests are offline and forbid model
runtime imports. Their fake sessions and subprocesses never synthesize speech:

```sh
python -B -m unittest -v test_one_call test_conversion_recipe test_output_signature test_private_raw_cause test_setup_safety
```

The dedicated workflow runs only those pure tests. It installs no TTS runtime
and invokes no scientific experiment. The ordinary repository CI is separate.

The adapter is disabled by default. A separately requested future execution
must name a fresh output directory and the public plan hash explicitly:

```sh
runtime/venv/bin/python -B run_single_melo.py --execute-approved-single-melo --plan execution-plan.json --plan-sha256 c30090ca35c6c40b26675312f02e3fb70a06f70e29b086e9afb0e4554650dab7 --model runtime/model.onnx --out generation-once
```

This consumes at most one durable write-ahead attempt; no resume or retry is
provided. Limits:60 seconds per call,600 seconds overall including model load,
441,000 native samples/10 seconds,20 MiB evidence,4 MiB logs, and a sampled
owned-process-group2-GiB RSS stop threshold. The RSS threshold is not a
continuous hard memory guarantee. The pinned output signature allows only the
known unit batch/channel shape refinements; every other dimension stays strict.

MeloTTS and sherpa-onnx notices are retained with their respective scopes.
They do not establish separate training-data or voice rights, speaker novelty,
detector quality, word boundaries, product accuracy or deployment readiness.

`inspect_waveforms.py DATA_DIRECTORY OUTPUT_JSON` reproduces the fixed
standard-library waveform check from the data archive. It verifies input hashes,
reports numerical gap candidates, and supplies no word or phoneme timing. The
portable copy changes paths and reporting text only; measured values were
compared exactly with the retained measurement. It does not invoke a model.
