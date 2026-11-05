# Active research Python

Host tooling and new research workers use **CPython 3.12**. The exact active
patch and the `core`, `tts`, and `asr` dependency roles are declared once in
[`configs/runtime/research-runtime.json`](../configs/runtime/research-runtime.json).
The current target is **3.12.16**. Selecting an interpreter and checking package
metadata do not establish model quality or successful end-to-end inference.
The combined ASR role completed 192 native recognitions over 64 existing clips
on this patch, with all 727 input/result files re-read and hash checked. The
manifest records that scoped evidence separately; TTS model validation on this
patch is still pending, so the overall model-smoke status remains pending.
See the [active ASR guide](ACTIVE_ASR_BATCH.md) for conditions and limits.

The device runtime remains pure C. It has no Python runtime dependency.
Python is a host build, data, training and evaluation tool.

## CI and local entry

After checking out this repository, active workflows use:

```yaml
- uses: ./.github/actions/setup-research-python
```

The action reads the manifest with `jq`, selects its exact version using the
pinned `actions/setup-python`, and verifies the resulting interpreter. It does
not import Torch, download models, or install the ML dependency roles. This
avoids relying on whichever `python3` happens to be in a runner image.

The hosted compiler job also installs the single YAML parser used by workflow
checks from [`host-tools.requirements.lock`](../configs/runtime/host-tools.requirements.lock).
That CPython 3.12 / Linux x86_64 wheel is pinned by version and SHA256; the job
does not rely on packages preinstalled for the runner's system Python.

For a local environment, install the patch reported by the manifest and run:

```sh
python3.12 tools/research_runtime.py check
python3.12 tools/research_runtime.py run -- tools/verify_model_registry.py --help
```

The checker rejects a different implementation, minor, or patch with the
expected and observed versions. There is no Python 3.8 compatibility lane.

For a manually extracted Linux `actions/python-versions` archive, `bin/python3.12`
can require the archive's `lib` directory in `LD_LIBRARY_PATH`. A venv created
from that interpreter shares this requirement; activating the venv does not
repair the loader path. Configure these variables in each shell that uses the
extracted runtime, with absolute paths to the actual archive:

```sh
export KWS_RESEARCH_PYTHON=/absolute/path/to/extracted-python/bin/python3.12
export LD_LIBRARY_PATH="/absolute/path/to/extracted-python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$KWS_RESEARCH_PYTHON" tools/research_runtime.py check
bash training/setup_core_runtime.sh .venv
.venv/bin/python tools/research_runtime.py check --role core --packages
```

`setup_core_runtime.sh` honors `KWS_RESEARCH_PYTHON`; otherwise it uses `python3`
from `PATH`. The loader setting is inherited by its venv and subprocesses.
Do not infer that a raw extracted archive is usable from an earlier import
receipt: verify the interpreter, package metadata and actual workload in the
current process environment. Hosted CI uses `actions/setup-python` to configure
the interpreter rather than directly executing a raw archive.

## Three dependency roles

| Role | Scope | Why separate |
|---|---|---|
| `core` | Governance, KWS training, evaluation and export | Current KWS uses NumPy 2.3.2 and CPU Torch 2.13.0. Pure tools need no ML imports. |
| `tts` | Qwen TTS | Torch/Torchaudio 2.11.0, Transformers 4.57.3, NumPy 1.26.4 and SoundFile 0.13.1. |
| `asr` | Qwen ASR, SenseVoice and native Whisper | Torch 2.12.1, Transformers 4.57.6, NumPy 1.26.4 and SoundFile 0.14.0. Keep models in separate processes. |

Prepare the active core role from the repository root:

```sh
bash training/setup_core_runtime.sh .venv
.venv/bin/python tools/research_runtime.py check --role core --packages
.venv/bin/python tools/research_runtime.py run --role core --packages -- training/train_ctc.py --help
```

The helper checks the interpreter before installing, gets package pins from
the manifest, uses the configured PyPI and CPU Torch indexes, and runs
`pip check`. These are **direct dependency pins**, not a complete transitive
wheel lock. An immutable training OCI image digest still binds the complete
installed environment. `training/requirements.lock` is a derived assertion
file for that image, not a second source of Python policy:

```sh
python3.12 tools/research_runtime.py training-lock > training/requirements.lock
python3.12 tools/research_runtime.py training-lock --check training/requirements.lock
```

The TTS and ASR roles reference retained dependency locks by SHA256. Their
dependencies are deliberately not installed by the core helper:

```sh
python3.12 tools/research_runtime.py check-locks
python3.12 tools/research_runtime.py check --role asr --packages
```

The retained primary ASR and Whisper package inventories have no conflicting
versions; Whisper adds `more-itertools`. That supports a combined ASR role,
but it is not evidence that the combined environment has executed a model.
The existing successful three-model preparation used separate primary and
Whisper environments on Python 3.12.14. Historical locks, completed campaign
source, and their receipts remain unchanged. Frozen one-shot workflows and
the runner environment probe continue to report their original environment.

## Completing the patch migration

Before marking the new model environment validated, run bounded real checks:
one core forward/backward and export/replay path, tiny nonclone TTS, and all
three ASR implementations on the same small audio set. Retain interpreter,
package and model identities with the actual output receipts. Imports and
source-shape tests alone do not satisfy this check. A future patch update
changes the manifest and derived assertion file together; it does not rewrite
old campaign evidence.

Python 3.12.16 is the current security release in this series. NumPy 1.26.4
officially supports Python 3.9–3.12, which makes 3.12 the common supported minor
for the current role pins. Moving to 3.13 would also require changing those
NumPy-dependent environments; it is a separate upgrade, not a prerequisite for
the current research loop.

- [Python 3.12.16 release](https://www.python.org/downloads/release/python-31216/)
- [NumPy 1.26.4 supported Python versions](https://numpy.org/doc/stable/release/1.26.4-notes.html)
- [Python version support status](https://devguide.python.org/versions/)
