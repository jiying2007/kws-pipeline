# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reviewed SoundFile loader adaptation for the bounded CosyVoice3 pilot.

Adapted from cosyvoice/utils/file_utils.py at official source revision
074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc. The decoding call is changed from
torchaudio.load(..., backend='soundfile') to soundfile.read. Channel averaging
and default TorchAudio Resample behavior remain the stock operations. The
minimum input rate is checked only when the rates differ, as in stock code.

This module imports only the standard library until a public function is
called. The runner must verify the source tree and locked installed runtime
before calling bind_frontend or qualify_adapter. This module is not a source
hash/dependency admission gate and does not acquire any source or package.
"""

import importlib
from pathlib import Path
import sys


def load_wav(wav, target_sr, min_sr=16000):
    """Decode float32, average channels, and resample exactly as stock does.

    SoundFile returns frames-by-channels because always_2d is explicit; Torch
    receives channels-by-frames before its float32 mean. Keep the original
    error type/message, including the upstream message's target_sr wording.
    Explicit raising preserves the safety check even under Python -O.
    """
    import soundfile
    import torch
    import torchaudio

    samples, sample_rate = soundfile.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(samples).transpose(0, 1)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise AssertionError(
                "wav sample rate {} must be greater than {}".format(
                    sample_rate, target_sr
                )
            )
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def _module_file(module, expected):
    """Reject cached/shadow source rather than patch a different checkout."""
    location = getattr(module, "__file__", None)
    if location is None or Path(location).resolve() != expected.resolve():
        raise RuntimeError(
            "unexpected source for {}: {!r}; expected {}".format(
                module.__name__, location, expected
            )
        )


def bind_frontend(source_root):
    """Patch file_utils first, import the actual frontend, verify both aliases.

    Return the actual imported cosyvoice.cli.frontend module. Call once in a
    fresh single-threaded process. An existing frontend binding is rejected,
    even if it happens to point at this loader. Keep this source root on
    sys.path for subsequent authorized model imports. On any failure the
    caller must stop/discard the process, not continue or retry model loading.
    No CosyVoiceFrontEnd, model, or ONNX Runtime session is constructed here.
    """
    frontend_name = "cosyvoice.cli.frontend"
    if frontend_name in sys.modules:
        raise RuntimeError("frontend already imported; use a fresh process")

    root = Path(source_root).resolve(strict=True)
    expected_utils = root / "cosyvoice" / "utils" / "file_utils.py"
    expected_frontend = root / "cosyvoice" / "cli" / "frontend.py"
    for path in (expected_utils, expected_frontend):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError("missing or escaped approved source: {}".format(path))

    # A cached package from another tree wins over sys.path. Reject it before
    # changing any binding. Hash validation of this root belongs to the caller.
    for name, module in tuple(sys.modules.items()):
        if name == "cosyvoice" or name.startswith("cosyvoice."):
            if module is None:
                raise RuntimeError("partially imported source module: " + name)
            location = getattr(module, "__file__", None)
            paths = getattr(module, "__path__", ())
            if location is not None:
                if not Path(location).resolve().is_relative_to(root):
                    raise RuntimeError("cached source outside approved root: " + name)
            elif not paths or any(
                not Path(path).resolve().is_relative_to(root) for path in paths
            ):
                raise RuntimeError("unverifiable cached source module: " + name)

    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    file_utils = importlib.import_module("cosyvoice.utils.file_utils")
    _module_file(file_utils, expected_utils)
    # The stock frontend uses `from ... import load_wav`: ordering is mandatory.
    file_utils.load_wav = load_wav
    if frontend_name in sys.modules:
        raise RuntimeError("frontend imported before loader binding; stop")
    frontend = importlib.import_module(frontend_name)
    _module_file(frontend, expected_frontend)
    if frontend.load_wav is not load_wav or file_utils.load_wav is not load_wav:
        raise RuntimeError("actual frontend/file_utils loader binding mismatch")
    return frontend


def qualify_adapter(source_root, scratch_dir):
    """Run the real numeric qualification; return a JSON-serializable receipt.

    This explicit call imports heavy dependencies. Do not call it in local
    stdlib/static checks. See qualify_soundfile_adapter.py for the CLI and the
    exact deterministic PCM fixtures. Its successful binding remains active.
    """
    from qualify_soundfile_adapter import qualify_adapter as qualify

    return qualify(source_root, scratch_dir)
