"""Deferred official feature-only reconstruction; no donor/model/decoder setup.

Original full audited_sources() also hashes missing resources/base.pt. This
adapter instead binds the two relevant exact source files, preserves their
fbank/accept_wave AST, and has no need to recover/download that donor checkpoint.
Pure AST inspection below does not import Torch or run any feature calculation.
"""
import ast
import struct
from types import SimpleNamespace
from contracts import require, sha

KALDI_SHA = '5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29'
STREAM_SHA = '2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53'


def frontend_asts(kaldi_path, stream_path):
    require(sha(kaldi_path) == KALDI_SHA and sha(stream_path) == STREAM_SHA, 'official feature source drift')
    tree = ast.parse(kaldi_path.read_text())
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.Import) and any(a.name == 'torchaudio' for a in n.names))
                 and not (isinstance(n, ast.FunctionDef) and n.name in ('mfcc', '_get_dct_matrix'))]
    require(not any(isinstance(n, ast.Name) and n.id == 'torchaudio' for n in ast.walk(tree)), 'unused module dependency')
    stream = ast.parse(stream_path.read_text())
    spotter = next(n for n in stream.body if isinstance(n, ast.ClassDef) and n.name == 'KeyWordSpotter')
    method = next(n for n in spotter.body if isinstance(n, ast.FunctionDef) and n.name == 'accept_wave')
    require(not any(isinstance(n, ast.Attribute) and n.attr in ('model', 'forward', 'decode_keywords')
                    for n in ast.walk(method)), 'accept_wave has model/decoder access')
    return tree, ast.Module(body=[method], type_ignores=[])


def build_feature_only_reference(kaldi_path, stream_path, observer=None):
    # Deferred until dependency + exact52-frontend-pass preparation release.
    import torch
    import numpy as np
    trees = frontend_asts(kaldi_path, stream_path)
    kaldi_ns = {}
    exec(compile(trees[0], str(kaldi_path), 'exec'), kaldi_ns)

    def hamming_fbank(waveform, **kwargs):
        require(waveform.dtype == torch.float32 and waveform.ndim == 2 and waveform.shape[0] == 1,
                'original wrapper waveform schema')
        result = kaldi_ns['fbank'](waveform, window_type='hamming', **kwargs)
        if observer is not None: observer(int(waveform.shape[1]), result)
        return result

    namespace = dict(torch=torch, np=np, F=torch.nn.functional, struct=struct,
                     kaldi=SimpleNamespace(fbank=hamming_fbank))
    exec(compile(trees[1], str(stream_path), 'exec'), namespace)
    return namespace['accept_wave']


def reconstruct_precmvn(pcm_bytes, accept_wave):
    import torch
    import numpy as np
    require(type(pcm_bytes) is bytes and len(pcm_bytes) % 2 == 0, 'PCM16 bytes')
    state = SimpleNamespace(sample_rate=16000, wave_remained=np.array([]), num_mel_bins=80,
        frame_length=25, frame_shift=10, downsampling=3, context_expansion=True,
        left_context=2, right_context=2, feature_remained=None, feats_ctx_offset=0,
        device=torch.device('cpu'))
    chunks, calls = [], []
    for start in range(0, len(pcm_bytes), 9600):
        actual = pcm_bytes[start:start+9600]
        x = accept_wave(state, actual)
        calls.append(dict(call_samples=len(actual)//2, selected_rows=0 if x is None else len(x)))
        if x is not None and len(x):
            require(x.dtype == torch.float32 and x.ndim == 2 and x.shape[1] == 400 and torch.isfinite(x).all().item(), 'official PRE-CMVN features')
            chunks.append(x)
    require(bool(chunks), 'no rows')
    # No padding or additional call at EOF. A fresh state for every waveform.
    return torch.cat(chunks, 0), calls


if __name__ == '__main__':
    raise SystemExit('PREPARATION_ONLY: reconstruction is not authorized or run')
