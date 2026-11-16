"""Fixed trailing PCM -> frozen ONNX embedding; no startup/EOF silence.

The frontend's ReduceMax may depend on its supplied window. Recomputing only
the already available one-second trailing window avoids whole-clip future input.
This is a new explicit feature recipe, not claimed parity with upstream's stream.
"""
from __future__ import annotations
from collections import deque
import math
from numbers import Integral
from pathlib import Path

from weak_embedding_contract import (ASSETS, EMBED_HOP, EMBED_WIDTH, HEAD_HISTORY,
                                     MEL_SAMPLES, digest, need)


def inspect_mel_convolution_geometry(path):
    """Future worker: read graph metadata before making any clip-support claim.

The expected two parallel STFT convolutions are checked, not inferred from the
97-frame output alone. No inference is performed by this function. The fixed
asset hash must already have been checked by the caller.
"""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    initializers = {x.name: x for x in model.graph.initializer}
    convolutions = []
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        weight = initializers.get(node.input[1])
        need(weight is not None and list(weight.dims) == [257, 1, 512], 'STFT_CONV_WEIGHT_GEOMETRY')
        kernel = list(attributes.get('kernel_shape', [512]))
        stride = list(attributes.get('strides', [1]))
        pads = list(attributes.get('pads', [0, 0]))
        dilation = list(attributes.get('dilations', [1]))
        need(kernel == [512] and stride == [160] and pads == [0, 0] and dilation == [1]
             and attributes.get('group', 1) == 1 and
             attributes.get('auto_pad', b'NOTSET') in (b'', b'NOTSET'), 'STFT_CONV_ATTRIBUTES')
        convolutions.append({'name': node.name, 'kernel': kernel, 'stride': stride,
                             'pads': pads, 'dilation': dilation, 'weight_shape': list(weight.dims)})
    need(len(convolutions) == 2 and
         not any(n.op_type in ('Pad', 'Slice', 'Resize', 'AveragePool', 'MaxPool') for n in model.graph.node),
         'FIXED_STFT_PATH_GEOMETRY')
    frame_count = (MEL_SAMPLES - 512) // 160 + 1
    support_start = (frame_count - 76) * 160
    support_end = (frame_count - 1) * 160 + 512
    need(frame_count == 97 and support_start == 3360 and support_end == 15872, 'MEL_FRAME_SUPPORT')
    return {'status': 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED', 'convolutions': convolutions,
            'selected_mel_input_support': [support_start, support_end], 'mel_frames': frame_count,
            'head_support_start_before_available':
                (HEAD_HISTORY - 1) * EMBED_HOP + MEL_SAMPLES - support_start,
            'support_end_before_available': MEL_SAMPLES - support_end}


class CausalFeatureStream:
    def __init__(self, embedding_callback):
        self.callback = embedding_callback
        self.samples = deque(maxlen=MEL_SAMPLES)
        self.observed = 0
        self.next_end = MEL_SAMPLES
        self.embedding_calls = 0

    def accept(self, samples):
        records = []
        for value in samples:
            need(isinstance(value, Integral) and not isinstance(value, bool) and
                 -32768 <= value <= 32767, 'PCM16_VALUE')
            self.samples.append(int(value))
            self.observed += 1
            if self.observed == self.next_end:
                vector = tuple(float(x) for x in self.callback(tuple(self.samples)))
                need(len(vector) == EMBED_WIDTH and all(math.isfinite(x) for x in vector),
                     'EMBEDDING_96_FINITE')
                records.append({'available_sample': self.observed, 'embedding': vector})
                self.embedding_calls += 1
                self.next_end += EMBED_HOP
        return records

    def finish(self):
        """Unobserved samples never become model input or an extra output."""
        return []


class FrozenOnnxEmbedding:
    def __init__(self, asset_directory):
        # Imported only by a separately authorized real research worker.
        import numpy as np
        import onnxruntime as ort
        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        sessions = {}
        for name, (size, expected_sha) in ASSETS.items():
            path = Path(asset_directory) / name
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == size
                 and digest(path.read_bytes()) == expected_sha, 'FROZEN_ASSET:' + name)
            if name == 'melspectrogram.onnx':
                self.geometry = inspect_mel_convolution_geometry(path)
            sessions[name] = ort.InferenceSession(str(path), sess_options=options,
                                                  providers=['CPUExecutionProvider'])
        self.mel = sessions['melspectrogram.onnx']
        self.embedding = sessions['embedding_model.onnx']
        need([(x.name, x.type) for x in self.mel.get_inputs()] == [('input', 'tensor(float)')]
             and [(x.name, x.type) for x in self.embedding.get_inputs()] ==
             [('input_1', 'tensor(float)')], 'FROZEN_GRAPH_INPUTS')

    def __call__(self, pcm16):
        np = self.np
        need(len(pcm16) == MEL_SAMPLES, 'ONE_SECOND_OBSERVED_PCM')
        raw = np.asarray(pcm16, dtype=np.float32).reshape(1, MEL_SAMPLES)
        # Upstream takes float32 PCM16-scale values, not values divided by 32768.
        mel = self.mel.run(None, {'input': raw})[0]
        need(mel.shape == (1, 1, 97, 32) and mel.dtype == np.float32 and
             bool(np.isfinite(mel).all()), 'FROZEN_MEL_GEOMETRY')
        window = ((mel[0, 0, -76:, :] / np.float32(10)) + np.float32(2)).reshape(1, 76, 32, 1)
        vector = self.embedding.run(None, {'input_1': window})[0]
        need(vector.shape == (1, 1, 1, 96) and vector.dtype == np.float32 and
             bool(np.isfinite(vector).all()), 'FROZEN_EMBEDDING_GEOMETRY')
        return vector.reshape(96)


def windows_at_ends(records, ends):
    """Select finite causal head inputs; missing history is an error, never padding."""
    positions = {r['available_sample']: i for i, r in enumerate(records)}
    need(len(positions) == len(records), 'DUPLICATE_EMBEDDING_CLOCK')
    result = []
    for end in ends:
        need(end in positions and positions[end] + 1 >= HEAD_HISTORY, 'NO_FULL_HEAD_HISTORY')
        i = positions[end]
        selected = records[i + 1 - HEAD_HISTORY:i + 1]
        need([r['available_sample'] for r in selected] ==
             list(range(end - (HEAD_HISTORY - 1) * EMBED_HOP, end + 1, EMBED_HOP)),
             'NONCONTIGUOUS_HEAD_HISTORY')
        result.append([list(r['embedding']) for r in selected])
    return result
