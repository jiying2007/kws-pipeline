"""Dependency-free source contracts, not model execution or resource admission.

The limits below bound metadata and prospective allocations. They are NOT an
SSC305 budget, a selected candidate recipe, or evidence of runtime feasibility.
All arithmetic describes the declared float32 reference source. No weights,
audio, backend, checkpoint or target device are opened by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

MAX_SPEC_BYTES = 16384
MAX_HIDDEN_DIM = 1024
MAX_LAYERS = 32
MAX_KERNEL_SIZE = 63
MAX_DILATION = 1024
MAX_PARAMETER_ELEMENTS = 1048576
MAX_CACHE_ELEMENTS = 1048576
MAX_BATCH_SIZE = 256
REFERENCE_DTYPE = 'float32'
REFERENCE_ELEMENT_BYTES = 4
SCHEMA = 'candidate-encoder-source-spec-v1'
LAYOUT_SCHEMA = 'candidate-encoder-research-layout-description-v1'
ARCHITECTURES = ('rnn64_control', 'causal_ds_tcn_ctc')
C0_SOURCE_PATH = 'training/model.py'
C0_SOURCE_COMMIT = 'fdd616be55190ece294e67a92ea8c4e14511705a'
C0_SOURCE_BLOB = 'f531fbfedf0ac33025744a0ba9188ccc67e892af'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high,
            f'{label} must be an exact integer in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class EncoderSpec:
    architecture: str
    feature_dim: int
    hidden_dim: int
    vocab_size: int
    kernels: tuple[int, ...] = ()
    dilations: tuple[int, ...] = ()

    def __post_init__(self):
        require(type(self.architecture) is str and self.architecture in ARCHITECTURES,
                'unsupported architecture')
        integer(self.feature_dim, 32, 32, 'feature_dim')
        integer(self.vocab_size, 5, 5, 'vocab_size')
        integer(self.hidden_dim, 1, MAX_HIDDEN_DIM, 'hidden_dim')
        require(type(self.kernels) is tuple and type(self.dilations) is tuple,
                'kernels and dilations must be immutable tuples')
        require(len(self.kernels) == len(self.dilations), 'layer dimensions disagree')
        if self.architecture == 'rnn64_control':
            require(self.hidden_dim == 64 and not self.kernels,
                    'C0 must preserve the 32x64x5 RNN and have no convolution layers')
        else:
            integer(len(self.kernels), 1, MAX_LAYERS, 'layer_count')
            for index, (kernel, dilation) in enumerate(zip(self.kernels, self.dilations)):
                integer(kernel, 1, MAX_KERNEL_SIZE, f'kernels[{index}]')
                integer(dilation, 1, MAX_DILATION, f'dilations[{index}]')
        require(self.parameter_elements <= MAX_PARAMETER_ELEMENTS,
                'parameter elements exceed source-allocation safety cap, not target budget')
        require(self.cache_elements <= MAX_CACHE_ELEMENTS,
                'cache elements exceed source-allocation safety cap, not target budget')

    @property
    def history_frames(self):
        """Per-layer finite convolution history; C0 has recurrent state instead."""
        return tuple((kernel - 1) * dilation
                     for kernel, dilation in zip(self.kernels, self.dilations))

    @property
    def weight_elements(self):
        h = self.hidden_dim
        if self.architecture == 'rnn64_control':
            return self.feature_dim * h + h * h + h * self.vocab_size
        return self.feature_dim * h + h * self.vocab_size + sum(
            h * kernel + h * h for kernel in self.kernels)

    @property
    def bias_elements(self):
        return self.hidden_dim + self.vocab_size + (
            2 * len(self.kernels) * self.hidden_dim
            if self.architecture == 'causal_ds_tcn_ctc' else 0)

    @property
    def parameter_elements(self):
        return self.weight_elements + self.bias_elements

    @property
    def cache_elements(self):
        if self.architecture == 'rnn64_control':
            return self.hidden_dim
        return self.hidden_dim * sum(self.history_frames)

    def state_shapes(self, batch_size):
        integer(batch_size, 1, MAX_BATCH_SIZE, 'batch_size')
        require(batch_size * self.cache_elements <= MAX_CACHE_ELEMENTS,
                'batched cache exceeds source-allocation safety cap, not target budget')
        if self.architecture == 'rnn64_control':
            return ((batch_size, self.hidden_dim),)
        return tuple((batch_size, self.hidden_dim, history) for history in self.history_frames)

    def to_dict(self):
        return dict(schema=SCHEMA, architecture=self.architecture,
                    feature_dim=self.feature_dim, hidden_dim=self.hidden_dim,
                    vocab_size=self.vocab_size, kernels=list(self.kernels),
                    dilations=list(self.dilations))


def rnn64_spec():
    return EncoderSpec('rnn64_control', 32, 64, 5)


def causal_ds_tcn_spec(*, hidden_dim, kernels, dilations):
    """No selected/default C1 dimensions: callers must supply every layer."""
    return EncoderSpec('causal_ds_tcn_ctc', 32, hidden_dim, 5, kernels, dilations)


def spec_from_dict(value):
    fields = {'schema', 'architecture', 'feature_dim', 'hidden_dim',
              'vocab_size', 'kernels', 'dilations'}
    require(type(value) is dict and set(value) == fields, 'spec missing or extra fields')
    require(type(value['schema']) is str and value['schema'] == SCHEMA, 'unsupported schema')
    require(type(value['kernels']) is list and type(value['dilations']) is list,
            'serialized layers must be JSON arrays')
    require(len(value['kernels']) <= MAX_LAYERS and len(value['dilations']) <= MAX_LAYERS,
            'too many serialized layers')
    return EncoderSpec(value['architecture'], value['feature_dim'], value['hidden_dim'],
                       value['vocab_size'], tuple(value['kernels']), tuple(value['dilations']))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def loads_spec(raw):
    require(type(raw) is bytes and len(raw) <= MAX_SPEC_BYTES, 'spec bytes exceed limit or wrong type')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid spec encoding or nesting') from error
    return spec_from_dict(value)


def load_spec(path):
    """Read only the explicitly supplied bounded JSON spec; never referenced assets."""
    with Path(path).open('rb') as stream:
        return loads_spec(stream.read(MAX_SPEC_BYTES + 1))


def spec_sha256(spec):
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def tensor_inventory(spec):
    """Canonical logical tensor shapes, not an exported checkpoint key guarantee."""
    require(type(spec) is EncoderSpec, 'exact EncoderSpec required')
    h, f, v = spec.hidden_dim, spec.feature_dim, spec.vocab_size
    rows = []
    def add(name, shape, kind):
        count = 1
        for dim in shape:
            count *= dim
        rows.append(dict(name=name, shape=list(shape), kind=kind,
                         dtype=REFERENCE_DTYPE, elements=count, bytes=count * REFERENCE_ELEMENT_BYTES))
    add('in_proj.weight', (h, f), 'weight')
    add('in_proj.bias', (h,), 'bias')
    if spec.architecture == 'rnn64_control':
        add('rec_proj.weight', (h, h), 'weight')
    else:
        for index, kernel in enumerate(spec.kernels):
            add(f'blocks.{index}.depthwise.weight', (h, 1, kernel), 'weight')
            add(f'blocks.{index}.depthwise.bias', (h,), 'bias')
            add(f'blocks.{index}.pointwise.weight', (h, h, 1), 'weight')
            add(f'blocks.{index}.pointwise.bias', (h,), 'bias')
    add('out_proj.weight', (v, h), 'weight')
    add('out_proj.bias', (v,), 'bias')
    return rows


def static_costs(spec):
    """Count one stream/frame in declared dense source; exclude runtime estimates."""
    require(type(spec) is EncoderSpec, 'exact EncoderSpec required')
    h, layers = spec.hidden_dim, len(spec.kernels)
    is_c1 = spec.architecture == 'causal_ds_tcn_ctc'
    return dict(
        status='STATIC_SOURCE_ARITHMETIC_NOT_RESOURCE_ADMISSION',
        mac_convention='one-multiply-accumulate-is-one-mac-bias-and-nonlinearity-separate',
        state_bytes_scope='logical-float32-tensor-payload-not-framework-or-autograd-memory',
        allocation_safety_caps_are_target_budgets=False,
        reference_dtype=REFERENCE_DTYPE,
        weight_elements=spec.weight_elements, bias_elements=spec.bias_elements,
        parameter_elements=spec.parameter_elements,
        float32_weight_bytes=spec.weight_elements * REFERENCE_ELEMENT_BYTES,
        float32_bias_bytes=spec.bias_elements * REFERENCE_ELEMENT_BYTES,
        float32_parameter_bytes=spec.parameter_elements * REFERENCE_ELEMENT_BYTES,
        state_elements_per_stream=spec.cache_elements,
        float32_state_bytes_per_stream=spec.cache_elements * REFERENCE_ELEMENT_BYTES,
        macs_per_frame=spec.weight_elements,
        assumed_sample_rate_hz=16000, assumed_frame_hop_samples=320,
        assumed_frames_per_audio_second=50,
        macs_per_audio_second_at_assumed_hop=spec.weight_elements * 50,
        non_mac_ops_per_frame=dict(bias_adds=spec.bias_elements,
            residual_or_recurrent_adds=h * layers if is_c1 else h,
            relu_elements=h * (1 + 2 * layers) if is_c1 else 0,
            tanh_elements=0 if is_c1 else h),
        omitted_costs=['frontend', 'decoder', 'softmax-or-log-softmax', 'cache-copy',
                      'layout-copy', 'allocation', 'kernel-launch-and-runtime-overhead'],
        encoder_receptive_field_frames=1 + sum(spec.history_frames) if is_c1 else None,
        recurrent_history_unbounded=not is_c1,
        encoder_right_context_frames=0, encoder_output_stride_frames=1,
        quantized_weights_bytes=None, scale_bytes=None, scratch_bytes=None,
        frontend_ring_bytes=None, total_runtime_arena_bytes=None,
        chunk_activation_bytes=None, autograd_graph_bytes=None,
        measured_cpu=None, measured_latency_ms=None, measured_peak_rss_bytes=None,
        resource_admitted=False, numerical_parity='NOT_RUN')


def research_export_layout(spec):
    """Describe a prospective logical layout. This does not serialize model data.

    Offsets are relative to a hypothetical float32 payload without a header.
    They are neither KWSP offsets nor an ABI, file-format or exporter promise.
    """
    rows = tensor_inventory(spec)
    offset = 0
    for row in rows:
        row['hypothetical_payload_offset'] = offset
        offset += row['bytes']
    states = [dict(name='recurrent' if spec.architecture == 'rnn64_control' else f'block.{i}.input_history',
                   shape_per_stream=shape[1:], dtype=REFERENCE_DTYPE,
                   reset='all-zero', time_order='oldest-to-newest' if len(shape) == 3 else None)
              for i, shape in enumerate(spec.state_shapes(1))]
    return dict(schema=LAYOUT_SCHEMA, artifact_kind='JSON-description-only-no-weight-payload',
        architecture_spec=spec.to_dict(), architecture_spec_sha256=spec_sha256(spec),
        implementation_status='SOURCE_PRESENT_NUMERICAL_NOT_RUN',
        c0_reference=dict(path=C0_SOURCE_PATH, commit=C0_SOURCE_COMMIT, git_blob_sha1=C0_SOURCE_BLOB,
                          supplied_instance_origin_verified=False) if spec.architecture == 'rnn64_control' else None,
        source_semantics=dict(normalization='none', dropout='none', dtype=REFERENCE_DTYPE,
            convolution_stride=1 if spec.architecture == 'causal_ds_tcn_ctc' else None,
            convolution_padding=0 if spec.architecture == 'causal_ds_tcn_ctc' else None,
            depthwise_tap_order='oldest-to-current-cross-correlation' if spec.architecture == 'causal_ds_tcn_ctc' else None),
        logical_tensors=rows, hypothetical_float32_payload_bytes=offset,
        payload_written=False, binary_header_bytes=None, abi_assigned=False,
        payload_alignment_bytes=4, payload_byte_order=None,
        alignment_scope='hypothetical-float32-tensor-payload-only',
        kwsp_compatible=False, candidate_c_runtime_available=False,
        states=states, state_ownership='caller-owned-explicit-tuple',
        state_chunk_boundary='preserve-history-no-implicit-reset',
        discontinuity='caller-must-reset-all-layers', eof='no-invented-frames-or-flush',
        step_input_order='B,F', step_output_order='B,V',
        chunk_input_order='B,T,F', chunk_output_order='T,B,V',
        empty_chunk='zero-output-frames-and-unchanged-state',
        full_forward='reset-before-input',
        static_costs=static_costs(spec),
        execution_authorized=False, training_ready=False, product_qualified=False,
        numerical_parity='NOT_RUN', candidate_c_export_parity='NOT_RUN')
