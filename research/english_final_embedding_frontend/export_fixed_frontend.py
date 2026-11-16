"""Pure local-file export for the two pinned openWakeWord v0.5.1 feature graphs.

No network, inference, optimization, training, or ONNXRuntime import. The public
CLI accepts only exact pinned asset bytes. lower_graph/emit_c also accept small
invented graph dictionaries so source tests need no ML package or real asset.
Operator semantics are the ONNX default-domain opset-13 schemas. Unsupported
types, attributes, dynamic shape operands, ranks, or output forms raise ValueError.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct

MAX_RANK = 6
MAX_INPUTS = 8
MAX_FLOATS = 16 * 1024 * 1024  # exporter bound, not a board memory claim
NONE = 0xffffffff
ASSETS = {
    'melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f',
                            'input', (1, 16000), 'output', (1, 1, 97, 32)),
    'embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f',
                             'input_1', (1, 76, 32, 1), 'conv2d_19', (1, 1, 1, 96)),
}
COUNTS = {
    'melspectrogram.onnx': {'Add': 1, 'Cast': 1, 'Clip': 2, 'Constant': 9,
        'Conv': 2, 'Div': 1, 'Log': 1, 'MatMul': 1, 'Mul': 1, 'Pow': 2,
        'ReduceMax': 1, 'Sub': 2, 'Transpose': 2, 'Unsqueeze': 3},
    'embedding_model.onnx': {'Conv': 20, 'LeakyRelu': 19, 'Max': 19,
                             'MaxPool': 5, 'Reshape': 2},
}
OPCODES = {'Add': 'EF_ADD', 'Sub': 'EF_SUB', 'Mul': 'EF_MUL', 'Div': 'EF_DIV',
    'Pow': 'EF_POW', 'Max': 'EF_MAX', 'Log': 'EF_LOG', 'Clip': 'EF_CLIP',
    'LeakyRelu': 'EF_LEAKY_RELU', 'Transpose': 'EF_TRANSPOSE',
    'ReduceMax': 'EF_REDUCE_MAX', 'MatMul': 'EF_MATMUL', 'Conv': 'EF_CONV',
    'MaxPool': 'EF_MAX_POOL'}
ALLOWED = {
    **{x: set() for x in ('Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Log', 'Clip',
                          'MatMul', 'Reshape', 'Unsqueeze')},
    'Constant': {'value', 'value_float', 'value_floats', 'value_int', 'value_ints'},
    'Cast': {'to'}, 'LeakyRelu': {'alpha'}, 'Transpose': {'perm'},
    'ReduceMax': {'axes', 'keepdims'},
    'Conv': {'auto_pad', 'dilations', 'group', 'kernel_shape', 'pads', 'strides'},
    'MaxPool': {'auto_pad', 'ceil_mode', 'dilations', 'kernel_shape', 'pads',
                'storage_order', 'strides'},
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def f32(value):
    value = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    need(math.isfinite(value), 'NONFINITE_CONSTANT')
    return value


def json_floats(value):
    """Describe original IEEE infinities without emitting nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'ieee754_float32': struct.pack('>f', value).hex(),
                'value': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {key: json_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_floats(item) for item in value]
    return value


class GraphExportError(ValueError):
    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.graph_diagnostic = diagnostic


def integer(value, what):
    need(type(value) is int and -0x7fffffff <= value <= 0x7fffffff, what)
    return value


def shape(value):
    value = tuple(value)
    need(len(value) <= MAX_RANK and all(type(x) is int and x > 0 for x in value),
         'STATIC_POSITIVE_SHAPE')
    need(math.prod(value) <= MAX_FLOATS, 'TENSOR_SIZE_BOUND')
    return value


def int_list(value, count, what, minimum=None):
    need(isinstance(value, (list, tuple)) and len(value) == count, what)
    result = [integer(x, what) for x in value]
    need(minimum is None or all(x >= minimum for x in result), what)
    return result


def broadcast(a, b):
    n = max(len(a), len(b))
    aa, bb = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    need(all(x == y or x == 1 or y == 1 for x, y in zip(aa, bb)), 'BROADCAST_SHAPE')
    return shape(max(x, y) for x, y in zip(aa, bb))


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = 'float32'
    data: tuple | None = None
    base: str | None = None
    born: int = -1


def constant(name, spec):
    need(set(spec) == {'shape', 'dtype', 'data'}, 'CONSTANT_FIELDS')
    dims = tuple(spec['shape'])
    # Empty integer vectors are allowed only as compile-time shape metadata.
    need(len(dims) <= MAX_RANK and all(type(x) is int and x >= 0 for x in dims)
         and math.prod(dims) <= MAX_FLOATS, 'CONSTANT_SHAPE')
    dtype = spec['dtype']
    need(dtype in ('float32', 'int32', 'int64'), 'CONSTANT_DTYPE')
    values = tuple(spec['data'])
    need(len(values) == math.prod(dims), 'CONSTANT_COUNT')
    if dtype == 'float32':
        need(all(x > 0 for x in dims), 'EMPTY_FLOAT_TENSOR')
        values = tuple(f32(x) for x in values)
    else:
        lo, hi = (-2**31, 2**31 - 1) if dtype == 'int32' else (-2**63, 2**63 - 1)
        need(all(type(x) is int and lo <= x <= hi for x in values), 'INTEGER_CONSTANT')
    return Tensor(name, dims, dtype, values, name)


def spatial(x, weight, attrs, pool):
    rank = len(x) - 2
    need(rank in (1, 2), 'CONV_POOL_ONLY_1D_2D')
    kernel = int_list(attrs.get('kernel_shape', weight[2:] if weight else []), rank,
                      'KERNEL_SHAPE', 1)
    stride = int_list(attrs.get('strides', [1] * rank), rank, 'STRIDES', 1)
    dilation = int_list(attrs.get('dilations', [1] * rank), rank, 'DILATIONS', 1)
    auto = attrs.get('auto_pad', 'NOTSET')
    if isinstance(auto, bytes):
        auto = auto.decode('ascii')
    need(auto in ('NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER'), 'AUTO_PAD')
    need(auto == 'NOTSET' or 'pads' not in attrs, 'AUTO_PAD_WITH_EXPLICIT_PADS')
    pads = int_list(attrs.get('pads', [0] * (2 * rank)), 2 * rank, 'PADS', 0)
    ceil = integer(attrs.get('ceil_mode', 0), 'CEIL_MODE') if pool else 0
    need(ceil in (0, 1), 'CEIL_MODE')
    # Deprecated auto_pad + ceil_mode definitions vary among historical
    # implementations. Reject that combination instead of silently choosing.
    need(not (pool and ceil and auto != 'NOTSET'), 'POOL_AUTO_PAD_WITH_CEIL_UNSUPPORTED')
    group = integer(attrs.get('group', 1), 'GROUP') if not pool else 1
    need(group > 0, 'GROUP')
    if pool:
        need(integer(attrs.get('storage_order', 0), 'STORAGE_ORDER') == 0,
             'POOL_INDICES_STORAGE_ORDER_UNSUPPORTED')
        channels = x[1]
    else:
        need(len(weight) == len(x) and tuple(kernel) == weight[2:]
             and weight[1] * group == x[1] and weight[0] % group == 0,
             'CONV_WEIGHT_GROUP_SHAPE')
        channels = weight[0]
    outputs = []
    for i in range(rank):
        effective = dilation[i] * (kernel[i] - 1) + 1
        if auto.startswith('SAME_'):
            out = (x[i + 2] + stride[i] - 1) // stride[i]
            total = max(0, (out - 1) * stride[i] + effective - x[i + 2])
            pads[i] = total // 2 if auto == 'SAME_UPPER' else (total + 1) // 2
            pads[i + rank] = total - pads[i]
        numerator = x[i + 2] + pads[i] + pads[i + rank] - effective
        out = (numerator + (stride[i] - 1 if ceil else 0)) // stride[i] + 1
        if pool and ceil and (out - 1) * stride[i] >= x[i + 2] + pads[i]:
            out -= 1
        need(out > 0, 'EMPTY_SPATIAL_OUTPUT')
        if pool:
            # Refuse all-padding windows, including dilation holes. They would
            # have no finite maximum; none is needed by the pinned feature path.
            for j in range(out):
                start = j * stride[i] - pads[i]
                need(any(0 <= start + k * dilation[i] < x[i + 2]
                         for k in range(kernel[i])), 'EMPTY_POOL_WINDOW')
        outputs.append(out)
    two = lambda xs, default: xs + ([default] if rank == 1 else [])
    p = [rank, group] + two(kernel, 1) + two(stride, 1) + two(dilation, 1)
    p += two(pads[:rank], 0) + two(pads[rank:], 0) + [ceil]
    return shape((x[0], channels, *outputs)), p


def lower_graph(spec):
    """Static lowering only. Input dict is also the synthetic test fixture API."""
    need(set(spec) <= {'input', 'output', 'initializers', 'nodes', 'output_shape'}, 'GRAPH_FIELDS')
    inp = spec['input']
    need(set(inp) == {'name', 'shape', 'dtype'} and inp['dtype'] == 'float32', 'GRAPH_INPUT')
    tensors = {inp['name']: Tensor(inp['name'], shape(inp['shape']), base=inp['name'])}
    unbounded = {}

    def graph_constant(name, data):
        # An explicit +/-infinity used solely as the corresponding Clip bound
        # is an absent limit for finite float32 runtime values. No nonfinite
        # weight, arithmetic input, output or NaN is accepted by this rule.
        values = data.get('data', ())
        nonfinite = data.get('dtype') == 'float32' and any(
            not math.isfinite(float(v)) for v in values)
        if not nonfinite:
            return constant(name, data)
        need(set(data) == {'shape', 'dtype', 'data'} and tuple(data['shape']) == () and
             len(values) == 1 and math.isinf(float(values[0])) and name != spec['output'],
             'NONFINITE_CONSTANT_NOT_SCALAR_CLIP_BOUND:' + name)
        position = 2 if values[0] > 0 else 1
        uses = [{'node_index': index, 'node_name': node.get('name', ''),
                 'op': node['op'], 'input_index': slot}
                for index, node in enumerate(spec['nodes'])
                for slot, source in enumerate(node.get('inputs', [])) if source == name]
        need(uses and all(use['op'] == 'Clip' and use['input_index'] == position for use in uses),
             'NONFINITE_CONSTANT_NOT_EXCLUSIVE_CLIP_LIMIT:' + name)
        unbounded[name] = {'tensor': name, 'original_value': json_floats(float(values[0])),
                          'clip_input_index': position, 'uses': uses,
                          'lowered_to': 'ABSENT_OPTIONAL_CLIP_BOUND_FOR_FINITE_INPUTS'}
        # This placeholder is removed from every use and therefore never
        # enters the emitted constant array or C runtime graph.
        return constant(name, {'shape': [], 'dtype': 'float32', 'data': [0.0]})

    for name, data in spec.get('initializers', {}).items():
        need(name not in tensors and name, 'DUPLICATE_TENSOR')
        tensors[name] = graph_constant(name, data)
    nodes, original = [], []
    for raw in spec['nodes']:
        need(set(raw) <= {'op', 'inputs', 'output', 'attrs', 'name', 'domain'}, 'NODE_FIELDS')
        op, inputs, out = raw['op'], list(raw.get('inputs', [])), raw['output']
        attrs = dict(raw.get('attrs', {}))
        need(op in ALLOWED and not raw.get('domain'), 'OP_OR_DOMAIN:' + str(op))
        need(set(attrs) <= ALLOWED[op], 'UNKNOWN_ATTRIBUTE:' + op)
        need(type(out) is str and out and out not in tensors, 'DUPLICATE_OUTPUT')
        need(len(inputs) <= MAX_INPUTS and all(not n or n in tensors for n in inputs), 'NODE_INPUTS')
        original.append({'op': op, 'name': raw.get('name', ''), 'inputs': inputs,
                         'output': out, 'attrs': attrs})
        if op == 'Constant':
            need(not inputs and len(attrs) == 1, 'CONSTANT_FORM')
            key, val = next(iter(attrs.items()))
            if key == 'value':
                data = val
            else:
                plural = key.endswith('s')
                values = list(val) if plural else [val]
                data = {'shape': [len(values)] if plural else [],
                        'dtype': 'int64' if key.startswith('value_int') else 'float32', 'data': values}
            tensors[out] = graph_constant(out, data)
            continue
        inputs = ['' if name in unbounded else name for name in inputs]
        need(inputs and inputs[0], 'REQUIRED_INPUT')
        xs = [tensors[n] if n else None for n in inputs]
        x = xs[0]
        if op in ('Reshape', 'Unsqueeze'):
            need(len(xs) == 2 and xs[1] is not None and xs[1].dtype == 'int64'
                 and xs[1].data is not None and len(xs[1].shape) == 1, 'STATIC_INT64_SHAPE_INPUT')
            values = list(xs[1].data)
            if op == 'Reshape':
                need(len(values) <= MAX_RANK and values.count(-1) <= 1 and all(v >= -1 for v in values),
                     'RESHAPE_SPEC')
                for i, val in enumerate(values):
                    if val == 0:
                        need(i < len(x.shape), 'RESHAPE_ZERO_AXIS')
                        values[i] = x.shape[i]
                known = math.prod(v for v in values if v != -1)
                if -1 in values:
                    need(known > 0 and math.prod(x.shape) % known == 0, 'RESHAPE_INFERENCE')
                    values[values.index(-1)] = math.prod(x.shape) // known
                newshape = shape(values)
                need(math.prod(newshape) == math.prod(x.shape), 'RESHAPE_COUNT')
            else:
                rank = len(x.shape) + len(values)
                need(rank <= MAX_RANK and all(-rank <= a < rank for a in values), 'UNSQUEEZE_AXES')
                axes = {a % rank for a in values}
                need(len(axes) == len(values), 'UNSQUEEZE_DUPLICATE_AXES')
                old = iter(x.shape)
                newshape = shape(1 if i in axes else next(old) for i in range(rank))
            tensors[out] = Tensor(out, newshape, x.dtype, x.data, x.base, x.born)
            continue
        if op == 'Cast':
            need(len(xs) == 1 and set(attrs) == {'to'}, 'CAST_FORM')
            need(type(attrs['to']) is int and attrs['to'] in (1, 6, 7), 'CAST_TYPE')
            dtype = {1: 'float32', 6: 'int32', 7: 'int64'}[attrs['to']]
            if dtype == x.dtype:
                tensors[out] = Tensor(out, x.shape, x.dtype, x.data, x.base, x.born)
            else:
                need(x.data is not None, 'RUNTIME_CAST_NOT_FLOAT_IDENTITY')
                converted = [f32(v) if dtype == 'float32' else int(v) for v in x.data]
                tensors[out] = constant(out, {'shape': x.shape, 'dtype': dtype, 'data': converted})
            continue
        need(all(t is None or t.dtype == 'float32' for t in xs), 'FLOAT_RUNTIME_ONLY')
        p, alpha = [], 0.0
        if op in ('Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max'):
            need(all(t is not None for t in xs) and
                 (1 <= len(xs) <= MAX_INPUTS if op == 'Max' else len(xs) == 2), 'ELEMENTWISE_INPUTS')
            yshape = x.shape
            for t in xs[1:]:
                yshape = broadcast(yshape, t.shape)
        elif op in ('Log', 'LeakyRelu'):
            need(len(xs) == 1, 'UNARY_INPUTS')
            yshape = x.shape
            if op == 'LeakyRelu':
                alpha = f32(attrs.get('alpha', .01))
        elif op == 'Clip':
            need(1 <= len(xs) <= 3 and all(t is None or t.shape == () for t in xs[1:]), 'CLIP_SCALARS')
            # Dynamic float scalar limits are supported; no implicit narrowing.
            inputs += [''] * (3 - len(inputs))
            yshape = x.shape
        elif op == 'Transpose':
            need(len(xs) == 1, 'TRANSPOSE_INPUTS')
            p = int_list(attrs.get('perm', list(reversed(range(len(x.shape))))),
                         len(x.shape), 'TRANSPOSE_PERM')
            need(sorted(p) == list(range(len(x.shape))), 'TRANSPOSE_PERM')
            yshape = shape(x.shape[i] for i in p)
        elif op == 'ReduceMax':
            need(len(xs) == 1, 'REDUCE_INPUTS')
            axes = attrs.get('axes', list(range(len(x.shape))))
            need(isinstance(axes, (list, tuple)) and all(type(a) is int for a in axes), 'REDUCE_AXES')
            if not axes:
                axes = list(range(len(x.shape)))
            need(all(-len(x.shape) <= a < len(x.shape) for a in axes), 'REDUCE_AXES')
            normalized = {a % len(x.shape) for a in axes}
            need(len(normalized) == len(axes), 'REDUCE_DUPLICATE_AXES')
            keep = integer(attrs.get('keepdims', 1), 'KEEPDIMS')
            need(keep in (0, 1), 'KEEPDIMS')
            yshape = shape(1 if i in normalized else d for i, d in enumerate(x.shape)) if keep else \
                shape(d for i, d in enumerate(x.shape) if i not in normalized)
            p = [sum(1 << a for a in normalized), keep]
        elif op == 'MatMul':
            need(len(xs) == 2 and xs[1] is not None and x.shape and xs[1].shape, 'MATMUL_INPUTS')
            b = xs[1].shape
            aa = (1, x.shape[0]) if len(x.shape) == 1 else x.shape
            bb = (b[0], 1) if len(b) == 1 else b
            need(aa[-1] == bb[-2], 'MATMUL_INNER')
            y = (*broadcast(aa[:-2], bb[:-2]), aa[-2], bb[-1])
            if len(b) == 1:
                y = y[:-1]
            if len(x.shape) == 1:
                y = y[:-1] if len(b) == 1 else y[:-2] + y[-1:]
            yshape = shape(y)
        elif op in ('Conv', 'MaxPool'):
            pool = op == 'MaxPool'
            need(len(xs) == 1 if pool else len(xs) in (2, 3) and xs[1] is not None,
                 'CONV_POOL_INPUTS')
            weight = None if pool else xs[1].shape
            yshape, p = spatial(x.shape, weight, attrs, pool)
            if not pool and len(xs) == 3 and xs[2] is not None:
                need(xs[2].shape == (yshape[1],), 'CONV_BIAS')
        else:
            raise ValueError('UNLOWERED_OPERATOR:' + op)
        tensors[out] = Tensor(out, yshape, base=out, born=len(nodes))
        nodes.append({'op': op, 'inputs': inputs, 'output': out, 'p': p + [0] * (16 - len(p)),
                      'alpha': alpha})
    output = spec['output']
    need(output in tensors and tensors[output].dtype == 'float32', 'GRAPH_OUTPUT')
    if 'output_shape' in spec:
        need(tensors[output].shape == shape(spec['output_shape']), 'DECLARED_OUTPUT_SHAPE')
    used = {inp['name'], output}
    for n in nodes:
        used.update(i for i in n['inputs'] if i)
        used.add(n['output'])
    bases = {tensors[n].base for n in used}
    used.update(bases)
    selected = {n: t for n, t in tensors.items() if n in used}
    need(all(t.dtype == 'float32' for t in selected.values()), 'LOWERED_FLOAT_TENSORS')
    last = {b: tensors[b].born for b in bases}
    for i, n in enumerate(nodes):
        for name in n['inputs']:
            if name:
                last[tensors[name].base] = max(last[tensors[name].base], i)
    last[tensors[output].base] = len(nodes)
    offsets, free, active, high = {}, [], {}, 0
    for i, n in enumerate(nodes):
        for old in list(active):
            if last[old] < i:
                free.append(active.pop(old))
        free.sort()
        merged = []
        for off, size in free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        free = merged
        name = n['output']
        count = (math.prod(tensors[name].shape) + 3) // 4 * 4
        choice = next((j for j, (_, size) in enumerate(free) if size >= count), None)
        if choice is None:
            off, high = high, high + count
        else:
            off, size = free.pop(choice)
            if size > count:
                free.append((off + count, size - count))
        offsets[name], active[name] = off, (off, count)
    need(high <= MAX_FLOATS, 'WORKSPACE_SIZE_BOUND')
    weights, constant_offsets = [], {}
    for name, t in tensors.items():
        if name in bases and t.data is not None:
            constant_offsets[name] = len(weights)
            weights.extend(t.data)
    entries, ids = [], {}
    for name, t in selected.items():
        ids[name] = len(entries)
        if t.base == inp['name']:
            storage, offset = 'EF_INPUT', 0
        elif t.base in constant_offsets:
            storage, offset = 'EF_CONSTANT', constant_offsets[t.base]
        else:
            storage, offset = 'EF_WORK', offsets[t.base]
        entries.append({'name': name, 'shape': t.shape, 'count': math.prod(t.shape),
                        'storage': storage, 'offset': offset, 'base': t.base})
    for n in nodes:
        n['input_ids'] = [ids[x] if x else NONE for x in n['inputs']]
        n['output_id'] = ids[n['output']]
    return {'tensors': entries, 'nodes': nodes, 'input': ids[inp['name']], 'output': ids[output],
            'constants': weights, 'workspace_floats': high, 'original_nodes': original,
            'unbounded_clip_limits': list(unbounded.values()),
            'lifetimes': {b: {'born': tensors[b].born, 'last': last[b], 'offset': offsets.get(b)}
                          for b in sorted(bases)}}


def c_float(v):
    return f32(v).hex() + 'f'


def emit_c(graph, symbol):
    need(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', symbol) is not None, 'C_SYMBOL')
    lines = ['#include "fixed_frontend.h"', '']
    values = graph['constants']
    lines.append(f'static const float {symbol}_constants[{max(1, len(values))}] = {{')
    for i in range(0, len(values), 8):
        lines.append('  ' + ', '.join(c_float(x) for x in values[i:i + 8]) + ',')
    if not values:
        lines.append('  0.0f,')
    lines.append('};')
    lines.append(f'static const ef_tensor {symbol}_tensors[] = {{')
    for t in graph['tensors']:
        dims = list(t['shape']) + [0] * (MAX_RANK - len(t['shape']))
        lines.append('  {%d, {%s}, %d, %s, %d},' %
            (len(t['shape']), ', '.join(map(str, dims)), t['count'], t['storage'], t['offset']))
    lines.append('};')
    lines.append(f'static const ef_node {symbol}_nodes[{max(1, len(graph["nodes"]))}] = {{')
    for n in graph['nodes']:
        ins = n['input_ids'] + [NONE] * (MAX_INPUTS - len(n['input_ids']))
        lines.append('  {%s, %d, {%s}, %d, {%s}, %s},' %
            (OPCODES[n['op']], len(n['input_ids']), ', '.join('EF_NONE' if v == NONE else str(v) for v in ins),
             n['output_id'], ', '.join(map(str, n['p'])), c_float(n['alpha'])))
    if not graph['nodes']:
        lines.append('  {0},')
    lines.extend(['};', f'const ef_graph {symbol} = {{',
        f'  {symbol}_tensors, {len(graph["tensors"])}, {symbol}_nodes, {len(graph["nodes"])},',
        f'  {graph["input"]}, {graph["output"]}, {symbol}_constants, {len(values)}, {graph["workspace_floats"]}', '};'])
    return '\n'.join(lines) + '\n'


def tensor_proto(t):
    # No numpy or onnxruntime: parse only inline float32/int32/int64 tensors.
    need(not t.external_data and t.data_location == 0 and not t.HasField('segment'), 'EXTERNAL_OR_SEGMENT_TENSOR')
    mapping = {1: ('float32', 'f', 'float_data', 4), 6: ('int32', 'i', 'int32_data', 4),
               7: ('int64', 'q', 'int64_data', 8)}
    need(t.data_type in mapping, 'ONNX_TENSOR_DTYPE')
    dtype, fmt, field, width = mapping[t.data_type]
    count = math.prod(t.dims)
    need(count <= MAX_FLOATS, 'ONNX_TENSOR_SIZE')
    if t.raw_data:
        need(len(t.raw_data) == count * width and not getattr(t, field), 'ONNX_RAW_TENSOR')
        data = list(struct.unpack('<' + fmt * count, t.raw_data))
    else:
        data = list(getattr(t, field))
    return {'shape': list(t.dims), 'dtype': dtype, 'data': data}


def read_pinned_onnx(path, expected):
    size, sha, input_name, input_shape, output_name, output_shape = expected
    need(path.is_file() and not path.is_symlink(), 'ASSET_FILE')
    raw = path.read_bytes()
    need(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, 'PINNED_ASSET_SHA:' + path.name)
    import onnx  # protobuf/checker only; never execute the model
    model = onnx.load_model_from_string(raw)
    need(model.ir_version == 7 and not model.functions and not model.training_info,
         'ONNX_IR_OR_FUNCTIONS')
    domains = [(x.domain, x.version) for x in model.opset_import]
    need(len({d for d, _ in domains}) == len(domains) and ('', 13) in domains
         and all((d, v) in (('', 13), ('ai.onnx.ml', 2)) for d, v in domains), 'ONNX_OPSETS')
    graph = model.graph
    need(not graph.sparse_initializer and len(graph.input) == 1 and len(graph.output) == 1,
         'ONNX_GRAPH_INTERFACE')
    for item, name, dims in ((graph.input[0], input_name, input_shape),
                             (graph.output[0], output_name, output_shape)):
        need(item.name == name and item.type.HasField('tensor_type') and
             item.type.tensor_type.elem_type == 1 and item.type.tensor_type.HasField('shape'), 'ONNX_IO_TYPE')
        declared = item.type.tensor_type.shape.dim
        need(len(declared) == len(dims) and
             all(not d.HasField('dim_value') or d.dim_value == v for d, v in zip(declared, dims)), 'ONNX_IO_DIM')
    initializers = {}
    for t in graph.initializer:
        need(t.name and t.name not in initializers, 'DUPLICATE_INITIALIZER')
        initializers[t.name] = tensor_proto(t)
    rows = []
    for node in graph.node:
        need(len(node.output) == 1 and node.output[0], 'SINGLE_NODE_OUTPUT_ONLY')
        attrs = {}
        for a in node.attribute:
            need(a.name not in attrs and not a.ref_attr_name, 'DUPLICATE_OR_REFERENCE_ATTRIBUTE')
            if a.type == onnx.AttributeProto.TENSOR:
                val = tensor_proto(a.t)
            elif a.type == onnx.AttributeProto.FLOAT:
                val = float(a.f)
            elif a.type == onnx.AttributeProto.INT:
                val = int(a.i)
            elif a.type == onnx.AttributeProto.STRING:
                val = a.s.decode('ascii')
            elif a.type == onnx.AttributeProto.INTS:
                val = list(a.ints)
            elif a.type == onnx.AttributeProto.FLOATS:
                val = list(a.floats)
            else:
                raise ValueError('ONNX_ATTRIBUTE_TYPE:' + a.name)
            attrs[a.name] = val
        rows.append({'op': node.op_type, 'domain': node.domain, 'name': node.name,
                     'inputs': list(node.input), 'output': node.output[0], 'attrs': attrs})
    need(dict(Counter(n['op'] for n in rows)) == COUNTS[path.name], 'AUDITED_OPERATOR_COUNTS')
    onnx.checker.check_model(model, full_check=False)
    spec = {'input': {'name': input_name, 'shape': input_shape, 'dtype': 'float32'},
            'output': output_name, 'output_shape': output_shape,
            'initializers': initializers, 'nodes': rows}
    try:
        return lower_graph(spec)
    except (ValueError, OverflowError) as error:
        diagnostic = {'asset_name': path.name, 'asset_sha256': sha,
            'reason': str(error), 'original_nodes': json_floats(rows),
            'initializers': {name: {key: value[key] for key in ('shape', 'dtype')}
                             for name, value in initializers.items()},
            'model_inference_calls': 0}
        raise GraphExportError(path.name + ':' + str(error), diagnostic) from error


def export_pair(asset_dir, output_dir):
    need(not output_dir.exists(), 'FRESH_OUTPUT_DIRECTORY_REQUIRED')
    graphs = {name: read_pinned_onnx(asset_dir / name, expected) for name, expected in ASSETS.items()}
    output_dir.mkdir(parents=True)
    symbols = {'melspectrogram.onnx': 'ef_fixed_mel', 'embedding_model.onnx': 'ef_fixed_embedding'}
    source = '\n'.join(emit_c(g, symbols[name]) for name, g in graphs.items())
    (output_dir / 'fixed_models.c').write_text(source)
    (output_dir / 'fixed_models.h').write_text('#ifndef EF_FIXED_MODELS_H\n#define EF_FIXED_MODELS_H\n'
        '#include "fixed_frontend.h"\nextern const ef_graph ef_fixed_mel;\n'
        'extern const ef_graph ef_fixed_embedding;\n#endif\n')
    report = {'schema': 'fixed-feature-pure-export-v1', 'real_inference_performed': False,
              'numerical_parity_established': False, 'ssc305_qualification': False, 'graphs': {}}
    for name, g in graphs.items():
        report['graphs'][name] = {'asset_sha256': ASSETS[name][1], 'asset_bytes': ASSETS[name][0],
            'input': g['tensors'][g['input']], 'output': g['tensors'][g['output']],
            'constant_bytes': len(g['constants']) * 4, 'workspace_bytes': g['workspace_floats'] * 4,
            'tensors': g['tensors'], 'nodes': g['nodes'], 'lifetimes': g['lifetimes'],
            'original_node_attributes': json_floats(g['original_nodes']),
            'unbounded_clip_limits': g['unbounded_clip_limits']}
    report['generated_c_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    (output_dir / 'export_manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True,
                                                              allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = export_pair(args.assets, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'graphs'}, sort_keys=True))


if __name__ == '__main__':
    main()
