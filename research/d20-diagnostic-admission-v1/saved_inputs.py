"""Bind the four frozen D20 saved files to bytes; never execute an operator.

The public entry point accepts no hash/layout override. Private byte/slice helpers
are testable with small synthetic data but do not grant authenticity or admission.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import zipfile

import contract

require = contract.require
SOURCE_FILES = {
    'torch_npz': ('local-certificate-v2-failed/qualification/run-once/D20/pcm9600.npz', 301434),
    'c_npz': ('local-certificate-v2-failed/qualification/run-once/D20/pcm9600-C.npz', 442116),
    'export_manifest': ('host-runtime-ab/exports/D20/manifest.json', 8034),
    'weights': ('host-runtime-ab/exports/D20/a20.f32', 1565280),
}
COUNTS = (9, 10)
MAX_HEADER_BYTES = 4096


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_frozen(path, size, sha256):
    # Open only one regular file, then retain the exact authenticated bytes.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size == size, 'frozen file size/type mismatch')
        raw = source.read(size + 1)
    require(len(raw) == size and _sha(raw) == sha256, 'frozen file bytes mismatch')
    return raw


def _npy_payload(raw, dtype, shape):
    """Narrow extension of cosyvoice3_cross_voice.packager.native_header.

    Headers have the same 4096-byte cap and literal-only parsing. No dtype
    conversion or float unpacking is used: signed-zero bits remain untouched.
    The caller supplies the already verified saved member's exact shape/dtype.
    """
    require(raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00'), 'NPY format mismatch')
    width = 2 if raw[6] == 1 else 4
    length = int.from_bytes(raw[8:8 + width], 'little')
    require(0 < length <= MAX_HEADER_BYTES, 'NPY header limit')
    end = 8 + width + length
    header = ast.literal_eval(raw[8 + width:end].decode('ascii'))
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'NPY header mismatch')
    require(header['descr'] == dtype and header['fortran_order'] is False, 'NPY dtype/order mismatch')
    require(type(header['shape']) is tuple and all(type(n) is int for n in header['shape'])
            and header['shape'] == shape, 'NPY shape mismatch')
    require(dtype in ('<f4', '<i8'), 'unsupported saved dtype')
    payload = raw[end:]
    require(len(payload) == math.prod(shape) * (4 if dtype == '<f4' else 8), 'NPY payload size mismatch')
    return payload, end


def _row(payload, shape, row):
    require(type(row) is int and 0 <= row < shape[0], 'row out of bounds')
    width = math.prod(shape[1:]) * 4
    require(len(payload) == shape[0] * width, 'array byte count mismatch')
    offset = row * width
    return payload[offset:offset + width], offset


def _row_identity(row):
    require(type(row) is int and 0 <= row < sum(COUNTS), 'global row out of bounds')
    return (0, row) if row < COUNTS[0] else (1, row - COUNTS[0])


def _input_key(part, stage):
    require(type(part) is int and part in (0, 1) and type(stage) is int and 0 <= stage < 21,
            'invalid saved stage')
    return f'cmvn{part}' if stage == 0 else f'stage{part}_{stage - 1}'


def _weight_inventory():
    # Exact native_a20/export_a20.py inventory, without importing its exporter.
    result = [('global_cmvn.mean', (400,)), ('global_cmvn.istd', (400,))]

    def affine(name, ni, no):
        result.extend([(name + '.linear.weight', (no, ni)), (name + '.linear.bias', (no,))])

    affine('backbone.in_linear1', 400, 140)
    affine('backbone.in_linear2', 140, 250)
    for layer in range(4):
        prefix = f'backbone.fsmn.{layer}'
        result.extend([(prefix + '.0.linear.weight', (128, 250)),
                       (prefix + '.1.conv_left.weight', (128, 1, 10, 1)),
                       (prefix + '.1.conv_right.weight', (128, 1, 2, 1))])
        affine(prefix + '.2', 128, 250)
    affine('backbone.out_linear1', 250, 140)
    affine('backbone.out_linear2', 140, 6)
    return result


def _stage_weights(stage):
    if stage in (2, 6, 10, 14, 18):
        return ()
    if stage in contract.MEMORY_STAGES:
        prefix = f'backbone.fsmn.{(stage - 4) // 4}.1'
        return (prefix + '.conv_left.weight', prefix + '.conv_right.weight')
    prefix = {0: 'backbone.in_linear1', 1: 'backbone.in_linear2',
              19: 'backbone.out_linear1', 20: 'backbone.out_linear2'}.get(stage)
    if prefix is None:
        layer, operation = divmod(stage - 3, 4)
        require(0 <= layer < 4 and operation in (0, 2), 'invalid weight stage')
        prefix = f'backbone.fsmn.{layer}.{operation}'
        if operation == 0:
            return (prefix + '.linear.weight',)
    return (prefix + '.linear.weight', prefix + '.linear.bias')


def _weights(manifest, raw):
    require(manifest.get('schema') == 'paired-bare-state-f32-export-v1'
            and manifest.get('arm') == 'D20' and manifest.get('steps') == 1200
            and manifest.get('dtype') == 'little-endian IEEE754 float32'
            and manifest.get('symbols') == ['<blank>', '你', '好', '小', '窝', '屋'], 'export schema mismatch')
    require(manifest.get('payload_bytes') == len(raw) and manifest.get('payload_sha256') == _sha(raw),
            'export payload mismatch')
    tensors = manifest.get('tensors')
    inventory = _weight_inventory()
    require(isinstance(tensors, list) and len(tensors) == len(inventory), 'export inventory mismatch')
    bound = {}
    offset = 0
    for tensor, (name, shape) in zip(tensors, inventory):
        size = math.prod(shape) * 4
        require(tensor.get('name') == name and tensor.get('shape') == list(shape)
                and all(type(n) is int for n in tensor['shape']), 'export tensor identity mismatch')
        require(type(tensor.get('offset')) is int and tensor['offset'] == offset
                and type(tensor.get('bytes')) is int and tensor['bytes'] == size, 'export tensor offset mismatch')
        data = raw[offset:offset + size]
        require(len(data) == size and _sha(data) == tensor.get('sha256'), 'export tensor bytes mismatch')
        bound[name] = dict(source='weights', name=name, shape=list(shape), dtype='<f4', order='C',
                           source_offset=offset, bundle_offset=offset, bytes=size, sha256=_sha(data))
        offset += size
    require(offset == len(raw) == SOURCE_FILES['weights'][1], 'export tensor coverage mismatch')
    return bound


def _layout(kind):
    """Exact layouts observed in the two hash-verified saved archives."""
    require(kind in ('c_npz', 'torch_npz'), 'unknown saved archive')
    result = {'input': ('<f4', (19, 400)), 'counts': ('<i8', (2,))}
    for part, n in enumerate(COUNTS):
        result[f'cmvn{part}'] = ('<f4', (n, 400))
        for stage, width in enumerate(contract.DIMS):
            result[f'stage{part}_{stage}'] = ('<f4', (n, width))
        for key in ('logits', 'probabilities'):
            result[f'{key}{part}'] = ('<f4', (n, 6))
        result[f'cache{part}'] = ('<f4', (1, 128, 11, 4))
        if kind == 'c_npz':
            result[f'cmvn_same{part}'] = ('<f4', (n, 400))
            for key in ('cache_before_', 'cache_after_'):
                result[f'{key}{part}'] = ('<f4', (n, 128, 11, 4))
    return result


def _arrays(raw, kind):
    # Called only after all four full-file identities pass. This is not a general
    # archive API: fixed names, encoding, header size and payload sizes only.
    layout = _layout(kind)
    arrays, identities = {}, {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == len(layout) and
                {member.filename for member in members} == {key + '.npy' for key in layout},
                'saved NPZ member set mismatch')
        for member in members:
            key = member.filename[:-4]
            dtype, shape = layout[key]
            size = math.prod(shape) * (4 if dtype == '<f4' else 8) + 128
            require(member.compress_type == zipfile.ZIP_DEFLATED and member.flag_bits & 1 == 0
                    and member.file_size == size, 'saved NPZ member layout mismatch')
            with archive.open(member) as source:
                data = source.read(size + 1)
            require(len(data) == size and data[:8] == b'\x93NUMPY\x01\x00', 'saved NPY length/version mismatch')
            payload, offset = _npy_payload(data, dtype, shape)
            require(offset == 128, 'saved NPY header offset mismatch')
            arrays[key] = payload
            identities[key] = dict(member=member.filename, member_bytes=size, member_sha256=_sha(data),
                                   dtype=dtype, shape=list(shape), order='C', data_offset=offset,
                                   payload_bytes=len(payload), payload_sha256=_sha(payload))
    require(arrays['counts'] == struct.pack('<qq', *COUNTS), 'saved 9+10 frame identity mismatch')
    return arrays, identities


def _build_plan(arrays, raw_weights, weights):
    """Byte mapping only. Synthetic callers do not receive an authenticity claim."""
    payload = bytearray(raw_weights)

    def append(key, part_row):
        shape = _layout('c_npz')[key][1]
        data, offset = _row(arrays[key], shape, part_row)
        binding = dict(source='c_npz', array_key=key, row_in_part=part_row,
                       array_byte_offset=offset, member_byte_offset=128 + offset,
                       bundle_offset=len(payload), bytes=len(data), sha256=_sha(data))
        payload.extend(data)
        return binding

    plan = dict(schema='d20-saved-single-op-plan-v1', frozen_sources=dict(contract.FROZEN_SOURCES),
                gates={name: list(value) for name, value in contract.GATES.items()},
                required_backend={'torch': '2.11.0+cpu', 'numpy': '1.26.4'},
                limits=dict(contract.LIMITS), anchor='C_SAVED_OBSERVED', input_chaining=False,
                historical_chunk_equivalence=False, original_status={'D20': 'FAIL', 'D90': 'NOT_RUN'}, jobs=[])
    for row in range(sum(COUNTS)):
        part, part_row = _row_identity(row)
        cache = append(f'cache_before_{part}', part_row)
        for stage, output_width in enumerate(contract.DIMS):
            binding = append(_input_key(part, stage), part_row)
            names = _stage_weights(stage)
            require(all(name in weights for name in names), 'stage weight missing')
            job = dict(row=row, part=part, row_in_part=part_row, stage=stage,
                       input_origin='frozen_saved_pre_input', dtype='<f4', order='C',
                       input_shape=[400 if stage == 0 else contract.DIMS[stage - 1]],
                       output_shape=[output_width], input_bytes=binding['bytes'],
                       input_sha256=binding['sha256'], input_binding=binding, weight_names=list(names))
            if stage in contract.MEMORY_STAGES:
                job.update(cache_origin='C_SAVED_OBSERVED', cache_shape=[128, 11, 4],
                           cache_bytes=cache['bytes'], cache_sha256=cache['sha256'], cache_binding=cache,
                           cache_state='before_row_update', cache_layer=(stage - 4) // 4)
            for backend in ('C', 'Torch'):
                plan['jobs'].append(dict(job, backend=backend))
    contract.validate(plan)
    result = bytes(payload)
    # Independently dereference each generated pair, including offsets, instead
    # of accepting the paired digest strings as proof of shared source bytes.
    for first, second in zip(plan['jobs'][::2], plan['jobs'][1::2]):
        for name in ('input_binding', 'cache_binding'):
            if name not in first:
                continue
            a, b = first[name], second[name]
            left = result[a['bundle_offset']:a['bundle_offset'] + a['bytes']]
            right = result[b['bundle_offset']:b['bundle_offset'] + b['bytes']]
            require(a == b and left == right and len(left) == a['bytes'] and _sha(left) == a['sha256'],
                    'backend saved-byte binding mismatch')
    return plan, result


def bind_saved(root):
    """Authenticate exactly four saved files and return a plan plus opaque bytes.

    There is no execute mode, dependency import, library load or source override.
    This does not prove that a future backend actually consumed these bytes.
    """
    files = {name: _read_frozen(Path(root) / path, size, contract.FROZEN_SOURCES[name])
             for name, (path, size) in SOURCE_FILES.items()}
    # No archive/header/JSON interpretation occurs before all four hashes pass.
    c_arrays, c_identities = _arrays(files['c_npz'], 'c_npz')
    _, torch_identities = _arrays(files['torch_npz'], 'torch_npz')
    weights = _weights(json.loads(files['export_manifest'].decode('utf-8')), files['weights'])
    plan, payload = _build_plan(c_arrays, files['weights'], weights)
    report = dict(schema='d20-saved-input-binding-v1', saved_byte_identity_verified=True,
                  saved_backend_pair_bytes_equal=True, backend_dispatch_identity_verified=False,
                  cmvn_binding='C saved cmvn output; not input or cmvn_same; no recomputation',
                  cmvn_observation='saved_separate_call_same_c_function_and_input',
                  historical_stage0_input_hook_recorded=False,
                  cache_binding='C saved cache_before per row; no Torch reconstruction',
                  input_chaining=False, historical_chunk_equivalence=False,
                  original_status={'D20': 'FAIL', 'D90': 'NOT_RUN'},
                  export_symbols=['<blank>', '你', '好', '小', '窝', '屋'],
                  planned_rows=len(plan['jobs']), numerical_imports=0, model_calls=0, operator_calls=0,
                  numerical_admission=False, execution_ready=False,
                  blockers=['historical_c_source_build_fp_binding_unverified',
                            'single_stage_wrappers_and_dispatch_unverified',
                            'exact_backend_isolated_import_and_threading_unverified',
                            'hard_memory_cpu_network_gpu_scope_unverified',
                            'independent_numerical_acceptance_missing', 'new_numerical_authorization_required'],
                  sources={name: dict(logical_path=path, bytes=size, sha256=contract.FROZEN_SOURCES[name])
                           for name, (path, size) in SOURCE_FILES.items()},
                  arrays={'c_npz': c_identities, 'torch_npz': torch_identities}, weights=weights,
                  payload=dict(file='inputs.bin', bytes=len(payload), sha256=_sha(payload)), plan=plan)
    return report, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, payload = bind_saved(args.saved_root)
    # No overwrite, no public upload. The binary contains saved model/input bytes.
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    with (args.output / 'inputs.bin').open('xb') as target:
        target.write(payload)
    serialized = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with (args.output / 'binding.json').open('xb') as target:
        target.write(serialized)
    # Verify the written files, not only the in-memory objects. Nothing here
    # accepts a caller-authored manifest as evidence of authenticity.
    _read_frozen(args.output / 'inputs.bin', len(payload), _sha(payload))
    _read_frozen(args.output / 'binding.json', len(serialized), _sha(serialized))
    summary = {name: report[name] for name in ('schema', 'planned_rows', 'saved_byte_identity_verified',
               'backend_dispatch_identity_verified', 'execution_ready', 'numerical_admission',
               'model_calls', 'operator_calls')}
    summary.update(output_bytes_verified=True, payload=report['payload'],
                   manifest=dict(file='binding.json', bytes=len(serialized), sha256=_sha(serialized)))
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
