"""Separate v2 conformance harness. No execution without exact review and launch.

The existing v1 Stage B and canonical failures are immutable historical results.
This module changes no model arithmetic and performs no audio/model downloads.
Imports are inert: only main() may load a checkpoint or execute model kernels.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
DIMS = [140, 250, 250] + [128, 128, 250, 250] * 4 + [140, 6]
AFFINES = [(0, 'backbone.in_linear1', 'cmvn', 400, 140),
           (1, 'backbone.in_linear2', 'stage0', 140, 250)]
for _i in range(4):
    AFFINES.extend([(3 + 4 * _i, f'backbone.fsmn.{_i}.0', f'stage{2 + 4 * _i}', 250, 128),
                    (5 + 4 * _i, f'backbone.fsmn.{_i}.2', f'stage{4 + 4 * _i}', 128, 250)])
AFFINES.extend([(19, 'backbone.out_linear1', 'stage18', 250, 140),
                (20, 'backbone.out_linear2', 'stage19', 140, 6)])
SOURCE = BASE / 'kws-alternate-fsmn-precheck-v1/reference/upstream/wekws/model'
HOOK_LIBRARY = ROOT / 'v2/harness/build/liba20_local_hooks.so'
MODEL_LIBRARY = ROOT / 'build/liba20.so'
DECODER_LIBRARY = ROOT / 'decoder/build/liba20decoder.so'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def specification():
    return {
        'fixtures': {
            'normalized_dyadic': {'generator': 'PCG64', 'seed': 20261003, 'shape': [31, 400],
                         'integer_inclusive_range': [-64, 64], 'divisor': 64,
                         'raw_conversion': 'FP32 z; FP32 divide by istd; FP32 add mean'},
            'feature_uniform': {'generator': 'PCG64', 'seed': 3102026, 'shape': [37, 400],
                          'uniform_range': [-16, 24], 'raw_conversion': 'float32'},
            'impulse_family': {'shape': [19, 400], 'rows': [0, 3, 6, 9, 12],
                          'columns': [0, 37, 139, 249, 399], 'values': [2, -2, 2, -2, 2],
                          'raw_conversion': 'FP32 z; FP32 divide by istd; FP32 add mean'}},
        'split_parts': {'normalized_dyadic': [1, 3, 9, 18], 'feature_uniform': [1, 3, 9, 8, 16],
                        'impulse_family': [1, 3, 7, 8]},
        'fixture_row_identities': 87, 'reference_route_rows': 174, 'native_route_rows': 174,
        'gates': {'logit_atol': 1e-4, 'logit_rtol': 1e-5, 'probability_atol': 1e-5,
                  'rowsum_atol': 1e-5, 'cmvn_atol': 2e-4, 'event_score_atol': 1e-5,
                  'event_discrete_fields': 'exact',
                  'local_affine_gamma': 'n+1', 'local_memory_gamma': 14,
                  'u32': 2.0 ** -24, 'u64': 2.0 ** -53,
                  'affine_underflow_operations': '2*n+1', 'memory_underflow_operations': 27,
                  'underflow_quantum': 2.0 ** -149},
        'reference_convention': 'official whole and predeclared split chunk routes',
        'historical_stage_b': 'FAIL_UNCHANGED', 'historical_canonical': 'FAIL_UNCHANGED',
        'audio_reads': 0, 'training_updates': 0,
    }


def required_bindings():
    local = ['v2/harness/fixtures.py', 'v2/harness/local_hooks.c', 'v2/harness/build_hooks.py',
             'v2/harness/build/liba20_local_hooks.so', 'v2/harness/build/receipt.json',
             'v2/NUMERICAL-CONTRACT.md', 'v2/fixtures.npz', 'v2/fixtures-manifest.json',
             'v2/prepare_inputs.py', 'v2/harness/events.py', 'v2/harness/unit_math.py', 'native/model/a20_fsmn.c', 'native/model/a20_fsmn.h',
             'native/model/load.c', 'native/model/sha256.c', 'native/model/sha256.h',
             'build/liba20.so', 'decoder/a20_decoder.c', 'decoder/a20_decoder.h',
             'decoder/build/liba20decoder.so', 'export/a20.f32', 'export/manifest.json',
             'export/a20_identity.h', 'export_a20.py', 'evidence/stage-b/report.json',
             'evidence/canonical/report.json', 'review/CANONICAL-OUTPUT-REVIEW.json',
             'evidence/stage-a/acceptance.json']
    files = [ROOT / p for p in local] + [SOURCE / 'fsmn.py', SOURCE / 'cmvn.py',
             BASE / 'kws-pretrained-encoder-adaptation-v1/training-run/A-terminal.reference.pt',
             BASE / 'kws-alternate-fsmn-precheck-v1/reference/upstream/wekws/bin/stream_kws_ctc.py',
             BASE / 'kws-alternate-fsmn-precheck-v1/reference/run_baseline.py',
             BASE / 'kws-alternate-fsmn-precheck-v1/decoder_tail_fix.py']
    return [str(p.relative_to(BASE)) for p in files]


def verify_bindings(protocol):
    assert set(required_bindings()) <= set(protocol['bindings']), 'missing required identity binding'
    for name, digest in protocol['bindings'].items():
        path = (BASE / name).resolve()
        assert path.is_relative_to(BASE) and len(digest) == 64
        assert sha(path) == digest, f'binding changed: {name}'


def preflight(args):
    assert args.protocol.resolve() == ROOT / 'v2/protocol.json'
    assert args.review.resolve() == ROOT / 'review/V2-REVIEW.json'
    assert args.launch.resolve() == ROOT / 'v2/launch.json'
    assert args.out.resolve() == ROOT / 'v2/evidence'
    protocol = json.loads(args.protocol.read_bytes())
    review = json.loads(args.review.read_bytes())
    launch = json.loads(args.launch.read_bytes())
    assert protocol['schema'] == 'a20-native-local-rounding-strict-final-v2'
    assert protocol['experiment'] == specification(), 'experiment differs from fixed harness contract'
    assert review['approved'] is True and review['protocol_sha256'] == sha(args.protocol)
    assert launch['authorized'] is True and launch['protocol_sha256'] == sha(args.protocol)
    assert launch['review_sha256'] == sha(args.review)
    assert launch['pid'] == os.getpid(), 'launch must name this child PID'
    assert launch['output_directory'] == str(args.out.resolve())
    assert os.path.abspath(sys.executable) == protocol['runtime']['python']
    assert 0 < protocol['max_cpu_seconds'] <= 120
    verify_bindings(protocol)
    assert json.loads((ROOT / 'evidence/stage-a/acceptance.json').read_bytes())['passed'] is True
    return protocol


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_fixtures(np, state):
    mean = state['global_cmvn.mean'].numpy()
    istd = state['global_cmvn.istd'].numpy()
    z = np.random.Generator(np.random.PCG64(20261003)).integers(-64, 65, (31, 400), dtype=np.int64).astype(np.float32)
    z = np.divide(z, np.float32(64), dtype=np.float32)
    derived = {'normalized_dyadic': np.add(mean, np.divide(z, istd, dtype=np.float32), dtype=np.float32),
              'feature_uniform': np.random.Generator(np.random.PCG64(3102026)).uniform(-16, 24, (37, 400)).astype(np.float32)}
    z = np.zeros((19, 400), np.float32)
    z[[0, 3, 6, 9, 12], [0, 37, 139, 249, 399]] = [2, -2, 2, -2, 2]
    derived['impulse_family'] = np.add(mean, np.divide(z, istd, dtype=np.float32), dtype=np.float32)
    manifest = json.loads((ROOT / 'v2/fixtures-manifest.json').read_bytes())
    assert manifest['fixture_sha256'] == sha(ROOT / 'v2/fixtures.npz')
    assert manifest['rows'] == 87 and manifest['whole_network_forwards'] == 0
    with np.load(ROOT / 'v2/fixtures.npz', allow_pickle=False) as frozen:
        assert set(frozen.files) == set(derived)
        inputs = {key: frozen[key].copy() for key in derived}
    for key, value in inputs.items():
        exact(np, value, derived[key], key + '_independently_rederived_frozen_input')
        assert hashlib.sha256(value.tobytes()).hexdigest() == manifest['arrays'][key]['raw_f32le_sha256']
    assert sum(len(x) for x in inputs.values()) == 87
    assert len({row.tobytes() for value in inputs.values() for row in value}) == manifest['numerically_distinct_vectors'] == 74
    assert all(x.shape[1] == 400 and x.dtype == np.float32 and np.isfinite(x).all() for x in inputs.values())
    return inputs


def exact(np, left, right, name):
    assert left.shape == right.shape and left.dtype == right.dtype, f'{name}: shape or dtype'
    assert np.isfinite(left).all() and np.isfinite(right).all(), f'{name}: nonfinite'
    assert left.tobytes() == right.tobytes(), f'{name}: structural bitwise mismatch'


def cache_sequence(np, before, stages):
    cache = before.copy()
    previous, after = [], []
    for row in range(len(stages[3])):
        previous.append(cache.copy())
        cache[:, :-1, :] = cache[:, 1:, :].copy()
        for layer in range(4):
            cache[:, -1, layer] = stages[3 + 4 * layer][row]
        after.append(cache.copy())
    return np.stack(previous), np.stack(after), cache


def freeze_reference(np, torch, state, inputs, out):
    fsmn = module('a20_v2_official_fsmn', SOURCE / 'fsmn.py')
    cmvn = module('a20_v2_official_cmvn', SOURCE / 'cmvn.py')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_cmvn = cmvn.GlobalCMVN(state['global_cmvn.mean'], state['global_cmvn.istd'], True)
            self.backbone = fsmn.FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 6)
            self.load_state_dict(state, strict=True)

    model = Model().eval()
    arrays = {name + '_input': value for name, value in inputs.items()}
    calls, row_count = [], 0
    with torch.no_grad():
        for name, value in inputs.items():
            for route, parts in [('whole', [len(value)]), ('split', specification()['split_parts'][name])]:
                cache, offset = None, 0
                assert sum(parts) == len(value)
                for chunk, count in enumerate(parts):
                    prefix = f'{name}_{route}_{chunk}'
                    before = np.zeros((128, 11, 4), np.float32) if cache is None else cache.numpy()[0].copy()
                    x = model.global_cmvn(torch.from_numpy(value[offset:offset + count]).unsqueeze(0))
                    arrays[prefix + '_cmvn'] = x.numpy()[0].copy()
                    caches = [None] * 4 if cache is None else [cache[:, :, :, i:i + 1] for i in range(4)]
                    stages = []
                    for layer in [model.backbone.in_linear1, model.backbone.in_linear2, model.backbone.relu]:
                        x, _ = layer(x)
                        stages.append(x.numpy()[0].copy())
                    for i, block in enumerate(model.backbone.fsmn):
                        x, _ = block[0](x)
                        stages.append(x.numpy()[0].copy())
                        x, caches[i] = block[1]((x, caches[i]))
                        stages.append(x.numpy()[0].copy())
                        x, _ = block[2](x)
                        stages.append(x.numpy()[0].copy())
                        x, _ = block[3](x)
                        stages.append(x.numpy()[0].copy())
                    for layer in [model.backbone.out_linear1, model.backbone.out_linear2]:
                        x, _ = layer(x)
                        stages.append(x.numpy()[0].copy())
                    cache = torch.cat(caches, dim=-1)
                    for stage, values in enumerate(stages):
                        assert values.shape == (count, DIMS[stage]) and np.isfinite(values).all()
                        arrays[prefix + f'_stage{stage}'] = values
                    for stage in [2, 6, 10, 14, 18]:
                        exact(np, stages[stage], np.maximum(stages[stage - 1], np.float32(0)), prefix + f'_reference_relu{stage}')
                    previous, after, final = cache_sequence(np, before, stages)
                    exact(np, final, cache.numpy()[0], prefix + '_reference_cache_reconstruction')
                    arrays[prefix + '_cache_before'] = before
                    arrays[prefix + '_cache_previous_rows'] = previous
                    arrays[prefix + '_cache_rows'] = after
                    arrays[prefix + '_cache'] = cache.numpy()[0].copy()
                    arrays[prefix + '_probabilities'] = torch.softmax(x, dim=-1).numpy()[0].copy()
                    calls.append({'prefix': prefix, 'input': name, 'route': route, 'offset': offset,
                                  'frames': count, 'reset': chunk == 0})
                    offset += count
                    row_count += count
    assert row_count == 174 and len(calls) == 16
    with (out / 'reference.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    return arrays, calls, row_count


def gamma(np, count, unit):
    # count*unit is exact here (small integer times a power of two), but
    # explicitly enclose it so this helper remains valid if called more widely.
    numerator = np.nextafter(count * unit, np.inf)
    denominator = np.nextafter(1.0 - numerator, -np.inf)
    assert denominator > 0
    return float(np.nextafter(numerator / denominator, np.inf))


def oracle_bounds(np, products, residual, count, underflow_ops):
    """FP32 products are exactly representable in FP64, including subnormal inputs.

    The positive L1 sum is outward-enclosed before either error bound is formed.
    count is n+1 for affine and 14 for the thirteen-term memory expression.
    The largest possible intermediate absolute value is <= exact L1, so the
    conservative safety check below excludes overflow in all FP32 associations.
    """
    assert products.dtype == np.float64 and residual.dtype == np.float64
    assert np.isfinite(products).all() and np.isfinite(residual).all()
    g32, g64 = gamma(np, count, 2.0 ** -24), gamma(np, count, 2.0 ** -53)
    oracle = products.sum(axis=-1, dtype=np.float64) + residual
    l1_computed = np.abs(products).sum(axis=-1, dtype=np.float64) + np.abs(residual)
    denominator = np.nextafter(1.0 - g64, -np.inf)
    l1_upper = np.nextafter(l1_computed / denominator, np.inf)
    b32 = np.nextafter(g32 * l1_upper, np.inf)
    b64 = np.nextafter(g64 * l1_upper, np.inf)
    bound = np.nextafter(np.nextafter(b32 + b64, np.inf) + np.nextafter(underflow_ops * 2.0 ** -149, np.inf), np.inf)
    assert np.isfinite(oracle).all() and np.isfinite(bound).all()
    assert (l1_upper < np.finfo(np.float32).max / 4.0).all(), 'possible FP32 overflow not covered'
    tiny = np.finfo(np.float32).tiny
    diagnostics = {'product_subnormal_count': int(((np.abs(products) > 0) & (np.abs(products) < tiny)).sum()),
                   'exact_oracle_subnormal_count': int(((np.abs(oracle) > 0) & (np.abs(oracle) < tiny)).sum())}
    return oracle, l1_upper, bound, diagnostics


def freeze_oracles(np, state, reference, calls, out):
    arrays, counts = {}, {'product_subnormal_count': 0, 'exact_oracle_subnormal_count': 0,
                          'operand_subnormal_count': 0}
    for tensor in state.values():
        a = tensor.numpy()
        counts['operand_subnormal_count'] += int(((np.abs(a) > 0) & (np.abs(a) < np.finfo(np.float32).tiny)).sum())
    for call in calls:
        prefix = call['prefix']
        for stage, stem, previous, ni, no in AFFINES:
            x = reference[prefix + '_' + previous].astype(np.float64)
            w = state[stem + '.linear.weight'].numpy().astype(np.float64)
            b = state.get(stem + '.linear.bias')
            b = np.zeros(no, np.float64) if b is None else b.numpy().astype(np.float64)
            products = x[:, None, :] * w[None, :, :]
            oracle, l1, bound, diag = oracle_bounds(np, products, np.broadcast_to(b, (len(x), no)), ni + 1, 2 * ni + 1)
            key = prefix + f'_affine{stage}'
            arrays[key + '_oracle'], arrays[key + '_l1_upper'], arrays[key + '_bound'] = oracle, l1, bound
            for k, value in diag.items():
                counts[k] += value
        cache = reference[prefix + '_cache_previous_rows'].astype(np.float64)
        for layer in range(4):
            projection = reference[prefix + f'_stage{3 + 4 * layer}'].astype(np.float64)
            left = state[f'backbone.fsmn.{layer}.1.conv_left.weight'].numpy().reshape(128, 10).astype(np.float64)
            right = state[f'backbone.fsmn.{layer}.1.conv_right.weight'].numpy().reshape(128, 2).astype(np.float64)
            products = np.concatenate([cache[:, :, :10, layer] * left[None, :, :],
                                      (cache[:, :, 10, layer] * right[None, :, 0])[:, :, None],
                                      (projection * right[None, :, 1])[:, :, None]], axis=-1)
            oracle, l1, bound, diag = oracle_bounds(np, products, cache[:, :, 9, layer], 14, 27)
            key = prefix + f'_memory{4 + 4 * layer}'
            arrays[key + '_oracle'], arrays[key + '_l1_upper'], arrays[key + '_bound'] = oracle, l1, bound
            for k, value in diag.items():
                counts[k] += value
    for key, a in reference.items():
        if a.dtype == np.float32:
            counts['operand_subnormal_count'] += int(((np.abs(a) > 0) & (np.abs(a) < np.finfo(np.float32).tiny)).sum())
    with (out / 'oracle.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    return arrays, counts


def compare(np, checks, name, actual, reference, limit, gate):
    assert actual.shape == reference.shape, name + ': dimension mismatch'
    assert np.isfinite(actual).all() and np.isfinite(reference).all(), name + ': nonfinite'
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    limit = np.broadcast_to(np.asarray(limit, np.float64), difference.shape)
    assert np.isfinite(limit).all() and (limit > 0).all()
    bad = difference > limit
    record = {'name': name, 'gate': gate, 'passed': not bool(bad.any()), 'elements': int(difference.size),
              'failed_elements': int(bad.sum()), 'max_abs': float(difference.max(initial=0)),
              'max_ratio': float((difference / limit).max(initial=0))}
    if bad.any():
        index = tuple(int(x) for x in np.argwhere(bad)[0])
        record['first_failure'] = {'index': list(index), 'actual': float(actual[index]),
                                   'reference': float(reference[index]), 'abs_error': float(difference[index]),
                                   'bound': float(limit[index])}
    checks.append(record)


def historical_status():
    old = json.loads((ROOT / 'evidence/stage-b/report.json').read_bytes())
    failed = [x for x in old['checks'] if not x['passed']]
    canonical = json.loads((ROOT / 'evidence/canonical/report.json').read_bytes())
    review = json.loads((ROOT / 'review/CANONICAL-OUTPUT-REVIEW.json').read_bytes())
    assert old['passed'] is False and len(failed) == 65
    assert sum('_torch_partition_' in x['name'] for x in failed) == 8
    assert canonical['native_canonical_gate_passed'] is False
    assert review['native_canonical_failed_stage_elements'] == 1
    return {'stage_b': 'FAIL_UNCHANGED', 'stage_b_native_failed_checks': 57,
            'stage_b_torch_partition_failed_checks': 8, 'canonical': 'FAIL_UNCHANGED',
            'canonical_failed_stage_elements': 1, 'canonical_failed_perrow_cache_elements': 11,
            'interpretation': 'v2 is an explicitly different prospective contract; no historical result is promoted'}


def run(args, protocol):
    start = time.monotonic()
    import numpy as np
    import torch
    assert np.__version__ == protocol['runtime']['numpy']
    assert torch.__version__ == protocol['runtime']['torch']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # Verify gradual underflow without silently changing the floating environment.
    tiny = torch.tensor([2.0 ** -149, 2.0 ** -126], dtype=torch.float32)
    assert torch.equal((tiny * torch.tensor([1., .5])).view(torch.int32),
                       torch.tensor([1, 0x00400000], dtype=torch.int32))
    exporter = module('a20_v2_export', ROOT / 'export_a20.py')
    state, payload, _ = exporter.read_state()
    assert (ROOT / 'export/a20.f32').read_bytes() == payload
    assert hashlib.sha256(payload).hexdigest() == 'a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d'
    inputs = make_fixtures(np, state)
    arrays, calls, reference_rows = freeze_reference(np, torch, state, inputs, args.out)
    oracles, subnormals = freeze_oracles(np, state, arrays, calls, args.out)
    events = module('a20_v2_events', ROOT / 'v2/harness/events.py')
    reference_events = events.freeze_reference_events(arrays, calls)
    write_json(args.out / 'reference-events.json', reference_events)
    write_json(args.out / 'reference-freeze.json', {
        'protocol_sha256': sha(args.protocol), 'review_sha256': sha(args.review),
        'raw_fixtures_sha256': sha(ROOT / 'v2/fixtures.npz'), 'reference_events_sha256': sha(args.out / 'reference-events.json'),
        'reference_sha256': sha(args.out / 'reference.npz'), 'oracle_sha256': sha(args.out / 'oracle.npz'),
        'calls': calls, 'fixture_row_identities': 87, 'reference_route_rows': reference_rows,
        'distinct_numerical_vectors': len({row.tobytes() for value in inputs.values() for row in value}),
        'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in inputs.items()},
        'state_sha256': exporter.STATE_SHA, 'candidate_full_rows_before_freeze': 0,
        'candidate_local_calls_before_freeze': 0, 'all_reference_tensors_frozen_before_native': True})
    # No native library is loaded until every official stage/cache and oracle is persisted.
    fp = C.POINTER(C.c_float)
    class Native(C.Structure):
        _fields_ = [('weights', fp), ('cache', C.c_float * 5632), ('fault', C.c_int)]
    model = C.CDLL(str(MODEL_LIBRARY))
    hook = C.CDLL(str(HOOK_LIBRARY))
    decoder = C.CDLL(str(DECODER_LIBRARY))
    model.a20_init_verified.argtypes = [C.POINTER(Native), fp, C.c_size_t]
    model.a20_step.argtypes = [C.POINTER(Native), fp, fp, fp]
    model.a20_cmvn.argtypes = [C.POINTER(Native), fp, fp]
    model.a20_reset.argtypes = [C.POINTER(Native)]
    model.a20_reset.restype = None
    hook.a20v2_affine.argtypes = [fp, fp, fp, fp, C.c_int, C.c_int]
    hook.a20v2_memory.argtypes = [C.POINTER(Native), C.c_int, fp, fp]
    hook.a20v2_fp_environment.argtypes = []
    decoder.a20d_softmax6.argtypes = [fp, fp]
    assert hook.a20v2_fp_environment() == 1, 'requires IEEE nearest and gradual underflow'
    ptr = lambda value: value.ctypes.data_as(fp)
    weights = np.frombuffer(payload, dtype='<f4').copy()
    native, local = Native(), Native()
    assert model.a20_init_verified(C.byref(native), ptr(weights), weights.size) == 0
    assert model.a20_init_verified(C.byref(local), ptr(weights), weights.size) == 0
    checks, native_arrays, local_arrays = [], {}, {}
    native_rows = affine_calls = memory_calls = 0
    native_event_runner = events.NativeEvents(decoder)
    native_event_records = []
    for call in calls:
        prefix, count = call['prefix'], call['frames']
        if call['reset']:
            model.a20_reset(C.byref(native))
            native_event_runner.reset()
            assert bytes(native.cache) == bytes(5632 * 4) and native.fault == 0
        before = np.ctypeslib.as_array(native.cache).reshape(128, 11, 4).copy()
        stages = [[] for _ in DIMS]
        zs, probs, caches = [], [], []
        for x in inputs[call['input']][call['offset']:call['offset'] + count]:
            y, trace, z, probability = np.zeros(6, np.float32), np.zeros((21, 250), np.float32), np.zeros(400, np.float32), np.zeros(6, np.float32)
            assert model.a20_cmvn(C.byref(native), ptr(x), ptr(z)) == 0
            assert model.a20_step(C.byref(native), ptr(x), ptr(y), ptr(trace)) == 0
            assert decoder.a20d_softmax6(ptr(y), ptr(probability)) == 0
            exact(np, y, trace[20, :6], prefix + '_returned_logits')
            for stage, dim in enumerate(DIMS):
                stages[stage].append(trace[stage, :dim].copy())
            zs.append(z)
            probs.append(probability)
            caches.append(np.ctypeslib.as_array(native.cache).reshape(128, 11, 4).copy())
            native_rows += 1
        stages = [np.stack(values) for values in stages]
        for stage in [2, 6, 10, 14, 18]:
            exact(np, stages[stage], np.maximum(stages[stage - 1], np.float32(0)), prefix + f'_native_relu{stage}')
        native_previous, reconstructed, final = cache_sequence(np, before, stages)
        exact(np, reconstructed, np.stack(caches), prefix + '_native_perrow_cache_reconstruction')
        exact(np, final, caches[-1], prefix + '_native_final_cache_reconstruction')
        native_arrays[prefix + '_cache_previous_rows'] = native_previous
        native_arrays[prefix + '_cache_rows'] = np.stack(caches)
        native_arrays[prefix + '_cache'] = caches[-1]
        native_arrays[prefix + '_cmvn'] = np.stack(zs)
        native_arrays[prefix + '_probabilities'] = np.stack(probs)
        for stage, value in enumerate(stages):
            native_arrays[prefix + f'_stage{stage}'] = value
            reference = arrays[prefix + f'_stage{stage}']
            compare(np, checks, prefix + f'_propagated_stage{stage}', value, reference,
                    1e-4 + 1e-5 * np.abs(reference.astype(np.float64)), 'diagnostic_old_B')
        compare(np, checks, prefix + '_propagated_cache_rows', np.stack(caches), arrays[prefix + '_cache_rows'],
                1e-4 + 1e-5 * np.abs(arrays[prefix + '_cache_rows'].astype(np.float64)), 'diagnostic_old_B')
        compare(np, checks, prefix + '_propagated_cache', caches[-1], arrays[prefix + '_cache'],
                1e-4 + 1e-5 * np.abs(arrays[prefix + '_cache'].astype(np.float64)), 'diagnostic_old_B')
        for suffix, limit in [('cmvn', 2e-4), ('probabilities', 1e-5),
                              ('stage20', 1e-4 + 1e-5 * np.abs(arrays[prefix + '_stage20'].astype(np.float64)))]:
            compare(np, checks, prefix + '_strict_' + suffix, native_arrays[prefix + '_' + suffix], arrays[prefix + '_' + suffix], limit, 'hard_final')
        for engine, value in [('native', native_arrays), ('reference', arrays)]:
            probabilities = value[prefix + '_probabilities']
            assert probabilities.shape == (count, 6) and ((probabilities >= 0) & (probabilities <= 1)).all()
            compare(np, checks, prefix + '_' + engine + '_rowsum', probabilities.astype(np.float64).sum(axis=1), np.ones(count), 1e-5, 'hard_final')
        event_record = native_event_runner.process(native_arrays[prefix + '_stage20'])
        native_event_records.append({'prefix': prefix, **event_record})
        events.compare_events(reference_events[len(native_event_records) - 1], event_record)
        # Independent local tests use the exact same frozen official input tensors.
        for stage, stem, previous, ni, no in AFFINES:
            x = arrays[prefix + '_' + previous]
            w = state[stem + '.linear.weight'].numpy()
            b = state.get(stem + '.linear.bias')
            b = None if b is None else b.numpy()
            result = np.zeros((count, no), np.float32)
            own_result = np.zeros_like(result)
            own_input = native_arrays[prefix + '_' + previous]
            for index in range(count):
                assert hook.a20v2_affine(ptr(own_input[index]), ptr(own_result[index]), ptr(w), None if b is None else ptr(b), ni, no) == 0
                affine_calls += 1
                assert hook.a20v2_affine(ptr(x[index]), ptr(result[index]), ptr(w), None if b is None else ptr(b), ni, no) == 0
                affine_calls += 1
            exact(np, own_result, native_arrays[prefix + f'_stage{stage}'], prefix + f'_own_input_affine{stage}')
            key = prefix + f'_affine{stage}'
            local_arrays[key + '_own_native_reproduction'] = own_result
            local_arrays[key + '_native'] = result
            for engine, value in [('native', result), ('reference', arrays[prefix + f'_stage{stage}'])]:
                compare(np, checks, key + '_' + engine, value, oracles[key + '_oracle'], oracles[key + '_bound'], 'hard_local')
        for layer in range(4):
            result = np.zeros((count, 128), np.float32)
            own_result = np.zeros_like(result)
            for index in range(count):
                own_before = native_previous[index].copy()
                C.memmove(C.addressof(local.cache), own_before.ctypes.data, own_before.nbytes)
                own_projection = native_arrays[prefix + f'_stage{3 + 4 * layer}'][index]
                assert hook.a20v2_memory(C.byref(local), layer, ptr(own_projection), ptr(own_result[index])) == 0
                own_expected = own_before.copy()
                own_expected[:, :-1, layer] = own_before[:, 1:, layer]
                own_expected[:, -1, layer] = own_projection
                exact(np, np.ctypeslib.as_array(local.cache).reshape(128, 11, 4), own_expected, prefix + f'_own_memory{layer}_cache')
                memory_calls += 1
                before = arrays[prefix + '_cache_previous_rows'][index].copy()
                C.memmove(C.addressof(local.cache), before.ctypes.data, before.nbytes)
                projection = arrays[prefix + f'_stage{3 + 4 * layer}'][index]
                assert hook.a20v2_memory(C.byref(local), layer, ptr(projection), ptr(result[index])) == 0
                expected = before.copy()
                expected[:, :-1, layer] = before[:, 1:, layer]
                expected[:, -1, layer] = projection
                exact(np, np.ctypeslib.as_array(local.cache).reshape(128, 11, 4), expected, prefix + f'_local_memory{layer}_cache')
                memory_calls += 1
            exact(np, own_result, native_arrays[prefix + f'_stage{4 + 4 * layer}'], prefix + f'_own_input_memory{layer}')
            key = prefix + f'_memory{4 + 4 * layer}'
            local_arrays[key + '_own_native_reproduction'] = own_result
            local_arrays[key + '_native'] = result
            for engine, value in [('native', result), ('reference', arrays[prefix + f'_stage{4 + 4 * layer}'])]:
                compare(np, checks, key + '_' + engine, value, oracles[key + '_oracle'], oracles[key + '_bound'], 'hard_local')
    assert reference_rows == native_rows == 174 and affine_calls == 4176 and memory_calls == 1392
    for name in inputs:
        split = [call for call in calls if call['input'] == name and call['route'] == 'split']
        for suffix in ['cmvn', 'probabilities', 'cache_rows'] + [f'stage{stage}' for stage in range(21)]:
            joined_native = np.concatenate([native_arrays[call['prefix'] + '_' + suffix] for call in split], axis=0)
            exact(np, joined_native, native_arrays[name + '_whole_0_' + suffix], name + '_native_partition_' + suffix)
            joined_reference = np.concatenate([arrays[call['prefix'] + '_' + suffix] for call in split], axis=0)
            whole_reference = arrays[name + '_whole_0_' + suffix]
            limit = 2e-4 if suffix == 'cmvn' else 1e-5 if suffix == 'probabilities' else 1e-4 + 1e-5 * np.abs(whole_reference.astype(np.float64))
            compare(np, checks, name + '_torch_partition_' + suffix, joined_reference, whole_reference, limit,
                    'hard_final_partition' if suffix in ['cmvn', 'probabilities', 'stage20'] else 'diagnostic_reference_partition')
        exact(np, native_arrays[split[-1]['prefix'] + '_cache'], native_arrays[name + '_whole_0_cache'], name + '_native_partition_final_cache')
        compare(np, checks, name + '_torch_partition_final_cache', arrays[split[-1]['prefix'] + '_cache'], arrays[name + '_whole_0_cache'],
                1e-4 + 1e-5 * np.abs(arrays[name + '_whole_0_cache'].astype(np.float64)), 'diagnostic_reference_partition')
    model.a20_reset(C.byref(native))
    assert bytes(native.cache) == bytes(5632 * 4) and native.fault == 0
    assert hook.a20v2_fp_environment() == 1
    with (args.out / 'native.npz').open('xb') as stream:
        np.savez(stream, **native_arrays)
    with (args.out / 'local.npz').open('xb') as stream:
        np.savez(stream, **local_arrays)
    write_json(args.out / 'native-events.json', native_event_records)
    subnormals['native_output_subnormal_count'] = sum(int(((np.abs(value) > 0) & (np.abs(value) < np.finfo(np.float32).tiny)).sum()) for value in native_arrays.values())
    subnormals['local_output_subnormal_count'] = sum(int(((np.abs(value) > 0) & (np.abs(value) < np.finfo(np.float32).tiny)).sum()) for value in local_arrays.values())
    verify_bindings(protocol)
    hard = [row for row in checks if row['gate'].startswith('hard_')]
    report = {'schema': 'a20-v2-local-rounding-strict-final-report-v1',
              'passed': all(row['passed'] for row in hard), 'checks': checks,
              'historical': historical_status(), 'protocol_sha256': sha(args.protocol),
              'review_sha256': sha(args.review), 'launch_sha256': sha(args.launch),
              'files_sha256': {name: sha(args.out / name) for name in ['reference.npz', 'oracle.npz', 'reference-freeze.json', 'reference-events.json', 'native-events.json', 'native.npz', 'local.npz']},
              'fixture_row_identities': 87, 'distinct_numerical_vectors': len({row.tobytes() for value in inputs.values() for row in value}),
              'reference_whole_network_route_rows': reference_rows, 'native_whole_network_route_rows': native_rows,
              'native_local_affine_calls': affine_calls, 'native_local_memory_calls': memory_calls,
              'native_own_input_affine_calls': affine_calls // 2, 'native_gold_input_affine_calls': affine_calls // 2,
              'native_own_input_memory_calls': memory_calls // 2, 'native_gold_input_memory_calls': memory_calls // 2,
              'extra_reference_local_forwards': 0, 'own_input_hook_reproduction_bitexact': True,
              'event_fields_exact_score_atol_1e5': True, 'reference_decoder_chunks': len(calls), 'native_decoder_chunks': len(calls), 'all_reference_tensors_frozen_before_native': True,
              'native_partition_bitexact': True, 'native_perrow_cache_reconstruction_bitexact': True,
              'reference_cache_reconstruction_bitexact': True, 'relu_structure_bitexact': True,
              'local_memory_cache_copy_bitexact': True, 'reset_cache_bitwise_zero': True,
              'subnormal_diagnostics': subnormals, 'fp32_overflow_excluded_by_l1': True,
              'round_to_nearest_gradual_underflow_verified': True,
              'hard_failed_checks': sum(not row['passed'] for row in hard),
              'diagnostic_old_B_failed_checks': sum(not row['passed'] for row in checks if row['gate'] == 'diagnostic_old_B'),
              'diagnostic_reference_partition_failed_checks': sum(not row['passed'] for row in checks if row['gate'] == 'diagnostic_reference_partition'),
              'audio_reads': 0, 'training_updates': 0, 'actual12_approved': False, 'benchmark_approved': False,
              'wall_seconds': time.monotonic() - start, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'scope': 'Prospective synthetic local-kernel conformance and strict final outputs only; no universal, intermediate-pointwise, acoustic, or product qualification'}
    write_json(args.out / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, sort_keys=True))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'v2/protocol.json')
    parser.add_argument('--review', type=Path, default=ROOT / 'review/V2-REVIEW.json')
    parser.add_argument('--launch', type=Path, default=ROOT / 'v2/launch.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'v2/evidence')
    parser.add_argument('--describe', action='store_true', help='print required contract without loading any model')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'experiment': specification(), 'required_bindings': required_bindings()}, indent=2, sort_keys=True))
        return 0
    protocol = preflight(args)
    args.out.mkdir(parents=False, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU, (protocol['max_cpu_seconds'], protocol['max_cpu_seconds']))
    try:
        return run(args, protocol)
    except BaseException as error:
        # Exclusive write prevents replacement of any successful or partial evidence.
        if not (args.out / 'execution-error.json').exists():
            write_json(args.out / 'execution-error.json', {'passed': False, 'error_type': type(error).__name__,
                       'error': str(error), 'protocol_sha256': sha(args.protocol),
                       'historical_stage_b': 'FAIL_UNCHANGED', 'historical_canonical': 'FAIL_UNCHANGED'})
        raise


if __name__ == '__main__':
    sys.exit(main())
