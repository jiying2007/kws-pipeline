"""Corrected-reference conformance harness for the one fixed precision64 candidate. No execution without exact review and launch.

The existing v1 Stage B and canonical failures are immutable historical results.
Affine and FSMN accumulation use FP64; all stored tensors and CMVN stay FP32.
Imports are inert. Only approved main() may load the exact export or execute kernels.
Official references and an independent mathematical oracle are frozen before native execution.
This reference-authority correction is explicitly post-diagnosis; no earlier FAIL is revised.
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
HOOK_LIBRARY = ROOT / 'precision64/harness/build/liba20_local_hooks.so'
MODEL_LIBRARY = ROOT / 'precision64/build/liba20_precision64.so'
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
            'normalized_dyadic': {'generator': 'PCG64', 'seed': 20261004, 'shape': [31, 400],
                         'integer_inclusive_range': [-64, 64], 'divisor': 64,
                         'raw_conversion': 'FP32 z; FP32 divide by istd; FP32 add mean'},
            'feature_uniform': {'generator': 'PCG64', 'seed': 3102027, 'shape': [37, 400],
                          'uniform_range': [-16, 24], 'raw_conversion': 'float32'},
            'impulse_family': {'shape': [19, 400], 'rows': [1, 4, 7, 10, 13],
                          'columns': [1, 38, 140, 250, 398], 'values': [2, -2, 2, -2, 2],
                          'raw_conversion': 'FP32 z; FP32 divide by istd; FP32 add mean'}},
        'split_parts': {'normalized_dyadic': [1, 3, 9, 18], 'feature_uniform': [1, 3, 9, 8, 16],
                        'impulse_family': [1, 3, 7, 8]},
        'fixture_row_identities': 87, 'reference_route_rows': 174, 'reused_reference_route_rows': 0, 'mathematical_oracle_rows': 87, 'native_route_rows': 174,
        'fresh_fixture_freeze_before_outputs': True, 'distinct_numerical_vectors': 74,
        'unseen_unique_input_values_claim': False, 'postdiagnosis_reference_correction': True, 'accumulation_bits': 64, 'storage_bits': 32, 'cmvn_bits': 32,
        'gates': {'logit_atol': 1e-4, 'logit_rtol': 1e-5, 'probability_atol': 1e-5,
                  'rowsum_atol': 1e-5, 'cmvn_atol': 2e-4, 'event_score_atol': 1e-5,
                  'event_discrete_fields': 'exact',
                  'local_affine_gamma': 'n+1', 'local_memory_gamma': 14,
                  'u32': 2.0 ** -24, 'u64': 2.0 ** -53,
                  'affine_underflow_operations': '2*n+1', 'memory_underflow_operations': 27,
                  'underflow_quantum': 2.0 ** -149,
                  'precision64_native_local_bound': 'u32*abs(oracle)+(1+u32)*(2*gamma_k(u64)*L1_upper)+min_subnormal_f32',
                  'precision64_bound_rounding': 'outward at every positive arithmetic operation',
                  'strict_raw_logit_authority': 'high-precision post-CMVN exact-real network enclosure',
                  'strict_raw_logit_interval_rule': 'up(max(native-lower,upper-native)) <= down(1e-4+1e-5*min_abs_interval)',
                  'mathematical_oracle_max_abs_uncertainty': '1e-20',
                  'official_raw_logits': 'diagnostic_only',
                  'official_probabilities_and_events': 'hard_unchanged'},
        'reference_convention': 'official whole/split for probabilities/events/local gates; ideal post-CMVN mathematical intervals for final raw logits',
        'historical_stage_b': 'FAIL_UNCHANGED', 'historical_canonical': 'FAIL_UNCHANGED', 'historical_v2': 'FAIL_UNCHANGED',
        'audio_reads': 0, 'training_updates': 0,
        'candidate_precision64_previous_execution': 'NOT_RUN_SUPERSEDED_REFERENCE_GATE',
    }


def required_bindings():
    local = ['oracle-corrected/harness/fixtures.py', 'oracle-corrected/harness/unit_contract.py',
             'oracle-corrected/harness/unit-receipt.json',
             'oracle-corrected/harness/README.md', 'oracle-corrected/fixtures.npz',
             'oracle-corrected/fixtures-manifest.json', 'oracle-corrected/prepare_inputs.py',
             'oracle-corrected/oracle/decimal_fsmn.py',
             'oracle-corrected/NUMERICAL-CONTRACT.md',
             'precision64/harness/fixtures.py', 'precision64/harness/local_hooks.c',
             'precision64/harness/events.py', 'precision64/harness/unit_math.py',
             'precision64/harness/build/liba20_local_hooks.so', 'precision64/harness/build/receipt.json',
             'precision64/model/a20_fsmn.c', 'precision64/model/a20_fsmn.h',
             'precision64/model/load.c', 'precision64/model/sha256.c', 'precision64/model/sha256.h',
             'precision64/prepare_harness.py', 'precision64/harness/README.md', 'precision64/unit-receipt.json',
             'precision64/build.py', 'precision64/build/liba20_precision64.so', 'precision64/build/receipt.json',
             'precision64/source.diff', 'precision64/NUMERICAL-CONTRACT.md',
             'v2/evidence/reference.npz', 'v2/evidence/oracle.npz', 'v2/evidence/reference-events.json',
             'v2/evidence/reference-freeze.json', 'v2/evidence/report.json',
             'v2/harness/fixtures.py', 'v2/harness/local_hooks.c', 'v2/harness/build_hooks.py',
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
    assert args.protocol.resolve() == ROOT / 'oracle-corrected/protocol.json'
    assert args.review.resolve() == ROOT / 'review/ORACLE-CORRECTED-REVIEW.json'
    assert args.launch.resolve() == ROOT / 'oracle-corrected/launch.json'
    assert args.out.resolve() == ROOT / 'oracle-corrected/evidence'
    protocol = json.loads(args.protocol.read_bytes())
    review = json.loads(args.review.read_bytes())
    launch = json.loads(args.launch.read_bytes())
    assert protocol['schema'] == 'a20-native-oracle-corrected-strict-final-v1'
    assert protocol['experiment'] == specification(), 'experiment differs from fixed harness contract'
    assert review['approved'] is True and review['protocol_sha256'] == sha(args.protocol)
    assert launch['authorized'] is True and launch['protocol_sha256'] == sha(args.protocol)
    assert launch['review_sha256'] == sha(args.review)
    assert launch['pid'] == os.getpid(), 'launch must name this child PID'
    assert launch['output_directory'] == str(args.out.resolve())
    assert os.path.abspath(sys.executable) == protocol['runtime']['python']
    assert 0 < protocol['max_cpu_seconds'] <= 60
    assert 0 < protocol['max_wall_seconds'] <= 120
    assert 0 < protocol['max_rss_kib'] <= 2097152
    assert 0 < protocol['artifact_cap_bytes'] <= 104857600
    assert 0 < protocol['file_cap_bytes'] <= 20971520
    assert protocol['actual_audio_execution'] is False and protocol['benchmark_execution'] is False
    verify_bindings(protocol)
    assert json.loads((ROOT / 'evidence/stage-a/acceptance.json').read_bytes())['passed'] is True
    return protocol


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_fixtures(np, state):
    mean = state['global_cmvn.mean']
    istd = state['global_cmvn.istd']
    z = np.random.Generator(np.random.PCG64(20261004)).integers(-64, 65, (31, 400), dtype=np.int64).astype(np.float32)
    z = np.divide(z, np.float32(64), dtype=np.float32)
    derived = {'normalized_dyadic': np.add(mean, np.divide(z, istd, dtype=np.float32), dtype=np.float32),
              'feature_uniform': np.random.Generator(np.random.PCG64(3102027)).uniform(-16, 24, (37, 400)).astype(np.float32)}
    z = np.zeros((19, 400), np.float32)
    z[[1, 4, 7, 10, 13], [1, 38, 140, 250, 398]] = [2, -2, 2, -2, 2]
    derived['impulse_family'] = np.add(mean, np.divide(z, istd, dtype=np.float32), dtype=np.float32)
    manifest = json.loads((ROOT / 'oracle-corrected/fixtures-manifest.json').read_bytes())
    assert manifest['fixture_sha256'] == sha(ROOT / 'oracle-corrected/fixtures.npz')
    assert manifest['rows'] == 87 and manifest['whole_network_forwards'] == 0
    with np.load(ROOT / 'oracle-corrected/fixtures.npz', allow_pickle=False) as frozen:
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


def mixed_bound(np, oracle, l1_upper, count):
    """Outward enclosure of comparison error to the same saved FP64 oracle.

    Both exact-product double reductions have error <= E64. Conversion of the
    native double to FP32 contributes <= u32*abs(native_double)+min_subnormal.
    Triangle inequality gives u32*abs(oracle)+(1+u32)*2*E64+min_subnormal.
    """
    assert oracle.dtype == l1_upper.dtype == np.float64
    assert np.isfinite(oracle).all() and np.isfinite(l1_upper).all() and (l1_upper >= 0).all()
    up = lambda x: np.nextafter(x, np.inf)
    e64 = up(gamma(np, count, 2.0 ** -53) * l1_upper)
    quantization = up((2.0 ** -24) * np.abs(oracle))
    double_difference = up(2.0 * e64)
    propagated = up(up(1.0 + 2.0 ** -24) * double_difference)
    bound = up(up(quantization + propagated) + 2.0 ** -149)
    assert np.isfinite(bound).all() and (bound > 0).all()
    return bound


def freeze_ideal(np, math_oracle, state, inputs, arrays, calls, out):
    # A single ideal sequence is meaningful for both native routes only when
    # the official post-CMVN starting values are exactly the same.
    for name in inputs:
        split = [call for call in calls if call['input'] == name and call['route'] == 'split']
        joined = np.concatenate([arrays[call['prefix'] + '_cmvn'] for call in split], axis=0)
        exact(np, joined, arrays[name + '_whole_0_cmvn'], name + '_whole_split_cmvn_ideal_guard')
    evaluator = math_oracle.Decimal80FSMN(state)
    cases, hashes = {}, {}
    counters = {'reference_rows': 0, 'affine_calls': 0, 'memory_calls': 0, 'weighted_products': 0}
    for name, raw in inputs.items():
        cmvn = arrays[name + '_whole_0_cmvn']
        assert cmvn.shape == (len(raw), 400) and cmvn.dtype == np.float32
        result = evaluator.evaluate(cmvn)
        assert result['input_rows'] == len(raw) and result['precision_digits'] == 80
        assert result['counters'] == {'reference_rows': len(raw), 'affine_calls': 12 * len(raw),
                                      'memory_calls': 4 * len(raw), 'weighted_products': 388984 * len(raw)}
        filename = 'ideal-' + name + '.json'
        write_json(out / filename, result)
        assert (out / filename).stat().st_size <= 20971520, 'ideal per-case file limit'
        hashes[filename] = sha(out / filename)
        cases[name] = math_oracle.slice_interval_result(result, 0, len(raw))
        for key in counters:
            counters[key] += result['counters'][key]
    assert counters == {'reference_rows': 87, 'affine_calls': 1044,
                        'memory_calls': 348, 'weighted_products': 33841608}
    return {'cases': cases, 'files_sha256': hashes, 'reference_rows': counters['reference_rows'],
            'counters': counters, 'whole_split_cmvn_bitexact': True}


def compare_ideal(math_oracle, checks, prefix, logits, case, offset, count, hard=True):
    selected = math_oracle.slice_interval_result(case, offset, count)
    certificate = math_oracle.compare_float32_logits(logits, selected)
    record = {'name': prefix + '_stage20_mathematical_interval',
              'gate': 'hard_final' if hard else 'diagnostic_official_raw_vs_ideal',
              'passed': certificate['passed'], 'elements': certificate['elements'],
              'failed_elements': certificate['failed_elements'],
              'max_abs': float(certificate['maximum_absolute_error_upper']),
              'max_ratio': float(certificate['maximum_tolerance_ratio_upper']),
              'interval_certificate': certificate}
    if certificate['failures']:
        failure = certificate['failures'][0]
        record['first_failure'] = {'index': failure['index'], 'actual': float(failure['actual']),
            'reference': float(failure['oracle_center']),
            'abs_error': float(failure['worst_absolute_error_upper']),
            'bound': float(failure['tolerance_lower'])}
    checks.append(record)


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
    v2 = json.loads((ROOT / 'v2/evidence/report.json').read_bytes())
    assert v2['passed'] is False and v2['hard_failed_checks'] == 1
    return {'v2': 'FAIL_UNCHANGED', 'stage_b': 'FAIL_UNCHANGED', 'stage_b_native_failed_checks': 57,
            'stage_b_torch_partition_failed_checks': 8, 'canonical': 'FAIL_UNCHANGED',
            'canonical_failed_stage_elements': 1, 'canonical_failed_perrow_cache_elements': 11,
            'precision64_original_gate': 'NOT_RUN_SUPERSEDED_REFERENCE_GATE',
            'interpretation': 'Reference-authority correction follows Decimal80 diagnosis; all earlier observed failures remain failures'}


def run(args, protocol):
    start = time.monotonic()
    import numpy as np
    import torch
    assert np.__version__ == protocol['runtime']['numpy']
    assert torch.__version__ == protocol['runtime']['torch']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tiny = torch.tensor([2.0 ** -149, 2.0 ** -126], dtype=torch.float32)
    assert torch.equal((tiny * torch.tensor([1., .5])).view(torch.int32),
                       torch.tensor([1, 0x00400000], dtype=torch.int32))
    exporter = module('a20_corrected_export', ROOT / 'export_a20.py')
    torch_state, payload, unused = exporter.read_state()
    assert payload == (ROOT / 'export/a20.f32').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == 'a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d'
    state = {key: value.numpy() for key, value in torch_state.items()}
    inputs = make_fixtures(np, state)
    old_harness = module('a20_corrected_pinned_reference', ROOT / 'v2/harness/fixtures.py')
    assert old_harness.specification()['split_parts'] == specification()['split_parts']
    arrays, calls, reference_rows = old_harness.freeze_reference(np, torch, torch_state, inputs, args.out)
    oracles, subnormals = old_harness.freeze_oracles(np, torch_state, arrays, calls, args.out)
    events = module('a20_corrected_events', ROOT / 'v2/harness/events.py')
    reference_events = events.freeze_reference_events(arrays, calls)
    write_json(args.out / 'reference-events.json', reference_events)
    math_oracle = module('a20_corrected_math_oracle', ROOT / 'oracle-corrected/oracle/decimal_fsmn.py')
    ideal = freeze_ideal(np, math_oracle, state, inputs, arrays, calls, args.out)
    write_json(args.out / 'reference-freeze.json', {
        'schema': 'a20-oracle-corrected-reference-freeze-v1',
        'protocol_sha256': sha(args.protocol), 'review_sha256': sha(args.review),
        'raw_fixtures_sha256': sha(ROOT / 'oracle-corrected/fixtures.npz'),
        'reference_sha256': sha(args.out / 'reference.npz'),
        'oracle_sha256': sha(args.out / 'oracle.npz'),
        'reference_events_sha256': sha(args.out / 'reference-events.json'),
        'ideal_sha256': ideal['files_sha256'],
        'calls': calls, 'fixture_row_identities': 87, 'reference_route_rows': reference_rows,
        'mathematical_oracle_rows': ideal['reference_rows'], 'mathematical_oracle_counters': ideal['counters'],
        'whole_split_reference_cmvn_bitexact_for_ideal': ideal['whole_split_cmvn_bitexact'],
        'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in inputs.items()},
        'state_sha256': exporter.STATE_SHA, 'candidate_full_rows_before_freeze': 0,
        'candidate_local_calls_before_freeze': 0, 'all_reference_tensors_frozen_before_native': True,
        'all_mathematical_intervals_frozen_before_native': True,
        'postdiagnosis_reference_correction': True})
    # No native library is loaded until official reference and ideal interval evidence is persisted.
    fp = C.POINTER(C.c_float)
    class Native(C.Structure):
        _fields_ = [('weights', fp), ('cache', C.c_float * 5632), ('fault', C.c_int)]
    model = C.CDLL(str(MODEL_LIBRARY))
    hook = C.CDLL(str(HOOK_LIBRARY))
    decoder = C.CDLL(str(DECODER_LIBRARY))
    model.a20_accumulation_bits.argtypes = []
    model.a20_accumulation_bits.restype = C.c_int
    hook.a20_accumulation_bits.argtypes = []
    hook.a20_accumulation_bits.restype = C.c_int
    assert model.a20_accumulation_bits() == hook.a20_accumulation_bits() == 64
    model.a20_model_bytes.restype = C.c_size_t
    assert model.a20_model_bytes() == C.sizeof(Native)
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
        for suffix, limit in [('cmvn', 2e-4), ('probabilities', 1e-5)]:
            compare(np, checks, prefix + '_strict_' + suffix, native_arrays[prefix + '_' + suffix], arrays[prefix + '_' + suffix], limit, 'hard_final')
        compare_ideal(math_oracle, checks, prefix, native_arrays[prefix + '_stage20'],
                      ideal['cases'][call['input']], call['offset'], count)
        compare_ideal(math_oracle, checks, prefix + '_official_diagnostic', arrays[prefix + '_stage20'],
                      ideal['cases'][call['input']], call['offset'], count, hard=False)
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
            w = state[stem + '.linear.weight']
            b = state.get(stem + '.linear.bias')
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
            mixed = mixed_bound(np, oracles[key + '_oracle'], oracles[key + '_l1_upper'], ni + 1)
            local_arrays[key + '_precision64_bound'] = mixed
            compare(np, checks, key + '_native_precision64', result, oracles[key + '_oracle'], mixed, 'hard_local_precision64')
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
            mixed = mixed_bound(np, oracles[key + '_oracle'], oracles[key + '_l1_upper'], 14)
            local_arrays[key + '_precision64_bound'] = mixed
            compare(np, checks, key + '_native_precision64', result, oracles[key + '_oracle'], mixed, 'hard_local_precision64')
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
                    'hard_final_partition' if suffix in ['cmvn', 'probabilities'] else 'diagnostic_reference_partition')
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
    report = {'schema': 'a20-oracle-corrected-strict-final-report-v1',
              'passed': all(row['passed'] for row in hard), 'checks': checks,
              'historical': historical_status(), 'protocol_sha256': sha(args.protocol),
              'review_sha256': sha(args.review), 'launch_sha256': sha(args.launch),
              'files_sha256': {name: sha(args.out / name) for name in ['reference.npz', 'oracle.npz', 'reference-freeze.json', 'reference-events.json', 'native-events.json', 'native.npz', 'local.npz']},
              'fixture_row_identities': 87, 'distinct_numerical_vectors': len({row.tobytes() for value in inputs.values() for row in value}),
              'reference_whole_network_route_rows': reference_rows, 'reused_reference_whole_network_route_rows': 0,
              'mathematical_oracle_rows': ideal['reference_rows'], 'mathematical_oracle_counters': ideal['counters'],
              'whole_split_reference_cmvn_bitexact_for_ideal': ideal['whole_split_cmvn_bitexact'], 'ideal_files_sha256': ideal['files_sha256'],
              'fresh_fixture_freeze_before_outputs': True, 'postdiagnosis_reference_correction': True, 'accumulation_bits': 64, 'weight_activation_cache_bits': 32, 'cmvn_bits': 32, 'native_whole_network_route_rows': native_rows,
              'native_local_affine_calls': affine_calls, 'native_local_memory_calls': memory_calls,
              'native_own_input_affine_calls': affine_calls // 2, 'native_gold_input_affine_calls': affine_calls // 2,
              'native_own_input_memory_calls': memory_calls // 2, 'native_gold_input_memory_calls': memory_calls // 2,
              'extra_reference_local_forwards': 0, 'own_input_hook_reproduction_bitexact': True,
              'event_fields_exact_score_atol_1e5': True,
              'strict_raw_logit_authority': 'mathematical_interval', 'official_raw_logits_diagnostic_only': True, 'reference_decoder_chunks': len(calls), 'reused_reference_decoder_chunks': 0, 'native_decoder_chunks': len(calls), 'all_reference_tensors_frozen_before_native': True,
              'native_partition_bitexact': True, 'native_perrow_cache_reconstruction_bitexact': True,
              'reference_cache_reconstruction_bitexact': True, 'relu_structure_bitexact': True,
              'local_memory_cache_copy_bitexact': True, 'reset_cache_bitwise_zero': True,
              'subnormal_diagnostics': subnormals, 'fp32_overflow_excluded_by_l1': True,
              'round_to_nearest_gradual_underflow_verified': True,
              'hard_failed_checks': sum(not row['passed'] for row in hard),
              'diagnostic_old_B_failed_checks': sum(not row['passed'] for row in checks if row['gate'] == 'diagnostic_old_B'),
              'diagnostic_official_raw_vs_ideal_failed_checks': sum(not row['passed'] for row in checks if row['gate'] == 'diagnostic_official_raw_vs_ideal'),
              'diagnostic_reference_partition_failed_checks': sum(not row['passed'] for row in checks if row['gate'] == 'diagnostic_reference_partition'),
              'audio_reads': 0, 'training_updates': 0, 'actual12_approved': False, 'benchmark_approved': False,
              'wall_seconds': time.monotonic() - start, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'scope': 'One unchanged precision64 candidate on newly frozen synthetic fixtures; explicit post-diagnosis final-raw-logit authority correction; original numeric tolerance and official probability/event gates unchanged; no universal, acoustic or product qualification'}
    write_json(args.out / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, sort_keys=True))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'oracle-corrected/protocol.json')
    parser.add_argument('--review', type=Path, default=ROOT / 'review/ORACLE-CORRECTED-REVIEW.json')
    parser.add_argument('--launch', type=Path, default=ROOT / 'oracle-corrected/launch.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'oracle-corrected/evidence')
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
                       'historical_stage_b': 'FAIL_UNCHANGED', 'historical_canonical': 'FAIL_UNCHANGED', 'historical_v2': 'FAIL_UNCHANGED'})
        raise


if __name__ == '__main__':
    sys.exit(main())
