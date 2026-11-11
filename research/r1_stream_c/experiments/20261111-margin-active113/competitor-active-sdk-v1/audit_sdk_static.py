#!/usr/bin/env python3
"""Independent saved-source/build inspection; never compile or execute model code.

Only nm is executed, to inspect already-built object/archive symbol metadata.
The receipt distinguishes mechanical identity checks from human source review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


PINS = {
    'PLAN.json': 'cd30ab66d17747d347ae243c03d9dbeb6dc3cbc6b0b540e582c076ae6574778f',
    'sdk-v2/MATERIALIZATION.json': 'f64aa68f11d1e44f6597f10ca39282db7da07a96f47117f32f03eb8a62ce997a',
    'build-v2/BUILD_RECEIPT.json': '64ae9d331797d510019a7817e8e9043194fbfb4660993cb4905797c4d7a35f11',
    'pack-v2/PACK.json': '6d09395d508320daf74ddd4eaaceaa116a31f369ed3516f744d589b13e207cf4',
}
VALIDATION_PINS = {
    'DECLARED_PLAN.json': 'a02e36d066526b3f99a2f93296d3debe370760f31ef00b2870f0949f08077e8d',
    'build/BUILD_RECEIPT.json': '0ee8394dea3319f00e7f361d829f8a0a13b20bbed9ddb0d8b5ee6df422bc1391',
    'PREFLIGHT.json': 'b0f299a501828585c4f1b45202fe40347a816d2f2a13d7214375b03a0a3a50c9',
}
PAIRS = (
    ('core/r1d_pcm.c', 'core/r1a113_pcm.c'),
    ('core/r1d_pcm.h', 'core/r1a113_pcm.h'),
    ('core/reference64_fbank.c', 'core/reference64_fbank.c'),
    ('core/reference64_fbank.h', 'core/reference64_fbank.h'),
    ('core/vendor/fft_twiddles_f64.h', 'core/vendor/fft_twiddles_f64.h'),
    ('core/sha256.c', 'core/sha256.c'),
    ('core/sha256.h', 'core/sha256.h'),
    ('src/fsmn_detector.c', 'src/active113_detector.c'),
    ('include/fsmn_detector.h', 'include/active113_detector.h'),
)
PREFIXES = ('active113_', 'r1active113_', 'r1a113_')
LEGACY_PREFIXES = ('fsmn_', 'r1dfsmn', 'r1d_', 'r1dcf')
ASSERTIONS = 0


def need(ok, message):
    global ASSERTIONS
    ASSERTIONS += 1
    if not ok:
        raise ValueError(message)


def raw(path):
    need(stat.S_ISREG(path.lstat().st_mode), 'regular non-symlink file required: ' + str(path))
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def identity(path):
    data = raw(path)
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def load(path):
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw(path), parse_constant=bad)


def namespace(text):
    for before, after in (
        ('r1dfsmn', 'r1active113'), ('R1DFSMN', 'R1ACTIVE113'),
        ('r1d_', 'r1a113_'), ('R1D_', 'R1A113_'),
        ('R1DCF', 'R1A113CF'), ('r1dcf', 'r1a113cf'),
        ('fsmn_', 'active113_'), ('FSMN_', 'ACTIVE113_'),
    ):
        text = text.replace(before, after)
    return text


def function(text, anchor):
    start = text.index(anchor)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def nm(path, mode):
    # This reads object symbol tables. It never loads or executes the object.
    result = subprocess.run(
        ['/usr/bin/nm', '-g', mode, str(path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=10, check=True,
    )
    need(not result.stderr, 'nm emitted diagnostic')
    minimum = 3 if mode == '--defined-only' else 2
    return sorted({line.split()[-1] for line in result.stdout.splitlines()
                   if len(line.split()) >= minimum})


def symbol_hash(symbols):
    return hashlib.sha256(('\n'.join(symbols) + '\n').encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True, type=Path)
    ap.add_argument('--baseline-sdk', required=True, type=Path)
    ap.add_argument('--baseline-build', required=True, type=Path)
    ap.add_argument('--validation', required=True, type=Path)
    ap.add_argument('--input-manifest', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    os.umask(0o077)
    need(not args.output.exists(), 'fresh audit output required')
    sdk = args.candidate / 'sdk-v2'
    for relative, digest in PINS.items():
        need(sha(args.candidate / relative) == digest, 'reviewed SDK receipt changed: ' + relative)
    for relative, digest in VALIDATION_PINS.items():
        need(sha(args.validation / relative) == digest, 'reviewed validation receipt changed: ' + relative)
    plan = load(args.candidate / 'PLAN.json')
    material = load(sdk / 'MATERIALIZATION.json')
    build = load(args.candidate / 'build-v2/BUILD_RECEIPT.json')
    pack = load(args.candidate / 'pack-v2/PACK.json')
    need(material['preparation_plan_sha256'] == build['preparation_plan_sha256'] ==
         pack['preparation_plan_sha256'] == PINS['PLAN.json'], 'preparation binding')
    need(build['materialization_manifest_sha256'] == PINS['sdk-v2/MATERIALIZATION.json'] and
         material['pack_manifest_sha256'] == PINS['pack-v2/PACK.json'], 'materialization binding')
    need(build['status'] == 'PASS_BUILD_AND_INVENTED_LOADER_CHECKS' and
         build['neural_model_calls'] == build['pcm_calls'] == 0, 'loader-only build scope')
    need(pack['status'] == 'PACKED_NO_MODEL_EXECUTION' and
         pack['neural_model_calls'] == pack['pcm_reads'] == pack['training'] == 0 and
         pack['parameter_values_modified'] is False and pack['default_source_or_model_modified'] is False,
         'pack preparation scope')
    source_inventory = {}
    for name, digest in plan['source_sha256'].items():
        path = args.candidate / name
        value = identity(path)
        need(value['sha256'] == digest, 'prepared source changed: ' + name)
        source_inventory[name] = value
    for name, digest in material['base_source_sha256'].items():
        need(sha(args.baseline_sdk / name) == digest, 'baseline source changed: ' + name)
    for name, expected in material['generated_sources'].items():
        need(identity(sdk / name) == expected, 'generated source changed: ' + name)
    for key in ('model', 'frontend'):
        descriptor = pack[key]
        need(identity(args.candidate / 'pack-v2' / descriptor['path']) ==
             {k: descriptor[k] for k in ('bytes', 'sha256')}, 'packed byte identity: ' + key)
    need(pack['model']['bytes'] == 1582624 and pack['model']['sha256'] ==
         '38818ee7ba00c9e9bb762f9d1d2efeca9c1bac349b8f1cf835913fd30ef75984', 'compact model identity')
    need(pack['selection_sha256'] == material['selection_sha256'] == build['selection_sha256'] ==
         '8289f2ca46e40717747c1ed8d08152d8db2fbadcac5f4a84fdbb20d06375dc42', 'selection binding')
    need(pack['screen_result_sha256'] == material['screen_result_sha256'] == build['screen_result_sha256'] ==
         '6fb94b8e60b7572166f5f5221a78e327d67d5dafb1a00b105a87254ad39f15a5', 'cached screen binding')
    namespace_only = {}
    for before, after in PAIRS:
        same = namespace(raw(args.baseline_sdk / before).decode()) == raw(sdk / after).decode()
        need(same, 'namespace-only source drift: ' + after)
        namespace_only[after] = True
    original = namespace(raw(args.baseline_sdk / 'core/r1dfsmn.c').decode())
    current = raw(sdk / 'core/r1active113.c').decode()
    helpers = {}
    for anchor in ('static void affine(', 'static void memory(', 'static void trace_stage(',
                   'static int finite_values(', 'static void relu('):
        same = function(original, anchor) == function(current, anchor)
        need(same, 'core helper drift: ' + anchor)
        helpers[anchor] = True
    old_step = function(original, 'r1active113_status r1active113_step(')
    new_step = function(current, 'r1active113_status r1active113_step(')
    start = '    const float *w = m->weights;'
    old_trunk = old_step[old_step.index(start):old_step.index('    affine_qhead(m, a, y);')]
    new_trunk = new_step[new_step.index(start):new_step.index('    m->other_winner_id=affine_qhead(m, a, y);')]
    need(old_trunk == new_trunk, 'stage 0-19 trunk arithmetic drift')
    need('_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53' in current, 'binary64 guard absent')
    architectures = {}
    for arch in ('host', 'arm'):
        item = build['builds'][arch]
        library = Path(item['static_library']['path'])
        baseline_library = args.baseline_build / arch / 'libfsmn_sdk_static.a'
        need(identity(library) == {k: item['static_library'][k] for k in ('bytes', 'sha256')}, 'library byte identity')
        for obj in item['objects'].values():
            need(identity(Path(obj['path'])) == {k: obj[k] for k in ('bytes', 'sha256')}, 'object byte identity')
        defined = nm(library, '--defined-only')
        undefined = nm(library, '--undefined-only')
        baseline_symbols = nm(baseline_library, '--defined-only')
        need(defined == sorted(item['namespace_symbols']) and len(defined) == 46, 'defined symbol receipt mismatch')
        need(all(s.startswith(PREFIXES) for s in defined), 'unprefixed exported symbol')
        need(not set(defined) & set(baseline_symbols), 'baseline global symbol collision')
        need(not any(s.startswith(LEGACY_PREFIXES) for s in undefined), 'unresolved legacy backend reference')
        need(item['loader_test_result'] == {'assertions': 87, 'neural_model_calls': 0, 'pcm_calls': 0,
             'status': 'PASS_INVENTED_LOADER_AND_ID_CHECKS'}, 'saved invented-loader result')
        architectures[arch] = {
            'static_library': {k: item['static_library'][k] for k in ('bytes', 'sha256')},
            'baseline_static_library': identity(baseline_library),
            'defined_symbol_count': len(defined), 'defined_symbols_sha256': symbol_hash(defined),
            'undefined_symbols_sha256': symbol_hash(undefined),
            'baseline_defined_symbol_count': len(baseline_symbols),
            'baseline_defined_symbols_sha256': symbol_hash(baseline_symbols),
            'legacy_backend_undefined_references': 0, 'baseline_global_symbol_collisions': 0,
            'all_defined_symbols_namespaced': True,
            'saved_loader_result': item['loader_test_result'], 'loader_checks_reexecuted': False,
        }
    declaration = load(args.validation / 'DECLARED_PLAN.json')
    validation_build = load(args.validation / 'build/BUILD_RECEIPT.json')
    preflight = load(args.validation / 'PREFLIGHT.json')
    need(validation_build['status'] == 'BUILT_PROBES_NO_PCM_OR_NEURAL_EXECUTION' and
         validation_build['declaration_sha256'] == VALIDATION_PINS['DECLARED_PLAN.json'] and
         preflight['status'] == 'READY_NO_PCM_EXECUTION' and preflight['pcm_calls'] == preflight['neural_calls'] == 0,
         'validation preparation-only scope')
    need(declaration['limits'] == {'total_seconds': 900, 'child_seconds': 30, 'cpu_threads': 1,
         'capture_calls': 260, 'host_cpu_calls': 18, 'host_memory_calls': 12,
         'total_pcm_calls': 290, 'retry': False}, 'fixed execution budget')
    for key, count in (('cpu', 9), ('memory', 6)):
        expected = [['baseline', 'active113'] if i % 2 == 0 else ['active113', 'baseline'] for i in range(count)]
        need(declaration['resource_orders'][key] == expected, 'fixed resource order')
    for name, value in declaration['source_files'].items():
        need(identity(Path(name)) == value, 'validation source drift')
    probe_symbols = {}
    for arch in ('host', 'arm'):
        probe_symbols[arch] = {}
        for kind in ('plain', 'capture'):
            descriptor = validation_build['builds'][arch]['binaries'][kind]
            path = Path(descriptor['path'])
            need(identity(path) == {k: descriptor[k] for k in ('bytes', 'sha256')}, 'probe byte identity')
            symbols = nm(path, '--defined-only')
            wrapped = '__wrap_r1active113_step' in symbols
            need(wrapped == (kind == 'capture'), 'plain/capture wrapper separation')
            probe_symbols[arch][kind] = {'binary_sha256': descriptor['sha256'],
                'defined_symbols_sha256': symbol_hash(symbols), 'step_wrapper_present': wrapped}
    # Scan exact observation identifiers only; original model token IDs are public semantics.
    manifest = load(args.input_manifest)
    need(sha(args.input_manifest) == '63d2f75e5e93e454bdf23624277fa728bba0cef6eee9cce311f037612198d6ab',
         'privacy inventory binding')
    observation_ids = {r['source_id'].encode() for r in manifest['records']}
    public_sources = [args.candidate / name for name in source_inventory]
    public_sources += [sdk / name for name in material['generated_sources']]
    public_sources += [Path(name) for name in declaration['source_files']]
    for path in public_sources:
        data = raw(path)
        need(not any(value in data for value in observation_ids), 'private observation ID in reviewed source')
    receipt = {
        'schema': 'active113-independent-static-review-v1',
        'status': 'PASS_INDEPENDENT_STATIC_SOURCE_AND_SAVED_BUILD_REVIEW',
        'audit_source_sha256': sha(Path(__file__)), 'sdk_receipt_sha256': PINS,
        'preparation_source_files': source_inventory, 'generated_source_files': material['generated_sources'],
        'base_source_sha256': material['base_source_sha256'],
        'compact_model_sha256': pack['model']['sha256'], 'compact_model_bytes': pack['model']['bytes'],
        'frontend_sha256': pack['frontend']['sha256'], 'selection_sha256': pack['selection_sha256'],
        'cached_screen_sha256': pack['screen_result_sha256'],
        'namespace_only_source_comparisons': namespace_only, 'unchanged_core_helpers': helpers,
        'stage_0_through_19_arithmetic_exact_after_namespace_mapping': True,
        'stage_0_through_19_source_sha256': hashlib.sha256(new_trunk.encode()).hexdigest(),
        'binary64_guard_present': True, 'architectures': architectures,
        'manual_source_review': {
            'scope': 'Source reasoning bound to the exact generated and prepared source hashes above; not runtime validation.',
            'loader': 'Strict compact layout, fixed model identity, alignment, finite-value and overlap checks before publication; copy and borrowed-view lifetime contracts are distinct.',
            'head': 'Five unchanged scalar precise rows plus 113 unchanged quantized rows; original par4 four-lane reduction, scales and biases; no full 2601-logit equivalence claim.',
            'winner': 'Six numerical scores plus actual selected OTHER original-ID metadata; highest score then lowest original ID; unchanged detector receives that original ID.',
            'abi': 'Distinct public ABI and namespaced public/backend symbols; six scores cannot be consumed as a 2601-logit buffer.',
            'capture': 'One real network step per emitted frame; existing trace yields stage19 hidden140; the ordinary callback provides six scores and OTHER ID.',
            'functional_comparison': 'Complete F/E bytes and per-frame clocks, hidden140, exact5, selected OTHER score/ID, and global original ID are compared to retained fixed references.',
            'resource': 'Plain host probes exclude capture code and link wrappers; same mmap/feed/process-CPU timing basis; resource phase follows all 260 functional comparisons.',
        },
        'validation_static_review': {'receipt_sha256': VALIDATION_PINS,
            'limits': declaration['limits'], 'resource_orders': declaration['resource_orders'],
            'source_files': declaration['source_files'], 'probe_symbol_separation': probe_symbols,
            'independent_preexecution_blockers': []},
        'source_privacy': {'source_files_scanned': len(public_sources),
            'private_observation_identifier_count': len(observation_ids),
            'exact_private_observation_id_matches': 0,
            'raw_weights_audio_hidden_features_predictions_in_review_receipt': False,
            'original_token_ids_are_model_semantics': True},
        'reviewer_execution': {'model_calls': 0, 'neural_head_calls': 0, 'detector_calls': 0,
            'pcm_calls': 0, 'training': 0, 'compilation': 0, 'loader_test_runs': 0,
            'nm_metadata_inspections': 10, 'model_bytes_read_for_hash_only': True},
        'interpretation': 'Static/build review permits the separately frozen bounded validation. It proves no runtime accuracy, independent FAR/FRR, physical-board qualification or generalization.',
        'model_promotion_authorized': False, 'default_source_or_model_modified': False,
        'assertions': ASSERTIONS,
    }
    with args.output.open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({'status': receipt['status'], 'assertions': receipt['assertions'],
                      'audit_sha256': sha(args.output), 'audit_source_sha256': receipt['audit_source_sha256'],
                      'model_calls': 0, 'pcm_calls': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
