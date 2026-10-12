"""Fixed, stdlib-only public receipts. Never publish private runtime dictionaries.

Reviewed lock files are the authority for every package/member/model name and
pin. A projection failure is evidence unavailability, never successful setup or
model loading. Numeric bounds below bound metadata; they do not grant budgets.
"""
from pathlib import Path
import errno
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / 'qwen6_tts'
GIB = 1024 ** 3
MAX_BYTES = 2 ** 63 - 1
SAFE_EXCEPTIONS = frozenset({'ValueError', 'RuntimeError', 'TimeoutError', 'TimeoutExpired',
    'OSError', 'FileNotFoundError', 'PermissionError', 'MetadataMismatch', 'ModuleNotFoundError',
    'ImportError', 'MemoryError', 'CalledProcessError', 'AttributeError', 'TypeError',
    'KeyboardInterrupt', 'SystemExit', 'OtherException'})
STAGES = {'bootstrap_pip': 'bootstrap', 'install_build_tools': 'install',
    'install_runtime_wheels': 'install', 'check_build_requirements': 'preflight',
    'build_wheel': 'build', 'verify_built_metadata': 'metadata', 'install_built_wheels': 'install',
    'verify_setup': 'verification', 'verify_installed_before_models': 'verification',
    'import_native_preflight': 'preflight'}
DIAGNOSTIC_CODES = frozenset({'requires_dist_mismatch', 'metadata_identity_mismatch',
    'metadata_hash_mismatch', 'build_requirement_not_locked', 'build_tool_version_drift',
    'build_frontend_missing_dependencies'})
PYTHON = {'implementation': 'CPython', 'version': '3.12.14', 'system': 'Linux',
          'machine': 'x86_64', 'byteorder': 'little'}
LIMITS = {'setup_wall_seconds': 1200, 'generation_wall_seconds': 1200,
    'setup_cpu_seconds_per_process': 2400, 'generation_cpu_seconds_per_process': 3600,
    'setup_sampled_rss_bytes': 4 * GIB, 'generation_sampled_rss_bytes': 7 * GIB,
    'affinity_cores': 4, 'job_stage_wall_seconds': 2400}
RAW_DEFINITION = 'wrapper-returned postdecoder float32, before our fixed resampling; not decoder-preclamp signal'


def require(value, message='invalid public metadata'):
    if not value:
        raise ValueError(message)


def keys(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
            'exact public metadata keys')


def integer(value, low=0, high=MAX_BYTES):
    require(type(value) is int and low <= value <= high, 'bounded metadata integer')


def number(value, high=86400):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= high,
            'bounded finite metadata number')


def boolean(value):
    require(type(value) is bool, 'real metadata boolean')


def digest(value, size=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(size) + '}', value) is not None,
            'fixed digest')


def same(value, expected):
    require(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) ==
            json.dumps(expected, sort_keys=True, allow_nan=False, separators=(',', ':')),
            'reviewed metadata changed')


def bounded(value, maximum):
    require(len(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')) <= maximum,
            'public metadata byte bound')


def copied(value):
    return json.loads(json.dumps(value, allow_nan=False))


def exception(value):
    require(type(value) is str and value in SAFE_EXCEPTIONS, 'fixed exception class')


def identity_check(value, expected):
    keys(value, {'repository', 'executed_head_sha', 'run_id', 'run_attempt', 'job_key'})
    require(value['repository'] == 'jiying2007/kws-data' and value['job_key'] == 'k1-tts', 'public run scope')
    digest(value['executed_head_sha'], 40)
    integer(value['run_id'], 1, 10 ** 20 - 1)
    integer(value['run_attempt'], 1, 1)
    same(value, expected)


def inventory(helper_root=None):
    root = Path(helper_root or HELPERS)
    blobs = {key: (root / name).read_bytes() for key, name in
             [('runtime_lock', 'runtime-lock.json'), ('model_locks', 'model-locks.json')]}
    require(all(len(raw) <= 2 * 1024 ** 2 for raw in blobs.values()), 'bounded reviewed locks')
    runtime, models = json.loads(blobs['runtime_lock']), json.loads(blobs['model_locks'])
    require(set(models) == {'qwen_tts'} and len(runtime['files']) == 92, 'frozen lock inventory')
    packages = {re.sub(r'[-_.]+', '-', row['name']).lower(): row for row in runtime['files']}
    require(len(packages) == 92, 'unique package inventory')
    model = models['qwen_tts']
    sources = model['source_lock']
    assets = [{key: row[key] for key in ('filename', 'bytes', 'sha256')} for row in model['asset_lock']['files']]
    require(len(sources) == 17 and len(assets) == 13, 'frozen source/model inventory')
    locks = {key: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for key, raw in blobs.items()}
    return packages, sources, assets, locks


def unavailable(value):
    if type(value) is dict and value.get('status') in ('not_available', 'projection_failed'):
        if value['status'] == 'not_available':
            keys(value, {'status'})
        else:
            keys(value, {'status', 'error_class'})
            exception(value['error_class'])
        return True
    return False


def _diagnostic(value):
    required = {'bytes', 'tail_bytes', 'codes', 'exception_types', 'stream_scope', 'raw_text_disclosed'}
    keys(value, required, {'tail_sha256', 'tail_truncated'})
    integer(value['bytes']); integer(value['tail_bytes'], 0, min(16384, value['bytes']))
    require(value['stream_scope'] == 'stderr_only' and value['raw_text_disclosed'] is False, 'no raw diagnostic text')
    for key, allowed in [('codes', DIAGNOSTIC_CODES), ('exception_types', SAFE_EXCEPTIONS - {'KeyboardInterrupt', 'SystemExit', 'OtherException'})]:
        require(type(value[key]) is list and len(value[key]) <= len(allowed), 'bounded diagnostic enums')
        require(all(type(item) is str and item in allowed for item in value[key]), 'fixed diagnostic enum')
        require(value[key] == sorted(set(value[key])), 'unique sorted diagnostic enums')
    require(('tail_sha256' in value) == ('tail_truncated' in value), 'complete tail identity')
    if 'tail_sha256' in value:
        digest(value['tail_sha256']); boolean(value['tail_truncated'])
        require(value['tail_truncated'] == (value['bytes'] > value['tail_bytes']), 'tail coverage')
    else:
        require(value['bytes'] == value['tail_bytes'] == 0 and value['codes'] == value['exception_types'] == [], 'unavailable stderr')


def _disk(value):
    keys(value, {'installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes'})
    for amount in value.values():
        integer(amount)


def _phase(value, packages):
    keys(value, {'stage', 'package'})
    require(type(value['stage']) is str and value['stage'] in set(STAGES) | {'download_package', 'download_model_asset', 'extract_source'}, 'fixed setup phase')
    require(value['package'] is None or type(value['package']) is str and value['package'] in set(packages) | {'pip', 'qwen_tts'}, 'fixed setup package')


def _commands(values, packages):
    require(type(values) is list and len(values) <= 16, 'bounded setup command count')
    for row in values:
        keys(row, {'stage', 'package', 'command_role', 'status', 'returncode'},
             {'wall_seconds', 'exception_type', 'stderr_diagnostic', 'metadata_diagnostic'})
        _phase({'stage': row['stage'], 'package': row['package']}, packages)
        require(row['stage'] in STAGES and row['command_role'] == STAGES[row['stage']], 'fixed command role')
        require(row['status'] in ('started', 'failed', 'success'), 'command status')
        if row['returncode'] is not None:
            integer(row['returncode'], -128, 255)
        if row['status'] == 'success':
            require(type(row['returncode']) is int and row['returncode'] == 0, 'success returncode')
        if 'wall_seconds' in row:
            number(row['wall_seconds'])
        if 'exception_type' in row:
            exception(row['exception_type'])
        if 'stderr_diagnostic' in row:
            _diagnostic(row['stderr_diagnostic'])
        if 'metadata_diagnostic' in row:
            d = row['metadata_diagnostic']
            keys(d, {'schema', 'metadata_sha256', 'metadata_bytes', 'status', 'code'}, {'exception_type'})
            require(d['schema'] == 'fixed30-built-metadata-diagnostic-v1' and d['status'] in ('started', 'failed', 'success') and
                    d['code'] in ('full_requires_dist_match', 'requires_dist_mismatch', 'metadata_validation_failed'), 'fixed metadata diagnostic')
            digest(d['metadata_sha256']); integer(d['metadata_bytes'], 0, 2 * 1024 ** 2)
            if 'exception_type' in d:
                exception(d['exception_type'])


def _preflight(value, packages):
    if value == {'status': 'diagnostic_rejected', 'code': 'unsafe_diagnostic_payload'}:
        same(value, {'status': 'diagnostic_rejected', 'code': 'unsafe_diagnostic_payload'})
        return
    keys(value, {'schema', 'profile', 'stage', 'status', 'weight_loads', 'model_constructions', 'forward_calls', 'raw_text_disclosed'},
         {'native_io', 'versions', 'exception_type', 'code'})
    require(value['schema'] == 'native-import-preflight-v1' and value['profile'] == 'tts', 'TTS preflight only')
    require(value['stage'] in {'entry', 'verify_no_model_assets', 'runtime_prepare', 'numpy', 'soundfile', 'scipy_signal', 'native_io', 'qwen_tts', 'tts_transitive_native', 'complete'}, 'fixed preflight stage')
    require(value['status'] in ('started', 'success', 'failed_no_retry') and value['raw_text_disclosed'] is False, 'preflight state')
    for key in ('weight_loads', 'model_constructions', 'forward_calls'):
        integer(value[key], 0, 0)
    if 'exception_type' in value:
        exception(value['exception_type'])
    if 'code' in value:
        require(value['code'] == 'IMPORT_OR_NATIVE_PREFLIGHT_FAILED', 'preflight error code')
    if 'versions' in value:
        allowed = {'numpy', 'soundfile', 'scipy', 'torch', 'transformers', 'qwen-tts', 'onnxruntime', 'torchaudio', 'sox', 'cffi'}
        keys(value['versions'], set(), allowed)
        for name, version in value['versions'].items():
            same(version, packages[name]['version'])
    if 'native_io' in value:
        native = value['native_io']
        keys(native, {'frames', 'float32_values_sha256', 'float_wav_sha256', 'pcm_wav_sha256', 'pcm16_values',
                     'bundled_native_module', 'bundled_native_sha256', 'libsndfile_version', 'empty_nonfinite_rejection_checked'})
        integer(native['frames'], 12, 12)
        same(native['pcm16_values'], [-32768, -32768, -16384, -2, -1, 0, 0, 1, 16384, 32767, 32767, 32767])
        require(native['bundled_native_module'] == '_soundfile_data/libsndfile_x86_64.so' and native['empty_nonfinite_rejection_checked'] is True, 'fixed native probe')
        require(type(native['libsndfile_version']) is str and re.fullmatch(r'[0-9]{1,3}(?:\.[0-9]{1,3}){1,3}', native['libsndfile_version']), 'numeric native version')
        for key in native:
            if key.endswith('_sha256'):
                digest(native[key])
    if value['status'] == 'success':
        require(value['stage'] == 'complete' and 'native_io' in value and 'versions' in value and 'exception_type' not in value, 'complete import evidence')


def _validate_setup(value, helper_root=None):
    if unavailable(value):
        return
    packages, sources, assets, locks = inventory(helper_root)
    required = {'schema', 'status', 'python', 'locks', 'initial_free_bytes', 'network_configuration',
                'command_results', 'wheels', 'source_members', 'model_assets', 'disk_checkpoints'}
    keys(value, required, {'actual_download_bytes', 'wall_seconds', 'bootstrap_pip', 'active_phase', 'exception_type', 'import_preflight'})
    require(value['schema'] == 'fixed30-setup-receipt-v1' and value['status'] in ('verified', 'incomplete', 'error'), 'truthful setup state')
    same(value['python'], PYTHON); same(value['locks'], locks)
    integer(value['initial_free_bytes'], 12 * GIB)
    require(value['network_configuration'] == 'offline_configured_not_kernel_isolated', 'setup network semantics')
    require(type(value['disk_checkpoints']) is list and len(value['disk_checkpoints']) <= 128, 'bounded disk checkpoints')
    for row in value['disk_checkpoints']:
        _disk(row)
    _commands(value['command_results'], packages)
    if 'actual_download_bytes' in value:
        integer(value['actual_download_bytes'], 0, 4 * GIB)
    if 'wall_seconds' in value:
        number(value['wall_seconds'])
    if 'exception_type' in value:
        exception(value['exception_type'])
        require(value['status'] == 'error', 'setup exception status')
    if value['status'] == 'error':
        require('exception_type' in value, 'error class required')
    if 'active_phase' in value:
        _phase(value['active_phase'], packages)
    if 'bootstrap_pip' in value:
        row = value['bootstrap_pip']
        keys(row, {'source', 'bytes', 'sha256', 'version', 'metadata_sha256'})
        require(row['source'] == 'exact_python_bundled_ensurepip_not_independent_lock', 'bootstrap scope')
        require(type(row['version']) is str and re.fullmatch(r'[0-9]{1,3}(?:\.[0-9]{1,3}){1,3}', row['version']), 'numeric pip version')
        integer(row['bytes'], 1, 16 * 1024 ** 2); digest(row['sha256']); digest(row['metadata_sha256'])
    keys(value['wheels'], set(), set(packages))
    for name, row in value['wheels'].items():
        pin = packages[name]
        keys(row, {'bytes', 'sha256', 'name', 'version', 'metadata_sha256', 'metadata_bytes'})
        same(row['name'], pin['name']); same(row['version'], pin['version'])
        integer(row['bytes'], 1, 4 * GIB); integer(row['metadata_bytes'], 1, 2 * 1024 ** 2)
        digest(row['sha256']); digest(row['metadata_sha256'])
        if pin['kind'] == 'wheel':
            for key, lockkey in [('bytes', 'size_bytes'), ('sha256', 'sha256'), ('metadata_sha256', 'metadata_sha256'), ('metadata_bytes', 'metadata_bytes')]:
                if lockkey != 'metadata_bytes' or pin.get(lockkey) is not None:
                    same(row[key], pin[lockkey])
    require(type(value['source_members']) is list and len(value['source_members']) in (0, len(sources)), 'all-or-absent verified source inventory')
    for row, pin in zip(value['source_members'], sources):
        keys(row, {'distribution', 'relative_path', 'sha256', 'bytes'})
        same({k: row[k] for k in pin}, pin)
        integer(row['bytes'], 0, 16 * 1024 ** 2)
    keys(value['model_assets'], set(), {'qwen_tts'})
    if 'qwen_tts' in value['model_assets']:
        same(value['model_assets']['qwen_tts'], assets)
    if 'import_preflight' in value:
        _preflight(value['import_preflight'], packages)
    if value['status'] == 'verified':
        require(set(value['wheels']) == set(packages) and len(value['source_members']) == len(sources) and
                set(value['model_assets']) == {'qwen_tts'} and 'bootstrap_pip' in value and
                value.get('import_preflight', {}).get('status') == 'success', 'complete package verification evidence')
        require(any(row['stage'] == 'verify_setup' and row['status'] == 'success' for row in value['command_results']), 'actual verification command')
    bounded(value, 512 * 1024)


def project_setup(private_setup_receipt, helper_root=None):
    """Project only fixed public metadata; failures raise ValueError, never succeed."""
    if private_setup_receipt is None:
        return {'status': 'not_available'}
    setup = private_setup_receipt
    required = {'schema', 'python', 'locks', 'initial_free_bytes', 'network_configuration', 'commands', 'wheels', 'disk_checkpoints'}
    keys(setup, required, {'python_executable', 'started_unix_seconds', 'ended_unix_seconds', 'builds', 'downloads',
         'actual_download_bytes', 'wall_seconds', 'bootstrap_pip', 'phase', 'exception', 'verification', 'package_verification', 'import_preflight'})
    result = {key: copied(setup[key]) for key in required - {'commands', 'wheels'}}
    for key in ('actual_download_bytes', 'wall_seconds', 'import_preflight'):
        if key in setup:
            result[key] = copied(setup[key])
    result['status'] = 'error' if 'exception' in setup else ('verified' if 'verification' in setup else 'incomplete')
    if 'exception' in setup:
        keys(setup['exception'], {'type', 'message'})
        result['exception_type'] = setup['exception']['type'] if setup['exception']['type'] in SAFE_EXCEPTIONS else 'OtherException'
    if 'phase' in setup:
        result['active_phase'] = copied(setup['phase'])
    if 'bootstrap_pip' in setup:
        row = setup['bootstrap_pip']
        keys(row, {'source', 'filename', 'bytes', 'sha256', 'version', 'metadata_sha256'})
        require(row['filename'] == 'pip-' + row['version'] + '-py3-none-any.whl', 'bundled pip filename')
        result['bootstrap_pip'] = {key: copied(row[key]) for key in row if key != 'filename'}
    result['command_results'] = []
    require(type(setup['commands']) is list and len(setup['commands']) <= 16, 'bounded private commands')
    for row in setup['commands']:
        keys(row, {'argv', 'log', 'stage', 'package', 'command_role', 'status', 'returncode'},
             {'wall_seconds', 'exception_type', 'stderr_diagnostic', 'metadata_diagnostic'})
        output = {key: copied(row[key]) for key in row if key not in ('argv', 'log', 'metadata_diagnostic')}
        if 'metadata_diagnostic' in row:
            diagnostic = row['metadata_diagnostic']
            keys(diagnostic, {'schema', 'metadata_sha256', 'metadata_bytes', 'status', 'code'}, {'exception_type', 'difference'})
            output['metadata_diagnostic'] = {key: copied(diagnostic[key]) for key in diagnostic if key != 'difference'}
        result['command_results'].append(output)
    result['wheels'] = {}
    require(type(setup['wheels']) is dict and len(setup['wheels']) <= 92, 'bounded wheel evidence')
    for name, row in setup['wheels'].items():
        keys(row, {'path', 'bytes', 'sha256', 'name', 'version', 'metadata_sha256', 'metadata_bytes', 'requires_dist'},
             {'normalized_requires_dist', 'retained_metadata'})
        result['wheels'][name] = {key: copied(row[key]) for key in ('bytes', 'sha256', 'name', 'version', 'metadata_sha256', 'metadata_bytes')}
    verification = setup.get('verification', setup.get('package_verification'))
    if verification is None:
        result['source_members'], result['model_assets'] = [], {}
    else:
        keys(verification, {'installed', 'source_members', 'model_assets', 'disk', 'python'})
        same(verification['python'], PYTHON); _disk(verification['disk'])
        result['source_members'] = copied(verification['source_members'])
        result['model_assets'] = copied(verification['model_assets'])
    _validate_setup(result, helper_root)
    return result


def _validate_load(value, helper_root=None):
    if unavailable(value):
        return
    keys(value, {'status', 'sidecars', 'loaders', 'network_events'})
    require(value['status'] == 'loaded_validated', 'model load completion evidence')
    _, _, assets, _ = inventory(helper_root)
    expected = [{'path': row['filename'], 'sha256': row['sha256']} for row in assets if row['filename'].endswith('.json')]
    require(type(value['sidecars']) is list and len(value['sidecars']) == len(expected), 'exact sidecar inventory')
    for row in value['sidecars']:
        keys(row, {'path', 'sha256'})
        require(type(row['path']) is str and type(row['sha256']) is str, 'sidecar types')
    same(sorted(value['sidecars'], key=lambda row: row['path']), sorted(expected, key=lambda row: row['path']))
    require(type(value['loaders']) is list and 1 <= len(value['loaders']) <= 32, 'bounded local loader calls')
    safety = {'local_files_only': 'True', 'trust_remote_code': 'False', 'force_download': 'False',
              'use_safetensors': 'True', 'weights_only': 'True', 'attn_implementation': 'eager',
              'dtype': 'torch.float32', 'device_map': 'cpu'}
    for row in value['loaders']:
        keys(row, {'loader', 'model_relative_path', 'safety_kwargs'}, {'completed_config_attention'})
        require(row['loader'] in ('AutoModel', 'AutoConfig', 'AutoProcessor', 'AutoFeatureExtractor', 'AutoTokenizer'), 'fixed model loader')
        require(row['model_relative_path'] in ('qwen_tts', 'qwen_tts/speech_tokenizer'), 'fixed public model root')
        base = {'local_files_only', 'trust_remote_code', 'force_download'}
        keys(row['safety_kwargs'], set(safety) if row['loader'] == 'AutoModel' else base, set(safety))
        for key, flag in row['safety_kwargs'].items():
            same(flag, safety[key])
        if 'completed_config_attention' in row:
            require(row['loader'] == 'AutoConfig' and row['completed_config_attention'] == 'eager_recursively', 'fixed config attention')
    require(any(row['loader'] == 'AutoModel' and row['model_relative_path'] == 'qwen_tts' for row in value['loaders']), 'top-level model loader evidence')
    require(type(value['network_events']) is list and len(value['network_events']) <= 32, 'bounded offline audit')
    for row in value['network_events']:
        keys(row, {'event', 'blocked'})
        require(row['event'] in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request') and row['blocked'] is True, 'fixed blocked network event')
    bounded(value, 32768)


def project_load(private_load_receipt, runtime, helper_root=None):
    """Only the post-validation worker receipt can establish loaded_validated."""
    if private_load_receipt is None:
        return {'status': 'not_available'}
    keys(private_load_receipt, {'sidecars', 'loaders', 'network_events'})
    root = Path(runtime)
    require(root.is_absolute(), 'absolute owned runtime argument')
    out = {'status': 'loaded_validated', 'sidecars': copied(private_load_receipt['sidecars']),
           'network_events': copied(private_load_receipt['network_events']), 'loaders': []}
    require(type(private_load_receipt['loaders']) is list and len(private_load_receipt['loaders']) <= 32, 'bounded private loaders')
    for row in private_load_receipt['loaders']:
        keys(row, {'loader', 'path', 'safety_kwargs'}, {'completed_config_attention'})
        require(type(row['path']) is str and row['path'] in
                (str(root / 'models/qwen_tts'), str(root / 'models/qwen_tts/speech_tokenizer')), 'exact private load roots')
        out['loaders'].append({key: copied(row[key]) for key in row if key != 'path'} |
                             {'model_relative_path': Path(row['path']).relative_to(root / 'models').as_posix()})
    _validate_load(out, helper_root)
    return out


def validate_source(value, identity, source_freeze, model_provenance, plan_sha256, *, helper_root=None):
    """source_freeze is reviewed freeze JSON augmented with its file SHA256."""
    keys(value, {'schema', 'identity', 'source_tree', 'source_freeze_sha256', 'frozen_files', 'source_files',
                 'model_provenance', 'plan_sha256', 'raw_definition', 'runtime_setup', 'runtime_model_load'})
    require(value['schema'] == 'k1-punctuation6-tts-source-v1', 'source schema')
    identity_check(value['identity'], identity); digest(value['source_tree'], 40)
    keys(source_freeze, {'schema', 'files', 'source_freeze_sha256'})
    require(source_freeze['schema'] == 'k1-punctuation6-tts-source-freeze-v1', 'source freeze authority')
    same(value['source_freeze_sha256'], source_freeze['source_freeze_sha256']); digest(value['source_freeze_sha256'])
    require(type(source_freeze['files']) is dict and 1 <= len(source_freeze['files']) <= 256, 'bounded fixed source inventory')
    same(value['source_files'], source_freeze['files'])
    integer(value['frozen_files'], len(source_freeze['files']), len(source_freeze['files']))
    for path, row in value['source_files'].items():
        require(type(path) is str and len(path) <= 240 and
                (path.startswith('research/k1-punctuation6-v1/') or path.startswith('research/qwen6_tts/') or path == '.github/workflows/k1-punctuation6-tts-v1.yml') and
                all(part not in ('', '.', '..') for part in path.split('/')) and '\\' not in path, 'frozen public source path')
        keys(row, {'bytes', 'sha256'}); integer(row['bytes'], 0, 8 * 1024 ** 2); digest(row['sha256'])
    same(value['model_provenance'], model_provenance)
    same(value['plan_sha256'], plan_sha256); digest(value['plan_sha256'])
    require(value['raw_definition'] == RAW_DEFINITION, 'fixed raw signal definition')
    _validate_setup(value['runtime_setup'], helper_root)
    _validate_load(value['runtime_model_load'], helper_root)
    bounded(value, 2 * 1024 ** 2)
    return True


def _safe_error(value):
    keys(value, {'error_type'}, {'errno'})
    require(value['error_type'] in ('TimeoutError', 'ValueError', 'TypeError', 'OSError', 'RuntimeError', 'KeyboardInterrupt', 'SystemExit', 'OtherError'), 'fixed filesystem error')
    if 'errno' in value:
        integer(value['errno'], 1, 4095)
        require(value['errno'] in errno.errorcode and value['error_type'] in ('OSError', 'TimeoutError'), 'real OS errno')


def _filesystem(value):
    keys(value, {'schema', 'semantics', 'maximum_scans_per_check_scope', 'scopes'})
    require(value['schema'] == 'qwen6-filesystem-sampling-v1' and value['semantics'] == 'non_atomic_observed_bytes_not_peak_or_quota', 'non-atomic disk semantics')
    integer(value['maximum_scans_per_check_scope'], 2, 2)
    keys(value['scopes'], set(), {'workspace', 'installed', 'buildtmp'})
    statuses = {'not_started', 'started', 'deadline_exhausted', 'root_missing', 'cap_exceeded', 'failed', 'complete', 'incomplete_after_one_resample'}
    measurements = {'observed_bytes', 'regular_files', 'symlinks', 'directories', 'vanished_entries', 'missing_root',
                    'root_missing_at_start', 'root_observed', 'complete', 'coverage', 'stop_reason'}
    for scope, row in value['scopes'].items():
        keys(row, {'checks', 'scans', 'incomplete_scans', 'vanished_entries', 'absent_optional_roots',
                   'observed_root_once', 'last_samples', 'selected_sample', 'status'})
        for key in ('checks', 'scans', 'incomplete_scans', 'vanished_entries', 'absent_optional_roots'):
            integer(row[key], 0, 10 ** 9)
        require(row['scans'] <= 2 * row['checks'] and row['incomplete_scans'] <= row['scans'] and row['absent_optional_roots'] <= row['scans'], 'sampling count consistency')
        boolean(row['observed_root_once'])
        require(row['status'] in statuses and type(row['last_samples']) is list and len(row['last_samples']) <= 2, 'fixed sampling status/count')
        require(len(row['last_samples']) <= row['scans'], 'last scans included in aggregate count')
        if row['selected_sample'] is not None:
            integer(row['selected_sample'], 1, len(row['last_samples']))
            require(row['status'] == 'complete', 'selected complete sample')
        for index, sample in enumerate(row['last_samples'], 1):
            keys(sample, {'attempt', 'status'}, measurements | {'error', 'fd_close_error_count', 'fd_close_first_error'})
            integer(sample['attempt'], index, index)
            require(sample['status'] in ('started', 'observed', 'cap_exceeded', 'failed'), 'fixed scan status')
            present = set(sample) & measurements
            require(not present or present == measurements, 'no fabricated partial counters')
            if present:
                integer(sample['observed_bytes'])
                for key in ('regular_files', 'symlinks', 'directories', 'vanished_entries'):
                    integer(sample[key], 0, 10 ** 9)
                for key in ('missing_root', 'root_missing_at_start', 'root_observed', 'complete'):
                    boolean(sample[key])
                require(sample['coverage'] in ('incomplete_lower_bound', 'complete_non_atomic_sample', 'absent_optional_root'), 'fixed coverage semantics')
                require(sample['stop_reason'] in (None, 'deadline_exhausted', 'cap_exceeded'), 'fixed scan stop reason')
                if sample['coverage'] == 'complete_non_atomic_sample':
                    require(sample['complete'] is True and sample['missing_root'] is False and sample['vanished_entries'] == 0, 'complete scan counters')
                elif sample['coverage'] == 'absent_optional_root':
                    require(scope != 'workspace' and sample['complete'] is True and sample['root_missing_at_start'] is True and sample['missing_root'] is True and sample['observed_bytes'] == 0, 'explicit optional-root absence')
                else:
                    require(sample['complete'] is False, 'incomplete scan cannot claim completeness')
            else:
                require(sample['status'] in ('started', 'failed'), 'unmeasured scan cannot be observed')
            if 'error' in sample:
                _safe_error(sample['error'])
                require(sample['status'] in ('failed', 'cap_exceeded'), 'failed scan error state')
            if sample['status'] in ('failed', 'cap_exceeded'):
                require('error' in sample, 'failed scan safe error evidence')
            require(('fd_close_error_count' in sample) == ('fd_close_first_error' in sample), 'close error evidence pair')
            if 'fd_close_error_count' in sample:
                integer(sample['fd_close_error_count'], 1, 10 ** 9); _safe_error(sample['fd_close_first_error'])
        if row['status'] == 'complete':
            require(row['selected_sample'] is not None, 'complete row selection')
            selected = row['last_samples'][row['selected_sample'] - 1]
            require(selected.get('complete') is True and selected['status'] == 'observed', 'complete selected observation')
        elif row['status'] == 'incomplete_after_one_resample':
            require(len(row['last_samples']) == 2 and all(sample.get('complete') is False for sample in row['last_samples']), 'bounded incomplete resample')


def _stage(value, stage):
    keys(value, {'returncode', 'process_returncode', 'filesystem_sampling', 'filesystem_error', 'termination_reason',
         'wall_seconds', 'reaped_children_cpu_seconds', 'sample_count', 'sample_interval_seconds', 'max_sampled',
         'cpu_limit_seconds_per_process', 'wall_limit_seconds', 'affinity_core_limit', 'rss_mode', 'rss_threshold_bytes',
         'cpu_mode', 'setup', 'stderr_diagnostic'})
    integer(value['returncode'], -128, 255); integer(value['process_returncode'], -128, 255)
    reason = value['termination_reason']
    require(reason in {'normal_exit', 'process_failure_or_kernel_limit', 'whole_job_deadline', 'wall_deadline',
            'sampled_group_rss', 'sampled_workspace_cap', 'sampled_installed_cap', 'sampled_buildtmp_cap',
            'sampled_filesystem_error', 'sampled_log_cap'}, 'fixed termination reason')
    if value['returncode'] == 0:
        require(value['process_returncode'] == 0 and reason == 'normal_exit', 'truthful stage success')
    for key in ('wall_seconds', 'reaped_children_cpu_seconds'):
        number(value[key], 10 ** 7 if key == 'reaped_children_cpu_seconds' else 86400)
    integer(value['sample_count'], 0, 10 ** 7)
    number(value['sample_interval_seconds']); require(value['sample_interval_seconds'] == .25, 'fixed sampling interval')
    integer(value['cpu_limit_seconds_per_process'], 2400 if stage == 'setup' else 3600, 2400 if stage == 'setup' else 3600)
    number(value['wall_limit_seconds'], 1200); require(value['wall_limit_seconds'] > 0, 'remaining wall budget')
    integer(value['affinity_core_limit'], 4, 4)
    integer(value['rss_threshold_bytes'], (4 if stage == 'setup' else 7) * GIB, (4 if stage == 'setup' else 7) * GIB)
    require(value['rss_mode'] == 'sampled_process_group_threshold_not_continuous_hard_limit' and
            value['cpu_mode'] == 'kernel_RLIMIT_CPU_per_process_not_aggregate_tree_limit', 'truthful resource accounting')
    require(value['setup'] is (stage == 'setup'), 'stage setup flag')
    keys(value['max_sampled'], {'rss_bytes', 'live_cpu_seconds', 'workspace_bytes', 'installed_bytes', 'buildtmp_bytes'})
    for key, amount in value['max_sampled'].items():
        if amount is not None:
            number(amount, 10 ** 7) if key == 'live_cpu_seconds' else integer(amount)
    _filesystem(value['filesystem_sampling'])
    for scope in ('workspace', 'installed', 'buildtmp'):
        row = value['filesystem_sampling']['scopes'].get(scope)
        maximum = value['max_sampled'][scope + '_bytes']
        if row is None:
            require(maximum is None, 'unsampled scope stays unknown')
        elif row['status'] == 'complete':
            selected = row['last_samples'][row['selected_sample'] - 1]
            require(maximum is not None and maximum >= selected['observed_bytes'], 'preserved complete sample maximum')
        elif row['checks'] == 1:
            require(maximum is None, 'incomplete first scan stays unknown')
        if row is not None and row['status'] in ('deadline_exhausted', 'root_missing', 'cap_exceeded', 'failed', 'incomplete_after_one_resample'):
            require(value['returncode'] != 0, 'failed filesystem sampling cannot claim stage success')
        if maximum is not None:
            require(maximum <= {'workspace': 10, 'installed': 3, 'buildtmp': 2}[scope] * GIB, 'selected disk sample stayed within unchanged threshold')
    if value['filesystem_error'] is not None:
        _safe_error(value['filesystem_error'])
        require(value['returncode'] != 0, 'filesystem failure stops stage')
    _diagnostic(value['stderr_diagnostic'])


def validate_resources(value, identity):
    keys(value, {'schema', 'identity', 'status', 'preflight_http', 'stages', 'limits', 'cpu_accounting', 'memory_disk_accounting'},
         {'error_class', 'wall_seconds', 'evidence_projection_error_class', 'complete_provenance_available'})
    require(value['schema'] == 'k1-punctuation6-tts-resources-v1' and value['status'] in ('started', 'stopped_no_retry', 'completed_six_once'), 'resource scope/status')
    identity_check(value['identity'], identity); same(value['limits'], LIMITS)
    require(value['cpu_accounting'] == 'RLIMIT_CPU per process, not hard aggregate tree cap' and
            value['memory_disk_accounting'] == 'sampled thresholds, not continuous hard limits', 'resource accounting scope')
    keys(value['preflight_http'], {'body_bytes', 'requests', 'retries'})
    integer(value['preflight_http']['body_bytes'], 0, 5 * 1024 ** 2)
    integer(value['preflight_http']['requests'], 5, 5); integer(value['preflight_http']['retries'], 0, 0)
    keys(value['stages'], set(), {'setup', 'generation'})
    require('generation' not in value['stages'] or 'setup' in value['stages'] and value['stages']['setup']['returncode'] == 0, 'setup precedes generation')
    for stage, receipt in value['stages'].items():
        _stage(receipt, stage)
    if 'error_class' in value:
        exception(value['error_class']); require(value['status'] == 'stopped_no_retry', 'error status')
    if 'wall_seconds' in value:
        number(value['wall_seconds'])
    if 'evidence_projection_error_class' in value:
        exception(value['evidence_projection_error_class'])
        require(value.get('complete_provenance_available') is False, 'projection failure cannot claim complete provenance')
    if 'complete_provenance_available' in value:
        boolean(value['complete_provenance_available'])
        if value['complete_provenance_available'] is False:
            require('evidence_projection_error_class' in value, 'explicit projection failure reason')
    if value['status'] == 'completed_six_once':
        require(set(value['stages']) == {'setup', 'generation'} and all(row['returncode'] == 0 for row in value['stages'].values()), 'actual successful supervised stages')
    bounded(value, 256 * 1024)
    return True
