"""Offline stdlib fixtures only: no downloads, packages, models or audio execution."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import tempfile
import time
import unittest

import tts_public_schema as schema

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[2] / 'test-results' / 'public-schema'
IDENTITY = {'repository': 'jiying2007/kws-data', 'executed_head_sha': 'a' * 40,
            'run_id': 123, 'run_attempt': 1, 'job_key': 'k1-tts'}


def private_setup():
    _, _, _, locks = schema.inventory()
    return {'schema': 'fixed30-setup-receipt-v1', 'python': dict(schema.PYTHON), 'locks': locks,
            'initial_free_bytes': 12 * schema.GIB, 'network_configuration': 'offline_configured_not_kernel_isolated',
            'commands': [], 'wheels': {}, 'disk_checkpoints': [], 'builds': [],
            'python_executable': '/private/interpreter', 'started_unix_seconds': 123,
            'downloads': [{'private_url': '/private/do-not-publish'}]}


def source_fixture():
    path = 'research/k1-punctuation6-v1/plan.proposed.json'
    raw = (HERE / 'plan.proposed.json').read_bytes()
    freeze = {'schema': 'k1-punctuation6-tts-source-freeze-v1',
              'files': {path: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}},
              'source_freeze_sha256': 'b' * 64}
    provenance = json.loads((HERE / 'model-provenance.json').read_bytes())
    value = {'schema': 'k1-punctuation6-tts-source-v1', 'identity': dict(IDENTITY), 'source_tree': 'c' * 40,
             'source_freeze_sha256': freeze['source_freeze_sha256'], 'frozen_files': 1, 'source_files': copy.deepcopy(freeze['files']),
             'model_provenance': copy.deepcopy(provenance), 'plan_sha256': hashlib.sha256(raw).hexdigest(),
             'raw_definition': schema.RAW_DEFINITION, 'runtime_setup': {'status': 'not_available'},
             'runtime_model_load': {'status': 'not_available'}}
    return value, freeze, provenance


def validate_source(value):
    _, freeze, provenance = source_fixture()
    return schema.validate_source(value, IDENTITY, freeze, provenance, freeze['files']['research/k1-punctuation6-v1/plan.proposed.json']['sha256'])


def diagnostic():
    return {'bytes': 0, 'tail_bytes': 0, 'codes': [], 'exception_types': [],
            'stream_scope': 'stderr_only', 'raw_text_disclosed': False}


def stage_fixture(stage='setup'):
    return {'returncode': 0, 'process_returncode': 0,
            'filesystem_sampling': {'schema': 'qwen6-filesystem-sampling-v1',
                'semantics': 'non_atomic_observed_bytes_not_peak_or_quota', 'maximum_scans_per_check_scope': 2, 'scopes': {}},
            'filesystem_error': None, 'termination_reason': 'normal_exit', 'wall_seconds': .01,
            'reaped_children_cpu_seconds': 0., 'sample_count': 0, 'sample_interval_seconds': .25,
            'max_sampled': dict.fromkeys(['rss_bytes', 'live_cpu_seconds', 'workspace_bytes', 'installed_bytes', 'buildtmp_bytes']),
            'cpu_limit_seconds_per_process': 2400 if stage == 'setup' else 3600, 'wall_limit_seconds': 1200.,
            'affinity_core_limit': 4, 'rss_mode': 'sampled_process_group_threshold_not_continuous_hard_limit',
            'rss_threshold_bytes': (4 if stage == 'setup' else 7) * schema.GIB,
            'cpu_mode': 'kernel_RLIMIT_CPU_per_process_not_aggregate_tree_limit', 'setup': stage == 'setup',
            'stderr_diagnostic': diagnostic()}


def resource_fixture():
    return {'schema': 'k1-punctuation6-tts-resources-v1', 'identity': dict(IDENTITY), 'status': 'started',
            'preflight_http': {'body_bytes': 123, 'requests': 5, 'retries': 0}, 'stages': {},
            'limits': dict(schema.LIMITS), 'cpu_accounting': 'RLIMIT_CPU per process, not hard aggregate tree cap',
            'memory_disk_accounting': 'sampled thresholds, not continuous hard limits'}


def model_load(runtime='/private/owned-runtime'):
    _, _, assets, _ = schema.inventory()
    return {'sidecars': [{'path': row['filename'], 'sha256': row['sha256']} for row in assets if row['filename'].endswith('.json')],
            'loaders': [{'loader': 'AutoModel', 'path': runtime + '/models/qwen_tts',
                         'safety_kwargs': {'local_files_only': 'True', 'trust_remote_code': 'False', 'force_download': 'False',
                            'use_safetensors': 'True', 'weights_only': 'True', 'attn_implementation': 'eager',
                            'dtype': 'torch.float32', 'device_map': 'cpu'}}], 'network_events': []}


class PublicSchemaTests(unittest.TestCase):
    def test_unavailable_source(self):
        self.assertTrue(validate_source(source_fixture()[0]))

    def test_source_extra_private_field_at_every_boundary(self):
        for target in ('top', 'identity', 'source_file', 'provenance', 'setup', 'load'):
            value = source_fixture()[0]
            row = {'top': value, 'identity': value['identity'], 'source_file': next(iter(value['source_files'].values())),
                   'provenance': value['model_provenance'], 'setup': value['runtime_setup'], 'load': value['runtime_model_load']}[target]
            row['private_history'] = 'do not publish'
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_source(value)

    def test_source_pins_and_fake_boolean_rejected(self):
        for mutation in ('model', 'source_hash', 'source_path', 'plan_hash', 'freeze_hash', 'count', 'run'):
            value = source_fixture()[0]
            if mutation == 'model': value['model_provenance']['model_revision'] = 'e' * 40
            elif mutation == 'source_hash': next(iter(value['source_files'].values()))['sha256'] = 'f' * 64
            elif mutation == 'source_path': value['source_files']['/private/file'] = next(iter(value['source_files'].values()))
            elif mutation == 'plan_hash': value['plan_sha256'] = 'f' * 64
            elif mutation == 'freeze_hash': value['source_freeze_sha256'] = 'f' * 64
            elif mutation == 'count': value['frozen_files'] = True
            else: value['identity']['run_attempt'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_source(value)

    def test_setup_empty_is_incomplete_not_loaded(self):
        public = schema.project_setup(private_setup())
        self.assertEqual(public['status'], 'incomplete')
        self.assertEqual(public['source_members'], [])
        self.assertEqual(public['model_assets'], {})
        self.assertNotIn('/private', json.dumps(public))
        value = source_fixture()[0]; value['runtime_setup'] = public
        self.assertTrue(validate_source(value))
        self.assertEqual(value['runtime_model_load'], {'status': 'not_available'})

    def test_setup_unknown_private_keys_rejected(self):
        private = private_setup(); private['private_history'] = 'private'
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_exception_text_never_projected(self):
        private = private_setup(); private['exception'] = {'type': 'UnknownPrivateClass', 'message': '/private/password=secret'}
        public = schema.project_setup(private)
        self.assertEqual(public['status'], 'error'); self.assertEqual(public['exception_type'], 'OtherException')
        self.assertNotIn('secret', json.dumps(public)); self.assertNotIn('UnknownPrivateClass', json.dumps(public))
        public['exception_type'] = 'ValueError: /private/secret'
        value = source_fixture()[0]; value['runtime_setup'] = public
        with self.assertRaises(ValueError): validate_source(value)

    def test_setup_lock_identity_rejected(self):
        private = private_setup(); private['locks']['runtime_lock']['sha256'] = 'a' * 64
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_runtime_source_member_exact_lock(self):
        private = private_setup(); _, sources, _, _ = schema.inventory()
        private['package_verification'] = {'installed': [], 'source_members': [dict(row, bytes=123) for row in sources],
            'model_assets': {}, 'disk': {key: 0 for key in ('installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes')},
            'python': dict(schema.PYTHON)}
        public = schema.project_setup(private)
        self.assertEqual(len(public['source_members']), 17)
        for key, mutation in [('relative_path', '/private/file'), ('sha256', 'b' * 64), ('bytes', True), ('private', 'secret')]:
            modified = copy.deepcopy(private); modified['package_verification']['source_members'][0][key] = mutation
            with self.subTest(key=key), self.assertRaises(ValueError): schema.project_setup(modified)

    def test_runtime_model_assets_exact_lock(self):
        private = private_setup(); _, sources, assets, _ = schema.inventory()
        private['package_verification'] = {'installed': [], 'source_members': [dict(row, bytes=123) for row in sources],
            'model_assets': {'qwen_tts': copy.deepcopy(assets)},
            'disk': {key: 0 for key in ('installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes')}, 'python': dict(schema.PYTHON)}
        self.assertEqual(schema.project_setup(private)['model_assets']['qwen_tts'], assets)
        for key, value in [('filename', '/private/model'), ('sha256', 'f' * 64), ('bytes', False), ('raw_model', 'secret')]:
            altered = copy.deepcopy(private); altered['package_verification']['model_assets']['qwen_tts'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): schema.project_setup(altered)

    def test_verified_setup_requires_real_evidence(self):
        private = private_setup()
        private['verification'] = {'installed': [], 'source_members': [], 'model_assets': {},
            'disk': {key: 0 for key in ('installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes')}, 'python': dict(schema.PYTHON)}
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_complete_setup_fixture_is_not_model_load_evidence(self):
        private = private_setup(); packages, sources, assets, _ = schema.inventory()
        for name, pin in packages.items():
            private['wheels'][name] = {'path': 'downloads/' + pin['filename'],
                'bytes': pin['size_bytes'] if pin['kind'] == 'wheel' else 50000,
                'sha256': pin['sha256'] if pin['kind'] == 'wheel' else 'a' * 64,
                'name': pin['name'], 'version': pin['version'], 'metadata_sha256': pin['metadata_sha256'] or 'b' * 64,
                'metadata_bytes': pin.get('metadata_bytes') or 1024, 'requires_dist': list(pin['requires_dist'])}
        private['bootstrap_pip'] = {'source': 'exact_python_bundled_ensurepip_not_independent_lock',
            'filename': 'pip-25.0.1-py3-none-any.whl', 'bytes': 123, 'sha256': 'a' * 64, 'version': '25.0.1', 'metadata_sha256': 'b' * 64}
        private['commands'] = [{'argv': ['/private/interpreter'], 'log': 'logs/private', 'stage': 'verify_setup',
            'package': None, 'command_role': 'verification', 'status': 'success', 'returncode': 0}]
        private['verification'] = {'installed': [], 'source_members': [dict(row, bytes=123) for row in sources],
            'model_assets': {'qwen_tts': assets}, 'disk': {key: 0 for key in ('installed_bytes', 'buildtmp_bytes', 'workspace_bytes', 'free_bytes')},
            'python': dict(schema.PYTHON)}
        private['import_preflight'] = {'schema': 'native-import-preflight-v1', 'profile': 'tts', 'stage': 'complete',
            'status': 'success', 'weight_loads': 0, 'model_constructions': 0, 'forward_calls': 0, 'raw_text_disclosed': False,
            'versions': {'numpy': packages['numpy']['version']},
            'native_io': {'frames': 12, 'float32_values_sha256': 'c' * 64, 'float_wav_sha256': 'd' * 64,
                'pcm_wav_sha256': 'e' * 64, 'pcm16_values': [-32768, -32768, -16384, -2, -1, 0, 0, 1, 16384, 32767, 32767, 32767],
                'bundled_native_module': '_soundfile_data/libsndfile_x86_64.so', 'bundled_native_sha256': 'f' * 64,
                'libsndfile_version': '1.2.2', 'empty_nonfinite_rejection_checked': True}}
        public = schema.project_setup(private)
        self.assertEqual(public['status'], 'verified'); self.assertEqual(len(public['wheels']), 92)
        value = source_fixture()[0]; value['runtime_setup'] = public
        self.assertTrue(validate_source(value)); self.assertEqual(value['runtime_model_load'], {'status': 'not_available'})
        private['verification']['model_assets']['qwen_tts'].pop()
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_wheel_identity_projection(self):
        private = private_setup(); packages, _, _, _ = schema.inventory(); pin = packages['accelerate']
        private['wheels']['accelerate'] = {'path': '/private/download', 'bytes': pin['size_bytes'], 'sha256': pin['sha256'],
            'name': pin['name'], 'version': pin['version'], 'metadata_sha256': pin['metadata_sha256'],
            'metadata_bytes': pin['metadata_bytes'], 'requires_dist': ['private untrusted body ignored']}
        public = schema.project_setup(private)
        self.assertNotIn('private', json.dumps(public)); self.assertEqual(public['wheels']['accelerate']['sha256'], pin['sha256'])
        private['wheels']['accelerate']['sha256'] = 'c' * 64
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_bootstrap_is_only_bounded_identity(self):
        private = private_setup(); private['bootstrap_pip'] = {'source': 'exact_python_bundled_ensurepip_not_independent_lock',
            'filename': 'pip-25.0.1-py3-none-any.whl', 'bytes': 123, 'sha256': 'a' * 64, 'version': '25.0.1', 'metadata_sha256': 'b' * 64}
        public = schema.project_setup(private)
        self.assertNotIn('filename', public['bootstrap_pip'])
        private['bootstrap_pip']['version'] = '/private/secret'
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_command_argv_and_metadata_body_not_copied(self):
        private = private_setup(); private['commands'] = [{'argv': ['/private/secret'], 'log': '/private/log',
            'stage': 'verify_built_metadata', 'package': 'sox', 'command_role': 'metadata', 'status': 'failed', 'returncode': 1,
            'stderr_diagnostic': diagnostic(), 'metadata_diagnostic': {'schema': 'fixed30-built-metadata-diagnostic-v1',
                'metadata_sha256': 'a' * 64, 'metadata_bytes': 123, 'status': 'failed', 'code': 'requires_dist_mismatch',
                'difference': {'private': 'never copied'}}}]
        public = schema.project_setup(private); self.assertNotIn('private', json.dumps(public))
        private['commands'][0]['stderr_diagnostic']['raw'] = 'secret'
        with self.assertRaises(ValueError): schema.project_setup(private)

    def test_preflight_no_model_claim_and_versions_pinned(self):
        private = private_setup(); private['import_preflight'] = {'schema': 'native-import-preflight-v1', 'profile': 'tts',
            'stage': 'numpy', 'status': 'started', 'weight_loads': 0, 'model_constructions': 0, 'forward_calls': 0,
            'raw_text_disclosed': False, 'versions': {'numpy': schema.inventory()[0]['numpy']['version']}}
        schema.project_setup(private)
        for key, value in [('weight_loads', False), ('versions', {'numpy': '/private/secret'}), ('private', 'secret')]:
            altered = copy.deepcopy(private); altered['import_preflight'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): schema.project_setup(altered)

    def test_model_load_exact_relative_projection(self):
        private = model_load(); public = schema.project_load(private, '/private/owned-runtime')
        self.assertEqual(public['status'], 'loaded_validated'); self.assertEqual(public['loaders'][0]['model_relative_path'], 'qwen_tts')
        self.assertNotIn('/private', json.dumps(public))
        value = source_fixture()[0]; value['runtime_model_load'] = public
        self.assertTrue(validate_source(value))

    def test_load_unknown_fields_roots_flags_and_sidecars_rejected(self):
        for mutation in ('extra', 'path', 'relative', 'flag', 'sidecar_pin', 'sidecar_extra', 'event'):
            private = model_load()
            if mutation == 'extra': private['loaders'][0]['credential'] = 'secret'
            elif mutation == 'path': private['loaders'][0]['path'] = '/private/another-model'
            elif mutation == 'relative': private['loaders'][0]['path'] = '/private/owned-runtime/models/qwen_tts/../qwen_tts'
            elif mutation == 'flag': private['loaders'][0]['safety_kwargs']['local_files_only'] = True
            elif mutation == 'sidecar_pin': private['sidecars'][0]['sha256'] = 'c' * 64
            elif mutation == 'sidecar_extra': private['sidecars'][0]['body'] = 'secret'
            else: private['network_events'] = [{'event': 'socket.connect', 'blocked': True, 'host': '/private/secret'}]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): schema.project_load(private, '/private/owned-runtime')

    def test_projection_failed_receipts_are_explicit(self):
        value = source_fixture()[0]
        value['runtime_setup'] = {'status': 'projection_failed', 'error_class': 'ValueError'}
        value['runtime_model_load'] = {'status': 'projection_failed', 'error_class': 'OtherException'}
        self.assertTrue(validate_source(value))
        resources = resource_fixture(); resources.update(evidence_projection_error_class='ValueError', complete_provenance_available=False)
        self.assertTrue(schema.validate_resources(resources, IDENTITY))
        resources['complete_provenance_available'] = True
        with self.assertRaises(ValueError): schema.validate_resources(resources, IDENTITY)
        value['runtime_setup']['error_class'] = 'ValueError: secret'
        with self.assertRaises(ValueError): validate_source(value)

    def test_resource_started_and_completed(self):
        value = resource_fixture(); self.assertTrue(schema.validate_resources(value, IDENTITY))
        value['stages'] = {name: stage_fixture(name) for name in ('setup', 'generation')}; value['status'] = 'completed_six_once'
        self.assertTrue(schema.validate_resources(value, IDENTITY))

    def test_resource_unknown_private_fields(self):
        for target in ('top', 'identity', 'preflight', 'limits', 'stage', 'max_sampled', 'filesystem', 'stderr'):
            value = resource_fixture(); stage = stage_fixture(); value['stages']['setup'] = stage
            row = {'top': value, 'identity': value['identity'], 'preflight': value['preflight_http'], 'limits': value['limits'],
                   'stage': stage, 'max_sampled': stage['max_sampled'], 'filesystem': stage['filesystem_sampling'], 'stderr': stage['stderr_diagnostic']}[target]
            row['private_text'] = 'secret'
            with self.subTest(target=target), self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)

    def test_resource_bad_scalar_types_and_nonfinite(self):
        for bad in (True, '1', float('nan'), float('inf'), -1, 10 ** 100):
            value = resource_fixture(); value['stages']['setup'] = stage_fixture(); value['stages']['setup']['wall_seconds'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)
        for field in ('returncode', 'sample_count', 'cpu_limit_seconds_per_process', 'affinity_core_limit', 'rss_threshold_bytes'):
            value = resource_fixture(); value['stages']['setup'] = stage_fixture(); value['stages']['setup'][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)

    def test_no_fabricated_zero_for_unsampled_scopes(self):
        value = resource_fixture(); value['stages']['setup'] = stage_fixture()
        value['stages']['setup']['max_sampled']['workspace_bytes'] = 0
        with self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)

    def test_no_new_budget_or_hard_cap_claim(self):
        value = resource_fixture(); value['limits']['setup_cpu_seconds_per_process'] = 2401
        with self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)
        value = resource_fixture(); value['stages']['setup'] = stage_fixture(); value['stages']['setup']['cpu_mode'] = 'hard_tree_limit'
        with self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)

    def test_real_bounded_sampler_preserved_and_mutations_rejected(self):
        spec = importlib.util.spec_from_file_location('schema_test_supervision', schema.HELPERS / 'supervision.py')
        supervision = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervision)
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary); (root / 'fixture').write_bytes(b'fixed test bytes')
            sampler = supervision.FilesystemSampler(root, time.monotonic() + 10)
            amounts = {}
            for scope, path, cap in [('workspace', root, 10 * schema.GIB), ('installed', root / 'venv', 3 * schema.GIB), ('buildtmp', root / 'buildtmp', 2 * schema.GIB)]:
                amounts[scope + '_bytes'] = sampler.sample_bytes(path, scope, cap, optional_root=scope != 'workspace')
            stage = stage_fixture(); stage['filesystem_sampling'] = sampler.receipt; stage['max_sampled'].update(amounts)
            value = resource_fixture(); value['stages']['setup'] = stage
            before = copy.deepcopy(value); self.assertTrue(schema.validate_resources(value, IDENTITY)); self.assertEqual(value, before)
            for mutation in ('row', 'sample', 'count', 'complete', 'coverage', 'error'):
                changed = copy.deepcopy(value); row = changed['stages']['setup']['filesystem_sampling']['scopes']['workspace']; sample = row['last_samples'][0]
                if mutation == 'row': row['private_path'] = '/private/data'
                elif mutation == 'sample': sample['private_text'] = 'secret'
                elif mutation == 'count': row['scans'] = True
                elif mutation == 'complete': sample['complete'] = 1
                elif mutation == 'coverage': sample['coverage'] = 'hard_continuous_quota'
                else: sample['error'] = {'error_type': 'OSError', 'message': 'secret'}
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): schema.validate_resources(changed, IDENTITY)

    def test_incomplete_and_failed_samples_stay_unknown(self):
        sample = {'attempt': 1, 'status': 'observed', 'observed_bytes': 7, 'regular_files': 1, 'symlinks': 0,
            'directories': 0, 'vanished_entries': 1, 'missing_root': False, 'root_missing_at_start': False,
            'root_observed': True, 'complete': False, 'coverage': 'incomplete_lower_bound', 'stop_reason': None}
        row = {'checks': 1, 'scans': 2, 'incomplete_scans': 2, 'vanished_entries': 2, 'absent_optional_roots': 0,
            'observed_root_once': True, 'last_samples': [sample, dict(sample, attempt=2)], 'selected_sample': None,
            'status': 'incomplete_after_one_resample'}
        stage = stage_fixture(); stage.update(returncode=1, process_returncode=0, termination_reason='sampled_filesystem_error', filesystem_error={'error_type': 'ValueError'})
        stage['filesystem_sampling']['scopes']['workspace'] = row
        value = resource_fixture(); value.update(status='stopped_no_retry', error_class='ValueError'); value['stages']['setup'] = stage
        self.assertTrue(schema.validate_resources(value, IDENTITY))
        self.assertIsNone(stage['max_sampled']['workspace_bytes'])
        row['last_samples'] = [{'attempt': 1, 'status': 'failed', 'error': {'error_type': 'OSError', 'errno': 13}}]
        row.update(scans=1, incomplete_scans=1, vanished_entries=0, status='failed')
        self.assertTrue(schema.validate_resources(value, IDENTITY))
        row['last_samples'][0]['error']['private_path'] = '/private/file'
        with self.assertRaises(ValueError): schema.validate_resources(value, IDENTITY)

    def test_stderr_tail_bounded_and_no_arbitrary_exception(self):
        value = resource_fixture(); stage = stage_fixture(); value['stages']['setup'] = stage
        for key, bad in [('bytes', True), ('tail_bytes', 16385), ('exception_types', ['SecretError']), ('codes', ['/private/path']), ('raw_text_disclosed', 0)]:
            altered = copy.deepcopy(value); altered['stages']['setup']['stderr_diagnostic'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): schema.validate_resources(altered, IDENTITY)

    def test_bounded_list_counts(self):
        private = model_load(); private['loaders'] *= 33
        with self.assertRaises(ValueError): schema.project_load(private, '/private/owned-runtime')
        private = private_setup(); private['disk_checkpoints'] = [{}] * 129
        with self.assertRaises(ValueError): schema.project_setup(private)


if __name__ == '__main__':
    unittest.main()
