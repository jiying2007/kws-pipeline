"""Synthetic byte inventories only: no model imports, downloads or inference."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = next((parent for parent in Path(__file__).resolve().parents
    if (parent / "tools/asr_native_contracts.py").is_file()), HERE)
REPO = Path(os.environ.get("KWS_NATIVE_BASE", str(DEFAULT_BASE)))
SOURCE = HERE / "source" if (HERE / "source").is_dir() else REPO
sys.path.insert(0, str(SOURCE / 'research/qwen12-native-admission-v1'))
from optional_tools import policy_config
FILE = SOURCE / 'research/qwen12-native-admission-v1/native_locks.py'
spec = importlib.util.spec_from_file_location('native_locks_under_test', FILE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


class NativeLocksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime'
        self.code = self.root / 'code'
        self.output = self.root / 'output'
        self.output.mkdir()
        write(self.code / 'research/qwen12-native-admission-v1/native_worker.py', b'# mock staged worker\n')
        write(self.code / 'research/qwen6_asr/core/asr_stage/__init__.py', b'')
        self.site = self.runtime / 'venv/lib/python3.12/site-packages'
        self.family = 'sensevoice'
        self.frontend = native.canonical({'attributes': {'fs': 16000}, 'observed': 'mock only'})
        self.make_family('sensevoice')

    def tearDown(self):
        self.temp.cleanup()

    def make_family(self, family):
        self.family = family
        model, _, revision = native.MODELS[family]
        source_name = 'funasr/models/sense_voice/model.py' if family == 'sensevoice' else 'qwen_asr/inference/qwen3_asr.py'
        body = b'# mock original runtime source\n'
        write(self.site / source_name, body)
        source = [{'distribution': 'funasr' if family == 'sensevoice' else 'qwen-asr',
                   'relative_path': source_name, 'sha256': native.digest(body)}]
        self.original_source = self.site / source_name
        names = {native.TOKENIZER: b'mock tokenizer\x00', 'model.pt': b'mock weight body never read'} if family == 'sensevoice' else {
            'tokenizer_config.json': b'{}', 'model.safetensors': b'mock weight body never read'}
        rows = []
        for name, data in sorted(names.items()):
            write(self.runtime / 'models' / family / name, data)
            rows.append({'filename': name, **native.identity(data), 'source_url': 'https://example.invalid/mock/' + name})
        asset = {'model_id': model, 'revision': revision, 'schema': 'asr-exact-model-assets-v1', 'files': rows}
        self.lock = {'model_id': model, 'asset_lock': asset, 'asset_lock_sha256': native.digest(native.canonical(asset)),
                     'source_lock': source, 'source_lock_sha256': native.digest(native.canonical(source))}
        verified = [{k: v for k, v in row.items() if k != 'source_url'} for row in rows]
        self.receipt = {'assets': {'model_id': model, 'revision': revision, 'asset_lock_sha256': self.lock['asset_lock_sha256'],
                                  'verified_files': verified, 'verified_bytes': sum(r['bytes'] for r in rows)}}
        self.verified = {'model_assets': {family: copy.deepcopy(verified)},
                         'source_members': [{**source[0], 'bytes': len(body)}]}
        self.extensions = None
        if family == 'sensevoice':
            write(self.code / native.PARSER, (REPO / native.PARSER).read_bytes())
            write(self.output / 'native-frontend-config.json', self.frontend)
        else:
            self.frontend = None
            self.extensions = {}
            for name in native.QWEN_EXTENSIONS:
                raw = b'# mock ' + name.encode() + b'\n'
                write(self.site / name, raw)
                self.extensions[name] = native.identity(raw)

    def build(self, **changes):
        values = dict(family=self.family, code_root=self.code, loaded_receipt=self.receipt,
                      verified_runtime=self.verified, executed_head_sha='a' * 40, run_id='38123456789',
                      installed_source_inventory=self.extensions, frontend_config_bytes=self.frontend,
                      execution_policy_bytes=native.canonical(policy_config()) if self.family == 'sensevoice' else None)
        values.update(changes)
        return native.build_native_locks(self.runtime, self.lock, self.output, **values)

    def verify_resolver_shape(self, source_raw, model_raw):
        source, model = json.loads(source_raw), json.loads(model_raw)
        self.assertEqual(source['schema'], 'retained-asr-source-lock-v1')
        self.assertEqual(model['schema'], 'retained-asr-model-lock-v1')
        self.assertEqual(source['executed_head_sha'], 'a' * 40)
        self.assertEqual(model['model_sha256'], native.digest(native.canonical(model['assets'])))
        for field in ('provider_id', 'model_id', 'model_revision', 'model_family', 'model_sha256'):
            self.assertTrue(model[field])
        for lock, field in ((source, 'files'), (model, 'assets')):
            paths = set()
            for row in lock[field]:
                self.assertNotIn(row['path'], paths)
                self.assertGreater(row['bytes'], 0)
                self.assertRegex(row['sha256'], r'^[0-9a-f]{64}$')
                self.assertEqual(native.relative(row['path']), row['path'])
                paths.add(row['path'])
        prefix = 'output/' + self.family + '/'
        for row in source['files']:
            self.assertTrue(row['path'].startswith(prefix + 'retained-source/'))
            self.assertEqual(native.identity((self.output / row['path'][len(prefix):]).read_bytes()),
                             {k: row[k] for k in ('bytes', 'sha256')})
        return source, model

    def test_sense_optional_execution_policy_required_and_exact(self):
        for value in (None, b'{}', native.canonical(dict(policy_config(), availability='available'))):
            with self.subTest(value=value),self.assertRaises(ValueError):self.build(execution_policy_bytes=value)
            # Each failed build already retained source files; fresh synthetic output per try.
            import shutil
            shutil.rmtree(self.output);self.output.mkdir()
            write(self.output/'native-frontend-config.json',self.frontend)

    def test_sense_full_archive_paths_source_and_dependency_members(self):
        source, model = self.verify_resolver_shape(*self.build())
        prefix = 'output/sensevoice/'
        expected = {prefix + 'retained-source/code/' + native.PARSER,
                    prefix + 'retained-assets/' + native.TOKENIZER,
                    prefix + 'native-frontend-config.json'}
        inventory = source['files'] + model['assets']
        for member in expected:
            rows = [r for r in inventory if r['path'] == member]
            self.assertEqual(len(rows), 1)
            self.assertEqual(native.identity((self.output / member[len(prefix):]).read_bytes()),
                             {k: rows[0][k] for k in ('bytes', 'sha256')})
        snapshot = next(r for r in model['assets'] if r['path'].endswith('native-frontend-config.json'))
        self.assertEqual(snapshot['kind'], 'runtime-validation-config')
        self.assertEqual(snapshot['role'], 'sensevoice-live-frontend-validation')
        self.assertEqual(len(source['empty_source_files']), 1)
        empty = source['empty_source_files'][0]
        self.assertEqual(empty['bytes'], 0)
        self.assertEqual((self.output / empty['path'][len(prefix):]).read_bytes(), b'')
        self.assertFalse((self.output / 'retained-assets/model.pt').exists())

    def test_qwen_extended_sources_all_retained(self):
        self.make_family('qwen06')
        source, _ = self.verify_resolver_shape(*self.build())
        paths = {r['path'] for r in source['files']}
        for name in native.QWEN_EXTENSIONS:
            self.assertIn('output/qwen06/retained-source/site-packages/' + name, paths)

    def test_weight_body_not_opened(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in ('model.pt', 'model.safetensors'):
                raise AssertionError('weight body must not be read')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            self.build()

    def test_actual_source_drift_fails(self):
        self.original_source.write_bytes(b'# different source same purpose\n')
        with self.assertRaisesRegex(ValueError, 'bytes/hash drift'):
            self.build()
        self.assertFalse((self.output / 'native-source-lock.json').exists())

    def test_original_source_pin_drift_fails(self):
        self.lock['source_lock'][0]['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'original source pin'):
            self.build()

    def test_loaded_asset_receipt_drift_fails(self):
        self.receipt['assets']['verified_files'][0]['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'inventories disagree'):
            self.build()

    def test_asset_bytes_drift_fails(self):
        path = self.runtime / 'models/sensevoice' / native.TOKENIZER
        body = path.read_bytes()
        path.write_bytes(bytes([body[0] ^ 1]) + body[1:])
        with self.assertRaisesRegex(ValueError, 'bytes/hash drift'):
            self.build()

    def test_missing_asset_fails(self):
        (self.runtime / 'models/sensevoice' / native.TOKENIZER).unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_staged_source_symlink_fails(self):
        (self.code / 'alias.py').symlink_to(self.original_source)
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.build()

    def test_unsafe_relative_paths(self):
        for name in ('../escape', '/absolute', 'a//b', './a', 'a/../b', 'a\\b', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                native.relative(name)

    def test_invalid_actual_run_binding(self):
        for field, value in (('executed_head_sha', 'main'), ('executed_head_sha', 'A' * 40), ('run_id', 'mock-run'), ('run_id', '0')):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.build(**{field: value})

    def test_frontend_retained_bytes_drift(self):
        (self.output / 'native-frontend-config.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'frontend retained bytes differ'):
            self.build()

    def test_frontend_noncanonical_fails(self):
        body = json.dumps({'mock': 1}, indent=2).encode()
        (self.output / 'native-frontend-config.json').write_bytes(body)
        with self.assertRaisesRegex(ValueError, 'canonical frontend'):
            self.build(frontend_config_bytes=body)

    def test_parser_drift_fails(self):
        (self.code / native.PARSER).write_bytes(b'# wrong parser\n')
        with self.assertRaisesRegex(ValueError, 'actual retained SenseVoice parser'):
            self.build()

    def test_missing_qwen_extension_fails(self):
        self.make_family('qwen06')
        self.extensions.pop(native.QWEN_EXTENSIONS[0])
        with self.assertRaisesRegex(ValueError, 'source extensions required'):
            self.build()

    def test_qwen_extension_drift_fails(self):
        self.make_family('qwen06')
        (self.site / native.QWEN_EXTENSIONS[0]).write_bytes(b'# altered\n')
        with self.assertRaisesRegex(ValueError, 'bytes/hash drift'):
            self.build()

    def test_true_native_v2_resolver_accepts_generated_locks_and_dependencies(self):
        sys.path[:0] = [str(REPO / name) for name in ('tools', 'training', 'tests')]
        from asr_resolver_fixtures import trusted_fixture
        from asr_native_fixtures import rebind_native_fixture
        import asr_source_resolver as resolver
        source_raw, model_raw = self.build()
        source, model = json.loads(source_raw), json.loads(model_raw)
        with trusted_fixture() as f:
            f.use_native_v2()
            index = 1
            prefix = 'asr-1'
            f.documents[prefix + '/source-lock.json'] = source
            f.documents[prefix + '/model-lock.json'] = model
            identity = f.record['recognizer_policy']['recognizers'][index]
            for key in ('provider_id', 'model_id', 'model_revision', 'model_family', 'model_sha256'):
                identity[key] = model[key]
            config = f.documents[prefix + '/native-config.json']
            config['source_lock_sha256'] = native.digest(source_raw)
            config['asset_lock_sha256'] = native.digest(model_raw)
            deps = {
                'tokenizer_sha256': 'output/sensevoice/retained-assets/' + native.TOKENIZER,
                'parser_contract_sha256': 'output/sensevoice/retained-source/code/' + native.PARSER,
                'frontend_config_sha256': 'output/sensevoice/native-frontend-config.json'}
            refs = f.record['evidence_bindings']['recognizers'][index]
            refs['native_dependencies'] = {key: f.ref(member) for key, member in deps.items()}
            for member in native.public_body_members(source_raw, model_raw):
                f.extra_members[member] = (self.output / member.removeprefix('output/sensevoice/')).read_bytes()
            for key, member in deps.items():
                config['effective_config'][key] = native.digest(f.extra_members[member])
            capture = f.native_captures[index]
            capture['source_lock_sha256'] = config['source_lock_sha256']
            capture['asset_lock_sha256'] = config['asset_lock_sha256']
            rebind_native_fixture(config, capture)
            identity['native_decode_config'] = copy.deepcopy(config)
            identity['native_decode_config_sha256'] = native.digest(native.canonical(config))
            f.rebind_captures()
            f.rebuild()
            with patch.object(resolver, '_SOURCE_ROOTS', (f.trust_root,)):
                admission = f.admission()
            self.assertTrue(admission['ctc_training_allowed'])
            # Genuine resolver must reject dependency bytes changed after locks.
            member = deps['parser_contract_sha256']
            f.extra_members[member] += b'\n# drift'
            f.rebuild()
            with patch.object(resolver, '_SOURCE_ROOTS', (f.trust_root,)), self.assertRaises(ValueError):
                f.admission()

    def test_public_body_allowlist_excludes_unretained_assets(self):
        self.make_family('qwen06')
        # Add measured large-tokenizer rows without opening them during lock creation.
        for name in ('merges.txt', 'vocab.json'):
            body = b'mock metadata-only public tokenizer body'
            write(self.runtime / 'models/qwen06' / name, body)
            row = {'filename': name, **native.identity(body)}
            self.lock['asset_lock']['files'].append({**row, 'source_url': 'https://example.invalid/' + name})
            self.receipt['assets']['verified_files'].append(row.copy())
            self.verified['model_assets']['qwen06'].append(row.copy())
            self.receipt['assets']['verified_bytes'] += len(body)
        pin = native.digest(native.canonical(self.lock['asset_lock']))
        self.lock['asset_lock_sha256'] = self.receipt['assets']['asset_lock_sha256'] = pin
        source_raw, model_raw = self.build()
        members = native.public_body_members(source_raw, model_raw)
        model = json.loads(model_raw)
        for name in ('merges.txt', 'vocab.json', 'model.safetensors'):
            self.assertNotIn('output/qwen06/retained-assets/' + name, members)
            self.assertFalse((self.output / 'retained-assets' / name).exists())
            row = next(r for r in model['assets'] if r['path'].endswith('/' + name))
            self.assertFalse(row['body_retained'])
            self.assertEqual(row['kind'], 'weights' if name == 'model.safetensors' else 'model-tokenizer-asset')

    def make_whisper(self):
        self.family = 'whisper'
        official = {'schema': 'whisper-screen-official-source-lock-v1',
                    'source_repository': 'https://github.com/openai/whisper',
                    'source_revision': native.WHISPER_REVISION, 'files': []}
        assets = {'schema': 'whisper-screen-loaded-assets-lock-v1',
                  'model_id': 'openai/whisper-small-native', 'files': {}}
        for name in sorted(native.WHISPER_ASSETS | {'whisper/decoding.py', 'whisper/audio.py'}):
            body = ('# mock ' + name).encode()
            observed = native.identity(body)
            if name == 'small.pt':
                write(self.runtime / 'model' / name, body)
                observed['sha256'] = native.WHISPER_MODEL_SHA
            else:
                write(self.runtime / 'source' / name, body)
                official['files'].append({'path': name, 'size_bytes': len(body), 'sha256': observed['sha256']})
            if name in native.WHISPER_ASSETS:
                assets['files'][name] = observed
        write(self.runtime / 'asset-lock.json', native.canonical(assets))
        return dict(code_root=self.code, official_source_lock=official, verified_asset_lock=assets,
                    executed_head_sha='a' * 40, run_id='38123456789')

    def test_whisper_locks_body_paths_and_small_metadata_no_weight_reads(self):
        values = self.make_whisper()
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'small.pt':
                raise AssertionError('Whisper checkpoint must not be opened by helper')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            source_raw, model_raw = native.build_whisper_locks(self.runtime, self.output, **values)
        source, model = self.verify_resolver_shape(source_raw, model_raw)
        self.assertEqual(model['model_id'], 'openai/whisper-small-native')
        self.assertEqual(model['model_revision'], native.WHISPER_REVISION)
        members = native.public_body_members(source_raw, model_raw)
        self.assertIn('output/whisper/retained-assets/whisper/assets/multilingual.tiktoken', members)
        self.assertNotIn('output/whisper/retained-assets/whisper/assets/gpt2.tiktoken', members)
        self.assertNotIn('output/whisper/retained-assets/small.pt', members)

    def test_whisper_official_source_drift_fails(self):
        values = self.make_whisper()
        (self.runtime / 'source/whisper/decoding.py').write_bytes(b'# changed')
        with self.assertRaisesRegex(ValueError, 'bytes/hash drift'):
            native.build_whisper_locks(self.runtime, self.output, **values)

    def test_whisper_checkpoint_inventory_drift_fails(self):
        values = self.make_whisper()
        values['verified_asset_lock']['files']['small.pt']['sha256'] = 'b' * 64
        write(self.runtime / 'asset-lock.json', native.canonical(values['verified_asset_lock']))
        with self.assertRaisesRegex(ValueError, 'official small checkpoint hash'):
            native.build_whisper_locks(self.runtime, self.output, **values)

    def make_wheel(self):
        selected = native.QWEN_EXTENSIONS[:2]
        path = self.runtime / 'downloads/transformers-mock.whl'
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, 'w') as archive:
            for member in selected:
                body = b'# mock locked wheel ' + member.encode()
                archive.writestr(member, body)
                write(self.site / member, body)
        row = {'name': 'transformers', 'kind': 'wheel', 'filename': path.name,
               'size_bytes': path.stat().st_size, 'sha256': native.digest(path.read_bytes())}
        return selected, path, {'files': [row]}

    def test_locked_wheel_matches_selected_installed_source(self):
        selected, _, lock = self.make_wheel()
        result = native.verify_installed_extension_sources(self.runtime, lock, selected)
        self.assertEqual(set(result), set(selected))
        for name in selected:
            self.assertEqual({k: result[name][k] for k in ('bytes', 'sha256')},
                             native.identity((self.site / name).read_bytes()))

    def test_locked_wheel_hash_drift_fails(self):
        selected, path, lock = self.make_wheel()
        body = bytearray(path.read_bytes()); body[10] ^= 1; path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, 'wheel hash drift'):
            native.verify_installed_extension_sources(self.runtime, lock, selected)

    def test_wheel_installed_source_disagreement_fails(self):
        selected, _, lock = self.make_wheel()
        (self.site / selected[0]).write_bytes(b'# drift\n')
        with self.assertRaisesRegex(ValueError, 'differs from locked wheel'):
            native.verify_installed_extension_sources(self.runtime, lock, selected)

    def test_wheel_unsafe_selection_fails(self):
        _, _, lock = self.make_wheel()
        with self.assertRaises(ValueError):
            native.verify_installed_extension_sources(self.runtime, lock, ('../escape.py',))



class OptionalToolPolicyTest(unittest.TestCase):
    """Synthetic import probes, never dependencies/models or real subprocesses."""
    def setUp(self):
        import optional_tools
        import subprocess
        from types import ModuleType, SimpleNamespace
        self.policy = optional_tools
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.temp.name) / 'runtime'
        self.source = self.runtime / 'venv/lib/python3.12/site-packages' / optional_tools.SOURCE
        self.raw = (('\n' * 29) + 'def is_ffmpeg_installed():\n'
            '    """Synthetic same-shaped optional probe."""\n'
            '    try:\n'
            '        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)\n'
            '        return "ffmpeg version" in output.decode("utf-8")\n'
            '    except (subprocess.CalledProcessError, FileNotFoundError):\n'
            '        return False\n').encode()
        write(self.source, self.raw)
        self.module = ModuleType(optional_tools.MODULE)
        self.module.__file__ = str(self.source)
        self.module.__spec__ = SimpleNamespace(origin=str(self.source))
        self.module.subprocess = subprocess
        exec(compile(self.raw, str(self.source), 'exec', dont_inherit=True,
             optimize=sys.flags.optimize), self.module.__dict__)
        self.guard = {'schema': 'qwen12-inference-process-guard-v1', 'family': 'sensevoice',
                      'fork_vfork_nonthread_clone': 'EPERM'}
        self.receipts = []
        self.hash_patch = patch.object(optional_tools, 'SOURCE_SHA', native.digest(self.raw))
        self.hash_patch.start()
        self.old_module = sys.modules.pop(optional_tools.MODULE, None)
        self.old_funasr = sys.modules.pop('funasr', None)

    def tearDown(self):
        sys.modules.pop(self.policy.MODULE, None)
        if self.old_module is not None: sys.modules[self.policy.MODULE] = self.old_module
        if self.old_funasr is not None: sys.modules['funasr'] = self.old_funasr
        self.hash_patch.stop()
        self.temp.cleanup()

    def context(self):
        return self.policy.unavailable_ffmpeg(self.runtime, self.guard, self.receipts.append)

    def finish_import(self):
        self.module.use_ffmpeg = self.module.is_ffmpeg_installed()
        sys.modules[self.policy.MODULE] = self.module

    def test_exact_probe_is_unavailable_without_process_and_restored(self):
        with patch.object(self.policy.subprocess, 'check_output') as original:
            with self.context(): self.finish_import()
            self.assertIs(self.policy.subprocess.check_output, original)
            original.assert_not_called()
        self.assertFalse(self.module.use_ffmpeg)
        self.assertEqual(self.receipts[0]['intercepted_probes'], 1)
        self.assertTrue(self.receipts[0]['completed'])
        self.assertEqual(self.receipts[0]['actual_installed_status'], 'unknown')
        self.assertFalse(self.receipts[0]['external_process_executed'])

    def test_other_caller_permission_error_not_swallowed_and_restored(self):
        error = PermissionError(1, 'synthetic secret must not enter receipt')
        with patch.object(self.policy.subprocess, 'check_output', side_effect=error) as original:
            with self.assertRaises(PermissionError) as caught:
                with self.context(): self.policy.subprocess.check_output(['different-command'])
            self.assertIs(caught.exception, error)
            self.assertIs(self.policy.subprocess.check_output, original)
        self.assertFalse(self.receipts[0]['completed'])
        self.assertNotIn('secret', json.dumps(self.receipts))

    def test_receipt_error_does_not_replace_original_failure(self):
        error = PermissionError(1, 'original synthetic failure')
        def failed_emit(_): raise OSError('receipt failed')
        with self.assertRaises(PermissionError) as caught:
            with self.policy.unavailable_ffmpeg(self.runtime, self.guard, failed_emit): raise error
        self.assertIs(caught.exception, error)
        with self.assertRaisesRegex(OSError, 'receipt failed'):
            with self.policy.unavailable_ffmpeg(self.runtime, self.guard, failed_emit): self.finish_import()

    def test_source_hash_drift_rejected_before_policy_install(self):
        self.source.write_bytes(self.raw + b'# drift\n')
        with self.assertRaisesRegex(ValueError, 'source drift'):
            with self.context(): self.fail('entered')
        self.assertEqual(self.receipts, [])

    def test_already_imported_module_rejected(self):
        for name in ('funasr', self.policy.MODULE):
            with self.subTest(name=name), patch.dict(sys.modules, {name: self.module}):
                with self.assertRaisesRegex(ValueError, 'fresh Sense'):
                    with self.context(): self.fail('entered')

    def test_absent_or_wrong_guard_rejected(self):
        for guard in ({}, {'schema': 'qwen12-inference-process-guard-v1', 'family': 'qwen06',
                           'fork_vfork_nonthread_clone': 'EPERM'}):
            self.guard = guard
            with self.assertRaisesRegex(ValueError, 'process guard'):
                with self.context(): self.fail('entered')

    def test_spoofed_origin_not_intercepted(self):
        self.module.__spec__.origin = '/secret/elsewhere.py'
        with patch.object(self.policy.subprocess, 'check_output', side_effect=PermissionError(1, 'blocked')) as original:
            with self.assertRaises(PermissionError):
                with self.context(): self.finish_import()
            original.assert_called_once()
        self.assertEqual(self.receipts[0]['intercepted_probes'], 0)

    def test_function_code_spoof_not_intercepted(self):
        fake = self.raw.replace(b'["ffmpeg", "-version"]', b'["unexpected", "-version"]')
        exec(compile(fake, str(self.source), 'exec'), self.module.__dict__)
        with patch.object(self.policy.subprocess, 'check_output', side_effect=PermissionError(1, 'blocked')):
            with self.assertRaises(PermissionError):
                with self.context(): self.finish_import()
        self.assertEqual(self.receipts[0]['intercepted_probes'], 0)

    def test_missing_probe_and_enabled_backend_fail_closed(self):
        for enabled in (False, True, None):
            sys.modules.pop(self.policy.MODULE, None)
            self.module.use_ffmpeg = enabled
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                with self.context(): sys.modules[self.policy.MODULE] = self.module

    def test_probe_count_bounded_and_observed_not_assumed_once(self):
        with self.context():
            self.finish_import()
            self.assertFalse(self.module.is_ffmpeg_installed())
        self.assertEqual(self.receipts[-1]['intercepted_probes'], 2)
        sys.modules.pop(self.policy.MODULE)
        with self.assertRaisesRegex(ValueError, 'bounded optional'):
            with self.context():
                for _ in range(self.policy.MAX_PROBES + 1): self.module.is_ffmpeg_installed()
        self.assertEqual(self.receipts[-1]['intercepted_probes'], self.policy.MAX_PROBES)
        self.assertFalse(self.receipts[-1]['completed'])

    def test_policy_config_stable_independent_of_dynamic_probe_count(self):
        before = self.policy.policy_config()
        with self.context(): self.finish_import()
        self.assertEqual(before, self.policy.policy_config())
        self.assertNotIn('intercepted_probes', before)


class SetupGuardDispatchTests(unittest.TestCase):
    """AST-only setup dispatch fixtures; no dependencies, models or real guard."""
    def setUp(self):
        self.setup_source = SOURCE / "research/qwen12-native-admission-v1/setup_adapter.py"
        self.retained = SOURCE / "research/qwen6_asr"
        self.events = []
        self.saved = {}
        self.active = {'guard': False, 'policy': False}
        self.fail_execute = False
        self.fail_guard = False
        self.fail_failure_write = False
        self.fail_diagnostic = False
        self.execute_error = ValueError('fixture import failure')
        source = ast.parse(self.setup_source.read_bytes(), filename=str(self.setup_source))
        selected = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ('_require', '_run_child')]
        self.assertEqual(len(selected), 2)
        def bind(profile, runtime, stage):
            self.events.append('bind')
            self.assertEqual(stage, 'setup')
            return SimpleNamespace(verify_setup=lambda *a, **kw: {'ok':True})
        def install(family):
            self.events.append('guard')
            self.assertEqual(family, 'sensevoice')
            self.assertFalse(self.active['guard'])
            if self.fail_guard: raise RuntimeError('fixture guard failure')
            self.active['guard'] = True
            return {'schema':'qwen12-inference-process-guard-v1','family':family,'rlimit_cpu_seconds':600,'fork_vfork_nonthread_clone':'EPERM'}
        def write_once(path, value):
            self.events.append('write:'+path.name)
            if path.name == 'import-preflight-failure.json' and self.fail_failure_write:
                raise OSError('fixture evidence write failure')
            self.assertNotIn(path.name, self.saved)
            self.saved[path.name] = value
        def register_codes():
            self.assertTrue(self.active['guard'])
            self.events.append('diagnostic-register')
        def diagnostic_stage(stage):
            self.assertEqual(stage, 'pre_model_import')
            self.events.append('diagnostic-stage')
        def register_probe(code):
            self.assertTrue(self.active['guard'])
            self.events.append('probe-code-register')
        def failure_diagnostic(error):
            if self.fail_diagnostic: raise RuntimeError('fixture diagnostic failure')
            result = {'schema':'qwen12-bounded-failure-v1','stage':'pre_model_import',
                'frames':[],'message_disclosed':False}
            if isinstance(error, OSError): result['errno'] = error.errno
            return result
        @contextmanager
        def unavailable(runtime, guard, emit, *, register_code=None):
            self.assertTrue(self.active['guard'])
            self.assertEqual(runtime, Path('/runtime'))
            self.events.append('policy-enter')
            self.assertIs(register_code, register_probe)
            register_code(None)
            self.active['policy'] = True
            completed = False
            try:
                yield
                completed = True
            finally:
                self.active['policy'] = False
                self.events.append('policy-exit')
                emit({'schema':'fixture-policy','completed':completed})
        def execute(profile, runtime, target):
            self.events.append('execute')
            self.assertEqual(target, runtime/'import-preflight.json')
            if profile == 'asr':
                self.assertTrue(self.active['guard'])
                self.assertTrue(self.active['policy'])
            else:
                self.assertFalse(self.active['guard'])
                self.assertFalse(self.active['policy'])
            if self.fail_execute: raise self.execute_error
        self.modules = {'cpu_budget':SimpleNamespace(install_inference_guard=install),
                        'campaign_contract':SimpleNamespace(write_once=write_once),
                        'optional_tools':SimpleNamespace(unavailable_ffmpeg=unavailable),
                        'native_worker':SimpleNamespace(_register_diagnostic_codes=register_codes,
                            _diagnostic_stage=diagnostic_stage,register_optional_probe_code=register_probe,
                            _safe_error_type=lambda error: type(error).__name__ if type(error) in
                                (ValueError,PermissionError,OSError) else 'OtherException',
                            _failure_diagnostic=failure_diagnostic),
                        'import_preflight':SimpleNamespace(execute=execute)}
        self.ns = {'Path':Path,'sys':sys,'json':json,'bind':bind,
                   'runpy':SimpleNamespace(run_module=lambda *a, **kw:self.events.append('run-module'))}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(self.setup_source),'exec'),self.ns)
    def run_dispatch(self,profile='asr',role='preflight',arguments=None):
        if arguments is None:arguments=[profile,'/runtime']
        with patch.dict(sys.modules,self.modules):
            self.ns['_run_child'](profile,'/runtime',role,arguments)
    def test_guard_and_policy_before_existing_execute(self):
        self.run_dispatch()
        self.assertEqual(self.events,['bind','guard','diagnostic-register','diagnostic-stage',
            'write:import-process-guard.json','policy-enter','probe-code-register','execute',
            'policy-exit','write:import-optional-tool-policy.json'])
        value=self.saved['import-process-guard.json']
        self.assertEqual(value['cpu_accounting'],'setup')
        self.assertEqual(value['extra_child_cpu_limit_seconds'],600)
        self.assertFalse(value['container_matches_decode_scope'])
        self.assertFalse(self.active['policy'])
        self.assertTrue(self.saved['import-optional-tool-policy.json']['completed'])
    def test_failure_propagates_after_policy_receipt(self):
        self.fail_execute=True
        with self.assertRaisesRegex(ValueError,'fixture import failure'):self.run_dispatch()
        self.assertFalse(self.saved['import-optional-tool-policy.json']['completed'])
        self.assertFalse(self.active['policy'])
        self.assertEqual(self.events.count('execute'),1)
        record=self.saved['import-preflight-failure.json']
        self.assertEqual(record['stage'],'pre_model_import')
        self.assertEqual(record['family'],'shared_qwen_sense')
        self.assertEqual(record['error_type'],'ValueError')
        self.assertFalse(record['diagnostic']['message_disclosed'])
        self.assertNotIn('fixture import failure',json.dumps(record))
    def test_failure_receipt_write_does_not_replace_original_error(self):
        self.fail_execute=True
        self.fail_failure_write=True
        with self.assertRaises(ValueError) as caught:self.run_dispatch()
        self.assertIs(caught.exception,self.execute_error)
        self.assertNotIn('import-preflight-failure.json',self.saved)
        self.assertFalse(self.active['policy'])
    def test_diagnostic_error_yields_safe_fallback_and_original_error(self):
        self.fail_execute=True
        self.fail_diagnostic=True
        with self.assertRaises(ValueError) as caught:self.run_dispatch()
        self.assertIs(caught.exception,self.execute_error)
        self.assertEqual(self.saved['import-preflight-failure.json']['diagnostic'],
            {'schema':'qwen12-bounded-failure-v1','status':'unavailable'})
    def test_permission_errno_retained_without_exception_message(self):
        self.fail_execute=True
        self.execute_error=PermissionError(13,'sensitive fixture message')
        with self.assertRaises(PermissionError) as caught:self.run_dispatch()
        self.assertIs(caught.exception,self.execute_error)
        record=self.saved['import-preflight-failure.json']
        self.assertEqual(record['error_type'],'PermissionError')
        self.assertEqual(record['diagnostic']['errno'],13)
        self.assertNotIn('sensitive',json.dumps(record))
    def test_guard_failure_blocks_import(self):
        self.fail_guard=True
        with self.assertRaisesRegex(RuntimeError,'fixture guard failure'):self.run_dispatch()
        self.assertEqual(self.events,['bind','guard'])
        self.assertEqual(self.saved,{})
    def test_tts_branch_unchanged(self):
        self.run_dispatch(profile='tts')
        self.assertEqual(self.events,['bind','execute'])
        self.assertEqual(self.saved,{})
    def test_invalid_arguments_rejected_before_guard(self):
        with self.assertRaises(ValueError):self.run_dispatch(arguments=['asr','/other'])
        self.assertEqual(self.events,['bind'])
    def test_existing_module_dispatch_remains_unguarded(self):
        previous=list(sys.argv)
        try:self.run_dispatch(role='module:pip',arguments=['--version'])
        finally:sys.argv=previous
        self.assertEqual(self.events,['bind','run-module'])
        self.assertEqual(self.saved,{})
    def test_preflight_import_is_stdlib_only_until_execute(self):
        tree=ast.parse((self.retained/'import_preflight.py').read_bytes())
        imports=[a.name.split('.')[0] for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(set(imports)&{'torch','transformers','funasr','numpy'})
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        self.assertTrue(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.walk(execute)))
    def test_preflight_command_precedes_model_download(self):
        tree=ast.parse((self.retained/'setup_locked.py').read_bytes())
        source=(self.retained/'setup_locked.py').read_text()
        lines=source.splitlines()
        preflight=[i for i,line in enumerate(lines,1) if "stage='import_native_preflight'" in line]
        self.assertEqual(len(preflight),1)
        later=[i for i,line in enumerate(lines,1) if i>preflight[0] and 'for name, plan in models.items()' in line]
        self.assertEqual(len(later),1)
        self.assertLess(preflight[0],later[0])


class CleanupLifecycleSmokeTests(unittest.TestCase):
    """Stdlib-only synthetic Docker/kernel faults; never executes a real client."""
    @classmethod
    def setUpClass(cls):
        path = SOURCE / 'research/qwen12-native-admission-v1/cleanup_lifecycle_smoke.py'
        spec = importlib.util.spec_from_file_location('cleanup_smoke_under_test', path)
        cls.smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.smoke)
        previous = list(sys.path)
        try:
            sys.path.insert(0, str(SOURCE / 'research/source-screen32-v1'))
            cls.original = cls.smoke.host.load('cleanup_smoke_original_fixture', SOURCE / 'research/source-screen32-v1/hosted_run.py')
        finally:
            sys.path[:] = previous

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = {'GITHUB_REPOSITORY': self.smoke.REPOSITORY, 'GITHUB_EVENT_NAME': 'pull_request',
                    'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_JOB': self.smoke.JOB, 'GITHUB_ACTIONS': 'true',
                    'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64',
                    'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '38123456789', 'GITHUB_REF': 'refs/pull/123/merge',
                    'GITHUB_HEAD_REF': self.smoke.BRANCH, 'RUNNER_TEMP': str(self.root)}
        repo = {'full_name': self.smoke.REPOSITORY, 'private': False}
        self.event = {'action': 'opened', 'number': 123, 'repository': copy.deepcopy(repo),
                      'pull_request': {'merge_commit_sha': 'a' * 40,
                          'head': {'repo': copy.deepcopy(repo), 'sha': 'b' * 40, 'ref': self.smoke.BRANCH},
                          'base': {'repo': copy.deepcopy(repo)}}}
        self.run = self.smoke.admission(self.env, self.event)

    def tearDown(self):
        self.temp.cleanup()

    def test_admission_binds_actual_head_and_github_merge_separately(self):
        self.assertEqual(self.run['executed_head_sha'], 'b' * 40)
        self.assertEqual(self.run['github_sha'], 'a' * 40)
        self.assertEqual(self.run['pull_request_merge_commit_sha_observed'], 'a' * 40)
        self.assertEqual(self.run['run_attempt'], 1)
        self.assertEqual(self.run['job_key'], self.smoke.JOB)

    def test_optional_event_merge_sha_is_recorded_independently(self):
        for kind, observed in (('missing', None), ('null', None), ('different', 'c' * 40)):
            event = copy.deepcopy(self.event)
            if kind == 'missing':
                del event['pull_request']['merge_commit_sha']
            else:
                event['pull_request']['merge_commit_sha'] = observed
            with self.subTest(kind=kind):
                run = self.smoke.admission(self.env, event)
                self.assertEqual(run['pull_request_merge_commit_sha_observed'], observed)
                self.assertEqual(run['github_sha'], 'a' * 40)
                self.assertEqual(run['executed_head_sha'], 'b' * 40)
                self.assertNotEqual(run['pull_request_merge_commit_sha_observed'], run['github_sha'])

    def test_malformed_nonnull_event_merge_sha_fails_closed(self):
        for observed in ('', 'c' * 39, 'c' * 41, 'G' * 40, 'c' * 40 + '\n', 123, True, [], {}):
            event = copy.deepcopy(self.event)
            event['pull_request']['merge_commit_sha'] = observed
            with self.subTest(observed=observed), self.assertRaisesRegex(ValueError, 'observed PR merge SHA'):
                self.smoke.admission(self.env, event)

    def test_optional_merge_observation_never_relaxes_actual_head_binding(self):
        event = copy.deepcopy(self.event)
        event['pull_request']['merge_commit_sha'] = None
        event['pull_request']['head']['sha'] = 'c' * 40
        run = self.smoke.admission(self.env, event)
        # The runner selected a different head from the event: stop before any
        # source bytes, image acquisition, or container effects can be used.
        with patch.object(self.smoke.subprocess, 'run',
                return_value=SimpleNamespace(stdout=b'b' * 40 + b'\n', stderr=b'')) as git, \
                self.assertRaisesRegex(ValueError, 'actual PR head'):
            self.smoke.source_binding(run, self.root)
        git.assert_called_once()

    def test_invalid_context_stops_before_any_effect(self):
        changes = [({'GITHUB_RUN_ATTEMPT': '2'}, {}), ({'GITHUB_HEAD_REF': 'wrong'}, {}),
                   ({'GITHUB_JOB': 'qwen12'}, {}), ({'RUNNER_ENVIRONMENT': 'self-hosted'}, {}),
                   ({'GITHUB_REPOSITORY': 'other/repository'}, {}), ({'GITHUB_SHA': 'g' * 40}, {})]
        changes += [({}, {'action': action}) for action in ('synchronize', 'reopened', 'ready_for_review')]
        for env_delta, event_delta in changes:
            with self.subTest(env=env_delta, event=event_delta), \
                    patch.object(self.smoke, 'source_binding', side_effect=AssertionError('effects forbidden')), \
                    self.assertRaises(ValueError):
                self.smoke.execute({**self.env, **env_delta}, {**self.event, **event_delta})
        for section in ('head', 'base'):
            event = copy.deepcopy(self.event)
            event['pull_request'][section]['repo']['full_name'] = 'foreign/repository'
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.smoke.admission(self.env, event)
        event = copy.deepcopy(self.event)
        event['pull_request']['head']['ref'] = 'wrong'
        with self.assertRaises(ValueError):
            self.smoke.admission(self.env, event)

    def test_source_binding_compares_actual_git_bytes_and_sha(self):
        for name in self.smoke.SOURCE_FILES:
            write(self.root / name, ('fixture source ' + name).encode())
        def git(argv, **kwargs):
            self.assertEqual(kwargs['timeout'], 5)
            body = (('b' * 40 + '\n').encode() if argv[-2:] == ['rev-parse', 'HEAD'] else
                    (self.root / argv[-1].split(':', 1)[1]).read_bytes())
            return SimpleNamespace(stdout=body, stderr=b'')
        with patch.object(self.smoke.subprocess, 'run', side_effect=git):
            result = self.smoke.source_binding(self.run, self.root)
        self.assertEqual(result['staged_code'], {self.smoke.GATE: result['source_files'][self.smoke.GATE]})
        self.assertEqual(result['runtime_files'], [])
        self.assertEqual(result['input_files'], [])
        body = {k: v for k, v in result.items() if k != 'binding_sha256'}
        self.assertEqual(result['binding_sha256'], native.digest(native.canonical(body)))
        with patch.object(self.smoke.subprocess, 'run', return_value=SimpleNamespace(stdout=b'c' * 40 + b'\n', stderr=b'')), \
                self.assertRaisesRegex(ValueError, 'actual PR head'):
            self.smoke.source_binding(self.run, self.root)

    def test_minimal_gate_stage_excludes_repo_runtime_audio_and_credentials(self):
        binding = {'staged_code': {self.smoke.GATE: self.smoke.identity(SOURCE / self.smoke.GATE)}}
        with patch.object(self.smoke, 'ROOT', SOURCE):
            code = self.smoke.stage_gate(self.root, binding)
        self.assertEqual([str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()], [self.smoke.GATE])
        for name in ('runtime', 'input', 'output'):
            self.assertEqual(list((self.root / name).iterdir()), [])
        with patch.object(self.original.os, 'getuid', return_value=1000), patch.object(self.original.os, 'getgid', return_value=1000):
            docker = self.smoke.SmokeDocker(self.root, 10**12)
            args = self.smoke.smoke_arguments(self.original, docker, 'synthetic-smoke', code)
        self.assertEqual(args[args.index(self.smoke.IMAGE) + 1:],
            ['python', '-I', '-B', '/code/' + self.smoke.GATE, 'python', '-I', '-B', '-c', 'pass'])
        self.assertEqual(args[args.index('--network') + 1], 'none')
        self.assertEqual(args[args.index('--pull') + 1], 'never')
        self.assertEqual(args.count('--mount'), 5)
        self.assertNotIn(str(SOURCE), '\n'.join(args))
        self.assertTrue({'--read-only', '--cap-drop', '--security-opt'} <= set(args))

    def test_sixty_cpu_seconds_uses_actual_inherited_close(self):
        usage = [5]
        ledger = self.smoke.SmokeCpuLedger()
        ledger.register('scope', 'setup', lambda: usage[0])
        usage[0] = 59999999
        self.assertEqual(ledger.sample(), [])
        ledger.close('scope', 60000000)
        self.assertTrue(ledger.failed)
        self.assertTrue(ledger.receipt()['all_final_usage_verified'])
        self.assertEqual(ledger.receipt()['total_threshold_seconds'], 60)
        self.assertIs(self.smoke.SmokeCpuLedger.close, self.smoke.CpuLedger.close)
        self.assertIs(self.smoke.SmokeDocker.remove, self.smoke.host.MeteredDocker.remove)
        self.assertIs(self.smoke.SmokeDocker.finish, self.smoke.host.MeteredDocker.finish)

    def test_kernel_scope_reads_actual_limits_and_rejects_drift(self):
        cgroup = self.root / 'cgroup'
        cgroup.mkdir()
        values = {'memory.max': str(12 * 1024**3), 'memory.swap.max': '0', 'cpu.max': '400000 100000',
                  'pids.max': '256', 'cgroup.procs': '123456'}
        for name, value in values.items():
            (cgroup / name).write_text(value)
        status = 'NoNewPrivs:\t1\nSeccomp:\t2\nUid:\t1000 1000 1000 1000\n' + ''.join(
            key + ':\t0000000000000000\n' for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'))
        original = self.smoke.bounded_text
        def read(path):
            return status if str(path) == '/proc/123456/status' else original(path)
        row = {'cpu_path': cgroup / 'cpu.stat', 'cgroup_inode': cgroup.stat().st_ino}
        with patch.object(self.smoke, 'bounded_text', side_effect=read), patch.object(self.smoke.os, 'getuid', return_value=1000):
            self.assertTrue(self.smoke.observed_kernel_scope(row)['observed_before_gate_release'])
            (cgroup / 'memory.swap.max').write_text('1')
            with self.assertRaisesRegex(ValueError, 'kernel limits'):
                self.smoke.observed_kernel_scope(row)

    def fixture(self, *, fault=None):
        import time
        code = self.root / 'code'
        for name in ('code', 'runtime', 'input', 'output'):
            (self.root / name).mkdir()
        docker = self.smoke.SmokeDocker(self.root, time.monotonic() + 180)
        name = 'kws-cleanup-smoke-' + self.run['run_id'] + '-1'
        cid = 'd' * 64
        image_env = ['PATH=/usr/local/bin:/usr/bin:/bin']
        events = []
        state = {'Running': True, 'Paused': False, 'Pid': 123456, 'OOMKilled': False}
        record = {}
        def raw(*args, timeout=5):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 5)
            events.append(args[0])
            if args[0] == 'create':
                command = list(args)
                mounts = []
                for i, item in enumerate(command[:-1]):
                    if item != '--mount': continue
                    pieces = dict((part.split('=', 1) if '=' in part else (part, True)) for part in command[i + 1].split(','))
                    mounts.append({'Destination': pieces['dst'], 'Source': pieces['src'], 'Type': 'bind', 'RW': not pieces.get('readonly', False)})
                env = [command[i + 1] for i, item in enumerate(command[:-1]) if item == '--env']
                tmpfs = dict(item.split(':', 1) for i, item in enumerate(command) if i and command[i - 1] == '--tmpfs')
                record.update(Id=cid, Image=self.smoke.IMAGE_CONFIG, Mounts=mounts, State=state,
                    Config={'Image': self.smoke.IMAGE, 'Cmd': command[command.index(self.smoke.IMAGE) + 1:],
                            'Entrypoint': None, 'User': '1000:1000', 'Env': image_env + env},
                    HostConfig={'LogConfig': {'Type': 'local', 'Config': {'max-size': '1m', 'max-file': '1', 'compress': 'false'}},
                        'Memory': 12 * 1024**3, 'MemorySwap': 12 * 1024**3, 'NanoCpus': 4000000000, 'PidsLimit': 256,
                        'ReadonlyRootfs': True, 'Privileged': False, 'CgroupnsMode': 'private', 'PidMode': '', 'IpcMode': 'private',
                        'CapAdd': None, 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'NetworkMode': 'none',
                        'RestartPolicy': {'Name': 'no'}, 'Tmpfs': tmpfs,
                        'Ulimits': [{'Name': 'core', 'Soft': 0, 'Hard': 0}, {'Name': 'nofile', 'Soft': 1024, 'Hard': 1024},
                                    {'Name': 'fsize', 'Soft': 8388608, 'Hard': 8388608}]})
                if fault == 'contract': record['HostConfig']['NetworkMode'] = 'bridge'
                if fault == 'create': raise self.smoke.subprocess.TimeoutExpired(['docker'], 5)
                return cid + '\n'
            self.assertEqual(args[-1], cid)  # No name-based post-create Docker actions.
            if args[0] == 'inspect':
                if '--format' in args: return json.dumps(state)
                return json.dumps([record])
            if args[0] == 'start':
                if fault == 'start': raise RuntimeError('sensitive start detail')
                return cid + '\n'
            if args[0] == 'pause':
                if fault == 'pause': raise RuntimeError('private pause detail')
                state['Paused'] = True
                return cid + '\n'
            if args[0] == 'rm':
                if fault == 'remove': raise RuntimeError('private remove detail')
                return cid + '\n'
            if args[0] == 'ps': return cid + '\n'
            self.fail('unexpected Docker operation')
        original_stat, original_read = Path.stat, Path.read_text
        def stat(path, *args, **kwargs):
            if str(path) == '/sys/fs/cgroup/mock-smoke': return SimpleNamespace(st_ino=98765, st_mode=0o040755)
            return original_stat(path, *args, **kwargs)
        def read(path, *args, **kwargs):
            if str(path) == '/proc/123456/cgroup': return '0::/mock-smoke\n'
            if str(path) == '/sys/fs/cgroup/mock-smoke/cpu.stat': return 'usage_usec 100\n'
            return original_read(path, *args, **kwargs)
        def scope(row):
            events.append('kernel-scope')
            self.assertFalse((row['gate'] / 'go').exists())
            if fault == 'scope': raise ValueError('bad kernel scope')
            return {'observed_before_gate_release': True}
        completed = docker.completed
        def finish_child(scope_name):
            self.assertTrue((docker.names[scope_name]['gate'] / 'go').is_file())
            body = {'schema': 'qwen12-gated-command-v1', 'exit_code': 1 if fault == 'child' else 0}
            (docker.names[scope_name]['gate'] / 'result.json').write_text(json.dumps(body))
            return completed(scope_name)
        with patch.object(docker, 'raw', side_effect=raw), patch.object(docker, 'completed', side_effect=finish_child), \
                patch.object(Path, 'stat', stat), patch.object(Path, 'read_text', read), \
                patch.object(self.smoke, 'observed_kernel_scope', side_effect=scope), \
                patch.object(self.smoke.os, 'getuid', return_value=1000), patch.object(self.smoke.os, 'getgid', return_value=1000), \
                patch.object(self.smoke.signal, 'setitimer'):
            result = self.smoke.lifecycle(self.original, docker, code, self.run, image_env)
        return result, docker, events

    def test_real_lifecycle_implementation_success_single_final_close(self):
        result, docker, events = self.fixture()
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['actual_contract_verified_before_start'])
        self.assertEqual(events, ['create', 'inspect', 'start', 'inspect', 'kernel-scope', 'pause', 'inspect', 'rm'])
        self.assertTrue(docker.ledger.receipt()['all_final_usage_verified'])
        self.assertEqual(len(docker.cleanup), 1)
        self.assertTrue(docker.cleanup[0]['cleanup_ok'])
        self.assertEqual(docker.cleanup[0]['accounting_status'], 'final_usage_verified')
        before = copy.deepcopy(docker.cleanup)
        docker.finish()
        self.assertEqual(before, docker.cleanup)

    def test_actual_contract_failure_never_starts_and_still_removes(self):
        result, docker, events = self.fixture(fault='contract')
        self.assertEqual(result['status'], 'failed_no_retry')
        self.assertEqual(result['primary_error']['phase'], 'inspect')
        self.assertEqual(events, ['create', 'inspect', 'rm'])
        self.assertTrue(docker.cleanup[0]['cleanup_ok'])

    def test_unknown_create_does_not_guess_ownership_or_retry(self):
        result, docker, events = self.fixture(fault='create')
        self.assertEqual(events, ['create'])
        self.assertIsNotNone(result['primary_error'])
        self.assertIsNotNone(result['cleanup_error'])
        self.assertEqual(docker.cleanup[0]['ownership'], 'unknown')
        self.assertFalse(docker.cleanup[0]['cleanup_ok'])

    def test_bad_actual_kernel_scope_does_not_release_gate(self):
        result, docker, events = self.fixture(fault='scope')
        self.assertEqual(result['status'], 'failed_no_retry')
        self.assertEqual(result['primary_error']['phase'], 'start')
        self.assertNotIn('go', [p.name for p in next(iter(docker.names.values()))['gate'].iterdir()])
        self.assertEqual(events[-3:], ['pause', 'inspect', 'rm'])

    def test_child_failure_and_cleanup_failure_both_remain(self):
        result, docker, events = self.fixture(fault='child')
        self.assertEqual(result['primary_error']['phase'], 'wait')
        self.assertIsNone(result['cleanup_error'])
        self.assertEqual(result['status'], 'failed_no_retry')
        self.assertTrue(docker.cleanup[0]['cleanup_ok'])

    def test_unverified_pause_does_not_block_owned_removal(self):
        result, docker, events = self.fixture(fault='pause')
        self.assertIsNone(result['primary_error'])
        self.assertIsNotNone(result['cleanup_error'])
        self.assertEqual(events[-2:], ['pause', 'rm'])
        self.assertFalse(docker.cleanup[0]['cleanup_ok'])
        self.assertNotIn('private', json.dumps(docker.cleanup))

    def test_single_image_acquisition_uses_original_and_bounds_inspect(self):
        calls = []
        def docker(*args, timeout=30):
            calls.append(('inspect', timeout))
            return json.dumps([{'Config': {'Env': ['PATH=/bin']}}])
        original = SimpleNamespace(docker=docker)
        def pull(lock, path):
            calls.append(('pull', lock['image']))
            original.docker('image', 'inspect', lock['image'])
            return {'daemon_completion_verified': True}
        original.pull_image = pull
        @contextmanager
        def alarm(deadline):
            calls.append(('alarm', deadline))
            yield
        lock = {'image': self.smoke.IMAGE, 'platform': 'linux/amd64', 'manifest': {'config': {'digest': self.smoke.IMAGE_CONFIG}}}
        with patch.object(self.smoke, 'phase_alarm', alarm), patch.object(self.smoke.time, 'monotonic', return_value=10):
            receipt, env = self.smoke.acquire_image(original, lock, self.root, 180)
        self.assertEqual(calls, [('alarm', 130), ('pull', self.smoke.IMAGE), ('inspect', 5)])
        self.assertIs(original.docker, docker)
        self.assertEqual(env, ['PATH=/bin'])

    def test_phase_alarm_restores_absolute_host_remainder_after_failure(self):
        now = [10.0]
        calls = []
        handlers = {}
        def outer(signum, frame):
            raise TimeoutError('outer')
        def set_handler(sig, handler):
            handlers['current'] = handler
        def timer(which, seconds, interval=0):
            calls.append((seconds, interval))
        with patch.object(self.smoke.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(self.smoke.signal, 'getsignal', return_value=outer), \
                patch.object(self.smoke.signal, 'getitimer', return_value=(170.0, 0.0)), \
                patch.object(self.smoke.signal, 'signal', side_effect=set_handler), \
                patch.object(self.smoke.signal, 'setitimer', side_effect=timer):
            with self.assertRaises(TimeoutError):
                with self.smoke.phase_alarm(130.0):
                    now[0] = 130.0
                    try:
                        handlers['current'](self.smoke.signal.SIGALRM, None)
                    finally:
                        now[0] = 150.0  # Legacy client cleanup consumes twenty seconds.
        self.assertEqual(calls, [(120.0, 0), (50.0, 0), (0, 0), (30.0, 0.0)])
        self.assertIs(handlers['current'], outer)

    def test_execute_retains_cpu_cleanup_and_partial_lifecycle_on_exception(self):
        retained = []
        fake = SimpleNamespace(ledger=SimpleNamespace(receipt=lambda: {'final': 'synthetic'}),
                               cleanup=[{'status': 'removal_unconfirmed'}])
        def read(path, *args):
            if Path(path).name == 'execution-release.json': return {'enabled': False}
            if Path(path).name == 'cleanup-receipt.json': return {'containers': fake.cleanup}
            return {}
        def fail_lifecycle(*args, result):
            result.update(status='failed_no_retry', primary_error={'phase': 'inspect', 'error_type': 'ValueError'})
            raise RuntimeError('secret fixture content')
        with patch.object(self.smoke, 'source_binding', return_value={}), \
                patch.object(self.smoke, 'stage_gate', return_value=self.root / 'code'), \
                patch.object(self.smoke, 'read_json', side_effect=read), \
                patch.object(self.smoke.host, 'load', return_value=SimpleNamespace()), \
                patch.object(self.smoke, 'acquire_image', return_value=({'daemon_completion_verified': True}, [])), \
                patch.object(self.smoke, 'SmokeDocker', return_value=fake), \
                patch.object(self.smoke, 'lifecycle', side_effect=fail_lifecycle), \
                patch.object(self.smoke, 'emit_receipt', side_effect=lambda value: retained.append(copy.deepcopy(value))), \
                patch.object(self.smoke.signal, 'setitimer'):
            self.assertEqual(self.smoke.execute(self.env, self.event), 1)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['cpu'], {'final': 'synthetic'})
        self.assertEqual(retained[0]['cleanup'], {'containers': fake.cleanup})
        self.assertEqual(retained[0]['lifecycle']['primary_error']['phase'], 'inspect')
        self.assertNotIn('secret', json.dumps(retained))

    def test_image_failure_never_starts_lifecycle_or_second_acquire(self):
        retained = []
        with patch.object(self.smoke, 'source_binding', return_value={}), \
                patch.object(self.smoke, 'stage_gate', return_value=self.root / 'code'), \
                patch.object(self.smoke, 'read_json', return_value={'enabled': False}), \
                patch.object(self.smoke.host, 'load', return_value=SimpleNamespace()), \
                patch.object(self.smoke, 'acquire_image', side_effect=TimeoutError('private image failure')) as acquire, \
                patch.object(self.smoke, 'lifecycle', side_effect=AssertionError('must not start')) as lifecycle, \
                patch.object(self.smoke, 'emit_receipt', side_effect=lambda value: retained.append(copy.deepcopy(value))), \
                patch.object(self.smoke.signal, 'setitimer'):
            self.assertEqual(self.smoke.execute(self.env, self.event), 1)
        acquire.assert_called_once()
        lifecycle.assert_not_called()
        error = retained[0]['primary_error']
        self.assertEqual(set(error), {'error_type', 'frames', 'frames_truncated'})
        self.assertEqual(error['error_type'], 'TimeoutError')
        self.assertIs(error['frames_truncated'], False)
        self.assertGreater(len(error['frames']), 0)
        self.assertLessEqual(len(error['frames']), self.smoke.host.MAX_HOST_ERROR_FRAMES)
        self.assertTrue(all(frame == {'file': '<redacted>', 'function': '<redacted>', 'line': None}
                            for frame in error['frames']))
        self.assertLessEqual(len(json.dumps(error).encode()), self.smoke.host.MAX_HOST_ERROR_BYTES)
        self.assertEqual(retained[0]['model_calls'], 0)
        self.assertNotIn('private image failure', json.dumps(retained))

    def test_receipt_bound_is_enforced_without_printing_body(self):
        with patch('builtins.print') as printed, self.assertRaisesRegex(ValueError, '32 KiB'):
            self.smoke.emit_receipt({'padding': 'x' * self.smoke.RECEIPT_BYTES})
        printed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
