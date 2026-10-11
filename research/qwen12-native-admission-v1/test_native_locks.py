"""Synthetic byte inventories only: no model imports, downloads or inference."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = next((parent for parent in Path(__file__).resolve().parents
    if (parent / "tools/asr_native_contracts.py").is_file()), HERE)
REPO = Path(os.environ.get("KWS_NATIVE_BASE", str(DEFAULT_BASE)))
SOURCE = HERE / "source" if (HERE / "source").is_dir() else REPO
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
                      installed_source_inventory=self.extensions, frontend_config_bytes=self.frontend)
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


if __name__ == '__main__':
    unittest.main()
