"""Offline fixtures only: never run pip, download packages or load a model."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phased_setup as p


class PhasedTests(unittest.TestCase):
    def test_original_locks_and_no_model_import(self):
        wheels, source = p.load_locks()
        self.assertEqual((len(wheels), len(source)), (22, 22))
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(p.MODEL_BYTES, 483617219)

    def test_changed_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in p.LOCK_HASHES:
                (root / name).write_bytes((p.LOCKS / name).read_bytes() + b' ')
            with self.assertRaises(p.SetupFault): p.load_locks(root)

    def test_model_requires_install_before_any_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root / 'runtime'; out = root / 'out'
            runtime.mkdir(); out.mkdir()
            setup = p.PhasedSetup(runtime, out)
            with patch.object(setup, 'materialize') as download:
                with self.assertRaises(FileNotFoundError):
                    setup.run('model', p.Diagnostics(out, 'model'), {}, object())
                download.assert_not_called()

    def test_failed_phase_blocks_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'phased-setup-failure.json').write_text('{}')
            with self.assertRaises(p.SetupFault):
                p.PhasedSetup(root, root).run('source', p.Diagnostics(root, 'source'), {}, object())

    def test_downloader_retains_exact_size_hash(self):
        import io
        class Opener:
            def open(self, url, timeout): return io.BytesIO(b'fixture')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'out'; out.mkdir()
            row = {'path': 'x', 'url': 'https://raw.githubusercontent.com/fixed',
                   'size_bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}
            setup = p.PhasedSetup(root, out)
            setup.materialize([row], 'source', p.Diagnostics(out, 'source'), Opener())
            self.assertEqual((root / 'source/x').read_bytes(), b'fixture')
            self.assertNotIn('torch', sys.modules)

    def test_body_oversize_fails(self):
        import io
        class Opener:
            def open(self, url, timeout): return io.BytesIO(b'oversized')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'path': 'x', 'url': 'https://raw.githubusercontent.com/fixed',
                   'size_bytes': 2, 'sha256': '0' * 64}
            with self.assertRaises(p.SetupFault):
                p.PhasedSetup(root, root).materialize([row], 'source', p.Diagnostics(root, 'source'), Opener())

    def test_model_rejects_tampered_installed_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root / 'runtime'; out = root / 'out'
            runtime.mkdir(); out.mkdir()
            installed = runtime / 'venv/lib/python3.12/site-packages'; installed.mkdir(parents=True)
            module = installed / 'public.py'; module.write_text('x=1')
            p.write_once(runtime / 'installed-inventory.json', {
                'schema': 'whisper-screen-installed-runtime-v1', 'files': {'public.py': p.identity(module)}})
            p.write_once(out / 'phase-install-complete.json', {
                'status': 'complete', 'phase': 'install', 'lock_sha256': p.LOCK_HASHES})
            module.write_text('x=2')
            setup = p.PhasedSetup(runtime, out)
            with patch.object(setup, 'materialize') as download:
                with self.assertRaises(ValueError):
                    setup.run('model', p.Diagnostics(out, 'model'), {}, object())
                download.assert_not_called()

    def test_finalize_preserves_adapter_contract_without_legacy_body_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root / 'runtime'; out = root / 'out'
            runtime.mkdir(); out.mkdir()
            for name in ('source/whisper/assets', 'model', 'wheels'):
                (runtime / name).mkdir(parents=True)
            (runtime / 'source/whisper/assets/test').write_bytes(b'source')
            (runtime / 'model/small.pt').write_bytes(b'model')
            (runtime / 'wheels/a.whl').write_bytes(b'wheel')
            (runtime / 'installed-inventory.json').write_text('{}')
            sources = [{'path': 'whisper/assets/test', 'size_bytes': 6,
                        'sha256': p.file_hash(runtime / 'source/whisper/assets/test')}]
            wheels = [{'filename': 'a.whl', 'size_bytes': 5,
                       'sha256': p.file_hash(runtime / 'wheels/a.whl')}]
            setup = p.PhasedSetup(runtime, out)
            with patch.object(setup, 'prior'), patch.object(p, 'MODEL_BYTES', 5), \
                 patch.object(p, 'MODEL_SHA', p.file_hash(runtime / 'model/small.pt')), \
                 patch.object(p, 'validated_acquisition', return_value={'total_read_bytes': 42}):
                setup.finalize(wheels, sources, {'network_disabled': True})
            asset = p.read_json(runtime / 'asset-lock.json')
            self.assertEqual(asset['schema'], 'whisper-screen-loaded-assets-lock-v1')
            self.assertEqual(set(asset['files']), {'small.pt', 'whisper/assets/test'})
            receipt = p.read_json(out / 'setup-receipt.json')
            self.assertEqual(receipt['schema'], 'whisper-screen-setup-v1')
            self.assertNotIn('total_download_bytes', receipt)
            self.assertEqual(receipt['asset_lock_sha256'], p.file_hash(runtime / 'asset-lock.json'))
            self.assertFalse((out / 'small.pt').exists())

    def receipt(self):
        return {'schema': 'bounded-host-acquisition-v1', 'status': 'complete',
                'aggregate_cap': p.LIMITS['download_bytes'], 'total_read_bytes': 24,
                'byte_scope': 'bidirectional_tunnel_socket_recv_before_forwarding',
                'assets': [{'asset_id': 'do-not-copy-remote-content', 'kind': kind,
                            'status': 'verified', 'observed_ciphertext_bytes': 1,
                            'budget_charge_bytes': 1} for kind in ['wheel'] * 22 + ['source', 'model']]}

    def test_receipt_truthful_and_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'; path.write_text(json.dumps(self.receipt()))
            value = p.validated_acquisition(path)
            self.assertEqual(value['total_read_bytes'], 24)
            self.assertNotIn('total_download_bytes', value)
            self.assertNotIn('do-not-copy', json.dumps(value))

    def test_receipt_rejects_reset_or_bad_metrics(self):
        for change in ('reset', 'over_cap', 'missing_plan', 'bad_kind', 'bad_status', 'boolean'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                value = self.receipt()
                if change == 'reset': value['total_read_bytes'] = 23
                elif change == 'over_cap': value['total_read_bytes'] = p.LIMITS['download_bytes'] + 1
                elif change == 'missing_plan': value['assets'].pop()
                elif change == 'bad_kind': value['assets'][-1]['kind'] = 'wheel'
                elif change == 'bad_status': value['assets'][0]['status'] = 'failed'
                else: value['assets'][0]['observed_ciphertext_bytes'] = True
                path = Path(tmp) / 'receipt.json'; path.write_text(json.dumps(value))
                with self.assertRaises(p.SetupFault): p.validated_acquisition(path)

    def test_diagnostics_never_echo_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); d = p.Diagnostics(root, 'model')
            d.fail(ValueError('private secret model content'))
            self.assertNotIn('private secret', ''.join(x.read_text() for x in root.iterdir()))

    def test_install_only_offline_arguments_and_verified_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root / 'runtime'; output = root / 'out'
            runtime.mkdir(); output.mkdir(); (runtime / 'wheels').mkdir()
            wheel = runtime / 'wheels/a.whl'; wheel.write_bytes(b'fixture')
            wheels = [{'filename': 'a.whl', 'size_bytes': 7, 'sha256': p.file_hash(wheel)}]
            calls = []
            def fake_venv(path):
                self.assertEqual(tempfile.gettempdir(), str(root))
                self.assertEqual(os.environ['TMPDIR'], str(root))
                self.assertEqual(os.environ['PIP_CONFIG_FILE'], '/dev/null')
                self.assertNotIn('HTTPS_PROXY', os.environ)
                self.assertNotIn('AWS_SECRET_ACCESS_KEY', os.environ)
                installed = path / 'lib/python3.12/site-packages'; installed.mkdir(parents=True)
                (installed / 'public.py').write_text('x=1')
            def fake_run(args, **kwargs): calls.append((args, kwargs))
            with patch.dict(os.environ, {'HTTPS_PROXY': 'fixture', 'AWS_SECRET_ACCESS_KEY': 'fixture'}), \
                 patch.object(p, 'BOOTSTRAP_TMP', root), \
                 patch.object(p.venv.EnvBuilder, 'create', side_effect=fake_venv), \
                 patch.object(p.subprocess, 'run', side_effect=fake_run), \
                 patch.object(p.shutil, 'disk_usage', return_value=type('D', (), {'free': 10 * 1024**3})()):
                p.PhasedSetup(runtime, output).install(wheels, p.Diagnostics(output, 'install'))
            self.assertIn('--no-index', calls[0][0]); self.assertIn('--no-deps', calls[0][0])
            self.assertEqual(calls[1][0][-2:], ['pip', 'check'])
            self.assertFalse(any('PROXY' in key for key in calls[0][1]['env']))
            p.installed_inventory(runtime)

    def test_bootstrap_restores_environment_and_temp_cache_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = dict(os.environ)
            cached = tempfile.tempdir
            with patch.object(p, 'BOOTSTRAP_TMP', Path(tmp)):
                with self.assertRaises(RuntimeError):
                    with p.bootstrap_environment({'TMPDIR': tmp, 'PIP_CONFIG_FILE': '/dev/null'}):
                        self.assertEqual(tempfile.gettempdir(), tmp)
                        self.assertEqual(set(os.environ), {'TMPDIR', 'PIP_CONFIG_FILE'})
                        raise RuntimeError('fixture')
            self.assertEqual(dict(os.environ), saved)
            self.assertEqual(tempfile.tempdir, cached)

    def test_bootstrap_rejects_tempfile_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = dict(os.environ)
            with patch.object(p, 'BOOTSTRAP_TMP', Path(tmp) / 'missing'):
                with self.assertRaises(p.SetupFault):
                    with p.bootstrap_environment({'TMPDIR': str(Path(tmp) / 'missing')}):
                        self.fail('must reject fallback')
            self.assertEqual(dict(os.environ), saved)


if __name__ == '__main__': unittest.main()
