"""Fictional kernel/container/PCM fixtures only; no Docker or ML calls."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_scope as scope
import hosted_run as host
import tts_worker as tts


def kernel():
    return {'/proc/self/cgroup': '0::/\n',
            '/proc/self/mountinfo': '1 0 0:1 / / ro - overlay overlay ro\n2 1 0:2 / /sys/fs/cgroup ro - cgroup2 cgroup ro\n',
            '/proc/self/status': 'NoNewPrivs:\t1\nSeccomp:\t2\nNSpid:\t1\n' +
              ''.join(name + ':\t0000000000000000\n' for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')),
            **{'/sys/fs/cgroup/' + name: value for name, value in
               {'memory.max': str(scope.MEMORY), 'memory.swap.max': '0', 'cpu.max': '400000 100000', 'pids.max': '256'}.items()}}


class ScopeTests(unittest.TestCase):
    def run_scope(self, values, interfaces=('lo',), uid=1001, network=True):
        path = mock.Mock(); path.iterdir.return_value = [types.SimpleNamespace(name=name) for name in interfaces]
        with mock.patch.object(scope, 'read', side_effect=lambda path: values[path]), \
                mock.patch.object(scope.os, 'geteuid', return_value=uid), mock.patch.object(scope, 'Path', return_value=path):
            return scope.verify_runtime_scope(network_required=network)

    def test_actual_limit_readback(self):
        result = self.run_scope(kernel())
        self.assertTrue(result['network_disabled']); self.assertEqual(result['limits']['memory.swap.max'], '0')

    def test_unknown_unlimited_or_wrong_limits_fail(self):
        for key, values in [('memory.max', ('max', str(scope.MEMORY + 1))), ('memory.swap.max', ('max', '1')),
                            ('cpu.max', ('max 100000', '800000 100000')), ('pids.max', ('max', '512'))]:
            for value in values:
                data = kernel(); data['/sys/fs/cgroup/' + key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError): self.run_scope(data)

    def test_namespace_privilege_and_writable_root_fail(self):
        for key, value in [('/proc/self/cgroup', '0::/user.slice/other'),
                           ('/proc/self/mountinfo', kernel()['/proc/self/mountinfo'].replace('/ / ro', '/ / rw')),
                           ('/proc/self/status', kernel()['/proc/self/status'].replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0')),
                           ('/proc/self/status', kernel()['/proc/self/status'].replace('0000000000000000', '0000000000000001'))]:
            data = kernel(); data[key] = value
            with self.assertRaises(RuntimeError): self.run_scope(data)
        with self.assertRaises(RuntimeError): self.run_scope(kernel(), uid=0)

    def test_network_exception_is_setup_only(self):
        with self.assertRaises(RuntimeError): self.run_scope(kernel(), ('eth0', 'lo'))
        self.assertFalse(self.run_scope(kernel(), ('eth0', 'lo'), network=False)['network_disabled'])


class HostTests(unittest.TestCase):
    def command(self, profile='tts', stage='tts'):
        with mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=1001):
            return host.container_args('fixed', 'python@sha256:' + 'a' * 64, Path('/stage/code'),
                Path('/stage/runtime'), Path('/stage/output'), profile, stage,
                Path('/stage/input') if profile == 'asr' else None)

    def test_fixed_container_command_has_real_limits_and_offline_readonly_inputs(self):
        command = self.command()
        for option, value in [('--memory', str(scope.MEMORY)), ('--memory-swap', str(scope.MEMORY)),
                               ('--cpus', '4'), ('--pids-limit', '256'), ('--network', 'none'),
                               ('--cgroupns', 'private'), ('--cap-drop', 'ALL')]:
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn('--read-only', command)
        self.assertIn('type=bind,src=/stage/runtime,dst=/runtime,readonly', command)
        self.assertFalse(any(value in command for value in ('--privileged', '--pid=host', '--network=host')))
        self.assertNotIn('/input', ' '.join(command))

    def test_setup_network_and_asr_input_are_explicit(self):
        self.assertEqual(self.command(stage='setup')[self.command(stage='setup').index('--network') + 1], 'bridge')
        command = self.command('asr', 'qwen06')
        self.assertIn('type=bind,src=/stage/input,dst=/input,readonly', command)
        self.assertIn('/code/research/source-screen32-v1/asr_worker.py', command)
        self.assertNotIn('tts_worker.py', ' '.join(command))

    def test_no_arbitrary_stage_or_command(self):
        with self.assertRaises(ValueError): self.command(stage='bash')

    def test_frozen_profiles_bind_sources_and_keep_asr_blind(self):
        freeze = json.loads((HERE / 'execution-freeze.json').read_text())
        for name, digest in freeze['files'].items():
            self.assertEqual(host.file_hash(host.ROOT / name), digest, name)
        reused = set(json.loads((HERE / 'reuse-pins.json').read_text()))
        self.assertTrue(reused <= set(freeze['profiles']['tts']))
        for name in freeze['profiles']['asr']:
            self.assertNotIn(name, ('research/source-screen32-v1/contract.py', 'research/source-screen32-v1/plan.json'))
            raw = (host.ROOT / name).read_bytes()
            for text in ('你好小窝', '小窝小窝', '你好小屋', '小屋小屋', '成年女性', '成年男性', 'screen32-001'):
                self.assertNotIn(text.encode(), raw, name)
        with tempfile.TemporaryDirectory() as temporary:
            staged = Path(temporary)
            host.stage_code(freeze, 'tts', staged)
            result = host.subprocess.run([sys.executable, '-I', '-B', str(staged / 'research/source-screen32-v1/contract.py')],
                                         capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['local_reuse_pins_verified'], len(reused))

    def test_release_false_blocks_before_docker(self):
        release = {'schema': 'screen32-execution-release-v1', 'experiment': host.EXPERIMENT,
                   'approved': False, 'source_freeze_sha256': 'a' * 64}
        with mock.patch.object(host, 'bounded_json', return_value=release), \
                mock.patch.object(host, 'file_hash', return_value='a' * 64), \
                mock.patch.object(host, 'docker') as docker, self.assertRaises(ValueError):
            host.verify_admission({})
        docker.assert_not_called()

    def test_failed_image_pull_does_not_claim_daemon_transfer_cutoff(self):
        process = mock.Mock(); process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen', return_value=process), \
                mock.patch.object(host, 'network_received', side_effect=[{'eth0': 1}, {'eth0': 1 + host.IMAGE_TRANSFER_RESERVE + 1}]), \
                mock.patch.object(host, 'docker') as docker:
            target = Path(temporary) / 'image-transfer.json'
            with self.assertRaises(ValueError): host.pull_image({'image': 'fictional-image'}, target)
            record = json.loads(target.read_text())
            self.assertEqual(record['status'], 'stopped_before_setup_daemon_completion_unverified')
            self.assertTrue(record['daemon_fetch_may_continue'])
            self.assertFalse(record['daemon_completion_verified'])
            self.assertFalse(record['cumulative_image_transfer_bytes_verified'])
            process.terminate.assert_called_once(); docker.assert_not_called()

    def test_terminal_tts_load_failure_retains_all32(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = host.terminal_tts(Path(temporary))
            self.assertEqual(len(receipt['ledger']), 32)
            self.assertTrue(all(row['attempts'] == 0 and row['status'] == 'NOT_RUN' for row in receipt['ledger']))

    def test_tts_claim_precedes_receipt_even_on_interrupt(self):
        for body in (b"screen32-001\n", b""):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tts.atomic(root / 'tts-receipt.json', {'schema': 'screen32-tts-receipt-v1', 'status': 'started',
                    'ledger': tts.initial_ledger(), 'outcomes': [], 'human_gold': False, 'training_admitted': False})
                (root / 'screen32-001.claim').write_bytes(body)
                receipt = host.terminal_tts(root)
                self.assertEqual(receipt['ledger'][0]['attempts'], 1)
                self.assertEqual(receipt['ledger'][0]['status'], 'FAILED_NO_RETRY')
                self.assertTrue(all(row['attempts'] == 0 for row in receipt['ledger'][1:]))
                self.assertEqual(receipt['outcomes'][0]['status'], 'failed_no_retry')

    def test_host_kills_container_on_watchdog_failure(self):
        calls = []
        def docker(*args, **kwargs):
            calls.append(args)
            if args == ('inspect', 'test'): return '[{}]'
            return ''
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host, 'docker', side_effect=docker), \
                mock.patch.object(host, 'validate_container_inspect'), \
                mock.patch.object(host.time, 'monotonic', side_effect=[0, 2000]), self.assertRaises(ValueError):
            host.supervise_container(['create'], 'test', Path(temporary), 'setup', 7200)
        self.assertEqual(calls[-1], ('rm', '--force', 'test'))

    def test_primary_projection_preserves_unknown_and_never_creates_human_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = {'schema': 'blind-source-screen32-job-v1', 'clips': [{'audio_id': 'clip-000001',
                   'audio_path': 'audio/' + 'a' * 64 + '.wav', 'wav_sha256': 'a' * 64}]}
            for model in ('qwen06', 'sensevoice'):
                (root / model).mkdir()
                tts.atomic(root / model / 'terminal-outcomes.json', [{'opaque_id': 'clip-000001',
                    'status': 'success', 'raw_text': 'fictional words', 'completeness': 'unknown', 'quality_flags': []}])
            host.freeze_primary(root, job)
            dispute = json.loads((root / 'frozen-disputes.json').read_text())
            self.assertEqual(dispute['unresolved_audio_ids'], ['clip-000001'])
            self.assertEqual(dispute['whisper_audio_ids'], [])
            self.assertFalse(dispute['human_gold']); self.assertFalse(dispute['training_admitted'])
            freeze = json.loads((root / 'primary-raw-freeze.json').read_text())
            self.assertEqual(freeze['primary_raw_sha256'], host.file_hash(root / 'primary-raw.json'))

    def test_truncated_setup_receipt_still_publishes_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'tts').mkdir(); (root / 'runtime').mkdir()
            (root / 'host-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json').write_text('{}')
            (root / 'runtime/setup-receipt.json').write_bytes(b'{"started":')
            artifact = host.public_evidence(root, 'tts')
            summary = json.loads((artifact / 'setup-public.json').read_text())
            self.assertEqual(summary['verification_evidence'], 'UNAVAILABLE_OR_TRUNCATED')
            self.assertTrue((artifact / 'tts/tts-receipt.json').exists())

    def test_known_interrupted_temporaries_do_not_hide_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'tts').mkdir()
            (root / 'host-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json.tmp').write_bytes(b'{')
            artifact = host.public_evidence(root, 'tts')
            freeze = json.loads((artifact / 'artifact-freeze.json').read_text())
            self.assertIn('tts/tts-receipt.json', freeze['files'])
            self.assertEqual(freeze['incomplete_temporary_files_excluded'], ['tts/tts-receipt.json.tmp'])
            self.assertFalse((artifact / 'tts/tts-receipt.json.tmp').exists())

    def test_blind_handoff_only_selected_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'out'; output.mkdir()
            pcm = struct.pack('<hh', 100, -100)
            body = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 40, b'WAVE', b'fmt ', 16,
                               1, 1, 16000, 32000, 2, 16, b'data', 4) + pcm
            audio = output / 'screen32-001.wav'; audio.write_bytes(body)
            ledger = tts.initial_ledger(); ledger[0].update(status='GENERATED', attempts=1, audio={
                'wav_sha256': host.file_hash(audio), 'pcm_sha256': host.hashlib.sha256(pcm).hexdigest(),
                'frames': 2, 'sample_rate_hz': 16000, 'channels': 1, 'sample_width_bytes': 2})
            (output / 'private-plan.json').write_text('fictional source cue')
            receipt = host.pack_blind(output, root / 'blind', ledger)
            self.assertEqual(receipt['clips'], 1)
            self.assertEqual(len(receipt['files']), 2)
            self.assertNotIn('screen32-001', (root / 'blind/job.json').read_text())
            self.assertEqual(len(host.validate_blind_input(root / 'blind')['clips']), 1)
            (root / 'blind/leak.json').write_text('{}')
            with self.assertRaises(ValueError): host.validate_blind_input(root / 'blind')


class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.adapter = types.SimpleNamespace(generate_cell=self.generate)
    def generate(self, cell):
        self.calls.append(cell)
        return [0.0], {'native': {'quality_flags': [], 'native_float_values_sha256': 'a' * 64}}
    def writer(self, array, native, cell):
        index = int(cell[-3:]); digest = format(index, '064x')
        return {'wav_sha256': digest, 'pcm_sha256': digest, 'frames': 1, 'sample_rate_hz': 16000,
                'channels': 1, 'sample_width_bytes': 2}, {}
    def run_cells(self, path, writer=None):
        with mock.patch.object(tts, 'verify_runtime_scope'), mock.patch.object(tts.signal, 'alarm'):
            return tts.run_cells(self.adapter, {}, path, writer or self.writer)
    def test_sixteen_once_durable_and_original32_denominator(self):
        with tempfile.TemporaryDirectory() as path:
            receipt = self.run_cells(Path(path))
            self.assertEqual(len(self.calls), 16)
            self.assertEqual(sum(row['attempts'] for row in receipt['ledger']), 16)
            self.assertTrue(all(row['status'] == 'NOT_RUN' for row in receipt['ledger'][16:]))
            with self.assertRaises(ValueError): self.run_cells(Path(path))
            self.assertEqual(len(self.calls), 16)
    def test_writer_failure_consumes_cell_stops_and_retains_partial(self):
        def writer(array, native, cell): raise OSError('fictional output failure')
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(OSError): self.run_cells(Path(path), writer)
            receipt = json.loads((Path(path) / 'tts-receipt.json').read_text())
            self.assertEqual(self.calls, ['screen32-001'])
            self.assertEqual(receipt['ledger'][0]['status'], 'FAILED_NO_RETRY')
            self.assertEqual(receipt['ledger'][0]['attempts'], 1)
            self.assertTrue(all(row['attempts'] == 0 for row in receipt['ledger'][1:]))
            self.assertTrue((Path(path) / 'screen32-001.claim').exists())


if __name__ == '__main__': unittest.main()
