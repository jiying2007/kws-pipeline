"""Stdlib fake-file/command fixtures; no AFE, model or external command runs."""
from __future__ import annotations
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import final_afe_identity as identity
import run_final_afe_corpus as runner


class AFEExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executable = self.root / 'fixture-executable'
        self.executable.write_bytes(b'not an executable; never run')
        self.config = self.root / 'config.json'
        self.config.write_text('{}')
        self.adapter = self.root / 'adapter.json'
        self.spec = dict(schema_version=1, executable_path=str(self.executable),
            command_argv=['{executable}', '{input}', '{output}', '{result}', '--config', str(self.config)],
            config_files=[str(self.config)], **{key: 'fixture' for key in
            ('sku', 'microphone_revision', 'enclosure_revision', 'audio_route', 'toolchain')})
        self.write_adapter()
        self.expected = self.root / 'identity.json'
        self.expected.write_text(json.dumps(identity.inspect_adapter(self.adapter)[2]))
        self.wav = self.root / 'input.wav'
        with wave.open(str(self.wav), 'wb') as output:
            output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            output.writeframes(b'\0\0' * 160)
        self.row = dict(recording='unit', input_path=self.wav.name,
            input_sha256=identity.sha256_file(self.wav), distance_m=1, azimuth_deg=0,
            tags=['synthetic'], expected=[], **{key: 'fixture' for key in
            ('speaker_id', 'session_id', 'source_id', 'room_id', 'device_id')})
        self.manifest = self.root / 'manifest.json'
        self.write_manifest([self.row])
        self.output = self.root / 'new-output'

    def write_adapter(self):
        self.adapter.write_text(json.dumps(self.spec))

    def write_manifest(self, rows):
        self.manifest.write_text(json.dumps(dict(schema_version=1,
            qualification_id='fixture-only', deployment_tag='fixture-only', recordings=rows)))

    def fake_command(self, argv, **kwargs):
        self.assertEqual(argv[0], str(self.executable))
        self.assertEqual(argv[-1], str(self.config))
        Path(argv[2]).write_bytes(Path(argv[1]).read_bytes())
        Path(argv[3]).write_text('{"latency_samples": 0}')
        return types.SimpleNamespace(returncode=0, stderr='')

    def invoke(self, effect=None):
        args = ['fixture', '--manifest', str(self.manifest), '--audio-root', str(self.root),
                '--adapter', str(self.adapter), '--expected-identity', str(self.expected),
                '--output-dir', str(self.output)]
        with mock.patch.object(sys, 'argv', args), mock.patch.object(runner.subprocess, 'run',
                side_effect=effect or self.fake_command) as command, contextlib.redirect_stdout(io.StringIO()):
            self.last_command = command
            result = runner.main()
        return result, command

    def test_valid_fixture_creates_fresh_bound_evidence(self):
        result, command = self.invoke()
        self.assertEqual(result, 0)
        command.assert_called_once()
        evidence = json.loads((self.output / 'afe-evidence.jsonl').read_text())
        self.assertEqual(evidence['input_pcm_sha256'], self.row['input_sha256'])
        self.assertEqual(evidence['executable_sha256'], identity.sha256_file(self.executable))
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)

    def test_unhashed_launcher_cannot_precede_executable_placeholder(self):
        self.spec['command_argv'].insert(0, '/unhashed/runner')
        self.write_adapter()
        with self.assertRaisesRegex(ValueError, r'command_argv\[0\]'):
            identity.inspect_adapter(self.adapter)

    def test_missing_or_aliased_config_argument_is_rejected(self):
        for value in ('/different/config.json', str(self.root / '.' / 'missing.json')):
            self.spec['command_argv'][-1] = value
            self.write_adapter()
            with self.assertRaisesRegex(ValueError, 'explicit canonical'):
                identity.inspect_adapter(self.adapter)
        alias = self.root / 'alias.json'
        alias.symlink_to(self.config)
        self.spec['command_argv'][-1] = str(alias)
        self.write_adapter()
        with self.assertRaisesRegex(ValueError, 'explicit canonical'):
            identity.inspect_adapter(self.adapter)

    def test_equals_config_argument_is_supported(self):
        self.spec['command_argv'][-2:] = ['--config=' + str(self.config)]
        self.write_adapter()
        self.assertEqual(identity.inspect_adapter(self.adapter)[2]['config_files'][0]['sha256'],
                         identity.sha256_file(self.config))

    def test_duplicate_placeholder_is_rejected(self):
        self.spec['command_argv'].append('{output}')
        self.write_adapter()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            identity.inspect_adapter(self.adapter)

    def test_preexisting_outputs_fail_before_command(self):
        self.output.mkdir()
        (self.output / 'post-afe').mkdir()
        (self.output / 'post-afe/unit.wav').write_bytes(self.wav.read_bytes())
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.last_command.assert_not_called()

    def test_noop_success_cannot_create_evidence(self):
        with self.assertRaisesRegex(ValueError, 'regular output/result'):
            self.invoke(lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=''))
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_output_symlink_is_rejected(self):
        def command(argv, **kwargs):
            Path(argv[2]).symlink_to(self.wav)
            Path(argv[3]).write_text('{"latency_samples":0}')
            return types.SimpleNamespace(returncode=0, stderr='')
        with self.assertRaisesRegex(ValueError, 'regular output/result'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_input_mutation_during_command_is_rejected(self):
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            self.wav.write_bytes(b'changed input')
            return result
        with self.assertRaisesRegex(ValueError, 'raw input changed'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_config_mutation_during_command_is_rejected(self):
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            self.config.write_text('{"changed":true}')
            return result
        with self.assertRaisesRegex(ValueError, 'identity drifted during'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_executable_mutation_during_command_is_rejected(self):
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            self.executable.write_bytes(b'changed executable')
            return result
        with self.assertRaisesRegex(ValueError, 'identity drifted during'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_redirected_adapter_cannot_mask_changed_cached_executable(self):
        self.write_manifest([self.row, {**self.row, 'recording': 'second'}])
        backup = self.root / 'same-bytes-backup'
        backup.write_bytes(self.executable.read_bytes())
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            self.executable.write_bytes(b'changed cached executable')
            self.spec['executable_path'] = str(backup)
            self.write_adapter()
            return result
        with self.assertRaisesRegex(ValueError, 'identity drifted during'):
            self.invoke(command)
        self.last_command.assert_called_once()
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_later_command_cannot_change_previous_output(self):
        self.write_manifest([self.row, {**self.row, 'recording': 'second'}])
        calls = []
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            calls.append(argv)
            if len(calls) == 2:
                previous = Path(calls[0][2])
                raw = bytearray(previous.read_bytes()); raw[-1] ^= 1
                previous.write_bytes(raw)
            return result
        with self.assertRaisesRegex(ValueError, 'before evidence publication'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())
        self.assertFalse((self.output / 'afe-corpus-summary.json').exists())

    def test_later_command_cannot_change_previous_input(self):
        second_input = self.root / 'second-input.wav'
        second_input.write_bytes(self.wav.read_bytes())
        self.write_manifest([self.row, {**self.row, 'recording': 'second',
            'input_path': second_input.name, 'input_sha256': identity.sha256_file(second_input)}])
        calls = []
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            calls.append(argv)
            if len(calls) == 2:
                self.wav.write_bytes(b'previous input changed')
            return result
        with self.assertRaisesRegex(ValueError, 'before evidence publication'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-corpus-summary.json').exists())

    def test_later_command_cannot_symlink_previous_result(self):
        self.write_manifest([self.row, {**self.row, 'recording': 'second'}])
        calls = []
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            calls.append(argv)
            if len(calls) == 2:
                previous = Path(calls[0][3]); previous.unlink()
                previous.symlink_to(Path(argv[3]))
            return result
        with self.assertRaisesRegex(ValueError, 'regular output/result'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-corpus-summary.json').exists())

    def test_later_command_cannot_change_previous_timeline(self):
        self.write_manifest([self.row, {**self.row, 'recording': 'second'}])
        calls = []
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            Path(argv[3]).write_text('{"latency_samples":0,"timing_contract":"afe-output-sample-v1"}')
            calls.append(argv)
            if len(calls) == 2:
                (self.output / 'post-afe/unit.timeline.tsv').write_text('changed timeline')
            return result
        with self.assertRaisesRegex(ValueError, 'before evidence publication'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-corpus-summary.json').exists())

    def test_adapter_content_cannot_change_with_equal_public_identity(self):
        def command(argv, **kwargs):
            result = self.fake_command(argv, **kwargs)
            self.adapter.write_text(json.dumps(self.spec, indent=2))
            return result
        with self.assertRaisesRegex(ValueError, 'identity drifted during'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_symlink_output_parent_is_rejected(self):
        def command(argv, **kwargs):
            external = self.root / 'other-output'
            external.mkdir()
            Path(argv[2]).parent.rmdir()
            Path(argv[2]).parent.symlink_to(external, target_is_directory=True)
            return self.fake_command(argv, **kwargs)
        with self.assertRaisesRegex(ValueError, 'regular output/result'):
            self.invoke(command)
        self.assertFalse((self.output / 'afe-evidence.jsonl').exists())

    def test_unsafe_and_duplicate_ids_fail_before_any_command(self):
        for name in ('../escape', '/absolute', 'a/b', 'a\\b', '..', ' padded'):
            self.write_manifest([{**self.row, 'recording': name}])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'safe filenames'):
                self.invoke()
            self.last_command.assert_not_called()
        self.write_manifest([self.row, copy.deepcopy(self.row)])
        with self.assertRaisesRegex(ValueError, 'safe filenames'):
            self.invoke()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
