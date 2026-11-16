"""Source-only tests with invented tensors, bytes and process callbacks.

No real model, ONNX session, model compiler, or production Actions environment
is used. The full-head verifier is neither imported nor modified.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('backbone_source_fixture', HERE / 'backbone_parity_once.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fake_pins():
    pins = worker.pins_template()
    pins.update(enabled=True, worker_code_sha256='a' * 64,
        frontend_sources_sha256={name: 'b' * 64 for name in worker.FRONTEND_FILES},
        reference_sources_sha256={name: 'c' * 64 for name in worker.REFERENCE_FILES})
    return pins


def fake_context():
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': worker.REPOSITORY,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
        'GITHUB_RUN_ID': '987654', 'GITHUB_SHA': 'd' * 40}
    event = {'before': worker.STATIC_SOURCE_HEAD, 'after': 'd' * 40, 'ref': 'refs/heads/main',
             'repository': {'full_name': worker.REPOSITORY, 'private': False}}
    return env, event


def fake_row(sample=16000):
    return {'sample': sample, **{stage: [0.0] * count for stage, count in worker.STAGE_COUNTS.items()}}


def put_zip(path, files, pin):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    raw = path.read_bytes()
    pin.update(zip_bytes=len(raw), zip_sha256=worker.sha(raw))


def artificial_export(root):
    """Disabled readback fixture; invented C bytes are never compiled."""
    pins = worker.pins_template()
    pins['static_export']['run_id'] = '1001'
    pins['static_export']['artifact_id'] = 1002
    pipeline = root / 'pipeline'
    frontend = pipeline / 'research/frontend'
    frontend.mkdir(parents=True)
    code, hashes = {}, {}
    for name in worker.FRONTEND_FILES:
        raw = ('ARTIFICIAL SOURCE FIXTURE: ' + name).encode()
        (frontend / name).write_bytes(raw)
        hashes[name] = worker.sha(raw)
        code['research/frontend/' + name] = {'bytes': len(raw), 'sha256': hashes[name]}
    graphs = {name: {'asset_bytes': size, 'asset_sha256': digest, 'constant_bytes': 4,
        'workspace_bytes': 4, 'input': {'name': 'ARTIFICIAL_INPUT'}, 'output': {'name': 'ARTIFICIAL_OUTPUT'}}
        for name, (size, digest) in worker.ASSETS.items()}
    generated = {'fixed_models.c': b'/* ARTIFICIAL C, NEVER COMPILED */\n',
                 'fixed_models.h': b'/* ARTIFICIAL HEADER */\n'}
    manifest = {'schema': 'fixed-feature-pure-export-v1', 'real_inference_performed': False,
        'numerical_parity_established': False, 'ssc305_qualification': False, 'graphs': graphs,
        'generated_c_sha256': worker.sha(generated['fixed_models.c'])}
    generated['export_manifest.json'] = worker.canonical(manifest) + b'\n'
    entries = {name: {'member': 'generated/' + name, 'bytes': len(raw), 'sha256': worker.sha(raw)}
               for name, raw in generated.items()}
    pins['static_export']['generated'] = entries
    assets = {name: {'bytes': size, 'sha256': digest} for name, (size, digest) in worker.ASSETS.items()}
    report = {'schema': worker.STATIC_SCHEMA, 'status': 'EXPORTED_STATIC_GRAPHS_ONLY',
        'repository': worker.REPOSITORY, 'run_id': '1001', 'run_attempt': 1,
        'source_head': worker.STATIC_SOURCE_HEAD, 'export_attempted': True,
        'parser_versions': {'numpy': '1.26.4', 'onnx': '1.17.0'},
        'model_calls': 0, 'onnx_inference_calls': 0, 'c_inference_calls': 0,
        'optimizer_updates': 0, 'tts_asr_phone_calls': 0, 'numerical_parity_established': False,
        'ssc305_qualification': False, 'assets': assets, 'expected_assets': assets,
        'code_files': code, 'graphs': graphs,
        'generated': {'generated/' + name: {k: entry[k] for k in ('bytes', 'sha256')}
                      for name, entry in entries.items()}}
    files = {'RESULT.json': worker.canonical(report) + b'\n',
             **{'generated/' + name: raw for name, raw in generated.items()}}
    pins['static_export']['result_sha256'] = worker.sha(files['RESULT.json'])
    archive = root / 'static.zip'
    put_zip(archive, files, pins['static_export'])
    args = argparse.Namespace(pipeline_source=pipeline, frontend_source=frontend, static_export_zip=archive)
    return args, pins, report, files, hashes, generated


class BackboneSourceTests(unittest.TestCase):
    def test_default_and_template_are_zero_ml_and_zero_write(self):
        for args in ([], ['--template']):
            code = ('import importlib.util,sys;'
                    f's=importlib.util.spec_from_file_location("fixture",{str(HERE / "backbone_parity_once.py")!r});'
                    f'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main({args!r});'
                    'sys.exit(any(k in sys.modules for k in ("numpy","onnx","onnxruntime","torch")))')
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], cwd=tmp,
                                         capture_output=True, text=True, check=True)
                row = json.loads(result.stdout)
                self.assertFalse(row.get('enabled', row.get('execution_enabled')))
                self.assertFalse(row['head_parity_established'])
                self.assertFalse(row['kws_model_delivered'])
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_explicit_execute_cannot_bypass_disabled_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pins.json').write_bytes(worker.canonical(worker.pins_template()))
            cli = ['--execute', '--pins', str(root / 'pins.json')]
            for name in ('pipeline-source', 'frontend-source', 'assets', 'static-export-zip', 'output'):
                cli += ['--' + name, str(root / name)]
            with self.assertRaisesRegex(ValueError, 'DISABLED'):
                worker.main(cli)
            self.assertFalse((root / 'output').exists())

    def test_first_public_pipeline_successor_guard_has_no_retry_or_dispatch(self):
        pins, (env, event) = fake_pins(), fake_context()
        worker.execution_guard(pins, env, event, 'a' * 64)
        for key, value in (('GITHUB_ACTIONS', 'false'), ('GITHUB_REPOSITORY', 'jiying2007/kws-data'),
                           ('GITHUB_EVENT_NAME', 'workflow_dispatch'), ('GITHUB_REF', 'refs/heads/topic'),
                           ('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_ID', worker.STATIC_EXPORT['run_id'])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FIRST_NEW_PUBLIC_PIPELINE'):
                worker.execution_guard(pins, {**env, key: value}, event, 'a' * 64)
        for changed in ({**event, 'before': 'e' * 40}, {**event, 'after': 'e' * 40},
                        {**event, 'ref': 'refs/heads/topic'},
                        {**event, 'repository': {**event['repository'], 'private': True}}):
            with self.assertRaises(ValueError):
                worker.execution_guard(pins, env, changed, 'a' * 64)
        with self.assertRaises(ValueError):
            worker.execution_guard(pins, env, event, 'f' * 64)

    def test_scope_plan_source_and_static_artifact_pins_cannot_be_changed(self):
        for change in ('plan', 'head', 'training', 'retry', 'static', 'source', 'reference', 'admission'):
            pins = fake_pins()
            if change == 'plan': pins['plan_sha256'] = '0' * 64
            elif change == 'head': pins['head_evaluations'] = 1
            elif change == 'training': pins['head_training'] = {'pretend': True}
            elif change == 'retry': pins['no_retry'] = False
            elif change == 'static': pins['static_export']['artifact_id'] += 1
            elif change == 'source': del pins['frontend_sources_sha256']['fixed_frontend.c']
            elif change == 'reference': del pins['reference_sources_sha256']['causal_embedding.py']
            else: pins['qualification_allowed'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_pins(pins)

    def test_new_protocol_and_all_frontend_limits_are_frozen(self):
        self.assertEqual(worker.sha(worker.canonical(worker.PLAN)),
                         '5657e4387909bf7b47ba4ed7adec723005a8428357bd75d3515c3ddfe26e550b')
        self.assertEqual(worker.PLAN['full_head_plan_unchanged_sha256'],
                         'cdb8cce44b3ba88c92afeaf2b1d64124190c46198091f375d036a099964a16fb')
        self.assertEqual(worker.LIMITS, {'raw_mel': (0.01, 2e-5),
            'normalized_mel': (0.001, 2e-5), 'embedding': (0.001, 2e-4)})
        self.assertEqual(worker.PLAN['maximum_reference_onnx_session_runs'], 156)
        self.assertEqual(worker.PLAN['embedding_windows_per_backend'], 78)
        self.assertEqual(worker.PLAN['head_evaluations'], 0)
        self.assertEqual(worker.PLAN['optimizer_updates'], 0)
        self.assertEqual(worker.PLAN['total_timeout_seconds'], 1800)

    def test_three_original_pcm_hashes_and_chunks_consume_exactly_48000(self):
        signals = worker.synthetic_pcm()
        self.assertEqual(list(signals), ['zero', 'sine400', 'lcg14'])
        for name, raw in signals.items():
            self.assertEqual(worker.sha(raw), worker.PCM_SHA[name])
            self.assertEqual(len(raw), 96000)
            parts = list(worker.chunks(raw))
            self.assertEqual([len(p) for p in parts[:4]], [1, 1279, 257, 4096])
            self.assertEqual(sum(map(len, parts)), 48000)
            self.assertEqual(b''.join(struct.pack('<%dh' % len(p), *p) for p in parts), raw)
        self.assertEqual(worker.SAMPLES, tuple(16000 + i * 1280 for i in range(26)))
        self.assertEqual(signals['sine400'][:80], signals['sine400'][80:160])

    def test_three_stage_trace_rejects_head_stage_extra_eof_and_nonfinite(self):
        row = fake_row()
        raw = worker.trace_bytes([row])
        parsed = worker.parse_trace(raw, (16000,))
        self.assertEqual(set(parsed[0]), {'sample', 1, 2, 3})
        self.assertEqual(len(raw), 8 + 22576)
        head_stage = struct.pack('<IIQ8f', 4, 8, 16000, *([0.0] * 8))
        for changed in (raw[:-1], raw + b'\0', raw + head_stage):
            with self.assertRaisesRegex(ValueError, 'TRACE_LENGTH_OR_MAGIC'):
                worker.parse_trace(changed, (16000,))
        changed = raw[:8] + struct.pack('<IIQ', 1, 3104, 16001) + raw[24:]
        with self.assertRaisesRegex(ValueError, 'TRACE_STAGE_COUNT_CLOCK'):
            worker.parse_trace(changed, (16000,))
        with self.assertRaisesRegex(ValueError, 'NONFINITE_FEATURE_TRACE'):
            worker.parse_trace(raw[:24] + struct.pack('<f', float('nan')) + raw[28:], (16000,))

    def test_frontend_tolerances_and_exact_clocks_do_not_validate_a_head(self):
        reference = [fake_row()]
        passed = worker.comparisons(reference, copy.deepcopy(reference))
        self.assertTrue(passed['passed'])
        self.assertFalse(passed['head_checked'])
        self.assertEqual(set(passed['metrics']), {'raw_mel', 'normalized_mel', 'embedding'})
        for stage, delta in ((1, 0.010001), (2, 0.001001), (3, 0.001001)):
            actual = copy.deepcopy(reference)
            actual[0][stage][0] = delta
            self.assertFalse(worker.comparisons(reference, actual)['passed'])
        changed = copy.deepcopy(reference)
        changed[0]['sample'] += 1
        with self.assertRaisesRegex(ValueError, 'EXACT_EMBEDDING_CLOCKS'):
            worker.comparisons(reference, changed)

    def test_c_summary_requires_head_disabled_zero_ready_and_zero_events(self):
        rows = [fake_row(sample) for sample in worker.SAMPLES]
        summary = {'samples': 48000, 'embeddings': 26, 'head_enabled': False, 'head_ready': 0,
                   'events': 0, 'workspace_bytes': 454656, 'frontend_state_bytes': 32128,
                   'ssc305_qualification': False}
        self.assertEqual(worker.replay_summary(json.dumps(summary), rows), summary)
        for key, value in (('head_enabled', True), ('head_ready', 1), ('events', 1),
                           ('samples', 49280), ('embeddings', 27), ('ssc305_qualification', True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'BACKBONE_ONLY_C_PROCESS'):
                worker.replay_summary(json.dumps({**summary, key: value}), rows)

    def test_reference_budget_is_checked_before_forward_and_failure_is_not_retried(self):
        class InventedSession:
            calls = 0
            def run(self, output, inputs):
                self.calls += 1
                return ['ARTIFICIAL RESULT']
        session = InventedSession()
        accounting = {'onnx_session_attempts': 0, 'onnx_session_completions': 0}
        for _ in range(156):
            worker.bounded_ort_call(session, {}, accounting)
        with self.assertRaisesRegex(ValueError, 'REFERENCE_SESSION_BUDGET_EXHAUSTED'):
            worker.bounded_ort_call(session, {}, accounting)
        self.assertEqual(session.calls, 156)
        self.assertEqual(accounting, {'onnx_session_attempts': 156, 'onnx_session_completions': 156})
        with patch.object(session, 'run', side_effect=RuntimeError('ARTIFICIAL_FAILURE')) as call:
            failed = {'onnx_session_attempts': 0, 'onnx_session_completions': 0}
            with self.assertRaisesRegex(RuntimeError, 'ARTIFICIAL_FAILURE'):
                worker.bounded_ort_call(session, {}, failed)
            self.assertEqual(failed, {'onnx_session_attempts': 1, 'onnx_session_completions': 0})
            call.assert_called_once()

    def test_compile_commands_have_no_head_flag_library_or_weight(self):
        commands = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def invented_compile(command, timeout, log):
                commands.append(command)
                elf = bytearray(64)
                elf[:4], elf[5] = b'\x7fELF', 1
                machine = 40 if 'arm-linux-gnueabihf-gcc' in str(command[0]) else 62
                elf[18:20] = struct.pack('<H', machine)
                Path(command[-1]).write_bytes(elf)
                self.assertEqual(timeout, 180)
                return {'ARTIFICIAL_COMPILER_FIXTURE': True}
            with patch.object(worker.shutil, 'which', side_effect=lambda name: '/invented/' + name), \
                 patch.object(worker, 'timed_process', side_effect=invented_compile), \
                 patch.object(worker.subprocess, 'check_output', return_value='ARTIFICIAL TOOL VERSION\n'):
                programs = worker.build_replays(root / 'frontend', root / 'generated', root)
            self.assertEqual(set(programs), {'native', 'arm'})
            self.assertEqual(len(commands), 2)
            for command in commands:
                joined = ' '.join(map(str, command))
                self.assertNotIn('EF_WITH_HEAD', joined)
                self.assertNotIn('english_causal_head', joined)
                self.assertNotIn('causal-head', joined)
                self.assertIn('-fno-fast-math', command)
                self.assertIn('-ffp-contract=off', command)

    def test_static_readback_returns_original_three_bytes_without_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, pins, report, _, hashes, generated = artificial_export(Path(tmp))
            actual, payload = worker.static_export_archive(args, pins, hashes)
            self.assertEqual(actual, report)
            self.assertEqual(payload, generated)
            (args.frontend_source / 'fixed_frontend.c').write_bytes(b'CHANGED SOURCE')
            with self.assertRaisesRegex(ValueError, 'STATIC_EXPORT_SOURCE_BYTES'):
                worker.static_export_archive(args, pins, hashes)

    def test_static_failure_inference_claim_zip_or_c_tampering_is_rejected(self):
        for mode in ('failed', 'inference', 'zip', 'c', 'traversal'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args, pins, report, files, hashes, _ = artificial_export(Path(tmp))
                if mode == 'zip':
                    args.static_export_zip.write_bytes(args.static_export_zip.read_bytes() + b'CHANGED')
                else:
                    if mode == 'failed': report['status'] = 'FAILED_NO_RETRY'
                    elif mode == 'inference': report['onnx_inference_calls'] = 1
                    elif mode == 'c': files['generated/fixed_models.c'] = b'CHANGED C'
                    else: files['../outside'] = b'INVALID'
                    files['RESULT.json'] = worker.canonical(report) + b'\n'
                    pins['static_export']['result_sha256'] = worker.sha(files['RESULT.json'])
                    put_zip(args.static_export_zip, files, pins['static_export'])
                with self.assertRaises(ValueError):
                    worker.static_export_archive(args, pins, hashes)

    def test_child_timeout_receipt_preserves_partial_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'timeout.json'
            with patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fixture'], 300)) as run:
                with self.assertRaisesRegex(TimeoutError, 'BOUNDED_CHILD_PROCESS_TIMEOUT'):
                    worker.timed_process(['ARTIFICIAL_PROCESS'], 300, log)
                run.assert_called_once()
            receipt = worker.read_json(log)
            self.assertEqual(receipt['status'], 'TIMEOUT')
            self.assertEqual(receipt['timeout_seconds'], 300)

    def test_create_only_receipts_and_manifest_keep_nonqualification_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker.once(root / 'fixture.json', {'ARTIFICIAL_FIXTURE': True})
            with self.assertRaises(FileExistsError):
                worker.once(root / 'fixture.json', {})
            worker.manifest(root)
            record = worker.read_json(root / 'MANIFEST.json')
            self.assertEqual(record['profile'], 'FIXED_SHARED_BACKBONE_ONLY')
            self.assertEqual(record['plan_sha256'], worker.FIXED_PLAN_SHA)
            for flag in (*worker.FALSE_FLAGS, 'head_parity_established', 'kws_model_delivered', 'ssc305_qualification'):
                self.assertIs(record[flag], False)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    unittest.main()
