"""Synthetic stdlib receipt fault tests. No models, setup, or external writes."""
import copy
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import blind_pcm as pcm
import public_evidence as pe
import worker
from execution_common import canonical, sha


def put(path, value, blind=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((pcm.canonical if blind else canonical)(value))
    return sha(path.read_bytes())


def make_tts(root, plan, engine, count=None):
    """Handmade synthetic evidence only, never an actual generation receipt."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    claim = worker.stage_claim(engine, plan)
    put(root / 'stage-claim.json', claim)
    started = put(root / 'model-load-started.json', dict(claim, load_seconds=1200))
    receipt = dict(schema='kw1-public-tts-load-identity-v1', engine=engine,
                   scientific_load_sha256='a' * 64, runtime_identity_sha256='b' * 64, source_pins_sha256='c' * 64)
    loaded = put(root / 'model-load-completed.json', dict(claim, started_sha256=started, receipt=receipt))
    cells = [c for c in plan['slots'] if c['engine'] == engine]
    outcomes = []
    for cell in cells[:len(cells) if count is None else count]:
        sid = cell['slot_id']; index = int(sid[-2:]) + 1
        values = struct.pack('<24f', *[((index + 1) * (j + 1)) / 10000 for j in range(24)])
        raw = worker.float_wav(values)
        filename = sid + '.raw-float.wav'; (root / filename).write_bytes(raw)
        termination = dict(iterations=3, last_token_ids=[2150], eos=2150, ended_with_eos=True, hit_token_cap_without_eos=False)
        _, samples = worker.native_samples(raw)
        native = dict(sample_rate_hz=24000, channels=1, frames=24, subtype='FLOAT', source_values_sha256=sha(values))
        outcome = dict(cell, status='completed', attempts=1, error_type=None, file=filename,
                       raw_wav_sha256=sha(raw), native=native, metrics=worker.signal_metrics(samples, termination), termination=termination)
        call_sha = put(root / 'cells' / sid / 'started.json', worker.started_record(cell, plan, loaded))
        put(root / 'cells' / sid / 'completed.json', dict(schema='kw1-tts-cell-completed-v1', started_sha256=call_sha, outcome=outcome))
        outcomes.append(outcome)
    return outcomes


def make_inputs(root, count=16, frames=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(count):
        size = 16 if frames is None else frames[i]
        raw = struct.pack('<' + str(size) + 'h', *[(i + 1) * (j + 1) for j in range(size)])
        value_sha = sha(raw); (root / (value_sha + '.pcm')).write_bytes(raw)
        clips.append(dict(opaque_id=value_sha, file=value_sha + '.pcm', sha256=value_sha,
                          bytes=len(raw), frames=size, sample_rate_hz=16000, channels=1, sample_format='s16le'))
    put(root / 'job.json', dict(schema=pcm.JOB_SCHEMA, clips=sorted(clips, key=lambda r: r['opaque_id'])), True)
    return clips


def make_asr(inputs, root, recognizer='qwen_asr', count=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    job, raw, data = pcm.bind_input(inputs); mapping = pcm.identity_map(job, raw, data)
    claim = dict(schema='kw1-blind-asr-stage-claim-v1', recognizer=recognizer,
                 job_sha256=sha(raw), identity_map_sha256=sha(pcm.canonical(mapping)), attempt=1)
    put(root / 'stage-claim.json', claim, True); put(root / 'identity-map.json', mapping, True)
    put(root / 'model-load-started.json', dict(claim, load_seconds=1200), True)
    receipt = dict(schema='kw1-public-model-load-identity-v1', recognizer=recognizer,
                   scientific_load_sha256='a' * 64, runtime_identity_sha256='b' * 64, source_pins_sha256='c' * 64)
    loaded = put(root / 'model-load-completed.json', dict(claim, receipt=receipt), True)
    rows = mapping['clips'][:len(mapping['clips']) if count is None else count]
    for identity in rows:
        oid = identity['opaque_id']; cell = root / 'cells' / oid
        started = dict(schema='kw1-blind-asr-cell-started-v1', recognizer=recognizer,
                       opaque_id=oid, pcm_sha256=identity['pcm_sha256'], wav_sha256=identity['wav_sha256'],
                       job_sha256=sha(raw), identity_map_sha256=sha(pcm.canonical(mapping)), load_completed_sha256=loaded, attempt=1)
        call_sha = put(cell / 'started.json', started, True)
        result = dict(raw_text='你好小窝', completeness='complete', quality_flags=[])
        evidence = dict(schema='kw1-public-decoder-evidence-v1', recognizer=recognizer, scientific_result_sha256='d' * 64)
        evidence_sha = put(cell / 'decoder-evidence.json', dict(result=result, evidence=evidence), True)
        outcome = dict(opaque_id=oid, recognizer=recognizer, pcm_sha256=identity['pcm_sha256'],
                       wav_sha256=identity['wav_sha256'], status='SUCCESS', attempts=1, error_type=None, **result)
        put(cell / 'completed.json', dict(schema='kw1-blind-asr-cell-completed-v1', started_sha256=call_sha,
                                         decoder_evidence_sha256=evidence_sha, outcome=outcome), True)
    return mapping


def make_work(root, all_complete=True):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); plan = worker.frozen_plan()
    admission = dict(schema='kw1-admission-receipt-v1', admitted=True, run_id=999999999,
                     source_sha='a' * 40, workflow_id=999999999, config_sha256='b' * 64,
                     source_files={'research/kw1-preset-supplement-v1/plan.json': sha(Path(worker.__file__).with_name('plan.json').read_bytes())},
                     max_tts_calls=16, max_asr_calls=48, retry=False)
    put(root / 'campaign.claim.json', admission)
    if not all_complete:
        return plan, admission
    native = []
    from test_worker import make_stage
    for engine in pe.ENGINES:
        output = root / 'tts' / engine / 'output'
        make_stage(output, engine, plan)
        native.extend(pe.validate_tts_archive(output, engine, plan)['outcomes'])
    clips = make_inputs(root / 'blind-input', frames=[(2 * r['native']['frames'] + 2)//3 for r in native])
    mapping = []
    for source, clip in zip(native, clips):
        mapping.append(dict(slot_id=source['slot_id'], native_sha256=source['raw_wav_sha256'],
                            native_values_sha256=source['native']['source_values_sha256'], derived=clip,
                            derivation=dict(method='scipy.signal.resample_poly', up=2, down=3, window=['kaiser', 5.0],
                            padtype='constant', quantization='floor_float64_times_32768_to_int16', numpy_version='2.3.2',
                            scipy_version='1.17.1', gain_normalization=False, trimming=False)))
    put(root / 'identity-map.json', mapping)
    import blind_asr_worker as asr
    from test_blind_asr import FakeModel
    for name in pe.RECOGNIZERS:
        output = root / 'asr' / name / 'output'; output.mkdir(parents=True)
        scratch = root / ('scratch-' + name); scratch.mkdir()
        FakeModel.calls = 0
        FakeModel.result = dict(raw_text='你好小窝', completeness='complete', quality_flags=[])
        with ExitStack() as stack:
            for fn, value in [('verify_sources', {}), ('verify_scope', {'schema': 'fake-scope'}),
                              ('prepare_environment', None), ('verify_runtime', {'fake': True})]:
                stack.enter_context(patch.object(asr, fn, return_value=value))
            stack.enter_context(patch.object(asr, 'SCRATCH', scratch))
            stack.enter_context(patch.object(asr, 'RetainedBackend', FakeModel))
            asr.run(name, root / 'blind-input', output, root / 'not-loaded')
        if FakeModel.calls != 16:
            raise AssertionError('synthetic ASR fixed sixteen-call count')
    from docker_backend import DockerBackend, STAGES
    backend = DockerBackend.__new__(DockerBackend)
    backend._failed = False; backend._next = len(STAGES)
    backend._summaries = [dict(stage=name, status='complete', attempt=1, wall_seconds=1.5,
                               container_configuration_verified=name != 'image', exit_code=0, oom_killed=False,
                               host=dict(available_memory_bytes=16 * 1024**3, available_disk_bytes=30 * 1024**3)) for name in STAGES]
    backend._runtime_cleanups = [dict(runtime=name, status='removed_after_container_cleanup', logical_regular_bytes=123,
        scope='only this campaign owned runtime; claims/results/source code retained')
        for name in ('runtime-qwen06', 'runtime-qwen17', 'runtime-primary', 'runtime-whisper')]
    put(root / 'backend/summary.json', backend.public_summary())
    return plan, admission


class ArchiveFaultTests(unittest.TestCase):
    def assert_unknown(self, value):
        self.assertIs(value['verified'], False); self.assertEqual(value['status'], 'UNVERIFIED')
        self.assertIsNone(value['attempts']); self.assertEqual(value['outcomes'], []); self.assertEqual(value['files'], {})
        self.assertNotIn(b'CANARY_PRIVATE', canonical(value))

    def test_complete_synthetic_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, _ = make_work(Path(tmp) / 'work')
            host, stages, resources, report, ledger = pe.collect(plan, Path(tmp) / 'work')
            self.assertTrue(host['verified']); self.assertEqual(report['consumed_calls'], {'tts': 16, 'asr': 48})
            self.assertEqual(report['target_qualified'], 16); self.assertEqual(len(ledger['slots']), 64)
            self.assertTrue(resources['verified'])

    def test_tts_required_parent_fault_matrix(self):
        variants = ('missing', 'empty', 'truncated', 'invalid', 'schema', 'boolean')
        for name in ('stage-claim.json', 'model-load-started.json', 'model-load-completed.json',
                     'cells/kw1s-00/started.json', 'cells/kw1s-00/completed.json'):
            for kind in variants:
                with self.subTest(file=name, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); plan = worker.frozen_plan(); make_tts(root, plan, 'qwen06')
                    path = root / name; raw = path.read_bytes()
                    if kind == 'missing': path.unlink()
                    elif kind == 'empty': path.write_bytes(b'')
                    elif kind == 'truncated': path.write_bytes(raw[:len(raw)//2])
                    elif kind == 'invalid': path.write_bytes(b'CANARY_PRIVATE')
                    elif kind == 'schema': path.write_bytes(b'{"extra":"CANARY_PRIVATE"}')
                    else:
                        value = pe.json_bytes(raw)
                        if 'attempt' in value: value['attempt'] = True
                        elif 'outcome' in value: value['outcome']['attempts'] = True
                        else: value['load_seconds'] = True
                        path.write_bytes(canonical(value))
                    self.assert_unknown(pe.assess_tts_archive(root, 'qwen06', plan))

    def test_asr_required_parent_fault_matrix(self):
        for basename in ('stage-claim.json', 'identity-map.json', 'model-load-started.json', 'model-load-completed.json',
                         'started.json', 'decoder-evidence.json', 'completed.json'):
            for kind in ('missing', 'empty', 'truncated', 'invalid', 'schema'):
                with self.subTest(file=basename, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); inp = root / 'input'; out = root / 'output'; make_inputs(inp)
                    mapping = make_asr(inp, out); oid = mapping['clips'][0]['opaque_id']
                    path = out / (f'cells/{oid}/{basename}' if basename in pe.CELL_FILES else basename)
                    raw = path.read_bytes()
                    if kind == 'missing': path.unlink()
                    elif kind == 'empty': path.write_bytes(b'')
                    elif kind == 'truncated': path.write_bytes(raw[:len(raw)//2])
                    elif kind == 'invalid': path.write_bytes(b'CANARY_PRIVATE')
                    else: path.write_bytes(b'{"extra":"CANARY_PRIVATE"}')
                    self.assert_unknown(pe.assess_asr_archive(inp, out, 'qwen_asr'))

    def test_asr_prefix_only_without_later_receipts(self):
        for length in (0, 1, 100):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as tmp:
                inp = Path(tmp) / 'input'; out = Path(tmp) / 'output'; make_inputs(inp)
                mapping = make_asr(inp, out, count=1); cell = out / 'cells' / mapping['clips'][0]['opaque_id']
                (cell / 'completed.json').unlink(); (cell / 'decoder-evidence.json').unlink()
                raw = (cell / 'started.json').read_bytes(); (cell / 'started.json').write_bytes(raw[:length])
                value = pe.assess_asr_archive(inp, out, 'qwen_asr')
                self.assertTrue(value['verified']); self.assertEqual(value['attempts'], 1)
                self.assertEqual(value['outcomes'][0]['status'], 'FAILED')
                (cell / 'started.json').write_bytes(b'garbage-not-prefix')
                self.assert_unknown(pe.assess_asr_archive(inp, out, 'qwen_asr'))

    def test_asr_boolean_parent_never_old_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / 'input'; out = Path(tmp) / 'output'; make_inputs(inp)
            mapping = make_asr(inp, out); path = out / 'stage-claim.json'
            value = pe.json_bytes(path.read_bytes()); value['attempt'] = True; put(path, value, True)
            cell = out / 'cells' / mapping['clips'][0]['opaque_id']; (cell / 'started.json').write_bytes(b'garbage-not-prefix')
            (cell / 'completed.json').write_bytes(b'{"CANARY_PRIVATE":true}')
            self.assertTrue(pe.active(out)); self.assert_unknown(pe.assess_asr_archive(inp, out, 'qwen_asr'))

    def test_missing_and_symlink_active_empty_not_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); empty = root / 'empty'; empty.mkdir(); link = root / 'link'; link.symlink_to(empty)
            self.assertFalse(pe.active(empty)); self.assertTrue(pe.active(root / 'missing')); self.assertTrue(pe.active(link))
            self.assert_unknown(pe.assess_tts_archive(root / 'missing', 'qwen06', worker.frozen_plan()))

    def test_extra_private_bodies_are_not_projected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = worker.frozen_plan(); make_tts(root, plan, 'qwen06')
            (root / 'private.json').write_text('{"secret":"CANARY_PRIVATE"}')
            result = pe.assess_tts_archive(root, 'qwen06', plan)
            self.assertTrue(result['verified']); self.assertNotIn('private.json', result['files'])

    def test_orphan_native_and_unknown_cell_files_are_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = worker.frozen_plan(); make_tts(root, plan, 'qwen06', count=1)
            shutil.rmtree(root / 'cells/kw1s-00')
            self.assert_unknown(pe.assess_tts_archive(root, 'qwen06', plan))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = worker.frozen_plan(); make_tts(root, plan, 'qwen06', count=1)
            (root / 'cells/kw1s-00/unknown.json').write_text('CANARY_PRIVATE')
            self.assert_unknown(pe.assess_tts_archive(root, 'qwen06', plan))

    def test_duplicate_native_and_numeric_identity_drift_are_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = worker.frozen_plan(); make_tts(root, plan, 'qwen06')
            first = pe.json_bytes((root / 'cells/kw1s-00/completed.json').read_bytes())['outcome']
            second_path = root / 'cells/kw1s-01/completed.json'; second = pe.json_bytes(second_path.read_bytes())
            for key in ('raw_wav_sha256', 'native', 'metrics', 'termination'):
                second['outcome'][key] = first[key]
            (root / 'kw1s-01.raw-float.wav').write_bytes((root / 'kw1s-00.raw-float.wav').read_bytes())
            put(second_path, second); self.assert_unknown(pe.assess_tts_archive(root, 'qwen06', plan))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'work'; plan, _ = make_work(root)
            path = root / 'identity-map.json'; mapping = pe.json_bytes(path.read_bytes())
            mapping[0]['derivation']['numpy_version'] = '1.26.4'; put(path, mapping)
            self.assertFalse(pe.assess_host_inputs(root, plan)['verified'])

    def test_cross_stage_later_calls_are_unverified(self):
        for stop_stage, later_stage in (('tts/qwen06', 'tts/qwen17'), ('asr/qwen_asr', 'asr/sensevoice'),
                                       ('asr/sensevoice', 'asr/whisper_small')):
            with self.subTest(stage=stop_stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'work'; plan, _ = make_work(root)
                output = root / stop_stage / 'output'
                (output / 'model-load-completed.json').write_bytes(b'CANARY_PRIVATE')
                _, stages, _, report, attempts = pe.collect(plan, root)
                self.assert_unknown(stages[later_stage]); self.assertEqual(len(attempts['slots']), 64)
                self.assertEqual(report['status'], 'stopped_no_retry')

    def test_resource_unknown_body_and_typed_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'backend').mkdir(); path = root / 'backend/summary.json'
            path.write_text('{"secret":"CANARY_PRIVATE"}')
            result = pe.assess_resources(root); self.assertFalse(result['verified']); self.assertIsNone(result['body'])
            self.assertNotIn(b'CANARY_PRIVATE', canonical(result))
            value = dict(schema='kw1-container-summary-v1', status='started', stages=[], runtime_cleanups=[],
                         max_wall_seconds=18000, max_attempts_per_stage=1, raw_setup_exported=False, api_tokens_passed=False)
            put(path, value); self.assertTrue(pe.assess_resources(root)['verified'])
            value['max_attempts_per_stage'] = True; put(path, value); self.assertFalse(pe.assess_resources(root)['verified'])


if __name__ == '__main__':
    unittest.main()
